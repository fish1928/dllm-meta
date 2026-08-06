"""
Cross-dataset generalization experiments for router_llada_v2.

Examples
--------
Run the complete experiment suite:

    python ablation_test_stage_cross_dataset.py \
        --dataset gsm8k,128 \
        --dataset ifeval,256

By default, a dataset named ``gsm8k`` is read from ``./stats_gsm8k``.
Use --data-root to change the parent folder:

    python ablation_test_stage_cross_dataset.py \
        --data-root /data/router_stats \
        --dataset gsm8k,128 \
        --dataset ifeval,256

An explicit folder can optionally be supplied as a third field:

    --dataset gsm8k,128,/data/gsm8k_stats

The default ``--experiments all`` trains three models for two datasets:

1. train on GSM8K; evaluate on GSM8K and IFEval;
2. train on IFEval; evaluate on IFEval and GSM8K;
3. train on the balanced mixture; evaluate on each dataset and their pooled
   evaluation blocks.

Different block lengths are supported because each block reference retains its
own dataset specification and calls build_geometry() with that dataset's
size_block.

The fixed default router configuration follows the previous winning setup:

    features:      attn_last + pos_delta + mask_density
    normalization: rank for attention; geometry remains raw
    router:        pointwise gated MLP, hidden=64, blocks=2
    loss:          Plackett-Luce
    horizon:       5

Results are saved under ``stage_cross_dataset`` in
``ablation_test_report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from attn_order_eval import ScoreOrderEval, summ
from router_llada import (
    Feature_attn_all,
    Feature_attn_last,
    Feature_conf,
    Feature_entropy,
    Feature_log_scaled,
    Feature_margin,
    Feature_mask_density,
    Feature_minmax_row,
    Feature_pos_delta,
    Feature_rank_normed,
    Feature_softmax_row,
    Feature_step_progress,
    Feature_x0_stability,
    Feature_znormed_row,
    FactoryLoss,
    FactoryRouter,
    build_geometry,
    load_stat,
)


# ---------------------------------------------------------------------------
# Dataset specifications and block references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str
    size_block: int


@dataclass(frozen=True)
class BlockRef:
    dataset: str
    id_sample: int
    pos_base: int


@dataclass
class DatasetSplit:
    spec: DatasetSpec
    ids_train: List[int]
    ids_eval: List[int]
    blocks_train: List[BlockRef]
    blocks_eval: List[BlockRef]


# ---------------------------------------------------------------------------
# JSON report helpers
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_report(report_path: str) -> Dict[str, Any]:
    if not os.path.exists(report_path):
        return {}
    with open(report_path, "r", encoding="utf-8") as file:
        return json.load(file)


def reset_stage(stage: str, report_path: str) -> None:
    report = load_report(report_path)
    report[stage] = []
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def save_result(
    *,
    stage: str,
    name: str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    error: Optional[str],
    report_path: str,
) -> None:
    report = load_report(report_path)
    records = report.setdefault(stage, [])

    record = {
        "name": name,
        "config": jsonable(config),
        "metrics": jsonable(metrics),
        "error": error,
    }

    for index, previous in enumerate(records):
        if previous.get("name") == name:
            records[index] = record
            break
    else:
        records.append(record)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


# ---------------------------------------------------------------------------
# Command-line parsing
# ---------------------------------------------------------------------------


def parse_dataset_argument(value: str, data_root: str) -> DatasetSpec:
    """
    Parse either:

        NAME,SIZE_BLOCK
        NAME,SIZE_BLOCK,FOLDER

    The two-field form resolves to DATA_ROOT/stats_NAME.
    """
    parts = [part.strip() for part in value.split(",")]

    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            "--dataset must be NAME,SIZE_BLOCK or NAME,SIZE_BLOCK,FOLDER"
        )

    name = parts[0]
    if not name:
        raise argparse.ArgumentTypeError("dataset name cannot be empty")

    try:
        size_block = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid block size in --dataset {value!r}"
        ) from exc

    if size_block <= 1:
        raise argparse.ArgumentTypeError("size_block must be greater than 1")

    if len(parts) == 3:
        folder = os.path.abspath(os.path.expanduser(parts[2]))
    else:
        folder = os.path.abspath(
            os.path.join(data_root, f"stats_{name}")
        )

    return DatasetSpec(
        name=name,
        folder=folder,
        size_block=size_block,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate one router across datasets with different "
            "block sizes."
        )
    )

    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME,SIZE_BLOCK[,FOLDER]",
        help=(
            "Repeat once per dataset, e.g. --dataset gsm8k,128 "
            "--dataset ifeval,256. The default folder is stats_NAME."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Parent folder used by the two-field --dataset form.",
    )
    parser.add_argument(
        "--experiments",
        choices=["all", "individual", "mixed"],
        default="all",
        help=(
            "individual trains one model per dataset; mixed trains one model "
            "on all datasets; all runs both."
        ),
    )
    parser.add_argument(
        "--mix-strategy",
        choices=["balanced", "proportional"],
        default="balanced",
        help=(
            "balanced gives each training dataset the same number of block "
            "updates per epoch; proportional uses every block once."
        ),
    )
    parser.add_argument(
        "--features",
        default="attn_last,pos_delta,mask_density,conf",
        help="Comma-separated router features.",
    )
    parser.add_argument(
        "--normalization",
        choices=[
            "raw",
            "rank",
            "znorm_row",
            "minmax_row",
            "log_znorm",
            "softmax_attn",
        ],
        default="rank",
    )
    parser.add_argument(
        "--loss",
        choices=[
            "uniform_within_h",
            "decay_within_h",
            "bce_within_h",
            "plackett_luce",
        ],
        default="plackett_luce",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument(
        "--filter-result",
        choices=["all", "pass", "fail"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--dim-hidden", type=int, default=64)
    parser.add_argument("--num-blocks-mlp", type=int, default=2)
    parser.add_argument(
        "--stage",
        default="stage_cross_dataset",
    )
    parser.add_argument(
        "--report-path",
        default="ablation_test_report.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing records for this stage instead of resetting it.",
    )

    return parser


# ---------------------------------------------------------------------------
# Feature construction following router_llada_v2's compositional pattern
# ---------------------------------------------------------------------------


def make_feature(
    name: str,
    folder_data: str,
    num_layers: int,
):
    if name == "attn_last":
        return Feature_attn_last(folder_data)
    if name == "attn_all":
        return Feature_attn_all(
            folder_data,
            num_layers=num_layers,
        )
    if name == "conf":
        return Feature_conf(folder_data)
    if name == "margin":
        return Feature_margin(folder_data)
    if name == "entropy":
        return Feature_entropy(folder_data)
    if name == "pos_delta":
        return Feature_pos_delta(folder_data)
    if name == "mask_density":
        return Feature_mask_density(folder_data)
    if name == "x0_stability":
        return Feature_x0_stability(folder_data)
    if name == "step_progress":
        return Feature_step_progress(folder_data)

    raise ValueError(f"unknown feature: {name}")


def normalize_feature(
    name: str,
    feature,
    normalization: str,
):
    """
    Preserve geometry/progress features in their bounded raw form, matching
    the previous ablation scripts. Apply the requested normalization to model-
    state and attention features.
    """
    keep_raw = {
        "pos_delta",
        "mask_density",
        "x0_stability",
        "step_progress",
    }

    if normalization == "raw" or name in keep_raw:
        return feature

    if normalization == "rank":
        return Feature_rank_normed(feature)
    if normalization == "znorm_row":
        return Feature_znormed_row(feature)
    if normalization == "minmax_row":
        return Feature_minmax_row(feature)
    if normalization == "log_znorm":
        if name in {"attn_last", "attn_all"}:
            return Feature_znormed_row(
                Feature_log_scaled(feature)
            )
        return Feature_znormed_row(feature)
    if normalization == "softmax_attn":
        if name in {"attn_last", "attn_all"}:
            return Feature_softmax_row(
                feature,
                temperature=1.0,
            )
        return Feature_znormed_row(feature)

    raise ValueError(f"unsupported normalization: {normalization}")


def build_feature_bundle(
    spec: DatasetSpec,
    feature_names: Sequence[str],
    normalization: str,
    num_layers: int,
):
    features = []

    for name in feature_names:
        feature = make_feature(
            name,
            spec.folder,
            num_layers,
        )
        feature = normalize_feature(
            name,
            feature,
            normalization,
        )
        features.append(feature)

    return features


# ---------------------------------------------------------------------------
# Split and block discovery
# ---------------------------------------------------------------------------


def read_result(spec: DatasetSpec, id_sample: int) -> str:
    path = os.path.join(
        spec.folder,
        str(id_sample),
        "generated.json",
    )

    if not os.path.exists(path):
        return "unknown"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file).get("result", "unknown")


def list_sample_ids(
    spec: DatasetSpec,
    filter_result: str,
) -> List[int]:
    if not os.path.isdir(spec.folder):
        raise FileNotFoundError(
            f"dataset folder does not exist: {spec.folder}"
        )

    ids = sorted(
        int(name)
        for name in os.listdir(spec.folder)
        if name.isdigit()
        and os.path.isdir(os.path.join(spec.folder, name))
    )

    if filter_result in {"pass", "fail"}:
        ids = [
            id_sample
            for id_sample in ids
            if read_result(spec, id_sample) == filter_result
        ]

    if len(ids) < 2:
        raise RuntimeError(
            f"dataset {spec.name!r} needs at least two samples after "
            f"filtering; found {len(ids)}"
        )

    return ids


def list_blocks(
    spec: DatasetSpec,
    ids_sample: Iterable[int],
) -> List[BlockRef]:
    blocks: List[BlockRef] = []

    for id_sample in ids_sample:
        folder_base = os.path.join(
            spec.folder,
            str(id_sample),
        )

        pos_root_path = os.path.join(
            folder_base,
            ".pos_root",
        )
        with open(pos_root_path, "r", encoding="utf-8") as file:
            pos_root = int(file.read())

        unmask_files = [
            name
            for name in os.listdir(folder_base)
            if name.startswith("unmask_")
            and name.endswith(".pt")
        ]

        for id_block in range(len(unmask_files)):
            blocks.append(
                BlockRef(
                    dataset=spec.name,
                    id_sample=id_sample,
                    pos_base=(
                        pos_root
                        + id_block * spec.size_block
                    ),
                )
            )

    return blocks


def build_dataset_split(
    spec: DatasetSpec,
    holdout: float,
    filter_result: str,
) -> DatasetSplit:
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout must be between 0 and 1")

    ids_all = list_sample_ids(
        spec,
        filter_result,
    )

    n_eval = max(1, int(len(ids_all) * holdout))
    n_eval = min(n_eval, len(ids_all) - 1)

    ids_train = ids_all[:-n_eval]
    ids_eval = ids_all[-n_eval:]

    blocks_train = list_blocks(
        spec,
        ids_train,
    )
    blocks_eval = list_blocks(
        spec,
        ids_eval,
    )

    if not blocks_train:
        raise RuntimeError(
            f"dataset {spec.name!r} has no training blocks"
        )
    if not blocks_eval:
        raise RuntimeError(
            f"dataset {spec.name!r} has no evaluation blocks"
        )

    return DatasetSplit(
        spec=spec,
        ids_train=ids_train,
        ids_eval=ids_eval,
        blocks_train=blocks_train,
        blocks_eval=blocks_eval,
    )


# ---------------------------------------------------------------------------
# Multi-dataset trainer
# ---------------------------------------------------------------------------


def evaluate_ndcg(
    evaluator: ScoreOrderEval,
    horizon: int,
):
    if hasattr(evaluator, "ndcg"):
        return evaluator.ndcg(horizon)
    if hasattr(evaluator, "ndgc"):
        return evaluator.ndgc(horizon)
    if hasattr(evaluator, "ndcg_at_h"):
        return evaluator.ndcg_at_h(horizon)

    raise AttributeError(
        "ScoreOrderEval must provide ndcg(h), ndgc(h), or ndcg_at_h(h)"
    )


class MultiDatasetRouterTrainer:
    """
    RouterTrainer-style class with dataset-specific block lengths and feature
    bundles.
    """

    def __init__(
        self,
        *,
        splits: Dict[str, DatasetSplit],
        feature_bundles: Dict[str, Sequence],
        h: int,
        device: str,
        lr: float,
        weight_decay: float,
        seed: int,
        mix_strategy: str,
    ):
        self.splits = splits
        self.feature_bundles = feature_bundles
        self.h = h
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.mix_strategy = mix_strategy

        self.router = None
        self.loss = None
        self.train_dataset_names: List[str] = []

        self._validate_feature_bundles()

    def _validate_feature_bundles(self) -> None:
        signatures = {}

        for name, features in self.feature_bundles.items():
            signatures[name] = [
                (feature.get_name(), feature.dim())
                for feature in features
            ]

        reference_name = next(iter(signatures))
        reference = signatures[reference_name]

        for name, signature in signatures.items():
            if signature != reference:
                raise ValueError(
                    "feature bundles must have identical names and dimensions; "
                    f"{reference_name}={reference}, {name}={signature}"
                )

    def register_router(self, router):
        self.router = router.to(self.device)
        return self

    def register_loss(self, loss):
        self.loss = loss() if isinstance(loss, type) else loss
        return self

    def _build_block_x(self, ref: BlockRef) -> torch.Tensor:
        split = self.splits[ref.dataset]
        features = self.feature_bundles[ref.dataset]

        x = torch.cat(
            [
                feature.load_block(
                    ref.id_sample,
                    ref.pos_base,
                    split.spec.size_block,
                )
                for feature in features
            ],
            dim=-1,
        )

        return x.to(self.device)

    def _load_order(self, ref: BlockRef) -> torch.Tensor:
        split = self.splits[ref.dataset]
        folder_base = os.path.join(
            split.spec.folder,
            str(ref.id_sample),
        )

        unmask = load_stat(
            folder_base,
            "unmask",
            ref.pos_base,
            split.spec.size_block,
        )

        order = (
            unmask.squeeze(-1).long()
            - ref.pos_base
        )

        return order.to(self.device)

    def _load_block(
        self,
        ref: BlockRef,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._build_block_x(ref), self._load_order(ref)

    @staticmethod
    def _repeat_to_length(
        refs: Sequence[BlockRef],
        length: int,
        rng: random.Random,
    ) -> List[BlockRef]:
        if not refs:
            raise ValueError("cannot sample from an empty block list")

        output: List[BlockRef] = []

        while len(output) < length:
            cycle = list(refs)
            rng.shuffle(cycle)
            output.extend(cycle)

        return output[:length]

    def _epoch_training_refs(
        self,
        train_dataset_names: Sequence[str],
        epoch: int,
    ) -> List[BlockRef]:
        rng = random.Random(
            self.seed + 1_000_003 * epoch
        )

        pools = {
            name: list(self.splits[name].blocks_train)
            for name in train_dataset_names
        }

        if len(train_dataset_names) == 1:
            refs = pools[train_dataset_names[0]]
            rng.shuffle(refs)
            return refs

        if self.mix_strategy == "proportional":
            refs = [
                ref
                for name in train_dataset_names
                for ref in pools[name]
            ]
            rng.shuffle(refs)
            return refs

        if self.mix_strategy != "balanced":
            raise ValueError(
                f"unknown mix strategy: {self.mix_strategy}"
            )

        # Equal number of optimizer updates per dataset. Smaller datasets are
        # repeated with a new shuffled cycle rather than discarded.
        target_length = max(
            len(pool)
            for pool in pools.values()
        )

        balanced_pools = {
            name: self._repeat_to_length(
                pools[name],
                target_length,
                rng,
            )
            for name in train_dataset_names
        }

        refs: List[BlockRef] = []

        for index in range(target_length):
            names_this_round = list(train_dataset_names)
            rng.shuffle(names_this_round)

            for name in names_this_round:
                refs.append(
                    balanced_pools[name][index]
                )

        return refs

    def train(
        self,
        *,
        train_dataset_names: Sequence[str],
        num_epochs: int,
        log_every: int = 1,
    ):
        if self.router is None or self.loss is None:
            raise RuntimeError(
                "register_router() and register_loss() before train()"
            )

        if not self.router.trainable():
            raise RuntimeError(
                "cross-dataset training requires a trainable router"
            )

        self.train_dataset_names = list(train_dataset_names)
        torch.manual_seed(self.seed)

        optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.router.train()

        for epoch in range(num_epochs):
            refs = self._epoch_training_refs(
                train_dataset_names,
                epoch,
            )

            losses: List[float] = []
            losses_by_dataset: Dict[str, List[float]] = {
                name: []
                for name in train_dataset_names
            }

            for ref in refs:
                x, order = self._load_block(ref)
                size_block = self.splits[
                    ref.dataset
                ].spec.size_block

                gap, candidate_mask = build_geometry(
                    order.cpu(),
                    size_block,
                )
                gap = gap.to(self.device)
                candidate_mask = candidate_mask.to(
                    self.device
                )

                optimizer.zero_grad(set_to_none=True)

                scores = self.router(x)
                loss = self.loss(
                    scores,
                    gap,
                    candidate_mask,
                    self.h,
                )

                loss.backward()
                optimizer.step()

                value = float(loss.item())
                losses.append(value)
                losses_by_dataset[ref.dataset].append(
                    value
                )

            if not losses:
                raise RuntimeError(
                    "no training blocks were processed"
                )

            if epoch % log_every == 0:
                dataset_text = ", ".join(
                    f"{name}={sum(values) / len(values):.4f}"
                    for name, values in losses_by_dataset.items()
                    if values
                )

                print(
                    f"epoch {epoch}: "
                    f"loss={sum(losses) / len(losses):.4f}, "
                    f"updates={len(losses)}, {dataset_text}"
                )

        return self

    @torch.no_grad()
    def evaluate_refs(
        self,
        refs: Sequence[BlockRef],
        horizon: int,
    ) -> Dict[str, Any]:
        if self.router is None:
            raise RuntimeError(
                "register_router() before evaluate()"
            )

        self.router.eval()

        recall_values = []
        pr_auc_values = []
        ndcg_values = []
        block_counts: Dict[str, int] = {}

        for ref in refs:
            x, order = self._load_block(ref)
            scores = self.router(x)

            evaluator = ScoreOrderEval(
                scores.cpu(),
                order.cpu(),
            )

            recall_values.append(
                evaluator.recall_at_h(horizon)
            )
            pr_auc_values.append(
                evaluator.pr_auc(horizon)
            )
            ndcg_values.append(
                evaluate_ndcg(
                    evaluator,
                    horizon,
                )
            )

            block_counts[ref.dataset] = (
                block_counts.get(ref.dataset, 0) + 1
            )

        if not recall_values:
            raise RuntimeError(
                "no evaluation blocks were processed"
            )

        return {
            f"recall@{horizon}": summ(
                torch.cat(recall_values)
            ),
            f"pr_auc@{horizon}": summ(
                torch.cat(pr_auc_values)
            ),
            f"ndcg@{horizon}": summ(
                torch.cat(ndcg_values)
            ),
            "n_blocks": len(refs),
            "blocks_by_dataset": block_counts,
        }

    def evaluate_all(
        self,
        *,
        eval_dataset_names: Sequence[str],
        horizon: int,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        pooled_refs: List[BlockRef] = []

        for name in eval_dataset_names:
            refs = self.splits[name].blocks_eval
            metrics[name] = self.evaluate_refs(
                refs,
                horizon,
            )
            pooled_refs.extend(refs)

        metrics["mixed_micro"] = self.evaluate_refs(
            pooled_refs,
            horizon,
        )

        return metrics


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def scenario_name(
    train_dataset_names: Sequence[str],
) -> str:
    if len(train_dataset_names) == 1:
        return f"train_{train_dataset_names[0]}"
    return "train_mixed_" + "_".join(
        train_dataset_names
    )


def run_scenario(
    *,
    train_dataset_names: Sequence[str],
    specs: Dict[str, DatasetSpec],
    splits: Dict[str, DatasetSplit],
    args: argparse.Namespace,
    feature_names: Sequence[str],
) -> Tuple[Optional[MultiDatasetRouterTrainer], Optional[Dict[str, Any]]]:
    name = scenario_name(train_dataset_names)
    eval_dataset_names = list(specs)

    config = {
        "train_datasets": list(train_dataset_names),
        "eval_datasets": eval_dataset_names,
        "datasets": {
            dataset_name: {
                **asdict(specs[dataset_name]),
                "num_train_samples": len(
                    splits[dataset_name].ids_train
                ),
                "num_eval_samples": len(
                    splits[dataset_name].ids_eval
                ),
                "num_train_blocks": len(
                    splits[dataset_name].blocks_train
                ),
                "num_eval_blocks": len(
                    splits[dataset_name].blocks_eval
                ),
            }
            for dataset_name in specs
        },
        "features": list(feature_names),
        "normalization": args.normalization,
        "loss": args.loss,
        "router": "mlp",
        "router_kwargs": {
            "dim_hidden": args.dim_hidden,
            "num_blocks_mlp": args.num_blocks_mlp,
        },
        "h": args.horizon,
        "num_epochs": args.epochs,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "holdout": args.holdout,
        "filter_result": args.filter_result,
        "seed": args.seed,
        "mix_strategy": (
            args.mix_strategy
            if len(train_dataset_names) > 1
            else "single_dataset"
        ),
    }

    print(
        f"\n[{args.stage}] {name}: "
        f"train={list(train_dataset_names)}, "
        f"eval={eval_dataset_names}"
    )

    try:
        # Recreate feature objects for every scenario so no state is shared
        # between models.
        feature_bundles = {
            dataset_name: build_feature_bundle(
                spec,
                feature_names,
                args.normalization,
                args.num_layers,
            )
            for dataset_name, spec in specs.items()
        }

        # Register one representative feature bundle to define router.dim_in.
        # During loading, the trainer uses the corresponding dataset's bundle.
        representative_name = train_dataset_names[0]

        torch.manual_seed(args.seed)

        router = FactoryRouter.create(
            "mlp",
            dim_hidden=args.dim_hidden,
            num_blocks_mlp=args.num_blocks_mlp,
        ).register_features(
            *feature_bundles[representative_name]
        )

        loss = FactoryLoss.create(args.loss)

        trainer = MultiDatasetRouterTrainer(
            splits=splits,
            feature_bundles=feature_bundles,
            h=args.horizon,
            device=args.device,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            mix_strategy=args.mix_strategy,
        )

        trainer.register_router(router)
        trainer.register_loss(loss)
        trainer.train(
            train_dataset_names=train_dataset_names,
            num_epochs=args.epochs,
        )

        metrics = {
            "evaluation": trainer.evaluate_all(
                eval_dataset_names=eval_dataset_names,
                horizon=args.horizon,
            )
        }

        os.makedirs(
            args.checkpoint_dir,
            exist_ok=True,
        )
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"{args.stage}__{name}.pt",
        )

        torch.save(
            {
                "state_dict": trainer.router.state_dict(),
                "router": trainer.router.describe(),
                "config": config,
            },
            checkpoint_path,
        )
        metrics["checkpoint"] = checkpoint_path

        save_result(
            stage=args.stage,
            name=name,
            config=config,
            metrics=metrics,
            error=None,
            report_path=args.report_path,
        )

        print(
            json.dumps(
                jsonable(metrics),
                indent=2,
            )
        )

        return trainer, metrics

    except Exception:
        error = traceback.format_exc()

        save_result(
            stage=args.stage,
            name=name,
            config=config,
            metrics=None,
            error=error,
            report_path=args.report_path,
        )

        print(error)
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    data_root = os.path.abspath(
        os.path.expanduser(args.data_root)
    )

    dataset_specs = [
        parse_dataset_argument(value, data_root)
        for value in args.dataset
    ]

    if len(dataset_specs) < 2:
        parser.error(
            "cross-dataset evaluation requires at least two --dataset values"
        )

    names = [spec.name for spec in dataset_specs]
    if len(names) != len(set(names)):
        parser.error("dataset names must be unique")

    feature_names = [
        name.strip()
        for name in args.features.split(",")
        if name.strip()
    ]
    if not feature_names:
        parser.error("--features cannot be empty")

    specs = {
        spec.name: spec
        for spec in dataset_specs
    }

    splits = {
        spec.name: build_dataset_split(
            spec,
            holdout=args.holdout,
            filter_result=args.filter_result,
        )
        for spec in dataset_specs
    }

    print("Dataset summary:")
    for name, split in splits.items():
        print(
            f"  {name}: folder={split.spec.folder}, "
            f"size_block={split.spec.size_block}, "
            f"train_samples={len(split.ids_train)}, "
            f"eval_samples={len(split.ids_eval)}, "
            f"train_blocks={len(split.blocks_train)}, "
            f"eval_blocks={len(split.blocks_eval)}"
        )

    if not args.append:
        reset_stage(
            args.stage,
            args.report_path,
        )

    scenarios: List[List[str]] = []

    if args.experiments in {"all", "individual"}:
        scenarios.extend([[name] for name in names])

    if args.experiments in {"all", "mixed"}:
        scenarios.append(names)

    for train_dataset_names in scenarios:
        run_scenario(
            train_dataset_names=train_dataset_names,
            specs=specs,
            splits=splits,
            args=args,
            feature_names=feature_names,
        )


if __name__ == "__main__":
    main()
