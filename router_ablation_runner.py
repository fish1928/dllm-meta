#!/usr/bin/env python3
"""Staged ablation runner for the LLaDA sparse-decoding router.

This script is intentionally separate from ``router_llada_v2.py``. It imports
that module and adds:

* reproducible stratified train/validation/test splits;
* declarative feature/wrapper/loss/router experiment configurations;
* staged ablation-plan generation (A, B1, B2, C1, C2, C3, D, E, F);
* shuffled training, validation, early stopping, and best checkpoints;
* fixed-budget ranking metrics, including next-hit, recall, MRR, NDCG, and AP;
* nontrivial-step metrics that exclude rows with <= h remaining candidates;
* JSON plans/results suitable for sharded execution across multiple GPUs.

Typical workflow
----------------
1. Generate and run sanity checks:
   python router_ablation_runner.py make-plan --stage A \
       --folder-data /path/to/stats --work-dir runs/router_ablation
   python router_ablation_runner.py run-plan \
       --plan runs/router_ablation/plans/stage_A.json --device cuda:0

2. Generate and run the loss x normalization screen:
   python router_ablation_runner.py make-plan --stage B1 ...
   python router_ablation_runner.py run-plan --plan .../stage_B1.json --device cuda:0

3. Promote the strongest B1 configurations and continue sequentially:
   python router_ablation_runner.py make-plan --stage B2 \
       --folder-data /path/to/stats --work-dir runs/router_ablation
   python router_ablation_runner.py run-plan --plan .../stage_B2.json --device cuda:0
   python router_ablation_runner.py make-plan --stage C1 ...
   ...

Use ``--num-shards`` and ``--shard-index`` with run-plan to split a plan across
GPUs or machines. Each experiment writes into a unique directory and is skipped
when a completed result already exists unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    tmp.replace(path)


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def stable_hash(value: Any, length: int = 10) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "experiment"


def mean_or_nan(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def std_or_nan(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0 if values else float("nan")


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Declarative experiment specification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WrapperSpec:
    """One wrapper in inner-to-outer application order."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    wrappers: tuple[WrapperSpec, ...] = ()
    sign: float = 1.0


@dataclass(frozen=True)
class RouterSpec:
    name: str = "mlp"
    kwargs: dict[str, Any] = field(
        default_factory=lambda: {"dim_hidden": 64, "num_blocks_mlp": 2}
    )


@dataclass(frozen=True)
class LossSpec:
    name: str = "uniform_within_h"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    stage: str
    features: list[FeatureSpec]
    router: RouterSpec = field(default_factory=RouterSpec)
    loss: LossSpec = field(default_factory=LossSpec)
    normalization_recipe: str = "raw"
    h: int = 5
    size_block: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-6
    seed: int = 233
    evaluate_test: bool = False
    cache_blocks: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        features = []
        for item in value["features"]:
            wrappers = tuple(WrapperSpec(**wrapper) for wrapper in item.get("wrappers", []))
            features.append(
                FeatureSpec(
                    name=item["name"],
                    kwargs=dict(item.get("kwargs", {})),
                    wrappers=wrappers,
                    sign=float(item.get("sign", 1.0)),
                )
            )
        return cls(
            name=value["name"],
            stage=value["stage"],
            features=features,
            router=RouterSpec(**value.get("router", {})),
            loss=LossSpec(**value.get("loss", {})),
            normalization_recipe=value.get("normalization_recipe", "raw"),
            h=int(value.get("h", 5)),
            size_block=int(value.get("size_block", 64)),
            lr=float(value.get("lr", 1e-3)),
            weight_decay=float(value.get("weight_decay", 1e-4)),
            max_epochs=int(value.get("max_epochs", 30)),
            patience=int(value.get("patience", 5)),
            min_delta=float(value.get("min_delta", 1e-6)),
            seed=int(value.get("seed", 233)),
            evaluate_test=bool(value.get("evaluate_test", False)),
            cache_blocks=bool(value.get("cache_blocks", False)),
            metadata=dict(value.get("metadata", {})),
        )


# -----------------------------------------------------------------------------
# Data split and block discovery
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockRef:
    id_sample: int
    pos_base: int
    block_index: int
    result: str


_UNMASK_RE = re.compile(r"^unmask_(-?\d+)_(-?\d+)\.pt$")


def read_sample_result(folder_data: Path, id_sample: int) -> str:
    path = folder_data / str(id_sample) / "generated.json"
    if not path.exists():
        return "unknown"
    try:
        value = json_load(path)
        return str(value.get("result", "unknown"))
    except (OSError, ValueError, TypeError):
        return "unknown"


def discover_sample_ids(folder_data: Path) -> list[int]:
    ids = sorted(int(path.name) for path in folder_data.iterdir() if path.is_dir() and path.name.isdigit())
    if not ids:
        raise RuntimeError(f"No numeric sample folders found under {folder_data}")
    return ids


def _allocate_group(
    ids: list[int],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 1:
        return ids, [], []
    if n == 2:
        return [ids[0]], [ids[1]], []

    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
    while n_val + n_test > n - 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    test = ids[:n_test]
    val = ids[n_test : n_test + n_val]
    train = ids[n_test + n_val :]
    return train, val, test


def create_split_manifest(
    folder_data: Path,
    manifest_path: Path,
    seed: int = 233,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    overwrite: bool = False,
) -> dict[str, Any]:
    if manifest_path.exists() and not overwrite:
        manifest = json_load(manifest_path)
        current_ids = set(discover_sample_ids(folder_data))
        manifest_ids = set(manifest["train"] + manifest["val"] + manifest["test"])
        missing = sorted(manifest_ids - current_ids)
        extra = sorted(current_ids - manifest_ids)
        if missing:
            raise RuntimeError(f"Split manifest references missing sample folders: {missing[:10]}")
        if extra:
            raise RuntimeError(
                f"New sample folders are absent from the existing split manifest: {extra[:10]}. "
                "Use --overwrite-split intentionally to regenerate all splits."
            )
        return manifest

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for id_sample in discover_sample_ids(folder_data):
        groups[read_sample_result(folder_data, id_sample)].append(id_sample)

    rng = random.Random(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for result_name in sorted(groups):
        g_train, g_val, g_test = _allocate_group(
            groups[result_name], rng, train_ratio, val_ratio, test_ratio
        )
        train.extend(g_train)
        val.extend(g_val)
        test.extend(g_test)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # For tiny or heavily fragmented datasets, guarantee global validation/test
    # sets when enough total samples are available.
    if not val and len(train) >= 2:
        val.append(train.pop())
    if not test and len(train) >= 2:
        test.append(train.pop())
    if not train:
        raise RuntimeError("The split procedure produced an empty training set")

    manifest = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "stratify_field": "generated.json:result",
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "result_counts": {
            split_name: dict(
                sorted(
                    (
                        result_name,
                        sum(read_sample_result(folder_data, i) == result_name for i in ids),
                    )
                    for result_name in groups
                )
            )
            for split_name, ids in (("train", train), ("val", val), ("test", test))
        },
    }
    manifest["fingerprint"] = stable_hash(
        {key: manifest[key] for key in ("seed", "ratios", "train", "val", "test")},
        length=16,
    )
    json_dump(manifest, manifest_path)
    return manifest


def list_blocks(folder_data: Path, ids: Sequence[int], size_block: int) -> list[BlockRef]:
    refs: list[BlockRef] = []
    for id_sample in ids:
        sample_folder = folder_data / str(id_sample)
        result = read_sample_result(folder_data, id_sample)
        found: list[tuple[int, int]] = []
        for path in sample_folder.iterdir():
            match = _UNMASK_RE.match(path.name)
            if not match:
                continue
            start, end = int(match.group(1)), int(match.group(2))
            if end - start != size_block:
                continue
            found.append((start, end))
        found.sort()
        if not found:
            raise RuntimeError(
                f"No unmask_<start>_<end>.pt blocks of size {size_block} found in {sample_folder}"
            )
        for block_index, (pos_base, _) in enumerate(found):
            refs.append(
                BlockRef(
                    id_sample=id_sample,
                    pos_base=pos_base,
                    block_index=block_index,
                    result=result,
                )
            )
    return refs


# -----------------------------------------------------------------------------
# Feature and checkpoint helpers
# -----------------------------------------------------------------------------


class SignedFeature:
    """Lightweight wrapper that multiplies a feature by a constant sign."""

    def __init__(self, inner: Any, sign: float):
        self.feature_inner = inner
        self.sign = float(sign)
        self.folder_data = inner.folder_data

    def dim(self) -> int:
        return self.feature_inner.dim()

    def get_name(self) -> str:
        return f"sign{self.sign:g}({self.feature_inner.get_name()})"

    def load_block(self, id_sample: int, pos_base: int, size_block: int) -> torch.Tensor:
        return self.feature_inner.load_block(id_sample, pos_base, size_block) * self.sign


def build_feature(router_module: Any, folder_data: str, spec: FeatureSpec) -> Any:
    feature = router_module.FactoryFeature.create(spec.name, folder_data, **spec.kwargs)
    for wrapper in spec.wrappers:
        feature = router_module.FactoryFeature.wrap(wrapper.name, feature, **wrapper.kwargs)
    if spec.sign != 1.0:
        feature = SignedFeature(feature, spec.sign)
    return feature


def walk_feature_tree(feature: Any) -> Iterator[Any]:
    inner = getattr(feature, "feature_inner", None)
    if inner is not None:
        yield from walk_feature_tree(inner)
    yield feature


def fit_feature_tree(feature: Any, blocks: list[tuple[int, int]], size_block: int) -> None:
    """Fit every dataset-fitted wrapper, inner to outer."""

    for node in walk_feature_tree(feature):
        fitted = getattr(node, "fitted", None)
        fit = getattr(node, "fit", None)
        if callable(fitted) and callable(fit) and not fitted():
            fit(blocks, size_block)


def collect_feature_state(feature: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"name": feature.get_name() if hasattr(feature, "get_name") else type(feature).__name__}
    get_state = getattr(feature, "get_state", None)
    if callable(get_state):
        state["state"] = get_state()
    inner = getattr(feature, "feature_inner", None)
    if inner is not None:
        state["inner"] = collect_feature_state(inner)
    return state


def save_router_checkpoint(
    path: Path,
    router: Any,
    config: ExperimentConfig,
    split_fingerprint: str,
    validation_metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": router.state_dict(),
            "config": config.to_dict(),
            "split_fingerprint": split_fingerprint,
            "features_state": [collect_feature_state(feature) for feature in router.features],
            "validation_metrics": dict(validation_metrics),
        },
        path,
    )


# -----------------------------------------------------------------------------
# Loss construction
# -----------------------------------------------------------------------------


def compute_balanced_pos_weight(
    router_module: Any,
    folder_data: Path,
    refs: Sequence[BlockRef],
    size_block: int,
    h: int,
) -> float:
    positives = 0
    negatives = 0
    for ref in refs:
        folder_base = folder_data / str(ref.id_sample)
        unmask = router_module.load_stat(str(folder_base), "unmask", ref.pos_base, size_block)
        order = unmask.squeeze(-1).long() - ref.pos_base
        gap, cand_mask = router_module.build_geometry(order, size_block)
        pos = (gap >= 1) & (gap <= h) & cand_mask
        positives += int(pos.sum())
        negatives += int((cand_mask & ~pos).sum())
    if positives == 0:
        raise RuntimeError("Cannot compute balanced BCE pos_weight: no positive labels")
    return negatives / positives


def build_loss(
    router_module: Any,
    config: ExperimentConfig,
    folder_data: Path,
    train_refs: Sequence[BlockRef],
) -> tuple[Any, dict[str, Any]]:
    kwargs = dict(config.loss.kwargs)
    runtime: dict[str, Any] = {}

    if config.loss.name == "bce_within_h":
        mode = kwargs.pop("mode", "unweighted")
        if mode == "balanced":
            pos_weight = compute_balanced_pos_weight(
                router_module, folder_data, train_refs, config.size_block, config.h
            )
            kwargs["pos_weight"] = pos_weight
            runtime["bce_pos_weight"] = pos_weight
        elif mode != "unweighted":
            raise ValueError(f"Unsupported BCE mode: {mode}")
        runtime["bce_mode"] = mode

    return router_module.FactoryLoss.create(config.loss.name, **kwargs), runtime


# -----------------------------------------------------------------------------
# Ranking metrics
# -----------------------------------------------------------------------------


class MetricBucket:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def add(self, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            if math.isfinite(float(value)):
                self.values[name].append(float(value))

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in sorted(self.values):
            values = self.values[name]
            result[name] = {
                "mean": finite_or_none(mean_or_nan(values)),
                "std": finite_or_none(std_or_nan(values)),
                "count": len(values),
            }
        return result


def _stable_descending_argsort(values: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:  # older PyTorch
        return torch.argsort(values, descending=True)


def compute_row_metrics(
    scores_row: torch.Tensor,
    gap_row: torch.Tensor,
    cand_row: torch.Tensor,
    h: int,
) -> dict[str, float] | None:
    candidate_positions = torch.nonzero(cand_row, as_tuple=False).flatten()
    n_cand = int(candidate_positions.numel())
    if n_cand == 0:
        return None

    candidate_scores = scores_row[candidate_positions]
    ranked_local = _stable_descending_argsort(candidate_scores)
    ranked_positions = candidate_positions[ranked_local]
    ranked_gaps = gap_row[ranked_positions]

    k = min(h, n_cand)
    top_gaps = ranked_gaps[:k]
    next_target_rank = torch.nonzero(ranked_gaps == 1, as_tuple=False).flatten()
    if next_target_rank.numel() != 1:
        raise RuntimeError("Expected exactly one gap==1 candidate in every non-empty row")
    next_rank_1based = int(next_target_rank.item()) + 1

    positives = (gap_row[candidate_positions] >= 1) & (gap_row[candidate_positions] <= h)
    n_pos = int(positives.sum())
    recall = float(((top_gaps >= 1) & (top_gaps <= h)).sum().item() / max(n_pos, 1))
    next_hit = float(next_rank_1based <= h)
    mrr = 1.0 / next_rank_1based

    ranked_positive = (ranked_gaps >= 1) & (ranked_gaps <= h)
    if n_pos > 0:
        cumulative = ranked_positive.float().cumsum(0)
        ranks = torch.arange(1, n_cand + 1, dtype=torch.float32)
        precisions = cumulative / ranks
        recalls = cumulative / float(n_pos)
        average_precision = float(precisions[ranked_positive].mean().item())

        # Trapezoidal area under the discrete precision-recall curve. Average
        # precision is also reported because many ranking libraries use AP as
        # their summary of a PR curve.
        precision_curve = torch.cat([torch.ones(1), precisions])
        recall_curve = torch.cat([torch.zeros(1), recalls])
        pr_auc = float(torch.trapz(precision_curve, recall_curve).item())
    else:
        average_precision = float("nan")
        pr_auc = float("nan")

    relevance = torch.where(
        (ranked_gaps >= 1) & (ranked_gaps <= h),
        (h + 1 - ranked_gaps).float(),
        torch.zeros_like(ranked_gaps, dtype=torch.float32),
    )
    gains = torch.pow(2.0, relevance[:k]) - 1.0
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    dcg = float((gains / discounts).sum().item())
    ideal = torch.sort(relevance, descending=True).values[:k]
    idcg = float(((torch.pow(2.0, ideal) - 1.0) / discounts).sum().item())
    ndcg = dcg / idcg if idcg > 0 else float("nan")

    return {
        f"next_hit@{h}": next_hit,
        f"recall@{h}": recall,
        "mrr_next": mrr,
        f"ndcg@{h}": ndcg,
        f"average_precision@{h}": average_precision,
        f"pr_auc@{h}": pr_auc,
        "n_candidates": float(n_cand),
    }


def aggregate_block_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    names = sorted({name for row in rows for name in row})
    return {
        name: mean_or_nan([float(row[name]) for row in rows if name in row and math.isfinite(float(row[name]))])
        for name in names
    }


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class AblationTrainer:
    def __init__(
        self,
        router_module: Any,
        folder_data: Path,
        config: ExperimentConfig,
        split_manifest: Mapping[str, Any],
        device: str,
        output_dir: Path,
    ) -> None:
        self.m = router_module
        self.folder_data = folder_data
        self.config = config
        self.split_manifest = split_manifest
        self.device = torch.device(device)
        self.output_dir = output_dir

        self.train_refs = list_blocks(folder_data, split_manifest["train"], config.size_block)
        self.val_refs = list_blocks(folder_data, split_manifest["val"], config.size_block)
        self.test_refs = list_blocks(folder_data, split_manifest["test"], config.size_block)
        if not self.val_refs:
            raise RuntimeError("Validation split contains no blocks")

        features = [build_feature(self.m, str(folder_data), spec) for spec in config.features]
        self.router = self.m.FactoryRouter.create(config.router.name, **config.router.kwargs)
        self.router.register_features(*features)

        train_blocks = [(ref.id_sample, ref.pos_base) for ref in self.train_refs]
        for feature in self.router.features:
            fit_feature_tree(feature, train_blocks, config.size_block)

        self.router = self.router.to(self.device)
        self.loss, self.loss_runtime = build_loss(
            self.m, config, folder_data, self.train_refs
        )
        self.cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _load_block(self, ref: BlockRef) -> tuple[torch.Tensor, torch.Tensor]:
        key = (ref.id_sample, ref.pos_base)
        if self.config.cache_blocks and key in self.cache:
            x_cpu, order_cpu = self.cache[key]
        else:
            x_cpu = self.router.build_block_x(
                ref.id_sample, ref.pos_base, self.config.size_block
            ).float().cpu()
            folder_base = self.folder_data / str(ref.id_sample)
            unmask = self.m.load_stat(
                str(folder_base), "unmask", ref.pos_base, self.config.size_block
            )
            order_cpu = (unmask.squeeze(-1).long() - ref.pos_base).cpu()
            if self.config.cache_blocks:
                self.cache[key] = (x_cpu, order_cpu)
        return x_cpu.to(self.device), order_cpu.to(self.device)

    def _train_epoch(self, optimizer: torch.optim.Optimizer, refs: list[BlockRef], seed: int) -> float:
        self.router.train()
        refs = list(refs)
        random.Random(seed).shuffle(refs)
        losses: list[float] = []
        for ref in refs:
            x, order = self._load_block(ref)
            gap, cand_mask = self.m.build_geometry(order.cpu(), self.config.size_block)
            gap = gap.to(self.device)
            cand_mask = cand_mask.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            scores = self.router(x)
            loss = self.loss(scores, gap, cand_mask, self.config.h)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss for {self.config.name}, block={ref}"
                )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        return mean_or_nan(losses)

    @torch.no_grad()
    def evaluate(self, refs: Sequence[BlockRef], split_name: str) -> dict[str, Any]:
        self.router.eval()
        overall = MetricBucket()
        nontrivial = MetricBucket()
        by_stratum: dict[str, MetricBucket] = defaultdict(MetricBucket)
        by_result: dict[str, MetricBucket] = defaultdict(MetricBucket)
        by_block_index: dict[int, MetricBucket] = defaultdict(MetricBucket)
        per_block: list[dict[str, Any]] = []

        for ref in refs:
            x, order = self._load_block(ref)
            scores = self.router(x).detach().float().cpu()
            order_cpu = order.detach().cpu()
            gap, cand_mask = self.m.build_geometry(order_cpu, self.config.size_block)

            row_values: list[dict[str, float]] = []
            for t in range(scores.shape[0]):
                metrics = compute_row_metrics(scores[t], gap[t], cand_mask[t], self.config.h)
                if metrics is None:
                    continue
                row_values.append(metrics)
                overall.add(metrics)
                by_result[ref.result].add(metrics)
                by_block_index[ref.block_index].add(metrics)

                n_cand = int(metrics["n_candidates"])
                if n_cand > 2 * self.config.h:
                    stratum = "early"
                elif n_cand > self.config.h:
                    stratum = "middle"
                else:
                    stratum = "late"
                by_stratum[stratum].add(metrics)
                if n_cand > self.config.h:
                    nontrivial.add(metrics)

            per_block.append(
                {
                    "id_sample": ref.id_sample,
                    "pos_base": ref.pos_base,
                    "block_index": ref.block_index,
                    "result": ref.result,
                    "metrics": {
                        key: finite_or_none(value)
                        for key, value in aggregate_block_metrics(row_values).items()
                    },
                    "n_rows": len(row_values),
                }
            )

        report = {
            "split": split_name,
            "overall": overall.summary(),
            "nontrivial": nontrivial.summary(),
            "by_stratum": {name: bucket.summary() for name, bucket in sorted(by_stratum.items())},
            "by_result": {name: bucket.summary() for name, bucket in sorted(by_result.items())},
            "by_block_index": {
                str(index): bucket.summary() for index, bucket in sorted(by_block_index.items())
            },
            "n_blocks": len(refs),
            "per_block": per_block,
        }
        return report

    def _selection_tuple(self, report: Mapping[str, Any]) -> tuple[float, float, float]:
        scope = report["nontrivial"] if report["nontrivial"] else report["overall"]

        def value(name: str) -> float:
            item = scope.get(name)
            if not item or item.get("mean") is None:
                return float("-inf")
            return float(item["mean"])

        h = self.config.h
        return value(f"next_hit@{h}"), value(f"recall@{h}"), value(f"ndcg@{h}")

    def run(self) -> dict[str, Any]:
        set_all_seeds(self.config.seed)
        started = time.time()
        history: list[dict[str, Any]] = []

        if self.router.trainable():
            optimizer = torch.optim.AdamW(
                self.router.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            best_tuple = (float("-inf"),) * 3
            best_state: dict[str, torch.Tensor] | None = None
            best_epoch = -1
            epochs_without_improvement = 0

            for epoch in range(self.config.max_epochs):
                train_loss = self._train_epoch(
                    optimizer, self.train_refs, seed=self.config.seed + epoch
                )
                val_report = self.evaluate(self.val_refs, "val")
                selection = self._selection_tuple(val_report)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": finite_or_none(train_loss),
                        "selection": list(selection),
                        "val": {key: value for key, value in val_report.items() if key != "per_block"},
                    }
                )

                improved = selection[0] > best_tuple[0] + self.config.min_delta
                if not improved and abs(selection[0] - best_tuple[0]) <= self.config.min_delta:
                    improved = selection[1:] > best_tuple[1:]

                if improved:
                    best_tuple = selection
                    best_epoch = epoch
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.router.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                print(
                    f"[{self.config.name}] epoch={epoch:03d} loss={train_loss:.6f} "
                    f"selection={selection} best_epoch={best_epoch}",
                    flush=True,
                )
                if epochs_without_improvement >= self.config.patience:
                    break

            if best_state is None:
                raise RuntimeError("No valid best checkpoint was produced")
            self.router.load_state_dict(best_state)
        else:
            best_epoch = -1

        val_report = self.evaluate(self.val_refs, "val")
        test_report = self.evaluate(self.test_refs, "test") if self.config.evaluate_test else None
        elapsed = time.time() - started

        result = {
            "status": "complete",
            "config": self.config.to_dict(),
            "config_hash": stable_hash(self.config.to_dict(), length=16),
            "split_fingerprint": self.split_manifest["fingerprint"],
            "loss_runtime": self.loss_runtime,
            "best_epoch": best_epoch,
            "selection_tuple": list(self._selection_tuple(val_report)),
            "val": val_report,
            "test": test_report,
            "elapsed_seconds": elapsed,
            "parameter_count": sum(parameter.numel() for parameter in self.router.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.router.parameters() if parameter.requires_grad
            ),
            "router_description": self.router.describe(),
            "history": history,
        }

        save_router_checkpoint(
            self.output_dir / "best.pt",
            self.router,
            self.config,
            self.split_manifest["fingerprint"],
            val_report,
        )
        json_dump(result, self.output_dir / "result.json")
        return result


# -----------------------------------------------------------------------------
# Feature recipes and staged plan generation
# -----------------------------------------------------------------------------


STATE_FEATURES = {"conf", "margin", "entropy"}
ATTENTION_FEATURES = {"attn_last", "attn_all"}
RAW_FEATURES = {"pos_delta", "step_progress", "mask_density", "x0_stability"}


NORMALIZATION_RECIPES = (
    "raw",
    "znorm_row",
    "rank",
    "minmax_row",
    "znorm_global",
    "log_znorm_row",
    "softmax_attn",
)


def wrappers_for_feature(
    feature_name: str,
    recipe: str,
    softmax_temperature: float = 1.0,
) -> tuple[WrapperSpec, ...]:
    if recipe == "raw":
        return ()
    if feature_name in RAW_FEATURES:
        return ()

    if recipe == "znorm_row":
        return (WrapperSpec("znorm_row"),)
    if recipe == "rank":
        return (WrapperSpec("rank"),)
    if recipe == "minmax_row":
        return (WrapperSpec("minmax_row"),)
    if recipe == "znorm_global":
        return (WrapperSpec("znorm_global"),)
    if recipe == "log_znorm_row":
        if feature_name in ATTENTION_FEATURES:
            return (WrapperSpec("log"), WrapperSpec("znorm_row"))
        if feature_name in STATE_FEATURES:
            return (WrapperSpec("znorm_row"),)
        return ()
    if recipe == "softmax_attn":
        if feature_name in ATTENTION_FEATURES:
            return (WrapperSpec("softmax_row", {"temperature": softmax_temperature}),)
        if feature_name in STATE_FEATURES:
            return (WrapperSpec("znorm_row"),)
        return ()
    raise ValueError(f"Unknown normalization recipe: {recipe}")


def feature_spec(
    name: str,
    recipe: str,
    num_layers: int,
    mask_density_window: int,
    softmax_temperature: float = 1.0,
    sign: float = 1.0,
) -> FeatureSpec:
    kwargs: dict[str, Any] = {}
    if name == "attn_all":
        kwargs["num_layers"] = num_layers
    if name == "mask_density":
        kwargs["window"] = mask_density_window
    return FeatureSpec(
        name=name,
        kwargs=kwargs,
        wrappers=wrappers_for_feature(name, recipe, softmax_temperature),
        sign=sign,
    )


def make_features(
    names: Sequence[str],
    recipe: str,
    num_layers: int,
    mask_density_window: int,
    softmax_temperature: float = 1.0,
    signs: Mapping[str, float] | None = None,
) -> list[FeatureSpec]:
    signs = signs or {}
    return [
        feature_spec(
            name,
            recipe,
            num_layers,
            mask_density_window,
            softmax_temperature,
            signs.get(name, 1.0),
        )
        for name in names
    ]


def loss_variants() -> list[LossSpec]:
    return [
        LossSpec("uniform_within_h"),
        LossSpec("decay_within_h"),
        LossSpec("bce_within_h", {"mode": "unweighted"}),
        LossSpec("bce_within_h", {"mode": "balanced"}),
        LossSpec("plackett_luce"),
    ]


def loss_label(loss: LossSpec) -> str:
    if loss.name == "bce_within_h":
        return f"bce-{loss.kwargs.get('mode', 'unweighted')}"
    return loss.name.replace("_within_h", "")


def feature_set_key(config: ExperimentConfig) -> tuple[str, ...]:
    return tuple(spec.name for spec in config.features)


def recipe_loss_key(config: ExperimentConfig) -> tuple[str, str, str]:
    return (
        config.normalization_recipe,
        config.loss.name,
        str(config.loss.kwargs.get("mode", "")),
    )


def base_experiment(
    *,
    name: str,
    stage: str,
    features: list[FeatureSpec],
    recipe: str,
    loss: LossSpec,
    seed: int,
    h: int,
    size_block: int,
    max_epochs: int,
    patience: int,
    evaluate_test: bool = False,
    router: RouterSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=safe_name(name),
        stage=stage,
        features=features,
        router=router or RouterSpec(),
        loss=loss,
        normalization_recipe=recipe,
        h=h,
        size_block=size_block,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        evaluate_test=evaluate_test,
        metadata=metadata or {},
    )


def clone_with_seed(config: ExperimentConfig, seed: int, stage: str | None = None) -> ExperimentConfig:
    cloned = copy.deepcopy(config)
    cloned.seed = seed
    old_stage = cloned.stage
    if stage is not None:
        cloned.stage = stage

    # Plans append a short hexadecimal hash to each readable name. Remove both
    # the prior seed and optional hash before creating the promoted name.
    base_name = re.sub(r"-seed\d+(?:-[0-9a-f]{10})?$", "", cloned.name)
    if stage is not None:
        base_name = re.sub(rf"^{re.escape(old_stage)}-", f"{stage}-", base_name)
    cloned.name = safe_name(base_name + f"-seed{seed}")
    return cloned


def stage_a_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    common = dict(
        num_layers=args.num_layers,
        mask_density_window=args.mask_density_window,
    )
    configs: list[ExperimentConfig] = []

    def add(name: str, router: RouterSpec, names: list[str], signs: Mapping[str, float] | None = None) -> None:
        configs.append(
            base_experiment(
                name=f"A-{name}-seed{args.seed}",
                stage="A",
                features=make_features(names, "raw", signs=signs, **common),
                recipe="raw",
                loss=LossSpec("uniform_within_h"),
                seed=args.seed,
                h=args.h,
                size_block=args.size_block,
                max_epochs=args.full_epochs,
                patience=args.full_patience,
                router=router,
                metadata={"purpose": "sanity_baseline"},
            )
        )

    add("random", RouterSpec("mockup_random", {"seed": args.seed}), ["pos_delta"])
    add("nearest-right", RouterSpec("mockup_nearest_right", {}), ["pos_delta"])
    add("raw-attn-last", RouterSpec("mockup_raw", {}), ["attn_last"])
    add("raw-conf", RouterSpec("mockup_raw", {}), ["conf"])
    add("raw-margin", RouterSpec("mockup_raw", {}), ["margin"])
    add("raw-neg-entropy", RouterSpec("mockup_raw", {}), ["entropy"], {"entropy": -1.0})
    add("raw-x0-stability", RouterSpec("mockup_raw", {}), ["x0_stability"])
    add(
        "linear-core",
        RouterSpec("linear", {}),
        ["attn_last", "conf", "margin", "pos_delta", "mask_density"],
    )
    return configs


def stage_b1_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    anchors = {
        "attn": ["attn_last"],
        "core": ["attn_last", "conf", "margin", "pos_delta", "mask_density"],
    }
    configs: list[ExperimentConfig] = []
    for anchor_name, names in anchors.items():
        for recipe in NORMALIZATION_RECIPES:
            for loss in loss_variants():
                label = f"B1-{anchor_name}-{recipe}-{loss_label(loss)}-seed{args.seed}"
                configs.append(
                    base_experiment(
                        name=label,
                        stage="B1",
                        features=make_features(
                            names,
                            recipe,
                            args.num_layers,
                            args.mask_density_window,
                            args.softmax_temperature,
                        ),
                        recipe=recipe,
                        loss=loss,
                        seed=args.seed,
                        h=args.h,
                        size_block=args.size_block,
                        max_epochs=args.screen_epochs,
                        patience=args.screen_patience,
                        metadata={"anchor": anchor_name, "screen": True},
                    )
                )
    return configs


def load_completed_results(work_dir: Path, stages: set[str] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    root = work_dir / "results"
    if not root.exists():
        return results
    for path in root.rglob("result.json"):
        try:
            value = json_load(path)
        except (OSError, ValueError):
            continue
        if value.get("status") != "complete":
            continue
        stage = value.get("config", {}).get("stage")
        if stages is not None and stage not in stages:
            continue
        value["_result_path"] = str(path)
        results.append(value)
    return results


def result_score(result: Mapping[str, Any]) -> tuple[float, float, float]:
    selection = result.get("selection_tuple", [])
    if len(selection) != 3:
        return (float("-inf"),) * 3
    return tuple(float(value) for value in selection)  # type: ignore[return-value]


def aggregate_result_groups(
    results: Sequence[Mapping[str, Any]],
    key_fn: Any,
) -> list[tuple[Any, tuple[float, float, float], list[Mapping[str, Any]]]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        config = ExperimentConfig.from_dict(result["config"])
        groups[key_fn(config)].append(result)

    aggregated = []
    for key, members in groups.items():
        scores = [result_score(member) for member in members]
        mean_score = tuple(statistics.fmean(score[i] for score in scores) for i in range(3))
        aggregated.append((key, mean_score, members))
    aggregated.sort(key=lambda item: item[1], reverse=True)
    return aggregated


def stage_b2_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    results = load_completed_results(Path(args.work_dir), {"B1"})
    if not results:
        raise RuntimeError("No completed B1 results found; run stage B1 first")
    results.sort(key=result_score, reverse=True)
    promoted = results[: args.promote_count]
    configs: list[ExperimentConfig] = []
    for result in promoted:
        base = ExperimentConfig.from_dict(result["config"])
        for seed in args.full_seeds:
            cfg = clone_with_seed(base, seed, stage="B2")
            cfg.max_epochs = args.full_epochs
            cfg.patience = args.full_patience
            cfg.metadata = {**cfg.metadata, "promoted_from": result["config"]["name"]}
            configs.append(cfg)
    return configs


def best_recipe_and_loss(work_dir: Path) -> tuple[str, LossSpec]:
    results = load_completed_results(work_dir, {"B2"})
    if not results:
        raise RuntimeError("No completed B2 results found; run stage B2 first")
    grouped = aggregate_result_groups(results, recipe_loss_key)
    recipe, loss_name, mode = grouped[0][0]
    kwargs = {"mode": mode} if loss_name == "bce_within_h" and mode else {}
    return recipe, LossSpec(loss_name, kwargs)


def stage_c1_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    recipe, loss = best_recipe_and_loss(Path(args.work_dir))
    sets = {
        "attn-last": ["attn_last"],
        "attn-all": ["attn_all"],
        "attn-all-plus-last": ["attn_all", "attn_last"],
        "conf": ["conf"],
        "margin": ["margin"],
        "entropy": ["entropy"],
        "pos-delta": ["pos_delta"],
        "mask-density": ["mask_density"],
        "x0-stability": ["x0_stability"],
    }
    configs: list[ExperimentConfig] = []
    for set_name, names in sets.items():
        for seed in args.full_seeds:
            configs.append(
                base_experiment(
                    name=f"C1-{set_name}-{recipe}-{loss_label(loss)}-seed{seed}",
                    stage="C1",
                    features=make_features(
                        names,
                        recipe,
                        args.num_layers,
                        args.mask_density_window,
                        args.softmax_temperature,
                    ),
                    recipe=recipe,
                    loss=loss,
                    seed=seed,
                    h=args.h,
                    size_block=args.size_block,
                    max_epochs=args.full_epochs,
                    patience=args.full_patience,
                    metadata={"feature_set_name": set_name},
                )
            )
    return configs


def best_attention_set(work_dir: Path) -> tuple[str, ...]:
    results = load_completed_results(work_dir, {"C1"})
    if not results:
        raise RuntimeError("No completed C1 results found; run stage C1 first")
    allowed = {
        ("attn_last",),
        ("attn_all",),
        ("attn_all", "attn_last"),
    }
    filtered = [
        result
        for result in results
        if feature_set_key(ExperimentConfig.from_dict(result["config"])) in allowed
    ]
    grouped = aggregate_result_groups(filtered, feature_set_key)
    if not grouped:
        raise RuntimeError("C1 results contain no attention representation experiments")
    return tuple(grouped[0][0])


def stage_c2_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    work_dir = Path(args.work_dir)
    recipe, loss = best_recipe_and_loss(work_dir)
    attention = list(best_attention_set(work_dir))

    add_one = {
        "attention": attention,
        "plus-conf": attention + ["conf"],
        "plus-margin": attention + ["margin"],
        "plus-entropy": attention + ["entropy"],
        "plus-pos-delta": attention + ["pos_delta"],
        "plus-mask-density": attention + ["mask_density"],
        "plus-x0-stability": attention + ["x0_stability"],
        "plus-step-progress": attention + ["step_progress"],
    }
    state = ["conf", "margin", "entropy", "x0_stability"]
    geometry = ["pos_delta", "mask_density"]
    groups = {
        "state-only": state,
        "geometry-only": geometry,
        "attention-plus-state": attention + state,
        "attention-plus-geometry": attention + geometry,
        "attention-plus-state-plus-geometry": attention + state + geometry,
        "all-plus-progress": attention + state + geometry + ["step_progress"],
    }
    feature_sets = {**add_one, **groups}

    configs: list[ExperimentConfig] = []
    for set_name, names in feature_sets.items():
        for seed in args.full_seeds:
            configs.append(
                base_experiment(
                    name=f"C2-{set_name}-{recipe}-{loss_label(loss)}-seed{seed}",
                    stage="C2",
                    features=make_features(
                        names,
                        recipe,
                        args.num_layers,
                        args.mask_density_window,
                        args.softmax_temperature,
                    ),
                    recipe=recipe,
                    loss=loss,
                    seed=seed,
                    h=args.h,
                    size_block=args.size_block,
                    max_epochs=args.full_epochs,
                    patience=args.full_patience,
                    metadata={
                        "feature_set_name": set_name,
                        "selected_attention": attention,
                    },
                )
            )
    return configs


def best_feature_config(work_dir: Path, stages: set[str]) -> ExperimentConfig:
    results = load_completed_results(work_dir, stages)
    if not results:
        raise RuntimeError(f"No completed results found for stages {sorted(stages)}")
    grouped = aggregate_result_groups(results, feature_set_key)
    return ExperimentConfig.from_dict(grouped[0][2][0]["config"])


def stage_c3_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    work_dir = Path(args.work_dir)
    best = best_feature_config(work_dir, {"C2"})
    full_names = [spec.name for spec in best.features]
    if len(full_names) < 2:
        raise RuntimeError("The best C2 feature set has fewer than two features; leave-one-out is undefined")

    configs: list[ExperimentConfig] = []
    candidates = [("full", full_names)] + [
        (f"minus-{name}-{index}", full_names[:index] + full_names[index + 1 :])
        for index, name in enumerate(full_names)
    ]
    for set_name, names in candidates:
        for seed in args.full_seeds:
            configs.append(
                base_experiment(
                    name=f"C3-{set_name}-{best.normalization_recipe}-{loss_label(best.loss)}-seed{seed}",
                    stage="C3",
                    features=make_features(
                        names,
                        best.normalization_recipe,
                        args.num_layers,
                        args.mask_density_window,
                        args.softmax_temperature,
                    ),
                    recipe=best.normalization_recipe,
                    loss=best.loss,
                    seed=seed,
                    h=args.h,
                    size_block=args.size_block,
                    max_epochs=args.full_epochs,
                    patience=args.full_patience,
                    metadata={"leave_one_out_from": full_names, "variant": set_name},
                )
            )
    return configs


def top_unique_recipe_losses(work_dir: Path, count: int) -> list[tuple[str, LossSpec]]:
    results = load_completed_results(work_dir, {"B2"})
    grouped = aggregate_result_groups(results, recipe_loss_key)
    output = []
    for (recipe, loss_name, mode), _, _ in grouped[:count]:
        kwargs = {"mode": mode} if loss_name == "bce_within_h" and mode else {}
        output.append((recipe, LossSpec(loss_name, kwargs)))
    return output


def top_unique_recipes(work_dir: Path, count: int) -> list[str]:
    results = load_completed_results(work_dir, {"B2"})
    grouped = aggregate_result_groups(results, lambda cfg: cfg.normalization_recipe)
    return [str(item[0]) for item in grouped[:count]]


def top_unique_losses(work_dir: Path, count: int) -> list[LossSpec]:
    results = load_completed_results(work_dir, {"B2"})

    def key(cfg: ExperimentConfig) -> tuple[str, str]:
        return cfg.loss.name, str(cfg.loss.kwargs.get("mode", ""))

    grouped = aggregate_result_groups(results, key)
    output = []
    for (loss_name, mode), _, _ in grouped[:count]:
        kwargs = {"mode": mode} if loss_name == "bce_within_h" and mode else {}
        output.append(LossSpec(loss_name, kwargs))
    return output


def top_feature_sets(work_dir: Path, count: int) -> list[tuple[str, ...]]:
    results = load_completed_results(work_dir, {"C2", "C3"})
    grouped = aggregate_result_groups(results, feature_set_key)
    return [tuple(item[0]) for item in grouped[:count]]


def stage_d_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    work_dir = Path(args.work_dir)
    feature_sets = top_feature_sets(work_dir, 3)
    recipes = top_unique_recipes(work_dir, 2)
    losses = top_unique_losses(work_dir, 2)
    if len(feature_sets) < 1 or len(recipes) < 1 or len(losses) < 1:
        raise RuntimeError("Insufficient B2/C2/C3 results to construct stage D")

    configs: list[ExperimentConfig] = []
    for set_index, names in enumerate(feature_sets):
        for recipe in recipes:
            for loss in losses:
                for seed in args.final_seeds:
                    configs.append(
                        base_experiment(
                            name=(
                                f"D-fset{set_index}-{recipe}-{loss_label(loss)}-seed{seed}"
                            ),
                            stage="D",
                            features=make_features(
                                names,
                                recipe,
                                args.num_layers,
                                args.mask_density_window,
                                args.softmax_temperature,
                            ),
                            recipe=recipe,
                            loss=loss,
                            seed=seed,
                            h=args.h,
                            size_block=args.size_block,
                            max_epochs=args.full_epochs,
                            patience=args.full_patience,
                            metadata={"feature_set": list(names), "factorial_confirmation": True},
                        )
                    )
    return configs


def best_full_config(work_dir: Path, stages: set[str]) -> ExperimentConfig:
    results = load_completed_results(work_dir, stages)
    if not results:
        raise RuntimeError(f"No completed results found for stages {sorted(stages)}")

    def key(cfg: ExperimentConfig) -> tuple[Any, ...]:
        return (
            feature_set_key(cfg),
            cfg.normalization_recipe,
            cfg.loss.name,
            str(cfg.loss.kwargs.get("mode", "")),
            cfg.router.name,
        )

    grouped = aggregate_result_groups(results, key)
    return ExperimentConfig.from_dict(grouped[0][2][0]["config"])


def stage_e_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    best = best_full_config(Path(args.work_dir), {"D"})
    routers = [
        RouterSpec("linear", {}),
        RouterSpec("mlp", {"dim_hidden": 64, "num_blocks_mlp": 2}),
        RouterSpec("set_attention", {"dim_model": 32, "num_heads": 1, "dim_hidden": 64}),
    ]
    configs: list[ExperimentConfig] = []
    names = [spec.name for spec in best.features]
    for router in routers:
        for seed in args.final_seeds:
            configs.append(
                base_experiment(
                    name=f"E-{router.name}-{best.normalization_recipe}-{loss_label(best.loss)}-seed{seed}",
                    stage="E",
                    features=make_features(
                        names,
                        best.normalization_recipe,
                        args.num_layers,
                        args.mask_density_window,
                        args.softmax_temperature,
                    ),
                    recipe=best.normalization_recipe,
                    loss=best.loss,
                    seed=seed,
                    h=args.h,
                    size_block=args.size_block,
                    max_epochs=args.full_epochs,
                    patience=args.full_patience,
                    router=router,
                    metadata={"architecture_ablation_from": best.name},
                )
            )
    return configs


def stage_f_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    results = load_completed_results(Path(args.work_dir), {"D", "E"})
    if not results:
        raise RuntimeError("No completed D/E results found; run stages D and E first")

    def key(cfg: ExperimentConfig) -> tuple[Any, ...]:
        return (
            feature_set_key(cfg),
            cfg.normalization_recipe,
            cfg.loss.name,
            str(cfg.loss.kwargs.get("mode", "")),
            cfg.router.name,
        )

    grouped = aggregate_result_groups(results, key)
    finalists = grouped[: args.finalist_count]
    configs: list[ExperimentConfig] = []
    for finalist_index, (_, _, members) in enumerate(finalists):
        base = ExperimentConfig.from_dict(members[0]["config"])
        for seed in args.final_seeds:
            cfg = clone_with_seed(base, seed, stage="F")
            cfg.name = safe_name(f"F-finalist{finalist_index}-{base.router.name}-seed{seed}")
            cfg.evaluate_test = True
            cfg.max_epochs = args.full_epochs
            cfg.patience = args.full_patience
            cfg.metadata = {**cfg.metadata, "finalist_index": finalist_index, "test_once": True}
            configs.append(cfg)
    return configs


def generate_stage_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    stage = args.stage.upper()
    generators = {
        "A": stage_a_configs,
        "B1": stage_b1_configs,
        "B2": stage_b2_configs,
        "C1": stage_c1_configs,
        "C2": stage_c2_configs,
        "C3": stage_c3_configs,
        "D": stage_d_configs,
        "E": stage_e_configs,
        "F": stage_f_configs,
    }
    if stage not in generators:
        raise ValueError(f"Unsupported stage {stage}; choose from {sorted(generators)}")
    configs = generators[stage](args)

    # Add a short hash so configurations with similar readable labels cannot collide.
    for config in configs:
        base_name = re.sub(r"-[0-9a-f]{10}$", "", config.name)
        config.name = f"{base_name}-{stable_hash(config.to_dict())}"
    return configs


# -----------------------------------------------------------------------------
# Plan execution and summaries
# -----------------------------------------------------------------------------


def write_plan(args: argparse.Namespace) -> Path:
    folder_data = Path(args.folder_data).resolve()
    work_dir = Path(args.work_dir).resolve()
    split_path = work_dir / "split.json"
    split = create_split_manifest(
        folder_data,
        split_path,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite_split,
    )
    configs = generate_stage_configs(args)
    plan = {
        "version": 1,
        "stage": args.stage.upper(),
        "folder_data": str(folder_data),
        "work_dir": str(work_dir),
        "router_module": args.router_module,
        "split_path": str(split_path),
        "split_fingerprint": split["fingerprint"],
        "created_at_unix": time.time(),
        "experiments": [config.to_dict() for config in configs],
    }
    path = work_dir / "plans" / f"stage_{args.stage.upper()}.json"
    json_dump(plan, path)
    print(f"Wrote {len(configs)} experiments to {path}")
    return path


def run_plan(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan).resolve()
    plan = json_load(plan_path)
    folder_data = Path(plan["folder_data"])
    work_dir = Path(plan["work_dir"])
    split = json_load(Path(plan["split_path"]))
    if split["fingerprint"] != plan["split_fingerprint"]:
        raise RuntimeError("Split manifest fingerprint does not match the plan")

    if str(folder_data.parent) not in sys.path:
        sys.path.insert(0, str(folder_data.parent))
    # Usually router_llada_v2.py is beside this runner or in the current project.
    runner_dir = str(Path(__file__).resolve().parent)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    router_module = importlib.import_module(plan.get("router_module", "router_llada_v2"))

    configs = [ExperimentConfig.from_dict(value) for value in plan["experiments"]]
    selected = [
        config
        for index, config in enumerate(configs)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(
        f"Running {len(selected)} of {len(configs)} experiments "
        f"(shard {args.shard_index}/{args.num_shards}) on {args.device}",
        flush=True,
    )

    failures: list[dict[str, str]] = []
    for index, config in enumerate(selected, start=1):
        output_dir = work_dir / "results" / config.stage / config.name
        result_path = output_dir / "result.json"
        if result_path.exists() and not args.overwrite:
            try:
                result = json_load(result_path)
                if result.get("status") == "complete":
                    print(f"[{index}/{len(selected)}] skip complete: {config.name}")
                    continue
            except (OSError, ValueError):
                pass

        print(f"[{index}/{len(selected)}] start: {config.name}", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_dump(config.to_dict(), output_dir / "config.json")
        try:
            trainer = AblationTrainer(
                router_module=router_module,
                folder_data=folder_data,
                config=config,
                split_manifest=split,
                device=args.device,
                output_dir=output_dir,
            )
            trainer.run()
        except Exception as error:  # preserve all completed work and continue the plan
            failure = {
                "status": "failed",
                "config": config.to_dict(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            json_dump(failure, output_dir / "failure.json")
            failures.append({"name": config.name, "error": repr(error)})
            print(f"FAILED {config.name}: {error!r}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        raise SystemExit(1)


def summarize_results(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    stages = set(stage.upper() for stage in args.stages) if args.stages else None
    results = load_completed_results(work_dir, stages)
    rows = []
    for result in results:
        cfg = ExperimentConfig.from_dict(result["config"])
        selection = result_score(result)
        rows.append(
            {
                "stage": cfg.stage,
                "name": cfg.name,
                "features": "+".join(spec.name for spec in cfg.features),
                "recipe": cfg.normalization_recipe,
                "loss": loss_label(cfg.loss),
                "router": cfg.router.name,
                "seed": cfg.seed,
                f"next_hit@{cfg.h}": selection[0],
                f"recall@{cfg.h}": selection[1],
                f"ndcg@{cfg.h}": selection[2],
                "best_epoch": result.get("best_epoch"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_path": result.get("_result_path"),
            }
        )
    rows.sort(
        key=lambda row: (
            row.get(f"next_hit@{args.h}", float("-inf")),
            row.get(f"recall@{args.h}", float("-inf")),
            row.get(f"ndcg@{args.h}", float("-inf")),
        ),
        reverse=True,
    )
    output = work_dir / "summary.json"
    json_dump(rows, output)

    print(f"Completed results: {len(rows)}")
    for row in rows[: args.top]:
        print(
            f"{row['stage']:>2}  hit={row[f'next_hit@{args.h}']:.5f} "
            f"recall={row[f'recall@{args.h}']:.5f} ndcg={row[f'ndcg@{args.h}']:.5f}  "
            f"{row['name']}"
        )
    print(f"Wrote {output}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def add_common_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", required=True, choices=["A", "B1", "B2", "C1", "C2", "C3", "D", "E", "F"])
    parser.add_argument("--folder-data", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--router-module", default="router_llada_v2")
    parser.add_argument("--h", type=int, default=5)
    parser.add_argument("--size-block", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--mask-density-window", type=int, default=3)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--split-seed", type=int, default=233)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--screen-epochs", type=int, default=10)
    parser.add_argument("--screen-patience", type=int, default=3)
    parser.add_argument("--full-epochs", type=int, default=30)
    parser.add_argument("--full-patience", type=int, default=5)
    parser.add_argument("--promote-count", type=int, default=12)
    parser.add_argument("--full-seeds", type=int, nargs="+", default=[233, 239, 251])
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[233, 239, 251, 257, 263])
    parser.add_argument("--finalist-count", type=int, default=3)
    parser.add_argument("--overwrite-split", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-plan", help="Generate a staged experiment plan")
    add_common_plan_args(make)
    make.set_defaults(func=write_plan)

    run = subparsers.add_parser("run-plan", help="Execute experiments from a plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--limit", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(func=run_plan)

    summary = subparsers.add_parser("summarize", help="Rank completed experiments")
    summary.add_argument("--work-dir", required=True)
    summary.add_argument("--stages", nargs="*")
    summary.add_argument("--h", type=int, default=5)
    summary.add_argument("--top", type=int, default=20)
    summary.set_defaults(func=summarize_results)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run-plan":
        if args.num_shards <= 0:
            parser.error("--num-shards must be positive")
        if not (0 <= args.shard_index < args.num_shards):
            parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    args.func(args)


if __name__ == "__main__":
    main()
