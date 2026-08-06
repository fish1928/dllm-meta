import json
import os
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from router_llada import (
    FeatureWrapperBase,
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
    Feature_znormed_global,
    Feature_znormed_row,
    FactoryLoss,
    FactoryRouter,
    RouterTrainer,
    build_geometry,
    load_stat,
)


REPORT_PATH = "ablation_test_report.json"


class Feature_negated(FeatureWrapperBase):
    """Use -x as the feature value. Useful for the raw entropy baseline."""

    def load_block(self, id_sample, pos_base, size_block):
        return -self.feature_inner.load_block(id_sample, pos_base, size_block)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_report(report_path: str = REPORT_PATH) -> Dict[str, Any]:
    if not os.path.exists(report_path):
        return {}
    with open(report_path, "r", encoding="utf-8") as file:
        return json.load(file)


def reset_stage(stage: str, report_path: str = REPORT_PATH) -> None:
    """Remove old records for one stage while preserving all other stages."""
    report = _load_report(report_path)
    report[stage] = []
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def save_result(
    stage: str,
    name: str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    report_path: str = REPORT_PATH,
) -> None:
    report = _load_report(report_path)
    records = report.setdefault(stage, [])

    record = {
        "name": name,
        "config": _jsonable(config),
        "metrics": _jsonable(metrics),
        "error": error,
    }

    # Re-running the same experiment replaces its previous record.
    for index, old in enumerate(records):
        if old.get("name") == name:
            records[index] = record
            break
    else:
        records.append(record)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def make_feature(name: str, folder_data: str, num_layers: int = 32):
    if name == "attn_last":
        return Feature_attn_last(folder_data)
    if name == "attn_all":
        return Feature_attn_all(folder_data, num_layers=num_layers)
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
    raise ValueError(f"Unknown feature: {name}")


def normalize_feature(name: str, feature, normalization: str):
    """
    Apply the normalization recipe used in the ablation plan.

    Geometric, binary, and progress features stay raw because their absolute
    values carry meaning or they are already bounded.
    """
    keep_raw = {"pos_delta", "mask_density", "x0_stability", "step_progress"}
    if name in keep_raw or normalization == "raw":
        return feature

    if normalization == "rank":
        return Feature_rank_normed(feature)
    if normalization == "znorm_row":
        return Feature_znormed_row(feature)
    if normalization == "minmax_row":
        return Feature_minmax_row(feature)
    if normalization == "znorm_global":
        return Feature_znormed_global(feature)

    if normalization == "log_znorm":
        if name in {"attn_last", "attn_all"}:
            return Feature_znormed_row(Feature_log_scaled(feature))
        return Feature_znormed_row(feature)

    if normalization == "softmax_attn":
        if name in {"attn_last", "attn_all"}:
            return Feature_softmax_row(feature, temperature=1.0)
        return Feature_znormed_row(feature)

    raise ValueError(f"Unknown normalization: {normalization}")


def build_features(
    feature_names: Sequence[str],
    folder_data: str,
    normalization: str,
    num_layers: int = 32,
):
    features = []
    for name in feature_names:
        feature = make_feature(name, folder_data, num_layers=num_layers)
        feature = normalize_feature(name, feature, normalization)
        features.append(feature)
    return features


def build_loss(loss_name: str, pos_weight: Optional[float] = None):
    if loss_name == "bce_balanced":
        if pos_weight is None:
            raise ValueError("bce_balanced requires pos_weight")
        return FactoryLoss.create("bce_within_h", pos_weight=pos_weight)
    return FactoryLoss.create(loss_name)


def estimate_balanced_pos_weight(
    folder_data: str,
    h: int,
    size_block: int,
    holdout: float = 0.2,
    filter_result: str = "all",
    seed: int = 233,
) -> float:
    """Count positive and negative candidate labels on the training split."""
    splitter = RouterTrainer(
        folder_data,
        h=h,
        size_block=size_block,
        device="cpu",
        holdout=holdout,
        filter_result=filter_result,
        seed=seed,
    )

    positives = 0
    negatives = 0
    for id_sample, pos_base in splitter._list_blocks(splitter.ids_train):
        folder_base = os.path.join(folder_data, str(id_sample))
        unmask = load_stat(folder_base, "unmask", pos_base, size_block)
        order = unmask.squeeze(-1).long() - pos_base
        gap, cand_mask = build_geometry(order, size_block)
        positive_mask = cand_mask & (gap >= 1) & (gap <= h)
        positives += int(positive_mask.sum())
        negatives += int((cand_mask & ~positive_mask).sum())

    if positives == 0:
        raise RuntimeError("No positive labels were found while estimating pos_weight")
    return negatives / positives


def run_experiment(
    *,
    stage: str,
    name: str,
    folder_data: str,
    feature_names: Sequence[str],
    normalization: str,
    loss_name: str,
    router_name: str = "mlp",
    router_kwargs: Optional[Dict[str, Any]] = None,
    loss_pos_weight: Optional[float] = None,
    h: int = 5,
    size_block: int = 128,
    device: str = "cuda:0",
    num_layers: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    holdout: float = 0.2,
    filter_result: str = "all",
    seed: int = 233,
    evaluate_result_groups: bool = False,
    checkpoint_path: Optional[str] = None,
    report_path: str = REPORT_PATH,
):
    router_kwargs = router_kwargs or {}
    config = {
        "folder_data": folder_data,
        "features": list(feature_names),
        "normalization": normalization,
        "loss": loss_name,
        "loss_pos_weight": loss_pos_weight,
        "router": router_name,
        "router_kwargs": router_kwargs,
        "h": h,
        "size_block": size_block,
        "device": device,
        "num_layers": num_layers,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "holdout": holdout,
        "filter_result": filter_result,
        "seed": seed,
    }

    print(f"\n[{stage}] {name}")
    try:
        # Set the seed before constructing the router so initialization is repeatable.
        torch.manual_seed(seed)

        features = build_features(
            feature_names,
            folder_data,
            normalization,
            num_layers=num_layers,
        )
        router = FactoryRouter.create(router_name, **router_kwargs).register_features(*features)
        loss = build_loss(loss_name, pos_weight=loss_pos_weight)

        trainer = RouterTrainer(
            folder_data,
            h=h,
            size_block=size_block,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            holdout=holdout,
            filter_result=filter_result,
            seed=seed,
        )
        trainer.register_router(router).register_loss(loss)
        trainer.train(num_epochs=num_epochs)

        metrics: Dict[str, Any] = {"all": trainer.evaluate()}

        if evaluate_result_groups:
            ids_pass = [i for i in trainer.ids_eval if trainer._read_result(i) == "pass"]
            ids_fail = [i for i in trainer.ids_eval if trainer._read_result(i) == "fail"]
            if ids_pass:
                metrics["pass"] = trainer.evaluate(ids_sample=ids_pass)
            if ids_fail:
                metrics["fail"] = trainer.evaluate(ids_sample=ids_fail)

        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            trainer.router.save_checkpoint(checkpoint_path)
            metrics["checkpoint"] = checkpoint_path

        save_result(stage, name, config, metrics=metrics, report_path=report_path)
        print(json.dumps(_jsonable(metrics), indent=2))
        return trainer, metrics

    except Exception:
        error = traceback.format_exc()
        save_result(stage, name, config, error=error, report_path=report_path)
        print(error)
        return None, None
