#!/usr/bin/env python3
"""Import the NEW ablation report (ablation_test_report_new.json, stages
new_a / new_b_arch / new_b_horizon) into sqlite.

Derived flags baked into the experiments table so the candidate queries stay
one-liners:
  normalization_deployable  normalization in (rank, softmax_attn) -- the only
                            recipes router_deploy.build_online_x implements
  feature_conf / feature_conf_aged  fresh vs aged confidence (fresh conf is
                            the KNOWN LEAK: strong offline, unrecoverable
                            online)
  router_trainable          0 for the mockup_* reference rows
  deployable                normalization_deployable AND no fresh conf AND a
                            trainable router: rows eligible to become E2E
                            candidates

Usage:
  python import_ablation_new_to_sqlite.py \
      --json ../ablation_test_report_new.json --db ablation_new.db
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

METRIC_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?:\(\s*n\s*=\s*(\d+)\s*\))?\s*$"
)

NORMALIZATIONS_DEPLOYABLE = {"rank", "softmax_attn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="ablation_test_report_new.json")
    parser.add_argument("--db", default="ablation_new.db")
    parser.add_argument("--schema", default="create_ablation_db_new.sql")
    parser.add_argument("--stage", action="append", dest="stages")
    return parser.parse_args()


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_metric_name(name: str) -> str:
    if name.startswith("ndgc@"):    # typo in attn_order_eval's metric key
        return "ndcg@" + name.split("@", 1)[1]
    return name


def parse_metric(value: Any) -> tuple[float | None, int | None, str | None]:
    if isinstance(value, bool):
        return None, None, None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return (numeric if math.isfinite(numeric) else None), None, None
    if isinstance(value, str):
        match = METRIC_PATTERN.fullmatch(value)
        if match is None:
            return None, None, value
        numeric = float(match.group(1))
        count = int(match.group(2)) if match.group(2) else None
        return numeric, count, value
    return None, None, None


def upsert_experiment(connection: sqlite3.Connection, stage: str, record: dict[str, Any]) -> int:
    config = record.get("config") or {}
    features = list(config.get("features") or [])
    feature_set = set(features)
    router_kwargs = config.get("router_kwargs") or {}
    router = config.get("router") or ""
    normalization = config.get("normalization")

    normalization_deployable = int(normalization in NORMALIZATIONS_DEPLOYABLE)
    router_trainable = int(not router.startswith("mockup"))
    deployable = int(
        normalization_deployable
        and "conf" not in feature_set    # fresh conf: the known offline-only leak
        and router_trainable
    )

    error = record.get("error")
    values = {
        "stage": stage,
        "name": str(record.get("name", "")),
        "status": "error" if error else "ok",
        "error_message": error,
        "dataset_group": config.get("dataset_group"),
        "datasets_json": to_json(config.get("datasets") or []),
        "size_blocks_json": to_json(config.get("size_blocks") or []),
        "normalization": normalization,
        "normalization_deployable": normalization_deployable,
        "loss": config.get("loss"),
        "loss_pos_weight": config.get("loss_pos_weight"),
        "router": router,
        "router_trainable": router_trainable,
        "dim_hidden": router_kwargs.get("dim_hidden"),
        "num_blocks_mlp": router_kwargs.get("num_blocks_mlp"),
        "dim_model": router_kwargs.get("dim_model"),
        "num_heads": router_kwargs.get("num_heads"),
        "dim_in": None,
        "h": config.get("h"),
        "max_conf_age": config.get("max_conf_age"),
        "device": config.get("device"),
        "num_layers": config.get("num_layers"),
        "num_epochs": config.get("num_epochs"),
        "learning_rate": config.get("lr"),
        "weight_decay": config.get("weight_decay"),
        "holdout": config.get("holdout"),
        "filter_result": config.get("filter_result"),
        "seed": config.get("seed"),
        "feature_count": len(features),
        "feature_attn_last": int("attn_last" in feature_set),
        "feature_attn_all": int("attn_all" in feature_set),
        "feature_conf": int("conf" in feature_set),
        "feature_conf_aged": int("conf_aged" in feature_set),
        "feature_margin": int("margin" in feature_set),
        "feature_pos_delta": int("pos_delta" in feature_set),
        "feature_mask_density": int("mask_density" in feature_set),
        "deployable": deployable,
        "features_json": to_json(features),
        "config_json": to_json(config),
        "raw_record_json": to_json(record),
    }

    # dim_in from the router metadata stored under metrics.all.router
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        runtime = metrics.get("all", {})
        if isinstance(runtime, dict) and isinstance(runtime.get("router"), dict):
            values["dim_in"] = runtime["router"].get("dim_in")

    columns = list(values)
    placeholders = ", ".join(f":{column}" for column in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in columns if column not in {"stage", "name"}
    )
    connection.execute(
        f"""
        INSERT INTO experiments ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(stage, name) DO UPDATE SET {updates}, imported_at=CURRENT_TIMESTAMP
        """,
        values,
    )

    row = connection.execute(
        "SELECT id FROM experiments WHERE stage=? AND name=?",
        (stage, values["name"]),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Could not retrieve {stage}/{values['name']}")
    return int(row[0])


def replace_metrics(connection: sqlite3.Connection, experiment_id: int, metrics: Any) -> int:
    connection.execute("DELETE FROM experiment_metrics WHERE experiment_id=?", (experiment_id,))
    if not isinstance(metrics, dict):
        return 0

    inserted = 0
    for result_group, group in metrics.items():
        if not isinstance(group, dict):
            continue
        for raw_name, payload in group.items():
            if raw_name == "router":    # metadata, not a metric
                continue
            metric_value, sample_count, raw_text = parse_metric(payload)
            if isinstance(payload, dict) and metric_value is None:
                continue
            connection.execute(
                """
                INSERT INTO experiment_metrics (
                    experiment_id, result_group, metric_name, metric_name_raw,
                    metric_value, sample_count, metric_raw_text, metric_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    str(result_group),
                    canonical_metric_name(str(raw_name)),
                    str(raw_name),
                    metric_value,
                    sample_count,
                    raw_text,
                    to_json(payload),
                ),
            )
            inserted += 1
    return inserted


def main() -> None:
    args = parse_args()

    with Path(args.json).open("r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError("The JSON root must be an object keyed by stage")

    selected_stages = set(args.stages) if args.stages else None

    connection = sqlite3.connect(Path(args.db))
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(Path(args.schema).read_text(encoding="utf-8"))

        experiment_count = 0
        metric_count = 0
        with connection:
            for stage, records in report.items():
                if selected_stages is not None and stage not in selected_stages:
                    continue
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    experiment_id = upsert_experiment(connection, stage, record)
                    metric_count += replace_metrics(connection, experiment_id, record.get("metrics"))
                    experiment_count += 1

        print(f"Imported {experiment_count} experiments and {metric_count} metric rows into {args.db}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
