#!/usr/bin/env python3
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="ablation_test_report.json")
    parser.add_argument("--db", default="ablation_test.db")
    parser.add_argument("--schema", default="create_ablation_db.sql")
    parser.add_argument("--stage", action="append", dest="stages")
    return parser.parse_args()


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_metric_name(name: str) -> str:
    # Correct the typo present in the current report.
    if name.startswith("ndgc@"):
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

    if isinstance(value, dict):
        numeric = None
        count = None

        for key in ("mean", "avg", "average", "value", "score", "median"):
            if key in value:
                numeric, _, _ = parse_metric(value[key])
                if numeric is not None:
                    break

        for key in ("n", "count", "sample_count", "num_samples"):
            if key in value:
                raw_count = value[key]
                if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                    count = raw_count
                elif isinstance(raw_count, str) and raw_count.isdigit():
                    count = int(raw_count)
                break

        return numeric, count, None

    return None, None, None


def ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    existing = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column not in existing:
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
        )


def apply_schema_and_migrate(
    connection: sqlite3.Connection,
    schema_path: Path,
) -> None:
    # First create any missing tables.
    connection.executescript(schema_path.read_text(encoding="utf-8"))

    # Then upgrade databases created by the old schema.
    ensure_column(connection, "experiments", "router_runtime", "TEXT")
    ensure_column(connection, "experiments", "dim_in", "INTEGER")
    ensure_column(connection, "experiments", "router_features_json", "TEXT")

    ensure_column(
        connection,
        "experiment_metrics",
        "metric_name_raw",
        "TEXT NOT NULL DEFAULT ''",
    )
    ensure_column(
        connection,
        "experiment_metrics",
        "sample_count",
        "INTEGER",
    )
    ensure_column(
        connection,
        "experiment_metrics",
        "metric_raw_text",
        "TEXT",
    )

    # Recreate the view after migration.
    connection.execute("DROP VIEW IF EXISTS stage_b_summary")
    connection.executescript(
        """
        CREATE VIEW stage_b_summary AS
        SELECT
            e.*,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@3'
                THEN m.metric_value END) AS recall_at_3,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@3'
                THEN m.sample_count END) AS recall_at_3_n,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@5'
                THEN m.metric_value END) AS recall_at_5,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@5'
                THEN m.sample_count END) AS recall_at_5_n,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@10'
                THEN m.metric_value END) AS recall_at_10,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@10'
                THEN m.sample_count END) AS recall_at_10_n,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='pr_auc@5'
                THEN m.metric_value END) AS pr_auc_at_5,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='pr_auc@5'
                THEN m.sample_count END) AS pr_auc_at_5_n,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='ndcg@5'
                THEN m.metric_value END) AS ndcg_at_5,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='ndcg@5'
                THEN m.sample_count END) AS ndcg_at_5_n,
            MAX(CASE WHEN m.result_group='all' AND m.metric_name='n_blocks'
                THEN m.metric_value END) AS n_blocks
        FROM experiments AS e
        LEFT JOIN experiment_metrics AS m
            ON m.experiment_id=e.id
        WHERE e.stage='stage_b'
        GROUP BY e.id;
        """
    )


def router_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    all_metrics = metrics.get("all")
    if not isinstance(all_metrics, dict):
        return {}
    router = all_metrics.get("router")
    return router if isinstance(router, dict) else {}


def upsert_experiment(
    connection: sqlite3.Connection,
    stage: str,
    record: dict[str, Any],
) -> int:
    config = record.get("config") or {}
    features = list(config.get("features") or [])
    feature_set = set(features)
    router_kwargs = config.get("router_kwargs") or {}
    runtime = router_metadata(record)

    error = record.get("error")
    values = {
        "stage": stage,
        "name": str(record.get("name", "")),
        "status": "error" if error else "ok",
        "error_message": error,
        "normalization": config.get("normalization"),
        "loss": config.get("loss"),
        "loss_pos_weight": config.get("loss_pos_weight"),
        "router": config.get("router"),
        "router_runtime": runtime.get("router"),
        "dim_hidden": router_kwargs.get("dim_hidden"),
        "num_blocks_mlp": router_kwargs.get("num_blocks_mlp"),
        "dim_in": runtime.get("dim_in"),
        "router_features_json": (
            to_json(runtime.get("features"))
            if isinstance(runtime.get("features"), list)
            else None
        ),
        "h": config.get("h"),
        "size_block": config.get("size_block"),
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
        "feature_margin": int("margin" in feature_set),
        "feature_entropy": int("entropy" in feature_set),
        "feature_pos_delta": int("pos_delta" in feature_set),
        "feature_mask_density": int("mask_density" in feature_set),
        "feature_x0_stability": int("x0_stability" in feature_set),
        "feature_step_progress": int("step_progress" in feature_set),
        "features_json": to_json(features),
        "config_json": to_json(config),
        "raw_record_json": to_json(record),
    }

    columns = list(values)
    placeholders = ", ".join(f":{column}" for column in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in columns
        if column not in {"stage", "name"}
    )

    connection.execute(
        f"""
        INSERT INTO experiments ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(stage, name) DO UPDATE SET
            {updates},
            imported_at=CURRENT_TIMESTAMP
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


def replace_metrics(
    connection: sqlite3.Connection,
    experiment_id: int,
    metrics: Any,
) -> int:
    connection.execute(
        "DELETE FROM experiment_metrics WHERE experiment_id=?",
        (experiment_id,),
    )

    if not isinstance(metrics, dict):
        return 0

    inserted = 0

    for result_group, group in metrics.items():
        if not isinstance(group, dict):
            continue

        for raw_name, payload in group.items():
            # Router is metadata, not a metric.
            if raw_name == "router":
                continue

            metric_value, sample_count, raw_text = parse_metric(payload)

            # Skip dictionary metadata with no recognized scalar value.
            if isinstance(payload, dict) and metric_value is None:
                continue

            metric_name = canonical_metric_name(str(raw_name))

            connection.execute(
                """
                INSERT INTO experiment_metrics (
                    experiment_id,
                    result_group,
                    metric_name,
                    metric_name_raw,
                    metric_value,
                    sample_count,
                    metric_raw_text,
                    metric_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    str(result_group),
                    metric_name,
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

    json_path = Path(args.json)
    db_path = Path(args.db)
    schema_path = Path(args.schema)

    with json_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError("The JSON root must be an object keyed by stage")

    selected_stages = set(args.stages) if args.stages else None

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        apply_schema_and_migrate(connection, schema_path)

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

                    experiment_id = upsert_experiment(
                        connection,
                        stage,
                        record,
                    )
                    metric_count += replace_metrics(
                        connection,
                        experiment_id,
                        record.get("metrics"),
                    )
                    experiment_count += 1

        print(
            f"Imported {experiment_count} experiments "
            f"and {metric_count} metric rows into {db_path}"
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
