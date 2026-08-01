PRAGMA foreign_keys = ON;

-- One row per ablation experiment.
CREATE TABLE IF NOT EXISTS experiments (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    stage                 TEXT NOT NULL,
    name                  TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_message         TEXT,

    normalization         TEXT,
    loss                  TEXT,
    loss_pos_weight       REAL,
    router                TEXT,
    dim_hidden            INTEGER,
    num_blocks_mlp        INTEGER,

    h                     INTEGER,
    size_block            INTEGER,
    device                TEXT,
    num_layers            INTEGER,
    num_epochs            INTEGER,
    learning_rate         REAL,
    weight_decay          REAL,
    holdout               REAL,
    filter_result         TEXT,
    seed                  INTEGER,

    feature_count         INTEGER NOT NULL DEFAULT 0,

    -- SQLite represents Boolean values as INTEGER 0/1.
    feature_attn_last      INTEGER NOT NULL DEFAULT 0 CHECK (feature_attn_last IN (0, 1)),
    feature_attn_all       INTEGER NOT NULL DEFAULT 0 CHECK (feature_attn_all IN (0, 1)),
    feature_conf           INTEGER NOT NULL DEFAULT 0 CHECK (feature_conf IN (0, 1)),
    feature_margin         INTEGER NOT NULL DEFAULT 0 CHECK (feature_margin IN (0, 1)),
    feature_entropy        INTEGER NOT NULL DEFAULT 0 CHECK (feature_entropy IN (0, 1)),
    feature_pos_delta      INTEGER NOT NULL DEFAULT 0 CHECK (feature_pos_delta IN (0, 1)),
    feature_mask_density   INTEGER NOT NULL DEFAULT 0 CHECK (feature_mask_density IN (0, 1)),
    feature_x0_stability   INTEGER NOT NULL DEFAULT 0 CHECK (feature_x0_stability IN (0, 1)),
    feature_step_progress  INTEGER NOT NULL DEFAULT 0 CHECK (feature_step_progress IN (0, 1)),

    -- Preserve the original structures for auditing and future extensions.
    features_json         TEXT NOT NULL,
    config_json           TEXT NOT NULL,
    raw_record_json       TEXT NOT NULL,

    imported_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (stage, name)
);

-- One row per top-level metric in each result group, such as:
-- result_group='all', metric_name='recall@5', metric_value=0.77.
CREATE TABLE IF NOT EXISTS experiment_metrics (
    experiment_id         INTEGER NOT NULL,
    result_group          TEXT NOT NULL,
    metric_name           TEXT NOT NULL,
    metric_value          REAL,
    metric_json           TEXT NOT NULL,

    PRIMARY KEY (experiment_id, result_group, metric_name),
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_experiments_stage_status
    ON experiments(stage, status);

CREATE INDEX IF NOT EXISTS idx_experiments_design
    ON experiments(stage, normalization, loss, router);

CREATE INDEX IF NOT EXISTS idx_experiments_feature_flags
    ON experiments(
        feature_attn_last,
        feature_attn_all,
        feature_conf,
        feature_margin,
        feature_entropy,
        feature_pos_delta,
        feature_mask_density,
        feature_x0_stability,
        feature_step_progress
    );

CREATE INDEX IF NOT EXISTS idx_metrics_lookup
    ON experiment_metrics(result_group, metric_name, metric_value);

-- Convenience view for the current Stage B evaluation settings.
-- If H changes, the generic experiment_metrics table still contains every metric.
CREATE VIEW IF NOT EXISTS stage_b_summary AS
SELECT
    e.*,
    MAX(CASE
        WHEN m.result_group = 'all' AND m.metric_name = 'recall@3'
        THEN m.metric_value
    END) AS recall_at_3,
    MAX(CASE
        WHEN m.result_group = 'all' AND m.metric_name = 'recall@5'
        THEN m.metric_value
    END) AS recall_at_5,
    MAX(CASE
        WHEN m.result_group = 'all' AND m.metric_name = 'recall@10'
        THEN m.metric_value
    END) AS recall_at_10,
    MAX(CASE
        WHEN m.result_group = 'all' AND m.metric_name = 'pr_auc@5'
        THEN m.metric_value
    END) AS pr_auc_at_5,
    MAX(CASE
        WHEN m.result_group = 'all' AND m.metric_name = 'n_blocks'
        THEN m.metric_value
    END) AS n_blocks
FROM experiments AS e
LEFT JOIN experiment_metrics AS m
    ON m.experiment_id = e.id
WHERE e.stage = 'stage_b'
GROUP BY e.id;
