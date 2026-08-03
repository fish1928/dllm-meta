PRAGMA foreign_keys = ON;

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
    router_runtime        TEXT,
    dim_hidden            INTEGER,
    num_blocks_mlp        INTEGER,
    dim_in                INTEGER,
    router_features_json  TEXT,

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

    feature_attn_last      INTEGER NOT NULL DEFAULT 0 CHECK (feature_attn_last IN (0, 1)),
    feature_attn_all       INTEGER NOT NULL DEFAULT 0 CHECK (feature_attn_all IN (0, 1)),
    feature_conf           INTEGER NOT NULL DEFAULT 0 CHECK (feature_conf IN (0, 1)),
    feature_margin         INTEGER NOT NULL DEFAULT 0 CHECK (feature_margin IN (0, 1)),
    feature_entropy        INTEGER NOT NULL DEFAULT 0 CHECK (feature_entropy IN (0, 1)),
    feature_pos_delta      INTEGER NOT NULL DEFAULT 0 CHECK (feature_pos_delta IN (0, 1)),
    feature_mask_density   INTEGER NOT NULL DEFAULT 0 CHECK (feature_mask_density IN (0, 1)),
    feature_x0_stability   INTEGER NOT NULL DEFAULT 0 CHECK (feature_x0_stability IN (0, 1)),
    feature_step_progress  INTEGER NOT NULL DEFAULT 0 CHECK (feature_step_progress IN (0, 1)),

    features_json         TEXT NOT NULL,
    config_json           TEXT NOT NULL,
    raw_record_json       TEXT NOT NULL,

    imported_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (stage, name)
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    experiment_id         INTEGER NOT NULL,
    result_group          TEXT NOT NULL,

    metric_name           TEXT NOT NULL,
    metric_name_raw       TEXT NOT NULL,

    metric_value          REAL,
    sample_count          INTEGER,
    metric_raw_text       TEXT,
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

CREATE INDEX IF NOT EXISTS idx_metrics_lookup
    ON experiment_metrics(result_group, metric_name, metric_value);

DROP VIEW IF EXISTS stage_b_summary;

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
    ON m.experiment_id = e.id
WHERE e.stage = 'stage_b'
GROUP BY e.id;
