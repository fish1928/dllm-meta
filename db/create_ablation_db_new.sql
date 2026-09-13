-- Schema for the NEW ablation suite (new_a / new_b_arch / new_b_horizon).
-- Use a separate db file from the legacy one (default: ablation_new.db).
--
-- One row per experiment; metrics live long-form in experiment_metrics with
-- result_group = 'all' (n-weighted aggregate) or 'ds_<task>' (per dataset).

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    stage                 TEXT NOT NULL,
    name                  TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_message         TEXT,

    dataset_group         TEXT,
    datasets_json         TEXT,
    size_blocks_json      TEXT,

    normalization         TEXT,
    normalization_deployable INTEGER NOT NULL DEFAULT 0
                          CHECK (normalization_deployable IN (0, 1)),
    loss                  TEXT,
    loss_pos_weight       REAL,

    router                TEXT,
    router_trainable      INTEGER NOT NULL DEFAULT 1 CHECK (router_trainable IN (0, 1)),
    dim_hidden            INTEGER,
    num_blocks_mlp        INTEGER,
    dim_model             INTEGER,
    num_heads             INTEGER,
    dim_in                INTEGER,

    h                     INTEGER,
    max_conf_age          INTEGER,
    device                TEXT,
    num_layers            INTEGER,
    num_epochs            INTEGER,
    learning_rate         REAL,
    weight_decay          REAL,
    holdout               REAL,
    filter_result         TEXT,
    seed                  INTEGER,

    feature_count         INTEGER NOT NULL DEFAULT 0,
    feature_attn_last     INTEGER NOT NULL DEFAULT 0 CHECK (feature_attn_last IN (0, 1)),
    feature_attn_all      INTEGER NOT NULL DEFAULT 0 CHECK (feature_attn_all IN (0, 1)),
    feature_conf          INTEGER NOT NULL DEFAULT 0 CHECK (feature_conf IN (0, 1)),
    feature_conf_aged     INTEGER NOT NULL DEFAULT 0 CHECK (feature_conf_aged IN (0, 1)),
    feature_margin        INTEGER NOT NULL DEFAULT 0 CHECK (feature_margin IN (0, 1)),
    feature_pos_delta     INTEGER NOT NULL DEFAULT 0 CHECK (feature_pos_delta IN (0, 1)),
    feature_mask_density  INTEGER NOT NULL DEFAULT 0 CHECK (feature_mask_density IN (0, 1)),

    -- deployable = every piece survives the online path: normalization is
    -- implemented in router_deploy.build_online_x AND no fresh-conf leak
    deployable            INTEGER NOT NULL DEFAULT 0 CHECK (deployable IN (0, 1)),

    features_json         TEXT NOT NULL,
    config_json           TEXT NOT NULL,
    raw_record_json       TEXT NOT NULL,
    imported_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (stage, name)
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    experiment_id         INTEGER NOT NULL,
    result_group          TEXT NOT NULL,      -- 'all' or 'ds_<task>'
    metric_name           TEXT NOT NULL,      -- canonical (ndgc -> ndcg)
    metric_name_raw       TEXT NOT NULL,
    metric_value          REAL,
    sample_count          INTEGER,
    metric_raw_text       TEXT,
    metric_json           TEXT NOT NULL,

    PRIMARY KEY (experiment_id, result_group, metric_name),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_experiments_stage_status
    ON experiments(stage, status);
CREATE INDEX IF NOT EXISTS idx_experiments_design
    ON experiments(stage, dataset_group, normalization, loss, router);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup
    ON experiment_metrics(result_group, metric_name, metric_value);

-- wide summary: one row per experiment with the headline metrics pivoted out
DROP VIEW IF EXISTS summary_wide;
CREATE VIEW summary_wide AS
SELECT
    e.id, e.stage, e.name, e.status, e.dataset_group,
    e.features_json, e.normalization, e.normalization_deployable, e.loss,
    e.router, e.router_trainable, e.dim_hidden, e.num_blocks_mlp, e.h,
    e.feature_conf, e.feature_conf_aged, e.deployable, e.seed,

    MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@3'
        THEN m.metric_value END)  AS recall_at_3,
    MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@5'
        THEN m.metric_value END)  AS recall_at_5,
    MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@10'
        THEN m.metric_value END)  AS recall_at_10,
    MAX(CASE WHEN m.result_group='all' AND m.metric_name='recall@' || e.h
        THEN m.metric_value END)  AS recall_at_h_train,
    MAX(CASE WHEN m.result_group='all' AND m.metric_name='pr_auc@' || e.h
        THEN m.metric_value END)  AS pr_auc_at_h,
    MAX(CASE WHEN m.result_group='all' AND m.metric_name='ndcg@' || e.h
        THEN m.metric_value END)  AS ndcg_at_h,
    MAX(CASE WHEN m.result_group='all' AND m.metric_name='n_blocks'
        THEN m.metric_value END)  AS n_blocks

FROM experiments AS e
LEFT JOIN experiment_metrics AS m ON m.experiment_id = e.id
GROUP BY e.id;

-- deployable candidates only: recipe fully implementable online, no fresh conf
DROP VIEW IF EXISTS summary_deployable;
CREATE VIEW summary_deployable AS
SELECT * FROM summary_wide
WHERE deployable = 1 AND status = 'ok';

-- per-dataset recall@5, long form (pivot in the query as needed)
DROP VIEW IF EXISTS per_dataset_recall5;
CREATE VIEW per_dataset_recall5 AS
SELECT
    e.id, e.stage, e.name, e.dataset_group, e.features_json,
    e.normalization, e.loss, e.router, e.h, e.deployable,
    m.result_group AS ds, m.metric_value AS recall_at_5, m.sample_count AS n
FROM experiments AS e
JOIN experiment_metrics AS m ON m.experiment_id = e.id
WHERE m.metric_name = 'recall@5'
  AND m.result_group LIKE 'ds_%'
  AND e.status = 'ok';
