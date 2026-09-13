-- Prepared queries over ablation_new.db (schema: create_ablation_db_new.sql).
-- Run interactively:  sqlite3 ablation_new.db
-- or one-shot:        sqlite3 -header -column ablation_new.db < queries_best_candidates.sql
--
-- Headline metric: recall_at_5 on result_group 'all' (n-weighted across the
-- group's datasets). `summary_deployable` already excludes fresh-conf rows,
-- non-deployable normalizations, and mockups -- pick E2E candidates there.
-- `summary_wide` keeps everything for diagnosis (leak sizing, mockup floors).

.headers on
.mode column

-- ============================================================
-- Q1. Top-10 deployable candidates PER dataset group (new_a)
-- ============================================================
SELECT dataset_group, name, features_json, normalization, loss,
       ROUND(recall_at_5, 3) AS r5, ROUND(recall_at_10, 3) AS r10,
       ROUND(ndcg_at_h, 3) AS ndcg
FROM (
    SELECT *, ROW_NUMBER() OVER (
                PARTITION BY dataset_group ORDER BY recall_at_5 DESC) AS rank_in_group
    FROM summary_deployable
    WHERE stage = 'new_a'
)
WHERE rank_in_group <= 10
ORDER BY dataset_group, recall_at_5 DESC;

-- ============================================================
-- Q2. A COMMON winner: recipes ranked by MINIMUM recall@5 across the three
--     dataset groups (a recipe must exist in all groups to qualify)
-- ============================================================
SELECT features_json, normalization, loss,
       COUNT(*)                       AS n_groups,
       ROUND(MIN(recall_at_5), 3)     AS worst_group_r5,
       ROUND(AVG(recall_at_5), 3)     AS mean_r5
FROM summary_deployable
WHERE stage = 'new_a'
GROUP BY features_json, normalization, loss
HAVING n_groups = 3
ORDER BY worst_group_r5 DESC
LIMIT 15;

-- ============================================================
-- Q3. THE LEAK, SIZED: fresh conf vs aged conf vs no conf, holding the rest
--     of the recipe fixed. Large fresh-aged gap = offline metric you cannot
--     have online; pick by the conf_aged / none columns.
-- ============================================================
SELECT w.dataset_group, w.normalization, w.loss,
       ROUND(MAX(CASE WHEN w.feature_conf = 1     THEN w.recall_at_5 END), 3) AS r5_conf_fresh,
       ROUND(MAX(CASE WHEN w.feature_conf_aged = 1 THEN w.recall_at_5 END), 3) AS r5_conf_aged,
       ROUND(MAX(CASE WHEN w.feature_conf = 0 AND w.feature_conf_aged = 0
                      THEN w.recall_at_5 END), 3)                              AS r5_no_conf
FROM summary_wide AS w
WHERE w.stage = 'new_a' AND w.status = 'ok' AND w.router_trainable = 1
  AND w.name LIKE 'attn_last_geo%'          -- the paired anchors
GROUP BY w.dataset_group, w.normalization, w.loss
ORDER BY w.dataset_group, r5_conf_aged DESC;

-- ============================================================
-- Q4. Normalization marginal effect (mean over anchors x losses), all rows
--     incl. exploratory norms; deployable column flags implementability
-- ============================================================
SELECT dataset_group, normalization, normalization_deployable,
       COUNT(*) AS n, ROUND(AVG(recall_at_5), 3) AS mean_r5,
       ROUND(MAX(recall_at_5), 3) AS best_r5
FROM summary_wide
WHERE stage = 'new_a' AND status = 'ok' AND router_trainable = 1
GROUP BY dataset_group, normalization
ORDER BY dataset_group, mean_r5 DESC;

-- ============================================================
-- Q5. Loss marginal effect
-- ============================================================
SELECT dataset_group, loss,
       COUNT(*) AS n, ROUND(AVG(recall_at_5), 3) AS mean_r5,
       ROUND(MAX(recall_at_5), 3) AS best_r5
FROM summary_wide
WHERE stage = 'new_a' AND status = 'ok' AND router_trainable = 1
GROUP BY dataset_group, loss
ORDER BY dataset_group, mean_r5 DESC;

-- ============================================================
-- Q6. Feature-set marginal effect (best over norms x losses per anchor)
-- ============================================================
SELECT dataset_group, features_json, feature_conf, feature_conf_aged,
       COUNT(*) AS n, ROUND(MAX(recall_at_5), 3) AS best_r5,
       ROUND(AVG(recall_at_5), 3) AS mean_r5
FROM summary_wide
WHERE stage = 'new_a' AND status = 'ok' AND router_trainable = 1
GROUP BY dataset_group, features_json
ORDER BY dataset_group, best_r5 DESC;

-- ============================================================
-- Q7. Does the winner generalize INSIDE a mixed group? per-dataset recall@5
--     of the top deployable mix5 recipes (watch for one task carrying the mean)
-- ============================================================
SELECT p.name, p.ds, ROUND(p.recall_at_5, 3) AS r5, p.n
FROM per_dataset_recall5 AS p
WHERE p.stage = 'new_a' AND p.dataset_group = 'mix5' AND p.deployable = 1
  AND p.name IN (
      SELECT name FROM summary_deployable
      WHERE stage = 'new_a' AND dataset_group = 'mix5'
      ORDER BY recall_at_5 DESC LIMIT 5)
ORDER BY p.name, p.ds;

-- ============================================================
-- Q8. Architecture ranking (new_b_arch), with the mockup floors for scale
-- ============================================================
SELECT dataset_group, router, dim_hidden, num_blocks_mlp, router_trainable,
       ROUND(recall_at_5, 3) AS r5, ROUND(ndcg_at_h, 3) AS ndcg
FROM summary_wide
WHERE stage = 'new_b_arch' AND status = 'ok'
ORDER BY dataset_group, recall_at_5 DESC;

-- ============================================================
-- Q9. Horizon curve (new_b_horizon): recall@5 stays the comparable basis;
--     recall_at_h_train shows what each h was optimized for
-- ============================================================
SELECT dataset_group, h,
       ROUND(recall_at_5, 3) AS r5_common_basis,
       ROUND(recall_at_h_train, 3) AS r_at_trained_h,
       ROUND(pr_auc_at_h, 3) AS pr_auc
FROM summary_wide
WHERE stage = 'new_b_horizon' AND status = 'ok'
ORDER BY dataset_group, h;

-- ============================================================
-- Q10. Failed experiments to retry (RERUN=1 after fixing)
-- ============================================================
SELECT stage, name, SUBSTR(error_message, 1, 120) AS error_head
FROM experiments
WHERE status = 'error'
ORDER BY stage, name;
