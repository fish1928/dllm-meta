# Router Ablation Runner

This runner imports your existing `router_llada_v2.py` module and leaves the router framework unchanged.

## Files

- `router_ablation_runner.py`: plan generation, training, evaluation, promotion, and summaries.
- `run_ablation_stage_2gpu.sh`: optional two-GPU launcher for one stage.

Place both files in the same Python project as `router_llada_v2.py`, `attn_order_eval.py`, and `tools_debug.py`.

## Metrics

Model selection uses validation rows with more than `h` candidates remaining, avoiding the trivially easy late steps. The lexicographic selection tuple is:

1. `next_hit@h`
2. `recall@h`
3. `ndcg@h`

Reports also include MRR, average precision, trapezoidal PR-AUC, early/middle/late strata, pass/fail categories, and block-index breakdowns.

## Split

The first `make-plan` command creates `WORK_DIR/split.json` using a 70/15/15 sample-level split stratified by `generated.json.result`. Every later stage reuses this manifest. The runner refuses to silently add newly discovered samples to an existing split.

## Stages

- `A`: sanity checks and non-trainable baselines.
- `B1`: 70 one-seed screening experiments: 2 feature anchors × 7 normalization recipes × 5 loss variants.
- `B2`: promote the best 12 B1 configurations and run three full-training seeds.
- `C1`: single-feature and attention-representation experiments.
- `C2`: add-one and semantic feature-group experiments.
- `C3`: leave-one-out experiments based on the strongest C2 set.
- `D`: compact 3 × 2 × 2 feature/loss/normalization factorial confirmation.
- `E`: linear versus pointwise MLP versus set-attention router.
- `F`: final held-out test evaluation of the strongest configurations.

Stages B2–F read completed results from earlier stages and construct the next plan automatically.

## Single-GPU example

```bash
DATA_DIR=/path/to/collected/router_stats
WORK_DIR=/path/to/router_ablation_runs
RUNNER=router_ablation_runner.py

python "$RUNNER" make-plan \
  --stage A \
  --folder-data "$DATA_DIR" \
  --work-dir "$WORK_DIR" \
  --router-module router_llada_v2 \
  --h 5 \
  --size-block 64 \
  --num-layers 32

python "$RUNNER" run-plan \
  --plan "$WORK_DIR/plans/stage_A.json" \
  --device cuda:0

python "$RUNNER" summarize \
  --work-dir "$WORK_DIR" \
  --stages A \
  --h 5
```

Repeat `make-plan` and `run-plan` for `B1`, `B2`, `C1`, `C2`, `C3`, `D`, `E`, and `F` in order.

## Two-GPU example

```bash
bash run_ablation_stage_2gpu.sh B1 \
  /path/to/collected/router_stats \
  /path/to/router_ablation_runs
```

The launcher creates the plan once, then executes even-indexed experiments on `cuda:0` and odd-indexed experiments on `cuda:1`.

## Important options

```text
--screen-epochs 10
--screen-patience 3
--full-epochs 30
--full-patience 5
--promote-count 12
--full-seeds 233 239 251
--final-seeds 233 239 251 257 263
--softmax-temperature 1.0
--mask-density-window 3
```

Use `--overwrite` with `run-plan` to rerun completed experiments. Use `--overwrite-split` only when intentionally rebuilding all train/validation/test assignments.

## Outputs

Each experiment creates:

```text
WORK_DIR/results/<stage>/<experiment>/
  config.json
  best.pt
  result.json
```

`result.json` includes the full configuration, epoch history, validation metrics, optional test metrics, per-block records, parameter counts, runtime, and the fitted balanced-BCE weight when applicable.

## Scope

Stages A–F cover offline router ablation. Actual sparse-decoding quality and latency require integration with your LLaDA decoding loop, which is outside the supplied router framework.
