#!/bin/bash
#################################################
# Run the in-framework d2Cache sweep for all four threads one by one
# (each = 6 canonical benchmarks via run_bench_d2cache.bash).
# Resume-safe: runs with an existing runner.json are skipped, so rerunning
# after a crash or kill continues where it stopped.
#
#   LIMIT=450 DEVICE=cuda:0 nohup bash run_bench_d2cache_all.bash > d2c_all.log 2>&1 &
#   THREADS="dream_base dream_instruct" DEVICE=cuda:1 bash run_bench_d2cache_all.bash
#   CONF_MODE=frozen bash run_bench_d2cache_all.bash        # released-code variant
#
# All run_bench_d2cache.bash env knobs pass through: DEVICE, LIMIT,
# FILTER_TASK, CONF_MODE, D2C_K, D2C_SIGMA, D2C_ROLLOUT_P, FOLDER_RESULTS.
#################################################

set -u
cd "$(dirname "$0")"

THREADS=${THREADS:-"llada_base llada_instruct dream_base dream_instruct"}

for thread in $THREADS; do
    echo "==================== d2cache: $thread ===================="
    THREAD=$thread bash run_bench_d2cache.bash || echo "[warn] thread $thread failed, continuing"
done
