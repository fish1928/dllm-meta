#!/bin/bash
#################################################
# Full-denoising baseline on the extended benchmark set (run_llada_semi).
#
# Same tasks, generation lengths, few-shot counts and LIMIT as
# run_bench_router.bash, so accuracy and wall-clock are directly comparable.
# No cache, no router: every position is queried at every step.
#
# Per run outputs (under FOLDER_RESULTS):
#   baseline__<task>/                 lm_eval metrics (--output_path)
#   baseline__<task>__runner.json     per-sample duration / has_done
#   logs/baseline__<task>.log         full stdout+stderr
#
# Usage:
#   DEVICE=cuda:0 LIMIT=100 bash run_bench_baseline.bash
#   FILTER_TASK=bbh bash run_bench_baseline.bash
#
# Runtime warning: full denoising queries every position each step, so this is
# several times slower per sample than the router runs -- that gap is the
# measurement. Use the SAME LIMIT as the router sweep for a valid comparison.
#################################################

set -u

DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-100}

FOLDER_RESULTS=${FOLDER_RESULTS:-results_bench_baseline}
FOLDER_LOGS=${FOLDER_LOGS:-$FOLDER_RESULTS/logs}
ID_MODEL=${ID_MODEL:-GSAI-ML/LLaDA-8B-Base}

FILTER_TASK=${FILTER_TASK:-}

NUM_BLOCKS=${NUM_BLOCKS:-1}
NUM_UNMASK=1
ID_MASK=126336

# must stay identical to run_bench_router.bash
BENCHMARKS=(
    "minerva_math:512:4:no"
    "bbh:256:3:no"
    "mbpp:512:3:yes"
    "humaneval:512:0:yes"
    "truthfulqa_gen:256:0:no"
)

mkdir -p "$FOLDER_RESULTS" "$FOLDER_LOGS"

num_run=0
num_skip=0
num_fail=0

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r task len_target nshot unsafe <<< "$entry"

    if [ -n "$FILTER_TASK" ] && [[ "$task" != *"$FILTER_TASK"* ]]; then
        continue
    fi

    tag="baseline__${task}"
    path_runner="$FOLDER_RESULTS/${tag}__runner.json"

    if [ -f "$path_runner" ]; then
        echo "SKIP $tag: runner report already exists"
        num_skip=$((num_skip + 1))
        continue
    fi

    flag_unsafe=""
    if [ "$unsafe" = "yes" ]; then
        flag_unsafe="--confirm_run_unsafe_code"
    fi

    echo "=== $tag (len_target=$len_target, nshot=$nshot, limit=$LIMIT) ==="
    num_run=$((num_run + 1))

    accelerate launch --num_processes=1 run_benchmark_main.py \
        --tasks "$task" --limit "$LIMIT" --model test --batch_size 1 \
        --num_fewshot "$nshot" --device "$DEVICE" $flag_unsafe \
        --output_path "$FOLDER_RESULTS/$tag" \
        --model_args "id_model=$ID_MODEL,size_batch=1,len_target=$len_target,num_blocks=$NUM_BLOCKS,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,runner=run_llada_semi,path_report=$path_runner" \
        2>&1 | tee "$FOLDER_LOGS/${tag}.log"

    if [ ! -f "$path_runner" ]; then
        echo "FAILED: $tag (see $FOLDER_LOGS/${tag}.log)"
        num_fail=$((num_fail + 1))
    fi
done

echo
echo "baseline sweep complete: $num_run launched, $num_skip skipped, $num_fail failed -> $FOLDER_RESULTS"
