#!/bin/bash
#################################################
# Mockup collection driver for the extended benchmark set (stage 1 of the
# four-thread pipeline): builds the tail-percent CSV pools that oracle
# collection reads. Resume-safe: an existing CSV is skipped.
#
# Env overrides:
#   PERCENT        tail fraction kept per task          (default 0.1)
#   FOLDER_OUTPUT  where CSVs land                      (default benchmark_mockup)
#   FILTER_TASK    run only this task (include filter)  (default: all)
#
# Fewshot settings mirror the eval sweeps (run_bench_*.bash) so mockup prompts
# match eval prompts. Group tasks (minerva_math, bbh) are merged into ONE csv
# each (rows keep their subtask in task_name; load_benchmark_mockup can filter).
#
# followbench is NOT in stock lm_eval; the entry stays here so the run picks it
# up automatically once a custom task yaml is provided (--include_path), and
# fails soft with a note until then.
#################################################

set -u

PERCENT=${PERCENT:-0.1}
FOLDER_OUTPUT=${FOLDER_OUTPUT:-benchmark_mockup}
FILTER_TASK=${FILTER_TASK:-}
PCT=$(awk "BEGIN{printf \"%d\", ${PERCENT}*100}")

# task:num_fewshot:merge  (merge -> one combined csv for group tasks)
# gsm8k appears twice: 5-shot for the base threads, 0-shot for the instruct
# threads (their prompts are rebuilt at runtime: chat template / official CoT)
SPECS=(
    "gsm8k:5:"
    "gsm8k:0:"
    "minerva_math:4:merge"
    "bbh:3:merge"
    "mbpp:3:"
    "humaneval:0:"
    "truthfulqa_gen:0:"
    "ifeval:0:"
    "followbench:0:"
)

mkdir -p "$FOLDER_OUTPUT"

for spec in "${SPECS[@]}"; do
    IFS=':' read -r task fewshot merge <<< "$spec"

    if [ -n "$FILTER_TASK" ] && [ "$task" != "$FILTER_TASK" ]; then
        continue
    fi

    tag="${fewshot}shot"
    path_csv="$FOLDER_OUTPUT/mockup_${task}_${tag}_p${PCT}.csv"
    if [ -f "$path_csv" ]; then
        echo "[skip] $path_csv exists"
        continue
    fi

    model_args="percent=$PERCENT,folder_output=$FOLDER_OUTPUT,tag=$tag"
    if [ -n "$merge" ]; then
        model_args="$model_args,merge=$task"
    fi

    # code benchmarks refuse to run without the unsafe-code confirmation even
    # under --predict_only (the gate sits at task load, next to HF_ALLOW_CODE_EVAL)
    flags_extra=""
    if [ "$task" = "mbpp" ] || [ "$task" = "humaneval" ]; then
        flags_extra="--confirm_run_unsafe_code"
    fi

    echo "[mockup] $task (num_fewshot=$fewshot)"
    python save_benchmark_mockup.py \
        --tasks "$task" \
        --model mockup \
        --num_fewshot "$fewshot" \
        --model_args "$model_args" \
        --predict_only \
        --output_path "$FOLDER_OUTPUT/lm_eval_logs" \
        $flags_extra \
        || echo "[warn] $task failed -- if this is followbench, it needs a custom task yaml (--include_path); see save_benchmark_mockup.py header"
done
