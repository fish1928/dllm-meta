#!/bin/bash
#################################################
# Oracle-trajectory collection driver (stage 2): runs one THREAD's collector
# over every benchmark mockup CSV. Resume-safe at task granularity: an existing
# output folder is skipped (delete it to recollect).
#
#   THREAD=llada_base     bash run_collect_oracle.bash
#   THREAD=llada_instruct NUM_BLOCKS_LIST="32 8 1" bash run_collect_oracle.bash
#
# Env overrides:
#   THREAD           llada_base | llada_instruct | dream_base | dream_instruct
#   DEVICE           cuda device                         (default cuda:0)
#   LIMIT            cap on mockup rows per task         (default: all)
#   FILTER_TASK      run only this task (include filter) (default: all)
#   NUM_BLOCKS_LIST  space-separated sweep               (default "1";
#                    llada_instruct is the thread meant for sweeping this)
#   PERCENT          mockup tail percent, for filenames  (default 0.1)
#   FOLDER_MOCKUP    mockup CSV folder                   (default benchmark_mockup)
#   FOLDER_OUTPUT    stats root                          (default stats_oracle)
#   OFFICIAL_GSM8K   1 -> llada_instruct gsm8k uses the OpenCompass 4-shot CoT
#                    prompt rebuild (default 1; needs the 0-shot gsm8k mockup)
#
# len_target per task mirrors the eval sweeps: 512 for minerva_math/mbpp/
# humaneval, 256 otherwise. gsm8k pool: 5-shot CSV for base threads, 0-shot
# for instruct threads (prompts are rebuilt at runtime there).
#################################################

set -u

THREAD=${THREAD:-llada_base}
DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-}
FILTER_TASK=${FILTER_TASK:-}
NUM_BLOCKS_LIST=${NUM_BLOCKS_LIST:-1}
PERCENT=${PERCENT:-0.1}
FOLDER_MOCKUP=${FOLDER_MOCKUP:-benchmark_mockup}
FOLDER_OUTPUT=${FOLDER_OUTPUT:-stats_oracle}
OFFICIAL_GSM8K=${OFFICIAL_GSM8K:-1}

PCT=$(awk "BEGIN{printf \"%d\", ${PERCENT}*100}")

case "$THREAD" in
    llada_base|llada_instruct|dream_base|dream_instruct) ;;
    *) echo "unknown THREAD=$THREAD"; exit 1 ;;
esac

case "$THREAD" in
    *_instruct) TAG_GSM8K="0shot" ;;
    *)          TAG_GSM8K="5shot" ;;
esac

# task:len_target:mockup_tag
SPECS=(
    "gsm8k:256:$TAG_GSM8K"
    "minerva_math:512:4shot"
    "bbh:256:3shot"
    "mbpp:512:3shot"
    "humaneval:512:0shot"
    "truthfulqa_gen:256:0shot"
    "ifeval:256:0shot"
    "followbench:256:0shot"
)

for spec in "${SPECS[@]}"; do
    IFS=':' read -r task len_target tag <<< "$spec"

    if [ -n "$FILTER_TASK" ] && [ "$task" != "$FILTER_TASK" ]; then
        continue
    fi

    path_csv="$FOLDER_MOCKUP/mockup_${task}_${tag}_p${PCT}.csv"
    if [ ! -f "$path_csv" ]; then
        echo "[warn] $path_csv missing -- run run_save_mockups.bash first (or the task has no mockup yet); skipping $task"
        continue
    fi

    for num_blocks in $NUM_BLOCKS_LIST; do
        folder_out="$FOLDER_OUTPUT/${THREAD}_${task}_b${num_blocks}"
        if [ -d "$folder_out" ]; then
            echo "[skip] $folder_out exists"
            continue
        fi

        flags_extra=""
        if [ "$THREAD" = "llada_instruct" ] && [ "$task" = "gsm8k" ] && [ "$OFFICIAL_GSM8K" = "1" ]; then
            flags_extra="--use_official_gsm8k_prompt"
        fi

        echo "[oracle] thread=$THREAD task=$task len_target=$len_target num_blocks=$num_blocks"
        python "run_collect_metrics_${THREAD}.py" \
            --path_mockup "$path_csv" \
            --folder_output "$folder_out" \
            --len_target "$len_target" \
            --num_blocks "$num_blocks" \
            --device "$DEVICE" \
            ${LIMIT:+--limit "$LIMIT"} \
            $flags_extra \
            || echo "[warn] collection failed: $THREAD/$task/b$num_blocks -- continuing"
    done
done
