#!/bin/bash
#################################################
# End-to-end sweep: every router bundle in FOLDER_ROUTERS is evaluated on
# gsm8k (len_target 128) and ifeval (len_target 256) through the mlp-llada
# runner. Fixed: model=llada, h=5, refresh_interval=16, num_blocks=1,
# num_unmask_per_step=1.
#
# Per run outputs:
#   results_e2e/<version>__<task>/            lm_eval metrics (--output_path)
#   results_e2e/<version>__<task>__runner.json   per-sample durations/has_done
#
# Usage:
#   FOLDER_ROUTERS=routers_e2e LIMIT=200 DEVICE=cuda:0 bash run_e2e_eval.bash
#   FILTER=data-mix bash run_e2e_eval.bash     # only bundles matching a substring
#################################################

set -u

FOLDER_ROUTERS=${FOLDER_ROUTERS:-routers_e2e}
FOLDER_RESULTS=${FOLDER_RESULTS:-results_e2e}
LIMIT=${LIMIT:-200}
DEVICE=${DEVICE:-cuda:0}
FILTER=${FILTER:-}
ID_MODEL=${ID_MODEL:-GSAI-ML/LLaDA-8B-Base}

H=5
REFRESH=16
NUM_BLOCKS=1
NUM_UNMASK=1
ID_MASK=126336

mkdir -p "$FOLDER_RESULTS"

for path_pt in "$FOLDER_ROUTERS"/*.pt; do
    name=$(basename "$path_pt" .pt)

    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
        continue
    fi

    if [ ! -f "${path_pt%.pt}.json" ]; then
        echo "SKIP $name: missing sidecar json"
        continue
    fi

    for task in gsm8k; do
        if [ "$task" = "gsm8k" ]; then
            LEN_TARGET=128
            NSHOT=5
        else
            LEN_TARGET=256
            NSHOT=1
        fi

        tag="${name}__${task}"
        if [ -f "$FOLDER_RESULTS/${tag}__runner.json" ]; then
            echo "SKIP $tag: runner report already exists"
            continue
        fi

        echo "=== $tag (len_target=$LEN_TARGET, limit=$LIMIT) ==="
        accelerate launch --num_processes=1 run_benchmark_main.py \
            --tasks "$task" --limit "$LIMIT" --model test --batch_size 1 \
            --num_fewshot "$NSHOT" --device "$DEVICE" \
            --output_path "$FOLDER_RESULTS/$tag" \
            --model_args "id_model=$ID_MODEL,size_batch=1,len_target=$LEN_TARGET,num_blocks=$NUM_BLOCKS,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,step_refresh_remainder=$REFRESH,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=$H,path_router=$path_pt,path_report=$FOLDER_RESULTS/${tag}__runner.json" \
            || echo "FAILED: $tag"
    done
done

echo "sweep complete -> $FOLDER_RESULTS"
