#!/bin/bash
#################################################
# Extended-benchmark sweep for the two winning router settings.
#
# Settings (from the e2e grid on gsm8k):
#   best:        attn_last + geo, no confidence, softmax_attn, plackett_luce
#   second best: same + aged confidence
# each crossed with the three training sets (gsm8k / ifeval / mix) -> 6 routers.
#
# Fixed decoding parameters (same as the e2e stage):
#   model=llada, h=5, refresh_interval=16, num_blocks=1, num_unmask_per_step=1
#
# Per run outputs (under FOLDER_RESULTS):
#   <router>__<task>/                 lm_eval metrics (--output_path)
#   <router>__<task>__runner.json     per-sample duration / has_done
#   logs/<router>__<task>.log         full stdout+stderr
#
# Usage:
#   DEVICE=cuda:0 LIMIT=100 bash run_bench_router.bash
#   FILTER_TASK=humaneval FILTER_ROUTER=data-mix bash run_bench_router.bash
#
# Runtime warning: 6 routers x 5 benchmarks = 30 runs. At ~11 s/sample for
# 512-token generation, LIMIT=100 is roughly 18 min/run (~9 h total).
# Start with LIMIT=20 to shake out task-config issues; the script is
# resume-safe, so a later larger LIMIT only needs the runner.json removed.
#################################################

set -u

DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-100}

FOLDER_ROUTERS=${FOLDER_ROUTERS:-routers_e2e}
FOLDER_RESULTS=${FOLDER_RESULTS:-results_bench_router}
FOLDER_LOGS=${FOLDER_LOGS:-$FOLDER_RESULTS/logs}
ID_MODEL=${ID_MODEL:-GSAI-ML/LLaDA-8B-Base}

FILTER_ROUTER=${FILTER_ROUTER:-}
FILTER_TASK=${FILTER_TASK:-}

H=5
REFRESH=16
NUM_BLOCKS=${NUM_BLOCKS:-1}
NUM_UNMASK=1
ID_MASK=126336

BEST="feat-attn_last_geo__conf-none__norm-softmax_attn__loss-plackett_luce"
SECOND="feat-attn_last_geo_conf__conf-aged__norm-softmax_attn__loss-plackett_luce"

ROUTERS=(
    "${BEST}__data-gsm8k"
    "${BEST}__data-ifeval"
    "${BEST}__data-mix"
    "${SECOND}__data-gsm8k"
    "${SECOND}__data-ifeval"
    "${SECOND}__data-mix"
)

# task:len_target:num_fewshot:needs_unsafe_code
# NOTE on task names/shots -- adjust to the lm_eval version on this box:
#   bbh          : if the group resolves to bbh_cot_fewshot, the shots live in
#                  the prompt template and NSHOT must be 0
#   minerva_math : 4-shot is the convention
#   mbpp/humaneval: execute generated code -> --confirm_run_unsafe_code
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

for name in "${ROUTERS[@]}"; do
    path_pt="$FOLDER_ROUTERS/$name.pt"

    if [ -n "$FILTER_ROUTER" ] && [[ "$name" != *"$FILTER_ROUTER"* ]]; then
        continue
    fi

    if [ ! -f "$path_pt" ] || [ ! -f "${path_pt%.pt}.json" ]; then
        echo "SKIP router $name: bundle or sidecar missing under $FOLDER_ROUTERS"
        num_skip=$((num_skip + 1))
        continue
    fi

    for entry in "${BENCHMARKS[@]}"; do
        IFS=':' read -r task len_target nshot unsafe <<< "$entry"

        if [ -n "$FILTER_TASK" ] && [[ "$task" != *"$FILTER_TASK"* ]]; then
            continue
        fi

        tag="${name}__${task}"
        path_runner="$FOLDER_RESULTS/${tag}__runner.json"

        if [ -f "$path_runner" ]; then
            echo "SKIP $tag: runner report already exists"
            num_skip=$((num_skip + 1))
            continue
        fi

        flag_unsafe=""
        allow_code_eval=""
        if [ "$unsafe" = "yes" ]; then
            # two independent gates for code benchmarks:
            #   --confirm_run_unsafe_code : lm_eval's own flag
            #   HF_ALLOW_CODE_EVAL=1      : HF evaluate's code_eval metric gate,
            #                               checked at TASK LOAD time (mbpp/utils.py
            #                               runs a pass@k smoke test on import)
            flag_unsafe="--confirm_run_unsafe_code"
            allow_code_eval="1"
        fi

        echo "=== $tag (len_target=$len_target, nshot=$nshot, limit=$LIMIT) ==="
        num_run=$((num_run + 1))

        HF_ALLOW_CODE_EVAL="$allow_code_eval" \
        accelerate launch --num_processes=1 run_benchmark_main.py \
            --tasks "$task" --limit "$LIMIT" --model test --batch_size 1 \
            --num_fewshot "$nshot" --device "$DEVICE" $flag_unsafe \
            --output_path "$FOLDER_RESULTS/$tag" \
            --model_args "id_model=$ID_MODEL,size_batch=1,len_target=$len_target,num_blocks=$NUM_BLOCKS,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,step_refresh_remainder=$REFRESH,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=$H,path_router=$path_pt,path_report=$path_runner" \
            2>&1 | tee "$FOLDER_LOGS/${tag}.log"

        if [ ! -f "$path_runner" ]; then
            echo "FAILED: $tag (see $FOLDER_LOGS/${tag}.log)"
            num_fail=$((num_fail + 1))
        fi
    done
done

echo
echo "router sweep complete: $num_run launched, $num_skip skipped, $num_fail failed -> $FOLDER_RESULTS"
