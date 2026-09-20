#!/bin/bash
#################################################
# Deciding e2e for the conf/margin question: three routers, identical
# decoding, only the router bundle differs.
#
#   cm_clean    attn_last + geo                       (no conf/margin)
#   cm_policy   + conf/margin, policy-aged training   (offline recall@5 0.681)
#   cm_aged     + conf/margin, random-aged training   (offline recall@5 0.680)
#   cm_age      + conf/margin values WITH the true per-position age as an
#               input channel (the aged router; runner tracks snapshot.age)
#
# Decoding: run_llada_semi_mlp_v2, h=5, Kr=16 (generation refresh),
# Kp=96 (prompt refresh) -- the best-known config (~0.45 gsm8k @ ~125 TF).
# Tasks default to the deciding pair gsm8k + bbh (bbh is where the clean
# recipe collapsed offline: 0.347 vs 0.714 bundle recall); extend BENCHMARKS
# below for the full suite.
#
# Bundles come from: python ablation_tests/train_e2e_confmargin.py
#
# Usage:
#   DEVICE=cuda:0 LIMIT=150 bash run_bench_confmargin.bash
#   FILTER_ROUTER=cm_policy FILTER_TASK=bbh bash run_bench_confmargin.bash
# Resume-safe per run (skips when the runner report exists).
#################################################

set -u

DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-150}
THREAD=${THREAD:-llada_base}

FOLDER_ROUTERS=${FOLDER_ROUTERS:-routers_e2e}
FOLDER_RESULTS=${FOLDER_RESULTS:-results_bench_confmargin}
FOLDER_LOGS=${FOLDER_LOGS:-$FOLDER_RESULTS/logs}
ID_MODEL=${ID_MODEL:-GSAI-ML/LLaDA-8B-Base}

FILTER_ROUTER=${FILTER_ROUTER:-}
FILTER_TASK=${FILTER_TASK:-}

H=${H:-5}
KR=${KR:-16}       # step_refresh_remainder        (generation clock)
KP=${KP:-96}       # step_refresh_remainder_prompt (prompt clock)
NUM_BLOCKS=${NUM_BLOCKS:-1}
NUM_UNMASK=1
ID_MASK=126336
RUNNER=${RUNNER:-run_llada_semi_mlp_v2}

ROUTERS=(
    "${THREAD}__cm_clean"
    "${THREAD}__cm_policy"
    "${THREAD}__cm_aged"
    "${THREAD}__cm_age"
)

# task:len_target:num_fewshot:needs_unsafe_code -- canonical gen lengths;
# the deciding pair first, the rest of the suite commented for later
BENCHMARKS=(
    "gsm8k:256:5:no"
    "bbh:256:3:no"
    # "minerva_math:512:4:no"
    # "mbpp:512:3:yes"
    # "humaneval:512:0:yes"
    # "truthfulqa_gen:256:0:no"
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
        echo "SKIP router $name: bundle or sidecar missing under $FOLDER_ROUTERS (train first: ablation_tests/train_e2e_confmargin.py)"
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
            flag_unsafe="--confirm_run_unsafe_code"
            allow_code_eval="1"
        fi

        # LIMIT is a per-TASK budget; lm_eval applies --limit per SUBTASK
        limit_task="$LIMIT"
        case "$task" in
            minerva_math) limit_task=$(( (LIMIT + 6) / 7 )) ;;
            bbh)          limit_task=$(( (LIMIT + 26) / 27 )) ;;
            humaneval)    [ "$LIMIT" -gt 148 ] && limit_task=148 ;;
        esac

        echo "=== $tag (len_target=$len_target, nshot=$nshot, limit=$limit_task/subtask) ==="
        num_run=$((num_run + 1))

        HF_ALLOW_CODE_EVAL="$allow_code_eval" \
        accelerate launch --num_processes=1 run_benchmark_main.py \
            --tasks "$task" --limit "$limit_task" --model test --batch_size 1 \
            --num_fewshot "$nshot" --device "$DEVICE" $flag_unsafe \
            --output_path "$FOLDER_RESULTS/$tag" \
            --model_args "id_model=$ID_MODEL,size_batch=1,len_target=$len_target,num_blocks=$NUM_BLOCKS,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,step_refresh_remainder=$KR,step_refresh_remainder_prompt=$KP,select_only_in_h=True,runner=$RUNNER,h=$H,path_router=$path_pt,path_report=$path_runner" \
            2>&1 | tee "$FOLDER_LOGS/${tag}.log"

        if [ ! -f "$path_runner" ]; then
            echo "FAILED: $tag (see $FOLDER_LOGS/${tag}.log)"
            num_fail=$((num_fail + 1))
        fi
    done
done

echo
echo "conf/margin sweep complete: $num_run launched, $num_skip skipped, $num_fail failed -> $FOLDER_RESULTS"
