#!/bin/bash
#################################################
# d2Cache (in-framework reimplementation) on the extended benchmark set,
# per thread. Runs the SAME harness, prompts, lengths, fewshots and LIMIT
# conventions as run_bench_baseline.bash / run_bench_router.bash, so accuracy
# and wall-clock are directly comparable across baseline / router / d2cache.
#
# Method arm only (their published protocol: blockless full canvas, k=32,
# sigma=10, rollout p=0.1, inflation off). The no-cache reference is the
# existing run_bench_baseline results. d2c_conf_mode:
#   live   (default) their INTENDED design -- conf table refreshed at queried rows
#   frozen           their RELEASED code -- conf immutable after prefill
#
# Usage:
#   THREAD=llada_base   DEVICE=cuda:0 LIMIT=450 bash run_bench_d2cache.bash
#   THREAD=dream_instruct DEVICE=cuda:1 LIMIT=450 CONF_MODE=frozen bash run_bench_d2cache.bash
#
# Env: THREAD (llada_base|llada_instruct|dream_base|dream_instruct), DEVICE,
#      LIMIT, FILTER_TASK, CONF_MODE, D2C_K, D2C_SIGMA, D2C_ROLLOUT_P,
#      FOLDER_RESULTS. Resume: a run with an existing runner.json is skipped.
#################################################

set -u

THREAD=${THREAD:-llada_base}
DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-100}
FILTER_TASK=${FILTER_TASK:-}
CONF_MODE=${CONF_MODE:-live}
D2C_K=${D2C_K:-32}
D2C_SIGMA=${D2C_SIGMA:-10.0}
D2C_ROLLOUT_P=${D2C_ROLLOUT_P:-0.1}

FOLDER_RESULTS=${FOLDER_RESULTS:-results_bench_d2cache/$THREAD}
FOLDER_LOGS=${FOLDER_LOGS:-$FOLDER_RESULTS/logs}

NUM_UNMASK=1

case "$THREAD" in
    llada_base)
        ID_MODEL="GSAI-ML/LLaDA-8B-Base";        ID_MASK=126336; RUNNER=run_llada_d2cache; FLAGS_MODEL="" ;;
    llada_instruct)
        ID_MODEL="GSAI-ML/LLaDA-8B-Instruct";    ID_MASK=126336; RUNNER=run_llada_d2cache
        FLAGS_MODEL=",use_chat_template=True,truncate_at_eos=True" ;;
    dream_base)
        ID_MODEL="Dream-org/Dream-v0-Base-7B";   ID_MASK=151666; RUNNER=run_dream_d2cache; FLAGS_MODEL="" ;;
    dream_instruct)
        ID_MODEL="Dream-org/Dream-v0-Instruct-7B"; ID_MASK=151666; RUNNER=run_dream_d2cache
        FLAGS_MODEL=",use_chat_template=True,truncate_at_eos=True" ;;
    *)
        echo "unknown THREAD=$THREAD"; exit 1 ;;
esac

# task:len_target:nshot:unsafe -- canonical dllm-meta spec; gsm8k included here
# (the baseline/router sweeps carry it separately per thread)
BENCHMARKS=(
    "gsm8k:256:5:no"
    "minerva_math:512:4:no"
    "bbh:256:3:no"
    "mbpp:512:3:yes"
    "humaneval:512:0:yes"
    "truthfulqa_gen:256:0:no"
)
if [[ "$THREAD" == *_instruct ]]; then
    # instruct protocol: 0-shot gsm8k (chat template carries the format)
    BENCHMARKS[0]="gsm8k:256:0:no"
fi

mkdir -p "$FOLDER_RESULTS" "$FOLDER_LOGS"

num_run=0
num_skip=0
num_fail=0

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r task len_target nshot unsafe <<< "$entry"

    if [ -n "$FILTER_TASK" ] && [[ "$task" != *"$FILTER_TASK"* ]]; then
        continue
    fi

    tag="d2cache-${CONF_MODE}__${task}"
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

    # LIMIT is a per-TASK budget; lm_eval applies --limit per SUBTASK, so
    # divide (ceil) for group tasks -- identical to the other sweeps so all
    # runs share subsets. humaneval capped at 148 (head-split scheme: docs
    # 148..163 are router training data).
    limit_task="$LIMIT"
    case "$task" in
        minerva_math) limit_task=$(( (LIMIT + 6) / 7 )) ;;
        bbh)          limit_task=$(( (LIMIT + 26) / 27 )) ;;
        humaneval)    [ "$LIMIT" -gt 148 ] && limit_task=148 ;;
    esac

    echo "=== $tag (len_target=$len_target, nshot=$nshot, limit=$limit_task) ==="
    num_run=$((num_run + 1))

    HF_ALLOW_CODE_EVAL="$allow_code_eval" \
    accelerate launch --num_processes=1 run_benchmark_main.py \
        --tasks "$task" --limit "$limit_task" --model test --batch_size 1 \
        --num_fewshot "$nshot" --device "$DEVICE" $flag_unsafe \
        --output_path "$FOLDER_RESULTS/$tag" \
        --model_args "id_model=$ID_MODEL,size_batch=1,len_target=$len_target,num_blocks=1,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,runner=$RUNNER,d2c_k=$D2C_K,d2c_sigma=$D2C_SIGMA,d2c_rollout_p=$D2C_ROLLOUT_P,d2c_conf_mode=$CONF_MODE,path_report=$path_runner$FLAGS_MODEL" \
        2>&1 | tee "$FOLDER_LOGS/${tag}.log"

    if [ ! -f "$path_runner" ]; then
        echo "FAILED: $tag (see $FOLDER_LOGS/${tag}.log)"
        num_fail=$((num_fail + 1))
    fi
done

echo "d2cache sweep [$THREAD/$CONF_MODE]: ran $num_run, skipped $num_skip, failed $num_fail -> $FOLDER_RESULTS"
