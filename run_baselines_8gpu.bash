#!/bin/bash
#################################################
# BASELINE CACHING METHODS on 8 GPUs -- fast-dllm / dllm-cache / d2cache,
# all four threads, same benchmark protocol and limits as the full-denoise
# campaign (run_full_denoise_8gpu.bash), via the same self-balancing queue.
#
#   methods:  fastdllm  (run_*_fastdllm:  DualCache ONLY, uniform greedy
#                        1-token/step decode -- the paper's parallel decoding
#                        is not implemented here, so only the cache is
#                        measured; block 32 -> num_blocks=len/32)
#             dllmcache (run_*_dllm_cache: adaptive V-sim update,
#                        v_rate 0.25, Kp 64, Kr 8, one block -- one consistent
#                        setting across threads/tasks, inside the paper's
#                        per-task ranges: their Kp 5-100 avg ~57, Kr 1-8)
#             d2cache   (run_*_d2cache:   certainty-density top-k + rollout,
#                        k 32, sigma 10, p 0.1, conf live, one block)
#   threads:  llada_base llada_instruct dream_base dream_instruct
#   limits:   gsm8k FULL (1319) | minerva 500/subtask (x7) | bbh 125/subtask (x27)
#             mbpp FULL (500)   | humaneval FULL (164)     | truthfulqa FULL (817)
#   protocol: instruct gsm8k = official 4-shot CoT prompt; everything else NO
#             chat template; instruct threads add truncate_at_eos=True (these
#             method runners are thread-shared); code tasks + fence cut.
#
# BUDGET WARNING: 72 jobs. dllm-cache runs near full-denoise speed by design
# (full-window V-proj + LM head floor), so its 24 jobs alone cost about one
# full-denoise campaign; fastdllm and d2cache are several-x cheaper. Slice with
# the METHODS / THREADS / TASKS filters if the clock is tight, and rate-check
# the *__runner.json files after ~2 hours.
#
# RESUME: same contract as the full-denoise script (.done sentinel written only
# after a clean exit WITH a runner report; stale claims cleared at startup).
#
# Usage:
#   SMOKE=1 nohup bash run_baselines_8gpu.bash > baselines_smoke.log 2>&1 &
#   nohup bash run_baselines_8gpu.bash > baselines.log 2>&1 &
#   METHODS="fastdllm d2cache" TASKS="gsm8k humaneval" bash run_baselines_8gpu.bash
#   DRY_RUN=1 bash run_baselines_8gpu.bash
#################################################

set -u

GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
DRY_RUN=${DRY_RUN:-}
SMOKE=${SMOKE:-}    # SMOKE=1: real runs at --limit 1 into results_smoke_baselines
if [ -n "$SMOKE" ]; then
    FOLDER_RESULTS=${FOLDER_RESULTS:-results_smoke_baselines}
else
    FOLDER_RESULTS=${FOLDER_RESULTS:-results_baselines}
fi
FOLDER_LOGS="$FOLDER_RESULTS/logs"
FOLDER_CLAIMS="$FOLDER_RESULTS/claims"
PORT_BASE=${PORT_BASE:-14000}

METHODS=${METHODS:-"dllmcache d2cache fastdllm"}
THREADS=${THREADS:-"llada_base llada_instruct dream_base dream_instruct"}
TASKS=${TASKS:-"minerva_math bbh gsm8k truthfulqa_gen mbpp humaneval"}

# method hyperparameters (override via env); fastdllm has none beyond num_blocks
DLLMC_VRATE=${DLLMC_VRATE:-0.25}
DLLMC_KR=${DLLMC_KR:-8}
DLLMC_KP=${DLLMC_KP:-64}
D2C_K=${D2C_K:-32}
D2C_SIGMA=${D2C_SIGMA:-10.0}
D2C_ROLLOUT_P=${D2C_ROLLOUT_P:-0.1}
D2C_CONF_MODE=${D2C_CONF_MODE:-live}

mkdir -p "$FOLDER_RESULTS" "$FOLDER_LOGS" "$FOLDER_CLAIMS"

task_params () {    # sets LEN, NSHOT, LIMIT, MODE for $1
    case "$1" in
        minerva_math)   LEN=512; NSHOT=4; LIMIT=500;  MODE=plain ;;
        bbh)            LEN=256; NSHOT=3; LIMIT=125;  MODE=plain ;;
        gsm8k)          LEN=256; NSHOT=5; LIMIT=full; MODE=plain ;;    # instructs switch to official below
        truthfulqa_gen) LEN=256; NSHOT=0; LIMIT=full; MODE=plain ;;
        mbpp)           LEN=512; NSHOT=3; LIMIT=full; MODE=code ;;
        humaneval)      LEN=512; NSHOT=0; LIMIT=full; MODE=code ;;
    esac
}

thread_params () {    # sets ID_MODEL, ID_MASK, FAMILY, IS_INSTRUCT for $1
    case "$1" in
        llada_base)     ID_MODEL="GSAI-ML/LLaDA-8B-Base";          ID_MASK=126336; FAMILY=llada; IS_INSTRUCT= ;;
        llada_instruct) ID_MODEL="GSAI-ML/LLaDA-8B-Instruct";      ID_MASK=126336; FAMILY=llada; IS_INSTRUCT=1 ;;
        dream_base)     ID_MODEL="Dream-org/Dream-v0-Base-7B";     ID_MASK=151666; FAMILY=dream; IS_INSTRUCT= ;;
        dream_instruct) ID_MODEL="Dream-org/Dream-v0-Instruct-7B"; ID_MASK=151666; FAMILY=dream; IS_INSTRUCT=1 ;;
    esac
}

method_params () {    # sets RUNNER, NUM_BLOCKS, ARGS_METHOD for method=$1 family=$2 len=$3
    case "$1" in
        fastdllm)
            RUNNER="run_${2}_fastdllm"
            NUM_BLOCKS=$(( $3 / 32 ))
            ARGS_METHOD="" ;;
        dllmcache)
            RUNNER="run_${2}_dllm_cache"
            NUM_BLOCKS=1
            ARGS_METHOD=",dllmc_v_rate=$DLLMC_VRATE,step_refresh_remainder=$DLLMC_KR,step_refresh_remainder_prompt=$DLLMC_KP" ;;
        d2cache)
            RUNNER="run_${2}_d2cache"
            NUM_BLOCKS=1
            ARGS_METHOD=",d2c_k=$D2C_K,d2c_sigma=$D2C_SIGMA,d2c_rollout_p=$D2C_ROLLOUT_P,d2c_conf_mode=$D2C_CONF_MODE" ;;
    esac
}

# job list: TASKS outer (largest first), then methods (dllmcache = slowest first),
# then threads -- keeps the expensive tiers at the front of the queue
JOBS=()
for task in $TASKS; do
    for method in $METHODS; do
        for thread in $THREADS; do
            JOBS+=( "${method}:${thread}:${task}" )
        done
    done
done

# clear stale claims (claimed but not done -> a previous run died mid-job)
for claim in "$FOLDER_CLAIMS"/*/; do
    [ -d "$claim" ] || continue
    tag=$(basename "$claim")
    if [ ! -f "$FOLDER_RESULTS/$tag.done" ]; then
        rm -rf "$claim"
        echo "[startup] cleared stale claim: $tag"
    fi
done

run_job () {    # $1 = gpu id, $2 = job index, $3 = job spec
    local gpu="$1" idx="$2" spec="$3"
    IFS=':' read -r method thread task <<< "$spec"
    local tag="${method}__${thread}__${task}"

    task_params "$task"
    thread_params "$thread"
    method_params "$method" "$FAMILY" "$LEN"

    local mode="$MODE" nshot="$NSHOT"
    local args_extra="" flag_unsafe="" allow_code=""
    if [ -n "$IS_INSTRUCT" ]; then
        args_extra=",truncate_at_eos=True"
        if [ "$task" = "gsm8k" ]; then
            mode=official; nshot=0
        fi
    fi
    case "$mode" in
        official) args_extra="$args_extra,use_official_gsm8k_prompt=True" ;;
        code)     args_extra="$args_extra,stop_at_code_fence=True"; flag_unsafe="--confirm_run_unsafe_code"; allow_code="1" ;;
    esac

    local flag_limit=""
    [ "$LIMIT" != "full" ] && flag_limit="--limit $LIMIT"
    [ -n "$SMOKE" ] && flag_limit="--limit 1"    # 1 doc (1/subtask for groups)

    local path_runner="$FOLDER_RESULTS/${tag}__runner.json"
    local port=$(( PORT_BASE + idx ))

    echo "[gpu$gpu] START $tag (len=$LEN nshot=$nshot limit=$LIMIT mode=$mode blocks=$NUM_BLOCKS) $(date '+%F %T')"
    if [ -n "$DRY_RUN" ]; then
        echo "  DRY: CUDA_VISIBLE_DEVICES=$gpu HF_ALLOW_CODE_EVAL=$allow_code accelerate launch --num_processes=1 --main_process_port $port run_benchmark_main.py --tasks $task $flag_limit --num_fewshot $nshot $flag_unsafe --model_args \"id_model=$ID_MODEL,runner=$RUNNER,num_blocks=$NUM_BLOCKS,len_target=$LEN$ARGS_METHOD$args_extra\""
        return 0    # no .done in DRY_RUN: a real launch afterwards must not skip
    fi

    CUDA_VISIBLE_DEVICES="$gpu" HF_ALLOW_CODE_EVAL="$allow_code" \
    accelerate launch --num_processes=1 --main_process_port "$port" run_benchmark_main.py \
        --tasks "$task" $flag_limit --model test --batch_size 1 \
        --num_fewshot "$nshot" --device cuda $flag_unsafe \
        --output_path "$FOLDER_RESULTS/$tag" \
        --model_args "id_model=$ID_MODEL,size_batch=1,len_target=$LEN,num_blocks=$NUM_BLOCKS,num_unmask_per_step=1,id_mask=$ID_MASK,runner=$RUNNER,path_report=$path_runner$ARGS_METHOD$args_extra" \
        > "$FOLDER_LOGS/${tag}.log" 2>&1
    local status=$?

    if [ $status -eq 0 ] && [ -f "$path_runner" ]; then
        touch "$FOLDER_RESULTS/$tag.done"
        echo "[gpu$gpu] DONE  $tag $(date '+%F %T')"
    else
        rm -rf "$FOLDER_CLAIMS/$tag"    # release so a rerun retries it
        echo "[gpu$gpu] FAILED $tag (exit $status) -- see $FOLDER_LOGS/${tag}.log"
    fi
}

worker () {    # one worker per GPU: claim jobs atomically until none remain
    local gpu="$1"
    local idx=0
    for spec in "${JOBS[@]}"; do
        idx=$(( idx + 1 ))
        IFS=':' read -r method thread task <<< "$spec"
        local tag="${method}__${thread}__${task}"
        [ -f "$FOLDER_RESULTS/$tag.done" ] && continue
        if mkdir "$FOLDER_CLAIMS/$tag" 2>/dev/null; then
            run_job "$gpu" "$idx" "$spec"
        fi
    done
    echo "[gpu$gpu] worker finished: no unclaimed jobs left"
}

for gpu in $GPU_LIST; do
    worker "$gpu" &
done
wait

echo
n_done=$(ls "$FOLDER_RESULTS"/*.done 2>/dev/null | wc -l)
echo "ALL WORKERS DONE: $n_done/${#JOBS[@]} jobs complete -> $FOLDER_RESULTS"

# verdict table: per job, done/failed + the lm_eval metrics actually produced
if [ -z "$DRY_RUN" ]; then
    python - "$FOLDER_RESULTS" <<'PYEOF'
import glob, json, os, sys
folder = sys.argv[1]
print(f'\n===== verdict ({folder}) =====')
tags = sorted(os.path.basename(p)[:-len('__runner.json')]
              for p in glob.glob(os.path.join(folder, '*__runner.json')))
for tag in tags:
    done = os.path.exists(os.path.join(folder, tag + '.done'))
    paths = glob.glob(os.path.join(folder, tag, '**', 'results_*.json'), recursive=True)
    metrics = 'no lm_eval results'
    if paths:
        results = json.load(open(max(paths, key=os.path.getmtime))).get('results', {})
        parts = []
        for name, entry in results.items():
            for key, value in entry.items():
                if isinstance(value, float) and 'stderr' not in key and key != 'alias':
                    parts.append(f'{key}={value:.3f}')
            break    # first (group/main) entry is enough for the verdict
        metrics = ' '.join(parts) or 'no numeric metrics'
    print(f'  {"OK  " if done else "FAIL"} {tag:56s} {metrics}')
PYEOF
fi
echo "summary: python build_bench_html.py --results $FOLDER_RESULTS"
