#!/bin/bash
#################################################
# FULL-DENOISING BASELINES on 8 GPUs -- the no-cache ceiling row for all
# four threads, big limits, via a self-balancing work queue.
#
#   threads:  llada_base (run_llada_semi)      llada_instruct (run_llada_instruct)
#             dream_base (run_dream_semi)      dream_instruct (run_dream_instruct)
#   limits:   gsm8k FULL (1319) | minerva 500 (72/subtask) | bbh 125 (5/subtask)
#             mbpp FULL (500)   | humaneval FULL (164)     | truthfulqa FULL (817)
#   protocol: instruct threads' gsm8k = official 4-shot CoT prompt; everything
#             else NO chat template (all threads); code tasks + fence cut;
#             llada_instruct num_blocks = len/32, all others one-block.
#
# Scheduling: 24 jobs, listed LARGEST FIRST; N_GPUS workers each claim jobs
# atomically (mkdir) and run one job per GPU at a time.
#
# RESUME: a job is done only when its .done sentinel exists (written after a
# clean exit WITH a runner report). Rerunning the script re-queues everything
# unfinished; stale claims from a crashed run are cleared at startup.
#
# Usage:
#   nohup bash run_full_denoise_8gpu.bash > full_denoise.log 2>&1 &
#   N_GPUS=8 GPU_LIST="0 1 2 3 4 5 6 7" ...   (defaults)
#   DRY_RUN=1 bash run_full_denoise_8gpu.bash
#################################################

set -u

GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
DRY_RUN=${DRY_RUN:-}
SMOKE=${SMOKE:-}    # SMOKE=1: real runs at --limit 1 into results_smoke_env --
                    # exercises model downloads, tokenizers, runners, lm_eval
                    # task loading (antlr/math_verify/code-eval gates), and
                    # GPU pinning; ~30-60 min on 8 GPUs. Validate the
                    # environment with this BEFORE the real launch.
if [ -n "$SMOKE" ]; then
    FOLDER_RESULTS=${FOLDER_RESULTS:-results_smoke_env}
else
    FOLDER_RESULTS=${FOLDER_RESULTS:-results_full_denoise}
fi
FOLDER_LOGS="$FOLDER_RESULTS/logs"
FOLDER_CLAIMS="$FOLDER_RESULTS/claims"
PORT_BASE=${PORT_BASE:-13000}

mkdir -p "$FOLDER_RESULTS" "$FOLDER_LOGS" "$FOLDER_CLAIMS"

# thread:task:len:nshot:limit:mode
#   limit: number = --limit N (per subtask for groups); full = no --limit
#   mode:  plain | official (gsm8k instruct) | code (plain + fence + unsafe)
# LARGEST JOBS FIRST so the queue balances across workers.
JOBS=(
    "llada_base:gsm8k:256:5:full:plain"
    "llada_instruct:gsm8k:256:0:full:official"
    "dream_base:gsm8k:256:5:full:plain"
    "dream_instruct:gsm8k:256:0:full:official"
    "llada_base:truthfulqa_gen:256:0:full:plain"
    "llada_instruct:truthfulqa_gen:256:0:full:plain"
    "dream_base:truthfulqa_gen:256:0:full:plain"
    "dream_instruct:truthfulqa_gen:256:0:full:plain"
    "llada_base:mbpp:512:3:full:code"
    "llada_instruct:mbpp:512:3:full:code"
    "dream_base:mbpp:512:3:full:code"
    "dream_instruct:mbpp:512:3:full:code"
    "llada_base:minerva_math:512:4:72:plain"
    "llada_instruct:minerva_math:512:4:72:plain"
    "dream_base:minerva_math:512:4:72:plain"
    "dream_instruct:minerva_math:512:4:72:plain"
    "llada_base:humaneval:512:0:full:code"
    "llada_instruct:humaneval:512:0:full:code"
    "dream_base:humaneval:512:0:full:code"
    "dream_instruct:humaneval:512:0:full:code"
    "llada_base:bbh:256:3:5:plain"
    "llada_instruct:bbh:256:3:5:plain"
    "dream_base:bbh:256:3:5:plain"
    "dream_instruct:bbh:256:3:5:plain"
)

thread_params () {    # sets ID_MODEL, ID_MASK, RUNNER for $1
    case "$1" in
        llada_base)     ID_MODEL="GSAI-ML/LLaDA-8B-Base";        ID_MASK=126336; RUNNER=run_llada_semi ;;
        llada_instruct) ID_MODEL="GSAI-ML/LLaDA-8B-Instruct";    ID_MASK=126336; RUNNER=run_llada_instruct ;;
        dream_base)     ID_MODEL="Dream-org/Dream-v0-Base-7B";   ID_MASK=151666; RUNNER=run_dream_semi ;;
        dream_instruct) ID_MODEL="Dream-org/Dream-v0-Instruct-7B"; ID_MASK=151666; RUNNER=run_dream_instruct ;;
    esac
}

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
    IFS=':' read -r thread task len nshot limit mode <<< "$spec"
    local tag="${thread}__${task}"

    thread_params "$thread"

    local num_blocks=1
    [ "$thread" = "llada_instruct" ] && num_blocks=$(( len / 32 ))

    local args_extra="" flag_unsafe="" allow_code=""
    case "$mode" in
        official) args_extra=",use_official_gsm8k_prompt=True" ;;
        code)     args_extra=",stop_at_code_fence=True"; flag_unsafe="--confirm_run_unsafe_code"; allow_code="1" ;;
    esac

    local flag_limit=""
    [ "$limit" != "full" ] && flag_limit="--limit $limit"
    [ -n "$SMOKE" ] && flag_limit="--limit 1"    # 1 doc (1/subtask for groups)

    local path_runner="$FOLDER_RESULTS/${tag}__runner.json"
    local port=$(( PORT_BASE + idx ))

    echo "[gpu$gpu] START $tag (len=$len nshot=$nshot limit=$limit mode=$mode) $(date '+%F %T')"
    if [ -n "$DRY_RUN" ]; then
        echo "  DRY: CUDA_VISIBLE_DEVICES=$gpu HF_ALLOW_CODE_EVAL=$allow_code accelerate launch --num_processes=1 --main_process_port $port run_benchmark_main.py --tasks $task $flag_limit --model test --batch_size 1 --num_fewshot $nshot --device cuda $flag_unsafe --output_path $FOLDER_RESULTS/$tag --model_args \"id_model=$ID_MODEL,...runner=$RUNNER,num_blocks=$num_blocks$args_extra\""
        touch "$FOLDER_RESULTS/$tag.done"
        return 0
    fi

    CUDA_VISIBLE_DEVICES="$gpu" HF_ALLOW_CODE_EVAL="$allow_code" \
    accelerate launch --num_processes=1 --main_process_port "$port" run_benchmark_main.py \
        --tasks "$task" $flag_limit --model test --batch_size 1 \
        --num_fewshot "$nshot" --device cuda $flag_unsafe \
        --output_path "$FOLDER_RESULTS/$tag" \
        --model_args "id_model=$ID_MODEL,size_batch=1,len_target=$len,num_blocks=$num_blocks,num_unmask_per_step=1,id_mask=$ID_MASK,runner=$RUNNER,path_report=$path_runner$args_extra" \
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
        IFS=':' read -r thread task _rest <<< "$spec"
        local tag="${thread}__${task}"
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
# (proves scoring deps work: strict/flexible, math_verify, pass@1, bleu_acc)
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
    print(f'  {"OK  " if done else "FAIL"} {tag:42s} {metrics}')
PYEOF
fi
echo "summary: python build_bench_html.py --results $FOLDER_RESULTS"
