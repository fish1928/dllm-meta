#!/bin/bash
#################################################
# OUR METHOD (learned MLP router) on 8 GPUs -- all four threads, same
# benchmark protocol, limits, and queue machinery as run_full_denoise_8gpu /
# run_baselines_8gpu.
#
#   routers:  llada_base / llada_instruct -> <thread>__cm_clean
#             dream_base / dream_instruct -> <thread>__cm_age
#             (bundles + json sidecars under routers_e2e/; h=8 for all runs
#             via the H_BUNDLE default)
#   clocks:   UNIFIED KP=64 KR=8 across all threads (the same setting the
#             dllm-cache baseline runs at).
#             !! deviation from the thread-tuned finals: dream ran its final
#             e2e at Kp=16 because Kp=96 collapsed it -- Kp=64 on the dream
#             threads is untested territory; if dream gsm8k comes out far
#             below its ~0.6 near-dense line, rerun the dream threads with
#             KP=16 into a fresh FOLDER_RESULTS and report that column.
#   limits:   gsm8k FULL (1319) | minerva 500/subtask (x7) | bbh 125/subtask (x27)
#             mbpp FULL (500)   | humaneval FULL (164)     | truthfulqa FULL (817)
#   protocol: instruct gsm8k = official CoT prompt (nshot 0); ALL other
#             instruct cells template-free (incl. dream_instruct minerva --
#             flat rule, matching the full-denoise and baseline campaigns);
#             code tasks + fence cut; llada_instruct num_blocks=len/32,
#             all others one block; select_only_in_h=True, 1 token/step.
#
# RESUME: .done sentinel after clean exit WITH a runner report; stale claims
# cleared at startup. SMOKE=1 -> --limit 1 into results_smoke_ours.
#
# Usage:
#   SMOKE=1 nohup bash run_ours_8gpu.bash > ours_smoke.log 2>&1 &
#   nohup bash run_ours_8gpu.bash > ours.log 2>&1 &
#   THREADS="dream_base dream_instruct" KP=16 FOLDER_RESULTS=results_ours_kp16 bash run_ours_8gpu.bash
#   DRY_RUN=1 bash run_ours_8gpu.bash
#################################################

set -u

GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
DRY_RUN=${DRY_RUN:-}
SMOKE=${SMOKE:-}
if [ -n "$SMOKE" ]; then
    FOLDER_RESULTS=${FOLDER_RESULTS:-results_smoke_ours}
else
    FOLDER_RESULTS=${FOLDER_RESULTS:-results_ours}
fi
FOLDER_LOGS="$FOLDER_RESULTS/logs"
FOLDER_CLAIMS="$FOLDER_RESULTS/claims"
FOLDER_ROUTERS=${FOLDER_ROUTERS:-routers_e2e}
PORT_BASE=${PORT_BASE:-15000}

THREADS=${THREADS:-"llada_base llada_instruct dream_base dream_instruct"}
TASKS=${TASKS:-"minerva_math bbh gsm8k truthfulqa_gen mbpp humaneval"}

KP=${KP:-64}            # step_refresh_remainder_prompt -- unified with the baselines
KR=${KR:-8}             # step_refresh_remainder
H_BUNDLE=${H_BUNDLE:-8}    # h=8 for ALL runs (campaign decision); set H_BUNDLE=""
                           # to fall back to each bundle's spec value
NUM_UNMASK=1

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

thread_params () {    # sets ID_MODEL, ID_MASK, RUNNER, ROUTER, IS_INSTRUCT for $1
    case "$1" in
        llada_base)     ID_MODEL="GSAI-ML/LLaDA-8B-Base";          ID_MASK=126336; RUNNER=run_llada_semi_mlp_v2;  ROUTER="llada_base__cm_clean";     IS_INSTRUCT= ;;
        llada_instruct) ID_MODEL="GSAI-ML/LLaDA-8B-Instruct";      ID_MASK=126336; RUNNER=run_llada_instruct_mlp; ROUTER="llada_instruct__cm_clean"; IS_INSTRUCT=1 ;;
        dream_base)     ID_MODEL="Dream-org/Dream-v0-Base-7B";     ID_MASK=151666; RUNNER=run_dream_semi_mlp;     ROUTER="dream_base__cm_age";       IS_INSTRUCT= ;;
        dream_instruct) ID_MODEL="Dream-org/Dream-v0-Instruct-7B"; ID_MASK=151666; RUNNER=run_dream_instruct_mlp; ROUTER="dream_instruct__cm_age";   IS_INSTRUCT=1 ;;
    esac
}

spec_field () {    # spec_field <path_json> <field> [default]
    python - "$1" "$2" "${3:-}" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))
PYEOF
}

# fail fast: every thread's bundle must exist before anything is claimed
missing=0
for thread in $THREADS; do
    thread_params "$thread"
    for f in "$FOLDER_ROUTERS/$ROUTER.pt" "$FOLDER_ROUTERS/$ROUTER.json"; do
        if [ ! -f "$f" ]; then
            echo "MISSING bundle file: $f (thread $thread)"
            missing=1
        fi
    done
done
[ "$missing" = "1" ] && { echo "ABORT: train/copy the router bundles first"; exit 1; }

# largest-first job list: tasks outer (minerva/bbh tiers first), threads inner
JOBS=()
for task in $TASKS; do
    for thread in $THREADS; do
        JOBS+=( "${thread}:${task}" )
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
    IFS=':' read -r thread task <<< "$spec"
    local tag="ours__${thread}__${task}"

    task_params "$task"
    thread_params "$thread"

    local path_pt="$FOLDER_ROUTERS/$ROUTER.pt"
    local path_spec="$FOLDER_ROUTERS/$ROUTER.json"
    local h_run
    if [ -n "$H_BUNDLE" ]; then h_run="$H_BUNDLE"; else h_run=$(spec_field "$path_spec" h 5); fi

    local num_blocks=1
    [ "$thread" = "llada_instruct" ] && num_blocks=$(( LEN / 32 ))

    local mode="$MODE" nshot="$NSHOT"
    local args_extra="" flag_unsafe="" allow_code=""
    if [ -n "$IS_INSTRUCT" ] && [ "$task" = "gsm8k" ]; then
        mode=official; nshot=0
    fi
    case "$mode" in
        official) args_extra=",use_official_gsm8k_prompt=True" ;;
        code)     args_extra=",stop_at_code_fence=True"; flag_unsafe="--confirm_run_unsafe_code"; allow_code="1" ;;
    esac

    local flag_limit=""
    [ "$LIMIT" != "full" ] && flag_limit="--limit $LIMIT"
    [ -n "$SMOKE" ] && flag_limit="--limit 1"    # 1 doc (1/subtask for groups)

    local path_runner="$FOLDER_RESULTS/${tag}__runner.json"
    local port=$(( PORT_BASE + idx ))

    local model_args="id_model=$ID_MODEL,size_batch=1,len_target=$LEN,num_blocks=$num_blocks,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,step_refresh_remainder=$KR,step_refresh_remainder_prompt=$KP,select_only_in_h=True,runner=$RUNNER,h=$h_run,path_router=$path_pt,path_report=$path_runner$args_extra"

    echo "[gpu$gpu] START $tag (len=$LEN blocks=$num_blocks nshot=$nshot h=$h_run limit=$LIMIT mode=$mode) $(date '+%F %T')"
    if [ -n "$DRY_RUN" ]; then
        echo "  DRY: CUDA_VISIBLE_DEVICES=$gpu HF_ALLOW_CODE_EVAL=$allow_code accelerate launch --num_processes=1 --main_process_port $port run_benchmark_main.py --tasks $task $flag_limit --num_fewshot $nshot $flag_unsafe --model_args \"$model_args\""
        return 0    # no .done in DRY_RUN: a real launch afterwards must not skip
    fi

    CUDA_VISIBLE_DEVICES="$gpu" HF_ALLOW_CODE_EVAL="$allow_code" \
    accelerate launch --num_processes=1 --main_process_port "$port" run_benchmark_main.py \
        --tasks "$task" $flag_limit --model test --batch_size 1 \
        --num_fewshot "$nshot" --device cuda $flag_unsafe \
        --output_path "$FOLDER_RESULTS/$tag" \
        --model_args "$model_args" \
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
        IFS=':' read -r thread task <<< "$spec"
        local tag="ours__${thread}__${task}"
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
    print(f'  {"OK  " if done else "FAIL"} {tag:52s} {metrics}')
PYEOF
fi
echo "summary: python build_bench_html.py --results $FOLDER_RESULTS"
