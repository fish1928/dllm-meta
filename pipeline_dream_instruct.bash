#!/bin/bash
#################################################
# OVERNIGHT PIPELINE -- dream_instruct, end to end:
#
#   stage 0  retire old dream_instruct collections + train splits
#   stage 1  collect oracles for ALL 6 benchmarks, LIMIT=500, full compute:
#              gsm8k          official 4-shot CoT prompt (in chat template)
#              everything else PLAIN (no chat template) -- including minerva
#   stage 2  print every oracle ceiling (eval_summary)
#   stage 3  head-split train sets (tail 10%)
#   stage 4  train the 4 routers (clean/policy/aged/age), H_TRAIN=8, layers 28
#   stage 5  conf/margin sweep, 4 arms x 6 tasks, LIMIT=100,
#            h=5 at inference, Kr=16, Kp=64  (self-contained: does NOT use
#            run_bench_confmargin.bash, so a concurrent final on the other
#            card keeps its script untouched)
#   stage 6  build the HTML summary
#
# Resume-safe: finished collections (eval_summary.json), existing bundles,
# and finished sweep runs (__runner.json) are skipped -- rerun the script to
# continue after any failure. A failed stage prints [FAIL]/[ABORT] banners;
# stage 4 refuses to train if any training-task folder is missing.
#
# NOTE Kp=64: dream threads measured healthy at Kp=16 and collapsed at Kp=96;
# 64 is between and UNTESTED -- if gsm8k comes in far under its ~0.6 ceiling,
# rerun stage 5 with KP=16 (env below) before blaming the router.
#
# Usage:
#   DEVICE=cuda:0 nohup bash pipeline_dream_instruct.bash > pipeline_di.log 2>&1 &
#   DRY_RUN=1 bash pipeline_dream_instruct.bash        # print commands only
#################################################

set -u

DEVICE=${DEVICE:-cuda:0}
DRY_RUN=${DRY_RUN:-}
LIMIT_COLLECT=${LIMIT_COLLECT:-500}
LIMIT_SWEEP=${LIMIT_SWEEP:-100}
H_TRAIN=${H_TRAIN:-8}
H_INFER=${H_INFER:-5}
KR=${KR:-16}
KP=${KP:-64}
STAMP=${STAMP:-$(date +%m%d)}
RESULTS=${RESULTS:-results_clean_dream_instruct_${STAMP}}
RETIRED="stats_retired_di_${STAMP}"

# task:len:mockup_tag:mode   mode: official | plain | code (plain + fence + unsafe)
COLLECT_SPECS=(
    "gsm8k:256:0shot:official"
    "bbh:256:3shot:plain"
    "truthfulqa_gen:256:0shot:plain"
    "minerva_math:512:4shot:plain"
    "humaneval:512:0shot:code"
    "mbpp:512:3shot:code"
)
TASKS_TRAIN=(gsm8k minerva_math bbh humaneval truthfulqa_gen)    # training mix (no mbpp, matches the other threads)
ARMS=(cm_clean cm_policy cm_aged cm_age)

banner () { echo; echo "########## [$(date '+%F %T')] $1 ##########"; }
run () { if [ -n "$DRY_RUN" ]; then echo "  DRY: $*"; else "$@"; fi }

# ============ stage 0: retire old data ============
banner "stage 0: retire old dream_instruct collections -> $RETIRED"
mkdir -p "$RETIRED"
for entry in "${COLLECT_SPECS[@]}"; do
    task="${entry%%:*}"
    if [ -d "stats_oracle/dream_instruct_${task}_b1" ] \
            && [ ! -f "stats_oracle/dream_instruct_${task}_b1/.pipeline_${STAMP}" ]; then
        run mv "stats_oracle/dream_instruct_${task}_b1" "$RETIRED/dream_instruct_${task}_b1"
    fi
    run rm -rf "stats_train/dream_instruct_${task}_b1"
done

# ============ stage 1: collect (full compute, LIMIT=500) ============
for entry in "${COLLECT_SPECS[@]}"; do
    IFS=':' read -r task len tag mode <<< "$entry"
    folder="stats_oracle/dream_instruct_${task}_b1"
    if [ -f "$folder/eval_summary.json" ]; then
        banner "collect $task: already complete, skipping"
        continue
    fi
    mockup="benchmark_mockup/mockup_${task}_${tag}_p100.csv"
    if [ ! -f "$mockup" ]; then
        echo "[FAIL] mockup missing: $mockup -- $task skipped (training gate will catch it)"
        continue
    fi
    flags="--plain_prompt"
    [ "$mode" = "official" ] && flags="--use_official_gsm8k_prompt"
    banner "collect $task (len=$len, $mode, limit=$LIMIT_COLLECT)"
    run python -u run_collect_metrics_dream_instruct.py \
        --path_mockup "$mockup" --folder_output "$folder" \
        --len_target "$len" --num_blocks 1 --device "$DEVICE" \
        --limit "$LIMIT_COLLECT" $flags \
        || echo "[FAIL] collection $task"
    [ -z "$DRY_RUN" ] && [ -d "$folder" ] && touch "$folder/.pipeline_${STAMP}"
done

# ============ stage 2: ceilings ============
banner "stage 2: oracle ceilings"
for entry in "${COLLECT_SPECS[@]}"; do
    task="${entry%%:*}"
    summary="stats_oracle/dream_instruct_${task}_b1/eval_summary.json"
    if [ -f "$summary" ]; then
        python -c "import json; d=json.load(open('$summary')); print('$task:', {k: {'acc': v.get('accuracy'), 'mv': v.get('mean_math_verify'), 'n': v.get('n')} for k, v in d.items() if isinstance(v, dict)})" || true
    else
        echo "$task: NO SUMMARY (collection failed?)"
    fi
done

# ============ stage 3: split ============
for entry in "${COLLECT_SPECS[@]}"; do
    task="${entry%%:*}"
    src="stats_oracle/dream_instruct_${task}_b1"
    dst="stats_train/dream_instruct_${task}_b1"
    if [ ! -f "$src/eval_summary.json" ]; then
        echo "[skip] split $task: collection incomplete"
        continue
    fi
    banner "split $task"
    run python make_train_split.py --folder_src "$src" --folder_dst "$dst" \
        || echo "[FAIL] split $task"
done

# ============ stage 4: train the 4 routers (H=$H_TRAIN) ============
banner "stage 4: gate + train"
missing=""
for task in "${TASKS_TRAIN[@]}"; do
    [ -d "stats_train/dream_instruct_${task}_b1" ] || missing="$missing $task"
done
if [ -n "$missing" ] && [ -z "$DRY_RUN" ]; then
    echo "[ABORT] training folders missing:$missing -- fix collections and rerun the pipeline"
    exit 1
fi
run rm -f routers_e2e/dream_instruct__cm_clean.pt routers_e2e/dream_instruct__cm_clean.json \
        routers_e2e/dream_instruct__cm_policy.pt routers_e2e/dream_instruct__cm_policy.json \
        routers_e2e/dream_instruct__cm_aged.pt routers_e2e/dream_instruct__cm_aged.json \
        routers_e2e/dream_instruct__cm_age.pt routers_e2e/dream_instruct__cm_age.json
if [ -n "$DRY_RUN" ]; then
    echo "  DRY: FOLDER_TRAIN=stats_train THREAD=dream_instruct NUM_LAYERS=28 H=$H_TRAIN DEVICE=$DEVICE python -u ablation_tests/train_e2e_confmargin.py"
else
    FOLDER_TRAIN=stats_train THREAD=dream_instruct NUM_LAYERS=28 H="$H_TRAIN" DEVICE="$DEVICE" \
        python -u ablation_tests/train_e2e_confmargin.py || { echo "[ABORT] training failed"; exit 1; }
fi

# ============ stage 5: sweep (h=$H_INFER, Kr=$KR, Kp=$KP) ============
banner "stage 5: conf/margin sweep -> $RESULTS"
mkdir -p "$RESULTS"
port=12500
for arm in "${ARMS[@]}"; do
    path_pt="routers_e2e/dream_instruct__${arm}.pt"
    if [ ! -f "$path_pt" ]; then
        echo "[skip] $arm: bundle missing"
        continue
    fi
    for entry in "${COLLECT_SPECS[@]}"; do
        IFS=':' read -r task len tag mode <<< "$entry"
        nshot="${tag%shot}"
        tag_run="${arm}__${task}"
        path_runner="$RESULTS/${tag_run}__runner.json"
        if [ -f "$path_runner" ]; then
            echo "[skip] $tag_run: report exists"
            continue
        fi

        args_extra=""
        flag_unsafe=""
        allow_code=""
        case "$mode" in
            official) args_extra=",use_official_gsm8k_prompt=True"; nshot=0 ;;
            code)     args_extra=",stop_at_code_fence=True"; flag_unsafe="--confirm_run_unsafe_code"; allow_code="1" ;;
        esac

        limit_task="$LIMIT_SWEEP"
        case "$task" in
            minerva_math) limit_task=$(( (LIMIT_SWEEP + 6) / 7 )) ;;
            bbh)          limit_task=$(( (LIMIT_SWEEP + 26) / 27 )) ;;
            humaneval)    [ "$LIMIT_SWEEP" -gt 148 ] && limit_task=148 ;;
        esac

        port=$((port + 1))
        banner "sweep $tag_run (len=$len, nshot=$nshot, mode=$mode, limit=$limit_task/subtask)"
        if [ -n "$DRY_RUN" ]; then
            echo "  DRY: HF_ALLOW_CODE_EVAL=$allow_code accelerate launch --num_processes=1 --main_process_port $port run_benchmark_main.py --tasks $task --limit $limit_task --num_fewshot $nshot $flag_unsafe --model_args \"...h=$H_INFER,Kr=$KR,Kp=$KP$args_extra...\""
            continue
        fi
        HF_ALLOW_CODE_EVAL="$allow_code" \
        accelerate launch --num_processes=1 --main_process_port "$port" run_benchmark_main.py \
            --tasks "$task" --limit "$limit_task" --model test --batch_size 1 \
            --num_fewshot "$nshot" --device "$DEVICE" $flag_unsafe \
            --output_path "$RESULTS/$tag_run" \
            --model_args "id_model=Dream-org/Dream-v0-Instruct-7B,size_batch=1,len_target=$len,num_blocks=1,num_unmask_per_step=1,id_mask=151666,step_refresh_remainder=$KR,step_refresh_remainder_prompt=$KP,select_only_in_h=True,runner=run_dream_instruct_mlp,h=$H_INFER,path_router=$path_pt,path_report=$path_runner$args_extra" \
            || echo "[FAIL] $tag_run"
    done
done

# ============ stage 6: html ============
banner "stage 6: summary"
run python build_bench_html.py --results "$RESULTS" --out "bench_summary_${RESULTS}.html"

banner "PIPELINE DONE -> $RESULTS ; bench_summary_${RESULTS}.html"
