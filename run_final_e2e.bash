#!/bin/bash
#################################################
# FINAL full-suite e2e for the base threads at the head-split budget
# (LIMIT=450: e2e docs 0..449, disjoint from the routers' training docs
# 450..499; humaneval capped at 148 by the same scheme).
#
# One chosen bundle per thread (the sweep winners):
#   llada_base -> routers_e2e/llada_base__cm_clean.pt   (Kr=16, Kp=96)
#   dream_base -> routers_e2e/dream_base__cm_age.pt     (Kr=16, Kp=16 --
#                 dream is prompt-staleness sensitive; Kp=96 collapses it)
#
# All six canonical tasks, canonical gen lengths, per-subtask LIMIT division.
# h is read from the bundle spec. Resume-safe per run (delete a run's
# __runner.json to redo it). Results: results_final_<THREAD>/.
#
# Usage:
#   THREAD=llada_base DEVICE=cuda:0 nohup bash run_final_e2e.bash > final_llada_base.log 2>&1 &
#   THREAD=dream_base  DEVICE=cuda:1 nohup bash run_final_e2e.bash > final_dream_base.log 2>&1 &
#   DRY_RUN=1 THREAD=dream_base bash run_final_e2e.bash     # print commands only
#   FILTER_TASK=gsm8k ... bash run_final_e2e.bash           # subset
# Summary afterwards: python build_bench_html.py --results results_final_<THREAD>
#################################################

set -u

THREAD=${THREAD:-llada_base}
DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-450}
DRY_RUN=${DRY_RUN:-}
FILTER_TASK=${FILTER_TASK:-}

FOLDER_ROUTERS=${FOLDER_ROUTERS:-routers_e2e}
FOLDER_RESULTS=${FOLDER_RESULTS:-results_final_${THREAD}}
FOLDER_LOGS=${FOLDER_LOGS:-$FOLDER_RESULTS/logs}

KR=${KR:-16}
NUM_UNMASK=1

case "$THREAD" in
    llada_base)
        ID_MODEL="GSAI-ML/LLaDA-8B-Base";      ID_MASK=126336
        RUNNER=${RUNNER:-run_llada_semi_mlp_v2}
        ROUTER=${ROUTER:-llada_base__cm_clean}
        KP=${KP:-96} ;;
    dream_base)
        ID_MODEL="Dream-org/Dream-v0-Base-7B"; ID_MASK=151666
        RUNNER=${RUNNER:-run_dream_semi_mlp}
        ROUTER=${ROUTER:-dream_base__cm_age}
        KP=${KP:-16} ;;
    *) echo "run_final_e2e.bash covers the base threads only (llada_base | dream_base)"; exit 1 ;;
esac

# task:len_target:num_fewshot:needs_unsafe_code -- canonical spec
BENCHMARKS=(
    "gsm8k:256:5:no"
    "minerva_math:512:4:no"
    "bbh:256:3:no"
    "mbpp:512:3:yes"
    "humaneval:512:0:yes"
    "truthfulqa_gen:256:0:no"
)

path_pt="$FOLDER_ROUTERS/$ROUTER.pt"
path_spec="${path_pt%.pt}.json"
if [ ! -f "$path_pt" ] || [ ! -f "$path_spec" ]; then
    echo "ABORT: bundle $path_pt (+.json) missing"; exit 1
fi

if [ -n "${H_BUNDLE:-}" ]; then
    :    # env override wins (e.g. bundle trained at H=8, inference at h=5)
else
H_BUNDLE=$(python - "$path_spec" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1])).get('h', 5))
PYEOF
)
fi

mkdir -p "$FOLDER_RESULTS" "$FOLDER_LOGS"
echo "[final] thread=$THREAD router=$ROUTER h=$H_BUNDLE Kr=$KR Kp=$KP limit=$LIMIT -> $FOLDER_RESULTS"

num_run=0
num_skip=0
num_fail=0

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r task len_target nshot unsafe <<< "$entry"

    if [ -n "$FILTER_TASK" ] && [[ "$task" != *"$FILTER_TASK"* ]]; then
        continue
    fi

    tag="${ROUTER}__${task}"
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

    # LIMIT is a per-TASK budget; lm_eval applies --limit per SUBTASK.
    # humaneval: head-split cap (train docs 148..163 -> e2e never past 148)
    limit_task="$LIMIT"
    case "$task" in
        minerva_math) limit_task=$(( (LIMIT + 6) / 7 )) ;;
        bbh)          limit_task=$(( (LIMIT + 26) / 27 )) ;;
        humaneval)    [ "$LIMIT" -gt 148 ] && limit_task=148 ;;
    esac

    args_extra=""
    if [ "$task" = "humaneval" ] || [ "$task" = "mbpp" ]; then
        args_extra=",stop_at_code_fence=True"    # cut at markdown fence (Dream tail rescue)
    fi

    model_args="id_model=$ID_MODEL,size_batch=1,len_target=$len_target,num_blocks=1,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,step_refresh_remainder=$KR,step_refresh_remainder_prompt=$KP,select_only_in_h=True,runner=$RUNNER,h=$H_BUNDLE,path_router=$path_pt,path_report=$path_runner$args_extra"

    echo "=== [$THREAD] $tag (len=$len_target, nshot=$nshot, limit=$limit_task/subtask) ==="

    if [ -n "$DRY_RUN" ]; then
        echo "  HF_ALLOW_CODE_EVAL=$allow_code_eval accelerate launch --num_processes=1 run_benchmark_main.py --tasks $task --limit $limit_task --model test --batch_size 1 --num_fewshot $nshot --device $DEVICE $flag_unsafe --output_path $FOLDER_RESULTS/$tag --model_args \"$model_args\""
        continue
    fi

    num_run=$((num_run + 1))
    HF_ALLOW_CODE_EVAL="$allow_code_eval" \
    accelerate launch --num_processes=1 run_benchmark_main.py \
        --tasks "$task" --limit "$limit_task" --model test --batch_size 1 \
        --num_fewshot "$nshot" --device "$DEVICE" $flag_unsafe \
        --output_path "$FOLDER_RESULTS/$tag" \
        --model_args "$model_args" \
        2>&1 | tee "$FOLDER_LOGS/${tag}.log"

    if [ ! -f "$path_runner" ]; then
        echo "FAILED: $tag (see $FOLDER_LOGS/${tag}.log)"
        num_fail=$((num_fail + 1))
    fi
done

echo
echo "[$THREAD] final e2e complete: $num_run launched, $num_skip skipped, $num_fail failed -> $FOLDER_RESULTS"
echo "summary: python build_bench_html.py --results $FOLDER_RESULTS"
