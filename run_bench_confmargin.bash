#!/bin/bash
#################################################
# Deciding e2e for the conf/margin question: the cm_* routers trained by
# ablation_tests/train_e2e_confmargin.py, identical decoding per thread,
# only the router bundle differs.
#
#   cm_clean    attn_last + geo                       (no conf/margin)
#   cm_policy   + conf/margin, policy-aged training
#   cm_aged     + conf/margin, random-aged training
#   cm_age      + conf/margin values WITH the true per-position age channel
#
# THREAD-aware (llada_base | llada_instruct | dream_base | dream_instruct):
#   - model id / mask id / runner per thread
#   - llada_instruct: block width 32 -> num_blocks = len_target/32 PER TASK;
#     gsm8k uses the official 0-shot CoT prompt; other tasks chat template
#   - dream_instruct: chat template, 0-shot gsm8k
#   - h is read from each bundle's json spec (matches how it was trained)
#   - bundles whose spec needs margin/age are SKIPPED on runners without the
#     online margin/age wiring (currently only the llada_base runners have it)
#
# NOTE Kp semantics: prompt refresh is decoupled everywhere now -- KP unset/0
# means the prompt is NEVER refreshed. KSURFIX (llada_instruct only) adds the
# suffix clock.
#
# Bundles come from: python ablation_tests/train_e2e_confmargin.py
#
# Usage:
#   THREAD=llada_instruct DEVICE=cuda:1 LIMIT=100 bash run_bench_confmargin.bash
#   THREAD=dream_base DEVICE=cuda:1 LIMIT=100 bash run_bench_confmargin.bash
#   DRY_RUN=1 THREAD=llada_instruct bash run_bench_confmargin.bash   # print commands only
#   FILTER_ROUTER=cm_clean FILTER_TASK=bbh ... bash run_bench_confmargin.bash
# Resume-safe per run (skips when the runner report exists).
#################################################

set -u

THREAD=${THREAD:-llada_base}
DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-150}
DRY_RUN=${DRY_RUN:-}

FOLDER_ROUTERS=${FOLDER_ROUTERS:-routers_e2e}
FOLDER_RESULTS=${FOLDER_RESULTS:-results_bench_confmargin_${THREAD}}
FOLDER_LOGS=${FOLDER_LOGS:-$FOLDER_RESULTS/logs}

FILTER_ROUTER=${FILTER_ROUTER:-}
FILTER_TASK=${FILTER_TASK:-}

KR=${KR:-16}        # step_refresh_remainder        (generation clock)
KP=${KP:-96}        # step_refresh_remainder_prompt (prompt clock; 0 = never)
KSURFIX=${KSURFIX:-}    # step_refresh_remainder_surfix (llada_instruct only; empty = off)
NUM_UNMASK=1

case "$THREAD" in
    llada_base)
        ID_MODEL="GSAI-ML/LLaDA-8B-Base";        ID_MASK=126336
        RUNNER=${RUNNER:-run_llada_semi_mlp_v2}; MARGIN_WIRED=1 ;;
    llada_instruct)
        ID_MODEL="GSAI-ML/LLaDA-8B-Instruct";    ID_MASK=126336
        RUNNER=${RUNNER:-run_llada_instruct_mlp}; MARGIN_WIRED=1 ;;
    dream_base)
        ID_MODEL="Dream-org/Dream-v0-Base-7B";   ID_MASK=151666
        RUNNER=${RUNNER:-run_dream_semi_mlp};    MARGIN_WIRED=1 ;;
    dream_instruct)
        ID_MODEL="Dream-org/Dream-v0-Instruct-7B"; ID_MASK=151666
        RUNNER=${RUNNER:-run_dream_instruct_mlp};  MARGIN_WIRED=1 ;;
    *) echo "unknown THREAD=$THREAD"; exit 1 ;;
esac

ROUTERS=(
    "${THREAD}__cm_clean"
    "${THREAD}__cm_policy"
    "${THREAD}__cm_aged"
    "${THREAD}__cm_age"
)

# task:len_target:num_fewshot:needs_unsafe_code -- canonical gen lengths.
# The deciding pair first; extend for the full suite. gsm8k few-shot is
# overridden to 0 for instruct threads below.
BENCHMARKS=(
    "gsm8k:256:5:no"
    "bbh:256:3:no"
    # "minerva_math:512:4:no"
    # "mbpp:512:3:yes"
    # "humaneval:512:0:yes"
    # "truthfulqa_gen:256:0:no"
)

mkdir -p "$FOLDER_RESULTS" "$FOLDER_LOGS"

spec_field () {    # spec_field <path_json> <field> [default]
    python - "$1" "$2" "${3:-}" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
value = spec.get(sys.argv[2], sys.argv[3])
print(value)
PYEOF
}

spec_needs_margin () {    # 1 when the bundle's features need the margin/age wiring
    python - "$1" <<'PYEOF'
import json, sys
features = json.load(open(sys.argv[1])).get('features', [])
print(1 if any(f in ('margin', 'conf_age', 'margin_age') for f in features) else 0)
PYEOF
}

num_run=0
num_skip=0
num_fail=0

for name in "${ROUTERS[@]}"; do
    path_pt="$FOLDER_ROUTERS/$name.pt"
    path_spec="${path_pt%.pt}.json"

    if [ -n "$FILTER_ROUTER" ] && [[ "$name" != *"$FILTER_ROUTER"* ]]; then
        continue
    fi

    if [ ! -f "$path_pt" ] || [ ! -f "$path_spec" ]; then
        echo "SKIP router $name: bundle or sidecar missing under $FOLDER_ROUTERS (train first: ablation_tests/train_e2e_confmargin.py)"
        num_skip=$((num_skip + 1))
        continue
    fi

    if [ "$MARGIN_WIRED" != "1" ] && [ "$(spec_needs_margin "$path_spec")" = "1" ]; then
        echo "SKIP router $name: spec needs online margin/age but runner $RUNNER is not wired for it yet"
        num_skip=$((num_skip + 1))
        continue
    fi

    H_BUNDLE=$(spec_field "$path_spec" h 5)

    for entry in "${BENCHMARKS[@]}"; do
        IFS=':' read -r task len_target nshot unsafe <<< "$entry"

        if [ -n "$FILTER_TASK" ] && [[ "$task" != *"$FILTER_TASK"* ]]; then
            continue
        fi

        # per-thread task protocol
        args_extra=""
        case "$THREAD" in
            llada_instruct)
                num_blocks=$(( len_target / 32 ))    # block width 32
                if [ "$task" = "gsm8k" ]; then
                    nshot=0
                    args_extra=",use_official_gsm8k_prompt=True"
                else
                    args_extra=",use_chat_template=True"
                fi
                if [ -n "$KSURFIX" ]; then
                    args_extra="$args_extra,step_refresh_remainder_surfix=$KSURFIX"
                fi
                ;;
            dream_instruct)
                num_blocks=1
                if [ "$task" = "gsm8k" ]; then
                    nshot=0
                    # official 4-shot CoT prompt (implies chat template) -- the
                    # 0-shot bare-chat protocol caps dream_instruct at ~0.37
                    # while the model is capable of ~0.8; oracle + router are
                    # collected/trained under this prompt too
                    args_extra=",use_official_gsm8k_prompt=True"
                else
                    args_extra=",use_chat_template=True"
                fi
                ;;
            *)
                num_blocks=1
                ;;
        esac

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

        model_args="id_model=$ID_MODEL,size_batch=1,len_target=$len_target,num_blocks=$num_blocks,num_unmask_per_step=$NUM_UNMASK,id_mask=$ID_MASK,step_refresh_remainder=$KR,step_refresh_remainder_prompt=$KP,select_only_in_h=True,runner=$RUNNER,h=$H_BUNDLE,path_router=$path_pt,path_report=$path_runner$args_extra"

        echo "=== [$THREAD] $tag (len=$len_target, blocks=$num_blocks, nshot=$nshot, h=$H_BUNDLE, limit=$limit_task/subtask) ==="

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
done

echo
echo "[$THREAD] conf/margin sweep complete: $num_run launched, $num_skip skipped, $num_fail failed -> $FOLDER_RESULTS"
