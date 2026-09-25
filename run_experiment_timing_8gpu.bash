#!/bin/bash
#################################################
# TIMING TABLE -- wall-clock per doc for every method x thread x gen length.
#
#   methods:  dense | ours | dllmcache | fastdllm | d2cache
#   threads:  llada_base llada_instruct dream_base dream_instruct
#   task:     gsm8k 5-shot PLAIN for ALL threads (timing wants uniform prompt
#             geometry; accuracy is not the point of these runs)
#   lengths:  gen 256 and gen 512   |   --limit 5 (5 docs per cell)
#
# 40 jobs on the 8-GPU work queue (claims + .done + resume, like the
# campaigns). Timing comes from the per-sample runner reports (generation
# only, model load excluded). Aggregate + analytic TFLOPs:
#   python run_experiment_timing_report.py --folder results_experiment_timing
#
# Settings mirror the accuracy campaigns: ours Kp=64/Kr=8/h=8 (H env),
# dllmcache rho=0.25/Kr=8/Kp=64, fastdllm nb=len/32, d2cache k=32/p=0.1.
#
# Usage:
#   nohup bash run_experiment_timing_8gpu.bash > timing.log 2>&1 &
#   GPU_LIST="0 1 2" DRY_RUN=1 ... as usual
#################################################

set -u

GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
DRY_RUN=${DRY_RUN:-}
FOLDER_RESULTS=${FOLDER_RESULTS:-results_experiment_timing}
FOLDER_LOGS="$FOLDER_RESULTS/logs"
FOLDER_CLAIMS="$FOLDER_RESULTS/claims"
FOLDER_ROUTERS=${FOLDER_ROUTERS:-routers_e2e}
PORT_BASE=${PORT_BASE:-18000}
N_DOCS=${N_DOCS:-5}

KP=${KP:-64}
KR=${KR:-8}
H=${H:-8}

METHODS=${METHODS:-"dense ours dllmcache fastdllm d2cache"}
THREADS=${THREADS:-"llada_base llada_instruct dream_base dream_instruct"}
LENS=${LENS:-"256 512"}

mkdir -p "$FOLDER_RESULTS" "$FOLDER_LOGS" "$FOLDER_CLAIMS"

thread_params () {    # ID_MODEL, ID_MASK, FAMILY, RUNNER_DENSE, RUNNER_OURS, ROUTER
    case "$1" in
        llada_base)     ID_MODEL="GSAI-ML/LLaDA-8B-Base";          ID_MASK=126336; FAMILY=llada; RUNNER_DENSE=run_llada_semi;     RUNNER_OURS=run_llada_semi_mlp_v2;  ROUTER="llada_base__cm_clean" ;;
        llada_instruct) ID_MODEL="GSAI-ML/LLaDA-8B-Instruct";      ID_MASK=126336; FAMILY=llada; RUNNER_DENSE=run_llada_instruct; RUNNER_OURS=run_llada_instruct_mlp; ROUTER="llada_instruct__cm_clean" ;;
        dream_base)     ID_MODEL="Dream-org/Dream-v0-Base-7B";     ID_MASK=151666; FAMILY=dream; RUNNER_DENSE=run_dream_semi;     RUNNER_OURS=run_dream_semi_mlp;     ROUTER="dream_base__cm_age" ;;
        dream_instruct) ID_MODEL="Dream-org/Dream-v0-Instruct-7B"; ID_MASK=151666; FAMILY=dream; RUNNER_DENSE=run_dream_instruct; RUNNER_OURS=run_dream_instruct_mlp; ROUTER="dream_instruct__cm_clean" ;;
    esac
}

method_params () {    # RUNNER, NUM_BLOCKS, ARGS_METHOD for method=$1 thread=$2 len=$3
    thread_params "$2"
    local blocks_instr=1
    [ "$2" = "llada_instruct" ] && blocks_instr=$(( $3 / 32 ))
    case "$1" in
        dense)
            RUNNER=$RUNNER_DENSE; NUM_BLOCKS=$blocks_instr; ARGS_METHOD="" ;;
        ours)
            RUNNER=$RUNNER_OURS; NUM_BLOCKS=$blocks_instr
            ARGS_METHOD=",step_refresh_remainder=$KR,step_refresh_remainder_prompt=$KP,select_only_in_h=True,h=$H,path_router=$FOLDER_ROUTERS/$ROUTER.pt" ;;
        dllmcache)
            RUNNER="run_${FAMILY}_dllm_cache"; NUM_BLOCKS=1
            ARGS_METHOD=",dllmc_v_rate=0.25,step_refresh_remainder=$KR,step_refresh_remainder_prompt=$KP" ;;
        fastdllm)
            RUNNER="run_${FAMILY}_fastdllm"; NUM_BLOCKS=$(( $3 / 32 ))
            ARGS_METHOD="" ;;
        d2cache)
            RUNNER="run_${FAMILY}_d2cache"; NUM_BLOCKS=1
            ARGS_METHOD=",d2c_k=32,d2c_sigma=10.0,d2c_rollout_p=0.1,d2c_conf_mode=live" ;;
    esac
}

JOBS=()
for len in $LENS; do
    for method in $METHODS; do
        for thread in $THREADS; do
            JOBS+=( "${method}:${thread}:${len}" )
        done
    done
done

for claim in "$FOLDER_CLAIMS"/*/; do
    [ -d "$claim" ] || continue
    tag=$(basename "$claim")
    if [ ! -f "$FOLDER_RESULTS/$tag.done" ]; then
        rm -rf "$claim"; echo "[startup] cleared stale claim: $tag"
    fi
done

run_job () {
    local gpu="$1" idx="$2" spec="$3"
    IFS=':' read -r method thread len <<< "$spec"
    local tag="${method}__${thread}__g${len}"

    thread_params "$thread"
    method_params "$method" "$thread" "$len"

    local path_runner="$FOLDER_RESULTS/${tag}__runner.json"
    local port=$(( PORT_BASE + idx ))
    local model_args="id_model=$ID_MODEL,size_batch=1,len_target=$len,num_blocks=$NUM_BLOCKS,num_unmask_per_step=1,id_mask=$ID_MASK,runner=$RUNNER,path_report=$path_runner$ARGS_METHOD"

    echo "[gpu$gpu] START $tag $(date '+%F %T')"
    if [ -n "$DRY_RUN" ]; then
        echo "  DRY: CUDA_VISIBLE_DEVICES=$gpu accelerate launch --num_processes=1 --main_process_port $port run_benchmark_main.py --tasks gsm8k --limit $N_DOCS --num_fewshot 5 --model_args \"$model_args\""
        return 0
    fi

    CUDA_VISIBLE_DEVICES="$gpu" \
    accelerate launch --num_processes=1 --main_process_port "$port" run_benchmark_main.py \
        --tasks gsm8k --limit "$N_DOCS" --model test --batch_size 1 \
        --num_fewshot 5 --device cuda \
        --output_path "$FOLDER_RESULTS/$tag" \
        --model_args "$model_args" \
        > "$FOLDER_LOGS/${tag}.log" 2>&1
    local status=$?

    if [ $status -eq 0 ] && [ -f "$path_runner" ]; then
        touch "$FOLDER_RESULTS/$tag.done"
        echo "[gpu$gpu] DONE  $tag $(date '+%F %T')"
    else
        rm -rf "$FOLDER_CLAIMS/$tag"
        echo "[gpu$gpu] FAILED $tag (exit $status) -- see $FOLDER_LOGS/${tag}.log"
    fi
}

worker () {
    local gpu="$1"
    local idx=0
    for spec in "${JOBS[@]}"; do
        idx=$(( idx + 1 ))
        IFS=':' read -r method thread len <<< "$spec"
        local tag="${method}__${thread}__g${len}"
        [ -f "$FOLDER_RESULTS/$tag.done" ] && continue
        if mkdir "$FOLDER_CLAIMS/$tag" 2>/dev/null; then
            run_job "$gpu" "$idx" "$spec"
        fi
    done
    echo "[gpu$gpu] worker finished"
}

for gpu in $GPU_LIST; do
    worker "$gpu" &
done
wait

n_done=$(ls "$FOLDER_RESULTS"/*.done 2>/dev/null | wc -l)
echo "ALL WORKERS DONE: $n_done/${#JOBS[@]} -> $FOLDER_RESULTS"
echo "report: python run_experiment_timing_report.py --folder $FOLDER_RESULTS"
