#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <STAGE> <DATA_DIR> <WORK_DIR> [ROUTER_MODULE]" >&2
  exit 2
fi

STAGE="$1"
DATA_DIR="$2"
WORK_DIR="$3"
ROUTER_MODULE="${4:-router_llada_v2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/router_ablation_runner.py"

COMMON_ARGS=(
  --stage "$STAGE"
  --folder-data "$DATA_DIR"
  --work-dir "$WORK_DIR"
  --router-module "$ROUTER_MODULE"
  --h 5
  --size-block 64
  --num-layers 32
  --screen-epochs 10
  --screen-patience 3
  --full-epochs 30
  --full-patience 5
  --promote-count 12
  --full-seeds 233 239 251
  --final-seeds 233 239 251 257 263
)

python "$RUNNER" make-plan "${COMMON_ARGS[@]}"
PLAN="$WORK_DIR/plans/stage_${STAGE^^}.json"

python "$RUNNER" run-plan \
  --plan "$PLAN" \
  --device cuda:0 \
  --num-shards 2 \
  --shard-index 0 &
PID0=$!

python "$RUNNER" run-plan \
  --plan "$PLAN" \
  --device cuda:1 \
  --num-shards 2 \
  --shard-index 1 &
PID1=$!

status=0
wait "$PID0" || status=$?
wait "$PID1" || status=$?

python "$RUNNER" summarize \
  --work-dir "$WORK_DIR" \
  --stages "${STAGE^^}" \
  --h 5

exit "$status"
