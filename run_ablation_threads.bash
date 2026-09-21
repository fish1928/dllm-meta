#!/bin/bash
#################################################
# Sequential fast ablation for the remaining threads on one card.
#
#   llada_instruct  (NUM_LAYERS=32)
#   dream_base      (NUM_LAYERS=28)
#   dream_instruct  (NUM_LAYERS=28)
#
# NUM_EPOCHS=4 for every search stage (final trainings use EPOCHS_FINAL=20
# inside the ablation script). Each thread writes its own report
# (ablation_test_report_fast_<THREAD>.json), bundles
# (routers_final/<THREAD>__*), and HTML (ablation_report_<THREAD>.html).
#
# Preflight before any training:
#   - prints the four staleness tracks (no / fresh / aged / policy
#     conf+margin) with their stage-A anchors, so coverage is visible
#   - lists missing oracle folders per thread (those datasets drop out)
#
# Resume-safe end to end (the ablation script skips finished runs), and a
# failed thread does not stop the next one.
#
# Usage:
#   DEVICE=cuda:1 nohup bash run_ablation_threads.bash > ablation_threads.log 2>&1 &
#   THREADS="dream_base" bash run_ablation_threads.bash      # subset
#################################################

set -u

DEVICE=${DEVICE:-cuda:1}
FOLDER_TRAIN=${FOLDER_TRAIN:-stats_train}
NUM_EPOCHS=${NUM_EPOCHS:-4}
NUM_BLOCKS=${NUM_BLOCKS:-1}
THREADS=${THREADS:-llada_instruct dream_base dream_instruct}

FOLDER_LOGS=logs_ablation
mkdir -p "$FOLDER_LOGS"

layers_for () {
    case "$1" in
        dream_base|dream_instruct) echo 28 ;;
        *)                         echo 32 ;;
    esac
}

# llada_instruct oracles were swept at b8/b16/b32 (no b1); its deployment
# convention is a constant block WIDTH of 32, which is _b8 for 256-length
# tasks and _b16 for 512-length ones -- BLOCK_SIZE mode picks per task.
# The one-block threads (dream_*, llada_base) keep plain _b1 selection.
block_size_for () {
    case "$1" in
        llada_instruct) echo 32 ;;
        *)              echo "" ;;
    esac
}

# every task any dataset group uses (mix + gsm8k + ifeval groups)
TASKS="gsm8k minerva_math bbh humaneval truthfulqa_gen ifeval"

echo "===== preflight: staleness-track coverage (stage A anchors) ====="
python - <<'EOF'
import sys
sys.path[:0] = ['ablation_tests', '.']
from ablation_test_fast import FEATURE_ANCHORS, track_of
groups = {}
for name, feats in FEATURE_ANCHORS.items():
    groups.setdefault(track_of(feats), []).append(name)
for track in ('clean', 'fresh', 'aged', 'policy'):
    anchors = groups.get(track, [])
    assert anchors, f'track {track} has NO anchors -- fix FEATURE_ANCHORS'
    print(f'  {track:7s} ({len(anchors)}): {", ".join(anchors)}')
EOF
if [ $? -ne 0 ]; then
    echo "preflight FAILED: a staleness track has no anchors"; exit 1
fi

echo
for thread in $THREADS; do
    block_size=$(block_size_for "$thread")
    missing=""
    for task in $TASKS; do
        if [ -n "$block_size" ]; then
            # BLOCK_SIZE mode: any b* collection may hold the right width;
            # the python side picks by inferred block width
            ls -d "$FOLDER_TRAIN/${thread}_${task}_b"* >/dev/null 2>&1 || missing="$missing $task"
        else
            [ -d "$FOLDER_TRAIN/${thread}_${task}_b${NUM_BLOCKS}" ] || missing="$missing $task"
        fi
    done
    if [ -n "$missing" ]; then
        echo "[$thread] WARNING missing oracle folders:$missing (those datasets drop out of their groups)"
    else
        echo "[$thread] all oracle folders present"
    fi
done

echo
for thread in $THREADS; do
    num_layers=$(layers_for "$thread")
    block_size=$(block_size_for "$thread")
    log="$FOLDER_LOGS/ablation_fast_${thread}.log"
    echo "===== $thread (layers=$num_layers, epochs=$NUM_EPOCHS, block_size=${block_size:-n/a}, device=$DEVICE) -> $log ====="

    FOLDER_TRAIN="$FOLDER_TRAIN" THREAD="$thread" DEVICE="$DEVICE" \
    NUM_LAYERS="$num_layers" NUM_EPOCHS="$NUM_EPOCHS" NUM_BLOCKS="$NUM_BLOCKS" \
    BLOCK_SIZE="$block_size" \
        python -u ablation_tests/ablation_test_fast.py >> "$log" 2>&1
    status=$?

    if [ $status -ne 0 ]; then
        echo "[$thread] FAILED (exit $status) -- see $log; continuing with the next thread"
    else
        echo "[$thread] done"
    fi

    # per-thread HTML regardless of exit status (partial reports still render)
    python ablation_tests/build_report_html.py --thread "$thread" || true
done

echo
echo "all threads processed. Reports: ablation_report_<thread>.html; bundles: routers_final/"
