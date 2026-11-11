#!/bin/bash
# Run one named experiment, then evaluate it. Detached and resumable.
#
#   scripts/run_experiment.sh <name> [extra train.py args...]
#
# Examples:
#   scripts/run_experiment.sh leaky_d2 --no_use_groups
#   scripts/run_experiment.sh grouped_d4 --mamba_depth 4
#
# Every experiment inherits the SAME base config, so runs differ only by the
# arguments passed on the command line. That is what makes the comparisons
# between runs legitimate rather than confounded.
#
# Base config is pinned to the first grouped run so all later runs are
# comparable to it:
#   epochs 16 (cosine schedule length), batch 4, mamba_depth 2, warmup 3,
#   patience 5, lr 1e-4, seed 42, Macenko on, batch mixing off.
#
# NOTE ON BUDGET: the reference grouped run completed 5 of 16 epochs before being
# stopped. To compare against it fairly, stop a new run at the same epoch count
# (STOP_AFTER_EPOCH below) rather than letting it run longer. The LR trajectory
# is then identical, because both use a cosine schedule defined over 16 epochs.

set -euo pipefail
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)

NAME="${1:-}"
[[ -n "$NAME" ]] || { echo "usage: $0 <name> [extra train.py args...]"; exit 1; }
shift || true

CKPT="./checkpoints/$NAME"
LOGD="./logs/$NAME"
RUNLOG="./logs/${NAME}_run.log"
RESULTS="./results/$NAME"

# 0 = run to --epochs. Set to N to stop once N epochs have completed, for
# budget-matched comparisons.
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-16}"
MAMBA_DEPTH="${MAMBA_DEPTH:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# DATA_ROOT / STAIN let a run read pre-normalised images (see
# tools/precompute_stain_norm.py) instead of applying Macenko on the fly,
# which starves the GPU on 4-vCPU instances.
DATA_ROOT="${DATA_ROOT:-./data}"
STAIN="${STAIN:-1}"

mkdir -p "$CKPT" "$LOGD" "$RESULTS"

ARGS=(
  --data_root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --mamba_depth "$MAMBA_DEPTH"
  --image_size "$IMAGE_SIZE"
  --warmup_epochs 3
  --patience 5
  --lr 1e-4
  --num_workers "$NUM_WORKERS"
  $([[ "$STAIN" == "1" ]] && echo --stain_normalize || echo --no_stain_normalize)
  --seed "$SEED"
  --checkpoint_dir "$CKPT"
  --log_dir "$LOGD"
)

if [[ -f "$CKPT/last_model.pth" ]]; then
  echo "resuming from $CKPT/last_model.pth"
  ARGS+=(--resume "$CKPT/last_model.pth")
fi

echo "experiment : $NAME"
echo "started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "extra args : $*"
echo "stop after : ${STOP_AFTER_EPOCH:-none} epochs"
echo

# Train in the background so this script can enforce STOP_AFTER_EPOCH.
# caffeinate (macOS) keeps the machine awake; it does not exist on Linux, so
# fall back to running train.py directly there.
if command -v caffeinate >/dev/null 2>&1; then
  KEEPAWAKE=(caffeinate -is)
else
  KEEPAWAKE=()
fi
"${KEEPAWAKE[@]}" python3 -u train.py "${ARGS[@]}" "$@" &
TRAIN_PID=$!

if [[ "$STOP_AFTER_EPOCH" != "0" ]]; then
  (
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      # `grep -c` already prints 0 when nothing matches AND exits non-zero, so
      # `|| echo 0` appended a second line and the comparison below got "0\n0".
      # Take the first line and default only if grep produced nothing at all.
      done_epochs=$(grep -ac "Val   — loss" "$RUNLOG" 2>/dev/null | head -1)
      done_epochs=${done_epochs:-0}
      if [[ "$done_epochs" =~ ^[0-9]+$ ]] && [[ "$done_epochs" -ge "$STOP_AFTER_EPOCH" ]]; then
        echo ">>> reached $STOP_AFTER_EPOCH epochs, stopping training for a budget-matched comparison"
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        break
      fi
      sleep 30
    done
  ) &
fi

wait "$TRAIN_PID" 2>/dev/null || true
echo "training ended $(date '+%H:%M:%S')"

if [[ ! -f "$CKPT/best_model.pth" ]]; then
  echo "ERROR: no checkpoint produced; see $RUNLOG"
  exit 1
fi

# Split parameters are recovered from the checkpoint, so the test set matches
# whatever this run actually trained against, grouped or per-file.
echo "evaluating"
python3 -u evaluate.py \
  --checkpoint "$CKPT/best_model.pth" \
  --data_root "$DATA_ROOT" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --num-xai-samples 8 \
  --output_dir "$RESULTS"

echo "done $(date '+%H:%M:%S') — results in $RESULTS"
