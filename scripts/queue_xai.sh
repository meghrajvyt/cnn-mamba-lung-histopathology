#!/bin/bash
# Post-queue analyses on AWS: the cost benchmark (tools/efficiency.py), then
# quantitative XAI metrics (tools/xai_metrics.py) for every LC25000 224px run.
# Run from the project root.
#
#   scripts/queue_xai.sh                  score all finished runs now
#   WAIT_FOR=queue scripts/queue_xai.sh   first wait for tmux session 'queue' to end
#
# Per run:
#   blur canvas (primary)    pathway attribution + deletion/insertion
#   mean canvas (secondary)  deletion/insertion only
# Leaky-split runs are skipped: their test sets overlap their training data, so
# their saliency is not comparable. Completed outputs are skipped, so the
# script is safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)
source /opt/pytorch/bin/activate

N="${XAI_SAMPLES:-300}"
STEPS="${XAI_STEPS:-25}"

if [[ -n "${WAIT_FOR:-}" ]]; then
  echo "waiting for tmux session '$WAIT_FOR' ($(date -u '+%H:%M UTC'))"
  while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 300; done
fi
# Cost benchmark first: it needs an idle GPU and takes a few minutes.
if [[ -f results/_efficiency/efficiency.json ]]; then
  echo "[skip] efficiency benchmark"
else
  echo "[run ] efficiency benchmark ($(date -u '+%H:%M'))"
  python3 -u tools/efficiency.py --out results/_efficiency/efficiency.json \
    > logs/efficiency.log 2>&1 && echo "[ok  ] efficiency" \
    || echo "[FAIL] efficiency -- see logs/efficiency.log"
fi

echo "XAI queue: $N images, $STEPS steps ($(date -u '+%H:%M UTC'))"

for ck in checkpoints/*_d4_224px_s*/best_model.pth; do
  run=$(basename "$(dirname "$ck")")
  [[ "$run" == leaky* ]] && continue
  for canvas in blur mean; do
    out="results/$run/xai_metrics"
    extra=()
    if [[ "$canvas" == mean ]]; then
      out="${out}_meandel"
      extra=(--skip_attribution)
    fi
    [[ -f "$out/xai_metrics.json" ]] && { echo "[skip] $run $canvas"; continue; }
    echo "[run ] $run $canvas ($(date -u '+%H:%M'))"
    if python3 -u tools/xai_metrics.py --checkpoint "$ck" \
         --num_samples "$N" --steps "$STEPS" --deletion_baseline "$canvas" \
         --num_workers 4 --output_dir "$out" "${extra[@]}" \
         > "logs/xai_${run}_${canvas}.log" 2>&1; then
      echo "[ok  ] $run $canvas"
    else
      echo "[FAIL] $run $canvas -- see logs/xai_${run}_${canvas}.log"
    fi
  done
done
# Blinded pathologist rating kit: the selected hybrid (combined and Mamba
# maps), the CNN pathway and Swin-T, seed 42, 15 test images per class.
SEL=$(sed -n 's/^selected: \([^ ]*\).*/\1/p' logs/selected_hybrid.txt 2>/dev/null | tail -1)
SEL="${SEL:-hybrid_res}"
if [[ -f results/_rating_kit/key.csv ]]; then
  echo "[skip] rating kit"
else
  echo "[run ] rating kit for $SEL ($(date -u '+%H:%M'))"
  python3 -u tools/make_rating_kit.py --per_class 15 \
    --runs "${SEL}_d4_224px_s42:combined" "${SEL}_d4_224px_s42:mamba" \
           cnn_only_d4_224px_s42:cnn swin_tiny_d4_224px_s42:swin \
    > logs/rating_kit.log 2>&1 && echo "[ok  ] rating kit" \
    || echo "[FAIL] rating kit -- see logs/rating_kit.log"
fi
echo "XAI queue finished ($(date -u '+%H:%M UTC'))"
