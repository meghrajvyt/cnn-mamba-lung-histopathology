#!/bin/bash
# LungHist700 explanation analysis for Paper 1 (Table T8c), run after the
# LC25000 XAI queue so the GPU is free (Grad-CAM++ beside training risks OOM).
#
#   WAIT_FOR=xai scripts/queue_xai_lh700.sh   wait for tmux session 'xai' to end
#   CANVAS=mean scripts/queue_xai_lh700.sh    secondary analysis: deleted pixels are
#                                       replaced by the normalisation mean
#                                       (insertion still starts from the blurred
#                                       image); faithfulness only, written to
#                                       results/<run>/xai_metrics_meandel
#
# Per three-class patient-level fold model (hybrid-res, CNN pathway,
# EfficientNet-B0, ViT-S, Swin-T): blur-canvas deletion/insertion faithfulness
# on that fold's own test images, plus (hybrid-res) pathway attribution on all
# of them. Per seven-class hybrid-res fold: pathway attribution only.
# Fold 0 of the three-class task is the learning-rate-selection run at the
# chosen rate (the same model whose predictions are reported for fold 0).
#
# Sample size is decided once, before any scored run, from a 12-image timing
# run: every test image if the estimated total fits XAI_LH_BUDGET_MIN minutes
# (default 300), otherwise the same stratified per-fold sample for every
# model (at least 60 images per fold). The decision is written to
# logs/xai_lh700_plan.txt. Completed outputs are skipped; safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)
source /opt/pytorch/bin/activate
MAG="${MAG:-all}"
STEPS="${XAI_STEPS:-25}"
BUDGET_MIN="${XAI_LH_BUDGET_MIN:-300}"
CANVAS="${CANVAS:-blur}"
case "$CANVAS" in
  blur) SUBDIR=xai_metrics ;;
  mean) SUBDIR=xai_metrics_meandel ;;
  *) echo "CANVAS must be blur or mean, got '$CANVAS'"; exit 1 ;;
esac
ARCHS=(hybrid_res cnn_only effnet_b0 vit_small swin_tiny)
PLAN=logs/xai_lh700_plan.txt

if [[ -n "${WAIT_FOR:-}" ]]; then
  echo "waiting for tmux session '$WAIT_FOR' ($(date -u '+%H:%M UTC'))"
  # '=' forces an exact name: plain -t xai would also match this session (xai_lh700).
  while tmux has-session -t "=$WAIT_FOR" 2>/dev/null; do sleep 300; done
fi
while pgrep -f "[t]rain.py|[x]ai_metrics.py --checkpoint checkpoints/[^l]" >/dev/null; do
  echo "GPU busy, waiting ($(date -u '+%H:%M'))"; sleep 300
done
echo "LungHist700 XAI queue started, canvas=$CANVAS ($(date -u '+%H:%M UTC'))"

ckpt_3c() {  # fold arch -> checkpoint path
  local f=$1 a=$2
  if [[ $f == 0 ]]; then
    local lr
    lr=$(grep "^selected: $a lr=" logs/lh700_lr_selection.txt | tail -1 | sed 's/.*lr=\([^ ]*\).*/\1/')
    echo "checkpoints/lh700_lrsel_patient${MAG}_f0_${a}_lr${lr}/best_model.pth"
  else
    echo "checkpoints/lh700_patient${MAG}_f${f}_${a}/best_model.pth"
  fi
}

# ---------------------------------------------------------------- plan
if [[ -f $PLAN ]]; then
  # The line carries a trailing comment, so keep only the leading digits.
  N=$(sed -n 's/^num_samples=\([0-9][0-9]*\).*/\1/p' "$PLAN")
  [[ -n $N ]] || { echo "[FAIL] cannot read num_samples from $PLAN"; exit 1; }
  echo "plan exists: num_samples=$N"
else
  ck=$(ckpt_3c 1 hybrid_res)
  t0=$SECONDS
  python3 -u tools/xai_metrics.py --checkpoint "$ck" --num_samples 12 --steps "$STEPS" \
    --deletion_baseline blur --skip_attribution --num_workers 4 \
    --output_dir /tmp/xai_lh700_timing > logs/xai_lh700_timing.log 2>&1 \
    || { echo "[FAIL] timing run -- see logs/xai_lh700_timing.log"; exit 1; }
  dt=$((SECONDS - t0))
  # Hybrid scores 4 maps (cnn, mamba, combined, random), the others 2, so
  # total ≈ 691 images × (1 + 4 × 0.5) hybrid-equivalents.
  est_min=$(python3 -c "print(round($dt / 12 * 691 * 3 / 60))")
  if (( est_min <= BUDGET_MIN )); then
    N=100000
  else
    N=$(python3 -c "import math; per=$dt/12; n=int($BUDGET_MIN*60/(per*3*5)); print(max(60, 3*math.floor(n/3)))")
  fi
  {
    echo "date=$(date -u '+%Y-%m-%d %H:%M UTC')"
    echo "timing_images=12 timing_seconds=$dt steps=$STEPS"
    echo "estimated_minutes_all_images=$est_min budget_minutes=$BUDGET_MIN"
    echo "num_samples=$N  # 100000 = every test image of the fold"
  } > "$PLAN"
  cat "$PLAN"
  rm -rf /tmp/xai_lh700_timing
fi

run_one() {  # name checkpoint outdir extra-args...
  local name=$1 ck=$2 out=$3; shift 3
  local log="logs/xai_${name}.log"   # blur logs keep their original names
  [[ $CANVAS == blur ]] || log="logs/xai_${name}_${CANVAS}.log"
  [[ -f "$out/xai_metrics.json" && -f "$out/xai_per_sample.npz" ]] && { echo "[skip] $name"; return; }
  [[ -f "$ck" ]] || { echo "[FAIL] $name -- missing $ck"; return; }
  echo "[run ] $name ($(date -u '+%H:%M'))"
  local t0=$SECONDS
  if python3 -u tools/xai_metrics.py --checkpoint "$ck" --steps "$STEPS" \
       --deletion_baseline "$CANVAS" --num_workers 4 --output_dir "$out" "$@" \
       > "$log" 2>&1; then
    echo "[ok  ] $name in $(( (SECONDS - t0) / 60 )) min"
  else
    echo "[FAIL] $name -- see $log"
  fi
}

# Blur: hybrid-res first (faithfulness + attribution), then seven-class
# attribution, then the comparators. Mean: faithfulness only, three-class.
EXTRA=()
[[ $CANVAS == mean ]] && EXTRA=(--skip_attribution)
for a in "${ARCHS[@]}"; do
  for f in 0 1 2 3 4; do
    name="lh700_patient${MAG}_f${f}_${a}"
    run_one "$name" "$(ckpt_3c $f $a)" "results/$name/$SUBDIR" --num_samples "$N" \
      ${EXTRA[@]+"${EXTRA[@]}"}
  done
  if [[ $a == hybrid_res && $CANVAS == blur ]]; then
    for f in 0 1 2 3 4; do
      name="lh700_7c_patient${MAG}_f${f}_hybrid_res"
      run_one "$name" "checkpoints/$name/best_model.pth" "results/$name/xai_metrics" \
        --skip_faithfulness
    done
  fi
done

if [[ $CANVAS == blur ]]; then
  echo "regenerating tables and figures ($(date -u '+%H:%M'))"
  python3 tools/paper_tables.py --mag "$MAG" > logs/paper_tables_final.log 2>&1 || echo "[FAIL] paper_tables"
  python3 tools/supp_tables.py --mag "$MAG" > logs/supp_tables_final.log 2>&1 || echo "[FAIL] supp_tables"
  python3 tools/paper_figures.py --mag "$MAG" > logs/paper_figures_final.log 2>&1 || echo "[FAIL] paper_figures"
fi
echo "LungHist700 XAI queue finished, canvas=$CANVAS ($(date -u '+%H:%M UTC'))"
