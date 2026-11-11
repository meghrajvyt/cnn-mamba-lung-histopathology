#!/bin/bash
# Run the full experimental study, in the order that matters for publication.
#
#   scripts/run_study.sh              run everything still outstanding
#   scripts/run_study.sh --list       show the plan and what is already done
#   scripts/run_study.sh --tier 1     run only the fatal-gap experiments
#   scripts/run_study.sh --dry-run    print the commands without executing
#
# Safe to interrupt and re-run: completed experiments are skipped, and a
# partially-trained one resumes from its last checkpoint.
#
# TIERS, ordered by how likely the gap is to cause rejection (from the original
# planning audit, which is no longer part of the repository). The paper's
# LC25000 runs were made by scripts/queue_paper1.sh, which calls this script
# with --only and --tier.
#
#   Tier 1  The claims the paper cannot be submitted without.
#           - grouped vs leaky, which is the headline leakage result
#           - cnn_only / mamba_only, without which "hybrid" means nothing
#           - effnet_b0, the "would a plain CNN do?" control
#   Tier 2  Architecture justification and the efficiency argument.
#           - fusion ablation, transformer baselines
#   Tier 3  Multi-seed repeats, for mean +/- std and paired significance tests.
#
# Batch size comes from batch_config.json if present (produced by
# tools/find_batch_size.py), because a 4 GB card and a 16 GB card need very
# different settings and guessing wastes hours on OOM crashes.

set -uo pipefail
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)

# Detect whether an accelerator exists. CPU-only machines need a materially
# smaller configuration: the full 224px depth-4 model costs ~165 min/epoch on an
# 8-core CPU, which is 3+ days for a single run and makes a study impossible.
ACCEL=$(python3 - <<'PY' 2>/dev/null || echo cpu
import torch
print("cuda" if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available() else "cpu")
PY
)

case "$ACCEL" in
  cuda)
    # A discrete GPU can afford the architecture as described.
    DEFAULT_EPOCHS=50; DEFAULT_DEPTH=4; DEFAULT_SIZE=224; DEFAULT_BATCH=""
    DEFAULT_GC=1; DEFAULT_WORKERS=4
    ;;
  mps)
    # Measured on an M2: 224px depth 4 costs 82 min/epoch (41 h per run), which
    # makes a study impractical. 128px keeps depth 4 at 24.8 min/epoch, ~12 h per
    # run. Resolution is a preprocessing choice; depth is architectural, so
    # reducing resolution is the cheaper concession scientifically.
    # Reduced resolution on MPS requires the adaptive pool_size selection in
    # CNNPathway — a fixed pool of 7 restricts Apple silicon to 224px.
    DEFAULT_EPOCHS=30; DEFAULT_DEPTH=4; DEFAULT_SIZE=128; DEFAULT_BATCH=8
    # Throughput does not scale with batch on MPS (bandwidth-bound) and memory is
    # unified, so recomputation only costs time here.
    DEFAULT_GC=0; DEFAULT_WORKERS=4
    ;;
  *)
    # CPU: 128px depth 2 is ~30 min/epoch against ~165 for 224px depth 4.
    # Depth has to give here too; the saving is needed.
    DEFAULT_EPOCHS=30; DEFAULT_DEPTH=2; DEFAULT_SIZE=128; DEFAULT_BATCH=8
    DEFAULT_GC=0; DEFAULT_WORKERS=4
    ;;
esac

EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
DEPTH="${DEPTH:-$DEFAULT_DEPTH}"
IMAGE_SIZE="${IMAGE_SIZE:-$DEFAULT_SIZE}"
SEEDS="${SEEDS:-42}"
BATCH="${BATCH:-$DEFAULT_BATCH}"
GRAD_CKPT="${GRAD_CKPT:-$DEFAULT_GC}"
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_WORKERS}"
TIER="${TIER:-0}"
# Comma-separated experiment names to run (e.g. grouped,cnn_only). Empty = all.
# Lets multi-seed repeats target only the comparisons that decide the claims.
ONLY="${ONLY:-}"
DRY=0
LIST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --tier) TIER="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --workers) NUM_WORKERS="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# ── batch size ──────────────────────────────────────────────────────────
if [[ -z "$BATCH" ]]; then
  if [[ -f batch_config.json ]]; then
    BATCH=$(python3 -c "
import json
d = json.load(open('batch_config.json'))
r = d.get('results', {})
h = r.get('hybrid') or next(iter(r.values()), {})
print(h.get('recommended_batch') or 4)
" 2>/dev/null || echo 4)
    echo "batch size $BATCH (from batch_config.json)"
  else
    BATCH=4
    echo "batch size $BATCH (default; run tools/find_batch_size.py to tune for this GPU)"
  fi
fi

GC_FLAG=""
[[ "$GRAD_CKPT" == "1" ]] && GC_FLAG="--grad_checkpoint"

# ── the plan: tier|name|extra train.py args ────────────────────────────
# Every entry inherits the same base config, so runs differ only by these args.
PLAN=(
  "1|grouped|"
  "1|leaky|--no_use_groups"
  "1|cnn_only|--arch cnn_only"
  "1|mamba_only|--arch mamba_only"
  "1|effnet_b0|--arch effnet_b0"
  "2|fusion_pooled|--arch hybrid_pooled"
  "2|vit_small|--arch vit_small"
  "2|swin_tiny|--arch swin_tiny"
  # Tier 3: the same baselines on the leaky per-file split, so leakage
  # inflation is measured per architecture, not only for the hybrid.
  "3|leaky_cnn_only|--arch cnn_only --no_use_groups"
  "3|leaky_effnet_b0|--arch effnet_b0 --no_use_groups"
  "3|leaky_vit_small|--arch vit_small --no_use_groups"
  "3|leaky_swin_tiny|--arch swin_tiny --no_use_groups"
  # Tier 4: residual feature-level hybrid (models/hybrid_res.py), two fusions.
  "4|hybrid_res|--arch hybrid_res"
  "4|hybrid_res_xattn|--arch hybrid_res_xattn"
  # Tier 5: leaky split for the hybrid variants (run only for the selected one).
  "5|leaky_hybrid_res|--arch hybrid_res --no_use_groups"
  "5|leaky_hybrid_res_xattn|--arch hybrid_res_xattn --no_use_groups"
  # Tier 6: standalone state-space comparator, added after the pre-submission
  # review (docs/added_analyses_protocol.md).
  "6|vmamba_tiny|--arch vmamba_tiny"
)

is_done() {
  [[ -f "./results/$1/evaluation_results.json" && -f "./results/$1/predictions.npz" ]]
}

if [[ "$LIST" == "1" ]]; then
  echo
  printf '%-6s %-22s %-9s %s\n' TIER EXPERIMENT STATUS ARGS
  for seed in $SEEDS; do
    for row in "${PLAN[@]}"; do
      IFS='|' read -r tier name extra <<< "$row"
      [[ "$TIER" != "0" && "$tier" != "$TIER" ]] && continue
      [[ -n "$ONLY" && ",$ONLY," != *",$name,"* ]] && continue
      full="${name}_d${DEPTH}_${IMAGE_SIZE}px_s${seed}"
      if is_done "$full"; then st="done"
      elif [[ -f "./checkpoints/$full/last_model.pth" ]]; then st="partial"
      else st="pending"; fi
      printf '%-6s %-22s %-9s %s\n' "$tier" "$full" "$st" "${extra:-<baseline>}"
    done
  done
  echo
  exit 0
fi

case "$ACCEL" in
  cuda) echo "CUDA GPU detected — full configuration (224px, depth 4)" ;;
  mps)  echo "Apple silicon detected — ${IMAGE_SIZE}px at full depth ${DEPTH}"
        echo "  (224px depth 4 costs ~82 min/epoch here; ${IMAGE_SIZE}px costs"
        echo "   ~25. Depth is preserved because it is architectural, whereas"
        echo "   resolution is a preprocessing choice.)" ;;
  *)    echo "CPU-only — reduced configuration (${IMAGE_SIZE}px, depth ${DEPTH})"
        echo "  (224px depth 4 is ~165 min/epoch on 8 cores, not a feasible"
        echo "   study, so depth is reduced here as well.)" ;;
esac
echo "  Report the configuration actually used. Comparisons remain valid because"
echo "  every run in this study shares it."
echo
echo "study: epochs=$EPOCHS depth=$DEPTH image_size=$IMAGE_SIZE batch=$BATCH"
echo "       seeds='$SEEDS' grad_ckpt=$GRAD_CKPT workers=$NUM_WORKERS"
[[ "$TIER" != "0" ]] && echo "tier filter: $TIER"
echo

TOTAL=0; RAN=0; SKIPPED=0; FAILED=0
START_ALL=$(date +%s)

for seed in $SEEDS; do
  for row in "${PLAN[@]}"; do
    IFS='|' read -r tier name extra <<< "$row"
    [[ "$TIER" != "0" && "$tier" != "$TIER" ]] && continue
    [[ -n "$ONLY" && ",$ONLY," != *",$name,"* ]] && continue

    TOTAL=$((TOTAL + 1))
    full="${name}_d${DEPTH}_${IMAGE_SIZE}px_s${seed}"

    if is_done "$full"; then
      echo "[skip] $full already evaluated"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    # Baselines have no Mamba pathway, so depth and checkpointing are no-ops
    # (VMamba-T has its own fixed depth and does not use either flag).
    depth_flag="--mamba_depth $DEPTH"
    gc="$GC_FLAG"
    case "$extra" in
      *effnet_b0*|*vit_small*|*swin_tiny*|*cnn_only*|*vmamba_tiny*) depth_flag=""; gc="" ;;
    esac

    cmd=(scripts/run_experiment.sh "$full" --epochs "$EPOCHS" --batch_size "$BATCH"
         --image_size "$IMAGE_SIZE" --num_workers "$NUM_WORKERS"
         --seed "$seed" $depth_flag $gc $extra)

    echo "=========================================================="
    echo "[run ] $full  ($(date '+%H:%M:%S'))"
    echo "       ${cmd[*]}"
    echo "=========================================================="

    if [[ "$DRY" == "1" ]]; then
      RAN=$((RAN + 1))
      continue
    fi

    t0=$(date +%s)
    if EPOCHS="$EPOCHS" MAMBA_DEPTH="$DEPTH" BATCH_SIZE="$BATCH" SEED="$seed" \
       IMAGE_SIZE="$IMAGE_SIZE" NUM_WORKERS="$NUM_WORKERS" \
       "${cmd[@]}" >> "./logs/${full}_study.log" 2>&1; then
      echo "[ok  ] $full in $((($(date +%s) - t0) / 60)) min"
      RAN=$((RAN + 1))
    else
      echo "[FAIL] $full — see ./logs/${full}_study.log"
      echo "       continuing with the rest of the study"
      FAILED=$((FAILED + 1))
    fi
  done
done

echo
echo "=========================================================="
echo "study finished in $(((($(date +%s) - START_ALL)) / 60)) min"
echo "  planned $TOTAL | ran $RAN | skipped $SKIPPED | failed $FAILED"
echo "=========================================================="

if [[ "$DRY" == "0" && "$RAN" -gt 0 ]]; then
  echo
  echo "aggregating"
  python3 aggregate_results.py --results-dirs ./results/*/ \
    --output-dir ./results/_aggregate 2>/dev/null \
    || echo "  (aggregate_results.py needs >=2 completed runs)"

  echo
  echo "paired comparisons"
  python3 compare_models.py --all ./results \
    --json-out ./results/_aggregate/comparisons.json 2>/dev/null \
    || echo "  (needs >=2 runs evaluated on the same test set)"
fi

echo
echo "next: review ./results/_aggregate, then scripts/status.sh for per-run detail"
