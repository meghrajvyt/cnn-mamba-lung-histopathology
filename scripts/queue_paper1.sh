#!/bin/bash
# Paper 1 experiment queue for the AWS instance.
#
#   tmux new-session -d -s queue "scripts/queue_paper1.sh > logs/queue_paper1.log 2>&1"
#
# DATA. Raw LC25000 images (./data), no stain normalisation. All runs before
# 2026-10-11 used ./data_macenko, produced by a defective Macenko
# implementation (see data/transforms.py MacenkoNormalize); they were archived
# and are not reused. Colour augmentation (ColorJitter) remains in training.
#
# PRE-SPECIFIED SELECTION RULE (committed before any tier-4 result existed,
# commit 9195061, where it was headed "pre-registered": it is fixed in this
# repository's history, not in a public registry; unchanged by the data
# correction):
#   Between hybrid_res and hybrid_res_xattn, the variant with the higher
#   best validation accuracy at seed 42 is selected. Ties go to hybrid_res
#   (fewer parameters). Test-set results play no part in the choice. The
#   selected variant is then compared with every baseline on the test set,
#   across seeds 42/1337/2024, with McNemar tests.
#
# Order is by value to the paper's claims; finished runs are skipped, so the
# script can be restarted at any point.
set -uo pipefail
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)
source /opt/pytorch/bin/activate
export DATA_ROOT=./data STAIN=0 BATCH=32

echo "Paper 1 queue on raw images ($(date -u '+%H:%M UTC'))"

# 1. Baselines, seed 42 (cheap: ~1.5 h together)
scripts/run_study.sh --only "cnn_only,effnet_b0,vit_small,swin_tiny" \
  > logs/study_baselines_s42.log 2>&1
echo "baselines finished ($(date -u '+%H:%M UTC'))"

# 2. Improved hybrid variants, seed 42
scripts/run_study.sh --tier 4 > logs/study_tier4.log 2>&1
echo "hybrid variants finished ($(date -u '+%H:%M UTC'))"

# 3. Select by validation accuracy only
SEL=$(python3 - <<'PY'
import torch
best = {}
for v in ("hybrid_res", "hybrid_res_xattn"):
    p = f"checkpoints/{v}_d4_224px_s42/best_model.pth"
    try:
        best[v] = float(torch.load(p, map_location="cpu", weights_only=False)["best_val_acc"])
    except Exception as e:
        print(f"# {v}: unavailable ({e})")
for v, a in best.items():
    print(f"# {v}: best_val_acc={a:.4f}")
# max() keeps the first key on ties, so hybrid_res wins ties
print(max(best, key=best.get) if best else "hybrid_res")
PY
)
echo "$SEL" | grep '^#' | tee -a logs/selected_hybrid.txt
SELECTED=$(echo "$SEL" | grep -v '^#' | tail -1)
echo "selected: $SELECTED ($(date -u '+%H:%M UTC'))" | tee -a logs/selected_hybrid.txt

# 4. Original dual-pathway hybrid, seed 42: the design the residual hybrid
#    replaces, needed to show why the change was made
scripts/run_study.sh --only grouped > logs/study_grouped_s42.log 2>&1
echo "original hybrid finished ($(date -u '+%H:%M UTC'))"

# 5. LungHist700 (second dataset): patient- vs image-level 5-fold CV, then
#    external test of every LC25000 checkpoint that exists at this point
scripts/queue_lh700.sh "$SELECTED" > logs/queue_lh700.log 2>&1
echo "LungHist700 finished ($(date -u '+%H:%M UTC'))"

# 6. Seeds 1337/2024 for the selected hybrid and the baselines
scripts/run_study.sh --seeds "1337 2024" \
  --only "$SELECTED,cnn_only,effnet_b0,vit_small,swin_tiny" \
  > logs/study_seeds_main.log 2>&1
echo "seed repeats finished ($(date -u '+%H:%M UTC'))"

# 7. Leaky split for each baseline, then for the selected hybrid
scripts/run_study.sh --tier 3 > logs/study_tier3.log 2>&1
scripts/run_study.sh --only "leaky_$SELECTED" > logs/study_tier5.log 2>&1

# 8. External test for the checkpoints trained since step 5 (finished CV
#    runs and external results are skipped, so only the new ones run)
scripts/queue_lh700.sh "$SELECTED" >> logs/queue_lh700.log 2>&1
echo "all queued work finished ($(date -u '+%H:%M UTC'))"
