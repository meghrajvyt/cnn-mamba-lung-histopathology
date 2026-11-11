#!/bin/bash
# LungHist700 experiments (second dataset). Run from the project root on AWS.
#
#   scripts/queue_lh700.sh <selected_hybrid>          e.g. hybrid_res
#
# 0. Learning-rate selection per architecture (see SMALL-DATA PROTOCOL below).
# 1. 3-class 5-fold CV at MAG with patient-level folds (leakage-free).
# 2. 7-class differentiation grading, patient-level 5-fold CV at MAG.
# 3. 3-class image-level folds (the leaky protocol, patients shared between
#    train and test); the gap to step 1 is the patient-leakage inflation.
# 4. External test: every LC25000-trained checkpoint scored on all LungHist700
#    images at 20x and 40x, without retraining (cross-dataset generalisation).
#
# MAG defaults to "all" (both magnifications, 691 images; a patient's 20x and
# 40x images always share a fold). Decided before any LungHist700 result
# existed, replacing 20x only (359 images): it roughly doubles the training
# data per fold, which the seven-class task needs, and matches how the
# closest prior work (PCMamba, Transl. Oncol. 2026) used the dataset.
#
# SMALL-DATA PROTOCOL (protocol 2, fixed before any protocol-2 result).
# Protocol 1 reused the LC25000 recipe (50 epochs, early stopping on validation
# loss with patience 5, LR 1e-4 / backbone 1e-5). On LungHist700 that is ~13
# steps per epoch, and most models stopped after 6-33 epochs with training
# accuracy still rising; a 100-epoch EfficientNet-B0 diagnostic on fold 0
# reached only 68% training and 50% validation accuracy, i.e. it underfits at
# that learning rate. Protocol 1 is archived and not reported. Protocol 2:
#   * fixed 100 epochs, no early stopping; best-validation-accuracy checkpoint;
#   * per-architecture learning rate chosen from {1e-4, 1e-3} (backbone x0.1)
#     by best validation accuracy on fold 0 of the 3-class patient-level
#     split; ties keep 1e-4. Test data play no part. The chosen rate is used
#     for every LungHist700 run of that architecture;
#   * images read from 512 x 512 copies of the 1200 x 1200 crops (faster JPEG
#     decoding; every model still sees 224 x 224).
# Completed runs are skipped, so the script is safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)
source /opt/pytorch/bin/activate

SELECTED="${1:-hybrid_res}"
MAG="${MAG:-all}"
ARCHS="${ARCHS:-$SELECTED cnn_only effnet_b0 vit_small swin_tiny}"
SPLITS=./splits/lunghist700
SPLITS7=./splits/lunghist700_7c
# Raw images, no stain normalisation, matching the LC25000 runs.
DATA3=./data_lh700_512
DATA7=./data_lh700_7c_512
export STAIN=0
export EPOCHS="${EPOCHS:-100}" BATCH_SIZE="${BATCH_SIZE:-32}" IMAGE_SIZE=224 MAMBA_DEPTH=4
PATIENCE="${PATIENCE:-100}"
LRS="${LRS:-1e-4 1e-3}"
LR_FILE=logs/lh700_lr_selection.txt

done_run() { [[ -f "results/$1/evaluation_results.json" && -f "results/$1/predictions.npz" ]]; }

train() {  # train <name> <arch> <lr> <data_root> <split_file>
  local name=$1 arch=$2 lr=$3 data=$4 split=$5 gc=""
  if done_run "$name"; then echo "[skip] $name"; return 0; fi
  [[ "$arch" == hybrid* ]] && gc="--grad_checkpoint"
  echo "[run ] $name lr=$lr ($(date -u '+%H:%M'))"
  if DATA_ROOT="$data" scripts/run_experiment.sh "$name" --arch "$arch" --lr "$lr" \
       --patience "$PATIENCE" --split_file "$split" $gc > "logs/${name}_run.log" 2>&1; then
    echo "[ok  ] $name"
  else
    echo "[FAIL] $name -- see logs/${name}_run.log"
  fi
}

best_val() {  # best validation accuracy recorded in a run's checkpoint
  python3 -c "import torch,sys; print(float(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['best_val_acc']))" \
    "checkpoints/$1/best_model.pth" 2>/dev/null || echo -1
}

echo "LungHist700 queue: selected=$SELECTED mag=$MAG epochs=$EPOCHS patience=$PATIENCE lrs='$LRS' archs='$ARCHS' ($(date -u '+%H:%M UTC'))"

# 0. Learning-rate selection on fold 0 (3-class, patient-level), validation only.
declare -A LR_OF
for arch in $ARCHS; do
  best=-1; chosen="${LRS%% *}"   # fallback if every selection run failed
  for lr in $LRS; do
    sel="lh700_lrsel_patient${MAG}_f0_${arch}_lr${lr}"
    train "$sel" "$arch" "$lr" "$DATA3" "$SPLITS/patient_${MAG}_fold0.json"
    v=$(best_val "$sel")
    echo "# $arch lr=$lr best_val_acc=$v" | tee -a "$LR_FILE"
    # strictly greater: ties keep the earlier (smaller) learning rate
    if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$v" "$best"; then
      best=$v; chosen=$lr
    fi
  done
  LR_OF[$arch]=$chosen
  echo "selected: $arch lr=$chosen ($(date -u '+%H:%M UTC'))" | tee -a "$LR_FILE"
  # The fold-0 patient-level run at the chosen rate IS the selection run:
  # reuse its predictions instead of training it again.
  main="lh700_patient${MAG}_f0_${arch}"
  if ! done_run "$main" && done_run "lh700_lrsel_patient${MAG}_f0_${arch}_lr${chosen}"; then
    cp -r "results/lh700_lrsel_patient${MAG}_f0_${arch}_lr${chosen}" "results/$main"
    echo "[copy] $main <- lr ${chosen} selection run"
  fi
done

# Order by value: primary 3-class patient-level, then 7-class grading, then the
# image-level (leaky) reference used only for the inflation estimate.
# 1. 3-class, patient-level.
for fold in 0 1 2 3 4; do
  for arch in $ARCHS; do
    train "lh700_patient${MAG}_f${fold}_${arch}" "$arch" "${LR_OF[$arch]}" \
          "$DATA3" "$SPLITS/patient_${MAG}_fold${fold}.json"
  done
done

# 2. 7-class differentiation grading (aca/scc x well/moderately/poorly, plus
#    normal), patient-level folds only.
for fold in 0 1 2 3 4; do
  for arch in $ARCHS; do
    train "lh700_7c_patient${MAG}_f${fold}_${arch}" "$arch" "${LR_OF[$arch]}" \
          "$DATA7" "$SPLITS7/patient_${MAG}_fold${fold}.json"
  done
done

# 3. 3-class, image-level (leaky reference).
for fold in 0 1 2 3 4; do
  for arch in $ARCHS; do
    train "lh700_image${MAG}_f${fold}_${arch}" "$arch" "${LR_OF[$arch]}" \
          "$DATA3" "$SPLITS/image_${MAG}_fold${fold}.json"
  done
done

# 4. External test. Inference only (seconds per checkpoint), so every
#    magnification is scored: the LC25000 acquisition scale is undocumented.
EXT_MAGS="${EXT_MAGS:-20x 40x}"
for emag in $EXT_MAGS; do
  echo "external test on LungHist700 $emag ($(date -u '+%H:%M UTC'))"
  for ck in checkpoints/*_d4_224px_s*/best_model.pth; do
    run=$(basename "$(dirname "$ck")")
    out="results/external_${emag}/${run}"
    [[ -f "$out/predictions.npz" ]] && { echo "[skip] external $emag $run"; continue; }
    if python3 -u evaluate.py --checkpoint "$ck" --data_root "$DATA3" \
         --split_file "$SPLITS/external_${emag}.json" --batch_size 32 \
         --num_workers 4 --num-xai-samples 0 --output_dir "$out" \
         > "logs/external_${emag}_${run}.log" 2>&1; then
      echo "[ok  ] external $emag $run"
    else
      echo "[FAIL] external $emag $run -- see logs/external_${emag}_${run}.log"
    fi
  done
done
echo "LungHist700 queue finished ($(date -u '+%H:%M UTC'))"
