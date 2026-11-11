#!/bin/bash
# Analyses added after the pre-submission review; the protocol, fixed before
# any of them was run, is docs/added_analyses_protocol.md. Run on AWS from the
# project root.
#
#   scripts/queue_review.sh e1 b2      learning-rate sensitivity for the five
#                                      original architectures (A2), then the
#                                      7 x 7 random-map re-scoring (A3)
#   scripts/queue_review.sh vmamba     VMamba-T (A1): kernel check, LungHist700
#                                      selection and folds, its learning-rate
#                                      sensitivity (A2), LC25000 three seeds,
#                                      external test, cost
#
#   WAIT_FOR=<tmux session>            first wait for that session to end
#   SHUTDOWN_AFTER_MIN=<m>             when the parts are done, stop the
#                                      instance m minutes later, or as soon
#                                      as logs/review_synced exists (0 = never)
#
# Completed runs are skipped, so the script is safe to re-run. Run a copy
# (e.g. logs/queue_review_running.sh) so that a git pull cannot change the
# file while bash is reading it; the copy still changes to the project root.
set -uo pipefail
cd "$(dirname "$0")/.."
source /opt/pytorch/bin/activate

MAG=all
SPLITS=./splits/lunghist700
SPLITS7=./splits/lunghist700_7c
DATA3=./data_lh700_512
DATA7=./data_lh700_7c_512
export STAIN=0 EPOCHS=100 BATCH_SIZE=32 IMAGE_SIZE=224 MAMBA_DEPTH=4
PATIENCE=100
LRS="1e-4 1e-3"                                # the pre-specified grid
SENS_LR=3e-3                                   # A2
SEL_FILE=logs/lh700_lr_selection.txt           # primary record, cited in Methods
VM_SEL_FILE=logs/lh700_lr_selection_vmamba.txt
SENS_FILE=logs/lh700_lr_sensitivity.txt
ORIGINAL_ARCHS="hybrid_res cnn_only effnet_b0 vit_small swin_tiny"
VM=vmamba_tiny

done_run() { [[ -f "results/$1/evaluation_results.json" && -f "results/$1/predictions.npz" ]]; }

train() {  # train <name> <arch> <lr> <data_root> <split_file>  (as scripts/queue_lh700.sh)
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

greater() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$1" "$2"; }

primary_lr() {  # arch -> rate chosen by the pre-specified selection
  local f=$SEL_FILE
  [[ $1 == "$VM" ]] && f=$VM_SEL_FILE
  grep "^selected: $1 lr=" "$f" 2>/dev/null | tail -1 | sed 's/.*lr=\([^ ]*\).*/\1/'
}

lh700_runs() {  # lh700_runs <arch> <lr> <prefix>: three-class folds 1-4, seven-class folds 0-4
  local a=$1 lr=$2 p=$3 fold
  for fold in 1 2 3 4; do
    train "${p}patient${MAG}_f${fold}_${a}" "$a" "$lr" "$DATA3" "$SPLITS/patient_${MAG}_fold${fold}.json"
  done
  for fold in 0 1 2 3 4; do
    train "${p}7c_patient${MAG}_f${fold}_${a}" "$a" "$lr" "$DATA7" "$SPLITS7/patient_${MAG}_fold${fold}.json"
  done
}

reuse_fold0() {  # the fold-0 run at the chosen rate is the selection run itself
  local sel=$1 main=$2
  if ! done_run "$main" && done_run "$sel"; then
    cp -r "results/$sel" "results/$main"
    echo "[copy] $main <- $sel"
  fi
}

sensitivity() {  # A2 for one architecture
  local a=$1 p v vp sel
  p=$(primary_lr "$a")
  [[ -n $p ]] || { echo "[FAIL] sensitivity $a -- no primary selection recorded"; return; }
  sel="lh700_lrsel_patient${MAG}_f0_${a}_lr${SENS_LR}"
  train "$sel" "$a" "$SENS_LR" "$DATA3" "$SPLITS/patient_${MAG}_fold0.json"
  v=$(best_val "$sel")
  vp=$(best_val "lh700_lrsel_patient${MAG}_f0_${a}_lr${p}")
  [[ $v != -1 && $vp != -1 ]] || { echo "[FAIL] sensitivity $a -- selection checkpoint missing"; return; }
  grep -q "^# $a lr=$SENS_LR " "$SENS_FILE" 2>/dev/null ||
    echo "# $a lr=$SENS_LR best_val_acc=$v (pre-specified choice lr=$p best_val_acc=$vp)" >> "$SENS_FILE"
  if greater "$v" "$vp"; then
    grep -q "^sensitivity: $a " "$SENS_FILE" ||
      echo "sensitivity: $a lr=$SENS_LR ($(date -u '+%Y-%m-%d %H:%M UTC')); its LungHist700 runs are repeated" >> "$SENS_FILE"
    reuse_fold0 "$sel" "lh700_lr${SENS_LR}_patient${MAG}_f0_${a}"
    lh700_runs "$a" "$SENS_LR" "lh700_lr${SENS_LR}_"
  else
    grep -q "^sensitivity: $a " "$SENS_FILE" ||
      echo "sensitivity: $a lr=$p ($(date -u '+%Y-%m-%d %H:%M UTC')); lr=$SENS_LR is not better" >> "$SENS_FILE"
  fi
}

part_e1() {
  echo "== A2: learning-rate sensitivity, original architectures ($(date -u '+%H:%M UTC'))"
  local a
  for a in $ORIGINAL_ARCHS; do sensitivity "$a"; done
}

ckpt_3c() {  # fold arch -> checkpoint of the three-class patient-level fold model
  local f=$1 a=$2
  if [[ $f == 0 ]]; then
    echo "checkpoints/lh700_lrsel_patient${MAG}_f0_${a}_lr$(primary_lr "$a")/best_model.pth"
  else
    echo "checkpoints/lh700_patient${MAG}_f${f}_${a}/best_model.pth"
  fi
}

xai_rg7() {  # xai_rg7 <name> <canvas> <checkpoint> <out> extra-args...
  local name=$1 canvas=$2 ck=$3 out=$4; shift 4
  [[ -f "$out/xai_metrics.json" && -f "$out/xai_per_sample.npz" ]] && { echo "[skip] rg7 $name $canvas"; return; }
  [[ -f "$ck" ]] || { echo "[FAIL] rg7 $name -- missing $ck"; return; }
  echo "[run ] rg7 $name $canvas ($(date -u '+%H:%M'))"
  if python3 -u tools/xai_metrics.py --checkpoint "$ck" --steps 25 --deletion_baseline "$canvas" \
       --random_grid 7 --skip_attribution --num_workers 4 --output_dir "$out" "$@" \
       > "logs/xai_rg7_${name}_${canvas}.log" 2>&1; then
    echo "[ok  ] rg7 $name $canvas"
  else
    echo "[FAIL] rg7 $name $canvas -- see logs/xai_rg7_${name}_${canvas}.log"
  fi
}

part_b2() {
  echo "== A3: faithfulness against a 7 x 7 random grid ($(date -u '+%H:%M UTC'))"
  local ck run canvas sub n a f name
  # LC25000: the runs scored by scripts/queue_xai.sh (every non-leaky 224-px run).
  for ck in checkpoints/*_d4_224px_s*/best_model.pth; do
    run=$(basename "$(dirname "$ck")")
    [[ "$run" == leaky* || "$run" == "$VM"* ]] && continue
    for canvas in blur mean; do
      sub=xai_metrics_rg7; [[ $canvas == mean ]] && sub=xai_metrics_meandel_rg7
      xai_rg7 "$run" "$canvas" "$ck" "results/$run/$sub" --num_samples 300
    done
  done
  # LungHist700: the three-class fold models of scripts/queue_xai_lh700.sh.
  n=$(sed -n 's/^num_samples=\([0-9][0-9]*\).*/\1/p' logs/xai_lh700_plan.txt)
  [[ -n $n ]] || { echo "[FAIL] cannot read num_samples from logs/xai_lh700_plan.txt"; return; }
  for a in $ORIGINAL_ARCHS; do
    for f in 0 1 2 3 4; do
      name="lh700_patient${MAG}_f${f}_${a}"
      for canvas in blur mean; do
        sub=xai_metrics_rg7; [[ $canvas == mean ]] && sub=xai_metrics_meandel_rg7
        xai_rg7 "$name" "$canvas" "$(ckpt_3c "$f" "$a")" "results/$name/$sub" --num_samples "$n"
      done
    done
  done
}

part_vmamba() {
  echo "== A1: VMamba-T ($(date -u '+%H:%M UTC'))"
  if ! python3 -u tools/check_vmamba_kernel.py --out logs/vmamba_kernel_check.txt \
         > logs/vmamba_kernel_check.log 2>&1; then
    echo "[FAIL] VMamba-T kernel check -- see logs/vmamba_kernel_check.log; VMamba-T runs skipped"
    return
  fi
  echo "[ok  ] kernel check"
  local lr best v chosen sel emag ck run out
  # Pre-specified LungHist700 selection, as for every other architecture.
  if [[ -z $(primary_lr "$VM") ]]; then
    best=-1; chosen="${LRS%% *}"
    for lr in $LRS; do
      sel="lh700_lrsel_patient${MAG}_f0_${VM}_lr${lr}"
      train "$sel" "$VM" "$lr" "$DATA3" "$SPLITS/patient_${MAG}_fold0.json"
      v=$(best_val "$sel")
      echo "# $VM lr=$lr best_val_acc=$v" >> "$VM_SEL_FILE"
      if greater "$v" "$best"; then best=$v; chosen=$lr; fi   # ties keep the smaller rate
    done
    echo "selected: $VM lr=$chosen ($(date -u '+%H:%M UTC'))" >> "$VM_SEL_FILE"
  fi
  lr=$(primary_lr "$VM")
  reuse_fold0 "lh700_lrsel_patient${MAG}_f0_${VM}_lr${lr}" "lh700_patient${MAG}_f0_${VM}"
  lh700_runs "$VM" "$lr" "lh700_"
  sensitivity "$VM"
  # LC25000, grouped split, three seeds, through the same study runner.
  echo "[run ] LC25000 $VM, three seeds ($(date -u '+%H:%M'))"
  DATA_ROOT=./data STAIN=0 BATCH=32 EPOCHS=50 \
    scripts/run_study.sh --only "$VM" --seeds "42 1337 2024" > "logs/study_${VM}.log" 2>&1 \
    && echo "[ok  ] LC25000 $VM" || echo "[FAIL] LC25000 $VM -- see logs/study_${VM}.log"
  # External test of its LC25000 checkpoints (as scripts/queue_lh700.sh, step 4).
  for emag in 20x 40x; do
    for ck in checkpoints/${VM}_d4_224px_s*/best_model.pth; do
      [[ -f "$ck" ]] || continue
      run=$(basename "$(dirname "$ck")")
      out="results/external_${emag}/${run}"
      [[ -f "$out/predictions.npz" ]] && { echo "[skip] external $emag $run"; continue; }
      python3 -u evaluate.py --checkpoint "$ck" --data_root "$DATA3" \
        --split_file "$SPLITS/external_${emag}.json" --batch_size 32 \
        --num_workers 4 --num-xai-samples 0 --output_dir "$out" \
        > "logs/external_${emag}_${run}.log" 2>&1 \
        && echo "[ok  ] external $emag $run" || echo "[FAIL] external $emag $run"
    done
  done
  # Cost, on the idle GPU. Hybrid-res and Swin-T are measured again in the same
  # session, as a check that this session's timings match Supplementary Table S12.
  if [[ -f results/_efficiency/efficiency_vmamba.json ]]; then
    echo "[skip] VMamba-T cost"
  else
    python3 -u tools/efficiency.py --archs hybrid_res swin_tiny "$VM" \
      --out results/_efficiency/efficiency_vmamba.json \
      > logs/efficiency_vmamba.log 2>&1 && echo "[ok  ] VMamba-T cost" \
      || echo "[FAIL] VMamba-T cost -- see logs/efficiency_vmamba.log"
  fi
}

finish() {
  local m=${SHUTDOWN_AFTER_MIN:-0} i
  (( m > 0 )) || return 0
  touch logs/review_done
  echo "parts done ($(date -u '+%H:%M UTC')); stopping the instance in $m min, or once logs/review_synced exists"
  for ((i = 0; i < m; i++)); do [[ -f logs/review_synced ]] && break; sleep 60; done
  while pgrep -f "[t]rain.py|[x]ai_metrics.py|[e]valuate.py" > /dev/null; do sleep 60; done
  echo "stopping the instance ($(date -u '+%H:%M UTC'))"
  sudo shutdown -h now
}

if [[ -n "${WAIT_FOR:-}" ]]; then
  echo "waiting for tmux session '$WAIT_FOR' ($(date -u '+%H:%M UTC'))"
  while tmux has-session -t "=$WAIT_FOR" 2>/dev/null; do sleep 120; done
fi
echo "review queue: parts '$*' ($(date -u '+%Y-%m-%d %H:%M UTC'))"
for part in "$@"; do
  case $part in
    e1) part_e1 ;;
    b2) part_b2 ;;
    vmamba) part_vmamba ;;
    *) echo "unknown part '$part' (use e1, b2, vmamba)" ;;
  esac
done
echo "review queue finished ($(date -u '+%H:%M UTC'))"
finish
