#!/bin/bash
# Compact status for one or all experiments.
#   scripts/status.sh            all experiments
#   scripts/status.sh leaky_d2   one experiment
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)

show() {
  local name="$1"
  # run_experiment.sh used standalone writes <name>_run.log; run_study.sh writes
  # <name>_study.log for the same experiment. Check both.
  local log="./logs/${1}_run.log"
  [[ -f "$log" ]] || log="./logs/${1}_study.log"
  [[ -f "$log" ]] || return 0

  echo "=== $name ==="
  if pgrep -f "checkpoint_dir ./checkpoints/$name" >/dev/null 2>&1; then
    echo "  status : RUNNING"
  else
    echo "  status : stopped"
  fi

  # Per-epoch history. grep -a because the log contains tqdm control bytes.
  local n=0
  while IFS=$'\t' read -r tr_line va_line; do
    n=$((n + 1))
    printf '  ep%-3d %s  |  %s\n' "$n" "$tr_line" "$va_line"
  done < <(paste \
      <(grep -a "Train — loss" "$log" 2>/dev/null | sed 's/.*Train — //') \
      <(grep -a "Val   — loss" "$log" 2>/dev/null | sed 's/.*Val   — //'))
  [[ "$n" -eq 0 ]] && echo "  (no completed epochs yet)"

  grep -a "New best val accuracy" "$log" 2>/dev/null | tail -1 | sed 's/.*INFO | train | */  /'
  grep -a "Early stopping triggered" "$log" 2>/dev/null | tail -1 | sed 's/.*| train | */  /'

  # In-flight progress. LC_ALL=C: the tqdm bar has multi-byte glyphs that make
  # tr fail with "Illegal byte sequence" under UTF-8.
  local cur
  cur=$(tail -c 3000 "$log" 2>/dev/null | LC_ALL=C tr '\r' '\n' \
        | grep -aE "Train Epoch|Val   Epoch" | tail -1 \
        | sed 's/^ *//' | cut -c1-95)
  [[ -n "$cur" ]] && echo "  now    : $cur"

  if [[ -f "./results/$name/evaluation_results.json" ]]; then
    python3 - "$name" <<'PY'
import json, sys
n = sys.argv[1]
d = json.load(open(f"./results/{n}/evaluation_results.json"))
m = d["metrics"]
keys = ["accuracy", "balanced_accuracy", "f1_macro", "auc_roc", "mcc", "cohen_kappa"]
parts = [f"{k}={m[k]:.4f}" for k in keys if isinstance(m.get(k), (int, float))]
print("  TEST   : " + "  ".join(parts))
PY
  fi
  echo
}

if [[ -n "${1:-}" ]]; then
  show "$1"
else
  seen=""
  for log in ./logs/*_run.log ./logs/*_study.log; do
    [[ -f "$log" ]] || continue
    b=$(basename "$log"); b="${b%_run.log}"; b="${b%_study.log}"
    case " $seen " in *" $b "*) continue ;; esac
    seen="$seen $b"
    show "$b"
  done
fi
