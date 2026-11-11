#!/bin/bash
# One-shot setup on a fresh cloud GPU instance (AWS, Vast.ai, RunPod, Lambda).
#
#   bash scripts/aws_bootstrap.sh
#
# Verifies the GPU, installs dependencies, fetches the dataset, checks the
# grouping covers it, and sizes the batch to the card. Stops at the first real
# problem rather than failing later inside a long run.

set -uo pipefail
cd "$(dirname "$0")/.."    # repository root (this script lives in scripts/)

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok  %s\n' "$*"; }
warn() { printf '   !!  %s\n' "$*"; }
die()  { printf '\n   FAILED: %s\n' "$*" >&2; exit 1; }

say "machine"
echo "   cores : $(nproc 2>/dev/null || echo '?')"
echo "   ram   : $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}' || echo '?')"
echo "   disk  : $(df -h . | awk 'NR==2{print $4" free"}')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | sed 's/^/   gpu   : /'
else
  warn "nvidia-smi not found — this may be a CPU-only box, which is supported"
  warn "but much slower; see README.md, 'Reproducing the study'"
fi

say "python environment"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv || die "could not create a virtualenv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install -q --upgrade pip
ok "venv active: $(python3 --version)"

say "pytorch"
HAS_TORCH=$(python3 -c "import torch" 2>/dev/null && echo 1 || echo 0)
if [[ "$HAS_TORCH" == "0" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Match the CUDA wheel to the driver. Installing plain `torch` on a GPU box
    # silently yields a CPU-only build, which is the most common setup failure.
    CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
    case "$CUDA_VER" in
      12*) IDX="https://download.pytorch.org/whl/cu121" ;;
      11*) IDX="https://download.pytorch.org/whl/cu118" ;;
      *)   IDX="https://download.pytorch.org/whl/cu121" ;;
    esac
    echo "   driver CUDA $CUDA_VER -> $IDX"
    python3 -m pip install -q torch torchvision --index-url "$IDX" \
      || die "torch install failed"
  else
    python3 -m pip install -q torch torchvision || die "torch install failed"
  fi
fi

read -r TORCH_VER CUDA_OK <<< "$(python3 -c "
import torch; print(torch.__version__, torch.cuda.is_available())")"
echo "   torch : $TORCH_VER   cuda available: $CUDA_OK"

if command -v nvidia-smi >/dev/null 2>&1 && [[ "$CUDA_OK" != "True" ]]; then
  die "A GPU is present but torch cannot see it (likely a +cpu build).
   Fix: pip uninstall -y torch torchvision
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
fi
[[ "$CUDA_OK" == "True" ]] && ok "GPU usable" || warn "running CPU-only"

say "dependencies"
python3 -m pip install -q -r requirements.txt || die "requirements install failed"
ok "installed"

say "dataset"
if [[ -d data/lung_aca && -d data/lung_n && -d data/lung_scc ]]; then
  ok "already present"
else
  if [[ -f download_dataset.py ]]; then
    python3 download_dataset.py || warn "download_dataset.py failed; place LC25000 under ./data manually"
  else
    warn "no download script; place the LC25000 lung subset under ./data"
  fi
fi
for c in lung_aca lung_n lung_scc; do
  n=$(find "data/$c" -name '*.jpeg' 2>/dev/null | wc -l | tr -d ' ')
  echo "   data/$c : ${n} images"
  [[ "$n" == "5000" ]] || warn "expected 5000 in data/$c"
done

say "grouping covers this copy"
python3 - <<'PY' || die "grouping check failed — do not train until this passes"
import sys
sys.path.insert(0, ".")
from data.dataset import (HistopathologyDataset, load_group_assignments,
                          _sample_group_key)
ds = HistopathologyDataset(root="./data", transform=None, stain_normalize=False)
g = load_group_assignments()
if g is None:
    raise SystemExit("   lc25000_groups.json missing: the split would silently "
                     "fall back to PER-FILE and inflate every result")
missing = [p for p, _ in ds.samples if _sample_group_key(p) not in g]
print(f"   classes  : {ds.class_names}")
print(f"   samples  : {len(ds)}")
print(f"   groups   : {len(set(g.values()))}")
print(f"   uncovered: {len(missing)}")
if len(ds.class_names) != 3:
    raise SystemExit(f"   expected 3 classes, got {ds.class_names}")
if missing:
    raise SystemExit(
        f"   {len(missing)} files absent from the grouping, e.g. "
        f"{_sample_group_key(missing[0])}.\n"
        f"   Regenerate: python3 tools/group_lc25000.py --data-root ./data "
        f"--out lc25000_groups.json")
print("   ok")
PY

say "sizing the batch to this hardware"
if [[ "$CUDA_OK" == "True" ]]; then
  python3 tools/find_batch_size.py --all-archs --depth 4 || warn "probe failed; run_study.sh will fall back to a default"
else
  python3 tools/benchmark_cpu.py --epochs 30 || warn "cpu benchmark failed"
fi

say "ready"
cat <<'EOF'
   Next:

     scripts/run_study.sh --list              show the plan
     scripts/run_study.sh --tier 1            the experiments the paper needs most

   Run it under tmux so an SSH drop does not kill it:

     tmux new -s study
     nohup scripts/run_study.sh --tier 1 > logs/study_tier1.log 2>&1 &

   Monitor:

     scripts/status.sh
     tail -f logs/study_tier1.log

   On a spot instance, interruption is safe: re-running scripts/run_study.sh skips
   completed experiments and resumes partial ones from last_model.pth.

   To avoid paying for an idle box, chain a shutdown:

     (scripts/run_study.sh --tier 1; sudo shutdown -h now) &
EOF
