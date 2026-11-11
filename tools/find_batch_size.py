#!/usr/bin/env python3
"""Find the largest batch size that fits in this machine's GPU memory.

WHY THIS EXISTS
---------------
On a small GPU (4-8 GB) the difference between a run that completes and one that
dies an hour in is a single hyperparameter, and the failure mode is a crash rather
than a warning. This probes actual forward+backward+step memory for the given
architecture and writes a config the study runner can consume, so nobody has to
guess.

The dominant memory term for this model is the selective scan's
(B, L, d_inner, d_state) intermediates, which is why --grad_checkpoint matters far
more here than it does on a 16 GB card.

Usage:
    python tools/find_batch_size.py                          # probe defaults
    python tools/find_batch_size.py --arch hybrid --depth 4
    python tools/find_batch_size.py --all-archs --out batch_config.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import ModelConfig  # noqa: E402
from models.baselines import ARCHITECTURES, build_model  # noqa: E402


def device_info() -> Dict:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"device": "cuda", "name": p.name,
                "total_gb": round(p.total_memory / 1e9, 2),
                "capability": f"sm_{p.major}{p.minor}",
                "tensor_cores": p.major >= 7}
    if torch.backends.mps.is_available():
        return {"device": "mps", "name": "Apple MPS", "total_gb": None,
                "capability": None, "tensor_cores": None}
    return {"device": "cpu", "name": "CPU", "total_gb": None,
            "capability": None, "tensor_cores": None}


def try_batch(arch: str, batch: int, depth: int, image_size: int,
              grad_ckpt: bool, amp: bool, dev: str) -> Optional[Dict]:
    """Attempt one training step. Returns metrics, or None on OOM."""
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cfg = ModelConfig()
    cfg.cnn_pretrained = False      # weights don't affect memory; skip the fetch
    cfg.num_classes = 3
    cfg.image_size = image_size
    cfg.mamba_depth = depth
    cfg.grad_checkpoint = grad_ckpt

    model = opt = None
    try:
        model = build_model(arch, cfg).to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(dev, enabled=amp)
        lf = nn.CrossEntropyLoss()
        x = torch.randn(batch, 3, image_size, image_size, device=dev)
        y = torch.randint(0, 3, (batch,), device=dev)

        for _ in range(2):          # two steps: optimiser state is allocated lazily
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev, enabled=amp):
                loss = lf(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        peak = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else None
        return {"ok": True, "peak_gb": round(peak, 2) if peak else None}

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return None
    finally:
        del model, opt
        if dev == "cuda":
            torch.cuda.empty_cache()


def probe(arch: str, depth: int, image_size: int, grad_ckpt: bool,
          amp: bool, dev: str, safety: float, max_batch: int) -> Dict:
    """Largest power-of-two batch that fits, then a safety margin applied."""
    best, best_peak, batch = 0, None, 1
    while batch <= max_batch:
        res = try_batch(arch, batch, depth, image_size, grad_ckpt, amp, dev)
        status = "ok" if res else "OOM"
        peak = f"{res['peak_gb']:.2f} GB" if (res and res["peak_gb"]) else "-"
        print(f"    batch {batch:3d}: {status:3s}  peak {peak}", flush=True)
        if res is None:
            break
        best, best_peak, batch = batch, res["peak_gb"], batch * 2

    # Back off one step. Peak memory grows with activation size, which varies
    # slightly with data; running at the exact ceiling invites an OOM hours in.
    recommended = max(1, int(best * safety)) if best else 0
    return {"arch": arch, "max_batch": best, "recommended_batch": recommended,
            "peak_gb_at_max": best_peak, "grad_checkpoint": grad_ckpt,
            "mamba_depth": depth, "image_size": image_size, "amp": amp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="hybrid", choices=list(ARCHITECTURES))
    ap.add_argument("--all-archs", action="store_true",
                    help="probe every architecture the study needs")
    ap.add_argument("--depth", type=int, default=4, help="mamba_depth")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--no-grad-checkpoint", dest="grad_ckpt",
                    action="store_false", default=True,
                    help="probe without activation recomputation")
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    ap.add_argument("--safety", type=float, default=0.5,
                    help="fraction of the max batch to recommend (default 0.5)")
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--out", default=None, help="write JSON config here")
    args = ap.parse_args()

    info = device_info()
    print("=" * 66)
    print(f"device : {info['name']} ({info['device']})")
    if info["total_gb"]:
        print(f"memory : {info['total_gb']} GB")
        print(f"compute: {info['capability']}, tensor cores: {info['tensor_cores']}")
    print(f"probing: depth={args.depth} image_size={args.image_size} "
          f"grad_checkpoint={args.grad_ckpt} amp={args.amp}")
    print("=" * 66)

    if info["device"] == "cpu":
        print("\nNo GPU visible. If this machine has an NVIDIA card, the installed")
        print("torch is probably the CPU build. Reinstall with a CUDA wheel:")
        print("  pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    dev = info["device"]
    archs = list(ARCHITECTURES) if args.all_archs else [args.arch]
    results = {}
    for arch in archs:
        print(f"\n{arch}:")
        # Single-pathway and timm baselines have no Mamba blocks, so depth is
        # irrelevant and grad checkpointing does nothing for them.
        depth = args.depth if arch in ("hybrid", "hybrid_pooled", "mamba_only") else 1
        gc = args.grad_ckpt if arch in ("hybrid", "hybrid_pooled", "mamba_only") else False
        results[arch] = probe(arch, depth, args.image_size, gc, args.amp,
                              dev, args.safety, args.max_batch)

    print("\n" + "=" * 66)
    print(f"{'arch':16s} {'max':>5s} {'recommended':>12s} {'peak@max':>10s}")
    print("=" * 66)
    for arch, r in results.items():
        peak = f"{r['peak_gb_at_max']:.2f} GB" if r["peak_gb_at_max"] else "-"
        print(f"{arch:16s} {r['max_batch']:>5d} {r['recommended_batch']:>12d} {peak:>10s}")

    if any(r["max_batch"] == 0 for r in results.values()):
        print("\nWARNING: some architectures do not fit at batch 1. Options:")
        print("  - reduce --mamba_depth (2 instead of 4)")
        print("  - keep --grad_checkpoint enabled")
        print("  - lower --image_size (must keep CNN feature maps divisible by 7:")
        print("    224 works; 160 and 128 do not with pool_size=7)")

    payload = {"device": info, "probe_settings": vars(args), "results": results}
    out = args.out or "batch_config.json"
    pathlib.Path(out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}")
    print("scripts/run_study.sh reads this automatically if present.")


if __name__ == "__main__":
    main()
