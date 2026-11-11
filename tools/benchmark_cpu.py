#!/usr/bin/env python3
"""Measure CPU training throughput and project epoch/run time.

WHY THIS EXISTS
---------------
CPU-only training of this model is viable only in a reduced configuration, and
which reductions pay off is hardware-specific. This measures the actual cost of
the levers that matter so the choice is made on numbers rather than hope.

The three levers, in order of effect:

  image_size   Mamba cost is roughly linear in token count, and tokens scale as
               (size/16)^2. 224 -> 128 cuts tokens 196 -> 64, about 3x. Note this
               is only possible on CPU/CUDA: the MPS backend rejects the CNN
               pathway's AdaptiveAvgPool2d(7) unless feature maps are divisible
               by 7, which forces 224 there.
  mamba_depth  Step time is close to linear in depth; 4 -> 2 is about 2x.
  threads      A many-core Xeon needs torch.set_num_threads set deliberately;
               the default is not always the core count.

Usage:
    python tools/benchmark_cpu.py                    # sweep the useful configs
    python tools/benchmark_cpu.py --threads 0        # also sweep thread counts
    python tools/benchmark_cpu.py --train-images 10533 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import ModelConfig  # noqa: E402
from models.baselines import build_model  # noqa: E402


def step_time(arch: str, size: int, depth: int, batch: int,
              iters: int = 3, dev: str = "cpu",
              amp: bool = False) -> Optional[float]:
    """Seconds per training step, or None if it fails."""
    cfg = ModelConfig()
    cfg.cnn_pretrained = False
    cfg.num_classes = 3
    cfg.image_size = size
    cfg.mamba_depth = depth

    try:
        model = build_model(arch, cfg).to(dev).train()
    except Exception as e:
        print(f"      build failed: {type(e).__name__}: {str(e)[:50]}")
        return None

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(dev, enabled=amp)
    x = torch.randn(batch, 3, size, size, device=dev)
    y = torch.randint(0, 3, (batch,), device=dev)

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(dev, enabled=amp):
            loss = lf(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    try:
        one()                                  # warm caches / lazy allocations
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            one()
        sync()
        return (time.perf_counter() - t0) / iters
    except Exception as e:
        print(f"      step failed: {type(e).__name__}: {str(e)[:50]}")
        return None
    finally:
        del model, opt
        if dev == "mps":
            torch.mps.empty_cache()
        elif dev == "cuda":
            torch.cuda.empty_cache()


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24:.1f} days"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", type=int, default=10533,
                    help="training-set size (grouped split default)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threads", type=int, default=-1,
                    help="torch threads; -1 leaves default, 0 sweeps counts")
    ap.add_argument("--out", default="cpu_benchmark.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="device to benchmark; mps/cuda also enable AMP")
    args = ap.parse_args()
    DEV = args.device
    AMP = DEV in ("mps", "cuda")

    ncpu = os.cpu_count() or 1
    print("=" * 74)
    print(f"cpu cores        : {ncpu}")
    print(f"torch            : {torch.__version__}")
    print(f"torch threads    : {torch.get_num_threads()}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    try:
        import torch.backends.mkldnn as mkldnn
        print(f"mkldnn (oneDNN)  : {mkldnn.is_available()}")
    except Exception:
        pass
    print(f"projecting       : {args.train_images} train images, "
          f"{args.epochs} epochs, batch {args.batch}")
    print("=" * 74)

    if torch.cuda.is_available():
        print("\nNOTE: a CUDA device is visible. This tool measures CPU only; "
              "use tools/find_batch_size.py for the GPU path.")

    # ── optional thread sweep ────────────────────────────────────────────
    if args.threads == 0:
        print("\nthread sweep (hybrid, 128px, depth 2):")
        best_t, best_dt = torch.get_num_threads(), None
        for t in sorted({1, 2, 4, max(1, ncpu // 2), ncpu}):
            torch.set_num_threads(t)
            dt = step_time("hybrid", 128, 2, args.batch, args.iters, DEV, AMP)
            if dt is None:
                continue
            print(f"  threads {t:3d}: {dt:6.2f} s/step  {args.batch / dt:6.2f} img/s")
            if best_dt is None or dt < best_dt:
                best_t, best_dt = t, dt
        torch.set_num_threads(best_t)
        print(f"  -> fastest at {best_t} threads")
    elif args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"\ntorch threads set to {args.threads}")

    # ── config sweep ─────────────────────────────────────────────────────
    # Ordered from the full architecture down to the smallest sensible variant.
    configs = [
        ("hybrid", 224, 4, "full architecture, as described in the paper"),
        ("hybrid", 224, 2, "depth 2"),
        ("hybrid", 160, 2, "depth 2 + 160px"),
        ("hybrid", 128, 2, "depth 2 + 128px  <- usually the CPU sweet spot"),
        ("hybrid", 128, 4, "depth 4 + 128px"),
        ("cnn_only", 224, 2, "CNN-only ablation"),
        ("effnet_b0", 224, 1, "plain CNN baseline"),
    ]

    print(f"\n{'arch':11s} {'size':>5s} {'depth':>6s} {'s/step':>8s} "
          f"{'img/s':>7s} {'min/epoch':>10s} {'full run':>10s}")
    print("-" * 74)

    results: List[Dict] = []
    for arch, size, depth, note in configs:
        dt = step_time(arch, size, depth, args.batch, args.iters, DEV, AMP)
        if dt is None:
            continue
        ips = args.batch / dt
        mins = args.train_images / ips / 60
        run_h = mins * args.epochs / 60
        results.append({"arch": arch, "image_size": size, "mamba_depth": depth,
                        "s_per_step": round(dt, 3), "img_per_s": round(ips, 2),
                        "min_per_epoch": round(mins, 1),
                        "run_hours": round(run_h, 1), "note": note})
        print(f"{arch:11s} {size:>5d} {depth:>6d} {dt:>8.2f} {ips:>7.2f} "
              f"{mins:>10.1f} {fmt_hours(run_h):>10s}")
        print(f"            {note}")

    # ── what a study would cost ──────────────────────────────────────────
    if results:
        best = min(results, key=lambda r: r["run_hours"] if r["arch"] == "hybrid"
                   else float("inf"))
        print("\n" + "=" * 74)
        print("STUDY COST at the fastest hybrid config "
              f"({best['image_size']}px, depth {best['mamba_depth']})")
        print("=" * 74)
        per_run = best["run_hours"]
        for label, n in (("tier 1 (5 runs)", 5), ("tier 1+2 (8 runs)", 8),
                         ("3 seeds x tier 1 (15 runs)", 15)):
            print(f"  {label:28s} {fmt_hours(per_run * n)}")
        print(f"\n  (baselines are cheaper than the hybrid, so these are "
              f"upper bounds)")

    payload = {
        "cpu_cores": ncpu, "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "settings": vars(args), "results": results,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
