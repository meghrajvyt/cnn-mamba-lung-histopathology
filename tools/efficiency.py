#!/usr/bin/env python3
"""Model cost under identical conditions: parameters, FLOPs, latency, memory.

For every architecture, built exactly as for training (224 px, Mamba depth 4,
3 classes; weights are irrelevant to cost, so pretrained loading is skipped):

  params            total parameter count
  gflops            forward FLOPs for one image, counted by
                    torch.utils.flop_counter (matmul, convolution and
                    attention kernels only; the element-wise selective-scan
                    recurrence is not counted, so the Mamba models'
                    figures are lower bounds - latency is the fair measure)
  latency_b1_ms     median batch-1 forward time with IQR
  throughput_b64    images per second at batch 64
  peak_mem_b64_mb   peak allocated GPU memory for one batch-64 forward

Timing uses eval mode, no_grad and fp16 autocast (as evaluate.py does on
CUDA), after warm-up, with CUDA synchronisation around every timed call.
Run it on an otherwise idle GPU.

Usage:
    python3 tools/efficiency.py --out results/_efficiency/efficiency.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ModelConfig  # noqa: E402
from models.baselines import build_model  # noqa: E402

ARCHS = ("hybrid_res", "hybrid_res_xattn", "hybrid", "cnn_only",
         "effnet_b0", "vit_small", "swin_tiny")


def build(arch: str, image_size: int, depth: int) -> torch.nn.Module:
    cfg = ModelConfig()
    cfg.image_size = image_size
    cfg.mamba_depth = depth
    cfg.num_classes = 3
    cfg.cnn_pretrained = False
    return build_model(arch, cfg)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure(model, device, image_size, amp, warmup, iters, big_batch):
    model.eval().to(device)
    ac = torch.autocast(device.type, dtype=torch.float16,
                        enabled=amp and device.type in ("cuda", "mps"))
    x1 = torch.randn(1, 3, image_size, image_size, device=device)
    xb = torch.randn(big_batch, 3, image_size, image_size, device=device)

    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        model(x1.float())
    gflops = fc.get_total_flops() / 1e9

    with ac:
        for _ in range(warmup):
            model(x1)
        sync(device)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x1)
            sync(device)
            times.append((time.perf_counter() - t0) * 1e3)

        for _ in range(max(2, warmup // 5)):
            model(xb)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        n_b = max(5, iters // 10)
        t0 = time.perf_counter()
        for _ in range(n_b):
            model(xb)
        sync(device)
        thr = n_b * big_batch / (time.perf_counter() - t0)
        peak = (torch.cuda.max_memory_allocated(device) / 2 ** 20
                if device.type == "cuda" else None)

    t = np.array(times)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return {
        "params": int(sum(p.numel() for p in model.parameters())),
        "gflops": round(gflops, 3),
        "latency_b1_ms": {"median": round(float(med), 3),
                          "iqr": [round(float(q1), 3), round(float(q3), 3)]},
        "throughput_b%d" % big_batch: round(float(thr), 1),
        "peak_mem_b%d_mb" % big_batch: None if peak is None else round(peak, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "results/_efficiency/efficiency.json"))
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True
    out = {"device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "torch": torch.__version__, "image_size": args.image_size,
           "amp_fp16": not args.no_amp, "models": {}}
    for arch in args.archs:
        torch.manual_seed(0)
        model = build(arch, args.image_size, args.depth)
        r = measure(model, device, args.image_size, not args.no_amp,
                    args.warmup, args.iters, args.batch)
        out["models"][arch] = r
        print(f"{arch:18s} params {r['params'] / 1e6:6.2f}M  GFLOPs {r['gflops']:6.2f}  "
              f"b1 {r['latency_b1_ms']['median']:7.2f} ms  "
              f"b{args.batch} {r['throughput_b%d' % args.batch]:8.1f} img/s  "
              f"mem {r['peak_mem_b%d_mb' % args.batch]} MB", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
