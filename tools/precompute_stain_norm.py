#!/usr/bin/env python3
"""Apply Macenko stain normalisation once and cache the results to disk.

WHY THIS EXISTS
---------------
Measured on an AWS g5.xlarge (A10G, 4 vCPU): with Macenko applied on the fly,
training ran at 0% GPU utilisation with the CPU at 100%. The GPU sat idle waiting
for the 4 dataloader workers to normalise each 768x768 image, every epoch.

NOTE (2026-10-11): ./data_macenko and ./data_lh700_macenko were produced by the
previous, incorrect MacenkoNormalize (see its docstring) and must not be used.
The Paper 1 study now trains on raw images; the corrected normaliser fails on
LungHist700, whose darker background breaks the fixed Io = 240 assumption.

Macenko is deterministic (fixed reference stain matrix), so normalising the same
image every epoch is wasted work. Doing it once and writing the output to disk
moves that cost out of the training loop entirely. Training then reads
pre-normalised images with --no_stain_normalize.

Output keeps the same <class>/<filename> layout, so lc25000_groups.json keys
still match and the leakage-free grouped split is unaffected.

Usage:
    python tools/precompute_stain_norm.py --src ./data --dst ./data_macenko
    python tools/precompute_stain_norm.py --src ./data --dst ./data_macenko --workers 4
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data.transforms import MacenkoNormalize  # noqa: E402

CLASSES = ("lung_aca", "lung_n", "lung_scc")
EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}

_normalizer = None


def _init():
    global _normalizer
    _normalizer = MacenkoNormalize()


def _process(job):
    src, dst, quality = job
    if os.path.exists(dst):
        return "skip"
    img = np.array(Image.open(src).convert("RGB"))
    out, ok = _normalizer.normalize_with_status(img)
    tmp = dst + ".tmp"
    Image.fromarray(out).save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, dst)  # atomic, so an interrupted run never leaves a half file
    # Unnormalised fallbacks are written too (layout must stay complete) but
    # counted, so a silent failure rate is visible.
    return "done" if ok else "fallback"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./data")
    ap.add_argument("--dst", default="./data_macenko")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality for the cached images")
    args = ap.parse_args()

    src_root, dst_root = pathlib.Path(args.src), pathlib.Path(args.dst)
    jobs = []
    for c in CLASSES:
        (dst_root / c).mkdir(parents=True, exist_ok=True)
        for f in sorted((src_root / c).iterdir()):
            if f.suffix.lower() in EXTS:
                jobs.append((str(f), str(dst_root / c / f.name), args.quality))

    print(f"{len(jobs)} images, {args.workers} workers -> {dst_root}", flush=True)
    t0, counts = time.time(), {"done": 0, "skip": 0, "fallback": 0}
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for i, r in enumerate(ex.map(_process, jobs, chunksize=16), 1):
            counts[r] += 1
            if i % 500 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                eta = (len(jobs) - i) / max(rate, 1e-9) / 60
                print(f"  {i}/{len(jobs)}  {rate:.1f} img/s  eta {eta:.1f} min",
                      flush=True)

    for c in CLASSES:
        n = sum(1 for _ in (dst_root / c).iterdir())
        print(f"  {c}: {n} images")
    print(f"done in {(time.time() - t0) / 60:.1f} min "
          f"({counts['done']} written, {counts['skip']} already present, "
          f"{counts['fallback']} left unnormalised because estimation failed)")


if __name__ == "__main__":
    main()
