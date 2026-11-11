#!/usr/bin/env python3
"""Supplementary Figs. S2 and S3: example Grad-CAM++ maps, illustrative only.

Reuses the panels written by tools/make_rating_kit.py (no rating was
performed). Each kit panel holds the image and its overlay side by side under
a 40-pixel header with the panel id; both are cropped out here. Picks, per
class, the first images (sorted by file name) that have a panel for every
model and map in the kit, so all columns show the same images. Rows are
images; the first column is the H&E image and the others are the overlays.
Panels whose model misclassified the image are labelled.

    Supplementary Fig. S2: --kit results/_rating_kit (LC25000, seed 42)
    Supplementary Fig. S3: --kit results/_rating_kit_lh700 --name figS3_examples_lh700
                           (LungHist700, fold 1 of the three-class patient-level CV)

Usage:
    python3 tools/figS2_examples.py [--kit results/_rating_kit] [--per_class 2]
        [--name figS2_examples] [--out docs/manuscript/figures]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import figstyle as fs  # noqa: E402

# Geometry of tools/make_rating_kit.py panels (display 512, header 40).
DISPLAY, HEADER, MARGIN, GAP = 512, 40, 10, 10
IMAGE_BOX = (MARGIN, HEADER, MARGIN + DISPLAY, HEADER + DISPLAY)
OVERLAY_BOX = (2 * MARGIN + DISPLAY, HEADER, 2 * MARGIN + 2 * DISPLAY, HEADER + DISPLAY)
COLUMNS = [("hybrid_res", "combined", "Hybrid-res,\ncombined map"),
           ("hybrid_res", "mamba", "Hybrid-res,\nMamba map"),
           ("cnn_only", "cnn", "CNN pathway"),
           ("swin_tiny", "swin", "Swin-T")]
CLASS_LABEL = {"lc": {"lung_aca": "Adenocarcinoma", "lung_n": "Benign tissue",
                      "lung_scc": "Squamous cell carcinoma"},
               "lh": {"lung_aca": "Adenocarcinoma", "lung_n": "Normal tissue",
                      "lung_scc": "Squamous cell carcinoma"}}
CELL_MM = 33.0     # printed size of each image
PIXELS = 400       # about 300 dpi at CELL_MM


def arch(run: str) -> str:
    """'hybrid_res_d4_224px_s42' or 'lh700_patientall_f1_hybrid_res' -> 'hybrid_res'."""
    if run.startswith("lh700_"):
        return run.split("_", 3)[3]
    return run.split("_d4_")[0]


def crop(path: Path, box) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (2 * DISPLAY + 3 * MARGIN, DISPLAY + HEADER + MARGIN):
        raise SystemExit(f"{path}: unexpected kit panel size {im.size}")
    return np.asarray(im.crop(box).resize((PIXELS, PIXELS), Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kit", default=str(ROOT / "results/_rating_kit"))
    ap.add_argument("--per_class", type=int, default=2)
    ap.add_argument("--name", default="figS2_examples")
    ap.add_argument("--out", default=str(ROOT / "docs/manuscript/figures"))
    args = ap.parse_args()
    kit = Path(args.kit)
    rows = list(csv.DictReader(open(kit / "key.csv")))
    labels = CLASS_LABEL["lh" if "lh700" in rows[0]["run"] else "lc"]
    by_img = defaultdict(dict)
    for r in rows:
        by_img[r["image"]][(arch(r["run"]), r["map"])] = r
    wanted = [(a, m) for a, m, _ in COLUMNS]
    full = sorted(i for i, d in by_img.items() if all(k in d for k in wanted))
    pick = []
    for cls in sorted({i.split("/")[0] for i in full}):
        pick += [i for i in full if i.startswith(cls + "/")][: args.per_class]
    if not pick:
        raise SystemExit("no image has a panel for every column")

    fs.apply()
    ncol = 1 + len(COLUMNS)
    fig = fs.figure(12 + ncol * CELL_MM + 1, 10 + len(pick) * CELL_MM, layout=None)
    W, H = fig.get_figwidth() / fs.MM, fig.get_figheight() / fs.MM
    for i, img in enumerate(pick):
        for j in range(ncol):
            x0, y0 = 12 + j * CELL_MM, H - 10 - (i + 1) * CELL_MM
            ax = fig.add_axes([(x0 + 0.4) / W, (y0 + 0.4) / H,
                               (CELL_MM - 0.8) / W, (CELL_MM - 0.8) / H])
            ax.set_axis_off()
            if j == 0:
                r0 = by_img[img][wanted[0]]
                ax.imshow(crop(kit / "panels" / f"{r0['panel_id']}.png", IMAGE_BOX))
                fig.text((x0 - 1.5) / W, (y0 + CELL_MM / 2) / H,
                         labels.get(r0["true_class"], r0["true_class"]),
                         rotation=90, ha="right", va="center")
            else:
                r = by_img[img][wanted[j - 1]]
                ax.imshow(crop(kit / "panels" / f"{r['panel_id']}.png", OVERLAY_BOX))
                if r["correct"] != "True":
                    ax.text(0.03, 0.03, "Misclassified", transform=ax.transAxes,
                            ha="left", va="bottom",
                            bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": "none"})
            if i == 0:
                fig.text((x0 + CELL_MM / 2) / W, (H - 9.0) / H,
                         "H&E image" if j == 0 else COLUMNS[j - 1][2],
                         ha="center", va="bottom")
    fs.save(fig, Path(args.out), args.name, jpeg_quality=92)
    print(f"  {len(pick)} images x {len(COLUMNS)} maps: {', '.join(pick)}")


if __name__ == "__main__":
    main()
