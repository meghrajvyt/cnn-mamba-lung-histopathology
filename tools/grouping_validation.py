#!/usr/bin/env python3
"""Validate the recovered LC25000 source groups (Supplementary Fig. S1 + stats).

Re-measures, with the exact verification rule of tools/group_lc25000.py
(SIFT at 384 px, at most 600 keypoints, ratio test 0.75, 4-DOF RANSAC, 5 px,
plain and mirrored orientation), the geometric-inlier counts of

  * sibling pairs: two images assigned to the same group, and
  * non-sibling pairs: two images of the same class in different groups,

for a seeded sample of pairs per class, and draws:
  (a) group-size distribution per class,
  (b) inlier distributions for sibling vs non-sibling pairs (log scale),
  (c) example groups (six members each).

Within-group pairs can legitimately fall below the acceptance threshold:
copies rotated far apart are joined through intermediate copies or the
merge pass rather than matched directly, so (b) reports the full distribution.

Outputs (in --out): figS1_grouping_stats.json (summary, cited in the paper),
figS1_grouping_data.json (per-pair inlier counts, group sizes, example
members) and figS1_grouping.pdf/.png. With --plot_only the figure is redrawn
from figS1_grouping_data.json without recomputing SIFT/RANSAC; the LC25000
images are still needed for panel c.

Usage:
    python3 tools/grouping_validation.py [--pairs 150] [--out docs/manuscript/figures]
    python3 tools/grouping_validation.py --plot_only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import figstyle as fs  # noqa: E402
from PIL import Image  # noqa: E402

CLASSES = ("lung_aca", "lung_n", "lung_scc")
CLASS_LABEL = {"lung_aca": "Adenocarcinoma", "lung_n": "Benign tissue",
               "lung_scc": "Squamous cell carcinoma"}
CLASS_COLOUR = {"lung_aca": fs.VERMILLION, "lung_n": fs.GREEN, "lung_scc": fs.BLUE}
THRESH = 15


def compute(data_root: Path, pairs: int, seed: int) -> dict:
    from data.dataset import load_group_assignments
    from group_lc25000 import _ransac_inliers, sift_descriptors

    rng = random.Random(seed)
    groups = load_group_assignments()
    stats = {"threshold": THRESH, "per_class": {}}
    inl = {"sibling": [], "non-sibling": []}
    sizes, examples = {}, {}
    for c in CLASSES:
        members = defaultdict(list)
        for k, g in groups.items():
            if k.startswith(c + "/"):
                members[g].append(k)
        sizes[c] = sorted(len(v) for v in members.values())
        gids = sorted(members)
        multi = [g for g in gids if len(members[g]) >= 2]
        sib = []
        for _ in range(pairs):
            g = rng.choice(multi)
            a, b = rng.sample(members[g], 2)
            sib.append((a, b))
        non = []
        for _ in range(pairs):
            g1, g2 = rng.sample(gids, 2)
            non.append((rng.choice(members[g1]), rng.choice(members[g2])))
        files = sorted({k for p in sib + non for k in p})
        idx = {k: i for i, k in enumerate(files)}
        plain, mirror = sift_descriptors([data_root / k for k in files], 384, 600)

        def count(p):
            # Best of plain and mirrored orientation, as verify() accepts a
            # pair if either reaches the threshold.
            i, j = idx[p[0]], idx[p[1]]
            return max(_ransac_inliers(plain[0][i], plain[1][i], plain[0][j], plain[1][j]),
                       _ransac_inliers(plain[0][i], plain[1][i], mirror[0][j], mirror[1][j]))

        s_in = [int(count(p)) for p in sib]
        n_in = [int(count(p)) for p in non]
        inl["sibling"] += s_in
        inl["non-sibling"] += n_in
        stats["per_class"][c] = {
            "groups": len(gids), "size_median": float(np.median(sizes[c])),
            "size_min": int(min(sizes[c])), "size_max": int(max(sizes[c])),
            "singletons": int(sum(s == 1 for s in sizes[c])),
            "sibling_inliers": {"mean": float(np.mean(s_in)), "median": float(np.median(s_in)),
                                "frac_ge_thresh": float(np.mean(np.array(s_in) >= THRESH))},
            "non_sibling_inliers": {"mean": float(np.mean(n_in)), "max": int(max(n_in)),
                                    "frac_ge_thresh": float(np.mean(np.array(n_in) >= THRESH))},
        }
        typical = [g for g in gids if 18 <= len(members[g]) <= 22]
        examples[c] = rng.sample(members[rng.choice(typical or multi)], 6)
        print(c, json.dumps(stats["per_class"][c]), flush=True)
    for kind in inl:
        v = np.array(inl[kind])
        stats[kind] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                       "max": int(v.max()), "frac_ge_thresh": float((v >= THRESH).mean())}
    return {"stats": stats, "inliers": inl, "sizes": sizes, "examples": examples,
            "pairs_per_class": pairs, "seed": seed}


def plot(data: dict, data_root: Path, out: Path) -> None:
    fs.apply()
    sizes, inl, examples = data["sizes"], data["inliers"], data["examples"]
    fig = fs.figure(180, 152)
    sub = fig.subfigures(2, 1, height_ratios=[1, 1.75], hspace=0.04)
    ax_a, ax_b = sub[0].subplots(1, 2, gridspec_kw={"wspace": 0.12})

    bins = np.arange(0.5, max(max(v) for v in sizes.values()) + 1.5, 2)
    for c in CLASSES:
        ax_a.hist(sizes[c], bins=bins, histtype="step", lw=fs.LINE_PT, color=CLASS_COLOUR[c],
                  label=f"{CLASS_LABEL[c]} ({len(sizes[c])} groups)")
    ax_a.axvline(20, color=fs.BLACK, ls=(0, (1, 1.5)), lw=fs.LINE_PT)
    ax_a.set_xlabel("Images per recovered group (expected 20)")
    ax_a.set_ylabel("Groups")
    peak = max(np.histogram(sizes[c], bins=bins)[0].max() for c in CLASSES)
    ax_a.set_ylim(0, 1.6 * peak)  # headroom for the legend above the bars
    ax_a.legend(loc="upper right")
    ax_a.set_title("Group sizes")

    top = max(max(inl["sibling"]), 10) + 1
    b = np.logspace(0, np.log10(top), 28)
    ax_b.hist(np.array(inl["non-sibling"]) + 1, bins=b, histtype="step", lw=fs.LINE_PT,
              color=fs.BLACK, label="Different groups")
    ax_b.hist(np.array(inl["sibling"]) + 1, bins=b, histtype="step", lw=fs.LINE_PT,
              color=fs.VERMILLION, label="Same group")
    ax_b.axvline(THRESH + 1, color=fs.BLACK, ls=(0, (3, 2)), lw=fs.LINE_PT)
    ax_b.set_xscale("log")
    ax_b.set_xlim(1, top)
    # Plain tick labels (1, 10, 100): 10^n labels would set the exponent in
    # smaller type, against the one-size rule.
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax_b.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax_b.xaxis.set_major_formatter(ScalarFormatter())
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("RANSAC inliers + 1 (log scale)")
    ax_b.set_ylabel("Pairs")
    ax_b.legend(loc="upper right")
    ax_b.set_title(f"Geometric verification ({data['pairs_per_class']} pairs per class and kind)")

    grid = sub[1].subplots(3, 6, gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for r, c in enumerate(CLASSES):
        for j, k in enumerate(examples[c]):
            ax = grid[r][j]
            im = Image.open(data_root / k).convert("RGB").resize((300, 300), Image.LANCZOS)
            ax.imshow(np.asarray(im))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(CLASS_LABEL[c].replace(" cell carcinoma", "\ncell carcinoma"))
    sub[1].suptitle("Six members of one recovered group per class")
    fs.panel_letters(fig, [ax_a, ax_b], "ab")
    fs.panel_letters(fig, [grid[0][0]], "c")
    fs.save(fig, out, "figS1_grouping", jpeg_quality=92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", default=str(ROOT / "data"))
    ap.add_argument("--pairs", type=int, default=150, help="pairs per class and kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "manuscript" / "figures"))
    ap.add_argument("--plot_only", action="store_true",
                    help="redraw from figS1_grouping_data.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "figS1_grouping_data.json"
    if args.plot_only:
        data = json.loads(cache.read_text())
    else:
        data = compute(Path(args.data_root), args.pairs, args.seed)
        print(json.dumps({k: data["stats"][k] for k in data["inliers"]}, indent=1))
        (out / "figS1_grouping_stats.json").write_text(json.dumps(data["stats"], indent=1))
        cache.write_text(json.dumps(data, indent=1))
    plot(data, Path(args.data_root), out)


if __name__ == "__main__":
    main()
