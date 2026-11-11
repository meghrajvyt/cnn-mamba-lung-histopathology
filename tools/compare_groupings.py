#!/usr/bin/env python3
"""Compare our automatic LC25000 grouping with LC25000-clean's corrected groups.

LC25000-clean (Batchkala, Li & Rittscher, MICCAI DEMI 2024,
doi:10.1007/978-3-031-73748-0_2) released a manually corrected group id for
every image. This script measures how well our SIFT/RANSAC grouping
(lc25000_groups.json) agrees with it, and, more importantly, whether OUR
grouped train/val/test split is leakage-free under THEIR ground truth.

Usage:
    curl -sL -o lc25000_image_groups.csv \
      https://raw.githubusercontent.com/GeorgeBatch/LC25000-clean/main/kaggle/lc25000_image_groups.csv
    python tools/compare_groupings.py --theirs lc25000_image_groups.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data.dataset import grouped_split, load_group_assignments  # noqa: E402

CLASSES = ("lung_aca", "lung_n", "lung_scc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json_out", default=None,
                    help="Also write the numbers to this JSON file.")
    args = ap.parse_args()
    out = {"seed": args.seed}

    theirs = {}
    with open(args.theirs) as f:
        for r in csv.DictReader(f):
            if r["label"] in CLASSES:
                theirs[f"{r['label']}/{r['filename']}"] = (r["label"], int(r["group_id"]))
    ours = load_group_assignments()

    keys = sorted(set(theirs) & set(ours))
    print(f"images: theirs={len(theirs)} ours={len(ours)} matched={len(keys)}")
    if len(keys) != len(ours):
        print("  WARNING: filename sets differ; comparison covers matched images only")

    # Group counts per class
    for c in CLASSES:
        tg = {theirs[k][1] for k in keys if k.startswith(c + "/")}
        og = {ours[k] for k in keys if k.startswith(c + "/")}
        print(f"  {c:9s} groups: theirs={len(tg)} ours={len(og)}")

    # Agreement
    try:
        from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
        a = [theirs[k] for k in keys]
        b = [ours[k] for k in keys]
        a_ids = [f"{l}:{g}" for l, g in a]
        h, cpl, v = homogeneity_completeness_v_measure(a_ids, b)
        ari = adjusted_rand_score(a_ids, b)
        out.update(ari=ari, homogeneity=h, completeness=cpl, v_measure=v)
        print(f"\nagreement: ARI={ari:.4f} "
              f"homogeneity={h:.4f} completeness={cpl:.4f} V={v:.4f}")
        print("  (homogeneity<1: some of our groups mix several true tiles = over-merged;"
              "\n   completeness<1: some true tiles are split across our groups = leak risk)")
    except ImportError:
        pass

    # Leakage audit: does OUR split separate THEIR groups?
    label_idx = {c: i for i, c in enumerate(CLASSES)}
    samples = [(f"/x/{k}", label_idx[k.split("/")[0]]) for k in keys]
    tr, va, te, _ = grouped_split(samples, ours, 3, 0.7, 0.15, args.seed)
    split_of = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        for i in idx:
            split_of[keys[i]] = name
    spans = defaultdict(set)
    for k in keys:
        spans[theirs[k]].add(split_of[k])
    leaky = {g: s for g, s in spans.items() if len(s) > 1}
    tt = [g for g, s in leaky.items() if "train" in s and "test" in s]
    n_test_imgs = sum(1 for k in keys if split_of[k] == "test")
    leaked_test = sum(1 for k in keys if split_of[k] == "test" and theirs[k] in set(tt))
    print(f"\nleakage audit of OUR seed-{args.seed} split under THEIR ground truth:")
    print(f"  true tiles spanning >1 split : {len(leaky)} of {len(spans)}")
    print(f"  true tiles spanning train+test: {len(tt)}")
    print(f"  test images with a sibling in train: {leaked_test} of {n_test_imgs} "
          f"({100 * leaked_test / max(n_test_imgs, 1):.2f}%)")
    sizes = Counter(len(s) for s in spans.values())
    print(f"  (per-true-tile split counts: {dict(sizes)})")
    if args.json_out:
        import json
        out.update(matched_images=len(keys), true_tiles=len(spans),
                   tiles_spanning_splits=len(leaky), tiles_train_and_test=len(tt),
                   test_images=n_test_imgs, test_images_with_train_sibling=leaked_test,
                   test_fraction_with_train_sibling=leaked_test / max(n_test_imgs, 1))
        pathlib.Path(args.json_out).write_text(json.dumps(out, indent=1))
        print(f"written {args.json_out}")


if __name__ == "__main__":
    main()
