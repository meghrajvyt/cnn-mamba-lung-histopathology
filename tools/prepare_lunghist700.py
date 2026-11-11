#!/usr/bin/env python3
"""Prepare LungHist700 for training and external testing.

LungHist700 (Diosdado et al., Scientific Data 11:1088, 2024; CC BY 4.0,
doi:10.6084/m9.figshare.25459174): 691 lung histology images (1600x1200) from
45 patients, at 20x and 40x, superclasses aca / scc / nor with differentiation
subclasses. data.csv gives superclass, subclass, resolution, image_id and
patient_id for every image.

What this does
--------------
1. Maps superclasses onto LC25000's class names so one model can be tested on
   both datasets: aca -> lung_aca, nor -> lung_n, scc -> lung_scc.
2. Centre-crops each image to 1200x1200 (avoids squashing 4:3 histology to 1:1
   at resize time) and writes <out>/<class>/<original filename>.
3. Writes split files (data/dataset.py ``split_file`` format) under --splits:
     patient_<mag>_fold<k>.json  5-fold CV, patients never cross splits
     image_<mag>_fold<k>.json    5-fold CV over images, ignoring patients
                                 (the leaky protocol, for measuring inflation)
     external_<mag>.json         every image -> "test" (cross-dataset testing)
   <mag> is 20x, 40x or all. Within each fold, validation is a further
   patient-level (or image-level) split of the training portion.

Patients appear in several classes (9 of 45 have both tumour and normal
tissue), so splitting is done with StratifiedGroupKFold on patients rather
than the per-class group partitioning used for LC25000.

--granularity sub keeps the seven differentiation classes instead (aca_bd,
aca_md, aca_pd, nor, scc_bd, scc_md, scc_pd; bd/md/pd = well / moderately /
poorly differentiated), with folds stratified on those seven labels. No
external split is written in that mode, since LC25000 has no grades.

Usage:
    python tools/prepare_lunghist700.py --src ~/lh700/x/data \
        --out ./data_lh700 --splits ./splits/lunghist700
    python tools/prepare_lunghist700.py --src ~/lh700/x/data --granularity sub \
        --out ./data_lh700_7c --splits ./splits/lunghist700_7c
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CLASS_MAP = {"aca": "lung_aca", "nor": "lung_n", "scc": "lung_scc"}
N_FOLDS = 5


def source_path(src: pathlib.Path, r: dict) -> pathlib.Path:
    sub = r["subclass"]
    folder = r["superclass"] + (f"_{sub}" if sub else "")
    stem = "_".join([r["superclass"]] + ([sub] if sub else [])
                    + [r["resolution"], r["image_id"]])
    return src / "images" / folder / f"{stem}.jpg"


def centre_crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def make_folds(keys, labels, groups, seed, by_patient):
    """Yield (train, val, test) key lists for each of N_FOLDS folds."""
    keys, labels, groups = map(np.asarray, (keys, labels, groups))
    outer = (StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
             if by_patient else StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed))
    split_args = (keys, labels, groups) if by_patient else (keys, labels)
    for rest, test in outer.split(*split_args):
        # Validation: one quarter of the remainder, split the same way.
        inner = (StratifiedGroupKFold(4, shuffle=True, random_state=seed)
                 if by_patient else StratifiedKFold(4, shuffle=True, random_state=seed))
        inner_args = ((keys[rest], labels[rest], groups[rest]) if by_patient
                      else (keys[rest], labels[rest]))
        tr_i, va_i = next(inner.split(*inner_args))
        yield keys[rest][tr_i].tolist(), keys[rest][va_i].tolist(), keys[test].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with data.csv and images/")
    ap.add_argument("--out", default="./data_lh700")
    ap.add_argument("--splits", default="./splits/lunghist700")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--granularity", choices=("super", "sub"), default="super",
                    help="'super': 3 classes named as in LC25000; 'sub': the 7 "
                         "differentiation classes")
    ap.add_argument("--size", type=int, default=0,
                    help="Resize the square crop to SIZE x SIZE (bilinear, "
                         "antialiased); 0 keeps 1200 x 1200. Models see 224 x 224 "
                         "either way; a smaller stored copy only makes JPEG "
                         "decoding cheaper.")
    args = ap.parse_args()

    src, out, splits = map(pathlib.Path, (args.src, args.out, args.splits))
    rows = list(csv.DictReader(open(src / "data.csv")))
    splits.mkdir(parents=True, exist_ok=True)

    meta = {}
    for r in rows:
        p = source_path(src, r)
        if args.granularity == "super":
            cls = CLASS_MAP[r["superclass"]]
        else:  # the source folder name: aca_bd, ..., nor, ..., scc_pd
            cls = r["superclass"] + (f"_{r['subclass']}" if r["subclass"] else "")
        key = f"{cls}/{p.name}"
        dst = out / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            im = centre_crop_square(Image.open(p).convert("RGB"))
            if args.size:
                im = im.resize((args.size, args.size), Image.BILINEAR)
            im.save(dst, format="JPEG", quality=args.quality)
        meta[key] = {"patient_id": int(r["patient_id"]), "class": cls,
                     "superclass": r["superclass"], "subclass": r["subclass"],
                     "resolution": r["resolution"]}
    (splits / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"prepared {len(meta)} images -> {out}")

    class_idx = {c: i for i, c in enumerate(sorted({m["class"] for m in meta.values()}))}
    print(f"classes: {class_idx}")
    for mag in ("20x", "40x", "all"):
        sel = sorted(k for k, m in meta.items() if mag == "all" or m["resolution"] == mag)
        labels = [class_idx[meta[k]["class"]] for k in sel]
        groups = [meta[k]["patient_id"] for k in sel]

        if args.granularity == "super":
            (splits / f"external_{mag}.json").write_text(
                json.dumps({k: "test" for k in sel}, indent=0))

        for by_patient, tag in ((True, "patient"), (False, "image")):
            for fold, (tr, va, te) in enumerate(
                    make_folds(sel, labels, groups, args.seed, by_patient)):
                mapping = {**{k: "train" for k in tr}, **{k: "val" for k in va},
                           **{k: "test" for k in te}}
                (splits / f"{tag}_{mag}_fold{fold}.json").write_text(
                    json.dumps(mapping, indent=0))

                pid = lambda ks: {meta[k]["patient_id"] for k in ks}
                p_tr, p_va, p_te = pid(tr), pid(va), pid(te)
                leaked = sum(1 for k in te if meta[k]["patient_id"] in p_tr)
                cls_te = Counter(meta[k]["class"] for k in te)
                if by_patient:
                    assert not (p_tr & p_te or p_tr & p_va or p_va & p_te), \
                        f"patient leakage in {tag}_{mag}_fold{fold}"
                print(f"  {tag:7s} {mag:3s} fold{fold}: train {len(tr):3d} / val {len(va):3d} / "
                      f"test {len(te):3d} | test patients {len(p_te):2d} | "
                      f"test imgs with patient in train {leaked:3d} | "
                      f"test classes {dict(sorted(cls_te.items()))}")
    print(f"split files -> {splits}")


if __name__ == "__main__":
    main()
