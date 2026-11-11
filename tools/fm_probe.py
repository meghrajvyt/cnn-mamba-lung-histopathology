#!/usr/bin/env python3
"""Pathology foundation-model reference: Phikon linear probe.

Phikon (Owkin; ViT-B/16 pretrained with iBOT on 40M TCGA tiles; Filiot et al.,
medRxiv 2023; Owkin non-commercial licence, academic use) is used frozen: the
CLS embedding of each image (224 x 224, ImageNet normalisation, the same eval
transform as every other model) feeds a standardised multinomial logistic
regression. The regularisation strength is chosen on the validation split of
each task (grid below, ties to the stronger penalty); test data are used only
for the final prediction. Neither LC25000 (VA hospital) nor LungHist700
(Valladolid) is part of TCGA, so pretraining does not contaminate the tests.

Every split of the trained models is reproduced exactly, and predictions are
written in the same order and format as evaluate.py, so tools/paper_tables.py
treats the probe like any other model:

  results/phikon_lp_d4_224px_s{42,1337,2024}/        LC25000 grouped split
  results/leaky_phikon_lp_d4_224px_s42/              LC25000 per-file split
  results/external_{20x,40x}/phikon_lp_d4_224px_s*/  LC25000 -> LungHist700
  results/lh700_{patient,image}all_f{k}_phikon_lp/   LungHist700 3-class CV
  results/lh700_7c_patientall_f{k}_phikon_lp/        LungHist700 7-class CV

Needs `transformers` (kept outside the training venv on AWS):
    PYTHONPATH=~/fm_pkgs python3 tools/fm_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import grouped_split, load_group_assignments  # noqa: E402
from data.transforms import HistopathologyEvalTransform  # noqa: E402

LC_CLASSES = ["lung_aca", "lung_n", "lung_scc"]
C7 = ["aca_bd", "aca_md", "aca_pd", "nor", "scc_bd", "scc_md", "scc_pd"]
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
NAME = "phikon_lp"


# ------------------------------------------------------------ features

class Files(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[Path]):
        self.paths, self.tf = list(paths), HistopathologyEvalTransform(224)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def embed(paths: Sequence[Path], device, batch: int, workers: int) -> np.ndarray:
    from transformers import ViTModel
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(device).eval()
    dl = torch.utils.data.DataLoader(Files(paths), batch_size=batch, num_workers=workers)
    out, t0 = [], time.time()
    for i, x in enumerate(dl):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model(pixel_values=x.to(device)).last_hidden_state[:, 0]
        out.append(h.float().cpu().numpy())
        if i % 20 == 0:
            print(f"  embedded {min((i + 1) * batch, len(paths))}/{len(paths)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out)


def cached(tag: str, keys: List[str], root: Path, cache: Path, device, args) -> Dict[str, np.ndarray]:
    f = cache / f"phikon_{tag}.npz"
    if f.is_file():
        z = np.load(f, allow_pickle=True)
        feats = dict(zip(z["keys"].tolist(), z["feats"]))
        if all(k in feats for k in keys):
            return feats
    print(f"embedding {len(keys)} images from {root}", flush=True)
    F = embed([root / k for k in keys], device, args.batch, args.workers)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, keys=np.array(keys), feats=F)
    return dict(zip(keys, F))


# --------------------------------------------------------------- probe

def fit_predict(Xtr, ytr, Xva, yva, Xte, n_classes):
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in C_GRID:  # ascending: ties keep the stronger penalty
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        acc = float((clf.predict(sc.transform(Xva)) == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    acc, C, clf = best
    probs = np.zeros((len(Xte), n_classes))
    probs[:, clf.classes_] = clf.predict_proba(sc.transform(Xte))
    return probs, {"C": C, "val_accuracy": acc}, (sc, clf)


def save(run_dir: Path, labels, probs, class_names, info):
    run_dir.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(1)
    np.savez(run_dir / "predictions.npz", labels=np.asarray(labels), preds=preds,
             probs=probs, class_names=np.array(class_names), checkpoint=np.array(NAME))
    acc = float((preds == np.asarray(labels)).mean())
    (run_dir / "evaluation_results.json").write_text(json.dumps(
        {"metrics": {"accuracy": acc}, "probe": info, "model": "owkin/phikon (frozen) + "
         "logistic regression"}, indent=1))
    print(f"  {run_dir.name}: acc {acc:.4f} (C={info['C']}, val {info['val_accuracy']:.4f})",
          flush=True)


def lab(keys, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[k.split("/")[0]] for k in keys])


def stack(feats, keys):
    return np.stack([feats[k] for k in keys])


def read_split(f: Path):
    m = json.loads(f.read_text())
    m = m.get("assignments", m)
    return [sorted(k for k, v in m.items() if v == s) for s in ("train", "val", "test")]


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--mag", default="all")
    ap.add_argument("--lh_root", default=str(ROOT / "data_lh700_512"),
                    help="LungHist700 image root (3-class layout), the same copy "
                         "the trained models read")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    R = Path(args.results)
    cache = R / "_fm"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # LC25000 ------------------------------------------------------------
    groups = load_group_assignments()
    keys = sorted(groups)
    y_all = lab(keys, LC_CLASSES)
    feats = cached("lc25000", keys, ROOT / "data", cache, device, args)
    X_all = stack(feats, keys)
    samples = [(f"/x/{k}", int(y)) for k, y in zip(keys, y_all)]

    lh_feats = None
    ext = {}
    for mag in ("20x", "40x"):
        f = ROOT / "splits" / "lunghist700" / f"external_{mag}.json"
        if f.is_file():
            ext[mag] = read_split(f)[2]
    fold0 = read_split(ROOT / "splits/lunghist700" / f"patient_{args.mag}_fold0.json")
    lh_keys = sorted({k for v in ext.values() for k in v} | {k for s in fold0 for k in s})
    lh_root = Path(args.lh_root)
    lh_feats = cached(f"lunghist700_{lh_root.name}", lh_keys, lh_root, cache, device, args)

    for seed in args.seeds:
        tr, va, te = grouped_split(samples, groups, 3, 0.7, 0.15, seed)[:3]
        probs, info, (sc, clf) = fit_predict(X_all[tr], y_all[tr], X_all[va], y_all[va],
                                             X_all[te], 3)
        save(R / f"{NAME}_d4_224px_s{seed}", y_all[te], probs, LC_CLASSES, info)
        for mag, ek in ext.items():  # external test with this LC25000 probe
            pe = np.zeros((len(ek), 3))
            pe[:, clf.classes_] = clf.predict_proba(sc.transform(stack(lh_feats, ek)))
            save(R / f"external_{mag}" / f"{NAME}_d4_224px_s{seed}", lab(ek, LC_CLASSES),
                 pe, LC_CLASSES, {**info, "external": mag})

    # per-file (leaky) split, seed 42, replicating data/dataset.py
    ind = list(range(len(keys)))
    # Same float expressions as data/dataset.py (1 - 0.7 = 0.30000000000000004
    # gives a 4,501-image held-out set), so the split matches the trained runs.
    train_ratio, val_ratio = 0.7, 0.15
    tr, temp = next(StratifiedShuffleSplit(1, test_size=1 - train_ratio, random_state=42)
                    .split(ind, y_all))
    rel = (1 - train_ratio - val_ratio) / (1 - train_ratio)
    va_rel, te_rel = next(StratifiedShuffleSplit(1, test_size=rel, random_state=42)
                          .split(temp, y_all[temp]))
    va, te = temp[va_rel], temp[te_rel]
    probs, info, _ = fit_predict(X_all[tr], y_all[tr], X_all[va], y_all[va], X_all[te], 3)
    save(R / f"leaky_{NAME}_d4_224px_s42", y_all[te], probs, LC_CLASSES, info)

    # LungHist700 CV -----------------------------------------------------
    by_name = {Path(k).name: v for k, v in lh_feats.items()}
    for task, sdir, classes, protos in (
            ("3c", "lunghist700", LC_CLASSES, ("patient", "image")),
            ("7c", "lunghist700_7c", C7, ("patient",))):
        for proto in protos:
            for k in range(5):
                f = ROOT / "splits" / sdir / f"{proto}_{args.mag}_fold{k}.json"
                tr_k, va_k, te_k = read_split(f)
                X = lambda ks: np.stack([by_name[Path(x).name] for x in ks])
                probs, info, _ = fit_predict(X(tr_k), lab(tr_k, classes), X(va_k),
                                             lab(va_k, classes), X(te_k), len(classes))
                name = (f"lh700_{proto}{args.mag}_f{k}_{NAME}" if task == "3c"
                        else f"lh700_7c_{proto}{args.mag}_f{k}_{NAME}")
                save(R / name, lab(te_k, classes), probs, classes, info)
    print("done")


if __name__ == "__main__":
    main()
