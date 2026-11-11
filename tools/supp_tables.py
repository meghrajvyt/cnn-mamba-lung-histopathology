#!/usr/bin/env python3
"""Supplementary tables for Paper 1, recomputed from predictions.

  S2  per-class precision, recall, F1 and one-vs-rest AUC
      (LC25000: mean over seeds; LungHist700: pooled out-of-fold)
  S3  calibration: expected calibration error (15 equal-width bins on the
      top-class probability) and multiclass Brier score
  S4  per-fold LungHist700 accuracy
  S7  Phikon probe: L2 strength C chosen on validation, per task and fold
  S8  LungHist700 learning-rate selection (fold 0, validation accuracy)

Image-level McNemar tests (S1) and the clean-subset sensitivity (S5) are in
tools/paper_tables.py (T2/T4/T5 and T7); faithfulness with the mean canvas
(S6) comes from the xai_metrics_meandel outputs.

Usage:
    python3 tools/supp_tables.py [--results results] [--out results/_paper]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from paper_tables import (BASELINES, LABEL, LC_ARCHS, LC_CLASSES, SECONDARY, Builder,  # noqa: E402
                          md_table, selected_hybrid)

C7 = ["aca_bd", "aca_md", "aca_pd", "nor", "scc_bd", "scc_md", "scc_pd"]


def ece(labels, probs, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(labels, probs) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def per_class(labels, preds, probs, n_classes) -> Dict[str, np.ndarray]:
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=range(n_classes),
                                                 zero_division=0)
    auc = []
    for c in range(n_classes):
        y = labels == c
        auc.append(roc_auc_score(y, probs[:, c]) if 0 < y.sum() < len(y) else np.nan)
    return {"precision": p, "recall": r, "f1": f, "auc": np.array(auc)}


def fmt(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{100 * v:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--splits", default=str(ROOT / "splits"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mag", default="all")
    args = ap.parse_args()
    R = Path(args.results)
    out = Path(args.out) if args.out else R / "_paper"
    out.mkdir(parents=True, exist_ok=True)
    sel = selected_hybrid(R)
    b = Builder(R, Path(args.splits), sel, None, args.mag)
    md: List[str] = [f"# Paper 1 supplementary tables (selected hybrid: {LABEL.get(sel, sel)})", ""]
    data: Dict = {}

    # ---------------------------------------------------- S2 / S3 LC25000
    s2, s3 = [], []
    for a in LC_ARCHS:
        runs = b.lc_runs(a)
        if not runs:
            continue
        pcs = [per_class(p["labels"], p["preds"], p["probs"], 3) for p in runs.values()]
        mean = {k: np.nanmean([pc[k] for pc in pcs], 0) for k in pcs[0]}
        for ci, c in enumerate(LC_CLASSES):
            s2.append(["LC25000", LABEL[a], c] + [fmt(mean[k][ci]) for k in
                                                  ("precision", "recall", "f1", "auc")])
        data.setdefault("S2", {})[f"lc25000/{a}"] = {
            k: dict(zip(LC_CLASSES, map(float, mean[k]))) for k in mean}
        e = [ece(p["labels"], p["probs"]) for p in runs.values()]
        br = [brier(p["labels"], p["probs"]) for p in runs.values()]
        s3.append(["LC25000", LABEL[a], str(len(runs)), f"{100 * np.mean(e):.2f}",
                   f"{np.mean(br):.4f}"])
        data.setdefault("S3", {})[f"lc25000/{a}"] = {"ece": e, "brier": br}

    # ------------------------------------------- S2 / S3 / S4 LungHist700
    s4 = []
    for task, classes, title in (("3c", list(LC_CLASSES), "LungHist700 3-class"),
                                 ("7c", C7, "LungHist700 7-class")):
        for a in [sel] + BASELINES + SECONDARY:
            r = b.lh_pooled(task, "patient", a)
            if r is None:
                continue
            pooled, accs, _ = r
            keys = sorted(pooled)
            y, p, pr = b.arrays(pooled, keys)
            pc = per_class(y, p, pr, len(classes))
            for ci, c in enumerate(classes):
                s2.append([title, LABEL[a], c] + [fmt(pc[k][ci]) for k in
                                                  ("precision", "recall", "f1", "auc")])
            s3.append([title, LABEL[a], "pooled", f"{100 * ece(y, pr):.2f}", f"{brier(y, pr):.4f}"])
            s4.append([title, LABEL[a]] + [f"{100 * x:.1f}" for x in accs])
            key = f"lh700_{task}/{a}"
            data.setdefault("S2", {})[key] = {
                k: dict(zip(classes, map(float, pc[k]))) for k in pc}
            data.setdefault("S3", {})[key] = {"ece": ece(y, pr), "brier": brier(y, pr)}
            data.setdefault("S4", {})[key] = [float(x) for x in accs]

    md += ["## S2. Per-class metrics (%)", "LC25000: mean over seeds; LungHist700: pooled "
           "out-of-fold, patient-level.", "",
           md_table(["Dataset", "Model", "Class", "Precision", "Recall", "F1", "AUC"], s2), "",
           "## S3. Calibration", "ECE: 15 equal-width bins on the top-class probability (%); "
           "Brier: multiclass, summed over classes, mean over images.", "",
           md_table(["Dataset", "Model", "Seeds", "ECE (%)", "Brier"], s3), "",
           "## S4. Per-fold LungHist700 accuracy (%), patient-level", "",
           md_table(["Task", "Model", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"], s4), ""]

    # ------------------------------------------------------- S7 Phikon C
    s7 = []
    for d in sorted(R.glob("*phikon_lp*")) + sorted(R.glob("external_*/*phikon_lp*")):
        f = d / "evaluation_results.json"
        if f.is_file():
            pr = json.loads(f.read_text()).get("probe", {})
            s7.append([str(d.relative_to(R)), str(pr.get("C", "—")),
                       fmt(pr.get("val_accuracy"))])
            data.setdefault("S7", {})[str(d.relative_to(R))] = {
                "C": pr.get("C"), "val_accuracy": pr.get("val_accuracy")}
    md += ["## S7. Phikon linear probe: L2 strength chosen on validation", "",
           md_table(["Run", "C", "Validation accuracy (%)"], s7), ""]

    # ------------------------------------------------ S8 LR selection
    s8 = []
    lf = R.parent / "logs" / "lh700_lr_selection.txt"
    if lf.is_file():
        seen = set()
        for line in lf.read_text().splitlines():
            # The queue appended the record again when it was re-run; the
            # repeated lines carry the same values (tools/paper_tables.py
            # Builder.lr_records checks this), so list each once.
            entry = re.sub(r" \(\d\d:\d\d UTC\)$", "", line)
            if entry in seen:
                continue
            seen.add(entry)
            m = re.match(r"# (\S+) lr=(\S+) best_val_acc=(\S+)", line)
            if m:
                s8.append([LABEL.get(m.group(1), m.group(1)), m.group(2),
                           fmt(float(m.group(3)))])
                data.setdefault("S8", {}).setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
            m = re.match(r"selected: (\S+) lr=(\S+)", line)
            if m:
                s8.append([LABEL.get(m.group(1), m.group(1)), f"**selected {m.group(2)}**", ""])
    md += ["## S8. LungHist700 learning-rate selection (fold 0, 3-class, patient-level)",
           "Validation accuracy only; ties keep 1e-4.", "",
           md_table(["Model", "Learning rate", "Best validation accuracy (%)"], s8)
           if s8 else "Not yet run.", ""]

    (out / "supp_tables.md").write_text("\n".join(md))
    (out / "supp_tables.json").write_text(json.dumps(data, indent=1))
    print(f"written {out / 'supp_tables.md'}")


if __name__ == "__main__":
    main()
