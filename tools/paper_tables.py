#!/usr/bin/env python3
"""Build every Paper 1 results table from the evaluated runs.

Reads results/<run>/predictions.npz (labels, preds, probs) and recomputes all
metrics from predictions, so every table uses one definition. Tables:

  T1  LC25000, leakage-controlled split: mean (range) over seeds per model
  T2  LC25000 paired tests: selected hybrid vs each baseline, per seed
      (McNemar, paired bootstrap CI of the accuracy difference, Holm across
      the baselines within each seed)
  T3  Leakage inflation: LC25000 per-file vs grouped (seed 42, different test
      sets, unpaired bootstrap); LungHist700 image- vs patient-level folds
      (same images, paired)
  T4  LungHist700 3-class, patient-level 5-fold CV: pooled out-of-fold
      metrics, per-fold accuracy mean (range), paired tests vs the hybrid
  T5  LungHist700 7-class grading: as T4, plus accuracy after collapsing
      grades to the three superclasses
  T6  External test LC25000 -> LungHist700 at 20x and 40x
  T7  Sensitivity: LC25000 accuracy on the strictly clean test subset
      (test images whose LC25000-clean tile has no member in training)
  T8  Explainability: faithfulness vs random; pathway attribution
  T9  Computational cost (tools/efficiency.py)

Added after the pre-submission review (docs/added_analyses_protocol.md):
VMamba-T in T1, T2, T4-T7 and T9 as a secondary comparator (paired tests
outside the Holm family: T2_secondary, T4/T5 secondary_tests), and
  T10 LungHist700 learning-rate sensitivity (A2)
  T11 Faithfulness against a 7 x 7 random map (A3)

Image identity is reconstructed from the split definitions (the dataset
orders samples by "<class>/<file>"), and checked against the stored labels,
so pooled and paired analyses align images exactly.

Usage:
    python3 tools/paper_tables.py [--results results] [--out results/_paper]
        [--lc25000_clean_csv lc25000_image_groups.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compare_models import mcnemar, paired_accuracy_diff_ci  # noqa: E402
from data.dataset import grouped_split, load_group_assignments  # noqa: E402

LABEL = {
    "hybrid_res": "Hybrid-res", "hybrid_res_xattn": "Hybrid-res-xattn",
    "grouped": "Original hybrid", "cnn_only": "CNN pathway",
    "effnet_b0": "EfficientNet-B0", "vit_small": "ViT-S/16", "swin_tiny": "Swin-T",
    "phikon_lp": "Phikon linear probe", "vmamba_tiny": "VMamba-T",
    "hybrid": "Original hybrid",  # key used by tools/efficiency.py
}
LC_ARCHS = ["hybrid_res", "hybrid_res_xattn", "grouped", "cnn_only",
            "effnet_b0", "vit_small", "swin_tiny", "phikon_lp", "vmamba_tiny"]
# Comparators of the selected hybrid (the Holm family): four trained
# baselines and the frozen pathology foundation-model probe.
BASELINES = ["cnn_only", "effnet_b0", "vit_small", "swin_tiny", "phikon_lp"]
# Added after the pre-submission review (docs/added_analyses_protocol.md, A1):
# compared with the selected hybrid as a secondary comparison, outside the
# Holm family, which stays as pre-specified.
SECONDARY = ["vmamba_tiny"]
# A2: the learning rate added to the LungHist700 grid, and the run-name
# prefix of the runs repeated at it (scripts/queue_review.sh).
SENS_LR = "3e-3"
SENS_PREFIX = f"lh700_lr{SENS_LR}_"
LC_CLASSES = ("lung_aca", "lung_n", "lung_scc")
METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "auc_macro", "mcc")
SHORT = {"accuracy": "Acc", "balanced_accuracy": "BalAcc", "f1_macro": "F1",
         "auc_macro": "AUC", "mcc": "MCC"}


# ----------------------------------------------------------------- loading

def load_pred(d: Path) -> Optional[Dict[str, np.ndarray]]:
    f = d / "predictions.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=True)
    return {"labels": z["labels"].astype(int), "preds": z["preds"].astype(int),
            "probs": z["probs"].astype(float)}


def metrics(labels, preds, probs) -> Dict[str, float]:
    out = {"n": int(len(labels)),
           "accuracy": float((labels == preds).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
           "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
           "mcc": float(matthews_corrcoef(labels, preds))}
    try:
        out["auc_macro"] = float(roc_auc_score(labels, probs, multi_class="ovr",
                                               average="macro"))
    except ValueError:
        out["auc_macro"] = float("nan")
    return out


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def cluster_paired(ca: np.ndarray, cb: np.ndarray, clusters: Sequence,
                   n: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Cluster-aware paired comparison of two correctness vectors.

    Images are not independent: LC25000 test images come in sibling groups of
    ~20 augmented copies of one tissue field, and LungHist700 images in
    patients. The independent unit is the cluster, so:
      * CI: percentile bootstrap resampling whole clusters;
      * p: sign-flip randomisation test on per-cluster differences in the
        number of correct predictions (exchangeability of the two models'
        labels within each cluster under H0).
    """
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    d = np.bincount(inv, weights=(ca.astype(float) - cb.astype(float)), minlength=len(uniq))
    size = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    boots = d[idx].sum(1) / size[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    t_obs = abs(d.sum())
    signs = rng.choice([-1.0, 1.0], size=(n, len(uniq)))
    t_perm = np.abs((signs * d).sum(1))
    p = (1 + np.sum(t_perm >= t_obs - 1e-12)) / (n + 1)
    return {"n_clusters": int(len(uniq)), "ci": [float(lo), float(hi)], "p": float(p)}


def cluster_acc_ci(c: np.ndarray, clusters: Sequence, n: int = 10000,
                   seed: int = 42) -> List[float]:
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = np.bincount(inv, weights=c.astype(float), minlength=len(uniq))
    size = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n, len(uniq)))
    b = k[idx].sum(1) / size[idx].sum(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def fmt_p(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pct(x: Optional[float], nd: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def mean_range(vals: List[float]) -> str:
    """'97.15 (95.63–98.10)': mean and range in %. With three seeds or five
    folds a range says more than a standard deviation (Scientific Reports
    statistical guidelines)."""
    if not vals:
        return "—"
    v = np.array(vals) * 100
    if len(v) == 1:
        return f"{v[0]:.2f}"
    return f"{v.mean():.2f} ({v.min():.2f}–{v.max():.2f})"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------- image identities

def lc25000_test_keys(seed: int = 42) -> List[str]:
    """Test-set keys in prediction order for the grouped LC25000 split."""
    groups = load_group_assignments()
    keys = sorted(groups)
    idx = {c: i for i, c in enumerate(LC_CLASSES)}
    samples = [(f"/x/{k}", idx[k.split("/")[0]]) for k in keys]
    _, _, te, _ = grouped_split(samples, groups, 3, 0.7, 0.15, seed)
    return [keys[i] for i in te]


def lc25000_leaky_test_keys(seed: int = 42) -> List[str]:
    """Test keys of the per-file split, replicating data/dataset.py."""
    from sklearn.model_selection import StratifiedShuffleSplit
    keys = sorted(load_group_assignments())
    idx = {c: i for i, c in enumerate(LC_CLASSES)}
    labels = [idx[k.split("/")[0]] for k in keys]
    ind = list(range(len(keys)))
    # Same float expressions as data/dataset.py: 1 - 0.7 is 0.30000000000000004,
    # which makes sklearn round the held-out set up to 4,501 images.
    train_ratio, val_ratio = 0.7, 0.15
    tr, temp = next(StratifiedShuffleSplit(1, test_size=1 - train_ratio, random_state=seed)
                    .split(ind, labels))
    rel = (1 - train_ratio - val_ratio) / (1 - train_ratio)
    _, te_rel = next(StratifiedShuffleSplit(1, test_size=rel, random_state=seed)
                     .split(temp, [labels[i] for i in temp]))
    return [keys[i] for i in temp[te_rel]]


def split_test_keys(split_file: Path) -> List[str]:
    m = json.loads(split_file.read_text())
    m = m.get("assignments", m)
    return sorted(k for k, v in m.items() if v == "test")


def keyed(pred: Dict, keys: List[str], classes: List[str]) -> Dict[str, tuple]:
    """Attach keys to predictions, checking labels match the key's class."""
    if len(keys) != len(pred["labels"]):
        raise ValueError(f"{len(keys)} keys vs {len(pred['labels'])} predictions")
    cidx = {c: i for i, c in enumerate(classes)}
    exp = np.array([cidx[k.split("/")[0]] for k in keys])
    if not np.array_equal(exp, pred["labels"]):
        raise ValueError("label order does not match the split definition")
    return {k: (int(pred["labels"][i]), int(pred["preds"][i]), pred["probs"][i])
            for i, k in enumerate(keys)}


# ------------------------------------------------------------------ tables

class Builder:
    def __init__(self, results: Path, splits: Path, selected: str,
                 clean_csv: Optional[Path], mag: str = "all"):
        self.R, self.S, self.sel, self.mag = results, splits, selected, mag
        self.clean_csv = clean_csv
        self.md: List[str] = []
        self.data: Dict = {"selected": selected}

    def lc_runs(self, arch: str, leaky: bool = False) -> Dict[int, Dict]:
        prefix = ("leaky_" if leaky else "") + arch
        out = {}
        for d in sorted(self.R.glob(f"{prefix}_d4_224px_s*")):
            m = re.fullmatch(rf"{re.escape(prefix)}_d4_224px_s(\d+)", d.name)
            p = load_pred(d) if m else None
            if p is not None:
                out[int(m.group(1))] = p
        return out

    # T1
    def t1(self):
        rows, store = [], {}
        for a in LC_ARCHS:
            runs = self.lc_runs(a)
            if not runs:
                continue
            per = {s: metrics(**p) for s, p in runs.items()}
            store[a] = per
            rows.append([LABEL[a] + (" (selected)" if a == self.sel else ""),
                         str(len(per))] +
                        [mean_range([m[k] for m in per.values()]) for k in METRICS])
        self.data["T1"] = store
        self.md += ["## T1. LC25000, leakage-controlled split (test size varies with the "
                    "seed's group partition)",
                    "Mean (minimum–maximum) over seeds, in %.", "",
                    md_table(["Model", "Seeds"] + [SHORT[k] for k in METRICS], rows), ""]

    # T2
    def baselines(self) -> List[str]:
        return [b for b in BASELINES if b != self.sel]

    def t2(self):
        hyb = self.lc_runs(self.sel)
        rows, store, store_sec = [], {}, {}
        for seed in sorted(hyb):
            res, sec = [], []
            # The grouped split depends on the seed, so rebuild its test keys.
            keys = lc25000_test_keys(seed)
            if len(keys) != len(hyb[seed]["labels"]):
                continue
            groups = load_group_assignments()
            clusters = [groups[k] for k in keys]
            for b in self.baselines() + SECONDARY:
                other = self.lc_runs(b).get(seed)
                if other is None or len(other["labels"]) != len(keys):
                    continue
                ca = hyb[seed]["preds"] == hyb[seed]["labels"]
                cb = other["preds"] == other["labels"]
                (sec if b in SECONDARY else res).append(
                    {"baseline": b, "diff": float(ca.mean() - cb.mean()),
                     "cluster": cluster_paired(ca, cb, clusters),
                     "ci_image": paired_accuracy_diff_ci(ca, cb),
                     "mcnemar": mcnemar(ca, cb)})
            for r, ph in zip(res, holm([r["cluster"]["p"] for r in res])):
                r["p_holm"] = ph
            for r in sec:  # outside the family: no adjustment
                r["p_holm"] = None
            for r in res + sec:
                c = r["cluster"]
                rows.append([str(seed), LABEL[r["baseline"]],
                             f"{100 * r['diff']:+.2f}",
                             f"[{100 * c['ci'][0]:+.2f}, {100 * c['ci'][1]:+.2f}]",
                             fmt_p(c["p"]),
                             "secondary" if r["p_holm"] is None else fmt_p(r["p_holm"]),
                             f"{r['mcnemar']['n10']} / {r['mcnemar']['n01']}",
                             fmt_p(r["mcnemar"]["p_value"])])
            store[seed] = res
            if sec:
                store_sec[seed] = sec
        self.data["T2"] = store
        if store_sec:
            self.data["T2_secondary"] = store_sec
        self.md += [f"## T2. LC25000 paired tests: {LABEL.get(self.sel, self.sel)} vs baselines",
                    "Δ = hybrid − baseline accuracy (points). Primary inference treats the "
                    "source-image group as the unit (test images are ~20 augmented copies of "
                    "~107 tissue fields): group bootstrap 95% CI and group sign-flip "
                    "randomisation p, Holm-adjusted across baselines within each seed. "
                    "Image-level McNemar (discordant = only-hybrid / only-baseline correct) "
                    "is shown for reference; it treats siblings as independent and so "
                    "overstates significance. 'secondary': VMamba-T, added after the "
                    "pre-submission review, is compared outside the Holm family (p "
                    "unadjusted).", "",
                    md_table(["Seed", "Baseline", "Δ acc", "Group 95% CI", "Group p",
                              "Holm p", "Discordant", "McNemar p"], rows), ""]

    # LungHist700 pooled out-of-fold predictions
    def lh_pooled(self, task: str, proto: str, arch: str, prefix: str = "lh700_"):
        """task '3c' or '7c'. Returns (keyed dict, per-fold acc list) or None.
        prefix: SENS_PREFIX for the runs repeated at the sensitivity rate."""
        mag = self.mag
        name = (f"{prefix}{proto}{mag}_f{{}}_{arch}" if task == "3c"
                else f"{prefix}7c_{proto}{mag}_f{{}}_{arch}")
        sdir = self.S / ("lunghist700" if task == "3c" else "lunghist700_7c")
        classes = (list(LC_CLASSES) if task == "3c" else
                   ["aca_bd", "aca_md", "aca_pd", "nor", "scc_bd", "scc_md", "scc_pd"])
        pooled, accs = {}, []
        for f in range(5):
            p = load_pred(self.R / name.format(f))
            if p is None:
                return None
            keys = split_test_keys(sdir / f"{proto}_{mag}_fold{f}.json")
            pooled.update(keyed(p, keys, classes))
            accs.append(float((p["preds"] == p["labels"]).mean()))
        return pooled, accs, classes

    @staticmethod
    def arrays(pooled: Dict, keys: List[str]):
        y = np.array([pooled[k][0] for k in keys])
        p = np.array([pooled[k][1] for k in keys])
        pr = np.stack([pooled[k][2] for k in keys])
        return y, p, pr

    def lh_model(self, task: str, arch: str, prefix: str = "lh700_"):
        """Pooled patient-level metrics of one model, or None if a fold is
        missing. Returns (keys, y, p, metrics, per-fold accuracies, row)."""
        r = self.lh_pooled(task, "patient", arch, prefix)
        if r is None:
            return None
        pooled, accs, classes = r
        keys = sorted(pooled)
        y, p, pr = self.arrays(pooled, keys)
        m = metrics(y, p, pr)
        patients = self.patients(task, keys)
        m["accuracy_ci_patient"] = cluster_acc_ci(p == y, patients)
        lo, hi = m["accuracy_ci_patient"]
        # Sensitivity: drop patients whose images were the validation set
        # used to choose learning rates (3-class fold 0), since they also
        # appear in other folds' test sets.
        sel = self.lr_selection_patients()
        keep = np.array([pt not in sel for pt in patients])
        m["accuracy_excl_lr_selection"] = float((p[keep] == y[keep]).mean()) if keep.any() else None
        m["n_excl_lr_selection"] = int(keep.sum())
        if m["accuracy_excl_lr_selection"] is not None:
            m["excl_lr_selection_shift"] = m["accuracy_excl_lr_selection"] - m["accuracy"]
        nc = len(classes)
        m["confusion"] = np.bincount(y * nc + p, minlength=nc * nc).reshape(nc, nc).tolist()
        row = ([LABEL[arch], str(m["n"])] + [pct(m[k]) for k in METRICS] +
               [f"[{100 * lo:.1f}, {100 * hi:.1f}]", mean_range(accs),
                pct(m["accuracy_excl_lr_selection"])])
        if task == "7c":
            sup = lambda v: np.array([classes[i].split("_")[0] for i in v])
            m["superclass_accuracy"] = float((sup(y) == sup(p)).mean())
            # Errors that keep the subtype but miss the grade.
            err = p != y
            m["grade_only_error_fraction"] = (float((sup(y)[err] == sup(p)[err]).mean())
                                              if err.any() else None)
            m["grade_only_error_rate"] = float((err & (sup(y) == sup(p))).mean())
            row.append(pct(m["superclass_accuracy"]))
        return keys, y, p, m, accs, row

    def lh_tests(self, task: str, ref, others: Dict[str, tuple]):
        """Paired tests of the selected hybrid (ref = (keys, y, p)) against each
        model in `others` ({arch: (keys, y, p)}): the family of baselines with
        Holm, and the secondary comparators without adjustment."""
        tests, sec = [], []
        for a, (keys, y, p) in others.items():
            if keys != ref[0]:
                continue
            ca, cb = ref[2] == ref[1], p == y
            (sec if a in SECONDARY else tests).append(
                {"baseline": a, "diff": float(ca.mean() - cb.mean()),
                 "cluster": cluster_paired(ca, cb, self.patients(task, keys)),
                 "mcnemar": mcnemar(ca, cb)})
        for t, ph in zip(tests, holm([t["cluster"]["p"] for t in tests])):
            t["p_holm"] = ph
        for t in sec:
            t["p_holm"] = None
        return tests, sec

    @staticmethod
    def test_rows(tests: List[Dict]) -> List[List[str]]:
        return [[LABEL[t["baseline"]], f"{100 * t['diff']:+.2f}",
                 f"[{100 * t['cluster']['ci'][0]:+.2f}, {100 * t['cluster']['ci'][1]:+.2f}]",
                 fmt_p(t["cluster"]["p"]),
                 "secondary" if t["p_holm"] is None else fmt_p(t["p_holm"]),
                 f"{t['mcnemar']['n10']} / {t['mcnemar']['n01']}",
                 fmt_p(t["mcnemar"]["p_value"])] for t in tests]

    def t_lh(self, task: str, tag: str, title: str):
        archs = [self.sel] + self.baselines() + SECONDARY
        rows, store, ref, others = [], {}, None, {}
        for a in archs:
            r = self.lh_model(task, a)
            if r is None:
                rows.append([LABEL[a], "incomplete"] +
                            ["—"] * (len(METRICS) + 3 + (task == "7c")))
                continue
            keys, y, p, m, accs, row = r
            rows.append(row)
            store[a] = {"pooled": m, "fold_acc": accs}
            if a == self.sel:
                ref = (keys, y, p)
            else:
                others[a] = (keys, y, p)
        tests, sec = self.lh_tests(task, ref, others) if ref is not None else ([], [])
        self.data[tag] = {"models": store, "tests": tests}
        if sec:
            self.data[tag]["secondary_tests"] = sec
        hdr = (["Model", "N"] + [SHORT[k] for k in METRICS] +
               ["Acc 95% CI (patient)", "Fold acc mean (range)", "Acc excl. LR-selection patients"])
        if task == "7c":
            hdr.append("Superclass acc")
        trows = self.test_rows(tests + sec)
        self.md += [f"## {tag}. {title}",
                    "Pooled out-of-fold predictions (each image predicted once by a model "
                    "that never saw its patient), in %. CIs resample patients.", "",
                    md_table(hdr, rows), ""]
        if trows:
            self.md += [f"Paired tests, {LABEL.get(self.sel, self.sel)} vs baseline (pooled; "
                        "patient bootstrap CI and patient sign-flip p, Holm across baselines; "
                        "image-level McNemar for reference; 'secondary': VMamba-T, outside "
                        "the Holm family, p unadjusted):", "",
                        md_table(["Baseline", "Δ acc", "Patient 95% CI", "Patient p",
                                  "Holm p", "Discordant", "McNemar p"], trows), ""]

    def lr_selection_patients(self) -> set:
        """Patients in the validation set of 3-class patient-level fold 0."""
        if not hasattr(self, "_lr_sel"):
            f = self.S / "lunghist700" / f"patient_{self.mag}_fold0.json"
            meta_f = self.S / "lunghist700" / "metadata.json"
            if f.is_file() and meta_f.is_file():
                m = json.loads(f.read_text())
                meta = json.loads(meta_f.read_text())
                self._lr_sel = {int(meta[k]["patient_id"]) for k, v in m.items() if v == "val"}
            else:
                self._lr_sel = set()
        return self._lr_sel

    def patients(self, task: str, keys: List[str]) -> List[int]:
        sdir = self.S / ("lunghist700" if task == "3c" else "lunghist700_7c")
        if not hasattr(self, "_meta"):
            self._meta = {}
        if task not in self._meta:
            self._meta[task] = json.loads((sdir / "metadata.json").read_text())
        return [int(self._meta[task][k]["patient_id"]) for k in keys]

    # T3
    def t3(self):
        rows, store = [], {}
        for a in [self.sel] + self.baselines():
            g, l = self.lc_runs(a).get(42), self.lc_runs(a, leaky=True).get(42)
            if g is None or l is None:
                continue
            cg, cl = g["preds"] == g["labels"], l["preds"] == l["labels"]
            # Different test sets: resample source groups independently in each.
            groups = load_group_assignments()
            gk, lk = lc25000_test_keys(42), lc25000_leaky_test_keys(42)
            if len(gk) != len(cg) or len(lk) != len(cl):
                continue
            rng = np.random.default_rng(42)

            def group_boot(c, keys):
                uniq, inv = np.unique([groups[k] for k in keys], return_inverse=True)
                k_ = np.bincount(inv, weights=c.astype(float))
                s_ = np.bincount(inv).astype(float)
                ix = rng.integers(0, len(uniq), size=(10000, len(uniq)))
                return k_[ix].sum(1) / s_[ix].sum(1)

            boots = group_boot(cl, lk) - group_boot(cg, gk)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            d = float(cl.mean() - cg.mean())
            store[f"lc25000/{a}"] = {"grouped": float(cg.mean()), "leaky": float(cl.mean()),
                                     "inflation": d, "ci": [float(lo), float(hi)]}
            rows.append(["LC25000 (per-file vs grouped)", LABEL[a], pct(cg.mean()),
                         pct(cl.mean()), f"{100 * d:+.2f}",
                         f"[{100 * lo:+.2f}, {100 * hi:+.2f}]", "—"])
        for a in [self.sel] + self.baselines():
            rp, ri = self.lh_pooled("3c", "patient", a), self.lh_pooled("3c", "image", a)
            if rp is None or ri is None or sorted(rp[0]) != sorted(ri[0]):
                continue
            keys = sorted(rp[0])
            yp, pp, _ = self.arrays(rp[0], keys)
            yi, pi, _ = self.arrays(ri[0], keys)
            cp, ci_ = pp == yp, pi == yi
            cl = cluster_paired(ci_, cp, self.patients("3c", keys))
            lo, hi = cl["ci"]
            d = float(ci_.mean() - cp.mean())
            store[f"lunghist700/{a}"] = {"patient": float(cp.mean()), "image": float(ci_.mean()),
                                         "inflation": d, "cluster": cl,
                                         "mcnemar": mcnemar(ci_, cp)}
            rows.append([f"LungHist700 {self.mag} (image- vs patient-level)", LABEL[a],
                         pct(cp.mean()),
                         pct(ci_.mean()), f"{100 * d:+.2f}", f"[{100 * lo:+.2f}, {100 * hi:+.2f}]",
                         fmt_p(cl["p"])])
        self.data["T3"] = store
        self.md += ["## T3. Leakage inflation (seed 42 for LC25000; pooled CV for LungHist700)",
                    "Inflation = leaky − leakage-controlled accuracy (points). LC25000: the two "
                    "protocols have different test sets, so the CI resamples source groups "
                    "independently in each. LungHist700: both protocols predict the same "
                    "images, so the CI and sign-flip p are paired and resample patients.", "",
                    md_table(["Dataset", "Model", "Controlled", "Leaky", "Inflation",
                              "95% CI", "p"], rows), ""]

    # T6
    def t6(self):
        rows, store = [], {}
        for a in LC_ARCHS:
            row, per = [LABEL[a]], {}
            for mag in ("20x", "40x"):
                vals = {}
                for d in sorted((self.R / f"external_{mag}").glob(f"{a}_d4_224px_s*")):
                    p = load_pred(d)
                    if p is not None:
                        vals[d.name] = metrics(**p)
                per[mag] = vals
                row += [str(len(vals)), mean_range([v["accuracy"] for v in vals.values()]),
                        mean_range([v["balanced_accuracy"] for v in vals.values()]),
                        mean_range([v["f1_macro"] for v in vals.values()]),
                        mean_range([v["auc_macro"] for v in vals.values()])]
            if any(per[m] for m in per):
                rows.append(row)
                store[a] = per
        self.data["T6"] = store
        # Reference: accuracy of always predicting the most frequent class.
        meta_f = self.S / "lunghist700" / "metadata.json"
        if meta_f.is_file():
            meta = json.loads(meta_f.read_text())
            maj = {}
            for mag in ("20x", "40x"):
                cls = [v["class"] for v in meta.values() if v.get("resolution") == mag]
                if cls:
                    maj[mag] = max(cls.count(c) for c in set(cls)) / len(cls)
            self.data["T6_majority"] = maj
        hdr = ["Model"] + [f"{m} {h}" for m in ("20×", "40×")
                           for h in ("n", "Acc", "BalAcc", "F1", "AUC")]
        self.md += ["## T6. External test: LC25000-trained models on LungHist700 (3-class)",
                    "Mean (minimum–maximum) over seeds (n = seeds), in %. No retraining or "
                    "adaptation.", "",
                    md_table(hdr, rows), ""]

    # T7
    def t7(self):
        if not self.clean_csv or not self.clean_csv.is_file():
            self.md += ["## T7. Clean-subset sensitivity", "",
                        "Skipped: pass --lc25000_clean_csv (LC25000-clean "
                        "kaggle/lc25000_image_groups.csv).", ""]
            return
        theirs = {}
        with open(self.clean_csv) as f:
            for r in csv.DictReader(f):
                if r["label"] in LC_CLASSES:
                    theirs[f"{r['label']}/{r['filename']}"] = (r["label"], int(r["group_id"]))
        groups = load_group_assignments()
        keys = sorted(groups)
        idx = {c: i for i, c in enumerate(LC_CLASSES)}
        samples = [(f"/x/{k}", idx[k.split("/")[0]]) for k in keys]
        tr, _, te, _ = grouped_split(samples, groups, 3, 0.7, 0.15, 42)
        train_tiles = {theirs[keys[i]] for i in tr if keys[i] in theirs}
        test_keys = [keys[i] for i in te]
        clean = np.array([k in theirs and theirs[k] not in train_tiles for k in test_keys])
        rows, store = [], {}
        for a in LC_ARCHS:
            p = self.lc_runs(a).get(42)
            if p is None:
                continue
            c = p["preds"] == p["labels"]
            store[a] = {"all": float(c.mean()), "clean": float(c[clean].mean()),
                        "n_clean": int(clean.sum())}
            rows.append([LABEL[a], pct(c.mean()), pct(c[clean].mean()),
                         f"{100 * (c[clean].mean() - c.mean()):+.2f}"])
        self.data["T7"] = store
        self.md += ["## T7. Sensitivity: strictly clean LC25000 test subset (seed 42)",
                    f"Clean = test images whose LC25000-clean tile has no member in training "
                    f"({int(clean.sum())} of {len(clean)}).", "",
                    md_table(["Model", "All test", "Clean subset", "Δ"], rows), ""]

    # T8
    def _lc_faithfulness(self, sub: str, attribution: bool = False):
        """LC25000 faithfulness records of every run with outputs in `sub`
        (one canvas and random grid): per run and map, the image-level margin
        over the random map and a source-group test, Holm across all rows.
        Returns (records {run/map: rec}, md rows, pathway attribution {run: pa})."""
        sys.path.insert(0, str(ROOT / "tools"))
        from cluster_stats import cluster_mean_test  # noqa: E402
        gmap = load_group_assignments()
        recs: Dict = {}
        tests, rows, attr, arows = [], [], {}, []
        for a in LC_ARCHS:
            for d in sorted(self.R.glob(f"{a}_d4_224px_s*")):
                f, zf = d / sub / "xai_metrics.json", d / sub / "xai_per_sample.npz"
                if not f.is_file() or not zf.is_file():
                    continue
                x = json.loads(f.read_text())
                z = np.load(zf)
                grp = [gmap.get(k) for k in x.get("faithfulness_images") or []]
                for mname, s in (x.get("faithfulness") or {}).get("maps", {}).items():
                    if mname == "random":
                        continue
                    r = s["ins_minus_del"]
                    diff = z[f"{mname}_ins_minus_del_auc"] - z["random_ins_minus_del_auc"]
                    g = (cluster_mean_test(diff, grp)
                         if grp and None not in grp and len(grp) == len(diff) else None)
                    rec = {"arch": a, "run": d.name, "map": mname,
                           "deletion": s["deletion"]["mean"],
                           "insertion": s["insertion"]["mean"],
                           "ins_minus_del": r["mean"],
                           "random_ins_minus_del": float(z["random_ins_minus_del_auc"].mean()),
                           "minus_random": r["minus_random"]["mean"],
                           "wilcoxon_p": r.get("wilcoxon_p_vs_random"), "group": g,
                           "accuracy_on_subset": x["faithfulness"].get("accuracy_on_subset")}
                    recs[f"{d.name}/{mname}"] = rec
                    tests.append(rec)
                pa = x.get("pathway_attribution")
                if pa and attribution:
                    pa["acc_drop"] = pa["acc_full"] - pa["acc_mamba_ablated"]
                    pa["flipped_fraction"] = pa["predictions_flipped"] / x["test_size"]
                    attr[d.name] = pa
                    arows.append([d.name, pct(pa["acc_full"]), pct(pa["acc_mamba_ablated"]),
                                  str(pa["predictions_flipped"]),
                                  fmt_p(pa["mcnemar_full_vs_ablated"]["p_value"]),
                                  f"{pa['delta_to_cnn_norm_ratio']['mean']:.3f}",
                                  f"{pa['pred_prob_drop_when_ablated']['mean']:.4f}"])
        ps = [t["group"]["p"] if t["group"] else 1.0 for t in tests]
        for t, ph in zip(tests, holm(ps) if ps else []):
            t["group_p_holm"] = ph if t["group"] else None
            rows.append([t["run"], t["map"], f"{t['deletion']:.3f}",
                         f"{t['insertion']:.3f}", f"{t['ins_minus_del']:.3f}",
                         f"{t['minus_random']:+.3f}", fmt_p(t["wilcoxon_p"]),
                         fmt_p(t["group"]["p"]) if t["group"] else "—",
                         fmt_p(ph) if t["group"] else "—"])
        return recs, rows, attr, arows

    @staticmethod
    def _lc_summary(recs: Dict):
        """Per-(architecture, map) summary over seeds of LC25000 faithfulness
        records. Returns ({arch/map: summary}, md rows)."""
        summ: Dict = {}
        for rec in recs.values():
            summ.setdefault((rec["arch"], rec["map"]), []).append(rec)
        out, srows = {}, []
        for (a, m), rs in summ.items():
            v = np.array([r["minus_random"] for r in rs])
            n_sig = sum(1 for r in rs if (r.get("group_p_holm") or 1) < 0.05)
            n_neg = sum(1 for r in rs if r["minus_random"] < 0
                        and (r.get("group_p_holm") or 1) < 0.05)
            out[f"{a}/{m}"] = {
                "n_seeds": len(rs), "minus_random_mean": float(v.mean()),
                "minus_random_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "minus_random_min": float(v.min()), "minus_random_max": float(v.max()),
                "n_seeds_holm_sig": n_sig, "n_seeds_holm_sig_negative": n_neg,
                "ins_minus_del_mean": float(np.mean([r["ins_minus_del"] for r in rs]))}
            srows.append([LABEL.get(a, a), m, str(len(rs)),
                          f"{v.mean():+.3f}" + (f" ({v.min():+.3f} to {v.max():+.3f})"
                                                if len(v) > 1 else ""),
                          f"{n_sig}/{len(rs)}" + (f" ({n_neg} below random)" if n_neg else "")])
        return out, srows

    def t8(self):
        """LC25000 explanations. Faithfulness per run and canvas with a
        group-level test (scored images are nested in source groups), Holm
        across all rows of a canvas; per-architecture summary over seeds;
        pathway attribution of the residual hybrids."""
        store: Dict = {"blur": {}, "mean": {}, "attribution": {}}
        rows: Dict[str, List] = {}
        arows: List = []
        for canvas, sub in (("blur", "xai_metrics"), ("mean", "xai_metrics_meandel")):
            recs, rows[canvas], attr, ar = self._lc_faithfulness(sub, canvas == "blur")
            store[canvas] = recs
            if canvas == "blur":
                store["attribution"], arows = attr, ar
        # Per-architecture summary over seeds, per canvas.
        srows: Dict[str, List] = {}
        for canvas in ("blur", "mean"):
            summ, srows[canvas] = self._lc_summary(store[canvas])
            if summ:
                store[f"summary_{canvas}"] = summ
        store["summary"] = store.get("summary_blur", {})
        self.data["T8"] = store
        hdr = ["Run", "Map", "Deletion", "Insertion", "Ins − Del", "vs random",
               "Wilcoxon p", "Group p", "Group Holm p"]
        note = ("Insertion − deletion AUC (higher = more faithful); 'vs random' is the "
                "paired difference from a random map on the same 300 images. Group p: "
                "sign-flip test on per-source-group mean differences (primary), Holm "
                "across all rows of the canvas; image-level Wilcoxon p for reference.")
        self.md += ["## T8a. Saliency faithfulness on LC25000 (blur canvas, 300 test images)",
                    note, "", md_table(hdr, rows["blur"]), "",
                    "## T8a-sum. Faithfulness summary per architecture (blur canvas)",
                    "Δ vs random: mean (minimum to maximum) over seeds. Significant seeds: "
                    "group Holm p < 0.05, in either direction; seeds in which the map was "
                    "significantly below the random map are counted in brackets.", "",
                    md_table(["Model", "Map", "Seeds", "Δ vs random", "Significant seeds"],
                             srows["blur"]), "",
                    "## T8d. Saliency faithfulness on LC25000 (mean-colour canvas, secondary)",
                    note, "", md_table(hdr, rows["mean"]), "",
                    "Summary over seeds (mean-colour canvas):", "",
                    md_table(["Model", "Map", "Seeds", "Δ vs random", "Significant seeds"],
                             srows["mean"]), "",
                    "## T8b. Pathway attribution (residual hybrids, full LC25000 test set)", "",
                    md_table(["Run", "Acc full", "Acc Mamba ablated", "Flipped",
                              "McNemar p", "‖δ‖/‖f_cnn‖", "Prob drop"], arows), ""]

    # T8c
    LH_XAI_ARCHS = ("hybrid_res", "cnn_only", "effnet_b0", "vit_small", "swin_tiny")

    def _lh_xai(self, task: str, arch: str, sub: str = "xai_metrics"):
        """Load the five fold xai outputs for a LungHist700 patient-level run
        (sub: 'xai_metrics' = blur canvas, 'xai_metrics_meandel' = mean canvas).
        Returns a list of (json, npz arrays, split_file) or None if incomplete."""
        name = (f"lh700_patient{self.mag}_f{{}}_{arch}" if task == "3c"
                else f"lh700_7c_patient{self.mag}_f{{}}_{arch}")
        sdir = self.S / ("lunghist700" if task == "3c" else "lunghist700_7c")
        out = []
        for f in range(5):
            d = self.R / name.format(f) / sub
            if not (d / "xai_metrics.json").is_file() or not (d / "xai_per_sample.npz").is_file():
                return None
            z = np.load(d / "xai_per_sample.npz")
            out.append((json.loads((d / "xai_metrics.json").read_text()),
                        {k: z[k] for k in z.files}, sdir / f"patient_{self.mag}_fold{f}.json"))
        return out

    def _lh_faithfulness(self, sub: str):
        """Per-patient faithfulness of every (architecture, map) on the
        three-class patient-level folds, for one canvas. Returns (store, rows);
        empty when the outputs of that canvas are incomplete."""
        sys.path.insert(0, str(ROOT / "tools"))
        from cluster_stats import cluster_mean_test  # noqa: E402
        store: Dict = {}
        rows, tests = [], []
        for a in self.LH_XAI_ARCHS:
            runs = self._lh_xai("3c", a, sub)
            if runs is None:
                rows.append([LABEL[a], "incomplete"] + ["—"] * 7)
                continue
            maps = [k[:-len("_ins_minus_del_auc")] for k in runs[0][1]
                    if k.endswith("_ins_minus_del_auc") and not k.startswith("random")]
            for m in maps:
                rnd, val, pts = [], [], []
                for x, z, _ in runs:
                    imgs = x.get("faithfulness_images") or []
                    pts += [self.patients("3c", [k])[0] for k in imgs]
                    val.append(z[f"{m}_ins_minus_del_auc"])
                    rnd.append(z["random_ins_minus_del_auc"])
                val, rnd = np.concatenate(val), np.concatenate(rnd)
                if len(pts) != len(val):
                    raise ValueError(f"{a}/{m}: {len(pts)} image ids vs {len(val)} AUCs")
                r = cluster_mean_test(val - rnd, pts)
                r.update(map_mean=float(val.mean()), random_mean=float(rnd.mean()))
                store[f"{a}/{m}"] = r
                tests.append((a, m, r))
        if not tests:
            return {}, rows
        for (a, m, r), ph in zip(tests, holm([t[2]["p"] for t in tests])):
            r["p_holm"] = ph
            rows.append([LABEL[a], m, str(r["n_images"]), str(r["n_clusters"]),
                         f"{r['map_mean']:.3f}", f"{r['random_mean']:.3f}",
                         f"{r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]",
                         f"{100 * r['frac_clusters_positive']:.0f}", fmt_p(ph)])
        return store, rows

    def t8c(self):
        """LungHist700 explanations with the patient as the unit of inference.

        (a) Faithfulness, 3-class patient-level folds: for every scored image,
            d = (ins − del AUC of the map) − (ins − del AUC of the random map);
            per-patient mean of d, bootstrap CI and sign-flip p over patients
            (tools/cluster_stats.py), Holm across all (model, map) rows.
        (b) Pathway attribution of the selected hybrid, pooled over the five
            out-of-fold test sets: accuracy with and without the Mamba term,
            patient-level paired CI/p of the change (cluster_paired)."""
        store: Dict = {"attribution": {}}
        store["faithfulness"], frows = self._lh_faithfulness("xai_metrics")
        # Secondary canvas: deleted pixels replaced by the normalisation mean
        # (insertion unchanged). Holm is applied within each canvas.
        fm, frows_mean = self._lh_faithfulness("xai_metrics_meandel")
        if fm:
            store["faithfulness_mean"] = fm
        arows = []
        for task in ("3c", "7c"):
            runs = self._lh_xai(task, self.sel)
            if runs is None:
                arows.append([task, "incomplete"] + ["—"] * 6)
                continue
            y, pf, pa, ratio, keys = [], [], [], [], []
            for x, z, sf in runs:
                if "attr_labels" not in z:
                    raise ValueError(f"no pathway attribution in {x.get('run')}")
                k = split_test_keys(sf)
                cls = x["class_names"]
                exp = np.array([cls.index(s.split("/")[0]) for s in k])
                if not np.array_equal(exp, z["attr_labels"]):
                    raise ValueError(f"attribution order differs from {sf.name}")
                keys += k
                y.append(z["attr_labels"]); pf.append(z["attr_prob_full"])
                pa.append(z["attr_prob_ablated"]); ratio.append(z["attr_ratio"])
            y, pf, pa, ratio = map(np.concatenate, (y, pf, pa, ratio))
            pred_f, pred_a = pf.argmax(1), pa.argmax(1)
            cf, ca = pred_f == y, pred_a == y
            pts = self.patients(task, keys)
            cp = cluster_paired(cf, ca, pts)
            drop = pf[np.arange(len(y)), pred_f] - pa[np.arange(len(y)), pred_f]
            # The full-model predictions must reproduce the reported out-of-fold ones.
            pooled = self.lh_pooled(task, "patient", self.sel)
            n_mismatch = (None if pooled is None else
                          int((np.array([pooled[0][k][1] for k in keys]) != pred_f).sum()))
            same = n_mismatch == 0
            s = {"n": int(len(y)), "acc_full": float(cf.mean()), "acc_ablated": float(ca.mean()),
                 "acc_change": float(cf.mean() - ca.mean()), "cluster": cp,
                 "flipped": int((pred_f != pred_a).sum()),
                 "flipped_fraction": float((pred_f != pred_a).mean()),
                 "ratio_mean": float(ratio.mean()), "prob_drop_mean": float(drop.mean()),
                 "matches_reported_predictions": bool(same),
                 "n_mismatch_vs_reported": n_mismatch, "mcnemar": mcnemar(cf, ca)}
            store["attribution"][task] = s
            arows.append([task, str(s["n"]), pct(s["acc_full"]), pct(s["acc_ablated"]),
                          f"{100 * s['acc_change']:+.2f} [{100 * cp['ci'][0]:+.2f}, "
                          f"{100 * cp['ci'][1]:+.2f}]", fmt_p(cp["p"]),
                          f"{s['flipped']} ({100 * s['flipped_fraction']:.1f}%)",
                          f"{s['ratio_mean']:.3f}"])
        self.data["T8c"] = store
        self.md += [f"## T8c. LungHist700 explanations, patient-level folds ({self.mag})",
                    "(a) Faithfulness (blur canvas): Δ = mean over patients of the per-image "
                    "difference in insertion − deletion AUC between the map and a random map; "
                    "patient bootstrap 95% CI; '% patients > 0'; sign-flip p over patients, "
                    "Holm across all rows.", "",
                    md_table(["Model", "Map", "Images", "Patients", "Map ins−del",
                              "Random ins−del", "Δ vs random [95% CI]", "% patients > 0",
                              "Holm p"], frows), ""]
        if fm:
            self.md += ["(a2) Faithfulness, mean-colour canvas (deleted pixels replaced by "
                        "the normalisation mean; insertion as for the blur canvas); same "
                        "statistics, Holm across the rows of this canvas.", "",
                        md_table(["Model", "Map", "Images", "Patients", "Map ins−del",
                                  "Random ins−del", "Δ vs random [95% CI]", "% patients > 0",
                                  "Holm p"], frows_mean), ""]
        self.md += [f"(b) Pathway attribution of {LABEL.get(self.sel, self.sel)}, pooled "
                    "out-of-fold; accuracy change = full − Mamba term removed, patient "
                    "bootstrap CI and sign-flip p.", "",
                    md_table(["Task", "N", "Acc full", "Acc Mamba ablated",
                              "Change [95% CI]", "Patient p", "Flipped", "‖δ‖/‖f_cnn‖"],
                             arows), ""]

    # T10
    def lr_records(self) -> Dict[str, Dict]:
        """Fold-0 validation accuracy per learning rate, the pre-specified
        choice and the sensitivity rate of every architecture, from the records
        written by the experiment queues (logs/lh700_lr_selection.txt,
        logs/lh700_lr_selection_vmamba.txt, logs/lh700_lr_sensitivity.txt). A
        record appended twice must repeat the same values."""
        out: Dict[str, Dict] = {}

        def put(a: str, key: str, value, lr: Optional[str] = None):
            d = out.setdefault(a, {"val_acc": {}})
            tgt, k = (d["val_acc"], lr) if lr else (d, key)
            if k in tgt and tgt[k] != value:
                raise ValueError(f"conflicting records for {a} {key} {lr or ''}: "
                                 f"{tgt[k]} vs {value}")
            tgt[k] = value

        logs = self.R.parent / "logs"
        for name in ("lh700_lr_selection.txt", "lh700_lr_selection_vmamba.txt",
                     "lh700_lr_sensitivity.txt"):
            f = logs / name
            if not f.is_file():
                continue
            for line in f.read_text().splitlines():
                m = re.match(r"# (\S+) lr=(\S+) best_val_acc=(\S+)", line)
                if m:
                    put(m.group(1), "val_acc", float(m.group(3)), m.group(2))
                    continue
                m = re.match(r"(selected|sensitivity): (\S+) lr=(\S+)", line)
                if m:
                    put(m.group(2), "primary" if m.group(1) == "selected" else "sensitivity",
                        m.group(3))
        # The records must follow the rules of the protocol: the pre-specified
        # choice is the better of 1e-4 and 1e-3, the sensitivity rate the best
        # of 1e-4, 1e-3 and 3e-3; ties keep the smaller rate.
        for a, d in out.items():
            v = d["val_acc"]
            for key, grid in (("primary", ("1e-4", "1e-3")),
                              ("sensitivity", ("1e-4", "1e-3", SENS_LR))):
                if key in d and all(g in v for g in grid):
                    best = max(grid, key=lambda g: (v[g], -float(g)))
                    if best != d[key]:
                        raise ValueError(f"{a}: recorded {key} rate {d[key]} but the "
                                         f"rule gives {best}")
        return out

    def t10(self):
        """A2 (docs/added_analyses_protocol.md): LungHist700 learning-rate
        sensitivity. The fold-0 validation accuracy of every architecture at
        1e-4, 1e-3 and 3e-3; for every architecture whose sensitivity rate
        differs from its pre-specified rate, its pooled results at that rate;
        and the paired tests of the selected hybrid repeated with every model
        at its sensitivity rate (Holm over the same five comparisons)."""
        rec = self.lr_records()
        if not any("sensitivity" in r for r in rec.values()):
            return
        archs = [self.sel] + self.baselines() + SECONDARY
        rates = {a: {"val_acc": rec.get(a, {}).get("val_acc", {}),
                     "primary": rec.get(a, {}).get("primary"),
                     "sensitivity": rec.get(a, {}).get("sensitivity")} for a in archs}
        changed = [a for a in archs
                   if rates[a]["sensitivity"] not in (None, rates[a]["primary"])]
        store: Dict = {"rates": rates, "changed": changed}
        rate_rows = [[LABEL[a]] + [pct(rates[a]["val_acc"].get(lr), 1)
                                   for lr in ("1e-4", "1e-3", SENS_LR)]
                     + [rates[a]["primary"] or "—", rates[a]["sensitivity"] or "pending"]
                     for a in archs if a in rec]
        self.md += ["## T10. LungHist700 learning-rate sensitivity (added after the "
                    "pre-submission review)",
                    "Fold-0 validation accuracy (%, three-class patient-level split) at each "
                    "rate; the pre-specified choice is the better of 1e-4 and 1e-3, the "
                    "sensitivity rate the best of 1e-4, 1e-3 and 3e-3 (ties keep the smaller "
                    "rate). Primary results are those at the pre-specified rate.", "",
                    md_table(["Model", "1e-4", "1e-3", SENS_LR, "Pre-specified",
                              "Sensitivity"], rate_rows), ""]
        # Architectures whose 3e-3 run has not been recorded yet. The Phikon
        # probe is not trained with a learning rate, so it is never pending.
        pending = [a for a in archs if a in rec and rates[a]["sensitivity"] is None]
        store["pending"] = pending
        for task, title in (("3c", "three classes"), ("7c", "seven classes")):
            models, ref, others, missing = {}, None, {}, []
            rows = []
            for a in archs:
                prefix = SENS_PREFIX if a in changed else "lh700_"
                r = self.lh_model(task, a, prefix)
                if r is None:
                    missing.append(a)
                    continue
                keys, y, p, m, accs, row = r
                if a in changed:
                    models[a] = {"pooled": m, "fold_acc": accs, "lr": rates[a]["sensitivity"]}
                    rows.append([f"{row[0]} ({rates[a]['sensitivity']})"] + row[1:])
                if a == self.sel:
                    ref = (keys, y, p)
                elif a not in pending:
                    others[a] = (keys, y, p)
            family = [self.sel] + self.baselines()
            # Holm needs the whole family at its sensitivity rates; the
            # secondary test needs VMamba-T's own record and runs.
            complete = not any(a in missing or a in pending for a in family)
            tests, sec = self.lh_tests(task, ref, others) if complete else ([], [])
            store[task] = {"models": models, "tests": tests, "secondary_tests": sec,
                           "missing": missing, "complete": complete}
            missing = missing + [a for a in pending if a not in missing]
            hdr = (["Model (rate)", "N"] + [SHORT[k] for k in METRICS] +
                   ["Acc 95% CI (patient)", "Fold acc mean (range)",
                    "Acc excl. LR-selection patients"] + (["Superclass acc"] if task == "7c"
                                                          else []))
            self.md += [f"LungHist700, {title}, pooled out-of-fold, models whose sensitivity "
                        f"rate differs from the pre-specified one, at that rate:", "",
                        md_table(hdr, rows) if rows else "None.", ""]
            if complete:
                self.md += [f"Paired tests repeated with every model at its sensitivity rate "
                            f"({title}; as in {'T4' if task == '3c' else 'T5'}, Holm across "
                            "the same five comparisons):", "",
                            md_table(["Baseline", "Δ acc", "Patient 95% CI", "Patient p",
                                      "Holm p", "Discordant", "McNemar p"],
                                     self.test_rows(tests + sec)), ""]
            else:
                self.md += [f"Paired tests pending: runs missing for "
                            f"{', '.join(LABEL[a] for a in missing)}.", ""]
        self.data["T10"] = store

    # T11
    def t11(self):
        """A3 (docs/added_analyses_protocol.md): every faithfulness analysis
        re-scored against a 7 x 7 random map (outputs *_rg7), with the same
        statistics as T8 and T8c, next to the margins over the 14 x 14 random
        map of the primary analysis."""
        store: Dict = {"lc25000": {}, "lunghist700": {}}
        rows = []
        for canvas, sub, sub14 in (("blur", "xai_metrics_rg7", "blur"),
                                   ("mean", "xai_metrics_meandel_rg7", "mean")):
            recs, _, _, _ = self._lc_faithfulness(sub)
            if not recs:
                continue
            summ, _ = self._lc_summary(recs)
            # Complete when every run and map of the primary analysis is re-scored
            # (otherwise Holm would run over a smaller family).
            prim_runs = set((self.data.get("T8") or {}).get(sub14, {}))
            store["lc25000"][canvas] = {"runs": recs, "summary": summ,
                                        "complete": set(recs) == prim_runs}
            prim = (self.data.get("T8") or {}).get(f"summary_{sub14}", {})
            for key, s in summ.items():
                a, m = key.split("/")
                p = prim.get(key, {})
                rows.append(["LC25000", canvas, LABEL.get(a, a), m, str(s["n_seeds"]),
                             f"{p['minus_random_mean']:+.3f}" if p else "—",
                             f"{p['n_seeds_holm_sig']}/{p['n_seeds']}" if p else "—",
                             f"{s['minus_random_mean']:+.3f} ({s['minus_random_min']:+.3f} to "
                             f"{s['minus_random_max']:+.3f})",
                             f"{s['n_seeds_holm_sig']}/{s['n_seeds']}"
                             + (f" ({s['n_seeds_holm_sig_negative']} below random)"
                                if s["n_seeds_holm_sig_negative"] else "")])
        t8c = self.data.get("T8c") or {}
        for canvas, sub, key14 in (("blur", "xai_metrics_rg7", "faithfulness"),
                                   ("mean", "xai_metrics_meandel_rg7", "faithfulness_mean")):
            fs_, _ = self._lh_faithfulness(sub)
            if not fs_:
                continue
            store["lunghist700"][canvas] = {"maps": fs_,
                                            "complete": set(fs_) == set(t8c.get(key14, {}))}
            prim = t8c.get(key14, {})
            for key, r in fs_.items():
                a, m = key.split("/")
                p = prim.get(key)
                rows.append(["LungHist700", canvas, LABEL.get(a, a), m, str(r["n_clusters"]),
                             f"{p['mean']:+.3f}" if p else "—",
                             fmt_p(p["p_holm"]) if p else "—",
                             f"{r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]",
                             fmt_p(r["p_holm"])])
        if not rows:
            return
        self.data["T11"] = store
        self.md += ["## T11. Faithfulness against a 7 × 7 random map (sensitivity; added "
                    "after the pre-submission review)",
                    "Margin = map minus random map in insertion − deletion AUC, on the same "
                    "images. LC25000: mean (minimum to maximum) over seeds of the image-level "
                    "margin; 'significant' counts seeds with source-group Holm p < 0.05, Holm "
                    "across all rows of a canvas and grid (T8). LungHist700: mean over "
                    "patients [patient bootstrap 95% CI] and Holm p over the rows of a canvas "
                    "and grid (T8c). The 14 × 14 columns repeat the primary analysis.", "",
                    md_table(["Dataset", "Canvas", "Model", "Map", "Seeds / patients",
                              "14 × 14: margin", "14 × 14: significant / Holm p",
                              "7 × 7: margin", "7 × 7: significant / Holm p"], rows), ""]

    # T9
    def t9(self):
        f = self.R / "_efficiency" / "efficiency.json"
        if not f.is_file():
            self.md += ["## T9. Computational cost", "", "Not yet measured.", ""]
            return
        e = json.loads(f.read_text())
        for r in e.get("models", {}).values():
            r["params_m"] = r["params"] / 1e6
        # VMamba-T was measured in a later session (scripts/queue_review.sh) on
        # the same instance type and software, together with two of the models
        # above as a check that the sessions' timings agree.
        f2 = self.R / "_efficiency" / "efficiency_vmamba.json"
        if f2.is_file():
            e2 = json.loads(f2.read_text())
            for r in e2.get("models", {}).values():
                r["params_m"] = r["params"] / 1e6
            for a in SECONDARY:
                if a in e2.get("models", {}):
                    e["models"][a] = dict(e2["models"][a], session=2)
            e["session2"] = e2
            e["session_check"] = {
                a: {"latency_b1_ms": [e["models"][a]["latency_b1_ms"]["median"],
                                      r["latency_b1_ms"]["median"]],
                    "latency_ratio": r["latency_b1_ms"]["median"]
                    / e["models"][a]["latency_b1_ms"]["median"]}
                for a, r in e2.get("models", {}).items()
                if a not in SECONDARY and a in e["models"]}
        self.data["T9"] = e
        rows = []
        for a, r in e["models"].items():
            thr = next(v for k, v in r.items() if k.startswith("throughput_b"))
            mem = next(v for k, v in r.items() if k.startswith("peak_mem_b"))
            rows.append([LABEL.get(a, a) + (" (second session)" if r.get("session") else ""),
                         f"{r['params'] / 1e6:.2f}", f"{r['gflops']:.2f}",
                         f"{r['latency_b1_ms']['median']:.2f}", f"{thr:.0f}",
                         "—" if mem is None else f"{mem:.0f}"])
        note = ("GFLOPs exclude the element-wise selective scan (lower bound for Mamba "
                "models).")
        if e.get("session_check"):
            note += (" VMamba-T was measured in a second session on the same instance type "
                     "and software, in which " + " and ".join(
                         f"{LABEL.get(a, a)} took {c['latency_b1_ms'][1]:.2f} ms at batch 1 "
                         f"({c['latency_b1_ms'][0]:.2f} ms above)"
                         for a, c in e["session_check"].items()) + ".")
        self.md += [f"## T9. Computational cost ({e.get('gpu') or e['device']}, "
                    f"{e['image_size']} px, fp16={e['amp_fp16']})",
                    note, "",
                    md_table(["Model", "Params (M)", "GFLOPs", "Latency b1 (ms)",
                              "Throughput b64 (img/s)", "Peak mem b64 (MB)"], rows), ""]


def selected_hybrid(results_root: Path) -> str:
    f = results_root.parent / "logs" / "selected_hybrid.txt"
    if f.is_file():
        for line in reversed(f.read_text().splitlines()):
            m = re.match(r"selected:\s*(\S+)", line)
            if m:
                return m.group(1)
    return "hybrid_res"


def _mean_sd_dict(per_run: Dict) -> Dict:
    """{run: {metric: value}} -> {metric: {mean, sd, min, max, n}} (sample SD)."""
    out: Dict = {}
    runs = [m for m in per_run.values() if isinstance(m, dict)]
    for k in METRICS:
        v = np.array([m[k] for m in runs if k in m], dtype=float)
        if len(v):
            out[k] = {"mean": float(v.mean()), "n": int(len(v)),
                      "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "min": float(v.min()), "max": float(v.max())}
    return out


def seed_summaries(data: Dict) -> Dict:
    """Mean, range and SD over seeds (T1, T6), stored so that every number
    quoted in the manuscript can be traced to tables.json."""
    s: Dict = {"T1": {}, "T6": {}}
    for arch, per in data.get("T1", {}).items():
        if isinstance(per, dict):
            s["T1"][arch] = _mean_sd_dict(per)
    for arch, per in data.get("T6", {}).items():
        s["T6"][arch] = {mag: _mean_sd_dict(runs) for mag, runs in per.items()}
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--splits", default=str(ROOT / "splits"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--selected", default=None,
                    help="Default: read logs/selected_hybrid.txt")
    ap.add_argument("--lc25000_clean_csv", default=str(ROOT / "lc25000_image_groups.csv"))
    ap.add_argument("--mag", default="all", choices=("20x", "40x", "all"),
                    help="LungHist700 CV magnification (scripts/queue_lh700.sh MAG)")
    args = ap.parse_args()

    R = Path(args.results)
    out = Path(args.out) if args.out else R / "_paper"
    out.mkdir(parents=True, exist_ok=True)
    sel = args.selected or selected_hybrid(R)
    b = Builder(R, Path(args.splits), sel, Path(args.lc25000_clean_csv), args.mag)
    b.md += [f"# Paper 1 results (selected hybrid: {LABEL.get(sel, sel)})", ""]
    b.t1(); b.t2(); b.t3()
    b.t_lh("3c", "T4", f"LungHist700 3-class, patient-level 5-fold CV ({args.mag})")
    b.t_lh("7c", "T5", f"LungHist700 7-class differentiation grading, patient-level "
                       f"5-fold CV ({args.mag})")
    b.t6(); b.t7(); b.t8(); b.t8c(); b.t9()
    # Added after the pre-submission review (docs/added_analyses_protocol.md).
    b.t10(); b.t11()
    b.data["summary"] = seed_summaries(b.data)
    # Derived quantities quoted in the text, stored so they can be traced.
    der: Dict = {"lc_seed_range": {}, "params_added_vs_cnn_m": {}}
    for a, per in b.data.get("T1", {}).items():
        accs = [m["accuracy"] for m in per.values()]
        if len(accs) > 1:
            der["lc_seed_range"][a] = max(accs) - min(accs)
    t9 = (b.data.get("T9") or {}).get("models", {})
    if "cnn_only" in t9:
        for a, r in t9.items():
            der["params_added_vs_cnn_m"][a] = (r["params"] - t9["cnn_only"]["params"]) / 1e6
    # A3: how far each margin moved between the 14 x 14 and the 7 x 7 random
    # map, and which maps changed side, per dataset and canvas.
    t11 = b.data.get("T11")
    if t11:
        t8, t8c = b.data.get("T8", {}), b.data.get("T8c", {})
        change: Dict = {}
        for ds, canvas, new, old, k_new, k_old in (
                ("lc25000", "blur", t11["lc25000"].get("blur", {}).get("summary", {}),
                 t8.get("summary_blur", {}), "minus_random_mean", "minus_random_mean"),
                ("lc25000", "mean", t11["lc25000"].get("mean", {}).get("summary", {}),
                 t8.get("summary_mean", {}), "minus_random_mean", "minus_random_mean"),
                ("lunghist700", "blur", t11["lunghist700"].get("blur", {}).get("maps", {}),
                 t8c.get("faithfulness", {}), "mean", "mean"),
                ("lunghist700", "mean", t11["lunghist700"].get("mean", {}).get("maps", {}),
                 t8c.get("faithfulness_mean", {}), "mean", "mean")):
            common = [k for k in new if k in old]
            if not common:
                continue
            d_ = {k: new[k][k_new] - old[k][k_old] for k in common}
            change.setdefault(ds, {})[canvas] = {
                "max_abs_change": max(abs(v) for v in d_.values()),
                "min_change": min(d_.values()), "max_change": max(d_.values()),
                "changed_sign": [k for k in common
                                 if (new[k][k_new] > 0) != (old[k][k_old] > 0)]}
        der["rg7_vs_rg14"] = change
    b.data["derived"] = der
    (out / "tables.md").write_text("\n".join(b.md))
    (out / "tables.json").write_text(json.dumps(b.data, indent=1, default=lambda o: (
        o.tolist() if isinstance(o, np.ndarray) else float(o))))
    print(f"written {out / 'tables.md'} and tables.json")


if __name__ == "__main__":
    main()
