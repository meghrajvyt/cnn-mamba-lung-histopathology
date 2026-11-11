#!/usr/bin/env python3
"""Paired statistical comparison between two evaluated models.

WHY THIS EXISTS
---------------
Reporting "model A scored 92.1% and model B scored 89.4%" does not establish that
A is better. The two were evaluated on the *same* test set, so the comparison is
paired, and a paired test is both more powerful and more honest than eyeballing a
gap. Reviewers in this area routinely ask for it.

McNemar's test is the standard choice for comparing two classifiers on one test
set (Dietterich 1998). It ignores samples both models get right or both get wrong
— those carry no information about which is better — and asks whether the
*disagreements* are lopsided beyond chance.

Usage:
    python compare_models.py results/grouped_d2 results/leaky_d2
    python compare_models.py results/hybrid results/cnn_only --name-a Hybrid --name-b "CNN only"
    python compare_models.py --all results/          # every pairwise comparison
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_predictions(run_dir: str | Path) -> Dict[str, np.ndarray]:
    """Load a run's per-sample predictions written by evaluate.py.

    Raises:
        FileNotFoundError: If predictions.npz is absent, which means the run was
            evaluated before prediction-saving existed and must be re-evaluated.
    """
    p = Path(run_dir) / "predictions.npz"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found. This run was evaluated before per-sample "
            f"predictions were saved; re-run evaluate.py on its checkpoint to "
            f"enable paired testing."
        )
    z = np.load(p, allow_pickle=True)
    return {
        "labels": z["labels"],
        "preds": z["preds"],
        "probs": z["probs"],
        "class_names": list(z["class_names"]) if "class_names" in z else None,
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray,
            exact_threshold: int = 25) -> Dict[str, float]:
    """McNemar's test on two boolean correctness vectors.

    Args:
        correct_a: Per-sample correctness of model A.
        correct_b: Per-sample correctness of model B.
        exact_threshold: Below this many discordant pairs, use the exact binomial
            test instead of the chi-square approximation, which is unreliable on
            small counts.

    Returns:
        Dict with n01, n10, statistic, p_value, test name, and the odds ratio.
    """
    # Discordant pairs: exactly the samples where the models disagree.
    n01 = int(np.sum(~correct_a & correct_b))   # only B correct
    n10 = int(np.sum(correct_a & ~correct_b))   # only A correct
    n = n01 + n10

    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0,
                "test": "none (models never disagree)", "odds_ratio": float("nan")}

    if n < exact_threshold:
        # Exact two-sided binomial test against p=0.5.
        k = min(n01, n10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
        stat = float(k)
        test = f"exact binomial (n={n} < {exact_threshold})"
    else:
        # Chi-square with Edwards' continuity correction.
        stat = (abs(n01 - n10) - 1.0) ** 2 / n
        # Survival function of chi-square with 1 df, without SciPy:
        # P(X > x) = erfc(sqrt(x/2))
        p = math.erfc(math.sqrt(stat / 2.0))
        test = "chi-square with continuity correction"

    odds = (n10 / n01) if n01 > 0 else float("inf")
    return {"n01": n01, "n10": n10, "statistic": float(stat),
            "p_value": float(p), "test": test, "odds_ratio": float(odds)}


def bootstrap_ci(correct: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for accuracy.

    Gives an interval from a single run, which is not a substitute for
    multi-seed variance but is better than reporting a bare point estimate.
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = correct[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_accuracy_diff_ci(correct_a: np.ndarray, correct_b: np.ndarray,
                            n_boot: int = 10000, alpha: float = 0.05,
                            seed: int = 42) -> Tuple[float, float]:
    """Bootstrap CI for the accuracy *difference*, resampling pairs together.

    Resampling both models on the same indices preserves the pairing, which is
    the whole point: it removes variance due to which samples are in the test set.
    """
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compare(dir_a: str, dir_b: str, name_a: Optional[str] = None,
            name_b: Optional[str] = None) -> Dict:
    """Run the full paired comparison between two evaluated runs."""
    a = load_predictions(dir_a)
    b = load_predictions(dir_b)
    name_a = name_a or Path(dir_a).name
    name_b = name_b or Path(dir_b).name

    if len(a["labels"]) != len(b["labels"]):
        raise ValueError(
            f"Test sets differ in size ({len(a['labels'])} vs {len(b['labels'])}). "
            f"A paired test requires the SAME test set. This usually means the "
            f"two runs used different splits — note that a grouped run and a "
            f"per-file run are NOT comparable this way."
        )
    if not np.array_equal(a["labels"], b["labels"]):
        raise ValueError(
            "Test-set labels differ between runs, so the samples are not aligned "
            "and a paired test is invalid. Re-evaluate both with the same seed, "
            "ratios and grouping."
        )

    correct_a = (a["preds"] == a["labels"])
    correct_b = (b["preds"] == b["labels"])
    acc_a, acc_b = float(correct_a.mean()), float(correct_b.mean())

    mc = mcnemar(correct_a, correct_b)
    ci_a = bootstrap_ci(correct_a)
    ci_b = bootstrap_ci(correct_b)
    ci_d = paired_accuracy_diff_ci(correct_a, correct_b)

    return {
        "model_a": name_a, "model_b": name_b,
        "n_test": int(len(a["labels"])),
        "accuracy_a": acc_a, "accuracy_b": acc_b,
        "accuracy_diff": acc_a - acc_b,
        "ci95_a": ci_a, "ci95_b": ci_b, "ci95_diff": ci_d,
        "mcnemar": mc,
        "significant_at_0.05": bool(mc["p_value"] < 0.05),
    }


def format_report(r: Dict) -> str:
    """Human-readable comparison, phrased so it can go straight into a paper."""
    mc = r["mcnemar"]
    sig = "SIGNIFICANT" if r["significant_at_0.05"] else "not significant"
    p = mc["p_value"]
    p_str = "< 0.001" if p < 0.001 else f"= {p:.4f}"

    lines = [
        f"Paired comparison: {r['model_a']} vs {r['model_b']}",
        f"  test samples      : {r['n_test']}",
        f"  {r['model_a'][:22]:22s} accuracy: {r['accuracy_a']:.4f}  "
        f"95% CI [{r['ci95_a'][0]:.4f}, {r['ci95_a'][1]:.4f}]",
        f"  {r['model_b'][:22]:22s} accuracy: {r['accuracy_b']:.4f}  "
        f"95% CI [{r['ci95_b'][0]:.4f}, {r['ci95_b'][1]:.4f}]",
        f"  difference        : {r['accuracy_diff']:+.4f}  "
        f"95% CI [{r['ci95_diff'][0]:+.4f}, {r['ci95_diff'][1]:+.4f}]",
        "",
        f"  McNemar ({mc['test']})",
        f"    only {r['model_a'][:18]} correct : {mc['n10']}",
        f"    only {r['model_b'][:18]} correct : {mc['n01']}",
        f"    statistic = {mc['statistic']:.4f}, p {p_str}  -> {sig} at alpha=0.05",
    ]
    if math.isfinite(mc["odds_ratio"]):
        lines.append(f"    odds ratio = {mc['odds_ratio']:.3f}")

    # A CI on the difference that contains zero is the honest reading, even when
    # the point estimate looks convincing.
    lo, hi = r["ci95_diff"]
    if lo <= 0 <= hi:
        lines += ["", "  NOTE: the 95% CI for the difference includes zero, so "
                      "this gap is not",
                  "        established at alpha=0.05 on a single run. Report it "
                      "with the interval,",
                  "        and prefer multi-seed results before claiming "
                      "superiority."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="two result directories to compare")
    ap.add_argument("--all", metavar="ROOT",
                    help="compare every pair of runs under ROOT")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--json-out", default=None, help="write results as JSON")
    args = ap.parse_args()

    reports: List[Dict] = []

    if args.all:
        root = Path(args.all)
        runs = sorted(d for d in root.iterdir()
                      if d.is_dir() and (d / "predictions.npz").is_file())
        if len(runs) < 2:
            print(f"Need at least two evaluated runs under {root}; found "
                  f"{len(runs)}. Runs must contain predictions.npz.")
            sys.exit(1)
        for a, b in itertools.combinations(runs, 2):
            try:
                r = compare(str(a), str(b))
            except ValueError as e:
                print(f"skipping {a.name} vs {b.name}: {e}\n")
                continue
            reports.append(r)
            print(format_report(r)); print("-" * 72)
    else:
        if len(args.dirs) != 2:
            ap.error("provide exactly two result directories, or use --all ROOT")
        r = compare(args.dirs[0], args.dirs[1], args.name_a, args.name_b)
        reports.append(r)
        print(format_report(r))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
