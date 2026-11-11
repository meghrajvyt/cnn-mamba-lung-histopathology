#!/usr/bin/env python3
"""Main-text tables for the Paper 1 manuscript, rendered from tables.json.

Generated rather than typed, so every cell traces to results/_paper/tables.json
(itself recomputed from predictions by tools/paper_tables.py).

  Table 1  LC25000: mean (range) over seeds, and paired tests vs the selected hybrid
  Table 2  LungHist700 three- and seven-class, pooled patient-level CV, paired tests

Usage:
    python3 tools/manuscript_tables.py [--tables results/_paper/tables.json] [--table 2]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"hybrid_res": "Hybrid-res (proposed)", "hybrid_res_xattn": "Hybrid-res-xattn",
         "grouped": "Original dual-pathway hybrid", "cnn_only": "CNN pathway",
         "effnet_b0": "EfficientNet-B0", "vit_small": "ViT-S/16", "swin_tiny": "Swin-T",
         "phikon_lp": "Phikon linear probe", "vmamba_tiny": "VMamba-T"}
ORDER = ["hybrid_res", "cnn_only", "effnet_b0", "vit_small", "swin_tiny", "phikon_lp"]
# Added after the pre-submission review (docs/added_analyses_protocol.md): a
# secondary comparator, compared outside the Holm family.
SECONDARY = ["vmamba_tiny"]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def rate(lr: str) -> str:
    """'3e-3' -> '3 × 10⁻³', '1e-3' -> '10⁻³'."""
    m, e = lr.split("e")
    return ("" if m == "1" else f"{m} × ") + "10" + str(int(e)).translate(SUP)


def pct(x: Optional[float], nd: int = 2) -> str:
    return "—" if x is None else f"{100 * x:.{nd}f}"


def ci(c, nd: int = 1, signed: bool = False) -> str:
    f = f"{{:{'+' if signed else ''}.{nd}f}}"
    return f"{f.format(100 * c[0])} to {f.format(100 * c[1])}"


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(header: List[str], rows: List[List[str]]) -> str:
    # Typographic minus for negative numbers, as in the text.
    rows = [[re.sub(r"(?<![\w])-(?=\d)", "\u2212", c) for c in r] for r in rows]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def _lh_row(label: str, m3: Dict, m7: Dict) -> List[str]:
    return [label, f"{pct(m3['accuracy'])} ({ci(m3['accuracy_ci_patient'])})",
            pct(m3["f1_macro"]), f"{pct(m7['accuracy'])} ({ci(m7['accuracy_ci_patient'])})",
            pct(m7["f1_macro"]), pct(m7["superclass_accuracy"])]


def _lh_test(label: str, x: Dict, y: Dict, mark: str = "") -> List[str]:
    """mark: appended to unadjusted p values (secondary comparisons)."""
    def p(t):
        return fmt_p(t["p_holm"]) if t["p_holm"] is not None else fmt_p(t["cluster"]["p"]) + mark
    return [label, f"{100 * x['diff']:+.2f} ({ci(x['cluster']['ci'], 2, True)})", p(x),
            f"{100 * y['diff']:+.2f} ({ci(y['cluster']['ci'], 2, True)})", p(y)]


def sensitivity(d: Dict) -> Dict[str, Dict]:
    """Models whose LungHist700 learning-rate sensitivity rate differs from the
    pre-specified one and whose runs and tests at that rate are complete:
    {arch: {"lr", "m3", "m7", "t3", "t7"}} (T10, docs/added_analyses_protocol.md A2)."""
    t10 = d.get("T10") or {}
    out = {}
    for a in t10.get("changed", []):
        s3, s7 = t10.get("3c", {}), t10.get("7c", {})
        m3, m7 = s3.get("models", {}).get(a), s7.get("models", {}).get(a)
        if not (m3 and m7 and s3.get("complete") and s7.get("complete")):
            continue
        tests = {t["baseline"]: t for t in s3["tests"] + s3["secondary_tests"]}
        tests7 = {t["baseline"]: t for t in s7["tests"] + s7["secondary_tests"]}
        out[a] = {"lr": m3["lr"], "m3": m3["pooled"], "m7": m7["pooled"],
                  "t3": tests.get(a), "t7": tests7.get(a)}
    return out


def table2(d: Dict) -> str:
    t4, t5 = d["T4"], d["T5"]
    sens = sensitivity(d)
    rows = []
    for a in ORDER + SECONDARY:
        m3 = t4["models"].get(a, {}).get("pooled")
        m7 = t5["models"].get(a, {}).get("pooled")
        if not m3 or not m7:
            continue
        rows.append(_lh_row(LABEL[a], m3, m7))
        if a in sens:
            s = sens[a]
            rows.append(_lh_row(f"{LABEL[a]}, {rate(s['lr'])} (sensitivity)", s["m3"], s["m7"]))
    perf = table(["Model", "3-class accuracy (95% CI)", "3-class macro F1",
                  "7-class accuracy (95% CI)", "7-class macro F1",
                  "7-class superclass accuracy"], rows)
    tests = {t["baseline"]: t for t in t4["tests"] + t4.get("secondary_tests", [])}
    tests7 = {t["baseline"]: t for t in t5["tests"] + t5.get("secondary_tests", [])}
    trows = []
    for a in ORDER[1:] + SECONDARY:
        if a in tests and a in tests7:
            trows.append(_lh_test(LABEL[a], tests[a], tests7[a], "†"))
        s = sens.get(a)
        if s and s["t3"] and s["t7"]:
            trows.append(_lh_test(f"{LABEL[a]}, {rate(s['lr'])} (sensitivity)",
                                  s["t3"], s["t7"], "†"))
    paired = table(["Hybrid-res minus", "3-class Δ accuracy, points (95% CI)", "Holm p",
                    "7-class Δ accuracy, points (95% CI)", "Holm p"], trows)
    extra = ""
    if any(a in tests for a in SECONDARY):
        extra += (" VMamba-T was added after a pre-submission review and is compared "
                  "outside the Holm family († unadjusted p).")
    if sens:
        extra += (" Sensitivity rows: " + ", ".join(
            f"{LABEL[a]} trained at {rate(s['lr'])}" for a, s in sens.items())
            + ", the best of three rates on fold-0 validation (pre-specified grid "
            "10⁻⁴ and 10⁻³; Methods); its Holm p comes from the five comparisons "
            "repeated with each model at its sensitivity rate.")
    return ("**Table 2. LungHist700 results under patient-level five-fold "
            "cross-validation.** Pooled out-of-fold predictions (n = 691 images, "
            "45 patients); each image is predicted once by a model that never saw "
            "its patient. Accuracy and macro F1 in %. The 95% confidence intervals (CI) "
            "and two-sided p values resample patients (percentile bootstrap and "
            "sign-flip randomisation test), Holm-adjusted over the five comparisons "
            "per task. Superclass accuracy: seven-class predictions collapsed to "
            "adenocarcinoma, squamous cell carcinoma or normal." + extra + "\n\n"
            + perf + "\n\n" + paired)


def test_sizes(d: Dict) -> str:
    """Test-set sizes per seed, e.g. '2,195–2,330'."""
    n = sorted({m["n"] for per in d.get("T1", {}).values() for m in per.values()})
    if not n:
        return "—"
    return f"{n[0]:,}" if n[0] == n[-1] else f"{n[0]:,}–{n[-1]:,}"


def table1(d: Dict) -> str:
    s = d.get("summary", {}).get("T1", {})
    rows = []
    for a in ["hybrid_res", "hybrid_res_xattn", "grouped"] + ORDER[1:] + SECONDARY:
        m = s.get(a)
        if not m:
            continue
        def mr(k):
            # Mean and range over seeds: with three seeds a range says more
            # than a standard deviation (journal statistical guidelines).
            v = m[k]
            return (f"{100 * v['mean']:.2f} ({100 * v['min']:.2f}–{100 * v['max']:.2f})"
                    if v["n"] > 1 else f"{100 * v['mean']:.2f}")
        rows.append([LABEL[a], str(m["accuracy"]["n"]), mr("accuracy"),
                     mr("balanced_accuracy"), mr("f1_macro"), mr("auc_macro")])
    perf = table(["Model", "Seeds", "Accuracy", "Balanced accuracy", "Macro F1",
                  "Macro AUC"], rows)
    trows = []
    sec = d.get("T2_secondary", {})
    for seed, res in sorted(d.get("T2", {}).items(), key=lambda kv: int(kv[0])):
        for r in res + sec.get(seed, []):
            trows.append([seed, LABEL[r["baseline"]],
                          f"{100 * r['diff']:+.2f} ({ci(r['cluster']['ci'], 2, True)})",
                          fmt_p(r["cluster"]["p"]), fmt_p(r["p_holm"])])
    paired = table(["Seed", "Hybrid-res minus", "Δ accuracy, points (95% CI)", "p",
                    "Holm p"], trows)
    extra = (" VMamba-T was added after a pre-submission review and is compared outside "
             "the Holm family (unadjusted p only)." if sec else "")
    return ("**Table 1. LC25000 results under the leakage-controlled (grouped) "
            "split.** Mean (minimum–maximum) over three training seeds, in %; the "
            "single-seed variants show one value. Each seed uses its own group-level "
            f"partition ({test_sizes(d)} test images). Paired tests per seed: the 95% "
            "confidence interval (CI) and the two-sided p value resample recovered "
            "source groups (percentile bootstrap and sign-flip randomisation test), "
            "Holm-adjusted over the five comparisons per seed." + extra + " AUC, area "
            "under the receiver operating characteristic curve (one-vs-rest, "
            "macro-averaged).\n\n"
            + perf + "\n\n" + paired)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", default=str(ROOT / "results/_paper/tables.json"))
    ap.add_argument("--table", choices=("1", "2", "all"), default="all")
    ap.add_argument("--inject", default=None, metavar="MANUSCRIPT",
                    help="Replace the text between <!-- TABLEn:BEGIN --> and "
                         "<!-- TABLEn:END --> markers in this file instead of printing.")
    args = ap.parse_args()
    d = json.loads(Path(args.tables).read_text())
    render = {"1": table1, "2": table2}
    wanted = ["1", "2"] if args.table == "all" else [args.table]
    if args.inject:
        path = Path(args.inject)
        text = path.read_text(encoding="utf-8")
        for n in wanted:
            pat = re.compile(rf"(<!-- TABLE{n}:BEGIN -->\n).*?(<!-- TABLE{n}:END -->)", re.S)
            if pat.search(text):
                text = pat.sub(lambda m: m.group(1) + render[n](d) + "\n" + m.group(2), text)
                print(f"Table {n} injected into {path}")
        path.write_text(text, encoding="utf-8")
    else:
        print("\n\n".join(render[n](d) for n in wanted))


if __name__ == "__main__":
    main()
