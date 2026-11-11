#!/usr/bin/env python3
"""Paper 1 main-text figures 2-6, drawn from results/_paper/tables.json
(run tools/paper_tables.py first) in the journal style of tools/figstyle.py.

  fig_leakage       Fig. 2  leakage-controlled vs leaky accuracy per model
                            (a LC25000 seed 42, b LungHist700 three-class)
  fig_confusion     Fig. 3  pooled out-of-fold confusion matrices of the
                            selected hybrid (a three-class, b seven-class)
  fig_external      Fig. 4  external accuracy at 20x and 40x
  fig_faithfulness  Fig. 5  Grad-CAM++ faithfulness relative to a random map
                            (a, b LC25000 blur / mean canvas; c, d LungHist700)
  fig_cost          Fig. 6  accuracy against parameters and latency
                            (a LC25000, b LungHist700)

Every plotted value is read from tables.json, so the figures show exactly
the numbers of the tables and the text. Fig. 1 is tools/fig1_design.py.

Usage:
    python3 tools/paper_figures.py [--tables results/_paper/tables.json]
        [--out results/_paper/figures] [--only fig_leakage ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import figstyle as fs  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, ScalarFormatter  # noqa: E402

LABEL = fs.LABEL
# Point and bar colours: the proposed hybrid in vermillion, everything else
# in neutral or blue tones (colour is redundant with the axis labels).
MODEL_COLOUR = {"hybrid_res": fs.VERMILLION, "hybrid_res_xattn": fs.ORANGE,
                "grouped": "#CC79A7", "cnn_only": fs.BLUE, "effnet_b0": fs.SKY,
                "vit_small": fs.GREEN, "swin_tiny": fs.BLACK, "phikon_lp": fs.GREY,
                "vmamba_tiny": fs.YELLOW}
LEAK_ORDER = ["hybrid_res", "cnn_only", "effnet_b0", "vit_small", "swin_tiny", "phikon_lp"]
# VMamba-T, added after the pre-submission review, comes last.
EXT_ORDER = ["hybrid_res", "hybrid_res_xattn", "grouped", "cnn_only", "effnet_b0",
             "vit_small", "swin_tiny", "phikon_lp", "vmamba_tiny"]
FAITH_ROWS = [("hybrid_res", "cnn"), ("hybrid_res", "mamba"), ("hybrid_res", "combined"),
              ("cnn_only", "cnn"), ("effnet_b0", "cnn"), ("vit_small", "vit"),
              ("swin_tiny", "swin")]
MAP_LABEL = {"cnn": "CNN map", "mamba": "Mamba map", "combined": "combined map"}
CLASS3 = ["ACA", "Normal", "SCC"]                       # lung_aca, lung_n, lung_scc
CLASS7 = ["ACA-WD", "ACA-MD", "ACA-PD", "Normal", "SCC-WD", "SCC-MD", "SCC-PD"]


# --------------------------------------------------------------- Fig. 2
def fig_leakage(T: Dict, out: Path) -> None:
    t3 = T["T3"]
    fig = fs.figure(180, 58)
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.08})
    panels = [("lc25000", "grouped", "leaky",
               "LC25000: source-group versus per-file split (seed 42)"),
              ("lunghist700", "patient", "image",
               "LungHist700: patient- versus image-level folds")]
    names = [LABEL[a] for a in LEAK_ORDER]
    y = np.arange(len(LEAK_ORDER))[::-1]
    for k, (ax, (ds, lo_k, hi_k, title)) in enumerate(zip(axes, panels)):
        lo = np.array([100 * t3[f"{ds}/{a}"][lo_k] for a in LEAK_ORDER])
        hi = np.array([100 * t3[f"{ds}/{a}"][hi_k] for a in LEAK_ORDER])
        for yi, a, b in zip(y, lo, hi):
            ax.plot([a, b], [yi, yi], color=fs.GREY, lw=fs.LINE_PT, zorder=1)
        ax.scatter(lo, y, s=16, color=fs.BLACK, linewidths=fs.LINE_PT, zorder=3,
                   label="Leakage-controlled")
        ax.scatter(hi, y, s=16, facecolors="white", edgecolors=fs.BLACK,
                   linewidths=fs.LINE_PT, zorder=3, label="Leaky")
        # Inflation in percentage points, in a column right of the axes.
        for yi, a, b in zip(y, lo, hi):
            ax.annotate(f"+{b - a:.2f}", xy=(1, yi), xycoords=("axes fraction", "data"),
                        xytext=(3, 0), textcoords="offset points", ha="left", va="center")
        ax.annotate("Inflation", xy=(1, 1), xycoords="axes fraction", xytext=(3, 2),
                    textcoords="offset points", ha="left", va="bottom")
        span = hi.max() - lo.min()
        ax.set_xlim(lo.min() - 0.12 * span, min(100.0, hi.max()) + 0.12 * span
                    if hi.max() < 100 else 100 + 0.06 * span)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_ylim(-0.6, len(LEAK_ORDER) - 0.4)
        ax.set_yticks(y, names if k == 0 else [""] * len(y))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(title)
    # Lower left of panel a is empty: every model scores above 98.5% there.
    axes[0].legend(loc="lower left", handletextpad=0.2)
    fs.panel_letters(fig, axes, "ab")
    fs.save(fig, out, "fig_leakage")


# --------------------------------------------------------------- Fig. 3
def fig_confusion(T: Dict, out: Path) -> None:
    sel = T.get("selected", "hybrid_res")
    m3 = np.array(T["T4"]["models"][sel]["pooled"]["confusion"])
    m7 = np.array(T["T5"]["models"][sel]["pooled"]["confusion"])
    fig = fs.figure(180, 80)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 7])
    ax3, ax7 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    im = None
    for ax, cm, labels, title in ((ax3, m3, CLASS3, "Three classes"),
                                  (ax7, m7, CLASS7, "Seven classes (differentiation grade)")):
        frac = cm / cm.sum(1, keepdims=True).clip(1)
        im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if frac[i, j] > 0.55 else fs.BLACK)
        ax.set_xticks(range(len(labels)), labels, rotation=45 if len(labels) > 3 else 0,
                      ha="right" if len(labels) > 3 else "center")
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        ax.set_anchor("N")  # top-align the two panels (and their letters)
    cb = fig.colorbar(im, ax=ax7, ticks=[0, 0.5, 1], shrink=0.75, aspect=25, pad=0.02,
                      anchor=(0.0, 1.0))
    cb.outline.set_linewidth(fs.LINE_PT)
    cb.set_label("Fraction of true class")
    fs.panel_letters(fig, [ax3, ax7], "ab")
    fs.save(fig, out, "fig_confusion")


# --------------------------------------------------------------- Fig. 4
def fig_external(T: Dict, out: Path) -> None:
    t6, maj = T["T6"], T["T6_majority"]
    models = [a for a in EXT_ORDER if a in t6]
    fig = fs.figure(120, 66)
    ax = fig.subplots()
    x = np.arange(len(models))
    w = 0.38
    for off, mag, col, lab in ((-w / 2, "20x", fs.SKY, "20×"), (w / 2, "40x", fs.BLUE, "40×")):
        vals = [[100 * v["accuracy"] for v in t6[a][mag].values()] for a in models]
        ax.bar(x + off, [np.mean(v) for v in vals], w, color=col, label=lab, zorder=1)
        for xi, v in zip(x, vals):
            ax.scatter(np.full(len(v), xi + off), v, s=5, color=fs.BLACK, linewidths=0,
                       zorder=3)
        # Always predicting the most frequent class, drawn over each bar.
        ax.hlines(np.full(len(x), 100 * maj[mag]), x + off - w / 2, x + off + w / 2,
                  colors=fs.BLACK, linestyles=(0, (1, 1.5)), lw=fs.LINE_PT, zorder=2)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.set_xticks(x, [LABEL[a] for a in models], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel("Accuracy (%)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=fs.BLACK, ls=(0, (1, 1.5)), lw=fs.LINE_PT))
    labels.append("Majority class")
    handles.append(Line2D([], [], color=fs.BLACK, marker="o", ms=2.5, ls="none", mew=0))
    labels.append("Seed")
    ax.legend(handles, labels, loc="upper left", ncol=4)
    fs.save(fig, out, "fig_external")


# --------------------------------------------------------------- Fig. 5
def _faith_names() -> List[str]:
    return [LABEL[a] + (f", {MAP_LABEL[m]}" if a == "hybrid_res" else "") for a, m in FAITH_ROWS]


def fig_faithfulness(T: Dict, out: Path) -> None:
    t8, lh = T["T8"], T["T8c"]
    fig = fs.figure(180, 106)
    axes = fig.subplots(2, 2, gridspec_kw={"wspace": 0.06, "hspace": 0.12})
    y = np.arange(len(FAITH_ROWS))[::-1]
    names = _faith_names()
    panels = [(axes[0][0], "LC25000, blur canvas", "lc", "blur"),
              (axes[0][1], "LC25000, mean-colour canvas", "lc", "mean"),
              (axes[1][0], "LungHist700, blur canvas", "lh", "faithfulness"),
              (axes[1][1], "LungHist700, mean-colour canvas", "lh", "faithfulness_mean")]
    for k, (ax, title, ds, key) in enumerate(panels):
        for yi, (a, m) in zip(y, FAITH_ROWS):
            hyb = a == "hybrid_res"
            if ds == "lc":
                vals = [r["minus_random"] for r in t8[key].values()
                        if r["arch"] == a and r["map"] == m]
                ax.barh(yi, np.mean(vals), height=0.62, zorder=1,
                        color="#F2B48C" if hyb else fs.LIGHT_GREY)
                ax.scatter(vals, [yi] * len(vals), s=7, color=fs.BLACK, linewidths=0,
                           zorder=3)
            else:
                r = lh[key][f"{a}/{m}"]
                ax.errorbar(r["mean"], yi, xerr=[[r["mean"] - r["ci"][0]], [r["ci"][1] - r["mean"]]],
                            fmt="o", ms=3, mew=0, color=fs.VERMILLION if hyb else fs.BLACK,
                            ecolor=fs.VERMILLION if hyb else fs.BLACK, elinewidth=fs.LINE_PT,
                            capthick=fs.LINE_PT, capsize=2, zorder=3)
        ax.axvline(0, color=fs.BLACK, lw=fs.LINE_PT, zorder=2)
        ax.set_ylim(-0.6, len(FAITH_ROWS) - 0.4)
        ax.set_yticks(y, names if k % 2 == 0 else [""] * len(y))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(title)
        # Every panel has its own scale, so every panel carries the axis label.
        ax.set_xlabel("Difference in insertion − deletion AUC from random map")
    handles = [Patch(color=fs.LIGHT_GREY, label="LC25000: mean over seeds"),
               Line2D([], [], color=fs.BLACK, marker="o", ms=2.5, ls="none", mew=0,
                      label="LC25000: seed"),
               Line2D([], [], color=fs.BLACK, marker="o", ms=3, mew=0, lw=fs.LINE_PT,
                      label="LungHist700: mean over patients, 95% CI")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fs.panel_letters(fig, [axes[0][0], axes[0][1], axes[1][0], axes[1][1]], "abcd")
    fs.save(fig, out, "fig_faithfulness")


# --------------------------------------------------------------- Fig. 6
def fig_cost(T: Dict, out: Path) -> None:
    e = dict(T["T9"]["models"])
    if "hybrid" in e and "grouped" not in e:  # efficiency.py's name for the original hybrid
        e["grouped"] = e["hybrid"]
    t1, t4 = T["T1"], T["T4"]["models"]
    full = max(len(per) for per in t1.values())
    acc_lc = {a: 100 * np.mean([m["accuracy"] for m in per.values()])
              for a, per in t1.items() if len(per) == full and a in e}
    acc_lh = {a: 100 * v["pooled"]["accuracy"] for a, v in t4.items() if a in e}
    fig = fs.figure(180, 66)
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.1})

    def area(ms):  # marker area (pt^2) grows linearly with latency
        return 12 + 3.5 * ms
    for ax, acc, title in ((axes[0], acc_lc, "LC25000 (mean over three seeds)"),
                           (axes[1], acc_lh,
                            "LungHist700, three classes (pooled cross-validation)")):
        pts = sorted(((e[a]["params"] / 1e6, v, a) for a, v in acc.items()))
        xmin, xmax = 2.6, 60.0
        lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
        pad = 0.18 * (hi - lo)
        ymin, ymax = lo - pad, hi + pad

        def frac(x, yv):  # position in axes fractions (log x)
            return ((np.log10(x) - np.log10(xmin)) / (np.log10(xmax) - np.log10(xmin)),
                    (yv - ymin) / (ymax - ymin))
        for x, yv, a in pts:
            r = np.sqrt(area(e[a]["latency_b1_ms"]["median"])) / 2  # marker radius, pt
            ax.scatter(x, yv, s=area(e[a]["latency_b1_ms"]["median"]),
                       color=MODEL_COLOUR[a], edgecolors=fs.BLACK, linewidths=fs.LINE_PT,
                       zorder=3)
            fx, fy = frac(x, yv)
            # Label to the right, unless another point sits just right of it at
            # about the same height; then to the left.
            crowded = any(0 < frac(x2, y2)[0] - fx < 0.2 and abs(frac(x2, y2)[1] - fy) < 0.08
                          for x2, y2, a2 in pts if a2 != a)
            ax.annotate(LABEL[a], (x, yv), textcoords="offset points",
                        xytext=(-(r + 2) if crowded else r + 2, 0), va="center",
                        ha="right" if crowded else "left")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 30]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("Parameters (millions, log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
    # Marker-size key, drawn in the empty lower right of panel b.
    ax = axes[1]
    for xk, ms in ((0.70, 10), (0.80, 50), (0.92, 90)):
        ax.scatter([xk], [0.13], s=area(ms), transform=ax.transAxes, clip_on=False,
                   facecolors="white", edgecolors=fs.BLACK, linewidths=fs.LINE_PT, zorder=3)
        ax.annotate(f"{ms} ms", xy=(xk, 0.13), xycoords="axes fraction",
                    xytext=(0, -(np.sqrt(area(90)) / 2 + 2)), textcoords="offset points",
                    ha="center", va="top")
    ax.annotate("Batch-1 latency", xy=(0.81, 0.13), xycoords="axes fraction",
                xytext=(0, np.sqrt(area(90)) / 2 + 3), textcoords="offset points",
                ha="center", va="bottom")
    fs.panel_letters(fig, axes, "ab")
    fs.save(fig, out, "fig_cost")


FIGS = {"fig_leakage": fig_leakage, "fig_confusion": fig_confusion,
        "fig_external": fig_external, "fig_faithfulness": fig_faithfulness,
        "fig_cost": fig_cost}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", default=str(ROOT / "results/_paper/tables.json"))
    ap.add_argument("--out", default=str(ROOT / "results/_paper/figures"))
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS), default=None)
    ap.add_argument("--mag", default="all",
                    help="ignored (the magnification is fixed when tables.json is built); "
                         "accepted so that the experiment queues keep working")
    args = ap.parse_args()
    tj = Path(args.tables)
    if not tj.is_file():
        raise SystemExit("run tools/paper_tables.py first")
    T = json.loads(tj.read_text())
    fs.apply()
    for name in args.only or FIGS:
        FIGS[name](T, Path(args.out))


if __name__ == "__main__":
    main()
