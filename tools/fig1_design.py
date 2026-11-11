#!/usr/bin/env python3
"""Fig. 1: the residual CNN-Mamba hybrid (a) and the evaluation design (b).

Drawn with matplotlib in the journal style of tools/figstyle.py (7 pt Arial,
lines of at least 1 pt, TrueType fonts, 180 mm wide). Boxes are sized from
the measured text, and the script refuses to write the figure if any text
leaves its box or overlaps other text. The numbers in panel b are read from
the result files (group count, agreement with LC25000-clean).

Usage:
    python3 tools/fig1_design.py [--out docs/manuscript/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

W_MM = 180.0
PITCH = fs.FONT_PT * 1.25 * 25.4 / 72      # line pitch in mm (7 pt, 1.25 leading)
PAD_X, PAD_Y = 1.5, 1.2                     # box padding, mm
GAP = 4.0                                   # gap between columns, mm
DASH_GAP = 1.2                              # space around a dashed (Grad-CAM++) group, mm
CNN_FILL, MAMBA_FILL = "#E3EEF8", "#FBE5D8"  # light blue / light vermillion


@dataclass
class Box:
    lines: List[str]
    fill: str = "white"
    x: float = 0.0
    y: float = 0.0      # lower-left corner, mm
    w: float = 0.0
    h: float = 0.0
    dashed: List[Tuple[int, int]] = field(default_factory=list)  # line ranges (first, last)
    texts: list = field(default_factory=list)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def right(self) -> float:
        return self.x + self.w


class Canvas:
    def __init__(self, height_mm: float):
        self.fig = plt.figure(figsize=(W_MM * fs.MM, height_mm * fs.MM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W_MM)
        self.ax.set_ylim(0, height_mm)
        self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()
        self.texts = []

    def width(self, s: str, bold: bool = False) -> float:
        t = self.ax.text(0, 0, s, fontsize=fs.FONT_PT, fontweight="bold" if bold else "normal")
        w = t.get_window_extent(self.r).width / self.fig.dpi * 25.4
        t.remove()
        return w

    def text(self, x, y, s, ha="left", va="center", bold=False):
        t = self.ax.text(x, y, s, fontsize=fs.FONT_PT, ha=ha, va=va,
                         fontweight="bold" if bold else "normal")
        self.texts.append(t)
        return t

    @staticmethod
    def offsets(box: Box) -> List[float]:
        """Distance from the box's inner top to the top of each line; a dashed
        group gets DASH_GAP of extra space above and below, so its outline
        never crosses the neighbouring lines."""
        out, y = [], 0.0
        for i in range(len(box.lines)):
            if any(i == f for f, _ in box.dashed):
                y += DASH_GAP
            out.append(y)
            y += PITCH
            if any(i == last for _, last in box.dashed):
                y += DASH_GAP
        return out + [y]  # last entry: total text height

    def size(self, box: Box) -> None:
        box.w = max(self.width(s) for s in box.lines) + 2 * PAD_X
        box.h = self.offsets(box)[-1] + 2 * PAD_Y

    def draw(self, box: Box, centred: bool = False) -> None:
        self.ax.add_patch(FancyBboxPatch((box.x, box.y), box.w, box.h,
                                         boxstyle="round,pad=0,rounding_size=1.0",
                                         lw=fs.LINE_PT, ec=fs.BLACK, fc=box.fill))
        off = self.offsets(box)
        for i, s in enumerate(box.lines):
            yc = box.top - PAD_Y - off[i] - 0.5 * PITCH
            box.texts.append(self.text(box.cx if centred else box.x + PAD_X, yc, s,
                                       ha="center" if centred else "left"))
        for first, last in box.dashed:
            y1 = box.top - PAD_Y - off[first] + DASH_GAP / 2
            y0 = box.top - PAD_Y - off[last] - PITCH - DASH_GAP / 2
            self.ax.add_patch(Rectangle((box.x + 0.6, y0), box.w - 1.2, y1 - y0, fill=False,
                                        lw=fs.LINE_PT, ls=(0, (3, 2)), ec=fs.BLACK))

    def line_y(self, box: Box, i: int) -> float:
        """Vertical centre of line i of a sized and placed box."""
        return box.top - PAD_Y - self.offsets(box)[i] - 0.5 * PITCH

    def line(self, pts, colour=fs.BLACK, lw=fs.LINE_PT):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=colour, lw=lw, solid_capstyle="projecting",
                     solid_joinstyle="miter")

    def arrow(self, a, b, colour=fs.BLACK, lw=fs.LINE_PT):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6,
                                          lw=lw, color=colour, shrinkA=0, shrinkB=0.3))

    def wrap(self, s: str, width_mm: float, indent: str = "") -> List[str]:
        """Greedy word wrap by measured width; continuation lines get `indent`."""
        out, cur = [], ""
        for word in s.split():
            trial = (cur + " " + word) if cur else word
            if cur and self.width(trial) > width_mm:
                out.append(cur)
                cur = indent + word
            else:
                cur = trial
        return out + [cur]

    def check(self, boxes: List[Box]) -> None:
        """Every box's text inside its box; no two texts overlap."""
        mm = lambda bb: (bb.x0 / self.fig.dpi * 25.4, bb.y0 / self.fig.dpi * 25.4,  # noqa: E731
                         bb.x1 / self.fig.dpi * 25.4, bb.y1 / self.fig.dpi * 25.4)
        self.fig.canvas.draw()
        problems = []
        for b in boxes:
            for t in b.texts:
                x0, y0, x1, y1 = mm(t.get_window_extent(self.r))
                if x0 < b.x - 0.05 or x1 > b.right + 0.05 or y0 < b.y - 0.05 or y1 > b.top + 0.05:
                    problems.append(f"text {t.get_text()!r} leaves its box")
        ext = [(t.get_text(), mm(t.get_window_extent(self.r))) for t in self.texts]
        for i in range(len(ext)):
            for j in range(i + 1, len(ext)):
                (s1, a), (s2, b) = ext[i], ext[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 0.1 and oy > 0.4:  # adjacent lines touch by design
                    problems.append(f"texts overlap: {s1!r} / {s2!r}")
        for _, (x0, y0, x1, y1) in ext:
            if x0 < 0 or x1 > W_MM or y0 < 0:
                problems.append("text outside the figure")
        if problems:
            raise SystemExit("Fig. 1 layout problems:\n  " + "\n  ".join(problems))


def numbers() -> Tuple[int, float]:
    stats = json.loads((ROOT / "docs/manuscript/figures/figS1_grouping_stats.json").read_text())
    groups = sum(v["groups"] for v in stats["per_class"].values())
    ari = json.loads((ROOT / "results/_paper/grouping_agreement.json").read_text())["ari"]
    return groups, ari


def build(out: Path) -> None:
    fs.apply()
    groups, ari = numbers()
    H = 111.0
    c = Canvas(H)

    # ------------------------------------------------------------ panel a
    top_a = H - 1.0
    c.text(0.5, top_a - 1.5, "a", bold=True)
    c.text(4.0, top_a - 1.5, "Residual CNN–Mamba hybrid (hybrid-res)")
    inp = Box(["H&E image", "224 × 224 × 3"])
    bb = Box(["EfficientNet-B0", "ImageNet-pretrained", "(learning rate × 0.1)"])
    s1 = Box(["Stage 1: 24 channels,", "56 × 56"])
    s4 = Box(["Stage 4: 320 channels,", "7 × 7"])
    s3 = Box(["Stage 3: 112 channels,", "14 × 14 (stride 16)"])
    cnn = Box(["CNN pathway",
               "Each stage: average-pool to 7 × 7,",
               "1 × 1 conv-BN-ReLU → 64 channels",
               "Concatenate → 49 tokens × 192, LN",
               "Mean over tokens → CNN descriptor (192)"],
              fill=CNN_FILL, dashed=[(2, 2)])
    mamba = Box(["Feature Mamba branch (on stage 3)",
                 "196 positions: Linear 112 → 192, LN,",
                 "+ learned positional embedding",
                 "4 × Vision Mamba block:",
                 "    LN → SS2D (4 scan orders, SSM) → +",
                 "    LN → MLP (768) → +",
                 "LN → mean over tokens → g (192)"],
                fill=MAMBA_FILL, dashed=[(3, 5)])
    delta = Box(["δ = Wg + b", "W = 0 and b = 0 at initialisation",
                 "(the model starts as the CNN pathway)"])
    head = Box(["f = LN(CNN descriptor + δ)",
                "Linear 192 → 128, GELU, dropout 0.3,",
                "Linear 128 → C → class probabilities"])
    for b in (inp, bb, s1, s4, s3, cnn, mamba, delta, head):
        c.size(b)
    # Columns, left to right; boxes in a column share a width.
    col_a = max(inp.w, bb.w)
    inp.w = bb.w = col_a
    col_b = max(s1.w, s4.w, s3.w)
    s1.w = s4.w = s3.w = col_b
    col_c = max(cnn.w, mamba.w)
    cnn.w = mamba.w = col_c
    x_a = 0.5
    x_b = x_a + col_a + GAP + 1.0   # room for the bus and an arrowhead
    x_c = x_b + col_b + GAP + 1.0
    x_d = x_c + col_c + GAP + 1.0
    # Lanes: CNN on top, Mamba below.
    lane1_top = top_a - 5.0
    cnn.x, cnn.y = x_c, lane1_top - cnn.h
    mamba.x, mamba.y = x_c, cnn.y - 6.0 - mamba.h
    inp.x, inp.y = x_a, cnn.top - inp.h
    bb.x, bb.y = x_a, mamba.cy - bb.h / 2
    s1.x, s1.y = x_b, cnn.top - s1.h
    s4.x, s4.y = x_b, s1.y - 2.5 - s4.h
    s3.x, s3.y = x_b, mamba.cy - s3.h / 2
    plus_r = 2.0
    plus_x, plus_y = x_d + plus_r, cnn.cy
    delta.x, delta.y = x_d, mamba.cy - delta.h / 2
    head.x, head.y = plus_x + plus_r + 3.5, cnn.cy - head.h / 2
    if head.right > W_MM - 0.5 or delta.right > W_MM - 0.5:
        raise SystemExit(f"panel a too wide: head ends at {head.right:.1f} mm, "
                         f"delta at {delta.right:.1f} mm")
    for b in (inp, bb, s1, s4, s3, delta, head):
        c.draw(b, centred=b in (inp, bb, s1, s4, s3))
    c.draw(cnn)
    c.draw(mamba)
    c.ax.add_patch(Circle((plus_x, plus_y), plus_r, fill=False, lw=fs.LINE_PT, ec=fs.BLACK))
    c.text(plus_x, plus_y, "+", ha="center")
    # Data flow.
    c.arrow((inp.cx, inp.y), (bb.cx, bb.top))
    # Backbone to the three stages through a vertical bus.
    bus = bb.right + 1.8
    c.line([(bb.right, bb.cy), (bus, bb.cy)])
    c.line([(bus, s3.cy), (bus, s1.cy)])
    for s in (s1, s4, s3):
        c.arrow((bus, s.cy), (s.x, s.cy))
    # Stages to the CNN pathway (stage 3 routed up the gap) and to the Mamba branch.
    c.arrow((s1.right, s1.cy), (cnn.x, s1.cy))
    c.arrow((s4.right, s4.cy), (cnn.x, s4.cy))
    riser, y_in = s3.right + 1.8, cnn.y + 2.0
    c.line([(s3.right, s3.cy + 1.0), (riser, s3.cy + 1.0), (riser, y_in)])
    c.arrow((riser, y_in), (cnn.x, y_in))
    c.arrow((s3.right, s3.cy - 1.0), (mamba.x, s3.cy - 1.0))
    c.arrow((cnn.right, plus_y), (plus_x - plus_r, plus_y))
    c.arrow((mamba.right, mamba.cy), (delta.x, mamba.cy))
    c.arrow((plus_x, delta.top), (plus_x, plus_y - plus_r), colour=fs.VERMILLION, lw=1.4)
    c.text(plus_x + 1.5, (delta.top + plus_y - plus_r) / 2, "Ablation: δ := 0")
    c.arrow((plus_x + plus_r, plus_y), (head.x, plus_y))
    # Notes under the lanes, across the panel.
    notes = ("Dashed outlines mark the layers used for Grad-CAM++ (CNN 7 × 7, Mamba "
             "14 × 14); the combined map is the mean of the two. Hybrid-res-xattn forms g "
             "by gated bidirectional cross-attention between the 49 CNN and 196 Mamba "
             "tokens (4 heads); the variant was chosen on validation accuracy only.")
    y = mamba.y - 2.5
    for line in c.wrap(notes, W_MM - 1.0):
        c.text(0.5, y, line)
        y -= PITCH
    bottom_a = y + PITCH / 2

    # ------------------------------------------------------------ panel b
    top_b = bottom_a - 4.0
    c.text(0.5, top_b - 1.5, "b", bold=True)
    c.text(4.0, top_b - 1.5, "Evaluation design (one training recipe for every model)")
    cols = [
        ("LC25000 (15,000 images, three classes)",
         [f"Source groups recovered by an embedding shortlist and SIFT/RANSAC: "
          f"{groups} groups (ARI {ari:.3f} with LC25000-clean)",
          "Grouped 70/15/15 split, three seeds (leakage-controlled)",
          "Per-file split (leaky reference)",
          "Inference unit: source group (about 107 per test set)",
          "Clean-subset sensitivity analysis"]),
        ("LungHist700 (691 images, 45 patients)",
         ["Three classes: patient-level five-fold CV; image-level five-fold CV as the "
          "leaky reference",
          "Seven-class differentiation grading: patient-level five-fold CV",
          "Pooled out-of-fold predictions; inference unit: patient",
          "20× and 40× together; a patient never spans folds"]),
        ("Transfer, explanations and cost",
         ["External test: LC25000-trained models on LungHist700 at 20× and 40×, "
          "without adaptation",
          "Pathway attribution: size and removal of δ",
          "Faithfulness: deletion and insertion AUC against a random map, with two "
          "deletion canvases",
          "Comparators: CNN pathway, original hybrid, EfficientNet-B0, ViT-S, Swin-T, "
          "VMamba-T, Phikon probe",
          "Parameters, operations, latency and memory"]),
    ]
    col_w = (W_MM - 1.0 - 2 * 3.0) / 3
    boxes_b = []
    for k, (heading, bullets) in enumerate(cols):
        lines = [heading]
        for bl in bullets:
            lines += c.wrap("• " + bl, col_w - 2 * PAD_X, indent="   ")
        boxes_b.append(Box(lines))
    h_b = max(len(b.lines) for b in boxes_b) * PITCH + 2 * PAD_Y
    for k, b in enumerate(boxes_b):
        b.w, b.h = col_w, h_b
        b.x, b.y = 0.5 + k * (col_w + 3.0), top_b - 5.0 - h_b
        c.draw(b)
    if boxes_b[0].y < 0.5:
        raise SystemExit(f"figure too short: panel b ends at {boxes_b[0].y:.1f} mm")
    c.check([inp, bb, s1, s4, s3, cnn, mamba, delta, head] + boxes_b)
    unused = boxes_b[0].y - 0.5
    print(f"  layout ok; {unused:.1f} mm spare at the bottom")
    fs.save(c.fig, out, "fig1_design", expected_width_mm=W_MM)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs/manuscript/figures"))
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
