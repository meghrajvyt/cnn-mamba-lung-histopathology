#!/usr/bin/env python3
"""Shared figure style for Paper 1, implementing the Scientific Reports figure
rules of docs/manuscript/format_spec.md:

  F1/F2  one sans-serif typeface (Arial; Liberation Sans or Helvetica as
         fallbacks) at 7 pt in every figure, panel letters included (bold,
         lower case, same size); TrueType fonts embedded (pdf.fonttype 42),
         never Type 3
  F3     sentence case: only the first letter of a label is capitalised
  F4     no line thinner than 1 pt
  F5     white background, no grid, Okabe-Ito colours, no coloured text
  F6     bar charts start at zero (each figure function is responsible)
  F9/F10 vector PDF, 300-dpi PNG preview, width at most 180 mm

Import and call ``apply()`` before creating figures; create figures with
``figure(width_mm, height_mm)`` and write them with ``save()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4            # inches per millimetre
FONT_PT = 7.0
LINE_PT = 1.0
MAX_WIDTH_MM = 180.0
FONTS = ["Arial", "Liberation Sans", "Helvetica"]

# Okabe-Ito palette (colour-blind safe). The proposed hybrid is always
# vermillion; other models are blues and greys.
VERMILLION = "#D55E00"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
YELLOW = "#F0E442"
GREY = "#7F7F7F"
LIGHT_GREY = "#BDBDBD"
BLACK = "#000000"

# Model names as printed in figures (sentence case; matches the text).
LABEL = {"hybrid_res": "Hybrid-res", "hybrid_res_xattn": "Hybrid-res-xattn",
         "grouped": "Original hybrid", "hybrid": "Original hybrid",
         "cnn_only": "CNN pathway", "effnet_b0": "EfficientNet-B0",
         "vit_small": "ViT-S/16", "swin_tiny": "Swin-T", "phikon_lp": "Phikon probe",
         "vmamba_tiny": "VMamba-T"}


def apply() -> None:
    """Set the journal style for every subsequent matplotlib figure."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONTS,
        "font.size": FONT_PT, "axes.titlesize": FONT_PT, "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT, "legend.title_fontsize": FONT_PT,
        "figure.titlesize": FONT_PT, "figure.labelsize": FONT_PT,
        "axes.titleweight": "normal", "axes.titlepad": 3.0, "axes.labelpad": 2.0,
        "axes.linewidth": LINE_PT, "lines.linewidth": LINE_PT,
        "lines.markeredgewidth": LINE_PT, "patch.linewidth": LINE_PT,
        "hatch.linewidth": LINE_PT, "grid.linewidth": LINE_PT,
        "xtick.major.width": LINE_PT, "ytick.major.width": LINE_PT,
        "xtick.minor.width": LINE_PT, "ytick.minor.width": LINE_PT,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": BLACK,
        "text.color": BLACK, "axes.labelcolor": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "legend.columnspacing": 1.0, "errorbar.capsize": 2.0,
        "axes.unicode_minus": True, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "pdf.compression": 9, "savefig.dpi": 300, "figure.dpi": 150,
        "image.interpolation": "antialiased",
    })


def figure(width_mm: float, height_mm: float, **kw):
    """A figure of the given printed size (constrained layout unless
    ``layout`` is given)."""
    if width_mm > MAX_WIDTH_MM + 1e-6:
        raise ValueError(f"figure width {width_mm} mm exceeds {MAX_WIDTH_MM} mm")
    kw.setdefault("layout", "constrained")
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), **kw)


def panel_letters(fig, axes, letters: str = "abcdefgh") -> None:
    """Bold lower-case panel letters, same size as all other text, on the
    title line of each axes and aligned with the left edge of its y-axis
    labels. They are left titles, so constrained layout makes room for them."""
    titles = []
    for ax, letter in zip(axes, letters):
        if letter != letter.lower():
            raise ValueError("panel letters are lower case")
        titles.append((ax, ax.set_title(letter, loc="left", fontweight="bold",
                                        fontsize=FONT_PT)))
    for _ in range(2):  # the layout moves once the letters move
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax, t in titles:
            x0 = min(ax.bbox.x0, ax.yaxis.get_tightbbox(r).x0
                     if ax.yaxis.get_visible() and ax.yaxis.get_tightbbox(r) else ax.bbox.x0)
            t.set_x((x0 - ax.bbox.x0) / ax.bbox.width)
            t.set_ha("left")


def jpeg_images(pdf: Path, quality: int = 92) -> None:
    """Re-encode the raster images of a PDF as JPEG (DCT), keeping text and
    lines vector. Matplotlib stores images losslessly, which makes figures of
    H&E photographs several megabytes; at quality 92 the difference is not
    visible. Only for photographs, never for plots."""
    from pypdf import PdfWriter

    writer = PdfWriter(clone_from=str(pdf))
    for page in writer.pages:
        for img in page.images:
            im = img.image
            if im.mode != "RGB":
                im = im.convert("RGB")
            img.replace(im, quality=quality)
    with open(pdf, "wb") as f:
        writer.write(f)


def save(fig, out_dir: Path, name: str, png: bool = True,
         expected_width_mm: Optional[float] = None,
         jpeg_quality: Optional[int] = None) -> Path:
    """Write <name>.pdf (vector) and <name>.png (300 dpi preview) at the
    figure's own size (no bbox cropping, so the printed width is exact).
    With ``jpeg_quality`` the PDF's raster images are stored as JPEG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w_mm = fig.get_figwidth() / MM
    if w_mm > MAX_WIDTH_MM + 0.01:
        raise ValueError(f"{name}: width {w_mm:.1f} mm exceeds {MAX_WIDTH_MM} mm")
    if expected_width_mm is not None and abs(w_mm - expected_width_mm) > 0.01:
        raise ValueError(f"{name}: width {w_mm:.1f} mm, expected {expected_width_mm} mm")
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf, metadata={"Creator": "tools/figstyle.py", "CreationDate": None})
    if png:
        fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    if jpeg_quality is not None:
        jpeg_images(pdf, jpeg_quality)
    print(f"  wrote {pdf.relative_to(out_dir.parents[1]) if len(out_dir.parents) > 1 else pdf}")
    return pdf
