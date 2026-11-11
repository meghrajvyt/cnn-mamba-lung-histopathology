"""The submitted figure PDFs follow the Scientific Reports figure rules
(docs/manuscript/format_spec.md F1-F5, F10): one embedded TrueType sans-serif
typeface, every text run at 7 pt, black or white text only, no stroke thinner
than 1 pt, one page no wider than 180 mm."""
import math
import re
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf")
from pypdf.generic import ContentStream  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    "docs/manuscript/figures/fig1_design.pdf",
    "results/_paper/figures/fig_leakage.pdf",
    "results/_paper/figures/fig_confusion.pdf",
    "results/_paper/figures/fig_external.pdf",
    "results/_paper/figures/fig_faithfulness.pdf",
    "results/_paper/figures/fig_cost.pdf",
    "docs/manuscript/figures/figS1_grouping.pdf",
    "docs/manuscript/figures/figS2_examples.pdf",
    "docs/manuscript/figures/figS3_examples_lh700.pdf",
]
ALLOWED_FONTS = {"ArialMT", "Arial-BoldMT", "LiberationSans", "LiberationSans-Bold",
                 "Helvetica", "Helvetica-Bold"}
MAX_WIDTH_PT = 180 / 25.4 * 72
STROKE_OPS = {b"S", b"s", b"B", b"B*", b"b", b"b*"}


def _fonts(page):
    res = page["/Resources"].get_object()
    fonts = (res.get("/Font") or {})
    fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
    out = []
    for ref in fonts.values():
        f = ref.get_object()
        sub, base = str(f["/Subtype"]), str(f.get("/BaseFont", ""))
        if sub == "/Type0":
            d = f["/DescendantFonts"][0].get_object()
            sub = str(d["/Subtype"])
            desc = d["/FontDescriptor"].get_object()
        else:
            desc = f.get("/FontDescriptor")
            desc = desc.get_object() if desc is not None else {}
        out.append((re.sub(r"^[A-Z]{6}\+", "", base.lstrip("/")), sub,
                    "/FontFile2" in desc))
    return out


def _walk(reader, page):
    """Yield (event, value) for text sizes, text colours and stroke widths,
    following Form XObjects with the inherited graphics state."""
    res = page["/Resources"].get_object()
    xobj = res.get("/XObject") or {}
    xobj = xobj.get_object() if hasattr(xobj, "get_object") else xobj

    def run(contents, state, depth=0):
        stack = []
        lw, scale, fill = state
        for ops, op in ContentStream(contents, reader).operations:
            if op == b"q":
                stack.append((lw, scale, fill))
            elif op == b"Q" and stack:
                lw, scale, fill = stack.pop()
            elif op == b"w":
                lw = float(ops[0])
            elif op == b"cm":
                a, b, c, d = (float(v) for v in ops[:4])
                scale *= math.sqrt(abs(a * d - b * c))
            elif op == b"rg":
                fill = tuple(round(float(v), 3) for v in ops)
            elif op == b"g":
                fill = (round(float(ops[0]), 3),) * 3
            elif op == b"Tf":
                yield "size", float(ops[1])
            elif op in (b"Tj", b"TJ", b"'", b'"'):
                yield "text_colour", fill
            elif op in STROKE_OPS:
                yield "stroke", lw * scale
            elif op == b"Do" and depth < 4:
                o = xobj.get(ops[0])
                o = o.get_object() if o is not None else None
                if o is not None and o.get("/Subtype") == "/Form":
                    yield from run(o, (lw, scale, fill), depth + 1)

    yield from run(page.get_contents(), (1.0, 1.0, (0.0, 0.0, 0.0)))


@pytest.fixture(scope="module", params=FIGURES, ids=[Path(f).stem for f in FIGURES])
def figure(request):
    path = ROOT / request.param
    assert path.is_file(), f"{request.param} missing: run the figure scripts"
    reader = pypdf.PdfReader(str(path))
    return request.param, reader


def test_one_page_at_most_180_mm(figure):
    name, reader = figure
    assert len(reader.pages) == 1
    assert float(reader.pages[0].mediabox.width) <= MAX_WIDTH_PT + 0.5, name


def test_fonts_are_embedded_truetype_arial(figure):
    name, reader = figure
    fonts = _fonts(reader.pages[0])
    assert fonts, f"{name}: no fonts (text converted to outlines?)"
    for base, sub, embedded in fonts:
        assert sub in ("/CIDFontType2", "/TrueType"), f"{name}: {base} is {sub}"
        assert base in ALLOWED_FONTS, f"{name}: font {base}"
        assert embedded, f"{name}: {base} not embedded"


def test_every_text_run_is_7_pt_black_or_white(figure):
    name, reader = figure
    events = list(_walk(reader, reader.pages[0]))
    sizes = {v for e, v in events if e == "size"}
    assert sizes == {7.0}, f"{name}: text sizes {sorted(sizes)}"
    colours = {v for e, v in events if e == "text_colour"}
    assert colours <= {(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)}, f"{name}: text colours {colours}"


def test_no_stroke_thinner_than_1_pt(figure):
    name, reader = figure
    widths = [v for e, v in _walk(reader, reader.pages[0]) if e == "stroke"]
    # Image-only figures (Supplementary Figs. S2, S3) may draw no lines at all.
    if widths:
        assert min(widths) >= 1.0 - 1e-6, f"{name}: thinnest stroke {min(widths):.3f} pt"
