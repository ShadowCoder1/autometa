"""Figure-region detection: line-art figures, caption assignment, caption scoring (regression tests for the
degenerate-rect / caption-conflict fixes)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pymupdf
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from canopy.ingest.pdf import _overlaps, caption_score, ingest_pdf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "tests" / "fixtures" / "pdfs"
BOCK = PDF_DIR / "bock2005.pdf"


def _errorbar_pdf(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.arange(1, 11)
    ax.errorbar(x, x * 1.5, yerr=1.0, fmt="o-")
    ax.set_xlabel("block")
    ax.set_ylabel("error (deg)")
    fig.savefig(path)
    plt.close(fig)


def test_overlaps_handles_degenerate_rects():
    line = pymupdf.Rect(10, 50, 200, 50)          # zero-height horizontal line
    box = pymupdf.Rect(0, 0, 100, 100)
    assert not box.intersects(line)               # the PyMuPDF behaviour that hid line art
    assert _overlaps(box, line)
    assert not _overlaps(pymupdf.Rect(0, 0, 5, 5), line)


def test_line_art_figure_is_detected(tmp_path):
    pdf = tmp_path / "synth.pdf"
    _errorbar_pdf(pdf)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=100)
    kinds = [f.kind for f in rec.figures]
    assert kinds == ["vector"], kinds
    assert rec.figures[0].n_drawings >= 15


def _two_figure_page(path: Path, captions_below: bool = True) -> None:
    """A page with two stacked line-art graphics and two captions ('Fig. 1 ...' / 'Fig. 2 ...')."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for top, label in ((60, 1), (420, 2)):
        # axes + ticks + a polyline: zero-area strokes, one PDF path each (as matplotlib/R emit them)
        segments = [((80, top + 200), (300, top + 200)), ((80, top), (80, top + 200))]
        for i in range(10):
            segments.append(((80 + i * 22, top + 200), (80 + i * 22, top + 205)))
            segments.append(((75, top + i * 20), (80, top + i * 20)))
        pts = [(80 + i * 22, top + 180 - i * 12) for i in range(10)]
        segments += list(zip(pts, pts[1:]))
        for a, b in segments:
            shape = page.new_shape()
            shape.draw_line(a, b)
            shape.finish(color=(0, 0, 0), width=0.8)
            shape.commit()
        cap_y = top + 225 if captions_below else top - 25
        page.insert_text((80, cap_y), f"Fig. {label} The dependent measure plotted against block for both groups.",
                         fontsize=8)
    doc.save(path)


def test_two_stacked_figures_get_their_own_captions(tmp_path):
    pdf = tmp_path / "two.pdf"
    _two_figure_page(pdf)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=100)
    labels = [f.label for f in rec.figures]
    assert labels == ["Fig. 1", "Fig. 2"], [(f.label, f.bbox) for f in rec.figures]
    assert [f.kind for f in rec.figures] == ["vector", "vector"]
    assert rec.figures[0].bbox.y1 < rec.figures[1].bbox.y0     # not merged into one region


def test_two_stacked_figures_with_captions_above(tmp_path):
    pdf = tmp_path / "two_above.pdf"
    _two_figure_page(pdf, captions_below=False)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=100)
    labels = [f.label for f in rec.figures]
    assert labels == ["Fig. 1", "Fig. 2"], [(f.label, f.bbox) for f in rec.figures]
    assert [f.kind for f in rec.figures] == ["vector", "vector"]


@pytest.mark.parametrize("text,real", [
    ("Fig. 5 The changes in proprioceptive estimates are plotted as a function of block.", True),
    ("Fig. 5 the changes were plotted as a function of block.", False),
    ("Figure 2 illustrates the mean error in each block.", False),
    ("Fig. 3. Mean pointing errors (± SD) per episode.", True),
    ("Fig. 4 | Adaptation curves for both groups.", True),
])
def test_caption_score_bands(text, real):
    assert (caption_score(text) >= 0.6) is real


def test_bock_roster_is_stable(tmp_path):
    rec = ingest_pdf(BOCK, tmp_path / "bock", render_pages=False, fig_dpi=100)
    assert [(f.id, f.page) for f in rec.figures] == [("fig01", 3), ("fig02", 4)]
    assert [(t.id, t.page) for t in rec.tables] == [("p1t1", 1)]


# --------------------------------------------------------------------------- C1: crop completeness
#
# A figure crop must contain the panel the map names AND the text that calibrates that panel's
# axes. The geometry every "unchanged" claim below is measured against is the geometry the
# recorded run ingested, INLINED here as literals: `runs/` is gitignored and holds no tracked
# file, so a test that reads it does not run on any machine but the one it was written on — and a
# regression proof that skips proves nothing.
HEUER = ROOT / "tests" / "fixtures" / "pdfs" / "heuer2008.pdf"
CRESSMAN = ROOT / "tests" / "fixtures" / "pdfs" / "cressman2010.pdf"

#: Heuer & Hegele 2008, sha12 3570e4ce2a9c — `runs/rerun-fixed/papers/*/ingest/paper.json`
RUN_HEUER = {
    "fig01": (42.67610168457031, 377.91925048828125, 294.680908203125, 709.9192504882812),
    "fig02": (89.48899841308594, 217.89501953125, 258.3399963378906, 444.63299560546875),
    "fig03": (171.47900390625, 112.64300537109375, 451.67401123046875, 397.5169982910156),
    "fig04": (352.7749938964844, 261.5980224609375, 550.7310180664062, 521.8740234375),
    "fig05": (168.7729949951172, 107.5489501953125, 451.69000244140625, 383.70098876953125),
    "fig06": (350.2239990234375, 247.448974609375, 551.8159790039062, 500.3630065917969),
}
#: Cressman et al. 2010, sha12 5039533c85ef — the paper supplying two of the three pooled cells
RUN_CRESSMAN = {
    "fig01": (300.1419982910156, 341.5080261230469, 550.2520141601562, 720.614013671875),
    "fig02": (45.02399826049805, 521.18798828125, 550.2520141601562, 719.9340209960938),
    "fig03": (45.02399826049805, 459.6190185546875, 550.2520141601562, 719.9340209960938),
    "fig04": (45.02399826049805, 452.8160400390625, 550.2520141601562, 719.9340209960938),
    "fig05": (45.02399826049805, 53.2440185546875, 295.1340026855469, 336.6330261230469),
    "fig06": (45.02399826049805, 351.44903564453125, 295.1340026855469, 610.2330322265625),
}
#: Bock 2005, sha12 b511dbb76fa6 — the third pooled cell, two scanned rasters
RUN_BOCK = {
    "fig01": (297.3070068359375, 45.024009704589844, 553.0864868164062, 176.02239990234375),
    "fig02": (42.18895721435547, 45.02397155761719, 297.9684143066406, 211.79559326171875),
}


def _bbox(fig) -> tuple[float, float, float, float]:
    return (fig.bbox.x0, fig.bbox.y0, fig.bbox.x1, fig.bbox.y1)


@pytest.fixture(scope="module")
def heuer(tmp_path_factory):
    return ingest_pdf(HEUER, tmp_path_factory.mktemp("heuer"), render_pages=False, fig_dpi=100)


@pytest.fixture(scope="module")
def cressman(tmp_path_factory):
    return ingest_pdf(CRESSMAN, tmp_path_factory.mktemp("cressman"), render_pages=False,
                      fig_dpi=100)


def test_heuer_multi_panel_figures_recover_their_own_tick_ladders(heuer):
    """fig02/fig04/fig06 held ZERO numeric words before this rule: the top panel sat 296 pt above
    its caption, was capped out of the caption assignment, became loose and was dropped, so the
    crop the map called 'panel a' did not contain panel a."""
    figs = {f.id: f for f in heuer.figures}
    ladders = {fid: [p.n_ladder for p in figs[fid].panels] for fid in ("fig02", "fig04", "fig06")}
    assert ladders == {"fig02": [6, 5, 7], "fig04": [5, 4, 5], "fig06": [4, 7, 9]}, ladders
    # the crop the run actually sent: not one numeric label anywhere in it
    from canopy.ingest.pdf import _numeric_words, _words_in
    doc = pymupdf.open(HEUER)
    for fid in ("fig02", "fig04", "fig06"):
        page = doc[figs[fid].page - 1]
        words = page.get_text("words")
        old_rect = pymupdf.Rect(*RUN_HEUER[fid])
        assert _numeric_words(words, old_rect) == [], fid
        assert len(_words_in(words, old_rect)) <= 2, fid
    for fid in ("fig02", "fig04", "fig06"):
        fig = figs[fid]
        assert [p.letter for p in fig.panels] == ["a", "b", "c"]
        assert all(p.calibrated for p in fig.panels), [p.n_ladder for p in fig.panels]
        # every panel keeps its own image: a reader asked for panel b never sees three ladders
        assert len({p.crop_png for p in fig.panels}) == len(fig.panels)
        # the recovered region is strictly taller and wider than the one the run recorded
        x0, y0, x1, y1 = RUN_HEUER[fid]
        assert fig.bbox.y0 < y0 and fig.bbox.x0 < x0, (_bbox(fig), RUN_HEUER[fid])
        assert fig.caption_growth_pt > 0.0


def test_the_bottom_panel_crop_carries_the_x_tick_ladder(heuer):
    """F4. The x labels are TEXT, so they lie below the drawing cluster and a 6 pt pad cuts them
    off — measured, they sit 0.1-1.4 pt below the crop's own bottom edge. Two of Heuer's four
    figure cells ask for LATE ADAPTATION, which is a position on that axis, and the per-panel
    assertion cannot notice the loss because it counts inside each panel's y-span and the x
    labels are outside every panel's y-span by construction."""
    from canopy.ingest.pdf import _numeric_words

    doc = pymupdf.open(HEUER)
    figs = {f.id: f for f in heuer.figures}
    want = {"fig02": ["0", "45", "90", "135", "180", "227", "270", "313"],
            "fig04": ["0", "45", "90", "135", "180", "227", "270", "313"],
            "fig06": ["0", "45", "90", "135", "180"]}
    for fid, labels in want.items():
        fig = figs[fid]
        page = doc[fig.page - 1]
        words = page.get_text("words")
        bottom = fig.panels[-1].bbox.rect()
        got = _numeric_words(words, bottom)
        assert all(x in got for x in labels), (fid, got)
        # and the region rect, which must not move, still does not carry them
        assert not set(labels) <= set(_numeric_words(words, fig.bbox.rect())), fid
        # the panel above it is not dragged into the panel below by the growth
        assert fig.panels[0].bbox.y1 < fig.panels[1].bbox.y0, fid


def test_the_calibration_assertion_is_per_panel_never_global():
    """The test that can tell the two rules apart. Numbers sitting in ONE panel's y-span satisfy a
    global `count_numeric >= 6` — they are the shared x-axis labels — and calibrate nothing on the
    other panel's value axis."""
    from canopy.ingest.pdf import _panels_of

    top = pymupdf.Rect(100, 100, 300, 200)
    bottom = pymupdf.Rect(100, 300, 300, 400)
    cap = pymupdf.Rect(90, 460, 300, 500)
    # a six-rung ladder, all of it inside the BOTTOM panel's y-span
    words = [(105, 302 + 15 * i, 118, 310 + 15 * i, str(10 * i), 0, 0, 0) for i in range(6)]
    panels = _panels_of([top, bottom], cap, words)
    assert sum(p["n_numeric"] for p in panels) >= 6          # a global count passes
    assert [p["calibrated"] for p in panels] == [False, True]  # the per-panel form does not


def test_a_side_by_side_figure_stays_two_panels_and_a_grid_stays_four():
    """F2. Growing EVERY panel's left edge to the caption column made the right-hand panel's rect
    swallow the left-hand one whole; `_overlapping` then fired on rects WE had grown and the
    decomposition collapsed into its own union. Measured before the fix: a two-column figure gave
    1 panel carrying 10 numeric words (both ladders) and a 2x2 grid gave 1 — which is the global
    `count_numeric >= 6` the decision withdrew, reached by another road."""
    from canopy.ingest.pdf import _panels_of

    def ladder(x, y0, y1, n=5, step=10):
        return [(x - 14, y0 + (y1 - y0) * i / (n - 1) - 4, x - 2,
                 y0 + (y1 - y0) * i / (n - 1) + 4, str(step * i), 0, 0, 0) for i in range(n)]

    cap = pymupdf.Rect(48, 560, 500, 600)
    two = [pymupdf.Rect(100, 100, 290, 300), pymupdf.Rect(310, 100, 500, 300)]
    words = ladder(100, 110, 290) + ladder(310, 110, 290, step=5)
    panels = _panels_of(two, cap, words)
    assert len(panels) == 2, [(p["rect"], p["n_numeric"]) for p in panels]
    assert [p["n_ladder"] for p in panels] == [5, 5]         # each panel keeps its OWN ladder
    assert panels[0]["rect"].x0 == 48.0                      # leftmost column: the caption column
    assert panels[1]["rect"].x0 == 290.0                     # inner column: the neighbour's edge

    grid = [pymupdf.Rect(100, 100, 290, 290), pymupdf.Rect(310, 100, 500, 290),
            pymupdf.Rect(100, 320, 290, 510), pymupdf.Rect(310, 320, 500, 510)]
    gwords = (ladder(100, 110, 280) + ladder(310, 110, 280) + ladder(100, 330, 500)
              + ladder(310, 330, 500))
    assert len(_panels_of(grid, cap, gwords)) == 4

    # …and the collapse that IS wanted still happens: clusters that genuinely sit on top of each
    # other (a raster under its own vector overlay, an inset) are one panel, not four
    stacked_on_top = [pymupdf.Rect(100, 100, 400, 400), pymupdf.Rect(120, 120, 380, 380)]
    assert len(_panels_of(stacked_on_top, cap, [])) == 1


@pytest.mark.parametrize("text,ok", [
    ("-10", True), ("−10", True), ("–10", True), ("10%", True), ("0.5", True),
    ("−0.5", True), ("0.3°", True), ("1,5", True), ("+60°", True),
    ("±6.9", False), ("block", False), ("1e5", False), ("", False),
])
def test_a_tick_label_may_carry_a_unicode_minus_or_a_unit(text, ok):
    """F5. U+2212 is matplotlib's default minus glyph and negative-going adaptation axes are the
    norm in this literature: on the ASCII-only pattern such a figure scored 0 numeric words, was
    marked uncalibrated and bought ZERO read-outs. Never a wrong number — a silent hard refusal,
    with no re-acquire path behind it."""
    from canopy.ingest.pdf import NUMERIC_WORD_RE

    assert bool(NUMERIC_WORD_RE.match(text)) is ok


@pytest.mark.parametrize("text,value", [
    ("−12,5°", -12.5), ("20%", 20.0), ("-10", -10.0), ("block", None),
])
def test_the_number_a_tick_label_carries(text, value):
    from canopy.ingest.pdf import numeric_value

    assert numeric_value(text) == value


def test_a_ladder_is_a_sorted_column_and_a_schematic_is_not_calibrated():
    """F11. The assertion certified "three bare numbers anywhere in the rect". Measured on
    Cressman 2010's Fig. 1 — an experimental-setup schematic with no value axis at all —
    `n_numeric = 6, calibrated = True` on the words ['1','10','5','30','30','30']. What a value
    axis looks like on the page is a stack of numbers at one x, running in one direction."""
    from canopy.ingest.pdf import _panels_of, _tick_ladder

    rect = pymupdf.Rect(100, 100, 400, 400)
    cap = pymupdf.Rect(90, 460, 400, 500)
    scattered = [(150, 130, 160, 138, "30", 0, 0, 0), (300, 210, 310, 218, "1", 0, 0, 0),
                 (220, 300, 232, 308, "10", 0, 0, 0), (350, 350, 358, 358, "5", 0, 0, 0),
                 (170, 370, 182, 378, "30", 0, 0, 0), (260, 140, 272, 148, "30", 0, 0, 0)]
    assert len(_tick_ladder(scattered, rect)) < 3
    assert _panels_of([rect], cap, scattered)[0]["calibrated"] is False

    column = [(104, 120 + 40 * i, 118, 130 + 40 * i, str(50 - 10 * i), 0, 0, 0) for i in range(6)]
    assert len(_tick_ladder(column, rect)) == 6
    assert _panels_of([rect], cap, column)[0]["calibrated"] is True

    # an x-axis ladder is a ROW, and it calibrates the axis nobody is reading a value off
    row = [(110 + 45 * i, 380, 124 + 45 * i, 390, str(45 * i), 0, 0, 0) for i in range(6)]
    assert len(_tick_ladder(row, rect)) < 3


def test_a_page_number_just_outside_the_region_does_not_flip_the_raster_exemption():
    """F6. The exemption hung on "not one word whose centre is inside the region", and on Bock the
    only thing between that and buying zero read-outs on two pooled cells was **5.6 pt** of white
    space above the page number ('261' on p3, '262' on p4). One word inside would have made
    `text_layer: present`, `n_numeric: 1 < 3`, `panel_uncalibrated` — the exact outcome C1 rule 6
    exists to prevent. Had the rule been measured on the CLIP (union + 6 pt) rather than on the
    union, this paper would be broken today. The exemption is about the absence of a printed
    LADDER among the figure's own words, and a folio is not one of the figure's own words."""
    from canopy.ingest.pdf import _figure_regions, _page_furniture, _tick_ladder, _words_in

    doc = pymupdf.open(BOCK)
    for pno in (3, 4):
        page = doc[pno - 1]
        words = page.get_text("words")
        regions = [r for r in _figure_regions(page, pno) if r["kind"] == "raster"]
        assert regions, pno
        union = regions[0]["bbox"]
        furniture = _page_furniture(page, [pymupdf.Rect(union)])
        grown = pymupdf.Rect(union.x0 - 12, union.y0 - 12, union.x1 + 12, union.y1 + 12)
        # 12 pt of pad is all it takes to swallow the folio, and nothing else on the page
        inside = _words_in(words, grown)
        assert len(inside) == 1 and inside[0].isdigit(), (pno, inside)
        kept = [w for w in words
                if not any(r.x0 <= (w[0] + w[2]) / 2 <= r.x1 and r.y0 <= (w[1] + w[3]) / 2 <= r.y1
                           for r in furniture)]
        assert _words_in(kept, grown) == [], (pno, _words_in(kept, grown))
        assert _tick_ladder(kept, grown) == []
        assert regions[0]["text_layer"] == "none" and regions[0]["n_region_ladder"] == 0
        assert all(p["calibrated"] for p in regions[0]["panels"])


@pytest.mark.parametrize("caption,panels", [
    # punctuated enumerations: read from the text, no bold needed
    ("Figure 2. a: Mean adaptive shifts, b: mean aftereffects, and c: the difference.",
     ["a", "b", "c"]),
    ("Fig. 4 Adaptation curves (a) and aftereffects (b).", ["a", "b"]),
    ("Fig. 8 Panel a, panel b, panel c and panel d.", ["a", "b", "c", "d"]),
    # prose that merely contains the letters is NOT an enumeration
    ("Fig. 6 Reaching in a rotated field; b denotes the baseline.", []),
    ("Fig. 7 e.g. the a and b conditions, with c.", []),
    ("Fig. 1 Experimental setup and design. a Side view of the setup. b and c Top view.", []),
    ("Fig. 2 Mean tracking errors of young and old subjects before and during exposure to a "
     "rotated visual feedback.", []),
    ("Fig. 5 The changes are plotted as a function of changes in reach aftereffects after a "
     "misaligned cursor.", []),
    ("Fig. 4 Adaptation curves (b) and aftereffects (d).", []),
])
def test_a_caption_enumeration_is_punctuated_or_bold_never_an_english_article(caption, panels):
    """F1's corroboration, and the thing that makes it safe to read a BARE letter.

    The shipped pattern demanded a bracket or a colon after the letter and was blind to every
    caption in Cressman 2010 (measured: `[]` on all six). Loosening it to a bare letter needs an
    acceptance rule, and "in order from `a`, at least two" is not enough on its own: it reads
    "e.g. the a and b conditions, with c." as a three-panel figure. From the TEXT, an enumeration
    is punctuated — more of its letters carry a mark than not. The unpunctuated kind is read from
    the publisher's own bold spans instead (next test)."""
    from canopy.ingest.pdf import caption_panels

    assert caption_panels(caption) == panels


def test_the_publisher_marks_its_own_panel_letters_in_bold(cressman):
    """Springer sets the panel letter bold and the English article roman, which is the only sound
    way to tell "a Side view of the setup" from "in a rotated field". Measured on Cressman 2010:
    Fig. 1's letters are `AdvPTimesB` and Fig. 3's the same, while every stray article in those
    captions — and every letter in the four captions that enumerate nothing — is `AdvPTimes`."""
    from canopy.ingest.pdf import caption_panels

    got = {f.id: f.caption_panels for f in cressman.figures}
    assert got["fig01"] == ["a", "b", "c"]
    assert got["fig03"] == ["a", "b"]                       # the two-panel figure behind F1
    assert got["fig05"] == [] and got["fig06"] == []         # captions that enumerate nothing
    # and the text alone does NOT claim them: it is the bold spans that carry this
    for fid in ("fig01", "fig03"):
        fig = next(f for f in cressman.figures if f.id == fid)
        assert caption_panels(fig.caption) == [], fid


def test_no_caption_of_the_three_fixture_pdfs_is_a_false_positive(heuer, cressman, tmp_path):
    """The measurement the rules above are licensed by, run over the corpus rather than quoted."""
    bock = ingest_pdf(BOCK, tmp_path / "bock", render_pages=False, fig_dpi=100)
    got = {f.id: f.caption_panels for rec in (heuer, cressman, bock) for f in rec.figures}
    enumerating = {fid: v for fid, v in got.items() if v}
    # every enumeration found is a run from "a" of at least two letters …
    assert all(v == [chr(ord("a") + i) for i in range(len(v))] for v in enumerating.values())
    assert all(len(v) >= 2 for v in enumerating.values())
    # … and no figure whose caption enumerates nothing claims otherwise
    assert got["fig02"] == [] and got["fig05"] == [] and got["fig06"] == []   # Bock/Cressman ids


def test_a_raster_figure_skips_the_text_rules_and_keeps_its_crop(tmp_path):
    """Bock's two figures are scanned rasters with zero extractable words, and they supplied two of
    the three autonomously pooled cells. Wiring the assertion to them would buy zero read-outs."""
    rec = ingest_pdf(BOCK, tmp_path / "bock", render_pages=False, fig_dpi=100)
    for fig in rec.figures:
        assert fig.text_layer == "none"
        assert fig.caption_growth_pt == 0.0
        assert len(fig.panels) == 1 and fig.panels[0].calibrated       # assertion SKIPPED, not failed
        assert fig.panels[0].n_numeric == 0 and fig.panels[0].n_ladder == 0
        assert fig.panels[0].crop_png == fig.crop_png                  # one panel: the same image
        assert _bbox(fig) == pytest.approx(RUN_BOCK[fig.id], abs=1e-6)


def test_cressman_geometry_is_untouched_by_the_growth_rule(cressman):
    """The regression proof: the paper that supplies two of the three pooled cells cannot move.
    Its caption `x0` already equals its graphics' `x0` on every figure, so the asymmetric rule is a
    no-op there — and every figure stays a single panel, so the digitiser is handed the same
    image it was handed before."""
    assert {f.id for f in cressman.figures} == set(RUN_CRESSMAN)
    for fig in cressman.figures:
        assert fig.caption_growth_pt == 0.0, (fig.id, fig.caption_growth_pt)
        assert _bbox(fig) == pytest.approx(RUN_CRESSMAN[fig.id], abs=1e-6), fig.id
        assert len(fig.panels) == 1, (fig.id, [p.letter for p in fig.panels])
        assert fig.panels[0].crop_png == fig.crop_png
    # the two figures the pooled cells are read from carry a printed ladder of their own
    by_id = {f.id: f for f in cressman.figures}
    assert by_id["fig03"].panels[0].n_ladder >= 3
    assert by_id["fig05"].panels[0].n_ladder >= 3


def _head_ingest(tmp_path: Path):
    """`canopy.ingest.pdf` as it stands at HEAD, importable beside the working tree's own."""
    import subprocess
    import sys

    pkg = tmp_path / "headingest"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in ("pdf.py", "images.py"):
        out = subprocess.run(["git", "show", f"HEAD:canopy/ingest/{name}"], cwd=ROOT,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        (pkg / name).write_text(out.stdout)
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib
        return importlib.import_module("headingest.pdf")
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.parametrize("pdf,recorded", [("cressman2010.pdf", RUN_CRESSMAN),
                                          ("bock2005.pdf", RUN_BOCK)])
def test_the_region_crops_of_the_pooled_papers_are_byte_identical_to_head(pdf, recorded, tmp_path):
    """The binding regression proof, run rather than asserted: the same PDF through HEAD's
    ingester and through this one must produce the SAME PNG BYTES for every figure region. These
    two papers supply all three of the run's autonomously pooled cells; if their crops move, every
    number read off them is a different number. Panel crops are new artefacts and are free to
    change — this test is about the region."""
    head = _head_ingest(tmp_path / "head_pkg")
    before = head.ingest_pdf(PDF_DIR / pdf, tmp_path / "before", render_pages=False, fig_dpi=100)
    after = ingest_pdf(PDF_DIR / pdf, tmp_path / "after", render_pages=False, fig_dpi=100)
    assert [f.id for f in after.figures] == [f.id for f in before.figures]
    for old, new in zip(before.figures, after.figures):
        assert _bbox(new) == _bbox(old), old.id
        assert _bbox(new) == pytest.approx(recorded[new.id], abs=1e-6), old.id
        for rel in (old.crop_png, old.claude_png):
            assert (tmp_path / "after" / rel).read_bytes() == \
                   (tmp_path / "before" / rel).read_bytes(), f"{old.id} {rel}"


def test_two_figures_on_one_page_do_not_merge_without_the_distance_cap(cressman):
    """M2, the prerequisite the cap used to hide: Cressman p9 carries Fig. 5 and Fig. 6, whose
    clusters sit 227 pt and 25 pt from the OTHER figure's caption."""
    p9 = [f for f in cressman.figures if f.page == 9]
    assert [f.label for f in p9] == ["Fig. 5", "Fig. 6"]
    assert p9[0].bbox.y1 < p9[1].bbox.y0, [_bbox(f) for f in p9]


def test_a_caption_may_not_reach_across_another_figure(tmp_path):
    """The merge hazard the distance cap used to hide. Once a caption may claim a graphic at any
    distance, the only thing left between "two panels of one figure" and "two different figures"
    is reading order: another caption's graphic sitting BETWEEN two of mine says mine are two
    figures. Without this the merged region carries two unrelated y ladders."""
    from canopy.ingest.pdf import _enforce_contiguity

    far = ("cap_far", "Fig. 1 The dependent measure.", "Fig. 1", 0.9)
    near = ("cap_near", "Fig. 2 The other measure.", "Fig. 2", 0.9)
    top = dict(rect=pymupdf.Rect(80, 60, 300, 200))
    middle = dict(rect=pymupdf.Rect(80, 260, 300, 400))
    bottom = dict(rect=pymupdf.Rect(80, 460, 300, 600))
    candidates = [top, middle, bottom]
    choice = {id(top): (far, "below"), id(middle): (near, "below"), id(bottom): (far, "below")}
    ranked = {id(top): [(far, "below", (0, 10))],
              id(middle): [(near, "below", (0, 10))],
              id(bottom): [(far, "below", (0, 300)), (near, "below", (0, 400))]}
    _enforce_contiguity(candidates, choice, ranked)
    assert choice[id(top)][0] is far                 # the contiguous run keeps the caption
    assert choice[id(middle)][0] is near
    assert choice[id(bottom)][0] is near, "the straggler must fall back, not merge two figures"


def test_a_straggler_with_no_other_caption_becomes_loose():
    from canopy.ingest.pdf import _enforce_contiguity

    far = ("cap_far", "Fig. 1 The dependent measure.", "Fig. 1", 0.9)
    near = ("cap_near", "Fig. 2 The other measure.", "Fig. 2", 0.9)
    top = dict(rect=pymupdf.Rect(80, 60, 300, 200))
    middle = dict(rect=pymupdf.Rect(80, 260, 300, 400))
    bottom = dict(rect=pymupdf.Rect(80, 460, 300, 600))
    choice = {id(top): (far, "below"), id(middle): (near, "below"), id(bottom): (far, "below")}
    ranked = {id(top): [(far, "below", (0, 10))], id(middle): [(near, "below", (0, 10))],
              id(bottom): [(far, "below", (0, 300))]}
    _enforce_contiguity([top, middle, bottom], choice, ranked)
    assert choice[id(bottom)] == (None, None)


# --------------------------------------------------------------------------- fix round 2
def _panel_pdf(path: Path, clusters, labels, caption: str, cap_xy=(90, 520)) -> None:
    """A page of line-art clusters, free-standing text labels, and one full-width caption.

    Zero-area strokes, one PDF path each, as matplotlib and R emit them — the shape the ingester's
    cluster detector actually sees.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for (x0, y0, x1, y1) in clusters:
        segs = [((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))]
        for i in range(12):
            segs.append(((x0 + i * (x1 - x0) / 12.0, y1), (x0 + i * (x1 - x0) / 12.0, y1 + 4)))
            segs.append(((x0 - 4, y0 + i * (y1 - y0) / 12.0), (x0, y0 + i * (y1 - y0) / 12.0)))
        for a, b in segs:
            shape = page.new_shape()
            shape.draw_line(a, b)
            shape.finish(color=(0, 0, 0), width=0.8)
            shape.commit()
    for (x, y, text) in labels:
        page.insert_text((x, y), text, fontsize=7)
    page.insert_text(cap_xy, caption, fontsize=8)
    doc.save(path)


TWO_PANEL_CAPTION = ("Fig. 1 The dependent measure plotted against block for both groups, with "
                     "the standard deviation shown. a: upper panel, b: lower panel.")


def test_one_point_of_overlap_does_not_hand_a_panel_its_neighbours_ladder():
    """N1. The neighbour landmark demanded STRICT non-overlap, so one point of x-overlap between
    two side-by-side clusters — an error-bar cap, a tick, a shared frame line — made the left
    neighbour invisible; the right panel then grew to the caption column and swallowed it whole,
    and the per-panel assertion certified the right panel **on the left panel's ladder**.
    `_overlapping` does not fire on 0.6 % of the smaller area, so the collapse that used to at
    least make the union obvious no longer happened either. F2's hazard, one point away."""
    from canopy.ingest.pdf import _panels_of

    def ladder(x, base, step):
        return [(x - 14, 106 + 40 * i, x - 2, 114 + 40 * i, str(base + step * i), 0, 0, 0)
                for i in range(5)]

    cap = pymupdf.Rect(48, 560, 520, 600)
    words = ladder(100, 0, 10) + ladder(296, 0, -5)      # b's ladder sits in the gutter
    for gutter, tag in ((310, "clean gutter"), (279, "1 pt of overlap")):
        panels = _panels_of([pymupdf.Rect(100, 100, 280, 300),
                             pymupdf.Rect(gutter, 100, 480, 300)], cap, words)
        assert len(panels) == 2, tag
        got = [_ladder_of(p, words) for p in panels]
        assert [len(x) for x in got] == [5, 5], (tag, got)
        assert got[0][0] == "0" and got[0][1] == "10", (tag, got)     # a's own, ascending
        assert got[1][1] == "-5", (tag, got)                          # b's own, descending
        assert panels[1]["rect"].x0 < gutter, tag        # grown left, but never over its own x0
        assert panels[1]["rect"].x0 >= 278, (tag, panels[1]["rect"])  # …and not to the caption


def _ladder_of(panel, words):
    from canopy.ingest.pdf import _tick_ladder

    return _tick_ladder(words, panel["rect"])


def test_a_vector_figure_labelled_only_in_its_margins_still_gets_its_ladder(tmp_path):
    """N2. The caption-column growth was gated on text found in the UNGROWN union — the very rect
    C1 proved does not contain the ladder. A clean vector plot whose only text is its axis labels
    therefore got no growth at all, was recorded as `text_layer: none` (i.e. "scanned raster",
    about a vector plot with a full text layer), was swept into the raster exemption, and the crop
    handed to the readers held no y labels — so an honest reader answers `calibration_source:
    inferred` and C2 drops every reading. Heuer survives it only because its legend box happens to
    sit inside the plot area."""
    pdf = tmp_path / "margins.pdf"
    labels = ([(120, 110 + 30 * i, str(-10 * i)) for i in range(5)]
              + [(120, 310 + 30 * i, str(20 - 5 * i)) for i in range(5)])
    _panel_pdf(pdf, [(150, 100, 400, 250), (150, 300, 400, 450)], labels, TWO_PANEL_CAPTION)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=72)
    fig = rec.figures[0]
    assert fig.kind == "vector"
    assert fig.text_layer == "present", "a vector plot with a text layer is not a scanned raster"
    assert fig.caption_growth_pt > 0.0
    assert [p.n_ladder for p in fig.panels] == [5, 5]
    assert all(p.calibrated for p in fig.panels)


def test_a_ladder_printed_to_the_right_of_its_panel_is_found(tmp_path):
    """N3. `_panel_left` and `_panel_bottom` grow to landmarks; the right edge kept the 6 pt
    constant. A ladder printed to the RIGHT of its plot area — a secondary axis, or the
    right-labelled member of a pair, which is what Cressman's own Fig. 3b is — sits 8 pt outside
    it, and the panel is then refused with zero read-outs although it is perfectly readable.
    Never a wrong number; a lost cell, of the same class as the ASCII-only minus."""
    pdf = tmp_path / "right.pdf"
    labels = ([(120, 110 + 30 * i, str(-10 * i)) for i in range(5)]
              + [(408, 310 + 30 * i, str(20 - 5 * i)) for i in range(5)])
    _panel_pdf(pdf, [(150, 100, 400, 250), (150, 300, 400, 450)], labels, TWO_PANEL_CAPTION)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=72)
    panels = rec.figures[0].panels
    assert [p.letter for p in panels] == ["a", "b"]
    assert [p.n_ladder for p in panels] == [5, 5], [p.n_ladder for p in panels]
    assert all(p.calibrated for p in panels)
    # the retry is spent only where the left growth found nothing: panel a is not widened
    assert panels[1].bbox.x1 > panels[0].bbox.x1


def test_two_stacked_ladders_are_two_ladders():
    """N4. The monotone test alone continues a run across any gap in y, so two panels stacked with
    descending axes — the common case — certified as one long ladder on exactly the union
    `n_ladder` exists to discriminate. Measured before: `50,40,30` above `20,10,0` came back as a
    six-rung ladder, and `50,40,30` above `0,10,20` still stole one rung."""
    from canopy.ingest.pdf import _tick_ladder

    rect = pymupdf.Rect(50, 50, 400, 700)

    def rungs(y0, values, step=20):
        return [(104, y0 + step * i, 118, y0 + step * i + 8, str(v), 0, 0, 0)
                for i, v in enumerate(values)]

    both_down = rungs(100, [50, 40, 30]) + rungs(400, [20, 10, 0])
    assert _tick_ladder(both_down, rect) == ["50", "40", "30"]
    mixed = rungs(100, [50, 40, 30]) + rungs(400, [0, 10, 20])
    assert _tick_ladder(mixed, rect) == ["50", "40", "30"]
    # …and a ladder that merely SUPPRESSES a label is still one ladder: Cressman 2010's Fig. 5
    # prints 120..20 and -20,-40 and leaves the zero off — one gap of exactly 2.0x
    suppressed = rungs(100, [120, 100, 80, 60, 40, 20])
    suppressed += rungs(100 + 20 * 7, [-20, -40])
    assert _tick_ladder(suppressed, rect) == ["120", "100", "80", "60", "40", "20", "-20", "-40"]


def test_a_panel_with_no_ladder_beside_one_that_has_a_ladder_is_refused(tmp_path):
    """N5, restated under P2. The premise the per-panel assertion needs is "some panel here owns
    an axis": where a figure prints one, a panel that does not own a share of it is refused rather
    than read off its neighbour's — which is the hazard the whole rule exists for. (Before N5 the
    exemption and the assertion shared a threshold, so a figure on which NO panel could be
    calibrated was exempted wholesale while the same figure with one well-labelled sibling refused
    the others — precisely backwards.)"""
    pdf = tmp_path / "one_ladder.pdf"
    labels = [(120, 110 + 30 * i, str(-10 * i)) for i in range(5)]      # panel a only
    _panel_pdf(pdf, [(150, 100, 400, 250), (150, 300, 400, 450)], labels, TWO_PANEL_CAPTION)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=72)
    fig = rec.figures[0]
    assert fig.n_region_ladder == 5                       # the figure does print an axis
    assert [p.n_ladder for p in fig.panels] == [5, 0]
    assert [p.calibrated for p in fig.panels] == [True, False]


def test_a_stray_numeral_does_not_refuse_a_multi_panel_raster_figure(tmp_path):
    """P2. Keying the exemption on "any numeric word" re-created exactly the fragility F6 was
    raised for. A scanned or mixed figure with two panels and a SINGLE extractable number — a
    scale bar's "10", an inset label, whatever the OCR layer happens to carry — measured
    `n_region_numeric=1, n_panels=2`, the exemption did not fire, and a named panel of it bought
    ZERO read-outs. F6 was raised because a page number 5.6 pt away came within that of costing
    Bock's two pooled cells; one numeral must not do what the folio could not."""
    pdf = tmp_path / "stray.pdf"
    _panel_pdf(pdf, [(150, 100, 400, 250), (150, 300, 400, 450)], [(120, 140, "10")],
               TWO_PANEL_CAPTION)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=72)
    fig = rec.figures[0]
    assert len(fig.panels) == 2
    assert fig.n_region_numeric == 1 and fig.n_region_ladder < 3
    assert all(p.calibrated for p in fig.panels), "one numeral is not an axis to be refused for"


def test_a_wordless_raster_region_is_still_exempt(tmp_path):
    """The other half of N5, and the one that must not move: Bock's two figures carry no numeric
    word at any pad and supply two of the three pooled cells."""
    rec = ingest_pdf(BOCK, tmp_path / "bock", render_pages=False, fig_dpi=100)
    for fig in rec.figures:
        assert fig.n_region_numeric == 0 and fig.n_region_ladder == 0
        assert fig.panels[0].calibrated is True


def test_a_guessed_region_is_not_certified_without_a_ladder():
    """N7. `_whole_region_panel` certified unconditionally — including a `caption_only` rect
    proposed above a caption that received no graphics (confidence 0.3) and a `loose` graphic with
    no caption at all (0.4). A rect the ingester only guessed at must not be recorded as
    calibrated merely because nobody split it."""
    from canopy.ingest.pdf import _whole_region_panel

    rect = pymupdf.Rect(50, 50, 400, 400)
    scattered = [(150, 130, 160, 138, "30", 0, 0, 0), (300, 210, 310, 218, "1", 0, 0, 0),
                 (220, 300, 232, 308, "10", 0, 0, 0)]
    assert _whole_region_panel(rect, scattered)[0]["calibrated"] is False
    column = [(104, 120 + 40 * i, 118, 130 + 40 * i, str(50 - 10 * i), 0, 0, 0) for i in range(4)]
    assert _whole_region_panel(rect, column)[0]["calibrated"] is True
    assert _whole_region_panel(rect, [])[0]["calibrated"] is True     # nothing to build one from


# --------------------------------------------------------------------------- fix round 3
@pytest.mark.parametrize("right_column,label", [
    ([(430, 310 + 30 * i, str(500 - 100 * i)) for i in range(5)], "another figure's ladder"),
    ([(430, 310 + 30 * i, str(1 + i)) for i in range(5)], "a numbered list in the body text"),
])
def test_the_right_retry_stops_at_whatever_belongs_to_something_else(right_column, label,
                                                                     tmp_path):
    """P1. The empty-ladder retry was bounded by the CAPTION, and a caption is routinely wider
    than the figure it captions — Heuer's own fig03 and fig05 overhang by 26 pt, and a full-width
    caption over a column figure is the ordinary two-column case. So the retry crossed into the
    next column and certified the panel on a FOREIGN ladder: a neighbouring figure's
    `500,400,300,200,100`, or a numbered list's `1,2,3,4,5`, accepted as "panel b's own axis" on a
    rect spanning the whole page and handed to the reader as such.

    That is worse than the lost cell N3 fixed. N3's failure was a refusal; this one is a
    wrong-ladder certification — the hazard the per-panel assertion exists for — and it defeats
    `panel_uncalibrated` at the same time."""
    pdf = tmp_path / "columns.pdf"
    own = [(120, 110 + 30 * i, str(-10 * i)) for i in range(5)]        # panel a's own ladder
    _panel_pdf(pdf, [(150, 100, 400, 250), (150, 300, 400, 450)], own + right_column,
               TWO_PANEL_CAPTION)
    rec = ingest_pdf(pdf, tmp_path / "out", render_pages=False, fig_dpi=72)
    panels = rec.figures[0].panels
    assert panels[0].n_ladder == 5 and panels[0].calibrated       # panel a is unaffected
    assert panels[1].n_ladder == 0, (label, panels[1].n_ladder)
    assert panels[1].calibrated is False, label
    assert panels[1].bbox.x1 < 430, (label, panels[1].bbox.x1)     # never reached the column


def test_bold_that_is_not_selective_enumerates_nothing(tmp_path):
    """P3. The bold path returned before the punctuation test was ever reached, so a publisher
    that sets whole captions bold — a real house style, and `BOLD_FONT_RE` matches broadly — got
    back exactly the false positive the text rule now rejects (this caption is the reviewer's own:
    "Reaching in a rotated field; b denotes the baseline"). Bold is evidence only where it is
    SELECTIVE, so the same kind of majority test the text path applies to punctuation is applied
    to weight: bold that is half the caption or more marks nothing.

    The three cases below are the three that matter, and the middle one is the one that proves the
    rule: the letters really are there as bold single-letter spans, and they are refused on weight
    alone. (In the all-bold case the writer merges the run into one span, so there are no
    single-letter spans left to find — the rule and the PDF's own layout agree.)"""
    from canopy.ingest.pdf import _bold_letters, _caption_blocks

    words = ("Fig. 1 Reaching in a rotated field for both groups over many blocks ; "
             "b denotes the baseline and the measure is the same throughout .").split()

    def build(path, mode: str):
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        for (x0, y0, x1, y1) in ((150, 100, 400, 250), (150, 300, 400, 450)):
            for i in range(12):
                for a, b in ((((x0 + i * 20, y1), (x0 + i * 20, y1 + 4))),
                             (((x0 - 4, y0 + i * 12), (x0, y0 + i * 12)))):
                    shape = page.new_shape()
                    shape.draw_line(a, b)
                    shape.finish(color=(0, 0, 0), width=0.8)
                    shape.commit()
        for i in range(5):
            page.insert_text((120, 110 + 30 * i), str(-10 * i), fontsize=7)
        x = 90.0
        for i, word in enumerate(words):
            bold = (word in ("a", "b") if mode == "letters"
                    else (i % 2 == 0 or word in ("a", "b")) if mode == "half" else True)
            font = "hebo" if bold else "helv"
            page.insert_text((x, 520), word, fontsize=8, fontname=font)
            x += pymupdf.get_text_length(word + " ", fontname=font, fontsize=8)
        doc.save(path)

    for mode, want in (("letters", ["a", "b"]), ("half", []), ("all", [])):
        pdf = tmp_path / f"bold_{mode}.pdf"
        build(pdf, mode)
        page = pymupdf.open(pdf)[0]
        caption = [c for c in _caption_blocks(page) if c[3] >= 0.6][0]
        if mode == "half":     # the letters ARE on the page as bold single-letter spans
            block = min((b for b in page.get_text("dict")["blocks"] if b.get("type") == 0),
                        key=lambda b: abs(b["bbox"][0] - caption[0].x0)
                        + abs(b["bbox"][1] - caption[0].y0))
            singles = [sp["text"].strip() for line in block["lines"] for sp in line["spans"]
                       if len(sp["text"].strip()) == 1]
            assert "a" in singles and "b" in singles, singles
        assert _bold_letters(page, caption[0]) == want, mode
        rec = ingest_pdf(pdf, tmp_path / f"out_{mode}", render_pages=False, fig_dpi=72)
        assert rec.figures[0].caption_panels == want, mode


def test_a_retry_ladder_further_away_than_a_tick_label_ever_sits_is_not_its_ladder(tmp_path):
    """P1's second bound, the one that holds where the page puts nothing in between (round 3b).

    A panel under a wider one has no sibling and no obstacle to its right — the page is empty
    there — so the retry runs to the region's own edge and finds the wide panel's right-hand
    labels. "Further away than the panel is wide" was too generous a cap on a wide panel: at
    350 pt it permits a ladder 350 pt away, most of a page. Tick labels sit ADJACENT to the axis
    they label (6-12 pt from the drawing on this corpus), so the cap is
    `min(panel width, RIGHT_LADDER_REACH_PT)` and a ladder further off is not this panel's,
    however wide the panel happens to be."""
    def build(path, panel_b_x1, ladder_x):
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        for (x0, y0, x1, y1) in ((150, 100, 500, 250), (150, 300, panel_b_x1, 450)):
            segs = [((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))]
            for i in range(12):
                segs.append(((x0 + i * (x1 - x0) / 12.0, y1),
                             (x0 + i * (x1 - x0) / 12.0, y1 + 4)))
                segs.append(((x0 - 4, y0 + i * (y1 - y0) / 12.0),
                             (x0, y0 + i * (y1 - y0) / 12.0)))
            for a, b in segs:
                shape = page.new_shape()
                shape.draw_line(a, b)
                shape.finish(color=(0, 0, 0), width=0.8)
                shape.commit()
        for i in range(5):
            page.insert_text((120, 110 + 30 * i), str(-10 * i), fontsize=7)
        for i in range(5):
            page.insert_text((ladder_x, 310 + 30 * i), str(20 - 5 * i), fontsize=7)
        page.insert_text((90, 520), TWO_PANEL_CAPTION, fontsize=8)
        doc.save(path)

    cases = [
        # (panel b's right edge, the ladder's x, may it certify?)
        (210, 430, False),        # narrow panel, ladder 214 pt off — beyond both caps
        (210, 216, True),         # narrow panel, ladder 6 pt off — adjacent, and its own
        (300, 430, False),        # WIDE panel, ladder 128 pt off: inside the panel's own width,
                                  # and still further than any tick label sits (3b's case)
        (300, 316, True),         # WIDE panel, ladder 10 pt off — adjacent
    ]
    for panel_b_x1, ladder_x, certifies in cases:
        pdf = tmp_path / f"reach_{panel_b_x1}_{ladder_x}.pdf"
        build(pdf, panel_b_x1, ladder_x)
        rec = ingest_pdf(pdf, tmp_path / f"out_{panel_b_x1}_{ladder_x}", render_pages=False,
                         fig_dpi=72)
        panels = rec.figures[0].panels
        where = (panel_b_x1, ladder_x)
        assert panels[0].calibrated is True, where          # the wide panel is unaffected
        assert panels[1].calibrated is certifies, where
        assert (panels[1].n_ladder == 5) is certifies, where


def test_a_raster_figure_whose_native_density_wins_the_dpi_choice_still_renders():
    """`get_pixmap(dpi=…)` needs an int; the raster branch handed it `72 * nat_w / width * upscale`
    raw and one nine-paper-run PDF died in ingest with `TypeError: in method
    'fz_pixmap_xres_set'`. The choice is rounded, on both branches."""
    import inspect
    from canopy.ingest import pdf as ingest_pdf_module
    src = inspect.getsource(ingest_pdf_module.ingest_pdf)
    assert "dpi = int(round(max(72, min(fig_dpi" in src
    assert "dpi = int(fig_dpi)" in src
    # and the arithmetic that used to leak a float is rounded to an int the way PyMuPDF wants
    fig_dpi, nat_w, width = 300, 1000, 500.0
    raw = max(72, min(fig_dpi, 72 * nat_w / max(width, 1) * ingest_pdf_module.RASTER_UPSCALE))
    assert isinstance(int(round(raw)), int)


def _page_with(draw_box=None, caption_rect=None, caption_text="", far_text=None):
    """A one-page PDF: an optional cluster of vector strokes and an optional caption block."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    if draw_box is not None:
        x0, y0, x1, y1 = draw_box
        for i in range(24):                       # >= 15 overlapping drawings = one cluster
            frac = i / 23.0
            page.draw_line((x0, y0 + frac * (y1 - y0)), (x1, y0 + frac * (y1 - y0)))
        page.draw_rect((x0, y0, x1, y1))
    if caption_rect is not None:
        page.insert_textbox(pymupdf.Rect(*caption_rect), caption_text, fontsize=9)
    if far_text is not None:
        page.insert_textbox(pymupdf.Rect(*far_text[0]), far_text[1], fontsize=9)
    return doc, page


def test_a_side_caption_taller_than_its_panel_still_attaches():
    """The side band is measured against the SMALLER of caption and graphic: the two share rows,
    so the smaller box must spend half its height beside the other. Measured against the caption
    alone, a margin caption taller than the panel it names could never attach — the panels became
    loose, the area gate dropped them (a panel is a few percent of the page), and the paid readers
    were handed the empty column above the caption instead of the figure."""
    from canopy.ingest.pdf import _figure_regions

    # panel 100x60 in the left column; caption in the adjacent column, 12pt gap, 150pt tall —
    # vertical overlap is the panel's full 60pt: under 0.5*caption (75) yet over 0.5*panel (30)
    caption = ("Fig. 3. Learning time course in the horizontal and sagittal conditions during "
               "the rotation phase, with average reach direction for the right and the left limb "
               "shown separately across the baseline and rotation blocks of the experiment, "
               "together with the average amount of explicit learning measured from the "
               "verbally reported aiming direction and the average implicit learning computed "
               "from the difference between the actual reach direction and the reported aiming "
               "location for every participant of both groups.")
    doc, page = _page_with(draw_box=(70, 200, 170, 260),
                           caption_rect=(182, 180, 320, 420), caption_text=caption)
    regions = _figure_regions(page, 2)
    attached = [r for r in regions if r["kind"] != "caption_only" and r["caption"]]
    assert attached, [r["kind"] for r in regions]
    assert "Learning time course" in attached[0]["caption"]
    doc.close()


def test_a_caption_only_region_counts_the_primitives_under_it():
    """`caption_only` used to hardcode `n_images=0, n_drawings=0` — a claim, not a measurement.
    Those two numbers are the record's only statement of whether anything readable is under the
    rect, and the digitizer now refuses to pay for a region that measures zero of both; so a
    hardcoded zero over real ink would turn a readable region into a refused one, and a real zero
    must stay zero so the refusal fires."""
    from canopy.ingest.pdf import _figure_regions

    caption = ("Fig. 7. Overview of the experimental apparatus and the sequence of trial phases "
               "used in every condition of the study, including all timing parameters.")
    # a caption with its figure genuinely elsewhere: body text above, no graphics near
    doc, page = _page_with(caption_rect=(70, 500, 320, 560), caption_text=caption,
                           far_text=((70, 300, 320, 480),
                                     "Participants completed the task described previously. " * 8))
    regions = _figure_regions(page, 3)
    only = [r for r in regions if r["kind"] == "caption_only"]
    assert only and only[0]["n_images"] == 0 and only[0]["n_drawings"] == 0
    doc.close()


# --------------------------------------------------- fix F: panel letters verify themselves
def test_f_the_fixture_corpus_is_silent_and_keeps_its_letters(heuer, cressman):
    """The measured NO-GO of the first design draft, pinned: with one-sided discriminative
    tokens, Heuer fig02/04/06 all falsely DISPUTED (shared axis vocabulary voted every panel at
    the last letter's description, which had swallowed the caption's trailing prose). Verified
    correct figures must come back silent — no dispute, no letter moved."""
    from canopy.ingest.pdf import PaperRecord
    for rec in (heuer, cressman):
        for f in rec.figures:
            assert not f.panel_labels_disputed, (f.id, f.panel_label_note)
            assert not f.panel_label_note, (f.id, f.panel_label_note)
            assert [p.letter for p in f.panels] == \
                [chr(ord("a") + i) for i in range(len(f.panels))], f.id
    # …and the new fields plus `primitives_measured` survive the disk round-trip: region dicts
    # are rebuilt field-by-field into FigureRegion, where an unthreaded key silently vanishes
    # (`primitives_measured` was set on caption_only dicts and dropped at exactly that seam)
    loaded = PaperRecord.load(heuer.out_dir)
    for f in loaded.figures:
        assert f.panel_labels_disputed is False
        if f.kind == "caption_only":
            assert f.primitives_measured is True, \
                "the $0 no-graphics refusal needs the measurement to survive ingest"


def test_f_caption_segmentation_is_all_letters_or_nothing():
    from canopy.ingest.pdf import _caption_panel_texts
    cap = ("Figure 3. (A) Setup of the task. (B) Learning curves for both groups. "
           "(D) Adaptation in the first block. (E) Adaptation in the last block. "
           "Error bars are 95% CIs.")
    spans = _caption_panel_texts(cap, ["a", "b", "d", "e"])
    assert set(spans) == {"a", "b", "d", "e"}
    assert "first block" in spans["d"] and "last block" in spans["e"]
    # the trailing prose belongs to the FIGURE, not the last panel: descriptions stop at their
    # first sentence boundary, so "Error bars…" never fills a description with shared vocabulary
    assert "Error bars" not in spans["e"]
    # a run the caption cannot place in order refuses wholesale rather than guessing
    assert _caption_panel_texts(cap, ["a", "b", "c"]) == {}
    assert _caption_panel_texts("Fig. 2. Reaching in a rotated field.", ["a", "b"]) == {}


def test_f_a_clean_swap_rebinds_and_anything_less_disputes_or_stays_silent():
    import pymupdf
    from canopy.ingest.pdf import _verify_panel_letters

    def word(x0, text):
        return (x0, 0.0, x0 + 40.0, 10.0, text, 0, 0, 0)

    panels = [dict(rect=pymupdf.Rect(0, 0, 90, 50)), dict(rect=pymupdf.Rect(95, 0, 190, 50))]
    cap = "Fig. 3. (D) Error in the first block. (E) Error in the last block."
    crossed = [word(0, "Last"), word(42, "block"), word(100, "First"), word(142, "block")]
    rebound, note = _verify_panel_letters(panels, cap, ["d", "e"], crossed)
    assert rebound == ["e", "d"] and "first" in note.lower()
    # one panel votes a sibling, the sibling is silent: never a re-bind — a dispute
    rebound, note = _verify_panel_letters(panels, cap, ["d", "e"], crossed[:2])
    assert rebound is None and note
    # correct labels: silence
    straight = [word(0, "First"), word(42, "block"), word(100, "Last"), word(142, "block")]
    assert _verify_panel_letters(panels, cap, ["d", "e"], straight) == (None, "")
    # trailing-letter captions ("Movement time (a) and endpoint error (b)") shift every
    # description one panel back — a segmentation artifact, silenced, never disputed
    trailing_cap = "Fig. 2. Movement time (a) and endpoint error (b) across blocks."
    trailing = [word(0, "Movement"), word(42, "time"), word(100, "endpoint"), word(142, "error")]
    assert _verify_panel_letters(panels, trailing_cap, ["a", "b"], trailing) == (None, "")
    # captions whose descriptions share all their words with both panels: nothing discriminative
    generic_cap = "Fig. 4. (A) Error across blocks. (B) Error across blocks."
    generic = [word(0, "Error"), word(100, "Error")]
    assert _verify_panel_letters(panels, generic_cap, ["a", "b"], generic) == (None, "")


# --------------------------------------------- ticket 2: what the caption itself states
def test_caption_dispersion_credits_only_anchored_explicit_statements():
    from canopy.ingest.pdf import caption_dispersion

    said = caption_dispersion
    assert said("Data are presented across all subjects (mean ± SE).") == \
        {"type": "SE", "quote": "Data are presented across all subjects (mean ± SE)."}
    assert said("Error bars represent the standard error of the mean.")["type"] == "SE"
    assert said("Values are means (±SEM) for the last block.")["type"] == "SE"
    assert said("Shaded regions indicate 95% confidence intervals.")["type"] == "CI95"
    assert said("Error bars, s.d.")["type"] == "SD"
    assert said("Whiskers denote the standard deviation.")["type"] == "SD"
    # the refusals, each a way a wrong type would have been invented:
    assert said("Error bars show mean ± 2 SE.") == {}, "a multiplied bar is not its bare type"
    assert said("The standard deviation of movement time was analysed.") == {}, \
        "the OUTCOME's SD, not the bars' — the anchor rule"
    assert said("The standard error of the estimate is shown.") == {}, "regression SEE"
    assert said("The standard deviation of the mean is displayed on error bars.") == {}, \
        "historically = SEM; a caption printing it is ambiguous"
    assert said("Error bars show SD in A and SEM in B.") == {}, "two types, unscoped"
    assert said("se was small in every condition.") == {}, "bare lowercase is a word"
    assert said("Fehlerbalken zeigen die Standardabweichung.") == {}, \
        "non-English matches nothing — the conservative default IS the non-English behaviour"
    assert said("") == {}


def test_caption_panel_dispersions_scope_by_letter_and_never_generalize():
    from canopy.ingest.pdf import _caption_semantics, caption_panel_dispersions

    cap = "(a) Baseline reach error. (b) Aftereffects; error bars show SD. Mean ± SEM elsewhere."
    per = caption_panel_dispersions(cap, ["a", "b"])
    assert per.get("b", {}).get("type") == "SD" and "a" not in per
    # a statement living inside letter b's own description must never speak for its siblings
    scoped = _caption_semantics("(a) Baseline. (b) Errors; error bars show SD.", ["a", "b"])
    assert scoped["type"] == "" and scoped["panels"]["b"]["type"] == "SD"
    # trailing figure-wide prose beyond the letters' sentence caps stays figure-level
    wide = _caption_semantics("(a) Baseline. (b) Errors. Error bars show the SEM throughout.",
                              ["a", "b"])
    assert wide["type"] == "SE" and wide["panels"] == {}


def test_caption_series_keys_parse_both_shapes_and_refuse_the_rest():
    from canopy.ingest.pdf import caption_series_keys

    balitsky = ("Circles represent adaptation with the right hand and squares represent "
                "adaptation with the left hand. Black lines represent cursor view trials.")
    keys = {k["descriptor"].lower(): k for k in caption_series_keys(balitsky)}
    assert "right hand" in keys["circles"]["series_text"]
    assert "left hand" in keys["squares"]["series_text"]
    assert keys["black lines"]["line_style"], "a line key never feeds marker matching"
    addison = ("Mean reaction time for dominant arm performance (open circles) and for "
               "nondominant arm performance (filled circles).")
    parsed = caption_series_keys(addison)
    assert {k["descriptor"] for k in parsed} == {"open circles", "filled circles"}
    assert caption_series_keys("Circles represent filled symbols.") == [], \
        "marker vocabulary inside the series text keys nothing"
    assert caption_series_keys("Circles show left hand. Circles show right hand.") == [], \
        "identical descriptors distinguish nothing — both dropped"
    assert caption_series_keys("The right hand adapted faster than the left.") == []


def test_figure_region_from_before_the_caption_fields_loads_with_defaults():
    from canopy.ingest.pdf import Bbox, FigureRegion

    old = dict(id="fig01", page=1, bbox=Bbox(x0=0, y0=0, x1=10, y1=10), caption="c", label="F",
               kind="raster", n_images=1, n_drawings=0, native_px=None, crop_png="a",
               claude_png="b", crop_dpi=72.0, claude_scale=1.0, confidence=0.9)
    fig = FigureRegion(**old)
    assert fig.caption_dispersion == "" and fig.caption_dispersion_panels == {}
    assert fig.caption_series_keys == []


def test_the_marker_vocabulary_has_one_home():
    from canopy.digitize import digitizer
    from canopy.ingest import pdf

    assert digitizer._FILL_WORDS is pdf.MARKER_FILL_WORDS
    assert digitizer._SHAPE_WORDS is pdf.MARKER_SHAPE_WORDS


def test_series_keys_never_mint_from_spread_statements_or_annotation_prose():
    """M-1's junk classes, each a verbatim shape from a real run that once minted a key."""
    from canopy.ingest.pdf import caption_series_keys

    junk = [
        "Error bars represent the standard error of the mean.",
        "Error bars represent SE of the mean across participants.",
        "Vertical bars show 95% confidence intervals.",
        "Bars are standard deviations with respect to mean performance.",
        "Error bars represent SE for the left hand.",       # would BIND and cap falsely
        "Stars indicate the significant GROUP by TRIAL interactions.",
        "The crosshair represents the cursor feedback position.",
        "Movements start at the position shown by the cursor.",
    ]
    for sentence in junk:
        assert caption_series_keys(sentence) == [], sentence
    # the paren-crossing junk: shape A must not eat half a parenthetical; shape B still keys it
    both = ("Data for naive performance (open circles) and opposite-arm performance "
            "(filled circles) are shown separately.")
    keys = caption_series_keys(both)
    assert {k["descriptor"] for k in keys} == {"open circles", "filled circles"}
    assert all(")" not in k["descriptor"] and "shown separately" not in k["series_text"]
               for k in keys)
    # …and the legitimate bar-series key survives the error-bar refusal
    assert caption_series_keys("Black bars represent cursor view trials.") \
        [0]["descriptor"] == "Black bars"
