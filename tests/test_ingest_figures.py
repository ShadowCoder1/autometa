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
BOCK = ROOT / "tests" / "fixtures" / "pdfs" / "bock2005.pdf"


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
