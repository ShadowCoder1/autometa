"""Task 6a: path A — exact geometry read out of the PDF drawing commands.

Ground truth is the vector figure itself: Heuer & Hegele (2008) Fig. 3 (page 6) is a 2x2 panel line chart
with filled/open circle markers and capped error bars, so every mark position is exact to the point and the
calibration should be near-perfect. Bock 2005 Fig. 1 is a raster figure and must degrade gracefully.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canopy.digitize.calibrate import px_to_value
from canopy.digitize.vector import (
    Mark,
    Segment,
    VectorScene,
    calibrate_from_scene,
    snap_to_vector,
    vector_candidates,
)
from canopy.ingest.pdf import PaperRecord, ingest_pdf

PDF_DIR = Path(__file__).resolve().parent / "fixtures" / "pdfs"


@pytest.fixture(scope="module")
def heuer(tmp_path_factory) -> PaperRecord:
    return ingest_pdf(PDF_DIR / "heuer2008.pdf", tmp_path_factory.mktemp("heuer"))


@pytest.fixture(scope="module")
def bock(tmp_path_factory) -> PaperRecord:
    return ingest_pdf(PDF_DIR / "bock2005.pdf", tmp_path_factory.mktemp("bock_vec"))


@pytest.fixture(scope="module")
def fig3(heuer) -> VectorScene:
    fig = next(f for f in heuer.figures if f.id == "fig03")
    assert fig.page == 6 and fig.kind == "vector"
    return vector_candidates(heuer, fig)


# ------------------------------------------------------------------ scene contents
def test_vector_candidates_finds_labels_and_marks(fig3):
    assert fig3.fig_id == "fig03" and fig3.page == 6
    assert fig3.crop_px_per_pt == pytest.approx(500 / 72.0)
    numeric = [t for t in fig3.tick_labels if t.value is not None]
    assert len(numeric) >= 6            # observed: 32 (4 y labels x 4 panels + the shared x axis labels)
    assert len(fig3.marks) >= 20        # observed: 67 (66 marker discs incl. legend keys + the legend box)
    assert {t.value for t in numeric} >= {-60.0, -40.0, -20.0, -10.0, 0.0, 10.0}
    assert all(t.source == "vector" and t.confidence == 1.0 for t in fig3.tick_labels)
    assert any("young" in t.text for t in fig3.texts)      # legend keys keep their position


def test_vector_candidates_classifies_the_scene(fig3):
    assert len(fig3.axis_lines) >= 4                       # observed 12: 4 panel y axes, x axes, zero lines
    assert len(fig3.tick_lines) >= 16                      # observed 32: 4 y ticks per panel + x ticks
    assert len(fig3.whiskers) >= 40                        # observed 128: two stems per plotted mean
    assert all(s.role == "tick" for s in fig3.tick_lines)
    assert all(s.role == "axis" for s in fig3.axis_lines)
    # ticks sit on an axis line and are much shorter than it
    longest_tick = max(s.length for s in fig3.tick_lines)
    assert longest_tick < 0.25 * min(s.length for s in fig3.axis_lines)


def test_marks_and_labels_are_in_crop_pixels(fig3, heuer):
    from PIL import Image
    fig = next(f for f in heuer.figures if f.id == "fig03")
    w, h = Image.open(Path(heuer.out_dir) / fig.crop_png).size
    inside = [m for m in fig3.marks if 0 <= m.x <= w and 0 <= m.y <= h]
    assert len(inside) >= 0.9 * len(fig3.marks)
    assert fig3.origin_pt[0] == pytest.approx(fig.bbox.x0, abs=0.3)
    assert fig3.origin_pt[1] == pytest.approx(fig.bbox.y0, abs=0.3)


def test_scene_serializes_to_json(fig3):
    d = json.loads(fig3.to_json())
    assert d["fig_id"] == "fig03"
    assert len(d["marks"]) == len(fig3.marks)
    assert set(d["marks"][0]) >= {"x", "y", "kind", "fill"}
    assert set(d["tick_labels"][0]) >= {"text", "value", "bbox", "center"}


# ------------------------------------------------------------------ calibration
def test_calibrate_from_scene_is_near_exact(fig3):
    cal = calibrate_from_scene(fig3, axis="y")
    assert cal is not None
    assert cal.rmse < 0.5                                   # pixels, at 500 dpi
    assert len(cal.ticks) >= 4
    assert cal.a < 0                                        # image rows grow downward


def test_calibrate_selects_the_panel_nearest_a_point(fig3):
    # bottom-right panel ("aftereffect / without explicit knowledge"), y axis -20..10:
    # its first young mean is drawn at PDF point (331.7, 346.7) -> about -9.5 deg
    px_per_pt = fig3.crop_px_per_pt
    x = (331.72 - fig3.origin_pt[0]) * px_per_pt
    y = (346.67 - fig3.origin_pt[1]) * px_per_pt
    cal = calibrate_from_scene(fig3, axis="y", near=(x, y))
    assert cal is not None
    assert cal.rmse < 0.5
    assert px_to_value(cal, y) == pytest.approx(-9.5, abs=0.3)
    assert px_to_value(cal, y) > px_to_value(cal, y + 10)   # downward is more negative


def test_calibrate_returns_none_without_ticks():
    empty = VectorScene(fig_id="x", page=1, crop_px_per_pt=1.0, origin_pt=(0.0, 0.0))
    assert calibrate_from_scene(empty) is None


# ------------------------------------------------------------------ snapping
def test_snap_to_vector_finds_the_nearest_mark(fig3):
    px_per_pt = fig3.crop_px_per_pt
    x = (331.72 - fig3.origin_pt[0]) * px_per_pt
    y = (346.67 - fig3.origin_pt[1]) * px_per_pt
    got = snap_to_vector(fig3, x + 4, y - 5, radius=20)
    assert got is not None
    assert got[0] == pytest.approx(x, abs=1.5)
    assert got[1] == pytest.approx(y, abs=1.5)


def test_snap_to_vector_returns_none_when_far_away(fig3):
    assert snap_to_vector(fig3, -500.0, -500.0, radius=8) is None


def test_snap_to_vector_prefers_marks_over_whisker_ends():
    scene = VectorScene(fig_id="s", page=1, crop_px_per_pt=1.0, origin_pt=(0.0, 0.0))
    scene.marks.append(Mark(x=100.0, y=100.0, w=5.0, h=5.0, kind="circle", fill="#000000"))
    scene.whiskers.append(Segment(100.0, 100.0, 105.0, 100.0, role="whisker"))
    assert snap_to_vector(scene, 104.0, 100.0, radius=10) == (100.0, 100.0)   # mark wins although further
    assert snap_to_vector(scene, 104.0, 100.0, radius=2) == (105.0, 100.0)    # only the whisker end is close


def test_snap_to_vector_returns_a_whisker_end_when_no_mark_is_near(fig3):
    far = [s for s in fig3.whiskers
           if all(((s.x1 - m.x) ** 2 + (s.y1 - m.y) ** 2) ** 0.5 > 25 for m in fig3.marks)]
    assert far, "expected error-bar caps that sit well away from any marker"
    s = far[0]
    assert snap_to_vector(fig3, s.x1 + 1.0, s.y1 - 1.0, radius=8) == pytest.approx((s.x1, s.y1))


# ------------------------------------------------------------------ raster figure degrades gracefully
def test_vector_candidates_on_a_raster_figure(bock):
    fig = next(f for f in bock.figures if f.id == "fig01")
    assert fig.kind == "raster"
    scene = vector_candidates(bock, fig)
    assert scene.fig_id == "fig01"
    assert scene.marks == [] and scene.tick_lines == []
    assert calibrate_from_scene(scene) is None
    assert snap_to_vector(scene, 100.0, 100.0) is None
