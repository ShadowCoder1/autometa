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
    whisker_ends,
)
from canopy.ingest.pdf import Bbox, FigureRegion, PaperRecord, ingest_pdf

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
    # error bars in this figure are all vertical: a flat data segment must not pick one up as its "cap"
    assert {s.orientation for s in fig3.whiskers} == {"v"}
    assert fig3.warnings == []


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
    x, y = fig3.to_px(331.72, 346.67)
    cal = calibrate_from_scene(fig3, axis="y", near=(x, y))
    assert cal is not None
    assert cal.rmse < 0.5
    assert px_to_value(cal, y) == pytest.approx(-9.5, abs=0.3)
    assert px_to_value(cal, y) > px_to_value(cal, y + 10)   # downward is more negative


def _log_pdf(path: Path, *, plain_labels: bool) -> list[float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    ys = [2.0, 20.0, 200.0, 2000.0, 20000.0]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot([1, 2, 3, 4, 5], ys, "o-", color="black")
    ax.set_yscale("log")
    ax.set_ylim(1, 100000)
    if plain_labels:                                   # "1000" rather than the default 10^3
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
        ax.yaxis.set_major_formatter(ScalarFormatter())
    fig.savefig(path)
    plt.close(fig)
    return ys


def test_calibrate_from_scene_picks_a_log_axis(tmp_path):
    """A log y axis must not be read as linear (scale="auto" fits both and keeps the better one)."""
    ys = _log_pdf(tmp_path / "log.pdf", plain_labels=True)
    paper = ingest_pdf(tmp_path / "log.pdf", tmp_path / "ingest")
    scene = vector_candidates(paper, _whole_page_figure(paper))
    cal = calibrate_from_scene(scene, axis="y")
    assert cal is not None and cal.scale == "log"
    assert cal.rmse < 0.5
    for mark, want in zip(sorted(scene.marks, key=lambda m: m.x), ys):
        assert px_to_value(cal, mark.y) == pytest.approx(want, rel=0.02)
    assert calibrate_from_scene(scene, axis="y", scale="linear").rmse > cal.rmse


def test_superscript_exponent_labels_are_refused_not_guessed(tmp_path):
    """The text layer flattens 10^3 to "103"; a linear fit through 100..105 is perfect and wrong."""
    _log_pdf(tmp_path / "sup.pdf", plain_labels=False)
    paper = ingest_pdf(tmp_path / "sup.pdf", tmp_path / "ingest")
    scene = vector_candidates(paper, _whole_page_figure(paper))
    assert [t.text for t in scene.texts if t.text.startswith("10")] == ["100", "101", "102", "103", "104", "105"]
    assert all(t.value is None for t in scene.texts)
    assert any("superscript" in w for w in scene.warnings)
    assert calibrate_from_scene(scene, axis="y") is None
    assert calibrate_from_scene(scene, axis="x") is not None       # the plain x labels still calibrate


def test_calibrate_returns_none_without_ticks():
    empty = VectorScene(fig_id="x", page=1, crop_px_per_pt=1.0, origin_pt=(0.0, 0.0))
    assert calibrate_from_scene(empty) is None


# ------------------------------------------------------------------ snapping
def test_snap_to_vector_finds_the_nearest_mark(fig3):
    x, y = fig3.to_px(331.72, 346.67)
    assert fig3.to_pt(x, y) == pytest.approx((331.72, 346.67), abs=1e-6)   # round trip
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


def test_vector_pixels_reproduce_the_rendered_raster(fig3, heuer):
    """Regression guard on the PDF-point -> crop-pixel mapping, measured against the rendered PNG.

    Every axis line the vector scene reports is looked for in the raster at the pixel position the mapping
    predicts. A wrong scale drifts across the image and a wrong origin (or a missed half-pixel convention)
    biases every line the same way, so both the worst error and the spread are asserted.
    """
    from canopy.digitize.cv import load_gray, snap_horizontal_edge, snap_vertical_edge
    fig = next(f for f in heuer.figures if f.id == "fig03")
    img = load_gray(Path(heuer.out_dir) / fig.crop_png)
    h, w = img.shape
    offsets = []
    for seg in fig3.axis_lines:
        lo, hi = (min(seg.y0, seg.y1), max(seg.y0, seg.y1)) if seg.orientation == "v" else \
                 (min(seg.x0, seg.x1), max(seg.x0, seg.x1))
        along = lo + 0.35 * (hi - lo)                      # off-centre: away from the panel's data cluster
        window = int(seg.width) + 4                        # must reach past the stroke itself
        if seg.orientation == "v":
            if not (window < seg.x0 < w - window - 1):
                continue
            got, conf = snap_vertical_edge(img, seg.x0, along, window=window, band=2)
            predicted = seg.x0
        else:
            if not (window < seg.y0 < h - window - 1):
                continue
            got, conf = snap_horizontal_edge(img, along, seg.y0, window=window, band=2)
            predicted = seg.y0
        assert conf > 0.3, f"no ink where the vector scene says the axis is: {seg}"
        offsets.append(got - predicted)
    assert len(offsets) >= 8
    assert max(abs(o) for o in offsets) < 0.5                      # measured: 0.13 px worst case
    assert max(offsets) - min(offsets) < 0.4                       # measured: 0.17 px, i.e. no drift
    assert abs(sum(offsets) / len(offsets)) < 0.25                 # array indices, not device edges (0.5)


# ------------------------------------------------------------------ synthetic vector PDF (end-to-end path A)
def _errorbar_pdf(path: Path, *, capsize: float, markers: bool = True) -> dict:
    """A matplotlib error-bar chart saved as a vector PDF, with the plotted truth returned."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [1, 2, 3, 4, 5, 6]
    ys = [10.0, 14.0, 12.5, 17.0, 15.5, 19.0]
    err = [1.5, 2.0, 1.0, 2.5, 1.75, 1.25]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.errorbar(xs, ys, yerr=err, fmt="o-" if markers else "none", color="black", capsize=capsize)
    ax.set_ylim(0, 25)
    ax.set_xlim(0, 7)
    fig.savefig(path)
    plt.close(fig)
    return dict(xs=xs, ys=ys, err=err)


def _whole_page_figure(paper: PaperRecord, fig_id: str = "figA") -> FigureRegion:
    """matplotlib PDFs are one figure per page; ingestion's region finder needs captions, so take the page."""
    page = paper.pages[0]
    return FigureRegion(id=fig_id, page=1, bbox=Bbox(0.0, 0.0, page.width_pt, page.height_pt), caption="",
                        label="", kind="vector", n_images=0, n_drawings=0, native_px=None, crop_png="",
                        claude_png="", crop_dpi=200.0, claude_scale=1.0, confidence=1.0)


@pytest.fixture(scope="module")
def capped(tmp_path_factory) -> tuple[VectorScene, dict]:
    out = tmp_path_factory.mktemp("errorbar")
    truth = _errorbar_pdf(out / "capped.pdf", capsize=4)
    paper = ingest_pdf(out / "capped.pdf", out / "ingest")
    return vector_candidates(paper, _whole_page_figure(paper)), truth


@pytest.fixture(scope="module")
def capless(tmp_path_factory) -> tuple[VectorScene, dict]:
    out = tmp_path_factory.mktemp("errorbar_nocap")
    truth = _errorbar_pdf(out / "capless.pdf", capsize=0)
    paper = ingest_pdf(out / "capless.pdf", out / "ingest")
    return vector_candidates(paper, _whole_page_figure(paper)), truth


def _y_calibration(scene: VectorScene):
    cal = calibrate_from_scene(scene, axis="y")
    assert cal is not None and cal.rmse < 0.5
    return cal


def test_capped_error_bars_are_paired_across_paths(capped):
    """matplotlib emits every stroke as its own path: stem, upper cap and lower cap are three drawings."""
    scene, truth = capped
    assert len(scene.marks) == len(truth["xs"])          # tick marks and caps are strokes, not marks
    assert len(scene.whiskers) >= len(truth["xs"])
    assert scene.warnings == []
    cal = _y_calibration(scene)
    seen = 0
    for mark in scene.marks:
        ends = whisker_ends(scene, mark)
        assert ends is not None, f"no error bar found for the mark at {mark.x:.1f},{mark.y:.1f}"
        top, bottom = ends
        mean = px_to_value(cal, mark.y)
        i = min(range(len(truth["ys"])), key=lambda k: abs(truth["ys"][k] - mean))
        assert mean == pytest.approx(truth["ys"][i], abs=0.05)
        assert px_to_value(cal, top) - mean == pytest.approx(truth["err"][i], abs=0.05)
        assert mean - px_to_value(cal, bottom) == pytest.approx(truth["err"][i], abs=0.05)
        seen += 1
    assert seen == len(truth["xs"])


def test_capless_error_bars_are_found_through_their_marker(capless):
    scene, truth = capless
    assert len(scene.whiskers) >= len(truth["xs"])
    assert scene.warnings == []
    cal = _y_calibration(scene)
    for mark in scene.marks:
        top, bottom = whisker_ends(scene, mark)
        mean = px_to_value(cal, mark.y)
        i = min(range(len(truth["ys"])), key=lambda k: abs(truth["ys"][k] - mean))
        assert px_to_value(cal, top) - mean == pytest.approx(truth["err"][i], abs=0.05)
        assert mean - px_to_value(cal, bottom) == pytest.approx(truth["err"][i], abs=0.05)


def test_unattributable_error_bars_raise_a_warning(tmp_path):
    """Stems with neither caps nor markers cannot be attributed — say so instead of silently finding none."""
    _errorbar_pdf(tmp_path / "bare.pdf", capsize=0, markers=False)
    paper = ingest_pdf(tmp_path / "bare.pdf", tmp_path / "ingest")
    scene = vector_candidates(paper, _whole_page_figure(paper))
    assert scene.whiskers == []
    assert any("error bar" in w for w in scene.warnings)
    assert "warnings" in json.loads(scene.to_json())
