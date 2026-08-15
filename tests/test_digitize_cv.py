"""Task 6a: raster CV core — axis/tick detection, tick-label OCR, sub-pixel snapping, bars, markers, overlay.

Synthetic figures are generated with matplotlib so the ground truth (bar tops, tick rows, marker centres)
is known exactly through `ax.transData`; the real-figure checks run on the Bock 2005 Fig. 1 crop produced
by ingestion. Offline; OCR asserts skip when the tesseract binary is missing.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from canopy.digitize.calibrate import fit_axis, pair_ticks, px_to_value
from canopy.digitize.cv import (
    Marker,
    detect_bars,
    detect_markers,
    detect_whiskers,
    find_axes,
    find_cap_ends,
    find_tick_marks,
    load_color,
    load_gray,
    ocr_tick_labels,
    snap_horizontal_edge,
    snap_vertical_edge,
)
from canopy.digitize.cv import run_tesseract, tesseract_path
from canopy.digitize.overlay import draw_overlay
from canopy.ingest.pdf import PaperRecord, ingest_pdf

PDF_DIR = Path(__file__).resolve().parent / "fixtures" / "pdfs"
needs_tesseract = pytest.mark.skipif(tesseract_path() is None,      # same resolver the code uses
                                     reason="tesseract binary not installed (see CANOPY_TESSERACT)")


# ------------------------------------------------------------------ synthetic figure fixtures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@pytest.fixture(scope="module")
def bar_chart(tmp_path_factory) -> dict:
    """A 4-bar chart at 150 dpi; truth = bar rectangles and y-tick rows in image pixels."""
    plt = _mpl()
    heights = [12.0, 27.5, 40.0, 33.25]
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    bars = ax.bar([0, 1, 2, 3], heights, width=0.6, color="#4c72b0")
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["A", "B", "C", "D"])
    fig.canvas.draw()
    path = tmp_path_factory.mktemp("cv") / "bars.png"
    fig.savefig(path, dpi=150)
    h_px = int(round(fig.get_size_inches()[1] * 150))
    truth_bars = []
    for h, patch in zip(heights, bars):
        bb = patch.get_window_extent()
        truth_bars.append(dict(value=h, x0=bb.x0, x1=bb.x1, top_y=h_px - bb.y1, base_y=h_px - bb.y0))
    ticks = [(h_px - ax.transData.transform((0, v))[1], v) for v in ax.get_yticks()]
    axes_bb = ax.get_window_extent()
    truth = dict(path=path, bars=truth_bars, ticks=ticks, height=h_px,
                 axes_left=axes_bb.x0, axes_right=axes_bb.x1,
                 axes_top=h_px - axes_bb.y1, axes_bottom=h_px - axes_bb.y0)
    plt.close(fig)
    return truth


@pytest.fixture(scope="module")
def scatter_chart(tmp_path_factory) -> dict:
    plt = _mpl()
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ys = [3.0, 5.5, 4.25, 7.0, 6.5, 8.0, 5.0, 9.5, 7.75, 6.0, 8.5, 4.5]
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.scatter(xs, ys, s=70, color="#222222")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    fig.canvas.draw()
    path = tmp_path_factory.mktemp("cv") / "scatter.png"
    fig.savefig(path, dpi=150)
    h_px = int(round(fig.get_size_inches()[1] * 150))
    pts = [(ax.transData.transform((x, y))[0], h_px - ax.transData.transform((x, y))[1]) for x, y in zip(xs, ys)]
    plt.close(fig)
    return dict(path=path, points=pts)


@pytest.fixture(scope="module")
def bock(tmp_path_factory) -> PaperRecord:
    return ingest_pdf(PDF_DIR / "bock2005.pdf", tmp_path_factory.mktemp("bock"))


# ------------------------------------------------------------------ loading
def test_load_gray_and_color(bar_chart):
    g = load_gray(bar_chart["path"])
    c = load_color(bar_chart["path"])
    assert g.ndim == 2 and g.dtype == np.uint8
    assert c.shape == (g.shape[0], g.shape[1], 3)
    assert g.shape[0] == bar_chart["height"]


# ------------------------------------------------------------------ sub-pixel snapping
def test_snap_horizontal_edge_finds_an_antialiased_line_within_half_a_pixel():
    img = np.full((200, 200), 255, np.uint8)
    truth = 123.4
    cv2.line(img, (10 * 16, int(round(truth * 16))), (190 * 16, int(round(truth * 16))), 0,
             thickness=1, shift=4, lineType=cv2.LINE_AA)
    y, conf = snap_horizontal_edge(img, x=100.0, y=120.0, window=8)
    assert abs(y - truth) < 0.5
    assert conf > 0.5


def test_snap_horizontal_edge_reports_no_confidence_on_blank_area():
    img = np.full((200, 200), 255, np.uint8)
    y, conf = snap_horizontal_edge(img, x=100.0, y=50.0, window=6)
    assert conf == 0.0
    assert y == 50.0


def test_snap_horizontal_edge_ignores_a_second_line_outside_the_window():
    img = np.full((200, 200), 255, np.uint8)
    cv2.line(img, (10 * 16, int(60.25 * 16)), (190 * 16, int(60.25 * 16)), 0, 1, cv2.LINE_AA, 4)
    cv2.line(img, (10 * 16, int(75.0 * 16)), (190 * 16, int(75.0 * 16)), 0, 1, cv2.LINE_AA, 4)
    y, conf = snap_horizontal_edge(img, x=100.0, y=61.0, window=5)
    assert abs(y - 60.25) < 0.5


def test_snap_vertical_edge_finds_an_antialiased_column():
    img = np.full((200, 200), 255, np.uint8)
    truth = 77.7
    cv2.line(img, (int(round(truth * 16)), 10 * 16), (int(round(truth * 16)), 190 * 16), 0,
             thickness=1, shift=4, lineType=cv2.LINE_AA)
    x, conf = snap_vertical_edge(img, x=80.0, y=100.0, window=8)
    assert abs(x - truth) < 0.5
    assert conf > 0.5


# ------------------------------------------------------------------ error-bar cap ends
def _error_bar_image() -> np.ndarray:
    img = np.full((240, 240), 255, np.uint8)
    cv2.line(img, (100, 60), (100, 180), 0, 2)          # stem
    cv2.line(img, (92, 60), (108, 60), 0, 2)            # top cap
    cv2.line(img, (92, 180), (108, 180), 0, 2)          # bottom cap
    cv2.circle(img, (100, 120), 5, 0, -1)               # marker
    return img


def test_find_cap_ends_locates_both_caps():
    top, bottom = find_cap_ends(_error_bar_image(), x=100.0, y_center=120.0, max_len_px=100.0)
    assert top is not None and bottom is not None
    assert abs(top - 60.0) < 1.0
    assert abs(bottom - 180.0) < 1.0


def test_find_cap_ends_returns_none_without_ink():
    img = np.full((240, 240), 255, np.uint8)
    assert find_cap_ends(img, x=50.0, y_center=100.0, max_len_px=60.0) == (None, None)


def test_find_cap_ends_respects_max_len():
    top, bottom = find_cap_ends(_error_bar_image(), x=100.0, y_center=120.0, max_len_px=20.0)
    assert top is not None and abs(top - 100.0) <= 2.0     # walk stopped inside the stem
    assert bottom is not None and abs(bottom - 140.0) <= 2.0


def test_detect_whiskers_finds_the_caps_of_a_marker():
    img = _error_bar_image()
    top, bottom = detect_whiskers(img, dict(x=100.0, y=120.0))
    assert abs(top - 60.0) < 1.0 and abs(bottom - 180.0) < 1.0          # the caps drawn at y=60 and y=180
    assert detect_whiskers(img, dict(x=100.0, y=120.0)) == find_cap_ends(img, 100.0, 120.0, 240 * 0.35)
    assert detect_whiskers(img, Marker(x=100.0, y=120.0, colour="#000000", kind="circle")) == (top, bottom)
    assert detect_whiskers(img, (100.0, 120.0)) == (top, bottom)


def test_detect_whiskers_rejects_a_dict_without_coordinates():
    with pytest.raises(ValueError, match="x_center"):
        detect_whiskers(_error_bar_image(), dict(left=1.0, top=2.0))


# ------------------------------------------------------------------ axes, ticks, OCR
def test_find_axes_on_bar_chart(bar_chart):
    img = load_gray(bar_chart["path"])
    axes = find_axes(img)
    assert axes.y_axis_x is not None and abs(axes.y_axis_x - bar_chart["axes_left"]) < 2.0
    assert axes.x_axis_y is not None and abs(axes.x_axis_y - bar_chart["axes_bottom"]) < 2.0
    x0, y0, x1, y1 = axes.plot_bbox
    assert abs(x0 - bar_chart["axes_left"]) < 2.0
    assert abs(x1 - bar_chart["axes_right"]) < 3.0
    assert abs(y1 - bar_chart["axes_bottom"]) < 2.0
    assert abs(y0 - bar_chart["axes_top"]) < 3.0
    assert axes.confidence > 0.8


def test_find_axes_returns_low_confidence_on_a_blank_image():
    axes = find_axes(np.full((100, 100), 255, np.uint8))
    assert axes.y_axis_x is None and axes.x_axis_y is None
    assert axes.confidence == 0.0


def test_find_tick_marks_on_bar_chart(bar_chart):
    img = load_gray(bar_chart["path"])
    ticks = find_tick_marks(img, find_axes(img))
    left = ticks["left"]
    assert len(left) == 6                                    # 0,10,20,30,40,50
    for row, _value in bar_chart["ticks"]:
        assert min(abs(row - t) for t in left) < 1.5
    assert len(ticks["bottom"]) == 4                          # A,B,C,D


@needs_tesseract
def test_ocr_tick_labels_reads_the_y_axis(bar_chart):
    img = load_gray(bar_chart["path"])
    axes = find_axes(img)
    labels = ocr_tick_labels(img, axes, side="left")
    values = sorted(t.value for t in labels if t.value is not None)
    expected = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0}
    assert len(expected & set(values)) >= 4
    for t in labels:
        assert t.source == "ocr"
        assert t.bbox[2] <= axes.y_axis_x + 2                 # labels live left of the axis


@needs_tesseract
def test_ocr_plus_ticks_calibrate_the_bar_chart(bar_chart):
    """End-to-end: OCR'd labels + detected tick rows must reproduce the plotted bar values."""
    img = load_gray(bar_chart["path"])
    axes = find_axes(img)
    ticks = find_tick_marks(img, axes)
    labels = ocr_tick_labels(img, axes, side="left")
    pairs = pair_ticks(labels, ticks["left"])
    cal = fit_axis(pairs)
    assert len(pairs) >= 4
    assert cal.rmse < 1.5
    for bar in bar_chart["bars"]:
        assert abs(px_to_value(cal, bar["top_y"]) - bar["value"]) < 0.3


@needs_tesseract
def test_ocr_tick_labels_reads_categorical_labels_below_the_x_axis(bar_chart):
    img = load_gray(bar_chart["path"])
    axes = find_axes(img)
    labels = ocr_tick_labels(img, axes, side="bottom", ticks=find_tick_marks(img, axes)["bottom"])
    assert [t.text for t in labels] == ["A", "B", "C", "D"]
    assert all(t.value is None for t in labels)


def test_ocr_tick_labels_without_an_axis_or_with_a_bad_side(bar_chart):
    img = load_gray(bar_chart["path"])
    axes = find_axes(img)
    blind = replace(axes, y_axis_x=None, x_axis_y=None)
    assert ocr_tick_labels(img, blind, side="left") == []
    assert ocr_tick_labels(img, blind, side="bottom") == []
    with pytest.raises(ValueError):
        ocr_tick_labels(img, axes, side="right")


def test_ocr_status_reports_a_missing_binary(bar_chart, monkeypatch):
    """A missing tesseract must be distinguishable from a figure that genuinely has no labels."""
    monkeypatch.setenv("CANOPY_TESSERACT", "")
    monkeypatch.setattr("canopy.digitize.cv.shutil.which", lambda _name: None)
    img = load_gray(bar_chart["path"])
    labels = ocr_tick_labels(img, find_axes(img), side="left")
    assert labels == []
    assert labels.status == "missing"
    assert run_tesseract("nonexistent.png").status == "missing"


@needs_tesseract
def test_ocr_status_is_ok_when_labels_are_read(bar_chart):
    img = load_gray(bar_chart["path"])
    labels = ocr_tick_labels(img, find_axes(img), side="left")
    assert labels.status == "ok"
    assert list(labels) == [t for t in labels]                  # behaves as a plain list


# ------------------------------------------------------------------ bars
def test_detect_bars_recovers_bar_tops(bar_chart):
    img = load_color(bar_chart["path"])
    axes = find_axes(img)
    bars = detect_bars(img, axes)
    assert len(bars) == 4
    for got, want in zip(bars, bar_chart["bars"]):
        assert abs(got.top_y - want["top_y"]) < 1.0
        assert abs(got.x0 - want["x0"]) < 2.0
        assert abs(got.x1 - want["x1"]) < 2.0
        assert abs(got.x_center - 0.5 * (want["x0"] + want["x1"])) < 2.0
        assert abs(got.base_y - want["base_y"]) < 2.0
        assert not got.narrow
        assert got.colour.lower() == "#4c72b0"


def test_detect_bars_flags_narrow_bars(tmp_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.bar([0, 1, 2], [10.0, 20.0, 30.0], width=0.03, color="#333333")
    ax.set_ylim(0, 40)
    ax.set_xlim(-0.5, 2.5)          # 3 data units over ~697 px -> bars are ~7 px wide
    path = tmp_path / "narrow.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    img = load_color(path)
    bars = detect_bars(img, find_axes(img))
    assert len(bars) == 3
    assert all(b.narrow for b in bars)
    assert all(b.x1 - b.x0 < 8 for b in bars)


# ------------------------------------------------------------------ markers
def test_detect_markers_on_scatter(scatter_chart):
    img = load_color(scatter_chart["path"])
    marks = detect_markers(img, find_axes(img))
    assert len(marks) >= 11
    for tx, ty in scatter_chart["points"]:
        assert min((m.x - tx) ** 2 + (m.y - ty) ** 2 for m in marks) ** 0.5 < 2.0
    assert sum(m.kind == "circle" for m in marks) >= 8


def test_detect_markers_separates_markers_joined_by_a_line(tmp_path):
    plt = _mpl()
    xs = [1, 2, 3, 4, 5, 6, 7]
    ys = [2.0, 4.0, 3.0, 5.5, 4.5, 6.0, 5.0]
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.plot(xs, ys, "o-", color="black", markersize=8, linewidth=1.5)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    fig.canvas.draw()
    path = tmp_path / "line.png"
    fig.savefig(path, dpi=150)
    h_px = int(round(fig.get_size_inches()[1] * 150))
    truth = [(ax.transData.transform((x, y))[0], h_px - ax.transData.transform((x, y))[1]) for x, y in zip(xs, ys)]
    plt.close(fig)
    img = load_color(path)
    marks = detect_markers(img, find_axes(img))
    assert len(marks) >= 6
    hits = sum(1 for tx, ty in truth if min((m.x - tx) ** 2 + (m.y - ty) ** 2 for m in marks) ** 0.5 < 2.5)
    assert hits >= 6


# ------------------------------------------------------------------ real figure: Bock 2005 Fig. 1
def test_bock_figure_axis_and_ticks(bock):
    fig = next(f for f in bock.figures if f.id == "fig01")
    assert fig.page == 3
    img = load_gray(Path(bock.out_dir) / fig.crop_png)
    axes = find_axes(img)
    assert axes.y_axis_x is not None
    ticks = find_tick_marks(img, axes)
    assert len(ticks["left"]) >= 5


@needs_tesseract
def test_bock_figure_calibrates_from_ocr(bock):
    fig = next(f for f in bock.figures if f.id == "fig01")
    img = load_gray(Path(bock.out_dir) / fig.crop_png)
    axes = find_axes(img)
    ticks = find_tick_marks(img, axes)
    labels = ocr_tick_labels(img, axes, side="left", ticks=ticks["left"])
    assert sorted(t.value for t in labels if t.value is not None) == [-40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
    pairs = pair_ticks(labels, ticks["left"])
    assert len(pairs) >= 4
    cal = fit_axis(pairs)
    assert cal.rmse < 3.0
    # the y axis of Fig. 1 runs -40 .. 60 degrees over the plot height
    x0, y0, x1, y1 = axes.plot_bbox
    assert px_to_value(cal, y0) > px_to_value(cal, y1)
    assert 40.0 < px_to_value(cal, y0) < 80.0


# ------------------------------------------------------------------ overlay
def test_draw_overlay_marks_are_visible(bar_chart, tmp_path):
    out = tmp_path / "overlay.png"
    marks = [
        dict(x=200.0, y=150.0, label="bar 1 top", color="#e6194b", kind="point"),
        dict(x=0.0, y=300.0, label="zero", color="#3cb44b", kind="hline"),
        dict(x=400.0, y=0.0, label="episode 20", color="#4363d8", kind="vline"),
        dict(x=500.0, y=200.0, x1=560.0, y1=260.0, label="box", color="#f58231", kind="box"),
        dict(x=120.0, y=420.0, label="caption", color="#911eb4", kind="text"),
    ]
    path = draw_overlay(bar_chart["path"], marks, out)
    assert path == out and out.exists()
    src = cv2.imread(str(bar_chart["path"]))
    got = cv2.imread(str(out))
    assert got.shape == src.shape
    patch = got[145:156, 195:206].reshape(-1, 3)
    assert np.any(np.all(np.abs(patch.astype(int) - np.array([75, 25, 230])) <= 12, axis=1))   # BGR of #e6194b
    row = got[300, :, :].astype(int)
    assert np.any(np.all(np.abs(row - np.array([75, 180, 60])) <= 12, axis=1))                 # BGR of #3cb44b
    assert not np.array_equal(src, got)


def test_draw_overlay_numbers_every_mark(bar_chart, tmp_path):
    out = tmp_path / "numbered.png"
    marks = [dict(x=120.0 + 90 * i, y=100.0, color="#e6194b", kind="point") for i in range(3)]
    draw_overlay(bar_chart["path"], marks, out)
    got = cv2.imread(str(out))
    src = cv2.imread(str(bar_chart["path"]))
    for i, mark in enumerate(marks, start=1):
        x, y = int(mark["x"]), int(mark["y"])
        ring = got[y - 12:y + 12, x - 12:x + 12]
        assert np.any(np.all(np.abs(ring.astype(int) - np.array([75, 25, 230])) <= 12, axis=2)), f"no ring {i}"
        # the number is written to the right of the ring: coloured glyph ink that was not there before
        label = got[y - 18:y + 18, x + 12:x + 46]
        was = src[y - 18:y + 18, x + 12:x + 46]
        glyph = np.all(np.abs(label.astype(int) - np.array([75, 25, 230])) <= 40, axis=2)
        assert glyph.sum() >= 12, f"no digits drawn next to mark {i}"
        assert not np.array_equal(label, was)


@needs_tesseract
def test_ocr_tick_labels_keeps_wide_numeric_labels_whole(tmp_path):
    """A long label ("-100000") must not lose its outer glyph to the neighbouring tick.

    Components are grouped into labels *before* they are assigned to ticks; assigning each glyph on its own
    turns "-100000" into "100000" (a sign error) or "10000" (a wrong value).
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
    xs = [-100000, 0, 100000, 200000]
    ax.plot(xs, [1.0, 2.0, 1.5, 2.5], "o-", color="black")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_xlim(-160000, 260000)
    path = tmp_path / "wide_labels.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    img = load_gray(path)
    axes = find_axes(img)
    labels = ocr_tick_labels(img, axes, side="bottom", ticks=find_tick_marks(img, axes)["bottom"])
    assert sorted(t.value for t in labels if t.value is not None) == [-100000.0, 0.0, 100000.0, 200000.0]
