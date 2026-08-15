"""Path A: exact figure geometry from the PDF drawing commands (no pixels, no LLM, no guessing).

For a vector figure the paper already contains the numbers we are trying to read: every marker, error-bar
cap, tick and tick label is a drawing or text operator at an exact PDF coordinate. `vector_candidates`
collects them into a `VectorScene`, expressed in the **crop pixels of `FigureRegion.crop_png`** so that the
scene, the raster CV of `canopy.digitize.cv` and any pixel coordinates a vision model returns all live in
one coordinate system.

Coordinate mapping (verified against `canopy.ingest.pdf.ingest_pdf`, which renders the crop with
``page.get_pixmap(dpi=fig.crop_dpi, clip=fig.bbox)``)::

    zoom      = fig.crop_dpi / 72                       # crop pixels per PDF point
    origin_pt = ((fig.bbox * Matrix(zoom, zoom)).irect.x0 / zoom, ... .y0 / zoom)
    crop_px   = (pdf_pt - origin_pt) * zoom

`origin_pt` is the *rounded* clip origin PyMuPDF actually used, so the mapping is exact rather than off by
the sub-pixel remainder of `bbox.x0 * zoom`. PyMuPDF returns drawings and text in the page's displayed
coordinate space (rotation already applied), the same space the clip rectangle lives in, so rotated pages
need no special case.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pymupdf

from ..ingest.pdf import FigureRegion, PaperRecord
from .calibrate import AxisCalibration, TickLabel, fit_axis, pair_ticks, parse_number

Role = Literal["axis", "tick", "whisker", "data", "other"]


@dataclass
class Segment:
    """A straight stroked segment of the figure, in crop pixels."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float = 0.0                       # stroke width, crop pixels
    colour: str | None = None                # "#rrggbb"
    role: Role = "other"

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def orientation(self) -> str:
        dx, dy = abs(self.x1 - self.x0), abs(self.y1 - self.y0)
        if dy <= max(0.5, 0.02 * dx):
            return "h"
        if dx <= max(0.5, 0.02 * dy):
            return "v"
        return "d"

    def to_dict(self) -> dict:
        return dict(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1, width=self.width, colour=self.colour,
                    role=self.role)


@dataclass
class Mark:
    """A filled shape that plots a datum: a marker disc, a bar, a box. Centre and size in crop pixels."""

    x: float
    y: float
    w: float
    h: float
    kind: str                                # circle | rect | triangle | polygon | path
    fill: str | None = None
    stroke: str | None = None

    def to_dict(self) -> dict:
        return dict(x=self.x, y=self.y, w=self.w, h=self.h, kind=self.kind, fill=self.fill, stroke=self.stroke)


@dataclass
class VectorScene:
    """Everything a vector figure declares about itself, in crop pixels."""

    fig_id: str
    page: int
    crop_px_per_pt: float
    origin_pt: tuple[float, float]
    tick_labels: list[TickLabel] = field(default_factory=list)   # numeric labels (axis ticks)
    texts: list[TickLabel] = field(default_factory=list)         # other words: legend keys, axis titles
    tick_lines: list[Segment] = field(default_factory=list)
    axis_lines: list[Segment] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    whiskers: list[Segment] = field(default_factory=list)        # error-bar stems (datum end first)
    hlines: list[Segment] = field(default_factory=list)          # other horizontal strokes
    vlines: list[Segment] = field(default_factory=list)          # other vertical strokes
    data_segments: list[Segment] = field(default_factory=list)   # oblique strokes (connecting lines)

    def to_pt(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Crop pixels -> PDF points on the page."""
        return (self.origin_pt[0] + x_px / self.crop_px_per_pt, self.origin_pt[1] + y_px / self.crop_px_per_pt)

    def to_px(self, x_pt: float, y_pt: float) -> tuple[float, float]:
        """PDF points -> crop pixels."""
        return ((x_pt - self.origin_pt[0]) * self.crop_px_per_pt, (y_pt - self.origin_pt[1]) * self.crop_px_per_pt)

    def to_json(self) -> str:
        return json.dumps(dict(
            fig_id=self.fig_id, page=self.page, crop_px_per_pt=self.crop_px_per_pt, origin_pt=list(self.origin_pt),
            tick_labels=[t.to_dict() for t in self.tick_labels], texts=[t.to_dict() for t in self.texts],
            tick_lines=[s.to_dict() for s in self.tick_lines], axis_lines=[s.to_dict() for s in self.axis_lines],
            marks=[m.to_dict() for m in self.marks], whiskers=[s.to_dict() for s in self.whiskers],
            hlines=[s.to_dict() for s in self.hlines], vlines=[s.to_dict() for s in self.vlines],
            data_segments=[s.to_dict() for s in self.data_segments]), indent=1)


# ----------------------------------------------------------------------------- extraction
def _hex(colour) -> str | None:
    if colour is None:
        return None
    r, g, b = (int(round(max(0.0, min(1.0, float(c))) * 255)) for c in tuple(colour)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _overlaps(r, clip) -> bool:
    """Rect overlap test that also accepts degenerate rects.

    `pymupdf.Rect.intersects` is False for empty rectangles, and a perfectly horizontal or vertical line —
    an axis, a tick, an error-bar stem — has exactly that shape, so using it would silently drop them.
    """
    return r.x0 <= clip.x1 and r.x1 >= clip.x0 and r.y0 <= clip.y1 and r.y1 >= clip.y0


def vector_candidates(paper: PaperRecord, fig: FigureRegion, margin_pt: float = 18.0) -> VectorScene:
    """Read the exact geometry of a figure region out of the PDF -> `VectorScene` in crop pixels.

    `margin_pt` widens the clip so tick labels printed just outside the detected figure box are still
    collected (they can land at negative crop coordinates, which is correct and harmless). Raster figures
    simply produce an (almost) empty scene.
    """
    doc = pymupdf.open(paper.source_path)
    try:
        page = doc[fig.page - 1]
        zoom = fig.crop_dpi / 72.0
        irect = (fig.bbox.rect() * pymupdf.Matrix(zoom, zoom)).irect
        origin = (irect.x0 / zoom, irect.y0 / zoom)
        scene = VectorScene(fig_id=fig.id, page=fig.page, crop_px_per_pt=zoom, origin_pt=origin)
        clip = pymupdf.Rect(fig.bbox.x0 - margin_pt, fig.bbox.y0 - margin_pt,
                            fig.bbox.x1 + margin_pt, fig.bbox.y1 + margin_pt)
        size_px = (float(irect.width), float(irect.height))
        _collect_text(page, clip, scene)
        _collect_drawings(page, clip, scene, size_px)
    finally:
        doc.close()
    return scene


def _collect_text(page: pymupdf.Page, clip: pymupdf.Rect, scene: VectorScene) -> None:
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        if not _overlaps(pymupdf.Rect(x0, y0, x1, y1), clip):
            continue
        bx0, by0 = scene.to_px(x0, y0)
        bx1, by1 = scene.to_px(x1, y1)
        label = TickLabel(text=text, value=parse_number(text), bbox=(bx0, by0, bx1, by1),
                          center=((bx0 + bx1) / 2.0, (by0 + by1) / 2.0), source="vector", confidence=1.0)
        (scene.tick_labels if label.value is not None else scene.texts).append(label)


def _collect_drawings(page: pymupdf.Page, clip: pymupdf.Rect, scene: VectorScene,
                      size_px: tuple[float, float]) -> None:
    big = max(size_px)
    axis_min = 0.2 * big                     # a line this long across the figure is an axis / reference line
    tick_max = max(6.0, 0.02 * big)          # a stroke this short touching an axis is a tick
    mark_max = max(12.0, 0.06 * min(size_px))
    area_px = size_px[0] * size_px[1]
    loose: list[list[Segment]] = []          # stroked segments, grouped by drawing (path)
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None or not _overlaps(rect, clip):
            continue
        fill = _hex(d.get("fill"))
        stroke = _hex(d.get("color"))
        width = float(d.get("width") or 0.0) * scene.crop_px_per_pt
        x0, y0 = scene.to_px(rect.x0, rect.y0)
        x1, y1 = scene.to_px(rect.x1, rect.y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if fill is not None and w * h <= 0.6 * area_px and (w > 0 or h > 0):
            scene.marks.append(Mark(x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0, w=w, h=h,
                                    kind=_mark_kind(d["items"], w, h, mark_max), fill=fill, stroke=stroke))
            continue
        segs = []
        for item in d["items"]:
            if item[0] == "l":
                segs.append(_segment(scene, item[1], item[2], width, stroke))
            elif item[0] == "re":
                r = item[1]
                corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                for i in range(4):
                    segs.append(_segment(scene, corners[i], corners[(i + 1) % 4], width, stroke))
            elif item[0] in ("c", "qu"):
                pts = [p for p in item[1:] if hasattr(p, "x")]
                if len(pts) >= 2:
                    segs.append(_segment(scene, pts[0], pts[-1], width, stroke))
        if segs:
            loose.append(segs)
    _classify(scene, loose, axis_min, tick_max)


def _segment(scene: VectorScene, p, q, width: float, colour: str | None) -> Segment:
    ax, ay = scene.to_px(p.x if hasattr(p, "x") else p[0], p.y if hasattr(p, "y") else p[1])
    bx, by = scene.to_px(q.x if hasattr(q, "x") else q[0], q.y if hasattr(q, "y") else q[1])
    return Segment(ax, ay, bx, by, width=width, colour=colour)


def _mark_kind(items, w: float, h: float, mark_max: float) -> str:
    ops = [it[0] for it in items]
    if w > mark_max or h > mark_max:
        return "rect" if ops.count("re") else "path"
    if ops and all(o == "c" for o in ops):
        return "circle"
    if ops == ["re"]:
        return "rect"
    if ops.count("l") in (2, 3) and "c" not in ops:
        return "triangle"
    if "l" in ops:
        return "polygon"
    return "path"


def _classify(scene: VectorScene, paths: list[list[Segment]], axis_min: float, tick_max: float) -> None:
    """Sort stroked segments into axes, ticks, error-bar stems, and everything else."""
    for segs in paths:                                            # axes first: ticks are defined by them
        for s in segs:
            if s.orientation in ("h", "v") and s.length >= axis_min:
                s.role = "axis"
                scene.axis_lines.append(s)
    for segs in paths:
        for s in segs:
            if s.role == "axis":
                continue
            if s.orientation in ("h", "v") and s.length <= tick_max and _on_axis(s, scene.axis_lines):
                s.role = "tick"
                scene.tick_lines.append(s)
        caps = [s for s in segs if s.role == "other" and s.orientation in ("h", "v") and s.length <= tick_max]
        for s in segs:
            if s.role != "other" or s.orientation not in ("h", "v"):
                continue
            end = _capped_end(s, caps)
            if end is None:
                continue
            if end == 0:                                          # keep the datum end first
                s.x0, s.y0, s.x1, s.y1 = s.x1, s.y1, s.x0, s.y0
            s.role = "whisker"
            scene.whiskers.append(s)
        for s in segs:
            if s.role != "other":
                continue
            if s.orientation == "h":
                scene.hlines.append(s)
            elif s.orientation == "v":
                scene.vlines.append(s)
            else:
                s.role = "data"
                scene.data_segments.append(s)


def _on_axis(seg: Segment, axes: list[Segment], tol: float = 2.0) -> bool:
    """True when one end of `seg` touches a perpendicular axis line."""
    for ax in axes:
        if ax.orientation == seg.orientation:
            continue
        if ax.orientation == "v":
            near_x = min(abs(seg.x0 - ax.x0), abs(seg.x1 - ax.x0)) <= tol
            within = min(ax.y0, ax.y1) - tol <= seg.y0 <= max(ax.y0, ax.y1) + tol
            if near_x and within:
                return True
        else:
            near_y = min(abs(seg.y0 - ax.y0), abs(seg.y1 - ax.y0)) <= tol
            within = min(ax.x0, ax.x1) - tol <= seg.x0 <= max(ax.x0, ax.x1) + tol
            if near_y and within:
                return True
    return False


def _capped_end(stem: Segment, caps: list[Segment], tol: float = 1.5) -> int | None:
    """Index of the end of `stem` that carries an error-bar cap (0 or 1), else None.

    A cap is a short perpendicular segment whose *midpoint* sits on the end of the stem; the corner of a
    box, where the perpendicular side meets end-to-end, is half a side away and does not qualify.
    """
    for cap in caps:
        if cap is stem or cap.orientation == stem.orientation:
            continue
        cx, cy = (cap.x0 + cap.x1) / 2.0, (cap.y0 + cap.y1) / 2.0
        for i, (ex, ey) in enumerate(((stem.x0, stem.y0), (stem.x1, stem.y1))):
            if math.hypot(cx - ex, cy - ey) <= max(tol, 0.25 * cap.length):
                return i
    return None


# ----------------------------------------------------------------------------- use
def snap_to_vector(scene: VectorScene, x_px: float, y_px: float, radius: float = 8.0) -> tuple[float, float] | None:
    """Snap an approximate pixel coordinate to the nearest exact mark centre, else whisker end.

    Marks win over whisker ends whenever any mark is within `radius`, because a vision model asked for "the
    mean" points at the marker and only lands on the cap when it is off by an error bar.
    """
    best = None
    for m in scene.marks:
        d = math.hypot(m.x - x_px, m.y - y_px)
        if d <= radius and (best is None or d < best[0]):
            best = (d, (m.x, m.y))
    if best is not None:
        return best[1]
    for s in scene.whiskers:
        for x, y in ((s.x1, s.y1), (s.x0, s.y0)):
            d = math.hypot(x - x_px, y - y_px)
            if d <= radius and (best is None or d < best[0]):
                best = (d, (x, y))
    return best[1] if best is not None else None


def calibrate_from_scene(scene: VectorScene, axis: Literal["x", "y"] = "y",
                         near: tuple[float, float] | None = None) -> AxisCalibration | None:
    """Calibrate one axis of a vector figure -> `AxisCalibration` in crop pixels, or None.

    Each axis line is calibrated on its own (multi-panel figures have one per panel) using the tick lines
    that touch it and the numeric labels beside it. The returned calibration is the one nearest `near` when
    given, otherwise the best supported (most ticks, then lowest rmse). Figures that draw no tick lines fall
    back to the label centres alone.
    """
    want = "v" if axis == "y" else "h"
    cands: list[tuple[AxisCalibration, Segment | None]] = []
    for ax in scene.axis_lines:
        if ax.orientation != want:
            continue
        ticks = [t for t in scene.tick_lines if _tick_of(t, ax, axis)]
        if len(ticks) < 2:
            continue
        coords = sorted((t.y0 if axis == "y" else t.x0) for t in ticks)
        labels = _labels_for(scene, ax, axis)
        pairs = pair_ticks(labels, coords, axis=axis)
        if len(pairs) < 2:
            continue
        try:
            cands.append((fit_axis(pairs, axis=axis), ax))
        except ValueError:
            continue
    if not cands:
        fallback = _fallback_from_labels(scene, axis, near)
        return fallback
    if near is not None:
        return min(cands, key=lambda c: _distance_to(c[1], near))[0]
    return max(cands, key=lambda c: (len(c[0].ticks), -c[0].rmse))[0]


def _tick_of(tick: Segment, ax: Segment, axis: str, tol: float = 2.0) -> bool:
    if axis == "y":
        if tick.orientation != "h":
            return False
        return (min(abs(tick.x0 - ax.x0), abs(tick.x1 - ax.x0)) <= tol
                and min(ax.y0, ax.y1) - tol <= tick.y0 <= max(ax.y0, ax.y1) + tol)
    if tick.orientation != "v":
        return False
    return (min(abs(tick.y0 - ax.y0), abs(tick.y1 - ax.y0)) <= tol
            and min(ax.x0, ax.x1) - tol <= tick.x0 <= max(ax.x0, ax.x1) + tol)


def _labels_for(scene: VectorScene, ax: Segment, axis: str, gutter: float = 0.35) -> list[TickLabel]:
    """Numeric labels that belong to one axis line: in its gutter (outside, within `gutter` x its length),
    and within its span."""
    span = ax.length
    out = []
    for t in scene.tick_labels:
        cx, cy = t.center
        if axis == "y":
            along, across = cy, ax.x0 - cx
            in_span = min(ax.y0, ax.y1) - 0.1 * span <= along <= max(ax.y0, ax.y1) + 0.1 * span
        else:
            along, across = cx, cy - ax.y0
            in_span = min(ax.x0, ax.x1) - 0.1 * span <= along <= max(ax.x0, ax.x1) + 0.1 * span
        if in_span and -0.05 * span <= across <= gutter * span:
            out.append(t)
    return out


def _distance_to(ax: Segment | None, point: tuple[float, float]) -> float:
    if ax is None:
        return float("inf")
    px, py = point
    x = min(max(px, min(ax.x0, ax.x1)), max(ax.x0, ax.x1))
    y = min(max(py, min(ax.y0, ax.y1)), max(ax.y0, ax.y1))
    return math.hypot(px - x, py - y)


def _fallback_from_labels(scene: VectorScene, axis: str, near: tuple[float, float] | None) -> AxisCalibration | None:
    """No tick lines: calibrate from the centres of one column (or row) of numeric labels."""
    labels = [t for t in scene.tick_labels if t.value is not None]
    if len(labels) < 2:
        return None
    across = np.array([t.center[0] if axis == "y" else t.center[1] for t in labels])
    order = np.argsort(across)
    clusters: list[list[int]] = []
    spread = max(8.0, 0.05 * (float(across.max() - across.min()) or 8.0))
    for i in order:
        if clusters and across[i] - across[clusters[-1][-1]] <= spread:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    if near is not None:
        pick = min(clusters, key=lambda c: min(math.hypot(labels[i].center[0] - near[0],
                                                          labels[i].center[1] - near[1]) for i in c))
    else:
        pick = max(clusters, key=len)
    pairs = pair_ticks([labels[i] for i in pick], [], axis=axis)
    if len(pairs) < 2:
        return None
    try:
        return fit_axis(pairs, axis=axis)
    except ValueError:
        return None
