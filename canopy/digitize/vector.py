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
    crop_px   = (pdf_pt - origin_pt) * zoom - 0.5

`origin_pt` is the *rounded* clip origin PyMuPDF actually used, so the mapping is exact rather than off by
the sub-pixel remainder of `bbox.x0 * zoom`.

**Half-pixel convention.** PDF device space counts pixel *edges*: device x = 0 is the left edge of column 0,
so a hairline drawn at device x = 100 straddles columns 99 and 100. Numpy arrays (and therefore
`canopy.digitize.cv`, and the pixel coordinates a vision model returns for an image) count pixel *indices*,
whose centres are at integer positions. This module normalises by the `- 0.5` above, so **every coordinate
in a `VectorScene` is an array index**, directly comparable with `snap_vertical_edge`/`snap_horizontal_edge`
output and with model-supplied pixel coordinates. `tests/test_digitize_vector.py` pins this against the
rendered raster.

PyMuPDF returns drawings and text in the page's displayed coordinate space (rotation already applied), the
same space the clip rectangle lives in, so rotated pages need no special case.
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
    """Everything a vector figure declares about itself, in crop pixels.

    NOTE ON `tick_labels`: this is *every numeric word in the clipped region* — a caption's "Fig. 3", an
    n = 12 annotation and a panel letter all land here. It is a candidate pool, **not** a list of axis
    ticks. `calibrate_from_scene` does the geometric filtering (labels in one axis' gutter, paired with
    that axis' tick lines, outliers rejected); consumers must go through it rather than reading this list
    as if every entry were a tick.
    """

    fig_id: str
    page: int
    crop_px_per_pt: float
    origin_pt: tuple[float, float]
    tick_labels: list[TickLabel] = field(default_factory=list)   # numeric words: CANDIDATE ticks only
    texts: list[TickLabel] = field(default_factory=list)         # other words: legend keys, axis titles
    tick_lines: list[Segment] = field(default_factory=list)
    axis_lines: list[Segment] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    whiskers: list[Segment] = field(default_factory=list)        # error-bar stems (datum end first)
    hlines: list[Segment] = field(default_factory=list)          # other horizontal strokes
    vlines: list[Segment] = field(default_factory=list)          # other vertical strokes
    data_segments: list[Segment] = field(default_factory=list)   # oblique strokes (connecting lines)
    warnings: list[str] = field(default_factory=list)            # what the classifier could not attribute

    def to_pt(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Crop pixels (array indices) -> PDF points on the page."""
        return (self.origin_pt[0] + (x_px + 0.5) / self.crop_px_per_pt,
                self.origin_pt[1] + (y_px + 0.5) / self.crop_px_per_pt)

    def to_px(self, x_pt: float, y_pt: float) -> tuple[float, float]:
        """PDF points -> crop pixels, as **array indices** (see the module docstring on the -0.5)."""
        return ((x_pt - self.origin_pt[0]) * self.crop_px_per_pt - 0.5,
                (y_pt - self.origin_pt[1]) * self.crop_px_per_pt - 0.5)

    def to_json(self) -> str:
        return json.dumps(dict(
            fig_id=self.fig_id, page=self.page, crop_px_per_pt=self.crop_px_per_pt, origin_pt=list(self.origin_pt),
            tick_labels=[t.to_dict() for t in self.tick_labels], texts=[t.to_dict() for t in self.texts],
            tick_lines=[s.to_dict() for s in self.tick_lines], axis_lines=[s.to_dict() for s in self.axis_lines],
            marks=[m.to_dict() for m in self.marks], whiskers=[s.to_dict() for s in self.whiskers],
            hlines=[s.to_dict() for s in self.hlines], vlines=[s.to_dict() for s in self.vlines],
            data_segments=[s.to_dict() for s in self.data_segments], warnings=list(self.warnings)), indent=1)


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
    sizes = _span_sizes(page)
    exponents = 0
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if not _overlaps(rect, clip):
            continue
        bx0, by0 = scene.to_px(x0, y0)
        bx1, by1 = scene.to_px(x1, y1)
        value = parse_number(text)
        if value is not None and _is_superscript_composite(rect, sizes):
            value = None                          # "10^3" reaches the text layer as "103": not a number
            exponents += 1
        label = TickLabel(text=text, value=value, bbox=(bx0, by0, bx1, by1),
                          center=((bx0 + bx1) / 2.0, (by0 + by1) / 2.0), source="vector", confidence=1.0)
        (scene.tick_labels if label.value is not None else scene.texts).append(label)
    if exponents:
        scene.warnings.append(
            f"{exponents} label(s) mix font sizes (superscript exponents such as 10^3, which the PDF text "
            f"layer flattens to '103'): their values are unusable, read this axis from the image instead")


def _span_sizes(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, float]]:
    """(rect, font size) of every text span on the page, for spotting superscripts."""
    out = []
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                out.append((pymupdf.Rect(span["bbox"]), float(span.get("size") or 0.0)))
    return out


def _is_superscript_composite(word: pymupdf.Rect, sizes: list[tuple[pymupdf.Rect, float]],
                              ratio: float = 1.15) -> bool:
    """True when one 'word' is really a base plus a raised, smaller exponent (10^3 -> "103").

    A linear fit through 10^0..10^5 read as 100..105 is *perfect* and completely wrong, so a mixed-size
    label must never be used as a tick value.
    """
    found = []
    for rect, size in sizes:
        if size <= 0:
            continue
        inter = rect & word
        if inter.is_empty or rect.get_area() <= 0:
            continue
        if inter.get_area() >= 0.5 * rect.get_area():
            found.append(size)
    return bool(found) and max(found) / min(found) >= ratio


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
        if fill is not None and _is_datum_fill(fill, w, h, area_px):
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
        segs = [seg for seg in segs if seg.length >= 1.0]     # drop sub-pixel chords (flattened curves)
        if segs:
            loose.append(segs)
    _classify(scene, loose, axis_min, tick_max)


def _is_datum_fill(fill: str, w: float, h: float, area_px: float) -> bool:
    """Is this filled path a plotted datum rather than line art or a background panel?

    Writers emit plain strokes as fill+stroke paths, so a "filled" shape with no area (a tick mark, an
    error-bar cap) is a line and must be classified as one. Large pale fills are panels, not data.
    """
    if w < 1.0 or h < 1.0:                         # degenerate: a stroked line, not a shape
        return False
    area = w * h
    if area > 0.6 * area_px:
        return False
    return not (area > 0.05 * area_px and fill in ("#ffffff", "#fefefe"))


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
    """Sort stroked segments into axes, ticks, error-bar stems, and everything else.

    Classification is figure-wide, not per path: matplotlib (and most plotting libraries) emit a stem and
    each of its caps as separate `get_drawings()` entries, so pairing within one path finds nothing.
    """
    segments = [s for segs in paths for s in segs]
    for s in segments:                                            # axes first: ticks are defined by them
        if s.orientation in ("h", "v") and s.length >= axis_min:
            s.role = "axis"
            scene.axis_lines.append(s)
    for s in segments:
        if s.role == "other" and s.orientation in ("h", "v") and s.length <= tick_max \
                and _on_axis(s, scene.axis_lines):
            s.role = "tick"
            scene.tick_lines.append(s)

    straight = [s for s in segments if s.role == "other" and s.orientation in ("h", "v")]
    caps, cell = _spatial_index(s for s in straight if s.length <= 3.0 * tick_max)
    for s in sorted(straight, key=lambda seg: -seg.length):        # longest first: a stem claims its cap
        if s.role != "other":
            continue
        found = _capped_end(s, caps, cell)
        if found is None:
            continue
        end, cap = found
        if end == 0:                                              # keep the datum end first
            s.x0, s.y0, s.x1, s.y1 = s.x1, s.y1, s.x0, s.y0
        s.role = "whisker"
        cap.role = "cap"                                          # consumed: a cap is not itself a stem
        scene.whiskers.append(s)
    _mark_anchored_whiskers(scene, [s for s in straight if s.role == "other"])
    _drop_minority_orientation(scene)

    for s in segments:
        if s.role not in ("other", "cap"):
            continue
        if s.orientation == "h":
            scene.hlines.append(s)
        elif s.orientation == "v":
            scene.vlines.append(s)
        else:
            s.role = "data"
            scene.data_segments.append(s)
    _warn_unattributed(scene, tick_max)


def _drop_minority_orientation(scene: VectorScene, dominance: float = 0.8) -> None:
    """Error bars in one figure share an orientation; drop the odd ones out.

    A data segment that happens to run flat and end on a marker can pick up that marker's (short) error bar
    as if it were its cap. Such accidents are always a small minority against the real family, so when one
    orientation holds at least `dominance` of the whiskers the rest are demoted. A chart with genuine x and
    y error bars splits about 50/50 and keeps both; a figure with a handful of real error bars perpendicular
    to a large family would lose them (noted, not seen in practice).
    """
    if len(scene.whiskers) < 4:
        return
    counts = {"h": 0, "v": 0}
    for s in scene.whiskers:
        counts[s.orientation] += 1
    total = sum(counts.values())
    for orientation, n in counts.items():
        if n and n / total < 1.0 - dominance:
            for s in [w for w in scene.whiskers if w.orientation == orientation]:
                s.role = "other"
                scene.whiskers.remove(s)


def _mark_anchored_whiskers(scene: VectorScene, candidates: list[Segment], min_family: int = 3) -> None:
    """Error bars drawn without caps: a stroke anchored to a plotted mark, in a family of at least three.

    matplotlib's default `capsize` is 0, so the only thing tying a stem to its datum is the marker it runs
    through. The family rule keeps a legend key (one or two lines through a sample marker) out.
    """
    if not scene.marks:
        return
    by_orientation: dict[str, list[Segment]] = {}
    for s in candidates:
        if _anchor_mark(s, scene.marks) is not None:
            by_orientation.setdefault(s.orientation, []).append(s)
    for family in by_orientation.values():
        if len(family) < min_family:
            continue
        for s in family:
            s.role = "whisker"
            scene.whiskers.append(s)


def _anchor_mark(seg: Segment, marks: list[Mark], tol: float | None = None) -> "Mark | None":
    """The plotted mark this stroke runs through or ends on, if any."""
    for m in marks:
        t = tol if tol is not None else max(2.0, 0.5 * min(m.w, m.h))
        if seg.orientation == "v":
            across, along, lo, hi = abs(m.x - seg.x0), m.y, min(seg.y0, seg.y1), max(seg.y0, seg.y1)
        else:
            across, along, lo, hi = abs(m.y - seg.y0), m.x, min(seg.x0, seg.x1), max(seg.x0, seg.x1)
        if across <= t and lo - t <= along <= hi + t:
            return m
    return None


def _warn_unattributed(scene: VectorScene, tick_max: float) -> None:
    short = [s for s in scene.vlines + scene.hlines if s.length <= 3.0 * tick_max]
    stems = [s for s in scene.vlines + scene.hlines if s.length > 3.0 * tick_max]
    if not scene.whiskers and stems and (short or scene.marks):
        scene.warnings.append(
            f"no error bar could be attributed: {len(stems)} unclassified straight strokes and "
            f"{len(short)} short perpendicular ones, but no caps or markers to tie them to")


def _spatial_index(segments, cell: float = 24.0) -> tuple[dict[tuple[int, int], list[Segment]], float]:
    """Bucket segments by the cell of their midpoint, so cap lookup stays linear in the segment count."""
    index: dict[tuple[int, int], list[Segment]] = {}
    for s in segments:
        cx, cy = (s.x0 + s.x1) / 2.0, (s.y0 + s.y1) / 2.0
        index.setdefault((int(cx // cell), int(cy // cell)), []).append(s)
    return index, cell


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


def _capped_end(stem: Segment, caps: dict[tuple[int, int], list[Segment]], cell: float,
                tol: float = 1.5) -> tuple[int, Segment] | None:
    """The end of `stem` that carries an error-bar cap -> (end index 0|1, the cap), else None.

    A cap is a short perpendicular segment whose *midpoint* sits on the end of the stem, drawn with the same
    pen (same stroke width — stem and caps come from one plotting call). The corner of a box, where the
    perpendicular side meets end-to-end, is half a side away and does not qualify. `caps` is the spatial
    index built by `_spatial_index`, because stem and cap are usually separate `get_drawings()` entries.
    """
    for i, (ex, ey) in enumerate(((stem.x0, stem.y0), (stem.x1, stem.y1))):
        gx, gy = int(ex // cell), int(ey // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cap in caps.get((gx + dx, gy + dy), ()):
                    if cap is stem or cap.role != "other" or cap.orientation == stem.orientation:
                        continue
                    if 1.5 * cap.length > stem.length:            # a cap is small across a longer stem
                        continue
                    if abs(cap.width - stem.width) > 0.5 * max(cap.width, stem.width, 1.0):
                        continue                                  # different pen: not the same error bar
                    cx, cy = (cap.x0 + cap.x1) / 2.0, (cap.y0 + cap.y1) / 2.0
                    if math.hypot(cx - ex, cy - ey) <= max(tol, 0.5 * cap.width):
                        return i, cap
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


def whisker_ends(scene: VectorScene, mark, axis: Literal["x", "y"] = "y",
                 tol: float | None = None) -> tuple[float, float] | None:
    """Extent of the error bar attached to a mark -> (low, high) crop pixels, or None.

    Mirrors `canopy.digitize.cv.detect_whiskers`: for a y axis the pair is (top row, bottom row). Figures
    that draw one stem per direction (from the datum outwards) and figures that draw a single stem through
    the datum both work, because the extremes of every anchored stroke are unioned.
    """
    if isinstance(mark, Mark):
        probe = mark
    else:
        probe = Mark(x=float(mark[0]), y=float(mark[1]), w=2.0, h=2.0, kind="path")
    want = "v" if axis == "y" else "h"
    ends: list[float] = []
    for s in scene.whiskers:
        if s.orientation != want or _anchor_mark(s, [probe], tol) is None:
            continue
        ends.extend((s.y0, s.y1) if want == "v" else (s.x0, s.x1))
    if not ends:
        return None
    return min(ends), max(ends)


def calibrate_from_scene(scene: VectorScene, axis: Literal["x", "y"] = "y",
                         near: tuple[float, float] | None = None,
                         scale: Literal["linear", "log", "auto"] = "auto") -> AxisCalibration | None:
    """Calibrate one axis of a vector figure -> `AxisCalibration` in crop pixels, or None.

    Each axis line is calibrated on its own (multi-panel figures have one per panel) using the tick lines
    that touch it and the numeric labels beside it. The returned calibration is the one nearest `near` when
    given, otherwise the best supported (most ticks, then lowest rmse). Figures that draw no tick lines fall
    back to the label centres alone.

    `scale="auto"` (the default) fits both a linear and a log axis and keeps whichever reproduces the tick
    positions better, so a log-scaled figure is not silently read as linear; pass "linear"/"log" to force.
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
        cal = _best_fit(pairs, axis, scale)
        if cal is not None:
            cands.append((cal, ax))
    if not cands:
        return _fallback_from_labels(scene, axis, near, scale)
    if near is not None:
        return min(cands, key=lambda c: _distance_to(c[1], near))[0]
    return max(cands, key=lambda c: (len(c[0].ticks), -c[0].rmse))[0]


def _best_fit(pairs: list[tuple[float, float]], axis: str, scale: str) -> AxisCalibration | None:
    """Fit the requested scale, or try both and keep the one that reproduces the ticks best."""
    wanted = ("linear", "log") if scale == "auto" else (scale,)
    best: AxisCalibration | None = None
    for sc in wanted:
        try:
            cal = fit_axis(pairs, scale=sc, axis=axis)
        except ValueError:
            continue
        if best is None or (len(cal.ticks) >= 3 and cal.rmse < best.rmse):
            best = cal
    return best


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


def _fallback_from_labels(scene: VectorScene, axis: str, near: tuple[float, float] | None,
                          scale: str = "auto") -> AxisCalibration | None:
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
    return _best_fit(pairs, axis, scale)
