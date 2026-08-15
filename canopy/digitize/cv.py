"""Raster computer vision for figure digitizing: axes, ticks, tick-label OCR, sub-pixel snapping,
bar/marker/whisker detection.

Everything works in **image pixels of the given PNG** (origin top-left, y downward); callers turn pixels
into data values with `canopy.digitize.calibrate`. Nothing here calls an LLM: these are the deterministic
measurements a vision model's read-out is snapped to and checked against.

The two roles of this module:
  * *shared core* — `find_axes` / `find_tick_marks` / `ocr_tick_labels` give the calibration for any route;
  * *path B* — `detect_bars` / `detect_markers` / `detect_whiskers` read the marks directly, as an
    independent voter alongside the vision-model routes.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibrate import TickLabel, parse_number

ImageLike = str | Path | np.ndarray      # a PNG path or an already-loaded array

MIN_BAR_WIDTH_PX = 8          # amendment F: bars narrower than this are flagged (unreliable read-out)
_MIN_INK = 8.0                # darkness (0..255) below which a patch counts as blank
NUMERIC_CHARS = "0123456789.,-+eE"      # tesseract whitelist used as an OCR fallback for tick labels


# ----------------------------------------------------------------------------- dataclasses
@dataclass
class Axes:
    """Detected plot frame. Positions are pixel coordinates in the image given to `find_axes`."""

    y_axis_x: float | None                            # column of the vertical axis line
    x_axis_y: float | None                            # row of the horizontal axis line
    y_axis_span: tuple[float, float] | None           # (y0, y1) rows covered by the vertical axis
    x_axis_span: tuple[float, float] | None           # (x0, x1) columns covered by the horizontal axis
    y_axis_width: float                               # stroke width of the vertical axis (scan lines), px
    x_axis_width: float                               # stroke width of the horizontal axis (scan lines), px
    plot_bbox: tuple[float, float, float, float]      # x0, y0, x1, y1 of the plotting area
    confidence: float                                 # 1.0 both axes, 0.6 one, 0.0 none

    def to_dict(self) -> dict:
        return dict(y_axis_x=self.y_axis_x, x_axis_y=self.x_axis_y,
                    y_axis_span=list(self.y_axis_span) if self.y_axis_span else None,
                    x_axis_span=list(self.x_axis_span) if self.x_axis_span else None,
                    y_axis_width=self.y_axis_width, x_axis_width=self.x_axis_width,
                    plot_bbox=list(self.plot_bbox), confidence=self.confidence)


@dataclass
class Bar:
    """One bar of a bar chart, in image pixels."""

    x_center: float
    x0: float
    x1: float
    top_y: float                                      # sub-pixel row of the bar's free end
    base_y: float                                     # row of the baseline it grows from
    colour: str                                       # "#rrggbb" of the fill
    narrow: bool = False                              # width < MIN_BAR_WIDTH_PX -> read-out unreliable

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def to_dict(self) -> dict:
        return dict(x_center=self.x_center, x0=self.x0, x1=self.x1, top_y=self.top_y, base_y=self.base_y,
                    colour=self.colour, narrow=self.narrow)


@dataclass
class Marker:
    """A plotted point (scatter/line marker), in image pixels."""

    x: float
    y: float
    colour: str                                       # "#rrggbb"
    kind: str                                         # circle | square | triangle | open | blob
    size: float = 0.0                                 # mean of the marker's width and height, px

    def to_dict(self) -> dict:
        return dict(x=self.x, y=self.y, colour=self.colour, kind=self.kind, size=self.size)


# ----------------------------------------------------------------------------- image helpers
def load_gray(png: ImageLike) -> np.ndarray:
    """Load a figure PNG as a uint8 grayscale array."""
    return _as_gray(png)


def load_color(png: ImageLike) -> np.ndarray:
    """Load a figure PNG as a uint8 BGR array (OpenCV channel order)."""
    return _as_bgr(png)


def _read(path: ImageLike, flags: int) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def _as_gray(img: ImageLike) -> np.ndarray:
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _read(img, cv2.IMREAD_GRAYSCALE)


def _as_bgr(img: ImageLike) -> np.ndarray:
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(_as_gray(img), cv2.COLOR_GRAY2BGR)
    return _read(img, cv2.IMREAD_COLOR)


def _threshold(gray: np.ndarray) -> float:
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(min(max(thr, 60.0), 220.0))


def _dark(gray: np.ndarray, thr: float | None = None) -> np.ndarray:
    return gray < (_threshold(gray) if thr is None else thr)


def _hex(bgr) -> str:
    b, g, r = (int(round(float(c))) for c in bgr[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _runs(v: np.ndarray) -> tuple[int, int, int]:
    """Longest run of True in a 1-D boolean array -> (length, start, end_exclusive)."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], v.astype(np.int8), [0]))))
    if idx.size == 0:
        return 0, 0, 0
    starts, ends = idx[0::2], idx[1::2]
    k = int(np.argmax(ends - starts))
    return int(ends[k] - starts[k]), int(starts[k]), int(ends[k])


def _groups(idx: np.ndarray, gap: int = 2) -> list[list[int]]:
    """Split sorted indices into groups separated by more than `gap`."""
    out: list[list[int]] = []
    for i in idx.tolist():
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def snap_window_for(width_px: float) -> int:
    """Amendment F snap window: a quarter of the mark's width, never below 3 px."""
    return int(max(3, round(0.25 * float(width_px))))


# ----------------------------------------------------------------------------- axes and ticks
def find_axes(img: ImageLike, min_frac: float = 0.4) -> Axes:
    """Find the axis lines (long dark straight runs) and the plotting area.

    Vertical/horizontal candidates are columns/rows whose longest dark run covers at least `min_frac` of
    the inked area; the *leftmost* of the longest vertical group is taken as the y axis and the *bottom-most*
    of the longest horizontal group as the x axis (the usual convention; a missing axis stays None and the
    plot area then falls back to the other axis' extent).
    """
    gray = _as_gray(img)
    h, w = gray.shape
    dark = _dark(gray)
    if not dark.any():
        return Axes(None, None, None, None, 0.0, 0.0, (0.0, 0.0, float(w), float(h)), 0.0)
    rows_ink = np.flatnonzero(dark.any(axis=1))
    cols_ink = np.flatnonzero(dark.any(axis=0))
    ink = (float(cols_ink[0]), float(rows_ink[0]), float(cols_ink[-1] + 1), float(rows_ink[-1] + 1))
    ink_h, ink_w = ink[3] - ink[1], ink[2] - ink[0]
    weight = (255.0 - gray.astype(np.float32)) * dark

    v = [_runs(dark[:, c]) for c in range(w)]
    hh = [_runs(dark[r, :]) for r in range(h)]
    y_axis = _pick_line([r[0] for r in v], min_frac * ink_h, weight.sum(axis=0), v, prefer="first")
    x_axis = _pick_line([r[0] for r in hh], min_frac * ink_w, weight.sum(axis=1), hh, prefer="last")

    y_x = y_axis[0] if y_axis else None
    y_span = (float(y_axis[1]), float(y_axis[2])) if y_axis else None
    y_wid = y_axis[3] if y_axis else 0.0
    x_y = x_axis[0] if x_axis else None
    x_span = (float(x_axis[1]), float(x_axis[2])) if x_axis else None
    x_wid = x_axis[3] if x_axis else 0.0
    # the vertical axis fixes the left edge and (with the horizontal one) the vertical extent; a horizontal
    # axis may be an interior zero line, so it only ever extends the plot area, never shrinks it
    bbox = (y_x if y_x is not None else (x_span[0] if x_span else ink[0]),
            y_span[0] if y_span else (x_y if x_y is not None else ink[1]),
            max(x_span[1], y_x) if (x_span and y_x is not None) else (x_span[1] if x_span else ink[2]),
            max(y_span[1], x_y) if (y_span and x_y is not None) else (y_span[1] if y_span else
                                                                     (x_y if x_y is not None else ink[3])))
    conf = 1.0 if (y_x is not None and x_y is not None) else (0.6 if (y_x is not None or x_y is not None) else 0.0)
    return Axes(y_x, x_y, y_span, x_span, y_wid, x_wid, tuple(float(v_) for v_ in bbox), conf)


def _pick_line(run_lens: list[int], min_len: float, weights: np.ndarray, runs: list[tuple[int, int, int]],
               prefer: str) -> tuple[float, float, float, float] | None:
    """Choose one axis line from candidate rows/columns -> (centre, span0, span1, stroke width)."""
    lens = np.asarray(run_lens, dtype=float)
    cand = np.flatnonzero(lens >= max(min_len, 3.0))
    if cand.size == 0:
        return None
    best = float(lens[cand].max())
    groups = [g for g in _groups(cand) if max(lens[i] for i in g) >= 0.8 * best]
    if not groups:
        return None
    g = groups[0] if prefer == "first" else groups[-1]
    ws = np.array([max(weights[i], 1e-6) for i in g])
    centre = float(np.average(np.asarray(g, dtype=float), weights=ws))
    span0 = float(min(runs[i][1] for i in g))
    span1 = float(max(runs[i][2] for i in g))
    return centre, span0, span1, float(len(g))


def find_tick_marks(img: ImageLike, axes: Axes, max_len: float | None = None) -> dict[str, list[float]]:
    """Find tick marks along the detected axes -> {"left": rows, "bottom": columns} (sub-pixel centres).

    A tick is a short dark run that starts at the axis line and stops well before `max_len` (default 2% of
    the plot size); the run length of every scan line is compared with the median run of the whole axis, so
    an anti-aliased halo along the axis does not masquerade as ticks. Ticks pointing outwards and inwards
    are both tried and the side with more ticks wins.
    """
    gray = _as_gray(img)
    dark = _dark(gray)
    x0, y0, x1, y1 = axes.plot_bbox
    if max_len is None:
        max_len = max(4.0, 0.02 * max(x1 - x0, y1 - y0))
    out: dict[str, list[float]] = {"left": [], "bottom": []}
    if axes.y_axis_x is not None and axes.y_axis_span is not None:
        rows = np.arange(int(max(0, axes.y_axis_span[0] - 2)), int(min(gray.shape[0], axes.y_axis_span[1] + 2)))
        band = axes.y_axis_width / 2.0 + 1.0
        outward = _ticks_along(dark[rows, :], axes.y_axis_x - band, max_len, -1)
        inward = _ticks_along(dark[rows, :], axes.y_axis_x + band, max_len, +1)
        best = outward if len(outward) >= len(inward) else inward
        out["left"] = [float(rows[0] + c) for c in best]
    if axes.x_axis_y is not None and axes.x_axis_span is not None:
        cols = np.arange(int(max(0, axes.x_axis_span[0] - 2)), int(min(gray.shape[1], axes.x_axis_span[1] + 2)))
        band = axes.x_axis_width / 2.0 + 1.0
        sub = dark[:, cols].T                                   # scan lines along the axis
        outward = _ticks_along(sub, axes.x_axis_y + band, max_len, +1)
        inward = _ticks_along(sub, axes.x_axis_y - band, max_len, -1)
        best = outward if len(outward) >= len(inward) else inward
        out["bottom"] = [float(cols[0] + c) for c in best]
    return out


def _ticks_along(scan: np.ndarray, start: float, max_len: float, direction: int) -> list[float]:
    """Tick centres (index into `scan`'s first axis) for runs leaving the axis at `start`.

    `scan` is a boolean array of shape (n_scanlines, n_pixels): each row is one line perpendicular to the
    axis. `direction` is -1 when ticks point towards lower pixel indices (left/up) and +1 otherwise.
    """
    n, width = scan.shape
    limit = int(np.ceil(max_len)) + 1
    s = int(round(start))
    if direction < 0:
        lo, hi = max(0, s - limit + 1), max(0, s + 1)
        strip = scan[:, lo:hi][:, ::-1]
    else:
        lo, hi = min(width, s), min(width, s + limit)
        strip = scan[:, lo:hi]
    if strip.shape[1] < 2:
        return []
    run = np.argmin(np.cumprod(strip.astype(np.int8), axis=1), axis=1).astype(float)
    run[strip.all(axis=1)] = strip.shape[1]                      # never terminated -> not a tick
    base = float(np.median(run))
    excess = run - base
    hits = np.flatnonzero((excess >= 2.0) & (run <= max_len + 1) & (run < strip.shape[1]))
    if hits.size == 0:
        return []
    max_thick = max(4.0, 0.02 * n)
    centres = []
    for g in _groups(hits, gap=1):
        if (g[-1] - g[0] + 1) > max_thick:
            continue
        wts = excess[g]
        if wts.sum() <= 0:
            continue
        centres.append(float(np.average(np.asarray(g, dtype=float), weights=wts)))
    return centres


# ----------------------------------------------------------------------------- OCR of tick labels
def tesseract_path() -> str | None:
    """Path to the tesseract CLI (`CANOPY_TESSERACT` overrides), or None when it is not installed."""
    return os.environ.get("CANOPY_TESSERACT") or shutil.which("tesseract")


@dataclass
class OcrResult:
    """Outcome of one tesseract call: the word boxes and *why* they may be empty.

    `status` lets a caller record "OCR unavailable" in provenance instead of silently reporting no ticks.
    """

    words: list[dict]
    status: str = "ok"                                # ok | missing | failed | timeout

    def __bool__(self) -> bool:
        return bool(self.words)


class TickLabels(list):
    """`list[TickLabel]` that also carries the OCR `status` of the run that produced it."""

    status: str = "ok"


def run_tesseract(png: str | Path, psm: int = 11, lang: str = "eng", whitelist: str | None = None) -> OcrResult:
    """Run tesseract on a PNG -> `OcrResult` (word boxes + status; never raises)."""
    exe = tesseract_path()
    if not exe:
        return OcrResult([], "missing")
    cmd = [exe, str(png), "stdout", "--psm", str(psm), "-l", lang]
    if whitelist:
        cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
    cmd += ["tsv"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return OcrResult([], "timeout")
    except (OSError, subprocess.SubprocessError):
        return OcrResult([], "failed")
    if proc.returncode != 0:
        return OcrResult([], "failed")
    if not proc.stdout.strip():
        return OcrResult([], "ok")
    words = []
    for row in csv.DictReader(proc.stdout.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            words.append(dict(text=text, left=float(row["left"]), top=float(row["top"]),
                              width=float(row["width"]), height=float(row["height"]),
                              conf=float(row["conf"]), line=(row["block_num"], row["par_num"], row["line_num"])))
        except (KeyError, TypeError, ValueError):
            continue
    return OcrResult(words, "ok")


def ocr_tick_labels(img: ImageLike, axes: Axes, side: str = "left", upscale: int = 4, psm: int = 13,
                    ticks: list[float] | None = None) -> list[TickLabel]:
    """OCR the tick labels in the gutter beside an axis (amendment F: upscale 4x before OCR).

    `side` is "left" (labels left of the y axis) or "bottom" (labels below the x axis). The gutter is
    segmented into connected components first: components far from the axis or much taller than the rest
    (a rotated axis title, a legend) are dropped, the remainder are grouped into one label per tick (using
    `ticks` from `find_tick_marks` when given, otherwise by text line) and each label is OCR'd on its own
    tight crop. Layout modes are tried in order (raw line, single line, then a digits-only whitelist) until
    the text parses as a number, and a leading dash-shaped component sets the sign even when OCR drops it —
    on scanned figures tesseract loses minus signs and reads "0" as ")", and a sign error is not recoverable
    downstream.

    Returns TickLabels in the coordinates of `img`; `value` is None when the text is not numeric.
    """
    gray = _as_gray(img)
    h, w = gray.shape
    x0, y0, x1, y1 = axes.plot_bbox
    tick_len = max(4.0, 0.02 * max(x1 - x0, y1 - y0))
    if side == "left":
        if axes.y_axis_x is None:
            return TickLabels()
        gx1 = int(max(0, axes.y_axis_x - axes.y_axis_width / 2.0 - tick_len))
        gx0 = int(max(0, gx1 - max(60.0, 0.35 * (x1 - x0))))
        pad = 0.04 * (y1 - y0)
        gy0, gy1 = int(max(0, y0 - pad)), int(min(h, y1 + pad))
    elif side == "bottom":
        if axes.x_axis_y is None:
            return TickLabels()
        gy0 = int(min(h, axes.x_axis_y + axes.x_axis_width / 2.0 + tick_len))
        gy1 = int(min(h, gy0 + max(40.0, 0.25 * (y1 - y0))))
        pad = 0.04 * (x1 - x0)
        gx0, gx1 = int(max(0, x0 - pad)), int(min(w, x1 + pad))
    else:
        raise ValueError(f"side must be 'left' or 'bottom', got {side!r}")
    if gx1 - gx0 < 4 or gy1 - gy0 < 4:
        return TickLabels()

    crop = gray[gy0:gy1, gx0:gx1]
    out = TickLabels()
    statuses: list[str] = []
    for comps in _label_groups(crop, side, ticks, offset=(gx0, gy0)):
        bx0 = float(min(c[0] for c in comps) + gx0)
        by0 = float(min(c[1] for c in comps) + gy0)
        bx1 = float(max(c[0] + c[2] for c in comps) + gx0)
        by1 = float(max(c[1] + c[3] for c in comps) + gy0)
        pad_px = max(2, int(0.15 * (by1 - by0)))
        sub = gray[max(0, int(by0) - pad_px):int(by1) + pad_px, max(0, int(bx0) - pad_px):int(bx1) + pad_px]
        if sub.size == 0:
            continue
        text, conf, status = _ocr_line(sub, upscale, psm)
        statuses.append(status)
        if not text:
            continue
        value = parse_number(text)
        if value is not None and value > 0 and _has_leading_minus(comps):
            text, value = "-" + text.lstrip("-"), -value
        out.append(TickLabel(text=text, value=value, bbox=(bx0, by0, bx1, by1),
                             center=((bx0 + bx1) / 2.0, (by0 + by1) / 2.0), source="ocr",
                             confidence=max(0.0, min(1.0, conf / 100.0))))
    for bad in ("missing", "timeout", "failed"):
        if bad in statuses:
            out.status = bad
            break
    else:
        out.status = "failed" if (statuses and not out) else "ok"
    return out


def _has_leading_minus(comps: list[tuple[int, int, int, int]]) -> bool:
    """True when the leftmost component of a label is a dash: wide, flat and clear of the digits."""
    if len(comps) < 2:
        return False
    first, *rest = sorted(comps, key=lambda c: c[0])
    x, y, w, h = first
    body_h = float(np.median([c[3] for c in rest]))
    return (w >= 1.5 * h and h <= 0.55 * body_h and x + w <= min(c[0] for c in rest) + 1
            and y > min(c[1] for c in rest))


def _ocr_line(patch: np.ndarray, upscale: int, psm: int) -> tuple[str, float, str]:
    """OCR one tight single-line crop -> (text, tesseract confidence 0..100, status).

    Tries the requested page-segmentation mode, then single-line, then a digits-only whitelist, and returns
    the first reading that parses as a number; when no mode yields a number (a categorical label such as
    "young"), the most confident reading wins.
    """
    big = cv2.resize(patch, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=int(np.percentile(patch, 95)))
    modes = [(psm, None), (7, None), (7, NUMERIC_CHARS), (13, NUMERIC_CHARS)]
    others: list[tuple[float, str]] = []
    status = "ok"
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "label.png"
        cv2.imwrite(str(png), big)
        for mode, whitelist in dict.fromkeys(modes):
            res = run_tesseract(png, psm=mode, whitelist=whitelist)
            if res.status != "ok":
                return "", 0.0, res.status
            if not res.words:
                continue
            words = sorted(res.words, key=lambda ww: ww["left"])
            text = "".join(ww["text"] for ww in words)
            conf = min(ww["conf"] for ww in words)
            if parse_number(text) is not None:
                return text, conf, status
            others.append((conf, text))
    if not others:
        return "", 0.0, status
    conf, text = max(others)
    return text, conf, status


def _label_groups(crop: np.ndarray, side: str, ticks: list[float] | None,
                  offset: tuple[int, int] = (0, 0)) -> list[list[tuple[int, int, int, int]]]:
    """Group the gutter's ink into one list of glyph components per tick label (gutter coordinates)."""
    gx0, gy0 = offset
    ink = _dark(crop).astype(np.uint8)
    if not ink.any():
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps = [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]),
              int(stats[i, cv2.CC_STAT_HEIGHT])) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3]
    if not comps:
        return []
    med_h = float(np.median([c[3] for c in comps]))
    comps = [c for c in comps if c[3] <= max(2.5 * med_h, med_h + 4) and c[2] <= 0.9 * crop.shape[1]]
    if not comps:
        return []
    med_h = float(np.median([c[3] for c in comps]))
    comps = _band_nearest_axis(comps, side, crop.shape, gap_tol=max(4.0, 0.35 * med_h))
    if not comps:
        return []
    groups = _group_glyphs(comps, side, med_h)
    if not ticks:
        return groups
    return _assign_groups_to_ticks(groups, ticks, side, med_h, offset)


def _group_glyphs(comps: list[tuple[int, int, int, int]], side: str,
                  med_h: float) -> list[list[tuple[int, int, int, int]]]:
    """Glyph components -> one group per label, by the whitespace between them along the axis.

    Grouping happens *before* any tick assignment: the outer digit of "-100000" is nearer to the
    neighbouring tick than to its own, so assigning glyph by glyph silently truncates wide labels.
    """
    if side == "left":                                   # labels stack: glyphs of one label share rows
        start, end, tol = (lambda c: c[1]), (lambda c: c[1] + c[3]), 0.25 * med_h
    else:                                                # labels sit side by side: glyphs nearly touch
        start, end, tol = (lambda c: c[0]), (lambda c: c[0] + c[2]), max(2.0, 0.45 * med_h)
    groups: list[list[tuple[int, int, int, int]]] = []
    edge = 0.0
    for c in sorted(comps, key=start):
        if groups and start(c) - edge <= tol:
            groups[-1].append(c)
            edge = max(edge, end(c))
        else:
            groups.append([c])
            edge = end(c)
    return groups


def _assign_groups_to_ticks(groups: list[list[tuple[int, int, int, int]]], ticks: list[float], side: str,
                            med_h: float, offset: tuple[int, int]) -> list[list[tuple[int, int, int, int]]]:
    """Keep the label group nearest each tick (groups spanning two ticks are split glyph by glyph)."""
    origin = offset[1] if side == "left" else offset[0]   # ticks are image coordinates, comps gutter ones
    span = float(np.median(np.diff(sorted(ticks)))) if len(ticks) > 1 else 4.0 * med_h

    def centre(g):
        if side == "left":
            return (min(c[1] for c in g) + max(c[1] + c[3] for c in g)) / 2.0 + origin
        return (min(c[0] for c in g) + max(c[0] + c[2] for c in g)) / 2.0 + origin

    def extent(g):
        if side == "left":
            return max(c[1] + c[3] for c in g) - min(c[1] for c in g)
        return max(c[0] + c[2] for c in g) - min(c[0] for c in g)

    candidates: list[list[tuple[int, int, int, int]]] = []
    for g in groups:
        if extent(g) > 0.8 * span and len(g) > 1:        # two labels ran together: fall back to per glyph
            candidates.extend([c] for c in g)
        else:
            candidates.append(g)
    best: dict[int, tuple[float, list]] = {}
    for g in candidates:
        pos = centre(g)
        j = int(np.argmin([abs(pos - t) for t in ticks]))
        d = abs(pos - ticks[j])
        if d <= min(0.5 * span, extent(g) / 2.0 + 1.2 * med_h + 4.0) and (j not in best or d < best[j][0]):
            best[j] = (d, g)
    return [best[j][1] for j in sorted(best)]


def _band_nearest_axis(comps: list[tuple[int, int, int, int]], side: str, shape: tuple[int, ...],
                       gap_tol: float) -> list[tuple[int, int, int, int]]:
    """Keep only the strip of components closest to the axis.

    Walking away from the axis, the first clear gap wider than `gap_tol` ends the tick labels: whatever lies
    beyond it is the axis title, a legend or another panel. Gaps between glyphs of one label are far smaller
    than the gap that separates the labels from the next thing along.
    """
    length = shape[1] if side == "left" else shape[0]
    occupied = np.zeros(length + 1, dtype=bool)
    for x, y, w, h in comps:
        if side == "left":
            occupied[max(0, x):min(length, x + w)] = True
        else:
            occupied[max(0, y):min(length, y + h)] = True
    order = range(length - 1, -1, -1) if side == "left" else range(length)
    start = None
    gap = 0
    edge = None
    for i in order:
        if occupied[i]:
            if start is None:
                start = i
            gap = 0
            edge = i
        elif start is not None:
            gap += 1
            if gap > gap_tol:
                break
    if edge is None:
        return comps
    if side == "left":
        return [c for c in comps if c[0] + c[2] > edge - 1]
    return [c for c in comps if c[1] < edge + 1]


# ----------------------------------------------------------------------------- sub-pixel snapping
def snap_horizontal_edge(img: ImageLike, x: float, y: float, window: int = 6,
                         band: int = 2) -> tuple[float, float]:
    """Refine the row of a horizontal feature near (x, y) -> (row, confidence 0..1).

    Looks at a column band of +-`band` px around `x` and +-`window` rows around `y`. A *line* (dark with
    lighter rows on both sides) is located by the intensity-weighted centroid of its half-maximum run; a
    *step edge* (a filled bar/box whose ink runs to the edge of the window) is located by integrating the
    ink coverage, which is where the centroid would be badly biased. Confidence 0 means "no ink here".
    Pass `window=snap_window_for(bar_width)` for bar tops (amendment F).
    """
    return _snap(_as_gray(img), x, y, window, band, horizontal=True)


def snap_vertical_edge(img: ImageLike, x: float, y: float, window: int = 6,
                       band: int = 2) -> tuple[float, float]:
    """Refine the column of a vertical feature near (x, y) -> (column, confidence 0..1). See `snap_horizontal_edge`."""
    return _snap(_as_gray(img), x, y, window, band, horizontal=False)


def _snap(gray: np.ndarray, x: float, y: float, window: int, band: int, horizontal: bool) -> tuple[float, float]:
    h, w = gray.shape
    if horizontal:
        a0, a1 = int(round(y)) - window, int(round(y)) + window + 1       # rows scanned
        b0, b1 = int(round(x)) - band, int(round(x)) + band + 1           # columns averaged
        limit_a, limit_b = h, w
    else:
        a0, a1 = int(round(x)) - window, int(round(x)) + window + 1
        b0, b1 = int(round(y)) - band, int(round(y)) + band + 1
        limit_a, limit_b = w, h
    a0, a1 = max(0, a0), min(limit_a, a1)
    b0, b1 = max(0, b0), min(limit_b, b1)
    if a1 - a0 < 2 or b1 - b0 < 1:
        return (float(y) if horizontal else float(x)), 0.0
    patch = (gray[a0:a1, b0:b1] if horizontal else gray[b0:b1, a0:a1].T).astype(np.float32)
    bg = float(np.percentile(patch, 90))
    ink = np.clip(bg - patch, 0.0, None)
    profile = ink.mean(axis=1)
    peak = float(profile.max())
    if peak < _MIN_INK:
        return (float(y) if horizontal else float(x)), 0.0
    k = int(np.argmax(profile))
    lo = hi = k
    while lo > 0 and profile[lo - 1] >= 0.5 * peak:
        lo -= 1
    while hi < len(profile) - 1 and profile[hi + 1] >= 0.5 * peak:
        hi += 1
    alpha = np.clip(profile / peak, 0.0, 1.0)
    n = len(profile)
    if lo == 0 and hi == n - 1:                                            # entirely inked: cannot localize
        pos = a0 + float(np.average(np.arange(n), weights=profile))
        return pos, 0.15
    if hi == n - 1:                                                        # ink runs to the far side: top edge
        pos = a0 + (lo + 0.5) - float(alpha[: lo + 1].sum())
    elif lo == 0:                                                          # ink runs to the near side: bottom edge
        pos = a0 + (hi - 0.5) + float(alpha[hi:].sum())
    else:                                                                  # isolated line: centroid
        idx = np.arange(lo, hi + 1, dtype=float)
        pos = a0 + float(np.average(idx, weights=profile[lo:hi + 1]))
    strongest = ink[lo:hi + 1].max(axis=0)
    coverage = float((strongest >= 0.5 * peak).mean())
    conf = float(np.clip(peak / 128.0 * coverage, 0.0, 1.0))
    return pos, conf


def find_cap_ends(img: ImageLike, x: float, y_center: float, max_len_px: float,
                  band: int = 1) -> tuple[float | None, float | None]:
    """Walk up and down the whisker at column `x` from `y_center` -> (top row, bottom row).

    The walk stops at the first gap of more than 2 px (or after `max_len_px`); the end is then refined
    sub-pixel on the cap. Either side is None when there is no ink to walk along.
    """
    gray = _as_gray(img)
    h, w = gray.shape
    dark = _dark(gray)
    c0, c1 = max(0, int(round(x)) - band), min(w, int(round(x)) + band + 1)
    if c1 <= c0:
        return None, None
    col = dark[:, c0:c1].any(axis=1)
    y0 = int(round(y_center))
    if not (0 <= y0 < h):
        return None, None
    ends: list[float | None] = []
    for step in (-1, +1):
        last = None
        gap = 0
        r = y0
        while 0 <= r < h and abs(r - y0) <= max_len_px:
            if col[r]:
                last = r
                gap = 0
            else:
                gap += 1
                if gap > 2:
                    break
            r += step
        if last is None:
            ends.append(None)
            continue
        ends.append(_refine_end(gray, dark, x, last, band))
    return ends[0], ends[1]


def _refine_end(gray: np.ndarray, dark: np.ndarray, x: float, last: int, band: int) -> float:
    """Sub-pixel row of a whisker end: the centre line of its cap, measured beside the stem."""
    h, w = dark.shape
    best = None
    xi = int(round(x))
    for r in range(max(0, last - 2), min(h, last + 3)):
        row = dark[r]
        if not row[xi]:
            continue
        left = xi
        while left > 0 and row[left - 1]:
            left -= 1
        right = xi
        while right < w - 1 and row[right + 1]:
            right += 1
        if best is None or (right - left) > (best[2] - best[1]):
            best = (r, left, right)
    if best is None or (best[2] - best[1] + 1) < 2 * band + 3:
        return float(last)                                    # no cap: the stem simply stops here
    r, left, right = best
    side = (x + right) / 2.0 if right - xi >= xi - left else (x + left) / 2.0
    refined, conf = snap_horizontal_edge(gray, side, float(r), window=3, band=1)
    return float(refined) if conf > 0 else float(r)


def detect_whiskers(img: ImageLike, marker_or_bar, max_len_px: float | None = None
                    ) -> tuple[float | None, float | None]:
    """Error-bar ends for a detected `Marker`, `Bar`, dict or (x, y) tuple -> (top row, bottom row)."""
    gray = _as_gray(img)
    if isinstance(marker_or_bar, Marker):
        x, y = marker_or_bar.x, marker_or_bar.y
    elif isinstance(marker_or_bar, Bar):
        x, y = marker_or_bar.x_center, marker_or_bar.top_y
    elif isinstance(marker_or_bar, dict):
        xv = marker_or_bar.get("x", marker_or_bar.get("x_center"))
        yv = marker_or_bar.get("y", marker_or_bar.get("top_y"))
        if xv is None or yv is None:
            raise ValueError("detect_whiskers: dict needs 'x'/'y' (or 'x_center'/'top_y'), got "
                             f"{sorted(marker_or_bar)}")
        x, y = float(xv), float(yv)
    else:
        x, y = float(marker_or_bar[0]), float(marker_or_bar[1])
    if max_len_px is None:
        max_len_px = 0.35 * gray.shape[0]
    return find_cap_ends(gray, float(x), float(y), max_len_px)


# ----------------------------------------------------------------------------- path B: marks by colour
def _plot_region(axes: Axes, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Plot area inset by a few pixels so the axis strokes themselves are not part of the content."""
    h, w = shape[:2]
    x0, y0, x1, y1 = axes.plot_bbox
    pad = max(3.0, 0.005 * max(x1 - x0, y1 - y0), axes.y_axis_width + 1, axes.x_axis_width + 1)
    return (int(max(0, min(w - 1, x0 + pad))), int(max(0, min(h - 1, y0 + pad))),
            int(max(1, min(w, x1 - pad))), int(max(1, min(h, y1 - pad))))


def _background(region: np.ndarray) -> np.ndarray:
    """Modal colour of a BGR region (the paper/panel background)."""
    q = (region.reshape(-1, 3) // 32).astype(np.int32)
    code = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    vals, counts = np.unique(code, return_counts=True)
    mode = vals[int(np.argmax(counts))]
    return np.median(region.reshape(-1, 3)[code == mode], axis=0)


def _ink_from_colour(region: np.ndarray, bg: np.ndarray, tol: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """(boolean ink mask, grayscale 'ink' image) for a BGR region against background colour `bg`."""
    dist = np.linalg.norm(region.astype(np.float32) - bg.astype(np.float32), axis=2)
    scale = max(float(np.percentile(dist, 99.5)), tol * 2.0)
    ink_gray = np.clip(255.0 - dist / scale * 255.0, 0, 255).astype(np.uint8)
    return dist > tol, ink_gray


def detect_bars(img_bgr: ImageLike, axes: Axes, min_width_px: int = MIN_BAR_WIDTH_PX,
                tol: float = 40.0) -> list[Bar]:
    """Detect bars that grow from the x axis, by colour segmentation -> list of `Bar` sorted left to right.

    A bar is a run of columns whose non-background ink reaches down to the baseline; its top is the median
    of the per-column tops (so an error bar drawn on top of the bar does not raise it) refined sub-pixel
    with the amendment's adaptive window (0.25x bar width, min 3 px). Bars narrower than `min_width_px` are
    returned with `narrow=True` — their read-out should be treated as unreliable, not silently dropped.
    """
    bgr = _as_bgr(img_bgr)
    x0, y0, x1, y1 = _plot_region(axes, bgr.shape)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    region = bgr[y0:y1, x0:x1]
    bg = _background(region)
    mask, ink_gray = _ink_from_colour(region, bg, tol)
    hh, ww = mask.shape
    if axes.x_axis_y is not None:                      # bars grow from the x axis, not from the crop edge
        base_row = int(round(axes.x_axis_y - axes.x_axis_width / 2.0 - 1)) - y0
        base_row = min(max(base_row, 0), hh - 1)
    else:
        base_row = hh - 1
    while base_row > 0 and not mask[base_row, :].any():
        base_row -= 1
    tops: list[float | None] = []
    for c in range(ww):
        if not mask[base_row, c]:
            tops.append(None)
            continue
        r = base_row
        gap = 0
        top = base_row
        while r >= 0:
            if mask[r, c]:
                top = r
                gap = 0
            else:
                gap += 1
                if gap > 1:
                    break
            r -= 1
        tops.append(float(top))
    idx = np.flatnonzero(np.array([t is not None for t in tops]))
    if idx.size == 0:
        return []
    bars: list[Bar] = []
    height_tol = max(2.0, 0.02 * hh)
    for g in _groups(idx, gap=1):
        if len(g) < 2:
            continue
        vals = np.array([tops[c] for c in g], dtype=float)
        med = float(np.median(vals))
        body = [c for c, t in zip(g, vals) if abs(t - med) <= height_tol]
        if len(body) < max(2, len(g) // 4):
            continue
        med = float(np.median([tops[c] for c in body]))
        if base_row - med < 2:
            continue
        bx0, bx1 = float(g[0]) - 0.5, float(g[-1]) + 0.5
        width = bx1 - bx0
        cx = 0.5 * (bx0 + bx1)
        top_y, conf = snap_horizontal_edge(ink_gray, cx, med, window=snap_window_for(width),
                                           band=max(1, int(width // 4)))
        if conf <= 0.0:
            top_y = med
        inset = int(min(2, max(0, width // 4)))
        interior = region[int(med) + 2:base_row - 1, g[0] + inset:g[-1] + 1 - inset].reshape(-1, 3)
        colour = _hex(np.median(interior, axis=0)) if interior.size else _hex(region[int(med) + 1, int(cx)])
        base_y = axes.x_axis_y if axes.x_axis_y is not None else float(base_row) + y0 + 0.5
        bars.append(Bar(x_center=cx + x0, x0=bx0 + x0, x1=bx1 + x0, top_y=top_y + y0,
                        base_y=float(base_y), colour=colour, narrow=width < min_width_px))
    return bars


def detect_markers(img_bgr: ImageLike, axes: Axes, tol: float = 40.0) -> list[Marker]:
    """Detect plotted point markers -> list of `Marker` sorted left to right.

    Markers are found as the *thick* parts of the ink: a distance transform separates blobs (large inradius)
    from the thin lines joining them, so markers on a line chart are still detected individually. `kind` is
    a coarse shape guess (circle/square/triangle/open) from the fill ratio of the marker's bounding box.
    """
    bgr = _as_bgr(img_bgr)
    x0, y0, x1, y1 = _plot_region(axes, bgr.shape)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    region = bgr[y0:y1, x0:x1]
    bg = _background(region)
    mask, _ = _ink_from_colour(region, bg, tol)
    if not mask.any():
        return []
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    p99 = float(np.percentile(dist[mask], 99))
    thr = max(1.5, 0.55 * p99)
    cores = (dist >= thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
    limit = 0.25 * min(x1 - x0, y1 - y0)
    out: list[Marker] = []
    for i in range(1, n):
        cx0, cy0, cw, ch, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH],
                                  stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA])
        if area < 2 or max(cw, ch) > limit:
            continue
        sel = labels[cy0:cy0 + ch, cx0:cx0 + cw] == i
        wts = dist[cy0:cy0 + ch, cx0:cx0 + cw] * sel
        if wts.sum() <= 0:
            continue
        ys, xs = np.mgrid[cy0:cy0 + ch, cx0:cx0 + cw]
        mx = float((xs * wts).sum() / wts.sum())
        my = float((ys * wts).sum() / wts.sum())
        pad = int(np.ceil(2.5 * thr)) + 2
        px0, py0 = max(0, cx0 - pad), max(0, cy0 - pad)
        px1, py1 = min(mask.shape[1], cx0 + cw + pad), min(mask.shape[0], cy0 + ch + pad)
        local = mask[py0:py1, px0:px1].astype(np.uint8)
        ln, llab = cv2.connectedComponents(local, connectivity=8)
        lid = llab[int(round(my)) - py0, int(round(mx)) - px0]
        blob = (llab == lid) if lid > 0 else local.astype(bool)
        touches = bool(blob[0, :].any() or blob[-1, :].any() or blob[:, 0].any() or blob[:, -1].any())
        shape_mask = sel if touches else blob
        kind, size = _classify_shape(shape_mask)
        if touches:
            size = max(size, 2.0 * thr)
        colour_px = region[cy0:cy0 + ch, cx0:cx0 + cw].reshape(-1, 3)[sel.reshape(-1)]
        out.append(Marker(x=mx + x0, y=my + y0, colour=_hex(np.median(colour_px, axis=0)), kind=kind, size=size))
    out.sort(key=lambda m: (m.x, m.y))
    return out


def _classify_shape(shape_mask: np.ndarray) -> tuple[str, float]:
    ys, xs = np.nonzero(shape_mask)
    if ys.size == 0:
        return "blob", 0.0
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    fill = float(ys.size) / (w * h)
    size = 0.5 * (w + h)
    aspect = w / h if h else 99.0
    if not (0.7 <= aspect <= 1.45):
        return "blob", size
    if fill > 0.9:
        return "square", size
    if fill >= 0.68:
        return "circle", size
    if fill >= 0.35:
        return "triangle", size
    return "open", size
