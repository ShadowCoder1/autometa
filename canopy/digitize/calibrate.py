"""Axis calibration: turn detected tick marks/labels into a pixel -> data-value mapping.

All pixel coordinates in this package are **image pixels of the figure crop** (`FigureRegion.crop_png`),
origin top-left, x to the right, y downward. A y-axis therefore normally has a *negative* slope `a`
(rows grow downward while values grow upward).

    linear:  value = a * px + b
    log:     value = 10 ** (a * px + b)

`AxisCalibration.rmse` is the RMS residual of the kept ticks **in pixels** (not data units), so it is
directly comparable with the sub-pixel snapping error; multiply by `pixel_resolution()` for data units.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Axis = Literal["x", "y"]
Scale = Literal["linear", "log"]

# unicode characters that stand in for "-" and for a thousands space in typeset/OCR'd tick labels
_MINUSES = "−–—‐‑­˗﹣－"
_SPACES = "        　"
_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_DECIMAL_COMMA_RE = re.compile(r"^[+-]?\d+,\d{1,2}$")
_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class TickLabel:
    """A number printed next to an axis, found by OCR (raster) or read from the PDF text layer (vector)."""

    text: str
    value: float | None                          # parsed number, None when the text is not numeric
    bbox: tuple[float, float, float, float]      # x0, y0, x1, y1 in crop pixels
    center: tuple[float, float]                  # centre of bbox, crop pixels
    source: str = ""                             # "ocr" | "vector"
    confidence: float = 1.0                      # 0..1 (OCR confidence; 1.0 for vector text)

    def to_dict(self) -> dict:
        return dict(text=self.text, value=self.value, bbox=list(self.bbox), center=list(self.center),
                    source=self.source, confidence=self.confidence)


@dataclass
class AxisCalibration:
    """Least-squares mapping between crop pixels and data values along one axis."""

    axis: Axis
    scale: Scale
    a: float
    b: float
    rmse: float                                  # RMS residual of `ticks`, in PIXELS
    ticks: list[tuple[float, float]]             # kept (pixel, value) pairs
    dropped: list[tuple[float, float]] = field(default_factory=list)   # robustly rejected ticks

    def to_dict(self) -> dict:
        return dict(axis=self.axis, scale=self.scale, a=self.a, b=self.b, rmse=self.rmse,
                    ticks=[[p, v] for p, v in self.ticks], dropped=[[p, v] for p, v in self.dropped])


# ----------------------------------------------------------------------------- numeric parsing
def parse_number(text: str | None) -> float | None:
    """Parse a tick label into a float, or None if it is not a number.

    Handles unicode minus/en-dash, thin and non-breaking spaces, thousands separators ("1,234", "1 234"),
    decimal commas ("1,5"), a trailing percent sign and scientific notation.
    """
    if not text:
        return None
    s = str(text).strip()
    for ch in _MINUSES:
        s = s.replace(ch, "-")
    for ch in _SPACES:
        s = s.replace(ch, " ")
    s = s.replace("%", "").replace("°", "")
    s = re.sub(r"\s+", "", s)                      # "1 234" -> "1234"
    if not s:
        return None
    if _THOUSANDS_RE.match(s):
        s = s.replace(",", "")
    elif _DECIMAL_COMMA_RE.match(s):
        s = s.replace(",", ".")
    if not _NUMBER_RE.match(s):
        return None
    try:
        return float(s)
    except ValueError:                             # pragma: no cover - regex already guarantees this
        return None


# ----------------------------------------------------------------------------- fitting
def _lsq(px: np.ndarray, val: np.ndarray) -> tuple[float, float]:
    A = np.column_stack([px, np.ones_like(px)])
    (a, b), *_ = np.linalg.lstsq(A, val, rcond=None)
    return float(a), float(b)


def fit_axis(ticks: list[tuple[float, float]], scale: Scale = "linear", axis: Axis = "y") -> AxisCalibration:
    """Least-squares fit of `value = a*px + b` (or `10**(a*px+b)` for log axes) through (pixel, value) ticks.

    Needs >= 2 ticks. With >= 4 ticks one robust pass runs: the tick with the largest pixel residual is
    dropped and the fit repeated when that residual exceeds 3x the median residual (and 0.25 px), which
    catches a single mis-OCR'd label or a mis-detected tick line. Dropped ticks are kept in `.dropped`.
    """
    if len(ticks) < 2:
        raise ValueError(f"need >= 2 ticks to calibrate an axis, got {len(ticks)}")
    px = np.asarray([float(p) for p, _ in ticks], dtype=float)
    raw = np.asarray([float(v) for _, v in ticks], dtype=float)
    if len(set(px.tolist())) != len(px):
        raise ValueError("duplicate tick pixel positions")
    if scale == "log":
        if np.any(raw <= 0):
            raise ValueError("log axis needs strictly positive tick values")
        val = np.log10(raw)
    elif scale == "linear":
        val = raw
    else:                                          # pragma: no cover - guarded by the type hint
        raise ValueError(f"unknown scale {scale!r}")

    if float(np.ptp(val)) == 0.0:
        raise ValueError("degenerate axis: all tick values identical")
    keep = np.ones(len(px), dtype=bool)
    a, b = _lsq(px, val)
    if not math.isfinite(a) or not math.isfinite(b) or a == 0.0:
        raise ValueError("degenerate axis fit (no usable slope)")
    dropped: list[tuple[float, float]] = []
    if len(px) >= 4:
        resid_px = np.abs(px - _inverse(a, b, val))
        med = float(np.median(resid_px))
        worst = int(np.argmax(resid_px))
        if resid_px[worst] > max(3.0 * med, 0.25):
            keep[worst] = False
            dropped.append((float(px[worst]), float(raw[worst])))
            a, b = _lsq(px[keep], val[keep])
            if not math.isfinite(a) or not math.isfinite(b) or a == 0.0:
                raise ValueError("degenerate axis fit (all tick values identical?)")
    resid = px[keep] - _inverse(a, b, val[keep])
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    kept = [(float(p), float(v)) for p, v, k in zip(px, raw, keep) if k]
    return AxisCalibration(axis=axis, scale=scale, a=a, b=b, rmse=rmse, ticks=kept, dropped=dropped)


def _inverse(a: float, b: float, val: np.ndarray | float):
    """Pixel position(s) predicted for fitted (already log-transformed for log axes) values."""
    return (val - b) / a


def px_to_value(cal: AxisCalibration, px: float) -> float:
    """Data value at pixel position `px`."""
    lin = cal.a * float(px) + cal.b
    return 10.0 ** lin if cal.scale == "log" else lin


def value_to_px(cal: AxisCalibration, value: float) -> float:
    """Pixel position of a data value (inverse of `px_to_value`)."""
    if cal.scale == "log":
        if value <= 0:
            raise ValueError("log axis: value must be > 0")
        value = math.log10(value)
    return (float(value) - cal.b) / cal.a


def pixel_resolution(cal: AxisCalibration, px: float | None = None) -> float:
    """Data units per pixel — the floor on digitization precision.

    Constant for a linear axis; for a log axis it is evaluated at `px` (default: the middle of the ticks).
    """
    if cal.scale != "log":
        return abs(cal.a)
    if px is None:
        pxs = [p for p, _ in cal.ticks] or [0.0]
        px = (min(pxs) + max(pxs)) / 2.0
    return abs(cal.a) * math.log(10.0) * px_to_value(cal, px)


# ----------------------------------------------------------------------------- pairing
def pair_ticks(labels: list[TickLabel], tick_lines: list[float], axis: Axis = "y",
               max_dist: float | None = None) -> list[tuple[float, float]]:
    """Pair numeric tick labels with detected tick-line positions, returning (pixel, value) pairs.

    Each label is matched to the nearest unused tick line along `axis` (y-centres for a y-axis, x-centres
    for an x-axis); labels further than `max_dist` (default: half the median tick spacing, min 3 px) stay
    unpaired. When `tick_lines` is empty the label centres are used as the pixel positions.

    Pairs whose (pixel, value) relation is inconsistent with the robust (Theil-Sen) trend of the others —
    a mis-read digit, or a label matched to the wrong tick — are rejected. Fewer than 2 surviving pairs
    returns [] (an axis cannot be calibrated from one point).
    """
    coord = (lambda t: t.center[1]) if axis == "y" else (lambda t: t.center[0])
    numeric = [t for t in labels if t.value is not None]
    if not numeric:
        return []
    lines = sorted(float(t) for t in tick_lines)
    if not lines:
        pairs = [(coord(t), float(t.value)) for t in numeric]
    else:
        if max_dist is None:
            gaps = np.diff(lines) if len(lines) > 1 else np.array([6.0])
            max_dist = max(3.0, float(np.median(gaps)) / 2.0)
        cands = sorted((abs(coord(t) - ln), i, j) for i, t in enumerate(numeric) for j, ln in enumerate(lines)
                       if abs(coord(t) - ln) <= max_dist)
        used_l: set[int] = set()
        used_t: set[int] = set()
        pairs = []
        for _, i, j in cands:
            if i in used_l or j in used_t:
                continue
            used_l.add(i)
            used_t.add(j)
            pairs.append((lines[j], float(numeric[i].value)))
    pairs.sort()
    return _drop_inconsistent(pairs)


def _drop_inconsistent(pairs: list[tuple[float, float]], tol_frac: float = 0.25) -> list[tuple[float, float]]:
    """Drop pairs that disagree with the robust trend (median pairwise slope) of the rest."""
    if len(pairs) < 2:
        return []
    if len(pairs) < 3:
        return pairs
    px = np.array([p for p, _ in pairs], dtype=float)
    val = np.array([v for _, v in pairs], dtype=float)
    slopes = [(val[j] - val[i]) / (px[j] - px[i]) for i in range(len(px)) for j in range(i + 1, len(px))
              if px[j] != px[i]]
    slope = float(np.median(slopes)) if slopes else 0.0
    if slope == 0.0:
        return pairs
    intercept = float(np.median(val - slope * px))
    resid_px = np.abs(px - (val - intercept) / slope)
    med = float(np.median(resid_px))
    spacing = float(np.median(np.abs(np.diff(px)))) if len(px) > 1 else 1.0
    tol = max(3.0 * med, tol_frac * spacing)
    kept = [pr for pr, r in zip(pairs, resid_px) if r <= tol]
    return kept if len(kept) >= 2 else []
