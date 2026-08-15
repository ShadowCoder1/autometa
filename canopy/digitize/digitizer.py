"""`digitize()` — read one figure with four independent routes and reconcile them.

Build order is amendment F's:

* **D** read-out — the vision model reads the values (several model x variant samples);
* **C** VLM coordinates + CV snap — the model points at pixels, code calibrates and sub-pixel
  snaps them, code converts to values;
* **B** raster CV — bars/markers/caps found by colour segmentation, matched to the target by the
  nearest VLM coordinate; an independent voter whenever segmentation succeeds;
* **A** vector-exact — the PDF's own geometry, when the figure has one and its tick labels
  survive cross-checking.

Every sample becomes its own `Candidate` (so task 8 can vote on the raw routes), plus one ensemble
`Candidate` per group carrying the median, `sigma` and the disagreement verdict. Nothing here
computes an effect size.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ingest.pdf import FigureRegion, PaperRecord
from ..llm.client import LLMClient
from ..models import Candidate, DatasetSpec, DispersionType, Source, SourceKind
from .calibrate import AxisCalibration, fit_axis, pair_ticks, pixel_resolution, \
    px_to_value, value_to_px
from .cv import (Axes, Bar, Marker, MIN_BAR_WIDTH_PX, detect_bars, detect_markers, find_axes,
                 find_cap_ends, find_tick_marks, load_color, load_gray, ocr_tick_labels,
                 snap_horizontal_edge, snap_window_for)
from .vector import VectorScene, calibrate_from_scene, snap_to_vector, vector_candidates, \
    whisker_ends
from .vlm import (READOUT_VARIANTS, CoordReadout, FigureView, PROMPT_VERSION, ReadOut,
                  TargetSpec, coords, overlay_verify, read_out, summarize_tool_calls)

__all__ = ["digitize", "RouteSample", "ReadoutSpec", "DigitizeResult", "ensemble_stats",
           "dual_tolerance", "ROUTE_LABELS"]

ROUTE_LABELS = {"A": "vector", "B": "raster_cv", "C": "vlm_coords", "D": "readout"}
GROUPS = ("A", "B")
MAX_OVERLAY_ITERATIONS = 2
_MEAN_TOL_FRACTION = 0.02        # amendment F: 2% of the y-axis range
_ERR_TOL_FRACTION = 0.10         # amendment F: 10% of the error half-length
_MARK_MERGE_PX = 1.5             # marks closer than this are one mark on the overlay
_MAD_TO_SIGMA = 1.4826


# ----------------------------------------------------------------------------- samples
@dataclass(frozen=True)
class ReadoutSpec:
    """One planned path-D sample: which model, which prompt variant, which repeat of that prompt."""

    model: str
    variant: str
    sample: int = 0                              # 0 = first use of this (model, variant) pair

    @property
    def resample(self) -> bool:
        return self.sample > 0

    def to_dict(self) -> dict[str, Any]:
        return dict(model=self.model, variant=self.variant, sample=self.sample,
                    resample=self.resample)


@dataclass
class RouteSample:
    """One route's answer for one group. `route` is A/B/C/D; `mean`/`error` are in data units."""

    route: str
    group: str
    model: str = ""
    variant: str = ""
    sample: int = 0                              # path D only: >0 = a re-ask of the same prompt
    mean: float | None = None
    error: float | None = None                   # error-bar HALF-length
    x_px: float | None = None
    y_px: float | None = None
    cap_top_px: float | None = None
    cap_bottom_px: float | None = None
    snap_conf: float | None = None
    sigma: float | None = None
    label_read: str = ""
    status: str = "found"
    notes: str = ""
    cal_source: str = ""
    call_ids: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    dropped: bool = False
    drop_reason: str = ""

    @property
    def extractor_id(self) -> str:
        parts = ["digitize", ROUTE_LABELS.get(self.route, self.route)]
        if self.model:
            parts.append(self.model)
        if self.variant:
            parts.append(self.variant)
        name = ":".join(parts)
        return f"{name}#{self.sample + 1}" if self.sample else name

    @property
    def usable(self) -> bool:
        return (not self.dropped) and self.mean is not None

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "tool_calls"}
        d["extractor_id"] = self.extractor_id
        d["n_tool_calls"] = len(self.tool_calls)
        return d


@dataclass
class DigitizeResult:
    """Everything one `digitize()` produced (the candidates plus the working it showed)."""

    candidates: list[Candidate] = field(default_factory=list)
    samples: list[RouteSample] = field(default_factory=list)
    calibration: AxisCalibration | None = None
    overlay_path: str = ""
    cost_usd: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- small maths
def ensemble_stats(values: Sequence[float]) -> tuple[float, float]:
    """Median and a robust spread (1.4826 x MAD) — the amendment-F ensemble estimator."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("ensemble_stats needs at least one value")
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    return med, _MAD_TO_SIGMA * mad


def dual_tolerance(means: Sequence[float], errors: Sequence[float], *, axis_range: float,
                   tick_spacing: float, px_units: float) -> dict[str, Any]:
    """Amendment F's dual tolerance: are the routes close enough to accept without review?"""
    out: dict[str, Any] = {"mean_agrees": True, "error_agrees": True, "reasons": []}
    mean_tol = max(_MEAN_TOL_FRACTION * abs(axis_range), 0.5 * abs(tick_spacing))
    out["mean_tolerance"] = mean_tol
    if len(means) >= 2:
        spread = max(means) - min(means)
        out["mean_spread"] = spread
        if mean_tol > 0 and spread > mean_tol:
            out["mean_agrees"] = False
            out["reasons"].append(
                f"means span {spread:.4g} across routes, over the {mean_tol:.4g} tolerance "
                f"(max of 2% of the {abs(axis_range):.4g} axis range and half a "
                f"{abs(tick_spacing):.4g} tick)")
    live = [abs(e) for e in errors]
    if len(live) >= 2:
        med = statistics.median(live)
        err_tol = max(_ERR_TOL_FRACTION * med, 0.5 * abs(px_units))
        spread = max(live) - min(live)
        out["error_tolerance"] = err_tol
        out["error_spread"] = spread
        if err_tol > 0 and spread > err_tol:
            out["error_agrees"] = False
            out["reasons"].append(
                f"error half-lengths span {spread:.4g}, over the {err_tol:.4g} tolerance "
                f"(max of 10% of {med:.4g} and half a pixel = {abs(px_units) / 2:.4g})")
    out["agrees"] = out["mean_agrees"] and out["error_agrees"]
    return out


# ----------------------------------------------------------------------------- CV core
@dataclass
class _Core:
    gray: Any
    colour: Any
    axes: Axes
    tick_rows: list[float]
    labels: Any
    cal: AxisCalibration | None
    ocr_status: str
    bars: list[Bar]
    markers: list[Marker]

    @property
    def is_bar_chart(self) -> bool:
        return bool(self.bars)


def _cv_core(crop_png: Path) -> _Core:
    """The shared deterministic pass every pixel route reuses (axes, ticks, OCR, LS fit)."""
    gray = load_gray(crop_png)
    colour = load_color(crop_png)
    axes = find_axes(gray)
    rows = find_tick_marks(gray, axes).get("left", []) if axes.y_axis_x is not None else []
    labels = ocr_tick_labels(gray, axes, side="left", ticks=rows or None)
    status = str(getattr(labels, "status", "missing"))
    pairs = pair_ticks(list(labels), rows, axis="y")
    cal: AxisCalibration | None = None
    if len(pairs) >= 2:
        try:
            cal = fit_axis(pairs, axis="y")
        except ValueError:                                    # degenerate ticks
            cal = None
    try:
        bars = detect_bars(colour, axes)
    except Exception:                                         # pragma: no cover - defensive
        bars = []
    markers: list[Marker] = []
    if not bars:
        try:
            markers = detect_markers(colour, axes)
        except Exception:                                     # pragma: no cover - defensive
            markers = []
    return _Core(gray=gray, colour=colour, axes=axes, tick_rows=rows, labels=labels, cal=cal,
                 ocr_status=status, bars=bars, markers=markers)


def _axis_range(cal: AxisCalibration | None, core: "_Core") -> tuple[float, str]:
    """The y range the tolerance is a percentage OF — the plotted axis span when we can measure it.

    The tick-value range only covers the labelled ticks; a figure whose data runs past its top tick
    (or whose axis is drawn well beyond it) has a larger real range, and using the smaller number
    makes the 2 % tolerance too tight. Prefer the calibrated span of the detected plot box, and say
    in provenance which one was used.
    """
    if cal is not None:
        x0, y0, x1, y1 = core.axes.plot_bbox
        if y1 - y0 > 1:
            try:
                span = abs(px_to_value(cal, y0) - px_to_value(cal, y1))
            except Exception:                                 # pragma: no cover - defensive
                span = 0.0
            if span > 0:
                return span, "plot_bbox"
    span, _ = _tick_stats(cal)
    return span, "tick_range"


def _tick_stats(cal: AxisCalibration | None) -> tuple[float, float]:
    """(value range, median spacing) of a calibration's surviving ticks."""
    if cal is None or len(cal.ticks) < 2:
        return 0.0, 0.0
    values = sorted(v for _, v in cal.ticks)
    gaps = [b - a for a, b in zip(values, values[1:]) if b > a]
    return values[-1] - values[0], (statistics.median(gaps) if gaps else 0.0)


def _model_calibration(readout: CoordReadout) -> AxisCalibration | None:
    pairs = [(t.y_px, t.value) for t in readout.ticks]
    seen: dict[float, float] = {}
    for px, value in pairs:
        seen.setdefault(round(px, 3), value)
    clean = [(px, value) for px, value in seen.items()]
    if len(clean) < 2:
        return None
    try:
        return fit_axis(clean, axis="y")
    except ValueError:
        return None


def _calibrations_agree(a: AxisCalibration, b: AxisCalibration, tol: float) -> bool:
    """Do two y calibrations read the same value at the same pixel, within `tol` data units?"""
    pixels = [px for px, _ in a.ticks] or [0.0, 100.0]
    lo, hi = min(pixels), max(pixels)
    probes = [lo, (lo + hi) / 2.0, hi]
    try:
        return all(abs(px_to_value(a, p) - px_to_value(b, p)) <= tol for p in probes)
    except Exception:                                         # pragma: no cover - defensive
        return False


def _choose_calibration(core: _Core, coord: CoordReadout | None
                        ) -> tuple[AxisCalibration | None, str, str]:
    """Prefer the OCR/tick fit, but only when the model's own ticks corroborate it."""
    cal_model = _model_calibration(coord) if coord is not None else None
    if core.cal is not None and cal_model is not None:
        axis_range, spacing = _tick_stats(core.cal)
        tol = max(_MEAN_TOL_FRACTION * abs(axis_range), 0.5 * abs(spacing))
        if tol <= 0:
            tol = abs(axis_range) * _MEAN_TOL_FRACTION or 1e-9
        if _calibrations_agree(core.cal, cal_model, tol):
            return core.cal, "cv_ocr", "OCR ticks corroborated by the model's tick read"
        return cal_model, "vlm_ticks", ("OCR fit disagreed with the model's ticks; used the "
                                        "model's tick coordinates")
    if core.cal is not None:
        return core.cal, "cv_ocr", "no model ticks to cross-check against"
    if cal_model is not None:
        return cal_model, "vlm_ticks", "no usable OCR ticks"
    return None, "none", "no y calibration could be built"


# ----------------------------------------------------------------------------- read-out plan
def _readout_plan(models: Sequence[str], n_readouts: int) -> list[ReadoutSpec]:
    """Vote within route = modality x model family (amendment G), with no vote counted twice.

    Samples are taken from DISTINCT (model, variant) pairs — primary/direct, primary/ticks-first,
    then one variant per other model family, then the remaining variants. Only when every pair is
    spent does a pair get re-sampled, and such a sample is flagged `resample` so the ensemble and
    task 8 can see that it is a weaker vote: an identical prompt on an identical image is the same
    question asked twice, and under replay it would be a byte-identical copy that silently pins the
    median and deflates the MAD.
    """
    from ..config import MODELS

    primary = models[0] if models else MODELS["primary"]
    families = [primary]
    for model in list(models[1:]) + [MODELS["secondary"]]:
        if model not in families:
            families.append(model)
    others = families[1:]
    first, second, *rest = READOUT_VARIANTS
    pairs = [(primary, first), (primary, second)]
    pairs += [(m, first) for m in others]
    pairs += [(primary, v) for v in rest]
    pairs += [(m, v) for m in others for v in (second, *rest)]

    if n_readouts <= 0:
        return []
    specs = [ReadoutSpec(m, v) for m, v in pairs[:n_readouts]]
    overflow = 0
    while len(specs) < n_readouts:                      # every distinct pair is spent
        model, variant = pairs[overflow % len(pairs)]
        specs.append(ReadoutSpec(model, variant, sample=1 + overflow // len(pairs)))
        overflow += 1
    return specs


# ----------------------------------------------------------------------------- route D
def _samples_from_readout(reading: ReadOut) -> list[RouteSample]:
    out: list[RouteSample] = []
    for group in GROUPS:
        row = reading.group(group)
        if row is None:
            continue
        error = row.error_half_length
        if error is None and row.mean is not None:
            arms = [abs(cap - row.mean) for cap in (row.error_upper, row.error_lower)
                    if cap is not None]
            error = statistics.mean(arms) if arms else None
        out.append(RouteSample(
            route="D", group=group, model=reading.model, variant=reading.variant,
            sample=reading.sample,
            mean=row.mean, error=abs(error) if error is not None else None,
            label_read=row.label_read, status=reading.status,
            notes=row.notes, snap_conf=row.confidence,
            call_ids=list(reading.call_ids), tool_calls=list(reading.tool_calls),
            cost_usd=reading.cost_usd / max(1, len(reading.groups)),
            extra={"legend_says": reading.legend_says, "x_read": row.x_read,
                   "tick_labels": list(reading.tick_labels), "unit": reading.unit,
                   "panel": reading.panel, "same_prompt_resample": reading.sample > 0}))
    return out


# ----------------------------------------------------------------------------- route C
def _snap_point(core: _Core, x: float, y: float, width: float | None,
                bar: Bar | None = None) -> tuple[float, float]:
    """Sub-pixel refine a datum row. On a bar, snap off-centre: the error-bar stem runs down the
    middle of the bar, so the centre column has no step edge at the bar top at all."""
    window = snap_window_for(width) if width else 6
    column = x
    if bar is not None and bar.width >= 4:
        column = min(bar.x0, bar.x1) + 0.25 * bar.width
    return snap_horizontal_edge(core.gray, column, y, window=window)


def _values_from_pixels(cal: AxisCalibration, y: float, cap_top: float | None,
                        cap_bottom: float | None) -> tuple[float, float | None]:
    mean = px_to_value(cal, y)
    arms = [abs(px_to_value(cal, cap) - mean) for cap in (cap_top, cap_bottom) if cap is not None]
    return mean, (statistics.mean(arms) if arms else None)


def _samples_from_coords(coord: CoordReadout, core: _Core, cal: AxisCalibration | None,
                         cal_source: str) -> list[RouteSample]:
    out: list[RouteSample] = []
    for group in GROUPS:
        row = coord.group(group)
        if row is None or row.y_px is None:
            continue
        sample = RouteSample(route="C", group=group, model=coord.model, variant="",
                             label_read=row.label_read, status=coord.status,
                             call_ids=list(coord.call_ids), tool_calls=list(coord.tool_calls),
                             cost_usd=coord.cost_usd / max(1, len(coord.groups)),
                             cal_source=cal_source, notes=row.notes,
                             extra={"raw_y_px": row.y_px, "panel": coord.panel,
                                    "unit": coord.unit})
        x = row.x_px if row.x_px is not None else (core.axes.plot_bbox[0] + core.axes.plot_bbox[2]) / 2
        bar = _bar_at(core, x)
        snapped, conf = _snap_point(core, x, row.y_px, row.bar_width, bar)
        sample.x_px, sample.y_px, sample.snap_conf = x, snapped, conf
        top, bottom = row.cap_top_px, row.cap_bottom_px
        span = abs(core.axes.plot_bbox[3] - core.axes.plot_bbox[1]) or float(core.gray.shape[0])
        found_top, found_bottom = find_cap_ends(core.gray, x, snapped, max_len_px=span)
        if bar is not None:
            found_top = _cap_beyond_bar_top(found_top, bar)
            found_bottom = _cap_beyond_bar_top(found_bottom, bar)
        sample.cap_top_px = _closest(top, found_top)
        sample.cap_bottom_px = _closest(bottom, found_bottom)
        sample.extra["cap_source"] = {"model": [top, bottom], "cv": [found_top, found_bottom]}
        if cal is not None:
            sample.mean, sample.error = _values_from_pixels(
                cal, snapped, sample.cap_top_px, sample.cap_bottom_px)
            sample.sigma = _pixel_sigma(cal, snapped)
        else:
            sample.notes = (sample.notes + " no y calibration; pixels only").strip()
        out.append(sample)
    return out


def _bar_at(core: _Core, x: float | None) -> Bar | None:
    """The bar whose body contains column `x` (bar charts hide the lower error-bar arm)."""
    if x is None or not core.bars:
        return None
    for bar in core.bars:
        if min(bar.x0, bar.x1) - 1.0 <= x <= max(bar.x0, bar.x1) + 1.0:
            return bar
    return None


def _cap_beyond_bar_top(cap: float | None, bar: Bar | None, tol: float = 2.0) -> float | None:
    """Keep only an error-bar end on the bar's FREE side; anything else is the bar or the axis.

    A bar hides the arm drawn over its own body, and `find_cap_ends` cannot walk through the fill:
    it stops at the baseline, or runs on into the axis line and the tick marks under it. So the
    only end the raster path can legitimately measure is the one beyond the bar's free end, and
    the visible half-length is that single arm.
    """
    if cap is None or bar is None:
        return None
    if bar.top_y <= bar.base_y:                      # bar grows upwards: free side is above
        return cap if cap < bar.top_y - tol else None
    return cap if cap > bar.top_y + tol else None


def _closest(model_px: float | None, cv_px: float | None, tol: float = 12.0) -> float | None:
    """Prefer the CV cap when it corroborates the model's, else keep whichever exists."""
    if cv_px is None:
        return model_px
    if model_px is None:
        return cv_px
    return cv_px if abs(cv_px - model_px) <= tol else model_px


def _pixel_sigma(cal: AxisCalibration, px: float) -> float:
    res = abs(pixel_resolution(cal, px))
    return max(res, abs(cal.rmse) * res)


# ----------------------------------------------------------------------------- route B
def _samples_from_raster(coord: CoordReadout | None, core: _Core,
                         cal: AxisCalibration | None, cal_source: str) -> list[RouteSample]:
    """Colour-segmented bars/markers, matched to the target by the nearest VLM coordinate."""
    if cal is None or coord is None or not (core.bars or core.markers):
        return []
    out: list[RouteSample] = []
    span = abs(core.axes.plot_bbox[3] - core.axes.plot_bbox[1]) or float(core.gray.shape[0])
    for group in GROUPS:
        row = coord.group(group)
        if row is None or row.x_px is None:
            continue
        sample = RouteSample(route="B", group=group, cal_source=cal_source,
                             label_read=row.label_read)
        if core.bars:
            bar = min(core.bars, key=lambda b: abs(b.x_center - row.x_px))
            if abs(bar.x_center - row.x_px) > max(3.0 * max(bar.width, 1.0), 24.0):
                continue
            sample.x_px, sample.y_px = bar.x_center, bar.top_y
            sample.extra = {"bar": bar.to_dict(), "narrow": bar.narrow}
            top, bottom = find_cap_ends(core.gray, bar.x_center, bar.top_y, max_len_px=span)
            top, bottom = _cap_beyond_bar_top(top, bar), _cap_beyond_bar_top(bottom, bar)
            if bottom is None and top is not None:
                sample.notes = "error read from the upper arm only (the bar hides the lower arm)"
            if bar.narrow:
                sample.notes = (sample.notes + "; bar is narrower than "
                                f"{MIN_BAR_WIDTH_PX} px").strip("; ")
        else:
            if row.y_px is None:
                continue
            marker = min(core.markers,
                         key=lambda m: (m.x - row.x_px) ** 2 + (m.y - row.y_px) ** 2)
            if abs(marker.x - row.x_px) > max(3.0 * max(marker.size, 1.0), 24.0):
                continue
            sample.x_px, sample.y_px = marker.x, marker.y
            sample.extra = {"marker": marker.to_dict()}
            top, bottom = find_cap_ends(core.gray, marker.x, marker.y, max_len_px=span)
        sample.cap_top_px, sample.cap_bottom_px = top, bottom
        sample.mean, sample.error = _values_from_pixels(cal, sample.y_px, top, bottom)
        sample.sigma = _pixel_sigma(cal, sample.y_px)
        sample.snap_conf = 1.0
        out.append(sample)
    return out


# ----------------------------------------------------------------------------- route A
def _vector_scale_trusted(cal_vec: AxisCalibration, core: _Core,
                          readouts: Iterable[ReadOut]) -> tuple[bool, str]:
    """The Heuer text layer has two wrong tick labels — never trust a vector scale unchecked."""
    if len(cal_vec.ticks) < 3:
        return False, f"only {len(cal_vec.ticks)} vector tick label(s)"
    axis_range, spacing = _tick_stats(cal_vec)
    if spacing <= 0:
        return False, "vector tick values do not increase"
    values = sorted(v for _, v in cal_vec.ticks)
    gaps = [b - a for a, b in zip(values, values[1:])]
    if max(gaps) > 2.5 * min(gaps):
        return False, f"vector tick values are unevenly spaced ({values})"
    tol = max(_MEAN_TOL_FRACTION * abs(axis_range), 0.5 * abs(spacing))
    if core.cal is not None:
        if _calibrations_agree(cal_vec, core.cal, tol):
            return True, "vector scale agrees with the OCR tick fit"
        return False, "vector scale disagrees with the OCR tick fit"
    read_ticks = {round(v, 6) for r in readouts for v in r.tick_labels}
    if read_ticks:
        hits = sum(1 for v in values if any(abs(v - r) <= tol for r in read_ticks))
        if hits >= 3:
            return True, "vector tick values match the read-outs' tick ladder"
        return False, "vector tick values do not match the read-outs' tick ladder"
    return cal_vec.rmse < 0.5, f"vector fit rmse {cal_vec.rmse:.3g} px, no independent check"


def _samples_from_vector(paper: PaperRecord, fig: FigureRegion, coord: CoordReadout | None,
                         core: _Core, readouts: Sequence[ReadOut]
                         ) -> tuple[list[RouteSample], dict[str, Any]]:
    info: dict[str, Any] = {"attempted": False}
    if fig.kind not in ("vector", "mixed") or coord is None:
        info["skipped"] = f"figure kind {fig.kind!r}" if coord is not None else "no VLM coordinates"
        return [], info
    info["attempted"] = True
    try:
        scene: VectorScene = vector_candidates(paper, fig)
    except Exception as exc:                                  # pragma: no cover - defensive
        info["error"] = f"{type(exc).__name__}: {exc}"
        return [], info
    info["warnings"] = list(scene.warnings)
    info["n_tick_labels"] = len(scene.tick_labels)
    info["n_marks"] = len(scene.marks)
    if not scene.tick_labels:
        info["skipped"] = "no tick-label spans in the figure's text layer"
        return [], info
    out: list[RouteSample] = []
    for group in GROUPS:
        row = coord.group(group)
        if row is None or row.x_px is None or row.y_px is None:
            continue
        near = (row.x_px, row.y_px)
        cal_vec = calibrate_from_scene(scene, axis="y", near=near)
        if cal_vec is None:
            info.setdefault("per_group", {})[group] = "no vector y calibration near the point"
            continue
        trusted, why = _vector_scale_trusted(cal_vec, core, readouts)
        info.setdefault("per_group", {})[group] = {
            "rmse_px": cal_vec.rmse, "ticks": cal_vec.ticks, "trusted": trusted, "why": why}
        if not trusted:
            continue
        snapped = snap_to_vector(scene, row.x_px, row.y_px, radius=12.0)
        if snapped is None:
            continue
        sample = RouteSample(route="A", group=group, cal_source="vector",
                             label_read=row.label_read, x_px=snapped[0], y_px=snapped[1],
                             snap_conf=1.0, notes=why,
                             extra={"vector_warnings": list(scene.warnings)})
        mark = min(scene.marks, key=lambda m: (m.x - snapped[0]) ** 2 + (m.y - snapped[1]) ** 2,
                   default=None)
        ends = whisker_ends(scene, mark) if mark is not None else None
        if ends is not None:
            sample.cap_top_px, sample.cap_bottom_px = min(ends), max(ends)
        sample.mean, sample.error = _values_from_pixels(
            cal_vec, snapped[1], sample.cap_top_px, sample.cap_bottom_px)
        sample.sigma = 0.0                                    # exact geometry
        out.append(sample)
    return out, info


def _corroborate_vector_whiskers(samples: list[RouteSample], px_units: float) -> None:
    """6a can invent whiskers on flat line segments — only keep one another route confirms."""
    others = {}
    for s in samples:
        if s.route != "A" and s.error is not None:
            others.setdefault(s.group, []).append(abs(s.error))
    for s in samples:
        if s.route != "A" or s.error is None:
            continue
        peers = others.get(s.group)
        if not peers:
            s.extra["whisker_uncorroborated"] = True
            s.notes = (s.notes + "; vector whisker kept without corroboration").strip("; ")
            continue
        tol = max(_ERR_TOL_FRACTION * statistics.median(peers), abs(px_units))
        if min(abs(abs(s.error) - p) for p in peers) > tol:
            s.extra["whisker_rejected"] = {"vector": s.error, "peers": peers, "tol": tol}
            s.notes = (s.notes + "; vector whisker disagreed with the other routes "
                                 "(6a can mistake a flat segment for an error bar)").strip("; ")
            s.error = None


# ----------------------------------------------------------------------------- overlay verify
def _overlay_marks(samples: Sequence[RouteSample], cal: AxisCalibration | None,
                   core: _Core, labels: dict[str, str]) -> tuple[list[dict], list[list[int]]]:
    """One numbered mark per distinct resolved pixel, with the sample indices behind each."""
    marks: list[dict[str, Any]] = []
    owners: list[list[int]] = []
    x_by_group: dict[str, float] = {}
    for s in samples:
        if s.x_px is not None:
            x_by_group.setdefault(s.group, s.x_px)
    mid_x = (core.axes.plot_bbox[0] + core.axes.plot_bbox[2]) / 2.0
    for i, s in enumerate(samples):
        if s.dropped or s.mean is None:
            continue
        y = s.y_px
        if y is None:
            if cal is None:
                continue
            y = value_to_px(cal, s.mean)
        x = s.x_px if s.x_px is not None else x_by_group.get(s.group, mid_x)
        for j, mark in enumerate(marks):
            if abs(mark["x"] - x) <= _MARK_MERGE_PX and abs(mark["y"] - y) <= _MARK_MERGE_PX:
                owners[j].append(i)
                mark["_routes"].append(_route_tag(s))
                break
        else:
            marks.append({"x": x, "y": y, "kind": "point", "_routes": [_route_tag(s)],
                          "_group": s.group})
            owners.append([i])
    for mark in marks:
        who = ", ".join(dict.fromkeys(mark.pop("_routes")))
        group = mark.pop("_group")
        mark["label"] = (f"group {group} ({labels.get(group, '?')}) mean, read by {who}")
    return marks, owners


def _route_tag(s: RouteSample) -> str:
    tag = ROUTE_LABELS.get(s.route, s.route)
    if s.route == "D":
        return f"{tag}/{s.model.split('-')[-1]}/{s.variant}"
    return tag


# ----------------------------------------------------------------------------- legend check
_LEGEND_PATTERNS = (
    (DispersionType.SE, ("standard error", "std. error", "s.e.m", "sem", " se ", "±se", "+/- se")),
    (DispersionType.SD, ("standard deviation", "std. dev", "s.d.", " sd ", "±sd", "+/- sd")),
    (DispersionType.CI95, ("95% confidence", "95% ci", "confidence interval")),
    (DispersionType.IQR, ("interquartile", "iqr")),
    (DispersionType.RANGE, ("min-max", "range of")),
)


def _legend_dispersion(text: str) -> DispersionType | None:
    padded = f" {(text or '').lower()} "
    for kind, needles in _LEGEND_PATTERNS:
        if any(n in padded for n in needles):
            return kind
    return None


# ----------------------------------------------------------------------------- entry point
def digitize(client: LLMClient, paper: PaperRecord, fig: FigureRegion, target: TargetSpec, *,
             models: Sequence[str] = ("claude-opus-5",), n_readouts: int = 3,
             want_uncertainty: bool = True, source: Source | None = None,
             dataset: DatasetSpec | None = None, out_dir: str | Path | None = None,
             caption: str | None = None, cell_key: str = "",
             verify: bool = True, result: bool = False) -> list[Candidate] | DigitizeResult:
    """Read `fig` for `target` with every available route and return the `Candidate`s.

    Returns one `Candidate` per (group, route sample) plus one ensemble `Candidate` per group.
    Pass `result=True` to get the full `DigitizeResult` (samples, calibration, provenance) instead.
    """
    crop = _asset(paper, fig.crop_png)
    work = Path(out_dir) if out_dir is not None else crop.parent
    work.mkdir(parents=True, exist_ok=True)
    text = caption if caption is not None else (fig.caption or "")
    key = cell_key or f"{paper.sha256[:12]}/{fig.id}/{target.outcome_key}"

    view = FigureView(crop, work_dir=work)
    core = _cv_core(crop)

    # --- path D: read-outs (several model x variant samples)
    readouts: list[ReadOut] = []
    samples: list[RouteSample] = []
    plan = _readout_plan(models, n_readouts)
    for spec in plan:
        suffix = f"/{spec.sample}" if spec.resample else ""
        reading = read_out(client, crop, text, target, spec.model, spec.variant,
                           sample=spec.sample, view=view,
                           cell_key=f"{key}/D/{spec.variant}{suffix}")
        readouts.append(reading)
        samples.extend(_samples_from_readout(reading))

    # --- path C: VLM coordinates + CV snap
    primary = models[0] if models else "claude-opus-5"
    coord = coords(client, crop, text, target, primary, view=view, cell_key=f"{key}/C")
    cal, cal_source, cal_why = _choose_calibration(core, coord)
    samples.extend(_samples_from_coords(coord, core, cal, cal_source))

    # --- path B: raster CV, matched by the nearest VLM coordinate
    samples.extend(_samples_from_raster(coord, core, cal, cal_source))

    # --- path A: vector-exact
    vec_samples, vec_info = _samples_from_vector(paper, fig, coord, core, readouts)
    samples.extend(vec_samples)

    _, tick_spacing = _tick_stats(cal)
    axis_range, axis_range_source = _axis_range(cal, core)
    px_units = abs(pixel_resolution(cal)) if cal is not None else 0.0
    _corroborate_vector_whiskers(samples, px_units)

    # --- overlay verify: drop what the model says is misplaced, then recompute
    labels = {"A": target.group_a_label or "group A", "B": target.group_b_label or "group B"}
    overlay_path = ""
    verify_log: list[dict[str, Any]] = []
    if verify:
        for iteration in range(1, MAX_OVERLAY_ITERATIONS + 1):
            marks, owners = _overlay_marks(samples, cal, core, labels)
            if not marks:
                break
            out_png = work / f"{fig.id}.overlay{iteration}.png"
            verdicts = overlay_verify(client, crop, marks, target, primary, out_png=out_png,
                                      cell_key=f"{key}/verify{iteration}")
            overlay_path = verdicts.overlay_path or str(out_png)
            bad = [v for v in verdicts if v.bad]
            verify_log.append({"iteration": iteration, "overlay": overlay_path,
                               "n_marks": len(marks), "call_ids": list(verdicts.call_ids),
                               "cost_usd": verdicts.cost_usd,
                               "verdicts": [v.to_dict() for v in verdicts]})
            if not bad:
                break
            dropped = 0
            for v in bad:
                for idx in owners[v.number - 1]:
                    if not samples[idx].dropped:
                        samples[idx].dropped = True
                        samples[idx].drop_reason = f"overlay verify: {v.verdict} — {v.reason}"
                        dropped += 1
            verify_log[-1]["dropped_samples"] = dropped
            if not dropped or not any(s.usable for s in samples):
                break

    provenance = _base_provenance(fig, core, cal, cal_source, cal_why, coord, vec_info,
                                  verify_log, target)
    provenance["axis_range"] = axis_range
    provenance["axis_range_source"] = axis_range_source
    provenance.update(_late_window_provenance(target, samples, plan))
    candidates = _build_candidates(samples, target=target, fig=fig, paper=paper, source=source,
                                   dataset=dataset, core=core, cal=cal, crop=crop,
                                   overlay_path=overlay_path, base=provenance,
                                   axis_range=axis_range, tick_spacing=tick_spacing,
                                   px_units=px_units, readouts=readouts,
                                   want_uncertainty=want_uncertainty, labels=labels)
    cost = sum(r.cost_usd for r in readouts) + coord.cost_usd + sum(
        entry.get("cost_usd", 0.0) for entry in verify_log)
    if result:
        return DigitizeResult(candidates=candidates, samples=samples, calibration=cal,
                              overlay_path=overlay_path, cost_usd=cost, provenance=provenance)
    return candidates


def _late_window_provenance(target: TargetSpec, samples: Sequence[RouteSample],
                            plan: Sequence[ReadoutSpec]) -> dict[str, Any]:
    """Which time-series rule was actually APPLIED, and what x the models say they read at.

    The configured rule is only an instruction; what matters downstream is whether it applied at
    all (it does not on a non-time-series figure) and which point the read-outs actually landed on.
    """
    x_reads = sorted({str(s.extra.get("x_read", "")).strip()
                      for s in samples if str(s.extra.get("x_read", "")).strip()})
    applied = target.late_window_sd if target.x_hint else "not_a_time_series"
    return {
        "late_window_rule": applied,
        "late_window_rule_configured": target.late_window_sd,
        "late_window_x_read": x_reads,
        "late_window_x_agrees": len(x_reads) <= 1,
        "readout_plan": [spec.to_dict() for spec in plan],
    }


def _asset(paper: PaperRecord, rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else Path(paper.out_dir) / rel


def _base_provenance(fig: FigureRegion, core: _Core, cal: AxisCalibration | None, cal_source: str,
                     cal_why: str, coord: CoordReadout, vec_info: dict[str, Any],
                     verify_log: list[dict[str, Any]], target: TargetSpec) -> dict[str, Any]:
    return {
        "figure_id": fig.id, "figure_kind": fig.kind, "crop_dpi": fig.crop_dpi,
        "sent_scale": coord.scale,
        "axes": core.axes.to_dict(),
        "tick_rows": [round(r, 3) for r in core.tick_rows],
        "ocr_status": core.ocr_status,
        "ocr_ticks": [(lb.text, lb.value, round(lb.center[1], 2)) for lb in core.labels],
        "cal": cal.to_dict() if cal is not None else None,
        "cal_source": cal_source, "cal_note": cal_why,
        "pixel_resolution": abs(pixel_resolution(cal)) if cal is not None else None,
        "bars": [b.to_dict() for b in core.bars],
        "markers": [m.to_dict() for m in core.markers[:24]],
        "narrow_bar": any(b.narrow for b in core.bars),
        "vector": vec_info,
        "vector_warnings": vec_info.get("warnings", []),
        "overlay_iterations": verify_log,
        "x_hint": target.x_hint,
        "prompt_version": PROMPT_VERSION,
    }


def _source_kind(core: _Core, target: TargetSpec, source: Source | None) -> SourceKind | None:
    if source is not None and source.kind is not None:
        return source.kind
    if target.quantity == "box":
        return SourceKind.figure_box
    if target.quantity == "points":
        return SourceKind.figure_points
    return SourceKind.figure_bar if core.is_bar_chart else SourceKind.figure_line


def _group_n(dataset: DatasetSpec | None, group: str) -> tuple[int | None, str]:
    if dataset is None:
        return None, ""
    spec = dataset.group_a if group == "A" else dataset.group_b
    return spec.n, spec.n_evidence


def _unit(target: TargetSpec, readouts: Sequence[ReadOut]) -> str:
    """The mapper's unit wins; otherwise the first unit any read-out managed to read."""
    if target.unit_hint:
        return target.unit_hint
    return next((r.unit for r in readouts if r.unit), "")


def _build_candidates(samples: list[RouteSample], *, target: TargetSpec, fig: FigureRegion,
                      paper: PaperRecord, source: Source | None, dataset: DatasetSpec | None,
                      core: _Core, cal: AxisCalibration | None, crop: Path, overlay_path: str,
                      base: dict[str, Any], axis_range: float, tick_spacing: float,
                      px_units: float, readouts: Sequence[ReadOut], want_uncertainty: bool,
                      labels: dict[str, str]) -> list[Candidate]:
    kind = _source_kind(core, target, source)
    mapper_type = source.error_bar_type if source is not None else DispersionType.UNKNOWN
    legend_text = " ".join(r.legend_says for r in readouts if r.legend_says)
    legend_type = _legend_dispersion(legend_text)
    unit = _unit(target, readouts)
    page = source.page if source is not None and source.page else fig.page
    locator = (source.locator if source is not None and source.locator
               else (target.panel_hint or fig.label or fig.id))

    out: list[Candidate] = []
    for group in GROUPS:
        mine = [s for s in samples if s.group == group]
        if not mine:
            continue
        for s in mine:
            out.append(_candidate(
                s.mean, s.error, group=group, sample=s, target=target, fig=fig, paper=paper,
                dataset=dataset, source=source, kind=kind, mapper_type=mapper_type, unit=unit,
                page=page, locator=locator, crop=crop, overlay_path=overlay_path,
                extractor_id=s.extractor_id, sigma=s.sigma, status=_sample_status(s),
                notes=("; ".join(x for x in (s.notes, s.drop_reason) if x)),
                provenance={**base, "route_sample": s.to_dict(),
                            "tool_calls": summarize_tool_calls(s.tool_calls),
                            "dropped": s.dropped},
                call_id=(s.call_ids[-1] if s.call_ids else ""), model=s.model))

        live = [s for s in mine if s.usable]
        if not live:
            status, reason = _absent_status(mine)
            out.append(_candidate(
                None, None, group=group, sample=None, target=target, fig=fig, paper=paper,
                dataset=dataset, source=source, kind=kind, mapper_type=mapper_type, unit=unit,
                page=page, locator=locator, crop=crop, overlay_path=overlay_path,
                extractor_id="digitize:ensemble", sigma=None, status=status, notes=reason,
                provenance={**base, "needs_review": True, "needs_review_reason": reason,
                            "per_route": [s.to_dict() for s in mine],
                            "dropped_samples": [s.to_dict() for s in mine if s.dropped],
                            "tool_calls": _aggregate_tool_calls(mine)},
                call_id=_verify_call_id(base), model=""))
            continue

        live, zero_notes = _drop_zero_confidence(live)
        means = [s.mean for s in live]
        errors = [s.error for s in live if s.error is not None]
        mean, mad_sigma = ensemble_stats(means)
        error = statistics.median(errors) if errors else None
        agreement = dual_tolerance(means, errors, axis_range=axis_range,
                                   tick_spacing=tick_spacing, px_units=px_units)
        sigma = None
        if want_uncertainty:
            floor = px_units
            if cal is not None:
                floor = max(floor, abs(cal.rmse) * px_units)
            sigma = max(mad_sigma, floor)
        conflict = (legend_type is not None and mapper_type != DispersionType.UNKNOWN
                    and legend_type != mapper_type)
        reasons = list(agreement["reasons"])
        reasons += [n for n in zero_notes if "excluded" not in n]
        if conflict:
            reasons.append(f"the figure's legend reads {legend_type.value} but the mapper recorded "
                           f"{mapper_type.value}")
        status = _ensemble_status(agreement["agrees"], conflict, zero_notes)
        provenance = {
            **base,
            "legend_says": legend_text,
            "legend_dispersion": legend_type.value if legend_type else None,
            "mapper_dispersion": mapper_type.value if mapper_type else None,
            "per_route": [s.to_dict() for s in mine],
            "route_values": {s.extractor_id: {"mean": s.mean, "error": s.error} for s in live},
            "snap_confidences": {s.extractor_id: s.snap_conf for s in mine
                                 if s.snap_conf is not None},
            "n_routes": len(live), "mad_sigma": mad_sigma,
            "agreement": agreement,
            "zero_confidence_snaps": zero_notes,
            "resampled_routes": [s.extractor_id for s in live if s.sample > 0],
            "tool_calls": _aggregate_tool_calls(mine),
            "needs_review": status == "ambiguous",
            "needs_review_reason": "; ".join(reasons),
            "dropped_samples": [s.to_dict() for s in mine if s.dropped],
        }
        out.append(_candidate(
            mean, error, group=group, sample=None, target=target, fig=fig, paper=paper,
            dataset=dataset, source=source, kind=kind, mapper_type=mapper_type, unit=unit,
            page=page, locator=locator, crop=crop, overlay_path=overlay_path,
            extractor_id="digitize:ensemble", sigma=sigma, status=status,
            notes="; ".join(reasons), provenance=provenance,
            call_id=_verify_call_id(base), model=live[0].model or ""))
    return out


def _ensemble_status(agrees: bool, legend_conflict: bool, zero_notes: Sequence[str]) -> str:
    """`found` only when the routes agree, the legend does not contradict the mapper, AND no
    zero-confidence snap had to be kept as a vote (a snap that found no ink is not evidence)."""
    kept_zero = any("kept" in n for n in zero_notes)
    return "found" if agrees and not legend_conflict and not kept_zero else "ambiguous"


def _drop_zero_confidence(live: list[RouteSample]) -> tuple[list[RouteSample], list[str]]:
    """A pixel route whose snap found no ink at all is not a vote — it is a miss.

    `snap_horizontal_edge` returns confidence 0 when the window is off-image or blank, so the row
    it hands back is the row we asked about, unrefined. Excluded from the median whenever at least
    two other samples survive; kept (and flagged) when dropping it would leave us with nothing.
    """
    zero = [s for s in live if s.route != "D" and s.snap_conf == 0.0]
    if not zero:
        return live, []
    kept = [s for s in live if not any(s is z for z in zero)]
    if len(kept) < 2:
        return live, [f"{s.extractor_id} snapped with zero confidence (kept: too few routes left)"
                      for s in zero]
    return kept, [f"{s.extractor_id} excluded: snap found no ink (confidence 0)" for s in zero]


def _absent_status(mine: Sequence[RouteSample]) -> tuple[str, str]:
    """Why a group has no value — and, crucially, whether that means the datum is not on the page.

    `not_on_these_pages` is a claim about the PAPER: the models looked and the quantity is not in
    this figure. Losing every route to overlay verification is a claim about US: the datum is there,
    we could not read it reliably. That is `ambiguous`, and conflating the two would let a failed
    read silently exclude a study from the meta-analysis.
    """
    if not mine:
        return "ambiguous", "no route sample was produced for this group"
    if any(s.dropped for s in mine):
        dropped = [s.extractor_id for s in mine if s.dropped]
        return "ambiguous", ("every usable route sample was dropped by overlay verification "
                             f"({', '.join(dropped)}); the datum is on the page but we could not "
                             f"read it reliably")
    if all(s.status == "not_on_these_pages" for s in mine):
        return "not_on_these_pages", "every route reported this group is not plotted in this figure"
    return "ambiguous", "no route produced a value for this group"


def _aggregate_tool_calls(mine: Sequence[RouteSample]) -> list[dict[str, Any]]:
    """Every tool call behind a group's routes, tagged with the route that made it."""
    out: list[dict[str, Any]] = []
    for s in mine:
        for call in summarize_tool_calls(s.tool_calls):
            out.append({"route": s.extractor_id, **call})
    return out


def _sample_status(s: RouteSample) -> str:
    """A route sample is `found` when it produced a value, else it inherits the model's verdict."""
    if s.mean is not None:
        return "ambiguous" if s.status == "ambiguous" else "found"
    return "not_on_these_pages" if s.status == "not_on_these_pages" else "ambiguous"


def _verify_call_id(base: dict[str, Any]) -> str:
    log = base.get("overlay_iterations") or []
    for entry in reversed(log):
        ids = entry.get("call_ids") or []
        if ids:
            return ids[-1]
    return ""


def _candidate(mean: float | None, error: float | None, *, group: str, sample: RouteSample | None,
               target: TargetSpec, fig: FigureRegion, paper: PaperRecord,
               dataset: DatasetSpec | None, source: Source | None, kind: SourceKind | None,
               mapper_type: DispersionType, unit: str, page: int, locator: str, crop: Path,
               overlay_path: str, extractor_id: str, sigma: float | None, status: str,
               notes: str, provenance: dict[str, Any], call_id: str, model: str) -> Candidate:
    n, n_quote = _group_n(dataset, group)
    return Candidate(
        candidate_id=f"{dataset.dataset_id if dataset else fig.id}:{target.outcome_key}:"
                     f"{group}:{extractor_id}",
        paper_id=paper.sha256,
        dataset_id=dataset.dataset_id if dataset else "",
        outcome_key=target.outcome_key,
        kind="group_stats", group=group, status=status, source_kind=kind,
        n=n, n_quote=n_quote, mean=mean,
        dispersion_value=(abs(error) if error is not None else None),
        dispersion_type=mapper_type, unit=unit,
        raw_value_semantics="unknown",
        analysis_metric=(source.analysis_metric if source is not None else "unknown"),
        error_bar_scope=(source.error_bar_scope if source is not None else "unknown"),
        page=page, locator=locator,
        crop_path=str(crop), overlay_path=overlay_path,
        pixel_provenance=provenance, sigma=sigma,
        route="figure", model=model, prompt_version=PROMPT_VERSION,
        llm_call_id=call_id, extractor_id=extractor_id, notes=notes)
