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
from ..models import (Candidate, DatasetSpec, DigitizeSettings, DispersionType, Source,
                      SourceKind)
from .calibrate import AxisCalibration, fit_axis, pair_ticks, pixel_resolution, \
    px_to_value, value_to_px
from .overlay import draw_overlay
from .cv import (Axes, Bar, Marker, MIN_BAR_WIDTH_PX, detect_bars, detect_markers, find_axes,
                 find_cap_ends, find_tick_marks, load_color, load_gray, ocr_tick_labels,
                 snap_horizontal_edge, snap_window_for)
from .vector import VectorScene, calibrate_from_scene, snap_to_vector, vector_candidates, \
    whisker_ends
from .vlm import (READOUT_VARIANTS, CoordReadout, FigureView, PROMPT_VERSION, ReadOut,
                  TargetSpec, coords, overlay_verify, read_out, summarize_tool_calls)

__all__ = ["digitize", "RouteSample", "ReadoutSpec", "DigitizeResult", "ensemble_stats",
           "dual_tolerance", "resolve_arms", "ROUTE_LABELS", "CalibrationChoice",
           "CAL_PREFERENCE"]

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
    one_sided: str | None = None                 # "up"/"down" when only one whisker arm is drawn
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


#: an "arm" shorter than this fraction of the other one is not the far end of a symmetric error
#: bar: it is the marker's own edge, the bar top, or the axis line the cap walk ran into
_ONE_SIDED_RATIO = 0.5


def resolve_arms(up: float | None, down: float | None, floor: float = 0.0
                 ) -> tuple[float | None, str | None]:
    """`(half-length, one_sided)` from the two arm lengths of an error bar.

    **This is where a one-armed whisker used to be halved.** Averaging the two arms is right only
    when both are really arms; when a figure draws the whisker on ONE side (common when two series
    overlap — Bock 2005 draws it up for the old group and down for the young), the "other arm" is
    whatever the cap search stopped at, usually a pixel or two away, and the average came out at
    half the true SD. On the first live run that turned an 11.0-unit SD into 6.0 and made four
    routes that agreed about the means look like a disagreement (task 13-14 report, limitation 7a).

    So an arm counts only if it is longer than `floor` (in the same units as the arms — pass two
    pixels' worth) and at least `_ONE_SIDED_RATIO` of the other. What survives alone IS the
    half-length, and `one_sided` says which side it was read from.
    """
    arms = {"up": up, "down": down}
    live = {side: abs(v) for side, v in arms.items() if v is not None and abs(v) > floor}
    if not live:
        return None, None
    if len(live) == 1:
        side, value = next(iter(live.items()))
        return value, side
    longest = max(live, key=lambda s: live[s])
    shortest = min(live, key=lambda s: live[s])
    if live[shortest] < _ONE_SIDED_RATIO * live[longest]:
        return live[longest], longest
    return statistics.mean(live.values()), None


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
    scale_note: str = "linear"          # linear | log | scale_ambiguous | no_fit
    band_retried: bool = False          # the OCR label band had to be re-cut (P4)

    @property
    def is_bar_chart(self) -> bool:
        return bool(self.bars)


#: a scale wins only when the loser's residual is at least twice its own; otherwise the ticks do
#: not say which scale the axis is drawn on, and a log axis read as linear is wrong by an order of
#: magnitude with a perfect-looking fit (critique §1 P1, miss 2)
_SCALE_RMSE_RATIO = 0.5
_MIN_TICKS_FOR_SCALE = 3        # two ticks fit BOTH scales exactly: they carry no scale evidence


def fit_best_scale(pairs: Sequence[tuple[float, float]], axis: str = "y"
                   ) -> tuple[AxisCalibration | None, str]:
    """`(calibration, scale note)` — fit a linear AND a log axis and keep the better-supported one.

    Nothing in the raster path used to infer the scale: `fit_axis` was called with the default
    "linear" and a log-scaled figure was read an order of magnitude out, with a residual that
    looked perfect. Both scales are fitted here; the winner keeps the fit only when the loser's
    pixel residual is at least twice as large, otherwise the note is `scale_ambiguous` and the
    caller treats the calibration as an unconfirmed single witness.
    """
    fits: dict[str, AxisCalibration] = {}
    for scale in ("linear", "log"):
        try:
            fits[scale] = fit_axis(list(pairs), scale=scale, axis=axis)   # type: ignore[arg-type]
        except ValueError:
            continue
    if "log" not in fits:
        # a log axis cannot be fitted through a non-positive tick, and ONE mis-signed label is
        # enough to hide a log axis behind a linear fit for good (tesseract reads "10" as "-10"
        # about as often as it drops a minus). Ask the positive ticks on their own, and keep the
        # log answer only if it wins on them decisively.
        recovered = _log_without_nonpositive_ticks(pairs, axis)
        if recovered is not None:
            return recovered, "log"
    if not fits:
        return None, "no_fit"
    if len(fits) == 1:
        scale, cal = next(iter(fits.items()))
        return cal, scale
    if len(pairs) < _MIN_TICKS_FOR_SCALE:
        return fits["linear"], "linear"       # too few ticks to tell; linear is the honest prior
    win = min(fits, key=lambda sc: fits[sc].rmse)
    lose = next(sc for sc in fits if sc != win)
    rmse_win, rmse_lose = abs(fits[win].rmse), abs(fits[lose].rmse)
    if rmse_lose <= 0.0:
        return fits[win], "scale_ambiguous"
    if rmse_win / rmse_lose < _SCALE_RMSE_RATIO:
        return fits[win], win
    return fits[win], "scale_ambiguous"


def _log_without_nonpositive_ticks(pairs: Sequence[tuple[float, float]], axis: str
                                   ) -> AxisCalibration | None:
    """The log fit through the positive ticks, when the ticks it drops were the wrong ones."""
    positive = [(p, v) for p, v in pairs if v > 0]
    if len(positive) < _MIN_TICKS_FOR_SCALE or len(positive) == len(pairs):
        return None
    try:
        as_log = fit_axis(positive, scale="log", axis=axis)                # type: ignore[arg-type]
        as_linear = fit_axis(positive, scale="linear", axis=axis)          # type: ignore[arg-type]
    except ValueError:
        return None
    if abs(as_linear.rmse) <= 0.0:
        return None
    if abs(as_log.rmse) / abs(as_linear.rmse) >= _SCALE_RMSE_RATIO:
        return None
    as_log.dropped = list(as_log.dropped) + [(p, v) for p, v in pairs if v <= 0]
    return as_log


def _cv_core(crop_png: Path, prefer_markers: bool = False) -> _Core:
    """The shared deterministic pass every pixel route reuses (axes, ticks, OCR, LS fit)."""
    gray = load_gray(crop_png)
    colour = load_color(crop_png)
    axes = find_axes(gray)
    rows = find_tick_marks(gray, axes).get("left", []) if axes.y_axis_x is not None else []
    labels = ocr_tick_labels(gray, axes, side="left", ticks=rows or None)
    status = str(getattr(labels, "status", "missing"))
    pairs = pair_ticks(list(labels), rows, axis="y")
    cal: AxisCalibration | None = None
    scale_note = "no_fit"
    if len(pairs) >= 2:
        cal, scale_note = fit_best_scale(pairs, axis="y")
    try:
        bars = detect_bars(colour, axes)
    except Exception:                                         # pragma: no cover - defensive
        bars = []
    markers: list[Marker] = []
    # miss 8: a line plot whose points `detect_bars` reported as 4-px "bars" never reached
    # `detect_markers`, so route B was silently absent from the figure it should read best. Run
    # both detectors whenever the bars are all narrow or the mapper says this is a line/point plot.
    if not bars or prefer_markers or all(b.narrow for b in bars):
        try:
            markers = detect_markers(colour, axes)
        except Exception:                                     # pragma: no cover - defensive
            markers = []
    return _Core(gray=gray, colour=colour, axes=axes, tick_rows=rows, labels=labels, cal=cal,
                 ocr_status=status, bars=bars, markers=markers,
                 scale_note=scale_note,
                 band_retried=bool(getattr(labels, "band_retried", False)))


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


def _model_calibration(readout: CoordReadout) -> tuple[AxisCalibration | None, str]:
    pairs = [(t.y_px, t.value) for t in readout.ticks]
    seen: dict[float, float] = {}
    for px, value in pairs:
        seen.setdefault(round(px, 3), value)
    clean = [(px, value) for px, value in seen.items()]
    if len(clean) < 2:
        return None, "no_fit"
    return fit_best_scale(clean, axis="y")


def _ladder_from_values(values: Sequence[float], rows: Sequence[float]
                        ) -> list[tuple[float, float]] | None:
    """Pair a read-out's tick VALUES with the tick rows the CV pass measured, or `None`.

    The read-outs already report the ladder they read (`tick_labels`) and it costs nothing extra,
    but a list of values is not a calibration: it needs pixels. `find_tick_marks` supplies those,
    and the two are paired only when the counts line up exactly, or when the values are an evenly
    strided subset of the rows (a figure that labels every second tick). Anything else is refused
    rather than guessed — a mis-paired ladder is the failure this witness exists to catch.
    """
    vals = sorted({float(v) for v in values}, reverse=True)    # y: value falls as the row grows
    lines = sorted(float(r) for r in rows)
    if len(vals) < 2 or len(lines) < 2:
        return None
    if len(vals) == len(lines):
        return list(zip(lines, vals))
    if len(vals) < len(lines) and (len(lines) - 1) % (len(vals) - 1) == 0:
        stride = (len(lines) - 1) // (len(vals) - 1)
        picked = lines[::stride]
        if len(picked) == len(vals):
            return list(zip(picked, vals))
    return None


def _readout_calibration(readouts: Sequence[ReadOut], rows: Sequence[float]
                         ) -> tuple[AxisCalibration | None, str]:
    """The ladder the read-outs themselves reported, as a calibration — the free fourth witness."""
    fits: list[AxisCalibration] = []
    for reading in readouts:
        pairs = _ladder_from_values(reading.tick_labels, rows)
        if not pairs:
            continue
        cal, _scale = fit_best_scale(pairs, axis="y")
        if cal is not None:
            fits.append(cal)
    if not fits:
        return None, "no read-out tick ladder could be paired with the detected tick marks"
    best, support = fits[0], 0
    for cal in fits:
        tol = _agreement_tolerance(cal)
        agree = sum(1 for other in fits if _calibrations_agree(cal, other, tol))
        if agree > support or (agree == support and len(cal.ticks) > len(best.ticks)):
            best, support = cal, agree
    return best, (f"{support} of {len(fits)} read-out tick ladder(s) agree "
                  f"({len(best.ticks)} ticks)")


def _agreement_tolerance(cal: AxisCalibration) -> float:
    """max(2 % of this witness's span, half a tick) — the tolerance two mappings agree within."""
    span, spacing = _tick_stats(cal)
    tol = max(_MEAN_TOL_FRACTION * abs(span), 0.5 * abs(spacing))
    return tol if tol > 0 else max(abs(span) * _MEAN_TOL_FRACTION, 1e-9)


def _calibrations_agree(a: AxisCalibration, b: AxisCalibration, tol: float) -> bool:
    """Do two y calibrations read the same value at the same pixel, within `tol` data units?"""
    pixels = [px for px, _ in a.ticks] or [0.0, 100.0]
    lo, hi = min(pixels), max(pixels)
    probes = [lo, (lo + hi) / 2.0, hi]
    try:
        return all(abs(px_to_value(a, p) - px_to_value(b, p)) <= tol for p in probes)
    except Exception:                                         # pragma: no cover - defensive
        return False


#: which witness's numbers are used when several agree — the OCR ladder is the most precise, the
#: PDF's own text layer next, then the model's tick pixels, then the ladder the read-outs reported
CAL_PREFERENCE = ("cv_ocr", "vector", "vlm_ticks", "readout_ticks")
#: a read-out mean this far past the top tick means the ladder is not the one the values were
#: drawn against (Cressman: ticks max 4.0, read-outs 31.3/33.3)
_MAGNITUDE_SLACK = 0.2
#: two read-out means are "the same number" for the magnitude test when they are this close —
#: deliberately loose, because the question is a factor of ten, not the last digit
_READOUT_AGREE_FRACTION = 0.05


@dataclass
class CalibrationChoice:
    """Which y calibration won the witness vote, and how much corroboration it had.

    `status` is the thing every downstream check keys on:

    * `confirmed`      — at least two independent witnesses read the same MAPPING;
    * `single_witness` — one witness only (or an unresolvable log/linear question): usable, but
                         nothing may be auto-accepted on it;
    * `cal_refuted`    — the read-outs agree on values the ladder cannot draw, so the ladder is
                         wrong: the pixel routes get no calibration and the read-outs stand alone;
    * `none`           — no calibration could be built at all.
    """

    cal: AxisCalibration | None = None
    source: str = "none"
    status: str = "none"
    why: str = ""
    witnesses: dict[str, Any] = field(default_factory=dict)
    agreeing: list[str] = field(default_factory=list)
    scale_note: str = ""
    refutation: dict[str, Any] | None = None

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @property
    def usable_for_pixels(self) -> AxisCalibration | None:
        """The calibration the pixel routes may convert with — none once it has been refuted."""
        return None if self.status == "cal_refuted" else self.cal


def _readout_means(readouts: Sequence[ReadOut]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for reading in readouts:
        for row in reading.groups:
            if row.mean is not None and row.group in GROUPS:
                out.setdefault(row.group, []).append(float(row.mean))
    return out


def _magnitude_refutes(cal: AxisCalibration, readouts: Sequence[ReadOut]
                       ) -> dict[str, Any] | None:
    """Do two agreeing read-outs report a value the ladder could not have drawn?

    This is the load-bearing half of "two witnesses": two readers of the same truncated glyph
    agree with each other and are both wrong, and no power-of-ten rule catches Cressman's 8.3x.
    What does catch it is the values: a ladder whose top tick is 4 cannot carry a datum at 33.
    """
    values = sorted(v for _, v in cal.ticks)
    if len(values) < 2:
        return None
    span = values[-1] - values[0]
    tick_max = max(abs(v) for v in values)
    by_group = _readout_means(readouts)
    agreeing = [means for means in by_group.values()
                if len(means) >= 2 and (max(means) - min(means))
                <= _READOUT_AGREE_FRACTION * max(abs(m) for m in means)]
    if not agreeing:
        return None
    biggest = max(abs(m) for means in agreeing for m in means)
    limit = tick_max + _MAGNITUDE_SLACK * abs(span)
    if biggest <= limit:
        return None
    return {"tick_max": tick_max, "tick_span": span, "limit": limit,
            "readout_max_abs_mean": biggest,
            "readout_means": {g: v for g, v in by_group.items()},
            "why": (f"{len(agreeing)} group(s) of read-outs agree on a value of {biggest:.4g}, "
                    f"which the tick ladder (max {tick_max:.4g}, span {span:.4g}) cannot draw")}


def _choose_calibration(core: _Core, coord: CoordReadout | None,
                        cal_vec: AxisCalibration | None = None,
                        readouts: Sequence[ReadOut] = ()) -> CalibrationChoice:
    """Vote four independent witnesses on ONE mapping, and say how corroborated the winner is.

    The witnesses are the OCR tick ladder, the model's own tick pixels (path C), the PDF text
    layer's ladder (route A, hoisted here so it can vote before the pixel routes convert anything)
    and the ladder the read-outs reported paired with the CV tick rows. They are compared as
    MAPPINGS, not as tick values: two readers of the same cut-off glyph agree on "4" and are both
    an order of magnitude out, whereas a mapping comparison asks what value each witness puts at
    the same pixel.
    """
    witnesses: dict[str, AxisCalibration] = {}
    notes: dict[str, str] = {}
    if core.cal is not None:
        witnesses["cv_ocr"] = core.cal
        notes["cv_ocr"] = f"OCR ladder {[v for _, v in core.cal.ticks]}"
    cal_model, model_scale = _model_calibration(coord) if coord is not None else (None, "")
    if cal_model is not None:
        witnesses["vlm_ticks"] = cal_model
        notes["vlm_ticks"] = f"the model's tick pixels ({len(cal_model.ticks)} ticks)"
    if cal_vec is not None:
        witnesses["vector"] = cal_vec
        notes["vector"] = f"the PDF text layer ({len(cal_vec.ticks)} ticks)"
    cal_read, read_note = _readout_calibration(readouts, core.tick_rows)
    if cal_read is not None:
        witnesses["readout_ticks"] = cal_read
    notes["readout_ticks"] = read_note

    record = {name: {"ticks": cal.ticks, "scale": cal.scale, "rmse_px": cal.rmse,
                     "note": notes.get(name, "")}
              for name, cal in witnesses.items()}
    if not witnesses:
        return CalibrationChoice(status="none", why="no y calibration could be built",
                                 witnesses=record)

    # --- the magnitude test runs FIRST, witness by witness: a ladder the agreed read-out values
    # cannot be drawn on is not a candidate, and dropping it lets a surviving witness (Cressman's
    # own read-out ladder, 45..-5) supply the axis instead of the cell losing its calibration
    refutations = {name: _magnitude_refutes(cal, readouts) for name, cal in witnesses.items()}
    refuted = {name: why for name, why in refutations.items() if why is not None}
    for name, why in refuted.items():
        record[name]["refuted"] = why
    survivors = {name: cal for name, cal in witnesses.items() if name not in refuted}
    if not survivors:
        first = sorted(witnesses, key=_rank)[0]
        return CalibrationChoice(cal=witnesses[first], source=first, status="cal_refuted",
                                 why=refuted[first]["why"], witnesses=record,
                                 refutation=refuted[first],
                                 scale_note=core.scale_note if first == "cv_ocr" else "")
    witnesses = survivors

    best: list[str] = []
    for name, cal in witnesses.items():
        tol = _agreement_tolerance(cal)
        cluster = sorted((other for other, cal_b in witnesses.items()
                          if _calibrations_agree(cal, cal_b, tol)),
                         key=lambda n: CAL_PREFERENCE.index(n) if n in CAL_PREFERENCE else 99)
        if len(cluster) > len(best) or (len(cluster) == len(best) and best
                                        and _rank(cluster[0]) < _rank(best[0])):
            best = cluster
    source = best[0]
    cal = witnesses[source]
    scale_note = core.scale_note if source == "cv_ocr" else (
        model_scale if source == "vlm_ticks" else cal.scale)

    choice = CalibrationChoice(cal=cal, source=source, witnesses=record, agreeing=list(best),
                               scale_note=scale_note)
    if len(best) >= 2:
        choice.status = "confirmed"
        choice.why = (f"{len(best)} independent witnesses agree on the mapping "
                      f"({', '.join(best)}); the {source} ladder supplies the numbers")
    else:
        choice.status = "single_witness"
        choice.why = (f"only the {source} ladder calibrates this axis; "
                      f"{'; '.join(v for k, v in notes.items() if k != source and v) or 'no other witness produced one'}")
    if scale_note == "scale_ambiguous":
        choice.status = "single_witness"
        choice.why = (choice.why + "; the ticks do not say whether this axis is linear or "
                                   "logarithmic, so the mapping is unconfirmed")
    if refuted:
        choice.refutation = next(iter(refuted.values()))
        choice.why = (f"{choice.why}; the {', '.join(sorted(refuted))} ladder was refused — "
                      f"{choice.refutation['why']}")
    return choice


def _rank(name: str) -> int:
    return CAL_PREFERENCE.index(name) if name in CAL_PREFERENCE else 99


# ----------------------------------------------------------------------------- read-out plan
def _readout_plan(models: Sequence[str], n_readouts: int) -> list[ReadoutSpec]:
    """Vote within route = modality x model family (amendment G), with no vote counted twice.

    Samples are taken from DISTINCT (model, variant) pairs, **a second model family before a
    second prompt variant**: primary/direct, secondary/direct, primary/ticks-first, then the rest.
    That order is the fix for F2. The adaptive plan stops at two read-outs when they agree, and
    with the old order those two were opus/direct and opus/ticks-first — one family, two prompts,
    which `vote.route_key` correctly counts as ONE voter. The cell could then never be accepted by
    agreement however right it was, and the sonnet read that would have settled it was never
    bought. Two prompts of one model share a failure mode; two models do not.

    Only when every pair is spent does a pair get re-sampled, and such a sample is flagged
    `resample` so the ensemble and task 8 can see that it is a weaker vote: an identical prompt on
    an identical image is the same question asked twice, and under replay it would be a
    byte-identical copy that silently pins the median and deflates the MAD.
    """
    from ..config import MODELS

    primary = models[0] if models else MODELS["primary"]
    families = [primary]
    for model in list(models[1:]) + [MODELS["secondary"]]:
        if model not in families:
            families.append(model)
    others = families[1:]
    first, second, *rest = READOUT_VARIANTS
    pairs = [(primary, first)]
    pairs += [(m, first) for m in others]
    pairs += [(primary, second)]
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
        error, one_sided = row.error_half_length, None
        if row.mean is not None:
            up = None if row.error_upper is None else abs(row.error_upper - row.mean)
            down = None if row.error_lower is None else abs(row.error_lower - row.mean)
            resolved, one_sided = resolve_arms(up, down)
            if error is None:                     # the model gave caps but no half-length
                error = resolved
        if row.error_sides in ("up", "down"):     # the model was asked outright, and answered
            one_sided = row.error_sides
        out.append(RouteSample(
            route="D", group=group, model=reading.model, variant=reading.variant,
            sample=reading.sample,
            mean=row.mean, error=abs(error) if error is not None else None,
            label_read=row.label_read, status=reading.status,
            notes=row.notes, snap_conf=row.confidence,
            call_ids=list(reading.call_ids), tool_calls=list(reading.tool_calls),
            cost_usd=reading.cost_usd / max(1, len(reading.groups)),
            one_sided=one_sided,
            extra={"legend_says": reading.legend_says, "x_read": row.x_read,
                   "tick_labels": list(reading.tick_labels), "unit": reading.unit,
                   "panel": reading.panel, "same_prompt_resample": reading.sample > 0,
                   "error_sides": row.error_sides}))
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
                        cap_bottom: float | None, floor_px: float = 2.0
                        ) -> tuple[float, float | None, str | None]:
    """`(mean, error half-length, one_sided)` for one datum, in data units.

    The floor below which an "arm" is not an arm is two pixels' worth of data units — or half the
    marker's own height when the marker is bigger than that. Bock 2005's route C walked down from
    a 15-px square and stopped on the square's own lower edge, 7.8 px below its centre: a "cap"
    that is inside the marker is the marker (critique acceptance item 9).
    """
    mean = px_to_value(cal, y)
    up = None if cap_top is None else abs(px_to_value(cal, cap_top) - mean)
    down = None if cap_bottom is None else abs(px_to_value(cal, cap_bottom) - mean)
    error, one_sided = resolve_arms(
        up, down, floor=max(2.0, float(floor_px)) * abs(pixel_resolution(cal, y)))
    return mean, error, one_sided


def _marker_floor_px(core: _Core, x: float | None, y: float | None) -> float:
    """Half the height of the marker a datum sits on — the shortest arm that is not its own edge."""
    if x is None or y is None or not core.markers:
        return 2.0
    marker = min(core.markers, key=lambda m: (m.x - x) ** 2 + (m.y - y) ** 2)
    if abs(marker.x - x) > max(3.0 * max(marker.size, 1.0), 24.0):
        return 2.0
    return max(2.0, 0.5 * float(marker.size))


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
            sample.mean, sample.error, sample.one_sided = _values_from_pixels(
                cal, snapped, sample.cap_top_px, sample.cap_bottom_px,
                floor_px=_marker_floor_px(core, x, snapped))
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
        sample.mean, sample.error, sample.one_sided = _values_from_pixels(
            cal, sample.y_px, top, bottom,
            floor_px=_marker_floor_px(core, sample.x_px, sample.y_px))
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


def _vector_scene(paper: PaperRecord, fig: FigureRegion
                  ) -> tuple[VectorScene | None, dict[str, Any]]:
    """The figure's own geometry, built ONCE and early.

    It used to be built inside route A, which runs after `_choose_calibration` — so the PDF's own
    tick ladder, the one witness that is exact when it exists, could never vote on the calibration
    the pixel routes had already used. Building it here lets it.
    """
    info: dict[str, Any] = {"attempted": False}
    if fig.kind not in ("vector", "mixed"):
        info["skipped"] = f"figure kind {fig.kind!r}"
        return None, info
    info["attempted"] = True
    try:
        scene: VectorScene = vector_candidates(paper, fig)
    except Exception as exc:                                  # pragma: no cover - defensive
        info["error"] = f"{type(exc).__name__}: {exc}"
        return None, info
    info["warnings"] = list(scene.warnings)
    info["n_tick_labels"] = len(scene.tick_labels)
    info["n_marks"] = len(scene.marks)
    if not scene.tick_labels:
        info["skipped"] = "no tick-label spans in the figure's text layer"
        return None, info
    return scene, info


def _samples_from_vector(scene: VectorScene | None, info: dict[str, Any],
                         coord: CoordReadout | None, core: _Core, readouts: Sequence[ReadOut]
                         ) -> list[RouteSample]:
    if scene is None or coord is None:
        if coord is None:
            info.setdefault("skipped", "no VLM coordinates")
        return []
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
        sample.mean, sample.error, sample.one_sided = _values_from_pixels(
            cal_vec, snapped[1], sample.cap_top_px, sample.cap_bottom_px)
        sample.sigma = 0.0                                    # exact geometry
        out.append(sample)
    return out


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


# ----------------------------------------------------------------------------- buying calls
def _means_by_group(samples: Sequence[RouteSample]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for s in samples:
        if s.usable and s.mean is not None:
            out.setdefault(s.group, []).append(float(s.mean))
    return out


def model_families(samples: Sequence[RouteSample]) -> list[str]:
    """The distinct model families behind a set of samples, ignoring the routes that have none.

    Route A and route B are code, not models, so they carry no family; route C and every read-out
    do. This is the number `vote.route_key` cannot see, because it reads one `model` field per
    candidate and the ensemble has to pick one.
    """
    from ..verify.vote import model_family

    return sorted({model_family(s.model) for s in samples if s.model})


def _needs_another_readout(samples: Sequence[RouteSample], *, axis_range: float,
                           tick_spacing: float, px_units: float) -> tuple[bool, str]:
    """Is one more vision read-out worth its price? (task 15 §A3, amended by task 16 §R1c)

    Yes when a group has fewer than two usable routes, when the routes that read it disagree about
    the MEAN beyond amendment F's tolerance, or — this is F2 — when every route that answered
    comes from ONE model family. Two prompts of one model that agree are one voter agreeing with
    itself: they share the model's failure modes, `vote.route_key` counts them as a single route,
    and the cell can never be accepted by agreement. A second family is what makes the agreement
    mean something, and it is bought before any adaptive stop.

    A third pass that only confirms two agreeing families buys nothing: the ensemble is already a
    median of routes that agree, and the confidence gate looks at the means.
    """
    means = _means_by_group(samples)
    if not means:
        return True, "no route has produced a value yet"
    for group in GROUPS:
        mine = [s for s in samples if s.group == group and s.usable]
        values = means.get(group)
        if values is None:
            continue                      # a group that is genuinely not plotted is not a reason
        if len(values) < 2:
            return True, f"group {group} has only {len(values)} usable route"
        families = model_families(mine)
        if len(families) < 2:
            return True, (f"group {group} was read by only one model family "
                          f"({', '.join(families) or 'none'}); two prompts of one model share its "
                          f"failure modes and count as one route in the vote")
        agreement = dual_tolerance(values, [], axis_range=axis_range, tick_spacing=tick_spacing,
                                   px_units=px_units)
        if not agreement["mean_agrees"]:
            return True, f"group {group}: {'; '.join(agreement['reasons'])}"
    return False, "the routes agreed across two model families, so no further read-out was bought"


def _overlay_wanted(verify: bool, policy: str, samples: Sequence[RouteSample], *,
                    axis_range: float, tick_spacing: float, px_units: float) -> tuple[bool, str]:
    """Whether to spend the overlay-verification call, and why (task 15 §A3).

    It exists to catch a mark that landed on the wrong datum, and a mark on the wrong datum shows
    up as routes that disagree. When every route agrees and none was dropped there is nothing for
    it to find, so under `on_disagreement` it is not bought.
    """
    if not verify or policy == "never":
        return False, "overlay verification is switched off"
    if policy == "always":
        return True, "overlay verification runs on every figure"
    if any(s.dropped for s in samples):
        return True, "a route sample was already dropped"
    if any(s.route != "D" and s.snap_conf == 0.0 for s in samples):
        return True, "a pixel route snapped with zero confidence"
    needed, reason = _needs_another_readout(samples, axis_range=axis_range,
                                            tick_spacing=tick_spacing, px_units=px_units)
    if needed:
        return True, reason
    return False, "every route agreed and none was dropped"


# ----------------------------------------------------------------------------- entry point
def digitize(client: LLMClient, paper: PaperRecord, fig: FigureRegion, target: TargetSpec, *,
             models: Sequence[str] = ("claude-opus-5",), n_readouts: int = 3,
             want_uncertainty: bool = True, source: Source | None = None,
             dataset: DatasetSpec | None = None, out_dir: str | Path | None = None,
             caption: str | None = None, cell_key: str = "",
             verify: bool = True, result: bool = False,
             settings: DigitizeSettings | None = None) -> list[Candidate] | DigitizeResult:
    """Read `fig` for `target` with every available route and return the `Candidate`s.

    Returns one `Candidate` per (group, route sample) plus one ensemble `Candidate` per group.
    Pass `result=True` to get the full `DigitizeResult` (samples, calibration, provenance) instead.

    Calls are bought, not spent by default (task 15 §A3): `settings.readouts_min` read-outs run
    first, the pixel routes are resolved, and a further read-out is asked for only when the routes
    disagree about a mean. The overlay-verification call is spent under the same rule.
    """
    cfg = settings or DigitizeSettings(readouts_min=min(2, n_readouts), readouts_max=n_readouts)
    n_max = max(0, min(int(cfg.readouts_max), n_readouts))
    n_min = max(0, min(int(cfg.readouts_min), n_max))
    crop = _asset(paper, fig.crop_png)
    work = Path(out_dir) if out_dir is not None else crop.parent
    work.mkdir(parents=True, exist_ok=True)
    text = caption if caption is not None else (fig.caption or "")
    # the `digitize:` prefix is what `canopy.llm.costs.stage_of` attributes to the
    # digitiser when the run reports where its money went
    key = f"digitize:{cell_key or f'{paper.sha256[:12]}/{fig.id}/{target.outcome_key}'}"

    # one deterministic CV pass, shared by the pixel routes AND by the view's `list_regions`
    core = _cv_core(crop, prefer_markers=_wants_markers(target, source))
    view = FigureView(crop, work_dir=work, axes=core.axes, tick_rows=core.tick_rows,
                      tick_labels=core.labels, bars=core.bars, markers=core.markers)

    readouts: list[ReadOut] = []
    samples: list[RouteSample] = []
    plan = _readout_plan(models, n_max)

    def read(spec: ReadoutSpec) -> None:
        suffix = f"/{spec.sample}" if spec.resample else ""
        reading = read_out(client, crop, text, target, spec.model, spec.variant,
                           sample=spec.sample, view=view,
                           cell_key=f"{key}/D/{spec.variant}{suffix}")
        readouts.append(reading)
        samples.extend(_samples_from_readout(reading))

    # --- path D: the first `readouts_min` read-outs
    for spec in plan[:n_min]:
        read(spec)

    # --- path C: VLM coordinates + CV snap
    primary = models[0] if models else "claude-opus-5"
    coord = coords(client, crop, text, target, primary, view=view, cell_key=f"{key}/C")
    # route A's scene is built HERE, before anything converts a pixel, so the PDF's own tick
    # ladder is one of the witnesses the calibration vote sees
    scene, vec_info = _vector_scene(paper, fig)
    cal_vec = calibrate_from_scene(scene, axis="y") if scene is not None else None
    choice = _choose_calibration(core, coord, cal_vec, readouts)
    cal, cal_source, cal_why = choice.cal, choice.source, choice.why
    pixel_cal = choice.usable_for_pixels
    samples.extend(_samples_from_coords(coord, core, pixel_cal, cal_source))

    # --- path B: raster CV, matched by the nearest VLM coordinate
    samples.extend(_samples_from_raster(coord, core, pixel_cal, cal_source))

    _, tick_spacing = _tick_stats(pixel_cal)
    axis_range, axis_range_source = _axis_range(pixel_cal, core)
    if pixel_cal is None and choice.status == "cal_refuted":
        # the ladder is wrong, so nothing derived from it may set a tolerance; the read-outs'
        # own magnitude is what is left, and it is recorded as such
        magnitudes = [abs(m) for means in _readout_means(readouts).values() for m in means]
        axis_range = 2.0 * max(magnitudes) if magnitudes else 0.0
        axis_range_source = "readout_magnitude"
    px_units = abs(pixel_resolution(pixel_cal)) if pixel_cal is not None else 0.0

    # --- path D again, but only while the routes so far do not agree, or come from one family
    bought = 0
    bought_because: list[str] = []
    stop_reason = f"the plan holds no read-out past the first {n_min}"
    for spec in plan[n_min:]:
        needed, reason = _needs_another_readout(samples, axis_range=axis_range,
                                                tick_spacing=tick_spacing, px_units=px_units)
        if not needed:
            stop_reason = reason
            break
        bought_because.append(reason)                 # WHY the money was spent, not why it stopped
        read(spec)
        bought += 1
    else:
        if len(plan) > n_min:
            stop_reason = "every read-out in the plan was spent"
    buy_reason = "; ".join(bought_because) or stop_reason

    # --- path A: vector-exact (after every read-out, so it sees every tick ladder)
    samples.extend(_samples_from_vector(scene, vec_info, coord, core, readouts))
    _corroborate_vector_whiskers(samples, px_units)

    # --- overlay verify: drop what the model says is misplaced, then recompute
    labels = {"A": target.group_a_label or "group A", "B": target.group_b_label or "group B"}
    overlay_path = ""
    verify_log: list[dict[str, Any]] = []
    do_verify, verify_reason = _overlay_wanted(
        verify, cfg.overlay_verify, samples, axis_range=axis_range, tick_spacing=tick_spacing,
        px_units=px_units)
    if do_verify:
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
    else:
        # the model is not asked, but the picture is still drawn: it costs nothing, and it is what
        # a reviewer opens to see where the routes landed
        marks, _ = _overlay_marks(samples, cal, core, labels)
        if marks:
            out_png = work / f"{fig.id}.overlay1.png"
            draw_overlay(crop, marks, out_png)
            overlay_path = str(out_png)

    provenance = _base_provenance(fig, core, choice, coord, vec_info, verify_log, target)
    provenance["axis_range"] = axis_range
    provenance["axis_range_source"] = axis_range_source
    provenance["call_plan"] = {
        "readouts_min": n_min, "readouts_max": n_max, "readouts_run": len(readouts),
        "extra_readouts_bought": bought, "extra_readout_reason": buy_reason,
        "readout_stop_reason": stop_reason,
        "overlay_verify": bool(do_verify), "overlay_verify_reason": verify_reason,
        "list_regions_offered": view.has_regions}
    # the families that actually answered — `vote.route_key` reads ONE model per candidate and the
    # ensemble has to pick one, so without this the vote cannot tell two families from two prompts
    provenance["model_families"] = model_families(samples)
    provenance["readout_families"] = sorted({r.model for r in readouts})
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


def _base_provenance(fig: FigureRegion, core: _Core, choice: CalibrationChoice,
                     coord: CoordReadout, vec_info: dict[str, Any],
                     verify_log: list[dict[str, Any]], target: TargetSpec) -> dict[str, Any]:
    # a REFUTED ladder is never written as `cal`: `verify.figures` reads that key as the axis a
    # value must lie inside, and handing it a ladder we have just disproved would convict a
    # correct value of being off-axis (F1). It is kept beside it, named for what it is.
    cal = choice.usable_for_pixels
    return {
        "figure_id": fig.id, "figure_kind": fig.kind, "crop_dpi": fig.crop_dpi,
        "sent_scale": coord.scale,
        "axes": core.axes.to_dict(),
        "tick_rows": [round(r, 3) for r in core.tick_rows],
        "ocr_status": core.ocr_status,
        "ocr_band_retried": core.band_retried,
        "ocr_ticks": [(lb.text, lb.value, round(lb.center[1], 2)) for lb in core.labels],
        "cal": cal.to_dict() if cal is not None else None,
        "cal_source": choice.source, "cal_note": choice.why,
        "cal_status": choice.status,
        "cal_witnesses": choice.witnesses,
        "cal_agreeing": list(choice.agreeing),
        "cal_scale": choice.scale_note,
        "cal_refuted_ladder": (choice.cal.to_dict()
                               if choice.status == "cal_refuted" and choice.cal else None),
        "cal_refutation": choice.refutation,
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


def _wants_markers(target: TargetSpec, source: Source | None) -> bool:
    """Should `detect_markers` run even if `detect_bars` claimed to find bars? (miss 8)"""
    if target.quantity in ("points", "box"):
        return True
    kind = source.kind if source is not None else None
    return kind in (SourceKind.figure_line, SourceKind.figure_points)


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
        # a route that found no whisker does not get a vote on the whisker's length
        with_error = [s for s in live if s.error is not None]
        errors = [s.error for s in with_error]
        mean, mad_sigma = ensemble_stats(means)
        error, error_mad = ensemble_stats(errors) if errors else (None, 0.0)
        agreement = dual_tolerance(means, errors, axis_range=axis_range,
                                   tick_spacing=tick_spacing, px_units=px_units)
        floor = px_units
        if cal is not None:
            floor = max(floor, abs(cal.rmse) * px_units)
        sigma = max(mad_sigma, floor) if want_uncertainty else None
        # amendment F applies PER QUANTITY (controller ruling, task 15 §B4): routes that agree
        # about the mean and differ about the half-length give an agreed mean and an uncertain
        # dispersion — the dispersion's own sigma carries that disagreement instead of the cell
        # being thrown away.
        dispersion_sigma = None
        if want_uncertainty and errors:
            # the MAD collapses to zero when a majority of the routes happen to agree exactly
            # ([11, 16, 16] has MAD 0), and a disagreement that wide is not zero uncertainty; half
            # the span is the honest floor for "somewhere between the smallest and largest read".
            spread = (max(errors) - min(errors)) if len(errors) > 1 else 0.0
            widen = 0.5 * spread if not agreement["error_agrees"] else 0.0
            dispersion_sigma = max(error_mad, widen, floor)
        conflict = (legend_type is not None and mapper_type != DispersionType.UNKNOWN
                    and legend_type != mapper_type)
        reasons = list(agreement["reasons"])
        reasons += [n for n in zero_notes if "excluded" not in n]
        if conflict:
            reasons.append(f"the figure's legend reads {legend_type.value} but the mapper recorded "
                           f"{mapper_type.value}")
        status = _ensemble_status(agreement["mean_agrees"], conflict, zero_notes)
        dispersion_only = (status == "found" and not agreement["error_agrees"])
        sides = sorted({s.one_sided for s in with_error if s.one_sided})
        if dispersion_only:
            reasons.append(
                f"the routes agree about the mean and disagree about the error half-length "
                f"({', '.join(f'{v:.4g}' for v in sorted(errors))}); the median of the "
                f"{len(errors)} route(s) that found a whisker is used and its own uncertainty is "
                f"widened to {dispersion_sigma:.4g}"
                + (f" (whisker drawn on one side only: {'/'.join(sides)})" if sides else ""))
        provenance = {
            **base,
            "model_families": model_families(mine),
            "legend_says": legend_text,
            "legend_dispersion": legend_type.value if legend_type else None,
            "mapper_dispersion": mapper_type.value if mapper_type else None,
            "per_route": [s.to_dict() for s in mine],
            "route_values": {s.extractor_id: {"mean": s.mean, "error": s.error} for s in live},
            "snap_confidences": {s.extractor_id: s.snap_conf for s in mine
                                 if s.snap_conf is not None},
            "n_routes": len(live), "n_routes_with_error": len(with_error),
            "mad_sigma": mad_sigma, "dispersion_mad_sigma": error_mad,
            "agreement": agreement,
            "mean_agreement": agreement["mean_agrees"],
            "error_agreement": agreement["error_agrees"],
            "one_sided": sides[0] if len(sides) == 1 else (sides or None),
            "zero_confidence_snaps": zero_notes,
            "resampled_routes": [s.extractor_id for s in live if s.sample > 0],
            "tool_calls": _aggregate_tool_calls(mine),
            "needs_review": status == "ambiguous" or dispersion_only,
            "needs_review_kind": ("mean" if status == "ambiguous"
                                  else ("dispersion" if dispersion_only else None)),
            "needs_review_reason": "; ".join(reasons),
            "dropped_samples": [s.to_dict() for s in mine if s.dropped],
        }
        out.append(_candidate(
            mean, error, group=group, sample=None, target=target, fig=fig, paper=paper,
            dataset=dataset, source=source, kind=kind, mapper_type=mapper_type, unit=unit,
            page=page, locator=locator, crop=crop, overlay_path=overlay_path,
            extractor_id="digitize:ensemble", sigma=sigma, status=status,
            notes="; ".join(reasons), provenance=provenance,
            call_id=_verify_call_id(base),
            # NOT `live[0].model`: `vote.route_key` would then stamp the ensemble with one family
            # and the vote could never see that two families agreed inside it. The families are
            # recorded explicitly in provenance, where `confidence` reads them.
            model=("" if len(model_families(mine)) > 1 else (live[0].model or "")),
            dispersion_sigma=dispersion_sigma))
    return out


def _ensemble_status(mean_agrees: bool, legend_conflict: bool, zero_notes: Sequence[str]) -> str:
    """`found` when the routes agree about the MEAN, the legend does not contradict the mapper,
    and no zero-confidence snap had to be kept as a vote (a snap that found no ink is not
    evidence).

    A disagreement about the error half-length alone does NOT make the cell ambiguous: it makes
    the dispersion uncertain, which travels in `dispersion_sigma` and in the review flag
    (controller ruling, task 15 §B4).
    """
    kept_zero = any("kept" in n for n in zero_notes)
    return "found" if mean_agrees and not legend_conflict and not kept_zero else "ambiguous"


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
               notes: str, provenance: dict[str, Any], call_id: str, model: str,
               dispersion_sigma: float | None = None) -> Candidate:
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
        pixel_provenance=provenance, sigma=sigma, dispersion_sigma=dispersion_sigma,
        route="figure", model=model, prompt_version=PROMPT_VERSION,
        llm_call_id=call_id, extractor_id=extractor_id, notes=notes)
