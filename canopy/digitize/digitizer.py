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


#: An error bar is `mean ± half-length`: its two arms are the SAME length by construction, up to
#: how precisely each cap can be located. So two arms are two arms only while they agree, and
#: "agree" means a couple of pixels' worth (the caller's `floor`) or a quarter of the longer arm,
#: whichever is larger — NOT the 2:1 disparity the first cut allowed. A shorter arm outside that
#: tolerance is not half of a symmetric bar; it is whatever the cap walk stopped on (the marker's
#: own edge, the bar top, the axis line, the next series' mark), and averaging it in halves the
#: spread. In `runs/proof` a genuine 7.78-deg arm was averaged with a 4.82-deg non-arm at ratio
#: 0.62 and reported as 6.30.
_ARM_ASYMMETRY_FRACTION = 0.25


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
    pixels' worth) and within `_ARM_ASYMMETRY_FRACTION` of the other. What survives alone IS the
    half-length, and `one_sided` says which side it was read from.

    When the two arms disagree, the LONGER one is the reading. That is not a coin toss: every way
    a cap walk goes wrong stops it EARLY (on the marker's own edge, on the bar it grew out of, on
    a neighbouring series' mark), so the short arm is the suspect one. It is also the conservative
    direction — a larger dispersion shrinks |d| rather than inflating it.
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
    slack = max(abs(float(floor)), _ARM_ASYMMETRY_FRACTION * live[longest])
    if live[longest] - live[shortest] > slack:
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
                         ) -> tuple[AxisCalibration | None, str, int]:
    """The ladder the read-outs reported, as a calibration, and how many of them agreed on it."""
    fits: list[AxisCalibration] = []
    for reading in readouts:
        pairs = _ladder_from_values(reading.tick_labels, rows)
        if not pairs:
            continue
        cal, _scale = fit_best_scale(pairs, axis="y")
        if cal is not None:
            fits.append(cal)
    if not fits:
        return None, "no read-out tick ladder could be paired with the detected tick marks", 0
    best, support = fits[0], 0
    for cal in fits:
        tol = _agreement_tolerance(cal)
        agree = sum(1 for other in fits if _calibrations_agree(cal, other, tol))
        if agree > support or (agree == support and len(cal.ticks) > len(best.ticks)):
            best, support = cal, agree
    return best, (f"{support} of {len(fits)} read-out tick ladder(s) agree "
                  f"({len(best.ticks)} ticks)"), support


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


#: which witness's numbers are used when several agree.
#:
#: The PDF's own text layer goes first, because it is not a measurement: it is the typeset number
#: the publisher drew the tick with, carried at `confidence = 1.0` (`calibrate.py`), and its fit
#: residual on this corpus is 0.003 px against tesseract's 0.20-0.92 px. OCR of a rendered glyph
#: is a guess, and a guess that has already failed by an order of magnitude ("45/35/25/15" read as
#: "4/3/2/1") and returned six unusable labels on one axis. A ladder that is READ outranking a
#: ladder that is PRINTED is backwards in any figure whose PDF still carries its text.
#:
#: This is only a tie-break: `_magnitude_refutes` runs on every witness first, so a stale or
#: off-panel text layer is refuted on the values before the preference is consulted, and
#: `trusted_readout` still hoists a ladder two independent read-outs agree on above all of them.
CAL_PREFERENCE = ("vector", "cv_ocr", "vlm_ticks", "readout_ticks")
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
    """Do agreeing read-outs report values the ladder could not have drawn — in EITHER direction?

    This is the load-bearing half of "two witnesses": two readers of the same truncated glyph
    agree with each other and are both wrong, and no power-of-ten rule catches Cressman's 8.3x.
    What does catch it is the values: a ladder whose top tick is 4 cannot carry a datum at 33.

    The first cut tested only the upper side, and in absolute value — so a ladder misread the
    other way (its labels read as 1/2/3/4 where the data sits at −31, or an axis whose ticks were
    read too small at the bottom) walked straight through. The test is stated on the ladder's own
    SIGNED range now, widened by a fifth of its span at each end, and it refuses only when EVERY
    agreeing group is outside that frame: one group off the end is a bad read of one series, and
    the whole cell being off the end is a bad ladder.
    """
    values = sorted(v for _, v in cal.ticks)
    if len(values) < 2:
        return None
    lo, hi = values[0], values[-1]
    span = hi - lo
    slack = _MAGNITUDE_SLACK * abs(span)
    low_limit, high_limit = lo - slack, hi + slack
    by_group = _readout_means(readouts)
    agreeing = [means for means in by_group.values()
                if len(means) >= 2 and (max(means) - min(means))
                <= _READOUT_AGREE_FRACTION * max(abs(m) for m in means)]
    if not agreeing:
        return None
    every = [m for means in agreeing for m in means]
    if any(low_limit <= m <= high_limit for m in every):
        return None
    biggest = max(every, key=abs)
    side = "above" if biggest > high_limit else "below"
    return {"tick_max": max(abs(v) for v in values), "tick_span": span,
            "tick_low": lo, "tick_high": hi,
            "limit": high_limit if side == "above" else low_limit,
            "low_limit": low_limit, "high_limit": high_limit, "side": side,
            "readout_max_abs_mean": max(abs(m) for m in every),
            "readout_means": {g: v for g, v in by_group.items()},
            "why": (f"{len(agreeing)} group(s) of read-outs agree on values around {biggest:.4g}, "
                    f"{side} everything the tick ladder ({lo:.4g}..{hi:.4g}) can draw")}


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
    cal_read, read_note, read_support = _readout_calibration(readouts, core.tick_rows)
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
        first = sorted(witnesses, key=_rank)[0]                 # nothing survived: report the top
        return CalibrationChoice(cal=witnesses[first], source=first, status="cal_refuted",
                                 why=refuted[first]["why"], witnesses=record,
                                 refutation=refuted[first],
                                 scale_note=core.scale_note if first == "cv_ocr" else "")
    witnesses = survivors

    # when two witnesses disagree and neither has corroboration, the tie cannot be broken by a
    # fixed preference: `cv_ocr` first is what read 45/35/25/15 as 4/3/2/1 in the first place. A
    # ladder that TWO OR MORE read-outs independently reported goes first instead — two models
    # reading the printed labels beat one tesseract pass, which is the whole lesson of F1.
    trusted_readout = "readout_ticks" in witnesses and read_support >= 2
    order = (("readout_ticks", *[n for n in CAL_PREFERENCE if n != "readout_ticks"])
             if trusted_readout else CAL_PREFERENCE)

    def rank(name: str) -> int:
        base = order.index(name) if name in order else len(order)
        # a ladder built from two ticks has no residual and no scale evidence — it reproduces a
        # linear and a log axis exactly and equally, so nothing about it can be checked against
        # itself (`_MIN_TICKS_FOR_SCALE` states the same threshold for the linear/log question).
        # Whatever its provenance, it goes behind every witness that CAN be checked.
        cal_here = witnesses.get(name)
        unchecked = cal_here is None or len(cal_here.ticks) < _MIN_TICKS_FOR_SCALE
        return base + (len(order) + 1 if unchecked else 0)

    best: list[str] = []
    for name, cal in witnesses.items():
        tol = _agreement_tolerance(cal)
        cluster = sorted((other for other, cal_b in witnesses.items()
                          if _calibrations_agree(cal, cal_b, tol)), key=rank)
        if len(cluster) > len(best) or (len(cluster) == len(best) and best
                                        and rank(cluster[0]) < rank(best[0])):
            best = cluster
    source = best[0]
    cal = witnesses[source]
    scale_note = core.scale_note if source == "cv_ocr" else (
        model_scale if source == "vlm_ticks" else cal.scale)

    choice = CalibrationChoice(cal=cal, source=source, witnesses=record, agreeing=list(best),
                               scale_note=scale_note)
    record.setdefault("readout_ticks", {})["support"] = read_support
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
#: how a dispersion that the CODE built rather than the paper stated is named on the row. The SD
#: of one target direction is not the SD of a subject's mean across eight of them: unless the
#: between-direction variance is fully shared it OVERSTATES the denominator and shrinks |d|. The
#: row says so, is capped, and the pooler can exclude it in a sensitivity analysis.
MEAN_OF_POINT_SD = "mean_of_point_sd"


#: the three answers `_categorical_role` can give, and what each one means for the read
CATEGORICAL_GROUPS = "groups"           # the x categories ARE the comparison arms
CATEGORICAL_CONDITIONS = "conditions"   # the outcome is the average across the categories
CATEGORICAL_UNRESOLVED = "unknown"      # nothing said which, so nothing may be averaged

_LABEL_JUNK = __import__("re").compile(r"[^a-z0-9]+")
#: a label this short matches too much to be evidence of anything ("SD", "n", "A")
_MIN_LABEL_CHARS = 3


def _label_key(text: Any) -> str:
    return _LABEL_JUNK.sub("", str(text or "").lower())


def _labels_are_the_same(a: Any, b: Any) -> bool:
    """Do a plotted x category and a protocol group label name the same thing?

    Equal after stripping case and punctuation, or one contained in the other — a figure axis
    says "Elderly" where the protocol says "Elderly adults", and an axis that says "old" is not
    evidence about a group called "older adults" unless one spells the other.
    """
    left, right = _label_key(a), _label_key(b)
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = sorted((left, right), key=len)
    return len(short) >= _MIN_LABEL_CHARS and short in long


def _categorical_role(target: TargetSpec | None, readings: Sequence[Any]
                      ) -> tuple[str, str]:
    """`(role, why)` — are this figure's x categories the CONDITIONS or the GROUPS themselves?

    `x_axis_kind: "categorical"` conflates two opposite figure shapes. On one, each series runs
    across the axis and the review wants the average across it. On the other, the axis IS the
    comparison — one bar per group — and averaging across it computes `(A + B) / 2` for both arms,
    annihilating the contrast and reporting Cohen's d = 0.0 with every route in perfect agreement.
    No number distinguishes them; the CATEGORY NAMES do.

    So the question asked here is the only one that can settle it: *do the categories the readers
    named map onto the groups the protocol is comparing?* If they do, each category is a group's
    own value and must be read as that group's; if they demonstrably do not (and there is more
    than one of them), they are conditions and collapsing is right. Anything else is unresolved,
    and an unresolved cell produces no number rather than a wrong one.

    Evidence, most authoritative first:

    1. `target.categorical_x`, when a caller has stated it outright;
    2. a reader listed categories matching BOTH group labels — the axis carries both arms;
    3. each group's own points are one category matching that group's own label;
    4. each group's `x_read` names its own group's label and not the other's;
    5. some series has two or more categories, none of which names either group.
    """
    if target is not None and target.categorical_x in (CATEGORICAL_GROUPS,
                                                       CATEGORICAL_CONDITIONS):
        return target.categorical_x, f"the caller stated the x categories are {target.categorical_x}"
    labels = {"A": getattr(target, "group_a_label", "") if target else "",
              "B": getattr(target, "group_b_label", "") if target else ""}
    if not any(labels.values()):
        return CATEGORICAL_UNRESOLVED, "the protocol gave no group labels to match categories to"

    # matching categories onto groups is only possible when both groups are named; a single
    # label matching a single category says nothing about what the axis IS
    both_labelled = all(labels.values())
    conditions_seen: list[str] = []
    for reading in readings:
        rows = {g: reading.group(g) for g in GROUPS}
        for group, row in rows.items():
            if row is None:
                continue
            categories = [p.x_label for p in row.points if str(p.x_label or "").strip()]
            hits = {g: [c for c in categories if _labels_are_the_same(c, labels[g])]
                    for g in GROUPS if labels[g]}
            if both_labelled and all(hits.get(g) for g in GROUPS):
                named = ", ".join(sorted({c for cs in hits.values() for c in cs}))
                return CATEGORICAL_GROUPS, (
                    f"a reader listed the x categories as {named!r}, which are the two groups "
                    f"being compared — each category is a group's own value, not a point to "
                    f"average over")
            if both_labelled and len(categories) == 1 and hits.get(group):
                other = [g for g in GROUPS if g != group][0]
                if not _labels_are_the_same(categories[0], labels.get(other, "")):
                    return CATEGORICAL_GROUPS, (
                        f"group {group}'s only x category is {categories[0]!r}, its own group "
                        f"label — the axis puts one point per group")
            if len(categories) >= 2 and not any(hits.get(g) for g in GROUPS if labels[g]):
                conditions_seen.append(
                    f"group {group} spans {len(categories)} categories "
                    f"({', '.join(map(str, categories[:4]))}), none of them a group label")
        reads = {g: str(getattr(rows[g], "x_read", "") or "") for g in GROUPS if rows[g]}
        if both_labelled and len(reads) == 2 and all(reads.values()):
            own = all(_labels_are_the_same(reads[g], labels[g]) for g in GROUPS if labels[g])
            cross = any(_labels_are_the_same(reads[g], labels[o])
                        for g, o in (("A", "B"), ("B", "A")) if labels[o])
            if own and not cross:
                return CATEGORICAL_GROUPS, (
                    f"each group was read at its own category on the x axis "
                    f"({reads['A']!r}, {reads['B']!r})")
    if conditions_seen:
        return CATEGORICAL_CONDITIONS, conditions_seen[0]
    return CATEGORICAL_UNRESOLVED, ("nothing in the readings says whether the x categories are "
                                    "conditions to average across or the groups themselves")


def _same_points(row_a: Any, row_b: Any) -> bool:
    """Did both groups come back with the very same points — the same categories at the same
    heights? Then one series was read twice and the "contrast" between the two averages is zero
    by construction. Two real arms of a comparison do not plot on top of each other."""
    def key(row: Any) -> list[tuple[str, Any]]:
        return [(_label_key(p.x_label), p.mean) for p in row.points]

    if row_a is None or row_b is None or not row_a.points or not row_b.points:
        return False
    return key(row_a) == key(row_b)


def _row_at_own_category(row: Any, label: str) -> Any:
    """The group's row with `mean`/`error_half_length` filled in from its OWN x category.

    On a group chart the reader may put the number in `points` (it was asked for points) rather
    than in `mean`. That single point IS the group's value; nothing is averaged.
    """
    from dataclasses import replace as _replace

    if row.mean is not None or not row.points:
        return row
    mine = [p for p in row.points if _labels_are_the_same(p.x_label, label)]
    chosen = mine[0] if len(mine) == 1 else (row.points[0] if len(row.points) == 1 else None)
    if chosen is None:
        return row
    return _replace(row, mean=chosen.mean,
                    error_half_length=(row.error_half_length if row.error_half_length is not None
                                       else chosen.error_half_length))


def _collapse_points(row: Any) -> tuple[float | None, float | None, int]:
    """`(mean of the points, mean of their half-lengths, how many)` for a categorical x axis."""
    means = [p.mean for p in row.points if p.mean is not None]
    errors = [abs(p.error_half_length) for p in row.points if p.error_half_length is not None]
    if len(means) < 2:
        return None, None, len(means)
    return (statistics.fmean(means),
            statistics.fmean(errors) if errors else None,
            len(means))


#: words with which a reader says a cap was NOT measured — it was hidden, or it was made up
_CAP_UNMEASURED = ("hidden", "obscured", "occluded", "coincident", "shared", "overlap",
                   "inferred", "assumed", "symmetry", "not drawn", "not visible", "invisible",
                   "cut off", "clipped", "estimated", "guessed", "behind")
#: …and words with which it says the opposite, which veto the clause they appear in
_CAP_MEASURED = ("visible", "drawn", "seen", "measured", "read", "clear", "distinct")
_CAP_NOUNS = ("cap", "whisker", "error bar", "errorbar", "arm", "error", "bar")
_CAP_SIDES = {"up": ("upper", "top", "above", "positive"),
              "down": ("lower", "bottom", "below", "negative")}
_CLAUSE_SPLIT = __import__("re").compile(r"[,;.\n]")


def _unmeasured_cap_side(text: str) -> str | None:
    """Which cap the reader's own prose says it did not measure — `"up"`, `"down"`, or None.

    The read-out prompt asks for the cap on an undrawn side to be left null. When the model fills
    it in anyway and then admits so in `notes`, the fabricated cap is what makes a one-armed bar
    look two-armed: in `runs/proof` a reader reported `error_sides: "both"` with
    `error_lower: -29.0` and the note *"lower cap inferred by symmetry"*, and another reported
    `"both"` beside *"upper error cap hidden, half-length inferred from the visible lower arm"*.
    A cap the reader says it did not see is not evidence, whatever the enum beside it says.

    Read clause by clause, because one sentence often reports both states ("upper cap hidden,
    half-length from the visible lower arm"). A clause counts only if it names a side, names a
    cap-like thing, says the cap is absent or invented, and does NOT also say it was seen. If
    both sides come out unmeasured the prose is not usable — a bar with no arms at all is not a
    reading — and None is returned.
    """
    found: set[str] = set()
    for clause in _CLAUSE_SPLIT.split((text or "").lower()):
        if not any(noun in clause for noun in _CAP_NOUNS):
            continue
        if not any(word in clause for word in _CAP_UNMEASURED):
            continue
        if any(word in clause for word in _CAP_MEASURED):
            continue
        sides = [side for side, words in _CAP_SIDES.items()
                 if any(word in clause for word in words)]
        if len(sides) == 1:
            found.add(sides[0])
    return found.pop() if len(found) == 1 else None


def _samples_from_readout(reading: ReadOut, collapse: bool = False,
                          target: TargetSpec | None = None) -> list[RouteSample]:
    role, role_why = (_categorical_role(target, [reading]) if collapse
                      else (CATEGORICAL_CONDITIONS, ""))
    # the guarantee that no cell can report the same mean for both arms out of the same points:
    # when both series come back with identical categories at identical heights, one series was
    # read twice, and averaging either of them is averaging the contrast away
    twinned = collapse and _same_points(reading.group("A"), reading.group("B"))
    out: list[RouteSample] = []
    for group in GROUPS:
        row = reading.group(group)
        if row is None:
            continue
        if collapse and role == CATEGORICAL_GROUPS:
            # the x categories ARE the groups, so this figure is an ordinary group chart and the
            # single category belonging to this group is its value. Nothing is averaged.
            label = (target.group_a_label if group == "A" else target.group_b_label) if target else ""
            resolved = _row_at_own_category(row, label)
            if resolved.mean is None and row.points:
                # the axis is the groups, but none of the categories this reader named can be
                # matched to THIS group — so we cannot say which of them is its value, and
                # picking one would be a guess about the contrast itself
                from dataclasses import replace as _replace
                resolved = _replace(row, notes=(
                    f"{row.notes}; the x categories are the two groups, but none of the "
                    f"{len(row.points)} this reader named ("
                    f"{', '.join(str(p.x_label) for p in row.points[:4])}) can be matched to "
                    f"group {group} ({label!r}), so no value is taken from it").strip("; "))
            row = resolved
            collapse_here = False
        else:
            collapse_here = collapse
        if collapse_here:
            mean, error, n_points = _collapse_points(row)
            if twinned:
                mean, error = None, None
            sample = RouteSample(
                route="D", group=group, model=reading.model, variant=reading.variant,
                sample=reading.sample, mean=mean, error=error,
                label_read=row.label_read, status=reading.status if mean is not None else
                "ambiguous", notes=row.notes, snap_conf=row.confidence,
                call_ids=list(reading.call_ids), tool_calls=list(reading.tool_calls),
                cost_usd=reading.cost_usd / max(1, len(reading.groups)),
                extra={"legend_says": reading.legend_says, "x_read": row.x_read,
                       "tick_labels": list(reading.tick_labels), "unit": reading.unit,
                       "panel": reading.panel, "axis_read": reading.axis_read,
                       "axis_direction_note": reading.axis_direction_note,
                       "collapsed_across_x": mean is not None, "n_points": n_points,
                       "points": [p.to_dict() for p in row.points],
                       "categorical_x_role": role, "categorical_x_role_why": role_why,
                       "dispersion_approximation": MEAN_OF_POINT_SD if error is not None else "",
                       "same_prompt_resample": reading.sample > 0})
            if twinned:
                sample.notes = (sample.notes + "; this reader returned the SAME points for both "
                                "groups, so their averages would be one number reported twice "
                                "and the contrast between them zero by construction; no value "
                                "is taken from it").strip("; ")
            elif mean is None and role == CATEGORICAL_UNRESOLVED:
                sample.notes = (sample.notes + f"; {role_why}, so nothing may be averaged across "
                                "it").strip("; ")
            elif mean is None:
                sample.notes = (sample.notes + "; the reader gave fewer than two points, so the "
                                              "average across the categorical axis could not be "
                                              "formed").strip("; ")
            out.append(sample)
            continue
        error, one_sided = row.error_half_length, None
        unmeasured = _unmeasured_cap_side(row.notes)
        if row.mean is not None:
            up = None if row.error_upper is None else abs(row.error_upper - row.mean)
            down = None if row.error_lower is None else abs(row.error_lower - row.mean)
            if unmeasured == "up":
                up = None
            elif unmeasured == "down":
                down = None
            resolved, one_sided = resolve_arms(up, down)
            if error is None:                     # the model gave caps but no half-length
                error = resolved
        if row.error_sides in ("up", "down"):     # the model was asked outright, and answered
            one_sided = row.error_sides
        if unmeasured is not None:
            # the prose beats the enum when they contradict: `error_sides` is one token the model
            # picked, the note is the model reporting what it could actually SEE
            one_sided = "down" if unmeasured == "up" else "up"
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
                   "error_sides": row.error_sides, "axis_read": reading.axis_read,
                   "unmeasured_cap": unmeasured,
                   "categorical_x_role": role if collapse else "",
                   "categorical_x_role_why": role_why if collapse else "",
                   "axis_direction_note": reading.axis_direction_note}))
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


def _drop_caps_on_other_series(samples: Sequence[RouteSample], core: _Core) -> None:
    """Blank any cap this series' whisker could only have reached through ANOTHER series' mark.

    Overlapping series are the ordinary case in a two-group figure, and a cap search that starts
    at one series' datum and walks toward the other has two ways to come back with the wrong
    number: it stops ON the other mark (a short "arm" that is really the neighbour's glyph), or it
    runs straight past it and stops on the neighbour's own CAP (a long "arm" that is really the
    neighbour's whisker). Both were reproduced on a figure whose half-lengths are known by
    construction, and the first is what `runs/proof` recorded for Bock 2005 Fig. 1: a "lower cap"
    6.4 px past the young group's triangle, averaged with a genuine 7.78-deg upper arm to give
    6.30. The route said so itself — *"lower cap taken as the upper of the coincident cap pair
    near y=729"*.

    So the rule is directional and geometric: walking from this datum toward the other group's
    datum, anything at or beyond the point where the other group's own marker begins is that
    series' ink. It applies only in the column the walk ran down — marks a column apart cannot be
    what a vertical search stopped on.

    Dropping the arm is safe. An error bar is `mean ± half-length`, so the arm on the far side
    reads the same number; the cost of being wrong is one corroborating arm, never the
    measurement. The half-length that survives is recorded as one-sided, which is what it is.
    """
    for sample in samples:
        if sample.y_px is None or sample.x_px is None:
            continue
        for other in samples:
            if other is sample or other.group == sample.group:
                continue
            if other.y_px is None or other.x_px is None:
                continue
            near = max(2.0, _marker_floor_px(core, other.x_px, other.y_px))
            if abs(other.x_px - sample.x_px) > max(4.0, 2.0 * near):
                continue
            toward = 1.0 if other.y_px > sample.y_px else -1.0     # image rows grow downward
            limit = other.y_px - toward * near                     # where the other glyph begins
            for name, attr in (("upper", "cap_top_px"), ("lower", "cap_bottom_px")):
                cap = getattr(sample, attr)
                if cap is None or cap == sample.y_px:
                    continue
                if (cap - sample.y_px) * toward <= 0:              # points away from the other mark
                    continue
                if (cap - limit) * toward < 0:                     # stops short of the other glyph
                    continue
                setattr(sample, attr, None)
                sample.notes = (f"{sample.notes}; the {name} cap is at or past group "
                                f"{other.group}'s own mark (y {other.y_px:.1f}), so it is that "
                                f"series' ink, not this one's whisker").strip("; ")
                sample.extra.setdefault("caps_on_other_series", []).append(name)


def _samples_from_coords(coord: CoordReadout, core: _Core, cal: AxisCalibration | None,
                         cal_source: str) -> list[RouteSample]:
    out: list[RouteSample] = []
    floors: list[tuple[RouteSample, float]] = []
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
        out.append(sample)
        floors.append((sample, _marker_floor_px(core, x, snapped)))
    # every group's datum has to be known before any cap can be judged against it, so the guard
    # and the conversion both run once the loop has seen the whole reading
    _drop_caps_on_other_series(out, core)
    for sample, floor_px in floors:
        if cal is not None:
            sample.mean, sample.error, sample.one_sided = _values_from_pixels(
                cal, sample.y_px, sample.cap_top_px, sample.cap_bottom_px, floor_px=floor_px)
            sample.sigma = _pixel_sigma(cal, sample.y_px)
        else:
            sample.notes = (sample.notes + " no y calibration; pixels only").strip()
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
    floors: list[tuple[RouteSample, float]] = []
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
        sample.snap_conf = 1.0
        out.append(sample)
        floors.append((sample, _marker_floor_px(core, sample.x_px, sample.y_px)))
    _drop_caps_on_other_series(out, core)
    for sample, floor_px in floors:
        sample.mean, sample.error, sample.one_sided = _values_from_pixels(
            cal, sample.y_px, sample.cap_top_px, sample.cap_bottom_px, floor_px=floor_px)
        sample.sigma = _pixel_sigma(cal, sample.y_px)
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


# ----------------------------------------------------------------------------- which axis?
_AXIS_SIDES = ("left", "right", "top", "bottom")
_AXIS_ORIENT = {"y": "y", "vertical": "y", "x": "x", "horizontal": "x"}
#: the printed axis title, which readers quote — everything else they write ("linear", the tick
#: ladder, pixel positions) is commentary and varies wildly in length between models
_QUOTED = __import__("re").compile(r"[\"'\u2018\u2019\u201c\u201d]([^\"'\u2018\u2019\u201c\u201d]{3,})"
                                   r"[\"'\u2018\u2019\u201c\u201d]")


def _axis_norm(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


@dataclass(frozen=True)
class _AxisFeatures:
    """What a free-text axis description positively STATES. Empty means "did not say"."""

    side: str = ""
    orientation: str = ""
    title: str = ""


def _axis_features(text: str) -> _AxisFeatures:
    words = _axis_norm(text).split()
    side = next((w for w in words if w in _AXIS_SIDES), "")
    orientation = ""
    for word in words:
        if word in _AXIS_ORIENT:
            orientation = _AXIS_ORIENT[word]
            break
        if word.endswith("axis") and len(word) > 4:          # "yaxis" after normalisation
            orientation = _AXIS_ORIENT.get(word[0], "")
            if orientation:
                break
    quoted = _QUOTED.search(text or "")
    return _AxisFeatures(side=side, orientation=orientation,
                         title=_axis_norm(quoted.group(1)) if quoted else "")


def _axis_similar(a: str, b: str) -> bool:
    """Do two free-text axis descriptions name the same ladder?

    They are treated as the SAME unless something they BOTH state positively disagrees. That
    default is the whole point. The first cut compared the raw strings with `SequenceMatcher` and
    called anything under 0.6 a different axis — but one model writes `left y-axis, "RMSE (mm)"`
    and another writes the same axis as `Left y-axis, printed title "RMSE (mm)". Linear. Tick
    ladder: 0 (y=899.5 px), 20 (y=654.9), …`, which scores nowhere near 0.6. On the live re-run
    that evicted a correct reader from six of Cressman's eight cells and left two of them with no
    usable route at all, so the paper produced `not_convertible` where the old code produced the
    right answer.

    A false pool costs nothing here — two readers who really are on different axes disagree about
    the VALUE, and the ensemble's own tolerance catches that. A false split destroys a reading.
    """
    from difflib import SequenceMatcher

    left, right = _axis_features(a), _axis_features(b)
    if left.side and right.side and left.side != right.side:
        return False                     # "left y-axis" and "right y-axis" are never one axis
    if left.orientation and right.orientation and left.orientation != right.orientation:
        return False                     # a value read off x is not a value read off y
    if left.title and right.title:
        # both quoted a printed title: compare THOSE, not the commentary around them
        return SequenceMatcher(None, left.title, right.title).ratio() >= 0.6
    return True


def _reconcile_axes(samples: list[RouteSample], target: TargetSpec) -> dict[str, Any]:
    """Refuse to pool two read-outs that answered off DIFFERENT value axes (critique miss 1).

    Cressman Fig. 3b carries a left y-axis in degrees and a right one in per cent of the imposed
    distortion. Both are correct readings of the same marks and they differ by a factor; the
    ensemble median of the two is a number that is on neither axis. OCR only ever reads the left
    gutter, so the raster path was safe by luck — the read-outs and the vector text layer see both.
    """
    named = [s for s in samples if s.route == "D" and str(s.extra.get("axis_read", "")).strip()]
    info: dict[str, Any] = {"axis_reads": sorted({str(s.extra.get("axis_read")) for s in named}),
                            "axis_direction_notes": sorted(
                                {str(s.extra.get("axis_direction_note")) for s in samples
                                 if str(s.extra.get("axis_direction_note", "")).strip()})}
    if len(named) < 2:
        info["axis_agreement"] = "not_enough_readers"
        return info
    clusters: list[list[RouteSample]] = []
    for sample in named:
        for cluster in clusters:
            if _axis_similar(str(sample.extra["axis_read"]), str(cluster[0].extra["axis_read"])):
                cluster.append(sample)
                break
        else:
            clusters.append([sample])
    if len(clusters) < 2:
        info["axis_agreement"] = "agreed"
        return info

    def rank(cluster: list[RouteSample]) -> tuple[int, int]:
        text = _axis_norm(str(cluster[0].extra["axis_read"]))
        hint = _axis_norm(target.unit_hint)
        return (len(cluster), 1 if hint and hint.split()[0] in text else 0)

    keep = max(clusters, key=rank)
    # …and never to the point of leaving a group with nothing. A conflict says two readers are on
    # two ladders; it does not say the minority reader is worthless, and evicting the only reading
    # a group has turns a disagreement into a missing row (which is what happened on the live
    # re-run: two Cressman cells lost every route and the paper came out `not_convertible`).
    survivors = {group: sum(1 for s in samples
                            if s.group == group and s.usable and s in keep)
                 for group in GROUPS}
    dropped: list[str] = []
    for cluster in clusters:
        if cluster is keep:
            continue
        for sample in cluster:
            if survivors.get(sample.group, 0) < 1:
                sample.notes = (sample.notes + f"; read off {str(sample.extra['axis_read'])!r}, "
                                f"which no other reader named — kept because group "
                                f"{sample.group} has no other reading").strip("; ")
                continue
            sample.dropped = True
            sample.drop_reason = (
                f"read off {str(sample.extra['axis_read'])!r}, while the ensemble is on "
                f"{str(keep[0].extra['axis_read'])!r} — two value axes are not one number")
            dropped.append(sample.extractor_id)
    info["axis_agreement"] = "conflict"
    info["axis_kept"] = str(keep[0].extra["axis_read"])
    info["axis_dropped_samples"] = sorted(set(dropped))
    return info


# ----------------------------------------------------------------------------- which series?
_FILL_WORDS = {"open": "open", "unfilled": "open", "hollow": "open", "white": "open",
               "empty": "open", "outline": "open",
               "filled": "filled", "solid": "filled", "black": "filled", "closed": "filled",
               "dark": "filled"}
_SHAPE_WORDS = ("square", "circle", "triangle", "diamond", "star", "cross", "bar")


def _marker_words(text: str) -> tuple[str, str]:
    """`(fill, shape)` a free-text series description resolves to; `""` for what it does not say."""
    words = _axis_norm(text).split()
    fill = next((_FILL_WORDS[w] for w in words if w in _FILL_WORDS), "")
    shape = next((sh for sh in _SHAPE_WORDS if any(w.startswith(sh) for w in words)), "")
    return fill, shape


def _detected_descriptor(marker: Marker) -> tuple[str, str]:
    """`(fill, shape)` for a marker the pixel pass found. `kind` conflates the two, so unpack it."""
    kind = str(marker.kind)
    if kind == "open":
        return "open", ""
    if kind in _SHAPE_WORDS:
        return "filled", kind
    return "", ""


def _descriptors_match(described: tuple[str, str], detected: tuple[str, str]) -> bool:
    """Do a reader's words and a detected marker agree on everything BOTH of them state?"""
    return all(not a or not b or a == b for a, b in zip(described, detected))


def _nearest_marker(core: _Core, x: float | None, y: float | None) -> Marker | None:
    if y is None or not core.markers:
        return None
    if x is None:
        return min(core.markers, key=lambda m: abs(m.y - y))
    return min(core.markers, key=lambda m: (m.x - x) ** 2 + (m.y - y) ** 2)


def _series_identity(samples: Sequence[RouteSample], core: _Core) -> dict[str, Any]:
    """Does each group's described marker exist, is it that group's, and is it the OTHER group's?

    Group assignment rested on one free-text `label_read` from one model ("Elderly: Misaligned
    (open white squares)"), and `_needs_another_readout` looks at means only — so two routes
    agreeing numerically on the WRONG series stopped the plan (critique misses 4 and 5).
    `detect_markers` knows fill and shape; this is where the two are put side by side.

    Three distinct answers, and the first cut only acted on the first:

    * `conflict` — both groups resolve to the SAME marker, so one series is being read twice;
    * `transposed` — each group's described marker is the one found where the OTHER group's value
      was measured. Both readings are of real series; they are on the wrong rows, which is a sign
      flip in the effect size and nothing else catches it;
    * `marker_mismatch` — the described marker is not among the ones the pixel pass found at all.
      These used to be recorded as prose and nothing read them.
    """
    described: dict[str, tuple[str, str]] = {}
    for group in GROUPS:
        texts = [s.label_read for s in samples if s.group == group and s.label_read]
        for text in texts:
            fill, shape = _marker_words(text)
            if fill or shape:
                described[group] = (fill, shape)
                break
    info: dict[str, Any] = {"described": {g: list(v) for g, v in described.items()},
                            "conflict": False, "transposed": False, "marker_mismatch": False,
                            "notes": []}
    if len(described) == 2 and described["A"] == described["B"] and any(described["A"]):
        info["conflict"] = True
        info["notes"].append(
            f"both groups were described as the same marker ({' '.join(w for w in described['A'] if w)}) "
            f"— one of the two series is being read for both groups")

    # what the pixel pass actually found where each group's value was measured
    detected: dict[str, tuple[str, str]] = {}
    for group in GROUPS:
        pixel = next((s for s in samples if s.group == group and s.y_px is not None), None)
        marker = _nearest_marker(core, pixel.x_px if pixel else None,
                                 pixel.y_px if pixel else None)
        if marker is not None:
            detected[group] = _detected_descriptor(marker)
    if detected:
        info["detected"] = {g: list(v) for g, v in detected.items()}
    kinds = {str(m.kind) for m in core.markers}
    if kinds:
        info["detected_marker_kinds"] = sorted(kinds)

    if (len(described) == 2 and len(detected) == 2 and described["A"] != described["B"]
            and any(described["A"]) and any(described["B"])
            and _descriptors_match(described["A"], detected["B"])
            and _descriptors_match(described["B"], detected["A"])
            and not _descriptors_match(described["A"], detected["A"])):
        info["transposed"] = True
        info["notes"].append(
            "each group's described marker is the one found where the OTHER group's value was "
            "measured — the two series are transposed, which flips the sign of the effect")

    if kinds:
        for group, (fill, shape) in described.items():
            if fill == "open" and "open" not in kinds:
                info["marker_mismatch"] = True
                info["notes"].append(
                    f"group {group} was described as an OPEN marker, but every marker the pixel "
                    f"pass found is filled ({', '.join(sorted(kinds))})")
            if shape and shape not in kinds and shape in ("square", "circle", "triangle"):
                info["marker_mismatch"] = True
                info["notes"].append(
                    f"group {group} was described as a {shape}, which the pixel pass did not "
                    f"find among {', '.join(sorted(kinds))}")
    return info


# ----------------------------------------------------------------------------- topology vote
#: dispersions whose two arms are the SAME length by construction, so that one arm of the bar is
#: the whole half-length. An IQR or a min-max whisker is not one of these: its arms genuinely
#: differ and a single arm is not a half-length, so nothing below applies to it.
_SYMMETRIC_DISPERSIONS = (DispersionType.SD, DispersionType.SE, DispersionType.CI95,
                          DispersionType.CI90)


def _dispersion_topology_vote(with_error: Sequence[RouteSample], kind: DispersionType
                              ) -> tuple[list[RouteSample], str]:
    """The routes that may vote on the half-length when they disagree about the WHISKER'S SHAPE.

    Two routes can agree that a bar exists and still be measuring different objects: one reports
    a whisker drawn on one side only and hands back the arm it saw, the other reports two arms and
    hands back their mean. Medianing those is medianing a measurement with a half-measurement, and
    it produces a number no route reported.

    When they conflict, the one-armed reads are the evidence, and the reason is an asymmetry in
    what can go wrong rather than a preference for any route or model:

    * if the whisker really is one-armed, the two-armed read's "other arm" is whatever its cap
      walk stopped on — the marker's own edge, the bar, a neighbouring series' mark — so its
      half-length is contaminated by a non-measurement;
    * if the whisker really is two-armed, the one-armed read measured one arm of a bar whose arms
      are equal by construction, and so has the half-length right anyway.

    A one-armed read is therefore correct under both hypotheses and a two-armed read under only
    one. `runs/proof` shows the cost of not doing this: on Cressman 2010 Fig. 3b one route read a
    one-armed 1.8 (noting the upper cap was hidden — and the pixels agree: the open square's only
    cap is 1.7-2.0 below it) while the other read a two-armed 3.0, and the median 2.4 was
    published with an empty flags column.

    Returns `(routes that vote, why)`; `why` is empty when there was nothing to settle.
    """
    if kind not in _SYMMETRIC_DISPERSIONS:
        return list(with_error), ""
    one_armed = [s for s in with_error if s.one_sided]
    two_armed = [s for s in with_error if not s.one_sided]
    if not one_armed or not two_armed:
        return list(with_error), ""
    sides = "/".join(sorted({s.one_sided for s in one_armed if s.one_sided}))
    return one_armed, (
        f"{len(one_armed)} route(s) read this whisker as drawn on one side only ({sides}) and "
        f"{len(two_armed)} read two arms; a one-armed read is the half-length whichever is true, "
        f"a two-armed read only if two arms are really drawn, so the half-length comes from the "
        f"{len(one_armed)} one-armed read(s)")


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
    the MEAN **or about the error half-length** beyond amendment F's tolerance, or — this is F2 —
    when every route that answered comes from ONE model family. Two prompts of one model that
    agree are one voter agreeing with itself: they share the model's failure modes,
    `vote.route_key` counts them as a single route, and the cell can never be accepted by
    agreement. A second family is what makes the agreement mean something, and it is bought
    before any adaptive stop.

    The dispersion used to be invisible here: `dual_tolerance` was called with an empty error
    list, and it only evaluates the error branch when it is given two or more. So the plan stopped
    on the means alone, on a cell whose spread was in open dispute. `runs/proof` records exactly
    that for Cressman 2010 Fig. 3b — `error_agrees: false`, `error_spread: 1.2` against a
    tolerance of 0.24, a third read-out planned, affordable and never bought — and the two routes'
    half-lengths imply d = -0.2548 or d = -0.1478 depending on which you believe. A spread the
    effect size divides by is exactly as load-bearing as the mean it subtracts, so an adaptive
    stopping rule that covers one and not the other is not a stopping rule.

    A pass that only confirms two agreeing families buys nothing: the ensemble is already a median
    of routes that agree on both quantities.
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
        # the errors that will actually vote in the ensemble — `_build_candidates` medians the
        # half-lengths of exactly these samples, so these are the numbers whose disagreement costs
        errors = [s.error for s in mine if s.error is not None]
        agreement = dual_tolerance(values, errors, axis_range=axis_range,
                                   tick_spacing=tick_spacing, px_units=px_units)
        if not agreement["agrees"]:
            return True, f"group {group}: {'; '.join(agreement['reasons'])}"
    return False, "the routes agreed across two model families, so no further read-out was bought"


def _overlay_wanted(verify: bool, policy: str, samples: Sequence[RouteSample], *,
                    axis_range: float, tick_spacing: float, px_units: float,
                    series_conflict: bool = False) -> tuple[bool, str]:
    """Whether to spend the overlay-verification call, and why (task 15 §A3).

    It exists to catch a mark that landed on the wrong datum, and a mark on the wrong datum shows
    up as routes that disagree. When every route agrees and none was dropped there is nothing for
    it to find, so under `on_disagreement` it is not bought.
    """
    if not verify or policy == "never":
        return False, "overlay verification is switched off"
    if policy == "always":
        return True, "overlay verification runs on every figure"
    if series_conflict:
        # exactly what the overlay call is for: the routes agree on a NUMBER while disagreeing
        # about which series it belongs to, and no amount of numeric agreement settles that
        return True, ("both groups were described as the same marker, so the marks are checked "
                      "against the picture before the numbers are trusted")
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
    # P6's OFF path, decided BEFORE any model call. A figure whose x axis is a set of conditions
    # carries the outcome as the average across that axis; reading one point of it is a different
    # number, not a less precise one, so the cell says it cannot be converted rather than
    # returning something wrong (Heuer & Hegele Fig 2a, acceptance item 15).
    if (source is not None and source.x_axis_kind == "categorical"
            and not target.collapse_across_x and target.categorical_x != CATEGORICAL_GROUPS):
        return _categorical_unsupported(fig, target, paper, source, dataset, crop, result)
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
        samples.extend(_samples_from_readout(reading, collapse=target.collapse_across_x,
                                             target=target))

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
    pixel_samples = _samples_from_coords(coord, core, pixel_cal, cal_source)
    # --- path B: raster CV, matched by the nearest VLM coordinate
    pixel_samples += _samples_from_raster(coord, core, pixel_cal, cal_source)
    cat_role, cat_role_why = ((_categorical_role(target, readouts)) if target.collapse_across_x
                              else (CATEGORICAL_CONDITIONS, ""))
    if target.collapse_across_x and cat_role != CATEGORICAL_GROUPS:
        # a pixel route resolves ONE datum; when the outcome is the mean of every datum on the
        # axis its answer is a different number and must not enter the ensemble. When the x
        # categories ARE the groups there is nothing to average: one datum per group is exactly
        # the quantity, and dropping these routes threw away correct readings.
        for pixel in pixel_samples:
            pixel.dropped = True
            pixel.drop_reason = (
                "this route reads one point, and the outcome is the average across the "
                f"categorical x axis ({cat_role_why})" if cat_role == CATEGORICAL_CONDITIONS else
                "this route reads one point, and nothing has established whether the x categories "
                f"are conditions to average across or the groups themselves ({cat_role_why})")
    samples.extend(pixel_samples)

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

    # --- two readers off two different value axes are not two reads of one number
    axis_info = _reconcile_axes(samples, target)
    series_info = _series_identity(samples, core)

    # --- overlay verify: drop what the model says is misplaced, then recompute
    labels = {"A": target.group_a_label or "group A", "B": target.group_b_label or "group B"}
    overlay_path = ""
    verify_log: list[dict[str, Any]] = []
    do_verify, verify_reason = _overlay_wanted(
        verify, cfg.overlay_verify, samples, axis_range=axis_range, tick_spacing=tick_spacing,
        px_units=px_units, series_conflict=bool(series_info.get("conflict")))
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
    provenance.update(axis_info)
    provenance["series_identity"] = series_info
    provenance["axis_range"] = axis_range
    provenance["axis_range_source"] = axis_range_source
    provenance["call_plan"] = {
        "readouts_min": n_min, "readouts_max": n_max, "readouts_run": len(readouts),
        "extra_readouts_bought": bought, "extra_readout_reason": buy_reason,
        "readout_stop_reason": stop_reason,
        "overlay_verify": bool(do_verify), "overlay_verify_reason": verify_reason,
        "list_regions_offered": view.has_regions}
    # the families that actually answered — `vote.route_key` reads ONE model per candidate and the
    # ensemble has to pick one, so without this the vote cannot tell two families from two prompts.
    # "Answered" means produced a value: a reader that returned no mean corroborates nothing, and
    # counting it here credited an abstention as agreement (`runs/proof`, Bock group A).
    provenance["model_families"] = model_families([s for s in samples if s.usable])
    collapsed = [s for s in samples if s.extra.get("collapsed_across_x")]
    provenance["collapse_across_x"] = bool(target.collapse_across_x)
    provenance["x_axis_kind"] = source.x_axis_kind if source is not None else "unknown"
    provenance["categorical_x"] = target.categorical_x
    provenance["categorical_x_role"] = cat_role if target.collapse_across_x else ""
    provenance["categorical_x_role_why"] = cat_role_why
    if collapsed:
        provenance["collapsed_across_x"] = True
        provenance["n_points"] = min(int(s.extra.get("n_points") or 0) for s in collapsed)
        provenance["dispersion_approximation"] = MEAN_OF_POINT_SD
    provenance["readout_families"] = sorted({r.model for r in readouts})
    provenance.update(_late_window_provenance(target, samples, plan,
                                              x_tick_px=_x_tick_spacing(core)))
    candidates = _build_candidates(samples, target=target, fig=fig, paper=paper, source=source,
                                   dataset=dataset, core=core, cal=cal, crop=crop,
                                   overlay_path=overlay_path, base=provenance,
                                   axis_range=axis_range, tick_spacing=tick_spacing,
                                   px_units=px_units, readouts=readouts,
                                   want_uncertainty=want_uncertainty, labels=labels)
    _refuse_identical_collapse(candidates)
    cost = sum(r.cost_usd for r in readouts) + coord.cost_usd + sum(
        entry.get("cost_usd", 0.0) for entry in verify_log)
    if result:
        return DigitizeResult(candidates=candidates, samples=samples, calibration=cal,
                              overlay_path=overlay_path, cost_usd=cost, provenance=provenance)
    return candidates


def _refuse_identical_collapse(candidates: Sequence[Candidate]) -> None:
    """Neither arm keeps a value when BOTH were averaged across the axis to the same number.

    The last net under `_same_points`, and it does not depend on having seen the points: two
    groups whose averages across a categorical axis agree to the last digit on the mean AND on
    the spread are one series reported twice, not a comparison. Cohen's d would be exactly 0.0,
    every route would agree with every other (they are the same number), the sign check has no
    direction to contradict, and the row would be pooled. Real data does not do this.
    """
    ensembles = {c.group: c for c in candidates if c.extractor_id == "digitize:ensemble"}
    a, b = ensembles.get("A"), ensembles.get("B")
    if a is None or b is None or a.mean is None or b.mean is None:
        return
    if not (a.pixel_provenance.get("collapsed_across_x")
            and b.pixel_provenance.get("collapsed_across_x")):
        return
    if a.mean != b.mean or a.dispersion_value != b.dispersion_value:
        return
    reason = ("both groups came back as the same average across the categorical x axis "
              f"(mean {a.mean}, spread {a.dispersion_value}) — that is one series reported twice, "
              "and the effect size between them would be exactly zero by construction")
    for cand in (a, b):
        cand.mean = None
        cand.dispersion_value = None
        cand.status = "ambiguous"
        cand.notes = "; ".join(x for x in (cand.notes, reason) if x)
        cand.pixel_provenance["identical_collapse_refused"] = True
        cand.pixel_provenance["needs_review"] = True
        cand.pixel_provenance["needs_review_reason"] = reason


_X_PX_RE = __import__("re").compile(r"x\s*(?:=|≈|~|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*px")


def _x_pixels(samples: Sequence[RouteSample]) -> list[float]:
    """Every x position a reader named, in pixels — from `x_px` or from the text it wrote."""
    out: list[float] = []
    for s in samples:
        if s.x_px is not None:
            out.append(float(s.x_px))
            continue
        match = _X_PX_RE.search(str(s.extra.get("x_read", "")))
        if match:
            out.append(float(match.group(1)))
    return out


CATEGORICAL_UNSUPPORTED = "categorical_x_unsupported"


def _categorical_unsupported(fig: FigureRegion, target: TargetSpec, paper: PaperRecord,
                             source: Source, dataset: DatasetSpec | None, crop: Path,
                             result: bool) -> list[Candidate] | DigitizeResult:
    """One ensemble candidate per group saying, with no number in it, why there is no number."""
    reason = (f"the x axis of {source.locator or fig.id} is recorded as categorical, and nothing "
              f"says which kind. If its categories are CONDITIONS the outcome is the average "
              f"across them and reading one point would be a different quantity — turn on "
              f"`collapse_across_categorical_x` to read every point. If its categories are the "
              f"two GROUPS THEMSELVES (one bar per group) there is nothing to average and this is "
              f"an ordinary group chart, which the mapper has mis-classified; averaging across "
              f"that axis would give both groups the same mean and an effect size of exactly zero")
    provenance = {"figure_id": fig.id, "figure_kind": fig.kind, "crop_dpi": fig.crop_dpi,
                  "x_axis_kind": source.x_axis_kind, "collapse_across_x": False,
                  CATEGORICAL_UNSUPPORTED: True, "needs_review": True,
                  "needs_review_reason": reason, "prompt_version": PROMPT_VERSION}
    out = [_candidate(None, None, group=group, sample=None, target=target, fig=fig, paper=paper,
                      dataset=dataset, source=source, kind=source.kind,
                      mapper_type=source.error_bar_type, unit=target.unit_hint,
                      page=source.page or fig.page,
                      locator=source.locator or fig.label or fig.id, crop=crop, overlay_path="",
                      extractor_id="digitize:ensemble", sigma=None, status="ambiguous",
                      notes=reason, provenance=provenance, call_id="", model="")
           for group in GROUPS]
    if result:
        return DigitizeResult(candidates=out, samples=[], calibration=None, overlay_path="",
                              cost_usd=0.0, provenance=provenance)
    return out


def _late_window_provenance(target: TargetSpec, samples: Sequence[RouteSample],
                            plan: Sequence[ReadoutSpec], x_tick_px: float = 0.0) -> dict[str, Any]:
    """Which time-series rule was actually APPLIED, and what x the models say they read at.

    The configured rule is only an instruction; what matters downstream is whether it applied at
    all (it does not on a non-time-series figure) and which point the read-outs actually landed on.

    Agreement is measured in PIXELS, not in prose (critique miss 6). Cressman's two reads are
    "Block 33 (last block, x ≈ 1451 px)" and "Block 33 (…x=1449 px)" — the same block, two pixels
    apart, and a string comparison scored it as a disagreement. Two genuinely different blocks
    phrased identically would have scored as agreement, which is the worse half of the same bug.
    """
    x_reads = sorted({str(s.extra.get("x_read", "")).strip()
                      for s in samples if str(s.extra.get("x_read", "")).strip()})
    pixels = _x_pixels(samples)
    tolerance = max(0.5 * abs(x_tick_px), 4.0)
    if len(pixels) >= 2:
        spread = max(pixels) - min(pixels)
        agrees, how = spread <= tolerance, "pixels"
    else:
        spread, agrees, how = None, len(x_reads) <= 1, "text"
    applied = target.late_window_sd if target.x_hint else "not_a_time_series"
    return {
        "late_window_rule": applied,
        "late_window_rule_configured": target.late_window_sd,
        "late_window_x_read": x_reads,
        "late_window_x_px": sorted(pixels),
        "late_window_x_spread_px": spread,
        "late_window_x_tolerance_px": tolerance,
        "late_window_x_compared": how,
        "late_window_x_agrees": bool(agrees),
        "readout_plan": [spec.to_dict() for spec in plan],
    }


def _x_tick_spacing(core: _Core) -> float:
    """Median gap between x tick marks, in pixels — half of one is "the same x position"."""
    try:
        columns = sorted(find_tick_marks(core.gray, core.axes).get("bottom", []))
    except Exception:                                         # pragma: no cover - defensive
        return 0.0
    gaps = [b - a for a, b in zip(columns, columns[1:]) if b > a]
    if gaps:
        return float(statistics.median(gaps))
    x0, _, x1, _ = core.axes.plot_bbox
    return 0.02 * abs(x1 - x0)


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
    # miss 10: `UNKNOWN` dispersion is a needs_human factory — `confidence._sd_of` returns None for
    # it, no route reaches the figure gate, and the cell fails on a spread the FIGURE stated
    # plainly. When the mapper could not say and the legend says outright, the legend is the
    # evidence; where it came from travels in provenance.
    dispersion_from = "mapper"
    if mapper_type is DispersionType.UNKNOWN and legend_type is not None:
        mapper_type, dispersion_from = legend_type, "legend"
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
        all_errors = [s.error for s in with_error]
        voting, topology_note = _dispersion_topology_vote(with_error, mapper_type)
        errors = [s.error for s in voting]
        mean, mad_sigma = ensemble_stats(means)
        error, error_mad = ensemble_stats(errors) if errors else (None, 0.0)
        # the agreement verdict is computed over EVERY route that found a whisker, not over the
        # subset that supplied the number: segregating the vote settles which reading to use, it
        # does not make the disagreement go away, and the record has to keep showing it
        agreement = dual_tolerance(means, all_errors, axis_range=axis_range,
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
            # The span is the one across ALL the routes that found a whisker, for the same reason
            # the agreement verdict is: choosing which read to believe is not evidence that the
            # others were never made.
            spread = (max(all_errors) - min(all_errors)) if len(all_errors) > 1 else 0.0
            widen = 0.5 * spread if not agreement["error_agrees"] else 0.0
            dispersion_sigma = max(error_mad, widen, floor)
        conflict = (dispersion_from == "mapper" and legend_type is not None
                    and mapper_type != DispersionType.UNKNOWN and legend_type != mapper_type)
        reasons = list(agreement["reasons"])
        reasons += [n for n in zero_notes if "excluded" not in n]
        if conflict:
            reasons.append(f"the figure's legend reads {legend_type.value} but the mapper recorded "
                           f"{mapper_type.value}")
        status = _ensemble_status(agreement["mean_agrees"], conflict, zero_notes)
        dispersion_only = (status == "found" and not agreement["error_agrees"])
        # NO calibration at all is the weakest evidence a figure cell can rest on, and it used to
        # be the quietest: the read-out routes need no ladder to produce a number, so the cell
        # sailed through with `cal_status="none"` and an empty reason. One witness is treated as
        # suspicious; zero must not be treated as fine. The digitiser says so on the row itself,
        # because it is the only stage that knows the ladder was never built.
        no_calibration = base.get("cal_status") == "none"
        if no_calibration:
            reasons.append(
                "no y calibration could be built for this figure, so nothing independent checked "
                "that these values lie on the axis they were read from")
        sides = sorted({s.one_sided for s in with_error if s.one_sided})
        if dispersion_only:
            reasons.append(
                f"the routes agree about the mean and disagree about the error half-length "
                f"({', '.join(f'{v:.4g}' for v in sorted(all_errors))}); the median of the "
                f"{len(errors)} route(s) that supplied one is used and its own uncertainty is "
                f"widened to {dispersion_sigma:.4g}"
                + (f" (whisker drawn on one side only: {'/'.join(sides)})" if sides else ""))
        if topology_note:
            reasons.append(topology_note)
        mine_collapsed = [s for s in mine if s.extra.get("collapsed_across_x")]
        provenance = {
            **base,
            # `live`, never `mine`: a vote may only credit readers that cast a ballot. The
            # confidence score pays +0.25 for "two independent model families read it and
            # agreed", and `mine` includes readers whose `mean` is None — in `runs/proof`,
            # Bock group A was credited with a sonnet reader that produced no value, and
            # that phantom family is the whole reason group A was released while group B,
            # on the same figure, was withheld.
            "model_families": model_families(live),
            "collapsed_across_x": bool(mine_collapsed),
            "n_points": (min(int(s.extra.get("n_points") or 0) for s in mine_collapsed)
                         if mine_collapsed else None),
            "dispersion_approximation": MEAN_OF_POINT_SD if mine_collapsed else "",
            "legend_says": legend_text,
            "legend_dispersion": legend_type.value if legend_type else None,
            "mapper_dispersion": mapper_type.value if mapper_type else None,
            "dispersion_type_from": dispersion_from,
            "per_route": [s.to_dict() for s in mine],
            "route_values": {s.extractor_id: {"mean": s.mean, "error": s.error} for s in live},
            "snap_confidences": {s.extractor_id: s.snap_conf for s in mine
                                 if s.snap_conf is not None},
            "n_routes": len(live), "n_routes_with_error": len(with_error),
            "n_routes_voting_on_error": len(voting),
            "dispersion_topology_note": topology_note,
            "dispersion_route_errors": {s.extractor_id: {"error": s.error,
                                                         "one_sided": s.one_sided}
                                        for s in with_error},
            "mad_sigma": mad_sigma, "dispersion_mad_sigma": error_mad,
            "agreement": agreement,
            "mean_agreement": agreement["mean_agrees"],
            "error_agreement": agreement["error_agrees"],
            "one_sided": sides[0] if len(sides) == 1 else (sides or None),
            "zero_confidence_snaps": zero_notes,
            "resampled_routes": [s.extractor_id for s in live if s.sample > 0],
            "tool_calls": _aggregate_tool_calls(mine),
            "cal_missing": no_calibration,
            "needs_review": status == "ambiguous" or dispersion_only or no_calibration,
            "needs_review_kind": ("mean" if status == "ambiguous"
                                  else ("dispersion" if dispersion_only
                                        else ("calibration" if no_calibration else None))),
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
            model=("" if len(model_families(live)) > 1 else (live[0].model or "")),
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
