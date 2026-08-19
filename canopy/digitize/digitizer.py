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

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ingest.pdf import (FigureRegion, MIN_PANEL_NUMERIC, PanelRegion, PaperRecord,
                          caption_panels)
from ..llm.client import LLMClient
from ..models import (Candidate, DatasetSpec, DigitizeSettings, DispersionType, Source,
                      SourceKind)
from ..verify.panels import _label_key, _labels_are_the_same, _MIN_LABEL_CHARS, _names_group
from .calibrate import AxisCalibration, fit_axis, pair_ticks, pixel_resolution, \
    px_to_value, value_to_px
from .overlay import draw_overlay
from .cv import (Axes, Bar, Marker, MIN_BAR_WIDTH_PX, detect_bars, detect_markers, find_axes,
                 find_cap_ends, find_tick_marks, load_color, load_gray, ocr_tick_labels,
                 snap_horizontal_edge, snap_window_for)
from .vector import VectorScene, calibrate_from_scene, snap_to_vector, vector_candidates, \
    whisker_ends
from .vlm import (CAL_SOURCE_INFERRED, CAL_SOURCE_UNKNOWN, READOUT_VARIANTS, CoordReadout,
                  FigureView, PROMPT_VERSION, ReadOut, TargetSpec, VISIBLE_NO, VISIBLE_PARTIAL,
                  VISIBLE_UNKNOWN, coords, overlay_verify, read_out, summarize_tool_calls)

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


#: the stages that may take a route sample out of the ensemble. Recorded on the sample and
#: carried into `pixel_provenance["dropped_by"]`, because "why is there no value here" was
#: answered by a hard-coded string for a year: a categorical drop printed as "dropped by overlay
#: verification", and an audit of a nulled cell went looking at the overlay layer, which had done
#: nothing to it. A stage that drops a reading says so under its own name.
DROPPED_BY_AXIS = "axis_identity"
DROPPED_BY_CATEGORICAL = "categorical_x"
DROPPED_BY_OVERLAY = "overlay_verify"
DROPPED_BY_LEGIBILITY = "legibility"


def _drop(sample: "RouteSample", stage: str, reason: str) -> None:
    """Take a route sample out of the ensemble, naming the stage that did it and why."""
    sample.dropped = True
    sample.drop_reason = reason
    sample.extra["dropped_by"] = stage


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


#: witnesses whose tick VALUES are trustworthy enough that, disputed, they prove the rows they sit
#: on are another axis. The PDF text layer is typeset numbers; OCR and the model's tick read are
#: measurements of glyphs and can be wrong about the values while sitting on the right rows.
_TRUSTED_VALUE_WITNESSES = frozenset({"vector"})

#: how many tick values two witnesses must share before they can be the same axis
_MIN_SHARED_TICKS = 2


def _agreed_tick_labels(readouts: Sequence[ReadOut]) -> set[float]:
    """The tick values at least two read-outs both listed for the value axis they read from.

    Every read-out reports the ladder it read the numbers off (`tick_labels`), and it costs
    nothing. Two readers listing the same labels is independent evidence of WHICH axis the
    numbers are in — the thing no pixel-exact ladder can know on a figure with more than one
    value axis. Cressman's Fig. 3 has two panels side by side and a second, percentage axis on
    the right of the second one; the PDF text layer's ladder (45/35/25/15/5, rmse 0.003 px) was
    the neighbouring panel's, and it won on precision.
    """
    sets = [set(float(v) for v in reading.tick_labels) for reading in readouts
            if len(reading.tick_labels) >= 2]
    agreed: set[float] = set()
    for i, one in enumerate(sets):
        for other in sets[i + 1:]:
            agreed |= one & other
    return agreed


def _shares_the_axis(cal: AxisCalibration, agreed: set[float]) -> bool:
    """Does this ladder carry at least `_MIN_SHARED_TICKS` of the tick values the readers agree on?

    A ladder with more ticks than the readers listed (minor ticks, or the readers listed a
    subset) still shares them; a ladder from another panel or the other axis of a dual-axis
    panel shares none, or one (a common zero). Nothing is decided when the readers agreed on
    fewer than three labels — that is not enough to say what the axis is.
    """
    if len(agreed) < 3:
        return True
    values = {float(v) for _, v in cal.ticks}
    return len(values & agreed) >= _MIN_SHARED_TICKS


def _readout_calibration(readouts: Sequence[ReadOut], rows: Sequence[float],
                         model_ticks: Sequence[tuple[float, float]] = (),
                         rows_disputed: bool = False
                         ) -> tuple[AxisCalibration | None, str, int]:
    """The ladder the read-outs reported, as a calibration, and how many of them agreed on it.

    The values come from the read-outs; the pixels come from the CV tick rows or, when those
    belong to another axis of the crop, from the model's own tick pixels — paired only where the
    read-out listed the very values the model put at those pixels. The CV rows of a two-panel
    crop are the first panel's; a read-out ladder that cannot be paired with them is not absent,
    it is on the other panel.
    """
    fits: list[AxisCalibration] = []
    model_by_value = {float(v): float(px) for px, v in model_ticks}
    for reading in readouts:
        listed = {float(v) for v in reading.tick_labels}
        pairs = None
        # a VALUED match first: the model put these very values at these pixels, so pairing the
        # read-out's labels with them is evidence, not coincidence
        shared = sorted(listed & set(model_by_value)) if model_by_value else []
        if len(shared) >= 3 and len(shared) >= len(listed) - 1:
            pairs = [(model_by_value[v], v) for v in shared]
        # a COUNT match with the CV rows only after that, and never with rows that belong to an
        # axis the readers have disputed — four labels landing on four rows of the neighbouring
        # panel is exactly the mis-pairing this witness exists to catch
        elif not rows_disputed:
            pairs = _ladder_from_values(reading.tick_labels, rows)
        if not pairs:
            continue
        cal, _scale = fit_best_scale(pairs, axis="y")
        if cal is not None:
            fits.append(cal)
    if not fits:
        return None, ("no read-out tick ladder could be paired with the detected tick marks"
                      + (" (the detected rows are an axis the readers dispute)"
                         if rows_disputed else "")), 0
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
    #: witnesses set aside because their tick values are not the axis the readers report
    disputed: list[str] = field(default_factory=list)

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
    # --- the axis-identity question is settled FIRST, before any read-out ladder is built: a
    # ladder that shares none of the tick values two readers agree the axis carries is a ladder
    # of ANOTHER axis — the neighbouring panel's, or the right-hand axis of a dual-axis panel.
    # Precision cannot rescue it: the PDF text layer's 45/35/25/15/5 fitted panel a to 0.003 px
    # while the readers read panel b's 0/10/20/30, and every pixel route then converted the
    # right pixels with the wrong ruler.
    agreed = _agreed_tick_labels(readouts)
    disputed = {name for name, cal in witnesses.items() if not _shares_the_axis(cal, agreed)}
    # The CV tick rows are the rows of whichever axis the pixel witnesses sit on. When a witness
    # whose VALUES can be trusted — the PDF's own typeset text — is disputed, those rows really
    # are another axis, and the read-out labels must not be count-paired with them. A disputed
    # OCR ladder testifies to nothing of the kind: OCR misreading the glyphs at these rows (F1's
    # 45→4) leaves them the readers' own axis, and pairing the readers' labels with them is
    # exactly how F1 was fixed.
    disputed_rows = [px for name in disputed if name in _TRUSTED_VALUE_WITNESSES
                     for px, _ in witnesses[name].ticks]
    rows_disputed = bool(disputed_rows) and bool(core.tick_rows) and (
        sum(1 for r in core.tick_rows if any(abs(r - px) <= 3.0 for px in disputed_rows))
        >= max(2, len(core.tick_rows) // 2))
    model_ticks = (cal_model.ticks if cal_model is not None and "vlm_ticks" not in disputed
                   else ())
    cal_read, read_note, read_support = _readout_calibration(
        readouts, core.tick_rows, model_ticks, rows_disputed=rows_disputed)
    if cal_read is not None:
        witnesses["readout_ticks"] = cal_read
    notes["readout_ticks"] = read_note

    record = {name: {"ticks": cal.ticks, "scale": cal.scale, "rmse_px": cal.rmse,
                     "note": notes.get(name, "")}
              for name, cal in witnesses.items()}
    if not witnesses:
        return CalibrationChoice(status="none", why="no y calibration could be built",
                                 witnesses=record)
    for name in disputed:
        record[name]["disputed_by_readouts"] = (
            f"its ticks {sorted(float(v) for _, v in witnesses[name].ticks)} share fewer than "
            f"{_MIN_SHARED_TICKS} values with the {sorted(agreed)} two readers report for this "
            f"axis — a ladder of another axis of the crop")
    all_witnesses = dict(witnesses)
    if disputed and len(disputed) < len(witnesses):
        witnesses = {name: cal for name, cal in witnesses.items() if name not in disputed}
        for name in disputed:
            notes[name] = f"{notes.get(name, name)} — disputed by the readers' tick labels"

    # --- the magnitude test, witness by witness (disputed ones included, so the record carries
    # both facts): a ladder the agreed read-out values cannot be drawn on is not a candidate, and
    # dropping it lets a surviving witness (Cressman's own read-out ladder, 45..-5) supply the
    # axis instead of the cell losing its calibration
    for name in disputed:
        if name in all_witnesses and name not in witnesses:
            why = _magnitude_refutes(all_witnesses[name], readouts)
            if why is not None:
                record[name]["refuted"] = why
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
    choice.disputed = sorted(disputed)
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
    if disputed and source in disputed:
        # every witness was disputed and the top one is being used anyway: say so, and nothing
        # downstream may treat this ruler as corroborated
        choice.status = "single_witness"
        choice.why = (f"{choice.why}; the readers' tick labels {sorted(agreed)} dispute EVERY "
                      f"ladder, including this one — the axis these numbers are in is unsettled")
    elif disputed:
        choice.why = (f"{choice.why}; the {', '.join(sorted(disputed))} ladder was set aside — "
                      f"it shares no tick values with the axis the readers report")
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


#: the four answers `_categorical_role` can give, and what each one means for the read
CATEGORICAL_GROUPS = "groups"           # the x categories ARE the comparison arms
CATEGORICAL_CONDITIONS = "conditions"   # the outcome is the average across the categories
#: the series are the groups and the SOURCE names one of the x categories: the point at that
#: category is this group's value. Neither of the two above — nothing is averaged (there is one
#: point per series to average), and the categories do not name the groups (they name the
#: conditions), which is why the two rules that came before it both produced nothing here.
CATEGORICAL_POINT_AT_CATEGORY = "point_at_category"
CATEGORICAL_UNRESOLVED = "unknown"      # nothing said which, so nothing may be averaged

#: how a locator can name a position on the x axis instead of quoting the category's name. Only
#: phrases that name a POSITION count: a bare "left" is prose about where a panel sits, and a
#: locator saying so is not a locator naming a category.
POSITIONAL_X_PHRASES = ("left side of the x axis", "left-most", "leftmost", "first", "last",
                        "right-most", "rightmost")
#: …matched on WORD boundaries. As a bare substring test, `last` fires inside "plasticity",
#: "lastly" and "elastic", and a locator that names no position at all reads as one.
_POSITIONAL_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in POSITIONAL_X_PHRASES)
                            + r")\b")
#: the category a locator quotes or brackets: `'without strategy'`, `"block 20"`, `(pre-test)`
_LOCATOR_PHRASE = re.compile(r"'([^']{2,60})'|\"([^\"]{2,60})\"|\(([^()]{2,60})\)")
_WORD_RE = re.compile(r"[a-z0-9]+")
#: a locator SAYING it spans the axis: "at each of the 8 target directions", "averaged across
#: blocks", "all eight targets". Whatever else such a locator brackets or quotes, it has already
#: said the value is not at one place on the x axis (whole-branch review, BLOCKER 1).
_ENUMERATION_CUE = re.compile(r"\b(?:each of|all|every|across|average(?:d)? (?:over|across))\b")
#: what separates the items of a bracketed LIST — `(0, 45, 90)`, `(block 1 and block 20)`,
#: `(pre/post)`, `(days 1–3)`
_LIST_SPLIT = re.compile(r",|/|;|\band\b|–|—")

#: The vocabulary question — *do these words name this group?* — is asked in two places now: here,
#: to decide whether a categorical x axis IS the comparison, and in `verify.panels`, to decide
#: whether a caption puts this group in the panel a reading came from. Two answers to one question
#: is how a figure gets read one way and checked another, so `_labels_are_the_same`, `_names_group`
#: and `_label_key` live in `verify.panels` and are imported at the top of this file — they are
#: still reachable under their old names here, for the callers and tests that knew them.


def _locator_phrases(locator: str) -> list[str]:
    """Every phrase a locator quotes or brackets — the places it names something verbatim."""
    out: list[str] = []
    for match in _LOCATOR_PHRASE.finditer(str(locator or "")):
        phrase = next(g for g in match.groups() if g is not None).strip()
        if phrase:
            out.append(phrase)
    return out


def _agreed_x_read(readings: Sequence[Any]) -> str:
    """The one x position every series was read at, when they all name the same one.

    A positional phrase in the locator ("the left-most point") says WHERE without saying WHAT, so
    on its own it is not a category. It becomes one when the readers, independently, all say they
    read at the same place — and "without strategy" and "without strategy (left end of x axis)"
    are the same place, which is why this is `_labels_are_the_same` and not string equality.
    """
    reads = [str(getattr(row, "x_read", "") or "").strip()
             for reading in readings for row in getattr(reading, "groups", ())
             if str(getattr(row, "x_read", "") or "").strip()]
    if not reads:
        return ""
    shortest = min(reads, key=len)
    return shortest if all(_labels_are_the_same(shortest, r) for r in reads) else ""


def _same_category(a: Any, b: Any) -> bool:
    """Do a locator's phrase and a plotted x category name the same category?

    `_labels_are_the_same` for words, because a paper spells its own labels its own way
    ("Elderly" / "Elderly adults"). NOT for digits: a numeric category is the number it prints
    and nothing else, so a substring rule makes "135" the category named by "(n = 135)", by
    "(1350)" and by the joined key of the whole list `0, 45, 90, 135, …` — which is exactly how
    an average across eight target directions became the value at one of them (BLOCKER 1).
    """
    left, right = _label_key(a), _label_key(b)
    if not left or not right:
        return False
    if left.isdigit() or right.isdigit():
        return left == right
    return _labels_are_the_same(a, b)


def _locator_category(locator: str, readings: Sequence[Any]) -> str:
    """The x category this locator names, or `""` — a phrase it quotes, or a position it points
    at that every reader agrees on.

    Three ways of naming NO category, and each of them is a locator that says something else:

    * it ENUMERATES ("at each of the 8 target directions", "averaged across blocks"). Whatever
      it brackets, it has already said the value is not at one place on the axis;
    * one of its bracketed phrases is a LIST of two or more items. A list names several
      categories, and several is not one;
    * more than one of its phrases matches a category. Taking the first in reading order picks
      whichever point the figure happens to plot leftmost, which is a decision about the data
      made by a sort order.

    A position is only a category once it lands on one. "The left-most point" beside readers who
    all say they read "every point on the x axis" names no category at all: taking their word for
    it there would turn a series mean into a point read, which is the opposite mistake to the one
    D3 exists to fix. So the position has to match a category some reader actually listed — and a
    reading that listed NO categories is not the exception to that rule but its plainest case: it
    has said nothing about where its mean sits, so nothing here can say the mean is a point.
    """
    if not str(locator or "").strip():
        return ""
    labels = [str(p.x_label).strip() for reading in readings
              for row in getattr(reading, "groups", ()) for p in getattr(row, "points", ())
              if str(getattr(p, "x_label", "") or "").strip()]
    if _ENUMERATION_CUE.search(str(locator).lower()):
        return ""
    matched: dict[str, str] = {}
    for phrase in _locator_phrases(locator):
        if len([p for p in _LIST_SPLIT.split(phrase) if p.strip()]) > 1:
            return ""
        for label in labels:
            if _same_category(phrase, label):
                matched[_label_key(label)] = label
    if matched:
        return next(iter(matched.values())) if len(matched) == 1 else ""
    if not _POSITIONAL_RE.search(str(locator).lower()):
        return ""
    agreed = _agreed_x_read(readings)
    if not agreed or not labels:
        return ""
    return next((label for label in labels if _same_category(agreed, label)), "")


def _vocab_words(names: Sequence[str]) -> set[str]:
    """The words a group is called by, long enough to tell one group from another."""
    out: set[str] = set()
    for name in names:
        out |= {w for w in _WORD_RE.findall(str(name or "").lower())
                if len(w) >= _MIN_LABEL_CHARS}
    return out


def _series_names_its_group(label_read: Any, own: Sequence[str], other: Sequence[str]) -> bool:
    """Does this series' description name its OWN arm, and not the other one?

    `_names_group` asks whether two LABELS are the same thing; a series description is not a
    label — "dashed purple (older non-instructed)" names a line style, a colour and a condition
    besides the group, so nothing in it spells "older adults". What separates the two arms is the
    vocabulary one of them has and the other does not, so that is what is looked for, on whole
    words: "young" must not answer for "younger" by being a piece of it.
    """
    if _names_group(label_read, own) and not _names_group(label_read, other):
        return True
    words = set(_WORD_RE.findall(str(label_read or "").lower()))
    mine, theirs = _vocab_words(own), _vocab_words(other)
    return bool(words & (mine - theirs)) and not (words & (theirs - mine))


def _point_at(row: Any, category: str) -> Any | None:
    """The one point of this series that IS the named category — or `None`.

    A series with a single point was taken as that category whatever the reader had called it,
    and on Vachon's Fig 4 that returned group B's only point ('with strategy', 25.0) as B's value
    at 'without strategy': two halves of one comparison read off two different conditions, in one
    candidate, with nothing on the record to say so (review MINOR 23). The label the source asked
    for and the label the reader wrote down are the whole of the evidence here, so one
    contradicting the other is not a match. An UNLABELLED point still stands: it contradicts
    nothing, and refusing it would throw away every reading of a single-category figure.
    """
    named = [point for point in row.points if _labels_are_the_same(point.x_label, category)]
    if len(named) == 1:
        return named[0]
    if len(row.points) == 1 and not str(row.points[0].x_label or "").strip():
        return row.points[0]
    return None


def _locatable_point(row: Any, category: str) -> bool:
    """Can this series' value be pinned to the named category, without picking one of several?

    A reading that listed no points cannot: its `mean` is what the read-out prompt asks to be the
    series as a whole, and calling that the point at a category another READER listed is how the
    two halves of the predicate come apart. The category and the point at it are asked of one
    reading or of none.
    """
    return bool(row.points) and _point_at(row, category) is not None


def _series_are_the_groups(readings: Sequence[Any], vocab: dict[str, tuple[str, ...]],
                           category: str) -> str:
    """How ONE reader told the two arms apart, when both of its series name their own group only
    and each of them has a value at `category`.

    Both halves are asked of the same reading on purpose: a reader that distinguished the arms
    and a different reader that read at the named category are not, together, a reader that did
    both, and the value D3 takes comes from a single read-out.
    """
    for reading in readings:
        rows = {g: reading.group(g) for g in GROUPS}
        if any(rows[g] is None for g in GROUPS):
            continue
        if not all(_locatable_point(rows[g], category) for g in GROUPS):
            continue
        reads = {g: str(getattr(rows[g], "label_read", "") or "").strip() for g in GROUPS}
        if not all(reads.values()) or _label_key(reads["A"]) == _label_key(reads["B"]):
            continue
        if all(_series_names_its_group(reads[g], vocab[g], vocab[other])
               for g, other in (("A", "B"), ("B", "A"))):
            return f"{reads['A']!r} and {reads['B']!r}"
    return ""


def _categorical_role(target: TargetSpec | None, readings: Sequence[Any], *,
                      locator: str = "") -> tuple[str, str]:
    """`(role, why)` — what ARE this figure's x categories, and what does that make the read?

    `x_axis_kind: "categorical"` conflates three different figure shapes. On the first, each
    series runs across the axis and the review wants the average across it. On the second, the
    axis IS the comparison — one bar per group — and averaging across it computes `(A + B) / 2`
    for both arms, annihilating the contrast and reporting Cohen's d = 0.0 with every route in
    perfect agreement. On the third the series are the groups and the categories are conditions,
    and the source names ONE of those conditions: the point there is the value, and there is
    nothing to average because each series has one point in the frame. No number distinguishes
    the three; the CATEGORY NAMES and the LOCATOR do.

    So the question asked here is the only one that can settle it: *do the categories the readers
    named map onto the groups the protocol is comparing?* If they do, each category is a group's
    own value and must be read as that group's; if they demonstrably do not (and there is more
    than one of them), they are conditions — and then it matters whether the locator picked one
    of them out. Anything else is unresolved, and an unresolved cell produces no number rather
    than a wrong one.

    Evidence, most authoritative first:

    1. `target.categorical_x`, when a caller has stated it outright;
    2. a reader listed categories matching BOTH group labels — the axis carries both arms;
    3. each group's own points are one category matching that group's own label;
    4. each group's `x_read` names its own group's label and not the other's;
    5. `locator` names one x category, the two series name the two groups, and no reader came
       back with two or more categories that name the groups — the point at that category is
       this group's value (D3);
    6. some series has two or more categories, none of which names either group.
    """
    if target is not None and target.categorical_x in (CATEGORICAL_GROUPS,
                                                       CATEGORICAL_CONDITIONS):
        return target.categorical_x, f"the caller stated the x categories are {target.categorical_x}"
    # a group's NAMES, not its one label: a paper labels its bars in its own words, and
    # "Old adults" is not a substring of "Older adults". The protocol already wrote down the
    # vocabulary of the review; without it this test fails to resolve and the cell yields nothing.
    labels = {"A": getattr(target, "group_a_label", "") if target else "",
              "B": getattr(target, "group_b_label", "") if target else ""}
    vocab = {g: tuple(n for n in (labels[g], *getattr(target, f"group_{g.lower()}_synonyms", ()))
                      if str(n or "").strip())
             for g in GROUPS}
    if not any(vocab.values()):
        return CATEGORICAL_UNRESOLVED, "the protocol gave no group labels to match categories to"

    # matching categories onto groups is only possible when both groups are named; a single
    # label matching a single category says nothing about what the axis IS
    both_labelled = all(vocab.values())
    conditions_seen: list[str] = []
    #: how many categories one series named that name a group — rule 5 is only reachable while
    #: this stays below two, which is rule 2's evidence and rule 2 wins it
    group_labelled_points = 0
    for reading in readings:
        rows = {g: reading.group(g) for g in GROUPS}
        for group, row in rows.items():
            if row is None:
                continue
            categories = [p.x_label for p in row.points if str(p.x_label or "").strip()]
            hits = {g: [c for c in categories if _names_group(c, vocab[g])]
                    for g in GROUPS if vocab[g]}
            group_labelled_points = max(
                group_labelled_points,
                len({_label_key(c) for cs in hits.values() for c in cs}))
            if both_labelled and all(hits.get(g) for g in GROUPS):
                named = ", ".join(sorted({c for cs in hits.values() for c in cs}))
                return CATEGORICAL_GROUPS, (
                    f"a reader listed the x categories as {named!r}, which are the two groups "
                    f"being compared — each category is a group's own value, not a point to "
                    f"average over")
            if both_labelled and len(categories) == 1 and hits.get(group):
                other = [g for g in GROUPS if g != group][0]
                if not _names_group(categories[0], vocab.get(other, ())):
                    return CATEGORICAL_GROUPS, (
                        f"group {group}'s only x category is {categories[0]!r}, its own group "
                        f"label — the axis puts one point per group")
            if len(categories) >= 2 and not any(hits.get(g) for g in GROUPS if vocab[g]):
                conditions_seen.append(
                    f"group {group} spans {len(categories)} categories "
                    f"({', '.join(map(str, categories[:4]))}), none of them a group label")
        reads = {g: str(getattr(rows[g], "x_read", "") or "") for g in GROUPS if rows[g]}
        if both_labelled and len(reads) == 2 and all(reads.values()):
            own = all(_names_group(reads[g], vocab[g]) for g in GROUPS if vocab[g])
            cross = any(_names_group(reads[g], vocab[o])
                        for g, o in (("A", "B"), ("B", "A")) if vocab[o])
            if own and not cross:
                return CATEGORICAL_GROUPS, (
                    f"each group was read at its own category on the x axis "
                    f"({reads['A']!r}, {reads['B']!r})")
    # D3, ranked above the conditions fallback: the categories are conditions, but the SOURCE
    # named one of them, and the series in front of the reader are the two groups. There is no
    # axis left to average over — each series has one point in the frame, at the category that
    # was asked for — so collapsing it produces nothing and dropping the pixel routes for
    # "reading one point" throws away the only routes that read the right point. It fires only
    # when the protocol permitted the collapse in the first place; without that permission the
    # figure is not being read across an axis at all and the ordinary path applies.
    if getattr(target, "collapse_across_x", False) and both_labelled and group_labelled_points < 2:
        category = _locator_category(locator, readings)
        series = _series_are_the_groups(readings, vocab, category) if category else ""
        if category and series:
            return CATEGORICAL_POINT_AT_CATEGORY, (
                f"the source names one x category ({category!r}) and the two series are the "
                f"groups themselves ({series}) — the point at that category IS this group's "
                f"value, and there is no second point of it on the axis to average with")
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


def _row_at_own_category(row: Any, names: Sequence[str]) -> Any:
    """The group's row with `mean`/`error_half_length` taken from its OWN x category.

    The point at this group's own category outranks whatever is in `mean`, and that ordering is
    the whole fix. `llm/prompts/digitize_readout.md` tells a reader on a categorical axis to
    "leave `mean` as your reading of the series as a whole" — which on a chart whose categories
    ARE the two groups is a number spanning both bars. Honouring it gave both arms the same mean
    and a Cohen's d of exactly 0.0, pooled, with every route agreeing because they were the same
    number. `mean` is used only when no category can be matched to this group.
    """
    from dataclasses import replace as _replace

    if not row.points:
        return row
    mine = [p for p in row.points if _names_group(p.x_label, names)]
    chosen = mine[0] if len(mine) == 1 else (row.points[0] if len(row.points) == 1 else None)
    if chosen is None or chosen.mean is None:
        return row
    return _replace(row, mean=chosen.mean,
                    error_half_length=(row.error_half_length if row.error_half_length is not None
                                       else chosen.error_half_length))


def _row_at_locator_category(row: Any, category: str) -> Any:
    """The group's row with `mean`/`error_half_length` taken from the category the SOURCE names.

    The sibling of `_row_at_own_category`, keyed on the locator's category rather than on the
    group's label: here the categories are conditions and the series are the groups, so what
    picks the point out is which condition was asked for (`_point_at`: a single point counts
    when the reader's own label for it agrees, or when the reader gave it none), and `mean` —
    which the read-out prompt asks to be the series as a whole — is used only when no point can
    be matched.

    A point PICKED OUT of a series brings only its own dispersion. `error_half_length`,
    `error_upper` and `error_lower` are the reader's statements about the SERIES: reporting them
    beside one category's mean makes the whole series' spread the band at that category, and
    `abs(error_upper - mean)` downstream — computed after the mean was swapped — is the distance
    from one category's cap to another category's mean, a number that is nowhere in the reading.
    The error is not a footnote: it sets this study's SE and therefore its weight in the pooled
    estimate. So the honest state is a mean with no spread, which the resolver already holds.
    """
    from dataclasses import replace as _replace

    if not row.points:
        return row
    chosen = _point_at(row, category)
    if chosen is None or chosen.mean is None:
        return row
    picked_out = len(row.points) > 1
    return _replace(row, mean=chosen.mean,
                    error_half_length=(chosen.error_half_length
                                       if chosen.error_half_length is not None
                                       else (None if picked_out else row.error_half_length)),
                    error_upper=(None if picked_out else row.error_upper),
                    error_lower=(None if picked_out else row.error_lower))


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
_CLAUSE_SPLIT = re.compile(r"[,;.\n]")


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
                          target: TargetSpec | None = None,
                          locator: str = "") -> list[RouteSample]:
    role, role_why = (_categorical_role(target, [reading], locator=locator) if collapse
                      else (CATEGORICAL_CONDITIONS, ""))
    category = (_locator_category(locator, [reading])
                if role == CATEGORICAL_POINT_AT_CATEGORY else "")
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
            synonyms = (getattr(target, f"group_{group.lower()}_synonyms", ()) if target else ())
            names = tuple(n for n in (label, *synonyms) if str(n or "").strip())
            resolved = _row_at_own_category(row, names)
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
        elif collapse and role == CATEGORICAL_POINT_AT_CATEGORY:
            # the categories are conditions, the source named one of them, and this series has
            # its point there. Nothing is averaged and nothing is approximated: the spread that
            # travels with the value is the paper's own band at that category.
            row = _row_at_locator_category(row, category)
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
                   "categorical_x_category": category,
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
def _own_x_px(s: RouteSample, scale: float) -> float | None:
    """The x THIS sample read at, in crop pixels: its `x_px`, or the pixel it named in prose.

    A read-out has no `x_px` field; it says where it read in `x_read` ("episode 20 (last filled
    square, x≈1242 px)"), in the pixels of the image it was shown, which is the crop scaled by
    `scale`. That is the reader's own claim about x, and the only x a `wrong_x` verdict on its
    mark can fairly be about.
    """
    if s.x_px is not None:
        return float(s.x_px)
    match = _X_PX_RE.search(str(s.extra.get("x_read", "")))
    if match and scale > 0:
        return float(match.group(1)) / scale
    return None


def _overlay_marks(samples: Sequence[RouteSample], cal: AxisCalibration | None,
                   core: _Core, labels: dict[str, str], scale: float = 1.0
                   ) -> tuple[list[dict], list[list[int]]]:
    """One numbered mark per distinct resolved pixel, with the sample indices behind each.

    Every mark records whether each owner's x was its OWN (`x_px`, or the pixel it named) or
    BORROWED from another sample of the group. Bock's Fig. 1: three read-outs said "episode 20,
    x≈1242 px" and 32.0; their mark was drawn at the coordinate route's x≈970 (episode 14), the
    verifier said "wrong x" — correctly, of the mark — and all three readers were dropped for a
    position that was never theirs.
    """
    marks: list[dict[str, Any]] = []
    owners: list[list[int]] = []
    own: dict[int, float] = {}
    for i, s in enumerate(samples):
        x = _own_x_px(s, scale)
        if x is not None:
            own[i] = x
    x_by_group: dict[str, float] = {}
    for i, s in enumerate(samples):
        if i in own:
            x_by_group.setdefault(s.group, own[i])
    mid_x = (core.axes.plot_bbox[0] + core.axes.plot_bbox[2]) / 2.0
    for i, s in enumerate(samples):
        if s.dropped or s.mean is None:
            continue
        y = s.y_px
        if y is None:
            if cal is None:
                continue
            y = value_to_px(cal, s.mean)
        borrowed = i not in own
        x = own.get(i, x_by_group.get(s.group, mid_x))
        for j, mark in enumerate(marks):
            if abs(mark["x"] - x) <= _MARK_MERGE_PX and abs(mark["y"] - y) <= _MARK_MERGE_PX:
                owners[j].append(i)
                mark["_routes"].append(_route_tag(s))
                mark["_borrowed"][i] = borrowed
                break
        else:
            marks.append({"x": x, "y": y, "kind": "point", "_routes": [_route_tag(s)],
                          "_group": s.group, "_borrowed": {i: borrowed}})
            owners.append([i])
    for mark in marks:
        who = ", ".join(dict.fromkeys(mark.pop("_routes")))
        group = mark.pop("_group")
        mark["borrowed_x"] = mark.pop("_borrowed")            # sample index → borrowed?
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
_QUOTED = re.compile(r"[\"'\u2018\u2019\u201c\u201d]([^\"'\u2018\u2019\u201c\u201d]{3,})"
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
            _drop(sample, DROPPED_BY_AXIS,
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
    """Do a reader's words and a detected marker agree on everything BOTH of them state?

    Vacuously true when either side states nothing — which is the right answer to "does this
    contradict?" and the wrong one to "does this corroborate?". Use `_descriptors_corroborate`
    for the second question.
    """
    return all(not a or not b or a == b for a, b in zip(described, detected))


def _descriptors_corroborate(described: tuple[str, str], detected: tuple[str, str]) -> bool:
    """Do they agree on at least one thing they BOTH state? Silence corroborates nothing."""
    return (_descriptors_match(described, detected)
            and any(a and b for a, b in zip(described, detected)))


#: how far a detected marker may be from the point a route measured and still be the marker that
#: route was looking at — three of its own widths, or 24 px for a marker too small to scale by
_MARKER_NEAR_PX = 24.0


def _nearest_marker(core: _Core, x: float | None, y: float | None
                    ) -> tuple[Marker | None, float]:
    """`(marker, distance)` — the detected marker closest to a measured point, and how far it is.

    The distance is returned because "the nearest marker in the whole panel" is not the same
    claim as "the marker at this datum", and a check that cannot resolve its input must not turn
    that into a positive finding about the input.
    """
    if y is None or not core.markers:
        return None, float("inf")
    if x is None:
        marker = min(core.markers, key=lambda m: abs(m.y - y))
        return marker, abs(marker.y - y)
    marker = min(core.markers, key=lambda m: (m.x - x) ** 2 + (m.y - y) ** 2)
    return marker, ((marker.x - x) ** 2 + (marker.y - y) ** 2) ** 0.5


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

    # what the pixel pass actually found where each group's value was measured, and how sure it
    # is: a marker it could not classify, or one it found halfway across the panel, is not
    # evidence about this datum either way
    detected: dict[str, tuple[str, str]] = {}
    resolved: dict[str, bool] = {}
    distances: dict[str, float] = {}
    for group in GROUPS:
        pixel = next((s for s in samples if s.group == group and s.y_px is not None), None)
        marker, distance = _nearest_marker(core, pixel.x_px if pixel else None,
                                           pixel.y_px if pixel else None)
        if marker is None:
            resolved[group] = False
            continue
        descriptor = _detected_descriptor(marker)
        near = distance <= max(3.0 * max(float(marker.size), 1.0), _MARKER_NEAR_PX)
        detected[group] = descriptor
        distances[group] = round(float(distance), 2)
        resolved[group] = bool(near and any(descriptor))
    if detected:
        info["detected"] = {g: list(v) for g, v in detected.items()}
    info["detected_resolved"] = dict(resolved)
    info["detected_distance_px"] = distances
    kinds = {str(m.kind) for m in core.markers}
    if kinds:
        info["detected_marker_kinds"] = sorted(kinds)

    # `transposed` is the one positive finding here that flips the sign of an effect size, so it
    # is the one that must never be asserted on silence. `_descriptors_match` is vacuously true
    # when the detector resolved nothing, and with it the swap used to be "corroborated" at a
    # point where nothing had been measured at all. Both crossings now have to be a real
    # agreement between two descriptors that each state something, at a marker actually found on
    # that datum, and both self-comparisons have to fail. General rule: a check that cannot
    # resolve its input reports that it could not, never a positive finding about the input.
    if (len(described) == 2 and len(detected) == 2 and described["A"] != described["B"]
            and any(described["A"]) and any(described["B"])
            and all(resolved.get(g) for g in GROUPS)
            and _descriptors_corroborate(described["A"], detected["B"])
            and _descriptors_corroborate(described["B"], detected["A"])
            and not _descriptors_match(described["A"], detected["A"])
            and not _descriptors_match(described["B"], detected["B"])):
        info["transposed"] = True
        info["notes"].append(
            "each group's described marker is the one found where the OTHER group's value was "
            "measured — the two series are transposed, which flips the sign of the effect")
    elif len(described) == 2 and not all(resolved.get(g) for g in GROUPS):
        unresolved = [g for g in GROUPS if not resolved.get(g)]
        info["notes"].append(
            f"the pixel pass could not resolve a marker on group {', '.join(unresolved)}'s "
            f"datum, so nothing here can say whether the two series are the right way round")

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


def _dispersion_topology_conflict(with_error: Sequence[RouteSample], kind: DispersionType) -> str:
    """The note for routes that disagree about the WHISKER'S SHAPE, or "" when they do not.

    Two routes can agree that a bar exists and still be describing different objects: one reports
    a whisker drawn on one side only and hands back the arm it saw, the other reports two arms and
    hands back their mean. That is a disagreement about the FIGURE, not only about a number, and
    the median of the two is a value neither route reported.

    **It is recorded and reviewed, not arbitrated.** I tried arbitrating it — taking the one-armed
    reads, on the argument that a one-armed read is the half-length whether the bar has one arm
    (it measured the only one) or two (the arms of `mean ± half-length` are equal), whereas a
    two-armed read is right only in the second case. The argument is sound about what the readings
    MEAN and says nothing about how accurately each cap was located, which is what the error is
    made of. Measured over the six scorable cells of the two completed runs it lost: mean |d| error
    0.0570 -> 0.0637, helping one cell by 0.014 and hurting two by 0.056 between them. On the worst
    of them the two-armed reader had the cap right to 0.02 deg of the pixels and the one-armed
    reader was 2.1 deg out, while its TOPOLOGY claim was the more faithful of the two. Topology is
    not a proxy for accuracy.

    So the ensemble still medians every route that found a whisker, and the conflict is surfaced:
    it sets the cell for review, and `_needs_another_readout` — which now sees the dispersion —
    buys a further reading while budget remains. Asking for more evidence is the honest response to
    two readers describing different pictures; picking one of them by rule is not.

    Not raised for an IQR or a min-max whisker, whose arms genuinely differ, so "one-armed" is not
    a claim about the same quantity there.
    """
    if kind not in _SYMMETRIC_DISPERSIONS:
        return ""
    one_armed = [s for s in with_error if s.one_sided]
    two_armed = [s for s in with_error if not s.one_sided]
    if not one_armed or not two_armed:
        return ""
    sides = "/".join(sorted({s.one_sided for s in one_armed if s.one_sided}))
    return (f"{len(one_armed)} route(s) read this whisker as drawn on one side only ({sides}) and "
            f"{len(two_armed)} read two arms — they disagree about the shape of the bar, not only "
            f"about its length, so the half-length they were medianed into is a value neither of "
            f"them reported")


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
    if any(s.dropped and not s.extra.get(ILLEGIBLE) for s in samples):
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
    # C1 rule 3-4, decided BEFORE any model call: read the panel the map names, not the union of
    # every panel; and a panel whose own axis ladder is not inside its rect buys NO read-outs.
    # Paying three models to read numbers off an image that does not contain the numbers is the
    # exact spend that produced Heuer's `value_outside_axis` flags.
    fig, panel_info = resolve_panel(fig, target, source)
    crop = _asset(paper, fig.crop_png)
    if panel_info.get("uncalibrated"):
        return _panel_uncalibrated(fig, target, paper, source, dataset, crop, panel_info, result)
    text = caption if caption is not None else (fig.caption or "")
    # WHERE in the figure the map sent us. On a categorical x axis this is evidence about the
    # quantity itself — a locator that names one of the categories is asking for the point there,
    # not for the average across them (D3) — so it travels with the read-outs, not just with the
    # candidates that come out at the end.
    locator = source.locator if source is not None else ""
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
        fresh = _samples_from_readout(reading, collapse=target.collapse_across_x, target=target,
                                      locator=locator)
        _mark_illegible(reading, fresh)          # C2: legibility is reported, then acted on
        samples.extend(fresh)

    # --- path D: the first `readouts_min` read-outs
    for spec in plan[:n_min]:
        read(spec)
    # C2: when a MAJORITY of the readers say the named target is not in this image, that is a
    # fact about the image and the route abstains — before the pixel routes are paid to measure
    # the same picture, and without ever adopting the minority reading that produced numbers.
    seen = legibility(readouts)
    if seen["abstain"]:
        return _target_not_visible(fig, target, paper, source, dataset, crop, seen, readouts,
                                   panel_info, result)

    # --- path C: VLM coordinates + CV snap
    primary = models[0] if models else "claude-opus-5"
    coord = coords(client, crop, text, target, primary, view=view, cell_key=f"{key}/C")
    # route A's scene is built HERE, before anything converts a pixel, so the PDF's own tick
    # ladder is one of the witnesses the calibration vote sees
    scene, vec_info = _vector_scene(paper, fig)
    cal_vec = calibrate_from_scene(scene, axis="y") if scene is not None else None
    # C2, filtered ONCE at the point of disqualification: a reading that could not see the target
    # or built its own ladder is not a witness to anything — not to the value, not to the axis
    # identity, not to the ladder, not to the unit. See `voting()` for the 31.5 -> 23.64 that
    # dropping only its samples produced.
    choice = _choose_calibration(core, coord, cal_vec, voting(readouts))
    cal, cal_source, cal_why = choice.cal, choice.source, choice.why
    pixel_cal = choice.usable_for_pixels
    pixel_samples = _samples_from_coords(coord, core, pixel_cal, cal_source)
    # --- path B: raster CV, matched by the nearest VLM coordinate
    pixel_samples += _samples_from_raster(coord, core, pixel_cal, cal_source)
    cat_role, cat_role_why = ((_categorical_role(target, voting(readouts), locator=locator))
                              if target.collapse_across_x else (CATEGORICAL_CONDITIONS, ""))
    if (target.collapse_across_x
            and cat_role not in (CATEGORICAL_GROUPS, CATEGORICAL_POINT_AT_CATEGORY)):
        # a pixel route resolves ONE datum; when the outcome is the mean of every datum on the
        # axis its answer is a different number and must not enter the ensemble. When the x
        # categories ARE the groups — or when the source named the one category to read at —
        # there is nothing to average: one datum per group is exactly the quantity, and dropping
        # these routes threw away correct readings.
        for pixel in pixel_samples:
            _drop(pixel, DROPPED_BY_CATEGORICAL, (
                "this route reads one point, and the outcome is the average across the "
                f"categorical x axis ({cat_role_why})" if cat_role == CATEGORICAL_CONDITIONS else
                "this route reads one point, and nothing has established whether the x categories "
                f"are conditions to average across or the groups themselves ({cat_role_why})"))
    samples.extend(pixel_samples)

    _, tick_spacing = _tick_stats(pixel_cal)
    axis_range, axis_range_source = _axis_range(pixel_cal, core)
    if pixel_cal is None and choice.status == "cal_refuted":
        # the ladder is wrong, so nothing derived from it may set a tolerance; the read-outs'
        # own magnitude is what is left, and it is recorded as such
        magnitudes = [abs(m) for means in _readout_means(voting(readouts)).values()
                      for m in means]
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

    # --- C10: a route that came back with a spread and no mean is a PARTIAL read, not a missing
    # witness. "One family read this" and "two families read it and one came back empty" must
    # never print the same reason, and the cheapest answer is not to re-price the confidence —
    # it is to re-ask that one reader for that one group. Exactly one such call is bought.
    partial_before = _partial_reads(samples)
    reread: dict[str, Any] = {}
    if partial_before:
        reread = _rebuy_partial(partial_before, readouts, samples, read)
    # ...and a hole the re-read FILLED is no longer a hole. Leaving the entry standing made the
    # cell print "a second model family returned a spread but no mean for group A" about a family
    # that had just supplied the mean and was, on the same record, counted among `model_families`
    # — two provenance facts contradicting each other. The pre-re-read list is kept under its own
    # name, because what was bought and why is also a fact.
    partial = _still_partial(partial_before, samples)

    # C2 again, now that every read-out in the plan has been spent: a majority that could not see
    # the target is the same fact whether it arrives on the second reader or the fourth.
    seen = legibility(readouts)
    if seen["abstain"]:
        return _target_not_visible(fig, target, paper, source, dataset, crop, seen, readouts,
                                   panel_info, result, coord=coord)

    # --- path A: vector-exact (after every read-out, so it sees every tick ladder)
    samples.extend(_samples_from_vector(scene, vec_info, coord, core, voting(readouts)))
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
            marks, owners = _overlay_marks(samples, cal, core, labels, scale=view.scale)
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
            if not choice.confirmed:
                # The mark is drawn with the calibration. Under a ruler only one witness built —
                # or one the readers dispute — "the circle is not on the datum" is a disagreement
                # between the mark and the reading, and the ruler is the suspect: it says
                # nothing about which of the two is wrong. Three read-outs across two families
                # were dropped this way on Cressman's Fig. 3b for sitting "3.4 deg below the bar
                # top" — the bar top measured with the neighbouring panel's ladder. A conviction
                # needs a corroborated ruler; without one the verdict is recorded, not applied.
                verify_log[-1]["not_applied"] = (
                    f"{len(bad)} mark(s) judged off the datum, but the calibration is "
                    f"{choice.status}, so the mark's own placement is uncorroborated and no "
                    f"read-out is dropped on it")
                for v in bad:
                    for idx in owners[v.number - 1]:
                        samples[idx].extra["overlay_disputed"] = f"{v.verdict} — {v.reason}"
                break
            for v in bad:
                borrowed = marks[v.number - 1].get("borrowed_x") or {}
                for idx in owners[v.number - 1]:
                    if samples[idx].dropped:
                        continue
                    if borrowed.get(idx, False):
                        # The mark's x was borrowed from another sample of the group. Whatever the
                        # verdict is called, a mark that is not on the datum at a position the
                        # reader never claimed says nothing about which coordinate is wrong — the
                        # y the reader read, or the x it never gave. Cressman's Fig. 3a: the
                        # coordinate route landed 90 px right of block 33, on the next panel's
                        # axis label; the verifier said `not_on_datum` (its reasons all about x),
                        # and three readers who had said "Block 33" and 31.1/31.0/31.4 were
                        # dropped for it. The verdict is recorded; the value is judged by the vote.
                        samples[idx].extra["overlay_disputed"] = (
                            f"{v.verdict} at a borrowed x — {v.reason}")
                        continue
                    _drop(samples[idx], DROPPED_BY_OVERLAY,
                          f"overlay verify: {v.verdict} — {v.reason}")
                    dropped += 1
            verify_log[-1]["dropped_samples"] = dropped
            if not dropped or not any(s.usable for s in samples):
                break
    else:
        # the model is not asked, but the picture is still drawn: it costs nothing, and it is what
        # a reviewer opens to see where the routes landed
        marks, _ = _overlay_marks(samples, cal, core, labels, scale=view.scale)
        if marks:
            out_png = work / f"{fig.id}.overlay1.png"
            draw_overlay(crop, marks, out_png)
            overlay_path = str(out_png)

    provenance = _base_provenance(fig, core, choice, coord, vec_info, verify_log, target)
    provenance.update(axis_info)
    provenance["panel"] = panel_info
    provenance["legibility"] = seen
    provenance[PARTIAL_READ] = partial
    provenance["partial_read_before_reread"] = partial_before
    provenance["partial_read_reread"] = reread
    provenance["series_identity"] = series_info
    provenance["axis_range"] = axis_range
    provenance["axis_range_source"] = axis_range_source
    provenance["call_plan"] = {
        "readouts_min": n_min, "readouts_max": n_max, "readouts_run": len(readouts),
        "extra_readouts_bought": bought, "extra_readout_reason": buy_reason,
        "readout_stop_reason": stop_reason,
        "overlay_verify": bool(do_verify), "overlay_verify_reason": verify_reason,
        "partial_read_rereads": int(bool(reread.get("bought"))),
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
                                   px_units=px_units, readouts=voting(readouts),
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
    if a.mean != b.mean or a.dispersion_value != b.dispersion_value:
        return
    # No `collapsed_across_x` gate: it does not matter HOW the two arms came to be the same
    # number. Gating on it left the groups branch — where `collapse_here` is False — with no net
    # at all, which is exactly where a reader's series-wide `mean` lands on both arms.
    how = ("the same average across the categorical x axis"
           if a.pixel_provenance.get("collapsed_across_x") else "the very same number")
    reason = (f"both groups came back as {how} "
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


_X_PX_RE = re.compile(r"x\s*(?:=|≈|~|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*px")


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
PANEL_UNCALIBRATED = "panel_uncalibrated"
PANEL_NOT_IN_CROP = "panel_not_in_crop"
#: the map named a panel and ingestion could not hand that panel over on its own. The reading is
#: taken from whatever rect there is and RECORDED as such — never denied, never an abstention.
PANEL_NOT_ISOLATED = "panel_not_isolated"
ONE_READER_BLIND = "one_reader_could_not_see_it"
PARTIAL_READ = "partial_read"
#: marks a sample dropped because its READER was disqualified, not because its number was
#: disputed. The overlay call exists to settle a disagreement between routes about where a datum
#: sits; "this reader could not see the panel" is not that disagreement, and paying for a picture
#: of it buys nothing.
ILLEGIBLE = "illegible"


def _reader_id(reading: ReadOut) -> str:
    name = ":".join(x for x in ("digitize:readout", reading.model, reading.variant) if x)
    return f"{name}#{reading.sample + 1}" if reading.sample else name


def legibility(readouts: Sequence[ReadOut]) -> dict[str, Any]:
    """What the readers reported about whether the target is IN the picture — a fact, not a note.

    "It is not in this picture" is the one thing a reader can tell us that no amount of averaging
    can recover, and until now it had nowhere to go but prose. A MAJORITY saying so is a fact
    about the image: the route abstains and the cell is queued for a human (nothing re-acquires
    the crop yet), and the minority reading that did produce numbers is never adopted — it read
    something, but not the thing that was asked for. A minority saying so is a fact about that
    reader: its samples do not vote, and the cell carries the disagreement.

    A `partial` reader is in neither camp. It is not saying the target is absent, so it is not a
    `no` vote; and it is not a witness to the value either, because half a panel is not the panel.
    It is therefore left out of the DENOMINATOR: counted there, every extra half-view of a
    half-cropped figure would make abstention less likely, which is backwards.
    """
    blind = [r for r in readouts if r.target_visible == VISIBLE_NO]
    partly = [r for r in readouts if r.target_visible == VISIBLE_PARTIAL]
    inferred = [r for r in readouts if r.calibration_source == CAL_SOURCE_INFERRED]
    silent = [r for r in readouts if r.target_visible == VISIBLE_UNKNOWN
              or r.calibration_source == CAL_SOURCE_UNKNOWN]
    total = len(readouts) - len(partly)
    majority = bool(blind) and total >= 2 and len(blind) * 2 > total
    return {
        "n_readouts": len(readouts),
        "n_voting": total,
        "target_not_visible": [_reader_id(r) for r in blind],
        "target_not_visible_why": [r.target_visible_reason or r.notes for r in blind],
        "target_partly_visible": [_reader_id(r) for r in partly],
        "target_partly_visible_why": [r.target_visible_reason or r.notes for r in partly],
        "calibration_inferred": [_reader_id(r) for r in inferred],
        "said_nothing": [_reader_id(r) for r in silent],
        "abstain": majority,
        ONE_READER_BLIND: bool(blind) and not majority,
    }


def voting(readouts: Sequence[ReadOut]) -> list[ReadOut]:
    """The read-outs that may witness ANYTHING — the value, the ladder, the axis, the unit.

    C2 says a disqualified reading is dropped before the ensemble, and dropping its SAMPLES is
    not the same thing as dropping its VOTE. Measured on the synthetic bar figure (truth 31.5,
    ticks 0..60) with one honest reader and two disqualified ones — a `target_visible: no` reader
    carrying nothing but a tick ladder, and a `calibration_source: inferred` one — both listing
    the neighbouring panel's `0,5,...,30`: the ensemble printed **23.64** instead of 31.5, on a
    ruler chosen by the two readings the rule had already declared non-voting, with the
    provenance note "2 of 3 read-out tick ladder(s) agree". A reader that read nothing was
    credited as a witness and moved the printed number by a quarter of its value.

    So the filter is applied ONCE, here, at the point of disqualification, and the filtered list
    is what every witness function sees. `legibility()` and the cost/provenance sums keep the
    full list: what a disqualified reader SAID is still a fact about the figure, and its money
    was still spent.

    `partial` is left in: a reader that could see part of the panel could still read the printed
    ladder off it, and F10 disqualifies it for the VALUE only.
    """
    return [r for r in readouts
            if r.target_visible != VISIBLE_NO and r.calibration_source != CAL_SOURCE_INFERRED]


def _partial_reads(samples: Sequence[RouteSample]) -> list[dict[str, Any]]:
    """Routes that produced part of a reading and not the rest, named by route and by group.

    Corroboration is credited only from readers that produced THE QUANTITY being corroborated, so
    a reader that returned `mean=None, error=6.5` corroborates no mean — that rule stands, and it
    is why Bock's group A scored 0.22 against group B's 0.45 off the very same three readers. The
    asymmetry it leaves is the defect: the cell reported "only one independent route produced this
    value", which is what a genuinely single-family cell reports, when in truth a second family
    read the figure and came back with half an answer. Naming the half-answer is what lets the
    cheapest fix apply: re-ask that one reader for that one group.
    """
    out = []
    for sample in samples:
        if sample.route != "D" or sample.dropped:
            continue
        if sample.mean is None and sample.error is not None:
            out.append({"route": sample.extractor_id, "group": sample.group, "missing": "mean",
                        "has": {"error": sample.error}, "model": sample.model,
                        "variant": sample.variant, "sample": sample.sample})
        elif sample.mean is not None and sample.error is None:
            out.append({"route": sample.extractor_id, "group": sample.group, "missing": "error",
                        "has": {"mean": sample.mean}, "model": sample.model,
                        "variant": sample.variant, "sample": sample.sample})
    return out


def _still_partial(partial: Sequence[dict[str, Any]],
                   samples: Sequence[RouteSample]) -> list[dict[str, Any]]:
    """The partial reads that are STILL partial once the targeted re-read has come back.

    C10 names the half-answer and buys the missing number back; it never said what happens to the
    name once the number arrives. What happened was that the cell reported a hole and counted the
    same family as whole at the same time. The producer owns this contract, so the retraction is
    made here rather than left for every consumer to guess at: a reader that later supplied the
    missing quantity for that group is no longer a partial read of it.
    """
    out = []
    for item in partial:
        want = "mean" if item["missing"] == "mean" else "error"
        filled = any(s.route == "D" and s.group == item["group"] and not s.dropped
                     and s.model == item["model"] and s.variant == item["variant"]
                     and s.sample != item["sample"] and getattr(s, want) is not None
                     for s in samples)
        if not filled:
            out.append(item)
    return out


def _rebuy_partial(partial: Sequence[dict[str, Any]], readouts: Sequence[ReadOut],
                   samples: Sequence[RouteSample], read) -> dict[str, Any]:
    """Re-ask ONE reader that came back with half an answer. Exactly one call, ever.

    Re-asking the reader that actually looked is cheaper and better evidence than buying a fresh
    family: the missing half is the only thing in doubt. It is capped at one call per cell so a
    figure nobody can read cannot spend the run's budget proving it.
    """
    missing_mean = [item for item in partial if item["missing"] == "mean"]
    if not missing_mean:                          # a missing spread is not a missing witness
        return {"bought": False, "why": "no route was missing the mean"}
    first = missing_mean[0]
    used = {(r.model, r.variant, r.sample) for r in readouts}
    nxt = max((s for (m, v, s) in used if m == first["model"] and v == first["variant"]),
              default=0) + 1
    spec = ReadoutSpec(model=first["model"], variant=first["variant"], sample=nxt)
    if (spec.model, spec.variant, spec.sample) in used:   # pragma: no cover - defensive
        return {"bought": False, "why": "that reader has already been re-asked"}
    read(spec)
    filled = any(s.route == "D" and s.group == first["group"] and s.model == spec.model
                 and s.variant == spec.variant and s.sample == spec.sample and s.mean is not None
                 for s in samples)
    return {"bought": True, "route": first["route"], "group": first["group"],
            "spec": spec.to_dict(), "resolved": filled,
            "why": (f"{first['route']} reported a spread for group {first['group']} and no mean; "
                    f"the missing half is re-asked of the reader that produced the other half "
                    f"before this cell goes to a human")}


def _mark_illegible(reading: ReadOut, samples: list[RouteSample]) -> None:
    """A reader that could not see the target, or built its own ladder, does not vote."""
    if reading.calibration_source == CAL_SOURCE_INFERRED:
        why = ("this reading's axis was inferred rather than read off printed labels, so its "
               "numbers are its own construction and cannot corroborate anyone else's")
    elif reading.target_visible == VISIBLE_NO:
        why = ("this reader reports the named target is not in this image"
               + (f" ({reading.target_visible_reason})" if reading.target_visible_reason else ""))
    elif reading.target_visible == VISIBLE_PARTIAL:
        why = ("this reader could see only part of the named target, so its value is a reading "
               "of part of a panel and does not vote"
               + (f" ({reading.target_visible_reason})" if reading.target_visible_reason else ""))
    else:
        return
    for sample in samples:
        _drop(sample, DROPPED_BY_LEGIBILITY, why)
        sample.extra[ILLEGIBLE] = True

#: how a locator names a panel: "Fig. 3a", "Figure 2, panel a", "(b)", "panel B"
_PANEL_RES = [
    re.compile(r"\bpanels?\s+\(?([a-h])\b", re.I),
    # "Fig. 3a" — the letter must be welded to the number. A space between them is prose:
    # "Fig. 2, a comparison of ..." names no panel, and reading one out of it is the guess this
    # function exists to refuse.
    re.compile(r"\bfig(?:ure)?s?\.?\s*s?\d+([a-h])(?![a-z0-9])",
                             re.I),
    re.compile(r"\bfig(?:ure)?s?\.?\s*s?\d+\s*[.,]?\s*\(([a-h])\)",
                             re.I),
    re.compile(r"[(\[]([a-h])[)\]]", re.I),
]


def panel_named(*texts: str) -> str:
    """The panel letter a locator names, or "" — the map's own words, never a guess.

    "Fig. 3a", "Figure 2, panel a ('adaptive shift')" and "(b)" all name a panel; "Fig. 1" does
    not. Nothing is inferred from position: a figure whose panel is not named keeps the whole
    region, because picking one panel out of three on no evidence is exactly the guess this rule
    exists to prevent.
    """
    for text in texts:
        for pattern in _PANEL_RES:
            match = pattern.search(str(text or ""))
            if match:
                return match.group(1).lower()
    return ""


def panel_view(fig: FigureRegion, panel: PanelRegion) -> FigureRegion:
    """`fig` as seen through ONE of its panels: same paper, same page, the panel's own rect.

    Every route downstream works off `fig.bbox` and `fig.crop_png` (the vector scene, the CV pass,
    the overlay), so handing them a panel-shaped `FigureRegion` is what makes "read panel b" mean
    the pixels of panel b rather than the union of three panels and three ladders.
    """
    from dataclasses import replace

    return replace(fig, id=panel.id, bbox=panel.bbox, crop_png=panel.crop_png,
                   claude_png=panel.claude_png, crop_dpi=panel.crop_dpi or fig.crop_dpi,
                   claude_scale=panel.claude_scale, panels=[panel])


def resolve_panel(fig: FigureRegion, target: TargetSpec,
                  source: Source | None) -> tuple[FigureRegion, dict[str, Any]]:
    """Which image this cell is actually read from, and why. `(figure_or_panel, provenance)`.

    A multi-panel union is 338 pt tall on Heuer 2008 Fig. 2 and carries three different y ladders
    plus an x ladder; a value read off it can be calibrated with the wrong one, and the digitiser's
    own record shows that happening (Cressman Fig. 3b, "the bar top measured with the neighbouring
    panel's ladder"). The panel rect is the smallest unit ingestion can name — not a guarantee of
    exactly one ladder: Heuer's Fig. 3 resolves to a single "panel" that holds three
    (['-20','-10','0','10'], ['-60','-40','-20','0'], ['-60','-40','-20','0']), which is why a
    named panel that could NOT be isolated is recorded rather than described as isolated.
    """
    enumerated = list(fig.caption_panels or caption_panels(fig.caption or ""))
    info: dict[str, Any] = {"figure_id": fig.id, "panel_named": "", "panel_used": "",
                            "n_panels": len(fig.panels), "text_layer": fig.text_layer,
                            "caption_enumerates": enumerated}
    letter = panel_named(target.panel_hint, source.locator if source is not None else "",
                         target.series_hint)
    info["panel_named"] = letter
    if len(fig.panels) <= 1:
        panel = fig.panels[0] if fig.panels else None
        if letter:
            _record_not_isolated(info, fig, letter, enumerated)
        else:
            info["panel_why"] = ("nothing names a panel and ingestion found one rect; the region "
                                 "and the panel are the same rect")
        if panel is not None and not panel.calibrated:
            info["uncalibrated"] = True
            info["panel_used"] = panel.id
            info["panel_numeric_labels"] = panel.n_numeric
            info["panel_ladder_labels"] = panel.n_ladder
        return fig, info
    if not letter:
        info["panel_why"] = (f"nothing names a panel of this {len(fig.panels)}-panel figure, so "
                             f"the whole region is read; a panel picked on no evidence would be a "
                             f"guess about which axis calibrates the value")
        return fig, info
    match = next((p for p in fig.panels if p.letter == letter), None)
    if match is None:
        _record_not_isolated(info, fig, letter, enumerated)
        return fig, info
    info["panel_used"] = match.id
    info["panel_numeric_labels"] = match.n_numeric
    info["panel_ladder_labels"] = match.n_ladder
    info["panel_why"] = f"the locator names panel {letter!r}, which is its own rect in the figure"
    if not match.calibrated:
        info["uncalibrated"] = True
    return panel_view(fig, match), info


def _record_not_isolated(info: dict[str, Any], fig: FigureRegion, letter: str,
                         enumerated: Sequence[str]) -> None:
    """C1 rule 3's third clause: a named panel that could not be isolated is a FLAGGED read.

    The rule says "never the multi-panel union" and both it and its acceptance tests assume the
    decomposition succeeds. On this corpus it fails on the one multi-panel figure that feeds
    pooled cells — Cressman 2010's Fig. 3 is a single drawing cluster, and its map locators are
    "Fig. 3a" and "Fig. 3b" — so the code was left with nothing to do about it and DENIED the
    problem instead, writing "the figure is one panel; the region and the panel are the same
    rect" over a two-panel figure. That sentence is false, and it is false on the two cells this
    run pools. Nothing else catches it: a reader handed that union answers `target_visible: yes`
    (panel b IS in the image) and `calibration_source: printed_labels` (the labels ARE printed),
    so C2 sees nothing, and the axis-identity nets need the readers to disagree with each other —
    three readers making the same mistake sail through.

    Refusing the cell would be worse than reading it: the union is readable, and the axis-identity
    rules already repaired Cressman Fig. 3 once. So the reading is taken and the fact is recorded,
    with the caption's own enumeration as corroboration where the caption enumerates anything.
    """
    n = len(fig.panels)
    info[PANEL_NOT_ISOLATED] = letter
    info["needs_review"] = True
    said = (f"; the caption enumerates {', '.join(enumerated)}" if enumerated else "")
    found = ([p.letter for p in fig.panels] if n else [])
    info["panel_why"] = (
        f"the locator names panel {letter!r} and ingestion could not hand that panel over on its "
        f"own (it resolved {fig.id} to {n} rect(s) {found}{said}), so the image read is the whole "
        f"region — which carries every panel's axes, and a value read off it can be scaled with "
        f"the wrong ladder")
    info["needs_review_reason"] = info["panel_why"]


def _no_value(fig: FigureRegion, target: TargetSpec, paper: PaperRecord, source: Source | None,
              dataset: DatasetSpec | None, crop: Path, reason: str, provenance: dict[str, Any],
              result: bool) -> list[Candidate] | DigitizeResult:
    """One ensemble candidate per group, carrying no number and the reason there is none."""
    out = [_candidate(None, None, group=group, sample=None, target=target, fig=fig, paper=paper,
                      dataset=dataset, source=source,
                      kind=(source.kind if source is not None and source.kind is not None
                            else SourceKind.figure_line),
                      mapper_type=(source.error_bar_type if source is not None
                                   else DispersionType.UNKNOWN),
                      unit=target.unit_hint,
                      page=(source.page if source is not None and source.page else fig.page),
                      locator=(source.locator if source is not None and source.locator
                               else target.panel_hint or fig.label or fig.id),
                      crop=crop, overlay_path="", extractor_id="digitize:ensemble", sigma=None,
                      status="ambiguous", notes=reason, provenance=provenance, call_id="",
                      model="")
           for group in GROUPS]
    if result:
        return DigitizeResult(candidates=out, samples=[], calibration=None, overlay_path="",
                              cost_usd=provenance.get("cost_usd", 0.0), provenance=provenance)
    return out


def _panel_uncalibrated(fig: FigureRegion, target: TargetSpec, paper: PaperRecord,
                        source: Source | None, dataset: DatasetSpec | None, crop: Path,
                        panel_info: dict[str, Any], result: bool
                        ) -> list[Candidate] | DigitizeResult:
    """One ensemble candidate per group saying the named panel carries no ladder of its own.

    `n_ladder` counts the rungs of the longest printed LADDER inside THIS panel's rect — a
    roughly collinear, value-monotone column, not three bare numbers anywhere in the rect. Fewer
    than `MIN_PANEL_NUMERIC` and there is no axis to calibrate against: whatever a reader returned
    would be scaled from a neighbouring panel's ladder, which is a different quantity in the same
    units. A figure that carries no ladder ANYWHERE (a scanned raster, a setup schematic) never
    reaches here: the assertion has no premise there and is skipped rather than failed.
    """
    named = panel_info.get("panel_used") or fig.id
    reason = (f"{named} carries a printed ladder of "
              f"{panel_info.get('panel_ladder_labels', 0)} label(s) inside its own rect, fewer "
              f"than the {MIN_PANEL_NUMERIC} an axis needs to be calibrated from the figure's "
              f"own text. A value read here would be scaled with a ladder that belongs "
              f"to another panel. Re-acquire the crop (the whole page is the fallback) rather "
              f"than paying a reader for a number the picture cannot support")
    provenance = {"figure_id": fig.id, "figure_kind": fig.kind, "crop_dpi": fig.crop_dpi,
                  PANEL_UNCALIBRATED: True, "panel": panel_info, "needs_review": True,
                  "needs_review_reason": reason, "prompt_version": PROMPT_VERSION,
                  "readouts_bought": 0}
    return _no_value(fig, target, paper, source, dataset, crop, reason, provenance, result)


def _target_not_visible(fig: FigureRegion, target: TargetSpec, paper: PaperRecord,
                        source: Source | None, dataset: DatasetSpec | None, crop: Path,
                        seen: dict[str, Any], readouts: Sequence[ReadOut],
                        panel_info: dict[str, Any], result: bool,
                        coord: CoordReadout | None = None,
                        verify_log: Sequence[dict[str, Any]] = ()
                        ) -> list[Candidate] | DigitizeResult:
    """The readers looked and the thing is not there — so the cell re-acquires, it does not guess.

    The alternative is what happened before this rule: two readers say the panel is not in the
    crop, a third returns numbers off whatever IS in the crop, and the third one's numbers become
    the cell's value with three error flags attached to them.
    """
    who = ", ".join(seen["target_not_visible"])
    why = "; ".join(x for x in seen["target_not_visible_why"] if x)
    reason = (f"{len(seen['target_not_visible'])} of {seen['n_readouts']} readers report that the "
              f"target named for this cell is not in {fig.id}'s image ({who})"
              + (f": {why}" if why else "")
              + ". A reading taken from this crop would be of something else, so the crop is "
                "re-acquired rather than the minority reading adopted")
    provenance = {"figure_id": fig.id, "figure_kind": fig.kind, "crop_dpi": fig.crop_dpi,
                  PANEL_NOT_IN_CROP: True, "panel": panel_info, "legibility": seen,
                  "needs_review": True, "needs_review_reason": reason,
                  "prompt_version": PROMPT_VERSION, "readouts_bought": len(readouts),
                  # every call this cell paid for, not only the read-outs. The second legibility
                  # check runs AFTER `coords()` and after any extra read-out and re-read, and
                  # reporting only the read-outs there is E10's "reported cost is not incurred
                  # cost" reintroduced in new code.
                  "cost_usd": (sum(r.cost_usd for r in readouts)
                               + (coord.cost_usd if coord is not None else 0.0)
                               + sum(float(e.get("cost_usd", 0.0)) for e in verify_log)),
                  "readout_families": sorted({r.model for r in readouts})}
    return _no_value(fig, target, paper, source, dataset, crop, reason, provenance, result)


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
        "cal_disputed": list(choice.disputed),
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
    """The unit the READERS say the axis is in; the mapper's hint only when none of them said.

    The mapper's hint is the outcome's unit, which is not the same thing as the unit of the axis
    a reading came off. Cressman's Fig. 3b was read twice, once off its degrees axis and once off
    the percentage axis beside it, and both ensembles were stamped with the outcome's "degrees" —
    so nothing downstream could tell them apart, and 18.5° and 61.5% went into one vote. Two
    readers naming the same unit outrank the hint; a lone reader's unit is taken when the hint
    is empty or agrees with it in kind.
    """
    from ..verify.units import unit_key

    named = [r.unit for r in readouts if str(r.unit or "").strip()]
    if named:
        keys = [unit_key(u) for u in named]
        best = max(set(keys), key=keys.count)
        if best and (keys.count(best) >= 2 or not target.unit_hint
                     or unit_key(target.unit_hint) in ("", best)):
            return next(u for u, k in zip(named, keys) if k == best)
    return target.unit_hint or (named[0] if named else "")


def _collapse_provenance(samples: Sequence[RouteSample]) -> dict[str, Any]:
    """What these samples — and only these — say about averaging across the x axis.

    Asked once per candidate rather than once per cell. `collapsed_across_x` is a claim about the
    number on THIS row: a pool where one reader averaged eight target directions and another read
    the point the locator named contains both kinds of row, and stamping the cell's answer on
    every one of them puts `collapsed_across_x` beside `categorical_point_read` — the pair
    DECISION D3 says cannot co-exist (whole-branch review, MAJOR beside BLOCKER 1).
    """
    collapsed = [s for s in samples if s.extra.get("collapsed_across_x")]
    return {"collapsed_across_x": bool(collapsed),
            "n_points": (min(int(s.extra.get("n_points") or 0) for s in collapsed)
                         if collapsed else None),
            "dispersion_approximation": MEAN_OF_POINT_SD if collapsed else ""}


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
                            # the CELL's base says a collapse happened somewhere in the pool;
                            # this row is one reader, and D3 forbids `collapsed_across_x` beside
                            # a point read. Every candidate answers for its own sample, the way
                            # the ensemble below already answers for its own group.
                            **_collapse_provenance([s]),
                            "tool_calls": summarize_tool_calls(s.tool_calls),
                            "dropped": s.dropped,
                            "dropped_by": str(s.extra.get("dropped_by", ""))},
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
                            **_collapse_provenance(mine),
                            "per_route": [s.to_dict() for s in mine],
                            "dropped_samples": [s.to_dict() for s in mine if s.dropped],
                            "dropped_by": ", ".join(sorted({str(s.extra.get("dropped_by", ""))
                                                            for s in mine if s.dropped}
                                                           - {""})),
                            "tool_calls": _aggregate_tool_calls(mine)},
                call_id=_verify_call_id(base), model=""))
            continue

        live, zero_notes = _drop_zero_confidence(live)
        means = [s.mean for s in live]
        # a route that found no whisker does not get a vote on the whisker's length
        with_error = [s for s in live if s.error is not None]
        errors = all_errors = [s.error for s in with_error]
        topology_note = _dispersion_topology_conflict(with_error, mapper_type)
        mean, mad_sigma = ensemble_stats(means)
        error, error_mad = ensemble_stats(errors) if errors else (None, 0.0)
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
        # C1 rule 3, third clause: the map named a panel, ingestion could not isolate it, and the
        # value was therefore read off a rect carrying more axes than one. Recorded on the row,
        # because it is exactly the condition under which a reading can be scaled with the
        # neighbouring panel's ladder and every route agree about it.
        not_isolated = (base.get("panel") or {}).get(PANEL_NOT_ISOLATED) or ""
        if not_isolated:
            reasons.append((base.get("panel") or {}).get("needs_review_reason")
                           or f"panel {not_isolated!r} could not be isolated from this figure")
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
        provenance = {
            **base,
            # `live`, never `mine`: a vote may only credit readers that cast a ballot. The
            # confidence score pays +0.25 for "two independent model families read it and
            # agreed", and `mine` includes readers whose `mean` is None — in `runs/proof`,
            # Bock group A was credited with a sonnet reader that produced no value, and
            # that phantom family is the whole reason group A was released while group B,
            # on the same figure, was withheld.
            "model_families": model_families(live),
            **_collapse_provenance(mine),
            "legend_says": legend_text,
            "legend_dispersion": legend_type.value if legend_type else None,
            "mapper_dispersion": mapper_type.value if mapper_type else None,
            "dispersion_type_from": dispersion_from,
            "per_route": [s.to_dict() for s in mine],
            "route_values": {s.extractor_id: {"mean": s.mean, "error": s.error} for s in live},
            "snap_confidences": {s.extractor_id: s.snap_conf for s in mine
                                 if s.snap_conf is not None},
            "n_routes": len(live), "n_routes_with_error": len(with_error),
            "dispersion_topology_conflict": bool(topology_note),
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
            PANEL_NOT_ISOLATED: bool(not_isolated),
            "needs_review": (status == "ambiguous" or dispersion_only or no_calibration
                             or bool(topology_note) or bool(not_isolated)),
            "needs_review_kind": ("mean" if status == "ambiguous"
                                  else ("dispersion" if dispersion_only or topology_note
                                        else ("calibration" if no_calibration or not_isolated
                                              else None))),
            "needs_review_reason": "; ".join(reasons),
            "dropped_samples": [s.to_dict() for s in mine if s.dropped],
            "dropped_by": ", ".join(sorted({str(s.extra.get("dropped_by", ""))
                                            for s in mine if s.dropped} - {""})),
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
    this figure. Losing every route to a drop is a claim about US: the datum is there, we could
    not read it reliably. That is `ambiguous`, and conflating the two would let a failed read
    silently exclude a study from the meta-analysis.

    WHICH stage dropped them is the second thing this line has to get right, and for a year it
    did not: the cause was hard-coded to "overlay verification" whatever had actually happened,
    so Vachon's Fig 4 — nulled by the categorical rule, untouched by the overlay layer — reported
    the overlay as its cause and sent a whole-run audit to the wrong stage. Every sample's own
    `drop_reason` is named here instead, distinct ones joined in the order they were dropped.
    """
    if not mine:
        return "ambiguous", "no route sample was produced for this group"
    if any(s.dropped for s in mine):
        dropped = [s.extractor_id for s in mine if s.dropped]
        reasons = list(dict.fromkeys(s.drop_reason.strip() for s in mine
                                     if s.dropped and s.drop_reason.strip()))
        why = "; ".join(reasons) or "no stage recorded why"
        return "ambiguous", (f"every usable route sample was dropped ({', '.join(dropped)}): "
                             f"{why}; the datum is on the page but we could not read it reliably")
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
