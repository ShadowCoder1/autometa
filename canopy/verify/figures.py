"""What a digitised candidate says about the figure it was read from — parsed in ONE place.

`canopy.digitize` records a lot in `Candidate.pixel_provenance`, and both the checks and the vote
need the same three things out of it: is this value from a picture, what value range was the
picture drawn over, and how far apart may two reads of it be before they disagree. Parsing that in
two modules invited them to drift apart (and to expect keys the digitizer never writes), so it
lives here.

The digitizer's own record is the best source and is preferred in this order:

1. `pixel_provenance["agreement"]["mean_tolerance"]` — the dual tolerance the digitizer already
   computed for this cell, i.e. exactly max(2% of the axis range, half a tick);
2. `pixel_provenance["axis_range"]` (a span in data units) with the tick spacing implied by
   `pixel_provenance["cal"]["ticks"]`, which are `[pixel, value]` pairs;
3. explicit `y_min` / `y_max` / `y_tick` keys, which no digitizer route writes today but which a
   hand-built candidate (and every test in this repo) may;
4. the `axis_range` argument the caller passes.

Nothing here computes a statistic and nothing raises: a figure whose calibration failed simply
returns `None` for the parts that are unknown, and the callers say so.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from ..models import Candidate, SourceKind

__all__ = ["FIGURE_KINDS", "is_figure", "FigureCalibration", "figure_calibration", "axis_limits",
           "figure_tolerance", "calibration_status", "routes_agree", "AXIS_FRACTION",
           "TICK_FRACTION", "FALLBACK_FRACTION", "offer_gate", "GATE_MARGIN_TICKS",
           "GATE_MARGIN_FRACTION", "PLOTBOX_SANE_FACTOR"]

AXIS_FRACTION = 0.02            # a digitised mean may differ by 2% of the axis range …
TICK_FRACTION = 0.5             # … or half a tick, whichever is looser (amendment F)
FALLBACK_FRACTION = 0.02        # a figure with no calibration at all: 2% of the value itself

FIGURE_KINDS: frozenset[SourceKind] = frozenset({
    SourceKind.figure_bar, SourceKind.figure_line, SourceKind.figure_points,
    SourceKind.figure_box})

#: key spellings for an explicit value axis (hand-built candidates; the digitizer writes none of
#: these, it writes `cal.ticks` and `axis_range` — see `figure_calibration`)
_MIN_KEYS = ("y_min", "ymin", "min", "axis_min", "value_min")
_MAX_KEYS = ("y_max", "ymax", "max", "axis_max", "value_max")
_TICK_KEYS = ("y_tick", "ytick", "tick", "tick_spacing", "y_tick_spacing")


def is_figure(cand: Candidate) -> bool:
    """A value read off a picture rather than out of characters."""
    return (cand.source_kind in FIGURE_KINDS
            or cand.extractor_id.startswith("digitize:")
            or cand.route.startswith(("figure", "digitize")))


@dataclass
class FigureCalibration:
    """The parts of a figure's calibration the verification layer uses."""

    low: float | None = None          # lowest labelled tick value
    high: float | None = None         # highest labelled tick value
    span: float | None = None         # the plotted value range (may exceed the tick range)
    tick: float | None = None         # median spacing between labelled ticks
    mean_tolerance: float | None = None   # the digitizer's own dual tolerance, when it recorded one
    source: str = "none"              # agreement | cal_ticks | explicit_keys | axis_range | none

    @property
    def tick_span(self) -> float | None:
        if self.low is None or self.high is None:
            return None
        return abs(self.high - self.low)

    @property
    def slack(self) -> float:
        """How far outside the LABELLED ticks a value may still be inside the drawn axis.

        The digitizer reports `axis_range` from the plot box when it can, and a plot box is usually
        larger than the range its labelled ticks cover. Treating the tick range as the axis would
        flag values that are plainly inside the frame, so the extra span is handed out as slack.
        """
        span, ticks = self.span, self.tick_span
        if span is None or ticks is None or span <= ticks:
            return 0.0
        return (span - ticks) / 2


def _number(node: Any) -> float | None:
    return float(node) if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def _tick_values(cal: Any) -> list[float]:
    """The value half of `cal["ticks"]`, which the digitizer writes as `[pixel, value]` pairs."""
    if not isinstance(cal, dict):
        return []
    values: list[float] = []
    for entry in cal.get("ticks") or []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            value = _number(entry[1])
            if value is not None:
                values.append(value)
    return sorted(values)


def _explicit(cal: Any) -> tuple[float | None, float | None, float | None]:
    if not isinstance(cal, dict):
        return None, None, None
    axis = cal.get("y") if isinstance(cal.get("y"), dict) else cal
    low = next((_number(axis.get(k)) for k in _MIN_KEYS if _number(axis.get(k)) is not None), None)
    high = next((_number(axis.get(k)) for k in _MAX_KEYS if _number(axis.get(k)) is not None), None)
    tick = next((_number(axis.get(k)) for k in _TICK_KEYS if _number(axis.get(k)) is not None),
                None)
    if low is not None and high is not None and low > high:
        low, high = high, low
    return low, high, tick


def figure_calibration(pixel_provenance: dict[str, Any] | None,
                       axis_range: float | None = None) -> FigureCalibration:
    """Everything the verification layer knows about one figure's value axis."""
    provenance = pixel_provenance if isinstance(pixel_provenance, dict) else {}
    cal = provenance.get("cal")
    out = FigureCalibration()

    ticks = _tick_values(cal)
    low, high, tick = _explicit(cal)
    if ticks:
        out.low, out.high, out.source = ticks[0], ticks[-1], "cal_ticks"
        gaps = [b - a for a, b in zip(ticks, ticks[1:]) if b > a]
        out.tick = float(statistics.median(gaps)) if gaps else None
    if out.low is None and low is not None and high is not None:
        out.low, out.high, out.source = low, high, "explicit_keys"
    if out.tick is None and tick is not None:
        out.tick = tick

    span = _number(provenance.get("axis_range"))
    if span is None or span <= 0:
        span = out.tick_span
    if (span is None or span <= 0) and axis_range:
        span, out.source = abs(float(axis_range)), out.source if out.source != "none" \
            else "axis_range"
    out.span = span if span and span > 0 else None

    agreement = provenance.get("agreement")
    if isinstance(agreement, dict):
        recorded = _number(agreement.get("mean_tolerance"))
        if recorded is not None and recorded > 0:
            out.mean_tolerance, out.source = recorded, "agreement"
    return out


#: how corroborated the axis a digitised value was read against is — written by
#: `digitize._choose_calibration`, read by the checks. An older record (or a hand-built candidate)
#: carries none, and "unknown" is the honest answer for it: it is neither confirmed nor refuted.
CAL_STATUSES = ("confirmed", "single_witness", "cal_refuted", "none")


def calibration_status(pixel_provenance: dict[str, Any] | None) -> str:
    """`confirmed` | `single_witness` | `cal_refuted` | `none` | `unknown` for one candidate."""
    provenance = pixel_provenance if isinstance(pixel_provenance, dict) else {}
    status = provenance.get("cal_status")
    return status if status in CAL_STATUSES else "unknown"


def routes_agree(pixel_provenance: dict[str, Any] | None) -> bool | None:
    """Did the digitizer's own routes agree about the MEAN? `None` when it did not say."""
    provenance = pixel_provenance if isinstance(pixel_provenance, dict) else {}
    agreed = provenance.get("mean_agreement")
    if isinstance(agreed, bool):
        return agreed
    agreement = provenance.get("agreement")
    if isinstance(agreement, dict) and isinstance(agreement.get("mean_agrees"), bool):
        return bool(agreement["mean_agrees"])
    return None


def axis_limits(pixel_provenance: dict[str, Any] | None) -> tuple[float, float] | None:
    """`(low, high)` of the value axis a digitised candidate was read against, with slack.

    Deliberately generous: the bounds come from the LABELLED ticks, widened by however much the
    plotted axis exceeded them, so `value_outside_axis` only fires on a value that could not have
    been drawn in that frame at all.
    """
    cal = figure_calibration(pixel_provenance)
    if cal.low is None or cal.high is None:
        return None
    slack = cal.slack
    return cal.low - slack, cal.high + slack


def figure_tolerance(cand: Candidate, axis_range: float | None = None) -> float | None:
    """max(2% of the axis range, half a tick) — `None` when nothing calibrates this figure."""
    cal = figure_calibration(cand.pixel_provenance, axis_range)
    if cal.mean_tolerance is not None:
        return cal.mean_tolerance
    options = [AXIS_FRACTION * cal.span] if cal.span else []
    if cal.tick:
        options.append(TICK_FRACTION * abs(cal.tick))
    return max(options) if options else None


# ------------------------------------------------------------- ticket 3: the offer gate
#: one full labelled-tick interval past the outermost label. An axis frame extends past its
#: labels by at most about one labelled interval — publishers either label the frame's end or
#: stop one division short, and a datum drawn beyond that would have forced another printed
#: label. Clipped bars and whiskers render AT the frame edge, i.e. inside this same interval,
#: so a clipped read passes the gate.
GATE_MARGIN_TICKS = 1.0
#: the floor for sparsely labelled ladders: 5% of the labelled span
GATE_MARGIN_FRACTION = 0.05
#: a "plot box" more than twice its labelled ladder is not one panel's frame: a real single
#: panel whose frame exceeded its labels by a full extra ladder-length would have half its area
#: unlabelled, which is not a shape publishers print. (Measured over a real corpus as
#: corroboration: sane single-panel frames ran ~1.0–1.3× their tick span; the one that ran 5×
#: was a multi-panel crop whose box handed −147.9 a pass on a −30..30 axis.)
PLOTBOX_SANE_FACTOR = 2.0


def offer_gate(pixel_provenance: dict[str, Any] | None, mean: Any) -> dict[str, Any] | None:
    """`None` when `mean` could plausibly be drawn in this frame (or nothing reliable says
    otherwise); else a record naming the bounds and why. Fails OPEN on every doubt: a wrong
    calibration must never suppress a right value.

    This is deliberately NOT `axis_limits` + `value_outside_axis`: those keep their generous
    semantics (conviction stays hard — `error`, unconditional). The gate is a cheaper-severity
    screen with strictly harder RELIABILITY preconditions, which is the only combination that
    closes the plot-box hole without risking the misread-ladder failure ("a correct read sent
    to a human by a wrong calibration"):

    - the calibration must be `confirmed` (two independent witnesses agreed on the mapping —
      `single_witness`, `cal_refuted`, `none` and old records all leave the gate inert);
    - linear only — a one-tick additive margin is meaningless on a log ladder;
    - at least three labelled ticks — a two-point fit is exact through its rungs and verifies
      nothing between them, so it may not exclude anybody;
    - the plot-box slack `axis_limits` hands out is honoured ONLY when the box is believable:
      not a `panel_not_isolated` crop (the box then spans the neighbours' panels — the exact
      hole the −147.9 case fell through), and no more than `PLOTBOX_SANE_FACTOR` times the
      labelled span.
    """
    provenance = pixel_provenance if isinstance(pixel_provenance, dict) else {}
    if calibration_status(provenance) != "confirmed":
        return None
    cal = provenance.get("cal")
    if not isinstance(cal, dict) or str(cal.get("scale") or "") != "linear":
        return None
    ticks = _tick_values(cal)
    if len(ticks) < 3:
        return None
    tick_low, tick_high = ticks[0], ticks[-1]
    gaps = [b - a for a, b in zip(ticks, ticks[1:]) if b > a]
    tick_span = tick_high - tick_low
    if not gaps or tick_span <= 0:
        return None
    tick = float(statistics.median(gaps))
    margin = max(GATE_MARGIN_TICKS * tick, GATE_MARGIN_FRACTION * tick_span)
    low, high = tick_low - margin, tick_high + margin
    # per-route candidates carry the panel record nested under "panel"; ensembles lift it to
    # the top level — read both spellings or the gate trusts exactly the boxes it must not
    panel = provenance.get("panel")
    not_isolated = bool(provenance.get("panel_not_isolated")
                        or (isinstance(panel, dict) and panel.get("panel_not_isolated")))
    source = str(provenance.get("axis_range_source") or "")
    span = _number(provenance.get("axis_range"))
    box_note = ""
    if source == "plot_bbox" and span:
        if not not_isolated and span <= PLOTBOX_SANE_FACTOR * tick_span:
            limits = axis_limits(provenance)
            if limits is not None:
                low, high = min(low, limits[0]), max(high, limits[1])
        else:
            box_note = (f"; the wider plot-box span ({span:g}) was not trusted because "
                        + ("the crop holds more than one panel" if not_isolated else
                           f"it exceeds {PLOTBOX_SANE_FACTOR:g}× the labelled ladder"))
    value = _number(mean)
    if value is None or low <= value <= high:
        return None
    return {"low": low, "high": high, "margin": margin,
            "tick_low": tick_low, "tick_high": tick_high, "axis_range_source": source,
            "why": (f"the value {value:g} lies outside this panel's printed tick range "
                    f"[{tick_low:g}, {tick_high:g}] (margin ±{margin:g}){box_note}")}
