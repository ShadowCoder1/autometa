"""Agreement vote (spec §3.3(2), amendment G) — pure code, no model.

A number is believed when two readers that fail *differently* wrote it down. "Differently" is
`route_key` = modality (text / table / one digitizer path / statistic) × model family: two prompt
variants of the same model on the same pages share a failure mode, so they are one voter, and
their answers are collapsed to that route's median before anything is compared.

Tolerance comes from the evidence, not from a constant:

* characters — half of the last digit the paper actually printed ("31.5" tolerates ±0.05);
* pictures   — max(2% of the axis range, half a tick), the resolution a reader can honestly claim.

Agreement across at least two routes is *accepted by vote*. Two text extractors that disagree set
`needs_third_candidate`, which the orchestrator satisfies by running a third cheap candidate
(amendment G) before anything is escalated to the adjudicator.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any, Iterable, Literal, Sequence

from pydantic import Field

from ..models import CanopyModel, Candidate, DispersionType, GroupKey, SourceKind
from .grounding import is_short_quote as _is_short_quote

__all__ = ["vote", "vote_groups", "VoteResult", "RouteValue", "route_key", "modality",
           "model_family", "precision_tolerance", "figure_tolerance", "AXIS_FRACTION",
           "TICK_FRACTION"]

AXIS_FRACTION = 0.02            # a digitised mean may differ by 2% of the axis range …
TICK_FRACTION = 0.5             # … or half a tick, whichever is looser (amendment F)
FALLBACK_FRACTION = 0.02        # a figure with no calibration at all: 2% of the value itself
MAX_DECIMALS = 12               # beyond this a "printed" precision is float noise

_FIGURE_KINDS = frozenset({SourceKind.figure_bar, SourceKind.figure_line, SourceKind.figure_points,
                           SourceKind.figure_box})
_AXIS_MIN_KEYS = ("y_min", "ymin", "min", "axis_min", "value_min")
_AXIS_MAX_KEYS = ("y_max", "ymax", "max", "axis_max", "value_max")
_TICK_KEYS = ("y_tick", "ytick", "tick", "tick_spacing", "y_tick_spacing")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


# ----------------------------------------------------------------------------- routes
def model_family(model: str) -> str:
    """`claude-opus-5-20260401` → `claude-opus`: the family, without version or date."""
    name = (model or "").split("/")[-1]
    parts = [p for p in name.split("-") if p]
    while parts and parts[-1].isdigit():
        parts.pop()
    return "-".join(parts) or name


def modality(cand: Candidate) -> str:
    """How this value was obtained — the half of a route that is not the model."""
    if cand.extractor_id.startswith("digitize:"):
        pieces = cand.extractor_id.split(":")
        return f"figure:{pieces[1]}" if len(pieces) > 1 and pieces[1] else "figure"
    if cand.source_kind in _FIGURE_KINDS:
        return "figure"
    if cand.kind in ("test_statistic", "reported_d"):
        return "statistic"
    if cand.source_kind is SourceKind.table:
        return "table"
    return "text"


def route_key(cand: Candidate) -> str:
    return f"{modality(cand)}/{model_family(cand.model)}"


def is_text_route(key: str) -> bool:
    return key.split("/", 1)[0] in ("text", "table")


def is_figure_route(key: str) -> bool:
    return key.split("/", 1)[0].startswith("figure")


# ----------------------------------------------------------------------------- tolerances
def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def precision_tolerance(cand: Candidate) -> float:
    """Half of the last digit the paper printed for this candidate's mean."""
    value = cand.mean
    if value is None:
        return 0.0
    tokens = _NUMBER_RE.findall(cand.value_as_written or "")
    decimals: int | None = None
    seen: list[int] = []
    for token in tokens:
        places = _decimals(token)
        seen.append(places)
        if places <= MAX_DECIMALS and round(value, places) == float(token):
            decimals = places
            break
    if decimals is None:
        decimals = min(seen) if seen else _decimals(repr(float(value)))
    return 0.5 * 10 ** (-min(decimals, MAX_DECIMALS))


def axis_calibration(cand: Candidate) -> dict[str, Any]:
    cal = (cand.pixel_provenance or {}).get("cal")
    if not isinstance(cal, dict):
        return {}
    axis = cal.get("y")
    return axis if isinstance(axis, dict) else cal


def figure_tolerance(cand: Candidate, axis_range: float | None = None) -> float | None:
    """max(2% of the axis range, half a tick) — `None` when nothing calibrates this figure."""
    cal = axis_calibration(cand)
    low = next((cal[k] for k in _AXIS_MIN_KEYS if isinstance(cal.get(k), (int, float))), None)
    high = next((cal[k] for k in _AXIS_MAX_KEYS if isinstance(cal.get(k), (int, float))), None)
    tick = next((cal[k] for k in _TICK_KEYS if isinstance(cal.get(k), (int, float))), None)
    span = abs(float(high) - float(low)) if low is not None and high is not None else axis_range
    options = [AXIS_FRACTION * abs(span) for span in ([span] if span else [])]
    if tick:
        options.append(TICK_FRACTION * abs(float(tick)))
    return max(options) if options else None


def candidate_tolerance(cand: Candidate, axis_range: float | None = None) -> float:
    """The tolerance this one candidate can honestly claim for its own mean."""
    if is_figure_route(route_key(cand)):
        tolerance = figure_tolerance(cand, axis_range)
        if tolerance is not None:
            return tolerance
        return FALLBACK_FRACTION * abs(cand.mean) if cand.mean else 0.0
    return precision_tolerance(cand)


# ----------------------------------------------------------------------------- result
class RouteValue(CanopyModel):
    """One voter: every candidate of one modality × model family, collapsed to their median."""

    route_key: str
    value: float | None = None
    dispersion_value: float | None = None
    n: int | None = None
    sigma: float | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    spread: float | None = None          # furthest candidate from this route's own median
    consistent: bool = True


class VoteResult(CanopyModel):
    """What the vote decided for one (dataset × outcome × group) cell."""

    group: GroupKey | None = None
    agreement: Literal["agree", "disagree", "single", "none"] = "none"
    mean: float | None = None
    dispersion_value: float | None = None
    dispersion_type: DispersionType = DispersionType.UNKNOWN
    ci_low: float | None = None
    ci_high: float | None = None
    points: list[float] = Field(default_factory=list)
    n: int | None = None
    unit: str = ""
    sigma: float | None = None
    mad: float | None = None
    #: True when every agreeing reader's quote is in the paper, False when one is not, None when
    #: the value came from a picture and there is no quote to ground
    grounded: bool | None = None
    short_quote: bool = False
    method: str = "none"
    tolerance: float | None = None
    agreeing_ids: list[str] = Field(default_factory=list)
    disagreeing_ids: list[str] = Field(default_factory=list)
    needs_third_candidate: bool = False
    routes: list[RouteValue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def value(self) -> float | None:
        """The agreed mean (`None` when the readers did not agree)."""
        return self.mean

    @property
    def accepted_by_vote(self) -> bool:
        return self.agreement == "agree"


# ----------------------------------------------------------------------------- the vote
def _usable(candidates: Iterable[Candidate], group: str | None) -> list[Candidate]:
    return [c for c in candidates
            if c.kind == "group_stats" and c.status == "found" and c.mean is not None
            and (group is None or c.group == group)]


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _mode(values: Sequence[Any]) -> Any | None:
    present = [v for v in values if v is not None and v != ""]
    return Counter(present).most_common(1)[0][0] if present else None


def _route_values(rows: Sequence[Candidate], tolerance: float,
                  notes: list[str]) -> list[RouteValue]:
    grouped: dict[str, list[Candidate]] = {}
    for cand in rows:
        grouped.setdefault(route_key(cand), []).append(cand)
    routes: list[RouteValue] = []
    for key in sorted(grouped):
        members = grouped[key]
        means = [c.mean for c in members]
        median = _median(means)
        spread = max(abs(m - median) for m in means)
        consistent = spread <= tolerance
        if not consistent:
            notes.append(f"route {key} disagrees with itself: {means} (spread {spread:.4g} > "
                         f"tolerance {tolerance:.4g}); its median {median:.4g} still votes")
        dispersions = [c.dispersion_value for c in members if c.dispersion_value is not None]
        sigmas = [c.sigma for c in members if c.sigma is not None]
        routes.append(RouteValue(
            route_key=key, value=median,
            dispersion_value=_median(dispersions) if dispersions else None,
            n=_mode([c.n for c in members]), sigma=_median(sigmas) if sigmas else None,
            candidate_ids=[c.candidate_id for c in members],
            spread=spread, consistent=consistent))
    return routes


def _best_cluster(routes: Sequence[RouteValue], tolerance: float) -> list[RouteValue]:
    """The largest set of routes that all sit within `tolerance` of one of them."""
    best: list[RouteValue] = []
    for seed in routes:
        cluster = [r for r in routes if abs(r.value - seed.value) <= tolerance]
        if (len(cluster), sum(len(r.candidate_ids) for r in cluster)) > \
                (len(best), sum(len(r.candidate_ids) for r in best)):
            best = cluster
    return best


def _fill_values(result: VoteResult, agreeing: Sequence[Candidate], notes: list[str]) -> None:
    """Dispersion, n, unit and digitisation sigma of the readers that carried the vote."""
    types = [c.dispersion_type for c in agreeing
             if c.dispersion_type not in (DispersionType.UNKNOWN, DispersionType.NONE)]
    chosen = _mode(types) or DispersionType.UNKNOWN
    if len({t for t in types}) > 1:
        notes.append("the readers that agreed on the mean disagree about the dispersion type "
                     f"({sorted({t.value for t in types})}); {chosen.value} carried the majority")
    result.dispersion_type = chosen
    matching = [c for c in agreeing if c.dispersion_type is chosen and c.dispersion_value is not None]
    if matching:
        result.dispersion_value = _median([c.dispersion_value for c in matching])
    lows = [c.ci_low for c in agreeing if c.ci_low is not None]
    highs = [c.ci_high for c in agreeing if c.ci_high is not None]
    result.ci_low = _median(lows) if lows else None
    result.ci_high = _median(highs) if highs else None
    result.points = next((list(c.points) for c in agreeing if c.points), [])
    result.n = _mode([c.n for c in agreeing])
    result.unit = _mode([c.unit for c in agreeing]) or ""
    sigmas = [c.sigma for c in agreeing if c.sigma is not None]
    result.sigma = _median(sigmas) if sigmas else None
    quoted = [c for c in agreeing if c.quote.strip()]
    if quoted:
        result.grounded = all(c.grounded is not False for c in quoted)
        result.short_quote = any(_is_short_quote(c.quote) for c in quoted)
    ambiguous = [c.candidate_id for c in agreeing if (c.pixel_provenance or {}).get("needs_review")]
    if ambiguous:
        notes.append(f"the digitizer marked {ambiguous} for review")


def vote(candidates: Sequence[Candidate], axis_range: float | None = None, *,
         group: GroupKey | None = None) -> VoteResult:
    """The vote for ONE group. Pass `group` when `candidates` covers more than one."""
    rows = _usable(candidates, group)
    groups = {c.group for c in rows}
    if group is None and len(groups) > 1:
        raise ValueError(f"vote() decides one group at a time; got {sorted(g or '?' for g in groups)}"
                         f" — pass group=... or use vote_groups()")
    decided = group or (next(iter(groups)) if groups else None)
    notes: list[str] = []
    result = VoteResult(group=decided)
    if not rows:
        result.notes = ["no candidate reported a value for this cell"]
        return result

    tolerances = [candidate_tolerance(c, axis_range) for c in rows]
    tolerance = max(tolerances)
    any_figure = any(is_figure_route(route_key(c)) for c in rows)
    if any_figure and all(figure_tolerance(c, axis_range) is None
                          for c in rows if is_figure_route(route_key(c))):
        notes.append(f"no figure calibration was recorded; falling back to "
                     f"{FALLBACK_FRACTION:.0%} of the value as the tolerance")

    routes = _route_values(rows, tolerance, notes)
    result.routes = routes
    result.tolerance = tolerance
    by_id = {c.candidate_id: c for c in rows}

    if len(routes) == 1:
        route = routes[0]
        result.agreement = "single"
        result.method = "single"
        result.mean = route.value
        result.agreeing_ids = list(route.candidate_ids)
        result.mad = 0.0
        _fill_values(result, [by_id[cid] for cid in route.candidate_ids], notes)
        result.notes = notes
        return result

    result.method = "figure_tolerance" if any_figure else "printed_precision"
    cluster = _best_cluster(routes, tolerance)
    cluster_keys = {r.route_key for r in cluster}
    text_routes = {r.route_key for r in routes if is_text_route(r.route_key)}

    if len(cluster) >= 2:
        values = [r.value for r in cluster]
        centre = _median(values)
        result.agreement = "agree"
        result.mean = centre
        result.mad = _median([abs(v - centre) for v in values])
        result.agreeing_ids = [c.candidate_id for c in rows if route_key(c) in cluster_keys]
        result.disagreeing_ids = [c.candidate_id for c in rows if route_key(c) not in cluster_keys]
        _fill_values(result, [by_id[cid] for cid in result.agreeing_ids], notes)
    else:
        result.agreement = "disagree"
        result.disagreeing_ids = [c.candidate_id for c in rows]
        result.needs_third_candidate = len(text_routes) == 2
        notes.append("no two independent routes agreed: "
                     + ", ".join(f"{r.route_key}={r.value:.4g}" for r in routes)
                     + f" (tolerance {tolerance:.4g})")
    result.notes = notes
    return result


def vote_groups(candidates: Sequence[Candidate],
                axis_range: float | None = None) -> dict[str, VoteResult]:
    """One `VoteResult` per group present in `candidates` (the shape Task 10 iterates)."""
    keys = sorted({c.group for c in candidates if c.group in ("A", "B")})
    return {key: vote(candidates, axis_range, group=key) for key in keys}
