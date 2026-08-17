"""Agreement vote (spec §3.3(2), amendment G) — pure code, no model.

A number is believed when two readers that fail *differently* wrote it down. "Differently" is
`route_key` = modality (text / table / one digitizer path / statistic) × model family: two prompt
variants of the same model on the same pages share a failure mode, so they are one voter, and
their answers are collapsed to that route's median before anything is compared.

Tolerance comes from the evidence, not from a constant, and it is applied PER PAIR — never once
for the whole cell:

* two printed values — the COARSER of the two printed precisions. A paper that prints 31.5 in one
  place and 31.51 in another is printing one number at two precisions, so ±0.05 (half a unit in
  the last digit of the LESS precise reading) is what separates a rounding from a discrepancy:
  31.5 and 31.51 agree, 31.5 and 31.6 do not, and 31.5 and 33.0 do not at any precision.
* anything against a picture — the figure tolerance, max(2% of the axis range, half a tick), the
  resolution a reader of that picture can honestly claim. A figure may only corroborate a printed
  value or be flagged against it; it never sets the resolved value and never overrules the print.

The comparison is therefore **staged**, and the stages are not interchangeable. Text and table
routes settle the value FIRST, among themselves (spec §3.3(2): text/table agree when equal after
normalisation, ± printed precision). **One printed route is already that consensus** — the
digitizer contributes four or five routes to a figure cell, and a printed value must not be
outvoted by however many ways one picture was measured. Only then are figure routes reconciled
with the printed consensus, at the figure tolerance. A figure can corroborate it, or conflict with
it (`figure_conflict`, a note, and the printed value stands); it can never replace it, and it can
never bridge two printed readers who disagree with each other.

The resolved value is a printed one whenever any printed route exists: the most-reported
text/table value and, on a tie, the most precisely printed of them (31.51 over 31.5) — never a
blend of a transcription and a measurement. Only a cell with no text or table route at all takes
the figure per-cell median.

A lone printed route that no figure corroborates keeps its value but is reported as `single` with
`method="printed_uncorroborated"`, so `confidence` caps it below automatic acceptance: exactly one
route stands behind it, and the pictures that could have confirmed it did not.

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
from .figures import (AXIS_FRACTION, FALLBACK_FRACTION, FIGURE_KINDS, TICK_FRACTION,
                      figure_tolerance)
from .grounding import is_short_quote as _is_short_quote
from .units import unit_key

__all__ = ["vote", "vote_groups", "VoteResult", "RouteValue", "route_key", "modality",
           "digitizer_path",
           "model_family", "precision_tolerance", "figure_tolerance", "candidate_tolerance",
           "AXIS_FRACTION", "TICK_FRACTION"]

MAX_DECIMALS = 12               # beyond this a "printed" precision is float noise

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


# ----------------------------------------------------------------------------- routes
def model_family(model: str) -> str:
    """`claude-opus-5-20260401` → `claude-opus`: the family, without version or date."""
    name = (model or "").split("/")[-1]
    parts = [p for p in name.split("-") if p]
    while parts and parts[-1].isdigit():
        parts.pop()
    return "-".join(parts) or name


def digitizer_path(extractor_id: str) -> str:
    """`digitize:readout:claude-opus-5:direct` → `figure:readout` — which way the picture was
    measured, without the model or the prompt variant. Two prompts of one path are one path."""
    pieces = (extractor_id or "").split(":")
    return f"figure:{pieces[1]}" if len(pieces) > 1 and pieces[1] else "figure"


def modality(cand: Candidate) -> str:
    """How this value was obtained — the half of a route that is not the model."""
    if cand.extractor_id.startswith("digitize:"):
        return digitizer_path(cand.extractor_id)
    if cand.source_kind in FIGURE_KINDS:
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
    tolerance: float = 0.0               # what THIS route can honestly claim about its own value
    spread: float | None = None          # furthest candidate from this route's own median
    consistent: bool = True
    #: the unit this route's members share, once routes are split by unit ("" = none stated)
    unit: str = ""
    #: a route whose members disagree with each other and corroborate nothing abstains: its
    #: `value` is None and it is kept here for the record, not counted in the vote
    abstained: bool = False


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
    #: a figure route read a different value from the text/table consensus that carried the vote —
    #: the text value is kept, and this says the picture did not corroborate it
    figure_conflict: bool = False
    method: str = "none"
    tolerance: float | None = None
    agreeing_ids: list[str] = Field(default_factory=list)
    disagreeing_ids: list[str] = Field(default_factory=list)
    needs_third_candidate: bool = False
    routes: list[RouteValue] = Field(default_factory=list)
    #: candidates whose unit is not the outcome's, set aside before the vote (Cressman's Fig. 3b
    #: read off its percentage axis when the outcome is in degrees) — the checks flag them
    unit_set_aside_ids: list[str] = Field(default_factory=list)
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


def _reported_value(routes: Sequence["RouteValue"]) -> float:
    """A value a source actually reported: the most-reported one, most precisely printed.

    Never the arithmetic middle of two different readings — an effect size built on a number no
    paper contains cannot be checked against the paper. When two readings agree within the coarser
    of their printed precisions, the finer one is the better record of what the paper prints
    (31.51, not 31.5), so ties are broken by tolerance and then by value, deterministically.
    """
    counts = Counter(r.value for r in routes)
    best = max(counts.values())
    tied = [r for r in routes if counts[r.value] == best]
    return float(min(tied, key=lambda r: (r.tolerance, r.value)).value)


def _route_tolerance(members: Sequence[Candidate], axis_range: float | None) -> float:
    """What one route can claim about its own value.

    A text route takes the coarsest printed precision among its members, for the same reason a
    pair of them does: a value printed at one decimal is not contradicted by the same value
    printed at two. A figure route's members share one calibration, so min and max agree.
    """
    return max(candidate_tolerance(c, axis_range) for c in members)


def _pair_tolerance(a: RouteValue, b: RouteValue) -> float:
    """How far apart two routes may be and still be the same value: the LOOSER of their claims.

    One expression, two readings of it, both deliberate:

    * **two printed readings** — the COARSER of the two precisions. "31.5" in the text and "31.51"
      in a table are one number printed at two precisions, and half a unit in the last digit of
      the less precise one is exactly what separates that from a discrepancy. 31.5 against 31.6,
      or 32 against 33.0, is a different number at any precision either of them claims.
    * **anything against a picture** — the figure tolerance, max(2% of the axis range, half a
      tick), because a value measured off an axis cannot be read more precisely than that axis.

    What keeps a picture from *governing* is not this number — a tolerance tight enough to stop
    that would also make corroboration impossible, since no digitised read lands within ±0.005 of
    a printed value. It is the staging in `vote()`: figures are reconciled with the printed
    consensus only after it has been decided among the printed sources, they never join in
    deciding it, and they never set the resolved value.
    """
    return max(a.tolerance, b.tolerance)


def _route_values(rows: Sequence[Candidate], axis_range: float | None,
                  notes: list[str]) -> list[RouteValue]:
    """One voter per modality × model family × UNIT, each voting only what its members corroborate.

    Two things a route may not do. It may not hold readings in different units — a bar read off
    a degrees axis and the same bar read off the percentage axis beside it are two quantities,
    and Cressman's Fig. 3b put 18.5° and 61.5% into one route whose "median" of the two, 39.99,
    was in no unit at all. And a route whose members disagree may not vote a middle that none of
    them read: with two members the median IS their average, a number nobody produced. The
    largest cluster of mutually-agreeing members votes; if no two members agree and there are
    more than one, the route abstains and is kept for the record.
    """
    grouped: dict[tuple[str, str], list[Candidate]] = {}
    for cand in rows:
        grouped.setdefault((route_key(cand), unit_key(cand.unit)), []).append(cand)
    units_per_key: dict[str, set[str]] = {}
    for key, unit in grouped:
        units_per_key.setdefault(key, set()).add(unit)
    routes: list[RouteValue] = []
    for key, unit in sorted(grouped):
        members = grouped[(key, unit)]
        label = key if len(units_per_key[key]) == 1 else f"{key}[{unit or 'no unit'}]"
        means = [c.mean for c in members]
        median = _median(means)
        spread = max(abs(m - median) for m in means)
        tolerance = _route_tolerance(members, axis_range)
        consistent = spread <= tolerance
        voters = members
        value: float | None = median
        if not consistent:
            # the median of an odd count is a member's own reading, and it stands if at least
            # one other member is within tolerance of it (an outlier does not move it); the
            # median of two far-apart members is their average, which nobody read
            at_median = [c for c in members if c.mean == median]
            partners = [c for c in members if abs(c.mean - median) <= tolerance]
            if at_median and len(partners) >= 2:
                voters = partners
                notes.append(f"route {label} disagrees with itself: {means}; its median "
                             f"{median:.4g} is a reading {len(partners)} of {len(members)} "
                             f"agree with, and it votes; the rest are set aside")
            else:
                cluster = _largest_cluster(members, tolerance)
                if len(cluster) >= 2:
                    voters = cluster
                    value = _median([c.mean for c in cluster])
                    notes.append(f"route {label} disagrees with itself: {means}; the "
                                 f"{len(cluster)} of {len(members)} that agree vote "
                                 f"{value:.4g}, the rest are set aside")
                else:
                    value = None
                    notes.append(f"route {label} disagrees with itself: {means} (spread "
                                 f"{spread:.4g} > tolerance {tolerance:.4g}) and no two of its "
                                 f"readings agree — it abstains; a middle none of them read is "
                                 f"not a reading")
        dispersions = [c.dispersion_value for c in voters if c.dispersion_value is not None]
        sigmas = [c.sigma for c in voters if c.sigma is not None]
        routes.append(RouteValue(
            route_key=label, value=value,
            dispersion_value=_median(dispersions) if dispersions else None,
            n=_mode([c.n for c in voters]), sigma=_median(sigmas) if sigmas else None,
            candidate_ids=[c.candidate_id for c in voters], tolerance=tolerance,
            spread=spread, consistent=consistent, unit=unit, abstained=value is None))
    return routes


def _largest_cluster(members: Sequence[Candidate], tolerance: float) -> list[Candidate]:
    """The biggest set of members that all sit within `tolerance` of one of them."""
    best: list[Candidate] = []
    for seed in members:
        cluster = [c for c in members if abs(c.mean - seed.mean) <= tolerance]
        if len(cluster) > len(best):
            best = cluster
    return best


def _best_cluster(routes: Sequence[RouteValue]) -> list[RouteValue]:
    """The largest set of routes that all sit within their PAIRWISE tolerance of one of them."""
    best: list[RouteValue] = []
    for seed in routes:
        cluster = [r for r in routes if abs(r.value - seed.value) <= _pair_tolerance(seed, r)]
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
         group: GroupKey | None = None, unit_hint: str = "") -> VoteResult:
    """The vote for ONE group. Pass `group` when `candidates` covers more than one.

    `unit_hint` is the unit the map/protocol recorded for this outcome. A candidate that names a
    DIFFERENT unit is a different expression of the quantity (a percentage axis beside a degrees
    axis) and is set aside before the vote — unless every candidate is in that other unit, in
    which case the hint is the odd one out and nothing is set aside.
    """
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

    hint = unit_key(unit_hint)
    if hint:
        other = [c for c in rows if unit_key(c.unit) and unit_key(c.unit) != hint]
        if other and len(other) < len(rows):
            result.unit_set_aside_ids = [c.candidate_id for c in other]
            rows = [c for c in rows if c not in other]
            notes.append(f"{len(other)} candidate(s) read in "
                         f"{sorted({unit_key(c.unit) for c in other})} were set aside: the "
                         f"outcome is recorded in {hint!r}, and a value in another unit is "
                         f"another expression of it, not a second reading")

    if any(is_figure_route(route_key(c)) for c in rows) and all(
            figure_tolerance(c, axis_range) is None
            for c in rows if is_figure_route(route_key(c))):
        notes.append(f"no figure calibration was recorded; falling back to "
                     f"{FALLBACK_FRACTION:.0%} of the value as the tolerance")

    routes = _route_values(rows, axis_range, notes)
    result.routes = routes
    by_id = {c.candidate_id: c for c in rows}
    routes = [r for r in routes if not r.abstained]        # kept on the record, not in the vote
    if not routes:
        result.agreement, result.method = "disagree", "none"
        result.disagreeing_ids = [c.candidate_id for c in rows]
        result.notes = notes + ["every route disagrees with itself; nothing corroborated votes"]
        return result

    if len(routes) == 1:
        route = routes[0]
        result.agreement, result.method = "single", "single"
        result.mean = route.value
        result.tolerance = route.tolerance
        result.agreeing_ids = list(route.candidate_ids)
        result.mad = 0.0
        _fill_values(result, [by_id[cid] for cid in route.candidate_ids], notes)
        result.notes = notes
        return result

    text = [r for r in routes if is_text_route(r.route_key)]
    others = [r for r in routes if not is_text_route(r.route_key)]

    # --- stage 1: what the printed sources say, decided among themselves. ONE printed route is
    # already a consensus: a value the paper prints is not outvoted by a picture, however many
    # ways that picture was measured.
    if text:
        consensus = _best_cluster(text) if len(text) >= 2 else list(text)
        if len(text) >= 2 and len(consensus) < 2:
            result.agreement, result.method = "disagree", "printed_precision"
            result.tolerance = max(_pair_tolerance(a, b) for a in text for b in text if a is not b)
            result.disagreeing_ids = [c.candidate_id for c in rows]
            result.needs_third_candidate = len(text) == 2
            notes.append("the printed sources disagree: "
                         + ", ".join(f"{r.route_key}={r.value:.4g}" for r in text)
                         + f" (tolerance {result.tolerance:.4g}); a figure read cannot decide "
                           f"between them")
            if others:
                notes.append("routes not consulted: "
                             + ", ".join(f"{r.route_key}={r.value:.4g}" for r in others))
            result.notes = notes
            return result

        value = _reported_value(consensus)
        conflicts = [r for r in text if r not in consensus]
        for route in conflicts:
            notes.append(f"printed route {route.route_key} read {route.value:.4g}, outside the "
                         f"{max(r.tolerance for r in consensus):.4g} tolerance around the value "
                         f"{value:.4g} the other printed sources agree on")
        precisions = {r.tolerance for r in consensus}
        if len(precisions) > 1:
            notes.append("the printed sources agree but at different precisions "
                         + ", ".join(f"{r.route_key}={r.value:.6g} (±{r.tolerance:g})"
                                     for r in consensus)
                         + f"; the most precise reading {value:.6g} is kept")

        # --- stage 2: does anything else corroborate what the paper prints?
        winners = list(consensus)
        reconciliation = 0.0
        for route in others:
            tolerance = max([route.tolerance] + [c.tolerance for c in consensus])
            reconciliation = max(reconciliation, tolerance)
            if abs(route.value - value) <= tolerance:
                winners.append(route)
            else:
                conflicts.append(route)
                notes.append(f"route {route.route_key} read {route.value:.4g}, outside the "
                             f"{tolerance:.4g} tolerance around the printed value {value:.4g}; "
                             f"the printed value stands")
        result.figure_conflict = any(is_figure_route(r.route_key) for r in conflicts)

        if len(consensus) >= 2:
            result.tolerance = max(_pair_tolerance(a, b) for a in consensus for b in consensus
                                   if a is not b)
            result.method = "printed_precision"
        else:
            result.tolerance = reconciliation or consensus[0].tolerance
            result.method = "figure_tolerance" if others else "single"

        if len(winners) >= 2:
            _decide(result, rows, winners, value, by_id, notes)
            return result

        # one printed route, and nothing corroborated it
        keys = {r.route_key for r in winners}
        result.agreement = "single"
        result.method = "printed_uncorroborated" if others else "single"
        result.mean = value
        result.mad = 0.0
        result.agreeing_ids = [c.candidate_id for c in rows if route_key(c) in keys]
        result.disagreeing_ids = [c.candidate_id for c in rows if route_key(c) not in keys]
        if others:
            notes.append("no other route corroborated the printed value; it stands on one "
                         "reader alone")
        _fill_values(result, [by_id[cid] for cid in result.agreeing_ids], notes)
        result.notes = notes
        return result

    # --- no printed source at all: the pictures vote among themselves
    cluster = _best_cluster(routes)
    result.method = "figure_tolerance" if any(is_figure_route(r.route_key) for r in routes) \
        else "printed_precision"
    if len(cluster) >= 2:
        result.tolerance = max(_pair_tolerance(a, b) for a in cluster for b in cluster if a is not b)
        _decide(result, rows, cluster, _median([r.value for r in cluster]), by_id, notes)
        return result

    result.agreement = "disagree"
    result.tolerance = max(r.tolerance for r in routes)
    result.disagreeing_ids = [c.candidate_id for c in rows]
    notes.append("no two independent routes agreed: "
                 + ", ".join(f"{r.route_key}={r.value:.4g}" for r in routes)
                 + f" (tolerances {[round(r.tolerance, 4) for r in routes]})")
    result.notes = notes
    return result


def _decide(result: VoteResult, rows: Sequence[Candidate], winners: Sequence[RouteValue],
            value: float, by_id: dict[str, Candidate], notes: list[str]) -> None:
    """Record an agreed cell: the value the sources reported, and who stood behind it."""
    keys = {r.route_key for r in winners}
    result.agreement = "agree"
    result.mean = value
    result.mad = _median([abs(r.value - value) for r in winners])
    result.agreeing_ids = [c.candidate_id for c in rows if route_key(c) in keys]
    result.disagreeing_ids = [c.candidate_id for c in rows if route_key(c) not in keys]
    _fill_values(result, [by_id[cid] for cid in result.agreeing_ids], notes)
    result.notes = notes


def vote_groups(candidates: Sequence[Candidate], axis_range: float | None = None,
                unit_hint: str = "") -> dict[str, VoteResult]:
    """One `VoteResult` per group present in `candidates` (the shape Task 10 iterates)."""
    keys = sorted({c.group for c in candidates if c.group in ("A", "B")})
    return {key: vote(candidates, axis_range, group=key, unit_hint=unit_hint) for key in keys}
