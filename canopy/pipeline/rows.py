"""The preparation every row goes through before `resolve_effect` — written once.

Between "the two cells have verdicts" and "the resolver may have them" there is a step that is
not arithmetic on either cell: the printed statistic and the printed effect size the row may
convert from, the flags the ROW carries about its dataset (a paper that reported more than two
groups, a dispersion the digitiser built rather than read), and the Cochrane 16.5.4 adjustment
for comparisons that share one control arm. None of it is visible from a single `Verdict`, and
all of it changes the number that reaches the forest plot.

It lived inside `run._resolve` and nowhere else, so the review layer's re-pool — which rebuilds a
row through `resolve_effect` after every human answer — rebuilt it from the two verdicts ALONE.
The whole-diff review measured what that costs: a shared control silently un-splits (a row's
variance falls 26% and its weight rises the moment a reviewer clicks "yes, this value is right"),
the statistic a row converts from disappears (so the row becomes `not_convertible` and no printed
t, F, p or d can reach the plot by any human path), and the row's own policy flags are dropped.

So the preparation is a function, in a module of its own, and both callers call it:
`run._resolve` for a whole paper, `overrides._rebuild_row` for one row plus the siblings it
shares a control with. The rule is the one the review named: **the re-pool of an unanswered row
must produce the byte-identical record the run produced.** `tests/test_run_records.py` asserts
exactly that.

Nothing here calls a model, and nothing here decides a bucket: it assembles inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Collection, Mapping, Sequence

from ..models import Candidate, DatasetSpec, StatsSettings, Verdict
from ..verify.checks import best_statistic
from ..verify.units import same_unit
from ..verify.vote import LOCATOR_DROPPED, locator_key, modality
from .resolve import (GROUP_ROUTES, GroupValues, ReportedValues, ResolvedValues,
                      StatisticValues, apply_shared_control, available_routes, multi_group_flags)

__all__ = ["PreparedRow", "ENSEMBLE", "DISPERSION_APPROXIMATED", "cell_candidates",
           "statistic_values", "reported_values", "approximation_flags", "prepare_row_values",
           "prepare_rows", "shared_control_siblings", "converted_route",
           "converting_candidate", "reported_candidate", "vote_candidates", "fallback_values"]

ENSEMBLE = "digitize:ensemble"

#: on the ROW, so a sensitivity analysis can pool with and without the rows whose dispersion the
#: code built rather than the paper stated (task 16 P6: the critique's amendment to the statistic)
DISPERSION_APPROXIMATED = "dispersion_approximated"


def converted_route(route: str) -> bool:
    """Did this row get an effect size from something other than the two cells' own numbers?

    One test, in one place, because both the review layer's bucket rules and the questions page
    ask it and a second copy is a second thing to keep in step. It is the RESOLVER's own answer —
    a route it recorded — rather than a list of route names held here, which the resolver could
    grow past without anyone noticing (whole-diff H2).
    """
    return bool(route) and route != "not_convertible"


@dataclass
class PreparedRow:
    """One (dataset × outcome) with everything `resolve_effect` needs, and nothing decided yet."""

    dataset: DatasetSpec
    outcome_key: str
    values: ResolvedValues
    #: D1: the same-locator candidate pairs this row could be built from INSTEAD, if the values
    #: above turn out to convert to nothing (`resolve.resolve_effect_with_fallback`). Prepared
    #: here rather than at either call site, because both call sites must offer the resolver the
    #: same alternatives or the re-pool of an unanswered row would not be the row the run built.
    alternatives: list[ResolvedValues] = field(default_factory=list)
    #: C2: HOW the direction this row is signed with was settled — `agreed`, `single_witness`,
    #: `tiebreak_ballot` or `human`, copied off the two cells. It is prepared here for the same
    #: reason everything else here is: both call sites stamp it onto the record from this, so a
    #: re-pool cannot quietly turn "a bought ballot outvoted a reader" into "two readers agreed".
    orientation_source: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.dataset.dataset_id, self.outcome_key)


def cell_candidates(candidates: Sequence[Candidate], dataset_id: str,
                    outcome_key: str) -> list[Candidate]:
    return [c for c in candidates
            if c.dataset_id == dataset_id and c.outcome_key == outcome_key]


def vote_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """The candidates the verification layer may see: one figure reading per group, not five.

    `digitize()` returns a `Candidate` per (group, route sample) *and* one ensemble candidate per
    group. The route samples belong in the stage file and the provenance bundle — that is where a
    reviewer checks how the picture was measured — but they must not enter the vote: the
    digitiser's four or five ways of measuring one figure would otherwise outvote the value the
    paper printed, and the ensemble (amendment F's median-of-routes, with the per-route detail in
    its `pixel_provenance`) is already their consensus. Controller ruling, fix round 1.

    It lives here, beside `fallback_values`, because D1's precedence override has to offer the
    resolver the same readings the vote weighed. Offered the raw samples instead, "the first
    alternative that converts" would be whichever measurement path the digitiser happened to list
    first — a number no reviewer ever saw and no verifier ever read.
    """
    return [c for c in candidates
            if not c.extractor_id.startswith("digitize:") or c.extractor_id == ENSEMBLE]


def approximation_flags(cell: Sequence[Candidate]) -> list[str]:
    """Row flags for anything in this cell whose dispersion is an approximation, not a reading."""
    kinds = {str((c.pixel_provenance or {}).get("dispersion_approximation") or "")
             for c in cell if c.extractor_id == ENSEMBLE}
    named = sorted(k for k in kinds if k)
    return [DISPERSION_APPROXIMATED, *[f"{DISPERSION_APPROXIMATED}:{k}" for k in named]] \
        if named else []


def statistic_values(candidates: Sequence[Candidate]) -> StatisticValues | None:
    """The best statistic the paper printed for this contrast — t/F first, then p."""
    # the same selection `canopy.verify.checks` screens, from the same function: the cell-level
    # gate and the row must agree about WHICH statistic this row is built from, or the gate holds
    # a cell for the degrees of freedom of a number nothing converts (review L4).
    cand = best_statistic(candidates)
    if cand is None:
        return None
    # P-B / C5: WHAT the statistic contrasts and WHAT it was averaged over. Dropping them here
    # left every statistic reaching the resolver with `contrast_kind="unknown"`, which the gate
    # refuses — fail-closed, so never a wrong number, but no printed statistic could ever rescue
    # a cell whose groups the paper does not print.
    return StatisticValues(
        stat_type=cand.stat_type or "unknown", value=cand.stat_value, df=cand.df,
        df1=cand.df1, df2=cand.df2, tails=cand.tails, p_kind=cand.p_kind or "unknown",
        p_value=cand.p_value, design=cand.design, direction=cand.direction,
        contrast_kind=cand.contrast_kind or "unknown",
        within_factors=list(cand.within_factors),
        outcome_averages_over=list(cand.outcome_averages_over))


def reported_candidate(candidates: Sequence[Candidate]) -> Candidate | None:
    """The printed effect size `reported_values` would build the row from, if any."""
    for cand in candidates:
        if cand.kind == "reported_d" and cand.status == "found" and cand.reported_value is not None:
            return cand
    return None


def converting_candidate(candidates: Sequence[Candidate], route: str = "") -> Candidate | None:
    """The ONE candidate this row's effect size was converted FROM.

    The same selection the resolver made, from the same functions — `checks.best_statistic` for a
    statistic route and `reported_candidate` for a printed d. The review layer has to name the
    number the row actually used: a paper often prints several statistics near an outcome, and
    "the first one in the stage file" is not the one `statistic_values` chose. It picked an
    INADMISSIBLE `F(1,22)` listed above the `t(22)` the row converted from, so the question asked
    the reviewer to accept a statistic that had nothing to do with the effect size beside it
    (whole-diff re-review N1).
    """
    if route == "reported_d":
        return reported_candidate(candidates)
    return best_statistic(candidates) or reported_candidate(candidates)


def reported_values(candidates: Sequence[Candidate]) -> ReportedValues | None:
    cand = reported_candidate(candidates)
    if cand is None:
        return None
    # P-B: a printed d has no test statistic behind it but it always has a CONTRAST.
    # "the aftereffect differed from zero, d = 1.30" is a one-sample effect, and it pooled
    # as the between-group difference because this field was extracted, carried on the
    # candidate, and dropped here. Without it every reported d reaches the resolver as
    # `contrast_kind="unknown"`, which the gate refuses — fail-closed, so never a wrong
    # number, but no printed effect size could ever fill a cell.
    return ReportedValues(value=cand.reported_value, scale=cand.reported_scale,
                          standardizer=cand.standardizer, ci_low=cand.reported_ci_low,
                          ci_high=cand.reported_ci_high,
                          positive_means=cand.positive_means,
                          contrast_kind=cand.contrast_kind or "unknown")


def prepare_row_values(dataset: DatasetSpec, outcome_key: str, verdict_a: Verdict,
                       verdict_b: Verdict, candidates: Sequence[Candidate],
                       settings: StatsSettings, *,
                       higher_is_better: bool | None = None) -> ResolvedValues:
    """One row's inputs: the two cells, what the paper printed, and the row's own policy flags.

    The shared-control adjustment is NOT here — it needs every row of the cluster at once, so it
    belongs to `prepare_rows` below.
    """
    cell = cell_candidates(candidates, dataset.dataset_id, outcome_key)
    values = ResolvedValues.from_verdicts(
        verdict_a, verdict_b, test_statistic=statistic_values(cell),
        reported=reported_values(cell), higher_is_better=higher_is_better)
    values.flags = sorted(set(values.flags)
                          | set(multi_group_flags(dataset, settings.multi_group_policy))
                          | set(approximation_flags(cell)))
    return values


def _some_spread(values: GroupValues) -> bool:
    """Did this reading bring a spread of ANY kind? Whether it converts is the resolver's call."""
    return (len(values.points) >= 2 or values.dispersion_value is not None
            or (values.ci_low is not None and values.ci_high is not None))


def _normalised(locator: str) -> str:
    return " ".join((locator or "").split()).casefold()


def _same_place(cand_a: Candidate, cand_b: Candidate) -> bool:
    """Were these two readings taken at the same place? The locator KEY, or the locator text.

    `vote.locator_key` is the hash of the normalised locator, so for a figure reading a shared key
    already IS a shared place. For every other modality it is `""` by D2's design — a sentence
    that names no place is not a second place — which would leave "the same (modality, key)" true
    of group A read from one sentence and group B from another. The text is compared directly
    there, so a pair is never assembled out of two different paragraphs.
    """
    if locator_key(cand_a):
        return True
    return _normalised(cand_a.locator) == _normalised(cand_b.locator)


def _reading(cand: Candidate, settled: GroupValues | None) -> GroupValues | None:
    """One candidate as one group's numbers, or `None` when it is not a whole set of them."""
    values = GroupValues.from_candidate(cand)
    # the group size is a fact about the ANALYSIS, not about where the number was read, and the
    # row's n may already have been divided between the comparisons that share a control arm
    # (Cochrane 16.5.4). So the size the cell settled on wins whenever it has one: taking the
    # candidate's would quietly un-split a shared control on exactly the rows this fallback is for.
    if settled is not None and settled.n:
        values.n = settled.n
    if (values.n or 0) < 2 or not _some_spread(values):
        return None
    if values.mean is None and values.median is None and len(values.points) < 2:
        return None
    return values


def _first_reading(cands: Sequence[Candidate], settled: GroupValues | None
                   ) -> tuple[Candidate, GroupValues] | None:
    """The first of one group's candidates that IS a whole set of numbers, with the candidate.

    Not simply the first candidate: `_reading` refuses one that brought no spread or no n, and
    taking the first and stopping would discard the whole pair when a later reading of the same
    route and place is complete. The candidate travels back with the values because the unit and
    the locator are checked between the two readings the pair was actually built from.
    """
    for cand in cands:
        values = _reading(cand, settled)
        if values is not None:
            return cand, values
    return None


def fallback_values(cell: Sequence[Candidate], primary: ResolvedValues,
                    settings: StatsSettings) -> list[ResolvedValues]:
    """The candidate PAIRS this row could be built from instead — D1's alternatives, in order.

    A pair, never two readings: one route, one place, both groups. The rules are the vote's own,
    for the vote's own reasons.

    * **One reading per group per route** (`vote_candidates`), so the digitiser's five measurement
      paths are one alternative and not five; and per group the FIRST of them that is a whole set
      of numbers, so an incomplete first candidate does not discard a complete second one.
    * **Nothing the panel check set aside** (`vote.LOCATOR_DROPPED`), which is where the vote
      itself drops them (`verify.vote._usable`): a reading whose panel the caption gives to the
      other group is that group's number, not a second reading of this one (D2).
    * **The same locator** (`vote.locator_key`, and the locator text where a modality has no key),
      because two panels of one figure are two quantities: Langan's Fig. 1 plots the young adults
      in panel A and the older adults in panel B, and a "pair" spanning both is a difference
      between two different pictures.
    * **The same unit** (`verify.units.same_unit`), because a mean in degrees minus a mean in
      per-cent is not an effect size.
    * **Ordered by the protocol's `route_precedence`**, so that when more than one pair converts
      the row is built from the one the protocol prefers — the fallback changes WHICH value is
      used, never the order they are preferred in.

    Whether a pair converts is not decided here: `resolve_effect_with_fallback` finds out by
    resolving it, through the ordinary resolver and its ordinary gates. This function's answer is
    "these are the pairs that exist", which is why a reading whose dispersion type nobody recorded
    is still listed — it is a real pair, and the refusal it earns should come from one place.
    """
    readings: dict[tuple[str, str], dict[str, list[Candidate]]] = {}
    for cand in vote_candidates(cell):
        if cand.kind != "group_stats" or cand.status != "found" or cand.group not in ("A", "B"):
            continue
        if (cand.pixel_provenance or {}).get(LOCATOR_DROPPED):
            continue        # verify.panels gave this panel to the other group (D2)
        readings.setdefault((modality(cand), locator_key(cand)), {}) \
                .setdefault(cand.group, []).append(cand)

    out: list[ResolvedValues] = []
    for pair in readings.values():
        read_a = _first_reading(pair.get("A", ()), primary.group_a)
        read_b = _first_reading(pair.get("B", ()), primary.group_b)
        if read_a is None or read_b is None:
            continue
        (cand_a, group_a), (cand_b, group_b) = read_a, read_b
        if not same_unit(cand_a.unit, cand_b.unit) or not _same_place(cand_a, cand_b):
            continue
        out.append(ResolvedValues(
            dataset_id=primary.dataset_id, outcome_key=primary.outcome_key,
            group_a=group_a, group_b=group_b, higher_is_better=primary.higher_is_better,
            route_available=list(primary.route_available), confidence=primary.confidence,
            flags=list(primary.flags), objections=dict(primary.objections)))

    order = list(settings.route_precedence)

    def rank(values: ResolvedValues) -> int:
        routes, _ = available_routes(values)
        name = next((route for route in routes if route in GROUP_ROUTES), "")
        return order.index(name) if name in order else len(order)

    return sorted(out, key=rank)               # stable: pairs of one rank keep the reading order


def _default_cluster(dataset: DatasetSpec) -> str:
    return dataset.cluster_id or dataset.dataset_id


def prepare_rows(cells: Sequence[tuple[DatasetSpec, str, Verdict, Verdict]],
                 candidates: Sequence[Candidate], settings: StatsSettings, *,
                 cluster_of: Callable[[DatasetSpec], str] = _default_cluster,
                 directions: Mapping[tuple[str, str], bool] | None = None,
                 ) -> list[PreparedRow]:
    """Every row of one paper (or one shared-control cluster), ready for `resolve_effect`.

    `cells` is `(dataset, outcome_key, verdict_A, verdict_B)` in the order the map lists them —
    the order matters, because `apply_shared_control`'s `keep_first` and `combine_arms`
    strategies are defined by it and `split_n` divides by how many rows are in the group.

    `directions` forces `higher_is_better` on named rows (the review layer's `orientation`
    answer, which is a decision about the measure rather than about either cell).
    """
    forced = dict(directions or {})
    prepared = [PreparedRow(
        dataset=dataset, outcome_key=key,
        values=prepare_row_values(dataset, key, verdict_a, verdict_b, candidates, settings,
                                  higher_is_better=forced.get((dataset.dataset_id, key))),
        # either cell's, because both were signed by the one verdict for the measure; A's first
        # only so that the answer is deterministic when one cell was never verified
        orientation_source=(verdict_a.orientation_source or verdict_b.orientation_source))
        for dataset, key, verdict_a, verdict_b in cells]

    # rows in one paper that share a control arm are not independent (Cochrane 16.5.4)
    shared: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(prepared):
        if row.dataset.shared_control:
            shared.setdefault((cluster_of(row.dataset), row.outcome_key), []).append(index)
    for indices in shared.values():
        if len(indices) < 2:
            continue
        adjusted = apply_shared_control([prepared[i].values for i in indices],
                                        settings.shared_control_strategy)
        for position, index in enumerate(indices):
            if position < len(adjusted):
                prepared[index].values = adjusted[position]

    # LAST, after the shared-control adjustment: an alternative takes the row's group sizes from
    # the values above, and those are the split ones (Cochrane 16.5.4). Built before this loop,
    # every fallback row would carry a control arm's full n.
    for row in prepared:
        row.alternatives = fallback_values(
            cell_candidates(candidates, row.dataset.dataset_id, row.outcome_key),
            row.values, settings)
    return prepared


def shared_control_siblings(dataset: DatasetSpec, outcome_key: str,
                            datasets: Sequence[DatasetSpec],
                            cluster_of: Callable[[DatasetSpec], str],
                            has_cell: Callable[[str, str], bool],
                            keys: Collection[str] | None = None,
                            ) -> list[tuple[DatasetSpec, str]]:
    """`(dataset, outcome_key)` for every row this one shares a control arm with, itself included.

    A row that shares no control is its own only sibling, so a caller never has to branch. The
    membership test is the run's: the same cluster, the same outcome, `shared_control` set, and
    both cells verified — otherwise the re-pool would split a control `k` ways where the run
    split it `k − 1` ways, and change a row nobody answered.
    """
    if not dataset.shared_control:
        return [(dataset, outcome_key)]
    cluster = cluster_of(dataset)
    out: list[tuple[DatasetSpec, str]] = []
    seen: set[str] = set()
    for other in datasets:
        if not other.shared_control or cluster_of(other) != cluster:
            continue
        if other.dataset_id in seen:
            continue
        for sources in other.outcomes:
            if sources.outcome_key != outcome_key:
                continue
            if keys is not None and sources.outcome_key not in keys:
                continue
            if has_cell(other.dataset_id, outcome_key):
                seen.add(other.dataset_id)
                out.append((other, outcome_key))
            break
    return out or [(dataset, outcome_key)]
