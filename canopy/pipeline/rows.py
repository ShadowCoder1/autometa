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

from dataclasses import dataclass
from typing import Callable, Collection, Mapping, Sequence

from ..models import Candidate, DatasetSpec, StatsSettings, Verdict
from ..verify.checks import best_statistic
from .resolve import (ReportedValues, ResolvedValues, StatisticValues, apply_shared_control,
                      multi_group_flags)

__all__ = ["PreparedRow", "ENSEMBLE", "DISPERSION_APPROXIMATED", "cell_candidates",
           "statistic_values", "reported_values", "approximation_flags", "prepare_row_values",
           "prepare_rows", "shared_control_siblings", "converted_route",
           "converting_candidate", "reported_candidate"]

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

    @property
    def key(self) -> tuple[str, str]:
        return (self.dataset.dataset_id, self.outcome_key)


def cell_candidates(candidates: Sequence[Candidate], dataset_id: str,
                    outcome_key: str) -> list[Candidate]:
    return [c for c in candidates
            if c.dataset_id == dataset_id and c.outcome_key == outcome_key]


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
                                  higher_is_better=forced.get((dataset.dataset_id, key))))
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
