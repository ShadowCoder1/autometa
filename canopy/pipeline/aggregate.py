"""Amendment A's `one_row_per_paper`: a paper's rows are COMBINED, never just picked from.

A paper that reports two contrasts for the same outcome cannot contribute both to a
random-effects pool: the rows are not independent and the pool would count that paper twice.
Choosing the "best" row instead throws away evidence and lets the choice rule move the result. So
the primary analysis combines them, by the rule the design justifies (Borenstein et al.,
*Introduction to Meta-Analysis*, ch. 24 — "Multiple outcomes or time-points within a study"):

* **Independent samples** (the paper ran the contrast on different participants and the rows do
  not share an arm) — a fixed-effect combination: weights 1/vᵢ, variance 1/Σw. This is
  `canopy.stats.meta.fixed_effects`; nothing here does that arithmetic itself.
* **Dependent rows** (the same participants across conditions, experiments or time points) — the
  composite is the MEAN of the rows, and its variance carries the correlation between them:

      V = (1/m²) · ( Σᵢ Vᵢ + Σᵢ≠ⱼ r·√(Vᵢ·Vⱼ) )

  with `r = settings.within_paper_r` (default 0.5). Treating dependent rows as independent would
  understate that variance, which is the one direction of error a meta-analysis must not take.
* **Shared-control designs** are NOT aggregated here: `resolve.apply_shared_control` has already
  applied the protocol's `shared_control_strategy` (Cochrane 16.5.4's n/k split by default), and
  splitting the shared arm is the accepted alternative to combining. Those rows stay as they are,
  with a note saying so.

Whether a paper's rows are independent is not something this module may guess. It reads
`EffectSizeRecord.sample_id`, which the orchestrator stamps from the mapper's own dataset
description (the paper's experiment label, and only when that dataset is a first exposure). An
empty `sample_id` means "we cannot claim this row came from its own sample", and rows that cannot
claim separate samples are treated as dependent — the conservative direction, and a recorded one:
the composite's `conversion_chain` names its members and the rule that combined them.

Nothing is dropped silently. Every row a composite replaced is returned in `superseded`, with an
`exclusions.csv` entry (`aggregated_into:<composite id>`), and the extraction table keeps it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models import EffectSizeRecord, StatsSettings
from ..verify.confidence import unverified_variance_bucket
from ..stats.effect_sizes import ci_smd
from ..stats.meta import fixed_effects

__all__ = ["Aggregation", "aggregate_one_row_per_paper", "composite_row", "composite_variance",
           "cluster_of", "AGGREGATED_FLAG", "DEPENDENT_FLAG", "INDEPENDENT_FLAG"]

AGGREGATED_FLAG = "aggregated_within_paper"
DEPENDENT_FLAG = "dependent_composite"
INDEPENDENT_FLAG = "independent_fixed_effect"
#: `apply_shared_control` marks the rows it adjusted with one of these; they are not aggregated
SHARED_CONTROL_PREFIX = "shared_control"


@dataclass
class Aggregation:
    """What `one_row_per_paper` did: the rows to pool, and everything it replaced."""

    rows: list[EffectSizeRecord] = field(default_factory=list)
    superseded: list[EffectSizeRecord] = field(default_factory=list)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def cluster_of(record: EffectSizeRecord) -> str:
    """The paper a row belongs to — one composite per cluster per outcome."""
    return record.cluster_id or record.paper_id or record.dataset_id


def _poolable(record: EffectSizeRecord) -> bool:
    return (record.es is not None and math.isfinite(record.es)
            and record.var is not None and record.var > 0 and math.isfinite(record.var))


def composite_variance(vi: Sequence[float], r: float) -> float:
    """Borenstein's variance of the MEAN of `m` dependent effects (ch. 24).

    `V = (1/m²)(Σ Vᵢ + Σ_{i≠j} r √(Vᵢ Vⱼ))`. With `r = 0` this is the variance of the mean of
    independent effects; with `r = 1` it is the variance of a single one of them.
    """
    values = [float(v) for v in vi]
    m = len(values)
    if m == 0:
        raise ValueError("composite_variance needs at least one variance")
    if any(v <= 0 for v in values):
        raise ValueError("every variance in a composite must be positive")
    cross = sum(r * math.sqrt(values[i] * values[j])
                for i in range(m) for j in range(m) if i != j)
    return (sum(values) + cross) / (m * m)


def _shared_control(record: EffectSizeRecord) -> bool:
    return any(str(f).startswith(SHARED_CONTROL_PREFIX) for f in record.flags)


def _independent(members: Sequence[EffectSizeRecord]) -> bool:
    """True only when every member names its OWN participant sample, and they all differ."""
    keys = [str(m.sample_id or "").strip() for m in members]
    return all(keys) and len(set(keys)) == len(keys)


def _common(values: Sequence[Any], default: Any = "") -> Any:
    unique = {v for v in values}
    return next(iter(unique)) if len(unique) == 1 else default


def _worst_confidence(members: Sequence[EffectSizeRecord], settings: StatsSettings) -> str:
    """The least confident member's bucket — a composite is no better than its weakest row."""
    order = list(settings.primary_analysis_includes)
    ranked = sorted(members, key=lambda r: order.index(r.confidence) if r.confidence in order
                    else len(order))
    return str(ranked[-1].confidence)


def composite_row(members: Sequence[EffectSizeRecord], settings: StatsSettings, *,
                  dependent: bool) -> EffectSizeRecord:
    """One row standing for a paper's several rows, with the variance its design implies."""
    if len(members) < 2:
        raise ValueError("a composite needs at least two rows")
    yi = [float(m.es) for m in members]
    vi = [float(m.var) for m in members]
    ids = [m.dataset_id for m in members]
    r = float(settings.within_paper_r)

    if dependent:
        estimate = sum(yi) / len(yi)
        var = composite_variance(vi, r)
        rule = (f"dependent rows (same participants): composite mean of {len(members)} rows, "
                f"variance (1/m²)(ΣVᵢ + Σᵢ≠ⱼ r√(VᵢVⱼ)) with r = {r:g}")
    else:
        fixed = fixed_effects(yi, vi, level=float(settings.ci_level))
        estimate, var = float(fixed.estimate), float(fixed.se) ** 2
        rule = (f"independent samples within one paper: fixed-effect combination of "
                f"{len(members)} rows (weights 1/vᵢ, variance 1/Σw)")

    se = math.sqrt(var)
    ci_low, ci_high = ci_smd(estimate, se, level=float(settings.ci_level), dist="z")

    digitised = [float(m.var_with_digitization or m.var) for m in members]
    if dependent:
        var_digit = composite_variance(digitised, r)
    else:
        var_digit = float(fixed_effects(yi, digitised, level=float(settings.ci_level)).se) ** 2

    first = members[0]
    moderators = {name: _common([m.moderators.get(name, "") for m in members], "mixed")
                  for name in first.moderators}
    routes = sorted({m.route for m in members if m.route})
    n_a = [m.n_a for m in members if m.n_a]
    n_b = [m.n_b for m in members if m.n_b]
    total = (sum if not dependent else max)

    record = EffectSizeRecord(
        paper_id=first.paper_id, cluster_id=cluster_of(first),
        dataset_id="+".join(ids), outcome_key=first.outcome_key,
        label=f"{first.label or first.dataset_id} (+{len(members) - 1} more)",
        route=routes[0] if len(routes) == 1 else "composite",
        n_a=total(n_a) if n_a else None, n_b=total(n_b) if n_b else None,
        es=estimate, var=var, se=se, ci_low=ci_low, ci_high=ci_high,
        estimator=first.estimator, variance_method=first.variance_method,
        level=float(settings.ci_level),
        higher_is_better=_common([m.higher_is_better for m in members], None),
        orientation_applied=all(m.orientation_applied for m in members),
        conversion_chain=f"{rule}; members: " + ", ".join(
            f"{m.dataset_id} ({m.es:+.4g}, v = {m.var:.4g})" for m in members),
        conversion_steps=[rule, *(f"{m.dataset_id}: {m.conversion_chain}" for m in members
                                  if m.conversion_chain)],
        routes_available=sorted({route for m in members for route in m.routes_available}),
        inputs={f"es[{m.dataset_id}]": m.es for m in members}
               | {f"var[{m.dataset_id}]": m.var for m in members},
        var_with_digitization=var_digit if var_digit > var else None,
        digitization_var=(var_digit - var) if var_digit > var else None,
        digitization_var_share=((var_digit - var) / var_digit) if var_digit > var else None,
        confidence=_worst_confidence(members, settings),
        analysis_metric=_common([m.analysis_metric for m in members], "unknown"),
        flags=sorted({f for m in members for f in m.flags}
                     | {AGGREGATED_FLAG, DEPENDENT_FLAG if dependent else INDEPENDENT_FLAG}),
        #: one entry per MEMBER ARM, never merged. The line above unions the members' codes, and a
        #: rule about one number read off that union is satisfied by two members that each pass on
        #: their own — which is exactly the shape `unverified_variance_bucket` screens for.
        arm_flags={f"{m.dataset_id}|{arm}": list(codes)
                   for m in members for arm, codes in m.arm_flags.items()},
        moderators=moderators, citation=first.citation, sample_id="",
        notes="; ".join(x for x in [*(m.notes for m in members), rule] if x)[:2000])

    # The row gates live in `resolve._finish`, and a composite never goes through it: this function
    # builds an `EffectSizeRecord` directly and it POOLS. So the one gate whose keys a union can
    # manufacture is re-derived here, on the attribution kept above (adversarial review, fix round).
    #
    # Its two neighbours are not re-derived, and deliberately: `conversion_gate_bucket` is keyed on
    # `route`, and a composite's route is `"composite"` or a single shared member route — it can
    # only fire on a converted statistic, which a composite is not; C9's `|d|` screen is keyed on
    # `record.d`, which a composite has none of (its estimate is a weighted mean of its members'
    # effect sizes, each already screened). Neither can be satisfied by a union of members that
    # passed on their own, which is the property that made this one bind.
    weighed, said = unverified_variance_bucket(record.confidence, record.arm_flags)
    if said:
        record.confidence = weighed
        record.conversion_steps = [*record.conversion_steps, *said]
        record.conversion_chain = "; ".join([record.conversion_chain, *said])
    return record


def aggregate_one_row_per_paper(rows: Sequence[EffectSizeRecord],
                                settings: StatsSettings) -> Aggregation:
    """Combine each paper's rows for ONE outcome into a single row (amendment A).

    Call with the rows of a single outcome that already passed the confidence filter. Rows from
    papers that contributed only one row come back untouched; the rest are replaced by a
    composite, and every replaced row is listed in `superseded` with its `exclusions.csv` entry.
    """
    out = Aggregation()
    clusters: dict[str, list[EffectSizeRecord]] = {}
    for record in rows:
        clusters.setdefault(cluster_of(record), []).append(record)

    for cluster, members in clusters.items():
        if len(members) == 1:
            out.rows.extend(members)
            continue
        if any(_shared_control(m) for m in members):
            out.rows.extend(members)
            out.notes.append(
                f"{cluster}: {len(members)} rows share a control arm — the protocol's "
                f"shared_control_strategy ({settings.shared_control_strategy}) already adjusted "
                f"them, so they are pooled as they are rather than aggregated")
            continue

        usable = [m for m in members if _poolable(m)]
        unusable = [m for m in members if not _poolable(m)]
        for record in unusable:
            out.superseded.append(record)
            out.exclusions.append(_exclusion(record, "superseded_by_dataset_rule",
                                             "one_row_per_paper: this paper contributes one row "
                                             "per outcome and this row carries no usable effect "
                                             "size to combine"))
        if len(usable) == 1:
            out.rows.extend(usable)
            continue
        if not usable:
            out.notes.append(f"{cluster}: no row could be combined for this outcome")
            continue

        dependent = not _independent(usable)
        composite = composite_row(usable, settings, dependent=dependent)
        out.rows.append(composite)
        for record in usable:
            out.superseded.append(record)
            out.exclusions.append(_exclusion(
                record, f"aggregated_into:{composite.dataset_id}", composite.conversion_chain))
        how = (f"dependent, r = {settings.within_paper_r:g}" if dependent
               else "independent samples, fixed-effect")
        out.notes.append(f"{cluster}: {len(usable)} rows combined into "
                         f"{composite.dataset_id} ({how})")
    return out


def _exclusion(record: EffectSizeRecord, reason: str, detail: str) -> dict[str, Any]:
    return {"paper_id": record.paper_id, "filename": "", "dataset_id": record.dataset_id,
            "outcome_key": record.outcome_key, "stage": "resolve", "reason": reason,
            "quote": "", "decider": "code", "detail": detail}
