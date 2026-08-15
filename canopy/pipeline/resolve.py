"""Effect-size resolution (Task 9): resolved values in, one `EffectSizeRecord` out.

This is the boundary the whole pipeline exists to reach. Everything before it locates, transcribes
and verifies; here the numbers become a standardised mean difference — in code, from named
formulas in `canopy.stats`, with the arithmetic written out.

Three things make a record trustworthy rather than merely present:

* **The route is chosen, not stumbled into.** `settings.route_precedence` orders the ways a paper
  can yield an effect size (amendment A's default puts a printed mean ± SD first and a reported d
  last, because a value we convert ourselves can be checked and one the authors computed cannot).
  Routes that were passed over are recorded with the reason.
* **A conversion that cannot be justified is refused, not approximated.** A statistic from a design
  that carries no between-group contrast, a p reported only as "< .05", an effect size standardised
  within participants, a measure whose direction nobody could establish — each is recorded with
  `route="not_convertible"` and a reason, so it appears in the review queue instead of the forest.
* **The chain is human-readable and carries its numbers**, e.g.
  ``SD_A = SE 2.01 × √19 = 8.761; pooled SD = 7.505; d = (44.67 − 46.14)/7.505 = −0.1959``.

Digitised inputs additionally carry σ, which reaches the record through the delta method
(`canopy.stats.conversions.partial_variance`) under `settings.digitization_variance`.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Sequence

from pydantic import Field

from ..models import (CanopyModel, ConfidenceBucket, DatasetSpec, Direction, DispersionType,
                      EffectSizeRecord, GroupKey, OutcomeDef, PKind, ReportedScale, Standardizer,
                      StatsSettings, TestDesign, Verdict)
from ..stats import effect_sizes as es
from ..stats.conversions import (mean_sd_from_five_number, mean_sd_from_median_iqr,
                                 combine_groups, partial_variance, split_control)
from ..stats.effect_sizes import NotConvertible, SMDResult

__all__ = ["resolve_effect", "available_routes", "apply_shared_control", "multi_group_flags",
           "GroupValues", "StatisticValues", "ReportedValues", "ResolvedValues",
           "GROUP_ROUTES", "DIGITIZATION_SHARE_FLAG"]

#: the route names in `StatsSettings.route_precedence` that are built from two groups' statistics
GROUP_ROUTES: tuple[str, ...] = ("text_mean_sd", "table", "text_mean_se_ci", "figure")
#: digitisation variance above this share of the sampling variance is flagged (spec §3.4)
DIGITIZATION_SHARE_FLAG = 0.10
#: amendment A: `ci_to_sd_dist="auto"` means t(n−1) below this group size, z at or above it
CI_T_BELOW_N = 100

_TEXT_ROUTES = frozenset({"", "text", "adjudicated", "author_data", "unknown"})
_BETWEEN_SCALES = frozenset({"cohens_d", "hedges_g"})
_WITHIN_STANDARDIZERS = frozenset({"dz_paired", "partial_eta"})
_CI_LEVELS = {DispersionType.CI95: 0.95, DispersionType.CI90: 0.90}


# ----------------------------------------------------------------------------- inputs
class GroupValues(CanopyModel):
    """What verification resolved for ONE group — whatever form the paper printed it in."""

    n: int | None = None
    mean: float | None = None
    dispersion_value: float | None = None
    dispersion_type: DispersionType = DispersionType.UNKNOWN
    ci_level: float | None = None                 # defaults from the dispersion type
    ci_low: float | None = None
    ci_high: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    points: list[float] = Field(default_factory=list)
    sigma: float | None = None                    # digitisation uncertainty on the mean
    dispersion_sigma: float | None = None         # digitisation uncertainty on the spread
    unit: str = ""
    route: str = ""                               # text | table | figure | adjudicated | ...
    label: str = ""

    @classmethod
    def from_verdict(cls, verdict: Verdict) -> "GroupValues":
        """The verification layer's `Verdict` as resolution inputs.

        A `Verdict` records one interval per cell; what that interval *is* depends on the
        dispersion type the readers agreed on, so an IQR becomes quartiles around the median and a
        range becomes a minimum and a maximum.
        """
        values = cls(n=verdict.n, mean=verdict.mean, dispersion_value=verdict.dispersion_value,
                     dispersion_type=verdict.dispersion_type, points=list(verdict.points),
                     sigma=verdict.sigma, unit=verdict.unit, route=verdict.route)
        if verdict.dispersion_type is DispersionType.IQR:
            values.median, values.q1, values.q3 = verdict.mean, verdict.ci_low, verdict.ci_high
        elif verdict.dispersion_type is DispersionType.RANGE:
            values.minimum, values.maximum = verdict.ci_low, verdict.ci_high
        else:
            values.ci_low, values.ci_high = verdict.ci_low, verdict.ci_high
        return values


class StatisticValues(CanopyModel):
    """A statistic the paper printed for the A-vs-B contrast."""

    stat_type: Literal["t", "F", "p", "chi2", "unknown"] = "unknown"
    value: float | None = None
    df: float | None = None
    df1: float | None = None
    df2: float | None = None
    tails: int | None = None
    p_kind: PKind = "unknown"
    p_value: float | None = None
    design: TestDesign = "unknown"
    direction: Direction = "unknown"


class ReportedValues(CanopyModel):
    """An effect size the paper itself printed."""

    value: float | None = None
    scale: ReportedScale = "unknown"
    standardizer: Standardizer = "unknown"
    ci_low: float | None = None
    ci_high: float | None = None
    positive_means: Direction = "unknown"


_BUCKET_ORDER: dict[str, int] = {"auto_accept": 2, "accept_with_note": 1, "needs_human": 0}


class ResolvedValues(CanopyModel):
    """Everything Task 9 needs about one (dataset × outcome), after verification."""

    dataset_id: str = ""
    outcome_key: str = ""
    group_a: GroupValues | None = None
    group_b: GroupValues | None = None
    test_statistic: StatisticValues | None = None
    reported: ReportedValues | None = None
    higher_is_better: bool | None = None
    #: routes the orchestrator is willing to use for this cell; empty means "whatever the data
    #: supports" (it exists so a re-run can exclude, say, every figure-derived route)
    route_available: list[str] = Field(default_factory=list)
    confidence: ConfidenceBucket = "needs_human"
    flags: list[str] = Field(default_factory=list)

    def group(self, key: str) -> GroupValues | None:
        return self.group_a if key == "A" else self.group_b

    @classmethod
    def from_verdicts(cls, verdict_a: Verdict, verdict_b: Verdict, *,
                      test_statistic: StatisticValues | None = None,
                      reported: ReportedValues | None = None,
                      higher_is_better: bool | None = None) -> "ResolvedValues":
        """Two `Verdict`s into one set of inputs; the weaker confidence governs the pair."""
        buckets = sorted((verdict_a.confidence, verdict_b.confidence),
                         key=lambda b: _BUCKET_ORDER.get(b, 0))
        direction = higher_is_better
        if direction is None:
            direction = verdict_a.higher_is_better if verdict_a.higher_is_better is not None \
                else verdict_b.higher_is_better
        flags = sorted({flag.code for verdict in (verdict_a, verdict_b)
                        for flag in verdict.flags if flag.severity in ("warn", "error")})
        return cls(dataset_id=verdict_a.dataset_id or verdict_b.dataset_id,
                   outcome_key=verdict_a.outcome_key or verdict_b.outcome_key,
                   group_a=GroupValues.from_verdict(verdict_a),
                   group_b=GroupValues.from_verdict(verdict_b),
                   test_statistic=test_statistic, reported=reported,
                   higher_is_better=direction, confidence=buckets[0], flags=flags)


# ----------------------------------------------------------------------------- route availability
def _modality(route: str) -> str:
    name = (route or "").lower()
    if name.startswith(("figure", "digitize")):
        return "figure"
    if name == "table":
        return "table"
    return "text" if name in _TEXT_ROUTES else "text"


def _has_spread(group: GroupValues) -> bool:
    kind = group.dispersion_type
    if len(group.points) >= 2:
        return True
    if kind is DispersionType.SD or kind is DispersionType.SE:
        return group.dispersion_value is not None and group.dispersion_value > 0
    if kind in _CI_LEVELS:
        return ((group.ci_low is not None and group.ci_high is not None)
                or (group.dispersion_value is not None and group.dispersion_value > 0))
    if kind is DispersionType.IQR:
        return (group.q1 is not None and group.q3 is not None) or group.dispersion_value is not None
    if kind is DispersionType.RANGE:
        return group.minimum is not None and group.maximum is not None
    return False


def _group_ready(group: GroupValues | None) -> bool:
    if group is None:
        return False
    if len(group.points) >= 2:
        return True
    centre = group.mean if group.mean is not None else group.median
    return centre is not None and (group.n or 0) >= 2 and _has_spread(group)


def _group_route_name(values: ResolvedValues) -> str:
    a, b = values.group_a, values.group_b
    modalities = {_modality(a.route), _modality(b.route)}
    if "figure" in modalities:
        return "figure"
    if "table" in modalities:
        return "table"
    both_sd = all(g.dispersion_type is DispersionType.SD and not g.points for g in (a, b))
    return "text_mean_sd" if both_sd else "text_mean_se_ci"


def available_routes(values: ResolvedValues) -> tuple[list[str], dict[str, str]]:
    """`(routes the data supports, why each other route is not there)`.

    The two groups yield at most ONE route name — the one that describes where the numbers came
    from and what form they were in — so a cell never claims to support both `text_mean_sd` and
    `figure` at once.
    """
    routes: list[str] = []
    reasons: dict[str, str] = {}
    if _group_ready(values.group_a) and _group_ready(values.group_b):
        routes.append(_group_route_name(values))
    else:
        missing = [key for key in ("A", "B") if not _group_ready(values.group(key))]
        reasons["group_statistics"] = (f"group(s) {', '.join(missing)} have no mean, group size "
                                       f"and dispersion")

    stat = values.test_statistic
    if stat is not None and stat.stat_type in ("t", "F") and stat.value is not None:
        routes.append("test_statistic")
    else:
        reasons["test_statistic"] = "no t or F statistic was resolved for this contrast"
    if stat is not None and stat.p_value is not None:
        routes.append("p_value")
    else:
        reasons["p_value"] = "no p value was resolved for this contrast"

    reported = values.reported
    if reported is not None and reported.value is not None and reported.scale != "unknown":
        routes.append("reported_d")
    else:
        reasons["reported_d"] = "the paper reports no effect size for this contrast"
    return routes, reasons


# ----------------------------------------------------------------------------- conversions
def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def _ci_dist(n: float, settings: StatsSettings) -> Literal["z", "t"]:
    if settings.ci_to_sd_dist in ("z", "t"):
        return settings.ci_to_sd_dist
    return "t" if n < CI_T_BELOW_N else "z"          # amendment A: t(n−1) when n < 100


def _mean_sd(group: GroupValues, side: str, settings: StatsSettings, steps: list[str],
             flags: list[str]) -> tuple[float, float, int]:
    """One group's `(mean, sd, n)` and the conversion steps that produced them."""
    kind = group.dispersion_type
    if len(group.points) >= 2:
        mean, sd, n = es.mean_sd_from_points(group.points)
        steps.append(f"group {side}: {n} digitised points → mean {_fmt(mean)}, SD {_fmt(sd)}")
        if group.n is not None and group.n != n:
            flags.append("points_n_mismatch")
            steps.append(f"group {side}: {n} points were read where n = {group.n} was analysed")
        return mean, sd, n

    n = int(group.n)
    if kind is DispersionType.SD:
        steps.append(f"group {side}: mean {_fmt(group.mean)}, SD {_fmt(group.dispersion_value)} "
                     f"as printed (n = {n})")
        return float(group.mean), float(group.dispersion_value), n

    if kind is DispersionType.SE:
        sd = es.sd_from_se(group.dispersion_value, n)
        steps.append(f"SD_{side} = SE {_fmt(group.dispersion_value)} × √{n} = {_fmt(sd)}")
        return float(group.mean), sd, n

    if kind in _CI_LEVELS:
        level = group.ci_level or _CI_LEVELS[kind]
        dist = _ci_dist(n, settings)
        label = f"t({n - 1})" if dist == "t" else "z"
        if group.ci_low is not None and group.ci_high is not None:
            sd = es.sd_from_ci(group.ci_low, group.ci_high, n, level=level, dist=dist)
            steps.append(f"SD_{side} = half of the {level:.0%} interval "
                         f"[{_fmt(group.ci_low)}, {_fmt(group.ci_high)}] ÷ {label} × √{n} "
                         f"= {_fmt(sd)}")
        else:
            sd = es.sd_from_ci_halfwidth(group.dispersion_value, n, level=level, dist=dist)
            steps.append(f"SD_{side} = interval half-width {_fmt(group.dispersion_value)} ÷ "
                         f"{label} × √{n} = {_fmt(sd)}")
        return float(group.mean), sd, n

    if kind is DispersionType.IQR:
        if group.q1 is not None and group.q3 is not None:
            median = group.median if group.median is not None else group.mean
            mean, sd = mean_sd_from_median_iqr(median, group.q1, group.q3, n)
            flags.append("median_iqr_conversion")
            steps.append(f"group {side}: median {_fmt(median)} with quartiles "
                         f"[{_fmt(group.q1)}, {_fmt(group.q3)}] → mean {_fmt(mean)} (Luo 2018), "
                         f"SD {_fmt(sd)} (Wan 2014 eq. 16)")
            return mean, sd, n
        sd = es.sd_from_iqr(0.0, group.dispersion_value, n)
        flags.append("median_iqr_conversion")
        steps.append(f"SD_{side} = IQR {_fmt(group.dispersion_value)} ÷ η({n}) = {_fmt(sd)} "
                     f"(Wan 2014 eq. 16)")
        return float(group.mean), sd, n

    if kind is DispersionType.RANGE:
        if group.q1 is not None and group.q3 is not None and group.median is not None:
            mean, sd = mean_sd_from_five_number(group.minimum, group.q1, group.median, group.q3,
                                                group.maximum, n)
            flags.append("five_number_conversion")
            steps.append(f"group {side}: five-number summary → mean {_fmt(mean)} (Luo 2018), "
                         f"SD {_fmt(sd)} (Shi 2020)")
            return mean, sd, n
        sd = es.sd_from_range(group.minimum, group.maximum, n)
        flags.append("range_to_sd")
        steps.append(f"SD_{side} = range [{_fmt(group.minimum)}, {_fmt(group.maximum)}] ÷ ξ({n}) "
                     f"= {_fmt(sd)} (Wan 2014 eq. 9 — a range is a weak estimate of a spread)")
        return float(group.mean), sd, n

    raise NotConvertible(f"group {side} has dispersion type {kind.value}, which is not a spread "
                         f"an effect size can be built from")


# ----------------------------------------------------------------------------- the routes
def _from_groups(values: ResolvedValues, settings: StatsSettings, inputs: dict[str, Any],
                 steps: list[str], flags: list[str]) -> SMDResult:
    mean_a, sd_a, n_a = _mean_sd(values.group_a, "A", settings, steps, flags)
    mean_b, sd_b, n_b = _mean_sd(values.group_b, "B", settings, steps, flags)
    inputs.update(mean_a=mean_a, sd_a=sd_a, n_a=n_a, mean_b=mean_b, sd_b=sd_b, n_b=n_b)
    pooled = es.pooled_sd(sd_a, n_a, sd_b, n_b)
    steps.append(f"pooled SD = √(((({n_a}−1)·{_fmt(sd_a)}² + ({n_b}−1)·{_fmt(sd_b)}²) / "
                 f"({n_a}+{n_b}−2)) = {_fmt(pooled)}")
    raw = (mean_a - mean_b) / pooled
    steps.append(f"d = ({_fmt(mean_a)} − {_fmt(mean_b)}) / {_fmt(pooled)} = {_fmt(raw, 5)}")
    return es.smd_from_means(mean_a, sd_a, n_a, mean_b, sd_b, n_b,
                             higher_is_better=values.higher_is_better,
                             estimator=settings.estimator, variance=settings.variance,
                             level=settings.ci_level)


def _sizes(values: ResolvedValues, dataset: DatasetSpec) -> tuple[int, int]:
    """Group sizes for a route that has no group statistics of its own."""
    a = (values.group_a.n if values.group_a and values.group_a.n else None) or dataset.group_a.n
    b = (values.group_b.n if values.group_b and values.group_b.n else None) or dataset.group_b.n
    if not a or not b:
        raise NotConvertible("the analysed group sizes are unknown, so a statistic cannot be "
                             "converted into an effect size")
    return int(a), int(b)


def _a_greater(direction: str, what: str) -> bool:
    if direction == "a_greater":
        return True
    if direction == "b_greater":
        return False
    raise NotConvertible(f"{what} carries no direction: the paper does not say which group scored "
                         f"higher, and a statistic has no sign of its own")


def _from_statistic(values: ResolvedValues, dataset: DatasetSpec, settings: StatsSettings,
                    inputs: dict[str, Any], steps: list[str], as_p: bool) -> SMDResult:
    stat = values.test_statistic
    n_a, n_b = _sizes(values, dataset)
    inputs.update(stat_value=stat.value, df=stat.df, df1=stat.df1, df2=stat.df2,
                  p_value=stat.p_value, n_a=n_a, n_b=n_b)
    common = dict(higher_is_better=values.higher_is_better, estimator=settings.estimator,
                  variance=settings.variance, level=settings.ci_level)

    if as_p:
        if stat.p_kind != "exact":
            raise NotConvertible(f"the p value is reported as {stat.p_kind!r}, not exactly, so the "
                                 f"statistic behind it cannot be recovered")
        a_greater = _a_greater(stat.direction, f"p = {stat.p_value:g}")
        steps.append(f"p = {_fmt(stat.p_value)} (two-tailed, df {stat.df}) → t → "
                     f"d = t·√(1/{n_a} + 1/{n_b})")
        return es.smd_from_p(stat.p_value, n_a, n_b, a_greater=a_greater, df=stat.df,
                             design=stat.design, two_tailed=(stat.tails or 2) == 2, **common)

    if stat.stat_type == "t":
        a_greater = _a_greater(stat.direction, f"t = {stat.value:g}")
        steps.append(f"t = {_fmt(stat.value)} (df {stat.df:g}, design {stat.design}) → "
                     f"d = t·√(1/{n_a} + 1/{n_b})" if stat.df is not None else
                     f"t = {_fmt(stat.value)} (design {stat.design}) → "
                     f"d = t·√(1/{n_a} + 1/{n_b})")
        return es.smd_from_t(stat.value, n_a, n_b, df=stat.df, design=stat.design,
                             positive_means_a_greater=a_greater, **common)

    a_greater = _a_greater(stat.direction, f"F = {stat.value:g}")
    steps.append(f"F({stat.df1:g}, {stat.df2:g}) = {_fmt(stat.value)} (design {stat.design}) → "
                 f"d = √F·√(1/{n_a} + 1/{n_b})" if stat.df1 is not None and stat.df2 is not None
                 else f"F = {_fmt(stat.value)} (design {stat.design})")
    return es.smd_from_f(stat.value, n_a, n_b, a_greater=a_greater, df1=stat.df1, df2=stat.df2,
                         design=stat.design, **common)


def _from_reported(values: ResolvedValues, dataset: DatasetSpec, settings: StatsSettings,
                   inputs: dict[str, Any], steps: list[str]) -> SMDResult:
    reported = values.reported
    n_a, n_b = _sizes(values, dataset)
    inputs.update(reported_value=reported.value, n_a=n_a, n_b=n_b)
    if reported.scale not in _BETWEEN_SCALES:
        raise NotConvertible(f"a reported effect size on the {reported.scale!r} scale is not a "
                             f"difference between two groups in standard-deviation units")
    if reported.standardizer in _WITHIN_STANDARDIZERS:
        raise NotConvertible(f"the reported effect size was standardised by "
                             f"{reported.standardizer!r}, not by the between-participant standard "
                             f"deviation")
    a_greater = _a_greater(reported.positive_means, f"the reported {reported.scale}")
    steps.append(f"the paper reports {reported.scale} = {_fmt(reported.value)} "
                 f"(standardised by {reported.standardizer}), positive means "
                 f"{reported.positive_means}")
    return es.smd_from_reported(reported.value, n_a, n_b,
                                positive_means_a_greater=a_greater,
                                is_hedges_g=(reported.scale == "hedges_g"),
                                higher_is_better=values.higher_is_better,
                                estimator=settings.estimator, variance=settings.variance,
                                level=settings.ci_level)


# ----------------------------------------------------------------------------- digitisation
def _digitization_variance(values: ResolvedValues, inputs: dict[str, Any],
                           settings: StatsSettings) -> tuple[float | None, dict[str, float]]:
    """Σ (∂es/∂x · σ_x)² over the digitised inputs, by central finite differences (spec §3.4)."""
    a, b = values.group_a, values.group_b
    if a is None or b is None or "sd_a" not in inputs:
        return None, {}
    sigmas = {"m_a": a.sigma, "m_b": b.sigma, "sd_a": a.dispersion_sigma,
              "sd_b": b.dispersion_sigma}
    if not any(sigma for sigma in sigmas.values()):
        return None, {}
    n_a, n_b = inputs["n_a"], inputs["n_b"]

    def estimate(m_a: float, sd_a: float, m_b: float, sd_b: float) -> float:
        d = es.cohens_d(m_a, sd_a, n_a, m_b, sd_b, n_b)
        return d if settings.estimator == "cohen" else es.hedges_g(d, n_a, n_b)

    point = {"m_a": inputs["mean_a"], "sd_a": inputs["sd_a"], "m_b": inputs["mean_b"],
             "sd_b": inputs["sd_b"]}
    return partial_variance(estimate, point, sigmas)


# ----------------------------------------------------------------------------- entry point
def resolve_effect(dataset: DatasetSpec, outcome_def: OutcomeDef, resolved_values: ResolvedValues,
                   settings: StatsSettings) -> EffectSizeRecord:
    """One (dataset × outcome) into one `EffectSizeRecord`, by the protocol's route precedence."""
    values = resolved_values
    record = EffectSizeRecord(
        cluster_id=dataset.cluster_id, dataset_id=values.dataset_id or dataset.dataset_id,
        outcome_key=values.outcome_key or outcome_def.key,
        label=dataset.label or outcome_def.label, estimator=settings.estimator,
        variance_method=settings.variance, level=settings.ci_level,
        higher_is_better=values.higher_is_better, confidence=values.confidence,
        moderators=dict(dataset.moderators), flags=list(values.flags))

    routes, why_missing = available_routes(values)
    record.routes_available = list(routes)
    if values.higher_is_better is None:
        return _not_convertible(record, "the direction of this measure is unresolved, so an "
                                        "effect size built from it could not be signed",
                                ["orientation_unresolved"])

    rejected: dict[str, str] = {}
    inputs: dict[str, Any] = {}
    for name in settings.route_precedence:
        if name not in routes:
            continue
        if values.route_available and name not in values.route_available:
            rejected[name] = "the orchestrator did not offer this route for this cell"
            continue
        steps: list[str] = []
        flags: list[str] = []
        attempt: dict[str, Any] = {}
        try:
            result = _run_route(name, values, dataset, settings, attempt, steps, flags)
        except (NotConvertible, ValueError) as exc:
            rejected[name] = str(exc)
            inputs = attempt or inputs
            continue
        return _finish(record, name, result, attempt, steps, flags, rejected, values, settings)

    reason = "; ".join(f"{name}: {why}" for name, why in rejected.items()) \
        or "; ".join(f"{name}: {why}" for name, why in why_missing.items()) \
        or "no route produced an effect size"
    record.routes_rejected = rejected
    record.inputs = {k: (float(v) if isinstance(v, (int, float)) else None)
                     for k, v in inputs.items()}
    return _not_convertible(record, reason, [])


def _run_route(name: str, values: ResolvedValues, dataset: DatasetSpec, settings: StatsSettings,
               inputs: dict[str, Any], steps: list[str], flags: list[str]) -> SMDResult:
    if name in GROUP_ROUTES:
        return _from_groups(values, settings, inputs, steps, flags)
    if name in ("test_statistic", "p_value"):
        return _from_statistic(values, dataset, settings, inputs, steps, name == "p_value")
    if name == "reported_d":
        return _from_reported(values, dataset, settings, inputs, steps)
    raise NotConvertible(f"unknown route {name!r} in route_precedence")


def _not_convertible(record: EffectSizeRecord, reason: str,
                     flags: Sequence[str]) -> EffectSizeRecord:
    record.route = "not_convertible"
    record.not_convertible_reason = reason
    record.flags = sorted(set(record.flags) | set(flags) | {"not_convertible"})
    record.conversion_steps = [f"not convertible: {reason}"]
    record.conversion_chain = record.conversion_steps[0]
    return record


def _finish(record: EffectSizeRecord, name: str, result: SMDResult, inputs: dict[str, Any],
            steps: list[str], flags: list[str], rejected: dict[str, str],
            values: ResolvedValues, settings: StatsSettings) -> EffectSizeRecord:
    record.route = name
    record.routes_rejected = rejected
    record.n_a, record.n_b = int(result.n_a), int(result.n_b)
    record.d, record.g, record.es = result.d, result.g, result.es
    record.var, record.se = result.var, result.se
    record.ci_low, record.ci_high = result.ci_low, result.ci_high
    record.orientation_applied = True

    steps.append(f"oriented: a larger raw value means more of the construct = "
                 f"{values.higher_is_better}, so d = {_fmt(result.d, 5)}")
    if settings.estimator == "hedges":
        steps.append(f"g = J({record.n_a}+{record.n_b}−2) × d = {_fmt(result.g, 5)}")
    steps.append(f"var ({settings.variance}) = {_fmt(result.var)}, SE = {_fmt(result.se)}, "
                 f"{settings.ci_level:.0%} CI [{_fmt(result.ci_low)}, {_fmt(result.ci_high)}]")
    for flag in result.details.get("flags", []):
        flags.append(flag)

    digitization, _ = _digitization_variance(values, inputs, settings)
    if digitization is not None and settings.digitization_variance != "off":
        record.digitization_var = digitization
        record.var_with_digitization = result.var + digitization
        record.digitization_var_share = digitization / result.var if result.var else None
        steps.append(f"digitisation variance (delta method) = {_fmt(digitization)}, "
                     f"{digitization / result.var:.1%} of the sampling variance")
        if record.digitization_var_share and record.digitization_var_share > DIGITIZATION_SHARE_FLAG:
            flags.append("digitization_variance_large")
        if settings.digitization_variance == "primary":
            record.var = record.var_with_digitization
            record.se = math.sqrt(record.var)
            record.ci_low, record.ci_high = es.ci_smd(result.es, record.se, settings.ci_level)
            steps.append(f"digitisation variance included in the primary analysis: SE = "
                         f"{_fmt(record.se)}, {settings.ci_level:.0%} CI "
                         f"[{_fmt(record.ci_low)}, {_fmt(record.ci_high)}]")

    record.conversion_steps = steps
    record.conversion_chain = "; ".join(steps)
    record.inputs = {k: (float(v) if isinstance(v, (int, float)) else None)
                     for k, v in inputs.items()}
    record.flags = sorted(set(record.flags) | set(flags))
    return record


# ----------------------------------------------------------------------------- dependence policies
def apply_shared_control(rows: Sequence[ResolvedValues],
                         strategy: str = "split_n", *,
                         shared: GroupKey = "B") -> list[ResolvedValues]:
    """`k` comparisons that share one arm, adjusted per `StatsSettings.shared_control_strategy`.

    The orchestrator (Task 10) groups rows by `(cluster_id, outcome_key)` and the shared arm, then
    calls this once per group; a single comparison is returned untouched. Nothing here computes an
    effect size — the returned rows go through `resolve_effect` exactly like any other.

    * `split_n`          — the shared arm contributes n/k to each comparison (Cochrane 16.5.4);
    * `combine_arms`     — the other arms are merged into one, leaving a single independent row;
    * `keep_first`       — only the first comparison survives;
    * `keep_all_flagged` — every comparison survives, each flagged as dependent.
    """
    rows = list(rows)
    if len(rows) < 2:
        return [row.model_copy(deep=True) for row in rows]
    other: GroupKey = "A" if shared == "B" else "B"

    if strategy == "split_n":
        out = []
        for row in rows:
            copy = row.model_copy(deep=True)
            group = copy.group(shared)
            if group is not None and group.n:
                group.n = int(round(split_control(group.n, len(rows))))
            copy.flags = sorted(set(copy.flags) | {"shared_control_split"})
            out.append(copy)
        return out

    if strategy == "combine_arms":
        merged = rows[0].model_copy(deep=True)
        arm = merged.group(other)
        for row in rows[1:]:
            addition = row.group(other)
            if arm is None or addition is None or arm.mean is None or addition.mean is None:
                raise ValueError("combine_arms needs a mean, SD and n in every arm being combined")
            mean, sd, n = combine_groups(arm.mean, arm.dispersion_value or 0.0, arm.n or 0,
                                         addition.mean, addition.dispersion_value or 0.0,
                                         addition.n or 0)
            arm.mean, arm.dispersion_value, arm.n = mean, sd, n
            arm.dispersion_type = DispersionType.SD
        merged.flags = sorted(set(merged.flags) | {"shared_control_combined"})
        return [merged]

    if strategy == "keep_first":
        first = rows[0].model_copy(deep=True)
        first.flags = sorted(set(first.flags) | {"shared_control_kept_first"})
        return [first]

    if strategy == "keep_all_flagged":
        out = []
        for row in rows:
            copy = row.model_copy(deep=True)
            copy.flags = sorted(set(copy.flags) | {"shared_control_repeated"})
            out.append(copy)
        return out

    raise ValueError(f"unknown shared-control strategy {strategy!r}")


def multi_group_flags(dataset: DatasetSpec, policy: str = "closest_to_definition") -> list[str]:
    """Flags for a dataset whose paper reported more than the two groups being contrasted.

    The mapper (Task 4) chose the pair and recorded every group it saw plus its rationale, so all
    that is left here is to say whether that choice may stand under the protocol's policy.
    `combine_matching` cannot be applied without per-group protocol matches, which the mapper does
    not record, so it is reported as needing a human rather than guessed at.
    """
    if len(dataset.all_groups_listed) <= 2:
        return []
    flags = [f"multi_group_{policy}"]
    if policy == "needs_human" or policy == "combine_matching":
        flags.append("multi_group_needs_human")
    if not dataset.chosen_pair_rationale:
        flags.append("multi_group_no_rationale")
    return flags
