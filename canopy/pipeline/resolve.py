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

from ..models import (Candidate, CanopyModel, ConfidenceBucket, ContrastKind, DatasetSpec,
                      Direction, DispersionType, EffectSizeRecord, GroupKey, OutcomeDef, PKind,
                      ReportedScale, Standardizer, StatsSettings, TestDesign, Verdict)
from ..stats import effect_sizes as es
from ..stats.conversions import (mean_sd_from_five_number, mean_sd_from_median_iqr,
                                 combine_groups, partial_variance, split_control)
from ..stats.effect_sizes import NotConvertible, SMDResult
from ..verify.confidence import (DF_SHORTFALL_PREFIX, IMPLAUSIBLE_DISPERSION, ROW_REFUSAL_CODES,
                                 conversion_gate_bucket, dispersion_plausibility_bucket)
from ..verify.vote import modality as reading_modality

__all__ = ["resolve_effect", "resolve_effect_with_fallback", "available_routes",
           "apply_shared_control", "multi_group_flags", "ROW_REFUSAL_CODES",
           "GroupValues", "StatisticValues", "ReportedValues", "ResolvedValues",
           "GROUP_ROUTES", "DIGITIZATION_SHARE_FLAG", "GROUP_STATISTICS_MISSING",
           "PRECEDENCE_OVERRIDE"]

#: the route names in `StatsSettings.route_precedence` that are built from two groups' statistics
GROUP_ROUTES: tuple[str, ...] = ("text_mean_sd", "table", "text_mean_se_ci", "figure")
#: digitisation variance above this share of the sampling variance is flagged (spec §3.4)
DIGITIZATION_SHARE_FLAG = 0.10
#: the record's own note that no route built from the two groups' statistics was available — the
#: paper printed a value for each group and no spread for either, so the four group routes were
#: never even attempted. It is the ONE condition D1's precedence override reads (Blocker 1): the
#: reason lives in `routes_rejected` as prose, and prose is not something another stage can act on.
GROUP_STATISTICS_MISSING = "group_statistics_missing"
#: D1: this row's effect size came from a same-locator candidate pair rather than from the value
#: the precedence list chose, because that value converts to nothing. Always with
#: `confidence = "needs_human"` — the row is held, not released.
PRECEDENCE_OVERRIDE = "precedence_override"
#: how much of a locator the override's reason prints — enough to name the panel, not the sentence
_LOCATOR_CHARS = 72
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
    #: WHERE this group's numbers were read, in the reader's own words. Two panels of one figure
    #: are two quantities (`verify.vote.locator_key`), so D1's fallback pair may only be built
    #: from two readings taken at the SAME place. Empty for a value the verdict settled — a
    #: verdict is one cell's answer and has no place of its own.
    locator: str = ""
    #: the candidate this group came from, when it came from ONE candidate rather than a verdict.
    #: The precedence override names the readings it used, and the verifier's objections are
    #: recorded against candidate ids, so without this the override could neither cite its
    #: evidence nor find the objection to it.
    candidate_id: str = ""

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

    @classmethod
    def from_candidate(cls, cand: Candidate) -> "GroupValues":
        """ONE extractor's reading as resolution inputs — D1's fallback, and nothing else.

        The ordinary path is `from_verdict`: the verification layer weighs the readings and
        records one answer per cell. This reads a single candidate instead, because the override
        is defined on a candidate PAIR (`rows.fallback_values`) — the two readings taken at one
        place in one figure — and a pair has no verdict of its own.

        The dispersion is read the way `from_verdict` reads it, from the same fields, so an IQR is
        quartiles around a median here too; a candidate and the verdict built from it must not
        mean different things by `ci_low`.
        """
        values = cls(n=cand.n, mean=cand.mean, dispersion_value=cand.dispersion_value,
                     dispersion_type=cand.dispersion_type, points=list(cand.points),
                     sigma=cand.sigma, dispersion_sigma=cand.dispersion_sigma, unit=cand.unit,
                     route=reading_modality(cand), locator=cand.locator,
                     candidate_id=cand.candidate_id)
        if cand.dispersion_type is DispersionType.IQR:
            values.median, values.q1, values.q3 = cand.mean, cand.ci_low, cand.ci_high
        elif cand.dispersion_type is DispersionType.RANGE:
            values.minimum, values.maximum = cand.ci_low, cand.ci_high
        else:
            values.ci_low, values.ci_high = cand.ci_low, cand.ci_high
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
    #: what the statistic contrasts and what its model averaged over — the extractor's record,
    #: carried here so the conversion gate can refuse on it (P-B, C5). The defaults are the
    #: refusing ones: "nobody recorded this" is not "it is fine".
    contrast_kind: ContrastKind = "unknown"
    within_factors: list[str] = Field(default_factory=list)
    outcome_averages_over: list[str] = Field(default_factory=list)


class ReportedValues(CanopyModel):
    """An effect size the paper itself printed."""

    value: float | None = None
    scale: ReportedScale = "unknown"
    standardizer: Standardizer = "unknown"
    ci_low: float | None = None
    ci_high: float | None = None
    positive_means: Direction = "unknown"
    #: WHAT the printed effect size contrasts (P-B). A printed d has no test statistic behind it
    #: but it always has a contrast: "the aftereffect differed from zero, d = 1.30" is a
    #: one-sample effect, and it pooled as the between-group difference because this field was
    #: extracted, carried on the `Candidate`, and read by nobody on this route. The default is
    #: the refusing one, as it is for a statistic.
    contrast_kind: ContrastKind = "unknown"


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
    #: `candidate_id -> the verifier's objection to it`, for the readings this cell's verifiers
    #: doubted or refuted. The verdict's flags carry codes and this carries the sentence, because
    #: D1's precedence override has to quote the objection to the reading it falls back to: a row
    #: built from a candidate a verifier refuted is still held, and the reviewer who is asked to
    #: decide it must be shown the objection rather than have to go and find it.
    objections: dict[str, str] = Field(default_factory=dict)

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
                   higher_is_better=direction, confidence=buckets[0], flags=flags,
                   objections={note.candidate_id: note.reason
                               for verdict in (verdict_a, verdict_b) for note in verdict.verifiers
                               if note.verdict in ("refuted", "ambiguous")
                               and note.candidate_id and note.reason})


# ----------------------------------------------------------------------------- route availability
def _modality(route: str) -> str:
    """`text` | `table` | `figure` | `unknown` — where a group's numbers came from.

    An unrecognised route string is `unknown`, not text: it is named in the record's flags so a
    reviewer sees that the pipeline could not tell where the value was read, instead of the value
    quietly claiming the highest-precedence route in the list.
    """
    name = (route or "").lower()
    if name.startswith(("figure", "digitize")):
        return "figure"
    if name == "table":
        return "table"
    return "text" if name in _TEXT_ROUTES else "unknown"


def _unknown_routes(values: ResolvedValues) -> list[str]:
    return sorted({g.route for g in (values.group_a, values.group_b)
                   if g is not None and _modality(g.route) == "unknown"})


def _stated_level(group: GroupValues) -> float | None:
    """An interval whose confidence level the extractor stated but whose type has no enum member.

    `DispersionType` names the 95% and 90% intervals; a paper that prints a 99% interval reaches
    here with `dispersion_type=UNKNOWN` and an explicit `ci_level`, and that is enough to convert
    it. Nothing sets `ci_level` by accident, so this never reinterprets a genuinely unknown spread.
    """
    if group.ci_level and (group.ci_low is not None or group.dispersion_value is not None):
        return float(group.ci_level)
    return None


def _centre(group: GroupValues) -> float:
    """The group's central value: its mean, or the median when only a median was reported."""
    centre = group.mean if group.mean is not None else group.median
    if centre is None:
        raise NotConvertible("this group reports neither a mean nor a median")
    return float(centre)


def _scaled_sigma(dispersion_sigma: float | None, as_read: float | None,
                  sd: float) -> float | None:
    """A digitisation uncertainty converted alongside the dispersion it belongs to.

    `GroupValues.dispersion_sigma` is in the units of the spread AS READ — half a pixel on an
    error-bar cap is half a pixel's worth of SE if the bar is an SE bar. Converting the bar to an
    SD without converting its uncertainty understates the digitisation variance by the same factor
    (√n for SE → SD), so σ travels through the identical linear factor as the value.
    """
    if dispersion_sigma is None or dispersion_sigma <= 0 or not as_read:
        return dispersion_sigma if dispersion_sigma else None
    return abs(dispersion_sigma) * abs(sd / as_read)


def _has_spread(group: GroupValues) -> bool:
    kind = group.dispersion_type
    if len(group.points) >= 2:
        return True
    if kind is DispersionType.SD or kind is DispersionType.SE:
        return group.dispersion_value is not None and group.dispersion_value > 0
    if kind in _CI_LEVELS or _stated_level(group) is not None:
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
    # P-B: the p route inverts a p BACK to the t it came from, so it is only a route for a
    # statistic that could have been a t or an F in the first place. Offered on `p_value is not
    # None` alone it walked around the chi-square refusal: chi2(1) = 7.58, p = .006 on 12/12
    # became d = 1.2414, a standardised mean difference invented out of a contingency table.
    if stat is not None and stat.p_value is not None and stat.stat_type in ("t", "F", "p"):
        routes.append("p_value")
    elif stat is not None and stat.p_value is not None:
        reasons["p_value"] = (f"a {stat.stat_type} is not a t, an F or a p, so its p value cannot "
                              f"be inverted back to a statistic this contrast could use")
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
             flags: list[str]) -> tuple[float, float, int, float | None]:
    """One group's `(mean, sd, n, sd_sigma)` and the conversion steps that produced them."""
    kind = group.dispersion_type
    if len(group.points) >= 2:
        mean, sd, n = es.mean_sd_from_points(group.points)
        steps.append(f"group {side}: {n} digitised points → mean {_fmt(mean)}, SD {_fmt(sd)}")
        if group.n is not None and group.n != n:
            flags.append("points_n_mismatch")
            steps.append(f"group {side}: {n} points were read where n = {group.n} was analysed")
        return mean, sd, n, None

    n = int(group.n)
    centre = _centre(group)
    if kind is DispersionType.SD:
        steps.append(f"group {side}: mean {_fmt(centre)}, SD {_fmt(group.dispersion_value)} "
                     f"as printed (n = {n})")
        return centre, float(group.dispersion_value), n, group.dispersion_sigma

    if kind is DispersionType.SE:
        sd = es.sd_from_se(group.dispersion_value, n)
        sigma = _scaled_sigma(group.dispersion_sigma, group.dispersion_value, sd)
        steps.append(f"SD_{side} = SE {_fmt(group.dispersion_value)} × √{n} = {_fmt(sd)}")
        if sigma is not None and group.dispersion_sigma:
            steps.append(f"digitisation uncertainty on the SE bar, {_fmt(group.dispersion_sigma)}, "
                         f"scales with it: ±{_fmt(sigma)} on the SD")
        return centre, sd, n, sigma

    if kind in _CI_LEVELS or _stated_level(group) is not None:
        level = group.ci_level or _CI_LEVELS.get(kind) or 0.95
        dist = _ci_dist(n, settings)
        label = f"t({n - 1})" if dist == "t" else "z"
        if group.ci_low is not None and group.ci_high is not None:
            half = (group.ci_high - group.ci_low) / 2
            sd = es.sd_from_ci(group.ci_low, group.ci_high, n, level=level, dist=dist)
            steps.append(f"SD_{side} = half of the {level:.0%} interval "
                         f"[{_fmt(group.ci_low)}, {_fmt(group.ci_high)}] ÷ {label} × √{n} "
                         f"= {_fmt(sd)}")
        else:
            half = group.dispersion_value
            sd = es.sd_from_ci_halfwidth(group.dispersion_value, n, level=level, dist=dist)
            steps.append(f"SD_{side} = interval half-width {_fmt(group.dispersion_value)} ÷ "
                         f"{label} × √{n} = {_fmt(sd)}")
        sigma = _scaled_sigma(group.dispersion_sigma, half, sd)
        if sigma is not None and group.dispersion_sigma:
            steps.append(f"digitisation uncertainty on the interval, "
                         f"{_fmt(group.dispersion_sigma)}, scales with it: ±{_fmt(sigma)} on the SD")
        return centre, sd, n, sigma

    if kind is DispersionType.IQR:
        if group.q1 is not None and group.q3 is not None:
            median = group.median if group.median is not None else _centre(group)
            mean, sd = mean_sd_from_median_iqr(median, group.q1, group.q3, n)
            flags.append("median_iqr_conversion")
            steps.append(f"group {side}: median {_fmt(median)} with quartiles "
                         f"[{_fmt(group.q1)}, {_fmt(group.q3)}] → mean {_fmt(mean)} (Luo 2018), "
                         f"SD {_fmt(sd)} (Wan 2014 eq. 16)")
            return mean, sd, n, _scaled_sigma(group.dispersion_sigma, group.q3 - group.q1, sd)
        sd = es.sd_from_iqr(0.0, group.dispersion_value, n)
        flags.append("median_iqr_conversion")
        steps.append(f"SD_{side} = IQR {_fmt(group.dispersion_value)} ÷ η({n}) = {_fmt(sd)} "
                     f"(Wan 2014 eq. 16)")
        return centre, sd, n, _scaled_sigma(group.dispersion_sigma, group.dispersion_value, sd)

    if kind is DispersionType.RANGE:
        if group.q1 is not None and group.q3 is not None and group.median is not None:
            mean, sd = mean_sd_from_five_number(group.minimum, group.q1, group.median, group.q3,
                                                group.maximum, n)
            flags.append("five_number_conversion")
            steps.append(f"group {side}: five-number summary → mean {_fmt(mean)} (Luo 2018), "
                         f"SD {_fmt(sd)} (Shi 2020)")
            return mean, sd, n, _scaled_sigma(group.dispersion_sigma,
                                              group.maximum - group.minimum, sd)
        sd = es.sd_from_range(group.minimum, group.maximum, n)
        flags.append("range_to_sd")
        steps.append(f"group {side}: mean {_fmt(centre)}; SD_{side} = range "
                     f"[{_fmt(group.minimum)}, {_fmt(group.maximum)}] ÷ ξ({n}) = {_fmt(sd)} "
                     f"(Wan 2014 eq. 9 — a range is a weak estimate of a spread)")
        return centre, sd, n, _scaled_sigma(group.dispersion_sigma,
                                            group.maximum - group.minimum, sd)

    raise NotConvertible(f"group {side} has dispersion type {kind.value}, which is not a spread "
                         f"an effect size can be built from")


# ----------------------------------------------------------------------------- the routes
def _from_groups(values: ResolvedValues, settings: StatsSettings, inputs: dict[str, Any],
                 steps: list[str], flags: list[str]) -> SMDResult:
    mean_a, sd_a, n_a, sigma_sd_a = _mean_sd(values.group_a, "A", settings, steps, flags)
    mean_b, sd_b, n_b, sigma_sd_b = _mean_sd(values.group_b, "B", settings, steps, flags)
    inputs.update(mean_a=mean_a, sd_a=sd_a, n_a=n_a, mean_b=mean_b, sd_b=sd_b, n_b=n_b,
                  sigma_mean_a=values.group_a.sigma, sigma_mean_b=values.group_b.sigma,
                  sigma_sd_a=sigma_sd_a, sigma_sd_b=sigma_sd_b)
    for name in _unknown_routes(values):
        flags.append("unknown_source_route")
        steps.append(f"the source route {name!r} is not one this pipeline names; the values were "
                     f"treated as printed text — a reviewer should confirm where they came from")
    pooled = es.pooled_sd(sd_a, n_a, sd_b, n_b)
    if pooled <= 0:
        raise NotConvertible(
            f"the pooled standard deviation is {pooled:g}: with no within-group variance a "
            f"standardised mean difference is unbounded, so these values cannot be pooled")
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


def _df_shortfall_explained(values: ResolvedValues) -> str:
    """The `df_off_by_<gap>` code the CELL raised, or `""` when nothing explained the shortfall."""
    return next((f for f in values.flags if f.startswith(DF_SHORTFALL_PREFIX)), "")


def _from_statistic(values: ResolvedValues, dataset: DatasetSpec, settings: StatsSettings,
                    inputs: dict[str, Any], steps: list[str], as_p: bool) -> SMDResult:
    stat = values.test_statistic
    n_a, n_b = _sizes(values, dataset)
    inputs.update(stat_value=stat.value, df=stat.df, df1=stat.df1, df2=stat.df2,
                  p_value=stat.p_value, n_a=n_a, n_b=n_b)
    common = dict(higher_is_better=values.higher_is_better, estimator=settings.estimator,
                  variance=settings.variance, level=settings.ci_level,
                  # P-B / C5: the gate that refuses a statistic answering a different question
                  # than this cell asks. Always passed — an unrecorded contrast is a refusal.
                  contrast_kind=stat.contrast_kind or "unknown",
                  within_factors=list(stat.within_factors),
                  outcome_averages_over=list(stat.outcome_averages_over),
                  # C9: the conversion gate now demands df == n_a + n_b - 2 exactly. Whether a
                  # shortfall is EXPLAINED is a fact about the paper, decided once by
                  # `checks._shortfall_is_explained` and recorded on the cell as `df_off_by_<gap>`.
                  # Reading the record here keeps one rule in one place; a cell that reached
                  # `df_shortfall_unexplained` (or no verdict at all) says nothing, which refuses.
                  df_shortfall_explained=_df_shortfall_explained(values))

    # P-B: refused HERE, before any branch, because every branch is a conversion. A chi-square
    # tests an association in a contingency table and a statistic of no recorded kind has no
    # formula at all; the guard used to sit below the `as_p` branch, so a chi2 carrying an exact
    # p converted through the p route as if it were a t.
    if stat.stat_type == "chi2":
        raise NotConvertible(
            f"chi2 = {stat.value:g} is not a difference between two group means: no conversion "
            f"from a chi-square to a standardised mean difference exists for a continuous outcome"
            if stat.value is not None else
            "a chi-square is not a difference between two group means: no conversion from a "
            "chi-square to a standardised mean difference exists for a continuous outcome")
    if stat.stat_type == "unknown":
        raise NotConvertible(
            "the paper's statistic was transcribed without saying whether it is a t, an F or a p, "
            "so there is no formula to apply to it")

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
    # P-B: what it contrasts decides before what it is measured in — an effect size for a test
    # against a constant is not this cell's number whatever scale it is printed on.
    ok, why = es.contrast_ok(reported.contrast_kind or "unknown")
    if not ok:
        raise NotConvertible(f"the reported {reported.scale} cannot carry this cell: {why}")
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
                                level=settings.ci_level,
                                contrast_kind=reported.contrast_kind or "unknown")


# ----------------------------------------------------------------------------- digitisation
def _digitization_variance(values: ResolvedValues, inputs: dict[str, Any],
                           settings: StatsSettings) -> tuple[float | None, dict[str, float]]:
    """Σ (∂es/∂x · σ_x)² over the digitised inputs, by central finite differences (spec §3.4)."""
    a, b = values.group_a, values.group_b
    if a is None or b is None or "sd_a" not in inputs:
        return None, {}
    # the σ recorded on a spread is in the units of that spread AS READ; `_mean_sd` converted it
    # alongside the value, and those converted numbers are what the derivatives here multiply
    sigmas = {"m_a": inputs.get("sigma_mean_a"), "m_b": inputs.get("sigma_mean_b"),
              "sd_a": inputs.get("sigma_sd_a"), "sd_b": inputs.get("sigma_sd_b")}
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
    # Blocker 1. "group(s) A, B have no mean, group size and dispersion" is a sentence in
    # `routes_rejected`, and a sentence is not something a later stage can act on. As a FLAG it
    # is the one condition D1's precedence override reads — and it is recorded before the
    # orientation refusal below, because whether the paper printed a spread is a fact about the
    # paper, not about whether anyone settled which direction is better.
    if "group_statistics" in why_missing:
        record.flags = sorted(set(record.flags) | {GROUP_STATISTICS_MISSING})
    if values.higher_is_better is None:
        return _not_convertible(record, "the direction of this measure is unresolved, so an "
                                        "effect size built from it could not be signed",
                                ["orientation_unresolved"])

    rejected: dict[str, str] = {}
    inputs: dict[str, Any] = {}
    for name in settings.route_precedence:
        if name not in routes:
            why = why_missing.get(name) or (why_missing.get("group_statistics")
                                            if name in GROUP_ROUTES else None)
            if why:
                rejected[name] = why
            continue
        if values.route_available and name not in values.route_available:
            rejected[name] = "the orchestrator did not offer this route for this cell"
            continue
        steps: list[str] = []
        flags: list[str] = []
        attempt: dict[str, Any] = {}
        try:
            result = _run_route(name, values, dataset, settings, attempt, steps, flags)
        except NotConvertible as exc:
            rejected[name] = str(exc)
            inputs = attempt or inputs
            continue
        except Exception as exc:            # a broken cell must never take the paper down with it
            rejected[name] = f"{type(exc).__name__}: {exc}"
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


# ------------------------------------------------------------------- D1: the precedence override
def resolve_effect_with_fallback(dataset: DatasetSpec, outcome_def: OutcomeDef,
                                 primary: ResolvedValues,
                                 alternatives: Sequence[ResolvedValues],
                                 settings: StatsSettings) -> EffectSizeRecord:
    """`resolve_effect`, plus D1: a value that converts to nothing yields to one that converts.

    The precedence list prefers a printed value over a measured one, and it is right to: a number
    the paper prints can be quoted and checked, and one read off a picture cannot. But precedence
    presumes the preferred value CAN be converted. Heuer & Hegele 2008 is the case that shows the
    gap: the paper prints 27.7° and 18.9° for the two age groups and prints no spread for either,
    so the printed pair yields no effect size at all, while the figure both readers measured sits
    in the same cell with a mean, an SE and an n. Under strict precedence the cell contributes
    nothing — not a weaker number, NO number — and the review queue offers a human no way to
    supply one, because no route was ever available to override.

    So, scoped: *the printed value is preferred whenever it converts; when it cannot, the row may
    be built from a convertible same-locator candidate pair, and is HELD.* Held is the whole of
    the safety: `confidence = "needs_human"` on every overridden row, the swap written on the
    record (`route_overridden_from`, `precedence_override_reason`), and the flag
    `precedence_override` so a reader can find every one of them. Nothing is released by this.

    Four conditions, all necessary. The row got no effect size at all: `group_statistics_missing`
    says the four group routes were unavailable, and it does NOT say the row converted to nothing
    — a printed t beside a spreadless mean still converts — so the route is tested too, and the
    override can never demote a row that has a number. An unresolved orientation still refuses: an
    effect size nobody can sign is not improved by measuring it more precisely. And a row the
    resolver already refused (`ROW_REFUSAL_CODES`) stays refused — the fallback is not a way
    around a screen.

    `alternatives` come from `rows.fallback_values`, which is where the pairing rules live (same
    locator, same unit, one reading per route). The first that converts wins, and the ordering is
    the protocol's own `route_precedence`.
    """
    record = resolve_effect(dataset, outcome_def, primary, settings)
    if not _may_fall_back(record, primary):
        return record
    for alternative in alternatives:
        values = _with_row_context(alternative, primary)
        attempt = resolve_effect(dataset, outcome_def, values, settings)
        if _rank(attempt.route, settings) >= _rank(record.route, settings):
            continue        # not convertible, or a route precedence ranks BELOW the one we have
        # both flags: the swap, and the fact that made it necessary. The alternative HAS group
        # statistics, so `resolve_effect` would never raise the second on it — and a row that did
        # not say the printed value has no spread would be a row whose reader cannot tell an
        # override from an ordinary figure read (acceptance item 4).
        attempt.flags = sorted(set(attempt.flags) | {PRECEDENCE_OVERRIDE, GROUP_STATISTICS_MISSING})
        attempt.route_overridden_from = _group_route_name(primary)
        attempt.precedence_override_reason = _override_reason(primary, values, attempt,
                                                              alternatives)
        attempt.confidence = "needs_human"          # a row this was done to is never released
        return attempt
    return record


def _may_fall_back(record: EffectSizeRecord, primary: ResolvedValues) -> bool:
    """May this row be rebuilt from a candidate pair? The conditions of D1, in one place."""
    # D1 is scoped to a value that converts to NOTHING. `group_statistics_missing` says the four
    # GROUP routes were unavailable — it does not say the row got no effect size: a paper that
    # prints means with no spread and a t beside them converts through `test_statistic`, which
    # `figure` outranks in the default precedence. Without this line the rank guard below would
    # have taken a released row's number away and held the row, on a flag about a route that was
    # never used (review finding 2).
    if record.route != "not_convertible":
        return False
    if GROUP_STATISTICS_MISSING not in record.flags:
        return False
    if primary.higher_is_better is None or primary.group_a is None or primary.group_b is None:
        return False
    return not (set(record.flags) & ROW_REFUSAL_CODES)


def _rank(route: str, settings: StatsSettings) -> int:
    """Where a route sits in the protocol's precedence list; off the end when it is not on it.

    `not_convertible` is not a route in the list, so it ranks last — which is what makes the one
    comparison in the loop above cover both cases: a row that converted is only overridden by a
    route the protocol prefers to it, and a row that converted to nothing is overridden by any.
    """
    order = list(settings.route_precedence)
    return order.index(route) if route in order else len(order)


def _with_row_context(alternative: ResolvedValues, primary: ResolvedValues) -> ResolvedValues:
    """The alternative pair, carrying the ROW's context — its flags, orientation and bucket.

    The candidates supply two groups' numbers and nothing else. Everything else about the row is
    a fact about the cell and the dataset, not about where the numbers were read: the multi-group
    policy flag, the cell's warnings, the direction of the measure. Dropping them would build the
    override row out from under the checks the primary row was subject to.

    The printed statistic and the printed effect size are deliberately NOT carried: this row is
    the candidate pair or it is nothing, and a fallback that quietly converted a printed t would
    be an override nobody asked for.
    """
    values = alternative.model_copy(deep=True)
    values.dataset_id = values.dataset_id or primary.dataset_id
    values.outcome_key = values.outcome_key or primary.outcome_key
    values.higher_is_better = primary.higher_is_better
    values.route_available = list(primary.route_available)
    values.confidence = primary.confidence
    values.flags = sorted(set(values.flags) | set(primary.flags))
    values.objections = dict(primary.objections)
    values.test_statistic = None
    values.reported = None
    return values


def _shown(group: GroupValues | None, *, spread: bool) -> str:
    """One group's numbers as a reader would write them — `"-27 ± 4.713 (SE)"`, `"no value"`."""
    if group is None or (group.mean is None and group.median is None and not group.points):
        return "no value"
    centre = _fmt(group.mean if group.mean is not None else group.median) \
        if (group.mean is not None or group.median is not None) else f"{len(group.points)} points"
    if not spread or group.dispersion_value is None:
        return centre
    kind = group.dispersion_type.value if group.dispersion_type else ""
    named = f" ({kind})" if kind and kind not in ("UNKNOWN", "NONE") else ""
    return f"{centre} ± {_fmt(group.dispersion_value)}{named}"


def _override_reason(primary: ResolvedValues, alternative: ResolvedValues,
                     record: EffectSizeRecord, alternatives: Sequence[ResolvedValues] = ()) -> str:
    """Why this row is not the number the precedence list asked for — in one readable sentence.

    It names both pairs, where the second was read, which candidates it is, and — because a
    reviewer decides this row — any objection this cell's verifiers raised against exactly those
    candidates. A refutation does not block the fallback (the printed value it argued for is the
    one that converts to nothing), but it is the first thing the human should see.

    And when the cell was read in more than one place, it says so. Vachon d2's aftereffect has a
    pair under Fig 4's left panel and another under Fig 3's top-right, and "the first by route
    precedence" chose between two pictures. The row is held either way, so nothing is released on
    that choice — but a reviewer who is not told a choice was made cannot revisit it.
    """
    pair = (alternative.group_a, alternative.group_b)
    ids = ", ".join(g.candidate_id for g in pair if g is not None and g.candidate_id)
    locator = next((g.locator for g in pair if g is not None and g.locator), "")
    said = (f"{_group_route_name(primary)} resolved "
            f"{_shown(primary.group_a, spread=False)}/{_shown(primary.group_b, spread=False)} "
            f"with no dispersion; {record.route} under {locator!r} resolved "
            f"{_shown(pair[0], spread=True)}/{_shown(pair[1], spread=True)} "
            f"(candidates {ids or 'unnamed'}), which converts")
    objection = "; ".join(dict.fromkeys(
        primary.objections.get(g.candidate_id, "") for g in pair
        if g is not None and primary.objections.get(g.candidate_id)))
    if objection:
        said += f"; verifier objection on those candidates: {objection[:200]!r}"
    elsewhere = sorted({(other.group_a.locator or "")[:_LOCATOR_CHARS]
                        for other in alternatives if other.group_a is not None
                        and other.group_a.locator and other.group_a.locator != locator})
    if elsewhere:
        said += (f"; this cell was also read at {len(elsewhere)} other place(s) "
                 f"({'; '.join(elsewhere)}), and this pair is the first by route precedence")
    return said


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
        share = (f"{record.digitization_var_share:.1%} of the sampling variance"
                 if record.digitization_var_share is not None
                 else "the sampling variance is zero, so its share is undefined")
        steps.append(f"digitisation variance (delta method) = {_fmt(digitization)}, {share}")
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

    # C9, the v1 defect this item names: the conversion gate's flags are raised HERE, while the
    # effect size is being built, and `record.confidence` was set from the two cells' buckets long
    # before that. Until this call, a `df_missing` row carried the flag and pooled anyway. The cap
    # is on the BUCKET rather than the score on purpose — "below auto_accept" still includes
    # `accept_with_note`, which pools, so a score cap could never withhold the row.
    capped, why = conversion_gate_bucket(record.confidence, record.route, record.flags)
    record.confidence = capped
    if why:                              # said even when the row was already held, so the reason
        steps.extend(why)                # a reviewer reads names the conversion, not just the cell
        record.conversion_steps = steps
        record.conversion_chain = "; ".join(steps)

    # C9's second half, on the number that reaches the plot (review H1). The cell-level check
    # screens raw candidates: only the SD-typed ones, an arbitrary one of them, and always before
    # the vote and the adjudicator. Here `record.d` is the resolved value — post-vote,
    # post-adjudication, post-conversion — so the screen finally sees the denominator the row was
    # actually divided by, whatever form the paper printed it in.
    screened, said = dispersion_plausibility_bucket(record.confidence, record.d,
                                                    denominator=_denominator_note(values, inputs),
                                                    route=name)
    if said:
        record.confidence = screened
        _add_row_refusal(record, IMPLAUSIBLE_DISPERSION)
        steps.extend(said)
        record.conversion_steps = steps
        record.conversion_chain = "; ".join(steps)
    return record


def _add_row_refusal(record: EffectSizeRecord, code: str) -> None:
    """Put a ROW-level refusal on the record — the only way `_finish` may add one.

    The review layer decides whether a cell can be released by consulting `ROW_REFUSAL_CODES`
    (`canopy.pipeline.overrides`), so a code this function could add without being in that set
    would release cells under a row the resolver refused. The check is here rather than in a test
    alone because the failure is silent everywhere else: the row is held, the cells are not, and
    nothing says so.
    """
    if code not in ROW_REFUSAL_CODES:
        raise KeyError(
            f"{code!r} is not in ROW_REFUSAL_CODES, so the review layer does not know it holds "
            f"this row's cells. Add it there — that set is the contract, not a description")
    record.flags = sorted(set(record.flags) | {code})


def _denominator_note(values: ResolvedValues, inputs: dict[str, Any]) -> str:
    """The two dispersions `_mean_sd` actually divided by, and what each was converted FROM.

    A reviewer told "|d| = 17.3 is implausible" and nothing else has to re-derive the conversion
    before they can see where it went wrong; told "SD_A = 1.732 (from SE 0.5)" they can see it at
    once. Empty for a route that has no group dispersions of its own (a statistic, a printed d).
    """
    said: list[str] = []
    for side, group in (("A", values.group_a), ("B", values.group_b)):
        sd = inputs.get(f"sd_{side.lower()}")
        if not isinstance(sd, (int, float)):
            continue
        printed = ""
        converted = (DispersionType.SD, DispersionType.UNKNOWN, DispersionType.NONE)
        if (group is not None and group.dispersion_value is not None
                and group.dispersion_type not in converted):
            printed = f" (from {group.dispersion_type.value} {_fmt(group.dispersion_value)})"
        mean = inputs.get(f"mean_{side.lower()}")
        against = (f" against a mean of {_fmt(float(mean))}"
                   if isinstance(mean, (int, float)) else "")
        said.append(f"SD_{side} = {_fmt(float(sd))}{printed}{against}")
    return "; ".join(said)


# ----------------------------------------------------------------------------- dependence policies
#: which arm `apply_shared_control` treats as the SHARED one when a caller does not say. Named,
#: rather than left as a positional default, because a second module has to know which arm the
#: adjustment touched in order not to undo it: `overrides._apply_group_n` re-splits the answered
#: size for exactly this arm, and reading "B" out of this signature by eye is how that coupling
#: silently breaks the day the default moves (review finding 3).
SHARED_CONTROL_ARM: GroupKey = "B"


def apply_shared_control(rows: Sequence[ResolvedValues],
                         strategy: str = "split_n", *,
                         shared: GroupKey = SHARED_CONTROL_ARM) -> list[ResolvedValues]:
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
