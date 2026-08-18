"""Standardized mean differences and the conversions needed to obtain them from what papers report.

Conventions (used everywhere in Canopy):
* Two groups: **A** and **B** (e.g. A = older adults, B = younger adults). Raw d is (M_A − M_B) / SD_pooled.
* ``higher_is_better`` says whether a larger raw score on the outcome means *better / more of the construct*
  (e.g. adaptation magnitude → True; error / deviation → False). ``orient()`` flips the sign so that a
  positive returned effect ALWAYS means "A shows more of the construct than B".
* Formulas follow Borenstein et al. (2009), the Cochrane Handbook (ch. 6), Viechtbauer (2010, metafor) and the
  R `meta` package source (metacont), so results can be checked against R exactly
  (see validation/fixtures/make_r_fixtures.R and tests/test_effect_sizes.py).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import stats as sps
from scipy.special import gammaln

VarianceMethod = Literal["borenstein", "hedges_olkin_df", "meta_exact", "meta_exact_g", "meta_hedges_approx"]
Estimator = Literal["cohen", "hedges"]


# ----------------------------------------------------------------------------- small-sample constants
def J_exact(m: float) -> float:
    """Exact Hedges small-sample correction J(m) = Γ(m/2) / (√(m/2) Γ((m−1)/2)), m = degrees of freedom."""
    if m <= 1:
        raise ValueError("J requires m > 1 degrees of freedom")
    return math.exp(gammaln(m / 2) - math.log(math.sqrt(m / 2)) - gammaln((m - 1) / 2))


def J_approx(m: float) -> float:
    """Approximate J = 1 − 3/(4m − 1) (Hedges 1981)."""
    return 1 - 3 / (4 * m - 1)


def K_exact(m: float) -> float:
    """meta's K(m) = 1 − (m−2)/(m J(m)²): coefficient of es² in the exact SMD variance."""
    return 1 - (m - 2) / (m * J_exact(m) ** 2)


# ----------------------------------------------------------------------------- core SMD
def pooled_sd(sd1: float, n1: float, sd2: float, n2: float) -> float:
    if n1 + n2 - 2 <= 0:
        raise ValueError("need n1 + n2 > 2 for a pooled SD")
    return math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))


def cohens_d(m1: float, sd1: float, n1: float, m2: float, sd2: float, n2: float) -> float:
    """Cohen's d = (m1 − m2) / pooled SD (identical to meta::metacont(method.smd="Cohen") and metafor
    escalc(measure="SMD", correct=FALSE))."""
    sp = pooled_sd(sd1, n1, sd2, n2)
    if sp <= 0:
        raise ValueError("pooled SD must be > 0")
    return (m1 - m2) / sp


def hedges_g(d: float, n1: float, n2: float, exact: bool = True) -> float:
    """Hedges' g = J(n1+n2−2)·d. exact=True uses the gamma-function J (metafor & meta defaults)."""
    m = n1 + n2 - 2
    return (J_exact(m) if exact else J_approx(m)) * d


def var_smd(es: float, n1: float, n2: float, method: VarianceMethod = "borenstein") -> float:
    """Sampling variance of a standardized mean difference.

    * ``borenstein``      (n1+n2)/(n1 n2) + es²/(2(n1+n2))       — Borenstein 2009 eq. 4.20; metafor escalc SMD
    * ``hedges_olkin_df`` (n1+n2)/(n1 n2) + es²/(2(n1+n2−2))     — common textbook variant (used in the
                                                                     Cisneros notebooks)
    * ``meta_exact``      1/n1 + 1/n2 + (J·d)²·K                 — meta::metacont(method.smd="Cohen", exact.smd=TRUE);
                                                                     pass the raw d
    * ``meta_exact_g``    1/n1 + 1/n2 + g²·K                     — meta::metacont(method.smd="Hedges", exact.smd=TRUE);
                                                                     pass g
    * ``meta_hedges_approx`` 1/n1 + 1/n2 + g²/(2(n1+n2−3.94))    — meta Hedges, exact.smd=FALSE
    """
    N = n1 + n2
    base = 1 / n1 + 1 / n2
    if method == "borenstein":
        return base + es ** 2 / (2 * N)
    if method == "hedges_olkin_df":
        return base + es ** 2 / (2 * (N - 2))
    if method == "meta_exact":
        m = N - 2
        return base + (J_exact(m) * es) ** 2 * K_exact(m)
    if method == "meta_exact_g":
        return base + es ** 2 * K_exact(N - 2)
    if method == "meta_hedges_approx":
        return base + es ** 2 / (2 * (N - 3.94))
    raise ValueError(f"unknown variance method {method!r}")


def se_smd(es: float, n1: float, n2: float, method: VarianceMethod = "borenstein") -> float:
    return math.sqrt(var_smd(es, n1, n2, method))


def ci_smd(es: float, se: float, level: float = 0.95, dist: Literal["z", "t"] = "z", df: float | None = None):
    if dist == "z":
        q = sps.norm.ppf(1 - (1 - level) / 2)
    else:
        if df is None:
            raise ValueError("df required for t-based CI")
        q = sps.t.ppf(1 - (1 - level) / 2, df)
    return es - q * se, es + q * se


# ----------------------------------------------------------------------------- SD conversions
def sd_from_se(se: float, n: float) -> float:
    """SD = SE·√n."""
    return se * math.sqrt(n)


def sd_from_ci(low: float, high: float, n: float, level: float = 0.95, dist: Literal["z", "t"] = "z") -> float:
    """SD from a confidence interval of a group MEAN. dist='t' uses t(n−1) (better for small n, Cochrane 6.5.2.2)."""
    half = (high - low) / 2
    if dist == "z":
        mult = sps.norm.ppf(1 - (1 - level) / 2)
    else:
        mult = sps.t.ppf(1 - (1 - level) / 2, n - 1)
    return half / mult * math.sqrt(n)


def sd_from_ci_halfwidth(halfwidth: float, n: float, level: float = 0.95, dist: Literal["z", "t"] = "z") -> float:
    return sd_from_ci(-halfwidth, halfwidth, n, level, dist)


def sd_from_iqr(q1: float, q3: float, n: float) -> float:
    """Wan et al. (2014) eq. 16: SD ≈ (q3 − q1) / η(n), η(n) = 2Φ⁻¹((0.75n − 0.125)/(n + 0.25))."""
    eta = 2 * sps.norm.ppf((0.75 * n - 0.125) / (n + 0.25))
    return (q3 - q1) / eta


def sd_from_range(low: float, high: float, n: float) -> float:
    """Wan et al. (2014) eq. 9: SD ≈ (max − min) / ξ(n), ξ(n) = 2Φ⁻¹((n − 0.375)/(n + 0.25))."""
    xi = 2 * sps.norm.ppf((n - 0.375) / (n + 0.25))
    return (high - low) / xi


def mean_from_median_iqr(median: float, q1: float, q3: float) -> float:
    """Wan 2014 / Luo 2018 (large-n approx): mean ≈ (q1 + median + q3)/3."""
    return (q1 + median + q3) / 3


def mean_sd_from_points(points: Sequence[float]) -> tuple[float, float, int]:
    """Sample mean, sample SD (ddof=1) and n from individual data points (e.g. digitized dots)."""
    x = np.asarray(points, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 points")
    return float(x.mean()), float(x.std(ddof=1)), int(x.size)


def pooled_sd_from_diff_ci(low: float, high: float, n1: float, n2: float, level: float = 0.95,
                           dist: Literal["z", "t"] = "z") -> float:
    """Pooled SD implied by a CI of the between-group MEAN DIFFERENCE (Cochrane 6.5.2.3)."""
    half = (high - low) / 2
    mult = sps.norm.ppf(1 - (1 - level) / 2) if dist == "z" else sps.t.ppf(1 - (1 - level) / 2, n1 + n2 - 2)
    se_diff = half / mult
    return se_diff / math.sqrt(1 / n1 + 1 / n2)


def pooled_sd_from_diff_se(se_diff: float, n1: float, n2: float) -> float:
    return se_diff / math.sqrt(1 / n1 + 1 / n2)


# ----------------------------------------------------------------------------- from test statistics
def d_from_t(t: float, n1: float, n2: float) -> float:
    """Independent-samples t → d = t·√(1/n1 + 1/n2). Sign of t is (group1 − group2)."""
    return t * math.sqrt(1 / n1 + 1 / n2)


def t_from_p(p: float, df: float, two_tailed: bool = True) -> float:
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")
    return float(sps.t.isf(p / 2, df) if two_tailed else sps.t.isf(p, df))


def d_from_p(p: float, n1: float, n2: float, two_tailed: bool = True, direction: int = 1) -> float:
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (group1 > group2) or −1")
    return direction * d_from_t(t_from_p(p, n1 + n2 - 2, two_tailed), n1, n2)


def d_from_f(F: float, n1: float, n2: float, direction: int) -> float:
    """Between-subjects F(1, df) with two groups: F = t², so d = √(F(n1+n2)/(n1 n2)); direction gives the sign
    (+1 if group1 mean > group2 mean)."""
    if F < 0:
        raise ValueError("F must be ≥ 0")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (group1 > group2) or −1 (group1 < group2)")
    return direction * math.sqrt(F) * math.sqrt(1 / n1 + 1 / n2)


def d_from_r(r: float) -> float:
    """Point-biserial r → d (equal-n approx): d = 2r/√(1−r²)."""
    return 2 * r / math.sqrt(1 - r ** 2)


def orient(d: float, higher_is_better: bool) -> float:
    return d if higher_is_better else -d


# ----------------------------------------------------------------------------- convertibility
class NotConvertible(ValueError):
    """A printed statistic that cannot become a standardised mean difference (amendment C).

    Raised rather than returned, because the alternative — a plausible number from a design that
    does not carry a between-group contrast — is the failure mode this whole layer exists to
    prevent. `canopy.pipeline.resolve` catches it and records the row with `route="not_convertible"`
    and the reason, so the statistic is visible to a reviewer instead of silently dropped.
    """


#: the only designs whose test statistic is a comparison of two independent groups
CONVERTIBLE_DESIGNS: frozenset[str] = frozenset({"independent_t", "one_way_between"})
#: how far the printed degrees of freedom may sit from n_a + n_b − 2 before the statistic is
#: refused (±2 covers a paper that reports df after one exclusion, or rounds a Welch correction)
DF_TOLERANCE = 2.0
#: why each other design cannot stand in for the two group means. The reason given for
#: `mixed_main_effect` used to be "tested against a different error term", which is false: the
#: between-subjects portion of a split-plot IS the one-way ANOVA on the subject means, so
#: F_between(1, N-2) = t^2. The real reason to refuse is AGGREGATION SCOPE — the main effect is
#: computed on scores averaged over every level of the model's within-subject factors, so it
#: answers the outcome's question only when the outcome asks for that same average (C5, P-B).
_DESIGN_REASONS: dict[str, str] = {
    "mixed_main_effect": "the main effect of the grouping factor in a mixed analysis is computed "
                         "on scores averaged over every level of the model's within-subject "
                         "factors, so it estimates the group contrast at that average and not "
                         "necessarily at the window this outcome asks about",
    "interaction": "an interaction term is not a comparison of the two groups",
    "paired": "a within-participant comparison carries no between-group variance",
    "ancova": "a covariate-adjusted statistic is not the raw contrast of the two groups",
    "welch": "unequal-variance degrees of freedom do not match n_a + n_b - 2",
    "unknown": "the paper does not say what kind of test this is",
}

#: the only thing a statistic may contrast if it is to stand in for the two group means (P-B)
CONVERTIBLE_CONTRAST = "groups"
#: why each other contrast cannot, whatever its degrees of freedom
_CONTRAST_REASONS: dict[str, str] = {
    "against_constant": "it tests one group against a constant (zero, chance, a baseline value), "
                        "not group A against group B — its degrees of freedom may still equal "
                        "n_a + n_b - 2 and it is still not this contrast",
    "interaction": "an interaction term is not a comparison of the two groups",
    "within": "a within-subject effect is not a comparison of the two groups",
    "unknown": "nobody recorded what this statistic contrasts, and a statistic is not assumed to "
               "compare the two groups because its design label allows it",
}


def _factor_tokens(name: str) -> frozenset[str]:
    """A factor name as a set of comparable words.

    Extractors write a factor with its levels ("target direction (8 levels)") and the outcome
    side writes the same factor plainly ("target direction", "blocks"). So: drop anything in
    parentheses (that is the level count), drop digits, lower-case, split on non-letters, and take
    the singular of each word. What is left is the factor's NAME, and two names match only when
    they are the same name (see `_same_factor`).
    """
    text = re.sub(r"\([^)]*\)", " ", str(name).lower())
    words = [w for w in re.split(r"[^a-z]+", text) if w]
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


#: "5 levels" / "eight conditions" is a factor's SIZE, not a choice of one of them
_LEVEL_COUNT = re.compile(r"\b\d+\s*(?:levels?|conditions?|values?)\b", re.I)
#: words that pick ONE level of a factor instead of naming the factor
_LEVEL_SELECTOR = re.compile(r"\b(?:last|first|final|only|single|one)\b", re.I)


def _names_one_level(text: str) -> bool:
    """Does this outcome-side entry name one LEVEL of a factor rather than the factor itself?

    "block (last block only)" and "block 5" are the outcome KEEPING a level — the opposite of
    averaging over the factor — and the parenthetical is exactly the part `_factor_tokens` drops,
    so without this the two strings compare equal to "block (5 levels)".
    """
    stripped = _LEVEL_COUNT.sub(" ", str(text))
    return bool(_LEVEL_SELECTOR.search(stripped) or re.search(r"\d", stripped))


def _same_factor(factor: str, averaged_over: str) -> bool:
    """Is `averaged_over` (from the OUTCOME) the same factor as `factor` (from the MODEL)?

    Equality of the normalised name, not containment. The subset test this replaces let one
    verbose outcome string ("the eight target directions in the last block") cover BOTH of a
    model's within factors, and let "the last block" satisfy the gate for "block (5 levels)"
    because it contains the word "block" — which is the statistic C5 exists to refuse. The
    arguments are not interchangeable: only the outcome side is checked for a level selector,
    because only the outcome side is claiming to average over the whole factor.
    """
    tokens_factor, tokens_averaged = _factor_tokens(factor), _factor_tokens(averaged_over)
    if not tokens_factor or not tokens_averaged:
        return False
    if _names_one_level(averaged_over):
        return False
    return tokens_factor == tokens_averaged


def contrast_ok(contrast_kind: str) -> tuple[bool, str]:
    """`(ok, reason)` — P-B's code-side counterpart to the prompt's exclusions.

    The prompt tells the extractor that a test against a constant is not a `value` source. This
    says the same thing where it binds: a statistic whose recorded contrast is anything but
    `groups` is refused, so the rule survives a model that labels it wrongly.
    """
    if contrast_kind == CONVERTIBLE_CONTRAST:
        return True, ""
    reason = _CONTRAST_REASONS.get(contrast_kind,
                                   "it is not recorded as a comparison of the two groups")
    return False, f"contrast {contrast_kind!r} cannot carry this cell: {reason}"


def aggregation_scope_ok(design: str, within_factors: Sequence[str] | None,
                         outcome_averages_over: Sequence[str] | None,
                         error_df: float | None, n_a: float,
                         n_b: float) -> tuple[bool, str, list[str]]:
    """`(ok, reason, flags)` — C5: does this statistic answer the question the outcome asks?

    A main-effect t/F from a model containing within-subject factors estimates the group contrast
    AVERAGED OVER every level of those factors. It may fill a cell only when the outcome's own
    measurement window is that same average, which is what `outcome_averages_over` records.

    Absence of evidence about the model's factors is treated as evidence of risk. Papers write
    "a 2 x 8 ANOVA" without naming the factors, and an extractor that read such a sentence records
    an empty list — indistinguishable from a model that truly had none. So an empty/absent
    `within_factors` opens the route only for the one shape where no within-subject factor can be
    hiding: `design == "independent_t"` AND the error df equal n_a + n_b - 2 EXACTLY.

    `within_factors=None` means this call site records nothing at all about the model (the plain
    arithmetic helpers, called with numbers rather than with an extraction) and the scope check is
    not applied; the pipeline always passes a list.
    """
    if within_factors is None:
        return True, "", []
    named = [f for f in within_factors if str(f).strip()]
    if named:
        averaged = [f for f in (outcome_averages_over or []) if str(f).strip()]
        missing = [f for f in named if not any(_same_factor(f, w) for w in averaged)]
        if missing:
            return False, (f"the statistic comes from a model containing the within-subject "
                           f"factor(s) {', '.join(repr(str(f)) for f in missing)}, so it estimates "
                           f"the group contrast averaged over every level of them; this outcome's "
                           f"measurement window does not average over "
                           f"{'them' if len(missing) > 1 else 'it'}"
                           + (f" (it averages over {', '.join(sorted(str(f) for f in (outcome_averages_over or [])))})"
                              if outcome_averages_over else " (it averages over nothing recorded)")), []
        return True, "", ["aggregation_scope_matched"]
    expected = n_a + n_b - 2
    if design == "independent_t" and error_df is not None and float(error_df) == float(expected):
        return True, "", []
    return False, ("the model's within-subject factors were not recorded, so the statistic's "
                   "estimand is unknown: it may be the group contrast averaged over blocks, "
                   "targets or sessions this outcome does not average over. Only an "
                   "`independent_t` whose error df equal n_a + n_b - 2 exactly "
                   f"(here design {design!r}, df {('none' if error_df is None else format(float(error_df), 'g'))} "
                   f"against {expected:g}) can be trusted without them"), []


def convertibility(design: str, df: float | None, n_a: float, n_b: float, *,
                   contrast_kind: str | None = None,
                   within_factors: Sequence[str] | None = None,
                   outcome_averages_over: Sequence[str] | None = None,
                   df_shortfall_explained: str = "",
                   ) -> tuple[bool, str, list[str]]:
    """`(ok, reason, flags)` — may a statistic from this design and these dfs become an SMD?

    Missing degrees of freedom are allowed but flagged: a paper that prints "t = 5.25, p < .001"
    with two groups of twelve is usually reporting the two-group test, and the flag says the claim
    was never checked against the group sizes.

    `contrast_kind` (P-B) and `within_factors` / `outcome_averages_over` (C5) are the extraction's
    record of WHAT the statistic contrasts and WHAT it was averaged over. Passing `None` for them
    means this call site has no such record — the plain arithmetic helpers — and the corresponding
    check is skipped; every pipeline call passes them, and `"unknown"` / `[]` are refusals, not
    permissions.

    **C9 — the degrees of freedom must EQUAL n_a + n_b - 2, unless the shortfall is explained.**
    A tolerance is not an explanation. `F(1,36)` reported for two groups of twenty is two df short
    of the 38 those groups have, and that is the shape of a two-covariate ANCOVA reported as a
    one-way: the number is a real F, computed on a model these two groups are only part of. The
    old `±2` window admitted it silently, and no arithmetic here can tell the difference. So the
    default is refusal, and `df_shortfall_explained` is the caller's record of WHY a gap is
    admissible — decided once, in `canopy.verify.checks._shortfall_is_explained`, against the
    paper's own reported participant total and whether the group sizes were printed at all. An
    explained gap is still never automatic: it is flagged `df_off_by_<gap>` and
    `confidence.conversion_gate_bucket` caps the row that carries it.
    """
    if contrast_kind is not None:
        ok, reason = contrast_ok(contrast_kind)
        if not ok:
            return False, reason, []
    if design not in CONVERTIBLE_DESIGNS:
        reason = _DESIGN_REASONS.get(design, "it is not a comparison of two independent groups")
        return False, f"design {design!r} cannot carry this contrast: {reason}", []
    scope_ok, scope_reason, scope_flags = aggregation_scope_ok(
        design, within_factors, outcome_averages_over, df, n_a, n_b)
    if not scope_ok:
        return False, scope_reason, []
    expected = n_a + n_b - 2
    if df is None:
        return True, "", [*scope_flags, "df_missing"]
    gap = abs(float(df) - expected)
    if gap and not df_shortfall_explained:
        return False, (f"the printed degrees of freedom ({float(df):g}) do not equal "
                       f"n_a + n_b - 2 = {expected:g} and nothing on the record explains the "
                       f"shortfall, so this statistic was not computed on these two groups as "
                       f"analysed"), []
    if gap > DF_TOLERANCE:
        return False, (f"the printed degrees of freedom ({float(df):g}) are {gap:g} from "
                       f"n_a + n_b - 2 = {expected:g}, further than any explanation covers "
                       f"({df_shortfall_explained})"), []
    return True, "", [*scope_flags, *([f"df_off_by_{gap:g}"] if gap else [])]


def _gate(design: str, df: float | None, n_a: float, n_b: float, what: str, *,
          contrast_kind: str | None = None, within_factors: Sequence[str] | None = None,
          outcome_averages_over: Sequence[str] | None = None,
          df_shortfall_explained: str = "") -> list[str]:
    ok, reason, flags = convertibility(design, df, n_a, n_b, contrast_kind=contrast_kind,
                                       within_factors=within_factors,
                                       outcome_averages_over=outcome_averages_over,
                                       df_shortfall_explained=df_shortfall_explained)
    if not ok:
        raise NotConvertible(f"{what}: {reason}")
    return flags


# ----------------------------------------------------------------------------- one-call API
@dataclass
class SMDResult:
    d: float                       # oriented Cohen's d (positive = A more of the construct than B)
    g: float                       # oriented Hedges' g (exact J)
    es: float                      # the effect size actually used downstream (d or g per `estimator`)
    se: float
    var: float
    ci_low: float
    ci_high: float
    n_a: float
    n_b: float
    route: str                     # means_sd | t_stat | f_stat | p_value | reported_d | ...
    estimator: str                 # cohen | hedges
    variance_method: str
    level: float = 0.95
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _finish(d_raw: float, n_a: float, n_b: float, higher_is_better: bool, estimator: Estimator,
            variance: VarianceMethod, level: float, route: str, details: dict,
            flags: Sequence[str] = ()) -> SMDResult:
    d = orient(d_raw, higher_is_better)
    g = hedges_g(d, n_a, n_b, exact=True)
    es = d if estimator == "cohen" else g
    var = var_smd(es, n_a, n_b, variance)
    se = math.sqrt(var)
    lo, hi = ci_smd(es, se, level)
    details = {**details, "flags": list(flags)}
    return SMDResult(d=d, g=g, es=es, se=se, var=var, ci_low=lo, ci_high=hi, n_a=n_a, n_b=n_b, route=route,
                     estimator=estimator, variance_method=variance, level=level, details=details)


def smd_from_means(m_a: float, sd_a: float, n_a: float, m_b: float, sd_b: float, n_b: float, *,
                   higher_is_better: bool = True, estimator: Estimator = "cohen",
                   variance: VarianceMethod = "borenstein", level: float = 0.95) -> SMDResult:
    d_raw = cohens_d(m_a, sd_a, n_a, m_b, sd_b, n_b)
    return _finish(d_raw, n_a, n_b, higher_is_better, estimator, variance, level, "means_sd",
                   dict(m_a=m_a, sd_a=sd_a, m_b=m_b, sd_b=sd_b, pooled_sd=pooled_sd(sd_a, n_a, sd_b, n_b)))


def smd_from_t(t: float, n_a: float, n_b: float, *, df: float | None = None,
               design: str = "unknown", positive_means_a_greater: bool = True,
               higher_is_better: bool = True, estimator: Estimator = "cohen",
               variance: VarianceMethod = "borenstein", level: float = 0.95,
               contrast_kind: str | None = None, within_factors: Sequence[str] | None = None,
               outcome_averages_over: Sequence[str] | None = None,
               df_shortfall_explained: str = "") -> SMDResult:
    """d from an independent-samples t (amendment C: gated by `design` and `df`).

    ``positive_means_a_greater`` states the paper's sign convention. Raises `NotConvertible` when
    the design cannot carry a between-group contrast, when what it contrasts is not the two groups
    (P-B), when its estimand is an average over factors this outcome does not average over (C5),
    or when the printed degrees of freedom do not match the analysed group sizes.
    """
    flags = _gate(design, df, n_a, n_b, f"t = {t:g}", contrast_kind=contrast_kind,
                  within_factors=within_factors, outcome_averages_over=outcome_averages_over,
                  df_shortfall_explained=df_shortfall_explained)
    t_ab = t if positive_means_a_greater else -t
    return _finish(d_from_t(t_ab, n_a, n_b), n_a, n_b, higher_is_better, estimator, variance, level, "t_stat",
                   dict(t=t, df=df, design=design,
                        positive_means_a_greater=positive_means_a_greater), flags)


def smd_from_f(F: float, n_a: float, n_b: float, *, a_greater: bool, df1: float | None = None,
               df2: float | None = None, design: str = "unknown", higher_is_better: bool = True,
               estimator: Estimator = "cohen", variance: VarianceMethod = "borenstein",
               level: float = 0.95, contrast_kind: str | None = None,
               within_factors: Sequence[str] | None = None,
               outcome_averages_over: Sequence[str] | None = None,
               df_shortfall_explained: str = "") -> SMDResult:
    """d from a between-subjects F(1, df) (amendment C: gated). ``a_greater`` = A's raw mean is higher.

    An F with more than one numerator degree of freedom compares more than two groups, so it is
    refused before the design is even considered; the error degrees of freedom `df2` are then
    checked against n_a + n_b - 2 like a t.
    """
    if df1 is not None and float(df1) != 1.0:
        raise NotConvertible(
            f"F = {F:g}: df1 = {float(df1):g} means the test compares more than two groups, so it "
            f"is not the contrast of this pair")
    flags = _gate(design, df2, n_a, n_b, f"F = {F:g}", contrast_kind=contrast_kind,
                  within_factors=within_factors, outcome_averages_over=outcome_averages_over,
                  df_shortfall_explained=df_shortfall_explained)
    return _finish(d_from_f(F, n_a, n_b, 1 if a_greater else -1), n_a, n_b, higher_is_better, estimator, variance,
                   level, "f_stat", dict(F=F, df1=df1, df2=df2, design=design,
                                         a_greater=a_greater), flags)


def smd_from_p(p: float, n_a: float, n_b: float, *, a_greater: bool, df: float | None = None,
               design: str = "unknown", two_tailed: bool = True, higher_is_better: bool = True,
               estimator: Estimator = "cohen", variance: VarianceMethod = "borenstein",
               level: float = 0.95, contrast_kind: str | None = None,
               within_factors: Sequence[str] | None = None,
               outcome_averages_over: Sequence[str] | None = None,
               df_shortfall_explained: str = "") -> SMDResult:
    """d from an exact p value (amendment C: gated exactly like the t it is inverted from)."""
    flags = _gate(design, df, n_a, n_b, f"p = {p:g}", contrast_kind=contrast_kind,
                  within_factors=within_factors, outcome_averages_over=outcome_averages_over,
                  df_shortfall_explained=df_shortfall_explained)
    d_raw = d_from_p(p, n_a, n_b, two_tailed, 1 if a_greater else -1)
    return _finish(d_raw, n_a, n_b, higher_is_better, estimator, variance, level, "p_value",
                   dict(p=p, df=df, design=design, two_tailed=two_tailed,
                        a_greater=a_greater), flags)


def smd_from_reported(d_reported: float, n_a: float, n_b: float, *, positive_means_a_greater: bool = True,
                      higher_is_better: bool = True, is_hedges_g: bool = False, estimator: Estimator = "cohen",
                      variance: VarianceMethod = "borenstein", level: float = 0.95,
                      contrast_kind: str | None = None) -> SMDResult:
    """Use an effect size the paper itself reports (Cohen's d or Hedges' g for A vs B).

    `contrast_kind` is what the printed effect size CONTRASTS (P-B). A printed d has no test
    statistic behind it, but it always has a contrast: "the aftereffect differed from zero,
    d = 1.30" is a one-sample effect, and pooling it as the between-group difference is a wrong
    number, not an imprecise one. `None` means this call site records nothing — the plain
    arithmetic helpers — and the check is skipped, exactly as it is for `convertibility`; the
    pipeline always passes it, and `"unknown"` is a refusal.
    """
    if contrast_kind is not None:
        ok, reason = contrast_ok(contrast_kind)
        if not ok:
            raise NotConvertible(f"the reported effect size cannot carry this cell: {reason}")
    d_raw = d_reported if positive_means_a_greater else -d_reported
    if is_hedges_g:  # back out d so `d`/`g` fields are consistent
        d_raw = d_raw / J_exact(n_a + n_b - 2)
    return _finish(d_raw, n_a, n_b, higher_is_better, estimator, variance, level, "reported_d",
                   dict(d_reported=d_reported, is_hedges_g=is_hedges_g))
