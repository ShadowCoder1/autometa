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
#: why each other design cannot stand in for the two group means
_DESIGN_REASONS: dict[str, str] = {
    "mixed_main_effect": "the grouping factor in a mixed analysis is tested against a different "
                         "error term than a two-group comparison",
    "interaction": "an interaction term is not a comparison of the two groups",
    "paired": "a within-participant comparison carries no between-group variance",
    "ancova": "a covariate-adjusted statistic is not the raw contrast of the two groups",
    "welch": "unequal-variance degrees of freedom do not match n_a + n_b - 2",
    "unknown": "the paper does not say what kind of test this is",
}


def convertibility(design: str, df: float | None, n_a: float,
                   n_b: float) -> tuple[bool, str, list[str]]:
    """`(ok, reason, flags)` — may a statistic from this design and these dfs become an SMD?

    Missing degrees of freedom are allowed but flagged: a paper that prints "t = 5.25, p < .001"
    with two groups of twelve is usually reporting the two-group test, and the flag says the claim
    was never checked against the group sizes.
    """
    if design not in CONVERTIBLE_DESIGNS:
        reason = _DESIGN_REASONS.get(design, "it is not a comparison of two independent groups")
        return False, f"design {design!r} cannot carry this contrast: {reason}", []
    expected = n_a + n_b - 2
    if df is None:
        return True, "", ["df_missing"]
    gap = abs(float(df) - expected)
    if gap > DF_TOLERANCE:
        return False, (f"the printed degrees of freedom ({float(df):g}) do not match "
                       f"n_a + n_b - 2 = {expected:g}, so this statistic was not computed on "
                       f"these two groups"), []
    return True, "", ([f"df_off_by_{gap:g}"] if gap else [])


def _gate(design: str, df: float | None, n_a: float, n_b: float, what: str) -> list[str]:
    ok, reason, flags = convertibility(design, df, n_a, n_b)
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
               variance: VarianceMethod = "borenstein", level: float = 0.95) -> SMDResult:
    """d from an independent-samples t (amendment C: gated by `design` and `df`).

    ``positive_means_a_greater`` states the paper's sign convention. Raises `NotConvertible` when
    the design cannot carry a between-group contrast, or when the printed degrees of freedom do
    not match the analysed group sizes.
    """
    flags = _gate(design, df, n_a, n_b, f"t = {t:g}")
    t_ab = t if positive_means_a_greater else -t
    return _finish(d_from_t(t_ab, n_a, n_b), n_a, n_b, higher_is_better, estimator, variance, level, "t_stat",
                   dict(t=t, df=df, design=design,
                        positive_means_a_greater=positive_means_a_greater), flags)


def smd_from_f(F: float, n_a: float, n_b: float, *, a_greater: bool, df1: float | None = None,
               df2: float | None = None, design: str = "unknown", higher_is_better: bool = True,
               estimator: Estimator = "cohen", variance: VarianceMethod = "borenstein",
               level: float = 0.95) -> SMDResult:
    """d from a between-subjects F(1, df) (amendment C: gated). ``a_greater`` = A's raw mean is higher.

    An F with more than one numerator degree of freedom compares more than two groups, so it is
    refused before the design is even considered; the error degrees of freedom `df2` are then
    checked against n_a + n_b - 2 like a t.
    """
    if df1 is not None and float(df1) != 1.0:
        raise NotConvertible(
            f"F = {F:g}: df1 = {float(df1):g} means the test compares more than two groups, so it "
            f"is not the contrast of this pair")
    flags = _gate(design, df2, n_a, n_b, f"F = {F:g}")
    return _finish(d_from_f(F, n_a, n_b, 1 if a_greater else -1), n_a, n_b, higher_is_better, estimator, variance,
                   level, "f_stat", dict(F=F, df1=df1, df2=df2, design=design,
                                         a_greater=a_greater), flags)


def smd_from_p(p: float, n_a: float, n_b: float, *, a_greater: bool, df: float | None = None,
               design: str = "unknown", two_tailed: bool = True, higher_is_better: bool = True,
               estimator: Estimator = "cohen", variance: VarianceMethod = "borenstein",
               level: float = 0.95) -> SMDResult:
    """d from an exact p value (amendment C: gated exactly like the t it is inverted from)."""
    flags = _gate(design, df, n_a, n_b, f"p = {p:g}")
    d_raw = d_from_p(p, n_a, n_b, two_tailed, 1 if a_greater else -1)
    return _finish(d_raw, n_a, n_b, higher_is_better, estimator, variance, level, "p_value",
                   dict(p=p, df=df, design=design, two_tailed=two_tailed,
                        a_greater=a_greater), flags)


def smd_from_reported(d_reported: float, n_a: float, n_b: float, *, positive_means_a_greater: bool = True,
                      higher_is_better: bool = True, is_hedges_g: bool = False, estimator: Estimator = "cohen",
                      variance: VarianceMethod = "borenstein", level: float = 0.95) -> SMDResult:
    """Use an effect size the paper itself reports (Cohen's d or Hedges' g for A vs B)."""
    d_raw = d_reported if positive_means_a_greater else -d_reported
    if is_hedges_g:  # back out d so `d`/`g` fields are consistent
        d_raw = d_raw / J_exact(n_a + n_b - 2)
    return _finish(d_raw, n_a, n_b, higher_is_better, estimator, variance, level, "reported_d",
                   dict(d_reported=d_reported, is_hedges_g=is_hedges_g))
