"""Amendment C: the conversions a review needs when a paper does not print mean ± SD.

Four jobs, all pure arithmetic with a citation attached:

* `combine_groups` — Cochrane 6.5.2.10. Two arms of one study collapsed into the single arm the
  contrast needs. The formula is *exact*: fed the summaries of two halves of a sample it returns
  the whole sample's mean and SD to the last digit (`tests/test_stats_conversions.py` checks that
  against a raw sample generated in R).
* `split_control` — Cochrane 16.5.4. One control arm compared with k intervention arms would enter
  a meta-analysis k times; dividing its n by k keeps the rows nearly independent.
* `mean_sd_from_median_iqr` / `mean_sd_from_five_number` — Luo et al. (2018) for the mean, Shi et
  al. (2020) for the SD, with Wan et al. (2014) available as the fallback the older literature
  uses. Papers that report a median and quartiles are otherwise simply lost.
* `partial_variance` — the delta method by central finite differences, so an uncertainty attached
  to a digitised input (a mean read off a bar, a cap read off an error bar) can be propagated into
  the variance of an effect size without anyone deriving ∂d/∂x by hand for each route.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping

from scipy import stats as sps

__all__ = ["combine_groups", "split_control", "mean_sd_from_median_iqr",
           "mean_sd_from_five_number", "partial_variance", "xi_range", "eta_iqr"]


# ----------------------------------------------------------------------------- combining arms
def combine_groups(m1: float, sd1: float, n1: float, m2: float, sd2: float,
                   n2: float) -> tuple[float, float, int]:
    """Two arms into one (Cochrane Handbook 6.5.2.10): `(mean, sd, n)`.

    SD² = [ (n1−1)sd1² + (n2−1)sd2² + n1 n2 (m1−m2)² / N ] / (N−1), which is the variance of the
    pooled sample — the spread *between* the arms is part of the combined spread, so this is not
    a pooled SD and must never be used as one.
    """
    if n1 < 1 or n2 < 1:
        raise ValueError("each arm needs at least one participant")
    n = n1 + n2
    if n < 2:
        raise ValueError("a combined arm needs at least two participants")
    mean = (n1 * m1 + n2 * m2) / n
    ss = (n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2 + n1 * n2 * (m1 - m2) ** 2 / n
    return float(mean), float(math.sqrt(ss / (n - 1))), int(n)


def split_control(n: float, k: int) -> float:
    """A control arm shared by `k` comparisons contributes n/k to each (Cochrane 16.5.4).

    It is an approximation — the rows stay correlated — but it stops one control group from being
    counted k times, which is the error that actually moves a pooled estimate.
    """
    if k < 1:
        raise ValueError("a shared control is shared by at least one comparison")
    return float(n) / k


# ----------------------------------------------------------------------------- five-number summaries
def xi_range(n: float) -> float:
    """Wan et al. (2014) eq. 9: ξ(n) = 2Φ⁻¹((n − 0.375)/(n + 0.25)), the range → SD divisor."""
    return 2 * sps.norm.ppf((n - 0.375) / (n + 0.25))


def eta_iqr(n: float) -> float:
    """Wan et al. (2014) eq. 16: η(n) = 2Φ⁻¹((0.75n − 0.125)/(n + 0.25)), the IQR → SD divisor."""
    return 2 * sps.norm.ppf((0.75 * n - 0.125) / (n + 0.25))


def _check_order(*values: float) -> None:
    if list(values) != sorted(values):
        raise ValueError(f"a five-number summary must be ordered, got {values}")


def mean_sd_from_median_iqr(median: float, q1: float, q3: float,
                            n: float) -> tuple[float, float]:
    """`(mean, sd)` from a median and the two quartiles (Luo 2018 S2 mean, Wan/Shi IQR SD).

    Luo et al. (2018) eq. (15): mean ≈ (0.7 + 0.39/n)·(q1+q3)/2 + (0.3 − 0.39/n)·median — the
    weights shift towards the mid-quartile range as n grows. Shi et al. (2020) keep Wan's η(n) for
    the SD when only the quartiles are known.
    """
    _check_order(q1, median, q3)
    if n < 1:
        raise ValueError("n must be at least 1")
    weight = 0.7 + 0.39 / n
    mean = weight * (q1 + q3) / 2 + (1 - weight) * median
    return float(mean), float((q3 - q1) / eta_iqr(n))


def mean_sd_from_five_number(minimum: float, q1: float, median: float, q3: float, maximum: float,
                             n: float, method: str = "shi") -> tuple[float, float]:
    """`(mean, sd)` from min, q1, median, q3, max (Luo 2018 S3 mean; Shi 2020 or Wan 2014 SD).

    * mean — Luo et al. (2018) eq. (20).
    * sd, `method="shi"` — Shi et al. (2020) eq. (10): the range and IQR estimators combined with
      the weight 1/(1 + 0.07 n^0.6) on the range term, which is optimal rather than Wan's 1/2.
    * sd, `method="wan"` — Wan et al. (2014) eq. (10): the unweighted average of the same two.
    """
    _check_order(minimum, q1, median, q3, maximum)
    if n < 1:
        raise ValueError("n must be at least 1")
    root = n ** 0.75
    w_range = 2.2 / (2.2 + root)
    w_quart = 0.7 - 0.72 / root
    mean = (w_range * (minimum + maximum) / 2 + w_quart * (q1 + q3) / 2
            + (1 - w_range - w_quart) * median)

    from_range = (maximum - minimum) / xi_range(n)
    from_iqr = (q3 - q1) / eta_iqr(n)
    if method == "wan":
        sd = 0.5 * from_range + 0.5 * from_iqr
    elif method == "shi":
        weight = 1 / (1 + 0.07 * n ** 0.6)
        sd = weight * from_range + (1 - weight) * from_iqr
    else:
        raise ValueError(f"unknown method {method!r} (have 'shi', 'wan')")
    return float(mean), float(sd)


# ----------------------------------------------------------------------------- delta method
def partial_variance(func: Callable[..., float], values: Mapping[str, float],
                     sigmas: Mapping[str, float | None],
                     rel_step: float = 1e-5) -> tuple[float, dict[str, float]]:
    """Σ (∂f/∂x_i · σ_i)² by central finite differences — the delta method, numerically.

    `func` is called with `**values`; only the inputs that have a positive σ are perturbed, so an
    exactly-known input costs nothing. Returns `(variance, per-input contributions)` so a record
    can say which digitised quantity dominates its own uncertainty.
    """
    parts: dict[str, float] = {}
    for name, sigma in sigmas.items():
        if sigma is None or sigma <= 0 or name not in values:
            continue
        base = float(values[name])
        step = max(abs(base) * rel_step, min(abs(sigma) * rel_step, 1e-6), 1e-9)
        high = dict(values); high[name] = base + step
        low = dict(values); low[name] = base - step
        derivative = (func(**high) - func(**low)) / (2 * step)
        parts[name] = float((derivative * sigma) ** 2)
    return float(math.fsum(parts.values())), parts
