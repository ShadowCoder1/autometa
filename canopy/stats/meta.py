"""Random-/fixed-effects meta-analysis for a vector of effect sizes with sampling variances.

Reproduces R `metafor::rma(yi, vi, method=...)` and `meta::metagen(TE, seTE, ...)` (see tests/test_meta.py):
* tau² estimators: REML (Fisher scoring with step-halving, as in metafor), DL, PM (Paule–Mandel), FE.
* I² is reported two ways: ``I2`` = (Q − df)/Q as printed by `meta` (what forest plots show), and
  ``I2_tau`` = tau²/(tau² + s²) as printed by `metafor` for non-DL estimators.
* Prediction interval: ``pi_low/pi_high`` use t(k−2) on sqrt(se² + tau²) (meta's default "HTS",
  Higgins–Thompson–Spiegelhalter 2009); ``pi_low_z/pi_high_z`` are metafor's z-based defaults.
* Hartung–Knapp (hakn=True): variance = Σw(y−μ)² / ((k−1)Σw), t(k−1) CI, identical in meta and metafor.
* Q-profile confidence interval for tau² (Viechtbauer 2007), as in metafor::confint.
* Egger regression exactly as the Cisneros Rmd computed it: lm(TE/seTE ~ 1/seTE), t-test on the intercept.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from scipy import optimize, stats as sps

Tau2Method = Literal["REML", "DL", "PM", "FE", "ML"]


@dataclass
class MetaResult:
    k: int
    method: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    z: float                 # z (or t if hakn) statistic
    p: float
    tau2: float
    tau: float
    se_tau2: float | None
    Q: float
    Q_df: int
    Q_p: float
    I2: float                # meta convention (Q-based), proportion 0..1
    I2_tau: float            # metafor convention (tau²-based), proportion 0..1
    H2: float
    weights: np.ndarray      # normalized RE weights (sum 1)
    weights_pct: np.ndarray  # RE weights in %
    weights_raw: np.ndarray  # 1/(vi + tau²)
    pi_low: float            # prediction interval, t(k-2) (meta HTS)
    pi_high: float
    pi_df: int
    pi_low_z: float          # prediction interval, z-based (metafor default)
    pi_high_z: float
    hakn: bool = False
    level: float = 0.95
    tau2_ci_low: float | None = None
    tau2_ci_high: float | None = None
    iterations: int = 0
    converged: bool = True
    yi: np.ndarray = field(default_factory=lambda: np.array([]))
    vi: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ("weights", "weights_pct", "weights_raw", "yi", "vi"):
            d[key] = [float(x) for x in np.asarray(d[key]).ravel()]
        return d


@dataclass
class FixedResult:
    k: int
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    z: float
    p: float
    Q: float
    Q_df: int
    Q_p: float
    weights_pct: np.ndarray


@dataclass
class EggerResult:
    intercept: float
    intercept_se: float
    t: float
    p: float
    slope: float
    k: int


# ----------------------------------------------------------------------------- helpers
def _check(yi, vi):
    yi = np.asarray(yi, dtype=float).ravel()
    vi = np.asarray(vi, dtype=float).ravel()
    if yi.shape != vi.shape:
        raise ValueError("yi and vi must have the same length")
    if yi.size < 2:
        raise ValueError("need at least k=2 studies")
    if np.any(vi <= 0) or np.any(~np.isfinite(vi)) or np.any(~np.isfinite(yi)):
        raise ValueError("vi must be finite and > 0; yi finite")
    return yi, vi


def cochran_Q(yi, vi):
    w = 1 / vi
    mu = np.sum(w * yi) / np.sum(w)
    return float(np.sum(w * (yi - mu) ** 2)), int(yi.size - 1)


def _Q_gen(tau2, yi, vi):
    w = 1 / (vi + tau2)
    mu = np.sum(w * yi) / np.sum(w)
    return float(np.sum(w * (yi - mu) ** 2))


def tau2_DL(yi, vi) -> float:
    Q, df = cochran_Q(yi, vi)
    w = 1 / vi
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    return float(max(0.0, (Q - df) / C))


def tau2_PM(yi, vi, tol: float = 1e-10) -> float:
    k = yi.size
    if _Q_gen(0.0, yi, vi) <= k - 1:
        return 0.0
    hi = 1.0
    while _Q_gen(hi, yi, vi) > k - 1:
        hi *= 4
        if hi > 1e12:
            return float(hi)
    return float(optimize.brentq(lambda t: _Q_gen(t, yi, vi) - (k - 1), 0.0, hi, xtol=tol))


def tau2_REML(yi, vi, tau2_init: float | None = None, tol: float = 1e-8, max_iter: int = 200,
              step_adj: float = 1.0, ml: bool = False):
    """Fisher-scoring REML (or ML) exactly following metafor::rma.uni's iterative scheme (intercept-only)."""
    k = yi.size
    tau2 = tau2_DL(yi, vi) if tau2_init is None else float(tau2_init)
    tau2 = max(tau2, 0.0)
    conv = False
    it = 0
    for it in range(1, max_iter + 1):
        w = 1 / (vi + tau2)
        sw = np.sum(w)
        # P = W - w w'/sum(w) ; (Py)_i = w_i (y_i - μ) so y'P²y = Σ w² (y - μ)² ;
        # tr(P) = Σw - Σw²/Σw ; tr(P²) = Σw² - 2Σw³/Σw + (Σw²)²/(Σw)²   (metafor rma.uni Fisher scoring)
        mu = np.sum(w * yi) / sw
        yPPy = np.sum(w ** 2 * (yi - mu) ** 2)
        if ml:
            adj = (yPPy - sw) / np.sum(w ** 2)
        else:
            trP = sw - np.sum(w ** 2) / sw
            trPP = np.sum(w ** 2) - 2 * np.sum(w ** 3) / sw + (np.sum(w ** 2) ** 2) / sw ** 2
            adj = (yPPy - trP) / trPP
        adj *= step_adj
        # step-halving to keep tau2 >= 0 (metafor: while tau2 + adj < tau2.min: adj /= 2)
        while tau2 + adj < 0:
            adj /= 2
        new = tau2 + adj
        change = abs(new - tau2)
        tau2 = new
        if change < tol:
            conv = True
            break
    # standard error of tau2 (metafor: se.tau2 = sqrt(2 / sum(diag(P%*%P))) for REML)
    w = 1 / (vi + tau2)
    sw = np.sum(w)
    trPP = np.sum(w ** 2) - 2 * np.sum(w ** 3) / sw + (np.sum(w ** 2) ** 2) / sw ** 2
    se_tau2 = float(np.sqrt(2 / trPP)) if trPP > 0 else None
    return float(tau2), se_tau2, it, conv


def tau2_qprofile_ci(yi, vi, level: float = 0.95, upper: float = 1e5):
    """Q-profile CI for tau² (Viechtbauer 2007), as metafor::confint(rma) does."""
    k = yi.size
    lo_q = sps.chi2.ppf(1 - (1 - level) / 2, k - 1)   # Q_gen(tau2) = chi2_{0.975} → lower bound
    hi_q = sps.chi2.ppf((1 - level) / 2, k - 1)       # Q_gen(tau2) = chi2_{0.025} → upper bound
    q0 = _Q_gen(0.0, yi, vi)
    if q0 <= lo_q:
        lb = 0.0
    else:
        lb = float(optimize.brentq(lambda t: _Q_gen(t, yi, vi) - lo_q, 0.0, upper))
    if q0 <= hi_q:
        ub = 0.0
    else:
        hi = 1.0
        while _Q_gen(hi, yi, vi) > hi_q and hi < upper:
            hi *= 4
        ub = float(optimize.brentq(lambda t: _Q_gen(t, yi, vi) - hi_q, 0.0, min(hi, upper))) if hi < upper else float("inf")
    return lb, ub


# ----------------------------------------------------------------------------- public API
def fixed_effects(yi, vi, level: float = 0.95) -> FixedResult:
    yi, vi = _check(yi, vi)
    w = 1 / vi
    mu = float(np.sum(w * yi) / np.sum(w))
    se = float(np.sqrt(1 / np.sum(w)))
    zq = sps.norm.ppf(1 - (1 - level) / 2)
    Q, df = cochran_Q(yi, vi)
    z = mu / se
    return FixedResult(k=yi.size, estimate=mu, se=se, ci_low=mu - zq * se, ci_high=mu + zq * se, z=z,
                       p=float(2 * sps.norm.sf(abs(z))), Q=Q, Q_df=df, Q_p=float(sps.chi2.sf(Q, df)),
                       weights_pct=100 * w / np.sum(w))


def random_effects(yi, vi, method: Tau2Method = "REML", hakn: bool = False, level: float = 0.95,
                   tau2_ci: bool = False) -> MetaResult:
    yi, vi = _check(yi, vi)
    k = yi.size
    se_tau2 = None
    iters, conv = 0, True
    if method == "DL":
        tau2 = tau2_DL(yi, vi)
    elif method == "PM":
        tau2 = tau2_PM(yi, vi)
    elif method == "FE":
        tau2 = 0.0
    elif method in ("REML", "ML"):
        tau2, se_tau2, iters, conv = tau2_REML(yi, vi, ml=(method == "ML"))
    else:
        raise ValueError(f"unknown method {method!r}")

    w = 1 / (vi + tau2)
    sw = np.sum(w)
    mu = float(np.sum(w * yi) / sw)
    se = float(np.sqrt(1 / sw))
    Q, df = cochran_Q(yi, vi)
    Q_p = float(sps.chi2.sf(Q, df))
    # heterogeneity summaries
    I2_meta = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    wf = 1 / vi
    s2 = (k - 1) * np.sum(wf) / (np.sum(wf) ** 2 - np.sum(wf ** 2))  # "typical" within-study variance
    I2_tau = float(tau2 / (tau2 + s2)) if (tau2 + s2) > 0 else 0.0
    H2 = float((tau2 + s2) / s2) if s2 > 0 else float("nan")

    if hakn:
        # Hartung–Knapp: var = Σ w (y-μ)² / ((k−1) Σ w); t(k−1)
        var_hk = float(np.sum(w * (yi - mu) ** 2) / ((k - 1) * sw))
        se_used = float(np.sqrt(var_hk))
        q = sps.t.ppf(1 - (1 - level) / 2, k - 1)
        stat = mu / se_used
        p = float(2 * sps.t.sf(abs(stat), k - 1))
    else:
        se_used = se
        q = sps.norm.ppf(1 - (1 - level) / 2)
        stat = mu / se_used
        p = float(2 * sps.norm.sf(abs(stat)))
    ci_low, ci_high = mu - q * se_used, mu + q * se_used

    # prediction intervals
    pi_sd = float(np.sqrt(se_used ** 2 + tau2))
    if k >= 3:
        tq = sps.t.ppf(1 - (1 - level) / 2, k - 2)
        pi_low, pi_high = mu - tq * pi_sd, mu + tq * pi_sd
    else:
        pi_low = pi_high = float("nan")
    zq = sps.norm.ppf(1 - (1 - level) / 2)
    pi_low_z, pi_high_z = mu - zq * pi_sd, mu + zq * pi_sd

    lb = ub = None
    if tau2_ci:
        lb, ub = tau2_qprofile_ci(yi, vi, level)

    return MetaResult(k=k, method=method, estimate=mu, se=se_used, ci_low=float(ci_low), ci_high=float(ci_high),
                      z=float(stat), p=p, tau2=float(tau2), tau=float(np.sqrt(tau2)), se_tau2=se_tau2, Q=Q, Q_df=df,
                      Q_p=Q_p, I2=I2_meta, I2_tau=I2_tau, H2=H2, weights=w / sw, weights_pct=100 * w / sw,
                      weights_raw=w, pi_low=float(pi_low), pi_high=float(pi_high), pi_df=k - 2,
                      pi_low_z=float(pi_low_z), pi_high_z=float(pi_high_z), hakn=hakn, level=level,
                      tau2_ci_low=lb, tau2_ci_high=ub, iterations=iters, converged=conv, yi=yi, vi=vi)


def egger_test(yi, sei) -> EggerResult:
    """Egger's regression test as OLS of (TE/seTE) on (1/seTE): the intercept tests for small-study effects."""
    yi = np.asarray(yi, float).ravel()
    sei = np.asarray(sei, float).ravel()
    y = yi / sei
    x = 1 / sei
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    k = yi.size
    sigma2 = float(resid @ resid / (k - 2))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_int = float(np.sqrt(cov[0, 0]))
    t = float(beta[0] / se_int)
    p = float(2 * sps.t.sf(abs(t), k - 2))
    return EggerResult(intercept=float(beta[0]), intercept_se=se_int, t=t, p=p, slope=float(beta[1]), k=k)


def per_study_ci(yi, vi, level: float = 0.95):
    yi = np.asarray(yi, float); se = np.sqrt(np.asarray(vi, float))
    q = sps.norm.ppf(1 - (1 - level) / 2)
    return yi - q * se, yi + q * se
