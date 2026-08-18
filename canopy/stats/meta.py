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
    pi_low_z: float          # prediction interval, z-based (metafor default; meta method.predict="S")
    pi_high_z: float
    pi_low_v: float = float("nan")   # prediction interval, t(k-1) (meta ≥7 default method.predict="V", Veroniki 2019)
    pi_high_v: float = float("nan")
    hakn: bool = False
    level: float = 0.95
    tau2_ci_low: float | None = None
    tau2_ci_high: float | None = None
    iterations: int = 0
    converged: bool = True
    yi: np.ndarray = field(default_factory=lambda: np.array([]))
    vi: np.ndarray = field(default_factory=lambda: np.array([]))
    #: non-empty when the Hartung-Knapp adjustment could not be applied and the ordinary
    #: random-effects standard error was used instead. Empty on every ordinary pool.
    hakn_fallback: str = ""

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
    """Egger's regression test. `intercept` is the BIAS coefficient in both variants.

    Classic (`predictor="precision"`): OLS of TE/seTE on 1/seTE — the fitted intercept is the bias
    term and the fitted slope is the effect a study of infinite precision would show. Modified
    (`predictor="sqrt_inv_n"`, Pustejovsky & Rodgers 2019): weighted least squares of TE on
    √(1/n_a + 1/n_b) with weights 1/vi, so the predictor does not contain the effect estimate that
    is being tested; there the bias term is the slope and `estimate` is the fitted intercept.
    Either way `intercept`/`t`/`p` test asymmetry and `estimate` is the limit estimate.
    """

    #: THE ASYMMETRY COEFFICIENT, whichever variant produced it — the quantity the test is about.
    #: Classic: the fitted intercept of TE/seTE on 1/seTE. Pustejovsky-Rodgers: the fitted slope on
    #: √(1/n_a + 1/n_b). `t` and `p` always test THIS number against zero.
    intercept: float
    intercept_se: float
    t: float
    p: float
    #: the other coefficient of the same fit: the effect a study of infinite precision would show
    slope: float
    k: int
    predictor: str = "precision"           # precision | sqrt_inv_n
    df: int = 0
    #: the limit estimate — identical to `slope`, under the name a report should print
    estimate: float = float("nan")
    estimate_ci_low: float = float("nan")
    estimate_ci_high: float = float("nan")

    @property
    def bias_coefficient(self) -> float:
        """`intercept` under the name that says what it is, whichever predictor was used."""
        return self.intercept

    @property
    def limit_estimate(self) -> float:
        """`estimate` under the name that says what it is: the effect at zero standard error."""
        return self.estimate


@dataclass
class LeaveOneOut:
    """One row of a leave-one-out analysis: the pooled result without study `omitted`."""

    omitted: int
    label: str
    k: int
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    tau2: float
    I2: float                # metafor convention (tau²-based), proportion 0..1
    Q: float
    Q_p: float


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

    hakn_fallback = ""
    if hakn:
        # Hartung–Knapp: var = Σ w (y-μ)² / ((k−1) Σ w); t(k−1)
        var_hk = float(np.sum(w * (yi - mu) ** 2) / ((k - 1) * sw))
        se_used = float(np.sqrt(var_hk))
        if not np.isfinite(se_used) or se_used == 0.0:
            # Every row sits exactly on the pooled estimate (k identical effect sizes), so HK's
            # between-study term is zero and its statistic is 0/0. That is not an infinitely
            # precise result — it is an adjustment with nothing to adjust. Fall back to the
            # ordinary random-effects SE and say so on the record. Raising here killed the WHOLE
            # RUN at the pooling step, after every paper had already been paid for.
            hakn_fallback = "se_used was zero (all rows identical); standard SE used"
            se_used = se
            q = sps.norm.ppf(1 - (1 - level) / 2)
            stat = mu / se_used if se_used else 0.0
            p = float(2 * sps.norm.sf(abs(stat)))
        else:
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
    tv = sps.t.ppf(1 - (1 - level) / 2, k - 1)
    pi_low_v, pi_high_v = mu - tv * pi_sd, mu + tv * pi_sd

    lb = ub = None
    if tau2_ci:
        lb, ub = tau2_qprofile_ci(yi, vi, level)

    return MetaResult(k=k, method=method, estimate=mu, se=se_used, ci_low=float(ci_low), ci_high=float(ci_high),
                      z=float(stat), p=p, tau2=float(tau2), tau=float(np.sqrt(tau2)), se_tau2=se_tau2, Q=Q, Q_df=df,
                      Q_p=Q_p, I2=I2_meta, I2_tau=I2_tau, H2=H2, weights=w / sw, weights_pct=100 * w / sw,
                      weights_raw=w, pi_low=float(pi_low), pi_high=float(pi_high), pi_df=k - 2,
                      pi_low_z=float(pi_low_z), pi_high_z=float(pi_high_z), pi_low_v=float(pi_low_v),
                      pi_high_v=float(pi_high_v), hakn=hakn, hakn_fallback=hakn_fallback,
                      level=level,
                      tau2_ci_low=lb, tau2_ci_high=ub, iterations=iters, converged=conv, yi=yi, vi=vi)


def prediction_interval(res: "MetaResult", method: str = "V") -> tuple[float, float, float]:
    """Return (low, high, df) for the requested convention: 'HTS' t(k-2) [meta ≤6 default, Cisneros 2024 figures],
    'V' t(k-1) [meta ≥7 default], 'z' normal [metafor default; meta 'S']."""
    m = method.upper()
    if m == "HTS":
        return res.pi_low, res.pi_high, res.k - 2
    if m == "V":
        return res.pi_low_v, res.pi_high_v, res.k - 1
    if m in ("Z", "S"):
        return res.pi_low_z, res.pi_high_z, float("inf")
    raise ValueError(f"unknown prediction-interval method {method!r}")


def egger_test(yi, sei, n_a=None, n_b=None, level: float = 0.95) -> EggerResult:
    """Egger's test for small-study effects, classic or Pustejovsky-Rodgers.

    Without group sizes: OLS of (TE/seTE) on (1/seTE) — the regression the reference review ran,
    and `metafor::regtest(predictor="sei", model="lm")`.

    With group sizes: the predictor becomes √((n_a+n_b)/(n_a·n_b)) = √(1/n_a + 1/n_b) and the fit
    is weighted least squares with weights 1/vi (Pustejovsky & Rodgers 2019;
    `metafor::regtest(predictor="sqrtninv", ni=n_a·n_b/(n_a+n_b), model="lm")`). For a standardised
    mean difference seTE is a function of the effect estimate itself, which makes the classic test
    reject too often; the sample-size predictor removes that dependence.

    Raises `ValueError` below k = 3, matching `metafor::regtest`: the fit spends two degrees of
    freedom on its two coefficients, so with two studies there is nothing left to test the
    asymmetry coefficient against and any p value would be fabricated.
    """
    yi = np.asarray(yi, float).ravel()
    sei = np.asarray(sei, float).ravel()
    if yi.shape != sei.shape:
        raise ValueError("yi and sei must have the same length")
    k = yi.size
    if k < 3:
        raise ValueError("Egger's test needs at least 3 studies")
    q = sps.t.ppf(1 - (1 - level) / 2, k - 2)

    if n_a is None or n_b is None:
        y = yi / sei
        X = np.column_stack([np.ones(k), 1 / sei])
        beta, cov, _ = _weighted_ls(X, y, np.ones(k))
        bias, corrected = 0, 1
        predictor = "precision"
    else:
        n_a = np.asarray(n_a, float).ravel()
        n_b = np.asarray(n_b, float).ravel()
        if n_a.shape != yi.shape or n_b.shape != yi.shape:
            raise ValueError("n_a and n_b must have the same length as yi")
        X = np.column_stack([np.ones(k), np.sqrt(1 / n_a + 1 / n_b)])
        beta, cov, _ = _weighted_ls(X, yi, 1 / sei ** 2)
        bias, corrected = 1, 0
        predictor = "sqrt_inv_n"

    se_bias = float(np.sqrt(cov[bias, bias]))
    t = float(beta[bias] / se_bias)
    se_est = float(np.sqrt(cov[corrected, corrected]))
    return EggerResult(intercept=float(beta[bias]), intercept_se=se_bias, t=t,
                       p=float(2 * sps.t.sf(abs(t), k - 2)), slope=float(beta[corrected]), k=k,
                       predictor=predictor, df=k - 2, estimate=float(beta[corrected]),
                       estimate_ci_low=float(beta[corrected] - q * se_est),
                       estimate_ci_high=float(beta[corrected] + q * se_est))


def _weighted_ls(X, y, w):
    """Weighted least squares with an estimated scale — what R's `lm(..., weights=w)` does."""
    W = np.asarray(w, float).ravel()
    XtW = X.T * W
    xtwx = XtW @ X
    beta = np.linalg.solve(xtwx, XtW @ y)
    resid = y - X @ beta
    df = X.shape[0] - X.shape[1]
    sigma2 = float((W * resid ** 2).sum() / df)
    return beta, sigma2 * np.linalg.inv(xtwx), sigma2


def leave_one_out(yi, vi, method: Tau2Method = "REML", level: float = 0.95,
                  labels=None) -> list[LeaveOneOut]:
    """Re-pool k times, omitting one study each time (`metafor::leave1out`).

    A pooled estimate that one study can move is a different finding from one that no study can,
    and this is the cheapest way to show which it is.
    """
    yi, vi = _check(yi, vi)
    k = yi.size
    if k < 3:
        raise ValueError("leave-one-out needs at least 3 studies")
    names = list(labels) if labels is not None else [str(i) for i in range(k)]
    if len(names) != k:
        raise ValueError("labels must have one entry per study")
    rows: list[LeaveOneOut] = []
    for i in range(k):
        keep = np.arange(k) != i
        res = random_effects(yi[keep], vi[keep], method=method, level=level)
        rows.append(LeaveOneOut(omitted=i, label=names[i], k=res.k, estimate=res.estimate,
                                se=res.se, ci_low=res.ci_low, ci_high=res.ci_high, tau2=res.tau2,
                                I2=res.I2_tau, Q=res.Q, Q_p=res.Q_p))
    return rows


def funnel_data(yi, vi, method: Tau2Method = "REML", labels=None,
                levels: tuple[float, ...] = (0.95, 0.99), n_points: int = 50) -> dict:
    """Everything a funnel plot needs, computed once: the points and the pseudo-CI contours.

    The contours are the region a study of a given standard error would fall in if the pooled
    estimate were the truth, so the plot can be drawn by any renderer without repeating the stats.
    """
    yi, vi = _check(yi, vi)
    sei = np.sqrt(vi)
    res = random_effects(yi, vi, method=method)
    se_max = float(sei.max())
    grid = np.linspace(0.0, se_max, max(2, n_points))
    contours = {}
    for level in levels:
        q = sps.norm.ppf(1 - (1 - level) / 2)
        contours[str(level)] = {"se": [float(s) for s in grid],
                                "low": [float(res.estimate - q * s) for s in grid],
                                "high": [float(res.estimate + q * s) for s in grid]}
    names = list(labels) if labels is not None else [str(i) for i in range(yi.size)]
    return {"yi": [float(v) for v in yi], "sei": [float(v) for v in sei], "labels": names,
            "estimate": float(res.estimate), "se_max": se_max, "method": method,
            "contours": contours}


def per_study_ci(yi, vi, level: float = 0.95):
    yi = np.asarray(yi, float); se = np.sqrt(np.asarray(vi, float))
    q = sps.norm.ppf(1 - (1 - level) / 2)
    return yi - q * se, yi + q * se
