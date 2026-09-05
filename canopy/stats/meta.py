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
* Cluster-robust pooling (``clusters=`` given): the FULL robumeta 2.1 CORR fit — Hedges, Tipton &
  Johnson (2010) working model, method-of-moments tau², CR2-adjusted sandwich SE and Satterthwaite
  df (Tipton 2015, robumeta ``small=TRUE``). This is a different estimator, not a corrected SE on
  the same estimate: the point estimate uses the CORR cluster weights and generally differs from
  the row-weighted REML one. Validated against robumeta 2.1 itself (tests/test_meta.py).
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
    Q_df: float              # k−1 ordinarily; NON-INTEGER under cluster-robust pooling (robumeta df_Q)
    Q_p: float
    I2: float                # meta convention (Q-based), proportion 0..1
    I2_tau: float            # metafor convention (tau²-based), proportion 0..1; NaN under RVE
    H2: float
    weights: np.ndarray      # normalized RE weights (sum 1)
    weights_pct: np.ndarray  # RE weights in %
    weights_raw: np.ndarray  # 1/(vi + tau²), or the CORR working weights 1/(k_j(v̄_j+τ²)) under RVE
    pi_low: float            # prediction interval, t(k-2) (meta HTS); t(m-2) over CLUSTERS under RVE
    pi_high: float
    pi_df: float
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
    # ---- cluster-robust (RVE) fields; defaults describe an ordinary (independent-rows) pool ----
    #: RVE was requested AND applied: estimate/se/ci/z/p are the full robumeta CORR fit.
    robust: bool = False
    #: RVE was requested at all (True even when `robust_fallback` explains why it wasn't applied).
    robust_requested: bool = False
    n_clusters: int = 0                    # m, distinct cluster labels among the pooled rows
    rho: float = float("nan")              # assumed within-cluster correlation (settings.rve_rho)
    df_robust: float = float("nan")        # Satterthwaite df of the pooled test
    se_model: float = float("nan")         # the working-model SE sqrt(1/ΣW) the robust SE replaced
    robust_small_sample: bool = False      # df_robust < 4 — robumeta: "do not trust the results"
    #: non-empty when RVE was requested but could not be applied; says why. The numbers then
    #: carry the ordinary independent-rows pool, and every report surface prints this reason.
    robust_fallback: str = ""
    #: a stability caveat that does NOT invalidate the numbers (e.g. one cluster carries
    #: essentially all the weight); printed beside the result, never instead of it.
    robust_note: str = ""

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
    I2: float                # metafor convention (tau²-based), proportion 0..1; NaN under RVE
    Q: float
    Q_p: float
    m: int = 0               # remaining clusters (cluster-robust leave-one-out only; 0 otherwise)


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


# ----------------------------------------------------------------------------- cluster-robust
def cluster_robust(yi, vi, clusters, rho: float = 0.8, level: float = 0.95) -> MetaResult:
    """The full robumeta 2.1 CORR fit, intercept-only: HTJ (2010) working model with
    method-of-moments tau², CR2-adjusted sandwich SE and Satterthwaite df (Tipton 2015,
    ``small=TRUE``). Validated digit-for-digit against robumeta itself (tests/test_meta.py).

    Raises ``ValueError`` for fewer than 2 distinct clusters — robumeta itself dies opaquely
    inside ``eigen`` there, and no meaningful number exists. Callers that must not raise go
    through :func:`random_effects`, which records the fallback instead.
    """
    yi, vi = _check(yi, vi)
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be between 0 and 1 inclusive")
    labels = [str(c) for c in clusters]
    if len(labels) != yi.size:
        raise ValueError("clusters must have one label per row")
    # group rows by cluster; dict preserves first-seen order but every quantity below is a sum
    # over clusters, so the result is invariant to row order and to relabeling (tested).
    members: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        members.setdefault(label, []).append(index)
    m = len(members)
    if m < 2:
        raise ValueError("RVE needs at least 2 clusters")
    k_j = np.array([len(rows) for rows in members.values()], dtype=float)
    vbar = np.array([float(np.mean(vi[rows])) for rows in members.values()])
    ybar = np.array([float(np.mean(yi[rows])) for rows in members.values()])

    # F2 — preliminary (tau-free) fit, weights 1/(k_j v̄_j) constant within cluster
    mu_prelim = float(np.sum(ybar / vbar) / np.sum(1.0 / vbar))
    w_prelim = np.repeat(1.0 / (k_j * vbar), k_j.astype(int))
    order = np.concatenate([np.asarray(rows) for rows in members.values()])
    QE = float(np.sum(w_prelim * (yi[order] - mu_prelim) ** 2))

    # F3 — CORR method-of-moments tau² (rho enters ONLY additively, robu.R:344-346)
    sumW = float(np.sum(1.0 / vbar))
    denom = sumW - float(np.sum((1.0 / vbar) ** 2)) / sumW
    termA = float(np.sum(1.0 / (k_j * vbar))) / sumW
    termB = float(np.sum((k_j - 1.0) / (k_j * vbar))) / sumW
    tau2 = max(0.0, (QE - m + termA) / denom + rho * termB / denom)

    # F4 — heterogeneity as robumeta reports it (df_Q non-integer; no p is defined)
    df_Q = m - termA - rho * termB
    I2 = float(max(0.0, (QE - df_Q) / QE)) if QE > 0 else 0.0

    # F5 — final fit and CR2 sandwich
    W = 1.0 / (vbar + tau2)                        # cluster weights
    S = float(np.sum(W))
    mu = float(np.sum(W * ybar) / S)
    h = W / S                                      # cluster leverages, sum exactly 1
    se_robust = float(np.sqrt(np.sum(W ** 2 * (ybar - mu) ** 2 / (1.0 - h)) / S ** 2))
    note = ""
    if float(np.min(1.0 - h)) < 1e-8:
        # robumeta clamps a near-zero eigenvalue at 1e-10 and we deliberately do not replicate
        # the clamp: past this point both implementations print garbage, and the honest output
        # is the caveat, not a stabilised-looking number.
        note = "one cluster carries essentially all the weight; results are not stable"

    # F6 — Satterthwaite df from the weights alone; cross-term via (Σg)² − Σg², g = h²/(1−h)
    g = h ** 2 / (1.0 - h)
    df_S = float(1.0 / (np.sum(h ** 2) + np.sum(g) ** 2 - np.sum(g ** 2)))

    # F8 — inference on t(df_S)
    stat = mu / se_robust if se_robust > 0 else float("inf") * np.sign(mu or 1.0)
    p = float(2 * sps.t.sf(abs(stat), df_S))
    tq = sps.t.ppf(1 - (1 - level) / 2, df_S)
    ci_low, ci_high = mu - tq * se_robust, mu + tq * se_robust

    # D8 — working-model prediction interval over CLUSTERS, with the caveat carried by the report
    pi_sd = float(np.sqrt(se_robust ** 2 + tau2))
    if m >= 3:
        t2 = sps.t.ppf(1 - (1 - level) / 2, m - 2)
        pi_low, pi_high = mu - t2 * pi_sd, mu + t2 * pi_sd
    else:
        pi_low = pi_high = float("nan")
    zq = sps.norm.ppf(1 - (1 - level) / 2)
    tv = sps.t.ppf(1 - (1 - level) / 2, m - 1)

    # F9 — per-row working weights W_j/k_j; they sum to S, so shares sum to 1 (100%)
    per_row = np.empty(yi.size)
    for (label, rows), Wj, kj in zip(members.items(), W, k_j):
        per_row[np.asarray(rows)] = Wj / kj
    return MetaResult(k=yi.size, method="CORR-MoM", estimate=mu, se=se_robust,
                      ci_low=float(ci_low), ci_high=float(ci_high), z=float(stat), p=p,
                      tau2=float(tau2), tau=float(np.sqrt(tau2)), se_tau2=None,
                      Q=QE, Q_df=float(df_Q), Q_p=float("nan"), I2=I2,
                      I2_tau=float("nan"), H2=float("nan"),
                      weights=per_row / S, weights_pct=100 * per_row / S, weights_raw=per_row,
                      pi_low=float(pi_low), pi_high=float(pi_high), pi_df=float(m - 2),
                      pi_low_z=float(mu - zq * pi_sd), pi_high_z=float(mu + zq * pi_sd),
                      pi_low_v=float(mu - tv * pi_sd), pi_high_v=float(mu + tv * pi_sd),
                      hakn=False, level=level, yi=yi, vi=vi,
                      robust=True, robust_requested=True, n_clusters=m, rho=float(rho),
                      df_robust=df_S, se_model=float(np.sqrt(1.0 / S)),
                      robust_small_sample=bool(df_S < 4), robust_note=note)


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
                   tau2_ci: bool = False, *, clusters=None, rho: float = 0.8) -> MetaResult:
    """`clusters=None` (the default) is the ordinary independent-rows pool, bit-identical to
    what this function always produced. With `clusters` given (one label per row), the pool is
    the full robumeta CORR fit (:func:`cluster_robust`) — `method` is then ignored, because the
    CORR method-of-moments tau² is part of that method. The ONE degenerate condition — fewer
    than 2 distinct clusters — falls back to the independent pool with the reason recorded in
    `robust_fallback`, following the hakn_fallback precedent: raising here would kill a whole
    run at the pooling step after every paper had been paid for.
    """
    if clusters is not None:
        if hakn:
            raise ValueError("hakn and cluster-robust pooling are mutually exclusive "
                             "small-sample corrections; choose one")
        labels = [str(c) for c in clusters]
        if len(set(labels)) >= 2:
            return cluster_robust(yi, vi, labels, rho=rho, level=level)
        result = random_effects(yi, vi, method=method, hakn=False, level=level, tau2_ci=tau2_ci)
        result.robust_requested = True
        result.robust_fallback = ("fewer than 2 clusters (all pooled rows share one cluster); "
                                  "pooled as independent rows")
        return result
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
    'V' t(k-1) [meta ≥7 default], 'z' normal [metafor default; meta 'S'].

    Under a cluster-robust pool the df base is the CLUSTER count (the stored intervals were
    computed on it): counting rows there would fake precision the dependence removed.
    """
    m = method.upper()
    base = res.n_clusters if getattr(res, "robust", False) else res.k
    if m == "HTS":
        return res.pi_low, res.pi_high, base - 2
    if m == "V":
        return res.pi_low_v, res.pi_high_v, base - 1
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
                  labels=None, *, clusters=None, rho: float = 0.8) -> list[LeaveOneOut]:
    """Re-pool k times, omitting one study each time (`metafor::leave1out`).

    A pooled estimate that one study can move is a different finding from one that no study can,
    and this is the cheapest way to show which it is.

    With `clusters` given, the unit omitted is the CLUSTER, not the row: under a cluster-robust
    pool, dropping one of a paper's five rows is not a meaningful "without this study". One
    result row per cluster; `label` is then the omitted cluster's label; needs >= 3 clusters.
    """
    yi, vi = _check(yi, vi)
    k = yi.size
    if clusters is not None:
        cl = [str(c) for c in clusters]
        if len(cl) != k:
            raise ValueError("clusters must have one label per row")
        distinct = list(dict.fromkeys(cl))
        if len(distinct) < 3:
            raise ValueError("cluster-robust leave-one-out needs at least 3 clusters")
        names = list(labels) if labels is not None else distinct
        if len(names) != len(distinct):
            raise ValueError("labels must have one entry per cluster")
        arr = np.array(cl)
        rows: list[LeaveOneOut] = []
        for i, label in enumerate(distinct):
            keep = arr != label
            res = random_effects(yi[keep], vi[keep], method=method, level=level,
                                 clusters=arr[keep], rho=rho)
            rows.append(LeaveOneOut(omitted=i, label=names[i], k=res.k, estimate=res.estimate,
                                    se=res.se, ci_low=res.ci_low, ci_high=res.ci_high,
                                    tau2=res.tau2, I2=res.I2_tau, Q=res.Q, Q_p=res.Q_p,
                                    m=res.n_clusters))
        return rows
    if k < 3:
        raise ValueError("leave-one-out needs at least 3 studies")
    names = list(labels) if labels is not None else [str(i) for i in range(k)]
    if len(names) != k:
        raise ValueError("labels must have one entry per study")
    rows = []
    for i in range(k):
        keep = np.arange(k) != i
        res = random_effects(yi[keep], vi[keep], method=method, level=level)
        rows.append(LeaveOneOut(omitted=i, label=names[i], k=res.k, estimate=res.estimate,
                                se=res.se, ci_low=res.ci_low, ci_high=res.ci_high, tau2=res.tau2,
                                I2=res.I2_tau, Q=res.Q, Q_p=res.Q_p))
    return rows


def funnel_data(yi, vi, method: Tau2Method = "REML", labels=None,
                levels: tuple[float, ...] = (0.95, 0.99), n_points: int = 50,
                center: float | None = None) -> dict:
    """Everything a funnel plot needs, computed once: the points and the pseudo-CI contours.

    The contours are the region a study of a given standard error would fall in if the pooled
    estimate were the truth, so the plot can be drawn by any renderer without repeating the stats.

    `center` overrides the internally re-pooled estimate — REQUIRED whenever the run's pooled
    estimate came from a different model (cluster-robust), so funnel.json and pooled.json can
    never disagree about what "the pooled estimate" is.
    """
    yi, vi = _check(yi, vi)
    sei = np.sqrt(vi)
    estimate = float(center) if center is not None else random_effects(yi, vi, method=method).estimate
    se_max = float(sei.max())
    grid = np.linspace(0.0, se_max, max(2, n_points))
    contours = {}
    for level in levels:
        q = sps.norm.ppf(1 - (1 - level) / 2)
        contours[str(level)] = {"se": [float(s) for s in grid],
                                "low": [float(estimate - q * s) for s in grid],
                                "high": [float(estimate + q * s) for s in grid]}
    names = list(labels) if labels is not None else [str(i) for i in range(yi.size)]
    return {"yi": [float(v) for v in yi], "sei": [float(v) for v in sei], "labels": names,
            "estimate": float(estimate), "se_max": se_max, "method": method,
            "contours": contours}


def per_study_ci(yi, vi, level: float = 0.95):
    yi = np.asarray(yi, float); se = np.sqrt(np.asarray(vi, float))
    q = sps.norm.ppf(1 - (1 - level) / 2)
    return yi - q * se, yi + q * se
