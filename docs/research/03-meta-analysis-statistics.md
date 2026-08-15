# 03 — Meta-analysis statistics reference (replicating R `meta`/`metafor` in Python)

Status: research brief, 2026-08-14. Target: `canopy.stats` must reproduce, to floating-point tolerance,
`meta::metagen(sm="SMD", random=TRUE, method.tau="REML", hakn=FALSE, prediction=TRUE)` and
`metafor::rma(yi, vi, method="REML")`. Everything below was checked against **metafor 4.6-0** and **meta 8.2-1**
installed locally (`/usr/local/bin/Rscript`, R 4.4.0; note CRAN is ahead — metafor 5.0-1 (2026-04-26) and meta 8.5-0
(2026-05-25) — so pin the golden fixtures to the versions actually used) and against a from-scratch NumPy/SciPy implementation
(`scratchpad/ref.py`) that reproduced all R numbers to ≥ 9 significant digits. The worked example in §7 is that run.

Notation: study *i* has groups A (reference, e.g. young) and B (comparison, e.g. older) with means m_A, m_B,
SDs s_A, s_B and sizes n_A, n_B; N = n_A + n_B; m = N − 2 (df); k = number of studies; y_i = effect, v_i = its
sampling variance; w_i = 1/v_i (common-effect weights); w_i* = 1/(v_i + τ²) (random-effects weights).

---

## 1. Standardized mean difference: Cohen's d, Hedges' g, variances, CI

| Quantity | Formula | Source |
|---|---|---|
| Pooled SD | s_p = √[((n_A−1)s_A² + (n_B−1)s_B²)/(N−2)] | Borenstein 2009 eq. 4.19; Cochrane 6.5.1.2 |
| Cohen's d | d = (m_B − m_A)/s_p | Cohen 1988; Borenstein eq. 4.18 |
| Var(d), large-sample ("Rmd" formula) | V_d = (n_A+n_B)/(n_A n_B) + d²/(2(N−2)) → SE(d)=√V_d | Formula as used in the Cisneros Rmd. **Note:** Borenstein 2009 eq. 4.20 actually reads V_d = (n_A+n_B)/(n_A n_B) + d²/(2(n_A+n_B)) — denominator 2N, not 2(N−2) (verified in the book text); Hedges & Olkin 1985 eq. 15 likewise uses 2N. The N−2 variant is a df-based approximation whose primary source we could not pin down — cite it as "Cisneros Rmd formula", not Borenstein |
| Hedges' correction, exact | J(m) = Γ(m/2) / (√(m/2) Γ((m−1)/2)); compute as exp(lgamma(m/2) − ½ln(m/2) − lgamma((m−1)/2)) | Hedges 1981; metafor `.cmicalc()` |
| Hedges' correction, approx. | J ≈ 1 − 3/(4m − 1) | Hedges 1981; Borenstein eq. 4.22; Lakens 2013 |
| Hedges' g | g = J·d | |
| Var(g) — **metafor default** (`vtype="LS"`) | v = 1/n_A + 1/n_B + g²/(2N) — note **g** (not d) and **2N** (not 2(N−2)) | Hedges 1982 eq. 8; Hedges & Olkin 1985 eq. 15; metafor `escalc()` source |
| Var(g) — Borenstein (`vtype="LS2"`) | v = J²·[1/n_A + 1/n_B + d²/(2N)] | Borenstein 2009 eq. 12.17 |
| Var(g) — unbiased (`vtype="UB"`, "Hedges 1983 exact") | v = 1/n_A + 1/n_B + (1 − (m−2)/(m J²))·g² | Hedges 1983 eq. 9 |
| Var(g) — `meta::metacont(exact.smd=TRUE)` default | same as UB above (White & Thomas 2005); verified numerically: metacont seTE = √(UB v) | meta docs |
| 95% CI of d or g | y ± z_{0.975}·SE, z = 1.959964 | |
| statsmodels `effectsize_smd` | J ≈ 1 − 3/(4N − 9); v = N/(n_A n_B) + g²/(2(N − 3.94)) — **differs** from all of the above (5th–4th decimal) | statsmodels source |

Numerical size of the choices (worked study 1: d = 0.53007, n = 12/12): SE from Rmd formula 0.415996; metafor LS
√v = 0.414877 (for g = 0.51176); UB √v = 0.416090; metacont seTE = 0.416089. Differences ≈ 0.3 % of SE — invisible
in a forest plot but visible in a 1e-6 tolerance test, so **the pipeline must record which variant it uses**.
For the Cisneros replication use *d with the Rmd formula*; expose g/LS/UB as options.

Exact-vs-approximate J: at m = 10 the difference is 3.3e-4; at m = 30, 3.6e-5; at m = 100, 3.1e-6. Use the exact
lgamma form (metafor and meta both do). Guard m ≤ 1 → NaN.

Recommendation: implement `smd(mA,sA,nA,mB,sB,nB, *, hedges: bool, vtype: {"rmd","LS","LS2","UB"})` returning
(y, v, J, s_p). Default for Canopy = `hedges=False, vtype="rmd"` when replicating Cisneros; `hedges=True, vtype="LS"`
when the protocol says "as metafor".

---

## 2. Recovering SDs (and means) from what papers actually report

All conversions produce a per-group SD to feed §1; each rule needs its own provenance tag because they carry
different error.

| Reported | SD estimate | Notes / source |
|---|---|---|
| SE of a group mean | SD = SE·√n | Cochrane 6.5.2.2 |
| 95 % CI of a group mean, large n (≥ ~60/group) | SD = √n·(U − L)/3.92 (90 %: 3.29; 99 %: 5.15) | Cochrane 6.5.2.2 |
| 95 % CI of a group mean, small n | SD = √n·(U − L)/(2·t_{0.975, n−1}) | Cochrane 6.5.2.2 ("replaced with slightly larger numbers specific to the t distribution"). n=15: t=2.1448 → divisor 4.2896 not 3.92; SD 2.709 vs 2.964 (−8.6 %) |
| 95 % CI of a **difference** in means | SE_MD = (U − L)/(2·t_{0.975, N−2}) (or /3.92 if large); s_p = SE_MD / √(1/n_A + 1/n_B) | Cochrane 6.5.2.3 |
| Exact p (or t) for a two-group comparison | t = t⁻¹(1 − p/2, N−2); SE_MD = |MD|/t; s_p as above — or go straight to d (§3) | Cochrane 6.5.2.3 |
| Median + IQR (q1, q3), n | Wan 2014 eq. 16: SD ≈ (q3 − q1)/η(n), η(n) = 2Φ⁻¹((0.75n − 0.125)/(n + 0.25)); large-n limit η→1.349 (Cochrane's "IQR/1.35") | Wan 2014; Cochrane 6.5.2.5 |
| Median + min/max, n | Wan 2014 eq. 9: SD ≈ (b − a)/ξ(n), ξ(n) = 2Φ⁻¹((n − 0.375)/(n + 0.25)); (b−a)/4 or /6 only as fallbacks (Hozo 2005) | Wan 2014 |
| Five-number summary | Wan 2014 eq. 13: SD ≈ ½[(b−a)/ξ(n) + (q3−q1)/η(n)]. **Shi 2020 (metafor default):** SD ≈ w(b−a)/ξ(n) + (1−w)(q3−q1)/η(n), w = 1/(1 + 0.07 n^0.6) | Shi 2020 eq. 10; metafor `conv.fivenum()` |
| Mean from median (Luo 2018) | (a,m,b): x̄ ≈ [4/(4+n^0.75)]·(a+b)/2 + [n^0.75/(4+n^0.75)]·m. (q1,m,q3): x̄ ≈ (0.7 + 0.39/n)(q1+q3)/2 + (0.3 − 0.39/n)m. Five numbers: w1 = 2.2/(2.2+n^0.75), w2 = 0.7 − 0.72/n^0.55, x̄ ≈ w1(a+b)/2 + w2(q1+q3)/2 + (1−w1−w2)m | Luo 2018; metafor source |
| Individual data points (digitized dots / supplementary data) | SD = sample SD with n−1; SE = SD/√n; prefer this over any of the above; record n actually counted vs n reported | — |
| Error bars in a figure | Must know whether they are SD, SE or CI (caption/methods). If SE → SD = SE·√n; if 95 % CI → SD = √n·half-width/1.96 (or /t) | Cochrane 6.5.2.2 |
| Bootstrapped/robust CIs, or asymmetric bars | Do not convert; escalate to human review | — |

Worked numbers (n = 20, a=3, q1=8, m=10, q3=13, b=20): ξ(20) = 3.73648, η(20) = 1.25337; SD_Wan,range = 4.5497,
SD_Wan,IQR = 3.9892, SD_Wan,5 = 4.2695, SD_Shi,5 = 4.3833 (w = 0.70304); mean_Luo(a,m,b) = 10.4459,
mean_Luo(q1,m,q3) = 10.3597, mean_Luo,5 = 10.5638; IQR/1.35 = 3.7037.
Skew screening (Shi 2023, implemented in metafor `conv.fivenum(test=TRUE)`): flag when
|(a + b − 2m)/(b − a)| > 1/ln(n+9) + 2.5/(n+1) or |(q1 + q3 − 2m)/(q3 − q1)| > 2.65/√n − 6/n².
Log-normal variants exist (`dist="lnorm"`) — out of scope unless a protocol asks.

---

## 3. d from test statistics

| Reported | Formula | Caveat |
|---|---|---|
| Independent-samples (Student) t, df = N−2 | d = t·√(1/n_A + 1/n_B) (= t·√(N/(n_A n_B))); sign from direction of means | Exact algebraic identity with pooled-SD d. Lakens 2013 eq. 2 |
| Welch t | d = t_W·√(s_A²/n_A + s_B²/n_B) / s_p — needs SDs; naive t·√(1/n_A+1/n_B) is biased when SDs differ (example: 0.874 vs 0.833) | If SDs unavailable, use naive form and flag |
| One-way ANOVA F(1, N−2), two groups, between-subjects | d = √(F·N/(n_A n_B)); sign from means | F = t² |
| Two-tailed exact p, two-group test | t = t⁻¹(1 − p/2, N−2) → d as above. **Use t, not z**: p = .018, df = 31: t = 2.498 (z = 2.366, −5 %) | Cochrane 6.5.2.3 |
| Inequality p ("p < .05") | Lower bound only: t = t⁻¹(0.975, df); df=31 → d ≥ 0.713. Record as **bound**, not estimate; usually exclude or sensitivity-analyse | Borenstein ch. 7 |
| Partial η² (between-subjects, two groups) | F = η²_p/(1−η²_p)·df_error → d as above; equal n: d = 2√(η²/(1−η²)) | Only valid when the factor is between-subjects and there is no other between factor sharing the error term |
| Point-biserial r | d = r/√(1−r²)·√((n_A+n_B)²/(n_A n_B)); equal n: d = 2r/√(1−r²) | Borenstein eq. 7.5–7.7 |
| Paired / repeated-measures t or F | **Not convertible** to a between-group d: it estimates the within-subject change (d_z, SD of differences, needs r), not group B − group A. Only usable if each group's own change score is the outcome (then two independent d's still needed) | Lakens 2013 (d_z vs d_av); Borenstein ch. 4 |
| Mixed-design interaction F (Group × Block) | The interaction is a difference-of-differences; d = √(F·N/(n_A n_B)) gives an SMD of the *interaction contrast*, not of the late-adaptation level. Use only when the protocol's outcome is that contrast; else flag as "not directly convertible" | Cisneros analysts used group main-effect t/F only |
| Group main effect F(1, N−2) from a mixed ANOVA | Uses between-subjects error → OK, but the estimate averages over levels (blocks); acceptable proxy if the outcome is the between-block mean; flag | |
| Regression coefficient / Bayes factor / non-parametric U | Do not convert automatically; escalate | |

Every conversion must set `effect_source ∈ {means_sd, means_se, means_ci, digitized, t, F, p_exact, p_bound, r, eta2}`
so the verifier can weight trust and the report can footnote it (Cisneros did exactly this: WPD / text / statistic).

---

## 4. Pooling: random-effects model, heterogeneity, intervals

### 4.1 Between-study variance τ²

**DerSimonian–Laird (closed form)**: Q = Σ w_i (y_i − ȳ_w)², ȳ_w = Σ w_i y_i / Σ w_i, C = Σ w_i − Σ w_i²/Σ w_i,
τ²_DL = max(0, (Q − (k−1))/C).

**Paule–Mandel**: solve Σ w_i*(τ²)(y_i − ȳ*(τ²))² = k − 1 by root-finding (metafor: `uniroot` on [0, tau2.max],
tol = .Machine$double.eps^0.25 ≈ 1.22e-4, tau2.max = max(100, 10·mad(y)²)); τ² = 0 if the LHS at 0 is already < k−1.
statsmodels' `method_re="iterated"` is PM with a looser stop (matches metafor to ~1e-6).

**REML — exactly as metafor `rma.uni` (Fisher scoring)**. Intercept-only case (X = 1):

```
inputs y[k], v[k]; con: threshold=1e-5, maxiter=100, stepadj=1, tau2.min=0, tol=eps^0.25
# initial value = Hedges (HE) estimator truncated at 0:
tau2 = max(0, ( sum((y-mean(y))^2) - (k-1)/k*sum(v) ) / (k-1))    # (RSS - tr(PV))/(k-p), P = I - 11'/k
change = threshold+1; iter = 0
while change > threshold:
    iter += 1; old = tau2
    w = 1/(v+tau2); W = diag(w)
    P = W - w w'/sum(w)                       # W - W X (X'WX)^-1 X'W
    PP = P P
    adj = ( y'PP y - tr(P) ) / tr(PP)         # REML score / expected information
    adj *= stepadj
    while tau2 + adj < tau2.min: adj /= 2     # step-halving keeps tau2 >= 0
    tau2 = tau2 + adj
    change = |old - tau2|
    if iter > maxiter: error "Fisher scoring did not converge"
# ll0 check (metafor con$ll0check=TRUE): if REML loglik at tau2=0 exceeds loglik at the estimate by > tol
# and tau2 > threshold, set tau2 = 0 (local-maximum guard).
llR(t) = -(k-1)/2 ln(2pi) - 1/2 sum ln(v+t) - 1/2 ln(sum 1/(v+t)) - 1/2 sum w*(y-mu*)^2
tau2 = max(tau2.min, tau2)
```

(ML replaces `tr(P)` with Σw and `tr(PP)` with Σw²; EB uses adj = (y'Py·k/(k−1) − k)/Σw.)
Worked example trace (§7): HE start 0.0859 → 0.1298 → 0.1246 → 0.1251 → 0.1251 → converged after 5 iterations at
τ² = 0.1250834209 (identical to metafor `verbose=TRUE`). Because the update is Fisher scoring, not Newton, results
match R only if the same recursion is used; a generic `scipy.optimize.minimize` on the REML likelihood converges to
the same optimum but only to the optimizer's tolerance (PyMARE: 0.12508276 vs 0.12508342). `meta::metagen`
calls `metafor::rma.uni` internally for REML, so meta and metafor τ² are bit-identical.

**SE(τ²)** for REML (metafor): se = √(2 / tr(P·P)) at the final τ² (0.18200934 in the example) — report only if asked.

**CI for τ² — Q-profile (Viechtbauer 2007)** — metafor `confint()` and meta's default for every estimator except DL
(meta uses Jackson 2013 for DL): with Q_gen(t) = Σ (y_i − ȳ*(t))²/(v_i + t), solve Q_gen(t) = χ²_{k−1, 0.975}
(lower bound) and = χ²_{k−1, 0.025} (upper bound) on [0, tau2.max]; bound = 0 when Q_gen(0) is below the target.
Example: (0, 1.7590172) for τ²; τ = (0, 1.3262795). I²/H² CIs are transformations of the τ² bounds.

### 4.2 Pooled estimate and test

μ̂ = Σ w_i* y_i / Σ w_i*, SE(μ̂) = 1/√Σ w_i*, z = μ̂/SE, p = 2·(1 − Φ(|z|)), CI = μ̂ ± 1.959964·SE.
Random-effects study weights (forest-plot %) = w_i*/Σ w_i*.

### 4.3 Heterogeneity statistics — two definitions of I², pick one deliberately

Q as in 4.1 (common-effect weights), df = k−1, p_Q = 1 − F_χ²(Q; k−1).
* **meta (default `method.I2="Q"`)**: I² = max(0, (Q − df)/Q); H = √(Q/df). Example: 48.38 %, H = 1.392.
* **metafor `rma()`**: "typical" within-study variance v_t = (k−1)/(Σw − Σw²/Σw); I² = 100·τ²/(v_t + τ²);
  H² = τ²/v_t + 1. Example: **49.15 %**, H² = 1.967. These differ whenever τ² ≠ τ²_DL (REML here).
Canopy should report the meta version when replicating a `meta` analysis and offer the metafor version as
`i2_method="tau2"`; label which one is on the plot.

### 4.4 Prediction interval — the exact `meta` rule depends on version

PI = μ̂ ± t_{df, 0.975}·√(τ² + SE(μ̂)²).
* `meta` < 8.0-0 default (`method.predict="HTS"`, Higgins–Thompson–Spiegelhalter 2009): **df = k − 2**, SE = classic
  RE SE. Example: (−1.008326, 1.667627). This is very likely what the Cisneros Rmd produced (CRAN versions up to
  7.0-0, released 2024-01-11, used HTS); confirm from their `sessionInfo()` — an analysis re-run after 2024-10-30
  would silently switch to "V".
* `meta` ≥ 8.0-0 (2024-10-30) default (`method.predict="V"`, Veroniki 2019): **df = k − 1**. Example: (−0.837633, 1.496934).
  (Corrected from "7.1": the meta NEWS file dates the default change to 8.0-0.)
* `method.predict="S"` (Skipka): z instead of t → (−0.494365, 1.153666).
* `metafor::predict.rma()` default: **z** (i.e. same as "S"); with `test="knha"`/`"t"`: t_{k−p} with the HK SE
  (example knha: (−0.825917, 1.485218)); `pi.type="Riley"`: t_{k−2}·√(τ² + SE²) with classic SE = meta "HTS" exactly.
* meta `"HK"` = HK SE with t_{k−1}; `"HK-PR"` = HK SE with t_{k−2} (Partlett & Riley 2017).
Special cases: k = 2 → HTS PI undefined (df = 0), meta prints NA; k ≤ 2 → return NaN. When τ² = 0, metafor's default
PI collapses to the CI, meta HTS/V still widens it (t_{k−2}·SE): 4-study zero-τ² example CI (0.0167, 0.4076) vs
meta HTS PI (−0.2169, 0.6412).

### 4.5 Hartung–Knapp (`hakn=TRUE` / `method.random.ci="HK"` / metafor `test="knha"`)

q = Σ w_i*(y_i − μ̂)²/(k − 1); SE_HK = √(q/Σ w_i*) = SE·√q; test t = μ̂/SE_HK on k−1 df; CI = μ̂ ± t_{k−1,0.975}·SE_HK.
Example: SE_HK 0.219413, t = 1.5024, df 4, p 0.2074, CI (−0.279539, 0.938840) — same in meta and metafor.
Ad-hoc guard (`adhoc.hakn.ci="se"`, metafor `test="adhoc"`): SE = max(SE_HK, SE) → CI (−0.301481, 0.960782).
Not the Cisneros default (hakn=FALSE) but should be an option; IntHout 2014 recommends it as default.

### 4.6 Publication bias

* **Egger's test as in the Rmd**: `lm(TE/seTE ~ 1/seTE)`; the **intercept** is the bias estimate, tested with
  t on k−2 df. Algebraically identical to weighted `lm(TE ~ seTE, weights = 1/seTE²)` (`meta::metabias(method.bias=
  "Egger")`, `metafor::regtest(model="lm")`). Example: intercept 1.7602 (SE 3.8268), t = 0.4600, df 3, p = 0.6768 —
  identical across the three. Note `metafor::regtest()` *default* is `model="rma"` (mixed-effects meta-regression
  on sei) — different numbers; use `model="lm"` for the Rmd. meta requires k ≥ 10 by default (`k.min`).
* **Trim-and-fill** (Duval & Tweedie 2000): metafor `trimfill(rma)` default estimator L0, side chosen from the sign
  of the Egger-type regression, iterative until k0 stabilises, then refit with the same τ² method. Example: k0 = 1
  (left), pooled 0.2494 (−0.1456, 0.6443). `meta::trimfill` gave k0 = 1 but 0.2444 (−0.1543, 0.6431) — the two
  packages fill/re-estimate slightly differently, so pick metafor as the oracle for this feature.
* **Vevea–Hedges step-function selection model** (optional): `weightr::weightfunct(y, v, steps=c(0.025, 1))` ≡
  `metafor::selmodel(rma, type="stepfun", steps=c(0.025,1))` (ML τ²). Needs k ≳ 10 and p-values in every interval;
  the 5-study example fails ("one or more intervals do not contain any observed p-values") — implement as
  optional, gate on k and interval counts.

---

## 5. Sign and direction conventions

Define once per protocol: `positive_means = "B higher on outcome"` with B = comparison group (older adults) and A =
reference (young), d = (m_B − m_A)/s_p. Then apply an outcome-level `direction` flag:
* `higher_is_more` (adaptation magnitude, aftereffect size, % adaptation, late-block hand angle toward target):
  keep sign; d > 0 = older adapt more.
* `higher_is_worse` (error, residual error, distance from target, RMSE, absolute deviation, time-to-criterion): the
  paper's larger value means *less* adaptation → multiply d by −1 so that + still means "enhanced adaptation in
  older adults" (Cochrane 6.5.1.2 / MECIR: multiply one set of means by −1 *before* standardizing).
* Some papers report a signed hand angle where the compensatory direction is negative (e.g. rotation −30°, hand
  angle +25° means 25° of adaptation): normalise to |adaptation| per group before subtracting; the LLM extractor
  must emit `raw_value_semantics` (`signed_toward_target`, `signed_error`, `magnitude`) and the stats layer applies
  the flip, never the extractor.
* When d comes from t/F/p, the sign is *not* in the statistic; it must be inferred from the means/text and stored
  with provenance (`sign_source`). Unresolvable sign → exclude and flag.
* Cross-check rule for the verifier: after orientation, at least one of {means, text claim "older adults adapted
  less/more"} must agree with sign(d); disagreement blocks inclusion.

---

## 6. Existing Python implementations — audit

| Package | Version tested | What it does | Verdict |
|---|---|---|---|
| `statsmodels.stats.meta_analysis` (`combine_effects`, `effectsize_smd`) | 0.14.6 | DL exact (τ² 0.12125456, matches R); `method_re="iterated"` = Paule–Mandel (~1e-6 off); **no REML**, no Q-profile CI, no PI, no HK; `effectsize_smd` uses J = 1−3/(4N−9) and 2(N−3.94) → g/var differ from metafor at 4th decimal | Use only as a secondary cross-check for DL |
| PyMARE | 0.0.10 (installed in scratch venv) | DL, HE, PM ("Hedges", "SampleSize"), REML/ML via `VarianceBasedLikelihoodEstimator` (SciPy optimizer): τ² 0.12508276 vs 0.12508342, μ̂ 0.32965022 vs 0.32965030; I² uses (Q−df)/Q. No PI, no HK, no Q-profile CI, no Egger | Good independent check at 1e-6; not exact |
| PythonMeta | 1.26 | Fixed/random (DL) only, aimed at RevMan-style plots; sparse tests | Not recommended |
| SciPy | 1.17 (installed; PyPI latest 1.18.0) | Nothing meta-analytic beyond `stats` primitives | Building block only |
| R via `Rscript` (metafor 4.6-0, meta 8.2-1 installed here) | — | The oracle | Use for CI tests, not at runtime |

**Recommendation**: implement from scratch (~250 lines, NumPy/SciPy only) mirroring metafor's algorithms verbatim
(§4.1 pseudo-code, `brentq` with `xtol=eps**0.25` for PM/Q-profile), and validate against R in CI. Keep an
optional `--engine r` that shells out to `Rscript` for auditors.

### Validation plan (unit + property tests)

1. **Golden fixtures from R** (generated by `tests/r/gen_fixtures.R`, checked in as JSON): the 5-study example
   below, the 4-study τ²=0 example, k = 2 and k = 3 cases, a k = 40 synthetic set with wide v_i range, one set with
   a huge outlier (τ² ≫), one with all-equal y (Q = 0, HK degenerate). For each: τ² (REML, DL, PM), μ̂, SE, z, p,
   CI, Q, p_Q, I² (both defs), H/H², τ² CI (QP), PI (HTS, V, S, knha), HK CI, weights, Egger (lm form),
   trim-and-fill k0/estimate, `escalc` yi/vi for LS/LS2/UB.
2. **Tolerances**: τ², μ̂, SE, CI, PI, weights: `abs=1e-8` (REML with the same recursion agrees to ~1e-12; PM/QP
   bounded by `uniroot`'s 1.22e-4 tolerance in R, so use `abs=1e-4` **relative to the root**, or re-solve at tight
   tolerance in both and compare at 1e-8); p-values `abs=1e-10`; I² `abs=1e-6` (%); Egger t/p `abs=1e-8`;
   trim-and-fill k0 exact match and estimate `abs=1e-6`.
3. **Cisneros end-to-end**: feed the published per-study d/SE table into both engines; require identical pooled
   estimate/CI/PI to 4 decimals as printed in the paper (their rounding), and identical to R to 1e-8.
4. **Property tests** (hypothesis): τ² ≥ 0; PI ⊇ CI when df ≥ 1; DL ≤ … no ordering guaranteed, but μ̂_RE → μ̂_FE
   as τ² → 0; d(t) round-trips d → t → d; J(m)·d ≤ d for d>0; sign flip invariance (negating all y negates μ̂,
   leaves τ², Q, I², |z| unchanged).
5. **Regression guard**: pin the golden JSON to metafor/meta versions in its header; CI job re-generates with the
   installed R and fails loudly on drift (e.g. meta changed the PI default at 8.0-0, 2024-10-30).

---

## 7. Worked example (turn into `tests/test_meta_stats.py`)

Input (young = A, older = B):

| study | m_A | s_A | n_A | m_B | s_B | n_B |
|---|---|---|---|---|---|---|
| S1 | 10.0 | 2.0 | 12 | 11.2 | 2.5 | 12 |
| S2 | 12.5 | 3.0 | 20 | 11.0 | 3.2 | 18 |
| S3 | 8.0 | 2.5 | 15 | 9.5 | 2.0 | 15 |
| S4 | 15.0 | 4.0 | 30 | 17.0 | 4.5 | 25 |
| S5 | 11.0 | 3.5 | 10 | 13.0 | 3.0 | 10 |

Per-study (Rmd formulas):
d = [0.5300713252, −0.4844875426, 0.6625891564, 0.4723959089, 0.6135719911];
SE(d) = [0.4159957644, 0.3298722012, 0.3757300289, 0.2746608754, 0.4587564892].
Hedges: J = [0.96545, 0.97900, 0.97293, 0.98577, 0.95763]; g = [0.5117577101, −0.4743115548, 0.6446542694,
0.4656740521, 0.5875850250]; metafor LS v = [0.1721228324, 0.1085157062, 0.1402596521, 0.0753047181, 0.2086314040];
UB v = [0.1731302480, 0.1088392820, 0.1412469552, 0.0754488141, 0.2106156387].

Pooling y = d, v = SE²:

| quantity | value |
|---|---|
| τ²_HE (start) | 0.0858546355 |
| τ²_REML (5 Fisher-scoring iterations) | **0.1250834209** (τ = 0.3536713458; SE(τ²) 0.1820093439) |
| τ²_DL | 0.1212545615 |
| τ²_PM | 0.1085057664 |
| Q, df, p | 7.7485839265, 4, 0.1012343546 |
| I² (meta, (Q−df)/Q) / H | 48.3776644 % / 1.3918139 |
| I² (metafor, τ²/(τ²+v_t)) / H² | 49.1543936 % / 1.9667383 |
| μ̂ (REML) | **0.3296502955** |
| SE(μ̂) | 0.2273164917 |
| z, p | 1.4501820480, 0.1470077598 |
| 95 % CI | (−0.1158818414, 0.7751824324) |
| RE weights (%) | 17.33, 22.09, 19.41, 25.77, 15.40 |
| τ² CI (Q-profile) | (0, 1.7590172285); τ: (0, 1.3262795) |
| PI, HTS (t₃ = 3.1824, SE_pred 0.4204238436) | (−1.0083260122, 1.6676266032) |
| PI, V (t₄ = 2.7764) | (−0.8376334272, 1.4969340182) |
| PI, S / metafor default (z) | (−0.4943652962, 1.1536658872) |
| HK: SE, t, df, p, CI | 0.2194134595, 1.5024160153, 4, 0.2074071001, (−0.2795391301, 0.9388397211) |
| HK ad hoc ("se") CI | (−0.3014814653, 0.9607820563) |
| metafor knha PI (t₄·√(τ²+SE_HK²)) | (−0.8259169877, 1.4852175787) |
| DL pooled: μ̂, SE, CI | 0.3292782347, 0.2255687036, (−0.1128283004, 0.7713847698) |
| PM pooled: μ̂, SE, CI | 0.3279726969, 0.2196350770, (−0.1025041438, 0.7584495376) |
| Egger `lm(TE/seTE ~ 1/seTE)` | intercept 1.7601976623 (SE 3.8267673723), t 0.4599698625, df 3, p 0.6768255227; slope −0.2999545337 |
| REML log-likelihood | −2.873193453 |
| trim-and-fill (metafor L0) | k0 = 1 (left, SE 1.7009); μ̂ 0.249353 (−0.145565, 0.644271); τ² 0.109498 |

Second fixture (τ² = 0 path): y = [0.20, 0.25, 0.22, 0.18], v = [0.04, 0.05, 0.03, 0.045] → τ²_REML = 0 (HE start
< 0, first step-halved to 0, ll0 check idle), μ̂ = 0.2121546961, SE 0.0997233743, CI (0.0167004741, 0.4076089182), Q =
0.0573664825 (p 0.996408), I² = 0, metafor PI = CI, meta HTS PI (t₂ = 4.3027) = (−0.2169203525, 0.6412297448).

Third fixture (conversions, §2–3): all numbers in §2/§3 text (Wan/Luo/Shi n = 20 block; CI→SD n = 15; MD-CI →
s_p = 2.1391 with t₂₈ vs 2.2356 with z; t = 2.5, n = 15/18 → d = 0.8740074, SE 0.3668011; F(1,31) = 6.25 → same d;
p = .018 → d = 0.8733431; η²_p = .15 → d = 0.8176964; r = .40 → d 0.8728716 (equal n) / 0.8765010 (15/18)).

---

## 8. Implementation notes for `canopy/stats/`

* Modules: `effects.py` (SMD variants, conversions with `source` tags), `pool.py` (τ² estimators, pooled stats,
  PI/CI variants, HK), `hetero.py` (Q, I² both defs, Q-profile CI), `bias.py` (Egger lm/rma, trim-and-fill L0,
  optional selection model), `r_oracle.py` (Rscript bridge for tests), `report.py` (metagen-style print).
* Configuration object mirrors R names verbatim so the protocol JSON can say
  `{"sm":"SMD","method.tau":"REML","hakn":false,"method.predict":"HTS","method.I2":"Q","vtype":"rmd"}` and the
  forest-plot footer prints them; the default profile "cisneros-2024" = REML, hakn FALSE, HTS PI, meta I², d
  with the Rmd SE; profile "metafor" = REML, z PI, metafor I², g with LS.
* Use `float64` throughout, `scipy.special.gammaln`, `scipy.stats.t/norm/chi2`, `scipy.optimize.brentq`.
* Guard rails: k = 1 → no pooling; k = 2 → PI NaN for HTS; v_i ≤ 0 → error; Q = 0 → HK undefined (metafor sets
  s²_w = 1 when RSS ≤ eps → CI = classic); all-identical y → τ² = 0.
* Never round before pooling; the analysts' published table is rounded to 2–3 dp, so an end-to-end replication
  from *raw* extracted numbers will differ from the paper's pooled estimate at the 3rd decimal — the tolerance in
  the paper-replication test must be set from the paper's own rounding, not 1e-8.

---

## Sources

* metafor `rma.uni` reference (control defaults, Fisher scoring, step-halving, KNHA): https://wviechtb.github.io/metafor/reference/rma.uni.html
* metafor `predict.rma` (z default; t with k−p df under test="t"/"knha"; `pi.type="Riley"` k−2): https://wviechtb.github.io/metafor/reference/predict.rma.html
* metafor source, `rma.uni.r` (REML/ML/EB update, HE start, ll0 check, I²/H² via v_t): https://raw.githubusercontent.com/wviechtb/metafor/master/R/rma.uni.r
* metafor source, `escalc.r` (SMD `vtype` LS/LS2/UB/AV/H0): https://raw.githubusercontent.com/wviechtb/metafor/master/R/escalc.r
* metafor `.cmicalc` (exact J via lgamma): https://raw.githubusercontent.com/wviechtb/metafor/master/R/misc.func.hidden.escalc.r
* metafor `conv.fivenum` (Luo/Wan/Shi formulas, skew tests): https://wviechtb.github.io/metafor/reference/conv.fivenum.html and https://raw.githubusercontent.com/wviechtb/metafor/master/R/conv.fivenum.r
* meta package (CRAN; local help for 8.2-1: `method.predict` V/HTS/HK/HK-PR/KR/S, `method.tau.ci` J/QP, `adhoc.hakn.ci`): https://cran.r-project.org/package=meta and https://cran.r-project.org/web/packages/meta/meta.pdf
* Cochrane Handbook ch. 6 (SD from SE/CI/t/p, IQR/1.35, direction of scales): https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-06
* Cochrane Handbook ch. 10 (RE model, heterogeneity, prediction interval): https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10
* Viechtbauer W (2010) Conducting meta-analyses in R with the metafor package, JSS 36(3): https://www.jstatsoft.org/article/view/v036i03
* Viechtbauer W (2007) Confidence intervals for the amount of heterogeneity in meta-analysis (Q-profile), Stat Med 26:37–52: https://doi.org/10.1002/sim.2514
* Borenstein M, Hedges LV, Higgins JPT, Rothstein HR (2009) Introduction to Meta-Analysis, Wiley (eq. 4.18–4.24, 7.x, 12.17): https://doi.org/10.1002/9780470743386
* Hedges LV (1981) Distribution theory for Glass's estimator of effect size, J Educ Stat 6:107–128: https://doi.org/10.3102/10769986006002107
* Higgins JPT, Thompson SG (2002) Quantifying heterogeneity in a meta-analysis, Stat Med 21:1539–1558: https://doi.org/10.1002/sim.1186
* Higgins JPT, Thompson SG, Spiegelhalter DJ (2009) A re-evaluation of random-effects meta-analysis (PI with t_{k−2}), JRSS-A 172:137–159: https://doi.org/10.1111/j.1467-985X.2008.00552.x
* IntHout J, Ioannidis JPA, Borm GF (2014) The Hartung-Knapp-Sidik-Jonkman method…, BMC Med Res Methodol 14:25: https://doi.org/10.1186/1471-2288-14-25
* Knapp G, Hartung J (2003) Improved tests for a random effects meta-regression with a single covariate, Stat Med 22:2693–2710: https://doi.org/10.1002/sim.1482
* Veroniki AA et al. (2016) Methods to estimate the between-study variance and its uncertainty in meta-analysis, Res Synth Methods 7:55–79: https://doi.org/10.1002/jrsm.1164
* Veroniki AA et al. (2019) Methods to calculate uncertainty in the estimated overall effect size from a random-effects meta-analysis, Res Synth Methods 10:23–43: https://doi.org/10.1002/jrsm.1319
* Partlett C, Riley RD (2017) Random effects meta-analysis: coverage performance of 95% CI and PI…, Stat Med 36:301–317: https://doi.org/10.1002/sim.7140
* Lakens D (2013) Calculating and reporting effect sizes…, Front Psychol 4:863: https://doi.org/10.3389/fpsyg.2013.00863
* Wan X, Wang W, Liu J, Tong T (2014) Estimating the sample mean and SD from the sample size, median, range and/or IQR, BMC Med Res Methodol 14:135: https://doi.org/10.1186/1471-2288-14-135
* Luo D, Wan X, Liu J, Tong T (2018) Optimally estimating the sample mean from the sample size, median, mid-range, and/or mid-quartile range, Stat Methods Med Res 27:1785–1805: https://doi.org/10.1177/0962280216669183
* Shi J, Luo D, Weng H, et al. (2020) Optimally estimating the sample standard deviation from the five-number summary, Res Synth Methods 11:641–654: https://doi.org/10.1002/jrsm.1429 (preprint https://arxiv.org/abs/2003.02130)
* DerSimonian R, Laird N (1986) Meta-analysis in clinical trials, Control Clin Trials 7:177–188: https://doi.org/10.1016/0197-2456(86)90046-2
* Paule RC, Mandel J (1982) Consensus values and weighting factors, J Res NBS 87:377–385: https://doi.org/10.6028/jres.087.022
* Egger M et al. (1997) Bias in meta-analysis detected by a simple, graphical test, BMJ 315:629–634: https://doi.org/10.1136/bmj.315.7109.629
* Duval S, Tweedie R (2000) Trim and fill, Biometrics 56:455–463: https://doi.org/10.1111/j.0006-341X.2000.00455.x
* Vevea JL, Hedges LV (1995) A general linear model for estimating effect size in the presence of publication bias, Psychometrika 60:419–435: https://doi.org/10.1007/BF02294384 ; weightr package: https://cran.r-project.org/package=weightr ; metafor `selmodel`: https://wviechtb.github.io/metafor/reference/selmodel.html
* statsmodels `combine_effects` / `effectsize_smd`: https://www.statsmodels.org/stable/generated/statsmodels.stats.meta_analysis.combine_effects.html
* PyMARE: https://pymare.readthedocs.io/ (PyPI 0.0.10)
* PythonMeta: https://pypi.org/project/PythonMeta/

---

## Verification notes (fact-check)

Fact-checked 2026-08-15 by re-running the worked example against the locally installed R (metafor 4.6-0, meta 8.2-1,
R 4.4.0) and Python (statsmodels 0.14.6, SciPy 1.17.0), and against primary sources on the web. Script:
`scratchpad/chk*.R` (fact-check session).

| # | Claim | Verdict | Source |
|---|---|---|---|
| 1 | Local versions: metafor 4.6-0, meta 8.2-1, Rscript at `/usr/local/bin` | **Confirmed** (added note: CRAN now has metafor 5.0-1, 2026-04-26 and meta 8.5-0, 2026-05-25) | `packageVersion()`; https://cran.r-project.org/package=metafor ; https://cran.r-project.org/package=meta |
| 2 | Borenstein 2009 eq. 4.20 uses 2(N−2) in the second term of V_d | **Corrected** — eq. 4.20 is (n1+n2)/(n1n2) + d²/(2(n1+n2)); the 2(N−2) form is the Cisneros Rmd's, not Borenstein's | Borenstein et al. 2009, ch. 4 text (eq. 4.20), https://doi.org/10.1002/9780470743386 |
| 3 | metafor `vtype` LS = 1/n1+1/n2+g²/(2N); LS2 = J²(1/n1+1/n2+d²/(2N)); UB = 1/n1+1/n2+(1−(m−2)/(m J²))g²; exact J via lgamma with m ≤ 1 → NA | **Confirmed** (source comments cite Hedges 1982 eq. 8 / H&O 1985 eq. 15; Borenstein eq. 12.17; Hedges 1983 eq. 9) | https://raw.githubusercontent.com/wviechtb/metafor/master/R/escalc.r ; `metafor:::.cmicalc` |
| 4 | statsmodels `effectsize_smd`: J = 1−3/(4N−9), var = N/(n1n2)+g²/(2(N−3.94)); DL τ² 0.12125456; `iterated` ≈ PM to ~1e-6 | **Confirmed** (iterated τ² 0.10850739 vs metafor PM 0.10850577) | statsmodels 0.14.6 source (`inspect.getsource`) |
| 5 | Wan 2014 equation numbers (range eq. 9, IQR eq. 16) and ξ(n)/η(n) definitions; five-number = eq. 13 (added) | **Confirmed** | https://pmc.ncbi.nlm.nih.gov/articles/PMC4383202/ |
| 6 | Shi 2020 weight w = 1/(1+0.07n^0.6); Luo 2018 weights 4/(4+n^0.75), 0.7+0.39/n, 2.2/(2.2+n^0.75), 0.7−0.72/n^0.55; skew tests 1/ln(n+9)+2.5/(n+1) and 2.65/√n−6/n²; worked n=20 numbers (ξ 3.73648, η 1.25337, SD_Shi 4.3833, mean 10.5638 etc.) | **Confirmed** | metafor `conv.fivenum` source + `conv.fivenum()` output |
| 7 | Cochrane 6.5.2.2 (÷3.92 / 3.29 / 5.15, t for <60 per group), 6.5.2.3 (t/p for differences), 6.5.2.5 (IQR ≈ 1.35 SD), 6.5.1.2 (SMD) | **Confirmed** | https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-06 |
| 8 | metafor `rma.uni` control defaults: threshold 1e-5, maxiter 100, stepadj 1, tau2.min 0, tol eps^0.25, ll0check TRUE, tau2.max = max(100, 10·mad(yi)²) | **Confirmed** | `deparse(rma.uni)` (metafor 4.6-0) |
| 9 | meta default PI switched from HTS (k−2) to V (k−1) at version 7.1 | **Corrected** — the switch is in meta 8.0-0 (2024-10-30) per NEWS: "By default, prediction intervals are based on k − 1 instead of k − 2 degrees of freedom (Veroniki et al., 2019)". 7.0-0 (2024-01-11) still HTS | https://cran.r-project.org/web/packages/meta/news/news.html |
| 10 | meta `method.tau.ci` default: Jackson (J) for DL, Q-profile otherwise; `metabias` k.min = 10; `method.I2` "Q" default, "tau2" = metafor definition | **Confirmed** | meta 8.2-1 `?meta-package`, `formals(meta:::metabias.meta)$k.min`, `gs("method.predict")` = "V" |
| 11 | All §7 worked-example numbers (d, SE, g, LS/UB v, τ²_REML/DL/PM, SE(τ²), μ̂, SE, CI, I² both defs, H/H², Q-profile CI, PI HTS/V/S/knha, HK CI + adhoc, Egger lm intercept/t/p, logLik, trim-and-fill metafor vs meta, metacont seTE = √UB) and the τ²=0 second fixture | **Confirmed** to all printed digits | Re-run in R (metafor 4.6-0, meta 8.2-1) |
| 12 | PyPI latest: PyMARE 0.0.10, PythonMeta 1.26, statsmodels 0.14.6 | **Confirmed**; PyMARE REML numbers (0.12508276 / 0.32965022) **unverifiable** here (package not installed in this venv) | `pip index versions` |
| 13 | SciPy version "1.17" | **Confirmed** as installed (1.17.0); PyPI latest is 1.18.0 (annotated) | `pip index versions scipy` |
