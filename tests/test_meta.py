"""Random-effects meta-analysis tests vs R fixtures (metafor::rma and meta::metagen)."""
import json
from pathlib import Path

import numpy as np
import pytest

from canopy.stats.meta import (
    egger_test,
    fixed_effects,
    random_effects,
)

FIX = json.loads((Path(__file__).parent.parent / "validation/fixtures/r_reference.json").read_text())
SYN = FIX["synthetic_k5"]


def _syn():
    return np.array(SYN["yi"]), np.array(SYN["vi"])


@pytest.mark.parametrize("method", ["REML", "DL", "PM"])
def test_random_effects_matches_metafor(method):
    yi, vi = _syn()
    ref = SYN[method]
    r = random_effects(yi, vi, method=method)
    assert r.estimate == pytest.approx(ref["b"], abs=1e-6)
    assert r.se == pytest.approx(ref["se"], abs=1e-6)
    assert r.ci_low == pytest.approx(ref["ci.lb"], abs=1e-5)
    assert r.ci_high == pytest.approx(ref["ci.ub"], abs=1e-5)
    assert r.z == pytest.approx(ref["zval"], abs=1e-4)
    assert r.p == pytest.approx(ref["pval"], abs=1e-5)
    assert r.tau2 == pytest.approx(ref["tau2"], abs=1e-5)
    assert r.Q == pytest.approx(ref["QE"], abs=1e-6)
    assert r.Q_p == pytest.approx(ref["QEp"], abs=1e-6)
    assert r.I2_tau == pytest.approx(ref["I2"] / 100, abs=1e-5)  # metafor: tau²-based I²
    assert r.H2 == pytest.approx(ref["H2"], abs=1e-4)
    np.testing.assert_allclose(r.weights_pct, ref["weights_pct"], atol=1e-4)
    # metafor predict(): z-based prediction interval
    assert r.pi_low_z == pytest.approx(ref["pi.lb"], abs=1e-5)
    assert r.pi_high_z == pytest.approx(ref["pi.ub"], abs=1e-5)


def test_random_effects_matches_meta_metagen_conventions():
    """meta::metagen: I² = (Q-df)/Q (DL-style, tau-estimator independent); PI uses t(k-2) (HTS)."""
    yi, vi = _syn()
    ref = SYN["meta_REML"]
    r = random_effects(yi, vi, method="REML")
    assert r.estimate == pytest.approx(ref["TE.random"], abs=1e-6)
    assert r.I2 == pytest.approx(ref["I2"], abs=1e-6)
    # metafor/meta iterate REML to |Δτ²|<1e-5; we iterate to 1e-8, so agree to ~1e-5
    assert r.pi_low == pytest.approx(ref["lower.predict"], abs=2e-5)
    assert r.pi_high == pytest.approx(ref["upper.predict"], abs=2e-5)
    assert r.pi_df == ref["df.predict"]
    fe = fixed_effects(yi, vi)
    assert fe.estimate == pytest.approx(ref["TE.common"], abs=1e-6)
    assert fe.se == pytest.approx(ref["seTE.common"], abs=1e-6)


def test_fixed_effects_matches_metafor_FE():
    yi, vi = _syn()
    ref = SYN["FE"]
    fe = fixed_effects(yi, vi)
    assert fe.estimate == pytest.approx(ref["b"], abs=1e-6)
    assert fe.se == pytest.approx(ref["se"], abs=1e-6)
    np.testing.assert_allclose(fe.weights_pct, ref["weights_pct"], atol=1e-5)


@pytest.mark.parametrize("key", ["cisneros_late", "cisneros_aft"])
def test_cisneros_gold_pooling(key):
    """Reproduce Figures 3/4 of Cisneros et al. from the human TE/seTE table."""
    fx = FIX[key]
    yi = np.array(fx["TE"]); sei = np.array(fx["seTE"])
    ref = fx["REML"]
    r = random_effects(yi, sei ** 2, method="REML")
    assert r.k == fx["k"]
    assert r.estimate == pytest.approx(ref["TE.random"], abs=1e-6)
    assert r.se == pytest.approx(ref["seTE.random"], abs=1e-6)
    assert r.ci_low == pytest.approx(ref["lower.random"], abs=1e-6)
    assert r.ci_high == pytest.approx(ref["upper.random"], abs=1e-6)
    assert r.tau2 == pytest.approx(ref["tau2"], abs=1e-5)
    assert r.Q == pytest.approx(ref["Q"], abs=1e-5)
    assert r.I2 == pytest.approx(ref["I2"], abs=1e-6)
    assert r.pi_low == pytest.approx(ref["lower.predict"], abs=2e-5)
    assert r.pi_high == pytest.approx(ref["upper.predict"], abs=2e-5)
    np.testing.assert_allclose(r.weights_pct, ref["w.random.pct"], atol=1e-4)
    # Hartung-Knapp
    rk = random_effects(yi, sei ** 2, method="REML", hakn=True)
    mk = fx["metafor_REML_knha"]
    assert rk.se == pytest.approx(mk["se"], abs=1e-5)
    assert rk.ci_low == pytest.approx(mk["ci.lb"], abs=2e-5)
    assert rk.ci_high == pytest.approx(mk["ci.ub"], abs=2e-5)
    assert rk.p == pytest.approx(mk["pval"], abs=1e-5)
    # Egger regression as in the Cisneros Rmd
    eg = egger_test(yi, sei)
    assert eg.intercept == pytest.approx(fx["egger_rmd"]["intercept"], abs=1e-6)
    assert eg.p == pytest.approx(fx["egger_rmd"]["intercept_p"], abs=1e-6)


def test_tau2_ci_q_profile():
    yi, vi = _syn()
    r = random_effects(yi, vi, method="REML", tau2_ci=True)
    ref = SYN["tau2_ci"]
    assert r.tau2_ci_low == pytest.approx(ref["lb"], abs=1e-4)
    assert r.tau2_ci_high == pytest.approx(ref["ub"], abs=1e-3)


def test_degenerate_inputs():
    with pytest.raises(ValueError):
        random_effects(np.array([0.1]), np.array([0.01]))  # k < 2
    r = random_effects(np.array([0.2, 0.2, 0.2]), np.array([0.01, 0.02, 0.03]))
    assert r.tau2 == 0 and r.I2 == 0


def test_prediction_interval_conventions():
    from canopy.stats.meta import prediction_interval
    yi, vi = _syn()
    r = random_effects(yi, vi, method="REML")
    lo, hi, df = prediction_interval(r, "V")      # meta 8.x default: t(k-1)
    assert (lo, hi, df) == pytest.approx((-0.885220, 1.180695, 4), abs=2e-5)
    lo, hi, df = prediction_interval(r, "HTS")    # t(k-2)
    assert (lo, hi) == pytest.approx((-1.036270, 1.331745), abs=2e-5) and df == 3
    lo, hi, df = prediction_interval(r, "z")      # metafor / meta 'S'
    assert (lo, hi) == pytest.approx((-0.581454, 0.876928), abs=2e-5)


def test_hartung_knapp_with_identical_rows_pools_instead_of_dividing_by_zero():
    """Controller ruling. `var_hk = Sum w (y - mu)^2 / ((k-1) Sum w)` is exactly 0 when every row
    sits on the pooled estimate, and `mu / se_used` then raised `ZeroDivisionError` — at the
    POOLING step, after every paper in the run had already been paid for.

    Two identical effect sizes is not contrived: two papers reporting the same `d` do it, and
    `hakn: true` is what the validation protocol sets. Zero between-study spread is not infinite
    precision, it is an adjustment with nothing to adjust, so the ordinary random-effects standard
    error is used and the fallback is named on the result.
    """
    import numpy as np

    from canopy.stats.meta import random_effects

    yi = np.array([-0.5, -0.5, -0.5])
    vi = np.array([0.04, 0.04, 0.04])
    result = random_effects(yi, vi, method="REML", hakn=True)     # must not raise

    assert result.hakn_fallback == "se_used was zero (all rows identical); standard SE used"
    assert result.as_dict()["hakn_fallback"] == result.hakn_fallback   # it reaches pooled.json
    assert result.estimate == pytest.approx(-0.5)
    plain = random_effects(yi, vi, method="REML", hakn=False)
    assert result.se == pytest.approx(plain.se)                   # the standard RE SE, as ruled
    assert (result.ci_low, result.ci_high) == pytest.approx((plain.ci_low, plain.ci_high))
    assert np.isfinite(result.p) and np.isfinite(result.z)
    assert result.hakn is True                                    # what was ASKED for is recorded


def test_an_ordinary_hartung_knapp_pool_carries_no_fallback_note():
    """The negative control: the note appears only when the adjustment could not be applied, so a
    reader can trust its absence."""
    import numpy as np

    from canopy.stats.meta import random_effects

    yi = np.array([-0.5, -0.2, -0.9])
    vi = np.array([0.04, 0.05, 0.06])
    adjusted = random_effects(yi, vi, method="REML", hakn=True)
    plain = random_effects(yi, vi, method="REML", hakn=False)

    assert adjusted.hakn_fallback == ""
    assert plain.hakn_fallback == ""
    assert adjusted.se != pytest.approx(plain.se)     # the adjustment really was applied


# --------------------------------------------------------------------- cluster-robust (RVE)
# Golden numbers: authentic robumeta 2.1 outputs, computed once by sourcing the CRAN tarball's
# R/robu.R under R 4.4.0 (no package install; robumeta has no dependencies beyond base R):
#   robu(y ~ 1, data=d, studynum=cl, var.eff.size=v, modelweights="CORR", rho=<rho>, small=TRUE)
# and independently reproduced to <= 1e-13 by a from-scratch numpy transcription of the design
# formulas (design docs: scratchpad rve_design/, adversarial-review verified). Tests need no R.

F1_Y = [0.10, 0.30, 0.50, 0.20, 0.40, 0.60, -0.10, 0.15, 0.45]
F1_V = [0.04, 0.05, 0.06, 0.08, 0.10, 0.05, 0.07, 0.03, 0.09]
F1_C = ["A", "A", "A", "B", "B", "C", "D", "D", "E"]          # 5 clusters, sizes 3/2/1/2/1


def test_cluster_robust_matches_robumeta_f1():
    """Fixture F1 at rho=0.8: every reported quantity equals robumeta 2.1's own output."""
    from canopy.stats.meta import cluster_robust

    r = cluster_robust(F1_Y, F1_V, F1_C, rho=0.8)
    assert r.estimate == pytest.approx(0.327006697455228, abs=1e-12)
    assert r.se == pytest.approx(0.113902213196822, abs=1e-12)
    assert r.z == pytest.approx(2.87094243629984, rel=1e-10)
    assert r.df_robust == pytest.approx(3.72857069735625, rel=1e-10)
    assert r.p == pytest.approx(0.0493825783827924, abs=1e-8)
    assert r.ci_low == pytest.approx(0.00147395971467, abs=1e-10)
    assert r.ci_high == pytest.approx(0.652539435195791, abs=1e-10)
    assert r.tau2 == pytest.approx(0.00606703066914499, abs=1e-12)
    assert 100 * r.I2 == pytest.approx(8.78639265795739, rel=1e-10)
    assert r.Q == pytest.approx(4.46234984984985, rel=1e-10)
    assert r.Q_df == pytest.approx(4.07027027027027, rel=1e-10)   # NON-integer by design
    assert np.isnan(r.Q_p) and np.isnan(r.I2_tau) and np.isnan(r.H2)   # robumeta defines none
    assert r.robust and r.robust_requested and r.n_clusters == 5 and r.k == 9
    assert r.robust_small_sample                                  # df 3.73 < 4 fires the warning
    assert sum(r.weights_pct) == pytest.approx(100.0)


def test_cluster_robust_rho_enters_only_additively():
    """The rho sweep of fixture F4, plus the df_Q = N−1 identity at rho = 1 (termA+termB = 1)."""
    from canopy.stats.meta import cluster_robust

    for rho, b, tau2 in ((0.0, 0.326547538498597, 0.00171758828996284),
                         (0.5, 0.326840480348116, 0.00443598977695168),
                         (1.0, 0.327113797047426, 0.00715439126394053)):
        r = cluster_robust(F1_Y, F1_V, F1_C, rho=rho)
        assert r.estimate == pytest.approx(b, abs=1e-12), rho
        assert r.tau2 == pytest.approx(tau2, abs=1e-12), rho
    assert cluster_robust(F1_Y, F1_V, F1_C, rho=1.0).Q_df == pytest.approx(4.0, abs=1e-12)


def test_cluster_robust_two_clusters_df_is_exactly_one():
    """Fixture F2: with N = 2 clusters the Satterthwaite df is 1 for ANY weights (identity F6a),
    and every golden quantity still matches robumeta."""
    from canopy.stats.meta import cluster_robust

    r = cluster_robust([0.20, 0.50, 0.10, 0.30, 0.60], [0.05, 0.04, 0.06, 0.05, 0.08],
                       ["A", "A", "B", "B", "B"], rho=0.8)
    assert r.df_robust == 1.0                                      # exact, not approx
    assert r.estimate == pytest.approx(0.343002915451895, abs=1e-12)
    assert r.se == pytest.approx(0.00822550202895972, abs=1e-12)
    assert r.p == pytest.approx(0.0152637587963100, abs=1e-8)
    assert r.tau2 == pytest.approx(0.003, abs=1e-14)
    assert r.ci_low == pytest.approx(0.238488002614113, abs=1e-10)
    assert r.ci_high == pytest.approx(0.447517828289677, abs=1e-10)
    assert r.robust_small_sample


def test_cluster_robust_singletons_collapse_to_dl_exactly():
    """Fixture F3b: all-singleton clusters make the CORR MoM tau² EXACTLY DerSimonian–Laird and
    the estimate exactly the DL random-effects estimate — while the SE stays the CR2 robust one
    (assert it DIFFERS, to catch an implementation that quietly falls through to the model SE).
    rho must vanish entirely on singletons."""
    from canopy.stats.meta import cluster_robust, random_effects, tau2_DL

    y = [0.05, 0.60, -0.20, 0.45, 0.90, 0.10]
    v = [0.03, 0.04, 0.05, 0.02, 0.06, 0.035]
    labels = [f"s{i}" for i in range(6)]
    r = cluster_robust(y, v, labels, rho=0.8)
    dl = random_effects(y, v, method="DL")
    assert r.tau2 == pytest.approx(tau2_DL(np.asarray(y), np.asarray(v)), abs=1e-12)
    assert r.tau2 == pytest.approx(0.0896141021470397, abs=1e-12)
    assert r.estimate == pytest.approx(dl.estimate, abs=1e-12)
    assert r.estimate == pytest.approx(0.308859389164304, abs=1e-12)
    assert r.se == pytest.approx(0.151774172311562, abs=1e-12)
    assert abs(r.se - dl.se) > 1e-3
    assert r.df_robust == pytest.approx(4.94195533407425, rel=1e-10)
    assert not r.robust_small_sample                               # df 4.94 — no warning
    r0 = cluster_robust(y, v, labels, rho=0.0)
    assert (r0.estimate, r0.se, r0.df_robust) == (r.estimate, r.se, r.df_robust)


def test_cluster_robust_estimate_differs_from_the_independent_pool():
    """The full CORR fit is a DIFFERENT estimator, not a corrected SE on the same estimate:
    on F1 the two point estimates differ in the second decimal (adversarial review B1)."""
    from canopy.stats.meta import cluster_robust, random_effects

    robust = cluster_robust(F1_Y, F1_V, F1_C, rho=0.8)
    independent = random_effects(F1_Y, F1_V, method="REML")
    assert abs(robust.estimate - independent.estimate) > 0.05


def test_cluster_robust_invariant_to_row_order_and_relabeling():
    from canopy.stats.meta import cluster_robust

    base = cluster_robust(F1_Y, F1_V, F1_C, rho=0.8)
    order = [8, 2, 5, 0, 7, 1, 4, 3, 6]
    shuffled = cluster_robust([F1_Y[i] for i in order], [F1_V[i] for i in order],
                              [F1_C[i] + "_renamed" for i in order], rho=0.8)
    for name in ("estimate", "se", "df_robust", "tau2", "p", "ci_low", "ci_high"):
        assert getattr(shuffled, name) == pytest.approx(getattr(base, name), abs=1e-12), name


def test_cluster_robust_refuses_a_single_cluster_and_bad_rho():
    from canopy.stats.meta import cluster_robust

    with pytest.raises(ValueError, match="at least 2 clusters"):
        cluster_robust([0.1, 0.2], [0.05, 0.05], ["A", "A"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        cluster_robust(F1_Y, F1_V, F1_C, rho=1.5)


def test_random_effects_records_the_single_cluster_fallback_instead_of_raising():
    """The hakn_fallback precedent: a paid run must not die at the pooling step. One cluster →
    the ordinary independent pool, with the reason on the record for every report surface."""
    from canopy.stats.meta import random_effects

    result = random_effects([0.1, 0.2], [0.05, 0.05], clusters=["A", "A"], rho=0.8)
    plain = random_effects([0.1, 0.2], [0.05, 0.05])
    assert result.robust_requested and not result.robust
    assert "fewer than 2 clusters" in result.robust_fallback
    assert result.estimate == pytest.approx(plain.estimate)
    assert result.se == pytest.approx(plain.se)
    assert result.as_dict()["robust_fallback"] == result.robust_fallback


def test_random_effects_without_clusters_is_bit_identical_to_before():
    """clusters=None is contractually a no-op: the whole as_dict must match a no-arg call."""
    from canopy.stats.meta import random_effects

    a = random_effects(F1_Y, F1_V, method="REML", hakn=True).as_dict()
    b = random_effects(F1_Y, F1_V, method="REML", hakn=True, clusters=None).as_dict()
    assert a == b
    assert a["robust"] is False and a["robust_requested"] is False


def test_random_effects_refuses_hakn_with_clusters():
    from canopy.stats.meta import random_effects

    with pytest.raises(ValueError, match="mutually exclusive"):
        random_effects(F1_Y, F1_V, hakn=True, clusters=F1_C)


def test_cluster_robust_leave_one_out_drops_clusters():
    from canopy.stats.meta import leave_one_out

    rows = leave_one_out(F1_Y, F1_V, clusters=F1_C, rho=0.8)
    assert len(rows) == 5                                          # one per cluster, not per row
    assert {r.m for r in rows} == {4}                              # 4 clusters remain each time
    assert rows[0].k == 6                                          # cluster A had 3 of 9 rows
    with pytest.raises(ValueError, match="at least 3 clusters"):
        leave_one_out([0.1, 0.2, 0.3], [0.04, 0.04, 0.04], clusters=["A", "A", "B"])


def test_cluster_robust_prediction_interval_counts_clusters():
    from canopy.stats.meta import cluster_robust, prediction_interval

    r = cluster_robust(F1_Y, F1_V, F1_C, rho=0.8)
    low, high, df = prediction_interval(r, "HTS")
    assert df == 3                                                 # m−2 over clusters, not k−2=7
    assert (low, high) == (r.pi_low, r.pi_high)
    _, _, df_v = prediction_interval(r, "V")
    assert df_v == 4


def test_funnel_data_center_overrides_the_internal_pool():
    from canopy.stats.meta import funnel_data

    data = funnel_data(F1_Y, F1_V, center=0.327006697455228)
    assert data["estimate"] == pytest.approx(0.327006697455228)
    default = funnel_data(F1_Y, F1_V)
    assert abs(default["estimate"] - data["estimate"]) > 0.05      # the override really bites
