"""Effect-size + conversion tests, checked against R (meta 8.2 / metafor 4.6) fixtures."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from canopy.stats import effect_sizes as es

FIX = json.loads((Path(__file__).parent.parent / "validation/fixtures/r_reference.json").read_text())


@pytest.mark.parametrize("case", FIX["smd_cases"])
def test_cohens_d_matches_meta_metacont(case):
    d = es.cohens_d(case["m1"], case["sd1"], case["n1"], case["m2"], case["sd2"], case["n2"])
    assert d == pytest.approx(case["metacont_cohen_TE"], abs=1e-9)
    assert d == pytest.approx(case["escalc_d_yi"], abs=1e-9)


@pytest.mark.parametrize("case", FIX["smd_cases"])
def test_hedges_g_matches_metafor_escalc(case):
    d = es.cohens_d(case["m1"], case["sd1"], case["n1"], case["m2"], case["sd2"], case["n2"])
    g = es.hedges_g(d, case["n1"], case["n2"])  # exact gamma-based J (metafor / meta default)
    assert g == pytest.approx(case["escalc_g_yi"], abs=1e-9)
    assert es.hedges_g(d, case["n1"], case["n2"], exact=False) == pytest.approx(
        d * (1 - 3 / (4 * (case["n1"] + case["n2"] - 2) - 1)), abs=1e-12)


@pytest.mark.parametrize("case", FIX["smd_cases"])
def test_variance_conventions(case):
    n1, n2 = case["n1"], case["n2"]
    d = es.cohens_d(case["m1"], case["sd1"], n1, case["m2"], case["sd2"], n2)
    g = es.hedges_g(d, n1, n2)
    # Borenstein / metafor correct=FALSE: (n1+n2)/(n1 n2) + d²/(2(n1+n2))
    assert es.var_smd(d, n1, n2, method="borenstein") == pytest.approx(case["escalc_d_vi"], abs=1e-9)
    # metafor default for g: same shape with g
    assert es.var_smd(g, n1, n2, method="borenstein") == pytest.approx(case["escalc_g_vi"], abs=1e-9)
    # meta::metacont Cohen exact.smd=TRUE
    assert math.sqrt(es.var_smd(d, n1, n2, method="meta_exact")) == pytest.approx(case["metacont_cohen_seTE"], abs=1e-9)
    # meta::metacont Hedges exact.smd=TRUE (uses g)
    assert math.sqrt(es.var_smd(g, n1, n2, method="meta_exact_g")) == pytest.approx(case["metacont_hedges_exact_seTE"], abs=1e-9)


def test_hedges_olkin_df_variance_reproduces_cisneros_notebook():
    # Bock 2005 (Late_Adaptation_v2.ipynb): d=-1.6757675, n=12/12 → SE 0.480093033465216
    d = -1.6757675199941424
    v = es.var_smd(d, 12, 12, method="hedges_olkin_df")
    assert math.sqrt(v) == pytest.approx(0.480093033465216, abs=1e-9)


def test_sd_conversions():
    assert es.sd_from_se(2.0, 16) == pytest.approx(8.0)
    # 95% CI half-width 3.92 for n=100 → SE=2 → SD=20 (z multiplier)
    assert es.sd_from_ci(10 - 3.92, 10 + 3.92, 100, dist="z") == pytest.approx(20.0, rel=1e-3)
    # t multiplier for small n is wider than z, so implied SD is smaller
    sd_z = es.sd_from_ci(0, 4, 8, dist="z")
    sd_t = es.sd_from_ci(0, 4, 8, dist="t")
    assert sd_t < sd_z
    # IQR → SD (Wan et al. 2014): for large n approaches IQR/1.349
    assert es.sd_from_iqr(0, 1.349, 10_000) == pytest.approx(1.0, rel=2e-3)
    # range → SD (Wan 2014): for n=25 ≈ range/3.93 (Hozo-ish); just check monotone/positive
    assert 0 < es.sd_from_range(0, 10, 25) < 10 / 3
    # points → mean/sd (sample SD, ddof=1)
    m, sd, n = es.mean_sd_from_points([1, 2, 3, 4])
    assert (m, n) == (2.5, 4) and sd == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    # CI of a mean difference → pooled SD
    # diff CI [1.0, 5.0], n=20/20 → SE_diff = 4/3.92 → sd_pooled = SE_diff / sqrt(1/20+1/20)
    sd_p = es.pooled_sd_from_diff_ci(1.0, 5.0, 20, 20, dist="z")
    assert sd_p == pytest.approx((4 / 3.919928) / math.sqrt(0.1), rel=1e-4)


@pytest.mark.parametrize("case", FIX["t_cases"])
def test_d_from_t(case):
    d = es.d_from_t(case["t"], case["n1"], case["n2"])
    assert d == pytest.approx(case["d_raw"], abs=1e-9)
    g = es.hedges_g(d, case["n1"], case["n2"])
    assert g == pytest.approx(case["escalc_g_yi"], abs=1e-9)
    assert es.var_smd(g, case["n1"], case["n2"], "borenstein") == pytest.approx(case["escalc_g_vi"], abs=1e-9)


@pytest.mark.parametrize("case", FIX["p_cases"])
def test_d_from_p(case):
    t = es.t_from_p(case["p"], case["n1"] + case["n2"] - 2, two_tailed=True)
    assert t == pytest.approx(case["t_from_p"], abs=1e-9)
    d = es.d_from_t(t, case["n1"], case["n2"])
    g = es.hedges_g(d, case["n1"], case["n2"])
    assert g == pytest.approx(case["escalc_g_yi"], abs=1e-9)


def test_d_from_f_and_sign():
    # F(1,df) between-subjects with two groups: F = t² → d = sqrt(F*(n1+n2)/(n1 n2)); sign from direction
    n1 = n2 = 10
    d_pos = es.d_from_f(11.4, n1, n2, direction=+1)
    assert d_pos == pytest.approx(math.sqrt(11.4 * (20 / 100)))
    assert es.d_from_f(11.4, n1, n2, direction=-1) == pytest.approx(-d_pos)
    with pytest.raises(ValueError):
        es.d_from_f(11.4, n1, n2, direction=0)


def test_orientation():
    # a positive raw difference on an ERROR measure means WORSE performance → flip
    assert es.orient(0.5, higher_is_better=False) == -0.5
    assert es.orient(0.5, higher_is_better=True) == 0.5


def test_ci_for_d():
    lo, hi = es.ci_smd(0.5, 0.2, level=0.95)
    assert lo == pytest.approx(0.5 - 1.959964 * 0.2, abs=1e-5)
    assert hi == pytest.approx(0.5 + 1.959964 * 0.2, abs=1e-5)


def test_compute_smd_high_level():
    """The one-call API used by the pipeline: returns d, g, variance and provenance of the route."""
    r = es.smd_from_means(m_a=31.51, sd_a=11.12, n_a=12, m_b=12.28, sd_b=11.82, n_b=12,
                          higher_is_better=False, estimator="cohen", variance="meta_exact")
    # group A (older) has HIGHER error → worse → negative oriented d
    assert r.d == pytest.approx(-1.675767519994, abs=1e-9)
    assert r.se == pytest.approx(0.4809018707246, abs=1e-9)
    assert r.route == "means_sd"
    assert r.ci_low < r.d < r.ci_high
