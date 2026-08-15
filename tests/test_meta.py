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
