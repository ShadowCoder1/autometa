"""Amendment A: `one_row_per_paper` AGGREGATES a paper's rows — it does not pick one of them.

The three designs and the three answers (controller ruling, fix round 1):

* independent samples inside one paper → a fixed-effect combination (weights 1/v, variance 1/Σw);
* the same participants across conditions/experiments → Borenstein's composite, whose variance
  carries `settings.within_paper_r`;
* a shared control arm → `resolve.apply_shared_control` already split it; nothing is aggregated.
"""
from __future__ import annotations

import csv
import math

import pytest

from canopy.models import Citation, EffectSizeRecord, StatsSettings
from canopy.pipeline.aggregate import (aggregate_one_row_per_paper, composite_variance,
                                       composite_row)
from canopy.stats.meta import fixed_effects


def _row(dataset_id: str, es: float, var: float, *, paper: str = "p1", sample_id: str = "",
         **kwargs) -> EffectSizeRecord:
    return EffectSizeRecord(
        paper_id=paper, cluster_id=paper, dataset_id=dataset_id, outcome_key="late_adaptation",
        label=dataset_id, route="text_mean_sd", n_a=12, n_b=12, es=es, d=es, var=var,
        se=math.sqrt(var), estimator="cohen", variance_method="hedges_olkin_df",
        confidence="auto_accept", sample_id=sample_id,
        citation=Citation(first_author="Bock", year=2005),
        conversion_chain=f"{dataset_id}: d from means", **kwargs)


def _settings(**kwargs) -> StatsSettings:
    return StatsSettings(one_row_per_paper=True, **kwargs)


# --------------------------------------------------------------------------- the two composites
def test_independent_rows_are_combined_as_a_fixed_effect():
    """Different participants in one paper: 1/v weights, variance 1/Σw — nothing hand-rolled."""
    rows = [_row("d1", -1.20, 0.16, sample_id="p1|exp:Experiment 1"),
            _row("d2", -0.40, 0.36, sample_id="p1|exp:Experiment 2")]
    result = aggregate_one_row_per_paper(rows, _settings())

    assert len(result.rows) == 1
    composite = result.rows[0]
    expected = fixed_effects([-1.20, -0.40], [0.16, 0.36])
    assert composite.es == pytest.approx(expected.estimate)
    assert composite.var == pytest.approx(1.0 / (1 / 0.16 + 1 / 0.36))
    assert composite.se == pytest.approx(math.sqrt(composite.var))
    assert composite.ci_low < composite.es < composite.ci_high
    assert "independent" in composite.conversion_chain
    assert "d1" in composite.conversion_chain and "d2" in composite.conversion_chain
    assert "independent_fixed_effect" in composite.flags
    assert composite.dataset_id == "d1+d2"
    assert composite.n_a == 24                       # different participants: the samples add


def test_dependent_rows_carry_the_within_paper_correlation():
    """The same participants twice: the composite is their mean and r inflates its variance."""
    rows = [_row("d1", -1.20, 0.16), _row("d2", -0.40, 0.36)]      # no sample_id → dependent
    result = aggregate_one_row_per_paper(rows, _settings(within_paper_r=0.5))

    composite = result.rows[0]
    expected_var = (0.16 + 0.36 + 2 * 0.5 * math.sqrt(0.16 * 0.36)) / 4
    assert composite.es == pytest.approx((-1.20 + -0.40) / 2)
    assert composite.var == pytest.approx(expected_var)
    assert "dependent_composite" in composite.flags
    assert "r = 0.5" in composite.conversion_chain
    assert composite.n_a == 12                       # the same 12 people, not 24

    independent = fixed_effects([-1.20, -0.40], [0.16, 0.36])
    assert composite.var > independent.se ** 2       # dependence costs precision, never buys it


def test_within_paper_r_is_the_protocols_setting_not_a_constant():
    rows = [_row("d1", -1.0, 0.20), _row("d2", -0.5, 0.20)]
    at_zero = aggregate_one_row_per_paper(rows, _settings(within_paper_r=0.0)).rows[0]
    at_one = aggregate_one_row_per_paper(rows, _settings(within_paper_r=1.0)).rows[0]
    assert at_zero.var == pytest.approx((0.20 + 0.20) / 4)      # the mean of two independent rows
    assert at_one.var == pytest.approx(0.20)                    # perfectly correlated: one row
    assert at_one.var > at_zero.var


def test_composite_variance_is_the_borenstein_formula():
    assert composite_variance([0.25], 0.5) == pytest.approx(0.25)
    assert composite_variance([0.25, 0.25], 0.0) == pytest.approx(0.125)
    assert composite_variance([0.25, 0.25], 1.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        composite_variance([0.25, 0.0], 0.5)


# --------------------------------------------------------------------------- what is not aggregated
def test_a_paper_that_contributed_one_row_is_left_alone():
    rows = [_row("d1", -1.0, 0.2), _row("d9", -0.3, 0.2, paper="p2")]
    result = aggregate_one_row_per_paper(rows, _settings())
    assert [r.dataset_id for r in result.rows] == ["d1", "d9"]
    assert result.superseded == [] and result.exclusions == []


def test_shared_control_rows_are_split_by_the_protocol_not_aggregated():
    """`apply_shared_control` already handled the dependence; aggregating on top would double it."""
    rows = [_row("d1", -1.0, 0.2, flags=["shared_control_split"]),
            _row("d2", -0.5, 0.2, flags=["shared_control_split"])]
    result = aggregate_one_row_per_paper(rows, _settings())
    assert [r.dataset_id for r in result.rows] == ["d1", "d2"]
    assert result.superseded == []
    assert any("share a control arm" in note for note in result.notes)


def test_a_row_with_no_usable_effect_size_is_superseded_not_silently_dropped():
    rows = [_row("d1", -1.0, 0.2), _row("d2", -0.5, 0.2),
            _row("d3", -0.5, 0.2).model_copy(update={"es": None, "var": None})]
    result = aggregate_one_row_per_paper(rows, _settings())
    reasons = {e["dataset_id"]: e["reason"] for e in result.exclusions}
    assert reasons["d3"] == "superseded_by_dataset_rule"
    assert reasons["d1"].startswith("aggregated_into:")


# --------------------------------------------------------------------------- the paper trail
def test_every_aggregated_row_reaches_the_exclusions_table(tmp_path):
    from canopy.report.tables import EXCLUSION_REASONS, exclusions_table

    rows = [_row("d1", -1.2, 0.16), _row("d2", -0.4, 0.36)]
    result = aggregate_one_row_per_paper(rows, _settings())
    assert [e["reason"] for e in result.exclusions] == ["aggregated_into:d1+d2"] * 2
    assert [r.dataset_id for r in result.superseded] == ["d1", "d2"]

    out = exclusions_table(result.exclusions, tmp_path / "exclusions")
    written = list(csv.DictReader(out["csv"].open(newline="", encoding="utf-8")))
    assert [r["reason"] for r in written] == ["aggregated", "aggregated"]
    assert all(r["reason_as_given"] == "aggregated_into:d1+d2" for r in written)
    assert all("composite" in r["detail"] or "independent" in r["detail"] for r in written)
    assert "aggregated" in EXCLUSION_REASONS


def test_composite_needs_at_least_two_rows():
    with pytest.raises(ValueError):
        composite_row([_row("d1", -1.0, 0.2)], _settings(), dependent=True)


# --------------------------------------------------------------------------- the profiles
def test_profiles_carry_the_published_analysis_choice():
    """Cisneros 2024 pooled all 50 datasets; `metafor` keeps one row per paper."""
    from canopy.protocol import apply_profile

    assert apply_profile(StatsSettings(profile="cisneros2024")).one_row_per_paper is False
    assert apply_profile(StatsSettings(profile="metafor")).one_row_per_paper is True
    # a protocol that says so explicitly still wins over its profile
    explicit = apply_profile(StatsSettings(profile="cisneros2024", one_row_per_paper=True))
    assert explicit.one_row_per_paper is True
