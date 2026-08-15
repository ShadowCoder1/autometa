"""Task 9: resolved values in, one `EffectSizeRecord` out.

This is the only place a study's numbers become an effect size, and it is all code: the route is
chosen by the protocol's precedence list, every conversion is a named formula from
`canopy.stats`, and the chain of arithmetic is written out with its numbers so a reviewer can
check it without rerunning anything.

The reference value is Bock 2005 as the published review scored it: TE −1.676, seTE 0.4801
(`validation/reference/cisneros2024/late_gsheet.csv`), from 31.51 ± 11.12 (n = 12, older) versus
12.28 ± 11.82 (n = 12, younger) on a measure where a larger raw value means MORE error.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from canopy.models import (DatasetSpec, DispersionType, GroupSpec, OutcomeDef, StatsSettings,
                           Verdict)
from canopy.pipeline.resolve import (GroupValues, ReportedValues, ResolvedValues, StatisticValues,
                                     apply_shared_control, available_routes, multi_group_flags,
                                     resolve_effect)
from canopy.stats import effect_sizes as es

GOLD = Path(__file__).parent.parent / "validation/reference/cisneros2024/late_gsheet.csv"

LATE = OutcomeDef(key="late_adaptation", label="Late adaptation",
                  definition="Direction error over the final adaptation block.",
                  units_hint="degrees")


def dataset(**kwargs) -> DatasetSpec:
    return DatasetSpec(dataset_id="b511dbb76fa6:d1", cluster_id="b511dbb76fa6", label="pointing",
                       group_a=GroupSpec(label="old subjects", n=12),
                       group_b=GroupSpec(label="young subjects", n=12),
                       moderators={"task": "VMA"}, **kwargs)


def bock_values(**kwargs) -> ResolvedValues:
    base = dict(
        dataset_id="b511dbb76fa6:d1", outcome_key="late_adaptation", higher_is_better=False,
        group_a=GroupValues(n=12, mean=31.51, dispersion_value=11.12,
                            dispersion_type=DispersionType.SD, unit="deg", route="text"),
        group_b=GroupValues(n=12, mean=12.28, dispersion_value=11.82,
                            dispersion_type=DispersionType.SD, unit="deg", route="text"),
        confidence="auto_accept")
    base.update(kwargs)
    return ResolvedValues(**base)


def gold_row(author="Bock "):
    with GOLD.open() as handle:
        for row in csv.DictReader(handle):
            if row["Author"] == author:
                return row
    raise AssertionError(f"no gold row for {author!r}")


# --------------------------------------------------------------------------- the reference row
def test_the_means_route_reproduces_the_published_effect_for_bock():
    gold = gold_row()
    record = resolve_effect(dataset(), LATE, bock_values(),
                            StatsSettings(estimator="cohen", variance="hedges_olkin_df"))
    assert record.route == "text_mean_sd"
    assert record.d == pytest.approx(-1.6757675, abs=1e-6)
    assert record.d == pytest.approx(float(gold["TE"]), abs=5e-4)
    assert record.es == record.d
    assert record.se == pytest.approx(float(gold["seTE"]), abs=1e-4)
    assert record.ci_low == pytest.approx(float(gold["CI_low"]), abs=1e-3)
    assert record.ci_high == pytest.approx(float(gold["CI_high"]), abs=1e-3)
    assert record.n_a == 12 and record.n_b == 12
    assert record.higher_is_better is False and record.orientation_applied is True
    assert record.not_convertible_reason == ""


def test_the_record_carries_the_dataset_identity():
    record = resolve_effect(dataset(), LATE, bock_values(), StatsSettings())
    assert record.dataset_id == "b511dbb76fa6:d1" and record.cluster_id == "b511dbb76fa6"
    assert record.outcome_key == "late_adaptation" and record.label == "pointing"
    assert record.moderators == {"task": "VMA"}
    assert record.confidence == "auto_accept"
    assert record.estimator == "cohen" and record.variance_method == "borenstein"
    assert record.level == 0.95


def test_the_chain_writes_out_the_arithmetic_with_its_numbers():
    record = resolve_effect(dataset(), LATE, bock_values(), StatsSettings())
    chain = record.conversion_chain
    assert "31.51" in chain and "12.28" in chain
    assert "11.4" in chain or "11.47" in chain          # the pooled SD
    assert "-1.6758" in chain or "1.6758" in chain
    assert "oriented" in chain.lower()
    assert record.conversion_steps and chain == "; ".join(record.conversion_steps)
    assert record.inputs["mean_a"] == 31.51 and record.inputs["sd_b"] == 11.82


def test_an_unoriented_measure_is_never_signed():
    record = resolve_effect(dataset(), LATE, bock_values(higher_is_better=None), StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "direction" in record.not_convertible_reason
    assert "orientation_unresolved" in record.flags


def test_hedges_g_is_used_when_the_profile_asks_for_it():
    record = resolve_effect(dataset(), LATE, bock_values(),
                            StatsSettings(estimator="hedges", variance="borenstein"))
    assert record.g == pytest.approx(es.hedges_g(-1.6757675, 12, 12), abs=1e-6)
    assert record.es == record.g and record.estimator == "hedges"


# --------------------------------------------------------------------------- conversions
def test_the_se_route_converts_to_sd_before_anything_else():
    """The worked example in the design spec: SD_A = SE 2.01 × √19 = 8.76, d = −0.196."""
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=19, mean=44.67, dispersion_value=2.01,
                            dispersion_type=DispersionType.SE, route="text"),
        group_b=GroupValues(n=20, mean=46.14, dispersion_value=6.08,
                            dispersion_type=DispersionType.SD, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "text_mean_se_ci"
    assert record.inputs["sd_a"] == pytest.approx(es.sd_from_se(2.01, 19), abs=1e-9)
    assert record.d == pytest.approx(-0.19589, abs=1e-4)
    assert "SE 2.01" in record.conversion_chain and "8.76" in record.conversion_chain
    assert "√19" in record.conversion_chain


def test_the_ci_route_uses_a_t_multiplier_for_a_small_sample():
    """Amendment A: `ci_to_sd_dist="auto"` means t(n−1) whenever n < 100."""
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=12, mean=31.51, ci_low=24.45, ci_high=38.57,
                            dispersion_type=DispersionType.CI95, route="text"),
        group_b=GroupValues(n=12, mean=12.28, ci_low=4.77, ci_high=19.79,
                            dispersion_type=DispersionType.CI95, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    expected = es.sd_from_ci(24.45, 38.57, 12, level=0.95, dist="t")
    assert record.inputs["sd_a"] == pytest.approx(expected, abs=1e-9)
    assert "t(11)" in record.conversion_chain
    assert record.route == "text_mean_se_ci"


def test_a_large_sample_confidence_interval_uses_z():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=120, mean=10.0, ci_low=8.0, ci_high=12.0,
                            dispersion_type=DispersionType.CI95, route="text"),
        group_b=GroupValues(n=120, mean=9.0, ci_low=7.0, ci_high=11.0,
                            dispersion_type=DispersionType.CI95, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.inputs["sd_a"] == pytest.approx(es.sd_from_ci(8.0, 12.0, 120, dist="z"), abs=1e-9)
    assert "z" in record.conversion_chain


def test_the_distribution_can_be_forced():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=12, mean=31.51, ci_low=24.45, ci_high=38.57,
                            dispersion_type=DispersionType.CI95, route="text"),
        group_b=GroupValues(n=12, mean=12.28, ci_low=4.77, ci_high=19.79,
                            dispersion_type=DispersionType.CI95, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings(ci_to_sd_dist="z"))
    assert record.inputs["sd_a"] == pytest.approx(es.sd_from_ci(24.45, 38.57, 12, dist="z"),
                                                  abs=1e-9)


def test_an_interval_whose_level_has_no_enum_member_is_still_converted():
    """`DispersionType` names only the 95% and 90% intervals; a 99% interval arrives with an
    UNKNOWN type and its level in `ci_level`, which is enough to convert it."""
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=12, mean=31.51, ci_low=20.0, ci_high=43.02, ci_level=0.99,
                            route="text"),
        group_b=GroupValues(n=12, mean=12.28, ci_low=1.0, ci_high=23.56, ci_level=0.99,
                            route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.inputs["sd_a"] == pytest.approx(
        es.sd_from_ci(20.0, 43.02, 12, level=0.99, dist="t"), abs=1e-9)
    assert "99%" in record.conversion_chain and "t(11)" in record.conversion_chain


def test_a_genuinely_unknown_spread_is_not_reinterpreted():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=12, mean=31.51, dispersion_value=6.0, route="text"),
        group_b=GroupValues(n=12, mean=12.28, dispersion_value=6.0, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible"


def test_a_median_and_quartiles_become_a_mean_and_sd():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=20, median=31.0, q1=24.0, q3=39.0,
                            dispersion_type=DispersionType.IQR, route="text"),
        group_b=GroupValues(n=20, median=12.0, q1=6.0, q3=19.0,
                            dispersion_type=DispersionType.IQR, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    from canopy.stats.conversions import mean_sd_from_median_iqr
    mean, sd = mean_sd_from_median_iqr(31.0, 24.0, 39.0, 20)
    assert record.inputs["mean_a"] == pytest.approx(mean, abs=1e-9)
    assert record.inputs["sd_a"] == pytest.approx(sd, abs=1e-9)
    assert "Luo" in record.conversion_chain or "median" in record.conversion_chain
    assert "median_iqr_conversion" in record.flags


def test_a_range_is_used_but_flagged():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=20, mean=31.0, minimum=10.0, maximum=55.0,
                            dispersion_type=DispersionType.RANGE, route="text"),
        group_b=GroupValues(n=20, mean=12.0, minimum=1.0, maximum=30.0,
                            dispersion_type=DispersionType.RANGE, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.inputs["sd_a"] == pytest.approx(es.sd_from_range(10.0, 55.0, 20), abs=1e-9)
    assert "range_to_sd" in record.flags


def test_individual_points_become_a_mean_and_sd():
    a = [30.0, 32.0, 31.0, 33.0, 31.5]
    b = [12.0, 13.0, 11.0, 12.5, 12.5]
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(points=a, route="figure"),
        group_b=GroupValues(points=b, route="figure"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    mean, sd, n = es.mean_sd_from_points(a)
    assert record.inputs["mean_a"] == pytest.approx(mean) and record.n_a == n
    assert record.inputs["sd_a"] == pytest.approx(sd)
    assert record.route == "figure"


def test_a_median_with_no_mean_never_crashes_the_paper():
    """A paper that reports a median and quartiles reports no mean; the median IS the centre."""
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=20, median=31.0, q1=24.0, q3=39.0,
                            dispersion_type=DispersionType.IQR, route="text"),
        group_b=GroupValues(n=20, median=12.0, q1=6.0, q3=19.0,
                            dispersion_type=DispersionType.IQR, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route != "not_convertible" and record.d is not None


def test_a_range_around_a_median_uses_the_median_as_the_centre():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=20, median=31.0, minimum=10.0, maximum=55.0,
                            dispersion_type=DispersionType.RANGE, route="text"),
        group_b=GroupValues(n=20, median=12.0, minimum=1.0, maximum=30.0,
                            dispersion_type=DispersionType.RANGE, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.inputs["mean_a"] == pytest.approx(31.0)
    assert record.inputs["sd_a"] == pytest.approx(es.sd_from_range(10.0, 55.0, 20), abs=1e-9)
    assert "range_to_sd" in record.flags


def test_a_standard_error_around_a_median_does_not_crash():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=20, median=31.0, dispersion_value=2.0,
                            dispersion_type=DispersionType.SE, route="text"),
        group_b=GroupValues(n=20, median=12.0, dispersion_value=2.0,
                            dispersion_type=DispersionType.SE, route="text"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.inputs["mean_a"] == pytest.approx(31.0) and record.d is not None


def test_groups_with_no_variance_are_refused_with_a_reason():
    values = ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(points=[5.0, 5.0, 5.0], route="figure"),
        group_b=GroupValues(points=[5.0, 5.0, 5.0], route="figure"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "pooled standard deviation" in record.not_convertible_reason
    assert "not_convertible" in record.flags


def test_an_unexpected_failure_becomes_a_record_not_a_crash(monkeypatch):
    """One broken cell must never take the whole paper down with it: whatever goes wrong inside a
    route is recorded as a refusal with the error text, and the next route is tried."""
    from canopy.pipeline import resolve as module

    def explode(*args, **kwargs):
        raise RuntimeError("scipy fell over")

    monkeypatch.setattr(module.es, "smd_from_means", explode)
    record = resolve_effect(dataset(), LATE, bock_values(), StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "RuntimeError" in record.not_convertible_reason
    assert "scipy fell over" in record.not_convertible_reason


def test_an_unrecognised_source_route_is_never_silently_treated_as_printed_text():
    values = bock_values()
    values.group_a.route = "carrier_pigeon"
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert "unknown_source_route" in record.flags
    assert "carrier_pigeon" in record.conversion_chain
    assert record.d is not None                            # still resolved, just not silently


# --------------------------------------------------------------------------- other routes
def test_a_two_group_t_converts_and_says_so():
    values = ResolvedValues(
        higher_is_better=False,
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "test_statistic"
    assert record.d == pytest.approx(-es.d_from_t(5.25, 12, 12), abs=1e-9)
    assert "t = 5.25" in record.conversion_chain and "df 22" in record.conversion_chain


def test_a_mixed_design_f_is_kept_but_not_converted():
    values = ResolvedValues(
        higher_is_better=False,
        test_statistic=StatisticValues(stat_type="F", value=7.58, df1=1.0, df2=22.0,
                                      design="mixed_main_effect", direction="b_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None and record.es is None
    assert "mixed_main_effect" in record.not_convertible_reason
    assert "not_convertible" in record.flags
    assert record.inputs["stat_value"] == 7.58


def test_a_one_way_between_f_converts():
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="F", value=7.58, df1=1.0, df2=22.0,
                                      design="one_way_between", direction="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "test_statistic"
    assert record.d == pytest.approx(es.d_from_f(7.58, 12, 12, 1), abs=1e-9)


def test_a_statistic_with_no_stated_direction_cannot_be_signed():
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="unknown"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible"
    assert "direction" in record.not_convertible_reason


def test_a_p_value_converts_through_its_t():
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="p", p_value=0.03, p_kind="exact", df=22.0,
                                      design="independent_t", direction="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "p_value"
    assert record.d == pytest.approx(es.d_from_p(0.03, 12, 12, True, 1), abs=1e-9)


def test_an_inexact_p_value_is_not_converted():
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="p", p_value=0.05, p_kind="less_than", df=22.0,
                                      design="independent_t", direction="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and "less_than" in record.not_convertible_reason


def test_a_reported_hedges_g_is_used_as_printed():
    values = ResolvedValues(
        higher_is_better=True,
        reported=ReportedValues(value=0.62, scale="hedges_g", standardizer="pooled_sd_between",
                                positive_means="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings(estimator="hedges"))
    assert record.route == "reported_d"
    assert record.g == pytest.approx(0.62, abs=1e-9)
    assert record.d == pytest.approx(0.62 / es.J_exact(22), abs=1e-9)
    assert "hedges_g" in record.conversion_chain


def test_a_within_participant_effect_size_is_refused():
    values = ResolvedValues(
        higher_is_better=True,
        reported=ReportedValues(value=0.62, scale="cohens_d", standardizer="dz_paired",
                                positive_means="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and "dz_paired" in record.not_convertible_reason


# --------------------------------------------------------------------------- precedence
def test_the_precedence_list_decides_when_several_routes_exist():
    values = bock_values(
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="a_greater"),
        reported=ReportedValues(value=-1.7, scale="cohens_d", standardizer="pooled_sd_between",
                                positive_means="b_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "text_mean_sd"
    assert set(record.routes_available) >= {"text_mean_sd", "test_statistic", "reported_d"}
    assert record.routes_rejected == {}


def test_a_protocol_may_reorder_the_precedence():
    values = bock_values(
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="a_greater"))
    settings = StatsSettings(route_precedence=["test_statistic", "text_mean_sd"])
    record = resolve_effect(dataset(), LATE, values, settings)
    assert record.route == "test_statistic"


def test_a_route_the_caller_did_not_offer_is_not_used():
    values = bock_values(
        route_available=["test_statistic"],
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "test_statistic"
    assert "text_mean_sd" in record.routes_rejected


def test_a_route_that_fails_its_gate_falls_through_to_the_next():
    values = ResolvedValues(
        higher_is_better=False,
        test_statistic=StatisticValues(stat_type="F", value=7.58, df1=1.0, df2=22.0,
                                      design="mixed_main_effect", direction="b_greater"),
        reported=ReportedValues(value=-1.7, scale="cohens_d", standardizer="pooled_sd_between",
                                positive_means="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "reported_d"
    assert "test_statistic" in record.routes_rejected
    assert "mixed_main_effect" in record.routes_rejected["test_statistic"]


def test_routes_passed_over_are_listed_with_their_reason():
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                       design="independent_t", direction="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "test_statistic"
    for skipped in ("text_mean_sd", "table", "text_mean_se_ci", "figure"):
        assert skipped in record.routes_rejected, record.routes_rejected
        assert "dispersion" in record.routes_rejected[skipped]


def test_available_routes_reports_why_a_route_is_missing():
    routes, reasons = available_routes(bock_values())
    assert routes == ["text_mean_sd"]
    assert "test_statistic" in reasons and "reported_d" in reasons


def test_nothing_usable_at_all_is_recorded_not_dropped():
    record = resolve_effect(dataset(), LATE, ResolvedValues(higher_is_better=True),
                            StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert record.not_convertible_reason
    assert record.confidence == "needs_human"


# --------------------------------------------------------------------------- digitisation variance
def figure_values(sigma=0.5) -> ResolvedValues:
    return ResolvedValues(
        higher_is_better=False,
        group_a=GroupValues(n=12, mean=31.51, dispersion_value=11.12,
                            dispersion_type=DispersionType.SD, route="figure", sigma=sigma),
        group_b=GroupValues(n=12, mean=12.28, dispersion_value=11.82,
                            dispersion_type=DispersionType.SD, route="figure", sigma=sigma))


def test_digitisation_variance_matches_the_analytic_delta_method():
    record = resolve_effect(dataset(), LATE, figure_values(0.5),
                            StatsSettings(digitization_variance="sensitivity"))
    sp = es.pooled_sd(11.12, 12, 11.82, 12)
    analytic = 2 * (0.5 / sp) ** 2
    assert record.digitization_var == pytest.approx(analytic, rel=1e-5)
    assert record.var_with_digitization == pytest.approx(record.var + record.digitization_var)
    assert record.digitization_var_share == pytest.approx(record.digitization_var / record.var,
                                                          rel=1e-9)


def test_sensitivity_mode_leaves_the_primary_variance_alone():
    record = resolve_effect(dataset(), LATE, figure_values(0.5),
                            StatsSettings(digitization_variance="sensitivity"))
    plain = resolve_effect(dataset(), LATE, figure_values(0.5),
                           StatsSettings(digitization_variance="off"))
    assert record.var == pytest.approx(plain.var)
    assert record.se == pytest.approx(plain.se)
    assert plain.digitization_var is None


def test_primary_mode_adds_it_to_the_variance_and_widens_the_interval():
    sensitivity = resolve_effect(dataset(), LATE, figure_values(2.0),
                                 StatsSettings(digitization_variance="sensitivity"))
    primary = resolve_effect(dataset(), LATE, figure_values(2.0),
                             StatsSettings(digitization_variance="primary"))
    assert primary.var == pytest.approx(sensitivity.var_with_digitization)
    assert primary.se > sensitivity.se
    assert primary.ci_high - primary.ci_low > sensitivity.ci_high - sensitivity.ci_low


def se_values(dispersion_sigma=0.5) -> ResolvedValues:
    """Both groups reported as mean ± SE, with a digitisation uncertainty ON THE SE BAR."""
    return ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=20, mean=31.51, dispersion_value=2.0,
                            dispersion_type=DispersionType.SE, route="figure",
                            sigma=0.3, dispersion_sigma=dispersion_sigma),
        group_b=GroupValues(n=20, mean=12.28, dispersion_value=2.0,
                            dispersion_type=DispersionType.SE, route="figure",
                            sigma=0.3, dispersion_sigma=dispersion_sigma))


def sd_values(dispersion_sigma=0.5) -> ResolvedValues:
    """The SAME standard deviations, already printed as SDs — so nothing needs converting."""
    sd = 2.0 * math.sqrt(20)
    return ResolvedValues(
        higher_is_better=True,
        group_a=GroupValues(n=20, mean=31.51, dispersion_value=sd,
                            dispersion_type=DispersionType.SD, route="figure",
                            sigma=0.3, dispersion_sigma=dispersion_sigma),
        group_b=GroupValues(n=20, mean=12.28, dispersion_value=sd,
                            dispersion_type=DispersionType.SD, route="figure",
                            sigma=0.3, dispersion_sigma=dispersion_sigma))


def test_a_digitisation_uncertainty_is_converted_with_the_dispersion_it_belongs_to():
    """Half a unit of uncertainty on an SE bar is half a unit OF SE; carried onto an SD that is
    √20 = 4.47x larger without conversion it would understate the digitisation variance."""
    record = resolve_effect(dataset(), LATE, se_values(),
                            StatsSettings(digitization_variance="sensitivity"))
    assert record.inputs["sigma_sd_a"] == pytest.approx(0.5 * math.sqrt(20), rel=1e-9)
    assert record.inputs["sigma_mean_a"] == pytest.approx(0.3)
    assert "scales with it" in record.conversion_chain


def test_an_uncertainty_on_a_printed_sd_is_left_alone():
    record = resolve_effect(dataset(), LATE, sd_values(),
                            StatsSettings(digitization_variance="sensitivity"))
    assert record.inputs["sigma_sd_a"] == pytest.approx(0.5)          # unchanged, nothing converted
    assert record.inputs["sd_a"] == pytest.approx(2.0 * math.sqrt(20))
    assert "scales with it" not in record.conversion_chain


def test_converting_the_bar_without_its_uncertainty_would_understate_the_variance():
    """Same SDs, same means, same n: the two records differ ONLY in whether the uncertainty on the
    spread was converted with it. The SE route must come out larger, by the square of √20."""
    converted = resolve_effect(dataset(), LATE, se_values(),
                               StatsSettings(digitization_variance="sensitivity"))
    printed = resolve_effect(dataset(), LATE, sd_values(),
                             StatsSettings(digitization_variance="sensitivity"))
    assert converted.inputs["sd_a"] == pytest.approx(printed.inputs["sd_a"])
    assert converted.digitization_var > printed.digitization_var
    # the mean partials are identical, so the difference is entirely in the SD partials
    mean_only = resolve_effect(dataset(), LATE, sd_values(dispersion_sigma=None),
                               StatsSettings(digitization_variance="sensitivity"))
    from_sd_converted = converted.digitization_var - mean_only.digitization_var
    from_sd_printed = printed.digitization_var - mean_only.digitization_var
    assert from_sd_converted == pytest.approx(from_sd_printed * 20, rel=1e-4)


def test_a_large_digitisation_variance_is_flagged():
    small = resolve_effect(dataset(), LATE, figure_values(0.2),
                           StatsSettings(digitization_variance="sensitivity"))
    large = resolve_effect(dataset(), LATE, figure_values(3.0),
                           StatsSettings(digitization_variance="sensitivity"))
    assert "digitization_variance_large" not in small.flags
    assert "digitization_variance_large" in large.flags


# --------------------------------------------------------------------------- shared controls
def three_arms():
    shared = GroupValues(n=30, mean=10.0, dispersion_value=3.0,
                         dispersion_type=DispersionType.SD, route="text")
    return [ResolvedValues(dataset_id=f"p:d{i}", higher_is_better=True,
                           group_a=GroupValues(n=10, mean=12.0 + i, dispersion_value=3.0,
                                               dispersion_type=DispersionType.SD, route="text"),
                           group_b=shared.model_copy())
            for i in range(3)]


def test_a_shared_control_is_split_between_its_comparisons():
    rows = apply_shared_control(three_arms(), "split_n", shared="B")
    assert len(rows) == 3
    assert [r.group_b.n for r in rows] == [10, 10, 10]
    assert all("shared_control_split" in r.flags for r in rows)
    assert [r.group_a.n for r in rows] == [10, 10, 10]


def test_combining_the_arms_leaves_one_row():
    rows = apply_shared_control(three_arms(), "combine_arms", shared="B")
    assert len(rows) == 1
    assert rows[0].group_a.n == 30
    assert rows[0].group_b.n == 30
    assert rows[0].group_a.mean == pytest.approx((12.0 + 13.0 + 14.0) / 3)
    assert "shared_control_combined" in rows[0].flags


def test_keeping_the_first_comparison_drops_the_rest():
    rows = apply_shared_control(three_arms(), "keep_first", shared="B")
    assert len(rows) == 1 and rows[0].dataset_id == "p:d0"
    assert "shared_control_kept_first" in rows[0].flags


def test_keeping_every_comparison_flags_them_all():
    rows = apply_shared_control(three_arms(), "keep_all_flagged", shared="B")
    assert len(rows) == 3
    assert all("shared_control_repeated" in r.flags for r in rows)
    assert [r.group_b.n for r in rows] == [30, 30, 30]


def test_one_comparison_is_not_a_shared_control():
    rows = apply_shared_control(three_arms()[:1], "split_n", shared="B")
    assert len(rows) == 1 and rows[0].group_b.n == 30 and rows[0].flags == []


# --------------------------------------------------------------------------- multi-group policy
def test_two_groups_need_no_multi_group_ruling():
    assert multi_group_flags(dataset(), "closest_to_definition") == []


def test_more_than_two_groups_records_the_policy_that_chose_the_pair():
    from canopy.models import GroupSpec

    many = dataset()
    many.all_groups_listed = [GroupSpec(label=name) for name in ("young", "middle", "old")]
    many.chosen_pair_rationale = "the protocol names the youngest and oldest groups"
    flags = multi_group_flags(many, "extremes")
    assert flags == ["multi_group_extremes"]


def test_a_policy_this_layer_cannot_apply_asks_for_a_human():
    """`combine_matching` needs a per-group protocol match the mapper does not record."""
    from canopy.models import GroupSpec

    many = dataset()
    many.all_groups_listed = [GroupSpec(label=name) for name in ("young", "middle", "old")]
    many.chosen_pair_rationale = "the two extremes"
    for policy in ("combine_matching", "needs_human"):
        assert "multi_group_needs_human" in multi_group_flags(many, policy)


def test_a_chosen_pair_with_no_rationale_is_flagged():
    from canopy.models import GroupSpec

    many = dataset()
    many.all_groups_listed = [GroupSpec(label=name) for name in ("young", "middle", "old")]
    flags = multi_group_flags(many, "closest_to_definition")
    assert "multi_group_no_rationale" in flags


# --------------------------------------------------------------------------- plumbing
def test_group_values_are_built_from_a_verdict():
    verdict = Verdict(dataset_id="ds1", outcome_key="late_adaptation", group="A", n=12,
                      mean=31.51, dispersion_value=11.12, dispersion_type=DispersionType.SD,
                      unit="deg", route="text", sigma=0.4, higher_is_better=False)
    values = GroupValues.from_verdict(verdict)
    assert values.n == 12 and values.mean == 31.51 and values.route == "text"
    assert values.sigma == 0.4 and values.unit == "deg"


def test_resolved_values_are_built_from_two_verdicts():
    a = Verdict(dataset_id="ds1", outcome_key="late_adaptation", group="A", n=12, mean=31.51,
                dispersion_value=11.12, dispersion_type=DispersionType.SD, route="text",
                higher_is_better=False, confidence="auto_accept")
    b = Verdict(dataset_id="ds1", outcome_key="late_adaptation", group="B", n=12, mean=12.28,
                dispersion_value=11.82, dispersion_type=DispersionType.SD, route="text",
                higher_is_better=False, confidence="accept_with_note")
    values = ResolvedValues.from_verdicts(a, b)
    assert values.higher_is_better is False and values.dataset_id == "ds1"
    assert values.confidence == "accept_with_note"          # the weaker of the two governs
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.d == pytest.approx(-1.6757675, abs=1e-6)


def test_an_interval_from_a_verdict_becomes_quartiles_for_an_iqr():
    verdict = Verdict(dataset_id="ds1", outcome_key="late_adaptation", group="A", n=20,
                      mean=31.0, ci_low=24.0, ci_high=39.0, dispersion_type=DispersionType.IQR,
                      route="text")
    values = GroupValues.from_verdict(verdict)
    assert (values.q1, values.q3, values.median) == (24.0, 39.0, 31.0)


def test_a_record_round_trips_through_json():
    from canopy.models import EffectSizeRecord

    record = resolve_effect(dataset(), LATE, bock_values(), StatsSettings())
    assert EffectSizeRecord.model_validate_json(record.model_dump_json()) == record


def test_resolving_never_changes_the_values_it_was_given():
    values = bock_values()
    before = values.model_dump()
    resolve_effect(dataset(), LATE, values, StatsSettings())
    assert values.model_dump() == before
