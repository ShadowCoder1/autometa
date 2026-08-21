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
import re
import sys
from pathlib import Path

import pytest

from canopy.models import (DatasetSpec, DispersionType, GroupSpec, OutcomeDef, StatsSettings,
                           Verdict)
from canopy.pipeline.resolve import (GroupValues, ReportedValues, ResolvedValues, StatisticValues,
                                     apply_shared_control, available_routes, multi_group_flags,
                                     resolve_effect, resolve_effect_with_fallback)
from canopy.pipeline.rows import ENSEMBLE, cell_candidates, fallback_values, prepare_row_values
from canopy.stats import effect_sizes as es
from tests.helpers import nine

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
                                      design="independent_t", direction="a_greater",
                                      contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "test_statistic"
    assert record.d == pytest.approx(-es.d_from_t(5.25, 12, 12), abs=1e-9)
    assert "t = 5.25" in record.conversion_chain and "df 22" in record.conversion_chain


def test_a_mixed_design_f_is_kept_but_not_converted():
    values = ResolvedValues(
        higher_is_better=False,
        test_statistic=StatisticValues(stat_type="F", value=7.58, df1=1.0, df2=22.0,
                                      design="mixed_main_effect", direction="b_greater",
                                      contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None and record.es is None
    assert "mixed_main_effect" in record.not_convertible_reason
    # P-B: the reason names aggregation scope, not the false "different error term"
    assert "averaged over" in record.not_convertible_reason
    assert "error term" not in record.not_convertible_reason
    assert "not_convertible" in record.flags
    assert record.inputs["stat_value"] == 7.58


def test_a_one_way_between_f_with_no_recorded_factors_is_blocked_by_default():
    """C5, fail-closed: an F whose model nobody described is refused even when the design label
    allows it and the error df match exactly.

    A mixed main effect's between-subjects error df are ALSO n_a + n_b - 2 (Heuer's F(1,38) with
    20/20), so the degrees of freedom cannot tell a one-way ANOVA from a main effect averaged over
    eight targets. Only the design label separates them, and a label is what the fail-closed rule
    declines to trust: absence of evidence about the model's factors is evidence of risk.
    """
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="F", value=7.58, df1=1.0, df2=22.0,
                                      design="one_way_between", direction="a_greater",
                                      contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "within-subject factors were not recorded" in record.not_convertible_reason


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
                                      design="independent_t", direction="a_greater",
                                      contrast_kind="groups"))
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
                                positive_means="a_greater", contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings(estimator="hedges"))
    assert record.route == "reported_d"
    assert record.g == pytest.approx(0.62, abs=1e-9)
    assert record.d == pytest.approx(0.62 / es.J_exact(22), abs=1e-9)
    assert "hedges_g" in record.conversion_chain


def test_a_within_participant_effect_size_is_refused():
    values = ResolvedValues(
        higher_is_better=True,
        reported=ReportedValues(value=0.62, scale="cohens_d", standardizer="dz_paired",
                                positive_means="a_greater", contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and "dz_paired" in record.not_convertible_reason


def test_a_printed_effect_size_for_a_test_against_a_constant_is_refused():
    """F5 / P-B, end to end: "the aftereffect differed from zero, d = 1.30" is not the contrast.

    `contrast_kind` was carried on the `Candidate` and read by nobody on this route, so a d of
    1.30 for a one-sample test against zero pooled as the between-group effect. A printed effect
    size always has a contrast behind it, and that is the field that records it.
    """
    values = ResolvedValues(
        higher_is_better=True,
        reported=ReportedValues(value=1.30, scale="cohens_d", standardizer="pooled_sd_between",
                                positive_means="a_greater", contrast_kind="against_constant"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "constant" in record.not_convertible_reason


def test_a_printed_effect_size_whose_contrast_nobody_recorded_is_refused_rather_than_assumed():
    """The default is the refusing one, as it is for a test statistic."""
    values = ResolvedValues(
        higher_is_better=True,
        reported=ReportedValues(value=1.30, scale="cohens_d", standardizer="pooled_sd_between",
                                positive_means="a_greater"))
    assert values.reported.contrast_kind == "unknown"
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "nobody recorded" in record.not_convertible_reason


def test_a_printed_effect_size_for_the_two_groups_still_converts():
    values = ResolvedValues(
        higher_is_better=True,
        reported=ReportedValues(value=1.30, scale="cohens_d", standardizer="pooled_sd_between",
                                positive_means="a_greater", contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "reported_d" and record.d == pytest.approx(1.30, abs=1e-9)


# --------------------------------------------------------------------------- precedence
def test_the_precedence_list_decides_when_several_routes_exist():
    values = bock_values(
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="a_greater"),
        reported=ReportedValues(value=-1.7, scale="cohens_d", standardizer="pooled_sd_between",
                                positive_means="b_greater", contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "text_mean_sd"
    assert set(record.routes_available) >= {"text_mean_sd", "test_statistic", "reported_d"}
    assert record.routes_rejected == {}


def test_a_protocol_may_reorder_the_precedence():
    values = bock_values(
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="a_greater",
                                      contrast_kind="groups"))
    settings = StatsSettings(route_precedence=["test_statistic", "text_mean_sd"])
    record = resolve_effect(dataset(), LATE, values, settings)
    assert record.route == "test_statistic"


def test_a_route_the_caller_did_not_offer_is_not_used():
    values = bock_values(
        route_available=["test_statistic"],
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                      design="independent_t", direction="a_greater",
                                      contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "test_statistic"
    assert "text_mean_sd" in record.routes_rejected


def test_a_route_that_fails_its_gate_falls_through_to_the_next():
    values = ResolvedValues(
        higher_is_better=False,
        test_statistic=StatisticValues(stat_type="F", value=7.58, df1=1.0, df2=22.0,
                                      design="mixed_main_effect", direction="b_greater",
                                      contrast_kind="groups"),
        reported=ReportedValues(value=-1.7, scale="cohens_d", standardizer="pooled_sd_between",
                                positive_means="a_greater", contrast_kind="groups"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "reported_d"
    assert "test_statistic" in record.routes_rejected
    assert "mixed_main_effect" in record.routes_rejected["test_statistic"]


def test_routes_passed_over_are_listed_with_their_reason():
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                       design="independent_t", direction="a_greater",
                                       contrast_kind="groups"))
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


# ------------------------------------------------- the conversion gate: whose question does this answer?
def test_a_statistic_with_no_recorded_contrast_is_refused_by_the_conversion_gate():
    """P-B, code-side: `contrast_kind` is a required extractor field and `unknown` is a refusal.

    Nothing today reaches this — the run's statistic route converts no cell — but the moment a
    loosening opens one, "this is the two groups" would otherwise be enforced only by the model
    that wrote the label.
    """
    values = ResolvedValues(
        higher_is_better=False,
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                       design="independent_t", direction="a_greater"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "nobody recorded" in record.not_convertible_reason


def test_a_test_against_zero_is_refused_even_when_the_extractor_mislabelled_its_design():
    values = ResolvedValues(
        higher_is_better=False,
        test_statistic=StatisticValues(stat_type="t", value=5.25, df=22.0,
                                       design="independent_t", direction="a_greater",
                                       contrast_kind="against_constant"))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "constant" in record.not_convertible_reason


def test_a_chi_square_is_refused_instead_of_being_square_rooted_like_an_f():
    """P-B: "A chi2 is never a `value` source for a continuous outcome."

    Route availability already refuses to call a chi2 a `test_statistic` route; the conversion
    itself had no guard, so a chi2 that reached `_from_statistic` by any other path fell through
    to the F branch and was square-rooted like one.
    """
    from canopy.pipeline.resolve import _from_statistic
    from canopy.stats.effect_sizes import NotConvertible

    stat = StatisticValues(stat_type="chi2", value=7.58, df=22.0, design="independent_t",
                           direction="a_greater", contrast_kind="groups")
    values = ResolvedValues(higher_is_better=True, test_statistic=stat)
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "no t or F statistic" in record.routes_rejected["test_statistic"]
    with pytest.raises(NotConvertible) as exc:
        _from_statistic(values, dataset(), StatsSettings(), {}, [], as_p=False)
    assert "no conversion" in str(exc.value) and "chi-square" in str(exc.value)


def test_a_chi_square_with_an_exact_p_is_refused_by_every_route():
    """F1, reproduced from the review: the p route walked around the chi2 refusal.

    `available_routes` offered `p_value` for any statistic carrying a p, and `_from_statistic`
    ran the `as_p` branch BEFORE the chi2 guard, so chi2(1) = 7.58, p = .006 on 12/12 became
    d = 1.2414 — a number invented out of a contingency table. The guard belongs where the
    conversion starts, and the route must not be offered at all for a statistic that is not a
    t, an F or a p.
    """
    from canopy.pipeline.resolve import _from_statistic
    from canopy.stats.effect_sizes import NotConvertible

    stat = StatisticValues(stat_type="chi2", value=7.58, df=22.0, design="independent_t",
                           direction="a_greater", contrast_kind="groups", p_kind="exact",
                           p_value=0.006, tails=2)
    values = ResolvedValues(higher_is_better=True, test_statistic=stat)
    routes, reasons = available_routes(values)
    assert "p_value" not in routes and "test_statistic" not in routes
    assert "chi2" in reasons["p_value"] or "not a t" in reasons["p_value"]
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    for as_p in (True, False):
        with pytest.raises(NotConvertible) as exc:
            _from_statistic(values, dataset(), StatsSettings(), {}, [], as_p=as_p)
        assert "chi-square" in str(exc.value)


def test_a_chi_square_with_an_exact_p_and_no_degrees_of_freedom_is_refused_too():
    """The same input with `df=None` — the review's second reproduction, which converted with
    flags `['aggregation_scope_matched', 'df_missing']`."""
    stat = StatisticValues(stat_type="chi2", value=7.58, df=None, design="one_way_between",
                           direction="a_greater", contrast_kind="groups", p_kind="exact",
                           p_value=0.006, tails=2, within_factors=["condition"],
                           outcome_averages_over=["condition"])
    values = ResolvedValues(higher_is_better=True, test_statistic=stat)
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None


def test_a_statistic_of_no_recorded_kind_is_refused_on_the_p_route_as_well():
    """`stat_type: unknown` has no formula either way: the p route is not a way around that."""
    stat = StatisticValues(stat_type="unknown", value=7.58, df=22.0, design="independent_t",
                           direction="a_greater", contrast_kind="groups", p_kind="exact",
                           p_value=0.006, tails=2)
    values = ResolvedValues(higher_is_better=True, test_statistic=stat)
    routes, _ = available_routes(values)
    assert routes == []
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None


def test_a_main_effect_averaged_over_a_factor_this_outcome_keeps_is_refused_with_the_factor_named():
    """C5 end to end through the resolver: the reason a reviewer reads names the factor."""
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="t", value=2.1, df=22.0,
                                       design="independent_t", direction="a_greater",
                                       contrast_kind="groups",
                                       within_factors=["target direction (8 levels)"],
                                       outcome_averages_over=["block"]))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "not_convertible" and record.d is None
    assert "target direction (8 levels)" in record.not_convertible_reason


def test_the_same_statistic_converts_when_the_outcome_asks_for_that_average():
    values = ResolvedValues(
        higher_is_better=True,
        test_statistic=StatisticValues(stat_type="t", value=2.1, df=22.0,
                                       design="independent_t", direction="a_greater",
                                       contrast_kind="groups",
                                       within_factors=["target direction (8 levels)"],
                                       outcome_averages_over=["target direction"]))
    record = resolve_effect(dataset(), LATE, values, StatsSettings())
    assert record.route == "test_statistic"
    assert record.d == pytest.approx(es.d_from_t(2.1, 12, 12), abs=1e-9)


# ------------------------------- the row-level refusals, as ONE constant the review layer reads
def test_every_row_level_refusal_the_resolver_can_add_is_in_the_exported_set():
    """Consistency item 1 (questions area, fix round 3, concern 1).

    A row refusal is the case where BOTH cells read as fine, so every rule that asks "may this
    cell be released?" has to consult the row. The review layer does that against
    `ROW_REFUSAL_CODES`; if `_finish` could add a code that is not in it, the review layer would
    release cells under a row the resolver refused — silently, because the row is still held and
    nothing says the cells disagree with it.

    So `_add_row_refusal` is the only writer, it refuses an unlisted code, and this reads the
    source to prove `_finish` has no second way to put a flag on the record.
    """
    import inspect

    from canopy.models import EffectSizeRecord
    from canopy.pipeline.resolve import ROW_REFUSAL_CODES, _add_row_refusal, _finish

    body = inspect.getsource(_finish)
    # everything before the first gate is the CONVERSION's own provenance (`median_iqr_conversion`,
    # `range_to_sd`, …) — recorded, never holding. From the first gate on, the row's bucket is
    # being decided, and the only thing that may add a code there is the guarded writer.
    deciding = body[body.index("conversion_gate_bucket("):]
    writes = [line.strip() for line in deciding.splitlines()
              if "record.flags" in line and line.strip().startswith("record.flags")]
    assert writes == [], f"_finish assigns record.flags outside _add_row_refusal: {writes}"
    added = re.findall(r"_add_row_refusal\(record,\s*([A-Za-z_][\w.]*)\)", body)
    assert added, "no row refusal is added at all — has the screen moved?"
    resolved = [getattr(sys.modules["canopy.pipeline.resolve"], name) for name in added]
    assert set(resolved) <= ROW_REFUSAL_CODES, sorted(set(resolved) - ROW_REFUSAL_CODES)

    with pytest.raises(KeyError):
        _add_row_refusal(EffectSizeRecord(), "a_code_the_review_layer_never_heard_of")


def test_the_review_layer_reads_the_resolvers_set_rather_than_a_copy_of_it():
    """The same object, not two sets with the same members today: a mirror is a list kept in sync
    by hand, and the first code added to one and not the other is the bug this pins."""
    from canopy.pipeline.overrides import ROW_REFUSALS
    from canopy.pipeline.resolve import ROW_REFUSAL_CODES
    from canopy.verify.confidence import ROW_REFUSAL_CODES as declared

    assert ROW_REFUSAL_CODES is declared
    assert ROW_REFUSALS is declared


# ----------------------------------------------- D1: a printed value with no spread yields
#
# Heuer & Hegele 2008 d1 late adaptation. The adjudicator kept the two numbers the paper PRINTS —
# 27.7° and 18.9°, the initial direction error in the last practice block — over the digitisation
# of Figure 2a, because a value that cannot be quoted cannot be kept. The paper prints no spread
# for either, so the printed pair converts to nothing at all and the cell reaches the review queue
# as `not_convertible`, while the figure the readers DID measure sits in the same cell with a mean,
# an SE and an n for both groups. D1's ruling: the printed value is preferred whenever it CONVERTS,
# and when it cannot the row may be built from a convertible same-locator candidate pair — held,
# never released, with the swap written on the record.
def _cell(paper, ds_id, key, hib):
    p = nine.protocol()
    ds = nine.dataset(paper, ds_id)
    cands = nine.candidates(paper)
    vs = {v.group: v for v in nine.verdicts(paper)
          if v.dataset_id == ds_id and v.outcome_key == key}
    primary = prepare_row_values(ds, key, vs["A"], vs["B"], cands, p.stats, higher_is_better=hib)
    # the fixture's verdicts carry an orientation of their own, so `higher_is_better=None` above
    # reaches `from_verdicts` as "nobody forced one" rather than as "nobody settled one". The
    # tests below mean the second, and say so here rather than by editing a real run's verdict.
    primary.higher_is_better = hib
    return p, ds, primary, fallback_values(cell_candidates(cands, ds_id, key), primary, p.stats)


def test_group_statistics_missing_is_flagged_on_the_record():
    p, ds, primary, _ = _cell("3570e4ce2a9c", "3570e4ce2a9c:d1", "late_adaptation", False)
    assert "group_statistics_missing" in resolve_effect(ds, p.outcome("late_adaptation"),
                                                        primary, p.stats).flags


def test_precedence_override_uses_figure_when_text_has_no_dispersion():
    p, ds, primary, alts = _cell("3570e4ce2a9c", "3570e4ce2a9c:d1", "late_adaptation", False)
    rec = resolve_effect_with_fallback(ds, p.outcome("late_adaptation"), primary, alts, p.stats)
    assert rec.route == "figure" and abs(rec.es + 0.627) < 0.01 and rec.confidence == "needs_human"
    assert "precedence_override" in rec.flags and rec.route_overridden_from == "text_mean_se_ci"
    assert "group_statistics_missing" in rec.flags   # the swap AND what made it necessary
    assert "27.7" in rec.precedence_override_reason
    assert "verifier objection" in rec.precedence_override_reason


def test_precedence_override_refuses_when_orientation_is_unresolved():
    p, ds, primary, alts = _cell("3570e4ce2a9c", "3570e4ce2a9c:d1", "late_adaptation", None)
    rec = resolve_effect_with_fallback(ds, p.outcome("late_adaptation"), primary, alts, p.stats)
    assert rec.route == "not_convertible" and "precedence_override" not in rec.flags


def test_fallback_pairs_never_cross_locators():
    p, ds, primary, alts = _cell("d1f2946e7e81", "d1f2946e7e81:d1", "late_adaptation", True)
    assert alts and all(a.group_a.locator == a.group_b.locator for a in alts)


def test_rows_that_already_convert_are_untouched():
    for r in nine.records():
        if r.route != "not_convertible":
            assert "precedence_override" not in r.flags   # only not_convertible rows fall back


def test_a_read_the_panel_check_set_aside_is_not_an_alternative():
    """Fix round 1, finding 1. D2's caption check marks a reading whose panel the caption gives to
    the OTHER group (`pixel_provenance["locator_dropped"]`), and the mark is on the CANDIDATE so
    that every caller of the vote honours it, not only the orchestrator that filtered its own list.
    The fallback is such a caller: `run._resolve` and `overrides._prepare` both hand it the full
    candidate list with the marks still on it. Un-honoured, the override's "pair" could be one
    group's real read plus another group's panel — a wrong effect size that D1 then admits into the
    best-guess line by rule."""
    from canopy.verify.vote import LOCATOR_DROPPED

    p, ds, primary, alts = _cell("3570e4ce2a9c", "3570e4ce2a9c:d1", "late_adaptation", False)
    assert alts, "the cell has a pair to begin with, so setting it aside is what this measures"

    def set_aside(cand):
        if cand.extractor_id != ENSEMBLE:
            return cand
        marked = cand.model_copy(deep=True)
        marked.pixel_provenance = {**marked.pixel_provenance,
                                   LOCATOR_DROPPED: "the caption gives panel a to the other group"}
        return marked

    cell = [set_aside(c) for c in cell_candidates(nine.candidates("3570e4ce2a9c"),
                                                  "3570e4ce2a9c:d1", "late_adaptation")]
    assert fallback_values(cell, primary, p.stats) == []
    rec = resolve_effect_with_fallback(ds, p.outcome("late_adaptation"), primary,
                                       fallback_values(cell, primary, p.stats), p.stats)
    assert rec.route == "not_convertible" and "precedence_override" not in rec.flags


def test_a_row_that_converted_from_a_printed_statistic_is_never_overridden():
    """Fix round 1, finding 2. `group_statistics_missing` says the GROUP routes were unavailable —
    it does not say the row got no effect size. A paper that prints means with no spread and a t
    beside them converts through `test_statistic`, and `figure` outranks `test_statistic` in the
    default precedence, so the rank guard alone would have swapped a released row's number for a
    digitised one and held it. D1 is scoped to a value that converts to NOTHING."""
    p, ds, primary, alts = _cell("3570e4ce2a9c", "3570e4ce2a9c:d1", "late_adaptation", False)
    assert alts, "the figure pair the fallback would otherwise have taken is there"
    primary.test_statistic = StatisticValues(stat_type="t", value=2.1, df=38.0,
                                             design="independent_t", direction="a_greater",
                                             contrast_kind="groups")
    rec = resolve_effect_with_fallback(ds, p.outcome("late_adaptation"), primary, alts, p.stats)
    assert rec.route == "test_statistic" and "precedence_override" not in rec.flags
    assert "group_statistics_missing" in rec.flags     # raised, but on its own not a trigger
    assert rec.model_dump() == resolve_effect(ds, p.outcome("late_adaptation"), primary,
                                              p.stats).model_dump()


def test_a_vote_that_disagreed_is_not_a_missing_spread():
    """Fix round 2, finding 2 (BLOCKER). Buch d2's aftereffect cell A has `agreement: "disagree"`
    and NO mean at all: the vote weighed the readings and refused to settle one. D1 is scoped to a
    value the paper PRINTS that merely carries no spread — "the printed value is preferred whenever
    it converts; when it cannot…" — and a refusal on evidence is not that case. Falling back there
    put a digitised pair (d = −1.35) on the best-guess forest for a cell the pipeline never
    resolved a value for."""
    p, ds, primary, alts = _cell("592b3b55a318", "592b3b55a318:d2", "aftereffect", False)
    assert primary.group_a.mean is None and primary.disagreed == ["A"]
    assert len(alts) >= 2, "the pairs are there — the point is that none of them is taken"
    rec = resolve_effect_with_fallback(ds, p.outcome("aftereffect"), primary, alts, p.stats)
    assert rec.route == "not_convertible" and "precedence_override" not in rec.flags
    assert rec.es is None and rec.route_overridden_from == ""


def test_two_pairs_that_convert_hold_the_row_instead_of_taking_the_first():
    """Fix round 2, finding 2 (BLOCKER), second half. The same cell is read in two places — Fig 4's
    left panel (d = −1.35) and Fig 3's top-right (d = −0.08), both `figure`, so neither is preferred
    by the protocol's precedence list. Taking "the first that converts" is taking the order
    `rows.fallback_values` happened to build them in, and a factor of seventeen on the best-guess
    forest rested on it. D1 supplies a value the precedence list could not; it does not choose
    BETWEEN values. The row is held, and the reason names both pairs so the card can offer them."""
    p, ds, primary, alts = _cell("592b3b55a318", "592b3b55a318:d2", "aftereffect", False)
    # the vote's refusal above is the FIRST gate and would stop the fallback before the choice
    # this test is about is reached, so cell A is given the printed value the vote lacked — a mean
    # with no spread, which is exactly the case D1 was written for.
    primary.disagreed = []
    primary.group_a.mean, primary.group_a.n = -2.5, 5
    rec = resolve_effect_with_fallback(ds, p.outcome("aftereffect"), primary, alts, p.stats)
    assert rec.route == "not_convertible" and "precedence_override" not in rec.flags
    assert rec.es is None
    assert "Fig 4, left panel" in rec.not_convertible_reason
    assert "Fig 3, top-right panel" in rec.not_convertible_reason
    assert rec.conversion_chain.endswith(rec.not_convertible_reason)


# ------------------------- which ARM a doubt is about (adversarial review, fix round MINOR 4)
# `confidence.resolve_cell` puts the whole cell's flag list on BOTH groups' verdicts, so "which
# arm?" is answered by the flag's `candidate_ids` and the candidate's group — never by which
# verdict carries the code, because every code is on both.
def _armed_verdict(group, *flags):
    from canopy.models import CheckFlag, Verdict

    return Verdict(dataset_id="ds1", outcome_key="late_adaptation", group=group,
                   confidence="accept_with_note",
                   flags=[CheckFlag(code=code, severity="warn", message=code,
                                    candidate_ids=list(cids))
                          for code, cids in flags])


def _armed_candidates():
    from canopy.models import Candidate

    return [Candidate(candidate_id=cid, dataset_id="ds1", outcome_key="late_adaptation",
                      kind="group_stats", group=group, status="found", mean=1.0)
            for cid, group in (("a1", "A"), ("a2", "A"), ("b1", "B"))]


def test_a_doubt_is_attributed_to_the_arm_whose_reading_raised_it():
    from canopy.pipeline.resolve import _codes_by_arm

    a = _armed_verdict("A", ("dispersion_unknown", ["a1"]), ("n_missing", ["a2"]))
    b = _armed_verdict("B", ("dispersion_unknown", ["a1"]), ("n_missing", ["a2"]))
    by_arm = _codes_by_arm(a, b, _armed_candidates())
    assert by_arm == {"A": ["dispersion_unknown", "n_missing"], "B": []}


def test_a_code_raised_on_no_candidate_belongs_to_no_arm():
    """It describes the CELL, not one of its numbers, so a per-number rule may not fire on it. It
    is still in the row's `flags`, which is the union and where every other reader looks."""
    from canopy.pipeline.resolve import _codes_by_arm

    a = _armed_verdict("A", ("dispersion_unknown", []), ("n_missing", []))
    b = _armed_verdict("B", ("dispersion_unknown", []), ("n_missing", []))
    assert _codes_by_arm(a, b, _armed_candidates()) == {"A": [], "B": []}


def test_an_answer_that_clears_a_code_on_one_arm_clears_it_for_that_arm_only():
    """`overrides._apply_value` strips a cleared code from the ANSWERED cell's flags and leaves the
    other cell's list alone. Reading both verdicts for one arm would find the code on the arm whose
    reviewer had just retired it, and the row would stay held for ever — so each arm is attributed
    from its OWN verdict."""
    from canopy.pipeline.resolve import _codes_by_arm

    answered = _armed_verdict("A", ("dispersion_unknown", ["a1"]))          # `n_missing` retired
    other = _armed_verdict("B", ("dispersion_unknown", ["a1"]), ("n_missing", ["a2"]))
    by_arm = _codes_by_arm(answered, other, _armed_candidates())
    assert by_arm["A"] == ["dispersion_unknown"]                            # the pair is broken
    assert "n_missing" not in by_arm["A"]


# --------------------------------------------------------------- the map's group sizes

def _sized(spec_n: tuple[int | None, int | None] = (12, 12), *,
           evidence: str = "twelve older and twelve younger adults were analysed",
           kind: DispersionType = DispersionType.SD,
           settled: tuple[int | None, int | None] = (None, None)) -> tuple:
    """A pair the paper printed, whose group sizes only the map carries."""
    spec = DatasetSpec(dataset_id="b511dbb76fa6:d1", cluster_id="b511dbb76fa6", label="pointing",
                       group_a=GroupSpec(label="old", n=spec_n[0], n_evidence=evidence),
                       group_b=GroupSpec(label="young", n=spec_n[1], n_evidence=evidence))
    values = bock_values(
        group_a=GroupValues(n=settled[0], mean=31.51, dispersion_value=11.12,
                            dispersion_type=kind, unit="deg", route="text"),
        group_b=GroupValues(n=settled[1], mean=12.28, dispersion_value=11.82,
                            dispersion_type=kind, unit="deg", route="text"))
    return spec, values


def test_a_printed_pair_whose_size_only_the_map_carries_still_converts():
    """The Hermans shape: means and SDs in one sentence, group sizes under Participants.

    The map reads the participants section — two agents, adjudicated on disagreement — and records
    the size with the quote that proves it. The extractors, reading the results sentence, have no
    size to carry, and the row was refused for want of a number the run already held. Refused
    SILENTLY, because a row that converts to nothing is in neither analysis line.
    """
    from canopy.pipeline.rows import N_FROM_MAP, _fill_group_n

    spec, values = _sized()
    assert _fill_group_n(values, spec) == ["A", "B"]
    assert (values.group_a.n, values.group_b.n) == (12, 12)
    assert values.group_a.n_from_map and values.group_b.n_from_map
    record = resolve_effect(spec, LATE, values, StatsSettings())
    assert record.route == "text_mean_sd"
    assert record.es is not None
    assert "from the map's participants section" in record.conversion_chain, \
        "the chain claimed the paper printed a size it printed three pages earlier"


def test_the_maps_size_may_not_complete_a_row_it_would_scale():
    """A size that reconstructs the SD is a question for a person, not a gap for the code.

    `SD = SE x sqrt(n)`, so for an SE, a CI, an IQR or a range the size is multiplied into the
    MAGNITUDE: a size read out of a recruitment sentence would silently scale the effect. Only a
    printed SD gives an estimate the size cannot move — there `n` merely weights the pooling.
    """
    from canopy.pipeline.rows import _fill_group_n

    for kind in (DispersionType.SE, DispersionType.CI95, DispersionType.IQR,
                 DispersionType.RANGE):
        spec, values = _sized(kind=kind)
        assert _fill_group_n(values, spec) == [], f"{kind.value} was completed from the map"
        assert values.group_a.n is None


def test_the_map_never_overrules_a_size_the_reading_carried():
    """A disagreement between a transcribed size and the map's is `n_mismatch`'s to raise."""
    from canopy.pipeline.rows import _fill_group_n

    spec, values = _sized(spec_n=(12, 12), settled=(18, 16))
    assert _fill_group_n(values, spec) == []
    assert (values.group_a.n, values.group_b.n) == (18, 16)


def test_a_size_that_is_not_a_size_leaves_the_row_where_it_was():
    """Below `MIN_N`, or recorded with no quote: neither is evidence of a group size."""
    from canopy.pipeline.rows import _fill_group_n

    for spec_n, evidence in (((1, 12), "one subject"), ((12, 12), ""), ((None, 12), "n/a")):
        spec, values = _sized(spec_n=spec_n, evidence=evidence)
        assert "A" not in _fill_group_n(values, spec)


def test_a_group_size_never_rescues_a_missing_mean():
    """A size is a size. It is not evidence that the paper reported the contrast."""
    from canopy.pipeline.rows import _fill_group_n

    spec, values = _sized()
    values.group_a.mean = None
    assert _fill_group_n(values, spec) == ["B"]
    assert values.group_a.n is None
