"""Task 8, step 1: the code-only consistency checks (spec §3.3(1) + amendment G).

Every rule here is arithmetic or bookkeeping over candidates that already exist, so nothing in
this file needs a model: hand-built `Candidate`s in, `CheckFlag`s out. A check never repairs a
value and never drops a candidate — it says what is wrong, on which candidates, and how badly,
and `canopy.verify.confidence` decides what that means for the cell.
"""
from __future__ import annotations

import pytest

from canopy.models import (Candidate, DatasetSpec, DispersionType, GroupSpec, OutcomeSources,
                           Source, SourceKind)
from canopy.verify.checks import (CHECK_SEVERITY, GROUP_LABEL_MISMATCH_NOTE, codes, run_checks)


# --------------------------------------------------------------------------- fixtures
def make_dataset(**kwargs) -> DatasetSpec:
    outcome = kwargs.pop("outcome", None) or OutcomeSources(
        outcome_key="late_adaptation", measure_name="mean direction error", units="deg",
        higher_is_better=False,
        sources=[Source(kind=SourceKind.text_mean_sd, page=3, locator="Results ¶2")])
    return DatasetSpec(
        dataset_id="ds1", cluster_id="paper1", label="pointing",
        group_a=GroupSpec(label="old subjects", n=12),
        group_b=GroupSpec(label="young subjects", n=12),
        outcomes=[outcome], **kwargs)


def cand(group="A", **kwargs) -> Candidate:
    base = dict(candidate_id=f"c{group}", paper_id="paper1", dataset_id="ds1",
                outcome_key="late_adaptation", kind="group_stats", group=group, status="found",
                source_kind=SourceKind.text_mean_sd, n=12, mean=31.51, dispersion_value=11.12,
                dispersion_type=DispersionType.SD, unit="deg", grounded=True, route="text",
                model="claude-opus-5", extractor_id="text:table_first:claude-opus-5", page=3,
                quote="the mean direction error was 31.51 +/- 11.12 deg")
    base.update(kwargs)
    return Candidate(**base)


def pair(**overrides) -> list[Candidate]:
    """One clean, mutually consistent candidate per group (the Bock late-adaptation numbers)."""
    return [cand("A"), cand("B", candidate_id="cB", mean=12.28, dispersion_value=11.82,
                 quote="the mean direction error was 12.28 +/- 11.82 deg", **overrides)]


# --------------------------------------------------------------------------- the clean case
def test_a_consistent_pair_raises_nothing():
    assert run_checks(make_dataset(), "late_adaptation", pair()) == []


def test_every_code_has_a_declared_severity():
    """A flag whose severity is not declared could never be weighed by `confidence`."""
    for code, severity in CHECK_SEVERITY.items():
        assert severity in ("info", "warn", "error"), code


def test_flags_are_stable_and_sorted_worst_first():
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", grounded=False, n=1), cand("B", mean=None, status="found")])
    severities = [f.severity for f in flags]
    rank = {"error": 0, "warn": 1, "info": 2}
    assert severities == sorted(severities, key=lambda s: rank[s])
    assert flags == run_checks(make_dataset(), "late_adaptation",
                               [cand("A", grounded=False, n=1), cand("B", mean=None,
                                                                     status="found")])


# --------------------------------------------------------------------------- spec §3.3(1)
def test_an_n_below_two_is_an_error():
    flags = run_checks(make_dataset(), "late_adaptation", [cand("A", n=1), cand("B")])
    flag = next(f for f in flags if f.code == "n_too_small")
    assert flag.severity == "error" and flag.candidate_ids == ["cA"] and "1" in flag.message
    assert "n_not_integer" not in codes(flags)           # a whole number, just too few of them


def test_a_row_only_table_match_is_flagged_as_a_possible_group_mix_up():
    """`grounding.check_table_cell` writes ROW_ONLY when the numbers are in the named row but not
    in the named column — exactly what reading the other group's cell looks like."""
    from canopy.verify.grounding import ROW_ONLY

    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", notes=f"table cell check: {ROW_ONLY}: found in p1t1 row 'old' "
                                        f"but not in column 'mean'"), cand("B")])
    flag = next(f for f in flags if f.code == "quote_row_only")
    assert flag.severity == "warn" and flag.candidate_ids == ["cA"]


def test_a_value_matched_without_its_sign_is_flagged():
    from canopy.verify.grounding import SIGN_NOTE

    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", notes=f"table cell check: row-confirmed; {SIGN_NOTE} for "
                                        f"[31.51]"), cand("B")])
    flag = next(f for f in flags if f.code == "sign_not_confirmed")
    assert flag.severity == "warn"


# --------------------------------------------------------------------------- the sign check
def test_a_stated_direction_that_the_numbers_contradict_is_an_error():
    from canopy.models import OrientationVerdict

    verdict = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=False,
                                 agreed=True, needs_human=False,
                                 direction_stated_in_text="b_greater")
    flags = run_checks(make_dataset(), "late_adaptation", pair(), orientation=verdict)
    flag = next(f for f in flags if f.code == "sign_mismatch")
    assert flag.severity == "error"
    assert "31.51" in flag.message and "12.28" in flag.message
    assert set(flag.candidate_ids) == {"cA", "cB"}


def test_a_stated_direction_the_numbers_agree_with_is_not_flagged():
    from canopy.models import OrientationVerdict

    verdict = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=False,
                                 agreed=True, needs_human=False,
                                 direction_stated_in_text="a_greater")
    assert "sign_mismatch" not in codes(
        run_checks(make_dataset(), "late_adaptation", pair(), orientation=verdict))


def test_identical_means_carry_no_direction_to_contradict():
    """A stated direction with equal extracted means is a magnitude problem, not a sign problem:
    the effect size is zero, and `sign_mismatch` would tell a reviewer the sign is inverted."""
    from canopy.verify.checks import sign_check

    assert sign_check("a_greater", 12.0, 12.0) is None
    assert sign_check("b_greater", 12.0, 12.0) is None
    assert sign_check("unknown", 31.51, 12.28) is None
    assert sign_check("a_greater", None, 12.28) is None
    assert sign_check("b_greater", 31.51, 12.28) is not None


def test_a_missing_n_is_flagged_but_not_fatal():
    flags = run_checks(make_dataset(), "late_adaptation", [cand("A", n=None), cand("B")])
    assert "n_missing" in codes(flags)
    assert CHECK_SEVERITY["n_missing"] == "warn"


def test_an_n_that_contradicts_the_map_is_flagged():
    flags = run_checks(make_dataset(), "late_adaptation", [cand("A", n=9), cand("B")])
    flag = next(f for f in flags if f.code == "n_mismatch")
    assert "9" in flag.message and "12" in flag.message


def test_group_ns_that_do_not_sum_to_the_reported_total_are_flagged():
    flags = run_checks(make_dataset(), "late_adaptation", pair(), total_n=30)
    flag = next(f for f in flags if f.code == "n_sum_mismatch")
    assert "24" in flag.message and "30" in flag.message


@pytest.mark.parametrize("dispersion", [0.0, -1.5])
def test_a_dispersion_that_is_not_positive_is_an_error(dispersion):
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", dispersion_value=dispersion), cand("B")])
    flag = next(f for f in flags if f.code == "sd_nonpositive")
    assert flag.severity == "error"


def test_a_dispersion_of_effectively_zero_is_flagged():
    """Amendment G: an SD read as ~0 turns a mean difference into an infinite effect size."""
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", dispersion_value=0.0001), cand("B")])
    assert "sd_near_zero" in codes(flags) and "sd_nonpositive" not in codes(flags)


def test_an_se_that_does_not_imply_the_reported_sd_is_flagged():
    """SE·√n ≈ SD: the same group read twice, once as SD and once as SE."""
    se_reader = cand("A", candidate_id="cA2", dispersion_value=9.0,
                     dispersion_type=DispersionType.SE, model="claude-sonnet-5",
                     extractor_id="text:narrative_first:claude-sonnet-5")
    flags = run_checks(make_dataset(), "late_adaptation", [cand("A"), se_reader, cand("B")])
    flag = next(f for f in flags if f.code == "se_sd_inconsistent")
    assert set(flag.candidate_ids) == {"cA", "cA2"}


def test_a_consistent_se_and_sd_pair_is_not_flagged():
    se_reader = cand("A", candidate_id="cA2", dispersion_value=11.12 / 12 ** 0.5,
                     dispersion_type=DispersionType.SE, model="claude-sonnet-5")
    flags = run_checks(make_dataset(), "late_adaptation", [cand("A"), se_reader, cand("B")])
    assert "se_sd_inconsistent" not in codes(flags)


def test_an_asymmetric_confidence_interval_is_flagged():
    lopsided = cand("A", dispersion_type=DispersionType.CI95, dispersion_value=None,
                    ci_low=20.0, ci_high=50.0)
    flags = run_checks(make_dataset(), "late_adaptation", [lopsided, cand("B")])
    assert "ci_asymmetric" in codes(flags)


def test_a_symmetric_confidence_interval_is_not_flagged():
    fine = cand("A", dispersion_type=DispersionType.CI95, dispersion_value=None,
                ci_low=31.51 - 6.3, ci_high=31.51 + 6.3)
    assert "ci_asymmetric" not in codes(run_checks(make_dataset(), "late_adaptation",
                                                  [fine, cand("B")]))


def test_a_figure_value_outside_the_calibrated_axis_is_an_error():
    off_axis = cand("A", source_kind=SourceKind.figure_bar, route="figure", mean=140.0,
                    extractor_id="digitize:ensemble",
                    pixel_provenance={"cal": {"y_min": 0.0, "y_max": 60.0, "y_tick": 10.0}},
                    quote="")
    flags = run_checks(make_dataset(), "late_adaptation", [off_axis, cand("B")])
    flag = next(f for f in flags if f.code == "value_outside_axis")
    assert flag.severity == "error" and "60" in flag.message


def test_a_figure_value_inside_the_axis_passes():
    on_axis = cand("A", source_kind=SourceKind.figure_bar, route="figure", mean=31.51,
                   extractor_id="digitize:ensemble", quote="",
                   pixel_provenance={"cal": {"y_min": 0.0, "y_max": 60.0, "y_tick": 10.0}})
    assert "value_outside_axis" not in codes(
        run_checks(make_dataset(), "late_adaptation", [on_axis, cand("B")]))


def test_an_unknown_dispersion_type_is_flagged():
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", dispersion_type=DispersionType.UNKNOWN), cand("B")])
    assert "dispersion_unknown" in codes(flags)


def test_a_group_with_no_dispersion_at_all_is_flagged():
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", dispersion_value=None,
                             dispersion_type=DispersionType.NONE), cand("B")])
    assert "dispersion_missing" in codes(flags)


def test_an_ungrounded_quote_is_an_error():
    flags = run_checks(make_dataset(), "late_adaptation", [cand("A", grounded=False), cand("B")])
    flag = next(f for f in flags if f.code == "quote_not_grounded")
    assert flag.severity == "error" and flag.candidate_ids == ["cA"]


def test_a_short_quote_is_noted_but_not_an_error():
    """Grounding flags a short quote rather than refusing it (`SHORT_QUOTE_NOTE`)."""
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", quote="31.51", notes="short quote: 5 characters ground too "
                                                       "easily to stand as evidence"), cand("B")])
    flag = next(f for f in flags if f.code == "quote_short")
    assert flag.severity == "info"


def test_the_same_numbers_under_two_outcomes_are_flagged():
    other = cand("A", candidate_id="cOther", outcome_key="aftereffect")
    flags = run_checks(make_dataset(), "late_adaptation", pair(), other_candidates=[other])
    flag = next(f for f in flags if f.code == "duplicate_across_outcomes")
    assert "aftereffect" in flag.message and set(flag.candidate_ids) == {"cA", "cOther"}


def test_a_swapped_group_label_is_an_error():
    swapped = cand("A", notes=f"{GROUP_LABEL_MISMATCH_NOTE}: extractor put 'young subjects' in "
                              f"group A, but that label matches group B ('young subjects')")
    flags = run_checks(make_dataset(), "late_adaptation", [swapped, cand("B")])
    flag = next(f for f in flags if f.code == "group_label_swapped")
    assert flag.severity == "error"


def test_the_swap_marker_is_the_one_the_extractor_actually_writes():
    """This check reads a marker another module writes; if that wording moves, fail here."""
    from canopy.agents.extract_common import group_label_check

    message = group_label_check("A", "young subjects", make_dataset())
    assert message.startswith(GROUP_LABEL_MISMATCH_NOTE)


def test_an_implausible_effect_size_is_flagged():
    huge = [cand("A", mean=100.0, dispersion_value=1.0),
            cand("B", candidate_id="cB", mean=1.0, dispersion_value=1.0)]
    flags = run_checks(make_dataset(), "late_adaptation", huge)
    flag = next(f for f in flags if f.code == "effect_implausible")
    assert "3" in flag.message


def test_a_test_statistic_without_degrees_of_freedom_is_flagged():
    stat = Candidate(candidate_id="cT", dataset_id="ds1", outcome_key="late_adaptation",
                     kind="test_statistic", stat_type="t", stat_value=5.25, design="independent_t",
                     grounded=True, quote="the difference was significant (t=5.25)")
    flags = run_checks(make_dataset(), "late_adaptation", [stat])
    assert "test_stat_missing_df" in codes(flags)


def test_a_test_statistic_with_degrees_of_freedom_passes():
    stat = Candidate(candidate_id="cT", dataset_id="ds1", outcome_key="late_adaptation",
                     kind="test_statistic", stat_type="t", stat_value=5.25, df=22.0,
                     design="independent_t", grounded=True, quote="t(22)=5.25")
    assert "test_stat_missing_df" not in codes(run_checks(make_dataset(), "late_adaptation",
                                                          [stat]))


# --------------------------------------------------------------------------- amendment G extras
def test_two_units_for_one_outcome_are_flagged():
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", unit="mm"), cand("B", unit="deg")])
    flag = next(f for f in flags if f.code == "unit_mismatch")
    assert "mm" in flag.message and "deg" in flag.message


def test_a_unit_that_contradicts_the_map_is_flagged():
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", unit="mm"), cand("B", unit="mm")])
    assert "unit_mismatch" in codes(flags)


def test_a_figure_n_that_differs_from_the_analysed_n_is_flagged():
    figure = cand("A", source_kind=SourceKind.figure_points, route="figure", n=10, quote="",
                  extractor_id="digitize:ensemble")
    flags = run_checks(make_dataset(), "late_adaptation", [figure, cand("B")])
    assert "figure_n_mismatch" in codes(flags)


def test_too_few_digitised_points_for_the_analysed_n_is_flagged():
    points = cand("A", source_kind=SourceKind.figure_points, route="figure", quote="",
                  extractor_id="digitize:ensemble", points=[1.0, 2.0, 3.0])
    flags = run_checks(make_dataset(), "late_adaptation", [points, cand("B")])
    flag = next(f for f in flags if f.code == "points_undercount")
    assert "3" in flag.message and "12" in flag.message


def test_mixed_analysis_metrics_within_a_paper_are_flagged():
    a = cand("A", analysis_metric="endpoint")
    b = cand("B", analysis_metric="change_from_baseline")
    flags = run_checks(make_dataset(), "late_adaptation", [a, b])
    flag = next(f for f in flags if f.code == "metric_mixed")
    assert "endpoint" in flag.message and "change_from_baseline" in flag.message


def test_one_analysis_metric_plus_unknowns_is_not_mixed():
    a = cand("A", analysis_metric="endpoint")
    b = cand("B", analysis_metric="unknown")
    assert "metric_mixed" not in codes(run_checks(make_dataset(), "late_adaptation", [a, b]))


def test_a_dispersion_type_that_contradicts_the_map_is_flagged():
    outcome = OutcomeSources(
        outcome_key="late_adaptation", units="deg", higher_is_better=False,
        sources=[Source(kind=SourceKind.figure_bar, page=4, figure_id="fig02", locator="Fig 2",
                        error_bar_type=DispersionType.SE, error_bar_agreement="agreed")])
    figure = cand("A", source_kind=SourceKind.figure_bar, route="figure", page=4, quote="",
                  extractor_id="digitize:ensemble", dispersion_type=DispersionType.SD)
    flags = run_checks(make_dataset(outcome=outcome), "late_adaptation", [figure])
    flag = next(f for f in flags if f.code == "dispersion_type_conflict")
    assert "SE" in flag.message and "SD" in flag.message


def test_an_unknown_figure_error_bar_is_flagged():
    outcome = OutcomeSources(
        outcome_key="late_adaptation", units="deg", higher_is_better=False,
        sources=[Source(kind=SourceKind.figure_bar, page=4, figure_id="fig02", locator="Fig 2",
                        error_bar_type=DispersionType.UNKNOWN)])
    flags = run_checks(make_dataset(outcome=outcome), "late_adaptation", pair())
    assert "figure_error_bar_unknown" in codes(flags)


def test_an_unconfirmed_figure_error_bar_is_noted():
    outcome = OutcomeSources(
        outcome_key="late_adaptation", units="deg", higher_is_better=False,
        sources=[Source(kind=SourceKind.figure_bar, page=4, figure_id="fig02", locator="Fig 2",
                        error_bar_type=DispersionType.SE, error_bar_agreement="unconfirmed")])
    flags = run_checks(make_dataset(outcome=outcome), "late_adaptation", pair())
    assert "error_bar_unconfirmed" in codes(flags)


def test_an_outcome_with_no_direction_is_flagged():
    outcome = OutcomeSources(outcome_key="late_adaptation", units="deg", higher_is_better=None)
    flags = run_checks(make_dataset(outcome=outcome), "late_adaptation", pair())
    flag = next(f for f in flags if f.code == "orientation_unknown")
    assert "direction" in flag.message.lower()


def test_a_resolved_orientation_clears_the_flag():
    from canopy.models import OrientationVerdict

    outcome = OutcomeSources(outcome_key="late_adaptation", units="deg", higher_is_better=None)
    verdict = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=False,
                                 agreed=True, needs_human=False)
    flags = run_checks(make_dataset(outcome=outcome), "late_adaptation", pair(),
                       orientation=verdict)
    assert "orientation_unknown" not in codes(flags)


# --------------------------------------------------------------------------- what is NOT checked
def test_a_candidate_that_found_nothing_carries_no_value_checks():
    """`not_on_these_pages` is an honest gap, not a bad value: it must not raise value flags."""
    empty = cand("A", status="not_on_these_pages", mean=None, dispersion_value=None, n=None,
                 dispersion_type=DispersionType.UNKNOWN, unit="", quote="", grounded=None)
    flags = run_checks(make_dataset(), "late_adaptation", [empty, cand("B")])
    assert not [f for f in flags if f.candidate_ids == ["cA"]], [f.code for f in flags]


def test_checks_never_change_a_candidate():
    candidates = [cand("A", n=1, grounded=False), cand("B")]
    before = [c.model_dump() for c in candidates]
    run_checks(make_dataset(), "late_adaptation", candidates)
    assert [c.model_dump() for c in candidates] == before
