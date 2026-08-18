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


def test_the_row_only_marker_is_matched_as_grounding_writes_it():
    """The bare words could appear in anyone's prose; the marker is the constant plus the
    punctuation `check_table_cell` writes after it."""
    from canopy.verify.checks import ROW_ONLY_MARKER
    from canopy.verify.grounding import ROW_ONLY

    assert ROW_ONLY_MARKER.startswith(ROW_ONLY) and ROW_ONLY_MARKER != ROW_ONLY
    prose = cand("A", notes="the reviewer wondered whether this was a row-only match")
    assert "quote_row_only" not in codes(run_checks(make_dataset(), "late_adaptation",
                                                    [prose, cand("B")]))


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
    flag = next(f for f in flags if f.code == "implausible_dispersion")
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


def test_a_reader_the_resolved_means_contradict_is_an_error_on_the_cell():
    """Fix round F5. `combine_orientation` records the code in the verdict's notes and this module
    puts it in front of `confidence`; the severity is declared here, and it is an `error`.

    The reader the filter discards is the one that made the only mechanically checkable claim on
    this measure, and it said the opposite of what this cell's own resolved means say. That is
    evidence about the VALUES, not only about the polarity — which is why it is not a warning
    worth -0.08 that a good score can absorb. Before this it was, and combined with the row-4
    defect it is exactly what let an inverted sign through.
    """
    from canopy.models import OrientationVerdict
    from canopy.verify.checks import orientation_note

    assert CHECK_SEVERITY["orientation_reader_contradicts_values"] == "error"
    verdict = OrientationVerdict(
        outcome_key="late_adaptation", higher_is_better=False, agreed=True, needs_human=False,
        notes=orientation_note("orientation_reader_contradicts_values",
                               "claude-opus-5 states b greater on this measure, but the resolved "
                               "raw means say A = 31.51 and B = 12.28"))
    flags = run_checks(make_dataset(), "late_adaptation", pair(), orientation=verdict)
    flag = next(f for f in flags if f.code == "orientation_reader_contradicts_values")
    assert flag.severity == "error"
    assert "claude-opus-5" in flag.message


def test_a_swapped_group_label_suspends_the_discard_check_like_a_transposed_series():
    """Fix round, caller guarantee 7: `group_label_swapped` says the two groups may be the wrong
    way round, which is exactly what `series_transposed` says about a figure. A discard run
    against means that may be the other group's throws out the reader that read the paper right,
    so the orientation check must abstain rather than choose while it is open."""
    from canopy.verify.checks import DISPUTED_MEANS_FLAGS

    assert "group_label_swapped" in DISPUTED_MEANS_FLAGS
    assert CHECK_SEVERITY["group_label_swapped"] == "error"
    assert DISPUTED_MEANS_FLAGS <= set(CHECK_SEVERITY)


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


# --------------------------------------------------------------------------- task 16: P3 as split
def _figure_cand(group="A", cal_status="confirmed", ticks=((0.0, 45.0), (100.0, 15.0)),
                 mean=31.3, agree=True, **kwargs) -> Candidate:
    provenance = {"figure_id": "fig03", "cal_status": cal_status, "mean_agreement": agree,
                  "cal_note": "an axis note",
                  "cal": {"ticks": [list(t) for t in ticks]} if ticks else None}
    return cand(group, candidate_id=f"fig{group}", source_kind=SourceKind.figure_line,
                extractor_id="digitize:ensemble", route="figure", mean=mean, quote="",
                pixel_provenance=provenance, **kwargs)


def test_a_value_outside_a_CONFIRMED_axis_is_still_an_error():
    dataset = make_dataset()
    flags = run_checks(dataset, "late_adaptation", [_figure_cand(mean=900.0)])
    assert "value_outside_axis" in codes(flags)
    assert CHECK_SEVERITY["value_outside_axis"] == "error"


def test_a_value_outside_an_UNCONFIRMED_axis_convicts_the_axis_instead():
    """F1: a ladder misread as 1..4 sent a correct read of 31.3 to a human as an `error`."""
    dataset = make_dataset()
    flags = run_checks(dataset, "late_adaptation",
                       [_figure_cand(cal_status="cal_refuted", ticks=((0.0, 4.0), (100.0, 1.0)))])
    assert "value_outside_axis" not in codes(flags)
    assert "calibration_refuted" in codes(flags)
    refuted = next(f for f in flags if f.code == "calibration_refuted")
    assert refuted.severity == "error" and "fig03" in refuted.message


def test_a_single_witness_axis_inside_its_own_frame_is_only_a_warning():
    dataset = make_dataset()
    flags = run_checks(dataset, "late_adaptation",
                       [_figure_cand(cal_status="single_witness", agree=True, mean=31.3)])
    assert codes(flags) == ["calibration_single_witness"]
    assert CHECK_SEVERITY["calibration_single_witness"] == "warn"


def test_agreeing_readers_outside_a_single_witness_frame_are_still_an_error():
    """Suppressing this outright left a ladder misread the OTHER way with nothing to stop it.

    F1 is not this case: there the ladder was refuted, and `digitize` writes no `cal` for a refuted
    ladder, so there are no limits to test a value against. Here the ladder stands, one witness
    built it, and two readers agree on a value it cannot draw — which is evidence.
    """
    dataset = make_dataset()
    flags = run_checks(dataset, "late_adaptation",
                       [_figure_cand(cal_status="single_witness", agree=True, mean=900.0)])
    assert "value_outside_axis" in codes(flags)
    assert "calibration_single_witness" in codes(flags)
    assert "calibration_disputed" not in codes(flags)
    # …and a refuted ladder still cannot convict the value, because it leaves no ladder behind
    refuted = _figure_cand(cal_status="cal_refuted", ticks=None, mean=31.3)
    assert "value_outside_axis" not in codes(run_checks(dataset, "late_adaptation", [refuted]))


def test_a_single_witness_axis_whose_readers_also_disagree_is_a_humans_problem():
    dataset = make_dataset()
    flags = run_checks(dataset, "late_adaptation",
                       [_figure_cand(cal_status="single_witness", agree=False)])
    assert "calibration_disputed" in codes(flags)
    assert CHECK_SEVERITY["calibration_disputed"] == "error"


def test_a_record_with_no_calibration_status_is_checked_the_way_it_always_was():
    """Every candidate written before task 16 (and every hand-built one) still gets the axis check."""
    dataset = make_dataset()
    plain = cand("A", candidate_id="old", source_kind=SourceKind.figure_bar, mean=900.0, quote="",
                 extractor_id="digitize:ensemble",
                 pixel_provenance={"cal": {"ticks": [[0.0, 45.0], [100.0, 15.0]]}})
    flags = run_checks(dataset, "late_adaptation", [plain])
    assert "value_outside_axis" in codes(flags)
    assert not [f for f in flags if f.code.startswith("calibration_")]


# --------------------------------------------------------------------------- metric scope
def _metric_pair(metric_a: str, metric_b: str, outcome_key="late_adaptation") -> list[Candidate]:
    return [cand("A", candidate_id="ma", outcome_key=outcome_key, analysis_metric=metric_a),
            cand("B", candidate_id="mb", outcome_key=outcome_key, mean=12.28,
                 analysis_metric=metric_b)]


def test_metric_mixed_is_scoped_to_one_outcome():
    """F4: Cressman's late adaptation IS an endpoint and its aftereffect IS a difference."""
    dataset = make_dataset()
    within = _metric_pair("endpoint", "change_from_baseline")
    assert "metric_mixed" in codes(run_checks(dataset, "late_adaptation", within))

    clean = _metric_pair("endpoint", "endpoint")
    others = _metric_pair("change_from_baseline", "change_from_baseline",
                          outcome_key="aftereffect")
    flags = run_checks(dataset, "late_adaptation", clean, other_candidates=others)
    assert "metric_mixed" not in codes(flags)
    # …but the paper-wide drift is still recorded, one severity down
    assert "metric_mixed_across_outcomes" in codes(flags)
    assert CHECK_SEVERITY["metric_mixed_across_outcomes"] == "info"


# --------------------------------------------------------------------------- unit coherence
def test_two_readings_a_decade_apart_are_a_unit_problem_not_a_reading_error():
    """Cressman's aftereffect A: read-outs at 61.6, a coords route at 0.061, all labelled "deg"."""
    dataset = make_dataset()
    rows = [cand("A", candidate_id="readout", mean=61.6, dispersion_value=3.0),
            cand("A", candidate_id="coords", mean=0.0610, dispersion_value=0.15)]
    flags = run_checks(dataset, "late_adaptation", rows)
    assert "unit_incoherent" in codes(flags)
    assert "unit_mismatch" not in codes(flags)          # the unit STRINGS agree perfectly
    flag = next(f for f in flags if f.code == "unit_incoherent")
    assert set(flag.candidate_ids) == {"readout", "coords"}


def test_a_mean_and_a_spread_in_different_units_on_one_candidate_are_flagged():
    dataset = make_dataset()
    rows = [cand("A", candidate_id="readout", mean=61.6, dispersion_value=3.0),
            cand("A", candidate_id="vector", mean=0.0383, dispersion_value=2.106)]
    flags = run_checks(dataset, "late_adaptation", rows)
    incoherent = [f for f in flags if f.code == "unit_incoherent"]
    assert any(f.candidate_ids == ["vector"] for f in incoherent), \
        "the mean/dispersion mismatch on one candidate was not named"


def test_two_groups_that_simply_differ_are_not_a_unit_problem():
    """The check is per GROUP: A at 0.5 and B at 30 is a large effect, not a unit mix."""
    dataset = make_dataset()
    rows = [cand("A", mean=0.5, dispersion_value=0.2),
            cand("B", candidate_id="cB", mean=30.0, dispersion_value=6.0)]
    assert "unit_incoherent" not in codes(run_checks(dataset, "late_adaptation", rows))


# --------------------------------------------------------------------------- series and axis
def test_both_groups_on_one_marker_is_flagged_and_withholds_the_cell():
    """Misses 4/5: numeric agreement says nothing about WHICH curve was read."""
    from canopy.verify.confidence import CONTRADICTING_FLAGS

    dataset = make_dataset()
    conflicted = _figure_cand(cal_status="confirmed")
    conflicted.pixel_provenance = {**conflicted.pixel_provenance, "series_identity": {
        "conflict": True, "notes": ["both groups were described as the same marker (open square)"]}}
    flags = run_checks(dataset, "late_adaptation", [conflicted])
    assert "series_identity_conflict" in codes(flags)
    flag = next(f for f in flags if f.code == "series_identity_conflict")
    assert "same marker" in flag.message and flag.severity == "warn"
    # EXPECTATION CHANGED (was `in CAPPING_FLAGS`): one series read for both groups means one
    # group's number IS the other's, which is a different quantity, not a weaker one.
    assert "series_identity_conflict" in CONTRADICTING_FLAGS


def test_readers_who_answered_off_two_axes_are_reported_on_the_row():
    """Miss 1: a left axis in degrees and a right one in per cent are both correct, and differ."""
    dataset = make_dataset()
    split = _figure_cand(cal_status="confirmed")
    split.pixel_provenance = {**split.pixel_provenance, "axis_agreement": "conflict",
                              "axis_kept": "left y-axis (deg)",
                              "axis_dropped_samples": ["digitize:readout:claude-haiku-4-5:direct"]}
    flags = run_checks(dataset, "late_adaptation", [split])
    assert "axis_conflict" in codes(flags)
    flag = next(f for f in flags if f.code == "axis_conflict")
    assert "left y-axis (deg)" in flag.message
    assert "digitize:readout:claude-haiku-4-5:direct" in flag.message


def test_a_figure_whose_readers_agreed_about_the_axis_raises_nothing():
    dataset = make_dataset()
    calm = _figure_cand(cal_status="confirmed")
    calm.pixel_provenance = {**calm.pixel_provenance, "axis_agreement": "agreed",
                             "series_identity": {"conflict": False, "notes": []}}
    assert not [f for f in run_checks(dataset, "late_adaptation", [calm])
                if f.code in ("axis_conflict", "series_identity_conflict")]


def _series_cand(**series) -> Candidate:
    cand = _figure_cand(cal_status="confirmed")
    cand.pixel_provenance = {**cand.pixel_provenance, "series_identity": {
        "conflict": False, "transposed": False, "marker_mismatch": False, "notes": [], **series}}
    return cand


def test_a_transposition_and_a_marker_quibble_are_no_longer_the_same_finding():
    """Two findings with opposite consequences must not share a code.

    "The marker vocabulary did not line up" is a soft doubt about a shape word. "Each group's
    value was measured on the other group's marker" is a determination that the effect's sign is
    inverted, which no downstream stage can recover from. They shared `series_marker_mismatch`,
    so the second could only ever cost a cell 0.08 and a note.
    """
    from canopy.verify.confidence import CAPPING_FLAGS, CONTRADICTING_FLAGS

    dataset = make_dataset()
    swapped = _series_cand(transposed=True, notes=["the two series are transposed"],
                           described={"A": ["filled", "square"], "B": ["open", "circle"]},
                           detected={"A": ["open", "circle"], "B": ["filled", "square"]})
    assert "series_transposed" in codes(run_checks(dataset, "late_adaptation", [swapped]))
    assert "series_marker_mismatch" not in codes(run_checks(dataset, "late_adaptation", [swapped]))
    assert "series_transposed" in CONTRADICTING_FLAGS

    quibble = _series_cand(marker_mismatch=True, notes=["group A was described as a square"])
    assert "series_marker_mismatch" in codes(run_checks(dataset, "late_adaptation", [quibble]))
    assert "series_transposed" not in codes(run_checks(dataset, "late_adaptation", [quibble]))
    assert "series_marker_mismatch" in CAPPING_FLAGS


def test_a_transposition_nobody_could_see_at_both_points_is_only_a_doubt():
    """A detector's "I could not tell" must never be read as "it is not there".

    The producer's test is "these two descriptors agree on everything BOTH of them state", which
    is vacuously true against a marker whose shape and fill the pixel pass could not resolve. A
    conviction that inverts an effect's sign may not rest on a vacuous match, so a transposition
    that is not positively corroborated at BOTH measured points falls back to the soft doubt.
    """
    dataset = make_dataset()
    vacuous = _series_cand(transposed=True, notes=["the two series are transposed"],
                           described={"A": ["filled", "square"], "B": ["filled", "triangle"]},
                           detected={"A": ["filled", "triangle"], "B": ["", ""]})
    found = codes(run_checks(dataset, "late_adaptation", [vacuous]))
    assert "series_transposed" not in found
    assert "series_marker_mismatch" in found
    message = next(f for f in run_checks(dataset, "late_adaptation", [vacuous])
                   if f.code == "series_marker_mismatch").message
    assert "could not" in message or "not corroborated" in message


def test_a_transposition_with_no_descriptors_recorded_at_all_is_only_a_doubt():
    """An older record carries the boolean and not the descriptors it was derived from."""
    dataset = make_dataset()
    bare = _series_cand(transposed=True, notes=["transposed"])
    assert "series_transposed" not in codes(run_checks(dataset, "late_adaptation", [bare]))
    assert "series_marker_mismatch" in codes(run_checks(dataset, "late_adaptation", [bare]))


def test_a_dispersion_taken_from_the_legend_is_flagged_and_caps():
    """UNKNOWN used to go to a human; believing the legend must not silently pool instead."""
    from canopy.verify.confidence import CAPPING_FLAGS

    dataset = make_dataset()
    cand = _figure_cand(cal_status="confirmed")
    cand.dispersion_type = DispersionType.SD
    cand.pixel_provenance = {**cand.pixel_provenance, "dispersion_type_from": "legend",
                             "legend_says": "error bars are the standard deviation"}
    flags = run_checks(dataset, "late_adaptation", [cand])
    assert "dispersion_type_from_legend" in codes(flags)
    assert "standard deviation" in next(
        f for f in flags if f.code == "dispersion_type_from_legend").message
    assert "dispersion_type_from_legend" in CAPPING_FLAGS
    # …and a type the MAP determined raises nothing
    mapped = _figure_cand(cal_status="confirmed")
    mapped.pixel_provenance = {**mapped.pixel_provenance, "dispersion_type_from": "mapper"}
    assert "dispersion_type_from_legend" not in codes(
        run_checks(dataset, "late_adaptation", [mapped]))


def test_an_axis_conflict_withholds_the_cell_it_survives():
    """A value off the wrong ladder is wrong by a factor; keeping the majority does not prove it.

    EXPECTATION CHANGED (was `in CAPPING_FLAGS`): under the R2 floor this flag could not move a
    cell out of the accept band at all. A figure read off the wrong panel in the wrong unit,
    carrying exactly this flag, replayed at 0.4500 — the acceptance threshold.
    """
    from canopy.verify.confidence import CONTRADICTING_FLAGS

    assert "axis_conflict" in CONTRADICTING_FLAGS
    dataset = make_dataset()
    split = _figure_cand(cal_status="confirmed")
    split.pixel_provenance = {**split.pixel_provenance, "axis_agreement": "conflict",
                              "axis_kept": "left y-axis (deg)", "axis_dropped_samples": ["r1"]}
    assert "axis_conflict" in codes(run_checks(dataset, "late_adaptation", [split]))


def test_no_calibration_at_all_is_flagged_at_least_as_loudly_as_one_witness():
    """Zero witnesses to the scale cannot be quieter than one.

    `cal_status="none"` means no y calibration could be built for the figure at all — the
    read-out routes need no ladder to produce a number, so such a cell reached the score with
    nothing said about its axis, while a cell whose axis ONE witness had established was flagged
    and capped. Absence of a check is a finding and has to be reported as one.
    """
    from canopy.verify.confidence import CALIBRATION_PENALTY, CAPPING_FLAGS

    dataset = make_dataset()
    missing = _figure_cand(cal_status="none", ticks=None)
    found = codes(run_checks(dataset, "late_adaptation", [missing]))
    assert "calibration_missing" in found
    flag = next(f for f in run_checks(dataset, "late_adaptation", [missing])
                if f.code == "calibration_missing")
    assert flag.severity == "warn"
    assert "calibration_missing" in CAPPING_FLAGS
    assert (CALIBRATION_PENALTY["calibration_missing"]
            >= CALIBRATION_PENALTY["calibration_single_witness"])
    # …and a figure whose axis two witnesses confirmed says nothing of the kind
    assert "calibration_missing" not in codes(
        run_checks(dataset, "late_adaptation", [_figure_cand(cal_status="confirmed")]))


def test_every_calibration_state_is_decided_about_rather_than_falling_through():
    """The axis test listed the states that pass, so a state nobody listed was silently skipped.

    That is how "we could not establish the scale" ended up checked less than "we established it
    with one witness". The map is total, and a state added to `CAL_STATUSES` without a decision
    here fails at import rather than quietly disabling a check.
    """
    from canopy.verify.checks import AXIS_TESTABLE
    from canopy.verify.figures import CAL_STATUSES

    assert set(AXIS_TESTABLE) == set(CAL_STATUSES) | {"unknown"}


def test_the_same_group_read_in_the_recorded_unit_and_another_is_an_expression_not_a_dispute():
    """Cressman's Fig. 3b: the same bars off the degrees axis and the percent axis beside it."""
    flags = run_checks(make_dataset(), "late_adaptation",
                       [cand("A", unit="deg"), cand("A", unit="% of the 30° distortion",
                                                    candidate_id="A-pct", mean=58.0),
                        cand("B", unit="deg"), cand("B", unit="%", candidate_id="B-pct",
                                                    mean=61.0)])
    assert "unit_mismatch" not in codes(flags)
    flag = next(f for f in flags if f.code == "unit_other_expression")
    assert flag.severity == "info" and "%" in flag.message


# =============================================================== ceiling C9 — the conversion gate
# `canopy.stats.effect_sizes.convertibility` answers "may this become an SMD?" while the effect
# size is being built, and its answer reaches the row as a flag that changes no bucket. These are
# the same questions, asked on the cell, where `confidence` can act on them.
from canopy.verify.checks import CHECK_SEVERITY_PREFIXES, DF_SHORTFALL_TOLERANCE, severity_of


def evidenced(n_a=20, n_b=20) -> DatasetSpec:
    """A dataset whose group sizes the paper PRINTED, so `n_a + n_b - 2` is not an estimate."""
    spec = make_dataset()
    spec.group_a = GroupSpec(label="old subjects", n=n_a,
                             n_evidence="twenty older adults were tested")
    spec.group_b = GroupSpec(label="young subjects", n=n_b,
                             n_evidence="twenty younger adults were tested")
    return spec


def stat(**kwargs) -> Candidate:
    base = dict(candidate_id="cT", paper_id="paper1", dataset_id="ds1",
                outcome_key="late_adaptation", kind="test_statistic", status="found",
                stat_type="t", stat_value=5.25, design="independent_t", grounded=True,
                model="claude-opus-5", quote="the difference was significant, t = 5.25, p < .001")
    base.update(kwargs)
    return Candidate(**base)


def test_a_statistic_with_no_degrees_of_freedom_cannot_claim_these_two_groups():
    """The failing input: "t = 5.25, p < .001" as a post-hoc from a three-group ANOVA."""
    flags = run_checks(evidenced(n_a=12, n_b=12), "late_adaptation", [stat()])
    assert "df_missing" in codes(flags)
    assert CHECK_SEVERITY["df_missing"] == "error"
    assert "post-hoc" in next(f for f in flags if f.code == "df_missing").message


def test_an_f_with_only_a_numerator_df_is_still_missing_the_one_that_matters():
    """`test_stat_missing_df` never fired here: `df1` is not None, so the row converted."""
    flags = run_checks(evidenced(), "late_adaptation",
                       [stat(stat_type="F", stat_value=27.6, df1=1.0, design="one_way_between")])
    assert "test_stat_missing_df" not in codes(flags)
    assert "df_missing" in codes(flags)


def test_matching_degrees_of_freedom_raise_nothing():
    assert codes(run_checks(evidenced(), "late_adaptation", [stat(df=38.0)])) == []


def test_an_unexplained_df_shortfall_is_refused_even_inside_the_old_tolerance():
    """`F(1,36)` at n = 20/20 is the shape of a two-covariate ANCOVA. Gap is exactly 2.0, which
    `DF_TOLERANCE = 2.0` admitted with `gap > DF_TOLERANCE`."""
    flags = run_checks(evidenced(), "late_adaptation",
                       [stat(stat_type="F", stat_value=27.6, df1=1.0, df2=36.0,
                             design="one_way_between")])
    assert "df_shortfall_unexplained" in codes(flags)
    assert CHECK_SEVERITY["df_shortfall_unexplained"] == "error"
    assert not any(c.startswith("df_off_by_") for c in codes(flags))


def test_the_only_explanation_the_record_can_carry_is_an_n_nobody_printed():
    """C9's acceptance case, AMENDED under review M2. C9 licensed two explanations for a df that
    is not exactly n_a + n_b - 2; only one of them is checkable on anything the pipeline records.

    "t(37) at n = 20/20 with a stated single dropout is allowed and named `df_off_by_1`" needed a
    participant total the PAPER states, and nothing extracts one: the `StudyMap` has no field for
    it and the orchestrator's only call passed the analysed group sizes back in as if they were
    the total, which made the branch dead in every real run. It is deleted rather than left
    standing, and the case is refused — fail-closed, a fill-rate cost and never a wrong number —
    until the mapper grows a stated-total field. That is future mapper work, not a rule change.
    """
    from canopy.verify.checks import _shortfall_is_explained

    printed = evidenced()                                     # both n's printed in the paper
    assert _shortfall_is_explained(printed) == ""
    assert "df_shortfall_unexplained" in codes(
        run_checks(printed, "late_adaptation", [stat(df=37.0)]))
    assert not any(c.startswith("df_off_by_") for c in
                   codes(run_checks(printed, "late_adaptation", [stat(df=37.0)])))


def test_the_gate_screens_the_statistic_the_row_would_convert_and_not_every_other_one():
    """Review L4. The gate flagged EVERY convertible-design statistic in the cell, so a paper that
    prints an unrelated one-way F beside the outcome held a cell the row resolves from a printed
    t — and, before M5, bought an adjudication for it. Bound to `best_statistic`, which is the
    same selection `canopy.pipeline.run._statistic_values` makes when it builds the row."""
    from canopy.verify.checks import best_statistic

    usable = stat(candidate_id="cT", df=38.0)                       # t(38) at n = 20/20: exact
    unrelated = stat(candidate_id="cF", stat_type="F", stat_value=27.6, df=None, df1=1.0,
                     design="one_way_between",
                     quote="the main effect of block was significant, F(1) = 27.6")
    assert best_statistic([unrelated, usable]) is usable             # t outranks F
    assert codes(run_checks(evidenced(), "late_adaptation", [unrelated, usable])) == []
    # …and with nothing better in the cell, the same unrelated F is screened exactly as before
    assert "df_missing" in codes(run_checks(evidenced(), "late_adaptation", [unrelated]))


def test_a_statistic_the_extractor_ruled_inadmissible_is_not_screened_either():
    """`_statistic_values` skips it, so the row never converts from it and holding a cell for its
    degrees of freedom holds the cell for a number nothing would have used."""
    from canopy.verify.checks import best_statistic

    ruled_out = stat(candidate_id="cX", admissible=False,
                     admissible_reason="this is the practice block, not the adaptation block")
    assert best_statistic([ruled_out]) is None
    assert "df_missing" not in codes(run_checks(evidenced(), "late_adaptation", [ruled_out]))


def test_an_inferred_group_size_excuses_a_shortfall_because_the_expectation_is_an_estimate():
    flags = run_checks(make_dataset(), "late_adaptation", [stat(df=21.0)])   # n printed nowhere
    assert "df_off_by_1" in codes(flags)
    assert severity_of("df_off_by_1") == "warn"
    assert "inferred rather than printed" in next(
        f for f in flags if f.code == "df_off_by_1").message


def test_a_shortfall_bigger_than_the_tolerance_is_refused_however_it_is_explained():
    flags = run_checks(make_dataset(), "late_adaptation", [stat(df=18.0)])   # n printed nowhere
    assert "df_shortfall_unexplained" in codes(flags)
    assert DF_SHORTFALL_TOLERANCE == 2.0


def test_a_design_that_cannot_carry_the_contrast_is_not_the_conversion_gates_business():
    """`convertibility` refuses these outright; a df quibble about them would be noise."""
    for design in ("paired", "interaction", "mixed_main_effect", "ancova"):
        found = codes(run_checks(evidenced(), "late_adaptation", [stat(df=37.0, design=design)]))
        assert "df_shortfall_unexplained" not in found and "df_off_by_1" not in found, design


def test_every_family_code_resolves_to_a_declared_severity():
    assert CHECK_SEVERITY_PREFIXES
    for prefix, severity in CHECK_SEVERITY_PREFIXES.items():
        assert severity_of(f"{prefix}1") == severity
    with pytest.raises(KeyError):
        severity_of("a_code_nobody_declared")


# ----------------------------------------------------------- C9, second half: the |d| screen
def test_an_implausible_effect_names_the_denominator_as_an_early_warning():
    """The cell-level half. It NAMES the suspect denominator, which is what a reviewer needs, and
    it is a `warn`: the binding screen is on the resolved |d| the row was actually divided by
    (`confidence.dispersion_plausibility_bucket`), because nothing here can see the vote, the
    adjudicator or an SE→SD conversion (review H1)."""
    huge = [cand("A", mean=10.0, dispersion_value=5.0),
            cand("B", candidate_id="cB", mean=40.0, dispersion_value=5.0)]
    flag = next(f for f in run_checks(make_dataset(), "late_adaptation", huge)
                if f.code == "implausible_dispersion")
    assert flag.severity == "warn" and CHECK_SEVERITY["implausible_dispersion"] == "warn"
    assert "denominator" in flag.message and "5" in flag.message


def test_the_early_warning_reads_an_se_bar_too_because_it_converts_arithmetically():
    """SE is the modal shape in the live corpus (32 against 25 SD of the 57 `found` group_stats
    candidates carrying either). An `SD`-only filter never looked at them, so the cheapest
    warning was silent on the majority."""
    se = [cand("A", mean=10.0, dispersion_value=0.5, dispersion_type=DispersionType.SE),
          cand("B", candidate_id="cB", mean=40.0, dispersion_value=0.5,
               dispersion_type=DispersionType.SE)]
    flag = next(f for f in run_checks(make_dataset(), "late_adaptation", se)
                if f.code == "implausible_dispersion")
    assert "1.73" in flag.message                # SE 0.5 x sqrt(12), not the printed 0.5
    # …and a spread no arithmetic converts is left to the row, which has the distribution rules
    iqr = [cand("A", mean=10.0, dispersion_value=0.5, dispersion_type=DispersionType.IQR),
           cand("B", candidate_id="cB", mean=40.0, dispersion_value=0.5,
                dispersion_type=DispersionType.IQR)]
    assert "implausible_dispersion" not in codes(run_checks(make_dataset(), "late_adaptation",
                                                            iqr))


def test_the_plausibility_line_still_clears_the_largest_effect_in_the_live_corpus():
    """Bock d2 late adaptation resolves at |d| = 2.9610 — nothing in the run changes today."""
    from canopy.stats.effect_sizes import cohens_d
    from canopy.verify.checks import MAX_PLAUSIBLE_D

    d = cohens_d(52.49055415155991, 9.15, 12, 27.98296334051485, 7.3, 12)
    assert abs(round(d, 4)) == 2.9610 and abs(d) < MAX_PLAUSIBLE_D
