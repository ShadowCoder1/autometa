"""Task 8, step 2: the agreement vote (spec §3.3(2) + amendment G).

Two readers that fail the same way are one reader, so a value is only "accepted by vote" when at
least two *heterogeneous* candidates agree — different modality (text / table / figure) or
different model family. The vote is pure code: it compares numbers with a tolerance that comes
from how the number was printed (text) or from the axis it was measured against (figures), and it
never asks a model anything.
"""
from __future__ import annotations

import pytest

from canopy.models import Candidate, DispersionType, SourceKind
from canopy.verify.vote import (VoteResult, figure_tolerance, model_family, precision_tolerance,
                                route_key, vote, vote_groups)

OPUS, SONNET = "claude-opus-5", "claude-sonnet-5"


def text_cand(cid, mean, *, model=OPUS, variant="table_first", written="", group="A", **kwargs):
    return Candidate(candidate_id=cid, dataset_id="ds1", outcome_key="late_adaptation",
                     kind="group_stats", group=group, status=kwargs.pop("status", "found"),
                     source_kind=SourceKind.text_mean_sd, mean=mean, n=12,
                     dispersion_value=kwargs.pop("dispersion_value", 11.12),
                     dispersion_type=kwargs.pop("dispersion_type", DispersionType.SD),
                     unit="deg", value_as_written=written or f"{mean} ± 11.12 deg",
                     grounded=True, route="text", model=model,
                     extractor_id=f"text:{variant}:{model}", **kwargs)


def figure_cand(cid, mean, *, model=OPUS, route="pathC", group="A", cal=None, **kwargs):
    provenance = {"cal": {"y_min": 0.0, "y_max": 60.0, "y_tick": 10.0} if cal is None else cal}
    provenance.update(kwargs.pop("pixel_provenance", {}))
    return Candidate(candidate_id=cid, dataset_id="ds1", outcome_key="late_adaptation",
                     kind="group_stats", group=group, status="found",
                     source_kind=SourceKind.figure_bar, mean=mean, n=12,
                     dispersion_value=kwargs.pop("dispersion_value", 11.0),
                     dispersion_type=DispersionType.SD, unit="deg",
                     route="figure", model=model, pixel_provenance=provenance,
                     extractor_id=f"digitize:{route}:{model}:v1", **kwargs)


# --------------------------------------------------------------------------- routes
def test_model_family_ignores_the_version_and_the_date():
    assert model_family("claude-opus-5") == "claude-opus"
    assert model_family("claude-opus-5-20260401") == "claude-opus"
    assert model_family("claude-haiku-4-5") == "claude-haiku"
    assert model_family(SONNET) != model_family(OPUS)


def test_a_route_is_a_modality_and_a_model_family():
    assert route_key(text_cand("a", 1.0)) == "text/claude-opus"
    assert route_key(text_cand("b", 1.0, model=SONNET)) == "text/claude-sonnet"
    assert route_key(figure_cand("c", 1.0)) == "figure:pathC/claude-opus"
    assert route_key(figure_cand("d", 1.0, route="ensemble")) == "figure:ensemble/claude-opus"


def test_two_variants_of_the_same_model_share_a_route():
    """Same modality, same family: they fail the same way, so they are one voter."""
    a = text_cand("a", 31.51, variant="table_first")
    b = text_cand("b", 31.51, variant="narrative_first")
    assert route_key(a) == route_key(b)
    result = vote([a, b])
    assert result.agreement == "single"


# --------------------------------------------------------------------------- tolerances
def test_printed_precision_is_half_of_the_last_printed_digit():
    assert precision_tolerance(text_cand("a", 31.51, written="31.51 ± 11.12")) == pytest.approx(0.005)
    assert precision_tolerance(text_cand("a", 31.5, written="31.5")) == pytest.approx(0.05)
    assert precision_tolerance(text_cand("a", 32.0, written="32")) == pytest.approx(0.5)


def test_printed_precision_falls_back_to_the_number_itself():
    assert precision_tolerance(text_cand("a", 31.51, written="")) == pytest.approx(0.005)


def test_figure_tolerance_is_the_looser_of_two_percent_of_range_and_half_a_tick():
    # range 60 → 2% = 1.2; tick 10 → half a tick = 5 → the tick wins
    assert figure_tolerance(figure_cand("a", 30.0)) == pytest.approx(5.0)
    # a fine tick makes the 2% rule bind
    tight = figure_cand("a", 30.0, cal={"y_min": 0.0, "y_max": 60.0, "y_tick": 1.0})
    assert figure_tolerance(tight) == pytest.approx(1.2)


def test_figure_tolerance_uses_the_axis_range_argument_when_there_is_no_calibration():
    bare = figure_cand("a", 30.0, cal={})
    assert figure_tolerance(bare, axis_range=50.0) == pytest.approx(1.0)
    assert figure_tolerance(bare) is None


# --------------------------------------------------------------------------- the vote
def test_two_heterogeneous_readers_that_agree_are_accepted():
    a = text_cand("a", 31.51, model=OPUS)
    b = text_cand("b", 31.51, model=SONNET, variant="narrative_first")
    result = vote([a, b])
    assert result.agreement == "agree"
    assert result.value == pytest.approx(31.51) and result.mean == pytest.approx(31.51)
    assert sorted(result.agreeing_ids) == ["a", "b"] and result.disagreeing_ids == []
    assert result.method == "printed_precision" and result.tolerance == pytest.approx(0.005)
    assert result.needs_third_candidate is False


def test_agreement_survives_a_difference_inside_the_printed_precision():
    """Both readers say the paper prints one decimal; 31.50 and 31.52 are the same printed value."""
    a = text_cand("a", 31.5, written="31.5")
    b = text_cand("b", 31.52, written="31.5", model=SONNET)
    result = vote([a, b])
    assert result.agreement == "agree" and result.tolerance == pytest.approx(0.05)
    assert result.value in (31.5, 31.52)                # a value a source actually reported


def test_the_coarser_printed_precision_decides_a_text_pair():
    """A paper that prints 31.5 in the text and 31.51 in a table is printing ONE number at two
    precisions. Half a unit in the last digit of the LESS precise reading is what separates that
    from a discrepancy, so they agree — and the finer reading is the better record of it."""
    a = text_cand("a", 31.51, written="31.51")
    b = text_cand("b", 31.5, written="31.5", model=SONNET)
    result = vote([a, b])
    assert result.agreement == "agree"
    assert result.tolerance == pytest.approx(0.05)      # the coarser claim, not 0.005
    assert result.value == pytest.approx(31.51)         # the more precise of the two
    assert result.needs_third_candidate is False
    assert any("different precisions" in note for note in result.notes)


def test_a_difference_the_coarser_precision_cannot_absorb_still_disagrees():
    """31.5 and 31.6 differ by a whole unit in the last printed digit: a different number."""
    a = text_cand("a", 31.5, written="31.5")
    b = text_cand("b", 31.6, written="31.6", model=SONNET)
    result = vote([a, b])
    assert result.agreement == "disagree" and result.tolerance == pytest.approx(0.05)
    assert result.needs_third_candidate is True


def test_a_coarse_reading_does_not_absorb_a_plainly_different_number():
    """Even a value printed as a whole number ("32", ±0.5) cannot absorb 33.0."""
    a = text_cand("a", 32.0, written="32")
    b = text_cand("b", 33.0, written="33.0", model=SONNET)
    result = vote([a, b])
    assert result.agreement == "disagree" and result.tolerance == pytest.approx(0.5)


def test_a_figure_read_cannot_bridge_two_text_readers_who_disagree():
    """The failure this layer exists to prevent: a loose figure tolerance must not be applied to a
    text-versus-text comparison, blending two contradictory transcriptions into one number."""
    a = text_cand("a", 31.51, written="31.51")
    b = text_cand("b", 33.0, written="33.0", model=SONNET, variant="narrative_first")
    figure = figure_cand("f", 32.4, model="claude-fable-5")
    result = vote([a, b, figure])
    assert result.agreement == "disagree"
    assert result.needs_third_candidate is True
    assert result.value is None
    assert sorted(result.agreeing_ids) == []
    assert any("cannot decide between them" in note for note in result.notes)


def test_a_figure_within_tolerance_corroborates_the_printed_value():
    a = text_cand("a", 31.51, written="31.51")
    b = text_cand("b", 31.51, written="31.51", model=SONNET, variant="narrative_first")
    figure = figure_cand("f", 32.4, model="claude-fable-5")
    result = vote([a, b, figure])
    assert result.agreement == "agree"
    assert result.value == pytest.approx(31.51)         # the printed value, not a blend
    assert sorted(result.agreeing_ids) == ["a", "b", "f"]
    assert result.figure_conflict is False


def test_a_figure_outside_tolerance_is_flagged_and_the_printed_value_stands():
    a = text_cand("a", 31.51, written="31.51")
    b = text_cand("b", 31.51, written="31.51", model=SONNET, variant="narrative_first")
    figure = figure_cand("f", 45.0, model="claude-fable-5")
    result = vote([a, b, figure])
    assert result.agreement == "agree"
    assert result.value == pytest.approx(31.51)
    assert result.figure_conflict is True
    assert result.disagreeing_ids == ["f"]
    assert any("the printed value stands" in note for note in result.notes)


def test_the_resolved_value_is_one_a_source_reported():
    """Three readers, two of whom report the same number: that number wins, not their mean."""
    rows = [text_cand("a", 31.5, written="31.5"),
            text_cand("b", 31.5, written="31.5", model=SONNET),
            text_cand("c", 31.52, written="31.5", model="claude-fable-5")]
    result = vote(rows)
    assert result.value == pytest.approx(31.5)


def test_two_text_readers_that_disagree_ask_for_a_third_candidate():
    a = text_cand("a", 31.51)
    b = text_cand("b", 13.5, model=SONNET, variant="narrative_first", written="13.5")
    result = vote([a, b])
    assert result.agreement == "disagree"
    assert result.needs_third_candidate is True
    assert sorted(result.disagreeing_ids) == ["a", "b"]
    assert result.value is None


def test_a_third_candidate_breaks_the_tie():
    a = text_cand("a", 31.51)
    b = text_cand("b", 13.5, model=SONNET, variant="narrative_first", written="13.5")
    c = text_cand("c", 31.51, model="claude-fable-5", variant="narrative_first")
    result = vote([a, b, c])
    assert result.agreement == "agree"
    assert sorted(result.agreeing_ids) == ["a", "c"] and result.disagreeing_ids == ["b"]
    assert result.needs_third_candidate is False


def test_one_reader_alone_is_never_accepted_by_vote():
    result = vote([text_cand("a", 31.51)])
    assert result.agreement == "single" and result.value == pytest.approx(31.51)
    assert result.agreeing_ids == ["a"] and result.method == "single"
    assert result.needs_third_candidate is False


def test_no_usable_candidate_is_an_empty_vote():
    empty = text_cand("a", None, status="not_on_these_pages")
    result = vote([empty])
    assert result.agreement == "none" and result.value is None
    assert result.agreeing_ids == [] and result.method == "none"


# --------------------------------------------------------------------------- figures
def test_figure_candidates_agree_within_the_figure_tolerance():
    a = figure_cand("a", 31.0, route="pathC")
    b = figure_cand("b", 33.0, route="pathD", model=SONNET)
    result = vote([a, b])
    assert result.agreement == "agree" and result.method == "figure_tolerance"
    assert result.tolerance == pytest.approx(5.0)
    assert result.value == pytest.approx(32.0)          # per-cell median of the agreeing routes


def test_figure_candidates_further_apart_than_the_tolerance_disagree():
    a = figure_cand("a", 20.0, route="pathC")
    b = figure_cand("b", 45.0, route="pathD", model=SONNET)
    result = vote([a, b])
    assert result.agreement == "disagree" and result.needs_third_candidate is False


def test_a_figure_and_a_text_reader_are_compared_with_the_figure_tolerance():
    a = text_cand("a", 31.51)
    b = figure_cand("b", 33.0, model=SONNET)
    result = vote([a, b])
    assert result.agreement == "agree" and result.method == "figure_tolerance"
    assert result.tolerance == pytest.approx(5.0)


def test_the_median_of_a_route_represents_it():
    """Three read-outs from one path are one voter, and its value is their median."""
    rows = [figure_cand("a", 30.0, route="pathD"), figure_cand("b", 31.0, route="pathD"),
            figure_cand("c", 38.0, route="pathD")]
    result = vote(rows + [text_cand("t", 31.51, model=SONNET)])
    route = next(r for r in result.routes if r.route_key.startswith("figure:pathD"))
    assert route.value == pytest.approx(31.0)
    assert route.candidate_ids == ["a", "b", "c"]
    assert result.agreement == "agree"


def test_a_route_that_disagrees_with_itself_is_recorded():
    rows = [figure_cand("a", 10.0, route="pathD"), figure_cand("b", 50.0, route="pathD")]
    result = vote(rows)
    route = result.routes[0]
    assert route.consistent is False and route.spread == pytest.approx(20.0)
    assert any("disagree" in note for note in result.notes)


def test_the_digitizer_sigma_and_the_spread_travel_with_the_result():
    a = figure_cand("a", 31.0, route="ensemble", sigma=0.8)
    b = text_cand("b", 31.51, model=SONNET)
    result = vote([a, b])
    assert result.sigma == pytest.approx(0.8)
    assert result.mad is not None


# --------------------------------------------------------------------------- resolved values
def test_the_vote_carries_the_dispersion_and_n_of_the_agreeing_readers():
    a = text_cand("a", 31.51, dispersion_value=11.12)
    b = text_cand("b", 31.51, dispersion_value=11.12, model=SONNET)
    result = vote([a, b])
    assert result.dispersion_value == pytest.approx(11.12)
    assert result.dispersion_type is DispersionType.SD
    assert result.n == 12 and result.unit == "deg"


def test_a_dispersion_disagreement_is_noted_without_blocking_the_mean():
    a = text_cand("a", 31.51, dispersion_value=11.12, dispersion_type=DispersionType.SD)
    b = text_cand("b", 31.51, dispersion_value=3.21, dispersion_type=DispersionType.SE,
                  model=SONNET)
    result = vote([a, b])
    assert result.agreement == "agree" and result.value == pytest.approx(31.51)
    assert any("dispersion" in note for note in result.notes)


# --------------------------------------------------------------------------- plumbing
def test_a_vote_must_be_about_one_group():
    with pytest.raises(ValueError):
        vote([text_cand("a", 31.51, group="A"), text_cand("b", 12.28, group="B")])


def test_the_group_can_be_selected_explicitly():
    rows = [text_cand("a", 31.51, group="A"), text_cand("b", 12.28, group="B")]
    assert vote(rows, group="B").value == pytest.approx(12.28)


def test_vote_groups_returns_one_result_per_group():
    rows = [text_cand("a", 31.51, group="A"), text_cand("b", 31.51, group="A", model=SONNET),
            text_cand("c", 12.28, group="B"), text_cand("d", 12.28, group="B", model=SONNET)]
    results = vote_groups(rows)
    assert sorted(results) == ["A", "B"]
    assert results["A"].agreement == "agree" and results["B"].value == pytest.approx(12.28)


def test_a_vote_result_round_trips_through_json():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    assert VoteResult.model_validate_json(result.model_dump_json()) == result


def test_voting_never_changes_a_candidate():
    rows = [text_cand("a", 31.51), text_cand("b", 13.5, model=SONNET)]
    before = [c.model_dump() for c in rows]
    vote(rows)
    assert [c.model_dump() for c in rows] == before
