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
from canopy.verify.vote import (VoteResult, figure_of, figure_tolerance, locator_key, modality,
                                model_family, precision_tolerance, route_key, vote,
                                vote_groups)
from tests.helpers import nine

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


# --------------------------------------------------------------------------- one printed route
def test_one_printed_route_is_the_consensus_and_four_figure_routes_corroborate_it():
    """The common shape after the digitizer merge: one text extractor against four ways of
    measuring one picture. The printed value is the answer; the pictures agree with it."""
    text = text_cand("t1", 31.51, written="31.51")
    figures = [figure_cand(f"f{i}", value, route=route, model=model)
               for i, (value, route, model) in enumerate(
                   [(31.0, "pathB", OPUS), (32.0, "pathC", OPUS), (30.5, "pathD", SONNET),
                    (31.4, "ensemble", SONNET)])]
    result = vote([text, *figures])
    assert result.agreement == "agree"
    assert result.value == pytest.approx(31.51)         # printed, not the figure median
    assert "t1" in result.agreeing_ids and len(result.agreeing_ids) == 5
    assert result.figure_conflict is False


def test_four_figure_routes_cannot_outvote_one_printed_route():
    """The probe: one text reader at 31.51 against four digitizer routes clustered near 45. The
    printed value stands, the pictures are recorded as conflicting, and nothing auto-accepts."""
    text = text_cand("t1", 31.51, written="31.51")
    figures = [figure_cand(f"f{i}", value, route=route, model=model)
               for i, (value, route, model) in enumerate(
                   [(45.0, "pathB", OPUS), (45.1, "pathC", OPUS), (44.9, "pathD", SONNET),
                    (45.2, "ensemble", SONNET)])]
    result = vote([text, *figures])
    assert result.value == pytest.approx(31.51)         # NOT 45.05
    assert result.agreeing_ids == ["t1"]
    assert sorted(result.disagreeing_ids) == ["f0", "f1", "f2", "f3"]
    assert result.figure_conflict is True
    assert result.agreement == "single"                 # nothing corroborated it
    assert result.method == "printed_uncorroborated"
    assert any("the printed value stands" in note for note in result.notes)
    assert any("stands on one reader alone" in note for note in result.notes)


def test_an_uncorroborated_printed_value_cannot_be_accepted_automatically():
    from canopy.models import OrientationVerdict, VerifierVerdict
    from canopy.verify.confidence import AUTO_ACCEPT, confidence

    text = text_cand("t1", 31.51, written="31.51")
    figures = [figure_cand(f"f{i}", 45.0 + i / 10, route=route)
               for i, route in enumerate(("pathB", "pathC", "pathD", "ensemble"))]
    result = vote([text, *figures])
    oriented = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=False,
                                  agreed=True, needs_human=False)
    bucket, score, _ = confidence(result, [VerifierVerdict(candidate_id="t1", verdict="confirmed",
                                                           model=SONNET)],
                                  [], None, orientation=oriented)
    assert bucket != "auto_accept" and score < AUTO_ACCEPT


def test_two_figure_routes_cannot_outvote_one_printed_route_either():
    text = text_cand("t1", 31.51, written="31.51")
    figures = [figure_cand("f0", 45.0, route="pathC"),
               figure_cand("f1", 45.1, route="pathD", model=SONNET)]
    result = vote([text, *figures])
    assert result.value == pytest.approx(31.51)
    assert result.agreement == "single" and result.method == "printed_uncorroborated"
    assert result.figure_conflict is True


def test_a_printed_outlier_is_noted_like_a_figure_one():
    """Three printed routes, one of which read something else: the majority carries the value and
    the odd one out is named, exactly as a conflicting figure would be."""
    rows = [text_cand("a", 31.5, written="31.5"),
            text_cand("b", 31.5, written="31.5", model=SONNET),
            text_cand("c", 44.0, written="44.0", model="claude-fable-5")]
    result = vote(rows)
    assert result.agreement == "agree" and result.value == pytest.approx(31.5)
    assert result.disagreeing_ids == ["c"]
    assert any("printed route" in note and "44" in note for note in result.notes)


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
    # the outlier (38) is on the route's record — `spread`, the note — but it did not read the
    # value the route votes, so it is not among the readings that carry it
    assert route.candidate_ids == ["a", "b"]
    assert not route.consistent and route.spread == pytest.approx(7.0)
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
    """Everything the result PUBLISHES survives JSON.

    `RouteValue.positions` deliberately does not: it indexes the row list of the call that
    produced it, so outside that call it is a set of integers pointing at nothing, and writing it
    into `verify.json` would invite exactly the misreading it exists to prevent. The comparison
    is therefore between what was written and what was read back.
    """
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    reloaded = VoteResult.model_validate_json(result.model_dump_json())
    assert reloaded.model_dump() == result.model_dump()
    assert any(r.positions for r in result.routes), "the routes did not record their members"
    assert all(not r.positions for r in reloaded.routes), "a position outlived its row list"


def test_voting_never_changes_a_candidate():
    rows = [text_cand("a", 31.51), text_cand("b", 13.5, model=SONNET)]
    before = [c.model_dump() for c in rows]
    vote(rows)
    assert [c.model_dump() for c in rows] == before


# --------------------------------------------------------------------------- units and middles
def _ens(cid, mean, unit, group="B", model=OPUS, **kw):
    """A digitiser ensemble candidate — the shape two figure sources of one cell produce."""
    cand = figure_cand(cid, mean, route="ensemble", group=group, model=model,
                       cal={"y_min": 0.0, "y_max": 30.0 if unit == "deg" else 100.0,
                            "y_tick": 10.0 if unit == "deg" else 25.0}, **kw)
    return cand.model_copy(update={"unit": unit})


def test_two_readings_in_different_units_are_two_quantities_not_one_route():
    """Cressman 2010 Fig. 3b in `runs/rerun-fixed`: the map listed the bars twice — the degrees
    axis and the percentage axis beside it — and both ensembles were stamped "deg". They landed
    in one route whose "median" of the two, 39.99, was in no unit at all, and it voted."""
    a = _ens("ens:deg", 18.49, "deg")
    b = _ens("ens:pct", 61.50, "%")
    # (the fix in the digitiser now stamps each ensemble with the unit its readers named)
    result = vote([a, b], group="B", unit_hint="degrees (CCW/left of target); also percentage")
    assert result.mean == pytest.approx(18.49)
    assert result.unit_set_aside_ids == ["ens:pct"]
    assert result.agreement == "single"
    assert any("set aside" in n and "another expression" in n for n in result.notes)


def test_without_a_unit_hint_two_units_are_still_two_routes_and_never_averaged():
    a = _ens("ens:deg", 18.49, "deg")
    b = _ens("ens:pct", 61.50, "%")
    result = vote([a, b], group="B")
    assert result.mean != pytest.approx(39.995), "the average of a degree and a percent voted"
    keys = {r.route_key for r in result.routes}
    assert any("[deg]" in k for k in keys) and any("[%]" in k for k in keys)
    assert result.agreement == "disagree"


def test_a_route_that_disagrees_with_itself_does_not_vote_a_middle_none_of_it_read():
    """Two members, far apart, same unit: their median is their average — nobody's reading."""
    a = _ens("ens:one", 18.49, "deg")
    b = _ens("ens:two", 61.50, "deg")
    result = vote([a, b], group="B")
    route = result.routes[0]
    assert route.abstained and route.value is None
    assert result.mean is None and result.agreement == "disagree"
    assert any("a middle none of them read is not a reading" in n for n in result.notes)


def test_a_route_with_a_corroborated_cluster_votes_the_cluster_not_the_outlier():
    a = _ens("ens:one", 18.4, "deg")
    b = _ens("ens:two", 18.6, "deg")
    c = _ens("ens:odd", 61.5, "deg")
    result = vote([a, b, c], group="B")
    route = result.routes[0]
    assert not route.abstained and route.value == pytest.approx(18.5, abs=0.11)
    assert set(route.candidate_ids) == {"ens:one", "ens:two"}


def test_when_every_candidate_is_in_the_other_unit_the_hint_is_the_odd_one_out():
    a = _ens("ens:pct1", 61.5, "%")
    b = _ens("ens:pct2", 61.7, "%", model=SONNET)
    result = vote([a, b], group="B", unit_hint="degrees")
    assert result.unit_set_aside_ids == []
    assert result.mean == pytest.approx(61.6, abs=0.15)


# ------------------------------------------------- D2: one figure, two locators, never averaged
# Real records: Langan 2022's Fig. 1 puts the young adults in panel A and the older adults in
# panel B, and the digitiser read the SAME cell off both. Their two values are two quantities —
# averaging them produced -18.5, a number no reader wrote down and no panel contains.
def _langan(group):
    return [c for c in nine.candidates("d1f2946e7e81")
            if c.dataset_id == "d1f2946e7e81:d1" and c.outcome_key == "late_adaptation"
            and c.group == group and c.extractor_id.endswith("ensemble")]


def test_two_locators_never_average():
    res = vote(_langan("B"))
    assert res.mean != -18.5 and res.agreement == "disagree" and res.method == "locator_conflict"
    assert "Fig. 1A" in " ".join(res.notes) and "Fig. 1B" in " ".join(res.notes)


def test_locator_key_partitions_only_figure_candidates():
    """A place is part of a route only for a reading that was measured somewhere.

    The modality of a digitised reading is `figure:<path>` — the path is how the picture was
    measured — so the family is the first segment, not the whole string. A text or table or
    statistic reading keeps the two-part route it always had.
    """
    for p in ("3570e4ce2a9c", "5039533c85ef", "b511dbb76fa6", "b7523a41b03a", "592b3b55a318",
              "d1f2946e7e81"):
        for c in nine.candidates(p):
            if modality(c).split(":", 1)[0] not in ("figure", "digitize"):
                assert locator_key(c) == "" and route_key(c).count("/") == 1
            else:
                assert route_key(c).count("/") == (2 if c.locator.strip() else 1)


def test_duplicate_ids_under_two_locators_are_all_considered_and_ids_keep_their_shape():
    cands = _langan("B")
    assert len({c.candidate_id for c in cands}) < len(cands)
    res = vote(cands)
    assert res.n_candidates_considered == len(cands)
    assert all(isinstance(i, str) for i in res.agreeing_ids + res.disagreeing_ids)


def test_two_figures_that_agree_are_corroboration_not_a_conflict():
    """Fix round 1, MAJOR 2. The conflict is about two places in ONE picture.

    Wang 2011 plots the gradual group's aftereffect in Fig 3 and again in Fig 4, and the digitiser
    read both: -2.85 and -3.45, agreeing to 0.1. That is the corroboration the vote exists to
    reward — the quantity was drawn twice and measured twice — not the panel-boundary coincidence
    `locator_conflict` is for. Treating it as a conflict withheld a value two independent pictures
    established, and told the reviewer the readings came from "the same figure", which is false.
    """
    cands = [c for c in nine.candidates("592b3b55a318")
             if c.dataset_id == "592b3b55a318:d2" and c.outcome_key == "aftereffect"
             and c.group == "B" and c.extractor_id.endswith("ensemble")]
    assert len({figure_of(c) for c in cands}) == 2, "the fixture no longer spans two figures"
    res = vote(cands)
    assert res.agreement == "agree" and res.method != "locator_conflict"
    assert res.mean == pytest.approx(-3.15, abs=0.01)
