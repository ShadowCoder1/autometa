"""Task 8, step 6: the confidence bucket and `resolve_cell` (spec §3.3(6), amendment F).

Confidence is code: it weighs where the value came from, whether its quote is in the paper, how
many *heterogeneous* readers agreed, what the adversarial verifier said, how tightly the digitizer
routes cluster, and every consistency flag. It never asks a model anything, and an error-severity
flag, an unresolved direction or an adjudicator asking for a human always wins.
"""
from __future__ import annotations

import pytest

from canopy.models import (AdjudicatedGroup, Adjudication, Candidate, CheckFlag, DatasetSpec,
                           DispersionType, GroupSpec, OrientationVerdict, OutcomeSources, Source,
                           SourceKind, VerifierVerdict)
from canopy.verify.confidence import (ACCEPT_WITH_NOTE, AUTO_ACCEPT, DELTA_D_LIMIT,
                                      DIGITIZATION_SE_SHARE, confidence, figure_gate, resolve_cell)
from canopy.verify.vote import vote

OPUS, SONNET = "claude-opus-5", "claude-sonnet-5"


def dataset() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="ds1", cluster_id="paper1",
        group_a=GroupSpec(label="old subjects", n=12),
        group_b=GroupSpec(label="young subjects", n=12),
        outcomes=[OutcomeSources(outcome_key="late_adaptation", units="deg",
                                 higher_is_better=False,
                                 sources=[Source(kind=SourceKind.text_mean_sd, page=3)])])


def text_cand(cid, mean, *, model=OPUS, group="A", **kwargs):
    return Candidate(candidate_id=cid, dataset_id="ds1", outcome_key="late_adaptation",
                     kind="group_stats", group=group, status=kwargs.pop("status", "found"),
                     source_kind=SourceKind.text_mean_sd, mean=mean, n=12,
                     dispersion_value=kwargs.pop("dispersion_value", 11.12),
                     dispersion_type=kwargs.pop("dispersion_type", DispersionType.SD),
                     unit="deg", value_as_written=f"{mean} ± 11.12 deg",
                     grounded=kwargs.pop("grounded", True),
                     route="text", model=model, extractor_id=f"text:v:{model}",
                     quote=kwargs.pop("quote",
                                      "the mean direction error was 31.51 ± 11.12 deg"), **kwargs)


def fig_cand(cid, mean, *, model=OPUS, group="A", route="pathC", sigma=None, sd=11.0, n=12):
    return Candidate(candidate_id=cid, dataset_id="ds1", outcome_key="late_adaptation",
                     kind="group_stats", group=group, status="found",
                     source_kind=SourceKind.figure_bar, mean=mean, n=n, dispersion_value=sd,
                     dispersion_type=DispersionType.SD, unit="deg", route="figure", model=model,
                     sigma=sigma, extractor_id=f"digitize:{route}:{model}:v1",
                     pixel_provenance={"cal": {"y_min": 0.0, "y_max": 60.0, "y_tick": 10.0}})


ORIENTED = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=False, agreed=True,
                              needs_human=False, direction_stated_in_text="a_greater")
CONFIRMED = [VerifierVerdict(candidate_id="a", verdict="confirmed", reason="page 3 prints it",
                             model=SONNET)]


# --------------------------------------------------------------------------- the buckets
def test_two_agreeing_grounded_readers_with_a_confirmation_are_accepted_automatically():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    bucket, score, reasons = confidence(result, CONFIRMED, [], None, orientation=ORIENTED)
    assert bucket == "auto_accept" and score >= AUTO_ACCEPT
    assert any("agree" in r for r in reasons) and any("confirmed" in r for r in reasons)


def test_a_single_reader_is_never_accepted_automatically():
    result = vote([text_cand("a", 31.51)])
    bucket, score, reasons = confidence(result, CONFIRMED, [], None, orientation=ORIENTED)
    assert bucket == "accept_with_note" and score < AUTO_ACCEPT
    assert any("one" in r or "single" in r for r in reasons)


def test_a_refuted_value_goes_to_a_human():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    refuted = [VerifierVerdict(candidate_id="a", verdict="refuted",
                               reason="that row is the baseline block", model=SONNET)]
    bucket, score, reasons = confidence(result, refuted, [], None, orientation=ORIENTED)
    assert bucket == "needs_human"
    assert any("refuted" in r for r in reasons)


def test_an_error_flag_sends_the_cell_to_a_human_whatever_the_score():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    flags = [CheckFlag(code="quote_not_grounded", severity="error", message="not in the paper",
                       candidate_ids=["a"])]
    bucket, _, reasons = confidence(result, CONFIRMED, flags, None, orientation=ORIENTED)
    assert bucket == "needs_human"
    assert any("quote_not_grounded" in r for r in reasons)


def test_warnings_cost_points_without_forcing_a_human():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    clean, _, _ = confidence(result, CONFIRMED, [], None, orientation=ORIENTED)
    warned, score, _ = confidence(
        result, CONFIRMED,
        [CheckFlag(code="n_mismatch", severity="warn", message="n 9 vs 12")], None,
        orientation=ORIENTED)
    assert warned in ("auto_accept", "accept_with_note")
    assert score < confidence(result, CONFIRMED, [], None, orientation=ORIENTED)[1]
    assert clean == "auto_accept"


def test_an_ungrounded_quote_costs_more_than_a_short_one():
    both = [text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)]
    result = vote(both)
    short = vote([text_cand("a", 31.51, quote="31.51"), text_cand("b", 31.51, model=SONNET)])
    ungrounded = vote([text_cand("a", 31.51, grounded=False),
                       text_cand("b", 31.51, model=SONNET)])
    base = confidence(result, CONFIRMED, [], None, orientation=ORIENTED)[1]
    assert confidence(short, CONFIRMED, [], None, orientation=ORIENTED)[1] < base
    assert confidence(ungrounded, CONFIRMED, [], None, orientation=ORIENTED)[1] \
        < confidence(short, CONFIRMED, [], None, orientation=ORIENTED)[1]


def test_a_cell_with_no_value_at_all_needs_a_human():
    result = vote([text_cand("a", None, status="not_on_these_pages")])
    bucket, score, reasons = confidence(result, [], [], None, orientation=ORIENTED)
    assert bucket == "needs_human" and score == 0.0
    assert any("no value" in r for r in reasons)


def test_a_disagreement_needs_a_human():
    result = vote([text_cand("a", 31.51), text_cand("b", 13.5, model=SONNET)])
    bucket, _, reasons = confidence(result, [], [], None, orientation=ORIENTED)
    assert bucket == "needs_human" and any("disagree" in r for r in reasons)


# --------------------------------------------------------------------------- orientation
def test_an_unresolved_direction_needs_a_human():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    unresolved = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=None,
                                    needs_human=True)
    bucket, _, reasons = confidence(result, CONFIRMED, [], None, orientation=unresolved)
    assert bucket == "needs_human" and any("direction" in r for r in reasons)


def test_a_missing_orientation_needs_a_human():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    bucket, _, _ = confidence(result, CONFIRMED, [], None)
    assert bucket == "needs_human"


# --------------------------------------------------------------------------- adjudication
def test_an_adjudicated_cell_is_never_accepted_automatically():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    ruling = Adjudication(dataset_id="ds1", outcome_key="late_adaptation", rationale="page 3",
                          groups=[AdjudicatedGroup(group="A", mean=31.51, n=12)])
    bucket, _, reasons = confidence(result, CONFIRMED, [], ruling, orientation=ORIENTED)
    assert bucket == "accept_with_note"
    assert any("adjudicat" in r for r in reasons)


def test_an_adjudicator_that_asks_for_a_human_gets_one():
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    ruling = Adjudication(dataset_id="ds1", outcome_key="late_adaptation", needs_human=True,
                          rationale="the paper reports two different blocks")
    bucket, _, _ = confidence(result, CONFIRMED, [], ruling, orientation=ORIENTED)
    assert bucket == "needs_human"


def test_a_row_only_table_match_caps_the_cell():
    """Numbers found in the named row but not the named column may be the other group's."""
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    flags = [CheckFlag(code="quote_row_only", severity="warn",
                       message="found in the row but not the column", candidate_ids=["a"])]
    bucket, score, reasons = confidence(result, CONFIRMED, flags, None, orientation=ORIENTED)
    assert bucket == "accept_with_note" and score <= 0.70
    assert any("other group" in r for r in reasons)


def test_a_figure_that_contradicts_the_printed_value_costs_points():
    text = [text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)]
    clean = confidence(vote(text), CONFIRMED, [], None, orientation=ORIENTED)
    conflicted = vote([*text, fig_cand("f", 45.0, model="claude-fable-5")])
    assert conflicted.figure_conflict is True
    scored = confidence(conflicted, CONFIRMED, [], None, orientation=ORIENTED)
    assert scored[1] < clean[1]
    assert any("figure read disagrees" in r for r in scored[2])


def test_a_grounded_adjudicated_quote_counts_as_evidence():
    """A ruling that quoted the paper is evidence; `ground_adjudication` already checked it."""
    rows = [text_cand("a", 31.51), text_cand("b", 13.5, model=SONNET)]
    quoted = Adjudication(
        dataset_id="ds1", outcome_key="late_adaptation", rationale="page 4 prints it",
        groups=[AdjudicatedGroup(group="A", mean=31.51, n=12, quote="the mean direction error "
                                                                   "was 31.51 deg",
                                 page=4, grounded=True, grounding_similarity=1.0)])
    bare = Adjudication(dataset_id="ds1", outcome_key="late_adaptation", rationale="page 4",
                        groups=[AdjudicatedGroup(group="A", mean=31.51, n=12)])
    with_quote = confidence(vote(rows), [], [], quoted, candidates=rows, orientation=ORIENTED)
    without = confidence(vote(rows), [], [], bare, candidates=rows, orientation=ORIENTED)
    assert with_quote[1] > without[1]
    assert with_quote[0] == "accept_with_note"


# --------------------------------------------------------------------------- amendment F
def test_digitised_routes_that_imply_the_same_effect_pass_the_gate():
    rows = [fig_cand("a1", 31.5, route="pathC", sigma=0.2),
            fig_cand("a2", 31.6, route="pathD", model=SONNET, sigma=0.2),
            fig_cand("b1", 12.3, route="pathC", group="B", sigma=0.2),
            fig_cand("b2", 12.4, route="pathD", group="B", model=SONNET, sigma=0.2)]
    ok, delta, se_share, reasons = figure_gate(rows, 12, 12)
    assert ok is True and delta < DELTA_D_LIMIT and se_share < DIGITIZATION_SE_SHARE


def test_digitised_routes_that_imply_different_effects_fail_the_gate():
    rows = [fig_cand("a1", 20.0, route="pathC", sigma=0.2),
            fig_cand("a2", 40.0, route="pathD", model=SONNET, sigma=0.2),
            fig_cand("b1", 12.3, route="pathC", group="B", sigma=0.2),
            fig_cand("b2", 12.4, route="pathD", group="B", model=SONNET, sigma=0.2)]
    ok, delta, _, reasons = figure_gate(rows, 12, 12)
    assert ok is False and delta > DELTA_D_LIMIT
    assert any("across routes" in r for r in reasons)


def test_a_large_digitisation_uncertainty_fails_the_gate():
    rows = [fig_cand("a1", 31.5, route="pathC", sigma=6.0),
            fig_cand("a2", 31.6, route="pathD", model=SONNET, sigma=6.0),
            fig_cand("b1", 12.3, route="pathC", group="B", sigma=6.0),
            fig_cand("b2", 12.4, route="pathD", group="B", model=SONNET, sigma=6.0)]
    ok, _, se_share, reasons = figure_gate(rows, 12, 12)
    assert ok is False and se_share > DIGITIZATION_SE_SHARE
    assert any("digitisation" in r for r in reasons)


def test_one_digitiser_route_cannot_satisfy_the_cross_route_gate():
    rows = [fig_cand("a1", 31.5, route="pathC", sigma=0.2),
            fig_cand("b1", 12.3, route="pathC", group="B", sigma=0.2)]
    ok, delta, _, reasons = figure_gate(rows, 12, 12)
    assert ok is False and delta is None
    assert any("one" in r for r in reasons)


def test_a_figure_only_cell_needs_the_gate_to_be_accepted_automatically():
    rows = [fig_cand("a1", 31.5, route="pathC", sigma=0.2),
            fig_cand("a2", 31.6, route="pathD", model=SONNET, sigma=0.2),
            fig_cand("b1", 12.3, route="pathC", group="B", sigma=0.2),
            fig_cand("b2", 12.4, route="pathD", group="B", model=SONNET, sigma=0.2)]
    result = vote(rows, group="A")
    good = confidence(result, CONFIRMED, [], None, candidates=rows, n_a=12, n_b=12,
                      orientation=ORIENTED)
    assert good[0] == "auto_accept"

    noisy = [fig_cand(c.candidate_id, c.mean, route=c.extractor_id.split(":")[1],
                      model=c.model, group=c.group, sigma=6.0) for c in rows]
    bad = confidence(vote(noisy, group="A"), CONFIRMED, [], None, candidates=noisy, n_a=12,
                     n_b=12, orientation=ORIENTED)
    assert bad[0] == "accept_with_note"
    assert any("digitisation" in r for r in bad[2])


def test_the_gate_does_not_apply_when_a_text_reader_agrees():
    rows = [fig_cand("a1", 31.5, route="pathC", sigma=6.0), text_cand("t", 31.51, model=SONNET)]
    bucket, _, reasons = confidence(vote(rows, group="A"), CONFIRMED, [], None, candidates=rows,
                                    n_a=12, n_b=12, orientation=ORIENTED)
    assert bucket == "auto_accept"


# --------------------------------------------------------------------------- resolve_cell
def test_resolve_cell_assembles_one_verdict_per_group():
    rows = [text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)]
    verdict = resolve_cell(dataset(), "late_adaptation", "A", rows, verdicts=CONFIRMED,
                           orientation=ORIENTED, n_a=12, n_b=12)
    assert verdict.dataset_id == "ds1" and verdict.outcome_key == "late_adaptation"
    assert verdict.group == "A" and verdict.agreement == "agree"
    assert verdict.mean == pytest.approx(31.51) and verdict.n == 12
    assert verdict.dispersion_type is DispersionType.SD and verdict.unit == "deg"
    assert verdict.route == "text" and verdict.confidence == "auto_accept"
    assert verdict.needs_human is False and verdict.higher_is_better is False
    assert sorted(verdict.candidate_ids) == ["a", "b"]
    assert verdict.verifier_verdict == "confirmed"
    assert verdict.vote_method == "printed_precision" and verdict.vote_tolerance is not None


def test_resolve_cell_runs_the_checks_when_none_are_given():
    rows = [text_cand("a", 31.51, grounded=False), text_cand("b", 31.51, model=SONNET)]
    verdict = resolve_cell(dataset(), "late_adaptation", "A", rows, orientation=ORIENTED)
    assert "quote_not_grounded" in [f.code for f in verdict.flags]
    assert verdict.needs_human is True and verdict.confidence == "needs_human"


def test_an_adjudicated_value_replaces_the_voted_one():
    rows = [text_cand("a", 31.51), text_cand("b", 13.5, model=SONNET)]
    ruling = Adjudication(dataset_id="ds1", outcome_key="late_adaptation",
                          rationale="the table on page 4 prints 31.51",
                          groups=[AdjudicatedGroup(group="A", mean=31.51, n=12,
                                                   dispersion_value=11.12,
                                                   dispersion_type=DispersionType.SD, unit="deg",
                                                   chosen_candidate_ids=["a"],
                                                   reason="page 4")])
    verdict = resolve_cell(dataset(), "late_adaptation", "A", rows, adjudication=ruling,
                           orientation=ORIENTED)
    assert verdict.adjudicated is True and verdict.mean == pytest.approx(31.51)
    assert verdict.candidate_ids == ["a"] and verdict.route == "adjudicated"
    assert verdict.confidence == "accept_with_note"
    assert "page 4" in verdict.adjudication_rationale or "31.51" in verdict.adjudication_rationale


def test_resolve_cell_passes_the_third_candidate_request_through():
    rows = [text_cand("a", 31.51), text_cand("b", 13.5, model=SONNET)]
    verdict = resolve_cell(dataset(), "late_adaptation", "A", rows, orientation=ORIENTED)
    assert verdict.needs_third_candidate is True and verdict.needs_human is True


def test_resolve_cell_records_the_orientation_evidence():
    rows = [text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)]
    oriented = ORIENTED.model_copy(update={"quotes": ["direction error, in degrees"],
                                           "reason": "the axis is labelled error"})
    verdict = resolve_cell(dataset(), "late_adaptation", "A", rows, orientation=oriented)
    assert verdict.higher_is_better is False
    assert "direction error" in verdict.orientation_evidence


def test_resolve_cell_round_trips_through_json():
    from canopy.models import Verdict

    rows = [text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)]
    verdict = resolve_cell(dataset(), "late_adaptation", "A", rows, verdicts=CONFIRMED,
                           orientation=ORIENTED)
    assert Verdict.model_validate_json(verdict.model_dump_json()) == verdict


def test_the_thresholds_are_ordered():
    assert 0.0 < ACCEPT_WITH_NOTE < AUTO_ACCEPT <= 1.0


# ------------------------------------------------------------------ task 16: caps never compound
def test_a_single_witness_calibration_caps_a_cell_but_never_sinks_it():
    """R2: a cap means `accept_with_note`, whatever else is capping the same cell."""
    from canopy.verify.confidence import ACCEPT_WITH_NOTE, CAPPING_FLAGS

    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    flags = [CheckFlag(code="calibration_single_witness", severity="warn",
                       message="only the OCR ladder calibrated this axis", candidate_ids=["a"])]
    bucket, score, reasons = confidence(result, CONFIRMED, flags, None, orientation=ORIENTED)
    assert bucket == "accept_with_note" and score <= 0.70
    assert any("uncorroborated" in r for r in reasons)
    assert "calibration_single_witness" in CAPPING_FLAGS
    assert score >= ACCEPT_WITH_NOTE


def test_every_cap_in_combination_still_means_accept_with_note():
    """No pile-up of caps may add up to `needs_human`: only evidence does that."""
    from canopy.verify.confidence import ACCEPT_WITH_NOTE, CAPPING_FLAGS

    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    flags = [CheckFlag(code=code, severity="warn", message=code, candidate_ids=["a"])
             for code in sorted(CAPPING_FLAGS)]
    bucket, score, _ = confidence(result, CONFIRMED, flags, None, orientation=ORIENTED)
    assert bucket == "accept_with_note", "caps composed downwards into needs_human"
    assert score >= ACCEPT_WITH_NOTE


def test_a_refuted_calibration_convicts_the_ladder_not_the_reading():
    """`calibration_refuted` is an `error` in the queue, but it does not force a human."""
    from canopy.verify.confidence import NON_FORCING_ERRORS

    result = vote([text_cand("a", 31.3), text_cand("b", 31.3, model=SONNET)])
    flags = [CheckFlag(code="calibration_refuted", severity="error",
                       message="the ticks max at 4 and the readers agree on 33.3",
                       candidate_ids=["a"])]
    bucket, score, reasons = confidence(result, CONFIRMED, flags, None, orientation=ORIENTED)
    assert "calibration_refuted" in NON_FORCING_ERRORS
    assert bucket == "accept_with_note" and score <= 0.70
    assert any("the calibration is wrong, not the value" in r for r in reasons)
    # …whereas a calibration nobody could settle IS a human's problem
    disputed = [CheckFlag(code="calibration_disputed", severity="error", message="x",
                          candidate_ids=["a"])]
    assert confidence(result, CONFIRMED, disputed, None, orientation=ORIENTED)[0] == "needs_human"


def test_two_model_families_inside_one_figure_route_count_as_agreement():
    """F2: two Opus prompts agreeing is one voter agreeing with itself; opus + sonnet is not."""
    from canopy.models import SourceKind

    def ensemble(cid, families):
        return Candidate(candidate_id=cid, paper_id="p", dataset_id="ds1",
                         outcome_key="late_adaptation", kind="group_stats", group="A",
                         status="found", source_kind=SourceKind.figure_bar, n=12, mean=31.5,
                         dispersion_value=11.0, dispersion_type=DispersionType.SD, unit="deg",
                         route="figure", extractor_id="digitize:ensemble", model="",
                         pixel_provenance={"model_families": families,
                                           "cal": {"ticks": [[0.0, 45.0], [100.0, 5.0]]}})

    one = ensemble("f1", ["claude-opus"])
    two = ensemble("f2", ["claude-opus", "claude-sonnet"])
    lonely = confidence(vote([one]), CONFIRMED, [], None, candidates=[one], n_a=12, n_b=12,
                        orientation=ORIENTED)
    shared = confidence(vote([two]), CONFIRMED, [], None, candidates=[two], n_a=12, n_b=12,
                        orientation=ORIENTED)
    assert vote([two]).agreement == "single", "the vote still sees one route, as it should"
    assert shared[1] > lonely[1]
    assert any("independent model families" in r for r in shared[2])
    assert any("only one independent route" in r for r in lonely[2])


# ------------------------------------------------------------------ R2 on a FIGURE-only cell
def _figure_ensemble(cid: str, group: str, mean: float, cal_status: str = "confirmed",
                     families=("claude-opus", "claude-sonnet")) -> Candidate:
    """A digitised cell with no printed reader and no verifier — the shape R2 is really about."""
    from canopy.models import SourceKind

    return Candidate(
        candidate_id=cid, paper_id="p", dataset_id="ds1", outcome_key="late_adaptation",
        kind="group_stats", group=group, status="found", source_kind=SourceKind.figure_line,
        n=12, mean=mean, dispersion_value=11.0, dispersion_type=DispersionType.SD, unit="deg",
        route="figure", extractor_id="digitize:ensemble", model="", sigma=0.2,
        pixel_provenance={"model_families": list(families), "cal_status": cal_status,
                          "cal": {"ticks": [[0.0, 45.0], [100.0, -5.0]]}})


def _cal_flags(code: str, n: int) -> list[CheckFlag]:
    """The code as `run_checks` really emits it: once per candidate of the cell."""
    severity = "error" if code == "calibration_refuted" else "warn"
    return [CheckFlag(code=code, severity=severity, message=code, candidate_ids=[f"c{i}"])
            for i in range(n)]


def _score(flags: list[CheckFlag], cal_status: str = "confirmed"):
    a = _figure_ensemble("fa", "A", 31.3, cal_status)
    b = _figure_ensemble("fb", "B", 12.3, cal_status)
    # no verifier verdict: a CONFIRMED verifier is +0.20 and hid this the first time round
    return confidence(vote([a]), [], flags, None, candidates=[a, b], n_a=12, n_b=12,
                      orientation=ORIENTED)


def test_one_axis_problem_on_six_candidates_is_still_one_axis_problem():
    """The regression: penalties counted FLAGS, so one code on six candidates spent -0.24.

    A clean figure cell scores 0.60. Under the old arithmetic three `calibration_single_witness`
    flags took it to 0.36 — `needs_human` — which is exactly the outcome task 16 exists to remove:
    Cressman's late adaptation, with the true ladder recovered, would still have gone to a human.
    """
    clean_bucket, clean_score, _ = _score([])
    assert (clean_bucket, clean_score) == ("accept_with_note", 0.60)
    for n in (1, 3, 6):
        bucket, score, _ = _score(_cal_flags("calibration_single_witness", n),
                                  cal_status="single_witness")
        assert bucket == "accept_with_note", f"{n} flags of one code buried the cell"
        assert score == 0.52, f"{n} flags of one code scored {score}, not once"


def test_the_axis_ladder_is_ordered_confirmed_then_single_witness_then_refuted():
    """An axis known WRONG must never outscore an axis merely uncorroborated."""
    confirmed = _score([])[1]
    single = _score(_cal_flags("calibration_single_witness", 2), "single_witness")[1]
    refuted = _score(_cal_flags("calibration_refuted", 2), "cal_refuted")[1]
    assert confirmed > single > refuted, (confirmed, single, refuted)
    for bucket, _s, _r in (_score([]), ):
        assert bucket == "accept_with_note"
    assert all(_score(_cal_flags(code, 2), status)[0] == "accept_with_note"
               for code, status in (("calibration_single_witness", "single_witness"),
                                    ("calibration_refuted", "cal_refuted")))


def test_no_pile_of_capping_flags_sends_a_figure_cell_to_a_human():
    """R2, on the route it actually bites: caps and axis doubt cost points, never the cell."""
    from canopy.verify.confidence import ACCEPT_WITH_NOTE, CAPPING_FLAGS

    flags: list[CheckFlag] = []
    for code in sorted(CAPPING_FLAGS):
        flags += _cal_flags(code, 4)
    bucket, score, _ = _score(flags, cal_status="single_witness")
    assert bucket == "accept_with_note", "capping flags composed downwards into needs_human"
    assert score >= ACCEPT_WITH_NOTE


def test_a_genuine_error_still_forces_a_human_on_the_same_cell():
    """The floor is for CAPS. Evidence that the reading is wrong is untouched by it."""
    disputed = [CheckFlag(code="calibration_disputed", severity="error", message="x",
                          candidate_ids=["c0"])]
    assert _score(disputed, "single_witness")[0] == "needs_human"
