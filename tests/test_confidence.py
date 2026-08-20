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
                     source_kind=kwargs.pop("source_kind", SourceKind.text_mean_sd),
                     mean=mean, n=12,
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


def test_a_row_only_table_match_withholds_the_cell():
    """Numbers found in the named row but not the named column may be the other group's.

    EXPECTATION CHANGED (was `accept_with_note`): `quote_row_only` is a report that the value may
    be a DIFFERENT QUANTITY — the neighbouring column is the other group — not that it is
    under-corroborated. Two readers agreeing on the other group's number agree on the wrong
    number, so no score can settle it and it belongs to `CONTRADICTING_FLAGS`.
    """
    result = vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])
    flags = [CheckFlag(code="quote_row_only", severity="warn",
                       message="found in the row but not the column", candidate_ids=["a"])]
    bucket, score, reasons = confidence(result, CONFIRMED, flags, None, orientation=ORIENTED)
    assert bucket == "needs_human" and score <= 0.70
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
    assert any("imply effects that differ by" in r for r in reasons)


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
    """No pile-up of caps may add up to `needs_human`: only evidence does that.

    SCOPE NARROWED: `CAPPING_FLAGS` is now only the under-corroborated family. The codes that
    report "this may be a different quantity" moved to `CONTRADICTING_FLAGS`, and the fact that
    they DO withhold is asserted by
    `test_a_flag_that_says_this_may_be_a_different_quantity_withholds_the_cell`.
    """
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
    """R2, on the route it actually bites: caps and axis doubt cost points, never the cell.

    SCOPE NARROWED, with its mirror image next door: this asserts the floor for the
    under-corroborated family, and `test_a_pile_of_contradictions_is_not_floored_at_
    accept_with_note` asserts that the contradicting family is NOT floored.
    """
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


# ---------------------------------------- "less corroborated" vs "this may be a different quantity"
# Controller ruling (task 16 R2, overturned in part): the two doubts below used to share one set
# and one floor. They have opposite consequences and must not.
def test_the_two_doubt_families_are_disjoint_and_every_code_belongs_to_exactly_one():
    from canopy.verify.checks import CHECK_SEVERITY
    from canopy.verify.confidence import CAP_REASONS, CAPPING_FLAGS, CONTRADICTING_FLAGS

    assert CAPPING_FLAGS.isdisjoint(CONTRADICTING_FLAGS)
    assert CAPPING_FLAGS and CONTRADICTING_FLAGS
    for code in CAPPING_FLAGS | CONTRADICTING_FLAGS:
        assert (code in CAPPING_FLAGS) ^ (code in CONTRADICTING_FLAGS), code
        assert code in CAP_REASONS, f"{code} caps a cell without telling the reviewer why"
        assert code in CHECK_SEVERITY, f"{code} is not a code any check can raise"


def test_every_orientation_doubt_is_in_a_family_and_says_why(): 
    """Review L3. Three of the four `ORIENTATION_FLAGS` are in a doubt family with a reason line;
    the fourth, `orientation_direction_conflict`, was in neither — so unlike its siblings it was
    deducted in the generic warn bucket ABOVE the R2 floor, could withhold a cell on its own at
    the margin, and printed to the reviewer as a bare code with no explanation. It says the same
    kind of thing they do: how thin the agreement about the DIRECTION was, which is a reason to
    look, never on its own evidence that the number is wrong."""
    from canopy.verify.checks import ORIENTATION_FLAGS
    from canopy.verify.confidence import CAP_REASONS, CAPPING_FLAGS, CONTRADICTING_FLAGS

    for code in ORIENTATION_FLAGS:
        assert (code in CAPPING_FLAGS) ^ (code in CONTRADICTING_FLAGS), code
        assert code in CAP_REASONS and len(CAP_REASONS[code]) > 40, code
    assert "orientation_direction_conflict" in CAPPING_FLAGS


def test_a_flag_that_says_this_may_be_a_different_quantity_withholds_the_cell():
    """No amount of agreement about a number establishes that it is the right number.

    Each of these codes reports that the value may have been measured somewhere other than where
    it was asked for — off another axis, off the other series, out of another column. The
    meta-analysis downstream pools whatever it is given and cannot recover from a value of the
    wrong quantity, so these must be able to withhold, unlike the merely under-corroborated ones.
    """
    from canopy.verify.confidence import CONTRADICTING_FLAGS

    for code in sorted(CONTRADICTING_FLAGS):
        bucket, _points, reasons = _score(_cal_flags(code, 4))
        assert bucket == "needs_human", f"{code} did not withhold the cell"
        assert any(code in r for r in reasons), code


def test_a_reader_the_numbers_contradict_withholds_the_cell_instead_of_costing_it_points():
    """Fix round F5 (orientation area). The deterministic orientation check discards the ONE
    reader that made a checkable claim about this cell's numbers — the reader whose own words say
    the opposite of what the resolved means say. What is in doubt afterwards is the values, not
    just how well corroborated the direction is, so the code belongs with the contradictions: it
    is exempt from the `accept_with_note` floor and it forces a human.
    """
    from canopy.verify.checks import CHECK_SEVERITY
    from canopy.verify.confidence import CAPPING_FLAGS, CONTRADICTING_FLAGS

    code = "orientation_reader_contradicts_values"
    assert code in CONTRADICTING_FLAGS and code not in CAPPING_FLAGS
    assert CHECK_SEVERITY[code] == "error"
    flags = [CheckFlag(code=code, severity=CHECK_SEVERITY[code], message=code,
                       candidate_ids=[f"c{i}"]) for i in range(2)]
    bucket, _points, reasons = _score(flags)
    assert bucket == "needs_human"
    assert any(code in reason for reason in reasons)


def test_a_pile_of_contradictions_is_not_floored_at_accept_with_note():
    """The R2 floor is for caps. A contradiction is evidence, and evidence is never restored."""
    from canopy.verify.confidence import ACCEPT_WITH_NOTE, CONTRADICTING_FLAGS

    flags: list[CheckFlag] = []
    for code in sorted(CONTRADICTING_FLAGS):
        flags += _cal_flags(code, 4)
    bucket, points, _reasons = _score(flags)
    assert bucket == "needs_human"
    assert points < ACCEPT_WITH_NOTE, "the floor restored a score the evidence took away"


def test_the_wrong_panel_figure_read_is_withheld_by_its_own_axis_conflict():
    """Regression from a real record, `runs/rerun-hardened/papers/b511dbb76fa6/verify.json`.

    That cell (`b511dbb76fa6:d2 late_adaptation`, both groups) carries `axis_conflict`,
    `dispersion_type_from_legend` and `figure_error_bar_unknown`, two model families, and an
    ambiguous verifier; it was digitised off the wrong panel in the wrong unit. The run recorded
    `needs_human` at 0.39. Commit `e9401f2` then moved `axis_conflict` under the R2 floor, which
    restored the 0.08 it had cost and landed the cell on exactly 0.4500 — the acceptance
    threshold — so a replay of the same inputs pooled it. Rule, stated generally: a reading whose
    own readers answered off different value axes may not be released by an arithmetic floor.
    """
    ambiguous = [VerifierVerdict(candidate_id="fa", verdict="ambiguous",
                                 reason="the figure prints no number", model=SONNET)]
    flags = (_cal_flags("axis_conflict", 2) + _cal_flags("dispersion_type_from_legend", 2)
             + _cal_flags("figure_error_bar_unknown", 2))
    a, b = _figure_ensemble("fa", "A", 52.77), _figure_ensemble("fb", "B", 39.0)
    bucket, score, reasons = confidence(vote([a]), ambiguous, flags, None, candidates=[a, b],
                                        n_a=12, n_b=12, orientation=ORIENTED)
    assert (bucket, score) == ("needs_human", 0.39), (bucket, score, reasons)


# ------------------------------------------------ what makes two readings two independent witnesses
def test_one_model_reading_one_sentence_twice_is_not_two_agreeing_routes():
    """`route_key` is modality x model family, and modality is a label the model wrote itself.

    A model that reports the same sentence once as `text` and once as `table` produced two route
    keys, which scored as agreement: +0.25 for "2 independent routes agree", no single-route cap,
    and a maximum-score `auto_accept` with no human and no note — off one model reading one
    sentence. Route independence is a property of the EVIDENCE, not of a label the reader wrote.
    """
    a = text_cand("a", 31.51)
    b = text_cand("b", 31.51, source_kind=SourceKind.table)
    result = vote([a, b])
    assert len(result.routes) == 2, "the vote still sees two routes, as it should"
    bucket, score, reasons = confidence(result, CONFIRMED, [], None, candidates=[a, b],
                                        orientation=ORIENTED)
    assert bucket == "accept_with_note", (bucket, score, reasons)
    assert score <= 0.70
    assert not any("independent routes agree" in r for r in reasons)
    assert any("one witness" in r for r in reasons), reasons


def test_two_model_families_quoting_one_sentence_are_still_two_witnesses():
    """A second family is a different failure mode even on the same sentence — that is the point."""
    a = text_cand("a", 31.51)
    b = text_cand("b", 31.51, model=SONNET, source_kind=SourceKind.table)
    bucket, _score, reasons = confidence(vote([a, b]), CONFIRMED, [], None, candidates=[a, b],
                                         orientation=ORIENTED)
    assert bucket == "auto_accept"
    assert any("independent routes agree" in r for r in reasons)


def test_one_model_reading_two_different_places_in_the_paper_is_two_witnesses():
    """The paper printing a value twice is corroboration of the transcription, not of the model."""
    a = text_cand("a", 31.51)
    b = text_cand("b", 31.51, source_kind=SourceKind.table,
                  quote="Table 2 gives 31.51 for the older group", page=7)
    bucket, _score, reasons = confidence(vote([a, b]), CONFIRMED, [], None, candidates=[a, b],
                                         orientation=ORIENTED)
    assert bucket == "auto_accept"
    assert any("independent routes agree" in r for r in reasons)


# ---------------------------------------------- amendment F's gate, on the input it really receives
def _route_row(extractor_id: str, mean: float | None, error: float | None, *,
               sigma: float | None = 0.08, dropped: bool = False) -> dict:
    """One row of `pixel_provenance["per_route"]`, shaped as `RouteSample.to_dict()` writes it."""
    return {"extractor_id": extractor_id, "mean": mean, "error": error, "sigma": sigma,
            "dropped": dropped, "status": "found", "group": "A", "route": "D"}


def _ensemble_with_routes(cid: str, group: str, mean: float, rows: list[dict], *,
                          dispersion_type=DispersionType.SD, sd: float = 11.0,
                          n: int = 12) -> Candidate:
    return Candidate(
        candidate_id=cid, paper_id="p", dataset_id="ds1", outcome_key="late_adaptation",
        kind="group_stats", group=group, status="found", source_kind=SourceKind.figure_line,
        n=n, mean=mean, dispersion_value=sd, dispersion_type=dispersion_type, unit="deg",
        route="figure", extractor_id="digitize:ensemble", model="", sigma=0.08,
        pixel_provenance={"model_families": ["claude-opus", "claude-sonnet"],
                          "cal_status": "confirmed", "per_route": rows,
                          "cal": {"ticks": [[0.0, 60.0], [100.0, 0.0]]}})


def test_the_figure_gate_can_pass_on_the_candidates_the_pipeline_really_hands_it():
    """The gate has to be satisfiable by the input `resolve_cell` actually receives.

    `run.py:vote_candidates` admits exactly one `digitize:ensemble` candidate per group, so the
    gate — which compared candidates of two different digitizer modalities — saw one route in
    every figure cell of every completed run, printed "1 digitizer route(s) read both groups" 21
    times, and capped every one of them. A check that the pipeline's own upstream filter makes
    unsatisfiable is worse than no check: it spends a reviewer's attention on a reason that reads
    like a finding. The routes were there all along, inside `pixel_provenance["per_route"]`.
    """
    a = _ensemble_with_routes("ea", "A", 31.5, [
        _route_row("digitize:readout:claude-opus-5:direct", 31.5, 11.0),
        _route_row("digitize:vlm_coords:claude-opus-5", 31.6, 11.1)])
    b = _ensemble_with_routes("eb", "B", 12.3, [
        _route_row("digitize:readout:claude-opus-5:direct", 12.3, 11.0),
        _route_row("digitize:vlm_coords:claude-opus-5", 12.4, 11.1)], sd=11.0)
    ok, delta, share, reasons = figure_gate([a, b], 12, 12)
    assert ok is True, reasons
    assert delta is not None and delta < DELTA_D_LIMIT
    assert share is not None and share < DIGITIZATION_SE_SHARE


def test_a_figure_cell_whose_paths_pass_the_gate_is_not_capped_by_it():
    """End to end: the gate's cap is now conditional on evidence, not on the pipeline's shape."""
    a = _ensemble_with_routes("ea", "A", 31.5, [
        _route_row("digitize:readout:claude-opus-5:direct", 31.5, 11.0),
        _route_row("digitize:vlm_coords:claude-opus-5", 31.6, 11.1)])
    b = _ensemble_with_routes("eb", "B", 12.3, [
        _route_row("digitize:readout:claude-opus-5:direct", 12.3, 11.0),
        _route_row("digitize:vlm_coords:claude-opus-5", 12.4, 11.1)])
    bucket, score, reasons = confidence(vote([a]), CONFIRMED, [], None, candidates=[a, b],
                                        n_a=12, n_b=12, orientation=ORIENTED)
    assert not any("cannot be accepted automatically" in r for r in reasons), reasons
    assert (bucket, score) == ("auto_accept", 0.80), (bucket, score, reasons)


def test_the_gate_fails_when_the_digitisers_paths_imply_different_effects():
    """Numbers from a real record: `runs/rerun-hardened/…/b511dbb76fa6:d2 late_adaptation`.

    That is the cell digitised off the wrong panel in the wrong unit. Its two surviving paths
    imply d = 0.783 and d = 0.608 — 0.174 apart, well over the limit — which no reader of the
    run could see, because the gate never compared them.
    """
    a = _ensemble_with_routes("ea", "A", 52.8, [
        _route_row("digitize:readout:claude-opus-5:direct", 52.8, 11.0),
        _route_row("digitize:readout:claude-sonnet-5:direct", 52.0, 9.0, dropped=True),
        _route_row("digitize:vlm_coords:claude-opus-5", 52.770661652842584, 9.96508568385304),
        _route_row("digitize:raster_cv", 52.81071631697411, None)],
        dispersion_type=DispersionType.SE, sd=10.48254284192652)
    b = _ensemble_with_routes("eb", "B", 28.4, [
        _route_row("digitize:readout:claude-opus-5:direct", 28.4, 6.4),
        _route_row("digitize:vlm_coords:claude-opus-5", 28.380926636087487, 12.98338661960863),
        _route_row("digitize:raster_cv", 29.00482401688621, 0.7982431512276698)],
        dispersion_type=DispersionType.SE, sd=6.4)
    ok, delta, _share, reasons = figure_gate([a, b], 12, 12)
    assert ok is False
    assert delta is not None and round(delta, 3) == 0.174, delta
    assert any("differ by" in r for r in reasons), reasons


def test_a_dropped_route_does_not_vote_in_the_gate():
    """A reading the ensemble threw away is not a witness to anything, here least of all."""
    kept = _route_row("digitize:readout:claude-opus-5:direct", 31.5, 11.0)
    thrown = _route_row("digitize:vlm_coords:claude-opus-5", 99.0, 11.0, dropped=True)
    a = _ensemble_with_routes("ea", "A", 31.5, [kept, thrown])
    b = _ensemble_with_routes("eb", "B", 12.3, [
        _route_row("digitize:readout:claude-opus-5:direct", 12.3, 11.0),
        _route_row("digitize:vlm_coords:claude-opus-5", 80.0, 11.0, dropped=True)])
    ok, delta, _share, reasons = figure_gate([a, b], 12, 12)
    assert ok is False and delta is None
    assert any("one" in r for r in reasons)


def test_the_gate_says_plainly_when_it_had_nothing_to_evaluate():
    """An absence of evidence must not be printed as a finding about the reading."""
    a = _ensemble_with_routes("ea", "A", 31.5, [])
    b = _ensemble_with_routes("eb", "B", 12.3, [])
    ok, delta, _share, reasons = figure_gate([a, b], 12, 12)
    assert ok is False and delta is None
    assert any("could not be evaluated" in r for r in reasons), reasons
    assert not any("route(s) read both groups" in r for r in reasons), reasons


# =========================================================== ceiling C8 + C11 (one change) + C9
# C8: an agent that could not be run is an ABSENCE, not a "don't know".
# C11: publish the margin, with the band re-derived AFTER C8.
# C9:  price the conversion gate's own flags before any statistic route.
from canopy.verify.confidence import (BUCKET_BOUNDARIES, CONVERTED_ROUTES, DECIDED_BY_A_HAIR,
                                      MARGIN_BAND, NOT_RUN, NO_VALUE_PRINTED,
                                      confidence_margin, conversion_gate_bucket, verifier_state)

#: exactly the shape `pipeline/run.py` writes when `verify_candidate` raised `TruncatedOutput` or
#: `LLMError` — no model, no prompt version, no call id, because there was no call
TRUNCATED = VerifierVerdict(
    candidate_id="a", verdict="ambiguous",
    reason="the verifier's answer was cut off at its output limit twice and could not be read "
           "(response hit max_tokens=8000); this candidate is unverified")


def _two_readers():
    return vote([text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)])


def _absence(candidate_id="a"):
    """A verifier that RAN and reported that the paper prints no independent value here."""
    return VerifierVerdict.model_construct(
        **{**VerifierVerdict(candidate_id=candidate_id, verdict="ambiguous", model=SONNET,
                             prompt_version="verifier/1@f78d94c0", llm_call_id="deadbeef",
                             reason="the paper never prints a numeric value for this cell").__dict__,
           "verdict": NO_VALUE_PRINTED})


# --------------------------------------------------------------------------- C8
def test_a_verifier_that_could_not_be_run_is_not_a_doubt():
    """The whole of C8: a failed call must cost exactly what a call nobody made costs."""
    never_scheduled = confidence(_two_readers(), [], [], None, orientation=ORIENTED)
    failed = confidence(_two_readers(), [TRUNCATED], [], None, orientation=ORIENTED)
    assert failed[1] == never_scheduled[1]
    assert failed[0] == never_scheduled[0]
    assert any("could not be run" in r and "unchanged" in r for r in failed[2])
    assert not any("could not settle it either way" in r for r in failed[2])


def test_the_failure_is_named_on_the_record_not_swallowed():
    _, _, reasons = confidence(_two_readers(), [TRUNCATED], [], None, orientation=ORIENTED)
    assert any("cut off at its output limit" in r for r in reasons)


def test_a_verifier_that_looked_and_could_not_tell_still_costs_its_penalty():
    """C8 removes a mis-classification, not the ambiguous penalty itself."""
    looked = VerifierVerdict(candidate_id="a", verdict="ambiguous", model=SONNET,
                             prompt_version="verifier/1@f78d94c0", llm_call_id="deadbeef",
                             reason="the table row could be either group")
    plain = confidence(_two_readers(), [], [], None, orientation=ORIENTED)[1]
    assert confidence(_two_readers(), [looked], [], None, orientation=ORIENTED)[1] == \
        round(plain - 0.05, 4)


def test_the_paper_printing_no_value_is_a_completed_check_not_an_ambiguity():
    """A successful cross-check whose answer is "there is nothing here to check against"."""
    plain = confidence(_two_readers(), [], [], None, orientation=ORIENTED)
    absent = confidence(_two_readers(), [_absence()], [], None, orientation=ORIENTED)
    assert absent[1] == plain[1] and absent[0] == plain[0]
    assert any("prints no independent value" in r and "unchanged" in r for r in absent[2])


def test_verifier_state_reads_the_record_not_the_label():
    assert verifier_state(TRUNCATED) == NOT_RUN
    assert verifier_state(VerifierVerdict(candidate_id="a", verdict="ambiguous", model=SONNET,
                                          llm_call_id="x")) == "ambiguous"
    assert verifier_state(_absence()) == NO_VALUE_PRINTED


def test_a_failed_call_never_outranks_a_verdict_an_agent_produced():
    from canopy.verify.confidence import _verifier_summary
    confirmed = VerifierVerdict(candidate_id="a", verdict="confirmed", model=SONNET,
                                llm_call_id="x", reason="page 3 prints it")
    assert _verifier_summary([TRUNCATED, confirmed], {"a"})[0] == "confirmed"
    assert _verifier_summary([TRUNCATED], {"a"})[0] == NOT_RUN
    assert _verifier_summary([], {"a"})[0] == NOT_RUN


# --------------------------------------------------------------------------- C11
def test_the_margin_is_published_on_every_cell():
    _, score, reasons = confidence(_two_readers(), CONFIRMED, [], None, orientation=ORIENTED)
    line = next(r for r in reasons if "margin" in r)
    distance, boundary, _ = confidence_margin(score)
    assert f"{distance:.4f}" in line and boundary in line


@pytest.mark.parametrize("score, distance, boundary, labelled", [
    (0.4500, 0.0000, "accept_with_note", True),      # pooled by nothing at all
    (0.7400, 0.0100, "auto_accept", True),           # held back by a hundredth
    (0.4800, 0.0300, "accept_with_note", True),      # exactly on the band
    (0.6000, 0.1500, "accept_with_note", False),     # decided by neither
])
def test_a_cell_within_a_hair_of_any_boundary_it_cleared_is_labelled(score, distance, boundary,
                                                                     labelled):
    assert confidence_margin(score) == (distance, boundary, labelled)


def test_the_band_catches_the_boundary_a_cell_missed_as_well_as_the_one_it_cleared():
    """A cell at 0.74 is as decided by a hair as one at 0.45 — against AUTO_ACCEPT, not for it."""
    assert confidence_margin(0.74)[1] == "auto_accept"
    assert confidence_margin(0.76)[1] == "auto_accept"
    assert BUCKET_BOUNDARIES == {"accept_with_note": ACCEPT_WITH_NOTE, "auto_accept": AUTO_ACCEPT}


def test_the_caps_are_not_boundaries():
    """Every capped cell lands exactly ON its ceiling, so a margin against it would say nothing."""
    from canopy.verify.confidence import ADJUDICATED_CAP, SINGLE_ROUTE_CAP
    assert ADJUDICATED_CAP not in BUCKET_BOUNDARIES.values()
    assert SINGLE_ROUTE_CAP not in BUCKET_BOUNDARIES.values()
    assert confidence_margin(0.70)[2] is False


# ------------------------------------------------- C8 + C11 together: the post-C8 regression
#: The six cells this run pooled, at the scores they reach ONCE C8 stops charging the verifier
#: penalty to an absence — replayed offline from `runs/rerun-fixed/*/verify.json` (Cressman d1
#: late/aftereffect and Bock d1 late, both groups each). v1 of this item claimed the band would
#: make all six "visibly marginal"; after C8 not one of them is inside it, and this asserts the
#: number so the claim can never go stale again.
POST_C8_POOLED_SCORES = (0.50, 0.50, 0.51, 0.51, 0.52, 0.52)


@pytest.mark.parametrize("score", POST_C8_POOLED_SCORES)
def test_after_c8_no_pooled_cell_is_decided_by_a_hair(score):
    distance, boundary, labelled = confidence_margin(score)
    assert boundary == "accept_with_note"
    assert 0.05 - 1e-9 <= distance <= 0.07 + 1e-9
    assert labelled is False


def test_before_c8_two_of_those_cells_sat_exactly_on_the_line():
    """The history C11 documents: at 0.4500 the bucket was decided by nothing at all."""
    for score in (0.45, 0.45, 0.46, 0.46, 0.47, 0.47):
        assert confidence_margin(score)[2] is (score <= ACCEPT_WITH_NOTE + MARGIN_BAND)
    assert confidence_margin(0.45) == (0.0, "accept_with_note", True)


def test_the_hair_label_is_its_own_state_not_a_sentence_a_reader_must_parse():
    """Review L1: the first cut of this fed a cell that scores 0.70 — five hundredths outside the
    band — so it asserted `<= 1` against zero lines and passed with C11 deleted. Both inputs here
    are INSIDE the band, one on each side of the boundary, and the assertion is `== 1`."""
    warn = [CheckFlag(code="n_mismatch", severity="warn", message="x", candidate_ids=["a"])]
    missed, score, reasons = confidence(_two_readers(), [], warn, None, orientation=ORIENTED)
    assert score == 0.72 and missed == "accept_with_note"      # 0.03 short of auto_accept
    hairs = [r for r in reasons if r.startswith(DECIDED_BY_A_HAIR)]
    assert len(hairs) == 1 and "misses auto_accept" in hairs[0]

    three = [CheckFlag(code=code, severity="warn", message="x", candidate_ids=["a"])
             for code in ("n_mismatch", "unit_mismatch", "sd_near_zero")]
    cleared, score, reasons = confidence(_two_readers(), CONFIRMED, three, None,
                                         orientation=ORIENTED)
    assert score == 0.76 and cleared == "auto_accept"          # 0.01 the other side of the line
    hairs = [r for r in reasons if r.startswith(DECIDED_BY_A_HAIR)]
    assert len(hairs) == 1 and "clears auto_accept" in hairs[0]


# --------------------------------------------------------------------------- C9
def _stat(**kwargs):
    base = dict(candidate_id="cT", dataset_id="ds1", outcome_key="late_adaptation",
                kind="test_statistic", stat_type="t", stat_value=5.25, design="independent_t",
                grounded=True, model=OPUS, quote="the difference was significant, t = 5.25, "
                                                 "p < .001")
    base.update(kwargs)
    return Candidate(**base)


def test_a_three_group_post_hoc_with_no_df_does_not_pool():
    """The failing input C9 was written against: t = 5.25, p < .001, n = 12/12, no df printed.

    `convertibility` returns ok with `df_missing`, so before C9 the row converted and pooled on a
    statistic nothing established belongs to these two groups.
    """
    from canopy.verify.checks import codes, run_checks

    flags = run_checks(dataset(), "late_adaptation", [text_cand("a", 31.51), _stat()])
    assert "df_missing" in codes(flags)
    bucket, _, reasons = confidence(_two_readers(), CONFIRMED, flags, None, orientation=ORIENTED)
    assert bucket == "needs_human"
    assert any("df_missing" in r for r in reasons)


def test_an_f_with_only_a_numerator_df_reached_the_row_unflagged_before_c9():
    """`test_stat_missing_df` fires only when df, df1 AND df2 are all None, so `F(1, ?)` passed
    every check there was — this is the case C9's cell-level half exists for."""
    from canopy.verify.checks import codes, run_checks

    flags = run_checks(dataset(), "late_adaptation",
                       [text_cand("a", 31.51),
                        _stat(stat_type="F", stat_value=27.6, df1=1.0, design="one_way_between")])
    assert "test_stat_missing_df" not in codes(flags)
    assert confidence(_two_readers(), CONFIRMED, flags, None, orientation=ORIENTED)[0] == \
        "needs_human"


def test_df_missing_caps_the_bucket_and_not_the_score():
    """"Below auto_accept" still includes accept_with_note, which POOLS. Assert the bucket."""
    for start in ("auto_accept", "accept_with_note"):
        bucket, reasons = conversion_gate_bucket(start, "test_statistic", ["df_missing"])
        assert bucket == "needs_human", start
        assert reasons and "THESE two groups" in reasons[0]


def test_the_conversion_gate_only_fires_on_a_route_that_converted_a_statistic():
    assert conversion_gate_bucket("auto_accept", "means_sd", ["df_missing"])[0] == "auto_accept"
    assert "test_statistic" in CONVERTED_ROUTES and "p_value" in CONVERTED_ROUTES


def test_an_explained_df_shortfall_is_allowed_but_never_automatic():
    bucket, reasons = conversion_gate_bucket("auto_accept", "t_stat", ["df_off_by_1"])
    assert bucket == "accept_with_note" and "df_off_by_1" in reasons[0]
    assert conversion_gate_bucket("accept_with_note", "t_stat", ["df_off_by_1"])[0] == \
        "accept_with_note"


# ------------------------------------ C9's |d| screen, on the number that reaches the plot
def _row(candidates, *, adjudication=None, flags=None):
    """The candidates carried the whole way a real cell travels: checks, vote, adjudication,
    `resolve_cell`, `ResolvedValues`, `resolve_effect`. Returns `(row, verdict_a, verdict_b)`.

    Written as a route rather than as a call to one function on purpose (review H1): the C9 screen
    was specified on the RESOLVED |d| and implemented on a pair of raw SD candidates, and no test
    that stops at the cell can tell the two apart.
    """
    from canopy.models import OutcomeDef, StatsSettings
    from canopy.pipeline.resolve import ResolvedValues, resolve_effect
    from canopy.verify.checks import run_checks
    from canopy.verify.vote import vote_groups

    flags = run_checks(dataset(), "late_adaptation", candidates) if flags is None else flags
    votes = vote_groups(candidates)
    cells = {group: resolve_cell(dataset(), "late_adaptation", group, candidates,
                                 vote_result=votes.get(group), verdicts=CONFIRMED, flags=flags,
                                 adjudication=adjudication, orientation=ORIENTED, n_a=12, n_b=12)
             for group in ("A", "B")}
    values = ResolvedValues.from_verdicts(cells["A"], cells["B"], higher_is_better=False,
                                          candidates=candidates)
    outcome = OutcomeDef(key="late_adaptation", label="late adaptation",
                         definition="directional error at the end of the block")
    return resolve_effect(dataset(), outcome, values, StatsSettings()), cells["A"], cells["B"]


def test_an_se_pair_whose_conversion_implies_an_impossible_d_is_refused():
    """C9 acceptance, route 1 — the modal shape in the live corpus (32 of the 57 `found`
    group_stats candidates carrying a spread type carry SE, against 25 SD). A = 10 ± 0.5 SE and
    B = 40 ± 0.5 SE at n = 12 convert to SDs of 1.732, which implies |d| = 17.3. Screening
    SD-typed candidates only never saw it, and the row pooled at `accept_with_note` with both
    cells at `auto_accept`."""
    se = [text_cand("a1", 10.0, dispersion_value=0.5, dispersion_type=DispersionType.SE),
          text_cand("a2", 10.0, dispersion_value=0.5, dispersion_type=DispersionType.SE,
                    model=SONNET),
          text_cand("b1", 40.0, dispersion_value=0.5, dispersion_type=DispersionType.SE,
                    group="B"),
          text_cand("b2", 40.0, dispersion_value=0.5, dispersion_type=DispersionType.SE,
                    group="B", model=SONNET)]
    row, cell_a, _ = _row(se)
    assert abs(row.d) > 3.0
    assert row.confidence == "needs_human", (row.d, cell_a.confidence)
    assert "implausible_dispersion" in row.flags
    # the screen's OWN line names the two dispersions `_mean_sd` used and where they came from
    said = next(step for step in row.conversion_steps if "implausible_dispersion" in step)
    assert "1.732" in said and "SE 0.5" in said and "17.3" in said


def test_the_screen_reads_the_voted_value_not_the_first_candidate_in_the_list():
    """C9 acceptance, route 2 — group A's first SD-typed candidate is sane (38 ± 5) while the two
    agreeing readers say 10 ± 0.5. Screening `rows[0]` compared 38 against 40 (|d| = 0.56) and
    passed the cell; the vote pooled 10 against 40, i.e. |d| = 60."""
    mixed = [text_cand("a0", 38.0, dispersion_value=5.0),
             text_cand("a1", 10.0, dispersion_value=0.5, model=SONNET),
             text_cand("a2", 10.0, dispersion_value=0.5, model="claude-haiku-5"),
             text_cand("b0", 40.0, dispersion_value=0.5, group="B"),
             text_cand("b1", 40.0, dispersion_value=0.5, group="B", model=SONNET)]
    row, cell_a, _ = _row(mixed)
    assert cell_a.mean == 10.0 and cell_a.dispersion_value == 0.5     # what the vote resolved
    assert round(row.d, 1) == 60.0
    assert row.confidence == "needs_human"


def test_a_dispersion_the_adjudicator_supplied_is_screened_like_any_other():
    """C9 acceptance, route 3 — the candidates are plausible (10 ± 5 against 12 ± 5, |d| = 0.4)
    and the adjudicator rules that the 5 was the range and the SD is 0.4. The screen ran inside
    `run_checks`, i.e. before the ruling existed, so the ruled value was never looked at."""
    ok = [text_cand("a1", 10.0, dispersion_value=5.0),
          text_cand("a2", 10.4, dispersion_value=5.0, model=SONNET),
          text_cand("b1", 12.0, dispersion_value=5.0, group="B"),
          text_cand("b2", 12.4, dispersion_value=5.0, group="B", model=SONNET)]
    quote = ("the aligned group reached 10.0 degrees of directional error at the end of the "
             "block, and the misaligned group reached the value printed beside it")
    ruling = Adjudication(
        dataset_id="ds1", outcome_key="late_adaptation", needs_human=False,
        rationale="the 5 is the range; the SD is 0.4",
        groups=[AdjudicatedGroup(group="A", mean=10.0, dispersion_value=0.4,
                                 dispersion_type=DispersionType.SD, n=12,
                                 chosen_candidate_ids=["a1"], quote=quote, grounded=True,
                                 reason="table 2"),
                AdjudicatedGroup(group="B", mean=12.0, dispersion_value=0.4,
                                 dispersion_type=DispersionType.SD, n=12,
                                 chosen_candidate_ids=["b1"], quote=quote, grounded=True,
                                 reason="table 2")])
    row, _, _ = _row(ok, adjudication=ruling)
    assert round(row.d, 1) == 5.0
    assert row.confidence == "needs_human"


def test_a_plausible_row_keeps_the_bucket_its_cells_earned():
    """The control: the same route with a believable denominator is not touched by the screen."""
    fine = [text_cand("a1", 10.0, dispersion_value=5.0),
            text_cand("a2", 10.0, dispersion_value=5.0, model=SONNET),
            text_cand("b1", 12.0, dispersion_value=5.0, group="B"),
            text_cand("b2", 12.0, dispersion_value=5.0, group="B", model=SONNET)]
    row, _, _ = _row(fine)
    assert abs(row.d) < 3.0
    assert row.confidence != "needs_human"
    assert "implausible_dispersion" not in row.flags


def test_the_cell_level_screen_is_an_early_warning_and_no_longer_decides_alone():
    """Means 10/40 with SDs of 5 imply |d| = 6: six readers can agree on a wrong denominator.

    The cell-level check keeps firing — it is the cheapest place to SAY it, and it names the
    denominator for the reviewer — but it is a `warn` now, because the binding screen is on the
    resolved value and an early warning that also withholds bought an adjudicator call for a
    denominator dispute the adjudicator's own answer would then not be screened for (M5)."""
    from canopy.verify.checks import CHECK_SEVERITY, codes, run_checks

    huge = [text_cand("a", 10.0, dispersion_value=5.0),
            text_cand("b", 10.0, dispersion_value=5.0, model=SONNET),
            text_cand("c", 40.0, dispersion_value=5.0, group="B"),
            text_cand("d", 40.0, dispersion_value=5.0, group="B", model=SONNET)]
    flags = run_checks(dataset(), "late_adaptation", huge)
    assert "implausible_dispersion" in codes(flags)
    assert CHECK_SEVERITY["implausible_dispersion"] == "warn"
    row, _, _ = _row(huge, flags=flags)
    assert row.confidence == "needs_human"           # …and the row is still refused


# ------------------------------- a denominator NOTHING on the record establishes (calibration D5)
# `dispersion_unknown` says the ± could be an SD or an SE; `n_missing` says the size the conversion
# scaled it by was not transcribed beside the number. Each alone is priced by the score. Together
# they are the same doubt twice, and it is the doubt that sets the magnitude: the only number that
# could tell an SD from an SE is the n, and the n was guessed too.
def _untyped_spread_and_no_n(cid, mean, *, group="A", model="claude-haiku-5"):
    """A third reader of one cell: a spread nobody could type, with no group size beside it."""
    return text_cand(cid, mean, dispersion_value=5.0, dispersion_type=DispersionType.UNKNOWN,
                     group=group, model=model).model_copy(update={"n": None})


def _doubly_unverified():
    """The defect's shape: a believable |d| = 0.4 the pipeline pooled at `accept_with_note`, on a
    denominator whose TYPE was never established and whose group size was never transcribed."""
    return [text_cand("a1", 10.0, dispersion_value=5.0),
            text_cand("a2", 10.0, dispersion_value=5.0, model=SONNET),
            _untyped_spread_and_no_n("a3", 10.0),
            text_cand("b1", 12.0, dispersion_value=5.0, group="B"),
            text_cand("b2", 12.0, dispersion_value=5.0, group="B", model=SONNET)]


def test_a_row_whose_spread_type_and_group_size_are_both_unverified_is_held():
    """The whole rule. |d| = 0.4 passes the plausibility screen, both cells score well enough to
    pool, every arithmetic check passes — and the magnitude still rests on nothing: read the ± as
    an SE instead of an SD and the same row is |d| = 1.7."""
    row, cell_a, cell_b = _row(_doubly_unverified())
    assert row.route == "text_mean_sd" and abs(row.d) < 3.0        # nothing else refuses it
    assert {"dispersion_unknown", "n_missing"} <= set(row.flags)
    assert cell_a.confidence != "needs_human" and cell_b.confidence != "needs_human"
    assert row.confidence == "needs_human", (row.d, row.flags)


def test_the_held_row_names_both_codes_so_the_question_card_can_say_why():
    """A row held with no reason a reviewer can read is a question nobody can answer."""
    row, _, _ = _row(_doubly_unverified())
    said = next(step for step in row.conversion_steps if "dispersion_unknown" in step)
    assert "n_missing" in said and "SE" in said and "√n" in said


def test_either_doubt_on_its_own_still_pools():
    """The rule is the CONJUNCTION, and the corpus is why: an untyped spread beside a printed n,
    and a missing n beside a typed SD, are each an ordinary warning the score already prices.
    Holding on either alone would bury readings that match the reference analysis."""
    typed_but_no_n = [text_cand("a1", 10.0, dispersion_value=5.0),
                      text_cand("a2", 10.0, dispersion_value=5.0, model=SONNET),
                      text_cand("a3", 10.0, dispersion_value=5.0,
                                model="claude-haiku-5").model_copy(update={"n": None}),
                      text_cand("b1", 12.0, dispersion_value=5.0, group="B"),
                      text_cand("b2", 12.0, dispersion_value=5.0, group="B", model=SONNET)]
    row, _, _ = _row(typed_but_no_n)
    assert "n_missing" in row.flags and "dispersion_unknown" not in row.flags
    assert row.confidence != "needs_human"

    untyped_with_n = [text_cand("a1", 10.0, dispersion_value=5.0),
                      text_cand("a2", 10.0, dispersion_value=5.0, model=SONNET),
                      text_cand("a3", 10.0, dispersion_value=5.0,
                                dispersion_type=DispersionType.UNKNOWN, model="claude-haiku-5"),
                      text_cand("b1", 12.0, dispersion_value=5.0, group="B"),
                      text_cand("b2", 12.0, dispersion_value=5.0, group="B", model=SONNET)]
    row, _, _ = _row(untyped_with_n)
    assert "dispersion_unknown" in row.flags and "n_missing" not in row.flags
    assert row.confidence != "needs_human"


def test_the_unverified_variance_gate_caps_the_bucket_and_not_the_score():
    """Same reason C9's gate does: "below auto_accept" still includes `accept_with_note`, which
    POOLS, so a score cap could never withhold the row it exists to withhold."""
    from canopy.verify.confidence import UNVERIFIED_VARIANCE_FLAGS, unverified_variance_bucket

    both = {"A": ["dispersion_unknown", "n_missing", "panel_not_isolated"], "B": []}
    for start in ("auto_accept", "accept_with_note"):
        bucket, reasons = unverified_variance_bucket(start, both)
        assert bucket == "needs_human", start
        assert reasons and all(code in reasons[0] for code in UNVERIFIED_VARIANCE_FLAGS)
    assert unverified_variance_bucket("auto_accept", {"A": ["n_missing"]})[0] == "auto_accept"
    assert unverified_variance_bucket(
        "auto_accept", {"A": ["dispersion_unknown"]})[0] == "auto_accept"
    assert unverified_variance_bucket("auto_accept", {})[0] == "auto_accept"


# ------------------------------------------------------- fix round: the conjunction is PER ARM
def _split_across_arms():
    """The doubt on A is not the doubt on B. Arm A's ± has no type but its n is printed, so the
    spread is checkable; arm B's spread is typed and only its n is missing. Neither arm is "the
    same doubt arriving twice", so neither justifies withholding the row."""
    return [text_cand("a1", 10.0, dispersion_value=5.0),
            text_cand("a2", 10.0, dispersion_value=5.0, model=SONNET),
            text_cand("a3", 10.0, dispersion_value=5.0,          # untyped, but n IS printed
                      dispersion_type=DispersionType.UNKNOWN, model="claude-haiku-5"),
            text_cand("b1", 12.0, dispersion_value=5.0, group="B"),
            text_cand("b2", 12.0, dispersion_value=5.0, group="B", model=SONNET),
            text_cand("b3", 12.0, dispersion_value=5.0, group="B",  # typed, but no n
                      model="claude-haiku-5").model_copy(update={"n": None})]


def test_the_conjunction_is_read_on_one_arm_and_never_across_the_pair():
    """`ResolvedValues.from_verdicts` unions both cells' codes onto the row, so a row-level
    conjunction fires whenever the two doubts sit on DIFFERENT arms. That row is not doubly
    unverified — each arm carries exactly one ordinary warning — and the docstring's own reason
    ("the only number that could tell an SD from an SE is the n, and the n was guessed too") is
    a statement about ONE number."""
    row, _, _ = _row(_split_across_arms())
    assert {"dispersion_unknown", "n_missing"} <= set(row.flags)   # the union still shows both
    assert row.confidence != "needs_human", (row.d, row.flags)

    same_arm, _, _ = _row(_doubly_unverified())                    # the control, unchanged
    assert same_arm.confidence == "needs_human"


def test_the_row_records_which_arm_carried_which_doubt():
    """The union is what made the cross-arm read possible, so the row keeps the codes per ARM as
    well. Unioning ARM SETS can never manufacture a conjunction; unioning code sets can."""
    row, _, _ = _row(_doubly_unverified())
    assert {"dispersion_unknown", "n_missing"} <= set(row.arm_flags["A"])
    assert not {"dispersion_unknown", "n_missing"} <= set(row.arm_flags.get("B") or [])

    split, _, _ = _row(_split_across_arms())
    assert "dispersion_unknown" in split.arm_flags["A"] and "n_missing" not in split.arm_flags["A"]
    assert "n_missing" in split.arm_flags["B"] and "dispersion_unknown" not in split.arm_flags["B"]


def test_the_held_arm_is_named_and_the_group_size_is_the_answer_asked_for():
    """Fix round MINOR 5: `dispersion_unknown` is in no `overrides.VALUE_CLEARS_*` family, so
    "either answer breaks the pair" was false for half the pair. Only the n is answerable, and the
    sentence a reviewer reads has to say so — and say which arm."""
    from canopy.pipeline.overrides import (VALUE_CLEARS_DISPERSION, VALUE_CLEARS_MEAN,
                                           VALUE_CLEARS_N, VALUE_CLEARS_SPREAD_TYPE)

    families = (VALUE_CLEARS_MEAN | VALUE_CLEARS_SPREAD_TYPE | VALUE_CLEARS_N
                | VALUE_CLEARS_DISPERSION)
    assert "dispersion_unknown" not in families      # nothing a reviewer types retires it
    assert "n_missing" in VALUE_CLEARS_N             # …and the n is what does

    row, _, _ = _row(_doubly_unverified())
    said = next(step for step in row.conversion_steps if "dispersion_unknown" in step)
    assert "group A" in said, said
    assert "group size" in said and "either" not in said.lower()


def test_the_gate_names_codes_the_checks_actually_raise():
    """A gate keyed on a code no check emits is a gate that never fires. `severity_of` raises on a
    code it has never heard of, which is exactly the typo this pins."""
    from canopy.verify.checks import severity_of
    from canopy.verify.confidence import UNVERIFIED_VARIANCE_FLAGS

    assert {severity_of(code) for code in UNVERIFIED_VARIANCE_FLAGS} == {"warn"}


def test_the_card_for_the_held_arm_names_both_doubts():
    """Fix round MINOR 7. The gate holds a ROW, and the question a reviewer answers is on a CELL —
    so the hold has to reach the card through the channel every other finding uses, which is
    `state.review_entry`'s reason (what `questions._why` reads). Because the conjunction is now
    read on ONE arm, the arm that triggered the hold is exactly the arm whose card explains it:
    no new channel, and no row-level code that would veto the best-guess line."""
    from canopy.models import Verdict
    from canopy.pipeline.state import review_entry

    held_arm = Verdict(
        dataset_id="ds1", outcome_key="late_adaptation", group="A", confidence="accept_with_note",
        flags=[CheckFlag(code="dispersion_unknown", severity="warn",
                         message="a spread of 5.0 was read but nobody could say what kind it is"),
               CheckFlag(code="n_missing", severity="warn",
                         message="no group size was transcribed with this value")])
    reason = review_entry(held_arm, paper_id="p1")["reason"]
    assert "dispersion_unknown" in reason and "n_missing" in reason


def test_a_doubly_unverified_row_is_still_available_to_the_best_guess_line():
    """It is held, not refused. The value may well be the best estimate anyone has of this
    contrast — what is missing is corroboration of its SCALE — so the strict line may not take it
    and the best-guess line may, under the rule it already has for a row held by its bucket."""
    from canopy.models import OutcomeDef, StatsSettings
    from canopy.pipeline.bestguess import CONTRADICTED, VETO_ROW_FLAGS, best_guess_rows
    from canopy.verify.confidence import UNVERIFIED_VARIANCE_FLAGS

    assert not (UNVERIFIED_VARIANCE_FLAGS & (VETO_ROW_FLAGS | CONTRADICTED))
    row, _, _ = _row(_doubly_unverified())
    outcome = OutcomeDef(key="late_adaptation", label="late adaptation",
                         definition="directional error at the end of the block")
    taken, decisions = best_guess_rows([], [row], outcome=outcome, settings=StatsSettings())
    assert [r.dataset_id for r in taken] == [row.dataset_id]
    assert decisions[0].rule == "low_confidence_value"


# ---------------------------------------------------------- C11 (M3): whose decision was it
def _a_cell_forced_by_an_error():
    """A cell whose bucket an `error` flag decided, with a score well above the line."""
    return [text_cand("a", 31.51), text_cand("b", 31.51, model=SONNET)]


def test_a_cell_the_score_did_not_decide_publishes_no_margin():
    """Review M3. C11's rule is "within 0.03 of any boundary the CELL CLEARED". A cell held by an
    error, a refutation, a contradiction or an unresolved direction cleared nothing — its score
    decided none of it — and printing "margin 0.2500 clears auto_accept (0.75)" beside it tells a
    reviewer sorting the queue that a held cell cleared the top boundary by a quarter.
    """
    from canopy.verify.confidence import DECIDED_BY_A_HAIR

    pair = _a_cell_forced_by_an_error()
    flags = [CheckFlag(code="sd_nonpositive", severity="error", message="the SD is 0")]
    bucket, score, reasons = confidence(vote(pair), CONFIRMED, flags, None, orientation=ORIENTED)
    assert bucket == "needs_human" and score >= ACCEPT_WITH_NOTE     # the score said "pool it"
    assert not any("clears" in r or "misses" in r for r in reasons), reasons
    assert not any(DECIDED_BY_A_HAIR in r for r in reasons)
    assert any("not decided by the score" in r for r in reasons), reasons

    verdict = resolve_cell(dataset(), "late_adaptation", "A", pair, vote_result=vote(pair),
                           verdicts=CONFIRMED, flags=flags, orientation=ORIENTED, n_a=12, n_b=12)
    assert verdict.confidence == "needs_human"
    assert verdict.confidence_margin is None and verdict.nearest_boundary == ""


def test_a_cell_the_score_did_decide_still_publishes_its_margin():
    """The control, including the cell the score sent to a human on its own: there the margin is
    a real statement about which side of the line the evidence landed."""
    pair = _a_cell_forced_by_an_error()
    bucket, score, reasons = confidence(vote(pair), CONFIRMED, [], None, orientation=ORIENTED)
    assert bucket != "needs_human"
    assert any("clears" in r or "misses" in r for r in reasons)
    verdict = resolve_cell(dataset(), "late_adaptation", "A", pair, vote_result=vote(pair),
                           verdicts=CONFIRMED, flags=[], orientation=ORIENTED, n_a=12, n_b=12)
    assert verdict.confidence_margin is not None and verdict.nearest_boundary

    # …and a cell the SCORE sent to a human keeps its margin too — the score decided that bucket
    thin = [text_cand("a", 31.51, grounded=False)]
    low, low_score, low_reasons = confidence(vote(thin), (), [], None, orientation=ORIENTED)
    assert low == "needs_human" and low_score < ACCEPT_WITH_NOTE
    assert any("misses" in r or "clears" in r for r in low_reasons), low_reasons
