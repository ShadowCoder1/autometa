"""DECISION B — the answer tier: rule-answered best guesses under objection-crossing.

Two halves, like `test_questions.py`: unit fixtures shaped like the mapped run's own cells
(hand-built `Verdict`/`Candidate` pairs, every number taken from the record the rules doc
verified), and the real-record measure over
`runs/20260827-080330-handdominanceandupper-limbsensorimotorad` when it is on this machine —
the virgin firing table (A-M: exactly three cells, one row), the four-step Coudière replay
(T-B4), and the post-log empty diff (T-B15 as A7 re-scoped). Nothing here calls a model; the
guard test at the bottom pins that the whole tier — and a full repool over it — runs with the
client unconstructable.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from canopy.models import (BEST_GUESS_RULE_NAMES, Candidate, CheckFlag, DatasetSpec,
                           EffectSizeRecord, GroupSpec, OutcomeDef, StatsSettings, StudyMap,
                           Verdict, VerifierVerdict)
from canopy.pipeline import bestguess
from canopy.pipeline.bestguess import (ANSWER_RULES, adjudication_index, best_guess_payload,
                                       best_guess_rows, decision_b_fired)
from canopy.review.questions import CellRetirement

REPO = Path(__file__).resolve().parents[1]
CASE_STUDY = REPO / "runs" / "20260827-080330-handdominanceandupper-limbsensorimotorad"
needs_run = pytest.mark.skipif(not (CASE_STUDY / "manifest.json").exists(),
                               reason="the case-study run is not on this machine")

OUTCOME = OutcomeDef(key="late_adaptation", label="Late adaptation", definition="test")


# ------------------------------------------------------------------ hand-built cell fixtures
def _ret(**kw) -> CellRetirement:
    return CellRetirement(already=kw.get("already", []), answered=kw.get("answered", False),
                          answered_value=kw.get("answered", False),
                          overruled=set(kw.get("overruled", ())),
                          retired_codes=set(kw.get("retired", ())),
                          auto_stale=set(kw.get("auto", ())), auto=[],
                          landed=kw.get("landed"))


def _flag(code: str, severity: str, ids) -> CheckFlag:
    return CheckFlag(code=code, severity=severity, candidate_ids=list(ids))


def _verdict(group: str, **kw) -> Verdict:
    return Verdict(dataset_id=kw.get("dataset_id", "x:d1"), outcome_key="late_adaptation",
                   group=group, agreement=kw.get("agreement", "single"),
                   mean=kw.get("mean"), dispersion_value=kw.get("disp"),
                   dispersion_type=kw.get("disp_type", "SE"), n=kw.get("n"),
                   route=kw.get("route", "adjudicated"), higher_is_better=False,
                   flags=kw.get("flags", []), verifiers=kw.get("verifiers", []),
                   agreeing_ids=kw.get("agreeing_ids", []),
                   vote_tolerance=kw.get("tol"), adjudicated=kw.get("adjudicated", True),
                   confidence="needs_human", needs_human=True)


def _cand(group: str, cid: str, mean, disp=None, disp_type="SE", n=None, route="figure",
          status="found", locator="Fig. 1", ci_low=None, ci_high=None,
          model="") -> Candidate:
    return Candidate(kind="group_stats", candidate_id=cid, dataset_id="x:d1",
                     outcome_key="late_adaptation", group=group, status=status, mean=mean,
                     dispersion_value=disp, dispersion_type=disp_type, n=n, route=route,
                     locator=locator, ci_low=ci_low, ci_high=ci_high, model=model)


def _held(**kw) -> EffectSizeRecord:
    return EffectSizeRecord(dataset_id=kw.get("dataset_id", "x:d1"),
                            outcome_key="late_adaptation", route="not_convertible",
                            higher_is_better=False, confidence="needs_human",
                            flags=kw.get("flags", []))


def _datasets():
    return {"x:d1": DatasetSpec(dataset_id="x:d1", group_a=GroupSpec(label="Old"),
                                group_b=GroupSpec(label="Young"))}


def _settings(**kw) -> StatsSettings:
    return StatsSettings(**kw)


def _run(record, verdicts, candidates, *, retirements=None, adjudications=None,
         settings=None, sink=None):
    return best_guess_rows([], [record], outcome=OUTCOME,
                           settings=settings or _settings(), verdicts=verdicts,
                           candidates=candidates, datasets=_datasets(),
                           retirements=retirements or {},
                           adjudications=adjudications or {}, cell_guesses=sink)


#: a Coudière-shaped settled-adjudication pair, small: both cells' ruling means on the
#: verdicts, spreads present, one crossable contradicting flag on the A ensemble
def _settled_pair(a_flags=(), b_flags=(), a_disp=0.28, b_disp=0.25, verifiers_b=()):
    flags = list(a_flags) + list(b_flags)
    va = _verdict("A", mean=3.94, disp=a_disp, n=22, flags=flags)
    vb = _verdict("B", mean=3.13, disp=b_disp, n=22, agreement="disagree", flags=flags,
                  verifiers=list(verifiers_b))
    cands = [_cand("A", "x:d1:late_adaptation:A:digitize:ensemble", 3.95, 0.275, n=22),
             _cand("B", "x:d1:late_adaptation:B:digitize:ensemble", 3.2, 0.2, n=22),
             _cand("A", "x:d1:late_adaptation:A:text:opus#0", 3.94, route="text",
                   locator="Results ¶1")]
    adj = {("x:d1", "late_adaptation"): {"needs_human": False, "chosen_candidate_ids": [],
                                         "rationale": "", "locators": {"A": "Results ¶1",
                                                                       "B": "Results ¶2"}}}
    return va, vb, cands, adj


A_ENS = "x:d1:late_adaptation:A:digitize:ensemble"
B_ENS = "x:d1:late_adaptation:B:digitize:ensemble"


def test_the_catalogue_is_pinned_to_the_models_literal_and_the_knob_validates():
    assert ANSWER_RULES == BEST_GUESS_RULE_NAMES
    assert bestguess.ORIENTATION_CONTRADICTED in bestguess.CONTRADICTED
    with pytest.raises(ValueError):
        StatsSettings(best_guess_rules=["not_a_rule"])
    with pytest.raises(ValueError):
        StatsSettings(best_guess_rules=["adjudicated_value", "adjudicated_value"])
    assert StatsSettings(best_guess_rules=[]).best_guess_rules == []


def test_adjudicated_value_crosses_a_displaced_reading_and_builds_the_row():
    va, vb, cands, adj = _settled_pair(
        a_flags=[_flag("axis_conflict", "warn", [A_ENS])],
        verifiers_b=[VerifierVerdict(candidate_id=B_ENS, verdict="refuted", alt_mean=3.13)])
    rows, dec = _run(_held(flags=["axis_conflict"]), [va, vb], cands, adjudications=adj)
    assert dec[0].admitted and dec[0].rule == "adjudicated_value"
    assert [e["finding"] for e in dec[0].stepped_past["A"]] == ["axis_conflict"]
    assert "verifier_refuted" in [e["finding"] for e in dec[0].stepped_past["B"]]
    # the built row went through the ordinary resolver: es/var real, inputs present
    assert dec[0].es is not None and dec[0].var and dec[0].var > 0
    assert rows and rows[-1].best_guess_rule == "adjudicated_value"
    # stepped-past subjects are never empty (§3: an empty list on a crossed cell is a bug)
    for entries in dec[0].stepped_past.values():
        for entry in entries:
            assert entry["subject"], entry


def test_with_no_cell_data_or_an_empty_knob_every_decision_is_byte_identical():
    va, vb, cands, adj = _settled_pair(a_flags=[_flag("axis_conflict", "warn", [A_ENS])])
    held = _held(flags=["axis_conflict"])
    _, legacy = best_guess_rows([], [held], outcome=OUTCOME, settings=_settings())
    _, off = _run(held, [va, vb], cands, adjudications=adj,
                  settings=_settings(best_guess_rules=[]))
    assert (off[0].veto, off[0].reason) == (legacy[0].veto, legacy[0].reason)
    # …and the payload with zero fires carries none of the new keys (T-B15's first half)
    payload = best_guess_payload(None, None, off, added_rows=[], loo_bg=[],
                                 settings=_settings(best_guess_rules=[]), fired=False)
    for key in ("rules_applied", "by_rule", "by_rule_totals", "cell_guesses"):
        assert key not in payload


def test_a2_an_empty_subject_finding_is_never_crossable_and_never_masked():
    # the reviewer's fixture: a settled adjudication grafted onto a cell carrying an
    # error-severity flag with candidate_ids: [] — rule (a) does NOT fire, and the refusal
    # names the finding
    va, vb, cands, adj = _settled_pair(a_flags=[_flag("calibration_disputed", "error", [])])
    sink = []
    _, dec = _run(_held(flags=["calibration_disputed"]), [va, vb], cands,
                  adjudications=adj, sink=sink)
    assert not dec[0].admitted
    assert not decision_b_fired(dec, sink)


def test_a2_orientation_reader_contradicts_values_is_an_absolute():
    va, vb, cands, adj = _settled_pair(
        a_flags=[_flag("orientation_reader_contradicts_values", "error", [A_ENS])])
    _, dec = _run(_held(flags=["orientation_reader_contradicts_values"]), [va, vb], cands,
                  adjudications=adj)
    assert not dec[0].admitted
    # …and it never appears in any stepped_past list
    assert not dec[0].stepped_past


def test_a3_a_refutation_disputing_the_entering_value_is_uncrossable():
    # the reviewer's constructed refutation: no candidate id, alt 3.78 against entering 3.94
    va, vb, cands, adj = _settled_pair(
        verifiers_b=[VerifierVerdict(candidate_id="", verdict="refuted", alt_mean=3.78)])
    _, dec = _run(_held(), [va, vb], cands, adjudications=adj)
    assert not dec[0].admitted
    # …and one with no alt_mean at all is inapplicable ⇒ disputes (alt-fields-or-bust)
    va, vb, cands, adj = _settled_pair(
        verifiers_b=[VerifierVerdict(candidate_id="", verdict="refuted")])
    _, dec = _run(_held(), [va, vb], cands, adjudications=adj)
    assert not dec[0].admitted


def test_g7_a_human_answer_disables_every_rule_including_donor_completion():
    va, vb, cands, adj = _settled_pair(b_disp=None)
    # B answered spread-less: the ladder must NOT top up the human's deliberately partial
    # value, and with A alone the row cannot build — no admission, no cell fire either
    retirements = {("x:d1", "late_adaptation", "B"): _ret(answered=True,
                                                          landed={"mean": 3.13, "seq": 79})}
    sink = []
    _, dec = _run(_held(), [va, vb], cands, adjudications=adj, retirements=retirements,
                  sink=sink)
    assert not dec[0].admitted
    guessed_cells = {g["group"] for g in sink}
    assert "B" not in guessed_cells


def test_t_b8_retirement_changes_the_crossing_set():
    # a (c)-shaped cell: refutation standing blocks; the log's overrule retires it and the
    # rule fires — the engine reads the log-derived sets, never `verdict.verifier_verdict`
    ref = VerifierVerdict(candidate_id=A_ENS, verdict="refuted", alt_mean=9.0)
    va = _verdict("A", mean=12.0, disp=1.0, n=6, route="figure", adjudicated=False, tol=1.0,
                  verifiers=[ref])
    vb = _verdict("B", mean=9.9, disp=0.6, n=6, route="figure", adjudicated=False, tol=1.0)
    cands = [_cand("A", "x:d1:late_adaptation:A:digitize:readout:opus:direct", 12.0, 1.0,
                   n=6, model="claude-opus-5"),
             _cand("A", "x:d1:late_adaptation:A:digitize:readout:sonnet:direct", 12.2, 1.1,
                   n=6, model="claude-sonnet-5"),
             _cand("B", "x:d1:late_adaptation:B:digitize:readout:opus:direct", 9.9, 0.6,
                   n=6, model="claude-opus-5"),
             _cand("B", "x:d1:late_adaptation:B:digitize:readout:sonnet:direct", 9.8, 0.6,
                   n=6, model="claude-sonnet-5")]
    held = _held()
    _, blocked = _run(held, [va, vb], cands)
    assert not blocked[0].admitted
    retirements = {("x:d1", "late_adaptation", "A"): _ret(overruled={"verifier_refuted"})}
    _, fired = _run(held, [va, vb], cands, retirements=retirements)
    assert fired[0].admitted and fired[0].rule == "agreed_solo_read"


def test_the_d1_boundary_blocks_agreed_solo_read_on_a_contradicting_code():
    # Poh's shape: two families agree, and the cell's own axis_conflict blocks (c) —
    # "agreement about a number never establishes that it is the right number"
    va = _verdict("A", mean=44.25, disp=1.0, n=10, route="figure", adjudicated=False,
                  tol=0.885, flags=[_flag("axis_conflict", "warn", [A_ENS])])
    vb = _verdict("B", mean=40.0, disp=1.0, n=10, route="figure", adjudicated=False, tol=1.0)
    cands = [_cand("A", "x:d1:late_adaptation:A:digitize:readout:opus:direct", 44.0, 1.0,
                   n=10, model="claude-opus-5"),
             _cand("A", "x:d1:late_adaptation:A:digitize:readout:sonnet:direct", 44.5, 1.0,
                   n=10, model="claude-sonnet-5")]
    _, dec = _run(_held(flags=["axis_conflict"]), [va, vb], cands)
    assert not dec[0].admitted and dec[0].veto == "contradicted_value"
    # …and calibration_disputed alone blocks too (the shared-instrument boundary)
    va2 = _verdict("A", mean=44.25, disp=1.0, n=10, route="figure", adjudicated=False,
                   tol=0.885, flags=[_flag("calibration_disputed", "error", [A_ENS])])
    _, dec = _run(_held(), [va2, vb], cands)
    assert not dec[0].admitted


# ------------------------------------------------------------------------- donor ladder (A6)
def test_t_b10_the_donor_ladder_gates_ranks_and_fences():
    settings = _settings()
    va, vb, cands, adj = _settled_pair(a_disp=None)
    # the shared-id designated group (D2's four readings) resolves to the largest converted SE
    group = [_cand("A", A_ENS, 3.95, 0.275, n=22, locator="Fig. 5, Adapt1 inset"),
             _cand("A", A_ENS, 3.91, 0.28, n=22, locator="Fig. 5, main panel"),
             _cand("A", A_ENS, 3.90, 0.2, n=22, locator="Fig. 6A inset"),
             _cand("A", A_ENS, 3.95, 0.15, n=22, locator="Fig. 6B inset")]
    va2 = _verdict("A", mean=3.94, n=22, disp=None, agreeing_ids=[A_ENS])
    donor, eligible, refusal = bestguess._donor_spread(
        bestguess._cell_context(outcome=OUTCOME, settings=settings, verdicts=[va2, vb],
                                candidates=group, datasets=_datasets(), retirements={},
                                adjudications={}, cell_guesses=None),
        va2, group, {"chosen_candidate_ids": []}, 3.94, 22)
    assert refusal == "" and donor.dispersion_value == 0.28
    assert donor.locator == "Fig. 5, main panel"
    # the wrong-series read fails the z ≤ 1 gate: |3.75 − 3.13| = 0.62 > 0.2
    pool = [_cand("B", B_ENS, 3.2, 0.2, n=22), _cand("B", B_ENS, 3.75, 0.2, n=22),
            _cand("B", B_ENS, 3.11, 0.25, n=22), _cand("B", B_ENS, 3.21, 0.175, n=22)]
    eligible = bestguess._eligible_donors(pool, 3.13, 22, settings)
    assert 3.75 not in [c.mean for c in eligible]
    assert sorted(c.dispersion_value for c in eligible) == [0.175, 0.2, 0.25]
    # a dispersion-less donor has no SE to be judged by and is ineligible (Kumar's −360 too)
    assert bestguess._eligible_donors([_cand("B", B_ENS, 3.13), 
                                       _cand("B", B_ENS, -360.05)], 3.13, 22, settings) == []
    # mixed types compare on the converted-SE scale: 0.9 SD at n 22 (SE 0.19) loses to 0.35 SE
    mixed = [_cand("B", B_ENS + "x", 3.13, 0.9, disp_type="SD", n=22),
             _cand("B", B_ENS, 3.13, 0.35, n=22)]
    ranked = sorted(mixed, key=lambda d: bestguess._donor_key(d, 22, settings))
    assert ranked[0].dispersion_value == 0.35
    # the reacquire read is absent from a step-3 browse (the reviewer's near-miss: 3.87±0.373
    # passes the gate and would win the tiebreak) — designation is the record's voice, the
    # browse is what gets fenced
    va3 = _verdict("A", mean=3.94, n=22, disp=None, agreeing_ids=[])
    browse = [_cand("A", A_ENS, 3.95, 0.275, n=22),
              _cand("A", A_ENS + ":reacquire", 3.87114660346871, 0.37304598697826175, n=22),
              _cand("A", "x:d1:late_adaptation:A:digitize:vlm_coords:x", 3.94, 0.5, n=22),
              _cand("A", "x:d1:late_adaptation:A:digitize:raster_cv", 3.94, 0.6, n=22)]
    ctx = bestguess._cell_context(outcome=OUTCOME, settings=settings, verdicts=[va3, vb],
                                  candidates=browse, datasets=_datasets(), retirements={},
                                  adjudications={}, cell_guesses=None)
    donor, eligible, refusal = bestguess._donor_spread(ctx, va3, browse, None, 3.94, 22)
    assert donor.dispersion_value == 0.275 and len(eligible) == 1
    # …but a designation that NAMES the reacquire id is honoured
    va4 = _verdict("A", mean=3.94, n=22, disp=None, agreeing_ids=[A_ENS + ":reacquire"])
    donor, _, _ = bestguess._donor_spread(ctx, va4, browse, None, 3.94, 22)
    assert donor.dispersion_value == 0.37304598697826175
    # A6's sanity ceiling: a straddler at 3× the eligible median is refused, by name
    wide = [_cand("A", A_ENS, 3.94, 0.2, n=22), _cand("A", A_ENS, 3.94, 0.21, n=22),
            _cand("A", A_ENS, 3.94, 0.19, n=22),
            _cand("A", A_ENS, 3.9, 0.62, n=22, locator="the straddler")]
    va5 = _verdict("A", mean=3.94, n=22, disp=None, agreeing_ids=[A_ENS])
    donor, _, refusal = bestguess._donor_spread(ctx, va5, wide, None, 3.94, 22)
    assert donor is None and "sanity ceiling" in refusal
    # disagreeing donor n ⇒ the slot is unanswerable ⇒ no guess
    two_ns = [_cand("A", A_ENS, 3.94, 0.2, n=20), _cand("A", A_ENS, 3.94, 0.2, n=22)]
    n, why = bestguess._unique_n(two_ns, _verdict("A", mean=3.94, disp=None))
    assert n is None and "disagree" in why


# ---------------------------------------------------------------------- route mapping (M5)
def test_t_b12_route_literals_map_to_precedence_entries_and_the_order_decides():
    order = _settings().route_precedence
    assert bestguess._mapped_entry(_cand("A", "c1", 1.0, 0.1, "SD", route="text"),
                                   order) == "text_mean_sd"
    for kind in ("SE", "CI95", "CI90"):
        assert bestguess._mapped_entry(_cand("A", "c1", 1.0, 0.1, kind, route="text"),
                                       order) == "text_mean_se_ci"
    # a mean-only printed value's authority is its mean — earliest text_* entry present
    assert bestguess._mapped_entry(_cand("A", "c1", 1.0, None, "UNKNOWN", route="text"),
                                   order) == "text_mean_sd"
    for literal in ("table", "figure", "test_statistic", "p_value", "reported_d"):
        assert bestguess._mapped_entry(_cand("A", "c1", 1.0, route=literal),
                                       order) == literal
    assert bestguess._mapped_entry(_cand("A", "c1", 1.0, route="martian"), order) == ""

    def kumar_cell(settings):
        text = [_cand("A", "x:d1:late_adaptation:A:text:opus#0", 26.386, 0.857, "CI95",
                      route="text", ci_low=24.519, ci_high=28.254, locator="Results"),
                _cand("A", "x:d1:late_adaptation:A:text:sonnet#0", 26.386, 0.857, "SE",
                      route="text", ci_low=24.519, ci_high=28.254, locator="Results")]
        figure = [_cand("A", A_ENS, 26.5, 1.7, n=13, locator="fig05 panel A"),
                  _cand("A", A_ENS, 26.25, None, "CI95", n=13, locator="fig05 panel D")]
        va = _verdict("A", mean=26.386, disp=0.857, disp_type="UNKNOWN", n=13,
                      agreement="disagree", tol=None,
                      flags=[_flag("locator_reads_conflict", "warn", [A_ENS])],
                      verifiers=[VerifierVerdict(candidate_id=A_ENS, verdict="refuted",
                                                 alt_mean=26.386, alt_n=13)])
        adj = {"needs_human": True, "chosen_candidate_ids": [], "locators": {}}
        ctx = bestguess._cell_context(outcome=OUTCOME, settings=settings, verdicts=[va],
                                      candidates=text + figure, datasets=_datasets(),
                                      retirements={}, adjudications={}, cell_guesses=None)
        return ctx, va, text + figure, adj

    ctx, va, cands, adj = kumar_cell(_settings())
    guess = bestguess._try_route_precedence(ctx, va, cands, None, adj)
    # the A4 arithmetic adjudicates the CI95-vs-SE split to the SEM, and the unsettled
    # adjudication is crossed as concurring (T-B13's second arm)
    assert guess.rule == "route_precedence" and guess.mean == 26.386
    assert (guess.dispersion_value, guess.dispersion_type) == (0.857, "SE")
    assert "adjudicated_unsettled" in [e["finding"] for e in guess.stepped_past]
    assert guess.n == 13
    # reorder the precedence to put figure first ⇒ the figure reading wins instead
    figure_first = _settings(route_precedence=["figure", "text_mean_sd", "table",
                                               "text_mean_se_ci", "test_statistic",
                                               "p_value", "reported_d"])
    ctx2, va2, cands2, adj2 = kumar_cell(figure_first)
    guess2 = bestguess._try_route_precedence(ctx2, va2, cands2, None, adj2)
    assert guess2.refusal or guess2.mean != 26.386   # the figure readings are the winner now

    # an unsettled adjudication that DISPUTES the winner blocks (T-B13's third arm)
    va3 = _verdict("A", mean=25.0, disp=None, disp_type="UNKNOWN", n=13,
                   agreement="disagree", tol=None)
    ctx3, _, cands3, _ = kumar_cell(_settings())
    guess3 = bestguess._try_route_precedence(ctx3, va3, cands3, None,
                                             {"needs_human": True,
                                              "chosen_candidate_ids": [], "locators": {}})
    assert guess3.refusal and "did not settle it at this value" in guess3.refusal


def test_a4_a_unanimous_label_the_bounds_refute_blocks_with_the_arithmetic_in_the_reason():
    # Kumar d1 B's shape: both text readings say CI95; the quoted CI's half-width is the
    # value × t(12) — the value is the SEM, a label no reading carries ⇒ no guess
    text = [_cand("B", "x:d1:late_adaptation:B:text:opus#1", 24.572, 1.039, "CI95",
                  route="text", ci_low=22.307, ci_high=26.837, locator="Results"),
            _cand("B", "x:d1:late_adaptation:B:text:sonnet#1", 24.572, 1.039, "CI95",
                  route="text", ci_low=22.307, ci_high=26.837, locator="Results")]
    figure = [_cand("B", B_ENS, 24.4, None, "CI95", n=13, locator="fig03 panel D")]
    vb = _verdict("B", mean=24.572, disp=1.039, disp_type="UNKNOWN", n=13,
                  agreement="disagree", tol=None,
                  flags=[_flag("dispersion_type_conflict", "warn", [B_ENS])])
    ctx = bestguess._cell_context(outcome=OUTCOME, settings=_settings(), verdicts=[vb],
                                  candidates=text + figure, datasets=_datasets(),
                                  retirements={}, adjudications={}, cell_guesses=None)
    guess = bestguess._try_route_precedence(ctx, vb, text + figure, None,
                                            {"needs_human": True, "chosen_candidate_ids": [],
                                             "locators": {}})
    assert guess.rule == "" and "refuted by the record's own bounds arithmetic" in guess.refusal
    assert "dispersion_type_conflict" in guess.refusal
    # a label conflict with NO bounds to adjudicate blocks too (doubt-against)
    boundless = [c.model_copy(update={"ci_low": None, "ci_high": None,
                                      "dispersion_type": kind})
                 for c, kind in zip(text, ("CI95", "SE"))]
    guess2 = bestguess._try_route_precedence(ctx, vb, boundless + figure, None, None)
    assert guess2.rule == "" and "no quoted bounds" in guess2.refusal
    # a winner-internal MEAN disagreement kills (b) outright
    split = [text[0], text[1].model_copy(update={"mean": 25.9})]
    guess3 = bestguess._try_route_precedence(ctx, vb, split + figure, None, None)
    assert guess3.rule == "" and "disagree about the mean" in guess3.refusal


# --------------------------------------------------------------------------- masking (A5)
def test_t_b17_masking_is_instance_scoped_and_exact():
    # an A fire cannot carry the row when B — the cell an uncrossed instance attaches to —
    # has no authority: the fire is recorded as blocked, never admitted
    va2, vb2, cands2, adj2 = _settled_pair(
        a_flags=[_flag("axis_conflict", "warn", [A_ENS])])
    vb2.flags = list(vb2.flags) + [_flag("series_identity_conflict", "warn", [B_ENS])]
    va2.flags = list(vb2.flags)                  # the pair shares its flag list (fact 1.1)
    vb3 = _verdict("B", mean=None, agreement="disagree", route="",
                   flags=list(va2.flags))
    sink = []
    _, dec = _run(_held(flags=["axis_conflict", "series_identity_conflict"]), [va2, vb3],
                  cands2, adjudications=adj2, sink=sink)
    assert not any(d.admitted for d in dec)
    # the A fire was recorded, blocked by the unusable sibling
    assert sink and sink[0]["group"] == "A" and sink[0]["row_blocked_by"]
    # …and an instance with EMPTY candidate_ids attached to no cell is never dropped: it
    # blocks the fire at the crossing stage already (A2), asserted above; here assert the
    # mask function itself keeps it
    flag = _flag("axis_conflict", "warn", [])
    copies = {"A": va2.model_copy(deep=True)}
    copies["A"].flags = [flag]
    ctx = bestguess._cell_context(outcome=OUTCOME, settings=_settings(),
                                  verdicts=[va2, vb3], candidates=cands2,
                                  datasets=_datasets(), retirements={}, adjudications={},
                                  cell_guesses=None)
    masked = bestguess._mask_pair(ctx, _held(), copies,
                                  {"A": bestguess._CellGuess(group="A", answered=True)})
    assert copies["A"].flags == [flag] and masked == {}


def test_t_b17_a_newly_derived_row_level_finding_still_vetoes_the_built_row():
    # the guessed values themselves trip the resolver's own screen (|d| implausible): the
    # veto is not maskable because it is derived from the entered numbers, not crossed
    va, vb, cands, adj = _settled_pair()
    va.mean, va.dispersion_value = 100.0, 0.1
    vb.mean, vb.dispersion_value = 0.0, 0.1
    sink = []
    _, dec = _run(_held(), [va, vb], cands, adjudications=adj, sink=sink)
    assert not any(d.admitted for d in dec)
    assert sink and "vetoed" in sink[0]["row_blocked_by"]


def test_t_b17_the_leak_fixture_an_uncrossed_sibling_instance_vetoes_the_built_row():
    # cell A fires and crosses ITS axis_conflict; cell B is a RELEASED, value-complete cell
    # carrying an uncrossed series_identity_conflict of its own readings — code-string
    # masking would drop it on the strength of A's crossing; instance-scoped masking keeps
    # it, the built row carries the code, and the row veto fires
    va, vb, cands, adj = _settled_pair(a_flags=[_flag("axis_conflict", "warn", [A_ENS])])
    vb.flags = list(va.flags) + [_flag("series_identity_conflict", "warn", [B_ENS])]
    va.flags = list(vb.flags)                    # the pair shares its flag list (fact 1.1)
    vb.needs_human = False
    vb.confidence = "accept_with_note"
    vb.adjudicated = False
    vb.agreement = "single"
    vb.route = "figure"
    sink = []
    _, dec = _run(_held(flags=["axis_conflict", "series_identity_conflict"]), [va, vb],
                  cands, adjudications=adj, sink=sink,
                  settings=_settings(best_guess_rules=["adjudicated_value"]))
    assert not any(d.admitted for d in dec)
    assert sink and "series_identity_conflict" in sink[0]["row_blocked_by"]


# ---------------------------------------------------------------- the real record (A-M etc.)
def _load_paper(paper12: str):
    def stage(name):
        return json.loads((CASE_STUDY / "papers" / paper12 / f"{name}.json").read_text())
    verify = stage("verify")
    verdicts = [Verdict.model_validate(v) for v in verify["verdicts"]]
    candidates = [Candidate.model_validate(c) for c in stage("extract")["candidates"]]
    candidates += [Candidate.model_validate(c) for c in verify.get("extra_candidates") or []]
    records = [EffectSizeRecord.model_validate(r) for r in stage("resolve")["records"]]
    study = StudyMap.model_validate(stage("map")["study"])
    return verdicts, candidates, records, {d.dataset_id: d for d in study.datasets}


def _virgin_fires():
    """Every answer-tier fire over every paper's virgin stage files — THE measure (A-M)."""
    from canopy.protocol import load_protocol

    protocol = load_protocol(CASE_STUDY / "protocol.yaml")
    fires: dict[tuple[str, str, str], dict] = {}
    admitted_rows: list[str] = []
    decisions_all = []
    for paper in sorted(p.name for p in (CASE_STUDY / "papers").iterdir()
                        if (p / "resolve.json").exists()):
        verdicts, candidates, records, datasets = _load_paper(paper)
        adjudications = adjudication_index(CASE_STUDY, [paper])
        for outcome in protocol.outcomes:
            held = [r for r in records if r.outcome_key == outcome.key
                    and not (r.confidence in protocol.stats.primary_analysis_includes
                             and r.es is not None and r.var)]
            if not held:
                continue
            sink: list[dict] = []
            _, decisions = best_guess_rows(
                [], held, outcome=outcome, settings=protocol.stats, verdicts=verdicts,
                candidates=candidates, datasets=datasets, retirements={},
                adjudications=adjudications, cell_guesses=sink)
            decisions_all.extend(decisions)
            for decision in decisions:
                if decision.admitted and decision.entered:
                    admitted_rows.append(f"{decision.dataset_id}|{outcome.key}")
                    for group, slots in decision.entered.items():
                        fires[(decision.dataset_id, outcome.key, group)] = {
                            "rule": next(r for r in ANSWER_RULES
                                         if r in decision.rule.split(";")
                                         and any(h["group"] == group and h["rule"] == r
                                                 for h in decision.answered_holds)),
                            "slots": slots,
                            "stepped": sorted(e["finding"] for e in
                                              decision.stepped_past.get(group, []))}
                if decision.admitted and decision.rule == "agreed_solo_read":
                    admitted_rows.append(f"RENAME {decision.dataset_id}|{outcome.key}")
            for guess in sink:
                fires[(guess["dataset_id"], outcome.key, guess["group"])] = {
                    "rule": guess["rule"], "slots": guess["entered"],
                    "stepped": sorted(e["finding"] for e in guess["stepped_past"]),
                    "blocked": guess["row_blocked_by"]}
    return fires, admitted_rows, decisions_all


@needs_run
def test_the_virgin_record_fires_exactly_three_cells_and_one_row_enters():
    fires, admitted_rows, _ = _virgin_fires()
    assert sorted(fires) == [("19ed99c0a7ad:d1", "late_adaptation", "A"),
                             ("19ed99c0a7ad:d1", "late_adaptation", "B"),
                             ("cfcf61c1c2b7:d2", "late_adaptation", "A")]
    a = fires[("19ed99c0a7ad:d1", "late_adaptation", "A")]
    assert a["rule"] == "adjudicated_value"
    assert a["slots"]["mean"]["value"] == 3.94
    assert (a["slots"]["dispersion"]["value"], a["slots"]["dispersion"]["type"]) == (0.28, "SE")
    assert a["slots"]["dispersion"]["passed_over"] == ["0.275", "0.2", "0.15"]   # D2's group
    assert a["slots"]["n"]["value"] == 22
    assert a["stepped"] == ["axis_conflict", "calibration_disputed", "locator_reads_conflict"]
    b = fires[("19ed99c0a7ad:d1", "late_adaptation", "B")]
    assert b["rule"] == "adjudicated_value"
    assert b["slots"]["mean"]["value"] == 3.13
    assert (b["slots"]["dispersion"]["value"], b["slots"]["dispersion"]["type"]) == (0.25, "SE")
    assert "3.75" not in json.dumps(b["slots"])          # the z ≤ 1 gate's exclusion (M2-2)
    assert b["stepped"] == ["axis_conflict", "calibration_disputed", "locator_reads_conflict",
                            "verifier_refuted"]
    k = fires[("cfcf61c1c2b7:d2", "late_adaptation", "A")]
    assert k["rule"] == "route_precedence"
    assert k["slots"]["mean"]["value"] == 26.386
    assert (k["slots"]["dispersion"]["value"], k["slots"]["dispersion"]["type"]) == (0.857, "SE")
    assert k["slots"]["n"]["value"] == 13
    assert k["stepped"] == ["adjudicated_unsettled", "locator_reads_conflict",
                            "verifier_refuted"]
    assert "d2 B" in k["blocked"] or "cell B" in k["blocked"]
    # ONE row enters, in late_adaptation; renames: zero (Kirby fails the family test, Poh
    # the D1 boundary)
    assert admitted_rows == ["19ed99c0a7ad:d1|late_adaptation"]
    # the invariant of §3: no stepped_past entry anywhere has an empty subject, except the
    # concurrence marker whose subject is the ruling itself
    _, _, decisions = _virgin_fires()
    for decision in decisions:
        for entries in decision.stepped_past.values():
            for entry in entries:
                # empty subjects are allowed only where the objection's ground is not a
                # reading: a concurring open adjudication, or an id-less refutation crossed
                # on its own supporting citation (A3 test 2 — review F4)
                assert entry["subject"] or entry["finding"] == "adjudicated_unsettled" \
                    or (entry["finding"] == "verifier_refuted"
                        and "citation" in entry["basis"])


@needs_run
def test_kumar_d1_b_is_blocked_by_the_bounds_arithmetic_not_guessed():
    from canopy.protocol import load_protocol

    protocol = load_protocol(CASE_STUDY / "protocol.yaml")
    verdicts, candidates, records, datasets = _load_paper("cfcf61c1c2b7")
    verdict = next(v for v in verdicts if v.dataset_id == "cfcf61c1c2b7:d1"
                   and v.outcome_key == "late_adaptation" and v.group == "B")
    ctx = bestguess._cell_context(
        outcome=protocol.outcome("late_adaptation"), settings=protocol.stats,
        verdicts=verdicts, candidates=candidates, datasets=datasets, retirements={},
        adjudications=adjudication_index(CASE_STUDY, ["cfcf61c1c2b7"]), cell_guesses=None)
    guess = bestguess._answer_cell(ctx, verdict)
    assert guess.rule == "" and guess.answered is False
    assert "refuted by the record's own bounds arithmetic" in guess.refusal
    assert "dispersion_type_conflict" in guess.refusal
    assert "22.307" in guess.refusal and "26.837" in guess.refusal


@needs_run
def test_the_run_holds_exactly_eight_empty_subject_orientation_contradictions():
    count = 0
    for paper in sorted(p.name for p in (CASE_STUDY / "papers").iterdir()
                        if (p / "verify.json").exists()):
        payload = json.loads((CASE_STUDY / "papers" / paper / "verify.json").read_text())
        for verdict in payload["verdicts"]:
            for flag in verdict.get("flags") or []:
                if not flag.get("candidate_ids") and flag.get("severity") == "error" \
                        and flag["code"] == "orientation_reader_contradicts_values":
                    count += 1
    assert count == 8                     # 3e8809d7f07a d1/d2 A+B, d8f649e55da7 d1/d2 afte


@needs_run
def test_t_b2_and_t_b16_evaluation_is_pure_and_repeatable():
    verdicts, candidates, records, datasets = _load_paper("19ed99c0a7ad")
    from canopy.protocol import load_protocol

    protocol = load_protocol(CASE_STUDY / "protocol.yaml")
    outcome = protocol.outcome("late_adaptation")
    held = [r for r in records if r.outcome_key == "late_adaptation"]
    adjudications = adjudication_index(CASE_STUDY, ["19ed99c0a7ad"])
    results = []
    for _ in range(2):
        sink: list[dict] = []
        _, decisions = best_guess_rows([], held, outcome=outcome, settings=protocol.stats,
                                       verdicts=verdicts, candidates=candidates,
                                       datasets=datasets, retirements={},
                                       adjudications=adjudications, cell_guesses=sink)
        results.append((json.dumps([{k: v for k, v in d.__dict__.items() if k != "row"}
                                    for d in decisions], default=str, sort_keys=True),
                        json.dumps(sink, default=str, sort_keys=True)))
    assert results[0] == results[1]


# ------------------------------------------------------------------- T-B4: the Coudière replay
COUDIERE = "19ed99c0a7ad"


def _coudiere_run(tmp_path: Path) -> Path:
    """A one-paper run dir built from the real Coudière stage files, verbatim."""
    run = tmp_path / "coudiere"
    (run / "papers" / COUDIERE).mkdir(parents=True)
    for name in ("map.json", "extract.json", "verify.json", "resolve.json", "ingest.json",
                 "file_id.txt"):
        source = CASE_STUDY / "papers" / COUDIERE / name
        if source.exists():
            shutil.copy(source, run / "papers" / COUDIERE / name)
    shutil.copy(CASE_STUDY / "protocol.yaml", run / "protocol.yaml")
    for name in ("prisma.json", "exclusions.json"):
        if (CASE_STUDY / name).exists():
            shutil.copy(CASE_STUDY / name, run / name)
    manifest = json.loads((CASE_STUDY / "manifest.json").read_text())
    manifest["papers"] = [p for p in manifest["papers"]
                          if p["paper_id"].startswith(COUDIERE)]
    manifest["human_review_queue"] = []
    manifest["outputs"] = {}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def _coudiere_log_lines() -> list[str]:
    return [line for line in (CASE_STUDY / "overrides.jsonl").read_text().splitlines()
            if f'"{COUDIERE}' in line]


def _repool(run: Path) -> dict:
    from canopy.pipeline.overrides import apply_overrides_and_repool

    apply_overrides_and_repool(run)
    return json.loads((run / "results" / "late_adaptation" / "pooled.json").read_text())


def _queue_rows(run: Path) -> list[dict]:
    return json.loads((run / "human_review_queue.json").read_text())


@needs_run
def test_t_b4_the_coudiere_replay_virgin_then_seq78_then_seq79_then_the_whole_log(tmp_path):
    run = _coudiere_run(tmp_path)
    lines = _coudiere_log_lines()
    seq78 = next(l for l in lines if json.loads(l).get("seq") == 78)
    seq79 = next(l for l in lines if json.loads(l).get("seq") == 79)

    # ---- step 1, virgin: both cells guessed, the row enters the bg line
    payload = _repool(run)
    bg = payload["best_guess"]
    assert bg["n_added"] == 1 and bg["rules_applied"] == list(BEST_GUESS_RULE_NAMES)
    entry = bg["added"][0]
    assert entry["rule"] == "adjudicated_value"
    assert entry["entered"]["A"]["mean"]["value"] == 3.94
    assert entry["entered"]["A"]["dispersion"]["value"] == 0.28
    assert entry["entered"]["B"]["dispersion"]["value"] == 0.25
    assert sorted(e["finding"] for e in entry["stepped_past"]["A"]) == \
        ["axis_conflict", "calibration_disputed", "locator_reads_conflict"]
    assert sorted(e["finding"] for e in entry["stepped_past"]["B"]) == \
        ["axis_conflict", "calibration_disputed", "locator_reads_conflict",
         "verifier_refuted"]
    assert entry["masked_on_answered"] == {}
    # the row's es is the resolver's own output at exactly these inputs — pinned by value
    # here once computed, never hand-derived (reviewer N-d)
    assert entry["es"] == pytest.approx(-0.6389289818377382)
    assert bg["cell_guesses"] == []
    # the d1 late questions are OPEN and their cards carry the guess sub-block
    queue = _queue_rows(run)
    mine = [q for q in queue if q["dataset_id"] == f"{COUDIERE}:d1"
            and q["outcome_key"] == "late_adaptation"]
    assert mine and all(q["best_guess_rule"] == "adjudicated_value" for q in mine)
    # per-group decoration (review F1): each row carries ITS cell's entered slots, never the
    # sibling's
    by_group = {q["group"]: q["best_guess_entered"] for q in mine}
    assert "3.94" in by_group["A"] and "3.13" not in by_group["A"]
    assert "3.13" in by_group["B"] and "3.94" not in by_group["B"]
    questions = json.loads((run / "questions.json").read_text())
    cards = [q for q in questions if q.get("dataset_id") == f"{COUDIERE}:d1"
             and q.get("outcome_key") == "late_adaptation"]
    assert cards and any("temporarily entered" in (q.get("best_guess") or {}).get("note", "")
                         for q in cards)
    # the csv schema carries the two trailing columns (A7 — always, empty when quiet)
    header = (run / "human_review_queue.csv").read_text().splitlines()[0]
    assert header.endswith("best_guess_rule,best_guess_entered")
    # the unfired outcome keeps a byte-identical block: no DECISION B keys at all
    other = json.loads((run / "results" / "aftereffect" / "pooled.json").read_text())
    assert "rules_applied" not in other["best_guess"]
    # the assembled caveat marks the fired outcome's conclusion
    assert any("by answering open review questions by rule" in s
               for s in payload["conclusion"]["best_guess_sentences"])

    # ---- step 2, + seq 78 (human A value): A is F7 for ever; the MIXED row persists
    (run / "overrides.jsonl").write_text(seq78 + "\n")
    log_bytes = (run / "overrides.jsonl").read_bytes()
    bg = _repool(run)["best_guess"]
    assert bg["n_added"] == 1
    entry = bg["added"][0]
    assert entry["rule"] == "adjudicated_value"          # from B alone
    assert list(entry["entered"]) == ["B"]
    assert "human's own answer stands" in entry["reason"]
    assert sorted(e["finding"] for e in entry["stepped_past"]["B"]) == \
        ["axis_conflict", "calibration_disputed", "locator_reads_conflict",
         "verifier_refuted"]
    # NOTE (deviation from A5's expected step-2 state, forced by the record): seq 78's mean
    # answer RETIRES axis_conflict and calibration_disputed on A through VALUE_CLEARS_MEAN
    # (overrides.py), so nothing is left standing on A to mask — `masked_on_answered` records
    # only instances that STAND on an answered cell, and here there are none.
    assert (run / "overrides.jsonl").read_bytes() == log_bytes    # guesses wrote nothing

    # ---- step 3, + seq 79 (human B value, dispersion null): the row FALLS OUT of the line
    (run / "overrides.jsonl").write_text(seq78 + "\n" + seq79 + "\n")
    bg = _repool(run)["best_guess"]
    assert bg["n_added"] == 0
    assert "rules_applied" not in bg                     # zero fire ⇒ fire-gated silence
    assert all(not q["best_guess_entered"] for q in _queue_rows(run))
    # the questions still show the landed value: the cells are still held, no guess anywhere
    header = (run / "human_review_queue.csv").read_text().splitlines()[0]
    assert header.endswith("best_guess_rule,best_guess_entered")

    # ---- step 4, + the whole Coudière log: the row is PRIMARY at the human values
    (run / "overrides.jsonl").write_text("\n".join(lines) + "\n")
    log_bytes = (run / "overrides.jsonl").read_bytes()
    payload = _repool(run)
    bg = payload["best_guess"]
    assert bg["n_added"] == 0 and "rules_applied" not in bg
    assert "stepped_past" not in json.dumps(payload)
    table = json.loads((run / "results" / "late_adaptation" /
                        "extraction_table.json").read_text())
    row = next(r for r in table if r["dataset_id"] == f"{COUDIERE}:d1")
    assert row["primary_row"] and row["es"] == pytest.approx(-0.589, abs=0.005)
    assert row["mean_a"] == 3.94 and row["mean_b"] == 3.13
    assert (run / "overrides.jsonl").read_bytes() == log_bytes


@needs_run
def test_no_model_calls_anywhere_in_the_tier_or_a_repool_over_it(tmp_path, monkeypatch):
    import canopy.llm.client as llm_client

    def boom(*args, **kwargs):
        raise AssertionError("the best-guess tier made (or prepared) a model call")

    monkeypatch.setattr(llm_client.LLMClient, "__init__", boom)
    run = _coudiere_run(tmp_path)
    payload = _repool(run)                               # a full repool, guard armed
    assert payload["best_guess"]["n_added"] == 1
    fires, _, _ = _virgin_fires()                        # …and the raw engine sweep too
    assert len(fires) == 3


# ------------------------------------------------ T-B15 as A7 re-scoped: the post-log measure
def _full_copy(tmp_path: Path, name: str, knob_off: bool) -> Path:
    """The whole case-study run's stage files and log, no figures — enough to repool."""
    run = tmp_path / name
    run.mkdir(parents=True)
    for top in ("manifest.json", "overrides.jsonl", "prisma.json", "exclusions.json"):
        if (CASE_STUDY / top).exists():
            shutil.copy(CASE_STUDY / top, run / top)
    if (CASE_STUDY / "provenance" / "provenance.json").exists():
        (run / "provenance").mkdir()
        shutil.copy(CASE_STUDY / "provenance" / "provenance.json",
                    run / "provenance" / "provenance.json")
    text = (CASE_STUDY / "protocol.yaml").read_text()
    if knob_off:
        text = text.replace("stats:\n", "stats:\n  best_guess_rules: []\n", 1)
    (run / "protocol.yaml").write_text(text)
    for paper in (CASE_STUDY / "papers").iterdir():
        if not (paper / "resolve.json").exists():
            continue
        dest = run / "papers" / paper.name
        dest.mkdir(parents=True)
        for name_ in ("map.json", "extract.json", "verify.json", "resolve.json",
                      "ingest.json", "file_id.txt"):
            if (paper / name_).exists():
                shutil.copy(paper / name_, dest / name_)
    return run


def _strip_knob(payload):
    """Drop the knob's own echo (`settings.best_guess_rules`) before comparing: the two
    repools differ by the knob BY CONSTRUCTION, and the claim under test is that nothing
    else differs."""
    if isinstance(payload, dict):
        return {k: _strip_knob(v) for k, v in payload.items() if k != "best_guess_rules"}
    if isinstance(payload, list):
        return [_strip_knob(v) for v in payload]
    return payload


@needs_run
@pytest.mark.slow
def test_the_post_seq302_record_repools_to_an_empty_diff_modulo_the_two_queue_columns(tmp_path):
    # A-M: all three virgin-firing cells are human-answered in the 302-line log (that is why
    # the review adopted them — F7), the only settled adjudication is released, and both
    # rename candidates fail — so the DEFAULT knob and `[]` must produce identical artefacts,
    # the two always-present (empty) queue columns included on BOTH sides (A7's stable schema)
    from canopy.pipeline.overrides import apply_overrides_and_repool

    off = _full_copy(tmp_path, "off", knob_off=True)
    on = _full_copy(tmp_path, "on", knob_off=False)
    apply_overrides_and_repool(off)
    apply_overrides_and_repool(on)
    compared = 0
    for rel in sorted(p.relative_to(on) for p in (on / "results").rglob("*")
                      if p.suffix in (".json", ".csv", ".md")):
        left, right = (off / rel), (on / rel)
        assert left.exists(), f"{rel} written only under the default knob"
        if rel.suffix == ".json":
            assert _strip_knob(json.loads(left.read_text())) == \
                _strip_knob(json.loads(right.read_text())), rel
        else:
            assert left.read_bytes() == right.read_bytes(), rel
        compared += 1
    assert compared > 10
    for name in ("human_review_queue.csv", "human_review_queue.json", "questions.json",
                 "questions.md"):
        assert (off / name).read_bytes() == (on / name).read_bytes(), name
    header = (on / "human_review_queue.csv").read_text().splitlines()[0]
    assert header.endswith("best_guess_rule,best_guess_entered")
    rows = json.loads((on / "human_review_queue.json").read_text())
    assert rows and all(r["best_guess_rule"] == "" and r["best_guess_entered"] == ""
                       for r in rows)


def test_the_queue_writer_tolerates_both_schemas():
    # the SHOULD-7 residual: a pre-change row (no guess keys) and a post-change row write
    # into ONE schema — the two trailing columns, empty where a row never carried them
    from canopy.pipeline.run import REVIEW_QUEUE_COLUMNS
    from canopy.report.tables import write_rows
    import tempfile

    old_shape = {"paper_id": "p", "dataset_id": "x:d1", "outcome_key": "late_adaptation",
                 "group": "A", "confidence": "needs_human", "route": "figure", "reason": "r",
                 "candidates": ""}
    new_shape = dict(old_shape, group="B", best_guess_rule="adjudicated_value",
                     best_guess_entered="B 3.13 ± 0.25 SE n 22")
    with tempfile.TemporaryDirectory() as scratch:
        paths = write_rows([old_shape, new_shape], Path(scratch) / "queue",
                           REVIEW_QUEUE_COLUMNS, formats=("csv", "json"))
        lines = paths["csv"].read_text().splitlines()
    assert lines[0].endswith("best_guess_rule,best_guess_entered")
    assert lines[1].endswith(",")                        # the old row: two empty cells
    assert lines[2].endswith("B 3.13 ± 0.25 SE n 22")


def test_a0_strict_rows_are_the_same_objects_even_while_the_tier_fires():
    va, vb, cands, adj = _settled_pair()
    strict = [EffectSizeRecord(dataset_id="s:d1", outcome_key="late_adaptation", es=0.5,
                               var=0.04, route="figure", higher_is_better=False,
                               confidence="auto_accept")]
    rows, dec = best_guess_rows(strict, [_held()], outcome=OUTCOME, settings=_settings(),
                                verdicts=[va, vb], candidates=cands, datasets=_datasets(),
                                retirements={}, adjudications=adj)
    assert rows[0] is strict[0] and dec[0].admitted
    assert len(rows) == 2                                # k_bg ≥ k_strict, always


def test_f2_is_absolute_no_reading_with_a_spread_means_no_guess():
    # Schabowsky's shape: a settled ruling mean and NOT ONE reading carrying a spread —
    # the ladder ends at F2 and the refusal says the record holds nothing borrowable
    va, vb, cands, adj = _settled_pair(a_disp=None, b_disp=None)
    cands = [c.model_copy(update={"dispersion_value": None}) for c in cands]
    sink = []
    _, dec = _run(_held(), [va, vb], cands, adjudications=adj, sink=sink)
    assert not any(d.admitted for d in dec)
    assert not decision_b_fired(dec, sink)


def test_the_caveat_is_the_constant_unfired_and_assembled_when_fired():
    from canopy.report.theme import BEST_GUESS_CAVEAT, best_guess_caveat

    assert best_guess_caveat() == BEST_GUESS_CAVEAT      # byte-identical when quiet
    fired = best_guess_caveat(fired=True, crossings=True, borrowed=True,
                              by_rule_totals={"adjudicated_value": 1})
    assert "entered at values already in the run's own record" in fired
    assert "every crossing is listed on the row" in fired
    assert "pull its effect toward null" in fired
    assert "adjudicated_value ×1" in fired


def test_t_b9_a_two_rule_row_joins_names_in_catalogue_order():
    # cell A: a cross-route disagreement with a clean text winner (b); cell B: the standing
    # value behind two model families (c); no adjudication anywhere — the row enters under
    # "route_precedence;agreed_solo_read", joined in catalogue order, never alphabetical
    va = _verdict("A", mean=20.0, disp=1.0, disp_type="UNKNOWN", n=10, agreement="disagree",
                  route="", adjudicated=False, tol=None)
    vb = _verdict("B", mean=15.0, disp=0.8, n=10, route="figure", adjudicated=False, tol=1.0)
    cands = [
        _cand("A", "x:d1:late_adaptation:A:text:opus#0", 20.0, 1.0, "SE", route="text",
              locator="Results"),
        _cand("A", "x:d1:late_adaptation:A:text:sonnet#0", 20.0, 1.0, "SE", route="text",
              locator="Results"),
        _cand("A", A_ENS, 22.0, 1.5, n=10),
        _cand("B", "x:d1:late_adaptation:B:digitize:readout:opus:direct", 15.0, 0.8, n=10,
              model="claude-opus-5"),
        _cand("B", "x:d1:late_adaptation:B:digitize:readout:sonnet:direct", 15.2, 0.8,
              n=10, model="claude-sonnet-5")]
    rows, dec = _run(_held(), [va, vb], cands)
    assert dec[0].admitted
    assert dec[0].rule == "route_precedence;agreed_solo_read"
    assert rows[-1].best_guess_rule == "route_precedence;agreed_solo_read"
    # …and a bare (a) fire never carries a joined name (asserted on the Coudière fixtures
    # above: rule == "adjudicated_value", no semicolon)


# ----------------------------------------------------------- review F1/F2/F3 regression tests
@needs_run
def test_f1_a_blocked_fire_projects_with_its_rule_the_doc_wording_and_only_its_own_row(tmp_path):
    # the reviewer's Kumar-only repool: d2 A fires cell-level (row blocked by d2 B), and the
    # projection must carry the RULE, use the cannot-enter wording, and leave the never-
    # guessed sibling row blank
    run = tmp_path / "kumar"
    (run / "papers" / "cfcf61c1c2b7").mkdir(parents=True)
    for name in ("map.json", "extract.json", "verify.json", "resolve.json", "ingest.json",
                 "file_id.txt"):
        source = CASE_STUDY / "papers" / "cfcf61c1c2b7" / name
        if source.exists():
            shutil.copy(source, run / "papers" / "cfcf61c1c2b7" / name)
    shutil.copy(CASE_STUDY / "protocol.yaml", run / "protocol.yaml")
    for name in ("prisma.json", "exclusions.json"):
        if (CASE_STUDY / name).exists():
            shutil.copy(CASE_STUDY / name, run / name)
    manifest = json.loads((CASE_STUDY / "manifest.json").read_text())
    manifest["papers"] = [p for p in manifest["papers"]
                          if p["paper_id"].startswith("cfcf61c1c2b7")]
    manifest["human_review_queue"] = []
    manifest["outputs"] = {}
    (run / "manifest.json").write_text(json.dumps(manifest))
    payload = _repool(run)
    bg = payload["best_guess"]
    assert bg["n_added"] == 0 and len(bg["cell_guesses"]) == 1
    assert bg["cell_guesses"][0]["rule"] == "route_precedence"
    rows = {(q["dataset_id"], q["group"]): q for q in _queue_rows(run)}
    fired = rows[("cfcf61c1c2b7:d2", "A")]
    assert fired["best_guess_rule"] == "route_precedence"
    assert fired["best_guess_entered"].startswith("A 26.386")
    assert "no authority" in fired["best_guess_blocked_by"]
    sibling = rows[("cfcf61c1c2b7:d2", "B")]
    assert sibling["best_guess_rule"] == "" and sibling["best_guess_entered"] == ""
    assert "best_guess_blocked_by" not in sibling
    d1 = rows[("cfcf61c1c2b7:d1", "B")]
    assert d1["best_guess_entered"] == ""                # A4-blocked is refused, not guessed
    questions = json.loads((run / "questions.json").read_text())
    notes = [(q.get("best_guess") or {}).get("note", "") for q in questions
             if q.get("dataset_id") == "cfcf61c1c2b7:d2"]
    assert any("a rule-answer is available" in n and "the row cannot enter" in n
               and "route_precedence" in n for n in notes)
    assert not any("temporarily entered" in n for n in notes)


def test_f2_rule_b_never_enters_a_mean_the_record_does_not_hold():
    # two text readings at 24.0 and 24.9 under the corpus's own 1.0 vote window: agreement
    # within the window is not ONE reading, and 24.45 exists nowhere in the record — refuse
    text = [_cand("A", "x:d1:late_adaptation:A:text:opus#0", 24.0, 1.0, "SE", route="text",
                  locator="Results"),
            _cand("A", "x:d1:late_adaptation:A:text:sonnet#0", 24.9, 1.0, "SE", route="text",
                  locator="Results")]
    figure = [_cand("A", A_ENS, 25.5, 1.5, n=13, locator="fig03 panel A")]
    va = _verdict("A", mean=None, agreement="disagree", route="", adjudicated=False, tol=1.0)
    ctx = bestguess._cell_context(outcome=OUTCOME, settings=_settings(), verdicts=[va],
                                  candidates=text + figure, datasets=_datasets(),
                                  retirements={}, adjudications={}, cell_guesses=None)
    guess = bestguess._try_route_precedence(ctx, va, text + figure, None, None)
    assert guess.rule == "" and "not one reading" in guess.refusal
    assert "24.45" not in guess.refusal                  # no averaged number anywhere
    # …and byte-identical readings still enter their own value (the Kumar witness shape)
    same = [text[0], text[0].model_copy(update={"candidate_id": text[0].candidate_id + "x"})]
    guess = bestguess._try_route_precedence(ctx, va, same + figure, None, None)
    assert guess.rule == "route_precedence" and guess.mean == 24.0


def test_f3_a_standing_unretired_finding_on_an_answered_cell_is_masked_and_recorded():
    # A5's second ground, non-trivially: cell A is human-landed and still carries a standing
    # locator_reads_conflict (in NO VALUE_CLEARS_* family) attached to its own reading; cell
    # B is guessed. The instance is masked from the row veto, recorded in
    # `masked_on_answered` — and in stepped_past NOWHERE (the two-list invariant).
    va, vb, cands, adj = _settled_pair()
    va.flags = [_flag("locator_reads_conflict", "warn", [A_ENS])]
    va.mean, va.dispersion_value, va.n = 3.95, 0.275, 22
    retirements = {("x:d1", "late_adaptation", "A"): _ret(
        answered=True, landed={"mean": 3.95, "dispersion_value": 0.275,
                               "dispersion_type": "SE", "seq": 78})}
    _, dec = _run(_held(flags=["locator_reads_conflict"]), [va, vb], cands,
                  adjudications=adj, retirements=retirements)
    decision = dec[0]
    assert decision.admitted and decision.rule == "adjudicated_value"
    masked = decision.masked_on_answered["A"]
    assert [m["finding"] for m in masked] == ["locator_reads_conflict"]
    assert masked[0]["note"].startswith("standing on a human-answered cell")
    assert "A" not in decision.stepped_past              # exactly one of the two lists
    assert "masked on the answered cell A: locator_reads_conflict" in decision.reason
    # …and the payload's added[] entry carries the non-empty list
    payload = best_guess_payload(None, None, dec, added_rows=[decision.row.model_copy(
        update={"best_guess_rule": decision.rule, "best_guess_reason": decision.reason})],
        loo_bg=[], settings=_settings(), fired=True)
    assert payload["added"][0]["masked_on_answered"]["A"]


@needs_run
def test_f3_the_run_path_log_gate_disables_the_tier_and_is_byte_identical_to_legacy(tmp_path):
    # integration §B: when a log exists, the run path's first pass evaluates with the tier
    # DISABLED — byte-identical decisions to the pre-tier engine, zero fires, no new keys —
    # and the repool that follows evaluates for real (T-B4 pins that half)
    from canopy.protocol import load_protocol
    from canopy.report.outputs import _best_guess_line

    run = _coudiere_run(tmp_path)
    protocol = load_protocol(run / "protocol.yaml")
    outcome = protocol.outcome("late_adaptation")
    verdicts, candidates, records, datasets = _load_paper(COUDIERE)
    held = [r for r in records if r.outcome_key == "late_adaptation"]
    off = _best_guess_line([], held, outcome, protocol.stats, run_dir=run,
                           verdicts=verdicts, candidates=candidates, datasets=datasets,
                           retirements={}, cell_rules_enabled=False)
    _, legacy = best_guess_rows([], held, outcome=outcome, settings=protocol.stats)
    assert off[5] is False and off[4] == []              # no fire, no cell guesses
    assert [(d.veto, d.reason) for d in off[1]] == [(d.veto, d.reason) for d in legacy]
    on = _best_guess_line([], held, outcome, protocol.stats, run_dir=run,
                          verdicts=verdicts, candidates=candidates, datasets=datasets,
                          retirements={}, cell_rules_enabled=True)
    assert on[5] is True and on[1][0].admitted


def test_f3_a_shared_control_row_is_refused_never_mis_split():
    va, vb, cands, adj = _settled_pair()
    datasets = {"x:d1": DatasetSpec(dataset_id="x:d1", group_a=GroupSpec(label="Old"),
                                    group_b=GroupSpec(label="Young"), shared_control=True)}
    sink: list[dict] = []
    _, dec = best_guess_rows([], [_held()], outcome=OUTCOME, settings=_settings(),
                             verdicts=[va, vb], candidates=cands, datasets=datasets,
                             retirements={}, adjudications=adj, cell_guesses=sink)
    assert not any(d.admitted for d in dec)
    assert sink and "control arm" in sink[0]["row_blocked_by"]
