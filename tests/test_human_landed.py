"""Ticket 1/4's shared primitive: the registry of human-landed values, the read tolerance, and
the one stale-hold rule the page and the apply loop both consult.

The registry is what lets three blind spots see one fact — the extract stage (an automated
re-read must not displace a human's number), the questions page (a card must not re-ask a
decision the log records) and the apply loop (a hold whose subject a human displaced is
released, not re-litigated). Everything here is a fold over raw log lines, so the tests write
raw lines, exactly as the loop reads them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canopy.pipeline.overrides import (READ_TOLERANCE_REL, append_override,
                                       apply_overrides_and_repool, human_landed_values,
                                       stale_hold_names, within_read_tolerance)
from tests.helpers import nine


def _raw(run: Path, **record) -> None:
    """One raw log line, the way an old or foreign writer would leave it — no validation."""
    path = run / "overrides.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    record.setdefault("seq", len([x for x in lines if x.strip()]) + 1)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# ----------------------------------------------------------------------------- the registry
def test_empty_run_has_an_empty_registry(tmp_path):
    landed = human_landed_values(tmp_path)
    assert landed.cells == {} and landed.dataset_n == {}
    assert landed.for_cell("x:d1", "late_adaptation", "A") is None


def test_fields_fold_sparsely_and_latest_wins(tmp_path):
    _raw(tmp_path, kind="value", dataset_id="x:d1", outcome_key="late", group="A",
         mean=78.5, dispersion_value=17.0, dispersion_type="CI95", n=8, justification="first")
    _raw(tmp_path, kind="value", dataset_id="x:d1", outcome_key="late", group="A",
         dispersion_type="SE", justification="type only — the mean must survive this")
    cell = human_landed_values(tmp_path).for_cell("x:d1", "late", "A")
    assert cell["mean"] == 78.5 and cell["n"] == 8
    assert cell["dispersion_type"] == "SE"
    assert cell["field_seqs"]["mean"] == 1 and cell["field_seqs"]["dispersion_type"] == 2


def test_a_group_less_raw_value_record_protects_both_groups(tmp_path):
    # nothing this codebase writes, but old and foreign logs may — and the raw scan this
    # registry replaced honoured them, so the registry keeps that width
    _raw(tmp_path, kind="value", dataset_id="x:d1", outcome_key="late", mean=5.0,
         justification="hand-written line with no group")
    landed = human_landed_values(tmp_path)
    assert landed.for_cell("x:d1", "late", "A")["mean"] == 5.0
    assert landed.for_cell("x:d1", "late", "B")["mean"] == 5.0


def test_unknown_dispersion_type_and_malformed_numbers_contribute_nothing(tmp_path):
    _raw(tmp_path, kind="value", dataset_id="x:d1", outcome_key="late", group="A",
         mean="12,5", dispersion_type="UNKNOWN", justification="page-copied UNKNOWN, bad mean")
    cell = human_landed_values(tmp_path).for_cell("x:d1", "late", "A")
    # the record contributes nothing usable, but the fold never raises
    assert cell is None or (cell["mean"] is None and cell["dispersion_type"] == "")


def test_mark_reviewed_contributes_overrules_and_never_a_value(tmp_path):
    _raw(tmp_path, kind="mark_reviewed", dataset_id="x:d1", outcome_key="late",
         overrules=["verifier_refuted"], justification="a hold is not a number")
    landed = human_landed_values(tmp_path)
    cell = landed.for_cell("x:d1", "late", "A")
    assert cell["mean"] is None and "verifier_refuted" in cell["overrules"]
    assert landed.for_cell("x:d1", "late", "B")["overrules"] == {"verifier_refuted"}


def test_group_n_fills_only_when_no_later_per_cell_n(tmp_path):
    _raw(tmp_path, kind="value", dataset_id="x:d1", outcome_key="late", group="A",
         mean=1.0, n=6, justification="per-cell n first")                      # seq 1
    _raw(tmp_path, kind="group_n", dataset_id="x:d1", n_a=16, n_b=12,
         justification="analysed sizes")                                      # seq 2
    landed = human_landed_values(tmp_path)
    # the later dataset-level answer supplies A's n (the answered size, pre-split)
    assert landed.for_cell("x:d1", "late", "A")["n"] == 16
    assert landed.for_cell("x:d1", "late", "B")["n"] == 12
    _raw(tmp_path, kind="value", dataset_id="x:d1", outcome_key="late", group="A",
         n=7, justification="later per-cell n wins")                          # seq 3
    assert human_landed_values(tmp_path).for_cell("x:d1", "late", "A")["n"] == 7


def test_stated_versus_landed(tmp_path):
    _raw(tmp_path, kind="value", dataset_id="x:d1", outcome_key="late", group="A",
         mean=3.13, justification="typed")
    # no summary yet: nothing says the answer waits, so it counts as landed (the page's rule)
    assert human_landed_values(tmp_path).for_cell("x:d1", "late", "A")["mean_landed"]
    (tmp_path / "overrides_applied.json").write_text(json.dumps(
        {"applied": 0, "pending": [{"seq": 1, "why": "no row for this in this run"}],
         "excluded": [], "outcomes": {}}))
    cell = human_landed_values(tmp_path).for_cell("x:d1", "late", "A")
    assert cell["mean"] == 3.13, "a pending number is still STATED — protection holds"
    assert not cell["mean_landed"], "…but it LANDED nothing, so no hold may be retired on it"


# ----------------------------------------------------------------------------- the tolerance
def test_read_tolerance_draws_the_codebase_s_own_two_lines():
    assert within_read_tolerance(31.3, 31.4)                 # 0.32% — the option-dedupe line
    assert not within_read_tolerance(31.3, 31.78)            # 1.5% — beyond it, no SD in hand
    assert within_read_tolerance(31.3, 31.78, sd=6.5)        # 0.48 < 0.1 × 6.5 — fix H's line
    assert not within_read_tolerance(None, 31.4)
    assert not within_read_tolerance(31.4, None)


def test_the_two_tolerance_constants_are_pinned_to_their_precedents():
    from canopy.verify.confidence import DELTA_D_LIMIT
    from canopy.verify.vote import NEGLIGIBLE_D

    assert READ_TOLERANCE_REL == 0.005, "the option-dedupe line `_value_options` draws"
    assert NEGLIGIBLE_D == DELTA_D_LIMIT, "fix H's SD line stays the confidence module's"


# ----------------------------------------------------------------------------- the one rule
def _names(**over) -> set[str]:
    base = dict(stage_mean=21.05, verifier_verdict="refuted", adjudicated=False,
                alt_means=[86.0], alt_quote_cites_number=True, disputes_flag_present=False,
                human_mean=86.0, sd=None)
    base.update(over)
    return stale_hold_names(**base)


def test_displaced_and_adopted_refutations_retire():
    assert _names() == {"verifier_refuted"}                       # adopted the alt, displaced too
    assert _names(alt_means=[90.0]) == {"verifier_refuted"}       # displaced only


def test_the_narrowing_keeps_live_objections():
    assert _names(alt_means=[]) == set(), "an identity objection has no structured target"
    assert _names(alt_quote_cites_number=False) == set()
    assert _names(disputes_flag_present=True) == set(), \
        "evidence that POSTDATES the human's number is a new objection, not a displaced one"
    assert _names(verifier_verdict="confirmed") == set()
    assert _names(human_mean=None) == set()
    assert _names(stage_mean=86.0, alt_means=[21.05]) == set(), \
        "the human's number IS the one the verifier judged — the objection is live"
    assert _names(stage_mean=18.9, alt_means=[18.9], human_mean=18.9) == set(), \
        "an alternative that AGREES with the reading is not a numeric objection (fix G's guard)"


def test_an_adjudication_retires_only_beside_its_refutation():
    assert _names(adjudicated=True) == {"verifier_refuted", "adjudicated"}
    assert _names(adjudicated=True, alt_means=[]) == set(), "a lone adjudication keeps its card"


# ----------------------------------------------------- page and analysis release in lockstep
@pytest.fixture()
def nine_tmp(tmp_path) -> Path:
    return nine.copy_to(tmp_path)


def test_adopting_the_verifier_s_value_releases_cell_page_and_analysis_together(nine_tmp):
    """The Carroll shape on real records: 3570:d2 aftereffect A was refuted (stage −1.719)
    against a printed −8.0. A reviewer lands −8.0 — the verifier's own number. The repool
    retires the refutation (and the adjudication convened for it), says so in the summary, and
    the page stops asking `verifier_refuted` for that group without dropping the cell's card."""
    from canopy.review.questions import questions_for_run

    dataset_id, outcome_key = "3570e4ce2a9c:d2", "aftereffect"
    record = append_override(nine_tmp, {
        "kind": "value", "dataset_id": dataset_id, "outcome_key": outcome_key, "group": "A",
        "mean": -8.0, "dispersion_value": 1.2, "dispersion_type": "SE", "n": 10,
        "justification": "the verifier's printed −8.0 is the value; adopted"})
    summary = apply_overrides_and_repool(nine_tmp)
    mine = [entry for entry in summary.get("auto_resolved") or []
            if entry["dataset_id"] == dataset_id and entry["group"] == "A"]
    assert {entry["name"] for entry in mine} == {"verifier_refuted", "adjudicated"}
    assert all(entry["from_seq"] == record["seq"] for entry in mine)
    kinds = {q["group"]: q["kind"] for q in questions_for_run(nine_tmp, fold=False)
             if q["dataset_id"] == dataset_id and q["outcome_key"] == outcome_key}
    assert kinds.get("A") != "verifier_refuted", \
        "the page must not re-ask an objection the record has answered"


def test_an_identity_refutation_is_not_retired_by_a_landed_number(nine_tmp):
    """3570:d1 late A's refutation carries NO alt_mean (an identity objection): a typed value
    does not displace its subject, so both the analysis and the page keep it."""
    from canopy.review.questions import questions_for_run

    dataset_id, outcome_key = "3570e4ce2a9c:d1", "late_adaptation"
    append_override(nine_tmp, {
        "kind": "value", "dataset_id": dataset_id, "outcome_key": outcome_key, "group": "A",
        "mean": 12.0, "dispersion_value": 1.0, "dispersion_type": "SE", "n": 10,
        "justification": "a number, which answers nothing about identity"})
    summary = apply_overrides_and_repool(nine_tmp)
    assert not [entry for entry in summary.get("auto_resolved") or []
                if entry["dataset_id"] == dataset_id and entry["group"] == "A"]
    kinds = [q["kind"] for q in questions_for_run(nine_tmp, fold=False)
             if q["dataset_id"] == dataset_id and q["outcome_key"] == outcome_key
             and q["group"] == "A"]
    assert kinds, "the cell keeps a card — its objection is live"
