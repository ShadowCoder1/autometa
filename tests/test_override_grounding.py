"""Where a reviewer's own number comes from, and what it carries once it lands.

Three failures an adversarial review measured on `pipeline/overrides.py`, each pinned here:

1. the one decision in the log that changes a pooled effect size — a `value` answer's mean — had no
   slot for its provenance, so nothing recorded where the number came from and nothing could check
   it. `eligibility`, `orientation`, `include_dataset` and `group_n` all carry a `quote`; the
   statistic did not;
2. after `_apply_value` replaced an arm's mean, the run's vote, checks and confidence score still
   described the number that had been replaced — so the cell was held on "the score is below the
   acceptance line" and released by `overrules: ["low_score"]`, the lever built for DISAGREEING
   with a score that exists. Worst case measured: one reviewer-originated row at d = 1.5, n = 60/60
   among five d = 0.2 studies took 31.9% of the weight and moved the pooled estimate from 0.20 to
   0.615, with every artefact reading normal;
3. an analysed-n answer about a MAPPED dataset whose cells the run never read was recorded on the
   state every later rebuild uses and then reported `pending` for ever.

Every test reads the committed `runs/nine` fixture, whose two papers with an `ingest/` record carry
the real page text — so "is this quote printed in the paper?" is asked of a real paper.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

#: a sentence Vachon (`b7523a41b03a`) really prints, on page 4 — `ground_quote` scores it 1.0
PRINTED = ("We excluded 4 younger (all from the non-instructed group) and 3 older "
           "(1 non-instructed, 2 instructed) participants.")
#: …and one it does not. Same shape, same register, invented numbers: the false witness a typed
#: number can stand on, which is the whole case for checking the quote at all.
NOT_PRINTED = ("We excluded 9 younger (all from the instructed group) and 11 older "
               "(4 instructed, 7 non-instructed) volunteers.")


def _nine(tmp_path: Path) -> Path:
    """A writable copy of the fixture run with its own recorded answer cleared."""
    from tests.helpers import nine

    run = nine.copy_to(tmp_path)
    (run / "overrides.jsonl").unlink(missing_ok=True)
    return run


def _repool(run: Path) -> dict[str, Any]:
    from canopy.pipeline.overrides import apply_overrides_and_repool

    return apply_overrides_and_repool(run)


def _queue(run: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The review queue the last re-pool wrote, by cell. A released cell is simply not in it."""
    rows = json.loads((run / "human_review_queue.json").read_text(encoding="utf-8"))
    return {(r["dataset_id"], r["outcome_key"], str(r["group"])): r for r in rows}


def _rows(run: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads((run / "results" / "extraction_table_all.json").read_text(encoding="utf-8"))
    return {(r["dataset_id"], r["outcome_key"]): r for r in rows}


# ============================================================ fix 1: a number with no provenance
def test_a_typed_mean_whose_quote_the_paper_does_not_contain_is_refused(tmp_path):
    """The case the review argued must be refused outright: a number with no witness at all.

    The record names words the paper never printed, and no reading in the run holds the number
    either — so there is nothing behind it but the reviewer's assertion. Before this, `_validate`'s
    `value` branch had no `quote` field to refuse: the mean was written, applied, pooled, and the
    only trace of where it came from was free prose nobody compared to anything.
    """
    from canopy.pipeline.overrides import OverrideRejected, append_override, read_overrides

    run = _nine(tmp_path)
    answer = {"kind": "value", "paper_id": "b7523a41b03a", "dataset_id": "b7523a41b03a:d1",
              "outcome_key": "late_adaptation", "group": "A", "mean": 44.4,
              "dispersion_value": 2.0, "dispersion_type": "SD", "n": 18,
              "quote": NOT_PRINTED,
              "justification": "read off the sentence I have quoted"}
    with pytest.raises(OverrideRejected) as refusal:
        append_override(run, answer)
    assert "not printed in the paper" in str(refusal.value)
    assert "44.4" in str(refusal.value), "the refusal must name the number it is about"
    assert read_overrides(run) == [], "a refused answer leaves nothing in the log"


def test_a_typed_mean_grounded_in_the_paper_records_the_page_it_was_found_on(tmp_path):
    """…and the same answer with the paper's own words is accepted, with the check on the record.

    `ground_quote` is the extract stage's own function, so a reviewer's quote is held to exactly
    the standard a model's is: the same normalisation, the same 0.95 threshold, the same page sweep.
    """
    from canopy.pipeline.overrides import VALUE_GROUNDED, append_override

    run = _nine(tmp_path)
    record = append_override(run, {
        "kind": "value", "paper_id": "b7523a41b03a", "dataset_id": "b7523a41b03a:d1",
        "outcome_key": "late_adaptation", "group": "A", "mean": 44.4,
        "quote": PRINTED, "justification": "the arm's size and value from this sentence"})
    assert record["quote"] == PRINTED
    assert record["grounding"]["state"] == VALUE_GROUNDED
    assert record["grounding"]["page"] == 4, "the page the quote was found on is on the record"
    assert record["grounding"]["similarity"] == 1.0
    # …and the record does not pretend the sentence holds the number. It is the exclusions sentence,
    # which is really printed and really does not contain 44.4: grounded provenance and provenance
    # that names the value are two different things, and a reader can see which this is. Recorded
    # rather than refused, because a reviewer may legitimately quote the sentence a table row
    # belongs to, or the printed value they converted this one from.
    assert record["grounding"]["quote_holds_the_number"] is False


def test_a_figure_read_no_quote_can_ground_is_recorded_ungrounded_and_named(tmp_path):
    """The case REFUSE would have broken, and why the rule stops short of it.

    Vachon's own means are in its figures — twenty pages of ingested text hold no sentence printing
    them, which is why the run digitized the plot (its cells' route is `figure`, off a `digitize`
    candidate). A reviewer who reads that plot has no quotable words for the
    number: the caption grounds beautifully and says nothing whatever about the value, so demanding
    one would buy assurance that is not there. The answer is recorded instead, with the state that
    says only the reviewer witnesses it, and the re-pool names it in `ungrounded` — where every
    number in the analysis that the paper was never asked about can be read off one list.
    """
    from canopy.pipeline.overrides import VALUE_UNGROUNDED, append_override

    run = _nine(tmp_path)
    record = append_override(run, {
        "kind": "value", "paper_id": "b7523a41b03a", "dataset_id": "b7523a41b03a:d1",
        "outcome_key": "late_adaptation", "group": "A", "mean": 44.4,
        "justification": "measured off Fig 2A against its own 0-60 ladder"})
    assert record["grounding"]["state"] == VALUE_UNGROUNDED

    summary = _repool(run)
    named = [entry for entry in summary["ungrounded"] if entry["seq"] == record["seq"]]
    assert len(named) == 1, summary["ungrounded"]
    assert named[0]["mean"] == 44.4 and named[0]["state"] == VALUE_UNGROUNDED
    assert named[0]["dataset_id"] == "b7523a41b03a:d1" and named[0]["group"] == "A"


def test_a_mean_the_run_itself_read_needs_no_quote(tmp_path):
    """…and neither does picking one of the run's own readings, which is the commonest answer.

    "That series is this group", "this is the right ladder", "keep the value you resolved": the
    number in each is a reading the extract stage took and `ground_candidate` already checked. The
    witness is the run's, not the reviewer's, so the record says so rather than demanding words for
    a number that is not new.
    """
    from canopy.pipeline.overrides import VALUE_READING_CONFIRMED, append_override

    run = _nine(tmp_path)
    resolved = 27.9                       # what the run resolved for this cell (verify.json)
    record = append_override(run, {
        "kind": "value", "paper_id": "b7523a41b03a", "dataset_id": "b7523a41b03a:d1",
        "outcome_key": "late_adaptation", "group": "A", "mean": resolved,
        "justification": "the value this run resolved is the one the panel shows"})
    assert record["grounding"]["state"] == VALUE_READING_CONFIRMED
    assert record["grounding"]["reading"] == pytest.approx(resolved)


def test_a_value_record_from_before_the_check_still_applies_and_is_named_as_never_grounded(tmp_path):
    """The migration, both halves of it.

    `runs/et_with_dbs` holds 24 `value` records and `runs/attention` 19, every one of them written
    before a reviewer's number was checked against anything, and both runs must keep re-pooling
    exactly as they did. So the requirement is on NEWLY VALIDATED records only — and the log is
    append-only, so a grandfathered record cannot be amended to say it was never grounded. The
    missing `grounding` field IS the marker, and the re-pool names every such number in
    `ungrounded`, so the grandfathering is visible rather than silent.
    """
    from canopy.pipeline.overrides import VALUE_NEVER_CHECKED, read_overrides

    run = _nine(tmp_path)
    old = {"kind": "value", "paper_id": "b7523a41b03a", "dataset_id": "b7523a41b03a:d1",
           "outcome_key": "late_adaptation", "group": "A", "mean": 44.4,
           "dispersion_value": 2.0, "dispersion_type": "SD", "n": 18,
           "justification": "a record the old log would have written", "seq": 1,
           "at": "2026-01-01T00:00:00+00:00", "actor": "a reviewer"}
    (run / "overrides.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    assert "grounding" not in read_overrides(run)[0], "the fixture must be a pre-check record"

    summary = _repool(run)
    assert summary["applied"] == 1 and not summary["pending"], summary["pending"]
    assert _rows(run)[("b7523a41b03a:d1", "late_adaptation")]["mean_a"] == 44.4, \
        "a grandfathered record still replaces the number"
    named = [entry for entry in summary["ungrounded"] if entry["seq"] == 1]
    assert len(named) == 1 and named[0]["state"] == VALUE_NEVER_CHECKED, summary["ungrounded"]


def test_a_quote_the_run_kept_no_page_text_for_is_recorded_unchecked_not_refused(tmp_path):
    """A check that cannot read the paper has nothing to say — and must never say the paper
    contradicts a quote it never saw.

    Four of the fixture's six papers carry an `ingest/` record with no page files, which is the
    ordinary state of an archived run: the text is the largest thing in a run directory and the
    first thing to be left behind. The precedent is `run._page_texts` ("a check that cannot read
    the paper has nothing to say, and that is not an error in the run") and `check_table_cell`'s
    `(None, reason)`. The state is recorded so a reader can see the difference between "checked"
    and "could not be checked".
    """
    from canopy.pipeline.overrides import VALUE_NO_TEXT, append_override

    run = _nine(tmp_path)
    record = append_override(run, {
        "kind": "value", "paper_id": "5039533c85ef", "dataset_id": "5039533c85ef:d1",
        "outcome_key": "late_adaptation", "group": "A", "mean": 41.4,
        "quote": "a sentence this run no longer keeps the pages to check",
        "justification": "read from the Results paragraph I have quoted"})
    assert record["grounding"]["state"] == VALUE_NO_TEXT
    assert "no ingested page text" in record["grounding"]["why"]
    assert [entry["state"] for entry in _repool(run)["ungrounded"]] == [VALUE_NO_TEXT]


# ====================================================== fix 2: no score is not a low score
def test_a_human_replaced_arm_is_not_released_by_overruling_low_score(tmp_path):
    """THE finding. Cressman's late-adaptation cells are held by a score of 0.29 and nothing else
    a reviewer cannot name: no refutation, no adjudication, no error, no contradiction.

    Before this, one answer replaced both arms' means and said `overrules: ["low_score"]`, and the
    cells were released — by disagreeing with a score that had been computed on the numbers the
    same answer had just thrown away. The score is now struck with the reading it described
    (`_void_verification`), the arm has NO score, and `low_score` cannot overrule the absence of
    the thing it was built to argue with.
    """
    from canopy.pipeline.overrides import append_override, recorded_holds

    run = _nine(tmp_path)
    cell = ("5039533c85ef:d1", "late_adaptation")
    assert recorded_holds(run, {"dataset_id": cell[0], "outcome_key": cell[1], "group": "A",
                                "paper_id": "5039533c85ef"}) >= {"low_score"}
    for group, mean in (("A", 41.4), ("B", 43.4)):
        append_override(run, {
            "kind": "value", "paper_id": "5039533c85ef", "dataset_id": cell[0],
            "outcome_key": cell[1], "group": group, "mean": mean,
            "dispersion_value": 3.0, "dispersion_type": "SD", "n": 20,
            "overrules": ["low_score"],
            "justification": "I have read the score and I disagree with it"})
    _repool(run)

    queue = _queue(run)
    for group in ("A", "B"):
        entry = queue.get((*cell, group))
        assert entry is not None, "a human-replaced arm with no verification left the queue"
        assert entry["confidence"] == "needs_human"
        assert entry["confidence_score"] is None, "a voided score is absent, not zero"
        assert "no score" in entry["reason"]
    assert _rows(run)[cell]["confidence"] == "needs_human"
    assert _rows(run)[cell]["primary_row"] is False, "a row nothing verified was pooled"


def test_the_answer_a_human_replaced_arm_does_have_is_offered_and_accepted(tmp_path):
    """…and the cell is not stranded: the one answer to it is nameable and the page offers it.

    Held with nothing to answer is the failure §C4 exists to remove, and it is what a new hold costs
    if only the analysis knows about it. `recorded_holds` names the finding off the LOG — the stage
    verdict still carries the score, because the voiding lives in the re-pool's working copies — and
    the page reads the same registry through the same rule, so the card that appears offers exactly
    the finding the analysis is enforcing.
    """
    from canopy.pipeline.overrides import NO_VERIFICATION, append_override, recorded_holds
    from canopy.review.questions import questions_for_run

    run = _nine(tmp_path)
    cell = ("5039533c85ef:d1", "late_adaptation")
    for group, mean in (("A", 41.4), ("B", 43.4)):
        append_override(run, {
            "kind": "value", "paper_id": "5039533c85ef", "dataset_id": cell[0],
            "outcome_key": cell[1], "group": group, "mean": mean,
            "dispersion_value": 3.0, "dispersion_type": "SD", "n": 20,
            "overrules": ["low_score"],
            "justification": "I have read the score and I disagree with it"})
    _repool(run)

    ref = {"dataset_id": cell[0], "outcome_key": cell[1], "group": "A",
           "paper_id": "5039533c85ef"}
    assert NO_VERIFICATION in recorded_holds(run, ref), "nothing could name the new hold"
    cards = [q for q in questions_for_run(run)
             if q["dataset_id"] == cell[0] and q["outcome_key"] == cell[1]]
    assert cards, "a held cell with no card is the failure C4 exists to remove"
    # a §C1 card folds two cells, so its options hang off its slots; a per-cell card carries its own
    offered = {name for card in cards
               for holder in (card, *(card.get("slots") or []))
               for option in (holder.get("options") or [])
               for name in (option.get("overrules") or [])}
    assert NO_VERIFICATION in offered, [card["kind"] for card in cards]
    assert "low_score" not in offered, "the page must not offer a lever the analysis ignores"

    append_override(run, {
        "kind": "mark_reviewed", "paper_id": "5039533c85ef", "dataset_id": cell[0],
        "outcome_key": cell[1], "confidence": "accept_with_note",
        "overrules": [NO_VERIFICATION],
        "justification": "I have read the figure and I accept both numbers as they now stand"})
    _repool(run)
    assert _rows(run)[cell]["confidence"] == "accept_with_note"
    assert (*cell, "A") not in _queue(run), "the answered cell is still being asked"


def test_no_score_and_a_low_score_are_answered_by_different_records():
    """The rule itself, on one arm, with nothing else in the way.

    `low_score` keeps its real purpose — a cell that WAS verified and scored below the line — and
    `no_verification` is the one answer to a cell whose score was struck with the number it
    described. Neither stands in for the other, which is the whole of the fix: an answer retires
    what it names and nothing else (§C4).
    """
    from canopy.models import Verdict
    from canopy.pipeline.overrides import (NO_VERIFICATION, _derived_bucket, _void_verification)

    scored = Verdict(dataset_id="d1", outcome_key="late_adaptation", group="A",
                     confidence="needs_human", needs_human=True, mean=10.0,
                     higher_is_better=True, agreement="single", confidence_score=0.29,
                     confidence_reasons=["one reader, grounded"])
    assert _derived_bucket(scored, mean_answered=True, overrules=()) == "needs_human"
    assert _derived_bucket(scored, mean_answered=True,
                           overrules={"low_score"}) == "accept_with_note"

    _void_verification(scored, 10.0)
    assert scored.confidence_score is None and scored.confidence_margin is None
    assert _derived_bucket(scored, mean_answered=True, overrules={"low_score"}) == "needs_human"
    assert _derived_bucket(scored, mean_answered=True,
                           overrules={NO_VERIFICATION}) == "accept_with_note"


def test_a_reading_that_agrees_with_the_human_keeps_its_provenance(tmp_path):
    """Voiding is not vandalism: the candidate that READ the number a human supplied is still
    where that number came from, and the extraction table builds `quote_a`, `page_a` and the figure
    crop out of exactly that list. A candidate that read something else is the provenance of a
    different number and goes with the score.
    """
    from canopy.models import Candidate, Verdict
    from canopy.pipeline.overrides import _void_verification

    agrees = Candidate(candidate_id="agrees", kind="group_stats", dataset_id="d1",
                       outcome_key="late_adaptation", group="A", mean=10.0)
    differs = Candidate(candidate_id="differs", kind="group_stats", dataset_id="d1",
                        outcome_key="late_adaptation", group="A", mean=14.0)
    verdict = Verdict(dataset_id="d1", outcome_key="late_adaptation", group="A", mean=10.0,
                      candidate_ids=["agrees", "differs"], confidence_score=0.29)
    _void_verification(verdict, 10.0, [agrees, differs])
    assert verdict.candidate_ids == ["agrees"]


# ============================================ fix 3: an answer that landed, reported as pending
def _forget_rows(run: Path, paper12: str, dataset_id: str) -> None:
    """Take every row of one dataset out of the resolve stage file — the shape a MAPPED dataset has
    when the extract stage never reached it (Roller's E1a, Wolpe's second experiment)."""
    path = run / "papers" / paper12 / "resolve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"] = [r for r in payload.get("records") or []
                          if r.get("dataset_id") != dataset_id]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_an_analysed_n_for_a_mapped_dataset_with_no_rows_is_applied_not_pending(tmp_path):
    """The answer landed: `_apply_analysed_n` wrote it onto the state every later rebuild reads,
    and then returned the refusal because it had touched no row.

    So a reviewer was told for ever that a decision on the record had not happened — and, because
    `HumanLanded` reads the summary's `pending` list to decide what has LANDED, the same line
    withheld hold-retirement from a size that was in fact being applied. The refusal for a dataset
    nobody mapped is a different statement and still stands.
    """
    from canopy.pipeline.overrides import append_override

    run = _nine(tmp_path)
    _forget_rows(run, "b7523a41b03a", "b7523a41b03a:d1")
    append_override(run, {"kind": "group_n", "paper_id": "b7523a41b03a",
                          "dataset_id": "b7523a41b03a:d1", "n_a": 18, "n_b": 16,
                          "quote": PRINTED,
                          "justification": "the analysed n, after the stated exclusions"})
    summary = _repool(run)
    assert summary["applied"] == 1, summary["pending"]
    assert not summary["pending"], summary["pending"]

    # …and an id this run never mapped is still a mistake, not an answer
    append_override(run, {"kind": "group_n", "dataset_id": "nobody:d9", "n_a": 3, "n_b": 4,
                          "justification": "a dataset this run never mapped"})
    summary = _repool(run)
    assert len(summary["pending"]) == 1 and "nobody:d9" in summary["pending"][0]["why"]


def test_an_analysed_n_that_touched_no_row_is_the_n_a_later_value_answer_builds_with(tmp_path):
    """…which is what makes reporting it applied true rather than merely kinder.

    The sizes go onto `_RunState` before anything is rebuilt, so the row a typed pair of values
    makes for a cell the run never read is divided by the answered n and not by the recruited one.
    """
    from canopy.pipeline.overrides import append_override

    run = _nine(tmp_path)
    _forget_rows(run, "b7523a41b03a", "b7523a41b03a:d1")
    append_override(run, {"kind": "group_n", "paper_id": "b7523a41b03a",
                          "dataset_id": "b7523a41b03a:d1", "n_a": 18, "n_b": 16,
                          "quote": PRINTED,
                          "justification": "the analysed n, after the stated exclusions"})
    for group, mean in (("A", 27.9), ("B", 27.35)):
        append_override(run, {
            "kind": "value", "paper_id": "b7523a41b03a", "dataset_id": "b7523a41b03a:d1",
            "outcome_key": "late_adaptation", "group": group, "mean": mean,
            "dispersion_value": 4.0, "dispersion_type": "SD",
            "justification": f"the value this run resolved for group {group}"})
    summary = _repool(run)
    assert not summary["pending"], summary["pending"]
    row = _rows(run)[("b7523a41b03a:d1", "late_adaptation")]
    assert (row["n_a"], row["n_b"]) == (18, 16), row
