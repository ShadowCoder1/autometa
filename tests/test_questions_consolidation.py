"""One decision = one card: the questions page after DECISION §C1/C3/C4.

The page used to ask a cell at a time. A reviewer looking at a figure does not decide "group A's
number" and then, as a separate act, "group B's number" — they read the picture once and settle
the row; and the direction of a measure is decided once for the paper, not once per cell that used
it. So the per-cell questions are still what the page is BUILT from, and what it SHOWS is the
decision: a dataset's pair, a measure's direction, a paper's eligibility, the precedence override,
the analysed n.

Everything here runs against `tests/fixtures/runs/nine` — real records from a real run. Two
findings the fixture predates (Task 4's `precedence_override`, Task 5's `n_before_exclusions`) are
injected onto a COPY through `helpers.nine`, in the shape their producing code writes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from canopy.review.questions import answers_to_overrides, questions_for_run
from tests.helpers import nine


@pytest.fixture()
def nine_tmp(tmp_path) -> Path:
    return nine.copy_to(tmp_path)


def _append(run: Path, records: list[dict[str, Any]]) -> None:
    from canopy.pipeline.overrides import append_override

    for record in records:
        append_override(run, record)


def _repool(run: Path) -> dict[str, Any]:
    from canopy.pipeline.overrides import apply_overrides_and_repool

    return apply_overrides_and_repool(run)


# ----------------------------------------------------------------- C1: the folds
def test_orientation_is_one_card_per_paper_outcome_measure(nine_tmp):
    """The direction of a measure is `_apply_orientation`'s own scope: one answer settles every
    dataset of that paper and outcome carrying the measure. Asking it once per cell asked the same
    question eight times and offered eight chances to answer it differently."""
    qs = [q for q in questions_for_run(nine_tmp) if q["kind"] == "orientation"]
    assert len(qs) == 4 and all(q["scope"] == "measure" and q["id"].split("|")[2] == ""
                                for q in qs)
    assert {m for q in qs for m in q["member_ids"]} == {
        q["id"] for q in json.load(open(nine_tmp / "questions.json")) if q["kind"] == "orientation"}
    # …and the card lists every cell it settles, so a reviewer can see the reach of the answer
    assert sum(len(q["cells"]) for q in qs) == 8
    for card in qs:
        assert card["answer_writes"] == "orientation"
        assert [o["key"] for o in card["options"]] == ["higher_is_more", "lower_is_more"]


def test_the_measure_card_shows_the_outcome_definition_and_every_cell_s_raw_means(nine_tmp):
    """A direction cannot be decided against a key: "aftereffect" is a name, and what the review
    scores is the protocol's definition. The cells' own means are the other half — which way this
    paper's numbers actually run — and both readers' ballots are already under `why`."""
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "b511dbb76fa6:m62a56549|aftereffect||orientation")
    assert "Performance after removal of the perturbation and feedback" in card["prompt"]
    for group in ("A", "B"):
        verdict = next(v for v in json.loads(
            (nine_tmp / "papers" / "b511dbb76fa6" / "verify.json").read_text())["verdicts"]
            if v["dataset_id"] == "b511dbb76fa6:d1" and v["outcome_key"] == "aftereffect"
            and v["group"] == group)
        assert f"{verdict['mean']:.4g}".rstrip("0").rstrip(".")[:4] in card["prompt"] \
            or str(round(verdict["mean"], 2)) in card["prompt"], verdict["mean"]
    assert "claude" in card["why"], "the two readers' ballots travel with the card"


def test_pair_card_id_is_kind_free_and_stable_across_a_slot_answer(nine_tmp):
    """The pair's id names the DATASET's decision, never one slot's kind — a slot whose kind
    changes when its own answer lands must not change the id of the card it sits in."""
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "5039533c85ef:d1|late_adaptation||pair")
    assert {s["kind"] for s in card["slots"]} == {"which_value", "confirm_value"}
    recs = answers_to_overrides(card, {"option": card["options"][0]["key"],
                                       "option_fingerprint": card["options"][0]["fingerprint"]})
    _append(nine_tmp, recs[:1])                       # answer group A's slot only
    _repool(nine_tmp)
    after = questions_for_run(nine_tmp)
    assert any(q["id"] == "5039533c85ef:d1|late_adaptation||pair" for q in after) \
        or not any(q["id"].startswith("5039533c85ef:d1|late_adaptation|") for q in after)


def test_pair_card_writes_one_override_per_group(nine_tmp):
    """"Answers clear only what they name": the combination is one decision and two records, each
    naming its own group and carrying only its own option's `clears`."""
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "5039533c85ef:d1|late_adaptation||pair")
    recs = answers_to_overrides(card, {"option": card["options"][0]["key"],
                                       "option_fingerprint": card["options"][0]["fingerprint"]})
    assert [r["group"] for r in recs] == ["A", "B"] and all(r["question_id"] == card["id"]
                                                            for r in recs)


def test_an_answer_given_through_a_card_is_seen_by_every_cell_it_names(nine_tmp):
    """Review finding 1. An answer given on the page is recorded against the CARD, so a cell that
    matched only its own `question_id` saw none of them: the codes the answer retired, the findings
    it overruled and the `settled` history were all empty at the next page build, and the cell was
    protected only by the re-pool having released it. The union the fold owes the cell is that a
    card id counts as this cell's id — the rest of the matching rules are unchanged.
    """
    (nine_tmp / "overrides.jsonl").unlink()          # the fixture ships one; start from nothing
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "b7523a41b03a:d2|late_adaptation||pair")
    assert {s["kind"] for s in card["slots"]} == {"which_series"}
    option = card["options"][0]
    records = answers_to_overrides(card, {"option": option["key"],
                                          "option_fingerprint": option["fingerprint"],
                                          "note": "read off the figure"})
    assert all(r["question_id"] == card["id"] for r in records)
    _append(nine_tmp, records)

    cells = {q["id"]: q for q in questions_for_run(nine_tmp, fold=False)
             if q["dataset_id"] == "b7523a41b03a:d2" and q["outcome_key"] == "late_adaptation"
             and q["route"] != "map"}   # the map's own measure decision is not a read cell
    assert len(cells) == 2
    for cell in cells.values():
        assert cell["answered"] is True, cell["id"]
        assert cell["answers"], cell["id"]
        # …and the history says what it settled, under the group that was asked
        assert "series_identity_conflict" in cell["settled"][0]["clears"], cell["id"]
        # the code the answer named is retired, so the cell has moved on to its next question
        assert cell["kind"] == "confirm_value", cell["id"]
    assert next(q for q in questions_for_run(nine_tmp)
                if q["id"] == card["id"])["status"] == "answered"


def test_the_row_map_is_read_once_per_page(nine_tmp, monkeypatch):
    """Review finding 2. Every card asks its row where it stands, and reading that per card
    re-globbed every `resolve.json` and the whole extraction table once per card — 25 full scans
    for one nine-paper page, growing as O(cards x papers). The page reads it once and threads it.
    """
    from canopy.review import questions as module

    calls: list[Path] = []
    real = module._rows_of                        # captured ONCE: patching a patch double-counts
    monkeypatch.setattr(module, "_rows_of",
                        lambda run, _real=real: (calls.append(run), _real(run))[1])
    for fold in (True, False):
        calls.clear()
        cards = module.questions_for_run(nine_tmp, fold=fold)
        assert cards
        assert len(calls) <= 1, f"fold={fold} read the row map {len(calls)} times"
        # …and the cards still know where their rows stand, so nothing was saved by not looking
        assert any(q["status_line"] for q in cards)


def test_pair_options_are_restamped_and_a_stale_echo_differs(nine_tmp):
    """A folded option's key is positional twice over, so the fingerprint has to cover both
    slots' numbers — otherwise `a1|b1` under one id means a different pair of numbers after any
    answer, and the server's echo check would wave it through."""
    def option() -> dict[str, Any]:
        return next(q for q in questions_for_run(nine_tmp)
                    if q["id"] == "5039533c85ef:d1|late_adaptation||pair")["options"][0]

    before = option()
    # the numbers a value option offers are the CANDIDATES' own, so that is where the run has to
    # change for the option under one key to become a different answer
    path = nine_tmp / "papers" / "5039533c85ef" / "extract.json"
    payload = json.loads(path.read_text())
    for cand in payload["candidates"]:
        if (cand.get("dataset_id"), cand.get("outcome_key"), cand.get("group")) == \
                ("5039533c85ef:d1", "late_adaptation", "A") and cand.get("mean") is not None:
            cand["mean"] = cand["mean"] + 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    after = option()
    assert before["key"] == after["key"] and before["fingerprint"] != after["fingerprint"]


def test_pair_card_shows_the_effect_size_each_combination_implies(nine_tmp):
    """What a combination MEANS is the number it puts in the forest plot, and it comes from the
    resolver — never from arithmetic the review layer invented for the page."""
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "5039533c85ef:d1|late_adaptation||pair")
    assert any(o["implied_d"] is not None and abs(o["implied_d"] + 0.2256) < 1e-3
               for o in card["options"])
    assert all("implied_d" in o and "implied_note" in o for o in card["options"])


def test_settled_history_on_the_vachon_pair(nine_tmp):
    """A recorded answer is history under the slot it was given for, never an "answered" tick on
    a card whose decision is still open — the cell has moved on to its next blocker."""
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "b7523a41b03a:d2|late_adaptation||pair")
    assert card["status"] == "open"
    settled = next(s for s in card["slots"] if s["group"] == "A")["settled"]
    assert "series_identity_conflict" in settled[0]["clears"] and settled[0]["at"]


def test_a_mixed_kind_cell_is_one_card_of_independent_slots(nine_tmp):
    """A `verifier_refuted` has no per-slot objection a combination could carry, so it is never a
    `pair` — but it is still the same cell as the axis question beside it. The second fold puts
    both on one `cell` card as slots answered ONE AT A TIME: each keeps its own options, each
    writes its own record, and the card is open until both are settled. Nothing is hidden: the
    slot IS the per-cell question, unchanged."""
    qs = questions_for_run(nine_tmp)
    assert not any(q["id"] == "3570e4ce2a9c:d2|late_adaptation||pair" for q in qs)
    mine = [q for q in qs if q["dataset_id"] == "3570e4ce2a9c:d2"
            and q["outcome_key"] == "late_adaptation" and q["route"] != "map"]
    assert [q["id"] for q in mine] == ["3570e4ce2a9c:d2|late_adaptation||cell"]
    card = mine[0]
    assert card["kind"] == "cell" and card["scope"] == "dataset" and card["slot_answers"] is True
    assert card["options"] == []
    assert card["member_ids"] == ["3570e4ce2a9c:d2|late_adaptation|A|which_axis",
                                  "3570e4ce2a9c:d2|late_adaptation|B|verifier_refuted"]
    assert [(s["kind"], s["group"], s["answerable"]) for s in card["slots"]] == [
        ("which_axis", "A", True), ("verifier_refuted", "B", True)]
    cells = {q["id"]: q for q in questions_for_run(nine_tmp, fold=False)}
    for slot in card["slots"]:
        assert slot["options"] == cells[slot["member_id"]]["options"]
    # one slot answered: one record, naming the card, through that slot's OWN question
    axis, refuted = card["slots"]
    stands = next(o for o in refuted["options"] if o["key"] == "stands")
    recs = answers_to_overrides(card, {"slots": [
        {"slot": refuted["member_id"], "option": "stands", "note": "read it; the value stands"}]})
    assert len(recs) == 1 and recs[0]["group"] == "B"
    # the record names the SLOT's own question, not the card: a record naming the card would be
    # read by the cell's next question as an answer to it too
    assert recs[0]["question_id"] == refuted["member_id"]
    assert "verifier_refuted" in recs[0]["overrules"]
    _append(nine_tmp, recs)
    # …and the cell is STILL OPEN: the axis question was not answered. The refutation is retired
    # (group B asks its next question, which is what lets the cell fold as a pair now), and the
    # overruling is on group B's record where the card shows it.
    cells_after = {q["id"]: q for q in questions_for_run(nine_tmp, fold=False)}
    assert refuted["member_id"] not in cells_after
    assert cells_after[axis["member_id"]]["status"] == "open"
    shown = [q for q in questions_for_run(nine_tmp) if q["dataset_id"] == "3570e4ce2a9c:d2"
             and q["outcome_key"] == "late_adaptation" and q["route"] != "map"]
    assert len(shown) == 1 and shown[0]["status"] == "open"
    assert any("verifier_refuted" in was["overrules"]
               for slot in shown[0]["slots"] if slot["group"] == "B" for was in slot["settled"])
    # both slots at once (on the card as it was): one record per slot, each on its own group
    both = answers_to_overrides(card, {"slots": [
        {"slot": axis["member_id"], "option": axis["options"][0]["key"], "note": "left axis"},
        {"slot": refuted["member_id"], "option": stands["key"], "note": "stands"}]})
    assert [r["group"] for r in both] == ["A", "B"]
    # a slot the card does not carry is refused whole
    from canopy.pipeline.overrides import OverrideRejected

    with pytest.raises(OverrideRejected):
        answers_to_overrides(card, {"slots": [{"slot": "nope|x||y", "option": "stands"}]})


def test_a_pair_with_more_than_nine_combinations_is_one_cell_card_of_two_slots(nine_tmp):
    """Six-by-six is not a decision anybody can read off one screen as COMBINATIONS — so it is
    not a pair. It is still one cell: two slots on one card, six options each, answered one at a
    time."""
    qs = questions_for_run(nine_tmp)
    assert not any(q["id"] == "592b3b55a318:d2|aftereffect||pair" for q in qs)
    mine = [q for q in qs if q["dataset_id"] == "592b3b55a318:d2"
            and q["outcome_key"] == "aftereffect"]
    assert len(mine) == 1 and mine[0]["kind"] == "cell"
    assert [len(s["options"]) for s in mine[0]["slots"]] == [6, 6]
    assert all(s["answerable"] for s in mine[0]["slots"])


# ----------------------------------------------------------------- C3: include_paper
def test_include_paper_question_per_mapper_exclusion(nine_tmp):
    qs = [q for q in questions_for_run(nine_tmp) if q["kind"] == "include_paper"]
    assert len(qs) == 3 and all("riterion" in q["prompt"] for q in qs) \
        and all(q["answer_writes"] == "eligibility" for q in qs)
    rec = answers_to_overrides(qs[0], {"option": "include"})[0]
    assert rec["kind"] == "eligibility" and rec["eligible"] is True
    assert all(q["scope"] == "paper" and q["id"].endswith("|||include_paper") for q in qs)


def test_a_papers_own_reviewer_exclusion_is_not_asked_again(nine_tmp):
    """"never for a human's": the card exists because a MODEL decided, and a decision a person
    took is not a question."""
    path = nine_tmp / "exclusions.json"
    rows = json.loads(path.read_text())
    for row in rows:
        if row.get("stage") == "map":
            row["decider"] = "human"
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert not [q for q in questions_for_run(nine_tmp) if q["kind"] == "include_paper"]


# ----------------------------------------------------------------- D1: precedence_override
def _override_the_heuer_row(run: Path) -> None:
    nine.with_record(
        run, "3570e4ce2a9c:d1", "late_adaptation", route="figure", es=-0.627,
        confidence="needs_human",
        flags=["precedence_override", "group_statistics_missing"],
        route_overridden_from="text_mean_se_ci",
        precedence_override_reason=(
            "text_mean_se_ci resolved 27.7/18.9 with no dispersion; figure under 'Figure 2, "
            "panel a' resolved -27.0 ± 4.7125/-40.275 ± 4.75 (candidates …), which converts"))


def test_precedence_override_card_replaces_the_which_value_pair(nine_tmp):
    _override_the_heuer_row(nine_tmp)
    qs = questions_for_run(nine_tmp)
    card = next(q for q in qs if q["id"] == "3570e4ce2a9c:d1|late_adaptation||precedence_override")
    assert not any(q["id"].endswith("|which_value")
                   and q["id"].startswith("3570e4ce2a9c:d1|late_adaptation") for q in qs)
    assert "27.7" in card["prompt"] and card["scope"] == "dataset"
    recs = answers_to_overrides(card, {"option": "use_candidates"})
    assert [r["kind"] for r in recs] == ["value", "value"]
    assert {r["mean"] for r in recs} == {-27.0, -40.275}


def test_the_other_two_answers_to_a_precedence_override(nine_tmp):
    """Keeping the printed value is a review decision on both cells; taking the dataset out is
    one exclusion. Neither is a number, and neither pretends to be."""
    _override_the_heuer_row(nine_tmp)
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "3570e4ce2a9c:d1|late_adaptation||precedence_override")
    assert [o["key"] for o in card["options"]] == ["use_candidates", "keep_printed", "exclude"]
    kept = answers_to_overrides(card, {"option": "keep_printed", "note": "the printed pair"})
    assert [r["kind"] for r in kept] == ["mark_reviewed", "mark_reviewed"]
    assert [r["group"] for r in kept] == ["A", "B"]
    gone = answers_to_overrides(card, {"option": "exclude", "note": "no usable spread"})
    assert [r["kind"] for r in gone] == ["exclude_dataset"]


def test_a_precedence_card_carries_the_refutations_on_its_row(nine_tmp):
    """D1's card replaces the value questions of its row; the refutations it does NOT answer ride
    on the same card as slots of their own, so the row is one place on the page — and the
    objection is still overruled by name, never retired by the precedence decision."""
    _override_the_heuer_row(nine_tmp)
    qs = questions_for_run(nine_tmp)
    card = next(q for q in qs if q["id"] == "3570e4ce2a9c:d1|late_adaptation||precedence_override")
    assert not any(q["dataset_id"] == "3570e4ce2a9c:d1" and q["outcome_key"] == "late_adaptation"
                   and q["id"] != card["id"] and q["route"] != "map" for q in qs)
    assert card["slot_answers"] is True
    refutations = [s for s in card["slots"] if s["kind"] == "verifier_refuted"]
    assert [s["group"] for s in refutations] == ["A", "B"] and all(s["answerable"]
                                                                   for s in refutations)
    assert all(not s["answerable"] for s in card["slots"] if s["kind"] != "verifier_refuted")
    assert [o["key"] for o in card["options"]] == ["use_candidates", "keep_printed", "exclude"]
    assert "3570e4ce2a9c:d1|late_adaptation|A|verifier_refuted" in card["member_ids"]
    # the precedence decision alone leaves both refutations open: the decision is recorded, and
    # the objections go back to a card of their own (a `cell` card of the two refutations)
    _append(nine_tmp, answers_to_overrides(card, {"option": "use_candidates", "note": "pair"}))
    again = {q["id"]: q for q in questions_for_run(nine_tmp)
             if q["dataset_id"] == "3570e4ce2a9c:d1" and q["outcome_key"] == "late_adaptation"}
    assert again[card["id"]]["status"] != "open" and again[card["id"]]["slot_answers"] is False
    assert "3570e4ce2a9c:d1|late_adaptation||cell" in again
    assert [s["kind"] for s in again["3570e4ce2a9c:d1|late_adaptation||cell"]["slots"]] == [
        "verifier_refuted", "verifier_refuted"]
    # the decision AND the refutations in one POST (on the card as it was shown): two values,
    # two overrulings, on their groups
    recs = answers_to_overrides(card, {"option": "use_candidates", "note": "pair", "slots": [
        {"slot": s["member_id"], "option": "stands"} for s in refutations]})
    assert [r["kind"] for r in recs] == ["value", "value", "mark_reviewed", "mark_reviewed"]
    assert [r["group"] for r in recs] == ["A", "B", "A", "B"]
    assert all("verifier_refuted" in r["overrules"] for r in recs[2:])
    _append(nine_tmp, recs[2:])
    done = [q for q in questions_for_run(nine_tmp) if q["dataset_id"] == "3570e4ce2a9c:d1"
            and q["outcome_key"] == "late_adaptation" and q["status"] == "open"]
    assert not any(s["kind"] == "verifier_refuted" for q in done for s in q["slots"]), done


def test_a_cell_card_refuses_every_answer_that_decides_nothing(nine_tmp):
    """Review findings 2, 3, 8, 9 on the second fold. A `cell` card offers no options of its own,
    so a card-level `option` is one it never showed; a slot entry with nothing in it, a slot named
    twice, and a bare `group` (the selector, not an answer) are all refused rather than recorded as
    "a human looked at it" — the overclaim §C4 removed."""
    from canopy.pipeline.overrides import OverrideRejected

    card = next(q for q in questions_for_run(nine_tmp) if q["kind"] == "cell")
    slot = card["slots"][0]
    with pytest.raises(OverrideRejected):
        answers_to_overrides(card, {"option": "stands", "note": "never offered here"})
    with pytest.raises(OverrideRejected):
        answers_to_overrides(card, {"slots": [{"slot": slot["member_id"]}]})
    with pytest.raises(OverrideRejected):
        answers_to_overrides(card, {"slots": [
            {"slot": slot["member_id"], "option": slot["options"][0]["key"]},
            {"slot": slot["member_id"], "option": slot["options"][-1]["key"]}]})
    with pytest.raises(OverrideRejected):
        answers_to_overrides(card, {"group": "A", "note": "picked a group, typed nothing"})
    # a bare `group` beside real slot answers is not a second, phantom answer
    recs = answers_to_overrides(card, {"group": slot["group"], "note": "x", "slots": [
        {"slot": slot["member_id"], "option": slot["options"][0]["key"]}]})
    assert len(recs) == 1 and recs[0]["group"] == slot["group"]
    # …and the two things a cell card DOES take at card level: the row's exclusion, and a typed
    # value that names its group (recorded as that slot's own question)
    gone = answers_to_overrides(card, {"exclude": True, "note": "unreadable"})
    assert [r["kind"] for r in gone] == ["exclude_dataset"]
    typed = answers_to_overrides(card, {"group": slot["group"], "mean": 12.5,
                                        "dispersion_value": 2.0, "dispersion_type": "SD",
                                        "n": 10, "note": "Table 1"})
    assert len(typed) == 1 and typed[0]["group"] == slot["group"]
    assert typed[0]["question_id"] == slot["member_id"] and typed[0]["kind"] == "value"


def test_a_precedence_card_never_carries_a_direction_or_a_typed_hint(nine_tmp):
    """Review finding 4: the attach uses the cell fold's own filter. A direction is the measure
    card's (one place, one answer — never twice on one card as two contradictory records) and a
    `no_value` is a typed hint, not a pick."""
    from canopy.review.questions import _NOT_CELL_FOLDABLE

    nine.with_record(nine_tmp, "d1f2946e7e81:d2", "late_adaptation",
                     flags=["precedence_override", "group_statistics_missing"],
                     precedence_override_reason="text converts to nothing")
    qs = questions_for_run(nine_tmp)
    card = next(q for q in qs
                if q["id"] == "d1f2946e7e81:d2|late_adaptation||precedence_override")
    assert not any(s["kind"] in _NOT_CELL_FOLDABLE for s in card["slots"] if s["answerable"])
    measure = [q for q in qs if q["kind"] == "orientation" and q["paper_id"].startswith("d1f29")]
    assert measure and len(measure[0]["slots"]) >= 2, "the measure card still owns the direction"


# ----------------------------------------------------------------- D4-lite: analysed_n
def test_analysed_n_is_one_card_per_paper_with_a_slot_per_dataset(nine_tmp):
    """Two flagged datasets of one paper: one card, one answerable slot each, and each slot's
    answer is the dataset's own `group_n` record — settling that slot and no other."""
    for dataset in ("b7523a41b03a:d1", "b7523a41b03a:d2"):
        for g in ("A", "B"):
            nine.with_flags(nine_tmp, dataset, "late_adaptation", g, ["n_before_exclusions"])
    qs = [q for q in questions_for_run(nine_tmp) if q["kind"] == "analysed_n"]
    assert len(qs) == 1 and qs[0]["scope"] == "paper" and qs[0]["slot_answers"] is True
    card = qs[0]
    assert card["id"].endswith("|||analysed_n") and card["options"] == []
    assert [s["dataset_id"] for s in card["slots"]] == ["b7523a41b03a:d1", "b7523a41b03a:d2"]
    assert card["member_ids"] == ["b7523a41b03a:d1|||analysed_n", "b7523a41b03a:d2|||analysed_n"]
    from canopy.pipeline.overrides import OverrideRejected

    with pytest.raises(OverrideRejected):
        answers_to_overrides(card, {"option": "typed", "n_a": 18, "n_b": 16})
    recs = answers_to_overrides(card, {"slots": [
        {"slot": "b7523a41b03a:d1|||analysed_n", "option": "typed", "n_a": 18, "n_b": 16}]})
    assert len(recs) == 1 and recs[0]["kind"] == "group_n"
    assert recs[0]["dataset_id"] == "b7523a41b03a:d1" and (recs[0]["n_a"], recs[0]["n_b"]) == (18, 16)
    assert recs[0]["question_id"] == "b7523a41b03a:d1|||analysed_n"
    _append(nine_tmp, recs)
    after = next(q for q in questions_for_run(nine_tmp) if q["kind"] == "analysed_n")
    assert after["status"] == "open" and after["id"] == card["id"]      # d2 is still open
    # review finding 7: the answered dataset is history on the card, not a second chance to
    # write a second `group_n` for the same arms
    d1, d2 = after["slots"]
    assert d1["answerable"] is False and d1["settled"] and "18/16" in d1["settled"][0]["justification"]
    assert d2["answerable"] is True
    with pytest.raises(OverrideRejected):
        answers_to_overrides(after, {"slots": [
            {"slot": "b7523a41b03a:d1|||analysed_n", "option": "typed", "n_a": 1, "n_b": 1}]})
    recs2 = answers_to_overrides(after, {"slots": [
        {"slot": "b7523a41b03a:d2|||analysed_n", "option": "typed", "n_a": 17, "n_b": 17}]})
    _append(nine_tmp, recs2)
    done = [q for q in questions_for_run(nine_tmp) if q["kind"] == "analysed_n"]
    assert done and all(q["status"] != "open" for q in done)


def test_analysed_n_is_one_card_per_dataset(nine_tmp):
    for g in ("A", "B"):
        nine.with_flags(nine_tmp, "b7523a41b03a:d1", "late_adaptation", g,
                        ["n_before_exclusions"])
    nine.with_flags(nine_tmp, "b7523a41b03a:d1", "aftereffect", "A", ["n_before_exclusions"])
    qs = [q for q in questions_for_run(nine_tmp) if q["kind"] == "analysed_n"]
    assert [q["id"] for q in qs] == ["b7523a41b03a:d1|||analysed_n"]
    rec = answers_to_overrides(qs[0], {"option": "typed", "n_a": 18, "n_b": 16})[0]
    assert rec["kind"] == "group_n" and (rec["n_a"], rec["n_b"]) == (18, 16)
    assert qs[0]["scope"] == "dataset" and qs[0]["answer_writes"] == "group_n"


def test_recruited_minus_excluded_is_offered_only_when_the_check_parsed_both_numbers(nine_tmp):
    """The subtraction is an answer only where the check did the reading: offering it on a cell
    whose exclusion count nobody parsed would be offering a number the tool made up."""
    nine.with_flags(nine_tmp, "b7523a41b03a:d1", "late_adaptation", "A", ["n_before_exclusions"],
                    detail={"recruited": 20, "excluded": 2, "quote": "two were excluded"})
    card = next(q for q in questions_for_run(nine_tmp) if q["kind"] == "analysed_n")
    minus = next(o for o in card["options"] if o["key"] == "recruited_minus_excluded")
    # group B was never flagged, so its size was never in doubt and keeps the one the run used
    assert minus["n_a"] == 18 and isinstance(minus["n_b"], int)
    rec = answers_to_overrides(card, {"option": "recruited_minus_excluded"})[0]
    assert rec["kind"] == "group_n" and rec["n_a"] == 18 and rec["n_b"] == minus["n_b"]
    assert rec["quote"]

    # …and a group the check flagged but could not finish reading has no subtraction to offer
    nine.with_flags(nine_tmp, "b7523a41b03a:d1", "late_adaptation", "B", ["n_before_exclusions"],
                    detail={"recruited": 19, "excluded": None, "quote": "nineteen took part"})
    again = next(q for q in questions_for_run(nine_tmp) if q["kind"] == "analysed_n")
    assert [o["key"] for o in again["options"]] == ["recorded", "typed"]


# ----------------------------------------------------------------- C4: impact
def test_low_impact_is_marked_not_answered(nine_tmp):
    """A badge and a lower section — never an answer the tool gave itself."""
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "5039533c85ef:d1|late_adaptation||pair")
    assert card["impact_band"] == "low" and card["status"] == "open"
    assert all(not q["answered"] for q in questions_for_run(nine_tmp)
               if q["impact_band"] == "low")


def test_a_card_with_no_pooled_number_is_priced_on_its_own_options(nine_tmp):
    """`row` is the fallback basis (§C4): the spread of the effect sizes the card's own answers
    imply. A row the pooler could not leave one out of is not a row whose answer is worthless —
    it is one nothing else can price, and `unknown` would sort it below cards that matter less."""
    path = nine_tmp / "human_review_queue.json"
    queue = json.loads(path.read_text())
    for entry in queue:
        if entry["dataset_id"] == "5039533c85ef:d1" and entry["outcome_key"] == "late_adaptation":
            entry["impact_abs_delta_pooled"] = None
    path.write_text(json.dumps(queue), encoding="utf-8")
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "5039533c85ef:d1|late_adaptation||pair")
    assert card["impact_basis"] == "row"
    implied = sorted(o["implied_d"] for o in card["options"] if o["implied_d"] is not None)
    assert abs(card["impact"] - (implied[-1] - implied[0])) < 1e-9


def test_ordering_is_impact_first_with_a_stated_basis(nine_tmp):
    qs = questions_for_run(nine_tmp)
    assert qs[0]["id"] == "b7523a41b03a:d2|late_adaptation||pair"
    assert all("impact_basis" in q for q in qs)
    seen_unknown = False
    for q in qs:
        if q["answered"]:
            break
        if q["impact_basis"] == "unknown":
            seen_unknown = True
        else:
            assert not seen_unknown, f"{q['id']} ({q['impact_basis']}) is below an unknown"


def test_nine_folds_within_the_measured_arithmetic(nine_tmp):
    """The fold's arithmetic on the recorded run, measured rather than asserted from the plan.

    DECISION §C's "20–24 open cards" is the target for the REGENERATED run, where D2 retires the
    four Langan refutations and D3 pools the Buch aftereffect reads. This fixture predates both,
    so it carries nine refutations and four unfoldable aftereffect cells; the fold is what is
    being measured here, and it takes the 34 recorded per-cell questions to 24 cards, plus the
    three papers whose eligibility nobody has been asked about.
    """
    qs = [q for q in questions_for_run(nine_tmp) if not q["answered"]]
    kinds = {kind: len([q for q in qs if q["kind"] == kind]) for kind in {q["kind"] for q in qs}}
    # the second fold (`cell`): cells that were two cards each — a refutation beside an axis
    # question, two refutations, two six-option groups — are one card each; the two refutations
    # left alone are the two cells whose other group is not held.
    #
    # `pair` 4 and `cell` 8, not 6 and 6, since a cell whose ROW converts to nothing asks for the
    # group's own statistics rather than for a confirmation of a number that builds nothing. Those
    # cells fold to `cell` rather than pairing, so two `pair` cards became four `cell` cards. The
    # totals below are the invariant that matters and neither of them moved.
    assert kinds == {"pair": 4, "orientation": 4, "cell": 8, "verifier_refuted": 2,
                     "include_paper": 3}
    assert len(qs) == 21
    # …out of the 34 per-cell questions the run recorded, plus the seven measures this run's maps
    # chose for themselves. Those are decisions already taken, shown so a reviewer can take them
    # again (`_settled_measure_questions`): they are `answered`, so none of them is in the 21 above,
    # and none of them folds with the cell it names — a card a reviewer must answer may not change
    # shape because another card mentions the same cell. The three paper-level cards are not a fold
    # of anything: no cell was ever read in those papers, so the unfolded path has none.
    unfolded = questions_for_run(nine_tmp, fold=False)
    assert len(unfolded) == 40
    settled = [q for q in unfolded if q["route"] == "map"]
    # six of this run's seven rulings, not seven: the Cressman aftereffect ruling set aside only
    # POOLED locations, which no reader may take a cell's value from, so it carries no alternative
    # a person could switch to. A card whose every option empties the cell is not a question.
    assert len(settled) == 6 and {q["kind"] for q in settled} == {"which_measure"}
    assert all(q["answered"] and q["status"] == "settled" for q in settled)


def test_every_answer_to_every_card_is_one_the_log_accepts(nine_tmp):
    """§C4's invariant, carried onto the cards: a card that offers an answer the log would refuse
    is a question nobody can answer. Folded or not, every option of every card — including the two
    findings this fixture predates — becomes at least one record `append_override` validates.
    """
    from canopy.pipeline.overrides import _validate

    _override_the_heuer_row(nine_tmp)
    nine.with_flags(nine_tmp, "b7523a41b03a:d1", "late_adaptation", "A", ["n_before_exclusions"],
                    detail={"recruited": 20, "excluded": 2, "quote": "two were excluded"})
    for fold in (True, False):
        cards = questions_for_run(nine_tmp, fold=fold)
        assert cards
        for card in cards:
            assert card["id"].count("|") == 3, card["id"]
            assert card["scope"] in ("cell", "dataset", "measure", "paper")
            assert card["impact_basis"] in ("pooled", "row", "unknown")
            assert card["impact_band"] in ("low", "high")
            slots = [s for s in card["slots"] if s.get("answerable")]
            assert card["options"] or card["answered"] or slots, f"{card['id']} asks nothing"
            assert bool(slots) == bool(card["slot_answers"]), card["id"]
            for option in card["options"]:
                assert option.get("fingerprint")
                if option.get("needs_input"):
                    continue          # a form, not an answer: it is waiting for typed numbers
                records = answers_to_overrides(card, {"option": option["key"], "note": "checked"})
                assert records, (card["id"], option["key"])
                for record in records:
                    _validate(record)
            # …and every option of every slot answered on its own, through `slots`
            for slot in slots:
                assert slot["options"], (card["id"], slot["member_id"])
                for option in slot["options"]:
                    assert option.get("fingerprint")
                    if option.get("needs_input"):
                        continue
                    records = answers_to_overrides(card, {"slots": [
                        {"slot": slot["member_id"], "option": option["key"], "note": "checked"}]})
                    assert len(records) == 1, (card["id"], slot["member_id"], option["key"])
                    _validate(records[0])


def test_the_unfolded_path_is_what_the_page_is_built_from(nine_tmp):
    """Every card's members are the per-cell questions, unchanged — the fold adds a view, it does
    not replace the thing being viewed."""
    cells = {q["id"]: q for q in questions_for_run(nine_tmp, fold=False)}
    for card in questions_for_run(nine_tmp):
        if card["scope"] == "cell":
            assert card["member_ids"] == [card["id"]]
        for member in card["member_ids"]:
            assert member in cells or card["scope"] in ("paper", "dataset"), member


# ------------------------------------------------- fix round 2: a card action does what it says
def test_exclude_these_cells_on_a_precedence_card_excludes(nine_tmp):
    """Fix round 2, finding 11. The page's "Exclude these cells" button posts `{exclude: true}`
    with no `option`, and `_precedence_answers` read only `option`: the empty key fell through to
    the `keep_printed` branch and wrote two `mark_reviewed` records saying "the printed value
    stands". The card ticked and nothing was excluded — the opposite decision, recorded."""
    _override_the_heuer_row(nine_tmp)
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "3570e4ce2a9c:d1|late_adaptation||precedence_override")
    recs = answers_to_overrides(card, {"exclude": True, "note": "neither value is usable"})
    assert [r["kind"] for r in recs] == ["exclude_dataset"]


def test_an_answer_that_decides_nothing_is_refused_by_the_precedence_card(nine_tmp):
    """…and the same fallthrough is why an unrecognised answer must be refused rather than read
    as "keep the printed value": a reviewer who typed a note and picked nothing decided nothing."""
    from canopy.pipeline.overrides import OverrideRejected

    _override_the_heuer_row(nine_tmp)
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "3570e4ce2a9c:d1|late_adaptation||precedence_override")
    for bad in ({"note": "not sure yet"}, {"option": "typo", "note": "hmm"}, {}):
        with pytest.raises(OverrideRejected):
            answers_to_overrides(card, bad)


def test_keep_printed_keeps_the_printed_value_on_both_lines(nine_tmp):
    """Fix round 2, finding 12. `keep_printed` wrote a plain `mark_reviewed`, and every rebuild
    goes through `resolve_effect_with_fallback`, which re-raised the override the reviewer had
    just refused: the row kept the candidate pair's effect size, `precedence_override` and its
    place on the best-guess line, while the card reported "answered … keeps no effect size from
    it". The decision is on the record now, and the rebuild offers that row no alternatives."""
    _override_the_heuer_row(nine_tmp)
    card = next(q for q in questions_for_run(nine_tmp)
                if q["id"] == "3570e4ce2a9c:d1|late_adaptation||precedence_override")
    kept = answers_to_overrides(card, {"option": "keep_printed",
                                       "note": "the printed pair is the paper's own number"})
    assert all(r["kind"] == "mark_reviewed" and r["keep_printed"] is True
               and r["confidence"] == "needs_human" for r in kept)
    _append(nine_tmp, kept)
    _repool(nine_tmp)
    row = {(r["dataset_id"], r["outcome_key"]): r for r in json.loads(
        (nine_tmp / "results" / "extraction_table_all.json").read_text())
        }[("3570e4ce2a9c:d1", "late_adaptation")]
    assert row["route"] == "not_convertible" and "precedence_override" not in row["flags"]
    assert not row.get("in_best_guess")


def test_an_eligibility_answer_that_decides_nothing_leaves_the_paper_out(nine_tmp):
    """Fix round 2, finding 13. `decision or ("exclude" if exclude else "include")` made INCLUDE
    the default of every malformed answer: a note-only submit, a typo'd option key, a decision
    spelled "excluded" or "no" all came back `eligible: True` — and an included paper buys a map
    and an extraction at the next `--resume`. A paper stays out unless a reviewer says the word."""
    from canopy.pipeline.overrides import OverrideRejected

    card = next(q for q in questions_for_run(nine_tmp) if q["kind"] == "include_paper")
    for bad in ({"note": "not sure yet"}, {"option": "typo", "note": "hmm"},
                {"decision": "excluded", "note": "hmm"}, {"decision": "no", "note": "hmm"}):
        with pytest.raises(OverrideRejected):
            answers_to_overrides(card, bad)
    out = answers_to_overrides(card, {"decision": "exclude", "note": "no older adults"})[0]
    assert out["eligible"] is False and "excluded by the reviewer" in out["justification"]
    keep = answers_to_overrides(card, {"option": "include", "note": "it does have both arms"})[0]
    assert keep["eligible"] is True and "included by the reviewer" in keep["justification"]


def test_a_later_per_cell_n_is_not_overwritten_by_an_earlier_dataset_n(nine_tmp):
    """Fix round 2, MINOR 30. The analysed size is kept on the run state so that it reaches every
    later rebuild of that dataset's rows — but it was re-applied to the arm unconditionally, so a
    per-cell answer made AFTER it (a reviewer reading the cell's own n out of the table) was
    overwritten on the way into the resolver: the review table showed the later number and the row
    was divided by the earlier one. The newer answer is the reviewer's current word."""
    ds, key = "b7523a41b03a:d1", "late_adaptation"
    _append(nine_tmp, [
        {"kind": "group_n", "paper_id": "b7523a41b03a" + "0" * 52, "dataset_id": ds,
         "n_a": 18, "n_b": 16, "justification": "the analysed sizes, from the participants para"},
        {"kind": "value", "paper_id": "b7523a41b03a" + "0" * 52, "dataset_id": ds,
         "outcome_key": key, "group": "A", "n": 12,
         "justification": "the table for this measure reports twelve in that arm"},
    ])
    _repool(nine_tmp)
    row = {(r["dataset_id"], r["outcome_key"]): r for r in json.loads(
        (nine_tmp / "results" / "extraction_table_all.json").read_text())}[(ds, key)]
    assert row["n_a"] == 12 and row["n_b"] == 16
    assert "√12" in row["conversion_chain"], "and the row is DIVIDED by the n the table shows"


# ----------------------------------------------------- the answered question that came back open
def _conflicted_readers(run: Path, dataset_id: str, outcome_key: str) -> None:
    """C3's "the two readers named opposite directions" on a cell whose direction was decided —
    the state the fixture predates, in the shape `verify.checks` writes it."""
    for group in ("A", "B"):
        nine.with_flags(run, dataset_id, outcome_key, group, ["orientation_direction_conflict"])


def test_a_recorded_direction_settles_the_measure_card_and_is_not_asked_again(nine_tmp):
    """One real run holds 879 identical answers to one measure card, because the card came back
    open after every one of them. A recorded direction retired `orientation_unknown` and nothing
    else, so `orientation_direction_conflict` — the readers' quarrel, which is a reason to ASK for
    a direction and not a second question — was still on the cell, `_kind` went on returning
    `orientation`, and the page asked the answered question for ever. One answer settles it.
    """
    _conflicted_readers(nine_tmp, "5039533c85ef:d1", "late_adaptation")
    card = next(q for q in questions_for_run(nine_tmp)
                if q["kind"] == "orientation" and q["paper_id"].startswith("5039533c85ef"))
    _append(nine_tmp, answers_to_overrides(
        card, {"option": card["options"][0]["key"], "note": "the paper's own wording, p. 3"}))

    after = questions_for_run(nine_tmp)
    assert not [q for q in after if q["id"] == card["id"] and q["status"] == "open"]
    assert not [q for q in after
                if q["kind"] == "orientation" and q["paper_id"] == card["paper_id"]
                and q["measure_name"] == card["measure_name"] and q["status"] == "open"], \
        "nor its successor for the same measure"


def test_every_code_that_asks_for_a_direction_is_one_a_direction_retires():
    """The page and the analysis read ONE set. `_apply_orientation` strips exactly what the page
    stops asking about, so a code that raises the direction question but is not in that set would
    be asked for ever — the shape of the 879-answer defect. Pinned rather than commented, because
    a comment cannot fail when someone adds the next orientation code to `_FLAG_TO_KIND`."""
    from canopy.pipeline.overrides import ORIENTATION_ANSWERED
    from canopy.review.questions import _FLAG_TO_KIND, _ORIENTATION_ASKING

    asks = {code for code, kind in _FLAG_TO_KIND if kind == "orientation"}
    assert asks <= ORIENTATION_ANSWERED, sorted(asks - ORIENTATION_ANSWERED)
    assert _ORIENTATION_ASKING is ORIENTATION_ANSWERED
    # …and the contradiction between a stated direction and the cell's own means is NOT one of
    # them: no direction answers it, and retiring it would swallow the only mechanical check there
    assert "orientation_reader_contradicts_values" not in ORIENTATION_ANSWERED


def test_a_direction_the_means_check_refuses_comes_back_as_the_contradiction_not_as_itself(
        nine_tmp):
    """The other half of the rule: settling the card must not swallow the one mechanical
    contradiction this pipeline has. C3's means check — a stated direction against this cell's own
    resolved raw means — is an `error` of its own (`orientation_reader_contradicts_values`), and a
    recorded direction does not retire it: the cell asks THAT, with its own three answers, and its
    `why` says the recorded answer is the one the means contradict. Never the same question again.
    """
    _conflicted_readers(nine_tmp, "5039533c85ef:d1", "late_adaptation")
    card = next(q for q in questions_for_run(nine_tmp)
                if q["kind"] == "orientation" and q["paper_id"].startswith("5039533c85ef"))
    _append(nine_tmp, answers_to_overrides(
        card, {"option": card["options"][0]["key"], "note": "the paper's own wording, p. 3"}))
    for group in ("A", "B"):
        nine.with_flags(nine_tmp, "5039533c85ef:d1", "late_adaptation", group,
                        ["orientation_reader_contradicts_values"])

    asked = [q for q in questions_for_run(nine_tmp, fold=False)
             if (q["dataset_id"], q["outcome_key"]) == ("5039533c85ef:d1", "late_adaptation")]
    assert asked and {q["kind"] for q in asked} == {"reader_contradicts_values"}
    for question in asked:
        assert "contradicted by this cell's own resolved raw means" in question["why"], \
            question["why"]
        assert question["options"], "and it is answerable, which the direction card no longer is"


# --------------------------------------------------- a number is the answer to "where is a number"
def _blanked_cell(run: Path) -> tuple[str, str]:
    """One cell of the fixture whose paper printed NOTHING — the state `no_value` is asked about.

    The candidates are dropped from the copied stage files rather than invented, which is the same
    device `test_integration_ceiling` uses for this question and leaves everything else the run
    recorded about the cell exactly as the pipeline wrote it.
    """
    from canopy.pipeline.state import read_stage, write_stage

    paper, cell = "b7523a41b03a", ("b7523a41b03a:d1", "aftereffect")
    for stage, field in (("extract", "candidates"), ("verify", "extra_candidates")):
        payload = read_stage(run, paper, stage)
        write_stage(run, paper, stage, {**payload, field: [
            c for c in payload.get(field) or []
            if (c.get("dataset_id"), c.get("outcome_key")) != cell]})
    return cell


def test_a_typed_value_settles_the_no_value_question_for_the_cell_it_names(nine_tmp):
    """A cell whose paper printed nothing asks "where is this value, if it is reported at all?".
    A reviewer typed both groups' numbers on the manual override form — which writes no
    `question_id`, because it is not the questions page — the row pooled, and both `no_value`
    cards stayed open: `answers_this` refuses every unnamed record, on the rule that a decision of
    the same KIND taken elsewhere is not an answer to the question in front of the reviewer. For
    this one kind it is: the question asks for a number and the record carries the whole of one."""
    cell = _blanked_cell(nine_tmp)
    asked = {q["group"]: q for q in questions_for_run(nine_tmp, fold=False)
             if (q["dataset_id"], q["outcome_key"]) == cell and q["route"] != "map"}
    assert {g: q["kind"] for g, q in asked.items()} == {"A": "no_value", "B": "no_value"}

    _append(nine_tmp, [{"kind": "value", "dataset_id": cell[0], "outcome_key": cell[1],
                        "group": "A", "mean": 12.5, "dispersion_value": 3.5,
                        "dispersion_type": "SD", "n": 9,
                        "justification": "Table 2, the older group's row, typed from the paper"}])
    after = {q["group"]: q for q in questions_for_run(nine_tmp, fold=False)
             if (q["dataset_id"], q["outcome_key"]) == cell}
    assert after["A"]["status"] != "open" and after["A"]["answered"] is True
    assert after["B"]["status"] == "open", "and the arm nobody typed a number for still asks"


def test_a_value_that_names_another_question_does_not_settle_the_no_value_card(nine_tmp):
    """"Answers clear only what they name" is the standing rule, and the exception above is for
    records that name NOTHING — the manual override form's. A record that does name a question
    names a different one, and reinterpreting it settles a card its reviewer never saw."""
    cell = _blanked_cell(nine_tmp)
    _append(nine_tmp, [{"kind": "value", "dataset_id": cell[0], "outcome_key": cell[1],
                        "group": "A", "question_id": f"{cell[0]}|{cell[1]}|A|which_value",
                        "mean": 12.5, "dispersion_value": 3.5, "dispersion_type": "SD", "n": 9,
                        "justification": "Table 2, the older group's row, typed from the paper"}])
    asked = {q["group"]: q for q in questions_for_run(nine_tmp, fold=False)
             if (q["dataset_id"], q["outcome_key"]) == cell}
    assert asked["A"]["status"] == "open" and asked["B"]["status"] == "open"


def test_a_mean_with_no_spread_and_no_n_leaves_the_no_value_card_open(nine_tmp):
    """A mean alone is a legal `value` record and is NOT a group's statistics: the row still has
    no effect size and stays `needs_human`, and both cells stay in the review queue. Ticking the
    card on it left a held row with no open question anywhere on the page — the §C4 state the
    review layer exists to remove, and a worse failure than the question it silenced."""
    cell = _blanked_cell(nine_tmp)
    _append(nine_tmp, [{"kind": "value", "dataset_id": cell[0], "outcome_key": cell[1],
                        "group": group, "mean": mean,
                        "justification": "the number in the text, no spread or n printed with it"}
                       for group, mean in (("A", 12.5), ("B", 7.2))])
    _repool(nine_tmp)
    row = next(r for r in json.loads(
        (nine_tmp / "results" / "extraction_table_all.json").read_text())
        if (r["dataset_id"], r["outcome_key"]) == cell)
    assert row["es"] is None and row["confidence"] == "needs_human", "the row is still held"
    open_cards = [q for q in questions_for_run(nine_tmp)
                  if q["status"] == "open"
                  and any((c["dataset_id"], c["outcome_key"]) == cell for c in q["cells"])]
    assert open_cards, "a held row must leave a question open somewhere on the page"


# ------------------------------------------------------ a later answer that states nothing new
def test_a_series_answer_that_names_no_spread_type_never_displaces_a_typed_one(nine_tmp):
    """Fix round: the Langan rows. Both groups' values were typed and the rows resolved; a later
    "this series is this group" was recorded off a candidate whose error bars nobody could
    identify, so it carried `dispersion_type: UNKNOWN` — and `_apply_value` wrote that over the
    spread type a person had typed. A row whose spreads have no type converts by no route at all,
    so cells holding both typed numbers read `one_group_only` with the numbers still on them.

    A recorded human value stays in force until a LATER record states another; an answer that
    states nothing about a field leaves that field where the last answer put it."""
    ds, key = "b7523a41b03a:d2", "late_adaptation"
    _append(nine_tmp, [
        {"kind": "value", "dataset_id": ds, "outcome_key": key, "group": group, "mean": mean,
         "dispersion_value": sd, "dispersion_type": "SD", "n": n,
         "justification": "typed off Figure 3; the caption says the bars are SDs"}
        for group, mean, sd, n in (("A", 30.2, 1.5, 19), ("B", 28.0, 1.6, 21))])
    _repool(nine_tmp)

    def row() -> dict[str, Any]:
        return next(r for r in json.loads(
            (nine_tmp / "results" / "extraction_table_all.json").read_text())
            if (r["dataset_id"], r["outcome_key"]) == (ds, key))

    typed = row()
    assert typed["es"] is not None and (typed["mean_a"], typed["mean_b"]) == (30.2, 28.0)

    _append(nine_tmp, [{"kind": "value", "dataset_id": ds, "outcome_key": key, "group": "A",
                        "mean": 30.2, "dispersion_value": 1.5, "dispersion_type": "UNKNOWN",
                        "n": 19, "clears": ["series_identity_conflict"],
                        "justification": "the upper series is the older group, per the legend"}])
    _repool(nine_tmp)
    after = row()
    assert (after["mean_a"], after["mean_b"]) == (30.2, 28.0)
    assert after["dispersion_type_a"] == "SD" and after["dispersion_type_b"] == "SD"
    assert after["es"] == typed["es"] and after["var"] == typed["var"]


def test_an_unknown_type_stated_about_a_new_spread_never_borrows_the_old_spread_s_label(nine_tmp):
    """The other side of the same rule, and the one that decides whether it is safe. Carrying a
    spread TYPE forward is honest only while the spread is the same number: a label is a statement
    about the number it was stated for. A record that types a DIFFERENT spread and says nobody
    could identify it has typed an unknown-type spread, and the row must go back to being held and
    visible — never converted with the label of the number this one replaced."""
    ds, key = "b7523a41b03a:d2", "late_adaptation"
    _append(nine_tmp, [
        {"kind": "value", "dataset_id": ds, "outcome_key": key, "group": group, "mean": mean,
         "dispersion_value": sd, "dispersion_type": "SD", "n": n,
         "justification": "typed off Figure 3; the caption says the bars are SDs"}
        for group, mean, sd, n in (("A", 30.2, 1.5, 19), ("B", 28.0, 1.6, 21))])
    _repool(nine_tmp)

    def row() -> dict[str, Any]:
        return next(r for r in json.loads(
            (nine_tmp / "results" / "extraction_table_all.json").read_text())
            if (r["dataset_id"], r["outcome_key"]) == (ds, key))

    typed_es = row()["es"]
    assert row()["dispersion_type_a"] == "SD" and typed_es is not None
    _append(nine_tmp, [{"kind": "value", "dataset_id": ds, "outcome_key": key, "group": "A",
                        "mean": 41.5, "dispersion_value": 6.2, "dispersion_type": "UNKNOWN",
                        "n": 19, "clears": ["series_identity_conflict"],
                        "justification": "the upper series is the older group; its bars are "
                                         "drawn but the caption never says what they are"}])
    _repool(nine_tmp)
    after = row()
    assert after["dispersion_a"] == 6.2, "the spread the record states does land"
    assert after["dispersion_type_a"] == "UNKNOWN", "and it carries no label nobody stated for it"
    # …so the typed pair converts by no route, which is what an untyped spread has always meant.
    # Whatever the row is built from afterwards is the resolver's own business (here D1's fallback
    # pair), and it is held either way — never the reviewer's 6.2 divided as though it were an SD.
    assert "group_statistics_missing" in after["flags"]
    assert after["confidence"] == "needs_human"
    assert after["es"] != typed_es, "and the pooled number is not the old label on the new spread"


def test_a_new_spread_with_no_stated_type_does_not_inherit_the_old_label_either(nine_tmp):
    """Re-review MINOR A: the same defect one branch up. A record that states a different spread
    and says nothing at all about its type has still replaced the number the old label described —
    the label may not survive onto it through the absence of a `dispersion_type` field."""
    ds, key = "b7523a41b03a:d2", "late_adaptation"
    _append(nine_tmp, [
        {"kind": "value", "dataset_id": ds, "outcome_key": key, "group": group, "mean": mean,
         "dispersion_value": sd, "dispersion_type": "SD", "n": n,
         "justification": "typed off Figure 3; the caption says the bars are SDs"}
        for group, mean, sd, n in (("A", 30.2, 1.5, 19), ("B", 28.0, 1.6, 21))])
    _repool(nine_tmp)
    _append(nine_tmp, [{"kind": "value", "dataset_id": ds, "outcome_key": key, "group": "A",
                        "mean": 41.5, "dispersion_value": 6.2, "n": 19,
                        "justification": "re-read the figure; the bars' meaning is not stated"}])
    _repool(nine_tmp)
    after = next(r for r in json.loads(
        (nine_tmp / "results" / "extraction_table_all.json").read_text())
        if (r["dataset_id"], r["outcome_key"]) == (ds, key))
    assert after["dispersion_a"] == 6.2
    assert after["dispersion_type_a"] == "UNKNOWN"
    assert after["confidence"] == "needs_human"
