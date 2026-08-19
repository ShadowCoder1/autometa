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
             if q["dataset_id"] == "b7523a41b03a:d2" and q["outcome_key"] == "late_adaptation"}
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


def test_a_mixed_kind_pair_is_not_folded(nine_tmp):
    """A fold that hid a hold would be worse than the repetition it removes: `verifier_refuted`
    has no per-slot objection option, so a cell carrying one keeps its own card."""
    qs = questions_for_run(nine_tmp)
    assert not any(q["id"] == "3570e4ce2a9c:d2|late_adaptation||pair" for q in qs)
    assert {q["id"] for q in qs if q["dataset_id"] == "3570e4ce2a9c:d2"
            and q["outcome_key"] == "late_adaptation"} == {
        "3570e4ce2a9c:d2|late_adaptation|A|which_axis",
        "3570e4ce2a9c:d2|late_adaptation|B|verifier_refuted"}


def test_a_pair_with_more_than_nine_combinations_stays_two_cards(nine_tmp):
    """Six-by-six is not a decision anybody can read off one screen."""
    qs = questions_for_run(nine_tmp)
    assert not any(q["id"] == "592b3b55a318:d2|aftereffect||pair" for q in qs)
    assert len([q for q in qs if q["dataset_id"] == "592b3b55a318:d2"
                and q["outcome_key"] == "aftereffect"]) == 2


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


# ----------------------------------------------------------------- D4-lite: analysed_n
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
    assert kinds == {"pair": 6, "orientation": 4, "verifier_refuted": 9, "which_axis": 3,
                     "which_value": 2, "include_paper": 3}
    assert len(qs) == 27
    # …out of the 34 per-cell questions the run recorded. The three paper-level cards are not a
    # fold of anything: no cell was ever read in those papers, so the unfolded path has none.
    assert len(questions_for_run(nine_tmp, fold=False)) == 34


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
            assert card["options"] or card["answered"], f"{card['id']} asks nothing"
            for option in card["options"]:
                assert option.get("fingerprint")
                if option.get("needs_input"):
                    continue          # a form, not an answer: it is waiting for typed numbers
                records = answers_to_overrides(card, {"option": option["key"], "note": "checked"})
                assert records, (card["id"], option["key"])
                for record in records:
                    _validate(record)


def test_the_unfolded_path_is_what_the_page_is_built_from(nine_tmp):
    """Every card's members are the per-cell questions, unchanged — the fold adds a view, it does
    not replace the thing being viewed."""
    cells = {q["id"]: q for q in questions_for_run(nine_tmp, fold=False)}
    for card in questions_for_run(nine_tmp):
        if card["scope"] == "cell":
            assert card["member_ids"] == [card["id"]]
        for member in card["member_ids"]:
            assert member in cells or card["scope"] in ("paper", "dataset"), member
