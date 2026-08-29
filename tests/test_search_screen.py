"""Title/abstract screening (`canopy.search.screen`), offline, against a scripted `FakeProvider`.

Nothing here opens a socket or reads a key: every test drives a real `LLMClient` whose provider is
`FakeProvider`, so the schema audit, the cost accounting and the budget reservation are the *real*
ones and only the model's answer is canned.

The first test is the point of the file. `client.structured` runs `assert_valid_output_schema`
before it sends anything (llm/client.py:368-370), and that audit rejects a string enum with no
"unknown-like" member — which is what the design's `["include", "exclude", "unsure"]` was. Every
screening call would have raised `AssertionError` before a byte left the machine.
"""
from __future__ import annotations

import copy

import pytest

from canopy.llm.client import LLMClient
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import (UNKNOWN_ENUM_MEMBERS, assert_no_derived_stats,
                                assert_valid_output_schema, find_schema_problems)
from canopy.search.models import Candidate, counts_of
from canopy.search.screen import (BATCH, MAX_ABSTRACT_CHARS, NO_ABSTRACT_MARKER, NO_MODEL_REASON,
                                  NO_VERDICT_REASON, SCREEN_DECISIONS, SCREEN_SCHEMA,
                                  TITLE_ONLY_NOTE, TRUNCATION_MARKER, ScreenOutcome,
                                  batches_of, screen_candidates)

MODEL = "claude-sonnet-5"                 # what MODELS["secondary"] is; see the module docstring
QUESTION = "does resistance training reduce tremor in Parkinson's disease?"
CRITERIA = ["adults with idiopathic Parkinson's disease",
            "a resistance-training arm and a comparator",
            "a tremor outcome reported numerically"]

#: an answer that costs about a dollar a call, so a budget test does not turn on rounding
EXPENSIVE = {"input_tokens": 1000, "output_tokens": 100_000,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# --------------------------------------------------------------------------------- fixtures
def _cand(i: int, *, title: str = "", abstract: str = "Randomised trial of progressive resistance "
                                                      "exercise reporting UPDRS tremor subscore.",
          **kw) -> Candidate:
    """One candidate with a well-formed, sortable key (`^[cu][0-9a-f]{12}$`)."""
    return Candidate(key=f"c{i:012x}", title=title or f"Paper number {i}", year=2013,
                     venue="Movement Disorders", abstract=abstract, **kw)


def _verdicts(*rows: tuple[str, str, str]) -> dict:
    return {"decisions": [{"ref": ref, "decision": decision, "reason": reason}
                          for ref, decision, reason in rows]}


def _client(payloads, *, usage=None) -> tuple[LLMClient, FakeProvider]:
    provider = FakeProvider(payloads, usage=usage)
    return LLMClient(provider=provider, cache_dir=None), provider


def _screen(client, candidates, **kw) -> ScreenOutcome:
    kw.setdefault("budget_usd", None)
    return screen_candidates(client, candidates, question=QUESTION, criteria=CRITERIA,
                             model=MODEL, cell_key_prefix="screen:test", **kw)


def _sent_text(provider: FakeProvider, index: int = 0) -> str:
    return provider.requests[index].messages[0]["content"]


# --------------------------------------------------------------------------------- the schema
def test_the_screen_schema_survives_canopys_own_output_audit():
    """B-BLK1: this is the audit that made the original design unable to run at all."""
    assert find_schema_problems(SCREEN_SCHEMA) == []
    assert_valid_output_schema(SCREEN_SCHEMA, "SCREEN_SCHEMA")
    assert_no_derived_stats(SCREEN_SCHEMA, name="SCREEN_SCHEMA")


def test_the_decision_enum_says_unknown_and_never_unsure():
    """The page shows the word "unsure"; the schema must say "unknown" — the audit accepts only
    members of `UNKNOWN_ENUM_MEMBERS`, and `counts_of` maps `unknown` to the `unsure` count."""
    enum = SCREEN_SCHEMA["properties"]["decisions"]["items"]["properties"]["decision"]["enum"]
    assert enum == ["include", "exclude", "unknown"] == list(SCREEN_DECISIONS)
    assert set(enum) & UNKNOWN_ENUM_MEMBERS == {"unknown"}
    assert "unsure" not in enum


def test_the_designs_original_enum_really_is_rejected():
    """Pins the blocker itself: swap `unknown` back for `unsure` and the audit refuses the call."""
    broken = copy.deepcopy(SCREEN_SCHEMA)
    broken["properties"]["decisions"]["items"]["properties"]["decision"]["enum"] = [
        "include", "exclude", "unsure"]
    problems = find_schema_problems(broken)
    assert len(problems) == 1 and "no unknown-like member" in problems[0]
    with pytest.raises(AssertionError):
        assert_valid_output_schema(broken, "SCREEN_SCHEMA")


def test_every_object_in_the_schema_is_strict():
    items = SCREEN_SCHEMA["properties"]["decisions"]["items"]
    assert SCREEN_SCHEMA["additionalProperties"] is False
    assert items["additionalProperties"] is False
    assert set(items["required"]) == set(items["properties"]) == {"ref", "decision", "reason"}


# --------------------------------------------------------------------------------- verdicts land
def test_a_verdict_lands_on_the_right_candidate_with_its_state_and_keep():
    cands = [_cand(1), _cand(2), _cand(3)]
    client, provider = _client([_verdicts(
        ("1", "include", "randomised resistance-training trial reporting a tremor subscore"),
        ("2", "exclude", "an animal model, so it fails the population criterion"),
        ("3", "unknown", "the abstract does not say whether tremor was measured"))])

    outcome = _screen(client, cands)

    assert len(provider.requests) == 1 and outcome.n_screened == 3
    assert [c.screen_decision for c in cands] == ["include", "exclude", "unknown"]
    assert [c.state for c in cands] == ["wanted", "excluded", "unsure"]
    assert [c.keep for c in cands] == [True, False, True]
    assert cands[1].screen_reason.startswith("an animal model")
    # the PRISMA ladder the page reads is derived from exactly these fields
    counts = counts_of(cands)
    assert (counts["screened"], counts["included"], counts["excluded"], counts["unsure"],
            counts["not_screened"]) == (3, 1, 1, 1, 0)


def test_the_models_reason_is_stored_verbatim():
    """The reason is the audit trail a reviewer defends — never paraphrased, never trimmed."""
    sentence = ("excluded: the cohort is 12 people with essential tremor, not Parkinson's — "
                'the abstract calls it "ET" throughout (n = 12; 2 dropouts).')
    cands = [_cand(1)]
    client, _ = _client([_verdicts(("1", "exclude", sentence))])

    _screen(client, cands)

    assert cands[0].screen_reason == sentence


def test_a_record_the_model_skipped_stays_not_screened_and_says_nobody_read_it():
    """Never silently dropped and never defaulted to `exclude`: "nobody read it" and "someone read
    it and said no" are different facts, and the page shows them in different buckets."""
    cands = [_cand(1), _cand(2), _cand(3)]
    client, _ = _client([_verdicts(("1", "include", "meets every criterion"),
                                   ("3", "exclude", "a review, not a primary study"))])

    outcome = _screen(client, cands)

    assert outcome.n_screened == 2 and outcome.batches[0].n_verdicts == 2
    assert cands[1].state == "not_screened"
    assert cands[1].screen_decision == "" and cands[1].keep is False
    assert cands[1].screen_reason == NO_VERDICT_REASON
    assert "nobody read it" in cands[1].screen_reason
    assert counts_of(cands)["screened"] == 2 and counts_of(cands)["not_screened"] == 1


def test_an_answer_the_enum_does_not_contain_becomes_unknown_and_says_so():
    """A model that answers with the page's word must not silently become an `exclude`."""
    cands = [_cand(1)]
    client, _ = _client([_verdicts(("1", "unsure", "the abstract is ambiguous about the outcome"))])

    _screen(client, cands)

    assert cands[0].screen_decision == "unknown" and cands[0].keep is True
    assert cands[0].screen_reason.startswith("the abstract is ambiguous about the outcome")
    assert '"unsure"' in cands[0].screen_reason and "recorded as unknown" in cands[0].screen_reason


def test_a_candidate_that_already_has_a_pdf_keeps_the_state_the_file_gave_it():
    """A screener's opinion does not un-fetch a paper; the decision and reason are still recorded."""
    cands = [_cand(1, pdf_path="uploads/ab12.pdf", state="fetched")]
    client, _ = _client([_verdicts(("1", "exclude", "wrong population"))])

    _screen(client, cands)

    assert cands[0].state == "fetched"
    assert cands[0].screen_decision == "exclude" and cands[0].keep is False


# --------------------------------------------------------------------------- abstracts and titles
def test_a_record_without_an_abstract_is_screened_on_its_title_and_the_reason_says_so():
    cands = [_cand(1, abstract="")]
    client, provider = _client([_verdicts(
        ("1", "unknown", "the title names Parkinson's but no outcome"))])

    _screen(client, cands)

    assert NO_ABSTRACT_MARKER in _sent_text(provider)     # the model is told, not left to guess
    assert cands[0].screened_on_title_only is True
    assert cands[0].screen_reason.startswith("the title names Parkinson's but no outcome")
    assert cands[0].screen_reason.endswith(TITLE_ONLY_NOTE)
    assert "no abstract" in cands[0].screen_reason


def test_a_record_with_an_abstract_is_not_marked_title_only():
    cands = [_cand(1)]
    client, _ = _client([_verdicts(("1", "include", "meets every criterion"))])

    _screen(client, cands)

    assert cands[0].screened_on_title_only is False
    assert TITLE_ONLY_NOTE not in cands[0].screen_reason


def test_a_long_abstract_is_truncated_before_it_is_sent():
    """1,800 characters, and the cut is visible — see the cost note on `screen_candidates`."""
    abstract = "A" * 2500 + " ENDOFABSTRACTMARKER"
    cands = [_cand(1, abstract=abstract)]
    client, provider = _client([_verdicts(("1", "include", "fine"))])

    _screen(client, cands)

    sent = _sent_text(provider)
    assert MAX_ABSTRACT_CHARS == 1800
    assert "A" * MAX_ABSTRACT_CHARS in sent and "A" * (MAX_ABSTRACT_CHARS + 1) not in sent
    assert TRUNCATION_MARKER in sent
    assert "ENDOFABSTRACTMARKER" not in sent
    assert QUESTION in sent and CRITERIA[0] in sent      # the question and criteria travel with it


# --------------------------------------------------------------------------------- batching
def test_forty_one_candidates_are_three_batches_of_twenty_twenty_one():
    cands = [_cand(i) for i in range(1, 42)]
    client, provider = _client([_verdicts(("1", "include", "fine"))])

    outcome = _screen(client, cands)

    assert BATCH == 20
    assert len(provider.requests) == 3
    assert [len(b.keys) for b in outcome.batches] == [20, 20, 1]
    assert all(b.sent for b in outcome.batches)


def test_the_batch_composition_is_pinned_by_key_so_a_resumed_search_replays_for_free():
    """Arrival order is a network fact; re-batching in a different order misses every cache entry
    and bills the user a second time with no warning (review §D3)."""
    cands = [_cand(i) for i in (7, 3, 9, 1, 5)]
    keys = sorted(c.key for c in cands)
    client, _ = _client([_verdicts(("1", "include", "fine"))])

    outcome = _screen(client, cands, batch_size=2)

    assert [b.keys for b in outcome.batches] == [keys[0:2], keys[2:4], keys[4:5]]
    assert [b.keys for b in outcome.batches] == [[c.key for c in b]
                                                 for b in batches_of(cands, 2)]


def test_the_reference_numbers_are_per_batch_not_global():
    """`ref` "1" in the second batch means the second batch's first record, not the search's."""
    cands = [_cand(1), _cand(2)]
    client, _ = _client([_verdicts(("1", "include", "first batch, first record")),
                         _verdicts(("1", "exclude", "second batch, first record"))])

    _screen(client, cands, batch_size=1)

    assert cands[0].screen_decision == "include" and cands[1].screen_decision == "exclude"
    assert cands[1].screen_reason == "second batch, first record"


def test_progress_is_reported_once_per_batch_with_n_and_total():
    cands = [_cand(i) for i in range(1, 6)]
    client, _ = _client([_verdicts(*[(str(i), "include", "fine") for i in range(1, 3)])])
    seen: list[dict] = []

    _screen(client, cands, batch_size=2, on_progress=seen.append)

    assert len(seen) == 3
    assert [e["n"] for e in seen] == [2, 4, 5]
    assert all(e["stage"] == "screen" and e["total"] == 5 for e in seen)


# --------------------------------------------------------------------------------- the cost cap
def test_the_cap_stops_between_batches_and_the_rest_say_which_cap_stopped_them():
    """A batch already paid for is kept; the batch that would not fit is never sent."""
    cands = [_cand(1), _cand(2), _cand(3)]
    client, provider = _client([_verdicts(("1", "include", "meets every criterion"))],
                               usage=EXPENSIVE)

    outcome = _screen(client, cands, batch_size=1, budget_usd=1.00)

    assert len(provider.requests) == 1                    # one batch sent, two never attempted
    assert outcome.stopped_because == "budget"
    assert outcome.cost_usd == pytest.approx(1.002)
    assert cands[0].screen_decision == "include" and cands[0].state == "wanted"
    for candidate in cands[1:]:
        assert candidate.state == "not_screened"
        assert candidate.screen_decision == "" and candidate.keep is False
        assert "$1.00 cost cap" in candidate.screen_reason
        assert "before this record was screened" in candidate.screen_reason
    assert [b.sent for b in outcome.batches] == [True, False, False]
    assert counts_of(cands)["not_screened"] == 2


def test_a_budget_exceeded_from_inside_the_client_is_the_same_clean_stop():
    """The client's own reservation refusing the call must read as the cap, not as a crash."""
    cands = [_cand(1), _cand(2)]
    provider = FakeProvider([_verdicts(("1", "include", "fine"))], usage=EXPENSIVE)
    client = LLMClient(provider=provider, cache_dir=None, budget_usd=0.001)

    outcome = _screen(client, cands, batch_size=1, budget_usd=100.0)

    assert outcome.stopped_because == "budget"
    assert provider.requests == []                        # refused before anything was sent
    assert all(c.state == "not_screened" and c.keep is False for c in cands)
    # the reason names the cap that FIRED ($0.001, the client's), not the one that did not ($100)
    assert "$0.0010 cost cap" in cands[0].screen_reason
    assert "$100.00" not in cands[0].screen_reason


def test_no_cap_screens_everything():
    cands = [_cand(1), _cand(2)]
    client, provider = _client([_verdicts(("1", "include", "fine"))], usage=EXPENSIVE)

    outcome = _screen(client, cands, batch_size=1, budget_usd=None)

    assert len(provider.requests) == 2 and outcome.stopped_because == ""


# --------------------------------------------------------------------------------- degradation
def test_no_client_screens_nothing_and_does_not_raise():
    cands = [_cand(1), _cand(2)]

    outcome = screen_candidates(None, cands, question=QUESTION, criteria=CRITERIA, model=MODEL,
                                budget_usd=1.0, cell_key_prefix="screen:test")

    assert outcome.stopped_because == "" and outcome.n_screened == 0
    assert outcome.batches == [] and outcome.cost_usd == 0.0
    for candidate in cands:
        assert candidate.state == "not_screened"
        assert candidate.screen_decision == "" and candidate.keep is False
        assert candidate.screen_reason == NO_MODEL_REASON
    assert counts_of(cands)["screened"] == 0 and counts_of(cands)["not_screened"] == 2


def test_no_candidates_is_an_empty_outcome_not_a_call():
    client, provider = _client([_verdicts(("1", "include", "fine"))])

    outcome = _screen(client, [])

    assert provider.requests == [] and outcome == ScreenOutcome()


def test_one_failed_batch_costs_twenty_decisions_and_not_the_whole_search():
    """A refusal or unparseable JSON is why a batch is twenty records and not two hundred."""
    cands = [_cand(1), _cand(2)]
    client, provider = _client(["this reply is not JSON at all {",
                                _verdicts(("1", "include", "meets every criterion"))])

    outcome = _screen(client, cands, batch_size=1)

    assert len(provider.requests) == 2 and outcome.n_screened == 1
    assert outcome.stopped_because == ""                  # a bad batch is not a stopped search
    assert "ParseError" in outcome.batches[0].error
    assert cands[0].state == "not_screened" and cands[0].screen_decision == ""
    assert "nobody read this record" in cands[0].screen_reason
    assert cands[1].screen_decision == "include"
