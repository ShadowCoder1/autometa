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
                                  batches_of, prompt_text, screen_candidates)

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


#: the six answers that make each decision legitimate under the v2 rubric, so a scripted verdict
#: passes the guard unchanged and the test is about what it says it is about
ANSWERS_FOR = {
    "include": {"q1": "yes", "q2": "yes", "q3": "yes", "q4": "yes", "q5": "named", "q6": "yes"},
    "exclude": {"q1": "yes", "q2": "yes", "q3": "no", "q4": "unknown", "q5": "unknown",
                "q6": "yes"},
    "unknown": {"q1": "yes", "q2": "unknown", "q3": "unknown", "q4": "unknown", "q5": "possible",
                "q6": "yes"},
}


def verdict(ref: str, decision: str, reason: str, quote: str = "a quote", **answers) -> dict:
    base = dict(ANSWERS_FOR.get(decision, ANSWERS_FOR["unknown"]))
    base.update(answers)
    return {"ref": ref, "decision": decision, "reason": reason, "quote": quote, **base}


def _verdicts(*rows: tuple[str, str, str]) -> dict:
    return {"decisions": [verdict(ref, decision, reason) for ref, decision, reason in rows]}


def _client(payloads, *, usage=None) -> tuple[LLMClient, FakeProvider]:
    provider = FakeProvider(payloads, usage=usage)
    return LLMClient(provider=provider, cache_dir=None), provider


def _screen(client, candidates, **kw) -> ScreenOutcome:
    kw.setdefault("budget_usd", None)
    return screen_candidates(client, candidates, question=QUESTION, criteria=CRITERIA,
                             model=MODEL, cell_key_prefix="screen:test", **kw)


def _sent_text(provider: FakeProvider, index: int = 0) -> str:
    return prompt_text(provider.requests[index].messages[0]["content"])


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
    assert set(items["required"]) == set(items["properties"]) == {
        "ref", "q1", "q2", "q3", "q4", "q5", "q6", "decision", "quote", "reason"}
    assert items["properties"]["q4"]["enum"] == ["yes", "unknown"], "never no from an abstract"
    assert items["properties"]["q5"]["enum"] == ["named", "possible", "no", "unknown"]


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


def test_the_reservation_covers_the_retry_the_client_may_make():
    """§M8: `client.structured` retries once at double `max_tokens` when the model runs out of
    output room, and that retry is a SECOND billed call. A reservation that priced only the first
    one let a measured batch of 20 bill $0.134 against a $0.051 check — the cap was not a cap.

    Asserted as arithmetic, because the number the cap is compared against is the thing that was
    wrong: it must be the first call PLUS the retry, not twice the first (the retry re-sends the
    whole prompt as well as doubling the output room).
    """
    from canopy.llm.costs import estimate_request_cost
    from canopy.search.screen import MAX_TOKENS, SYSTEM, _prompt_for

    cands = [_cand(1), _cand(2)]
    client, _ = _client([_verdicts(("1", "include", "fine"), ("2", "include", "fine"))])

    outcome = _screen(client, cands)

    messages = [{"role": "user", "content": _prompt_for(cands, question=QUESTION,
                                                        criteria=CRITERIA)}]
    first = estimate_request_cost(MODEL, SYSTEM, messages, MAX_TOKENS)
    retry = estimate_request_cost(MODEL, SYSTEM, messages, MAX_TOKENS * 2)
    assert outcome.batches[0].estimated_usd == pytest.approx(first + retry)
    assert outcome.batches[0].estimated_usd > first, "the retry is priced in, not hoped away"


def test_money_billed_inside_a_budget_exceeded_batch_is_charged_to_somebody():
    """§M9: `BudgetExceeded` can be raised by the RETRY's reservation, long after the first call
    was answered and billed. That branch used to leave `record.cost_usd` at zero, so a batch that
    really cost money was reported to the user — and to the cap's own arithmetic — as $0.00.

    `stop_reason="max_tokens"` is what makes the client retry; a budget that fits the first
    reservation and not the second is what makes the retry raise.
    """
    cands = [_cand(1)]
    provider = FakeProvider([_verdicts(("1", "include", "fine"))], stop_reason="max_tokens",
                            usage=EXPENSIVE)
    client = LLMClient(provider=provider, cache_dir=None, budget_usd=1.05)

    outcome = _screen(client, cands, budget_usd=100.0)

    assert outcome.stopped_because == "budget"
    assert len(provider.requests) == 1, "answered and billed once; the retry was refused"
    billed = client.total_cost()
    assert billed > 0.9, "the provider really was paid for that call"
    assert outcome.batches[0].cost_usd == pytest.approx(billed)
    assert outcome.cost_usd == pytest.approx(billed), "every cent reaches the record"


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


# ================================================================ the v2 rubric (design 03 §4)
"""The screener sees the protocol, answers six questions with a quote, and a guard enforces the
decision rules in code. The three answer-key papers the first search excluded on "no comparison"
framing (design 01) are walked through the guard exactly as design 04 §C walked them."""
from canopy.protocol import load_protocol  # noqa: E402
from canopy.search.screen import (PREAMBLE, PROMPT_VERSION, SYSTEM, _guard,  # noqa: E402
                                  preamble_for, related_term_in)
from tests.test_server import PROTOCOL as PROTOCOL_PATH  # noqa: E402

RUBRIC = [
    {"rule": "The study is written in English.", "kind": "language", "abstract_can_fail": True},
    {"rule": "The participants were neurologically healthy.", "kind": "population",
     "abstract_can_fail": True},
    {"rule": "The study reports at least one protocol outcome.", "kind": "outcome",
     "abstract_can_fail": False},
]
PLAN = {"blocks": [
    {"name": "groups", "terms": ["older adults", "younger adults"],
     "expanded": {"older adults": ["older", "adults"], "younger adults": ["younger"]},
     "pruned": []},
    {"name": "related_designs", "terms": ["untrained group", "control group"],
     "expanded": {"untrained group": ["untrained"], "control group": []}, "pruned": []}]}


def test_the_prompt_is_the_v2_one_and_the_preamble_carries_the_protocol():
    protocol = load_protocol(str(PROTOCOL_PATH))
    text = preamble_for(question=QUESTION, protocol=protocol, rubric=RUBRIC)
    assert PROMPT_VERSION == "search-screen-2"
    assert "presence of a DESIGN" in SYSTEM and "RELATED design" in SYSTEM
    assert "GROUP A — " in text and "GROUP B — " in text and "also called:" in text
    assert "OUTCOMES (any one suffices):" in text and "Late adaptation" in text
    hard = text.split("HARD RULES (an abstract may fail these):")[1].split("SOFT RULES")[0]
    soft = text.split("SOFT RULES (an abstract may meet these, never fail them):")[1]
    assert "1. The study is written in English." in hard
    assert "2. The participants were neurologically healthy." in hard
    assert "3. It is a primary study with results" in hard, "always the last hard rule"
    assert "1. The study reports at least one protocol outcome." in soft
    assert "q5 groups — does the DESIGN contain both GROUP A and GROUP B?" in text
    # no rubric: the protocol's eligibility list is SOFT and the primary-study rule alone is HARD
    bare = preamble_for(question=QUESTION, protocol=protocol, rubric=())
    assert "1. It is a primary study with results" in bare.split("HARD RULES")[1].split("SOFT")[0]
    assert protocol.eligibility[0] in bare.split("SOFT RULES")[1]


def test_the_user_message_is_two_blocks_with_one_cache_breakpoint_on_the_preamble():
    """System (≈ 380 tokens) + preamble (≈ 1,000) clears the 1,024-token cache minimum, so every
    batch after the first reads the preamble from cache. v1 sent one plain string and
    `cache_creation_input_tokens` was 0 on all eleven calls (design 04 MAJOR-4)."""
    cands = [_cand(1), _cand(2)]
    client, provider = _client([_verdicts(("1", "include", "fine"), ("2", "include", "fine"))])
    _screen(client, cands, protocol=load_protocol(str(PROTOCOL_PATH)), rubric=RUBRIC)
    content = provider.requests[0].messages[0]["content"]
    assert isinstance(content, list) and len(content) == 2
    assert content[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in content[1]
    assert content[0]["text"].startswith("The review asks:") and "GROUP A" in content[0]["text"]
    assert content[1]["text"].startswith("Screen the 2 records below")
    assert "1. Paper number 1" in content[1]["text"]
    # byte-identical preamble across batches: that is what a cache prefix is
    cands = [_cand(i) for i in range(1, 25)]
    client, provider = _client([_verdicts(*[(str(i), "include", "ok") for i in range(1, 21)]),
                                _verdicts(*[(str(i), "include", "ok") for i in range(1, 5)])])
    _screen(client, cands, protocol=load_protocol(str(PROTOCOL_PATH)), rubric=RUBRIC)
    first, second = (r.messages[0]["content"][0]["text"] for r in provider.requests[:2])
    assert first == second


def test_the_answers_and_the_quote_land_on_the_candidate():
    cands = [_cand(1)]
    client, _ = _client([{"decisions": [verdict("1", "include", "both groups named",
                                                quote="dominant and non-dominant", q4="yes")]}])
    _screen(client, cands)
    assert cands[0].screen_answers == {"q1": "yes", "q2": "yes", "q3": "yes", "q4": "yes",
                                       "q5": "named", "q6": "yes",
                                       "quote": "dominant and non-dominant"}
    assert cands[0].screen_decision == "include" and cands[0].state == "wanted"


def test_an_off_enum_answer_is_unknown_on_that_question():
    cands = [_cand(1)]
    client, _ = _client([{"decisions": [dict(verdict("1", "unknown", "x"), q4="no", q5="maybe")]}])
    _screen(client, cands)
    assert cands[0].screen_answers["q4"] == "unknown" and cands[0].screen_answers["q5"] == "unknown"


# --------------------------------------------------------------------------------- the guard
def _record(text: str) -> Candidate:
    return Candidate(key="c000000000009", title="A paper", abstract=text)


def test_guard_rule_1_an_exclude_with_no_failing_hard_question_is_unknown():
    answers = {"q1": "yes", "q2": "yes", "q3": "yes", "q4": "unknown", "q5": "possible",
               "q6": "yes"}
    decision, note = _guard("exclude", answers, _record("x"), PLAN)
    assert decision == "unknown" and "without a failing hard question" in note


def test_guard_rule_2_fires_when_q5_is_the_only_no_and_a_groups_word_is_present():
    """Design 04 MAJOR-3 / §C, the Poh shape: q3 = unknown (the abstract never names the
    perturbation), q5 = no, q6 = yes. Rule 1 lets it through (q5 is a failing question); rule 2
    must not, because the record talks about the untrained group."""
    poh = {"q1": "yes", "q2": "yes", "q3": "unknown", "q4": "unknown", "q5": "no", "q6": "yes"}
    record = _record("Transfer of learning to the untrained limb was measured.")
    decision, note = _guard("exclude", poh, record, PLAN)
    assert decision == "unknown" and '"untrained"' in note and "full text" in note
    # the same answers on a record that mentions no groups word: the exclude stands
    assert _guard("exclude", poh, _record("A single cohort adapted to the task."), PLAN) == (
        "exclude", "")
    # a real hard no beside q5 = no: the exclude stands even with a groups word present
    hard = dict(poh, q3="no")
    assert _guard("exclude", hard, record, PLAN) == ("exclude", "")


def test_guard_rule_3_an_include_without_named_groups_on_a_met_rubric_is_unknown():
    named = {"q1": "yes", "q2": "yes", "q3": "yes", "q4": "unknown", "q5": "named", "q6": "yes"}
    assert _guard("include", named, _record("x"), PLAN) == ("include", "")
    assert _guard("include", dict(named, q5="possible"), _record("x"), PLAN)[0] == "unknown"
    assert _guard("include", dict(named, q3="unknown"), _record("x"), PLAN)[0] == "unknown"
    assert _guard("include", dict(named, q6="no"), _record("x"), PLAN)[0] == "unknown"
    # a review is excluded on q6 alone, whatever else the abstract names
    review = {"q1": "yes", "q2": "unknown", "q3": "unknown", "q4": "unknown", "q5": "named",
              "q6": "no"}
    assert _guard("exclude", review, _record("older adults were compared"), PLAN) == (
        "exclude", "")


def test_related_term_in_is_literal_whole_word_and_hyphen_blind():
    assert related_term_in(_record("Older-adults and the untrained group"), PLAN) == "older adults"
    assert related_term_in(_record("An UNTRAINED cohort"), PLAN) == "untrained"
    assert related_term_in(_record("adults who were told"), PLAN) == "adults"
    assert related_term_in(_record("nothing here"), PLAN) == ""
    assert related_term_in(_record("older adults"), None) == ""


def test_the_04_rubric_walk_poh_wang2003_wang2011():
    """Design 04 §C, as the guard sees it: Poh → unknown, Wang 2003 → include, Wang 2011 →
    include. Answers as the walk gives them; the records mention the groups words the walk
    quotes."""
    walk = [
        ("Poh 2016", {"q1": "yes", "q2": "yes", "q3": "yes", "q4": "unknown", "q5": "possible",
                      "q6": "yes"}, "unknown", "transfer between the left and right limbs"),
        ("Wang & Sainburg 2003", {"q1": "yes", "q2": "yes", "q3": "yes", "q4": "unknown",
                                  "q5": "named", "q6": "yes"}, "include",
         "the other arm"),
        ("Wang 2011", {"q1": "yes", "q2": "yes", "q3": "yes", "q4": "unknown", "q5": "named",
                       "q6": "yes"}, "include", "with the left arm, then with the right arm"),
    ]
    for _name, answers, decision, text in walk:
        assert _guard(decision, answers, _record(text), PLAN)[0] == decision


# --------------------------------------------------------------------------------- the audit
def test_the_exclude_audit_re_asks_bounded_excludes_and_reverses_the_ones_that_change():
    """Excludes with q5 = no that still mention a groups word, at most 10 % of the round, most
    relevant first, asked again with one prefixed line; a changed verdict → unknown."""
    cands = [_cand(i, abstract="Older adults were tested in a single group.") for i in range(1, 6)]
    cands += [_cand(i, abstract="A single cohort, no groups named.") for i in range(6, 21)]
    for i, c in enumerate(cands):
        c.relevance = 20.0 - i
    first = {"decisions": [dict(verdict(str(i), "exclude", "single group", q3="no", q5="no"))
                           for i in range(1, 21)]}
    # the guard keeps these excludes (q3 = no is a hard no) — the audit is what re-asks them
    audit_answer = {"decisions": [dict(verdict("1", "unknown", "on reflection both groups",
                                               q3="unknown", q5="possible")),
                                  dict(verdict("2", "exclude", "still single group", q3="no",
                                               q5="no"))]}
    client, provider = _client([first, audit_answer])
    outcome = _screen(client, cands, plan=PLAN)

    assert len(provider.requests) == 2
    audit_text = prompt_text(provider.requests[1].messages[0]["content"])
    assert audit_text.count('This record contains the words: "older adults". Answer q5 again') == 2
    assert "Screen the 2 records below" in audit_text, "10 % of 20, most relevant first"
    reversed_ones = [c for c in cands if c.screen_answers.get("audit") == "reversed"]
    confirmed = [c for c in cands if c.screen_answers.get("audit") == "confirmed"]
    assert [c.key for c in reversed_ones] == [cands[0].key]
    assert cands[0].screen_decision == "unknown" and cands[0].state == "unsure" and cands[0].keep
    assert "asked again with the groups words in view" in cands[0].screen_reason
    assert [c.key for c in confirmed] == [cands[1].key]
    assert cands[1].screen_decision == "exclude"
    assert all(c.screen_decision == "exclude" and "audit" not in c.screen_answers
               for c in cands[2:])
    assert outcome.n_audited == 2 and outcome.n_reversed == 1
    assert outcome.batches[-1].audit is True and outcome.n_screened == 20
    assert any("asked again" in note for note in outcome.notes)
    client, provider = _client([first])
    outcome = _screen(client, cands, plan=PLAN, audit=False)
    assert len(provider.requests) == 1 and outcome.n_audited == 0
