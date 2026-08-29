"""Query building: one model call, and the keyless path made of the user's own words."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.protocol import load_protocol
from canopy.search.queries import (MAX_QUERIES, QUERY_SCHEMA, build_queries, template_queries)
from tests.test_server import PROTOCOL as PROTOCOL_PATH


def test_the_query_schema_passes_both_audits():
    """The audit `client.structured` runs on the way to the wire — the search's own version of
    the blocker that made screening impossible before it was ever called."""
    assert_valid_output_schema(QUERY_SCHEMA, "QUERY_SCHEMA")
    assert_no_derived_stats(QUERY_SCHEMA)


def _client(parsed, cost=0.0025):
    return SimpleNamespace(structured=lambda **kw: SimpleNamespace(parsed=parsed, cost_usd=cost))


def test_a_model_answer_becomes_queries_and_criteria():
    client = _client({"queries": [{"text": "  motor   adaptation ageing ", "why": "core"},
                                  {"text": "", "why": "empty ones are dropped"}],
                      "criteria": ["compares two age groups", "  "]})
    out = build_queries(client, "Does ageing change adaptation?", model="m")
    assert out["source"] == "model" and out["cost_usd"] == 0.0025
    assert [q["text"] for q in out["queries"]] == ["motor adaptation ageing"]
    assert out["criteria"] == ["compares two age groups"]


def test_a_failed_call_falls_back_to_the_users_own_words_and_says_so():
    """A search the user asked for does not die because a model blinked — but the record must
    never imply a model wrote queries it did not write."""
    def boom(**kw):
        raise RuntimeError("upstream 503")

    out = build_queries(SimpleNamespace(structured=boom), "Does ageing change adaptation?",
                        model="m")
    assert out["source"] == "template" and out["queries"]
    assert out["notes"] and "failed" in out["notes"][0]


def test_a_model_answer_with_no_usable_query_is_the_same_as_no_answer():
    out = build_queries(_client({"queries": [], "criteria": []}), "a question", model="m")
    assert out["source"] == "template" and out["notes"]


def test_the_template_query_carries_both_arms_of_the_comparison():
    """A comparison query whose vocabulary is all one arm finds papers about that group, not
    papers that compared it with anything — and the second arm is what makes a study eligible."""
    protocol = load_protocol(str(PROTOCOL_PATH))
    out = template_queries("Does ageing change sensorimotor adaptation?", protocol)
    assert out["source"] == "template" and out["cost_usd"] == 0.0
    crossed = next(q["text"] for q in out["queries"] if " AND " in q["text"])
    lowered = crossed.lower()
    assert "older" in lowered and "younger" in lowered
    assert any(q["text"] == "Late adaptation" for q in out["queries"]), "a wider net too"
    assert all("reports" in c or "compares" in c or "primary study" in c for c in out["criteria"])


def test_without_a_protocol_a_quoted_phrase_does_not_become_the_whole_search():
    """A quoted phrase is a good query and a bad search: it finds the papers that use exactly
    those words and misses every paper that says the same thing differently."""
    out = template_queries('Does "cognitive behavioural therapy" reduce anxiety in teenagers?')
    texts = [q["text"] for q in out["queries"]]
    assert "cognitive behavioural therapy" in texts
    assert any("anxiety" in t and "teenagers" in t for t in texts)
    assert all(q["why"] for q in out["queries"]), "every query says what it reaches for"


def test_queries_are_bounded_and_every_one_explains_itself():
    protocol = load_protocol(str(PROTOCOL_PATH))
    out = template_queries("ageing and adaptation", protocol)
    assert 0 < len(out["queries"]) <= MAX_QUERIES
    assert all(q["text"].strip() and q["why"].strip() for q in out["queries"])
