"""The search bench's own machinery (`validation/search_bench/bench.py`): verdict replay, the
answer-key matching rule, and the buckets. No network, no key file, no model.

The bench is the one place the answer keys are read, and these tests never read them either: the
rows below are made up. What is pinned is the replay discipline — a foreign `prompt_version` is
refused rather than replayed, an unrecorded record is counted rather than invented — because a
bench that quietly answered `include` for a verdict it never had would measure nothing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from canopy.llm.providers import LLMRequest
from canopy.search.models import Candidate
from canopy.search.screen import PROMPT_VERSION, SCREEN_SCHEMA, _prompt_for

BENCH = Path(__file__).resolve().parents[1] / "validation" / "search_bench" / "bench.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("search_bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def screening_request(candidates: list[Candidate]) -> LLMRequest:
    """The exact user message the screener sends, so the replay parses what the model sees."""
    prompt = _prompt_for(candidates, question="does X change Y?", criteria=["reports Y"])
    return LLMRequest(model="claude-sonnet-5", system="s", schema=SCREEN_SCHEMA,
                      messages=[{"role": "user", "content": prompt}])


def paper(key: str, title: str, year: int = 2015) -> Candidate:
    return Candidate(key=key, title=title, year=year, venue="Experimental Brain Research",
                     abstract="An abstract. 1. Not a record header.")


def test_verdicts_replay_by_normalised_title_and_count_what_they_lack(bench):
    verdicts = {"prompt_version": PROMPT_VERSION, "model": "claude-sonnet-5", "entries": {
        bench.normalise_title("Limb dominance results from asymmetries in predictive control"):
            {"decision": "include", "reason": "compares the two limbs"},
        bench.normalise_title("Interlimb transfer of visuomotor rotation"):
            {"decision": "exclude", "reason": "no per-limb outcome"}}}
    replay = bench.VerdictReplay(verdicts, PROMPT_VERSION)
    batch = [paper("c000000000001", "Limb Dominance Results From Asymmetries in Predictive "
                                    "Control"),
             paper("c000000000002", "A paper nobody screened"),
             paper("c000000000003", "Interlimb transfer of visuomotor rotation.")]

    answer = replay.answer(screening_request(batch))
    by_ref = {d["ref"]: d for d in answer["decisions"]}
    assert by_ref["1"]["decision"] == "include" and by_ref["1"]["reason"] == "compares the two limbs"
    assert by_ref["3"]["decision"] == "exclude"
    assert by_ref["2"]["decision"] == "unknown" and "no recorded verdict" in by_ref["2"]["reason"]
    assert replay.n_answered == 2 and replay.n_unrecorded == 1
    assert replay.n_version_mismatch == 0


def test_a_foreign_prompt_version_is_refused_not_replayed(bench):
    """v1 verdicts under a v2 rubric would 'measure' a rubric that never ran (04 MAJOR-6)."""
    verdicts = {"prompt_version": "search-screen-0", "entries": {
        bench.normalise_title("Limb dominance"): {"decision": "include", "reason": "x"}}}
    replay = bench.VerdictReplay(verdicts, PROMPT_VERSION)
    answer = replay.answer(screening_request([paper("c000000000001", "Limb dominance")]))
    assert answer["decisions"][0]["decision"] == "unknown"
    assert "search-screen-0" in answer["decisions"][0]["reason"]
    assert replay.n_version_mismatch == 1 and replay.n_answered == 0


def test_the_bench_provider_never_lets_a_screening_call_reach_a_model(bench):
    """`--live` records indexes and makes the plan call; screening stays replayed."""
    replay = bench.VerdictReplay({"prompt_version": PROMPT_VERSION, "entries": {}},
                                 PROMPT_VERSION)
    provider = bench.BenchProvider(verdicts=replay, queries_answer={"queries": [], "criteria": []},
                                   live_plan=False, live_screen=False)
    response = provider.complete(screening_request([paper("c000000000001", "T")]))
    assert json.loads(response.text)["decisions"][0]["decision"] == "unknown"
    assert provider.requests == ["screen"]
    # a plan call with no cache and no --live is a loud miss, never a silent template fallback
    from canopy.llm.errors import MissingFixture

    with pytest.raises(MissingFixture):
        provider.complete(LLMRequest(model="m", schema={"properties": {"blocks": {}}},
                                     messages=[]))


def test_matching_is_doi_first_then_title_jaccard_never_surname_year(bench):
    rows = [{"doi": "10.1000/abc", "title": "Limb dominance results from asymmetries"},
            {"doi": "", "title": "Effect of coordinate frame compatibility on the transfer of "
                                 "implicit and explicit learning across limbs"},
            {"doi": "10.1000/zzz", "title": "Something the search never found"}]
    candidates = [Candidate(key="c000000000001", doi="10.1000/ABC", title="a different title"),
                  Candidate(key="c000000000002", title="Effect of coordinate-frame compatibility "
                                                       "on the transfer of implicit and explicit "
                                                       "learning across limbs."),
                  Candidate(key="c000000000003", title="Handedness can be explained by a serial "
                                                       "hybrid control scheme", year=2014,
                            authors=["Yadav, V"])]
    assert bench.match(rows[0], candidates)[0].key == "c000000000001"
    matched, how = bench.match(rows[1], candidates)
    assert matched.key == "c000000000002" and how.startswith("title")
    assert bench.match(rows[2], candidates) == (None, "")


def test_buckets_follow_the_candidate_s_state(bench):
    assert bench.bucket_of(None) == "not_returned"
    assert bench.bucket_of(Candidate(key="c000000000001")) == "returned_not_screened"
    assert bench.bucket_of(Candidate(key="c000000000001", screen_decision="exclude")) == \
        "screened_out"
    assert bench.bucket_of(Candidate(key="c000000000001", screen_decision="include")) == \
        "screened_in_fetch_failed"
    assert bench.bucket_of(Candidate(key="c000000000001", screen_decision="unknown")) == \
        "screened_in_fetch_failed"
    assert bench.bucket_of(Candidate(key="c000000000001", screen_decision="include",
                                     pdf_path="staging/x.pdf")) == "success"


def test_nothing_under_canopy_reads_the_answer_keys():
    """The keys are grading only. A pipeline that read them would be tuning against its exam."""
    root = Path(__file__).resolve().parents[1] / "canopy"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "gold_nyamsuren" not in text and "gold_cisneros" not in text, path
        assert "gold_" not in text.replace("gold_standard", ""), path
        assert "bench.py" not in text and "validation/search_bench/gold" not in text, path
