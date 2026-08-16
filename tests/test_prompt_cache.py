"""Task 15 §A: the prompt-cache contract, offline and (optionally) live.

The first live run sent 3.7 M input tokens and read 160 k of them from the prompt cache — 4 %,
with 333 of its 353 calls carrying no `cache_control` marker at all. A live smoke test (recorded
in `task-15-report.md`) established the two rules these tests pin:

1. a cache READ needs the request prefix to be byte-identical **including the output schema**, so
   only calls that share a model AND a schema can share an entry — marking a prefix nothing will
   re-read costs 1.25x the input price and saves nothing;
2. the prefix ends at the last `cache_control` marker, so everything invariant (the document, the
   figure image) must come FIRST and everything per-call must come after it.

The offline tests inspect the requests a `FakeProvider` receives. The live one (skipped without
`CANOPY_LIVE=1`) asserts that a real second call actually reads the cache.
"""
from __future__ import annotations

from typing import Any

import pytest

from canopy.agents.extract_text import extract_group_stats
from canopy.agents.mapper import map_study
from canopy.agents.orientation import orientation_run
from canopy.agents.verifier import verify_candidate
from canopy.llm.client import LLMClient
from canopy.llm.providers import FakeProvider
from canopy.models import (Candidate, DatasetSpec, DispersionType, GroupSpec, OutcomeSources,
                           Source, SourceKind)

CAND = Candidate(candidate_id="c1", kind="group_stats", group="A", mean=42.5,
                 dispersion_value=6.9, dispersion_type=DispersionType.SD, n=12, page=2,
                 quote="that for old subjects was 42.5±6.9 s", model="claude-opus-5",
                 route="text", extractor_id="text:table_first:claude-opus-5")
DATASET = DatasetSpec(dataset_id="d1", group_a=GroupSpec(label="old", n=12),
                      group_b=GroupSpec(label="young", n=12))


def _client(payload: Any) -> tuple[LLMClient, FakeProvider]:
    provider = FakeProvider([payload])
    return LLMClient(provider=provider, cache_dir=None), provider


def _blocks(request) -> list[dict]:
    return request.messages[0]["content"]


def _marked(blocks: list[dict]) -> list[int]:
    return [i for i, b in enumerate(blocks) if "cache_control" in b]


# ------------------------------------------------------------------ shape: whole-paper agents
def test_verifier_sends_the_document_first_and_marks_it(paper):
    """The verifier runs once per candidate with ONE schema, so its document prefix is re-read."""
    client, provider = _client({"verdict": "confirmed", "reason": "it is printed", "checked": [],
                                "alt_mean": None, "alt_dispersion_value": None,
                                "alt_dispersion_type": "UNKNOWN", "alt_n": None, "alt_page": None,
                                "alt_quote": "", "better_source": "", "better_source_page": None,
                                "notes": ""})
    verify_candidate(client, paper, CAND, model="claude-sonnet-5", dataset=DATASET)
    blocks = _blocks(provider.requests[0])
    assert blocks[0]["type"] == "document"
    assert _marked(blocks) == [0], "the document is the whole invariant prefix"
    assert blocks[-1]["type"] == "text" and "CANDIDATE" in blocks[-1]["text"]


def test_orientation_sends_the_document_first_and_marks_it(paper):
    client, provider = _client({"raw_value_semantics": "higher_more_error",
                                "higher_is_better": "lower", "direction_stated_in_text": "unknown",
                                "quotes": ["a quote"], "reason": "an error measure"})
    orientation_run(client, paper, DATASET,
                    OutcomeSources(outcome_key="late_adaptation", measure_name="pointing error"),
                    model="claude-sonnet-5")
    blocks = _blocks(provider.requests[0])
    assert blocks[0]["type"] == "document" and _marked(blocks) == [0]


def test_the_mapper_does_not_pay_to_write_a_prefix_nothing_reads(paper, protocol):
    """Every mapper call carries a different output schema, and the schema is part of the prefix.

    Measured live (task 15 §A1): the same document under two schemas produced two cache WRITES and
    no read. A marker here would add 25 % to the most expensive document in the run for nothing.
    """
    payload = {"citation": {"authors": "Bock O", "year": 2005, "title": "t", "journal": "",
                            "doi": ""},
               "eligible": False, "eligibility_rationale": "not eligible",
               "exclusion_reason": "no age contrast", "datasets": [], "roster": [], "notes": ""}
    client, provider = _client(payload)
    map_study(client, paper, protocol)
    documents = [_blocks(r) for r in provider.requests
                 if any(b.get("type") == "document" for b in _blocks(r))]
    assert documents, "the mapper reads the whole PDF"
    for blocks in documents:
        assert blocks[0]["type"] == "document", "the document still comes first"
        assert _marked(blocks) == [], "no marker on a prefix no second call can read"


# ------------------------------------------------------------------ shape: page-context agents
def test_extractor_marks_the_last_page_block_and_asks_afterwards(paper, protocol):
    client, provider = _client({"groups": [], "notes": ""})
    sources = [Source(kind=SourceKind.text_mean_sd, page=2, locator="Results",
                      quote="27.4±7.2 s")]
    extract_group_stats(client, paper, protocol, DATASET, "late_adaptation", sources,
                        variant="table_first")
    blocks = _blocks(provider.requests[0])
    assert _marked(blocks) == [len(blocks) - 2], "the marker ends the page context, not the prompt"
    assert blocks[-1]["type"] == "text" and "OUTCOME" in blocks[-1]["text"]


# ------------------------------------------------------------------ live
@pytest.mark.live
def test_live_a_second_call_on_the_same_document_reads_the_cache(paper):
    """A real second call with the same model, schema and document must READ, not re-write.

    Budget: two Sonnet calls over the 5-page Bock PDF, about $0.10.
    """
    from canopy.agents.verifier import VERIFIER_SCHEMA
    from canopy.agents.verify_common import SYSTEM, whole_paper
    from canopy.config import load_env
    from canopy.llm.context import text_block

    load_env()
    client = LLMClient(cache_dir=None, budget_usd=2.0, allow_live=True)
    document, betas = whole_paper(client, paper, None)

    def ask(question: str):
        return client.structured(
            model="claude-sonnet-5", system=SYSTEM, schema=VERIFIER_SCHEMA, effort="medium",
            max_tokens=1000, betas=betas, cell_key="verify:cache-probe",
            messages=[{"role": "user", "content": [document, text_block(question)]}])

    first = ask("Say `confirmed` and nothing else; leave every other field empty or null.")
    second = ask("Say `ambiguous` and nothing else; leave every other field empty or null.")
    assert first.usage.get("cache_creation_input_tokens", 0) >= 1024, first.usage
    assert second.usage.get("cache_read_input_tokens", 0) >= 1024, (
        f"the second identical prefix did not read the cache: {second.usage}")
    assert second.cost_usd < first.cost_usd
