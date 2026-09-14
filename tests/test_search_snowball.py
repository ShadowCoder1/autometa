"""Citation chasing (`canopy.search.snowball`), offline: OpenAlex's filter answers are recorded at
the adapter's own keys, the screening call is scripted, and what is pinned is the design's
mechanics (03 §6 as amended by 04 MAJOR-2) — seeds are the UNEXPANDED includes and unknowns and
are expanded once, every id is hydrated before anything is ranked, `cited_by_seeds` counts the
seeds a paper is linked to, a round with no new include stops the chase, and a round asks at
most 250 requests.
"""
from __future__ import annotations

import json
from pathlib import Path

from canopy.search.indices import OpenAlex
from canopy.search.models import Candidate
from canopy.search.snowball import MAX_REQUESTS_PER_ROUND, round_cap, snowball
from canopy.search.transport import HttpResponse, RecordedTransport

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "search"
PLAN = {"blocks": [
    {"name": "groups", "terms": ["older adults"], "expanded": {"older adults": []}, "pruned": []},
    {"name": "task", "terms": ["rotation", "prism"], "expanded": {"rotation": [], "prism": []},
     "pruned": []},
    {"name": "phenomenon", "terms": ["adaptation"], "expanded": {"adaptation": []}, "pruned": []},
    {"name": "related_designs", "terms": [], "expanded": {}, "pruned": []}]}


def template_work():
    return json.loads((FIXTURES / "openalex_works.json").read_text())["results"][0]


def work(w: str, doi: str, title: str, refs=(), authors=("A1",)) -> dict:
    row = dict(template_work())
    row.update({"id": f"https://openalex.org/{w}", "doi": f"https://doi.org/{doi}",
                "display_name": title, "referenced_works": [f"https://openalex.org/{r}" for r in refs],
                "authorships": [{"author": {"id": f"https://openalex.org/{a}",
                                            "display_name": f"Author {a}"}} for a in authors]})
    return row


def record_filter(transport: RecordedTransport, expr: str, results, *, limit: int,
                  select: str | None = None, cursor: str = "*", next_cursor=None) -> None:
    url, params = OpenAlex().filter_request(expr, limit=limit, select=select, cursor=cursor)
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok", body=json.dumps(
        {"meta": {"count": len(results), "next_cursor": next_cursor},
         "results": list(results)}).encode()), params)


def seed(key: str, doi: str, decision: str, w: str = "", relevance: float = 1.0) -> Candidate:
    c = Candidate(key=key, title=f"Seed {key}", doi=doi, screen_decision=decision,
                  state="wanted" if decision == "include" else "unsure",
                  ids={"openalex": w} if w else {})
    c.relevance = relevance
    return c


def test_one_round_resolves_expands_hydrates_ranks_and_screens_then_stops_on_no_include():
    """Two seeds (one with an OpenAlex id, one by DOI), one shared reference, one forward
    citation each, one shared author with a task-block match; the round hydrates all four new
    ids, screens the two the cap allows, and stops because neither was an include."""
    resolve_select = "id,doi,referenced_works,authorships,cited_by_count"
    transport = RecordedTransport()
    s1 = seed("c000000000001", "10.1000/s1", "include", w="W1", relevance=9.0)
    s2 = seed("c000000000002", "10.1000/s2", "unknown", relevance=8.0)
    dropped = seed("c000000000003", "10.1000/s3", "exclude", w="W3")
    # resolve s2 by DOI; read s1's record by id
    record_filter(transport, "doi:10.1000/s2", [work("W2", "10.1000/s2", "Seed 2",
                                                    refs=("W10", "W11"), authors=("A1", "A2"))],
                  limit=50, select=resolve_select)
    record_filter(transport, "openalex_id:W1", [work("W1", "10.1000/s1", "Seed 1",
                                                    refs=("W10",), authors=("A1",))],
                  limit=50, select=resolve_select)
    # forward citations
    record_filter(transport, "cites:W1", [work("W20", "10.1000/w20", "Cites 1")], limit=200,
                  select="id,doi")
    record_filter(transport, "cites:W2", [work("W21", "10.1000/w21", "Cites 2")], limit=200,
                  select="id,doi")
    # the shared author A1, AND the task block
    record_filter(transport, "authorships.author.id:A1,title_and_abstract.search:(rotation OR "
                             "prism)", [work("W30", "10.1000/w30", "By A1")], limit=200,
                  select="id,doi")
    # hydration of every unseen id, in the order collected
    hydrated = [work("W10", "10.1000/w10", "Prism adaptation in older adults, cited by both"),
                work("W11", "10.1000/w11", "Only seed 2 cites this"),
                work("W20", "10.1000/w20", "Cites 1"), work("W21", "10.1000/w21", "Cites 2"),
                work("W30", "10.1000/w30", "By A1")]
    record_filter(transport, "openalex_id:W10|W11|W20|W21|W30", hydrated, limit=50)

    screened = []

    def screen(chosen, round_index):
        screened.append((round_index, [c.key for c in chosen]))
        for c in chosen:
            c.screen_decision = "unknown"
            c.state = "unsure"
        return type("Outcome", (), {"n_screened": len(chosen), "cost_usd": 0.01,
                                    "stopped_because": ""})()

    everything, rounds = snowball([s1, s2, dropped], transport=transport, plan=PLAN,
                                  screen=screen, cap=2)

    assert len(rounds) == 1 and rounds[0]["n_seeds"] == 2, "the exclude is never a seed"
    assert s1.expanded_as_seed and s2.expanded_as_seed and not dropped.expanded_as_seed
    assert s2.ids["openalex"] == "W2", "resolved by DOI"
    assert rounds[0]["n_raw_ids"] == 5 and rounds[0]["n_hydrated"] == 5 and rounds[0]["n_new"] == 5
    new = {c.ids["openalex"]: c for c in everything if c.found_by == ["snowball"]}
    assert set(new) == {"W10", "W11", "W20", "W21", "W30"}
    assert all(c.found_in_round == 1 and c.title for c in new.values()), "hydrated before ranked"
    assert new["W10"].cited_by_seeds == 2 and new["W11"].cited_by_seeds == 1
    # ranked, then the cap: the paper both seeds cite, with the block words in its title, first
    assert screened == [(1, [new["W10"].key, screened[0][1][1]])]
    assert rounds[0]["n_screened"] == 2 and rounds[0]["n_included"] == 0
    assert rounds[0]["stopped"] == "no new include"
    assert rounds[0]["requests"] == {"resolve": 2, "forward": 2, "author": 1,
                                     "backward_hydrate": 1}
    assert sum(rounds[0]["requests"].values()) <= MAX_REQUESTS_PER_ROUND


def test_a_second_round_starts_from_the_new_includes_only_and_never_re_expands():
    resolve_select = "id,doi,referenced_works,authorships,cited_by_count"
    transport = RecordedTransport()
    s1 = seed("c000000000001", "10.1000/s1", "include", w="W1", relevance=9.0)
    record_filter(transport, "openalex_id:W1", [work("W1", "10.1000/s1", "Seed 1", refs=("W10",))],
                  limit=50, select=resolve_select)
    record_filter(transport, "cites:W1", [], limit=200, select="id,doi")
    record_filter(transport, "openalex_id:W10", [work("W10", "10.1000/w10", "Round one find")],
                  limit=50)
    # round two: only W10 is a seed; it is read by id, cites nothing new, is cited by nobody
    record_filter(transport, "openalex_id:W10", [work("W10", "10.1000/w10", "Round one find",
                                                     refs=("W1",))], limit=50,
                  select=resolve_select)
    record_filter(transport, "cites:W10", [], limit=200, select="id,doi")

    def screen(chosen, round_index):
        for c in chosen:
            c.screen_decision = "include"
            c.state = "wanted"
        return type("Outcome", (), {"n_screened": len(chosen), "cost_usd": 0.0,
                                    "stopped_because": ""})()

    everything, rounds = snowball([s1], transport=transport, plan=PLAN, screen=screen, cap=10,
                                  max_rounds=3)
    assert [r["n_seeds"] for r in rounds] == [1, 1], "round two's seed is round one's include"
    assert rounds[0]["n_included"] == 1 and rounds[1]["n_new"] == 0
    assert rounds[1]["stopped"] == "no new include", "and nothing new means no round three"
    keys = [c["key"] for c in transport.calls]
    assert keys.count(keys[0]) == 1, "seed 1 was expanded once, in round one"
    assert len(everything) == 2


def test_the_round_cap_is_the_snowball_share_spread_over_the_rounds():
    assert round_cap(5.0, share=0.20, cost_per_record=0.003) == 111
    assert round_cap(None, share=0.20, cost_per_record=0.003) is None
    assert round_cap(0.0, share=0.20, cost_per_record=0.003) == 0


def test_openalex_refusing_falls_back_to_europe_pmc_for_seeds_with_a_pmid():
    """A 429 on the forward pass turns OpenAlex off for the round; a seed with a PMID is chased
    through Europe PMC's citations and references and hydrated through EXT_ID."""
    from canopy.search.indices import EuropePmc

    resolve_select = "id,doi,referenced_works,authorships,cited_by_count"
    transport = RecordedTransport()
    s1 = seed("c000000000001", "10.1000/s1", "include", w="W1")
    s1.ids["pmid"] = "111"
    record_filter(transport, "openalex_id:W1", [work("W1", "10.1000/s1", "Seed 1")], limit=50,
                  select=resolve_select)
    url, params = OpenAlex().filter_request("cites:W1", limit=200, select="id,doi")
    transport.record(url, HttpResponse(url=url, status=429, outcome="rate_limited",
                                       error="slow down"), params)
    epmc = EuropePmc()
    for kind, ids in (("citations", ["222"]), ("references", ["333"])):
        url, params = epmc.citations_request("111", kind, page=1, limit=1000)
        key = "citation" if kind == "citations" else "reference"
        transport.record(url, HttpResponse(url=url, status=200, outcome="ok", body=json.dumps(
            {f"{key}List": {key: [{"id": i, "source": "MED"} for i in ids]}}).encode()), params)
    body = json.loads((FIXTURES / "europepmc_search.json").read_text())
    rows = body["resultList"]["result"][:2]
    rows[0]["pmid"], rows[1]["pmid"] = "222", "333"
    url, params = epmc.hydrate_request(["222", "333"], limit=100)
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok", body=json.dumps(
        {"resultList": {"result": rows}}).encode()), params)

    def screen(chosen, round_index):
        return type("Outcome", (), {"n_screened": len(chosen), "cost_usd": 0.0,
                                    "stopped_because": ""})()

    everything, rounds = snowball([s1], transport=transport, plan=PLAN, screen=screen, cap=10)
    found = [c for c in everything if c.found_by == ["snowball"]]
    assert {c.ids["pmid"] for c in found} == {"222", "333"}
    assert all(c.cited_by_seeds == 1 and c.found_in_round == 1 for c in found)
    assert rounds[0]["requests"]["epmc_fallback"] == 2 and rounds[0]["requests"]["epmc_hydrate"] == 1
    assert any("stopped answering forward citations" in n for n in rounds[0]["notes"])
