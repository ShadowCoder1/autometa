"""The orchestrator (`canopy.search.run`), end to end, with no network and no paid call.

`run_search` is the real function in every test here. What is replaced is the world outside it: a
`RecordedTransport` answering the index calls out of the JSON those APIs really sent, and an
`LLMClient` whose provider is `FakeProvider` — so the schema audit, the cost accounting and the
budget reservation are all the genuine ones and only the model's words are canned.

The four properties being defended:

* **it never raises for something an index or a publisher did.** OpenAlex running out of its $0.10
  daily budget, a publisher serving a login page, a host answering 429 — each is a row in the
  record and a sentence in `notes`, and the search finishes.
* **the five rungs of `PHASES` all report, in order**, whatever happened inside them.
* **`unique_contributed` is measured.** Europe PMC is asked first because it is unmetered, not
  because this module believes anything about a research field (review §C4) — so the record has to
  carry the evidence for whether that ordering paid off.
* **`stopped_because` is set when a search is cut short**, because a cancelled or capped search
  still finishes `done` and a page told only "done" would never tell the user.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canopy.llm.client import LLMClient
from canopy.llm.providers import FakeProvider
from canopy.search.indices import EuropePmc, OpenAlex
from canopy.search.models import PHASES, counts_of, project
from canopy.search.run import EXCLUDED_BY_USER, MIN_TITLE_FRAGMENT, run_search
from canopy.search.transport import HttpResponse, RecordedTransport, fixture_key

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "search"
PDF = Path(__file__).resolve().parent / "fixtures" / "pdfs" / "bock2005.pdf"

QUESTION = "does resistance training reduce tremor in Parkinson's disease?"
QUERY = "resistance training parkinson tremor"
PER_QUERY = 25
MODEL_ROLES = {"primary": "claude-opus-5", "secondary": "claude-sonnet-5"}

#: what the query call answers. One query, so one request per index and one fixture per index.
QUERIES_PAYLOAD = {"queries": [{"text": QUERY, "why": "the words of the question itself"}],
                   "criteria": ["adults with Parkinson's disease",
                                "a resistance-training arm and a comparator",
                                "a tremor outcome reported numerically"]}


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def fake_probe(_path):
    """A probe that reads nothing. The real one spawns a child interpreter per PDF; what is under
    test here is the orchestrator's bookkeeping, not the PDF parser's (which has its own suite)."""
    return {"ok": True, "n_pages": 5}


def index_transport(*, epmc: dict | None = None, openalex: dict | None = None,
                    epmc_response: HttpResponse | None = None,
                    openalex_response: HttpResponse | None = None) -> RecordedTransport:
    """A transport with one recorded answer per (index, query), at the key the adapter will ask for.

    The keys come from each adapter's own `request()`, so an adapter that changed its parameters
    raises `MissingSearchFixture` here instead of quietly passing against a stale recording.
    """
    transport = RecordedTransport()
    for index, body, response in ((EuropePmc(), epmc, epmc_response),
                                  (OpenAlex(), openalex, openalex_response)):
        url, params = index.request(QUERY, limit=PER_QUERY)
        if response is None:
            response = HttpResponse(url=url, status=200, outcome="ok",
                                    body=json.dumps(body or {}).encode("utf-8"))
        transport.record(url, response, params)
    return transport


def serve_pdfs(transport: RecordedTransport, urls) -> RecordedTransport:
    for url in urls:
        transport.payloads[fixture_key(url)] = PDF
    return transport


def refuse(transport: RecordedTransport, url: str, *, status: int, outcome: str,
           error: str = "") -> RecordedTransport:
    transport.responses[fixture_key(url)] = HttpResponse(
        url=url, status=status, outcome=outcome, error=error or f"HTTP {status}")
    return transport


def scripted_client(*payloads) -> LLMClient:
    return LLMClient(provider=FakeProvider(list(payloads)), cache_dir=None)


def include_everything(n: int) -> dict:
    """A screener that wants every record, so the fetch stage gets a full workload.

    Which candidate lands on which `ref` depends on a hash sort, so a payload that included only
    some of them would make the fetch assertions depend on hashing — the tests would still pass and
    would no longer mean anything.
    """
    return {"decisions": [{"ref": str(i), "decision": "include",
                           "reason": "a resistance-training trial reporting tremor"}
                          for i in range(1, n + 1)]}


@pytest.fixture(autouse=True)
def no_contact_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """No address, so Unpaywall is skipped with its note and no test depends on the developer's
    own environment. The skip note is asserted below — it is part of the contract."""
    monkeypatch.delenv("CANOPY_CONTACT_EMAIL", raising=False)


@pytest.fixture
def phase_log() -> list[tuple[str, str, str, float]]:
    return []


# ------------------------------------------------------------------------------- the whole thing
def test_a_whole_search_end_to_end(tmp_path, phase_log):
    """queries → index → dedupe → screen → fetch, on real index JSON and a real PDF.

    Six Europe PMC records (four open access) and five OpenAlex works, none of which overlap, all
    screened in, and then fetched: the four Europe PMC `?pdf=render` copies land, the two OpenAlex
    `oa_url`s are the NCBI landing pages a live probe returned (`not_a_pdf` — which is exactly how
    a login wall or a Cloudflare page shows up), and the rest had no open-access route at all.
    """
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    render = [f"https://europepmc.org/articles/{pmcid}?pdf=render" for pmcid in
              ("PMC13317673", "PMC12941259", "PMC12982457", "PMC13065030")]
    serve_pdfs(transport, render)
    for landing in ("https://www.ncbi.nlm.nih.gov/pmc/articles/4366306",
                    "https://www.ncbi.nlm.nih.gov/pmc/articles/4586021"):
        refuse(transport, landing, status=200, outcome="not_a_pdf",
               error="www.ncbi.nlm.nih.gov served text/html, not a PDF")

    record = run_search(
        question=QUESTION, transport=transport,
        client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=60,
        staging_dir=tmp_path / "pdfs", per_query=PER_QUERY, probe=fake_probe,
        on_phase=lambda *event: phase_log.append(event))

    counts = counts_of(record.candidates, record.possible_duplicates)
    assert counts["records"] == 11 and counts["after_dedupe"] == 11
    assert counts["screened"] == 11 and counts["included"] == 11
    assert counts["fetched"] == 4, "the four Europe PMC OA copies"
    assert counts["paywalled"] == 7 and counts["wanted"] == 0
    assert record.query_source == "model" and record.cost_usd > 0

    fetched = [c for c in record.candidates if c.state == "fetched"]
    assert all(c.pdf_path.endswith(".pdf") and c.pdf_pages == 5 for c in fetched)
    # `pdf_path` is relative to the SEARCH directory, not to the staging directory inside it —
    # `staging_dir` here is `tmp_path/"pdfs"`, so the search directory is `tmp_path`. This
    # assertion used to join the staging directory onto a staging-relative path and pass only
    # because the fetcher wrote a bare filename, which is the convention that made every fetched
    # paper unresolvable at begin time (review §B1).
    assert all(c.pdf_path.startswith("pdfs/") for c in fetched)
    assert all((tmp_path / c.pdf_path).exists() for c in fetched)
    # every attempt is on the record, including the two that proved a landing page is not a paper
    assert sum(len(c.fetch_attempts) for c in record.candidates) == 6
    assert {a["outcome"] for c in record.candidates for a in c.fetch_attempts} == {"ok",
                                                                                  "not_a_pdf"}
    assert record.stopped_because == ""


def test_the_five_phases_report_in_order(tmp_path, phase_log):
    """The names are shared verbatim with the page; two of these rungs once differed and never lit."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    run_search(question=QUESTION, transport=transport,
               client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
               model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=0,
               staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe,
               on_phase=lambda *event: phase_log.append(event))

    names = [name for name, status, *_ in phase_log if status == "running"]
    assert names == list(PHASES)
    finished = [(name, status) for name, status, *_ in phase_log if status != "running"]
    assert [name for name, _ in finished] == list(PHASES)
    assert all(status in ("ok", "skipped", "error") for _, status in finished)
    assert all(seconds >= 0 for *_, seconds in phase_log)


def test_a_broken_progress_listener_cannot_kill_a_search(tmp_path):
    """A closed SSE connection is the caller's problem, not a reason to lose finished work."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))

    def explode(*_args):
        raise RuntimeError("the browser went away")

    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe,
                        on_phase=explode)
    assert len(record.candidates) == 11 and len(record.phases) == 2 * len(PHASES)


# ------------------------------------------------------------------------- the measured ordering
def test_sources_carry_n_returned_and_unique_contributed(tmp_path):
    """`unique_contributed` is why the index ordering is a fact and not an assumption (§C4)."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    rows = {row["name"]: row for row in record.sources}
    assert set(rows) == {"europepmc", "openalex"}
    assert rows["europepmc"]["n_returned"] == 6 and rows["openalex"]["n_returned"] == 5
    # these two recordings share no paper, so every row is a unique contribution
    assert rows["europepmc"]["unique_contributed"] == 6
    assert rows["openalex"]["unique_contributed"] == 5
    assert all(row["query"] == QUERY for row in record.sources)


def test_a_paper_both_indexes_found_is_counted_once_and_credited_to_neither(tmp_path):
    """The merge case, and the only one that makes `unique_contributed` mean anything.

    The OpenAlex body is the real recording with ONE work's DOI rewritten to a DOI Europe PMC also
    returned — the only way to exercise a cross-index merge without a second live recording, and
    the rewrite is confined to this test.
    """
    openalex = load("openalex_works")
    shared_doi = load("europepmc_search")["resultList"]["result"][0]["doi"]
    openalex["results"][0]["doi"] = f"https://doi.org/{shared_doi}"
    transport = index_transport(epmc=load("europepmc_search"), openalex=openalex)

    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(10)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    counts = counts_of(record.candidates, record.possible_duplicates)
    assert counts["records"] == 11 and counts["after_dedupe"] == 10, "eleven rows, ten papers"
    both = next(c for c in record.candidates if c.doi == shared_doi)
    assert sorted(both.found_by) == ["europepmc", "openalex"]

    rows = {row["name"]: row for row in record.sources}
    assert rows["europepmc"]["unique_contributed"] == 5
    assert rows["openalex"]["unique_contributed"] == 4
    assert sum(row["unique_contributed"] for row in record.sources) == 9, "the shared one is neither's"


# ---------------------------------------------------------------------------- degrading honestly
def test_openalex_running_out_of_budget_leaves_the_search_alive(tmp_path, phase_log):
    """OpenAlex is metered at $0.10 a day. Losing Europe PMC's results over that would be absurd."""
    transport = index_transport(
        epmc=load("europepmc_search"),
        openalex_response=HttpResponse(url="https://api.openalex.org/works", status=429,
                                       outcome="rate_limited",
                                       error="api.openalex.org asked us to slow down"))

    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(6)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe,
                        on_phase=lambda *event: phase_log.append(event))

    assert len(record.candidates) == 6, "Europe PMC's six survive"
    dead = next(row for row in record.sources if row["name"] == "openalex")
    assert dead["n_returned"] == 0 and dead["outcome"] == "rate_limited"
    assert "$0.10" in dead["note"] and "midnight UTC" in dead["note"]
    assert any("$0.10" in note for note in record.notes), "the user is told, in words"
    assert record.stopped_because == "", "one arm failing is not a stopped search"
    assert ("index", "ok") in [(name, status) for name, status, *_ in phase_log]


def test_both_indexes_failing_is_recorded_not_raised(tmp_path, phase_log):
    """A search that found nothing and says why beats a traceback the user cannot act on."""
    down = HttpResponse(url="https://example.org", status=503, outcome="http_error",
                        error="the index answered HTTP 503")
    transport = index_transport(epmc_response=down, openalex_response=down)

    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD), model_roles=MODEL_ROLES,
                        budget_usd=1.0, max_screened=200, max_fetch=0, staging_dir=tmp_path,
                        per_query=PER_QUERY, probe=fake_probe,
                        on_phase=lambda *event: phase_log.append(event))

    assert record.candidates == []
    assert [row["error"] for row in record.sources] == ["the index answered HTTP 503"] * 2
    assert ("index", "error") in [(name, status) for name, status, *_ in phase_log]
    assert len(record.phases) == 2 * len(PHASES), "every rung still reported"


def keyless_transport() -> RecordedTransport:
    """Europe PMC answering both template queries; OpenAlex answering both with nothing.

    The template path builds two queries out of the user's own words, so each index is asked
    twice, and `RecordedTransport` refuses any request it has no recording for.
    """
    index = EuropePmc()
    transport = RecordedTransport()
    for query_text in ("resistance training reduce tremor parkinson disease",
                       "resistance training reduce tremor"):
        url, params = index.request(query_text, limit=PER_QUERY)
        transport.record(url, HttpResponse(
            url=url, status=200, outcome="ok",
            body=json.dumps(load("europepmc_search")).encode()), params)
    url, params = OpenAlex().request("resistance training reduce tremor parkinson disease",
                                     limit=PER_QUERY)
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                       body=json.dumps({"results": [], "meta": {}}).encode()),
                     params)
    url, params = OpenAlex().request("resistance training reduce tremor", limit=PER_QUERY)
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                       body=json.dumps({"results": [], "meta": {}}).encode()),
                     params)
    return transport


#: the four Europe PMC records in the recording that carry a `?pdf=render` copy
KEYLESS_RENDER = [f"https://europepmc.org/articles/{p}?pdf=render"
                  for p in ("PMC13317673", "PMC12941259", "PMC12982457", "PMC13065030")]


def test_with_no_model_the_queries_are_the_user_s_own_words_and_nothing_is_screened(tmp_path):
    """The keyless path is a working feature, not a degraded one.

    Every paper reaches the PAGE with an abstract excerpt and a link — asserted through
    `project()`, which is what a browser actually receives. Asserting `candidate.abstract` here
    instead was green for months while `project()` stripped the abstract and the user got a list
    of bare titles (review §M15): the record having a thing is not the user reading it.
    """
    from canopy.search.models import project

    record = run_search(question=QUESTION, transport=serve_pdfs(keyless_transport(),
                                                                KEYLESS_RENDER),
                        client=None, model_roles=MODEL_ROLES, budget_usd=None, max_screened=200,
                        max_fetch=60, staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    counts = counts_of(record.candidates, record.possible_duplicates)
    assert record.query_source == "template" and record.cost_usd == 0.0
    assert counts["after_dedupe"] == 6 and counts["screened"] == 0
    rows = [project(c) for c in record.candidates]
    assert all(row["abstract_excerpt"] for row in rows), "the abstracts reach the screen"
    assert all(row["links"] for row in rows), "and every row has a link out"
    assert all(link["url"].startswith("https://") for row in rows for link in row["links"])
    assert any("no model was available" in note for note in record.notes)


def test_a_keyless_search_still_fetches_the_open_copies_and_can_become_a_run(tmp_path):
    """§B3: with no key nothing is screened, so nothing is `wanted` — and the old fetch stage
    therefore fetched NOTHING, leaving a list of titles with no PDF, no reason and no way to begin.

    What is asserted here is the whole keyless chain: unscreened records with an open-access route
    are fetched, the file is where `server/searches.py` looks for it, and every row that was NOT
    fetched says in one word why not.
    """
    record = run_search(question=QUESTION, transport=serve_pdfs(keyless_transport(),
                                                                KEYLESS_RENDER),
                        client=None, model_roles=MODEL_ROLES, budget_usd=None, max_screened=200,
                        max_fetch=60, staging_dir=tmp_path / "staging", per_query=PER_QUERY,
                        probe=fake_probe)

    counts = counts_of(record.candidates, record.possible_duplicates)
    assert counts["screened"] == 0, "nobody read a word of these"
    assert counts["fetched"] == 4, "…and the four open-access copies are on disk anyway"
    fetched = [c for c in record.candidates if c.state == "fetched"]
    assert all((tmp_path / c.pdf_path).is_file() for c in fetched)
    # nothing may be called paywalled: the screener never asked for any of these
    assert counts["paywalled"] == 0
    unfetched = [c for c in record.candidates if not c.pdf_path]
    assert unfetched and all(c.fetch_outcome for c in unfetched), \
        "a blank outcome reads as 'we tried and found nothing', which is not what happened"


def test_the_records_count_is_the_rows_the_indexes_returned_not_the_index_names(tmp_path):
    """§M11: `records` is the top-of-funnel number a reviewer publishes, and it was derived from
    `found_by`, which is unique — so one index answering two queries with the same six papers was
    reported as six records instead of twelve, four lines from a phase message that said twelve.

    The keyless transport asks Europe PMC the same recording for both template queries, which is
    exactly that case.
    """
    record = run_search(question=QUESTION, transport=keyless_transport(), client=None,
                        model_roles=MODEL_ROLES, budget_usd=None, max_screened=200, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    counts = counts_of(record.candidates, record.possible_duplicates)
    assert counts["after_dedupe"] == 6, "six distinct papers"
    assert counts["records"] == 12, "…from twelve rows, which is what the indexes returned"
    assert sum(row["n_returned"] for row in record.sources) == counts["records"], \
        "the flow strip and the per-index table are one measurement"
    dedupe_rung = next(p for p in record.phases
                       if p["name"] == "dedupe" and p["status"] == "ok")
    assert "12 record(s)" in dedupe_rung["message"], "…and so is the phase ladder"


def test_the_screening_batch_map_is_written_into_the_record(tmp_path):
    """§M10: `screen.py` said the map was in `search.json` and re-read on resume. It was in
    neither. It is written now — as the audit trail it really is, so a reader can price screening
    per batch — and the docstring no longer claims a resume that does not exist."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    blob = json.loads(json.dumps(record.to_json()))
    assert blob["batches"], "the map reaches disk"
    assert [k for batch in blob["batches"] for k in batch["keys"]] == \
        sorted(c.key for c in record.candidates), "every screened record is accounted for, once"
    assert all(set(batch) >= {"index", "keys", "sent", "estimated_usd", "cost_usd", "n_verdicts"}
               for batch in blob["batches"])
    assert sum(batch["cost_usd"] for batch in blob["batches"]) > 0
    from canopy.search import screen as screen_module

    assert "re-batching from it on resume" not in (screen_module.ScreenBatch.__doc__ or "")


def test_unpaywall_is_skipped_with_a_note_when_there_is_no_contact_address(tmp_path):
    """It rejects placeholders, so there is no honest way to call it. The record says so."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=60,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    assert any("CANOPY_CONTACT_EMAIL" in note for note in record.notes)
    assert not any(row["name"] == "unpaywall" for row in record.sources)


# ------------------------------------------------------------------------- stopping, and saying so
def test_cancelling_between_stages_skips_the_rest_and_names_the_reason(tmp_path, phase_log):
    """`stopped_because` is what the page's banner keys on — a cancelled search still ends `done`."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    seen = {"n": 0}

    def cancelled() -> bool:
        # cancellation is checked BETWEEN stages, so the stage in flight when the user pressed the
        # button finishes: the screener has already paid for the batch it is in, and a stage that
        # is half-done and half-recorded is worse than one that finished.
        seen["n"] += 1
        return True

    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD), model_roles=MODEL_ROLES,
                        budget_usd=1.0, max_screened=200, max_fetch=60, staging_dir=tmp_path,
                        per_query=PER_QUERY, probe=fake_probe, cancelled=cancelled,
                        on_phase=lambda *event: phase_log.append(event))

    assert record.stopped_because == "cancelled"
    assert record.candidates == [] and record.sources == []
    skipped = [name for name, status, *_ in phase_log if status == "skipped"]
    assert skipped == ["index", "dedupe", "screen", "fetch"]
    assert all("you stopped this search" in message
               for _, status, message, _ in phase_log if status == "skipped")
    assert transport.calls == [], "nothing went to an index after the cancellation"


def test_the_screening_cap_leaves_the_rest_listed_and_unscreened(tmp_path):
    """Capped is not dropped: the rows stay, with the reason, and the counts are about the list
    the user is actually looking at."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(4)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=4, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    counts = counts_of(record.candidates, record.possible_duplicates)
    assert counts["after_dedupe"] == 11, "nothing is dropped by a cap"
    assert counts["screened"] == 4 and counts["not_screened"] == 7
    assert any("screening cap of 4" in note for note in record.notes)


def test_the_fetch_cap_is_recorded_on_the_candidates_it_stopped(tmp_path):
    """`over_fetch_cap`, and the papers stay `wanted` — nothing was tried, so nothing is claimed."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    serve_pdfs(transport, [f"https://europepmc.org/articles/{p}?pdf=render"
                           for p in ("PMC13317673", "PMC12941259", "PMC12982457", "PMC13065030")])

    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=2,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    over = [c for c in record.candidates if c.fetch_outcome == "over_fetch_cap"]
    assert len(over) == 9 and all(c.state == "wanted" for c in over)
    assert all(c.fetch_attempts == [] for c in over)
    assert any("fetch cap of 2" in note for note in record.notes)


def test_a_question_that_produces_no_query_says_so_instead_of_searching_nothing(tmp_path,
                                                                                phase_log):
    """An empty search that reports "no query" is honest; one that reports "no results" is not."""
    record = run_search(question="   ", transport=RecordedTransport(), client=None,
                        model_roles=MODEL_ROLES, budget_usd=None, max_screened=10, max_fetch=0,
                        staging_dir=tmp_path, probe=fake_probe,
                        on_phase=lambda *event: phase_log.append(event))

    assert record.queries == [] and record.candidates == []
    assert ("queries", "error") in [(name, status) for name, status, *_ in phase_log]
    assert [name for name, status, *_ in phase_log if status == "skipped"] == [
        "index", "dedupe", "screen", "fetch"]


def test_the_record_is_json_serialisable_with_its_counts(tmp_path):
    """`search.json` is the artefact a reviewer reads without the server running."""
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(11)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=0,
                        staging_dir=tmp_path, per_query=PER_QUERY, probe=fake_probe)

    blob = json.loads(json.dumps(record.to_json()))
    assert blob["counts"]["after_dedupe"] == 11
    assert blob["search_id"].startswith("s") and blob["created_at"]
    assert [phase["name"] for phase in blob["phases"]] == [
        name for name in PHASES for _ in range(2)]
    assert all(set(row) >= {"name", "query", "n_returned", "error", "unique_contributed"}
               for row in blob["sources"])
    assert all(set(candidate) >= {"key", "state", "links", "found_by"}
               for candidate in blob["candidates"])


# ------------------------------------------------------- the papers a USER forbids this search
"""A user's exclusion list is an instruction, not a judgement.

Somebody validating this search against a meta-analysis they already have must be able to
guarantee that the meta-analysis itself is never fetched and never read — otherwise the recall
they measure is measured against a paper the tool was handed. The same door serves excluding your
own prior work, a review you are not counting, or a retraction.

Four properties, one test each:

* it happens **after the dedupe and before the screener**, so a forbidden paper never reaches a
  model, never reaches a publisher and never costs a cent;
* the record says **the person did it**, distinguishably from the screener's own verdict;
* a fragment too short to be safe is **refused**, not quietly turned into a wildcard;
* an entry that matched nothing is **reported**, because the alternative is a user certain they
  excluded something they did not.
"""
#: the first Europe PMC record in the fixture — a real DOI and a real title, written here the way
#: a person would paste them (a resolver prefix, the wrong case, a trailing full stop)
FORBIDDEN_DOI = "https://doi.org/10.1109/JBHI.2025.3644234"
FORBIDDEN_TITLE = ("Investigating the Effectiveness of Haptic Resistive Force Feedback to Improve "
                   "Tremors in Parkinson's Disease")
#: …and the second, retyped the way somebody remembers it: the wrong case, a comma the title does
#: not have, and plain spaces where the real title hyphenates ("8-Week", "Aerobic-Resistance")
FORBIDDEN_FRAGMENT = "EFFECTS of an 8 week, COMBINED aerobic resistance training!"


def prompts_of(client) -> str:
    """Every request the model actually received, as one searchable string.

    The point of an exclusion is that the screener never sees the paper, and "the count went down
    by one" does not prove that — only the prompts do.
    """
    return "\n".join(repr(request) for request in client.provider.requests)


def search_excluding(exclude, tmp_path, *, client=None, max_screened=200):
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    client = client or scripted_client(QUERIES_PAYLOAD, include_everything(11))
    record = run_search(question=QUESTION, transport=transport, client=client,
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=max_screened,
                        max_fetch=0, staging_dir=tmp_path, per_query=PER_QUERY,
                        probe=fake_probe, exclude=exclude)
    return record, client


def test_a_doi_the_user_excluded_never_reaches_the_screener(tmp_path):
    """The whole promise in one test: not screened, not fetched, not billed — and still on record."""
    record, client = search_excluding([FORBIDDEN_DOI], tmp_path)

    forbidden = [c for c in record.candidates if c.excluded_by_user]
    assert len(forbidden) == 1
    paper = forbidden[0]
    assert paper.doi.lower() == "10.1109/jbhi.2025.3644234"
    assert paper.state == "excluded" and paper.keep is False
    assert paper.excluded_by_user == FORBIDDEN_DOI, "the entry the USER typed, verbatim"
    # never read: no verdict was invented for it, and no prompt ever carried it
    assert paper.screen_decision == ""
    assert paper.title[:40] not in prompts_of(client)
    assert FORBIDDEN_TITLE[:40] not in prompts_of(client)
    # never fetched, and the reason on the row is the person's, not a screener's "not wanted"
    assert paper.fetch_attempts == [] and paper.pdf_path == ""
    assert paper.fetch_outcome == EXCLUDED_BY_USER

    counts = counts_of(record.candidates)
    assert counts["excluded_by_user"] == 1
    assert counts["after_dedupe"] == 11, "it is still in the funnel, not deleted from it"
    assert counts["screened"] == 10, "the other ten, and only the other ten"
    assert counts["excluded"] == 0, "no screener ruled anything out here"


def test_a_title_fragment_ignores_case_and_punctuation(tmp_path):
    """A person half-remembers a title; they do not retype its hyphens.

    `normalise_title` is `dedupe`'s own — the same function that decides two index rows are the
    same paper — so an exclusion cannot forbid a paper under one spelling and admit it under
    another.
    """
    record, client = search_excluding([FORBIDDEN_FRAGMENT], tmp_path)

    forbidden = [c for c in record.candidates if c.excluded_by_user]
    assert len(forbidden) == 1
    assert forbidden[0].doi == "10.1177/10538135261434253"
    assert forbidden[0].title[:40] not in prompts_of(client)
    assert counts_of(record.candidates)["excluded_by_user"] == 1
    assert record.exclusions == [{"entry": FORBIDDEN_FRAGMENT, "kind": "title",
                                 "read_as": "effects of an 8 week combined aerobic resistance "
                                            "training",
                                 "matched": 1, "note": ""}]


def test_a_fragment_too_short_to_be_safe_is_refused_not_widened(tmp_path):
    """`tremor` would forbid every tremor paper in a tremor search, silently.

    Containment is what makes a title fragment usable at all, and it is also what makes one
    careless word a wildcard. The refusal is REPORTED — a rule that quietly did not apply is
    worse than one that was never written.
    """
    record, _client = search_excluding(["tremor"], tmp_path)

    assert [c for c in record.candidates if c.excluded_by_user] == []
    counts = counts_of(record.candidates)
    assert counts["excluded_by_user"] == 0 and counts["screened"] == 11
    assert len(record.exclusions) == 1
    entry = record.exclusions[0]
    assert entry["entry"] == "tremor" and entry["kind"] == "refused" and entry["matched"] == 0
    assert str(MIN_TITLE_FRAGMENT) in entry["note"]
    assert any("tremor" in note and "not used" in note for note in record.notes)


def test_an_exclusion_that_matched_nothing_is_never_silent(tmp_path):
    """A mistyped DOI must not read as "this search found none of that paper"."""
    record, _client = search_excluding(["10.9999/nothing-here-matches-this",
                                        FORBIDDEN_DOI], tmp_path)

    rows = {entry["entry"]: entry for entry in record.exclusions}
    assert rows["10.9999/nothing-here-matches-this"]["kind"] == "doi"
    assert rows["10.9999/nothing-here-matches-this"]["matched"] == 0
    assert rows[FORBIDDEN_DOI]["matched"] == 1, "the one that worked still says so"
    assert any("10.9999/nothing-here-matches-this" in note for note in record.notes)
    # …and the entry that worked is not reported as a problem
    assert not any("nothing this search found matched" in note and FORBIDDEN_DOI in note
                   for note in record.notes)


def test_the_user_and_the_screener_are_never_confused_for_each_other(tmp_path):
    """Both land in the `excluded` state; a reader of `search.json` must still tell them apart."""
    verdicts = {"decisions": [{"ref": "1", "decision": "exclude",
                               "reason": "no tremor outcome is reported"}]
                + [{"ref": str(i), "decision": "include", "reason": "a resistance-training trial"}
                   for i in range(2, 11)]}
    record, _client = search_excluding(
        [FORBIDDEN_DOI], tmp_path,
        client=scripted_client(QUERIES_PAYLOAD, verdicts))

    by_user = [c for c in record.candidates if c.excluded_by_user]
    by_model = [c for c in record.candidates if c.screen_decision == "exclude"]
    assert len(by_user) == 1 and len(by_model) == 1
    assert by_user[0].state == by_model[0].state == "excluded"

    # the person's row names the person and the entry that caught it
    assert "you excluded this" in by_user[0].screen_reason.lower()
    assert FORBIDDEN_DOI in by_user[0].screen_reason
    assert by_user[0].screen_decision == ""
    # the screener's row is the screener's own sentence and claims nobody's authority but its own
    assert by_model[0].screen_reason == "no tremor outcome is reported"
    assert by_model[0].excluded_by_user == ""

    # …and the page is told which is which without having to read the prose
    assert project(by_user[0])["excluded_by_you"] is True
    assert project(by_model[0])["excluded_by_you"] is False
    # the two counts are separate rungs and cannot double-count: nothing read the user's paper
    counts = counts_of(record.candidates)
    assert counts["excluded_by_user"] == 1 and counts["excluded"] == 1
    assert counts["screened"] == 10


def test_the_exclusion_list_survives_the_round_trip_to_json(tmp_path):
    """A search whose recall is measured must state what it was forbidden to find."""
    record, _client = search_excluding([FORBIDDEN_DOI, "tremor"], tmp_path)

    blob = json.loads(json.dumps(record.to_json()))
    assert [row["entry"] for row in blob["exclusions"]] == [FORBIDDEN_DOI, "tremor"]
    assert blob["counts"]["excluded_by_user"] == 1
    forbidden = [c for c in blob["candidates"] if c["excluded_by_user"]]
    assert len(forbidden) == 1 and forbidden[0]["excluded_by_user"] == FORBIDDEN_DOI


def test_an_excluded_paper_is_not_fetched_even_when_an_open_copy_is_there(tmp_path):
    """"Never proposed" has to survive the one stage that would otherwise have downloaded it.

    `fetch._record_skips` writes `not_wanted` — "the screener read it and did not want it" — over
    every blank outcome in the `excluded` state. True of a screener's verdict and false of a
    person's, on precisely the row where the difference is the point, so the outcome is set before
    the fetch stage ever sees it. Nothing else in the fetch stage changes.
    """
    transport = index_transport(epmc=load("europepmc_search"), openalex=load("openalex_works"))
    render = [f"https://europepmc.org/articles/{pmcid}?pdf=render" for pmcid in
              ("PMC13317673", "PMC12941259", "PMC12982457", "PMC13065030")]
    serve_pdfs(transport, render)
    for landing in ("https://www.ncbi.nlm.nih.gov/pmc/articles/4366306",
                    "https://www.ncbi.nlm.nih.gov/pmc/articles/4586021"):
        refuse(transport, landing, status=200, outcome="not_a_pdf",
               error="www.ncbi.nlm.nih.gov served text/html, not a PDF")

    # 10.2196/97507 is PMC13317673 — an open-access copy this search would otherwise have kept
    record = run_search(question=QUESTION, transport=transport,
                        client=scripted_client(QUERIES_PAYLOAD, include_everything(10)),
                        model_roles=MODEL_ROLES, budget_usd=1.0, max_screened=200, max_fetch=60,
                        staging_dir=tmp_path / "pdfs", per_query=PER_QUERY, probe=fake_probe,
                        exclude=["10.2196/97507"])

    forbidden = next(c for c in record.candidates if c.excluded_by_user)
    assert forbidden.doi == "10.2196/97507"
    assert forbidden.pdf_path == "" and forbidden.fetch_attempts == []
    assert forbidden.fetch_outcome == EXCLUDED_BY_USER, "not the screener's 'not wanted'"
    # four Europe PMC copies were on offer and this search kept three of them
    assert counts_of(record.candidates)["fetched"] == 3
    assert not any(attempt["url"].endswith("PMC13317673?pdf=render")
                   for c in record.candidates for attempt in c.fetch_attempts)
