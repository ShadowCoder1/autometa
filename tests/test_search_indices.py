"""The index adapters (`canopy.search.indices`), against JSON these APIs really sent.

Every fixture under `tests/fixtures/search/` except one is a recording of a live, unauthenticated
GET made on 2026-08-29 — Europe PMC, OpenAlex and Crossref answering real queries. That matters
more here than anywhere else in the search: an adapter tested against JSON its own author invented
tests the author's beliefs about the API, and the two most expensive bugs in this module
(`isOpenAccess` being the STRING `"N"`, and OpenAlex having no `abstract` field at all) are exactly
the beliefs a hand-written fixture would have got wrong.

The one exception is `unpaywall_doi.json`, which is synthetic and says so in its own first key:
Unpaywall's endpoint requires a real contact address and rejects placeholders, and this test suite
had none it was allowed to send.

No test here opens a socket. Every request goes through `RecordedTransport`, which raises
`MissingSearchFixture` for anything unrecorded — so an adapter that quietly changed its URL or its
parameters fails loudly instead of passing against a stale recording.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canopy.search.indices import (INDEXES, MAX_ABSTRACT_WORDS, OA_ID_PREFIXES, Crossref,
                                   EuropePmc, OpenAlex, PubMed, Unpaywall, depth_for, links_for,
                                   oa_id_urls, parse_index_names, reconstruct_abstract,
                                   search_pages, source_record)
from canopy.search.models import Candidate
from canopy.search.transport import HttpResponse, MissingSearchFixture, RecordedTransport

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "search"


@pytest.fixture(autouse=True)
def no_contact_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset by default, so a developer's own `CANOPY_CONTACT_EMAIL` cannot change the request
    parameters under the fixtures and turn a green suite red on someone else's machine."""
    monkeypatch.delenv("CANOPY_CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("CANOPY_OPENALEX_KEY", raising=False)


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def replaying(index, query: str, fixture: str, *, limit: int, status: int = 200,
              outcome: str = "ok") -> RecordedTransport:
    """A transport that answers this adapter's OWN request with a recorded body.

    The URL and params come from `index.request(...)`, not from a string in the test, so the
    fixture key is whatever the adapter actually asks for. Guessing the key here would let the
    adapter drift from the recording and still pass.
    """
    url, params = index.request(query, limit=limit)
    transport = RecordedTransport()
    transport.record(url, HttpResponse(url=url, status=status, outcome=outcome,
                                       body=json.dumps(load(fixture)).encode("utf-8")), params)
    return transport


# ------------------------------------------------------------------------------------ Europe PMC
EPMC_QUERY = "(resistance training) AND parkinson AND tremor"


@pytest.fixture
def epmc_candidates() -> list[Candidate]:
    index = EuropePmc()
    transport = replaying(index, EPMC_QUERY, "europepmc_search", limit=6)
    candidates, record = index.search(transport, EPMC_QUERY, limit=6)
    assert record["n_returned"] == len(candidates) == 6
    return candidates


def test_europepmc_parses_the_fields_the_page_shows(epmc_candidates):
    """Title, authors, year, venue, DOI and abstract, off a real `resultType=core` response."""
    paper = next(c for c in epmc_candidates if c.ids.get("pmcid") == "PMC13317673")
    assert paper.year == 2026
    # `journalInfo.journal.title`, not the top-level `journalTitle` — which is `null` on every one
    # of these live records and would have blanked the venue on every row.
    assert paper.venue == "JMIR research protocols"
    assert paper.doi == "10.2196/97507"
    assert paper.abstract and len(paper.abstract) > 200
    assert paper.ids["pmid"] and paper.ids["europepmc_source"] == "MED"
    assert paper.key.startswith("c") and len(paper.key) == 13


def test_europepmc_author_is_surname_first_so_the_dedupe_can_read_it(epmc_candidates):
    """`authorList.author[].fullName` is `"Mehta N"`, and the last token of that is the INITIAL.

    `dedupe.first_author_surname` splits on a comma and otherwise takes the last token, so passing
    `fullName` straight through would make every Europe PMC surname a single letter — and the
    duplicate guard compares surnames.
    """
    from canopy.search.dedupe import first_author_surname

    paper = next(c for c in epmc_candidates if c.ids.get("pmcid") == "PMC13317673")
    assert paper.authors[0] == "Mehta, N"
    assert first_author_surname(paper) == "mehta"
    assert paper.study_label().startswith("Mehta")


def test_europepmc_open_access_flags_are_strings_and_are_read_as_strings(epmc_candidates):
    """THE trap: `isOpenAccess` is `"Y"`/`"N"`, and `bool("N")` is `True`.

    Two of the six recorded records are closed. A truthiness test would hand both of them a
    `?pdf=render` URL, the fetch stage would collect two 403s, and the record would report a
    publisher refusal for a paper no publisher was ever asked about.
    """
    payload = load("europepmc_search")
    rows = payload["resultList"]["result"]
    closed = [r["id"] for r in rows if r["isOpenAccess"] == "N"]
    assert closed, "the fixture must keep at least one closed record or this test proves nothing"
    assert all(bool(r["isOpenAccess"]) for r in rows), "…and bool() is True for every one of them"

    by_id = {c.ids["europepmc"]: c for c in epmc_candidates}
    for record_id in closed:
        assert "europepmc_render" not in by_id[record_id].ids
        assert not [k for k in by_id[record_id].ids if k.startswith("europepmc_oa_pdf")]
    open_ones = [c for c in epmc_candidates if c.ids["europepmc"] not in closed]
    assert all(c.ids["europepmc_render"].endswith("?pdf=render") for c in open_ones)


def test_europepmc_takes_only_the_oa_pdf_links(epmc_candidates):
    """`availabilityCode == "OA"` AND `documentStyle == "pdf"`.

    Every record in the recording also carries a `S | doi | DOI` entry pointing at doi.org. That is
    a subscription link to a landing page; treating it as an OA PDF route would spend an attempt
    per paper fetching an HTML redirect.
    """
    paper = next(c for c in epmc_candidates if c.ids.get("pmcid") == "PMC12941259")
    pdfs = [v for k, v in paper.ids.items() if k.startswith("europepmc_oa_pdf")]
    assert pdfs == ["https://europepmc.org/articles/PMC12941259?pdf=render"]
    assert not any("doi.org" in url for url in pdfs)


def test_europepmc_licence_is_the_index_s_own_word(epmc_candidates):
    """`cc by`, verbatim. Never inferred from "the download worked"."""
    paper = next(c for c in epmc_candidates if c.ids.get("pmcid") == "PMC13317673")
    assert paper.license == "cc by"
    closed = next(c for c in epmc_candidates if c.doi == "10.1109/jbhi.2025.3644234")
    assert closed.license == ""


def test_europepmc_keeps_a_record_with_no_abstract(monkeypatch):
    """1 of 2 sampled MEDLINE `core` records had no `abstractText` at all (review §C2).

    Such a record is kept and screened on its title. Dropping it would silence a paper over a
    metadata gap, and the screener has its own `screened_on_title_only` flag for exactly this.
    """
    index = EuropePmc()
    query = "parkinson AND FIRST_PDATE:[1978 TO 1986]"
    transport = replaying(index, query, "europepmc_no_abstract", limit=4)
    candidates, record = index.search(transport, query, limit=4)

    assert record["n_returned"] == 4
    without = [c for c in candidates if not c.abstract]
    assert len(without) == 3
    assert all(c.title for c in without), "a record with no abstract still has a title to screen on"


def test_europepmc_records_a_dead_index_instead_of_raising():
    """A 503 is data. The other index's results must survive it."""
    index = EuropePmc()
    url, params = index.request(EPMC_QUERY, limit=6)
    transport = RecordedTransport()
    transport.record(url, HttpResponse(url=url, status=503, outcome="http_error",
                                       error="www.ebi.ac.uk answered HTTP 503"), params)

    candidates, record = index.search(transport, EPMC_QUERY, limit=6)
    assert candidates == []
    assert record["name"] == "europepmc" and record["n_returned"] == 0
    assert "503" in record["error"] and record["outcome"] == "http_error"


def test_an_unrecorded_request_is_loud():
    """The fake never improvises. An adapter that changed its parameters must fail, not pass.

    A fixture double that answered "no results" for an unknown key would turn the exact failure
    this feature exists to prevent — an index silently returning nothing — into a green test.
    """
    with pytest.raises(MissingSearchFixture):
        EuropePmc().search(RecordedTransport(), "anything at all", limit=6)


# --------------------------------------------------------------------------------------- OpenAlex
OA_QUERY = 'parkinson AND (tremor OR bradykinesia) AND "resistance training"'
OA_OA_QUERY = "parkinson AND exercise AND tremor"


def test_openalex_reconstructs_the_inverted_abstract():
    """OpenAlex has NO `abstract` field. `{word: [positions]}` is the whole of it."""
    index = OpenAlex()
    transport = replaying(index, OA_QUERY, "openalex_works", limit=5)
    candidates, record = index.search(transport, OA_QUERY, limit=5)

    with_abstract = [c for c in candidates if c.abstract]
    assert len(with_abstract) == 3, "3 of the 5 recorded works carry an inverted index"
    raw = {w["id"]: w for w in load("openalex_works")["results"]}
    paper = next(c for c in candidates if c.ids["openalex"] == "W2069014105")
    inverted = raw["https://openalex.org/W2069014105"]["abstract_inverted_index"]
    first_word = min(((min(pos), word) for word, pos in inverted.items()))[1]
    assert paper.abstract.startswith(first_word)
    assert record["n_returned"] == 5


def test_openalex_keeps_the_works_whose_abstract_is_null():
    """~31 % of works have `abstract_inverted_index: null`. They are candidates, not casualties."""
    index = OpenAlex()
    transport = replaying(index, OA_QUERY, "openalex_works", limit=5)
    candidates, _ = index.search(transport, OA_QUERY, limit=5)

    blank = [c for c in candidates if not c.abstract]
    assert len(blank) == 2 and all(c.title and c.doi for c in blank)


def test_openalex_normalises_dois_and_strips_url_identifiers():
    """`doi` arrives as `https://doi.org/10.1016/…`; `id` as `https://openalex.org/W…`."""
    index = OpenAlex()
    transport = replaying(index, OA_QUERY, "openalex_works", limit=5)
    candidates, _ = index.search(transport, OA_QUERY, limit=5)

    paper = candidates[0]
    assert paper.doi == "10.1016/j.parkreldis.2009.04.009"
    assert paper.ids["openalex"] == "W2134994784"
    assert paper.ids["pmid"].isdigit(), "the pmid is a URL in the JSON and an id in the record"
    # PMCIDs come from Europe PMC, never from here: `has_pmcid:true` returns 0 works.
    assert "pmcid" not in paper.ids


def test_openalex_carries_the_oa_urls_in_fetch_order():
    """`best_oa_location.pdf_url`, then `locations[].pdf_url`, then `open_access.oa_url`."""
    index = OpenAlex()
    transport = replaying(index, OA_OA_QUERY, "openalex_oa_works", limit=4)
    candidates, _ = index.search(transport, OA_OA_QUERY, limit=4)

    with_pdf = next(c for c in candidates if c.ids.get("openalex_pdf"))
    assert with_pdf.ids["openalex_pdf"] == "https://www.einj.org/upload/pdf/inj-1836226-113.pdf"
    assert with_pdf.license == "cc-by-nc", "per-article licence, as the index stated it"
    # a duplicate location URL is not carried twice: the fetch stage has three attempts and
    # spending two of them on the same URL is spending one of them on nothing
    assert list(dict.fromkeys(u for _, u in oa_id_urls(with_pdf))) == [u for _, u in
                                                                      oa_id_urls(with_pdf)]

    landing = next(c for c in candidates if c.ids.get("openalex_oa_url")
                   and not c.ids.get("openalex_pdf"))
    assert oa_id_urls(landing)[-1][0] == "openalex_oa_url", "the landing page is tried last"


def test_openalex_records_how_the_query_was_parsed():
    """`meta.x_query.oql` is free, machine-readable proof that the query shown is the query run."""
    index = OpenAlex()
    transport = replaying(index, OA_QUERY, "openalex_works", limit=5)
    _, record = index.search(transport, OA_QUERY, limit=5)

    assert "stemmed" in record["parsed_as"] and "bradykinesia or tremor" in record["parsed_as"]
    assert record["total_hits"] == 16


def test_openalex_out_of_budget_leaves_the_search_alive():
    """OpenAlex is METERED: $0.10/day anonymous, $0.001 a search (review §C1).

    When the meter runs out it answers 429. That must produce a note a person can act on and NO
    exception — Europe PMC's results are the search, and losing them because the secondary arm ran
    out of a tenth of a dollar would be absurd.
    """
    index = OpenAlex()
    url, params = index.request(OA_QUERY, limit=5)
    transport = RecordedTransport()
    transport.record(url, HttpResponse(url=url, status=429, outcome="rate_limited",
                                       error="api.openalex.org asked us to slow down"), params)

    candidates, record = index.search(transport, OA_QUERY, limit=5)
    assert candidates == []
    assert record["outcome"] == "rate_limited" and record["n_returned"] == 0
    assert "$0.10" in record["note"] and "Europe PMC" in record["note"]
    assert "midnight UTC" in record["note"]


def test_openalex_uses_the_contact_email_when_there_is_one(monkeypatch):
    """`mailto` is etiquette, not economy — it no longer buys a separate quota. Still sent."""
    monkeypatch.setenv("CANOPY_CONTACT_EMAIL", "someone@example.org")
    _, params = OpenAlex().request(OA_QUERY, limit=5)
    assert params["mailto"] == "someone@example.org"
    monkeypatch.delenv("CANOPY_CONTACT_EMAIL")
    _, bare = OpenAlex().request(OA_QUERY, limit=5)
    assert "mailto" not in bare


def test_openalex_asks_for_title_and_abstract_not_full_text():
    """`search=` searches full text: ~7× the hits and ~7× the screening bill for passing mentions."""
    _, params = OpenAlex().request(OA_QUERY, limit=500)
    assert params["filter"].startswith("title_and_abstract.search:")
    assert "is_retracted:false" in params["filter"]
    assert params["per-page"] == 200, "the API's own maximum, and the cost is flat in per-page"
    # `type` is deliberately absent: reviews are overwhelmingly classified as articles, so
    # filtering on it would drop primary studies and keep reviews.
    assert "type:" not in params["filter"]


# --------------------------------------------------------------------------------------- Crossref
def test_crossref_is_identity_not_discovery():
    """It fills DOIs. It is never asked to propose papers, and it supplies no abstract.

    Crossref's abstracts are raw JATS, present for about a quarter of works, and Crossref itself
    warns some may be subject to publisher copyright. The recorded fixture's first item HAS one,
    and the adapter still drops it — which is the point of the assertion.
    """
    index = Crossref()
    query = "High-intensity resistance training amplifies muscle hypertrophy Parkinson"
    transport = replaying(index, query, "crossref_works", limit=3)
    candidates, record = index.search(transport, query, limit=3)

    assert index.discovery is False
    assert load("crossref_works")["message"]["items"][0]["abstract"].startswith("<jats:title>")
    assert all(c.abstract == "" for c in candidates)
    assert candidates[0].doi == "10.1002/mds.20997"
    assert candidates[0].authors[0] == "Dibble, Leland E."
    assert candidates[0].year == 2006 and candidates[0].venue == "Movement Disorders"
    assert record["n_returned"] == 3


def test_crossref_survives_a_date_with_no_parts():
    """`{"date-parts": [[None]]}` is a real Crossref answer, and `int(None)` is a crash."""
    index = Crossref()
    query = "High-intensity resistance training amplifies muscle hypertrophy Parkinson"
    transport = replaying(index, query, "crossref_works", limit=3)
    candidates, _ = index.search(transport, query, limit=3)

    undated = next(c for c in candidates if c.doi == "10.7717/peerj.17195/table-1")
    assert undated.year is None


# -------------------------------------------------------------------------------------- Unpaywall
def test_unpaywall_is_skipped_with_a_note_when_there_is_no_email():
    """It rejects placeholder addresses, so there is no honest way to call it without one.

    The skip is RECORDED per candidate, so the record shows exactly which papers went unresolved
    for want of one setting — and no request is made, which the empty call log proves.
    """
    transport = RecordedTransport()
    candidate = Candidate(key="c000000000001", doi="10.1097/npt.0000000000000086")

    ids, record = Unpaywall().locations(transport, candidate, email="")
    assert ids == {} and record["skipped"] is True
    assert "CANOPY_CONTACT_EMAIL" in record["note"]
    assert transport.calls == [], "nothing may go on the wire without an address to identify us"


def test_unpaywall_resolves_pdf_urls_best_first():
    """`url_for_pdf` first, then `url` — because `url_for_pdf` is often null and `url` may be a
    landing page, which the content-type gate downstream is what makes safe."""
    index = Unpaywall()
    candidate = Candidate(key="c000000000001", doi="10.1097/npt.0000000000000086")
    url, params = index.request(candidate.doi, email="someone@example.org")
    transport = RecordedTransport()
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                       body=json.dumps(load("unpaywall_doi")).encode()), params)

    ids, record = index.locations(transport, candidate, email="someone@example.org")
    assert list(ids.values())[0] == "https://europepmc.org/articles/PMC4366306?pdf=render"
    assert all(key.startswith("unpaywall_pdf") for key in ids)
    assert record["license"] == "cc-by-nc-nd" and record["is_oa"] is True


def test_unpaywall_says_so_when_a_candidate_has_no_doi():
    transport = RecordedTransport()
    ids, record = Unpaywall().locations(transport, Candidate(key="c000000000001"),
                                        email="someone@example.org")
    assert ids == {} and record["skipped"] is True and "no DOI" in record["note"]
    assert transport.calls == []


# ------------------------------------------------------------------------------- the small helpers
def test_links_are_built_here_because_the_page_may_not_build_them():
    """doi.org first, publisher second AND ONLY WHEN IT DIFFERS.

    A row offering "doi.org" and "publisher" that both point at the same URL is a row that lies
    about having two routes to the paper.
    """
    both = links_for("https://doi.org/10.1002/MDS.20997", "https://journal.example.org/article/1")
    assert [link["label"] for link in both] == ["doi.org", "publisher"]
    assert both[0]["url"] == "https://doi.org/10.1002/mds.20997", "normalised before it is linked"

    same = links_for("10.1002/mds.20997", "https://doi.org/10.1002/mds.20997")
    assert [link["label"] for link in same] == ["doi.org"]

    # a plain-http landing page is not offered: every URL this server hands a browser is https
    assert links_for("10.1002/mds.20997", "http://journal.example.org/1") == [
        {"label": "doi.org", "url": "https://doi.org/10.1002/mds.20997"}]
    assert links_for("not a doi at all") == []
    assert links_for("", "https://journal.example.org/1") == [
        {"label": "publisher", "url": "https://journal.example.org/1"}]


def test_reconstruct_abstract_orders_by_position_and_is_bounded():
    assert reconstruct_abstract({"world": [1], "hello": [0]}) == "hello world"
    assert reconstruct_abstract({"a": [0, 2], "b": [1]}) == "a b a"
    assert reconstruct_abstract(None) == "" and reconstruct_abstract({}) == ""
    # positions we cannot read are skipped, not fatal: the JSON is somebody else's
    assert reconstruct_abstract({"a": [0], "b": ["x"], "c": None}) == "a"
    huge = reconstruct_abstract({"w": list(range(MAX_ABSTRACT_WORDS * 3))})
    assert len(huge.split()) == MAX_ABSTRACT_WORDS


def test_oa_id_urls_is_the_fetch_order_even_after_two_indexes_merge():
    """A paper both indexes proposed carries both routes, and Europe PMC's copy is tried first.

    This is the property that makes `dedupe._merge`'s union of `ids` load-bearing: the fetch order
    of §3.5 survives the merge, whichever index happened to answer first on the day.
    """
    candidate = Candidate(key="c000000000001", ids={
        "openalex_oa_url": "https://example.org/landing",
        "openalex_pdf": "https://publisher.example.org/a.pdf",
        "europepmc_render": "https://europepmc.org/articles/PMC1?pdf=render",
        "unpaywall_pdf": "https://repo.example.org/a.pdf",
        "europepmc_oa_pdf": "https://europepmc.org/articles/PMC1?pdf=render&v=1",
        "openalex_location_pdf_2": "https://mirror.example.org/b.pdf",
        "openalex_location_pdf": "https://mirror.example.org/a.pdf",
        "pmcid": "PMC1", "doi": "10.1/ab"})

    assert [key for key, _ in oa_id_urls(candidate)] == [
        "europepmc_oa_pdf", "europepmc_render", "openalex_pdf",
        "openalex_location_pdf", "openalex_location_pdf_2", "unpaywall_pdf", "openalex_oa_url"]
    assert [key for key, _ in oa_id_urls(candidate)] == list(OA_ID_PREFIXES)[:3] + [
        "openalex_location_pdf", "openalex_location_pdf_2", "unpaywall_pdf", "openalex_oa_url"]
    # plain identifiers are not URLs and are never tried
    assert not any(url in ("PMC1", "10.1/ab") for _, url in oa_id_urls(candidate))


def test_source_record_always_has_the_four_fields():
    """`run.py` reads `n_returned` off every row and `counts` are derived from them."""
    row = source_record("europepmc", "tremor")
    assert set(row) == {"name", "query", "n_returned", "error"}
    assert row["n_returned"] == 0 and row["error"] == ""


def test_the_index_registry_names_the_discovery_arms_in_the_measured_order():
    """PubMed, OpenAlex, Europe PMC: by ranking quality measured on the first answer key (design
    04 §A: 20, 18, 8 of 23 in the top thousand), not by anyone's idea of a field."""
    from canopy.search.indices import DISCOVERY_INDEXES

    assert DISCOVERY_INDEXES == ("pubmed", "openalex", "europepmc")
    assert set(INDEXES) == {"pubmed", "europepmc", "openalex", "crossref", "unpaywall"}
    assert INDEXES["unpaywall"].discovery is False


def test_parse_index_names_drops_a_typo_instead_of_refusing_to_start():
    assert parse_index_names("europepmc,crossref") == ["europepmc", "crossref"]
    assert parse_index_names("europmc") == [], "a typo narrows the search; it does not crash it"
    assert parse_index_names("") == ["pubmed", "openalex", "europepmc"]


# ------------------------------------------------------------------- the author manuscripts (§7a)
def test_europepmc_offers_the_free_but_not_open_access_routes():
    """The four NIH author manuscripts of the first answer key: `isOpenAccess: N`, `inEPMC: Y`,
    `hasPDF: Y`, and an explicit `availabilityCode: F` PDF route. The old rule (`OA` only, render
    on `isOpenAccess`) offered none of them and the four were reported as fetch failures."""
    index = EuropePmc()
    query = "SRC:MED AND (EXT_ID:11 OR EXT_ID:12)"
    transport = replaying(index, query, "europepmc_author_manuscripts", limit=4)
    candidates, record = index.search(transport, query, limit=4)

    assert record["n_returned"] == 4
    for paper in candidates:
        assert paper.ids["europepmc_oa_flag"] == "N", "the flag is kept: it is a fact"
        assert paper.ids["europepmc_render"].endswith("?pdf=render")
        assert paper.ids["europepmc_oa_pdf"].endswith("?pdf=render"), "the F route"
        assert [url for _, url in oa_id_urls(paper)] == [paper.ids["europepmc_render"]], \
            "one URL, not the same one twice"


# ----------------------------------------------------------------------------- PubMed (§2)
ESEARCH = {"header": {"type": "esearch", "version": "0.3"},
           "esearchresult": {"count": "3", "retmax": "3", "retstart": "0",
                             "idlist": ["41752351", "42378169", "99999999"],
                             "translationset": [],
                             "querytranslation": '("dominant"[All Fields] OR "dominance"[All '
                                                 'Fields]) AND "adaptation"[All Fields]'}}


def pubmed_transport(query: str, *, limit: int = 1000) -> RecordedTransport:
    """esearch answering three ids, two of which Europe PMC holds (the recorded fixture)."""
    index = PubMed()
    url, params = index.request(query, limit=limit)
    transport = RecordedTransport()
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                       body=json.dumps(ESEARCH).encode()), params)
    hurl, hparams = EuropePmc().hydrate_request(ESEARCH["esearchresult"]["idlist"])
    transport.record(hurl, HttpResponse(url=hurl, status=200, outcome="ok",
                                        body=json.dumps(load("europepmc_search")).encode()),
                     hparams)
    return transport


def test_pubmed_asks_esearch_by_relevance_with_tool_and_email(monkeypatch):
    url, params = PubMed().request("(a OR b) AND c", limit=1000)
    assert url.startswith("https://eutils.ncbi.nlm.nih.gov/")
    assert params["db"] == "pubmed" and params["sort"] == "relevance"
    assert params["retmax"] == 1000 and params["retstart"] == 0 and params["retmode"] == "json"
    assert params["tool"] == "canopy-meta" and "email" not in params
    monkeypatch.setenv("CANOPY_CONTACT_EMAIL", "someone@example.org")
    assert PubMed().request("x", limit=5)[1]["email"] == "someone@example.org"
    assert PubMed().request("x", limit=5, cursor="1000")[1]["retstart"] == 1000
    assert PubMed().request("x", limit=50000)[1]["retmax"] == PubMed.MAX_PAGE


def test_pubmed_hydrates_its_ids_through_europe_pmc_in_pubmed_order():
    """esearch answers ids only. Each page is hydrated in batches of a hundred via
    `SRC:MED AND (EXT_ID:… OR …)`; the candidates come back in PubMed's order, `found_by`
    says `pubmed`, and an id Europe PMC does not hold is counted, never invented."""
    query = "(dominant OR dominance) AND adaptation"
    transport = pubmed_transport(query)
    candidates, record = PubMed().search(transport, query, limit=1000)

    assert [c.ids["pmid"] for c in candidates] == ["41752351", "42378169"]
    assert all(c.found_by == ["pubmed"] for c in candidates)
    assert candidates[0].abstract and candidates[0].ids["pmcid"] == "PMC12941259"
    assert record["total_hits"] == 3 and record["n_unhydrated"] == 1
    assert record["n_hydration_requests"] == 1
    assert '"dominance"[All Fields]' in record["parsed_as"], "the term mapping is on the record"
    hydration = [c for c in transport.calls if "EXT_ID" in str(c["params"].get("query", ""))]
    assert len(hydration) == 1 and hydration[0]["context"]["form"] == "hydrate"
    assert hydration[0]["context"]["exact"] is True, "a hydration batch never replays by tuple"


def test_search_pages_writes_the_entry_position_and_one_source_row():
    query = "(dominant OR dominance) AND adaptation"
    transport = pubmed_transport(query)
    row = {"query_id": "Q1", "index": "pubmed", "form": "", "text": query}
    candidates, source = search_pages(PubMed(), transport, row, depth=1000)

    assert [c.ranks["pubmed:Q1:"] for c in candidates] == [1, 2]
    assert source["name"] == "pubmed" and source["query_id"] == "Q1" and source["form"] == ""
    assert source["pages"] == 1 and source["n_returned"] == 2 and source["total_hits"] == 3
    assert source["more_available"] is False and source["retry_after_honoured"] is False
    assert source["n_hydration_requests"] == 1 and source["error"] == ""


# ---------------------------------------------------------------------- OpenAlex forms and key
def test_openalex_forms_cursor_and_key(monkeypatch):
    """`ta` is the title-and-abstract filter; `ft` is the same string as `search=`; both sorted
    by relevance and cursor-paged; the key rides as `api_key` only when set — and never into a
    fixture (`transport.IDENTITY_PARAMS`)."""
    _, ta = OpenAlex().request("a AND b", limit=200)
    assert ta["filter"] == "title_and_abstract.search:a AND b,is_retracted:false"
    assert "search" not in ta and ta["sort"] == "relevance_score:desc" and ta["cursor"] == "*"
    assert "referenced_works" in ta["select"] and "cited_by_count" in ta["select"]
    _, ft = OpenAlex().request("a AND b", limit=200, form="ft", cursor="IlsxNDIuNDU3")
    assert ft["search"] == "a AND b" and ft["filter"] == "is_retracted:false"
    assert ft["cursor"] == "IlsxNDIuNDU3"
    assert "api_key" not in ta
    monkeypatch.setenv("CANOPY_OPENALEX_KEY", "not-a-real-key")
    _, keyed = OpenAlex().request("a AND b", limit=200)
    assert keyed["api_key"] == "not-a-real-key"
    from canopy.search.transport import fixture_key

    assert fixture_key(OpenAlex.URL, keyed) == fixture_key(OpenAlex.URL, ta)
    _, count = OpenAlex().count_request("a AND b")
    assert count["per-page"] == 1 and count["select"] == "id" and "cursor" not in count
    _, ecount = EuropePmc().count_request("a AND b")
    assert ecount["pageSize"] == 1 and ecount["resultType"] == "lite"


def test_openalex_pages_follow_the_cursor_and_stop_when_it_runs_out(monkeypatch):
    """Ten rows a page: a full page with a cursor is followed, a short page ends the paging,
    and `depth` stops it first when it is smaller."""
    monkeypatch.setattr(OpenAlex, "MAX_PAGE", 10)
    index = OpenAlex()
    base = load("openalex_works")
    template = base["results"][0]
    first = {"meta": {"count": 11, "next_cursor": "cursor-2"},
             "results": [dict(template, id=f"https://openalex.org/W{i}",
                              doi=f"https://doi.org/10.1000/w{i}") for i in range(1, 11)]}
    second = {"meta": {"count": 11, "next_cursor": "cursor-3"},
              "results": [dict(template, id="https://openalex.org/W11",
                               doi="https://doi.org/10.1000/w11")]}
    transport = RecordedTransport()
    # each page asks for exactly what is still wanted, so the fixtures sit at those keys:
    # 10 on the first page; 2 on the second when the depth is 12; 5 when the depth is 5
    for cursor, limit, body in (("*", 10, first), ("cursor-2", 2, second), ("*", 5, first)):
        url, params = index.request(OA_QUERY, limit=limit, cursor=cursor)
        transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                           body=json.dumps(body).encode()), params)
    row = {"query_id": "Q1", "index": "openalex", "form": "ta", "text": OA_QUERY}
    candidates, source = search_pages(index, transport, row, depth=12)

    assert len(candidates) == 11 and source["pages"] == 2
    assert [c.ranks["openalex:Q1:ta"] for c in candidates] == list(range(1, 12))
    assert source["more_available"] is False, "the second page was short: that is the end"
    assert [c["context"]["page"] for c in transport.calls] == [1, 2]
    # …and `depth` stops the paging before the cursor does
    transport.calls.clear()
    candidates, source = search_pages(index, transport, row, depth=5)
    assert len(candidates) == 5 and source["pages"] == 1 and source["more_available"] is True
    candidates, source = search_pages(index, transport, row, depth=10)
    assert len(candidates) == 10 and source["pages"] == 1 and source["more_available"] is True


def test_a_429_with_retry_after_is_honoured_once_per_page():
    """Sleep what the server asked (at most 30 s), ask again once; the row says it happened."""
    index = OpenAlex()
    url, params = index.request(OA_QUERY, limit=5)
    transport = RecordedTransport()
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                       body=json.dumps(load("openalex_works")).encode()), params)
    key = next(iter(transport.responses))
    transport.sequences[key] = [
        HttpResponse(url=url, status=429, outcome="rate_limited", retry_after=12.0,
                     error="slow down"),
        transport.responses[key]]
    slept: list[float] = []
    row = {"query_id": "Q1", "index": "openalex", "form": "ta", "text": OA_QUERY}
    candidates, source = search_pages(index, transport, row, depth=5, sleep=slept.append)

    assert slept == [12.0] and len(candidates) == 5
    assert source["retry_after_honoured"] is True and source["error"] == ""

    # a second 429 on the same page is the row's error, not a second sleep
    transport.sequences[key] = [transport.sequences[key][0], transport.sequences[key][0]]
    transport._cursor.clear()
    slept.clear()
    candidates, source = search_pages(index, transport, row, depth=5, sleep=slept.append)
    assert slept == [12.0] and candidates == [] and source["outcome"] == "rate_limited"


def test_the_operator_throttle_is_named_and_the_key_is_the_advice():
    """Three 429s, three sentences: the anonymous operator limit, a busy cluster, the meter."""
    operators = ("Your query uses 29 boolean operators (OR/AND/NOT). Broad boolean searches are "
                 "heavy for our search cluster, so queries with more than 5 operators are limited")
    note = OpenAlex.degradation_note("rate_limited", 429, operators)
    assert "five AND/OR operators" in note and "CANOPY_OPENALEX_KEY" in note
    assert "PubMed and Europe PMC" in note
    busy = OpenAlex.degradation_note("rate_limited", 429, "Please retry in 12s")
    assert "retry" in busy.lower() and "CANOPY_OPENALEX_KEY" in busy
    meter = OpenAlex.degradation_note("rate_limited", 429, "")
    assert "$0.10" in meter and "midnight UTC" in meter


def test_depth_scales_every_index_and_europe_pmc_goes_three_times_deeper():
    assert (depth_for("pubmed"), depth_for("openalex"), depth_for("europepmc")) == (1000, 1000,
                                                                                     3000)
    assert (depth_for("pubmed", 200), depth_for("europepmc", 200)) == (200, 600)


def test_a_name_that_does_not_resolve_is_retried_up_the_ladder_and_survives():
    """The essential-tremor run lost PubMed entirely: `gaierror` on both strings, and the old
    branch retried only `timeout`/`network_error` — once, after 5 s. A DNS failure or a 5xx
    is transient like the others, and a WiFi blip of a minute must cost a pause, not an index."""
    from canopy.search import indices as mod

    class Flaky:
        name = "pubmed"
        discovery = True
        MAX_PAGE = 1000

        def __init__(self, failures):
            self.answers = list(failures)

        def page(self, transport, text, *, limit, cursor, form, timeout, context):
            if self.answers:
                outcome = self.answers.pop(0)
                status = 503 if outcome == "http_error" else None
                return [], {"outcome": outcome, "status": status, "error": outcome}
            cand = Candidate(key="c1", title="found after the blip", authors=[], year=2020)
            return [cand], {"outcome": "ok", "status": 200, "next_cursor": None}

    slept: list[float] = []
    row = {"query_id": "Q1", "index": "pubmed", "form": "", "text": "x"}

    # two DNS failures then a 5xx, then success: three pauses up the ladder, index kept
    index = Flaky(["dns_error", "dns_error", "http_error"])
    candidates, source = search_pages(index, RecordedTransport(), row, depth=10,
                                      sleep=slept.append)
    assert [c.title for c in candidates] == ["found after the blip"]
    assert source["retried"] == 3 and slept == list(mod.TRANSIENT_PAUSES_S)
    assert not source.get("error")

    # four failures exhaust the ladder: the row records the failure and does not loop forever
    slept.clear()
    index = Flaky(["dns_error"] * 4)
    candidates, source = search_pages(index, RecordedTransport(), row, depth=10,
                                      sleep=slept.append)
    assert candidates == [] and source["retried"] == 3 and source["error"] == "dns_error"

    # a failure the query caused (a 4xx) is not retried at all
    slept.clear()
    index = Flaky(["http_error"])
    index.page = lambda *a, **k: ([], {"outcome": "http_error", "status": 400, "error": "bad"})
    candidates, source = search_pages(index, RecordedTransport(), row, depth=10,
                                      sleep=slept.append)
    assert source["retried"] == 0 and slept == []
