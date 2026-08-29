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
                                   EuropePmc, OpenAlex, Unpaywall, links_for, oa_id_urls,
                                   parse_index_names, reconstruct_abstract, source_record)
from canopy.search.models import Candidate
from canopy.search.transport import HttpResponse, MissingSearchFixture, RecordedTransport

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "search"


@pytest.fixture(autouse=True)
def no_contact_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset by default, so a developer's own `CANOPY_CONTACT_EMAIL` cannot change the request
    parameters under the fixtures and turn a green suite red on someone else's machine."""
    monkeypatch.delenv("CANOPY_CONTACT_EMAIL", raising=False)


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


def test_the_index_registry_names_the_discovery_arms_in_order():
    from canopy.search.indices import DISCOVERY_INDEXES

    assert DISCOVERY_INDEXES == ("europepmc", "openalex")
    assert set(INDEXES) == {"europepmc", "openalex", "crossref", "unpaywall"}
    assert INDEXES["unpaywall"].discovery is False


def test_parse_index_names_drops_a_typo_instead_of_refusing_to_start():
    assert parse_index_names("europepmc,crossref") == ["europepmc", "crossref"]
    assert parse_index_names("europmc") == [], "a typo narrows the search; it does not crash it"
    assert parse_index_names("") == list(INDEXES and ("europepmc", "openalex"))
