"""The scholarly indexes, one adapter each, all of them the same two lines wide.

Every adapter is `search(transport, query, *, limit) -> (candidates, record)`. The second half of
that pair is the point: an index that answered nothing, answered slowly, or refused outright is
DATA — a dict with its name, the query it was asked, how many rows came back and its error — and
never an exception. A search that dies because one index had a bad afternoon is a search that
cannot be trusted to have looked, and the user would have no way to tell the two apart.

WHO IS HERE, AND WHY IN THIS ORDER
----------------------------------
* **Europe PMC** — first, and the reason is REQUEST ECONOMY, not subject matter. It is unmetered
  and keyless, so asking it costs nothing and cannot exhaust anything. It is deliberately *not*
  called "the primary spine because it is the right index for medicine": Canopy's own product rule
  is that nothing here knows a research field (`server/app.py`), and hard-coding a field's index as
  the spine would break it. Which index actually earned its place is a MEASURED fact — the record
  carries `unique_contributed` per index, and a reader can see for their own question whether the
  ordering paid off (review §C4).
* **OpenAlex** — second, and metered since February 2026: $0.10 of budget a day anonymously and
  $0.001 per `.search` request, so roughly 100 requests a day (review §C1, verified live). It is
  the cross-domain recall arm, and it must DEGRADE: when the budget is gone, or the service is
  down, the adapter returns no candidates and a note that says so in words, and the search carries
  on with what Europe PMC found.
* **Crossref** — metadata and DOI fill, never discovery. Its abstracts are raw JATS, present for
  about a quarter of works, and carry a copyright warning; using it to screen would be using the
  wrong tool badly. `discovery = False` says so in code, and `run.py` reads that flag.
* **Unpaywall** — OA resolution only, and only for a candidate that already has a DOI and no OA
  URL yet. Its `email` parameter is mandatory and it rejects placeholder addresses (the repo
  already learned this: `validation/papers_oa/fetch_oa_papers.py:53-58`), so with no
  `CANOPY_CONTACT_EMAIL` the adapter is SKIPPED WITH A NOTE rather than called with an invented
  address. Inventing one would be lying to a service that asked us who we are.

WHERE THE OA URLS GO
--------------------
`Candidate` has no field for "the PDF URLs an index knows about", and `models.py` is the contract
and not ours to widen. So they ride in `Candidate.ids` under the prefixed keys in `OA_ID_PREFIXES`,
in the order `fetch.py` will try them (§3.5). Two properties make this the right hiding place
rather than a hack: `dedupe._merge` unions `ids` across the records it merges, so a paper found by
both indexes arrives at the fetch stage carrying Europe PMC's route AND OpenAlex's; and `ids` is
already the audit trail, so every URL the fetcher will try is visible in `search.json` before it is
tried.

`links`, by contrast, is for a HUMAN: doi.org first, the publisher's landing page second and only
when it is a different URL. The page cannot build these — its own test forbids the literal
`https://` anywhere in the SPA — so every outbound link a user can click is one this module wrote.

THE TWO TRAPS THIS MODULE EXISTS TO NOT FALL INTO
-------------------------------------------------
1. **Europe PMC's OA flags are the STRINGS `"Y"` and `"N"`.** `if record["isOpenAccess"]:` is
   `True` for a closed paper. Every read of them goes through `_is_yes`, and
   `tests/test_search_indices.py` pins a real recorded `"N"` record against it.
2. **OpenAlex has no `abstract` field at all** — only `abstract_inverted_index`, which is `null`
   for roughly a third of works. A record with no abstract is KEPT and screened on its title
   (`screened_on_title_only`), never dropped: "we could not read this one's abstract" and "this
   paper does not exist" are different sentences.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from .dedupe import key_for, normalise_doi
from .models import Candidate
from .transport import DEFAULT_TIMEOUT, SearchTransport

__all__ = [
    "EuropePmc", "OpenAlex", "Crossref", "Unpaywall", "INDEXES", "DISCOVERY_INDEXES",
    "OA_ID_PREFIXES", "MAX_ABSTRACT_WORDS", "contact_email", "reconstruct_abstract",
    "links_for", "source_record", "oa_id_urls", "parse_index_names",
]

#: the OA URL id-keys `fetch.py` reads, BEST FIRST — this tuple *is* the fetch order of §3.5, and
#: the order is evidence-led: Europe PMC's `?pdf=render` served a real PDF on a live probe while
#: about half of publisher links 403'd a bot, so the index's own copy is tried before the
#: publisher's. `openalex_oa_url` is last because it is a URL, not necessarily a PDF (one live
#: probe returned an HTML landing page), and the content-type gate is what makes that safe.
OA_ID_PREFIXES: tuple[str, ...] = (
    "europepmc_oa_pdf",        # fullTextUrlList entry with availabilityCode OA + documentStyle pdf
    "europepmc_render",        # https://europepmc.org/articles/<PMCID>?pdf=render
    "openalex_pdf",            # best_oa_location.pdf_url
    "openalex_location_pdf",   # locations[].pdf_url, in OpenAlex's own order
    "unpaywall_pdf",           # best_oa_location.url_for_pdf, then oa_locations[], then .url
    "openalex_oa_url",         # open_access.oa_url — often a landing page
)

#: a defensive bound on abstract reconstruction. OpenAlex's inverted index is attacker-adjacent
#: data (we did not write it and do not bound it), the screener truncates at 1,800 characters
#: anyway, and a record with a million positions should cost one long abstract, not the machine.
MAX_ABSTRACT_WORDS = 4000

#: how many `locations[].pdf_url` entries are worth carrying. `MAX_ATTEMPTS` in `fetch.py` is 3, so
#: carrying twenty would be twenty rows of `search.json` that can never be tried.
MAX_LOCATION_PDFS = 3


def contact_email() -> str:
    """The address we identify ourselves with, or `""`.

    Read on every call rather than captured at import so a test can set and unset it, and so a
    server that gains the variable does not need restarting to become polite.
    """
    return os.environ.get("CANOPY_CONTACT_EMAIL", "").strip()


# --------------------------------------------------------------------------------- small helpers
def _is_yes(value: Any) -> bool:
    """Europe PMC's `"Y"`/`"N"` flags, read as the strings they are.

    THE trap of this module: `bool("N")` is `True`, so a truthiness test marks every closed paper
    open access and the fetch stage then reports the resulting 403s as publisher refusals. A real
    recorded `"N"` record is pinned against this in the tests.
    """
    return str(value or "").strip().upper() in ("Y", "YES", "TRUE")


def _text(value: Any, limit: int = 4000) -> str:
    """A trimmed string from whatever an index put in the field, never `None`, never unbounded."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    return " ".join(str(value).split())[:limit]


def _year(value: Any) -> int | None:
    """A publication year, or None. Europe PMC sends `"2026"`; OpenAlex sends `2026`; Crossref
    sends `{"date-parts": [[None]]}` for a record with no date, which must not raise."""
    try:
        year = int(str(value).strip()[:4])
    except (TypeError, ValueError):
        return None
    # a four-digit sanity window: a "year" of 12 or 20260 is a parse gone wrong, and a candidate
    # with a wrong year is worse than one with no year (dedupe buckets on it)
    return year if 1500 <= year <= 2200 else None


def _bare(value: Any) -> str:
    """`https://openalex.org/W123` → `W123`; `https://pubmed…/19497777` → `19497777`.

    OpenAlex returns every identifier as a URL. The record should hold the id, because that is
    what a person pastes into the other tool and what a re-search matches on.
    """
    text = str(value or "").strip()
    return text.rstrip("/").rsplit("/", 1)[-1] if text else ""


def links_for(doi: str, landing: str = "") -> list[dict[str, str]]:
    """The links a HUMAN may follow: doi.org first, the publisher's page second when it differs.

    Second *and only when it differs*, because a row offering "doi.org" and "publisher" that both
    point at `https://doi.org/10.1/x` is a row that lies about having two routes to the paper.
    The page renders at most two, in this order, and cannot construct either itself.
    """
    links: list[dict[str, str]] = []
    clean = normalise_doi(doi)
    if clean:
        links.append({"label": "doi.org", "url": f"https://doi.org/{clean}"})
    page = str(landing or "").strip()
    if page.startswith("https://") and page not in {link["url"] for link in links}:
        links.append({"label": "publisher", "url": page})
    return links


def source_record(name: str, query: str, *, n_returned: int = 0, error: str = "",
                  **extra: Any) -> dict[str, Any]:
    """The per-source row of `SearchRecord.sources`. Four fields always, whatever happened.

    `unique_contributed` is deliberately NOT set here: an adapter cannot know what the other
    adapters found. `run.py` fills it after the dedupe, which is the only place the answer exists.
    """
    row: dict[str, Any] = {"name": name, "query": query, "n_returned": int(n_returned),
                           "error": str(error or "")}
    row.update(extra)
    return row


def oa_id_urls(candidate: Candidate) -> list[tuple[str, str]]:
    """`[(id_key, url), …]` — every OA URL an index gave this candidate, best route first.

    The order is `OA_ID_PREFIXES`, and within one prefix it is the numeric suffix the adapter
    wrote (`openalex_location_pdf`, `openalex_location_pdf_2`, …), so "OpenAlex's own order"
    survives the trip through `ids` and through a dedupe merge.
    """
    out: list[tuple[str, str]] = []
    for prefix in OA_ID_PREFIXES:
        matching = [(key, value) for key, value in candidate.ids.items()
                    if key == prefix or key.startswith(prefix + "_")]
        matching.sort(key=lambda pair: (len(pair[0]), pair[0]))
        for key, value in matching:
            url = str(value or "").strip()
            if url and url not in {seen for _, seen in out}:
                out.append((key, url))
    return out


def _finish(candidate: Candidate) -> Candidate:
    """Give a freshly-parsed candidate its key, from the same identity the dedupe groups on.

    Through `dedupe.key_for` rather than a local hash so the two can never drift: if the key said
    one thing about a paper's identity and the dedupe said another, the same paper found again in a
    re-search would arrive under a key the page had never seen.
    """
    candidate.key = key_for(candidate)
    return candidate


def reconstruct_abstract(inverted: Mapping[str, Any] | None) -> str:
    """OpenAlex's `abstract_inverted_index` — `{word: [positions]}` — back into a sentence.

    `null` for roughly a third of works, and that is not an error: the empty string means the
    screener reads the title alone and says so on the row.
    """
    if not inverted or not isinstance(inverted, Mapping):
        return ""
    slots: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, (list, tuple)):
            continue
        for position in positions:
            try:
                slots.append((int(position), str(word)))
            except (TypeError, ValueError):
                continue
            if len(slots) >= MAX_ABSTRACT_WORDS:
                break
        if len(slots) >= MAX_ABSTRACT_WORDS:
            break
    slots.sort()
    return " ".join(word for _, word in slots).strip()


# ------------------------------------------------------------------------------------ Europe PMC
class EuropePmc:
    """MEDLINE + PMC + preprints, unmetered and keyless. `resultType=core` or nothing.

    `core` is not a preference: `lite` (the default) omits `abstractText` and `fullTextUrlList`
    entirely, and those two fields are the whole reason to call this index — one is what the
    screener reads and the other is the OA route that actually works.
    """

    name = "europepmc"
    discovery = True
    URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    MAX_PAGE = 1000                       # the API's own maximum

    def request(self, query: str, *, limit: int) -> tuple[str, dict[str, Any]]:
        """The URL and params for one call. Exposed so a test can record a fixture at exactly the
        key the adapter will ask for, rather than at a key a test author guessed."""
        return self.URL, {
            "query": query,
            "format": "json",
            "pageSize": max(1, min(int(limit), self.MAX_PAGE)),
            "resultType": "core",
            # offset paging was removed in 2016; `cursorMark=*` is the only first page there is
            "cursorMark": "*",
        }

    def search(self, transport: SearchTransport, query: str, *, limit: int = 200,
               timeout: float = DEFAULT_TIMEOUT) -> tuple[list[Candidate], dict[str, Any]]:
        url, params = self.request(query, limit=limit)
        response = transport.get_json(url, params=params, timeout=timeout)
        if not response.ok:
            return [], source_record(self.name, query, error=response.error or response.outcome,
                                     outcome=response.outcome, status=response.status,
                                     seconds=round(response.seconds, 3))
        payload = response.json()
        if not isinstance(payload, dict):
            return [], source_record(self.name, query,
                                     error="Europe PMC answered with something that was not JSON",
                                     outcome="unreadable", status=response.status)
        rows = ((payload.get("resultList") or {}).get("result") or [])
        candidates = [self.parse(row) for row in rows if isinstance(row, dict)]
        return candidates, source_record(
            self.name, query, n_returned=len(candidates), outcome=response.outcome,
            status=response.status, seconds=round(response.seconds, 3),
            total_hits=payload.get("hitCount"),
            # a cursor left over means we asked for fewer rows than exist. Recorded so "we found 40"
            # is never mistaken for "there are 40".
            more_available=bool(payload.get("nextCursorMark") and len(rows) >= params["pageSize"]))

    def parse(self, row: Mapping[str, Any]) -> Candidate:
        doi = normalise_doi(row.get("doi"))
        pmcid = _text(row.get("pmcid"), 32)
        ids: dict[str, str] = {}
        for key, value in (("europepmc", _text(row.get("id"), 32)),
                           ("europepmc_source", _text(row.get("source"), 8)),
                           ("pmid", _text(row.get("pmid"), 32)),
                           ("pmcid", pmcid), ("doi", doi)):
            if value:
                ids[key] = value

        # the OA routes, in the order §3.5 tries them. The rule is the index's own, and the same
        # one `validation/papers_oa/fetch_oa_papers.py:397` already uses: availabilityCode "OA"
        # AND documentStyle "pdf". `availability` ("Open access") is prose and is not parsed.
        n_pdf = 0
        for entry in ((row.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
            if not isinstance(entry, dict):
                continue
            if (str(entry.get("availabilityCode") or "").upper() == "OA"
                    and str(entry.get("documentStyle") or "").lower() == "pdf"):
                url = _text(entry.get("url"), 600)
                if url.startswith("http"):
                    n_pdf += 1
                    ids["europepmc_oa_pdf" + (f"_{n_pdf}" if n_pdf > 1 else "")] = url
        # the render route, from the flags — and `_is_yes`, because these are the strings "Y"/"N"
        if pmcid and _is_yes(row.get("isOpenAccess")) and _is_yes(row.get("hasPDF")):
            ids["europepmc_render"] = f"https://europepmc.org/articles/{pmcid}?pdf=render"

        landing = ""
        for entry in ((row.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
            if (isinstance(entry, dict) and str(entry.get("documentStyle") or "").lower() == "html"
                    and not landing):
                landing = _text(entry.get("url"), 600)

        return _finish(Candidate(
            key="", title=_text(row.get("title"), 600), authors=_epmc_authors(row),
            year=_year(row.get("pubYear")),
            # `journalInfo.journal.title`, NOT the top-level `journalTitle` — which comes back
            # `null` on live records (verified) and would blank every venue on the page.
            venue=_text(((row.get("journalInfo") or {}).get("journal") or {}).get("title"), 200),
            doi=doi, abstract=_text(row.get("abstractText"), 8000),
            found_by=[EuropePmc.name], ids=ids,
            # per-article, as the index stated it. Europe PMC says out loud that the OA subset's
            # licence terms are NOT identical across articles, so assuming CC-BY would be inventing
            # a permission on the publisher's behalf.
            license=_text(row.get("license"), 60),
            links=links_for(doi, landing)))


def _epmc_authors(row: Mapping[str, Any]) -> list[str]:
    """`["Mehta, N", …]` — surname first, comma, because that is what the rest of the search reads.

    `authorList.author[].fullName` is `"Mehta N"` with no comma, and `dedupe.first_author_surname`
    would read the *initial* as the surname. Rebuilding `lastName, initials` fixes that and makes
    Europe PMC's names the same shape as Crossref's. (A name like `van der Kolk` still normalises
    differently across indexes; that costs a duplicate PROPOSAL the user can accept, never a merge.)
    """
    authors: list[str] = []
    for entry in ((row.get("authorList") or {}).get("author") or []):
        if not isinstance(entry, dict):
            continue
        surname = _text(entry.get("lastName"), 120)
        initials = _text(entry.get("initials") or entry.get("firstName"), 60)
        authors.append(f"{surname}, {initials}" if surname and initials
                       else surname or _text(entry.get("fullName"), 160))
    if authors:
        return [a for a in authors if a][:60]
    # `authorString` is "Mehta N, Munoz MJ, …" — comma-separated between authors, so splitting on
    # the comma gives whole names and NOT "surname, initials". Kept as a last resort.
    return [part.strip() for part in _text(row.get("authorString"), 2000).split(",")
            if part.strip()][:60]


# --------------------------------------------------------------------------------------- OpenAlex
class OpenAlex:
    """Cross-domain recall — and METERED, which is the fact that shapes this class.

    $0.10 of budget a day anonymously, $0.001 per `.search` request (verified live, review §C1), so
    about 100 requests a day. The cost is flat in `per-page`, so one page of 200 costs exactly what
    one page of 1 does — hence one request per query and never a paginator.

    When the meter runs out the API answers 429, and 429 is `rate_limited` in the transport's
    vocabulary. This adapter turns that into a NOTE IN PLAIN WORDS and no candidates. It must never
    raise: the search still has Europe PMC, and a user whose OpenAlex budget expired should get a
    smaller search with an explanation, not a failure.
    """

    name = "openalex"
    discovery = True
    URL = "https://api.openalex.org/works"
    MAX_PAGE = 200
    #: top-level only — `best_oa_location.pdf_url` in `select` is REJECTED by the API; you select
    #: the whole object. (`filter` does take dotted paths, which is the confusing part.)
    SELECT = ("id,doi,ids,display_name,publication_year,type,language,open_access,"
              "best_oa_location,locations,primary_location,abstract_inverted_index,authorships")

    def request(self, query: str, *, limit: int) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {
            # `title_and_abstract.search`, not `search`: the latter searches FULL TEXT, which
            # measured ~7× the hits and would be ~7× the screening bill for passing mentions.
            # `is_retracted:false` is a filter, not a search term, and costs nothing extra.
            "filter": f"title_and_abstract.search:{query},is_retracted:false",
            "per-page": max(1, min(int(limit), self.MAX_PAGE)),
            "select": self.SELECT,
            "cursor": "*",
        }
        email = contact_email()
        if email:
            # etiquette rather than economy: since Feb 2026 `mailto` no longer buys a separate
            # quota, it decrements the same counter. It still tells them who to shout at.
            params["mailto"] = email
        return self.URL, params

    def search(self, transport: SearchTransport, query: str, *, limit: int = 200,
               timeout: float = DEFAULT_TIMEOUT) -> tuple[list[Candidate], dict[str, Any]]:
        url, params = self.request(query, limit=limit)
        response = transport.get_json(url, params=params, timeout=timeout)
        if not response.ok:
            return [], source_record(self.name, query, error=response.error or response.outcome,
                                     outcome=response.outcome, status=response.status,
                                     seconds=round(response.seconds, 3),
                                     note=self.degradation_note(response.outcome, response.status))
        payload = response.json()
        if not isinstance(payload, dict):
            return [], source_record(
                self.name, query, error="OpenAlex answered with something that was not JSON",
                outcome="unreadable", status=response.status,
                note="OpenAlex could not be read this time, so this query's results are Europe "
                     "PMC's alone")
        meta = payload.get("meta") or {}
        rows = payload.get("results") or []
        candidates = [self.parse(row) for row in rows if isinstance(row, dict)]
        return candidates, source_record(
            self.name, query, n_returned=len(candidates), outcome=response.outcome,
            status=response.status, seconds=round(response.seconds, 3),
            total_hits=meta.get("count"),
            # free, machine-readable proof that the query the user is shown is the query that ran.
            # A silent misparse (a stray parenthesis turning AND into OR) is otherwise invisible.
            parsed_as=_text((meta.get("x_query") or {}).get("oql"), 1000),
            cost_usd=meta.get("cost_usd"))

    @staticmethod
    def degradation_note(outcome: str, status: int = 0) -> str:
        """The sentence a user reads when OpenAlex did not answer. Never a stack trace.

        The 429 wording is the one that matters: OpenAlex's meter and OpenAlex being busy look
        identical from here, so the note says both and names the thing the user can act on.
        """
        if outcome == "rate_limited" or status == 429:
            return ("OpenAlex would not answer any more requests today — its free daily budget is "
                    "$0.10, about 100 searches, and it resets at midnight UTC. This search used "
                    "Europe PMC alone; running it again tomorrow, or setting an OpenAlex API key, "
                    "would widen it")
        if outcome in ("timeout", "network_error", "dns_error"):
            return ("OpenAlex could not be reached, so this search is Europe PMC's results alone "
                    "— it is narrower than it would have been, not wrong")
        return ("OpenAlex did not answer this query, so its results are missing from this search; "
                "Europe PMC's are not")

    def parse(self, row: Mapping[str, Any]) -> Candidate:
        raw_ids = row.get("ids") if isinstance(row.get("ids"), dict) else {}
        doi = normalise_doi(row.get("doi") or raw_ids.get("doi"))
        ids: dict[str, str] = {}
        for key, value in (("openalex", _bare(row.get("id") or raw_ids.get("openalex"))),
                           ("pmid", _bare(raw_ids.get("pmid"))), ("doi", doi)):
            if value:
                ids[key] = value
        # NOT pmcid: `has_pmcid:true` returns 0 works and `pmcid` appeared in 0/200 sampled
        # records. PMCIDs come from Europe PMC — which matters, because the PMC route is the OA
        # route that actually works.

        best = row.get("best_oa_location") if isinstance(row.get("best_oa_location"), dict) else {}
        open_access = row.get("open_access") if isinstance(row.get("open_access"), dict) else {}
        if _text(best.get("pdf_url"), 600):
            ids["openalex_pdf"] = _text(best.get("pdf_url"), 600)
        n_loc = 0
        for location in (row.get("locations") or []):
            if not isinstance(location, dict):
                continue
            url = _text(location.get("pdf_url"), 600)
            if url and url != ids.get("openalex_pdf") and n_loc < MAX_LOCATION_PDFS:
                n_loc += 1
                ids["openalex_location_pdf" + (f"_{n_loc}" if n_loc > 1 else "")] = url
        oa_url = _text(open_access.get("oa_url"), 600)
        if oa_url and oa_url not in ids.values():
            ids["openalex_oa_url"] = oa_url

        source = (row.get("primary_location") or {}).get("source") or {}
        return _finish(Candidate(
            key="", title=_text(row.get("display_name") or row.get("title"), 600),
            authors=_openalex_authors(row), year=_year(row.get("publication_year")),
            venue=_text(source.get("display_name"), 200), doi=doi,
            # `abstract_inverted_index` is null for ~31 % of works. Such a record is KEPT and
            # screened on its title; dropping it would silence a paper for a metadata gap.
            abstract=reconstruct_abstract(row.get("abstract_inverted_index")),
            found_by=[OpenAlex.name], ids=ids,
            license=_text(best.get("license") or best.get("license_id"), 60),
            links=links_for(doi, _text(best.get("landing_page_url")
                                       or (row.get("primary_location") or {})
                                       .get("landing_page_url"), 600))))


def _openalex_authors(row: Mapping[str, Any]) -> list[str]:
    """`"Leland E. Dibble"` → `"Dibble, Leland E."`, so every index here writes a name the same way.

    The last whitespace token is taken as the surname. That is wrong for `van der Kolk`, and the
    cost of being wrong is bounded: `dedupe`'s author guard is affirmative-only, so a surname the
    two indexes disagree about costs one duplicate PROPOSAL, never a silent merge.
    """
    authors: list[str] = []
    for entry in (row.get("authorships") or []):
        if not isinstance(entry, dict):
            continue
        name = _text((entry.get("author") or {}).get("display_name")
                     or entry.get("raw_author_name"), 160)
        if not name:
            continue
        if "," in name:
            authors.append(name)
        else:
            parts = name.split()
            authors.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else name)
    return authors[:60]


# --------------------------------------------------------------------------------------- Crossref
class Crossref:
    """DOI and metadata fill. `discovery = False`, and `run.py` honours that flag.

    Not a discovery index on purpose: its abstracts are raw JATS, present for about a quarter of
    works, and Crossref itself warns some of them may be subject to publisher copyright. It is
    excellent at one thing — turning a bibliographic string into a registered DOI — and that is the
    only thing asked of it.
    """

    name = "crossref"
    discovery = False
    URL = "https://api.crossref.org/works"
    MAX_ROWS = 100
    SELECT = "DOI,title,author,issued,container-title,type,abstract,license"

    def request(self, query: str, *, limit: int) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {"query.bibliographic": query,
                                  "rows": max(1, min(int(limit), self.MAX_ROWS)),
                                  "select": self.SELECT}
        email = contact_email()
        if email:
            # the polite pool. The contactable User-Agent alone is enough (verified:
            # `x-api-pool: polite-array`), and sending both is the documented best practice.
            params["mailto"] = email
        return self.URL, params

    def search(self, transport: SearchTransport, query: str, *, limit: int = 20,
               timeout: float = DEFAULT_TIMEOUT) -> tuple[list[Candidate], dict[str, Any]]:
        url, params = self.request(query, limit=limit)
        response = transport.get_json(url, params=params, timeout=timeout)
        if not response.ok:
            return [], source_record(self.name, query, error=response.error or response.outcome,
                                     outcome=response.outcome, status=response.status,
                                     seconds=round(response.seconds, 3))
        payload = response.json()
        message = (payload or {}).get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            return [], source_record(self.name, query,
                                     error="Crossref answered with something that was not JSON",
                                     outcome="unreadable", status=response.status)
        rows = message.get("items") or []
        candidates = [self.parse(row) for row in rows if isinstance(row, dict)]
        return candidates, source_record(self.name, query, n_returned=len(candidates),
                                         outcome=response.outcome, status=response.status,
                                         seconds=round(response.seconds, 3),
                                         total_hits=message.get("total-results"))

    def parse(self, row: Mapping[str, Any]) -> Candidate:
        doi = normalise_doi(row.get("DOI"))
        titles = row.get("title") or []
        containers = row.get("container-title") or []
        licenses = row.get("license") or []
        return _finish(Candidate(
            key="", title=_text(titles[0] if titles else "", 600),
            authors=_crossref_authors(row),
            # `{"date-parts": [[None]]}` is a real Crossref answer for an undated component record.
            year=_year((((row.get("issued") or {}).get("date-parts") or [[None]])[0] or [None])[0]),
            venue=_text(containers[0] if containers else "", 200), doi=doi,
            # deliberately NO abstract: Crossref's is raw JATS and copyright-flagged, and this
            # adapter exists for identity, not for the screener's reading material.
            abstract="", found_by=[Crossref.name],
            ids={"doi": doi} if doi else {},
            license=_text((licenses[0] or {}).get("URL") if licenses else "", 200),
            links=links_for(doi)))


def _crossref_authors(row: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for entry in (row.get("author") or []):
        if not isinstance(entry, dict):
            continue
        family = _text(entry.get("family"), 120)
        given = _text(entry.get("given"), 120)
        name = f"{family}, {given}" if family and given else family or _text(entry.get("name"), 160)
        if name:
            out.append(name)
    return out[:60]


# -------------------------------------------------------------------------------------- Unpaywall
class Unpaywall:
    """OA resolution for one DOI. Never a discovery index — it cannot search.

    `email` is MANDATORY (422 without it) and placeholder addresses are rejected, so with no
    `CANOPY_CONTACT_EMAIL` this adapter is skipped and SAYS SO. Inventing an address to satisfy a
    service that asked who we are is the kind of small dishonesty that ends in a blocked IP.

    Worth knowing when reading a record: `support.unpaywall.org` now redirects to
    `help.openalex.org` — Unpaywall is operated by OpenAlex, so it is a second index of the same
    underlying data with better repository coverage, NOT an independent check on OpenAlex's OA
    status. The record should not be read as two sources agreeing.
    """

    name = "unpaywall"
    discovery = False
    URL = "https://api.unpaywall.org/v2"

    def request(self, doi: str, *, email: str = "") -> tuple[str, dict[str, Any]]:
        return f"{self.URL}/{normalise_doi(doi)}", {"email": email or contact_email()}

    def locations(self, transport: SearchTransport, candidate: Candidate,
                  *, email: str = "", timeout: float = DEFAULT_TIMEOUT,
                  ) -> tuple[dict[str, str], dict[str, Any]]:
        """`({id_key: url}, record)` — OA URLs to merge into `candidate.ids`, best first.

        Returns the ids rather than mutating the candidate so the caller decides whether a lookup
        that found nothing is worth writing down (it is: "we asked and there is no OA copy" is the
        difference between `paywalled` and a guess).
        """
        address = (email or contact_email()).strip()
        doi = normalise_doi(candidate.doi)
        if not doi:
            return {}, source_record(self.name, candidate.key, error="",
                                     note="this candidate has no DOI, so Unpaywall could not be "
                                          "asked about it", skipped=True)
        if not address:
            # the honest skip. Recorded per call so the record shows exactly which candidates went
            # unresolved for want of one setting.
            return {}, source_record(
                self.name, doi, error="", skipped=True,
                note="Unpaywall was not asked: it requires a real contact email address and "
                     "rejects placeholders, so set CANOPY_CONTACT_EMAIL to use it")

        url, params = self.request(doi, email=address)
        response = transport.get_json(url, params=params, timeout=timeout)
        if not response.ok:
            return {}, source_record(self.name, doi, error=response.error or response.outcome,
                                     outcome=response.outcome, status=response.status,
                                     seconds=round(response.seconds, 3))
        payload = response.json()
        if not isinstance(payload, dict):
            return {}, source_record(self.name, doi, error="Unpaywall's answer was not JSON",
                                     outcome="unreadable", status=response.status)
        ids = self.parse(payload)
        return ids, source_record(self.name, doi, n_returned=len(ids), outcome=response.outcome,
                                  status=response.status, seconds=round(response.seconds, 3),
                                  license=_text((payload.get("best_oa_location") or {})
                                                .get("license"), 60),
                                  is_oa=bool(payload.get("is_oa")))

    @staticmethod
    def parse(payload: Mapping[str, Any]) -> dict[str, str]:
        """`best_oa_location.url_for_pdf`, then `oa_locations[].url_for_pdf`, then `.url`.

        `url_for_pdf` is often `null`, so `url` is a genuine fallback — and it is allowed to be a
        landing page, because the content-type gate and the `%PDF-` magic check downstream are what
        actually decide whether a fetched body is a paper.
        """
        ordered: list[str] = []
        best = payload.get("best_oa_location")
        locations = [best] + list(payload.get("oa_locations") or [])
        for location in locations:
            if isinstance(location, dict):
                for field in ("url_for_pdf", "url"):
                    url = _text(location.get(field), 600)
                    if url.startswith("http") and url not in ordered:
                        ordered.append(url)
        return {"unpaywall_pdf" + (f"_{i}" if i > 1 else ""): url
                for i, url in enumerate(ordered[:MAX_LOCATION_PDFS], start=1)}


#: every adapter, by name — `run.py` and the config read this rather than importing classes, so
#: `CANOPY_SEARCH_INDICES` can name one in a string.
INDEXES: dict[str, Any] = {index.name: index for index in
                           (EuropePmc(), OpenAlex(), Crossref(), Unpaywall())}

#: the ones that may propose a paper, in the order they are asked. Europe PMC first because it is
#: unmetered — a request economy, not a claim about which field the user is in.
DISCOVERY_INDEXES: tuple[str, ...] = tuple(
    name for name in ("europepmc", "openalex", "crossref")
    if getattr(INDEXES[name], "discovery", False))


def parse_index_names(raw: str | None = None,
                      available: Sequence[str] = DISCOVERY_INDEXES) -> list[str]:
    """`CANOPY_SEARCH_INDICES="europepmc,crossref"` → `["europepmc", "crossref"]`.

    An unknown name is dropped rather than raising: a typo in an environment variable should
    narrow a search and be visible in the record, not stop a server from starting.
    """
    text = raw if raw is not None else os.environ.get("CANOPY_SEARCH_INDICES", "")
    wanted = [part.strip().lower() for part in str(text or "").split(",") if part.strip()]
    if not wanted:
        return list(available)
    return [name for name in wanted if name in INDEXES]
