"""Turning "the screener wants this paper" into a PDF on disk — or into an honest `paywalled`.

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
`wanted` becomes `paywalled` only after a real attempt. `models.py` separates the two words for
exactly this reason: telling a user a paper is behind a paywall when nothing ever tried to fetch it
is a claim about a publisher that nobody checked. So a candidate this stage never reached — because
the fetch cap stopped first, because the job was cancelled — keeps `state == "wanted"` and gets
`fetch_outcome = "over_fetch_cap"` (or `"cancelled"`), which is a fact about *Canopy*, in Canopy's
own words. Silently leaving it `wanted` with no explanation would be the same failure one step
quieter. The corollary holds too: a candidate NOBODY SCREENED never becomes `paywalled`, however
the fetch went, because "the screener wanted this and the publisher would not give it" is two
claims and only the second one was tested.

WHAT THIS STAGE FETCHES, AND WHY IT IS NOT JUST `wanted` (review §B3)
--------------------------------------------------------------------
Two populations, in this order: everything the screener **wanted**, then everything **nobody
read** that carries an open-access route. The second half is what makes a search without an API
key a feature rather than a dead end — with no model nothing is screened, so nothing is `wanted`,
so the old rule fetched *nothing at all* and left the user a list of titles they could not open,
could not judge and could not begin a run from. An unscreened paper with an OA route costs
bandwidth and no money, and a PDF on disk is the difference between a list and a review.

Between those two, the papers the screener could NOT decide about (`unsure`) — fetched in
relevance order and only the first `max_fetch_unsure` of them (design 03 §5). Screening v2 is
built to answer `unknown` whenever the abstract does not settle the question, so roughly half of
what it reads lands here; every one of those PDFs is a paper the review will READ, at dollars per
paper, and a cap on them is the difference between a $5 search and a $700 run. The ones past the
cap stay `unsure`, stay ticked, and say `over_unsure_cap`. `excluded` is never fetched: the
screener read it and ruled it out, and a human who disagrees follows the link or uploads the PDF.
Every row this stage does not attempt records WHY it did not (`NOT_WANTED`, `NO_OA_LOCATION`,
`OVER_FETCH_CAP`, `OVER_UNSURE_CAP`, `OVER_FETCH_DEADLINE`, `CANCELLED`) — a blank `fetch_outcome`
reads the same as "we tried and found nothing", and that is the one thing it must never be
mistaken for.

HOW HARD ONE PAPER IS TRIED (design 03 §7)
------------------------------------------
Every route an index offered, up to `MAX_ATTEMPTS`, in `OA_ID_PREFIXES` order. Then, when a route
served an HTML page instead of a PDF, that page is read ONCE for a `citation_pdf_url` meta tag or
an `application/pdf` alternate link and the link is fetched (`via: "landing_page"`) — re-vetted by
the transport like any other URL. Then, when nothing worked and nothing was rate-limited, the
Internet Archive's copy of the best failed URL is asked for once (`via: "internet_archive"`). A URL
that already gave a definitive answer is never asked again, on any pass; only a 429 earns a retry.

WHAT A RATE LIMIT IS, AND WHAT IT IS NOT (review §C-BLK1, the blocker this module answers)
------------------------------------------------------------------------------------------
Two back-to-back fetches of `https://europepmc.org/articles/<PMCID>?pdf=render` were measured
returning `200` then `429` — no `Retry-After`, no rate-limit headers of any kind — and the same URL
succeeded a minute later. That host is sources #1 and #2 of the fetch order, so a naive
`ThreadPoolExecutor` over candidates lands almost every fetch on the one host that 429s, and the
headline number the whole feature is judged on ("found 40, fetched 31") collapses toward "fetched
9" while the record blames the publisher. Four things here prevent that:

* **`transport.HOST_INTERVALS["europepmc.org"] = 2.0`** — verified present, and `HostClock`'s
  parent-domain match makes the redirect target `europepmc.org/api/getPdf` the same host.
* **Fetching is SERIAL PER HOST.** Candidates are grouped by the host of their best OA route and
  one worker owns each group, so parallelism only ever happens *across distinct hosts*. That also
  removes the pressure the byte-budget race (review §S1) was about.
* **429 is `rate_limited`, its own outcome, and it NAMES THE HOST.** It is never folded into
  `http_error`, because "the publisher refused" and "we asked too fast" are different sentences and
  only one of them is true.
* **A rate-limited candidate is retried AFTER every other candidate**, not immediately — the whole
  point of a 429 is that the next second is the wrong time to ask again.

EVERY ATTEMPT IS RECORDED, INCLUDING THE ONES THAT WORKED OUT
-------------------------------------------------------------
`candidate.fetch_attempts` gets `{url, host, outcome, status}` for every URL tried, even when a
later one succeeded. "We tried the publisher, it refused, Europe PMC served it" is the sentence a
user needs when they wonder why a fetch took three tries — and when a paper is missing, the list of
what was tried is the only thing that lets them fix it by hand.

WHAT THIS MODULE DOES NOT DECIDE
--------------------------------
Which URLs exist: `indices.py` wrote them onto `candidate.ids` in the order §3.5 tries them, so the
routes a fetch will take are visible in `search.json` before any of them is taken. Whether a body
is a PDF: `transport.get_bytes` → `server.uploads.stream_upload` owns the `%PDF-` magic, both size
caps and the content-addressed name, and a fetched paper therefore lands under exactly the same
rules as one a human uploaded. And the licence: it is whatever the index said, copied verbatim,
never inferred from the fact that a download worked.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urljoin

import httpx

from .indices import oa_id_urls
from .models import Candidate
from .transport import DEFAULT_FETCH_TIMEOUT, SearchTransport

__all__ = ["MAX_ATTEMPTS", "MAX_HOST_WORKERS", "RETRYABLE", "NO_OA_LOCATION", "OVER_FETCH_CAP",
           "OVER_UNSURE_CAP", "OVER_FETCH_DEADLINE", "CANCELLED", "NOT_WANTED", "NO_PDF_LINK",
           "DEFAULT_MAX_FETCH_UNSURE", "FETCH_DEADLINE_S", "INTERNET_ARCHIVE", "FetchSummary",
           "host_of", "oa_sources", "pdf_link_in", "fetch_candidate", "fetch_candidates",
           "fetchable", "unsure_order", "default_probe"]

#: URLs tried per candidate. Eight, because the routes are now the index's copy, the best OA
#: location, up to five more locations and Unpaywall's, and the four author manuscripts the
#: first answer key needed were listed only past the third (design 03 §7b).
MAX_ATTEMPTS = 8

#: how many `unsure` papers may be fetched, best claim first — see the module docstring
DEFAULT_MAX_FETCH_UNSURE = 100

#: wall-clock seconds the whole fetch stage may take. 500 candidates × up to 8 routes with Europe
#: PMC at 2 s each is 20–40 minutes; the rows the deadline never reached say so and the cut lands
#: on the unread tail, because the workload is ordered wanted → unsure → unread.
FETCH_DEADLINE_S = 1200.0

#: the Wayback Machine's "latest capture" prefix. It answers with a redirect to the timestamped
#: copy on its own host, which the transport's hop vetting allows, and the `%PDF-` gate and the
#: probe decide whether what came back is a paper. The licence is never inferred from it.
INTERNET_ARCHIVE = "https://web.archive.org/web/2/"

#: how many DISTINCT hosts may be fetched from at once. Never two workers on one host — see the
#: module docstring. Three is the pool the design specified, applied to hosts instead of papers.
MAX_HOST_WORKERS = 3

#: outcomes worth a second pass, once the other candidates have had their turn: a 429, and the
#: three that say nothing about the paper at all — a name that did not resolve, a stalled
#: download, a dropped connection. A 403 or a `not_a_pdf` will say the same thing next minute,
#: and retrying it is just noise in the record and pressure on a host that already answered.
#: None of these four may ever turn a `wanted` paper into `paywalled`: nobody at the publisher
#: said no.
RETRYABLE = frozenset({"rate_limited", "dns_error", "timeout", "network_error"})

#: `fetch_outcome` values this module writes that did not come off a wire. They are Canopy's own
#: reasons, and they are spelled out so a reader never has to guess whether a blank meant
#: "no attempt" or "attempted and nothing found".
NO_OA_LOCATION = "no_oa_location"      # tried nothing, because no index offered a URL
OVER_FETCH_CAP = "over_fetch_cap"      # the search's own cap stopped before this one
OVER_UNSURE_CAP = "over_unsure_cap"    # an unsure paper past `max_fetch_unsure`; still ticked
OVER_FETCH_DEADLINE = "over_fetch_deadline"   # the stage's wall clock ran out first
CANCELLED = "cancelled"                # the user stopped the search
NOT_WANTED = "not_wanted"              # the screener read it and ruled it out
NO_PDF_LINK = "no_pdf_link"            # a landing page was read and named no PDF


@dataclass
class FetchSummary:
    """What the fetch stage did, in the vocabulary the page and the record share."""

    n_fetched: int = 0
    n_paywalled: int = 0
    #: tried, no PDF, and nobody had screened it. Kept apart from `n_paywalled` for the reason
    #: `models.py` keeps `wanted` and `paywalled` apart: "the screener wanted it and the publisher
    #: refused" is two claims, and only the second one was tested here.
    n_no_copy: int = 0
    n_rate_limited: int = 0          # still `wanted`: we were told to slow down, not refused
    n_unreachable: int = 0           # still `wanted`: a name, a stall, a dropped connection
    n_over_cap: int = 0
    #: `unsure` papers fetched, and `unsure` papers the cap on them left unfetched — two numbers
    #: the run-cost line at `begin` is built from
    n_unsure_fetched: int = 0
    n_unsure_over_cap: int = 0
    n_over_deadline: int = 0
    n_attempts: int = 0
    bytes_written: int = 0
    stopped_because: str = ""        # "" | "cancelled"
    #: hosts that answered 429, so the page can say "Europe PMC asked us to slow down" rather than
    #: implying a paywall
    slowed_hosts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def host_of(url: str) -> str:
    """The host of a URL, lowercased — `""` when it has none we can read.

    Used to group work, so it must never raise on a malformed URL an index handed us: a candidate
    with a nonsense URL belongs in its own group of one, not in a traceback.
    """
    try:
        return (httpx.URL(str(url)).host or "").lower()
    except (httpx.InvalidURL, ValueError, TypeError):
        return ""


def oa_sources(candidate: Candidate, limit: int = MAX_ATTEMPTS) -> list[tuple[str, str]]:
    """`[(id_key, url), …]` — the URLs this candidate will be fetched from, best route first.

    The order is `indices.OA_ID_PREFIXES` and it survives a dedupe merge, so a paper both indexes
    proposed is tried on Europe PMC's copy first and the publisher's second, whichever index
    happened to answer first on the day.
    """
    return oa_id_urls(candidate)[:max(0, int(limit))]


def default_probe(timeout: float = 20.0) -> Callable[[Path], Mapping[str, Any]]:
    """The real "can this actually be read?" check, as a callable `get_bytes` can hold.

    Imported lazily because `canopy.server.uploads` drags in the pipeline, and both this module and
    the transport must stay importable without it. A fetched file that the ingester cannot open is
    `unreadable` and is deleted before it ever gets its final name — a broken PDF in the run
    directory is worse than a missing one, because the next stage would try to extract from it.
    """
    from ..server.uploads import probe_pdf

    def probe(path: Path) -> Mapping[str, Any]:
        return probe_pdf(path, timeout=timeout)

    return probe


def _remembering(probe: Callable[[Path], Mapping[str, Any]] | None,
                 into: dict[str, Any]) -> Callable[[Path], Mapping[str, Any]] | None:
    """The same probe, with its answer copied into `into`. `None` in, `None` out.

    `None` must stay `None` rather than becoming a no-op callable: `get_bytes` treats "no probe" as
    "do not run one", and handing it a callable that always says `ok` would turn "we did not check"
    into "we checked and it was fine".
    """
    if probe is None:
        return None

    def remembering(path: Path) -> Mapping[str, Any]:
        result = probe(path)
        into.update(dict(result or {}))
        return result

    return remembering


def _attempt(url: str, outcome: str, status: int = 0, error: str = "",
             via: str = "") -> dict[str, Any]:
    """One row of `candidate.fetch_attempts`. Four fields the page and the record both read, and
    `via` when the URL came from somewhere other than an index's own list."""
    row: dict[str, Any] = {"url": url, "host": host_of(url), "outcome": outcome,
                           "status": int(status or 0)}
    if error:
        row["error"] = str(error)[:300]
    if via:
        row["via"] = via
    return row


def _slow_down_note(host: str) -> str:
    """The sentence a 429 earns. It blames us, because a 429 IS us."""
    return (f"{host or 'the index'} asked us to slow down (HTTP 429) — that is our request rate, "
            f"not a paywall; this paper is worth trying again in a minute")


def _record_path(written: Path, dest_dir: Path) -> str:
    """Where a fetched PDF lives, said the way the rest of the system says it.

    `dest_dir` is the search's `staging/`, so the search directory is its parent and the
    recorded string is `staging/<sha>.pdf`. Falling back to the bare name would re-create the
    mismatch this function exists to remove, so an unexpected layout says so loudly instead.
    """
    root = dest_dir.resolve().parent
    try:
        return str(written.resolve().relative_to(root))
    except ValueError:                                     # pragma: no cover - defensive
        return str(written.resolve())


#: `<meta …>` / `<link …>` tags and their attributes, read leniently: publisher HTML is not ours
#: and half of it would not validate. Only two tags are looked at and only two attributes read.
_TAG_RE = re.compile(r"<(meta|link)\b([^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")


def pdf_link_in(html: str, base: str) -> str:
    """The PDF a landing page names, absolute, or `""`.

    Two conventions cover nearly every publisher and repository: Google Scholar's
    `<meta name="citation_pdf_url" content="…">` (usually absolute) and
    `<link rel="alternate" type="application/pdf" href="…">` (often relative — hence `urljoin`
    against the page's FINAL URL, after its redirects). Read from at most `MAX_HTML_BYTES` of
    the page. Nothing else is parsed: a page's `<a>` links are a different, much larger question.
    """
    text = str(html or "")
    for kind, raw in _TAG_RE.findall(text):
        attrs: dict[str, str] = {}
        for name, a, b, c in _ATTR_RE.findall(raw):
            attrs[name.lower()] = unescape(a or b or c or "").strip()
        href = ""
        if kind.lower() == "meta":
            if (attrs.get("name") or attrs.get("property") or "").lower() == "citation_pdf_url":
                href = attrs.get("content", "")
        elif ("alternate" in (attrs.get("rel") or "").lower()
              and (attrs.get("type") or "").lower() == "application/pdf"):
            href = attrs.get("href", "")
        if href:
            joined = urljoin(str(base or ""), href)
            if joined.lower().startswith(("https://", "http://")):
                return joined
    return ""


def _already(candidate: Candidate, via: str) -> bool:
    """Has this candidate already had its one landing-page follow / archive lookup?"""
    return any(attempt.get("via") == via for attempt in candidate.fetch_attempts)


def fetch_candidate(candidate: Candidate, dest_dir: str | Path, *,
                    transport: SearchTransport,
                    probe: Callable[[Path], Mapping[str, Any]] | None = None,
                    max_bytes: float | None = None,
                    timeout: float = DEFAULT_FETCH_TIMEOUT,
                    limit: int = MAX_ATTEMPTS) -> str:
    """Try this candidate's OA URLs in order; return the `fetch_outcome` written on it.

    Mutates the candidate: `fetch_attempts` (every URL, always), `fetch_outcome`, and — only when
    something actually happened — `state`, `pdf_path`, `pdf_pages`, `pdf_bytes`.

    The three endings, and the difference between them is the whole honesty of this stage:

    * a PDF landed and passed the probe → `state = "fetched"`;
    * every URL was tried and none produced a paper → `state = "paywalled"`, because we asked;
    * a host told us to slow down and nothing else worked → the state STAYS `wanted`, the outcome
      is `rate_limited`, and the caller retries it later. Calling that a paywall would be blaming a
      publisher for our own request rate.

    Only a candidate that arrived `wanted` can be left `paywalled`. A `not_screened` or `unsure`
    one keeps its state whatever the fetch did, because `paywalled` in this tool means "the
    screener wanted it and no open copy exists", and half of that sentence was never established
    for a paper nobody read or nobody could decide about. Its `fetch_outcome` still says exactly
    what happened.

    Safe to call again on the same candidate: a URL that already answered definitively is
    skipped, so a second pass (after Unpaywall found a new route, or after a 429) asks only what
    is still open.
    """
    was_wanted = candidate.state == "wanted"
    sources = oa_sources(candidate, limit)
    if not sources:
        # nothing was tried, so nothing may be claimed about a publisher. This is still a real
        # answer to "can we read it?" — no index offered an open-access copy — and the page shows
        # it beside the links, which is what makes it actionable.
        candidate.fetch_outcome = NO_OA_LOCATION
        if was_wanted:
            candidate.state = "paywalled"
        return NO_OA_LOCATION

    answered = {a["url"]: a["outcome"] for a in candidate.fetch_attempts
                if a.get("outcome") not in RETRYABLE}
    rate_limited_host = ""
    transient_outcome = ""           # a lookup, a stall, a dropped connection: try again later
    last_outcome = ""

    def attempt(url: str, via: str = "") -> bool:
        """One download, recorded whatever happened. True when a paper landed."""
        nonlocal rate_limited_host, last_outcome
        # the probe is wrapped so its page count survives: `get_bytes` uses the probe's verdict
        # and throws the rest away, but `pdf_pages` is what the page prints under a fetched paper
        # and re-opening the file to count them again would be a second parse of the same bytes.
        seen: dict[str, Any] = {}
        download = transport.get_bytes(url, dest_dir, filename="paper.pdf", timeout=timeout,
                                       probe=_remembering(probe, seen),
                                       **({"max_bytes": max_bytes} if max_bytes else {}))
        response = download.response
        # recorded BEFORE the success test, so the winning attempt and the ones that failed before
        # it are one list in the order they happened
        candidate.fetch_attempts.append(_attempt(url, response.outcome, response.status,
                                                 response.error, via=via))
        if via != "internet_archive":
            # the archive's 404 is not the reason there is no PDF; the publisher's answer is.
            # The archive attempt is on the record, and `fetch_outcome` keeps the real one.
            last_outcome = response.outcome
        answered[url] = response.outcome
        if download.ok and download.path is not None:
            candidate.state = "fetched"
            candidate.fetch_outcome = "fetched"
            # RELATIVE TO THE SEARCH DIRECTORY, which is what `models.Candidate.pdf_path`
            # documents and what `server/searches.py` resolves against. Recording the bare
            # filename here made every fetched paper unresolvable at begin time — the file
            # sits in `staging/`, the reader looked beside it, found nothing, and the run was
            # refused for having no PDFs while the page showed them ticked and readable. The
            # two sides now agree, and `test_search_fetch.py` asserts they still do.
            candidate.pdf_path = _record_path(Path(download.path), Path(dest_dir))
            candidate.pdf_bytes = int(download.n_bytes)
            pages = seen.get("n_pages")
            candidate.pdf_pages = int(pages) if isinstance(pages, int) and pages > 0 else None
            return True
        if response.outcome == "rate_limited":
            rate_limited_host = host_of(url) or rate_limited_host
        elif response.outcome in RETRYABLE:
            transient_outcome = response.outcome
        return False

    for _id_key, url in sources:
        if url in answered:
            continue                     # it already said no, definitively; asking again is noise
        if attempt(url):
            return "fetched"

    # one landing-page follow per candidate: the route that served HTML instead of a PDF is read
    # for the PDF it names, and that link is fetched through the same gate as everything else
    if not rate_limited_host and not transient_outcome and not _already(candidate,
                                                                          "landing_page"):
        page = next((a["url"] for a in candidate.fetch_attempts
                     if a.get("outcome") == "not_a_pdf" and not a.get("via")), "")
        if page:
            link = _follow_landing_page(candidate, page, transport, timeout)
            if link and link not in answered and attempt(link, via="landing_page"):
                return "fetched"

    # the last resort, once, and never on top of a 429 or a hiccup: the archive's copy of the
    # best failed URL
    if (not rate_limited_host and not transient_outcome
            and not _already(candidate, "internet_archive")):
        failed = next((a["url"] for a in candidate.fetch_attempts
                       if a.get("outcome") not in ("ok", "rate_limited") and not a.get("via")), "")
        if failed and attempt(INTERNET_ARCHIVE + failed, via="internet_archive"):
            return "fetched"

    if rate_limited_host:
        # `wanted` on purpose: the search will come back to it, and until it does, the honest word
        # is still "wanted".
        candidate.fetch_outcome = "rate_limited"
        return "rate_limited"
    if transient_outcome:
        # the same: nothing about the paper was learned, so nothing about a paywall is said
        candidate.fetch_outcome = transient_outcome
        return transient_outcome
    if was_wanted:
        candidate.state = "paywalled"
    candidate.fetch_outcome = last_outcome or candidate.fetch_outcome or NO_OA_LOCATION
    return candidate.fetch_outcome


def _follow_landing_page(candidate: Candidate, page: str, transport: SearchTransport,
                         timeout: float) -> str:
    """Read one landing page for its PDF link. The read itself is recorded as an attempt row
    (`via: "landing_page"`), so "we looked and the page named no PDF" is on the record too."""
    response = transport.get_html(page, timeout=timeout)
    if not response.ok:
        candidate.fetch_attempts.append(_attempt(page, response.outcome, response.status,
                                                 response.error, via="landing_page"))
        return ""
    link = pdf_link_in(response.body.decode("utf-8", "replace"), response.url or page)
    if not link:
        candidate.fetch_attempts.append(_attempt(page, NO_PDF_LINK, response.status,
                                                 "the page names no PDF", via="landing_page"))
    return link


def _group_by_host(candidates: Sequence[Candidate]) -> list[list[Candidate]]:
    """Candidates grouped by the host of their best OA route, order preserved within a group.

    This grouping IS the concurrency policy: one worker per group means one worker per host, so
    two threads never ask Europe PMC for a PDF at the same time while two different hosts still
    run in parallel. A candidate whose later attempts fall on another host is still serialised
    there by the transport's own per-host clock — the grouping makes the common case right and the
    clock makes the uncommon case safe.
    """
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        sources = oa_sources(candidate)
        groups.setdefault(host_of(sources[0][1]) if sources else "", []).append(candidate)
    return list(groups.values())


def unsure_order(candidates: Sequence[Candidate]) -> list[Candidate]:
    """The `unsure` papers, best claim first: by `relevance` when the search ranked them, else in
    arrival order — and the cap on them cuts THIS order, so an unsure paper that reached the run
    is the one the ranking put highest, not the one an index happened to answer with first."""
    rows = [c for c in candidates if c.state == "unsure"]
    if any(getattr(c, "relevance", None) is not None for c in rows):
        rows.sort(key=lambda c: (-float(getattr(c, "relevance", None) or 0.0), c.key))
    return rows


def fetchable(candidates: Sequence[Candidate],
              max_fetch_unsure: int | None = DEFAULT_MAX_FETCH_UNSURE) -> list[Candidate]:
    """The candidates this stage may fetch, best claim first — see the module docstring.

    Wanted papers first, so a cap that bites cuts the tail and never a paper the screener asked
    for. Then the first `max_fetch_unsure` of the `unsure` papers by relevance, with a route.
    Then the records nobody read that carry an open-access route: with no model NOTHING is
    wanted, and a stage that fetched only `wanted` fetched nothing at all.
    """
    wanted = [c for c in candidates if c.state == "wanted"]
    unsure = [c for c in unsure_order(candidates) if oa_sources(c)]
    if max_fetch_unsure is not None:
        unsure = unsure[:max(0, int(max_fetch_unsure))]
    unread = [c for c in candidates if c.state == "not_screened" and oa_sources(c)]
    return wanted + unsure + unread


def _record_skips(candidates: Sequence[Candidate], attempting: set[str]) -> None:
    """Write WHY on every row this stage will not try. Nothing leaves here with a blank reason.

    A blank `fetch_outcome` is indistinguishable from "we tried every route and none worked", and
    the keyless search left every single row blank: nothing was screened, so nothing was wanted,
    so nothing was fetched and the record did not even say so (review §B3). A row that already
    carries an outcome — a previous pass, a cap, a cancellation — keeps the one it earned.
    """
    for candidate in candidates:
        if candidate.key in attempting or candidate.fetch_outcome or candidate.pdf_path:
            continue                     # attempted below, already answered, or already readable
        if candidate.state in ("not_screened", "unsure") and not oa_sources(candidate):
            # it survived `fetchable`'s filter only by having no route at all
            candidate.fetch_outcome = NO_OA_LOCATION
        elif candidate.state == "unsure":
            # a route exists and the cap on unsure papers stopped before it. Still ticked: the
            # screener could not rule it out, and the user may raise the cap or upload the PDF
            candidate.fetch_outcome = OVER_UNSURE_CAP
        elif candidate.state == "excluded":
            candidate.fetch_outcome = NOT_WANTED


def fetch_candidates(candidates: Sequence[Candidate], dest_dir: str | Path, *,
                     transport: SearchTransport,
                     probe: Callable[[Path], Mapping[str, Any]] | None = None,
                     max_fetch: int | None = None,
                     max_fetch_unsure: int | None = DEFAULT_MAX_FETCH_UNSURE,
                     max_bytes: float | None = None,
                     timeout: float = DEFAULT_FETCH_TIMEOUT,
                     max_workers: int = MAX_HOST_WORKERS,
                     deadline_s: float | None = FETCH_DEADLINE_S,
                     cancelled: Callable[[], bool] | None = None,
                     resolve_more: Callable[[Candidate], bool] | None = None,
                     now: Callable[[], float] = time.monotonic) -> FetchSummary:
    """Fetch every `fetchable` candidate, serial per host, and retry the rate-limited ones at the
    end.

    `max_fetch` is a cap on candidates ATTEMPTED, not on candidates fetched, and the ones it cuts
    off are recorded (`over_fetch_cap`) rather than dropped: a user who sees "40 wanted, 20
    fetched" and no explanation has been told a smaller lie than the truth. `max_fetch_unsure`
    bounds the unsure papers inside that workload (`over_unsure_cap` past it). `deadline_s` is
    the stage's wall clock, checked between candidates; the rows it never reaches say
    `over_fetch_deadline`.

    `resolve_more(candidate)` is the seam for the second Unpaywall pass: called once for a
    candidate whose every route failed definitively, it may add routes to `candidate.ids` and
    returns True when it did, in which case the candidate is tried once more (only the new
    routes — a URL that already answered is never asked again).
    """
    summary = FetchSummary()
    workload = fetchable(candidates, max_fetch_unsure=max_fetch_unsure)
    _record_skips(candidates, {c.key for c in workload})
    summary.n_unsure_over_cap = sum(1 for c in candidates if c.fetch_outcome == OVER_UNSURE_CAP)
    if summary.n_unsure_over_cap:
        summary.notes.append(
            f"{summary.n_unsure_over_cap} paper(s) the screener could not decide about were not "
            f"fetched: only the {max_fetch_unsure} most relevant unsure papers are, because each "
            f"one fetched is a paper the review will read in full — they stay ticked and listed "
            f"with their links, and raising the unsure cap would fetch them")
    if not workload:
        return summary

    attempted, over_cap = workload, []
    if max_fetch is not None and len(workload) > max_fetch:
        attempted, over_cap = workload[:max(0, max_fetch)], workload[max(0, max_fetch):]
    for candidate in over_cap:
        # the state is left exactly as it was, and it must be: nothing was tried, so nothing
        # about a publisher may be recorded. The outcome names OUR cap.
        candidate.fetch_outcome = OVER_FETCH_CAP
        summary.n_over_cap += 1
    if over_cap:
        summary.notes.append(
            f"the fetch cap of {max_fetch} stopped this search before {len(over_cap)} more "
            f"papers were tried — they are listed with their links, and raising the cap would "
            f"fetch them")

    started = now()

    def stopped() -> bool:
        return bool(cancelled and cancelled())

    def out_of_time() -> bool:
        return deadline_s is not None and (now() - started) > float(deadline_s)

    def run_group(group: Sequence[Candidate]) -> None:
        for candidate in group:
            if stopped():
                # a cancelled search leaves the untouched candidates `wanted`, with the reason
                candidate.fetch_outcome = candidate.fetch_outcome or CANCELLED
                continue
            if out_of_time():
                candidate.fetch_outcome = candidate.fetch_outcome or OVER_FETCH_DEADLINE
                continue
            outcome = fetch_candidate(candidate, dest_dir, transport=transport, probe=probe,
                                      max_bytes=max_bytes, timeout=timeout)
            if (resolve_more is not None and candidate.doi and not candidate.pdf_path
                    and outcome not in ("fetched", "rate_limited", NO_OA_LOCATION)):
                # every route it had said no. One more source of routes, and one more pass over
                # whatever is new — the old URLs are remembered and not asked twice.
                try:
                    found_more = bool(resolve_more(candidate))
                except Exception:               # noqa: BLE001 - one candidate's problem
                    found_more = False
                if found_more and not stopped() and not out_of_time():
                    fetch_candidate(candidate, dest_dir, transport=transport, probe=probe,
                                    max_bytes=max_bytes, timeout=timeout)

    groups = _group_by_host(attempted)
    _run_groups(groups, run_group, max_workers)

    # the second pass: only the ones a host told us to slow down for, and only after everyone else
    # has had their turn. Retrying a 429 immediately is asking the same question at the same rate.
    # `not c.pdf_path` rather than `state == "wanted"`: an unscreened candidate that was told to
    # slow down keeps `not_screened`, and keying the retry on the state would have quietly
    # dropped exactly the rows the keyless path exists to fetch.
    retry = [c for c in attempted if c.fetch_outcome in RETRYABLE and not c.pdf_path]
    if retry and not stopped() and not out_of_time():
        n_slowed = sum(1 for c in retry if c.fetch_outcome == "rate_limited")
        summary.notes.append(
            f"{len(retry)} paper(s) were retried at the end of the pass: {n_slowed} rate-limited, "
            f"{len(retry) - n_slowed} unreachable (a name that did not resolve, a stalled or "
            f"dropped connection)")
        _run_groups(_group_by_host(retry), run_group, max_workers)

    for candidate in attempted:
        summary.n_attempts += len(candidate.fetch_attempts)
        summary.bytes_written += int(candidate.pdf_bytes or 0)
        if candidate.state == "fetched":
            summary.n_fetched += 1
            if candidate.screen_decision == "unknown":
                summary.n_unsure_fetched += 1
        elif candidate.state == "paywalled":
            summary.n_paywalled += 1
        elif candidate.fetch_attempts and candidate.fetch_outcome not in RETRYABLE:
            # tried, nothing came back, and nobody had screened it (or nobody could decide) — so
            # it is counted here and not as a paywall, which would be a claim about a publisher
            # on a paper the screener never asked for
            summary.n_no_copy += 1
        if candidate.fetch_outcome == OVER_FETCH_DEADLINE:
            summary.n_over_deadline += 1
        if candidate.fetch_outcome in RETRYABLE and candidate.fetch_outcome != "rate_limited":
            summary.n_unreachable += 1
        if candidate.fetch_outcome == "rate_limited":
            summary.n_rate_limited += 1
            for attempt in candidate.fetch_attempts:
                if attempt.get("outcome") == "rate_limited":
                    host = str(attempt.get("host") or "")
                    if host not in summary.slowed_hosts:
                        summary.slowed_hosts.append(host)
    for host in summary.slowed_hosts:
        summary.notes.append(_slow_down_note(host))
    if summary.n_unreachable:
        summary.notes.append(
            f"{summary.n_unreachable} paper(s) could not be reached — the host did not resolve, "
            f"stalled or dropped the connection, twice — so nothing is said about a paywall; "
            f"they keep their links and are worth a search again later")
    if summary.n_over_deadline:
        summary.notes.append(
            f"the fetch stage's {int(deadline_s or 0) // 60}-minute deadline ran out before "
            f"{summary.n_over_deadline} paper(s) were tried — they are listed with their links, "
            f"and searching again with fewer papers, or uploading them, would get them")
    if stopped():
        summary.stopped_because = "cancelled"
    return summary


def _run_groups(groups: Sequence[Sequence[Candidate]],
                work: Callable[[Sequence[Candidate]], None], max_workers: int) -> None:
    """One worker per host group. Sequential when there is only one group — which is the common
    case, and a thread pool for one serial list is a thread pool that exists to be misread."""
    live = [group for group in groups if group]
    if len(live) <= 1:
        for group in live:
            work(group)
        return
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(live)))) as pool:
        for future in [pool.submit(work, group) for group in live]:
            # `.result()` rather than letting the pool swallow it: a bug in a worker must surface
            # here, where `run.py` records it as a stage failure, and not vanish into a future
            # nobody read.
            future.result()
