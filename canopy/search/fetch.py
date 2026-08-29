"""Turning "the screener wants this paper" into a PDF on disk — or into an honest `paywalled`.

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
`wanted` becomes `paywalled` only after a real attempt. `models.py` separates the two words for
exactly this reason: telling a user a paper is behind a paywall when nothing ever tried to fetch it
is a claim about a publisher that nobody checked. So a candidate this stage never reached — because
the fetch cap stopped first, because the job was cancelled — keeps `state == "wanted"` and gets
`fetch_outcome = "over_fetch_cap"` (or `"cancelled"`), which is a fact about *Canopy*, in Canopy's
own words. Silently leaving it `wanted` with no explanation would be the same failure one step
quieter.

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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from .indices import oa_id_urls
from .models import Candidate
from .transport import DEFAULT_FETCH_TIMEOUT, SearchTransport

__all__ = ["MAX_ATTEMPTS", "MAX_HOST_WORKERS", "RETRYABLE", "NO_OA_LOCATION", "OVER_FETCH_CAP",
           "CANCELLED", "FetchSummary", "host_of", "oa_sources", "fetch_candidate",
           "fetch_candidates", "default_probe"]

#: URLs tried per candidate. Three, because the order is quality-sorted: if the index's own copy,
#: the best OA location and the next location have all failed, the fourth is a landing page and the
#: minutes are better spent on the next paper.
MAX_ATTEMPTS = 3

#: how many DISTINCT hosts may be fetched from at once. Never two workers on one host — see the
#: module docstring. Three is the pool the design specified, applied to hosts instead of papers.
MAX_HOST_WORKERS = 3

#: outcomes worth a second pass, once the other candidates have had their turn. Only the one: a
#: 403 or a `not_a_pdf` will say the same thing next minute, and retrying it is just noise in the
#: record and pressure on a host that already answered.
RETRYABLE = frozenset({"rate_limited"})

#: `fetch_outcome` values this module writes that did not come off a wire. They are Canopy's own
#: reasons, and they are spelled out so a reader never has to guess whether a blank meant
#: "no attempt" or "attempted and nothing found".
NO_OA_LOCATION = "no_oa_location"      # tried nothing, because no index offered a URL
OVER_FETCH_CAP = "over_fetch_cap"      # the search's own cap stopped before this one
CANCELLED = "cancelled"                # the user stopped the search


@dataclass
class FetchSummary:
    """What the fetch stage did, in the vocabulary the page and the record share."""

    n_fetched: int = 0
    n_paywalled: int = 0
    n_rate_limited: int = 0          # still `wanted`: we were told to slow down, not refused
    n_over_cap: int = 0
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


def _attempt(url: str, outcome: str, status: int = 0, error: str = "") -> dict[str, Any]:
    """One row of `candidate.fetch_attempts`. Four fields the page and the record both read."""
    row: dict[str, Any] = {"url": url, "host": host_of(url), "outcome": outcome,
                           "status": int(status or 0)}
    if error:
        row["error"] = str(error)[:300]
    return row


def _slow_down_note(host: str) -> str:
    """The sentence a 429 earns. It blames us, because a 429 IS us."""
    return (f"{host or 'the index'} asked us to slow down (HTTP 429) — that is our request rate, "
            f"not a paywall; this paper is worth trying again in a minute")


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
    """
    sources = oa_sources(candidate, limit)
    if not sources:
        # nothing was tried, so nothing may be claimed about a publisher. This is still a real
        # answer to "can we read it?" — no index offered an open-access copy — and the page shows
        # it beside the links, which is what makes it actionable.
        candidate.fetch_outcome = NO_OA_LOCATION
        candidate.state = "paywalled"
        return NO_OA_LOCATION

    rate_limited_host = ""
    last_outcome = ""
    for _id_key, url in sources:
        # the probe is wrapped so its page count survives: `get_bytes` uses the probe's verdict and
        # throws the rest away, but `pdf_pages` is what the page prints under a fetched paper and
        # re-opening the file to count them again would be a second parse of the same bytes.
        seen: dict[str, Any] = {}
        download = transport.get_bytes(url, dest_dir, filename="paper.pdf", timeout=timeout,
                                       probe=_remembering(probe, seen),
                                       **({"max_bytes": max_bytes} if max_bytes else {}))
        response = download.response
        # recorded BEFORE the success test, so the winning attempt and the three that failed before
        # it are one list in the order they happened
        candidate.fetch_attempts.append(_attempt(url, response.outcome, response.status,
                                                 response.error))
        last_outcome = response.outcome
        if download.ok and download.path is not None:
            candidate.state = "fetched"
            candidate.fetch_outcome = "fetched"
            candidate.pdf_path = Path(download.path).name
            candidate.pdf_bytes = int(download.n_bytes)
            pages = seen.get("n_pages")
            candidate.pdf_pages = int(pages) if isinstance(pages, int) and pages > 0 else None
            return "fetched"
        if response.outcome == "rate_limited":
            rate_limited_host = host_of(url) or rate_limited_host

    if rate_limited_host:
        # `wanted` on purpose: the search will come back to it, and until it does, the honest word
        # is still "wanted".
        candidate.fetch_outcome = "rate_limited"
        return "rate_limited"
    candidate.state = "paywalled"
    candidate.fetch_outcome = last_outcome or NO_OA_LOCATION
    return candidate.fetch_outcome


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


def fetch_candidates(candidates: Sequence[Candidate], dest_dir: str | Path, *,
                     transport: SearchTransport,
                     probe: Callable[[Path], Mapping[str, Any]] | None = None,
                     max_fetch: int | None = None,
                     max_bytes: float | None = None,
                     timeout: float = DEFAULT_FETCH_TIMEOUT,
                     max_workers: int = MAX_HOST_WORKERS,
                     cancelled: Callable[[], bool] | None = None) -> FetchSummary:
    """Fetch every wanted candidate, serial per host, and retry the rate-limited ones at the end.

    `max_fetch` is a cap on candidates ATTEMPTED, not on candidates fetched, and the ones it cuts
    off are recorded (`over_fetch_cap`) rather than dropped: a user who sees "40 wanted, 20
    fetched" and no explanation has been told a smaller lie than the truth.
    """
    summary = FetchSummary()
    wanted = [c for c in candidates if c.state == "wanted"]
    if not wanted:
        return summary

    attempted, over_cap = wanted, []
    if max_fetch is not None and len(wanted) > max_fetch:
        attempted, over_cap = wanted[:max(0, max_fetch)], wanted[max(0, max_fetch):]
    for candidate in over_cap:
        # still `wanted`, and it must stay that way: nothing was tried, so nothing about a
        # publisher may be recorded. The outcome names OUR cap.
        candidate.fetch_outcome = OVER_FETCH_CAP
        summary.n_over_cap += 1
    if over_cap:
        summary.notes.append(
            f"the fetch cap of {max_fetch} stopped this search before {len(over_cap)} wanted "
            f"papers were tried — they are listed with their links, and raising the cap would "
            f"fetch them")

    def stopped() -> bool:
        return bool(cancelled and cancelled())

    def run_group(group: Sequence[Candidate]) -> None:
        for candidate in group:
            if stopped():
                # a cancelled search leaves the untouched candidates `wanted`, with the reason
                candidate.fetch_outcome = candidate.fetch_outcome or CANCELLED
                continue
            fetch_candidate(candidate, dest_dir, transport=transport, probe=probe,
                            max_bytes=max_bytes, timeout=timeout)

    groups = _group_by_host(attempted)
    _run_groups(groups, run_group, max_workers)

    # the second pass: only the ones a host told us to slow down for, and only after everyone else
    # has had their turn. Retrying a 429 immediately is asking the same question at the same rate.
    retry = [c for c in attempted if c.fetch_outcome in RETRYABLE and c.state == "wanted"]
    if retry and not stopped():
        summary.notes.append(
            f"{len(retry)} paper(s) were rate-limited on the first pass and retried at the end")
        _run_groups(_group_by_host(retry), run_group, max_workers)

    for candidate in attempted:
        summary.n_attempts += len(candidate.fetch_attempts)
        summary.bytes_written += int(candidate.pdf_bytes or 0)
        if candidate.state == "fetched":
            summary.n_fetched += 1
        elif candidate.state == "paywalled":
            summary.n_paywalled += 1
        if candidate.fetch_outcome == "rate_limited":
            summary.n_rate_limited += 1
            for attempt in candidate.fetch_attempts:
                if attempt.get("outcome") == "rate_limited":
                    host = str(attempt.get("host") or "")
                    if host not in summary.slowed_hosts:
                        summary.slowed_hosts.append(host)
    for host in summary.slowed_hosts:
        summary.notes.append(_slow_down_note(host))
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
