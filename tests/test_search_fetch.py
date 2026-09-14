"""The fetch stage (`canopy.search.fetch`) — the honesty of `paywalled`, and the 429 blocker.

Two things are being defended here, and they are the two the design review called out.

**The word `paywalled` is a claim about a publisher.** `models.py` keeps `wanted` and `paywalled`
apart so that "the screener wants this and we have not answered yet" can never be shown as "the
publisher will not let you have it". So the tests below check the boring direction as hard as the
interesting one: a candidate the cap cut off, and a candidate a cancellation skipped, must still be
`wanted` when the stage ends.

**A 429 is Canopy's fault, not the publisher's** (review §C-BLK1). Europe PMC's `?pdf=render` was
measured returning `200` then `429` on two back-to-back fetches, with no `Retry-After` — and it is
the first route in the fetch order, so almost every fetch lands there. The stage therefore fetches
serially per host, records `rate_limited` as its own outcome naming the host, and retries those
candidates only after every other candidate has had its turn.

Nothing here opens a socket. Downloads replay a real PDF from `tests/fixtures/pdfs/` through the
same `stream_upload` path a live fetch uses, so the `%PDF-` magic check, the size caps and the
content-addressed filename are the real ones.
"""
from __future__ import annotations

from types import SimpleNamespace

import threading
from pathlib import Path

import pytest

from canopy.search.fetch import (CANCELLED, INTERNET_ARCHIVE, MAX_ATTEMPTS, NOT_WANTED,
                                 NO_OA_LOCATION, NO_PDF_LINK, OVER_FETCH_CAP, OVER_FETCH_DEADLINE,
                                 OVER_UNSURE_CAP, _group_by_host, fetch_candidate,
                                 fetch_candidates, fetchable, host_of, oa_sources, pdf_link_in,
                                 unsure_order)
from canopy.search.models import Candidate
from canopy.search.transport import (Download, HttpResponse, RecordedTransport, fixture_key,
                                     html_key)

PDF = Path(__file__).resolve().parent / "fixtures" / "pdfs" / "bock2005.pdf"

EPMC = "https://europepmc.org/articles/PMC13317673?pdf=render"
PUBLISHER = "https://journal.example.org/articles/1.pdf"
MIRROR = "https://mirror.example.org/1.pdf"


def candidate(key: str = "c000000000001", **ids: str) -> Candidate:
    """One wanted candidate carrying the OA routes its indexes proposed."""
    return Candidate(key=key, title=f"Paper {key}", year=2015, state="wanted",
                     doi="10.1097/npt.0000000000000086", found_by=["europepmc"], ids=dict(ids))


def serving(*urls: str) -> RecordedTransport:
    """A transport that serves the real fixture PDF for each of `urls`."""
    return RecordedTransport(payloads={fixture_key(url): PDF for url in urls})


def refusing(transport: RecordedTransport, url: str, *, status: int, outcome: str,
             error: str = "", archived: bool = False) -> RecordedTransport:
    """Register a recorded FAILURE for one URL. `RecordedTransport` replays it as itself.

    The Internet Archive's copy of the same URL is refused with it (404, never archived) unless
    `archived` says a test is about the archive: a definitive failure now sends the fetcher there
    once, and a fake that had no answer for it would be a loud miss in every old test.
    """
    transport.responses[fixture_key(url)] = HttpResponse(
        url=url, status=status, outcome=outcome, error=error or f"HTTP {status}")
    if not archived:
        transport.responses[fixture_key(INTERNET_ARCHIVE + url)] = HttpResponse(
            url=INTERNET_ARCHIVE + url, status=404, outcome="http_error",
            error="web.archive.org answered HTTP 404")
    return transport


def no_archive(transport: RecordedTransport, *urls: str) -> RecordedTransport:
    """The archive has no copy of these either."""
    for url in urls:
        transport.responses[fixture_key(INTERNET_ARCHIVE + url)] = HttpResponse(
            url=INTERNET_ARCHIVE + url, status=404, outcome="http_error",
            error="web.archive.org answered HTTP 404")
    return transport


# ------------------------------------------------------------------------------- the fetch order
def test_oa_sources_is_the_order_the_indexes_wrote_and_is_capped():
    """Every route, best first, up to `MAX_ATTEMPTS` (eight — the handedness key's four author
    manuscripts were reachable only past the third, design 03 §7b)."""
    paper = candidate(europepmc_render=EPMC, openalex_pdf=PUBLISHER,
                      openalex_location_pdf=MIRROR, openalex_oa_url="https://example.org/landing")
    assert [url for _, url in oa_sources(paper)] == [EPMC, PUBLISHER, MIRROR,
                                                     "https://example.org/landing"]
    many = candidate(**{f"openalex_location_pdf_{i}": f"{MIRROR}?n={i}" for i in range(2, 12)})
    assert len(oa_sources(many)) == MAX_ATTEMPTS == 8
    assert oa_sources(Candidate(key="c000000000002")) == []


def test_host_of_never_raises_on_something_an_index_handed_us():
    assert host_of(EPMC) == "europepmc.org"
    assert host_of("not a url") == "" and host_of("") == ""


# ---------------------------------------------------------------------------- every attempt kept
def test_a_fetch_records_the_attempt_that_worked():
    paper = candidate(europepmc_render=EPMC)
    transport = serving(EPMC)

    outcome = fetch_candidate(paper, "/tmp", transport=transport, probe=None)
    assert outcome == "fetched" and paper.state == "fetched"
    assert paper.fetch_attempts == [{"url": EPMC, "host": "europepmc.org",
                                     "outcome": "ok", "status": 200}]
    assert paper.pdf_path.endswith(".pdf") and paper.pdf_bytes > 1000


def test_every_attempt_is_recorded_even_when_a_later_one_succeeds(tmp_path):
    """"We tried Europe PMC, it 404'd, the publisher served it" is the sentence a user needs.

    Keeping only the winner would make a paper that took two tries look like one that took one,
    and would hide a broken route the tool will keep trying first on every future search.
    """
    paper = candidate(europepmc_render=EPMC, openalex_pdf=PUBLISHER)
    transport = refusing(serving(PUBLISHER), EPMC, status=404, outcome="http_error",
                         error="europepmc.org answered HTTP 404")

    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "fetched"
    assert [(a["url"], a["outcome"], a["status"]) for a in paper.fetch_attempts] == [
        (EPMC, "http_error", 404), (PUBLISHER, "ok", 200)]
    assert paper.state == "fetched"


def test_a_paper_no_route_could_reach_is_paywalled_because_we_asked(tmp_path):
    paper = candidate(openalex_pdf=PUBLISHER)
    transport = refusing(RecordedTransport(), PUBLISHER, status=403, outcome="http_error")

    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "http_error"
    # the publisher's refusal, then the archive asked once for the same URL and refusing too
    assert paper.state == "paywalled" and len(paper.fetch_attempts) == 2
    assert paper.fetch_attempts[1]["via"] == "internet_archive"
    assert paper.fetch_attempts[1]["url"] == INTERNET_ARCHIVE + PUBLISHER


def test_a_paper_with_no_oa_route_is_paywalled_and_says_which_kind(tmp_path):
    """`no_oa_location` — no index offered a copy. The resolution ran; there was nothing to try."""
    paper = candidate()
    assert fetch_candidate(paper, tmp_path, transport=RecordedTransport(),
                           probe=None) == NO_OA_LOCATION
    assert paper.state == "paywalled" and paper.fetch_attempts == []


def test_an_unreadable_pdf_is_not_a_fetched_paper(tmp_path):
    """The bytes arrived, the parser could not open them, and nothing is left on disk.

    A broken PDF in the run directory is worse than a missing one: the next stage would try to
    extract numbers from it.
    """
    paper = candidate(europepmc_render=EPMC)

    def broken(_path):
        return {"ok": False, "error": "no readable page"}

    outcome = fetch_candidate(paper, tmp_path, transport=no_archive(serving(EPMC), EPMC),
                              probe=broken)
    assert outcome == "unreadable" and paper.state == "paywalled"
    assert paper.fetch_attempts[0]["outcome"] == "unreadable"
    assert paper.fetch_attempts[1]["via"] == "internet_archive", "…and the archive was asked"
    assert paper.fetch_outcome == "unreadable", "the archive's 404 is not why there is no PDF"
    assert list(tmp_path.glob("*.pdf")) == []


def test_the_probe_s_page_count_reaches_the_record(tmp_path):
    paper = candidate(europepmc_render=EPMC)

    def probe(_path):
        return {"ok": True, "n_pages": 7}

    fetch_candidate(paper, tmp_path, transport=serving(EPMC), probe=probe)
    assert paper.pdf_pages == 7


# ----------------------------------------------------------------------------- the 429 (§C-BLK1)
class Flaky(RecordedTransport):
    """The recording, except that the first request to a named host answers 429.

    Modelled on the measured behaviour: back-to-back `?pdf=render` fetches returned 200 then 429
    with no `Retry-After`, and the same URL succeeded a minute later. `order` is what lets a test
    assert that the retry happened *after* the other candidates and not immediately.
    """

    def __init__(self, *args, rate_limit_first: tuple[str, ...] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.order: list[str] = []
        self._limited = set(rate_limit_first)

    def get_bytes(self, url, dest_dir, **kwargs):        # type: ignore[override]
        self.order.append(url)
        host = host_of(url)
        if host in self._limited:
            self._limited.discard(host)
            return Download(HttpResponse(
                url=url, status=429, outcome="rate_limited",
                error=f"{host} asked us to slow down (HTTP 429)"))
        return super().get_bytes(url, dest_dir, **kwargs)


def test_a_429_is_rate_limited_and_the_paper_stays_wanted(tmp_path):
    """Never `http_error`, never `paywalled`. We were too fast; nobody refused us."""
    paper = candidate(europepmc_render=EPMC)
    transport = Flaky(payloads={fixture_key(EPMC): PDF}, rate_limit_first=("europepmc.org",))

    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "rate_limited"
    assert paper.state == "wanted", "a rate limit is our problem, so the paper is still wanted"
    assert paper.fetch_attempts[-1]["outcome"] == "rate_limited"
    assert paper.fetch_attempts[-1]["host"] == "europepmc.org"


def test_a_rate_limited_paper_is_retried_after_the_others(tmp_path):
    """The whole point of a 429 is that the next second is the wrong time to ask again."""
    first = candidate("c000000000001", europepmc_render=EPMC)
    second = candidate("c000000000002", openalex_pdf=PUBLISHER)
    third = candidate("c000000000003", openalex_pdf=MIRROR)
    transport = Flaky(payloads={fixture_key(u): PDF for u in (EPMC, PUBLISHER, MIRROR)},
                      rate_limit_first=("europepmc.org",))

    summary = fetch_candidates([first, second, third], tmp_path, transport=transport, probe=None)

    assert transport.order.index(EPMC) == 0, "it was tried first and answered 429"
    assert transport.order[-1] == EPMC, "…and retried only after the other two had their turn"
    assert first.state == "fetched" and summary.n_fetched == 3
    assert first.fetch_attempts[0]["outcome"] == "rate_limited"
    assert "retried at the end of the pass: 1 rate-limited" in " ".join(summary.notes)


def test_a_host_that_kept_saying_429_is_named_and_is_not_called_a_paywall(tmp_path):
    paper = candidate(europepmc_render=EPMC)
    transport = refusing(RecordedTransport(), EPMC, status=429, outcome="rate_limited")

    summary = fetch_candidates([paper], tmp_path, transport=transport, probe=None)
    assert summary.n_rate_limited == 1 and summary.n_paywalled == 0
    assert summary.slowed_hosts == ["europepmc.org"]
    note = " ".join(summary.notes)
    assert "europepmc.org asked us to slow down" in note and "not a paywall" in note
    assert paper.state == "wanted"


# ----------------------------------------------------------------------------- serial per host
def test_candidates_on_one_host_share_one_worker():
    """The grouping IS the concurrency policy: one group per host, one worker per group."""
    same = [candidate("c00000000000%d" % i, europepmc_render=f"{EPMC}&n={i}") for i in (1, 2, 3)]
    assert len(_group_by_host(same)) == 1

    mixed = same + [candidate("c000000000004", openalex_pdf=PUBLISHER)]
    groups = _group_by_host(mixed)
    assert len(groups) == 2 and sorted(len(g) for g in groups) == [1, 3]


class Watcher(RecordedTransport):
    """Counts how many workers are inside one host at a time, and proves two hosts overlap.

    The barrier is the deterministic half: if the stage ran the two hosts one after the other, the
    second worker never arrives, the barrier times out and the test fails with `BrokenBarrierError`
    instead of flaking on a sleep.
    """

    def __init__(self, *args, barrier: threading.Barrier, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.barrier = barrier
        self.lock = threading.Lock()
        self.inside: dict[str, int] = {}
        self.max_inside: dict[str, int] = {}
        self.arrived: set[str] = set()

    def get_bytes(self, url, dest_dir, **kwargs):        # type: ignore[override]
        host = host_of(url)
        with self.lock:
            self.inside[host] = self.inside.get(host, 0) + 1
            self.max_inside[host] = max(self.max_inside.get(host, 0), self.inside[host])
            # only the FIRST call from each host waits: the barrier is a rendezvous between hosts,
            # and a second call from an already-arrived host would wait alone and break it
            first_from_host = host not in self.max_inside or self.max_inside[host] == 1
            first_from_host = first_from_host and host not in self.arrived
            if first_from_host:
                self.arrived.add(host)
        try:
            if first_from_host:
                self.barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        finally:
            with self.lock:
                self.inside[host] -= 1
        return super().get_bytes(url, dest_dir, **kwargs)


def test_two_hosts_run_at_once_and_one_host_never_does(tmp_path):
    """Parallel across DISTINCT hosts, serial within one. Both halves asserted.

    Without this, three workers on `europepmc.org` triple the pressure on the one host that was
    measured 429ing on the second back-to-back request.
    """
    urls = [f"{EPMC}&n=1", f"{EPMC}&n=2", PUBLISHER, MIRROR]
    papers = [candidate(f"c00000000{i:04d}", **({"europepmc_render": u} if "europepmc" in u
                                                else {"openalex_pdf": u}))
              for i, u in enumerate(urls)]
    barrier = threading.Barrier(3)          # europepmc.org + journal + mirror, all at once
    transport = Watcher(payloads={fixture_key(u): PDF for u in urls}, barrier=barrier)

    summary = fetch_candidates(papers, tmp_path, transport=transport, probe=None)

    assert summary.n_fetched == 4
    assert transport.max_inside["europepmc.org"] == 1, "never two workers on one host"
    assert not barrier.broken, "the three hosts really did run at the same time"


# --------------------------------------------------------------- the cap and the cancellation
def test_the_fetch_cap_records_itself_instead_of_dropping_papers(tmp_path):
    """Over the cap is `wanted` with `over_fetch_cap` — never `paywalled`.

    Nothing was tried, so nothing about a publisher may be recorded. The outcome names OUR cap,
    which is the number the user can change.
    """
    papers = [candidate(f"c00000000{i:04d}", europepmc_render=f"{EPMC}&n={i}") for i in range(4)]
    transport = serving(*[f"{EPMC}&n={i}" for i in range(4)])

    summary = fetch_candidates(papers, tmp_path, transport=transport, probe=None, max_fetch=2)

    assert summary.n_fetched == 2 and summary.n_over_cap == 2
    cut = papers[2:]
    assert all(p.state == "wanted" for p in cut)
    assert all(p.fetch_outcome == OVER_FETCH_CAP for p in cut)
    assert all(p.fetch_attempts == [] for p in cut), "no attempt, so no claim about a publisher"
    assert "raising the cap would fetch them" in " ".join(summary.notes)


def test_cancelling_leaves_the_untouched_papers_wanted(tmp_path):
    """Cancelled is not paywalled either — and the reason is written on the candidate."""
    papers = [candidate(f"c00000000{i:04d}", europepmc_render=f"{EPMC}&n={i}") for i in range(3)]
    transport = serving(*[f"{EPMC}&n={i}" for i in range(3)])
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 1               # the first candidate runs; the rest are stopped

    summary = fetch_candidates(papers, tmp_path, transport=transport, probe=None,
                              cancelled=cancelled)

    assert summary.stopped_because == "cancelled"
    assert papers[0].state == "fetched"
    assert [p.state for p in papers[1:]] == ["wanted", "wanted"]
    assert all(p.fetch_outcome == CANCELLED for p in papers[1:])


def test_a_paper_the_screener_read_and_did_not_want_is_not_fetched(tmp_path):
    """An excluded paper and one a human already uploaded are not fetch material.

    The screener had an opinion about the first and the file already exists for the second, so
    neither is worth a request — but the excluded one still says, in one word, why nothing was
    tried, because a blank outcome reads exactly like "we tried every route and none worked".
    """
    excluded = Candidate(key="c000000000001", state="excluded", ids={"europepmc_render": EPMC})
    uploaded = Candidate(key="u000000000001", state="uploaded", ids={"europepmc_render": EPMC},
                         pdf_path="staging/ab.pdf")
    transport = serving(EPMC)

    summary = fetch_candidates([excluded, uploaded], tmp_path, transport=transport, probe=None)
    assert summary == type(summary)(), "an empty run, and no attempt on either"
    assert excluded.fetch_attempts == [] and uploaded.fetch_attempts == []
    assert excluded.fetch_outcome == NOT_WANTED
    assert uploaded.fetch_outcome == "", "it has the PDF; there was nothing to skip"
    assert transport.calls == []


# ------------------------------------------------------- the keyless path (review §B3)
def test_a_record_nobody_screened_is_fetched_and_is_never_called_paywalled(tmp_path):
    """With no API key NOTHING is `wanted`, so a stage that fetched only `wanted` fetched nothing.

    An unscreened paper with an open-access route costs bandwidth and no money, and the PDF is the
    difference between a list of titles and a review. What it must NOT become is `paywalled`:
    that word means "the screener wanted it and no open copy exists", and nobody read this one.
    """
    unread = Candidate(key="c000000000001", state="not_screened",
                       ids={"europepmc_render": EPMC})
    blocked = Candidate(key="c000000000002", state="not_screened",
                        ids={"openalex_pdf": PUBLISHER})
    transport = refusing(serving(EPMC), PUBLISHER, status=403, outcome="http_error")

    summary = fetch_candidates([unread, blocked], tmp_path, transport=transport, probe=None)

    assert unread.state == "fetched" and unread.pdf_path.endswith(".pdf")
    assert blocked.state == "not_screened", "nobody read it, so nobody may call it paywalled"
    assert blocked.fetch_outcome == "http_error"
    assert (summary.n_fetched, summary.n_paywalled, summary.n_no_copy) == (1, 0, 1)


def test_an_unscreened_record_with_no_route_says_so_instead_of_saying_nothing(tmp_path):
    """The keyless search's own bug: every row came back blank, so the record did not even say
    that nothing had been tried."""
    unread = Candidate(key="c000000000001", state="not_screened")
    transport = serving(EPMC)

    summary = fetch_candidates([unread], tmp_path, transport=transport, probe=None)

    assert unread.fetch_outcome == NO_OA_LOCATION and unread.state == "not_screened"
    assert transport.calls == [] and summary == type(summary)()


def test_wanted_papers_are_fetched_before_the_unread_ones(tmp_path):
    """The cap has to bite the tail, not the papers the screener actually asked for."""
    unread = Candidate(key="c000000000001", state="not_screened",
                       ids={"europepmc_render": f"{EPMC}&n=1"})
    wanted = candidate("c000000000002", europepmc_render=f"{EPMC}&n=2")
    transport = serving(f"{EPMC}&n=1", f"{EPMC}&n=2")

    fetch_candidates([unread, wanted], tmp_path, transport=transport, probe=None, max_fetch=1)

    assert wanted.state == "fetched", "the screener's own choice went first"
    assert unread.fetch_outcome == OVER_FETCH_CAP and unread.fetch_attempts == []


def test_the_summary_counts_what_the_page_prints(tmp_path):
    fetched = candidate("c000000000001", europepmc_render=EPMC)
    blocked = candidate("c000000000002", openalex_pdf=PUBLISHER)
    nothing = candidate("c000000000003")
    transport = refusing(serving(EPMC), PUBLISHER, status=403, outcome="http_error")

    summary = fetch_candidates([fetched, blocked, nothing], tmp_path, transport=transport,
                               probe=None)
    assert (summary.n_fetched, summary.n_paywalled, summary.n_rate_limited) == (1, 2, 0)
    assert summary.n_attempts == 3, ("the fetched one, the blocked one and the archive's copy "
                                     "of it; the third had nothing to try, and that is not an "
                                     "attempt")
    assert summary.bytes_written == fetched.pdf_bytes


@pytest.mark.parametrize("state", ["fetched", "paywalled"])
def test_a_second_fetch_pass_does_not_re_fetch_what_it_already_answered(tmp_path, state):
    """A paper this stage has already answered about is never asked about twice."""
    paper = Candidate(key="c000000000001", state=state, ids={"europepmc_render": EPMC})
    transport = serving(EPMC)
    fetch_candidates([paper], tmp_path, transport=transport, probe=None)
    assert transport.calls == []


def test_fetchable_is_the_three_populations_and_their_order():
    """The whole policy in one function: wanted first, then the unsure papers under their cap,
    then the records nobody read."""
    wanted = candidate("c000000000001", europepmc_render=EPMC)
    unread = Candidate(key="c000000000002", state="not_screened",
                       ids={"europepmc_render": EPMC})
    routeless = Candidate(key="c000000000003", state="not_screened")
    excluded = Candidate(key="c000000000004", state="excluded",
                         ids={"europepmc_render": EPMC})
    unsure = Candidate(key="c000000000005", state="unsure", screen_decision="unknown",
                       ids={"europepmc_render": EPMC})

    assert fetchable([unread, wanted, routeless, excluded, unsure]) == [wanted, unsure, unread]
    assert fetchable([unread, wanted, routeless, excluded, unsure],
                     max_fetch_unsure=0) == [wanted, unread]


# ------------------- the seam the unit tests on either side could not see
def test_a_fetched_pdf_is_findable_by_the_code_that_builds_the_run(tmp_path):
    """`fetch` writes `pdf_path`; `server/searches.py` resolves it. They must agree.

    They did not: the fetcher recorded a bare filename and the server resolved it against the
    search directory, where the file is not — it is in `staging/`. Every unit test on both
    sides passed, because each used its own convention, and the result was that no fetched
    paper could ever become a run: `begin` refused the search for having no PDFs while the
    page showed them ticked and readable. This test owns the seam rather than either side.
    """
    from canopy.search.fetch import _record_path
    from canopy.server.searches import SearchJobs

    search_dir = tmp_path / "20260101-000000-s"
    staging = search_dir / "staging"
    staging.mkdir(parents=True)
    written = staging / "abc123.pdf"
    written.write_bytes(b"%PDF-1.4\n")

    recorded = _record_path(written, staging)
    assert recorded == "staging/abc123.pdf", "relative to the SEARCH dir, as models.py says"

    candidate = Candidate(key="c" + "0" * 12, pdf_path=recorded)
    jobs = SearchJobs(tmp_path)
    job = SimpleNamespace(run_dir=search_dir)
    assert jobs.pdf_on_disk(job, candidate) == written.resolve(), \
        "the server must find the file the fetcher wrote"

    # …and the traversal guard still holds on a hand-edited record
    evil = Candidate(key="c" + "1" * 12, pdf_path="../../etc/passwd")
    assert jobs.pdf_on_disk(job, evil) is None


def test_a_recorded_path_never_escapes_the_search_directory(tmp_path):
    from canopy.search.fetch import _record_path

    staging = tmp_path / "s" / "staging"
    staging.mkdir(parents=True)
    inside = staging / "x.pdf"
    inside.write_bytes(b"%PDF-1.4\n")
    assert not _record_path(inside, staging).startswith("/")
    assert ".." not in _record_path(inside, staging)


# ------------------------------------------------------- how hard one paper is tried (design 03 §7)
def test_every_route_is_tried_before_giving_up(tmp_path):
    """Up to eight, in the indexes' order. The old cap of three never reached Unpaywall's route
    for a paper Europe PMC and OpenAlex both had a dead link for."""
    routes = {f"openalex_location_pdf_{i}": f"{MIRROR}?n={i}" for i in range(2, 6)}
    paper = candidate(europepmc_render=EPMC, openalex_pdf=PUBLISHER, openalex_location_pdf=MIRROR,
                      unpaywall_pdf="https://repo.example.org/late.pdf", **routes)
    transport = serving("https://repo.example.org/late.pdf")
    for url in [EPMC, PUBLISHER, MIRROR] + list(routes.values()):
        refusing(transport, url, status=403, outcome="http_error")

    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "fetched"
    assert len(paper.fetch_attempts) == 8 and paper.fetch_attempts[-1]["outcome"] == "ok"
    assert paper.fetch_attempts[-1]["url"] == "https://repo.example.org/late.pdf"


def test_a_landing_page_is_read_once_for_its_pdf_link_and_the_link_is_relative(tmp_path):
    """`<link rel="alternate" type="application/pdf" href="…">` is usually relative; it is joined
    against the page's FINAL url (after redirects) and then fetched through the same gate."""
    landing = "https://journal.example.org/article/42"
    paper = candidate(openalex_oa_url=landing)
    transport = serving("https://journal.example.org/article/42/download.pdf")
    refusing(transport, landing, status=200, outcome="not_a_pdf",
             error="journal.example.org served text/html, not a PDF")
    transport.responses[html_key(landing)] = HttpResponse(
        url="https://journal.example.org/article/42/", status=200, outcome="ok",
        body=(b'<html><head><title>x</title>'
              b'<link rel="alternate" type="application/pdf" href="download.pdf">'
              b'</head><body></body></html>'))

    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "fetched"
    assert [(a["url"], a["outcome"], a.get("via", "")) for a in paper.fetch_attempts] == [
        (landing, "not_a_pdf", ""),
        ("https://journal.example.org/article/42/download.pdf", "ok", "landing_page")]
    assert [c["method"] for c in transport.calls] == ["get_bytes", "get_html", "get_bytes"]


def test_a_landing_page_that_names_no_pdf_is_on_the_record(tmp_path):
    landing = "https://journal.example.org/article/43"
    paper = candidate(openalex_oa_url=landing)
    transport = refusing(RecordedTransport(), landing, status=200, outcome="not_a_pdf")
    transport.responses[html_key(landing)] = HttpResponse(
        url=landing, status=200, outcome="ok", body=b"<html><body>Log in</body></html>")

    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "not_a_pdf"
    assert paper.state == "paywalled"
    assert [(a["outcome"], a.get("via", "")) for a in paper.fetch_attempts] == [
        ("not_a_pdf", ""), (NO_PDF_LINK, "landing_page"), ("http_error", "internet_archive")]
    # a second pass reads nothing again: the page and the archive were each asked once
    fetch_candidate(paper, tmp_path, transport=transport, probe=None)
    assert len(paper.fetch_attempts) == 3


def test_pdf_link_in_reads_both_conventions_and_nothing_else():
    page = ('<html><head><meta name="citation_title" content="x">'
            '<meta content="https://cdn.example.org/p.pdf" name="citation_pdf_url">'
            '<a href="/evil.pdf">not this</a></head></html>')
    assert pdf_link_in(page, "https://journal.example.org/a/1") == "https://cdn.example.org/p.pdf"
    assert pdf_link_in('<link rel="alternate" type="application/pdf" href="../p.pdf">',
                       "https://j.example.org/a/b/1") == "https://j.example.org/a/p.pdf"
    assert pdf_link_in('<link rel="alternate" type="text/html" href="/p.pdf">', "https://x/") == ""
    assert pdf_link_in('<meta name="citation_pdf_url" content="javascript:alert(1)">',
                       "https://x/") == ""
    assert pdf_link_in("", "https://x/") == ""


def test_the_archive_is_asked_last_and_never_after_a_rate_limit(tmp_path):
    paper = candidate(openalex_pdf=PUBLISHER)
    transport = refusing(serving(INTERNET_ARCHIVE + PUBLISHER), PUBLISHER, status=403,
                         outcome="http_error", archived=True)
    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "fetched"
    assert paper.fetch_attempts[-1]["via"] == "internet_archive"
    assert paper.fetch_attempts[-1]["host"] == "web.archive.org"

    slowed = candidate("c000000000002", openalex_pdf=f"{PUBLISHER}?x=2")
    transport = refusing(RecordedTransport(), f"{PUBLISHER}?x=2", status=429,
                         outcome="rate_limited", archived=True)
    assert fetch_candidate(slowed, tmp_path, transport=transport, probe=None) == "rate_limited"
    assert len(slowed.fetch_attempts) == 1, "a 429 earns a retry later, not an archive lookup"
    assert slowed.state == "wanted"


def test_a_url_that_answered_definitively_is_never_asked_again(tmp_path):
    """Second passes — after Unpaywall found a route, after a 429 elsewhere — ask only what is
    still open. A 403 will say 403 again."""
    paper = candidate(europepmc_render=EPMC, openalex_pdf=PUBLISHER)
    transport = refusing(refusing(RecordedTransport(), EPMC, status=403, outcome="http_error"),
                         PUBLISHER, status=429, outcome="rate_limited")
    assert fetch_candidate(paper, tmp_path, transport=transport, probe=None) == "rate_limited"
    n = len(transport.calls)
    serve = serving(PUBLISHER)
    serve.responses.update({k: v for k, v in transport.responses.items()
                            if k != fixture_key(PUBLISHER)})
    assert fetch_candidate(paper, tmp_path, transport=serve, probe=None) == "fetched"
    assert [c["url"] for c in serve.calls] == [PUBLISHER], "only the rate-limited one again"
    assert n == 2


def test_unpaywall_after_every_route_failed_gets_one_more_pass(tmp_path):
    """The `resolve_more` seam: a candidate whose routes all said no is offered to Unpaywall,
    and only when that found something new is it tried again."""
    paper = candidate(openalex_pdf=PUBLISHER)
    late = "https://repo.example.org/found-later.pdf"
    transport = refusing(serving(late), PUBLISHER, status=403, outcome="http_error")
    asked = []

    def resolve_more(c):
        asked.append(c.key)
        c.ids["unpaywall_pdf"] = late
        return True

    summary = fetch_candidates([paper], tmp_path, transport=transport, probe=None,
                               resolve_more=resolve_more)
    assert asked == [paper.key] and paper.state == "fetched"
    assert [a["url"] for a in paper.fetch_attempts] == [PUBLISHER, INTERNET_ARCHIVE + PUBLISHER,
                                                        late]
    assert summary.n_fetched == 1


# ----------------------------------------------------------- the unsure papers, under their cap
def test_unsure_papers_are_fetched_after_wanted_in_relevance_order_and_capped(tmp_path):
    """Half of what a v2 screener reads is `unsure`, and every one fetched is a paper the review
    will read at dollars a paper: relevance decides which, and `max_fetch_unsure` how many."""
    wanted = candidate("c000000000001", europepmc_render=f"{EPMC}&w=1")
    low = Candidate(key="c000000000002", state="unsure", screen_decision="unknown", keep=True,
                    ids={"europepmc_render": f"{EPMC}&u=low"})
    high = Candidate(key="c000000000003", state="unsure", screen_decision="unknown", keep=True,
                     ids={"europepmc_render": f"{EPMC}&u=high"})
    routeless = Candidate(key="c000000000004", state="unsure", screen_decision="unknown",
                          keep=True)
    setattr(low, "relevance", 1.0)
    setattr(high, "relevance", 5.0)
    assert unsure_order([low, high, routeless]) == [high, low, routeless]
    transport = serving(f"{EPMC}&w=1", f"{EPMC}&u=low", f"{EPMC}&u=high")

    summary = fetch_candidates([low, high, routeless, wanted], tmp_path, transport=transport,
                               probe=None, max_fetch_unsure=1)

    assert wanted.state == "fetched" and high.state == "fetched"
    assert high.screen_decision == "unknown" and high.keep is True, "still the screener's unsure"
    assert low.state == "unsure" and low.fetch_outcome == OVER_UNSURE_CAP and low.keep is True
    assert routeless.fetch_outcome == NO_OA_LOCATION
    assert [c["url"] for c in transport.calls] == [f"{EPMC}&w=1", f"{EPMC}&u=high"]
    assert (summary.n_fetched, summary.n_unsure_fetched, summary.n_unsure_over_cap) == (2, 1, 1)
    assert any("most relevant unsure" in note for note in summary.notes)


def test_an_unsure_paper_no_route_could_reach_is_never_called_paywalled(tmp_path):
    unsure = Candidate(key="c000000000001", state="unsure", screen_decision="unknown", keep=True,
                       ids={"openalex_pdf": PUBLISHER})
    transport = refusing(RecordedTransport(), PUBLISHER, status=403, outcome="http_error")
    fetch_candidates([unsure], tmp_path, transport=transport, probe=None)
    assert unsure.state == "unsure" and unsure.fetch_outcome == "http_error"


def test_the_deadline_cuts_the_tail_and_says_so(tmp_path):
    """Wanted → unsure → unread is the workload order, so a wall clock that runs out lands on
    the papers nobody asked for. The rows it never reached carry the reason."""
    first = candidate("c000000000001", europepmc_render=f"{EPMC}&n=1")
    second = candidate("c000000000002", europepmc_render=f"{EPMC}&n=2")
    unread = Candidate(key="c000000000003", state="not_screened",
                       ids={"europepmc_render": f"{EPMC}&n=3"})
    transport = serving(f"{EPMC}&n=1", f"{EPMC}&n=2", f"{EPMC}&n=3")
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])

    summary = fetch_candidates([first, second, unread], tmp_path, transport=transport,
                               probe=None, deadline_s=50.0, now=lambda: next(ticks, 100.0))

    assert first.state == "fetched"
    assert second.state == "wanted" and second.fetch_outcome == OVER_FETCH_DEADLINE
    assert unread.fetch_outcome == OVER_FETCH_DEADLINE and unread.fetch_attempts == []
    assert summary.n_over_deadline == 2
    assert any("deadline" in note for note in summary.notes)
