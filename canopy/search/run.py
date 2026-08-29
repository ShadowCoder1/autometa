"""The orchestrator: queries → index → dedupe → screen → fetch, and a record of all five.

`run_search` is the only function here, and everything it needs from the outside world arrives as
an argument: the transport, the model client, the model names, the caps, the directory, the
progress callback and the cancellation check. Nothing is constructed here that could open a socket
or spend money, which is why the end-to-end test in `tests/test_search_run.py` runs the REAL
orchestrator with a `RecordedTransport` and a scripted client and still touches neither.

THE PROMISE
-----------
**It does not raise for an index or a fetch failure.** Every stage is wrapped: a dead index, a
model outage, a publisher refusal, a 429 — each becomes a row in `record.sources`, a note in
`record.notes` and a `status: "error"` phase event, and the search continues with what it has. The
one thing that ends a search early is the user cancelling it or a cap firing, and both are named in
`record.stopped_because` — which the page keys its banner on, because a capped search still
finishes `done` and a user told only "done" would never learn their search was cut short.

**Every stage reports.** `on_phase(name, status, message, seconds)` fires for each of the five
`PHASES` in order, with `status` one of `running`/`ok`/`skipped`/`error`. The names are the shared
vocabulary from `models.py`; two of these rungs used to differ between the backend and the page and
neither of them ever lit.

**`unique_contributed` is measured, not assumed.** After the dedupe, each source row gets the
number of surviving candidates that ONLY that index proposed. This is the correction the design
review asked for (§C4): Europe PMC goes first because it is unmetered, not because of anything
this module believes about a research field — and a user who wants to know whether that ordering
served their question can read the answer instead of trusting it.

WHAT COUNTS AS "SCREENED" AND WHAT COUNTS AS "WANTED"
-----------------------------------------------------
The screener sets `include`/`exclude`/`unknown` and, for an included paper, `state = "wanted"`.
Only the fetch stage may turn `wanted` into `fetched` or `paywalled` — so with no model, or under a
cap, the candidates stay `not_screened` and nothing is fetched, and the record says so in the
user's own arithmetic (`counts_of`). The keyless path is a working feature, not a degraded one: a
list of candidates with their abstracts and their links is still worth a person's afternoon.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dedupe import dedupe
from .fetch import default_probe, fetch_candidates
from .indices import INDEXES, DISCOVERY_INDEXES, Unpaywall, contact_email, source_record
from .models import PHASES, Candidate, SearchRecord, counts_of, new_key
from .queries import build_queries, template_queries
from .screen import screen_candidates
from .transport import SearchTransport

__all__ = ["run_search", "DEFAULT_PER_QUERY", "DEFAULT_MAX_SCREENED", "DEFAULT_MAX_FETCH",
           "new_search_id"]

#: rows asked of each index per query. One request per query per index: the metered index charges
#: the same for a page of 200 as for a page of 1, and the unmetered one pages 1,000 at a time, so a
#: paginator would spend requests to no purpose.
DEFAULT_PER_QUERY = 100
DEFAULT_MAX_SCREENED = 200
DEFAULT_MAX_FETCH = 60


def new_search_id(now: datetime | None = None) -> str:
    """`s20260829-1a2b3c` — sortable by date, unique by hash, and safe as a directory name."""
    moment = now or datetime.now(timezone.utc)
    return "s" + moment.strftime("%Y%m%d") + "-" + new_key("c", moment.isoformat())[1:7]


def _iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


class _Phases:
    """The five-rung ladder, reported once each, in order, whatever happens inside a stage.

    A context manager rather than two calls at the ends of a stage, because the failure mode is a
    stage that raises between them: the page would then show a spinner for a rung that finished
    minutes ago, and a user watching a search would have no way to know it had moved on.
    """

    def __init__(self, record: SearchRecord,
                 on_phase: Callable[[str, str, str, float], None] | None) -> None:
        self.record = record
        self._on_phase = on_phase

    def emit(self, name: str, status: str, message: str = "", seconds: float = 0.0) -> None:
        self.record.phases.append({"name": name, "status": status, "message": message,
                                   "seconds": round(float(seconds), 3)})
        if self._on_phase is None:
            return
        try:
            self._on_phase(name, status, message, round(float(seconds), 3))
        except Exception:                       # noqa: BLE001 - a listener may not kill a search
            # the caller's progress sink is not the search's business. A broken SSE connection or a
            # full queue must not lose the work that is already done.
            pass


def run_search(*, question: str,
               protocol: Any = None,
               transport: SearchTransport,
               client: Any | None = None,
               model_roles: Mapping[str, str],
               budget_usd: float | None,
               max_screened: int = DEFAULT_MAX_SCREENED,
               max_fetch: int = DEFAULT_MAX_FETCH,
               staging_dir: str | Path,
               on_phase: Callable[[str, str, str, float], None] | None = None,
               cancelled: Callable[[], bool] | None = None,
               index_names: Sequence[str] = DISCOVERY_INDEXES,
               per_query: int = DEFAULT_PER_QUERY,
               probe: Callable[[Path], Mapping[str, Any]] | None = None,
               search_id: str = "",
               now: datetime | None = None) -> SearchRecord:
    """Run the whole search and return its record. Never raises for anything an index or a
    publisher did.

    `client is None` means no model is available: the queries come from the user's own words
    (`template_queries`) and nothing is screened. That is a supported way to run, not a failure —
    `record.query_source` says which path was taken and the page shows it, because it changes what
    a reader should expect of the recall.

    `cancelled()` is checked BETWEEN stages and inside the fetch stage's per-host loop. Between,
    rather than inside every loop, because a stage that is half-done and half-recorded is worse
    than one that finished: the screener has already paid for the batch it is in.
    """
    started = time.monotonic()
    record = SearchRecord(search_id=search_id or new_search_id(now), question=question,
                          created_at=_iso(now))
    phases = _Phases(record, on_phase)
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    def stop_requested() -> bool:
        return bool(cancelled and cancelled())

    def skip_rest(from_index: int, why: str) -> SearchRecord:
        """Every remaining rung reported `skipped`, with the same reason, once.

        Silence would leave the page's ladder half-lit forever; a different sentence per rung would
        make one cancellation look like four unrelated problems.
        """
        record.stopped_because = why
        for name in PHASES[from_index:]:
            phases.emit(name, "skipped", _stop_message(why))
        return record

    # ---------------------------------------------------------------- 1. queries
    phases.emit("queries", "running")
    stage = time.monotonic()
    built = _build_queries(client, question, protocol, model_roles, record)
    record.queries = built["queries"]
    record.criteria = built["criteria"]
    record.query_source = built["source"]
    record.cost_usd += float(built.get("cost_usd") or 0.0)
    record.notes.extend(built.get("notes") or [])
    if not record.queries:
        # nothing to ask. Reported rather than raised: the user gets a search that found nothing
        # and a sentence saying why, which they can act on by rewording the question.
        phases.emit("queries", "error", "no query could be built from this question — try naming "
                                        "the groups you are comparing and the outcome you care "
                                        "about", time.monotonic() - stage)
        return skip_rest(1, "")
    phases.emit("queries", "ok",
                f"{len(record.queries)} quer{'y' if len(record.queries) == 1 else 'ies'} "
                f"from {'a model' if record.query_source == 'model' else 'your own words'}",
                time.monotonic() - stage)
    if stop_requested():
        return skip_rest(1, "cancelled")

    # ---------------------------------------------------------------- 2. index
    phases.emit("index", "running")
    stage = time.monotonic()
    found = _run_indexes(transport, record, index_names, per_query=per_query)
    phases.emit("index", "ok" if any(s["n_returned"] for s in record.sources) else "error",
                _index_message(record.sources, len(found)), time.monotonic() - stage)
    if stop_requested():
        return skip_rest(2, "cancelled")

    # ---------------------------------------------------------------- 3. dedupe
    phases.emit("dedupe", "running")
    stage = time.monotonic()
    merged, pairs = dedupe(found)
    record.candidates = merged
    record.possible_duplicates = pairs
    _measure_unique(record)
    phases.emit("dedupe", "ok",
                f"{len(merged)} distinct paper(s) from {len(found)} record(s)"
                + (f"; {len(pairs)} possible duplicate(s) for you to judge" if pairs else ""),
                time.monotonic() - stage)
    if stop_requested():
        return skip_rest(3, "cancelled")

    # ---------------------------------------------------------------- 4. screen
    phases.emit("screen", "running")
    stage = time.monotonic()
    to_screen, over_cap = _split_at_cap(record.candidates, max_screened)
    if over_cap:
        record.notes.append(
            f"the screening cap of {max_screened} stopped this search before {len(over_cap)} "
            f"records were read — they are listed unscreened, with their abstracts")
    outcome = _screen(client, to_screen, record, model_roles, budget_usd)
    record.cost_usd = round(record.cost_usd + outcome.cost_usd, 6)
    record.notes.extend(outcome.notes)
    if outcome.stopped_because:
        record.stopped_because = outcome.stopped_because
    counts = counts_of(record.candidates, record.possible_duplicates)
    phases.emit("screen", "skipped" if client is None else "ok",
                _screen_message(client, counts), time.monotonic() - stage)
    if stop_requested():
        return skip_rest(4, "cancelled")
    if record.stopped_because == "budget":
        # the cap fired inside screening. The fetch stage is deliberately still run: it costs no
        # money, and the papers the screener DID reach deserve their PDFs.
        record.notes.append("the cost cap stopped screening; the papers that were screened were "
                            "still fetched, because fetching costs nothing")

    # ---------------------------------------------------------------- 5. fetch
    phases.emit("fetch", "running")
    stage = time.monotonic()
    summary = _fetch(record, staging, transport=transport, max_fetch=max_fetch,
                     probe=probe, cancelled=cancelled)
    record.notes.extend(summary.notes)
    if summary.stopped_because and not record.stopped_because:
        record.stopped_because = summary.stopped_because
    phases.emit("fetch", "ok", _fetch_message(summary), time.monotonic() - stage)

    record.notes.append(f"the search took {time.monotonic() - started:.0f}s")
    return record


# ------------------------------------------------------------------------------------ the stages
def _build_queries(client: Any | None, question: str, protocol: Any,
                   model_roles: Mapping[str, str], record: SearchRecord) -> dict[str, Any]:
    """One model call, or the user's own words. Never an exception either way.

    `build_queries` already falls back to the template path when the call fails; this wrapper
    exists for the case it cannot cover — a client object that raises on attribute access, a model
    name that is not in the roles map — because the query stage failing must not be the thing that
    stops a free index search.
    """
    if client is None:
        result = template_queries(question, protocol)
        result.setdefault("notes", []).append(
            "no model was available, so these queries are made of your own words and nothing was "
            "screened — every paper found is listed for you to read")
        return result
    model = model_roles.get("queries") or model_roles.get("secondary") or "claude-sonnet-5"
    try:
        return build_queries(client, question, model=model, protocol=protocol)
    except Exception as exc:                    # noqa: BLE001 - reported, never raised
        result = template_queries(question, protocol)
        result.setdefault("notes", []).append(
            f"the query call failed ({type(exc).__name__}), so the queries below were built from "
            f"your own words instead")
        record.notes.append(f"query building fell back to your own words: {exc}"[:300])
        return result


def _run_indexes(transport: SearchTransport, record: SearchRecord,
                 index_names: Sequence[str], *, per_query: int) -> list[Candidate]:
    """Ask each index each query, and write down what every one of them said.

    Sequential across indexes and queries: the transport's per-host clock would serialise same-host
    calls anyway, and the indexes here are a handful of requests, not a crawl. An index that raises
    — which it should not, the transport returns outcomes as data — is caught and recorded, because
    "the adapter had a bug" is still not a reason to lose the other index's results.
    """
    found: list[Candidate] = []
    for name in index_names:
        index = INDEXES.get(name)
        if index is None or not getattr(index, "discovery", False):
            continue
        for query in record.queries:
            text = str(query.get("text") or "").strip()
            if not text:
                continue
            try:
                candidates, row = index.search(transport, text, limit=per_query)
            except Exception as exc:            # noqa: BLE001 - a bug here is one index's problem
                candidates, row = [], source_record(
                    name, text, error=f"{type(exc).__name__}: {exc}"[:300], outcome="unreadable")
            found.extend(candidates)
            record.sources.append(row)
            if row.get("error"):
                note = row.get("note") or f"{name} did not answer: {row['error']}"
                if note not in record.notes:
                    record.notes.append(str(note))
    return found


def _measure_unique(record: SearchRecord) -> None:
    """Fill `unique_contributed` on every source row — the measured half of the index ordering.

    "Unique" means: among the papers that SURVIVED the dedupe, how many were proposed by this index
    and by no other. Counted on the merged list rather than the raw rows, because a paper both
    indexes found is one paper, and counting it twice is precisely the flattery this number exists
    to prevent. Rows for the same index across several queries share the count — an index's unique
    yield is a property of the index, not of one of its queries.
    """
    unique: dict[str, int] = {}
    for candidate in record.candidates:
        if len(candidate.found_by) == 1:
            unique[candidate.found_by[0]] = unique.get(candidate.found_by[0], 0) + 1
    for row in record.sources:
        row["unique_contributed"] = unique.get(str(row.get("name") or ""), 0)


def _split_at_cap(candidates: Sequence[Candidate],
                  cap: int | None) -> tuple[list[Candidate], list[Candidate]]:
    """The first `cap` candidates and the rest. The rest are NOT dropped — they stay in the record,
    unscreened, with the reason, so the count a user reads is a fact about the list they see."""
    rows = list(candidates)
    if cap is None or cap < 0 or len(rows) <= cap:
        return rows, []
    return rows[:cap], rows[cap:]


def _screen(client: Any | None, candidates: Sequence[Candidate], record: SearchRecord,
            model_roles: Mapping[str, str], budget_usd: float | None) -> Any:
    """The screening stage, with its own failure caught.

    `screen_candidates` already turns a budget stop and a per-batch model error into recorded
    outcomes; this catches the rest (a client that raises on construction of the call, a model name
    the provider rejects) so a model problem cannot lose an index search that already ran.
    """
    from .screen import ScreenOutcome

    model = model_roles.get("screen") or model_roles.get("secondary") or "claude-sonnet-5"
    try:
        return screen_candidates(client, candidates, question=record.question,
                                 criteria=record.criteria, model=model, budget_usd=budget_usd,
                                 cell_key_prefix=f"screen:{record.search_id}")
    except Exception as exc:                    # noqa: BLE001 - reported, never raised
        outcome = ScreenOutcome()
        outcome.notes.append(
            f"screening failed ({type(exc).__name__}: {exc}); every record is listed unscreened "
            f"for you to read"[:300])
        for candidate in candidates:
            candidate.state = "not_screened"
        return outcome


def _fetch(record: SearchRecord, staging: Path, *, transport: SearchTransport,
           max_fetch: int, probe: Callable[[Path], Mapping[str, Any]] | None,
           cancelled: Callable[[], bool] | None) -> Any:
    """Resolve the long tail with Unpaywall, then fetch. Both halves failure-tolerant.

    The Unpaywall pass runs FIRST and only for candidates that are wanted, have a DOI and have no
    OA URL from either discovery index — that is the population it exists for, and asking it about
    a paper Europe PMC already offered would spend a request to learn something we know.
    """
    from .fetch import FetchSummary, oa_sources

    wanted = [c for c in record.candidates if c.state == "wanted"]
    unpaywall = Unpaywall()
    if contact_email():
        for candidate in wanted[:max_fetch]:
            if oa_sources(candidate) or not candidate.doi:
                continue
            if cancelled and cancelled():
                break
            try:
                ids, row = unpaywall.locations(transport, candidate)
            except Exception as exc:            # noqa: BLE001 - one candidate's problem
                ids, row = {}, source_record("unpaywall", candidate.doi,
                                             error=f"{type(exc).__name__}: {exc}"[:300])
            candidate.ids.update(ids)
            # the licence is whatever Unpaywall stated, and only when nothing else stated one
            if not candidate.license and row.get("license"):
                candidate.license = str(row["license"])
            record.sources.append(row)
    elif wanted:
        record.notes.append(
            "Unpaywall was not consulted for the papers no index offered a PDF for: it needs a "
            "real contact address, so set CANOPY_CONTACT_EMAIL to widen the fetch")

    try:
        return fetch_candidates(record.candidates, staging, transport=transport,
                                probe=probe if probe is not None else default_probe(),
                                max_fetch=max_fetch, cancelled=cancelled)
    except Exception as exc:                    # noqa: BLE001 - reported, never raised
        summary = FetchSummary()
        summary.notes.append(
            f"the fetch stage failed ({type(exc).__name__}: {exc}); the papers it did not reach "
            f"are listed with their links"[:300])
        return summary


# ----------------------------------------------------------------------------------- the wording
def _stop_message(why: str) -> str:
    return {"cancelled": "you stopped this search",
            "budget": "the cost cap stopped this search",
            "deadline": "this search ran out of time"}.get(why, "this stage did not run")


def _index_message(sources: Sequence[Mapping[str, Any]], n_found: int) -> str:
    live = [s for s in sources if not s.get("error")]
    dead = sorted({str(s.get("name")) for s in sources if s.get("error")})
    text = f"{n_found} record(s) from {len({str(s.get('name')) for s in live})} index(es)"
    return text + (f"; {', '.join(dead)} did not answer" if dead else "")


def _screen_message(client: Any | None, counts: Mapping[str, int]) -> str:
    if client is None:
        return (f"no model was available, so none of the {counts['after_dedupe']} papers were "
                f"screened — they are all listed for you to read")
    return (f"{counts['screened']} read: {counts['included']} wanted, {counts['unsure']} unsure, "
            f"{counts['excluded']} ruled out")


def _fetch_message(summary: Any) -> str:
    parts = [f"{summary.n_fetched} PDF(s) fetched"]
    if summary.n_paywalled:
        parts.append(f"{summary.n_paywalled} with no open-access copy")
    if summary.n_rate_limited:
        parts.append(f"{summary.n_rate_limited} rate-limited")
    if summary.n_over_cap:
        parts.append(f"{summary.n_over_cap} over the fetch cap")
    return ", ".join(parts)
