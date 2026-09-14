"""The orchestrator: queries → index → dedupe → screen → fetch, and a record of all five.

The first stage now builds concept BLOCKS (`blocks.py`) and two Boolean strings from them, has
width control measure and prune them (`width.py`), and sends each string to every index form in
the order the indexes were measured to rank (`indices.py`: PubMed, OpenAlex title/abstract,
Europe PMC three pages deep, OpenAlex full text when the string fits). The screening cap then
cuts a RANKED list (`rank.py`), not the arrival order, and is derived from the budget unless the
user set a number. The reasons are in design 03 §1–3 and are measured in
`validation/search_bench`.

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

**A user's exclusions are obeyed, never judged.** `exclude` is a list of DOIs and title fragments
a person has forbidden. It is applied between the dedupe and the screener — so a forbidden paper is
never sent to a model, never fetched and never billed — and every entry gets a row in
`record.exclusions` saying how it was read and how many papers it caught, including the ones that
caught none. `Candidate.excluded_by_user` carries the entry that did it, so a reader of
`search.json` can tell the tool's own verdict from the person's instruction without parsing prose.

WHAT COUNTS AS "SCREENED" AND WHAT COUNTS AS "WANTED"
-----------------------------------------------------
The screener sets `include`/`exclude`/`unknown` and, for an included paper, `state = "wanted"`.
Only the fetch stage may turn `wanted` into `fetched` or `paywalled` — so with no model, or under a
cap, the candidates stay `not_screened`, and the record says so in the user's own arithmetic
(`counts_of`).

**The keyless path is a working feature, and it is fetched.** With no model nothing is screened, so
nothing is `wanted` — and a fetch stage that fetched only `wanted` therefore fetched nothing at
all, leaving the user a list of bare titles with no PDF, which is not worth anyone's afternoon.
`fetch.fetchable` takes the wanted papers first and then every record nobody read that carries an
open-access route, so a search run without an API key ends with PDFs on disk, an abstract excerpt
and links on every row, and a Begin button that works once the reviewer ticks what they want. What
it does NOT get is a screener's verdict, and `counts_of` says `screened: 0` in so many words.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import threading

from .blocks import INDEX_FORMS, build_plan, query_rows, template_plan
from .cost import (COST_PER_RECORD, SHARE_SNOWBALL, predict, run_commit, run_cost_per_paper,
                   screening_cap)
from .dedupe import dedupe, normalise_doi, normalise_title
from .fetch import DEFAULT_MAX_FETCH_UNSURE, FETCH_DEADLINE_S, default_probe, fetch_candidates
from .indices import (DEFAULT_DEPTH, INDEXES, DISCOVERY_INDEXES, Unpaywall, contact_email,
                      depth_for, search_pages, source_record)
from .models import PHASES, Candidate, SearchRecord, counts_of, new_key
from .rank import rank
from .screen import screen_candidates
from .snowball import round_cap, snowball
from .transport import SearchTransport

__all__ = ["run_search", "DEFAULT_DEPTH", "DEFAULT_MAX_SCREENED", "DEFAULT_MAX_FETCH",
           "DEFAULT_MAX_FETCH_UNSURE", "MIN_TITLE_FRAGMENT", "EXCLUDED_BY_USER",
           "exclusion_rules", "new_search_id"]

#: the screening cap when neither a budget nor a number bounds it (`cost.screening_cap`). Kept
#: because the server's settings pin against it; a search with a budget derives its own.
DEFAULT_MAX_SCREENED = 200
#: candidates the fetch stage may ATTEMPT. Five hundred, up from sixty: fetching costs bandwidth
#: and no money, the stage has a wall clock of its own (`FETCH_DEADLINE_S`), and the sixty-cap
#: cut 201 papers of the first real search without the page being able to say so.
DEFAULT_MAX_FETCH = 500

#: the shortest NORMALISED title fragment that may be used as a substring rule.
#:
#: A title fragment matches by containment, which is the only way "the Cisneros review" can be
#: named without retyping its full title — and containment is also how one careless word deletes
#: half a search. `tremor` normalises to six characters and would silently forbid every paper with
#: `tremor` anywhere in its title, in a tool whose whole claim is that nothing disappears without
#: a reason. Twelve characters is about two ordinary words: long enough that a user has said
#: something specific, short enough that a distinctive phrase still works. A shorter entry that is
#: not a DOI is REFUSED and reported, never quietly widened and never quietly dropped.
MIN_TITLE_FRAGMENT = 12

#: `Candidate.fetch_outcome` on a paper the user forbade. It is set BEFORE the fetch stage runs,
#: because `fetch._record_skips` writes `not_wanted` ("the screener read it and did not want it")
#: over every blank outcome in the `excluded` state — a true sentence about a screener's verdict
#: and a false one about a person's, on the one row where the difference is the whole point.
EXCLUDED_BY_USER = "excluded_by_you"


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
               max_screened: int | None = None,
               max_fetch: int = DEFAULT_MAX_FETCH,
               max_fetch_unsure: int | None = DEFAULT_MAX_FETCH_UNSURE,
               staging_dir: str | Path,
               on_phase: Callable[[str, str, str, float], None] | None = None,
               cancelled: Callable[[], bool] | None = None,
               index_names: Sequence[str] = DISCOVERY_INDEXES,
               exclude: Sequence[str] = (),
               depth: int = DEFAULT_DEPTH,
               seed_dois: Sequence[str] = (),
               chase_citations: bool = True,
               probe: Callable[[Path], Mapping[str, Any]] | None = None,
               search_id: str = "",
               now: datetime | None = None) -> SearchRecord:
    """Run the whole search and return its record. Never raises for anything an index or a
    publisher did.

    `client is None` means no model is available: the queries come from the user's own words
    (`blocks.template_plan`) and nothing is screened. That is a supported way to run, not a failure —
    `record.query_source` says which path was taken and the page shows it, because it changes what
    a reader should expect of the recall.

    `cancelled()` is checked BETWEEN stages and inside the fetch stage's per-host loop. Between,
    rather than inside every loop, because a stage that is half-done and half-recorded is worse
    than one that finished: the screener has already paid for the batch it is in.

    `exclude` is a list of DOIs and title fragments the USER has forbidden. It is applied after the
    dedupe and before the screener, so a forbidden paper is never sent to a model, never fetched
    and never billed — and it is recorded as the person's decision, in `record.exclusions` and in
    `Candidate.excluded_by_user`, never as a verdict this tool reached.

    `max_fetch_unsure` bounds how many papers the screener could not decide about are fetched
    (best claim first). Each one fetched is a paper the review will read in full, at dollars a
    paper, and the record's last note prices that commitment out loud (`fetch.py`, design 03 §5).

    `max_screened` is an OVERRIDE: `None` means the cap is what `SHARE_SCREEN` of `budget_usd`
    buys at `COST_PER_RECORD` (`cost.screening_cap`), and the list it cuts is ranked. `depth` is
    the rows asked of each index form per string (Europe PMC three times that). `seed_dois` are
    recorded on the plan for the seed check (design 03 §1, step 7) and not yet acted on.
    """
    started = time.monotonic()
    record = SearchRecord(search_id=search_id or new_search_id(now), question=question,
                          created_at=_iso(now), depth=int(depth))
    # what this search is expected to spend, written before a cent is: the bench compares it
    # with what it did spend, and the page shows it beside the cap
    record.predicted = predict(budget_usd, max_fetch_unsure=max_fetch_unsure,
                               snowball=chase_citations)
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
    plan = _build_plan(client, question, protocol, model_roles, record, transport, seed_dois)
    record.plan = plan
    record.queries = query_rows(plan)
    record.criteria = [str(r.get("rule") or "") for r in plan.get("rubric") or []
                       if str(r.get("rule") or "").strip()]
    record.query_source = str(plan.get("source") or "template")
    record.cost_usd += float(plan.get("cost_usd") or 0.0)
    record.notes.extend(str(n) for n in (plan.get("notes") or []))
    if not any(row.get("text") for row in record.queries):
        # nothing to ask. Reported rather than raised: the user gets a search that found nothing
        # and a sentence saying why, which they can act on by rewording the question.
        phases.emit("queries", "error", "no query could be built from this question — try naming "
                                        "the groups you are comparing and the outcome you care "
                                        "about", time.monotonic() - stage)
        return skip_rest(1, "")
    phases.emit("queries", "ok", _plan_message(plan, record.queries), time.monotonic() - stage)
    if stop_requested():
        return skip_rest(1, "cancelled")

    # ---------------------------------------------------------------- 2. index
    phases.emit("index", "running")
    stage = time.monotonic()
    found = _run_indexes(transport, record, index_names, depth=depth)
    found.extend(_injected_seeds(record))
    phases.emit("index", "ok" if any(s.get("n_returned") for s in record.sources) else "error",
                _index_message(record.sources, len(found)), time.monotonic() - stage)
    if stop_requested():
        return skip_rest(2, "cancelled")

    # ---------------------------------------------------------------- 3. dedupe
    phases.emit("dedupe", "running")
    stage = time.monotonic()
    merged, pairs = dedupe(found)
    record.candidates = merged
    record.possible_duplicates = pairs
    _count_rows(merged, found)
    _measure_unique(record)
    # the user's own exclusions, here and nowhere else: after the dedupe, so a paper found twice is
    # forbidden once, and before the screener, so a forbidden paper never reaches a model. Reported
    # on the dedupe rung rather than on a sixth one — `PHASES` is the vocabulary the page draws and
    # this is not a stage the search performs, it is a person's instruction being obeyed.
    n_forbidden = _apply_exclusions(record, exclude)
    phases.emit("dedupe", "ok",
                f"{len(merged)} distinct paper(s) from {len(found)} record(s)"
                + (f"; {len(pairs)} possible duplicate(s) for you to judge" if pairs else "")
                + (f"; {n_forbidden} you excluded" if n_forbidden else ""),
                time.monotonic() - stage)
    if stop_requested():
        return skip_rest(3, "cancelled")

    # ---------------------------------------------------------------- 4. screen
    phases.emit("screen", "running")
    stage = time.monotonic()
    # a paper the user forbade is not offered to the screener AND does not spend the cap on its
    # way past: it was never going to be read, so counting it against the abstracts that could be
    # would cost the user a record they had not excluded.
    screenable = [c for c in record.candidates if not c.excluded_by_user]
    cap = screening_cap(budget_usd, max_screened)
    if cap is None and client is not None:
        cap = DEFAULT_MAX_SCREENED
    to_screen, over_cap = _rank_and_split(screenable, plan, cap)
    if over_cap:
        record.notes.append(
            f"the screening cap of {cap} "
            + ("(what 70 % of the budget buys at $0.003 a record) " if max_screened is None
               else "")
            + f"stopped this search before {len(over_cap)} records were read — they are the "
              f"{len(over_cap)} ranked least relevant, listed unscreened with their abstracts "
              f"and their rank")
    outcome = _screen(client, to_screen, record, model_roles, budget_usd, protocol)
    record.cost_usd = round(record.cost_usd + outcome.cost_usd, 6)
    # the batch map goes to disk. `screen.py` claimed for a while that it was already there and
    # re-read on resume; it was neither, and a false line in an audit trail is worse than a
    # missing one (review §M10). What it buys is real but smaller: a reader can price screening
    # per batch and see which twenty records one failed call cost.
    record.batches = [asdict(batch) for batch in outcome.batches]
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

    # ---------------------------------------------------------------- 5. snowball
    phases.emit("snowball", "running")
    stage = time.monotonic()
    if client is None:
        phases.emit("snowball", "skipped", "no model was available to screen what the citations "
                                           "would find", time.monotonic() - stage)
    elif not chase_citations:
        phases.emit("snowball", "skipped", "citation chasing was switched off",
                    time.monotonic() - stage)
    elif record.stopped_because == "budget":
        phases.emit("snowball", "skipped", "the cost cap had already stopped screening",
                    time.monotonic() - stage)
    else:
        _chase(client, record, transport, model_roles, budget_usd, protocol,
               cancelled=cancelled)
        phases.emit("snowball", "ok", _snowball_message(record.rounds), time.monotonic() - stage)
    if stop_requested():
        return skip_rest(5, "cancelled")

    # ---------------------------------------------------------------- 6. fetch
    phases.emit("fetch", "running")
    stage = time.monotonic()
    summary = _fetch(record, staging, transport=transport, max_fetch=max_fetch,
                     max_fetch_unsure=max_fetch_unsure, probe=probe, cancelled=cancelled)
    record.notes.extend(summary.notes)
    if summary.stopped_because and not record.stopped_because:
        record.stopped_because = summary.stopped_because
    phases.emit("fetch", "ok", _fetch_message(summary), time.monotonic() - stage)
    _say_the_run_cost(record)

    _say_what_it_cost(record)
    record.notes.append(f"the search took {time.monotonic() - started:.0f}s")
    return record


# ------------------------------------------------------------------------------------ the stages
def _build_plan(client: Any | None, question: str, protocol: Any,
                model_roles: Mapping[str, str], record: SearchRecord, transport: SearchTransport,
                seeds: Sequence[str]) -> dict[str, Any]:
    """One model call, expansion, width control, rendering — or the user's own words. Never an
    exception either way.

    `build_plan` already falls back to the template path when the call fails; this wrapper
    exists for the case it cannot cover — a client object that raises on attribute access, a
    model name that is not in the roles map, a width control that dies — because the query stage
    failing must not be the thing that stops a free index search.
    """
    if client is None:
        try:
            plan = build_plan(None, question, model="", protocol=protocol, transport=transport,
                              seeds=seeds)
        except Exception as exc:                # noqa: BLE001 - reported, never raised
            plan = template_plan(question, protocol)
            plan["notes"].append(f"width control failed ({type(exc).__name__}: {exc}"[:200]
                                 + "); the strings are sent unpruned")
            from .blocks import expand_plan, render_strings

            expand_plan(plan, protocol, question)
            render_strings(plan)
        plan["notes"].append(
            "no model was available, so these blocks are made of your own words and nothing was "
            "screened — every paper found is listed for you to read, and the open-access copies "
            "were still fetched")
        return plan
    model = model_roles.get("queries") or model_roles.get("secondary") or "claude-sonnet-5"
    billed_before = _client_cost(client)
    try:
        plan = build_plan(client, question, model=model, protocol=protocol, transport=transport,
                          seeds=seeds)
    except Exception as exc:                    # noqa: BLE001 - reported, never raised
        plan = template_plan(question, protocol)
        from .blocks import expand_plan, render_strings

        expand_plan(plan, protocol, question)
        render_strings(plan)
        plan["notes"].append(
            f"the query call failed ({type(exc).__name__}), so the blocks below were built from "
            f"your own words instead")
        record.notes.append(f"query building fell back to your own words: {exc}"[:300])
    # A call that failed after the provider answered was still BILLED, and the template path
    # reports `cost_usd: 0.0` because it never called anything — so a failed query call used to
    # be charged to nobody and shown to the user as $0.00 (review §M9). The client's own ledger
    # is the one that knows, and the difference across the call is the truth whichever path ran.
    spent = max(0.0, _client_cost(client) - billed_before)
    if spent > float(plan.get("cost_usd") or 0.0):
        plan["cost_usd"] = round(spent, 6)
    return plan


def _client_cost(client: Any) -> float:
    """What the client has billed so far, or 0.0 for a stand-in that keeps no ledger."""
    total = getattr(client, "total_cost", None)
    try:
        return float(total()) if callable(total) else 0.0
    except Exception:                           # pragma: no cover - a stand-in without a ledger
        return 0.0


def _run_indexes(transport: SearchTransport, record: SearchRecord,
                 index_names: Sequence[str], *, depth: int) -> list[Candidate]:
    """Ask each index form its own rows of `record.queries`, in the measured order, and write
    down what every one of them said.

    The order is `blocks.INDEX_FORMS` — PubMed, OpenAlex title/abstract, Europe PMC, OpenAlex
    full text — filtered by `index_names`; a row an index was never sent (the full-text form of
    a string too long for it) becomes a `sources` row that says `skipped` and why, so the page
    never mistakes "not asked" for "found nothing". Sequential across indexes and queries: the
    transport's per-host clock would serialise same-host calls anyway. An index that raises —
    which it should not, the transport returns outcomes as data — is caught and recorded,
    because "the adapter had a bug" is still not a reason to lose the other index's results.
    """
    found: list[Candidate] = []
    for name, form in INDEX_FORMS:
        if name not in index_names:
            continue
        index = INDEXES.get(name)
        if index is None or not getattr(index, "discovery", False):
            continue
        for row in record.queries:
            if str(row.get("index") or "") != name or str(row.get("form") or "") != form:
                continue
            if row.get("skipped") or not str(row.get("text") or "").strip():
                record.sources.append(source_record(
                    name, "", query_id=str(row.get("query_id") or ""), form=form, pages=0,
                    skipped=True, chars=int(row.get("chars") or 0),
                    note=str(row.get("why") or "not sent")))
                continue
            try:
                candidates, source = search_pages(index, transport, row,
                                                  depth=depth_for(name, depth))
            except Exception as exc:            # noqa: BLE001 - a bug here is one index's problem
                candidates, source = [], source_record(
                    name, str(row.get("text") or ""), query_id=str(row.get("query_id") or ""),
                    form=form, pages=0, error=f"{type(exc).__name__}: {exc}"[:300],
                    outcome="unreadable")
            found.extend(candidates)
            record.sources.append(source)
            if source.get("error"):
                note = source.get("note") or f"{name} did not answer: {source['error']}"
                if note not in record.notes:
                    record.notes.append(str(note))
    return found


def _injected_seeds(record: SearchRecord) -> list[Candidate]:
    """The seed papers no string reached even after a restore and a rewrite (design 03 §1):
    their Europe PMC records become candidates found by "seed", ticked for screening ahead of
    the cap, and NEVER counted as recall — the user handed them in."""
    from .indices import EuropePmc

    out: list[Candidate] = []
    for row in (record.plan or {}).get("seed_inject") or []:
        if not isinstance(row, Mapping):
            continue
        candidate = EuropePmc().parse(row)
        candidate.found_by = ["seed"]
        candidate.seed = True
        out.append(candidate)
    if out:
        record.sources.append(source_record("seed", "the seed DOIs no string reached",
                                            n_returned=len(out), query_id="", form="", pages=0,
                                            note="injected from the seed list, not found by a "
                                                 "search string"))
        record.notes.append(f"{len(out)} seed paper(s) no string reached were added as "
                            f"candidates found by \"seed\"")
    return out


def _count_rows(merged: Sequence[Candidate], found: Sequence[Candidate]) -> None:
    """Attribute every raw index row to the candidate that survived it (`Candidate.n_rows`).

    This is the only place the raw list and the merged list are both in scope, and it is the
    number `counts_of` publishes as `records`. Counting `len(found_by)` instead — which is what
    the page used to show — under-counts by exactly the rows one index returned for several
    queries: twelve rows were reported as four, four lines below a phase message that said
    "2 distinct paper(s) from 12 record(s)". The two now agree because they are one measurement.
    """
    owner: dict[str, Candidate] = {}
    for candidate in merged:
        candidate.n_rows = 0
        owner[candidate.key] = candidate
        for absorbed in candidate.merged_from:
            owner.setdefault(absorbed, candidate)
    for raw in found:
        target = owner.get(raw.key)
        if target is not None:
            target.n_rows += 1
    for candidate in merged:
        # a candidate the raw list cannot account for is one row, not zero: the honest floor is
        # "it exists", and a zero here would silently shrink the top of the funnel
        candidate.n_rows = max(1, candidate.n_rows)


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


def exclusion_rules(entries: Sequence[str]) -> list[dict[str, Any]]:
    """Each entry a person typed, read as a DOI or as a title fragment, with its own audit row.

    `normalise_doi` and `normalise_title` are `dedupe`'s — the same two functions that decide
    whether two index rows are the same paper. Reusing them is not tidiness: an exclusion written
    with a different notion of sameness would forbid a paper under one spelling and let the same
    paper back in under another, which is the exact failure the user asked to be protected from.

    Nothing is guessed. An entry that is a DOI is matched as a DOI; anything else is a title
    fragment; and a fragment too short to be safe (`MIN_TITLE_FRAGMENT`) is `refused` here rather
    than run — with the reason on the row, because a rule that silently did not apply is worse
    than one that never existed.
    """
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in entries or ():
        entry = str(raw or "").strip()
        if not entry or entry in seen:
            continue                    # the same line typed twice is one instruction, not two
        seen.add(entry)
        doi = normalise_doi(entry)
        if doi:
            rules.append({"entry": entry, "kind": "doi", "read_as": doi, "matched": 0, "note": ""})
            continue
        fragment = normalise_title(entry)
        if len(fragment) < MIN_TITLE_FRAGMENT:
            rules.append({"entry": entry, "kind": "refused", "read_as": fragment, "matched": 0,
                          "note": f"too short to use as a title fragment — it needs at least "
                                  f"{MIN_TITLE_FRAGMENT} letters and digits once punctuation is "
                                  f"ignored, or write the whole DOI instead"})
            continue
        rules.append({"entry": entry, "kind": "title", "read_as": fragment, "matched": 0,
                      "note": ""})
    return rules


def _rule_hits(rule: Mapping[str, Any], doi: str, title: str) -> bool:
    """Does this rule catch a candidate whose DOI and title are already normalised?

    A DOI must be EQUAL and a title fragment need only be CONTAINED — the asymmetry is deliberate.
    A DOI is a whole identifier and a substring of one names a different paper; a title is what a
    person can be expected to half-remember, and `MIN_TITLE_FRAGMENT` is what stops the containment
    rule from being a wildcard.
    """
    if rule["kind"] == "doi":
        return bool(doi) and doi == rule["read_as"]
    if rule["kind"] == "title":
        return bool(title) and rule["read_as"] in title
    return False                        # "refused" — a rule that was never allowed to run


def _apply_exclusions(record: SearchRecord, entries: Sequence[str]) -> int:
    """Mark every candidate the USER forbade, and write down what they forbade. Returns the count.

    The papers stay in `record.candidates`. Dropping them would make the funnel unaccountable —
    the reader could not see that the search DID find the paper and was told not to use it, which
    is precisely the fact somebody validating a search against a known review needs.

    `screen_decision` is deliberately left empty: nothing read these, so they are not among the
    abstracts `counts_of` says were read, and no model's verdict is ever invented for them.
    """
    rules = exclusion_rules(entries)
    if not rules:
        return 0
    forbidden = 0
    for candidate in record.candidates:
        doi, title = normalise_doi(candidate.doi), normalise_title(candidate.title)
        hits = [rule for rule in rules if _rule_hits(rule, doi, title)]
        if not hits:
            continue
        # every entry that catches this paper is credited, so an entry reported as matching
        # nothing really matched nothing; the REASON names the first, because a row shows one
        for rule in hits:
            rule["matched"] += 1
        candidate.state = "excluded"
        candidate.keep = False
        candidate.excluded_by_user = str(hits[0]["entry"])
        candidate.screen_reason = f'You excluded this: matched "{hits[0]["entry"]}"'
        candidate.fetch_outcome = EXCLUDED_BY_USER
        forbidden += 1
    record.exclusions = [{k: rule[k] for k in ("entry", "kind", "read_as", "matched", "note")}
                         for rule in rules]
    # an entry that did nothing is said out loud. A user who mistypes a DOI would otherwise read
    # "0 excluded" as "this search found none of that paper" rather than "your line was wrong".
    for rule in rules:
        if rule["kind"] == "refused":
            record.notes.append(f'"{rule["entry"]}" was not used as an exclusion: {rule["note"]}')
        elif not rule["matched"]:
            record.notes.append(f'nothing this search found matched the exclusion '
                                f'"{rule["entry"]}", so nothing was removed for it — check it')
    return forbidden


def _rank_and_split(candidates: Sequence[Candidate], plan: Mapping[str, Any],
                    cap: int | None) -> tuple[list[Candidate], list[Candidate]]:
    """Every candidate scored and sorted (`rank.py`), then the first `cap` and the rest.

    The rest are NOT dropped — they stay in the record, unscreened, with the reason and their
    rank, so the count a user reads is a fact about the list they see. Ranked before the cut,
    not after: the first search cut its list in arrival order and the one key paper the indexes
    returned late was lost to the order the network answered in (design 01).
    """
    ordered = rank(list(candidates), plan)
    for position, candidate in enumerate(ordered, start=1):
        candidate.relevance_why = f"rank {position}: {candidate.relevance_why}"
    if cap is None or cap < 0 or len(ordered) <= cap:
        return ordered, []
    # a seed the user handed in is read whatever its rank: it was never a candidate to cut
    chosen = ordered[:cap] + [c for c in ordered[cap:] if c.seed]
    rest = [c for c in ordered[cap:] if not c.seed]
    for candidate in rest:
        candidate.screen_reason = (f"ranked {ordered.index(candidate) + 1} of {len(ordered)}, "
                                   f"past the screening cap of {cap} — nobody read it")
    return chosen, rest


def _screen(client: Any | None, candidates: Sequence[Candidate], record: SearchRecord,
            model_roles: Mapping[str, str], budget_usd: float | None,
            protocol: Any = None, round_index: int = 0) -> Any:
    """The screening stage, with its own failure caught.

    `screen_candidates` already turns a budget stop and a per-batch model error into recorded
    outcomes; this catches the rest (a client that raises on construction of the call, a model name
    the provider rejects) so a model problem cannot lose an index search that already ran.
    """
    from .screen import ScreenOutcome

    model = (model_roles.get("screener") or model_roles.get("screen")
             or model_roles.get("secondary") or "claude-sonnet-5")
    try:
        return screen_candidates(client, candidates, question=record.question,
                                 protocol=protocol, rubric=list(record.plan.get("rubric") or []),
                                 plan=record.plan, criteria=record.criteria, model=model,
                                 budget_usd=budget_usd, round_index=round_index,
                                 cell_key_prefix=f"screen:{record.search_id}")
    except Exception as exc:                    # noqa: BLE001 - reported, never raised
        outcome = ScreenOutcome()
        outcome.notes.append(
            f"screening failed ({type(exc).__name__}: {exc}); every record is listed unscreened "
            f"for you to read"[:300])
        for candidate in candidates:
            candidate.state = "not_screened"
        return outcome


def _chase(client: Any, record: SearchRecord, transport: SearchTransport,
           model_roles: Mapping[str, str], budget_usd: float | None, protocol: Any, *,
           cancelled: Callable[[], bool] | None) -> None:
    """The citation-chasing rounds (`snowball.py`), each screened like the first pass and
    priced into the record. Never raises: a stage that dies is a note and the search goes on
    to fetch what it already has."""
    cap = round_cap(budget_usd, share=SHARE_SNOWBALL, cost_per_record=COST_PER_RECORD)

    def screen_round(chosen: Sequence[Candidate], round_index: int) -> Any:
        outcome = _screen(client, chosen, record, model_roles, budget_usd, protocol,
                          round_index=round_index)
        record.cost_usd = round(record.cost_usd + float(outcome.cost_usd or 0.0), 6)
        record.batches.extend(dict(asdict(b), round=round_index) for b in outcome.batches)
        record.notes.extend(outcome.notes)
        return outcome

    try:
        everything, rounds = snowball(record.candidates, transport=transport, plan=record.plan,
                                      screen=screen_round, cap=cap, budget_usd=budget_usd,
                                      cost_so_far=lambda: record.cost_usd, cancelled=cancelled)
    except Exception as exc:                    # noqa: BLE001 - reported, never raised
        record.notes.append(f"citation chasing failed ({type(exc).__name__}: {exc}); the search "
                            f"goes on with what the strings found"[:300])
        return
    record.candidates = everything
    record.rounds = rounds
    _measure_unique(record)
    for row in rounds:
        for note in row.get("notes") or []:
            if note not in record.notes:
                record.notes.append(str(note))


def _fetch(record: SearchRecord, staging: Path, *, transport: SearchTransport,
           max_fetch: int, max_fetch_unsure: int | None,
           probe: Callable[[Path], Mapping[str, Any]] | None,
           cancelled: Callable[[], bool] | None) -> Any:
    """Resolve the long tail with Unpaywall, then fetch, then Unpaywall again for what failed.
    Every half failure-tolerant.

    The first Unpaywall pass runs for candidates this stage would fetch, that have a DOI, and
    that have no OA URL from any discovery index — that is the population it exists for, and
    asking it about a paper Europe PMC already offered would spend a request to learn something
    we know. The list is the fetch workload itself (`fetchable`: wanted, then the unsure papers
    under their cap, then the unread), cut at `max_fetch`, so the cap bounds the requests this
    pass makes rather than being eaten by candidates it was going to skip anyway.

    The second pass is the fetch stage's `resolve_more` hook: a candidate whose every index
    route failed definitively is offered to Unpaywall once, and tried again only for the routes
    that are new. Unpaywall's own copy is often the repository PDF an index's landing page hid.
    """
    from .fetch import FetchSummary, fetchable, oa_sources

    workload = fetchable(record.candidates, max_fetch_unsure=max_fetch_unsure)[:max_fetch]
    unresolved = [c for c in workload if c.doi and not oa_sources(c)]
    unpaywall = Unpaywall()
    lock = threading.Lock()
    asked: set[str] = set()

    def resolve(candidate: Candidate) -> bool:
        """Ask Unpaywall once; merge what it said; True when a route we did not have landed."""
        with lock:
            if candidate.key in asked:
                return False
            asked.add(candidate.key)
        before = {url for _, url in oa_sources(candidate, limit=1000)}
        try:
            ids, row = unpaywall.locations(transport, candidate)
        except Exception as exc:            # noqa: BLE001 - one candidate's problem
            ids, row = {}, source_record("unpaywall", candidate.doi,
                                         error=f"{type(exc).__name__}: {exc}"[:300])
        candidate.ids.update(ids)
        # the licence is whatever Unpaywall stated, and only when nothing else stated one
        if not candidate.license and row.get("license"):
            candidate.license = str(row["license"])
        with lock:
            record.sources.append(row)
        return any(url not in before for _, url in oa_sources(candidate, limit=1000))

    if contact_email():
        for candidate in unresolved:
            if cancelled and cancelled():
                break
            resolve(candidate)
        resolve_more: Callable[[Candidate], bool] | None = resolve
    else:
        resolve_more = None
        if unresolved or workload:
            record.notes.append(
                "Unpaywall was not consulted — before the fetch for the papers no index offered "
                "a PDF for, nor after it for the ones every route failed on: it needs a real "
                "contact address, so set CANOPY_CONTACT_EMAIL to widen the fetch")

    try:
        return fetch_candidates(record.candidates, staging, transport=transport,
                                probe=probe if probe is not None else default_probe(),
                                max_fetch=max_fetch, max_fetch_unsure=max_fetch_unsure,
                                deadline_s=FETCH_DEADLINE_S, cancelled=cancelled,
                                resolve_more=resolve_more)
    except Exception as exc:                    # noqa: BLE001 - reported, never raised
        summary = FetchSummary()
        summary.notes.append(
            f"the fetch stage failed ({type(exc).__name__}: {exc}); the papers it did not reach "
            f"are listed with their links"[:300])
        return summary


def _say_what_it_cost(record: SearchRecord) -> None:
    """`record.predicted["actual"]`: what the search really billed, beside what was predicted
    for it — the plan call, the first pass (audit included), the citation-chasing rounds. The
    bench prints the two side by side; a prediction that is not measured is a guess."""
    first = [b for b in record.batches if not b.get("round")]
    rounds = [b for b in record.batches if b.get("round")]
    actual = {
        "plan_usd": round(float((record.plan or {}).get("cost_usd") or 0.0), 4),
        "screen_usd": round(sum(float(b.get("cost_usd") or 0.0) for b in first), 4),
        "audit_usd": round(sum(float(b.get("cost_usd") or 0.0) for b in first if b.get("audit")),
                           4),
        "snowball_usd": round(sum(float(b.get("cost_usd") or 0.0) for b in rounds), 4),
        "n_screened": sum(int(b.get("n_verdicts") or 0) for b in first if not b.get("audit")),
        "snowball_rounds": len([r for r in record.rounds if r.get("n_seeds")]),
        "total_usd": round(float(record.cost_usd or 0.0), 4),
    }
    predicted = dict(record.predicted or {})
    predicted["actual"] = actual
    record.predicted = predicted


def _say_the_run_cost(record: SearchRecord) -> None:
    """The bill a `begin` would start, said in the record's own words and numbers.

    The search costs cents a record; the review it becomes costs dollars a paper, and half of
    what a careful screener reads is `unsure` and now fetched. `record.predicted["run_commit"]`
    carries the numbers for the page and for `begin`; the note carries the sentence for a reader
    of `search.json` with no page in front of them.
    """
    # what `begin` would send: ticked AND readable, which is neither "fetched" (an unscreened
    # paper is fetched and unticked) nor "wanted" (a wanted paper may be paywalled)
    ticked = [c for c in record.candidates if c.keep and c.pdf_path]
    n_unsure = sum(1 for c in ticked if c.screen_decision == "unknown")
    per_paper, source = run_cost_per_paper()
    commit = run_commit(len(ticked) - n_unsure, n_unsure, per_paper)
    commit["per_paper_source"] = source
    predicted = dict(getattr(record, "predicted", None) or {})
    predicted["run_commit"] = commit
    record.predicted = predicted
    if commit["n_read"]:
        record.notes.append(
            f"{commit['n_read']} paper(s) have a PDF and are ticked for the review: "
            f"{commit['n_wanted']} the screener wanted and {commit['n_unsure']} it could not "
            f"decide about. Reading each in the review costs about ${per_paper:.2f} ({source}), "
            f"so beginning it as it stands commits about ${commit['usd']:.2f} — the "
            f"{commit['n_unsure']} unsure paper(s) are ≈ ${commit['unsure_usd']:.2f} of that. "
            f"Untick any you do not want before you begin")


# ----------------------------------------------------------------------------------- the wording
def _stop_message(why: str) -> str:
    return {"cancelled": "you stopped this search",
            "budget": "the cost cap stopped this search",
            "deadline": "this search ran out of time"}.get(why, "this stage did not run")


def _index_message(sources: Sequence[Mapping[str, Any]], n_found: int) -> str:
    live = [s for s in sources if not s.get("error") and not s.get("skipped")]
    dead = sorted({str(s.get("name")) for s in sources if s.get("error")})
    skipped = [s for s in sources if s.get("skipped")]
    text = f"{n_found} record(s) from {len({str(s.get('name')) for s in live})} index(es)"
    per_index: dict[str, int] = {}
    for source in live:
        label = str(source.get("name")) + (f" {source.get('form')}" if source.get("form") else "")
        per_index[label] = per_index.get(label, 0) + int(source.get("n_returned") or 0)
    if per_index:
        text += " (" + ", ".join(f"{k} {v}" for k, v in per_index.items()) + ")"
    if dead:
        text += f"; {', '.join(dead)} did not answer"
    if skipped:
        text += f"; {len(skipped)} string(s) too long for OpenAlex full text"
    return text


def _plan_message(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    blocks = [b for b in plan.get("blocks") or [] if b.get("terms")]
    n_terms = sum(len(b.get("terms") or []) for b in blocks)
    n_variants = sum(len(v) for b in blocks for v in (b.get("expanded") or {}).values())
    n_pruned = sum(len(b.get("pruned") or []) for b in blocks)
    n_strings = len(plan.get("strings") or {})
    sent = sum(1 for r in rows if r.get("text") and not r.get("skipped"))
    source = "a model" if plan.get("source") == "model" else "your own words"
    return (f"{n_strings} search string(s) from {len(blocks)} concept block(s) by {source}: "
            f"{n_terms} terms, {n_variants} variants, {n_pruned} pruned by width control; sent "
            f"to {sent} index form(s)")


def _screen_message(client: Any | None, counts: Mapping[str, int]) -> str:
    if client is None:
        return (f"no model was available, so none of the {counts['after_dedupe']} papers were "
                f"screened — they are all listed for you to read")
    return (f"{counts['screened']} read: {counts['included']} wanted, {counts['unsure']} unsure, "
            f"{counts['excluded']} ruled out")


def _snowball_message(rounds: Sequence[Mapping[str, Any]]) -> str:
    real = [r for r in rounds if r.get("n_seeds")]
    if not real:
        return "nothing to chase: no included or unsure paper to start from"
    n_new = sum(int(r.get("n_new") or 0) for r in real)
    n_screened = sum(int(r.get("n_screened") or 0) for r in real)
    n_inc = sum(int(r.get("n_included") or 0) for r in real)
    n_unk = sum(int(r.get("n_unknown") or 0) for r in real)
    requests = sum(sum(int(v) for v in (r.get("requests") or {}).values()) for r in real)
    return (f"{len(real)} round(s) from {sum(int(r.get('n_seeds') or 0) for r in real)} seed(s): "
            f"{n_new} new paper(s), {n_screened} screened, {n_inc} wanted, {n_unk} unsure "
            f"({requests} requests)" + (f"; stopped: {real[-1]['stopped']}"
                                       if real[-1].get("stopped") else ""))


def _fetch_message(summary: Any) -> str:
    parts = [f"{summary.n_fetched} PDF(s) fetched"
             + (f" ({summary.n_unsure_fetched} of them unsure)"
                if getattr(summary, "n_unsure_fetched", 0) else "")]
    if getattr(summary, "n_unsure_over_cap", 0):
        parts.append(f"{summary.n_unsure_over_cap} unsure over the unsure cap")
    if getattr(summary, "n_over_deadline", 0):
        parts.append(f"{summary.n_over_deadline} past the fetch deadline")
    if summary.n_paywalled:
        parts.append(f"{summary.n_paywalled} with no open-access copy")
    if summary.n_no_copy:
        # deliberately not folded into the line above: nobody screened these, so calling them
        # paywalled would put a claim about a publisher on a paper the screener never asked for
        parts.append(f"{summary.n_no_copy} unscreened with no copy we could fetch")
    if summary.n_rate_limited:
        parts.append(f"{summary.n_rate_limited} rate-limited")
    if getattr(summary, "n_unreachable", 0):
        parts.append(f"{summary.n_unreachable} unreachable")
    if summary.n_over_cap:
        parts.append(f"{summary.n_over_cap} over the fetch cap")
    return ", ".join(parts)
