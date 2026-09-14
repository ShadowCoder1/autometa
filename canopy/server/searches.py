"""A search is a job, in its own directory, with its own record — and it is not a run.

The whole file exists to keep those two words apart. A *run* reads papers and computes numbers; a
*search* only decides which papers a run will read. They share the job machinery (a thread, an
event log, `job.json`, a bearer token) because that machinery is about *background work*, not
about meta-analysis — so `SearchJobs` subclasses `JobManager` rather than copying it.

What it does NOT share is the directory. `JobManager.list()` globs `<its dir>/*/job.json`, so the
one honest way to keep a search out of `GET /api/runs` — with no filter to forget and no `kind`
check to get wrong in a later edit — is a second manager over a **sibling** directory:

    runs/       20260827-…-my-review/   job.json  protocol.yaml  uploads/  papers/  results/
    searches/   20260827-…-does-tdcs/   job.json  search.json    staging/  protocol.yaml

A search directory deliberately mirrors a run directory, one file per concern:

* `job.json`   — the registry entry (`JobManager`'s own format, `kind: "search"`);
* `search.json`— the `SearchRecord`: what was asked, what each index answered, what the screener
  said and why, every URL tried. This is the audit trail, and `begin` copies it into the run as
  `search/search.json` so the review can be read years later without this server;
* `staging/`   — PDFs, named `<sha256>.pdf` exactly as `uploads/` names them, so a fetched paper
  and an uploaded one are the same kind of thing and no candidate key ever becomes a path;
* `protocol.yaml` — the protocol text the user pasted when they started the search, if any.

Two state-machine decisions are pinned here rather than left to the endpoints, because both were
holes the design review found and both are the kind of thing that gets re-opened by accident:

1. **A search that is still running may not be edited.** `keep`, uploads, extra papers and
   `begin` all require a finished job. A running search is rewriting `search.json` from another
   thread; a `keep` toggle merged into that would be silently lost the next time the pipeline
   saved, and a `begin` half-way through the fetch stage would build a run out of whichever PDFs
   happened to have landed. The refusal is a 409 that says to wait.
2. **A search becomes exactly one run.** The run id is recorded on the search; a second `begin`
   returns the SAME run instead of spending the user's money twice, and editing a search after it
   has become a run is refused — the run already has its copy of the papers, so a change here
   would silently not be in it.

Nothing in this module knows anything about any research field, and nothing here computes a
statistic (`server/app.py`'s rule). It also never calls a model: the pipeline is injected as
`runner`, which is the seam `tests/test_search_server.py` fills with a fake.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..search.models import (KEY_RE, PHASES, Candidate, SearchRecord, counts_of, new_key,
                             project)
from .jobs import Job, JobManager
from .uploads import DEFAULT_INGEST_TIMEOUT, DEFAULT_MAX_TOTAL_MB, safe_filename

__all__ = ["DEFAULT_MAX_FETCH_UNSURE", "DEFAULT_MAX_SCREENED", "DEFAULT_MAX_USD",
           "MAX_EXCLUSIONS", "MAX_EXCLUSION_CHARS", "MAX_SEED_DOIS",
           "SEARCH_BYTE_BUDGET", "SearchJobs", "SearchOptions", "candidate_from_json",
           "default_search_runner", "pdf_name", "phase_rows", "record_from_json",
           "search_options_from", "skipped_of", "state_of", "recording_dir", "RECORD_ROOT"]

#: what a search costs at most, and how many abstracts it reads at most, when the page does not
#: say. Both are caps a *person* should be able to raise, so they are environment-tunable like
#: every other limit in this package — and both are shown by `/api/settings` so the New-search
#: form can pre-fill the numbers it is actually going to be held to.
#: 200 is `canopy.search.run.DEFAULT_MAX_SCREENED`, repeated rather than imported: reading it
#: would pull the whole search pipeline (and httpx) into every import of the server. Two numbers
#: that must agree is a smell — the pin is `test_the_default_cap_agrees_with_the_pipeline`.
DEFAULT_MAX_USD = float(os.environ.get("CANOPY_SEARCH_MAX_USD", "2") or 2)
DEFAULT_MAX_SCREENED = int(os.environ.get("CANOPY_SEARCH_MAX_SCREENED", "200") or 200)
#: 100 is `canopy.search.fetch.DEFAULT_MAX_FETCH_UNSURE`, repeated for the same reason as the
#: line above and pinned by the same test. Shown by `/api/settings` so the Find panel can price
#: what a search may commit the review to.
DEFAULT_MAX_FETCH_UNSURE = int(os.environ.get("CANOPY_SEARCH_MAX_FETCH_UNSURE", "100") or 100)

#: how many papers one search may be forbidden, and how long each entry may be.
#: A person naming their own prior work, a review they are validating against and a handful of
#: retractions is a short list; 50 lines of 300 characters is ~15 kB, which is a request body and
#: not a database. Fixed rather than environment-tunable: unlike the caps above, raising this
#: buys nobody anything a second search would not.
MAX_EXCLUSIONS = 50
MAX_EXCLUSION_CHARS = 300
MAX_SEED_DOIS = 20

#: how many bytes one search may KEEP on disk, in total, across every PDF it fetches.
#: Deliberately the SAME number the upload door enforces — same disk, same person, and a search
#: allowed to write more than a human may upload is a cap in name only. It matters because the
#: per-file cap does not bound a search: `run.DEFAULT_MAX_FETCH` (60) × `CANOPY_SEARCH_MAX_PDF_MB`
#: (50 MB) is ~3 GB against a 2 GB total. `transport.ByteBudget` was written for exactly this and
#: was never handed to a transport, so the arithmetic that was supposed to stop it never ran.
SEARCH_BYTE_BUDGET = DEFAULT_MAX_TOTAL_MB * 1e6

#: the fields each dataclass actually has, so a `search.json` written by an older (or newer)
#: build loads instead of raising `TypeError: unexpected keyword argument`. A record is an audit
#: trail: refusing to open one because it gained a field is the one failure mode it may not have.
_CANDIDATE_FIELDS = frozenset(f.name for f in dataclass_fields(Candidate))
_RECORD_FIELDS = frozenset(f.name for f in dataclass_fields(SearchRecord))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================================ options
class SearchOptions(BaseModel):
    """What a search takes from the page: two numbers, and the papers it may never propose.

    `extra="forbid"` for the same reason `RunOptions` forbids it — a typo'd cap is a search that
    quietly runs under the default and spends more than the user thought they had allowed. Which
    is also why `exclude` had to be added HERE rather than passed beside the options: an
    exclusion list this model did not know about would have 422'd the whole search.
    """

    model_config = ConfigDict(extra="forbid")

    max_usd: float | None = Field(default=None, gt=0)
    max_screened: int | None = Field(default=None, ge=1)
    #: how many papers the fetch stage may try, and how many of the papers the screener could
    #: not decide about are among them. The second is the one that prices the review: every
    #: unsure PDF fetched is a paper `begin` sends to be read at dollars a paper.
    max_fetch: int | None = Field(default=None, ge=0)
    max_fetch_unsure: int | None = Field(default=None, ge=0)
    #: rows asked of each index form per string (Europe PMC three times that); 200 / 1,000 /
    #: 3,000 on the page
    depth: int | None = Field(default=None, ge=50, le=5000)
    #: DOIs the user says must be found: checked against the strings at abstract level, a
    #: pruned term restored for them, one rewrite call, and injected as a last resort
    seed_dois: list[str] | None = Field(default=None, max_length=MAX_SEED_DOIS)
    #: citation chasing on (the default) or off
    snowball: bool | None = None
    #: DOIs and title fragments this search may never propose — the user's own decision, recorded
    #: as theirs. Bounded like every other list this server accepts: a body is not a place to put
    #: an unbounded amount of anything, and a page that can send 50 lines can send 50,000.
    exclude: list[str] | None = Field(default=None, max_length=MAX_EXCLUSIONS)

    @field_validator("seed_dois")
    @classmethod
    def _seed_entries(cls, entries: list[str] | None) -> list[str] | None:
        """Blank lines dropped; each entry must at least look like a DOI."""
        if entries is None:
            return None
        cleaned = [str(entry).strip() for entry in entries]
        cleaned = [entry for entry in cleaned if entry]
        for entry in cleaned:
            if len(entry) > MAX_EXCLUSION_CHARS or "10." not in entry:
                raise ValueError(f"a seed must be a DOI (10.xxxx/…): {entry[:60]!r} is not one")
        return cleaned

    @field_validator("exclude")
    @classmethod
    def _readable_entries(cls, entries: list[str] | None) -> list[str] | None:
        """Blank lines dropped, each entry length-capped, order and spelling otherwise untouched.

        Untouched on purpose: what the user typed is what `search.json` reports back to them, and
        a server that tidied an entry would show them a line they never wrote when it matched
        nothing. A too-long entry is refused rather than truncated for the same reason — a
        silently shortened DOI is a different DOI.
        """
        if entries is None:
            return None
        cleaned = [str(entry).strip() for entry in entries]
        cleaned = [entry for entry in cleaned if entry]
        for entry in cleaned:
            if len(entry) > MAX_EXCLUSION_CHARS:
                raise ValueError(f"one entry is longer than {MAX_EXCLUSION_CHARS} characters — "
                                 f"a DOI or a distinctive phrase from the title is enough")
        return cleaned

    def resolved(self) -> dict[str, Any]:
        """The caps this search will actually be held to — never `None`.

        Filled in HERE, at creation, rather than read from the environment at each use: the
        numbers land in `job.json`, so a search records what it was allowed to do even if the
        server's defaults change afterwards.

        `exclude` appears only when there is one. An empty key on every search that excluded
        nothing would be a fact about nothing, and `job.json` is read by people.
        """
        chosen: dict[str, Any] = {
            "max_usd": DEFAULT_MAX_USD if self.max_usd is None else float(self.max_usd)}
        # `max_screened` is an OVERRIDE (design 03 §3): absent, the pipeline screens what 70 %
        # of `max_usd` buys at $0.003 a record, ranked; present, exactly this many
        if self.max_screened is not None:
            chosen["max_screened"] = int(self.max_screened)
        if self.max_fetch is not None:
            chosen["max_fetch"] = int(self.max_fetch)
        if self.max_fetch_unsure is not None:
            chosen["max_fetch_unsure"] = int(self.max_fetch_unsure)
        if self.depth is not None:
            chosen["depth"] = int(self.depth)
        if self.snowball is not None:
            chosen["snowball"] = bool(self.snowball)
        if self.seed_dois:
            chosen["seed_dois"] = list(self.seed_dois)
        if self.exclude:
            chosen["exclude"] = list(self.exclude)
        return chosen


def search_options_from(raw: Any) -> SearchOptions:
    """The `options` object as a validated model, or a 422 naming the field that is wrong."""
    if raw is None:
        raw = {}
    if isinstance(raw, str):                               # a form field, or a page sending JSON
        try:
            raw = json.loads(raw or "{}")
        except ValueError:
            raise HTTPException(status_code=422, detail="options must be a JSON object")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="options must be a JSON object")
    # an untouched form field arrives as `""`; that means "use the default", not "zero"
    cleaned = {k: (None if isinstance(v, str) and not v.strip() else v) for k, v in raw.items()}
    try:
        return SearchOptions.model_validate(cleaned)
    except ValidationError as exc:
        first = exc.errors()[:4]
        detail = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', ()))}: {e.get('msg')}"
                           for e in first)
        raise HTTPException(status_code=422, detail=f"bad search options — {detail}")


# ============================================================================ record <-> json
def candidate_from_json(data: Mapping[str, Any]) -> Candidate:
    """One candidate out of `search.json`, ignoring anything this build does not know about."""
    kept = {k: v for k, v in dict(data).items() if k in _CANDIDATE_FIELDS}
    kept["key"] = str(kept.get("key") or "")
    return Candidate(**kept)


def record_from_json(data: Mapping[str, Any]) -> SearchRecord:
    """`search.json` as a `SearchRecord`. `counts` is dropped: it is derived, never stored."""
    kept = {k: v for k, v in dict(data).items() if k in _RECORD_FIELDS}
    kept["candidates"] = [candidate_from_json(c) for c in (kept.get("candidates") or [])
                          if isinstance(c, Mapping)]
    return SearchRecord(**kept)


# ============================================================================ views
def phase_rows(record: SearchRecord) -> list[dict[str, Any]]:
    """The ladder the page draws: every phase in `PHASES`, in order, whether it ran or not.

    A page that renders only the phases the record happens to mention draws a ladder that grows
    rungs as the search proceeds, so the user cannot see what is still coming. Every rung is
    always present; one that has not started says `pending`.
    """
    said: dict[str, Mapping[str, Any]] = {}
    for entry in record.phases:
        if isinstance(entry, Mapping) and entry.get("name"):
            said[str(entry["name"])] = entry
    rows = [_phase_row(name, said.pop(name, {})) for name in PHASES]
    # anything the pipeline reported that `PHASES` does not name goes last rather than vanishing:
    # a phase the page cannot place is still a phase that happened, and hiding it would make the
    # ladder disagree with the seconds and the cost beside it
    rows.extend(_phase_row(name, entry) for name, entry in said.items())
    return rows


def _phase_row(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": str(entry.get("status") or "pending"),
            "message": str(entry.get("message") or ""),
            "seconds": round(float(entry.get("seconds") or 0.0), 3)}


def pdf_name(candidate: Candidate) -> str:
    """What the RUN will call this paper — `bock_2005.pdf`, not `c3f9a1b2c4d5.pdf`.

    `make_run` records the first name it sees for each sha256 in `filenames.json`, and that string
    is what every later screen shows beside the paper. A candidate key there would make a review
    built from a search unreadable next to one built from a folder of PDFs.

    The name must end in `.pdf`: `PdfSource` refuses anything else, and a paper title used raw
    would 400 every `begin` with an error about the file rather than about the name.
    """
    if candidate.upload_filename.lower().endswith(".pdf"):
        return safe_filename(candidate.upload_filename)
    stem = "".join(ch if ch.isalnum() else "_"
                   for ch in (candidate.study_label() or candidate.key)).strip("_")
    return f"{(stem or candidate.key)[:80].lower()}.pdf"


def skipped_of(job: Job) -> list[dict[str, str]]:
    """The papers `begin` could not put in the run — `{key, name, reason}` each.

    Lives in `job.options`, which is `job.json`, so it survives the request that produced it: it
    is written once, when the run is built, and every later reader (a poll, a reload, another
    browser) gets the same list. `key` is the candidate key rather than the display name, because
    the page has to be able to point at the ROW a skip belongs to, and two papers can be called
    the same thing.
    """
    rows = [dict(s) for s in (job.options.get("skipped") or []) if isinstance(s, Mapping)]
    return [{"key": str(r.get("key") or ""), "name": str(r.get("name") or ""),
             "reason": str(r.get("reason") or "")} for r in rows]


def state_of(job: Job, record: SearchRecord) -> dict[str, Any]:
    """Everything `GET /api/searches/{id}` answers — the job's state and the record's contents.

    The candidates are `project()`ed, never sent raw: the record holds up to 4 kB of publisher
    abstract per row and the page shows none of it, so 200 candidates would otherwise be a
    megabyte of JSON on every poll.

    `stopped_because` is beside `status` and is NOT folded into it. A search that hit its cap
    still finishes `done`, and a user told only "done" would never learn that their question was
    answered from the first 60 abstracts of 400.
    """
    return {
        "search_id": job.run_id,
        "status": job.status,
        "question": record.question,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        # the LARGER of the two ledgers, never the record's alone. `job.cost_usd` is what the
        # client was actually billed (copied off `client.total_cost()` at every phase boundary);
        # `record.cost_usd` is assembled from the stages that managed to report a number, and a
        # stage that was billed and then failed — a `BudgetExceeded` batch, a query call the
        # provider charged for before it errored — is in the first and not the second. Taking the
        # record's would show a user their search getting CHEAPER as it finished.
        "cost_usd": round(max(float(record.cost_usd or 0.0), float(job.cost_usd or 0.0)), 6),
        "error": job.error,
        "stopped_because": record.stopped_because,
        "counts": counts_of(record.candidates, record.possible_duplicates),
        "phases": phase_rows(record),
        "sources": [dict(s) for s in record.sources if isinstance(s, Mapping)],
        "queries": [dict(q) for q in record.queries if isinstance(q, Mapping)],
        "query_source": record.query_source,
        "criteria": list(record.criteria),
        "notes": list(record.notes),
        "possible_duplicates": [dict(d) for d in record.possible_duplicates
                                if isinstance(d, Mapping)],
        # what beginning this search would cost, in papers and dollars (`run_commit`), beside the
        # predicted search cost: the number BLOCKER-3 said had to be visible before the button
        "predicted": dict(record.predicted or {}),
        # the concept blocks, width measurements and seed check behind the strings, and the
        # citation-chasing rounds: the page draws them above the strings and in the flow line
        "plan": dict(record.plan or {}),
        "rounds": [dict(r) for r in (record.rounds or []) if isinstance(r, Mapping)],
        # what the USER forbade, one row per entry, including the entries that caught nothing.
        # Sent on every poll like the counts are: an exclusion the page never mentions is one the
        # user has to take on faith, and a mistyped DOI they would take on faith wrongly.
        "exclusions": [dict(x) for x in record.exclusions if isinstance(x, Mapping)],
        # the papers `begin` dropped, with the reason for each. Sent on EVERY poll and not only
        # in `begin`'s own answer: the answer is seen once, by one tab, and then the page
        # navigates to the run — a reload used to lose the fact that 39 of 40 papers never made
        # it, leaving a one-paper review that claimed to be forty.
        "skipped": skipped_of(job),
        "candidates": [project(c) for c in record.candidates],
        # the run this search became, if it has: the page needs it to link there, and its
        # presence is also what makes the search read-only (see the module docstring)
        "run_id": str(job.options.get("run_id") or ""),
    }


# ============================================================================ the pipeline seam
def default_search_runner(record: SearchRecord, *, search_dir: Path, options: Mapping[str, Any],
                          client_factory: Callable[..., Any], cancel: threading.Event,
                          progress: Callable[[Mapping[str, Any]], None],
                          save: Callable[..., None]) -> SearchRecord:
    """The real search pipeline, behind the seam this module defines — and it is an ADAPTER.

    `canopy.search.run.run_search` takes what a *search* needs (a transport, model roles, a
    staging directory, two callbacks) and knows nothing about jobs, tokens or SSE. This function
    is the only place the two vocabularies meet, so the pipeline never grows a parameter because
    of the server and the server never grows one because of the pipeline.

    Everything is imported at CALL time. Late on purpose: the search pipeline pulls `httpx` and
    the whole transport in, and `create_app` is imported by the CLI, by every server test and by
    anything embedding this package — none of which should fail because a search dependency is
    missing. A build without it answers the request, starts the job, and reports one readable
    error on the search itself rather than failing to start at all.
    """
    try:
        from ..config import MODELS, api_key, live_enabled
        from ..search.run import run_search
        from ..search.transport import ByteBudget, HttpxTransport
    except ImportError as exc:                             # pragma: no cover - shipped together
        raise RuntimeError("this build has no search pipeline "
                           f"(canopy.search.run.run_search): {exc}") from None
    from .jobs import default_client_factory

    max_usd = options.get("max_usd")
    budget_usd = float(max_usd) if max_usd else None

    # A search with NO model is a supported way to run, not a failure: the queries come from the
    # user's own words and nothing is screened. So the client is built only when there is
    # something to build it from — an injected factory (a fake, a replayer) always counts, a real
    # key or `CANOPY_LIVE` counts, and nothing else does. This mirrors `JobManager.key_required`.
    client = None
    if client_factory is not default_client_factory or api_key() or live_enabled():
        client = client_factory(run_dir=search_dir, budget_usd=budget_usd, concurrency=1,
                                purpose="search")

    def on_phase(name: str, status: str, message: str, seconds: float) -> None:
        """Keep the ladder on disk live, and put the same words on the event stream.

        `run_search` returns its record only at the end, so without this a poll would see an
        empty search for however long the indexes take — and the phase ladder is the one thing
        the page has to show while it waits.
        """
        rows = [p for p in record.phases if str(p.get("name")) != name]
        rows.append({"name": name, "status": status, "message": message, "seconds": seconds})
        record.phases = rows
        # the cost comes off the CLIENT, not off the record: `run_search` fills its record's
        # `cost_usd` in at the end, and a page told "$0.00" for the whole of a screening pass is
        # being told the one number the cap is about, wrongly
        with contextlib.suppress(Exception):
            record.cost_usd = float(client.total_cost()) if client is not None else 0.0
        save(record)
        progress({"stage": name, "paper": "", "status": status,
                  "cost_so_far": record.cost_usd, "message": message})

    # S1: the whole point of `ByteBudget` is that ONE object is shared by every fetch worker for
    # the life of a search, and it only bounds anything if a transport is holding it. Built here
    # rather than inside `HttpxTransport` because the number is the SERVER's (this disk, this
    # deployment's cap), not the transport's — and what is already staged is subtracted, so a
    # search that a person has attached PDFs to cannot fetch its way past the total.
    budget = ByteBudget(max(0.0, SEARCH_BYTE_BUDGET - _staged_bytes(Path(search_dir) / "staging")))
    transport: Any = HttpxTransport(budget=budget)
    record_dir = recording_dir(record.search_id)
    if record_dir is not None:
        # every index answer and every fetch written down, so the bench can replay this search
        # with no network. Opt-in by environment and never on by default: a fixture directory
        # grows by megabytes per search.
        from ..search.transport import RecordingTransport

        transport = RecordingTransport(transport, record_dir)
        record.notes.append(f"every index response of this search was recorded under "
                            f"{record_dir} (CANOPY_SEARCH_RECORD)")

    # `run_search` builds and returns its OWN record — including the phases it just reported —
    # and `SearchJobs._run` saves whatever comes back, so the live copy above is replaced by the
    # finished one rather than merged with it.
    return run_search(
        question=record.question, protocol=_protocol_or_none(search_dir),
        transport=transport, client=client, model_roles=dict(MODELS),
        budget_usd=budget_usd,
        max_screened=(int(options["max_screened"]) if options.get("max_screened") else None),
        # already validated and length-capped by `SearchOptions`; absent means nothing was forbidden
        exclude=[str(entry) for entry in (options.get("exclude") or ())],
        staging_dir=search_dir / "staging", on_phase=on_phase, cancelled=cancel.is_set,
        search_id=record.search_id,
        # the fetch caps, the depth, the seeds and the snowball switch only when the page set
        # them; the pipeline's own defaults otherwise
        **{name: int(options[name]) for name in ("max_fetch", "max_fetch_unsure", "depth")
           if options.get(name) is not None},
        **({"seed_dois": [str(d) for d in options["seed_dois"]]} if options.get("seed_dois")
           else {}),
        **({"chase_citations": bool(options["snowball"])} if options.get("snowball") is not None
           else {}))


#: where `CANOPY_SEARCH_RECORD=1` puts a search's recordings: the bench's fixture tree, one
#: directory per search id. Any other value is taken as a directory of its own.
RECORD_ROOT = Path(__file__).resolve().parents[2] / "validation" / "search_bench" / "fixtures"


def recording_dir(search_id: str) -> Path | None:
    """The directory this search's index answers are recorded under, or None when recording is
    off. Read per call, not at import, so a test can set and unset it."""
    raw = os.environ.get("CANOPY_SEARCH_RECORD", "").strip()
    if not raw or raw.lower() in ("0", "false", "no"):
        return None
    if raw in ("1", "true", "yes"):
        return RECORD_ROOT / safe_filename(str(search_id or "search"))
    return Path(raw)


def _staged_bytes(staging: Path) -> int:
    """How many bytes of PDF a search is already holding. One definition, two callers.

    `SearchJobs.staged_bytes` charges it against the upload cap and `default_search_runner`
    subtracts it from the fetch budget; two copies of this sum would be two caps that agree until
    somebody edits one of them.
    """
    if not staging.is_dir():
        return 0
    return sum(p.stat().st_size for p in staging.glob("*.pdf") if p.is_file())


def _protocol_or_none(search_dir: Path) -> Any:
    """The protocol the user pasted when they started the search, if it parses.

    If it does not, the search still runs: a half-written protocol makes the queries worse (the
    builder falls back to the question alone), and refusing to search at all because a draft is
    incomplete would be the wrong trade at the one moment the user has not written it yet.
    """
    path = Path(search_dir) / "protocol.yaml"
    if not path.is_file():
        return None
    try:
        from ..protocol import load_protocol

        return load_protocol(path)
    except Exception:                                      # pragma: no cover - a draft protocol
        return None


# ============================================================================ the registry
class SearchJobs(JobManager):
    """Every search this server knows about — same machinery as runs, different directory.

    `runner` is the whole pipeline as one callable, and it is called with keywords only:

        runner(record, *, search_dir, options, client_factory, cancel, progress, save)

    * `record`         — the `SearchRecord` to fill in. Mutated in place, or a new one returned.
    * `search_dir`     — the search's own directory; PDFs go in `search_dir / "staging"`.
    * `options`        — `{"max_usd": float, "max_screened": int}`, already resolved, plus
                         `"exclude"` (the papers the user forbade) when they named any.
    * `client_factory` — the same seam runs use; a manager built with a fake never calls a model.
    * `cancel`         — a `threading.Event`; the pipeline checks it and stops.
    * `progress`       — one dict per step, published to the SSE stream (`cost_so_far` is read
                         off it, so the cost on the page is live rather than final-only).
    * `save`           — `save(record)` writes `search.json` NOW. The GET endpoint reads from
                         disk, so this is how a poll sees a phase finish; call it at every phase
                         boundary. The manager saves once more when the runner returns.

    It returns the record (or `None`, meaning "I mutated yours"). Anything it raises becomes the
    search's `error`, exactly as a run's exception does.
    """

    def __init__(self, searches_dir: str | Path,
                 client_factory: Callable[..., Any] | None = None,
                 runner: Callable[..., Any] | None = None,
                 max_active: int | None = None,
                 ingest_timeout: float = DEFAULT_INGEST_TIMEOUT):
        super().__init__(searches_dir, client_factory=client_factory,
                         ingest_timeout=ingest_timeout,
                         **({} if max_active is None else {"max_active": max_active}))
        self.runner = runner or default_search_runner
        # a fake runner never reaches a model however the client is built, so a server given one
        # must not tell the user to go and configure an API key
        self.uses_real_models = client_factory is None and runner is None
        self._record_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------ creation
    def create_search(self, *, question: str, options: Mapping[str, Any],
                      protocol_text: str = "") -> tuple[Job, SearchRecord]:
        """A directory, a job with `kind == "search"`, and an empty record on disk."""
        job = self.create(title=question.strip()[:60], options=dict(options))
        job.kind = "search"                                # never "run": the two lists are separate
        job.save()
        record = SearchRecord(search_id=job.run_id, question=question.strip(),
                              created_at=job.created_at)
        self.save_record(job, record)
        if protocol_text.strip():
            # kept verbatim, like a run's protocol: it is what the query builder reads, and what
            # `begin` falls back to when the page does not resend it
            (job.run_dir / "protocol.yaml").write_text(protocol_text, encoding="utf-8")
        return job, record

    # ------------------------------------------------------------------ the record on disk
    def record_path(self, job: Job) -> Path:
        return job.run_dir / "search.json"

    def staging_dir(self, job: Job) -> Path:
        return job.run_dir / "staging"

    def protocol_text(self, job: Job) -> str:
        path = job.run_dir / "protocol.yaml"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def load_record(self, job: Job) -> SearchRecord:
        """The record as it stands on disk. A missing or unreadable file is an EMPTY record.

        Empty rather than an exception: a search whose first phase has not saved yet is a normal
        state that the page polls through, and a 500 there would look like a broken server.
        """
        path = self.record_path(job)
        data: Any = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:                             # pragma: no cover - defensive
                data = {}
        record = record_from_json(data if isinstance(data, Mapping) else {})
        record.search_id = record.search_id or job.run_id
        record.question = record.question or job.title
        return record

    def save_record(self, job: Job, record: SearchRecord) -> None:
        """Write `search.json` atomically — a poll must never read half a record.

        The papers `begin` dropped ride along under `skipped`. They are not a `SearchRecord`
        field: nothing the pipeline does produces them, they are a fact about the run this search
        became, and `search.json` is the file a reviewer opens years later to ask "why is this
        paper not in the review?". A skip that lived only in `job.json` answered that question
        nowhere a person would look.
        """
        data = record.to_json()
        skipped = skipped_of(job)
        if skipped:
            data["skipped"] = skipped
        path = self.record_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str),
                       encoding="utf-8")
        with contextlib.suppress(OSError):                 # exotic filesystem
            tmp.chmod(0o600)
        tmp.replace(path)

    @contextlib.contextmanager
    def record_lock(self, job: Job) -> Iterator[None]:
        """Serialise read-modify-write on one search's record.

        Two uploads arriving together each load the record, add their own candidate and save;
        without this the second overwrites the first and one PDF is on disk but in nobody's list.
        """
        with self._lock:
            lock = self._record_locks.setdefault(job.run_id, threading.Lock())
        with lock:
            yield

    # ------------------------------------------------------------------ candidates
    def candidate(self, record: SearchRecord, key: str) -> Candidate | None:
        """The candidate with this key, or `None` — and the key is CHECKED before anything else.

        `KEY_RE` first, always. A candidate key arrives in a URL path, and although nothing here
        joins it onto a directory (a PDF is stored under its own sha256, never under a key), the
        one rule that keeps that true is that a key which is not `[cu]` + 12 hex digits never
        reaches any code below this line. A bad shape answers exactly as an unknown key does, so
        the endpoint's 404 is not an oracle for "which keys exist".
        """
        if not KEY_RE.match(str(key or "")):
            return None
        return next((c for c in record.candidates if c.key == key), None)

    def relative_pdf(self, job: Job, path: Path) -> str:
        """A staged PDF as the record stores it — `staging/<sha256>.pdf`, relative to the search."""
        return Path(path).resolve().relative_to(job.run_dir.resolve()).as_posix()

    def pdf_on_disk(self, job: Job, candidate: Candidate) -> Path | None:
        """The candidate's PDF, or `None` when there is not one — path-checked even so.

        `pdf_path` is written by this module and by the pipeline, never by a browser, but it is
        still a string in a JSON file that a person can edit; a `..` in it must not read a file
        outside the search.
        """
        if not candidate.pdf_path:
            return None
        target = (job.run_dir / candidate.pdf_path).resolve()
        if not str(target).startswith(str(job.run_dir.resolve()) + os.sep):
            return None
        return target if target.is_file() else None

    def attach_pdf(self, job: Job, record: SearchRecord, candidate: Candidate, *,
                   path: Path, filename: str, probe: Mapping[str, Any]) -> Candidate:
        """A human's PDF for a paper the search could not fetch. Saves the record."""
        candidate.pdf_path = self.relative_pdf(job, path)
        candidate.pdf_pages = int(probe.get("n_pages") or 0) or None
        candidate.pdf_bytes = path.stat().st_size
        candidate.upload_filename = filename
        # `uploaded` and not `fetched`: the page says who supplied each PDF, and a reviewer
        # reading `search.json` afterwards needs to know which papers a person went and got.
        candidate.state = "uploaded"
        # supplying the PDF IS the decision to include it; a user who then changes their mind
        # toggles `keep` back off, which is one click and leaves the file where it is
        candidate.keep = True
        self.save_record(job, record)
        return candidate

    def add_extra(self, job: Job, record: SearchRecord, *, path: Path, filename: str,
                  probe: Mapping[str, Any]) -> Candidate:
        """A paper the search missed, added by hand. Saves the record.

        Keyed on the file's own sha256, so sending the same PDF twice updates one candidate
        rather than adding two — the same rule `uploads/` uses for a run.
        """
        key = new_key("u", Path(path).stem)
        candidate = next((c for c in record.candidates if c.key == key), None)
        if candidate is None:
            candidate = Candidate(key=key, title=Path(filename).stem[:200],
                                  # `found_by` answers "how did this get here?" for every row,
                                  # and for this row the honest answer is: a person put it there
                                  found_by=["you"], state="extra")
            record.candidates.append(candidate)
        candidate.pdf_path = self.relative_pdf(job, path)
        candidate.pdf_pages = int(probe.get("n_pages") or 0) or None
        candidate.pdf_bytes = path.stat().st_size
        candidate.upload_filename = filename
        candidate.state = "extra"
        candidate.keep = True
        self.save_record(job, record)
        return candidate

    def set_keep(self, job: Job, record: SearchRecord, candidate: Candidate,
                 keep: bool) -> Candidate:
        """The one field a human owns. `state` is untouched: a paper dropped by hand still shows
        why it was found and what the screener thought of it."""
        candidate.keep = bool(keep)
        self.save_record(job, record)
        return candidate

    def kept_pdfs(self, job: Job, record: SearchRecord) -> list[tuple[Candidate, Path]]:
        """The papers `begin` will send to the run: kept AND with a readable PDF on disk.

        Both halves matter. A paywalled paper the user kept but never supplied has no bytes to
        send, and a fetched paper they unticked is not theirs to include — sending either would
        make the run disagree with the list the user was looking at.
        """
        out: list[tuple[Candidate, Path]] = []
        for candidate in record.candidates:
            if not candidate.keep:
                continue
            path = self.pdf_on_disk(job, candidate)
            if path is not None:
                out.append((candidate, path))
        return out

    def staged_bytes(self, job: Job) -> int:
        """How much this search already holds, so the total-size cap counts the whole search."""
        return _staged_bytes(self.staging_dir(job))

    def orphan_staged(self, job: Job, record: SearchRecord) -> list[Path]:
        """Staged PDFs no candidate in this record points at. Read-only, and lock-free by design.

        The check is separate from the repair so a poll can ask "is there anything to fix?"
        without taking `record_lock`: `begin` holds that lock for as long as it takes to copy
        forty PDFs into a run, and a `GET` that queued behind it would look like a hung page.
        """
        staging = self.staging_dir(job)
        if not staging.is_dir():
            return []
        known = {c.pdf_path for c in record.candidates if c.pdf_path}
        return [path for path in sorted(staging.glob("*.pdf"))
                if path.is_file() and self.relative_pdf(job, path) not in known]

    def recover_staged(self, job: Job, record: SearchRecord) -> int:
        """Give every staged PDF that no candidate points at a row of its own. Returns how many.

        `run_search` builds its OWN record and hands it back only when it returns, so a server
        that dies mid-search leaves `staging/` holding PDFs that `search.json` does not mention.
        Nothing could reach them again: `kept_pdfs` walks the candidates, so the files were
        invisible to `begin`, unreachable from the page — and still charged against the search's
        byte cap by `staged_bytes`, which is the worst of both. Meanwhile the page said "Nothing
        was lost". This is the half of that sentence the server can actually make true: the PDFs
        the search had already fetched come back as papers a person can read, untick, or begin a
        review from.

        What it does NOT do is invent metadata. There is no title, author or DOI for these rows —
        that knowledge died with the record — so the title says what the file is and where it came
        from. A plausible-looking citation on a paper nobody screened would be the one kind of
        fiction this package refuses everywhere else. The note it appends names what was lost
        (the queries, the index rows, the screening decisions), because "recovered" and "nothing
        happened" are different facts.

        Idempotent, and only ever called for a search that has STOPPED. While the pipeline is
        running every staged file is legitimately an orphan — its record is still in memory — and
        a recovery there would race the record the runner is about to return.
        """
        keys = {c.key for c in record.candidates}
        recovered = 0
        for path in self.orphan_staged(job, record):
            # keyed on the file's own sha256, exactly as `add_extra` is, so a second call to this
            # method — or a person who later uploads the same paper — updates one row, not two
            key = new_key("u", path.stem)
            if key in keys:
                continue
            keys.add(key)
            record.candidates.append(Candidate(
                key=key, title=("a PDF this search fetched before it was interrupted "
                                f"({path.stem[:12]}…)"),
                found_by=["recovered"], state="fetched", fetch_outcome="fetched",
                # ticked, because that is what it was: only a paper the screener wanted is ever
                # fetched, so unticking it here would silently drop a paper the search chose
                keep=True, pdf_path=self.relative_pdf(job, path), pdf_bytes=path.stat().st_size))
            recovered += 1
        if recovered:
            record.notes.append(
                f"this search stopped before it could save what it found: {recovered} PDF(s) it "
                f"had already fetched are listed below, recovered from disk, but the queries, the "
                f"index results and the screening decisions behind them were lost — search again "
                f"to rebuild that list")
            self.save_record(job, record)
        return recovered

    # ------------------------------------------------------------------ the run it became
    def run_id_of(self, job: Job) -> str:
        return str(job.options.get("run_id") or "")

    def remember_run(self, job: Job, run_id: str, skipped: list[dict[str, str]]) -> None:
        """Record which run this search became, so a second `begin` returns it rather than
        building a second one out of the same papers and the same money.

        `skipped` is `{key, name, reason}` per dropped paper — the candidate key included, so the
        page can mark the row rather than print a number.
        """
        job.options["run_id"] = str(run_id)
        job.options["skipped"] = [{"key": str(s.get("key") or ""),
                                   "name": str(s.get("name") or ""),
                                   "reason": str(s.get("reason") or "")} for s in skipped]
        job.save()

    def skipped_of(self, job: Job) -> list[dict[str, str]]:
        return skipped_of(job)

    # ------------------------------------------------------------------ running
    def _run(self, job: Job) -> None:                      # noqa: D102 - overrides JobManager
        """One search, in its own thread. The pipeline is `self.runner`; nothing here is a model.

        Mirrors `JobManager._run` deliberately — same status transitions, same `_finish`, same
        cancellation semantics — so a search and a run behave identically to the page that polls
        them and to the SSE stream that follows them.
        """
        from ..pipeline.run import RunCancelled

        record = self.load_record(job)
        job.status = "running"
        job.started_at = _now()
        job.save()

        def progress(event: Mapping[str, Any]) -> None:
            job.cost_usd = float(event.get("cost_so_far") or job.cost_usd)
            job.publish(dict(event))

        def save(current: SearchRecord | None = None) -> None:
            self.save_record(job, current or record)

        try:
            returned = self.runner(record, search_dir=job.run_dir, options=dict(job.options),
                                   client_factory=self.client_factory, cancel=job.cancel,
                                   progress=progress, save=save)
            record = returned if isinstance(returned, SearchRecord) else record
            # never DOWN. `job.cost_usd` is the live ledger `on_phase` copied off the client — the
            # money the provider actually billed — and `record.cost_usd` is the sum of the stages
            # that managed to report one. A batch billed inside a `BudgetExceeded`, or a query
            # call charged for and then failed, is in the first number and missing from the
            # second, so assigning the record's here made a finished search cost less than it had
            # cost a moment earlier, on the user's own screen.
            job.cost_usd = max(float(record.cost_usd or 0.0), float(job.cost_usd or 0.0))
            if job.cancel.is_set() and not record.stopped_because:
                record.stopped_because = "cancelled"
            self.save_record(job, record)
            if job.cancel.is_set():
                self._finish(job, "cancelled", "stopped by the reviewer")
            else:
                counts = counts_of(record.candidates, record.possible_duplicates)
                # `job.cost_usd`, the larger of the two ledgers (see above): the sentence a user
                # keeps must not be smaller than the number they watched climb
                self._finish(job, "done", f"{counts['after_dedupe']} paper(s), "
                                          f"{counts['fetched']} fetched, ${job.cost_usd:.2f}")
        except RunCancelled:
            record.stopped_because = record.stopped_because or "cancelled"
            self.save_record(job, record)
            self._finish(job, "cancelled", "stopped by the reviewer")
        except Exception as exc:
            # whatever the pipeline got to keeps its place on disk: a search that died in the
            # fetch stage still has its queries, its counts and its screened list, and throwing
            # them away would make the error unreadable
            with contextlib.suppress(Exception):           # pragma: no cover - defensive
                self.save_record(job, record)
            self._finish(job, "error", f"{type(exc).__name__}: {exc}")
