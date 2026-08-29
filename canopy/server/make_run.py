"""Everything that turns a protocol and some PDFs into a run directory — for BOTH doors.

`POST /api/runs` (a person's upload) and `POST /api/searches/{id}/begin` (papers a search found)
must produce byte-identical runs. The only way to guarantee that is for them to run the same
code, so the body of `create_run` lives here and the endpoint is now a caller like any other.
Nothing in this module knows what a search is, and nothing here imports `app` — the arrow points
one way, and `canopy/cli.py` imports `serve` from `app` inside a function, so there is no cycle.

Read `make_run`'s docstring before changing anything in it: every line of that function is a
behaviour some test or some incident put there, and several of them look like tidying
opportunities that are not.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Mapping, Sequence

import yaml
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..config import MODELS
from ..models import Protocol, StatsSettings
from ..protocol import apply_profile, available_profiles, dump_protocol, load_yaml_strict
from .jobs import Job, JobManager
from .uploads import (DEFAULT_MAX_FILES, UploadRejected, probe_pdf, safe_filename,
                      stream_upload)

__all__ = ["MAX_CONCURRENCY", "MAX_PROTOCOL_BYTES", "PdfSource", "RunOptions", "make_run",
           "options_from_json", "parse_protocol", "protocol_text_or_413", "validation_message"]

MAX_CONCURRENCY = 16
#: a protocol is a page of YAML; anything larger is a mistake or an attack
MAX_PROTOCOL_BYTES = 1_000_000


def validation_message(error: ValidationError) -> str:
    parts = []
    for item in error.errors()[:8]:
        where = ".".join(str(x) for x in item.get("loc", ()) if x != "__root__")
        parts.append(f"{where or 'protocol'}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def parse_protocol(text: str) -> Protocol:
    """The uploaded YAML as a `Protocol`, or a 422 a person can act on."""
    try:
        # strict: a repeated key would otherwise drop everything under the first copy in silence
        raw = load_yaml_strict(text, "the protocol")
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"the protocol is not valid YAML: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="the protocol must be a YAML mapping")
    try:
        protocol = Protocol.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422,
                            detail=f"the protocol is incomplete — {validation_message(exc)}")
    if protocol.stats.profile not in available_profiles():
        # caught at upload, where the person who named it is still looking: left for the run to
        # discover, the same typo is a paid job that dies on its first protocol load
        raise HTTPException(status_code=422,
                            detail=f"unknown stats profile {protocol.stats.profile!r} "
                                   f"(available: {available_profiles()})")
    return protocol


def protocol_text_or_413(text: str) -> str:
    """The protocol text, refused with the SAME 413 whichever door it came through.

    `POST /api/runs` has always answered an over-large protocol with this status and this
    sentence. A second door that silently truncated at the same limit — which is what slicing
    without checking does — would turn one honest 413 into a confusing 422 from the YAML
    parser, about a document the user sent whole.
    """
    if len(text.encode("utf-8", "replace")) > MAX_PROTOCOL_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"a protocol may not be larger than "
                                   f"{MAX_PROTOCOL_BYTES / 1e6:.0f} MB")
    return text


class RunOptions(BaseModel):
    """Everything the New-run form may ask for, and nothing else.

    This is a schema rather than a pile of `float(...)` calls because the caller is a browser and
    the answer to `{"budget_usd": "lots"}` must be a 422 that says which field is wrong — not a
    500 from a `ValueError` nobody caught.
    """

    model_config = ConfigDict(extra="forbid")

    budget_usd: float | None = Field(default=None, gt=0)
    max_usd_per_paper: float | None = Field(default=None, gt=0)
    max_papers: int | None = Field(default=None, ge=1)
    concurrency: int = Field(default=4, ge=1, le=MAX_CONCURRENCY)
    resume: bool = True
    start: bool = True
    name: str = Field(default="", max_length=60)
    profile: str | None = None
    models: dict[str, str] = Field(default_factory=dict)

    @field_validator("budget_usd", "max_usd_per_paper", "max_papers", "profile", mode="before")
    @classmethod
    def _blank_is_absent(cls, value: Any) -> Any:
        """An untouched form field arrives as `""`; that means "no cap", not "zero"."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("profile")
    @classmethod
    def _known_profile(cls, value: str | None) -> str | None:
        if value is not None and value not in available_profiles():
            raise ValueError(f"unknown stats profile (available: {available_profiles()})")
        return value

    @field_validator("models")
    @classmethod
    def _known_roles(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(value) - set(MODELS))
        if unknown:
            raise ValueError(f"unknown model role(s) {unknown} (roles: {sorted(MODELS)})")
        return {k: str(v)[:80] for k, v in value.items() if str(v).strip()}


def options_from_json(raw: str) -> RunOptions:
    """The `options` form field as a validated model, or a 422 naming the field that is wrong."""
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        raise HTTPException(status_code=422, detail="options must be a JSON object")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="options must be a JSON object")
    try:
        return RunOptions.model_validate(parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"bad options — {validation_message(exc)}")


@dataclass(frozen=True)
class PdfSource:
    """One PDF on its way into a run: what a person calls it, and how to open its bytes.

    `stream` is a factory, not an open file: `make_run` may be entered and abandoned (a refusal
    below the loop, a cleanup), and a caller that handed over a live handle would have to know
    which of those paths closed it. A fresh reader per call means nobody has to know.

    The name must end in `.pdf`: `stream_upload` refuses anything else, so a candidate title
    used as a filename would fail every begin and the failure would read as a bad PDF rather
    than a naming bug. It is NOT enforced in `__post_init__` — that raised a bare ValueError
    while the endpoint was building its source list, i.e. BEFORE `make_run`'s ordered refusals
    ran, which turned an upload that had always been an honest 400 into a 500. `stream_upload`
    makes that judgement, in the loop, where it becomes the status it always was.
    """

    name: str
    stream: Callable[[], IO[bytes]]


def make_run(manager: JobManager, *, protocol_text: str, options: RunOptions,
             sources: Sequence[PdfSource], max_upload_bytes: float, max_total_bytes: float,
             probe_timeout: float, max_files: int = DEFAULT_MAX_FILES,
             copy_into: Mapping[str, str] | None = None,
             on_source_rejected: str = "fail",
             rejected_out: list[dict[str, str]] | None = None,
             stream_upload: Callable[..., Any] = stream_upload,
             probe_pdf: Callable[..., Any] = probe_pdf) -> Job:
    """A protocol and some PDFs become a run directory and an unstarted job.

    Returns the job CREATED BUT NOT STARTED, so each caller keeps its own ordering of
    `job.save()` and `_start(...)`. That is deliberate and load-bearing: `_start` sits OUTSIDE
    the try/except below, so a 429 from a busy manager leaves the run directory on disk and
    startable later rather than deleting the upload the user just waited for.

    The refusals, in the order `POST /api/runs` has always made them — a helper that checks the
    same things in a different order is not the same endpoint:

    1. no protocol text                       → 422
    2. no sources                             → 422
    3. more sources than `max_files`          → 413
    4. `start` with no API key configured     → 400
    5. `start` with no free slot              → 429
    6. the protocol does not parse            → 422 (`parse_protocol`)

    (The 413 for an over-large protocol fires before all of these, in the caller, because only
    the caller knows whether the text arrived as a form field or an upload —
    `protocol_text_or_413` is the shared rule.)

    Invariants inside the loop, each of which some test or some incident put there:

    * `remaining` is charged for EVERY file, duplicates included. Two copies of one 40 MB PDF
      consume 80 MB of the allowance although only one file lands — both were read off the
      wire. Skipping the decrement for a duplicate would move when a 413 fires.
    * `first_time` is computed BEFORE `saved.setdefault`. Reversing those two lines makes it
      always False and silently stops probing everything after the first file.
    * `saved` keys on `str(path)` (`…/uploads/<sha>.pdf`) and the FIRST name wins, so
      `filenames.json` records what the person called the paper the first time they sent it.
    * `job.n_files` is the number of unique shas, not of uploaded files.
    * the cleanup catches `BaseException`, not `Exception`: a KeyboardInterrupt mid-upload must
      still remove the half-built run rather than leaving it to be resumed.

    `on_source_rejected` is the one behaviour the two doors do NOT share. An upload is a thing
    the person just chose, so a bad file fails the request ("fail") and they can fix it. A
    search's staged PDFs were fetched by a machine hours earlier, and destroying a 40-paper run
    because one of them times out in the probe is not a service — those are skipped and
    reported ("skip"). Each skip lands in `rejected_out` as `{name, reason}`; a caller that
    passes no list is saying it expects none, which is true of every "fail" caller.

    "skip" survives a bad source WHATEVER it raises, and that is the point of it: a staged file
    that has been deleted since it was fetched raises `FileNotFoundError` from `source.stream()`,
    and a probe handed a decompression bomb or a broken xref raises out of the parser instead of
    returning `ok: False`. Both used to escape the loop, hit the `BaseException` cleanup and
    `rmtree` the whole run — F3's own failure through a different exception type. In "fail" mode
    nothing is caught that was not caught before: those exceptions propagate exactly as they did.

    The `stream_upload` / `probe_pdf` parameters default to the module's own imports and exist
    so a caller can pass ITS module globals — which is what keeps
    `test_uploads_are_handled_one_file_at_a_time` honest: that test monkeypatches the names in
    `canopy.server.app`, and a helper that resolved its own would make the patch inert and the
    pin green for the wrong reason.
    """
    if not protocol_text.strip():
        raise HTTPException(status_code=422, detail="a run needs a protocol (YAML)")
    if not sources:
        raise HTTPException(status_code=422, detail="a run needs at least one PDF")
    if len(sources) > max_files:
        raise HTTPException(status_code=413,
                            detail=f"{len(sources)} files is over the {max_files} limit")
    if options.start and manager.key_required():
        raise HTTPException(status_code=400, detail="no ANTHROPIC_API_KEY is configured — "
                                                    "add one to .env and try again")
    if options.start and not manager.has_capacity():
        raise HTTPException(status_code=429,
                            detail=f"this server already has {manager.max_active} review(s) "
                                   f"running; wait for one to finish (or raise "
                                   f"CANOPY_MAX_ACTIVE_RUNS)")

    parsed = parse_protocol(protocol_text)
    if options.profile:
        # the picker's choice wins where it speaks and the protocol's own typed stats win
        # where they do: rebuild from the fields the YAML actually set, under the picked
        # profile, and resolve. (The guided form sends only `profile` in its stats block, so
        # for it this is the old wholesale replacement; a pasted protocol that also picked a
        # profile keeps its typed settings, which replacement silently discarded.)
        typed = {k: getattr(parsed.stats, k)
                 for k in parsed.stats.model_fields_set if k != "profile"}
        parsed.stats = apply_profile(StatsSettings(profile=options.profile, **typed))

    job = manager.create(title=parsed.title or options.name or "review",
                         options=options.model_dump())
    rejected: list[dict[str, str]] = [] if rejected_out is None else rejected_out
    try:
        if options.profile:
            dump_protocol(parsed, job.run_dir / "protocol.yaml")
        else:
            # the same rule as the CLI's copyfile branch: the run keeps the document the
            # person uploaded, verbatim. A re-dump manufactures explicitness — every field of
            # a full `model_dump` reads back as explicitly chosen, so `apply_profile` becomes
            # a no-op and the class defaults are frozen in as if somebody picked them; this
            # run-creation path is how a protocol asking for Hedges' g via `profile: metafor`
            # produced runs recorded as `estimator: cohen`. The verbatim file also preserves
            # the uploader's comments, which are the protocol's own audit trail.
            (job.run_dir / "protocol.yaml").write_text(protocol_text, encoding="utf-8")
        for relative, text in (copy_into or {}).items():
            # provenance the second door carries in (what a search did, so the run can be read
            # without it). Names are fixed by the caller and never come from a paper or an
            # index; the join is checked anyway, because a `..` here would write outside a run.
            target = (job.run_dir / relative).resolve()
            if not str(target).startswith(str(job.run_dir.resolve()) + "/"):
                raise ValueError(f"refusing to write {relative!r} outside the run directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        saved: dict[str, str] = {}
        remaining = float(max_total_bytes)
        skipping = on_source_rejected == "skip"
        for source in sources:
            name = safe_filename(source.name)
            try:
                path, size = stream_upload(source.stream(), job.run_dir / "uploads", name,
                                           max_bytes=max_upload_bytes,
                                           remaining_bytes=remaining)
            except UploadRejected as exc:
                if skipping:
                    rejected.append({"name": name, "reason": str(exc)})
                    continue
                raise HTTPException(status_code=exc.status_code, detail=str(exc))
            except OSError as exc:
                # skip mode only. `source.stream()` opens a file a machine staged hours ago; one
                # that has since been deleted raises FileNotFoundError here and never reaches
                # `UploadRejected`. In "fail" mode this re-raises untouched, so mode 1 sees the
                # same exception it always saw.
                if not skipping:
                    raise
                rejected.append({"name": name, "reason": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            remaining -= size
            first_time = str(path) not in saved      # `<sha256>.pdf`: the same paper twice
            saved.setdefault(str(path), name)
            if first_time:                           # probing a duplicate buys nothing, and a
                # folder of 79 files is often a dozen papers — each probe is a child process
                try:
                    probe = probe_pdf(path, timeout=probe_timeout)
                except Exception as exc:             # noqa: BLE001 - skip mode only, see below
                    # the probe parses bytes a publisher chose, so it RAISING is an ordinary
                    # outcome rather than a bug (`transport._store` reasons the same way about the
                    # same call). Turned into the refusal it would have returned; re-raised
                    # unchanged in "fail" mode, where the person who chose the file is waiting.
                    if not skipping:
                        raise
                    probe = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
                if not probe.get("ok"):
                    if on_source_rejected == "skip":
                        rejected.append({"name": name, "reason": str(probe.get("error") or
                                                                    "unreadable PDF")})
                        saved.pop(str(path), None)
                        path.unlink(missing_ok=True)
                        continue
                    raise HTTPException(status_code=400, detail=f"{name}: {probe['error']}")
        if not saved:
            # every source was skipped: a run with no papers is not a run, and the caller is
            # owed the reason rather than an empty directory that fails later
            raise HTTPException(status_code=422,
                                detail="none of these PDFs could be read: "
                                       + "; ".join(f"{r['name']} ({r['reason']})"
                                                   for r in rejected[:5]))
        (job.run_dir / "uploads" / "filenames.json").write_text(
            json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    except BaseException:
        shutil.rmtree(job.run_dir, ignore_errors=True)
        manager.forget(job.run_id)
        raise

    job.n_files = len(saved)
    job.save()
    return job
