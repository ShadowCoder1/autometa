"""The API the review UI talks to — and the whole of the server's attack surface.

Every endpoint under `/api/runs/{id}` needs that run's bearer token; everything else is either the
static SPA or information that says nothing about anybody's papers (which example protocols exist,
whether an API key is configured — never the key itself).

The shape of a session:

    POST /api/runs                      upload the folder + protocol, get {run_id, token}
    POST /api/runs/{id}/start           start a run created with `start: false` (after a dry run)
    GET  /api/runs/{id}/events          server-sent progress, replayable, ends with `end`
    GET  /api/runs/{id}                 the manifest and the run's state
    GET  /api/runs/{id}/results/{key}   pooled estimate, rows, sensitivity set, forest URL
    GET  /api/runs/{id}/evidence/{d}/{k}  candidates, verdicts, verifier text, images
    GET  /api/runs/{id}/files/{path}    the artefacts themselves, path-checked
    POST /api/runs/{id}/overrides       one reviewer decision, appended to overrides.jsonl
    POST /api/runs/{id}/repool          re-pool under those decisions, no model calls
    POST /api/runs/{id}/cancel          stop after the current step

Nothing here computes a statistic, and nothing here knows anything about any research field: every
label the UI shows comes from the user's own protocol.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urlsplit

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..config import MODELS, api_key, live_enabled, load_env
from ..models import Protocol, StatsSettings
from ..protocol import (apply_profile, available_profiles, dump_protocol, load_protocol,
                        load_yaml_strict)
from ..report import theme
from .jobs import Job, JobBusy, JobManager, TooManyRuns
from .overrides import (OverrideRejected, append_override, append_overrides,
                        apply_overrides_and_repool,
                        override_summary, read_overrides, repool_lock)
from .security import (PathRejected, is_attachment, is_loopback, media_type, safe_run_path,
                       token_matches)
from .uploads import (DEFAULT_MAX_FILES, DEFAULT_MAX_TOTAL_MB, DEFAULT_MAX_UPLOAD_MB,
                      DEFAULT_PROBE_TIMEOUT, UploadRejected, probe_pdf, safe_filename,
                      stream_upload)

__all__ = ["create_app", "serve"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "protocols"
MAX_CONCURRENCY = 16
#: a protocol is a page of YAML; anything larger is a mistake or an attack
MAX_PROTOCOL_BYTES = 1_000_000
#: methods that change something — a page on another origin may not use them
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


# ============================================================================ helpers
def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:                                     # pragma: no cover - defensive
        return default


def _protocol_of(job: Job) -> Protocol:
    path = job.run_dir / "protocol.yaml"
    if not path.exists():
        raise HTTPException(status_code=409, detail="this run has no protocol on disk")
    return load_protocol(path)


def _validation_message(error: ValidationError) -> str:
    parts = []
    for item in error.errors()[:8]:
        where = ".".join(str(x) for x in item.get("loc", ()) if x != "__root__")
        parts.append(f"{where or 'protocol'}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def _parse_protocol(text: str) -> Protocol:
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
                            detail=f"the protocol is incomplete — {_validation_message(exc)}")
    return protocol


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


def _options(raw: str) -> RunOptions:
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
        raise HTTPException(status_code=422, detail=f"bad options — {_validation_message(exc)}")


def _start(manager: JobManager, job: Job, *, dry_run: bool = False,
           max_papers: int = 3) -> None:
    """Hand a job to the worker, turning its two refusals into the HTTP answers they mean.

    The checks live inside the manager, under its lock: a capacity check made here and acted on
    there is a race, and two clicks on Run would win it.
    """
    try:
        if dry_run:
            manager.start_dry_run(job, max_papers=max_papers)
        else:
            manager.start(job)
    except JobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except TooManyRuns as exc:
        raise HTTPException(status_code=429, detail=str(exc))


def is_cross_site(request: Request) -> bool:
    """True when a browser says this request came from another site.

    `POST /api/runs` cannot require a token — it is where the token comes from — and a multipart
    POST is a "simple request", so no CORS preflight protects it: a page the user happens to be
    visiting could otherwise start a run on their machine, with their key and their money. Every
    browser sends `Sec-Fetch-Site`, and a cross-origin `fetch` also sends `Origin`; a non-browser
    client (curl, a script, a test) sends neither and is unaffected. Reading a response cross-origin
    is already impossible: no CORS header is ever sent.
    """
    site = (request.headers.get("sec-fetch-site") or "").lower()
    if site and site not in ("same-origin", "none"):
        return True
    origin = request.headers.get("origin")
    if origin:
        try:
            netloc = urlsplit(origin).netloc.lower()
        except ValueError:                                 # pragma: no cover - malformed header
            return True
        if netloc != (request.headers.get("host") or "").lower():
            return True
    return False


def _study_label(citation: Mapping[str, Any] | None, fallback: str = "") -> str:
    """`Bock 2005` — how a person names a paper, with the file name as the fallback.

    A sha256 is how Canopy identifies a paper; it is not how anybody reads one. Every list the UI
    shows joins the citation the mapper read back onto the id, and keeps the id as the second line.
    """
    citation = citation or {}
    name = str(citation.get("first_author") or "").strip()
    if not name:
        authors = str(citation.get("authors") or "").strip()
        name = authors.split(",")[0].strip() if authors else ""
    year = citation.get("year")
    if name and year:
        return f"{name} {year}"
    if name:
        return name
    stem = Path(str(fallback or "")).stem
    return stem or "this paper"


def _pages(*values: Any) -> str:
    """`p. 3` / `pp. 3, 5` — where in the paper the numbers were read."""
    seen: list[str] = []
    for value in values:
        text = str(value if value is not None else "").strip()
        if text and text not in seen and text.lower() not in ("none", "nan", "0"):
            seen.append(text)
    if not seen:
        return ""
    return ("p. " if len(seen) == 1 else "pp. ") + ", ".join(seen)


def _paper_rows(job: Job, manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """One row per paper: what the run did with it, under the name its authors would use.

    The status, the stages and the cost come from the manifest, so a monitor that opens after the
    run finished paints the same grid the live events would have drawn.
    """
    studies = _run_index(job)["studies"]
    rows: list[dict[str, Any]] = []
    for status in (manifest or {}).get("papers") or []:
        paper_id = str(status.get("paper_id") or "")
        citation = (studies.get(paper_id) or {}).get("citation") or {}
        filename = str(status.get("filename") or "")
        rows.append({
            "paper_id": paper_id, "sha12": paper_id[:12], "filename": filename,
            "study_label": _study_label(citation, filename),
            "first_author": citation.get("first_author") or "", "year": citation.get("year"),
            "title": citation.get("title") or "",
            "status": status.get("status", ""), "stages": status.get("stages") or {},
            "cost_usd": status.get("cost_usd", 0.0), "seconds": status.get("seconds", 0.0),
            "eligible": status.get("eligible"), "error": status.get("error", ""),
            "n_datasets": len((studies.get(paper_id) or {}).get("datasets") or []),
        })
    return rows


def _dataset_labels(job: Job) -> dict[str, dict[str, Any]]:
    """dataset_id → what to call it: the paper, the dataset's own label, the group names."""
    index = _run_index(job)
    out: dict[str, dict[str, Any]] = {}
    for paper_id, study in index["studies"].items():
        citation = study.get("citation") or {}
        for dataset in study.get("datasets") or []:
            dataset_id = str(dataset.get("dataset_id") or "")
            out[dataset_id] = {
                "paper_id": paper_id,
                "study_label": _study_label(citation, ""),
                "first_author": citation.get("first_author") or "",
                "year": citation.get("year"),
                "dataset_label": str(dataset.get("label") or dataset.get("experiment") or ""),
                "condition": str(dataset.get("condition") or ""),
                "group_a": (dataset.get("group_a") or {}).get("label", ""),
                "group_b": (dataset.get("group_b") or {}).get("label", ""),
            }
    return out


def _run_index(job: Job) -> dict[str, Any]:
    """dataset_id → paper_id, and the study maps, read from the run's own stage files."""
    datasets: dict[str, str] = {}
    studies: dict[str, dict[str, Any]] = {}
    for path in sorted(job.run_dir.glob("papers/*/map.json")):
        payload = _read_json(path, {}) or {}
        study = payload.get("study") or {}
        paper_id = str(study.get("paper_id") or path.parent.name)
        studies[paper_id] = study
        for dataset in study.get("datasets") or []:
            datasets[str(dataset.get("dataset_id"))] = paper_id
    return {"dataset_to_paper": datasets, "studies": studies}


def _stage(job: Job, paper_id: str, stage: str) -> dict[str, Any]:
    return _read_json(job.run_dir / "papers" / str(paper_id)[:12] / f"{stage}.json", {}) or {}


def _file_url(job: Job, path: str | Path) -> str:
    """A run-relative artefact path as an API URL, or "" when it is not inside the run.

    The name is percent-encoded: a candidate id contains `#` (`d1:outcome:A:extractor#0`), and an
    un-encoded `#` in a URL is a fragment, so the server would never see the file name at all.
    """
    try:
        relative = Path(path).resolve().relative_to(job.run_dir.resolve())
    except (ValueError, OSError):
        return ""
    return f"/api/runs/{job.run_id}/files/{quote(relative.as_posix(), safe='/')}"


# ============================================================================ the app
def create_app(runs_dir: str | Path = "runs", *,
               client_factory: Callable[..., Any] | None = None,
               allowed_hosts: Sequence[str] | None = None,
               max_upload_mb: float | None = None,
               probe_timeout: float | None = None,
               max_active_runs: int | None = None,
               loopback_only: bool = True) -> FastAPI:
    """Build the API. `client_factory` is the seam a test fills with a fake or replaying client."""
    app = FastAPI(title="Canopy", docs_url=None, redoc_url=None, openapi_url=None)
    manager = JobManager(runs_dir, client_factory=client_factory,
                         **({} if max_active_runs is None else {"max_active": max_active_runs}))
    app.state.runs_dir = str(manager.runs_dir)
    app.state.jobs = manager
    app.state.max_upload_bytes = (max_upload_mb if max_upload_mb is not None
                                  else DEFAULT_MAX_UPLOAD_MB) * 1e6
    app.state.probe_timeout = (probe_timeout if probe_timeout is not None
                               else DEFAULT_PROBE_TIMEOUT)
    app.state.max_total_bytes = max(app.state.max_upload_bytes, DEFAULT_MAX_TOTAL_MB * 1e6)
    app.state.allowed_hosts = [h.lower() for h in (allowed_hosts or [])]
    app.state.loopback_only = bool(loopback_only)

    # ------------------------------------------------------------------ middleware
    @app.middleware("http")
    async def guard(request: Request, call_next):          # noqa: ANN001 - starlette signature
        """Refuse a foreign Host header (DNS rebinding) and a cross-site write (drive-by runs)."""
        allowed = request.app.state.allowed_hosts
        if allowed:
            host = (request.headers.get("host") or "").split(":")[0].strip("[]").lower()
            if host not in allowed:
                return JSONResponse({"detail": f"unexpected Host header {host!r}"},
                                    status_code=400)
        if request.method in UNSAFE_METHODS and is_cross_site(request):
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    def run_of(run_id: str, request: Request) -> Job:
        """The job this request may touch, or 404 (unknown) / 401 (no or wrong token)."""
        job = manager.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such run")
        header = request.headers.get("authorization") or ""
        given = header[7:].strip() if header.lower().startswith("bearer ") else ""
        # EventSource and <img> cannot set headers, so a token in the query is accepted too;
        # it never leaves this machine and the run directory is readable anyway.
        given = given or (request.query_params.get("token") or "")
        if not token_matches(given, job.token):
            raise HTTPException(status_code=401, detail="this run needs its own token")
        return job

    # ------------------------------------------------------------------ static SPA
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():                              # pragma: no cover - build sanity
            raise HTTPException(status_code=500, detail="the UI files are missing")
        return FileResponse(page, media_type="text/html; charset=utf-8")

    # ------------------------------------------------------------------ settings
    @app.get("/api/settings")
    def settings() -> dict[str, Any]:
        """What the UI needs to know — and, of the API key, only whether there is one."""
        return {
            "api_key_configured": bool(api_key()),
            "live": bool(live_enabled()),
            "models": dict(MODELS),
            "profiles": available_profiles(),
            "max_upload_mb": round(app.state.max_upload_bytes / 1e6, 3),
            "max_files": DEFAULT_MAX_FILES,
            "max_total_mb": round(app.state.max_total_bytes / 1e6, 1),
            "loopback_only": app.state.loopback_only,
            "max_active_runs": manager.max_active,
            "uses_real_models": manager.uses_real_models,
        }

    # ------------------------------------------------------------------ protocols
    @app.get("/api/protocols/examples")
    def protocol_examples() -> dict[str, Any]:
        from ..cli import SKELETON

        examples = []
        for path in sorted(EXAMPLES_DIR.glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            try:
                title = str((yaml.safe_load(text) or {}).get("title") or path.stem)
            except yaml.YAMLError:                         # pragma: no cover - shipped examples
                title = path.stem
            examples.append({"name": path.stem, "title": title, "yaml": text})
        return {"examples": examples, "profiles": available_profiles(),
                "skeleton": SKELETON.format(name="my_review", title="My review",
                                            profiles=available_profiles())}

    @app.post("/api/protocols/draft")
    def draft_protocol(body: dict[str, Any]) -> dict[str, Any]:
        """One sentence in, a protocol skeleton out (amendment I). The user edits it after."""
        from .protocol_draft import draft_from_sentence

        sentence = str(body.get("sentence") or "").strip()
        if not sentence:
            raise HTTPException(status_code=422, detail="say what your review is about, in one "
                                                        "sentence")
        if manager.key_required():
            raise HTTPException(status_code=400, detail="drafting a protocol needs a model: set "
                                                        "ANTHROPIC_API_KEY in .env")
        client = manager.client_factory(run_dir=None, purpose="draft")
        try:
            drafted = draft_from_sentence(client, sentence[:2000],
                                          model=MODELS["primary"])
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"the draft failed: "
                                                        f"{type(exc).__name__}: {exc}")
        return drafted

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        """Every run on disk. The token comes back only on a loopback-only server."""
        rows = []
        for job in manager.list():
            row: dict[str, Any] = {"run_id": job.run_id, "title": job.title,
                                   "created_at": job.created_at, "status": job.status,
                                   "cost_usd": round(job.cost_usd, 4), "n_files": job.n_files,
                                   "kind": job.kind}
            if app.state.loopback_only:
                row["token"] = job.token
            rows.append(row)
        return {"runs": rows, "loopback_only": app.state.loopback_only}

    @app.post("/api/runs", status_code=201)
    def create_run(files: list[UploadFile] = File(default_factory=list),
                   protocol: UploadFile | None = File(default=None),
                   protocol_text: str = Form(default=""),
                   options: str = Form(default="{}")) -> dict[str, Any]:
        """A folder of PDFs and a protocol become a run directory and a background job.

        Each file is streamed from the wire to `<sha256>.pdf` and ingested before the next one is
        read, so the server holds one chunk of one upload at a time however large the folder is.
        """
        chosen = _options(options)
        text = protocol_text[:MAX_PROTOCOL_BYTES + 1]
        if not text and protocol is not None:
            text = protocol.file.read(MAX_PROTOCOL_BYTES + 1).decode("utf-8", "replace")
        if len(text.encode("utf-8", "replace")) > MAX_PROTOCOL_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"a protocol may not be larger than "
                                       f"{MAX_PROTOCOL_BYTES / 1e6:.0f} MB")
        if not text.strip():
            raise HTTPException(status_code=422, detail="a run needs a protocol (YAML)")
        if not files:
            raise HTTPException(status_code=422, detail="a run needs at least one PDF")
        if len(files) > DEFAULT_MAX_FILES:
            raise HTTPException(status_code=413,
                                detail=f"{len(files)} files is over the {DEFAULT_MAX_FILES} limit")
        if chosen.start and manager.key_required():
            raise HTTPException(status_code=400, detail="no ANTHROPIC_API_KEY is configured — "
                                                        "add one to .env and try again")
        if chosen.start and not manager.has_capacity():
            raise HTTPException(status_code=429,
                                detail=f"this server already has {manager.max_active} review(s) "
                                       f"running; wait for one to finish (or raise "
                                       f"CANOPY_MAX_ACTIVE_RUNS)")

        parsed = _parse_protocol(text)
        if chosen.profile:
            # the same rule as `canopy run --profile`: an explicit choice replaces the
            # protocol's own statistics block, and the run directory keeps what it used
            parsed.stats = apply_profile(StatsSettings(profile=chosen.profile))

        job = manager.create(title=parsed.title or chosen.name or "review",
                             options=chosen.model_dump())
        try:
            dump_protocol(parsed, job.run_dir / "protocol.yaml")
            saved: dict[str, str] = {}
            remaining = float(app.state.max_total_bytes)
            for upload in files:
                name = safe_filename(upload.filename or "upload.pdf")
                try:
                    path, size = stream_upload(upload.file, job.run_dir / "uploads", name,
                                               max_bytes=app.state.max_upload_bytes,
                                               remaining_bytes=remaining)
                except UploadRejected as exc:
                    raise HTTPException(status_code=exc.status_code, detail=str(exc))
                remaining -= size
                first_time = str(path) not in saved      # `<sha256>.pdf`: the same paper twice
                saved.setdefault(str(path), name)
                if first_time:                           # probing a duplicate buys nothing, and a
                    # folder of 79 files is often a dozen papers — each probe is a child process
                    probe = probe_pdf(path, timeout=app.state.probe_timeout)
                    if not probe.get("ok"):
                        raise HTTPException(status_code=400, detail=f"{name}: {probe['error']}")
            (job.run_dir / "uploads" / "filenames.json").write_text(
                json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        except BaseException:
            shutil.rmtree(job.run_dir, ignore_errors=True)
            manager.forget(job.run_id)
            raise

        job.n_files = len(saved)
        job.save()
        if chosen.start:
            _start(manager, job)
        return {"run_id": job.run_id, "token": job.token, "n_files": job.n_files,
                "status": job.status, "title": job.title}

    @app.get("/api/runs/{run_id}")
    def run_state(run_id: str, request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        manifest = _read_json(job.run_dir / "manifest.json")
        protocol = _protocol_of(job) if (job.run_dir / "protocol.yaml").exists() else None
        outcomes = []
        for outcome in (protocol.outcomes if protocol else []):
            pooled = _read_json(job.run_dir / "results" / outcome.key / "pooled.json", {}) or {}
            outcomes.append({"key": outcome.key, "label": outcome.label,
                             "k": pooled.get("k", 0), "estimate": pooled.get("estimate"),
                             "ci_low": pooled.get("ci_low"), "ci_high": pooled.get("ci_high"),
                             "n_needs_human": pooled.get("n_needs_human", 0),
                             "has_results": bool(pooled)})
        return {
            "run_id": job.run_id, "status": job.status, "kind": job.kind, "title": job.title,
            "created_at": job.created_at, "started_at": job.started_at,
            "finished_at": job.finished_at, "error": job.error,
            "cost_usd": round(job.cost_usd, 6), "n_files": job.n_files, "options": job.options,
            "manifest": manifest, "outcomes": outcomes,
            "papers": _paper_rows(job, manifest),
            "protocol": None if protocol is None else {
                "title": protocol.title, "research_question": protocol.research_question,
                "group_a": protocol.group_a.model_dump(mode="json"),
                "group_b": protocol.group_b.model_dump(mode="json"),
                "outcomes": [o.model_dump(mode="json") for o in protocol.outcomes],
                "moderators": list(protocol.moderators),
                "eligibility": list(protocol.eligibility),
                "stats": protocol.stats.model_dump(mode="json")},
            "n_overrides": len(read_overrides(job.run_dir)),
            "outputs": (manifest or {}).get("outputs", {}),
        }

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, request: Request) -> StreamingResponse:
        job = run_of(run_id, request)
        return StreamingResponse(
            manager.stream(job), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"})

    @app.post("/api/runs/{run_id}/start")
    def start_run(run_id: str, request: Request) -> dict[str, Any]:
        """Start a run that was created but held back — the dry-run-first path."""
        job = run_of(run_id, request)
        if manager.key_required():
            raise HTTPException(status_code=400, detail="no ANTHROPIC_API_KEY is configured — "
                                                        "add one to .env and try again")
        job.kind = "run"
        _start(manager, job)
        return {"run_id": job.run_id, "status": job.status}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        return {"run_id": job.run_id, "status": manager.cancel(job)}

    @app.get("/api/runs/{run_id}/results/{outcome_key}")
    def run_results(run_id: str, outcome_key: str, request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        protocol = _protocol_of(job)
        outcome = next((o for o in protocol.outcomes if o.key == outcome_key), None)
        if outcome is None:
            raise HTTPException(status_code=404, detail=f"no outcome {outcome_key!r} in this "
                                                        f"protocol")
        directory = job.run_dir / "results" / outcome_key
        pooled = _read_json(directory / "pooled.json", {}) or {}
        held = set(pooled.get("needs_human_dataset_ids") or [])
        names = _dataset_labels(job)
        papers = {p["paper_id"]: p for p in _paper_rows(job, _read_json(
            job.run_dir / "manifest.json", {}) or {})}
        rows = []
        for row in _read_json(directory / "extraction_table.json", []) or []:
            flags = row.get("flags") or []
            named = names.get(str(row.get("dataset_id") or ""), {})
            paper = papers.get(str(row.get("paper_id") or ""), {})
            rows.append({**row, "in_primary": row.get("dataset_id") not in held,
                         "overridden": "human_override" in flags,
                         # DECISION A: which line each row is in, and on whose authority. Named
                         # here even when the table predates them, so the page can ask the
                         # question of any run instead of only of a freshly pooled one.
                         "in_best_guess": row.get("in_best_guess"),
                         "best_guess_rule": row.get("best_guess_rule") or "",
                         "best_guess_reason": row.get("best_guess_reason") or "",
                         "study_label": (named.get("study_label")
                                         or _study_label(row, paper.get("filename", ""))),
                         "dataset_label": named.get("dataset_label") or row.get("label") or "",
                         "paper_filename": paper.get("filename", ""),
                         "pages": _pages(row.get("page_a"), row.get("page_b"))})
        summary = override_summary(job.run_dir)
        return {
            "outcome": outcome.model_dump(mode="json"),
            "pooled": pooled, "rows": rows,
            "sensitivity": _read_json(directory / "sensitivity.json", {}) or {},
            "leave_one_out": _read_json(directory / "leave_one_out.json", []) or [],
            "funnel": _read_json(directory / "funnel.json", {}) or {},
            "forest": {name: _file_url(job, directory / f"forest.{name}")
                       for name in ("svg", "png", "pdf")
                       if (directory / f"forest.{name}").exists()},
            # DECISION A/B/F: the second line, the paragraph and who drew the plots — all read
            # out of `pooled.json` rather than derived here, so the page, the report and the file
            # cannot end up saying three different things about one run.
            "best_guess": pooled.get("best_guess") or {},
            "forest_best_guess": {
                name: _file_url(job, directory / f"forest_best_guess.{name}")
                for name in ("svg", "png", "pdf")
                if (directory / f"forest_best_guess.{name}").exists()},
            "best_guess_caveat": theme.BEST_GUESS_CAVEAT,
            "conclusion": pooled.get("conclusion") or {},
            "renderer": pooled.get("renderer") or {},
            "renderer_best_guess": pooled.get("renderer_best_guess") or {},
            "figures": {name: _file_url(job, directory / f"{name}.png")
                        for name in ("sensitivity", "funnel")
                        if (directory / f"{name}.png").exists()},
            "review": [e for e in _review_queue(job) if e.get("outcome_key") == outcome_key],
            "papers": list(papers.values()),
            "excluded": [e for e in summary.get("excluded", [])
                         if e.get("outcome_key") in ("", outcome_key)],
            "settings": protocol.stats.model_dump(mode="json"),
        }

    def _review_queue(job: Job) -> list[dict[str, Any]]:
        from ..pipeline.state import sort_review_queue

        queue = _read_json(job.run_dir / "human_review_queue.json", None)
        if queue is None:
            manifest = _read_json(job.run_dir / "manifest.json", {}) or {}
            queue = manifest.get("human_review_queue") or []
        names = _dataset_labels(job)
        outcomes = {o.key: o.label for o in _protocol_of(job).outcomes} \
            if (job.run_dir / "protocol.yaml").exists() else {}
        labelled = []
        for entry in queue:
            if not isinstance(entry, dict):
                continue
            named = names.get(str(entry.get("dataset_id") or ""), {})
            pages = _pages(*[c.get("page") for c in entry.get("candidates") or []])
            labelled.append({**entry, "study_label": named.get("study_label", ""),
                             "dataset_label": named.get("dataset_label", ""),
                             "first_author": named.get("first_author", ""),
                             "year": named.get("year"), "pages": pages,
                             "outcome_label": outcomes.get(str(entry.get("outcome_key")), "")})
        return sort_review_queue(labelled)

    @app.get("/api/runs/{run_id}/review")
    def run_review(run_id: str, request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        return {"queue": _review_queue(job), "overrides": read_overrides(job.run_dir),
                "summary": override_summary(job.run_dir)}

    @app.get("/api/runs/{run_id}/evidence/{dataset_id}/{outcome_key}")
    def run_evidence(run_id: str, dataset_id: str, outcome_key: str,
                     request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        index = _run_index(job)
        paper_id = index["dataset_to_paper"].get(dataset_id, "")
        if not paper_id:
            raise HTTPException(status_code=404, detail=f"no dataset {dataset_id!r} in this run")

        extract = _stage(job, paper_id, "extract")
        verify = _stage(job, paper_id, "verify")
        candidates = [*(extract.get("candidates") or []),
                      *(verify.get("extra_candidates") or [])]
        mine = [c for c in candidates
                if c.get("dataset_id") == dataset_id and c.get("outcome_key") == outcome_key]
        verdicts = [v for v in (verify.get("verdicts") or [])
                    if v.get("dataset_id") == dataset_id and v.get("outcome_key") == outcome_key]
        record = next((r for r in _read_json(
            job.run_dir / "results" / outcome_key / "extraction_table.json", []) or []
            if r.get("dataset_id") == dataset_id), {})
        study = index["studies"].get(paper_id, {})
        dataset = next((d for d in study.get("datasets") or []
                        if d.get("dataset_id") == dataset_id), {})
        return {
            "run_id": job.run_id, "dataset_id": dataset_id, "outcome_key": outcome_key,
            "paper_id": paper_id, "citation": study.get("citation", {}),
            "study_label": _study_label(study.get("citation") or {}, ""),
            "dataset_label": str(dataset.get("label") or dataset.get("experiment") or ""),
            "dataset": dataset, "record": record,
            "conversion_chain": record.get("conversion_chain", ""),
            "conversion_steps": _conversion_steps(job, paper_id, dataset_id, outcome_key),
            "routes_available": record.get("routes_available", []),
            "routes_rejected": record.get("routes_rejected", {}),
            "candidates": [_candidate_view(c) for c in mine],
            "verdicts": [_verdict_view(v) for v in verdicts],
            "orientation": (verify.get("orientation") or {}).get(
                f"{outcome_key}|{(dataset.get('outcomes') or [{}])[0].get('measure_name', '')}",
                {}),
            "images": _images(job, paper_id, dataset_id, outcome_key, mine),
            "overrides": [o for o in read_overrides(job.run_dir)
                          if o.get("dataset_id") == dataset_id
                          and o.get("outcome_key") in ("", outcome_key)],
        }

    def _conversion_steps(job: Job, paper_id: str, dataset_id: str,
                          outcome_key: str) -> list[str]:
        for row in _stage(job, paper_id, "resolve").get("records") or []:
            if row.get("dataset_id") == dataset_id and row.get("outcome_key") == outcome_key:
                return list(row.get("conversion_steps") or [])
        return []

    def _images(job: Job, paper_id: str, dataset_id: str, outcome_key: str,
                candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """The picture behind each value: a page crop with the quote, or the digitiser's overlay."""
        bundle = _read_json(job.run_dir / "provenance" / "provenance.json", {}) or {}
        ingest_dir = job.run_dir / "papers" / str(paper_id)[:12] / "ingest"
        out: list[dict[str, Any]] = []
        for candidate in candidates:
            entry = bundle.get(candidate.get("candidate_id") or "", {})
            crop = entry.get("crop") or ""
            if not crop:
                relative = candidate.get("overlay_path") or candidate.get("crop_path") or ""
                crop = str(ingest_dir / relative) if relative else ""
            url = _file_url(job, crop) if crop and Path(crop).exists() else ""
            if not url:
                continue
            out.append({"candidate_id": candidate.get("candidate_id", ""), "url": url,
                        "group": candidate.get("group"), "page": candidate.get("page"),
                        "quote": candidate.get("quote", ""), "route": candidate.get("route", ""),
                        "matched": bool(entry.get("matched", False)),
                        "note": entry.get("note", "")})
        return out

    @app.get("/api/runs/{run_id}/files/{path:path}")
    def run_file(run_id: str, path: str, request: Request) -> FileResponse:
        job = run_of(run_id, request)
        try:
            target = safe_run_path(job.run_dir, path)
        except PathRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not target.is_file():
            raise HTTPException(status_code=404, detail="no such artefact")
        headers = {"Cache-Control": "no-store",
                   "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"}
        if is_attachment(target):
            headers["Content-Disposition"] = f'attachment; filename="{target.name}"'
        return FileResponse(target, media_type=media_type(target), headers=headers)

    # ------------------------------------------------------------------ review workflow
    @app.post("/api/runs/{run_id}/overrides", status_code=201)
    def post_override(run_id: str, request: Request, body: dict[str, Any]) -> dict[str, Any]:
        job = run_of(run_id, request)
        try:
            record = append_override(job.run_dir, body)
        except OverrideRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, "override": record, "n_overrides": len(read_overrides(job.run_dir))}

    @app.get("/api/runs/{run_id}/overrides")
    def get_overrides(run_id: str, request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        return {"overrides": read_overrides(job.run_dir),
                "summary": override_summary(job.run_dir)}

    # ------------------------------------------------------------------ questions
    @app.get("/api/runs/{run_id}/questions")
    def run_questions(run_id: str, request: Request) -> dict[str, Any]:
        """Every held cell as a question with its screenshot, answers and reason."""
        job = run_of(run_id, request)
        if not (job.run_dir / "manifest.json").exists():
            raise HTTPException(status_code=409, detail="this run has not finished yet")
        from ..review.questions import questions_for_run
        questions = questions_for_run(job.run_dir)
        for q in questions:
            image = q.get("image") or {}
            if image.get("path"):
                image["url"] = _file_url(job, str(job.run_dir / image["path"]))
        return {"run_id": job.run_id, "questions": questions,
                # three states, three counts: "not answered" is not the same as "not applied", and
                # a run whose every remaining answer is waiting for a re-run used to report "no
                # open questions" while nothing had been applied.
                "n_open": sum(1 for q in questions if q.get("status") == "open"),
                "n_pending": sum(1 for q in questions
                                 if q.get("status") == "pending_rerun")}

    @app.post("/api/runs/{run_id}/questions/{number}/answer", status_code=201)
    def answer_question(run_id: str, number: int, request: Request,
                        body: dict[str, Any]) -> dict[str, Any]:
        """An answer becomes an override with a justification naming the question, and the run
        is re-pooled so the forest plot reflects it.

        The body MUST carry the question's `id` (from `GET /questions`). Without it: **422**;
        with one that is no longer the question at this number: **409** — the list is rebuilt on
        every request and shortens as answers land, so a number alone would decide a different
        cell. An answer that picks an `option` must also echo that option's `fingerprint`: **422**
        without it, **409** when it no longer matches. Clients written against the earlier
        contract have to send both.

        **201** carries `override` (the first record, as before) and `overrides` (all of them):
        a card that settles both groups of a dataset writes one override per group.
        """
        job = run_of(run_id, request)
        from ..review.questions import answers_to_overrides, questions_for_run
        question = next((q for q in questions_for_run(job.run_dir) if q["number"] == number), None)
        if question is None:
            raise HTTPException(status_code=404, detail=f"no question #{number} in this run")
        # A number is a position in a list that is rebuilt on every request, and answering a
        # question removes it — so #2 means a different cell a moment later, and a retried POST
        # decides a cell nobody was shown. The answer therefore names the question it answers.
        wanted = str((body or {}).get("id") or "")
        if not wanted:
            raise HTTPException(status_code=422,
                                detail="an answer must name the question it answers: send the "
                                       "question's `id` with it (numbers move as answers land)")
        if wanted != question["id"]:
            raise HTTPException(status_code=409,
                                detail=f"question #{number} is now {question['id']!r}, not "
                                       f"{wanted!r} — the list moved; reload the questions")
        # the id names the question; it cannot name the OPTION. `v1…vN` are positional and the
        # list is rebuilt from the cell's current numbers, so after any value answer the same key
        # under the same id can denote a different number. An answer therefore echoes what it was
        # shown, and a stale one is refused rather than applied to a number nobody chose.
        chosen = str((body or {}).get("option") or "")
        if chosen:
            option = next((o for o in question.get("options") or []
                           if o.get("key") == chosen), None)
            if option is None:
                raise HTTPException(status_code=409,
                                    detail=f"question #{number} no longer offers {chosen!r}")
            echoed = str((body or {}).get("option_fingerprint") or "")
            if not echoed:
                raise HTTPException(status_code=422,
                                    detail="an answer that picks an option must echo its "
                                           "`fingerprint`: the options are rebuilt on every "
                                           "request, so a key alone can mean a different number")
            if echoed != option.get("fingerprint"):
                raise HTTPException(status_code=409,
                                    detail=f"{chosen!r} on question #{number} is not the option "
                                           f"you were shown ({option.get('label')!r}); reload "
                                           f"the questions")
        # one answer, one override per CELL it names (DECISION §C1). A card can settle both
        # groups of a dataset at once, and the record of it is still one record per group — each
        # naming its own group and carrying only its own option's `clears`. They are appended
        # together and the run is re-pooled ONCE, so a reviewer never sees the analysis in the
        # half-answered state between two records of one decision.
        # …and they are validated TOGETHER before any of them is written: a refusal on the
        # second record used to leave the first in the log, half a decision with no re-pool
        # behind it and nothing in the response to say so (review MINOR 29).
        try:
            records = append_overrides(job.run_dir,
                                       answers_to_overrides(question, body or {}))
        except OverrideRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        with repool_lock(job.run_dir):
            try:
                summary = apply_overrides_and_repool(job.run_dir)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=409, detail=str(exc))
        after = questions_for_run(job.run_dir)
        return {"ok": True, "override": records[0], "overrides": records, "repool": summary,
                "n_open": sum(1 for q in after if q.get("status") == "open"),
                "n_pending": sum(1 for q in after if q.get("status") == "pending_rerun")}

    @app.post("/api/runs/{run_id}/repool")
    def repool(run_id: str, request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        if not (job.run_dir / "manifest.json").exists():
            raise HTTPException(status_code=409, detail="this run has not finished yet")
        with repool_lock(job.run_dir):                     # two reviewers, one set of artefacts
            try:
                return apply_overrides_and_repool(job.run_dir)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=409, detail=str(exc))

    # ------------------------------------------------------------------ dry run
    @app.post("/api/runs/{run_id}/dry-run")
    def dry_run(run_id: str, request: Request,
                body: dict[str, Any] | None = None) -> dict[str, Any]:
        job = run_of(run_id, request)
        body = body or {}
        if manager.key_required():
            raise HTTPException(status_code=400, detail="a dry run needs a model: set "
                                                        "ANTHROPIC_API_KEY in .env")
        _start(manager, job, max_papers=int(body.get("max_papers") or 3), dry_run=True)
        wait = min(float(body.get("wait_seconds") or 0.0), 600.0)
        if wait > 0 and job.thread is not None:
            job.thread.join(timeout=wait)
        result = manager.dry_run_result(job)
        return {"run_id": job.run_id, "status": result.get("status", job.status),
                "maps": result.get("maps", []), "cost_usd": result.get("cost_usd", 0.0),
                "error": result.get("error", "")}

    @app.get("/api/runs/{run_id}/dry-run")
    def dry_run_result(run_id: str, request: Request) -> dict[str, Any]:
        job = run_of(run_id, request)
        result = manager.dry_run_result(job)
        return {"run_id": job.run_id, "status": result.get("status", job.status),
                "maps": result.get("maps", []), "cost_usd": result.get("cost_usd", 0.0),
                "error": result.get("error", "")}

    @app.get("/api/health", include_in_schema=False)
    def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    return app


# ----------------------------------------------------------------------------- candidate views
def _candidate_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """What the evidence drawer shows for one reading. Every string is escaped by the SPA."""
    keys = ("candidate_id", "group", "kind", "status", "source_kind", "n", "mean",
            "dispersion_value", "dispersion_type", "ci_low", "ci_high", "unit",
            "value_as_written", "raw_value_semantics", "analysis_metric", "error_bar_scope",
            "stat_type", "stat_value", "df", "df1", "df2", "p_value", "p_kind", "design",
            "reported_value", "reported_scale", "standardizer", "page", "page_corrected",
            "quote", "locator", "row_header", "col_header", "sigma", "grounded",
            "grounding_similarity", "route", "model", "prompt_version", "notes")
    return {key: candidate.get(key) for key in keys}


def _verdict_view(verdict: Mapping[str, Any]) -> dict[str, Any]:
    verifiers = verdict.get("verifiers") or []
    last = verifiers[-1] if verifiers else {}
    return {
        "group": verdict.get("group"), "confidence": verdict.get("confidence"),
        "confidence_score": verdict.get("confidence_score"),
        "confidence_reasons": verdict.get("confidence_reasons") or [],
        "agreement": verdict.get("agreement"), "route": verdict.get("route"),
        "n": verdict.get("n"), "mean": verdict.get("mean"),
        "dispersion_value": verdict.get("dispersion_value"),
        "dispersion_type": verdict.get("dispersion_type"), "unit": verdict.get("unit"),
        "needs_human": verdict.get("needs_human"),
        "overridden_by_human": verdict.get("overridden_by_human"),
        "override_justification": verdict.get("override_justification", ""),
        "flags": verdict.get("flags") or [],
        "adjudicated": verdict.get("adjudicated"),
        "adjudication_rationale": verdict.get("adjudication_rationale", ""),
        "higher_is_better": verdict.get("higher_is_better"),
        "orientation_evidence": verdict.get("orientation_evidence", ""),
        "candidate_ids": verdict.get("candidate_ids") or [],
        "verifier": {"verdict": verdict.get("verifier_verdict") or "not_run",
                     "reason": verdict.get("verifier_reason", ""),
                     "model": last.get("model", ""),
                     "checked": last.get("checked", []),
                     "better_source": last.get("better_source", ""),
                     "alt_mean": last.get("alt_mean"),
                     "alt_quote": last.get("alt_quote", ""),
                     "reopens": verdict.get("reopens", 0)},
    }


# ----------------------------------------------------------------------------- serve
def serve(host: str = "127.0.0.1", port: int = 8000, runs_dir: str | Path = "runs") -> None:
    """Run the UI with uvicorn. Loopback by default; anything else is said out loud."""
    import uvicorn

    load_env()                                             # once, at startup; never printed
    loopback = is_loopback(host)
    if not loopback:
        print(f"\n  WARNING: binding {host} exposes your runs — and the PDFs, quotes and API\n"
              f"  budget behind them — to every machine that can reach this port.\n"
              f"  Canopy has no user accounts: the only secret is a per-run token.\n"
              f"  Use 127.0.0.1 unless you have a reason not to.\n")
    app = create_app(runs_dir, loopback_only=loopback,
                     allowed_hosts=["localhost", "127.0.0.1", "::1"] if loopback else None)
    print(f"canopy UI on http://{host}:{port}  ·  runs in {Path(runs_dir).resolve()}  ·  "
          f"API key {'configured' if api_key() else 'NOT configured'}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
