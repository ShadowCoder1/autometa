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
from pydantic import ValidationError

from ..config import MODELS, api_key, live_enabled, load_env
from ..models import Protocol, StatsSettings
from ..protocol import apply_profile, available_profiles, dump_protocol, load_protocol
from .jobs import Job, JobManager
from .overrides import (OverrideRejected, append_override, apply_overrides_and_repool,
                        override_summary, read_overrides, repool_lock)
from .security import (PathRejected, is_attachment, is_loopback, media_type, safe_run_path,
                       token_matches)
from .uploads import (DEFAULT_MAX_FILES, DEFAULT_MAX_TOTAL_MB, DEFAULT_MAX_UPLOAD_MB,
                      DEFAULT_PROBE_TIMEOUT, UploadRejected, probe_many, safe_filename,
                      save_upload, validate_pdf)

__all__ = ["create_app", "serve"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "protocols"
MAX_CONCURRENCY = 16
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
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"the protocol is not valid YAML: {exc}")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="the protocol must be a YAML mapping")
    try:
        protocol = Protocol.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422,
                            detail=f"the protocol is incomplete — {_validation_message(exc)}")
    return protocol


def _options(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        raise HTTPException(status_code=422, detail="options must be a JSON object")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="options must be a JSON object")

    out: dict[str, Any] = {}
    for name in ("budget_usd", "max_usd_per_paper"):
        if parsed.get(name) not in (None, ""):
            value = float(parsed[name])
            if value <= 0:
                raise HTTPException(status_code=422, detail=f"{name} must be positive")
            out[name] = value
    if parsed.get("max_papers") not in (None, ""):
        out["max_papers"] = max(1, int(parsed["max_papers"]))
    out["concurrency"] = max(1, min(MAX_CONCURRENCY, int(parsed.get("concurrency") or 4)))
    out["resume"] = bool(parsed.get("resume", True))
    models = parsed.get("models") or {}
    if models:
        if not isinstance(models, dict):
            raise HTTPException(status_code=422, detail="models must be an object")
        unknown = sorted(set(models) - set(MODELS))
        if unknown:
            raise HTTPException(status_code=422,
                                detail=f"unknown model role(s) {unknown} (roles: {sorted(MODELS)})")
        out["models"] = {k: str(v)[:80] for k, v in models.items() if str(v).strip()}
    if parsed.get("profile"):
        profile = str(parsed["profile"])
        if profile not in available_profiles():
            raise HTTPException(status_code=422, detail=f"unknown stats profile {profile!r} "
                                                        f"(available: {available_profiles()})")
        out["profile"] = profile
    out["start"] = bool(parsed.get("start", True))
    out["name"] = str(parsed.get("name") or "")[:60]
    return out


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
               loopback_only: bool = True) -> FastAPI:
    """Build the API. `client_factory` is the seam a test fills with a fake or replaying client."""
    app = FastAPI(title="Canopy", docs_url=None, redoc_url=None, openapi_url=None)
    manager = JobManager(runs_dir, client_factory=client_factory)
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
        """A folder of PDFs and a protocol become a run directory and a background job."""
        chosen = _options(options)
        text = protocol_text
        if not text and protocol is not None:
            text = protocol.file.read().decode("utf-8", "replace")
        if not text.strip():
            raise HTTPException(status_code=422, detail="a run needs a protocol (YAML)")
        if not files:
            raise HTTPException(status_code=422, detail="a run needs at least one PDF")
        if len(files) > DEFAULT_MAX_FILES:
            raise HTTPException(status_code=413,
                                detail=f"{len(files)} files is over the {DEFAULT_MAX_FILES} limit")
        if chosen["start"] and manager.key_required():
            raise HTTPException(status_code=400, detail="no ANTHROPIC_API_KEY is configured — "
                                                        "add one to .env and try again")

        parsed = _parse_protocol(text)
        if chosen.get("profile"):
            # the same rule as `canopy run --profile`: an explicit choice replaces the
            # protocol's own statistics block, and the run directory keeps what it used
            parsed.stats = apply_profile(StatsSettings(profile=chosen["profile"]))

        payloads: list[tuple[str, bytes]] = []
        total = 0
        for upload in files:
            data = upload.file.read()
            try:
                validate_pdf(upload.filename or "upload.pdf", data, app.state.max_upload_bytes)
            except UploadRejected as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc))
            total += len(data)
            if total > app.state.max_total_bytes:
                raise HTTPException(status_code=413,
                                    detail=f"this upload is over the "
                                           f"{app.state.max_total_bytes / 1e6:.0f} MB total limit "
                                           f"(raise it with CANOPY_MAX_UPLOAD_TOTAL_MB)")
            payloads.append((safe_filename(upload.filename or "upload.pdf"), data))

        job = manager.create(title=parsed.title or chosen.get("name") or "review",
                             options=chosen)
        try:
            dump_protocol(parsed, job.run_dir / "protocol.yaml")
            saved = [save_upload(job.run_dir / "uploads", data) for _, data in payloads]
            names = {str(path): name for path, (name, _) in zip(saved, payloads)}
            (job.run_dir / "uploads" / "filenames.json").write_text(
                json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
            probes = probe_many(sorted(set(saved)), timeout=app.state.probe_timeout)
            unreadable = [f"{names.get(path, Path(path).name)}: {result['error']}"
                          for path, result in probes.items() if not result.get("ok")]
            if unreadable:
                raise HTTPException(status_code=400, detail="; ".join(unreadable[:5]))
        except BaseException:
            shutil.rmtree(job.run_dir, ignore_errors=True)
            manager.forget(job.run_id)
            raise

        job.n_files = len(set(saved))
        job.save()
        if chosen["start"]:
            manager.start(job)
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
        if job.status in ("running", "queued"):
            raise HTTPException(status_code=409, detail="this run is already going")
        if manager.key_required():
            raise HTTPException(status_code=400, detail="no ANTHROPIC_API_KEY is configured — "
                                                        "add one to .env and try again")
        job.kind = "run"
        manager.start(job)
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
        rows = []
        for row in _read_json(directory / "extraction_table.json", []) or []:
            flags = row.get("flags") or []
            rows.append({**row, "in_primary": row.get("dataset_id") not in held,
                         "overridden": "human_override" in flags})
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
            "figures": {name: _file_url(job, directory / f"{name}.png")
                        for name in ("sensitivity", "funnel")
                        if (directory / f"{name}.png").exists()},
            "review": [e for e in _review_queue(job) if e.get("outcome_key") == outcome_key],
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
        return sort_review_queue([e for e in queue if isinstance(e, dict)])

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
        if job.status in ("running", "queued"):
            raise HTTPException(status_code=409, detail="this run is already busy")
        if manager.key_required():
            raise HTTPException(status_code=400, detail="a dry run needs a model: set "
                                                        "ANTHROPIC_API_KEY in .env")
        manager.start_dry_run(job, max_papers=int(body.get("max_papers") or 3))
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
