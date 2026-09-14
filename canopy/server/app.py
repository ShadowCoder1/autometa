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

…and the other door in, where the user has a question rather than a folder. A search never
appears in the run list (it lives in a sibling directory, see `searches.py`) and it ends by
becoming a run through the very same `make_run`:

    POST /api/searches                  ask a question, get {search_id, token}
    GET  /api/searches/{id}             counts, phases, sources and the candidate list
    GET  /api/searches/{id}/events      the same replayable SSE stream a run has
    POST /api/searches/{id}/papers/{key}/decide   tick or untick one paper
    POST /api/searches/{id}/papers/{key}/upload   the PDF for a paywalled one
    POST /api/searches/{id}/papers      PDFs the search missed
    POST /api/searches/{id}/begin       those papers become a run — one search, one run

Nothing here computes a statistic, and nothing here knows anything about any research field: every
label the UI shows comes from the user's own protocol.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urlsplit

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..config import MODELS, api_key, live_enabled, load_env
from ..models import Protocol, StatsSettings
from ..protocol import (apply_profile, available_profiles, dump_protocol, load_protocol,
                        load_yaml_strict)
from ..report import theme
from ..search.models import project as _project_candidate
from .jobs import TERMINAL_STATES, Job, JobBusy, JobManager, TooManyRuns
from .make_run import (MAX_CONCURRENCY, MAX_PROTOCOL_BYTES, PdfSource, RunOptions, make_run,
                       options_from_json as _options, parse_protocol as _parse_protocol,
                       protocol_text_or_413, validation_message as _validation_message)
from .overrides import (OverrideRejected, append_override, append_overrides,
                        apply_overrides_and_repool,
                        override_summary, read_overrides, repool_lock)
from .searches import (DEFAULT_MAX_FETCH_UNSURE, DEFAULT_MAX_SCREENED, DEFAULT_MAX_USD,
                       SearchJobs, pdf_name, search_options_from, state_of)
from .security import (ACCESS_COOKIE, PathRejected, access_cookie_value, access_granted,
                       is_attachment, is_loopback, media_type, safe_run_path, token_matches)
from .uploads import (DEFAULT_MAX_FILES, DEFAULT_MAX_TOTAL_MB, DEFAULT_MAX_UPLOAD_MB,
                      DEFAULT_PROBE_TIMEOUT, UploadRejected, probe_pdf, safe_filename,
                      stream_upload)

__all__ = ["create_app", "serve"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "protocols"
#: methods that change something — a page on another origin may not use them
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
#: the one path a browser without the site's access cookie may reach
ACCESS_PATH = "/access"


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
def _access_page(*, wrong: bool) -> str:
    """A single field, no framework: the page a visitor sees before anything else loads."""
    note = ("<p class=\"bad\">That code is not right.</p>" if wrong
            else "<p>This AutoMeta server is private. Enter the access code you were given.</p>")
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>AutoMeta — access</title><style>"
        "html{color-scheme:light}body{margin:0;min-height:100vh;display:grid;place-items:center;"
        "background:#f4f5f7;color:#1b1f24;font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}"
        "form{background:#fff;border:1px solid #d9dde3;border-radius:6px;padding:28px 32px;width:min(92vw,380px)}"
        "h1{font-size:18px;margin:0 0 6px;letter-spacing:.01em}p{margin:0 0 18px;color:#5b6470}"
        ".bad{color:#a63a2b}label{display:block;font-size:12px;letter-spacing:.06em;"
        "text-transform:uppercase;color:#5b6470;margin-bottom:6px}"
        "input{width:100%;box-sizing:border-box;font:inherit;padding:9px 10px;border:1px solid #b6bcc4;border-radius:4px}"
        "input:focus{outline:2px solid #25344a;outline-offset:1px}"
        "button{margin-top:14px;width:100%;font:inherit;font-weight:600;padding:10px;border:0;"
        "border-radius:4px;background:#25344a;color:#fff;cursor:pointer}"
        "</style></head><body><form method=\"post\" action=\"/access\" autocomplete=\"off\">"
        "<h1>AutoMeta</h1>" + note +
        "<label for=\"code\">Access code</label>"
        "<input id=\"code\" name=\"code\" type=\"password\" autofocus required>"
        "<button type=\"submit\">Enter</button></form></body></html>")


def create_app(runs_dir: str | Path = "runs", *,
               client_factory: Callable[..., Any] | None = None,
               allowed_hosts: Sequence[str] | None = None,
               max_upload_mb: float | None = None,
               probe_timeout: float | None = None,
               max_active_runs: int | None = None,
               loopback_only: bool = True,
               searches_dir: str | Path | None = None,
               search_runner: Callable[..., Any] | None = None,
               access_code: str | None = None) -> FastAPI:
    """Build the API. `client_factory` is the seam a test fills with a fake or replaying client.

    `search_runner` is the same kind of seam for the OTHER pipeline: the whole paper search as one
    callable (see `searches.SearchJobs`), defaulting to `canopy.search.run.run_search`. A test
    passes a fake and the search endpoints then reach no index and no model.

    `searches_dir` defaults to a SIBLING of the runs directory, so that a search directory can
    never be a run directory: `GET /api/runs` must not offer a search as a review with no papers.
    The reason it is a sibling rather than `runs/searches/` is NOT that the nested layout would
    break — `JobManager.list()` globs `<dir>/*/job.json`, one component, and a nested search's
    `job.json` sits two deep, so it would be missed today. It is that "missed by a glob" is an
    accident of the pattern, and one `**` in a later edit would put every search on the review
    list. A sibling makes the separation structural. `test_a_search_never_appears_in_the_run_list`
    passes both directories explicitly, so it pins the endpoint and not this default.
    """
    app = FastAPI(title="AutoMeta", docs_url=None, redoc_url=None, openapi_url=None)
    manager = JobManager(runs_dir, client_factory=client_factory,
                         **({} if max_active_runs is None else {"max_active": max_active_runs}))
    searches = SearchJobs(searches_dir if searches_dir is not None
                          else Path(manager.runs_dir).parent / "searches",
                          client_factory=client_factory, runner=search_runner,
                          max_active=max_active_runs)
    app.state.runs_dir = str(manager.runs_dir)
    app.state.searches_dir = str(searches.runs_dir)
    app.state.jobs = manager
    app.state.searches = searches
    app.state.max_upload_bytes = (max_upload_mb if max_upload_mb is not None
                                  else DEFAULT_MAX_UPLOAD_MB) * 1e6
    app.state.probe_timeout = (probe_timeout if probe_timeout is not None
                               else DEFAULT_PROBE_TIMEOUT)
    app.state.max_total_bytes = max(app.state.max_upload_bytes, DEFAULT_MAX_TOTAL_MB * 1e6)
    app.state.allowed_hosts = [h.lower() for h in (allowed_hosts or [])]
    app.state.loopback_only = bool(loopback_only)
    # A site-wide access code, for a server that is reachable from the internet. Per-run tokens
    # keep one visitor's run from another, but nothing else stops a stranger from STARTING a
    # run — on this key, with this budget. Unset means no gate, so the loopback server is
    # unchanged; set, every request needs the cookie the code earns, except the form itself.
    app.state.access_code = (access_code or "").strip() or None

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
        code = request.app.state.access_code
        if code and request.url.path != ACCESS_PATH and not access_granted(
                request.cookies.get(ACCESS_COOKIE), request.headers.get("x-canopy-access"), code):
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "this site needs its access code"},
                                    status_code=401)
            return RedirectResponse(ACCESS_PATH, status_code=303)
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

    def search_of(search_id: str, request: Request) -> Job:
        """The search this request may touch — the same two refusals, in the same order.

        Deliberately a copy of `run_of` rather than a shared helper parameterised by manager:
        the two say different words ("no such run" / "no such search") and a future change to one
        must not silently change the other. What they DO share is the order — unknown id is 404
        before the token is looked at, so a wrong token on a real search and any token on a
        made-up one are told apart only by whoever holds the right token.
        """
        job = searches.get(search_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such search")
        header = request.headers.get("authorization") or ""
        given = header[7:].strip() if header.lower().startswith("bearer ") else ""
        given = given or (request.query_params.get("token") or "")
        if not token_matches(given, job.token):
            raise HTTPException(status_code=401, detail="this search needs its own token")
        return job

    # ------------------------------------------------------------------ static SPA
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------ site access code
    @app.get(ACCESS_PATH, response_class=HTMLResponse)
    def access_form(request: Request) -> Any:
        """The one page a browser without the cookie can see."""
        if not request.app.state.access_code:
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_access_page(wrong=False), headers={"Cache-Control": "no-store"})

    @app.post(ACCESS_PATH)
    def access_submit(request: Request, code: str = Form(default="")) -> Any:
        """A right code earns the cookie; the cookie holds a digest of the code, never the code."""
        expected = request.app.state.access_code
        if not expected:
            return RedirectResponse("/", status_code=303)
        if not token_matches(code.strip(), expected):
            return HTMLResponse(_access_page(wrong=True), status_code=401,
                                headers={"Cache-Control": "no-store"})
        response = RedirectResponse("/", status_code=303)
        # behind Cloud Run's proxy the app sees http; the browser saw https
        https = (request.url.scheme == "https"
                 or (request.headers.get("x-forwarded-proto") or "").lower() == "https")
        response.set_cookie(ACCESS_COOKIE, access_cookie_value(expected), max_age=30 * 86400,
                            httponly=True, secure=https, samesite="lax", path="/")
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():                              # pragma: no cover - build sanity
            raise HTTPException(status_code=500, detail="the UI files are missing")
        return FileResponse(page, media_type="text/html; charset=utf-8")

    def _run_cost() -> tuple[float, str]:
        """Dollars per paper the review will bill, from the last completed run when one exists."""
        try:
            from ..search.cost import run_cost_per_paper

            return run_cost_per_paper(manager.runs_dir)
        except Exception:                                  # pragma: no cover - a courtesy number
            return 7.0, "the default"

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
            # …and the same three questions about the OTHER door. `api_key_configured` above
            # already answers "is there a key"; `search_key_required` answers the one the New
            # search form actually asks — "will this search be worse without one?" — because a
            # search with no model still runs, on template queries and with nothing screened.
            "search_key_required": searches.key_required(),
            "search_uses_real_models": searches.uses_real_models,
            "search_max_usd": DEFAULT_MAX_USD,
            "search_max_screened": DEFAULT_MAX_SCREENED,
            # …and two facts the Find panel warns about: Unpaywall needs a real address, and
            # every unsure PDF a search fetches is a paper the review reads at this price
            "contact_email_set": bool(os.environ.get("CANOPY_CONTACT_EMAIL", "").strip()),
            # only whether one is set — never the key. Anonymous OpenAlex throttles any Boolean
            # search with more than five operators, which every block string is (design 03 §2)
            "openalex_key_set": bool(os.environ.get("CANOPY_OPENALEX_KEY", "").strip()),
            "run_cost_per_paper": _run_cost()[0],
            "run_cost_source": _run_cost()[1],
            "search_max_fetch_unsure": DEFAULT_MAX_FETCH_UNSURE,
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
        # the size rule stays HERE, above the shared helper: only this endpoint knows the text
        # may have arrived as an upload rather than a form field, and `protocol_text_or_413` is
        # the rule itself so the search's own door refuses the same document the same way
        protocol_text_or_413(text)
        # …and everything from here down is what BOTH doors do, so both do it in one place.
        # `stream_upload`/`probe_pdf` are passed as this module's globals on purpose: the
        # one-file-at-a-time pin monkeypatches them here, and a helper resolving its own copies
        # would make that pin green without testing anything.
        job = make_run(manager, protocol_text=text, options=chosen,
                       sources=[PdfSource(safe_filename(upload.filename or "upload.pdf"),
                                          (lambda u=upload: u.file)) for upload in files],
                       max_upload_bytes=app.state.max_upload_bytes,
                       max_total_bytes=app.state.max_total_bytes,
                       probe_timeout=app.state.probe_timeout,
                       max_files=DEFAULT_MAX_FILES,
                       stream_upload=stream_upload, probe_pdf=probe_pdf)
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
        # …and the same two checks for every SLOT answered on a card that takes its slots one at
        # a time (§C1, second fold): the slot must be one the card offers for answering on its
        # own, and a picked option must echo the fingerprint it was shown under.
        slot_answers = (body or {}).get("slots")
        if slot_answers is not None and not isinstance(slot_answers, list):
            raise HTTPException(status_code=422, detail="`slots` must be a list of slot answers")
        seen_slots: set[str] = set()
        for entry in slot_answers or []:
            if not isinstance(entry, dict):
                raise HTTPException(status_code=422, detail="each slot answer must be an object")
            named = str(entry.get("slot") or "")
            slot = next((s for s in question.get("slots") or []
                         if str(s.get("member_id") or "") == named and s.get("answerable")), None)
            if slot is None:
                raise HTTPException(status_code=409,
                                    detail=f"question #{number} has no slot {named!r} that is "
                                           f"answered on its own; reload the questions")
            if named in seen_slots:
                raise HTTPException(status_code=422,
                                    detail=f"slot {named!r} is answered twice in one request; "
                                           f"one answer per slot")
            seen_slots.add(named)
            picked = str(entry.get("option") or "")
            if not picked:
                # a typed value or a hint is an answer; an entry with nothing in it is not, and
                # `answers_to_overrides` refuses it below (422) rather than recording "reviewed"
                continue
            option = next((o for o in slot.get("options") or [] if o.get("key") == picked), None)
            if option is None:
                raise HTTPException(status_code=409,
                                    detail=f"slot {named!r} of question #{number} no longer "
                                           f"offers {picked!r}")
            echoed = str(entry.get("option_fingerprint") or "")
            if not echoed:
                raise HTTPException(status_code=422,
                                    detail="a slot answer that picks an option must echo its "
                                           "`fingerprint`")
            if echoed != option.get("fingerprint"):
                raise HTTPException(status_code=409,
                                    detail=f"{picked!r} on slot {named!r} of question #{number} "
                                           f"is not the option you were shown; reload the "
                                           f"questions")
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

    # ------------------------------------------------------------------ searches
    # A search is the OTHER way into a run: a question instead of a folder. Everything below is
    # additive — no route above it changed — and every one of these endpoints sits inside the same
    # cross-site guard and takes the same per-object bearer token as a run.
    def editable(job: Job) -> None:
        """A search may be edited only when it has stopped, and only until it becomes a run.

        Two 409s, both closing a hole the design review found:

        * **still running** — the pipeline is rewriting `search.json` from its own thread. A
          `keep` toggle or an upload merged into that would be lost the next time the pipeline
          saved, and a `begin` half-way through the fetch stage would build a review out of
          whichever PDFs happened to have landed by then.
        * **already begun** — the run has its own copy of the PDFs and of `search.json`. A change
          made here afterwards would look accepted and simply not be in the review.
        """
        if job.status not in TERMINAL_STATES:
            raise HTTPException(status_code=409,
                                detail="this search is still going — wait for it to finish "
                                       "before changing its papers")
        begun = searches.run_id_of(job)
        if begun:
            raise HTTPException(status_code=409,
                                detail=f"this search has already become review {begun}; its "
                                       f"papers are fixed. Start another search to look again.")

    def stage_upload(job: Job, upload: UploadFile) -> tuple[Path, str]:
        """One PDF from the wire into `staging/<sha256>.pdf`, with the same caps a run uses.

        The total cap counts what the search ALREADY holds, not just this request: without that,
        one file at a time is an unbounded directory.
        """
        name = safe_filename(upload.filename or "upload.pdf")
        remaining = app.state.max_total_bytes - searches.staged_bytes(job)
        try:
            path, _size = stream_upload(upload.file, searches.staging_dir(job), name,
                                        max_bytes=app.state.max_upload_bytes,
                                        remaining_bytes=remaining)
        except UploadRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc))
        return path, name

    def readable_pdf(job: Job, record: Any, path: Path, name: str) -> dict[str, Any]:
        """Probe a staged upload; on failure remove it — unless another candidate is using it.

        Two candidates can share one file (a PDF is stored under its own sha256), so an
        unconditional `unlink` here would delete a paper somebody else's row is pointing at.
        """
        probe = probe_pdf(path, timeout=app.state.probe_timeout)
        if not probe.get("ok"):
            relative = searches.relative_pdf(job, path)
            if not any(c.pdf_path == relative for c in record.candidates):
                path.unlink(missing_ok=True)
            why = str(probe.get("error") or "the PDF could not be read")
            raise HTTPException(status_code=400, detail=f"{name}: {why}")
        return probe

    def paper_of(record: Any, key: str) -> Any:
        """The candidate this URL names. A malformed key answers exactly as an unknown one does.

        `searches.candidate` checks `KEY_RE` before it looks at anything, so a key from a URL
        never reaches code that could join it onto a path — and the 404 says the same thing for
        `../../etc/passwd` as for a key that is merely not in this search.
        """
        candidate = searches.candidate(record, key)
        if candidate is None:
            raise HTTPException(status_code=404, detail="no such paper in this search")
        return candidate

    def begin_answer(run: Job, skipped: list[dict[str, str]],
                     run_commit: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """EXACTLY the body `POST /api/runs` returns, plus `skipped` and `run_commit`.

        Same keys in the same shape on purpose: the page's `attach()` already knows how to take a
        run over from that body, and a second door answering in its own dialect would need a
        second copy of the code that reads it. `run_commit` is what the review will read and what
        that costs — the sentence the page confirms before it navigates.
        """
        return {"run_id": run.run_id, "token": run.token, "n_files": run.n_files,
                "status": run.status, "title": run.title, "skipped": skipped,
                "run_commit": dict(run_commit or {})}

    @app.get("/api/searches/preview")
    def search_preview(max_usd: float | None = None, max_fetch_unsure: int | None = None,
                       snowball: bool = True) -> dict[str, Any]:
        """What a search with these numbers would spend, and what beginning it could commit —
        the Find panel's cost line, from the pipeline's own arithmetic (`cost.predict`)."""
        from ..search.cost import predict

        budget = float(max_usd) if max_usd is not None and max_usd > 0 else DEFAULT_MAX_USD
        unsure = (int(max_fetch_unsure) if max_fetch_unsure is not None and max_fetch_unsure >= 0
                  else DEFAULT_MAX_FETCH_UNSURE)
        per_paper, source = _run_cost()
        preview = predict(budget, max_fetch_unsure=unsure, per_paper=per_paper,
                          snowball=bool(snowball))
        preview["run_commit"]["per_paper_source"] = source
        return preview

    @app.get("/api/searches")
    def list_searches() -> dict[str, Any]:
        """Every search on disk, newest first. The token comes back only on a loopback server.

        This is the ONLY way back to a search the browser has forgotten, and forgetting is easy:
        the page keeps its tokens in `localStorage`, so clearing site data, opening the review on
        another browser, or simply starting a second search leaves a finished — possibly already
        paid-for — search with no route to it. The rows are shaped for that job and no other: a
        row carries `search_id` and, on a loopback server, `token`, which is exactly the pair
        `attachSearch(id, token)` takes. `run_id` says which of them already became a review.

        Deliberately the same shape and the same rules as `GET /api/runs`, including handing the
        token back on loopback: the run list is how the page already re-opens a review it has
        lost, and a search should not be harder to find than a run.
        """
        rows = []
        for job in searches.list():
            row: dict[str, Any] = {"search_id": job.run_id, "question": job.title,
                                   "created_at": job.created_at, "status": job.status,
                                   "cost_usd": round(job.cost_usd, 4),
                                   "run_id": str(job.options.get("run_id") or "")}
            if app.state.loopback_only:
                row["token"] = job.token
            rows.append(row)
        return {"searches": rows, "loopback_only": app.state.loopback_only}

    @app.post("/api/searches", status_code=201)
    def create_search(body: dict[str, Any]) -> dict[str, Any]:
        """A question becomes a search directory and a background job.

        No API-key check, unlike `POST /api/runs`: a search with no model still runs — the
        queries come from the protocol's own words and nothing is screened — and refusing it
        would take away the one part of Canopy that works without a card. `/api/settings` says
        `search_key_required` so the form can warn instead.
        """
        question = str((body or {}).get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=422,
                                detail="say what you are looking for, in one question")
        text = str((body or {}).get("protocol_text") or "")[:MAX_PROTOCOL_BYTES + 1]
        protocol_text_or_413(text)                         # the same 413, from both doors
        chosen = search_options_from((body or {}).get("options"))
        if not searches.has_capacity():
            # checked before the directory exists: a 429 that leaves a half-built search behind
            # puts a row on the user's list that can never be started
            raise HTTPException(status_code=429,
                                detail=f"this server already runs {searches.max_active} "
                                       f"search(es) at a time; wait for one to finish (or raise "
                                       f"CANOPY_MAX_ACTIVE_RUNS)")
        job, _record = searches.create_search(question=question[:4000],
                                              options=chosen.resolved(), protocol_text=text)
        try:
            _start(searches, job)
        except HTTPException:                              # lost the capacity race after all
            shutil.rmtree(job.run_dir, ignore_errors=True)
            searches.forget(job.run_id)
            raise
        return {"search_id": job.run_id, "token": job.token, "status": job.status}

    @app.get("/api/searches/{search_id}")
    def search_state(search_id: str, request: Request) -> dict[str, Any]:
        """The whole search as the page reads it — counts, ladder, sources and candidates.

        A stopped search is repaired first. The pipeline only writes its candidates when it
        returns, so a server that died mid-search left staged PDFs that nothing pointed at:
        `searches.recover_staged` gives each of them a row, which is the difference between "the
        server restarted" and "the papers this search fetched are gone". It is idempotent and it
        never runs for a search that is still going — see the method's own docstring.
        """
        job = search_of(search_id, request)
        record = searches.load_record(job)
        # the repair takes the record lock, so it is entered only when there is something to
        # repair: `begin` can hold that lock for as long as it takes to copy forty PDFs into a
        # run, and a poll that queued behind it would look to the user like a hung page
        if job.status in TERMINAL_STATES and searches.orphan_staged(job, record):
            with searches.record_lock(job):
                record = searches.load_record(job)
                searches.recover_staged(job, record)
        return state_of(job, record)

    @app.get("/api/searches/{search_id}/events")
    def search_events(search_id: str, request: Request) -> StreamingResponse:
        job = search_of(search_id, request)
        return StreamingResponse(
            searches.stream(job), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"})

    @app.post("/api/searches/{search_id}/cancel")
    def cancel_search(search_id: str, request: Request) -> dict[str, Any]:
        job = search_of(search_id, request)
        return {"search_id": job.run_id, "status": searches.cancel(job)}

    @app.post("/api/searches/{search_id}/papers/{key}/decide")
    def decide_paper(search_id: str, key: str, request: Request,
                     body: dict[str, Any]) -> dict[str, Any]:
        """Tick or untick one paper. `state` never moves: a paper dropped by hand still shows
        how it was found and what the screener said about it."""
        job = search_of(search_id, request)
        editable(job)
        if "keep" not in (body or {}):
            raise HTTPException(status_code=422, detail='send {"keep": true} or {"keep": false}')
        with searches.record_lock(job):
            record = searches.load_record(job)
            candidate = paper_of(record, key)
            searches.set_keep(job, record, candidate, bool(body["keep"]))
        return {"key": candidate.key, "keep": candidate.keep}

    @app.post("/api/searches/{search_id}/papers/{key}/upload")
    def upload_paper(search_id: str, key: str, request: Request,
                     file: UploadFile = File(...)) -> dict[str, Any]:
        """The PDF for a paper the search could not fetch — the paywall escape hatch.

        Probed here rather than at `begin` so the person who chose the file is still looking when
        they are told it is not a paper.
        """
        job = search_of(search_id, request)
        editable(job)
        with searches.record_lock(job):
            record = searches.load_record(job)
            candidate = paper_of(record, key)
            path, name = stage_upload(job, file)
            probe = readable_pdf(job, record, path, name)
            searches.attach_pdf(job, record, candidate, path=path, filename=name, probe=probe)
        return _project_candidate(candidate)

    @app.post("/api/searches/{search_id}/papers")
    def add_papers(search_id: str, request: Request,
                   files: list[UploadFile] = File(default_factory=list)) -> dict[str, Any]:
        """Papers the search missed. One bad file does not fail the request: the others land and
        the refusals come back named, because a person adding six PDFs by hand should not have to
        binary-search which one this server dislikes."""
        job = search_of(search_id, request)
        editable(job)
        if not files:
            raise HTTPException(status_code=422, detail="choose at least one PDF")
        if len(files) > DEFAULT_MAX_FILES:
            raise HTTPException(status_code=413,
                                detail=f"{len(files)} files is over the {DEFAULT_MAX_FILES} "
                                       f"limit")
        added: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        with searches.record_lock(job):
            record = searches.load_record(job)
            for upload in files:
                name = safe_filename(upload.filename or "upload.pdf")
                try:
                    path, name = stage_upload(job, upload)
                    probe = readable_pdf(job, record, path, name)
                except HTTPException as exc:
                    rejected.append({"filename": name, "reason": str(exc.detail)})
                    continue
                added.append(_project_candidate(
                    searches.add_extra(job, record, path=path, filename=name, probe=probe)))
        return {"added": added, "rejected": rejected}

    @app.post("/api/searches/{search_id}/begin", status_code=201)
    def begin_review(search_id: str, request: Request,
                     body: dict[str, Any] | None = None) -> dict[str, Any]:
        """The papers this search found become a run — through `make_run`, like any upload.

        A search becomes exactly ONE run. A second `begin` returns the first one instead of
        spending the user's budget twice on the same papers: the run id is recorded on the
        search, so a double-click, a retry after a dropped connection and a reloaded tab all get
        the same review back with its current status.

        That last sentence is only true because the id is re-read INSIDE `record_lock`. Read once
        before the lock, it is a plain time-of-check/time-of-use race: two begins arriving
        together both saw "not begun yet", both built a run, both started it and both spent the
        budget — and only the second was remembered, so the first was a paid, orphaned run
        directory that no search pointed at and nothing would ever collect. Two threads on one
        barrier reproduced it first try. The fast path below the status check stays because it
        keeps the ordinary repeat-begin cheap and keeps the refusal ORDER unchanged (an oversize
        protocol still 413s before anything else); the read inside the lock is the authority.
        """
        job = search_of(search_id, request)
        if job.status not in TERMINAL_STATES:
            raise HTTPException(status_code=409,
                                detail="this search is still going — wait for it to finish "
                                       "before starting the review")

        def already_begun() -> dict[str, Any] | None:
            """`begin`'s own answer for the run this search has already become, or None."""
            begun = searches.run_id_of(job)
            if not begun:
                return None
            run = manager.get(begun)
            if run is None:                                # deleted from under us
                raise HTTPException(status_code=409,
                                    detail=f"this search became review {begun}, which is no "
                                           f"longer on disk")
            return begin_answer(run, searches.skipped_of(job),
                                (searches.load_record(job).predicted or {}).get("run_commit"))

        answer = already_begun()
        if answer is not None:
            return answer

        body = body or {}
        # the protocol the page sends wins; the one pasted when the search started is the
        # fallback, so a user who wrote it once is not asked for it twice
        text = (str(body.get("protocol_text") or "") or searches.protocol_text(job))
        text = text[:MAX_PROTOCOL_BYTES + 1]
        protocol_text_or_413(text)
        chosen = _options(json.dumps(body.get("options") or {}))
        with searches.record_lock(job):
            # the authoritative idempotency check: whoever holds this lock is the only thread
            # that can be building this search's run (see the docstring)
            answer = already_begun()
            if answer is not None:
                return answer
            record = searches.load_record(job)
            # PDFs a dead server left behind get their rows here too, so a review begun after a
            # restart is built from the papers that are actually on disk
            searches.recover_staged(job, record)
            kept = searches.kept_pdfs(job, record)
            if not kept:
                raise HTTPException(status_code=422,
                                    detail="no paper in this search is both ticked and readable "
                                           "yet — tick the ones you want, and attach a PDF for "
                                           "any that are paywalled")
            rejected: list[dict[str, str]] = []
            # `make_run` reports a skip by the name it streamed, which is `safe_filename` of the
            # source name — so the map is keyed on exactly that string. A list per name, because
            # two candidates can be called the same thing and each of them is still its own row.
            keys_by_name: dict[str, list[str]] = {}
            for candidate, _path in kept:
                keys_by_name.setdefault(safe_filename(pdf_name(candidate)),
                                        []).append(candidate.key)
            run = make_run(
                manager, protocol_text=text, options=chosen,
                sources=[PdfSource(pdf_name(candidate), (lambda p=path: p.open("rb")))
                         for candidate, path in kept],
                max_upload_bytes=app.state.max_upload_bytes,
                max_total_bytes=app.state.max_total_bytes,
                probe_timeout=app.state.probe_timeout, max_files=DEFAULT_MAX_FILES,
                # the run carries its own provenance: what was asked, what each index answered
                # and why every paper is or is not in it, readable without this server. Written
                # here WITHOUT `skipped` — nothing has been skipped yet, and a key that said
                # `[]` before the loop ran would be a claim rather than an absence. The corrected
                # copy is written below, once the skips are facts.
                copy_into={"search/search.json": json.dumps(record.to_json(),
                                                            ensure_ascii=False, indent=1,
                                                            default=str)},
                # a staged PDF was fetched by a machine hours ago; one that no longer probes is
                # reported and skipped rather than destroying a 40-paper review
                on_source_rejected="skip", rejected_out=rejected,
                stream_upload=stream_upload, probe_pdf=probe_pdf)
            skipped: list[dict[str, str]] = []
            for row in rejected:
                name = str(row.get("name") or "")
                pending = keys_by_name.get(name) or []
                skipped.append({"key": pending.pop(0) if pending else "", "name": name,
                                "reason": str(row.get("reason") or "")})
            # the run's own copy, corrected: a review whose `search/search.json` lists forty kept
            # papers while `uploads/` holds one is an audit trail that lies about the run it is
            # inside. Rewritten rather than appended to, so the file is whole either way; the
            # relative path is this endpoint's own literal and never comes from a paper.
            (run.run_dir / "search" / "search.json").write_text(
                json.dumps({**record.to_json(), "skipped": skipped},
                           ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            # recorded BEFORE the run is started, so even a 429 from a busy server leaves the
            # search pointing at a real run directory that can be started later
            searches.remember_run(job, run.run_id, skipped)
            # …and into the SEARCH's own record, so a reload — or a reviewer opening
            # `search.json` next year — can still see which papers never made it and why
            searches.save_record(job, record)
        if chosen.start:
            _start(manager, run)
        return begin_answer(run, skipped, (record.predicted or {}).get("run_commit"))

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
              f"  AutoMeta has no user accounts: the only secret is a per-run token.\n"
              f"  Use 127.0.0.1 unless you have a reason not to.\n")
    access_code = os.environ.get("CANOPY_ACCESS_CODE") or None
    if not loopback and not access_code:
        print("  Set CANOPY_ACCESS_CODE to put a password on the whole site.\n")
    app = create_app(runs_dir, loopback_only=loopback,
                     allowed_hosts=["localhost", "127.0.0.1", "::1"] if loopback else None,
                     access_code=access_code)
    print(f"canopy UI on http://{host}:{port}  ·  runs in {Path(runs_dir).resolve()}  ·  "
          f"API key {'configured' if api_key() else 'NOT configured'}"
          + ("" if loopback else f"  ·  access code {'set' if access_code else 'NOT set'}"))
    uvicorn.run(app, host=host, port=port, log_level="warning")
