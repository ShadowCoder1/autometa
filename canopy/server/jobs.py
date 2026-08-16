"""Background runs: one thread per job, one event stream per subscriber, one file per job.

The pipeline is synchronous and thread-based (`canopy.pipeline.run`), so a job here is a plain
thread with three things around it:

* **an event log.** Every progress event the pipeline emits is numbered and kept, so a browser
  that connects late — or reconnects — sees the whole run rather than the tail of it. The SSE
  stream replays the log, then follows the live queue, then sends one terminal `end` event.
* **a cancel flag.** `run_pipeline` has no cancel parameter and this package may not change it, so
  cancellation is expressed through the one channel the pipeline already offers: the progress
  callback raises `RunCancelled` at the next per-paper event. The orchestrator treats that as that
  paper's failure, carries on to the next (which cancels immediately too), and still writes the
  manifest and the outputs for whatever finished. A cancelled run is therefore resumable, which is
  the behaviour a person pressing "stop" actually wants.
* **`job.json`.** The registry is on disk, in the run directory, so restarting `canopy serve` lists
  every past run with its status, its cost and its token.

The `client_factory` seam is how tests (and anything embedding this server) supply a fake or
replaying `LLMClient` instead of the Anthropic-backed one.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import os

from ..config import MODELS, api_key, live_enabled
from ..pipeline.run import RunCancelled
from ..pipeline.state import sha12
from ..protocol import load_protocol
from .uploads import DEFAULT_INGEST_TIMEOUT, ingest_pdf_subprocess

__all__ = ["Job", "JobManager", "JobBusy", "TooManyRuns", "RunCancelled",
           "default_client_factory", "sse_pack", "TERMINAL_STATES", "STREAM_END_STATES"]

TERMINAL_STATES = frozenset({"done", "error", "cancelled", "interrupted"})
#: a run nobody has started yet has nothing to stream either — the stream says so and closes
STREAM_END_STATES = TERMINAL_STATES | {"created"}
MAX_EVENTS = 20_000
HEARTBEAT_SECONDS = 15.0
#: how many reviews may run at once (each is a thread pool of its own, and an API budget)
MAX_ACTIVE_RUNS = max(1, int(os.environ.get("CANOPY_MAX_ACTIVE_RUNS", "2") or 2))


class JobBusy(RuntimeError):
    """This run is already going."""


class TooManyRuns(RuntimeError):
    """The server is already running as many reviews as it allows."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sse_pack(event: str, data: Mapping[str, Any], seq: int) -> str:
    """One server-sent event. `data` is JSON on a single line, whatever it contains."""
    payload = json.dumps(dict(data), ensure_ascii=False, default=str)
    return f"id: {seq}\nevent: {event}\ndata: {payload}\n\n"


def default_client_factory(run_dir: Path | None = None, *, budget_usd: float | None = None,
                           concurrency: int = 4, **_: Any) -> Any:
    """The real client: disk cache in the run directory, the run's budget, live calls allowed."""
    from ..llm.client import LLMClient

    return LLMClient(cache_dir=None if run_dir is None else Path(run_dir) / "cache",
                     budget_usd=budget_usd, max_concurrency=max(6, concurrency * 2),
                     allow_live=True)


# ----------------------------------------------------------------------------- one job
@dataclass
class Job:
    run_id: str
    run_dir: Path
    token: str
    title: str = ""
    created_at: str = field(default_factory=_now)
    status: str = "created"                    # created|queued|running|done|error|cancelled
    options: dict[str, Any] = field(default_factory=dict)
    n_files: int = 0
    cost_usd: float = 0.0
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    kind: str = "run"                          # run | dry-run
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    seq: int = 0
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    subscribers: list["queue.Queue[dict[str, Any]]"] = field(default_factory=list, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "token": self.token, "title": self.title,
                "created_at": self.created_at, "status": self.status, "options": self.options,
                "n_files": self.n_files, "cost_usd": self.cost_usd, "error": self.error,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "kind": self.kind}

    def save(self) -> None:
        path = self.run_dir / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:                                    # pragma: no cover - exotic filesystem
            pass
        tmp.replace(path)

    @classmethod
    def load(cls, run_dir: Path) -> "Job | None":
        path = Path(run_dir) / "job.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:                                 # pragma: no cover - defensive
            return None
        status = str(data.get("status", "created"))
        if status in ("running", "queued"):                # the server died while it was running
            status = "interrupted"
        return cls(run_id=str(data.get("run_id") or Path(run_dir).name), run_dir=Path(run_dir),
                   token=str(data.get("token", "")), title=str(data.get("title", "")),
                   created_at=str(data.get("created_at", "")), status=status,
                   options=dict(data.get("options") or {}), n_files=int(data.get("n_files") or 0),
                   cost_usd=float(data.get("cost_usd") or 0.0), error=str(data.get("error", "")),
                   started_at=str(data.get("started_at", "")),
                   finished_at=str(data.get("finished_at", "")),
                   kind=str(data.get("kind", "run")))

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATES

    # ------------------------------------------------------------------ events
    def publish(self, event: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.seq += 1
            numbered = {**dict(event), "seq": self.seq}
            self.events.append(numbered)
            if len(self.events) > MAX_EVENTS:              # keep the shape, drop the middle
                del self.events[1:len(self.events) - MAX_EVENTS + 1]
            for subscriber in list(self.subscribers):
                try:
                    subscriber.put_nowait(numbered)
                except queue.Full:                         # pragma: no cover - unbounded queues
                    pass
        return numbered

    def subscribe(self) -> tuple[list[dict[str, Any]], "queue.Queue[dict[str, Any]]"]:
        """A snapshot of the log and a queue of everything after it — with no gap between."""
        channel: "queue.Queue[dict[str, Any]]" = queue.Queue()
        with self.lock:
            history = list(self.events)
            self.subscribers.append(channel)
        return history, channel

    def unsubscribe(self, channel: "queue.Queue[dict[str, Any]]") -> None:
        with self.lock:
            if channel in self.subscribers:
                self.subscribers.remove(channel)


# ----------------------------------------------------------------------------- the registry
class JobManager:
    """Every run this server knows about: in memory while it runs, on disk for ever after."""

    def __init__(self, runs_dir: str | Path,
                 client_factory: Callable[..., Any] | None = None,
                 max_active: int = MAX_ACTIVE_RUNS,
                 ingest_timeout: float = DEFAULT_INGEST_TIMEOUT):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.client_factory = client_factory or default_client_factory
        self.uses_real_models = client_factory is None
        self.max_active = max(1, int(max_active))
        self.ingest_timeout = float(ingest_timeout)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ capacity
    def active(self) -> int:
        """Runs this process is working on right now. Call it holding `_lock`, or accept a race."""
        return sum(1 for job in self._jobs.values()
                   if job.status in ("queued", "running")
                   and job.thread is not None and job.thread.is_alive())

    def has_capacity(self) -> bool:
        with self._lock:
            return self.active() < self.max_active

    # ------------------------------------------------------------------ lookup
    def new_run_id(self, name: str = "") -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        clean = "".join(c for c in str(name).lower() if c.isalnum() or c in "-_")[:40]
        base = f"{stamp}-{clean}" if clean else stamp
        run_id, n = base, 2
        while (self.runs_dir / run_id).exists():
            run_id, n = f"{base}-{n}", n + 1
        return run_id

    def create(self, *, title: str, options: Mapping[str, Any], run_id: str | None = None) -> Job:
        from .security import mint_token

        run_id = run_id or self.new_run_id(title)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            run_dir.chmod(0o700)
        except OSError:                                    # pragma: no cover - exotic filesystem
            pass
        job = Job(run_id=run_id, run_dir=run_dir, token=mint_token(), title=str(title),
                  options=dict(options))
        job.save()
        with self._lock:
            self._jobs[run_id] = job
        return job

    def get(self, run_id: str) -> Job | None:
        """The live job, or the one `job.json` remembers. Never touches anything outside runs/."""
        with self._lock:
            job = self._jobs.get(run_id)
        if job is not None:
            return job
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            return None
        run_dir = self.runs_dir / run_id
        if not run_dir.is_dir() or not run_dir.resolve().is_relative_to(self.runs_dir.resolve()):
            return None
        loaded = Job.load(run_dir)
        if loaded is None:
            return None
        with self._lock:
            return self._jobs.setdefault(run_id, loaded)

    def forget(self, run_id: str) -> None:
        """Drop a run from the registry (used when its creation failed half-way)."""
        with self._lock:
            self._jobs.pop(run_id, None)

    def list(self) -> list[Job]:
        """Newest first. Reads every `job.json` under the runs directory."""
        for path in sorted(self.runs_dir.glob("*/job.json")):
            self.get(path.parent.name)
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: (j.created_at, j.run_id), reverse=True)

    # ------------------------------------------------------------------ running
    def start(self, job: Job) -> Job:
        """Queue a run. The busy/capacity checks and the hand-off are one atomic step.

        Both had races worth closing: two clicks on Run could start the same run twice, and two
        runs could pass a capacity check that neither had taken yet.
        """
        with self._lock:
            self._claim(job)
            job.status = "queued"
            job.error = ""
            job.save()
            job.thread = threading.Thread(target=self._run, args=(job,),
                                          name=f"canopy-{job.run_id}", daemon=True)
            job.thread.start()
        return job

    def _claim(self, job: Job) -> None:
        """Caller must hold `_lock`."""
        if job.status in ("queued", "running") or (job.thread is not None
                                                   and job.thread.is_alive()):
            raise JobBusy(f"run {job.run_id} is already going")
        if self.active() >= self.max_active:
            raise TooManyRuns(f"this server runs at most {self.max_active} review(s) at a time "
                              f"(raise it with CANOPY_MAX_ACTIVE_RUNS); the run is saved and can "
                              f"be started when one finishes")

    def cancel(self, job: Job) -> str:
        if job.finished:
            return job.status
        job.cancel.set()                                   # the pipeline checks this itself
        job.publish({"stage": "run", "paper": "", "status": "cancelling",
                     "cost_so_far": job.cost_usd, "message": "stopping after the current step"})
        return "cancelling"

    def _progress(self, job: Job) -> Callable[[dict[str, Any]], None]:
        """Publish, and nothing else. Cancellation is the pipeline's own `cancel_event`."""
        def progress(event: dict[str, Any]) -> None:
            job.cost_usd = float(event.get("cost_so_far") or job.cost_usd)
            job.publish(event)
        return progress

    def _ingest_fn(self) -> Callable[..., Any]:
        """Ingestion in a child process with a timeout — a hostile PDF cannot hang a run."""
        timeout = self.ingest_timeout

        def ingest(pdf: Path, out_dir: Path) -> Any:
            return ingest_pdf_subprocess(pdf, out_dir, timeout=timeout)
        return ingest

    def _finish(self, job: Job, status: str, message: str = "") -> None:
        job.status = status
        job.finished_at = _now()
        job.error = message if status == "error" else job.error
        job.save()
        job.publish({"stage": "run", "paper": "", "status": status, "cost_so_far": job.cost_usd,
                     "message": message})

    def _run(self, job: Job) -> None:
        from ..pipeline.run import run_pipeline

        options = job.options
        job.status = "running"
        job.started_at = _now()
        job.save()
        try:
            client = self.client_factory(run_dir=job.run_dir,
                                         budget_usd=options.get("budget_usd"),
                                         concurrency=int(options.get("concurrency") or 4),
                                         purpose="run")
            manifest = run_pipeline(
                job.run_dir / "uploads", job.run_dir / "protocol.yaml", job.run_dir,
                models=options.get("models") or None, budget_usd=options.get("budget_usd"),
                resume=bool(options.get("resume", True)), max_papers=options.get("max_papers"),
                concurrency=int(options.get("concurrency") or 4),
                progress=self._progress(job), max_usd_per_paper=options.get("max_usd_per_paper"),
                client=client, cancel_event=job.cancel, ingest_fn=self._ingest_fn())
            job.cost_usd = float(manifest.cost_usd)
            # `run_pipeline` re-applies `overrides.jsonl` itself, so a run and a `--resume` from
            # the command line behave identically here
            if job.cancel.is_set():
                self._finish(job, "cancelled", "stopped by the reviewer")
            else:
                self._finish(job, "done", f"{len(manifest.papers)} paper(s), "
                                          f"${manifest.cost_usd:.2f}")
        except RunCancelled:
            self._finish(job, "cancelled", "stopped by the reviewer")
        except Exception as exc:
            self._finish(job, "error", f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ dry run
    def start_dry_run(self, job: Job, max_papers: int = 3) -> Job:
        with self._lock:
            self._claim(job)
            job.status = "running"
            job.kind = "dry-run"
            job.save()
            job.thread = threading.Thread(target=self._dry_run, args=(job, max_papers),
                                          name=f"canopy-dry-{job.run_id}", daemon=True)
            job.thread.start()
        return job

    def dry_run_result(self, job: Job) -> dict[str, Any]:
        path = job.run_dir / "dry_run.json"
        if not path.exists():
            return {"status": "running" if job.status == "running" else "none", "maps": [],
                    "cost_usd": 0.0, "error": job.error}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:                                 # pragma: no cover - defensive
            return {"status": "error", "maps": [], "cost_usd": 0.0,
                    "error": "dry_run.json is unreadable"}

    def _dry_run(self, job: Job, max_papers: int) -> None:
        """Ingest and map two or three papers, and nothing else — no extraction, no pooling."""
        from ..agents.mapper import map_study
        from ..ingest.dedupe import dedupe_pdfs
        from ..ingest.pdf import ingest_pdf

        maps: list[dict[str, Any]] = []
        client = None
        try:
            protocol = load_protocol(job.run_dir / "protocol.yaml")
            models = {**MODELS, **(job.options.get("models") or {})}
            client = self.client_factory(run_dir=job.run_dir, budget_usd=None, concurrency=1,
                                         purpose="dry-run")
            pdfs = sorted((job.run_dir / "uploads").glob("*.pdf"))
            groups = dedupe_pdfs(pdfs)[:max(1, min(int(max_papers), 3))]
            for group in groups:
                label = sha12(group.sha256)
                job.publish({"stage": "dry-run", "paper": label, "status": "started",
                             "cost_so_far": float(client.total_cost()), "message": "ingesting"})
                paper = ingest_pdf(group.representative, job.run_dir / "dry_run" / label)
                if job.cancel.is_set():
                    raise RunCancelled("cancelled by the reviewer")
                study = map_study(client, paper, protocol, model_primary=models["primary"],
                                  model_check=models["secondary"],
                                  model_adjudicate=models["adjudicator"])
                maps.append({**study.model_dump(mode="json"), "filename": paper.filename})
                job.publish({"stage": "dry-run", "paper": label, "status": "done",
                             "cost_so_far": float(client.total_cost()),
                             "message": f"{len(study.datasets)} dataset(s), "
                                        f"eligible={study.eligible}"})
            cost = float(client.total_cost())
            (job.run_dir / "dry_run.json").write_text(json.dumps(
                {"status": "done", "maps": maps, "cost_usd": cost, "error": ""},
                ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            job.cost_usd = cost
            self._finish(job, "done", f"{len(maps)} paper(s) mapped, ${cost:.2f}")
        except RunCancelled:
            self._finish(job, "cancelled", "stopped by the reviewer")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            (job.run_dir / "dry_run.json").write_text(json.dumps(
                {"status": "error", "maps": maps, "cost_usd": 0.0, "error": message},
                ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            self._finish(job, "error", message)

    # ------------------------------------------------------------------ SSE
    def stream(self, job: Job) -> Iterator[str]:
        """Replay the log, follow the live queue, end with one terminal event."""
        history, channel = job.subscribe()
        seen = 0
        try:
            for event in history:
                seen = int(event.get("seq") or seen)
                yield sse_pack("progress", event, seen)
            last_beat = time.monotonic()
            while True:
                if job.status in STREAM_END_STATES and channel.empty():
                    break
                try:
                    event = channel.get(timeout=0.25)
                except queue.Empty:
                    if time.monotonic() - last_beat > HEARTBEAT_SECONDS:
                        last_beat = time.monotonic()
                        yield ": keep-alive\n\n"
                    continue
                if int(event.get("seq") or 0) <= seen:
                    continue
                seen = int(event.get("seq") or seen + 1)
                yield sse_pack("progress", event, seen)
            started = job.status != "created"
            yield sse_pack("end", {
                "stage": "run", "paper": "", "run_id": job.run_id,
                "status": job.status if started else "not_started",
                "cost_so_far": job.cost_usd,
                "message": job.error or ("" if started else
                                         "this run has not been started yet")}, seen + 1)
        finally:
            job.unsubscribe(channel)

    # ------------------------------------------------------------------ misc
    def key_required(self) -> bool:
        """True when this manager would make real API calls and no key is configured."""
        return self.uses_real_models and not api_key() and not live_enabled()
