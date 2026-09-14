"""Task 12 — the local web UI, offline.

Everything here runs with no API key and no network. The pipeline behind the server is the same
fake-provider pipeline Task 10 tests (`tests/test_pipeline_offline.py`): both fixture PDFs are
really ingested, the fake's quotes are real lines of the real papers, and every stage runs for
real. The server is given that client through its `client_factory`, which is the seam a live
deployment fills with the Anthropic-backed one.

Three groups of tests:

* **units** — upload validation, path safety, the SSE frame format, the override log;
* **the API** — create → poll → results → evidence → files → override → repool → cancel;
* **the security baseline** (amendment I) — token required, traversal refused, the key never
  echoed, and the SPA never writing untrusted text as HTML.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from canopy.llm.client import LLMClient
from canopy.llm.providers import FakeProvider

from .test_pipeline_offline import PDFS, FakeSpec, fake_router

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"
STATIC = ROOT / "canopy" / "server" / "static"
OUTCOME = "late_adaptation"


# ============================================================================ fixtures
@pytest.fixture(scope="module")
def specs(tmp_path_factory) -> "list[FakeSpec]":
    """Both fixture papers, really ingested once for the whole module."""
    from canopy.ingest.pdf import ingest_pdf

    root = tmp_path_factory.mktemp("server_papers")
    papers = [ingest_pdf(path, root / path.stem) for path in PDFS]
    return [FakeSpec(papers[0], 44.6, 30.2), FakeSpec(papers[1], 21.5, 17.9)]


@pytest.fixture
def make_app(tmp_path, specs):
    """`make_app(**kwargs) -> TestClient` with a fake-provider pipeline behind it."""
    from canopy.server.app import create_app

    def factory(**kwargs: Any) -> TestClient:
        def client_factory(run_dir: Path | None = None, **_: Any) -> LLMClient:
            return LLMClient(provider=FakeProvider([fake_router(specs)]), allow_live=True,
                             cache_dir=None)

        kwargs.setdefault("runs_dir", tmp_path / "runs")
        kwargs.setdefault("client_factory", client_factory)
        return TestClient(create_app(**kwargs))

    return factory


@pytest.fixture
def api(make_app) -> TestClient:
    return make_app()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_run(api: TestClient, *, pdfs=PDFS, options: dict[str, Any] | None = None,
               protocol: Path = PROTOCOL) -> Any:
    files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in pdfs]
    files.append(("protocol", ("protocol.yaml", protocol.read_bytes(), "text/yaml")))
    return api.post("/api/runs", files=files,
                    data={"options": json.dumps(options or {})})


def wait_done(api: TestClient, run_id: str, token: str, timeout: float = 300.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = api.get(f"/api/runs/{run_id}", headers=auth(token)).json()
        if body["status"] in ("done", "error", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


@pytest.fixture(scope="module")
def finished(tmp_path_factory, specs) -> dict[str, Any]:
    """One completed run, shared by the read-only tests (running it twice is pure waiting)."""
    from canopy.server.app import create_app

    def client_factory(run_dir: Path | None = None, **_: Any) -> LLMClient:
        return LLMClient(provider=FakeProvider([fake_router(specs)]), allow_live=True,
                         cache_dir=None)

    runs = tmp_path_factory.mktemp("finished_runs")
    api = TestClient(create_app(runs_dir=runs, client_factory=client_factory))
    created = create_run(api, options={"concurrency": 1})
    assert created.status_code == 201, created.text
    body = created.json()
    status = wait_done(api, body["run_id"], body["token"])
    assert status["status"] == "done", status.get("error")
    return {"api": api, "run_id": body["run_id"], "token": body["token"], "runs": runs,
            "status": status}


@pytest.fixture
def cloned(finished, tmp_path, specs) -> dict[str, Any]:
    """A private copy of the finished run, served by a fresh server.

    The review-workflow tests each need a run they may change. Running the pipeline again for
    every one of them would test the pipeline, not the review workflow — so the run directory is
    copied instead, and the new server finds it (and its token) in `job.json`, which is exactly
    what happens when `canopy serve` is restarted.
    """
    from canopy.server.app import create_app

    def client_factory(run_dir: Path | None = None, **_: Any) -> LLMClient:
        return LLMClient(provider=FakeProvider([fake_router(specs)]), allow_live=True,
                         cache_dir=None)

    runs = tmp_path / "cloned"
    runs.mkdir()
    shutil.copytree(Path(finished["runs"]) / finished["run_id"], runs / finished["run_id"])
    api = TestClient(create_app(runs_dir=runs, client_factory=client_factory))
    return {"api": api, "run_id": finished["run_id"], "token": finished["token"], "runs": runs}


# ============================================================================ units: uploads
def _stream(data: bytes, dest, name="paper.pdf", max_bytes=1_000_000, remaining=None):
    from canopy.server.uploads import stream_upload

    return stream_upload(io.BytesIO(data), dest, name, max_bytes=max_bytes,
                         remaining_bytes=remaining)


def test_a_non_pdf_upload_is_refused_by_its_magic_bytes(tmp_path):
    from canopy.server.uploads import UploadRejected

    _stream(b"%PDF-1.7\nreal enough", tmp_path)
    with pytest.raises(UploadRejected) as exc:
        _stream(b"<?php system($_GET[0]); ?>", tmp_path, "evil.pdf")
    assert "PDF" in str(exc.value)
    with pytest.raises(UploadRejected):
        _stream(b"%PDF-1.7 but the name lies", tmp_path, "evil.exe")
    with pytest.raises(UploadRejected):
        _stream(b"", tmp_path, "empty.pdf")
    assert not list(tmp_path.glob(".upload-*")), "a refused upload leaves nothing behind"


def test_an_oversize_upload_is_refused(tmp_path):
    from canopy.server.uploads import UploadRejected

    with pytest.raises(UploadRejected) as exc:
        _stream(b"%PDF-1.7" + b"x" * 5_000, tmp_path, "big.pdf", max_bytes=1_000)
    assert exc.value.status_code == 413
    with pytest.raises(UploadRejected) as total:
        _stream(b"%PDF-1.7" + b"x" * 5_000, tmp_path, "big.pdf", remaining=1_000)
    assert total.value.status_code == 413 and "total" in str(total.value)
    assert not list(tmp_path.iterdir())


def test_an_upload_is_streamed_to_disk_a_chunk_at_a_time(tmp_path, monkeypatch):
    """Nothing larger than one chunk is ever in memory, whatever the file's size."""
    from canopy.server import uploads

    monkeypatch.setattr(uploads, "CHUNK", 1024)
    data = b"%PDF-1.7\n" + b"x" * (8 * 1024)
    reads: list[int] = []

    class Counting(io.BytesIO):
        def read(self, size=-1):                            # noqa: D401 - a spy
            chunk = super().read(size)
            reads.append(len(chunk))
            return chunk

    path, size = uploads.stream_upload(Counting(data), tmp_path, "paper.pdf",
                                       max_bytes=1_000_000)
    assert size == len(data) and path.read_bytes() == data
    assert max(reads) <= 1024, f"a whole {max(reads)}-byte read is not streaming"
    assert len(reads) >= 8


def test_an_upload_is_stored_under_its_own_sha256(tmp_path):
    import hashlib

    data = PDFS[0].read_bytes()
    path, size = _stream(data, tmp_path, PDFS[0].name, max_bytes=1e9)
    assert path.name == f"{hashlib.sha256(data).hexdigest()}.pdf"
    assert path.read_bytes() == data and size == len(data)
    assert _stream(data, tmp_path, PDFS[0].name, max_bytes=1e9)[0] == path   # same bytes, one file


def test_a_hostile_pdf_cannot_hang_the_server(tmp_path):
    """Ingestion runs in a subprocess with a timeout (amendment I)."""
    from canopy.server.uploads import probe_pdf

    good = probe_pdf(PDFS[0], timeout=90)
    assert good["ok"] and good["n_pages"] > 0

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\n" + b"\x00" * 200)
    bad = probe_pdf(broken, timeout=90)
    assert bad["ok"] is False and bad["error"]

    stuck = probe_pdf(PDFS[0], timeout=0.0001)             # a PDF that never finishes
    assert stuck["ok"] is False and "timed out" in stuck["error"]


def test_the_pipelines_own_ingestion_can_run_in_a_child_process(tmp_path):
    """The run's ingest step, in a killable process, writing what a direct call would."""
    from canopy.ingest.pdf import PaperRecord
    from canopy.server.uploads import ingest_pdf_subprocess

    paper = ingest_pdf_subprocess(PDFS[0], tmp_path / "ingest", timeout=120)
    assert isinstance(paper, PaperRecord)
    assert paper.n_pages > 0 and paper.sha256
    assert (tmp_path / "ingest" / "pages" / "p001.png").exists(), "page rasters are needed later"
    assert paper.page_text(1).strip()

    with pytest.raises(TimeoutError) as exc:
        ingest_pdf_subprocess(PDFS[0], tmp_path / "slow", timeout=0.0001)
    assert "timed out" in str(exc.value)


def test_a_paper_that_will_not_ingest_fails_only_itself(tmp_path, specs):
    """A hostile PDF takes its child process down, not the run."""
    from canopy.pipeline.run import run_pipeline

    papers = tmp_path / "papers"
    papers.mkdir()
    shutil.copyfile(PDFS[0], papers / PDFS[0].name)

    def refuse(pdf: Path, out_dir: Path):
        raise TimeoutError("ingestion timed out after 300s: hostile.pdf")

    client = LLMClient(provider=FakeProvider([fake_router(specs)]), allow_live=True,
                       cache_dir=None)
    manifest = run_pipeline(papers, PROTOCOL, tmp_path / "run", client=client, concurrency=1,
                            ingest_fn=refuse)
    assert manifest.papers[0].status == "error"
    assert "timed out" in manifest.papers[0].error
    assert (tmp_path / "run" / "manifest.json").exists()


# ============================================================================ units: paths
def test_the_files_path_resolver_refuses_to_leave_the_run(tmp_path):
    from canopy.server.security import PathRejected, safe_run_path

    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    (run / "results" / "forest.svg").write_text("<svg/>")
    (tmp_path / "secret.env").write_text("ANTHROPIC_API_KEY=sk-ant-nope")

    assert safe_run_path(run, "results/forest.svg").name == "forest.svg"
    for bad in ("../secret.env", "/etc/passwd", "results/../../secret.env",
                "uploads/paper.pdf", "cache/anything.json", ".hidden/x.json",
                "results/forest.exe"):
        with pytest.raises(PathRejected):
            safe_run_path(run, bad)


def test_the_files_path_resolver_refuses_a_symlink_out_of_the_run(tmp_path):
    from canopy.server.security import PathRejected, safe_run_path

    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "outside.json").write_text("{}")
    (run / "escape.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(PathRejected):
        safe_run_path(run, "escape.json")


# ============================================================================ units: SSE
def test_sse_frames_carry_an_event_name_json_data_and_an_id():
    from canopy.server.jobs import sse_pack

    frame = sse_pack("progress", {"stage": "map", "paper": "abc", "status": "done",
                                  "cost_so_far": 0.5, "message": "2 datasets"}, seq=7)
    assert frame.endswith("\n\n")
    lines = frame.strip().splitlines()
    assert lines[0] == "id: 7"
    assert lines[1] == "event: progress"
    assert lines[2].startswith("data: ")
    assert json.loads(lines[2][len("data: "):])["stage"] == "map"
    assert "\n" not in json.loads(lines[2][len("data: "):])["message"]


def test_sse_data_is_one_line_even_when_a_message_has_newlines():
    from canopy.server.jobs import sse_pack

    frame = sse_pack("progress", {"message": "line one\nline two"}, seq=1)
    assert len([l for l in frame.strip().splitlines() if l.startswith("data: ")]) == 1


# ============================================================================ the API
def test_creating_a_run_returns_a_run_id_and_a_token(api):
    created = create_run(api, options={"concurrency": 1})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["run_id"] and body["token"] and len(body["token"]) >= 24
    assert body["n_files"] == 2
    assert body["status"] in ("queued", "running")


def test_a_run_endpoint_needs_its_own_token(api):
    body = create_run(api, options={"concurrency": 1}).json()
    run_id, token = body["run_id"], body["token"]

    assert api.get(f"/api/runs/{run_id}").status_code == 401
    assert api.get(f"/api/runs/{run_id}", headers=auth("not-the-token")).status_code == 401
    assert api.get(f"/api/runs/{run_id}", headers=auth(token)).status_code == 200
    assert api.get(f"/api/runs/{run_id}/results/{OUTCOME}").status_code == 401
    assert api.post(f"/api/runs/{run_id}/cancel").status_code == 401
    assert api.get(f"/api/runs/{run_id}/files/manifest.json").status_code == 401


def test_an_unknown_run_is_a_404_not_a_401_oracle(api):
    assert api.get("/api/runs/nope-not-a-run", headers=auth("x")).status_code == 404


def test_the_manifest_endpoint_reports_the_finished_run(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    body = api.get(f"/api/runs/{run_id}", headers=auth(token)).json()

    assert body["status"] == "done"
    assert body["manifest"]["papers"] and len(body["manifest"]["papers"]) == 2
    assert body["manifest"]["protocol_hash"]
    assert OUTCOME in [o["key"] for o in body["outcomes"]]
    assert body["protocol"]["title"]
    assert body["cost_usd"] >= 0.0


def test_results_carry_the_pooled_estimate_rows_and_sensitivity(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    body = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()

    assert body["pooled"]["k"] == 2
    assert body["pooled"]["estimate"] is not None
    assert len(body["rows"]) == 2
    row = body["rows"][0]
    assert row["dataset_id"] and row["es"] is not None and row["var"] > 0
    assert row["route"] and row["conversion_chain"]
    assert body["sensitivity"]["analyses"]                 # amendment H's sensitivity set
    assert body["forest"]["svg"].endswith("forest.svg")
    assert body["outcome"]["positive_direction_label"]


def test_an_unknown_outcome_is_404(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    assert api.get(f"/api/runs/{run_id}/results/not_an_outcome",
                   headers=auth(token)).status_code == 404


def test_evidence_carries_candidates_verdicts_and_image_urls(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    rows = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()["rows"]
    dataset_id = rows[0]["dataset_id"]

    body = api.get(f"/api/runs/{run_id}/evidence/{dataset_id}/{OUTCOME}",
                   headers=auth(token)).json()
    assert body["dataset_id"] == dataset_id
    assert body["candidates"], "the drawer must show what was read"
    assert {"group", "mean", "quote", "page", "route", "model"} <= set(body["candidates"][0])
    assert body["verdicts"] and body["verdicts"][0]["confidence"]
    assert body["verdicts"][0]["verifier"]["verdict"]      # the adversarial verifier's own words
    assert body["conversion_chain"]
    assert body["record"]["es"] is not None
    images = body["images"]
    assert images and images[0]["url"].startswith(f"/api/runs/{run_id}/files/")

    served = api.get(images[0]["url"], headers=auth(token))
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"


def test_the_files_endpoint_serves_artefacts_and_refuses_traversal(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]

    svg = api.get(f"/api/runs/{run_id}/files/results/{OUTCOME}/forest.svg", headers=auth(token))
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    assert svg.text.lstrip().startswith("<?xml") or "<svg" in svg.text
    assert svg.headers["x-content-type-options"] == "nosniff"

    for bad in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "cache/x.json",
                "uploads/anything.pdf", "protocol.yaml/../../../etc/hosts"):
        response = api.get(f"/api/runs/{run_id}/files/{bad}", headers=auth(token))
        assert response.status_code in (400, 404), f"{bad} was served: {response.status_code}"


def test_the_event_stream_replays_the_run_and_ends(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    with api.stream("GET", f"/api/runs/{run_id}/events", headers=auth(token)) as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert stream.headers["cache-control"] == "no-cache"
        lines = [line for line in stream.iter_lines()]

    events = [json.loads(line[len("data: "):]) for line in lines if line.startswith("data: ")]
    names = [line[len("event: "):] for line in lines if line.startswith("event: ")]
    assert names[-1] == "end"
    assert {"stage", "paper", "status", "cost_so_far", "message"} <= set(events[0])
    assert {e["stage"] for e in events[:-1]} >= {"ingest", "map", "extract", "verify", "resolve"}
    assert events[-1]["status"] == "done"


def test_the_review_queue_is_served_sorted_by_pooled_impact(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    body = api.get(f"/api/runs/{run_id}/review", headers=auth(token)).json()
    impacts = [e.get("impact_abs_delta_pooled") for e in body["queue"]]
    known = [i for i in impacts if i is not None]
    assert known == sorted(known, reverse=True)
    assert impacts[len(known):] == [None] * (len(impacts) - len(known))


def test_listing_runs_survives_a_restart(finished):
    """`canopy serve` restarted must still find the runs on disk (job.json)."""
    from canopy.server.app import create_app

    fresh = TestClient(create_app(runs_dir=finished["runs"]))
    listed = fresh.get("/api/runs").json()["runs"]
    assert finished["run_id"] in [r["run_id"] for r in listed]
    row = next(r for r in listed if r["run_id"] == finished["run_id"])
    assert row["status"] == "done" and row["title"]
    # a loopback client may reclaim the token (it can read the run directory anyway)
    assert row["token"] == finished["token"]
    assert fresh.get(f"/api/runs/{finished['run_id']}",
                     headers=auth(row["token"])).status_code == 200


# ============================================================================ names, not shas
def test_the_run_state_lists_its_papers_by_name_with_their_cost(finished):
    """A sha is how Canopy identifies a paper; it is not how anybody reads one."""
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    body = api.get(f"/api/runs/{run_id}", headers=auth(token)).json()

    papers = body["papers"]
    assert len(papers) == 2
    for paper in papers:
        assert paper["study_label"] and paper["study_label"] != paper["sha12"]
        assert str(paper["year"]) in paper["study_label"]
        assert paper["filename"].endswith(".pdf")
        assert paper["sha12"] == paper["paper_id"][:12]
        assert paper["stages"]["resolve"] in ("done", "skipped")
        assert isinstance(paper["cost_usd"], (int, float))
    assert body["manifest"]["papers"], "the raw manifest is still there for anything else"


def test_results_rows_carry_the_study_the_dataset_and_the_page(finished):
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]
    body = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()

    for row in body["rows"]:
        assert row["study_label"] and row["study_label"] != row["dataset_id"]
        assert row["dataset_label"]
        assert row["pages"].startswith("p"), row["pages"]
        assert row["paper_filename"].endswith(".pdf")
        assert row["dataset_id"], "the id stays, as the second line"
    assert body["papers"] and body["papers"][0]["study_label"]


def test_the_review_queue_is_named_too(cloned):
    """A flag a reviewer cannot place in a paper is a flag they cannot act on."""
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    rows = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()["rows"]
    dataset_id = rows[0]["dataset_id"]

    # make one cell need a human, the way a reviewer would
    api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "mark_reviewed", "dataset_id": dataset_id, "outcome_key": OUTCOME,
        "confidence": "needs_human", "justification": "the figure panel is ambiguous"})
    api.post(f"/api/runs/{run_id}/repool", headers=auth(token))

    queue = api.get(f"/api/runs/{run_id}/review", headers=auth(token)).json()["queue"]
    assert queue, "the cell we just flagged should be waiting"
    entry = next(e for e in queue if e["dataset_id"] == dataset_id)
    assert entry["study_label"] and str(entry["year"]) in entry["study_label"]
    assert entry["dataset_label"] and entry["outcome_label"]
    assert entry["dataset_id"] == dataset_id               # …and the id is still there

    evidence = api.get(f"/api/runs/{run_id}/evidence/{dataset_id}/{OUTCOME}",
                       headers=auth(token)).json()
    assert evidence["study_label"] and evidence["dataset_label"]


def test_the_monitor_paints_a_finished_run_from_the_manifest():
    """Item 2: a tab that slept through the run must still repaint it correctly."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderPapers(" in app_js
    assert "renderPapers(run.papers || [])" in app_js, "the grid is drawn from the manifest"
    assert "entry.cost.textContent = money(paper.cost_usd)" in app_js
    assert "state.live" in app_js, "live events stay as an overlay on top of it"
    assert "pollWhileRunning" in app_js


def test_the_spa_remembers_its_run_across_a_reload():
    """Item 3: reloading the page must not lose the run."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "window.localStorage" in app_js
    assert "canopy.tokens" in app_js and "canopy.current" in app_js
    assert "function rememberRun(" in app_js and "function forgetCurrentRun(" in app_js
    assert "rememberRun(body.run_id, body.token)" in app_js, "a new run is remembered"
    assert "attach(saved, savedToken)" in app_js, "and re-attached on load"
    assert 'if (screen === "new") { forgetCurrentRun(); }' in app_js


def test_the_monitor_says_when_the_run_has_already_finished():
    """Item 4: `Stop the run` on a finished run should say so, and offer the results."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="monitor-banner"' in page
    assert "stop.disabled = finished" in app_js
    assert "Open results" in app_js


def test_a_flag_can_be_accepted_in_one_click(cloned):
    """Item 5: the primary action is "accept as read", and it is a real override."""
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    rows = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()["rows"]
    posted = api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "mark_reviewed", "dataset_id": rows[0]["dataset_id"], "outcome_key": OUTCOME,
        "justification": "checked against the paper; the reading is right"})
    assert posted.status_code == 201
    assert posted.json()["override"]["confidence"] == "accept_with_note"

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '"Accept as read"' in app_js and "highlightRepool" in app_js
    assert app_js.index('"Accept as read"') < app_js.index('"Override value…"')
    assert '"Exclude dataset"' in app_js


# ============================================================================ review workflow
def test_an_override_is_appended_to_an_immutable_log(tmp_path):
    from canopy.server.overrides import append_override, read_overrides

    run = tmp_path / "run"
    run.mkdir()
    first = append_override(run, {"kind": "mark_reviewed", "dataset_id": "d1",
                                  "outcome_key": OUTCOME, "justification": "checked by hand"})
    before = (run / "overrides.jsonl").read_text()
    second = append_override(run, {"kind": "exclude_dataset", "dataset_id": "d1",
                                   "outcome_key": OUTCOME, "justification": "wrong task"})
    text = (run / "overrides.jsonl").read_text()

    assert text.startswith(before)                          # nothing was rewritten
    assert len(text.strip().splitlines()) == 2
    assert first["seq"] == 1 and second["seq"] == 2
    assert first["at"] and first["actor"]
    assert [o["kind"] for o in read_overrides(run)] == ["mark_reviewed", "exclude_dataset"]


def test_an_override_without_a_justification_is_refused(tmp_path):
    from canopy.server.overrides import OverrideRejected, append_override

    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(OverrideRejected):
        append_override(run, {"kind": "value", "dataset_id": "d1", "outcome_key": OUTCOME,
                              "group": "A", "mean": 1.0, "justification": ""})


def test_override_and_repool_change_the_pooled_estimate(cloned):
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]

    before = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()
    dataset_id = before["rows"][0]["dataset_id"]
    original = before["pooled"]["estimate"]

    posted = api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "value", "dataset_id": dataset_id, "outcome_key": OUTCOME, "group": "A",
        "mean": 12.0, "dispersion_value": 4.0, "dispersion_type": "SD", "n": 12,
        "justification": "the figure axis was misread; Table 2 prints 12.0 ± 4.0"})
    assert posted.status_code == 201, posted.text
    assert posted.json()["override"]["seq"] == 1

    repooled = api.post(f"/api/runs/{run_id}/repool", headers=auth(token))
    assert repooled.status_code == 200, repooled.text
    assert repooled.json()["applied"] == 1

    # …and it is not pooled yet. The run's score for this arm was computed on the value this answer
    # replaced, so the re-pool strikes it (`overrides._void_verification`) and the row is held by
    # the absence of verification until a person accepts a number nothing in the run checked. One
    # typed number inheriting the acceptance the replaced reading had earned is the hole the
    # grounding review measured at 31.9% of the pooled weight.
    held = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()
    assert "estimate" not in (held["pooled"] or {}), \
        "a row whose number nothing has verified was pooled"

    accepted = api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "mark_reviewed", "dataset_id": dataset_id, "outcome_key": OUTCOME,
        "confidence": "accept_with_note", "overrules": ["no_verification"],
        "justification": "I have read Table 2 and accept both arms as they now stand"})
    assert accepted.status_code == 201, accepted.text
    assert api.post(f"/api/runs/{run_id}/repool",
                    headers=auth(token)).json()["applied"] == 2

    after = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()
    changed = next(r for r in after["rows"] if r["dataset_id"] == dataset_id)
    assert changed["overridden"] is True
    assert changed["inputs"]["mean_a"] == 12.0
    assert after["pooled"]["estimate"] != pytest.approx(original)
    # …and the artefacts on disk were rewritten, not just the JSON response
    pooled = json.loads(api.get(f"/api/runs/{run_id}/files/results/{OUTCOME}/pooled.json",
                                headers=auth(token)).text)
    assert pooled["estimate"] == pytest.approx(after["pooled"]["estimate"])


def test_excluding_a_dataset_removes_it_from_the_analysis(cloned):
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    rows = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()["rows"]
    dataset_id = rows[0]["dataset_id"]

    api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "exclude_dataset", "dataset_id": dataset_id, "outcome_key": OUTCOME,
        "justification": "the participants also did experiment 2"})
    api.post(f"/api/runs/{run_id}/repool", headers=auth(token))

    after = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()
    assert dataset_id not in [r["dataset_id"] for r in after["rows"]]
    assert after["excluded"] and after["excluded"][0]["dataset_id"] == dataset_id
    # one row left: nothing can be pooled, and the old plot of two rows must not survive
    assert after["pooled"]["k"] == 0 and after["pooled"]["note"]
    assert after["forest"] == {}
    assert not (Path(api.app.state.runs_dir) / run_id / "results" / OUTCOME / "forest.svg").exists()
    assert api.get(f"/api/runs/{run_id}/files/results/{OUTCOME}/forest.svg",
                   headers=auth(token)).status_code == 404


def test_a_re_extract_request_is_recorded_and_reported_as_pending(cloned):
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    rows = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()["rows"]

    api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "re_extract", "dataset_id": rows[0]["dataset_id"], "outcome_key": OUTCOME,
        "group": "A", "hint": "read the right-hand panel of Figure 3, not the left",
        "justification": "the wrong panel was digitised"})
    repooled = api.post(f"/api/runs/{run_id}/repool", headers=auth(token)).json()

    assert repooled["pending"], "a re-extraction needs a model call, so repool cannot do it"
    assert "Figure 3" in repooled["pending"][0]["hint"]


def test_overrides_survive_a_resume(cloned, specs):
    """`canopy run --resume` rebuilds every artefact from the stage files — and re-applies the log.

    No server is involved in the resume: `run_pipeline` applies `overrides.jsonl` itself, which is
    what amendment I asks for, because a reviewer's decision must outlive the artefacts it changed.
    """
    from canopy.pipeline.run import run_pipeline

    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    run_dir = Path(api.app.state.runs_dir) / run_id

    rows = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()["rows"]
    dataset_id = rows[0]["dataset_id"]
    api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "value", "dataset_id": dataset_id, "outcome_key": OUTCOME, "group": "A",
        "mean": 12.0, "dispersion_value": 4.0, "dispersion_type": "SD", "n": 12,
        "justification": "Table 2 prints 12.0 ± 4.0"})
    # …and the acceptance the typed number needs before it pools: the run's score for that arm was
    # computed on the value this answer replaced, so the re-pool strikes it and the row is held
    # until a person says on the record that they accept a number nothing in the run verified.
    api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "mark_reviewed", "dataset_id": dataset_id, "outcome_key": OUTCOME,
        "confidence": "accept_with_note", "overrules": ["no_verification"],
        "justification": "I have read Table 2 and accept both arms as they now stand"})
    api.post(f"/api/runs/{run_id}/repool", headers=auth(token))
    overridden = api.get(f"/api/runs/{run_id}/results/{OUTCOME}",
                         headers=auth(token)).json()["pooled"]["estimate"]

    fresh = LLMClient(provider=FakeProvider([fake_router(specs)]), allow_live=True, cache_dir=None)
    events: list[dict[str, Any]] = []
    manifest = run_pipeline(run_dir / "uploads", run_dir / "protocol.yaml", run_dir,
                            client=fresh, concurrency=1, resume=True, progress=events.append)
    assert manifest.n_llm_calls == 0                        # a resume spends nothing

    after = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()
    assert after["pooled"]["estimate"] == pytest.approx(overridden), \
        "the resume rebuilt the artefacts and the override was not re-applied"
    changed = next(r for r in after["rows"] if r["dataset_id"] == dataset_id)
    assert changed["overridden"] is True and changed["inputs"]["mean_a"] == 12.0
    assert any(e["stage"] == "review" and "override" in e["message"] for e in events)
    # both records: the typed value and the acceptance it needed before the row could pool
    assert json.loads((run_dir / "overrides_applied.json").read_text())["applied"] == 2


def test_a_repool_writes_every_artefact_a_run_writes(cloned, monkeypatch):
    """A reviewed run directory must be the same *kind* of thing a finished one is."""
    from canopy.pipeline import overrides as pipeline_overrides

    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    run_dir = Path(api.app.state.runs_dir) / run_id
    rows = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()["rows"]
    dataset_id = rows[0]["dataset_id"]
    before_prisma = json.loads((run_dir / "prisma.json").read_text())
    assert before_prisma["included_datasets"] == 2

    # `all_rows` has no visible effect until a paper contributes two rows, so it is pinned at the
    # seam: without it the per-outcome table silently loses the rows an aggregation replaced
    seen: dict[str, Any] = {}
    real = pipeline_overrides.write_outcome_outputs

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.setdefault("all_rows", kwargs.get("all_rows"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline_overrides, "write_outcome_outputs", spy)

    api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "exclude_dataset", "dataset_id": dataset_id, "outcome_key": OUTCOME,
        "justification": "the participants also did experiment 2"})
    api.post(f"/api/runs/{run_id}/repool", headers=auth(token))

    assert seen["all_rows"] is not None, "the per-outcome table was written without all_rows"

    # the run-wide table is rebuilt from the reviewed rows
    table = (run_dir / "results" / "extraction_table_all.csv").read_text(encoding="utf-8")
    assert dataset_id not in table
    assert any(r["dataset_id"] in table for r in rows[1:])

    # the PRISMA flow is re-counted, and still adds up
    prisma = json.loads((run_dir / "prisma.json").read_text())
    assert prisma["included_datasets"] == 1
    assert prisma["datasets_excluded"] == before_prisma["datasets_excluded"] + 1
    assert prisma["files"] == before_prisma["files"]        # screening cannot change
    assert prisma.get("consistent", True) is True
    assert "human_override" in prisma["exclusion_reasons"]

    # and the report keeps the provenance it had
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "<h2>Provenance</h2>" in report
    assert "provenance.json" in report


# ============================================================================ cancel
def test_cancel_stops_the_run_and_still_writes_a_manifest(tmp_path, specs):
    """Cancellation is checked between papers and stages; what finished is kept."""
    import threading

    from canopy.server.app import create_app

    gate = threading.Event()

    def slow_client_factory(run_dir: Path | None = None, **_: Any) -> LLMClient:
        router = fake_router(specs)

        def wait_then_answer(request):
            gate.wait(timeout=60)
            return router(request)

        return LLMClient(provider=FakeProvider([wait_then_answer]), allow_live=True,
                         cache_dir=None)

    api = TestClient(create_app(runs_dir=tmp_path / "runs", client_factory=slow_client_factory))
    body = create_run(api, options={"concurrency": 1}).json()
    run_id, token = body["run_id"], body["token"]

    deadline = time.monotonic() + 60
    while api.get(f"/api/runs/{run_id}", headers=auth(token)).json()["status"] == "queued":
        assert time.monotonic() < deadline, "the job never started"
        time.sleep(0.02)

    cancelled = api.post(f"/api/runs/{run_id}/cancel", headers=auth(token))
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] in ("cancelling", "cancelled")
    gate.set()

    final = wait_done(api, run_id, token, timeout=120)
    assert final["status"] == "cancelled"
    manifest = json.loads((Path(api.app.state.runs_dir) / run_id / "manifest.json").read_text())
    assert manifest["papers"], "a cancelled run still writes what it knows"
    assert any(p["status"] == "cancelled" for p in manifest["papers"]), \
        "a paper stopped between stages is cancelled, not failed"
    assert all(p["status"] in ("cancelled", "resolved", "excluded") for p in manifest["papers"])


# ============================================================================ options + limits
@pytest.mark.parametrize("options, wrong", [
    ({"budget_usd": "lots"}, "budget_usd"),
    ({"budget_usd": -3}, "budget_usd"),
    ({"max_usd_per_paper": "1.2.3"}, "max_usd_per_paper"),
    ({"max_papers": "all of them"}, "max_papers"),
    ({"max_papers": 0}, "max_papers"),
    ({"concurrency": "four"}, "concurrency"),
    ({"concurrency": 999}, "concurrency"),
    ({"profile": "not_a_profile"}, "profile"),
    ({"models": {"wizard": "claude-opus-5"}}, "models"),
    ({"models": "claude-opus-5"}, "models"),
    ({"totally_unknown": 1}, "totally_unknown"),
])
def test_malformed_options_are_a_readable_422_not_a_500(api, options, wrong):
    """A browser sends what a person typed; the answer says which field is wrong."""
    files = [("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf")),
             ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))]
    response = api.post("/api/runs", files=files, data={"options": json.dumps(options)})
    assert response.status_code == 422, response.text
    assert wrong in response.json()["detail"]


def test_a_protocol_with_a_repeated_key_is_a_422_that_names_the_key(api):
    """The path F11 came in on: a form-built protocol with two `digitize:` blocks silently lost
    the first one. The CLI refuses it; the server has to as well, and readably."""
    body = PROTOCOL.read_text() + "\ndigitize:\n  readouts_min: 2\ndigitize:\n  readouts_max: 3\n"
    files = [("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf")),
             ("protocol", ("protocol.yaml", body.encode(), "text/yaml"))]
    response = api.post("/api/runs", files=files, data={"options": "{}"})
    assert response.status_code == 422, response.text
    assert "duplicate key 'digitize'" in response.json()["detail"]


def test_options_that_are_not_even_json_are_refused(api):
    files = [("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf")),
             ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))]
    for bad in ("not json", "[1, 2, 3]", "null"):
        response = api.post("/api/runs", files=files, data={"options": bad})
        assert response.status_code == 422 and "JSON object" in response.json()["detail"]


def test_an_untouched_number_field_means_no_cap(api):
    """The form sends `""` for a box nobody typed in — that is "no cap", not zero."""
    created = create_run(api, options={"budget_usd": "", "max_papers": "", "start": False})
    assert created.status_code == 201, created.text
    body = api.get(f"/api/runs/{created.json()['run_id']}",
                   headers=auth(created.json()["token"])).json()
    assert body["options"]["budget_usd"] is None and body["options"]["max_papers"] is None


def test_a_giant_protocol_is_refused(api):
    files = [("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf")),
             ("protocol", ("protocol.yaml", b"title: x\n" + b"# padding\n" * 200_000,
                           "text/yaml"))]
    response = api.post("/api/runs", files=files, data={"options": "{}"})
    assert response.status_code == 413 and "protocol" in response.json()["detail"]


def test_uploads_are_handled_one_file_at_a_time(api, monkeypatch, tmp_path):
    """Item 1 of the review: never hold the whole folder in memory.

    Asserted at the seam, because "how much was resident" is not observable from outside: each
    file must be streamed to disk and ingested before the next one is read off the wire.
    """
    from canopy.server import app as server_app

    order: list[str] = []
    real_stream = server_app.stream_upload

    def stream(source, dest, name, **kwargs):
        order.append(f"stream {name}")
        return real_stream(source, dest, name, **kwargs)

    def probe(path, timeout=0.0):
        order.append(f"probe {Path(path).name[:8]}")
        return {"ok": True, "n_pages": 1, "error": ""}

    monkeypatch.setattr(server_app, "stream_upload", stream)
    monkeypatch.setattr(server_app, "probe_pdf", probe)

    created = create_run(api, pdfs=PDFS, options={"start": False})
    assert created.status_code == 201, created.text
    assert [step.split()[0] for step in order] == ["stream", "probe", "stream", "probe"]


def test_a_second_run_is_refused_while_one_is_going(tmp_path, specs):
    """One laptop, one API budget: the number of concurrent runs is capped.

    The gate is opened in a `finally` because a run left blocked would be joined at interpreter
    exit (`ThreadPoolExecutor` threads are not daemons), turning a failed assertion into a
    four-minute test.
    """
    import threading

    from canopy.server.app import create_app

    gate = threading.Event()

    def slow(run_dir: Path | None = None, **_: Any) -> LLMClient:
        router = fake_router(specs)

        def wait_then_answer(request):
            gate.wait(timeout=60)
            return router(request)

        return LLMClient(provider=FakeProvider([wait_then_answer]), allow_live=True,
                         cache_dir=None)

    api = TestClient(create_app(runs_dir=tmp_path / "runs", client_factory=slow,
                                max_active_runs=1))
    assert api.get("/api/settings").json()["max_active_runs"] == 1
    first = create_run(api, options={"concurrency": 1}).json()
    try:
        deadline = time.monotonic() + 60
        while api.get(f"/api/runs/{first['run_id']}",
                      headers=auth(first["token"])).json()["status"] == "queued":
            assert time.monotonic() < deadline, "the first run never started"
            time.sleep(0.02)

        refused = create_run(api, options={"concurrency": 1})
        assert refused.status_code == 429
        assert "CANOPY_MAX_ACTIVE_RUNS" in refused.json()["detail"]

        held = create_run(api, options={"concurrency": 1, "start": False}).json()
        again = api.post(f"/api/runs/{held['run_id']}/start", headers=auth(held["token"]))
        assert again.status_code == 429                    # …and the same gate holds on /start
        assert "CANOPY_MAX_ACTIVE_RUNS" in again.json()["detail"]
        assert api.post(f"/api/runs/{first['run_id']}/start",
                        headers=auth(first["token"])).status_code == 409   # already going
    finally:
        api.post(f"/api/runs/{first['run_id']}/cancel", headers=auth(first["token"]))
        gate.set()
        wait_done(api, first["run_id"], first["token"], timeout=180)


def test_the_event_stream_of_a_run_nobody_started_says_so_and_closes(api):
    created = create_run(api, options={"start": False}).json()
    with api.stream("GET", f"/api/runs/{created['run_id']}/events",
                    headers=auth(created["token"])) as stream:
        lines = list(stream.iter_lines())
    names = [line[len("event: "):] for line in lines if line.startswith("event: ")]
    payloads = [json.loads(line[len("data: "):]) for line in lines if line.startswith("data: ")]
    assert names == ["end"]
    assert payloads[-1]["status"] == "not_started"
    assert "not been started" in payloads[-1]["message"]


# ============================================================================ protocols
def test_the_example_protocols_are_offered_with_their_yaml(api):
    body = api.get("/api/protocols/examples").json()
    names = [e["name"] for e in body["examples"]]
    assert "aging_sensorimotor_adaptation" in names
    example = next(e for e in body["examples"] if e["name"] == "aging_sensorimotor_adaptation")
    assert example["title"] and "outcomes:" in example["yaml"]
    assert "metafor" in body["profiles"] and body["skeleton"]


def test_a_protocol_is_drafted_from_one_sentence(make_app):
    """Amendment I's "draft from one sentence" — the model fills a protocol skeleton."""
    from canopy.protocol import load_protocol

    drafted = {
        "title": "Mindfulness training and working memory",
        "research_question": "Does mindfulness training improve working memory in adults?",
        "group_a": {"key": "A", "label": "Mindfulness training",
                    "definition": "Participants who completed a mindfulness programme.",
                    "synonyms": ["meditation group", "MBSR"]},
        "group_b": {"key": "B", "label": "Control",
                    "definition": "Participants who did not train.",
                    "synonyms": ["waitlist", "passive control"]},
        "outcomes": [{"key": "working_memory", "label": "Working memory",
                      "definition": "Any validated working-memory span score.",
                      "measurement_window": "Immediately after the intervention.",
                      "higher_is_better_hint": "A higher span is better.",
                      "positive_direction_label": "Better with training",
                      "negative_direction_label": "Worse with training",
                      "units_hint": "span score"}],
        "eligibility": ["Randomised or quasi-randomised design.",
                        "The study is written in English."],
        "dataset_rules": ["Use the first post-intervention timepoint."],
        "moderators": ["training_hours"],
        "notes": "",
    }

    def client_factory(run_dir: Path | None = None, **_: Any) -> LLMClient:
        return LLMClient(provider=FakeProvider([drafted]), allow_live=True, cache_dir=None)

    api = make_app(client_factory=client_factory)
    response = api.post("/api/protocols/draft",
                        json={"sentence": "Does mindfulness training improve working memory?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["protocol"]["title"] == "Mindfulness training and working memory"
    assert "working_memory" in body["yaml"]

    written = Path(api.app.state.runs_dir) / "draft.yaml"
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(body["yaml"], encoding="utf-8")
    assert load_protocol(written).outcomes[0].key == "working_memory"   # it really loads


def test_an_empty_draft_request_is_refused(api):
    assert api.post("/api/protocols/draft", json={"sentence": "  "}).status_code == 422


def test_a_dry_run_maps_a_couple_of_papers_without_running_the_pipeline(api):
    body = create_run(api, options={"start": False}).json()
    run_id, token = body["run_id"], body["token"]
    assert api.get(f"/api/runs/{run_id}", headers=auth(token)).json()["status"] == "created"

    started = api.post(f"/api/runs/{run_id}/dry-run", headers=auth(token),
                       json={"max_papers": 2, "wait_seconds": 120})
    assert started.status_code == 200, started.text
    maps = started.json()["maps"]
    assert 1 <= len(maps) <= 2
    assert maps[0]["datasets"] and maps[0]["citation"]["year"]
    assert maps[0]["eligible"] is True
    assert maps[0]["datasets"][0]["group_a"]["label"]

    fetched = api.get(f"/api/runs/{run_id}/dry-run", headers=auth(token)).json()
    assert fetched["status"] == "done" and len(fetched["maps"]) == len(maps)
    # a dry run maps only: no effect sizes, no report
    assert not (Path(api.app.state.runs_dir) / run_id / "report.html").exists()


# ============================================================================ security baseline
def test_the_api_key_is_never_echoed(monkeypatch, make_app, finished):
    planted = "sk-ant-THIS-MUST-NOT-APPEAR"
    monkeypatch.setenv("ANTHROPIC_API_KEY", planted)
    api, run_id, token = finished["api"], finished["run_id"], finished["token"]

    settings = api.get("/api/settings").json()
    assert settings["api_key_configured"] is True
    assert planted not in json.dumps(settings)

    for url in (f"/api/runs/{run_id}", f"/api/runs/{run_id}/results/{OUTCOME}",
                f"/api/runs/{run_id}/review", "/api/protocols/examples", "/api/runs"):
        response = api.get(url, headers=auth(token))
        assert planted not in response.text, url


def test_the_settings_endpoint_reports_only_whether_a_key_is_configured(monkeypatch, api):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = api.get("/api/settings").json()
    assert body["api_key_configured"] is False
    assert not any("key" in str(v).lower() and str(v).startswith("sk-") for v in body.values())


def test_an_upload_that_is_not_a_pdf_is_refused_by_the_api(api):
    files = [("files", ("evil.pdf", b"MZ\x90\x00 this is a windows binary", "application/pdf")),
             ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))]
    response = api.post("/api/runs", files=files, data={"options": "{}"})
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_an_oversize_upload_is_refused_by_the_api(make_app):
    api = make_app(max_upload_mb=0.001)
    response = create_run(api)
    assert response.status_code == 413


def test_a_run_needs_at_least_one_paper_and_a_protocol(api):
    assert api.post("/api/runs", files=[("protocol", ("p.yaml", PROTOCOL.read_bytes(),
                                                      "text/yaml"))],
                    data={"options": "{}"}).status_code == 422
    assert api.post("/api/runs",
                    files=[("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf"))],
                    data={"options": "{}"}).status_code == 422


def test_a_broken_protocol_is_refused_with_a_readable_message(api):
    files = [("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf")),
             ("protocol", ("protocol.yaml", b"title: nothing else\n", "text/yaml"))]
    response = api.post("/api/runs", files=files, data={"options": "{}"})
    assert response.status_code == 422
    assert "group_a" in response.json()["detail"] or "outcomes" in response.json()["detail"]


def test_a_page_on_another_site_cannot_start_a_run(api):
    """`POST /api/runs` mints the token, so it cannot require one — a browser must say it is ours.

    A multipart POST is a "simple request": no CORS preflight stands between a page the user is
    visiting and this server. Without this check such a page could start a run with the user's key.
    """
    files = [("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf")),
             ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))]
    data = {"options": "{}"}

    assert api.post("/api/runs", files=files, data=data,
                    headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert api.post("/api/runs", files=files, data=data,
                    headers={"Origin": "http://evil.example"}).status_code == 403
    assert api.post("/api/protocols/draft", json={"sentence": "hello"},
                    headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    # the browser's own page, and every non-browser client, are unaffected
    assert api.get("/api/settings", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200
    assert api.post("/api/runs", files=files, data=data,
                    headers={"Sec-Fetch-Site": "same-origin"}).status_code == 201


def test_a_foreign_host_header_is_refused_when_the_server_is_bound_to_loopback(make_app):
    api = make_app(allowed_hosts=["localhost", "127.0.0.1"])
    assert api.get("/api/settings", headers={"Host": "evil.example.com"}).status_code == 400
    assert api.get("/api/settings", headers={"Host": "127.0.0.1:8000"}).status_code == 200


def test_a_server_bound_to_the_world_does_not_hand_out_tokens(make_app, finished):
    """`--host 0.0.0.0` means the run list can be read by anyone; the tokens must not be."""
    from canopy.server.app import create_app

    public = TestClient(create_app(runs_dir=finished["runs"], loopback_only=False))
    listed = public.get("/api/runs").json()
    assert listed["loopback_only"] is False
    assert all("token" not in row for row in listed["runs"])
    assert public.get(f"/api/runs/{finished['run_id']}").status_code == 401


def test_a_run_can_be_created_now_and_started_later(api):
    """The dry-run-first path: upload, look at the maps, then commit to the run."""
    created = create_run(api, options={"start": False, "concurrency": 1}).json()
    run_id, token = created["run_id"], created["token"]
    assert api.get(f"/api/runs/{run_id}", headers=auth(token)).json()["status"] == "created"

    started = api.post(f"/api/runs/{run_id}/start", headers=auth(token))
    assert started.status_code == 200
    final = wait_done(api, run_id, token)
    assert final["status"] == "done"
    assert (Path(api.app.state.runs_dir) / run_id / "manifest.json").exists()


def test_the_spa_never_writes_untrusted_text_as_html():
    """Amendment I: escape all PDF/LLM text — the SPA uses textContent, never innerHTML."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in app_js
    assert "outerHTML" not in app_js
    assert "insertAdjacentHTML" not in app_js
    assert "document.write" not in app_js
    assert "eval(" not in app_js


def test_the_results_panes_are_reachable_with_a_keyboard():
    """Each pane is a tab panel that names its tab, and the drawer hands focus back."""
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    for name in ("forest", "table", "flags", "figures", "downloads"):
        assert f'aria-controls="pane-{name}"' in page
        assert f'id="pane-{name}" role="tabpanel"' in page
        assert f'aria-labelledby="tab-{name}"' in page
    assert 'role="dialog"' in page
    assert "state.returnFocus = document.activeElement" in app_js
    assert "state.returnFocus.focus()" in app_js


def test_hidden_really_hides_every_element_the_spa_toggles():
    """`show(node, false)` sets the `hidden` attribute — which a class that sets `display` would
    otherwise override, leaving the evidence drawer on screen after the close button was pressed."""
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css), \
        "the stylesheet must force [hidden] to win over any display rule"
    toggled = set(re.findall(r'show\(\$\("([a-z0-9-]+)"\)', app_js))
    assert "drawer" in toggled and "scrim" in toggled
    for element_id in toggled:
        assert f'id="{element_id}"' in page, element_id


def test_the_spa_makes_no_external_requests():
    """Local-first: nothing is fetched from a CDN, and there is no build step."""
    for name in ("index.html", "app.js", "styles.css"):
        text = (STATIC / name).read_text(encoding="utf-8")
        assert "http://" not in text.replace("http://www.w3.org", "")
        assert "https://" not in text.replace("https://www.w3.org", "")


def test_the_index_page_is_served(api):
    response = api.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>" in response.text
    assert api.get("/static/app.js").status_code == 200
    assert api.get("/static/styles.css").status_code == 200


# ============================================================================ questions
def test_a_run_offers_its_held_cells_as_questions_and_an_answer_repools(cloned):
    """Ask → answer → the answer is an override naming the question, and the pool moved."""
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    body = api.get(f"/api/runs/{run_id}/questions", headers=auth(token)).json()
    assert body["run_id"] == run_id and isinstance(body["questions"], list)
    questions = body["questions"]
    if not questions:
        pytest.skip("this fake run held nothing; the module is covered in test_questions.py")
    q = questions[0]
    for key in ("number", "kind", "prompt", "options", "why", "answer_writes", "image"):
        assert key in q
    if q["image"].get("path"):
        assert q["image"]["url"].startswith("/api/runs/")

    before = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()
    answer = ({"id": q["id"], "option": q["options"][0]["key"],
               "option_fingerprint": q["options"][0]["fingerprint"],
               "note": "checked the figure"}
              if q["options"] and q["options"][0].get("mean") is not None
              else {"id": q["id"], "mean": 12.0, "dispersion_value": 4.0, "dispersion_type": "SD",
                    "n": 12, "note": "typed from the table"})
    response = api.post(f"/api/runs/{run_id}/questions/{q['number']}/answer",
                        headers=auth(token), json=answer)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["ok"] and payload["override"]["justification"].startswith(
        f"answered question #{q['number']}")
    log = (Path(api.app.state.runs_dir) / run_id / "overrides.jsonl").read_text()
    assert f"answered question #{q['number']}" in log
    again = api.get(f"/api/runs/{run_id}/questions", headers=auth(token)).json()
    mine = next(x for x in again["questions"] if x["id"] == q["id"])
    assert mine["answered"] is True and again["n_open"] == body["n_open"] - 1
    # answering a value question is a value override, so the pooled estimate may move; either
    # way the run was re-pooled and still reads as a finished run
    after = api.get(f"/api/runs/{run_id}/results/{OUTCOME}", headers=auth(token)).json()
    assert "pooled" in after and (Path(api.app.state.runs_dir) / run_id / "prisma.json").exists()


def test_the_server_records_a_direction_for_a_measure_and_refuses_one_without_a_direction(cloned):
    """C4's `orientation` decision reaches the log through the ordinary override endpoint.

    The direction of a measure is not a cell value — it is decided once per (paper, outcome,
    measure) — so before C4 there was no kind a reviewer could post it as, and `resolve` refused
    every such row with `orientation_unresolved`.
    """
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    queue = api.get(f"/api/runs/{run_id}/review", headers=auth(token)).json()["queue"]
    paper_id = next((e.get("paper_id") for e in queue if e.get("paper_id")), None) \
        or api.get(f"/api/runs/{run_id}", headers=auth(token)).json()["papers"][0]["paper_id"]

    posted = api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "orientation", "paper_id": paper_id, "outcome_key": OUTCOME, "group": "A",
        "higher_is_better": False, "measure_name": "mean direction error",
        "quote": "“a smaller aftereffect means less adaptation”",
        "justification": "the outcome is an error measure, so smaller is more adaptation"})
    assert posted.status_code == 201, posted.text
    record = posted.json()["override"]
    assert record["kind"] == "orientation" and record["higher_is_better"] is False
    assert record["group"] is None, "a direction belongs to the measure, not to one group's cell"

    refused = api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "orientation", "paper_id": paper_id, "outcome_key": OUTCOME,
        "measure_name": "mean direction error",
        "justification": "I am not sure which way round it goes"})
    assert refused.status_code == 422
    assert "higher_is_better" in refused.json()["detail"]

    # H1/N4: a blank measure is recordable — an outcome whose map never named one has a blank
    # measure, and refusing the field would leave its direction question unanswerable. The scope is
    # enforced where it can be seen: the re-pool refuses a direction that lands on more than one
    # measure (`test_a_direction_with_no_measure_named_is_refused_instead_of_re_signing_the_whole_paper`).
    unscoped = api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json={
        "kind": "orientation", "paper_id": paper_id, "outcome_key": OUTCOME,
        "higher_is_better": True, "measure_name": "",
        "justification": "a larger value is more adaptation"})
    assert unscoped.status_code == 201 and unscoped.json()["override"]["measure_name"] == ""


def test_a_question_the_map_could_not_settle_is_asked_and_answered_through_the_server(cloned):
    """C6/C7: a map question blocks an extraction, so it never reaches the review queue — and
    before this it reached no page either. Here it is asked, answered, and the answer comes back
    as the record the extract stage reads, with the honest news that acting on it needs a resume.
    """
    from canopy.models import MapQuestion

    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    run_dir = Path(api.app.state.runs_dir) / run_id
    path = next(iter(sorted(run_dir.glob("papers/*/map.json"))))
    payload = json.loads(path.read_text())
    dataset_id = payload["study"]["datasets"][0]["dataset_id"]
    payload["study"]["open_questions"] = [MapQuestion(
        kind="include_dataset", dataset_id=dataset_id,
        question="Only one of the two mapping agents proposed this dataset. Is it in the review?",
        options=["include it", "exclude it"],
        quotes=["these participants never performed the adaptation task"]).model_dump(mode="json")]
    path.write_text(json.dumps(payload), encoding="utf-8")

    body = api.get(f"/api/runs/{run_id}/questions", headers=auth(token)).json()
    question = next(q for q in body["questions"] if q["kind"] == "include_dataset")
    assert question["dataset_id"] == dataset_id and question["answer_writes"] == "include_dataset"
    assert [o["key"] for o in question["options"]][0] == "include"
    assert "never performed the adaptation task" in question["why"]

    include = next(o for o in question["options"] if o["key"] == "include")
    posted = api.post(f"/api/runs/{run_id}/questions/{question['number']}/answer",
                      headers=auth(token),
                      json={"id": question["id"], "option": "include",
                            "option_fingerprint": include["fingerprint"],
                            "note": "a separate sample"})
    assert posted.status_code == 201, posted.text
    record = posted.json()["override"]
    assert record["kind"] == "include_dataset" and record["decision"] == "include"
    assert record["note"] == "a separate sample" and record["group"] is None
    pending = posted.json()["repool"]["pending"]
    assert pending and pending[0]["why"] == ("extraction was never bought for this; re-run with "
                                             "--resume to extract it")
    # and it now reads as answered rather than being asked again
    again = api.get(f"/api/runs/{run_id}/questions", headers=auth(token)).json()["questions"]
    assert next(q for q in again if q["kind"] == "include_dataset")["answered"] is True


def test_the_server_refuses_a_map_answer_that_names_no_decision_or_no_winning_measure(cloned):
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    paper_id = api.get(f"/api/runs/{run_id}", headers=auth(token)).json()["papers"][0]["paper_id"]
    for body, wanted in (
            ({"kind": "include_dataset", "paper_id": paper_id, "dataset_id": "x:d2",
              "decision": "maybe", "note": "I am not sure"}, "decision"),
            ({"kind": "include_dataset", "paper_id": paper_id, "dataset_id": "",
              "decision": "exclude", "note": "the controls never adapted"}, "dataset_id"),
            ({"kind": "which_measure", "paper_id": paper_id, "dataset_id": "x:d1",
              "outcome_key": OUTCOME, "note": "one of them"}, "winning")):
        refused = api.post(f"/api/runs/{run_id}/overrides", headers=auth(token), json=body)
        assert refused.status_code == 422, refused.text
        assert wanted in refused.json()["detail"]


def _with_a_map_question(cloned) -> dict:
    """A run holding one question whose number is stable enough to aim two POSTs at."""
    from canopy.models import MapQuestion

    run_dir = Path(cloned["api"].app.state.runs_dir) / cloned["run_id"]
    path = next(iter(sorted(run_dir.glob("papers/*/map.json"))))
    payload = json.loads(path.read_text())
    payload["study"]["open_questions"] = [MapQuestion(
        kind="include_dataset", dataset_id=payload["study"]["datasets"][0]["dataset_id"],
        question="Only one mapping agent proposed this dataset. Is it in the review?",
        options=["include it", "exclude it"]).model_dump(mode="json")]
    path.write_text(json.dumps(payload), encoding="utf-8")
    body = cloned["api"].get(f"/api/runs/{cloned['run_id']}/questions",
                             headers=auth(cloned["token"])).json()
    return body


def test_an_answer_must_name_the_question_it_answers_and_a_stale_number_is_refused(cloned):
    """H3. The number is a position in a list that is rebuilt on every request and shortens as
    answers land, so a retried POST to `/questions/2/answer` decided a *different* cell — the
    replay in the review produced two identical POSTs writing means to two different groups, and
    pooled a row nobody was shown. The answer now names the question, and a number that has since
    moved is a 409 rather than a decision.
    """
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    body = _with_a_map_question(cloned)
    question = next(q for q in body["questions"] if q["kind"] == "include_dataset")
    url = f"/api/runs/{run_id}/questions/{question['number']}/answer"

    fp = next(o for o in question["options"] if o["key"] == "include")["fingerprint"]
    nameless = api.post(url, headers=auth(token), json={"option": "include", "note": "keep it"})
    assert nameless.status_code == 422 and "id" in nameless.json()["detail"]

    stale = api.post(url, headers=auth(token),
                     json={"id": "d9|late_adaptation|A|which_axis", "option": "include",
                           "option_fingerprint": fp, "note": "keep it"})
    assert stale.status_code == 409, stale.text
    assert question["id"] in stale.json()["detail"]

    # N6: the id names the question, not the option — an answer echoes what it was shown
    unechoed = api.post(url, headers=auth(token),
                        json={"id": question["id"], "option": "include", "note": "keep it"})
    assert unechoed.status_code == 422 and "fingerprint" in unechoed.json()["detail"]
    moved = api.post(url, headers=auth(token),
                     json={"id": question["id"], "option": "include",
                           "option_fingerprint": "0" * 12, "note": "keep it"})
    assert moved.status_code == 409 and "not the option you were shown" in moved.json()["detail"]

    named = api.post(url, headers=auth(token),
                     json={"id": question["id"], "option": "include",
                           "option_fingerprint": fp, "note": "keep it"})
    assert named.status_code == 201, named.text


@pytest.fixture()
def nine_served(tmp_path) -> dict[str, Any]:
    """A server over the RECORDED nine-paper run — the only run in the suite that really holds
    cells a card folds. The fake pipeline run behind `cloned` holds none, so the endpoint's plural
    contract cannot be exercised there at all."""
    from canopy.server.app import create_app
    from tests.helpers import nine

    runs = tmp_path / "nine-runs"
    runs.mkdir()
    run_dir = nine.copy_to(runs)
    token = "n" * 43
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_dir.name, "token": token, "title": "nine", "created_at": "2026-08-18",
        "status": "done", "options": {}, "n_files": 9, "cost_usd": 0.0, "error": "",
        "started_at": "2026-08-18", "finished_at": "2026-08-18", "kind": "run"}), encoding="utf-8")
    return {"api": TestClient(create_app(runs_dir=runs)), "run_id": run_dir.name, "token": token}


def _pair(served: dict[str, Any]) -> dict[str, Any]:
    body = served["api"].get(f"/api/runs/{served['run_id']}/questions",
                             headers=auth(served["token"])).json()
    return next(q for q in body["questions"] if q["kind"] == "pair")


def test_answer_endpoint_appends_one_override_per_group_and_returns_all(nine_served):
    """§C1: one decision, one POST — and one override per cell the decision names. A client that
    only reads `override` still sees the first; one that records what happened reads `overrides`."""
    api, run_id, token = nine_served["api"], nine_served["run_id"], nine_served["token"]
    card = _pair(nine_served)
    option = card["options"][0]
    posted = api.post(f"/api/runs/{run_id}/questions/{card['number']}/answer",
                      headers=auth(token),
                      json={"id": card["id"], "option": option["key"],
                            "option_fingerprint": option["fingerprint"],
                            "note": "read off the figure"})
    assert posted.status_code == 201, posted.text
    payload = posted.json()
    assert len(payload["overrides"]) == 2
    assert [r["group"] for r in payload["overrides"]] == ["A", "B"]
    assert payload["override"] == payload["overrides"][0]
    assert all(r["question_id"] == card["id"] for r in payload["overrides"])
    log = [json.loads(line) for line in
           (Path(api.app.state.runs_dir) / run_id / "overrides.jsonl").read_text().splitlines()
           if line.strip()]
    assert [r["group"] for r in log[-2:]] == ["A", "B"]


def test_stale_option_fingerprint_still_409s(nine_served):
    """The echo check has to survive the fold: a folded option's key is positional twice over, so
    a key alone means even less than it did before."""
    api, run_id, token = nine_served["api"], nine_served["run_id"], nine_served["token"]
    card = _pair(nine_served)
    url = f"/api/runs/{run_id}/questions/{card['number']}/answer"
    option = card["options"][0]
    unechoed = api.post(url, headers=auth(token), json={"id": card["id"], "option": option["key"]})
    assert unechoed.status_code == 422 and "fingerprint" in unechoed.json()["detail"]
    stale = api.post(url, headers=auth(token),
                     json={"id": card["id"], "option": option["key"],
                           "option_fingerprint": "0" * 12})
    assert stale.status_code == 409 and "not the option you were shown" in stale.json()["detail"]
    named = api.post(url, headers=auth(token),
                     json={"id": card["id"], "option": option["key"],
                           "option_fingerprint": option["fingerprint"], "note": "the figure"})
    assert named.status_code == 201, named.text


def test_a_cell_card_is_answered_one_slot_at_a_time_through_the_endpoint(nine_served):
    """§C1, second fold: a `cell` card carries its slots' own options, and the POST names the slot
    each pick is for (`slots`). The same two guards as a card-level pick — the slot must be one
    the card answers on its own, and a pick must echo the fingerprint it was shown under — and
    one override per slot answered, on that slot's group."""
    api, run_id, token = nine_served["api"], nine_served["run_id"], nine_served["token"]
    body = api.get(f"/api/runs/{run_id}/questions", headers=auth(token)).json()
    card = next(q for q in body["questions"] if q["kind"] == "cell")
    assert card["slot_answers"] is True and card["options"] == []
    slots = [s for s in card["slots"] if s["answerable"]]
    assert len(slots) >= 2
    url = f"/api/runs/{run_id}/questions/{card['number']}/answer"
    first, second = slots[0], slots[1]
    option = first["options"][0]
    unknown = api.post(url, headers=auth(token), json={"id": card["id"], "slots": [
        {"slot": "nope|x||y", "option": option["key"], "option_fingerprint": option["fingerprint"]}]})
    assert unknown.status_code == 409 and "no slot" in unknown.json()["detail"]
    unechoed = api.post(url, headers=auth(token), json={"id": card["id"], "slots": [
        {"slot": first["member_id"], "option": option["key"]}]})
    assert unechoed.status_code == 422 and "fingerprint" in unechoed.json()["detail"]
    stale = api.post(url, headers=auth(token), json={"id": card["id"], "slots": [
        {"slot": first["member_id"], "option": option["key"], "option_fingerprint": "0" * 12}]})
    assert stale.status_code == 409 and "not the option you were shown" in stale.json()["detail"]
    twice = api.post(url, headers=auth(token), json={"id": card["id"], "slots": [
        {"slot": first["member_id"], "option": option["key"],
         "option_fingerprint": option["fingerprint"]},
        {"slot": first["member_id"], "option": option["key"],
         "option_fingerprint": option["fingerprint"]}]})
    assert twice.status_code == 422 and "twice" in twice.json()["detail"]
    empty = api.post(url, headers=auth(token), json={"id": card["id"], "slots": [
        {"slot": first["member_id"]}]})
    assert empty.status_code == 422 and "no answer" in empty.json()["detail"]
    posted = api.post(url, headers=auth(token), json={"id": card["id"], "note": "read it", "slots": [
        {"slot": first["member_id"], "option": option["key"],
         "option_fingerprint": option["fingerprint"]},
        {"slot": second["member_id"], "option": second["options"][0]["key"],
         "option_fingerprint": second["options"][0]["fingerprint"]}]})
    assert posted.status_code == 201, posted.text
    records = posted.json()["overrides"]
    assert len(records) == 2
    assert [r["group"] for r in records] == [first["group"], second["group"]]
    assert [r["question_id"] for r in records] == [first["member_id"], second["member_id"]]


def test_the_badge_counts_open_questions_and_reports_the_ones_waiting_for_a_re_run(cloned):
    """M7. `n_open` counted "not answered", so a run whose every remaining answer was waiting for
    a resume reported "No open questions" while nothing had been applied."""
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    before = _with_a_map_question(cloned)
    question = next(q for q in before["questions"] if q["kind"] == "include_dataset")
    assert before["n_open"] >= 1 and before["n_pending"] == 0

    fp = next(o for o in question["options"] if o["key"] == "include")["fingerprint"]
    answered = api.post(f"/api/runs/{run_id}/questions/{question['number']}/answer",
                        headers=auth(token),
                        json={"id": question["id"], "option": "include",
                              "option_fingerprint": fp, "note": "keep it"}).json()
    assert answered["n_pending"] == 1
    assert answered["n_open"] == before["n_open"] - 1
    after = api.get(f"/api/runs/{run_id}/questions", headers=auth(token)).json()
    assert after["n_pending"] == 1 and after["n_open"] == before["n_open"] - 1


def test_the_page_offers_no_inert_button_and_no_unscoped_direction():
    """M4 + H1, on the page itself: the exclude button is not drawn on a `which_measure` card (its
    override could never apply), and the direction form offers the measures the map names rather
    than a blank box that meant "every measure of this outcome".

    Fix round 2 (finding 11) widened the first rule rather than replacing it — the same button was
    inert or wrong on three more card kinds — so the condition is now a named predicate and
    `which_measure` is one of its clauses (`test_app_js_card_actions_do_what_they_say` pins the
    rest).
    """
    source = (Path(__file__).resolve().parents[1] / "canopy" / "server" / "static"
              / "app.js").read_text(encoding="utf-8")
    assert 'q.kind === "which_measure"' in source and "if (!inertExclude) {" in source
    assert "blank means every measure" not in source
    assert "o.outcome_key === evidence.outcome_key && o.measure_name" in source
    assert "payload.id = q.id" in source or "{ id: q.id," in source


def test_answering_a_question_that_does_not_exist_is_a_404(cloned):
    api, run_id, token = cloned["api"], cloned["run_id"], cloned["token"]
    response = api.post(f"/api/runs/{run_id}/questions/999/answer", headers=auth(token),
                        json={"option": "v1"})
    assert response.status_code == 404


def test_every_run_file_an_image_points_at_carries_the_runs_token():
    """A run's files are served behind its token, and an `<img>` sends no headers.

    The questions page built its evidence image straight from `q.image.url`, so the browser
    asked for the crop without a token, got a 401, and the reviewer was asked "which plotted
    series is this group?" beside a broken-image icon — the one question kind that is useless
    without the picture.
    """
    import re

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    # every src:/href: that carries a URL built from the API goes through withToken(...)
    for match in re.finditer(r"(?:src|href):\s*([^,}\n]+)", app_js):
        value = match.group(1).strip()
        if "/api/" in value or value.endswith(".url") or value in ("url", "imageUrl"):
            assert "withToken" in value or value == "imageUrl", value
    assert "var imageUrl = withToken(q.image.url);" in app_js


# ==================================================== DECISION A/B/F: both lines, on the page
@pytest.fixture(scope="module")
def client_nine(tmp_path_factory) -> TestClient:
    """The nine-paper run, RE-POOLED, served read-only.

    The recorded fixture was written before the best-guess line, the conclusion and the renderer
    block existed, so its `pooled.json` carries none of them. Re-pooling it here is the run's own
    path (`apply_overrides_and_repool`, no model call), which is how a reviewer's browser gets
    those keys on a run that predates them — and it means this test reads what the pipeline
    writes rather than a payload the test invented.
    """
    from canopy.pipeline.overrides import apply_overrides_and_repool
    from canopy.server.app import create_app
    from tests.helpers import nine

    runs = tmp_path_factory.mktemp("nine-lines")
    run_dir = nine.copy_to(runs)
    apply_overrides_and_repool(run_dir)
    token = "l" * 43
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_dir.name, "token": token, "title": "nine", "created_at": "2026-08-18",
        "status": "done", "options": {}, "n_files": 9, "cost_usd": 0.0, "error": "",
        "started_at": "2026-08-18", "finished_at": "2026-08-18", "kind": "run"}), encoding="utf-8")
    return TestClient(create_app(runs_dir=runs), headers=auth(token))


def test_results_endpoint_carries_both_lines_and_conclusion(client_nine):
    body = client_nine.get("/api/runs/nine/results/late_adaptation").json()
    assert body["best_guess"]["k"] >= 5 and body["conclusion"]["sentences"] \
        and body["renderer"]["renderer"] and "png" in body["forest_best_guess"]
    assert any(r["in_best_guess"] and not r["in_primary"] and r["best_guess_rule"]
               for r in body["rows"])


def test_app_js_has_the_hooks():
    """The SPA's half of DECISION A/C: the toggle, the badge, the card, the new kinds."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for needle in ("line-toggle", "state.line", "pill guess", "q-status", 'name: "option_"',
                   "body.overrides", "include_paper", "precedence_override", "analysed_n",
                   # §C1, second fold: slots answered one at a time, posted as `slots`
                   "q.slot_answers", 'name: "slot_" + i', "payload.slots = slotPicks",
                   # the free-text trigger on a slot card is a checkbox (a lone radio can never
                   # be unchecked), and a typed value beside slot picks is refused, not dropped
                   'name: "free_toggle"', "Not both in one submit"):
        assert needle in js, needle
    # the results page lands on the best-guess line when the run has one; a tab that switched
    # to strict stays there, and a run with no guess falls back to strict in `drawLineToggle`
    stored = js.split("function storedLine(")[1].split("\n  }")[0]
    assert 'stored === "strict" ? "strict" : "best_guess"' in stored


def test_app_js_card_actions_do_what_they_say():
    """Fix round 2, findings 11, 14 and 15 — the three places the page said one thing and did
    another.

    The wordless "Exclude these cells" button posted `{exclude: true}` on cards where an exclusion
    is not what the answer writes: on a `precedence_override` it recorded "the printed value
    stands" (the opposite decision), and on an `analysed_n` or a measure-scoped card it produced a
    422. The best-guess toggle enabled itself on a line that had admitted nothing, showing the
    strict number twice under two names. And the conclusion card rendered the whole paragraph —
    both lines' headline numbers — under a comment claiming no state of the page shows both, then
    printed every caveat a second time.
    """
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "inertExclude" in js and 'q.kind === "analysed_n"' in js and 'q.scope === "measure"' in js
    assert "Number(guess.n_added) >= 1 && Number(guess.k) >= 2" in js
    card = js.split("function renderConclusion(")[1].split("\n  }")[0]
    assert "conclusion.best_guess_sentences" in card and "conclusion.headline" in card
    assert "(conclusion.caveats || []).forEach" not in card


def test_a_hidden_protocol_pane_stops_demanding_its_fields():
    """The Run button is the form's SUBMIT, and the guided pane's `p-title` is `required`.

    Switching to the YAML tab only hides that pane, so the field stayed required and stayed in the
    form: the browser failed validation, tried to focus a control it could not see, logged "An
    invalid form control with name='' is not focusable" to the console, and did nothing. A run
    pasted as YAML could not be started at all — silently, with the reason nowhere on the page —
    while "Dry run", a plain `type="button"`, worked, which is what made it look like the button
    rather than the mode.

    Pinned against the PANE rather than against `p-title`, because the failure returns the moment
    anyone marks another field in either pane required.
    """
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    # the shape that makes this possible: one form, a submit button, a required field in a pane
    assert 'id="run-form"' in html and 'type="submit"' in html and "required" in html
    # …and the mode switch takes the requirement away with the pane
    assert 'showPane($("guided"), state.mode === "guided")' in js
    assert 'showPane($("yaml-mode"), state.mode === "yaml")' in js
    body = js.split("function demandFields(")[1].split("\n  }")[0]
    assert "field.required = false" in body and "data-was-required" in body
    assert "field.required = true" in body          # …and gives it back on the way in


def test_a_named_profile_in_uploaded_yaml_survives_the_run_dir_roundtrip(tmp_path):
    """The web UI's YAML path used to re-dump the parsed protocol into the run directory. A full
    `model_dump` writes every field, a later `load_protocol` counts every field in the YAML as
    explicitly chosen, and `apply_profile` becomes a no-op — so `profile: metafor` (Hedges' g,
    z prediction interval) was recorded and executed as the class defaults (Cohen's d, V), and a
    surviving row would have been Cohen's d labelled Hedges' g.

    The run directory now keeps the uploaded document verbatim (the CLI's copyfile rule), so the
    reload resolves the profile with true explicitness; and `dump_protocol` itself resolves before
    writing, so no writer can freeze unresolved defaults again.
    """
    from canopy.protocol import dump_protocol, load_protocol
    from canopy.models import Protocol

    text = (
        "title: Roundtrip\n"
        "group_a: {key: A, label: Old, definition: older}\n"
        "group_b: {key: B, label: Young, definition: younger}\n"
        "outcomes:\n"
        "  - {key: o1, label: O, definition: outcome}\n"
        "stats:\n"
        "  profile: metafor\n"
        "  hakn: true\n"                                   # a typed setting beside the profile
    )
    src = tmp_path / "protocol.yaml"
    src.write_text(text)
    loaded = load_protocol(src)
    assert loaded.stats.estimator == "hedges" and loaded.stats.pi_method == "z"
    assert loaded.stats.hakn is True                       # the typed value beat the profile

    # …and a dump/load cycle is a fixed point that keeps the resolution
    dumped = dump_protocol(Protocol.model_validate(
        __import__("yaml").safe_load(text)), tmp_path / "resolved.yaml")
    again = load_protocol(dumped)
    assert again.stats.estimator == "hedges" and again.stats.pi_method == "z"
    assert again.stats.hakn is True


# ------------------- the net that must exist BEFORE `create_run`'s body moves anywhere
def test_creating_a_run_refuses_in_exactly_this_order(api):
    """Every refusal `POST /api/runs` can make, and the ORDER it makes them in.

    Written before the `make_run` extraction, as its regression net: a helper that validates the
    same things in a different order is not the same endpoint, and the difference is invisible
    until a user sends a request that trips two rules at once. Each case below trips the rule
    named and every rule after it, so the asserted status is the FIRST one that fires.
    """
    pdf = ("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf"))
    good = ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))
    huge = ("protocol", ("protocol.yaml", b"title: x\n" + b"# padding\n" * 200_000, "text/yaml"))

    # 1. options that are not a JSON object — before anything looks at the protocol or the files
    assert api.post("/api/runs", files=[pdf, huge],
                    data={"options": "not json"}).status_code == 422
    # 2. an oversized protocol — before "a run needs a protocol" and before the file loop
    over = api.post("/api/runs", files=[pdf, huge], data={"options": "{}"})
    assert over.status_code == 413 and "protocol" in over.json()["detail"]
    # 3. no protocol at all
    none_yet = api.post("/api/runs", files=[pdf], data={"options": "{}"})
    assert none_yet.status_code == 422 and "protocol" in none_yet.json()["detail"].lower()
    # 4. a protocol but no PDFs — after the protocol checks, before parsing
    empty = api.post("/api/runs", files=[good], data={"options": "{}"})
    assert empty.status_code == 422 and "PDF" in empty.json()["detail"]
    # 5. unparseable YAML — after the "at least one PDF" rule
    bad_yaml = api.post("/api/runs",
                        files=[pdf, ("protocol", ("p.yaml", b"title: [unclosed", "text/yaml"))],
                        data={"options": "{}"})
    assert bad_yaml.status_code == 422
    # …and the happy path still 201s with the response shape the page reads
    ok = create_run(api, options={"start": False})
    assert ok.status_code == 201
    assert set(ok.json()) == {"run_id", "token", "n_files", "status", "title"}


def test_the_same_paper_twice_is_one_file_but_two_bites_of_the_allowance(api):
    """Two upload invariants that a "tidier" helper would quietly change.

    `n_files` counts unique sha256s, so the same PDF sent twice is ONE paper — but both copies
    are read off the wire, so both are charged against the total-size allowance. A helper that
    skipped the decrement for a duplicate would move when a 413 fires; one that counted files
    instead of shas would report a paper the run does not have.
    """
    twice = [PDFS[0], PDFS[0]]
    created = create_run(api, pdfs=twice, options={"start": False})
    assert created.status_code == 201, created.text
    assert created.json()["n_files"] == 1


def test_an_old_job_json_without_a_kind_still_loads(tmp_path):
    """Every job.json written before `kind` existed must still load, as a plain run."""
    from canopy.server.jobs import Job

    run_dir = tmp_path / "20260101-000000-old"
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_dir.name, "title": "an older review", "token": "t" * 32,
        "created_at": "2026-01-01T00:00:00+00:00", "status": "done", "options": {}}),
        encoding="utf-8")
    job = Job.load(run_dir)
    assert job is not None and job.kind == "run" and job.title == "an older review"


# ==================================================== the paper search, as the browser reads it
# Every pin below is a static one, in the style of the SPA tests above: the search screen is
# vanilla JS with no build step, so the file itself is the artefact under test. What they protect
# is the one thing this feature can most easily get wrong — the page and the server drifting into
# two different APIs, which is exactly what the design review found the first time.
def test_the_new_run_screen_still_starts_on_the_upload_flow():
    """Mode 2 is optional: the page a user loads is the page they had before it existed."""
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'data-source="upload"' in page and 'data-source="find"' in page
    assert '<div class="drop" id="drop">' in page          # unchanged, and never ships hidden
    assert 'class="src-btn is-on" data-source="upload"' in page
    for needle in ('id="find-panel"', 'id="find-btn"', 'id="step-search"'):
        opening = page.split(needle)[1].split(">")[0]
        assert "hidden" in opening, needle                 # …and all three ship hidden
    assert 'papersFrom: "upload"' in app_js                # …and the JS agrees


def test_the_papers_source_toggle_cannot_hijack_the_protocol_toggle():
    """`.seg-btn`'s handler is global and sets state.mode — a second control built from that class
    would silently hide the protocol editor. The source toggle uses `.src-btn`, the same dodge the
    results line toggle already uses."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert app_js.count('state.mode = button.getAttribute("data-mode")') == 1
    assert 'querySelectorAll(".src-btn")' in app_js
    assert 'class="src-btn' in page and 'class="seg-btn' in page
    assert 'setPapersFrom(button.getAttribute("data-source"))' in app_js
    assert 'showPane($("guided"), state.mode === "guided")' in app_js   # still the one pane switch


def test_the_find_panel_cannot_start_a_run_by_pressing_enter():
    """#f-max-usd is a number input inside #run-form, so Enter fires the form's default button —
    #run-btn, hidden or not. Only a DISABLED default button suppresses implicit submission; without
    this, Enter in the search budget would post a run with no PDFs."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    body = app_js.split("function setPapersFrom(")[1].split("\n  }")[0]
    assert 'var finding = state.papersFrom === "find";' in body
    assert '$("run-btn").disabled = finding;' in body
    assert 'show($("find-btn"), finding);' in body


def test_the_search_screen_is_registered_and_stays_out_of_the_way():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'SCREENS = ["new", "search", "monitor", "results", "runs"]' in app_js
    assert 'id="screen-search"' in page and 'data-screen="search"' in page
    assert 'id="step-search"' in page
    assert 'show($("step-search"), true)' in app_js


def test_the_search_counts_are_the_servers_own_vocabulary():
    """Five of ten count names differed between the two sides in the first draft of this feature,
    which renders `undefined` on screen. The page now reads COUNT_KEYS and nothing else."""
    from canopy.search.models import COUNT_KEYS

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    listed = app_js.split("var COUNT_KEYS = [")[1].split("];")[0]
    assert tuple(re.findall(r'"([a-z_]+)"', listed)) == COUNT_KEYS
    for name in sorted(set(re.findall(r"counts\.([a-z_]+)", app_js))):
        assert name in COUNT_KEYS, name


def test_the_search_ladder_uses_the_servers_own_phase_names():
    """`query` ≠ `queries` and `oa` ≠ `fetch`: two rungs that never lit, in the first draft."""
    from canopy.search.models import PHASES

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    listed = app_js.split("var SEARCH_PHASES = [")[1].split("];")[0]
    assert tuple(re.findall(r'\["([a-z]+)"', listed)) == PHASES
    # the pipeline's own status words, drawn with the monitor's glyphs — no new vocabulary
    ladder = app_js.split("function phaseClass(")[1].split("\n  }")[0]
    for status in ('"ok"', '"running"', '"skipped"', '"error"'):
        assert status in ladder, status


def test_the_search_sends_only_options_the_server_will_accept():
    """`SearchOptions` forbids extras, so one field it does not have is a 422 on the first request.

    The expected set is READ OFF the model rather than written out here. The hand-written tuple
    this replaced pinned the two fields of the day and had to be edited the moment a third was
    added — which is the one moment a mirror test is supposed to be doing its job, not being
    rewritten. Derived, it now fails for the drift it was written to catch (a field the page sends
    and the model forbids) and passes for a field both sides gained together.
    """
    from canopy.server.searches import SearchOptions

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    body = app_js.split("function searchOptions(")[1].split("\n  }")[0]
    assert "options.max_usd = " in body and "options.max_screened = " in body
    sent = set(re.findall(r"options\.([a-z_]+) =", body))
    assert sent <= set(SearchOptions.model_fields), "the page sends a field the server forbids"
    assert sent == {"max_usd", "max_screened", "max_fetch_unsure", "depth", "snowball",
                    "seed_dois", "exclude"}
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="f-max-usd"' in page and 'id="f-max-screened"' in page
    assert 'id="f-max-fetch-unsure"' in page and 'id="f-seeds"' in page
    assert 'id="f-depth"' in page and 'id="f-snowball"' in page
    assert 'id="f-exclude"' in page


def test_the_paywalled_link_comes_from_the_server_never_from_the_page():
    """test_the_spa_makes_no_external_requests forbids the literal; this says why it must stay
    forbidden — a publisher URL the page built out of a DOI would be a URL nobody audited.

    The ban is on HOST literals, not on the names of services. `"https:"` does appear in this
    file (the scheme check below), so `"https:" + "//doi.org/" + doi` is the concatenation this
    tripwire exists to catch, and the list now covers every host the search talks to rather than
    two of them. What it deliberately allows is a service NAMED in prose or in a DOM id — the
    Find panel has to be able to say "no OpenAlex key is set" in words a person can act on, and
    a sentence is not a URL. The proof that no href is ever assembled is the loop below.
    """
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    for host in ("doi.org", "openalex.org", "europepmc.org", "ebi.ac.uk", "ncbi.nlm.nih.gov",
                 "unpaywall.org", "eutils", "web.archive.org"):
        assert host not in app_js, host
    assert "var elsewhere = link.url;" in app_js          # the whole URL, as the server sent it
    assert 'protocol === "https:"' in app_js              # …and it must be one, before it is drawn
    assert 'rel: "noopener noreferrer"' in app_js and 'target: "_blank"' in app_js
    # every outbound href in the whole file is either a run file carrying its token or a whole
    # URL the server sent — there is no third way to make one, and no doi is ever concatenated
    for value in re.findall(r"href:\s*([^,}\n]+)", app_js):
        assert "withToken" in value or value.strip() in ("elsewhere", "imageUrl"), value


def test_every_string_the_search_shows_is_written_as_text():
    """A title, a venue and a screener's sentence all come from a publisher or a model. They are
    written with h({text: …}), which is textContent — the same rule as everywhere else."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    row = app_js.split("function candRow(")[1].split("\n  }")[0]
    for field in ("candidate.title", "metaLine(candidate)", "whyLine(candidate)"):
        assert f"text: {field}" in row or f'text: {field} || ""' in row, field
    assert "innerHTML" not in app_js                      # …and never the other way


def test_the_search_streams_and_polls_exactly_like_a_run():
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'new EventSource(withSearchToken(searchPath("/events")))' in app_js
    assert "function pollSearchWhileRunning(" in app_js and "state.searchPoll" in app_js
    assert "function withSearchToken(" in app_js
    # `budget` is not a job status and never will be: a capped search still finishes `done`
    assert "SEARCH_TERMINAL" not in app_js
    assert 'search.stopped_because === "budget"' in app_js


def test_the_search_token_never_rides_on_the_run_header():
    """api() attaches Authorization only for /api/runs — the path prefix is the whole protection —
    so the search passes its own bearer and the two can never be swapped."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'path.indexOf("/api/runs") === 0' in app_js     # api() itself is untouched
    assert 'Authorization: "Bearer " + state.searchToken' in app_js


def test_the_search_survives_a_reload_the_way_a_run_does():
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "canopy.searches" in app_js and "canopy.search" in app_js
    assert "function rememberSearch(" in app_js and "function forgetCurrentSearch(" in app_js
    assert "attachSearch(savedSearch, savedSearchToken)" in app_js
    assert "forgetCurrentSearch(); });" in app_js          # a dead id is forgotten, not shouted


def test_beginning_a_review_hands_over_to_the_ordinary_monitor():
    """The search must not grow a second monitor: begin calls attach(), which is the one that
    remembers the token, paints the manifest, opens the stream and navigates."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'searchPath("/begin")' in app_js and "attach(body.run_id, body.token)" in app_js
    assert "runOptions()" in app_js                        # one options row, mirrored not copied
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    for one in ('id="o-budget"', 'id="o-per-paper"', 'id="o-concurrency"', 'id="o-max-papers"'):
        assert page.count(one) == 1, one                   # …and never duplicated on a second screen


def test_the_begin_button_checks_the_protocol_itself_and_locks_while_it_works():
    """demandFields exists because a required control on a hidden pane cannot be focused, so
    #begin-btn is not a submit and never asks a form on another screen to validate. And begin
    re-reads every staged PDF: without the lock, a double click is two runs."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "reportValidity" not in app_js
    body = app_js.split("function beginRun(")[1].split("\n  }\n")[0]
    assert "protocolWritten()" in body and 'goto("new")' in body
    assert "beginning = true;" in body and "button.disabled = true;" in body
    # the gate asks whichever editor is in front of the user: #p-title is empty for pasted YAML
    gate = app_js.split("function protocolWritten(")[1].split("\n  }")[0]
    assert 'state.mode === "yaml" ? $("p-yaml").value : $("p-title").value' in gate


def test_a_paywalled_paper_can_be_opened_uploaded_or_skipped():
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    for needle in ('"Upload the PDF"', '"Skip"', "function uploadCandidate(",
                   "function decideCandidate(", '"/upload"', '"/decide"', "function uploadExtras("):
        assert needle in app_js, needle
    # a poll must never revert a tick whose POST is still in flight
    assert "state.pendingDecisions" in app_js and "function pendingKeep(" in app_js


def test_no_candidate_ever_falls_off_the_search_screen():
    """Every `CandidateState` the server can write lands in a list that exists in the markup, and
    the reason it is in that list is in the row, not in a tooltip."""
    from canopy.search.models import CandidateState
    from typing import get_args

    page = (STATIC / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    buckets = re.findall(r'"([a-z]+)"', app_js.split("var BUCKETS = [")[1].split("];")[0])
    for name in buckets:
        assert f'id="list-{name}"' in page, name
        assert f'id="{name}-count"' in page, name
    sorted_by = app_js.split("function bucketOf(")[1].split("\n  }")[0]
    for kind in get_args(CandidateState):
        assert f'"{kind}"' in sorted_by or kind == "not_screened", kind
    why = app_js.split("function whyLine(")[1].split("\n  }")[0]
    for kind in get_args(CandidateState):
        assert f'"{kind}"' in why or kind == "not_screened", kind


def test_every_search_failure_says_what_happened():
    """An index that fell over, a cap that ran out, a search the server forgot — each one named,
    and none of them empties a list."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    for needle in ("No records matched",
                   "Nothing found was open access",
                   "This search stopped when the server did",
                   "nothing was thrown away",
                   # not "finding papers needs a model" any more: a keyless search finds them,
                   # fetches the open-access copies and lists them — what it does not do is sort
                   # them for you (review §B3)
                   "Sorting the papers for you needs a model",
                   "That file is not a PDF",
                   "Upload at least one PDF to begin",
                   "Search stopped.",
                   "did not. The papers below are from the indexes that ",
                   "Describe your review in a sentence first."):
        assert needle in app_js, needle
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="card-unscreened"' in page and "Not screened" in page


def test_a_paper_with_no_pdf_gets_its_links_and_an_upload_slot_whatever_bucket_it_is_in():
    """§B3: offering them only to `locked` and `wanted` made a keyless search a dead end.

    With no model nothing is screened, so every paper lands in `unscreened` — and every one of
    them had no link to follow, nowhere to put a PDF, and no way to reach the Begin button.
    """
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    row = app_js.split("function candRow(")[1].split("\n  }")[0]
    assert "if (!candidate.pdf) {" in row, "the gate is the PDF, not the bucket"
    assert "linkNodes(candidate)" in row and "uploadSlot(candidate, row, problem)" in row
    assert '"locked"' not in row and '"wanted"' not in row
    # …and the abstract the server now sends is what a person judges an unscreened paper by
    assert "candidate.abstract_excerpt" in row
    # the unscreened list ships collapsed; when nothing was screened it IS the result set
    assert '$("card-unscreened").open = true;' in app_js


def test_the_search_shows_what_it_wrote_down_and_what_it_would_not_merge():
    """§M12 and §M13: the server sent `notes` and `possible_duplicates` and the page rendered
    neither — the tool telling the user something and then hiding it.

    `notes` is every degradation sentence the pipeline writes; the duplicate pairs are the half
    of the no-auto-merge rule that is supposed to cost the reader one click.
    """
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderNotes(" in app_js and "search.notes" in app_js
    assert "renderNotes(search);" in app_js
    assert "function renderDuplicates(" in app_js and "search.possible_duplicates" in app_js
    assert "renderDuplicates(search);" in app_js
    assert "function dropDuplicate(" in app_js, "the one click a false split is supposed to cost"
    # the cards are built by the page, so they must be built from the same helpers as every
    # other list — no markup, no innerHTML
    assert "function cardOnce(" in app_js and "innerHTML" not in app_js


def test_every_row_says_why_it_has_no_pdf_in_canopys_own_words():
    """`fetch_outcome` was recorded for every candidate the fetch stage did not reach and shown
    nowhere, so "no index offered a copy" and "the cap stopped us" read the same on screen."""
    from canopy.search.fetch import CANCELLED, NOT_WANTED, NO_OA_LOCATION, OVER_FETCH_CAP

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    words = app_js.split("var FETCH_WORDS = {")[1].split("\n  };")[0]
    for outcome in (NO_OA_LOCATION, OVER_FETCH_CAP, CANCELLED, NOT_WANTED, "rate_limited"):
        assert outcome + ":" in words, outcome
    assert "FETCH_WORDS[candidate.fetch]" in app_js


def test_a_partial_begin_names_the_papers_it_left_out():
    """§M4: a review that starts without 39 of its 40 papers is not a toast carrying a number."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "var skippedAtBegin = [];" in app_js
    assert "skippedAtBegin = (body.skipped || []).slice();" in app_js
    assert "Left out of the review: " in app_js       # …and it stays on the search screen
    banner = app_js.split("function searchBanner(")[1].split("\n  }\n")[0]
    assert "skippedAtBegin" in banner


def test_the_search_counts_are_written_in_real_plurals():
    """"1 paper", "3 papers" — the PRISMA line and every banner go through plural()."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    search = app_js.split("function renderFlow(")[1]
    assert "plural(" in search.split("\n  }")[0]
    banner = app_js.split("function searchBanner(")[1].split("\n  }\n")[0]
    assert banner.count("plural(") >= 3
    # …and no count is written with a bare " s" escape hatch anywhere in the search screen
    assert "paper(s)" not in search and "record(s)" not in search


def test_the_search_screen_is_operable_without_a_mouse():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="search-title" tabindex="-1"' in page
    assert '$("search-title").focus()' in app_js
    assert 'aria-live="polite"' in page.split('id="search-live"')[1].split(">")[0]
    assert 'id="search-log" aria-live="off"' in page       # the log does not speak
    assert "function announce(" in app_js and "state.announced" in app_js
    # the file input a keyboard has to reach is visually hidden, not `hidden`
    assert 'cls: "sr-only"' in app_js and ".sr-only {" in \
        (STATIC / "styles.css").read_text(encoding="utf-8")


def test_the_search_ui_adds_no_new_colour():
    """Offprint: one chromatic accent, and it lives where data lives."""
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    block = css.split("/* ─────────────────────────────────────────────────────────── paper search")[1]
    block = block.split("mobile */")[0]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block), "new colours entered the palette"
    for reused in (".cand", ".counts", ".begin-bar", ".phases", ".sr-only", ".find-panel"):
        assert reused in block, reused
    assert ".src-btn" in css                               # appended to the existing selector list


def test_a_file_whose_name_is_not_a_pdf_is_a_400_not_a_500(api):
    """Mode 1's own contract: an upload Canopy will not take is refused, not crashed on.

    The search feature briefly regressed this. Building the list of sources for `make_run`
    happens BEFORE `make_run`'s ordered refusals run, so a `.pdf`-name check that raised a bare
    ValueError there escaped as a 500 — on a request the endpoint had always answered with an
    honest 400. `safe_filename` truncates at 200 characters, so a long enough name reaches this
    path through the ordinary page, which is how a user would have met it.
    """
    long_name = ("a" * 260) + ".pdf"       # truncated past its extension by safe_filename
    response = api.post("/api/runs",
                        files=[("files", (long_name, PDFS[0].read_bytes(), "application/pdf")),
                               ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(),
                                             "text/yaml"))],
                        data={"options": json.dumps({"start": False})})
    assert response.status_code != 500, response.text
    assert response.status_code in (201, 400, 413)
    if response.status_code == 400:
        assert "PDF" in response.json()["detail"] or "pdf" in response.json()["detail"]


# ============================== the refusals and invariants the order test did not reach
def test_creating_a_run_refuses_the_per_file_and_manager_rules_in_order_too(api, monkeypatch,
                                                                           tmp_path):
    """The half of `POST /api/runs`'s refusal table the order pin above never covered.

    It covered five refusals and omitted the 413 too-many-files, the 400 no-key, the 429
    capacity and — the gap that mattered — EVERY per-file refusal, which is exactly where a
    regression walked through: a `.pdf`-suffix check moved out of the loop turned an honest 400
    into a 500. Each case asserts the SENTENCE, not only the status: the status says something
    went wrong and the sentence is the only part a person can act on.
    """
    from canopy.server.jobs import JobManager, TooManyRuns

    pdf = ("files", (PDFS[0].name, PDFS[0].read_bytes(), "application/pdf"))
    good = ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))
    tiny = b"%PDF-1.4\n%%EOF\n"

    def post(files, options="{}"):
        return api.post("/api/runs", files=files, data={"options": options})

    # 6. too many files — after the "at least one PDF" rule, before the key and the parse
    monkeypatch.setattr("canopy.server.app.DEFAULT_MAX_FILES", 2)
    many = post([("files", (f"p{n}.pdf", tiny, "application/pdf")) for n in range(3)] + [good])
    assert many.status_code == 413 and many.json()["detail"] == "3 files is over the 2 limit"

    # 7. `start` with no key — after the file count, before the capacity check
    monkeypatch.setattr(JobManager, "key_required", lambda self: True)
    monkeypatch.setattr(JobManager, "has_capacity", lambda self: False)
    no_key = post([pdf, good])
    assert no_key.status_code == 400
    assert no_key.json()["detail"] == ("no ANTHROPIC_API_KEY is configured — add one to .env "
                                       "and try again")
    # …and the file count still beats it, which is the ORDER half of the claim
    assert post([("files", (f"p{n}.pdf", tiny, "application/pdf")) for n in range(3)]
                + [good]).status_code == 413

    # 8. …then the capacity check, with its own sentence
    monkeypatch.setattr(JobManager, "key_required", lambda self: False)
    busy = post([pdf, good])
    assert busy.status_code == 429 and "CANOPY_MAX_ACTIVE_RUNS" in busy.json()["detail"]
    # neither fires when the run is not being started: both are `options.start` rules
    assert post([pdf, good], json.dumps({"start": False})).status_code == 201

    # 9. the PER-FILE refusals, all of them below the protocol parse and inside the loop
    monkeypatch.setattr(JobManager, "has_capacity", lambda self: True)
    off = json.dumps({"start": False})
    cases = [
        (("notes.txt", PDFS[0].read_bytes()), 400, "is not a PDF (the name must end in .pdf)"),
        (("notes", PDFS[0].read_bytes()), 400, "is not a PDF (the name must end in .pdf)"),
        # `safe_filename` truncates at 200 characters, so a long name loses its extension and
        # reaches the same rule — through the page's own `/\.pdf$/i` filter, which passes it
        (("x" * 250 + ".pdf", PDFS[0].read_bytes()), 400, "is not a PDF (the name must end in"),
        (("nope.pdf", b"<html>not a paper at all</html>"), 400, "does not start with %PDF-"),
        (("empty.pdf", b""), 400, "is empty"),
    ]
    for (name, payload), status, sentence in cases:
        answer = post([("files", (name, payload, "application/pdf")), good], off)
        assert answer.status_code == status, f"{name}: {answer.text}"
        assert sentence in answer.json()["detail"], name

    # …and an unparseable protocol still beats every one of them: the loop is below the parse
    both = api.post("/api/runs",
                    files=[("files", ("notes.txt", PDFS[0].read_bytes(), "application/pdf")),
                           ("protocol", ("p.yaml", b"title: [unclosed", "text/yaml"))],
                    data={"options": off})
    assert both.status_code == 422 and "YAML" in both.json()["detail"]


def test_a_duplicate_pdf_is_charged_to_the_allowance_even_though_it_lands_once(api, tmp_path):
    """A4a, which its own test named and never checked. `remaining` is charged for EVERY file,
    duplicates included: both copies came off the wire. Moving `remaining -= size` inside
    `if first_time:` leaves `n_files == 1` true and silently moves when a 413 fires — so this
    pins the 413 rather than the count."""
    payload = PDFS[0].read_bytes()
    api.app.state.max_total_bytes = len(payload) * 1.5          # room for one copy, not two

    twice = api.post("/api/runs",
                     files=[("files", ("a.pdf", payload, "application/pdf")),
                            ("files", ("a.pdf", payload, "application/pdf")),
                            ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))],
                     data={"options": json.dumps({"start": False})})
    assert twice.status_code == 413
    assert "total size limit" in twice.json()["detail"]
    assert list((tmp_path / "runs").glob("*/job.json")) == []   # and the run was cleaned up


def test_the_first_name_a_paper_arrived_under_is_the_one_the_run_keeps(api, tmp_path):
    """A4c. `saved` keys on the sha256 path and `setdefault` keeps the FIRST name, so
    `filenames.json` says what the person called the paper the first time they sent it — which
    is the string every later screen shows beside it."""
    payload = PDFS[0].read_bytes()
    created = api.post("/api/runs",
                       files=[("files", ("first.pdf", payload, "application/pdf")),
                              ("files", ("second.pdf", payload, "application/pdf")),
                              ("protocol", ("protocol.yaml", PROTOCOL.read_bytes(), "text/yaml"))],
                       data={"options": json.dumps({"start": False})}).json()

    assert created["n_files"] == 1
    names = json.loads((tmp_path / "runs" / created["run_id"] / "uploads" / "filenames.json")
                       .read_text(encoding="utf-8"))
    assert list(names.values()) == ["first.pdf"]


def test_a_keyboardinterrupt_mid_upload_still_removes_the_half_built_run(tmp_path):
    """A4e: the cleanup catches `BaseException`, not `Exception`. Narrowing it is invisible to
    every other test — and the run it would leave behind is a directory with half a folder of
    PDFs in it that the next `GET /api/runs` offers the user as resumable."""
    import io

    from canopy.server.jobs import JobManager
    from canopy.server.make_run import PdfSource, RunOptions, make_run

    def interrupted(path: Any, timeout: float = 0.0) -> dict[str, Any]:
        raise KeyboardInterrupt("^C while the folder was being read")

    manager = JobManager(tmp_path / "runs")
    with pytest.raises(KeyboardInterrupt):
        make_run(manager, protocol_text=PROTOCOL.read_text(encoding="utf-8"),
                 options=RunOptions(start=False),
                 sources=[PdfSource("a.pdf", lambda: io.BytesIO(PDFS[0].read_bytes()))],
                 max_upload_bytes=5e7, max_total_bytes=5e8, probe_timeout=1.0,
                 probe_pdf=interrupted)

    assert list((tmp_path / "runs").glob("*/job.json")) == []
    assert manager.list() == []                            # …and the registry forgot it too


def test_a_run_refused_at_the_starting_line_is_still_on_disk_and_startable(api, tmp_path,
                                                                          monkeypatch):
    """A4f: `_start` sits OUTSIDE `make_run`'s try/except, so a 429 from a busy manager leaves
    the run directory rather than deleting the upload the user just waited for. The existing
    capacity test gets its 429 from `make_run`'s PRE-check, before the directory exists, so it
    never touched this — the invariant survived on nobody's evidence."""
    from canopy.server.jobs import JobManager, TooManyRuns

    def busy(self, job, **_):
        raise TooManyRuns("this server already has 2 review(s) running")

    monkeypatch.setattr(JobManager, "start", busy)
    refused = create_run(api, options={"start": True})
    assert refused.status_code == 429 and "already has" in refused.json()["detail"]

    on_disk = list((tmp_path / "runs").glob("*/job.json"))
    assert len(on_disk) == 1, "the upload was thrown away with the refusal"
    run_id = on_disk[0].parent.name
    listed = next(r for r in api.get("/api/runs").json()["runs"] if r["run_id"] == run_id)
    assert listed["n_files"] == 2 and listed["status"] == "created"

    # …and it really is startable later, which is the whole point of keeping it
    monkeypatch.undo()
    assert api.post(f"/api/runs/{run_id}/start",
                    headers=auth(listed["token"])).status_code == 200
    api.post(f"/api/runs/{run_id}/cancel", headers=auth(listed["token"]))


# ============================================================ the page's half of an exclusion
def test_the_page_names_the_user_as_the_one_who_excluded_a_paper():
    """A paper the USER forbade sits in the same Excluded list as one the screener threw out.

    They must not read alike: the screener's line is a judgement a reviewer may argue with, and
    this one is an instruction that was obeyed. `project()` sends `excluded_by_you` for exactly
    this — the page should never have to parse the reason prose to tell whose decision it was.
    """
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    why = app_js.split("function whyLine(")[1].split("\n  }")[0]
    assert "candidate.excluded_by_you" in why
    # the person's branch answers before the screener's words are reached, so a row can never say
    # both "you excluded this" and "the screener read it and did not want it"
    assert why.index("excluded_by_you") < why.index('kind === "excluded"')
    assert "return" in why.split("candidate.excluded_by_you")[1].split("if (")[0], \
        "the person's branch answers; it does not fall through to the screener's wording"

    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "Never include these" in page and 'id="f-exclude"' in page
    # the field says what it costs and what it does, because "never proposed" is a strong promise
    panel = page.split('id="find-panel"')[1].split("</div>\n      </section>")[0]
    for promise in ("never read by", "never fetched", "never charged"):
        assert promise in panel, promise


def test_the_prisma_line_states_what_the_search_was_forbidden_to_find():
    """A recall measured against a known review is not a number unless the line says what was
    excluded — and a line the user typed that matched nothing has to be said out loud, or they
    will read "0 excluded" as "this search found none of that paper"."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    flow = app_js.split("function renderFlow(")[1].split("\n  }")[0]
    assert "counts.excluded_by_user" in flow
    assert "matched nothing" in flow
    assert "entry.matched" in flow, "an unmatched line is found by its own count, not guessed"
    # the count is on the strip too, under the name the server publishes it by
    from canopy.search.models import COUNT_KEYS

    assert "excluded_by_user" in COUNT_KEYS
    listed = app_js.split("var COUNT_KEYS = [")[1].split("];")[0]
    assert "excluded_by_user" in listed


def test_repool_under_an_edited_rve_protocol_flips_the_model_and_the_methods(cloned):
    """Editing `dependency: cluster_robust` into a run's protocol.yaml and repooling gives RVE
    numbers in pooled.json AND a methods page describing the new model — the manifest.settings
    staleness bug the RVE adversarial review verified (a repool used to write a methods
    paragraph describing the settings saved at original run time)."""
    api, run_id, token, runs = cloned["api"], cloned["run_id"], cloned["token"], cloned["runs"]

    protocol_path = runs / run_id / "protocol.yaml"
    text = protocol_path.read_text(encoding="utf-8")
    assert "stats:" in text
    text = text.replace("stats:", "stats:\n  dependency: cluster_robust\n"
                                  "  hakn: false\n  one_row_per_paper: false", 1)
    protocol_path.write_text(text, encoding="utf-8")

    repooled = api.post(f"/api/runs/{run_id}/repool", headers=auth(token))
    assert repooled.status_code == 200, repooled.text

    pooled = json.loads(api.get(f"/api/runs/{run_id}/files/results/{OUTCOME}/pooled.json",
                                headers=auth(token)).text)
    assert pooled["settings"]["dependency"] == "cluster_robust"
    assert pooled["robust"] is True or pooled["robust_fallback"]  # single-cluster data falls back
    methods = api.get(f"/api/runs/{run_id}/files/methods.md", headers=auth(token)).text
    assert "cluster-robust" in methods


# ============================================================================ site access code
def test_access_code_gates_everything_but_the_form(make_app):
    """With a code set, a browser sees only the form until it gives the code — and then holds a
    digest of it, never the code itself."""
    from canopy.server.security import ACCESS_COOKIE, access_cookie_value

    api = make_app(access_code="open-sesame", loopback_only=False)
    assert api.get("/", follow_redirects=False).status_code == 303
    assert api.get("/api/runs").status_code == 401
    assert api.get("/static/app.js", follow_redirects=False).status_code == 303

    page = api.get("/access")
    assert page.status_code == 200 and 'name="code"' in page.text
    assert page.headers["cache-control"] == "no-store"

    wrong = api.post("/access", data={"code": "nope"}, follow_redirects=False)
    assert wrong.status_code == 401 and ACCESS_COOKIE not in wrong.cookies
    assert api.get("/api/runs").status_code == 401                # a wrong answer earned nothing

    right = api.post("/access", data={"code": "open-sesame"}, follow_redirects=False)
    assert right.status_code == 303 and right.headers["location"] == "/"
    assert right.cookies.get(ACCESS_COOKIE) == access_cookie_value("open-sesame")
    assert "open-sesame" not in right.headers["set-cookie"]
    assert "HttpOnly" in right.headers["set-cookie"]
    assert api.get("/api/runs").status_code == 200                # the client kept the cookie
    assert api.get("/", follow_redirects=False).status_code == 200


def test_access_code_accepted_in_a_header_for_scripts(make_app):
    api = make_app(access_code="open-sesame", loopback_only=False)
    assert api.get("/api/runs", headers={"X-Canopy-Access": "open-sesame"}).status_code == 200
    assert api.get("/api/runs", headers={"X-Canopy-Access": "open-sesam"}).status_code == 401
    assert api.get("/api/runs", headers={"X-Canopy-Access": ""}).status_code == 401


def test_no_access_code_means_no_gate(make_app):
    """The loopback server is unchanged: nothing is gated and the form is not a page."""
    api = make_app()
    assert api.get("/api/runs").status_code == 200
    assert api.get("/access", follow_redirects=False).status_code == 303
    assert api.post("/access", data={"code": "anything"}, follow_redirects=False).status_code == 303


def test_access_cookie_is_a_digest_and_constant_time():
    from canopy.server.security import access_cookie_value, access_granted

    digest = access_cookie_value("open-sesame")
    assert len(digest) == 64 and "open-sesame" not in digest
    assert access_granted(digest, None, "open-sesame")
    tampered = digest[:-1] + ("0" if digest[-1] != "0" else "1")   # always a different digest
    assert not access_granted(tampered, None, "open-sesame")
    assert not access_granted("", "", "open-sesame")
    assert access_granted(None, None, None)                        # no code: no gate
    assert access_granted(None, "open-sesame", "open-sesame")
