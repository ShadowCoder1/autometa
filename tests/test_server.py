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
    assert json.loads((run_dir / "overrides_applied.json").read_text())["applied"] == 1


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
