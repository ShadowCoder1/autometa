"""The search half of the server: the job registry and the endpoints on top of it.

Everything here runs with **no network and no model**. The whole search pipeline is injected as
one callable (`create_app(search_runner=...)`), so the fake below writes the record a real search
would have written — queries, sources, three candidates, a staged PDF, a possible duplicate — and
nothing in this file can reach an index or an API. The PDF probe is faked too: it is a child
process running the real ingester, and this file is about the endpoints, not about PDF parsing.

What is pinned here, beyond the happy path:

* the **state machine** the design review found holes in — a running search cannot be edited, a
  search becomes exactly one run, and a search that has become a run cannot be edited afterwards;
* the **security baseline**, identical to a run's — a token is required, an unknown id is a 404
  rather than a 401 oracle, and a cross-site POST is refused;
* the **separation** — a search never appears in `GET /api/runs`, and a run directory written by
  an older build still loads.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from canopy.search.models import Candidate, SearchRecord, new_key

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"


# ============================================================================ fakes
def pdf_bytes(marker: str) -> bytes:
    """A file that passes `stream_upload` (the magic bytes) and nothing more.

    The probe is faked, so these never have to be real PDFs — and keeping them tiny is what makes
    this module run in a second rather than in the minutes a real ingest of two papers costs.
    """
    return b"%PDF-1.4\n% " + marker.encode() + b"\n%%EOF\n"


def sha_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


FETCHED = pdf_bytes("fetched-by-the-search")
UPLOADED = pdf_bytes("supplied-by-a-person")
EXTRA = pdf_bytes("a-paper-the-search-missed")
NOT_A_PDF = b"MZ\x90\x00 this is a windows binary"

KEY_FETCHED = new_key("c", "fetched")
KEY_PAYWALLED = new_key("c", "paywalled")
KEY_EXCLUDED = new_key("c", "excluded")


def fake_probe(path: Any, timeout: float = 0.0) -> dict[str, Any]:
    """Every staged file is a readable three-page paper — except one named to be refused."""
    if Path(path).read_bytes().startswith(b"%PDF-1.4\n% unreadable"):
        return {"ok": False, "error": "no readable page", "n_pages": 0, "n_chars": 0,
                "n_figures": 0, "has_text_layer": False}
    return {"ok": True, "n_pages": 3, "n_chars": 9000, "n_figures": 1, "has_text_layer": True}


def fake_search(record: SearchRecord, *, search_dir: Path, options: Any, client_factory: Any,
                cancel: Any, progress: Any, save: Any) -> SearchRecord:
    """What a real search would have left on disk, written directly.

    One fetched paper (with its bytes in `staging/`), one paywalled paper the screener wanted but
    could not get, and one the screener ruled out — which is exactly the three-way split the page
    has to render and the three-way split `begin` has to filter.
    """
    staging = search_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / f"{sha_of(FETCHED)}.pdf").write_bytes(FETCHED)

    record.query_source = "template"
    record.queries = [{"index": "openalex", "query": "tdcs AND motor learning", "n": 2}]
    record.criteria = ["adults", "randomised"]
    record.sources = [{"name": "openalex", "asked": "tdcs AND motor learning", "returned": 3,
                       "unique_contributed": 3, "error": ""}]
    record.phases = [{"name": name, "status": "done", "message": f"{name} finished",
                      "seconds": 0.01}
                     for name in ("queries", "index", "dedupe", "screen")]
    record.candidates = [
        Candidate(key=KEY_FETCHED, title="Adaptation in older adults", authors=["Bock, O"],
                  year=2005, doi="10.1000/fetched", found_by=["openalex"], state="fetched",
                  screen_decision="include", screen_reason="an adaptation study in older adults",
                  keep=True, pdf_path=f"staging/{sha_of(FETCHED)}.pdf", pdf_pages=3,
                  pdf_bytes=len(FETCHED), fetch_outcome="fetched",
                  links=[{"label": "doi", "url": "https://doi.org/10.1000/fetched"}]),
        Candidate(key=KEY_PAYWALLED, title="A paper behind a paywall", authors=["Smith, J"],
                  year=2011, doi="10.1000/paywalled", found_by=["openalex"], state="paywalled",
                  screen_decision="include", screen_reason="matches the question",
                  keep=True, fetch_outcome="no_oa_location"),
        Candidate(key=KEY_EXCLUDED, title="A paper about something else", year=1999,
                  found_by=["openalex"], state="excluded", screen_decision="exclude",
                  screen_reason="not a motor task", keep=False),
    ]
    record.possible_duplicates = [{"left": KEY_FETCHED, "right": KEY_PAYWALLED,
                                   "reason": "the titles are close and the years are one apart"}]
    record.cost_usd = 0.023
    progress({"stage": "search", "paper": "", "status": "running", "cost_so_far": 0.023,
              "message": "screened 3 abstracts"})
    save(record)
    record.phases.append({"name": "fetch", "status": "done", "message": "1 of 2 fetched",
                          "seconds": 0.02})
    save(record)
    return record


# ============================================================================ fixtures
@pytest.fixture(autouse=True)
def no_real_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both doors read the probe off `canopy.server.app`, so one patch covers upload AND begin."""
    monkeypatch.setattr("canopy.server.app.probe_pdf", fake_probe)


@pytest.fixture
def make_app(tmp_path: Path):
    """`make_app(**kwargs) -> TestClient`, with a fake search pipeline and no model anywhere."""
    from canopy.server.app import create_app

    def never(**_: Any) -> Any:                            # nothing in this file may call a model
        raise AssertionError("the search endpoints must not build an LLM client")

    def factory(**kwargs: Any) -> TestClient:
        kwargs.setdefault("runs_dir", tmp_path / "runs")
        kwargs.setdefault("searches_dir", tmp_path / "searches")
        kwargs.setdefault("client_factory", never)
        kwargs.setdefault("search_runner", fake_search)
        return TestClient(create_app(**kwargs))

    return factory


@pytest.fixture
def api(make_app) -> TestClient:
    return make_app()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def start_search(api: TestClient, question: str = "does tDCS help motor learning?",
                 **body: Any) -> tuple[str, str]:
    response = api.post("/api/searches", json={"question": question, **body})
    assert response.status_code == 201, response.text
    payload = response.json()
    return payload["search_id"], payload["token"]


def wait_done(api: TestClient, search_id: str, token: str,
              timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = api.get(f"/api/searches/{search_id}", headers=auth(token)).json()
        if body["status"] in ("done", "error", "cancelled", "interrupted"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"the search never finished: {body}")            # pragma: no cover


def finished(api: TestClient) -> tuple[str, str, dict[str, Any]]:
    search_id, token = start_search(api)
    return search_id, token, wait_done(api, search_id, token)


def begin(api: TestClient, search_id: str, token: str, **extra: Any):
    """Begin with `start: false` — this file tests the door, never the pipeline behind it."""
    body = {"protocol_text": PROTOCOL.read_text(encoding="utf-8"),
            "options": {"start": False}, **extra}
    return api.post(f"/api/searches/{search_id}/begin", json=body, headers=auth(token))


# ============================================================================ the happy path
def test_a_search_runs_and_the_page_can_read_every_field_it_draws(api):
    search_id, token, body = finished(api)

    assert body["status"] == "done"
    assert body["search_id"] == search_id
    assert body["question"] == "does tDCS help motor learning?"
    assert body["error"] == "" and body["stopped_because"] == ""
    assert body["cost_usd"] == pytest.approx(0.023)
    assert body["query_source"] == "template"
    assert [q["index"] for q in body["queries"]] == ["openalex"]
    assert [s["name"] for s in body["sources"]] == ["openalex"]
    assert len(body["possible_duplicates"]) == 1

    # the PRISMA ladder, derived from the candidates rather than counted alongside them
    assert body["counts"]["after_dedupe"] == 3
    assert body["counts"]["included"] == 2
    assert body["counts"]["excluded"] == 1
    assert body["counts"]["fetched"] == 1
    assert body["counts"]["paywalled"] == 1
    assert body["counts"]["possible_duplicates"] == 1

    # every phase in `PHASES`, in order, whether the pipeline mentioned it or not
    assert [p["name"] for p in body["phases"]] == ["queries", "index", "dedupe", "screen",
                                                   "fetch"]
    assert all(p["status"] == "done" for p in body["phases"])

    # the candidates arrive PROJECTED: display strings, no abstract, no nested fetch record
    row = next(c for c in body["candidates"] if c["key"] == KEY_FETCHED)
    assert row["study_label"] == "Bock 2005"
    assert row["state"] == "fetched" and row["keep"] is True
    assert row["pdf"] == {"pages": 3, "bytes": len(FETCHED)}
    assert row["reason"] == "an adaptation study in older adults"
    assert "abstract" not in row and "fetch_attempts" not in row


def test_the_record_on_disk_is_the_audit_trail(api, tmp_path):
    search_id, token, _ = finished(api)
    record = json.loads((tmp_path / "searches" / search_id / "search.json")
                        .read_text(encoding="utf-8"))
    assert record["question"] == "does tDCS help motor learning?"
    assert record["counts"]["after_dedupe"] == 3
    # the screener's own sentence, verbatim, beside the decision it explains
    excluded = next(c for c in record["candidates"] if c["key"] == KEY_EXCLUDED)
    assert excluded["screen_decision"] == "exclude"
    assert excluded["screen_reason"] == "not a motor task"


def test_the_event_stream_replays_and_ends(api):
    search_id, token, _ = finished(api)
    with api.stream("GET", f"/api/searches/{search_id}/events?token={token}") as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "event: progress" in text
    assert "screened 3 abstracts" in text
    assert "event: end" in text


# ============================================================================ decide / upload
def test_decide_toggles_keep_and_leaves_the_state_alone(api):
    search_id, token, _ = finished(api)

    response = api.post(f"/api/searches/{search_id}/papers/{KEY_EXCLUDED}/decide",
                        json={"keep": True}, headers=auth(token))
    assert response.status_code == 200
    assert response.json() == {"key": KEY_EXCLUDED, "keep": True}

    row = next(c for c in api.get(f"/api/searches/{search_id}", headers=auth(token))
               .json()["candidates"] if c["key"] == KEY_EXCLUDED)
    # a paper a human ticked back on still shows why the screener dropped it
    assert row["keep"] is True and row["state"] == "excluded"
    assert row["reason"] == "not a motor task"


def test_decide_needs_to_say_which_way(api):
    search_id, token, _ = finished(api)
    response = api.post(f"/api/searches/{search_id}/papers/{KEY_FETCHED}/decide", json={},
                        headers=auth(token))
    assert response.status_code == 422


def test_uploading_the_pdf_for_a_paywalled_paper(api, tmp_path):
    search_id, token, _ = finished(api)
    response = api.post(f"/api/searches/{search_id}/papers/{KEY_PAYWALLED}/upload",
                        files={"file": ("smith_2011.pdf", UPLOADED, "application/pdf")},
                        headers=auth(token))
    assert response.status_code == 200, response.text
    row = response.json()
    assert row["key"] == KEY_PAYWALLED
    assert row["state"] == "uploaded"          # who supplied it is part of the audit trail
    assert row["keep"] is True                 # supplying it IS the decision to include it
    assert row["pdf"] == {"pages": 3, "bytes": len(UPLOADED)}
    assert row["upload"]["filename"] == "smith_2011.pdf"
    # stored under its own sha256, never under anything from the request
    assert (tmp_path / "searches" / search_id / "staging" /
            f"{sha_of(UPLOADED)}.pdf").is_file()


def test_an_upload_that_is_not_a_pdf_is_refused(api, tmp_path):
    search_id, token, _ = finished(api)
    response = api.post(f"/api/searches/{search_id}/papers/{KEY_PAYWALLED}/upload",
                        files={"file": ("evil.pdf", NOT_A_PDF, "application/pdf")},
                        headers=auth(token))
    assert response.status_code == 400
    assert "%PDF-" in response.json()["detail"]
    assert list((tmp_path / "searches" / search_id / "staging").glob("*.pdf")) == [
        tmp_path / "searches" / search_id / "staging" / f"{sha_of(FETCHED)}.pdf"]


def test_an_upload_that_does_not_probe_is_refused_with_the_reason(api):
    search_id, token, _ = finished(api)
    response = api.post(f"/api/searches/{search_id}/papers/{KEY_PAYWALLED}/upload",
                        files={"file": ("broken.pdf", pdf_bytes("unreadable"),
                                        "application/pdf")}, headers=auth(token))
    assert response.status_code == 400
    assert "no readable page" in response.json()["detail"]


def test_papers_the_search_missed_are_added_and_the_bad_ones_named(api):
    search_id, token, _ = finished(api)
    response = api.post(
        f"/api/searches/{search_id}/papers",
        files=[("files", ("mine.pdf", EXTRA, "application/pdf")),
               ("files", ("junk.pdf", NOT_A_PDF, "application/pdf"))],
        headers=auth(token))
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["added"]) == 1
    assert body["added"][0]["state"] == "extra"
    assert body["added"][0]["keep"] is True
    assert body["added"][0]["key"].startswith("u")     # a person's paper, not an index's
    assert [r["filename"] for r in body["rejected"]] == ["junk.pdf"]

    # and it is in the counts the page draws
    counts = api.get(f"/api/searches/{search_id}", headers=auth(token)).json()["counts"]
    assert counts["extra"] == 1 and counts["after_dedupe"] == 4


def test_the_same_extra_pdf_twice_is_one_paper(api):
    search_id, token, _ = finished(api)
    for _ in range(2):
        api.post(f"/api/searches/{search_id}/papers",
                 files=[("files", ("mine.pdf", EXTRA, "application/pdf"))],
                 headers=auth(token))
    counts = api.get(f"/api/searches/{search_id}", headers=auth(token)).json()["counts"]
    assert counts["extra"] == 1


# ============================================================================ begin
def test_begin_builds_a_real_run_directory_through_make_run(api, tmp_path):
    search_id, token, _ = finished(api)
    api.post(f"/api/searches/{search_id}/papers/{KEY_PAYWALLED}/upload",
             files={"file": ("smith_2011.pdf", UPLOADED, "application/pdf")},
             headers=auth(token))

    response = begin(api, search_id, token)
    assert response.status_code == 201, response.text
    body = response.json()
    # the SAME body `POST /api/runs` returns, so the page's `attach()` needs no second branch
    assert set(body) == {"run_id", "token", "n_files", "status", "title", "skipped"}
    assert body["n_files"] == 2                # the fetched one and the uploaded one
    assert body["skipped"] == []

    run_dir = tmp_path / "runs" / body["run_id"]
    assert (run_dir / "protocol.yaml").is_file()
    assert (run_dir / "uploads" / f"{sha_of(FETCHED)}.pdf").is_file()
    assert (run_dir / "uploads" / f"{sha_of(UPLOADED)}.pdf").is_file()
    # the excluded paper had no PDF and was not kept: it is not in the run
    assert len(list((run_dir / "uploads").glob("*.pdf"))) == 2

    names = json.loads((run_dir / "uploads" / "filenames.json").read_text(encoding="utf-8"))
    # a human-readable name, never a candidate key — this is what every later screen shows
    assert sorted(names.values()) == ["bock_2005.pdf", "smith_2011.pdf"]

    # …and the run carries the search's own provenance, readable without this server
    copied = json.loads((run_dir / "search" / "search.json").read_text(encoding="utf-8"))
    assert copied["question"] == "does tDCS help motor learning?"
    assert copied["counts"]["paywalled"] == 0      # the upload moved it to `uploaded`
    assert len(copied["candidates"]) == 3

    # and the run is a run: it is in the run list, with its own token
    runs = api.get("/api/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [body["run_id"]]
    assert api.get(f"/api/runs/{body['run_id']}",
                   headers=auth(body["token"])).status_code == 200


def test_a_second_begin_returns_the_same_run(api, tmp_path):
    search_id, token, _ = finished(api)
    first = begin(api, search_id, token).json()
    second = begin(api, search_id, token)

    assert second.status_code == 201
    assert second.json()["run_id"] == first["run_id"]
    assert second.json()["token"] == first["token"]
    assert len(list((tmp_path / "runs").glob("*/job.json"))) == 1


def test_begin_with_nothing_kept_is_a_422_that_says_what_to_do(api):
    search_id, token, _ = finished(api)
    for key in (KEY_FETCHED, KEY_PAYWALLED):
        api.post(f"/api/searches/{search_id}/papers/{key}/decide", json={"keep": False},
                 headers=auth(token))
    response = begin(api, search_id, token)
    assert response.status_code == 422
    assert "tick" in response.json()["detail"]


def test_a_kept_paper_with_no_pdf_is_not_enough(api):
    """`keep` is a wish; `begin` needs bytes. The paywalled paper is kept and has none."""
    search_id, token, _ = finished(api)
    api.post(f"/api/searches/{search_id}/papers/{KEY_FETCHED}/decide", json={"keep": False},
             headers=auth(token))
    assert begin(api, search_id, token).status_code == 422


def test_begin_falls_back_to_the_protocol_the_search_was_started_with(api, tmp_path):
    search_id, token = start_search(api, protocol_text=PROTOCOL.read_text(encoding="utf-8"))
    wait_done(api, search_id, token)
    response = api.post(f"/api/searches/{search_id}/begin",
                        json={"options": {"start": False}}, headers=auth(token))
    assert response.status_code == 201, response.text
    assert (tmp_path / "runs" / response.json()["run_id"] / "protocol.yaml").is_file()


def test_begin_without_any_protocol_is_a_422(api):
    search_id, token, _ = finished(api)
    response = api.post(f"/api/searches/{search_id}/begin", json={"options": {"start": False}},
                        headers=auth(token))
    assert response.status_code == 422


def test_editing_a_search_after_it_has_become_a_run_is_refused(api):
    """The run already has its copy of the papers, so a change here would silently not be in it."""
    search_id, token, _ = finished(api)
    begin(api, search_id, token)
    for response in (
            api.post(f"/api/searches/{search_id}/papers/{KEY_FETCHED}/decide",
                     json={"keep": False}, headers=auth(token)),
            api.post(f"/api/searches/{search_id}/papers/{KEY_PAYWALLED}/upload",
                     files={"file": ("late.pdf", UPLOADED, "application/pdf")},
                     headers=auth(token)),
            api.post(f"/api/searches/{search_id}/papers",
                     files=[("files", ("late.pdf", EXTRA, "application/pdf"))],
                     headers=auth(token))):
        assert response.status_code == 409, response.text
        assert "already become" in response.json()["detail"]


# ============================================================================ the state machine
class Gate:
    """A runner that stops on command, so the "still running" refusals can be tested."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, record: SearchRecord, **kwargs: Any) -> SearchRecord:
        kwargs["save"](record)
        self.entered.set()
        self.release.wait(timeout=20)
        return record


@pytest.fixture
def gate():
    gate = Gate()
    yield gate
    gate.release.set()                                     # never leave a thread parked


def test_a_running_search_may_not_be_edited_or_begun(make_app, gate):
    api = make_app(search_runner=gate)
    search_id, token = start_search(api)
    assert gate.entered.wait(timeout=10)

    for response in (
            api.post(f"/api/searches/{search_id}/papers/{KEY_FETCHED}/decide",
                     json={"keep": False}, headers=auth(token)),
            api.post(f"/api/searches/{search_id}/papers",
                     files=[("files", ("mine.pdf", EXTRA, "application/pdf"))],
                     headers=auth(token)),
            begin(api, search_id, token)):
        assert response.status_code == 409, response.text
        assert "still going" in response.json()["detail"]


def test_a_search_can_be_cancelled(make_app, gate):
    api = make_app(search_runner=gate)
    search_id, token = start_search(api)
    assert gate.entered.wait(timeout=10)

    response = api.post(f"/api/searches/{search_id}/cancel", headers=auth(token))
    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"

    gate.release.set()
    body = wait_done(api, search_id, token)
    assert body["status"] == "cancelled"
    # the banner keys on `stopped_because`, never on the status alone
    assert body["stopped_because"] == "cancelled"


def test_a_search_that_raises_becomes_an_error_not_a_500(make_app):
    def explode(record: SearchRecord, **kwargs: Any) -> SearchRecord:
        raise RuntimeError("openalex said no")

    api = make_app(search_runner=explode)
    search_id, token = start_search(api)
    body = wait_done(api, search_id, token)
    assert body["status"] == "error"
    assert "openalex said no" in body["error"]


def test_at_capacity_a_new_search_is_a_429(make_app, gate):
    api = make_app(search_runner=gate, max_active_runs=1)
    start_search(api)
    assert gate.entered.wait(timeout=10)
    response = api.post("/api/searches", json={"question": "another one"})
    assert response.status_code == 429


def test_a_search_needs_a_question(api):
    assert api.post("/api/searches", json={"question": "   "}).status_code == 422
    assert api.post("/api/searches", json={}).status_code == 422


def test_the_caps_are_validated_not_ignored(api):
    assert api.post("/api/searches", json={"question": "q",
                                           "options": {"max_usd": -1}}).status_code == 422
    assert api.post("/api/searches", json={"question": "q",
                                           "options": {"max_udd": 3}}).status_code == 422
    # …and a blank form field means "use the default", not "zero"
    response = api.post("/api/searches", json={"question": "q",
                                               "options": {"max_usd": "", "max_screened": ""}})
    assert response.status_code == 201


def test_the_caps_are_recorded_on_the_search(api, tmp_path):
    from canopy.server.searches import DEFAULT_MAX_SCREENED

    search_id, token = start_search(api, options={"max_usd": 0.5})
    wait_done(api, search_id, token)
    job = json.loads((tmp_path / "searches" / search_id / "job.json").read_text(encoding="utf-8"))
    assert job["kind"] == "search"
    assert job["options"] == {"max_usd": 0.5, "max_screened": DEFAULT_MAX_SCREENED}


# ============================================================================ separation
def test_a_search_never_appears_in_the_run_list(api):
    search_id, token, _ = finished(api)
    assert api.get("/api/runs").json()["runs"] == []
    assert api.get(f"/api/runs/{search_id}", headers=auth(token)).status_code == 404

    listed = api.get("/api/searches").json()
    assert [s["search_id"] for s in listed["searches"]] == [search_id]
    assert listed["loopback_only"] is True
    assert listed["searches"][0]["token"] == token      # loopback only, like runs


def test_the_search_list_hides_the_token_off_loopback(make_app):
    api = make_app(loopback_only=False)
    search_id, _token, _ = finished(api)
    row = api.get("/api/searches").json()["searches"][0]
    assert row["search_id"] == search_id and "token" not in row


def test_an_old_run_job_json_still_loads(api, tmp_path):
    """A run directory written before searches existed has no `kind` and must still open."""
    run_dir = tmp_path / "runs" / "20240101-000000-old"
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": "20240101-000000-old", "token": "old-token", "title": "An old review",
        "created_at": "2024-01-01T00:00:00+00:00", "status": "done", "options": {},
        "n_files": 2, "cost_usd": 1.5, "error": "", "started_at": "", "finished_at": ""}),
        encoding="utf-8")

    rows = api.get("/api/runs").json()["runs"]
    assert [r["run_id"] for r in rows] == ["20240101-000000-old"]
    assert rows[0]["kind"] == "run"
    assert api.get("/api/runs/20240101-000000-old",
                   headers=auth("old-token")).status_code == 200


# ============================================================================ the default runner
def test_the_default_cap_agrees_with_the_pipeline(monkeypatch):
    """Two numbers that must agree. `searches.py` may not import the pipeline (it would pull
    httpx into every import of the server), so the copy is pinned here instead."""
    import os

    if os.environ.get("CANOPY_SEARCH_MAX_SCREENED"):       # pragma: no cover - operator override
        pytest.skip("the cap is overridden in this environment")
    pipeline = pytest.importorskip("canopy.search.run")
    from canopy.server import searches

    assert searches.DEFAULT_MAX_SCREENED == pipeline.DEFAULT_MAX_SCREENED


def test_the_default_runner_adapts_the_server_seam_to_the_pipeline(tmp_path, monkeypatch):
    """The one place the job vocabulary and the search vocabulary meet — so it is pinned.

    No network: `run_search` and the transport are both replaced. What is checked is the
    TRANSLATION — which of the server's words became which of the pipeline's.
    """
    pipeline = pytest.importorskip("canopy.search.run")
    from canopy.server import searches

    seen: dict[str, Any] = {}
    built: list[dict[str, Any]] = []
    saved: list[Any] = []

    def fake_run_search(**kwargs: Any) -> SearchRecord:
        seen.update(kwargs)
        return SearchRecord(search_id="s1", question=kwargs["question"], query_source="model")

    monkeypatch.setattr(pipeline, "run_search", fake_run_search)
    monkeypatch.setattr("canopy.search.transport.HttpxTransport", lambda **kw: "a transport")

    record = SearchRecord(search_id="s1", question="does tDCS help motor learning?")
    returned = searches.default_search_runner(
        record, search_dir=tmp_path, options={"max_usd": 1.5, "max_screened": 7},
        client_factory=lambda **kw: built.append(kw) or "a client",
        cancel=threading.Event(), progress=lambda event: None,
        save=lambda current=None: saved.append(current))

    assert seen["question"] == "does tDCS help motor learning?"
    assert seen["budget_usd"] == 1.5 and seen["max_screened"] == 7
    assert seen["staging_dir"] == tmp_path / "staging"
    assert seen["search_id"] == "s1"
    assert seen["cancelled"]() is False
    assert seen["protocol"] is None                        # this search was started without one
    # an INJECTED factory always counts as "a model is available", key or no key
    assert built and built[0]["purpose"] == "search" and built[0]["budget_usd"] == 1.5
    # …and the pipeline's own record is what comes back, not the one that went in
    assert returned.query_source == "model"

    # the ladder is live: a phase report lands on disk before `run_search` returns
    seen["on_phase"]("queries", "done", "3 queries", 0.4)
    assert record.phases == [{"name": "queries", "status": "done", "message": "3 queries",
                              "seconds": 0.4}]
    assert saved


def test_the_settings_endpoint_says_what_a_search_needs(api):
    from canopy.server.searches import DEFAULT_MAX_SCREENED, DEFAULT_MAX_USD

    body = api.get("/api/settings").json()
    assert body["search_max_usd"] == DEFAULT_MAX_USD
    assert body["search_max_screened"] == DEFAULT_MAX_SCREENED
    # a fake runner never reaches a model, so this server must not demand an API key
    assert body["search_key_required"] is False
    assert body["search_uses_real_models"] is False


# ============================================================================ security baseline
def test_a_search_needs_its_own_token(api):
    search_id, token, _ = finished(api)
    for response in (api.get(f"/api/searches/{search_id}"),
                     api.get(f"/api/searches/{search_id}", headers=auth("not-the-token")),
                     api.get(f"/api/searches/{search_id}/events"),
                     api.post(f"/api/searches/{search_id}/cancel"),
                     api.post(f"/api/searches/{search_id}/papers/{KEY_FETCHED}/decide",
                              json={"keep": False}),
                     api.post(f"/api/searches/{search_id}/begin", json={})):
        assert response.status_code == 401, response.text


def test_an_unknown_search_is_a_404_not_a_401_oracle(api):
    assert api.get("/api/searches/nope-not-a-search", headers=auth("x")).status_code == 404
    assert api.post("/api/searches/nope-not-a-search/cancel",
                    headers=auth("x")).status_code == 404
    # …and a path that tries to leave the searches directory is not special either
    assert api.get("/api/searches/..%2F..%2Fetc", headers=auth("x")).status_code == 404


def test_a_cross_site_post_is_refused(api):
    search_id, token, _ = finished(api)
    assert api.post("/api/searches", json={"question": "drive-by"},
                    headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert api.post(f"/api/searches/{search_id}/papers/{KEY_FETCHED}/decide",
                    json={"keep": False},
                    headers={**auth(token), "Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert api.post(f"/api/searches/{search_id}/begin", json={},
                    headers={**auth(token), "Sec-Fetch-Site": "cross-site"}).status_code == 403
    # a same-origin write is untouched
    assert api.post(f"/api/searches/{search_id}/papers/{KEY_FETCHED}/decide",
                    json={"keep": False},
                    headers={**auth(token), "Sec-Fetch-Site": "same-origin"}).status_code == 200


@pytest.mark.parametrize("key", ["..", "....", "not-a-key", "C1234567890AB", "c1234567890",
                                 "c1234567890abZ", "%2e%2e", "c" * 40])
def test_a_bad_paper_key_is_refused_without_touching_the_filesystem(api, tmp_path, key):
    search_id, token, _ = finished(api)
    search_dir = tmp_path / "searches" / search_id
    before = sorted(p.relative_to(search_dir).as_posix() for p in search_dir.rglob("*"))

    response = api.post(f"/api/searches/{search_id}/papers/{key}/decide", json={"keep": True},
                        headers=auth(token))
    assert response.status_code == 404
    # the same answer a key that is merely unknown gets, so the shape is not an oracle. A key
    # that is path-shaped (`..`) never reaches the endpoint at all: the router normalises the URL
    # and no route matches, which is Starlette's own 404 — also a refusal, one layer earlier.
    assert response.json()["detail"] in ("no such paper in this search", "Not Found")
    assert sorted(p.relative_to(search_dir).as_posix()
                  for p in search_dir.rglob("*")) == before


def test_a_well_formed_key_that_is_not_in_this_search_is_the_same_404(api):
    search_id, token, _ = finished(api)
    response = api.post(f"/api/searches/{search_id}/papers/{new_key('c', 'elsewhere')}/decide",
                        json={"keep": True}, headers=auth(token))
    assert response.status_code == 404
    assert response.json()["detail"] == "no such paper in this search"


def test_one_search_cannot_read_another(make_app):
    api = make_app()
    first, first_token, _ = finished(api)
    second, second_token, _ = finished(api)
    assert first != second
    assert api.get(f"/api/searches/{first}", headers=auth(second_token)).status_code == 401
