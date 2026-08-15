"""Task 10 — the resumable orchestrator and the CLI, offline.

Two kinds of test live here.

* **Fake-provider tests** are the primary ones: a `FakeProvider` answers every agent with
  model-shaped JSON built from the *real* Bock 2005 PDF (real ingestion, real page text, real
  quotes), so the whole pipeline runs end to end with no API key and no fixtures. These assert
  the artefacts, the resume behaviour, the budget guard and the progress events.
* **Replay tests** run the same pipeline against recorded fixtures. They are written in full and
  skip visibly when a fixture the pipeline needs has not been recorded yet — never weakened.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from canopy.llm.client import LLMClient, MissingFixture
from canopy.llm.providers import FakeProvider, LLMRequest

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "tests" / "fixtures" / "pdfs" / "bock2005.pdf"
PROTOCOL = ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"
REPLAY = ROOT / "tests" / "fixtures" / "llm"
RECORD_HINT = ("fixture not recorded yet — run: CANOPY_LIVE=1 CANOPY_RECORD=1 "
               ".venv/bin/python -m pytest tests/test_pipeline_offline.py -q")


# ============================================================================ state helpers
def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    """A run killed mid-write must not leave a stage file that `--resume` reads as complete."""
    import os

    from canopy.pipeline import state

    target = tmp_path / "deep" / "thing.json"
    state.atomic_write(target, '{"a": 1}')
    assert json.loads(target.read_text()) == {"a": 1}

    def boom(src, dst):
        raise RuntimeError("killed between write and rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError):
        state.atomic_write(target, '{"a": 999999}')
    assert json.loads(target.read_text()) == {"a": 1}      # the old file is untouched
    assert not list(target.parent.glob(".*.tmp"))          # …and no debris is left behind


def test_stage_files_round_trip_and_report_completion(tmp_path):
    from canopy.pipeline.state import (read_stage, stage_done, stage_path, write_stage)

    sha = "a" * 64
    assert stage_done(tmp_path, sha, "map") is False
    write_stage(tmp_path, sha, "map", {"study": {"paper_id": sha}})
    assert stage_done(tmp_path, sha, "map") is True
    assert read_stage(tmp_path, sha, "map")["study"]["paper_id"] == sha
    assert stage_path(tmp_path, sha, "map") == tmp_path / "papers" / sha[:12] / "map.json"
    with pytest.raises(ValueError):
        stage_path(tmp_path, sha, "not_a_stage")


def test_manifest_round_trips(tmp_path):
    from canopy.models import RunManifest
    from canopy.pipeline.state import load_manifest, save_manifest

    manifest = RunManifest(run_id="r", created_at="now", protocol_hash="h", cost_usd=1.25)
    save_manifest(tmp_path, manifest)
    assert load_manifest(tmp_path).cost_usd == 1.25


def test_paper_client_meters_cost_and_stops_one_paper_at_its_cap():
    from canopy.pipeline.state import PaperBudgetExceeded, PaperClient

    client = LLMClient(provider=FakeProvider({"ok": True}), allow_live=True, cache_dir=None)
    paper = PaperClient(client, "a" * 64, max_usd=0.02)
    schema = {"type": "object", "additionalProperties": False, "required": ["ok"],
              "properties": {"ok": {"type": "boolean"}}}
    calls = 0
    with pytest.raises(PaperBudgetExceeded) as excinfo:
        for i in range(50):
            calls += 1
            paper.structured(model="claude-opus-5", messages=[{"role": "user",
                                                               "content": f"hello {i}"}],
                             schema=schema)
    assert paper.cost_usd > 0.02
    assert "of its $0.0200 allowance" in str(excinfo.value)
    assert paper.n_calls == calls                          # every call it made is accounted for
    assert client.total_cost() >= paper.cost_usd           # …and shows up on the shared client


def test_paper_client_delegates_everything_else():
    from canopy.pipeline.state import PaperClient

    client = LLMClient(provider=FakeProvider(), cache_dir=None)
    paper = PaperClient(client, "b" * 64)
    assert paper.live is client.live
    assert paper.pdf_block(pdf_path=PDF)["type"] == "document"
    assert paper.calls() == client.calls()


def test_review_queue_is_sorted_by_pooled_impact(tmp_path):
    from canopy.models import EffectSizeRecord, StatsSettings
    from canopy.pipeline.state import pooled_impact, sort_review_queue

    settings = StatsSettings()
    primary = [EffectSizeRecord(dataset_id=f"d{i}", paper_id=f"p{i}", cluster_id=f"p{i}",
                                es=e, var=v)
               for i, (e, v) in enumerate([(-0.5, 0.09), (-0.4, 0.09), (-0.6, 0.09)])]
    tiny = EffectSizeRecord(dataset_id="dX", paper_id="pX", cluster_id="pX", es=-0.5, var=0.09)
    huge = EffectSizeRecord(dataset_id="dY", paper_id="pY", cluster_id="pY", es=3.0, var=0.09)
    small_impact = pooled_impact(primary, tiny, settings)
    big_impact = pooled_impact(primary, huge, settings)
    assert big_impact > small_impact >= 0
    assert pooled_impact(primary, EffectSizeRecord(dataset_id="dZ"), settings) is None

    queue = sort_review_queue([
        {"paper_id": "small", "impact_abs_delta_pooled": small_impact},
        {"paper_id": "unknown", "impact_abs_delta_pooled": None},
        {"paper_id": "big", "impact_abs_delta_pooled": big_impact}])
    assert [e["paper_id"] for e in queue] == ["big", "small", "unknown"]


# ============================================================================ fake pipeline
def _quote_with_numbers(paper) -> tuple[int, str, float, float]:
    """A real line of the real paper that prints two numbers — used as the fake's evidence."""
    pattern = re.compile(r"(\d+\.\d+)")
    for page in paper.pages:
        for line in paper.page_text(page.number).splitlines():
            found = pattern.findall(line)
            if len(found) >= 2 and len(line.split()) >= 6 and line.isascii():
                return page.number, line.strip(), float(found[0]), float(found[1])
    raise AssertionError("no numeric line found in the ingested paper")


def _properties(request: LLMRequest) -> set[str]:
    return set(((request.schema or {}).get("properties") or {}))


def _mapper_payload(paper, page: int, quote: str) -> dict[str, Any]:
    from canopy.agents.mapper import roster_entries

    return {
        "citation": {"authors": "Author, A.", "year": 2005, "title": paper.title,
                     "journal": "", "doi": paper.doi,
                     "first_author": (paper.title or paper.filename).split()[0][:20]},
        "eligible": True, "eligibility_rationale": "older and younger adults were compared",
        "exclusion_reason": "", "design_notes": "", "notes": "", "related_files": [],
        "roster": [{"id": entry["id"], "relevant": False,
                    "reason": "no value this review needs", "outcome_keys": []}
                   for entry in roster_entries(paper)],
        "datasets": [{
            "label": "pointing", "experiment": "1", "condition": "",
            "shared_control": False, "exposure_order": "first",
            "chosen_pair_rationale": "the paper compares exactly two age groups",
            "moderators": [{"name": "task_type", "value": "visuomotor"},
                           {"name": "perturbation_size_deg", "value": "60"},
                           {"name": "n_targets", "value": "8"}],
            "notes": "",
            "group_a": {"label": "elderly", "n": 12, "n_evidence": quote, "age_mean": 69.5,
                        "age_sd": 3.0, "age_range": "", "notes": ""},
            "group_b": {"label": "young", "n": 12, "n_evidence": quote, "age_mean": 26.0,
                        "age_sd": 2.0, "age_range": "", "notes": ""},
            "all_groups_listed": [],
        }],
    }


def _sources_payload(page: int, quote: str) -> dict[str, Any]:
    source = {"kind": "text_mean_sd", "page": page, "locator": "Results \u00b61", "quote": quote,
              "figure_id": None, "table_id": None, "error_bar_type": "SD",
              "error_bar_scope": "between_subject", "error_bar_evidence": quote,
              "analysis_metric": "endpoint", "values_in_text": quote, "notes": ""}
    return {"notes": "", "datasets": [{"dataset_index": 1, "outcomes": [{
        "outcome_key": "late_adaptation", "measure_name": "mean direction error",
        "units": "deg", "higher_is_better": False,
        "higher_is_better_evidence": quote, "operationalization": "mean of the last block",
        "analysis_metric": "endpoint", "sources": [source]}]}]}


def _crosscheck_payload() -> dict[str, Any]:
    return {"eligible": True, "eligibility_rationale": "two age groups", "notes": "",
            "roster_error_bars": [],
            "datasets": [{"label": "pointing", "experiment": "1", "condition": "",
                          "group_a": {"label": "elderly", "n": 12},
                          "group_b": {"label": "young", "n": 12},
                          "outcome_keys": ["late_adaptation"], "notes": ""}]}


def _text_group(group: str, page: int, quote: str, mean: float, sd: float) -> dict[str, Any]:
    return {"group": group, "group_label_as_written": "elderly" if group == "A" else "young",
            "status": "found", "page": page, "kind": "text_mean_sd", "quote": quote,
            "row_header": "", "col_header": "", "value_as_written": f"{mean} \u00b1 {sd}",
            "mean": mean, "dispersion_value": sd, "dispersion_type": "SD",
            "ci_low": None, "ci_high": None, "ci_level": None, "unit": "deg", "n": 12,
            "n_quote": quote, "raw_value_semantics": "higher_more_error",
            "analysis_metric": "endpoint", "error_bar_scope": "between_subject", "notes": ""}


class FakeSpec:
    """One paper's canned answers, plus the strings that identify its requests."""

    def __init__(self, paper, mean_a: float, mean_b: float, sd: float = 6.0):
        self.paper = paper
        self.page, self.quote, printed_a, printed_b = _quote_with_numbers(paper)
        self.mean_a = mean_a if mean_a is not None else printed_a
        self.mean_b = mean_b if mean_b is not None else printed_b
        self.sd = sd
        self.markers = [self.quote[:60]]
        self.markers += [f.caption[:40] for f in paper.figures if len(f.caption) > 20]
        self.markers += [t.caption[:40] for t in paper.tables if len(t.caption) > 20]
        self.markers = [m for m in self.markers if m.strip()]


def _request_text(request: LLMRequest) -> str:
    """Everything textual in a request — the base64 document blocks are skipped."""
    parts: list[str] = [str(request.system or "")]
    for message in request.messages:
        content = message.get("content") if isinstance(message, dict) else message
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def fake_router(specs: "list[FakeSpec]"):
    """Answer every agent with a shape its own parser accepts, built from the real papers.

    Which paper a request is about is decided by looking for that paper's own strings (its
    quote, its figure and table captions) in the request — the same way a reader would.
    """
    def which(request: LLMRequest) -> FakeSpec:
        text = _request_text(request)
        scored = sorted(specs, key=lambda s: -sum(1 for m in s.markers if m and m in text))
        return scored[0]

    def route(request: LLMRequest) -> Any:
        props = _properties(request)
        spec = which(request)
        page, quote, sd = spec.page, spec.quote, spec.sd
        if request.tools:                                  # the digitiser's tool loop
            final = next((t for t in request.tools if "submit" in str(t.get("name", ""))),
                         request.tools[-1])
            return [{"type": "tool_use", "id": "t1", "name": final["name"], "input": {}}]
        if {"citation", "datasets", "roster"} <= props:
            return _mapper_payload(spec.paper, page, quote)
        if "roster_error_bars" in props:
            return _crosscheck_payload()
        if "decisions" in props:
            return {"decisions": []}
        if "error_bar_rulings" in props:
            return {"eligible": True, "eligibility_rationale": "", "notes": "",
                    "error_bar_rulings": [], "datasets": []}
        if props == {"datasets", "notes"}:
            return _sources_payload(page, quote)
        if "statistics" in props:
            return {"notes": "", "statistics": []}
        if "groups" in props and "rationale" in props:     # adjudicator
            return {"rationale": "the printed values agree", "needs_human": False, "notes": "",
                    "groups": [{"group": g, "n": 12, "mean": m, "dispersion_value": sd,
                                "dispersion_type": "SD", "unit": "deg", "quote": quote,
                                "page": page, "locator": "Results",
                                "chosen_candidate_ids": [], "reason": "printed in the text",
                                "needs_human": False}
                               for g, m in (("A", spec.mean_a), ("B", spec.mean_b))]}
        if "groups" in props:                              # text extractor
            return {"notes": "",
                    "groups": [_text_group("A", page, quote, spec.mean_a, sd),
                               _text_group("B", page, quote, spec.mean_b, sd)]}
        if "verdict" in props:
            return {"verdict": "confirmed", "reason": "the value is printed on this page",
                    "checked": ["wrong_group", "se_vs_sd", "time_window"], "alt_mean": None,
                    "alt_dispersion_value": None, "alt_dispersion_type": "UNKNOWN",
                    "alt_n": None, "alt_page": None, "alt_quote": "", "better_source": "",
                    "better_source_page": None, "notes": ""}
        if "raw_value_semantics" in props:
            return {"raw_value_semantics": "higher_more_error", "higher_is_better": "lower",
                    "direction_stated_in_text": "a_greater", "quotes": [quote],
                    "reason": "a direction error is worse when larger"}
        raise AssertionError(f"the fake has no answer for schema {sorted(props)}")
    return route


PDFS = (ROOT / "tests" / "fixtures" / "pdfs" / "bock2005.pdf",
        ROOT / "tests" / "fixtures" / "pdfs" / "heuer2008.pdf")


@pytest.fixture(scope="module")
def fake_specs(tmp_path_factory) -> "list[FakeSpec]":
    """Both fixture papers, really ingested, with different effects so pooling has work to do."""
    from canopy.ingest.pdf import ingest_pdf

    root = tmp_path_factory.mktemp("fake_papers")
    papers = [ingest_pdf(path, root / path.stem) for path in PDFS]
    return [FakeSpec(papers[0], 44.6, 30.2), FakeSpec(papers[1], 21.5, 17.9)]


@pytest.fixture
def fake_client(fake_specs):
    return LLMClient(provider=FakeProvider([fake_router(fake_specs)]), allow_live=True,
                     cache_dir=None)


@pytest.fixture
def papers_dir(tmp_path):
    import shutil

    directory = tmp_path / "papers"
    directory.mkdir()
    for path in PDFS:
        shutil.copyfile(path, directory / path.name)
    return directory


def test_offline_run_produces_every_artefact(tmp_path, papers_dir, fake_client):
    from canopy.pipeline.run import run_pipeline

    events: list[dict[str, Any]] = []
    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1,
                            progress=events.append)

    assert (out / "manifest.json").exists()
    assert (out / "results" / "late_adaptation" / "forest.png").exists()
    assert (out / "report.html").exists() and (out / "methods.md").exists()
    assert (out / "prisma.json").exists() and (out / "methods_routes.png").exists()
    assert (out / "exclusions.csv").exists()

    table = out / "results" / "late_adaptation" / "extraction_table.csv"
    rows = list(csv.DictReader(table.open(newline="", encoding="utf-8")))
    assert len(rows) >= 1
    assert rows[0]["dataset_id"] and rows[0]["outcome_key"] == "late_adaptation"
    assert rows[0]["route"] and rows[0]["conversion_chain"]

    assert manifest.papers and manifest.papers[0].status == "resolved"
    assert manifest.papers[0].stages == {s: "done" for s in
                                         ("ingest", "map", "extract", "verify", "resolve")}
    assert manifest.n_llm_calls > 0
    assert manifest.protocol_hash

    stages = {e["stage"] for e in events}
    assert {"ingest", "map", "extract", "verify", "resolve", "pool", "run"} <= stages
    assert all({"stage", "paper", "status", "cost_so_far", "message"} == set(e) for e in events)


def test_resume_makes_no_model_calls_at_all(tmp_path, papers_dir, fake_client, fake_specs):
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    first = run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    assert first.n_llm_calls > 0

    second_client = LLMClient(provider=FakeProvider([fake_router(fake_specs)]),
                              allow_live=True, cache_dir=None)
    second = run_pipeline(papers_dir, PROTOCOL, out, client=second_client, concurrency=1,
                          resume=True)
    assert second_client.calls() == []                     # nothing was asked of any model
    assert second.n_llm_calls == 0
    assert all(status.stages == {s: "skipped" for s in
                                 ("ingest", "map", "extract", "verify", "resolve")}
               for status in second.papers)
    assert (out / "results" / "late_adaptation" / "forest.png").exists()


def test_no_resume_reruns_every_stage(tmp_path, papers_dir, fake_client, fake_specs):
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    again = LLMClient(provider=FakeProvider([fake_router(fake_specs)]),
                      allow_live=True, cache_dir=None)
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=again, concurrency=1, resume=False)
    assert manifest.n_llm_calls > 0
    assert all(v == "done" for v in manifest.papers[0].stages.values())


def test_a_paper_that_blows_its_own_budget_does_not_stop_the_run(tmp_path, papers_dir,
                                                                 fake_client):
    from canopy.pipeline.run import run_pipeline

    manifest = run_pipeline(papers_dir, PROTOCOL, tmp_path / "run", client=fake_client,
                            concurrency=1, max_usd_per_paper=1e-9)
    assert manifest.papers[0].status == "error"
    assert "allowance" in manifest.papers[0].error
    assert (tmp_path / "run" / "manifest.json").exists()   # the run still finished and wrote out


def test_an_ineligible_paper_is_excluded_with_its_reason(tmp_path, papers_dir, fake_specs):
    from canopy.pipeline.run import run_pipeline

    router = fake_router(fake_specs)

    def ineligible(request: LLMRequest) -> Any:
        """Both mapper agents must say no — one dissenter would (correctly) force adjudication."""
        payload = router(request)
        if isinstance(payload, dict) and "eligible" in payload:
            payload = {**payload, "eligible": False, "datasets": [],
                       "exclusion_reason": "single age group",
                       "eligibility_rationale": "only younger adults took part"}
        return payload

    client = LLMClient(provider=FakeProvider([ineligible]), allow_live=True, cache_dir=None)
    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1)
    assert manifest.papers[0].status == "excluded"
    rows = list(csv.DictReader((out / "exclusions.csv").open(newline="", encoding="utf-8")))
    assert any(r["reason"] == "not_eligible" for r in rows)


def test_validate_repools_a_finished_run_without_any_model_call(tmp_path, papers_dir, fake_client):
    from canopy.pipeline.run import revalidate, run_pipeline

    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    before = len(fake_client.calls())
    report = revalidate(out)
    assert len(fake_client.calls()) == before              # revalidate asks nothing of a model
    assert report["records"] >= 1
    assert "late_adaptation" in report["outcomes"]
    assert report["missing_outputs"] == []
    assert report["protocol_matches_manifest"] is True


def test_target_spec_is_built_from_the_mapper_and_the_protocol():
    from canopy.models import DispersionType, OutcomeSources, Source, SourceKind
    from canopy.pipeline.run import target_for_source
    from canopy.protocol import load_protocol

    protocol = load_protocol(PROTOCOL)
    from canopy.models import DatasetSpec, GroupSpec

    dataset = DatasetSpec(dataset_id="d1", group_a=GroupSpec(label="elderly", n=12),
                          group_b=GroupSpec(label="young", n=12))
    source = Source(kind=SourceKind.figure_line, page=3, locator="Fig 2A", figure_id="fig02",
                    error_bar_type=DispersionType.SE, quote="filled circles are the elderly")
    sources = OutcomeSources(outcome_key="late_adaptation", measure_name="direction error",
                             units="deg")
    target = target_for_source(source, dataset, sources, protocol, protocol.stats)
    assert target.outcome_key == "late_adaptation"
    assert target.group_a_label == "elderly" and target.group_b_label == "young"
    assert target.panel_hint == "Fig 2A"
    assert target.error_bar_type_hint == "SE"
    assert target.unit_hint == "deg"
    assert target.late_window_sd == protocol.stats.late_window_sd
    assert "perturbation" in target.x_hint          # the protocol's own measurement window
    assert "filled circles" in target.notes


# ============================================================================ CLI
def _runner():
    from typer.testing import CliRunner

    return CliRunner()


def test_cli_protocol_init_writes_a_loadable_commented_skeleton(tmp_path):
    from canopy.cli import app
    from canopy.protocol import load_protocol

    target = tmp_path / "my_review.yaml"
    result = _runner().invoke(app, ["protocol", "init", "my_review", "--out", str(target)])
    assert result.exit_code == 0, result.output
    text = target.read_text()
    assert text.count("#") > 10                            # it is commented, not bare
    protocol = load_protocol(target)                       # …and it actually loads
    assert protocol.outcomes and protocol.stats.profile
    again = _runner().invoke(app, ["protocol", "init", "my_review", "--out", str(target)])
    assert again.exit_code == 1 and "already exists" in again.output


def test_cli_protocol_init_can_start_from_an_example(tmp_path):
    from canopy.cli import app
    from canopy.protocol import load_protocol

    target = tmp_path / "copy.yaml"
    result = _runner().invoke(app, ["protocol", "init", "copy", "--out", str(target),
                                    "--from", "aging_sensorimotor_adaptation"])
    assert result.exit_code == 0, result.output
    assert load_protocol(target).outcomes[0].key == "late_adaptation"


def test_cli_validate_reports_a_finished_run(tmp_path, papers_dir, fake_client):
    from canopy.cli import app
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    result = _runner().invoke(app, ["validate", str(out)])
    assert result.exit_code == 0, result.output
    assert "late_adaptation" in result.output


def test_cli_validate_fails_loudly_when_an_artefact_is_missing(tmp_path, papers_dir, fake_client):
    from canopy.cli import app
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    (out / "results" / "late_adaptation" / "forest.png").unlink()
    result = _runner().invoke(app, ["validate", str(out)])
    assert result.exit_code == 1
    assert "missing output" in result.output


def test_cli_serve_says_what_is_missing_until_task_12_lands():
    from canopy.cli import app

    result = _runner().invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "report.html" in result.output


def test_cli_never_prints_the_api_key(tmp_path, papers_dir, monkeypatch):
    from canopy.cli import app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-THIS-MUST-NOT-APPEAR")
    result = _runner().invoke(app, ["protocol", "check", str(PROTOCOL)])
    assert result.exit_code == 0, result.output
    assert "THIS-MUST-NOT-APPEAR" not in result.output


# ============================================================================ replay on Bock
@pytest.fixture
def replay_client():
    """A replaying client (and a recording one under CANOPY_LIVE=1 CANOPY_RECORD=1)."""
    from canopy.config import live_enabled, load_env, record_enabled

    live, record = live_enabled(), record_enabled()
    if live:
        load_env()
    return LLMClient(replay_dir=REPLAY, record_dir=REPLAY if record else None, allow_live=live,
                     cache_dir=None)


@pytest.mark.replay
def test_bock_end_to_end_from_recorded_fixtures(tmp_path, papers_dir, replay_client):
    """The real pipeline over Bock 2005 with every model call replayed."""
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    try:
        manifest = run_pipeline(papers_dir, PROTOCOL, out, client=replay_client, concurrency=1)
    except MissingFixture:
        pytest.skip(RECORD_HINT)
    # the orchestrator turns one paper's failure into that paper's status rather than raising,
    # so an unrecorded fixture arrives here as an error on the paper, not as an exception
    if any("no fixture" in (status.error or "") for status in manifest.papers):
        pytest.skip(RECORD_HINT)

    assert (out / "manifest.json").exists()
    assert (out / "results" / "late_adaptation" / "forest.png").exists()
    rows = list(csv.DictReader(
        (out / "results" / "late_adaptation" / "extraction_table.csv").open(
            newline="", encoding="utf-8")))
    assert len(rows) >= 1
    assert manifest.papers[0].status == "resolved"
    assert any(status.eligible for status in manifest.papers)

    calls_before = len(replay_client.calls())
    again = run_pipeline(papers_dir, PROTOCOL, out, client=replay_client, concurrency=1,
                         resume=True)
    assert len(replay_client.calls()) == calls_before      # a resumed run asks nothing
    assert again.n_llm_calls == 0


def test_the_run_writes_a_provenance_bundle_the_report_links(tmp_path, papers_dir, fake_client):
    """Every value that reached a verdict gets an image showing where it was read."""
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    bundle = json.loads((out / "provenance" / "provenance.json").read_text())
    assert bundle, "the run resolved values but recorded no provenance for them"
    entry = next(iter(bundle.values()))
    assert entry["quote"] and entry["page"]
    assert Path(entry["crop"]).exists()
    assert entry["matched"] is True                        # the quote was found on the page

    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Provenance" in html and "provenance/" in html
