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
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from canopy.config import MODELS
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
    assert paper.cost_usd >= 0.02
    assert "of its $0.0200 allowance" in str(excinfo.value)
    # the cap is checked BEFORE the call, so the iteration that raised bought nothing: the caller
    # never loses a result it has already paid for, which is what let a $14 paper keep nothing (F7)
    assert paper.n_calls == calls - 1
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


def _sources_payload(page: int, quote: str, figure=None) -> dict[str, Any]:
    sources = [{"kind": "text_mean_sd", "page": page, "locator": "Results \u00b61", "quote": quote,
                "figure_id": None, "table_id": None, "error_bar_type": "SD",
                "error_bar_scope": "between_subject", "error_bar_evidence": quote,
                "analysis_metric": "endpoint", "values_in_text": quote, "notes": ""}]
    if figure is not None:                                 # \u2026and the same value plotted
        sources.append({
            "kind": "figure_bar", "page": figure.page, "locator": figure.label or "Fig 1",
            "quote": figure.caption[:200], "figure_id": figure.id, "table_id": None,
            "error_bar_type": "SD", "error_bar_scope": "between_subject",
            "error_bar_evidence": figure.caption[:200], "analysis_metric": "endpoint",
            "values_in_text": "", "notes": ""})
    return {"notes": "", "datasets": [{"dataset_index": 1, "outcomes": [{
        "outcome_key": "late_adaptation", "measure_name": "mean direction error",
        "units": "deg", "higher_is_better": False,
        "higher_is_better_evidence": quote, "operationalization": "mean of the last block",
        "analysis_metric": "endpoint", "sources": sources}]}]}


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
    """One paper's canned answers, plus the strings that identify its requests.

    The knobs are the four wiring branches a text-only, agreeing paper never reaches:
    `figure_id` makes the mapper report a figure source (so `digitize()` runs), `disagreement`
    makes the second text extractor read a different number (so the vote asks for a third
    candidate and the adjudicator is called), and `refute` makes every verifier refute (so the
    cell is re-opened up to the amendment-G limit).
    """

    def __init__(self, paper, mean_a: float, mean_b: float, sd: float = 6.0, *,
                 figure_id: str = "", disagreement: float = 0.0, refute: bool = False):
        self.paper = paper
        self.page, self.quote, printed_a, printed_b = _quote_with_numbers(paper)
        self.mean_a = mean_a if mean_a is not None else printed_a
        self.mean_b = mean_b if mean_b is not None else printed_b
        self.sd = sd
        self.disagreement = disagreement
        self.refute = refute
        self.figure = next((f for f in paper.figures if f.id == figure_id), None)
        self.figure_size = (600, 400)
        if self.figure is not None:
            from PIL import Image

            with Image.open(Path(paper.out_dir) / self.figure.crop_png) as image:
                self.figure_size = image.size
        self.markers = [self.quote[:60]]
        self.markers += [f.caption[:40] for f in paper.figures if len(f.caption) > 20]
        self.markers += [t.caption[:40] for t in paper.tables if len(t.caption) > 20]
        self.markers = [m for m in self.markers if m.strip()]

    # the digitiser's own two payloads, on the crop's real pixel geometry ------------------
    def _y_px(self, value: float) -> float:
        """A linear axis: value 0 at 90% of the height, ten units per 12% of the height."""
        return self.figure_size[1] * (0.9 - 0.012 * value)

    def coords_payload(self) -> dict[str, Any]:
        width = self.figure_size[0]
        groups = []
        for key, mean, x in (("A", self.mean_a, 0.35), ("B", self.mean_b, 0.65)):
            groups.append({
                "group": key, "label_read": "elderly" if key == "A" else "young",
                "x_px": width * x, "y_px": self._y_px(mean),
                "bar_x0_px": width * (x - 0.06), "bar_x1_px": width * (x + 0.06),
                "cap_top_px": self._y_px(mean + self.sd),
                "cap_bottom_px": self._y_px(mean - self.sd), "notes": ""})
        return {"status": "found", "panel": "Fig 1", "unit": "deg", "confidence": 0.9, "notes": "",
                "ticks": [{"value": v, "y_px": self._y_px(v)} for v in (0, 10, 20, 30, 40, 50)],
                "groups": groups}


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


def _digitizer_turn(request: LLMRequest, spec: "FakeSpec") -> Any:
    """The digitiser's three tool loops, scripted the way `tests/test_digitizer.py` scripts them."""
    from tests.test_digitizer import _readout_payload

    system = _system_text(request)
    tools = request.tools or []
    name = next((t["name"] for t in tools if "submit" in str(t.get("name", ""))),
                tools[-1]["name"] if tools else "submit")
    if "read numeric values" in system:
        payload: Any = _readout_payload(spec.mean_a, spec.sd, spec.mean_b, spec.sd)
    elif "locate features" in system:
        payload = spec.coords_payload()
    else:                                                  # overlay verification
        payload = {"marks": [], "notes": ""}
    return [{"type": "tool_use", "id": "t1", "name": name, "input": payload}]


def _system_text(request: LLMRequest) -> str:
    return request.system if isinstance(request.system, str) else json.dumps(request.system)


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
            return _digitizer_turn(request, spec)
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
            return _sources_payload(page, quote, spec.figure)
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
            # the second reader (a different model) is the one that can disagree
            shift = spec.disagreement if request.model != MODELS["primary"] else 0.0
            return {"notes": "",
                    "groups": [_text_group("A", page, quote, spec.mean_a + shift, sd),
                               _text_group("B", page, quote, spec.mean_b + shift, sd)]}
        if "verdict" in props:
            if spec.refute:
                return {"verdict": "refuted",
                        "reason": "the quoted sentence reports the baseline block, not the "
                                  "late block this outcome asks for",
                        "checked": ["wrong_group", "wrong_time_window", "se_vs_sd"],
                        "alt_mean": None, "alt_dispersion_value": None,
                        "alt_dispersion_type": "UNKNOWN", "alt_n": None, "alt_page": None,
                        "alt_quote": "", "better_source": "", "better_source_page": None,
                        "notes": ""}
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


@pytest.fixture(scope="module")
def hard_specs(tmp_path_factory) -> "list[FakeSpec]":
    """Bock again, but as the paper the wiring is afraid of.

    A plotted source (so the digitiser runs), two text readers that disagree (so the vote asks for
    a third candidate and the adjudicator is called) and a verifier that refuses to confirm (so
    the cell is re-opened).
    """
    from canopy.ingest.pdf import ingest_pdf

    root = tmp_path_factory.mktemp("hard_papers")
    paper = ingest_pdf(PDFS[0], root / PDFS[0].stem)
    return [FakeSpec(paper, 44.6, 30.2, figure_id="fig01", disagreement=9.0, refute=True)]


@pytest.fixture
def hard_client(hard_specs):
    return LLMClient(provider=FakeProvider([fake_router(hard_specs)]), allow_live=True,
                     cache_dir=None)


@pytest.fixture
def bock_dir(tmp_path):
    import shutil

    directory = tmp_path / "one_paper"
    directory.mkdir()
    shutil.copyfile(PDFS[0], directory / PDFS[0].name)
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
    assert rows[0]["primary_row"] in ("true", "false")

    combined = list(csv.DictReader(                      # one table for the whole run as well
        (out / "results" / "extraction_table_all.csv").open(newline="", encoding="utf-8")))
    assert len(combined) >= len(rows)
    assert {r["outcome_key"] for r in combined} >= {"late_adaptation"}
    assert any(r["primary_row"] == "true" for r in combined)

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
    # …and it says so in the exclusions table. Run 1's third paper died after mapping having spent
    # $14.37 and appeared in no output at all, so nobody counting papers could learn it was gone.
    rows = list(csv.DictReader(
        (tmp_path / "run" / "exclusions.csv").open(newline="", encoding="utf-8")))
    assert [r for r in rows if r["reason"] == "budget_exhausted"], [r["reason"] for r in rows]


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


def test_a_mapper_objection_reaches_the_run_instead_of_dying_where_it_was_made(
        tmp_path, papers_dir, fake_specs):
    """`StudyMap.needs_human` was written at `mapper.py` and read by nothing but `app.js`.

    Wrong outcome key, conflicting units, a group mapping two agents could not agree on — all of
    it was raised and then dropped before any table, queue or manifest saw it.
    """
    from canopy.pipeline.run import run_pipeline

    router = fake_router(fake_specs)

    def disagree(request: LLMRequest) -> Any:
        """Make the cross-check name a dataset the primary map does not have."""
        payload = router(request)
        if isinstance(payload, dict) and "roster_error_bars" in payload:      # the cross-check
            extra = dict((payload.get("datasets") or [{}])[0])
            extra["label"] = "a condition only the second agent saw"
            payload = {**payload, "datasets": list(payload.get("datasets") or []) + [extra]}
        return payload

    client = LLMClient(provider=FakeProvider([disagree]), allow_live=True, cache_dir=None)
    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1)
    mapped = [w for w in manifest.warnings if "map: " in w]
    assert mapped, manifest.warnings
    assert any("does not contain" in w for w in mapped), mapped


def test_a_paper_whose_every_dataset_resolves_to_nothing_is_still_accounted_for(
        tmp_path, papers_dir, fake_specs):
    """The success path is a way of leaving with nothing too.

    `b270022` closed the `datasets == []` door. Next to it: a paper mapped with datasets whose
    outcome keys are all outside the protocol resolves every one of them to nothing, finishes
    `resolved` with all five stages done, and appeared in no table at all — while the methods
    text said "0 were excluded" and PRISMA said `consistent: true`.
    """
    from canopy.pipeline.run import run_pipeline

    router = fake_router(fake_specs)

    def foreign_outcomes(request: LLMRequest) -> Any:
        payload = router(request)
        if isinstance(payload, dict) and "datasets" in payload:
            datasets = []
            for d in payload["datasets"]:
                d = dict(d)
                if "outcomes" in d:
                    d["outcomes"] = [{**o, "outcome_key": "not_in_this_protocol"}
                                     for o in d["outcomes"]]
                datasets.append(d)
            payload = {**payload, "datasets": datasets}
        return payload

    client = LLMClient(provider=FakeProvider([foreign_outcomes]), allow_live=True, cache_dir=None)
    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1)
    paper = manifest.papers[0]
    rows = list(csv.DictReader((out / "exclusions.csv").open(newline="", encoding="utf-8")))
    mine = [r for r in rows if r["paper_id"] == paper.paper_id and not r["dataset_id"]]
    assert mine, [(r["reason"], r["dataset_id"]) for r in rows]
    prisma = json.loads((out / "prisma.json").read_text())
    assert prisma["consistent"] is True, prisma["problems"]
    assert prisma["included_papers"] == 0
    assert prisma["eligible_papers"] - prisma["papers_with_no_rows"] == prisma["included_papers"]
    methods = (out / "methods.md").read_text(encoding="utf-8")
    assert "yielded no usable row" in methods


def test_the_methods_paragraph_reports_the_spend_the_manifest_recorded(tmp_path, papers_dir,
                                                                       fake_client):
    """Every run so far wrote "0 model calls were made at a cost of $0.00" into its methods text.

    `report.html` and `methods.md` read `n_llm_calls`/`cost_usd` off the manifest, and the report
    was built before those fields were filled in — so the same page showed real per-paper costs
    beside a zeroed total. A methods paragraph exists to be pasted into a paper.
    """
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    assert manifest.n_llm_calls > 0, "the fake run made no calls; the test proves nothing"
    methods = (out / "methods.md").read_text(encoding="utf-8")
    assert f"{manifest.n_llm_calls} model calls" in methods, methods[-400:]
    assert "0 model calls" not in methods
    html = (out / "report.html").read_text(encoding="utf-8")
    assert f"{manifest.n_llm_calls}" in html


def test_a_baseline_source_is_kept_for_the_record_and_never_read_as_the_value(
        tmp_path, papers_dir, fake_specs):
    """Cressman's Fig. 3a plots the aligned-cursor (baseline) curves beside the misaligned ones.

    The map listed both under late adaptation — the baseline "for baseline correction" — and the
    extractor digitised the baseline as if it were the outcome (3.9° beside 31.4°). A source now
    says what it is; only a `value` source is read for the number.
    """
    from canopy.pipeline.run import run_pipeline

    router = fake_router(fake_specs)
    seen: dict[str, int] = {"digitize_calls": 0}

    def with_baseline(request: LLMRequest) -> Any:
        payload = router(request)
        system = " ".join(str(b.get("text", "")) for m in request.messages
                          for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                          if isinstance(b, dict))
        if isinstance(payload, dict) and payload.get("datasets") \
                and "outcomes" in (payload["datasets"][0] or {}):
            for d in payload["datasets"]:
                for o in d.get("outcomes") or []:
                    twin = dict(o["sources"][0])
                    twin["locator"] = "aligned-cursor baseline curves"
                    twin["role"] = "baseline"
                    o["sources"] = list(o["sources"]) + [twin]
        if "read numeric values" in system or "locate features" in system:
            seen["digitize_calls"] += 1
        return payload

    client = LLMClient(provider=FakeProvider([with_baseline]), allow_live=True, cache_dir=None)
    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1)
    warnings = [w for w in manifest.papers[0].warnings if "baseline source" in w]
    assert warnings, manifest.papers[0].warnings
    assert "not read for the value" in warnings[0]
    # the map still carries the baseline location for a reader
    study = json.loads((out / "papers" / manifest.papers[0].paper_id[:12] / "map.json")
                       .read_text())["study"]
    roles = [src.get("role") for d in study["datasets"] for o in d["outcomes"]
             for src in o["sources"]]
    assert "baseline" in roles and "value" in roles
    # …and no candidate was ever produced from it
    extract = json.loads((out / "papers" / manifest.papers[0].paper_id[:12] / "extract.json")
                         .read_text())
    assert not [c for c in extract["candidates"] if "aligned-cursor" in str(c.get("locator", ""))]


def test_an_eligible_paper_with_no_dataset_never_leaves_the_run_in_silence(
        tmp_path, papers_dir, fake_specs):
    """It contributes no row, so it has to appear in the exclusions table and say why.

    `runs/proof`: Heuer & Hegele 2008 was mapped eligible with `datasets: []` and then vanished
    — absent from the forest plot, from the review queue and from the exclusions table alike, so
    a reader counting papers had no way to learn it had been dropped.
    """
    from canopy.pipeline.run import run_pipeline

    router = fake_router(fake_specs)

    def empty_map(request: LLMRequest) -> Any:
        payload = router(request)
        if isinstance(payload, dict) and "eligible" in payload:
            payload = {**payload, "datasets": []}
        return payload

    client = LLMClient(provider=FakeProvider([empty_map]), allow_live=True, cache_dir=None)
    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1)
    assert manifest.papers[0].status == "excluded"
    assert manifest.papers[0].eligible is True          # eligible, and contributed nothing anyway
    assert any("no dataset was mapped" in w for w in manifest.papers[0].warnings)
    rows = list(csv.DictReader((out / "exclusions.csv").open(newline="", encoding="utf-8")))
    entry = [r for r in rows if r["reason"] == "no_usable_data"]
    assert entry, [r["reason"] for r in rows]
    assert entry[0]["reason_as_given"] == "no_usable_data:no_datasets_mapped"
    assert "no dataset was mapped" in entry[0]["detail"]


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


def test_the_riskiest_branches_all_execute_offline(tmp_path, bock_dir, hard_client, monkeypatch):
    """digitize → third candidate → adjudication → a verifier re-open, in one green run.

    Also the controller's ruling on what may vote: the digitiser contributes four or five route
    samples per group, and only its ENSEMBLE candidate is allowed into the vote — a printed value
    must not be outvoted by however many ways one picture was measured.
    """
    from canopy.pipeline import run as run_module
    from canopy.pipeline.state import read_stage

    seen: list[list[Any]] = []
    real_vote = run_module.vote_groups

    def spy(candidates, *args, **kwargs):
        seen.append(list(candidates))
        return real_vote(candidates, *args, **kwargs)

    monkeypatch.setattr(run_module, "vote_groups", spy)

    out = tmp_path / "run"
    manifest = run_module.run_pipeline(bock_dir, PROTOCOL, out, client=hard_client, concurrency=1)
    sha = manifest.papers[0].paper_id
    assert manifest.papers[0].error == ""

    extract = read_stage(out, sha, "extract")
    extractors = {c["extractor_id"] for c in extract["candidates"]}
    assert any(e.startswith("digitize:") and e != "digitize:ensemble" for e in extractors), \
        "the digitiser's route samples are missing from the stage file"
    assert "digitize:ensemble" in extractors

    verify = read_stage(out, sha, "verify")
    assert len(verify["adjudications"]) >= 1               # the disagreement reached adjudication
    assert len(verify["extra_candidates"]) >= 1            # amendment G's third cheap reading
    assert verify["reopens"] >= 1                          # …and the verifier re-opened the cell

    voted = [cell for cell in seen
             if any(c.extractor_id.startswith("digitize:") for c in cell)]
    assert voted, "no figure candidate reached the vote at all"
    for cell in voted:
        figures = [c for c in cell if c.extractor_id.startswith("digitize:")]
        assert all(c.extractor_id == "digitize:ensemble" for c in figures), \
            "a per-route digitiser sample reached the vote"
        counts = Counter(c.group for c in figures)
        assert set(counts.values()) == {1}, f"more than one figure candidate per group: {counts}"


def test_only_the_ensemble_figure_candidate_is_offered_to_the_vote():
    """The unit behind the ruling: route samples stay in the stage file, the ensemble votes."""
    from canopy.models import Candidate
    from canopy.pipeline.run import vote_candidates

    def figure(extractor_id: str, group: str) -> Candidate:
        return Candidate(candidate_id=f"d1:late:{group}:{extractor_id}", dataset_id="d1",
                         outcome_key="late", kind="group_stats", group=group, mean=30.0,
                         route="figure", extractor_id=extractor_id, model="claude-opus-5")

    text = [Candidate(candidate_id=f"d1:late:{g}:text", dataset_id="d1", outcome_key="late",
                      kind="group_stats", group=g, mean=31.0, route="text_mean_sd",
                      extractor_id="text:table_first", model="claude-opus-5")
            for g in ("A", "B")]
    samples = [figure(f"digitize:{route}:claude-opus-5:direct", g)
               for route in ("readout", "coords", "raster", "vector") for g in ("A", "B")]
    ensemble = [figure("digitize:ensemble", g) for g in ("A", "B")]

    kept = vote_candidates([*text, *samples, *ensemble])
    assert [c.extractor_id for c in kept if c.extractor_id.startswith("digitize:")] == \
        ["digitize:ensemble", "digitize:ensemble"]
    assert [c.candidate_id for c in text] == [c.candidate_id for c in kept
                                              if not c.extractor_id.startswith("digitize:")]
    assert vote_candidates([]) == []


def test_one_row_per_paper_aggregates_the_primary_analysis_and_records_what_it_replaced():
    """Amendment A in the orchestrator's own split: combine, keep the paper trail, never select."""
    from canopy.models import EffectSizeRecord, StatsSettings
    from canopy.pipeline.aggregate import AGGREGATED_FLAG
    from canopy.pipeline.run import _split_rows

    def row(dataset_id, es, var, sample_id="", confidence="auto_accept"):
        return EffectSizeRecord(paper_id="p1", cluster_id="p1", sample_id=sample_id,
                                dataset_id=dataset_id, outcome_key="late_adaptation", es=es,
                                var=var, se=var ** 0.5, route="text_mean_sd",
                                confidence=confidence)

    rows = [row("d1", -1.2, 0.16, "p1|Experiment 1"), row("d2", -0.4, 0.36, "p1|Experiment 2"),
            row("d3", -3.0, 0.20, confidence="needs_human")]

    split = _split_rows(rows, StatsSettings(one_row_per_paper=True))
    assert len(split.primary) == 1                       # one row for this paper, not the best row
    assert AGGREGATED_FLAG in split.primary[0].flags
    assert [r.dataset_id for r in split.held] == ["d3"]  # held rows are never aggregated away
    assert {r.dataset_id for r in split.every} >= {"d1", "d2", "d3", "d1+d2"}
    assert sorted(e["dataset_id"] for e in split.exclusions) == ["d1", "d2"]
    assert all(e["reason"].startswith("aggregated_into:") for e in split.exclusions)

    off = _split_rows(rows, StatsSettings(one_row_per_paper=False))
    assert [r.dataset_id for r in off.primary] == ["d1", "d2"]
    assert off.exclusions == []


def test_prisma_still_adds_up_when_max_papers_stops_the_run_early(tmp_path, papers_dir,
                                                                  fake_client):
    """A partial run must not claim it excluded the papers it never opened."""
    from canopy.pipeline.run import run_pipeline

    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1, max_papers=1)
    prisma = json.loads((out / "prisma.json").read_text())
    assert prisma["unique_papers"] == 2                   # both papers are in the folder
    assert prisma["not_processed"] == 1                   # …one was never looked at
    assert prisma["papers_excluded"] == 0                 # …and nothing was rejected
    assert prisma["consistent"] is True, prisma["problems"]


def test_validate_reads_the_protocol_the_run_kept_beside_its_stage_files(tmp_path, papers_dir,
                                                                         fake_client):
    """`canopy run` copies the protocol into the run; re-pooling must use that copy."""
    import shutil

    from canopy.pipeline.run import revalidate, run_pipeline

    moved = tmp_path / "elsewhere" / "protocol.yaml"
    moved.parent.mkdir(parents=True)
    shutil.copyfile(PROTOCOL, moved)
    out = tmp_path / "run"
    run_pipeline(papers_dir, moved, out, client=fake_client, concurrency=1)
    shutil.copyfile(moved, out / "protocol.yaml")         # what the CLI writes
    moved.unlink()                                        # the user's own copy is gone

    report = revalidate(out)
    assert report["records"] >= 1
    assert report["protocol_matches_manifest"] is True
    assert "late_adaptation" in report["outcomes"]


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
    # the protocol's other names for each group travel too: whether a two-bar chart yields a
    # number at all turns on matching its x categories to the groups, and a paper labels its
    # bars in its own words ("Elderly", "Young adults") — see `digitizer._names_group`
    assert set(target.group_a_synonyms) == set(protocol.group_a.synonyms)
    assert set(target.group_b_synonyms) == set(protocol.group_b.synonyms)
    assert "elderly" in {n.lower() for n in target.group_a_synonyms}


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


def test_cli_serve_hands_off_to_the_web_ui_and_never_binds_a_port(monkeypatch):
    """`canopy serve` delegates to the Task-12 server, and says what is missing when it is absent.

    The server is stubbed out on purpose: a test that really called `serve` would bind a socket
    and never return (it did, once — that is why this test looks like this).
    """
    from canopy.cli import app

    try:
        from canopy.server import app as server_app
    except Exception:                                      # the UI is not in this build
        result = _runner().invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "report.html" in result.output
        return

    seen: dict[str, Any] = {}
    monkeypatch.setattr(server_app, "serve", lambda **kwargs: seen.update(kwargs))
    result = _runner().invoke(app, ["serve", "--port", "8123", "--host", "127.0.0.1"])
    assert result.exit_code == 0, result.output
    assert seen["port"] == 8123 and seen["host"] == "127.0.0.1"


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

    # …and every one of those links actually opens a file: candidate ids carry `:` and end in
    # `#N`, and a browser truncates a URL at the `#`
    from tests.test_report import local_links

    targets = local_links(html)
    assert any("provenance/" in t for t in targets)
    missing = sorted({t for t in targets if not (out / t).exists()})
    assert missing == [], f"dead links in the run's report.html: {missing}"


# ============================================================================ task 16: budget (F7)
def test_a_partial_stage_file_is_not_a_finished_stage(tmp_path):
    """The sharp edge P7 would otherwise introduce: `--resume` truncating a paper for good."""
    from canopy.pipeline.state import stage_done, write_stage

    sha = "b" * 64
    write_stage(tmp_path, sha, "extract", {"candidates": [], "complete": False})
    assert stage_done(tmp_path, sha, "extract") is False
    write_stage(tmp_path, sha, "extract", {"candidates": [], "complete": True})
    assert stage_done(tmp_path, sha, "extract") is True
    # a stage that never says anything about completeness is complete, as it always was
    write_stage(tmp_path, sha, "map", {"study": {}})
    assert stage_done(tmp_path, sha, "map") is True


def test_a_budget_death_mid_extraction_keeps_the_rows_it_already_bought(tmp_path, papers_dir,
                                                                       fake_specs):
    """F7: Buch 2003 spent $14.37 against a $14 cap and nothing at all was kept."""
    import json

    from canopy.pipeline.run import run_pipeline
    from canopy.pipeline.state import stage_done, stage_path

    client = LLMClient(provider=FakeProvider([fake_router(fake_specs)]), allow_live=True,
                       cache_dir=None)
    out = tmp_path / "run"
    # a cap that the mapper alone does not exhaust, but extraction does
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1,
                            max_usd_per_paper=0.02)
    paper = manifest.papers[0]
    assert paper.stages.get("extract") == "partial", (
        f"the cap no longer bites inside extraction (stages {paper.stages}) — pick a tighter one "
        f"rather than deleting the test, or F7 stops being covered")
    assert paper.status == "error" and "allowance" in paper.error
    payload = json.loads(stage_path(out, paper.paper_id, "extract").read_text())
    assert payload["complete"] is False
    assert payload["cells_budget_exhausted"], "no cell was recorded as budget_exhausted"
    assert any("budget_exhausted" in w for w in paper.warnings)
    # …and `--resume` re-enters the stage rather than reading the salvage as finished
    assert stage_done(out, paper.paper_id, "extract") is False


def test_an_extract_that_finished_is_marked_complete(tmp_path, papers_dir, fake_client):
    import json

    from canopy.pipeline.run import run_pipeline
    from canopy.pipeline.state import stage_done, stage_path

    out = tmp_path / "run"
    manifest = run_pipeline(papers_dir, PROTOCOL, out, client=fake_client, concurrency=1)
    paper = manifest.papers[0]
    assert paper.stages["extract"] == "done"
    payload = json.loads(stage_path(out, paper.paper_id, "extract").read_text())
    assert payload["complete"] is True
    assert payload["cells_extracted"] and not payload["cells_budget_exhausted"]
    assert stage_done(out, paper.paper_id, "extract") is True


def test_a_figure_only_cell_does_not_buy_three_text_calls(tmp_path):
    """Miss 9: Cressman's aftereffect bought four text readings of a cell with no printed values."""
    from canopy.models import Source, SourceKind
    from canopy.pipeline.run import _has_printed_source

    figures = [Source(kind=SourceKind.figure_bar, page=6, locator="Fig 3b", figure_id="fig03"),
               Source(kind=SourceKind.figure_points, page=9, locator="Fig 5", figure_id="fig05")]
    assert _has_printed_source(figures) is False
    assert _has_printed_source([*figures,
                                Source(kind=SourceKind.text_mean_sd, page=3, locator="Results")])
    assert _has_printed_source([Source(kind=SourceKind.table, page=4, locator="Table 1")])
    # a source of a kind nobody recognised is a location a human has to route: still worth reading
    assert _has_printed_source([Source(kind=SourceKind.unknown, page=4, locator="fitted model")])
    assert _has_printed_source([]) is False
