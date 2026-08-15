"""Task 11 — outputs: forest plot, tables, PRISMA, sensitivity, methods figure, HTML report.

The forest plot is exercised against the reference review's own published effect sizes
(`validation/reference/cisneros2024/late_gsheet.csv`, k = 50). That file is **validation input
only**: nothing in `canopy/` reads it, and no test here lets a number from it reach the pipeline.
It is used because a renderer that survives fifty real rows — long author strings, missing
moderators, a wide effect range — has been tested on something.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from canopy.models import (Citation, EffectSizeRecord, OutcomeDef, Protocol, StatsSettings)
from canopy.stats.meta import random_effects

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "validation" / "reference" / "cisneros2024" / "late_gsheet.csv"


# --------------------------------------------------------------------------- fixtures
def _float(raw: str) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


@pytest.fixture(scope="module")
def gold_rows() -> list[EffectSizeRecord]:
    """The reference review's 50 late-adaptation rows as `EffectSizeRecord`s (validation only)."""
    rows: list[EffectSizeRecord] = []
    with GOLD.open(newline="", encoding="utf-8") as fh:
        for i, raw in enumerate(csv.DictReader(fh)):
            te, se = _float(raw["TE"]), _float(raw["seTE"])
            if te is None or se is None or se <= 0:
                continue
            author = (raw["Author"] or "").strip()
            year = _float(raw["Year"])
            routes = ["text_mean_sd", "figure", "table", "test_statistic", "reported_d"]
            rows.append(EffectSizeRecord(
                paper_id=f"paper{i:02d}", cluster_id=f"paper{i:02d}",
                dataset_id=f"d{i:02d}", outcome_key="late_adaptation",
                label=f"{author} {int(year) if year else ''}".strip(),
                route=routes[i % len(routes)],
                n_a=int(_float(raw["N_old"]) or 0) or None,
                n_b=int(_float(raw["N_young"]) or 0) or None,
                es=te, se=se, var=se * se,
                ci_low=_float(raw["CI_low"]), ci_high=_float(raw["CI_high"]),
                estimator="cohen", variance_method="hedges_olkin_df",
                higher_is_better=False, orientation_applied=True,
                confidence="auto_accept" if i % 7 else "accept_with_note",
                moderators={"task_type": (raw["Task"] or "").strip(),
                            "perturbation_size_deg": (raw["Rotation_Size"] or "").strip(),
                            "n_targets": (raw["Target_N"] or "").strip()},
                citation=Citation(first_author=author.split()[0] if author else "",
                                  authors=author, year=int(year) if year else None),
                conversion_chain="reference values (validation input)"))
    return rows


@pytest.fixture(scope="module")
def outcome() -> OutcomeDef:
    return OutcomeDef(key="late_adaptation", label="Late adaptation",
                      definition="adaptive change late in the perturbation block",
                      positive_direction_label="Enhanced in Old",
                      negative_direction_label="Reduced in Old")


@pytest.fixture(scope="module")
def settings() -> StatsSettings:
    from canopy.protocol import apply_profile
    return apply_profile(StatsSettings(profile="cisneros2024"))


@pytest.fixture(scope="module")
def pooled_gold(gold_rows, settings):
    return random_effects([r.es for r in gold_rows], [r.var for r in gold_rows],
                          method=settings.tau2_method, hakn=settings.hakn,
                          level=settings.ci_level)


# --------------------------------------------------------------------------- forest plot
def test_gold_csv_is_the_reference_k50_set(gold_rows):
    assert len(gold_rows) == 50


def test_forest_plot_renders_all_three_formats(tmp_path, gold_rows, pooled_gold, outcome, settings):
    from canopy.report.forest import forest_plot

    out = forest_plot(gold_rows, pooled_gold, outcome, settings, tmp_path / "forest")
    assert set(out) == {"png", "svg", "pdf"}
    for path in out.values():
        assert path.exists() and path.stat().st_size > 0


def test_forest_svg_carries_labels_route_glyphs_and_footer(tmp_path, gold_rows, pooled_gold,
                                                           outcome, settings):
    from canopy.report.forest import forest_plot
    from canopy.report.theme import ROUTE_GLYPHS

    out = forest_plot(gold_rows, pooled_gold, outcome, settings, tmp_path / "forest")
    svg = out["svg"].read_text(encoding="utf-8")
    # column headers, direction labels from the protocol, and the pooled row
    # moderator headers are the protocol's names, humanised (`task_type` -> `task type`)
    for token in ("Author", "Year", "task type", "N (A/B)", "Enhanced in Old", "Reduced in Old",
                  "Random-effects"):
        assert token in svg, token
    # every route present in the rows contributes its glyph
    for route in {r.route for r in gold_rows}:
        assert ROUTE_GLYPHS[route] in svg, route
    # footer conventions (amendment A/H)
    for token in ("REML", "Hedges", "I²", "prediction interval", "k = 50"):
        assert token.lower() in svg.lower(), token


def test_needs_human_rows_are_hollow_excluded_and_counted(tmp_path, gold_rows, pooled_gold,
                                                          outcome, settings):
    from canopy.report.forest import forest_plot

    flagged = [gold_rows[0].model_copy(update={
                   "confidence": "needs_human", "dataset_id": "dQQ",
                   "citation": Citation(first_author="Heldforreview", year=1999)}),
               gold_rows[1].model_copy(update={"confidence": "needs_human", "dataset_id": "dRR"})]
    out = forest_plot(gold_rows, pooled_gold, outcome, settings, tmp_path / "forest",
                      needs_human_rows=flagged)
    svg = out["svg"].read_text(encoding="utf-8")
    assert "Heldforreview" in svg                       # the hollow row is drawn and labelled
    assert "Held for human review" in svg               # …under its own heading
    assert "2 rows excluded" in svg                     # …and counted in the conventions footer
    # a hollow square is drawn with the surface colour as its fill
    assert svg.count("fill: none") + svg.count('fill="none"') >= 2


def test_forest_plot_is_sorted_by_effect(tmp_path, gold_rows, pooled_gold, outcome, settings):
    from canopy.report.forest import forest_plot, forest_layout

    layout = forest_layout(gold_rows, pooled_gold, settings)
    order = [row.record.es for row in layout.rows]
    assert order == sorted(order)
    assert all(row.weight_pct > 0 for row in layout.rows)
    assert math.isclose(sum(row.weight_pct for row in layout.rows), 100.0, abs_tol=1e-6)


# --------------------------------------------------------------------------- tables
@pytest.fixture
def resolved_row() -> EffectSizeRecord:
    """One fully-provenanced row, as the orchestrator produces it."""
    return EffectSizeRecord(
        paper_id="a1b2c3d4e5f6", cluster_id="a1b2c3d4e5f6", dataset_id="d1",
        outcome_key="late_adaptation", label="pointing experiment", route="text_mean_sd",
        n_a=12, n_b=12, d=-1.6758, es=-1.6758, var=0.2305, se=0.4801,
        ci_low=-2.6168, ci_high=-0.7348, estimator="cohen", variance_method="hedges_olkin_df",
        higher_is_better=False, orientation_applied=True,
        conversion_chain="pooled SD = 7.505; d = (44.67 − 46.14)/7.505 = −0.1959",
        conversion_steps=["pooled SD = 7.505", "d = −0.1959"],
        routes_available=["text_mean_sd", "test_statistic"],
        routes_rejected={"test_statistic": "text_mean_sd came first in route_precedence"},
        inputs={"mean_a": 44.67, "sd_a": 8.76, "n_a": 12.0,
                "mean_b": 46.14, "sd_b": 6.11, "n_b": 12.0},
        confidence="auto_accept", flags=["error_bar_unconfirmed"],
        moderators={"task_type": "visuomotor", "perturbation_size_deg": "60"},
        citation=Citation(first_author="Bock", authors="Bock, O.", year=2005))


def _verdicts_for(record: EffectSizeRecord):
    from canopy.models import Candidate, DispersionType, Verdict

    verdicts, candidates = [], []
    for group, mean, sd, page in (("A", 44.67, 8.76, 3), ("B", 46.14, 6.11, 3)):
        cid = f"c-{group}"
        candidates.append(Candidate(
            candidate_id=cid, paper_id=record.paper_id, dataset_id=record.dataset_id,
            outcome_key=record.outcome_key, kind="group_stats", group=group, mean=mean,
            dispersion_value=sd, dispersion_type=DispersionType.SD, n=12, page=page,
            quote=f"{mean} ± {sd} s", crop_path=f"crops/{group}.png", unit="s",
            route="text", model="claude-opus-5"))
        verdicts.append(Verdict(
            dataset_id=record.dataset_id, outcome_key=record.outcome_key, group=group,
            agreement="agree", confidence="auto_accept", n=12, mean=mean, dispersion_value=sd,
            dispersion_type=DispersionType.SD, unit="s", route="text", candidate_ids=[cid],
            higher_is_better=False))
    return verdicts, candidates


def test_extraction_table_has_one_row_per_dataset_outcome_with_provenance(tmp_path, resolved_row):
    from canopy.report.tables import EXTRACTION_COLUMNS, extraction_table

    verdicts, candidates = _verdicts_for(resolved_row)
    out = extraction_table([resolved_row], tmp_path / "extraction_table",
                           verdicts=verdicts, candidates=candidates)
    assert set(out) == {"csv", "json", "xlsx"}
    assert all(p.exists() and p.stat().st_size > 0 for p in out.values())

    with out["csv"].open(newline="", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 1
    row = table[0]
    for column in EXTRACTION_COLUMNS:
        assert column in row, column
    assert row["dataset_id"] == "d1" and row["outcome_key"] == "late_adaptation"
    assert row["route"] == "text_mean_sd"
    assert row["mean_a"] == "44.67" and row["dispersion_a"] == "8.76"
    assert row["dispersion_type_a"] == "SD" and row["n_a"] == "12"
    assert "pooled SD" in row["conversion_chain"]
    assert row["confidence"] == "auto_accept"
    assert "error_bar_unconfirmed" in row["flags"]
    assert row["quote_a"].startswith("44.67") and row["crop_a"] == "crops/A.png"
    assert row["page_a"] == "3"
    assert row["mod_task_type"] == "visuomotor"

    payload = json.loads(out["json"].read_text())
    assert payload[0]["es"] == pytest.approx(-1.6758)
    assert payload[0]["routes_rejected"]["test_statistic"]


def test_extraction_table_survives_a_row_with_no_verdicts(tmp_path, resolved_row):
    from canopy.report.tables import extraction_table

    out = extraction_table([resolved_row], tmp_path / "t")
    row = list(csv.DictReader(out["csv"].open(newline="", encoding="utf-8")))[0]
    assert row["mean_a"] == "44.67"          # falls back to the record's own inputs
    assert row["quote_a"] == ""


def test_exclusions_table_normalises_the_reason_enum(tmp_path):
    from canopy.report.tables import EXCLUSION_REASONS, exclusions_table

    entries = [
        {"paper_id": "p1", "filename": "a.pdf", "stage": "map", "reason": "not_eligible",
         "quote": "young adults only", "decider": "mapper"},
        {"paper_id": "p2", "filename": "b.pdf", "stage": "ingest", "reason": "duplicate",
         "quote": "", "decider": "dedupe"},
        {"paper_id": "p3", "filename": "c.pdf", "stage": "resolve", "reason": "wat",
         "quote": "", "decider": "code"},
    ]
    out = exclusions_table(entries, tmp_path / "exclusions")
    rows = list(csv.DictReader(out["csv"].open(newline="", encoding="utf-8")))
    assert [r["reason"] for r in rows] == ["not_eligible", "duplicate", "other"]
    assert "wat" in rows[2]["reason_as_given"]
    assert set(EXCLUSION_REASONS) >= {"duplicate", "not_eligible", "no_usable_data",
                                      "needs_human", "not_convertible", "other"}


def test_leave_one_out_table_has_one_row_per_study(tmp_path, gold_rows, settings):
    from canopy.report.tables import leave_one_out_table

    out = leave_one_out_table(gold_rows, settings, tmp_path / "leave_one_out")
    rows = list(csv.DictReader(out["csv"].open(newline="", encoding="utf-8")))
    assert len(rows) == len(gold_rows)
    assert {"omitted_label", "omitted_dataset_id", "k", "estimate", "ci_low", "ci_high",
            "tau2", "I2"} <= set(rows[0])
    assert all(int(r["k"]) == len(gold_rows) - 1 for r in rows)


def test_sensitivity_set_covers_every_named_analysis(tmp_path, gold_rows, settings):
    from canopy.report.tables import SENSITIVITY_ANALYSES, sensitivity_analyses

    held = [gold_rows[0].model_copy(update={"confidence": "needs_human", "dataset_id": "dX"})]
    result = sensitivity_analyses(gold_rows, settings, needs_human_rows=held)
    names = {a["name"] for a in result["analyses"]}
    for required in SENSITIVITY_ANALYSES:
        assert any(n == required or n.startswith(f"{required}:") for n in names), required
    primary = next(a for a in result["analyses"] if a["name"] == "primary")
    assert primary["k"] == len(gold_rows)
    assert primary["estimate"] == pytest.approx(result["primary"]["estimate"])
    for analysis in result["analyses"]:
        assert "delta_vs_primary" in analysis and "k" in analysis and "description" in analysis


def test_sensitivity_outputs_write_json_and_small_multiples(tmp_path, gold_rows, settings):
    from canopy.report.tables import sensitivity_outputs

    out = sensitivity_outputs(gold_rows, settings, tmp_path / "sensitivity")
    assert out["json"].exists() and out["png"].exists()
    assert out["png"].stat().st_size > 0
    payload = json.loads(out["json"].read_text())
    assert payload["analyses"] and payload["primary"]["k"] == len(gold_rows)


def test_funnel_plot_uses_the_sample_size_predictor_when_n_is_known(tmp_path, gold_rows, settings):
    from canopy.report.tables import funnel_plot

    out = funnel_plot(gold_rows, settings, tmp_path / "funnel")
    assert out["png"].exists() and out["png"].stat().st_size > 0
    egger = json.loads(out["json"].read_text())["egger"]
    assert egger["predictor"] == "sqrt_inv_n"          # Pustejovsky-Rodgers, n known for every row
    assert egger["k"] == len(gold_rows)

    no_n = [r.model_copy(update={"n_a": None, "n_b": None}) for r in gold_rows]
    egger2 = json.loads(funnel_plot(no_n, settings, tmp_path / "funnel2")["json"].read_text())
    assert egger2["egger"]["predictor"] == "precision"


def test_prisma_counts_add_up(tmp_path):
    from canopy.report.tables import prisma_flow

    counts = {"files": 12, "duplicates_removed": 2, "unique_papers": 10,
              "papers_excluded": 3, "eligible_papers": 7, "datasets": 11,
              "datasets_excluded": 2, "included_datasets": 9, "included_papers": 6,
              "exclusion_reasons": {"not_eligible": 2, "no_usable_data": 1}}
    out = prisma_flow(counts, tmp_path / "prisma")
    payload = json.loads(out["json"].read_text())
    assert payload["files"] - payload["duplicates_removed"] == payload["unique_papers"]
    assert payload["unique_papers"] - payload["papers_excluded"] == payload["eligible_papers"]
    assert payload["datasets"] - payload["datasets_excluded"] == payload["included_datasets"]
    assert payload["consistent"] is True
    assert out["png"].exists() and out["png"].stat().st_size > 0


def test_prisma_flow_reports_an_inconsistent_chain_instead_of_hiding_it(tmp_path):
    from canopy.report.tables import prisma_flow

    out = prisma_flow({"files": 12, "duplicates_removed": 2, "unique_papers": 9,
                       "papers_excluded": 3, "eligible_papers": 7, "datasets": 11,
                       "datasets_excluded": 2, "included_datasets": 9, "included_papers": 6},
                      tmp_path / "prisma")
    payload = json.loads(out["json"].read_text())
    assert payload["consistent"] is False
    assert any("unique_papers" in problem for problem in payload["problems"])


# --------------------------------------------------------------------------- provenance
def _bock_quote(paper) -> tuple[int, str]:
    """A verbatim phrase from the real Bock PDF, so the crop has something true to find."""
    for page in paper.pages:
        text = paper.page_text(page.number)
        for line in text.splitlines():
            words = line.split()
            if len(words) >= 8 and all(w.isascii() for w in words):
                return page.number, " ".join(words[:8])
    raise AssertionError("no usable line in the ingested paper")


def test_quote_crop_finds_the_quote_and_highlights_it(tmp_path, paper):
    from canopy.report.provenance import quote_crop

    page, quote = _bock_quote(paper)
    out = quote_crop(paper, page, quote, tmp_path / "crop.png")
    assert out["matched"] is True
    assert out["path"].exists() and out["path"].stat().st_size > 0
    assert out["page"] == page
    x0, y0, x1, y1 = out["bbox_pt"]
    assert x1 > x0 and y1 > y0
    assert out["matched_words"] >= 6


def test_quote_crop_reports_a_quote_it_cannot_find_instead_of_pretending(tmp_path, paper):
    from canopy.report.provenance import quote_crop

    out = quote_crop(paper, 1, "zzqq nonexistent phrase that is not in this paper at all",
                     tmp_path / "crop.png")
    assert out["matched"] is False
    assert out["path"].exists()          # the page is still shown, just without a highlight
    assert out["note"]


def test_provenance_bundle_writes_one_entry_per_candidate(tmp_path, paper):
    from canopy.models import Candidate, DispersionType
    from canopy.report.provenance import provenance_bundle

    page, quote = _bock_quote(paper)
    cands = [Candidate(candidate_id="c1", paper_id=paper.sha256, dataset_id="d1",
                       outcome_key="late_adaptation", kind="group_stats", group="A", mean=42.5,
                       dispersion_value=6.9, dispersion_type=DispersionType.SD, n=12, page=page,
                       quote=quote, route="text", model="claude-opus-5"),
             Candidate(candidate_id="c2", paper_id=paper.sha256, dataset_id="d1",
                       outcome_key="late_adaptation", kind="group_stats", group="B", mean=30.0,
                       page=page, quote="", route="figure", model="claude-opus-5")]
    bundle = provenance_bundle(paper, cands, tmp_path / "provenance")
    assert set(bundle["entries"]) == {"c1", "c2"}
    assert bundle["entries"]["c1"]["matched"] is True
    assert Path(bundle["entries"]["c1"]["crop"]).exists()
    assert bundle["json"].exists()
    payload = json.loads(bundle["json"].read_text())
    assert payload["c1"]["mean"] == 42.5 and payload["c1"]["quote"] == quote


# --------------------------------------------------------------------------- methods figure
def test_route_counts_are_percentages_of_datasets(resolved_row):
    from canopy.report.methods_fig import route_counts

    rows = [resolved_row.model_copy(update={"dataset_id": f"d{i}", "route": route})
            for i, route in enumerate(["text_mean_sd", "text_mean_se_ci", "figure", "figure",
                                       "table", "test_statistic", "reported_d", "not_convertible"])]
    counts = route_counts(rows)
    assert counts["figure"]["n"] == 2
    assert counts["text"]["n"] == 2
    assert sum(v["n"] for v in counts.values()) == len(rows)
    assert pytest.approx(sum(v["pct"] for v in counts.values()), abs=1e-6) == 100.0


def test_methods_figure_renders_with_thumbnails(tmp_path, paper, resolved_row):
    from canopy.models import Candidate, DispersionType
    from canopy.report.methods_fig import methods_figure, route_examples

    page, quote = _bock_quote(paper)
    rows = [resolved_row.model_copy(update={"dataset_id": "d1", "route": "text_mean_sd"}),
            resolved_row.model_copy(update={"dataset_id": "d2", "route": "figure"})]
    cands = [Candidate(candidate_id="c1", paper_id=paper.sha256, dataset_id="d1",
                       outcome_key="late_adaptation", kind="group_stats", group="A", mean=42.5,
                       dispersion_value=6.9, dispersion_type=DispersionType.SD, page=page,
                       quote=quote, route="text", model="claude-opus-5")]
    examples = route_examples(rows, candidates=cands, papers=[paper],
                              out_dir=tmp_path / "thumbs", max_examples=3)
    assert examples["text"], "the text route should have at least one thumbnail"
    out = methods_figure(rows, tmp_path / "methods_routes", examples=examples)
    assert out["png"].exists() and out["png"].stat().st_size > 0
    svg = out["svg"].read_text(encoding="utf-8")
    assert "figure" in svg and "text" in svg


def test_methods_figure_works_with_no_thumbnails_at_all(tmp_path, resolved_row):
    from canopy.report.methods_fig import methods_figure

    out = methods_figure([resolved_row], tmp_path / "methods_routes")
    assert out["png"].exists() and out["png"].stat().st_size > 0


# --------------------------------------------------------------------------- per-outcome outputs
@pytest.fixture
def small_rows(resolved_row) -> list[EffectSizeRecord]:
    values = [(-1.68, 0.4801), (-0.62, 0.3200), (-0.20, 0.2800), (0.15, 0.3300), (0.65, 0.4700)]
    rows = []
    for i, (es, se) in enumerate(values):
        rows.append(resolved_row.model_copy(update={
            "paper_id": f"p{i}", "cluster_id": f"p{i}", "dataset_id": f"d{i}",
            "es": es, "d": es, "se": se, "var": se * se,
            "ci_low": es - 1.96 * se, "ci_high": es + 1.96 * se,
            "route": ["text_mean_sd", "figure", "table", "test_statistic", "reported_d"][i],
            "analysis_metric": "endpoint",
            "citation": Citation(first_author=f"Author{i}", year=2000 + i)}))
    return rows


def test_write_outcome_outputs_writes_the_whole_per_outcome_set(tmp_path, small_rows, outcome,
                                                                settings):
    from canopy.report import write_outcome_outputs

    pooled = random_effects([r.es for r in small_rows], [r.var for r in small_rows],
                            method=settings.tau2_method, hakn=settings.hakn)
    out = write_outcome_outputs(tmp_path, outcome, small_rows, pooled, settings)
    for key in ("forest_png", "forest_svg", "forest_pdf", "extraction_csv", "extraction_json",
                "extraction_xlsx", "leave_one_out_csv", "sensitivity_json", "sensitivity_png",
                "funnel_png", "funnel_json", "pooled_json"):
        assert key in out, key
        assert out[key].exists() and out[key].stat().st_size > 0, key
    assert out["forest_png"] == tmp_path / "results" / outcome.key / "forest.png"
    pooled_json = json.loads(out["pooled_json"].read_text())
    assert pooled_json["estimate"] == pytest.approx(pooled.estimate)
    assert pooled_json["k"] == len(small_rows)
    assert pooled_json["k_papers"] == len(small_rows)
    assert pooled_json["n_needs_human"] == 0
    assert pooled_json["settings"]["tau2_method"] == settings.tau2_method


def test_write_outcome_outputs_still_writes_when_there_is_nothing_to_pool(tmp_path, resolved_row,
                                                                          outcome, settings):
    from canopy.report import write_outcome_outputs

    held = [resolved_row.model_copy(update={"confidence": "needs_human"})]
    out = write_outcome_outputs(tmp_path, outcome, [], None, settings, needs_human_rows=held)
    assert out["pooled_json"].exists()
    payload = json.loads(out["pooled_json"].read_text())
    assert payload["k"] == 0 and payload["n_needs_human"] == 1
    assert payload["note"]
    assert "forest_png" not in out                       # nothing to draw, and it says so


# --------------------------------------------------------------------------- HTML report
def _manifest(tmp_path, protocol_hash="abc123"):
    from canopy.models import PaperStatus, RunManifest, StatsSettings

    return RunManifest(
        run_id="run-1", created_at="2026-08-15T10:00:00+00:00", protocol_hash=protocol_hash,
        protocol_path="examples/protocols/aging_sensorimotor_adaptation.yaml",
        canopy_version="0.1.0", git_commit="deadbee",
        papers=[PaperStatus(paper_id="p0", filename="a.pdf", status="resolved", eligible=True,
                            cost_usd=0.4, seconds=31.0),
                PaperStatus(paper_id="p1", filename="b.pdf", status="excluded", eligible=False,
                            cost_usd=0.1, seconds=9.0)],
        settings=StatsSettings(profile="cisneros2024"), models={"primary": "claude-opus-5"},
        cost_usd=0.5, n_llm_calls=17, seconds=40.0)


def test_html_report_links_every_artefact_and_escapes_paper_text(tmp_path, small_rows, outcome,
                                                                 settings, protocol):
    from canopy.report import write_html_report, write_outcome_outputs

    pooled = random_effects([r.es for r in small_rows], [r.var for r in small_rows],
                            method=settings.tau2_method)
    outputs = write_outcome_outputs(tmp_path, outcome, small_rows, pooled, settings)
    manifest = _manifest(tmp_path, protocol.hash())
    queue = [{"paper_id": "p0", "dataset_id": "d1", "outcome_key": outcome.key, "group": "A",
              "reason": "<script>alert(1)</script> verifier refuted",
              "candidates": [{"value": 1.0}], "impact_abs_delta_pooled": 0.21},
             {"paper_id": "p1", "dataset_id": "d2", "outcome_key": outcome.key, "group": "B",
              "reason": "vote disagreed", "candidates": [], "impact_abs_delta_pooled": 0.05}]
    out = write_html_report(tmp_path, manifest, protocol,
                            results={outcome.key: {"pooled": pooled, "outputs": outputs,
                                                   "rows": small_rows, "needs_human_rows": []}},
                            review_queue=queue,
                            exclusions=[{"paper_id": "p1", "filename": "b.pdf", "stage": "map",
                                         "reason": "not_eligible", "quote": "young only",
                                         "decider": "mapper"}])
    html = out["html"].read_text(encoding="utf-8")
    assert out["html"].name == "report.html"
    assert "results/late_adaptation/forest.png" in html
    assert "results/late_adaptation/extraction_table.csv" in html
    assert "methods.md" in html
    assert "&lt;script&gt;" in html and "<script>alert(1)</script>" not in html
    assert "0.21" in html and "vote disagreed" in html          # the review queue is shown
    assert "not_eligible" in html                                # …and so are the exclusions
    assert "http://" not in html and "https://" not in html.replace(
        'xmlns="http://www.w3.org/', "")                          # no external requests


def test_methods_paragraph_uses_only_manifest_numbers(tmp_path, small_rows, outcome, settings,
                                                      protocol):
    from canopy.report import methods_paragraph

    pooled = random_effects([r.es for r in small_rows], [r.var for r in small_rows],
                            method=settings.tau2_method)
    manifest = _manifest(tmp_path, protocol.hash())
    text = methods_paragraph(manifest, protocol,
                             results={outcome.key: {"pooled": pooled, "rows": small_rows,
                                                    "needs_human_rows": []}})
    assert "REML" in text and "Cohen" in text
    assert f"k = {pooled.k}" in text
    assert protocol.hash()[:12] in text
    assert "0.1.0" in text and "deadbee" in text
    assert f"{pooled.estimate:.2f}" in text


def test_human_review_queue_is_sorted_by_impact(tmp_path):
    from canopy.report import human_review_table

    rows = [{"paper_id": "p1", "reason": "a", "impact_abs_delta_pooled": 0.02},
            {"paper_id": "p2", "reason": "b", "impact_abs_delta_pooled": 0.30},
            {"paper_id": "p3", "reason": "c", "impact_abs_delta_pooled": None}]
    html = human_review_table(rows)
    assert html.index("p2") < html.index("p1") < html.index("p3")


def test_funnel_falls_back_when_every_study_has_the_same_group_sizes(tmp_path, small_rows,
                                                                     settings):
    """A constant √(1/n_A+1/n_B) makes the Pustejovsky-Rodgers design matrix singular."""
    from canopy.report.tables import funnel_plot

    same = [r.model_copy(update={"n_a": 12, "n_b": 12}) for r in small_rows]
    out = funnel_plot(same, settings, tmp_path / "funnel")
    payload = json.loads(out["json"].read_text())
    assert payload["egger"] is not None
    assert payload["egger"]["predictor"] == "precision"
    assert "could not be fitted" in payload["egger_note"]
    assert out["png"].exists() and out["png"].stat().st_size > 0


def test_extraction_xlsx_survives_control_characters_from_a_real_pdf(tmp_path, resolved_row):
    """PDF text routinely contains characters XLSX forbids; the CSV must still keep them."""
    from canopy.report.tables import extraction_table

    dirty = resolved_row.model_copy(update={
        "notes": "SD \x03 3.7 years \x0b as printed",
        "conversion_chain": "pooled SD \x01 = 7.5"})
    out = extraction_table([dirty], tmp_path / "t")
    assert out["xlsx"].exists() and out["xlsx"].stat().st_size > 0
    row = list(csv.DictReader(out["csv"].open(newline="", encoding="utf-8")))[0]
    assert "\x03" in row["notes"]                     # the CSV keeps what the extractor read


def test_route_example_filenames_are_safe_for_a_url(tmp_path, paper, resolved_row):
    """Candidate ids carry `:` and `#`; a thumbnail the HTML report links must not."""
    from canopy.models import Candidate
    from canopy.report.methods_fig import route_examples

    page, quote = _bock_quote(paper)
    rows = [resolved_row.model_copy(update={"dataset_id": "sha:d1", "route": "text_mean_sd",
                                            "paper_id": paper.sha256})]
    cands = [Candidate(candidate_id="sha:d1:late:A:text:table_first:claude-opus-5#1",
                       paper_id=paper.sha256, dataset_id="sha:d1", outcome_key="late_adaptation",
                       kind="group_stats", group="A", mean=1.0, page=page, quote=quote,
                       route="text")]
    examples = route_examples(rows, candidates=cands, papers=[paper], out_dir=tmp_path / "t")
    for paths in examples.values():
        for path in paths:
            assert ":" not in path.name and "#" not in path.name


def test_footer_does_not_claim_a_prediction_interval_it_could_not_compute(gold_rows, settings):
    """With k = 2 the prediction interval has df = 0; saying so beats printing a df of 0."""
    from canopy.report.theme import conventions_footer

    two = gold_rows[:2]
    pooled = random_effects([r.es for r in two], [r.var for r in two],
                            method=settings.tau2_method)
    lines = "\n".join(conventions_footer(settings, pooled, k_papers=2, k_datasets=2))
    assert "no prediction interval" in lines and "k = 2" in lines

    many = gold_rows[:6]
    pooled_many = random_effects([r.es for r in many], [r.var for r in many],
                                 method=settings.tau2_method)
    lines = "\n".join(conventions_footer(settings, pooled_many, k_papers=6, k_datasets=6))
    assert "prediction interval: HTS" in lines
