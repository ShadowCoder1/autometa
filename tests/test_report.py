"""Task 11 — outputs: forest plot, tables, PRISMA, sensitivity, methods figure, HTML report.

The forest plot is exercised against the reference review's own published effect sizes
(`validation/reference/cisneros2024/late_gsheet.csv`, k = 50). That file is **validation input
only**: nothing in `canopy/` reads it, and no test here lets a number from it reach the pipeline.
It is used because a renderer that survives fifty real rows — long author strings, missing
moderators, a wide effect range — has been tested on something.
"""
from __future__ import annotations

import csv
import html as _html
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

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


def test_footer_counts_needs_human_and_not_convertible_separately(tmp_path, gold_rows,
                                                                  pooled_gold, outcome, settings):
    """"Held for a human" and "no route could convert it" are different findings, so both print."""
    from canopy.report.forest import forest_plot
    from canopy.report.theme import conventions_footer

    lines = "\n".join(conventions_footer(settings, pooled_gold, k_papers=50, k_datasets=50,
                                         n_excluded=3, n_not_convertible=2))
    assert "3 rows excluded (needs_human 1, not convertible 2)" in lines

    held = [gold_rows[0].model_copy(update={"confidence": "needs_human", "dataset_id": "dQ"}),
            gold_rows[1].model_copy(update={"confidence": "needs_human", "dataset_id": "dR",
                                            "route": "not_convertible",
                                            "not_convertible_reason": "no dispersion was reported"})]
    svg = forest_plot(gold_rows, pooled_gold, outcome, settings,
                      tmp_path / "forest", needs_human_rows=held)["svg"].read_text()
    assert "2 rows excluded (needs_human 1, not convertible 1)" in svg


def test_the_override_marker_survives_a_long_author_label(tmp_path, gold_rows, pooled_gold,
                                                          outcome, settings):
    """The △ was appended to the label BEFORE `elide(…, 17)`, so a long label silently lost it.

    Which is the worst failure a marker can have: the rows whose author strings are longest are
    exactly the rows on which the plot then claimed nothing had been overridden. The footer's key
    kept saying a △ meant an override, with no △ anywhere on the figure.
    """
    from canopy.report.forest import _left_columns, forest_layout, forest_plot
    from canopy.report.theme import OVERRIDE_MARK

    name = "Vandenberghe-Lindqvist"
    assert len(name) > 17                              # longer than `labels.MAX_LABEL_CH`
    marked = gold_rows[0].model_copy(update={
        "flags": ["human_override"],
        "citation": Citation(first_author=name, authors=name, year=2011)})
    rows = [marked, *gold_rows[1:]]

    layout = forest_layout(rows, pooled_gold, settings)
    cells = _left_columns(layout)[0].cells
    mine = next(cell for cell, row in zip(cells, layout.all_rows) if row.overridden)
    assert mine.startswith("Vandenberghe-Lin") and mine.endswith(OVERRIDE_MARK)
    assert sum(1 for cell in cells if OVERRIDE_MARK in cell) == 1

    svg = forest_plot(rows, pooled_gold, outcome, settings,
                      tmp_path / "forest")["svg"].read_text(encoding="utf-8")
    assert mine in svg                                 # …and it is on the figure, not just the cell


def test_is_overridden_reads_the_flag_and_not_the_route(gold_rows):
    """`record.route == "human_override"` was a clause that could never fire (review finding).

    A row's route is a resolver route name — `text_mean_sd`, `figure:*`, `composite`,
    `not_convertible` — and nothing writes an override's own name into it. An override is recorded
    as the `human_override` FLAG, by `pipeline.overrides._rebuild_row`, on every row any override
    rebuilt. So the clause was deleted rather than corrected: it tested nothing, it hid that it
    tested nothing, and a route is not where this question is answered.
    """
    from canopy.report.theme import OVERRIDE_FLAG, is_overridden

    assert is_overridden(gold_rows[0].model_copy(update={"flags": [OVERRIDE_FLAG]}))
    assert not is_overridden(gold_rows[0])
    assert not is_overridden(gold_rows[0].model_copy(update={"route": OVERRIDE_FLAG}))


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


def test_an_overridden_arm_never_carries_the_replaced_candidate_quote(tmp_path, resolved_row):
    """A reviewer's number printed beside a reading's page and quote is a FALSE provenance claim.

    `verdict.candidate_ids` still names the candidates behind the value an override replaced, and a
    `value` override carries no quote, page or crop of its own — so the cell used to print the
    corrected mean next to the verbatim sentence, the page and the digitiser's overlay of a number
    that is no longer there. Two real rows of `runs/et_with_dbs` did that: one printed `mean_b =
    35.4` beside the sentence printing 27.1 for the OTHER arm, and one printed a reviewer's swapped
    series beside the overlay of the box it overruled. A reader checking such a quote concludes the
    number is wrong, when it is the quote that does not belong.

    The arm a human answered says so instead. The arm nobody answered keeps its provenance: blanking
    that too would throw away a quote that is still true of the number beside it.
    """
    from canopy.report.tables import HUMAN_PROVENANCE, extraction_table

    verdicts, candidates = _verdicts_for(resolved_row)
    answered = next(v for v in verdicts if v.group == "A")
    answered.mean = 41.0                  # what `overrides._apply_value` does to the answered cell
    answered.overridden_by_human = True
    answered.override_justification = "Table 2 prints 41.0 for the older group, not the text"
    out = extraction_table([resolved_row.model_copy(update={"flags": ["human_override"]})],
                           tmp_path / "t", verdicts=verdicts, candidates=candidates)
    row = list(csv.DictReader(out["csv"].open(newline="", encoding="utf-8")))[0]

    assert row["mean_a"] == "41"
    assert (row["page_a"], row["crop_a"], row["overlay_a"]) == ("", "", "")
    assert "44.67" not in row["quote_a"]               # the replaced reading's own sentence
    assert row["quote_a"] == f"{HUMAN_PROVENANCE}: {answered.override_justification}"
    assert row["page_b"] == "3" and row["quote_b"].startswith("46.14")
    assert row["crop_b"] == "crops/B.png"


def test_extraction_table_writes_an_infinite_value_instead_of_raising(tmp_path, resolved_row):
    """A variance that overflowed is a finding; `int(inf)` raises, so it must never be rounded."""
    from canopy.report.tables import extraction_table

    broken = resolved_row.model_copy(update={"var": float("inf"), "se": float("inf"),
                                             "digitization_var_share": float("-inf"),
                                             "ci_high": float("nan")})
    out = extraction_table([broken], tmp_path / "t")
    row = list(csv.DictReader(out["csv"].open(newline="", encoding="utf-8")))[0]
    assert row["var"] == "inf" and row["se"] == "inf"
    assert row["digitization_var_share"] == "-inf"
    assert row["ci_high"] == ""                        # NaN is "no value", and prints as none


def test_extraction_table_marks_which_rows_were_pooled(tmp_path, resolved_row):
    """The table keeps every row — the ones an aggregation replaced included — and says which."""
    from canopy.report.tables import extraction_table

    kept = resolved_row.model_copy(update={"dataset_id": "d1+d2"})
    member = resolved_row.model_copy(update={"dataset_id": "d1"})
    out = extraction_table([kept, member], tmp_path / "t", primary=[kept])
    rows = {r["dataset_id"]: r["primary_row"]
            for r in csv.DictReader(out["csv"].open(newline="", encoding="utf-8"))}
    assert rows == {"d1+d2": "true", "d1": "false"}

    unknown = extraction_table([kept], tmp_path / "u")   # no caller opinion: no claim made
    assert list(csv.DictReader(
        unknown["csv"].open(newline="", encoding="utf-8")))[0]["primary_row"] == ""


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
              "datasets_excluded": 2, "included_datasets": 9, "included_papers": 6, "papers_with_no_rows": 1,
              "exclusion_reasons": {"not_eligible": 2, "no_usable_data": 1}}
    out = prisma_flow(counts, tmp_path / "prisma")
    payload = json.loads(out["json"].read_text())
    assert payload["files"] - payload["duplicates_removed"] == payload["unique_papers"]
    assert payload["unique_papers"] - payload["papers_excluded"] == payload["eligible_papers"]
    assert payload["datasets"] - payload["datasets_excluded"] == payload["included_datasets"]
    assert payload["consistent"] is True
    assert out["png"].exists() and out["png"].stat().st_size > 0


def test_a_paper_that_was_eligible_and_left_no_row_breaks_the_chain_by_name(tmp_path):
    """Three runs printed `consistent: true` beside `eligible_papers: 3, included_papers: 2`.

    Every link of the chain was an identity of how the counts were built, so a missing paper
    could not make it false. `included_papers` is counted from the effect-size records and
    `eligible_papers` from the manifest; the link between them is the one that can fail.
    """
    from canopy.report.tables import prisma_flow

    counts = {"files": 3, "duplicates_removed": 0, "unique_papers": 3, "papers_excluded": 0,
              "eligible_papers": 3, "papers_with_no_rows": 0, "included_papers": 2,
              "datasets": 4, "datasets_excluded": 1, "included_datasets": 3,
              "exclusion_reasons": {}}
    payload = json.loads(prisma_flow(counts, tmp_path / "prisma")["json"].read_text())
    assert payload["consistent"] is False
    assert any("eligible_papers (3)" in problem and "included_papers (2)" in problem
               for problem in payload["problems"]), payload["problems"]
    counts["papers_with_no_rows"] = 1                     # …and named, it adds up again
    payload = json.loads(prisma_flow(counts, tmp_path / "prisma2")["json"].read_text())
    assert payload["consistent"] is True


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


# ------------------------------------------------------- DECISION A: the best-guess line
def _nine_outputs(tmp_path, key, *, held_update=None, one_row_per_paper=False):
    """Write one outcome of the `runs/nine` fixture into `tmp_path`, exactly as a run does.

    `held_update(record) -> dict | None` may hand a held row the fields a later stage will give
    it (a value a categorical point read produces, a direction a ballot settled); everything else
    is the record the resolver actually wrote.
    """
    from canopy.pipeline.run import _split_rows
    from canopy.report import write_outcome_outputs
    from canopy.report.tables import pool_rows
    from tests.helpers import nine

    protocol = nine.protocol()
    protocol.stats.one_row_per_paper = one_row_per_paper
    split = _split_rows(nine.records(key), protocol.stats)
    held = [r.model_copy(update=(held_update(r) or {})) if held_update else r for r in split.held]
    cells: dict = {}
    out = write_outcome_outputs(tmp_path, protocol.outcome(key), split.primary,
                                pool_rows(split.primary, protocol.stats), protocol.stats,
                                needs_human_rows=held, all_rows=split.every,
                                primary_pre_agg=split.primary_pre_agg, best_guess_cells=cells,
                                protocol=protocol, warnings=[])
    return out, json.loads(out["pooled_json"].read_text()), cells


def _same(fixture, written, path=""):
    """Every key the fixture recorded is present and equal — NaN included, subsets allowed.

    Lists are compared as multisets. `pooled.json`'s lists are all per-row (the held ids, `yi`,
    `vi`, the weights) and their ORDER is the order the run's papers finished — they are read
    here in path order instead — so a permutation of them is a property of concurrency, not of
    the analysis. Every pooled number, which is what the key claim is about, is compared exactly.
    """
    if isinstance(fixture, dict):
        assert isinstance(written, dict), path
        for key, value in fixture.items():
            assert key in written, f"{path}.{key} disappeared"
            _same(value, written[key], f"{path}.{key}")
        return
    if isinstance(fixture, float) and math.isnan(fixture):
        # the fixture predates fix round 2 (MINOR 21): a quantity with no estimable value was
        # written as the bare token `NaN`, which no JSON parser outside Python accepts. It is
        # `null` now — the same absence, spelled in the language of the file — and nothing else
        # may stand where the fixture recorded one.
        assert written is None or (isinstance(written, float) and math.isnan(written)), path
        return
    if isinstance(fixture, list):
        assert isinstance(written, list) and len(fixture) == len(written), path
        for a, b in zip(sorted(fixture), sorted(written)):
            _same(a, b, f"{path}[]")
        return
    assert fixture == written, path


def test_pooled_json_keeps_strict_keys_at_top_level_and_adds_best_guess(tmp_path):
    """A0: the second line is additive. Every number the primary analysis published is still

    published, at the same key, with the same value — the fixture's own `pooled.json` is the
    witness, because it was written before this feature existed.
    """
    from tests.helpers import nine

    _, payload, _ = _nine_outputs(tmp_path, "late_adaptation")
    fixture = json.loads(
        (nine.NINE / "results" / "late_adaptation" / "pooled.json").read_text())
    _same(fixture, payload)
    assert payload["analysis_lines"] == ["strict", "best_guess"]
    assert payload["best_guess"]["k"] >= 5
    assert payload["best_guess"]["k"] >= payload["k"]
    assert payload["best_guess"]["note"]


def test_every_held_row_is_named_in_exactly_one_of_the_best_guess_lists(tmp_path):
    _, payload, _ = _nine_outputs(tmp_path, "late_adaptation")
    added = {a["dataset_id"] for a in payload["best_guess"]["added"]}
    held_back = {n["dataset_id"] for n in payload["best_guess"]["not_added"]}
    assert not (added & held_back)
    assert added | held_back == set(payload["needs_human_dataset_ids"])
    assert all(n["veto"] and n["reason"] for n in payload["best_guess"]["not_added"])


def test_forest_best_guess_written_only_when_rows_were_added(tmp_path):
    """No added row means the best-guess line IS the strict line, and it gets no second plot."""
    late, late_payload, _ = _nine_outputs(tmp_path / "late", "late_adaptation")
    assert late_payload["best_guess"]["n_added"] >= 1
    assert late["forest_best_guess_png"].exists()

    aft, aft_payload, _ = _nine_outputs(tmp_path / "aft", "aftereffect")
    assert aft_payload["best_guess"]["n_added"] == 0
    assert "forest_best_guess_png" not in aft
    # and `k` means on both lines what it means on the strict one: what was POOLED. The line's
    # size is a different question with its own key, so the two ks can be read side by side.
    assert aft_payload["k"] == 0 and aft_payload["best_guess"]["k"] == 0
    assert aft_payload["best_guess"]["k_rows"] == 1


def test_aftereffect_has_best_guess_forest_without_strict_forest(tmp_path):
    """Strict k = 1, best-guess k = 3: the outcome no primary analysis can report, reported.

    The two Vachon aftereffect rows are given the value D3's categorical point read produces
    (acceptance 5) over the group sizes the SAME datasets carry on their late-adaptation rows —
    the paper's own n, not a number invented for the test.
    """
    from canopy.stats.effect_sizes import se_smd
    from tests.helpers import nine

    reads = {"b7523a41b03a:d1": 0.13, "b7523a41b03a:d2": 1.24}

    def point_read(record):
        if record.dataset_id not in reads:
            return None
        late = nine.record(record.dataset_id, "late_adaptation")
        es = reads[record.dataset_id]
        se = se_smd(es, late.n_a, late.n_b)
        return {"route": "figure", "es": es, "se": se, "var": se * se, "n_a": late.n_a,
                "n_b": late.n_b, "not_convertible_reason": "",
                "flags": [f for f in record.flags if f != "not_convertible"]
                         + ["categorical_point_read"]}

    out, payload, _ = _nine_outputs(tmp_path, "aftereffect", held_update=point_read)
    assert payload["k"] < 2 and "forest_png" not in out
    assert payload["best_guess"]["k"] >= 3 and payload["best_guess"]["n_added"] == 2
    assert out["forest_best_guess_png"].exists()
    assert "Best guess" in out["forest_best_guess_svg"].read_text(encoding="utf-8")


def test_each_line_aggregates_its_own_rows_and_a_mixed_composite_is_a_guess(tmp_path):
    """`one_row_per_paper` runs once per line, over that line's members (DECISION A).

    Buch contributes one strict row to neither line and two held rows to the best guess; the
    composite that stands for the paper is therefore a best-guess row wholesale, and says which
    of its members were guessed.
    """
    out, payload, cells = _nine_outputs(tmp_path, "late_adaptation", one_row_per_paper=True)
    composites = [a for a in payload["best_guess"]["added"] if "+" in a["dataset_id"]]
    buch = next(c for c in composites if c["dataset_id"].startswith("592b3b55a318"))
    for member in buch["dataset_id"].split("+"):
        assert member in buch["reason"]
        assert cells[(member, "late_adaptation")]["in_best_guess"] is True
    # …and a composite folding a STRICT row into a disputed admission is a guess wholesale
    # too, its reason naming the guessed member alone — the strict member is in the line
    # already and is not a guess to explain
    mixed = next(c for c in composites if c["dataset_id"].startswith("b7523a41b03a"))
    assert "b7523a41b03a:d2 (disputed_reading_guess)" in mixed["reason"]
    assert payload["k"] == 2 and payload["best_guess"]["k"] == 4
    assert out["forest_best_guess_png"].exists()


def test_extraction_columns_have_best_guess_before_moderators(tmp_path):
    from canopy.report.tables import EXTRACTION_COLUMNS

    added = ("in_best_guess", "best_guess_rule", "best_guess_reason", "best_guess_es",
             "best_guess_se")
    assert EXTRACTION_COLUMNS[-len(added):] == added      # nothing but `mod_*` follows them

    out, _, cells = _nine_outputs(tmp_path, "late_adaptation")
    table = {r["dataset_id"]: r for r in json.loads(out["extraction_json"].read_text())}
    columns = list(table["5039533c85ef:d1"])
    assert columns.index("best_guess_se") < min(i for i, c in enumerate(columns)
                                                if c.startswith("mod_"))
    guessed = table["5039533c85ef:d1"]
    assert guessed["in_best_guess"] is True
    assert guessed["best_guess_rule"] == "low_confidence_value" and guessed["best_guess_reason"]
    assert guessed["best_guess_es"] == guessed["es"]
    disputed = table["b7523a41b03a:d2"]
    assert disputed["in_best_guess"] is True
    assert disputed["best_guess_rule"] == "disputed_reading_guess"
    assert disputed["best_guess_es"] == disputed["es"]        # at its own value, never rescaled
    assert table["b511dbb76fa6:d1"]["in_best_guess"] is True      # a strict row is in the line
    assert table["b511dbb76fa6:d1"]["best_guess_rule"] == ""      # but it is not a guess


def test_include_needs_human_note_counts_what_it_could_not_include(tmp_path):
    out, _, _ = _nine_outputs(tmp_path, "late_adaptation")
    payload = json.loads(out["sensitivity_json"].read_text())
    entry = next(a for a in payload["analyses"] if a["name"] == "include_needs_human")
    assert "4 of 8 held row(s) could not be included" in entry["note"]
    assert entry["k"] == 2 + 4                            # the four that carry a usable value
    # on THIS fixture the two ks now coincide (the disputed row enters both); the sets still
    # differ in general — a `row_refusal` or unsigned row with a value is included here and
    # vetoed there, which is the difference between "add everything held" and "what a rule admits"
    best_guess = next(a for a in payload["analyses"] if a["name"] == "best_guess")
    assert best_guess["k"] == 6 and best_guess["delta_vs_primary"] is not None


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


def test_methods_paragraph_still_reports_an_outcome_the_protocol_no_longer_names(tmp_path,
                                                                                 small_rows,
                                                                                 settings,
                                                                                 protocol):
    """A run re-pooled against an edited protocol must not lose its pooled result silently."""
    from canopy.report import methods_paragraph

    pooled = random_effects([r.es for r in small_rows], [r.var for r in small_rows],
                            method=settings.tau2_method)
    text = methods_paragraph(_manifest(tmp_path, protocol.hash()), protocol,
                             results={"an_outcome_this_protocol_dropped": {
                                 "pooled": pooled, "rows": small_rows, "needs_human_rows": []}})
    assert "an_outcome_this_protocol_dropped" in text
    assert f"k = {pooled.k} datasets" in text
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


def test_the_review_table_prints_the_margin_and_the_line_it_belongs_to():
    """Review L2: the HTML column C11 added had no test at all. A cell whose bucket the score did
    not decide prints an em dash rather than a distance from a line it never approached (M3)."""
    from canopy.report.html import human_review_table

    table = human_review_table([
        {"paper_id": "p1", "dataset_id": "d1", "outcome_key": "late_adaptation", "group": "A",
         "reason": "one route only", "confidence_margin": 0.0100,
         "nearest_boundary": "auto_accept", "impact_abs_delta_pooled": 0.2, "candidates": []},
        {"paper_id": "p1", "dataset_id": "d2", "outcome_key": "late_adaptation", "group": "B",
         "reason": "the direction of this measure is unresolved", "confidence_margin": None,
         "nearest_boundary": "", "impact_abs_delta_pooled": 0.1, "candidates": []}])
    assert ">Margin<" in table
    assert "0.0100 from auto_accept" in table
    held = table[table.index("d2"):]
    assert "from auto_accept" not in held and "\u2014" in held


def test_the_footer_says_z_when_the_hartung_knapp_adjustment_had_nothing_to_adjust(settings):
    """Review M4. The R2 guard falls back to the ordinary random-effects SE and the normal `z`
    quantile, and records why — and the footer went on printing "CI from t(k-1), Hartung-Knapp"
    over a normal-z interval, which is a statement about the method that is simply false. A note
    nobody reads, beside a sentence the fallback falsifies, is worse than no note."""
    import numpy as np

    from canopy.report.theme import conventions_footer

    yi, vi = np.array([-0.5, -0.5, -0.5]), np.array([0.04, 0.04, 0.04])
    fell_back = random_effects(yi, vi, method=settings.tau2_method, hakn=True)
    assert fell_back.hakn_fallback and fell_back.hakn is True     # the premise: it fell back
    lines = "\n".join(conventions_footer(settings, fell_back, k_papers=3, k_datasets=3))
    assert "CI from z" in lines
    assert "nothing to adjust" in lines and fell_back.hakn_fallback in lines
    assert "CI from t(k\u22121), Hartung\u2013Knapp" not in lines

    # the control: a pool where the adjustment applied still says so
    applied = random_effects(np.array([-0.4, -0.6, -0.2]), vi, method=settings.tau2_method,
                             hakn=True)
    assert not applied.hakn_fallback
    ok = "\n".join(conventions_footer(settings, applied, k_papers=3, k_datasets=3))
    assert "Hartung\u2013Knapp" in ok and "nothing to adjust" not in ok


def test_the_methods_paragraph_does_not_call_a_z_statistic_a_t(settings, gold_rows, protocol,
                                                               tmp_path):
    """The same falsehood one layer up: `html.methods_paragraph` prints `t = ...` off
    `pooled.hakn`, which records what was ASKED for rather than what was used."""
    import numpy as np

    from canopy.report.html import methods_paragraph

    yi, vi = np.array([-0.5, -0.5, -0.5]), np.array([0.04, 0.04, 0.04])
    fell_back = random_effects(yi, vi, method=settings.tau2_method, hakn=True)
    manifest = _manifest(tmp_path, protocol.hash())
    manifest.settings = settings
    results = {"late_adaptation": {"pooled": fell_back, "rows": gold_rows[:3],
                                   "needs_human_rows": []}}
    text = methods_paragraph(manifest, protocol, results)
    pooled_sentence = next(line for line in text.splitlines() if "pooled" in line and "CI" in line)
    assert "z = " in pooled_sentence and "t = " not in pooled_sentence, pooled_sentence
    assert "nothing to adjust" in pooled_sentence


def local_links(page: str) -> list[str]:
    """Every in-run target the page asks a browser to fetch, decoded the way a browser would."""
    out: list[str] = []
    for raw in re.findall(r'(?:href|src)="([^"]*)"', page):
        value = _html.unescape(raw)
        if not value or value.startswith(("http://", "https://", "data:", "mailto:", "#")):
            continue
        out.append(unquote(urlparse(value).path))     # a browser drops everything after `#`
    return out


def test_provenance_image_names_survive_being_used_as_a_url(tmp_path, paper):
    """A candidate id ends in `#1`; a browser truncates the link there, so the file must not."""
    from canopy.models import Candidate, DispersionType
    from canopy.report.provenance import provenance_bundle

    page, quote = _bock_quote(paper)
    cid = "sha:d1:late_adaptation:A:text:table_first:claude-opus-5#1"
    cand = Candidate(candidate_id=cid, paper_id=paper.sha256, dataset_id="sha:d1",
                     outcome_key="late_adaptation", kind="group_stats", group="A", mean=42.5,
                     dispersion_value=6.9, dispersion_type=DispersionType.SD, n=12, page=page,
                     quote=quote, route="text", model="claude-opus-5")
    bundle = provenance_bundle(paper, [cand], tmp_path / "provenance")
    crop = Path(bundle["entries"][cid]["crop"])
    assert crop.exists() and crop.stat().st_size > 0
    assert "#" not in crop.name and ":" not in crop.name


def test_every_link_in_the_html_report_resolves(tmp_path, paper, small_rows, outcome, settings,
                                                protocol):
    """Not "a link is present" — every href and src the page emits must name a file that is there."""
    from canopy.models import Candidate, DispersionType
    from canopy.report import provenance_bundle, write_html_report, write_outcome_outputs

    page, quote = _bock_quote(paper)
    cands = [Candidate(candidate_id=f"sha:d{i}:late_adaptation:A:text:table_first:opus#{i + 1}",
                       paper_id=paper.sha256, dataset_id=f"d{i}", outcome_key=outcome.key,
                       kind="group_stats", group="A", mean=42.5 + i, dispersion_value=6.9,
                       dispersion_type=DispersionType.SD, n=12, page=page, quote=quote,
                       route="text", model="claude-opus-5")
             for i in range(2)]
    bundle = provenance_bundle(paper, cands, tmp_path / "provenance")
    pooled = random_effects([r.es for r in small_rows], [r.var for r in small_rows],
                            method=settings.tau2_method)
    outputs = write_outcome_outputs(tmp_path, outcome, small_rows, pooled, settings)
    out = write_html_report(
        tmp_path, _manifest(tmp_path, protocol.hash()), protocol,
        results={outcome.key: {"pooled": pooled, "outputs": outputs, "rows": small_rows,
                               "needs_human_rows": []}},
        provenance=bundle["entries"],
        run_outputs={"provenance_json": bundle["json"]})
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")   # linked from the footer

    targets = local_links(out["html"].read_text(encoding="utf-8"))
    assert any("provenance/" in t for t in targets), "the per-value evidence is not linked at all"
    missing = [t for t in targets if not (tmp_path / t).exists()]
    assert missing == [], f"dead links in report.html: {missing}"


def test_forest_weights_are_the_pooled_models_own_weights(gold_rows, pooled_gold, settings):
    """The plot must not invent a weighting: every square is `1/(vi + tau2)` from the MetaResult."""
    import numpy as np

    from canopy.report.forest import forest_layout

    layout = forest_layout(gold_rows, pooled_gold, settings)
    by_dataset = {row.record.dataset_id: row.weight_pct for row in layout.rows}
    expected = {r.dataset_id: w for r, w in zip(gold_rows, pooled_gold.weights_pct)}
    for dataset_id, weight in expected.items():
        assert by_dataset[dataset_id] == pytest.approx(float(weight), rel=1e-9)
    assert np.isclose(sum(by_dataset.values()), 100.0)


def test_the_effect_size_and_its_interval_are_separate_named_columns(tmp_path, gold_rows,
                                                                     pooled_gold, outcome,
                                                                     settings):
    """A reader scans a column of effect sizes; they should not have to parse brackets out of it.

    The estimate column is headed by what the number IS (`Cohen's d`, `Hedges' g` — from the
    profile's estimator), not by the outcome, which the title already names; the interval sits
    beside it under `95% CI`, the convention the reference review's own forest follows.
    """
    from canopy.report.forest import forest_plot
    from canopy.report.theme import estimator_label

    out = forest_plot(gold_rows, pooled_gold, outcome, settings, tmp_path / "forest")
    svg = out["svg"].read_text(encoding="utf-8")
    assert estimator_label(settings) in svg          # "Hedges' g" for this profile
    assert f"{settings.ci_level * 100:g}% CI" in svg
    # the combined form is gone: no cell prints an estimate and its interval together
    row = gold_rows[0]
    assert f"{row.es:.2f} [" not in svg


# ------------------------------------------------- DECISION A/B/F: the report's two lines
def _nine_report(run: Path):
    """The nine-paper run written as a real report: both outcomes, both analysis lines.

    Late adaptation pools two rows and admits three more by rule; aftereffect pools NOTHING
    (strict k = 1) while its best-guess line has three — the shape DECISION A gives its own
    heading. The aftereffect's two Vachon rows are given the value a categorical point read
    produces over the same datasets' own group sizes, exactly as
    `test_aftereffect_has_best_guess_forest_without_strict_forest` does.
    """
    from canopy.pipeline.run import _split_rows
    from canopy.report import write_html_report
    from canopy.report.tables import pool_rows
    from canopy.stats.effect_sizes import se_smd
    from tests.helpers import nine

    reads = {"b7523a41b03a:d1": 0.13, "b7523a41b03a:d2": 1.24}

    def point_read(record):
        if record.dataset_id not in reads:
            return None
        late = nine.record(record.dataset_id, "late_adaptation")
        es = reads[record.dataset_id]
        se = se_smd(es, late.n_a, late.n_b)
        return {"route": "figure", "es": es, "se": se, "var": se * se, "n_a": late.n_a,
                "n_b": late.n_b, "not_convertible_reason": "",
                "flags": [f for f in record.flags if f != "not_convertible"]
                         + ["categorical_point_read"]}

    protocol = nine.protocol()
    results = {}
    for key, update in (("late_adaptation", None), ("aftereffect", point_read)):
        out, _payload, _cells = _nine_outputs(run, key, held_update=update)
        split = _split_rows(nine.records(key), protocol.stats)
        held = [r.model_copy(update=(update(r) or {})) if update else r for r in split.held]
        results[key] = {"pooled": pool_rows(split.primary, protocol.stats), "outputs": out,
                        "rows": split.primary, "needs_human_rows": held}
    manifest = _manifest(run, protocol.hash())
    (run / "manifest.json").write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    write_html_report(run, manifest, protocol, results=results)
    return run


@pytest.fixture(scope="module")
def tmp_run(tmp_path_factory) -> Path:
    return _nine_report(tmp_path_factory.mktemp("nine_report"))


def test_report_html_sections_in_order(tmp_run):
    """The conclusion first, the primary analysis next, the guess last — and never unlabelled."""
    from canopy.report import theme
    from tests.helpers import nine

    html = (tmp_run / "report.html").read_text(encoding="utf-8")
    assert html.index("<h2>Conclusion</h2>") < html.index("Best guess (not the primary analysis)")
    # LA has a disputed admission, so its section prints the ASSEMBLED caveat: the constant is
    # its prefix, and the disputed clause follows it (adversarial finding 3's threading)
    assert theme.BEST_GUESS_CAVEAT in html
    assert "enter over a standing dispute" in html
    # the outcome's own heading (its strict cards) comes before its guess
    label = nine.protocol().outcome("late_adaptation").label
    assert html.index(f"<h2>{label}</h2>") < html.index("Best guess (not the primary analysis)")
    assert "results/late_adaptation/forest_best_guess.png" in html
    assert "low_confidence_value" in html and "disputed_reading_guess" in html
    # …and the guess's own artefacts are real files, not a second name for the strict ones
    missing = sorted({t for t in local_links(html) if not (tmp_run / t).exists()})
    assert missing == [], f"dead links in report.html: {missing}"


def test_aftereffect_report_says_best_guess_only(tmp_run):
    assert ("Best guess only — nothing could be pooled for the primary analysis."
            in (tmp_run / "report.html").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- cluster-robust (RVE)
@pytest.fixture
def rve_settings(settings) -> StatsSettings:
    return settings.model_copy(update={"dependency": "cluster_robust", "hakn": False,
                                       "one_row_per_paper": False, "rve_rho": 0.8})


@pytest.fixture
def clustered_rows(small_rows) -> list[EffectSizeRecord]:
    """The five rows re-labeled so two papers contribute two rows each: 3 clusters of 2/2/1."""
    clusters = ["p0", "p0", "p1", "p1", "p2"]
    return [row.model_copy(update={"paper_id": c, "cluster_id": c})
            for row, c in zip(small_rows, clusters)]


def test_pool_rows_clusters_by_the_k_papers_chain_under_rve(clustered_rows, rve_settings,
                                                            settings):
    from canopy.report.tables import pool_rows

    robust = pool_rows(clustered_rows, rve_settings)
    plain = pool_rows(clustered_rows, settings)
    assert robust.robust and robust.n_clusters == 3 and robust.rho == 0.8
    assert robust.estimate != pytest.approx(plain.estimate)   # the full CORR fit, not a re-label
    # the per-pool override wins over the settings default, both ways
    assert not pool_rows(clustered_rows, rve_settings, dependency="independent").robust
    assert pool_rows(clustered_rows, settings.model_copy(update={"one_row_per_paper": False}),
                     dependency="cluster_robust").robust


def test_pooled_json_records_rve_fields(tmp_path, clustered_rows, outcome, rve_settings):
    from canopy.report import write_outcome_outputs
    from canopy.report.tables import pool_rows

    pooled = pool_rows(clustered_rows, rve_settings)
    write_outcome_outputs(tmp_path, outcome, clustered_rows, pooled, rve_settings)
    payload = json.loads((tmp_path / "results" / outcome.key / "pooled.json").read_text())
    assert payload["robust"] is True
    assert payload["n_clusters"] == 3
    assert payload["rho"] == 0.8
    assert payload["df_robust"] == pytest.approx(pooled.df_robust)
    assert payload["settings"]["dependency"] == "cluster_robust"
    assert payload["Q_p"] is None                         # NaN lands as null, never as NaN text
    # funnel.json's center is the run's own pooled estimate — the two files may never disagree
    funnel = json.loads((tmp_path / "results" / outcome.key / "funnel.json").read_text())
    assert funnel["funnel"]["estimate"] == pytest.approx(pooled.estimate)
    assert "treats them as independent" in funnel["egger_note"]
    # sensitivity: the dependency entry exists and the HK entry stayed independent
    sens = json.loads((tmp_path / "results" / outcome.key / "sensitivity.json").read_text())
    by_name = {e["name"]: e for e in sens["analyses"]}
    assert by_name["dependency"]["dependency"] == "independent"
    assert by_name["hartung_knapp"]["dependency"] == "independent"
    assert by_name["primary"]["dependency"] == "cluster_robust"
    assert by_name["primary"]["n_clusters"] == 3


def test_rve_forest_footer_speaks_the_corr_model_and_prints_no_nan(clustered_rows, rve_settings):
    from canopy.report.tables import pool_rows
    from canopy.report.theme import conventions_footer

    pooled = pool_rows(clustered_rows, rve_settings)
    footer = "\n".join(conventions_footer(rve_settings, pooled, k_papers=3, k_datasets=5))
    assert "CORR method of moments (RVE)" in footer
    assert "Cluster-robust SE" in footer and "m = 3 clusters of 5 rows" in footer
    assert "Satterthwaite" in footer
    assert "heterogeneity test not defined under RVE" in footer
    assert "nan" not in footer                            # M2: no literal nan on any RVE plot
    assert "REML" not in footer                           # B1: never a wrong label on the tau²
    assert "df < 4" in footer                             # 3 clusters — the warning must fire


def test_rve_forest_weight_column_prints_the_pooler_weights(clustered_rows, rve_settings):
    from canopy.report.forest import forest_layout
    from canopy.report.tables import pool_rows, poolable_rows

    pooled = pool_rows(clustered_rows, rve_settings)
    layout = forest_layout(clustered_rows, pooled, rve_settings)
    printed = {row.record.dataset_id: row.weight_pct for row in layout.rows}
    expected = {r.dataset_id: float(w) for r, w in zip(poolable_rows(clustered_rows),
                                                       pooled.weights_pct)}
    for dataset_id, pct in expected.items():
        assert printed[dataset_id] == pytest.approx(pct, abs=1e-9), dataset_id


def test_rve_never_reaches_the_r_renderer(tmp_path, clustered_rows, outcome, rve_settings,
                                          monkeypatch):
    from canopy.report import forest_render
    from canopy.report.tables import pool_rows

    def _boom(*a, **k):                                   # noqa: ANN002, ANN003
        raise AssertionError("render_forest_r must not be called for an RVE forest")
    monkeypatch.setattr(forest_render, "render_forest_r", _boom)
    pooled = pool_rows(clustered_rows, rve_settings)
    paths, info = forest_render.render_forest(clustered_rows, pooled, outcome, rve_settings,
                                              None, tmp_path / "forest")
    assert paths["png"].exists()
    assert info.renderer == "canopy.report.forest (matplotlib)"
    assert "RVE" in info.reason
    assert info.crosscheck is None                        # canopy validate stays green


def test_rve_leave_one_out_drops_clusters_not_rows(clustered_rows, rve_settings):
    from canopy.report.tables import leave_one_out_rows

    table = leave_one_out_rows(clustered_rows, rve_settings)
    assert len(table) == 3                                # one per cluster, not 5 per row
    assert table[0]["omitted_dataset_id"] == ["d0", "d1"]  # the cluster's rows, joined
    assert all(entry["m"] == 2 for entry in table)


def test_sensitivity_dependency_entry_is_omitted_under_a_composite_primary(small_rows, settings):
    """one_row_per_paper primaries get the entry OMITTED with the reason recorded — RVE over
    composites is not the robumeta analysis of the raw rows (review M1)."""
    from canopy.report.tables import sensitivity_analyses

    composite = settings.model_copy(update={"one_row_per_paper": True})
    payload = sensitivity_analyses(small_rows, composite)
    entry = next(e for e in payload["analyses"] if e["name"] == "dependency")
    assert entry["estimate"] is None
    assert "not computed" in entry["note"]
    # …and under a non-composite independent primary the entry really is the RVE pool
    computed = sensitivity_analyses(small_rows, settings)
    entry = next(e for e in computed["analyses"] if e["name"] == "dependency")
    assert entry["dependency"] == "cluster_robust" and entry["estimate"] is not None
