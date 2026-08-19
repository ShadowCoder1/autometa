"""Tables and the diagnostic figures that go with them.

Six deliverables live here, and one rule governs all of them: **a number in a table was computed
somewhere else.** Pooling, heterogeneity, leave-one-out and Egger all come from `canopy.stats`;
this module selects rows, names the analysis, and writes files.

* `extraction_table` — one row per dataset × outcome with every raw value, the route, the
  conversion chain, the confidence bucket, the flags and the provenance (page, quote, crop). It is
  the artefact a reviewer checks the review with, so it holds what was read, not only what was
  computed.
* `exclusions_table` — what was dropped, at which stage, why (a closed reason list), on whose
  say-so, with the quote.
* `leave_one_out_table` — the pooled estimate without each study in turn.
* `sensitivity_analyses` / `sensitivity_outputs` — the named set from amendment H, as JSON plus
  small multiples, so "does this finding depend on a choice we made?" has a printed answer.
* `funnel_plot` — funnel with pseudo-CI contours and Egger's test, using the Pustejovsky–Rodgers
  sample-size predictor whenever the group sizes are known.
* `prisma_flow` — the PRISMA-style count chain, which **reports its own inconsistencies** rather
  than quietly printing numbers that do not add up.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..models import Candidate, EffectSizeRecord, StatsSettings, Verdict
from ..stats.meta import (EggerResult, MetaResult, egger_test, funnel_data, leave_one_out,
                          random_effects)
from . import theme
from .theme import ACCENT, ACCENT_SOFT, AXIS, GRID, INK, INK_SECONDARY, MARK, MUTED, figure_style

__all__ = ["extraction_table", "EXTRACTION_COLUMNS", "exclusions_table", "EXCLUSION_REASONS",
           "leave_one_out_table", "leave_one_out_rows", "sensitivity_analyses",
           "sensitivity_outputs",
           "SENSITIVITY_ANALYSES", "funnel_plot", "prisma_flow", "PRISMA_CHAIN", "pool_rows",
           "poolable_rows",
           "write_rows", "dump_json"]

#: the reasons a paper or a dataset can leave the review — a free-text reason becomes `other`.
#: A reason may carry a target after a colon (`aggregated_into:d1+d2`): the part before the colon
#: is the enum, the whole string is kept in `reason_as_given`.
EXCLUSION_REASONS: tuple[str, ...] = (
    "duplicate", "not_eligible", "ineligible_design", "no_usable_data", "outcome_not_reported",
    "not_convertible", "needs_human", "aggregated", "superseded_by_dataset_rule", "ingest_failed",
    "budget_exhausted", "error", "human_override",
    #: C7: the map adjudicator rejected a dataset on a named protocol rule, before any extraction
    #: was bought for it. The dataset never reaches the resolver, so without this it left the
    #: review without appearing anywhere — the same silence an ineligible paper used to leave.
    "map_adjudication", "other")
#: `aggregated_into:<row>` is written by the within-paper aggregation and read as `aggregated`
_REASON_ALIASES: dict[str, str] = {"aggregated_into": "aggregated"}

#: the analyses amendment H requires; `by_analysis_metric` fans out to one entry per metric
SENSITIVITY_ANALYSES: tuple[str, ...] = (
    "primary", "exclude_figure_derived", "exclude_test_statistic_derived", "include_needs_human",
    "best_guess", "hartung_knapp", "normal_z", "with_digitization_variance",
    "without_digitization_variance", "by_analysis_metric", "one_row_per_paper", "all_rows")

#: `(start, removals, end)` — every step must satisfy `start − Σ removals = end`. `not_processed`
#: is what `--max-papers` left out: a partial run still has to add up, so the papers it never
#: looked at are a removal of their own rather than being folded into "excluded".
PRISMA_CHAIN: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("files", ("duplicates_removed",), "unique_papers"),
    ("unique_papers", ("not_processed", "papers_excluded"), "eligible_papers"),
    #: the one link that is not an identity of how the counts are built: `included_papers` is
    #: counted from the effect-size records, `eligible_papers` from the manifest, and
    #: `papers_with_no_rows` from their difference — so a paper that was eligible and left no
    #: row anywhere shows up here as a removal with a name, instead of "consistent: true" beside
    #: `eligible_papers: 3, included_papers: 2` (which is what three runs printed).
    ("eligible_papers", ("papers_with_no_rows",), "included_papers"),
    ("datasets", ("datasets_excluded",), "included_datasets"),
)


# ----------------------------------------------------------------------------- small helpers
def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})
    return path


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # NaN is "no value" and prints as empty; ±inf is a value (a variance that overflowed, a
        # ratio with a zero denominator) and prints as itself. `int(inf)` raises, so neither may
        # reach the rounding branch.
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value == int(value):
            return str(int(value))
        return repr(round(value, 10)).rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def dump_json(payload: Any, path: Path) -> Path:
    """Write one JSON artefact (numpy arrays, dataclasses and Paths included), as valid JSON.

    `json.dumps` writes Python's own spelling of a value it has no JSON for — `NaN`, `Infinity`,
    `-Infinity` — and every one of those makes the whole FILE unreadable to a consumer that is
    not Python: R's jsonlite, `jq` and every browser refuse the document, not the number. A
    quantity that is not estimable is `null`, which is what the rest of these artefacts already
    write for it (review MINOR 21).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=1, default=_default),
                    encoding="utf-8")
    return path


def _plain(obj: Any) -> Any:
    """The payload with every float JSON can print — a non-finite one becomes `None`."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        return _plain(obj.item())
    if isinstance(obj, np.ndarray):
        return [_plain(float(x)) for x in obj.ravel()]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _plain(asdict(obj))
    if isinstance(obj, Mapping):
        return {key: _plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(value) for value in obj]
    return obj


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [float(x) for x in obj.ravel()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_xlsx(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> Path:
    """One sheet, one header row. Missing openpyxl is reported, never silently skipped.

    Quotes come out of real PDFs, and real PDFs contain control characters that the XLSX format
    forbids outright. They are stripped **here only**: the CSV and the JSON keep the text exactly
    as the extractor read it, because that is the copy a reviewer checks against the paper.
    """
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    path.parent.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    sheet = book.active
    sheet.title = "extraction"
    sheet.append(list(columns))
    for row in rows:
        sheet.append([ILLEGAL_CHARACTERS_RE.sub("", _cell(row.get(c))) for c in columns])
    sheet.freeze_panes = "A2"
    book.save(path)
    return path


def write_rows(rows: Sequence[Mapping[str, Any]], out_stem: str | Path, columns: Sequence[str],
               formats: Sequence[str] = ("csv", "json", "xlsx")) -> dict[str, Path]:
    """One table in every requested format (`csv`, `json`, `xlsx`)."""
    stem = Path(out_stem)
    out: dict[str, Path] = {}
    if "csv" in formats:
        out["csv"] = _write_csv(rows, stem.with_suffix(".csv"), columns)
    if "json" in formats:
        out["json"] = dump_json([dict(r) for r in rows], stem.with_suffix(".json"))
    if "xlsx" in formats:
        out["xlsx"] = _write_xlsx(rows, stem.with_suffix(".xlsx"), columns)
    return out


# ----------------------------------------------------------------------------- extraction table
EXTRACTION_COLUMNS: tuple[str, ...] = (
    "paper_id", "cluster_id", "sample_id", "dataset_id", "outcome_key", "label", "first_author",
    "year", "primary_row", "route", "analysis_metric", "confidence", "flags",
    "n_a", "mean_a", "dispersion_a", "dispersion_type_a", "unit_a", "route_a",
    "page_a", "quote_a", "crop_a", "overlay_a", "sigma_a",
    "n_b", "mean_b", "dispersion_b", "dispersion_type_b", "unit_b", "route_b",
    "page_b", "quote_b", "crop_b", "overlay_b", "sigma_b",
    "estimator", "variance_method", "es", "se", "var", "ci_low", "ci_high", "level",
    "higher_is_better", "orientation_applied", "conversion_chain", "routes_available",
    "routes_rejected", "not_convertible_reason", "digitization_var", "digitization_var_share",
    "inputs", "notes",
    # DECISION A: the second analysis line, per row — whether it is in the best guess, the rule
    # that admitted it (empty on a row the primary analysis already had) or the veto and reason
    # that kept it out, and the value the line used. They sit here, ahead of the `mod_*` block,
    # so a reviewer reads them next to the row's own value rather than past its moderators.
    "in_best_guess", "best_guess_rule", "best_guess_reason", "best_guess_es", "best_guess_se")


def _group_provenance(record: EffectSizeRecord, group: str, verdicts: Mapping[tuple, Verdict],
                      candidates: Mapping[str, Candidate]) -> dict[str, Any]:
    """Everything known about ONE group of one cell: verified values first, then where they came from."""
    key = (record.dataset_id, record.outcome_key, group)
    verdict = verdicts.get(key)
    suffix = group.lower()
    out: dict[str, Any] = {f"n_{suffix}": record.n_a if group == "A" else record.n_b}
    if verdict is not None:
        out.update({
            f"n_{suffix}": verdict.n if verdict.n is not None else out[f"n_{suffix}"],
            f"mean_{suffix}": verdict.mean,
            f"dispersion_{suffix}": verdict.dispersion_value,
            f"dispersion_type_{suffix}": getattr(verdict.dispersion_type, "value",
                                                 verdict.dispersion_type),
            f"unit_{suffix}": verdict.unit,
            f"route_{suffix}": verdict.route,
            f"sigma_{suffix}": verdict.sigma,
        })
        for cid in verdict.candidate_ids:
            cand = candidates.get(cid)
            if cand is None:
                continue
            out.setdefault(f"page_{suffix}", cand.page)
            if cand.quote and not out.get(f"quote_{suffix}"):
                out[f"quote_{suffix}"] = cand.quote
            if cand.crop_path and not out.get(f"crop_{suffix}"):
                out[f"crop_{suffix}"] = cand.crop_path
            if cand.overlay_path and not out.get(f"overlay_{suffix}"):
                out[f"overlay_{suffix}"] = cand.overlay_path
    else:                                        # no verdict: the record's own inputs still show
        inputs = record.inputs or {}
        out[f"mean_{suffix}"] = inputs.get(f"mean_{suffix}")
        out[f"dispersion_{suffix}"] = inputs.get(f"sd_{suffix}")
        if out[f"dispersion_{suffix}"] is not None:
            out[f"dispersion_type_{suffix}"] = "SD"
        if inputs.get(f"n_{suffix}") is not None:
            out[f"n_{suffix}"] = int(inputs[f"n_{suffix}"])
    return out


def extraction_row(record: EffectSizeRecord, verdicts: Mapping[tuple, Verdict],
                   candidates: Mapping[str, Candidate],
                   primary_row: bool | None = None,
                   best_guess: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "paper_id": record.paper_id, "cluster_id": record.cluster_id,
        "sample_id": record.sample_id, "primary_row": primary_row,
        "dataset_id": record.dataset_id, "outcome_key": record.outcome_key,
        "label": record.label, "first_author": record.citation.first_author,
        "year": record.citation.year, "route": record.route,
        "analysis_metric": record.analysis_metric, "confidence": record.confidence,
        "flags": list(record.flags), "estimator": record.estimator,
        "variance_method": record.variance_method, "es": record.es, "se": record.se,
        "var": record.var, "ci_low": record.ci_low, "ci_high": record.ci_high,
        "level": record.level, "higher_is_better": record.higher_is_better,
        "orientation_applied": record.orientation_applied,
        "conversion_chain": record.conversion_chain,
        "routes_available": list(record.routes_available),
        "routes_rejected": dict(record.routes_rejected),
        "not_convertible_reason": record.not_convertible_reason,
        "digitization_var": record.digitization_var,
        "digitization_var_share": record.digitization_var_share,
        "inputs": dict(record.inputs), "notes": record.notes,
    }
    for group in ("A", "B"):
        row.update(_group_provenance(record, group, verdicts, candidates))
    # the best-guess line's own view of this row. Without a decision for it the columns stay
    # empty — a blank is "this run did not compute a second line", which is not the same claim
    # as "this row is not in it", and the record's own fields are the fallback because a row that
    # already carries them was written by the line itself.
    row.update({"in_best_guess": None, "best_guess_rule": record.best_guess_rule,
                "best_guess_reason": record.best_guess_reason,
                "best_guess_es": None, "best_guess_se": None})
    if best_guess is not None:
        row.update({key: best_guess.get(key) for key in
                    ("in_best_guess", "best_guess_rule", "best_guess_reason",
                     "best_guess_es", "best_guess_se")})
    for name, value in record.moderators.items():
        row[f"mod_{name}"] = value
    return {key: row.get(key) for key in (*EXTRACTION_COLUMNS,
                                          *(k for k in row if k.startswith("mod_")))}


def extraction_table(rows: Sequence[EffectSizeRecord], out_stem: str | Path, *,
                     verdicts: Sequence[Verdict] = (), candidates: Sequence[Candidate] = (),
                     primary: Sequence[EffectSizeRecord] | None = None,
                     best_guess: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
                     formats: Sequence[str] = ("csv", "json", "xlsx")) -> dict[str, Path]:
    """One row per dataset × outcome, with the raw values and where each of them was read.

    `rows` is EVERY row the outcome produced — the ones that were pooled, the ones a paper's
    aggregation replaced and the ones held for review — because this is the table a reviewer
    checks the review with. `primary` names the subset that actually entered the pool; each row
    then carries `primary_row`. Without it the column is left empty rather than guessed.

    `best_guess` is `pipeline.bestguess.best_guess_cells` — `(dataset_id, outcome_key)` to the
    second line's view of that row. It is passed in rather than recomputed so that the run-wide
    table and the per-outcome ones show the SAME decision.
    """
    by_cell = {(v.dataset_id, v.outcome_key, v.group): v for v in verdicts}
    by_id = {c.candidate_id: c for c in candidates}
    in_primary = (None if primary is None
                  else {(r.dataset_id, r.outcome_key) for r in primary})
    table = [extraction_row(record, by_cell, by_id,
                            None if in_primary is None
                            else (record.dataset_id, record.outcome_key) in in_primary,
                            None if best_guess is None
                            else best_guess.get((record.dataset_id, record.outcome_key)))
             for record in rows]
    moderator_columns: list[str] = []
    for row in table:
        for key in row:
            if key.startswith("mod_") and key not in moderator_columns:
                moderator_columns.append(key)
    return write_rows(table, out_stem, (*EXTRACTION_COLUMNS, *moderator_columns), formats)


# ----------------------------------------------------------------------------- exclusions
EXCLUSION_COLUMNS: tuple[str, ...] = ("paper_id", "filename", "dataset_id", "outcome_key",
                                      "stage", "reason", "reason_as_given", "quote", "decider",
                                      "detail")


def exclusions_table(entries: Iterable[Mapping[str, Any]], out_stem: str | Path,
                     formats: Sequence[str] = ("csv", "json")) -> dict[str, Path]:
    """What left the review and why. An unrecognised reason becomes `other`, never disappears."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        given = str(entry.get("reason", "") or "")
        head = given.split(":", 1)[0]
        known = _REASON_ALIASES.get(head, head if head in EXCLUSION_REASONS else "other")
        rows.append({
            "paper_id": entry.get("paper_id", ""), "filename": entry.get("filename", ""),
            "dataset_id": entry.get("dataset_id", ""), "outcome_key": entry.get("outcome_key", ""),
            "stage": entry.get("stage", ""), "reason": known,
            "reason_as_given": given, "quote": entry.get("quote", ""),
            "decider": entry.get("decider", ""), "detail": entry.get("detail", ""),
        })
    return write_rows(rows, out_stem, EXCLUSION_COLUMNS, formats)


# ----------------------------------------------------------------------------- pooling helpers
def _label(record: EffectSizeRecord) -> str:
    name = theme.study_label(record)
    year = record.citation.year
    return f"{name} {year}" if year else name


def _poolable(rows: Sequence[EffectSizeRecord], *, digitization: bool = False
              ) -> tuple[list[EffectSizeRecord], list[float], list[float]]:
    keep, yi, vi = [], [], []
    for record in rows:
        var = record.var
        if digitization and record.var_with_digitization:
            var = record.var_with_digitization
        if record.es is None or not var or var <= 0 or not math.isfinite(record.es):
            continue
        keep.append(record)
        yi.append(float(record.es))
        vi.append(float(var))
    return keep, yi, vi


def poolable_rows(rows: Sequence[EffectSizeRecord], *,
                  digitization: bool = False) -> list[EffectSizeRecord]:
    """The rows `pool_rows` would actually pool, in the order it hands them to the pooler.

    Public because anything that reads a pooled result BY POSITION — the per-row weights, above
    all — has to use the same predicate the pooler used, not a second one that agrees today.
    """
    keep, _, _ = _poolable(rows, digitization=digitization)
    return keep


def pool_rows(rows: Sequence[EffectSizeRecord], settings: StatsSettings, *,
              hakn: bool | None = None, digitization: bool = False) -> MetaResult | None:
    """Random-effects pooling of whatever in `rows` can be pooled; `None` below k = 2."""
    _, yi, vi = _poolable(rows, digitization=digitization)
    if len(yi) < 2:
        return None
    return random_effects(yi, vi, method=settings.tau2_method,
                          hakn=settings.hakn if hakn is None else hakn,
                          level=settings.ci_level)


# ----------------------------------------------------------------------------- leave-one-out
LOO_COLUMNS: tuple[str, ...] = ("omitted_label", "omitted_dataset_id", "omitted_paper_id", "k",
                                "estimate", "se", "ci_low", "ci_high", "tau2", "I2", "Q", "Q_p")


def leave_one_out_rows(rows: Sequence[EffectSizeRecord],
                       settings: StatsSettings) -> list[dict[str, Any]]:
    """The leave-one-out table as data — empty below k = 3, where the question is meaningless.

    Separate from the writer because the best-guess line needs the same numbers in
    `pooled.json` (how far one guessed row moves the line) as it writes to
    `leave_one_out_best_guess.csv`, and computing them twice is how two artefacts of one run
    start to disagree.
    """
    keep, yi, vi = _poolable(rows)
    if len(keep) < 3:
        return []
    results = leave_one_out(yi, vi, method=settings.tau2_method, level=settings.ci_level,
                            labels=[_label(r) for r in keep])
    return [{"omitted_label": r.label, "omitted_dataset_id": keep[r.omitted].dataset_id,
             "omitted_paper_id": keep[r.omitted].paper_id, "k": r.k, "estimate": r.estimate,
             "se": r.se, "ci_low": r.ci_low, "ci_high": r.ci_high, "tau2": r.tau2, "I2": r.I2,
             "Q": r.Q, "Q_p": r.Q_p} for r in results]


def leave_one_out_table(rows: Sequence[EffectSizeRecord], settings: StatsSettings,
                        out_stem: str | Path,
                        formats: Sequence[str] = ("csv", "json"),
                        table: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Path]:
    """`metafor::leave1out` over the pooled rows: which study, if any, is carrying the result."""
    entries = leave_one_out_rows(rows, settings) if table is None else list(table)
    return write_rows(entries, out_stem, LOO_COLUMNS, formats)


# ----------------------------------------------------------------------------- sensitivity set
def _is_figure_route(route: str) -> bool:
    head = (route or "").split(":", 1)[0]
    return head.startswith(("figure", "digitize"))


def _is_statistic_route(route: str) -> bool:
    return (route or "").split(":", 1)[0] in ("test_statistic", "p_value")


def _one_per_paper(rows: Sequence[EffectSizeRecord],
                   settings: StatsSettings) -> list[EffectSizeRecord]:
    """One row per paper, by the SAME rule the primary analysis uses (amendment A).

    The sensitivity set exists to show what a choice costs, so this must be the choice the primary
    analysis would make — a Borenstein composite of the paper's rows, not a pick of the most
    precise one. Imported lazily because the aggregation lives in `canopy.pipeline` (it is part of
    the analysis, not of the report) and this package must not depend on that package at import
    time.
    """
    from ..pipeline.aggregate import aggregate_one_row_per_paper

    return aggregate_one_row_per_paper(rows, settings).rows


def _entry(name: str, description: str, rows: Sequence[EffectSizeRecord],
           settings: StatsSettings, *, hakn: bool | None = None, digitization: bool = False,
           primary: MetaResult | None = None) -> dict[str, Any]:
    keep, _, _ = _poolable(rows, digitization=digitization)
    result = pool_rows(rows, settings, hakn=hakn, digitization=digitization)
    entry: dict[str, Any] = {
        "name": name, "description": description, "k": len(keep),
        "k_papers": len({r.cluster_id or r.paper_id or r.dataset_id for r in keep}),
        "hakn": settings.hakn if hakn is None else hakn,
        "tau2_method": settings.tau2_method,
        "digitization_variance_included": bool(digitization),
    }
    if result is None:
        entry.update({"estimate": None, "se": None, "ci_low": None, "ci_high": None,
                      "tau2": None, "I2": None, "pi_low": None, "pi_high": None,
                      "delta_vs_primary": None,
                      "note": "fewer than two poolable rows — not pooled"})
        return entry
    from ..stats.meta import prediction_interval
    try:
        pi_low, pi_high, _ = prediction_interval(result, str(settings.pi_method))
    except ValueError:                                     # pragma: no cover - guarded upstream
        pi_low = pi_high = float("nan")
    entry.update({
        "estimate": result.estimate, "se": result.se, "ci_low": result.ci_low,
        "ci_high": result.ci_high, "tau2": result.tau2, "I2": result.I2,
        "I2_tau": result.I2_tau, "Q": result.Q, "Q_p": result.Q_p,
        "pi_low": None if math.isnan(pi_low) else pi_low,
        "pi_high": None if math.isnan(pi_high) else pi_high,
        "delta_vs_primary": None if primary is None else result.estimate - primary.estimate,
        "yi": [float(v) for v in result.yi], "vi": [float(v) for v in result.vi],
    })
    return entry


def sensitivity_analyses(rows: Sequence[EffectSizeRecord], settings: StatsSettings, *,
                         needs_human_rows: Sequence[EffectSizeRecord] = (),
                         best_guess_rows: Sequence[EffectSizeRecord] | None = None
                         ) -> dict[str, Any]:
    """The named analyses of amendment H, each as a pooled result next to the primary one.

    `best_guess_rows` is DECISION A's second line (the primary rows plus the held rows a rule
    admitted). Without it the entry is the primary set itself — which is exactly what the line is
    when no held row could be admitted, so the analysis is always present and never invented.
    """
    primary = pool_rows(rows, settings)
    def add(name, description, subset, **kw):
        return _entry(name, description, subset, settings, primary=primary, **kw)

    held_back = add("include_needs_human", "rows held for human review added back",
                    [*rows, *needs_human_rows])
    # amendment H's honesty rule: an analysis that says "held rows added back" while several of
    # them carry no effect size to add is reporting a set it did not analyse. The note counts
    # them, so the k beside it can be read for what it is.
    unusable = len(needs_human_rows) - len(_poolable(needs_human_rows)[0])
    if needs_human_rows:
        held_back["note"] = "; ".join(x for x in [
            held_back.get("note") or "",
            f"{unusable} of {len(needs_human_rows)} held row(s) could not be included "
            f"(no usable effect size and variance)"] if x)

    analyses: list[dict[str, Any]] = [
        add("primary", "every row in the primary analysis, as configured", rows),
        add("exclude_figure_derived", "rows read off a figure removed",
            [r for r in rows if not _is_figure_route(r.route)]),
        add("exclude_test_statistic_derived", "rows converted from a t/F/p statistic removed",
            [r for r in rows if not _is_statistic_route(r.route)]),
        held_back,
        add("best_guess", "the best-guess line: the primary analysis plus every held row a "
                          "named rule admitted at its own value (DECISION A)",
            list(rows) if best_guess_rows is None else list(best_guess_rows)),
        add("hartung_knapp", "Hartung-Knapp variance and t(k-1) confidence interval", rows,
            hakn=True),
        add("normal_z", "normal (z) confidence interval, no Hartung-Knapp", rows, hakn=False),
        add("with_digitization_variance",
            "digitisation uncertainty added to each figure-derived sampling variance", rows,
            digitization=True),
        add("without_digitization_variance", "sampling variance only", rows, digitization=False),
        add("one_row_per_paper", "each paper's rows combined into one (amendment A)",
            _one_per_paper(rows, settings)),
        add("all_rows", "every row, including several from the same paper", rows),
    ]
    metrics: list[str] = []
    for record in rows:
        if record.analysis_metric not in metrics:
            metrics.append(record.analysis_metric)
    for metric in metrics:
        analyses.append(add(f"by_analysis_metric:{metric}",
                            f"rows whose analysis metric is {metric}",
                            [r for r in rows if r.analysis_metric == metric]))

    return {"primary": {"k": 0 if primary is None else primary.k,
                        "estimate": None if primary is None else primary.estimate,
                        "ci_low": None if primary is None else primary.ci_low,
                        "ci_high": None if primary is None else primary.ci_high,
                        "tau2_method": settings.tau2_method, "hakn": settings.hakn,
                        "estimator": settings.estimator, "variance": settings.variance},
            "analyses": analyses}


def _sensitivity_figure(payload: Mapping[str, Any], out_stem: Path,
                        formats: Sequence[str]) -> dict[str, Path]:
    """Small multiples: one mini forest per analysis, all on one x-scale."""
    analyses = [a for a in payload["analyses"] if a.get("estimate") is not None]
    if not analyses:
        analyses = list(payload["analyses"])[:1]
    ncols = 3
    nrows = max(1, math.ceil(len(analyses) / ncols))
    values: list[float] = []
    for a in analyses:
        values.extend(v for v in (a.get("ci_low"), a.get("ci_high")) if v is not None)
        values.extend(float(v) for v in a.get("yi", []) or [])
    lo, hi = (min(values), max(values)) if values else (-1.0, 1.0)
    pad = 0.06 * max(hi - lo, 1e-6)
    primary = payload["primary"].get("estimate")

    with figure_style():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 1.6 * nrows),
                                 squeeze=False, sharex=True)
        flat = [a for row in axes for a in row]
        for index, ax in enumerate(flat):
            for name in ("top", "right", "left"):
                ax.spines[name].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
            ax.set_yticks([])
            ax.set_xlim(lo - pad, hi + pad)
            if index >= len(analyses):
                ax.set_visible(False)
                continue
            entry = analyses[index]
            yi = [float(v) for v in entry.get("yi", []) or []]
            if yi:
                order = np.argsort(yi)
                ys = np.linspace(0.94, 0.52, len(yi))
                ax.scatter(np.asarray(yi)[order], ys, s=6, marker="s", color=MARK, alpha=0.55,
                           linewidths=0, zorder=2)
            if primary is not None:
                ax.axvline(primary, color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=1)
            ax.axvline(0.0, color=GRID, lw=0.8, zorder=1)
            if entry.get("estimate") is not None:
                y = 0.28
                ax.fill([entry["ci_low"], entry["estimate"], entry["ci_high"], entry["estimate"]],
                        [y, y + 0.11, y, y - 0.11], color=ACCENT, zorder=3)
                if entry.get("pi_low") is not None:
                    ax.plot([entry["pi_low"], entry["pi_high"]], [0.08, 0.08],
                            color=ACCENT_SOFT, lw=1.8, solid_capstyle="butt", zorder=3)
            ax.set_ylim(-0.02, 1.06)
            ax.set_title(f"{entry['name']}  (k = {entry['k']})", fontsize=7.5, color=INK,
                         loc="left", pad=3)
            label = f"{theme.fmt(entry.get('estimate'))} " \
                    f"{theme.fmt_ci(entry.get('ci_low'), entry.get('ci_high'))}"
            ax.set_title(label, fontsize=6.8, color=INK_SECONDARY, loc="right", pad=3)
            ax.tick_params(axis="x", length=3, width=0.8, colors=MUTED, labelsize=7)
        # `sharex` hides tick labels on every axes but the bottom of each column; when the last
        # row is short, the bottom of some columns is an invisible axes, so re-enable them
        for column in range(ncols):
            visible = [flat[r * ncols + column] for r in range(nrows)
                       if r * ncols + column < len(analyses)]
            if visible:
                visible[-1].tick_params(axis="x", labelbottom=True)
        fig.text(0.005, 0.004,
                 "Each panel re-pools the same review under one changed choice; the dashed line is "
                 "the primary estimate.", fontsize=6.6, color=MUTED, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return {k: Path(v) for k, v in theme.save_figure(fig, out_stem, formats).items()}


def sensitivity_outputs(rows: Sequence[EffectSizeRecord], settings: StatsSettings,
                        out_stem: str | Path, *,
                        needs_human_rows: Sequence[EffectSizeRecord] = (),
                        best_guess_rows: Sequence[EffectSizeRecord] | None = None,
                        formats: Sequence[str] = ("png",)) -> dict[str, Path]:
    """`sensitivity.json` plus the small-multiples figure."""
    payload = sensitivity_analyses(rows, settings, needs_human_rows=needs_human_rows,
                                   best_guess_rows=best_guess_rows)
    stem = Path(out_stem)
    out = {"json": dump_json(payload, stem.with_suffix(".json"))}
    out.update(_sensitivity_figure(payload, stem, formats))
    return out


# ----------------------------------------------------------------------------- funnel + Egger
def _egger(keep: Sequence[EffectSizeRecord], yi, vi,
           settings: StatsSettings) -> tuple[EggerResult | None, str]:
    """Egger's test, preferring the sample-size predictor; returns `(result, note)`.

    The Pustejovsky–Rodgers predictor √(1/n_A + 1/n_B) is constant when every study used the same
    group sizes, which makes its design matrix singular — the asymmetry it tests is not identified.
    That is a property of the data, not a failure, so it falls back to the classic precision
    predictor and says so, rather than taking the whole outcome's outputs down with it.
    """
    if len(keep) < 3:
        return None, "fewer than three rows — Egger's test needs k ≥ 3"
    sei = [float(math.sqrt(v)) for v in vi]
    n_a = [r.n_a for r in keep]
    n_b = [r.n_b for r in keep]
    if all(a and b for a, b in zip(n_a, n_b)):
        predictor = np.sqrt(1.0 / np.asarray(n_a, float) + 1.0 / np.asarray(n_b, float))
        spread = float(np.ptp(predictor)) / max(float(np.mean(predictor)), 1e-12)
        if spread < 1e-9:
            note = ("every study used the same group sizes, so the sample-size predictor "
                    "\u221a(1/n_A + 1/n_B) is constant and its Egger regression could not be "
                    "fitted; the classic precision predictor was used instead")
            try:
                return egger_test(yi, sei, level=settings.ci_level), note
            except (np.linalg.LinAlgError, ValueError) as inner:
                return None, f"{note}; that failed too ({inner})"
        try:
            return egger_test(yi, sei, n_a=n_a, n_b=n_b, level=settings.ci_level), ""
        except (np.linalg.LinAlgError, ValueError) as exc:
            note = (f"the sample-size predictor could not be fitted ({exc}); the classic "
                    f"precision predictor was used instead")
            try:
                return egger_test(yi, sei, level=settings.ci_level), note
            except (np.linalg.LinAlgError, ValueError) as inner:
                return None, f"{note}; that failed too ({inner})"
    try:
        return egger_test(yi, sei, level=settings.ci_level), ""
    except (np.linalg.LinAlgError, ValueError) as exc:
        return None, f"Egger's test could not be fitted ({exc})"


def funnel_plot(rows: Sequence[EffectSizeRecord], settings: StatsSettings, out_stem: str | Path,
                formats: Sequence[str] = ("png", "svg")) -> dict[str, Path]:
    """Funnel with pseudo-CI contours plus Egger's test (JSON), drawn from `canopy.stats`."""
    keep, yi, vi = _poolable(rows)
    stem = Path(out_stem)
    if len(keep) < 2:
        payload = {"k": len(keep), "egger": None,
                   "note": "fewer than two poolable rows — no funnel drawn"}
        return {"json": dump_json(payload, stem.with_suffix(".json"))}

    data = funnel_data(yi, vi, method=settings.tau2_method, labels=[_label(r) for r in keep])
    egger, egger_note = _egger(keep, yi, vi, settings)
    payload = {"k": len(keep), "funnel": data, "egger_note": egger_note,
               "egger": None if egger is None else asdict(egger)}
    out = {"json": dump_json(payload, stem.with_suffix(".json"))}

    with figure_style():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        se_max = max(data["se_max"], 1e-6)
        for level, contour in sorted(data["contours"].items()):
            ax.plot(contour["low"], contour["se"], color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.plot(contour["high"], contour["se"], color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.annotate(f"{float(level) * 100:g}%", xy=(contour["high"][-1], se_max),
                        xytext=(2, -2), textcoords="offset points", fontsize=6.8, color=MUTED)
        ax.axvline(data["estimate"], color=ACCENT, lw=1.0, zorder=2)
        ax.scatter(data["yi"], data["sei"], s=22, marker="o", facecolors="none",
                   edgecolors=MARK, linewidths=0.9, zorder=3)
        ax.set_ylim(se_max * 1.06, 0.0)                    # precise studies at the top
        ax.set_xlabel("Effect size", fontsize=8.5)
        ax.set_ylabel("Standard error", fontsize=8.5)
        for name in ("top", "right"):
            ax.spines[name].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        lines = [f"k = {len(keep)} · pooled estimate {theme.fmt(data['estimate'], 3)} "
                 f"(τ² by {settings.tau2_method})"]
        if egger is not None:
            predictor = ("√(1/n_A + 1/n_B), Pustejovsky–Rodgers"
                         if egger.predictor == "sqrt_inv_n" else "1/seTE, classic Egger")
            lines.append(f"Egger asymmetry test on {predictor}: bias "
                         f"{theme.fmt(egger.intercept, 3)} (SE {theme.fmt(egger.intercept_se, 3)}), "
                         f"t({egger.df}) = {theme.fmt(egger.t, 2)}, p {theme.fmt_p(egger.p)}")
        if egger_note:
            lines.append(egger_note)
        fig.text(0.01, 0.005, "\n".join(lines), fontsize=6.8, color=MUTED, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.02 + 0.03 * len(lines), 1, 1))
        out.update({k: Path(v) for k, v in theme.save_figure(fig, stem, formats).items()})
    return out


# ----------------------------------------------------------------------------- PRISMA
def prisma_flow(counts: Mapping[str, Any], out_stem: str | Path,
                formats: Sequence[str] = ("png", "svg")) -> dict[str, Path]:
    """The PRISMA-style count chain as JSON and a figure.

    The chain is *checked*, not assumed: every `a − b = c` step that does not hold is listed in
    `problems` and `consistent` goes false, because a flow diagram whose numbers disagree with the
    run is worse than no diagram.
    """
    data = {k: v for k, v in counts.items()}
    problems: list[str] = []
    for start, removals, end in PRISMA_CHAIN:
        if not all(isinstance(data.get(k), int) for k in (start, end)):
            continue
        taken = {k: data.get(k) for k in removals if isinstance(data.get(k), int)}
        if data[start] - sum(taken.values()) != data[end]:
            printed = " − ".join(f"{k} ({v})" for k, v in taken.items()) or "0"
            problems.append(f"{start} ({data[start]}) − {printed} ≠ {end} ({data[end]})")
    data["problems"] = problems
    data["consistent"] = not problems
    stem = Path(out_stem)
    out = {"json": dump_json(data, stem.with_suffix(".json"))}

    steps = [("Files found", data.get("files")),
             ("Unique papers", data.get("unique_papers")),
             ("Eligible papers", data.get("eligible_papers")),
             ("Papers contributing rows", data.get("included_papers")),
             ("Datasets found", data.get("datasets")),
             ("Datasets included", data.get("included_datasets"))]
    asides = [("Duplicates removed", data.get("duplicates_removed"), 0),
              ("Not processed (--max-papers)", data.get("not_processed") or None, 1),
              ("Papers excluded", data.get("papers_excluded"), 1),
              ("Eligible, no usable row", data.get("papers_with_no_rows") or None, 2),
              ("Datasets excluded", data.get("datasets_excluded"), 4)]
    reasons = data.get("exclusion_reasons") or {}

    with figure_style():
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        fig, ax = plt.subplots(figsize=(7.2, 1.05 * len(steps) + 0.7))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(steps))
        ax.axis("off")
        box_w, box_h = 4.2, 0.56
        for i, (label, value) in enumerate(steps):
            y = len(steps) - i - 0.65
            ax.add_patch(FancyBboxPatch((0.4, y), box_w, box_h, boxstyle="round,pad=0.04,rounding_size=0.06",
                                        linewidth=0.9, edgecolor=AXIS, facecolor="none"))
            ax.text(0.6, y + box_h / 2, label, fontsize=9, color=INK, va="center", ha="left")
            ax.text(0.4 + box_w - 0.2, y + box_h / 2, "—" if value is None else str(value),
                    fontsize=9.5, fontweight="bold", color=INK, va="center", ha="right")
            if i + 1 < len(steps):
                ax.add_patch(FancyArrowPatch((0.4 + box_w / 2, y - 0.02),
                                             (0.4 + box_w / 2, y - 0.42),
                                             arrowstyle="-|>", mutation_scale=9, color=AXIS,
                                             linewidth=0.9))
        # several removals can share one gap (a partial run loses papers to `--max-papers` AND to
        # eligibility): they go in ONE box, stacked, rather than two boxes drawn over each other
        by_gap: dict[int, list[tuple[str, Any]]] = {}
        for label, value, after in asides:
            if value is None:
                continue
            by_gap.setdefault(after, []).append((label, value))
        for after, items in by_gap.items():
            tall = box_h * (1.0 + 0.62 * (len(items) - 1))
            y = len(steps) - after - 1.15 - (tall - box_h) / 2   # centred in the gap
            ax.add_patch(FancyBboxPatch((5.3, y), 4.3, tall,
                                        boxstyle="round,pad=0.04,rounding_size=0.06",
                                        linewidth=0.9, edgecolor=GRID, facecolor="none"))
            for i, (label, value) in enumerate(items):
                row_y = y + tall - (i + 0.5) * (tall / len(items))
                ax.text(5.5, row_y, label, fontsize=8.2, color=MUTED, va="center", ha="left")
                ax.text(9.4, row_y, str(value), fontsize=8.6, color=INK_SECONDARY,
                        va="center", ha="right")
            ax.add_patch(FancyArrowPatch((0.4 + box_w / 2, y + tall / 2), (5.25, y + tall / 2),
                                         arrowstyle="-|>", mutation_scale=8, color=GRID,
                                         linewidth=0.9))
        footer = []
        if reasons:
            footer.append("Excluded for: " + ", ".join(f"{k.replace('_', ' ')} ({v})"
                                                       for k, v in sorted(reasons.items())))
        if problems:
            footer.append("Inconsistent counts: " + "; ".join(problems))
        if footer:
            fig.text(0.01, 0.01, "\n".join(footer), fontsize=6.8, color=MUTED, ha="left",
                     va="bottom")
        fig.tight_layout(rect=(0, 0.02 + 0.03 * len(footer), 1, 1))
        out.update({k: Path(v) for k, v in theme.save_figure(fig, stem, formats).items()})
    return out
