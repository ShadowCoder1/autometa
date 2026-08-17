"""The orchestrator: a folder of PDFs in, a run directory of results out — resumably.

The shape of this file is the shape of the review. Per paper, five stages, each of which writes
its own JSON and is skipped when that file already exists:

    ingest → map → extract → verify → resolve

and then, once per outcome: pool the rows the protocol admits, write every artefact, and record
what was held back. Three things are deliberate.

* **Nothing is spent twice.** `--resume` is not an optimisation, it is how a run that hit a budget,
  a rate limit or a laptop lid is finished. A second run over a complete directory makes zero
  model calls, and that is asserted in the tests.
* **A paper that fails does not take the run with it.** Its status becomes `error`, its reason is
  recorded, and the others carry on — including when it alone blew `max_usd_per_paper`.
* **Every number that reaches the forest plot came through `canopy.verify` and `canopy.stats`.**
  This module chooses *what to ask*; it never decides what a value is, and it computes nothing.

The order inside `verify` is the one Tasks 8/9 specified: checks → vote → a third cheap reading
when the two text extractors disagree → an adversarial verifier on a different model family
(at most two re-opens) → adjudication only when the vote failed or a verifier refuted →
orientation, decided once per measure by two independent agents → confidence → `resolve_cell`.
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from ..agents.adjudicator import adjudicate
from ..agents.extract_stats import extract_test_statistics
from ..agents.extract_text import extract_group_stats
from ..agents.mapper import map_study
from ..agents.orientation import orientation as orientation_verdict
from ..agents.source_rank import (keep_for_vote, match_named_source, rank_sources,
                                 source_of)
from ..agents.verifier import MAX_REOPENS, verify_candidate, verifier_model_for
from ..config import MODELS, live_enabled
from ..digitize.digitizer import digitize
from ..digitize.vlm import TargetSpec
from ..ingest.dedupe import PaperGroup, dedupe_pdfs
from ..ingest.pdf import FigureRegion, PaperRecord, ingest_pdf
from ..llm.client import LLMClient
from ..llm.context import upload_pdf
from ..llm.costs import cache_stats, cache_summary_line, cost_by_stage
from ..llm.errors import BudgetExceeded
from ..models import (Adjudication, Candidate, CheckFlag, DatasetSpec, EffectSizeRecord,
                      OrientationVerdict, OutcomeSources, PaperStatus, Protocol, RunManifest,
                      SourceKind, Source, StatsSettings, StudyMap, Verdict, VerifierVerdict)
from ..protocol import load_protocol
from ..report import (exclusions_table, extraction_table, methods_figure, pool_rows,
                      prisma_flow, provenance_bundle, route_examples, write_html_report,
                      write_outcome_outputs, write_rows)
from ..stats.meta import MetaResult
from ..verify.checks import CHECK_SEVERITY, run_checks
from ..verify.confidence import resolve_cell
from ..verify.vote import VoteResult, vote_groups
from .aggregate import AGGREGATED_FLAG, Aggregation, aggregate_one_row_per_paper
from .overrides import OVERRIDES_FILE, apply_overrides_and_repool, read_overrides
from .resolve import (ReportedValues, ResolvedValues, StatisticValues, apply_shared_control,
                      multi_group_flags, resolve_effect)
from .state import (PaperBudgetExceeded, PaperClient, emit, load_manifest, paper_dir,
                    read_stage, review_entry, save_manifest, sha12, sort_review_queue,
                    stage_done, write_stage)

__all__ = ["run_pipeline", "RunContext", "PaperResult", "revalidate", "target_for_source",
           "vote_candidates", "sample_key"]

FIGURE_KINDS = frozenset({SourceKind.figure_bar, SourceKind.figure_line, SourceKind.figure_points,
                          SourceKind.figure_box})


# ----------------------------------------------------------------------------- context
class RunCancelled(RuntimeError):
    """`cancel_event` was set: this paper stops here and the run finishes with what it has."""


@dataclass
class RunContext:
    protocol: Protocol
    out_dir: Path
    client: Any
    models: dict[str, str]
    resume: bool = True
    max_usd_per_paper: float | None = None
    progress: Callable[[dict[str, Any]], None] | None = None
    warnings: list[str] = field(default_factory=list)
    #: set by the caller (the UI's stop button) — checked between papers and between stages
    cancel_event: threading.Event | None = None
    #: how a paper is ingested; the server passes a subprocess-backed one so that a PDF built to
    #: hang a parser takes a child process with it instead of the run
    ingest_fn: Callable[[Path, Path], PaperRecord] = ingest_pdf

    def stop_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RunCancelled("cancelled by the reviewer")

    @property
    def settings(self) -> StatsSettings:
        return self.protocol.stats

    def outcome_keys(self) -> set[str]:
        return {o.key for o in self.protocol.outcomes}


@dataclass
class PaperResult:
    status: PaperStatus
    paper: PaperRecord | None = None
    study: StudyMap | None = None
    candidates: list[Candidate] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    records: list[EffectSizeRecord] = field(default_factory=list)
    exclusions: list[dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------------- digitiser target
def target_for_source(source: Source, dataset: DatasetSpec, outcome_sources: OutcomeSources,
                      protocol: Protocol, settings: StatsSettings) -> TargetSpec:
    """The mapper's figure `Source` as the digitiser's `TargetSpec`.

    Every hint is copied from the protocol or from what the mapper read in the paper; the x hint is
    the protocol's own measurement window, which is what makes "read the late part of the block"
    a protocol statement rather than something this code knows.
    """
    outcome = protocol.outcome(outcome_sources.outcome_key)
    return TargetSpec(
        outcome_key=outcome_sources.outcome_key,
        group_a_label=dataset.group_a.label or protocol.group_a.label,
        group_b_label=dataset.group_b.label or protocol.group_b.label,
        series_hint=outcome_sources.measure_name or outcome.label,
        x_hint=(outcome.measurement_window or "").strip(),
        panel_hint=source.locator,
        quantity="mean_and_error",
        error_bar_type_hint=getattr(source.error_bar_type, "value", str(source.error_bar_type)),
        unit_hint=outcome_sources.units or outcome.units_hint,
        late_window_sd=settings.late_window_sd,
        collapse_across_x=(source.x_axis_kind == "categorical"
                           and protocol.digitize.collapse_across_categorical_x),
        notes="; ".join(part for part in (source.quote, source.values_in_text, source.notes)
                        if part)[:400])


#: source kinds that put NUMBERS in characters somewhere a text reader could find them. A source
#: of an unrecognised kind counts: it is a location a human has to route, and refusing to look at
#: it would be the pipeline deciding that for them.
_PRINTED_KINDS = frozenset(SourceKind) - FIGURE_KINDS


def _has_printed_source(sources: Sequence[Source]) -> bool:
    """Is there anything on this cell for a text or statistic reader to read?"""
    return any(s.kind in _PRINTED_KINDS and not s.figure_id for s in sources)


def _figure(paper: PaperRecord, figure_id: str) -> FigureRegion | None:
    return next((f for f in paper.figures if f.id == figure_id), None)


# ----------------------------------------------------------------------------- stages
def _ingest(ctx: RunContext, group: PaperGroup, status: PaperStatus) -> PaperRecord:
    directory = paper_dir(ctx.out_dir, group.sha256) / "ingest"
    if ctx.resume and stage_done(ctx.out_dir, group.sha256, "ingest"):
        status.stages["ingest"] = "skipped"
        return PaperRecord.load(directory)
    paper = ctx.ingest_fn(group.representative, directory)
    write_stage(ctx.out_dir, group.sha256, "ingest", {
        "sha256": paper.sha256, "filename": paper.filename, "source_path": paper.source_path,
        "out_dir": str(directory), "n_pages": paper.n_pages, "title": paper.title,
        "doi": paper.doi, "n_figures": len(paper.figures), "n_tables": len(paper.tables),
        "has_text_layer": paper.has_text_layer,
        "duplicates": [str(p) for p in group.duplicates], "duplicate_reason": group.reason,
        "warnings": list(paper.warnings)})
    status.stages["ingest"] = "done"
    status.warnings.extend(paper.warnings)
    return paper


def _map(ctx: RunContext, paper: PaperRecord, group: PaperGroup,
         status: PaperStatus) -> tuple[StudyMap, str]:
    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "map"):
        payload = read_stage(ctx.out_dir, paper.sha256, "map")
        status.stages["map"] = "skipped"
        return StudyMap.model_validate(payload["study"]), str(payload.get("pdf_file_id") or "")
    # amendment B: the paper is uploaded once and the mapper's call warms the prompt cache before
    # any fan-out, so the extractors read a cached document rather than re-sending the PDF.
    file_id = ""
    if ctx.client.live:
        try:
            file_id = upload_pdf(ctx.client, group.representative,
                                 paper_dir(ctx.out_dir, paper.sha256))
        except Exception as exc:                           # an upload failure is not fatal
            status.warnings.append(f"PDF upload failed, falling back to inline base64: {exc}")
    study = map_study(ctx.client, paper, ctx.protocol,
                      model_primary=ctx.models["primary"], model_check=ctx.models["secondary"],
                      model_adjudicate=ctx.models["adjudicator"], pdf_file_id=file_id or None)
    write_stage(ctx.out_dir, paper.sha256, "map",
                {"study": study.model_dump(mode="json"), "pdf_file_id": file_id})
    status.stages["map"] = "done"
    return study, file_id


def _extract(ctx: RunContext, paper: PaperRecord, study: StudyMap,
             status: PaperStatus) -> list[Candidate]:
    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "extract"):
        payload = read_stage(ctx.out_dir, paper.sha256, "extract")
        status.stages["extract"] = "skipped"
        return [Candidate.model_validate(c) for c in payload["candidates"]]

    keys = ctx.outcome_keys()
    figures_dir = paper_dir(ctx.out_dir, paper.sha256) / "figures"
    candidates: list[Candidate] = []
    done: list[str] = []
    exhausted: list[str] = []
    stopped = ""

    def save(complete: bool) -> None:
        # written after EVERY cell, not once at the end: Buch 2003 spent $14.37 against a $14 cap
        # and the unwind threw away every candidate the stage had already paid for (F7). Until the
        # stage finishes the file says `"complete": false`, so `--resume` re-enters it rather than
        # reading a salvaged half-paper as a finished one.
        write_stage(ctx.out_dir, paper.sha256, "extract",
                    {"candidates": [c.model_dump(mode="json") for c in candidates],
                     "complete": complete, "cells_extracted": list(done),
                     "cells_budget_exhausted": list(exhausted),
                     "budget_note": stopped})

    for dataset in study.datasets:
        for sources in dataset.outcomes:
            if sources.outcome_key not in keys:
                status.warnings.append(
                    f"{dataset.dataset_id}: the mapper reported outcome "
                    f"{sources.outcome_key!r}, which is not in the protocol — skipped")
                continue
            cell = f"{dataset.dataset_id}/{sources.outcome_key}"
            if stopped:                       # the cap stops NEW cells; it does not undo old ones
                exhausted.append(cell)
                continue
            try:
                _extract_cell(ctx, paper, dataset, sources, figures_dir, status,
                              out=candidates)
            except PaperBudgetExceeded as exc:
                stopped = str(exc)
                exhausted.append(cell)
                status.warnings.append(
                    f"{cell}: budget_exhausted — {exc}; the rows this cell had already produced "
                    f"are kept, the cells after it were not started")
                continue
            done.append(cell)
            save(complete=False)
    if exhausted:
        status.warnings.append(
            f"budget_exhausted: {len(done)} cell(s) extracted, {len(exhausted)} not started "
            f"({', '.join(exhausted)}) — re-run with --resume once the cap is raised")
        status.error = stopped or "the paper's budget was exhausted during extraction"
    save(complete=not exhausted)
    status.stages["extract"] = "partial" if exhausted else "done"
    return candidates


def _extract_cell(ctx: RunContext, paper: PaperRecord, dataset: DatasetSpec,
                  sources: OutcomeSources, figures_dir: Path,
                  status: PaperStatus, out: list[Candidate] | None = None) -> list[Candidate]:
    """Both text variants, the statistic reader, and the digitiser once per figure source.

    `out` is filled as each reader answers rather than returned at the end, so a budget death half
    way through a cell keeps the readings that were already paid for.
    """
    key = sources.outcome_key
    out = out if out is not None else []
    # the two heterogeneous text readings the vote needs (different model AND different prompt) —
    # bought only when there is something printed to read. A cell whose only source is a figure
    # used to buy three text calls and get three `not_on_these_pages` answers back (critique
    # miss 9); the figure routes are what carry such a cell, and they are unaffected.
    if _has_printed_source(sources.sources):
        out.extend(extract_group_stats(ctx.client, paper, ctx.protocol, dataset, key,
                                       sources.sources, variant="table_first",
                                       model=ctx.models["primary"]))
        out.extend(extract_group_stats(ctx.client, paper, ctx.protocol, dataset, key,
                                       sources.sources, variant="narrative_first",
                                       model=ctx.models["secondary"]))
        out.extend(extract_test_statistics(ctx.client, paper, ctx.protocol, dataset, key,
                                           sources.sources, model=ctx.models["primary"]))
    else:
        status.warnings.append(
            f"{dataset.dataset_id}/{key}: every source the mapper found for this outcome is a "
            f"figure, so the text and statistic readers were not bought")
    for source in sources.sources:
        if source.kind not in FIGURE_KINDS and not source.figure_id:
            continue
        figure = _figure(paper, source.figure_id or "")
        if figure is None:
            status.warnings.append(
                f"{dataset.dataset_id}/{key}: the mapper pointed at figure "
                f"{source.figure_id or source.locator!r}, which ingestion did not find")
            continue
        target = target_for_source(source, dataset, sources, ctx.protocol, ctx.settings)
        digitised = digitize(ctx.client, paper, figure, target, source=source, dataset=dataset,
                             out_dir=figures_dir, caption=figure.caption,
                             models=(ctx.models["primary"],),
                             settings=ctx.protocol.digitize,
                             n_readouts=ctx.protocol.digitize.readouts_max,
                             cell_key=f"{paper.sha256[:12]}/{dataset.dataset_id}/{key}/"
                                      f"{figure.id}")
        out.extend(digitised if isinstance(digitised, list) else digitised.candidates)
    return out


@dataclass
class _CellVerification:
    verdicts: list[Verdict] = field(default_factory=list)
    extra_candidates: list[Candidate] = field(default_factory=list)
    adjudication: Adjudication | None = None
    orientation: OrientationVerdict | None = None
    reopens: int = 0
    #: the code-side source rank for this cell, and what it held back from the vote
    source_rank: list[dict[str, Any]] = field(default_factory=list)
    held_back: list[str] = field(default_factory=list)
    reopened_source: str = ""


def _cell_candidates(candidates: Sequence[Candidate], dataset_id: str,
                     outcome_key: str) -> list[Candidate]:
    return [c for c in candidates
            if c.dataset_id == dataset_id and c.outcome_key == outcome_key]


ENSEMBLE = "digitize:ensemble"


def vote_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """The candidates the verification layer may see: one figure reading per group, not five.

    `digitize()` returns a `Candidate` per (group, route sample) *and* one ensemble candidate per
    group. The route samples belong in the stage file and the provenance bundle — that is where a
    reviewer checks how the picture was measured — but they must not enter the vote: the
    digitiser's four or five ways of measuring one figure would otherwise outvote the value the
    paper printed, and the ensemble (amendment F's median-of-routes, with the per-route detail in
    its `pixel_provenance`) is already their consensus. Controller ruling, fix round 1.
    """
    return [c for c in candidates
            if not c.extractor_id.startswith("digitize:") or c.extractor_id == ENSEMBLE]


def _winner(candidates: Sequence[Candidate], result: VoteResult | None,
            group: str) -> Candidate | None:
    """The candidate the verifier should try to refute: an agreeing reading, best-grounded first."""
    mine = [c for c in candidates
            if c.group == group and c.kind == "group_stats" and c.status == "found"
            and c.mean is not None]
    if not mine:
        return None
    agreeing = [c for c in mine if result is not None and c.candidate_id in result.agreeing_ids]
    pool = agreeing or mine
    return sorted(pool, key=lambda c: (0 if c.grounded else 1, 0 if c.quote else 1,
                                       c.candidate_id))[0]


def _source_marker(source: Source) -> str:
    return source.figure_id or source.table_id or f"p{source.page}:{source.locator}"


def _already_read(cell: Sequence[Candidate], sources: Sequence[Source]) -> set[str]:
    """Which of the mapper's locations this cell already has a reading from.

    Figure ids are the reliable half; a text or table source is matched the way the checks match
    one (its own id, or the page a candidate was read from), because a verifier naming the page-4
    sentence that two text readers already answered on must not buy a third reading of it.
    """
    seen: set[str] = set()
    for cand in cell:
        source = source_of(cand, sources)
        if source is not None:
            seen.add(_source_marker(source))
    return seen


def _reopen_on_better_source(ctx: RunContext, paper: PaperRecord, dataset: DatasetSpec,
                             sources: OutcomeSources, verdicts: Sequence[VerifierVerdict],
                             cell: Sequence[Candidate], status: PaperStatus
                             ) -> tuple[str, list[Candidate]] | None:
    """Re-extract ONE source a verifier named, when the mapper already had it and nobody read it."""
    already = _already_read(cell, sources.sources)
    for verdict in verdicts:
        named = (verdict.better_source or "").strip()
        if not named:
            continue
        source = match_named_source(named, sources.sources)
        if source is None:
            status.warnings.append(
                f"{dataset.dataset_id}/{sources.outcome_key}: a verifier named {named!r} as a "
                f"better source, which is not in the mapper's list for this outcome — not "
                f"re-opened, because acting on it would mean acting on a location a model invented")
            continue
        if _source_marker(source) in already:
            continue                     # the cell already read it; re-reading buys nothing
        # the re-extraction's own warnings belong to the paper, not to a throwaway status object
        extra = _extract_cell(ctx, paper, dataset,
                              sources.model_copy(update={"sources": [source]}),
                              paper_dir(ctx.out_dir, paper.sha256) / "figures", status)
        if extra:
            return named, list(extra)
        status.warnings.append(
            f"{dataset.dataset_id}/{sources.outcome_key}: re-opened on {named!r} at the verifier's "
            f"suggestion and it produced no candidate")
    return None


def _verify_cell(ctx: RunContext, paper: PaperRecord, dataset: DatasetSpec,
                 sources: OutcomeSources, candidates: Sequence[Candidate],
                 all_candidates: Sequence[Candidate], file_id: str,
                 orientation: OrientationVerdict | None,
                 status: PaperStatus) -> _CellVerification:
    key = sources.outcome_key
    outcome_def = ctx.protocol.outcome(key)
    # the digitiser's per-route samples stay in the stage file; only its ensemble votes
    cell = vote_candidates(candidates)
    others = vote_candidates([c for c in all_candidates
                              if c.dataset_id != dataset.dataset_id or c.outcome_key != key])
    # …and of the locations the mapper found, only the best-scoring ones do (P5). Extraction has
    # already read them all; this decides which readings the VOTE weighs against each other, so a
    # sentence with no n and no dispersion cannot outvote a figure with SE bars and n printed.
    ranked = rank_sources(sources.sources, sources)
    out_rank = [row.to_dict() for row in ranked]
    cell, held_back = keep_for_vote(cell, ranked)
    n_a, n_b = dataset.group_a.n, dataset.group_b.n
    total_n = (n_a or 0) + (n_b or 0) or None
    out = _CellVerification(orientation=orientation, source_rank=out_rank,
                            held_back=list(held_back))

    flags = run_checks(dataset, key, cell, other_candidates=others, orientation=orientation,
                       total_n=total_n)
    votes = vote_groups(cell)

    # amendment G: the two text extractors disagreed, so buy a third cheap reading — the secondary
    # model on the prompt variant it has not seen — and let it move that route's median.
    if any(v.needs_third_candidate for v in votes.values()):
        third = extract_group_stats(ctx.client, paper, ctx.protocol, dataset, key, sources.sources,
                                    variant="table_first", model=ctx.models["secondary"])
        if third:
            out.extra_candidates.extend(third)
            cell = [*cell, *third]
            flags = run_checks(dataset, key, cell, other_candidates=others,
                               orientation=orientation, total_n=total_n)
            votes = vote_groups(cell)

    verifier_verdicts = []
    refuted = False
    for group in ("A", "B"):
        winner = _winner(cell, votes.get(group), group)
        if winner is None:
            continue
        reopen = 0
        while True:
            # a re-open is a *different* reader asking the same question; `reopen` is part of the
            # cache key, so it is a fresh answer rather than the cached one
            preferred = (ctx.models["secondary"], ctx.models["primary"],
                         ctx.models["adjudicator"])[min(reopen, 2)]
            verdict = verify_candidate(
                ctx.client, paper, winner, model=verifier_model_for(winner, preferred=preferred),
                protocol=ctx.protocol, dataset=dataset, outcome=outcome_def,
                pdf_file_id=file_id or None, reopen=reopen)
            verifier_verdicts.append(verdict)
            if verdict.verdict != "refuted":
                break
            refuted = True
            if reopen >= MAX_REOPENS:
                break
            reopen += 1
            out.reopens += 1

    # P5's second half: the verifier's `better_source` answer was written, stored, printed — and
    # acted on nowhere. It is acted on now, ONCE, and only when it names a location the mapper
    # already found: a re-open on a page number a model invented would be worse than not looking.
    reopened = _reopen_on_better_source(ctx, paper, dataset, sources, verifier_verdicts, cell,
                                        status)
    if reopened is not None:
        named, extra = reopened
        out.reopened_source = named
        out.extra_candidates.extend(extra)
        cell = [*cell, *vote_candidates(extra)]
        flags = run_checks(dataset, key, cell, other_candidates=others, orientation=orientation,
                           total_n=total_n)
        flags.append(CheckFlag(
            code="reopened_on_better_source", severity=CHECK_SEVERITY["reopened_on_better_source"],
            message=(f"a verifier named {named!r} as a better source for this outcome and "
                     f"extraction was re-opened on it (one hop, once)"),
            candidate_ids=sorted(c.candidate_id for c in vote_candidates(extra))))
        votes = vote_groups(cell)

    disagreed = any(v.agreement == "disagree" for v in votes.values())
    errors = any(f.severity == "error" for f in flags)
    if disagreed or refuted or errors:
        out.adjudication = adjudicate(
            ctx.client, paper, dataset, sources, cell, verdicts=verifier_verdicts, flags=flags,
            model=ctx.models["adjudicator"], protocol=ctx.protocol, outcome_def=outcome_def,
            votes=votes, pdf_file_id=file_id or None)

    for group in ("A", "B"):
        out.verdicts.append(resolve_cell(
            dataset, key, group, cell, vote_result=votes.get(group),
            verdicts=verifier_verdicts, flags=flags, adjudication=out.adjudication,
            orientation=orientation, other_candidates=others, n_a=n_a, n_b=n_b))
    return out


def _verify(ctx: RunContext, paper: PaperRecord, study: StudyMap, candidates: list[Candidate],
            file_id: str, status: PaperStatus) -> list[Verdict]:
    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "verify"):
        payload = read_stage(ctx.out_dir, paper.sha256, "verify")
        status.stages["verify"] = "skipped"
        candidates.extend(Candidate.model_validate(c)
                          for c in payload.get("extra_candidates", []))
        return [Verdict.model_validate(v) for v in payload["verdicts"]]

    keys = ctx.outcome_keys()
    verdicts: list[Verdict] = []
    extra: list[Candidate] = []
    orientations: dict[tuple[str, str], OrientationVerdict] = {}
    adjudications: list[dict[str, Any]] = []
    source_ranks: dict[str, Any] = {}
    reopens = 0
    for dataset in study.datasets:
        for sources in dataset.outcomes:
            if sources.outcome_key not in keys:
                continue
            # spec §3.3(5): the direction of a measure is decided once per (outcome, measure) by
            # two independent agents — not once per value.
            measure = (sources.outcome_key, sources.measure_name or "")
            if measure not in orientations:
                orientations[measure] = orientation_verdict(
                    ctx.client, paper, dataset, sources,
                    models=(ctx.models["primary"], ctx.models["secondary"]),
                    protocol=ctx.protocol, outcome=ctx.protocol.outcome(sources.outcome_key),
                    pdf_file_id=file_id or None)
            cell = _cell_candidates([*candidates, *extra], dataset.dataset_id,
                                    sources.outcome_key)
            result = _verify_cell(ctx, paper, dataset, sources, cell, [*candidates, *extra],
                                  file_id, orientations[measure], status)
            extra.extend(result.extra_candidates)
            verdicts.extend(result.verdicts)
            reopens += result.reopens
            source_ranks[f"{dataset.dataset_id}|{sources.outcome_key}"] = {
                "ranked": result.source_rank, "held_back_from_vote": result.held_back,
                "reopened_on_better_source": result.reopened_source}
            if result.adjudication is not None:
                adjudications.append(result.adjudication.model_dump(mode="json"))
    candidates.extend(extra)
    write_stage(ctx.out_dir, paper.sha256, "verify", {
        "verdicts": [v.model_dump(mode="json") for v in verdicts],
        "extra_candidates": [c.model_dump(mode="json") for c in extra],
        "orientation": {f"{k[0]}|{k[1]}": v.model_dump(mode="json")
                        for k, v in orientations.items()},
        "adjudications": adjudications,
        "source_rank": source_ranks,
        "reopens": reopens})
    status.stages["verify"] = "done"
    return verdicts


#: on the ROW, so a sensitivity analysis can pool with and without the rows whose dispersion the
#: code built rather than the paper stated (task 16 P6: the critique's amendment to the statistic)
DISPERSION_APPROXIMATED = "dispersion_approximated"


def _approximation_flags(cell: Sequence[Candidate]) -> list[str]:
    """Row flags for anything in this cell whose dispersion is an approximation, not a reading."""
    kinds = {str((c.pixel_provenance or {}).get("dispersion_approximation") or "")
             for c in cell if c.extractor_id == ENSEMBLE}
    named = sorted(k for k in kinds if k)
    return [DISPERSION_APPROXIMATED, *[f"{DISPERSION_APPROXIMATED}:{k}" for k in named]] \
        if named else []


def sample_key(dataset: DatasetSpec, paper_id: str) -> str:
    """Which PARTICIPANT sample this dataset came from — or `""` when none can be claimed.

    `one_row_per_paper` has to know whether a paper's two rows are two samples (combine them as
    independent) or the same people twice (combine them as dependent, with a correlation). The
    only honest source for that is the mapper's own description of the dataset, and its contract
    (`canopy/llm/prompts/mapper.md`) is exactly this: a dataset is "one independent participant
    sample under one condition", `experiment` is the paper's own label, and `exposure_order` says
    whether these data are the participants' FIRST exposure.

    So a dataset claims its own sample only when the paper labelled the experiment it belongs to
    and the mapper called it a first exposure. Everything else — an unlabelled experiment, a
    repeated exposure, a counterbalanced set — returns `""`, which the aggregation reads as "same
    people, treat as dependent". Guessing the other way would understate the variance.
    """
    experiment = (dataset.experiment or "").strip()
    if not experiment or str(dataset.exposure_order) != "first":
        return ""
    return f"{paper_id}|{experiment}"


def _statistic_values(candidates: Sequence[Candidate]) -> StatisticValues | None:
    """The best statistic the paper printed for this contrast — t/F first, then p."""
    stats = [c for c in candidates if c.kind == "test_statistic" and c.status == "found"
             and c.admissible]
    ranked = sorted(stats, key=lambda c: ({"t": 0, "F": 1, "p": 2}.get(str(c.stat_type), 3),
                                          c.candidate_id))
    for cand in ranked:
        if cand.stat_value is not None or cand.p_value is not None:
            return StatisticValues(
                stat_type=cand.stat_type or "unknown", value=cand.stat_value, df=cand.df,
                df1=cand.df1, df2=cand.df2, tails=cand.tails, p_kind=cand.p_kind or "unknown",
                p_value=cand.p_value, design=cand.design, direction=cand.direction)
    return None


def _reported_values(candidates: Sequence[Candidate]) -> ReportedValues | None:
    for cand in candidates:
        if cand.kind == "reported_d" and cand.status == "found" and cand.reported_value is not None:
            return ReportedValues(value=cand.reported_value, scale=cand.reported_scale,
                                  standardizer=cand.standardizer, ci_low=cand.reported_ci_low,
                                  ci_high=cand.reported_ci_high,
                                  positive_means=cand.positive_means)
    return None


def _resolve(ctx: RunContext, paper: PaperRecord, study: StudyMap,
             candidates: Sequence[Candidate], verdicts: Sequence[Verdict],
             status: PaperStatus) -> list[EffectSizeRecord]:
    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "resolve"):
        payload = read_stage(ctx.out_dir, paper.sha256, "resolve")
        status.stages["resolve"] = "skipped"
        return [EffectSizeRecord.model_validate(r) for r in payload["records"]]

    keys = ctx.outcome_keys()
    by_cell = {(v.dataset_id, v.outcome_key, v.group): v for v in verdicts}
    prepared: list[tuple[DatasetSpec, str, ResolvedValues]] = []
    for dataset in study.datasets:
        for sources in dataset.outcomes:
            key = sources.outcome_key
            if key not in keys:
                continue
            verdict_a = by_cell.get((dataset.dataset_id, key, "A"))
            verdict_b = by_cell.get((dataset.dataset_id, key, "B"))
            if verdict_a is None or verdict_b is None:
                status.warnings.append(f"{dataset.dataset_id}/{key}: verification produced no "
                                       f"verdict for both groups — not resolved")
                continue
            cell = _cell_candidates(candidates, dataset.dataset_id, key)
            values = ResolvedValues.from_verdicts(
                verdict_a, verdict_b, test_statistic=_statistic_values(cell),
                reported=_reported_values(cell))
            values.flags = sorted(set(values.flags)
                                  | set(multi_group_flags(dataset, ctx.settings.multi_group_policy))
                                  | set(_approximation_flags(cell)))
            prepared.append((dataset, key, values))

    # rows in one paper that share a control arm are not independent (Cochrane 16.5.4)
    shared: dict[tuple[str, str], list[int]] = {}
    for index, (dataset, key, _) in enumerate(prepared):
        if dataset.shared_control:
            shared.setdefault((dataset.cluster_id or paper.sha256, key), []).append(index)
    for indices in shared.values():
        if len(indices) < 2:
            continue
        adjusted = apply_shared_control([prepared[i][2] for i in indices],
                                        ctx.settings.shared_control_strategy)
        for position, index in enumerate(indices):
            if position < len(adjusted):
                dataset, key, _ = prepared[index]
                prepared[index] = (dataset, key, adjusted[position])

    records: list[EffectSizeRecord] = []
    for dataset, key, values in prepared:
        record = resolve_effect(dataset, ctx.protocol.outcome(key), values, ctx.settings)
        record.paper_id = paper.sha256
        record.cluster_id = record.cluster_id or paper.sha256
        record.sample_id = sample_key(dataset, paper.sha256)
        record.citation = study.citation
        record.label = record.label or dataset.label or dataset.dataset_id
        verdict_a = by_cell.get((dataset.dataset_id, key, "A"))
        if verdict_a is not None and verdict_a.analysis_metric != "unknown":
            record.analysis_metric = verdict_a.analysis_metric
        records.append(record)

    write_stage(ctx.out_dir, paper.sha256, "resolve",
                {"records": [r.model_dump(mode="json") for r in records]})
    status.stages["resolve"] = "done"
    return records


# ----------------------------------------------------------------------------- one paper
def _run_paper(ctx: RunContext, group: PaperGroup) -> PaperResult:
    started = time.perf_counter()
    status = PaperStatus(paper_id=group.sha256, filename=Path(group.representative).name)
    result = PaperResult(status=status)
    paper_client = PaperClient(ctx.client, group.sha256, ctx.max_usd_per_paper)
    paper_ctx = RunContext(protocol=ctx.protocol, out_dir=ctx.out_dir, client=paper_client,
                           models=ctx.models, resume=ctx.resume,
                           max_usd_per_paper=ctx.max_usd_per_paper, progress=ctx.progress,
                           cancel_event=ctx.cancel_event, ingest_fn=ctx.ingest_fn)
    label = sha12(group.sha256)
    try:
        ctx.stop_if_cancelled()                            # before this paper starts at all
        emit(ctx.progress, "ingest", label, "started", cost_so_far=ctx.client.total_cost())
        paper = _ingest(paper_ctx, group, status)
        result.paper = paper
        status.status = "ingested"
        for duplicate in group.duplicates:
            result.exclusions.append({"paper_id": group.sha256, "filename": Path(duplicate).name,
                                      "stage": "ingest", "reason": "duplicate",
                                      "quote": "", "decider": "dedupe",
                                      "detail": f"same {group.reason} as {status.filename}"})
        emit(ctx.progress, "ingest", label, "done", cost_so_far=ctx.client.total_cost(),
             message=f"{paper.n_pages} pages, {len(paper.figures)} figures")

        ctx.stop_if_cancelled()                            # …and between every two stages
        emit(ctx.progress, "map", label, "started", cost_so_far=ctx.client.total_cost())
        study, file_id = _map(paper_ctx, paper, group, status)
        result.study = study
        status.status = "mapped"
        status.eligible = study.eligible
        # `StudyMap.needs_human` was written by the mapper and read by nothing outside the browser,
        # so every objection it raised — a wrong outcome key, a conflicting unit, a group mapping
        # two agents could not agree on, "no dataset in an eligible paper, twice" — died where it
        # was made. An objection that reaches no output is the same as no objection at all.
        for note in study.needs_human:
            status.warnings.append(f"map: {note}")
        if study.eligible is False:
            status.status = "excluded"
            result.exclusions.append({
                "paper_id": group.sha256, "filename": status.filename, "stage": "map",
                "reason": "not_eligible", "quote": study.eligibility_rationale,
                "decider": "mapper", "detail": study.exclusion_reason})
            emit(ctx.progress, "map", label, "excluded",
                 cost_so_far=ctx.client.total_cost(), message=study.exclusion_reason)
            return result
        if not study.datasets:
            # Eligible and empty. Without this the paper leaves the run without appearing
            # anywhere at all — no row in the forest plot, none in the review queue, none in the
            # exclusions table — and a reader counting papers would never learn it was dropped.
            # A paper that contributes nothing has to say so and give its reason.
            status.status = "excluded"
            why = "; ".join(study.disagreements[-2:]) or "the mapper listed no dataset"
            status.warnings.append(f"eligible but no dataset was mapped: {why}")
            result.exclusions.append({
                "paper_id": group.sha256, "filename": status.filename, "stage": "map",
                "reason": "no_usable_data:no_datasets_mapped",
                "quote": study.eligibility_rationale, "decider": "mapper",
                "detail": f"the paper was called eligible but no dataset was mapped, so it "
                          f"contributes no row: {why}"})
            emit(ctx.progress, "map", label, "excluded", cost_so_far=ctx.client.total_cost(),
                 message="eligible, but no dataset was mapped")
            return result
        emit(ctx.progress, "map", label, "done", cost_so_far=ctx.client.total_cost(),
             message=f"{len(study.datasets)} datasets")

        ctx.stop_if_cancelled()                            # …and between every two stages
        emit(ctx.progress, "extract", label, "started", cost_so_far=ctx.client.total_cost())
        result.candidates = _extract(paper_ctx, paper, study, status)
        status.status = "extracted"
        emit(ctx.progress, "extract", label, "done", cost_so_far=ctx.client.total_cost(),
             message=f"{len(result.candidates)} candidates")
        if status.stages.get("extract") == "partial":
            # the rows are on `result` and in the stage file; the paper stops here rather than
            # walking into a verify stage whose first call would raise anyway
            raise PaperBudgetExceeded(status.error)

        ctx.stop_if_cancelled()                            # …and between every two stages
        emit(ctx.progress, "verify", label, "started", cost_so_far=ctx.client.total_cost())
        result.verdicts = _verify(paper_ctx, paper, study, result.candidates, file_id, status)
        status.status = "verified"
        emit(ctx.progress, "verify", label, "done", cost_so_far=ctx.client.total_cost(),
             message=f"{sum(1 for v in result.verdicts if v.needs_human)} cells need a human")

        ctx.stop_if_cancelled()                            # …and between every two stages
        emit(ctx.progress, "resolve", label, "started", cost_so_far=ctx.client.total_cost())
        result.records = _resolve(paper_ctx, paper, study, result.candidates, result.verdicts,
                                  status)
        status.status = "resolved"
        emit(ctx.progress, "resolve", label, "done", cost_so_far=ctx.client.total_cost(),
             message=f"{len(result.records)} effect sizes")
    except RunCancelled:
        status.status = "cancelled"
        status.error = "cancelled by the reviewer"
        emit(ctx.progress, "paper", label, "cancelled", cost_so_far=ctx.client.total_cost(),
             message="stopped before the next stage; --resume will carry on from here")
    except PaperBudgetExceeded as exc:
        status.status = "error"
        status.error = str(exc)
        status.warnings.append(str(exc))
        _gone(result, group, status, "budget_exhausted", str(exc))
        emit(ctx.progress, "paper", label, "budget", cost_so_far=ctx.client.total_cost(),
             message=str(exc))
    except BudgetExceeded as exc:
        status.status = "error"
        status.error = f"run budget exhausted: {exc}"
        _gone(result, group, status, "budget_exhausted", status.error)
        emit(ctx.progress, "paper", label, "budget", cost_so_far=ctx.client.total_cost(),
             message=status.error)
    except Exception as exc:                               # one paper's failure is not the run's
        status.status = "error"
        status.error = f"{type(exc).__name__}: {exc}"
        status.warnings.append(traceback.format_exc(limit=4))
        _gone(result, group, status, "error", status.error)
        emit(ctx.progress, "paper", label, "error", cost_so_far=ctx.client.total_cost(),
             message=status.error)
    finally:
        status.cost_usd = round(paper_client.cost_usd, 6)
        status.seconds = round(time.perf_counter() - started, 3)
    return result


# ----------------------------------------------------------------------------- pooling
@dataclass
class _Split:
    """One outcome's rows, sorted into what is pooled, what is held, and what was combined away."""

    primary: list[EffectSizeRecord] = field(default_factory=list)
    held: list[EffectSizeRecord] = field(default_factory=list)
    every: list[EffectSizeRecord] = field(default_factory=list)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _split_rows(records: Sequence[EffectSizeRecord], settings: StatsSettings) -> _Split:
    """Apply the protocol's own rules: confidence first, then amendment A's `one_row_per_paper`.

    Aggregation happens AFTER the confidence filter — a row a human still has to look at is not
    combined into a number, it is held for that human — and it replaces the rows it combined, so
    a paper contributes exactly one row per outcome. Every replaced row stays in `every` (the
    extraction table shows it, with `primary_row` false) and in `exclusions`.
    """
    admitted = list(settings.primary_analysis_includes)
    split = _Split(every=list(records))
    for record in records:
        if record.confidence in admitted and record.es is not None and record.var:
            split.primary.append(record)
        else:
            split.held.append(record)
    if settings.one_row_per_paper and split.primary:
        aggregated: Aggregation = aggregate_one_row_per_paper(split.primary, settings)
        composites = [r for r in aggregated.rows if AGGREGATED_FLAG in r.flags]
        split.primary = aggregated.rows
        split.every = [*split.every, *composites]
        split.exclusions.extend(aggregated.exclusions)
        split.notes.extend(f"one_row_per_paper: {note}" for note in aggregated.notes)
    return split


def _primary_rows(records: Sequence[EffectSizeRecord], settings: StatsSettings
                  ) -> tuple[list[EffectSizeRecord], list[EffectSizeRecord], list[str]]:
    """`(pooled, held back, notes)` — the shape the server's re-pool also calls."""
    split = _split_rows(records, settings)
    return split.primary, split.held, split.notes


def _pool(rows: Sequence[EffectSizeRecord], settings: StatsSettings) -> MetaResult | None:
    """The one pooler: `canopy.report.tables.pool_rows`, which drops what cannot be pooled."""
    return pool_rows(rows, settings)


# ----------------------------------------------------------------------------- the run
def _discover(papers_dir: str | Path) -> list[Path]:
    directory = Path(papers_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such papers directory: {directory}")
    return sorted({p for p in directory.rglob("*") if p.suffix.lower() == ".pdf" and p.is_file()})


def run_pipeline(papers_dir: str | Path, protocol_path: str | Path, out_dir: str | Path, *,
                 models: dict[str, str] | None = None, budget_usd: float | None = None,
                 resume: bool = True, max_papers: int | None = None, concurrency: int = 4,
                 progress: Callable[[dict[str, Any]], None] | None = None,
                 max_usd_per_paper: float | None = None,
                 client: LLMClient | None = None,
                 allow_live: bool | None = None,
                 cancel_event: threading.Event | None = None,
                 ingest_fn: Callable[[Path, Path], PaperRecord] | None = None) -> RunManifest:
    """Run the whole review and write `<out_dir>`; returns the manifest it saved.

    `resume=True` (the default) skips any stage whose file already exists, so re-running after a
    budget stop, a crash or a new paper costs only what is genuinely new.

    `cancel_event` stops the run at the next paper or stage boundary: the papers that had not
    finished end `status="cancelled"`, everything that did finish is written, and `--resume`
    carries on from there. `ingest_fn(pdf, out_dir) -> PaperRecord` replaces the ingestion call —
    the server passes one that runs in a child process with a timeout, so a hostile PDF cannot
    hang the run.
    """
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(protocol_path)
    chosen = {**MODELS, **(models or {})}

    if client is None:
        client = LLMClient(cache_dir=out / "cache", budget_usd=budget_usd,
                           max_concurrency=max(6, concurrency * 2),
                           allow_live=bool(allow_live) if allow_live is not None
                           else live_enabled())
    ctx = RunContext(protocol=protocol, out_dir=out, client=client, models=chosen, resume=resume,
                     max_usd_per_paper=max_usd_per_paper, progress=progress,
                     cancel_event=cancel_event, ingest_fn=ingest_fn or ingest_pdf)

    paths = _discover(papers_dir)
    emit(progress, "discover", "", "done", message=f"{len(paths)} PDF file(s)")
    groups = dedupe_pdfs(list(paths))
    n_duplicates = sum(len(g.duplicates) for g in groups)
    if max_papers is not None:
        groups = groups[:max_papers]

    manifest = RunManifest(
        run_id=out.name or "run",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        protocol_hash=protocol.hash(), protocol_path=str(Path(protocol_path)),
        canopy_version=_version(), git_commit=_git_commit(), settings=protocol.stats,
        models=chosen)

    results: list[PaperResult] = []
    if groups:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = list(pool.map(lambda g: _run_paper(ctx, g), groups))
    manifest.papers = [r.status for r in results]
    manifest.warnings.extend(ctx.warnings)
    for result in results:
        manifest.warnings.extend(f"{sha12(result.status.paper_id)}: {w}"
                                 for w in result.status.warnings if not w.startswith("Traceback"))

    # The spend has to be on the manifest BEFORE the report is built: `report.html` and the
    # methods paragraph read `n_llm_calls` and `cost_usd` off it, and building them first printed
    # "0 model calls were made at a cost of $0.00" into every run's methods text while the
    # per-paper rows of the same page showed the real figures. A methods paragraph is written to
    # be pasted into a paper, so a wrong number there is published.
    _account(manifest, client)
    outputs = _write_outputs(ctx, manifest, results, paths, groups, n_duplicates)
    manifest.outputs = {k: str(Path(v).relative_to(out)) if Path(v).is_relative_to(out) else str(v)
                        for k, v in outputs.items()}
    _account(manifest, client)               # …and again, so the saved manifest counts the writing
    manifest.seconds = round(time.perf_counter() - started, 3)
    save_manifest(out, manifest)
    if read_overrides(out):
        # amendment I: this run has just rebuilt every artefact from the stage files, so a
        # reviewer's decisions would be undone by it. The log outlives what it changes — it is
        # re-applied here, which is what makes an override survive `canopy run --resume`.
        summary = apply_overrides_and_repool(out)
        emit(progress, "review", "", "done", cost_so_far=manifest.cost_usd,
             message=f"{summary['applied']} override(s) re-applied from {OVERRIDES_FILE}")
        manifest = load_manifest(out)
    emit(progress, "cache", "", "done", cost_so_far=manifest.cost_usd,
         message=cache_summary_line(manifest.cache))
    emit(progress, "run", "", "done", cost_so_far=manifest.cost_usd,
         message=f"{len(results)} paper(s), ${manifest.cost_usd:.2f}, "
                 f"{manifest.n_llm_calls} calls")
    return manifest


def _write_outputs(ctx: RunContext, manifest: RunManifest, results: Sequence[PaperResult],
                   paths: Sequence[Path], groups: Sequence[PaperGroup],
                   n_duplicates: int) -> dict[str, Path]:
    out = ctx.out_dir
    settings = ctx.settings
    records = [r for result in results for r in result.records]
    candidates = [c for result in results for c in result.candidates]
    verdicts = [v for result in results for v in result.verdicts]
    papers = [result.paper for result in results if result.paper is not None]
    exclusions = [e for result in results for e in result.exclusions]

    outputs: dict[str, Path] = {}
    review: list[dict[str, Any]] = []
    per_outcome: dict[str, dict[str, Any]] = {}
    by_dataset = {(r.dataset_id, r.outcome_key): r for r in records}

    every_row: list[EffectSizeRecord] = []
    primary_rows: list[EffectSizeRecord] = []
    for outcome in ctx.protocol.outcomes:
        mine = [r for r in records if r.outcome_key == outcome.key]
        split = _split_rows(mine, settings)
        primary, held = split.primary, split.held
        manifest.warnings.extend(f"{outcome.key}: {n}" for n in split.notes)
        exclusions.extend(split.exclusions)
        every_row.extend(split.every)
        primary_rows.extend(primary)
        pooled = _pool(primary, settings)
        emit(ctx.progress, "pool", "", "done", cost_so_far=ctx.client.total_cost(),
             message=f"{outcome.key}: k={len(primary)}, held={len(held)}")
        artefacts = write_outcome_outputs(out, outcome, primary, pooled, settings,
                                          needs_human_rows=held, verdicts=verdicts,
                                          candidates=candidates,
                                          moderators=ctx.protocol.moderators or None,
                                          all_rows=split.every)
        outputs.update({f"{outcome.key}.{k}": v for k, v in artefacts.items()})
        per_outcome[outcome.key] = {"pooled": pooled, "outputs": artefacts, "rows": primary,
                                    "needs_human_rows": held}
        for verdict in verdicts:
            if verdict.outcome_key != outcome.key or not verdict.needs_human:
                continue
            record = by_dataset.get((verdict.dataset_id, verdict.outcome_key))
            paper_id = record.paper_id if record is not None else ""
            review.append(review_entry(verdict, paper_id=paper_id, candidates=candidates,
                                       record=record, primary=primary, settings=settings))
        for record in mine:
            if record.route == "not_convertible":
                exclusions.append({
                    "paper_id": record.paper_id, "filename": "",
                    "dataset_id": record.dataset_id, "outcome_key": record.outcome_key,
                    "stage": "resolve", "reason": "not_convertible",
                    "quote": "", "decider": "code", "detail": record.not_convertible_reason})

    # one table for the whole run beside the per-outcome ones: every row of every outcome, with
    # the raw values, the route and whether it was pooled — the file a reviewer opens first
    outputs.update({f"extraction_table_all.{k}": v for k, v in extraction_table(
        every_row, out / "results" / "extraction_table_all", verdicts=verdicts,
        candidates=candidates, primary=primary_rows).items()})

    manifest.human_review_queue = sort_review_queue(review)
    if manifest.human_review_queue:
        outputs.update({f"human_review_queue.{k}": v for k, v in write_rows(
            manifest.human_review_queue, out / "human_review_queue",
            ["paper_id", "dataset_id", "outcome_key", "group", "confidence", "route", "reason",
             "impact_abs_delta_pooled", "candidates"], formats=("csv", "json")).items()})

    outputs.update({f"exclusions.{k}": v for k, v in
                    exclusions_table(exclusions, out / "exclusions").items()})

    counts = _prisma_counts(results, paths, n_duplicates, records, exclusions)
    outputs.update({f"prisma.{k}": v for k, v in prisma_flow(counts, out / "prisma").items()})

    examples = route_examples(records, candidates=candidates, verdicts=verdicts, papers=papers,
                              out_dir=out / "thumbnails", max_examples=3)
    outputs.update({f"methods_fig.{k}": v for k, v in
                    methods_figure(records, out / "methods_routes",
                                   examples=examples).items()})

    # the provenance bundle covers the values that actually reached a verdict, not every reading
    cited = {cid for verdict in verdicts for cid in verdict.candidate_ids}
    chosen = [c for c in candidates if c.candidate_id in cited]
    bundle = provenance_bundle(papers, chosen, out / "provenance") if papers and chosen else None
    if bundle is not None:
        outputs["provenance.json"] = bundle["json"]

    report = write_html_report(out, manifest, ctx.protocol, results=per_outcome,
                               review_queue=manifest.human_review_queue, exclusions=exclusions,
                               provenance=None if bundle is None else bundle["entries"],
                               run_outputs={"methods_fig_png": outputs.get("methods_fig.png"),
                                            "prisma_png": outputs.get("prisma.png"),
                                            "provenance_json": outputs.get("provenance.json")})
    outputs.update({f"report.{k}": v for k, v in report.items()})
    return outputs


def _gone(result: PaperResult, group: Any, status: PaperStatus, reason: str,
          detail: str) -> None:
    """A paper that died and left no row still has to appear in the exclusions table.

    Run 1's third paper failed after mapping, having spent $14.37, and showed up in no output at
    all — not as a row, not as a held row, not as an exclusion — so a reader counting papers had
    no way to learn it had been dropped or what it cost. A paper that kept some rows is not
    excluded, so this stays silent for the partial case that `d9fe6fe` protects.
    """
    if result.records:
        return
    result.exclusions.append({
        "paper_id": getattr(group, "sha256", ""), "filename": status.filename, "stage": "run",
        "reason": reason, "quote": "", "decider": "code", "detail": detail})


def _account(manifest: RunManifest, client: LLMClient) -> None:
    """Copy the client's spend onto the manifest. Called before the report and again after it."""
    calls = client.calls()
    manifest.n_llm_calls = len(calls)
    manifest.cache_hits = sum(1 for c in calls if c.get("cached"))
    manifest.cost_by_stage = cost_by_stage(calls)
    manifest.cache = cache_stats(calls)
    manifest.cost_usd = round(client.total_cost(), 6)


def _prisma_counts(results: Sequence[PaperResult], paths: Sequence[Path], n_duplicates: int,
                   records: Sequence[EffectSizeRecord],
                   exclusions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The PRISMA chain, counted so that it still adds up under `--max-papers`.

    `unique_papers` is every unique paper the folder held, not the subset this run looked at, and
    the ones a `--max-papers` cap left out are their own count (`not_processed`). Folding them
    into "excluded" would claim the review rejected papers it never opened.
    """
    eligible = [r for r in results if r.status.eligible]
    datasets = sum(len(r.study.datasets) for r in results if r.study is not None)
    included = [r for r in records if r.route != "not_convertible"]
    unique = max(0, len(paths) - n_duplicates)
    reasons: dict[str, int] = {}
    for entry in exclusions:
        # `aggregated_into:<row>` counts as `aggregated`, the same head the exclusions table uses
        name = str(entry.get("reason", "other") or "other").split(":", 1)[0]
        reasons[name] = reasons.get(name, 0) + 1
    return {
        "files": len(paths), "duplicates_removed": n_duplicates,
        "unique_papers": unique,
        "not_processed": max(0, unique - len(results)),
        "papers_excluded": len(results) - len(eligible),
        "eligible_papers": len(eligible),
        "datasets": datasets,
        "datasets_excluded": max(0, datasets - len({r.dataset_id for r in included})),
        "included_datasets": len({r.dataset_id for r in included}),
        "included_papers": len({r.paper_id for r in included}),
        "exclusion_reasons": reasons,
    }


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("canopy-meta")
    except Exception:                                      # pragma: no cover - not installed
        return ""


def _git_commit() -> str:
    import subprocess

    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                      # pragma: no cover - no git
        return ""


# ----------------------------------------------------------------------------- validate
def revalidate(run_dir: str | Path, protocol_path: str | Path | None = None) -> dict[str, Any]:
    """Re-pool a finished run from its own stage files, with no model calls at all.

    This is what `canopy validate` runs: it proves that the numbers in the report follow from the
    stage files that are on disk, and lists any artefact the manifest promises but cannot show.
    """
    out = Path(run_dir)
    manifest = load_manifest(out)
    # the run's OWN copy first: `canopy run` writes `<run_dir>/protocol.yaml`, and that is the
    # protocol these stage files were produced under. `manifest.protocol_path` points at wherever
    # the user's file was at the time, which may have moved, changed or never existed on this
    # machine — a run must be re-poolable from the directory alone.
    local = out / "protocol.yaml"
    protocol = load_protocol(protocol_path or (local if local.exists()
                                               else manifest.protocol_path))
    settings = protocol.stats
    records: list[EffectSizeRecord] = []
    stages_missing: list[str] = []
    for status in manifest.papers:
        if status.status in ("excluded", "error"):
            continue
        try:
            payload = read_stage(out, status.paper_id, "resolve")
        except FileNotFoundError:
            stages_missing.append(f"{sha12(status.paper_id)}/resolve.json")
            continue
        records.extend(EffectSizeRecord.model_validate(r) for r in payload["records"])

    outcomes: dict[str, Any] = {}
    for outcome in protocol.outcomes:
        mine = [r for r in records if r.outcome_key == outcome.key]
        primary, held, _notes = _primary_rows(mine, settings)
        pooled = _pool(primary, settings)
        outcomes[outcome.key] = {
            "k": 0 if pooled is None else pooled.k, "n_needs_human": len(held),
            "estimate": None if pooled is None else pooled.estimate,
            "ci_low": None if pooled is None else pooled.ci_low,
            "ci_high": None if pooled is None else pooled.ci_high,
        }
    missing = [name for name, rel in manifest.outputs.items() if not (out / rel).exists()]
    protocol_changed = protocol.hash() != manifest.protocol_hash
    return {"run_dir": str(out), "protocol_hash": protocol.hash(),
            "protocol_matches_manifest": not protocol_changed,
            "records": len(records), "outcomes": outcomes,
            "missing_outputs": sorted(missing), "missing_stages": stages_missing,
            "ok": not missing and not stages_missing and not protocol_changed}
