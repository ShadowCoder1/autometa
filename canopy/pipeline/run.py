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
from types import MappingProxyType
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..agents.adjudicator import adjudicate
from ..agents.extract_stats import extract_test_statistics
from ..agents.extract_text import extract_group_stats
from ..agents.mapper import (HUMAN_DECIDER, HUMAN_EXCLUSION_RULE, MAP_ADJUDICATOR,
                             apply_map_answers,
                             extraction_blocks, map_study, readable_sources,
                             source_unreadable_reason)
from ..agents.orientation import combine_orientation
from ..agents.orientation import orientation as orientation_verdict
from ..agents.orientation import TIEBREAK_MODEL, tiebreak_ballot
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
from ..llm.errors import BudgetExceeded, LLMError, TruncatedOutput
from ..models import (Adjudication, Candidate, CheckFlag, DatasetSpec, EffectSizeRecord,
                      OrientationVerdict, OutcomeSources, PaperStatus, Protocol, RunManifest,
                      SourceKind, Source, StatsSettings, StudyMap, Verdict, VerifierVerdict)
from ..protocol import load_protocol
from ..report import (exclusions_table, extraction_table, methods_figure, pool_rows,
                      prisma_flow, provenance_bundle, route_examples, write_html_report,
                      write_outcome_outputs, write_rows)
from ..stats.meta import MetaResult
from ..verify.checks import (CHECK_SEVERITY, DF_PROVENANCE_FLAGS, ORIENTATION_FLAGS,
                            n_before_exclusions, run_checks)
from ..verify.confidence import ROW_REFUSAL_CODES, resolve_cell
from ..verify.panels import apply_panel_check
from ..verify.vote import (LOCATOR_CONFLICT, LOCATOR_CONFLICT_NOTE, VoteResult,
                           model_family, vote_groups)
from .aggregate import AGGREGATED_FLAG, Aggregation, aggregate_one_row_per_paper
from .overrides import (OVERRIDES_FILE, apply_overrides_and_repool, map_answers, read_overrides)
from .resolve import resolve_effect_with_fallback
from .rows import (DISPERSION_APPROXIMATED, approximation_flags, cell_candidates, prepare_rows,
                   reported_values, statistic_values, vote_candidates)
from .state import (PaperBudgetExceeded, PaperClient, emit, load_manifest, paper_dir,
                    read_stage, review_entry, save_manifest, sha12, sort_review_queue,
                    stage_done, write_stage)

__all__ = ["run_pipeline", "RunContext", "PaperResult", "revalidate", "target_for_source",
           "vote_candidates", "sample_key", "cells_for_review", "REVIEW_QUEUE_COLUMNS",
           "buys_adjudication"]

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
    #: C2: buy ONE more orientation ballot per (paper, outcome, measure) when the two readers and
    #: the free deterministic check have all failed to settle a direction and this cell's raw
    #: means exist. On by default — a measure nobody settles is a row that cannot be signed, and
    #: the ballot is the cheapest thing in the run that can remove such a question. `--no-tiebreak`
    #: reproduces the behaviour before C2 exactly: no call, no cost, the question stays open.
    tiebreak: bool = True
    max_usd_per_paper: float | None = None
    progress: Callable[[dict[str, Any]], None] | None = None
    warnings: list[str] = field(default_factory=list)
    #: set by the caller (the UI's stop button) — checked between papers and between stages
    cancel_event: threading.Event | None = None
    #: how a paper is ingested; the server passes a subprocess-backed one so that a PDF built to
    #: hang a parser takes a child process with it instead of the run
    ingest_fn: Callable[[Path, Path], PaperRecord] = ingest_pdf
    #: this paper's answers to its map's open questions, read ONCE and shared by the three stages
    #: that ask for them (review L8). `_run_paper` builds one context per paper, so the cache is
    #: per paper by construction — and three reads of a file a reviewer may be editing mid-run
    #: could hand two stages different answers, which is the one thing `apply_map_answers` being
    #: pure and idempotent cannot protect against.
    answers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

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
        group_a_synonyms=tuple(protocol.group_a.synonyms or ()),
        group_b_synonyms=tuple(protocol.group_b.synonyms or ()),
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


def _answered_map(ctx: "RunContext", paper: PaperRecord, study: StudyMap) -> StudyMap:
    """This paper's map with the reviewer's answers to its open questions applied (C4/C6/C7).

    `map_answers` reads the review log; `apply_map_answers` applies only the records that answer a
    question THIS map actually asked. Neither is caught: an unreadable review log must stop the
    run, because "nobody has answered anything" and "the answers could not be read" block the same
    cells and mean opposite things — the first is a question waiting for a person, the second is a
    person's answer being silently discarded.

    Called by BOTH stages that read `extraction_blocks`, because both must see the same map: a
    person who answers a `which_measure` question between an extract and a `--resume` verify has
    unblocked the cell for the verify stage too, and `apply_map_answers` is pure and idempotent
    exactly so that asking twice is free and gives the same answer.
    """
    return apply_map_answers(study, _map_answers(ctx, paper))


def _map_answers(ctx: "RunContext", paper: PaperRecord) -> list[dict[str, Any]]:
    """This paper's answers, read from the review log once per paper and cached (review L8)."""
    cached = ctx.answers.get(paper.sha256)
    if cached is None:
        cached = map_answers(ctx.out_dir, paper.sha256)
        ctx.answers[paper.sha256] = cached
    return cached


#: map answers that can only be acted on by BUYING a reading. "Exclude it" needs no model call and
#: is applied by `overrides.apply_overrides_and_repool` at re-pool time; these two are decisions
#: about what to EXTRACT, so they stay pending until an extract stage has read the cells they
#: un-block — which is what `consumed_override_seqs` records (M8's contract).
_EXTRACTING_ANSWERS = ("include_dataset", "which_measure")


def _answer_cells(answer: Mapping[str, Any], study: StudyMap, keys: set[str],
                  blocked_cells: Mapping[tuple[str, str], str] = MappingProxyType({})
                  ) -> list[str]:
    """The `<dataset>/<outcome>` cells one answer asks the extract stage to read, if any.

    Cells another OPEN map question still blocks are not among them. "Include this dataset" is
    acted on when every cell it un-blocks has been read, and a sibling cell held by an unanswered
    `which_measure` is one no resume may buy — so counting it left the inclusion "pending re-run"
    for ever, and the reviewer was told to run `--resume` again after every resume (whole-diff L4).
    """
    if answer.get("kind") not in _EXTRACTING_ANSWERS:
        return []
    if answer.get("kind") == "include_dataset" and answer.get("decision") != "include":
        return []
    dataset_id = str(answer.get("dataset_id") or "")
    outcome_key = str(answer.get("outcome_key") or "")
    return [f"{dataset.dataset_id}/{sources.outcome_key}"
            for dataset in study.datasets if dataset.dataset_id == dataset_id
            for sources in dataset.outcomes
            if sources.outcome_key in keys
            and (not outcome_key or sources.outcome_key == outcome_key)
            and (dataset.dataset_id, sources.outcome_key) not in blocked_cells]


def _consumed_seqs(ctx: "RunContext", paper: PaperRecord, study: StudyMap, keys: set[str],
                   extracted: Sequence[str], already: Sequence[Any] = (),
                   blocked_cells: Mapping[tuple[str, str], str] = MappingProxyType({})
                   ) -> list[int]:
    """The `seq` of every answer this stage has now acted on, unioned with what is on record.

    An answer is acted on when every cell it asks for has been extracted — not merely when the
    stage that could act on it ran. A run that dies on its budget before reaching the dataset a
    reviewer just included has not consumed that answer, and telling the reviewer it did would
    retire a decision nothing bought. Cumulative across resumes, per the consumer's contract:
    the union is written, never this resume's seqs alone.
    """
    done = set(extracted)
    seqs = {int(seq) for seq in already if isinstance(seq, int)}
    for answer in _map_answers(ctx, paper):
        seq = answer.get("seq")
        cells = _answer_cells(answer, study, keys, blocked_cells)
        if isinstance(seq, int) and cells and done.issuperset(cells):
            seqs.add(seq)
    return sorted(seqs)


def _has_printed_source(sources: Sequence[Source]) -> bool:
    """Is there anything on this cell for a text or statistic reader to read?"""
    return any(s.kind in _PRINTED_KINDS and not s.figure_id for s in sources)


def _figure(paper: PaperRecord, figure_id: str) -> FigureRegion | None:
    return next((f for f in paper.figures if f.id == figure_id), None)


def _cell_caption(paper: PaperRecord, sources: OutcomeSources) -> str:
    """The caption of the figure this cell is read off — "" unless there is exactly one.

    A cell whose readings come from two different figures has two captions, and a panel letter
    means something different in each: "panel B" of Fig. 3 and "panel B" of Fig. 4 are unrelated
    claims. The check that reads captions is only entitled to speak when there is one figure to
    speak about, so two captions (or none) means it does not run. `verify.panels` fails closed
    for the same reason at every other step.
    """
    captions = {" ".join((figure.caption or "").split())
                for source in readable_sources(sources.sources) if source.figure_id
                for figure in [_figure(paper, source.figure_id)]
                if figure is not None and (figure.caption or "").strip()}
    return captions.pop() if len(captions) == 1 else ""


def _page_texts(paper: PaperRecord) -> list[str]:
    """Every page of the ingested paper as text, page 1 first — the corpus D4-lite reads.

    Free (the ingest stage already wrote the files) and forgiving: a record whose text files are
    not on this machine yields empty pages rather than failing the verify stage, because a check
    that cannot read the paper has nothing to say and that is not an error in the run.
    """
    out: list[str] = []
    for page in paper.pages:
        try:
            out.append(paper.page_text(page.number))
        except OSError:
            out.append("")
    return out


def _analysed_n_flags(pages: Sequence[str], cell: Sequence[Candidate],
                      vocabulary: Mapping[str, Sequence[str]]) -> list[CheckFlag]:
    """D4-lite, over one cell's candidates: is a group's n the count before its exclusions?

    ONE flag per (group, size), carrying every candidate that was scored on it. Raised per
    candidate it would say the same thing five times about one arm of one figure, and
    `confidence` prices a repeated code once — so the repetition would buy nothing and cost a
    reviewer the readable version of the finding.
    """
    out: list[CheckFlag] = []
    seen: dict[tuple[str, int], CheckFlag] = {}
    for cand in cell:
        if cand.group not in ("A", "B") or cand.n is None:
            continue
        key = (str(cand.group), int(cand.n))
        found = seen.get(key)
        if found is not None:
            found.candidate_ids = sorted({*found.candidate_ids, cand.candidate_id})
            continue
        flag = n_before_exclusions(pages, cand, vocabulary.get(str(cand.group), ()))
        if flag is None:
            continue
        seen[key] = flag
        out.append(flag)
    return out


def _group_vocabulary(dataset: DatasetSpec, protocol: Protocol) -> dict[str, list[str]]:
    """Every word the review has for each arm — the dataset's own label first, then the review's.

    The same vocabulary the digitiser matches x categories against, for the same reason: a paper
    labels its panels in its own words, and a check that only knows the protocol's one label for
    a group cannot read a caption that abbreviates it.
    """
    return {
        "A": [name for name in (dataset.group_a.label, protocol.group_a.label,
                                *(protocol.group_a.synonyms or ())) if str(name or "").strip()],
        "B": [name for name in (dataset.group_b.label, protocol.group_b.label,
                                *(protocol.group_b.synonyms or ())) if str(name or "").strip()],
    }


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
                {"study": study.model_dump(mode="json"), "pdf_file_id": file_id,
                 # M8's contract: `overrides.consumed_seqs` reads this key from `map.json` AND
                 # `extract.json` and unions them. The map stage buys no reading, so it consumes
                 # nothing: `apply_map_answers` is applied by the two stages that read
                 # `extraction_blocks`, and an `include_dataset: include` answer is not acted on
                 # until a reader has been bought for the cells it un-blocks. Claiming it here
                 # would tell a reviewer their decision had been applied while nothing had been
                 # extracted for it — the same false statement, one stage earlier, that H2 is
                 # about. The key is written empty rather than omitted so the record says which.
                 "consumed_override_seqs": []})
    status.stages["map"] = "done"
    return study, file_id


def _cells_still_unread(study: StudyMap, keys: set[str], blocked_datasets: Mapping[str, str],
                        blocked_cells: Mapping[tuple[str, str], str], extracted: Sequence[str],
                        candidates: Sequence[Candidate]) -> set[str]:
    """Cells this map now asks for that a finished extract stage has neither read nor recorded.

    The rule the "answers on resume" ruling needed and did not have (review H2): **the extract
    stage is not done while the map un-blocks a cell it has no candidates for.** Written against
    the map rather than against the answers, so anything that un-blocks a cell — an answered
    question, an edited protocol, a re-mapped paper — re-enters the stage for that cell and only
    that cell.

    `cells_extracted` is what the stage itself recorded reading, so a cell that WAS read and came
    back empty is never re-bought; the candidate check is the fallback for a stage file written
    before that key existed.
    """
    have = {(c.dataset_id, c.outcome_key) for c in candidates}
    read = set(extracted)
    return {f"{dataset.dataset_id}/{sources.outcome_key}"
            for dataset in study.datasets if dataset.dataset_id not in blocked_datasets
            for sources in dataset.outcomes
            if sources.outcome_key in keys
            and (dataset.dataset_id, sources.outcome_key) not in blocked_cells
            and f"{dataset.dataset_id}/{sources.outcome_key}" not in read
            and (dataset.dataset_id, sources.outcome_key) not in have}


def _extract(ctx: RunContext, paper: PaperRecord, study: StudyMap,
             status: PaperStatus) -> list[Candidate]:
    keys = ctx.outcome_keys()
    figures_dir = paper_dir(ctx.out_dir, paper.sha256) / "figures"
    # C7/C6: a question the MAP could not settle is settled before extraction, not after it. A
    # dataset only one mapping agent proposed, or an outcome the map gave two measures, buys no
    # reader at all — Bock's tracking-only control sample was extracted, digitised, verified and
    # signed at d = -2.9610 for >= $0.745 while the objection to including it sat in a warning.
    #
    # Read BEFORE the resume short-circuit, because `--resume` is the only path the answers exist
    # for: extraction was never bought for a blocked cell, so the answer that un-blocks it can
    # only be acted on by a later run. Reading them after the short-circuit made the whole
    # mechanism inert — the tool told the reviewer to re-run with `--resume`, the reviewer did,
    # and got the identical message back, for ever (review H2).
    study = _answered_map(ctx, paper, study)
    blocked_datasets, blocked_cells = extraction_blocks(study)
    candidates: list[Candidate] = []
    done: list[str] = []
    exhausted: list[str] = []
    stopped = ""
    consumed: list[Any] = []
    only: set[str] | None = None

    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "extract"):
        payload = read_stage(ctx.out_dir, paper.sha256, "extract")
        candidates = [Candidate.model_validate(c) for c in payload["candidates"]]
        done = [str(cell) for cell in payload.get("cells_extracted") or []]
        consumed = list(payload.get("consumed_override_seqs") or [])
        only = _cells_still_unread(study, keys, blocked_datasets, blocked_cells, done, candidates)
        if not only:
            # nothing new to read. The stage file is still rewritten when an answer has finished
            # being acted on, so a decision whose cells were already extracted stops being pending
            # instead of waiting for a reading nobody owes it.
            seqs = _consumed_seqs(ctx, paper, study, keys, done, consumed, blocked_cells)
            if seqs != sorted(int(s) for s in consumed if isinstance(s, int)):
                write_stage(ctx.out_dir, paper.sha256, "extract", {**payload,
                                                                   "consumed_override_seqs": seqs})
            status.stages["extract"] = "skipped"
            return candidates
        status.warnings.append(
            f"{len(only)} cell(s) this map asks for had never been extracted "
            f"({', '.join(sorted(only))}) — the extract stage was re-entered for those cells "
            f"only; every other cell keeps the reading an earlier run paid for")

    def save(complete: bool) -> None:
        # written after EVERY cell, not once at the end: Buch 2003 spent $14.37 against a $14 cap
        # and the unwind threw away every candidate the stage had already paid for (F7). Until the
        # stage finishes the file says `"complete": false`, so `--resume` re-enters it rather than
        # reading a salvaged half-paper as a finished one.
        write_stage(ctx.out_dir, paper.sha256, "extract",
                    {"candidates": [c.model_dump(mode="json") for c in candidates],
                     "complete": complete, "cells_extracted": list(done),
                     "cells_budget_exhausted": list(exhausted),
                     "budget_note": stopped,
                     # M8: which of the reviewer's answers a stage has acted on. Without it an
                     # answer that needs a model call is pending for ever and the review page can
                     # never say which decisions are still outstanding.
                     "consumed_override_seqs": _consumed_seqs(ctx, paper, study, keys, done,
                                                              consumed, blocked_cells)})

    for dataset in study.datasets:
        if dataset.dataset_id in blocked_datasets:
            status.warnings.append(
                f"{dataset.dataset_id}: not extracted — {blocked_datasets[dataset.dataset_id]}")
            continue
        for sources in dataset.outcomes:
            if sources.outcome_key not in keys:
                status.warnings.append(
                    f"{dataset.dataset_id}: the mapper reported outcome "
                    f"{sources.outcome_key!r}, which is not in the protocol — skipped")
                continue
            if (dataset.dataset_id, sources.outcome_key) in blocked_cells:
                status.warnings.append(
                    f"{dataset.dataset_id}/{sources.outcome_key}: not extracted — "
                    f"{blocked_cells[(dataset.dataset_id, sources.outcome_key)]}")
                continue
            cell = f"{dataset.dataset_id}/{sources.outcome_key}"
            if only is not None and cell not in only:
                continue                      # already read in an earlier run; not re-bought here
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
    # only the sources the map says carry the VALUE are read for it. A baseline plotted beside
    # the outcome, or a location that only defines the window, stays on the record for a reader
    # and is named here: the aligned-cursor curves of Cressman's Fig. 3a were digitised as late
    # adaptation (3.9° beside the misaligned curves' 31.4°) before the map could say which was
    # which.
    # WHICH locations may be read is the map's rule, decided in one place so the pipeline and the
    # map cannot drift apart: what the location IS for this outcome (`role`) and WHOSE numbers are
    # at it (`sample`). A second copy of that test here is how the two came to disagree before.
    readable = readable_sources(sources.sources)
    for skipped in sources.sources:
        why = source_unreadable_reason(skipped)
        if why:
            status.warnings.append(
                f"{dataset.dataset_id}/{key}: {skipped.locator[:80]!r} {why} — kept for the "
                f"record, not read for the value")
    # the two heterogeneous text readings the vote needs (different model AND different prompt) —
    # bought only when there is something printed to read. A cell whose only source is a figure
    # used to buy three text calls and get three `not_on_these_pages` answers back (critique
    # miss 9); the figure routes are what carry such a cell, and they are unaffected.
    if _has_printed_source(readable):
        out.extend(extract_group_stats(ctx.client, paper, ctx.protocol, dataset, key,
                                       readable, variant="table_first",
                                       model=ctx.models["primary"]))
        out.extend(extract_group_stats(ctx.client, paper, ctx.protocol, dataset, key,
                                       readable, variant="narrative_first",
                                       model=ctx.models["secondary"]))
        out.extend(extract_test_statistics(ctx.client, paper, ctx.protocol, dataset, key,
                                           readable, model=ctx.models["primary"]))
    else:
        status.warnings.append(
            f"{dataset.dataset_id}/{key}: every source the mapper found for this outcome is a "
            f"figure, so the text and statistic readers were not bought")
    for source in readable:
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


#: the helpers below live in `canopy.pipeline.rows` now, because the REVIEW layer's rebuild has
#: to make the same row this stage does (whole-diff H1/H2) — `vote_candidates` joined them for
#: D1, whose fallback must offer the resolver the readings the vote weighed and not the
#: digitiser's raw samples. They keep their names here so a caller that knew where they were
#: still finds them.
_cell_candidates = cell_candidates


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
    """Re-extract ONE source a verifier named, when the mapper already had it and nobody read it.

    Matched against `readable_sources` and not the whole list: a verifier that names the baseline
    plotted beside the outcome has named a real location, and re-opening on it would buy a
    reading of the wrong quantity — which is the one thing `readable_sources` exists to stop.
    """
    readable = readable_sources(sources.sources)
    already = _already_read(cell, readable)
    for verdict in verdicts:
        named = (verdict.better_source or "").strip()
        if not named:
            continue
        source = match_named_source(named, readable)
        if source is None:
            kept = match_named_source(named, sources.sources)
            why = (f"the map keeps that location for the record and not for this outcome's value "
                   f"({source_unreadable_reason(kept)})" if kept is not None else
                   f"which is not in the mapper's list for this outcome — acting on it would "
                   f"mean acting on a location a model invented")
            status.warnings.append(
                f"{dataset.dataset_id}/{sources.outcome_key}: a verifier named {named!r} as a "
                f"better source — not re-opened, because {why}")
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


def _recheck_orientation(dataset: DatasetSpec, verdict: OrientationVerdict | None,
                         votes: dict[str, VoteResult],
                         flags: Sequence[CheckFlag]) -> OrientationVerdict | None:
    """C3 row 1, run where the numbers it needs finally exist.

    The discard filter compares a reader's own `direction_stated_in_text` with the RESOLVED raw
    group means. Orientation is decided before any cell of the measure is resolved, so at that
    point there are no means and the filter is inert — which is why it never fired on a real run.
    Re-combining the SAME ballots here costs nothing (no model call: `combine_orientation` is
    pure) and is the first moment the comparison is possible.

    Two guards, both load-bearing:

    * **only against the dataset the readers were asked about.** The verdict is a property of the
      measure and is reused for every dataset carrying it; another dataset's means are a different
      comparison, and checking a reader against them would discard whoever read the paper right.
    * **`third_read` travels with the verdict**, so a measure a bought third read settled by
      majority is recombined the same way rather than silently falling back to a question.

    Consequently the check runs ONCE per measure. Its outcome is a property of the measure — a
    reader whose words contradict the numbers it was reading is not trustworthy about that measure
    anywhere — so `_verify` writes the checked verdict back and every later dataset of the same
    measure inherits it and takes the early return above.

    What is handed to the filter, exactly:

    * the RAW group means, in this dataset's own A/B assignment, as the vote resolved them.
      Nothing here orientates them: `higher_is_better` is applied in `resolve_effect`, far later,
      and a sign applied before the comparison would be the comparison arguing with itself.
    * `None` — never a default, never the other group's number — for a group that did not resolve.
      An absent mean disables the filter, which is the safe direction: its only power is to remove
      a witness.
    * `open_flags` from the SAME candidate set the means came from, so `series_marker_mismatch` and
      its family suppress a discard against means that may be the other group's.

    The filter can only remove a witness or abstain, so the worst this can do is turn a decided
    measure into a question. It can never invent a direction, and "no discard" is exactly today.
    """
    if verdict is None or verdict.dataset_id != dataset.dataset_id:
        return verdict
    vote_a, vote_b = votes.get("A"), votes.get("B")
    checked = combine_orientation(
        verdict.runs, verdict.outcome_key, verdict.measure_name,
        mean_a=vote_a.mean if vote_a is not None else None,
        mean_b=vote_b.mean if vote_b is not None else None,
        open_flags=[f.code for f in flags], third_read=verdict.third_read,
        dataset_id=verdict.dataset_id)
    # ALWAYS the checked verdict, never the one it replaces. An equality shortcut on
    # `(hib, needs_human, agreed)` looks safe — those are the fields anything downstream computes
    # with — but the verdict is also a RECORD, and the record is what a reviewer reads. Under the
    # shortcut 11 of the 13 replayed cells kept `means check: no_means` (written when orientation
    # was decided and there were no means yet) on a cell where the check had since actually run,
    # or had been suspended because the series identity was disputed. All three pooled cells were
    # among them. A verdict that says the check could not run, on a cell where it did, is a false
    # statement about the evidence — and it is the statement someone would rely on to decide the
    # check had never been wired in at all.
    return checked


#: error codes that must NOT, on their own, buy an adjudicator call. Two rulings, one shape.
#:
#: The adjudicator settles VALUE disputes — which of two readings of a number is right — and it
#: is bought with the whole paper in context. A cell whose only error is something it is
#: forbidden to decide, or something no model can supply, is already waiting for the person who
#: will answer it, and the call cannot change the answer.
#:
#: * `ORIENTATION_FLAGS` (controller ruling R1): by ruling the adjudicator may not decide a
#:   direction, so C3's contradiction and its siblings buy nothing.
#: * `DF_PROVENANCE_FLAGS` (review M5, widened by the fix round's controller ruling): "this t is
#:   printed with no degrees of freedom" is a fact about the paper. The flags are computed BEFORE
#:   the call and passed unchanged into `resolve_cell`, so the cell is `needs_human` whatever the
#:   ruling says. All three members are that one fact under three codes.
#:
#: Both come from `canopy.verify.checks` rather than from a list of strings here, so a code added
#: to either family is excluded here without anyone remembering to. Any OTHER error, a vote
#: disagreement or a refutation adjudicates exactly as before.
NO_ADJUDICATION_FLAGS: frozenset[str] = frozenset(ORIENTATION_FLAGS) | frozenset(
    DF_PROVENANCE_FLAGS)


def _tiebreak(ctx: RunContext, paper: PaperRecord, dataset: DatasetSpec,
              sources: OutcomeSources, orientation: OrientationVerdict | None,
              votes: Mapping[str, VoteResult], flags: Sequence[CheckFlag],
              tiebroken: set[tuple[str, str]] | None, status: PaperStatus,
              file_id: str = "") -> OrientationVerdict | None:
    """C2: at most ONE bought ballot per (paper, outcome, measure), or `None` for "not bought".

    Every condition is a reason not to spend money, and each of them is a rule rather than a
    heuristic:

    * `--no-tiebreak`, or a direction that is already settled — there is nothing to buy;
    * this measure already has a third read (`tiebroken`, and `third_read` on the verdict so that
      a resumed run does not buy a second one) — the ruling is "at most once per measure", and a
      paper with four datasets carrying one measure would otherwise buy four;
    * one of the two raw means did not resolve — the ballot's whole advantage over the first two
      readers is that it can check a claim about which group came out higher against the numbers,
      and with a number missing it is simply a third reader of the same prompt;
    * the ballot's model family is one an earlier reader already used — a reader that fails the
      same way as one that has already answered adds a vote, not evidence.

    The combination is `combine_orientation(..., third_read=True)`, which is the ordinary majority
    branch: the ballot can remove a question and can never settle one alone. A `BudgetExceeded` is
    a note on the verdict and not an exception — the run's money is the run's business, and a
    ballot nobody can pay for leaves exactly the question the cell already had.
    """
    measure = (sources.outcome_key, sources.measure_name or "")
    if not ctx.tiebreak or orientation is None or not orientation.needs_human:
        return None
    if orientation.third_read or (tiebroken is not None and measure in tiebroken):
        return None
    vote_a, vote_b = votes.get("A"), votes.get("B")
    mean_a = vote_a.mean if vote_a is not None else None
    mean_b = vote_b.mean if vote_b is not None else None
    if mean_a is None or mean_b is None:
        return None
    if tiebroken is not None:
        tiebroken.add(measure)                  # bought or refused, this measure is not asked twice
    if model_family(TIEBREAK_MODEL) in {model_family(run.model) for run in orientation.runs}:
        return orientation.model_copy(update={"notes": "; ".join(filter(None, [
            orientation.notes,
            f"no tiebreak ballot was bought: {TIEBREAK_MODEL} is the model family of a reader "
            f"that has already answered on this measure"]))})
    try:
        third = tiebreak_ballot(
            ctx.client, paper, dataset, sources, protocol=ctx.protocol,
            outcome=ctx.protocol.outcome(sources.outcome_key),
            pdf_file_id=file_id or None, mean_a=mean_a, mean_b=mean_b,
            group_a_label=dataset.group_a.label, group_b_label=dataset.group_b.label,
            n_a=dataset.group_a.n, n_b=dataset.group_b.n,
            unit=sources.units or "", prior_runs=orientation.runs)
    except (BudgetExceeded, PaperBudgetExceeded, LLMError) as exc:
        status.warnings.append(
            f"{dataset.dataset_id}/{sources.outcome_key}: no tiebreak ballot was bought for "
            f"{sources.measure_name or sources.outcome_key!r} ({type(exc).__name__}: "
            f"{str(exc)[:160]}) — the direction stays a question")
        return orientation.model_copy(update={"notes": "; ".join(filter(None, [
            orientation.notes, f"no tiebreak ballot was bought: {str(exc)[:200]}"]))})
    return combine_orientation(
        [*orientation.runs, third], sources.outcome_key, sources.measure_name or "",
        mean_a=mean_a, mean_b=mean_b, open_flags=[f.code for f in flags], third_read=True,
        dataset_id=dataset.dataset_id)


def buys_adjudication(flags: Sequence[CheckFlag]) -> bool:
    """Does this cell carry an `error` an adjudicator could actually settle?"""
    return any(f.severity == "error" and f.code not in NO_ADJUDICATION_FLAGS for f in flags)


def _verify_cell(ctx: RunContext, paper: PaperRecord, dataset: DatasetSpec,
                 sources: OutcomeSources, candidates: Sequence[Candidate],
                 all_candidates: Sequence[Candidate], file_id: str,
                 orientation: OrientationVerdict | None,
                 status: PaperStatus,
                 tiebroken: set[tuple[str, str]] | None = None) -> _CellVerification:
    key = sources.outcome_key
    outcome_def = ctx.protocol.outcome(key)
    # D2: the caption says which panel is whose, and a reading taken off another group's panel is
    # that group's number wearing this cell's name. It is settled BEFORE anything else looks at
    # the cell — the vote, the checks and the adversarial verifier all see the filtered list —
    # because a reading nobody may weigh is not a reading a verifier should be paid to refute.
    candidates, panel_flags = apply_panel_check(
        candidates, _cell_caption(paper, sources), _group_vocabulary(dataset, ctx.protocol))
    extra_flags: list[CheckFlag] = list(panel_flags)
    # the digitiser's per-route samples stay in the stage file; only its ensemble votes
    cell = vote_candidates(candidates)
    others = vote_candidates([c for c in all_candidates
                              if c.dataset_id != dataset.dataset_id or c.outcome_key != key])
    # …and of the locations the mapper found, only the best-scoring ones do (P5). Extraction has
    # already read them all; this decides which readings the VOTE weighs against each other, so a
    # sentence with no n and no dispersion cannot outvote a figure with SE bars and n printed.
    #
    # Through `readable_sources`, like every other reading. The rank was built over the mapper's
    # WHOLE list, and `keep_for_vote` holds a candidate back when a HIGHER-SCORING source carries
    # a dispersion it lacks — so a location the map says nobody may read (C6's `alternate`
    # operationalization with its own error bars, a `pooled` sample) scored top and evicted the
    # value the paper printed from the vote. A location no reading may come from cannot be the
    # reason another reading is not weighed (whole-diff M1).
    ranked = rank_sources(readable_sources(sources.sources), sources)
    out_rank = [row.to_dict() for row in ranked]
    cell, held_back = keep_for_vote(cell, ranked)
    n_a, n_b = dataset.group_a.n, dataset.group_b.n
    out = _CellVerification(orientation=orientation, source_rank=out_rank,
                            held_back=list(held_back))
    # D4-lite, before the checks that weigh it: every candidate carries a group size (a digitised
    # one carries the mapper's), and the paper's own words say whether that size is the one it
    # recruited or the one it analysed. Free, deterministic, and it repairs nothing — the flag
    # caps the cell and the reviewer's `group_n` answer supplies the analysed sizes.
    extra_flags.extend(_analysed_n_flags(_page_texts(paper), cell,
                                         _group_vocabulary(dataset, ctx.protocol)))

    # `run_checks`'s participant-total parameter wants the total the PAPER states for this
    # dataset. This call used to pass `n_a + n_b`, which is not that number — it is the two
    # analysed group sizes, so every check that compared the two was comparing a value with
    # itself (review M2: C9's "a stated exclusion explains the shortfall" branch and
    # `n_sum_mismatch` were both dead in every real run). Nothing on a `StudyMap` records a
    # stated total yet, so nothing is passed: an argument that is not the thing the parameter
    # names is worse than a missing one.
    flags = [*run_checks(dataset, key, cell, other_candidates=others, orientation=orientation),
             *extra_flags]
    votes = vote_groups(cell, unit_hint=sources.units or outcome_def.units_hint)

    # amendment G: the two text extractors disagreed, so buy a third cheap reading — the secondary
    # model on the prompt variant it has not seen — and let it move that route's median. Through
    # `readable_sources`, like every other reading: this call passed the mapper's WHOLE list, so
    # the one reader bought to settle a disagreement was the only one allowed to read a baseline
    # the other two were kept away from (mapper re-review, amendment G).
    if any(v.needs_third_candidate for v in votes.values()):
        third = extract_group_stats(ctx.client, paper, ctx.protocol, dataset, key,
                                    readable_sources(sources.sources),
                                    variant="table_first", model=ctx.models["secondary"])
        if third:
            out.extra_candidates.extend(third)
            cell = [*cell, *third]
            flags = [*run_checks(dataset, key, cell, other_candidates=others,
                                 orientation=orientation), *extra_flags]
            votes = vote_groups(cell, unit_hint=sources.units or outcome_def.units_hint)

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
            try:
                verdict = verify_candidate(
                    ctx.client, paper, winner,
                    model=verifier_model_for(winner, preferred=preferred),
                    protocol=ctx.protocol, dataset=dataset, outcome=outcome_def,
                    pdf_file_id=file_id or None, reopen=reopen)
            except (BudgetExceeded, PaperBudgetExceeded):
                raise                                   # money is the run's business, not the cell's
            except TruncatedOutput as exc:
                # one reader wrote past its output limit twice; that is a verdict the cell does
                # not have, not a reason for the paper to have no verdicts at all. Heuer &
                # Hegele lost its whole verify stage — 47 candidates, no rows — to one such call.
                # C8: an infrastructure failure is an ABSENCE, not evidence. Recorded as
                # `not_run`, it is priced at zero; recorded as `ambiguous` it cost -0.05, so a
                # cell whose verifier CRASHED scored below one that was never verified at all.
                verdict = VerifierVerdict(
                    candidate_id=winner.candidate_id, verdict="not_run",
                    reason=f"the verifier's answer was cut off at its output limit twice and "
                           f"could not be read ({exc}); no verdict was produced, so this "
                           f"candidate is unverified — which is not evidence against it")
                status.warnings.append(
                    f"{dataset.dataset_id}/{key}: verifier truncated at its output limit on "
                    f"{winner.candidate_id} — no verdict was produced, so it is recorded as "
                    f"not_run and the cell is unverified rather than doubted")
            except LLMError as exc:
                verdict = VerifierVerdict(
                    candidate_id=winner.candidate_id, verdict="not_run",
                    reason=f"the verifier could not be run ({type(exc).__name__}: "
                           f"{str(exc)[:160]}); no verdict was produced, so this candidate is "
                           f"unverified — which is not evidence against it")
                status.warnings.append(
                    f"{dataset.dataset_id}/{key}: verifier failed on {winner.candidate_id} "
                    f"({type(exc).__name__}) — no verdict was produced, so it is recorded as "
                    f"not_run and the cell is unverified rather than doubted")
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
        extra_flags.append(CheckFlag(
            code="reopened_on_better_source", severity=CHECK_SEVERITY["reopened_on_better_source"],
            message=(f"a verifier named {named!r} as a better source for this outcome and "
                     f"extraction was re-opened on it (one hop, once)"),
            candidate_ids=sorted(c.candidate_id for c in vote_candidates(extra))))
        flags = [*run_checks(dataset, key, cell, other_candidates=others,
                             orientation=orientation), *extra_flags]
        votes = vote_groups(cell, unit_hint=sources.units or outcome_def.units_hint)

    # C3 row 1: the vote and the checks are final, so the resolved means the discard filter needs
    # finally exist. Free (no model call), and it can only take a witness away.
    checked = _recheck_orientation(dataset, orientation, votes, flags)
    if checked is not orientation:
        orientation = out.orientation = checked
        flags = [*run_checks(dataset, key, cell, other_candidates=others,
                             orientation=orientation), *extra_flags]

    # C2, immediately after the free check and nowhere else. Here because this is the first and
    # only moment all three of its inputs exist together: the raw means (the vote has just
    # resolved them), the two earlier ballots, and the fact that nothing free settled the
    # direction. And here rather than in `_verify` after this function returns, because the two
    # cells below are signed from `orientation` — a ballot bought after they were written would
    # settle the MEASURE for the papers' later datasets and leave this one's own rows unsigned,
    # which is the cell the money was spent on.
    bought = _tiebreak(ctx, paper, dataset, sources, orientation, votes, flags, tiebroken,
                       status, file_id)
    if bought is not None:
        orientation = out.orientation = bought
        flags = [*run_checks(dataset, key, cell, other_candidates=others,
                             orientation=orientation), *extra_flags]

    # …and D2's first half, recorded once the vote is final: the readings that agreed came from
    # different places in one figure, so their agreement is a coincidence of the figure tolerance
    # rather than corroboration, and the vote refused to average them.
    for group in ("A", "B"):
        result = votes.get(group)
        if result is None or result.method != LOCATOR_CONFLICT:
            continue
        flags = [*flags, CheckFlag(
            code="locator_reads_conflict", severity=CHECK_SEVERITY["locator_reads_conflict"],
            message=next((note for note in result.notes
                          if note.startswith(LOCATOR_CONFLICT_NOTE)), LOCATOR_CONFLICT_NOTE),
            candidate_ids=sorted(set(result.disagreeing_ids)))]

    disagreed = any(v.agreement == "disagree" for v in votes.values())
    if disagreed or refuted or buys_adjudication(flags):
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


def _orientation_state(verdict: OrientationVerdict | None) -> tuple:
    """What a re-check could have CHANGED about a direction — the fields a reader is warned about.

    `notes` is included because that is where the check records a discard and its own state
    (`ran` / `no_means` / `disputed`); `agreed` and `higher_is_better` are the decision itself.
    """
    if verdict is None:
        return ()
    return (verdict.higher_is_better, verdict.needs_human, verdict.agreed, verdict.notes)


def _verify(ctx: RunContext, paper: PaperRecord, study: StudyMap, candidates: list[Candidate],
            file_id: str, status: PaperStatus) -> list[Verdict]:
    keys = ctx.outcome_keys()
    study = _answered_map(ctx, paper, study)
    # C6/C7 again, at the stage that BUYS the direction of a measure. The extraction gate already
    # skips a blocked cell, so it produces no candidate and no verifier or digitiser call follows
    # — but the two orientation reads are per (outcome, measure), not per candidate, and were
    # spent anyway on a dataset nobody has agreed to include.
    blocked_datasets, blocked_cells = extraction_blocks(study)
    verdicts: list[Verdict] = []
    extra: list[Candidate] = []
    orientations: dict[tuple[str, str], OrientationVerdict] = {}
    adjudications: list[dict[str, Any]] = []
    source_ranks: dict[str, Any] = {}
    reopens = 0
    #: C2: the measures a tiebreak ballot has already been offered for, bought or refused. Owned
    #: here because the ruling is per (paper, outcome, measure) and a cell cannot see its siblings.
    tiebroken: set[tuple[str, str]] = set()
    only: set[tuple[str, str]] | None = None

    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "verify"):
        payload = read_stage(ctx.out_dir, paper.sha256, "verify")
        verdicts = [Verdict.model_validate(v) for v in payload["verdicts"]]
        extra = [Candidate.model_validate(c) for c in payload.get("extra_candidates", [])]
        # the same rule as the extract stage's, one stage on: this stage is not done while a cell
        # that now HAS candidates has no verdict. Without it an answer acted on by the resumed
        # extract stage produced candidates and no row, which is the same dead end one layer up.
        only = {(c.dataset_id, c.outcome_key) for c in [*candidates, *extra]
                if c.dataset_id not in blocked_datasets and c.outcome_key in keys
                and (c.dataset_id, c.outcome_key) not in blocked_cells} - {
                    (v.dataset_id, v.outcome_key) for v in verdicts}
        if not only:
            status.stages["verify"] = "skipped"
            candidates.extend(extra)
            return verdicts
        status.warnings.append(
            f"{len(only)} newly extracted cell(s) had no verdict "
            f"({', '.join(f'{d}/{k}' for d, k in sorted(only))}) — the verify stage was "
            f"re-entered for those cells only")
        # the directions an earlier run already decided, in the shape the stage file stores them
        # (`"<outcome>|<measure>"`), so a measure that has a verdict is never re-bought
        orientations = {(name.split("|", 1)[0], name.split("|", 1)[1] if "|" in name else ""):
                        OrientationVerdict.model_validate(value)
                        for name, value in (payload.get("orientation") or {}).items()}
        adjudications = list(payload.get("adjudications") or [])
        source_ranks = dict(payload.get("source_rank") or {})
        reopens = int(payload.get("reopens") or 0)

    for dataset in study.datasets:
        if dataset.dataset_id in blocked_datasets:
            continue
        for sources in dataset.outcomes:
            if sources.outcome_key not in keys:
                continue
            if (dataset.dataset_id, sources.outcome_key) in blocked_cells:
                continue
            if only is not None and (dataset.dataset_id, sources.outcome_key) not in only:
                continue                      # verified in an earlier run; nothing is re-bought
            cell = _cell_candidates([*candidates, *extra], dataset.dataset_id,
                                    sources.outcome_key)
            if not cell:
                # nothing was extracted for this cell: no value to sign, nothing for a verifier
                # to check, and no direction is bought (the two orientation reads are per
                # (outcome, measure), so a dataset un-blocked but never extracted used to spend
                # two calls on empty verdicts — review H2). The cell is still WRITTEN, as two
                # value-less verdicts, so the review queue and the questions page can ask where
                # its value is: the first nine-paper run lost a paper's cells with "no verdict
                # worth writing", zero calls and no question — a hold nobody could see.
                status.warnings.append(
                    f"{dataset.dataset_id}/{sources.outcome_key}: no candidate was extracted for "
                    f"this cell, so no direction was bought; the cell is recorded as unresolved "
                    f"and asked about")
                n_a, n_b = dataset.group_a.n, dataset.group_b.n
                for group in ("A", "B"):
                    verdicts.append(resolve_cell(
                        dataset, sources.outcome_key, group, [], vote_result=None, verdicts=[],
                        flags=[], adjudication=None, orientation=None, other_candidates=[],
                        n_a=n_a, n_b=n_b))
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
            result = _verify_cell(ctx, paper, dataset, sources, cell, [*candidates, *extra],
                                  file_id, orientations[measure], status, tiebroken)
            if result.orientation is not None:
                # C3 row 1 may have discarded a reader once this cell's means existed. That is a
                # fact about the MEASURE, so every later dataset carrying it inherits the checked
                # verdict rather than the one the readers were first scored on, and the stage file
                # records what was actually used.
                if _orientation_state(result.orientation) != _orientation_state(
                        orientations[measure]):
                    # by STATE, not by identity: `_recheck_orientation` returns a fresh object
                    # whenever the dataset matches, so an identity test said "changed" about
                    # every measure of every paper, always — a warning that fires unconditionally
                    # is a warning a reader learns to skip (whole-diff L1).
                    status.warnings.append(
                        f"{dataset.dataset_id}/{sources.outcome_key}: the direction of "
                        f"{sources.measure_name or sources.outcome_key!r} was re-checked against "
                        f"the resolved means and changed — {result.orientation.notes}")
                orientations[measure] = result.orientation
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


#: the columns `human_review_queue.csv` carries, in order. `write_rows` passes
#: `extrasaction="ignore"` to `csv.DictWriter`, so a key `review_entry` publishes and this list
#: omits is dropped from the file in silence — which is how C11's three columns came to exist on
#: the dict, in the JSON and in the HTML table, and nowhere in the CSV a reviewer opens (M1).
REVIEW_QUEUE_COLUMNS: tuple[str, ...] = (
    "paper_id", "dataset_id", "outcome_key", "group", "confidence",
    #: C11: the score, how far it was from the line that decided the bucket, and which line —
    #: empty on a cell whose bucket the score did not decide (M3)
    "confidence_score", "confidence_margin", "nearest_boundary",
    "route", "reason", "impact_abs_delta_pooled", "candidates")

def cells_for_review(verdicts: Sequence[Verdict], held: Sequence[EffectSizeRecord],
                     exclusions: Sequence[Mapping[str, Any]] = ()) -> list[Verdict]:
    """Every cell that belongs in `human_review_queue`, by three rules — not one.

    The queue is built from CELLS and read by the review page and `questions_for_run`, so a cell
    missing from it is a question nobody is asked. Three things put one there, and the last two
    exist because a row is something that holds:

    1. the cell's own bucket is `needs_human`;
    2. the ROW is held by a rule that lives on the row and on neither cell — the conversion gate
       and C9's `|d|` screen both are, because both need the number the conversion produced. Such
       a row was refused, drawn hollow, and named nowhere a reviewer works. `not_convertible` is
       deliberately excluded: that row is a paper that did not report enough, it is recorded in
       `exclusions`, and the two findings are kept apart;
    3. …and for a row carrying a `ROW_REFUSAL_CODES` flag, BOTH cells, whether or not one of them
       has a finding of its own. Rule 2 subtracts the cells that are held in their own right,
       which is right for an ordinary cell finding — the healthy sibling has no question — and
       wrong here: neither cell can be released while the row's denominator is in doubt, so
       leaving the sibling out drops it the moment the first one is answered.

    Minus one: a cell a PERSON excluded is settled, not outstanding. The decision is on the record
    and in the exclusion table, and counting it as awaiting review counts it for ever.

    These are the rules `canopy.pipeline.overrides._rewrite` applies when it re-pools, and they
    live here so a fresh `canopy run` and a re-pool of it name the same cells: a queue that
    changes when a reviewer answers something unrelated is a queue nobody can work down.
    """
    cells_held = {(v.dataset_id, v.outcome_key) for v in verdicts if v.needs_human}
    row_held = {(r.dataset_id, r.outcome_key) for r in held
                if r.route != "not_convertible"} - cells_held
    row_held |= {(r.dataset_id, r.outcome_key) for r in held
                 if set(r.flags) & ROW_REFUSAL_CODES}
    gone_datasets = {str(e.get("dataset_id") or "") for e in exclusions
                     if e.get("decider") == HUMAN_DECIDER and not e.get("outcome_key")}
    gone_cells = {(str(e.get("dataset_id") or ""), str(e.get("outcome_key") or ""))
                  for e in exclusions
                  if e.get("decider") == HUMAN_DECIDER and e.get("outcome_key")}
    out: list[Verdict] = []
    for verdict in verdicts:
        key = (verdict.dataset_id, verdict.outcome_key)
        if not (verdict.needs_human or key in row_held):
            continue
        if verdict.dataset_id in gone_datasets or key in gone_cells:
            continue
        out.append(verdict)
    return out


_approximation_flags = approximation_flags


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


_statistic_values = statistic_values
_reported_values = reported_values


def _resolve(ctx: RunContext, paper: PaperRecord, study: StudyMap,
             candidates: Sequence[Candidate], verdicts: Sequence[Verdict],
             status: PaperStatus) -> list[EffectSizeRecord]:
    keys = ctx.outcome_keys()
    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "resolve"):
        payload = read_stage(ctx.out_dir, paper.sha256, "resolve")
        records = [EffectSizeRecord.model_validate(r) for r in payload["records"]]
        # the same rule again, at the last stage: not done while a cell that now has verdicts has
        # no record. This stage buys nothing — it is arithmetic over the verdicts — so it is
        # recomputed in full rather than merged, which cannot drift from a fresh run.
        missing = {(v.dataset_id, v.outcome_key) for v in verdicts} - {
            (r.dataset_id, r.outcome_key) for r in records}
        if not missing:
            status.stages["resolve"] = "skipped"
            return records
        status.warnings.append(
            f"{len(missing)} newly verified cell(s) had no resolved row "
            f"({', '.join(f'{d}/{k}' for d, k in sorted(missing))}) — the resolve stage was "
            f"re-run over every verdict this paper now has")

    by_cell = {(v.dataset_id, v.outcome_key, v.group): v for v in verdicts}
    cells: list[tuple[DatasetSpec, str, Verdict, Verdict]] = []
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
            cells.append((dataset, key, verdict_a, verdict_b))

    # the printed statistic, the printed effect size, the row's own policy flags and the
    # shared-control adjustment (Cochrane 16.5.4) — in `pipeline.rows`, because the review layer
    # rebuilds the same row after every human answer and must build the SAME row (whole-diff H1).
    prepared = prepare_rows(cells, candidates, ctx.settings,
                            cluster_of=lambda d: d.cluster_id or paper.sha256)

    records: list[EffectSizeRecord] = []
    for row in prepared:
        dataset, key, values = row.dataset, row.outcome_key, row.values
        # D1: `resolve_effect`, plus the one case precedence cannot reach on its own — a printed
        # value with no spread converts to nothing, and a same-locator candidate pair that does
        # convert may build the row instead (held, and stamped). `row.alternatives` is prepared by
        # the same function the review layer's rebuild calls, so both paths offer the same pairs.
        record = resolve_effect_with_fallback(dataset, ctx.protocol.outcome(key), values,
                                              row.alternatives, ctx.settings)
        record.orientation_source = row.orientation_source
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
                           models=ctx.models, resume=ctx.resume, tiebreak=ctx.tiebreak,
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
        # C7: a dataset the adjudicator rejected on a named protocol rule is never extracted, so
        # it reaches no resolver and no row — and until now it reached no table either. "Rejected
        # -> recorded in exclusions" is the half that makes the decision auditable: the rule it
        # was rejected under and the paper's own words it rests on, in the run's own record.
        #
        # Built from the ANSWERED map, because a person may exclude a dataset too (C4/C7's
        # `include_dataset` question, answered `exclude`). Such a row must not be attributed to the
        # adjudicator: the review log says who answered the question, and the record has to say
        # who decided, not merely that something did.
        # WHO decided, from the RECORD of the decision — the review log, which is where a
        # person's answer to the C7 question lives. The only test available before was whether the
        # cited rule was the "no rule named" placeholder, so every reviewer who CITED the protocol
        # was written down as the adjudicator: a person's judgement attributed to a model, and an
        # exclusion the table then listed twice (whole-diff L5 / re-review N2). Deliberately NOT a
        # new field on `DatasetSpec`: the study map is dumped verbatim into the map-adjudicator's
        # prompt, so a field added there changes that prompt and re-buys the call for every paper.
        by_hand = {str(answer.get("dataset_id") or "")
                   for answer in _map_answers(paper_ctx, paper)
                   if answer.get("kind") == "include_dataset"
                   and str(answer.get("decision") or "").strip().lower() == "exclude"}
        for dataset in _answered_map(paper_ctx, paper, study).datasets:
            if dataset.included:
                continue
            result.exclusions.append({
                "paper_id": group.sha256, "filename": status.filename,
                "dataset_id": dataset.dataset_id, "stage": "map",
                "reason": "map_adjudication:dataset_rule",
                "quote": dataset.exclusion_quote,
                # a person decided it if the LOG says so — or if the rule cited is the
                # placeholder only `apply_map_answers` writes, which is the one case the log is
                # not needed for and the only one the old test covered.
                "decider": (HUMAN_DECIDER
                            if dataset.dataset_id in by_hand
                            or dataset.exclusion_rule.strip() == HUMAN_EXCLUSION_RULE
                            else MAP_ADJUDICATOR),
                "detail": dataset.exclusion_rule})
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
        # the success path is a way of leaving with nothing too: every stage done, every dataset
        # resolved to no row (an outcome key outside the protocol, say). Say so.
        _gone(result, group, status, "no_usable_data:no_rows_resolved",
              "every stage finished and no dataset resolved to an effect size")
    except RunCancelled:
        status.status = "cancelled"
        status.error = "cancelled by the reviewer"
        _gone(result, group, status, "cancelled", status.error)
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
                 tiebreak: bool = True,
                 ingest_fn: Callable[[Path, Path], PaperRecord] | None = None) -> RunManifest:
    """Run the whole review and write `<out_dir>`; returns the manifest it saved.

    `resume=True` (the default) skips any stage whose file already exists, so re-running after a
    budget stop, a crash or a new paper costs only what is genuinely new.

    `tiebreak=False` turns C2's bought orientation ballot off: no third read is ever purchased
    and a direction the two readers and the free check could not settle stays a question.

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
                     tiebreak=tiebreak,
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
        for verdict in cells_for_review([v for v in verdicts if v.outcome_key == outcome.key],
                                        held, exclusions):
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
            REVIEW_QUEUE_COLUMNS, formats=("csv", "json")).items()})
        # …and the same cells as QUESTIONS: the picture the tool read, the answers it is choosing
        # between, and why it could not decide — what a reviewer actually wants to be shown
        try:
            from ..review.questions import questions_for_run, write_questions
            outputs.update({f"questions.{k}": v for k, v in write_questions(
                out, questions_for_run(out, queue=manifest.human_review_queue)).items()})
        except Exception as exc:                        # pragma: no cover - never fail a run on it
            manifest.warnings.append(f"questions could not be written: {type(exc).__name__}: {exc}")

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
    if result.records or any(e.get("stage") == "map" and not e.get("dataset_id")
                             for e in result.exclusions):
        return                                # it has rows, or the map stage already said why
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
    included_paper_ids = {r.paper_id for r in included}
    # an eligible paper that produced no row is a paper the review lost, whatever the reason —
    # excluded out loud at the map stage, budget, error, or every dataset resolving to nothing.
    # Counting `eligible is False` alone printed "0 were excluded" beside a paper that vanished.
    no_rows = [r for r in eligible if r.status.paper_id not in included_paper_ids]
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
        "papers_with_no_rows": len(no_rows),
        "datasets": datasets,
        "datasets_excluded": max(0, datasets - len({r.dataset_id for r in included})),
        "included_datasets": len({r.dataset_id for r in included}),
        "included_papers": len(included_paper_ids),
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
