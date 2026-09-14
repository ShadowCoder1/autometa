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

import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

from ..agents.adjudicator import adjudicate
from ..agents.extract_stats import extract_test_statistics
from ..agents.extract_text import extract_group_stats
from ..agents.mapper import (HUMAN_DECIDER, HUMAN_EXCLUSION_RULE, MAP_ADJUDICATOR,
                             apply_map_answers, c6_demoted,
                             extraction_blocks, map_answer_effects, map_answer_key,
                             map_study, readable_sources,
                             settled_measure, source_unreadable_reason, unreadable_cell)
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
from ..llm.context import reviewer_ruling_line, upload_pdf
from ..llm.costs import cache_stats, cache_summary_line, cost_by_stage
from ..llm.errors import BudgetExceeded, LLMError, TruncatedOutput
from ..models import (Adjudication, Candidate, CheckFlag, DatasetSpec, EffectSizeRecord,
                      OrientationVerdict, OutcomeSources, PaperStatus, Protocol, RunManifest,
                      SourceKind, Source, StatsSettings, StudyMap, UNREADABLE_SAMPLES,
                      Verdict, VerifierVerdict)
from ..protocol import load_protocol
from ..report import (exclusions_table, extraction_table, methods_figure, pool_rows,
                      prisma_flow, provenance_bundle, route_examples, write_html_report,
                      write_outcome_outputs, write_rows)
from ..stats.meta import MetaResult
from ..verify.checks import (CHECK_SEVERITY, DF_PROVENANCE_FLAGS, ORIENTATION_FLAGS,
                            n_before_exclusions, run_checks)
from ..verify.confidence import ROW_REFUSAL_CODES, resolve_cell
from ..verify.panels import apply_panel_check
from ..verify.vote import (LOCATOR_CONFLICT, LOCATOR_CONFLICT_NOTE, NEGLIGIBLE_SPLIT, VoteResult,
                           _verified_sd, model_family, vote_groups)
from .aggregate import AGGREGATED_FLAG, Aggregation, aggregate_one_row_per_paper
from .overrides import (HUMAN_OVERRIDE, OVERRIDES_FILE, HumanLanded, apply_overrides_and_repool,
                        eligibility_answers, human_landed_values, map_answers, read_overrides,
                        re_extract_answers, within_read_tolerance)
from .resolve import resolve_effect_with_fallback
from .rows import (DISPERSION_APPROXIMATED, approximation_flags, cell_candidates,
                   house_spread_type, prepare_rows,
                   reported_values, sample_key, statistic_values, vote_candidates)
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
    #: …and which of those answers actually CHANGED the map, per paper. An answer the map refuses —
    #: one naming a reading it does not carry, one that would leave both measures readable — is not
    #: a decision this run has acted on, and `_consumed_seqs` may not retire it (M8). Cached beside
    #: the answers themselves so the two are always read from the same log.
    effects: dict[str, dict[tuple[str, str, str], str]] = field(default_factory=dict)

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
                      protocol: Protocol, settings: StatsSettings,
                      reviewer_hint: str = "", categorical_answer: str = "") -> TargetSpec:
    """The mapper's figure `Source` as the digitiser's `TargetSpec`.

    Every hint is copied from the protocol or from what the mapper read in the paper; the x hint is
    the protocol's own measurement window, which is what makes "read the late part of the block"
    a protocol statement rather than something this code knows.

    `reviewer_hint` is the one line that comes from neither: a human's `re_extract` answer for this
    cell. It travels beside `panel_hint`, never instead of it.
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
        # `categorical_answer` is a reviewer's `categorical_axis_kind` decision for this cell:
        # "groups" becomes the caller statement resolver rule 1 consumes; "conditions" means the
        # outcome is the average across the axis, so the collapse the protocol did not switch on
        # globally is granted for this one cell — by a person, on the record.
        categorical_x=("groups" if categorical_answer == "groups" else "unknown"),
        collapse_across_x=(source.x_axis_kind == "categorical"
                           and (protocol.digitize.collapse_across_categorical_x
                                or categorical_answer == "conditions")),
        notes="; ".join(part for part in (source.quote, source.values_in_text, source.notes)
                        if part)[:400],
        reviewer_hint=reviewer_hint)


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
    effects: dict[tuple[str, str, str], str] = {}
    answered = apply_map_answers(study, _map_answers(ctx, paper), effects=effects)
    ctx.effects[paper.sha256] = effects
    return answered


def _applied_answers(ctx: "RunContext", paper: PaperRecord) -> set[tuple[str, str, str]]:
    """`(kind, dataset, outcome)` for every answer of this paper's that CHANGED its map.

    Filled in by `_answered_map`, which every stage that reads the map calls first. Empty when it
    has not run, and empty is the SAFE reading of "nobody knows": an answer nothing has recorded as
    applied stays pending, is asked about again, and costs a reviewer a second look — where the
    other default would retire a decision this run never took.
    """
    return {key for key, why in ctx.effects.get(paper.sha256, {}).items() if not why}


def _map_answers(ctx: "RunContext", paper: PaperRecord) -> list[dict[str, Any]]:
    """This paper's answers, read from the review log once per paper and cached (review L8)."""
    cached = ctx.answers.get(paper.sha256)
    if cached is None:
        cached = map_answers(ctx.out_dir, paper.sha256)
        ctx.answers[paper.sha256] = cached
    return cached


def _answered_eligibility(ctx: "RunContext", paper: PaperRecord) -> bool | None:
    """What a reviewer has decided about whether this PAPER is in the review (§C3), or `None`.

    The mapper's verdict is a model's reading of the protocol's criteria, and until §C3 it was
    final: a paper it called ineligible left the run with nobody having been asked. The card asks,
    and this is where the answer is acted on — before the exclusion, so a paper a reviewer put back
    is mapped and extracted like any other. The freshest answer wins, because a reviewer who
    changes their mind changes the review and the log still holds both.
    """
    answers = eligibility_answers(ctx.out_dir, paper.sha256)
    return bool(answers[-1]["eligible"]) if answers else None


def _eligibility_seqs(ctx: "RunContext", paper: PaperRecord, extracted: Sequence[str]) -> list[int]:
    """The `seq` of an inclusion this stage has now acted on — nothing until something was read.

    Same contract as `_consumed_seqs`: an answer is consumed when the reading it asked for has
    been bought, not when the stage that could buy it ran. A run that dies on its budget before
    reaching the paper a reviewer just included has not consumed that answer.
    """
    if not extracted:
        return []
    return sorted({int(answer["seq"])
                   for answer in eligibility_answers(ctx.out_dir, paper.sha256)
                   if isinstance(answer.get("seq"), int) and answer.get("eligible")})


#: §C3: the map stage's own record of the inclusions it has already bought a fresh map for.
#: Deliberately NOT `consumed_override_seqs`, which means "a reading was bought for this answer"
#: and is what the questions page reads to stop calling a decision pending — a map is not a
#: reading, and claiming one would tick off a card nothing had extracted for.
REMAPPED_FOR = "remapped_for_override_seqs"


def _needs_a_fresh_map(ctx: "RunContext", paper: PaperRecord) -> list[int]:
    """The inclusion(s) this paper's cached map cannot be acted on without re-mapping (§C3).

    A reviewer who includes a paper the mapper left with NO dataset is asking for a map: there is
    nothing on the cached one to extract, and the ordinary resume would exclude it again for the
    same reason. One map per answer, recorded in the stage file, so a paper nothing can be mapped
    in does not re-buy a map on every resume. A paper whose cached map HAS datasets is extracted
    off that map and buys nothing at all.
    """
    if not ctx.resume or not stage_done(ctx.out_dir, paper.sha256, "map"):
        return []
    if _answered_eligibility(ctx, paper) is not True:
        return []
    payload = read_stage(ctx.out_dir, paper.sha256, "map") or {}
    if (payload.get("study") or {}).get("datasets"):
        return []
    bought = {int(seq) for seq in payload.get(REMAPPED_FOR) or [] if isinstance(seq, int)}
    return sorted({int(answer["seq"])
                   for answer in eligibility_answers(ctx.out_dir, paper.sha256)
                   if answer.get("eligible") and isinstance(answer.get("seq"), int)
                   and answer["seq"] not in bought})


def _ruling_text(ctx: "RunContext", paper: PaperRecord, acting_on: Sequence[int] = ()) -> str:
    """The reviewer's inclusion ruling, as the mapper is told it (§C3).

    The mapper is given the criterion the person decided under and the paper's own words they
    relied on — the two fields the card records for exactly this reason.

    `acting_on` is the seqs this map is being bought FOR (`_needs_a_fresh_map`'s answer, which is
    the unconsumed ones). The ruling is read from the freshest of THOSE rather than from the
    freshest eligible record in the log, so the criterion the mapper is shown and the answer the
    map is charged to are the same record: with two inclusions where the later one already had its
    map, the mapper would otherwise be told a criterion nobody is currently asking about.
    """
    wanted = {int(seq) for seq in acting_on}
    including = [answer for answer in eligibility_answers(ctx.out_dir, paper.sha256)
                 if answer.get("eligible")
                 and (not wanted or answer.get("seq") in wanted)]
    if not including:
        return ""
    last = including[-1]
    return reviewer_ruling_line(str(last.get("rule") or ""), str(last.get("quote") or ""))


#: the extract stage's own record of the cells it has RE-read under a reviewer's hint and whose
#: readings changed, and — in `verify.json` / `resolve.json` — each later stage's record of the
#: ones it has already rebuilt for. Two keys, on disk, because the instruction to rebuild must be
#: exactly as durable as the `consumed_override_seqs` written beside it: `save()` records the
#: consumption as each cell is read, so an interruption between the extract stage and the verify
#: stage of the same paper would otherwise leave a verdict standing over readings the stage file
#: no longer holds — with the seq consumed, so no resume would look at the cell again and no card
#: anywhere would ask about it. The difference between the two keys is what a resume repairs, and
#: it is idempotent: a stage that dies before writing its own key simply repairs again.
#: (`REMAPPED_FOR` above is the same pattern for §C3's maps.)
REREAD_CELLS = "cells_reread_for_override"
#: what each cell's readings were taken under: `{cell: [metric, [locator, ...]]}`. The map alone
#: cannot say it — a resumed run re-applies the whole answer log to the pristine map every time,
#: so "what the map says now" is not "what this cell was read from", and a reviewer who changes
#: their mind back to the tool's own choice matches the pristine map while the numbers on the page
#: still belong to the measure they rejected.
READ_UNDER = "cells_read_under"
REREAD_APPLIED = "cells_reread_applied"


def _cells(names: Any) -> set[tuple[str, str]]:
    """`["d1/late_adaptation", …]` as `{(dataset_id, outcome_key), …}`."""
    return {(str(name).split("/", 1)[0], str(name).split("/", 1)[1])
            for name in names or () if isinstance(name, str) and "/" in name}


def _reread_on_record(ctx: "RunContext", paper: PaperRecord) -> list[str]:
    """Every cell the extract stage says it has re-read under a hint, freshest file wins."""
    if not stage_done(ctx.out_dir, paper.sha256, "extract"):
        return []
    payload = read_stage(ctx.out_dir, paper.sha256, "extract") or {}
    return sorted({str(name) for name in payload.get(REREAD_CELLS) or [] if isinstance(name, str)})


def _stale_cells(ctx: "RunContext", paper: PaperRecord, stage: str) -> set[tuple[str, str]]:
    """Cells a hinted re-read has changed that `stage` has not rebuilt for yet.

    Read from the stage files rather than carried in memory, so the in-process case and the
    interrupted one are the same case: what this stage owes is the difference between what the
    extract stage recorded re-reading and what this stage recorded acting on.
    """
    applied: Any = ()
    if stage_done(ctx.out_dir, paper.sha256, stage):
        applied = (read_stage(ctx.out_dir, paper.sha256, stage) or {}).get(REREAD_APPLIED)
    return _cells(_reread_on_record(ctx, paper)) - _cells(applied)


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


def _hinted_cells(ctx: "RunContext", paper: PaperRecord, study: StudyMap, keys: set[str],
                  blocked_datasets: Mapping[str, str],
                  blocked_cells: Mapping[tuple[str, str], str],
                  consumed: Collection[int] = (),
                  unreachable: dict[str, str] | None = None
                  ) -> dict[str, list[dict[str, Any]]]:
    """`{cell: [re_extract answer, ...]}` for the cells a reviewer has asked to have read again.

    Only cells THIS map still asks for: a hint on a dataset the review has since excluded, or on a
    cell an unanswered map question blocks, is a reading no resume may buy — the same rule
    `_answer_cells` applies to an inclusion, and for the same reason (whole-diff L4). Such a hint
    is not thrown away in silence: `unreachable` collects `{cell: why}` for it, which is what the
    stage warns about and what the review page prints instead of a re-read it will never buy.

    A cell is in only while one of its hints is unconsumed. That is what makes the hint buy ONE
    reading rather than one per resume for ever: the seqs go into `consumed_override_seqs` when the
    reading is bought, and the next resume sees nothing left to act on.
    """
    done = {int(seq) for seq in consumed if isinstance(seq, int)}
    unreachable = {} if unreachable is None else unreachable
    live = {f"{dataset.dataset_id}/{sources.outcome_key}"
            for dataset in study.datasets if dataset.dataset_id not in blocked_datasets
            for sources in dataset.outcomes
            if sources.outcome_key in keys
            and (dataset.dataset_id, sources.outcome_key) not in blocked_cells}
    cells: dict[str, list[dict[str, Any]]] = {}
    for answer in re_extract_answers(ctx.out_dir, paper.sha256):
        cell = f"{answer.get('dataset_id') or ''}/{answer.get('outcome_key') or ''}"
        if cell in live:
            cells.setdefault(cell, []).append(answer)
        elif answer.get("seq") not in done:
            # C4's honesty clause: a hint the map puts out of reach is not silently dropped. It
            # buys nothing here and nothing on any later resume, so the run SAYS which cell and
            # what is in the way — the reviewer was otherwise told "the next --resume re-reads
            # this cell" for ever while every resume walked straight past it.
            why = unreadable_cell(study, str(answer.get("dataset_id") or ""),
                                  str(answer.get("outcome_key") or ""), keys)
            unreachable[cell] = why or "this cell is not one the extract stage reads"
    return {cell: answers for cell, answers in cells.items()
            if any(answer.get("seq") not in done for answer in answers)}


#: how much of one hint the readers are shown, and how many hints. The line they end up in is
#: truncated whole (`llm.context.reviewer_hint_line`), so an unbounded join would let a further
#: hint fall off the end — producing a byte-identical prompt, hence the same cache key, hence a
#: `seq` consumed by a reading nothing new was bought for. Bounded here instead, where the reason
#: is visible: four hints of 200 characters cannot reach the line's own limit.
_HINT_CHARS, _HINTS_SHOWN = 200, 4


def _hint_text(answers: Sequence[Mapping[str, Any]]) -> str:
    """Every hint a reviewer has given this cell, as the one line the readers see.

    Joined rather than reduced to the freshest: each is a place a person says the value is, and a
    reader given two locations reads both. Duplicates are dropped — the same words twice are one
    instruction, and two copies of it in a prompt read as emphasis nobody wrote — and only the
    most recent `_HINTS_SHOWN` are carried, so the line can never grow past the point where a new
    hint stops changing it.

    The `group` a hint may carry is deliberately not read here: the extractors answer both arms in
    one call, so there is no such thing as re-reading one of them, and the cell is what is re-read.
    That is said out loud on the card and in the resume's warning rather than silently assumed.
    """
    hints = [" ".join(str(answer.get("hint") or "").split())[:_HINT_CHARS] for answer in answers]
    return "; ".join(list(dict.fromkeys(hint for hint in hints if hint))[-_HINTS_SHOWN:])


def _categorical_answer(answers: Sequence[Mapping[str, Any]]) -> str:
    """The latest structural `categorical_x` a reviewer gave this cell, or "".

    The sibling of `_hint_text` for the `categorical_axis_kind` card's answer. Latest-wins, not
    joined: "groups" and "conditions" are one decision that may change its mind, like a map
    answer — not two places a person looked. Free text cannot carry it: the re-read's TargetSpec
    consumes this as a caller statement (`_categorical_role` rule 1), and a prose hint never
    reaches that field.
    """
    for answer in reversed(list(answers)):
        value = str(answer.get("categorical_x") or "")
        if value:
            return value
    return ""


def _absorb_reread(cell: tuple[str, str], candidates: list[Candidate],
                   fresh: Sequence[Candidate], superseded: list[Candidate],
                   replace: bool = False, protected: frozenset[str] = frozenset(),
                   protected_note: str = "") -> bool:
    """Merge a hinted re-read into the cell's readings, losing none of them. Did anything change?

    `replace` is the one case where the earlier readings may not stand: a reviewer has changed WHICH
    MEASURE this cell is, so every reading already on it was taken against a measure the review has
    since rejected. Keeping those as a fallback would let a rejected measure supply the cell's
    numbers whenever the re-read came back empty — the losing answer winning by default. They are
    superseded, not deleted, exactly as below; the cell may legitimately end up with nothing, which
    is the honest outcome of "the measure you want is not reported here".

    **A hinted re-read never removes a reading that found a value.** A hint is a request for a
    better reading, not permission to lose the one the run already paid for: the questions page
    writes a `re_extract` the moment a reviewer types into `hint` — on any card, not only a
    valueless one — so "somebody asked" is not evidence that the cell had nothing. A re-read that
    comes back `not_on_these_pages`, which is what a hint pointing at the wrong place produces,
    therefore leaves the cell exactly as it was; and it leaves BOTH arms as they were, because a
    hint recorded against one group said nothing about the other.

    Ids are deterministic (`dataset:outcome:group:extractor#index`), so a reader's two readings of
    one cell collide by construction and exactly one of each pair may stand: the fresh one when it
    found something, the earlier one otherwise. The loser is not dropped — it goes to
    `superseded_candidates` in the stage file, because a reading a run bought is evidence about
    the paper whether or not the analysis weighs it.

    **A re-read never displaces a value a human supplied** (`protected`: the groups of this cell
    with a reviewer-typed mean, from `overrides.human_landed_values`). One step past the rule
    above: the hint still buys the reading — a human asked — but a fresh reading for a protected
    group goes straight to the shelf with `protected_note` in its provenance, and the standing
    readings (the ones the human's number was decided against) stay live, so verify re-votes that
    group over unchanged evidence and the reviewer is never asked to re-assert their own number.
    BOTH doors are shut: a colliding fresh reading may not win, and a non-colliding one (a route
    the first round never ran) may not slip into the live pool either — that side door is exactly
    how the observed clobber recurred. `replace` bypasses protection deliberately: a measure
    switch is itself a later human decision about the same cell, and a value typed against the
    rejected measure describes a number the review no longer wants.

    Returns whether the cell's readings actually changed, which is what makes the verdict and the
    row built from them stale. A re-read that changed nothing has nothing to rebuild.
    """
    guarded: list[Candidate] = []
    live_fresh = list(fresh)
    if protected and not replace:
        guarded = [c for c in fresh if c.group in protected]
        live_fresh = [c for c in fresh if c.group not in protected]
        for reading in guarded:
            reading.pixel_provenance["set_aside_for_human_value"] = protected_note \
                or "a reviewer supplied this group's value; this reading is recorded, not weighed"
    won = {c.candidate_id for c in live_fresh if c.status == "found"}
    if replace:
        won = {c.candidate_id for c in candidates if (c.dataset_id, c.outcome_key) == cell}
    kept = [c for c in candidates if (c.dataset_id, c.outcome_key) != cell
            or c.candidate_id not in won]
    standing = {c.candidate_id for c in kept}
    added = [c for c in live_fresh if c.candidate_id not in standing]
    superseded.extend(c for c in candidates
                      if (c.dataset_id, c.outcome_key) == cell and c.candidate_id in won)
    superseded.extend(c for c in live_fresh if c.candidate_id in standing)
    superseded.extend(guarded)
    candidates[:] = [*kept, *added]
    return bool(added or won)


def _human_valued_groups(landed: HumanLanded, dataset_id: str, outcome_key: str
                         ) -> frozenset[str]:
    """The groups of this cell whose MEAN a reviewer has stated — the write-protected ones.

    STATED, not landed: a number stuck pending is still a human's stated number, and protection
    against machine overwrite must not lapse while it waits (the hold-retirement rules are the
    ones that demand landed values). Group-scoped: a value on group A says nothing about B.
    """
    out = []
    for group in ("A", "B"):
        landed_cell = landed.for_cell(dataset_id, outcome_key, group)
        if landed_cell is not None and landed_cell.get("mean") is not None:
            out.append(group)
    return frozenset(out)


def _shelve_protected(extra: Sequence[Candidate], protected: frozenset[str],
                      landed: HumanLanded, dataset_id: str, outcome_key: str,
                      status: PaperStatus, shelved_out: list[Candidate]) -> list[Candidate]:
    """The candidates of an automated repair that may enter the VOTE — the protected groups'
    readings are stamped with the set-aside note, reported, collected for the T1 flags, and
    returned OUT of the merge. They stay wherever the caller records them (a paid reading is
    evidence about the paper) but are never weighed as a protected group's value."""
    if not protected:
        return list(extra)
    kept: list[Candidate] = []
    for reading in extra:
        if reading.group not in protected:
            kept.append(reading)
            continue
        human = landed.for_cell(dataset_id, outcome_key, str(reading.group)) or {}
        seq = (human.get("field_seqs") or {}).get("mean", human.get("seq", 0))
        reading.pixel_provenance["set_aside_for_human_value"] = (
            f"set aside: a reviewer supplied this group's value (override seq {seq}); this "
            f"automated repair's reading is recorded here, never weighed as the cell's value "
            f"without a human decision")
        shelved_out.append(reading)
        if reading.mean is not None:
            status.warnings.append(
                f"{dataset_id}/{outcome_key}: an automated repair read group {reading.group} "
                f"as {reading.mean:g}, but a reviewer has supplied that group's value "
                f"(override seq {seq}) — the fresh reading is recorded, not weighed")
    return kept


def _consumed_seqs(ctx: "RunContext", paper: PaperRecord, study: StudyMap, keys: set[str],
                   extracted: Sequence[str], already: Sequence[Any] = (),
                   blocked_cells: Mapping[tuple[str, str], str] = MappingProxyType({}),
                   re_read: Sequence[int] = (),
                   awaiting: Collection[int] = ()) -> list[int]:
    """The `seq` of every answer this stage has now acted on, unioned with what is on record.

    An answer is acted on when every cell it asks for has been extracted — not merely when the
    stage that could act on it ran. A run that dies on its budget before reaching the dataset a
    reviewer just included has not consumed that answer, and telling the reviewer it did would
    retire a decision nothing bought. Cumulative across resumes, per the consumer's contract:
    the union is written, never this resume's seqs alone.

    …and only an answer THE MAP ACCEPTED. `apply_map_answers` drops an answer naming a reading the
    map does not carry, or one that would leave both measures readable: it changed nothing, so no
    reading can be the reading it asked for. Retiring it anyway told a reviewer their refused answer
    had been applied and took the card off the page with it — the record claiming a decision landed
    at the one moment it provably had not.

    `awaiting` is the answers whose cells are already in `cells_extracted` but were read
    against the measure the answer REJECTS. "Every cell it asks for has been read" is true of
    those the moment the resume starts, so they would be retired before the re-reading they
    exist to buy — the same false consumption one layer down, and the one that would have made
    a reversal a decision the record accepted and the numbers never felt. They are retired by
    `re_read` and by nothing else.
    """
    done = set(extracted)
    seqs = {int(seq) for seq in already if isinstance(seq, int)}
    applied = _applied_answers(ctx, paper)
    owed = {int(seq) for seq in awaiting if isinstance(seq, int)}
    for answer in _map_answers(ctx, paper):
        seq = answer.get("seq")
        cells = _answer_cells(answer, study, keys, blocked_cells)
        if map_answer_key(answer) not in applied or seq in owed:
            continue
        if isinstance(seq, int) and cells and done.issuperset(cells):
            seqs.add(seq)
    # …and §C3's inclusion, whose cells are every cell of the paper: it is acted on the moment
    # this paper has been read at all, because being read IS what it asked for. Without this the
    # card said "not applied yet" for ever on a paper the resume had extracted.
    seqs.update(_eligibility_seqs(ctx, paper, extracted))
    # …and the re-extraction hints whose cell this stage has just read AGAIN. Passed in rather
    # than derived, because "the cell is in `cells_extracted`" is true of every cell an earlier
    # run read without the hint: what retires a hint is the reading bought FOR it.
    seqs.update(int(seq) for seq in re_read if isinstance(seq, int))
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
         status: PaperStatus, *, remap_for: Sequence[int] = ()) -> tuple[StudyMap, str]:
    bought: set[int] = set()
    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "map"):
        payload = read_stage(ctx.out_dir, paper.sha256, "map")
        bought = {int(seq) for seq in payload.get(REMAPPED_FOR) or [] if isinstance(seq, int)}
        if not remap_for:
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
    # §C3: when this map is bought BECAUSE a reviewer overruled the mapper, the mapper is told so.
    # Without it the re-map is the identical question in the identical words, so it comes back off
    # the prompt cache with the identical answer — `eligible=True, status excluded, $0.00`, which
    # is exactly what Wolpe and Roller did — and the reviewer's decision buys nothing at all.
    study = map_study(ctx.client, paper, ctx.protocol,
                      model_primary=ctx.models["primary"], model_check=ctx.models["secondary"],
                      model_adjudicate=ctx.models["adjudicator"], pdf_file_id=file_id or None,
                      reviewer_ruling=_ruling_text(ctx, paper, remap_for) if remap_for else "")
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
                 "consumed_override_seqs": [],
                 # …and §C3's own key, which is NOT that: it records the inclusions this stage has
                 # already bought a fresh map for, so a paper nothing can be mapped in does not
                 # re-buy one on every resume. A map is not a reading, so it does not retire the
                 # answer — the extract stage does that.
                 REMAPPED_FOR: sorted({*bought, *(int(seq) for seq in remap_for)})})
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


def _readable_at(study: StudyMap, dataset_id: str, outcome_key: str) -> tuple[str, tuple[str, ...]]:
    """What one cell's numbers may be read as, and where — `(metric, locators)`.

    The whole of what a `which_measure` decision controls, in the form two maps can be compared on.
    """
    for dataset in study.datasets:
        if dataset.dataset_id != dataset_id:
            continue
        for outcome in dataset.outcomes:
            if outcome.outcome_key != outcome_key:
                continue
            return (str(getattr(outcome.analysis_metric, "value", outcome.analysis_metric)),
                    tuple(sorted(s.locator for s in readable_sources(outcome.sources))))
    return ("", ())


def _measure_reread_hints(ctx: "RunContext", paper: PaperRecord, as_mapped: StudyMap,
                          study: StudyMap, extracted: Sequence[str],
                          consumed: Collection[int] = (),
                          read_under: Mapping[str, Any] = MappingProxyType({}),
                          ) -> dict[str, list[dict[str, Any]]]:
    """Cells a reviewer's `which_measure` answer re-decided AFTER the run had already read them.

    A measure answer to an OPEN question lands on a cell nobody read, and reading it is how the
    answer is acted on. An answer that overrules a ruling the map made FOR ITSELF lands on a cell
    this run has already read — and read against the measure the reviewer has just rejected.
    Nothing in the stage file tells those readings apart from good ones, so without this the map
    says one thing and the numbers say another, and a decision a person took is a note in a file
    that no forest plot ever feels.

    Routed through the same hints the re-extraction card uses, so nothing here is new machinery: the
    displaced readings go to `superseded_candidates` rather than being lost, the cell is marked in
    `REREAD_CELLS` so the stages after this one rebuild it, and the seq is retired by the READING
    rather than by this function having run.

    Only a cell the answer actually CHANGED. A reviewer who agrees with the ruling has confirmed it,
    and buying a second reading of a cell to arrive at the same numbers spends a person's money to
    learn nothing.
    """
    done = set(extracted)
    already = {int(seq) for seq in consumed if isinstance(seq, int)}
    applied = _applied_answers(ctx, paper)
    out: dict[str, list[dict[str, Any]]] = {}
    for answer in _map_answers(ctx, paper):
        key = map_answer_key(answer)
        kind, dataset_id, outcome_key = key
        cell = f"{dataset_id}/{outcome_key}"
        if kind != "which_measure" or key not in applied or cell not in done:
            continue
        if answer.get("seq") in already or not settled_measure(as_mapped, dataset_id, outcome_key):
            continue                        # nothing to overrule, or the reading was already bought
        # what the cell was last READ under, not what the map said before any answer. A reviewer
        # who switches, waits for the resume, then changes back is asking for the ORIGINAL measure
        # — which matches the pristine map, so comparing against that found no change, retired the
        # answer without buying anything, and left the cell holding the numbers of the measure they
        # had just rejected twice over. The stage file records what each cell was read under
        # (`READ_UNDER`) exactly so this comparison has a fact to make.
        was, now = (read_under.get(cell) or _readable_at(as_mapped, dataset_id, outcome_key),
                    _readable_at(study, dataset_id, outcome_key))
        if was == now:
            continue                        # the reviewer confirmed the ruling; the reading stands
        where = str(answer.get("winning_location") or "").strip()
        metric = str(answer.get("winning_analysis_metric") or "").strip()
        note = str(answer.get("note") or "").strip()
        # Short, and the instruction FIRST. `_hint_text` clips each hint to `_HINT_CHARS`, and a
        # long preamble spent the whole budget on the locator: every hint this function built came
        # out over the cap, most losing the imperative and all of them losing the reviewer's own
        # words — the run buying a reading with a sentence that stopped mid-locator. Worse, two
        # answers that differ only in the clipped tail render the same prompt, so the cache would
        # return the first one's reading for the second one's question.
        out.setdefault(cell, []).append({**answer, "hint": _clipped_hint(
            f"Read {metric or now[0] or 'the measure named here'}"
            + (f", at {_short_locator(where)}" if where else "")
            + f", not {was[0] or 'the other measure'}: a reviewer chose it for this review."
            + (f" Their words: {note}" if note else ""))})
    return out


def _short_locator(where: str, limit: int = 70) -> str:
    """A locator short enough to leave room for the instruction around it. Map locators run to
    hundreds of characters (every target direction of a figure's x axis), and the hint they sit in
    is clipped as a whole — so an un-trimmed one eats the sentence that says what to do with it."""
    text = " ".join(str(where or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _clipped_hint(text: str, limit: int = _HINT_CHARS) -> str:
    """…and the whole hint, trimmed at a word rather than mid-word, so nothing reads as truncated
    nonsense in a prompt the run is about to pay for."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut[:cut.rfind(" ")] if " " in cut else cut).rstrip(",;: ") + "…"


def _awaiting_reread(remeasured: Mapping[str, Sequence[Mapping[str, Any]]]) -> set[int]:
    """The seqs of the measure answers that owe this stage a re-reading before they are acted on."""
    return {int(record["seq"]) for records in remeasured.values() for record in records
            if isinstance(record.get("seq"), int)}


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
    as_mapped = study                       # …and what the map said before any of them was applied
    study = _answered_map(ctx, paper, study)
    blocked_datasets, blocked_cells = extraction_blocks(study)
    candidates: list[Candidate] = []
    done: list[str] = []
    exhausted: list[str] = []
    stopped = ""
    consumed: list[Any] = []
    only: set[str] | None = None
    #: C4's other half: the cells a reviewer has answered "it is over there" about, and the seqs
    #: this stage retires by re-reading them. A hint is the one answer whose consequence is a
    #: READING, so it is acted on here or nowhere.
    hints: dict[str, list[dict[str, Any]]] = {}
    #: …of which these are the cells a reviewer re-decided the MEASURE of. Held apart because
    #: their earlier readings may not stand as a fallback: they were taken against a measure
    #: this review has since rejected (`_absorb_reread(replace=True)`).
    remeasured: dict[str, list[dict[str, Any]]] = {}
    #: `{cell: (metric, locators)}` — what each cell's readings were taken under (`READ_UNDER`)
    read_under: dict[str, Any] = {}
    re_read: list[int] = []
    #: the cells a hinted re-read has CHANGED, and the readings it displaced. Both go in the stage
    #: file: the first is what the stages after this one owe a rebuild for, the second is the
    #: evidence a re-read is not allowed to lose.
    reread: set[str] = set()
    superseded: list[Candidate] = []
    #: hints this map puts out of reach, and what is in the way. Never silently dropped: the
    #: promise on the card ("the next --resume re-reads this cell") is one no resume can keep.
    unreachable: dict[str, str] = {}
    #: what humans have stated about this run's cells — recomputed from the log on every entry,
    #: so an interrupted resume re-derives the same protection instead of trusting stage state
    landed = human_landed_values(ctx.out_dir)

    def say_what_cannot_be_read() -> None:
        """Warn once per hint nothing will ever act on — before any early return, in every branch.

        Said whether or not this stage is re-entered: the hint is on the record, no resume will
        buy the reading it asks for, and a reviewer re-running for that reading is entitled to
        know why rather than getting the identical "it is recorded" line back for ever.
        """
        for cell, why in sorted(unreachable.items()):
            status.warnings.append(
                f"{cell}: a reviewer's re-extraction hint cannot be acted on — {why}")

    def say_what_changed_nothing() -> None:
        """Warn once per map answer this map REFUSED, and why.

        The other half of the same honesty: an answer that changed nothing is no longer
        recorded as acted on (`_consumed_seqs`), so without this it would sit on the page as
        "waiting for a re-run" through every re-run there will ever be. A reviewer is owed the
        sentence saying their answer named something this map does not carry.
        """
        for (kind, dataset_id, outcome_key), why in sorted(
                ctx.effects.get(paper.sha256, {}).items()):
            if not why:
                continue
            where = f"{dataset_id}/{outcome_key}" if outcome_key else dataset_id
            status.warnings.append(
                f"{where}: a reviewer's {kind} answer changed nothing — {why}")

    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "extract"):
        payload = read_stage(ctx.out_dir, paper.sha256, "extract")
        candidates = [Candidate.model_validate(c) for c in payload["candidates"]]
        superseded = [Candidate.model_validate(c)
                      for c in payload.get("superseded_candidates") or []]
        reread = {str(cell) for cell in payload.get(REREAD_CELLS) or [] if isinstance(cell, str)}
        read_under = {str(cell): (str(what[0]), tuple(str(x) for x in what[1]))
                      for cell, what in (payload.get(READ_UNDER) or {}).items()
                      if isinstance(what, (list, tuple)) and len(what) == 2}
        done = [str(cell) for cell in payload.get("cells_extracted") or []]
        consumed = list(payload.get("consumed_override_seqs") or [])
        hints = _hinted_cells(ctx, paper, study, keys, blocked_datasets, blocked_cells,
                              consumed, unreachable)
        remeasured = _measure_reread_hints(ctx, paper, as_mapped, study, done, consumed,
                                           read_under)
        for cell, records in remeasured.items():
            hints.setdefault(cell, []).extend(records)
        say_what_cannot_be_read()
        say_what_changed_nothing()
        if remeasured:
            status.warnings.append(
                f"{len(remeasured)} cell(s) are being read again because a reviewer changed "
                f"which measure this review reads ({', '.join(sorted(remeasured))}) — the "
                f"readings taken against the measure they rejected do not stand as a fallback; "
                f"they are kept in this paper's extract stage file under "
                f"`superseded_candidates`, so a cell whose new measure the paper does not report "
                f"ends with no rows at all")
        # a cell nobody has read, or one a reviewer has asked to have read AGAIN. The second is
        # not "still unread" by any test the stage file can apply — it was read, and came back
        # with nothing usable — so it is unioned in rather than folded into that rule.
        only = _cells_still_unread(study, keys, blocked_datasets, blocked_cells, done,
                                   candidates) | set(hints)
        if not only:
            # nothing new to read. The stage file is still rewritten when an answer has finished
            # being acted on, so a decision whose cells were already extracted stops being pending
            # instead of waiting for a reading nobody owes it.
            seqs = _consumed_seqs(ctx, paper, study, keys, done, consumed, blocked_cells,
                                  awaiting=_awaiting_reread(remeasured))
            if seqs != sorted(int(s) for s in consumed if isinstance(s, int)):
                write_stage(ctx.out_dir, paper.sha256, "extract", {**payload,
                                                                   "consumed_override_seqs": seqs})
            status.stages["extract"] = "skipped"
            return candidates
        unread = sorted(only - set(hints))
        if unread:
            status.warnings.append(
                f"{len(unread)} cell(s) this map asks for had never been extracted "
                f"({', '.join(unread)}) — the extract stage was re-entered for those cells "
                f"only; every other cell keeps the reading an earlier run paid for")
        hinted = sorted(set(hints) - set(remeasured))
        if hinted:
            # …the cells re-read for a HINT. A remeasured cell is in `hints` too (it travels the
            # same machinery) but the opposite is true of it — its earlier readings do not stand
            # — so listing it here as well printed two contradictory sentences about one cell.
            status.warnings.append(
                f"{len(hinted)} cell(s) are being read again with a reviewer's hint "
                f"({', '.join(hinted)}) — the WHOLE cell is re-read, both groups, "
                f"whichever group the hint was recorded against; a reading that found a value is "
                f"replaced only by a re-reading that finds one")
    else:
        hints = _hinted_cells(ctx, paper, study, keys, blocked_datasets, blocked_cells,
                              consumed, unreachable)
        say_what_cannot_be_read()
        say_what_changed_nothing()

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
                     "consumed_override_seqs": _consumed_seqs(
                         ctx, paper, study, keys, done, consumed, blocked_cells, re_read,
                         awaiting=_awaiting_reread(remeasured)),
                     # the readings a hinted re-read displaced. Kept, not dropped: a reading the
                     # run bought is evidence about the paper whether or not the analysis weighs
                     # it, and a re-read that overwrote it unrecoverably would make the hint a
                     # way of losing data.
                     "superseded_candidates": [c.model_dump(mode="json") for c in superseded],
                     # …and which cells that re-reading actually changed, for the stages after
                     # this one. Written beside the consumption, so the two can never disagree.
                     REREAD_CELLS: sorted(reread),
                     READ_UNDER: {cell: list(what) for cell, what in read_under.items()}})

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
            hint = _hint_text(hints.get(cell, ()))
            categorical = _categorical_answer(hints.get(cell, ()))
            pair = (dataset.dataset_id, sources.outcome_key)
            protected = _human_valued_groups(landed, dataset.dataset_id, sources.outcome_key)
            protected_note = ""
            if protected:
                seqs = sorted({(landed.for_cell(dataset.dataset_id, sources.outcome_key, g)
                                or {}).get("field_seqs", {}).get("mean", 0) for g in protected})
                protected_note = (
                    f"set aside: a reviewer supplied this group's value (override seq "
                    f"{', '.join(str(s) for s in seqs)}); this re-read's reading is recorded "
                    f"here, never weighed as the cell's value without a human decision "
                    # repeated set-asides collide on deterministic candidate ids — the stamp is
                    # what tells one shelved batch from another
                    f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}]")
            # a hinted re-read is read into a scratch list and merged afterwards, never over the
            # cell's own readings: nothing may be removed before the reading meant to replace it
            # is known to have found anything. A cap that lands mid-cell takes the same path, so
            # the merge happens either way and the readings already paid for are kept.
            fresh: list[Candidate] = []
            try:
                _extract_cell(ctx, paper, dataset, sources, figures_dir, status,
                              out=fresh if hint else candidates, reviewer_hint=hint,
                              categorical_answer=categorical)
            except PaperBudgetExceeded as exc:
                if hint:
                    # `replace` here too: a cap landing mid-cell must not leave the rejected
                    # measure's readings beside whatever the new one produced — one cell holding
                    # two measures is the `metric_mixed` state C6 exists to prevent. The seq is
                    # not retired on this path, so the next resume reads the cell again.
                    _absorb_reread(pair, candidates, fresh, superseded,
                                   replace=cell in remeasured, protected=protected,
                                   protected_note=protected_note)
                stopped = str(exc)
                exhausted.append(cell)
                status.warnings.append(
                    f"{cell}: budget_exhausted — {exc}; the rows this cell had already produced "
                    f"are kept, the cells after it were not started")
                continue
            if cell not in done:
                done.append(cell)
            # what this reading was taken under, recorded beside the reading itself: it is the
            # only fact that can tell a reviewer's later change of mind from their first one.
            read_under[cell] = _readable_at(study, dataset.dataset_id, sources.outcome_key)
            if hint:
                # fix D': a re-read that found nothing says WHY, per group, before the merge
                # routes its candidates away — three cells on one real run failed a paid re-read
                # in total silence, and no one (human or tool) could tell what the blocker was.
                for g in ("A", "B"):
                    fresh_mine = [c for c in fresh if c.group == g]
                    if not fresh_mine:
                        status.warnings.append(
                            f"{cell}: the hinted re-read returned no candidates at all for "
                            f"group {g}")
                    elif not any(c.mean is not None for c in fresh_mine):
                        said = next((str(c.notes
                                         or (c.pixel_provenance or {}).get("needs_review_reason")
                                         or "") for c in fresh_mine
                                     if (c.notes or (c.pixel_provenance or {}
                                                     ).get("needs_review_reason"))), "")
                        status.warnings.append(
                            f"{cell}: the hinted re-read bought a fresh look for group {g} and "
                            f"found no value — {said[:200] or 'the readers gave no reason'}")
                        # …and the reason travels to the re-opened card: the cell's surviving
                        # ensemble records carry it in provenance, which the review layer reads
                        for old in candidates:
                            if (old.dataset_id == pair[0] and old.outcome_key == pair[1]
                                    and old.group == g and said
                                    and old.extractor_id == "digitize:ensemble"
                                    and old.pixel_provenance is not None):
                                old.pixel_provenance.setdefault(
                                    "reread_found_nothing", said[:300])
                # …and only now: the seqs are retired by the reading, not by the stage reaching
                # the cell. A re-read that changed the cell's readings makes the verdict and the
                # row built from them stale, and that is recorded in the stage file beside the
                # consumption it pairs with — never in memory alone (see `REREAD_CELLS`).
                if _absorb_reread(pair, candidates, fresh, superseded,
                                  replace=cell in remeasured, protected=protected,
                                  protected_note=protected_note):
                    reread.add(cell)
                # S-5: a hint whose readings were ALL set aside would otherwise vanish — bought,
                # shelved, and shown nowhere. The warning names what was read; the disagreement,
                # if any, becomes the T1 flag (and its one card) at the next verify pass.
                shelved_now = [c for c in fresh
                               if c.group in protected and c.mean is not None] \
                    if cell not in remeasured else []
                if shelved_now:
                    status.warnings.append(
                        f"{cell}: the hinted re-read bought fresh reading(s) for group(s) "
                        f"whose value a reviewer supplied — set aside, never weighed: "
                        + "; ".join(f"group {c.group} read {c.mean:g}"
                                    for c in shelved_now[:4])
                        + " — the reviewer's numbers stand")
                re_read.extend(int(answer["seq"]) for answer in hints[cell]
                               if isinstance(answer.get("seq"), int))
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
                  status: PaperStatus, out: list[Candidate] | None = None,
                  reviewer_hint: str = "", categorical_answer: str = "",
                  force_reacquire: str = "") -> list[Candidate]:
    """Both text variants, the statistic reader, and the digitiser once per figure source.

    `out` is filled as each reader answers rather than returned at the end, so a budget death half
    way through a cell keeps the readings that were already paid for.

    `reviewer_hint` is a human's `re_extract` answer for this cell, and it reaches EVERY reader —
    the two text variants, the statistic pass and each figure read-out — because a reviewer who
    says where the value is has not said which reader should find it there. It is added to the
    locations the mapper gave and never substituted for them.
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
                                       reviewer_hint=reviewer_hint,
                                       model=ctx.models["primary"]))
        out.extend(extract_group_stats(ctx.client, paper, ctx.protocol, dataset, key,
                                       readable, variant="narrative_first",
                                       reviewer_hint=reviewer_hint,
                                       model=ctx.models["secondary"]))
        out.extend(extract_test_statistics(ctx.client, paper, ctx.protocol, dataset, key,
                                           readable, reviewer_hint=reviewer_hint,
                                           model=ctx.models["primary"]))
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
        target = target_for_source(source, dataset, sources, ctx.protocol, ctx.settings,
                                   reviewer_hint, categorical_answer=categorical_answer)
        try:
            digitised = digitize(ctx.client, paper, figure, target, source=source,
                                 dataset=dataset,
                                 out_dir=figures_dir, caption=figure.caption,
                                 models=(ctx.models["primary"],),
                                 settings=ctx.protocol.digitize,
                                 n_readouts=ctx.protocol.digitize.readouts_max,
                                 # fix G: a caller with a refutation in hand sends the figure
                                 # read straight to the page render; text readers never see it
                                 force_reacquire=force_reacquire,
                                 cell_key=f"{paper.sha256[:12]}/{dataset.dataset_id}/{key}/"
                                          f"{figure.id}")
        except (BudgetExceeded, PaperBudgetExceeded):
            raise                               # money is the run's business, not the cell's
        except TruncatedOutput as exc:
            # C8, same rule as the verifier's own catch below: a read-out that wrote past its
            # output limit twice is a reading this cell DOES NOT HAVE — an absence, not
            # evidence, and not a reason for the whole paper to die with every other cell's
            # candidates unread (Cornelis 2022 lost a 30-file run's paper to one such call).
            # The cell keeps whatever its other sources produced; with none it resolves
            # `not_convertible` and is held, which is the honest state of an unread figure.
            status.warnings.append(
                f"{dataset.dataset_id}/{key}: digitize truncated at its output limit twice on "
                f"{figure.id} — that reading was not produced ({str(exc)[:120]}); the cell "
                f"keeps its other candidates and the figure is unread, not misread")
            continue
        out.extend(digitised if isinstance(digitised, list) else digitised.candidates)
    # fix C, the last resort: a group every readable source left without a value promotes ONE of
    # the map's own recorded alternates — the record the cell would otherwise die ignoring. Gated
    # entirely on record enums: never a C6-demoted loser (that is the which_measure decision's
    # territory and a pinned behaviour), never a sample that cannot carry the contrast. The
    # promotion is a recursive read of a one-source copy with the role flipped, so text and
    # figure alternates take their ordinary reader paths and no second promotion can fire (the
    # copy's source list holds no alternate). Bounded: one per cell per pass, priced under the
    # paper's own cap like any read.
    mine = [c for c in out if c.dataset_id == dataset.dataset_id and c.outcome_key == key]
    have = {c.group for c in mine if c.mean is not None}
    missing = [g for g in ("A", "B") if g not in have]
    if missing:
        read_markers = {_source_marker(s) for s in readable}
        # a mapper-native alternate can carry a DIFFERENT measure ("DE (primary); IEE
        # (alternative)") — promoting that reads a different quantity into the vote, the
        # metric_mixed failure C6 exists to prevent. So the alternate's metric must match a
        # readable value source's (or be unstated, which is the map saying nothing).
        value_metrics = {str(getattr(v, "analysis_metric", "") or "")
                         for v in readable} - {"", "unknown"}
        promoted = next(
            (s for s in sources.sources
             if s.role == "alternate" and not c6_demoted(s)
             and s.sample not in UNREADABLE_SAMPLES
             and _source_marker(s) not in read_markers
             and (str(getattr(s, "analysis_metric", "") or "") in ("", "unknown")
                  or not value_metrics
                  or str(getattr(s, "analysis_metric", "")) in value_metrics)), None)
        if promoted is not None:
            status.warnings.append(
                f"{dataset.dataset_id}/{key}: group(s) {', '.join(missing)} got no value from "
                f"any readable source, so the map's own alternate {promoted.locator[:80]!r} is "
                f"read as a last resort")
            out.extend(_extract_cell(
                ctx, paper, dataset,
                sources.model_copy(update={"sources":
                                           [promoted.model_copy(update={"role": "value"})]}),
                figures_dir, status, reviewer_hint=reviewer_hint,
                categorical_answer=categorical_answer))
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
    landed = human_landed_values(ctx.out_dir) if ctx is not None else HumanLanded({}, {})
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
        # nothing here was requested by a person, and the model call has not been bought yet —
        # so when a reviewer has supplied the refuted group's value, the cheapest honest
        # protection is not to buy it (the same rule fix G draws below). Skip, and say so.
        refuted_group = next((c.group for c in cell
                              if c.candidate_id == verdict.candidate_id), None)
        if refuted_group in ("A", "B"):
            human = landed.for_cell(dataset.dataset_id, sources.outcome_key, str(refuted_group))
            if human is not None and human.get("mean") is not None:
                status.warnings.append(
                    f"{dataset.dataset_id}/{sources.outcome_key}: a verifier named {named!r} as "
                    f"a better source for group {refuted_group}, but a reviewer has supplied "
                    f"that group's value (override seq "
                    f"{human['field_seqs'].get('mean', human['seq'])}) — the re-open was not "
                    f"bought; theirs wins")
                continue
        if _source_marker(source) in already:
            # "already read" only counts when the reading it bought actually SERVED the refuted
            # group (fix B). Kumar's insets were mapped, read, and refused by a gate that has
            # since been fixed; the verifier then named them and this skip threw its knowledge
            # away. A refutation is the statement that the standing reading did not serve the
            # group, so a named source is re-opened UNLESS it already gave that group a value —
            # then re-reading truly buys nothing and value-level refutations stay a question.
            group = next((c.group for c in cell
                          if c.candidate_id == verdict.candidate_id), None)
            if verdict.verdict != "refuted" or group is None:
                continue
            served = any(c.group == group and c.mean is not None
                         and source_of(c, readable) is source
                         and not str(c.candidate_id).endswith(":reopen") for c in cell)
            reopened_before = any(str(c.candidate_id).endswith(":reopen")
                                  and source_of(c, readable) is source for c in cell)
            if served or reopened_before:
                continue
        # the re-extraction's own warnings belong to the paper, not to a throwaway status object
        extra = _extract_cell(ctx, paper, dataset,
                              sources.model_copy(update={"sources": [source]}),
                              paper_dir(ctx.out_dir, paper.sha256) / "figures", status)
        if extra:
            # a distinct id: the reopened read must never share a candidate_id with the reading
            # it answers (ensemble ids omit the figure, so a same-source re-read would collide
            # and conflate flag/verdict lookups), and the suffix is the recurrence marker above
            for c in extra:
                c.candidate_id = f"{c.candidate_id}:reopen"
            return named, list(extra)
        status.warnings.append(
            f"{dataset.dataset_id}/{sources.outcome_key}: re-opened on {named!r} at the verifier's "
            f"suggestion and it produced no candidate")
    return None


def _reacquire_on_refutation(ctx: RunContext, paper: PaperRecord, dataset: DatasetSpec,
                             sources: OutcomeSources,
                             refuted: Sequence[tuple[VerifierVerdict, Candidate]],
                             cell: Sequence[Candidate], status: PaperStatus
                             ) -> tuple[str, list[Candidate]] | None:
    """Fix G: a figure read the verifier refuted AGAINST PRINTED VALUES is re-read from the page.

    The verifier already catches "the figure read contradicts what the paper prints" and files a
    `refuted` verdict with the printed evidence in its structured fields; fix E already knows how
    to re-acquire a read from the full page render. This is the wire between them — one bounded
    retry, the `crop_reacquired` cap on the re-read, and the knowledge stops dead-ending in a
    card. The trigger is the verdict's STRUCTURED shape only (`alt_quote` citing a number and an
    `alt_mean` that actually disagrees with the reading) — never a parse of its prose.

    `refuted` pairs each verdict with the exact Candidate it judged: ensemble candidate ids omit
    the figure, so a cell with two figure sources holds two candidates with identical ids and a
    lookup by id could re-acquire the wrong figure.
    """
    if any(str(c.candidate_id).endswith(":reacquire") for c in cell):
        return None                       # once per cell, ever — the marker survives resume
    if any(str(rec.get("kind")) == "re_extract"
           and rec.get("dataset_id") == dataset.dataset_id
           and rec.get("outcome_key") == sources.outcome_key
           for rec in read_overrides(ctx.out_dir)):
        return None                       # a pending hint is a human mid-decision on this whole
        # cell (a hint re-reads both groups); theirs wins
    #: …while a typed VALUE is group-scoped, like the value itself: a reviewer's number for
    #: group A says nothing about B, and blocking B's re-acquire on it left B's refutation
    #: dead-ended in a card. The registry (not a raw kind-scan) is what all three protection
    #: paths consult, so a group-less record in an old log still protects both groups.
    landed = human_landed_values(ctx.out_dir)
    readable = readable_sources(sources.sources)
    for verdict, winner in refuted:
        if verdict.verdict != "refuted":
            continue
        if winner.group in ("A", "B"):
            human = landed.for_cell(dataset.dataset_id, sources.outcome_key, str(winner.group))
            if human is not None and human.get("mean") is not None:
                continue                  # a human stated this group's value; theirs wins
        quote = str(verdict.alt_quote or "")
        if not any(ch.isdigit() for ch in quote) or verdict.alt_mean is None:
            continue
        prov = winner.pixel_provenance if isinstance(winner.pixel_provenance, dict) else {}
        prov_fid = str(prov.get("figure_id") or "")
        if not prov_fid or prov.get("crop_reacquired"):
            # not a figure read, or already a page-level read — re-buying the identical page
            # render replays the cache byte for byte and buys nothing
            continue
        if winner.mean is not None and \
                abs(float(verdict.alt_mean) - float(winner.mean)) \
                <= 1e-9 + 1e-6 * abs(float(winner.mean)):
            continue                      # the printed value AGREES with the reading; the
        # refutation is about something a wider image cannot settle
        # panel reads carry the PANEL id ("fig02a"); sources name the figure ("fig02") — match
        # by prefix with an alphabetic remainder, never by equality alone
        source = next((s for s in readable if s.figure_id
                       and (prov_fid == s.figure_id
                            or (prov_fid.startswith(s.figure_id)
                                and prov_fid[len(s.figure_id):].isalpha()))), None)
        if source is None:
            continue
        figure = _figure(paper, source.figure_id)
        page = next((p for p in (paper.pages or [])
                     if figure is not None and p.number == figure.page), None)
        if figure is None or page is None or not page.png:
            continue                      # nothing wider exists; the refutation stays for
        # adjudication exactly as before this fix
        note = (f"a verifier refuted this figure read against printed values "
                f"({quote[:160]!r}), so the reading was re-acquired from the full page render")
        extra = _extract_cell(ctx, paper, dataset,
                              sources.model_copy(update={"sources": [source]}),
                              paper_dir(ctx.out_dir, paper.sha256) / "figures", status,
                              force_reacquire=note)
        reacquired = [c for c in extra
                      if isinstance(c.pixel_provenance, dict)
                      and c.pixel_provenance.get("crop_reacquired")]
        if not reacquired:
            # the digitiser fell back to the ordinary read (or produced nothing) — a cached
            # replay of the refuted reading must never re-enter the vote as fresh corroboration
            status.warnings.append(
                f"{dataset.dataset_id}/{sources.outcome_key}: a verifier refuted the figure "
                f"read against printed values but the re-acquire produced no page-level "
                f"reading; the refutation stands for adjudication")
            return None
        for c in extra:
            # a distinct id: same collision guard as `:reopen` above, and the once-ever marker
            c.candidate_id = f"{c.candidate_id}:reacquire"
        return quote, list(extra)
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
                 tiebroken: set[tuple[str, str]] | None = None,
                 shelved: Sequence[Candidate] = ()) -> _CellVerification:
    key = sources.outcome_key
    outcome_def = ctx.protocol.outcome(key)
    #: T1: the groups of this cell whose value a reviewer has stated. Consulted by every
    #: automated repair below before its fresh readings may enter the VOTE, and by the flag
    #: block at the end. `shelved` is the extract stage's set-aside shelf — the readings
    #: protection kept out of the live pool, which are the only evidence that genuinely
    #: POSTDATES the human's number.
    landed_here = human_landed_values(ctx.out_dir)
    protected_cell = _human_valued_groups(landed_here, dataset.dataset_id, key)
    shelved_this_pass: list[Candidate] = []
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
    #: (verdict, the exact Candidate it judged) — fix G needs the object, not the id: ensemble
    #: candidate ids omit the figure, so two figure sources produce identical ids in one cell
    judged_pairs: list[tuple[VerifierVerdict, Candidate]] = []
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
            judged_pairs.append((verdict, winner))
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
        # M-2: `_extract_cell` reads the source for the WHOLE cell, so an automated repair
        # aimed at one group's refutation returns the sibling's reading too — and a sibling a
        # human has valued must not have it WEIGHED. It stays on the record (extra_candidates —
        # a paid reading is evidence about the paper, and the option it backs is a choice a
        # person may still make); it never enters the vote.
        extra = _shelve_protected(extra, protected_cell, landed_here, dataset.dataset_id, key,
                                  status, shelved_this_pass)
        cell = [*cell, *vote_candidates(extra)]
        extra_flags.append(CheckFlag(
            code="reopened_on_better_source", severity=CHECK_SEVERITY["reopened_on_better_source"],
            message=(f"a verifier named {named!r} as a better source for this outcome and "
                     f"extraction was re-opened on it (one hop, once)"),
            candidate_ids=sorted(c.candidate_id for c in vote_candidates(extra))))
        flags = [*run_checks(dataset, key, cell, other_candidates=others,
                             orientation=orientation), *extra_flags]
        votes = vote_groups(cell, unit_hint=sources.units or outcome_def.units_hint)
    else:
        # fix G, only when fix B did not act (one repair per cell per pass; a verifier that both
        # named a better source and cited print got the more specific cure above): a figure read
        # refuted against printed values is re-read from the full page render.
        reacq = _reacquire_on_refutation(ctx, paper, dataset, sources, judged_pairs, cell,
                                         status)
        if reacq is not None:
            quote, extra = reacq
            out.extra_candidates.extend(extra)
            extra = _shelve_protected(extra, protected_cell, landed_here, dataset.dataset_id,
                                      key, status, shelved_this_pass)      # M-2, as above
            cell = [*cell, *vote_candidates(extra)]
            extra_flags.append(CheckFlag(
                code="reacquired_on_refutation",
                severity=CHECK_SEVERITY["reacquired_on_refutation"],
                message=(f"a verifier refuted the figure read against printed values "
                         f"({quote[:160]!r}) and the reading was re-acquired from the full "
                         f"page render (one retry, once per cell)"),
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

    # fix H's record: disagreeing routes were settled as a negligible split (same side of zero,
    # same place, whole spread under NEGLIGIBLE_D of the cell's own SD). The cell pools under a
    # cap rather than holding for a human coin flip; the flag is what the cap and the reviewer
    # see.
    for group in ("A", "B"):
        result = votes.get(group)
        if result is None or result.method != NEGLIGIBLE_SPLIT:
            continue
        flags = [*flags, CheckFlag(
            code="negligible_split_resolved",
            severity=CHECK_SEVERITY["negligible_split_resolved"],
            message=next((note for note in result.notes if "negligible" in note
                          or "cannot visibly move" in note),
                         "disagreeing readings were settled as measurement noise"),
            candidate_ids=sorted(set(result.agreeing_ids)))]

    # T1's record, once the votes are final — derived from the SET-ASIDE readings only, never
    # from the vote. A protected group's live pool cannot gain fresh readings (protection
    # shelves them), so this pass's vote re-resolves the pre-human number for ever: comparing
    # IT with the human's value would raise "disputes" off evidence the reviewer already read
    # when they typed their number — and the disputes flag blocks T4's retirement, re-arming
    # the very loop the protection kills (M-3). The shelf is the one place evidence that
    # genuinely POSTDATES the human's number can be, so it alone speaks: agreement is a record
    # (info), disagreement holds the cell once, capped, for its one human look. A re-vote over
    # unchanged candidates raises nothing — `overridden_by_human` already records the standing
    # disagreement. (A `replace=True` measure-switch re-read bypasses protection and its
    # readings enter the live pool unflagged — the named residual of design edge 4.7.)
    for group in ("A", "B"):
        human = landed_here.for_cell(dataset.dataset_id, key, group)
        if human is None or human.get("mean") is None:
            continue
        fresh = [c for c in [*shelved, *shelved_this_pass]
                 if c.dataset_id == dataset.dataset_id and c.outcome_key == key
                 and c.group == group and c.mean is not None]
        if not fresh:
            continue
        reading = fresh[-1]                     # the latest set-aside batch speaks
        own_sd = _verified_sd([c for c in cell if c.group == group])
        agrees = within_read_tolerance(reading.mean, human["mean"], sd=own_sd)
        code = "reread_confirms_human_value" if agrees else "reread_disputes_human_value"
        flags = [*flags, CheckFlag(
            code=code, severity=CHECK_SEVERITY[code],
            message=(f"a reviewer supplied this group's value ({human['mean']:g}, override seq "
                     f"{human['field_seqs'].get('mean', human['seq'])}); a later re-read's own "
                     f"reading was {reading.mean:g}, set aside rather than adopted — "
                     + ("the two agree within read tolerance" if agrees else
                        "the fresh reading DISAGREES; the reviewer's value stands unless a "
                        "person decides otherwise")),
            candidate_ids=[reading.candidate_id])]

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
    #: T1: the extract stage's set-aside shelf — the readings protection kept out of the live
    #: pool because a reviewer had supplied the group's value. Read ONCE per paper (the stage
    #: file runs to megabytes) and handed to every cell: they are the only evidence that
    #: genuinely postdates a human's number, which is what the T1 flags speak from.
    shelf = [Candidate.model_validate(c)
             for c in (read_stage(ctx.out_dir, paper.sha256, "extract") or {}
                       ).get("superseded_candidates") or []
             if isinstance(c, dict)
             and (c.get("pixel_provenance") or {}).get("set_aside_for_human_value")]

    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "verify"):
        payload = read_stage(ctx.out_dir, paper.sha256, "verify")
        verdicts = [Verdict.model_validate(v) for v in payload["verdicts"]]
        extra = [Candidate.model_validate(c) for c in payload.get("extra_candidates", [])]
        # …except for a cell whose readings a hinted re-read has changed and this stage has not
        # rebuilt for: what it holds there is a verdict over readings the stage file no longer
        # has, and a re-opened candidate bought to settle them. Both are dropped, which puts the
        # cell back in the case below — candidates and no verdict — and it is verified from the
        # new reading like any other newly extracted cell. The debt is read off the stage files,
        # so an interrupted resume owes exactly the repair this one would have made.
        stale = _stale_cells(ctx, paper, "verify")
        if stale:
            verdicts = [v for v in verdicts if (v.dataset_id, v.outcome_key) not in stale]
            extra = [c for c in extra if (c.dataset_id, c.outcome_key) not in stale]
            status.warnings.append(
                f"{len(stale)} cell(s) were read again with a reviewer's hint "
                f"({', '.join(f'{d}/{k}' for d, k in sorted(stale))}) — the verdict an earlier "
                f"run reached over the readings that re-reading replaced was dropped, not kept")
        # the same rule as the extract stage's, one stage on: this stage is not done while a cell
        # that now HAS candidates has no verdict. Without it an answer acted on by the resumed
        # extract stage produced candidates and no row, which is the same dead end one layer up.
        only = {(c.dataset_id, c.outcome_key) for c in [*candidates, *extra]
                if c.dataset_id not in blocked_datasets and c.outcome_key in keys
                and (c.dataset_id, c.outcome_key) not in blocked_cells} - {
                    (v.dataset_id, v.outcome_key) for v in verdicts}
        if not only:
            if stale:
                # a re-read cell the map now blocks: there is nothing to verify and nothing left
                # to owe, so the debt is settled on the file rather than re-attempted, silently
                # and for nothing, on every resume after this one.
                write_stage(ctx.out_dir, paper.sha256, "verify",
                            {**payload,
                             "verdicts": [v.model_dump(mode="json") for v in verdicts],
                             "extra_candidates": [c.model_dump(mode="json") for c in extra],
                             REREAD_APPLIED: _reread_on_record(ctx, paper)})
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
                                  file_id, orientations[measure], status, tiebroken,
                                  shelved=shelf)
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
        # which hinted re-reads this stage has now rebuilt for. Written LAST, with the verdicts it
        # rebuilt: a stage that dies before this file exists still owes the repair, and says so.
        REREAD_APPLIED: _reread_on_record(ctx, paper),
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
    "route", "reason", "impact_abs_delta_pooled", "candidates",
    #: DECISION B (A7): the two guess columns are ALWAYS present — a one-time, stable schema —
    #: with EMPTY values whenever no answer-tier rule fired (`best_guess_rules: []` included),
    #: because a consumer script must see one schema forever, not one that flaps with the data
    "best_guess_rule", "best_guess_entered")


def decorate_review_queue(review: list[dict[str, Any]],
                          best_guess_cells: Mapping[tuple[str, str], Mapping[str, Any]]) -> None:
    """Fill the two DECISION B queue columns, in place — empty strings when nothing fired.

    Decoration at the two call sites (here and `overrides._rewrite`) keeps `review_entry` and
    `state.py` out of the diff and the run and repool paths symmetrical. A queue row shows a
    rule and an entered summary only when the answer tier actually ENTERED values on THAT
    GROUP's cell (review F1: the sibling of a guessed cell was never guessed and stays
    blank) — a DECISION A admission is not a guess about a cell and leaves both columns empty.
    """
    for entry in review:
        cell = best_guess_cells.get((str(entry.get("dataset_id") or ""),
                                     str(entry.get("outcome_key") or ""))) or {}
        mine = (cell.get("best_guess_cell_guesses") or {}).get(
            str(entry.get("group") or "")) or {}
        entry["best_guess_rule"] = str(mine.get("rule") or "")
        entry["best_guess_entered"] = str(mine.get("entered") or "")
        # the crossed codes — and, for a fire whose row could not enter, the reason why —
        # ride the queue's JSON (never the CSV: A7's schema is two columns) so the question
        # card can print them; absent whenever nothing fired (fire-gating)
        if entry["best_guess_entered"]:
            if mine.get("stepped_past"):
                entry["best_guess_stepped_past"] = list(mine["stepped_past"])
            if mine.get("blocked_by"):
                entry["best_guess_blocked_by"] = str(mine["blocked_by"])

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
       `exclusions`, and the two findings are kept apart — UNLESS a person's own answer is on it
       (`human_override`). "The paper reports nothing" is an exclusion; "somebody answered this
       row and it still converts to nothing" is a question, and dropping it from the queue is how
       a reviewer's own answer made a row disappear: they were asked to confirm a number, the
       confirmation released both cells, the row stayed unbuildable, and it left the analysis with
       nothing anywhere to say it had;
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
                if r.route != "not_convertible" or HUMAN_OVERRIDE in r.flags} - cells_held
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


_statistic_values = statistic_values
_reported_values = reported_values


def _resolve(ctx: RunContext, paper: PaperRecord, study: StudyMap,
             candidates: Sequence[Candidate], verdicts: Sequence[Verdict],
             status: PaperStatus) -> list[EffectSizeRecord]:
    keys = ctx.outcome_keys()
    if ctx.resume and stage_done(ctx.out_dir, paper.sha256, "resolve"):
        payload = read_stage(ctx.out_dir, paper.sha256, "resolve")
        records = [EffectSizeRecord.model_validate(r) for r in payload["records"]]
        # a row for a cell that has been read again is a row over numbers this run has replaced,
        # so it is not a record of anything: dropped here, it makes the cell "verified with no
        # row" and the stage recomputes — over every verdict, as below. Off the stage files, for
        # the reason `_verify` reads it there: this stage owes the same repair after a resume that
        # died before it as it owes inside the run that re-read the cell.
        stale = _stale_cells(ctx, paper, "resolve")
        records = [r for r in records if (r.dataset_id, r.outcome_key) not in stale]
        # the same rule again, at the last stage: not done while a cell that now has verdicts has
        # no record. This stage buys nothing — it is arithmetic over the verdicts — so it is
        # recomputed in full rather than merged, which cannot drift from a fresh run.
        missing = {(v.dataset_id, v.outcome_key) for v in verdicts} - {
            (r.dataset_id, r.outcome_key) for r in records}
        if not missing:
            if stale:
                # the re-read cell has no verdict either (its dataset is blocked now): there is
                # no row to rebuild, so the file is brought in line with what this stage returns
                # and the debt is settled rather than re-attempted on every resume.
                write_stage(ctx.out_dir, paper.sha256, "resolve",
                            {"records": [r.model_dump(mode="json") for r in records],
                             REREAD_APPLIED: _reread_on_record(ctx, paper)})
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
                            cluster_of=lambda d: d.cluster_id or paper.sha256,
                            # Rule A's premise, computed once per paper from the whole map so the
                            # run, the re-pool and the preview infer identically or not at all
                            house_spread=house_spread_type(study.datasets))

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
                {"records": [r.model_dump(mode="json") for r in records],
                 REREAD_APPLIED: _reread_on_record(ctx, paper)})
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
        study, file_id = _map(paper_ctx, paper, group, status,
                              remap_for=_needs_a_fresh_map(paper_ctx, paper))
        result.study = study
        status.status = "mapped"
        status.eligible = study.eligible
        # `StudyMap.needs_human` was written by the mapper and read by nothing outside the browser,
        # so every objection it raised — a wrong outcome key, a conflicting unit, a group mapping
        # two agents could not agree on, "no dataset in an eligible paper, twice" — died where it
        # was made. An objection that reaches no output is the same as no objection at all.
        for note in study.needs_human:
            status.warnings.append(f"map: {note}")
        # §C3: the reviewer's answer to the `include_paper` card, before the mapper's verdict is
        # acted on. `True` skips the exclusion and the paper is read like any other; `False` is a
        # person agreeing with the mapper, which changes nothing here and is recorded in the log.
        included = _answered_eligibility(paper_ctx, paper)
        if study.eligible is False and included is not True:
            status.status = "excluded"
            result.exclusions.append({
                "paper_id": group.sha256, "filename": status.filename, "stage": "map",
                "reason": "not_eligible", "quote": study.eligibility_rationale,
                "decider": "mapper", "detail": study.exclusion_reason})
            emit(ctx.progress, "map", label, "excluded",
                 cost_so_far=ctx.client.total_cost(), message=study.exclusion_reason)
            return result
        if study.eligible is False and included is True:
            status.eligible = True
            status.warnings.append(
                "the mapper called this paper ineligible and a reviewer included it: "
                f"{study.exclusion_reason}")
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
    #: `primary` BEFORE `one_row_per_paper` combined anything — the rows the best-guess line adds
    #: to (DECISION A). Aggregation happens once per analysis line, over that line's own rows, so
    #: a composite is never built half from the strict line and half from the other one. Equal to
    #: `primary` when the protocol does not aggregate.
    primary_pre_agg: list[EffectSizeRecord] = field(default_factory=list)


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
    split.primary_pre_agg = list(split.primary)
    if settings.one_row_per_paper and split.primary:
        aggregated: Aggregation = aggregate_one_row_per_paper(split.primary, settings)
        composites = [r for r in aggregated.rows if AGGREGATED_FLAG in r.flags]
        # the confidence filter again, because aggregation can DOWNGRADE: a composite built
        # from a doubly-unverified member arm comes back `needs_human`, and a held row is not
        # combined into the estimate — it is held for the human, same as before aggregation
        split.primary = [r for r in aggregated.rows if r.confidence in admitted]
        split.held = [*split.held,
                      *[r for r in aggregated.rows if r.confidence not in admitted]]
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
        #
        # What it re-applies is the review layer's own `apply_overrides_and_repool`, and that
        # function rewrites DERIVED artefacts only (`overrides._rewrite`: "No stage file is
        # touched"). So after this line `papers/<id>/resolve.json` still says what the model-driven
        # run resolved BEFORE any human touched it, and it is held that way deliberately:
        # `overrides._row_of` reads it as the one fixed baseline an answer's admissibility is judged
        # against, and an admissibility test that moved with the last re-pool would be no test.
        # The consequence is worth saying out loud, because a reviewer auditing a number will open
        # the per-paper file first and be misled: **what was pooled is in `results/`**
        # (`extraction_table_all.json`, `<outcome>/extraction_table.json`, `<outcome>/pooled.json`),
        # never in a stage file. Investigated in full after two agents read the same row off the two
        # files and reported different numbers — both were right, and `--resume` applies an override
        # exactly as the review layer does because it IS the review layer.
        #
        # The one real hazard in the ordering: `_write_outputs` above has already written a pooled
        # result from the stale stage files (k=1 where the log makes it k=2), and this call
        # overwrites it. A crash in that window leaves an under-pooled `results/` on disk — the
        # same window `has_log` already disables the best-guess tier for, below.
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
    #: filled per outcome by `write_outcome_outputs`, so the run-wide extraction table below
    #: shows the SAME best-guess decision the per-outcome one does rather than a second opinion
    best_guess_cells: dict[tuple[str, str], dict] = {}
    # DECISION B's inputs, and its run-path log gate (integration §B): when a log already
    # exists, this first pass runs the answer tier DISABLED — the immediately following
    # `apply_overrides_and_repool` evaluates it for real, strictly post-log-application, so a
    # guess is only ever computed AFTER the log is applied, or when no log exists. A crash in
    # the window leaves artefacts carrying NO guesses — the safe direction. The retirement
    # sets are the questions page's own helpers, computed once (G6).
    has_log = bool(read_overrides(out))
    datasets = {d.dataset_id: d for result in results if result.study
                for d in result.study.datasets}
    from ..review.questions import retirements_for_run

    retirements = {} if has_log else retirements_for_run(out, verdicts)
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
                                          all_rows=split.every,
                                          primary_pre_agg=split.primary_pre_agg,
                                          best_guess_cells=best_guess_cells,
                                          protocol=ctx.protocol,
                                          warnings=manifest.warnings,
                                          datasets=datasets, retirements=retirements,
                                          cell_rules_enabled=not has_log)
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
        candidates=candidates, primary=primary_rows, best_guess=best_guess_cells).items()})

    decorate_review_queue(review, best_guess_cells)
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
    crosscheck_failed = _forest_crosscheck_failures(out)
    return {"run_dir": str(out), "protocol_hash": protocol.hash(),
            "protocol_matches_manifest": not protocol_changed,
            "records": len(records), "outcomes": outcomes,
            "missing_outputs": sorted(missing), "missing_stages": stages_missing,
            "forest_crosscheck_failed": crosscheck_failed,
            "ok": (not missing and not stages_missing and not protocol_changed
                   and not crosscheck_failed)}


def _forest_crosscheck_failures(run_dir: Path) -> list[str]:
    """Every forest whose R cross-check failed, as `<outcome>/<line>` (DECISION F).

    A run whose picture and whose tables came out of two different pooled results is not a valid
    run, however complete its file list is — so `canopy validate` fails on it. The check reads the
    same `renderer` block the report printed under the figure, and a `pooled.json` that predates
    the renderer (or names no renderer, because nothing was drawn) simply has nothing to fail.
    """
    failures: list[str] = []
    for path in sorted((run_dir / "results").glob("*/pooled.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for name, line in (("renderer", "strict"), ("renderer_best_guess", "best guess")):
            block = payload.get(name)
            check = block.get("crosscheck") if isinstance(block, dict) else None
            if not isinstance(check, dict) or check.get("ok") is not False:
                continue
            named = ", ".join(str(quantity) for quantity in (check.get("failed") or []))
            failures.append(f"{path.parent.name}/{line}" + (f": {named}" if named else ""))
    return failures
