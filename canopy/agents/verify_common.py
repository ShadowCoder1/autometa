"""Shared machinery for the three verification agents (Task 8).

A verification agent differs from an extractor in what it is given and what it may conclude: it
sees the WHOLE paper (a `document` block, by Files-API id when one exists) rather than the pages
someone else chose, and it is asked to disagree. Nothing here computes a statistic; these are
prose renderers, the document block, and the bookkeeping that keeps a whole-paper call's budget
reservation honest.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.context import FILES_API_BETA
from ..llm.costs import register_file_pages
from ..models import (Candidate, CheckFlag, DatasetSpec, OutcomeDef, OutcomeSources, Protocol,
                      VerifierVerdict)
from .extract_common import SYSTEM, clip, enum_schema, groups_text, outcome_text, prompt_fingerprint

__all__ = ["SYSTEM", "clip", "enum_schema", "prompt_fingerprint", "outcome_prompt", "groups_prompt",
           "measure_prompt", "candidate_text", "candidates_text", "evidence_text", "whole_paper",
           "MAX_REOPENS"]

#: how often one cell may be re-opened after a refutation before it goes to a human (amendment G)
MAX_REOPENS = 2

_QUOTE_CHARS = 320


# ----------------------------------------------------------------------------- prose
def outcome_prompt(outcome_key: str, *, protocol: Protocol | None = None,
                   dataset: DatasetSpec | None = None,
                   outcome: OutcomeDef | None = None) -> str:
    """The outcome under discussion, from whatever the caller could supply."""
    if protocol is not None:
        try:
            return outcome_text(protocol, outcome_key, dataset)
        except KeyError:                                  # the protocol does not define this key
            pass
    lines = [f"OUTCOME KEY: {outcome_key}"]
    if outcome is not None:
        lines += [f"NAME: {outcome.label}", f"DEFINITION: {outcome.definition.strip()}"]
        if outcome.measurement_window:
            lines.append(f"MEASUREMENT WINDOW: {outcome.measurement_window.strip()}")
        if outcome.units_hint:
            lines.append(f"UNITS HINT: {outcome.units_hint.strip()}")
    found = None
    if dataset is not None:
        found = next((o for o in dataset.outcomes if o.outcome_key == outcome_key), None)
    if found is not None:
        lines += measure_prompt(found).splitlines()
    return "\n".join(lines)


def measure_prompt(outcome: OutcomeSources | None) -> str:
    """What the mapper established about the measure this paper actually reports."""
    if outcome is None:
        return "(the map recorded nothing about how this paper measures the outcome)"
    lines = [f"THE MEASURE THIS PAPER REPORTS: {outcome.measure_name or '(not recorded)'}"]
    if outcome.units:
        lines.append(f"UNITS THIS PAPER PRINTS: {outcome.units}")
    if outcome.operationalization:
        lines.append(f"HOW THIS PAPER MEASURED IT: {clip(outcome.operationalization, 400)}")
    if outcome.analysis_metric != "unknown":
        lines.append(f"WHAT THE NUMBER IS RELATIVE TO: {outcome.analysis_metric}")
    for source in outcome.sources[:6]:
        lines.append(f"LOCATION: page {source.page} | {source.kind.value}"
                     f"{' | ' + clip(source.locator, 120) if source.locator else ''}")
    return "\n".join(lines)


def groups_prompt(dataset: DatasetSpec | None) -> str:
    if dataset is None:
        return "(the two groups were not recorded on this call)"
    return groups_text(dataset)


def candidate_text(cand: Candidate) -> str:
    """One candidate as prose — everything a reader needs to attack it, and its id."""
    lines = [f"CANDIDATE ID: {cand.candidate_id}",
             f"EXTRACTED BY: {cand.extractor_id or cand.model or 'unknown reader'}",
             f"STATUS: {cand.status}"]
    if cand.group:
        lines.append(f"GROUP: {cand.group}")
    if cand.kind == "group_stats":
        lines.append(f"MEAN: {cand.mean}")
        lines.append(f"DISPERSION: {cand.dispersion_value} "
                     f"({cand.dispersion_type.value}, scope {cand.error_bar_scope})")
        if cand.ci_low is not None or cand.ci_high is not None:
            lines.append(f"INTERVAL: [{cand.ci_low}, {cand.ci_high}]")
        if cand.points:
            lines.append(f"INDIVIDUAL POINTS READ: {len(cand.points)}")
        lines.append(f"N: {cand.n}")
        lines.append(f"UNIT: {cand.unit or '(none recorded)'}")
        lines.append(f"WHAT THE NUMBER IS RELATIVE TO: {cand.analysis_metric}")
    else:
        lines.append(f"STATISTIC: {cand.stat_type} = {cand.stat_value} "
                     f"(df {cand.df}, df1 {cand.df1}, df2 {cand.df2}), design {cand.design}")
        lines.append(f"P: {cand.p_kind} {cand.p_value}")
        if cand.reported_value is not None:
            lines.append(f"REPORTED EFFECT SIZE: {cand.reported_value} "
                         f"({cand.reported_scale}, standardised by {cand.standardizer})")
        lines.append(f"DIRECTION THE PAPER STATES: {cand.direction}")
    lines.append(f"SOURCE: {cand.source_kind.value if cand.source_kind else 'unknown'} on page "
                 f"{cand.page}{' — ' + clip(cand.locator, 160) if cand.locator else ''}")
    if cand.row_header or cand.col_header:
        lines.append(f"TABLE CELL: row {cand.row_header!r}, column {cand.col_header!r}")
    if cand.value_as_written:
        lines.append(f"AS WRITTEN: {clip(cand.value_as_written, 200)}")
    if cand.quote:
        lines.append(f"QUOTE: {clip(cand.quote, _QUOTE_CHARS)}")
    lines.append(f"QUOTE GROUNDED IN THE PAGE TEXT: {cand.grounded} "
                 f"(similarity {cand.grounding_similarity})")
    if cand.sigma is not None:
        lines.append(f"DIGITISATION UNCERTAINTY: ±{cand.sigma}")
    if cand.notes:
        lines.append(f"READER NOTES: {clip(cand.notes, 400)}")
    return "\n".join(lines)


def candidates_text(candidates: Sequence[Candidate]) -> str:
    if not candidates:
        return "(no candidates)"
    return "\n\n".join(f"--- candidate {i + 1} ---\n{candidate_text(c)}"
                       for i, c in enumerate(candidates))


def evidence_text(votes: Any = None, verdicts: Iterable[VerifierVerdict] = (),
                  flags: Iterable[CheckFlag] = ()) -> str:
    """The vote, the consistency flags and every verifier verdict, as prose for the adjudicator."""
    lines: list[str] = []
    votes_map = votes if isinstance(votes, dict) else ({} if votes is None else {"": votes})
    for group, result in sorted(votes_map.items()):
        head = f"VOTE for group {group}" if group else "VOTE"
        lines.append(f"{head}: {result.agreement} — mean {result.mean}, dispersion "
                     f"{result.dispersion_value} ({result.dispersion_type.value}), n {result.n} "
                     f"[method {result.method}, tolerance {result.tolerance}]")
        for route in result.routes:
            lines.append(f"    route {route.route_key}: {route.value} "
                         f"from {route.candidate_ids}"
                         f"{'' if route.consistent else ' (disagrees with itself)'}")
        for note in result.notes:
            lines.append(f"    note: {note}")
    flag_list = list(flags)
    if flag_list:
        lines.append("CONSISTENCY FLAGS:")
        lines += [f"    [{f.severity}] {f.code}: {f.message} {f.candidate_ids}" for f in flag_list]
    verdict_list = list(verdicts)
    if verdict_list:
        lines.append("VERIFIER VERDICTS:")
        for verdict in verdict_list:
            lines.append(f"    {verdict.candidate_id or '(unnamed candidate)'} — "
                         f"{verdict.verdict} by {verdict.model}: {clip(verdict.reason, 400)}")
            if verdict.better_source:
                lines.append(f"        better source: {clip(verdict.better_source, 200)}")
            alt = {k: v for k, v in {"mean": verdict.alt_mean, "n": verdict.alt_n,
                                     "dispersion": verdict.alt_dispersion_value}.items()
                   if v is not None}
            if alt:
                lines.append(f"        alternative it proposes: {json.dumps(alt)}")
    return "\n".join(lines) or "(no evidence recorded)"


# ----------------------------------------------------------------------------- context
def whole_paper(client: LLMClient, paper: PaperRecord,
                pdf_file_id: str | None = None) -> tuple[dict[str, Any], list[str] | None]:
    """The `document` block for a whole-paper call, plus the betas it needs.

    A verifier that only sees the page a value came from cannot answer "is there a better source
    anywhere in the paper?", so every agent in this module gets the whole PDF. When it goes by
    Files-API id, the paper's page count is registered so the budget reservation reflects the real
    size instead of the flat guess in `canopy.llm.costs`.
    """
    if pdf_file_id:
        register_file_pages(pdf_file_id, paper.n_pages)
    block = client.pdf_block(pdf_path=None if pdf_file_id else paper.source_path,
                             file_id=pdf_file_id, cache=True)
    return block, ([FILES_API_BETA] if pdf_file_id else None)
