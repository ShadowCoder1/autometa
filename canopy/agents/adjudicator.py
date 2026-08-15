"""Adjudicator (spec §3.3(4)) — the strongest model, once, on a cell nobody else could settle.

It runs only when the vote failed or a verifier refuted, and it is the one agent that sees
everything: the whole paper, every candidate, the consistency flags, the vote and every verifier
verdict. It still may not compute: it chooses (or reads) values that are printed in the paper, and
says which candidate each one came from. `needs_human` is a legitimate answer and the prompt says
so — a cell in the review queue costs a reviewer a minute, a confidently wrong number costs the
review its result.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.context import text_block
from ..models import (Adjudication, AdjudicatedGroup, Candidate, CheckFlag, DatasetSpec,
                      DispersionType, OutcomeDef, Protocol, VerifierVerdict)
from ..verify.grounding import ground_candidate
from . import render_prompt
from .verify_common import (SYSTEM, candidates_text, enum_schema, evidence_text, groups_prompt,
                            number, outcome_prompt, prompt_fingerprint, strings, whole,
                            whole_paper)

__all__ = ["adjudicate", "ADJUDICATE_SCHEMA", "PROMPT_VERSION", "PROMPT_FILES"]

PROMPT_FILES = ("adjudicator",)
PROMPT_VERSION = f"adjudicator/1@{prompt_fingerprint(PROMPT_FILES)}"

MAX_TOKENS = 8000
EFFORT = "xhigh"

#: 12 leaf properties
ADJUDICATE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["groups", "rationale", "needs_human", "notes"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["group", "n", "mean", "dispersion_value", "dispersion_type", "unit",
                             "quote", "page", "locator", "chosen_candidate_ids", "reason",
                             "needs_human"],
                "properties": {
                    "group": enum_schema(["A", "B", "unknown"]),
                    "n": {"type": ["integer", "null"]},
                    "mean": {"type": ["number", "null"]},
                    "dispersion_value": {"type": ["number", "null"]},
                    "dispersion_type": enum_schema([d.value for d in DispersionType]),
                    "unit": {"type": "string"},
                    "quote": {"type": "string"},
                    "page": {"type": ["integer", "null"]},
                    "locator": {"type": "string"},
                    "chosen_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "needs_human": {"type": "boolean"},
                },
            },
        },
        "rationale": {"type": "string"},
        "needs_human": {"type": "boolean"},
        "notes": {"type": "string"},
    },
}


def _outcome_key(outcome: Any) -> str:
    return outcome if isinstance(outcome, str) else getattr(outcome, "key", "") or \
        getattr(outcome, "outcome_key", "")


def _group_row(raw: Mapping[str, Any], known_ids: set[str],
               unknown_ids: list[str]) -> AdjudicatedGroup | None:
    key = raw.get("group")
    if key not in ("A", "B"):
        return None
    chosen = strings(raw.get("chosen_candidate_ids"))
    unknown_ids.extend(cid for cid in chosen if cid not in known_ids)
    kind = raw.get("dispersion_type")
    return AdjudicatedGroup(
        group=key, n=whole(raw.get("n")), mean=number(raw.get("mean")),
        dispersion_value=number(raw.get("dispersion_value")),
        dispersion_type=DispersionType(kind if kind in {d.value for d in DispersionType}
                                       else "UNKNOWN"),
        unit=(raw.get("unit") or "").strip(),
        quote=(raw.get("quote") or "").strip(), page=whole(raw.get("page")),
        locator=(raw.get("locator") or "").strip(),
        chosen_candidate_ids=[cid for cid in chosen if cid in known_ids],
        reason=(raw.get("reason") or "").strip(),
        needs_human=bool(raw.get("needs_human")))


def ground_adjudication(adjudication: Adjudication, candidates: Sequence[Candidate],
                        paper: PaperRecord) -> Adjudication:
    """Give every adjudicated value a checked provenance — in place, and returned for chaining.

    A value equal to one of the candidates inherits that candidate's page, quote and grounding: the
    adjudicator picked a reading somebody already evidenced. A value nobody proposed has to stand
    on its own quote, which is grounded exactly as an extractor's would be (named page, then ±1,
    then the whole document). One that is neither a candidate's value nor findable in the paper
    sets `needs_human` — pure code, no second opinion needed to know an unquotable number is not
    usable.
    """
    for group in adjudication.groups:
        if group.mean is None:
            continue
        twin = _matching_candidate(group, candidates)
        if twin is not None:
            group.grounded = twin.grounded
            group.grounding_similarity = twin.grounding_similarity
            group.quote = group.quote or twin.quote
            group.page = group.page or twin.page
            group.locator = group.locator or twin.locator
            if twin.candidate_id not in group.chosen_candidate_ids:
                group.chosen_candidate_ids = [*group.chosen_candidate_ids, twin.candidate_id]
            if twin.grounded is False:
                group.needs_human = True
                group.reason = _note(group.reason, "the candidate this value came from is not "
                                                   "grounded in the paper")
            continue
        if not group.quote.strip():
            group.needs_human = True
            group.grounded = False
            group.reason = _note(group.reason, "this value matches no candidate and the "
                                               "adjudicator quoted nothing for it")
            continue
        probe = ground_candidate(Candidate(kind="group_stats", group=group.group, mean=group.mean,
                                           dispersion_value=group.dispersion_value,
                                           quote=group.quote, page=group.page), paper)
        group.grounded = probe.grounded
        group.grounding_similarity = probe.grounding_similarity
        group.page = probe.page
        if not probe.grounded:
            group.needs_human = True
            group.reason = _note(group.reason,
                                 f"this value matches no candidate and its quote is not in the "
                                 f"paper (best similarity {probe.grounding_similarity})")
    adjudication.needs_human = adjudication.needs_human or any(g.needs_human
                                                               for g in adjudication.groups)
    return adjudication


def _matching_candidate(group: AdjudicatedGroup,
                        candidates: Sequence[Candidate]) -> Candidate | None:
    """The candidate this ruling actually endorses: one it named, else one with the same value."""
    named = [c for c in candidates if c.candidate_id in set(group.chosen_candidate_ids)
             and c.group == group.group]
    same = [c for c in named if c.mean is not None and group.mean is not None
            and math.isclose(c.mean, group.mean, rel_tol=1e-9, abs_tol=1e-12)]
    if same:
        return same[0]
    for cand in candidates:
        if (cand.group == group.group and cand.status == "found" and cand.mean is not None
                and group.mean is not None
                and math.isclose(cand.mean, group.mean, rel_tol=1e-9, abs_tol=1e-12)):
            return cand
    return None


def _note(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def adjudicate(client: LLMClient, paper: PaperRecord, dataset: DatasetSpec, outcome: Any,
               candidates: Sequence[Candidate], verdicts: Iterable[VerifierVerdict] = (),
               flags: Iterable[CheckFlag] = (), model: str = MODELS["adjudicator"],
               effort: str = EFFORT, *, protocol: Protocol | None = None,
               outcome_def: OutcomeDef | None = None, votes: Any = None,
               pdf_file_id: str | None = None) -> Adjudication:
    """Settle one (dataset × outcome) cell. Call only when the vote failed or a verifier refuted."""
    outcome_key = _outcome_key(outcome)
    document, betas = whole_paper(client, paper, pdf_file_id)
    content = [document, text_block(render_prompt(
        "adjudicator",
        OUTCOME=outcome_prompt(outcome_key, protocol=protocol, dataset=dataset,
                               outcome=outcome_def if outcome_def is not None
                               else (outcome if isinstance(outcome, OutcomeDef) else None)),
        GROUPS=groups_prompt(dataset),
        CANDIDATES=candidates_text(candidates),
        EVIDENCE=evidence_text(votes, verdicts, flags)))]

    result = client.structured(
        model=model, system=SYSTEM, schema=ADJUDICATE_SCHEMA, effort=effort,
        max_tokens=MAX_TOKENS, betas=betas, prompt_version=PROMPT_VERSION,
        cell_key=f"adjudicate:{dataset.dataset_id}:{outcome_key}",
        messages=[{"role": "user", "content": content}])

    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    known_ids = {c.candidate_id for c in candidates}
    unknown_ids: list[str] = []
    rows = [row for row in (parsed.get("groups") or []) if isinstance(row, dict)]
    groups = [g for g in (_group_row(row, known_ids, unknown_ids) for row in rows) if g is not None]

    notes = (parsed.get("notes") or "").strip()
    if unknown_ids:
        notes = (f"{notes}; " if notes else "") + \
                f"dropped candidate ids that are not in this cell: {sorted(set(unknown_ids))}"
    answered = {g.group for g in groups}
    for missing in sorted({"A", "B"} - answered):
        groups.append(AdjudicatedGroup(group=missing, needs_human=True,
                                       reason="the adjudicator returned no answer for this group"))
        notes = (f"{notes}; " if notes else "") + f"no ruling for group {missing}"

    adjudication = Adjudication(
        dataset_id=dataset.dataset_id, outcome_key=outcome_key, groups=groups,
        rationale=(parsed.get("rationale") or "").strip(),
        needs_human=bool(parsed.get("needs_human")) or any(g.needs_human for g in groups),
        chosen_candidate_ids=sorted({cid for g in groups for cid in g.chosen_candidate_ids}),
        model=model, prompt_version=PROMPT_VERSION, llm_call_id=result.call_id, notes=notes)
    ground_adjudication(adjudication, candidates, paper)
    if not adjudication.rationale:
        adjudication.needs_human = True
        adjudication.notes = (f"{adjudication.notes}; " if adjudication.notes else "") + \
            "no rationale was given, so the ruling cannot be reviewed"
    return adjudication
