"""Orientation (spec §3.3(5), amendment G) — what does a LARGER raw value on this measure mean?

This is the question that decides the SIGN of every effect size built from a measure, and getting
it wrong inverts a result rather than blurring it. So it is answered once per (outcome, measure),
by two agents that must agree independently, from the measure's own definition and the paper's
words — never from what the measure is usually called. Code then applies `orient()`; nothing here
computes anything.

The paper's own statement of which group came out higher travels with the verdict
(`direction_stated_in_text`), so `canopy.verify.checks.sign_check` can compare it with the sign the
extracted numbers imply and flag `sign_mismatch` when they disagree.
"""
from __future__ import annotations

from typing import Any, Sequence, get_args

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.context import text_block
from ..models import (DatasetSpec, Direction, OrientationRun, OrientationVerdict, OutcomeDef,
                      OutcomeSources, Protocol, RawValueSemantics)
from . import render_prompt
from .verify_common import (SYSTEM, clip, enum_schema, enum_value, groups_prompt, measure_prompt,
                            outcome_prompt, prompt_fingerprint, whole_paper)

__all__ = ["orientation", "orientation_run", "combine_orientation", "ORIENTATION_SCHEMA",
           "PROMPT_VERSION", "PROMPT_FILES", "DEFAULT_MODELS"]

PROMPT_FILES = ("orientation",)
PROMPT_VERSION = f"orientation/1@{prompt_fingerprint(PROMPT_FILES)}"

MAX_TOKENS = 4000
#: two agents, and they must differ — one strong reader and one that fails differently
DEFAULT_MODELS: tuple[str, str] = (MODELS["primary"], MODELS["secondary"])
_EFFORT = {MODELS["primary"]: "high"}
_DEFAULT_EFFORT = "medium"

_HIGHER = {"higher": True, "lower": False, "unknown": None}

#: 5 leaf properties — one measure, one question
ORIENTATION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["raw_value_semantics", "higher_is_better", "direction_stated_in_text", "quotes",
                 "reason"],
    "properties": {
        "raw_value_semantics": enum_schema(get_args(RawValueSemantics)),
        "higher_is_better": enum_schema(list(_HIGHER)),
        "direction_stated_in_text": enum_schema(get_args(Direction)),
        "quotes": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}


def _quotes(raw: Any) -> list[str]:
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        text = clip(item, 400) if isinstance(item, str) else ""
        if text and text not in out:
            out.append(text)
    return out


def orientation_run(client: LLMClient, paper: PaperRecord, dataset: DatasetSpec | None,
                    outcome_sources: OutcomeSources, model: str, *,
                    protocol: Protocol | None = None, outcome: OutcomeDef | None = None,
                    pdf_file_id: str | None = None, effort: str | None = None) -> OrientationRun:
    """One agent's answer about one measure's direction."""
    document, betas = whole_paper(client, paper, pdf_file_id)
    content = [document, text_block(render_prompt(
        "orientation",
        OUTCOME=outcome_prompt(outcome_sources.outcome_key, protocol=protocol, dataset=None,
                               outcome=outcome),
        MEASURE=measure_prompt(outcome_sources),
        GROUPS=groups_prompt(dataset)))]

    result = client.structured(
        model=model, system=SYSTEM, schema=ORIENTATION_SCHEMA,
        effort=effort or _EFFORT.get(model, _DEFAULT_EFFORT), max_tokens=MAX_TOKENS, betas=betas,
        prompt_version=PROMPT_VERSION,
        cell_key=f"orientation:{outcome_sources.outcome_key}:"
                 f"{outcome_sources.measure_name or 'measure'}",
        messages=[{"role": "user", "content": content}])

    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    return OrientationRun(
        higher_is_better=_HIGHER[enum_value(parsed.get("higher_is_better"), list(_HIGHER),
                                             "unknown")],
        raw_value_semantics=enum_value(parsed.get("raw_value_semantics"),
                                        get_args(RawValueSemantics), "unknown"),
        direction_stated_in_text=enum_value(parsed.get("direction_stated_in_text"),
                                             get_args(Direction), "unknown"),
        quotes=_quotes(parsed.get("quotes")),
        reason=(parsed.get("reason") or "").strip(),
        model=model, prompt_version=PROMPT_VERSION, llm_call_id=result.call_id)


def combine_orientation(runs: Sequence[OrientationRun], outcome_key: str,
                        measure_name: str = "") -> OrientationVerdict:
    """Two independent answers into one verdict: they must agree, or a human decides."""
    verdict = OrientationVerdict(outcome_key=outcome_key, measure_name=measure_name,
                                 runs=list(runs),
                                 llm_call_ids=[r.llm_call_id for r in runs if r.llm_call_id])
    quotes: list[str] = []
    for run in runs:
        quotes += [q for q in run.quotes if q not in quotes]
    verdict.quotes = quotes
    verdict.reason = " | ".join(f"{r.model}: {r.reason}" for r in runs if r.reason)

    answers = [r.higher_is_better for r in runs]
    notes: list[str] = []
    if len(runs) < 2:
        notes.append("only one agent ruled on this measure; the direction of a measure needs two "
                     "independent answers")
    elif len(set(answers)) == 1 and answers[0] is not None:
        verdict.agreed = True
        verdict.needs_human = False
        verdict.higher_is_better = answers[0]
    elif all(answer is None for answer in answers):
        notes.append("neither agent could tell what a larger raw value means")
    else:
        notes.append(f"the agents disagree about the direction of this measure: "
                     + ", ".join(f"{r.model}={r.higher_is_better}" for r in runs))

    semantics = {r.raw_value_semantics for r in runs}
    verdict.raw_value_semantics = semantics.pop() if len(semantics) == 1 else "unknown"
    directions = {r.direction_stated_in_text for r in runs if r.direction_stated_in_text != "unknown"}
    if len(directions) == 1:
        verdict.direction_stated_in_text = directions.pop()
    elif directions:
        notes.append(f"the agents read the paper's stated direction differently ({sorted(directions)})")
    verdict.notes = "; ".join(notes)
    return verdict


def orientation(client: LLMClient, paper: PaperRecord, dataset: DatasetSpec | None,
                outcome_sources: OutcomeSources,
                models: str | Sequence[str] = DEFAULT_MODELS, *,
                protocol: Protocol | None = None, outcome: OutcomeDef | None = None,
                pdf_file_id: str | None = None) -> OrientationVerdict:
    """Decide the direction of ONE measure with two independent agents (spec §3.3(5)).

    Runs once per (outcome, measure), not once per value: the answer is a property of the measure.
    Disagreement, or two "unknown"s, leaves `needs_human` set and `higher_is_better` None, and
    Task 9 refuses to sign an effect size without it.
    """
    names = [models] if isinstance(models, str) else list(models)
    if not names:
        raise ValueError("orientation needs at least one model")
    runs = [orientation_run(client, paper, dataset, outcome_sources, name, protocol=protocol,
                            outcome=outcome, pdf_file_id=pdf_file_id) for name in names]
    return combine_orientation(runs, outcome_sources.outcome_key, outcome_sources.measure_name)
