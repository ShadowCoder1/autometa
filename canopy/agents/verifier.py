"""Adversarial verifier (spec §3.3(3), amendment G) — a second model paid to disagree.

The verifier is given the WHOLE paper, one candidate, and a list of the ways a reading of that
kind goes wrong (wrong group, wrong time window, wrong panel, SE where SD was meant, baseline
where post was meant, wrong units, a subgroup). It must try each one and say which it tried. It
answers `confirmed` / `refuted` / `ambiguous`, may propose the value the paper actually prints
(never one it worked out), and — independently of its verdict — says whether some *other* place in
the paper reports the same quantity more directly.

Two rules hold this together:

* the verifier's model family must differ from the one that produced the candidate, or the second
  opinion is the first opinion with a different prompt (`verifier_model_for` picks one);
* a cell may be re-opened at most `MAX_REOPENS` times (amendment G). The count travels in the
  verdict, and the cache key, so a re-open is a fresh question rather than the cached answer.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.context import text_block
from ..models import Candidate, DatasetSpec, DispersionType, OutcomeDef, Protocol, VerifierVerdict
from ..verify.vote import model_family
from . import render_prompt
from .verify_common import (MAX_REOPENS, SYSTEM, candidate_text, enum_schema, groups_prompt,
                            outcome_prompt, prompt_fingerprint, whole_paper)

__all__ = ["verify_candidate", "verifier_model_for", "VERIFIER_SCHEMA", "PROMPT_VERSION",
           "PROMPT_FILES", "REFUTATION_TARGETS", "MAX_REOPENS"]

PROMPT_FILES = ("verifier",)
PROMPT_VERSION = f"verifier/1@{prompt_fingerprint(PROMPT_FILES)}"

MAX_TOKENS = 4000
EFFORT = "high"

#: the named ways a transcribed value goes wrong; the prompt lists them and the model echoes back
#: which ones it actually checked, so a bare "confirmed" cannot pass as work
REFUTATION_TARGETS: tuple[str, ...] = (
    "wrong_group", "wrong_time_window", "wrong_panel", "se_vs_sd", "baseline_vs_post", "units",
    "subgroup", "transcription")

_VERDICTS = ("confirmed", "refuted", "ambiguous")

#: 12 leaf properties — a verifier answers about one candidate, so the schema stays small
VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["verdict", "reason", "checked", "alt_mean", "alt_dispersion_value",
                 "alt_dispersion_type", "alt_n", "alt_page", "alt_quote", "better_source",
                 "better_source_page", "notes"],
    "properties": {
        "verdict": enum_schema(_VERDICTS),
        "reason": {"type": "string"},
        "checked": {"type": "array", "items": {"type": "string"}},
        "alt_mean": {"type": ["number", "null"]},
        "alt_dispersion_value": {"type": ["number", "null"]},
        "alt_dispersion_type": enum_schema([d.value for d in DispersionType]),
        "alt_n": {"type": ["integer", "null"]},
        "alt_page": {"type": ["integer", "null"]},
        "alt_quote": {"type": "string"},
        "better_source": {"type": "string"},
        "better_source_page": {"type": ["integer", "null"]},
        "notes": {"type": "string"},
    },
}


# ----------------------------------------------------------------------------- helpers
def verifier_model_for(cand: Candidate, preferred: str = MODELS["secondary"],
                       fallback: str = MODELS["primary"]) -> str:
    """A model that did not produce this candidate — `preferred` unless it is the same family."""
    if model_family(preferred) != model_family(cand.model):
        return preferred
    if model_family(fallback) != model_family(cand.model):
        return fallback
    for name in MODELS.values():
        if model_family(name) != model_family(cand.model):
            return name
    raise ValueError(f"no configured model differs from {cand.model!r}; add one to canopy.config")


def _clean_checked(raw: Any) -> list[str]:
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, str) or not item.strip():
            continue
        name = "_".join(item.strip().lower().split())
        if name not in out:
            out.append(name)
    return out


def _whole(raw: Any) -> int | None:
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _number(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _enum_value(raw: Any, allowed: Sequence[str], fallback: str) -> str:
    value = raw.strip() if isinstance(raw, str) else ""
    return value if value in set(allowed) else fallback


def _note(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


# ----------------------------------------------------------------------------- the agent
def verify_candidate(client: LLMClient, paper: PaperRecord, cand: Candidate,
                     context_blocks: Sequence[dict[str, Any]] = (),
                     model: str = MODELS["secondary"], *,
                     protocol: Protocol | None = None, dataset: DatasetSpec | None = None,
                     outcome: OutcomeDef | None = None, pdf_file_id: str | None = None,
                     reopen: int = 0, effort: str = EFFORT) -> VerifierVerdict:
    """Try to refute one candidate with a different model and the whole paper in front of it.

    `context_blocks` are the extra blocks this candidate needs — the crop and the overlay for a
    figure reading, a page image for a table. They are sent *after* the document so the cached
    document prefix is shared by every verifier call on this paper.
    """
    if model_family(model) == model_family(cand.model):
        raise ValueError(
            f"the verifier must not be the model that produced the candidate: {model!r} and "
            f"{cand.model!r} are the same family — use verifier_model_for(cand)")
    if reopen > MAX_REOPENS:
        raise ValueError(f"cell {cand.candidate_id!r} has already been re-opened {reopen} times "
                         f"(limit {MAX_REOPENS})")

    document, betas = whole_paper(client, paper, pdf_file_id)
    content: list[dict[str, Any]] = [document, *context_blocks]
    content.append(text_block(render_prompt(
        "verifier",
        OUTCOME=outcome_prompt(cand.outcome_key, protocol=protocol, dataset=dataset,
                               outcome=outcome),
        GROUPS=groups_prompt(dataset),
        CANDIDATE=candidate_text(cand))))

    result = client.structured(
        model=model, system=SYSTEM, schema=VERIFIER_SCHEMA, effort=effort, max_tokens=MAX_TOKENS,
        betas=betas, prompt_version=PROMPT_VERSION,
        cache_key_extra=f"reopen-{reopen}" if reopen else "",
        cell_key=f"verify:{cand.candidate_id}",
        messages=[{"role": "user", "content": content}])

    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    checked = _clean_checked(parsed.get("checked"))
    verdict = VerifierVerdict(
        candidate_id=cand.candidate_id,
        verdict=_enum_value(parsed.get("verdict"), _VERDICTS, "ambiguous"),
        reason=(parsed.get("reason") or "").strip(),
        alt_mean=_number(parsed.get("alt_mean")),
        alt_dispersion_value=_number(parsed.get("alt_dispersion_value")),
        alt_dispersion_type=DispersionType(
            _enum_value(parsed.get("alt_dispersion_type"), [d.value for d in DispersionType],
                        "UNKNOWN")),
        alt_n=_whole(parsed.get("alt_n")),
        alt_page=_whole(parsed.get("alt_page")),
        alt_quote=(parsed.get("alt_quote") or "").strip(),
        better_source=(parsed.get("better_source") or "").strip(),
        checked=checked, reopen=reopen, model=model, prompt_version=PROMPT_VERSION,
        llm_call_id=result.call_id, notes=(parsed.get("notes") or "").strip())

    page = _whole(parsed.get("better_source_page"))
    if verdict.better_source and page is not None:
        verdict.better_source = f"{verdict.better_source} (page {page})"
    missed = [target for target in REFUTATION_TARGETS if target not in checked]
    if missed:
        verdict.notes = _note(verdict.notes, f"did not report checking: {', '.join(missed)}")
    if verdict.verdict == "refuted" and not verdict.reason:
        verdict.verdict = "ambiguous"
        verdict.notes = _note(verdict.notes, "refutation without a reason was downgraded to "
                                             "ambiguous")
    if reopen:
        verdict.notes = _note(verdict.notes, f"re-open {reopen} of {MAX_REOPENS}")
    return verdict
