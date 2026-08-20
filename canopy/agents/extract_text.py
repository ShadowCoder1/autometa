"""Text/table extractor — reads one outcome for one dataset's two groups off the pages the mapper
named, and returns `Candidate`s: raw transcribed values with provenance.

It never computes anything. Two independent variants read the same pages so that Task 8 has two
voters that fail differently:

* `table_first`     (Opus, effort high)    tables cell by cell, then the running text;
* `narrative_first` (Sonnet, effort medium) the sentences first, then the tables.

Both answer the same envelope, so their answers are comparable value by value. A group the paper
does not report on those pages comes back as `not_on_these_pages` with null numbers — an honest
gap is worth more than a plausible guess, and Task 8 can then route the cell to the digitizer.

Every candidate goes through `canopy.verify.grounding.ground_candidate`: its quote must be on the
page it claims (or ±1, or somewhere in the document), and a value said to come from a table row
must really be in that row.
"""
from __future__ import annotations

from typing import Any, Literal, Sequence, get_args

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.context import text_block
from ..models import (AnalysisMetric, Candidate, CandidateStatus, DatasetSpec, DispersionType,
                      ErrorBarScope, Protocol, RawValueSemantics, Source, SourceKind)
from . import render_prompt
from .extract_common import (SYSTEM, TEXT_SOURCE_KINDS, candidate_id, context_blocks,
                             enum_schema, ground_candidate, group_label_check, groups_text, note,
                             outcome_text, pages_of, prompt_fingerprint, sources_text,
                             table_ids_of, usable_sources)

__all__ = ["extract_group_stats", "EXTRACT_TEXT_SCHEMA", "PROMPT_VERSION", "VARIANTS",
           "Variant"]

Variant = Literal["table_first", "narrative_first"]

#: prompt file, model and effort per variant — two readers that differ in more than temperature
VARIANTS: dict[str, dict[str, str]] = {
    "table_first": {"prompt": "extract_table_first", "model": MODELS["primary"], "effort": "high"},
    "narrative_first": {"prompt": "extract_narrative_first", "model": MODELS["secondary"],
                        "effort": "medium"},
}
PROMPT_FILES = tuple(v["prompt"] for v in VARIANTS.values())
PROMPT_VERSION = f"extract_text/1@{prompt_fingerprint(PROMPT_FILES)}"

MAX_TOKENS = 8000
ROUTE = "text"

#: confidence levels a paper may state, as printed (a string: it is a label, not a number to use)
CI_LEVELS = ("95", "90", "99", "unknown")
#: the dispersion type a stated level implies, when the extractor left the type open. A level with
#: no member (99%) keeps its bounds in `ci_low`/`ci_high` and says so in the notes rather than
#: growing `DispersionType`, which Task 8/9 code is reading concurrently.
CI_DISPERSION = {"95": DispersionType.CI95, "90": DispersionType.CI90}


# ----------------------------------------------------------------------------- schema
#: one row per group; 20 leaf properties, well under the structured-output grammar limit
EXTRACT_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["groups", "notes"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["group", "group_label_as_written", "status", "page", "kind", "quote",
                             "row_header", "col_header", "value_as_written", "mean",
                             "dispersion_value", "dispersion_type", "ci_low", "ci_high",
                             "ci_level", "unit", "n", "n_quote", "raw_value_semantics",
                             "analysis_metric", "error_bar_scope", "notes"],
                "properties": {
                    "group": enum_schema(["A", "B", "unknown"]),
                    "group_label_as_written": {"type": "string"},
                    "status": enum_schema(get_args(CandidateStatus)),
                    "page": {"type": ["integer", "null"]},
                    "kind": enum_schema([k.value for k in SourceKind]),
                    "quote": {"type": "string"},
                    "row_header": {"type": "string"},
                    "col_header": {"type": "string"},
                    "value_as_written": {"type": "string"},
                    "mean": {"type": ["number", "null"]},
                    "dispersion_value": {"type": ["number", "null"]},
                    "dispersion_type": enum_schema([d.value for d in DispersionType]),
                    "ci_low": {"type": ["number", "null"]},
                    "ci_high": {"type": ["number", "null"]},
                    "ci_level": enum_schema(CI_LEVELS),
                    "unit": {"type": "string"},
                    "n": {"type": ["integer", "null"]},
                    "n_quote": {"type": "string"},
                    "raw_value_semantics": enum_schema(get_args(RawValueSemantics)),
                    "analysis_metric": enum_schema(get_args(AnalysisMetric)),
                    "error_bar_scope": enum_schema(get_args(ErrorBarScope)),
                    "notes": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}


# ----------------------------------------------------------------------------- parsing
_STATUSES = set(get_args(CandidateStatus))
_SOURCE_KINDS = {k.value for k in SourceKind}
_DISPERSIONS = {d.value for d in DispersionType}


def _enum_value(raw: Any, allowed: set[str] | tuple[str, ...], fallback: str) -> str:
    value = (raw or "").strip() if isinstance(raw, str) else ""
    return value if value in allowed else fallback


def _whole(raw: Any) -> int | None:
    """An integer the model sent (a bool is an int in Python, and never an answer here)."""
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _number(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _hint_conflict(cand: Candidate, sources: Sequence[Source]) -> str:
    """The mapper's error-bar hint for this page against what the extractor actually read.

    For a text source the mapper's reading was never confirmed by a second agent
    (`error_bar_agreement` stays "unconfirmed" by design), so the extractor's own reading — made
    with the sentence in front of it — governs, and the disagreement travels as a note.
    """
    if cand.dispersion_type is DispersionType.UNKNOWN:
        return ""
    for source in sources:
        if source.page != cand.page:
            continue
        hint = source.error_bar_type
        if hint in (DispersionType.UNKNOWN, DispersionType.NONE) or hint is cand.dispersion_type:
            continue
        return (f"dispersion type disagreement: this extractor read "
                f"{cand.dispersion_type.value}, the map said {hint.value} for "
                f"{source.locator or f'page {source.page}'}")
    return ""


def _candidate(row: dict[str, Any], *, index: int, paper: PaperRecord, dataset: DatasetSpec,
               outcome_key: str, sources: Sequence[Source], extractor_id: str, model: str,
               call_id: str) -> Candidate:
    group_raw = _enum_value(row.get("group"), {"A", "B"}, "")
    group = group_raw or None
    status = _enum_value(row.get("status"), _STATUSES, "ambiguous")
    label = (row.get("group_label_as_written") or "").strip()
    page = _whole(row.get("page"))
    notes = (row.get("notes") or "").strip()

    values = {name: _number(row.get(name))
              for name in ("mean", "dispersion_value", "ci_low", "ci_high")}
    dispersion_type = DispersionType(_enum_value(row.get("dispersion_type"), _DISPERSIONS,
                                                 "UNKNOWN"))
    ci_level = _enum_value(row.get("ci_level"), CI_LEVELS, "unknown")
    n, n_quote = _whole(row.get("n")), (row.get("n_quote") or "").strip()
    unit = (row.get("unit") or "").strip()
    written = (row.get("value_as_written") or "").strip()

    if status != "found":                        # amendment E: a non-`found` row carries no value
        reported = {**values, "n": n, "n_quote": n_quote, "unit": unit,
                    "value_as_written": written,
                    "dispersion_type": (None if dispersion_type is DispersionType.UNKNOWN
                                        else dispersion_type.value)}
        dropped = {name: value for name, value in reported.items()
                   if value is not None and value != ""}
        if dropped:
            notes = note(notes, f"numbers dropped because status is {status}: {dropped}")
        values = dict.fromkeys(values)
        dispersion_type, ci_level = DispersionType.UNKNOWN, "unknown"
        n, n_quote, unit, written = None, "", "", ""
    elif (values["ci_low"] is not None or values["ci_high"] is not None) \
            and dispersion_type is DispersionType.UNKNOWN:
        implied = CI_DISPERSION.get(ci_level)    # a stated level names the interval, nothing more
        if implied is not None:
            dispersion_type = implied
            notes = note(notes, f"dispersion type read from the stated {ci_level}% interval")
        elif ci_level != "unknown":              # e.g. 99%: the bounds are kept, the type is not
            notes = note(notes, f"the paper states a {ci_level}% interval, which has no "
                                f"DispersionType member: the bounds are transcribed and "
                                f"dispersion_type stays UNKNOWN")

    cand = Candidate(
        candidate_id=candidate_id(dataset.dataset_id, outcome_key, extractor_id, index, group),
        paper_id=paper.sha256, dataset_id=dataset.dataset_id, outcome_key=outcome_key,
        kind="group_stats", group=group, status=status,
        source_kind=SourceKind(_enum_value(row.get("kind"), _SOURCE_KINDS, "unknown")),
        n=n, n_quote=n_quote,
        mean=values["mean"], dispersion_value=values["dispersion_value"],
        dispersion_type=dispersion_type,
        ci_low=values["ci_low"], ci_high=values["ci_high"],
        unit=unit, value_as_written=written,
        raw_value_semantics=_enum_value(row.get("raw_value_semantics"),
                                        get_args(RawValueSemantics), "unknown"),
        analysis_metric=_enum_value(row.get("analysis_metric"), get_args(AnalysisMetric),
                                    "unknown"),
        error_bar_scope=_enum_value(row.get("error_bar_scope"), get_args(ErrorBarScope), "unknown"),
        page=page, quote=(row.get("quote") or "").strip(),
        row_header=(row.get("row_header") or "").strip(),
        col_header=(row.get("col_header") or "").strip(),
        route=ROUTE, model=model, prompt_version=PROMPT_VERSION, llm_call_id=call_id,
        extractor_id=extractor_id, notes=notes)

    if not group_raw:
        cand.notes = note(cand.notes, f"extractor did not say which group this row is for "
                                      f"(label as written: {label!r})")
    for warning in (group_label_check(group, label, dataset), _hint_conflict(cand, sources)):
        if warning:
            cand.notes = note(cand.notes, warning)
    own_label = dataset.group_a.label if group == "A" else dataset.group_b.label
    if group and label and label != own_label:          # the paper's words, kept for the reviewer
        cand.notes = note(cand.notes, f"group label as written: {label!r}")

    ground_candidate(cand, paper)
    # after grounding, so a corrected page names the location it was actually read from
    cand.locator = next((s.locator for s in sources if s.page == cand.page and s.locator), "")
    return cand


def _missing_group(group: str, *, paper: PaperRecord, dataset: DatasetSpec, outcome_key: str,
                   extractor_id: str, model: str, call_id: str, index: int) -> Candidate:
    """A group the extractor said nothing about: an explicit empty answer, never a silent gap."""
    return Candidate(
        candidate_id=candidate_id(dataset.dataset_id, outcome_key, extractor_id, index, group),
        paper_id=paper.sha256, dataset_id=dataset.dataset_id, outcome_key=outcome_key,
        kind="group_stats", group=group, status="not_on_these_pages", route=ROUTE, model=model,
        prompt_version=PROMPT_VERSION, llm_call_id=call_id, extractor_id=extractor_id,
        notes="the extractor returned no row for this group")


# ----------------------------------------------------------------------------- the agent
def extract_group_stats(client: LLMClient, paper: PaperRecord, protocol: Protocol,
                        dataset: DatasetSpec, outcome_key: str, sources: Sequence[Source], *,
                        variant: Variant = "table_first",
                        reviewer_hint: str = "",
                        model: str | None = None) -> list[Candidate]:
    """Transcribe one outcome for both groups from the text/table locations in `sources`.

    Returns one `Candidate` per group (`kind="group_stats"`), grounded. An empty list means there
    was nothing for this extractor to read — no text or table source inside the document — which is
    different from having read the pages and found nothing.

    `reviewer_hint` is a human's `re_extract` answer for this cell, added to the locations rather
    than replacing them. It changes the prompt, so a reading cached without it is never reused for
    the hinted re-read — which is the whole point of buying one.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (have {sorted(VARIANTS)})")
    settings = VARIANTS[variant]
    model = model or settings["model"]
    extractor_id = f"text:{variant}:{model}"

    usable = usable_sources(sources, TEXT_SOURCE_KINDS, paper)
    if not usable:
        return []
    pages = pages_of(usable)
    table_ids = table_ids_of(usable, paper)
    # only a source that really points at a table earns a page raster; a table that merely sits on
    # a page a *text* source names contributes its cells, not an image (amendment E)
    image_pages = [s.page for s in usable if s.kind is SourceKind.table or s.table_id]

    content = context_blocks(paper, pages, image_pages=image_pages, table_ids=table_ids)
    content.append(text_block(render_prompt(
        settings["prompt"],
        OUTCOME=outcome_text(protocol, outcome_key, dataset),
        GROUPS=groups_text(dataset),
        LOCATIONS=sources_text(usable, reviewer_hint=reviewer_hint))))

    result = client.structured(
        model=model, system=SYSTEM, schema=EXTRACT_TEXT_SCHEMA, effort=settings["effort"],
        max_tokens=MAX_TOKENS, prompt_version=PROMPT_VERSION,
        cell_key=f"extract-text:{dataset.dataset_id}:{outcome_key}:{variant}",
        messages=[{"role": "user", "content": content}])

    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    rows = parsed.get("groups") or []
    candidates = [_candidate(row, index=index, paper=paper, dataset=dataset,
                             outcome_key=outcome_key, sources=usable, extractor_id=extractor_id,
                             model=model, call_id=result.call_id)
                  for index, row in enumerate(rows) if isinstance(row, dict)]

    answered = {c.group for c in candidates}
    for group in ("A", "B"):
        if group not in answered:
            candidates.append(_missing_group(group, paper=paper, dataset=dataset,
                                             outcome_key=outcome_key, extractor_id=extractor_id,
                                             model=model, call_id=result.call_id,
                                             index=len(candidates)))
    overall = (parsed.get("notes") or "").strip()
    if overall:
        for cand in candidates:
            cand.notes = note(cand.notes, f"extractor notes: {overall}")
    return candidates
