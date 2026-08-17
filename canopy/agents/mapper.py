"""Study mapper — the first LLM agent: one paper in, one `StudyMap` out.

The mapper reads a whole PDF and answers *where things are*: is the paper eligible, which
independent A-vs-B contrasts (datasets) it supports, how many participants were analysed in each
group, and — for every outcome — every location in the paper where a usable number lives. It never
reads a value off a figure and never computes anything; the extractors (which can only look where
the mapper points them) do that later.

Calls, all protocol-driven — nothing about any research domain is hard-coded here:

1. study map            (Opus, effort high)   whole PDF + protocol + the deterministic roster:
                                              eligibility, datasets, groups, Ns, roster decisions;
2. source map           (Opus, effort high)   same PDF (cached) + those datasets: for every outcome
                                              every location where a number lives (an empty answer
                                              is retried once, then flagged);
3. roster follow-up     (Sonnet, medium)      only if the study map left a figure/table undecided;
4. independent check    (Sonnet, medium)      eligibility, datasets, Ns, source locations, and an
                                              error-bar reading of EVERY roster figure/table;
5. adjudication         (Opus, xhigh)         when eligibility, an n, or a group mapping disagrees.

Passes 1 and 2 are one job split in two because the combined JSON schema exceeds the
structured-output grammar limit; splitting also lets pass 2 spend its whole answer on locations.

Agreement is decided by *coverage*, not by the absence of a conflict (amendment D): a figure or
table error bar counts only where the second agent independently determined the same type for that
roster id — `Source.error_bar_agreement` records `agreed` / `conflict` / `unconfirmed`, and only
`agreed` passes without a human. Everything the agents disagree about lands in
`StudyMap.disagreements`; anything no two agents ever agreed on lands in `StudyMap.needs_human`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, get_args

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.errors import LLMError
from ..llm.context import FILES_API_BETA, figure_blocks, text_block
from ..models import (AnalysisMetric, Citation, DatasetSpec, DispersionType, ErrorBarScope,
                      ExposureOrder, GroupSpec, OutcomeSources, Protocol, RosterDecision, Source,
                      SourceKind, StudyMap, XAxisKind)
from . import load_prompt, render_prompt

__all__ = ["map_study", "protocol_text", "roster_text", "roster_entries", "dataset_text",
           "PROMPT_VERSION", "MAPPER_SCHEMA", "MAPPER_SOURCES_SCHEMA",
           "MAPPER_CROSSCHECK_SCHEMA", "MAPPER_ADJUDICATE_SCHEMA", "MAPPER_ROSTER_SCHEMA"]

#: every prompt file this agent uses — their content fingerprints `PROMPT_VERSION`
PROMPT_FILES = ("mapper", "mapper_sources", "mapper_crosscheck", "mapper_adjudicate",
                "mapper_roster")


def prompt_fingerprint() -> str:
    """sha256 over the prompt files, so an edited prompt cannot keep an old version string."""
    blob = "\0".join(load_prompt(name) for name in PROMPT_FILES)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


#: recorded on every call and in the run manifest; the suffix moves whenever a prompt changes
PROMPT_VERSION = f"mapper/1@{prompt_fingerprint()}"

#: shared across the three calls so the cached `document` prefix (system + PDF) can be reused
SYSTEM = ("You are a component of Canopy, an automated meta-analysis pipeline. You work only from "
          "the documents you are given, you quote them verbatim as evidence, and you never invent, "
          "estimate or compute a number. When a paper does not report something, you say so.")

CAPTION_CHARS = 400            # roster captions are trimmed so the prompt stays small
TABLE_ROWS = 3                 # rows of a table shown in the roster
QUOTE_MATCH_CHARS = 40         # shortest quote that may identify a source on its own


# ----------------------------------------------------------------------------- schemas
def _enum(values: Any) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


_GROUP_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["label", "n", "n_evidence", "age_mean", "age_sd", "age_range", "notes"],
    "properties": {
        "label": {"type": "string"},
        "n": {"type": ["integer", "null"]},
        "n_evidence": {"type": "string"},
        "age_mean": {"type": ["number", "null"]},
        "age_sd": {"type": ["number", "null"]},
        "age_range": {"type": "string"},
        "notes": {"type": "string"},
    },
}

_SOURCE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["kind", "page", "locator", "quote", "figure_id", "table_id", "error_bar_type",
                 "error_bar_scope", "error_bar_evidence", "analysis_metric", "values_in_text",
                 "x_axis_kind", "notes"],
    "properties": {
        "kind": _enum([k.value for k in SourceKind]),
        "page": {"type": "integer"},
        "locator": {"type": "string"},
        "quote": {"type": "string"},
        "figure_id": {"type": "string"},                 # roster id, "" when not a figure
        "table_id": {"type": "string"},
        "error_bar_type": _enum([d.value for d in DispersionType]),
        "x_axis_kind": _enum(get_args(XAxisKind)),
        "error_bar_scope": _enum(get_args(ErrorBarScope)),
        "error_bar_evidence": {"type": "string"},
        "analysis_metric": _enum(get_args(AnalysisMetric)),
        "values_in_text": {"type": "string"},
        "notes": {"type": "string"},
    },
}

_OUTCOME_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["outcome_key", "measure_name", "units", "higher_is_better",
                 "higher_is_better_evidence", "operationalization", "analysis_metric", "sources"],
    "properties": {
        "outcome_key": {"type": "string"},
        "measure_name": {"type": "string"},
        "units": {"type": "string"},
        "higher_is_better": _enum(["higher_is_better", "lower_is_better", "unknown"]),
        "higher_is_better_evidence": {"type": "string"},
        "operationalization": {"type": "string"},
        "analysis_metric": _enum(get_args(AnalysisMetric)),
        "sources": {"type": "array", "items": _SOURCE_SCHEMA},
    },
}

_ROSTER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "relevant", "reason", "outcome_keys"],
    "properties": {
        "id": {"type": "string"},
        "relevant": {"type": "boolean"},
        "reason": {"type": "string"},
        "outcome_keys": {"type": "array", "items": {"type": "string"}},
    },
}

MAPPER_SOURCES_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["datasets", "notes"],
    "properties": {
        "datasets": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["dataset_index", "outcomes"],
                "properties": {
                    "dataset_index": {"type": "integer"},        # 1-based, from the study map
                    "outcomes": {"type": "array", "items": _OUTCOME_SCHEMA},
                },
            },
        },
        "notes": {"type": "string"},
    },
}

MAPPER_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["citation", "eligible", "eligibility_rationale", "exclusion_reason",
                 "design_notes", "datasets", "related_files", "roster", "notes"],
    "properties": {
        "citation": {
            "type": "object", "additionalProperties": False,
            "required": ["authors", "year", "title", "journal", "doi", "first_author"],
            "properties": {"authors": {"type": "string"}, "year": {"type": ["integer", "null"]},
                           "title": {"type": "string"}, "journal": {"type": "string"},
                           "doi": {"type": "string"}, "first_author": {"type": "string"}},
        },
        "eligible": {"type": "boolean"},
        "eligibility_rationale": {"type": "string"},
        "exclusion_reason": {"type": "string"},
        "design_notes": {"type": "string"},
        "datasets": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["label", "experiment", "condition", "shared_control",
                             "exposure_order", "group_a", "group_b", "all_groups_listed",
                             "chosen_pair_rationale", "moderators", "notes"],
                "properties": {
                    "label": {"type": "string"},
                    "experiment": {"type": "string"},
                    "condition": {"type": "string"},
                    "shared_control": {"type": "boolean"},
                    "exposure_order": _enum(get_args(ExposureOrder)),
                    "group_a": _GROUP_SCHEMA,
                    "group_b": _GROUP_SCHEMA,
                    "all_groups_listed": {"type": "array", "items": _GROUP_SCHEMA},
                    "chosen_pair_rationale": {"type": "string"},
                    "moderators": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": False,
                                  "required": ["name", "value"],
                                  "properties": {"name": {"type": "string"},
                                                 "value": {"type": "string"}}},
                    },
                    "notes": {"type": "string"},
                },
            },
        },
        "related_files": {"type": "array", "items": {"type": "string"}},
        "roster": {"type": "array", "items": _ROSTER_DECISION_SCHEMA},
        "notes": {"type": "string"},
    },
}

_CHECK_GROUP_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["label", "n", "n_evidence"],
    "properties": {"label": {"type": "string"}, "n": {"type": ["integer", "null"]},
                   "n_evidence": {"type": "string"}},
}

MAPPER_CROSSCHECK_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["eligible", "eligibility_rationale", "datasets", "roster_error_bars", "notes"],
    "properties": {
        "eligible": {"type": "boolean"},
        "eligibility_rationale": {"type": "string"},
        "datasets": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["label", "experiment", "condition", "group_a", "group_b", "outcomes"],
                "properties": {
                    "label": {"type": "string"},
                    "experiment": {"type": "string"},
                    "condition": {"type": "string"},
                    "group_a": _CHECK_GROUP_SCHEMA,
                    "group_b": _CHECK_GROUP_SCHEMA,
                    "outcomes": {
                        "type": "array",
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["outcome_key", "sources"],
                            "properties": {
                                "outcome_key": {"type": "string"},
                                "sources": {
                                    "type": "array",
                                    "items": {
                                        "type": "object", "additionalProperties": False,
                                        "required": ["kind", "page", "locator", "figure_id",
                                                     "table_id", "error_bar_type", "quote"],
                                        "properties": {
                                            "kind": _enum([k.value for k in SourceKind]),
                                            "page": {"type": "integer"},
                                            "locator": {"type": "string"},
                                            "figure_id": {"type": "string"},
                                            "table_id": {"type": "string"},
                                            "error_bar_type": _enum([d.value for d in
                                                                     DispersionType]),
                                            "quote": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "roster_error_bars": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "error_bar_type", "error_bar_scope", "evidence"],
                "properties": {
                    "id": {"type": "string"},            # a roster id, verbatim
                    "error_bar_type": _enum([d.value for d in DispersionType]),
                    "error_bar_scope": _enum(get_args(ErrorBarScope)),
                    "evidence": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}

MAPPER_ADJUDICATE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["eligible", "eligibility_rationale", "datasets", "error_bar_rulings", "notes"],
    "properties": {
        "eligible": {"type": "boolean"},
        "eligibility_rationale": {"type": "string"},
        "datasets": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["primary_dataset_index", "label", "group_a", "group_b", "rationale"],
                "properties": {
                    "primary_dataset_index": {"type": ["integer", "null"]},
                    "label": {"type": "string"},
                    "group_a": _CHECK_GROUP_SCHEMA,
                    "group_b": _CHECK_GROUP_SCHEMA,
                    "rationale": {"type": "string"},
                },
            },
        },
        "error_bar_rulings": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["dataset_index", "outcome_key", "page", "locator", "figure_id",
                             "table_id", "error_bar_type", "evidence"],
                "properties": {
                    "dataset_index": {"type": "integer"},
                    "outcome_key": {"type": "string"},
                    "page": {"type": "integer"},
                    "locator": {"type": "string"},
                    "figure_id": {"type": "string"},
                    "table_id": {"type": "string"},
                    "error_bar_type": _enum([d.value for d in DispersionType]),
                    "evidence": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}

MAPPER_ROSTER_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["decisions"],
    "properties": {"decisions": {"type": "array", "items": _ROSTER_DECISION_SCHEMA}},
}


# ----------------------------------------------------------------------------- prompt context
_GROUP_POLICY = {
    "extremes": "pick the two groups that sit furthest apart on the dimension the definitions "
                "describe",
    "combine_matching": "when several groups match one definition, choose the one the paper "
                        "analyses as a whole and list the others, saying they could be combined",
    "closest_to_definition": "pick the single group that matches each definition most closely",
    "needs_human": "if more than two groups could match, do not choose a pair: leave the groups "
                   "empty and explain what a human has to decide",
}


def protocol_text(protocol: Protocol) -> str:
    """The protocol as prose for a prompt (definitions and rules only, no statistics settings)."""
    lines = [f"TITLE: {protocol.title}"]
    if protocol.research_question:
        lines.append(f"RESEARCH QUESTION: {protocol.research_question.strip()}")
    for group in (protocol.group_a, protocol.group_b):
        lines.append(f"\nGROUP {group.key} — \"{group.label}\"")
        lines.append(f"  definition: {group.definition.strip()}")
        if group.synonyms:
            lines.append(f"  the paper may call it: {', '.join(group.synonyms)}")
    policy = protocol.stats.multi_group_policy
    lines.append(f"\nGROUP SELECTION POLICY ({policy}): when the paper has more than two candidate "
                 f"groups, {_GROUP_POLICY.get(policy, 'follow the group definitions')}.")

    lines.append("\nOUTCOMES (use these keys verbatim):")
    for outcome in protocol.outcomes:
        lines.append(f"- key: {outcome.key} — {outcome.label}")
        lines.append(f"  definition: {outcome.definition.strip()}")
        if outcome.measurement_window:
            lines.append(f"  measurement window: {outcome.measurement_window.strip()}")
        if outcome.higher_is_better_hint:
            lines.append(f"  direction hint: {outcome.higher_is_better_hint.strip()}")
        if outcome.units_hint:
            lines.append(f"  units hint: {outcome.units_hint.strip()}")

    if protocol.eligibility:
        lines.append("\nELIGIBILITY (every criterion must hold):")
        lines += [f"{i}. {rule.strip()}" for i, rule in enumerate(protocol.eligibility, 1)]
    if protocol.dataset_rules:
        lines.append("\nDATASET RULES:")
        lines += [f"{i}. {rule.strip()}" for i, rule in enumerate(protocol.dataset_rules, 1)]
    if protocol.moderators:
        lines.append("\nMODERATORS to record for every dataset: " + ", ".join(protocol.moderators))
    if protocol.notes:
        lines.append(f"\nPROTOCOL NOTES: {protocol.notes.strip()}")
    return "\n".join(lines)


def roster_entries(paper: PaperRecord) -> list[dict[str, Any]]:
    """Every figure and table ingestion detected, in a stable order (figures by id, then tables).

    The order is part of the prompt, and the prompt is part of the fixture key — keep it stable.
    """
    entries: list[dict[str, Any]] = []
    for fig in sorted(paper.figures, key=lambda f: f.id):
        entries.append({"kind": "figure", "id": fig.id, "page": fig.page, "label": fig.label or "",
                        "caption": (fig.caption or "").strip(), "rows": []})
    for table in sorted(paper.tables, key=lambda t: t.id):
        entries.append({"kind": "table", "id": table.id, "page": table.page, "label": "",
                        "caption": (table.caption or "").strip(), "rows": table.rows})
    return entries


def _entry_line(entry: dict[str, Any]) -> str:
    head = f"- {entry['id']} | page {entry['page']}"
    if entry["label"]:
        head += f" | printed label \"{entry['label']}\""
    caption = _clip(entry["caption"], CAPTION_CHARS) or "(no caption found)"
    line = f"{head} | caption: {caption}"
    for row in entry["rows"][:TABLE_ROWS]:
        line += "\n    | " + " | ".join(_clip(str(cell), 40) for cell in row) + " |"
    if len(entry["rows"]) > TABLE_ROWS:
        line += f"\n    ... {len(entry['rows']) - TABLE_ROWS} more rows"
    return line


def roster_text(paper: PaperRecord, entries: list[dict[str, Any]] | None = None) -> str:
    """The roster as prose. `entries` restricts it (used by the follow-up call)."""
    entries = roster_entries(paper) if entries is None else entries
    figures = [e for e in entries if e["kind"] == "figure"]
    tables = [e for e in entries if e["kind"] == "table"]
    blocks = ["FIGURES:"] + ([_entry_line(e) for e in figures] or ["(none detected)"])
    blocks += ["", "TABLES:"] + ([_entry_line(e) for e in tables] or ["(none detected)"])
    return "\n".join(blocks)


def dataset_text(study: StudyMap) -> str:
    """The datasets of a study map as prose — the second pass attaches outcomes to these."""
    if not study.datasets:
        return "(no datasets)"
    lines = []
    for index, dataset in enumerate(study.datasets, start=1):
        head = f"- dataset_index {index} | label \"{dataset.label}\""
        if dataset.experiment:
            head += f" | experiment \"{dataset.experiment}\""
        if dataset.condition:
            head += f" | condition \"{dataset.condition}\""
        lines.append(head)
        for key in ("group_a", "group_b"):
            group: GroupSpec = getattr(dataset, key)
            lines.append(f"    group {key[-1].upper()}: \"{group.label}\" "
                         f"(analysed n = {group.n if group.n is not None else 'not reported'})")
        if dataset.notes:
            lines.append(f"    notes: {_clip(dataset.notes, 300)}")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


# ----------------------------------------------------------------------------- parsing helpers
_SOURCE_KINDS = {k.value for k in SourceKind}
_HIGHER_IS_BETTER = {"higher_is_better": True, "lower_is_better": False, "unknown": None}
#: locator fragments that name one panel/row/column inside a figure or table
_QUALIFIER_RES = (
    re.compile(r"\b(?:fig|figure|tab|table)\s*\.?\s*s?\d+\s*([a-z])\b", re.I),
    re.compile(r"\bpanel\s+([a-z0-9]+)\b", re.I),
    re.compile(r"\b(?:row|column|col)\s*[:=]?\s*['\"‘’“”]([^'\"‘’“”]+)", re.I),
    re.compile(r"\b(?:row|column|col)\s+([a-z0-9][a-z0-9 _-]*)", re.I),
)


def roster_ids(paper: PaperRecord) -> dict[str, set[str]]:
    """The ids an agent is allowed to cite (anything else would break `llm.context` lookups)."""
    return {"figure": {f.id for f in paper.figures}, "table": {t.id for t in paper.tables}}


def _note(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def _source(raw: dict[str, Any], ids: dict[str, set[str]], cell: str,
            flags: list[str]) -> Source:
    """A `Source` from either schema's source object, with the same validation on both paths.

    A kind we do not know becomes `SourceKind.unknown` (the location is kept — a human routes it);
    a figure/table id ingestion never produced is cleared and flagged, because `llm.context`
    raises on an unknown id when an extractor later asks for that crop.
    """
    data = dict(raw)
    data["figure_id"] = data.get("figure_id") or None
    data["table_id"] = data.get("table_id") or None
    if data.get("kind") not in _SOURCE_KINDS:
        data["notes"] = _note(data.get("notes") or "", f"kind {data.get('kind')!r} not recognised")
        data["kind"] = SourceKind.unknown.value
    source = Source.model_validate(data)
    for attr, kind in (("figure_id", "figure"), ("table_id", "table")):
        ident = getattr(source, attr)
        if ident and ident not in ids[kind]:
            flags.append(f"{cell} p{source.page} {source.locator}: {kind} id {ident!r} is not in "
                         f"the ingestion roster — needs human; quote: {_clip(source.quote, 160)!r}")
            source.notes = _note(source.notes, f"{attr} {ident!r} not in the ingestion roster")
            setattr(source, attr, None)
    return source


def _norm(text: str) -> str:
    """Locators/labels compared loosely: 'Fig. 1' == 'Figure 1', 'Table 2' == 'tab 2'."""
    lowered = (text or "").lower().replace("figure", "fig").replace("table", "tab")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def _qualifier(locator: str) -> str:
    """The panel/row/column a locator names, normalised — '' when it names none.

    'Fig 2B, last block' -> 'b';  "Table 1, row 'old'" -> 'old'.  Two sources that name *different*
    panels of the same figure are different locations, so this belongs in the match key.
    """
    found = [_norm(m.group(1)) for pattern in _QUALIFIER_RES
             if (m := pattern.search(locator or ""))]
    return "|".join(sorted({q for q in found if q}))


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _mapping_verdict(primary_a: str, primary_b: str, other_a: str, other_b: str) -> str:
    """How another agent's A/B labels line up with the primary's: agreed | swapped | different."""
    if not all(_norm(x) for x in (primary_a, primary_b, other_a, other_b)):
        return "unconfirmed"
    if (_sim(other_a, primary_a) > _sim(other_a, primary_b)
            and _sim(other_b, primary_b) > _sim(other_b, primary_a)):
        return "agreed"
    if (_sim(other_a, primary_b) > _sim(other_a, primary_a)
            and _sim(other_b, primary_a) > _sim(other_b, primary_b)):
        return "swapped"
    return "different"


def _source_key(source: Source, tier: int) -> tuple | None:
    ident = source.figure_id or source.table_id or ""
    locator = _norm(source.locator)
    if tier == 0:
        return (source.page, ident, locator)
    if tier == 1:
        return (source.page, ident, _qualifier(source.locator)) if ident else None
    return (source.page, locator) if locator else None


def _quotes_overlap(a: str, b: str) -> bool:
    """One quote contains the other — two agents quoting the same sentence, differently trimmed."""
    a, b = _norm(a), _norm(b)
    if min(len(a), len(b)) < QUOTE_MATCH_CHARS:
        return False
    return a in b or b in a


def _match_source(sources: list[Source], other: Source) -> Source | None:
    """Same location? id (+ panel/row) first, then locator, then the quote; the page always matches.

    Tier 1 falls back to the bare id only when neither locator names a panel/row: "Fig 2B, open
    circles" and "Fig 2A, filled squares" are different locations that share `figure_id`. The quote
    tier is for text sources only — two figure panels routinely quote the same caption.
    """
    for tier in (0, 1, 2):
        target = _source_key(other, tier)
        if target is None:
            continue
        for source in sources:
            if _source_key(source, tier) == target:      # tier 1 needs the same panel/row, and
                return source                            # ('' == '') is the both-unqualified case
    if other.figure_id or other.table_id:
        return None
    for source in sources:
        if source.figure_id or source.table_id:
            continue
        if source.page == other.page and _quotes_overlap(source.quote, other.quote):
            return source
    return None


def _same_location(source: Source, raw: dict[str, Any]) -> bool:
    if int(raw.get("page") or 0) != source.page:
        return False
    ident = (raw.get("figure_id") or raw.get("table_id") or "").strip()
    own = source.figure_id or source.table_id or ""
    if ident and own:
        if ident != own:
            return False
        ruled, mine = _qualifier(raw.get("locator", "")), _qualifier(source.locator)
        return not ruled or not mine or ruled == mine
    return not raw.get("locator") or _norm(raw.get("locator", "")) == _norm(source.locator)


# ----------------------------------------------------------------------------- map assembly
def _build_map(parsed: dict[str, Any], paper: PaperRecord) -> StudyMap:
    sha12 = paper.sha256[:12]
    datasets = [_build_dataset(raw, index, sha12)
                for index, raw in enumerate(parsed.get("datasets") or [], start=1)]
    return StudyMap(
        paper_id=paper.sha256,
        citation=Citation.model_validate(parsed.get("citation") or {}),
        eligible=parsed.get("eligible"),
        eligibility_rationale=parsed.get("eligibility_rationale") or "",
        exclusion_reason=parsed.get("exclusion_reason") or "",
        design_notes=parsed.get("design_notes") or "",
        datasets=datasets,
        related_files=list(parsed.get("related_files") or []),
        notes=parsed.get("notes") or "",
    )


def _build_dataset(raw: dict[str, Any], index: int, sha12: str) -> DatasetSpec:
    """One dataset of the study map; its outcomes arrive with the source map."""
    moderators = {m["name"]: m.get("value") or "" for m in raw.get("moderators") or []
                  if (m.get("name") or "").strip()}
    return DatasetSpec(
        dataset_id=f"{sha12}:d{index}",
        label=raw.get("label") or "",
        experiment=raw.get("experiment") or "",
        condition=raw.get("condition") or "",
        cluster_id=sha12,
        shared_control=bool(raw.get("shared_control")),
        exposure_order=raw.get("exposure_order") or "unknown",
        group_a=GroupSpec.model_validate(raw.get("group_a") or {}),
        group_b=GroupSpec.model_validate(raw.get("group_b") or {}),
        all_groups_listed=[GroupSpec.model_validate(g) for g in raw.get("all_groups_listed") or []],
        chosen_pair_rationale=raw.get("chosen_pair_rationale") or "",
        moderators=moderators,
        notes=raw.get("notes") or "")


def _outcome_key_ok(key: str, dataset_id: str, outcome_keys: set[str], flags: list[str]) -> bool:
    """Both agent paths validate outcome keys the same way (an unknown key reaches no analysis)."""
    if key in outcome_keys:
        return True
    flags.append(f"dataset {dataset_id}: outcome key {key!r} is not in the protocol — needs human")
    return False


def _merge_outcome(existing: OutcomeSources, raw: dict[str, Any], sources: list[Source],
                   dataset_id: str, flags: list[str]) -> None:
    """One outcome reported twice: keep both source lists, keep every field, flag real conflicts."""
    existing.sources += sources
    fields = {"measure_name": raw.get("measure_name") or "",
              "units": raw.get("units") or "",
              "operationalization": raw.get("operationalization") or "",
              "higher_is_better_evidence": raw.get("higher_is_better_evidence") or ""}
    for name, value in fields.items():
        current = getattr(existing, name)
        if not value or value == current:
            continue
        if not current:
            setattr(existing, name, value)
            continue
        flags.append(f"dataset {dataset_id} {existing.outcome_key}: reported twice with different "
                     f"{name} ({current!r} vs {value!r}) — needs human")
    direction = _HIGHER_IS_BETTER.get(raw.get("higher_is_better") or "")
    if direction is None:
        return
    if existing.higher_is_better is None:
        existing.higher_is_better = direction
    elif existing.higher_is_better != direction:
        flags.append(f"dataset {dataset_id} {existing.outcome_key}: reported twice with opposite "
                     f"higher_is_better — needs human")


def _apply_sources(study: StudyMap, parsed: dict[str, Any], outcome_keys: set[str],
                   ids: dict[str, set[str]], flags: list[str]) -> None:
    """Fold the source map's outcomes into the datasets of the study map."""
    for raw in parsed.get("datasets") or []:
        index = int(raw.get("dataset_index") or 0)
        if not 1 <= index <= len(study.datasets):
            flags.append(f"source map: outcomes reported for dataset_index {index}, which the "
                         f"study map does not contain — needs human")
            continue
        dataset = study.datasets[index - 1]
        for raw_outcome in raw.get("outcomes") or []:
            key = raw_outcome.get("outcome_key") or ""
            _outcome_key_ok(key, dataset.dataset_id, outcome_keys, flags)
            cell = f"dataset {dataset.dataset_id} {key}"
            sources = [_source(s, ids, cell, flags) for s in raw_outcome.get("sources") or []]
            existing = next((o for o in dataset.outcomes if o.outcome_key == key), None)
            if existing is not None:                    # the model split one outcome over two rows
                _merge_outcome(existing, raw_outcome, sources, dataset.dataset_id, flags)
                continue
            dataset.outcomes.append(OutcomeSources(
                outcome_key=key,
                measure_name=raw_outcome.get("measure_name") or "",
                units=raw_outcome.get("units") or "",
                higher_is_better=_HIGHER_IS_BETTER.get(raw_outcome.get("higher_is_better") or ""),
                higher_is_better_evidence=raw_outcome.get("higher_is_better_evidence") or "",
                operationalization=raw_outcome.get("operationalization") or "",
                analysis_metric=raw_outcome.get("analysis_metric") or "unknown",
                sources=sources))


def _flag_thin_outcomes(study: StudyMap, flags: list[str]) -> None:
    """A dataset with no numbers at all, or an outcome whose only source is unroutable."""
    if study.eligible is False:
        return
    for dataset in study.datasets:
        if not any(outcome.sources for outcome in dataset.outcomes):
            flags.append(f"dataset {dataset.dataset_id}: no sources found for any outcome "
                         f"— needs human")
            continue
        for outcome in dataset.outcomes:
            if outcome.sources and all(s.kind is SourceKind.unknown for s in outcome.sources):
                first = outcome.sources[0]
                flags.append(f"dataset {dataset.dataset_id} {outcome.outcome_key} p{first.page} "
                             f"{first.locator}: the only source is of an unlisted kind — needs "
                             f"human; quote: {_clip(first.quote, 160)!r}")


def _decisions(paper: PaperRecord, raw_decisions: Any, decided: dict[str, RosterDecision],
               flags: list[str]) -> None:
    """Fold raw `{id, relevant, reason, outcome_keys}` answers into `decided` (page/label in code)."""
    entries = {entry["id"]: entry for entry in roster_entries(paper)}
    for raw in raw_decisions or []:
        rid = (raw.get("id") or "").strip()
        entry = entries.get(rid)
        if entry is None:
            flags.append(f"roster {rid!r}: decided on an item ingestion did not detect "
                         f"— needs human")
            continue
        decided[rid] = RosterDecision(
            kind=entry["kind"], id=rid, page=entry["page"], label=entry["label"],
            relevant=bool(raw.get("relevant")), reason=raw.get("reason") or "",
            outcome_keys=list(raw.get("outcome_keys") or []))


# ----------------------------------------------------------------------------- cross-check
ADDED_BY_CROSSCHECK = "added by cross-check"


@dataclass
class _Conflicts:
    eligibility: bool = False
    n_mismatch: bool = False
    mapping: list[int] = field(default_factory=list)        # indices into StudyMap.datasets
    error_bars: list[dict[str, Any]] = field(default_factory=list)
    #: (dataset index, "group_a"/"group_b") -> the cross-check's n, so adjudication can check that
    #: an adopted number came from one of the two agents rather than from nowhere
    check_n: dict[tuple[int, str], int] = field(default_factory=dict)

    @property
    def needs_adjudication(self) -> bool:
        return self.eligibility or self.n_mismatch or bool(self.mapping)


def _diff(study: StudyMap, check: dict[str, Any], outcome_keys: set[str],
          ids: dict[str, set[str]], disagreements: list[str], flags: list[str]) -> _Conflicts:
    """Compare the cross-check with the map, append sources only it found, collect open conflicts."""
    conflicts = _Conflicts()
    checked_eligible = check.get("eligible")
    if (study.eligible is not None and checked_eligible is not None
            and bool(checked_eligible) != bool(study.eligible)):
        disagreements.append(f"eligibility: primary={study.eligible} "
                             f"cross-check={bool(checked_eligible)}")
        conflicts.eligibility = True

    check_datasets = check.get("datasets") or []
    if len(check_datasets) != len(study.datasets):
        disagreements.append(f"dataset count: primary={len(study.datasets)} "
                             f"cross-check={len(check_datasets)}")
    for index, (dataset, raw) in enumerate(zip(study.datasets, check_datasets)):
        _diff_dataset(index, dataset, raw, conflicts, outcome_keys, ids, disagreements, flags)
    for position in range(len(study.datasets), len(check_datasets)):
        _report_extra_dataset(check_datasets[position], position, disagreements, flags)
    for position in range(len(check_datasets), len(study.datasets)):
        flags.append(f"dataset {study.datasets[position].dataset_id}: group mapping unconfirmed "
                     f"(the cross-check found no counterpart dataset) — needs human")
    return conflicts


def _report_extra_dataset(raw: dict[str, Any], position: int, disagreements: list[str],
                          flags: list[str]) -> None:
    """A dataset only the cross-check saw: describe it in full, never invent it into the map."""
    groups = []
    for key in ("group_a", "group_b"):
        group = raw.get(key) or {}
        groups.append(f"{key}={group.get('label') or '?'!r} n={group.get('n')} "
                      f"({_clip(group.get('n_evidence') or '', 120)})")
    sources = [f"{s.get('kind')} p{s.get('page')} {s.get('locator')}"
               + (f" [{s.get('figure_id') or s.get('table_id')}]"
                  if (s.get("figure_id") or s.get("table_id")) else "")
               for outcome in raw.get("outcomes") or [] for s in outcome.get("sources") or []]
    described = (f"cross-check dataset {position + 1} not in the map: "
                 f"label={raw.get('label') or ''!r} experiment={raw.get('experiment') or ''!r} "
                 f"condition={raw.get('condition') or ''!r}; {'; '.join(groups)}; "
                 f"outcomes={[o.get('outcome_key') for o in raw.get('outcomes') or []]}; "
                 f"sources={sources}")
    disagreements.append(described)
    flags.append(f"cross-check reports a dataset the map does not contain "
                 f"({raw.get('label') or 'unlabelled'!r}) — needs human")


def _diff_dataset(index: int, dataset: DatasetSpec, raw: dict[str, Any], conflicts: _Conflicts,
                  outcome_keys: set[str], ids: dict[str, set[str]], disagreements: list[str],
                  flags: list[str]) -> None:
    for key in ("group_a", "group_b"):
        group: GroupSpec = getattr(dataset, key)
        other_n = (raw.get(key) or {}).get("n")
        if other_n is None:
            continue
        conflicts.check_n[(index, key)] = int(other_n)
        if group.n is not None and int(other_n) != int(group.n):
            disagreements.append(f"{dataset.dataset_id} {key} n: primary={group.n} "
                                 f"cross-check={int(other_n)}")
            conflicts.n_mismatch = True

    label_a = (raw.get("group_a") or {}).get("label") or ""
    label_b = (raw.get("group_b") or {}).get("label") or ""
    verdict = _mapping_verdict(dataset.group_a.label, dataset.group_b.label, label_a, label_b)
    if verdict == "unconfirmed":
        flags.append(f"dataset {dataset.dataset_id}: group mapping unconfirmed (the cross-check "
                     f"named no groups) — needs human")
    elif verdict != "agreed":
        disagreements.append(
            f"{dataset.dataset_id} group mapping ({verdict}): primary A={dataset.group_a.label!r} "
            f"B={dataset.group_b.label!r}; cross-check A={label_a!r} B={label_b!r}")
        conflicts.mapping.append(index)

    for raw_outcome in raw.get("outcomes") or []:
        key = raw_outcome.get("outcome_key") or ""
        if not _outcome_key_ok(key, dataset.dataset_id, outcome_keys, flags):
            continue                                   # never create an unusable outcome
        outcome = next((o for o in dataset.outcomes if o.outcome_key == key), None)
        cell = f"dataset {dataset.dataset_id} {key}"
        for raw_source in raw_outcome.get("sources") or []:
            source = _source(raw_source, ids, f"{cell} (cross-check)", flags)
            match = _match_source(outcome.sources, source) if outcome is not None else None
            if match is None:
                if outcome is None:
                    outcome = OutcomeSources(outcome_key=key)
                    dataset.outcomes.append(outcome)
                source.notes = _note(source.notes, ADDED_BY_CROSSCHECK)
                outcome.sources.append(source)
                disagreements.append(f"{dataset.dataset_id} {key}: source added by cross-check "
                                     f"— p{source.page} {source.locator}")
                continue
            if (match.figure_id or match.table_id) or match.error_bar_type is source.error_bar_type:
                continue                               # figure/table bars are settled by the roster
            disagreements.append(
                f"{dataset.dataset_id} {key} p{match.page} {match.locator}: error bar "
                f"primary={match.error_bar_type.value} "
                f"cross-check={source.error_bar_type.value}")


def _roster_determinations(check: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The cross-check's error-bar reading of every roster id it answered for."""
    out: dict[str, dict[str, Any]] = {}
    for raw in check.get("roster_error_bars") or []:
        ident = (raw.get("id") or "").strip()
        if ident:
            out[ident] = raw
    return out


def _agree_error_bars(study: StudyMap, determinations: dict[str, dict[str, Any]],
                      conflicts: _Conflicts, disagreements: list[str], flags: list[str]) -> None:
    """Amendment D: a figure/table error bar counts only when a second agent read it the same way.

    The cross-check reports an error-bar type for every id on the deterministic roster, so coverage
    — not just the absence of a conflict — decides: no determination for that id leaves the source
    `unconfirmed` and in the human queue. Text sources carry no id and are Task 8's job.
    """
    for index, dataset in enumerate(study.datasets):
        for outcome in dataset.outcomes:
            for source in outcome.sources:
                ident = source.figure_id or source.table_id
                if not ident:
                    continue
                cell = (f"dataset {dataset.dataset_id} {outcome.outcome_key} {source.locator} "
                        f"(p{source.page})")
                ruling = determinations.get(ident)
                if ruling is None or ADDED_BY_CROSSCHECK in source.notes:
                    source.error_bar_agreement = "unconfirmed"
                    flags.append(f"{cell}: error-bar type {source.error_bar_type.value} "
                                 f"unconfirmed by a second agent — needs human")
                    continue
                other = DispersionType(ruling.get("error_bar_type") or "UNKNOWN")
                if other is source.error_bar_type:
                    source.error_bar_agreement = "agreed"
                    scope = ruling.get("error_bar_scope") or "unknown"
                    if scope != source.error_bar_scope:
                        disagreements.append(f"{cell} error-bar scope: primary="
                                             f"{source.error_bar_scope} cross-check={scope}")
                    continue
                source.error_bar_agreement = "conflict"
                disagreements.append(f"{cell} error bar: primary={source.error_bar_type.value} "
                                     f"cross-check={other.value}")
                conflicts.error_bars.append({"dataset_index": index,
                                             "dataset_id": dataset.dataset_id,
                                             "outcome_key": outcome.outcome_key,
                                             "source": source, "check_type": other})


# ----------------------------------------------------------------------------- adjudication
def _apply_adjudication(study: StudyMap, adjudicated: dict[str, Any], conflicts: _Conflicts,
                        labels: list[tuple[str, str]], disagreements: list[str],
                        flags: list[str]) -> None:
    ruled_eligible = adjudicated.get("eligible")
    if ruled_eligible is not None and bool(ruled_eligible) != bool(study.eligible):
        disagreements.append(f"adjudicated eligibility: {bool(ruled_eligible)} "
                             f"(primary said {study.eligible})")
        study.eligible = bool(ruled_eligible)
        study.eligibility_rationale = (adjudicated.get("eligibility_rationale")
                                       or study.eligibility_rationale)
    elif conflicts.eligibility:
        disagreements.append(f"adjudicated eligibility: {study.eligible} (primary confirmed)")

    for raw in adjudicated.get("datasets") or []:
        index = raw.get("primary_dataset_index")
        if not index or not 1 <= int(index) <= len(study.datasets):
            flags.append(f"adjudicator reports dataset {raw.get('label', '')!r} that the map does "
                         f"not contain — needs human")
            continue
        position = int(index) - 1
        dataset = study.datasets[position]
        if position in conflicts.mapping:
            _resolve_mapping(dataset, raw, position, labels[position], conflicts, disagreements)
        for key in ("group_a", "group_b"):
            _adopt_group(dataset, key, raw.get(key) or {}, conflicts.check_n.get((position, key)),
                         disagreements, flags)

    _apply_error_bar_rulings(study, adjudicated.get("error_bar_rulings") or [], conflicts,
                             disagreements)


def _resolve_mapping(dataset: DatasetSpec, raw: dict[str, Any], position: int,
                     primary_labels: tuple[str, str], conflicts: _Conflicts,
                     disagreements: list[str]) -> None:
    ruled_a = (raw.get("group_a") or {}).get("label") or ""
    ruled_b = (raw.get("group_b") or {}).get("label") or ""
    verdict = _mapping_verdict(primary_labels[0], primary_labels[1], ruled_a, ruled_b)
    if verdict == "swapped":
        dataset.group_a, dataset.group_b = dataset.group_b, dataset.group_a
        disagreements.append(f"adjudicated {dataset.dataset_id} group mapping: swapped to "
                             f"A={dataset.group_a.label!r} B={dataset.group_b.label!r}")
        conflicts.mapping.remove(position)
    elif verdict == "agreed":
        disagreements.append(f"adjudicated {dataset.dataset_id} group mapping: primary confirmed")
        conflicts.mapping.remove(position)


def _adopt_group(dataset: DatasetSpec, key: str, raw: dict[str, Any], check_n: int | None,
                 disagreements: list[str], flags: list[str]) -> None:
    """Adopt the adjudicated n, but only when it is one of the two agents' answers.

    A number that agrees with neither agent has no second reader at all, so nothing is adopted and
    a human decides — the same rule the error-bar rulings follow.
    """
    group: GroupSpec = getattr(dataset, key)
    ruled_n = raw.get("n")
    if ruled_n is None:
        return
    ruled_n = int(ruled_n)
    if group.n is not None and ruled_n == group.n:
        disagreements.append(f"adjudicated {dataset.dataset_id} {key} n: {ruled_n} "
                             f"(primary confirmed)")
        return
    if group.n is not None and check_n is not None and ruled_n != check_n:
        disagreements.append(f"adjudicated {dataset.dataset_id} {key} n: {ruled_n} agrees with "
                             f"neither agent (primary={group.n} cross-check={check_n}) — not "
                             f"adopted")
        flags.append(f"dataset {dataset.dataset_id} {key}: n disagreement unresolved "
                     f"(primary={group.n}, cross-check={check_n}, adjudicator={ruled_n}) "
                     f"— needs human")
        return
    disagreements.append(f"adjudicated {dataset.dataset_id} {key} n: {ruled_n} "
                         f"(primary said {group.n})")
    group.n = ruled_n
    group.n_evidence = raw.get("n_evidence") or group.n_evidence
    group.label = raw.get("label") or group.label


def _apply_error_bar_rulings(study: StudyMap, rulings: list[dict[str, Any]], conflicts: _Conflicts,
                             disagreements: list[str]) -> None:
    for ruling in rulings:
        index = int(ruling.get("dataset_index") or 0) - 1
        ruled = ruling.get("error_bar_type")
        for conflict in list(conflicts.error_bars):
            if conflict["dataset_index"] != index:
                continue
            if conflict["outcome_key"] != (ruling.get("outcome_key") or ""):
                continue
            source: Source = conflict["source"]
            if not _same_location(source, ruling):
                continue
            if ruled == source.error_bar_type.value:
                disagreements.append(f"adjudicated {conflict['dataset_id']} "
                                     f"{conflict['outcome_key']} {source.locator} error bar: "
                                     f"{ruled} (primary confirmed)")
            elif ruled == conflict["check_type"].value:
                source.error_bar_type = DispersionType(ruled)
                source.error_bar_evidence = ruling.get("evidence") or source.error_bar_evidence
                disagreements.append(f"adjudicated {conflict['dataset_id']} "
                                     f"{conflict['outcome_key']} {source.locator} error bar: "
                                     f"{ruled} (cross-check confirmed)")
            else:
                continue                      # agrees with neither: the conflict stays open
            source.error_bar_agreement = "agreed"
            conflicts.error_bars.remove(conflict)


# ----------------------------------------------------------------------------- the agent
def map_study(client: LLMClient, paper: PaperRecord, protocol: Protocol, *,
              model_primary: str = MODELS["primary"], model_check: str = MODELS["secondary"],
              model_adjudicate: str = MODELS["adjudicator"],
              pdf_file_id: str | None = None) -> StudyMap:
    """Map one paper against one protocol: eligibility, datasets, group Ns and source locations."""
    sha12 = paper.sha256[:12]
    # No `cache_control` here: the mapper's four calls each carry a DIFFERENT output schema, and
    # the schema is part of the cached prefix (measured live, task 15 §A), so a marker would write
    # a fresh cache entry per call at 1.25x the input price and never be read. The whole-paper
    # agents that ARE called repeatedly with one schema (verifier, orientation, adjudicator) mark
    # theirs — see `canopy.agents.verify_common.whole_paper`.
    document = client.pdf_block(pdf_path=None if pdf_file_id else paper.source_path,
                                file_id=pdf_file_id, cache=False)
    betas = [FILES_API_BETA] if pdf_file_id else None
    entries = roster_entries(paper)
    ids = roster_ids(paper)
    outcome_keys = {o.key for o in protocol.outcomes}
    protocol_prompt = protocol_text(protocol)
    roster_prompt = roster_text(paper, entries)
    flags: list[str] = []
    disagreements: list[str] = []

    primary = client.structured(
        model=model_primary, system=SYSTEM, schema=MAPPER_SCHEMA, effort="high", max_tokens=16000,
        betas=betas, prompt_version=PROMPT_VERSION, cell_key=f"map:{sha12}",
        messages=[{"role": "user", "content": [document, text_block(render_prompt(
            "mapper", PROTOCOL=protocol_prompt, ROSTER=roster_prompt))]}])
    parsed = primary.parsed or {}
    study = _build_map(parsed, paper)
    call_ids = [primary.call_id]

    if study.eligible is not False and not study.datasets:
        # An eligible paper with no contrast in it is a self-contradiction, and it is the one
        # failure that leaves no trace behind: no dataset means no source pass, no candidate, no
        # row, and nothing in the review queue for a person to look at. Heuer & Hegele 2008 came
        # back exactly this way — eligible, 900 words of design notes, `datasets: []` — while the
        # cross-check of the very same paper found two. Ask once more before believing it.
        again = client.structured(
            model=model_primary, system=SYSTEM, schema=MAPPER_SCHEMA, effort="high",
            max_tokens=16000, betas=betas, prompt_version=PROMPT_VERSION,
            cache_key_extra="retry-no-datasets", cell_key=f"map:{sha12}",
            messages=[{"role": "user", "content": [document, text_block(render_prompt(
                "mapper", PROTOCOL=protocol_prompt, ROSTER=roster_prompt))]}])
        call_ids.append(again.call_id)
        retried = _build_map(again.parsed or {}, paper)
        disagreements.append(f"the primary map found no dataset in a paper it called eligible; "
                             f"asked again and got {len(retried.datasets)}")
        if retried.datasets:
            parsed, study = again.parsed or {}, retried
        else:
            flags.append("the mapper found no dataset in a paper it called eligible, twice — "
                         "needs human")

    if study.datasets:                       # nothing to locate when the paper has no contrast
        content = [document, text_block(render_prompt(
            "mapper_sources", PROTOCOL=protocol_prompt, ROSTER=roster_prompt,
            DATASETS=dataset_text(study)))]
        for attempt in ("", "retry-1"):      # an empty answer is a failure, not an answer
            located = client.structured(
                model=model_primary, system=SYSTEM, schema=MAPPER_SOURCES_SCHEMA, effort="high",
                max_tokens=16000, betas=betas, prompt_version=PROMPT_VERSION,
                cache_key_extra=attempt, cell_key=f"map-sources:{sha12}",
                messages=[{"role": "user", "content": content}])
            call_ids.append(located.call_id)
            parsed_sources = located.parsed or {}
            if _has_sources(parsed_sources):
                break
            disagreements.append(f"source map returned no locations{' (retry)' if attempt else ''}")
        _apply_sources(study, parsed_sources, outcome_keys, ids, flags)

    decided: dict[str, RosterDecision] = {}
    _decisions(paper, parsed.get("roster"), decided, flags)
    missing = [entry for entry in entries if entry["id"] not in decided]
    if missing:
        try:                                 # an optional cheap call must never kill the map
            follow_up = _roster_follow_up(client, paper, protocol_prompt, missing, model_check)
            call_ids.append(follow_up.call_id)
            _decisions(paper, (follow_up.parsed or {}).get("decisions"), decided, flags)
        except LLMError as exc:
            disagreements.append(f"roster follow-up failed ({type(exc).__name__}): "
                                 f"{_clip(str(exc), 160)}")
    for entry in entries:
        if entry["id"] not in decided:
            decided[entry["id"]] = RosterDecision(
                kind=entry["kind"], id=entry["id"], page=entry["page"], label=entry["label"],
                relevant=False, reason="mapper did not decide")
            flags.append(f"roster {entry['id']} (page {entry['page']}): mapper did not decide "
                         f"relevance — needs human")
    study.roster = [decided[entry["id"]] for entry in entries]

    check = client.structured(
        model=model_check, system=SYSTEM, schema=MAPPER_CROSSCHECK_SCHEMA, effort="medium",
        max_tokens=16000, betas=betas, prompt_version=PROMPT_VERSION,
        cell_key=f"map-crosscheck:{sha12}",
        messages=[{"role": "user", "content": [document, text_block(render_prompt(
            "mapper_crosscheck", PROTOCOL=protocol_prompt, ROSTER=roster_prompt))]}])
    call_ids.append(check.call_id)
    checked = check.parsed or {}
    labels = [(d.group_a.label, d.group_b.label) for d in study.datasets]
    conflicts = _diff(study, checked, outcome_keys, ids, disagreements, flags)
    _agree_error_bars(study, _roster_determinations(checked), conflicts, disagreements, flags)
    _flag_thin_outcomes(study, flags)

    if conflicts.needs_adjudication:
        verdict = client.structured(
            model=model_adjudicate, system=SYSTEM, schema=MAPPER_ADJUDICATE_SCHEMA, effort="xhigh",
            max_tokens=16000, betas=betas, prompt_version=PROMPT_VERSION,
            cell_key=f"map-adjudicate:{sha12}",
            messages=[{"role": "user", "content": [document, text_block(render_prompt(
                "mapper_adjudicate", PROTOCOL=protocol_prompt,
                MAP_A=json.dumps(study.model_dump(mode="json"), ensure_ascii=False, indent=1),
                MAP_B=json.dumps(checked, ensure_ascii=False, indent=1),
                DISAGREEMENTS=_disagreement_text(disagreements, conflicts)))]}])
        call_ids.append(verdict.call_id)
        _apply_adjudication(study, verdict.parsed or {}, conflicts, labels, disagreements, flags)

    for position in conflicts.mapping:
        dataset = study.datasets[position]
        flags.append(f"dataset {dataset.dataset_id}: group mapping disagreement, primary kept "
                     f"(A={labels[position][0]!r}/B={labels[position][1]!r}) — needs human")
    for conflict in conflicts.error_bars:
        source: Source = conflict["source"]
        flags.append(f"dataset {conflict['dataset_id']} {conflict['outcome_key']} "
                     f"{source.locator} (p{source.page}): error-bar type disagreement "
                     f"({source.error_bar_type.value} vs {conflict['check_type'].value}) "
                     f"— needs human")

    study.disagreements = disagreements
    study.needs_human = flags
    study.model = model_primary
    study.prompt_version = PROMPT_VERSION
    study.llm_call_ids = call_ids
    return study


def _has_sources(parsed: dict[str, Any]) -> bool:
    """True when the source map actually located something (an empty answer must be retried)."""
    return any(outcome.get("sources")
               for dataset in parsed.get("datasets") or []
               for outcome in dataset.get("outcomes") or [])


def _roster_follow_up(client: LLMClient, paper: PaperRecord, protocol_prompt: str,
                      missing: list[dict[str, Any]], model: str):
    """Cheap second look at the figures/tables the map left undecided (images + table cells)."""
    content: list[dict[str, Any]] = []
    content += figure_blocks(paper, [e["id"] for e in missing if e["kind"] == "figure"])
    for entry in (e for e in missing if e["kind"] == "table"):
        content.append(text_block(f"[table {entry['id']} on page {entry['page']}]\n"
                                  + _entry_line(entry)))
    content.append(text_block(render_prompt("mapper_roster", PROTOCOL=protocol_prompt,
                                            ROSTER=roster_text(paper, missing))))
    return client.structured(model=model, system=SYSTEM, schema=MAPPER_ROSTER_SCHEMA,
                             effort="medium", max_tokens=4000, prompt_version=PROMPT_VERSION,
                             cell_key=f"map-roster:{paper.sha256[:12]}",
                             messages=[{"role": "user", "content": content}])


def _disagreement_text(disagreements: list[str], conflicts: _Conflicts) -> str:
    lines = [f"- {line}" for line in disagreements] or ["- (none listed)"]
    if conflicts.error_bars:
        lines.append("\nOpen error-bar questions (one ruling each):")
        for conflict in conflicts.error_bars:
            source: Source = conflict["source"]
            lines.append(
                f"- dataset_index {conflict['dataset_index'] + 1}, outcome "
                f"{conflict['outcome_key']}, page {source.page}, locator {source.locator!r}, "
                f"figure_id {source.figure_id or ''!r}, table_id {source.table_id or ''!r}: "
                f"MAP A says {source.error_bar_type.value}, "
                f"MAP B says {conflict['check_type'].value}")
    return "\n".join(lines)
