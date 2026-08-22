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
from typing import (Any, Callable, Collection, Mapping, MutableMapping, Sequence,
                    get_args)

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.errors import LLMError
from ..llm.context import FILES_API_BETA, figure_blocks, text_block
from ..models import (C6_DEMOTION_NOTE, C6_WITHHELD_NOTE, HUMAN_DECIDER_NAME,
                      UNREADABLE_SAMPLES,
                      MAP_ADJUDICATOR_NAME, AnalysisMetric, Citation, DatasetSpec,
                      DispersionType, ErrorBarScope, ExposureOrder, GroupSpec, MapQuestion,
                      OutcomeSources, Protocol, RosterDecision, Source, SourceKind, SourceRole,
                      SourceSample, StudyMap, XAxisKind, c6_demoted_note)
from . import load_prompt, render_prompt

__all__ = ["apply_map_answers", "extraction_blocks", "unreadable_cell", "map_study", "measure_of",
           "open_map_questions", "protocol_text", "readable_sources", "roster_text",
           "MAP_ADJUDICATOR",
           "roster_entries", "dataset_text", "source_unreadable_reason", "split_metrics",
           "named_alternatives", "PROMPT_VERSION", "MAPPER_SCHEMA", "MAPPER_SOURCES_SCHEMA",
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
                 "x_axis_kind", "role", "sample", "sample_note", "notes"],
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
        "role": _enum(get_args(SourceRole)),
        "sample": _enum(get_args(SourceSample)),
        "sample_note": {"type": "string"},
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
                                                     "table_id", "error_bar_type",
                                                     "analysis_metric", "quote"],
                                        "properties": {
                                            "kind": _enum([k.value for k in SourceKind]),
                                            "page": {"type": "integer"},
                                            "locator": {"type": "string"},
                                            "figure_id": {"type": "string"},
                                            "table_id": {"type": "string"},
                                            "error_bar_type": _enum([d.value for d in
                                                                     DispersionType]),
                                            # C6/F6: a location that cannot say which measure it
                                            # reads is an unresolved second candidate, and every
                                            # location this agent added used to arrive that way —
                                            # the schema never asked it. `unknown` is still an
                                            # honest answer; it is just no longer the only one.
                                            "analysis_metric": _enum(get_args(AnalysisMetric)),
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
    "required": ["eligible", "eligibility_rationale", "datasets", "dataset_inclusions",
                 "measure_rulings", "error_bar_rulings", "notes"],
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
        "dataset_inclusions": {                      # C7: one row per single-mapper dataset
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["primary_dataset_index", "verdict", "rule", "quote", "rationale"],
                "properties": {
                    "primary_dataset_index": {"type": "integer"},      # 1-based, into MAP A
                    "verdict": _enum(["include", "exclude", "unknown"]),
                    "rule": {"type": "string"},                        # the protocol rule, verbatim
                    "quote": {"type": "string"},                       # the paper's own words
                    "rationale": {"type": "string"},
                },
            },
        },
        "measure_rulings": {                         # C6: one row per two-measure outcome
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["dataset_index", "outcome_key", "verdict", "winning_analysis_metric",
                             "winning_location", "losing_locations", "winner_quote", "loser_quote",
                             "rationale"],
                "properties": {
                    "dataset_index": {"type": "integer"},              # 1-based, into MAP A
                    "outcome_key": {"type": "string"},
                    "verdict": _enum(["winner", "toss_up", "unknown"]),
                    "winning_analysis_metric": _enum(get_args(AnalysisMetric)),
                    # C6/F2: two operationalizations routinely share one `analysis_metric`
                    # (Heuer's Experiment 2 prints both as `change_from_baseline`), and a ruling
                    # that can only name the metric then demotes nothing and settles nothing.
                    # The LOCATION is what separates them.
                    "winning_location": {"type": "string"},
                    "losing_locations": {"type": "array", "items": {"type": "string"}},
                    "winner_quote": {"type": "string"},
                    "loser_quote": {"type": "string"},
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
    """One outcome reported twice: keep both source lists, keep every field, flag real conflicts.

    A second report that names a DIFFERENT measure or unit is not a second look at this outcome —
    it is a different quantity wearing the same key, and its source locations must not join this
    outcome's list. They used to: the sources were concatenated before the conflict was even
    looked for, so an extractor pointed at this outcome could read either panel. Heuer & Hegele
    2008's late adaptation came back as the practice-block initial direction error rather than the
    adaptive shift that way. The reading is kept, the intruding locations are not, and the reason
    travels out as a flag.
    """
    fields = {"measure_name": raw.get("measure_name") or "",
              "units": raw.get("units") or "",
              "operationalization": raw.get("operationalization") or "",
              "higher_is_better_evidence": raw.get("higher_is_better_evidence") or ""}
    conflicts: list[str] = []
    for name, value in fields.items():
        current = getattr(existing, name)
        if not value or value == current:
            continue
        if not current:
            setattr(existing, name, value)
            continue
        conflicts.append(f"{name} ({current!r} vs {value!r})")
        flags.append(f"dataset {dataset_id} {existing.outcome_key}: reported twice with different "
                     f"{name} ({current!r} vs {value!r}) — needs human")
    #: `higher_is_better_evidence` is prose ABOUT the measure, not the measure — two agents wording
    #: the same direction differently is not two quantities.
    quantity = [c for c in conflicts if not c.startswith("higher_is_better_evidence")]
    if quantity:
        flags.append(f"dataset {dataset_id} {existing.outcome_key}: "
                     f"{len(sources)} source location(s) were NOT added to this outcome because "
                     f"they describe a different quantity ({'; '.join(quantity)}) — needs human")
    else:
        existing.sources += sources
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
    #: C7: indices of datasets only the PRIMARY proposed — an inclusion question, not a note
    single_mapper: list[int] = field(default_factory=list)
    #: C6: (dataset index, outcome_key) whose map names two measures — a `which_measure` decision
    measures: list[tuple[int, str]] = field(default_factory=list)
    #: (dataset index, "group_a"/"group_b") -> the cross-check's n, so adjudication can check that
    #: an adopted number came from one of the two agents rather than from nowhere
    check_n: dict[tuple[int, str], int] = field(default_factory=dict)

    @property
    def needs_adjudication(self) -> bool:
        # A dataset one agent never saw, and an outcome with two candidate measures, are both
        # decisions taken HERE — before any extraction is bought against them (C6, C7).
        return bool(self.eligibility or self.n_mismatch or self.mapping or self.single_mapper
                    or self.measures)


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
    matched = _pair_datasets(study.datasets, check_datasets)
    for index, dataset in enumerate(study.datasets):
        position = matched.get(index)
        if position is None:
            # C7: neither agent's list position says anything — this dataset is one the cross-check
            # has no counterpart for, by IDENTITY. It is an INCLUSION question, settled at the map
            # stage against a named protocol rule, and not a warning to be read after the fact:
            # Bock's tracking-only control sample was mapped by the primary alone, rejected in
            # prose by the cross-check, and still produced a fully signed d = -2.9610 because
            # nothing turned the objection into a decision.
            conflicts.single_mapper.append(index)
            disagreements.append(f"dataset {dataset.dataset_id} ({dataset.label or 'unlabelled'}) "
                                 f"has no counterpart in the cross-check's map")
            flags.append(_unmatched_dataset_flag(dataset.dataset_id))
            continue
        _diff_dataset(index, dataset, check_datasets[position], conflicts, outcome_keys, ids,
                      disagreements, flags)
    paired_positions = set(matched.values())
    for position, raw in enumerate(check_datasets):
        if position not in paired_positions:
            _report_extra_dataset(raw, position, disagreements, flags)
    return conflicts


#: below this, two agents' descriptions of a contrast are not the same contrast. Pairing by
#: position blocked the WRONG dataset whenever the mapper listed its datasets in another order
#: (F3), and pairing two different contrasts is worse than not pairing them: `_diff_dataset`
#: would compare their group labels and their Ns and an adjudicated n could be adopted from that
#: comparison. Measured on the recorded cross-checks of the nine-paper run, the same contrast
#: described by two agents scores 0.69-1.00 and a different contrast of the same paper 0.39-0.56.
DATASET_MATCH = 0.5
#: words that carry no identity: every dataset description is full of them
_IDENTITY_STOPWORDS = frozenset({"the", "a", "an", "of", "to", "and", "or", "vs", "versus", "in",
                                 "on", "with", "for", "by", "at", "day", "single", "experiment"})


def _identity_words(*parts: str) -> frozenset[str]:
    """The content words of a dataset description, singularised — its identity as a bag of words."""
    text = " ".join(part or "" for part in parts).lower()
    words = [w for w in re.split(r"[^a-z0-9]+", text)
             if w and w not in _IDENTITY_STOPWORDS]
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


def _dataset_similarity(dataset: DatasetSpec, raw: dict[str, Any]) -> float:
    """0..1 — how much two agents' descriptions look like the same contrast.

    Two measures, and the stronger one counts: how much of the shorter description's vocabulary
    the other one contains (two agents writing at different lengths about the same contrast —
    "Pointing adaptation to +60° rotation: old vs young" and "Bock (2005) Experiment 1 — Pointing
    adaptation to 60° visuomotor rotation" — share almost every content word), and how alike the
    three fields both schemas ask for read one by one (which separates two experiments of one
    paper that share most of their words but differ in their own labels).

    When neither side described the contrast in words at all, the group labels are the only
    identity left; when there are none of those either, there is no identity and no match — a
    question, never a pairing on nothing.
    """
    mine = (dataset.label, dataset.experiment, dataset.condition)
    theirs = (raw.get("label") or "", raw.get("experiment") or "", raw.get("condition") or "")
    scored = [_sim(a, b) for a, b in zip(mine, theirs) if _norm(a) and _norm(b)]
    if not scored:
        groups = [(dataset.group_a.label, (raw.get("group_a") or {}).get("label") or ""),
                  (dataset.group_b.label, (raw.get("group_b") or {}).get("label") or "")]
        scored = [_sim(a, b) for a, b in groups if _norm(a) and _norm(b)]
        return sum(scored) / len(scored) if scored else 0.0
    words_a, words_b = _identity_words(*mine), _identity_words(*theirs)
    shared = (len(words_a & words_b) / min(len(words_a), len(words_b))
              if words_a and words_b else 0.0)
    return max(shared, sum(scored) / len(scored))


def _pair_datasets(datasets: Sequence[DatasetSpec],
                   check_datasets: Sequence[dict[str, Any]]) -> dict[int, int]:
    """`{index into the map: index into the cross-check's list, for the same contrast}`.

    Best-first, one counterpart each: the strongest pair is taken, then the next strongest of
    what is left. Order is never used — "the dataset at position 2" is not an identity, and
    nothing about a model's output order is stable across runs.
    """
    scores = sorted(((_dataset_similarity(dataset, raw), index, position)
                     for index, dataset in enumerate(datasets)
                     for position, raw in enumerate(check_datasets)),
                    key=lambda item: (-item[0], item[1], item[2]))
    paired: dict[int, int] = {}
    taken: set[int] = set()
    for score, index, position in scores:
        if score < DATASET_MATCH or index in paired or position in taken:
            continue
        paired[index] = position
        taken.add(position)
    return paired


def _unmatched_dataset_flag(dataset_id: str) -> str:
    """The `needs_human` line for a dataset the cross-check has no counterpart for — written once
    and cleared once, so answering the inclusion question does not leave it standing."""
    return (f"dataset {dataset_id}: group mapping unconfirmed (the cross-check has no counterpart "
            f"dataset) — needs human")


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


# ----------------------------------------------------------------------------- one measure (C6)
#: words a mapper uses when its own text is naming a SECOND measure rather than describing one
ALTERNATIVE_MARKERS: tuple[str, ...] = ("alternatively", "candidate operationalization",
                                        "candidate operationalisation", "two candidate",
                                        "either of two", "or, alternatively")


def named_alternatives(outcome: OutcomeSources) -> str:
    """The marker in the mapper's own words that says this outcome has two measures, or `""`."""
    text = f"{outcome.measure_name} {outcome.operationalization}".lower()
    return next((marker for marker in ALTERNATIVE_MARKERS if marker in text), "")


#: The one member of `AnalysisMetric` that names a UNIT rather than a quantity: a reading divided
#: by the size of the perturbation the study imposed, written as a percentage of it. The other
#: three each name a different QUANTITY — `endpoint` a level, `change_from_baseline` a difference
#: from this group's own pre-manipulation reading, `baseline_corrected` a reading with a control
#: condition subtracted — and swapping one for another changes what was measured, which is what a
#: `which_measure` decision is for. Expressing any of those quantities as a percentage of the
#: perturbation changes only the scale it is written on: it is the same measurement, so the verify
#: stage treats the same marks read against a degree axis and a percent axis as corroboration
#: rather than a dispute (`unit_other_expression`, commit `4c59649`), and C6 must not demote one
#: of them and buy an adjudication call to settle a question nobody asked.
RE_EXPRESSION_METRIC = "percent_of_perturbation"


def _metric_of(source: Source) -> str:
    return str(getattr(source.analysis_metric, "value", source.analysis_metric))


def _value_metrics(outcome: OutcomeSources) -> list[tuple[Source, str]]:
    """Every `value` location of this outcome that recorded a metric, with that metric.

    `value` only: a `baseline` or `context` location beside the value is not a rival measure, and
    an `alternate` is one C6 already settled.
    """
    return [(source, metric) for source in outcome.sources
            if source.role == "value" and (metric := _metric_of(source)) != "unknown"]


def _metricless_values(outcome: OutcomeSources) -> list[Source]:
    """`value` locations that did not say which measure they read.

    These are UNRESOLVED, not agreement. Extraction reads `role in ("value", "unknown")`, so a
    location whose `analysis_metric` the mapper left blank was read for the cell's number while
    C6 could neither see it nor demote it: after a winner ruling on Heuer's `d1 late_adaptation`
    the losing operationalization's own paragraph was still read, through exactly such a sibling.
    """
    return [source for source in outcome.sources
            if source.role == "value" and _metric_of(source) == "unknown"]


def _element_key(source: Source) -> tuple[str, str] | None:
    """The plotted element a source points at — `None` when it points at prose.

    The same figure/table id AND the same panel/row qualifier means *the same marks*: this is the
    key `_match_source` already uses to decide that two panels of one figure are two locations.
    """
    ident = source.figure_id or source.table_id
    return (ident, _qualifier(source.locator)) if ident else None


def measure_of(source: Source, outcome: OutcomeSources) -> str:
    """The MEASURE one `value` location reads: its metric, with a re-expression resolved to the
    quantity it re-expresses.

    Controller ruling on C6 vs `4c59649`, as a general rule about the `AnalysisMetric` enum: a
    percentage OF THE PERTURBATION is a unit, not a quantity, so a location that reports it is
    reporting one of the outcome's other readings in another unit — one measure written twice,
    not two candidate operationalizations. Cressman's Fig. 3b prints both on one pair of bars
    (left axis in degrees, right axis in percent); Bock prints the degrees off Fig. 1 and the
    percentage in a sentence on the next page ("adaptation magnitude A=(I-F)/I"). Neither is a
    question for a human, and neither may buy the adjudication call that settles a real dispute.

    Which quantity it re-expresses is answered from THE PAPER'S OWN LAYOUT, never inferred from
    the absence of alternatives: only the same plotted element read against its other axis says
    so. A percentage printed anywhere else is its own candidate measure and the `which_measure`
    question is asked.

    The rule used to have a second branch — "an outcome whose other `value` locations read
    exactly ONE quantity leaves nothing else the percentage could be a percentage of" — and the
    controller dropped it, because the label is not evidence about the denominator or about the
    sample: Bock's prose `A = (I - F)/I` divides by the measured initial error (so `d(A) != d(F)`
    whenever `I` differs between the groups) and computes it over the pooled seniors, and branch
    (b) merged it into the Fig. 1 reading on nothing but the absence of a third candidate.
    """
    metric = _metric_of(source)
    if metric != RE_EXPRESSION_METRIC:
        return metric
    readings = _value_metrics(outcome)
    quantities = {m for _, m in readings if m != RE_EXPRESSION_METRIC}
    if not quantities:
        return metric               # nothing else was read: the percentage IS this measure
    key = _element_key(source)
    same_marks = {m for other, m in readings
                  if m != RE_EXPRESSION_METRIC and key is not None and _element_key(other) == key}
    if len(same_marks) == 1:
        return same_marks.pop()
    return metric


def split_metrics(outcome: OutcomeSources) -> list[str]:
    """The distinct MEASURES the outcome's own `value` locations read (2+ is a split).

    Measures, not metric labels: `measure_of` has already folded a reading and its re-expression
    in another unit into one.
    """
    return sorted({measure_of(source, outcome) for source, _ in _value_metrics(outcome)})


def _diff_measures(study: StudyMap, conflicts: _Conflicts, disagreements: list[str]) -> None:
    """C6: an outcome whose map names two measures is a `which_measure` decision, taken now.

    `measure_name` and `operationalization` are single-measure fields. When the mapper's own text
    names alternatives, or two `value` locations under one outcome measure different quantities,
    the cell has two candidate answers and extracting both mixes them: the run's `metric_mixed`,
    `unit_mismatch` and `value_outside_axis` flags on Heuer's late-adaptation cell are what that
    looks like downstream. The choice belongs to the protocol's `definition` and
    `measurement_window`, and it is made once, here.
    """
    for index, dataset in enumerate(study.datasets):
        for outcome in dataset.outcomes:
            marker = named_alternatives(outcome)
            metrics = split_metrics(outcome)
            # a `value` location that recorded no metric is a candidate nobody has resolved: it
            # cannot say whether it reads this outcome's measure or the other one, and it is read
            # for the value unless something says otherwise (F6).
            metricless = _metricless_values(outcome) if metrics else []
            # The mapper's own words name an alternative — but a mapper that ALSO demoted the
            # alternative's locations to `alternate` and left one measure among its `value`
            # locations has answered its own question ("DE (primary); IEE (alternative)", IEE
            # marked alternate): buying an adjudication and blocking extraction on those words
            # settled nothing on the first nine-paper run. The words open the question only while
            # the map still carries two readable candidates.
            demoted = any(str(getattr(source, "role", "")) == "alternate"
                          for source in outcome.sources)
            words_open = bool(marker) and not (demoted and len(metrics) <= 1)
            if not words_open and len(metrics) < 2 and not metricless:
                continue
            why = []
            if words_open:
                why.append(f"the map's own words name an alternative ({marker!r})")
            if len(metrics) >= 2:
                why.append(f"its value locations measure {' and '.join(metrics)}")
            if metricless:
                why.append(f"{len(metricless)} value location(s) cannot say which measure they "
                           f"read ({'; '.join(_clip(s.locator, 60) for s in metricless)})")
            conflicts.measures.append((index, outcome.outcome_key))
            disagreements.append(f"{dataset.dataset_id} {outcome.outcome_key}: two candidate "
                                 f"measures — {'; '.join(why)}")


#: what the record says about a `value` location that never said which measure it reads, once the
#: outcome's measure has been settled by someone. Defined in `models` beside the note the
#: demotion writes, because the review page has to read both to offer a set-aside location back.
WITHHELD_NOTE = C6_WITHHELD_NOTE


@dataclass
class _Settlement:
    """One `which_measure` answer read against the outcome's own locations — never applied yet.

    The same reading serves the adjudicator's ruling and a human's answer, so the rule that
    decides what a settlement DOES exists once. `ok` is False when the answer names nothing this
    outcome carries (never invent); `settles` is False when applying it would leave the rival
    readings readable, which is not a settlement however valid the answer looks.
    """

    ok: bool = False
    reason: str = ""
    metric: str = ""                                          # the winning metric, for the record
    measure: str = ""                                         # the measure that won
    losers: list[tuple[Source, str]] = field(default_factory=list)     # (location, its metric)
    withheld: list[Source] = field(default_factory=list)      # `value` locations with no metric
    settles: bool = False


def read_measure_answer(outcome: OutcomeSources, *, winning_metric: str = "",
                        winning_location: str = "",
                        losing_locations: Sequence[str] = (),
                        keep_group_siblings: bool = False) -> _Settlement:
    """Read a `which_measure` answer against one outcome. Pure: nothing is changed here.

    A LOCATION decides whenever one is named, and only then the metric. Two operationalizations
    routinely share one `analysis_metric` — Heuer's Experiment 2 prints both of its candidates as
    `change_from_baseline` — so an answer that can only name the metric names both of them at
    once: it demotes nothing, leaves both readable, and the run then extracts the two readings
    the question exists to choose between. When a location is named, every other `value` location
    loses EXCEPT the winner's own other-axis expression of the same marks (the unit rule), because
    "read this one" is what the person answering said.

    Every location that carries no metric at all is withheld too: it cannot say whether it reads
    the winner or the loser, and a location that cannot say which measure it reads cannot be read
    as the winner's number.
    """
    readings = [(source, metric, measure_of(source, outcome))
                for source, metric in _value_metrics(outcome)]
    if not readings:
        return _Settlement(reason="this outcome has no value location that names a measure")
    settlement = _winner(readings, winning_metric, winning_location)
    if not settlement.ok and winning_location and winning_metric:
        # a location the map does not carry cannot be honoured, but the answer also named a
        # measure: read that instead of throwing the answer away (it settles only if it demotes)
        settlement, winning_location = _winner(readings, winning_metric, ""), ""
    if not settlement.ok:
        return settlement

    winners = _winner_sources(readings, settlement, winning_location)
    # identity, not equality: two locations with the same fields are still two locations, and a
    # pydantic model compares by value
    won = {id(source) for source in winners}
    #: the marks the winning locations point at: the same marks in another unit are the winner
    #: written twice (`measure_of` branch (a)), and nothing else survives a named location
    elements = {_element_key(source) for source in winners if _element_key(source) is not None}
    named_losers = [name for name in losing_locations if str(name).strip()]
    losers: list[tuple[Source, str]] = []
    #: the winners' own sample answers: a location naming ONE group is half of a two-arm cell, and
    #: the other half is the same measure read for the other group. Two panels of one figure — young
    #: in A, older in B — are the ordinary layout, and "read this one" said of one panel is not a
    #: refusal of the other: it is a choice of MEASURE, and the group it happens to be printed for
    #: is not part of that choice. Demoting the sibling narrows the cell to one arm, and a two-arm
    #: contrast that has lost an arm produces no effect size at all.
    #:
    #: It protects only what the answer did not name: an answer that says outright which readings
    #: lose is authority over exactly those, which is how a split whose two sides share one metric
    #: is settled at all (Langan prints DE in Fig. 1A/1B and IEE in Fig. 1C/1D — four one_group
    #: panels, one metric, two measures). Off unless the caller asks, so the adjudicator's rulings
    #: settle exactly what they settled before: a ruling that stops settling re-blocks its cell, and
    #: a cell nobody may read is not an improvement on a cell read the wrong way.
    keep_siblings = (keep_group_siblings
                     and any(str(getattr(s, "sample", "")) == "one_group" for s in winners))
    for source, metric, measure in readings:
        if winning_location:
            wins = id(source) in won or (measure == settlement.measure
                                         and _element_key(source) in elements) \
                or (keep_siblings and measure == settlement.measure
                    and str(getattr(source, "sample", "")) == "one_group"
                    and not any(_names_location(source, name) for name in named_losers))
        else:
            wins = measure == settlement.measure
        if wins and named_losers and not (winning_location and id(source) in won):
            # the answer may also name the losing locations outright — the way to settle a split
            # whose two sides share one metric. It never demotes the winner's own location.
            wins = not any(_names_location(source, name) for name in named_losers)
        if not wins:
            losers.append((source, metric))
    settlement.losers = losers
    settlement.withheld = _metricless_values(outcome)
    remaining = len(readings) - len(losers)
    if remaining <= 0:
        # "the measure is X" has to leave X readable somewhere; an answer that demotes every
        # location is a deletion, and C6 demotes, it never deletes
        settlement.reason = ("the answer would demote every value location, leaving nothing for "
                             "the cell to be read from")
        settlement.settles = False
        return settlement
    # A settlement that leaves the rival readings readable has settled nothing. One MEASURE left
    # standing IS a settlement — however many locations read it (a young-adults panel and an
    # older-adults panel are two locations of one measure) — but two measures still readable
    # means the map still carries two candidate answers and the question stays open.
    lost = {id(source) for source, _ in losers}
    measures_left = {measure for source, _, measure in readings if id(source) not in lost}
    # …EXCEPT when the mapper's own words opened the question and nothing was ever demoted:
    # then two operationalizations may share one metric (Heuer's Experiment 2 prints both as
    # `change_from_baseline`), one metric-class left is not one measure, and only a demotion
    # settles it (the F2 rule — settle by location).
    words_open = bool(named_alternatives(outcome)) and not any(
        str(getattr(source, "role", "")) == "alternate" for source in outcome.sources)
    settlement.settles = (bool(losers) or remaining <= 1
                          or (len(measures_left) <= 1 and not words_open))
    return settlement


def _winner(readings: list[tuple[Source, str, str]], winning_metric: str,
            winning_location: str) -> _Settlement:
    """Which measure the answer names, or why it names none. The location decides when given."""
    if winning_location:
        matched = [(metric, measure) for source, metric, measure in readings
                   if _names_location(source, winning_location)]
        if not matched:
            return _Settlement(reason=f"no value location of this outcome is at "
                                      f"{winning_location!r}")
        measures = {measure for _, measure in matched}
        metrics = {metric for metric, _ in matched}
        if len(measures) != 1 or len(metrics) != 1:
            return _Settlement(reason=f"{winning_location!r} names locations that read "
                                      f"{' and '.join(sorted(measures))}")
        return _Settlement(ok=True, metric=metrics.pop(), measure=measures.pop())
    if not winning_metric:
        return _Settlement(reason="the answer names neither a measure nor a location")
    carried = sorted({metric for _, metric, _ in readings})
    if winning_metric not in carried:
        return _Settlement(reason=f"it names {winning_metric!r}, which none of the value "
                                  f"locations carries ({', '.join(carried) or 'none'})")
    won = {measure for _, metric, measure in readings if metric == winning_metric}
    if len(won) != 1:
        return _Settlement(reason=f"it names {winning_metric!r}, which this outcome's locations "
                                  f"read as {' and '.join(sorted(won))}")
    return _Settlement(ok=True, metric=winning_metric, measure=won.pop())


def _winner_sources(readings: list[tuple[Source, str, str]], settlement: _Settlement,
                    winning_location: str) -> list[Source]:
    if winning_location:
        return [source for source, _, _ in readings if _names_location(source, winning_location)]
    return [source for source, metric, _ in readings if metric == settlement.metric]


#: what the record says when a person takes a settled `which_measure` decision again. Appended, never substituted:
#: the reason a location was set aside is evidence about the paper whether or not this review ends
#: up reading it, so a reopened location carries both notes and a reader can see the whole history.
REOPENED_NOTE = ("put back as a candidate: the which_measure decision that demoted it is being "
                 "taken again")


def c6_demoted(source: Source) -> bool:
    """Was this location set aside by a `which_measure` decision — and so by a reversible one?

    An `alternate` the MAPPER itself wrote (a paper's own "DE (primary); IEE (alternative)") is the
    map's reading of the paper, not a decision anybody took about this review. C6 never made it, and
    an answer to a C6 question does not undo it.
    """
    return (str(getattr(source, "role", "")) == "alternate"
            and c6_demoted_note(source.notes or ""))


def reopened_outcome(outcome: OutcomeSources) -> OutcomeSources:
    """A COPY of this outcome with its `which_measure` demotions undone.

    The candidates as they stood before anyone chose between them. Without this a settled ruling is
    a decision nobody can take again: `_value_metrics` reads `value` locations only, so the reading
    C6 demoted is invisible to `read_measure_answer`, and an answer naming it comes back "which none
    of the value locations carries" — which is what a reviewer disagreeing with the tool's own
    choice was told on every one of the 25 measures it settled for itself on this corpus.

    A copy, because an answer that settles nothing must leave the map exactly as it was rather than
    half-reopened with two readings readable again.
    """
    copy = outcome.model_copy(deep=True)
    for source in copy.sources:
        if c6_demoted(source):
            source.role = "value"
            source.notes = _note(source.notes, REOPENED_NOTE)
    return copy


def apply_measure_settlement(outcome: OutcomeSources, settlement: _Settlement,
                             why: Callable[[str], str]) -> None:
    """Demote what the settlement demotes and record the winner. `why(metric)` writes the note.

    The losers are kept on the record with the reason they lost — C6 demotes, it never deletes —
    and the outcome's own `analysis_metric` becomes the winner's.
    """
    for source, metric in settlement.losers:
        source.role = "alternate"
        source.notes = _note(source.notes, f"{C6_DEMOTION_NOTE}: {why(metric)}")
    for source in settlement.withheld:
        source.role = "alternate"
        source.notes = _note(source.notes, WITHHELD_NOTE)
    outcome.analysis_metric = settlement.metric


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

    _apply_inclusion_rulings(study, adjudicated.get("dataset_inclusions") or [], conflicts,
                             disagreements, flags)
    _apply_measure_rulings(study, adjudicated.get("measure_rulings") or [], conflicts,
                           disagreements, flags)
    _apply_error_bar_rulings(study, adjudicated.get("error_bar_rulings") or [], conflicts,
                             disagreements)


def _apply_inclusion_rulings(study: StudyMap, rulings: list[dict[str, Any]], conflicts: _Conflicts,
                             disagreements: list[str], flags: list[str]) -> None:
    """C7: settle each single-mapper dataset. Reject ONLY on a named protocol rule, with a quote.

    A rejection that cites no rule, or quotes nothing from the paper, is not a rejection: it leaves
    the dataset open and a person answers. The asymmetry is deliberate — an adjudicator that cannot
    name the rule it is applying is guessing, and a guess that deletes a dataset is worse than a
    question.
    """
    for ruling in rulings:
        index = ruling.get("primary_dataset_index")
        position = int(index) - 1 if isinstance(index, int) else -1
        if position not in conflicts.single_mapper:
            if position < 0 or position >= len(study.datasets):
                flags.append(f"the adjudicator ruled on the inclusion of dataset {index!r}, which "
                             f"is not one of the datasets a single agent proposed — ignored")
            continue
        dataset = study.datasets[position]
        verdict = (ruling.get("verdict") or "").strip()
        rule = (ruling.get("rule") or "").strip()
        quote = (ruling.get("quote") or "").strip()
        if verdict == "exclude" and rule and quote:
            dataset.included = False
            dataset.exclusion_rule = rule
            dataset.exclusion_quote = quote
            dataset.notes = _note(dataset.notes,
                                  f"excluded at the map stage under {rule!r}: {quote}")
            disagreements.append(f"adjudicated {dataset.dataset_id} inclusion: EXCLUDED under "
                                 f"{rule!r} — {_clip(quote, 160)}")
            conflicts.single_mapper.remove(position)
            continue
        if verdict == "exclude":
            disagreements.append(f"adjudicated {dataset.dataset_id} inclusion: exclusion cited "
                                 f"{'no protocol rule' if not rule else 'no quote from the paper'}"
                                 f" — not applied")
            continue                          # stays open: a question, not a deletion
        if verdict == "include":
            dataset.notes = _note(dataset.notes,
                                  f"kept at the map stage{f' under {rule!r}' if rule else ''}"
                                  + (f": {quote}" if quote else ""))
            disagreements.append(f"adjudicated {dataset.dataset_id} inclusion: kept"
                                 + (f" under {rule!r}" if rule else ""))
            conflicts.single_mapper.remove(position)


def _apply_measure_rulings(study: StudyMap, rulings: list[dict[str, Any]], conflicts: _Conflicts,
                           disagreements: list[str], flags: list[str]) -> None:
    """C6: one outcome, one measure — the winner keeps `value`, the loser becomes `alternate`.

    A ruling counts only when it names the winner AND quotes the paper for winner and loser, and
    only when applying it actually settles something: a `winner` that demotes no location has
    left both readings in place, which is not an answer to "which of these two?". Anything else
    (a `toss_up`, a ruling with no quotes, a measure no location carries, a metric both
    operationalizations share) leaves the outcome open, and an open outcome buys no extraction.
    """
    for ruling in rulings:
        index = ruling.get("dataset_index")
        position = int(index) - 1 if isinstance(index, int) else -1
        key = (ruling.get("outcome_key") or "").strip()
        if (position, key) not in conflicts.measures:
            if position < 0 or position >= len(study.datasets):
                flags.append(f"the adjudicator ruled on the measure of dataset {index!r} "
                             f"{key!r}, which the map did not put in question — ignored")
            continue
        dataset = study.datasets[position]
        outcome = next((o for o in dataset.outcomes if o.outcome_key == key), None)
        if outcome is None:
            continue
        verdict = (ruling.get("verdict") or "").strip()
        winner_quote = (ruling.get("winner_quote") or "").strip()
        loser_quote = (ruling.get("loser_quote") or "").strip()
        if verdict != "winner" or not (winner_quote and loser_quote):
            disagreements.append(
                f"{dataset.dataset_id} {key} which_measure: not settled "
                f"({verdict or 'no verdict'}"
                f"{'' if winner_quote and loser_quote else ', quotes missing'})")
            continue
        settlement = read_measure_answer(
            outcome,
            winning_metric=(ruling.get("winning_analysis_metric") or "").strip(),
            winning_location=(ruling.get("winning_location") or "").strip(),
            losing_locations=[str(x) for x in (ruling.get("losing_locations") or [])])
        if not settlement.ok:
            disagreements.append(f"{dataset.dataset_id} {key} which_measure: {settlement.reason} "
                                 f"— not applied")
            continue
        if not settlement.settles:
            disagreements.append(
                f"{dataset.dataset_id} {key} which_measure: "
                + (settlement.reason or f"the ruling for {settlement.metric!r} demoted no "
                                        f"location, so both readings are still read")
                + " — it has settled nothing and the question stays open (name the winning "
                  "location, or the losing ones)")
            continue
        apply_measure_settlement(
            outcome, settlement,
            lambda metric, winner=settlement.metric: (
                f"this outcome's measure is {winner} ({_clip(winner_quote, 120)}); this location "
                f"measures {metric} ({_clip(loser_quote, 120)})"))
        outcome.measure_ruling = (f"measure: {settlement.metric}. winner: {winner_quote} | loser: "
                                  f"{loser_quote}"
                                  + (f" | {ruling.get('rationale')}" if ruling.get("rationale")
                                     else ""))
        disagreements.append(f"adjudicated {dataset.dataset_id} {key} which_measure: "
                             f"{settlement.metric} ({len(settlement.losers)} location(s) demoted "
                             f"to alternate"
                             + (f", {len(settlement.withheld)} withheld for naming no measure)"
                                if settlement.withheld else ")"))
        conflicts.measures.remove((position, key))


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
              pdf_file_id: str | None = None, reviewer_ruling: str = "") -> StudyMap:
    """Map one paper against one protocol: eligibility, datasets, group Ns and source locations.

    `reviewer_ruling` is §C3's other half: the one line that says a PERSON has already decided
    this paper belongs in the review (`llm.context.reviewer_ruling_line`). It rides additively in
    the two prompts that vote on eligibility, so the mapper is asked to map a paper rather than to
    judge one — and because it is prompt text, the call has a cache key of its own and cannot come
    back as the cached "not eligible, no datasets" answer the reviewer was overruling.
    """
    sha12 = paper.sha256[:12]
    ruling = ("\n\n" + reviewer_ruling) if reviewer_ruling else ""
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
            "mapper", PROTOCOL=protocol_prompt, ROSTER=roster_prompt) + ruling)]}])
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
                "mapper", PROTOCOL=protocol_prompt, ROSTER=roster_prompt) + ruling)]}])
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
            "mapper_crosscheck", PROTOCOL=protocol_prompt, ROSTER=roster_prompt) + ruling)]}])
    call_ids.append(check.call_id)
    checked = check.parsed or {}
    labels = [(d.group_a.label, d.group_b.label) for d in study.datasets]
    conflicts = _diff(study, checked, outcome_keys, ids, disagreements, flags)
    _agree_error_bars(study, _roster_determinations(checked), conflicts, disagreements, flags)
    _flag_thin_outcomes(study, flags)
    _diff_measures(study, conflicts, disagreements)          # C6, before anything is extracted

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

    # C7 / C6: what adjudication did not settle becomes a QUESTION, and a question blocks the
    # extraction of the cell it names. Nothing here deletes data: the dataset and both measures
    # stay on the record with the words that put them in doubt.
    for position in conflicts.single_mapper:
        dataset = study.datasets[position]
        study.open_questions.append(MapQuestion(
            kind="include_dataset", dataset_id=dataset.dataset_id,
            # what the map KNOWS is that the cross-check's map contains no dataset describing this
            # contrast; which agent "proposed" what is not something the diff can establish.
            question=(f"The cross-check's map has no counterpart for "
                      f"{dataset.label or dataset.dataset_id!r} "
                      f"({dataset.experiment or 'no experiment label'}; "
                      f"{dataset.condition or 'no condition'}), and no protocol rule was cited to "
                      f"exclude it. Does this review include it?"),
            options=["include it", "exclude it"],
            quotes=[q for q in (dataset.chosen_pair_rationale, dataset.group_a.n_evidence,
                                dataset.group_b.n_evidence) if q]))
        flags.append(_question_flag("include_dataset", dataset.dataset_id, ""))
    for position, key in conflicts.measures:
        dataset = study.datasets[position]
        outcome = next((o for o in dataset.outcomes if o.outcome_key == key), None)
        study.open_questions.append(MapQuestion(
            kind="which_measure", dataset_id=dataset.dataset_id, outcome_key=key,
            question=(f"The map names two measures for {key} in "
                      f"{dataset.label or dataset.dataset_id!r}. Which one does this review's "
                      f"definition and measurement window ask for?"),
            options=_measure_options(outcome),
            quotes=[q for q in ((outcome.measure_name if outcome else ""),
                                (outcome.operationalization if outcome else "")) if q]))
        flags.append(_question_flag("which_measure", dataset.dataset_id, key))
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


def extraction_blocks(study: StudyMap) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """`(datasets, cells)` the map says must not be extracted, each with the reason why.

    Two sources, both decided at the map stage and both refusing to spend rather than to guess:
    a dataset the adjudicator excluded on a named protocol rule (C7, `included=False`), and any
    cell named by an open map question — `include_dataset` when the inclusion of a single-mapper
    dataset is unsettled, `which_measure` when one outcome still carries two measures (C6).
    """
    datasets: dict[str, str] = {}
    cells: dict[tuple[str, str], str] = {}
    for dataset in study.datasets:
        if dataset.included:
            continue
        # the rule, or a sentence saying there is none: "excluded under ''" names nothing a
        # reviewer can check the exclusion against, and the honest message is only as useful as
        # the rule it prints.
        rule = (dataset.exclusion_rule or "").strip()
        datasets[dataset.dataset_id] = (
            (f"the map excluded this dataset under {rule!r}" if rule
             else "the map excluded this dataset under a protocol rule it does not record")
            + (f": {dataset.exclusion_quote}" if dataset.exclusion_quote else ""))
    for question in study.open_questions:
        why = f"an unanswered {question.kind} question: {question.question}"
        if question.outcome_key:
            cells[(question.dataset_id, question.outcome_key)] = why
        else:
            datasets.setdefault(question.dataset_id, why)
    return datasets, cells


def unreadable_cell(study: StudyMap, dataset_id: str, outcome_key: str,
                    outcome_keys: Collection[str] = ()) -> str:
    """Why no extract stage can be sent to this cell — `""` when one can.

    C4's honesty clause for the one answer whose consequence is a READING. A `re_extract` hint on
    a cell the map puts out of reach buys nothing on this resume and nothing on any later one, and
    telling the reviewer "the next --resume re-reads this cell" is then a promise nothing can keep:
    Roller's E1a and E3 hints sat on that line through every resume of a run that was never going
    to read them, because the map had excluded both datasets under a protocol rule and E3's map
    carried no source for the outcome at all.

    ONE rule, here beside `extraction_blocks` whose answer it reads, so the run's warning and the
    review page's pending line can never disagree about what is in the way.
    """
    if study.eligible is False:
        # a dataset-level ruling never names the PAPER's eligibility: a paper the screen threw
        # out stays out until the eligibility card that already exists says otherwise, and a
        # typed value or a hint on one of its cells is refused with the same sentence
        return ("the screen excluded this paper from the review; the paper's own eligibility "
                "question is the answer that lifts this, not a dataset ruling")
    dataset = next((d for d in study.datasets if d.dataset_id == dataset_id), None)
    if dataset is None:
        return f"no dataset {dataset_id!r} in this run's map of the paper"
    if outcome_keys and outcome_key not in outcome_keys:
        return (f"{outcome_key!r} is not an outcome of this review's protocol, so no reader is "
                f"ever sent to it")
    if all(sources.outcome_key != outcome_key for sources in dataset.outcomes):
        return (f"the map lists no {outcome_key!r} source for {dataset_id}, so there is no cell "
                f"here for a reader to be sent to")
    datasets, cells = extraction_blocks(study)
    return datasets.get(dataset_id) or cells.get((dataset_id, outcome_key)) or ""


def _measure_options(outcome: OutcomeSources | None) -> list[str]:
    """One option per candidate READING, not per metric label.

    Two operationalizations can share one `analysis_metric` (Heuer's Experiment 2 prints both as
    `change_from_baseline`), and a question whose only option is the metric they share cannot be
    answered: whoever answers it has to be able to name WHERE the winner is. A location that
    recorded no metric is offered too — it is a candidate nobody has resolved.
    """
    if outcome is None:
        return ["the first measure named", "the second measure named"]
    options = [f"{metric} — {_clip(source.locator, 90)}"
               for source, metric in _value_metrics(outcome)]
    options += [f"no measure recorded — {_clip(source.locator, 90)}"
                for source in _metricless_values(outcome)]
    return options or ["the first measure named", "the second measure named"]


def open_map_questions(study: StudyMap) -> list[MapQuestion]:
    """The map questions still waiting for a person — the list the review UI offers to answer.

    Copies: a pure reader that hands out the study's own models lets its caller edit the map
    through it, and this list is read by the review UI on every refresh.
    """
    return [question.model_copy(deep=True) for question in study.open_questions]


#: `Source.sample` answers that are not the two groups this contrast compares. `unknown` and
#: `both_groups` are read — `unknown` is the honest absence every map written before the field
#: existed carries, and refusing on it would stop reading every paper in the corpus.
#: A location whose sample is NOT the two groups this contrast compares is not read for the value:
#: a pooled analysis (Bock's `A=(I-F)/I` over the pooled seniors) or some other sample. A location
#: that carries ONE of the two groups is readable — a figure whose panels are one age group each
#: (Fig. 1A young / Fig. 1B old) is the ordinary layout, and the reader is asked for each group
#: separately; the run's first nine-paper pass read `one_group` as unreadable and extracted
#: NOTHING from such a paper, with zero calls and no question, so the rule is written here.
#: (defined in `models` beside `SourceSample`, because the review page must refuse to offer such a
#: location and may not import this package)
#: the roles a number may be read at: `value` is the outcome's own number, `unknown` is a role the
#: mapper did not fill in. `baseline`, `context` and `alternate` are on the record for a reader.
READABLE_ROLES: frozenset[str] = frozenset({"value", "unknown"})


def source_unreadable_reason(source: Source) -> str:
    """Why this location must not be read for its cell's value — `""` when it may be read.

    Two reasons, decided in one place so the pipeline and the map agree: what the location IS for
    this outcome (`role`), and WHOSE numbers are at it (`sample`). Cressman's aligned-cursor
    curves were digitised as late adaptation (3.9° beside the misaligned curves' 31.4°) for want
    of the first; Bock's adaptation magnitude `A=(I-F)/I` is computed over the pooled seniors and
    sat in the map as a `value` location of a two-group cell for want of the second.
    """
    role = str(getattr(source, "role", "") or "unknown")
    if role not in READABLE_ROLES:
        return (f"a {role} source — kept for the record, not read for the value")
    sample = str(getattr(source, "sample", "") or "unknown")
    if sample in UNREADABLE_SAMPLES:
        note = _clip(str(getattr(source, "sample_note", "") or ""), 120)
        return (f"reports a {sample} sample, not the two groups this contrast compares — kept "
                f"for the record, not read for the value" + (f" ({note})" if note else ""))
    return ""


def readable_sources(sources: Sequence[Source]) -> list[Source]:
    """The locations of one outcome a number may be read at (`source_unreadable_reason` == "")."""
    return [source for source in sources if not source_unreadable_reason(source)]


#: how the record names a person who answered a map question (never a model name). Defined in
#: `models` beside the other names a map decision writes, because the review page prints them
#: and may not import this package.
HUMAN_DECIDER = HUMAN_DECIDER_NAME
#: …and how it names the model that rules when nobody has answered. The exclusions table needs
#: both names, and the two are told apart by the review LOG — the record of who decided — never
#: by the shape of the rule that was cited (re-review N2).
MAP_ADJUDICATOR = MAP_ADJUDICATOR_NAME
#: the rule an exclusion cites when the person who made it cited none. A person may exclude a
#: dataset without quoting the protocol at an adjudicator's standard, but the record must not
#: claim a rule nobody named (C7 holds the ADJUDICATOR to a named rule; this is not that path).
HUMAN_EXCLUSION_RULE = "human decision"
_ANSWER_KINDS: tuple[str, ...] = ("include_dataset", "which_measure")


def _question_flag(kind: str, dataset_id: str, outcome_key: str) -> str:
    """The `needs_human` line that goes with one map question — written once, cleared once.

    Both writers share this text so that answering a question removes exactly the flag the
    question wrote, and a wording change cannot leave a stale flag standing after the answer.
    """
    if kind == "include_dataset":
        return (f"dataset {dataset_id}: proposed by one mapping agent only and no "
                f"protocol rule was cited to exclude it — inclusion needs human, nothing "
                f"extracted until then")
    return (f"dataset {dataset_id} {outcome_key}: two candidate measures and no ruling "
            f"that quotes both — which_measure needs human, nothing extracted until then")


def map_answer_key(answer: Mapping[str, Any]) -> tuple[str, str, str]:
    """The question one map answer addresses: `(kind, dataset_id, outcome_key)`.

    One rule, so `apply_map_answers` and everything that asks it what it applied name the same
    question. Only `which_measure` is per-outcome — an inclusion is a decision about the whole
    dataset, and keying it by an outcome would make two answers to one question look like two
    questions.
    """
    kind = str(answer.get("kind") or "").strip()
    return (kind, str(answer.get("dataset_id") or "").strip(),
            str(answer.get("outcome_key") or "").strip() if kind == "which_measure" else "")


def _answers_this_study(study: StudyMap, answer: Mapping[str, Any]) -> bool:
    """A record addresses this study when it names its paper id, whole or by the sha12 prefix the
    run uses in every dataset id. A record that names no paper is not addressed to one."""
    paper_id = str(answer.get("paper_id") or "").strip()
    return bool(paper_id) and (study.paper_id == paper_id
                               or (len(paper_id) >= 12 and study.paper_id.startswith(paper_id)))


def _names_location(source: Source, location: str) -> bool:
    """Does `location` name this source — its figure/table id, or its locator?"""
    wanted = _norm(location)
    if not wanted:
        return False
    ids = {_norm(ident) for ident in (source.figure_id, source.table_id) if ident}
    locator = _norm(source.locator)
    return wanted in ids or (bool(locator) and wanted in locator)


def _overrules_an_exclusion(study: StudyMap, kind: str, answer: Mapping[str, Any]) -> bool:
    """Is this an inclusion for a dataset the MAP excluded, which asked no question about it?

    §C3 one level down. A dataset the map adjudicator threw out on a protocol rule carries no open
    question, so `apply_map_answers` ignored an `include_dataset` answer for it and there was no
    record a reviewer could write that would ever put it back: the run refuses to re-read its
    cells and tells them why, and the reason names an answer that did nothing. A person overruling
    the mapper's reading of the protocol is exactly what the eligibility card already is for a
    whole paper, and this is the same decision about one contrast of it.

    Only an INCLUSION, and only over an exclusion the map itself made: "exclude it" needs an open
    question, because excluding a dataset nobody proposed excluding answers nothing, and an
    inclusion for a dataset already included changes nothing either.
    """
    if kind != "include_dataset" or str(answer.get("decision") or "").strip().lower() != "include":
        return False
    dataset_id = str(answer.get("dataset_id") or "").strip()
    return any(d.dataset_id == dataset_id and d.included is False for d in study.datasets)


def settled_measure(study: StudyMap, dataset_id: str, outcome_key: str) -> str:
    """The `which_measure` ruling this map settled for one cell on its own — `""` if it settled
    none. The one record that says a choice between two measures was MADE here."""
    return next((o.measure_ruling.strip()
                 for d in study.datasets if d.dataset_id == dataset_id
                 for o in d.outcomes
                 if o.outcome_key == outcome_key and (o.measure_ruling or "").strip()), "")


def _overrules_a_measure_ruling(study: StudyMap, kind: str, answer: Mapping[str, Any]) -> bool:
    """Is this a `which_measure` answer for an outcome the map already settled by itself?

    §C6's half of `_overrules_an_exclusion`, and the same hole. A settled ruling CLOSES the
    question, so the answer named no open question and `apply_map_answers` ignored it; and because
    settling had already demoted the rival reading out of `_value_metrics`, there was no record a
    reviewer could write that would ever put the other measure back. On the corpus this tool was
    validated against the map settled 25 measures for itself, asked about none of them, and gave
    the same cell opposite answers on two runs of the same paper — a choice that moves an effect
    size by an order of magnitude, taken silently and unappealably.

    Only a cell the map itself ruled on: an answer about an outcome nobody chose between answers
    nothing, and `_answer_measure` still refuses anything that names a reading the map never made.
    """
    return kind == "which_measure" and bool(settled_measure(
        study, str(answer.get("dataset_id") or "").strip(),
        str(answer.get("outcome_key") or "").strip()))


def _answer_inclusion(dataset: DatasetSpec, answer: Mapping[str, Any]) -> tuple[bool, str]:
    """C7 answered by a person: include it, or exclude it and say on what.

    `(False, why)` means the record was not an answer, and nothing about the dataset changed. The
    reason travels because a reviewer whose answer changed nothing is owed the sentence saying so.
    """
    decision = str(answer.get("decision") or "").strip().lower()
    note = str(answer.get("note") or "").strip()
    if decision not in ("include", "exclude"):
        return False, "the record names neither include nor exclude"
    if decision == "include":
        dataset.included = True
        # …and the exclusion it overrules goes with it: a dataset the review includes may not keep
        # the rule and the quote it was thrown out under, or `extraction_blocks` would keep
        # refusing to read a cell a person has just put back.
        dataset.exclusion_rule = ""
        dataset.exclusion_quote = ""
        dataset.notes = _note(dataset.notes, f"included by {HUMAN_DECIDER}"
                                             + (f": {note}" if note else ""))
        return True, ""
    dataset.included = False
    dataset.exclusion_rule = str(answer.get("rule") or "").strip() or HUMAN_EXCLUSION_RULE
    dataset.exclusion_quote = str(answer.get("quote") or "").strip()
    dataset.notes = _note(dataset.notes, f"excluded by {HUMAN_DECIDER} under "
                                         f"{dataset.exclusion_rule}"
                                         + (f": {note}" if note else ""))
    return True, ""


def confirms_the_measure(outcome: OutcomeSources, answer: Mapping[str, Any]) -> bool:
    """Does this answer AGREE with the measure this outcome already reads?

    Agreement names the measure and no location. A location is authority over which readings
    survive — naming one demotes every other — so an answer that means "yes, that one" must not
    carry one, and the option a review page offers for agreement must not either.

    Read against the outcome as it stands, so the same test serves both directions: before a resume
    a reversal disagrees with the map and is owed a reading; once the resume has applied it the map
    reads what the answer named and the same answer is agreement. Nothing has to remember which it
    was, which is what makes re-applying the whole log idempotent.
    """
    metric = str(answer.get("winning_analysis_metric") or "").strip()
    chosen = str(getattr(outcome.analysis_metric, "value", outcome.analysis_metric) or "").strip()
    # …and only where a decision has actually been TAKEN. On an outcome nobody has settled there is
    # nothing to agree with: `analysis_metric` is then the mapper's own guess beside two readable
    # measures, and treating "that one" as agreement would close the open question while leaving
    # both of them readable — the `metric_mixed` state, reached by answering the question that
    # exists to prevent it.
    return (bool((outcome.measure_ruling or "").strip())
            and bool(metric) and metric == chosen
            and not str(answer.get("winning_location") or "").strip()
            and not [str(x) for x in (answer.get("losing_locations") or []) if str(x).strip()])


def _answer_measure(dataset: DatasetSpec, outcome_key: str,
                    answer: Mapping[str, Any]) -> tuple[bool, str]:
    """C6 answered by a person: the losing `value` locations become `alternate`, as a ruling does.

    One rule, read once (`read_measure_answer`), so the two paths cannot drift apart. The answer
    may name the winning location or the winning metric; the review UI sends both, and the
    LOCATION decides, because a metric that both operationalizations carry cannot separate them.
    An answer that names something the map does not carry, that lands on two measures at once, or
    that would demote nothing while two readings stay readable, changes nothing and leaves the
    question open — a person's answer is authority over which measure the review wants, not a
    licence to write into the record a measure nobody read.

    A ruling the map already settled is REOPENED first (`reopened_outcome`), so the answer is read
    against every candidate the outcome ever carried rather than against the one the tool left
    standing. The tool's own choice between two measures is exactly the kind of decision a person
    is entitled to take again, and until this it was the one decision on the whole record that
    nothing a reviewer could write would change.
    """
    index, outcome = next(((i, o) for i, o in enumerate(dataset.outcomes)
                           if o.outcome_key == outcome_key), (-1, None))
    if outcome is None:
        return False, "this map has no such outcome"
    if confirms_the_measure(outcome, answer):
        # AGREEMENT CHANGES NOTHING. `read_measure_answer` cannot express "confirm": a metric-only
        # answer re-wins every location that reads that metric, so once the demotions are reopened
        # it puts back the very readings the ruling set aside (one Vachon cell went from one
        # readable location to three), and a location-named one demotes everything else. Both are
        # re-settlements, and a reviewer saying "yes, that one" is not asking for either. The
        # record gains a line saying a person looked; the map is not touched.
        outcome.measure_ruling = _note(outcome.measure_ruling,
                                       f"confirmed by {HUMAN_DECIDER}"
                                       + (f": {_clip(str(answer.get('note') or ''), 400)}"
                                          if str(answer.get("note") or "").strip() else ""))
        return True, ""
    candidate = reopened_outcome(outcome)
    settlement = read_measure_answer(
        candidate,
        winning_metric=str(answer.get("winning_analysis_metric") or "").strip(),
        winning_location=str(answer.get("winning_location") or "").strip(),
        losing_locations=[str(x) for x in (answer.get("losing_locations") or [])],
        # a person's answer may not cost the other arm of the contrast (see `keep_group_siblings`)
        keep_group_siblings=True)
    if not (settlement.ok and settlement.settles):
        # …and `candidate` is discarded: the map is untouched
        return False, settlement.reason or ("it demotes no location, so both readings are "
                                            "still read and nothing is settled")
    note = str(answer.get("note") or "").strip()
    apply_measure_settlement(
        candidate, settlement,
        lambda metric, winner=settlement.metric: (
            f"{HUMAN_DECIDER} chose {winner} as this outcome's measure; this location measures "
            f"{metric}" + (f" ({_clip(note, 120)})" if note else "")))
    candidate.measure_ruling = (f"measure: {settlement.metric}. decided by {HUMAN_DECIDER}"
                                + (f": {_clip(note, 400)}" if note else ""))
    dataset.outcomes[index] = candidate
    return True, ""


def apply_map_answers(study: StudyMap, answers: Sequence[Mapping[str, Any]],
                      effects: MutableMapping[tuple[str, str, str], str] | None = None
                      ) -> StudyMap:
    """A NEW `StudyMap` with the human answers to this map's open questions applied.

    Pure: no I/O, no model call, and the study handed in is never mutated — the review log is
    append-only and a resumed run re-applies the whole log to the stage file it read, so applying
    an answer twice must give the same map both times.

    An answer is applied only where it ANSWERS AN OPEN QUESTION of this study: the record must
    name this paper, a dataset the map has, and a question the map actually asked. Anything else
    — an unknown kind, another paper, a dataset that was never in question, a metric no `value`
    location carries — is ignored, silently and without changing the map. Two answers to one
    question are a reviewer changing their mind: the last one stands, and it is applied to the
    map as it arrived rather than on top of the first, so the order in the log decides the answer
    and nothing accumulates.

    `effects`, when given, collects `{question: why it changed nothing}` for every answer addressed
    to this study — `""` meaning it was applied. An answer this function drops (one naming a reading
    the map does not carry, one that would leave both measures readable, one addressed to a question
    nobody asked) changed nothing, and the run may not go on to record it as a decision it has acted
    on (M8). Without somewhere to say so, "I applied it" and "I was handed it" were the same fact to
    every caller: a reviewer whose answer was refused was told it had landed, and the card came off
    the page with the refusal on it.
    """
    answered = study.model_copy(deep=True)
    asked = {(q.kind, q.dataset_id, q.outcome_key) for q in answered.open_questions}
    latest: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for answer in answers:
        kind = str(answer.get("kind") or "").strip()
        if kind not in _ANSWER_KINDS or not _answers_this_study(answered, answer):
            continue
        key = map_answer_key(answer)
        if key in asked or _overrules_an_exclusion(answered, kind, answer) \
                or _overrules_a_measure_ruling(answered, kind, answer):
            latest[key] = answer
        elif effects is not None:
            effects[key] = ("no open question of this map asks it, and it overrules no ruling the "
                            "map made for itself")

    for (kind, dataset_id, outcome_key), answer in latest.items():
        dataset = next((d for d in answered.datasets if d.dataset_id == dataset_id), None)
        if dataset is None:
            if effects is not None:
                effects[(kind, dataset_id, outcome_key)] = "this map has no such dataset"
            continue
        did, why = (_answer_inclusion(dataset, answer) if kind == "include_dataset"
                    else _answer_measure(dataset, outcome_key, answer))
        if effects is not None:
            effects[(kind, dataset_id, outcome_key)] = why
        if not did:
            continue
        answered.open_questions = [q for q in answered.open_questions
                                   if (q.kind, q.dataset_id, q.outcome_key)
                                   != (kind, dataset_id, outcome_key)]
        # every line this question put in the human queue goes with it. The inclusion question
        # writes two — its own, and the "group mapping unconfirmed" line the cross-check diff
        # wrote for the same dataset — and leaving the second standing sends a settled paper back
        # to a person for a question they have already answered.
        stale = {_question_flag(kind, dataset_id, outcome_key)}
        if kind == "include_dataset":
            stale.add(_unmatched_dataset_flag(dataset_id))
        answered.needs_human = [f for f in answered.needs_human if f not in stale]
    return answered


def map_answer_effects(study: StudyMap, answers: Sequence[Mapping[str, Any]]
                       ) -> dict[tuple[str, str, str], str]:
    """What each of these answers DID to this map: `{question: why it changed nothing}`, `""` when
    it was applied.

    The same loop as `apply_map_answers`, so the two can never disagree about what landed. This is
    the fact `_consumed_seqs` needs and had no way to ask for: an answer that changed nothing was
    retired the moment the cells it named had been read by anything, which told a reviewer their
    refused answer was applied and took the card off the page with it.
    """
    out: dict[tuple[str, str, str], str] = {}
    apply_map_answers(study, answers, effects=out)
    return out


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
