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
                                              every location where a number lives;
3. roster follow-up     (Sonnet, medium)      only if the study map left a figure/table undecided;
4. independent check    (Sonnet, medium)      eligibility, datasets, Ns, source locations, error bars;
5. adjudication         (Opus, xhigh)         only when eligibility or an n disagrees.

Passes 1 and 2 are one job split in two because the combined JSON schema exceeds the
structured-output grammar limit; splitting also lets pass 2 spend its whole answer on locations.

Everything the two agents disagree about lands in `StudyMap.disagreements`; anything no two agents
ever agreed on lands in `StudyMap.needs_human` and is excluded from the primary analysis later.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, get_args

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.context import FILES_API_BETA, figure_blocks, text_block
from ..models import (AnalysisMetric, Citation, DatasetSpec, DispersionType, ErrorBarScope,
                      ExposureOrder, GroupSpec, OutcomeSources, Protocol, RosterDecision, Source,
                      SourceKind, StudyMap)
from . import render_prompt

__all__ = ["map_study", "protocol_text", "roster_text", "roster_entries", "dataset_text",
           "PROMPT_VERSION", "MAPPER_SCHEMA", "MAPPER_SOURCES_SCHEMA",
           "MAPPER_CROSSCHECK_SCHEMA", "MAPPER_ADJUDICATE_SCHEMA", "MAPPER_ROSTER_SCHEMA"]

#: bump whenever any mapper prompt or schema changes (recorded on every call, and in the manifest)
PROMPT_VERSION = "mapper/1"

#: shared across the three calls so the cached `document` prefix (system + PDF) can be reused
SYSTEM = ("You are a component of Canopy, an automated meta-analysis pipeline. You work only from "
          "the documents you are given, you quote them verbatim as evidence, and you never invent, "
          "estimate or compute a number. When a paper does not report something, you say so.")

CAPTION_CHARS = 400            # roster captions are trimmed so the prompt stays small
TABLE_ROWS = 3                 # rows of a table shown in the roster
UNKNOWN_KIND = "unknown"       # schema-only escape hatch (see `_source`)
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
                 "notes"],
    "properties": {
        "kind": _enum([k.value for k in SourceKind] + [UNKNOWN_KIND]),
        "page": {"type": "integer"},
        "locator": {"type": "string"},
        "quote": {"type": "string"},
        "figure_id": {"type": "string"},                 # roster id, "" when not a figure
        "table_id": {"type": "string"},
        "error_bar_type": _enum([d.value for d in DispersionType]),
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
    "required": ["eligible", "eligibility_rationale", "datasets", "notes"],
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
                                            "kind": _enum([k.value for k in SourceKind]
                                                          + [UNKNOWN_KIND]),
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


def _source(raw: dict[str, Any]) -> Source | None:
    """A `Source` from either schema's source object; None when the kind was not recognised."""
    if raw.get("kind") not in _SOURCE_KINDS:
        return None
    data = dict(raw)
    data["figure_id"] = data.get("figure_id") or None
    data["table_id"] = data.get("table_id") or None
    return Source.model_validate(data)


def _norm(text: str) -> str:
    """Locators/labels compared loosely: 'Fig. 1' == 'Figure 1', 'Table 2' == 'tab 2'."""
    lowered = (text or "").lower().replace("figure", "fig").replace("table", "tab")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _mapping_swapped(primary_a: str, primary_b: str, other_a: str, other_b: str) -> bool:
    """True when the other agent's A/B labels line up with the primary's B/A (a real swap)."""
    if not all(_norm(x) for x in (primary_a, primary_b, other_a, other_b)):
        return False
    return (_sim(other_a, primary_b) > _sim(other_a, primary_a)
            and _sim(other_b, primary_a) > _sim(other_b, primary_b))


def _source_key(source: Source, tier: int) -> tuple | None:
    ident = source.figure_id or source.table_id or ""
    locator = _norm(source.locator)
    if tier == 0:
        return (source.page, ident, locator)
    if tier == 1:
        return (source.page, ident) if ident else None
    return (source.page, locator) if locator else None


def _quotes_overlap(a: str, b: str) -> bool:
    """One quote contains the other — two agents quoting the same sentence, differently trimmed."""
    a, b = _norm(a), _norm(b)
    if min(len(a), len(b)) < QUOTE_MATCH_CHARS:
        return False
    return a in b or b in a


def _match_source(sources: list[Source], other: Source) -> Source | None:
    """Same location? id first, then locator, then the quote — page always has to match.

    The quote tier matters for text sources: two agents describe the same sentence with different
    locators ("Results, ANOVA on the adaptation phase" vs "Results, adaptation phase ANOVA").
    """
    for tier in (0, 1, 2):
        target = _source_key(other, tier)
        if target is None:
            continue
        for source in sources:
            if _source_key(source, tier) == target:
                return source
    for source in sources:
        if source.page == other.page and _quotes_overlap(source.quote, other.quote):
            return source
    return None


def _same_location(source: Source, raw: dict[str, Any]) -> bool:
    if int(raw.get("page") or 0) != source.page:
        return False
    ident = (raw.get("figure_id") or raw.get("table_id") or "").strip()
    own = source.figure_id or source.table_id or ""
    if ident and own:
        return ident == own
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


def _apply_sources(study: StudyMap, parsed: dict[str, Any], outcome_keys: set[str],
                   flags: list[str]) -> None:
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
            if key not in outcome_keys:
                flags.append(f"dataset {dataset.dataset_id}: outcome key {key!r} is not in the "
                             f"protocol — needs human")
            sources = []
            for raw_source in raw_outcome.get("sources") or []:
                source = _source(raw_source)
                if source is None:
                    flags.append(f"dataset {dataset.dataset_id} {key} "
                                 f"p{raw_source.get('page')} {raw_source.get('locator')}: "
                                 f"source kind not recognised — needs human")
                    continue
                sources.append(source)
            existing = next((o for o in dataset.outcomes if o.outcome_key == key), None)
            if existing is not None:                    # the model split one outcome over two rows
                existing.sources += sources
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
@dataclass
class _Conflicts:
    eligibility: bool = False
    n_mismatch: bool = False
    mapping: list[int] = field(default_factory=list)        # indices into StudyMap.datasets
    error_bars: list[dict[str, Any]] = field(default_factory=list)

    @property
    def needs_adjudication(self) -> bool:
        return self.eligibility or self.n_mismatch


def _diff(study: StudyMap, check: dict[str, Any], disagreements: list[str]) -> _Conflicts:
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
        _diff_dataset(index, dataset, raw, conflicts, disagreements)
    return conflicts


def _diff_dataset(index: int, dataset: DatasetSpec, raw: dict[str, Any], conflicts: _Conflicts,
                  disagreements: list[str]) -> None:
    for key in ("group_a", "group_b"):
        group: GroupSpec = getattr(dataset, key)
        other = raw.get(key) or {}
        other_n = other.get("n")
        if other_n is not None and group.n is not None and int(other_n) != int(group.n):
            disagreements.append(f"{dataset.dataset_id} {key} n: primary={group.n} "
                                 f"cross-check={int(other_n)}")
            conflicts.n_mismatch = True

    label_a = (raw.get("group_a") or {}).get("label") or ""
    label_b = (raw.get("group_b") or {}).get("label") or ""
    if _mapping_swapped(dataset.group_a.label, dataset.group_b.label, label_a, label_b):
        disagreements.append(
            f"{dataset.dataset_id} group mapping: primary A={dataset.group_a.label!r} "
            f"B={dataset.group_b.label!r}; cross-check A={label_a!r} B={label_b!r}")
        conflicts.mapping.append(index)

    for raw_outcome in raw.get("outcomes") or []:
        key = raw_outcome.get("outcome_key") or ""
        outcome = next((o for o in dataset.outcomes if o.outcome_key == key), None)
        for raw_source in raw_outcome.get("sources") or []:
            source = _source(raw_source)
            if source is None:
                continue
            match = _match_source(outcome.sources, source) if outcome is not None else None
            if match is None:
                if outcome is None:
                    outcome = OutcomeSources(outcome_key=key)
                    dataset.outcomes.append(outcome)
                source.notes = "added by cross-check"
                outcome.sources.append(source)
                disagreements.append(f"{dataset.dataset_id} {key}: source added by cross-check "
                                     f"— p{source.page} {source.locator}")
                continue
            if (match.error_bar_type is not DispersionType.UNKNOWN
                    and source.error_bar_type is not DispersionType.UNKNOWN
                    and match.error_bar_type is not source.error_bar_type):
                disagreements.append(
                    f"{dataset.dataset_id} {key} p{match.page} {match.locator}: error bar "
                    f"primary={match.error_bar_type.value} "
                    f"cross-check={source.error_bar_type.value}")
                conflicts.error_bars.append({"dataset_index": index, "dataset_id":
                                             dataset.dataset_id, "outcome_key": key,
                                             "source": match,
                                             "check_type": source.error_bar_type})


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
            _adopt_group(dataset, key, raw.get(key) or {}, disagreements)

    _apply_error_bar_rulings(study, adjudicated.get("error_bar_rulings") or [], conflicts,
                             disagreements)


def _resolve_mapping(dataset: DatasetSpec, raw: dict[str, Any], position: int,
                     primary_labels: tuple[str, str], conflicts: _Conflicts,
                     disagreements: list[str]) -> None:
    ruled_a = (raw.get("group_a") or {}).get("label") or ""
    ruled_b = (raw.get("group_b") or {}).get("label") or ""
    if _mapping_swapped(primary_labels[0], primary_labels[1], ruled_a, ruled_b):
        dataset.group_a, dataset.group_b = dataset.group_b, dataset.group_a
        disagreements.append(f"adjudicated {dataset.dataset_id} group mapping: swapped to "
                             f"A={dataset.group_a.label!r} B={dataset.group_b.label!r}")
        conflicts.mapping.remove(position)
    elif (_sim(ruled_a, primary_labels[0]) > _sim(ruled_a, primary_labels[1])
          and _sim(ruled_b, primary_labels[1]) > _sim(ruled_b, primary_labels[0])):
        disagreements.append(f"adjudicated {dataset.dataset_id} group mapping: primary confirmed")
        conflicts.mapping.remove(position)


def _adopt_group(dataset: DatasetSpec, key: str, raw: dict[str, Any],
                 disagreements: list[str]) -> None:
    """Adopt the adjudicated n (with its quote); every ruling is recorded, confirmations included."""
    group: GroupSpec = getattr(dataset, key)
    ruled_n = raw.get("n")
    if ruled_n is None:
        return
    if group.n is not None and int(ruled_n) == int(group.n):
        disagreements.append(f"adjudicated {dataset.dataset_id} {key} n: {int(ruled_n)} "
                             f"(primary confirmed)")
        return
    disagreements.append(f"adjudicated {dataset.dataset_id} {key} n: {int(ruled_n)} "
                         f"(primary said {group.n})")
    group.n = int(ruled_n)
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
            conflicts.error_bars.remove(conflict)


# ----------------------------------------------------------------------------- the agent
def map_study(client: LLMClient, paper: PaperRecord, protocol: Protocol, *,
              model_primary: str = MODELS["primary"], model_check: str = MODELS["secondary"],
              model_adjudicate: str = MODELS["adjudicator"],
              pdf_file_id: str | None = None) -> StudyMap:
    """Map one paper against one protocol: eligibility, datasets, group Ns and source locations."""
    sha12 = paper.sha256[:12]
    document = client.pdf_block(pdf_path=None if pdf_file_id else paper.source_path,
                                file_id=pdf_file_id, cache=True)
    betas = [FILES_API_BETA] if pdf_file_id else None
    entries = roster_entries(paper)
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

    if study.datasets:                       # nothing to locate when the paper has no contrast
        located = client.structured(
            model=model_primary, system=SYSTEM, schema=MAPPER_SOURCES_SCHEMA, effort="high",
            max_tokens=16000, betas=betas, prompt_version=PROMPT_VERSION,
            cell_key=f"map-sources:{sha12}",
            messages=[{"role": "user", "content": [document, text_block(render_prompt(
                "mapper_sources", PROTOCOL=protocol_prompt, ROSTER=roster_prompt,
                DATASETS=dataset_text(study)))]}])
        call_ids.append(located.call_id)
        _apply_sources(study, located.parsed or {}, {o.key for o in protocol.outcomes}, flags)

    decided: dict[str, RosterDecision] = {}
    _decisions(paper, parsed.get("roster"), decided, flags)
    missing = [entry for entry in entries if entry["id"] not in decided]
    if missing:
        follow_up = _roster_follow_up(client, paper, protocol_prompt, missing, model_check)
        call_ids.append(follow_up.call_id)
        _decisions(paper, (follow_up.parsed or {}).get("decisions"), decided, flags)
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
    conflicts = _diff(study, checked, disagreements)

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
        flags.append(f"dataset {dataset.dataset_id}: group mapping disagreement "
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
