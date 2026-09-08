"""Draft a protocol from one sentence (amendment I).

This is a *starting point*, never a finished protocol: the model turns one sentence — "does
treatment X change outcome Y in adults, compared with controls?" — into the *shape* of a protocol:
two groups with definitions and synonyms, one or more outcomes with windows and direction labels,
eligibility bullets. The reviewer then edits every line of it. It is the only place in Canopy where a model writes something a human is
expected to change rather than to check.

The schema deliberately contains no statistics: the profile is chosen in the form, not by a model,
and no field here could ever hold a computed number (`canopy.llm.schemas` enforces that for every
agent schema, and this one obeys the same rule).
"""
from __future__ import annotations

from typing import Any

import yaml

from ..models import Protocol, StatsSettings
from ..protocol import apply_profile

__all__ = ["DRAFT_SCHEMA", "draft_from_sentence", "protocol_to_yaml", "PROMPT_VERSION"]

PROMPT_VERSION = "draft-1"

_GROUP = {
    "type": "object", "additionalProperties": False,
    "required": ["label", "definition", "synonyms"],
    "properties": {
        "label": {"type": "string", "description": "Short name for this arm of the comparison."},
        "definition": {"type": "string",
                       "description": "How a paper's own wording identifies this group."},
        "synonyms": {"type": "array", "items": {"type": "string"},
                     "description": "Other words a paper might use for it."},
    },
}

_OUTCOME = {
    "type": "object", "additionalProperties": False,
    "required": ["key", "label", "definition", "measurement_window", "higher_is_better_hint",
                 "positive_direction_label", "negative_direction_label", "units_hint"],
    "properties": {
        "key": {"type": "string", "description": "lower_snake_case identifier."},
        "label": {"type": "string"},
        "definition": {"type": "string",
                       "description": "What counts as this outcome, in the words a reader of the "
                                      "papers would use."},
        "measurement_window": {"type": "string",
                               "description": "Which timepoint or block to read when a paper "
                                              "reports several."},
        "higher_is_better_hint": {"type": "string",
                                  "description": "How to decide whether a larger raw number means "
                                                 "more or less of the construct."},
        "positive_direction_label": {"type": "string",
                                     "description": "Forest-plot label for a positive effect."},
        "negative_direction_label": {"type": "string",
                                     "description": "Forest-plot label for a negative effect."},
        "units_hint": {"type": "string"},
    },
}

#: what the model fills in. Strict objects, every property required, and nothing derived.
DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "research_question", "group_a", "group_b", "outcomes", "eligibility",
                 "dataset_rules", "moderators", "notes"],
    "properties": {
        "title": {"type": "string"},
        "research_question": {"type": "string", "description": "One sentence."},
        "group_a": _GROUP,
        "group_b": _GROUP,
        "outcomes": {"type": "array", "items": _OUTCOME},
        "eligibility": {"type": "array", "items": {"type": "string"},
                        "description": "Screening rules; a paper failing any one is excluded."},
        "dataset_rules": {"type": "array", "items": {"type": "string"},
                          "description": "How to treat a paper contributing several comparisons."},
        "moderators": {"type": "array", "items": {"type": "string"},
                       "description": "lower_snake_case names recorded per dataset."},
        "notes": {"type": "string"},
    },
}

SYSTEM = ("You draft the protocol for a systematic review and meta-analysis. You write the "
          "*plan*: who is compared with whom, which outcomes are pooled, how each is measured and "
          "which studies are eligible. You never invent findings, numbers or citations, and you "
          "never mention a specific study. Where the request leaves something open, write the "
          "most conventional choice for that field and say so in `notes` so the reviewer can "
          "change it.")

PROMPT = """A researcher describes the review they want to run, in one sentence:

    {SENTENCE}

Draft the protocol. Rules:

* group A is the group the sentence puts first (the one an effect is "in"); group B is its
  comparator. Give each a definition written as *how a paper would identify it*, plus the
  synonyms papers actually use.
* one outcome per quantity that would be pooled separately. Keys are lower_snake_case.
* `measurement_window` says which timepoint or block to read when a paper reports several.
* `higher_is_better_hint` tells an extractor how to work out, per paper, whether a larger raw
  number means more or less of the construct — it does not decide it here.
* `positive_direction_label` / `negative_direction_label` become the two ends of the forest
  plot's x axis; write them from group A's point of view.
* eligibility bullets are the screening rules, one per line, including the language rule.
* moderators are the study characteristics worth recording for every dataset.

Return the draft only. The reviewer will edit every line of it."""


def draft_from_sentence(client: Any, sentence: str, *, model: str,
                        max_tokens: int = 8000) -> dict[str, Any]:
    """One structured call. Returns `{"protocol": {...}, "yaml": "...", "warnings": [...]}`."""
    result = client.structured(
        model=model, system=SYSTEM, schema=DRAFT_SCHEMA, effort="high", max_tokens=max_tokens,
        prompt_version=PROMPT_VERSION,
        messages=[{"role": "user", "content": PROMPT.format(SENTENCE=sentence)}])
    drafted = dict(result.parsed or {})
    protocol, warnings = _as_protocol(drafted)
    return {"protocol": protocol.model_dump(mode="json"), "yaml": protocol_to_yaml(protocol),
            "warnings": warnings, "cost_usd": round(float(getattr(result, "cost_usd", 0.0)), 6),
            "model": model}


def _as_protocol(drafted: dict[str, Any]) -> tuple[Protocol, list[str]]:
    """The model's answer as a real `Protocol` — so a draft that will not load never reaches a UI."""
    warnings: list[str] = []
    group_a = dict(drafted.get("group_a") or {})
    group_b = dict(drafted.get("group_b") or {})
    outcomes = list(drafted.get("outcomes") or [])
    if not outcomes:
        warnings.append("the draft has no outcome — one was added for you to fill in")
        outcomes = [{"key": "primary_outcome", "label": "Primary outcome",
                     "definition": "", "measurement_window": "", "higher_is_better_hint": "",
                     "positive_direction_label": "Higher in A",
                     "negative_direction_label": "Lower in A", "units_hint": ""}]
    seen: set[str] = set()
    for index, outcome in enumerate(outcomes):
        key = str(outcome.get("key") or f"outcome_{index + 1}").strip().lower()
        key = "".join(c if c.isalnum() else "_" for c in key).strip("_") or f"outcome_{index + 1}"
        if key in seen:
            key = f"{key}_{index + 1}"
        seen.add(key)
        outcome["key"] = key
    payload = {
        "title": str(drafted.get("title") or "Untitled review"),
        "research_question": str(drafted.get("research_question") or ""),
        "group_a": {"key": "A", **{k: group_a.get(k, "") if k != "synonyms"
                                   else list(group_a.get("synonyms") or [])
                                   for k in ("label", "definition", "synonyms")}},
        "group_b": {"key": "B", **{k: group_b.get(k, "") if k != "synonyms"
                                   else list(group_b.get("synonyms") or [])
                                   for k in ("label", "definition", "synonyms")}},
        "outcomes": outcomes,
        "eligibility": [str(x) for x in (drafted.get("eligibility") or [])],
        "dataset_rules": [str(x) for x in (drafted.get("dataset_rules") or [])],
        "moderators": [str(x) for x in (drafted.get("moderators") or [])],
        "notes": str(drafted.get("notes") or ""),
        "stats": {"profile": "metafor"},
    }
    if not payload["group_a"]["label"] or not payload["group_b"]["label"]:
        warnings.append("one of the two groups has no label — name it before you run anything")
        payload["group_a"]["label"] = payload["group_a"]["label"] or "Group A"
        payload["group_b"]["label"] = payload["group_b"]["label"] or "Group B"
    protocol = Protocol.model_validate(payload)
    protocol.stats = apply_profile(StatsSettings(profile="metafor"))
    warnings.append("this is a draft: check every definition against your own review's methods "
                    "before you run it")
    return protocol, warnings


def protocol_to_yaml(protocol: Protocol) -> str:
    """Just the parts a person edits — the statistics stay a named profile until they change it."""
    data = protocol.model_dump(mode="json")
    ordered = {
        "title": data["title"],
        "research_question": data["research_question"],
        "group_a": data["group_a"],
        "group_b": data["group_b"],
        "outcomes": data["outcomes"],
        "eligibility": data["eligibility"],
        "dataset_rules": data["dataset_rules"],
        "moderators": data["moderators"],
        "stats": {"profile": protocol.stats.profile},
        "notes": data["notes"],
    }
    body = yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=100)
    return ("# Drafted by AutoMeta from one sentence — a starting point, not a protocol.\n"
            "# Edit every line, then: canopy run --papers PDFS --protocol this.yaml --out runs/x\n"
            f"{body}")
