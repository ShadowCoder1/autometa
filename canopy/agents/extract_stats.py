"""Test-statistic / reported-effect-size extractor — the fallback route to an effect size.

When a paper prints no group means, a statistic it *does* print can still carry the contrast: an
independent-groups t or a one-way between-participants F converts to a standardised mean difference
(in code, in `canopy.stats`, in Task 9). Most statistics cannot: an interaction is not a group
comparison, a within-participant effect carries no between-group variance, and the main effect of
the grouping factor in a mixed analysis is computed against a different error term.

So this agent transcribes — statistic, degrees of freedom, p, the design as the paper describes it,
which group came out higher and the words that say so — and *code* decides admissibility from the
design. The model's own verdict can only make a statistic less admissible, never more; and even an
admissible flag is advisory, because Task 9 refuses non-convertible designs again on its own.
"""
from __future__ import annotations

from typing import Any, Sequence, get_args

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import LLMClient
from ..llm.context import text_block
from ..models import (Candidate, CandidateStatus, DatasetSpec, Direction, PKind, Protocol,
                      ReportedScale, Source, SourceKind, Standardizer, TestDesign)
from . import render_prompt
from .extract_common import (STAT_SOURCE_KINDS, SYSTEM, candidate_id, context_blocks, enum_schema,
                             ground_candidate, groups_text, note, outcome_text, pages_of,
                             prompt_fingerprint, sources_text, table_ids_of, usable_sources)

__all__ = ["extract_test_statistics", "admissibility", "EXTRACT_STATS_SCHEMA", "PROMPT_VERSION",
           "PROMPT_FILES", "ADMISSIBLE_DESIGNS"]

PROMPT_FILES = ("extract_stats",)
PROMPT_VERSION = f"extract_stats/1@{prompt_fingerprint(PROMPT_FILES)}"

MAX_TOKENS = 8000
EFFORT = "high"

#: kinds the model may answer with, mapped onto `CandidateKind`
_KINDS = {"test_statistic": "test_statistic", "reported_effect_size": "reported_d"}
_STAT_TYPES = ("t", "F", "p", "chi2", "unknown")
_YES_NO = ("yes", "no", "unknown")


# ----------------------------------------------------------------------------- admissibility
#: the only designs a standardised mean difference can be recovered from (amendment C: `smd_from_t`
#: / `smd_from_f` refuse everything else, so flagging it here just tells the reviewer why)
ADMISSIBLE_DESIGNS: frozenset[str] = frozenset({"independent_t", "one_way_between"})
#: why each other design cannot stand in for the two group means
DESIGN_REASONS: dict[str, str] = {
    "mixed_main_effect": "the effect of the grouping factor in a mixed analysis is tested against "
                         "a different error term than a two-group comparison, so its F does not "
                         "convert to a standardised mean difference",
    "interaction": "an interaction term is not a comparison of the two groups",
    "paired": "a within-participant comparison carries no between-group variance",
    "ancova": "a covariate-adjusted statistic is not the raw contrast of the two groups",
    "welch": "unequal-variance degrees of freedom do not match n_a + n_b - 2",
    "unknown": "the paper does not say what kind of test this is",
}
#: scales that are a difference between two groups in units of a between-participant SD
_BETWEEN_SCALES = frozenset({"cohens_d", "hedges_g", "glass_delta"})
_WITHIN_STANDARDIZERS = frozenset({"dz_paired", "partial_eta"})


def admissibility(kind: str, design: str, compares: str, model_ok: bool, model_reason: str,
                  reported_scale: str = "unknown",
                  standardizer: str = "unknown") -> tuple[bool, str]:
    """`(admissible, reason)` — code's own reading of whether this statistic could carry the effect.

    Advisory: Task 9 applies the same rule again before it converts anything. The extractor may
    veto (it saw the sentence), but it may never promote a design this table refuses.
    """
    if compares != "yes":                    # first: is it even about these two groups?
        return False, ("the extractor could not confirm that this compares the two groups of this "
                       f"contrast on this outcome (it answered {compares!r})")
    if kind == "reported_d":
        if reported_scale not in _BETWEEN_SCALES:
            return False, (f"a reported effect size on the {reported_scale!r} scale is not a "
                           f"difference between two groups in standard-deviation units")
        if standardizer in _WITHIN_STANDARDIZERS:
            return False, (f"the reported effect size was standardised by {standardizer!r}, not by "
                           f"the between-participant standard deviation")
    elif design not in ADMISSIBLE_DESIGNS:
        return False, DESIGN_REASONS.get(design, f"design {design!r} is not a comparison of two "
                                                 f"independent groups")
    if not model_ok:
        return False, model_reason or "the extractor marked this statistic inadmissible"
    return True, ""


# ----------------------------------------------------------------------------- schema
#: one row per statistic found; 28 leaf properties, under the structured-output grammar limit
EXTRACT_STATS_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["statistics", "notes"],
    "properties": {
        "statistics": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["kind", "status", "page", "locator", "quote", "effect_as_written",
                             "compares_the_two_groups", "design", "stat_type", "stat_value", "df",
                             "df1", "df2", "tails", "p_kind", "p_value", "direction",
                             "direction_quote", "reported_value", "reported_scale", "standardizer",
                             "reported_ci_low", "reported_ci_high", "positive_means", "admissible",
                             "admissible_reason", "notes"],
                "properties": {
                    "kind": enum_schema([*_KINDS, "unknown"]),
                    "status": enum_schema(get_args(CandidateStatus)),
                    "page": {"type": ["integer", "null"]},
                    "locator": {"type": "string"},
                    "quote": {"type": "string"},
                    "effect_as_written": {"type": "string"},
                    "compares_the_two_groups": enum_schema(_YES_NO),
                    "design": enum_schema(get_args(TestDesign)),
                    "stat_type": enum_schema(_STAT_TYPES),
                    "stat_value": {"type": ["number", "null"]},
                    "df": {"type": ["number", "null"]},
                    "df1": {"type": ["number", "null"]},
                    "df2": {"type": ["number", "null"]},
                    "tails": {"type": ["integer", "null"]},
                    "p_kind": enum_schema(get_args(PKind)),
                    "p_value": {"type": ["number", "null"]},
                    "direction": enum_schema(get_args(Direction)),
                    "direction_quote": {"type": "string"},
                    "reported_value": {"type": ["number", "null"]},
                    "reported_scale": enum_schema(get_args(ReportedScale)),
                    "standardizer": enum_schema(get_args(Standardizer)),
                    "reported_ci_low": {"type": ["number", "null"]},
                    "reported_ci_high": {"type": ["number", "null"]},
                    "positive_means": enum_schema(get_args(Direction)),
                    "admissible": {"type": "boolean"},
                    "admissible_reason": {"type": "string"},
                    "notes": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}


# ----------------------------------------------------------------------------- parsing
_STATUSES = set(get_args(CandidateStatus))


def _enum_value(raw: Any, allowed: Any, fallback: str) -> str:
    value = (raw or "").strip() if isinstance(raw, str) else ""
    return value if value in set(allowed) else fallback


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


def _route(kind: str, stat_type: str) -> str:
    """The route name Task 9's precedence list uses (`StatsSettings.route_precedence`)."""
    if kind == "reported_d":
        return "reported_d"
    return "p_value" if stat_type == "p" else "test_statistic"


def _candidate(row: dict[str, Any], *, index: int, paper: PaperRecord, dataset: DatasetSpec,
               outcome_key: str, extractor_id: str, model: str, call_id: str) -> Candidate:
    kind = _KINDS.get(_enum_value(row.get("kind"), _KINDS, "test_statistic"), "test_statistic")
    status = _enum_value(row.get("status"), _STATUSES, "ambiguous")
    design = _enum_value(row.get("design"), get_args(TestDesign), "unknown")
    stat_type = _enum_value(row.get("stat_type"), _STAT_TYPES, "unknown")
    compares = _enum_value(row.get("compares_the_two_groups"), _YES_NO, "unknown")
    reported_scale = _enum_value(row.get("reported_scale"), get_args(ReportedScale), "unknown")
    standardizer = _enum_value(row.get("standardizer"), get_args(Standardizer), "unknown")
    notes = (row.get("notes") or "").strip()
    page = _whole(row.get("page"))

    values = {name: _number(row.get(name)) for name in
              ("stat_value", "p_value", "reported_value", "reported_ci_low", "reported_ci_high",
               "df", "df1", "df2")}
    tails = _whole(row.get("tails"))
    p_kind = _enum_value(row.get("p_kind"), get_args(PKind), "unknown")
    if status != "found":                          # amendment E: a non-`found` row carries no value
        printed = {k: v for k, v in {**values, "tails": tails}.items() if v is not None}
        if p_kind != "unknown":
            printed["p_kind"] = p_kind
        if printed:
            notes = note(notes, f"numbers dropped because status is {status}: {printed}")
        values, tails, p_kind = dict.fromkeys(values), None, "unknown"

    admissible, reason = admissibility(
        kind, design, compares, bool(row.get("admissible")),
        (row.get("admissible_reason") or "").strip(), reported_scale, standardizer)
    if status != "found":                    # nothing was transcribed, so nothing is admissible
        admissible, reason = False, f"the extractor answered {status}"

    cand = Candidate(
        candidate_id=candidate_id(dataset.dataset_id, outcome_key, extractor_id, index),
        paper_id=paper.sha256, dataset_id=dataset.dataset_id, outcome_key=outcome_key,
        kind=kind, group=None, status=status,
        source_kind=(SourceKind.reported_effect_size if kind == "reported_d"
                     else SourceKind.test_statistic),
        stat_type=stat_type, stat_value=values["stat_value"],
        df=values["df"], df1=values["df1"], df2=values["df2"], tails=tails, p_kind=p_kind,
        p_value=values["p_value"], design=design,
        direction=_enum_value(row.get("direction"), get_args(Direction), "unknown"),
        admissible=admissible, admissible_reason=reason,
        reported_value=values["reported_value"], reported_scale=reported_scale,
        standardizer=standardizer, reported_ci_low=values["reported_ci_low"],
        reported_ci_high=values["reported_ci_high"],
        positive_means=_enum_value(row.get("positive_means"), get_args(Direction), "unknown"),
        page=page, quote=(row.get("quote") or "").strip(),
        locator=(row.get("locator") or "").strip(),
        route=_route(kind, stat_type), model=model, prompt_version=PROMPT_VERSION,
        llm_call_id=call_id, extractor_id=extractor_id, notes=notes)

    effect = (row.get("effect_as_written") or "").strip()
    if effect:
        cand.notes = note(cand.notes, f"effect as written: {effect}")
    direction_quote = (row.get("direction_quote") or "").strip()
    if direction_quote:
        cand.notes = note(cand.notes, f"direction evidence: {direction_quote}")
    if cand.direction != "unknown" and not direction_quote:
        cand.notes = note(cand.notes, f"direction {cand.direction} asserted without a quote")
    return ground_candidate(cand, paper)


def _nothing_found(paper: PaperRecord, dataset: DatasetSpec, outcome_key: str, extractor_id: str,
                   model: str, call_id: str) -> Candidate:
    """The extractor read the pages and found no statistic — recorded, not silently absent."""
    return Candidate(
        candidate_id=candidate_id(dataset.dataset_id, outcome_key, extractor_id, 0),
        paper_id=paper.sha256, dataset_id=dataset.dataset_id, outcome_key=outcome_key,
        kind="test_statistic", status="not_on_these_pages", stat_type="unknown",
        p_kind="unknown", admissible=False,
        admissible_reason="no statistic was found on these pages",
        route="test_statistic", model=model, prompt_version=PROMPT_VERSION, llm_call_id=call_id,
        extractor_id=extractor_id, source_kind=SourceKind.test_statistic,
        notes="the extractor returned no statistic for these pages")


# ----------------------------------------------------------------------------- the agent
def extract_test_statistics(client: LLMClient, paper: PaperRecord, protocol: Protocol,
                            dataset: DatasetSpec, outcome_key: str, sources: Sequence[Source], *,
                            model: str = MODELS["primary"]) -> list[Candidate]:
    """Transcribe every statistic and printed effect size for one outcome on the pages in `sources`.

    Returns grounded `Candidate`s of kind `test_statistic` / `reported_d`, each with code's own
    admissibility verdict. An empty list means there was nothing for this extractor to read.
    """
    usable = usable_sources(sources, STAT_SOURCE_KINDS, paper)
    if not usable:
        return []
    extractor_id = f"stats:{model}"
    pages = pages_of(usable)

    # a statistic is always printed as characters: page text and table cells, never a raster
    content = context_blocks(paper, pages, image_pages=(), table_ids=table_ids_of(usable, paper))
    content.append(text_block(render_prompt(
        "extract_stats",
        OUTCOME=outcome_text(protocol, outcome_key, dataset),
        GROUPS=groups_text(dataset),
        LOCATIONS=sources_text(usable))))

    result = client.structured(
        model=model, system=SYSTEM, schema=EXTRACT_STATS_SCHEMA, effort=EFFORT,
        max_tokens=MAX_TOKENS, prompt_version=PROMPT_VERSION,
        cell_key=f"extract-stats:{dataset.dataset_id}:{outcome_key}",
        messages=[{"role": "user", "content": content}])

    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    rows = [row for row in (parsed.get("statistics") or []) if isinstance(row, dict)]
    candidates = [_candidate(row, index=index, paper=paper, dataset=dataset,
                             outcome_key=outcome_key, extractor_id=extractor_id, model=model,
                             call_id=result.call_id)
                  for index, row in enumerate(rows)]
    if not candidates:
        candidates = [_nothing_found(paper, dataset, outcome_key, extractor_id, model,
                                     result.call_id)]
    overall = (parsed.get("notes") or "").strip()
    if overall:
        for cand in candidates:
            cand.notes = note(cand.notes, f"extractor notes: {overall}")
    return candidates
