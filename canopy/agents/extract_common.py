"""Shared machinery for the extractor agents (Tasks 5 and 7).

An extractor is told *where* to look (the mapper's `Source`s), *what* to look for (one protocol
outcome), and *for whom* (the dataset's two groups). It transcribes what the paper prints there and
nothing else: no arithmetic, no conversion, no estimate. Everything an extractor returns carries a
page and a verbatim quote, which `canopy.verify.grounding` then checks against the deterministic
page text — a value whose quote is not in the paper is a red flag, not a value.

This module holds what both extractors need: the prose renderers for the prompt, the context
builder (amendment E: text-only for text sources, a page image only where a table or figure has to
be looked at), source filtering, and the bookkeeping that turns a parsed row into a `Candidate`.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Sequence

from ..ingest.pdf import PaperRecord
from ..llm.client import EPHEMERAL
from ..llm.context import page_blocks, reviewer_hint_line, text_block
from ..models import DatasetSpec, GroupSpec, Protocol, Source, SourceKind
from ..verify.grounding import ground_candidate, normalize
# the mapper owns these three; sharing them (rather than copying) is what keeps one prompt-cache
# prefix and one schema idiom across every agent
from .mapper import SYSTEM, _clip as clip, _enum as enum_schema
from . import load_prompt

__all__ = ["SYSTEM", "enum_schema", "clip", "TEXT_SOURCE_KINDS", "STAT_SOURCE_KINDS",
           "outcome_text", "groups_text", "sources_text", "usable_sources", "pages_of",
           "table_ids_of", "context_blocks", "table_text", "prompt_fingerprint", "candidate_id",
           "ground_candidate", "note", "group_label_check"]

#: source kinds the text/table extractor can read (`unknown` is a number of an unlisted kind —
#: a fitted parameter, say; it is attempted as text and reported honestly if it is not usable)
TEXT_SOURCE_KINDS: frozenset[SourceKind] = frozenset({
    SourceKind.text_mean_sd, SourceKind.text_mean_se, SourceKind.text_mean_ci,
    SourceKind.table, SourceKind.author_data, SourceKind.unknown})
#: source kinds the test-statistic / reported-effect-size extractor reads
STAT_SOURCE_KINDS: frozenset[SourceKind] = frozenset({
    SourceKind.test_statistic, SourceKind.reported_effect_size, SourceKind.unknown})

TABLE_CELL_CHARS = 60          # per cell, when a table is rendered into the prompt


def note(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def prompt_fingerprint(names: Sequence[str]) -> str:
    """sha256 over the prompt files of one agent, so an edited prompt cannot keep its version."""
    blob = "\0".join(load_prompt(name) for name in names)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


# ----------------------------------------------------------------------------- prompt prose
def outcome_text(protocol: Protocol, outcome_key: str, dataset: DatasetSpec | None = None) -> str:
    """The one outcome an extractor is working on: the protocol's definition plus what the mapper
    already established about it in this paper (measure name, units, how it was operationalised)."""
    outcome = protocol.outcome(outcome_key)
    lines = [f"OUTCOME KEY: {outcome.key}", f"NAME: {outcome.label}",
             f"DEFINITION: {outcome.definition.strip()}"]
    if outcome.measurement_window:
        lines.append(f"MEASUREMENT WINDOW: {outcome.measurement_window.strip()}")
    if outcome.units_hint:
        lines.append(f"UNITS HINT: {outcome.units_hint.strip()}")
    if outcome.higher_is_better_hint:
        lines.append(f"DIRECTION HINT: {outcome.higher_is_better_hint.strip()}")
    found = None
    if dataset is not None:
        found = next((o for o in dataset.outcomes if o.outcome_key == outcome_key), None)
    if found is not None:
        if found.measure_name:
            lines.append(f"THE MEASURE THIS PAPER REPORTS: {found.measure_name}")
        if found.units:
            lines.append(f"UNITS THIS PAPER PRINTS: {found.units}")
        if found.operationalization:
            lines.append(f"HOW THIS PAPER MEASURED IT: {clip(found.operationalization, 400)}")
        if found.analysis_metric and found.analysis_metric != "unknown":
            lines.append(f"WHAT THE NUMBER IS RELATIVE TO: {found.analysis_metric}")
    return "\n".join(lines)


def _group_line(key: str, group: GroupSpec) -> str:
    line = (f"GROUP {key} — the paper calls it \"{group.label}\" "
            f"(analysed n = {group.n if group.n is not None else 'not reported'})")
    if group.n_evidence:
        line += f"\n    n evidence: {clip(group.n_evidence, 200)}"
    return line


def groups_text(dataset: DatasetSpec) -> str:
    """The two groups of this contrast, in the paper's own words."""
    lines = [f"DATASET: {dataset.label or dataset.dataset_id}"]
    if dataset.experiment:
        lines.append(f"EXPERIMENT: {dataset.experiment}")
    if dataset.condition:
        lines.append(f"CONDITION: {dataset.condition}")
    lines.append(_group_line("A", dataset.group_a))
    lines.append(_group_line("B", dataset.group_b))
    return "\n".join(lines)


def sources_text(sources: Sequence[Source], *, reviewer_hint: str = "") -> str:
    """Where a previous agent said the numbers live — a starting point, not a limit.

    `reviewer_hint` is a human's answer to "where is this value?" (a `re_extract` override). It is
    appended to the mapper's locations and never substituted for them: the hint is why this cell
    is being read a second time, not a ruling about where the first reading should have looked.
    """
    hint = reviewer_hint_line(reviewer_hint)
    if not sources:
        return "\n".join(["(no locations were recorded for this outcome)", hint] if hint
                         else ["(no locations were recorded for this outcome)"])
    lines = []
    for source in sources:
        head = f"- page {source.page} | {source.kind.value}"
        if source.locator:
            head += f" | {clip(source.locator, 200)}"
        if source.table_id:
            head += f" | table {source.table_id}"
        lines.append(head)
        if source.quote:
            lines.append(f"    quoted: {clip(source.quote, 400)}")
        if source.values_in_text:
            lines.append(f"    values it reported: {clip(source.values_in_text, 200)}")
        if source.error_bar_type.value not in ("UNKNOWN", "NONE"):
            lines.append(f"    a previous reader thought the ± value here is "
                         f"{source.error_bar_type.value} — check it yourself")
    if hint:
        lines.append(hint)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- context
def usable_sources(sources: Iterable[Source], kinds: frozenset[SourceKind],
                   paper: PaperRecord) -> list[Source]:
    """The sources this extractor can act on: right kind, page inside the document."""
    return [s for s in sources if s.kind in kinds and 1 <= s.page <= len(paper.pages)]


def pages_of(sources: Iterable[Source]) -> list[int]:
    return sorted({s.page for s in sources})


def table_ids_of(sources: Iterable[Source], paper: PaperRecord) -> list[str]:
    """Ingested tables the sources name, plus every table sitting on a page they point at."""
    known = {t.id for t in paper.tables}
    named = {s.table_id for s in sources if s.table_id and s.table_id in known}
    pages = set(pages_of(sources))
    named |= {t.id for t in paper.tables if t.page in pages}
    return sorted(named)


def table_text(paper: PaperRecord, table_ids: Sequence[str]) -> str:
    """The cells of the named tables as pipe-separated rows (amendment E: cells, not a picture)."""
    blocks = []
    for table in paper.tables:
        if table.id not in table_ids:
            continue
        head = f"[table {table.id} on page {table.page}]"
        if table.caption:
            head += f"\n{clip(table.caption, 400)}"
        rows = ["| " + " | ".join(clip(str(cell), TABLE_CELL_CHARS) for cell in row) + " |"
                for row in table.rows]
        blocks.append("\n".join([head] + (rows or ["(ingestion read no cells)"])))
    return "\n\n".join(blocks)


def context_blocks(paper: PaperRecord, pages: Sequence[int], *, image_pages: Sequence[int] = (),
                   table_ids: Sequence[str] = (), cache: bool = True) -> list[dict[str, Any]]:
    """Page text (+ the page image only where one is needed), then the table cells.

    Amendment E: a text source is read from text alone — a page raster costs tokens, adds nothing
    to a sentence that is already in the text layer, and invites the model to read a figure it was
    not asked to read. Pages carrying a table the extractor must look at do get their image.

    The blocks are the INVARIANT part of an extractor call — the same pages, in the same order,
    whatever question is asked of them — so the last one carries the `cache_control` marker and
    the variable prompt goes after it (amendment B; task 15 §A2a). Two calls share the entry only
    when the model AND the output schema match too, which is why the two text variants (Opus and
    Sonnet) still pay separately: they are meant to be independent readers.
    """
    blocks: list[dict[str, Any]] = []
    for number in pages:
        blocks += page_blocks(paper, [number], with_text=True,
                              with_images=number in set(image_pages))
    cells = table_text(paper, table_ids)
    if cells:
        blocks.append(text_block(cells))
    if cache and blocks:
        blocks[-1] = {**blocks[-1], "cache_control": dict(EPHEMERAL)}
    return blocks


# ----------------------------------------------------------------------------- candidates
def candidate_id(dataset_id: str, outcome_key: str, extractor_id: str, index: int,
                 group: str | None = None) -> str:
    return f"{dataset_id}:{outcome_key}:{group or '-'}:{extractor_id}#{index}"


def _tokens(label: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", normalize(label)) if len(t) >= 3}


def _hits(written: set[str], distinctive: set[str]) -> int:
    """How many of the written words name one of the words that tell the two groups apart."""
    return sum(any(word == other or word in other or other in word for other in distinctive)
               for word in written)


def group_label_check(group: str | None, label_as_written: str, dataset: DatasetSpec) -> str:
    """'' when the label the model wrote belongs to the group key it chose, else a warning.

    Only the words that *distinguish* the two groups count: both labels usually share most of their
    words ("old subjects" / "young subjects"), so a whole-string similarity flags nearly everything
    (it did, on the first live run). A swapped group mapping is the worst error a meta-analysis can
    make, so the mismatch travels with the candidate instead of being quietly corrected here.
    """
    if group not in ("A", "B") or not label_as_written.strip():
        return ""
    own_label = dataset.group_a.label if group == "A" else dataset.group_b.label
    other_label = dataset.group_b.label if group == "A" else dataset.group_a.label
    own, other = _tokens(own_label), _tokens(other_label)
    written = _tokens(label_as_written)
    if not written or own == other:
        return ""
    own_only, other_only = own - other, other - own
    if not own_only or not other_only:
        return ""
    if _hits(written, other_only) <= _hits(written, own_only):
        return ""
    other_key = "B" if group == "A" else "A"
    return (f"group label mismatch: extractor put {label_as_written!r} in group {group}, but that "
            f"label matches group {other_key} ({other_label!r})")
