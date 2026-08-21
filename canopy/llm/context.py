"""Context builders: turn deterministic ingestion artefacts into Claude content blocks.

Everything an agent sees about a paper is assembled here, so provenance stays traceable: page
text comes from `pages/pNNN.txt`, page images from `pages/pNNN.png` and figure images from
`figures/<id>.claude.png` — all pre-sized by `canopy.ingest.images.prepare_for_claude`, so
coordinates a model reports are pixels of exactly the image we sent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ingest.pdf import FigureRegion, PageRecord, PaperRecord
from .client import EPHEMERAL, LLMClient, image_block
from .errors import LiveCallsDisabled, LLMError

FILES_API_BETA = "files-api-2025-04-14"
FILE_ID_NAME = "file_id.txt"

#: how a reviewer's `re_extract` hint is put to a reader. ONE phrasing, here rather than beside
#: either consumer, because the text extractors and the digitiser's read-out must say the same
#: thing about it: a hint is one more place to look, ADDED to the locations the mapper found and
#: never a replacement for them. A reviewer saying where a value is does not un-say where the map
#: looked, and a reader given only the hint would stop being a second opinion about the paper.
REVIEWER_HINT_LABEL = "a reviewer says the value is at"


#: §C3: how a reviewer's inclusion ruling is put to the MAPPER. Same shape and same reason as the
#: hint above — the words go in the prompt TEXT, so a re-map asked after a person overruled the
#: mapper is a different question with a different cache key, and cannot come back as the cached
#: "not eligible, no datasets" answer the reviewer was overruling. Additive: the protocol and the
#: roster still say everything they said, and the ruling is one more thing the mapper is told.
REVIEWER_RULING_LABEL = "a reviewer has ruled this paper eligible under the protocol"


def reviewer_ruling_line(rule: str = "", quote: str = "") -> str:
    """`- <the ruling>; map its datasets`, with the criterion and the words it rests on.

    Empty when no reviewer has ruled, so an ordinary map is byte-identical to the map it always
    was and no paper re-buys one.
    """
    said = " ".join(str(rule or "").split())[:500]
    evidence = " ".join(str(quote or "").split())[:500]
    return "\n".join(part for part in (
        f"- {REVIEWER_RULING_LABEL}; map its datasets, groups and sources as you would for any "
        f"eligible paper. Do not re-decide eligibility: it has been decided by a person.",
        f"  the criterion they decided under: {said}" if said else "",
        f"  the paper's own words they relied on: {evidence}" if evidence else "") if part)


def reviewer_hint_line(hint: str) -> str:
    """`- a reviewer says the value is at: <hint>`, or "" when no reviewer has said anything.

    In the prompt TEXT rather than in a parameter beside it, which is what puts it in the cache
    key: a cached reading taken without the hint is a reading of a different question, and reusing
    it would let a resume report a re-extraction it never bought.
    """
    text = " ".join(str(hint or "").split())[:1000]
    return f"- {REVIEWER_HINT_LABEL}: {text}" if text else ""


# ----------------------------------------------------------------------------- lookups
def _page(paper: PaperRecord, number: int) -> PageRecord:
    if not isinstance(number, int) or number < 1 or number > len(paper.pages):
        raise ValueError(f"page {number!r} out of range for {paper.filename} "
                         f"(1..{len(paper.pages)})")
    return paper.pages[number - 1]


def _figure(paper: PaperRecord, fig_id: str) -> FigureRegion:
    for fig in paper.figures:
        if fig.id == fig_id:
            return fig
    raise KeyError(f"no figure {fig_id!r} in {paper.filename} "
                   f"(have {[f.id for f in paper.figures]})")


def _asset(paper: PaperRecord, rel: str) -> Path:
    path = Path(paper.out_dir) / rel
    if not rel or not path.exists():
        raise FileNotFoundError(f"missing ingestion artefact {rel!r} under {paper.out_dir}")
    return path


def text_block(text: str, cache: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = dict(EPHEMERAL)
    return block


# ----------------------------------------------------------------------------- builders
def page_blocks(paper: PaperRecord, pages: Sequence[int], with_text: bool = True,
                with_images: bool = True) -> list[dict[str, Any]]:
    """Per page: a `[page N text]` text block then the page image (in the given page order)."""
    blocks: list[dict[str, Any]] = []
    for number in pages:
        page = _page(paper, number)
        if with_text:
            blocks.append(text_block(f"[page {number} text]\n{paper.page_text(number)}"))
        if with_images:
            blocks.append(image_block(_asset(paper, page.png)))
    return blocks


def figure_blocks(paper: PaperRecord, fig_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Per figure: a caption/locator text block then the Claude-sized figure crop."""
    blocks: list[dict[str, Any]] = []
    for fig_id in fig_ids:
        fig = _figure(paper, fig_id)
        caption = (fig.caption or "").strip() or "(no caption found by ingestion)"
        header = (f"[figure {fig.id} on page {fig.page}"
                  f"{f' — printed label {fig.label}' if fig.label else ''}]\n{caption}")
        blocks.append(text_block(header))
        blocks.append(image_block(_asset(paper, fig.claude_png)))
    return blocks


def excerpt_for(paper: PaperRecord, pages: Sequence[int] = (), figures: Iterable[str] = (),
                with_text: bool = True, with_images: bool = True,
                cache: bool = True) -> list[dict[str, Any]]:
    """Pages then figures, with `cache_control` on the last block so the whole prefix is cached.

    Only mark the excerpt cacheable when at least two calls will reuse it (amendment B).
    """
    blocks = page_blocks(paper, pages, with_text=with_text, with_images=with_images)
    blocks += figure_blocks(paper, figures)
    if cache and blocks:
        blocks[-1] = {**blocks[-1], "cache_control": dict(EPHEMERAL)}
    return blocks


# ----------------------------------------------------------------------------- Files API
def upload_pdf(client: LLMClient, pdf_path: str | Path, out_dir: str | Path | None = None) -> str:
    """Upload a PDF once and remember its `file_id` in `<out_dir>/file_id.txt`.

    The id lets every agent reference the whole document without re-sending base64 bytes.
    """
    pdf_path = Path(pdf_path)
    directory = Path(out_dir) if out_dir is not None else pdf_path.parent
    cache_file = directory / FILE_ID_NAME
    if cache_file.exists():
        cached = cache_file.read_text().strip()
        if cached:
            return cached

    uploader = getattr(client.provider, "upload_file", None)
    if uploader is None:
        raise LLMError(f"provider {getattr(client.provider, 'name', '?')!r} cannot upload files")
    if getattr(client.provider, "is_live", False) and not client.live:
        raise LiveCallsDisabled(
            f"uploading {pdf_path.name} needs CANOPY_LIVE=1 (or a cached {FILE_ID_NAME})")

    file_id = uploader(pdf_path, "application/pdf", [FILES_API_BETA])
    if not file_id:
        raise LLMError(f"Files API returned no id for {pdf_path}")
    directory.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(file_id + "\n")
    return file_id
