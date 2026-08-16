"""Per-value provenance: the page crop with the quote highlighted, or the figure with its overlay.

A number in a meta-analysis is only as good as the ability to go back and look at it. Every
`Candidate` carries a page and either a quote (text/table) or a crop and pixel coordinates
(figure); this module turns that into an image a human can check in one glance, and a JSON record
of what was claimed beside it.

The quote search is deliberately literal: it matches the extractor's quote against the words
PyMuPDF read off the page, and when it cannot find them it says so (`matched=False`) and still
writes the page image. A provenance crop that silently highlights the wrong place would be worse
than none.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..ingest.pdf import FigureRegion, PaperRecord
from ..models import Candidate
from .naming import safe_name
from .theme import ACCENT

__all__ = ["quote_crop", "figure_provenance", "provenance_bundle", "normalise_words"]

_PUNCT = re.compile(r"[^\w.\-+]+", re.UNICODE)
#: unicode minus / thin spaces / non-breaking hyphens survive a PDF and break a naive comparison
_TRANSLATE = {ord("−"): "-", ord("–"): "-", ord("—"): "-", ord("‑"): "-",
              ord(" "): " ", ord(" "): " ", ord(" "): " ", ord(" "): " "}
MIN_RUN = 3                 # a match shorter than this is noise, not evidence
PAD_PT = 14.0               # padding around the highlighted words, in PDF points
CONTEXT_PT = 26.0           # extra context above/below so the crop is readable


def normalise_words(text: str) -> list[str]:
    """Words as they compare: NFKC, unicode dashes folded, punctuation dropped, lower-cased."""
    folded = unicodedata.normalize("NFKC", str(text)).translate(_TRANSLATE)
    return [w for w in (_PUNCT.sub("", part).lower() for part in folded.split()) if w]


def _page_words(paper: PaperRecord, page: int) -> list[dict[str, Any]]:
    record = paper.pages[page - 1]
    return json.loads((Path(paper.out_dir) / record.words_file).read_text())


def _longest_run(needle: Sequence[str], haystack: Sequence[str]) -> tuple[int, int]:
    """The longest contiguous run of `needle` that appears verbatim in `haystack`.

    Returns `(start_in_haystack, length)`. Longest-run rather than exact-match because an
    extractor's quote often carries a leading/trailing fragment the PDF tokenised differently.
    """
    best_start, best_len = -1, 0
    for i in range(len(needle)):
        for j in range(len(haystack)):
            length = 0
            while (i + length < len(needle) and j + length < len(haystack)
                   and needle[i + length] == haystack[j + length]):
                length += 1
            if length > best_len:
                best_start, best_len = j, length
    return best_start, best_len


def quote_crop(paper: PaperRecord, page: int, quote: str, out_path: str | Path, *,
               pad_pt: float = PAD_PT, context_pt: float = CONTEXT_PT) -> dict[str, Any]:
    """Crop the page around `quote` and highlight it; returns what was found, including failures."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"page": page, "quote": quote, "matched": False, "matched_words": 0,
                              "bbox_pt": None, "path": out, "note": ""}
    if page < 1 or page > len(paper.pages):
        result["note"] = f"page {page} is outside this paper (1..{len(paper.pages)})"
        return result
    record = paper.pages[page - 1]
    png = Path(paper.out_dir) / record.png
    if not record.png or not png.exists():
        result["note"] = "the page was not rasterised during ingestion"
        return result

    from PIL import Image, ImageDraw

    image = Image.open(png).convert("RGB")
    scale = record.png_scale or (image.width / max(record.width_pt, 1e-6))
    words = _page_words(paper, page)
    tokens = [normalise_words(w["text"]) for w in words]
    flat = [(t[0], i) for i, t in enumerate(tokens) if t]
    needle = normalise_words(quote)
    start, length = _longest_run(needle, [t for t, _ in flat]) if needle and flat else (-1, 0)

    box = None
    if length >= MIN_RUN:
        matched = [words[flat[start + k][1]] for k in range(length)]
        box = (min(w["x0"] for w in matched), min(w["y0"] for w in matched),
               max(w["x1"] for w in matched), max(w["y1"] for w in matched))
        result.update(matched=True, matched_words=length, bbox_pt=[float(v) for v in box])
    else:
        result["note"] = ("this quote is not on the page as ingestion read it — the page is shown "
                          "without a highlight")

    if box is None:
        image.save(out, optimize=True)
        return result

    highlight = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(highlight)
    px = [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]
    accent = tuple(int(ACCENT[i:i + 2], 16) for i in (1, 3, 5))
    draw.rectangle(px, fill=(*accent, 46), outline=(*accent, 235), width=max(2, int(scale)))
    image = Image.alpha_composite(image.convert("RGBA"), highlight).convert("RGB")

    crop = (max(0.0, (box[0] - pad_pt) * scale), max(0.0, (box[1] - pad_pt - context_pt) * scale),
            min(float(image.width), (box[2] + pad_pt) * scale),
            min(float(image.height), (box[3] + pad_pt + context_pt) * scale))
    image.crop(tuple(int(round(v)) for v in crop)).save(out, optimize=True)
    result["crop_px"] = [float(v) for v in crop]
    return result


def _figure(paper: PaperRecord, figure_id: str) -> FigureRegion | None:
    for figure in paper.figures:
        if figure.id == figure_id:
            return figure
    return None


def figure_provenance(paper: PaperRecord, figure_id: str, out_path: str | Path, *,
                      marks: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The figure crop, with the digitiser's read-out marks drawn on it when there are any."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = _figure(paper, figure_id)
    if figure is None:
        return {"figure_id": figure_id, "path": out, "matched": False,
                "note": f"no figure {figure_id!r} in this paper"}
    source = Path(paper.out_dir) / figure.crop_png
    if not source.exists():
        return {"figure_id": figure_id, "path": out, "matched": False,
                "note": "the figure crop is missing from the ingestion output"}
    if marks:
        from ..digitize.overlay import draw_overlay

        draw_overlay(source, [dict(m) for m in marks], out)
    else:
        from PIL import Image

        Image.open(source).convert("RGB").save(out, optimize=True)
    return {"figure_id": figure_id, "path": out, "matched": True, "page": figure.page,
            "caption": figure.caption, "note": ""}


def _paper_for(candidate: Candidate, papers: Mapping[str, PaperRecord]) -> PaperRecord | None:
    return papers.get(candidate.paper_id) or (next(iter(papers.values())) if len(papers) == 1
                                              else None)


def provenance_bundle(paper: PaperRecord | Iterable[PaperRecord], candidates: Sequence[Candidate],
                      out_dir: str | Path) -> dict[str, Any]:
    """One image and one JSON record per candidate — the bundle the report and the UI both link to.

    A figure candidate keeps the digitiser's own overlay when it wrote one (that image shows the
    pixels it measured); otherwise the figure crop is copied. A text candidate gets the page crop
    with its quote highlighted.
    """
    papers = {p.sha256: p for p in ([paper] if isinstance(paper, PaperRecord) else list(paper))}
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        record = _paper_for(candidate, papers)
        entry: dict[str, Any] = {
            "candidate_id": candidate.candidate_id, "paper_id": candidate.paper_id,
            "dataset_id": candidate.dataset_id, "outcome_key": candidate.outcome_key,
            "group": candidate.group, "kind": candidate.kind, "route": candidate.route,
            "model": candidate.model, "prompt_version": candidate.prompt_version,
            "llm_call_id": candidate.llm_call_id, "page": candidate.page,
            "quote": candidate.quote, "locator": candidate.locator, "mean": candidate.mean,
            "dispersion_value": candidate.dispersion_value,
            "dispersion_type": getattr(candidate.dispersion_type, "value",
                                       candidate.dispersion_type),
            "n": candidate.n, "unit": candidate.unit, "sigma": candidate.sigma,
            "grounded": candidate.grounded, "matched": False, "crop": "", "note": "",
        }
        # a candidate id carries `:` and ends in `#N`; a browser truncates a link at the `#`,
        # so the file it points at is named with the shared sanitiser and nothing else
        target = directory / f"{safe_name(candidate.candidate_id or 'candidate')}.png"
        existing = candidate.overlay_path or candidate.crop_path
        figure_id = str((candidate.pixel_provenance or {}).get("figure_id") or "")
        if existing and record is not None and (Path(record.out_dir) / existing).exists():
            entry.update(crop=str(Path(record.out_dir) / existing), matched=True,
                         note="the digitiser's own crop/overlay")
        elif figure_id and record is not None:
            figure = figure_provenance(record, figure_id, target)
            entry.update(crop=str(figure["path"]), matched=bool(figure["matched"]),
                         note=figure.get("note", ""))
        elif candidate.quote and candidate.page and record is not None:
            crop = quote_crop(record, int(candidate.page), candidate.quote, target)
            entry.update(crop=str(crop["path"]), matched=bool(crop["matched"]),
                         note=crop.get("note", ""), bbox_pt=crop.get("bbox_pt"))
        else:
            entry["note"] = "no quote, crop or figure id — nothing to show"
        entries[candidate.candidate_id or f"candidate{len(entries)}"] = entry
    path = directory / "provenance.json"
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=1, default=str),
                    encoding="utf-8")
    return {"entries": entries, "json": path, "dir": directory}
