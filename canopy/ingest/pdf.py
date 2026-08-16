"""PDF ingestion: deterministic parsing of a paper into pages, text, words, tables, figure regions and rasters.

Everything here is code (no LLM). Outputs are written under ``<out_dir>/`` and summarized in a ``PaperRecord``:

    paper.json                 PaperRecord (metadata, pages, figures, tables, files)
    pages/p001.txt             page text in reading order (blocks sorted by column, then y)
    pages/p001.words.json      words with bboxes (PDF points)             [for quote grounding / highlighting]
    pages/p001.png             page raster at PAGE_DPI, pre-resized to Claude's high-res tier (coords 1:1)
    figures/figNN.png          high-res crop of the figure region (+ caption margin)
    figures/figNN.claude.png   the same crop prepared for Claude (resized/upscaled per tier rules)
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image

from .images import prepare_for_claude


def _union(a: pymupdf.Rect, b: pymupdf.Rect) -> pymupdf.Rect:
    """Bounding box of two rects that also works for degenerate ones — `Rect | Rect` follows MuPDF and IGNORES an
    'empty' operand (zero width or height), so straight lines would never grow a cluster."""
    return pymupdf.Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _overlaps(a, b) -> bool:
    """Inclusive rectangle overlap that also works for degenerate (zero-area) rects such as straight lines —
    `pymupdf.Rect.intersects` returns False for those and silently drops axes, ticks and error-bar stems."""
    return not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0)


PAGE_DPI = 200          # page rasters sent to text/table extractors
FIG_DPI = 500           # crops of vector figures
RASTER_UPSCALE = 4      # crops of embedded raster figures (native pixels × this, capped by tier)
CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|figs\.?)\s*(s?\d+[a-z]?)\b", re.I)
TABLE_CAP_RE = re.compile(r"^\s*table\s+(s?\d+)\b", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


@dataclass
class Bbox:
    x0: float; y0: float; x1: float; y1: float

    @classmethod
    def from_rect(cls, r) -> "Bbox":
        return cls(float(r.x0), float(r.y0), float(r.x1), float(r.y1))

    def rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.x0, self.y0, self.x1, self.y1)


@dataclass
class TableRecord:
    id: str
    page: int                       # 1-based
    bbox: Bbox
    caption: str
    rows: list[list[str]]           # cell text (header row first if detected)


@dataclass
class FigureRegion:
    id: str                         # "fig03" (order of appearance)
    page: int                       # 1-based
    bbox: Bbox                      # region incl. all panels (PDF points, page coords)
    caption: str
    label: str                      # e.g. "Fig. 2" as printed (best effort)
    kind: str                       # raster | vector | mixed | caption_only
    n_images: int
    n_drawings: int
    native_px: tuple[int, int] | None   # for single raster images: native pixel size
    crop_png: str                   # path (relative to out_dir)
    claude_png: str                 # path (relative to out_dir)
    crop_dpi: float                 # effective DPI of crop_png relative to PDF points
    claude_scale: float             # claude_png px / crop_png px
    confidence: float               # heuristic 0..1 for region correctness


@dataclass
class PageRecord:
    number: int                     # 1-based
    width_pt: float
    height_pt: float
    n_chars: int
    n_images: int
    n_drawings: int
    png: str                        # relative path of the Claude-ready raster
    png_scale: float                # png px per PDF point
    text_file: str
    words_file: str


@dataclass
class PaperRecord:
    sha256: str
    source_path: str
    filename: str
    n_pages: int
    title: str
    doi: str
    first_page_text: str
    pages: list[PageRecord]
    figures: list[FigureRegion]
    tables: list[TableRecord]
    has_text_layer: bool
    out_dir: str
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def load(cls, out_dir: str | Path) -> "PaperRecord":
        d = json.loads((Path(out_dir) / "paper.json").read_text())
        d["pages"] = [PageRecord(**p) for p in d["pages"]]
        for f in d["figures"]:
            f["bbox"] = Bbox(**f["bbox"])
            if f["native_px"] is not None:
                f["native_px"] = tuple(f["native_px"])
        d["figures"] = [FigureRegion(**f) for f in d["figures"]]
        for t in d["tables"]:
            t["bbox"] = Bbox(**t["bbox"])
        d["tables"] = [TableRecord(**t) for t in d["tables"]]
        return cls(**d)

    def page_text(self, page: int) -> str:
        return (Path(self.out_dir) / self.pages[page - 1].text_file).read_text()


# ----------------------------------------------------------------------------- helpers
def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reading_order_text(page: pymupdf.Page) -> str:
    """Blocks sorted into columns (by x-center), then top-to-bottom; good enough for quote grounding."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not blocks:
        return ""
    mid = page.rect.width / 2
    # two-column heuristic: if many blocks are clearly left or right of the middle, split into columns
    left = [b for b in blocks if b[2] < mid + 20]
    right = [b for b in blocks if b[0] > mid - 20]
    if len(left) >= 3 and len(right) >= 3:
        span = [b for b in blocks if not (b[2] < mid + 20) and not (b[0] > mid - 20)]  # spanning blocks (title etc.)
        ordered = sorted(span, key=lambda b: b[1]) + sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
    else:
        ordered = sorted(blocks, key=lambda b: (round(b[1] / 5), b[0]))
    return "\n".join(b[4].rstrip() for b in ordered)


def _guess_title(doc: pymupdf.Document) -> str:
    meta = (doc.metadata or {}).get("title") or ""
    meta = meta.strip()
    if meta and len(meta) > 12 and not meta.lower().startswith(("untitled", "microsoft", "doi", "pii", "jcn")):
        return meta
    # largest-font text in the top 45% of page 1 (multi-line titles merged)
    page = doc[0]
    limit = page.rect.height * 0.45
    lines = []  # (size, y, text)
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0 or b["bbox"][1] > limit:
            continue
        for l in b["lines"]:
            txt = _norm_ws("".join(sp["text"] for sp in l["spans"]))
            if len(txt) < 3:
                continue
            size = max(sp["size"] for sp in l["spans"])
            lines.append((round(size, 1), l["bbox"][1], txt))
    if not lines:
        return meta
    sizes = sorted({sz for sz, _, _ in lines}, reverse=True)
    for sz in sizes:  # largest size whose merged text is a plausible title
        cand = " ".join(t for s_, _, t in sorted(lines, key=lambda x: x[1]) if s_ == sz)
        if 20 <= len(cand) <= 300 and not re.search(r"@|\bdoi\b|www\.|http", cand, re.I):
            return cand
    return meta


def _norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


_BODY_HEAD = r"^\s*(?i:fig(?:ure)?\.?|figs\.?)\s*[sS]?\d+[a-zA-Z]?\s*(?:[,–-]\s*\d+[a-zA-Z]?\s*)?"
#: "Figure 2 illustrates ..." — a body sentence, not a caption (verbs: any case)
BODY_VERB_RE = re.compile(_BODY_HEAD +
                          r"(illustrates|shows|show|depicts|displays|reveals|presents|summarizes|summarises|"
                          r"plots|gives|indicates|demonstrates|provides|compares|contains|represents|is|are|and|"
                          r"also|thus|therefore|which|where|as|for|in|of|but|that|"
                          r"revealed|showed|depicted|illustrated|confirms|suggests|highlights|makes)\b", re.I)
#: "Fig. 5 the ..." with a LOWERCASE function word is body text; "Fig. 5 The changes in ..." (capitalised) is how
#: Springer/Elsevier set real captions, so these words are matched case-sensitively.
BODY_FUNC_RE = re.compile(_BODY_HEAD + r"(the|a|an|we|this|these|it)\b")


def caption_score(txt: str) -> float:
    """1.0 = looks like a real figure caption ('Fig. 3. Mean ...', 'Figure 2 | ...'); 0 = body sentence."""
    t = _norm_ws(txt)
    if not CAPTION_RE.match(t):
        return 0.0
    if BODY_VERB_RE.match(t) or BODY_FUNC_RE.match(t):
        return 0.1
    m = CAPTION_RE.match(t)
    rest = t[m.end():].lstrip()
    score = 0.6
    if rest[:1] in ".:|" or rest[:1] == "":
        score += 0.3
    elif rest[:1].isupper() or rest[:1] in "([":
        score += 0.2
    if len(t) > 400 and score < 0.9:
        score -= 0.1
    return min(score, 1.0)


def _find_doi(text: str) -> str:
    m = DOI_RE.search(text)
    return m.group(1).rstrip(".;,)") if m else ""


def _cluster_rects(rects: list[pymupdf.Rect], gap: float = 12.0) -> list[pymupdf.Rect]:
    """Greedy union of rectangles that are within `gap` points of each other."""
    clusters: list[pymupdf.Rect] = []
    for r in sorted(rects, key=lambda r: (r.y0, r.x0)):
        merged = False
        for i, c in enumerate(clusters):
            grown = pymupdf.Rect(c.x0 - gap, c.y0 - gap, c.x1 + gap, c.y1 + gap)
            if _overlaps(grown, r):
                clusters[i] = _union(c, r)
                merged = True
                break
        if not merged:
            clusters.append(pymupdf.Rect(r))
    # second pass to merge clusters that became adjacent
    changed = True
    while changed:
        changed = False
        out: list[pymupdf.Rect] = []
        for c in clusters:
            for i, o in enumerate(out):
                grown = pymupdf.Rect(o.x0 - gap, o.y0 - gap, o.x1 + gap, o.y1 + gap)
                if _overlaps(grown, c):
                    out[i] = _union(o, c)
                    changed = True
                    break
            else:
                out.append(c)
        clusters = out
    return clusters


def _caption_blocks(page: pymupdf.Page):
    """(rect, normalized text, label, score) for blocks that start like a figure caption or an in-text reference."""
    caps = []
    for b in page.get_text("blocks"):
        if b[6] != 0:
            continue
        txt = _norm_ws(b[4])
        m = CAPTION_RE.match(txt)
        if m and len(txt) > 8:
            caps.append((pymupdf.Rect(b[:4]), txt, m.group(0).strip(), caption_score(txt)))
    return caps


def _figure_regions(page: pymupdf.Page, page_no: int) -> list[dict]:
    """Deterministic figure-region proposals. Candidate graphics = raster image placements + clusters of vector
    drawings; each candidate is attached to the best-scoring nearby caption (real captions beat body sentences like
    'Figure 2 illustrates ...'); candidates sharing a caption are merged (multi-panel figures)."""
    W, H = page.rect.width, page.rect.height
    img_rects: list[pymupdf.Rect] = []
    native: dict[tuple[int, int], tuple[int, int]] = {}
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            if r.width >= 40 and r.height >= 40 and r.width < W * 0.98:
                img_rects.append(pymupdf.Rect(r))
                native[(round(r.x0), round(r.y0))] = (im[2], im[3])
    draw_rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        if r.width > W * 0.9 and r.height < 3:      # page-wide rules
            continue
        if r.width < 0.5 and r.height < 0.5:
            continue
        draw_rects.append(pymupdf.Rect(r))
    clusters = [c for c in _cluster_rects(draw_rects, gap=10) if c.width > 60 and c.height > 40]
    counts = [sum(1 for r in draw_rects if _overlaps(c, r)) for c in clusters]
    clusters = [c for c, n in zip(clusters, counts) if n >= 15]
    counts = [n for n in counts if n >= 15]
    # drop vector clusters that are really tables (overlap a detected table region heavily)
    candidates = [dict(rect=r, kind="raster", n_draw=0) for r in img_rects] + \
                 [dict(rect=c, kind="vector", n_draw=n) for c, n in zip(clusters, counts)]
    caps = _caption_blocks(page)

    def ranked_captions(r: pymupdf.Rect):
        """All plausible captions for a graphic, best first: (cap, relation, key)."""
        out = []
        for cap in caps:
            cap_rect, cap_txt, label, score = cap
            rel = None
            horiz = min(r.x1, cap_rect.x1) - max(r.x0, cap_rect.x0)
            vert = min(r.y1, cap_rect.y1) - max(r.y0, cap_rect.y0)
            dist = None
            if horiz > 0.3 * min(r.width, cap_rect.width):
                below = cap_rect.y0 - r.y1          # caption below graphic (usual)
                above = r.y0 - cap_rect.y1          # caption above graphic (some journals)
                if -0.3 * r.height <= below <= 260:
                    dist, rel = max(below, 0), "below"
                elif -0.3 * cap_rect.height <= above <= 60:
                    dist, rel = max(above, 0) + 30, "above"
                elif vert > 0 and cap_rect.y0 > r.y0 + 0.4 * r.height:
                    dist, rel = 20, "inside"        # caption inside the lower part of a wide figure's span
            elif vert > 0.5 * cap_rect.height:
                gap = max(r.x0 - cap_rect.x1, cap_rect.x0 - r.x1)
                if 0 <= gap <= 45:                  # side (margin) caption, vertically aligned
                    dist, rel = 50 + gap, "side"
            if dist is None:
                continue
            out.append((cap, rel, (0 if score >= 0.6 else 1, dist)))   # real captions first, then nearest
            # (a band, not the raw score: a slightly 'nicer' caption 200 pt away must not beat the adjacent one)
        out.sort(key=lambda t: t[2])
        return out

    def assign_captions():
        """Best caption per graphic, then resolve conflicts. When several graphics claim ONE caption with mixed
        relations (one sees it below itself, another above — i.e. the caption sits between two different
        figures) the graphics are re-paired with their plausible captions in reading order, because figure
        numbers increase down the page. Multi-panel figures (all claims 'below'/'inside') still merge."""
        ranked = {id(c): ranked_captions(c["rect"]) for c in candidates}
        choice = {id(c): (ranked[id(c)][0][:2] if ranked[id(c)] else (None, None)) for c in candidates}
        claims: dict[int, list] = {}
        for c in candidates:
            cap, rel = choice[id(c)]
            if cap is not None:
                claims.setdefault(id(cap), []).append((c, rel))
        for cap_id, cl in claims.items():
            rels = {rel for _, rel in cl}
            if len(cl) < 2 or "above" not in rels or rels == {"above"}:
                continue
            graphics = sorted((c for c, _ in cl), key=lambda c: (c["rect"].y0, c["rect"].x0))
            pool = {}
            for c in graphics:
                for cap, rel, _ in ranked[id(c)]:
                    pool[id(cap)] = cap
            ordered = sorted(pool.values(), key=lambda cap: (cap[0].y0, cap[0].x0))
            paired = None
            if len(ordered) >= len(graphics):
                trial = list(zip(graphics, ordered))
                if all(any(id(cap) == id(t[0]) for t in ranked[id(c)]) for c, cap in trial):
                    paired = trial
            if paired is not None:
                for c, cap in paired:
                    rel = next(t[1] for t in ranked[id(c)] if id(t[0]) == id(cap))
                    choice[id(c)] = (cap, rel)
            else:                                   # fallback: the 'above' claimants take their next-best caption
                for c, rel in cl:
                    if rel == "above":
                        alt = [t for t in ranked[id(c)] if id(t[0]) != cap_id]
                        choice[id(c)] = alt[0][:2] if alt else (None, None)
        return {k: v[0] for k, v in choice.items()}

    assignment = assign_captions()
    groups: dict[str, dict] = {}
    loose = []
    for c in candidates:
        cap = assignment[id(c)]
        if cap is None:
            loose.append(c)
            continue
        key = cap[1][:80]
        g = groups.setdefault(key, dict(rects=[], kinds=set(), n_img=0, n_draw=0, cap=cap))
        g["rects"].append(c["rect"]); g["kinds"].add(c["kind"])
        g["n_img"] += int(c["kind"] == "raster"); g["n_draw"] += c["n_draw"]
    regions = []
    for g in groups.values():
        union = g["rects"][0]
        for r in g["rects"][1:]:
            union = _union(union, r)
        cap_rect, cap_txt, label, score = g["cap"]
        kind = "raster" if g["kinds"] == {"raster"} else "vector" if g["kinds"] == {"vector"} else "mixed"
        npx = native.get((round(union.x0), round(union.y0))) if (kind == "raster" and g["n_img"] == 1) else None
        regions.append(dict(bbox=union, caption=cap_txt, label=label, kind=kind, n_images=g["n_img"],
                            n_drawings=g["n_draw"], native_px=npx, confidence=min(0.95, 0.55 + 0.4 * score)))
    # real captions that received no graphics: propose the area above the caption (low confidence)
    matched = {id(g["cap"]) for g in groups.values()}
    for cap in caps:
        cap_rect, cap_txt, label, score = cap
        if score < 0.6 or id(cap) in matched:
            continue
        top = 0.0
        for b in page.get_text("blocks"):
            if b[6] == 0 and b[3] < cap_rect.y0 - 5 and b[3] > top and len(_norm_ws(b[4])) > 80 \
                    and min(b[2], cap_rect.x1) - max(b[0], cap_rect.x0) > 20:
                top = b[3]
        r = pymupdf.Rect(cap_rect.x0, max(top, cap_rect.y0 - 320), cap_rect.x1, cap_rect.y0)
        if r.height > 40:
            regions.append(dict(bbox=r, caption=cap_txt, label=label, kind="caption_only", n_images=0,
                                n_drawings=0, native_px=None, confidence=0.3))
    for c in loose:   # graphics without any caption nearby: keep if large (caption may be on the next page)
        r = c["rect"]
        if r.width * r.height <= 0.08 * W * H:
            continue
        if page_no == 1 and r.y1 < 0.2 * H:      # journal banner / logo rules on the title page
            continue
        if c["kind"] == "vector" and c["n_draw"] / (r.width * r.height / 1e4) < 1.0:
            continue                             # a few page-wide rules and boxes, not a chart (sparse line art)
        regions.append(dict(bbox=r, caption="", label="", kind=c["kind"], n_images=int(c["kind"] == "raster"),
                                n_drawings=c["n_draw"], native_px=native.get((round(r.x0), round(r.y0))), confidence=0.4))
    regions.sort(key=lambda d: (d["bbox"].y0, d["bbox"].x0))
    return regions


def _tables(page: pymupdf.Page, page_no: int) -> list[TableRecord]:
    out = []
    try:
        tabs = page.find_tables()
    except Exception:
        return out
    caps = [(pymupdf.Rect(b[:4]), b[4].strip()) for b in page.get_text("blocks") if b[6] == 0 and TABLE_CAP_RE.match(b[4].strip())]
    for i, t in enumerate(tabs.tables):
        try:
            rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in t.extract()]
        except Exception:
            continue
        if len(rows) < 2:
            continue
        r = pymupdf.Rect(t.bbox)
        cap = ""
        best = 1e9
        for cr, ct in caps:
            d = abs(cr.y1 - r.y0) if cr.y1 <= r.y0 + 5 else abs(r.y1 - cr.y0)
            if d < best:
                best, cap = d, ct
        out.append(TableRecord(id=f"p{page_no}t{i+1}", page=page_no, bbox=Bbox.from_rect(r), caption=cap, rows=rows))
    return out


# ----------------------------------------------------------------------------- main entry
def ingest_pdf(path: str | Path, out_dir: str | Path, page_dpi: int = PAGE_DPI, fig_dpi: int = FIG_DPI,
               render_pages: bool = True) -> PaperRecord:
    path = Path(path); out = Path(out_dir)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(path)
    sha = sha256_of(path)
    pages: list[PageRecord] = []
    figures: list[FigureRegion] = []
    tables: list[TableRecord] = []
    warnings: list[str] = []
    total_chars = 0
    fig_counter = 0
    for i, page in enumerate(doc):
        n = i + 1
        text = _reading_order_text(page)
        total_chars += len(text)
        (out / "pages" / f"p{n:03d}.txt").write_text(text)
        words = [dict(x0=w[0], y0=w[1], x1=w[2], y1=w[3], text=w[4], block=w[5], line=w[6], word=w[7])
                 for w in page.get_text("words")]
        (out / "pages" / f"p{n:03d}.words.json").write_text(json.dumps(words))
        png_rel, png_scale = "", 0.0
        if render_pages:
            pix = page.get_pixmap(dpi=page_dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            prep = prepare_for_claude(img)
            png_rel = f"pages/p{n:03d}.png"
            prep.image.save(out / png_rel, optimize=True)
            png_scale = (page_dpi / 72.0) * prep.scale
        pages.append(PageRecord(number=n, width_pt=page.rect.width, height_pt=page.rect.height, n_chars=len(text),
                                n_images=len(page.get_images()), n_drawings=len(page.get_drawings()), png=png_rel,
                                png_scale=png_scale, text_file=f"pages/p{n:03d}.txt", words_file=f"pages/p{n:03d}.words.json"))
        # tables
        tables.extend(_tables(page, n))
        # figures
        for reg in _figure_regions(page, n):
            fig_counter += 1
            fid = f"fig{fig_counter:02d}"
            r = reg["bbox"]
            margin = 6
            clip = pymupdf.Rect(max(0, r.x0 - margin), max(0, r.y0 - margin), min(page.rect.width, r.x1 + margin),
                                min(page.rect.height, r.y1 + margin))
            if reg["kind"] == "raster" and reg["native_px"] and reg["n_images"] == 1:
                # render at the native pixel density × upscale (capped) so we don't invent detail beyond the source
                nat_w = reg["native_px"][0]
                dpi = max(72, min(fig_dpi, 72 * nat_w / max(clip.width, 1) * RASTER_UPSCALE))
            else:
                dpi = fig_dpi
            pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
            crop = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            crop_rel = f"figures/{fid}.png"
            crop.save(out / crop_rel, optimize=True)
            prep = prepare_for_claude(crop)
            claude_rel = f"figures/{fid}.claude.png"
            prep.image.save(out / claude_rel, optimize=True)
            figures.append(FigureRegion(id=fid, page=n, bbox=Bbox.from_rect(clip), caption=reg["caption"], label=reg["label"],
                                        kind=reg["kind"], n_images=reg["n_images"], n_drawings=reg["n_drawings"],
                                        native_px=reg["native_px"], crop_png=crop_rel, claude_png=claude_rel,
                                        crop_dpi=dpi, claude_scale=prep.scale, confidence=reg["confidence"]))
    first_text = doc[0].get_text() if len(doc) else ""
    has_text = total_chars > 200 * max(1, len(doc)) * 0.2
    if not has_text:
        warnings.append("little or no text layer — scanned PDF? OCR fallback recommended")
    rec = PaperRecord(sha256=sha, source_path=str(path), filename=path.name, n_pages=len(doc), title=_guess_title(doc),
                      doi=_find_doi(first_text) or _find_doi(" ".join(str(v) for v in (doc.metadata or {}).values() if v)),
                      first_page_text=first_text[:3000], pages=pages, figures=figures, tables=tables,
                      has_text_layer=has_text, out_dir=str(out), warnings=warnings)
    (out / "paper.json").write_text(rec.to_json())
    return rec
