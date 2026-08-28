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

#: a word that is a bare number — a tick label, not prose. "3.5", "-10", "1,5" (comma decimal),
#: "−10" (U+2212, matplotlib's default minus glyph), "–10" (en dash), "20%", "0.3°".
#: The ASCII-only first cut refused 257 numeric-looking words across the three fixture PDFs and
#: scored **zero** on any matplotlib-produced axis with negative ticks — and a panel that scores
#: zero is refused with no read-outs at all. Never a wrong number; a silent hard refusal on the
#: commonest tick-label encoding in modern figures, so the minus signs and the two unit suffixes
#: that always ride a tick label are accepted.
NUMERIC_WORD_RE = re.compile(r"^[+\-−–]?\d+(?:[.,]\d+)?\s*[%°]?$")
#: how many numeric words, IN A LADDER, a panel needs before we believe its axis is calibrated.
#: A crop that carries a neighbouring panel's ladder and none of its own cannot be read: the
#: numbers a reader would use belong to a different set of axes (Heuer 2008 Fig. 2/4/6, where the
#: named panel was not even in the crop). The count is per panel and never global — a global count
#: is satisfied by the x-axis labels alone, which calibrate nothing on the value axis.
MIN_PANEL_NUMERIC = 3
#: …and how many rungs let a PANEL be read against its own ladder. Two, because two ticks are
#: `calibrate.fit_axis`'s own precondition — they define the affine map exactly; the third rung
#: only VERIFIES linearity, and the system already reads figures with ZERO ladder (under
#: `calibration_missing`), so refusing at two while reading at zero defended nothing. What the
#: original guard exists for — a crop scaled with a NEIGHBOURING panel's ladder — needs its own
#: rungs to be IN the panel's rect, which both thresholds require equally. A two-rung read is
#: capped and flagged (`calibration_two_point`), never silently trusted.
MIN_PANEL_CALIBRATED = 2
#: pad on the edges that are not grown to a landmark
PANEL_PAD = 6.0
#: how far a panel's LOWER edge may reach past its drawing cluster, when a caption block does not
#: stop it sooner. The x tick labels and the x-axis title are text, so they sit below the drawing
#: cluster: on Heuer 2008 the eight x labels of fig02/04/06 sit 0.1-1.4 pt below the crop's own
#: bottom edge and the axis title 8-13 pt below it, while the caption starts 34-39 pt below —
#: there is a landmark-shaped gap there, and 6 pt of pad does not reach it.
PANEL_GROW_DOWN = 24.0
#: the clearance a grown panel edge keeps from the landmark it grew to (a caption block, the next
#: panel down). Small, because the landmark is the thing being avoided, not approached.
LANDMARK_GAP = 1.0
#: how far to the right of a panel a ladder may sit and still be that panel's. Tick labels sit
#: ADJACENT to the axis they label — measured on this corpus they are 6-12 pt from the drawing —
#: so a ladder half a page away is never this panel's, however wide the panel happens to be.
#: The panel's own width alone was too generous a cap: on a 350 pt panel it permitted a ladder
#: 350 pt away, which is most of a page.
RIGHT_LADDER_REACH_PT = 48.0
#: growth smaller than this is not growth, it is float noise in the PDF's own coordinates. Every
#: Cressman 2010 figure has a caption `x0` equal to its graphics' `x0` to four decimal places, and
#: without a floor the "grow to the caption column" rule would move six crops by 0.0004 pt and
#: cost the byte-identity that proves the rule cannot regress that paper.
MIN_CAPTION_GROWTH = 1.0
TEXT_LAYER_PRESENT = "present"
TEXT_LAYER_NONE = "none"
#: fraction of the page height at the head and the foot inside which a short text block that
#: touches no graphic is page furniture (a folio, a running head), not figure text.
FURNITURE_BAND = 0.10
FURNITURE_MAX_CHARS = 120
#: how far apart two neighbouring gaps in one ladder may be before they are two ladders. A tick
#: ladder is evenly spaced; two panels stacked with descending axes are not, and read as one run
#: they certified a union on a ladder that is really two (measured: 50,40,30 above 20,10,0 came
#: back as a six-rung ladder). The tolerance is not 1.0 because a ladder may suppress a label —
#: Cressman 2010's Fig. 5 prints 120..20 and -20,-40 and leaves the zero off, one gap of exactly
#: 2.0x — so what is refused is a gap out of scale with its NEIGHBOUR, not a gap out of scale
#: with a constant.
LADDER_GAP_RATIO = 3.0
#: a span whose font is bold. Springer prints the panel letter of a caption in bold and the
#: English article in roman, which is the only sound way to tell "a Side view of the setup" from
#: "in a rotated field": measured on Cressman 2010, the caption letters of Fig. 1 and Fig. 3 are
#: `AdvPTimesB` and every stray article is `AdvPTimes`. Flags are not enough (this publisher sets
#: flags=4, serif, on both), so the font NAME is read as well.
BOLD_FONT_RE = re.compile(r"bold|black|heavy|semib|[a-z]B$|[a-z]-B\b|,B(?:old)?$", re.I)
#: a letter enumerating a panel inside a caption: "a: ...", "(b) ...", "b and c Top view ...",
#: "a Side view ...". Deliberately BARE — Springer sets its panel letters in bold with no
#: punctuation at all, so a pattern demanding a bracket or a colon is blind to every caption in
#: Cressman 2010 ("the a reach training trials and b aftereffect trials"). What stops a bare
#: letter being read as the English article is the ACCEPTANCE rule in `caption_panels`, not the
#: pattern: an enumeration counts only when the letters run in order from `a` and there are at
#: least two of them. Measured on all 12 real captions of the three fixture PDFs: 5 true
#: positives, 0 false positives (the six captions carrying a stray article "a" stop at one
#: letter and are rejected).
CAPTION_PANEL_RE = re.compile(r"(?:^|[\s(\[])\(?\[?([a-h])[)\]]?[:.,;]?(?=\s|$)")


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
class PanelRegion:
    """One panel of a figure: the rect a reader is handed when the map names that panel.

    A multi-panel figure's union carries several value ladders (Heuer 2008 Fig. 2 is 338 pt tall
    and holds three), so a reading taken off the union can be calibrated with the wrong one. The
    panel rect is the unit that has exactly one y ladder, and `calibrated` says whether that
    ladder is actually inside it.
    """

    id: str                         # "fig02a"
    letter: str                     # "a" — ordinal position in reading order, or the caption's own
    bbox: Bbox                      # PDF points, page coords (grown like the region)
    n_numeric: int                  # numeric words whose centres fall inside THIS panel's rect
    calibrated: bool                # n_ladder >= MIN_PANEL_CALIBRATED, or the region carries no ladder
    #: the longest LADDER among those words — a roughly collinear, value-monotone column. Three
    #: bare numbers anywhere in the rect are not an axis: Cressman 2010's Fig. 1 is an
    #: experimental-setup schematic with no value axis at all and its annotations
    #: ('1','10','5','30','30','30') certified it as calibrated under a plain count.
    n_ladder: int = 0
    crop_png: str = ""              # path (relative to out_dir); == the figure's for a 1-panel figure
    claude_png: str = ""
    crop_dpi: float = 0.0
    claude_scale: float = 1.0


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
    #: per-panel sub-rects, in reading order (top-to-bottom, then left-to-right)
    panels: list[PanelRegion] = field(default_factory=list)
    #: "none" when the region contains no FIGURE words at all (a scanned/raster figure). Page
    #: furniture and caption text are not figure words: Bock 2005's two figures are scanned
    #: rasters supplying two pooled cells, and the only thing between them and a `present` verdict
    #: was 5.6 pt of white space above a page number.
    text_layer: str = TEXT_LAYER_PRESENT
    #: the longest ladder anywhere the reader is handed — the grown region or any panel rect.
    n_region_ladder: int = 0
    #: numeric words anywhere the reader is handed. NONE of them means this figure prints no axis
    #: at all, and the per-panel assertion has no premise: it is SKIPPED rather than failed
    #: (Bock's two scanned rasters supply two pooled cells). A one-panel figure is exempt for a
    #: different reason — it cannot be reading off its neighbour's ladder.
    n_region_numeric: int = 0
    #: how far the region's left edge was grown to reach the caption block's own column (0.0 when
    #: the caption already starts at or right of the graphics — every Cressman 2010 figure)
    caption_growth_pt: float = 0.0
    #: the panels the CAPTION enumerates ("a: ... b: ... c: ..."), when it enumerates any. Read
    #: against `panels`, it is the page's own answer to "did ingestion find every panel?"
    caption_panels: list[str] = field(default_factory=list)
    #: fix F. `_panel_letters` assigns letters by ORDINAL reading order and, on its own, nothing
    #: checks them against what the panels actually print — one real figure's D-rect was lettered
    #: E, and every downstream reader faithfully read the wrong panel. `_verify_panel_letters`
    #: compares each panel's own words against the caption's per-letter descriptions: an
    #: unambiguous permutation re-binds the letters silently (the record is then RIGHT and no
    #: doubt survives); anything short of that sets this bit, and a letter-addressed read of a
    #: disputed figure prefers the full page render over a crop whose letter cannot be trusted.
    panel_labels_disputed: bool = False
    #: what the verification saw, for the human and the provenance note — which panel's text
    #: matched which letter's description on which tokens ("" when nothing was found).
    panel_label_note: str = ""

    #: True iff `n_images`/`n_drawings` were MEASURED against the page. Records written before
    #: this field hardcoded 0/0 on every caption_only region — a claim, not a measurement — and at
    #: least three such regions on disk sit over readable figures. The digitizer's no-graphics
    #: refusal requires this, so an old run's resume can never refuse a figure on a hardcoded zero.
    primitives_measured: bool = False

    #: ticket 2a: what the caption ITSELF says the error bars show — a deterministic parse of the
    #: publisher's own sentence (`caption_dispersion`), empty when the caption names nothing or
    #: names it ambiguously. One full run held ~35 review slots asking a human to read exactly
    #: this sentence off the screen. A `DispersionType` value ("SE", "SD", …) or "".
    caption_dispersion: str = ""
    #: the matched sentence, verbatim — the provenance every consumer must show
    caption_dispersion_quote: str = ""
    #: per-letter statements ("error bars in A show SD"), via `_caption_panel_texts`' own
    #: segmentation: {"a": {"type": "SD", "quote": "…"}}. Empty when segmentation cannot account
    #: for the whole caption — a scope it cannot place must not manufacture evidence.
    caption_dispersion_panels: dict[str, dict[str, str]] = field(default_factory=dict)
    #: ticket 2b: the caption's marker/series key ("circles represent adaptation with the right
    #: hand"), each {"descriptor", "series_text", "quote", "line_style"}. Explicit statements
    #: only; the digitiser resolves descriptors through its own `_marker_words` and binds series
    #: text to groups — this record never guesses either.
    caption_series_keys: list[dict] = field(default_factory=list)

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
            # records written before per-panel sub-rects existed simply have no panels
            panels = f.get("panels") or []
            f["panels"] = [PanelRegion(**{**p, "bbox": Bbox(**p["bbox"])}) for p in panels]
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


def _words_in(words: list, rect: pymupdf.Rect) -> list[str]:
    """Every word whose centre lies inside `rect` — how we ask whether a region has a text layer."""
    return [str(w[4]) for w in words
            if rect.x0 <= (w[0] + w[2]) / 2.0 <= rect.x1 and rect.y0 <= (w[1] + w[3]) / 2.0 <= rect.y1
            and str(w[4]).strip()]


def _numeric_words(words: list, rect: pymupdf.Rect) -> list[str]:
    """Bare numbers whose CENTRE lies inside `rect` — the labels that calibrate what is drawn there.

    Centres, not overlaps: a label straddling the edge of a neighbouring panel belongs to that
    neighbour, and counting it would let one panel's ladder certify another's.
    """
    return [m[0] for m in _numeric_marks(words, rect)]


def numeric_value(text) -> float | None:
    """The number a tick label carries, or None — "−12,5°" is -12.5, "block" is nothing."""
    body = str(text).strip()
    if not NUMERIC_WORD_RE.match(body):
        return None
    body = body.rstrip("%°").strip().replace("−", "-").replace("–", "-").replace(",", ".")
    try:
        return float(body)
    except ValueError:                                    # pragma: no cover - regex guarantees it
        return None


def _numeric_marks(words: list, rect: pymupdf.Rect) -> list[tuple]:
    """(text, x0, x1, y-centre, value) for every numeric word whose centre lies inside `rect`."""
    out = []
    for w in words:
        value = numeric_value(w[4])
        if value is None:
            continue
        cx, cy = (w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            out.append((str(w[4]).strip(), float(w[0]), float(w[2]), cy, value))
    return out


def _evenly_spaced(run: list[tuple]) -> list[tuple]:
    """The longest stretch of `run` whose rungs are evenly spaced down the page.

    A tick ladder is evenly spaced; two ladders stacked one above the other are not, and the
    monotone test alone cannot tell them apart when both descend — measured, `50,40,30` printed
    above `20,10,0` came back as one six-rung ladder, on exactly the union `n_ladder` exists to
    discriminate. The comparison is between NEIGHBOURING gaps rather than against a constant, so
    a ladder that suppresses one label (Cressman 2010 Fig. 5 leaves its zero off, one gap of
    exactly 2.0x) survives while a panel break (10x and up) does not.
    """
    if len(run) < 3:
        return run
    gaps = [run[i + 1][3] - run[i][3] for i in range(len(run) - 1)]
    cuts = [0]
    for i in range(1, len(gaps)):
        lo, hi = sorted((gaps[i - 1], gaps[i]))
        if lo <= 0 or hi > LADDER_GAP_RATIO * lo:
            cuts.append(i)
    cuts.append(len(gaps))
    best: list[tuple] = []
    for start, end in zip(cuts, cuts[1:]):
        stretch = run[start:end + 1]
        if len(stretch) > len(best):
            best = stretch
    return best


def _monotone_run(column: list[tuple]) -> list[tuple]:
    """The longest run of consecutive entries (already in y order) that is monotone in value AND
    evenly spaced — the two things that make a column of numbers one axis rather than two."""
    best: list[tuple] = []
    run: list[tuple] = []

    def close() -> None:
        nonlocal best
        even = _evenly_spaced(run)
        if len(even) > len(best):
            best = even

    for mark in column:
        if not run:
            run = [mark]
        elif mark[4] == run[-1][4]:                       # a repeat is not a step of a ladder
            close()
            run = [mark]
        elif len(run) == 1 or (mark[4] > run[-1][4]) == (run[-1][4] > run[-2][4]):
            run.append(mark)
        else:
            close()
            run = [run[-1], mark]
    close()
    return best


def _tick_ladder(words: list, rect: pymupdf.Rect) -> list[str]:
    """The longest LADDER of numeric words inside `rect`: a column, in order of value.

    "Three bare numbers anywhere in the rect" is not evidence that a panel carries its own value
    axis. Measured: Cressman 2010 Fig. 1 is an experimental-setup schematic with no value axis at
    all, and its diagram annotations ('1','10','5','30','30','30') certified it. What a value axis
    looks like on the page is a stack of numbers at one x, running in one direction — right- or
    centre-aligned, so the words' x RANGES overlap even where their centres do not. An x-axis
    ladder is a row, not a column, and is correctly worth nothing here: it calibrates the axis
    nobody is reading a value off.
    """
    return [m[0] for m in _ladder_marks(words, rect)]


def _ladder_marks(words: list, rect: pymupdf.Rect) -> list[tuple]:
    """`_tick_ladder`'s rungs as marks, so a caller can ask WHERE the ladder it found sits."""
    marks = _numeric_marks(words, rect)
    best: list[tuple] = []
    for anchor in marks:
        column = sorted((m for m in marks if not (m[2] < anchor[1] or anchor[2] < m[1])),
                        key=lambda m: m[3])
        run = _monotone_run(column)
        if len(run) > len(best):
            best = run
    return best


def _in_rects(word, rects: list[pymupdf.Rect]) -> bool:
    cx, cy = (word[0] + word[2]) / 2.0, (word[1] + word[3]) / 2.0
    return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in rects)


def _page_furniture(page: pymupdf.Page, graphics: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """The page's own running heads and folios — text that is ON the page, not IN the figure.

    The raster exemption used to hang on "not one word whose centre is inside the region", and on
    Bock 2005 the only thing between that and buying ZERO read-outs on two pooled cells was 5.6 pt
    of white space above the page number ('261' on p3, '262' on p4). A page number is not evidence
    that a scanned figure has a text layer. Furniture is defined by where it sits and what it
    touches, never by what it says: a short block in the head or foot band that overlaps none of
    the page's graphics.
    """
    top, bottom = FURNITURE_BAND * page.rect.height, page.rect.height * (1.0 - FURNITURE_BAND)
    out = []
    for b in page.get_text("blocks"):
        if b[6] != 0:
            continue
        txt = _norm_ws(b[4])
        if not txt or len(txt) > FURNITURE_MAX_CHARS:
            continue
        r = pymupdf.Rect(b[:4])
        if not (r.y1 <= top or r.y0 >= bottom):
            continue
        if any(_overlaps(r, g) for g in graphics):
            continue
        out.append(r)
    return out


def _in_order(letters: list[str]) -> list[str]:
    """`letters` if they are a run from `a` with at least two of them, else []."""
    if len(letters) < 2:
        return []
    return letters if letters == [chr(ord("a") + i) for i in range(len(letters))] else []


def _bold_letters(page: pymupdf.Page, cap_rect: pymupdf.Rect) -> list[str]:
    """The single letters a caption prints in BOLD, in order — the publisher's own enumeration.

    Springer sets the panel letter bold and the English article roman, and that is the only
    sound way to tell "a Side view of the setup" from "in a rotated field" without fitting a
    pattern to one paper's prose. Measured on Cressman 2010: Fig. 1 gives a, b, c and Fig. 3
    gives a, b, while every stray article in the same captions is roman and every caption that
    enumerates nothing gives []. Journals that punctuate instead ("a: ... b: ...", Heuer 2008)
    are read from the text by `caption_panels`, which needs no bold at all.
    """
    best, seen = None, []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        bx = block["bbox"]
        d = abs(bx[0] - cap_rect.x0) + abs(bx[1] - cap_rect.y0)
        if best is None or d < best[0]:
            best = (d, block)
    if best is None or best[0] > 2.0:
        return []
    bold_chars = total_chars = 0
    for line in best[1]["lines"]:
        for span in line["spans"]:
            text = str(span.get("text") or "").strip()
            bold = bool(int(span.get("flags") or 0) & 16) or bool(
                BOLD_FONT_RE.search(str(span.get("font") or "")))
            total_chars += len(text)
            if bold:
                bold_chars += len(text)
            if bold and len(text) == 1 and text.lower() in "abcdefgh" and text.lower() not in seen:
                seen.append(text.lower())
    # the same majority test the text path applies to punctuation, applied to weight: bold is
    # evidence only where it is SELECTIVE. A publisher that sets whole captions bold marks
    # nothing by it, and taking the letters out of such a caption is the very guess the
    # marker-majority rule exists to refuse ("Reaching in a rotated field; b denotes …", all
    # bold, would otherwise come back as a two-panel enumeration).
    if total_chars and bold_chars * 2 >= total_chars:
        return []
    return seen


def _other_page_content(candidates: list[dict], text_blocks: list[pymupdf.Rect],
                        caps: list, group: dict, union: pymupdf.Rect) -> list[pymupdf.Rect]:
    """Everything on this page that belongs to something OTHER than this figure.

    What bounds a panel's growth cannot be the figure's own idea of itself: the caption block is
    routinely wider than the graphic it captions, and on a two-column page the space to the right
    of a column figure belongs to the next column, not to the figure. So the page's other
    graphics, the other captions and any text block outside this figure's own column are all
    obstacles, and a panel edge stops at the nearest of them.

    A block sitting within `PANEL_GROW_DOWN` of the region's own edges is NOT an obstacle — that
    is the same landmark distance the bottom edge uses for "text that belongs to this panel", and
    it is where a panel's own tick labels and axis titles live. Beyond it, the page is somebody
    else's.
    """
    own = {id(r) for r in group["rects"]}
    near = pymupdf.Rect(union.x0 - PANEL_GROW_DOWN, union.y0 - PANEL_GROW_DOWN,
                        union.x1 + PANEL_GROW_DOWN, union.y1 + PANEL_GROW_DOWN)
    out = [c["rect"] for c in candidates if id(c["rect"]) not in own]
    out += [cap[0] for cap in caps if id(cap) != id(group["cap"])]
    out += [r for r in text_blocks if not _overlaps(r, near)]
    return out


def caption_panels(caption: str, bold: list[str] | None = None) -> list[str]:
    """The panels a caption ENUMERATES, or [] — never a guess off a bare English article.

    Two independent kinds of evidence, because journals enumerate in two ways:

    * **bold** — the publisher marked the letters itself. Passed in by the ingester, which is the
      only place the PDF's spans exist. This is what reads Cressman 2010, whose captions
      enumerate with no punctuation at all ("a Side view of the experimental setup").
    * **punctuation** — "a: ... b: ...", "(a) ... (b)", "Panel a, panel b, ...". Read from the
      text alone, and accepted only when MORE of the letters carry an enumeration mark than not:
      an enumeration is punctuated and prose is not. Without that half, "e.g. the a and b
      conditions, with c." and "Reaching in a rotated field; b denotes the baseline." both parse
      as three- and two-panel enumerations, which is a guess off English articles.

    Either way the letters must run in order from `a` and there must be at least two of them.
    """
    marked = unmarked = 0
    seen: list[str] = []
    for m in CAPTION_PANEL_RE.finditer(_norm_ws(caption)):
        letter = m.group(1).lower()
        if letter in seen:
            continue
        seen.append(letter)
        if m.group(0).rstrip()[-1:] in ")]:.,;":
            marked += 1
        else:
            unmarked += 1
    from_bold = _in_order(list(bold or []))
    if from_bold:
        return from_bold
    return _in_order(seen) if marked >= unmarked and marked else []


def _panel_letters(enumerated: list[str], n: int) -> list[str]:
    """One letter per panel, in reading order.

    Panel letters are NOT a growth target: verified on Heuer 2008, `a)`/`b)`/`c)` sit 34-39 pt left
    of the cluster union while caption and axis-title contamination begins at 26 pt, so no uniform
    pad captures the letters and excludes the caption. Identity therefore comes from ordinal
    position within the caption's own cluster list, matched against the enumeration the caption
    itself prints ("a: ... b: ... c: ..."). That is deterministic and needs no OCR of the letter.
    """
    seen = list(enumerated)
    alphabet = [chr(ord("a") + i) for i in range(n)]
    if len(seen) >= n and seen[:n] == alphabet:      # the caption enumerates them in order
        return seen[:n]
    return alphabet


def _content_tokens(texts) -> set[str]:
    """Lowercased word tokens with no numbers and no one-letter words — what a title is made of.

    De-hyphenated first: line-broken words survive `_norm_ws` as "par- ticipants", and a token
    ending in "-" matches nothing it should.
    """
    out: set[str] = set()
    for text in texts:
        for raw in re.split(r"[\s/]+", str(text).replace("- ", "").replace("-", " ")):
            token = raw.strip(".,;:()[]{}\"'!?%°").lower()
            if len(token) >= 2 and numeric_value(token) is None:
                out.add(token)
    return out


def _caption_panel_texts(caption: str, letters: list[str]) -> dict[str, str]:
    """The caption segmented into per-letter descriptions, or {} — never a guess.

    Two scans over the SAME normalized text, mirroring `caption_panels`' two kinds of evidence:
    marks first (an enumeration is punctuated and prose is not), then bare letters only when the
    punctuated scan cannot place the whole run (bold enumerations print no punctuation at all).
    Both are greedy on the NEXT EXPECTED letter, so an English article "a" sitting after the real
    "(a)" mark can never re-segment the caption. Every description ends at its first sentence
    boundary: the trailing prose of a caption ("Error bars are standard errors…") describes the
    figure, not the last panel, and measured on a real corpus it is exactly what filled the last
    letter's description with vocabulary every panel matches. All letters placed in order, or {}:
    a segmentation that cannot account for the whole run must not manufacture evidence.
    """
    text = _norm_ws(caption)
    # a LOCAL case-insensitive scan: `CAPTION_PANEL_RE` is lowercase-only and its acceptance
    # behaviour is pinned, but "(A) … (B)" is the commonest enumeration in print, and this
    # function only ever looks for the letters `caption_panels` (or the ordinal fallback)
    # already committed to — the next-expected discipline keeps stray capitals inert
    pattern = re.compile(CAPTION_PANEL_RE.pattern, re.IGNORECASE)
    for punctuated_only in (True, False):
        spans: dict[str, str] = {}
        expected = list(letters)
        marks: list[tuple[str, int, int]] = []
        for m in pattern.finditer(text):
            if not expected:
                break
            if punctuated_only and m.group(0).rstrip()[-1:] not in ")]:.,;":
                continue
            if m.group(1).lower() == expected[0]:
                marks.append((expected.pop(0), m.start(), m.end()))
        if expected or not marks:
            continue
        for i, (letter, _, begin) in enumerate(marks):
            end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
            body = text[begin:end]
            stop = re.search(r"\.\s", body)
            spans[letter] = body[:stop.end()] if stop else body
        return spans
    return {}


def _verify_panel_letters(panels: list[dict], caption: str, letters: list[str],
                          words: list) -> tuple[list[str] | None, str]:
    """`(rebound_letters, "")`, `(None, dispute_note)` or `(None, "")` — fix F's whole ruling.

    Each panel's own printed words are matched against the caption's per-letter descriptions on
    DOUBLY discriminative tokens: a token votes only when it belongs to exactly one panel AND
    exactly one description. One-sided discriminativeness is not enough — measured on a real
    corpus, a shared axis title ("target direction … deg") is unique to the LAST description
    (which inherits the caption's trailing prose) and voted every panel of three correct figures
    toward the same letter. A panel votes for the letter it beats every other letter on; votes
    that agree with the standing letters are silence; a full bijection of disagreeing votes is a
    re-bind (a true swap produces exactly that 2-cycle); anything less is a dispute — EXCEPT the
    uniform previous-letter chain, which is what "Movement time (a) and endpoint error (b)"
    trailing-letter captions produce from segmentation alone and is silenced as an artifact.
    """
    if len(panels) < 2 or len(letters) != len(panels):
        return None, ""
    descriptions = _caption_panel_texts(caption, letters)
    if set(descriptions) != set(letters):
        return None, ""
    panel_tokens = [_content_tokens(_words_in(words, p["rect"])) for p in panels]
    description_tokens = {letter: _content_tokens([descriptions[letter]]) for letter in letters}
    votes: dict[int, tuple[str, list[str]]] = {}
    for i, own in enumerate(panel_tokens):
        mine = {t for t in own if sum(t in other for other in panel_tokens) == 1}
        scores = {
            letter: sorted(t for t in mine & description_tokens[letter]
                           if sum(t in description_tokens[o] for o in letters) == 1)
            for letter in letters}
        best = max(scores, key=lambda letter: len(scores[letter]))
        if scores[best] and all(len(scores[o]) < len(scores[best])
                                for o in letters if o != best):
            votes[i] = (best, scores[best])
    if not votes or all(best == letters[i] for i, (best, _) in votes.items()):
        return None, ""
    proposed = [votes[i][0] if i in votes else letters[i] for i in range(len(panels))]
    evidence = "; ".join(
        f"the rect lettered {letters[i]!r} prints {', '.join(repr(t) for t in toks[:4])}, "
        f"which the caption describes under {best!r}"
        for i, (best, toks) in sorted(votes.items()) if best != letters[i])
    if sorted(proposed) == sorted(letters):
        return proposed, evidence
    shift = {(i, best) for i, (best, _) in votes.items() if best != letters[i]}
    if all(i > 0 and best == letters[i - 1] for i, best in shift):
        # every disagreeing vote points one letter BACK — the shape trailing-letter captions
        # ("Movement time (a) and endpoint error (b)") give the segmentation on a CORRECT
        # figure, because each description's words sit before its mark, not after
        return None, ""
    return None, evidence


# ------------------------------------------- ticket 2: what the caption itself states
#: the marker vocabulary, canonical HERE: the caption parser and the digitiser must speak one
#: language or a key parsed from print would fail to match the very words a reader used
#: (`digitize.digitizer` aliases these under its old names; a pinned test asserts identity).
MARKER_FILL_WORDS = {"open": "open", "unfilled": "open", "hollow": "open", "white": "open",
                     "empty": "open", "outline": "open",
                     "filled": "filled", "solid": "filled", "black": "filled",
                     "closed": "filled", "dark": "filled"}
MARKER_SHAPE_WORDS = ("square", "circle", "triangle", "diamond", "star", "cross", "bar")
#: words that identify a series by colour or line style without being a fill claim. A colour key
#: feeds the reader's prompt and the group binding but NEVER the marker-mismatch machinery —
#: mapping colour names to detected-marker hex is guesswork this record refuses to do.
MARKER_COLOUR_WORDS = frozenset({"black", "grey", "gray", "white", "red", "blue", "green",
                                 "orange", "purple", "dark", "light"})
MARKER_LINE_WORDS = frozenset({"line", "lines", "dashed", "dotted", "curve", "curves",
                               "trace", "traces"})
#: neutral carriers a descriptor may include beside a real marker word ("filled symbols")
_MARKER_CARRIER_WORDS = frozenset({"symbol", "symbols", "marker", "markers", "point", "points",
                                   "dot", "dots"})

#: a sentence is ABOUT the rendering of spread only when it says so: in this literature
#: "standard deviation" routinely names the OUTCOME ("the SD of heading direction was
#: analysed"), and crediting that would invent an error-bar type the figure never states.
_DISPERSION_ANCHOR_RE = re.compile(
    r"error\s+bars?|whiskers?|\bshaded\b|\bshading\b|"
    r"(?:vertical|horizontal)\s+(?:error\s+)?(?:bars?|lines?)|"
    r"\bbars?\s+(?:denote|indicate|represent|show|are)\b", re.IGNORECASE)

#: multiplier forms: "± 2 SE" draws a bar TWICE the statistic — crediting the bare type would
#: corrupt every digitised dispersion by that factor, so any such caption is refused whole
_DISPERSION_MULTIPLIER_RE = re.compile(
    r"(?:±\s*|\b(?:denote|indicate|represent|show|are)\s+)\d+(?:\.\d+)?\s*(?:×|x\s)?\s*"
    r"(?=S\.?E|S\.?D|SEM\b|SD\b|SE\b|s\.e|s\.d|standard\s)", re.IGNORECASE)

#: `(DispersionType value, pattern)` in scan order. Spelled-out phrases are case-insensitive
#: with their known impostors excluded by lookahead (the SEE of a regression, the historical
#: "standard deviation of the mean" = SEM); abbreviations count only dotted or uppercase —
#: bare lowercase "se"/"sd" are words in several languages and are never credited.
_CAPTION_DISPERSION_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("SE", re.compile(r"\bstandard\s+errors?\s+of\s+the\s+means?\b", re.IGNORECASE)),
    ("SE", re.compile(r"\bstandard\s+errors?\b(?!\s+of\s+the\s+(?:estimate|mean))",
                      re.IGNORECASE)),
    ("SE", re.compile(r"\bS\.E\.M\b\.?|\bs\.e\.m\.")),
    ("SE", re.compile(r"\bSEM\b")),
    ("SE", re.compile(r"\bSE\b|\bs\.e\.")),
    ("SD", re.compile(r"\bstandard\s+deviations?\b(?!\s+of\s+the\s+mean)", re.IGNORECASE)),
    ("SD", re.compile(r"\bS\.D\b\.?|\bs\.d\.")),
    ("SD", re.compile(r"\bSD\b|\bstd\.?\s*dev\w*")),
    ("CI95", re.compile(r"\b95\s*%\s*(?:confidence\s+intervals?|CI\b)", re.IGNORECASE)),
    ("CI90", re.compile(r"\b90\s*%\s*(?:confidence\s+intervals?|CI\b)", re.IGNORECASE)),
    ("IQR", re.compile(r"\binterquartile\b|\bIQR\b")),
    ("RANGE", re.compile(r"\bmin\s*[-–—]\s*max\b", re.IGNORECASE)),
)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def caption_dispersion(caption: str) -> dict[str, str]:
    """`{"type": "SE", "quote": "<the matched sentence>"}` or `{}` — never a guess.

    Conservative by construction, exactly like `_caption_panel_texts`: explicit statements only,
    refusal on any ambiguity. A match counts only when its sentence is ANCHORED to the rendering
    of spread (`_DISPERSION_ANCHOR_RE`) or the match itself sits in a `mean ± TYPE` form; two
    different anchored types with no scope separating them refuse the whole caption (the
    per-letter helper below may still resolve each); a multiplier form ("± 2 SE") refuses it
    outright. Non-English captions match nothing, which IS the conservative behaviour.
    """
    text = _norm_ws(caption).replace("+/-", "±").replace("+/−", "±")
    if not text or _DISPERSION_MULTIPLIER_RE.search(text):
        return {}
    found: dict[str, str] = {}
    for sentence in _sentences(text):
        anchored_sentence = bool(_DISPERSION_ANCHOR_RE.search(sentence))
        for kind, pattern in _CAPTION_DISPERSION_PATTERNS:
            for m in pattern.finditer(sentence):
                prefix = sentence[max(0, m.start() - 8):m.start()]
                if not anchored_sentence and "±" not in prefix:
                    continue
                found.setdefault(kind, sentence)
    if len(found) != 1:
        return {}
    kind, quote = next(iter(found.items()))
    return {"type": kind, "quote": quote}


def caption_panel_dispersions(caption: str, letters: list[str]) -> dict[str, dict[str, str]]:
    """Per-letter dispersion statements, through `_caption_panel_texts`' own segmentation.

    `{}` whenever segmentation cannot account for the whole caption: a statement whose scope
    cannot be placed must not be scoped by guesswork.
    """
    out: dict[str, dict[str, str]] = {}
    for letter, description in _caption_panel_texts(caption, letters).items():
        entry = caption_dispersion(description)
        if entry:
            out[letter] = entry
    return out


_SERIES_KEY_VERB_RE = re.compile(r"\b(represents?|denotes?|indicates?|shows?|are|is)\b",
                                 re.IGNORECASE)


def _marker_descriptor(words: list[str]) -> bool:
    """Is this 1–3 word phrase a marker descriptor — every word vocabulary, at least one real?

    Carriers ("symbols") may ride along; a phrase of only carriers describes nothing.
    """
    if not words or len(words) > 3:
        return False
    if any(ch in word for word in words for ch in "()"):
        # a phrase crossing a paren boundary ("circles)") is shape-B territory — a key minted
        # from it would carry half a parenthetical as its descriptor
        return False
    cleaned = [w.lower().strip(",;:.\"'") for w in words]
    real = 0
    for word in cleaned:
        if _is_shape_word(word) or word in MARKER_FILL_WORDS or word in MARKER_COLOUR_WORDS \
                or word in MARKER_LINE_WORDS:
            real += 1
        elif word not in _MARKER_CARRIER_WORDS:
            return False
    if cleaned == [cleaned[0]] and cleaned[0] in MARKER_COLOUR_WORDS:
        return False                    # a lone colour word keys nothing ("the green is …")
    return real >= 1


def _is_shape_word(word: str) -> bool:
    """Exactly a shape word (plural allowed) — never a prefix: "crosshair" is not a cross,
    "stars" of significance are stars but "start" is not, and "barely" is not a bar."""
    return any(re.fullmatch(rf"{shape}(?:s|es)?", word) for shape in MARKER_SHAPE_WORDS)


def _key_of(descriptor: str, series_text: str, quote: str) -> dict[str, str] | None:
    """One parsed key, or None when the pair fails the refusal rules."""
    series_text = series_text.strip(" ,;:.").strip()
    if not 2 <= len(series_text) <= 60:
        return None
    # a statement about SPREAD is never a series key: "Error bars represent the standard error
    # of the mean" once minted `bars = 'the standard error of the mean'` — an instruction-grade
    # false fact in every reader prompt of that figure. Same for significance annotations.
    if any(pattern.search(series_text) for _, pattern in _CAPTION_DISPERSION_PATTERNS) \
            or "significan" in series_text.lower():
        return None
    lowered = series_text.lower().split()
    if any(w in MARKER_FILL_WORDS or w in MARKER_COLOUR_WORDS or w in MARKER_LINE_WORDS
           or _is_shape_word(w) for w in lowered):
        return None                     # "circles represent filled symbols" keys nothing
    words = descriptor.lower().split()
    return {"descriptor": descriptor.strip(), "series_text": series_text, "quote": quote,
            "line_style": any(w.strip(",;:") in MARKER_LINE_WORDS for w in words)}


def caption_series_keys(caption: str) -> list[dict[str, str]]:
    """The caption's own marker→series key, or `[]` — explicit statements only.

    Two sentence shapes, measured on real captions: key-first ("Circles represent adaptation
    with the right hand") and parenthetical ("dominant arm performance (open circles)"). Two
    keys whose descriptors no longer distinguish anything (identical) are BOTH dropped;
    everything unrecognized parses to nothing, by construction.
    """
    text = _norm_ws(caption)
    out: list[dict[str, str]] = []

    def trailing_descriptor(prefix: str) -> list[str]:
        words = prefix.split()[-3:]
        while words and not _marker_descriptor(words):
            dropped = words[0]
            words = words[1:]           # "…right hand and squares" keys on "squares" alone
            if words and dropped.lower().strip(",;:.") == "error":
                return []               # "error bars" is a spread phrase, never a series key
        return words

    for sentence in _sentences(text):
        # shape A: descriptor VERB series-text, possibly chained with "and"/"while"/"whereas"
        spans: list[tuple[Any, list[str], int]] = []
        for m in _SERIES_KEY_VERB_RE.finditer(sentence):
            words = trailing_descriptor(sentence[:m.start()])
            if not words:
                continue
            at = sentence[:m.start()].rfind(" ".join(words))
            spans.append((m, words, at if at >= 0 else m.start()))
        for i, (m, descriptor_words, _) in enumerate(spans):
            # one key's series text ends where the NEXT key's descriptor begins, not at its verb
            stop = spans[i + 1][2] if i + 1 < len(spans) else len(sentence)
            tail = re.split(r"[,;.]", sentence[m.end():stop])[0]
            tail = re.sub(r"\s+(?:and|while|whereas)\s*$", "", tail)
            key = _key_of(" ".join(descriptor_words), tail, sentence)
            if key:
                out.append(key)
        # shape B: series-text (descriptor)
        for m in re.finditer(r"([^(),;.]{2,80})\(([^()]{2,40})\)", sentence):
            inner = m.group(2).split()
            if not _marker_descriptor(inner):
                continue
            lead = m.group(1).strip()
            lead = re.split(r"\b(?:for|and|with|of)\s+(?=\S+\s+\S+)", lead)[-1]
            key = _key_of(" ".join(inner), lead, sentence)
            if key:
                out.append(key)
    counted: dict[str, int] = {}
    for key in out:
        counted[key["descriptor"].lower()] = counted.get(key["descriptor"].lower(), 0) + 1
    return [key for key in out if counted[key["descriptor"].lower()] == 1]


def _caption_semantics(caption: str, letters: list[str]) -> dict[str, Any]:
    """Everything ticket 2 reads off one caption, with the figure/letter scoping rule applied.

    Figure-level dispersion is withheld when any letter-scoped statement names a DIFFERENT type,
    and when the figure-level match's own sentence lives inside a letter's description — a
    statement about panel b must never speak for its siblings.
    """
    per_panel = caption_panel_dispersions(caption, letters)
    figure = caption_dispersion(caption)
    if figure and per_panel:
        descriptions = _caption_panel_texts(caption, letters)
        if any(entry["type"] != figure["type"] for entry in per_panel.values()) \
                or any(figure["quote"] in desc for desc in descriptions.values()):
            figure = {}
    return {"type": str(figure.get("type") or ""), "quote": str(figure.get("quote") or ""),
            "panels": per_panel, "series_keys": caption_series_keys(caption)}


def _panel_left(rect: pymupdf.Rect, siblings: list[pymupdf.Rect],
                cap_x0: float | None) -> float:
    """How far left this panel may reach — to a landmark, and never over its neighbour.

    Growing EVERY panel's left edge to the caption column is what made side-by-side layouts
    collapse into their own union: the right-hand panel's rect then swallowed the left-hand one
    whole, `_overlapping` fired, and the "per-panel" assertion was evaluated on a rect holding
    two ladders — which is exactly the global count the decision withdrew, reached by another
    road. Measured with the first cut: a two-column figure gave 1 panel with 10 numeric words,
    a 2x2 grid gave 1. The landmark for an inner column is the nearest neighbour's right edge
    (the gap between two panels is where the right one's tick labels live); only the leftmost
    column has the caption's own `x0` as its landmark.
    """
    left_of = [s for s in siblings if s.x0 < rect.x0 and s.y1 > rect.y0 and s.y0 < rect.y1]
    if left_of:
        # x-RANGE, never strict non-overlap: one point of x-overlap between two side-by-side
        # clusters — an error-bar cap, a tick, a shared frame line — used to make the neighbour
        # invisible, and the right-hand panel then grew to the caption column and swallowed the
        # left-hand one whole. Measured: with a 1 pt overlap the right panel's rect became
        # (48..486) and the assertion certified it on the LEFT panel's ladder, while
        # `_overlapping` (0.6 % of the smaller area) did not fire either. A neighbour is a
        # cluster that starts to my left; the landmark is its right edge, clamped inside mine.
        return min(rect.x0 - LANDMARK_GAP, max(s.x1 for s in left_of))
    if cap_x0 is not None and rect.x0 - cap_x0 >= MIN_CAPTION_GROWTH:
        return cap_x0
    return rect.x0 - PANEL_PAD


def _panel_right(rect: pymupdf.Rect, siblings: list[pymupdf.Rect],
                 obstacles: list[pymupdf.Rect], cap_x1: float | None,
                 region_x1: float) -> float:
    """The mirror of `_panel_left`, spent only on a panel whose ladder came back EMPTY.

    A ladder printed to the RIGHT of its plot area — a secondary axis, a right-labelled member of
    a pair (`digitizer.py` records that Cressman 2010's Fig. 3b itself "carries a left y-axis in
    degrees and a right one in per cent") — is 8 pt outside a 6 pt pad, and the panel is then
    refused with zero read-outs although it is perfectly readable. Never a wrong number; a lost
    cell, of the same class as the ASCII-only minus.

    It is bounded by EVERYTHING ELSE ON THE PAGE, never by the caption. Bounded by the caption
    alone it was the first rule in this area that could reach outside the figure it describes,
    and the thing it reaches for is a ladder: on a page whose caption is wider than its figure —
    Heuer's own fig03 and fig05 overhang by 26 pt, and a full-width caption over a column figure
    is the ordinary two-column case — the retry crossed into the next column and certified the
    panel on a FOREIGN ladder (measured: a neighbouring figure's `500,400,300,200,100`, and a
    numbered list's `1,2,3,4,5`, both accepted as "panel b's own axis" on a rect spanning the
    whole page). That is worse than the lost cell N3 fixed: it is a wrong-ladder certification,
    the exact hazard the per-panel assertion exists for, and it defeats `panel_uncalibrated` at
    the same time.

    So the edge stops at the nearest thing to its right that belongs to something else — another
    figure's cluster, another caption, a body-text block outside this figure's own column — and
    the caption's `x1` is only the fallback when the page carries nothing there at all. It is
    still spent only on a panel whose ladder came back EMPTY, so it cannot make a calibrated
    panel worse, and `_panels_of` refuses the ladder it finds if it sits further away than a tick
    label ever sits (`RIGHT_LADDER_REACH_PT`, or the panel's own width where that is smaller).
    """
    def blocks(rects: list[pymupdf.Rect]) -> list[float]:
        return [r.x0 - LANDMARK_GAP for r in rects
                if r.x0 >= rect.x1 and r.y1 > rect.y0 and r.y0 < rect.y1]

    edge = region_x1 if cap_x1 is None else max(region_x1, cap_x1)
    bounds = [edge] + blocks([s for s in siblings if s.x1 > rect.x1]) + blocks(list(obstacles))
    return max(rect.x1 + PANEL_PAD, min(bounds))


def _panel_bottom(rect: pymupdf.Rect, siblings: list[pymupdf.Rect],
                  cap_y0: float | None) -> float:
    """How far down this panel may reach — to the caption block, never into the panel below.

    The caption block is a landmark on the BOTTOM as much as on the left, and the decision's own
    argument ("a real thing on the page; a pad in points is not") was never carried to this edge.
    It matters on precisely the figures C1 exists to repair: Heuer 2008's eight x tick labels sit
    0.1-1.4 pt below the crop and its x-axis title 8-13 pt below, with the caption a further
    20-25 pt down — and the target of two of those cells is a POSITION ON THAT X AXIS.
    The region rect is deliberately not moved: that would move Cressman's crops and destroy the
    byte-identity that proves this rule cannot regress the paper supplying two pooled cells.
    """
    if cap_y0 is not None and cap_y0 > rect.y1:
        bottom = min(cap_y0 - LANDMARK_GAP, rect.y1 + PANEL_GROW_DOWN)
    else:
        bottom = rect.y1 + PANEL_PAD
    below = [s.y0 for s in siblings if s.y0 >= rect.y1 and s.x1 > rect.x0 and s.x0 < rect.x1]
    if below:
        bottom = min(bottom, min(below) - LANDMARK_GAP)
    return max(bottom, rect.y1)


def _panels_of(rects: list[pymupdf.Rect], cap: pymupdf.Rect | None, words: list,
               obstacles: list[pymupdf.Rect] = ()) -> list[dict]:
    """Per-panel sub-rects for one caption's graphics, in reading order, each with its own ladder.

    Growth is ASYMMETRIC and to LANDMARKS, not by a constant: left to the nearest neighbour's
    right edge or (for the leftmost column) the caption block's own `x0`, down to the caption
    block or the next panel, a 6 pt pad on what is left. Re-measuring Heuer 2008 showed the pad
    each panel needs to reach its own tick ladder is 16/16/16, 17/13/16 and 20/11/15 pt on three
    figures — so no single constant tuned on one page generalises, and the 12 pt an earlier draft
    prescribed is insufficient on all three.
    """
    order = sorted(rects, key=_reading_order_index)
    if _overlapping(order):
        # Panels partition a figure; they do not sit on top of each other. When the CLUSTERS
        # overlap (a raster placement under its own vector overlay, an inset, a shared legend
        # box) the split is an artefact of how the graphics were emitted, not a panel structure,
        # and splitting on it would hand a reader an inset instead of the panel it asked for.
        # The question is asked of the clusters the page drew, never of the grown rects: growth
        # is our doing, and deciding "these are not panels" on rects we grew ourselves is how
        # every side-by-side figure collapsed into its union.
        whole = order[0]
        for rect in order[1:]:
            whole = _union(whole, rect)
        return _panels_of([whole], cap, words, obstacles)
    cap_x0 = float(cap.x0) if cap is not None else None
    cap_x1 = float(cap.x1) if cap is not None else None
    cap_y0 = float(cap.y0) if cap is not None else None
    region_x1 = max(r.x1 for r in order)
    out = []
    for rect in order:
        left = _panel_left(rect, order, cap_x0)
        bottom = _panel_bottom(rect, order, cap_y0)
        sub = pymupdf.Rect(left, rect.y0 - PANEL_PAD, rect.x1 + PANEL_PAD, bottom)
        ladder = _tick_ladder(words, sub)
        if not ladder:
            # the left growth found no ladder: look to the right, at the same kind of landmark
            wide = pymupdf.Rect(left, sub.y0,
                                _panel_right(rect, order, list(obstacles), cap_x1, region_x1),
                                bottom)
            found = _ladder_marks(words, wide)
            # …and a ladder that sits further from this panel than a tick label ever sits — or
            # than the panel is itself wide, on a narrow one — is not this panel's ladder,
            # whatever the page did or did not put between them
            reach = sub.x1 + min(sub.x1 - sub.x0, RIGHT_LADDER_REACH_PT)
            if found and max((m[1] + m[2]) / 2.0 for m in found) <= reach:
                sub, ladder = wide, [m[0] for m in found]
        out.append(dict(rect=sub, n_numeric=len(_numeric_words(words, sub)),
                        n_ladder=len(ladder),
                        calibrated=len(ladder) >= MIN_PANEL_CALIBRATED))
    return out


def _overlapping(rects: list[pymupdf.Rect]) -> bool:
    """True when any two of the page's own graphic clusters share a fifth of the smaller's area."""
    for i, ra in enumerate(rects):
        for rb in rects[i + 1:]:
            w = min(ra.x1, rb.x1) - max(ra.x0, rb.x0)
            h = min(ra.y1, rb.y1) - max(ra.y0, rb.y0)
            if w <= 0 or h <= 0:
                continue
            smaller = min(ra.width * ra.height, rb.width * rb.height)
            if smaller > 0 and (w * h) / smaller > 0.2:
                return True
    return False


def _whole_region_panel(rect: pymupdf.Rect, words: list) -> list[dict]:
    """A region nobody split into panels is one panel: itself.

    Used for the two GUESSED region kinds — a `caption_only` rect proposed above a caption that
    received no graphics (`confidence 0.3`) and a `loose` graphic with no caption at all (`0.4`).
    They do not get the one-panel exemption a real figure gets: a rect the ingester only guessed
    at must not be recorded as calibrated because nobody split it. Only a ladder, or the absence
    of any number to build one from, certifies it.
    """
    ladder = _tick_ladder(words, rect)
    numeric = _numeric_words(words, rect)
    return [dict(rect=pymupdf.Rect(rect), letter="a", n_numeric=len(numeric),
                 n_ladder=len(ladder),
                 calibrated=len(ladder) >= MIN_PANEL_CALIBRATED or not numeric)]


def _enforce_contiguity(candidates: list[dict], choice: dict, ranked: dict) -> None:
    """M2: the graphics merged under one caption must be CONTIGUOUS in reading order.

    Without a distance cap, one caption whose score dips (a "Fig. 4 (cont.)" header, a body
    sentence that outscores nothing) can otherwise reach past a neighbouring figure and pull
    two unrelated sets of axes into one region — the union would then carry two different
    ladders with no way to tell them apart. Interleaving is the signal: if another caption's
    graphic sits BETWEEN two of mine in reading order, mine are two figures, not two panels.
    The stragglers fall back to their own next-best caption, or become loose.
    """
    banned: dict[int, set[int]] = {}
    # Bounded by the number of (graphic, caption) pairs there ARE to ban, not by the number of
    # graphics: one pass bans a pair per broken caption, and a single graphic can need as many
    # bans as it has plausible captions. With more captions than candidates the old bound could
    # exit still non-contiguous, silently.
    for _ in range(sum(len(ranked.get(id(c)) or ()) for c in candidates) + 1):
        order = sorted((c for c in candidates if choice[id(c)][0] is not None),
                       key=lambda c: _reading_order_index(c["rect"]))
        by_cap: dict[int, list[int]] = {}
        for i, c in enumerate(order):
            by_cap.setdefault(id(choice[id(c)][0]), []).append(i)
        broken = [(cap_id, pos) for cap_id, pos in by_cap.items()
                  if len(pos) > 1 and pos[-1] - pos[0] + 1 != len(pos)]
        if not broken:
            return
        for cap_id, pos in broken:
            run_end = 0                          # keep the first contiguous run, re-home the rest
            while run_end + 1 < len(pos) and pos[run_end + 1] == pos[run_end] + 1:
                run_end += 1
            for i in pos[run_end + 1:]:
                c = order[i]
                out = banned.setdefault(id(c), set())
                out.add(cap_id)
                alt = [t for t in ranked[id(c)] if id(t[0]) not in out]
                choice[id(c)] = alt[0][:2] if alt else (None, None)


def _reading_order_index(rect: pymupdf.Rect) -> tuple[float, float]:
    return (round(rect.y0 / 10.0), rect.x0)


def _figure_regions(page: pymupdf.Page, page_no: int) -> list[dict]:
    """Deterministic figure-region proposals. Candidate graphics = raster image placements + clusters of vector
    drawings; each candidate is attached to the best-scoring nearby caption (real captions beat body sentences like
    'Figure 2 illustrates ...'); candidates sharing a caption are merged (multi-panel figures)."""
    W, H = page.rect.width, page.rect.height
    img_rects: list[pymupdf.Rect] = []
    native: dict[tuple[int, int], tuple[int, int]] = {}
    #: every image placement, unfiltered — what the caption_only MEASUREMENT counts. The detector
    #: below excludes full-page placements (a scanned page is not a figure candidate), but a
    #: caption sitting on a full-page scan still has readable ink above it, and measuring with the
    #: filtered list would refuse every figure of a scanned journal as "provably text".
    #: Hairline exclusions live at the measurement site: a running-head rule (page-wide, flat) is
    #: page furniture, and one of them under a caption_only rect must not dodge the refusal.
    raw_img_rects: list[pymupdf.Rect] = []
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            raw_img_rects.append(pymupdf.Rect(r))
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
                # NO distance cap below the graphic. A 260 pt cap threw away the top panel of
                # every tall multi-panel figure (Heuer 2008 Fig. 2's panel a sits 296.2 pt above
                # its caption, and Fig. 4 and Fig. 6 the same): the cluster was assigned no
                # caption, became "loose", and was dropped, so the crop the map named "panel a"
                # did not contain panel a. What keeps a far caption honest is not a constant, it
                # is the two guards below: a graphic takes the NEAREST caption of the best band
                # (so a caption of its own always wins), and a merged region's graphics must be
                # contiguous in reading order (so two figures on one page cannot merge).
                if -0.3 * r.height <= below:
                    dist, rel = max(below, 0), "below"
                elif -0.3 * cap_rect.height <= above <= 60:
                    dist, rel = max(above, 0) + 30, "above"
                elif vert > 0 and cap_rect.y0 > r.y0 + 0.4 * r.height:
                    dist, rel = 20, "inside"        # caption inside the lower part of a wide figure's span
            elif vert > 0.5 * min(cap_rect.height, r.height):
                # side (margin) caption: the caption and its graphic share rows, so the SMALLER
                # box must spend at least half its height beside the other. Measured against the
                # caption alone this assumed captions are shorter than panels; a margin caption is
                # routinely taller than the panel it names, and one real page's panels overlapped
                # their caption by 62pt against the 76 the caption's own height demanded — so the
                # caption matched nothing, the panels became loose, the area gate dropped them,
                # and the paid readers were handed the empty column above the caption.
                gap = max(r.x0 - cap_rect.x1, cap_rect.x0 - r.x1)
                if 0 <= gap <= 45:                  # vertically aligned, nearly touching columns
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
        _enforce_contiguity(candidates, choice, ranked)
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
    all_words = page.get_text("words")
    text_blocks = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")
                   if b[6] == 0 and _norm_ws(b[4])]
    # page furniture and caption prose are not figure text, and letting them answer "does this
    # region carry a text layer / a ladder?" is how a page number came within 5.6 pt of buying
    # zero read-outs on Bock's two pooled cells
    furniture = _page_furniture(page, [c["rect"] for c in candidates])
    prose = furniture + [cap[0] for cap in caps]
    words = [w for w in all_words if not _in_rects(w, prose)]
    regions = []
    for g in groups.values():
        union = g["rects"][0]
        for r in g["rects"][1:]:
            union = _union(union, r)
        cap_rect, cap_txt, label, score = g["cap"]
        kind = "raster" if g["kinds"] == {"raster"} else "vector" if g["kinds"] == {"vector"} else "mixed"
        npx = native.get((round(union.x0), round(union.y0))) if (kind == "raster" and g["n_img"] == 1) else None
        # growth is decided on the caption column alone, NOT on text found in the ungrown union —
        # that is the very rect C1 proved does not contain the ladder. A clean vector plot whose
        # only text is its axis labels then got no growth, was labelled `text_layer: none`, and
        # was swept into the raster exemption with an empty crop. Measured on all 14 real
        # figures: `union.x0 - caption.x0` is <= 0.0012 pt on every Cressman and Bock figure, so
        # dropping the gate is a no-op there and the byte-identity proof is untouched.
        grown, growth = union, 0.0
        if union.x0 - cap_rect.x0 >= MIN_CAPTION_GROWTH:
            # grow to a LANDMARK — the caption block's own column — not by a constant
            grown = pymupdf.Rect(cap_rect.x0, union.y0, union.x1, union.y1)
            growth = float(union.x0 - cap_rect.x0)
        panels = _panels_of(g["rects"], cap_rect, words,
                            _other_page_content(candidates, text_blocks, caps, g, union))
        # every question about this region's text is asked of what the READER IS HANDED — the
        # grown region and the panel rects — never of the raw cluster union, which on Heuer's
        # fig02/04/06 holds none of the tick labels at all (they sit 46-51 pt to its left).
        rects = [grown] + [p["rect"] for p in panels]
        has_text = any(_words_in(words, r) for r in rects)
        n_region_ladder = max(len(_tick_ladder(words, r)) for r in rects)
        n_region_numeric = max(len(_numeric_words(words, r)) for r in rects)
        if n_region_ladder < MIN_PANEL_NUMERIC or len(panels) <= 1:
            # The exemption, and it is about a LADDER rather than about numerals. The premise the
            # per-panel assertion needs is "some panel here owns an axis"; where the figure prints
            # no axis at all there is nothing for a panel to own a share of, and a scanned figure
            # is readable while its words are not (Bock 2005's two rasters supply two pooled
            # cells). Keying it on "any numeric word" re-created exactly the fragility F6 was
            # raised for: a single stray numeral on a two-panel raster — a scale bar's "10", an
            # inset label, whatever the OCR layer happens to carry — refused the whole figure and
            # bought zero read-outs. A ONE-panel figure is exempt for its own reason: it cannot be
            # reading off its neighbour's ladder, which is the whole hazard here.
            for panel in panels:
                panel["calibrated"] = True
        enumerated = caption_panels(cap_txt, bold=_bold_letters(page, cap_rect))
        assigned = _panel_letters(enumerated, len(panels))
        for panel, letter in zip(panels, assigned):
            panel["letter"] = letter
        # fix F: the ordinal assignment above is a guess about reading order, checked here
        # against the only two witnesses the page itself offers — each panel's own printed words
        # and the caption's per-letter descriptions. An unambiguous permutation is re-bound
        # BEFORE `_render_panels` mints ids and crops, so `fig02d` is born meaning the d panel;
        # anything short of a bijection is recorded as a dispute for the read side to honour.
        labels_disputed, label_note = False, ""
        rebound, note = _verify_panel_letters(panels, cap_txt, assigned, words)
        if rebound is not None:
            for panel, letter in zip(panels, rebound):
                panel["letter"] = letter
            label_note = f"re-bound: {note}"
        elif note:
            labels_disputed, label_note = True, note
        regions.append(dict(panel_labels_disputed=labels_disputed, panel_label_note=label_note,
                            bbox=grown, caption=cap_txt, label=label, kind=kind, n_images=g["n_img"],
                            n_drawings=g["n_draw"], native_px=npx, panels=panels,
                            text_layer=TEXT_LAYER_PRESENT if has_text else TEXT_LAYER_NONE,
                            n_region_ladder=n_region_ladder,
                            n_region_numeric=n_region_numeric, caption_panels=enumerated,
                            caption_growth_pt=round(growth, 4),
                            confidence=min(0.95, 0.55 + 0.4 * score)))
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
            # counted, never hardcoded: these two numbers are the record's only statement of
            # whether anything readable is under this rect, and writing zeros unconditionally was
            # a claim, not a measurement. Downstream, a region that MEASURES zero of both is
            # refused a paid read (it is provably text); one that carries primitives the caption
            # matching missed is still readable.
            n_img = sum(1 for ir in raw_img_rects if _overlaps(r, ir))
            n_draw = sum(1 for dr in draw_rects if _overlaps(r, dr)
                         and not (dr.height < 3 and dr.width > 0.5 * W))
            regions.append(dict(bbox=r, caption=cap_txt, label=label, kind="caption_only",
                                n_images=n_img, primitives_measured=True,
                                n_drawings=n_draw, native_px=None, confidence=0.3,
                                panels=_whole_region_panel(r, words), caption_growth_pt=0.0,
                                n_region_ladder=len(_tick_ladder(words, r)),
                                n_region_numeric=len(_numeric_words(words, r)),
                                caption_panels=caption_panels(cap_txt,
                                                              bold=_bold_letters(page, cap_rect)),
                                text_layer=TEXT_LAYER_PRESENT if _words_in(words, r) else TEXT_LAYER_NONE))
    for c in loose:   # graphics without any caption nearby: keep if large (caption may be on the next page)
        r = c["rect"]
        if r.width * r.height <= 0.08 * W * H:
            continue
        if page_no == 1 and r.y1 < 0.2 * H:      # journal banner / logo rules on the title page
            continue
        if c["kind"] == "vector" and c["n_draw"] / (r.width * r.height / 1e4) < 1.0:
            continue                             # a few page-wide rules and boxes, not a chart (sparse line art)
        regions.append(dict(bbox=r, caption="", label="", kind=c["kind"], n_images=int(c["kind"] == "raster"),
                                n_drawings=c["n_draw"], native_px=native.get((round(r.x0), round(r.y0))),
                                confidence=0.4, panels=_whole_region_panel(r, words), caption_growth_pt=0.0,
                                n_region_ladder=len(_tick_ladder(words, r)),
                                n_region_numeric=len(_numeric_words(words, r)), caption_panels=[],
                                text_layer=TEXT_LAYER_PRESENT if _words_in(words, r) else TEXT_LAYER_NONE))
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


def _render_panels(page: pymupdf.Page, out: Path, fid: str, reg: dict, dpi: float,
                   clip: pymupdf.Rect, crop_rel: str, claude_rel: str,
                   claude_scale: float) -> list[PanelRegion]:
    """One image per panel, so a reader asked for panel b is never handed three panels' ladders.

    A single-panel figure re-uses the figure's own crop byte-for-byte: there is nothing to split,
    and rendering a second, slightly different image of the same thing would change what every
    downstream route sees for no gain.
    """
    proposed = reg.get("panels") or []
    if len(proposed) <= 1:
        letter = (proposed[0]["letter"] if proposed else "a")
        n_numeric = int(proposed[0]["n_numeric"]) if proposed else 0
        n_ladder = int(proposed[0].get("n_ladder", 0)) if proposed else 0
        calibrated = bool(proposed[0]["calibrated"]) if proposed else True
        return [PanelRegion(id=f"{fid}{letter}", letter=letter, bbox=Bbox.from_rect(clip),
                            n_numeric=n_numeric, n_ladder=n_ladder, calibrated=calibrated,
                            crop_png=crop_rel, claude_png=claude_rel, crop_dpi=dpi,
                            claude_scale=claude_scale)]
    out_panels: list[PanelRegion] = []
    for panel in proposed:
        rect = panel["rect"]
        sub = pymupdf.Rect(max(0, rect.x0), max(0, rect.y0),
                           min(page.rect.width, rect.x1), min(page.rect.height, rect.y1))
        pid = f"{fid}{panel['letter']}"
        pix = page.get_pixmap(dpi=dpi, clip=sub, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        rel = f"figures/{pid}.png"
        image.save(out / rel, optimize=True)
        prep = prepare_for_claude(image)
        claude = f"figures/{pid}.claude.png"
        prep.image.save(out / claude, optimize=True)
        out_panels.append(PanelRegion(id=pid, letter=panel["letter"], bbox=Bbox.from_rect(sub),
                                      n_numeric=int(panel["n_numeric"]),
                                      n_ladder=int(panel.get("n_ladder", 0)),
                                      calibrated=bool(panel["calibrated"]), crop_png=rel,
                                      claude_png=claude, crop_dpi=dpi, claude_scale=prep.scale))
    return out_panels


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
                # PyMuPDF's `set_dpi` takes an int: a raster figure whose native density won the
                # `min` handed it a float and the paper died in ingest (`TypeError: in method
                # 'fz_pixmap_xres_set'`, nine-paper run) — round, never pass the ratio raw
                dpi = int(round(max(72, min(fig_dpi, 72 * nat_w / max(clip.width, 1) * RASTER_UPSCALE))))
            else:
                dpi = int(fig_dpi)
            pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
            crop = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            crop_rel = f"figures/{fid}.png"
            crop.save(out / crop_rel, optimize=True)
            prep = prepare_for_claude(crop)
            claude_rel = f"figures/{fid}.claude.png"
            prep.image.save(out / claude_rel, optimize=True)
            panels = _render_panels(page, out, fid, reg, dpi, clip, crop_rel, claude_rel, prep.scale)
            _sem = _caption_semantics(reg["caption"], list(reg.get("caption_panels") or []))
            figures.append(FigureRegion(id=fid, page=n, bbox=Bbox.from_rect(clip), caption=reg["caption"], label=reg["label"],
                                        kind=reg["kind"], n_images=reg["n_images"], n_drawings=reg["n_drawings"],
                                        native_px=reg["native_px"], crop_png=crop_rel, claude_png=claude_rel,
                                        crop_dpi=dpi, claude_scale=prep.scale, confidence=reg["confidence"],
                                        panels=panels, text_layer=reg.get("text_layer", TEXT_LAYER_PRESENT),
                                        n_region_ladder=int(reg.get("n_region_ladder", 0)),
                                        n_region_numeric=int(reg.get("n_region_numeric", 0)),
                                        caption_panels=list(reg.get("caption_panels") or []),
                                        caption_growth_pt=float(reg.get("caption_growth_pt", 0.0)),
                                        # every key set on a region dict must be threaded HERE or
                                        # it silently vanishes: `primitives_measured=True` was
                                        # set on caption_only dicts and dropped at this seam, so
                                        # the digitiser's $0 no-graphics refusal never fired on a
                                        # fresh ingest — a claim its own pinned test could not
                                        # see, because that test builds the record directly
                                        primitives_measured=bool(reg.get("primitives_measured", False)),
                                        panel_labels_disputed=bool(reg.get("panel_labels_disputed", False)),
                                        panel_label_note=str(reg.get("panel_label_note", "")),
                                        # ticket 2: deterministic parses of the caption itself,
                                        # computed at this one seam so every region path (grouped,
                                        # caption_only, loose) gets them from the same words
                                        **{"caption_dispersion": _sem["type"],
                                           "caption_dispersion_quote": _sem["quote"],
                                           "caption_dispersion_panels": _sem["panels"],
                                           "caption_series_keys": _sem["series_keys"]}))
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
