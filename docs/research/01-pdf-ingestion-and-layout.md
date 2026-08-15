# 01 — PDF ingestion & layout analysis for Canopy

*Research brief, 2026-08-14. Scope: how to turn a folder of research-paper PDFs into clean, page-anchored text, located figures/captions with high-resolution crops, structured tables, and OCR'd scans, ready for LLM extraction with provenance. Local checks were run on this Mac (Python 3.13.11, PyMuPDF 1.26.7 / MuPDF 1.26.12, OpenCV 4.13, torch 2.11 with MPS, tesseract 5.5.1, poppler 25.05) against three real papers (Tsay 2021 J Neurophysiol; a 2022 Nature article; a 2023 Exp Brain Res article).*

---

## 0. TL;DR recommendation

| Layer | Primary | Fallback / verifier | Why |
|---|---|---|---|
| Parse + render + text with bboxes | **PyMuPDF** (`pymupdf`, AGPL; 1.26.7 installed, 1.28.2 current) | `pdftotext -layout` (poppler) as a second opinion on reading order | Fast (whole 11-page paper: text+images+drawings+tables in ~1 s), gives spans/lines/blocks with bboxes, native raster extraction, vector-drawing clustering, table finder, page rendering at any DPI, built-in tesseract OCR hook. |
| Reading order / 2-column | Column-aware block sort on PyMuPDF `dict` output (own code, ~50 lines) | `pymupdf4llm` (+ `pymupdf_layout`, GNN layout, CPU-only) **as an optional extra** — `pymupdf-layout` 1.28.2 (2026-08-06) is now dual-licensed AGPL-3.0 / Artifex commercial (1.28.0 was still Polyform-Noncommercial); pin `>=1.28.2` and keep it optional because it is a separate proprietary-model package | Column-sort covers >90% of journal papers; layout model handles the ugly rest (floating boxes, footnotes, 3-col). |
| Figure + caption localisation | Heuristics: caption regex → nearest raster rects (`get_image_info`) ∪ vector clusters (`cluster_drawings`) → union bbox | **DocLayout-YOLO** (`doclayout-yolo`, ~40 MB `.pt`, runs on MPS) `figure`/`figure_caption`/`table`/`table_caption` on 150-dpi page renders; **Docling** (`docling`, layout "heron" RT-DETR) if we want an all-in-one converter | Heuristics are exact for born-digital PDFs; the detector catches figures with no vector/raster signal (e.g. figures made of text glyphs) and disambiguates multi-figure pages. |
| Figure crops | `page.get_pixmap(clip=bbox, dpi=400–600)` for vector or mixed figures; `doc.extract_image(xref)` for pure rasters *when the native raster is ≥ the render resolution* | — | Vector figures must be rendered; native rasters in journal PDFs are often already 1500–2150 px wide (measured), i.e. > 300 dpi at print size. |
| Tables | `page.find_tables()` (strategies `lines_strict` → `lines` → `text`) → pandas | Docling TableFormer (structure model) or Claude vision on a 300-dpi crop with structured-output JSON | Rule-based finder is good for ruled journal tables; ML/VLM for borderless tables. |
| Scans / no text layer | Detect (`len(page.get_text()) < N` and page is one big raster) → **`ocrmypdf --skip-text`** (tesseract) to add a text layer, then run the same pipeline | PyMuPDF `page.get_textpage_ocr(dpi=300)`; Claude native PDF/vision as a last resort | Keeps one code path: after OCR the scan is a normal PDF with a text layer. |
| Whole-paper "second reader" | **Anthropic native PDF input** (`document` block, base64 or Files API) with `citations` enabled → page numbers | — | Claude sees text + a rendered image of every page; cheap with caching/batches; no bboxes though. |
| De-dup | SHA-256 of bytes → DOI/title/first-2-pages text fingerprint → MinHash near-dup | — | Handles re-downloads, preprint vs. VoR, supplement-vs-main. |

Do **not** build the core on Marker, MinerU, olmOCR, Nougat, GROBID or PDFFigures2 (details in §1): they are either heavy (multi-GB VLMs, GPU-oriented), lossy about coordinates, JVM/Scala, or unmaintained. They remain useful as *optional* backends behind a common interface.

---

## 1. Landscape (mid-2026)

| Tool | What it is | Text w/ bboxes | Figures + captions w/ bboxes | Tables | Scans/OCR | Local footprint (Mac M-series) | Notes |
|---|---|---|---|---|---|---|---|
| **PyMuPDF** 1.28.2 (2026-08-06); 1.26.7 installed | MuPDF binding | Yes: `get_text("dict"/"rawdict"/"words"/"blocks")` with span/char bboxes, fonts, sizes | Rasters: `get_image_info(xrefs=True)`, `get_image_rects`; vector: `get_drawings()` + `cluster_drawings()`; captions: your regex on blocks | `find_tables()` (3 strategies) → `to_pandas()`/markdown | `get_textpage_ocr()` via tesseract (needs `TESSDATA_PREFIX`) | ~24 MB wheel (macOS arm64), no torch | AGPL-3 (or commercial). Fastest by an order of magnitude. Prints "Consider using pymupdf_layout" hint. |
| **pymupdf4llm** + **pymupdf_layout** | Markdown/chunk converter with GNN layout model | `to_markdown(page_chunks=True, extract_words=True)` → per-page `text`, `words`, `tables`, `images`, `graphics`, `page_boxes` (layout class + bbox) | `write_images=True, dpi=…` writes figure crops; layout classes incl. captions/headers/footers | Yes (PyMuPDF finder) | Plugin OCR (tesseract/RapidOCR/Paddle) | CPU-only, no torch | pymupdf4llm AGPL; **pymupdf_layout ≥1.28.2 is AGPL-3.0 / commercial** (≤1.28.0 was Polyform Noncommercial — do not pin older) — treat as optional. |
| **pdfplumber** 0.11.x | pdfminer.six wrapper | Yes: chars/words/lines with bboxes | No figure detection (has `.images`, `.rects`, `.curves`) | `extract_tables()` lines/text strategies, tunable | No | Pure Python | 10–50× slower than PyMuPDF; fine for tables/debug (`page.to_image().debug_tablefinder()`). |
| **Docling** 2.120.x (2.120.1, 2026-08-14; IBM, MIT) | Full converter: layout (RT-DETR "heron", ~78% mAP), TableFormer, picture classifier, OCR (RapidOCR/EasyOCR/tesseract) | Yes (DoclingDocument items with `prov[].page_no`, `bbox` — **BOTTOMLEFT origin**) | `PictureItem` with `caption_text(doc)`, `get_image(doc)` at `images_scale` (1.0 = 72 dpi); `generate_picture_images=True` | TableFormer → DataFrame/HTML | Built in | `pip install docling` pulls torch; models to `~/.cache/docling/models` (`docling-tools models download`), several hundred MB | Best "batteries-included" OSS option; slower (seconds/page on CPU/MPS). olmOCR-bench ≈50% (their harness). |
| **Marker 2.0** (Datalab, Jul 2026) | Rewrite; modes balanced/fast/`--disable_ocr` | Markdown/JSON blocks with bboxes | Extracts images; JSON has block types incl. Figure/Caption | Yes | Surya OCR | torch + Surya models (GB-scale); "fast" mode default on MPS | Code Apache-2.0, **model weights modified OpenRAIL-M** (free for research/personal/startups <$5M; commercial licence otherwise — check before bundling); 76.0% olmOCR-bench (their numbers). Good, but a big dependency for what we need. |
| **Surya 2** (Datalab) | Single ~650 M-param VLM: layout + OCR + tables | Yes | Layout labels incl. Picture, Caption, PageHeader… | Yes | Yes | torch/vLLM or llama.cpp | Underlies Marker; usable standalone for layout. |
| **MinerU 2.5/2.6** (OpenDataLab) | Pipeline backend + 1.2 B VLM backend; `vlm-mlx-engine` for Apple Silicon | Yes (middle JSON with bboxes) | Yes (image + caption blocks) | Yes | Yes | ~2.4 GB VLM + deps | 72.7% olmOCR-bench (pipeline). Heavier; Chinese-doc bias but fine on English. |
| **olmOCR 2** (Ai2) | Qwen2.5-VL-7B fine-tune → text | Text only (no bboxes) | No | Text | Yes (its point) | 7B (~15 GB; FP8 ~8 GB), vLLM/GPU; community GGUF ports | 82.4 olmOCR-bench. Not for figure localisation. |
| **Nougat** (Meta 2023) | Page→markdown VLM | No bboxes | No | Text | Yes | ~1.4 GB | Effectively unmaintained; superseded by the above. |
| **GROBID** 0.8.x | Java service, TEI XML | Coordinates optional (`teiCoordinates`) | Figures/tables in TEI with coords | Partial | No | Full Docker image **amd64 only** (emulated on ARM); CRF-only image has native arm64 since 0.8.1 | Great for header/refs metadata (title, DOI, authors) — consider only for citation-level metadata. |
| **PDFFigures2** (AI2) | Scala/JVM figure+caption extractor | — | Yes, JSON with bboxes | Yes | No | JVM, sbt | Old but still the classic baseline for figure/caption localisation; not Python. |
| **DocLayout-YOLO** (OpenDataLab) | YOLOv10-based layout detector | Boxes only | Classes: title, plain text, abandon, figure, **figure_caption**, table, **table_caption**, table_footnote, isolate_formula, formula_caption | boxes | — | `pip install doclayout-yolo`, `doclayout_yolo_docstructbench_imgsz1024.pt` (~40 MB), torch; MPS works | 79.7 mAP DocLayNet; ~real-time. |
| **PP-DocLayout(-V2)** | PaddleDetection RT-DETR variants, 23 classes | Boxes | Yes | Yes | — | Needs PaddlePaddle (awkward on macOS-arm) | 90.4 mAP@0.5 (L). Skip on Mac. |
| **Anthropic native PDF** | `document` content block | No bboxes; `citations` → `page_location` (1-indexed pages) | Claude sees page images; can *describe* figures, cannot return exact bboxes | Reads tables from image+text | Sees scanned pages (as images) | API only | 32 MB/request, 600 pages/request (100 if context <1M), no encryption. |

Benchmark caveat: olmOCR-bench (1,403 PDFs, ~8,400 unit tests) numbers above are vendor-reported and are about *text fidelity*, not figure localisation. There is no public benchmark that measures exactly what Canopy needs (figure/panel bbox recall + caption linkage on physiology/psychology journals), hence the eval plan in §8.

---

## 2. Anthropic native PDF input — what it does and when to use it

Per the PDF-support docs (fetched today):

- Each page is converted to an image **and** its text is extracted; both are given to the model. So Claude reads charts, but you get no coordinates back.
- Limits: 32 MB per request (payload incl. everything), 600 pages/request (100 when the request's context window is <1M tokens — irrelevant for Opus 5/Sonnet 5/Fable 5, all 1M), standard PDF only (no passwords).
- Cost: text ≈1,500–3,000 tokens/page **plus** image tokens per page (image tokens = ⌈w/28⌉×⌈h/28⌉ visual tokens; a page rendered near the standard 1568-px cap ≈1,568 tokens, up to 4,784 on the high-res tier for Claude ≥4.7). Rough per paper: 11 pages ≈ 40–80 k input tokens ≈ $0.20–0.40 on Opus 5 ($5/M input), ~60% of that on Sonnet 5 ($3/M list; $2/M intro through 2026-08-31), half again via the Batches API, and ~90% off on repeat reads with prompt caching (`cache_control` on the document block).
- Sources: base64, URL, or Files API `file_id` (beta header `files-api-2025-04-14`, 500 MB/file). Place the document block *before* the text prompt.
- `citations: {enabled: true}` returns `page_location` (`start_page_number`, 1-indexed) — usable as coarse provenance. Citations are incompatible with `output_config.format`, so a citation pass must be a separate call from the structured-JSON extraction pass.
- The docs don't state the rasterisation DPI of PDF pages, and dense pages "can fill the context window before reaching the page limit". Assume standard-tier resolution; do not rely on it for reading tick labels on small panels — send our own crops instead.
- Bedrock Converse quirk: visual PDF analysis requires citations enabled (otherwise text-only). Not relevant to direct API use.

**Role in Canopy:** a whole-paper reader/verifier agent (with page citations), and the scanned-PDF fallback. Not the primary source of figure crops or bboxes.

---

## 3. (a) Clean text with page numbers and 2-column reading order

**Primary: PyMuPDF `get_text("dict")` + own column sort.**

```python
import pymupdf, re, unicodedata
FLAGS = pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_MEDIABOX_CLIP | pymupdf.TEXT_DEHYPHENATE
doc = pymupdf.open(path)
for pno, page in enumerate(doc, start=1):
    d = page.get_text("dict", flags=FLAGS)          # blocks → lines → spans, each with bbox
    blocks = [b for b in d["blocks"] if b["type"] == 0]
    W = page.rect.width
    # 2-column heuristic: split at the page mid-line, sort (col, y0, x0); full-width blocks first
    def key(b):
        x0, y0, x1, y1 = b["bbox"]
        full = (x1 - x0) > 0.6 * W
        col = 0 if full or x0 < W * 0.5 else 1
        return (0 if full and y0 < 120 else 1, col, round(y0), x0)
    ordered = sorted(blocks, key=key)
```

Refinements that matter for our corpus:
- Detect the column gutter per page from the histogram of block `x0` values (Tsay 2021 pages: `x0 ∈ {44, 320}` → 2 columns; Nature: 3-column pages exist). Don't hard-code 0.5·W.
- Drop headers/footers/running titles: blocks whose text repeats on ≥50% of pages at similar `y`, or `y0 < 40` / `y1 > H−40` with small font. Drop the tiny per-page logos (Tsay: a 59×81 px raster on every page).
- Keep font size/flags from spans: section headers (bold/larger) and caption starts are detectable from typography, which improves both reading order and section segmentation ("Methods", "Results", "Experiment 2").
- Store every emitted paragraph with `{doc_id, page, bbox, text}` so quotes in extractions can be highlighted later (provenance requirement).
- Unicode: captions and numbers use ` `, ` `, ` `, minus signs `−`, `±`. Normalize with `unicodedata.normalize("NFKC", …)` **for matching only**; keep raw text for quotes.
- Hyphenation: `TEXT_DEHYPHENATE` joins line-end hyphens; still watch for "adap-tation" survivors.
- Second opinion: `pdftotext -layout -f N -l N` (poppler 25.05) is a cheap alternative ordering; if the two orderings disagree materially (e.g. Jaccard of 5-gram sequences < 0.9), flag the page for the layout-model path.

**Fallback (optional extra): `pymupdf4llm.to_markdown(path, page_chunks=True, extract_words=True, header=False, footer=False, use_ocr=…)`** returns per-page dicts with `text` (markdown), `words` (bboxes), `tables`, `images`, `graphics`, `page_boxes` (layout class + bbox). With `pymupdf_layout` installed, `import pymupdf4llm` activates the GNN layout automatically (CPU only, no torch; reported F1 0.864 on their layout eval). License gate: `pymupdf-layout` is AGPL-3.0/commercial from 1.28.2 (Polyform NC before) → still ship behind `pip install canopy[layout]` (large model download, separate licence text).

**Docling** gives the same via `DocumentConverter().convert(path).document.export_to_markdown()` plus per-item provenance; use if we adopt Docling for figures/tables anyway.

---

## 4. (b) Locating figures + captions, high-res crops, and linking "Fig. 3B"

What the three test papers showed (PyMuPDF, ~0.1 s/page):

| Paper | Figure encoding | Signals available |
|---|---|---|
| Tsay 2021 (JNP) | Pure embedded rasters (2150×576, 1500×737 px) | `get_image_info` gives placement rect + xref; captions "Figure 1. …" as separate blocks |
| Nature 2022 | Mixed: 850 vector drawings + ~18 rasters on one page; several vector clusters per figure | `cluster_drawings` → 8–17 clusters/page, needs merging; captions "Fig. 2 \| …" |
| Exp Brain Res 2023 | Rasters + a vector cluster covering the whole figure region | Caption "Fig. 1  …" (Unicode spaces) |

So: **no single signal is sufficient; combine caption anchoring with raster rects, vector clusters, and (fallback) a layout detector.**

Algorithm (per page):
1. **Captions**: from text blocks, regex `^\s*(Fig(?:ure)?\.?|Table|Extended Data (?:Fig|Table))\s*\.?\s*(\d+)` after NFKC + whitespace collapse; require caption-like typography (font size ≤ body, block width ≥ column width, starts a block). Reject in-body cross-references ("Figure 1d and Extended Data Table 1" was a false positive in the Nature paper) by requiring the block to be ≥ 2 lines *or* followed by a period/colon/pipe pattern.
2. **Candidate graphic regions**: `page.get_image_info(xrefs=True)` (filter width/height > 100 px and area > 1% of page) ∪ `page.cluster_drawings(x_tolerance=3, y_tolerance=3)` (filter `w > 60 pt and h > 40 pt`; drop clusters that intersect `find_tables()` bboxes or that are just horizontal rules) ∪ (optionally) DocLayout-YOLO `figure` boxes.
3. **Merge**: union overlapping/adjacent candidates (gap ≤ 12 pt); then assign each merged region to the nearest caption *below* it (or beside it, for side captions) within the same column span; the figure bbox = union(region) expanded by 4–6 pt, clipped to page.
4. **Panels**: parse panel letters from the caption ("A:", "(B)", "a,", "**a**" bold spans) and, when the crop must be split, either (i) split the region at large white gaps in the 150-dpi render (row/column projection profile in OpenCV), or (ii) ask Claude for panel bboxes on the crop (approximate — verify by re-cropping and checking the panel letter is visible). Store panels as children of the figure.
5. **Crops**: for each figure/panel: `page.get_pixmap(clip=bbox, dpi=400, colorspace=pymupdf.csRGB, alpha=False)` (vector and mixed); if the region is a single raster whose native pixel size exceeds the 400-dpi render size, also `doc.extract_image(xref)` and store both (native is loss-free; note `get_image_info()["transform"]` may rotate/crop, and SMask/alpha must be composited). 400 dpi on a 3.5-in column figure ≈ 1400 px wide; 600 dpi for small panels. Claude's high-res tier caps at 2576 px long edge / 4,784 tokens; keep master crops on disk and downsample per request.
6. **In-text references**: scan ordered text for `\b(Fig(?:ure)?s?\.?|Figs?\.)\s*(\d+)\s*([A-Za-z](?:[,–-][A-Za-z])*)?` and `Table\s*(\d+)`; record `{page, bbox(span), figure_id, panel}` per mention; also capture the sentence for the extractor ("… late adaptation was reduced in older adults (Fig. 3B)"). Prefer span-level bboxes from `get_text("dict")` for exact highlighting.

**Detector fallback details (DocLayout-YOLO):**
```python
from doclayout_yolo import YOLOv10
m = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")   # HF: juliozhao/DocLayout-YOLO-DocStructBench
pix = page.get_pixmap(dpi=150); res = m.predict(pix.pil_image(), imgsz=1024, conf=0.25, device="mps")
# boxes in pixel coords → divide by (150/72) to get PDF points; classes incl. figure, figure_caption, table, table_caption, abandon
```
Run it on every page whose heuristics found a caption without a region (or a region without a caption), and on a random 10% for QA. Docling (`PictureItem` + `caption_text(doc)` + `prov[0].bbox.to_top_left_origin(page_h)`) is the alternative if we prefer one converter for figures+tables+text; remember its bboxes are BOTTOMLEFT-origin and its `images_scale=1.0` means 72 dpi (set 4–6 for 300–430 dpi, or crop with PyMuPDF from its bboxes instead — cheaper).

---

## 5. (c) Tables → structured rows

1. `tabs = page.find_tables(strategy="lines_strict")` → if none, `"lines"` → `"text"`. `tab.to_pandas()`, `tab.header.names`, `tab.bbox`, `tab.cells`. Link to the caption ("Table 2") above; store `{page, bbox, caption, header, rows}` plus a 300-dpi crop.
2. Sanity filters: reject "tables" that are figure axes/legends (bbox inside a figure region), 1×N tables, or ones on the title page (Tsay p.1 gave a false positive with default settings).
3. Fallbacks: pdfplumber `extract_tables({"vertical_strategy":"text","horizontal_strategy":"text"})` for borderless tables; Docling TableFormer; or Claude vision on the crop with `output_config.format` JSON schema `{header: [...], rows: [[...]]}` plus a re-read check (row/col counts, numeric parse rate).
4. Numeric normalisation happens downstream, but the ingest layer should keep the raw cell strings (`12.3 ± 1.1`, `12.3 (1.1)`, `n = 14`) and cell bboxes for provenance.

---

## 6. (d) Scanned PDFs / OCR fallback

Detection: page has < ~50 text characters and ≥1 raster covering > 70% of the page ⇒ scanned. Also handle "hybrid" PDFs where a publisher pasted a low-quality OCR layer (check `get_text` for garbage: high ratio of non-alphanumerics or missing spaces).

Options, in order:
1. **`ocrmypdf --skip-text --rotate-pages --deskew --optimize 0 in.pdf out.pdf`** (`brew install ocrmypdf` or `pip install ocrmypdf`; uses tesseract 5.5.1, ghostscript). Produces a searchable PDF; the rest of the pipeline is unchanged (text with bboxes via PyMuPDF). Use `-l eng`, `--tesseract-oem 1`, `--oversample 300`.
2. **PyMuPDF**: `tp = page.get_textpage_ocr(flags=..., language="eng", dpi=300, full=False)` then `page.get_text("dict", textpage=tp)` — `full=False` only OCRs image areas (good for hybrid pages). Needs `TESSDATA_PREFIX` (`brew --prefix tesseract`/share/tessdata).
3. **`pdftoppm -r 300 -png`** + `pytesseract.image_to_data(..., output_type=DICT)` when we want word-level confidences.
4. **Claude native PDF/vision** for degraded scans; keep page citations as provenance and mark `ocr_source="claude"` (lower trust; require verifier agreement).

Figures in scanned pages: no vector/raster structure — use DocLayout-YOLO on the page render (it is trained on scanned/photographed docs too) and crop from the 300–400 dpi render.

---

## 7. (e) De-duplication and multi-experiment papers

- **Byte-identical**: SHA-256 of file bytes (fast path).
- **Same paper, different file** (re-download, watermarked copy, "(1).pdf"): fingerprint = SHA-1 of NFKC-normalized, whitespace-collapsed text of pages 1–2 with digits removed; plus DOI (`10\.\d{4,9}/[-._;()/:A-Za-z0-9]+`) from text or `doc.metadata` (Tsay: `subject='Journal of Neurophysiology 2021.125:12-22'`, title present) and normalized title. Any match ⇒ same work.
- **Near-duplicates** (preprint vs. version of record; main text vs. supplement): MinHash (datasketch, 5-word shingles) Jaccard > 0.8 ⇒ cluster; keep the VoR as canonical, attach the others as `related_files` (supplements are needed for tables!). Match supplement PDFs to the main paper by DOI/title in the first page ("Supplementary material for …").
- **Multi-experiment papers**: segment by headings/typography (`Experiment 1`, `Study 2`, `Exp. 1a`) using the section tree from §3; each figure/table/caption is assigned to the enclosing experiment section, and each caption panel may cite an experiment ("A: Experiment 1 …"). The extraction unit is `(work_id, experiment_id, outcome, group_pair)`; the meta-analysis dataset key is `doi + experiment + outcome`. This is what makes the tool general (Cisneros has several papers contributing multiple datasets).

---

## 8. Recommended stack, fallback chain, install, sizes, pitfalls

**Install (macOS/Apple Silicon, Python 3.13):**
```bash
# core (no torch)
pip install "pymupdf>=1.26.7" pdfplumber datasketch pytesseract ocrmypdf
brew install tesseract poppler ghostscript ocrmypdf   # tesseract 5.5.1 & poppler already present
export TESSDATA_PREFIX="$(brew --prefix tesseract)/share/tessdata"
# optional layout detector (torch 2.11 already installed; MPS ok)
pip install doclayout-yolo huggingface_hub
python -c "from huggingface_hub import hf_hub_download as d; print(d('juliozhao/DocLayout-YOLO-DocStructBench','doclayout_yolo_docstructbench_imgsz1024.pt'))"   # ~40 MB
# optional all-in-one converter
pip install docling && docling-tools models download        # torch + models, several hundred MB to ~1 GB
# optional layout for pymupdf4llm (pymupdf-layout>=1.28.2 is AGPL/commercial; <=1.28.0 was Polyform-NC)
pip install "canopy[layout]"  # -> pymupdf4llm pymupdf-layout>=1.28.2
```
Upgrade note: PyMuPDF 1.28.2 (2026-08-06) is current; 1.26.7 (installed) has every API used here. Pin `>=1.26.7,<2`.

**Fallback chain per page:** PyMuPDF heuristics → (if caption without region / region without caption / scanned page) DocLayout-YOLO on 150-dpi render → (if still unresolved or QA sample) Docling PictureItems → (if still unresolved) Claude vision on the page image asking for approximate figure boxes, then re-crop and confirm. Text: PyMuPDF column sort → poppler agreement check → pymupdf4llm/Docling layout → OCR path.

**Sizes/deps summary:** PyMuPDF wheel ~24 MB (macOS arm64; 18–26 MB by platform), no torch. DocLayout-YOLO ~40 MB + torch (already present). Docling: torch + heron layout (RT-DETR, ~100–200 MB) + TableFormer (~200 MB) + RapidOCR — budget ~1 GB. Marker 2 / Surya 2: torch + ~650 M-param VLM + detectors — budget 2–3 GB. MinerU 2.5 VLM: ~2.4 GB (MLX backend on Mac). olmOCR-2-7B: ~15 GB (FP8 ~8 GB), GPU/vLLM — not a local Mac path.

**Known pitfalls:**
- PyMuPDF `cluster_drawings` may return many small clusters (issue #4599 in early 1.26; tune `x_tolerance/y_tolerance`, post-merge). Header rules and table lines are drawings too — exclude via table bboxes and aspect ratio.
- `find_tables()` false positives on title pages/figure axes; always intersect with a caption or column geometry.
- Some rasters are placed with rotation/crop (`transform`), or split into strips (Nature page 3 had 6 near-identical 439×77 strips) — union by caption, don't trust one xref = one figure.
- Journal PDFs may embed the same figure twice (thumbnail + full); pick the largest placement rect.
- Unicode spaces and `Fig.` vs `Figure` vs `FIG.` in captions; "Extended Data Fig." exists; some journals put captions *above* figures or at the end of the manuscript (author manuscripts/preprints: figures on separate pages after references) — the caption-to-region assignment must allow "nearest region on the same or next page".
- Coordinates: PyMuPDF is top-left origin in points; Docling BOTTOMLEFT; YOLO pixel coords at the render DPI; Claude returns pixel coords in *its* resized image (approximate). Store everything as PDF points + page index and convert at the edges.
- Claude image limits: 8000×8000 px max, 10 MB base64, >20 images per request triggers a stricter per-image size limit (≤2000 px per side) — send panels one to a few at a time; use the Files API for repeated crops.
- Licensing: PyMuPDF/pymupdf4llm AGPL (fine for an open-source tool; note in README); pymupdf_layout AGPL/commercial from 1.28.2 (Polyform-NC ≤1.28.0); Marker code Apache-2.0 but its model weights are OpenRAIL-M (not permissive for commercial use); DocLayout-YOLO is **AGPL-3.0** (not Apache); Docling MIT.
- Nougat/olmOCR/Marker output has no reliable bboxes → cannot satisfy the provenance requirement alone.

---

## 9. Evaluation plan (figure-detection recall on our own PDFs)

1. **Gold set**: 25–30 PDFs from the Cisneros 2024 study list (mix of journals: JNP, J Neurosci, Exp Brain Res, PLoS, Frontiers, Psych Aging; include ≥3 author-manuscript/preprint layouts and ≥2 scans). Annotate per page: figure bboxes (whole figure), panel bboxes where letters exist, caption bbox + figure number, table bboxes + captions. Bootstrap annotations with Docling/DocLayout-YOLO, then correct by hand in a small viewer (PyMuPDF renders + `labelme`, or a 100-line Streamlit page); store as JSON `{doc, page, type, id, bbox_pt}`. Budget: ~2 h.
2. **Metrics**: figure recall/precision at IoU ≥ 0.5 (and ≥ 0.75), per journal; caption-linkage accuracy (right figure number); panel recall; "crop legibility" = Claude Sonnet 5 reads the axis tick labels correctly (spot check 30 crops); table cell F1 on 10 hand-transcribed tables; text: word-level agreement vs `pdftotext -layout` and reading-order correctness on 20 two-column pages (caption/heading sequence); OCR WER on the scanned pages; wall time per page.
3. **Ablations**: heuristics only vs +YOLO vs Docling; 150 vs 200 dpi detector input; crop dpi 300/400/600 vs Claude token cost.
4. **Regression**: keep the gold JSON in `validation/ingest/` and a pytest that runs the ingest on the gold PDFs (paths outside the repo, gitignored) and asserts recall ≥ 0.95 / linkage ≥ 0.98; print a per-journal table.

---

## Sources

- Anthropic PDF support: https://platform.claude.com/docs/en/build-with-claude/pdf-support.md
- Anthropic vision (image limits, visual-token formula, high-res tier): https://platform.claude.com/docs/en/build-with-claude/vision.md
- Anthropic Files API: https://platform.claude.com/docs/en/build-with-claude/files.md ; citations: https://platform.claude.com/docs/en/build-with-claude/citations.md
- PyMuPDF on PyPI (1.28.2, 2026-08-06, AGPL/commercial): https://pypi.org/project/pymupdf/ ; changelog: https://pymupdf.readthedocs.io/en/latest/changes.html
- PyMuPDF4LLM API (`to_markdown`, `page_chunks`, `page_boxes`): https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html ; layout announcement: https://pymupdf.io/blog/introducing-the-new-pymupdf4llm-now-including-layout ; pymupdf-layout (AGPL/commercial from 1.28.2; Polyform NC ≤1.28.0): https://pypi.org/project/pymupdf-layout/
- PyMuPDF cluster_drawings discussion / issue: https://github.com/pymupdf/PyMuPDF/discussions/2312 ; https://tessl.io/registry/tessl/pypi-pymupdf/1.26.0/files/docs/table-extraction.md
- Docling install/models: https://docling-project.github.io/docling/getting_started/installation/ ; export figures example: https://github.com/docling-project/docling/blob/main/docs/examples/export_figures.py ; layout models paper (heron): https://arxiv.org/abs/2509.11720 ; model card: https://huggingface.co/docling-project/docling-layout-heron
- Marker 2.0 release: https://github.com/datalab-to/marker/releases/tag/v2.0.0 ; benchmark write-up: https://www.marktechpost.com/2026/07/24/datalab-marker-v2-vs-mineru-docling-and-liteparse-benchmark-breakdown/
- Surya: https://github.com/datalab-to/surya ; https://pypi.org/project/surya-ocr/
- MinerU changelog (MLX backend, 2.5 VLM): https://opendatalab.github.io/MinerU/reference/changelog/ ; model: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
- olmOCR 2: https://allenai.org/blog/olmocr-2 ; https://huggingface.co/allenai/olmOCR-2-7B-1025 ; paper: https://arxiv.org/abs/2510.19817
- Nougat: https://github.com/facebookresearch/nougat
- GROBID coordinates & docker: https://grobid.readthedocs.io/en/latest/Coordinates-in-PDF/ ; https://grobid.readthedocs.io/en/latest/Grobid-docker/
- PDFFigures2: https://github.com/allenai/pdffigures2
- DocLayout-YOLO: https://github.com/opendatalab/DocLayout-YOLO ; https://pypi.org/project/doclayout-yolo/ ; weights: https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench ; paper: https://arxiv.org/abs/2410.12628
- PP-DocLayout: https://arxiv.org/abs/2503.17213
- Local probes (this machine, 2026-08-14): PyMuPDF 1.26.7 on Tsay et al. 2021, a Nature 2022 article, and an Exp Brain Res 2023 article — numbers quoted in §4.

---

## Verification notes (fact-check)

*Fact-check pass 2026-08-14/15 against primary sources (PyPI JSON API, GitHub API/raw files, Anthropic platform docs, arXiv, HF Hub API). Wrong claims were corrected in place above; this list records what was checked.*

| # | Claim | Verdict | Source |
|---|---|---|---|
| 1 | PyMuPDF 1.28.2 is current, released 2026-08-06, AGPL-3.0 / Artifex commercial | **Confirmed** (macOS arm64 wheel is ~24 MB, not ~30 MB — adjusted) | https://pypi.org/pypi/pymupdf/json ; https://pypi.org/project/pymupdf/ |
| 2 | `pymupdf_layout` is Polyform-Noncommercial licensed | **Corrected** → 1.28.2 (2026-08-06) metadata and repo LICENSE are AGPL-3.0 / Artifex commercial; 1.28.0 (2026-06-29) was "Polyform Noncommercial or Artifex Commercial"; 1.26.6 said "Commercial license" | https://pypi.org/pypi/pymupdf-layout/1.28.2/json ; https://pypi.org/pypi/pymupdf-layout/1.28.0/json ; https://github.com/ArtifexSoftware/pymupdf_layout/blob/master/LICENSE |
| 3 | Anthropic PDF input: 32 MB/request, 600 pages (100 when context window < 1M), no passwords; text 1,500–3,000 tokens/page plus image tokens; place PDF before text; prompt caching supported; Bedrock Converse needs citations for visual analysis | **Confirmed** | https://platform.claude.com/docs/en/build-with-claude/pdf-support.md |
| 4 | Vision: tokens = ⌈w/28⌉×⌈h/28⌉; standard tier 1568 px / 1568 tokens; high-res tier (Claude 4.7+) 2576 px / 4784 tokens; 8000×8000 px max; 10 MB base64; >20 images ⇒ ≤2000 px per side | **Confirmed** | https://platform.claude.com/docs/en/build-with-claude/vision.md |
| 5 | Files API: beta header `files-api-2025-04-14`, 500 MB/file, `document` block with `file_id`, `citations` allowed | **Confirmed** (org storage cap is 500 GB) | https://platform.claude.com/docs/en/build-with-claude/files.md |
| 6 | Cost: Opus 5 $5/M input; "half that on Sonnet 5"; Batches −50%; caching ~90% off | **Corrected** wording: Sonnet 5 list is $3/M input ($2/M intro through 2026-08-31) → ~60% (40% intro) of Opus, not exactly half. Opus $5/M, Batches 50%, cache-read ≈0.1× confirmed | https://platform.claude.com/docs/en/pricing.md ; https://platform.claude.com/docs/en/build-with-claude/batch-processing.md ; https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md |
| 7 | Citations (`page_location`, 1-indexed) incompatible with `output_config.format` (400) | **Confirmed** | https://platform.claude.com/docs/en/build-with-claude/citations.md ; https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md |
| 8 | DocLayout-YOLO: `pip install doclayout-yolo`, YOLOv10-based, `doclayout_yolo_docstructbench_imgsz1024.pt` ~40 MB, 79.7 mAP DocLayNet (DocSynth300K-pretrained), classes title/plain text/abandon/figure/figure_caption/table/table_caption/table_footnote/isolate_formula/formula_caption | **Confirmed** (weight file is 40,709,302 bytes). License is **AGPL-3.0**, not Apache — corrected in pitfalls | https://github.com/opendatalab/DocLayout-YOLO ; https://huggingface.co/api/models/juliozhao/DocLayout-YOLO-DocStructBench ; https://arxiv.org/abs/2410.12628 |
| 9 | olmOCR 2 = 82.4 olmOCR-bench, Qwen2.5-VL-7B fine-tune; olmOCR-bench = 1,403 PDFs / ~8,400 unit tests | **Confirmed** (Ai2 blog for 82.4 and base model; bench size from Marker README quoting olmOCR-bench) | https://allenai.org/blog/olmocr-2 ; https://github.com/datalab-to/marker (README) |
| 10 | Marker 2.0 released Jul 2026, 76.0% olmOCR-bench (balanced), fast 66.6%, `--disable_ocr` CPU mode; MinerU pipeline 72.7; Docling ≈50 (Marker harness) | **Confirmed** (v2.0.0 published 2026-07-20; Docling = 50.3). License **corrected**: code Apache-2.0, weights modified OpenRAIL-M (not "GPL-ish") | https://api.github.com/repos/datalab-to/marker/releases/tags/v2.0.0 ; https://github.com/datalab-to/marker/blob/master/README.md |
| 11 | Docling "2.5x", MIT; heron layout ≈78% mAP; `images_scale=1.0` ≈ 72 dpi; bboxes BOTTOMLEFT | **Corrected** version → 2.120.1 (2026-08-14). MIT, heron-101 78% mAP, scale=1 ~72 DPI confirmed. BOTTOMLEFT: `CoordOrigin` enum exists and the PDF pipeline emits BOTTOMLEFT prov boxes (convert with `to_top_left_origin`) — confirmed by docs example, not re-run locally | https://pypi.org/pypi/docling/json ; https://api.github.com/repos/docling-project/docling ; https://arxiv.org/abs/2509.11720 ; https://github.com/docling-project/docling/blob/main/docs/examples/export_figures.py ; https://github.com/docling-project/docling-core/blob/main/docling_core/types/doc/base.py |
| 12 | GROBID Docker images are amd64-only (emulated on ARM) | **Corrected/qualified**: full image amd64-only (emulated on macOS/arm64); CRF-only image ships native arm64 since 0.8.1 | https://grobid.readthedocs.io/en/latest/Grobid-docker/ |
| 13 | PP-DocLayout-L 90.4 mAP@0.5, 23 classes | **Confirmed** | https://arxiv.org/abs/2503.17213 |
| 14 | MinerU 2.5 VLM ~2.4 GB, olmOCR-2 ~15 GB / FP8 ~8 GB, Surya 2 ~650 M params, pymupdf4llm layout F1 0.864, PyMuPDF `cluster_drawings` issue #4599 | **Unverifiable in this pass** (not checked against primary sources; treat as approximate) | — |
