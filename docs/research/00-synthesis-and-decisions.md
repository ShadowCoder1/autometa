# 00 — Synthesis and design decisions for Canopy

*Written 2026-08-15 from research briefs 01–06 in this folder. This is the decision record the implementation should follow. Every decision cites the brief(s) it rests on as `[NN §x]`. Where briefs disagree, the resolution is stated in §0.3. Anything marked **OPEN** must be resolved during implementation and the resolution appended here.*

Brief index: **01** PDF ingestion & layout · **02** chart data extraction · **03** meta-analysis statistics (+ `03-meta-stats-reference-impl.py`, `03-meta-stats-r-oracle.R`) · **04** existing automated-review tools · **05** LLM extraction reliability engineering · **06** Anthropic API fact sheet.

---

## 0. Cross-cutting decisions

### 0.1 The one-sentence architecture

Deterministic, local ingestion (PyMuPDF) produces page-anchored text, table cells, figure crops and vector geometry with provenance; heterogeneous Claude agents **locate, label and critique** but never measure or compute; pixels are measured by OpenCV / PDF vector geometry; every scalar is grounded (quote+page or crop+pixel coords), voted on by ≥2 independent routes, adversarially verified, adjudicated only on dispute, and passed through code-only consistency checks; effect sizes and pooling are computed in NumPy/SciPy replicating `metafor`/`meta` bit-for-bit and validated against an R oracle. [02 §7 design rules; 04 §4–6; 05 §0, §4]

### 0.2 Cross-cutting invariants (apply to every stage)

| Invariant | Decision | Source |
|---|---|---|
| **Coordinate convention** | Store all geometry as PDF points, top-left origin, 1-indexed logical page; convert at the edges (Docling is BOTTOMLEFT; YOLO is pixels at render DPI; Claude returns pixels of the *resized* image it saw). | [01 §8 pitfalls; 02 §2.3; 06 §4] |
| **Extraction unit / dataset key** | `(work_id, experiment_id, outcome_id, group_pair)`; meta-analysis row key `doi + experiment + outcome`. This is what keeps the tool general (multi-experiment papers, shared control groups). | [01 §7; 04 §5 item 8] |
| **Provenance record** | Every scalar carries `{doc_id, page, bbox_pt, quote (verbatim) \| crop_path + pixel_coords + calibration, source_kind, effect_source, model_id, prompt_version, raw_response_id}`. Nothing enters the analysis without it. | [01 §3; 04 §4 "self-proving"; 05 §4.1] |
| **Model does not do arithmetic** | LLMs report values *as written* + units + error-bar type + n. SE→SD, CI→SD, t/F→d, pixel→value, sign orientation, pooling: all in code with unit tests. | [03 §5, §8; 04 §5 item 6; 05 §4.7] |
| **Ambiguity is flagged, never resolved by the model** | SD vs SE vs CI, baseline vs post, per-trial vs per-block, signed vs magnitude → `unknown` + `needs_human`, adjudicator may decide only with quotes. | [02 §6, §8; 04 §5 item 3; 05 §4.5] |
| **Default model roster** | Opus 5 (`claude-opus-5`, $5/$25, high-res vision, 512-token cache min) for figure work + primary extraction; Sonnet 5 (`claude-sonnet-5`, $2/$10 permanent, high-res vision, 1024 cache min) for second-route extraction, verifier votes, locator; Fable 5 (`claude-fable-5`, $10/$50, 30-day retention required, refusal→`fallbacks:"default"`) adjudicator only, optional; Haiku 4.5 only for cheap classification (standard vision tier, no `effort`). Query `client.models.retrieve()` at start-up instead of hard-coding capabilities. | [05 §3.2; 06 §1, §13] |
| **Sampling knobs are gone** | `temperature/top_p/top_k` 400 on Opus 4.7+/Sonnet 5/Fable 5 → diversity comes from model × prompt variant × crop/DPI variant × effort, never from sampling. | [05 §1, §4.2; 06 §1] |
| **Licensing** | Core is PyMuPDF (AGPL-3) + OpenCV (Apache) + Docling (MIT, optional). `pyproject.toml` currently says MIT — **OPEN**: either relicense the package AGPL-3 or make PyMuPDF an interchangeable backend. DocLayout-YOLO is AGPL-3, pymupdf-layout ≥1.28.2 AGPL/commercial (pin `>=1.28.2`, never ≤1.28.0 Polyform-NC), Marker weights OpenRAIL-M (do not bundle). | [01 §8 pitfalls, verification #2, #8, #10] |

### 0.3 Conflicts between briefs and how they are resolved

| Topic | Brief A says | Brief B says | Resolution |
|---|---|---|---|
| Sonnet 5 price | 01/02: $3/$15 list, $2/$10 intro through 2026-08-31 | 05/06 (fact-checked 08-15 against pricing page): **$2/$10 is now permanent**, Sept increase cancelled | Use $2/$10 (05 §7, 06 §1). Recompute any cost tables from 01/02 accordingly. |
| Mid-conversation `role:"system"` on Sonnet 5 | 05 §3.3: "not supported on Sonnet 5" (from bundled skill) | 06 §9/§14 (live page): **supported**, no beta header; only the tool-changes beta excludes Sonnet 5 | Trust 06 but wrap in a try/except that falls back to a user-turn `<system-reminder>` block on 400. |
| Prompt-cache invalidation by `effort` | 05 (initial): only `thinking` | 05 verification 5b / 06 §9: **`thinking` and `effort` both invalidate messages cache** | Hold both constant across agents that share a cached prefix. |
| Var(d) formula attribution | Cisneros Rmd: `d²/(2(N−2))` | 03 verification #2: Borenstein 4.20 / Hedges–Olkin use `2N` | Implement `vtype="rmd"` (default for the Cisneros profile, cited as "Cisneros Rmd formula") and `LS/LS2/UB`; label on the plot footer. |
| `meta` PI default | HTS (k−2 df) | V (k−1 df) since meta 8.0-0 (2024-10-30) | Profile "cisneros-2024" = HTS; profile "metafor" = z; expose `method.predict`. **OPEN**: confirm from the Cisneros `sessionInfo()`. |
| Files API org storage | bundled skill: 100 GB | 06 verification #6: **500 GB** | 500 GB. |
| Human WPD reliability | 02: ICC ≈ .99 (Drevon 2017) | 02 verification #6: exact coefficient behind paywall | Treat "≈.99" as unverified; the evaluation must adjudicate disagreements rather than assume human = truth (04 §3 F13). |

---

## 1. Stage: Ingestion (PDF → text, tables, figures, scans, dedup)

| Aspect | Chosen approach | Alternatives considered | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Parser / renderer | **PyMuPDF ≥1.26.7,<2** for text (`get_text("dict")` with span bboxes), rasters (`get_image_info(xrefs=True)`), vector drawings (`get_drawings`/`cluster_drawings`), tables (`find_tables`), page renders (`get_pixmap(clip, dpi)`), OCR hook | pdfplumber (10–50× slower), Docling (torch, seconds/page), Marker 2 / MinerU / olmOCR / Nougat (GB-scale VLMs, lossy or no bboxes), GROBID (JVM, amd64 image) | Whole 11-page paper in ~1 s with every signal Canopy needs; the only tool that satisfies the bbox-provenance requirement without a GPU. [01 §0, §1, §8] | AGPL licence (see §0.2); `cluster_drawings` over-fragments (tune tolerances, post-merge); prints layout-model hints. |
| Reading order | Own column-aware block sort on `dict` blocks, gutter detected per page from the `x0` histogram; header/footer removal by cross-page repetition; keep font size/flags for section segmentation | `pymupdf4llm`+`pymupdf_layout` GNN (optional extra `canopy[layout]`), Docling layout | Column sort covers >90% of journal papers; layout model only for the ugly rest. [01 §3] | 3-column Nature pages, floating boxes → poppler `pdftotext -layout` agreement check (5-gram Jaccard < 0.9 → route page to layout model). |
| Text normalisation | NFKC + whitespace collapse **for matching only**; raw text retained for quotes; `TEXT_DEHYPHENATE` | — | Captions/numbers use thin/nb spaces, `−`, `±`; quotes must be verbatim for the grounding check. [01 §3; 05 §4.1] | Hyphenation survivors ("adap-tation") → fuzzy ratio ≥0.95 fallback in the quote check. |
| Figure + caption localisation | Caption regex (`^(Fig(?:ure)?\.?\|Table\|Extended Data …)\s*(\d+)`) → candidate regions = raster rects ∪ vector clusters ∪ (fallback) DocLayout-YOLO `figure` boxes → merge (gap ≤12 pt) → assign to nearest caption below/beside, allow same-or-next page | Docling `PictureItem`, PDFFigures2 (JVM), Claude vision for boxes | No single signal suffices (Tsay: rasters only; Nature: 850 vector items/page; EBR: mixed). [01 §4] | In-body "Figure 1d" false positives (require ≥2 lines or punctuation), strips/thumbnails (union by caption, largest rect), captions above / at end of manuscript (next-page rule). |
| Panels | Parse panel letters from caption; split by white-gap projection profile at 150 dpi; ask Claude for panel bboxes only as fallback, then re-crop and confirm letter visible | Model-only panel boxes | Deterministic crop beats model self-cropping (EpiCurveBench top failure). [01 §4; 02 §6] | Irregular panel grids → verifier confirms panel letter in crop. |
| Crops | `get_pixmap(clip, dpi=400)` (600 for small panels), RGB, no alpha; also `extract_image(xref)` when native raster ≥ render size; store master crops, downsample per request with Anthropic `resized_size()` (2576 px / 4784 tokens) | Server-side PDF rasterisation | Vector figures must be rendered; Claude's own page images can't be mapped back to coordinates. [01 §4; 05 §1; 06 §3] | Claude limits: 8000×8000, 10 MB, >20 images ⇒ ≤2000 px per side → send few panels per call. |
| Tables | `find_tables()` `lines_strict → lines → text` → pandas; keep raw cell strings + cell bboxes; link to "Table N" caption; sanity filters (inside figure, 1×N, title page) | pdfplumber text strategy, Docling TableFormer, Claude vision + JSON schema on 300-dpi crop with row/col count re-check | Rule-based finder is good for ruled journal tables; fall back for borderless. [01 §5] | False positives on axes/title pages → intersect with caption/column geometry. |
| Scans / hybrid | Detect (<50 chars + raster >70% page, or garbage text layer) → `ocrmypdf --skip-text --rotate-pages --deskew` → same pipeline; per-page `get_textpage_ocr(full=False)` for hybrid pages; DocLayout-YOLO for figures on scans; Claude native PDF as last resort with `ocr_source="claude"` (lower trust) | Tesseract-only, Claude-only | Keeps one code path after OCR. [01 §6] | OCR WER on degraded scans → require verifier agreement; scanned PDFs are not citable by the citations API [05 §1]. |
| Dedup / grouping | SHA-256 → (DOI ∪ normalised title ∪ text fingerprint of pp.1–2) → MinHash 5-shingle Jaccard >0.8; keep VoR canonical, attach supplements as `related_files`; segment experiments by headings/typography | — | Re-downloads, preprint vs VoR, supplements (needed for tables), multi-experiment papers. [01 §7] | Sub-experiments sharing a control group → explicit `dataset_id` dedup rules visible to the human (04 §5 item 8). |
| Whole-paper second reader | Anthropic native PDF `document` block via Files API `file_id`, `citations` enabled → `page_location` | — | Cheap with caching/batches; coarse page provenance; also the scanned-PDF fallback. Not a source of bboxes. [01 §2; 06 §3, §5] | Dense papers can blow context; measure with `count_tokens`; whole-PDF read ≈100–170k tokens. |

**Open questions (ingestion):** (a) fraction of Cisneros PDFs that are vector vs raster (decides how often the exact vector path in §4 applies) [02 §8]; (b) whether Docling should be adopted as the *single* optional fallback for figures+tables+text or only DocLayout-YOLO (weigh 1 GB models vs 40 MB); (c) header/footer removal thresholds per journal; (d) whether to keep GROBID for citation metadata (title/DOI) or rely on PyMuPDF metadata + regex.

---

## 2. Stage: Figure/table detection & cropping — quality gate

(Detection mechanics are in §1; this table records the *acceptance* decisions.)

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Fallback chain per page | Heuristics → DocLayout-YOLO on 150-dpi render (when caption without region, region without caption, scanned) → Docling PictureItems (QA/unresolved) → Claude vision approximate boxes → re-crop & confirm | Detector-first | Heuristics are exact for born-digital PDFs; detector catches text-glyph figures and multi-figure pages. [01 §4, §8] | Detector on 10% random sample for QA; log which rung produced each box. |
| Which figures to crop | Only figures/tables the Locator agent (A0) marks relevant to a protocol outcome/group, plus any figure referenced by in-text mentions of the outcome | Crop everything | Cost control; A0 enumerates all candidates and records an explicit decision per candidate (omission is the #1 failure in the literature). [04 §3 F1, §5 item 4; 05 §3.2] | Missed figure-only data → the decision list is shown to the human; verifier can request additional crops. |
| Crop resolution policy | Master crop 400–600 dpi on disk; per-request downsample with `resized_size(…, 2576, 4784)`; zoom = crop, never upscale; keep axis labels in the crop; one panel per call for dense figures | Send whole page images | Small elements lose precision when downscaled; PlotPick/ReFocus/PixelCraft gains from crop/zoom. [02 §6; 05 §5; 06 §4] | Token cost ≈ $0.024/4784-token image on Opus 5 → cap images/request, use Sonnet 5 for ensemble members. |
| Table crops | 300-dpi crop stored beside cell strings/bboxes | — | Enables Claude-vision fallback and human review UI. [01 §5] | — |
| Gold set | 25–30 Cisneros PDFs, per-page annotated figure/panel/caption/table bboxes as JSON; regression pytest asserts recall ≥0.95 (IoU 0.5), caption linkage ≥0.98 | — | No public benchmark measures Canopy's need. [01 §9] | Annotation drift → bootstrap with YOLO/Docling then hand-correct; keep PDFs out of repo. |

**Open questions:** detector input DPI 150 vs 200 (ablate); crop DPI 300/400/600 vs Claude token cost (ablate); how to handle figures at manuscript end (author preprints) at scale.

---

## 3. Stage: Text & table extraction (LLM agents)

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Context strategy | **Targeted context**: A0 Locator (Sonnet 5, effort `medium`) reads the whole PDF (`file_id`, cached) once and emits a page map; A1–A5 receive only relevant page images (our rasters) + PyMuPDF text (4–10 pages ≈15–40k tokens) + crops | Whole PDF to every agent (~120k tokens/read, forces one model to share cache) | Cheaper, allows heterogeneous models, puts *our* images in front of the model so coordinates are usable. [05 §3.3, §7] | A0 misses a page → in-text figure/table reference scan (ingestion) is a second locator; human sees A0's decision list. |
| Two heterogeneous extractors | A1 Opus 5 `high` "table-first" (page images + text) and A2 Sonnet 5 `high` "narrative-first" (different field order, different DPI, with/without text layer) — different model **and** route | Same model twice; single pass | Diversity of routes beats repetition (Mayo dual-LLM 0.94 vs 0.89/0.90, hallucination 0.2% vs 2.3–2.8%; identical calls cannot fix consistent hallucinations). [04 §1, §4, §6; 05 §4.2] | Cost ×2 → batches, caching of shared excerpt. |
| Output contract | Structured outputs (`output_config.format` json_schema / `client.messages.parse` + pydantic), **quote-first field order** (`page, location_kind, quote, table_id/row_header/column_header, statistic_kind, n, values_as_written, unit, error_type_confidence, notes, value`), `additionalProperties:false`, all fields required, ≤16 union-typed params, no numeric bounds in wire schema (pydantic enforces client-side) | Free text + regex; tool-use JSON | Grammar-constrained JSON; field order forces grounding before value. [05 §4.1; 06 §8] | Citations incompatible with structured outputs (400) → grounding is schema-native + code check; citations only in the verifier pass. |
| Grounding check (code) | Quote must be a verbatim substring (after NFKC/minus/ligature normalisation) of PyMuPDF text of the stated page; fuzzy ≥0.95 fallback; table cells require row+column header present on page and numeric string inside quote; else `ungrounded` → verifier | Trust model | Deterministic anti-hallucination; Manalyzer self-proving +5.4 pts. [04 §4; 05 §4.1] | PDF text-layer quirks (ligatures, column order) → test on Cisneros PDFs [05 §9]. |
| Protocol compilation | User protocol (groups, outcomes, timepoints, sign convention, units) compiled into prompt text **and** schema enums; agents may only emit records for enumerated outcomes/groups | Generic "extract outcomes" prompt | +14.8% recall vs generic prompts; group/timepoint mis-mapping is the top wrong-number cause. [04 §4, §5 item 1] | Over-constrained enums miss synonyms → allow `other` with free-text label routed to human. |
| Figures in text pass | Extractor emits `location_kind="figure"` records with null values — it never eyeballs figures | Let extractor read bars | Model value read-out is 2–6% of axis range and worse on error bars. [02 §0; 05 §6] | — |
| Statistic conversions | Recorded as reported (`mean_sd\|mean_se\|mean_ci\|median_iqr\|t\|F\|p\|r\|eta2`), converted in code with `effect_source` tag | Model computes d | See §5. [03 §2–3] | Paired/RM t, interaction F, p-bounds → not auto-convertible; escalate. [03 §3] |
| Prompt style | Short frozen system prompt; goals + constraints, not step scripts; "report every doubt with confidence, filter downstream" | Long CRITICAL/MUST prompts | Opus 5 / Sonnet 5 follow instructions literally; over-prescription reduces quality; severity filters depress recall. [05 §4.5, §6; claude-api guidance] | Re-baseline prompts on each model release; log `prompt_version`. |

**Open questions:** effort sweep per agent (`medium` for A0; `high` vs `xhigh` for A1) on the gold set; whether A2 should be Fable 5 for the hardest papers; how many records per call (all groups vs one group per call) — trade recall vs "multi-instance compression" [04 §3 F1].

---

## 4. Stage: Figure digitisation (values ± error from charts)

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Division of labour | **VLM for semantics** (which mark = which group/condition, error-bar type from legend/caption, axis tick text, approximate seeds); **CV / vector geometry for measurement** (bar tops, cap ends, tick positions); **least-squares axis fit** for pixel→data | Pure VLM read-out; specialised chart models (DePlot, TinyChart, ChartMoE, ExChart) | Frontier VLMs read values at 2–6% of axis range and error bars proportionally worse; specialists collapse on real figures; PixelCraft/ExChart show coordinates→CV is the winning hybrid. [02 §0, §2, §3] | Anthropic calls localisation "approximate" → VLM coords are seeds only; snap to Canny edge within ±4 px. |
| Path A — vector | If drawings exist under panel bbox: rects/lines/curves + tick text from PyMuPDF → exact bars/whiskers/ticks; VLM only labels | — | Digitisation error essentially zero (<0.5% range) for matplotlib/R/Prism exports. Highest-value trick. [02 §4.2, §7] | Must confirm region isn't an embedded raster; rounding ~0.01 pt. |
| Path B — raster CV | 300–600 dpi crop → axis lines (LSD/Hough) → tick marks + tesseract OCR at 4× → RANSAC LS fit with residual RMS → bars/points/lines by colour → whisker + cap scan (±0.3 bar width) → pixel values ±0.5 px | WPD-style 2-point calibration | n-tick fit removes VLM tick-format biases (FairChart2Table); reimplement WPD algorithms (<200 lines) rather than embed AGPL WPD with proprietary AI backend. [02 §4.1–4.2] | Overlapping/hatched/low-res → Path C. |
| Path C — VLM coordinates + snap | Pre-resize with `resized_size(…,2576,4784)`; Claude Opus 5 (high-res tier) returns absolute pixel coords of tick labels, bar tops, cap ends via structured JSON; snap to nearest edge; LS fit; ensemble ×5 (crop scale ×1/×1.5, two prompts, Opus+Sonnet), per-value median, MAD | Normalised 0–1000 coords (documented to work poorly) | Coordinates map 1:1 when we pre-resize; ensembling +2.7–23% and dispersion predicts error (ρ≈−0.35). [02 §2.3, §6; 05 §5; 06 §4] | Cost ≈ $0.15–0.30/panel with Opus → Sonnet 5 members + batches. |
| Path D — VLM value read-out | Two models × two prompts, "give M and error per group", median/MAD — **cross-check only** | As primary | 3–6% error; used to catch gross errors. [02 §7] | Never sole source. |
| Client-side tools for the model | `crop_image(figure_id,bbox,zoom)` returning an `image` block in `tool_result`, `list_regions()` (OpenCV axes/ticks/legend), `overlay_points(coords)`; drive with `client.beta.messages.tool_runner`, log every call | Server-side code_execution crops | Generated files in the sandbox are **not** seen by Claude; no OpenCV there; Opus 5 guidance: give it crop/verify tools. [05 §1, §5; 06 §11] | Runner does not auto-resume `pause_turn` (n/a without server tools). |
| Overlay-verify | Draw measured tops/caps/calibration ticks on crop → **fresh** model (different from extractor) critiques ("does each mark sit on the line? list mismatches") → iterate ≤2× | Self-critique | PlotExtract render-and-compare 100% precision, ~82–89% recall. [02 §6] | Cost → only when paths disagree or MAD high. |
| Agreement rule | ≥2 paths per value when possible; agreement <2% of axis range → accept; else escalate/flag; tolerance = max(2% axis range, 0.5 × smallest tick interval) | Single path | Cross-path disagreement is the best error detector available. [02 §7; 05 §4.3] | — |
| Per-value uncertainty | `σ_pix² = 0.5² + σ_cap²; σ_val = |b|·σ_pix ⊕ fit residual; σ_ens = 1.4826·MAD`; propagate into `Var(d)_total` (delta method) or at least add to `SE(d)²`; flag if >10% of sampling variance or σ_val >2% range | Ignore | Digitisation error is a real variance component; report it. [02 §7.1] | Sensitivity analysis toggle in the report. |
| Error-bar semantics | Separate text agent extracts "error bars represent …" quote + n per group; unknown → `error_type="unknown"` + `needs_human`; conflicts (legend SEM vs methods SD) → adjudicator with both quotes stored | Guess SE | Dominant *systematic* risk (√n factor on d); pixels can't fix it. [02 §6, §8; 04 §5 item 3] | Cochrane rule: assume SE only with explicit evidence. |
| Special cases | Box plots → 5-number → Wan/Luo/Shi (`estmeansd`-style) with `error_source="boxplot_converted"`; grouped bars by legend swatch RGB; stacked bars usually rejected; log axes fit `log10`; broken axes piecewise or refuse; learning curves: use paper's own late-block value if any, else mean of last N with `approx_late_block` flag; half error bars assumed symmetric and noted; scatter/strip: recompute SD from points, count vs n | — | [02 §7.2; 03 §2] | Each special case gets its own provenance tag and larger σ. |

**Open questions:** (a) empirical snap-to-edge success rate at 2576 px (budget: 50 bars, 50 caps) [02 §8]; (b) how often Path A applies in Cisneros; (c) whether Gemini 3.x should be an optional second family for adjudication (adds a dependency; keep behind a flag); (d) OCR of tick labels — tesseract vs asking the VLM for tick text + snapping; (e) how to score "crop legibility" automatically.

---

## 5. Stage: Effect-size computation

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| SMD | `smd(mA,sA,nA,mB,sB,nB, hedges: bool, vtype: {"rmd","LS","LS2","UB"})` → `(y, v, J, s_p)`; exact `J` via `lgamma`; profile "cisneros-2024" = `hedges=False, vtype="rmd"` (SE(d)=√((n1+n2)/(n1n2)+d²/(2(N−2)))); profile "metafor" = `hedges=True, vtype="LS"` | statsmodels `effectsize_smd` (J=1−3/(4N−9), 2(N−3.94): differs at 4th decimal) | Variants differ ~0.3% of SE — invisible on the plot, visible at 1e-6; must record which. [03 §1] | Attribute the `rmd` formula to the Cisneros Rmd, not Borenstein (whose 4.20 uses 2N). |
| SD recovery | SE·√n; CI→SD with **t** for small n (`√n(U−L)/(2t_{.975,n−1})`), 3.92/3.29/5.15 large-n; MD-CI → s_p; median+IQR/range/5-number → Wan 2014 (η(n), ξ(n)), Shi 2020 default for 5-number, Luo 2018 for means, skew screen (Shi 2023) → flag; digitised points → sample SD (n−1) preferred | z everywhere | Cochrane 6.5.2.x; metafor `conv.fivenum` parity. [03 §2] | Bootstrapped/asymmetric bars → escalate, don't convert. |
| d from statistics | Student t: `d=t√(1/nA+1/nB)`; F(1,N−2): `√(F·N/(nA nB))`; exact p → t (not z); "p<.05" → **bound**, excluded by default; partial η² only for between-subjects; point-biserial r; Welch t needs SDs (else flag); **paired/RM t, interaction F, regression β, BF, U → not auto-convertible** | Convert everything | [03 §3] | `effect_source` tag on every row so verifier weights trust and report footnotes it. |
| Sign / direction | Protocol defines `positive_means`; outcome-level `direction ∈ {higher_is_more, higher_is_worse}` applied **once in code**; extractor emits `raw_value_semantics ∈ {signed_toward_target, signed_error, magnitude}`; d from t/F/p carries `sign_source`; verifier cross-check: ≥1 of {means, text claim} must agree with sign(d), else block | Extractor flips signs | Sign errors are catastrophic and a documented 15.5% failure class. [03 §5; 04 §3 F3, §5 item 2] | Unresolvable sign → exclude + flag. |
| Consistency checks (code) | Σ group n = N; SD>0, SE>0, SE·√n≈SD when both; CI symmetric & half-width ≈1.96·SE (or t); figure value within axis range; \|d\|≤3 flag; t/F→d only with df + group sizes; duplicate numbers across outcomes flag | LLM "check your work" | Deterministic, auditable; nothing silently fixed. [05 §4.4] | Thresholds tuned on gold set. |

**Open questions:** whether digitisation variance should enter the primary analysis or only a sensitivity analysis; default handling of studies with only p-bounds (exclude vs sensitivity); unit harmonisation across degrees / % / normalised units when the outcome is standardised anyway (usually irrelevant for SMD but matters for sign/magnitude semantics).

---

## 6. Stage: Verification / voting / adjudication

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Pipeline shape | **Locate → extract (×2 heterogeneous) → prove (grounding) → check (code) → vote → adversarial verify → adjudicate on dispute → confidence bucket → human queue** | Single pass + human; LLM self-reflection | Each stage measurably adds accuracy (Manalyzer 44.6→65.9→71.3→77.7); self-reflection alone ~+1.8%. [04 §4, §6] | Cost → verifier/vote traffic in Batches. |
| Voting rules | Text/table: exact after unit normalisation (±1 ulp of printed precision); 2 heterogeneous agree → `accepted-by-vote`; disagree → verifier+adjudicator. Figures: align by (group, condition, panel), per-cell median, MAD; tolerance as §4; CV measurement is a voter, not truth | Majority of identical calls | [05 §4.3] | Store every candidate (model, prompt variant, raw JSON, tokens) — the vote is auditable. |
| Adversarial verifier (A5) | Different model than the extractor it checks (Sonnet 5 `xhigh` or Opus 5 `high`); gets only candidate + source page(s) with **citations enabled**; prompt "try to REFUTE: wrong group/timepoint/SE-vs-SD/unit/row/subgroup/absent"; small XML/JSON envelope (`verdict ∈ {confirmed,refuted,ambiguous}`, reason, alternative, `page_location`) | Same-model self-check | Cross-family disagreement is more informative; citations give page-anchored evidence (separate call because citations ⊥ structured outputs). [05 §3.2, §4.5; 06 §5] | Verifier over-refutes → adjudicator sees both rationales; track refute precision on gold set. |
| Adjudicator (A6) | Fable 5 `high` (fallback Opus 5 `xhigh`), invoked only on dispute/ambiguity; sees all candidates + verdicts + crops + protocol; outputs `decision, chosen_source, rationale, needs_human, what_would_resolve_it`; targeted context 10–30k tokens; `fallbacks:"default"` + `server-side-fallback-2026-07-01`; check org retention ≥30 days | Always-on adjudicator | Bounds Fable cost (~30% of papers). [05 §3.2, §7; 06 §10] | Fable refusal/ZDR → fallback; Fable optional in config. |
| Confidence & human flags | Score computed in code from source kind (text 1.0 > table 0.9 > figure 0.7), grounding, # agreeing heterogeneous samples, verifier verdict, MAD, rule results → `auto-accept` / `accept-with-note` / `needs_human`; thresholds calibrated on Cisneros gold; target ≥98% of auto-accepted within tolerance | Ask model for probability | Model probabilities are uncalibrated. [05 §4.6] | Publish the calibration curve. |
| Human-in-the-loop UI | Crop + quote next to each value; one-click accept/override; overrides logged; A0's candidate list visible; discrepancy log | Fully automatic | SWAR: human-verified LLM draft 91.0% vs 89.0% dual human, −41 min/study; AutoForest 80→90.2% with expert edits. [04 §1, §4] | Tool must still run end-to-end with zero human input (requirement). |
| Audit bundle | Model IDs, prompt versions, per-value provenance, raw responses, votes, verdicts, overrides, discrepancy log — exported per run (PRISMA-trAIce / RAISE) | — | Required for Cochrane-style use and reproducibility. [04 §4, §6] | Size → gzip JSONL. |

**Open questions:** vote thresholds for 3-way disagreement; whether to run the verifier on `accepted-by-vote` items too (sampling rate?); how to bound adjudication loops (cap iterations at 2); how to present "abstain / not_reported" (treat as human item, not as correct [04 §4]).

---

## 7. Stage: Meta-analysis + forest plot

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Engine | From-scratch NumPy/SciPy (`canopy/stats/`: `effects.py, pool.py, hetero.py, bias.py, r_oracle.py, report.py`) mirroring metafor algorithms verbatim (REML Fisher scoring with HE start, step-halving, ll0 check; PM/Q-profile via `brentq` xtol=eps^0.25); optional `--engine r` shelling to `Rscript` for auditors | statsmodels (DL only, no REML/PI/HK), PyMARE (REML via generic optimiser, ~1e-6 off), PythonMeta | Only a verbatim recursion reproduces R to ~1e-12; `meta::metagen` calls `rma.uni` for REML so meta and metafor τ² are identical. [03 §4, §6] | R version drift (metafor 5.0-1 / meta 8.5-0 on CRAN vs 4.6-0 / 8.2-1 local) → pin fixtures to versions in header; CI regenerates and fails loudly. |
| Model & options | Random-effects, `method.tau="REML"`, `hakn=FALSE` default; options DL/PM, HK (+ad-hoc "se"), `method.I2 ∈ {"Q" (meta), "tau2" (metafor)}`, `method.predict ∈ {"HTS","V","S","knha"}`, Q-profile τ² CI; config keys mirror R names; profile "cisneros-2024" = REML, hakn F, HTS PI, meta I², d with rmd SE | Fixed-effect | Replicate the validation case exactly while exposing modern defaults. [03 §4, §8] | Two I² definitions and PI-df conventions differ visibly → label on plot footer. |
| Publication bias | Egger via `lm(TE/seTE ~ 1/seTE)` intercept (k≥10 default), trim-and-fill L0 (metafor as oracle), optional Vevea–Hedges step selection model gated on k and p-interval counts | — | [03 §4.6] | k<10 → report but flag. |
| Guard rails | k=1 no pooling; k=2 → HTS PI NaN; v_i≤0 error; Q=0 → HK degenerate; never round before pooling; paper-replication tolerance set by the paper's rounding, not 1e-8 | — | [03 §8] | — |
| Forest plot | matplotlib: per-dataset d + 95% CI, RE weight % (square size), pooled diamond, PI bar, τ², I² (labelled definition), Q/p, k; footer prints the R-equivalent call and formula variant; per-row footnote of `effect_source`; sensitivity toggle including digitisation variance | R plot via Rscript | Local-first, no R at runtime. [03 §8] | Long study lists → paginate/scale. |
| Tests | Golden JSON fixtures from R (5-study, 4-study τ²=0, k=2/3, k=40 wide-v, outlier, all-equal y) with tolerances abs 1e-8 (τ², μ̂, SE, CI, PI, weights), 1e-10 p, 1e-6 I²; property tests (τ²≥0, PI⊇CI, sign-flip invariance, d↔t round trip); Cisneros end-to-end from published d/SE table to 4 dp | — | [03 §6 validation plan] | Existing `validation/fixtures/r_reference.json` + `make_r_fixtures.R` should be pinned to versions. |

**Open questions:** confirm Cisneros `sessionInfo()` (meta version → HTS vs V) [03 §4.4]; whether to ship an R-oracle Docker/`renv` recipe for reproducibility; how to display digitisation-variance-inflated SEs on the plot.

---

## 8. Stage: Cost / caching / batching strategy

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Upload once | Each PDF → Files API `file_id` (beta `files-api-2025-04-14` on upload **and** every referencing call); rasterise once locally; `count_tokens` per paper per model recorded | Base64 every call | 500 MB/file, 500 GB/org, keeps requests <32 MB. [05 §3.1; 06 §6] | Files API ~100 RPM; workspace-scoped (never accept user `file_id`). |
| Cache layout | `tools → system(frozen protocol + schema, cache_control) → user(document/excerpt block, cache_control on last shared block) → task text`; identical `tools/system/thinking/effort` across agents sharing a prefix; fire first call, wait for first token, then fan out; 1-h TTL for batches | Ad-hoc | Prefix match; write 1.25× (5 m)/2× (1 h), read 0.1×; minimums 512 (Opus 5/Fable) / 1024 (Sonnet 5) / 4096 (Haiku); entry readable only after first response begins. [05 §7; 06 §9] | Silent invalidators (timestamps, unsorted JSON, varying tools) → assert `cache_read_input_tokens>0` in tests. |
| Batches | A1–A5 (extractors, verifiers, ensemble members) via Message Batches (50% off, ≤100k req / 256 MB, 24 h, join on `custom_id`, `client.beta.messages.batches` when betas needed); A0 sync (builds excerpts), A6 sync; UI path sync with `messages.stream()` | All sync | Not latency-sensitive; halves cost; no RPM pressure. [05 §7; 06 §7] | Batch cache hits best-effort (30–98%) → monitor, fall back to sync fan-out with warm-up; `fallbacks` and `max_tokens:0` not allowed in batches. |
| Budget targets | Recommended targeted design ≈ **$1.7–2.3/paper sync, ≈$0.9–1.2 batched**; figures ≈ $0.15–0.30/panel (Opus ensemble), ~5× less with Sonnet members; 44-paper validation ≈ $75–100 sync / $40–55 batched | Whole-PDF to 8 Opus agents ≈ $2.85 sync | [02 §8; 05 §7] | Thinking tokens dominate and are unpredictable → per-agent token log from day one; effort sweeps. |
| Rate limits | Start tier: 2M ITPM / 1,000 RPM per model (Opus 5, Sonnet 5 separate buckets); only uncached tokens count → caching is the throughput lever; Fable 5 only 500k ITPM at Start | — | [06 §2] | Back-off on 429 with `retry-after`; SDK retries ×2. |
| Model-side hygiene | Handle `stop_reason ∈ {refusal, max_tokens, pause_turn, model_context_window_exceeded}` before reading content; stream when `max_tokens` >~16k (SDK ValueError above 21,333 non-streaming); pin model IDs; log `usage.*` | — | [06 §10, §12] | — |

**Open questions:** whether Batches accept `file_id` sources via non-beta `client.messages.batches` (use beta client with `betas`) [06 §14]; real per-paper token counts on Cisneros PDFs (measure); whether whole-PDF reads on high-res models exceed ~150k tokens (then feed text-only `document` + our crops) [05 §9].

---

## 9. Stage: API surface (CLI, FastAPI, data model)

| Aspect | Chosen approach | Alternatives | Rationale | Risks → mitigations |
|---|---|---|---|---|
| Inputs | Folder of PDFs + protocol JSON/YAML: research question, group A (reference) / group B (comparison) definitions with synonyms, outcomes with `direction` and timepoint rules, `positive_means`, stats profile (`{"sm":"SMD","method.tau":"REML","hakn":false,"method.predict":"HTS","method.I2":"Q","vtype":"rmd"}`), model roster & effort, budget cap | Interactive wizard only | Protocol is compiled into prompts/enums; R-named config prints on the plot footer. [03 §8; 04 §4] | Validate protocol schema up front; refuse ambiguous sign conventions. |
| Run model | `canopy run <folder> --protocol p.yaml` (Typer CLI) and FastAPI: `POST /runs` (upload folder + protocol) → job id; `GET /runs/{id}` progress (per-paper stage, tokens, cost); `GET /runs/{id}/datasets` (rows with provenance + confidence); `POST /runs/{id}/datasets/{row}/override`; `GET /runs/{id}/forest.png|svg|csv`, `/audit.zip`; SSE/streaming progress | Batch CLI only | Web UI requirement; human review loop. [04 §4; 05 §7] | Long-running jobs → background worker + resumable stage cache keyed by content hash. |
| Storage | Local-first: SQLite (runs, papers, records, votes, verdicts, overrides) + on-disk artefacts (renders, crops, raw responses JSONL); everything content-addressed for resumability | Cloud DB | Local-first is a stated goal; audit bundle is a zip of the run dir. | Disk growth → configurable retention of renders. |
| Data model | `Work → Experiment → Figure/Table/Paragraph (provenance) → Candidate (per agent) → Vote → Verdict → Decision → DatasetRow (y, v, effect_source, flags) → Analysis`; every object carries `model_id/prompt_version/timestamp` | Flat CSV | Auditability + dedup rules. [01 §7; 04 §6] | — |
| Extensibility | Backend interfaces: `PdfParser` (PyMuPDF default; Docling optional), `LayoutDetector` (YOLO optional), `Digitizer` paths A–D pluggable, `LLMProvider` (Anthropic only for v1), stats profiles | Hard-wired | Keeps AGPL/optional deps swappable; avoids special-casing Cisneros. [01 §0, §8] | Interface creep → keep to those five. |
| Outputs | Forest plot (PNG/SVG), datasets CSV/XLSX with provenance columns, `analysis.json` (all stats incl. τ² CI, PI variants), audit bundle, discrepancy log, cost report | — | [03 §8; 04 §6] | — |

**Open questions:** auth for the FastAPI app (local-only default); whether to expose a "review-only" mode that ingests a human table and just runs stats/plot; export to RevMan/CSV formats.

---

## 10. Top 15 engineering requirements (measurable)

1. **Provenance completeness**: 100% of numeric values entering the analysis carry `page + (verbatim quote with bbox \| crop + pixel coords + calibration)`, `effect_source`, `model_id`, `prompt_version`; a deterministic checker rejects any row without them. [01 §3; 05 §4.1]
2. **Figure localisation**: on the 25–30-PDF gold set, figure recall ≥0.95 at IoU 0.5, caption→figure linkage ≥0.98, panel recall ≥0.90; regression pytest in CI. [01 §9]
3. **Digitisation accuracy** vs adjudicated WebPlotDigitizer values on Cisneros: means within 2% of axis range for ≥90% of raster values (<0.5% on vector figures), error-bar half-lengths within 5% for ≥85%; ICC ≥0.95 (ceiling ≈.98 given human error); every value reports σ (pixel + fit + MAD). [02 §7.3, §8]
4. **Downstream fidelity**: pooled estimate, CI, PI, τ², I² for both Cisneros outcomes reproduce the published values within the paper's rounding; per-study |Δd| ≤0.10 for ≥90% of datasets, with disagreements adjudicated against the PDF. [03 §6; 04 §6]
5. **Statistics parity**: `canopy.stats` matches metafor 4.6-0 / meta 8.2-1 golden fixtures at abs 1e-8 (τ², μ̂, SE, CI, PI, weights), 1e-10 (p), 1e-6 (I²); property tests pass; fixture header pins R package versions and CI regenerates. [03 §6–7]
6. **Grounding**: ≥99% of `auto-accept` text/table values pass the verbatim-quote check; ungrounded rate reported per run; zero values accepted from an ungrounded quote. [05 §4.1]
7. **Independence of routes**: every accepted numeric value has ≥2 candidates from routes differing in model **and** modality/prompt (text vs image; CV vs VLM); single-route values can never be `auto-accept`. [04 §4, §6; 05 §4.2]
8. **Calibration**: `auto-accept` bucket ≥98% within tolerance on the gold set; `needs_human` rate and its precision reported; thresholds stored in config, not code. [05 §4.6]
9. **Arithmetic in code**: no LLM output is used as d, SE, SD-from-SE, CI→SD, or pixel→value; static check that agents' schemas contain no derived-statistic fields. [03 §8; 05 §4.7]
10. **Semantic guards**: `error_type ∈ {SD,SE,CI,IQR,unknown}` and `raw_value_semantics` required on every group value; `unknown` never auto-accepted; sign applied once in code with a verifier cross-check that blocks contradictory rows. [02 §6; 03 §5; 04 §5]
11. **Cost & observability**: per-agent token/cost log (`input, cache_creation, cache_read, output`) and `cache_read_input_tokens>0` asserted for shared-prefix agents; per-paper cost ≤ $3 sync / ≤ $1.5 batched at 2026-08 list prices; a run-level budget cap aborts gracefully. [05 §7; 06 §1, §9]
12. **Determinism & auditability**: raw responses stored; run is resumable per stage by content hash; audit bundle (models, prompts, provenance, votes, verdicts, overrides, discrepancy log) exported per PRISMA-trAIce/RAISE. [04 §4, §6]
13. **Generality**: no code path references Cisneros-specific outcomes/groups; the protocol object alone configures groups, outcomes, directions, timepoints and stats profile; a second toy protocol (synthetic PDFs) must run end-to-end in CI. [context; 03 §8]
14. **Robust ingestion**: scanned/hybrid PDFs are detected and OCR'd (WER ≤10% on gold scans); dedup collapses byte/near duplicates and attaches supplements; multi-experiment papers yield one dataset per `(experiment, outcome)`. [01 §6–7]
15. **Zero-human completion**: with no overrides the pipeline still produces the forest plot, with `needs_human` rows marked and (configurably) excluded or included in a sensitivity analysis; UI overrides are optional and logged. [04 §4]

---

## 11. Evaluation protocol (accuracy vs the human gold standard, without overfitting)

**Assets.** (i) Cisneros 2024 human table: per-study group means/dispersion/n, digitisation source (WPD / text / statistic), d, SE, and the published pooled results for late adaptation and aftereffect (~44 studies). (ii) The PDFs. (iii) An ingestion gold set (§2). (iv) The R oracle. Keep PDFs and the human table **outside the repo** (paths in `validation/`, gitignored).

**Splits (the anti-overfitting rule).**
- **Dev set**: 12 papers chosen to span journals/figure types (vector, raster, box, learning-curve, text-only, t/F-only, ≥1 scan). All prompt tuning, effort sweeps, tolerance/threshold calibration, CV parameter tuning happen **only** here.
- **Held-out set**: the remaining papers, touched **once** per frozen release (`prompt_version`, model IDs, thresholds tagged in git). Any change after looking at held-out results requires a new tag and a note in the run log; report both dev and held-out numbers.
- **Out-of-domain smoke test**: 3–5 papers from an unrelated field (or synthetic PDFs with known ground truth generated by matplotlib/R at vector and raster resolution) to catch protocol-specific overfitting.

**Metrics (per level).**
1. *Ingestion*: figure/table recall & precision (IoU 0.5/0.75), caption linkage, panel recall, table cell F1 on 10 transcribed tables, reading-order correctness on 20 two-column pages, OCR WER, wall time/page. [01 §9]
2. *Value level*: relative error vs axis range for means and error-bar half-lengths; ICC and Bland–Altman (bias, LoA) vs human WPD values; exact-match rate for text/table numbers; SE/SD/CI-type agreement; n agreement; sign agreement. Stratify by route (vector/CV/VLM-coord/read-out), by figure type, by journal.
3. *Record level*: omission rate (human dataset without a Canopy row), spurious-row rate, group/timepoint mis-mapping rate, hallucination rate (value not present in source), verifier refute precision/recall, adjudicator agreement with adjudicated truth, `needs_human` precision.
4. *Analysis level*: Δd per study, Δpooled, ΔCI, ΔPI, Δτ², ΔI² vs (a) the published numbers and (b) the R oracle run on the *human* table; leave-one-out sensitivity of the pooled estimate to Canopy-flagged rows. [03 §6; 04 §6]
5. *Cost/ops*: $ per paper, tokens per agent, cache hit rate, wall time, batch success rate.

**Adjudicated truth, not "human = truth".** Every Canopy-vs-human disagreement beyond tolerance is blind-adjudicated against the PDF by a reviewer who sees both values and the crop/quote but not which is which; classify as {Canopy wrong, human wrong, both defensible, ambiguous source}. Report agreement *and* the human-error rate discovered (up to 63% of human-extracted studies contain ≥1 error in the literature). [04 §3 F13, §4]

**Ablations (dev set only).** Heuristics vs +YOLO vs Docling; crop DPI 300/400/600; single route vs dual route vs +verifier vs +adjudicator (Manalyzer-style ladder); ensemble size 1/3/5; Opus-only vs Opus+Sonnet; effort low/medium/high/xhigh per agent; with/without protocol-compiled enums; with/without digitisation variance in pooling.

**Reporting.** A `validation/report.md` generated by code with per-journal tables, calibration curve (confidence bucket vs observed accuracy), Bland–Altman plots, and the discrepancy log; freeze as `validation/results/<tag>/`. Publish the digitisation ICC/Bland–Altman as the paper's headline (no comparable published number exists). [02 §5; 04 §2]

---

## 12. Implementation order implied by these decisions

1. Data model + provenance record + protocol schema (§9) → 2. ingestion with gold-set tests (§1–2) → 3. stats engine parity (largely done: `canopy/stats/`, `validation/fixtures/`) (§7) → 4. text/table extractors + grounding + code checks (§3, §5) → 5. figure digitiser paths A/B, then C/D + overlay-verify (§4) → 6. voting/verifier/adjudicator + confidence (§6) → 7. batching/caching/cost log (§8) → 8. CLI + FastAPI + review UI + audit bundle (§9) → 9. evaluation protocol run and report (§11).
