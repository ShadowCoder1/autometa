# 05 — Reliability engineering for high-precision numeric extraction with Claude

*Canopy research brief · topic: reliability · written 2026-08-14 · sources: Anthropic platform docs (fetched today), 2025–2026 literature (cited inline)*

## 0. TL;DR

Canopy's extraction layer should be built as a **verify-everything, arithmetic-in-code** pipeline on the Claude Messages API:

1. **One PDF, uploaded once** to the Files API (`file_id`), and rasterized once locally with PyMuPDF at a resolution we control. Claude's own PDF rasterization is fine for *reading* but its page images are produced "at dimensions you don't control, so the returned coordinates can't be reliably mapped back onto the page" — so every figure/table crop that needs coordinates goes through **our** images, pre-resized with Anthropic's published resize algorithm so returned pixel coordinates are 1:1.
2. **Structured outputs everywhere** (`output_config.format` json_schema / `client.messages.parse` + pydantic), with **quote-first schemas** (`quote`, `page`, `location_kind`, then `value`) and a deterministic **verbatim-quote check** against PyMuPDF page text. Citations (`page_location`) run in a *separate* verifier pass because **citations and structured outputs are mutually exclusive (400)** and citations do not cover images.
3. **Heterogeneous parallel extractors** (Opus 5 + Sonnet 5, different prompt variants and different crops) → **numeric agreement voting with tolerance** (median-of-cells for figures) → **adversarial verifier prompted to refute** → **adjudicator (Fable 5 / Opus 5 xhigh)** only on disputes → **deterministic consistency checks in code** → calibrated confidence + human-review flags. Temperature is gone on Opus 4.7+/Sonnet 5/Fable 5, so diversity must come from prompts, crops, models and effort, not sampling knobs.
4. **Cost**: a 40-page paper is ~100–150k tokens as a document block (measure with `count_tokens`); with prompt caching (5-min TTL, 1.25× write / 0.1× read) and 6–8 agents the per-paper cost is roughly **$2–3 on Opus 5 synchronously, ~$1–1.5 via the Batches API (50% off)**; a targeted-context design (page-range excerpts + crops instead of whole-PDF to every agent) lands in the same range while allowing a heterogeneous ensemble. 44 papers ≈ $75–125 sync, roughly half in batch (Sonnet 5 at its now-standard $2/$10).

---

## 1. Ground truth about the platform (what is and is not possible, August 2026)

| Capability | Status / numbers | Source |
|---|---|---|
| PDF input | `document` block (base64 / url / `file_id`); 32 MB request, 600 pages per request on 1M-context models; each page is converted to an image **and** its text extracted; **1,500–3,000 text tokens/page** + image tokens per page; docs' example: ~7,000 tokens for a 3-page PDF with visual processing (~2,300/page all-in) | [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support) |
| Vision resolution | High-res tier (Claude 4.7 and later, i.e. Opus 5 / Sonnet 5 / Fable 5): **max long edge 2576 px, max 4,784 visual tokens/image**; standard tier 1,568 px / 1,568 tokens. Tokens = ⌈w/28⌉×⌈h/28⌉ (28-px patches). 1000×1000 = 1,296 tokens; 1920×1080 = 2,691 tokens on high-res; 4K = 4,784. Up to 600 images/request (100 on 200k models); >20 images/request → each side ≤2000 px | [Vision](https://platform.claude.com/docs/en/build-with-claude/vision) |
| Coordinates / bboxes | "Claude works best with absolute pixel coordinates. Ask for them explicitly … `[x1, y1, x2, y2]`"; **do not ask for normalized 0–1000 coords**; coordinates are relative to the *resized* image (padded to next multiple of 28 on bottom/right, padding does not shift origin); reference `resized_size()` implementation published; "small elements lose precision when an image is downscaled: for fine targets, crop the region of interest and send the crop (offset returned coordinates by the crop origin)"; **PDF pages rasterized server-side cannot be mapped back — rasterize yourself** | [Coordinates and bounding boxes](https://platform.claude.com/docs/en/build-with-claude/vision-coordinates) |
| Citations | `citations: {enabled: true}` per `document` block (all-or-none); PDF → `page_location` {`cited_text`, `document_index`, `start_page_number` (1-indexed), `end_page_number` (exclusive)}; text → `char_location`; custom content → `content_block_location`; works with `file_id`, prompt caching, batches; **"Citations and structured outputs are incompatible" (400)**; **"Only text citations are currently supported. Image citations are not yet possible."**; scanned PDFs without text layer are not citable; `cited_text` does not count toward output tokens | [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) |
| Structured outputs | `output_config.format` json_schema; Python `client.messages.parse(..., output_format=PydanticModel)` → `.parsed_output`; **no numeric constraints (`minimum`/`maximum`) or string length constraints in the API** — the Python/TS SDKs strip them and validate client-side; `additionalProperties: false` required; strict tool use via `strict: true`; first use of a schema pays a compile cost, then 24-h schema cache; works with batches, streaming, thinking; incompatible with citations and prefill | claude-api skill / [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| Sampling / determinism | `temperature`, `top_p`, `top_k` **removed (400)** on Opus 4.7/4.8/5, Sonnet 5, Fable 5; adaptive thinking is on by default on Opus 5 / Sonnet 5 / Fable 5 (Fable: cannot be disabled); `effort` ∈ low/medium/high/xhigh/max; raw chain-of-thought never returned on 5-series | claude-api skill / [Migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide) |
| Prompt caching | Prefix match, render order tools → system → messages; ≤4 breakpoints; min cacheable prefix **512 tokens (Opus 5, Fable 5), 1024 (Sonnet 5)**; write 1.25× (5-min TTL) or 2× (1-h TTL), read 0.1×; `document` blocks are cacheable (docs put `cache_control` on the document block); caches are model-scoped in practice (the caching page does not state this explicitly; a model switch is treated as a full miss); changing `thinking` **or `output_config.effort`** invalidates the messages cache (tools/system caches: model-specific); "a cache entry becomes readable only after the first response begins streaming" → for fan-out, send 1 request, await first token, then fire the rest; 20-block lookback | [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) + skill notes |
| Files API | beta header `files-api-2025-04-14` on upload **and** on every `messages.create` that references the file; `client.beta.files.upload(file=Path(...))` → `id`; reference as `{"type":"document","source":{"type":"file","file_id":...}}` or `{"type":"image",...}`; 500 MB/file, 500 GB/org, files persist until deleted; free to store, billed as input tokens when used | [Files](https://platform.claude.com/docs/en/build-with-claude/files) |
| Batches | 50% off all tokens; ≤100,000 requests or 256 MB per batch; most finish <1 h, hard 24-h expiry; results 29 days; supports PDFs, vision, tools, structured outputs, thinking, prompt caching (stackable) — but cache hits are best-effort ("30% to 98%"); recommend 1-h TTL for batches; `max_tokens: 0` pre-warm not allowed in a batch; no streaming; results in any order (key by `custom_id`) | [Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) |
| Code execution (server tool) | `code_execution_20260120` / `_20260521`; images/PDFs can be uploaded via `container_upload` + `file_id`; container: 1 CPU, 5 GiB RAM, 5 GiB disk, expires 30 days, no internet; **pillow, pypdfium2, pdf2image, pdfplumber, pypdf, numpy, scipy, matplotlib** preinstalled (OpenCV is *not* listed); output files land in `$OUTPUT_DIR` as `file_id`s — "**Claude doesn't see the `content` list**", i.e. generated crops are *not* fed back to the model as images; 1,550 free container-hours/month then $0.05/h (5-minute minimum per execution) | [Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool) |
| Tool results with images | A `tool_result` may contain `text`, `image`, `document`, `search_result` blocks — so a **client-side `crop_image` tool can return a PNG the model then sees** | [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) |
| Vision advice for Opus 5 / Fable 5 | Opus 5 migration guide: "The highest-leverage change is **giving it tools to iteratively analyze, crop, and visually verify its own work** … tool use is a markedly more cost-effective lever than raising thinking alone." Fable 5: "explicitly trained to use bash and crop tools on flipped/blurry/noisy inputs." | claude-api skill, [Migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide) |
| Model prices ($/M in/out) | Opus 5 $5/$25 · Sonnet 5 **$2/$10** (the launch "introductory" price is now the standard price — the planned Sept-2026 rise to $3/$15 was cancelled) · Fable 5 $10/$50 (30-day data retention required; may return `stop_reason: "refusal"` — opt into `fallbacks: "default"`) · Haiku 4.5 $1/$5 (200k context, standard vision tier) | [Pricing](https://platform.claude.com/docs/en/about-claude/pricing) |

**Two hard consequences for Canopy's design.** (a) The "quote-first" grounding cannot rely on the citations API for the same call that returns schema-constrained JSON; grounding must be schema-native (a `quote` field) plus a code-side verbatim check, with citations used in an independent verifier call. (b) Figure crops that Claude produces server-side never come back into its vision — the crop/zoom loop has to be a **client-side tool** (our FastAPI worker crops with PyMuPDF/OpenCV and returns an `image` block in `tool_result`), which is exactly what Anthropic recommends for Opus 5.

---

## 2. Empirical baseline (why verification is not optional)

- A systematic review of 27 LLM data-extraction studies (searched through Dec 2025) found overall accuracy **47%–99.9%**; **numerical data extracted less reliably (47–88%) than categorical/string variables (74–96%)**; omissions were the dominant error type (60–74%) with hallucination rates 0.08–6%; Claude models outperformed GPT in head-to-head event-count extraction (OR 1.70) — [Performance of LLMs in data extraction for evidence synthesis](https://www.sciencedirect.com/science/article/abs/pii/S1532046426001103), [PubMed](https://pubmed.ncbi.nlm.nih.gov/42501879/).
- Chart understanding remains the weak spot: even the best frontier model reached only ~59% on complex stacked bar charts in the 2025 ChartBench evaluation, and models "grasp semantics but fail at precise quantitative extraction" — [ChartBench (2025)](https://dl.acm.org/doi/10.1145/3772128.3772169); see also [Making Multimodal LLMs Reliable Chart Data Extractors (CHI 2026)](https://dl.acm.org/doi/10.1145/3772318.3790721).
- **Self-ensembling works for chart digitization**: repeatedly sampling tables from the same VLM and taking **per-cell medians**, with convergence detection and dispersion-based uncertainty, gave up to **23% relative improvement** on WB-ChartExtract — [Berkane, Wang & Majumder, arXiv 2605.27298 (May 2026)](https://arxiv.org/abs/2605.27298). This is the template for Canopy's figure-value voting.

Takeaways: numbers from **text** are near-solved with quote checks; **tables** need layout-aware extraction (row/column header capture); **figures** need multi-sample median aggregation, crop/zoom, axis calibration and a CV cross-check.

---

## 3. Recommended architecture

### 3.1 Ingestion (deterministic, once per paper)

1. `PyMuPDF` → per-page text with word bboxes (`page.get_text("words")`), tables (`page.find_tables()`), and image/drawing bboxes.
2. Rasterize each page at 200–300 DPI to PNG; store; also pre-compute the exact size Claude will see with the published `resized_size(w, h, max_edge=2576, max_tokens=4784)` and store the resized PNG (so coordinates are 1:1 and no rescale math is needed).
3. Upload the PDF once to the Files API → `file_id`; upload the resized page PNGs / figure crops as needed (or send base64 for small crops).
4. `client.messages.count_tokens` on the document block for each model to record the real per-paper token cost.

### 3.2 Agent roster (roles, models, effort)

| # | Agent | Model / effort | Input | Output (schema) | Notes |
|---|---|---|---|---|---|
| A0 | **Locator / triage** | Sonnet 5, effort `medium` | whole PDF (`file_id`, cached) + protocol | page map: where groups, N, demographics, outcome definitions, candidate tables/figures (page, panel, caption, why relevant), design flags | Only agent that reads the whole PDF every time; cheap; output drives targeted context for the rest |
| A1 | **Text/table extractor – variant "table-first"** | Opus 5, `high` | page-range excerpt (page images from A0's map + PyMuPDF text) | per-outcome record: group, n, statistic type (M±SD/M±SE/CI/t/F/median-IQR), values, units, **quote, page, kind ∈ {sentence, table_cell, figure_caption}**, table id / row / column headers | structured output; quote verified verbatim in code |
| A2 | **Text/table extractor – variant "narrative-first"** | Sonnet 5, `high` | same pages, different instructions/field order, images at different DPI | same schema | heterogeneous second opinion |
| A3 | **Figure digitizer – model-driven** | Opus 5, `high`, tools: `crop_image`, `zoom`, `measure_axis` | our resized figure PNG (+ caption text) | axis calibration (pixel↔value pairs, log/linear), per bar/point/error-cap **pixel coords** `[x,y]` and derived values, error-bar type (SD/SE/CI/unknown), n per group if printed | client-side crop tool returns image blocks; ask for absolute pixel coordinates; values computed **in code** from pixels + calibration |
| A4 | **Figure digitizer – CV-assisted** | Sonnet 5, `high` (or Opus 5 with a different crop set) | same figure + OpenCV-detected axis ticks/labels supplied as text; independent crops | same schema | independent second sample; feed the median-of-cells aggregator (§4.3) |
| A5 | **Adversarial verifier** | Sonnet 5 `xhigh` (or Opus 5 `high`) — must be a *different* model than the extractor it checks | candidate record + only the source page(s), citations enabled | verdict ∈ {confirmed, refuted, ambiguous}, `refutation_reason`, alternative value, `page_location` citation | prompt: "your job is to prove this value wrong"; not structured-output (citations on) — parse a small XML/JSON envelope |
| A6 | **Adjudicator** | Fable 5 `high` (fallback Opus 5 `xhigh`) | all candidates + verifier verdicts + crops + protocol rules | final value, chosen source, rationale, `needs_human` | invoked only when voting/verification disagree; keep the context targeted (≈10–30k tokens) to bound Fable cost |
| C1 | **Consistency checker** | **code, no LLM** | all records | pass/fail flags per rule (§4.4) | runs before and after adjudication |
| C2 | **Statistics converter** | **code** (numpy/scipy) | verified raw stats | d, SE(d), transformations (SE→SD by √n, t/F→d, CI→SD) | never let the model compute effect sizes |

Effort guidance follows Anthropic's Opus 5 notes: `high` default, `xhigh` for the hardest verification/adjudication, `low`/`medium` unusually strong on Opus 5/Sonnet 5 for triage; sweep effort on the Cisneros gold set before fixing it. Keep `thinking` **and `effort`** configuration identical across agents that share a cached document (changing either invalidates the messages cache) — so A2/A4 (`high`) and A5 (`xhigh`) on Sonnet 5 will not share a messages-cache entry unless effort is equalised or the excerpt is cached only through the system/document prefix.

### 3.3 Context strategy: whole-PDF vs targeted excerpts

- **Whole PDF to every agent** is simplest and caches well *within one model* (one write, N reads) but forces all agents onto one model to share the cache and costs ~120k tokens per read.
- **Targeted context** (recommended): A0 reads the whole PDF; A1–A5 receive only the relevant page images + text (typically 4–10 pages ≈ 15–40k tokens) plus figure crops. Cheaper per agent, allows heterogeneous models, and — critically — puts *our* rasterized images in front of the model so coordinates are usable. Cache the shared excerpt with one `cache_control` breakpoint on the last shared block, and place agent-specific instructions **after** the breakpoint (in the user turn; on Opus 5 / Fable 5 you may instead append a `{"role":"system"}` message — not supported on Sonnet 5).

---

## 4. Verification patterns

### 4.1 Quote-first, value-second (schema-native grounding)

Schema field order matters because JSON is generated left-to-right: put `quote`/`page`/`location` before `value`. Then in code:

- Normalise whitespace/ligatures/minus signs; check the quote is a verbatim substring of the PyMuPDF text of the stated page (fallback: fuzzy ratio ≥ 0.95 for hyphenation/line-break artefacts); if it fails → mark `ungrounded` and route to A5.
- For table cells: require `row_header` and `column_header` strings and check both occur on the page; check that the numeric string in `value_as_written` occurs inside the quote.
- For figures: `quote` is the caption/legend text; the grounding artefact is the **crop + pixel coordinates**, stored as provenance (page, bbox in original-page pixels after un-offsetting the crop origin).

### 4.2 Independent parallel extractors and heterogeneous ensembles

Because `temperature` no longer exists on the 5-series and Opus 4.7+, self-consistency comes from **prompt variants** (field order, "table-first" vs "narrative-first", asking for all groups vs one group per call), **input variants** (different DPI/crop windows, with vs without PyMuPDF text), **model variants** (Opus 5 vs Sonnet 5 vs Fable 5), and **effort variants**. Repeated identical calls still vary somewhat (the API never guaranteed determinism), but do not rely on that alone. Three to five samples per figure value is the sweet spot from the self-ensembling paper; stop early when the aggregated cell stabilises.

### 4.3 Numeric agreement voting with tolerance

- Text/table values: agreement = exact after unit normalisation (allow ±1 ulp of the printed precision, e.g. 12.3 vs 12.30). Two heterogeneous agents agree → **accepted-by-vote**; disagreement → A5 + A6.
- Figure values: align candidates by (group, condition, panel), take **per-cell median**; report dispersion (MAD) as uncertainty; tolerance = max(2% of axis range, 0.5 × smallest tick interval). If MAD > tolerance → more samples (up to 5) then adjudicate; also cross-check with the CV pixel measurement (bar-top / error-cap y-pixel via OpenCV line detection) — the CV number is another voter, not ground truth.
- Store every candidate (model, prompt variant, raw JSON, tokens) — the vote is auditable.

### 4.4 Deterministic consistency checks (code, run on every record)

| Rule | Check |
|---|---|
| Sample sizes | Σ per-group n = reported total N (± dropouts stated); n ≥ 2; integer |
| Dispersion | SD > 0; SE > 0; SE·√n ≈ SD when both reported; heuristic flag "reported SD looks like SE" if SD/mean is implausibly small for the outcome family and SD·√n matches a nearby number |
| Intervals | CI symmetric about the mean (|(hi+lo)/2 − M| < 0.5% of range) unless log-scale/skewed outcome; CI half-width ≈ 1.96·SE (or t-crit for small n) |
| Axis plausibility | figure-derived value within axis range; ordering of bars matches ordering of any values mentioned in text |
| Direction/sign | sign orientation rule from the protocol applied in code (e.g. "+ = enhanced adaptation in older adults"); flag when text says "older adults showed less…" but computed d > 0 |
| Effect-size sanity | |d| ≤ 3 by default (flag), SE(d) computed only from n and d via SE(d)=√((n1+n2)/(n1·n2)+d²/(2(n1+n2−2))) |
| Statistic conversion | t/F→d only with df and group sizes; F must be 1-df between-groups |
| Provenance | quote grounded (4.1); page within document; figure bbox within page |
| Duplicates | same numbers appearing for two different outcomes → flag copy-paste |

Any failed rule downgrades confidence and triggers A5/A6 or human review; nothing is silently "fixed" by an LLM.

### 4.5 Adversarial verifier and tie-break

The verifier gets *only* the candidate and the source page(s) with `citations` enabled and is instructed to find a reason the value is wrong (wrong group, wrong time point, SE vs SD, baseline vs post, per-trial vs per-block units, subgroup, different figure panel). Use a **different model** than the extractor (cross-family disagreement is more informative). The adjudicator sees everything and must output a decision plus `needs_human: true` when evidence is genuinely ambiguous (e.g. error-bar type unknown and no text disambiguates). Anthropic's own advice for review-style tasks: instruct the finder to report every doubt with confidence and severity, and filter downstream — do not ask it to "only report high-severity" issues.

### 4.6 Confidence calibration and human-review flags

Do not ask the model for a bare probability. Compute a score in code from: source kind (text 1.0 > table 0.9 > figure 0.7), grounding check, number of agreeing heterogeneous samples, verifier verdict, dispersion of figure samples, consistency-rule results. Then bucket: `auto-accept` (all green, ≥2 agree), `accept-with-note`, `needs_human` (any refutation, ungrounded quote, MAD > tolerance, unknown error-bar type, N mismatch). Calibrate thresholds on the Cisneros gold set (human WebPlotDigitizer values) — report the fraction of auto-accepted values within tolerance; target ≥ 98% before trusting auto-accept.

### 4.7 Arithmetic in code, always

The model reports what the paper says (numbers as written, units, error-bar type, n). Effect sizes, SE→SD conversions, CI→SD, t/F→d, sign orientation, and pixel→value calibration are computed in Python (numpy/scipy) with unit tests. The stats engine's own tests should include the Cisneros formulas as fixtures.

---

## 5. Vision specifics for figures

- Rasterize the figure region yourself; pre-resize with `resized_size()` (high-res tier: 2576 px / 4,784 tokens). A typical half-page panel at ~1200×900 px costs ⌈1200/28⌉×⌈900/28⌉ = 43×33 = 1,419 tokens; a full 2576×1449 page ≈ 4,784.
- Prompt for `[x1, y1, x2, y2]` / `[x, y]` **pixel coordinates**, request JSON via structured outputs; never ask for normalized coordinates. Coordinates apply to the image sent; offset crop coordinates by the crop origin. Ask separately for (i) axis calibration points (two tick labels with their pixel y), (ii) bar top / marker centre pixels, (iii) error-bar cap pixels — then compute values in code.
- Give the model **client-side tools**: `crop_image(page_or_figure_id, bbox, zoom)` returning an `image` block, `list_regions()` returning the OpenCV-detected axes/ticks/legend bboxes, and (optionally) `overlay_points(coords)` returning an image with the model's own points drawn so it can visually verify (Opus 5 guidance: analyze → crop → verify). Reference: `tool_result.content` supports `image` blocks.
- Do **not** route crops through the server-side code_execution tool for viewing — generated files come back as `file_id`s that "Claude doesn't see". Code execution remains useful for server-side pandas/pillow work if we ever want it, but Canopy already has a local Python worker, so keep everything client-side.
- Small elements lose precision when downscaled: crop tightly around the panel, keep axis labels in the crop, and send one panel per call for dense figures.

---

## 6. Prompt skeletons

**Shared system prompt (identical across agents that share a cache; keep it frozen):**
```
You are an extraction component inside an automated meta-analysis pipeline. You never compute effect sizes
or convert statistics; you report exactly what the source shows, with a verbatim quote and location for
every number, and you say "not_reported" rather than guessing. Coordinates are absolute pixels of the image
you were shown. Ambiguity (SD vs SE, baseline vs post, per-trial vs per-block) must be flagged, not resolved.
```

**A1/A2 extractor (user turn, after the cached excerpt):**
```
Protocol: <groups, outcomes, timepoints, inclusion rules>.
Task: extract every reported statistic for the outcomes above, one record per (group × outcome × timepoint).
For each record give, in this order: page, location_kind, quote (verbatim), table_id/row_header/column_header
(if table), statistic_kind (mean_sd|mean_se|mean_ci|median_iqr|t|F|other), n, values_as_written, unit,
error_type_confidence (high|medium|low), notes. If a value is only shown in a figure, emit a record with
location_kind="figure" and leave values null — do not eyeball figures here.
```
(Variant B: ask "narrative first — list every sentence in Results that mentions the outcomes, then map sentences to records"; and permute the schema field order.)

**A5 adversarial verifier (citations enabled, no structured output):**
```
A previous extractor claims: <record JSON>. Using only the attached page(s), try to REFUTE it: wrong group,
wrong timepoint, SE reported as SD (or vice versa), wrong unit, wrong table row/column, subgroup, or the
number does not appear at all. Cite the exact supporting or contradicting sentence(s). Answer with
<verdict>confirmed|refuted|ambiguous</verdict><reason>…</reason><alternative>…</alternative>.
```

**A3 figure digitizer (tools available):**
```
Figure <id> (page p, panel q). Step 1: identify axes; return two calibration ticks per numeric axis as
{label_value, pixel_x_or_y}. Step 2: for each bar/point matching the protocol groups, return the pixel
[x, y] of the top edge/marker centre and of each error-bar cap. Use crop_image to zoom on caps and tick
labels before answering; call overlay_points to check your points sit on the marks. Say which error-bar
type the caption/legend states; if not stated, error_type="unknown". Do not convert pixels to values.
```

**A6 adjudicator (Fable 5):** present candidates + verifier verdicts + crops; require `decision`, `chosen_source`, `rationale`, `needs_human`, `what_would_resolve_it`.

---

## 7. Cost model

**Unit costs (Anthropic list prices):** Opus 5 $5/$25 per M in/out; Sonnet 5 $2/$10 (now permanent — see verification notes); Fable 5 $10/$50. Cache write 1.25× (5-min) or 2× (1-h); cache read 0.1×. Batches −50% on everything.

**Per-page/image tokens:** PDF page ≈ 1,500–3,000 text tokens + page-image tokens (docs' worked example ≈ 2,300/page all-in); our own high-res page image ≤ 4,784 tokens; a typical figure-panel crop 1,000–2,000 tokens. Assume a 40-page paper document block ≈ **120k tokens** (measure per paper with `count_tokens`; range 100–200k).

| Design | Per-paper cost (sync) | With Batches |
|---|---|---|
| **Whole PDF to 8 Opus 5 agents, one 5-min cache** — 1 write 120k×$6.25/M = $0.75; 7 reads 7×120k×$0.50/M = $0.42; instructions 8×2k×$5/M = $0.08; output+thinking 8×8k×$25/M = $1.60 | **≈ $2.85** | ≈ $1.45 |
| **Targeted (recommended)** — A0 Sonnet 5 whole PDF ($0.24 + $0.06 out) ; A1/A3 Opus 5 2×(30k in + 8k out) = $0.70; A2/A4/A5×2 Sonnet 5 4×(30k in + 6k out) = $0.48; A6 Fable 5 on ~30% of papers (20k in + 10k out = $0.70 when used) | **≈ $1.7–2.3** | ≈ $0.9–1.2 |
| Heterogeneous whole-PDF (extra cache write per model: Sonnet +$0.30, Fable +$1.50) | ≈ $4–5 | ≈ $2–2.5 |

**44-paper validation run:** ≈ $75–100 sync on the recommended design (≈ $40–55 in batches) at the current Sonnet 5 $2/$10 price. Thinking tokens are billed as output and dominate; use `effort: medium` for A0 and sweep effort for A1–A5. Add a per-agent token log (`usage.input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`) from day one — without accounting none of this can be tuned.

**Caching mechanics to get right:** identical `tools`, `system`, `thinking` and `effort` config across agents sharing a prefix; document/excerpt block first with `cache_control` on the last shared block; agent-specific text after; fan-out only after the first response starts streaming (or send one sync warm-up then batch the rest with 1-h TTL); one write per model in a heterogeneous ensemble; verify `cache_read_input_tokens > 0` in tests.

**Batches vs sync:** Canopy's per-paper extraction is not latency-sensitive → run A1–A5 in batches (per paper or per corpus), A0 sync (needed to build the excerpts), A6 sync (few calls, interactive UI can show progress). Note that batch cache hits are best-effort and `max_tokens:0` warming is disallowed inside batches.

---

## 8. Concrete implementation notes (Python, `anthropic` SDK ≥ 0.122)

- `client.beta.files.upload(file=Path(pdf))` → `file_id`; document block `{"type":"document","source":{"type":"file","file_id":fid},"citations":{"enabled":True}}` for verifier calls; without citations for structured-output calls; header `betas=["files-api-2025-04-14"]` on both.
- Extraction: `client.messages.parse(model="claude-opus-5", output_format=ExtractionRecordList, thinking={"type":"adaptive"}, output_config={"effort":"high"}, messages=[...])` → `.parsed_output`; keep pydantic validators for ranges (SDK strips `minimum`/`maximum` before sending and validates client-side).
- Figure tools: `@beta_tool def crop_image(figure_id: str, x1:int,y1:int,x2:int,y2:int, zoom:float=2.0)` returning `[{"type":"image","source":{"type":"base64",...}}, {"type":"text","text":"crop origin (x1,y1); crop size WxH"}]`; drive with `client.beta.messages.tool_runner` and record every tool call for provenance.
- Handle `stop_reason == "refusal"` on Fable 5/Opus 5 (rare for this content) and enable `fallbacks: "default"` (`server-side-fallback-2026-07-01`) on Fable calls; check the org has 30-day retention before using Fable.
- Stream anything with `max_tokens` above ~16k; extractors typically need 4–16k.
- Token counting: `client.messages.count_tokens(model=..., messages=[{role:user, content:[document block]}])` per model, because tokenizers differ (Sonnet 5 uses the new tokenizer; Opus 5/Fable 5 share the Opus 4.7 tokenizer).

---

## 9. Risks and open questions

- **Server-side PDF page images** cost tokens we don't control and their coordinates are unusable; the pipeline should measure whether whole-PDF reads on high-res models blow past ~150k tokens/paper and, if so, feed only extracted text + our own crops (a `document` block with `source.type: "text"` for the text layer keeps citations working).
- **Citations vs structured outputs** exclusivity means grounding is two-pass; the verbatim-quote check must be robust to PDF text-layer quirks (ligatures, hyphenation, column order) — test on Cisneros PDFs.
- **Batch cache hit rate** (30–98%) can quietly double input cost; monitor `cache_read_input_tokens` and fall back to sync fan-out with a warm-up call if hits are poor.
- **Effort/thinking cost** is the dominant term and hard to predict; effort sweeps on the gold set are required.
- **Figure error bars**: even with median-of-samples, SD-vs-SE ambiguity is a semantic problem the vote cannot fix — needs the verifier + human flag path.
- **Model drift**: prompts and tolerances must be re-baselined at each model release; log model IDs with every record.

## Sources

- Anthropic — PDF support: https://platform.claude.com/docs/en/build-with-claude/pdf-support
- Anthropic — Vision (limits, token formula, resolution tiers): https://platform.claude.com/docs/en/build-with-claude/vision
- Anthropic — Coordinates and bounding boxes (resize/pad algorithm, pixel-coordinate prompting): https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
- Anthropic — Citations (page_location, incompatibility with structured outputs, no image citations): https://platform.claude.com/docs/en/build-with-claude/citations
- Anthropic — Structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Anthropic — Prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic — Files API: https://platform.claude.com/docs/en/build-with-claude/files
- Anthropic — Batch processing: https://platform.claude.com/docs/en/build-with-claude/batch-processing
- Anthropic — Code execution tool: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
- Anthropic — Handle tool calls (image blocks in tool_result): https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
- Anthropic — Define tools (descriptions, input_examples, tool_choice): https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
- Anthropic — Model migration guide (Opus 5 vision/crop-tool advice, sampling removal, effort): https://platform.claude.com/docs/en/about-claude/models/migration-guide
- Anthropic — Pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Anthropic — Token counting: https://platform.claude.com/docs/en/build-with-claude/token-counting
- Performance of large language models in data extraction for evidence synthesis: a systematic review (2026): https://www.sciencedirect.com/science/article/abs/pii/S1532046426001103 · https://pubmed.ncbi.nlm.nih.gov/42501879/
- Berkane, Wang, Majumder — Self-Ensembling Vision-Language Models for Chart Data Extraction (arXiv 2605.27298, May 2026): https://arxiv.org/abs/2605.27298
- ChartBench (2025): https://dl.acm.org/doi/10.1145/3772128.3772169
- Making Multimodal LLMs Reliable Chart Data Extractors (CHI 2026): https://dl.acm.org/doi/10.1145/3772318.3790721
- Collaborative LLMs for automated data extraction in living systematic reviews: https://pubmed.ncbi.nlm.nih.gov/39836495/

## Verification notes (fact-check)

Fact-check performed 2026-08-15 against primary sources (Anthropic platform docs fetched live; PubMed/arXiv/ACM records). Claims edited in place are marked **corrected**.

| # | Claim checked | Verdict | Source |
|---|---|---|---|
| 1 | PDF input: 32 MB request; 600 pages/request (100 when context < 1M); each page processed as image + extracted text; 1,500–3,000 text tokens/page; ~7,000 tokens for a 3-page PDF with visual processing | **confirmed** | https://platform.claude.com/docs/en/build-with-claude/pdf-support |
| 2 | Vision: high-res tier = Claude 4.7 and later, 2576 px long edge / 4,784 visual tokens; standard 1,568 px / 1,568 tokens; ⌈w/28⌉×⌈h/28⌉; 1000×1000 = 1,296; 1920×1080 = 2,691 (high-res); 3840×2160 → 2576×1449 = 4,784; 600 images/request (100 on 200k models); >20 images → resize so neither side exceeds 2000 px | **confirmed** | https://platform.claude.com/docs/en/build-with-claude/vision |
| 3 | Coordinates: ask for absolute pixel coordinates, not 0–1000 normalized; padding to next multiple of 28 on bottom/right, origin unchanged; `resized_size()` reference implementation; crop for fine targets and offset by crop origin; server-side PDF rasterization "at dimensions you don't control" so coordinates cannot be mapped back | **confirmed** (verbatim quotes match) | https://platform.claude.com/docs/en/build-with-claude/vision-coordinates |
| 4 | Citations: PDF `page_location` page numbers 1-indexed with exclusive end; citations + `output_config.format` → 400; "Only text citations are currently supported. Image citations are not yet possible."; scanned PDFs without extractable text not citable; `cited_text` not counted as output tokens; works with prompt caching, batches, `file_id` | **confirmed** | https://platform.claude.com/docs/en/build-with-claude/citations |
| 5 | Prompt caching: min cacheable prefix 512 (Opus 5 / Fable 5 / Mythos 5), 1,024 (Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5), 4,096 (Haiku 4.5); write 1.25× (5 m) / 2× (1 h), read 0.1×; ≤4 breakpoints; 20-block lookback; entry available only after first response begins | **confirmed** | https://platform.claude.com/docs/en/build-with-claude/prompt-caching |
| 5b | "Toggling thinking invalidates the messages cache (not tools/system)" | **corrected** → the docs say thinking *and* `output_config.effort` changes always invalidate message blocks, with a *model-specific* effect on tools/system caches. §1, §3.2 and §7 now say to hold both `thinking` and `effort` constant across agents sharing a prefix | https://platform.claude.com/docs/en/build-with-claude/prompt-caching |
| 5c | "Caches are model-scoped" | **unverifiable in the docs page** (no explicit statement found); retained as "in practice" — a model switch is a full miss per the claude-api skill's invalidation table | https://platform.claude.com/docs/en/build-with-claude/prompt-caching |
| 6 | Files API: beta header `files-api-2025-04-14` required on upload and on Messages requests referencing the file; 500 MB per file; storage per org; files persist until deleted; file operations free, content billed as input tokens | **corrected** → total storage is **500 GB per organization**, not 100 GB (500 MB/file confirmed) | https://platform.claude.com/docs/en/build-with-claude/files |
| 7 | Batches: 50% discount; ≤100,000 requests or 256 MB; most <1 h, expire at 24 h; results 29 days; cache hits best-effort "30% to 98%"; 1-h TTL recommended; `max_tokens: 0` not allowed in a batch | **confirmed** | https://platform.claude.com/docs/en/build-with-claude/batch-processing |
| 8 | Code execution: 1 CPU / 5 GiB RAM / 5 GiB disk, 30-day expiry, no internet; pillow, pypdf, pdfplumber, pypdfium2, pdf2image, numpy, scipy, matplotlib preinstalled (OpenCV not listed); `$OUTPUT_DIR` files returned as `file_id`s and "Claude doesn't see the `content` list"; 1,550 free hours/month then $0.05/h | **confirmed** (added: execution time has a 5-minute minimum) | https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool ; https://platform.claude.com/docs/en/about-claude/pricing |
| 9 | `tool_result.content` may contain `text`, `image`, `document`, `search_result` blocks (so a client-side crop tool can return an image the model sees) | **confirmed** | https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls |
| 10 | Prices: Opus 5 $5/$25; Fable 5 $10/$50; Haiku 4.5 $1/$5; Sonnet 5 "$3/$15 (intro $2/$10 through 2026-08-31)" | **corrected** → Sonnet 5 is **$2/$10** as the standard price ("the previously scheduled increase to $3/$15 … on September 1, 2026 will not occur"). Cost model in §7 recomputed (recommended design ≈ $1.7–2.3/paper sync, 44 papers ≈ $75–100). The pricing URL was also wrong (`/docs/en/pricing` 404s) → `/docs/en/about-claude/pricing` | https://platform.claude.com/docs/en/about-claude/pricing |
| 11 | Systematic review: 27 studies, searched through Dec 2025, accuracy 47%–99.9%, numerical 47–88% vs categorical/string 74–96%, omissions 60–74%, hallucination 0.08–6%, Claude vs GPT OR 1.70 for event counts | **confirmed** (Shankar, Lim & Qian, J Biomed Inform 2026;181:105086, PMID 42501879) | https://pubmed.ncbi.nlm.nih.gov/42501879/ |
| 12 | Berkane, Wang & Majumder, arXiv 2605.27298 (May 2026): per-cell median self-ensembling, convergence detection, dispersion-based uncertainty, up to 23% relative improvement on WB-ChartExtract | **confirmed** (submitted 2026-05-26) | https://arxiv.org/abs/2605.27298 |
| 13 | ChartBench (VRISP 2025, ACM DOI 10.1145/3772128.3772169): best model ~59% on stacked bar charts | **confirmed** (59.2% for the top model, GPT-5, on stacked bar charts; ACM full text is paywalled — verified via search snippets of the paper) | https://dl.acm.org/doi/10.1145/3772128.3772169 |
| 14 | Sampling params removed (400) on Opus 4.7+/Sonnet 5/Fable 5; adaptive thinking default on Opus 5/Sonnet 5/Fable 5; effort levels low…max; structured-output constraints (no `minimum`/`maximum`, 24-h schema cache); tokenizer note (Opus 5/Fable 5 share Opus 4.7 tokenizer, Sonnet 5 new tokenizer) | **confirmed against the bundled claude-api skill / migration guide** (not re-fetched separately) | https://platform.claude.com/docs/en/about-claude/models/migration-guide |

