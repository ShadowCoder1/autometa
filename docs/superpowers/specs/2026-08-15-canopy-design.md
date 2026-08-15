# Canopy — agent-verified automated meta-analysis from a folder of PDFs

*Design spec, 2026-08-15. Status: v1 (reviewed by an agent panel; see `docs/superpowers/specs/2026-08-15-design-review.md`).*

## 1. Goal

Upload a folder of research-paper PDFs plus a short **protocol** (what to compare, which outcomes) → Canopy
identifies eligible datasets, extracts the group statistics from **text, tables and figures**, has independent
agents **verify every number**, computes standardized mean differences with CIs in code, runs a random-effects
meta-analysis and returns a publication-grade **forest plot with full provenance** (which page/figure/quote each
number came from, how it was converted, who verified it, and what still needs a human).

First validation case: Cisneros et al. (2024) aging × sensorimotor adaptation (late adaptation, aftereffect).
The tool must be **general** — nothing in the pipeline may be specific to that study; the protocol carries all
domain content.

### Non-goals (v1)
* Literature search / retrieval (future "forward" mode).
* Non-SMD effect sizes (odds ratios, correlations) — the architecture leaves room, but v1 = SMD (Cohen's d /
  Hedges' g) between two independent groups.
* Multi-arm network meta-analysis, meta-regression UI (moderators are recorded and exported; regression is a
  script-level feature).

## 2. Concepts & data model (pydantic, `canopy/models.py`)

| Object | Meaning |
|---|---|
| `Protocol` | User input: research question; **group A** / **group B** definitions (labels + synonyms + rules, e.g. "older adults ≥ 60 y" vs "younger adults 18–35 y"); list of **outcomes** (name, definition, operationalization rules, `higher_is_better` guidance, what counts as the measurement window); eligibility criteria; dataset rules (multiple experiments → separate datasets; same participants → first exposure only); statistics settings (estimator cohen/hedges, variance convention, tau² method, HK, PI method, sign convention text); moderators to record (e.g. perturbation size, #targets, task type). Stored as YAML/JSON; editable in the UI. |
| `Paper` | One unique PDF (sha256-deduplicated; DOI/title near-duplicate check). Ingestion artefacts: per-page text + words with bboxes, tables, drawings, images, page rasters (300 DPI PNG + Claude-resized copy), figure regions with captions, Files-API `file_id`. |
| `StudyMap` | Output of the **mapper** agent: citation, eligibility verdict + rationale, task/design descriptors, list of `DatasetSpec`s (label, experiment/condition, group A/B labels + n + age + evidence quotes, moderators), and for each (dataset, outcome) a list of **`Source`s** (kind ∈ text_mean_sd / text_mean_se / text_mean_ci / table / figure_bar / figure_line / figure_points / figure_box / test_statistic / reported_effect_size / author_data; page; locator; quote; error-bar type; values if in text). |
| `Candidate` | One extractor's answer for one (dataset, outcome, group) quantity set: `{n, mean, dispersion_value, dispersion_type ∈ SD/SE/CI95/IQR/…, unit, page, kind, quote/crop, pixel provenance, model, prompt_variant, raw_json}` — or a test-statistic candidate `{stat_type t/F/p, value, df, direction}` — or a reported-d candidate. Never contains computed effect sizes. |
| `Verdict` | Result of verification for a quantity: agreement status, verifier verdict (confirmed/refuted/ambiguous + reason + alt value), adjudication (if any), consistency-check flags, **confidence bucket** (auto_accept / accept_with_note / needs_human), final resolved values. |
| `EffectSize` | Computed in code from resolved values: route (means_sd / t_stat / f_stat / p_value / reported_d), d, g, es used, var, se, CI, orientation applied, digitization variance (optional), the exact conversion chain as text. |
| `MetaResult` | Pooled RE result (+ FE), heterogeneity, PI, weights, Egger; per-outcome. |
| `RunManifest` | Everything about a run: protocol hash, papers, per-paper status, costs, timings, model versions, prompt versions, cache hits, warnings; makes runs reproducible/resumable. |

Every LLM call is a `LLMCall` record (request hash → response JSON, tokens, cost, latency, model) persisted in a
content-addressed cache (`runs/<run>/cache/`), so re-runs are free/idempotent and every number is auditable.

## 3. Pipeline (per outcome the run fans out per paper; stages are resumable)

```
folder ──► 0 Ingest (dedupe, parse, raster, figure regions, upload)     [code]
        ─► 1 Map   (whole PDF + protocol → StudyMap)                    [Opus 5]  ×1 (+ Sonnet 5 cross-check of eligibility & N)
        ─► 2 Extract per (dataset × outcome):
              2a Text/table extractors ×2 (heterogeneous)               [Opus 5 "table-first", Sonnet 5 "narrative-first"]
              2b Figure digitizer (if any figure source):
                   A vector-exact (PDF drawings)  →  C VLM coords + CV snap  →  D VLM read-out ensemble (×3–5)
                   + overlay-verify                                     [Opus 5 primary; Fable 5 optional member]
              2c Test-statistic / reported-d extractor (when relevant)  [Opus 5]
        ─► 3 Verify: consistency checks [code] → agreement vote [code] → adversarial verifier [Sonnet 5/Opus 5,
              different model than extractor] → adjudicator on disputes [Fable 5 or Opus 5 xhigh] → confidence [code]
        ─► 4 Effect sizes [code]  (route precedence + conversions + orientation)
        ─► 5 Pool + plot [code]  (RE model, forest plot, heterogeneity, PI, Egger; extraction table; provenance)
        ─► 6 Report [code + one Opus 5 call for the narrative "methods & flags" summary]
```

### 3.0 Ingestion (`canopy/ingest/`)
* Dedupe by sha256; then by normalized title/DOI (Levenshtein ≥ 0.95) → group duplicates, keep one, record aliases.
* PyMuPDF: text per page (`get_text("dict")` for spans with bboxes; reading order via blocks sorted by column),
  words, `find_tables()`, `get_drawings()`, `get_images()` + placement rects, links (DOI).
* Page rasters at 300 DPI (PNG) and a **Claude-resized** copy computed with the published resize rule
  (long edge ≤ 2576 px, ≤ 4784 visual tokens ⇒ w·h/784 ≤ 4784) so coordinates returned by the model are 1:1
  with what we hold.
* Figure regions: union of (a) placed raster images, (b) clusters of vector drawings, (c) caption blocks
  ("Fig./Figure N"), matched by proximity; each region rendered as a **high-res crop** (600 DPI for vector,
  native×upscale for raster) with a margin; panels detected later by the VLM on demand.
* Files API upload of the PDF (beta) → `file_id` (with base64 fallback); `count_tokens` recorded.
* Scanned PDFs (no text layer): OCR fallback via tesseract to give quote-grounding a text layer; flagged.

### 3.1 Mapper (`canopy/agents/mapper.py`) — Opus 5, effort high, whole PDF (cached document block)
Structured output = `StudyMap` (schema as in the spike, generalized: outcomes come from the protocol).
Rules baked into the prompt: report **analyzed** N (post-exclusion) and quote it; enumerate all datasets;
list every source location for each outcome; classify error-bar type from caption/legend/methods; note design
traps (same participants across conditions, contextual change between learning and aftereffect, transfer
conditions). A second cheaper model (Sonnet 5, medium) independently answers *eligibility + datasets + N* only;
disagreement → adjudicator (Opus 5 xhigh) with both maps.

### 3.2 Extraction (`canopy/agents/extract_text.py`, `canopy/digitize/`)
* **Text/table extractors** get *targeted context*: the pages listed by the mapper (page images + PyMuPDF text
  for those pages), the protocol, the dataset/outcome definition, and a **quote-first schema** (`quote`, `page`,
  `kind`, `row/col headers` … then `value_as_written`, parsed numbers, unit, dispersion type). Two heterogeneous
  variants (model × prompt order). Grounding check in code: normalized quote must be a substring (or fuzzy ≥ 0.95)
  of that page's text; numeric string must occur in the quote.
* **Figure digitizer** (`canopy/digitize/`), invoked per figure source, given: crop PNG(s), caption/legend text,
  the mapper's description of which panel/series/x-position is the target, and the group labels.
  * Path A — **vector-exact**: if the region contains vector drawings: extract tick-label spans (text + bbox) and
    tick/axis lines → least-squares axis calibration; extract candidate marks (small filled paths, rects) and
    vertical lines with caps; VLM (Opus 5) labels which mark = which group/x-position by returning approximate
    pixel coords; snap to nearest primitive; values from geometry (error < 0.5% of range).
  * Path C — **VLM coordinates + CV snap**: Opus 5 returns pixel coords for tick labels/marks/cap ends (structured);
    CV refines within ±6 px (edge/cap detection along the column); LS axis fit with residuals; ensemble ×3–5 over
    crops/zooms; per-cell median + MAD.
  * Path D — **VLM value read-out**: 2 prompts × (Opus 5, optional Fable 5/Sonnet 5): direct values with the model's
    own calibration reasoning; median; cross-check against A/C.
  * **Overlay-verify**: draw the resolved marks/caps on the crop; a fresh model instance answers "is each red marker
    exactly on the intended element? which are wrong?"; disagreements re-open the cell.
  * Output per group: mean, error-bar half-length (data units), error-bar type (from caption via text agent),
    n if printed, plus per-value σ (pixel resolution ⊕ MAD ⊕ axis residual), the calibration record and the
    overlay PNG (provenance).
  * Special cases handled: time-series (late window = mean of last N points or the paper's own late-block value —
    protocol decides), grouped bars, box plots (→ Wan/Luo/Shi conversions), individual points (→ mean/SD from
    points; count check vs n), log/broken axes (detected → flagged/handled), half error bars.
* **Test-statistic extractor**: finds independent-samples t / between-groups F(1,df) / exact p for the A-vs-B
  contrast on the outcome (with quotes), the direction of the difference (which group higher), and any reported d/g.

### 3.3 Verification (`canopy/verify/`)
1. **Consistency checks (code)**: n integer ≥ 2 and Σn vs total N; SD/SE > 0; SE·√n ≈ SD when both; CI symmetric;
   figure values inside axis range; dispersion type known; effect |d| ≤ 3 flag; t/F only with df + group sizes;
   quote grounded; duplicate numbers across outcomes; group-label sanity (A/B not swapped: agent must echo the
   group text label it used, checked against protocol synonyms).
2. **Agreement vote (code)**: text/table = exact after normalization (± printed precision); figures = tolerance
   max(2% axis range, 0.5×tick) using per-cell medians; agreement across ≥2 heterogeneous candidates ⇒ accepted-by-vote.
3. **Adversarial verifier (LLM, different model from the extractor)**: sees only the candidate + source page(s)
   (+ crop with overlay for figures) and is instructed to *refute*: wrong group / time window / panel / SE-vs-SD /
   baseline vs post / units / subgroup. Returns confirmed / refuted / ambiguous + reason + alternative.
4. **Adjudicator (LLM, strongest model, xhigh)**: only when vote fails or verifier refutes; sees all candidates and
   evidence; outputs the decision, rationale, `needs_human`.
5. **Sign/orientation check (LLM + code)**: an agent states, with quotes, whether *higher raw values mean more of the
   construct* for this measure (e.g. "error" → False); code applies `orient()`; a second agent must independently
   agree; disagreement → human flag. The direction of the group difference stated in the paper's text (if any) is
   compared with the computed sign.
6. **Confidence (code)** from: source kind, grounding, #agreeing heterogeneous candidates, verifier verdict, MAD,
   consistency flags → `auto_accept` / `accept_with_note` / `needs_human`. Nothing is silently "fixed" by an LLM.

### 3.4 Effect sizes (`canopy/stats/effect_sizes.py`, done)
Route precedence (configurable): reported d/g (A vs B, sign convention checked) → text/table M+SD → M+SE/CI →
figure M+error → t/F/p. Conversions per Borenstein/Cochrane; variance convention default `borenstein`
(metafor); estimator default `cohen` (protocol may choose `hedges`). Optional digitization variance added to
SE(d)² (delta method) and flagged when > 10% of sampling variance. Each `EffectSize` records the full conversion
chain as human-readable text ("SD_old = SE 2.01 × √19 = 8.76; d = (44.67 − 46.14)/pooled 7.52 = −0.196; oriented ×+1").

### 3.5 Pooling & outputs (`canopy/stats/meta.py` done; `canopy/report/`)
* RE model (REML default; DL/PM options), FE for reference, HK option, PI method (V default; HTS/z options),
  heterogeneity (Q, I² meta-style + tau²-based, tau², H²), Egger; subgroup by a protocol moderator; sorted forest.
* Forest plot (matplotlib, `dataviz` conventions): study label, year, moderator columns, N(A/B), square size ∝ RE
  weight, CI whiskers, diamond, PI bar, heterogeneity footer, x-axis labels from protocol ("Reduced in A ↔ Enhanced
  in A"); PNG + SVG + PDF; also an interactive HTML version (Plotly-free inline SVG with hover) in the UI.
* Extraction table (CSV/XLSX/JSON) with one row per dataset×outcome: all raw values, route, conversions, confidence,
  flags, provenance links.
* **Methods breakdown figure**: % of datasets by extraction route (text M±SD, table, figure, test statistic, reported)
  with representative examples — auto-generated from the run.
* Provenance bundle: per value, page crop with highlight (text) or overlay (figure) + JSON.
* Human-review queue: everything `needs_human`, with what's uncertain and both candidate values.

### 3.6 Web UI + CLI (`canopy/server/`, `canopy/cli.py`)
* CLI: `canopy run --papers DIR --protocol protocol.yaml --out runs/NAME [--models …] [--budget-usd N] [--batch]`,
  `canopy serve`, `canopy protocol init`, `canopy validate`.
* Server: FastAPI; jobs run in a background worker (asyncio + bounded concurrency; per-paper tasks); SSE progress
  stream (stage per paper, cost so far); results endpoints; static SPA (vanilla TS/JS + inline SVG forest, mobile-friendly).
* Pages: **New run** (protocol editor with presets + folder upload via `webkitdirectory`, model/budget options) →
  **Run monitor** (per-paper stage grid, live log, cost meter, cancel) → **Results** (forest plot with click-through to
  evidence, extraction table with filters, flags queue, downloads) → **Evidence drawer** (page image with highlight /
  figure crop with overlay, quotes, candidates, verifier reasoning, conversion chain).
* Local-first: runs on the user's machine with their API key (`.env` or UI settings); no data leaves except to Anthropic.

### 3.7 Cost & robustness
* Prompt caching: system prompt + protocol + document blocks cached; per-model prefixes kept byte-stable.
* Concurrency: default 4 papers × 3 agents in flight; rate-limit aware retries (SDK), overloaded (529) backoff.
* Budget: `--budget-usd`; the run stops accepting new LLM calls when exceeded and finishes with what it has (flagged).
* Refusals: `stop_reason == "refusal"` handled; server-side `fallbacks: "default"` on Opus 5/Fable 5 calls.
* Resumable: stage outputs and LLM cache on disk; `canopy run --resume`.
* Batches API optional (`--batch`) for the extraction fan-out (50% cheaper, minutes–hours latency).

## 4. Evaluation (`validation/`)
* Unit: stats vs R fixtures (done); ingestion golden tests (page text, figure regions on 3 sample PDFs); schema/prompt
  round-trip tests with recorded LLM responses (cassettes) so CI needs no key.
* End-to-end on the Cisneros corpus: (1) per-value agreement with the human WebPlotDigitizer values (means, SDs) —
  MAE in % of axis range, ICC; (2) per-dataset d agreement (scatter manual vs auto with CIs; concordance; sign
  agreement rate); (3) pooled estimate/CI/I² agreement; (4) auto-accept precision (fraction of auto-accepted values
  within tolerance); (5) cost and wall-clock per paper. Discrepancies are classified (tool error / human error /
  protocol ambiguity) — several human-notebook bugs are already known.
* Guardrail against overfitting: protocol written from the paper's Methods section only; no per-study rules; the gold
  tables are read only by `validation/` scripts.

## 5. Deliverables for the paper (`validation/scripts/`)
1. `compare_manual_vs_auto.py` → scatter of manual vs tool d (with CIs), per outcome; concordance stats.
2. `extraction_routes_figure.py` → % of datasets per route + representative example panels.
3. `example_text_ms.py`, `example_figure_only.py`, `example_test_statistic.py` → three runnable worked examples that
   show exactly how the model pulled the numbers (page image with highlight, crop with overlay, conversion chain).
4. `replot_forest.py` → auto forest plots with RE weights (+ side-by-side with the manual plot).
5. `pipeline_figure.md` → inputs for the user's own methods figure (stage list, agent roster, prompts version).

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Wrong group/condition/timepoint mapping (dominant failure mode in the literature) | mapper + independent cross-check; quote-first grounding; adversarial verifier; human queue |
| SE vs SD vs CI confusion | dedicated error-bar-type resolution from caption/legend/methods with quotes; conflict → adjudicate; unknown → human |
| Digitization error | vector-exact path; CV snapping; ensembles; overlay-verify; per-value σ; flag > 2% range |
| Sign errors | explicit `higher_is_better` agent + code orientation + text-direction cross-check |
| Cost blow-up | caching, targeted context, budget cap, batch mode, Sonnet members |
| Hallucinated quotes/pages | verbatim grounding check; ungrounded → verifier/human |
| Model refusals/outages | fallbacks, retries, resumable runs |

## 7. Open questions (to settle during implementation)
* Whether Fable 5 as adjudicator/ensemble member is worth 2× cost (default: Opus 5 xhigh; Fable optional flag).
* Best default `late-window` rule for time-series figures (protocol option: `paper_reported_block` | `last_point` | `mean_last_n`).
* How to present per-value digitization uncertainty in the forest plot (default: not shown; available in table).

## 8. Decisions adopted from the research synthesis (`docs/research/00-synthesis-and-decisions.md`)
* **License**: core depends on PyMuPDF (AGPL-3) → Canopy is licensed AGPL-3.0-or-later (pyproject updated). A permissive
  build would need a pypdfium2/pdfplumber backend behind the same `ingest` interface (not v1).
* **Statistical profiles**: `profile: metafor` (default: Cohen d, Borenstein variance, REML, PI z) and
  `profile: cisneros2024` (Cohen d, variance `hedges_olkin_df` = the Rmd formula, REML, PI HTS k−2, hakn=False,
  I² Q-based). Profiles are plain YAML in `canopy/profiles/`; the forest footer prints the conventions used.
* **Dataset key**: `(paper_id, experiment_id, outcome_id, group_pair)`; meta-analysis row key `doi+experiment+outcome`.
  Shared control groups across experiments are flagged (dependency) and, by default, only the first exposure enters.
* **Model roster & prices** (Aug 2026): Opus 5 $5/$25 (primary extraction + figures), Sonnet 5 $2/$10 (second route,
  verifier votes, locator cross-check), Fable 5 $10/$50 (optional adjudicator; needs `fallbacks:"default"`),
  Haiku 4.5 (none by default). Effort and thinking held constant across agents sharing a cached prefix.
* **Grounding**: verbatim quote check (NFKC + whitespace collapse for matching only; raw text kept); citations
  (page_location) only in the separate verifier pass because citations and structured outputs are mutually exclusive.
* **Ambiguity is flagged, never resolved by the model** (SD vs SE vs CI, baseline vs post, per-trial vs per-block).
* **Evaluation ceiling**: human WPD values also carry error; disagreements are adjudicated (both values inspected), not
  assumed to be tool errors.
