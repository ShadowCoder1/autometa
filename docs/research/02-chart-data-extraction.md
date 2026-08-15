# 02 — Extracting numeric values (mean ± error) from scientific charts

Research brief for Canopy, written 2026-08-15. Scope: state of the art (2024–2026) for turning bar/line/box/scatter figures with error bars into `group mean ± SE/SD/CI` values good enough for a random-effects meta-analysis, plus a concrete algorithm and fallback chain for our tool. Everything with a number has a source in the final section; anything I could not verify from a primary source is flagged **[unverified]**.

---

## 0. TL;DR

1. **Frontier VLMs (Claude Opus 4.7+/5, GPT-5.x, Gemini 3.x) are now the best general chart *readers*, and specialized chart models (DePlot, TinyChart, OneChart, ChartMoE, ChartCoder…) do not beat them on value extraction from real scientific figures.** On real-world charts specialized models collapse (OneChart 9.4 %, TinyChart 11.4 % vs Gemini 2.5 Pro 52.3 % ECS on EpiCurveBench; TinyChart 2.9 vs Gemini 3.1 Pro 59.2 mAP on ChartArena). The only place a specialist wins is a 7B model *fine-tuned on the exact distribution* (ExChart: 4.9 % adaptive-MAPE vs Gemini 2.5 Flash 6.7 % on its own synthetic-heavy benchmark).
2. **But "reads the value" ≠ "reads it to 1–2 % of axis range".** Direct value read-out by a frontier VLM on unlabeled bars/points is typically 2–6 % of the axis range per value (PlotExtract with Claude 3.5 Sonnet: MAEy ≈ 1–2.6 %, MAEx ≈ 3 %; ExChart's best frontier model 3.8–5.6 % adaptive MAPE on bars). That is fine for Cohen's *d* only if the between-group difference is much larger than that error; for small effects it is not, and error bars (which are short) are read *proportionally* worse.
3. **Pixel-level measurement beats value read-out.** The reliable recipe is: VLM for *semantics* (which bar is which group, what the error bar represents, where the axis ticks are), classical CV / vector geometry for *measurement* (bar tops, cap positions, tick pixel positions), least-squares axis fit for pixel→data. Where the figure is vector art inside the PDF (very common for matplotlib/R/Prism exports), PyMuPDF `page.get_drawings()` gives *exact* coordinates and the digitization error is essentially zero.
4. **Claude ≥ 4.7 returns absolute pixel coordinates in the image it sees, and if you pre-resize with Anthropic's published `resized_size()` (max long edge 2576 px, ≤ 4784 visual tokens of 28×28) the coordinates map 1:1 onto your image.** Anthropic still calls localization "approximate", so use VLM coordinates as *seeds* for CV refinement, not as final measurements.
5. **Techniques with measured gains:** cropping/zooming to a single panel (ReFocus +6.8 % on charts, PixelCraft +5.6 to +9.5 points on CharXiv/ChartQAPro/EvoChart; 2× upscaling +3 pp in PlotPick), giving the model the axis info explicitly (FairChart2Table: "prompting with y-axis information significantly enhances performance"), self-ensembling with per-cell medians (+2.7 % to +23 % relative, and dispersion predicts error, ρ ≈ −0.35), and render-and-compare verification (PlotExtract: 100 % precision on "is this extraction faithful?" at ≈ 82–89 % recall).
6. **Realistic targets for Canopy:** vector figures → < 0.5 % of axis range; raster bar/point charts with clean axes → 1–2 % of axis range for means, 2–5 % for error-bar half-lengths; hard cases (log axes, overlapping series, low-res scans) → 5–10 % and must be flagged for human review. Per-value uncertainty should be reported (from ensemble dispersion + pixel resolution) and propagated into SE(d) as an extra variance term.

---

## 1. What we actually need from a figure

For each dataset (study × outcome), per group: **M, error half-length, what the error is (SE / SD / 95 % CI / IQR), n**. Sign/orientation of the outcome. For time-series (e.g. adaptation curves): the mean of the last *N* cycles/blocks per group. Provenance: page, panel bbox, pixel positions of what was measured, the crop, and the calibration used.

Two things make this harder than "chart QA" benchmarks:

* **Error bars are short.** A 5 % axis-range error on a bar top is a 5 % error on the mean but can be a 50 % error on an SE that spans 10 % of the axis. SD ← SE·√n amplifies nothing, but *d* is divided by pooled SD, so a 20 % under-read of the error bar inflates *d* by 25 %.
* **Small between-group differences.** In Cisneros-style data (young vs older adults), the two bars often differ by less than 10 % of the axis range. A 2 % per-bar error is then a 20–30 % relative error on the difference.

So the target metric is *pixel error relative to axis range*, not RMS-F1 / ChartQA relaxed accuracy (5 % tolerance = "correct"), which is what most benchmarks report.

---

## 2. (a) Frontier vision-language models

### 2.1 Benchmarks (what they measure and where the models sit)

| Benchmark | What it measures | Best reported (primary source) |
|---|---|---|
| **CharXiv-Reasoning** (arXiv figures, 1 000 val Qs) | reasoning over real scientific charts; humans 80.5 % | Gemini 3 Pro 81.4 % vs Gemini 2.5 Pro 69.6 %, Claude Sonnet 4.5 68.5 %, GPT-5.1 69.5 % (Google's own runs, Nov 2025); GPT-5 (thinking) 81.1 % (OpenAI, Aug 2025). Aggregator leaderboards in Aug 2026 show Claude Opus 4.7 ≈ 91 %, Opus 4.8 ≈ 89.9 %, Sonnet 5 ≈ 88.3 %, Gemini 3.7 Flash ≈ 88.7 % **[unverified secondary]** — the benchmark is close to saturated. |
| **ChartQA** | QA over simple infographics, 5 % relaxed accuracy | frontier VLMs > 89 % RMS chart-to-table; "showing diminishing headroom" (EpiCurveBench). |
| **ChartX** (chart-to-table, RMS-F1) | synthetic multi-type charts | VLMs 88–96 % recall vs DePlot 71 %; box plots: DePlot 24 % vs VLMs 83–97 % (PlotPick, May 2026). |
| **ExChart-Bench** (3 600 chart-table pairs, adaptive MAPE) | value recovery without data labels | Gemini 2.5 Flash 6.7 %; GLM-4.5V 5.9 %; fine-tuned ExChart-7B 4.9 %; bars 3.8–5.7 % (CHI 2026). |
| **EpiCurveBench** (1 000 real epidemic curves, dense series) | dense time-series digitization (ECS metric) | Gemini 2.5 Pro (high) 52.3 %; GPT-5.2 45.4 %; Claude Opus 4.5 42.0 %; Qwen3-VL-235B 27.5 %; **OneChart 9.4 %, TinyChart 11.4 %**. Bars are "uniformly easier than line charts" (Gemini 72.7 % vs 47.5 %). |
| **ChartArena** (multi-lingual chart parsing, mAP) | chart-to-table incl. photos, hand-drawn | Gemini 3.1 Pro 59.2; Kimi K2.5 54.8; Qwen3.5-35B 54.1; **RRVF-7B 36.0 (best specialist); ChartMoE 8.5; ChartCoder 12.6; TinyChart 2.9**. |
| **ChartBench** (2025) | unlabeled charts, stacked bars | "GPT-5 achieves only 59.2 % on complex stacked bar charts". |

Take-aways: reasoning benchmarks are saturating; *precise numeric extraction* on real, dense, unlabeled figures is not (best ≈ 50 % on EpiCurveBench). Reasoning effort and code-execution tools help inconsistently (EpiCurveBench: high effort +11.8 ECS for Gemini, +0.7 for Claude, −1.5 for GPT-5.2; the dominant tool-use failure was *inaccurate self-cropping*).

### 2.2 Direct value read-out accuracy on scientific plots

* **PlotExtract** (Polak & Morgan, 2025; Claude 3.5 Sonnet, which they found better than GPT-4o): on *published* two-axis plots, MAEx ≈ 2.8 %, MAEy ≈ 2.4 % of range per point; on their *synthetic* set MAEx ≈ 3.0 %, MAEy ≈ 0.94 % (Table 1; the authors call the synthetic set the "worst-case" because of its random, sub-optimal presentation); precision/recall of point recovery ≈ 92–94 % (published) / ≈ 92 % (synthetic). Their render-and-compare filter had 100 % precision, 81.8 % (published) / 88.9 % (synthetic) recall on flagging faithful extractions. *(fact-check: the published/synthetic MAEy figures were swapped in the original draft — corrected.)*
* **Material Database Agent** (2026, one Debye-temperature curve, 5 points): MAE (K, values ~ hundreds) GPT-5.2 3.44, Claude Opus 4.6 1.61, GPT-5.4 1.27, GLM-5V-Turbo 1.09, Claude Opus 3 ≈ 186 (catastrophic). Tiny sample, but shows the generation gap and that MMMU-Pro correlates with digitization skill.
* **PlotPick** (2026; Haiku 4.5, Sonnet 4.6, Gemini 3 Flash/3.1 Flash-Lite, GPT-5.4 nano/mini; prompt = "Extract the data from this chart as a TSV"): ≥ 98 % accuracy when data labels are printed; 2× upscaling +3 pp; stacked/grouped bars systematically harder.
* **Self-Ensembling** (Berkane et al., May 2026): repeated sampling + per-cell median gives +2.7 % to +23.1 % relative RMS-F1; relative-MAD across samples correlates with error (Spearman −0.34 to −0.37); early-stopping keeps 99 % of the gain at ≈ 16 samples; < $15 per 1 000-chart benchmark.

**No study yet reports a formal LLM-vs-WebPlotDigitizer agreement study (ICC/Bland–Altman) on meta-analysis bar charts with error bars.** The closest are (i) the plant-science pipeline (bioRxiv Feb/Mar 2026): AI extraction vs human meta-analysis data, paper-level ICC 0.838, TOST equivalence at ±2 pp, ~$0.37/paper — but figure-derived values are not isolated; (ii) Tan & D'Souza 2026 finding "near-zero reliability for full meta-analytic association tuples" when an LLM does everything in one shot (role reversal, numeric misattribution) — an argument for our decomposition + verification design rather than a single prompt.

### 2.3 Pixel coordinates from frontier VLMs

* **Claude 4.7 and later** are on Anthropic's "high-resolution tier": max long edge **2576 px**, max **4784 visual tokens** (28×28 patches; ⌈w/28⌉×⌈h/28⌉), padded on bottom/right to a multiple of 28. Coordinates are absolute pixels **in the image Claude sees after resizing**; if you pre-resize with the reference `resized_size(w,h,max_edge=2576,max_tokens=4784)` they map 1:1 (docs, Aug 2026). Prompt for absolute pixels ("Return `[x1,y1,x2,y2]` in pixel coordinates"), *never* normalized 0–1000 (documented to work poorly). Docs explicitly say to crop small targets and send the crop, and that PDF uploads cannot be mapped back — rasterize pages yourself. Docs still list "Spatial reasoning: coordinate and localization outputs are approximate" as a limitation. Chart-element grounding is named as an intended use ("bounding boxes for … chart elements"). Older Claude models: 1568 px / 1568 tokens (a 1920×1080 image is downscaled to 1456×819).
* **Gemini 2.5/3.x**: pointing and boxes are native (normalized 0–1000 in the docs); Gemini 3 Pro advertises "pixel-precise coordinates" and chart "derendering" to code. Best CharXiv-R / ChartArena / EpiCurveBench scores at time of writing.
* **GPT-5.x**: strong chart reasoning (CharXiv-R 81.1 % at launch), no documented pixel-coordinate contract; COREval-style benchmarks report that many VLM families "fail to produce acceptable spatial localizations", with LLaVA/Gemini families the exceptions (2024–25).
* Independent evidence on chart grounding: ChartREG++ (2026) benchmarks point/box grounding of chart elements incl. tick marks (PCK@0.01) — even Gemini-2.5-Pro-Thinking "can pinpoint the target in its reasoning but cannot correctly predict the box". PixelCraft (Microsoft, 2025) got its gains by fine-tuning a *3B* grounding model and handing its coordinates to classical CV — a strong hint that VLM coordinates should seed CV, not replace it.

**Practical conclusion**: use Claude Opus 4.7+/Sonnet 5 (or Gemini 3.x as a second opinion) for semantics + approximate localization; refine with CV; measure in pixels; convert with a fitted axis.

---

## 3. (b) Specialized chart models

| Model (year) | Type | Verdict for value extraction |
|---|---|---|
| DePlot / MatCha (Google, 2022–23) | Pix2Struct chart→table | 71 % recall on ChartX vs 88–96 % for VLMs; 24 % on box plots (PlotPick). Legacy baseline only. |
| UniChart, ChartAssistant, ChartLlama, ChartInstruct (2023–24) | fine-tuned open VLMs | Below Qwen2.5-VL-7B zero-shot on ExChart-Bench. |
| TinyChart-3B (2024), OneChart (2024) | small chart→table specialists | Best specialists on ChartQA-era data; collapse on real charts (EpiCurveBench 11.4 % / 9.4 %; ChartArena 2.9). |
| ChartMoE-8B (ICLR 2025), ChartCoder-7B (2025), ChartGemma (2024) | MoE connector / chart-to-code | ChartArena 8.5 and 12.6 mAP — far below frontier. |
| RRVF-7B (2025, RL from render-verify feedback) | RL-trained chart→code | 36.0 mAP: best open specialist but still < open generalists (Qwen3.5-35B 54.1). |
| ExChart-7B (CHI 2026, Qwen2.5-VL base + coordinate-perception stage) | fine-tuned extractor | Beats Gemini 2.5 Flash on its own benchmark (4.87 % vs 6.72 % adaptive MAPE); training recipe (Stage 1: predict per-mark x,y; Stage 2: table) is a good template if we ever fine-tune. |
| ChartReader / ChartOCR / ChartDete (2019–23) | CV keypoint detectors | Only useful as ideas (keypoint heads for bar corners); no maintained weights that generalize. |

Bottom line: **do not add a specialized chart model to the pipeline** — it costs a GPU dependency and loses to Claude/Gemini on real figures. The lesson from ExChart/PixelCraft (train the model to see *coordinates*, then let CV do the arithmetic) is exactly the hybrid we should implement with prompting + CV instead of fine-tuning.

---

## 4. (c) Classical digitizers and hybrid CV

### 4.1 Tools

* **WebPlotDigitizer** (Ankit Rohatgi). v4.x is AGPL-3 (last GitHub release **4.7, Feb 2025**; NodeJS batch CLI since 4.2; Docker/npm build). **v5** at automeris.io: frontend AGPL, but the "AI Assist" backend (auto axis/chart detection) is **proprietary cloud** — not usable in a local-first tool. Automatic algorithms: Averaging Window, X-Step (with interpolation), Custom Independents (digitize at user-specified X), Blob Detector, Template Matching (scatter), Bar extraction (positive+negative bars). Human reliability with WPD is the de-facto gold standard: Drevon et al. 2017, 3 596 points from 168 series / 36 graphs / 18 studies, "high" intercoder reliability and validity (the ≈ .99 ICC / r figures are the commonly cited values but sit behind the paywall — treat as **[unverified exact value]**). **We should not embed WPD**; its algorithms are simple (color mask + column scan) and we can reimplement them in OpenCV in < 200 lines while keeping our own provenance.
* **metaDigitise** (R, Pick/Nakagawa/Noble 2019, MEE) — the meta-analysis community's tool: mean-error plots (bar/point + SE/SD/CI), box plots, scatter, histograms; user clicks calibration + error-bar ends; converts to mean/SD given n. Good spec for what our output schema must contain (it stores calibration, clicks, error type, n → mean, SD). Its box-plot → mean/SD conversion follows the same estimators we list in §7.
* **Engauge Digitizer**, **PlotDigitizer.com** (commercial), **dilawar/PlotDigitizer** (`pip install plotdigitizer`; OCR axis limits, batch CLI, log axes), **juicr** (R). All manual/semi-automatic; none does error bars automatically. Useful only as UX references.

### 4.2 Hybrid CV pipeline (what actually works, and what we should build)

1. **Panel & plot-area detection**: VLM returns bboxes of panels + the plot area (axes rectangle) → refine with Hough/LSD long straight lines to find the x/y axis lines and gridlines to ±0.5 px.
2. **Axis calibration**: OCR tick labels (tesseract on 3–4× upscaled axis strip, or ask the VLM for tick *text* + approximate pixel positions), snap each label to the nearest detected tick mark / gridline, then **least-squares fit** `value = a + b·pixel` (or `log10(value)` for log axes) over all ticks; reject with RANSAC; report residual RMS. Two-point calibration (WPD default) is strictly worse than an n-tick fit. FairChart2Table showed VLM value read-out is biased by tick digit length, tick count, range and format (K/M/sci) — the fit removes all of that.
3. **Mark detection**: bars by color/contour (connected components inside plot area, filter aspect ratio, ignore legend); points by template/blob matching per series color; lines by color mask + per-column median (WPD "averaging window").
4. **Error bars**: for each bar/point, scan a narrow vertical strip (±0.3 bar-width) above/below the mark for the vertical whisker segment (same color as bar edge or black); cap = short horizontal run at the whisker end (Hough or morphological hit-or-miss); if capless, take the whisker extremum. Handle "half error bars" (only upper) and error bars hidden behind the bar (only upper visible → assume symmetric). Record both ends; the half-length = |cap − mark| in pixels.
5. **pixel → data** with the fitted axis; propagate the fit's residual and ±0.5 px quantization into per-value uncertainty.
6. **Vector shortcut**: if the figure region contains PDF drawings (PyMuPDF `page.get_drawings()` → items `'re'` (rects), `'l'` (lines), `'c'` (curves) with fill/stroke color and `rect`), bars are exact rectangles, whiskers exact lines, ticks exact short lines, and tick labels come from `page.get_text("dict")` with bboxes — no OCR, no raster error. `Page.cluster_drawings()` groups a chart's paths. This covers a large fraction of modern journal PDFs (matplotlib/ggplot/Prism/MATLAB export vector by default) and is the single highest-value trick for accuracy. Always check the figure is *not* an embedded raster (`page.get_image_info()`).

---

## 5. (d) Figure extraction for meta-analysis / systematic reviews — what is published

* Human WPD extraction: ICC ≈ .99 (Drevon 2017); metaDigitise reports "little inter-observer bias". This is the bar we are measured against.
* LLM data extraction for SRs (text-level): GPT-4o/o3 72–96 % element accuracy depending on prompting/batching (Cambridge RSM 2025; JMIR AI 2025) — but these do not isolate figure-derived numbers.
* Plant-science meta-analysis pipeline (bioRxiv 2026, v1 dual-model Claude Sonnet 4 + Kimi K2.5 with Gemini 3 Flash tiebreak; v2 single Claude Opus 4.6 agent): paper-level ICC 0.838 vs humans, aggregate effects reproduced within 0.05–2.17 pp, ≈ $0.37/paper. Encouraging for the *whole pipeline*, silent on figure digitization specifically.
* Domain pipelines: PlotExtract (materials, Claude 3.5 Sonnet, ≈ 5 % error); GPT-4V for MOF plots (> 93 % identification, not value accuracy); EpiCurveBench (public health) — best 52 % ECS, "far from usable".
* **Gap**: nobody has published Bland–Altman/ICC of a VLM+CV pipeline against WPD on group-mean ± error charts. Canopy's Cisneros validation (44 studies, WPD-digitized ground truth) would be a publishable contribution; design the evaluation now (per-value relative error vs axis range, ICC, Bland–Altman, and downstream Δd, ΔI², Δpooled).

---

## 6. (e) Techniques that measurably help

| Technique | Evidence | How we use it |
|---|---|---|
| Crop/zoom to one panel, one series | ReFocus +6.8 % charts (GPT-4o); PixelCraft +5.6–9.5 pts; Anthropic docs: "for fine targets, crop the region and send the crop"; EpiCurveBench: model self-cropping is the top failure → **we crop deterministically**, not the model. | Panel bbox from VLM → deterministic crop with 5 % margin at high DPI. |
| Render vector PDF at high DPI (not screenshots) | Anthropic tier: ≤ 2576 px / 4784 tokens; PlotPick 2× upscale +3 pp; ExChart failure "large-magnitude ticks exceed digit counting" is an OCR issue. | Render page at 300–600 DPI, crop panel, then `resized_size()` to the tier limit; separately OCR axis strips at 4×. |
| Ask for tick labels + pixel positions first (calibrate before reading) | FairChart2Table: prompting with y-axis info "significantly enhances performance"; ExChart Stage-1 (coordinate perception) is the biggest gain. | Structured JSON: `{axis, ticks:[{label, px}], scale: linear|log, broken:bool}`; snap to CV-detected ticks. |
| Gridline anchoring | y-axis bias paper: gridlines "provide compensatory reference"; CV: gridlines give more calibration points. | Include gridlines in the least-squares fit. |
| Ensembles / self-consistency | Self-Ensembling: +2.7–23 % rel., dispersion→error ρ ≈ −0.35; ~16 samples enough. | 5–8 samples across (crop scale ×1/×1.5, two prompts, Opus+Sonnet or Opus+Gemini); per-value median; MAD → uncertainty. |
| Propose → overlay → verify | PlotExtract render-and-compare: 100 % precision, ≈ 82–89 % recall on flagging faithful extractions; PixelCraft "critic" agents; ReFocus draws boxes/lines. | Draw the measured bar tops, cap positions and calibration ticks on the crop; ask a *fresh* model "does each mark sit on the drawn line? list mismatches"; iterate ≤ 2×. |
| Read legend/caption/methods for SE vs SD vs CI and n | Not benchmarked but essential; ambiguity is the largest *systematic* error source (SE vs SD is a √n factor on *d*). | Separate text agent extracts "error bars represent …" quote + n per group with page provenance; if absent, flag; default per Cochrane: assume SE only with explicit evidence, else mark uncertain. |
| Effort/thinking & code tools | EpiCurveBench: inconsistent (Gemini +11.8, Claude +0.7, GPT −1.5); Claude High+Code cost 3×. | Use moderate effort for semantics; do *not* rely on model-side code execution for measurement — our CV does it deterministically. |

---

## 7. Recommended algorithm + fallback chain for Canopy

```
figure page → panel detect (VLM bbox) → for each relevant panel:
  ├─ A. VECTOR path (if drawings exist under panel bbox)          target err < 0.5 % range
  │     rects/lines/text from PyMuPDF → bars, whiskers, ticks, tick text
  │     → LS axis fit → values; VLM only labels which mark = which group/condition
  ├─ B. RASTER-CV path (raster figure or A fails self-checks)      target 1–2 % (means), 2–5 % (SE half-length)
  │     render 300–600 DPI crop → axis lines (LSD/Hough) → tick marks + OCR (tesseract 4×) 
  │     → LS axis fit (RANSAC, RMS residual) → bar/point/line detection by color
  │     → error-bar whisker+cap scan → pixel values (+½ px) ; VLM seeds bboxes and labels series
  ├─ C. VLM-COORDINATE path (CV cannot segment: overlapping/hatched/low-res)   target 2–5 %
  │     pre-resize crop with resized_size(…,2576,4784) → Claude Opus 4.7+ returns
  │     pixel coords of tick labels, bar tops, cap ends (structured JSON) → snap to
  │     nearest edge within ±4 px (Canny) → LS fit → values ; ensemble ×5, median, MAD
  └─ D. VLM VALUE READ-OUT (last resort / cross-check)                     target 3–6 %
        two models × two prompts, "give M and error for each group", median, MAD

after any path: OVERLAY-VERIFY (draw marks on crop → fresh model critiques) 
→ compare paths B/C/D when ≥2 available: agreement < 2 % range → accept, else escalate/flag
→ SE/SD/CI/n resolution from caption/legend/methods (text agent) → M, SD, n per group
→ store provenance {page, panel bbox, crop png, calibration ticks+fit, pixel measurements, samples, MAD}
```

Design rules:

* **Never let the model measure**: it labels, localizes approximately, and critiques. Pixels come from CV or vector geometry.
* **Everything is a fit with residuals**: axis calibration RMS, cap-detection confidence, ensemble MAD → per-value σ.
* **Two independent estimates per value** whenever cost allows (e.g. B and C, or B and D). Disagreement is the best error detector we have (self-ensembling ρ ≈ −0.35 for one model; cross-path disagreement should be stronger).
* **Model choice**: Claude Opus 4.7+/Opus 5 (high-res tier, pixel contract) as primary; Sonnet 5 for cheap ensemble members; Gemini 3.x optional second family for adjudication. Prompt caching for the shared instructions; Batches API for the ensemble runs; Files API to avoid re-uploading crops in overlay-verify turns.

### 7.1 Per-value uncertainty

For a value read from pixels: `σ_pix² = (0.5 px)² + σ_capdetect² ; σ_val = |b|·σ_pix ⊕ axis-fit residual`, with `b` = data units per pixel. From ensembles: `σ_ens = 1.4826·MAD`. Report `σ_M`, `σ_err` per group. Propagate: `Var(d)_total = SE(d)² + (∂d/∂M₁ σ_M₁)² + … + (∂d/∂SD σ_SD)²` (delta method); at minimum, add the digitization variance to `SE(d)²` and flag datasets where it exceeds ~10 % of sampling variance. Any value with `σ_val > 2 % of axis range` or path disagreement > 3 % goes to the human-review queue with the overlay image.

### 7.2 Special cases

* **Box plots** (median, Q1, Q3, whiskers, n): measure the five numbers in pixels; convert with Wan et al. 2014 (mean ≈ (Q1+m+Q3)/3 for large n; SD ≈ IQR/1.35 with the n-dependent η(n) correction), Luo et al. 2018 (optimal weighted mean), Shi et al. 2020 (SD), or McGrath et al. 2020 quantile-estimation (`estmeansd` R pkg, also handles skew) — Cochrane Handbook §6.5.2.5 (IQR → SD) and §6.5.2.9 (missing means) describe the Wan/Bland estimators (Luo/Shi/McGrath are not named there). Whiskers may be min/max, 1.5·IQR, or 5–95 %; read the caption; if min/max are given, use the (min,Q1,m,Q3,max) formulas. Mark such rows with `error_source = "boxplot_converted"` and larger σ (the estimators themselves add ~5–15 % SD error at n < 25).
* **Grouped bars**: series membership by color/hatch and legend; ask VLM for legend swatch bboxes → sample RGB → match bars. **Stacked bars**: segment boundaries via color changes along the bar column; values are differences; error bars on stacked bars are almost never per-segment — usually reject for meta-analysis unless caption is explicit.
* **Log axes**: detect from tick labels (1, 10, 100 or 10^k) or minor-tick spacing; fit `log10(value) = a + b·px`; error bars on log axes are asymmetric in data space — convert both ends separately and report asymmetric errors (do not average unless the analysis needs a symmetric SD; then work on log scale).
* **Broken axes**: detect the break glyph (double slash / gap) via VLM + tick discontinuity in the LS fit (large residual, monotone violation); fit piecewise, one segment per contiguous tick run; refuse if a mark lies inside the break.
* **Time-series / learning curves** ("late adaptation = mean of last N cycles"): digitize the series with per-x measurement (X-Step at the x positions of the last N ticks or "Custom Independents"), and per-point error bars if drawn; the SD of the *mean of the last N* is **not** the mean of the per-point SDs — per-point SEs are within-subject and correlated across cycles. Follow the protocol: use the paper's own reported late-block value if any; otherwise take mean of last N means, and use the error bar of the block/epoch average if the paper reports one, else the median per-point error bar as an approximation and flag `error_source = "approx_late_block"`. Also handle "aftereffect" = mean of first N no-feedback trials, same machinery.
* **Individual-point scatter/strip plots with mean line**: mean = horizontal marker; SD can be recomputed from the digitized points (blob/template detection, count vs n as a self-check).
* **Half error bars / hidden lower whisker**: assume symmetric; note in provenance.
* **Legend says "± SEM" but methods say SD**: text agent flags the conflict; adjudicator decides with quotes; both stored.

### 7.3 Expected error budget (to be validated on Cisneros)

| Path | Means (% of axis range) | Error-bar half-length | Notes |
|---|---|---|---|
| A vector | < 0.5 % | < 0.5 % | limited by rounding in the PDF (~0.01 pt) |
| B raster-CV | 1–2 % | 2–5 % (bars ≥ 8 px), else flag | 300–600 DPI, clean axes |
| C VLM-coords + snap | 2–5 % | 5–10 % | ensemble ×5 |
| D VLM read-out | 3–6 % | 5–15 % | cross-check only |
| Human WPD (reference) | ≈ 1 % (ICC .99) | ≈ 2–3 % | Drevon 2017 |

Downstream: with typical Cisneros bar heights, a 1–2 % mean error and 3–5 % SD error keep |Δd| ≲ 0.05–0.10 per study, which barely moves a pooled estimate over ~40 studies; the failure mode to design against is *gross* errors (wrong group/series, SE-vs-SD, wrong panel), hence the emphasis on semantic verification and dual paths rather than sub-pixel accuracy.

---

## 8. Open questions / risks

1. No published ICC of a VLM+CV pipeline vs WPD on mean±error charts — we must produce it ourselves; ground truth exists (Cisneros digitized values), but the human values also carry ~1–2 % error, so the agreement ceiling is ~ICC .98.
2. How much of the Cisneros corpus is vector vs raster figures determines how often path A applies (unknown until we run `get_drawings()` on the 44 PDFs).
3. Aggregator leaderboard numbers for Claude Opus 4.7/4.8/5 and Sonnet 5 on CharXiv (≈ 88–91 %) come from third-party sites; Anthropic's own release notes give no chart benchmark. Treat as indicative.
4. Anthropic's coordinate contract says "approximate"; our snap-to-edge step must be evaluated (what fraction of VLM points land within ±4 px of the true edge at 2576 px?). Budget a small experiment (50 bars, 50 caps).
5. Cost: Opus 5 at $5/M input, a 2576-px crop ≈ 4 784 tokens ≈ $0.024 per image per call; ensemble ×5 + verify ≈ $0.15–0.30 per panel; ~150 panels for a 44-study review ≈ $30–45 with Opus, ~5× less with Sonnet 5 members ($2/M input, now standard pricing; $1/M in batch) and Batches API (50 % off).
6. Error-bar semantics (SE vs SD vs CI) is the dominant *systematic* risk and cannot be solved by better pixels — needs the text-agent + adjudicator design from brief 05.

---

## Sources

Primary docs / papers consulted (accessed 2026-08-14/15):

* Anthropic, *Coordinates and bounding boxes* — https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
* Anthropic, *Vision* (resolution tiers 1568/2576 px, 1568/4784 tokens, limits, limitations) — https://platform.claude.com/docs/en/build-with-claude/vision
* Anthropic, *Introducing Claude Opus 4.7* (2026-04-16; 2576 px, xhigh) — https://www.anthropic.com/news/claude-opus-4-7
* Google, *Gemini 3 Pro: the frontier of vision AI* (CharXiv-R 80.5 % human baseline; pointing; derendering) — https://blog.google/innovation-and-ai/technology/developers-tools/gemini-3-pro-vision/
* Google DeepMind, *Gemini 3 Pro model evaluation methodology* — https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_model_evaluation.pdf ; CharXiv-R table (81.4/69.6/68.5/69.5) via https://simonwillison.net/2025/Nov/18/gemini-3/
* OpenAI, *Introducing GPT-5* (CharXiv-R 81.1 %, MMMU 84.2 %) — https://openai.com/index/introducing-gpt-5/
* Polak & Morgan, *PlotExtract* (arXiv 2503.12326; Claude 3.5 Sonnet; MAE ≈ 1–3 %; render-and-compare) — https://arxiv.org/abs/2503.12326
* Carstensen, *PlotPick* (arXiv 2605.06021; ChartX/PlotQA; VLMs vs DePlot; 2× upscaling) — https://arxiv.org/html/2605.06021
* Berkane & Majumder, *EpiCurveBench* (arXiv 2605.27195) — https://arxiv.org/pdf/2605.27195
* Berkane, Wang & Majumder, *Self-Ensembling VLMs for Chart Data Extraction* (arXiv 2605.27298) — https://arxiv.org/html/2605.27298
* *Making Multimodal LLMs Reliable Chart Data Extractors* / ExChart (CHI 2026; arXiv 2606.29808) — https://arxiv.org/html/2606.29808v1 ; https://dl.acm.org/doi/10.1145/3772318.3790721
* *ChartArena* (arXiv 2606.01348) — https://arxiv.org/html/2606.01348v3
* Song, Efat & Tavanapong, *Assessing Y-Axis Influence / FairChart2Table* (arXiv 2604.24987) — https://arxiv.org/pdf/2604.24987
* Zhang et al. (Microsoft), *PixelCraft* (arXiv 2509.25185) — https://arxiv.org/pdf/2509.25185
* Fu et al., *ReFocus* (arXiv 2501.05452) — https://arxiv.org/abs/2501.05452
* *ChartREG++* (arXiv 2605.07415) — https://arxiv.org/html/2605.07415 ; *ChartLens* — https://arxiv.org/pdf/2505.19360 ; *COREval* — https://arxiv.org/html/2411.18145
* *ChartBench* (ACM 2025; GPT-5 59.2 % stacked bars) — https://dl.acm.org/doi/10.1145/3772128.3772169
* Chandrasekhar et al., *Material Database Agent* (arXiv 2605.04278; per-model MAE on a digitized curve) — https://arxiv.org/pdf/2605.04278
* Plant-science AI extraction vs human meta-analysis (bioRxiv 2026.02.17.706322 v1/v2) — https://www.biorxiv.org/content/10.64898/2026.02.17.706322v1 ; https://www.biorxiv.org/content/10.64898/2026.02.17.706322v2
* Sallam et al. / RSM 2025, GPT-4o & o3 SR extraction — https://www.cambridge.org/core/journals/research-synthesis-methods/article/automating-the-data-extraction-process-for-systematic-reviews-using-gpt4o-and-o3/28C54A601B80E2BE6611C8B7B15FEDAF ; JMIR AI 2025 — https://ai.jmir.org/2025/1/e68097
* WebPlotDigitizer releases (4.7, Feb 2025; AGPL; NodeJS CLI) — https://github.com/automeris-io/WebPlotDigitizer/releases ; README (AI Assist proprietary) — https://github.com/automeris-io/WebPlotDigitizer/blob/master/README.md ; v5 docs — https://automeris.io/docs/digitize/
* Drevon, Fursa & Malcolm 2017, WPD reliability — https://journals.sagepub.com/doi/10.1177/0145445516673998
* Pick, Nakagawa & Noble 2019, metaDigitise — https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13118 ; https://github.com/daniel1noble/metaDigitise
* dilawar/PlotDigitizer (pip `plotdigitizer`) — https://github.com/dilawar/PlotDigitizer
* PyMuPDF Page API (`get_drawings`, `cluster_drawings`, `get_text("dict")`, `get_pixmap(dpi, clip)`) — https://pymupdf.readthedocs.io/en/latest/page.html
* Box-plot → mean/SD estimators: Wan et al. 2014 — https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-14-135 ; Luo et al. 2018 — https://doi.org/10.1177/0962280216669183 ; Shi et al. 2020 — https://doi.org/10.1002/jrsm.1429 ; McGrath et al. 2020 (`estmeansd`) — https://doi.org/10.1177/0962280219889080 ; Cochrane Handbook §6.5.2 — https://training.cochrane.org/handbook/current/chapter-06
* Aggregator leaderboard (unverified secondary) — https://benchlm.ai/benchmarks/charxiv

---

## Verification notes (fact-check)

Fact-check pass on 2026-08-15 against primary sources (Anthropic docs, arXiv full texts, GitHub, publisher pages). Verdicts: **confirmed** / **corrected** (fixed in place above) / **unverifiable** (left as-is, flagged).

| # | Claim (section) | Verdict | Source |
|---|---|---|---|
| 1 | Claude 4.7+ high-res tier: max long edge 2576 px, ≤ 4784 visual tokens, 28×28 patches, `⌈w/28⌉×⌈h/28⌉`; standard tier 1568 px / 1568 tokens; 1920×1080 → 1456×819 on standard tier (§0, §2.3, §6) | confirmed | https://platform.claude.com/docs/en/build-with-claude/vision (Resolution and token cost table) |
| 2 | Coordinates are absolute pixels in the post-resize image; reference `resized_size(w,h,max_edge=1568,max_tokens=1568)` (pass 2576/4784 for high-res); padding to multiple of 28 on bottom/right only; "does not work well" with normalized 0–1000; crop small targets; PDF uploads can't be mapped back; "chart elements" named as a use case; localization "approximate" (§2.3) | confirmed | https://platform.claude.com/docs/en/build-with-claude/vision-coordinates ; limitations list on the Vision page |
| 3 | Claude Opus 4.7 announced 2026-04-16 with 2576 px images and `xhigh` effort (Sources) | confirmed | https://www.anthropic.com/news/claude-opus-4-7 |
| 4 | Opus 5 = $5/M input → 4784-token image ≈ $0.024/call; Batches API −50 %; Sonnet 5 members ≈ 5× cheaper (§8.5) | confirmed (with note) — Sonnet 5 is $2/M input (the launch "introductory" price is now permanent), $1/M in batch, so ≈ 5× vs Opus is right; Opus 5 batch = $2.50/M | https://platform.claude.com/docs/en/about-claude/pricing |
| 5 | WebPlotDigitizer: last GitHub release 4.7 (Feb 2025); frontend AGPL-3; "AI Assist" cloud backend closed-source (§4.1) | confirmed (release dated 2025-02-13; README: "WPD frontend is distributed under GNU AGPL v3", "Automeris 'AI Assist' and other related cloud based systems are closed source and owned by Automeris LLC") | https://github.com/automeris-io/WebPlotDigitizer/releases ; https://github.com/automeris-io/WebPlotDigitizer/blob/master/README.md |
| 6 | Drevon, Fursa & Malcolm 2017 (Behav Modif 41(2):323–339): 3 596 points, ICC ≈ .99, r ≈ .99 (§4.1, §5, §7.3) | partially confirmed / exact coefficients unverifiable — abstract confirms 3 596 points, 168 series, 36 graphs, 18 studies and "high levels of intercoder reliability and validity"; the ≈ .99 values are behind the paywall and were not verified from a primary source (text annotated) | https://journals.sagepub.com/doi/10.1177/0145445516673998 ; https://pubmed.ncbi.nlm.nih.gov/27760807/ |
| 7 | CharXiv-R: Gemini 3 Pro 81.4 / Gemini 2.5 Pro 69.6 / Claude Sonnet 4.5 68.5 / GPT-5.1 69.5; GPT-5 (thinking) 81.1 %, MMMU 84.2 % (§2.1) | confirmed | https://simonwillison.net/2025/Nov/18/gemini-3/ (Google's table); OpenAI GPT-5 launch numbers via https://www.datacamp.com/blog/gpt-5 (openai.com blocked the fetch) |
| 8 | PlotExtract (arXiv 2503.12326): Claude 3.5 Sonnet > GPT-4o; MAE and validation P/R (§2.2, §0, §6) | **corrected** — published set: MAEx 2.8 %, MAEy 2.4 %; synthetic set: MAEx 3.0 %, MAEy 0.94 % (draft had the y-values swapped); validation recall 81.8 % (published) / 88.9 % (synthetic), precision 100 % (draft said ≈ 82–85 % / ≈ 85 %) | https://arxiv.org/html/2503.12326 (Table 1) |
| 9 | EpiCurveBench (arXiv 2605.27195): Gemini 2.5 Pro high 52.3, GPT-5.2 45.4, Claude Opus 4.5 42.0, Qwen3-VL 27.5, OneChart 9.4, TinyChart 11.4; bars 72.7 vs lines 47.5 (Gemini); high effort +11.8 / +0.7 / −1.5; Claude High+Code ≈ 3× cost ($82.41 vs $29.60); dominant tool failure = inaccurate cropping (§2.1, §6) | confirmed (Table 2 + text) | https://arxiv.org/pdf/2605.27195 |
| 10 | ExChart (arXiv 2606.29808, CHI 2026): 3 600 pairs; adaptive MAPE; ExChart-7B 4.87 %, Gemini 2.5 Flash 6.72 %, GLM-4.5V 5.94 %; bars 3.78–5.66 %; Qwen2.5-VL-7B base; two-stage coordinate-perception → table training (§2.1, §3) | confirmed | https://arxiv.org/html/2606.29808v1 |
| 11 | ChartArena (arXiv 2606.01348): Gemini 3.1 Pro 59.2, Kimi K2.5 54.8, Qwen3.5-35B 54.1, RRVF-7B 36.0, ChartMoE 8.5, ChartCoder 12.6, TinyChart 2.9 (§2.1, §3) | confirmed (EN mAP_high column of Table 2) | https://arxiv.org/html/2606.01348v3 |
| 12 | Self-Ensembling (arXiv 2605.27298): +2.7 % to +23.1 % relative RMS-F1; Spearman −0.34/−0.37 (rel-MAD vs error); early stopping keeps 99 % of gain at ≈ 16 samples; < $15 per benchmark (§2.2, §6) | confirmed | https://arxiv.org/html/2605.27298 |
| 13 | Plant-science pipeline (bioRxiv 2026.02.17.706322 v2): single Claude Opus 4.6 agent, paper-level ICC 0.838, TOST ±2 pp, ≈ $0.37/paper (§2.2, §5) | confirmed ($17 for 46 papers ≈ $0.37/paper; the "0.05–2.17 pp" aggregate figure was not re-checked) | https://www.biorxiv.org/content/10.64898/2026.02.17.706322v2 (biorxiv blocked direct fetch; abstract via search) |
| 14 | PyMuPDF 1.26: `page.get_drawings()` items `'re'`/`'l'`/`'c'` with `rect`, `fill`, `color`; `Page.cluster_drawings()`; `page.get_image_info()` (§4.2) | confirmed by running PyMuPDF 1.26.7 locally (item types `['re','l','c','l']`; `cluster_drawings(clip=None, drawings=None, x_tolerance=3, y_tolerance=3, final_filter=True)`) | https://pymupdf.readthedocs.io/en/latest/page.html |
| 15 | Cochrane Handbook §6.5.2 "endorses" Wan/Luo/Shi/McGrath box-plot estimators (§7.2) | **corrected** — §6.5.2.5 (IQR → SD) and §6.5.2.9 (missing means) cite Wan et al. 2014 and Bland 2015; Luo 2018 / Shi 2020 / McGrath 2020 are not named there; wording changed from "endorses" to "describes" | https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-06 |
| 16 | Wan et al. 2014: mean ≈ (Q1+m+Q3)/3, SD ≈ IQR/1.35 (η(n) → 2·Φ⁻¹(0.75) = 1.349) (§7.2) | confirmed (formula check; matches Wan 2014 §2/§3) | https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-14-135 |

Not re-checked in this pass (left as written, already flagged in the text where secondary): aggregator CharXiv numbers for Opus 4.7/4.8/Sonnet 5 (**[unverified secondary]**), Material Database Agent per-model MAEs, PlotPick figures, ReFocus/PixelCraft deltas, ChartBench 59.2 %, metaDigitise details, Gemini 3 Pro "pixel-precise" marketing claim.
