# 04 — Existing automated / LLM-assisted systematic-review extraction and meta-analysis tools (2023–2026)

*Research brief for Canopy. Written 2026-08-14. Sources at the end; numbers are quoted from the cited papers/pages unless flagged as approximate.*

## 0. TL;DR

- **Nobody has shipped what Canopy is building.** Commercial SR platforms (Covidence, DistillerSR, Laser AI, Nested Knowledge, Rayyan, Elicit) extract *text/table* fields with a human accept/reject loop; the ones that mention figures (Elicit, Nested Knowledge "High Fidelity Mode") describe reading figures qualitatively, and independent evaluations (Bianchi 2025) found Elicit "unable to extract information from figures". No SR tool does calibrated numeric digitization of bar/line charts with error bars end-to-end with provenance. Research prototypes that come closest are Manalyzer (VLM converts "tabular figures" to markdown tables; no coordinate calibration), AutoForest (papers → forest plot, text/tables, Claude Sonnet 4.5 + R `meta`), and generic chart-to-table VLM tools (PlotExtract, PlotPick, ExChart, self-ensembling) that report ~5% coordinate error and mostly ignore error bars.
- **Accuracy picture:** categorical/string fields 90–99% for frontier models; **numeric outcome data is the weak spot** (Yun 2024: GPT-4 exact-match 65.5% binary / 48.7% continuous; Li 2026 RSM: statistical-results recall 21–76%; Sallam 2025: o3 75%; Tan & D'Souza 2026: full effect-size tuples F1 ≈ 0.0–0.22). Best workflow-level results (otto-SR 93.1%, RTI SWAR 91.0% vs human 89.0%, Gartlehner 96.3%) all involve either LLM-as-judge scoring, human verification, or protocol-defined single-value fields — not raw figure digitization.
- **What demonstrably raises accuracy:** (1) two independent extractors + cross-critique/adjudication (Mayo collaborative-LLM: hallucinations 2.5% → 0.25%, accuracy 0.89/0.90 → 0.94; multi-model consensus: error 9.6–11.9% → 6.7%/1.5% true errors); (2) quote/evidence grounding ("self-proving" in Manalyzer: hit-rate 65.9 → 71.3; + checker → 77.7); (3) hierarchical extraction (locate relevant span first, extract second: 44.6 → 65.9); (4) protocol-specific, customized prompts (+14.8% recall vs generic; self-reflection alone ~+1.8%); (5) schema constraints + programmatic range/consistency checks; (6) human verification of a single-pass LLM extraction (SWAR).
- **Failure modes to design against (frequency-ordered from the literature):** omission/“unknown” (87.8% of errors in Li 2026; 5.5% missed in SWAR); wrong-value extraction incl. wrong arm/timepoint/outcome (Yun: 34–113 per task, dominant qualitative cause), role/direction reversal (15.5% of spurious tuples), cross-analysis binding drift (21.6% of tuple errors), hallucinated numbers (Yun: 12–38 per task; Helms Andersen: 4% confabulation), SD/SE/median confusion, arithmetic/derivation errors (Manalyzer Level-3 hit rate ≤ 3.4%), pre-vs-post confusion, negative-sign loss (“−−”), group-size denominators, PDF parsing errors (0.67% in otto-SR; the majority of GPT-4 plug-in errors in Konet 2024), and multiplicity/double-counting of sub-experiments (Shokraneh taxonomy).

---

## 1. Landscape table

| Tool / study | Type | What it automates | Reported accuracy (vs humans) | Figures? | Verification design | Notes |
|---|---|---|---|---|---|---|
| **otto-SR** (Cao et al., medRxiv 2025; ottosr.com) | research pipeline (Toronto) | search → screening (GPT-4.1) → extraction (o3-mini-high, PDF→markdown via Gemini 2.0 Flash) → meta-analysis | Extraction 93.1% (91.1–97.0) vs dual human 79.7% (69.1–91.0) vs Elicit 74.8%, n=4,559 data points/495 studies/7 reviews; blinded panel sided with otto-SR in 69.3% of disagreements with original authors. Reproduced 12 Cochrane reviews in 2 days; "matched" pooled estimates had overlapping 95% CI with Cochrane in all reviews; expanded analyses changed significance in 3 reviews | **No** — "limited to the main text… did not extract data from supplementary tables or figures"; studies with figure-only data were dropped from otto-SR analyses | Single LLM pass; LLM-as-judge accuracy scoring; blinded human adjudication of discrepancies for gold standard; error classes: inaccessible 0.83%, parsing 0.67%, both wrong 0.49% | Strongest end-to-end evidence; explicitly calls for "raw numerical data from figures" as future need |
| **Gartlehner 2024** (RSM 10.1002/jrsm.1710) | evaluation | Claude 2 extraction of 16 element types from 10 RCTs | 96.3% (160 elements); test-retest 96.9% / 95.0%; Claude found 21 human errors | No | Prompted per element; human gold with correction | Small convenience sample |
| **Konet 2024** (RSM 10.1002/jrsm.1732) | evaluation | Claude 2 vs GPT-4 (+ PDF plug-in) | Claude 2 96.3%, GPT-4 68.8% (most errors from PDF plug-in) | No | — | PDF parsing is a first-order error source |
| **RTI SWAR** (Annals Int Med 2025, Kahwati/Gartlehner) | prospective study-within-reviews | Claude 2.1/3 Opus/3.5 Sonnet first pass + human verification vs dual human | Accuracy 91.0% (90.4–91.6) vs 89.0% (88.3–89.6); concordance 77.2%; missed 5.5% vs 6.2%; misallocated 1.8% vs 1.9%; major errors 2.5% vs 2.7%; −41 min/study; 9,341 elements/63 studies/6 reviews | No | Human verifier over LLM draft; blinded adjudicators | "String data better than numerical/mixed" |
| **Khraisha 2024** (RSM 10.1002/jrsm.1715) | evaluation | GPT-4 screening + extraction, multi-language | Extraction "moderate" after adjusting for chance agreement (κ≈0.65) | No | Human-out-of-the-loop | Warns raw accuracy inflated by imbalance |
| **Schopow 2023** (JMIR Med Inform) | evaluation | ChatGPT vs humans on parameter extraction (κ) | High concordance except study design/clinical task/impact | No | Fleiss/Cohen κ | Early, small |
| **Yun et al. 2024** (MLHC; arXiv 2405.01686; github hyesunyun/llm-meta-analysis) | benchmark | Zero-shot numeric ICO extraction, 120 PMC RCT articles / 699 ICO prompts (183 binary, 516 continuous), 7 LLMs → fixed-effect MA | GPT-4 exact-match **0.655 binary, 0.487 continuous**; all LLMs <50% on continuous; MSE of log-OR 0.10, of SMD 0.29; remdesivir MA reproduced pooled log-OR, CI off by 0.1 | No (text only) | None | Best documented error taxonomy for numeric extraction (see §3) |
| **Li, Mathrani, Susnjak 2026** (RSM 10.1017/rsm.2025.10066; arXiv 2507.15152) | benchmark | 58 RCTs / 6 MAs; GPT-4o-mini, Gemini-2.0-flash, Grok-3; 4 prompting strategies | Precision high; recall for statistical results 21–76%; customized prompts +14.8% recall, ensemble +5.9%, self-reflection +1.8%; errors: 87.8% missing, 10.3% wrong value, <2% unit/overgeneralization | No | Rule-merged 3-model ensemble | Proposes tiering: statistical results = "human judgment essential" |
| **Tan & D'Souza 2026** (arXiv 2602.10881) | diagnostic benchmark | Schema-constrained tuple extraction, 52 papers/5 domains, GPT-5.2 & Qwen3-VL-235B | Single-property F1 0.45–0.62; multi-atom tuples 0.20–0.30; highest-arity (var, role, method, effect size) **F1 0.00**; long-context multi-doc regime collapses to 0.05 | Not the focus | — | Names role reversal (15.5%), cross-analysis binding drift (21.6%), multi-instance compression, numeric misattribution |
| **Sallam et al. 2025** (RSM 10.1017/rsm.2025.10030) | evaluation | GPT-4o / o3 on 290 RCTs (CBT-insomnia SR) | 72.6% / 75.3% accuracy; strings > numerics; "cannot yet replace humans for numeric data" | No | — | |
| **Mayo collaborative LLMs** (JAMIA 2025, Khan et al.) | workflow | GPT-4-turbo + Claude-3-Opus independent extraction, cross-critique on discordance | Concordant 87% (test); collaborative accuracy 0.94 vs 0.89/0.90 single; hallucinations 0.25% vs ~2.5%; cross-critique fixed 51% of discordant | No (abstracts) | Dual-LLM + cross-critique | Direct evidence for Canopy's dual-extract design |
| **Multi-model consensus** (Frontiers AI 2026, MS reports) | workflow | Claude 3.7 Sonnet + Gemini 2.0 Pro + o3-mini, ≥2 must agree | Single 9.6–11.9% error → consensus 6.7% total / 1.48% true; humans ~2% | No | Majority vote, abstain otherwise | Abstention counted as not-error — beware |
| **Helms Andersen 2025** (CESM 10.1002/cesm.70036) | evaluation | Elicit & ChatGPT as second reviewer, 30 articles | P/R/F1 88.8–92.2%; 7 confabulations = 4% of data | No | Propose AI replaces 2nd human extractor, human reconciles | |
| **Bianchi 2025** (CESM 10.1002/cesm.70033) | evaluation | Elicit vs human, 20 RCTs | "equal" 20.7%, partially equal 45.7%, deviating 4.3%, "more" 29.3% | **No** — "unable to extract information from figures" | — | |
| **Elicit** (elicit.com) | commercial | Screening + column extraction with quotes; "Systematic Review" product | Self-reported 94–99% extraction; independent: 81.4% vs 86.7% human (n.s.), 91% high-accuracy mode, 74.8% in otto-SR; 70/90 prompts >87% but unstable across accounts/versions (quotes matched 46%, reasoning 30%; high-accuracy mode 77% value match) | Marketing now says it reads figures/tables; independent evidence contradicts (2025) | Quote grounding to source section | Non-determinism across accounts is a documented reproducibility risk |
| **Covidence AI Extraction suggestions** | commercial | 14 study-characteristic fields with supporting quotes | Precision 92.2–99.3%, recall 98.8–100% (own curated eval); "better than typical human 80–85%" | No; **no outcome/numeric results at all** | Mandatory per-field accept/reject | |
| **DistillerSR AI (Smart Evidence Extraction)** | commercial | GenAI extraction from "tables and free text", full auto or HITL; NIST AI RMF | None public | Tables yes, figures not claimed | HITL review before acceptance; auditability | |
| **Laser AI** (Evidence Prime) | commercial | Suggests values from text and tables, links to source; extraction time −53% | None public | Tables yes | Human validate/edit/accept | One of 2 tools in Cochrane's 2026 platform study |
| **Nested Knowledge** | commercial | Core Smart Tags (PICO etc.), Adaptive Smart Tags (OpenAI), "High Fidelity Mode… pulls data directly from full-text tables and figures… diagrams and images" | None public | Claims figures qualitatively | Human tag review | Other Cochrane platform-study tool |
| **Rayyan** | commercial | Agentic screening; "AI-Assisted" prefill and "AI-Automated" extraction with human oversight (2025) | None public | Not claimed | HITL | |
| **EPPI-Reviewer** | academic platform | GPT-4 screening/coding integrations; one report 95% sens/100% spec screening, 86% no errors in coding | Screening only | No | Human validation | |
| **RobotReviewer / Trialstreamer** (Marshall, Wallace) | academic | RoB assessment, PICO tagging, living updates (RobotReviewer LIVE) | RoB "moderate" vs humans (Tian 2024); zero/few-shot LLM RoB "weak" (2024) | No | — | Not a numeric extractor |
| **MetaMate** (educational SR, CHI 2026 EA) | academic tool | LLM extraction of ed-research coding | P 81–96%, R 90–100%, F1 88–96% | No | Human review | |
| **MetaMind** (medRxiv 2025) | prototype | Multi-agent (LLaMA-3, Mistral, Qwen2) extraction of trial endpoints → scripted Bayesian NMA vs manual | Benchmarked vs manual NMA (numbers not retrieved) | No | Multi-model | |
| **Manalyzer** (arXiv 2505.20310) | prototype (multi-agent) | End-to-end MA: retrieval, hybrid review screening, VLM table/figure→markdown, hierarchical extraction, self-proving, checker, code-gen analysis | Benchmark 729 papers/3 domains/>10k points; hit-rate ablation: baseline 44.6/45.0/0 (L1 text/L2 table+image/L3 calc) → +hierarchical 65.9/55.3/0.5 → +self-proving 71.3/67.9/1.1 → +checker 77.7/70.6/3.4; +50% hit rate over LLM baseline | Yes-ish: VLM converts "tabular figures" to markdown; non-tabular figures summarised as bullets; **no axis calibration/digitizing** | Quote-citing (self-proving) + independent checker with feedback loop | Level-3 (needs calculation) remains near zero even with all mechanisms |
| **meta-pipe** (arXiv 2606.28363, 2026) | open-source pipeline | 10 stages, Claude Opus 4 + Haiku 3.5, R meta/metafor/netmeta, Quarto manuscript, GRADE, 12 overclaim patterns | **No validation reported**; single-pass extraction, uncalibrated confidence | **No** — "cannot reliably extract from figures, supplementary materials, or complex multi-page tables"; documented wrong-table / wrong-population / wrong-endpoint failures | Range/consistency checks; human gates | Notes 0.95^k error compounding |
| **AutoForest** (IBM/DCU/UCL, arXiv 2606.02403, 2026) | prototype + UI | Papers → ICO suggestion → events/n or mean/SD/n extraction with free-text justification → R `meta` forest plot (RR/MD, fixed/random, I², τ²) + RoB2 macros | Fully automatic >80% extraction accuracy; expert+tool 90.2%; students+tool 86.4% (> manual experts); time halved (n=8) | Text/tables via layout analysis (206 tables, TEDS/numeric-cell metrics); **not chart digitization** | Human edit; explanation text; admits no visual source highlighting | Closest published analogue to Canopy's output stage |
| **MetaSyn** (arXiv 2606.17041, 2026) | benchmark | 442 Nature-portfolio MAs; retrieval + end-to-end generation | Best inclusion recall 52.7% despite 90.9% retrieval ceiling — screening/eligibility is the bottleneck | — | — | Not extraction-focused |
| **PlotExtract** (Polak & Morgan 2025, arXiv 2503.12326) | figure method | Zero-shot CoT prompts to multimodal LLM to extract points | >90% precision, ~90% recall, x/y error ≈5% or lower on extractable plots | Yes (scatter/line) | Prompt chain, no calibration | Not SR-specific |
| **PlotPick** (arXiv 2605.06021, 2026) | open-source figure tool | Batch VLM chart→table for SR/MA; 6 VLMs (Claude Haiku 4.5/Sonnet 4.6, Gemini 3 Flash/3.1 Flash Lite, GPT-5.4 nano/mini) | ChartX recall 88.5–95.8% vs DePlot 70.5%; PlotQA RMSF1 86.3–99.1% vs 94.2%; box plots 83–97% vs 24%; upscaling +3 pp | Yes (bar/line/box/hist); grouped/stacked harder; small models make scale/magnitude errors | None beyond prompt | No error-bar or WebPlotDigitizer/human comparison |
| **ExChart** (arXiv 2606.29808, 2026) | benchmark + fine-tune | 3,600 unlabeled charts, 33,757 values | Best models ~4.9–5.9% adaptive MAPE (GLM-4.5V 5.94, ExChart 4.87); Claude not evaluated | Bar/line/scatter/pie/radar; **no error bars** | Coordinate-geometry pretraining | Systematic biases: zero-height bars corrupt neighbours, large tick magnitudes → digit errors |
| **Self-ensembling VLMs** (arXiv 2605.27298, 2026) | figure method | Sample ≤20 tables, align rows/cols (ANLS), per-cell median, MAD uncertainty | +1.5 to +8 RMSF1; value errors 19–40%, missing points 22–42% remain; cannot fix consistent hallucinations | Synthetic charts only; no error bars | Cell-wise consensus + uncertainty | Direct template for Canopy's figure-vote step |
| **WebPlotDigitizer v5** (automeris) | manual/CV tool | Manual axis calibration; bar-chart/blob/averaging-window auto-detect | (human reference tool used by Cisneros analysts) | Yes, manual | — | No AI, no error-bar mode |

Guidance / governance: **Cochrane–Campbell–JBI–CEE position statement 2025** (RAISE): AI allowed if it does not compromise rigour, always with human oversight, synthesists remain responsible, disclosure required; Cochrane platform study of Laser AI + Nested Knowledge with results due late 2026. **PRISMA-trAIce** (JMIR AI 2025): reporting checklist for AI in SLRs (tool identity/version, human–AI interaction, performance evaluation, limitations). **CHART** (2025) covers chatbot health-advice studies (not extraction). **"From promise to practice"** (2025) lists evaluation pitfalls: human reference standards contain errors (up to 63% of human-extracted studies have ≥1 error), contamination, stochastic variability, and recommends blinded discrepancy adjudication and a major/minor/inconsequential error taxonomy.

---

## 2. Does any tool extract numbers FROM FIGURES automatically, end-to-end?

Short answer: **not in a validated, calibrated, provenance-preserving way for meta-analysis inputs.**

- **Commercial SR platforms**: Covidence (metadata only), DistillerSR SEE and Laser AI (text + tables), Rayyan (text) do not claim chart digitization. Elicit's 2026 marketing says it reads "charts, tables, diagrams", but the two 2025 independent evaluations (Bianchi; otto-SR's Elicit arm at 74.8%) report figure inability / low accuracy, and reproducibility across accounts is poor. Nested Knowledge's "High Fidelity Mode" claims to read figures/images but publishes no numeric accuracy.
- **otto-SR and meta-pipe explicitly exclude figures**; otto-SR treated figure-only studies as unextractable and asks publishers to release "raw numerical data from figures".
- **Manalyzer** is the only end-to-end MA prototype that ingests figures, but via VLM "figure → markdown table" conversion (fine for figures that are really tables), and its Level-2 (table+image) hit rate tops out at 70.6% with Level-3 (derived values) at 3.4%.
- **AutoForest** is end-to-end to a forest plot but its evidence comes from text/tables (document layout analysis; TEDS/numeric-cell fidelity reported on 206 tables).
- **Chart-to-table research** (PlotExtract, PlotPick, ExChart, ChartLlama/DePlot lineage, self-ensembling) shows frontier VLMs read unlabeled bar/line/box charts with ~5% relative coordinate error, but: benchmarks are mostly synthetic; **none evaluate error-bar length extraction**, SE-vs-SD-vs-CI legend disambiguation, or group→condition→timepoint mapping; and consistent hallucinations survive self-ensembling.

Implication for Canopy: the figure digitizer (task #6: VLM + CV hybrid, explicit axis calibration, error-bar detection, zoomed crops) is genuinely novel relative to the field, and its validation against WebPlotDigitizer human values in Cisneros is the key evidence to publish. Expect residual ~3–5% coordinate error even when calibration is right; the bigger risks are semantic (which bar is which group; whether whiskers are SE or SD or 95% CI; log axes; broken axes; stacked/grouped decomposition).

---

## 3. Failure modes catalogued in the literature

| # | Failure mode | Evidence | Typical cause |
|---|---|---|---|
| F1 | **Omission / spurious "unknown"** | 87.8% of errors in Li 2026; GPT-4 "unknown for inferable" 20 (binary) and 142 (continuous) in Yun; SWAR missed 5.5% | value requires inference (split n across arms), lives in figure/supplement, dense results sections ("multi-instance compression": recall 0.45 → 0.32 as instances rise 1–5 → 31+) |
| F2 | **Wrong value: wrong arm / comparator / outcome / timepoint** | dominant qualitative cause of GPT-4 wrong numbers (Yun); meta-pipe "wrong table, wrong population, wrong endpoint"; Manalyzer "NO2 instead of SO2" | many similar outcomes; pre vs post; completers vs ITT n; several group sizes per ICO |
| F3 | **Role / direction reversal** | 15.5% of spurious tuples are exact IV/DV swaps (Tan & D'Souza) | surface order of mention; for Canopy: old−young vs young−old sign, "improvement" scored as lower error |
| F4 | **Cross-analysis binding drift** | 21.6% of high-arity tuple errors | variables reused across analyses/tables; pairs value from one table with method from another |
| F5 | **Hallucinated numbers** | GPT-4 12 (binary) / 38 (continuous) fabricated where reference unknown (Yun); 4% confabulation (Helms Andersen); ~2.5% single-LLM vs 0.25% dual (Mayo); Spinner 2020 events hallucinated in Yun case study | ungrounded generation; pressure to fill schema |
| F6 | **SD/SE/CI/IQR/median–mean confusion; unit/scale** | "rare but present" (Yun); Shokraneh "interpreting SE as SD"; <2% unit errors (Li) | legend/caption ambiguity; error bars unlabeled; %, degrees, normalized units |
| F7 | **Derived-value arithmetic errors** | Manalyzer Level-3 ≤3.4% hit; Yun: division/subtraction for group sizes; Consortium 2021 required calculation | LLM does math in-context instead of code |
| F8 | **Sign loss / formatting** | Yun: negative numbers typeset "--" ignored → "unknown"; PDF hyphenation | parser artifacts |
| F9 | **PDF parsing / OCR / table-structure errors** | Konet: most GPT-4 errors from PDF plug-in; otto-SR parsing 0.67%; Shokraneh "misread/misconverted"; ExChart digit-count errors on large ticks | layout, rotated tables, scanned PDFs |
| F10 | **Multiplicity / double counting** | Shokraneh "miscollected (multiplicity)", "deep error" (values from cited papers); Cochrane guidance on unit-of-analysis | multiple experiments/cohorts per paper; same cohort across papers; sub-experiments sharing a control group |
| F11 | **Non-determinism / version drift** | Elicit: 90% value agreement across accounts, 46% quote, 30% reasoning; high-accuracy mode 77% value; Gartlehner test-retest 96.9/95.0 with different items wrong 5/6 times | stochastic sampling; vendor model updates |
| F12 | **Screening/eligibility misapplication upstream** | MetaSyn: ≤52.7% inclusion recall; otto-SR found 54 extra eligible studies | protocol interpretation; not Canopy's core, but the "which sub-experiment qualifies" analogue applies |
| F13 | **Reference-standard errors** | otto-SR: adjudicators sided with LLM 69.3%; Claude found 21 human errors vs 6 own; humans 65.8–85.5% accurate; ≥1 error in up to 63% of human-extracted studies | validation must adjudicate disagreements rather than assume human = truth |

---

## 4. Verification strategies with measured effect

| Strategy | Evidence of effect | How Canopy should implement |
|---|---|---|
| **Two independent extractors + adjudication** (mirrors Cochrane dual extraction; Buscemi 2006: double 14.5% vs single+verify 17.7% errors) | Mayo dual-LLM: 0.94 vs 0.89/0.90, hallucination 0.25% vs 2.5%; consensus of 3 models 9.6–11.9% → 6.7% (1.5% true) | Two extractors that differ in *both* model and *modality/route* (e.g., Opus on page images vs Sonnet on layout text; figure digitizer CV path vs VLM path); adjudicator sees both + source crop, must cite quote/bbox |
| **Cross-critique on discordance** | Fixed 51% of discordant items (Mayo) | Adjudicator gets each extractor's rationale + evidence, not just values |
| **Quote / evidence grounding ("self-proving")** | Manalyzer +5.4 pts and reduces non-existent values; Elicit/Laser AI/Covidence all attach quotes; AutoForest users valued the "thought process" | Every scalar carries page, bbox, verbatim quote (or figure crop + pixel coordinates); a deterministic checker confirms the quote exists in the page text and the number appears in the quote |
| **Hierarchical / two-stage extraction (locate then extract)** | +21 pts hit rate (Manalyzer 44.6 → 65.9); Li 2026 recall gains from customized prompts | Stage 1: candidate-span/figure finder per outcome/group; Stage 2: focused extraction on the located span/crop with zoom |
| **Independent checker with feedback loop** | +6.4 pts (Manalyzer 71.3 → 77.7) | Verifier agent scores accuracy + consistency, can request re-extraction; cap iterations |
| **Protocol-customized prompts** | +14.8% recall vs generic; self-reflection only +1.8% | Compile the user protocol (groups, outcomes, sign convention, preferred timepoints, unit) into extractor prompts and schema enums; do not rely on generic "check your work" |
| **Ensembling / self-consistency at cell level** | +1.5–8 RMSF1 on charts; MAD gives uncertainty | For figures: N samples of VLM read at different zoom/crops + CV estimate; median + MAD; flag high-MAD cells for adjudication |
| **Schema constraints & programmatic checks** | meta-pipe range/consistency checks; AutoForest RoB2 rule macros; Tan & D'Souza recommend "neural–symbolic" enforcement | Typed JSON schema (n integers, SD>0, SE<SD, CI brackets contain mean, group sizes sum to N, timepoints exist in design); all arithmetic (SE→SD, t/F→d) in code, never by the model |
| **Explicit "not reported"/abstain paths** | Both Claude 2 and GPT-4 recognized missing elements (Konet); consensus pipeline abstains when <2 agree | Allow `not_reported` with reason; treat abstention as an item for human review, not as correct |
| **Human verification of LLM draft** | SWAR 91.0% vs 89.0%, −41 min/study; AutoForest 80% → 90.2% with expert edits | UI shows crop + quote next to each value with one-click accept/override; log overrides for audit |
| **Blinded adjudication for validation** | otto-SR method; "From promise to practice" recommendation | For Cisneros validation, disagreements between Canopy and the human table should be adjudicated against the PDF, not auto-scored as Canopy errors |
| **Report per PRISMA-trAIce / RAISE** | Position statement requires disclosure + human oversight | Emit an audit bundle: model IDs, prompt versions, per-value provenance, override log |

---

## 5. Prioritized list of failure modes Canopy's verification layer must catch

1. **Group / condition / timepoint mis-mapping** (F2, F4) — the most common *wrong-number* cause; catch with protocol-compiled enums, per-value "which figure panel / table row / sentence" provenance, and a mapping-consistency check across the two extractors.
2. **Sign / direction errors** (F3) — catastrophic for pooled d; enforce that raw values are stored as reported per group and the sign convention is applied once in code; verifier re-derives direction from the quote ("older adults showed larger aftereffects").
3. **Error-bar type and dispersion confusion** (F6) — SE vs SD vs 95% CI vs IQR; require an explicit `dispersion_type` field with quote/legend evidence; if unknown → flag, do not guess; sanity checks (SD/mean plausibility, SE < SD, CI width vs n).
4. **Omission of figure-only data** (F1) — the failure every prior system has; ensure the locator step enumerates all figures/tables mentioning the outcome and records an explicit decision per candidate.
5. **Hallucinated or unsupported numbers** (F5) — deterministic quote-in-page check; for figures, pixel-level re-measurement of the reported bar/whisker by CV; disagreement > tolerance → adjudication.
6. **Derived-value arithmetic** (F7) — never let the LLM compute d, SE, or SD-from-SE; verify inputs, compute in `numpy`, log formula.
7. **Sample-size denominators** (F1/F2) — n per group at the analysed timepoint (dropouts, completers); require quote; cross-check that n_young + n_old matches the reported total.
8. **Double counting / unit-of-analysis** (F10) — sub-experiments sharing a control group, multiple papers on the same cohort, multiple outcomes per dataset; explicit `dataset_id` with dedup rules and a human-visible list.
9. **Digitization scale errors** (F9/ExChart) — axis calibration failures, log axes, broken axes, digit-count errors on large ticks; recompute values from pixel coordinates with independently detected axis ticks (OCR) and require the two paths to agree within tolerance.
10. **PDF text-layer artefacts** (F8/F9) — minus signs, ligatures, superscripts, rotated tables; extract from page images as one route and from text layer as the other.
11. **Run-to-run instability** (F11) — pin model versions, temperature 0 where possible, store all raw responses; report agreement rate between extractors as a quality metric.

## 6. Design requirements / lessons for Canopy

- Treat **numeric outcome extraction as Tier-3** (Li 2026): default to dual independent extraction + adjudication + provenance for every scalar; string metadata can be single-pass.
- **Diversity of routes beats repetition of one route**: page-image VLM read vs text-layer read; CV digitizer vs VLM digitizer; different model families. Ensembling identical calls cannot fix consistent hallucinations (self-ensembling paper).
- **Locate → extract → prove → check** (Manalyzer's ablation is the cleanest evidence that each stage adds accuracy).
- **Ground truth for validation must be adjudicated**, since human WebPlotDigitizer values also carry error; report agreement, not just "accuracy vs humans", and treat systematic disagreements as findings.
- **Publish figure-digitization accuracy** as a first-class metric (relative error of means and error-bar lengths, and downstream Δd) — nothing comparable exists in the literature.
- **Provide the audit bundle** PRISMA-trAIce/RAISE ask for: tool + model versions, prompts, per-value provenance, human overrides, and a discrepancy log; this is also what makes Canopy usable in Cochrane-style workflows.
- **Expect compounding**: meta-pipe's 0.95^k warning — measure end-to-end pooled-estimate error (Δd, ΔCI, ΔI²) on the Cisneros case, not only per-field accuracy.

## Sources

- otto-SR preprint (medRxiv 2025): https://www.medrxiv.org/content/10.1101/2025.06.13.25329541v1 ; manuscript PDF: https://ottosr.com/manuscript.pdf ; announcement: https://ottosr.com/blog/announcement/
- Gartlehner et al. 2024, RSM: https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1710
- Konet et al. 2024, RSM: https://onlinelibrary.wiley.com/doi/10.1002/jrsm.1732
- Kahwati/Gartlehner et al. 2025, Annals of Internal Medicine SWAR: https://www.acpjournals.org/doi/10.7326/ANNALS-25-00739 ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC13091441/
- Khraisha et al. 2024, RSM: https://onlinelibrary.wiley.com/doi/10.1002/jrsm.1715
- Schopow et al. 2023, JMIR Med Inform: https://medinform.jmir.org/2023/1/e48933
- Yun et al. 2024, MLHC (arXiv 2405.01686): https://arxiv.org/abs/2405.01686 ; code: https://github.com/hyesunyun/llm-meta-analysis
- Li, Mathrani, Susnjak 2026, RSM "What level of automation is good enough?": https://arxiv.org/html/2507.15152 ; https://mro.massey.ac.nz/server/api/core/bitstreams/ac6c526f-1705-49d7-882a-b552020f4c8f/content
- Tan & D'Souza 2026, "Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis": https://arxiv.org/abs/2602.10881
- Sallam et al. 2025, RSM (GPT-4o/o3): https://pubmed.ncbi.nlm.nih.gov/41626895/ ; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12823200/
- Khan et al. 2025, JAMIA collaborative LLMs: https://pubmed.ncbi.nlm.nih.gov/39836495/ ; preprint https://pmc.ncbi.nlm.nih.gov/articles/PMC11469465/
- Multi-model consensus (Frontiers in AI 2026, MS): https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1658575/full
- Helms Andersen et al. 2025, CESM: https://onlinelibrary.wiley.com/doi/full/10.1002/cesm.70036
- Bianchi et al. 2025, CESM (Elicit vs humans): https://onlinelibrary.wiley.com/doi/full/10.1002/cesm.70033
- Elicit feasibility study, RSM 2025: https://www.cambridge.org/core/journals/research-synthesis-methods/article/using-elicit-ai-research-assistant-for-data-extraction-in-systematic-reviews-a-feasibility-study-across-environmental-and-life-sciences/C97DAEC70C3173A260F0B12E729E7250 ; Hilkenmeier et al. 2025: https://journals.sagepub.com/doi/10.1177/08944393251404052 ; Elicit self-evaluation: https://elicit.com/blog/how-we-evaluated-elicit-systematic-review
- Covidence AI extraction suggestions: https://support.covidence.org/help/ai-feature-extraction-suggestions ; responsible automation: https://support.covidence.org/help/covidences-approach-to-responsible-automation-ai
- DistillerSR Smart Evidence Extraction press release (2026-04): https://www.pharmiweb.com/press-release/2026-04-08/distillersr-launches-the-industrys-most-advanced-genai-capabilities-for-extracting-scientific-liter ; https://www.distillersr.com/products/distillersrai
- Laser AI: https://www.laser.ai/solutions ; Nested Knowledge AI docs: https://about.nested-knowledge.com/docs/artificial-intelligence-in-nested-knowledge/ ; Core Smart Tags: https://about.nested-knowledge.com/docs/core-smart-tags/
- Rayyan data extraction (2025): https://blog.rayyan.ai/2025/08/25/rayyan-data-extraction-from-manual-to-fully-automated-insights/
- RobotReviewer: https://github.com/ijmarshall/robotreviewer ; LLM RoB weak (2024): https://pubmed.ncbi.nlm.nih.gov/39176994/
- MetaMate (SREE 2024 / CHI EA 2026): https://eric.ed.gov/?id=ED663552 ; https://dl.acm.org/doi/10.1145/3772363.3798755
- MetaMind (medRxiv 2025): https://www.medrxiv.org/content/10.1101/2025.08.04.25332893.full.pdf
- Manalyzer (arXiv 2505.20310): https://arxiv.org/abs/2505.20310
- meta-pipe (arXiv 2606.28363): https://arxiv.org/abs/2606.28363
- AutoForest (arXiv 2606.02403): https://arxiv.org/pdf/2606.02403
- MetaSyn benchmark (arXiv 2606.17041): https://arxiv.org/html/2606.17041v5
- MetaBeeAI (Ecological Informatics 2026): https://www.sciencedirect.com/science/article/pii/S1574954126002190
- PlotExtract (Polak & Morgan 2025): https://arxiv.org/abs/2503.12326
- PlotPick (arXiv 2605.06021): https://arxiv.org/html/2605.06021
- ExChart (arXiv 2606.29808): https://arxiv.org/html/2606.29808v1
- Self-ensembling VLMs for chart extraction (arXiv 2605.27298): https://arxiv.org/html/2605.27298
- WebPlotDigitizer docs: https://automeris.io/docs/digitize/
- Cochrane/Campbell/JBI/CEE AI position statement 2025: https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.ED000178/full ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12603384/
- Cochrane AI platform study (Laser AI, Nested Knowledge): https://www.cochrane.org/about-us/news/cochrane-announces-selected-ai-tools-innovative-platform-study
- PRISMA-trAIce (JMIR AI 2025): https://ai.jmir.org/2025/1/e80247
- CHART statement (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12320030/
- "From promise to practice" evaluation pitfalls (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12703319/
- Shokraneh 2026, classification of LLM extraction errors: https://farhadinfo.medium.com/classification-of-llm-errors-in-data-extraction-for-systematic-reviews-and-factors-affecting-the-4549f5c68467
- Systematic review of LLM extraction performance (J Biomed Inform 2026): https://pubmed.ncbi.nlm.nih.gov/42501879/
- Buscemi 2006 single vs double extraction: https://pubmed.ncbi.nlm.nih.gov/16765272/ ; Mathes 2017 methodological review: https://link.springer.com/article/10.1186/s12874-017-0431-4 ; Cochrane Handbook ch. 5: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-05
- Cisneros et al. (Nature Human Behaviour 2026 / bioRxiv 2024): https://www.biorxiv.org/content/10.1101/2024.07.02.601091v1
