# Plan amendments after the design-review panel (binding; read with your task brief)

Source: `docs/superpowers/specs/2026-08-15-design-review.md` (findings numbered there). These amend
`2026-08-15-canopy-implementation.md`. Where the two conflict, THIS file wins.

## A. Data model / protocol (Task 1)
- `Candidate` gains: `status: Literal["found","not_on_these_pages","ambiguous"]` (numerics nullable when not found);
  `raw_value_semantics: Literal["higher_more_construct","higher_more_error","signed_direction","unknown"]`;
  `analysis_metric: Literal["endpoint","change_from_baseline","baseline_corrected","percent_of_perturbation","unknown"]`;
  `error_bar_scope: Literal["between_subject","within_subject_normalized","unknown"]`;
  `whisker_definition: Literal["min_max","iqr_1_5","percentile_5_95","sd","se","ci","unknown"]` (box plots);
  for test statistics: `design: Literal["independent_t","one_way_between","mixed_main_effect","interaction","ancova","paired","welch","unknown"]`, `df1`, `df2`, `tails`, `p_kind: Literal["exact","less_than","greater_than","ns"]`;
  for reported effect sizes: `standardizer: Literal["pooled_sd_between","dz_paired","glass_delta","partial_eta","unknown"]`, `reported_ci_low/high`;
  `n_quote` per group; `page_corrected: bool` (grounding moved it ±1 page).
- `DatasetSpec` gains `cluster_id` (paper sha12), `shared_control: bool`, `exposure_order: Literal["first","repeated","counterbalanced_collapsed","unknown"]`, `all_groups_listed: list[GroupSpec]` + `chosen_pair_rationale`.
- `StatsSettings`: `route_precedence` DEFAULT = `["text_mean_sd","table","text_mean_se_ci","figure","test_statistic","p_value","reported_d"]`; add `late_window_sd: Literal["paper_reported_block","block_closest_to_end","mean_of_block_sd"]="paper_reported_block"`, `ci_to_sd_dist: Literal["auto","z","t"]="auto"` (t(n−1) when n<100), `primary_analysis_includes: list = ["auto_accept","accept_with_note"]`, `shared_control_strategy: Literal["split_n","combine_arms","keep_first","keep_all_flagged"]="split_n"`, `multi_group_policy: Literal["extremes","combine_matching","closest_to_definition","needs_human"]="closest_to_definition"`, `digitization_variance: Literal["off","sensitivity","primary"]="sensitivity"`, `one_row_per_paper: bool=True` (primary analysis aggregates/splits so each paper contributes one row per outcome; sensitivity uses all rows).
- Profiles: `metafor` = Hedges g, `variance="borenstein"` (LS), REML, z CI, PI `z`; `cisneros2024` = Cohen d, `hedges_olkin_df`, REML, PI `HTS`, hakn False. Footer must enumerate estimator, variance formula, τ² method, HK, PI df, I² definition, k(papers)/k(datasets), rows excluded (needs_human) count.
- Static test: no agent output schema contains derived-statistic fields (`d`, `g`, `hedges`, `cohen`, `smd`, `effect_size`, `pooled_sd`).
- Every schema enum has an `unknown`/`not_reported` member where a value might be absent.

## B. LLM client (Task 2)
- `LLMCall` records: request_id, stop_reason, cache_creation/read tokens, effort, prompt_version, schema hash, served model, image hashes, cell key (opaque string), latency, cost. Provider seam: `LLMProvider` protocol with `AnthropicProvider`, `ReplayProvider` (fixtures), `FakeProvider` (tests); `LLMClient(provider=...)`.
- Check `stop_reason` everywhere; `max_tokens` truncation → raise `TruncatedOutput` (retry once with 2×); use streaming for `max_tokens > 16000`; `fallbacks:"default"` only for `claude-fable-5`; budget = per-call reservation, plus `max_usd_per_paper` guard in the pipeline.
- Cache `document`/image blocks only when ≥2 same-model readers will reuse them; warm-up call before fan-out.

## C. Stats additions (new small Task 2b — implement inside Task 9's dispatch, in `canopy/stats/`)
- `smd_from_t(t, n_a, n_b, df=None, design=...)`, `smd_from_f(F, n_a, n_b, df1, df2, design=...)`, `smd_from_p(...)`: refuse (raise `NotConvertible`) unless design ∈ {independent_t, one_way_between} and (df ≈ n_a+n_b−2 within ±2 or df missing → flag) — the record keeps the value with `route="not_convertible"`.
- `combine_groups(m1,sd1,n1,m2,sd2,n2)` (Cochrane 6.5.2.10), `split_control(n, k)`; `mean_sd_from_median_iqr(median,q1,q3,n)` (Luo 2018 mean, Shi 2020 SD, Wan 2014 fallback), `mean_sd_from_five_number(...)`; `egger_test(yi, sei, n_a=None, n_b=None)` → when n given use Pustejovsky–Rodgers covariate √((nA+nB)/(nA·nB)); `leave_one_out(yi, vi, method)`; funnel data helper. Extend R fixtures where cheap (escalc for combine, metafor regtest with sqrt(1/n) predictor).

## D. Mapper (Task 4)
- Mapper lists ALL groups (ages) and the chosen A/B pair with rationale (`multi_group_policy` from protocol); flags supplements/related files; emits a decision (`relevant`/`irrelevant`+reason) for EVERY figure and table detected by ingestion (deterministic Fig/Table reference scan supplies the list); records `exposure_order`, `shared_control`, `analysis_metric` per outcome source, `error_bar_type` AND `error_bar_scope` with quotes.
- Cross-check (Sonnet) also enumerates source locations; diff → disagreements listed; error-bar type and group mapping require independent agreement (second agent), else `needs_human`.

## E. Extractors (Tasks 5, 7)
- Output envelope `{status, ...}` (see A); text sources get text-only context (page text + table cells) plus page image only for tables/figures; grounding: numeric normalization both sides, table cell-level check, ±1 page then whole-document fallback (`page_corrected=True`), unicode minus/thin spaces.
- Test-statistic extractor emits `design/df1/df2/tails/p_kind` and is told to prefer per-timepoint contrasts; reported d includes `standardizer` and reported CI.

## F. Digitizer (Task 6)
- Provide model-driven zoom via `tool_runner`: client tools `crop_image(x0,y0,x1,y1, zoom)` (returns image block), `list_regions()`, `overlay_points(points)` (returns image with marks) — stable tool list, ≤ 8 tool calls, all logged. Build order: Path D (read-out) + overlay-verify + zoom tools first; then shared CV core (axis lines, ticks, OCR of tick labels via tesseract at 4×, LS fit); Path C (VLM coords + snap, adaptive window = 0.25×bar width, min 3 px); Path B (raster CV bar/point/cap detection by color) as an independent voter when segmentation succeeds; Path A (vector) last but required.
- Dual tolerance for votes: means within max(2% axis range, 0.5 tick); error half-lengths within max(10%, 0.5 px); auto_accept only if implied |Δd| < 0.1 across routes and digitization SE < 10% of sampling SE; flag bars < 8 px.
- Time-series: honor `late_window_sd` rule; record which rule was applied.

## G. Verification (Task 8)
- Vote within route (modality × model family); if the two text extractors disagree, add a third cheap candidate (Sonnet, other variant) before adjudication; verifier model ≠ winning candidate's model; verifier receives the WHOLE PDF (file_id) + the candidate and must also answer "is there a better source anywhere in the paper?"; ≤ 2 re-opens per cell.
- Orientation determined once per (outcome, measure) from `raw_value_semantics` + direction; two agents must agree.
- Extra checks: SD≈0, unit mismatch, figure n ≠ analyzed n, point undercount vs n, mixed metric within a paper.

## H. Report (Task 11)
- Primary forest = rows with confidence ∈ `primary_analysis_includes`; `needs_human` rows drawn hollow and excluded from pooling, counted in footer; route glyph per row; override marker. Also: PRISMA-style flow JSON + figure (files → unique papers → eligible → datasets → included), exclusion table (paper, stage, reason enum, quote, decider), methods paragraph templated from the manifest (no LLM numbers), funnel plot, leave-one-out table, sensitivity set (exclude figure-derived; exclude test-stat-derived; include needs_human; HK vs z; ± digitization variance; by analysis_metric; one-row-per-paper vs all rows) as JSON + small multiples PNG.
- Footer enumerates all statistical conventions (see A).

## I. Server/UI (Task 12)
- Security baseline: uploads saved as `<sha256>.pdf` after magic-byte + size checks; artefacts served by manifest id (path-traversal safe); bind 127.0.0.1 by default; per-run bearer token; API key never echoed; escape all PDF/LLM text; ingestion in a subprocess with timeout.
- Review workflow: per cell — override value(s) with justification, exclude dataset, eligibility override, "re-extract with hint", mark reviewed; immutable `overrides.jsonl`; overrides survive `--resume` and re-pool; queue sorted by |Δpooled| impact. Guided protocol form (stats under Advanced), "draft protocol from one sentence" (LLM), dry-run mapper on 2–3 papers. Deferred to v2: `--batch`, heavy interactive SVG (keep simple hover/click).

## J. Validation (Task 13)
- Dev/held-out split of papers (dev used for prompt tuning; held-out touched once at a frozen git tag); synthetic matplotlib figure corpus with known truth (`validation/synthetic/`) for digitizer accuracy; second toy protocol (any small OA corpus) in CI to prove generality; metric = agreement with blind-adjudicated truth (human and tool disagreements adjudicated by inspecting the source), plus mis-mapping/omission counts, bucket calibration; example scripts take `--paper/--run`, run offline from cassettes.

Rulings (controller): accepted all blockers/majors above; rejected making a second vendor mandatory (provider seam only);
kept simple hover SVG (deferred rich interactivity); `--batch` deferred. Cost if wrong: extra plumbing, no correctness risk.
