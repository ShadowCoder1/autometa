# Canopy design review — panel synthesis (2026-08-15)

Reviewed: spec v1 against `docs/research/00-synthesis-and-decisions.md`, existing `canopy/stats` + `canopy/ingest`
code, and the user's verbatim requirements (perfect-accuracy forest plot from a folder upload; agents verifying and
voting on every part; image models zooming into graphs, multiple systems; not biased to Cisneros; the five deliverables).

## 1. Overall verdict

**Approve with mandatory changes (major revision of §3.2–3.5, §4, §8).** The architecture (mapper → heterogeneous
extractors → code checks → vote → adversarial verifier → adjudicator → stats in code) is right and general. Three
blocking gaps: (a) safeguards already decided in the synthesis were dropped (dev/held-out split, `raw_value_semantics`,
Path B, PRISMA/AI-use reporting, supplements); (b) unsafe statistical routes (reported-d first, un-gated t/F,
single-rule dependency handling); (c) correlated-error holes in verification (one mapper feeds every agent; verifier
never sees the whole paper; schemas cannot say "not present"). No re-architecture needed.

## 2. Deduplicated findings, ranked (A = accept, R = reject/modify)

### Blockers
1. **Test-statistic route accepts non-convertible F/t** (mixed-ANOVA main effect, interaction, Welch, paired). — **A**:
   schema gains `design`, `df1/df2`, `tails`, `p_kind`, quote; `smd_from_t/f/p` take df and refuse unless design is
   independent-t/one-way-between and df ≈ nA+nB−2 → `not_convertible` (kept for audit).
2. **Route precedence puts reported d/g first.** — **A**: order = M±SD → M±SE/CI → figure → t/F → exact p → reported
   d (last, only with a quote establishing pooled-SD between-groups d/g); reported d/CI become consistency checks
   (|Δd|>0.1 flag); record `standardizer` per row.
3. **Dependency handling = one rule ("first exposure").** — **A**: add `cluster_id`, `shared_control` strategy
   {split_n, combine_arms, keep_first, keep_all_flagged}, `exposure_order` moderator; implement `combine_groups()`,
   `split_control()`; primary = one row per paper per outcome, sensitivity = all rows; footer prints k(papers)/k(datasets).
4. **Single mapper feeds every agent; verifier sees only mapper-chosen pages; no omission recovery.** — **A**: verifier
   gets the whole PDF (file_id, citations) and asks "is there a better source anywhere?"; Sonnet cross-check enumerates
   source locations and diffs them; deterministic Fig/Table reference scan; mapper emits a decision per figure/table;
   whole-paper search before `not_reported`; mis-mapping and omission become eval metrics.
5. **Schemas force a value → hallucination.** — **A**: every extractor output is
   `{status: found | not_on_these_pages | ambiguous}` with nullable numerics; closed enums with `unknown`; static test
   that no agent schema has derived-statistic fields.
6. **Anti-Cisneros-bias guardrail is only "protocol from Methods".** — **A**: restore §11 of the synthesis into §4:
   ~12-paper dev set, held-out touched once per frozen git tag, synthetic matplotlib/R PDF corpus with known truth,
   second toy protocol in CI; dev and held-out numbers reported separately in every deliverable.
7. **`needs_human`/`accept_with_note` fate in the plot undefined.** — **A**: primary = auto_accept + accept_with_note;
   needs_human excluded, drawn hollow, counted in footer, in an automatic sensitivity plot; route glyph + override marker.
8. **Paper-ready outputs missing.** — **A**: `canopy/report/` emits PRISMA-flow JSON+figure, exclusion table (stage,
   reason enum, quote, decider), methods paragraph templated from RunManifest (PRISMA-trAIce/RAISE; LLM only rephrases;
   every number checked against `analysis.json`), study table CSV/XLSX/DOCX, funnel, leave-one-out, and a standard
   sensitivity set (exclude figure-/test-stat-derived/needs_human; HK vs z; ± digitization variance; by `analysis_metric`).
9. **Web-app security unspecified.** — **A**: uploads stored as `<sha256>.pdf` with magic-byte/size checks; artefacts
   served by manifest ID; bind 127.0.0.1 + per-run token; CSRF; never echo the API key; escape all PDF/LLM text;
   PyMuPDF/OCR in a subprocess with timeout.

### Major
10. Within-subject (Cousineau–Morey/Loftus–Masson) error bars inflate d. — **A**: `error_bar_scope` field; never
    auto-accept `within_subject_normalized` for SD recovery.
11. Late-window dispersion rule unspecified. — **A**: protocol `late_window_sd ∈ {paper_reported_block,
    block_closest_to_end, mean_of_block_sd}` (default paper's own value), recorded per row; closes open question §7.
12. Endpoint vs change-score mixed in one SMD. — **A**: `analysis_metric` field on Candidate/EffectSize; auto
    subgroup/sensitivity by it.
13. `profile: metafor` mislabeled (metafor SMD = Hedges g, LS variance). — **A**: redefine as g/LS/REML/z; keep
    `cisneros2024`; footer enumerates estimator, variance formula, τ² method, HK, PI df, I² definition.
14. Digitization variance in primary weights is non-standard. — **A**: primary without; sensitivity with (delta
    method through both mean and SD, or Monte-Carlo); footnote symbol on figure-derived rows.
15. Egger on SMD has inflated type-I. — **A**: use √((nA+nB)/(nA·nB)) covariate (Pustejovsky–Rodgers); add funnel.
16. Figure vote tolerance is axis-relative but error bars are 2–5% of the axis. — **A**: dual tolerance (half-lengths
    within max(10%, 0.5 px)); auto_accept only if implied |Δd|<0.1 and digitization SE < 10% of sampling SE; flag
    bars < ~8 px.
17. `raw_value_semantics` dropped from spec. — **A**: restore on every Candidate; orientation = f(direction,
    semantics) applied once in code; second agent must agree on both.
18. Box/IQR conversions. — **A**: `whisker_definition` field; only `min_max` feeds range formulas; Luo 2018 mean +
    Shi 2020 SD + skew screen; route tag `median_iqr`.
19. Multi-group papers (3 ages, two older cohorts). — **A**: `multi_group_policy ∈ {extremes, combine_matching,
    closest_to_definition, needs_human}`; mapper lists all groups + chosen pair + rationale; ages as moderator.
20. Supplements treated as separate papers. — **A**: restore supplement detection + `related_files` grouping; UI tagging.
21. Not all StudyMap fields dual-read. — **A**: error-bar type and group mapping require independent agreement (as
    orientation does); moderators marked unverified unless confirmed.
22. No model-driven zoom ("image models zooming into graphs"). — **A**: digitizer tools `crop_image(bbox, zoom)`,
    `list_regions()`, `overlay_points()` via `tool_runner`; stable tool list; bounded iterations; calls logged.
23. Path B (pure raster CV) dropped; "multiple systems" is Anthropic-only. — **A** (partial): restore Path B as an
    independent voter; state that heterogeneity = model family × prompt × zoom × path; `LLMProvider` seam with a
    non-Anthropic VLM behind a flag. **R** making a second vendor mandatory in v1 — say so in the limitations text.
24. Figure-path build order/assumptions (Path A depends on vector share; Path C's ±6 px seed error unmeasured). — **A**:
    see §4 below; adaptive snap window (fraction of bar width).
25. Cost/latency unmodeled; cache plan wrong for a single reader; budget overshoots. — **A**: per-agent table
    (model, effort, `max_tokens`, image tier, $/paper); `canopy estimate`; cache document blocks only for ≥2
    same-model readers; warm-up before fan-out; per-call reservation, `--max-usd-per-paper`; wall-clock in UI.
26. Refusal/truncation handling. — **A**: `fallbacks:"default"` only on Opus/Fable sync, client retry elsewhere;
    streamed calls with explicit `max_tokens` (≥16k mapper/adjudicator) and timeout; check `stop_reason` everywhere.
27. Resumability. — **A**: temp-then-rename; outputs keyed by (paper_sha, stage, protocol_hash, prompt_version,
    model_id); `--resume` refuses on hash mismatch unless `--fork`; budget-truncated cells → needs_human; SQLite + blobs.
28. Grounding false negatives (table order, unicode minus, page semantics, scans). — **A**: cell-level table
    grounding, numeric normalization both sides, ±1 page then whole-doc fallback (`page_corrected`), crop hash for figures.
29. Two-extractor "voting" is unanimity-or-dispute. — **A**: vote within route; third cheap candidate before
    adjudication; route = modality × model family; verifier differs from the winning candidate's model; ≤2 re-opens.
30. Observability / key-free tests. — **A**: `LLMCall` gains request_id, stop_reason, cache create/read tokens,
    effort, prompt/schema version, served model, image hashes, cell keys; cassettes keyed by content hash;
    synthetic figure corpus.
31. Review workflow under-specified. — **A**: manual value entry with justification, dataset exclusion, eligibility
    override, re-extract-with-hint, review-all/sign-off, immutable override log, overrides survive `--resume`,
    queue sorted by |Δpooled|, mapper decision list visible.
32. Protocol authoring for non-programmers. — **A**: guided form (stats under "Advanced"), LLM-drafted protocol from
    one sentence, dry-run mapper on 2–3 papers.

### Minor (all accepted; one rejected)
CI→SD via t(n−1) for n<100, level recorded; p-route needs test type + tails, rounding as interval, p rows in
sensitivity; HK beside z, PI suppressed k<4, τ²/I² CIs; extra consistency checks (SD≈0, unit mismatch, figure n ≠
analyzed n, point undercount); DOI-less key fallback; §4 metric = agreement with blind-adjudicated truth + bucket
calibration; orientation agents once per (outcome, measure); merge overlay-verify into the verifier unless paths
disagree; **defer** `--batch` and interactive SVG to v2; Fable startup probe; text-only context for text sources;
per-group `n` with its own quote; `canopy files purge`; narrative report templated + numerically grounded; example
scripts take `--paper/--run`, run offline from cassettes/synthetic corpus, choose examples by route; ground
author/year on page 1; process pool for CV/OCR; enumerate endpoints (override, audit.zip, cost report), zip upload.
**R**: "PI shown regardless of k" — not in the spec; implementation note only.

## 3. Changes to apply to the spec (by section; numbers refer to findings above)

- **§2 models/protocol**: 1, 3, 5, 10, 11, 12, 17, 18, 19; add `primary_analysis_includes`, `ci_to_sd_dist`,
  per-group `n` source; closed enums with `unknown`.
- **§3.0 ingest**: 20; subprocess ingestion; `canopy files purge`.
- **§3.1 mapper**: 4, 21; two-stage mapper option.
- **§3.2 extraction**: 1, 22, 23, 24; text-only context for text sources.
- **§3.3 verification**: 4, 16, 28, 29; expanded consistency checks; orientation once per outcome.
- **§3.4 effect sizes**: 2, 3, 14, 18; df/design gates.
- **§3.5 outputs**: 7, 8, 13, 15; footer enumerates all conventions.
- **§3.6 UI/CLI**: 9, 31, 32; audit-bundle/override/cost endpoints; defer batch + interactive SVG.
- **§3.7 cost/robustness**: 25, 26, 27, 30.
- **§4 evaluation**: 6; adjudicated-truth metric; mis-mapping/omission metrics.
- **§5 scripts**: offline via cassettes/synthetic corpus, `--paper/--run` args, examples chosen by route.
- **§7/§1**: resolve late-window and digitization-variance questions (11, 14); status-line reference now valid.

## 4. Build order (de-risk first)

1. **Data model + `LLMProvider` seam** (Anthropic/Replay/Fake), `LLMCall` logging, static schema test — everything
   depends on it.
2. **Stats additions** on the validated base: df-gated t/F/p, new precedence, `combine_groups`/`split_control`,
   Luo/Shi, Pustejovsky Egger, profile rename, LOO/funnel/sensitivity; extend R fixtures.
3. **Corpus survey script** (1 h): vector vs raster share, tick-label text, error-bar semantics — decides Path A/B
   investment before any digitizer code.
4. **Mapper + text/table extractors + grounding + code checks** on the dev set, with location diff and omission
   scan; measure mis-mapping/omission first (dominant failure mode).
5. **Path D + overlay + zoom tools** end-to-end, then the shared CV core (axes, ticks, OCR, LS fit), then Path B,
   Path C (after measuring seed error), Path A last.
6. **Vote / whole-PDF verifier / adjudicator / confidence** with cassettes; calibrate buckets on dev set only.
7. **Report layer**: primary + sensitivity forest, PRISMA flow, exclusion table, templated methods, tables.
8. **CLI + local web app** with security baseline and review workflow; batch/interactive SVG deferred.
9. **Held-out run under a frozen tag + synthetic-corpus/second-protocol CI**; then the five deliverable scripts.
