# Hardening after the first user runs (2026-08-16)

Two live runs (3 papers each) exposed failures that a correct reading could not survive. This document
lists every failure observed, its root cause from the run records, and the fix — with the rule that
**no single component may be trusted for a number that decides an effect size**.

## Failure catalogue (with evidence from `runs/20260816-075831…` and `validation/out/run_cisneros_dev`)

| # | Failure | Root cause (from stage files) | Consequence |
|---|---|---|---|
| F1 | Cressman Fig 3A y-axis calibrated as a **5-unit** span (true: −5..45) | tesseract read the labels 45/35/25/15 as **4/3/2/1** (trailing digit cut off by the label band crop); the ladder 4,3,2,1 is evenly spaced so `pair_ticks`/`fit_axis` reported rmse 0.2 px — a perfect fit of wrong numbers | pixel routes (C/B/A) produced no value; `value_outside_axis` (error) fired on a CORRECT value; cell → needs_human |
| F2 | Only the two **Opus** read-outs voted (same model family ⇒ "single route") | adaptive plan stops at 2 read-outs when they agree; the two are opus/direct + opus/ticks_first; the Sonnet read-out never ran | agreement across independent readers never established ⇒ cannot auto-accept even when right |
| F3 | Bock one-armed SD whiskers ⇒ half-length disagreement discarded agreed means (yesterday) | per-cell dual tolerance | fixed in ad17509 (per-quantity) — keep, add tests on Cressman/Bock records |
| F4 | `metric_mixed` warned on Cressman because late adaptation is an endpoint and the aftereffect a baseline-subtracted difference | check scoped per PAPER | penalises every paper whose outcomes are legitimately different metrics |
| F5 | Heuer & Hegele 2008: mapper chose page-4 text (no n, no dispersion) and Exp 2 (Fig 6) instead of **Fig 2a/2b (p5, SE bars, n=20/20)** which the published review used | source ranking does not prefer sources that carry dispersion; the verifier's "better source anywhere?" answer was not acted on; no dataset-choice rule "prefer the experiment the protocol's rules select" | two `not_convertible` rows on a paper with a perfectly usable figure |
| F6 | Fig 2a's x-axis is **target direction** (categorical); the outcome is the average across 8 points per series | digitizer reads ONE point at an x-hint; no "collapse across a categorical axis" mode | even with the right source the value could not be read |
| F7 | Buch 2003 died at $14.37 vs a $14 cap after extraction; nothing kept | per-paper cap raises mid-stage; no partial-row salvage; the cap is checked call-by-call | $14 bought nothing |
| F8 | Silent half-scale OCR calibration (synthetic corpus, yesterday) | `_band_nearest_axis` cut inside wide labels | fixed in ccadbad (0.6×) — same class as F1: OCR is fragile |
| F9 | UI: table shows sha ids, COST blank, page does not repaint after sleep, run forgotten on reload, X did nothing, Cancel "not found" during upload | rendering only from live events; no persisted run id; CSS override | operator cannot tell which paper is which |
| F10 | Cost $10–15/paper (before caching fix); 358 calls for 3 papers | ~90 read-out/overlay tool-loop calls per paper; no cache reads (fixed 07f394f) | affordability |

## Design principles for the fix
1. **Two independent witnesses for every axis calibration and every value.** A calibration is *usable* only when two of {OCR ladder, VLM tick reading (path C), read-out tick lists, vector text layer} agree on the tick VALUES within 2 % of span AND the implied span is consistent with the read-out values' magnitude; otherwise the pixel routes fall back to the VLM's own tick pixels and the calibration is marked `single_witness` (never auto_accept on it alone).
2. **Two model families minimum in every read-out plan** (opus + sonnet) before any adaptive stopping; a third read-out only on disagreement. Model family = independent route.
3. **Checks compare against evidence, not against a fit**: `value_outside_axis` uses the label range agreed by ≥2 witnesses; when calibration is single-witness the check downgrades to `warn` ("calibration unconfirmed"), never `error`.
4. **Truncated glyph detection**: any OCR label whose bbox touches the crop band edge is rejected; label bands are widened; a ladder whose values are all single-digit while the read-out values are two-digit is refused (magnitude sanity).
5. **Source selection prefers usable sources**: the mapper ranks sources by (has dispersion for both groups, has n, matches the outcome window) and lists why lower-ranked ones lost; the verifier's `better_source` answer, when it names a figure with error bars, re-opens extraction on that source (one hop, once).
6. **Categorical-axis mode**: when the mapper's source says the quantity is an average across a categorical axis (targets, directions, conditions), the digitizer reads ALL points per series and the code averages them; dispersion = mean of the per-point half-lengths (recorded as `collapsed_across_x`, `n_points`), row is at most `accept_with_note`.
7. **Budget: finish the cell, keep the rows.** The per-paper cap stops NEW cells (`budget_exhausted`, partial rows kept, remaining cells needs_human); the run cap likewise.
8. **Checks scoped correctly**: `metric_mixed` per (paper, outcome); flags on the row name the scope.
9. **UI truth**: author/year/page from the manifest; cost from the manifest; run id persisted (localStorage) and re-attached on load; a run finished while the tab slept re-paints from the manifest.
10. **Regression tests from the real records**: Cressman and Bock stage files (small JSON) become fixtures: after the fix, Cressman late adaptation must be `accept_with_note` or better with d within 0.02 of −0.224 and axis span 50 ± 1; Bock late adaptation `accept_with_note` with d within 0.2 of −1.676; the synthetic corpus must show 0 calibration failures.

## Out of scope (recorded): batch mode; second vendor; interactive SVG.
