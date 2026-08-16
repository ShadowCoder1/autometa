# Validating Canopy against a published meta-analysis

Canopy's claim is that an agent can read a folder of PDFs and produce the same numbers a careful
human produced by hand. This directory is where that claim is checked, against
**Cisneros et al. (2024)**, a meta-analysis of ageing and upper-extremity sensorimotor adaptation
whose per-study extractions, analysis code and forest plots are public
([OSF s7h5e](https://osf.io/s7h5e/)).

Nothing in here is imported by the pipeline. `canopy/` never reads a gold file, and a test asserts
it: the tool has to earn its numbers.

---

## What is where

| path | what it is |
|---|---|
| `reference/cisneros2024/{late,aft,combined}_gsheet.csv` | **the gold data** — the human's per-dataset Cohen's d, 95 % CI and seTE, one row per (paper, experiment, measure) |
| `reference/cisneros2024/*.ipynb`, `*.Rmd`, `*.xlsx` | the human's own working: WebPlotDigitizer read-outs → SD → d, and the `meta::metagen` analysis |
| `reference/cisneros2024/SOURCES.md` | where each of those files came from |
| `reference/cisneros2024/join_overrides.csv` | hand-written joins for cases the matcher cannot make on evidence (currently empty) |
| `papers_oa/` | 10 open-access PDFs of included studies, plus `MANIFEST.csv` listing the 22 that are paywalled |
| `splits.json` | the frozen dev / held-out assignment of the 19 unique papers |
| `synthetic/` | a figure corpus with known truth, and the digitiser's accuracy on it |
| `out/` | everything the scripts produce (figures, tables, summaries) |
| `scripts/` | the scripts below |

---

## The corpus, and the dev / held-out split

The corpus is two folders joined and de-duplicated:

* `~/Downloads/Systematic Review/papers` — 79 files that are **9 unique papers** (the same PDFs
  saved many times over; `canopy.ingest.dedupe` collapses them by sha256, DOI and title);
* `validation/papers_oa` — 10 open-access PDFs fetched for the studies that were missing.

**19 unique papers** in total, against 50 gold datasets for late adaptation and 39 for
aftereffects — so most of the human's rows come from papers this machine does not have. The 22
paywalled ones are listed in `papers_oa/MANIFEST.csv`; dropping them into the papers folder is the
single biggest thing that would improve these numbers (see `docs/handoff.md`).

`splits.json` assigns each unique paper to `dev` or `heldout` by sorting on sha256 and
alternating. It depends only on the papers' bytes — not on file names, folder order or the day it
ran — and it is **frozen**: re-writing it re-shuffles the alternation, which is why
`--write-split` is a deliberate, separate command.

* **dev** (10 papers) is what prompts, thresholds and heuristics may be tuned on.
* **heldout** (9 papers) is run **once**, at the git tag `validation-heldout-v1`, after the code
  is frozen. Its numbers are the ones that mean anything; a held-out set you have looked at twice
  is a dev set.

```bash
# rewrite the split (rarely — it is frozen on purpose)
.venv/bin/python validation/scripts/run_cisneros.py --write-split

# see what a split selects, without spending anything
.venv/bin/python validation/scripts/run_cisneros.py --split heldout --list
```

---

## The scripts

Every script takes `--run RUN_DIR` and reads a finished run from disk. **None of them calls a
model** — a run directory already holds every candidate, verdict, quote, crop and conversion
chain, so an explanation is a read, never a re-extraction.

### 1 · `run_cisneros.py` — produce the run

```bash
# the dev split, live (needs credit)
.venv/bin/python validation/scripts/run_cisneros.py \
    --split dev --out validation/out/run_cisneros_dev \
    --budget-usd 60 --max-usd-per-paper 14 --concurrency 3

# the held-out split — ONCE, at the tag, after the code is frozen
git tag validation-heldout-v1
.venv/bin/python validation/scripts/run_cisneros.py \
    --split heldout --out validation/out/run_cisneros_heldout \
    --budget-usd 60 --max-usd-per-paper 14 --concurrency 3

# offline, no credit at all: the pipeline's own fake provider over the two fixture PDFs
.venv/bin/python validation/scripts/run_cisneros.py --demo --out validation/out/demo_run

# offline from recorded cassettes — ONLY once the WHOLE-PIPELINE ones exist. The agent tests'
# fixtures are not enough: the orchestrator asks each agent a different question (it passes the
# mapper's full Source list), so the pipeline needs its own recording. Until `pytest
# tests/test_pipeline_offline.py` has been run under CANOPY_LIVE=1 CANOPY_RECORD=1 (see
# "PENDING LIVE RECORDING" in docs/handoff.md), this raises MissingFixture rather than going
# live — which is the intended behaviour, not a bug.
.venv/bin/python validation/scripts/run_cisneros.py \
    --replay tests/fixtures/llm --papers tests/fixtures/pdfs \
    --out validation/out/run_bock_replay
```

`--resume` is the default and it is what makes this affordable: a stage with a file is skipped,
and every model call is content-addressed in `<out>/cache`, so a re-run after a budget stop pays
only for what is genuinely new. Raising `--max-usd-per-paper` and re-running is therefore cheap.

### 2 · `compare_manual_vs_auto.py` — the backward-validation scatter

```bash
.venv/bin/python validation/scripts/compare_manual_vs_auto.py --run validation/out/run_cisneros_dev
```

Manual d on x, Canopy's d on y, **95 % CI bars on both axes**, the identity line, a ±0.1 agreement
band, one marker shape per extraction route, and hollow markers for rows the tool held back for a
human. Writes `out/manual_vs_auto_<outcome>.png|svg`, `out/discrepancies_<outcome>.csv` and
`out/agreement_<outcome>.json` (Lin's CCC, MAE, RMSE, sign agreement, Bland–Altman bias and limits
of agreement, per-route and per-confidence-bucket calibration, and the pooled-estimate comparison).

**The join** is the fiddly part, so it is explicit and it reports itself. Auto rows are matched to
gold rows hardest-evidence-first — `override`, then `exact` (author + year + both group sizes +
experiment label), then `n_pair`, then `author_year` (only when exactly one row is left on each
side, because the spreadsheets and Crossref disagree about a couple of 2001/2002 papers) — and
every pair carries the rule that made it. A wrong pairing invents a disagreement or hides one, so
an unmatched row is always preferred to a guessed one; both unmatched sides are counted and listed.

**Classifying a discrepancy.** `discrepancies_<outcome>.csv` has three empty columns to fill in
while adjudicating, and enough context (route, page, quote, conversion chain, both CIs, both group
sizes) to do it without opening the run:

| `classification` | means |
|---|---|
| `tool_error` | the paper says something else; Canopy read it wrong |
| `human_error` | the paper says something else; the spreadsheet is wrong |
| `protocol_ambiguity` | both readings are defensible — the protocol does not say which block, phase or measure to use, and it should |
| `unresolved` | looked at, still not settled — say why in `adjudicator_note` |

Adjudicate by opening the paper, not by preferring either side: the metric amendment J asks for is
agreement with a **blind-adjudicated truth**, not agreement with the human.

### 3 · `replot_forest.py` — the auto forest beside the manual one

```bash
.venv/bin/python validation/scripts/replot_forest.py --run validation/out/run_cisneros_dev
```

Both forests are drawn by the same code (`canopy.report.forest`), sorted the same way, under the
same conventions, and **both sides' random-effects weights are recomputed** by
`canopy.stats.meta.random_effects` as `1/(vᵢ + τ²)` — precision, which is driven by sample size
*and* by how noisy the study was. Neither the spreadsheet's stored weights nor the run's are
copied, so the two plots are comparable by construction. Writes `forest_manual_*`, `forest_auto_*`,
`forest_side_by_side_*` and a JSON with k, pooled d, CI, τ², I² and the prediction interval for
both.

### 4 · `extraction_routes_figure.py` — where the numbers came from

```bash
.venv/bin/python validation/scripts/extraction_routes_figure.py --run validation/out/run_cisneros_dev
```

The share of datasets read from text, tables, figures, test statistics or a reported effect size,
with up to three **real** crops per route taken from the run's own provenance — a page crop with
the extractor's quote highlighted, or the digitiser's marks drawn on the figure. Writes
`extraction_routes.png|svg`, one figure per outcome, and `extraction_routes.csv` (one row per
dataset: route, confidence, pages, locators, conversion chain).

### 5 · the three example scripts — one number, end to end

```bash
.venv/bin/python validation/scripts/example_text_ms.py        --run validation/out/run_cisneros_dev
.venv/bin/python validation/scripts/example_figure_only.py    --run validation/out/run_cisneros_dev
.venv/bin/python validation/scripts/example_test_statistic.py --run validation/out/run_cisneros_dev
```

Each picks one cell that took its route, prints the whole story — what the mapper located, what
every extractor read (with its own quote), what the vote and the verifier decided, the conversion
chain, and the resulting d with its CI — and writes a figure showing the page crop with the quote
highlighted (or the digitiser's overlay) beside the chain and the effect size.

**What deliverable (c) actually shows today, and what it will show.** On the current run all three
scripts land on the *same* Bock 2005 cell (`b511dbb76fa6:d1 / late_adaptation`, `not_convertible`),
reached through two different fallback paths, plus one figure from the offline demo:

| script | what it shows now | what it will show |
|---|---|---|
| `example_figure_only.py` | Bock 2005 Fig. 1 — the four digitiser routes' read-outs and the ensemble's tolerance failure, via the **held-back fallback** | the same cell as a *resolved* row, once the one-armed-error-bar rule (handoff limitation 7a) is fixed |
| `example_test_statistic.py` | the same Bock cell — its seven F statistics and the ruling that **none** is admissible, via the **outranked fallback** | a paper whose t/F actually wins. This literature is overwhelmingly repeated-measures, whose F values do not convert to a between-groups d, so this may legitimately stay a fallback — and that is itself the finding, not a gap |
| `example_text_ms.py` | **nothing on this run** — neither Bock nor Wolpe prints its outcome means in the text, so it exits 2 and lists the routes the run does have. The committed figure `example_text_ms_offline_demo.*` is from the offline demo run: real PDF, real page crop with the quote highlighted, real arithmetic, **model answers from the FakeProvider** | **Kitchen 2021** (`validation/papers_oa/Kitchen_2021.pdf`), which is in the **dev** split and needs no tag — see the survey below |

So (c) is currently *one real cell seen two ways plus one demo*, not three independent papers.

**Which paper will carry the text-M/SD example — measured, not guessed.** Scanning all 19 unique
papers' text layers for `M = … SD/± …` near an outcome word (offline, seconds, no model):

| paper | split | M/SD-ish sentences | near an outcome word |
|---|---|---|---|
| **Kitchen 2021** | **dev** | 32 | **2** |
| Li 2021 | heldout | 14 | 2 |
| `paper_15209` (Cressman) | heldout | 11 | 2 |
| Hermans 2025 | heldout | 20 | 0 |
| Panouillères 2015 | dev | 15 | 0 |
| **Heuer & Hegele 2008** (`paper_1755470`) | heldout | **0** | **0** |

Heuer & Hegele 2008 — the obvious guess, and the one the task brief suggested — prints **no M ± SD
at all**; its numbers are entirely in figures. **Kitchen 2021 is the candidate**, and being in the
dev split it costs one more dev paper rather than the held-out tag.

Add `--paper path/to/one.pdf` to insist on a paper, `--outcome` / `--dataset` to narrow further.
If the run has no cell that took that route, the script says so and lists the routes the run *does*
have, rather than silently showing something else. `example_test_statistic.py` additionally falls
back to showing a t/F that was available but *outranked* (`route_precedence` puts test statistics
fifth) and labels it as the fallback it is.

### 6 · `synthetic_figures.py` — how wrong is the digitiser?

```bash
.venv/bin/python validation/scripts/synthetic_figures.py
```

A published figure has no ground truth; a matplotlib figure has it by construction. Twelve cases
vary chart form (bars, points, a time series), dispersion (SD/SE/CI), tick density, font family and
size, DPI, a negative range, sub-8-px bars and a noisy "scanned" render — each written as **both**
a PNG (what the raster routes see) and a one-page PDF (what the vector route sees).

Measured offline, with no model: **route A** (vector — the PDF's own drawing operators) and
**route B** (raster CV — axis lines, tick marks, tesseract on the tick labels, least-squares
calibration, colour-segmented bars/markers, edge-refined caps). Routes C and D are the VLM ones and
are **not** measured here; running them on this corpus once credit allows is a task in
`docs/handoff.md`.

Reported per route: read rate, MAE and median error as a **percentage of the axis range** (the same
unit as the digitiser's own 2 % vote tolerance), the error-bar half-length error, and — for route B,
which reports one — whether the digitisation `sigma` actually covers the truth 95 % of the time.
Detection failures are counted separately from read-out errors, because a harness that pairs two
series to one detected blob reports a huge "error" that is really a missed detection.

---

## Second protocol (generality)

Amendment J asks for a second, small review to show that nothing in the pipeline is specific to the
ageing question. `examples/protocols/clinical_vs_control_adaptation.yaml` changes everything a
protocol controls — a clinical-vs-control contrast, different outcomes (`adaptation_extent`,
`retention`), different moderators, and the `metafor` profile (Hedges' g, Borenstein variance,
normal prediction interval) instead of `cisneros2024` (Cohen's d, Hedges–Olkin df variance, HTS
interval). It runs on two OA PDFs that are already in this repository:

```bash
mkdir -p validation/papers_oa/toy_clinical
cp validation/papers_oa/BinyaminNetser_2023.pdf validation/papers_oa/Cornelis_2022.pdf \
   validation/papers_oa/toy_clinical/

.venv/bin/python -m canopy.cli run \
    --papers validation/papers_oa/toy_clinical \
    --protocol examples/protocols/clinical_vs_control_adaptation.yaml \
    --out validation/out/run_toy_clinical --budget-usd 25 --max-usd-per-paper 12
```

**Status: PENDING LIVE RUN.** The protocol loads, resolves the `metafor` profile and is asserted to
be genuinely different from the ageing one in `tests/test_validation_scripts.py`; running it needs
credit. Expect roughly $8–25 for the two papers at the costs measured below.

---

## What the first live run found

Three dev papers, 2026-08-16, on the real corpus. Everything below is in `out/`.

| | |
|---|---|
| **Anguera 2010** | correctly **excluded** at the mapping stage — it tested only young adults, and the mapper quoted the sentence that says so |
| **Bock 2005** | 2 cells, **both `needs_human`** |
| **Wolpe 2020** | 2 cells, **both `needs_human`** |
| pooled | **k = 0** for both outcomes; 4 rows held for review |
| the join | still matched all three cells to their gold rows — Bock late `−1.676`, Bock aftereffect `−0.5369`, Wolpe late `−0.374` |

So the honest headline is: **the tool found the right studies and the right cells, and then refused
to put a number in any of them.** That is the design working — but it means the backward-validation
scatter has no points yet, and the auto forest has no rows.

**Why, for Bock 2005 late adaptation** — the most useful thing this run produced. Four digitiser
routes read the group means off Fig. 1 as **31.2 – 32.8 deg** (older) and **10.6 – 13.4 deg**
(younger): agreement to within 3 %, implying d ≈ −1.5 against the human's −1.676. Every ensemble
then came back `ambiguous` because the **error-bar half-lengths** disagreed by 5 – 10 deg against a
~1.1 deg tolerance — because Bock draws only *one arm* of each SD whisker (upward for the old
group, downward for the young). Amendment F's dual tolerance is applied to the whole cell, so a
disagreement about the dispersion discards the means as well. One-armed error bars are common in
the literature precisely because they declutter overlapping series.

**And the test statistics:** all seven of Bock's F values were correctly ruled **inadmissible** —
mixed main effects tested against a different error term, and interaction terms — which is
amendment C's convertibility gate doing exactly its job. `example_test_statistic.py` shows it.

---

## What it costs

Measured on this corpus, live, on 2026-08-16 — not an estimate:

| | |
|---|---|
| the whole 3-paper run | **$26.25**, 353 unique model calls, ~50 min wall clock |
| Bock 2005 (5 pages, 2 figures, 50 candidates) | **$10.20** |
| Wolpe 2020 (11 pages, 4 figures, 79 candidates) | **$14.94** |
| Anguera 2010 (excluded at mapping) | $0.50 |
| mapping one paper | $0.27 – $1.81 |
| input tokens over the first 109 calls | **1 099 996**, of which **0 were cache reads** |

Two things follow.

1. **Budget $10–15 per paper**, not the $2–5 first estimated. A 19-paper run is therefore a
   $150–250 job, not a $15–40 one. Set `--max-usd-per-paper` accordingly: a paper that hits its cap
   ends as `error` and contributes **nothing**, so a cap that is too low buys you the cost without
   the result.
2. **Prompt caching is not paying off.** Zero cache reads across a whole run means every reader
   re-sends the page images and the PDF at full price. That is the single biggest cost lever in the
   system and it is a pipeline issue, not a validation one — recorded in `docs/handoff.md`.

Re-running is cheap: `--resume` skips finished stages and the content-addressed disk cache under
`<out>/cache` means an already-answered question is never paid for twice. Raising a cap and
re-running costs only the calls that had not been made.

---

## Tests

```bash
.venv/bin/python -m pytest tests/test_validation_scripts.py -q
```

**42 tests**, entirely offline: the gold reader (**every row of both spreadsheets must carry
`n_old`, `n_young`, TE and seTE** — the regression that made this test exist is below), the column
aliases, the join (including that one gold row is never claimed twice and that an override beats
every automatic rule), the agreement statistics (CCC is 1 only for the identity and is punished by
a scaled reading; a control test feeds the gold rows back in as if they were the tool's and asserts
CCC 1.0 / MAE 0.0), the split's determinism and its freedom from absolute paths, the synthetic
corpus including the log-axis case, and every script driven end to end against a real run directory
built by the pipeline's own `FakeProvider`.

> **The regression worth knowing about.** `late_gsheet.csv` names the column `N_young`;
> `aft_gsheet.csv` names it `N_yng`. Reading one spelling left all 40 aftereffect gold rows with
> `n_young = None`, which made `_n_distance` return `None` and every join rule that needs the group
> sizes silently unreachable — no error, just a column of blanks in the committed table. The
> aliases now live in one place (`COLUMN_ALIASES`), a missing required column raises
> `GoldSchemaError` at load time, and a test asserts completeness on both sheets.
