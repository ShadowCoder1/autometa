# Handoff

Everything that is not finished, in the order it is worth doing. Written 2026-08-16, at the end of
the implementation plan (`docs/superpowers/plans/2026-08-15-canopy-implementation.md`).

---

## 0 · Rotate the API key. Today.

The key in `.env` was **copied from the Wren project's server `.env`**. It is a working key that
belongs to another deployment, so Canopy's spending is currently indistinguishable from Wren's, and
a leak from either project compromises both.

1. Create a **new key** in the Anthropic console, scoped to this project, with a spend limit —
   `$150/month` is comfortable given the costs below.
2. Copy it to the clipboard from the console, then paste it straight into the file:
   ```bash
   printf 'ANTHROPIC_API_KEY=' > /Users/sritejpadmanabhan/Projects/canopy/.env
   pbpaste >> /Users/sritejpadmanabhan/Projects/canopy/.env
   ```
   The key never appears in a terminal argument, a shell history entry, or this document.
3. **Revoke** the Wren key if Wren has already been moved to its own, or leave it and simply stop
   sharing it.

`.env` is git-ignored and the key is never printed by any part of Canopy (a test plants a key and
asserts it never reaches the output), but a shared key is a shared blast radius.

---

## 1 · What has been spent so far

Every live call made while building this, from the task reports in
`.superpowers/sdd/2026-08-15-canopy-implementation/`:

| task | what was paid for | cost |
|---|---|---|
| 1–3 | LLM client live verification (cache hit, `count_tokens`) | $0.046 |
| 4 | mapper fixtures on Bock 2005, recorded twice | ≈ $1.36 |
| 5 + 7 | text and statistics extractor fixtures (7 calls) | $0.4116 |
| 6b | digitiser tool-loop fixtures (10 turns, opus) | $0.6856 |
| 8 + 9 | verification and stats — nothing live (account out of credit) | $0.00 |
| 10 + 11 | orchestrator and report — nothing live | $0.00 |
| 12 | web UI — nothing live, none needed | $0.00 |
| 13 + 14 | the Cisneros validation run (3 papers, 353 calls, two invocations) | **$26.25** |
| | **total** | **≈ $28.75** |

---

## 2 · Recordings that are still pending

Ten tests skip visibly, each printing its own recording command. The account **has credit again**
(verified 2026-08-16), so these can be recorded now. Do them in this order — the pipeline
end-to-end one reuses everything the others record.

```bash
cd /Users/sritejpadmanabhan/Projects/canopy

# 1 · text extractor — 4 skips, ~$0.15
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_extract_text.py -q

# 2 · verifier and orientation — 4 skips, ~$0.40
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_verifier.py -q

# 3 · digitiser: ticks_first tail, VLM coordinates, overlay verification — 1 skip, ≈$1.40
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_digitizer.py -q

# 4 · the whole pipeline over Bock 2005 from fixtures — 1 skip, ≈$2.00–2.80
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_pipeline_offline.py -q

# then confirm the suite has NO skips left, offline
.venv/bin/python -m pytest -q
```

**Total ≈ $4.00–4.75.** Two things to know before starting:

* the four text-extractor tests are currently wrapped in a temporary skip; delete the wrapper once
  their fixtures exist (noted in `.superpowers/.../progress.md`, Task 5+7);
* the pipeline's requests are **not** byte-identical to the ones the agent tests recorded — the
  orchestrator passes each agent the mapper's full `Source` list rather than a hand-picked subset —
  so step 4 will write **new** fixture files rather than reuse the earlier ones. That is expected.

**Zero replay skips is the gate before merging `build/v1` to `main`.**

---

## 3 · Finish the validation

The scripts, the split and the figures are done (`validation/README.md`). What is left is running
them over more of the corpus.

### 3a · The dev split

```bash
.venv/bin/python validation/scripts/run_cisneros.py \
    --split dev --out validation/out/run_cisneros_dev \
    --budget-usd 80 --max-usd-per-paper 14 --concurrency 3

.venv/bin/python validation/scripts/compare_manual_vs_auto.py  --run validation/out/run_cisneros_dev
.venv/bin/python validation/scripts/replot_forest.py           --run validation/out/run_cisneros_dev
.venv/bin/python validation/scripts/extraction_routes_figure.py --run validation/out/run_cisneros_dev
.venv/bin/python validation/scripts/example_text_ms.py         --run validation/out/run_cisneros_dev
.venv/bin/python validation/scripts/example_figure_only.py     --run validation/out/run_cisneros_dev
.venv/bin/python validation/scripts/example_test_statistic.py  --run validation/out/run_cisneros_dev
```

The three papers already in that directory are done and will replay free; the **seven remaining
dev papers** at the measured $10–15 each ≈ **$70–105**. `--resume` and the disk cache mean a stop
is never wasted: re-run the same command and it continues.

### 3b · The held-out split — once, at the tag

Do this **after** the code is frozen and nothing else will change. The point of a held-out set is
that it is looked at once; a held-out set you have run twice is a dev set.

```bash
git tag validation-heldout-v1
.venv/bin/python validation/scripts/run_cisneros.py \
    --split heldout --out validation/out/run_cisneros_heldout \
    --budget-usd 80 --max-usd-per-paper 14 --concurrency 3
# …then the same five analysis scripts, with --run validation/out/run_cisneros_heldout
```

9 papers ≈ **$90–135**.

**One thing to expect on the way.** All three `example_*.py` scripts currently land on the same
Bock 2005 cell (two of them through their fallback paths); the text-M/SD one has nothing to show
because neither Bock nor Wolpe prints its outcome means. An offline scan of all 19 papers' text
layers says the paper that will carry it is **Kitchen 2021** (dev split, no tag needed) — *not*
Heuer & Hegele 2008, which prints no M ± SD at all. The table is in `validation/README.md`.

### 3c · Adjudicate the discrepancies

`validation/out/discrepancies_<outcome>.csv` has three empty columns
(`classification`, `adjudicated_truth`, `adjudicator_note`) and everything needed to fill them in
without opening the run. Amendment J's metric is agreement with a **blind-adjudicated truth**, so
open the paper and decide what it actually says — do not default to either the human's answer or
the tool's. The four classifications are documented in `validation/README.md`.

### 3d · The second protocol (generality)

`examples/protocols/clinical_vs_control_adaptation.yaml` — a clinical-vs-control review with
different outcomes, different moderators and the `metafor` profile. Commands are in
`validation/README.md`, "Second protocol"; ≈ **$8–25** for its two OA papers.

### 3e · The VLM digitiser routes on the synthetic corpus

`validation/scripts/synthetic_figures.py` measures routes A (vector) and B (raster CV) offline and
says so. Routes C (VLM coordinates) and D (VLM read-out) need a model and have **never been
measured against known truth**. Extending the script to call `canopy.digitize.digitizer.digitize`
on the twelve synthetic figures would give the first real accuracy number for the routes that
actually decide most figure values — ≈ $10–15 for the corpus, and the highest-value unfinished
measurement in the project.

---

## 4 · Supply the paywalled papers

The corpus is **19 unique papers** against 50 gold datasets. Most of the human's rows come from
papers this machine does not have.

`validation/papers_oa/MANIFEST.csv` lists **22 studies with no legal open-access copy**, each with
its DOI. Download them through an institutional subscription and drop the PDFs into
`~/Downloads/Systematic Review/papers/` (any filename — de-duplication is by sha256, DOI and
title), then:

```bash
.venv/bin/python validation/scripts/run_cisneros.py --write-split   # the split MUST be rewritten
```

**Rewriting the split invalidates the held-out result**, because the alternation re-shuffles when
papers are added. So either add the papers *before* running held-out, or accept that held-out has
to be re-tagged (`validation-heldout-v2`) and re-run. Doing 4 before 3b is the cheaper order.

`fetch_oa_papers.py` will not help with these: it only follows Unpaywall / Europe PMC / publisher
OA pages and records anything else as `paywalled`, by design.

> **Note on `fetch_oa_papers.py`.** It sends `sritej.paddy@gmail.com` as the Unpaywall and Crossref
> contact address (`CONTACT_EMAIL`, line 58) — those APIs require a real one and reject the shared
> `example@example.com`. Anyone else running the script should substitute their own; it is a
> courtesy identifier, not a credential, but it is a personal email address sitting in a file that
> may be published.

---

## 5 · Known limitations and parked findings

Collected from every task report and the progress ledger. Nothing here is a surprise waiting to
happen — each was found, understood and deliberately left.

### Cost and performance

1. **Prompt caching is not hitting at all.** The 3-paper validation run made 109 calls with
   **1 099 996 input tokens and 0 cache reads**. Amendment B's design (cache `document`/image
   blocks when ≥ 2 same-model readers will reuse them, with a warm-up call before fan-out) is
   implemented but is not producing reads in a real run — most likely the cache-control markers are
   not surviving onto the blocks the fan-out actually sends, or the warm-up and the readers do not
   share a prefix. **This is the single biggest cost lever in the system**: at Opus input pricing
   those tokens are roughly two thirds of the bill. Worth half a day.
2. **A paper that hits `--max-usd-per-paper` contributes nothing.** It ends as `error` mid-stage,
   so the money is spent and no row is produced. Both eligible papers in the probe run did exactly
   that at a $4 cap. Either set the cap generously ($12–14) or make the cap stop a paper *between*
   stages so partial work is kept.
3. **A budget stop reads as a failure.** `BudgetExceeded` is caught per paper, so exhausting the
   global budget leaves several papers in `error` rather than one clean "run stopped: budget"
   status. Cosmetic, but alarming in the manifest.

### Statistics

4. **`canopy/stats/meta.py::egger_test` raises `numpy.linalg.LinAlgError`** when every study has
   the same group sizes (the Pustejovsky–Rodgers predictor is then constant), and with some k it
   does *not* raise but returns a meaningless fit. `canopy/report/tables.py::_egger` detects the
   degenerate predictor and falls back to the classic precision predictor with a note. The clean
   fix is for `egger_test` to check its predictor's spread and raise a typed `NotEstimable`.
   **Reported by Task 10/11, not fixed.**
5. **`DispersionType` has no `CI99` member.** A 99 % CI is worked around in
   `canopy/pipeline/resolve.py`. Add the enum member at the next fixture re-record.
6. **`one_row_per_paper` aggregation assumes dependence when it cannot prove independence.** A
   dataset claims its own participant sample only when the paper labelled the experiment *and* the
   mapper called it a first exposure; anything else is combined as dependent (`r = 0.5`, settable
   via `StatsSettings.within_paper_r`). That is the conservative direction — dependence never buys
   precision — but it does mean a paper with two genuinely independent unlabelled experiments gets
   a wider composite than it deserves.

### Digitiser

7a. **A one-armed error bar discards the whole cell.** This is the most consequential finding of
    the first live run. Bock 2005 Fig. 1 draws only one arm of each SD whisker (upward for the old
    group, downward for the young). The four digitiser routes read the **means** to within 3 % of
    each other — 31.2–32.8 deg (old) and 10.6–13.4 deg (young), implying d ≈ −1.5 against the
    human's −1.676 — and every ensemble still came back `ambiguous`, because the error-bar
    half-lengths spanned 5–10 deg against a ~1.1 deg tolerance. Amendment F's dual tolerance is
    applied per cell, so a disagreement about the *dispersion* throws away agreed *means* as well.
    One-armed error bars are common in the literature precisely because they declutter overlapping
    series. The fix is to separate the two verdicts: accept a mean the routes agree on, and mark
    the dispersion `needs_human` (or fall back to the widest arm with a flag), rather than
    discarding the row. `canopy/digitize/digitizer.py::dual_tolerance` and `_ensemble_status`.

7. **`ocr_tick_labels` can return only the last glyph of a tick label.** On two of twelve synthetic
   figures it read `15, 10, 5, 0` as `5, 0, 5, 0` and `−20 … 5` as `5, 0, 5, 0, 5, 0` — dropping a
   leading digit and the minus sign. `fit_axis` then returned a calibration with an **rmse of
   389–581 px** on a ~525-px axis, and **nothing downstream looks at that residual**, so the values
   were read at roughly half scale with no warning. Two fixes, both small: reject a fit whose rmse
   exceeds a fraction of the tick spacing, and treat a monotonic-tick-value violation as fatal.
   In production the model's own tick read cross-checks this (`_choose_calibration`), so it is only
   fully exposed when no VLM ticks are available — but it is a silent factor-of-two error.
8. **`_cv_core` tries `detect_bars` first and only falls back to `detect_markers` when it finds
   none.** On every point chart in the synthetic corpus `detect_bars` returned **one spurious
   bar**, which would stop production ever reaching the marker detector. Prefer the detector whose
   result is consistent with the mapper's `source_kind`, or run both and choose by agreement.
9. **Bars that do not start at the drawn x axis are not detected.** The negative-range synthetic
   case (bars hanging from zero, axis at the bottom) yielded one bar instead of two.
10. **Pure line-art vector figures may yield no `FigureRegion`.** `ingest_pdf._figure_regions`
    counts drawings with `Rect.intersects`, which is `False` for a degenerate rect. Carried
    forward from Task 1; a manual overlap test fixes it, and the Bock roster must be checked
    unchanged before and after (else the mapper and extractor fixtures need re-recording).
11. **A verifier re-open changes the reader, not the reading.** "≤ 2 re-opens" asks a stronger
    model the same question; it does not re-extract. The third-candidate mechanism covers
    "get more evidence". If a re-open should also buy a fresh reading, that is one line in
    `_verify_cell` and a real cost increase.

### Extraction

11a. **Not one of Bock 2005's seven F statistics was admissible**, and correctly so — they are
     mixed main effects tested against a different error term, plus interaction terms. This is
     amendment C's convertibility gate working, but it is worth knowing that on this literature
     the test-statistic route will rarely rescue a paper: repeated-measures designs are the norm,
     and their F values do not convert to a between-groups d.

12. **The CI-level `%` heuristic can false-refute a literal 95 % outcome value** (a paper whose
    outcome *is* a percentage near 95).
13. **A sign lookbehind corrupts numbers abutting `)` or `]` with no separator** — within one cell
    only.
14. **`3.5×10³` is NFKC-folded** during normalisation (pre-existing).
15. **Positional dataset pairing between agents.** The mapper and the cross-check line datasets up
    by position, not by identity. It has not bitten yet; a paper with several experiments in a
    different order would expose it.
16. **The re-recorded Bock map lists a second dataset** (tracking-only control groups) flagged
    "group mapping unconfirmed". Whether the protocol as written should include it is exactly the
    kind of question the discrepancy adjudication in 3c is for.

### Report and server

17. **`canopy/server/overrides.py::_rewrite` calls `write_outcome_outputs` without `all_rows=`**,
    so after an override re-pool that outcome's extraction table shows the composite row but not
    the rows it replaced. One keyword argument. (Carried to the Task 12 fix round.)
18. **`extraction_table` falls back to `record.inputs`** when no `Verdict` is passed; the keys it
    looks for are `canopy/pipeline/resolve.py`'s naming. A rename there silently blanks those
    columns for a direct caller. The orchestrator always passes verdicts.
19. **The HTML report is not CSP-hardened.** It escapes everything and makes no external request,
    but it is a file, not a served page with headers.
20. **One vendor.** The provider seam exists (`LLMProvider` + Anthropic / replay / fake), but only
    Anthropic is wired up, so "two models agreed" means two models from one family. The design
    review explicitly declined to make a second vendor mandatory; it remains the strongest
    available improvement to the verification story.

---

## 6 · Before this repository is published

* **Remove `tests/fixtures/pdfs/*.pdf`** — Bock 2005 and Heuer & Hegele 2008 are copyrighted
  publisher PDFs, kept for local development only. `tests/fixtures/README.md` says so. Replace them
  with openly licensed equivalents or the suite loses its two real papers.
* **Check `validation/papers_oa/*.pdf` licences** — they are open access, but individual licences
  (CC-BY vs "free to read") differ; `MANIFEST.csv` records the source URL for each.
* **`validation/reference/cisneros2024/`** is public OSF material (`SOURCES.md` gives the
  provenance), but the authors deserve a citation in any published comparison.
* **AGPL-3.0-or-later**, because PyMuPDF is. Running a modified Canopy as a network service
  obliges you to offer its source to users. PyMuPDF is used for text, word boxes, page rasters,
  figure regions and the whole vector digitiser route, so replacing it is not a small job.
* Substitute the Unpaywall contact email (section 4).
