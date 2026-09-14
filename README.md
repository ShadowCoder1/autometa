# AutoMeta

**A meta-analysis from a folder of paper PDFs — with every number traceable to the sentence, table
cell or pixel it came from.**

*(The Python package and CLI are still named `canopy`, the project's working name. AutoMeta is the
name the web interface and the published figures use.)*

You give AutoMeta a folder of PDFs and a protocol (what question, which two groups, which outcomes,
which studies count). It reads every paper the way a careful reviewer would — locating the values
in the text, the tables and the figures, reading each one with more than one model, checking those
readings against each other and against the PDF, converting them to effect sizes **in code**, and
pooling them into a random-effects forest plot. Anything it is not sure about it hands to you,
with the evidence, instead of guessing.

The thing that makes it usable is not the extraction. It is that you can click any number in the
report and see the page it was read off, with the sentence highlighted — or the figure, with the
digitiser's own marks drawn on it.

> **Status.** Working end to end and validated against one published review (see
> [`validation/README.md`](validation/README.md)). Not a released package: expect to read the code.

> **No paper PDFs are distributed here.** Journal articles are copyrighted by their publishers, so
> every third-party PDF is excluded from this repository — the review corpora, the open-access
> validation set, and the test fixtures alike. Each of those directories keeps its manifest, so you
> can fetch the same papers yourself from their DOIs; `validation/papers_oa/fetch_oa_papers.py`
> does it for the open-access set. Tests that read a fixture PDF will fail until you supply one
> (see [`tests/fixtures/pdfs/README.md`](tests/fixtures/pdfs/README.md)); the rest of the suite
> runs on a fresh clone.

---

## What it actually does

```
PDFs ──▶ ingest ──▶ map ──▶ extract ──▶ verify ──▶ resolve ──▶ pool ──▶ report
        (no LLM)   (agent)  (agents)    (agents)   (code)     (code)   (code)
```

1. **Ingest** (no model): de-duplicate the folder by sha256, DOI and title; rasterise pages; find
   figure regions and table blocks; keep the text layer with word boxes so any quote can be traced
   back to its pixels.
2. **Map** (agent): decide whether the paper is eligible against *your* criteria, list every
   group it compared, choose the A/B pair with a written rationale, and locate — with a quote —
   where each outcome's numbers live. A second, cheaper model cross-checks the group mapping and
   the error-bar type; they must agree or the cell goes to a human.
3. **Extract** (agents + code): two text extractors with different reading strategies, a
   test-statistic reader, and — when the numbers exist only in a plot — a four-route digitiser
   (the PDF's vector drawing operators; colour-segmented computer vision; a model pointing at
   pixels that are then snapped to detected edges; a model reading values off the axis).
4. **Verify** (agents + code): the readings vote within their route; deterministic checks look for
   SE mistaken for SD, the wrong group, the wrong time window, an implausible n; a verifier from a
   *different* model family is handed the whole PDF and asked both "is this right?" and "is there
   a better source anywhere in this paper?"; an adjudicator settles what is left.
5. **Resolve and pool** (code only): the accepted values become effect sizes through
   `canopy.stats` — never through a model. Route precedence, unit conversions, shared control
   arms, within-paper dependency, REML τ², Hartung–Knapp, prediction intervals, Egger's test,
   leave-one-out.
6. **Report** (code only): forest plot, extraction table, PRISMA flow, funnel, sensitivity set,
   a methods paragraph templated from the manifest, and a self-contained `report.html` linking a
   provenance image for every single value.

**No model ever does arithmetic.** Agents report what a paper says; every derived number comes from
`canopy/stats/`, and a test asserts that no agent's output schema even contains a field like `d`,
`g` or `pooled_sd`.

---

## Install

Python 3.11+.

```bash
git clone <this repo> && cd canopy
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

System dependencies:

* **tesseract** — reads tick labels off figure axes (`brew install tesseract`). Without it the
  digitiser loses its raster calibration and leans on the model's tick read.
* **R with `meta` and `metafor`** — *optional*, and only to regenerate the statistical reference
  fixtures in `validation/fixtures/` (`Rscript validation/fixtures/make_r_fixtures.R`). The test
  suite reads the committed fixtures and does not need R.

The API key goes in a `.env` at the repo root, which is git-ignored:

```
ANTHROPIC_API_KEY=sk-ant-...
```

It is loaded once at startup and **never printed** — `canopy run` reports only "API key
configured / NOT configured", and a test plants a key in the environment and asserts it never
reaches the output. Do not pass it on a command line and do not commit it.

The default test suite is **fully offline**: no key, no network.

```bash
.venv/bin/python -m pytest -q
```

---

## Use it

### Command line

Installing the package puts a `canopy` script in `.venv/bin/`; `.venv/bin/python -m canopy.cli`
does the same thing and is used below so the commands work from a bare checkout.

```bash
# a review, from a folder of PDFs
.venv/bin/python -m canopy.cli run \
    --papers ~/papers/my-review \
    --protocol examples/protocols/aging_sensorimotor_adaptation.yaml \
    --out runs/my-review \
    --budget-usd 60 --max-usd-per-paper 12 --concurrency 3

# start a protocol from a commented skeleton, or from a shipped example
# (`init` takes a NAME and writes <name>.yaml — passing "my-review.yaml" gets you
#  my-review.yaml.yaml)
.venv/bin/python -m canopy.cli protocol init my-review
.venv/bin/python -m canopy.cli protocol init my-review --from aging_sensorimotor_adaptation
.venv/bin/python -m canopy.cli protocol check my-review.yaml

# re-pool a finished run from its own stage files, with NO model calls, and check
# that every artefact the manifest promises is actually there
.venv/bin/python -m canopy.cli validate runs/my-review

# the local web UI
.venv/bin/python -m canopy.cli serve --port 8000
```

`--resume` is on by default: a stage that already has a file is skipped, and every model call is
content-addressed in `<out>/cache`. A run stopped by its budget can be continued with the same
command and pays only for what is new.

### Web UI

`canopy serve` binds **127.0.0.1** and issues a per-run bearer token. Three screens: build a
protocol with a guided form (or draft one from a sentence, or paste YAML) and pick a folder; watch
a per-paper stage grid with a live cost meter; then read the results — forest, extraction table,
the review queue sorted by how far each held-back cell would move the pooled estimate, the figures,
and an evidence drawer showing what each model read and the crop it read it from. Uploads are
magic-byte checked and stored as `<sha256>.pdf`; artefacts are served by manifest id, not by path.

### Hosting it

The same server runs anywhere a container runs. The `Dockerfile` builds it with R and the `meta`
package so the report's forest plot is the one the local install draws, and `deploy/cloudrun.sh`
puts it on Google Cloud Run in one command — always-on, one instance, runs on a bucket, secrets in
Secret Manager, an HTTPS URL at the end:

```bash
gcloud auth login && gcloud config set project <PROJECT_ID>
CANOPY_ENV_FILE=.env deploy/cloudrun.sh          # asks for the key and an access code, silently
```

Two things change off loopback. The run list stops handing out tokens, so a visitor sees only the
runs they started. And **`CANOPY_ACCESS_CODE` puts a password on the whole site** — without it,
anyone who finds the URL can start a review on your key, with your budget. The browser holds a
digest of the code in an `HttpOnly` cookie, never the code; scripts send it as `X-Canopy-Access`.
An always-on 2 vCPU / 4 GiB instance is roughly $100 a month before any model calls; the model
calls are what a review actually costs (see below).

---

## Writing a protocol

A protocol is the review's own Methods section, in YAML, written **before** any paper is read. It
contains no knowledge of any individual study.

```yaml
title: Ageing and upper-extremity sensorimotor adaptation
research_question: >-
  Do older adults differ from younger adults in adaptation to a visuomotor perturbation?

group_a:
  key: A
  label: Older adults
  definition: >-
    The older of the two age groups compared by the study, as the study itself labels it.
  synonyms: [older, older adults, elderly, aged, seniors]
group_b:
  key: B
  label: Younger adults
  definition: >-
    The younger of the two age groups compared by the study, as the study itself labels it.
  synonyms: [younger, younger adults, young, young adults]

outcomes:
  - key: late_adaptation
    label: Late adaptation
    definition: >-
      Adaptive change late in the perturbation block; use the block average when reported.
    measurement_window: >-
      The end of the initial perturbation block, with the perturbation still applied.
    higher_is_better_hint: >-
      Error-like measures are better when smaller; adaptation-like measures when larger.
      Decide per dataset from the paper's own wording and record the quote.
    positive_direction_label: Enhanced in Old        # the forest plot's x-axis labels
    negative_direction_label: Reduced in Old
    units_hint: degrees, cm, N, or a unitless index — record exactly what the paper prints.

eligibility:
  - The study used an upper-extremity adaptation task with a visuomotor or force-field perturbation.
  - The study reported data comparing older adults with younger adults.
dataset_rules:
  - When the same participants took part in two experiments, include only the first.
moderators: [task_type, perturbation_size_deg, n_targets]

stats:
  profile: cisneros2024
```

Two things carry most of the weight and are worth writing carefully:

* **`measurement_window`** — the single most common source of an honest disagreement between two
  extractions is *which* block, phase or timepoint the outcome means. Say it precisely.
* **`higher_is_better_hint`** — the sign of every effect depends on it. Canopy still decides it
  per dataset from the paper's own wording (two agents must agree), but the hint is what they
  reason against.

`positive_direction_label` / `negative_direction_label` are printed under the forest plot's x axis,
so a reader never has to work out which way is "better".

### Statistics profiles

`stats.profile` picks a set of conventions; anything you set explicitly in `stats:` wins over it.

| profile | conventions |
|---|---|
| `metafor` (default) | Hedges' g, Borenstein large-sample variance, REML τ², z-based CI and prediction interval, one row per paper |
| `cisneros2024` | Cohen's d, Hedges–Olkin df-based variance, REML τ², no Hartung–Knapp, HTS (t-based) prediction interval, and **all** of a paper's datasets pooled separately — the reference review's own choices |

Every figure and table carries a **conventions footer** that enumerates the estimator, the variance
formula, the τ² method, whether Hartung–Knapp was applied, the prediction-interval method *and its
degrees of freedom*, both I² definitions, k datasets from k papers, and how many rows were held
back. Two forest plots produced under different profiles can never be confused for each other.

Examples live in `examples/protocols/`: `aging_sensorimotor_adaptation.yaml` (the validation
review) and `clinical_vs_control_adaptation.yaml` (a deliberately different one).

---

## What you get back

```
runs/my-review/
├── report.html                      self-contained; every value links to its provenance image
├── methods.md                       a methods paragraph, every number read from the manifest
├── manifest.json                    protocol hash, models, per-paper status, cost, warnings
├── protocol.yaml                    the protocol this run actually used
├── prisma.{json,png,svg}            files → unique papers → eligible → datasets → included
├── exclusions.{csv,json}            every drop, with stage, reason, quote and who decided
├── human_review_queue.{csv,json}    sorted by |Δ pooled| — the biggest doubts first
├── methods_routes.{png,svg}         where the numbers came from, with real example crops
├── provenance/                      one image + one JSON per extracted value
├── papers/<sha12>/{ingest,map,extract,verify,resolve}.json
└── results/<outcome>/
    ├── forest.{png,svg,pdf}         primary rows solid, held-back rows hollow
    ├── extraction_table.{csv,json,xlsx}
    ├── leave_one_out.*  sensitivity.*  funnel.*  pooled.json
```

`canopy validate RUN_DIR` re-pools the whole run from those stage files with no model calls, and
exits non-zero if any artefact the manifest promises is missing. That is the guarantee: the
numbers in the report follow from files on disk that you can read yourself.

### The human-review workflow

A cell becomes `needs_human` when the readings disagree beyond tolerance, a check fires, the
verifier refuses to confirm, or the group mapping could not be independently agreed. Those rows are
drawn hollow, excluded from pooling, counted in the footer, and queued in
`human_review_queue.csv` **in order of how far each one would move the pooled estimate**.

In the UI you can, per cell: override a value with a justification, exclude the dataset, override
eligibility, ask for a re-extraction with a hint, or simply mark it reviewed. Every decision is
appended to an immutable `overrides.jsonl` and re-applied on every re-pool — so your review
survives `--resume`, and someone else can see exactly what you changed and why.

---

## What it costs

Measured on real papers (2026-08-16), not estimated:

| | |
|---|---|
| mapping one paper | $0.27 – $1.81 |
| Bock 2005 — 5 pages, 2 figures, 50 candidates | **$10.20** |
| Wolpe 2020 — 11 pages, 4 figures, 79 candidates | **$14.94** |
| a paper excluded at the mapping stage | $0.50 |
| the whole 3-paper run | **$26.25**, 353 model calls, ~50 min |

Budget **$10–15 per paper** and set `--max-usd-per-paper` to match: a paper that hits its cap ends
as `error` and contributes nothing, so a cap that is too low buys the cost without the result.
Figure-heavy papers are the expensive ones — the digitiser runs four routes and several tool loops
per plotted value.

Costs come down two ways: `--resume` plus the content-addressed disk cache (an already-answered
question is never paid for twice), and `--max-papers` for a dry run on two or three papers before
committing to a folder. A known inefficiency is recorded in `docs/handoff.md`: prompt caching is
currently not hitting, so page images are re-sent at full price to every reader.

---

## Validation

Canopy was checked against **Cisneros et al. (2024)**, a published meta-analysis of ageing and
sensorimotor adaptation whose per-study extractions and analysis code are public. The scripts that
do it — the backward-validation scatter of manual vs automatic Cohen's d with CIs on both axes,
the auto forest beside the manual one, the extraction-route figure, three runnable
"where did this number come from" examples, and a synthetic figure corpus with known truth for
digitiser accuracy — are in `validation/`, with commands, expected outputs and results in
[`validation/README.md`](validation/README.md).

---

## Limitations, honestly

* **`needs_human` is a real answer, not a failure.** Canopy holds back anything it cannot verify.
  A run over difficult papers can leave a large fraction of cells queued; that is the design.
* **Figures are read, not measured.** On a synthetic corpus with known truth, the two
  *deterministic* routes land within the digitiser's own 2 % tolerance on 100 % (vector) and 93 %
  (raster CV) of read-outs, with median errors of 0.00 % and 0.09 % of the axis range — but the
  two model-driven routes have **never been measured against known truth**, and a real published
  plot can have overlapping markers, log axes, superscripted tick labels or a raster scan of a
  raster. Every figure-derived value carries a digitisation sigma, and there is a sensitivity
  analysis that removes them all. See [`validation/README.md`](validation/README.md).
* **Eligibility is the model's judgement.** It is recorded with a quote and a rationale and it is
  overridable in the UI, but a paper wrongly excluded at the mapping stage never gets extracted.
* **Paywalled papers are not fetched.** `validation/papers_oa/fetch_oa_papers.py` finds only
  legally open copies (Unpaywall / Europe PMC); the rest are listed for a human to supply.
* **One vendor.** There is a provider seam (`LLMProvider`, with Anthropic, replay and fake
  implementations) but only one real provider is wired up, so "two models agreed" currently means
  two models from the same family.
* **Not a substitute for a systematic reviewer.** It is a very fast, very literal first pass that
  shows its working.

---

## Development

```bash
.venv/bin/python -m pytest -q             # fully offline: no key, no network
```

Agent tests replay **recorded fixtures** (`tests/fixtures/llm/`, one content-addressed JSON per
request), so the suite is deterministic and free. A missing fixture raises rather than silently
going live. To re-record after changing a prompt:

```bash
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_mapper.py -q
```

**Pending recordings.** Ten tests currently skip because their fixtures have never been recorded.
Each prints its own command; together they cost about **$4.00–4.75**. Do them in this order — the
last one reuses everything the others record:

```bash
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_extract_text.py -q      # 4 skips, ~$0.15
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_verifier.py -q          # 4 skips, ~$0.40
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_digitizer.py -q         # 1 skip,  ~$1.40
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_pipeline_offline.py -q  # 1 skip,  ~$2.00-2.80
```

Zero replay skips is the gate before merging to `main`. The caveats (a temporary skip wrapper to
delete, and why the pipeline's requests produce *new* fixtures rather than reusing the agents')
are in [`docs/handoff.md`](docs/handoff.md), together with everything else outstanding.

The architecture — stages, agents, models, verification gates, the data model and the provenance
guarantees — is in [`docs/architecture.md`](docs/architecture.md).

---

## Licence

**AGPL-3.0-or-later.** Canopy depends on [PyMuPDF](https://pymupdf.readthedocs.io/), which is
AGPL-3.0, and that licence propagates: if you run a modified Canopy as a network service, you must
offer its source to your users. Replacing PyMuPDF (it is used for text, word boxes, page rasters,
figure regions and the vector digitiser route) would be a substantial piece of work.

The PDFs under `tests/fixtures/pdfs/` are **copyrighted publisher files** kept for local
development only. They must be removed or replaced before this repository is published.
