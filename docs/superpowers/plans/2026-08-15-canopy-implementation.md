# Canopy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a folder of paper PDFs + a protocol into an agent-verified random-effects forest plot with full provenance (CLI + local web UI), and produce the Cisneros validation deliverables.

**Architecture:** Deterministic ingestion (PyMuPDF) → Claude agents that *locate/label/critique* (mapper, extractors, digitizer, verifiers, adjudicator) with structured outputs and quote/pixel grounding → all arithmetic in code (`canopy/stats`, validated vs R) → pooling/plots/report → FastAPI UI. Every LLM call is cached on disk (content-addressed) and costed; runs are resumable.

**Tech Stack:** Python 3.13, `anthropic` ≥ 0.122 (Messages API, structured outputs `output_config.format`, Files API beta, prompt caching), pydantic 2, PyMuPDF, OpenCV, numpy/scipy, matplotlib, FastAPI + uvicorn, vanilla JS/SVG frontend, pytest.

**Spec:** `docs/superpowers/specs/2026-08-15-canopy-design.md` (+ review `2026-08-15-design-review.md`, research `docs/research/00-synthesis-and-decisions.md`, API facts `docs/research/06-anthropic-api-fact-sheet.md`).

## Global Constraints
- Python ≥ 3.11; run everything with `~/Projects/canopy/.venv/bin/python`; tests: `python -m pytest -q` (no API key needed for the default test suite — LLM calls are replayed from fixtures in `tests/fixtures/llm/`; live tests are marked `@pytest.mark.live` and skipped unless `CANOPY_LIVE=1`).
- API key from `.env` (`ANTHROPIC_API_KEY`) via `python-dotenv` at CLI/server startup only; never printed/logged.
- Models: `claude-opus-5` (primary), `claude-sonnet-5` (second route/verifier), `claude-fable-5` (optional adjudicator, needs `betas=["server-side-fallback-2026-07-01"], fallbacks="default"`); adaptive thinking (do NOT pass `thinking` or `temperature`); `output_config={"effort": ..., "format": {"type":"json_schema","schema": ...}}` for structured outputs; schemas need `additionalProperties: false` + `required` everywhere; NO `minimum/maximum/minLength/pattern` keywords in schemas.
- Images sent to Claude are pre-resized with `canopy.ingest.images.prepare_for_claude` (2576 px / 4784 tokens rule) and coordinates are requested as absolute pixels of the sent image.
- LLMs never do arithmetic on values; conversions/effect sizes/pooling only via `canopy.stats`.
- Every scalar in the analysis carries provenance (page, quote or crop+pixels, model, prompt version, LLM call id).
- Nothing study-specific: all domain knowledge lives in the protocol YAML (`examples/protocols/*.yaml`).
- Commit after each task; run `python -m pytest -q` before committing.

---

## File structure (target)
```
canopy/
  models.py            pydantic: Protocol, GroupDef, OutcomeDef, StatsSettings, StudyMap, DatasetSpec, Source,
                       Candidate, Verdict, EffectSizeRecord, RunManifest (+ enums)
  protocol.py          load/validate protocol YAML, profiles (metafor, cisneros2024), examples
  llm/client.py        LLMClient: structured/text calls, images/PDF blocks, caching (disk), cost, retries, fallbacks
  llm/context.py       build page-image/text excerpt blocks from ingestion outputs; Files API upload helper
  llm/prompts/*.md     versioned prompt templates (mapper, extract_table_first, extract_narrative, digitize_*, verifier, adjudicator, orientation)
  agents/mapper.py     StudyMap agent (+ cross-check)
  agents/extract_text.py  text/table extractors (2 variants) → Candidates
  agents/extract_stats.py test-statistic / reported-d extractor → Candidates
  agents/verifier.py   adversarial verifier, adjudicator, orientation agent
  digitize/calibrate.py  axis calibration (LS fit, residuals), pixel↔data
  digitize/vlm.py      VLM read-out (D) and VLM coordinates (C) prompts + parsing
  digitize/cv.py       snap-to-edge/cap refinement, error-bar cap search, overlay drawing
  digitize/vector.py   vector-exact path from PDF drawings (tick labels/marks/whisker lines)
  digitize/digitizer.py  orchestrates A→C→D + ensembles + overlay-verify → figure Candidates with σ
  verify/grounding.py  quote grounding vs page text; page/bbox validation
  verify/checks.py     deterministic consistency checks
  verify/vote.py       agreement voting, confidence scoring, human flags
  pipeline/resolve.py  route precedence → EffectSizeRecord via canopy.stats
  pipeline/run.py      orchestrator (stages, resume, manifest, budget, concurrency)
  report/forest.py     forest plot (matplotlib; PNG/SVG/PDF) following `dataviz` skill
  report/tables.py     extraction table CSV/XLSX/JSON, provenance bundle, methods-breakdown figure
  report/html.py       static HTML report
  server/app.py        FastAPI app, jobs, SSE, static UI (server/static/*)
  cli.py               typer CLI: run, serve, protocol, validate
examples/protocols/aging_sensorimotor_adaptation.yaml   (written from the Cisneros Methods only)
validation/scripts/*.py                                  (deliverables)
tests/...                                                (fixture-driven; live tests opt-in)
```

---

### Task 1: Data models, protocol schema, profiles

**Files:** Create `canopy/models.py`, `canopy/protocol.py`, `canopy/profiles/metafor.yaml`, `canopy/profiles/cisneros2024.yaml`, `examples/protocols/aging_sensorimotor_adaptation.yaml`; Test `tests/test_models.py`.

**Interfaces (Produces):**
```python
class GroupDef(BaseModel): key: Literal["A","B"]; label: str; definition: str; synonyms: list[str] = []
class OutcomeDef(BaseModel): key: str; label: str; definition: str; measurement_window: str; higher_is_better_hint: str;
                              positive_direction_label: str  # e.g. "Enhanced in older"; negative_direction_label: str
class StatsSettings(BaseModel): profile: str = "metafor"; estimator: Literal["cohen","hedges"]="cohen";
    variance: Literal["borenstein","hedges_olkin_df","meta_exact","meta_exact_g","meta_hedges_approx"]="borenstein";
    tau2_method: Literal["REML","DL","PM"]="REML"; hakn: bool=False; pi_method: Literal["V","HTS","z"]="V";
    route_precedence: list[str] = ["reported_d","text_mean_sd","table","text_mean_se_ci","figure","test_statistic"];
    late_window_rule: str = "paper_reported_block_else_last_point"; add_digitization_variance: bool = False
class Protocol(BaseModel): title: str; research_question: str; group_a: GroupDef; group_b: GroupDef;
    outcomes: list[OutcomeDef]; eligibility: list[str]; dataset_rules: list[str]; moderators: list[str] = [];
    stats: StatsSettings = StatsSettings(); notes: str = ""
    def outcome(self, key) -> OutcomeDef; def hash(self) -> str   # sha256 of canonical JSON
def load_protocol(path) -> Protocol; def apply_profile(settings: StatsSettings) -> StatsSettings
```
Also in `models.py`: `SourceKind` enum (`text_mean_sd, text_mean_se, text_mean_ci, table, figure_bar, figure_line, figure_points, figure_box, test_statistic, reported_effect_size, author_data`), `DispersionType` (`SD, SE, CI95, CI90, IQR, RANGE, NONE, UNKNOWN`), `Source`, `GroupSpec(label, n, n_evidence, age_mean, age_sd)`, `DatasetSpec(dataset_id, label, experiment, condition, group_a: GroupSpec, group_b: GroupSpec, moderators: dict[str,str], outcomes: list[OutcomeSources])`, `OutcomeSources(outcome_key, measure_name, units, higher_is_better, higher_is_better_evidence, operationalization, sources: list[Source])`, `StudyMap(paper_id, citation: Citation, eligible, eligibility_rationale, datasets, notes, model, prompt_version, llm_call_ids)`, `Candidate` (fields per spec §2 incl. `kind: Literal["group_stats","test_statistic","reported_d"]`, `group: Literal["A","B",None]`, `n, mean, dispersion_value, dispersion_type, unit, page, quote, locator, crop_path, pixel_provenance: dict, sigma: float|None, model, prompt_version, llm_call_id, extractor_id`), `Verdict`, `EffectSizeRecord`, `RunManifest`.

- [ ] Step 1: write `tests/test_models.py`: loads `examples/protocols/aging_sensorimotor_adaptation.yaml`, asserts 2 outcomes with keys `late_adaptation`, `aftereffect`, group A label contains "Older", `apply_profile` for `cisneros2024` yields `variance=="hedges_olkin_df"`, `pi_method=="HTS"`, `Protocol.hash()` stable; a `Candidate(kind="group_stats", group="A", ...)` round-trips through JSON.
- [ ] Step 2: run → fails (module missing). Step 3: implement models/protocol/profiles/example protocol (write the protocol from the Cisneros paper's **Methods section only**: eligibility 4 criteria + 3 considerations; outcomes late adaptation ("average of overall performance when reported, else last measure of the group time series") and aftereffect; groups older vs younger; moderators task type, perturbation size, number of targets). Step 4: tests pass. Step 5: commit `feat(models): protocol/profile schemas and study/candidate models`.

### Task 2: LLM client with structured outputs, caching, cost, fixtures

**Files:** Create `canopy/llm/client.py`, `canopy/llm/costs.py`, `canopy/llm/cache.py`; Test `tests/test_llm_client.py`, fixtures dir `tests/fixtures/llm/`.

**Interfaces (Produces):**
```python
@dataclass class LLMResult: text: str; parsed: Any|None; usage: dict; cost_usd: float; model: str; call_id: str; cached: bool; stop_reason: str; latency_s: float; raw: dict
class LLMClient:
    def __init__(self, cache_dir: Path|None, budget_usd: float|None=None, max_concurrency: int=6, record_dir: Path|None=None, replay_dir: Path|None=None, default_effort="high")
    def structured(self, *, model: str, system: str|list, messages: list, schema: dict, effort: str|None=None, max_tokens=16000, cache_key_extra: str="", betas: list[str]|None=None, fallbacks: str|None=None) -> LLMResult
    def text(self, *, model, system, messages, effort=None, max_tokens=8000) -> LLMResult
    def count_tokens(self, *, model, system, messages) -> int
    def image_block(self, png_path: Path, cache: bool=False) -> dict     # base64 image block (media_type image/png)
    def pdf_block(self, pdf_path: Path, file_id: str|None=None, cache: bool=True) -> dict  # document block (file_id if given else base64) with cache_control when cache=True
    def total_cost(self) -> float; def calls(self) -> list[dict]
```
Behaviour: content-addressed cache key = sha256(model, system, messages (with image bytes hashed), schema, effort, max_tokens); disk cache in `cache_dir/<key>.json`; `replay_dir` = fixtures for tests (miss → raise `MissingFixture` unless `CANOPY_LIVE=1`); `record_dir` writes fixtures. Retries: rely on SDK `max_retries=4`; on `stop_reason == "refusal"` raise `RefusalError`; when `fallbacks` given, use `client.beta.messages.create(betas=[...], fallbacks=...)`. Cost table (`costs.py`): opus-5 5/25, sonnet-5 2/10, fable-5 10/50, opus-4-8 5/25, haiku-4-5 1/5 per MTok; cache write ×1.25, cache read ×0.1. Budget: raise `BudgetExceeded` before a call if `total_cost + estimated > budget`. Concurrency: threading semaphore. Streaming: use `client.messages.stream(...).get_final_message()` when `max_tokens > 16000`.

- [ ] Step 1: tests: (a) cache key stable across dict ordering; (b) replay from a fixture JSON returns parsed dict; (c) `MissingFixture` raised when absent and not live; (d) `estimate_cost(usage, model)` numbers (e.g. 100k input + 10k output on opus-5 = 0.75); (e) `image_block` produces valid base64 block; (f) budget guard raises.
- [ ] Step 2: run → fail. Step 3: implement (use `anthropic.Anthropic()`, `messages.create` with `output_config`; parse `json.loads` of first text block; keep `raw`). Step 4: pass. Step 5: commit `feat(llm): client with structured outputs, disk cache, cost accounting, fixture replay`.

### Task 3: Context builders + Files API

**Files:** Create `canopy/llm/context.py`; Test `tests/test_context.py` (uses ingestion outputs of `tests/fixtures/pdfs/bock2005.pdf` — copy `paper_1731234.pdf` there; keep tests offline).

**Interfaces:**
```python
def upload_pdf(client: LLMClient, pdf_path: Path) -> str          # Files API (betas=["files-api-2025-04-14"]) → file_id; cached in <out>/file_id.txt
def page_blocks(paper: PaperRecord, pages: list[int], with_text: bool=True) -> list[dict]   # per page: text block "[page N text]…" + image block of pages/pNNN.png
def figure_blocks(paper: PaperRecord, fig_ids: list[str]) -> list[dict]                       # caption text + claude png
def excerpt_for(paper: PaperRecord, pages: list[int], figures: list[str]) -> list[dict]        # composed, with cache_control on last block
```
- [ ] Step 1: tests: `page_blocks` on ingested Bock returns 2 blocks per page with correct order; `excerpt_for` last block has `cache_control`. Step 2 fail → Step 3 implement → Step 4 pass → Step 5 commit `feat(llm): context builders and Files API upload`.

### Task 4: Mapper agent (StudyMap) with cross-check

**Files:** Create `canopy/agents/mapper.py`, `canopy/llm/prompts/mapper.md`, `canopy/llm/prompts/mapper_crosscheck.md`; Test `tests/test_mapper.py` (fixture replay: record one live run on Bock 2005 into `tests/fixtures/llm/` with `CANOPY_LIVE=1 python -m pytest tests/test_mapper.py --record`).

**Interfaces:**
```python
def map_study(client: LLMClient, paper: PaperRecord, protocol: Protocol, *, model_primary="claude-opus-5", model_check="claude-sonnet-5", pdf_file_id: str|None=None) -> StudyMap
```
Prompt (generalize the spike): STUDY MAPPER role; protocol text; rules: analyzed N (post-exclusion) with quotes; enumerate every dataset (independent sample × condition) with the A-vs-B contrast; for each outcome list EVERY source location (kind, page 1-based, locator, verbatim quote, error-bar type from caption/legend/methods, values if in text); do not digitize; note design traps (same participants across conditions → first exposure only; contextual change between learning and aftereffect; transfer conditions); higher_is_better with evidence. Cross-check call (Sonnet, effort medium): eligibility + datasets + Ns only; if eligibility disagrees or any N differs → adjudicate with Opus (xhigh) given both maps; record `disagreements` in `StudyMap.notes`.
Schema = JSON schema of `StudyMap` minus bookkeeping fields (write it by hand mirroring the spike, `additionalProperties:false`).
- [ ] Step 1: test: replayed mapper on Bock returns eligible=True, ≥1 dataset with group n 12/12, at least one `figure_line` source on page 3 for late_adaptation, and error-bar type contains "standard deviation". Step 2 fail → 3 implement → 4 record fixture live then pass offline → 5 commit `feat(agents): study mapper with cross-check`.

### Task 5: Text/table extractors + quote grounding

**Files:** Create `canopy/agents/extract_text.py`, `canopy/verify/grounding.py`, prompts `extract_table_first.md`, `extract_narrative_first.md`; Test `tests/test_grounding.py`, `tests/test_extract_text.py`.

**Interfaces:**
```python
def ground_quote(quote: str, page_text: str) -> tuple[bool, float, str]   # (grounded, similarity, normalized_match)
def extract_group_stats(client, paper, protocol, dataset: DatasetSpec, outcome_key: str, sources: list[Source], *, variant: Literal["table_first","narrative_first"], model: str) -> list[Candidate]
```
Grounding: NFKC normalize, collapse whitespace, unify minus/hyphens and ± spacing; exact substring → (True, 1.0); else fuzzy `difflib.SequenceMatcher` sliding window ≥ 0.95 → True; else False. Extractor: targeted context = pages from sources (+ tables' rows as text) ; quote-first schema per group: `{group_label_as_written, page, kind, quote, row_header, col_header, value_as_written, n, mean, dispersion_value, dispersion_type, unit, notes}`; parse into `Candidate`s; set `grounded` via `ground_quote`. Two variants differ in instructions order/emphasis and model.
- [ ] Step 1: tests: grounding handles "−0.5" vs "-0.5", line-break hyphenation ("adap-\ntation"), and rejects an invented quote; extractor replay on a fixture returns 2 candidates (A,B) with grounded=True. Steps 2–5 as usual; commit `feat(agents): text/table extractors with quote grounding`.

### Task 6: Figure digitizer (calibration, VLM read-out, VLM coords + CV snap, vector-exact, ensembles, overlay)

**Files:** Create `canopy/digitize/{calibrate,vlm,cv,vector,digitizer}.py`, prompts `digitize_readout.md`, `digitize_coords.md`, `digitize_overlay_verify.md`; Test `tests/test_digitize_calibrate.py`, `tests/test_digitize_cv.py`, `tests/test_digitize_vector.py`, `tests/test_digitizer.py` (fixture replay on Bock Fig 1 crop; expected human values young 12.28 ± 11.82, old 31.51 ± 11.12 within 1.0 unit).

**Interfaces:**
```python
@dataclass class AxisCalibration: axis: Literal["x","y"]; scale: Literal["linear","log"]; a: float; b: float; rmse: float; ticks: list[tuple[float,float]]  # value = a*px + b
def fit_axis(ticks: list[tuple[float,float]], scale="linear") -> AxisCalibration; def px_to_value(cal, px) -> float
def read_out(client, crop_png, caption, target: TargetSpec, model, variant) -> ReadOut     # D
def coords(client, crop_png, caption, target, model) -> CoordReadout                       # C (ticks + marker/cap pixel coords)
def snap_horizontal_edge(img: np.ndarray, x: float, y: float, window: int=6) -> tuple[float, float]  # refined y, confidence
def find_cap_ends(img, x, y_center, max_len_px) -> tuple[float|None,float|None]
def vector_candidates(paper: PaperRecord, fig: FigureRegion) -> VectorScene   # tick labels (text+bbox), tick lines, marks, vertical whiskers (PDF points → crop px)
def snap_to_vector(scene: VectorScene, x_px, y_px, radius=8) -> tuple[float,float]|None
def draw_overlay(crop_png, marks: list[dict], out_png)
def digitize(client, paper, fig: FigureRegion, target: TargetSpec, *, models=("claude-opus-5",), n_readouts=3, want_uncertainty=True) -> list[Candidate]   # per group Candidate with mean, dispersion_value(half-length), sigma, pixel_provenance, overlay path
```
`TargetSpec` = `{outcome_key, group_a_label, group_b_label, series_hint (legend/marker text from mapper), x_hint (e.g. "last adaptation episode / block 20"), panel_hint, quantity: "mean_and_error"|"points"|"box"}`. Ensemble aggregation: per (group, quantity) median across all successful routes; `sigma = max(1.4826*MAD, pixel_res_units, calibration_rmse)`; disagreement between routes > max(2% axis range, 0.5 tick) → `needs_review=True`. Overlay-verify: draw resolved marks; ask a fresh Opus call "list mismatches"; if mismatch → drop that route's sample and recompute (max 2 iterations). Vector path only when `fig.kind in {"vector","mixed"}` and tick-label spans found; values then use vector geometry with `sigma≈0`.
- [ ] Step 1: tests: `fit_axis` on synthetic ticks (exact), log scale; `snap_horizontal_edge` on a synthetic image with a black horizontal line finds it within 0.5 px; `vector_candidates` on Heuer&Hegele p6 figure returns ≥ 6 tick labels and ≥ 20 marks; digitizer replay on Bock returns candidates within tolerance. Steps 2–5; commit `feat(digitize): figure digitizer (calibration, VLM+CV, vector-exact, ensembles, overlay verify)`.

### Task 7: Test-statistic / reported-d extractor
**Files:** Create `canopy/agents/extract_stats.py`, prompt `extract_stats.md`; Test `tests/test_extract_stats.py`.
**Interface:** `extract_test_statistics(client, paper, protocol, dataset, outcome_key, sources, model) -> list[Candidate]` — candidates with `kind="test_statistic"` (`stat_type ∈ t|F|p`, value, df1, df2, direction ∈ "A>B"|"A<B"|"unknown", design ∈ "between"|"mixed"|"unknown", quote, page) or `kind="reported_d"` (value, `positive_means` "A>B"/"B>A", is_hedges_g). Only between-subjects main effects of group on the target outcome are admissible (flag interactions/mixed designs `admissible=False` with reason).
- [ ] Steps 1–5 (fixture replay on Bock: F(1,22)=7.58 for Age on adaptation phase flagged as `design="mixed"`, admissible only as fallback with note); commit `feat(agents): test-statistic and reported-d extractor`.

### Task 8: Verification layer
**Files:** Create `canopy/verify/checks.py`, `canopy/verify/vote.py`, `canopy/agents/verifier.py`, prompts `verifier.md`, `adjudicator.md`, `orientation.md`; Test `tests/test_checks.py`, `tests/test_vote.py`, `tests/test_verifier.py`.
**Interfaces:**
```python
def run_checks(dataset: DatasetSpec, outcome_key, candidates: list[Candidate]) -> list[CheckFlag]   # rules from spec §3.3(1); CheckFlag(code, severity, message, candidate_ids)
def vote(candidates: list[Candidate], axis_range: float|None) -> VoteResult   # per group: agreed value(s), method, tolerance, agreeing_ids, disagreeing_ids
def verify_candidate(client, paper, cand: Candidate, context_blocks, model) -> VerifierVerdict   # confirmed|refuted|ambiguous + reason + alt values, model MUST differ from cand.model
def adjudicate(client, paper, dataset, outcome, candidates, verdicts, flags, model="claude-opus-5", effort="xhigh") -> Adjudication  # final values per group + rationale + needs_human
def orientation(client, paper, dataset, outcome_sources, model) -> OrientationVerdict   # higher_is_better bool + quotes; run twice (Opus, Sonnet); disagreement → needs_human
def confidence(vote, verdicts, flags, adjudication) -> tuple[Literal["auto_accept","accept_with_note","needs_human"], float, list[str]]
```
- [ ] Steps 1–5 with unit tests for each rule/scoring path (no LLM for checks/vote/confidence; verifier/adjudicator/orientation replay fixtures); commit `feat(verify): checks, voting, adversarial verifier, adjudicator, orientation, confidence`.

### Task 9: Effect-size resolution
**Files:** Create `canopy/pipeline/resolve.py`; Test `tests/test_resolve.py`.
**Interface:** `resolve_effect(dataset, outcome_def, resolved_values: ResolvedValues, settings: StatsSettings) -> EffectSizeRecord` where `ResolvedValues` holds per-group `{n, mean, dispersion_value, dispersion_type, ci_level}` or a test statistic or reported d, plus `higher_is_better`. Route precedence from settings; conversions via `canopy.stats.effect_sizes` (SE→SD `sd_from_se`, CI→SD `sd_from_ci` with `dist='t'` when n<60 else 'z' per Cochrane 6.5.2.2, IQR→SD `sd_from_iqr`, points→mean/SD); `conversion_chain: list[str]` human-readable; `route` string; optional digitization variance added: `var_total = var + (dd/dm_a·σ_a)² + …` via numerical partial derivatives (finite differences).
- [ ] Tests: means_sd route reproduces Bock d=−1.6758 (higher_is_better False, A=old); SE route; CI route with t; t-stat route; reported g; precedence when several routes exist; chain text contains the numbers. Commit `feat(pipeline): effect-size resolution with conversion chains`.

### Task 10: Orchestrator + CLI
**Files:** Create `canopy/pipeline/run.py`, `canopy/pipeline/state.py`, `canopy/cli.py`; Test `tests/test_pipeline_offline.py` (end-to-end on Bock with all LLM calls replayed).
**Interfaces:**
```python
def run_pipeline(papers_dir: Path, protocol_path: Path, out_dir: Path, *, models: dict|None=None, budget_usd: float|None=None, resume: bool=True, max_papers: int|None=None, concurrency: int=4, progress: Callable[[dict],None]|None=None) -> RunManifest
```
Stages per paper (each writes `<out>/papers/<sha12>/<stage>.json`, skipped when present and `resume`): ingest → map → extract (per dataset×outcome: text variants, stats, digitize if figure sources) → verify (checks, vote, verifier, adjudicate, orientation, confidence) → resolve. Then per outcome: pool (`canopy.stats.meta.random_effects` with settings) → outputs (Task 11). Manifest: protocol hash, per-paper status/timings/cost, warnings, `human_review_queue`. Progress callback events `{stage, paper, status, cost_so_far, message}`. CLI: `canopy run --papers DIR --protocol FILE --out DIR [--budget-usd] [--max-papers] [--concurrency] [--no-resume]`, `canopy serve [--port]`, `canopy protocol init NAME`, `canopy validate RUN_DIR`.
- [ ] Tests: offline e2e produces `manifest.json`, `results/late_adaptation/forest.png`, extraction table with ≥1 row; resume skips completed stages (assert LLM client call count 0 on second run). Commit `feat(pipeline): resumable orchestrator + CLI`.

### Task 11: Pooling, forest plot, tables, provenance, methods figure, HTML report
**Files:** Create `canopy/report/{forest,tables,html,methods_fig}.py`; Test `tests/test_report.py`.
Forest plot per spec §3.5 (read the `dataviz` skill palette guidance: neutral greys, one accent for the pooled diamond, colorblind-safe; label columns: Author, Year, moderators, N(A/B); square area ∝ RE weight; CI whiskers; diamond; PI bar; footer with RE model conventions and heterogeneity; x labels from protocol direction labels; sorted by effect). Outputs PNG (300 dpi), SVG, PDF. Tables: `extraction_table.csv/json/xlsx` (one row per dataset×outcome with all raw values, route, chain, confidence, flags, provenance paths). Methods figure: bar/donut of routes (% datasets) + up to 3 example thumbnails per route. HTML report: static page linking everything (used by the server too).
- [ ] Tests: forest plot renders for the Cisneros gold TE/seTE (k=50) without error and file sizes > 0; tables have expected columns. Commit `feat(report): forest plot, tables, methods figure, html report`.

### Task 12: Web UI (FastAPI + static SPA)
**Files:** Create `canopy/server/app.py`, `canopy/server/jobs.py`, `canopy/server/static/{index.html,app.js,styles.css}`; Test `tests/test_server.py` (TestClient: create job with 1 tiny PDF and offline replay → status → results JSON).
Endpoints: `POST /api/runs` (multipart: files[] from `webkitdirectory` upload + protocol yaml/json + options) → run_id; `GET /api/runs/{id}/events` (SSE progress); `GET /api/runs/{id}` (manifest); `GET /api/runs/{id}/results/{outcome}` (pooled + rows); `GET /api/runs/{id}/evidence/{dataset}/{outcome}` (candidates, verdicts, crops/overlays as URLs); `GET /api/runs/{id}/files/…` (static, path-traversal safe via `Path.resolve().is_relative_to`); `POST /api/runs/{id}/cancel`; `GET /api/protocols/examples`. UI: New run (protocol editor pre-filled from example; folder picker; model/budget), Run monitor (per-paper stage grid + cost meter + log via SSE), Results (inline SVG forest with hover/click → evidence drawer showing page image with highlighted quote or figure crop with overlay, candidates table, verifier text, conversion chain), Flags queue, downloads. Mobile: single column, sticky tabs. Use the `frontend-design` skill guidance (no generic AI look; keep it restrained and legible).
- [ ] Tests as above; commit `feat(server): local web UI with SSE progress and evidence drawer`.

### Task 13: Validation deliverables (Cisneros)
**Files:** Create `validation/scripts/{run_cisneros.py, compare_manual_vs_auto.py, extraction_routes_figure.py, example_text_ms.py, example_figure_only.py, example_test_statistic.py, replot_forest.py}`, `validation/README.md`.
`run_cisneros.py` runs the pipeline on the papers folder with `examples/protocols/aging_sensorimotor_adaptation.yaml`, profile `cisneros2024`. `compare_manual_vs_auto.py` joins run rows to `validation/reference/cisneros2024/{late,aft}_gsheet.csv` by (normalized first author, year, experiment/condition, N pair) → scatter manual vs auto d with CIs (both axes), identity line, concordance (Lin's CCC), MAE, sign agreement; writes `validation/out/manual_vs_auto_<outcome>.png` + CSV of discrepancies with classification columns. `extraction_routes_figure.py` → route % + example crops. Example scripts each run ONE paper end-to-end and print/plot exactly where the number came from (page highlight / overlay / quote + chain). `replot_forest.py` → auto forest plots (+ side-by-side with the manual TE/seTE forest).
- [ ] Runnable without re-calling the API when the run cache exists; commit `feat(validation): Cisneros validation scripts and figures`.

### Task 14: README, docs, handoff
**Files:** `README.md` (install, `.env`, run CLI/UI, protocol how-to, cost expectations, license note AGPL-3), `docs/architecture.md` (pipeline figure inputs), memory update.
- [ ] Commit `docs: README and architecture`.

## Self-review notes
Spec coverage: §3.0 ingestion (done pre-plan) · §3.1 T4 · §3.2 T5–T7 · §3.3 T8 · §3.4 T9 · §3.5 T11 · §3.6 T10/T12 · §3.7 T2/T10 · §4 T13 · §5 T13/T14. Types: `Candidate`, `StudyMap`, `DatasetSpec`, `Source`, `EffectSizeRecord`, `LLMClient`, `PaperRecord`, `FigureRegion` used consistently (defined in T1/T2 and existing `canopy/ingest`).
