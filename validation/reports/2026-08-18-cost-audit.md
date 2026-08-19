# Canopy API-cost audit

Scope: read-only code + artifact review of `/Users/sritejpadmanabhan/Projects/canopy`. No API calls made,
no `.env` read, no pipeline/tests run. Evidence is file:line citations from the repo plus data pulled from
`runs/nine/` (papers/*/extract.json, manifest.json, run.log, run1.log, run2.log).

**Important data-provenance note up front:** `runs/nine/manifest.json` currently reflects a small, partial
rerun (matches `run.log`: digitize $2.40/402 calls, verify $6.48/73 calls, map $0.41/2 calls, extract
$0/1 call — all for a single paper, `d1f2946e7e81`). The fuller 9-paper attempt that matches the owner's
quoted "$6–11/paper, digitize≈65%" figures is captured only in `run2.log`'s printed summary (the manifest
that produced it was overwritten by the later partial rerun). Where the two disagree, this report says
which log a number comes from.

---

## 1. Prompt caching in `canopy/llm/client.py`

**Mechanics.** `EPHEMERAL = {"type": "ephemeral"}` (client.py:34) is the only cache_control value used.
Two low-level helpers attach it to a single content block: `image_block(..., cache: bool = False)`
(client.py:87–95, **default off**) and `pdf_block(..., cache: bool = True)` (client.py:97–109, **default
on**). `mark_cacheable()` (client.py:117–133) copies a block list and stamps the marker on the **last**
block only, and `tool_loop()` (client.py:376–461) moves that marker forward turn-by-turn, capping the
rolling window at `MAX_CACHE_MARKERS = 4` (client.py:112, enforced client.py:459–461) — i.e. never more than
4 breakpoints in flight, well under the API's per-request cap.

Critically, `client.py` **never marks `system` itself** — `system` is a plain string, passed straight
through in `providers.py:181–182` (`kwargs["system"] = req.system`). Caching of the system prompt and tool
list happens implicitly: a single cache_control breakpoint on a later message block caches everything
*preceding* it too (tools → system → messages is the API's fixed prefix order), so one marker on the figure
image is enough to cache the system prompt and tool definitions along with it. This is spelled out in the
digitizer's own comment at `canopy/digitize/vlm.py:715–724`.

**Where digitize's prompt is assembled:** `canopy/digitize/vlm.py`, function `_run()` (vlm.py:706–733,
shared by `read_out()` and `coords()`), and the top-level orchestrator `canopy/digitize/digitizer.py:
digitize()` (2014–2298). `FigureView.image_block(cache: bool = False)` at vlm.py:511–515 is what actually
stamps the marker when called with `cache=True` (vlm.py:728: `view.image_block(cache=True)`).

**Order — confirmed static-first, then variable**, exactly as hypothesized:
- `system` = `load_prompt("digitize_readout")` (vlm.py:~752) → `canopy/llm/prompts/digitize_readout.md`,
  66 lines of protocol/instructions (axis identification, unit handling, zoom guidance), **byte-identical
  across every model/variant/sample** for a given figure (comment at vlm.py:750–751: "the system prompt is
  IDENTICAL for every variant").
- user message content, in order (vlm.py:727–730): `[header text, image (cache_control HERE)]` → then
  `task` block → then closing `ask` string.
- `task` is rendered from `canopy/llm/prompts/digitize_task.md` (11 lines: `{{TARGET}}`, `{{CAPTION}}`,
  `{{VARIANT}}` — the per-cell locator and per-pass instruction) and is appended **after** the cache
  breakpoint, so it is never itself cached, by design (vlm.py:717–724 docstring: "everything that varies
  ... comes after [the image]").
- `tools()` (vlm.py:600–636: `crop_image`, `overlay_points`, conditionally `list_regions`) are also static
  per figure and therefore ride inside the same cached prefix.

**A deliberate non-use of caching:** `canopy/agents/mapper.py:1592–1596` explicitly skips cache_control on
the mapper's whole-paper document block because the mapper's four calls each use a **different output
schema**, and schema is part of the cached prefix — marking would pay the 1.25× write premium on every call
and never get a read. This is a documented, reasoned decision, not an oversight.

**Whole-paper verifier prompt:** `whole_paper()` at `canopy/agents/verify_common.py:205–224` builds the
shared PDF document block with `cache=True` by default, and is called from `verifier.py:118`,
`orientation.py:114`, and `adjudicator.py:200` — all three pass the same `SYSTEM` constant
(`verify_common.py`) as `system=`, so repeat calls in each stage (once per candidate/measure/adjudication)
can read the cached PDF+system prefix instead of rewriting it.

**Second-family (sonnet-5) share of digitize spend — caches are per model, so this portion can never read
the opus-5 prefix.** `protocol.yaml` (`runs/nine/protocol.yaml:101–106`) sets `readouts_min: 2`,
`readouts_max: 3`. `_readout_plan()` (`canopy/digitize/digitizer.py:779–816`) **always** buys the first two
read-outs from two *different* model families — `(primary=claude-opus-5, "direct")` then
`(secondary=claude-sonnet-5, "direct")` — before any adaptive stop is even considered (comment,
digitizer.py:781–791: "a second family is what makes the agreement mean something, and it is bought before
any adaptive stop"). Every digitize cell therefore contains at least one guaranteed sonnet-5 call that can
never reuse the opus-5 cache prefix (or vice versa).

Measured from `runs/nine/papers/*/extract.json` (6 papers that reached extract; candidates whose
`extractor_id` starts with `digitize:`, n=249): of the 185 candidates attributable to an actual VLM read-out
(path D), **134 are opus-tagged (72.4%) and 51 are sonnet-tagged (27.6%)** (the remaining 64 digitize
candidates — 58 `ensemble`, 4 `raster_cv`, 2 `vector` — are synthesized/deterministic, not separate model
calls). So roughly **a quarter to a third of digitize's read-out call volume runs in a cache family that
structurally cannot be a cache hit against the opus prefix** — it's either that call's own first-ever write,
or (on a later cell touching the same figure) a same-family read, but never a read of the opus family's
entries. This is a direct, mechanical explanation for part of the "org's cache hit rate is low" email: it
isn't (only) a bug, it's a designed-in floor.

Aggregate cache performance actually observed: `run.log` (smaller rerun) — "69% of input tokens read from
cache" (2,218,167 read / 740,284 written); `run2.log` (fuller 9-paper run) — "60% of input tokens read from
cache (5,666,358 read, 2,953,473 written, 835,261 fresh) — saved about $19.16". `manifest.json`'s current
`cache` block (matching `run.log`): `cache_hit_ratio: 0.6879`, `saved_usd: 6.566891`.

---

## 2. Call fan-out for one cell

One `digitize()` call (`canopy/pipeline/run.py:493–499`) handles one `(figure, target)` pair; a target
commonly spans **two groups (A/B)**, both read together inside a single read-out call, so a single-group
"cell" is roughly half of one `digitize()` invocation's read-out cost — coords/overlay/CV calls are shared
across both groups.

Fan-out inside one `digitize()` call, under the current `runs/nine/protocol.yaml` settings
(`readouts_min=2, readouts_max=3, overlay_verify=on_disagreement, max_tool_calls=6`):

1. **Path D read-outs (VLM, `read_out()`, vlm.py:743–769).** `_readout_plan()`
   (digitizer.py:779–816) always buys `readouts_min=2` up front: `(opus-5, "direct")`,
   `(sonnet-5, "direct")` — two distinct model families, never the same family twice in the guaranteed
   slice (digitizer.py:2075–2076, the `for spec in plan[:n_min]` loop).
2. **Early stopping — yes, confirmed.** Before buying read-out #3, `_needs_another_readout()`
   (digitizer.py:1935–1978) is checked (loop at digitizer.py:2116–2130). It says "stop" only when, per
   group: ≥2 usable routes exist, those routes span ≥2 model families, **and**
   `dual_tolerance()` (digitizer.py:192–224) says the routes agree on *both* the mean and the error
   half-length within tolerance. If any of those fail, read-out #3 is bought:
   `(opus-5, "ticks_first")` (next pair in `_readout_plan`'s order, digitizer.py:804–809).
3. **Path C coords (VLM, `coords()`, vlm.py:838–851).** Exactly 1 call, opus-5/primary only, **bought
   unconditionally** — not gated by agreement (digitizer.py:2088).
4. **Path B (raster CV) / Path A (vector).** Not LLM calls — deterministic, reuse path C's pixel
   coordinates or the PDF's own vector geometry (digitizer.py:2100–2103, 2144–2145).
5. **Partial-reread.** Up to 1 extra call, only if a route reported a spread with no mean
   (`_rebuy_partial`, digitizer.py:2148–2156).
6. **Overlay-verify (VLM, `overlay_verify()`).** Up to `MAX_OVERLAY_ITERATIONS = 2`
   (digitizer.py:48, loop digitizer.py:2186–2200), opus-5/primary only, gated by `_overlay_wanted()`
   (digitizer.py:1984–2012) under the `on_disagreement` policy — fires on a series conflict, a dropped
   sample, a zero-confidence pixel snap, **or** the same disagreement condition as #2; otherwise 0 calls.

**Net per `digitize()` invocation** (≈2 group-cells): best case (early agreement, no overlay) = **3 VLM
tool_loop calls** (2 read-outs + 1 coords); worst case = **7** (3 read-outs + 1 coords + 1 reread + 2
overlay). Each of those is itself a `tool_loop()` that can spend several more *actual* API turns on
`crop_image`/`overlay_points` zoom before it submits (capped at `max_tool_calls=6` for read-outs via
`MAX_READOUT_TOOL_CALLS`, vlm.py:52, and 8 generally, vlm.py:51) — this is where a manifest's raw "N calls"
count (e.g. run2.log's 704 digitize calls) balloons past the ~3–7 logical passes per cell.

Retries are a separate, exceptional-path mechanism, not part of the routine multiplier: `client.py:527–535`
doubles `max_tokens` and retries once on truncation; `client.py:44–49` reissues once on a "degenerate reply"
(too-short/garbled justification). Both are rare and orthogonal to the adaptive fan-out above.

---

## 3. Wasted calls

From `runs/nine/papers/*/extract.json` → `candidates`, across the 6 papers that reached extract
(`5039533c85ef`, `b7523a41b03a`, `d1f2946e7e81`, `3570e4ce2a9c`, `592b3b55a318`, `b511dbb76fa6`):

| status | count | % of total | mean=None | mean=None % |
|---|---|---|---|---|
| found | 162 | 61.6% | 4 | 2.5% |
| ambiguous | 97 | 36.9% | 72 | 74.2% |
| not_on_these_pages | 4 | 1.5% | 4 | 100% |
| **total** | **263** | | | |

By source route: `figure` (=digitize) 249 (94.7%), `text` 8, `test_statistic` 4, `reported_d` 2 — digitize
dominates candidate volume, consistent with it dominating cost.

**Digitize-specific:** of the 249 digitize-attributed candidates, **70 have `mean=None` — 28.1% of digitize
candidates produced no usable value.** Model split among the 185 model-attributed (path-D) candidates: opus
134 / sonnet 51, as above.

Per-paper "no-return" rate varies a lot (ambiguous-or-worse ÷ total candidates): `b511dbb76fa6` 9%,
`5039533c85ef` 10%, `3570e4ce2a9c` 15%, `b7523a41b03a` 46%, `592b3b55a318` 47%, `d1f2946e7e81` 60% — poor
figure legibility on a handful of papers drives a disproportionate share of spend-with-no-return; this isn't
evenly distributed.

**manifest.json `cost_by_stage`** (current snapshot — the single-paper `d1f2946e7e81` rerun, matches
`run.log`): digitize $2.399392/402 calls, extract $0/1 call, map $0.409452/2 calls, verify $6.480135/73
calls; total `cost_usd` 9.288979, `n_llm_calls` 478, `cache_hits` 367, `cache_hit_ratio` 0.6879.

**`run2.log`** (the fuller 9-paper attempt, not reflected in the current manifest): `by stage: digitize
$22.38/704c · map $6.46/30c · verify $5.42/48c · extract $0.21/6c` → **total $34.47 / 788 calls**, prompt
cache "60% of input tokens read from cache (5,666,358 read, 2,953,473 written, 835,261 fresh) — saved about
$19.16". Digitize = 22.38/34.47 = **64.9% of spend** — matches the owner's ~65% figure. Of the 9 papers: 4
fully resolved, 2 excluded pre-digitize (map-only cost), 1 errored on an unrelated PyMuPDF bug
(`fz_pixmap_xres_set` TypeError, paper `790ab2b0bc43`), and **3 papers errored after hitting an $8.00
per-paper budget cap** (`d1f2946e7e81` spent $8.0536, `3570e4ce2a9c` $8.2156, `592b3b55a318` $8.3622 — all
just over the $8.00 allowance). Those 3 papers alone account for **$24.63 of $34.47 (71.4%) of the run's
total spend**, and none of them produced a resolved/pooled result in that run. (The underlying calls are
disk-cached — `runs/nine/cache/*.json`, 1092 files — so a rerun with a higher cap would replay most of that
spend near-free rather than pay it twice; but as shipped, that money bought nothing usable without a manual
retry.) This is arguably the largest concrete "waste" finding in the whole audit, separate from prompt
caching.

---

## 4. Message Batches API

**Not used anywhere.** `grep -rn "batches\|message_batches\|MessageBatch" canopy/` returns zero matches in
any `.py` file.

**Would the pipeline structure allow it?** Partially, and only for a specific slice. The blocker is that
almost every layer is adaptive/synchronous by design:
- Digitize's own `tool_loop()` (client.py:376–461) needs a live round-trip for each `crop_image`/
  `overlay_points` zoom turn — the Batches API's up-to-24h async window cannot support a mid-conversation
  zoom decision at all. Batching could only replace the **first turn** of a read-out/coords call, and only
  helps when that turn already terminates without zooming.
- The adaptive "buy read-out #3 only if #1 and #2 disagree" logic (`_needs_another_readout`,
  digitizer.py:1935–1978) needs #1 and #2's results before it can even decide whether to submit #3 — by
  construction this cannot be pre-batched with the first two.
- The **two mandatory upfront read-outs** (`readouts_min=2`, opus+sonnet, digitizer.py:2075–2076) are the
  one clean exception: neither depends on the other's outcome, so both could be submitted to one batch
  together. That's the actual scope of what's batchable without a design change.
- `coords()` (always bought, digitizer.py:2088) is independent of the read-outs too and could join the same
  batch.
- Verify/adjudicate calls (`verifier.py`, `adjudicator.py`) are more sequentially adjudication-shaped
  (verify a candidate → maybe reopen → maybe adjudicate) and would see less benefit.

**What would need to change concretely:**
1. `LLMClient._call()` (client.py:487–553) currently assumes synchronous request→response; it would need a
   "pending batch" state — submit now, resolve later — which the disk-cache layer (`cache.py`,
   `DiskCache.get/put`) and the replay-fixture layer (`ReplayProvider`) don't have a concept of today.
2. The live per-paper circuit breaker (`_reserve`/`_release`/`_assert_affordable`, client.py:301–332, which
   is exactly what's hitting the $8 cap in §3) doesn't compose with batches: a batch commits spend for the
   whole submitted set upfront, so the graceful-stop-mid-run behavior would need to move to *before*
   submission (decide the whole batch's scope up front) rather than the current call-by-call check.
3. `costs.py` (`estimate_cost`) would need to record the 50% batch discount as a distinct rate, and
   `cost_by_stage` bookkeeping would need a "batched" call source alongside `live`/`disk_cache`/`replay`
   (`LLMCall.source`, client.py:203).
4. Turnaround time changes from the current single-paper ~10 minutes (run2.log: 9 papers / 3176s ≈ 353s/
   paper) to potentially hours; realistic use is an offline full-corpus (re)run, not the interactive
   single-paper path.

---

## 5. Model routing

`MODELS` (`canopy/config.py:16–19`): `primary=claude-opus-5` (mapper, extractors, digitizer read-outs),
`secondary=claude-sonnet-5` (cross-check/second route/verifier), `adjudicator=claude-opus-5` (**same tier as
primary**, not an upgrade), `adjudicator_max=claude-fable-5` (optional max-effort adjudicator).

`MAP_ADJUDICATOR` (`mapper.py:1849`) is **not a model** — it's the string label `"map-adjudicator"` used
only to attribute a decision in the review log (mapper.py:1846–1852). The mapper's actual adjudication
calls run on `model_adjudicate: str = MODELS["adjudicator"]` (mapper.py:1589) = **claude-opus-5**. Mapper
adjudication never touches fable-5.

The only place `MODELS["adjudicator_max"]` (fable-5) is wired up at all is `TIEBREAK_MODEL` in
`canopy/agents/orientation.py:68`, and it is **off by default**: "`tiebreak=None` reproduces HEAD exactly,
no call, no cost" (orientation.py:64–67). When explicitly turned on, it buys "at most one extra ballot per
(outcome, measure), only where the free deterministic check already abstained, from a model family that
differs from at least one of the first two readers" (same comment). So in the runs/nine data, fable-5 spend
is effectively zero — it's a narrowly-scoped, currently-dormant escape hatch, not a routine cost driver.

Given this, and given the instruction not to recommend downgrading readers (opus 0.015–0.023 mean |d| error
vs sonnet 0.151 from the earlier audit): the model-tier risk here isn't "using too expensive a model
somewhere it shouldn't" — fable-5 is barely used and opus/sonnet are both doing accuracy-load-bearing work.
The available levers are about **how often** calls happen (fan-out, batching, graceful degradation), not
**which model** answers them.

---

## 6. Ranked cost levers

Percentages are estimates grounded in the numbers above (run2.log's $34.47/9-paper run, digitize=65%,
cache hit ~60%, 249 digitize candidates with 28.1% mean=None, sonnet=27.6% of read-out candidates, 3/9
papers losing 71.4% of run spend to an unhandled budget-cap error). They are directional, not audited to
the cent — per the project's own norm (memory: "Team decisions, not solo"), any of these should go through
panel → synthesis → adversarial review → implement → review → measure-on-cache before shipping, exactly as
`project_canopy_ceiling_build.md` already does.

**1. Make the per-paper budget cap fail soft instead of erroring out (~20–30% of nominal run cost
recovered from "produced nothing").**
What changes: when a paper hits its $8 allowance mid-digitize, finish with whatever candidates exist
(flag the paper's *later* cells as thin-coverage) instead of raising and discarding the run's output for
that paper. Grounded in: 3 of 9 papers in run2.log spent $24.63 combined (71.4% of the run) and produced no
resolved/pooled output; the spend is disk-cached (`runs/nine/cache/`) so it isn't literally gone, but it is
functionally wasted unless someone knows to retry with a higher cap. Accuracy risk: **low-moderate** — a
paper cut off mid-way has fewer read-outs/less overlay-verification for its later cells than for its early
ones, so those outputs need an explicit "partial coverage" flag into human review rather than being pooled
at full confidence. This is a control-flow fix, not a caching/architecture change — cheapest of the five to
implement.

**2. Batch the two mandatory upfront read-outs (opus+sonnet) via the Message Batches API
(~10–15% of total run cost).**
What changes: submit `(primary, "direct")` and `(secondary, "direct")` — the two calls `_readout_plan`
always buys regardless of agreement (digitizer.py:2075–2076) — as one Batches API job at 50% off, instead of
two live calls; keep the adaptive 3rd read-out, coords, reread, and overlay calls live (they depend on the
first two's results or on zoom). This is the one slice of digitize that is both (a) unconditional — always
bought — and (b) mutually independent, so it doesn't fight the adaptive-stop logic in §2 above. Accuracy
risk: **low** (identical model/prompt/schema, just billed and returned differently) but real engineering
cost: `LLMClient._call()` needs a pending-batch state, the live per-paper budget breaker needs to move to
"decide batch scope up front," and turnaround grows from minutes to potentially hours — best suited to an
offline full-corpus run, not interactive single-paper checks.

**3. Make route-C `coords()` conditional instead of unconditional (~8–12%).**
What changes: today `coords()` (1 opus call) is bought on *every* `digitize()` invocation regardless of
whether the two mandatory read-outs already agreed (digitizer.py:2088) — it isn't gated by
`_needs_another_readout` the way read-out #3 and overlay-verify are. Skipping it when read-outs already
agree across families **and** the CV pass found no bars/markers worth snapping to (`core.bars`/
`core.markers` empty) would remove ~1 of a cell's 3–4 calls on the "easy" fraction of cells. Accuracy risk:
**moderate-to-high** — route C/coords is what lets route B's raster-CV pixel-snap corroborate route D, and
it's part of `_choose_calibration`'s axis-ladder voting (digitizer.py:621–748); cutting it removes an
independent corroborating witness family, which is exactly the mechanism the earlier audit's accuracy
numbers depend on. Needs the most scrutiny of the five before implementing — a candidate for the panel
process, not a quick flip.

**4. Tighten `overlay_verify`'s trigger or cap it at 1 iteration instead of 2 (~3–7%).**
What changes: `_overlay_wanted()` (digitizer.py:1984–2012) fires on more than pure quantitative
disagreement — also on any dropped sample or a zero-confidence pixel snap — and when it fires it can run up
to `MAX_OVERLAY_ITERATIONS=2` (digitizer.py:48) full opus tool_loop calls. Narrowing the trigger to
disagreement-only, or capping at 1 iteration, trims a bounded slice of digitize spend. This is the softest
estimate in this report — it needs one more measurement pass (how many cells actually trigger overlay, and
how many resolve on iteration 1) before sizing it precisely; I didn't have a clean per-call route/tag census
for it within budget. Accuracy risk: **moderate** — overlay-verify is a real error-catcher (it exists
specifically to catch a mark that landed on the wrong datum); a 2nd iteration exists because the 1st
sometimes doesn't fully resolve it.

**5. Give the mapper's 4 schema-diverse calls a shared cache prefix instead of opting out entirely
(~2%).**
What changes: `mapper.py:1592–1596` currently disables caching entirely because each of the mapper's 4
calls uses a different output schema and schema sits inside the cached prefix. Restructuring so the
invariant part (whole-paper PDF + protocol + roster text) carries the cache_control breakpoint and the
schema-specific difference lives in a block *after* it — the same trick `vlm.py:_run()` already uses for
digitize (static image first, variable task after) — could let the mapper's 4 calls share a prefix read
instead of each writing fresh. Sized modestly because map is a smaller stage (run2.log: $6.46/$34.47 =
18.7% of run cost) — even a 60–70% hit rate on it (in line with what digitize/verify already achieve) is a
small slice of total spend. Accuracy risk: **low** (pure request-shape change, not a reasoning change), but
mapper.py's own comment shows this exact tradeoff was already considered once and rejected as not worth
getting wrong — a naive attempt risks paying the 1.25× write premium with no matching future read.

### 50-paper projection

Owner's anchor: ~$6–11/eligible paper, digitize ≈ 65% of spend, ~100 digitize calls/paper. In the run2.log
sample, 6 of 9 discovered papers were eligible-and-attempted (2 excluded pre-digitize by protocol criteria,
1 errored on an unrelated PyMuPDF bug); of those 6, half (3) hit the $8 budget cap before finishing.

**Current design, 50 papers, no levers:**
- ~28–33 eligible papers (using the observed ~55–65% eligibility mix) × $6–11 ≈ **$170–360** in nominal
  digitize+map+verify spend, of which ~65% (**~$110–235**) is digitize.
- At the observed ~1/3 cap-failure rate among eligible papers, **~9–11 of those papers stall at the $8 cap**
  and produce no resolved output without a manual retry — roughly **$70–120 of nominal spend sunk into
  papers that finish as "error," not as usable rows** in the pooled forest plot, unless someone notices and
  reruns them (cheaply, off the disk cache).

**With levers 1–5 applied (directional, not additive at face value — they overlap on digitize call
volume, so treat this as one combined estimate, not five stacked percentages):**
- Lever 1 alone converts essentially all of that ~$70–120 "stalled" spend into usable output — the 50-paper
  batch finishes with close to 50 usable papers' worth of extraction instead of ~39–41.
- Levers 2–5 combined trim roughly **20–30% off raw digitize/map token spend** (batching the two mandatory
  read-outs is the biggest piece; conditional coords and a tighter overlay trigger add smaller, riskier
  increments; mapper caching is marginal). Net nominal spend for the same 50-paper batch: **~$120–260**
  instead of $170–360.
- Combined effect: roughly **25–30% lower raw token spend, and close to 100% (vs ~65–70% today) of that
  spend converted into usable pooled output** — the effective cost per *usable* paper drops by something
  closer to 40–50%, because the biggest single problem this audit found isn't the cache hit rate itself,
  it's that a third of eligible papers currently buy nothing at all.

All five levers should go through the project's standard panel → synthesis → adversarial review →
implement → review → measure-on-cache loop before shipping — none of the accuracy-risk claims above are a
substitute for that.
