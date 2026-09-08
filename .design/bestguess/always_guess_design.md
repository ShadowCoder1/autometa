# Design: "always guess" best-guess tier + unit refusal + AutoMeta rename

Date: 2026-09-07. Panelist output; input = always_guess_brief.md. Status: for adversarial review.
Every anchor below was read in the working tree at design time.

## Verified ground (corrections and confirmations to the brief)

- `StatsSettings.best_guess_rules` gates ONLY the answer tier: `_cell_context`
  (bestguess.py:709-724) returns `None` when the list is empty, and `best_guess_rows`
  then falls through to `_decide` — the ROW rules (`RULES`, bestguess.py:67) and vetoes run
  unconditionally on every held row. `[]` pins "answer tier off", not row-rule behavior.
  The pinned `[]` tests (test_bestguess_cells.py:145, :790-842, :1025) compare off-vs-on
  **within one build**; a row-rule change moves both sides together, so they stay green.
- Fixture facts (runs/nine, read from resolve.json): the ONLY contradicted row with a live es
  is `b7523a41b03a:d2` LA (es +0.831, axis_conflict + series_identity_conflict). The other
  contradiction-vetoed rows (`3570e4ce2a9c:d2` LA/AE, `d1f2946e7e81:d1` AE) have es=None and
  fall through to the no-value vetoes. `3570e4ce2a9c:d2` AE carries the real `sign_mismatch`
  but no es; the sign-veto tests use a constructed row with es=-0.128.
- Resolved-pair units: scanning all six papers' verdicts with `verify.units.unit_key`, NO
  (dataset, outcome) pair in runs/nine has two non-empty, differing arm unit keys — so the unit
  refusal fires on zero fixture rows and both fixture lines are byte-identical.
- The web UI is ALREADY AutoMeta almost everywhere (index.html:8,17; html.py:209,702). Only
  stragglers remain (list in §Rename).

## Q1 — sign_mismatch: ADMIT at the resolved row's own value (never re-signed)

**Decision.** Delete both `contradicted_value` branches of `_veto` (bestguess.py:216-229,
flag-driven and `sign_mismatch`). A held row carrying a CONTRADICTING flag or `sign_mismatch`
that has a usable es is admitted under one new row rule, **`disputed_reading_guess`**, at the
es/var the resolver built — nothing recomputed, re-signed or rescaled. A contradicted row with
no es still dies at `one_group_only`/`no_variance` (the honest "cannot").

**Rationale.** The user's rule is decisive: the only reason not to guess is that the paper does
not provide the data; every one of these rows carries a number the resolver built. The recorded
evidence runs one way: Bagesteiro's 4 sign_mismatch AE cells, Scarpina's series conflicts and
Redding's two-places conflict were all later admitted by the human review at (mostly) the tool's
own values; post-review, the two live-es held rows (series_identity_conflict +0.386,
locator_reads_conflict +0.098) are exactly the rows the manual prompting had to rescue. There is
no case in the record where the resolved MEANS were on the wrong side and the paper's prose was
right. Note the sign_mismatch row is not unsigned: `higher_is_better` is settled and the es is
signed by the extracted means — the line still "never resolves a sign it does not have"
(orientation veto stays, Q4); what changes is that a sign the row HAS but the prose disputes is
shown with its dispute instead of suppressed. `orientation_reader_contradicts_values` (a member
of CONTRADICTING_FLAGS) admits under the same rule for the same reason — it is the same class of
means-vs-claim conflict. The MAJOR-2 failure (Heuer flipping the AE conclusion silently) is
answered by loudness, not suppression: the rule name is distinct, the reason opens with the
dispute, the caveat counts it (§Q5), and `sign_agrees_with_strict` + "points the other way"
(conclusion.py:408-412) still fire.

One rule name for both cases (not two): the payload/caveat counts stay legible, the surface
stays minimal, and the evidence dict distinguishes them (`contradicting_flags` carries the
flags verbatim, `sign_mismatch` included when present).

## Q2 — no knob for the row rule

**Decision.** `disputed_reading_guess` joins `RULES` ungated, like every row rule before it.
No change to `BEST_GUESS_RULE_NAMES` (models.py:125), the validator (models.py:197-205), or the
assert (bestguess.py:618) — those are the ANSWER-rule namespace and stay `("adjudicated_value",
"route_precedence", "agreed_solo_read")`.

**Rationale.** Row rules were never protocol-gated (`low_confidence_value` is not), and the knob's
closed set is validated against the answer catalogue — putting a row name in it would corrupt that
namespace and break test_bestguess_cells.py:119-125 for nothing. Primary results cannot move (best
guess only); the best-guess line of an old run growing rows on repool is the accepted, intended
effect (user: yes, explicitly). A protocol that must pin the old line pins the build, which is
already the rule for DECISION A behavior.

## Q3 — unit refusal now, pair re-selection as a recorded follow-up

**Decision.** New row-refusal code **`resolved_unit_mismatch`** (a NEW name — NOT the existing
candidate-level `unit_mismatch`), raised by `resolve._finish` when the RESOLVED pair's units
disagree. Pair re-selection ("pick the corroborated same-unit pair", what the human did for
Sainburg d2) is out of scope: it changes which numbers the resolver divides — a vote/adjudication
-adjacent change that needs its own panel — and refusal is correct meanwhile (a guess about
mm-vs-ratio is not a guess about the contrast). Record the follow-up in the brief's follow-up list.

**Why a new code (load-bearing).** `ROW_REFUSAL_CODES` is a row-level contract ("codes neither
cell carries", confidence.py:774-788), and `bestguess.VETO_ROW_FLAGS` intersects the row's FLAT
flags — which are the union of both cells' codes. Adding the existing `unit_mismatch` (raised by
checks.py:984,990 on candidate disagreement, severity warn) would instantly veto Buch's two
admitted rows (they carry it; test_bestguess.py:207-221 pins them admitted with it QUOTED) —
the exact opposite of always-guess. Reader disagreement about a unit is not evidence the two
numbers actually divided were in different units; the resolved pair is.

**The refusal condition, in `_finish` (resolve.py:1043), after the C9 block (:1138-1152):**

```python
# units of the two numbers actually divided — the RESOLVED arms, not any candidate
if name in GROUP_ROUTES and not same_unit(values.group_a.unit if values.group_a else "",
                                          values.group_b.unit if values.group_b else ""):
    record.confidence = "needs_human"
    _add_row_refusal(record, RESOLVED_UNIT_MISMATCH)
    steps.append(f"refused: the two groups' resolved values are in different units "
                 f"({unit_key(values.group_a.unit)} vs {unit_key(values.group_b.unit)}); "
                 f"their difference is not a contrast, so no line may take this row")
    record.conversion_steps = steps; record.conversion_chain = "; ".join(steps)
```

- `same_unit` (verify/units.py:51-55) already encodes "an empty unit is never a mismatch", so
  nothing is decided against a reading that did not say. `unit_other_expression` needs no guard:
  it is the case where the vote KEPT the mapped unit on both arms, so the resolved keys agree;
  if the vote kept different units on the two arms, the division is genuinely cross-unit and the
  refusal is correct even when one unit is "another expression".
- `GROUP_ROUTES` only (resolve.py:50): a test statistic / p / reported d consumed no
  unit-bearing pair.
- Constants: `RESOLVED_UNIT_MISMATCH = "resolved_unit_mismatch"` in confidence.py beside
  IMPLAUSIBLE_DISPERSION (:772); `ROW_REFUSAL_CODES = frozenset({IMPLAUSIBLE_DISPERSION,
  RESOLVED_UNIT_MISMATCH})` (:788, docstring "one member" updated); severity declared in
  checks.py's map (~:167): `"resolved_unit_mismatch": "error"`. resolve.py imports the constant.
- Everything downstream is automatic by existing contracts: overrides.py:458,1439,1771 and
  run.py:2247 hold both cells and refuse releases (questions stay open; a human answer that
  fixes a unit/value rebuilds through `_finish` and the refusal simply does not re-fire —
  supersede is structural); bestguess vetoes it as `row_refusal` (VETO_ROW_FLAGS is the same
  object, pinned test_bestguess.py:80-83).
- Strict byte-identity: zero fixture rows fire (verified above). On the Nyamsuren/Cisneros
  caches the implementer must diff strict `pooled.json` pre/post; the expected delta is zero
  (Sainburg d2 was already held — it moves best-guess only, +2.07 → no number). If any strict
  row moves, stop and escalate: that is a pooled cross-unit contrast, a finding, not a tuning.
- `k_bg >= k_strict` holds trivially: a refused row is in `held` and never admitted; bg = strict
  + admitted (invariant A0 untouched).

## Q4 — orientation: keep the veto

**Decision.** `orientation_unresolvable` stays. A tiebreak ballot that exists already signs the
row upstream (`orientation_by_majority`, hib set — the Heuer AE row shows the shape);
`higher_is_better is None` means no hint, no ballot, no checkable claim — the paper did not
provide a direction, which is the user's own "literally cannot". Both orientation-vetoed rows in
the record also lack an es, so relaxing would add zero rows while breaking the one invariant
every tier shares ("never resolves a direction it does not have"). No change, no test change.

## Q5 — what breaks, exactly (and the edits)

**bestguess.py** — `VETOES` → `("row_refusal", "orientation_unresolvable", "one_group_only",
"no_variance")` (:69); delete both branches at :216-229; `RULES` →
`("disputed_reading_guess", "inferred_premise", "low_confidence_value", "precedence_override")`
(:67) — disputed FIRST in `_rule` (:264): what the number IS being contested outranks how it was
built, preserving loudest-first. New branch (before `inferred`):

```python
named = sorted(set(record.flags) & CONTRADICTED)
if SIGN_DISPUTED in record.flags:
    named.append(SIGN_DISPUTED)
if named:
    return ("disputed_reading_guess",
            f"the evidence disputes what this reading is ({', '.join(named)}); the best-guess "
            f"line takes the resolved value as the tool's best answer — {value} — nothing "
            f"re-signed or rescaled, and the question stays open in human review",
            _evidence(record, contradicting_flags=named, disputed=_disputes(record)))
```

Docstring surgery: module docstring :33-39 (the "disputed direction is a different finding"
paragraph → "a disputed direction is shown WITH its dispute; only a direction the row does not
have is refused"), the SIGN_DISPUTED comment :75-81, the "by every tier" clause at :19-21.
`F3_ABSOLUTES` (:625) unchanged — the answer tier still never ENTERS values over a sign dispute.

**Automatic (no edit):** `best_guess_payload` — `added[].rule`, `note` counts, `by_rule`/
`by_rule_totals` are keyed off `row.best_guess_rule` generically; `mark_composites`;
`best_guess_cells` / tables.py:214 column list (new string value, same schema); JT generator
(validation/out/nyamsuren_tsay/jt_deliverable_generator.py:146-149 reads only
in_best_guess/es/se); SPA app.js (renders payload strings generically).

**theme.py + outputs.py** — one addition: `best_guess_caveat(..., disputed: int = 0)` appends,
whenever `disputed > 0` (fired or not): `" N row(s) enter over a standing dispute about what the
number is or which way it points; each names its dispute."` `best_guess_caveat()` with defaults
stays byte-equal to `BEST_GUESS_CAVEAT` (pins test_bestguess_cells.py:894-898). outputs.py:178-190
computes `disputed = sum("disputed_reading_guess" in (r.best_guess_rule or "") for r in added)`
and assembles when `fired or disputed`; the assembled caveat already threads to the forest and
report (:222). This is the MAJOR-2 lesson kept: a disputed-direction row is announced at line
level, not only per-row. `conclusion.py` VETO_PHRASES (:44-51): KEEP the `contradicted_value`
entry with a one-line "pre-change payloads reload through this" comment (the fallback would
misspell nothing, but byte-stable re-renders of old runs are free).

**Pinned tests that change (complete list):**
- tests/test_bestguess.py::test_contradicting_flags_veto_unless_retired (:50) → rewrite:
  `b7523a41b03a:d2` LA admitted, rule `disputed_reading_guess`, both flags in reason, es
  +0.831 unchanged; `3570e4ce2a9c:d2` LA and `d1f2946e7e81:d1` AE now veto `one_group_only`/
  `no_variance` (es=None rows — assert the veto is a no-value veto, not the flag name).
- ::test_a_disputed_sign_never_enters_the_best_guess_line (:272) → becomes
  test_a_disputed_sign_enters_under_disputed_reading_guess: admitted, `sign_mismatch` in
  reason and in `evidence["contradicting_flags"]`, `rows[0].es == row.es` (never re-signed).
- ::test_the_same_row_without_the_sign_dispute_is_still_admitted (:292) → still green, but the
  rule assertion stays `low_confidence_value` (flag retired ⇒ disputed rule does not fire) — verify.
- ::test_a_refused_row_is_vetoed_by_the_resolvers_own_code (:86) — green;
  `sorted(VETO_ROW_FLAGS)[0]` now picks "implausible_dispersion" alphabetically — still a member.
- tests/test_report.py::test_extraction_columns_have_best_guess_before_moderators (:691):
  `b7523a41b03a:d2` cell flips to `in_best_guess: True`, rule `disputed_reading_guess`,
  `best_guess_es == es`.
- ::test_include_needs_human_note_counts_what_it_could_not_include (:714): best_guess k 5→6
  (== include_needs_human's 6 on THIS fixture; update the comment — the sets still differ in
  general via row_refusal/unsigned rows).
- ::test_report_html_sections_in_order (:1085): `"contradicted_value" in html` →
  `"disputed_reading_guess" in html`; `theme.BEST_GUESS_CAVEAT in html` → assert the caveat
  PREFIX plus the disputed clause (LA now has a disputed admission).
- tests/test_bestguess_cells.py::test_the_d1_boundary_blocks_agreed_solo_read_on_a_contradicting_code
  (:231): the tier still refuses to ENTER (that assertion stays the point); the fallback decision
  on the es-None `_held()` row is now veto `no_variance` (both halves, :242 and :247).
- ::test_the_caveat_is_the_constant_unfired_and_assembled_when_fired (:894): add the
  disputed-clause case; the no-arg identity assertion stays.
- tests/test_resolve.py (:893-929): existing pins stay green (subset + identity). ADD:
  test_finish_refuses_a_resolved_pair_in_different_units (deg vs mm on a group route ⇒ flag
  present, confidence needs_human, steps sentence names both unit keys); test_an_empty_unit_never
  _refuses; test_a_statistic_route_never_unit_refuses.
- tests/test_questions.py:2379-2383 and tests/test_integration_ceiling.py:1001-1004 iterate
  ROW_REFUSAL_CODES — the new member is exercised automatically; expected green (same
  _add_row_refusal + needs_human shape as implausible_dispersion), verify in the run.
- tests/test_checks.py: severity map addition needs no test change (map-driven).

Everything else pinned on RULES/VETOES membership (test_bestguess.py:96,180; payload asserts)
is enum-driven and follows the tuples.

## Rename scope (user-facing Canopy → AutoMeta; package/CLI/paths stay `canopy`)

Change (4 sites): server/app.py:271 `FastAPI(title="Canopy")` → `"AutoMeta"`;
report/html.py:491-499 callout — "R `meta` and canopy disagreed… it is canopy's own" → AutoMeta
twice, but `<code>canopy validate</code>` STAYS (it is the CLI command); report/html.py:581
footer "· canopy {version}" → "· AutoMeta {version}" (`manifest.canopy_version` field name
stays); cli.py:312 "the Canopy web UI is not installed" → "the AutoMeta web UI…".

Leave: app.js localStorage keys `canopy.*` (:108-109,140,2284-2285 — functional, renaming logs
users out); all docstrings/comments (app.py:140,1006, protocol_draft.py:6, theme.py:1,
forest_r.py:6, forest_meta.R:3, search/*.py); mapper.py:76 SYSTEM prompt (an agent prompt, not
the web UI — changing it perturbs extraction behavior and prompt cache); every import, path, run
dir, `canopy_version` key, and CLI name. No test pins any of the changed strings (grep-verified).

## Invariants checklist

- Pure function of the record: yes — both changes are flag/field reads; no LLM, no IO.
- Primary/strict byte-identical: yes on fixtures (verified); cache diff required at implement
  time for the unit refusal (expected zero; a nonzero diff is an escalation, §Q3).
- `best_guess_rules: []` byte-identical off-vs-on: yes (knob gates answer tier only; both sides
  of the pinned comparisons move together).
- Questions open / human answers supersede: structural, untouched — admission reads the record;
  a retired flag stops the rule firing (test_bestguess.py:60 pattern), an answered unit stops
  the refusal re-firing on rebuild.
- `k_bg >= k_strict`: untouched (A0; refusal rows are held, never strict).
- New names flow: `disputed_reading_guess` → payload/note/cells/composites automatically, caveat
  via the one new parameter; `resolved_unit_mismatch` → review layer via ROW_REFUSAL_CODES
  membership, bestguess via VETO_ROW_FLAGS identity, prose via VETO_PHRASES fallback
  ("resolved unit mismatch") — add a phrase entry only if the reviewer wants nicer words.
- Nothing tuned to this corpus: both rules are flag-class-driven; the fixture scan is a
  regression check, not a condition.
