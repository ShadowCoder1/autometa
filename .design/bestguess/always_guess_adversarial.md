# Adversarial review: always_guess_design.md

Date: 2026-09-07. Reviewer: adversarial pass, every claim checked against the working tree.

## Verdict: GO-WITH-FIXES

One blocker (a real silent-wrong-number path the deleted veto was load-bearing for), three
majors (two false claims in the design's own "verified ground"/edit list, one missing edit),
two minors. All have small, well-defined fixes; no architectural rework. The core decision —
admit disputed held rows at their own value under a loud rule, refuse resolved cross-unit
pairs — survives review. The fixture scans in the design are otherwise accurate (verified:
no differing unit pairs in `tests/fixtures/runs/nine`; `b7523a41b03a:d2` LA is the only
contradiction-vetoed fixture row with a live es; the live run has exactly two live-es
contradiction-blocked rows, +0.386 and +0.098).

## 1. BLOCKER — deleting the contradicted branch guts the answer tier's built-row backstop

**Path.** `bestguess.py:1596` (`vetoed = _veto(built)` inside `_cell_decide`) is the ONLY
guard against a forcing CONTRADICTED flag instance that survives instance-scoped masking onto
a tier-BUILT row. `_mask_pair` (bestguess.py:1481-1488, comment :1510 "the leak fixture's
veto survives") deliberately keeps an instance attached to a cell that neither crossed it nor
was answered — e.g. a RELEASED, value-complete sibling cell carrying its own uncrossed
`series_identity_conflict`. The crossing calculus never sees that instance (`_answer_cell`
returns "not held" for a released cell before any F3/forcing check, bestguess.py:1346-1350;
`consumable()` passes it as healthy, :1546-1553). Today `_veto(built)`'s `contradicted_value`
branch catches the code on the built row and the tier refuses. Pinned:
`tests/test_bestguess_cells.py:445` (`test_t_b17_the_leak_fixture_...`).

**Failure scenario under the design.** Delete bestguess.py:216-229 and `_veto(built)` returns
None. `_cell_decide` ADMITS the built row under the answer-tier rule name
(`"adjudicated_value"`), reason "answers this row's open questions by rule — …". The
uncrossed `series_identity_conflict` appears in no rule, no reason, no `stepped_past` (it was
never crossed), no caveat count (the disputed count keys on `"disputed_reading_guess"`, a
DECISION-A rule this row never touches). The payload `added[]` entry and the queue's
`best_guess_rule` column show a clean answer-tier admission. A value the evidence says may be
the wrong QUANTITY enters the line with values the tier itself typed in, and the "loud label"
defense never fires anywhere. This is precisely the class the old veto existed to stop, and
T-B17 breaks — a test the design's "complete list" does not name.

**Minimal fix.** In `_cell_decide`, immediately after `vetoed = _veto(built)` handling
(bestguess.py:1596-1601), add a tier-only check:
`leaked = sorted(set(built.flags) & CONTRADICTED)`; if non-empty,
`_record_cell_guesses(ctx, record, guessed, f"the built row carries an uncrossed
contradiction ({', '.join(leaked)})")` and `return None`. DECISION A's new
`disputed_reading_guess` then correctly applies only to a HELD row's own resolved value,
never to values the tier entered. T-B17 survives with only its `row_blocked_by` wording
updated. (Sign is already safe: the pre-check at :1534 blocks the tier on any row-level
`sign_mismatch`, and the resolver never raises it.)

## 2. MAJOR — `resolved_unit_mismatch` needs a `_FLAG_TO_KIND` entry; the design says "no test change"

**Path.** `tests/test_questions.py:2370-2388`
(`test_no_row_refusal_can_ever_reach_the_confirmed_value_terminus`) iterates `ROW_REFUSALS`
and asserts each code is in `canopy/review/questions.py`'s `_FLAG_TO_KIND` (:120) AND that
its kind is not in `_ASKED_ONCE` (:448). `resolved_unit_mismatch` has no entry → the test
fails, and the design's claim "the new member is exercised automatically; expected green" is
false. The functional gap is real, not cosmetic: an unmapped code falls through to
`confirm_value`, which IS in `_ASKED_ONCE`, so a confirmed value could take the last card off
a cell while the row stays refused — the exact leak this interlock guards.

**Fix.** Add `("resolved_unit_mismatch", <kind>)` to `_FLAG_TO_KIND` with a kind outside
`_ASKED_ONCE`, plus the card wording for that kind (the kind-specific branches at
questions.py:907/914 and the option plumbing at :2469/:2575 show the shape
`dispersion_doubt` needed). Given finding 6 below, the card should explicitly ask for the
arm's UNIT, since the unit slot is what clears the refusal.

## 3. MAJOR — the assembled caveat does NOT reach the HTML report; the planned :1085 test edit will fail

**Path.** `canopy/report/html.py:417`: `_best_guess_section` prints
`theme.BEST_GUESS_CAVEAT` — the raw constant — unconditionally. The assembled caveat threads
only to `render_forest` (`outputs.py:222`, drawn into the PNG). The design's sentence "the
assembled caveat already threads to the forest and report (:222)" is wrong about the report,
and its rewrite of `tests/test_report.py:1085` ("assert the caveat PREFIX plus the disputed
clause") asserts a string that will never be in `report.html`.

**Failure scenario.** LA gains a disputed admission; the report's best-guess section — the
artifact a reviewer actually reads — shows the unmodified constant with no disputed clause.
The MAJOR-2 "answered by loudness" defense is silent in the one place it matters most.

**Fix.** Thread the assembled caveat into the html section (e.g. store it in the results
payload beside `best_guess` and give `_best_guess_section` a caveat argument defaulting to
the constant), then the planned :1085 assertion is correct. Alternatively drop that half of
the test edit — but then say so in the design instead of claiming the threading exists.

## 4. MAJOR — wrong predicted veto for `d1f2946e7e81:d1` AE (fixture fact error)

**Path.** `tests/fixtures/runs/nine/papers/d1f2946e7e81/resolve.json`: `d1f2946e7e81:d1`
aftereffect has `es=None` AND `higher_is_better=None` (so does its late_adaptation row). The
post-deletion veto order is row_refusal → `orientation_unresolvable` (bestguess.py:231) →
one_group_only/no_variance. That row therefore vetoes **orientation_unresolvable**, not the
"one_group_only/no_variance … no-value veto" §Q5 prescribes for the rewrite of
`tests/test_bestguess.py:50`. Only `3570e4ce2a9c:d2` LA (hib=False, es=None) lands a
no-value veto as predicted. Loud failure, but an implementer following the design's assert
text will chase a phantom; the design's "Every anchor … was read in the working tree" is
overstated here.

**Fix.** In the rewritten test: `d1f2946e7e81:d1` AE → assert veto `orientation_unresolvable`;
`3570e4ce2a9c:d2` LA → a no-value veto. Note also `3570e4ce2a9c:d2` AE (the real
`sign_mismatch` row) has hib=None in the fixture, so it stays vetoed by orientation — which
is why the sign tests' constructed row (hib=False) is the right vehicle, as the design says.

## 5. MINOR — the disputed caveat count misses a two-rule composite

**Path.** `outputs.py` disputed count is specified over post-aggregation `added` rows via
substring on `best_guess_rule`; `mark_composites` (bestguess.py:394) names a composite with
members under two different rules `"composite"`, so a disputed member escapes the count and
the caveat clause stays silent while a disputed value sits in the line.
**Fix.** Count decisions instead: `sum(1 for d in decisions if d.admitted and d.rule ==
"disputed_reading_guess")` — decisions are pre-aggregation and one per held row.

## 6. MINOR — unit-alias false positives now lock a row where they used to warn

**Path.** `verify/units.py:30-50`: an unknown token keeps its own spelling — "px" vs
"pixels", "trial" vs "trials" produce differing non-empty keys — and the new refusal turns
what was a warn (`unit_mismatch`) into a locked row with both cells `needs_human`. Direction
is conservative (a held row, never a wrong number), and the reviewer CAN clear it: verified
that `_apply_value` sets `verdict.unit` when the answer carries one (overrides.py:2122-2123),
the release gate reads the PROBE rebuilt through `_finish` (overrides.py:2147-2149 →
`_rebuild_row` :1918 → `resolve_effect_with_fallback`; the mark-reviewed path likewise,
:1164-1185), and GroupValues carries `unit` from the verdict (resolve.py:92, :125). The
actual Sainburg d2 AE answers (live run `overrides.jsonl` seq 369/370) supplied
`"unit": "unitless ratio (major/minor axis, change"` on BOTH arms → keys equal → the refusal
would not re-fire → the reviewer's historical fix WOULD have lifted it. **But** an answer
with same-unit values and an empty unit field leaves the stale verdict units standing
(`if override.get("unit")`) and the refusal re-fires forever with nothing telling the
reviewer the unit slot is the lever.
**Fix.** Make the finding-2 question card ask for the unit; optionally add plural-stripping
to `unit_key` for unknown tokens.

## Verified clean (no action)

- **Knob gating**: `best_guess_rules: []` gates only the answer tier (`_cell_context`,
  bestguess.py:714-717 returns None → `_decide` runs row rules regardless). Every `[]` pin
  compares off-vs-on within one build (`test_bestguess_cells.py:146` catalogue test, the
  byte-identical decision test, and the off/on repool diff at :816-847 repools BOTH sides
  with the same build) — a row-rule change moves both sides together. Design correct.
- **Cardinality**: no consumer assumes one `ROW_REFUSAL_CODES` member. All uses are set
  intersections (overrides.py:458, :1439, :1771; run.py:2247) or iteration
  (test_integration_ceiling.py:1001-1004 — shape-generic, passes with the new code;
  test_questions.py:2379 — see finding 2). `sorted(VETO_ROW_FLAGS)[0]` in
  test_bestguess.py:92 still picks `implausible_dispersion` ("i" < "r").
  `_add_row_refusal`'s membership self-check + test_resolve.py:893-930 subset/identity pins
  stay green once the constant is added.
- **Snippet viability**: `_finish` is at resolve.py:1043 with `name`, `values`, `record`,
  `steps` all in scope after the C9 block; `GROUP_ROUTES` at :50; `GroupValues.unit` exists
  (:92). `same_unit` empty-never-mismatches confirmed (units.py:53-55). The design's
  "zero fixture rows fire" claim re-verified by scanning all six papers' verdicts: true.
- **Downstream rule-name consumers**: JT generator reads only
  `in_best_guess`/`best_guess_es`/`best_guess_se` (jt_deliverable_generator.py:141-149) —
  tolerant. `app.js` renders `best_guess_rule` as an opaque string (:1771, :1986) — no
  switch on names. `conclusion.py` VETO_PHRASES spells out unknown names (:42-44 comment);
  keeping the `contradicted_value` entry for old-payload reloads is right; "points the other
  way" (:408-412) still fires. `best_guess_payload`/`by_rule`/`mark_composites`/
  extraction columns/queue writer are all name-generic.
- **Tier interlocks**: the new refusal code auto-blocks the tier via the pre-check
  (bestguess.py:1534, `VETO_ROW_FLAGS` identity). `decision_b_fired` untouched by the new
  row rule (not in ANSWER_RULES). `_rename_agreed_solo` requires rule ==
  `low_confidence_value`, so a disputed row is never renamed.
- **k_bg ≥ k_strict / A0**: untouched — the new rule only admits, the refusal only holds
  (a refused row was already `needs_human` before `_finish` returns, so it is in `held`).
  The design's own escalation clause covers the one genuine k_strict risk (a pooled
  cross-unit contrast surfacing on the cache diff).
- **Live-run pins**: the A-M measure (`test_bestguess_cells.py:526+`) counts only
  tier-entered rows; with the finding-1 fix in place, tier behavior is byte-identical and
  the pins hold. The live run's pooled.json shows no cell guess blocked by the contradicted
  veto, so the fires set cannot shift on this record either way.
- **Live-run measured impact**: post-review LA has four `contradicted_value` vetoes, of
  which exactly two carry a live es (71d8cd6a7200:d1 +0.386, e6d86c2fefc2:d2 +0.098; the
  other two es=None, hib set → no-value vetoes under the design). Brief's numbers accurate.
