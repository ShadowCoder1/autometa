# Code review: always-guess implementation (uncommitted diff vs always_guess_design.md + adversarial)

Date: 2026-09-07. Adversarial pass; every claim checked in the working tree.

## Verdict: SHIP-WITH-FIXES

The design and all four adversarial findings that demanded code (blocker leak check, _FLAG_TO_KIND
entry, caveat threading to html, decisions-based disputed count) are correctly implemented. One
MAJOR remains: the unit_doubt card's clearing answer cannot be typed in the web UI. Three minors.

## Verified correct (the hunted items)

- **_finish refusal** (resolve.py:1159-1167): placed last, after C9/inferred/variance/dispersion
  blocks; `_finish` has NO early returns, so no path skips it. `steps` is the same list object as
  `record.conversion_steps` (assigned :1083), so append+reassign is idempotent — no double-append.
  None arms are safe: `values.group_a.unit if values.group_a else ""` makes `same_unit` True
  (units.py:53-55), so the unguarded `values.group_a.unit` in the f-string is unreachable with a
  None arm. `needs_human` + `_add_row_refusal` matches the IMPLAUSIBLE_DISPERSION pattern; the
  membership self-check (:1180) passes (confidence.py:797 adds the code). Probe/release path:
  GroupValues are built only via `from_verdict` (resolve.py:126, :265) in both the run and the
  overrides rebuild, so no run-vs-rebuild asymmetry can mis-fire a refusal on release; a corrected
  unit rebuilds through `_finish` and the refusal re-derives away. Residual mis-fire vectors are
  finding 4 below.
- **_cell_decide leak check** (bestguess.py:1607-1616): placed after `_veto(built)` handling,
  before `_has_value` — order-insensitive, both refuse. No false positives from crossings or
  retirements: `_mask_pair` (:1481-1527) drops an instance iff every attached cell crossed,
  answered, or retired it, and `built.flags` derive from the masked copies — so a crossed or
  retired CONTRADICTED flag never reaches the intersection. Instances with empty candidate_ids
  survive, exactly as the deleted `_veto(built)` branch caught them — tier behavior is equivalent.
  `_record_cell_guesses` is the same sink the adjacent veto path uses, same `row_blocked_by`
  shape — right call. All three T-B17 tests still pass unedited: :443 asserts "vetoed" (the
  retained row-refusal path), :464 asserts the flag name, which the new message interpolates;
  the fallback `_decide` on `_held()` (route="not_convertible", no es) still vetoes, so
  `not any(d.admitted)` holds.
- **disputed_reading_guess** (bestguess.py:264-274): `value` defined at :259 before use; vetoes
  ran first so es/var/hib exist. NO duplication: `sign_mismatch` is not in CONTRADICTING_FLAGS
  (confidence.py:235-260, 7 members — verified), so the append at :266 never doubles a name.
  Evidence shape (`contradicting_flags` + `disputed`) mirrors the deleted veto's key and the
  sibling rules'. Fixture facts check out: 3570e4ce2a9c:d2 AE clean copy has no CONTRADICTED
  member (series_marker_mismatch is not one), so the unchanged low_confidence_value test stays
  green; d1f2946e7e81:d1 AE is es=None/hib=None → orientation_unresolvable, as the edited test
  asserts (adversarial finding 4 landed).
- **Caveat threading**: outputs.py:181 counts DECISIONS (finding 5's fix, composites cannot hide
  a member); `bg_payload["caveat"]` set only inside `if fired or disputed` (:183-198) and before
  `payload["best_guess"] = bg_payload` / dump_json (:251, :259) — quiet payloads stay
  byte-identical (key absent). html.py:415 falls back to the constant for old payloads.
  theme.best_guess_caveat() with defaults returns the constant byte-for-byte; the disputed clause
  appends to the constant prefix unfired (pinned in test_bestguess_cells.py:905-909).
- **unit_doubt plumbing**: QUESTION_KINDS, _FLAG_TO_KIND (ordered before the deliberate-last
  orientation entries), _answer_kind→"value", options, prompt all registered; NOT in _ASKED_ONCE
  (questions.py:454) so the interlock test (test_questions.py:2370-2388) is satisfied; row-refusal
  codes reach cards via `on_the_row` (:797-799) generically, same as implausible_dispersion.
  Server and app.js render kinds generically (zero kind switches found) — no enumeration missed.
- **Rename**: every hunk is a display string or comment. `canopy validate`/`canopy run`/`canopy
  serve`, localStorage `canopy.*` keys, `manifest.canopy_version`, imports, paths, CLI name all
  untouched; no test pins any renamed string (only docstrings mention Canopy). One mandated site
  missed (finding 2). protocol_draft.py:188 was renamed beyond the design's 4-site list — benign
  (user-facing generated header; the CLI string on the next line kept).
- **Test edits**: nothing asserts less than before except the noted disjunction (finding 5).
  The sign test now additionally pins es AND var unchanged; the report tests pin the new rule
  name, the disputed clause in html, and k 5→6 exactly as the (corrected) design predicted.

## Findings

1. **MAJOR — the unit_doubt card's clearing answer cannot be given from the web UI.**
   questions.py:1428-1440 instructs "Re-type the value with its unit — the unit is what clears
   the refusal", and _single_override:2700-2702 honors `answer["unit"]` — but the free-value form
   in app.js (~:1173-1184) has only mean / dispersion_value / dispersion_type / n / note inputs
   and never submits a `unit` key (sole "unit" reference in app.js is a placeholder hint). So a
   UI reviewer facing a false-positive refusal (unit alias, stamped unit — finding 4) has no path
   but "different_quantities"/Exclude, which throws the row away, or a hand-built API override.
   The operative half of adversarial findings 2+6's fix is missing. Not a blocker only because
   zero fixture/live rows fire today and the direction is conservative.
   Fix: add a text input `name="unit"` to the free-value block (placeholder from `q.unit`), sent
   like the other fields; the server allowlist (app.py:1255) already accepts it.

2. **MINOR — mandated rename site missed**: html.py:500-503 `_renderer_callout` still prints
   "R `meta` and canopy disagreed … it is canopy's own". Design §Rename names this site
   explicitly (change both, keep `<code>canopy validate</code>`). Two word swaps.

3. **MINOR — VETO_PHRASES["resolved_unit_mismatch"] is dead code with an overstating comment**
   (conclusion.py:48-51). The only consumer (`_veto_reasons`, :157-164) keys on
   `not_added[].veto`, which is always one of the four veto NAMES ("row_refusal", …) — never a
   refusal code — so "wherever a refusal is spelled by its code" happens nowhere. Harmless entry;
   either delete it or reword the comment so a later reader does not hunt for a consumer that
   does not exist. (The kept "contradicted_value" entry is correctly justified: old payloads.)

4. **MINOR — a value answer on ANY card silently stamps the card-derived unit onto the verdict,
   and the refusal now weaponizes that.** _single_override falls back to `question.get("unit")`
   (questions.py:2702) — derived from CANDIDATES (`_unit`, :4441: first candidate with a unit) —
   and _apply_value writes any truthy unit to the verdict (overrides.py:2122-2123). Where readers
   disagreed on units, the stamped unit can differ from the voted verdict unit, so answering an
   unrelated value question can newly create (or silently clear) a cross-unit pair on rebuild —
   a row locked by a legitimate release. Pre-change this was warn territory; now it refuses, and
   with finding 1 the lock is UI-unclearable. Also `_unit_prompt`'s sentence "a value alone
   leaves the stale unit standing" misdescribes this: a value alone stamps the CARD's unit, not
   nothing. Fix: on unit_doubt cards prefer typed unit → card unit as now, but on OTHER kinds
   prefer typed unit → "" (leave the verdict unit alone instead of stamping the candidate guess);
   then the prompt sentence becomes true as written.

5. **MINOR — one assertion weakened without need**: test_bestguess.py:739
   `veto in ("one_group_only", "no_variance")` — the fixture row (es=None, hib=False, inputs
   empty, no GROUP_STATISTICS_MISSING flag) deterministically vetoes `no_variance`; the design
   only asked not to assert the flag NAME. Pin the single name; the disjunction can hide a
   future drift between the two no-value vetoes.

## Notes for the run in flight

- test_report_html_sections_in_order now needs an outcome that prints the raw constant
  (`theme.BEST_GUESS_CAVEAT in html`). AE adds no rows in runs/nine (all held AE rows have
  es=None — verified), so its section is absent; the assertion holds only because LA's assembled
  caveat has the constant as PREFIX, which is true only while LA is UNFIRED on this fixture. If
  the suite shows LA fired, that assertion — not the feature — is what broke.
- Strict-line cache diff (design Q3's escalation clause) is still owed before commit: zero
  expected on Nyamsuren/Cisneros caches; any strict movement is a finding, not a tuning.
