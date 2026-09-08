# Brief: "always guess" best-guess tier + unit_mismatch row refusal + AutoMeta rename

Date: 2026-09-07. Owner: coordinator. Status: DESIGN → adversarial review → implement.

## User requirement (verbatim intent)

The best-guess line should have answered EVERY open question with the tool's best answer —
still shown in human review, human answers still superseding — and the only acceptable reason
not to guess is that the paper literally does not provide the data. All the manual prompting in
the Nyamsuren validation ("explain why it couldn't do these questions" → reviewer entering
values) should not have been needed for rows where the tool had a number. Do not break
anything. Also: rename the web UI from Canopy to AutoMeta.

## Current architecture (facts, verified in code)

`canopy/pipeline/bestguess.py` — pure function of the record, no LLM calls at repool time.
Row admission is already "admit unless veto" (`low_confidence_value` is the catch-all rule).
The vetoes (`_veto`, first match wins):
1. `row_refusal` — flags ∩ ROW_REFUSAL_CODES (today: implausible_dispersion only).
2. `contradicted_value` — flags ∩ CONTRADICTED (series_identity_conflict,
   locator_reads_conflict, categorical_x_unsupported…): "a best guess about the wrong quantity
   is not a guess about this contrast".
3. `contradicted_value` via `sign_mismatch` — stated direction contradicts extracted means.
4. `orientation_unresolvable` — higher_is_better is None.
5. `one_group_only` / `no_variance` — no computable es (the honest "cannot").

`ROW_REFUSAL_CODES` lives in `canopy/verify/confidence.py:788` (frozenset, one member);
`resolve._add_row_refusal` (resolve.py:1155) is the only writer and self-checks membership;
`bestguess.VETO_ROW_FLAGS = ROW_REFUSAL_CODES`; overrides.py re-exports as ROW_REFUSALS.
`unit_mismatch` is flagged in `canopy/verify/checks.py:984,990` (severity "warn") when arms'
units disagree — but the resolver still converts, which produced Sainburg 2002 AE d2's absurd
+2.07 SMD (dominant in mm vs non-dominant as a ratio) that only died in manual review.

`BEST_GUESS_RULE_NAMES` (models.py:125) = ("adjudicated_value", "route_precedence",
"agreed_solo_read") is the ANSWER-rule namespace (cells); ROW rules namespace is
`bestguess.RULES` = ("inferred_premise", "low_confidence_value", "precedence_override").
NOTE the assert at bestguess.py:618: `ANSWER_RULES == BEST_GUESS_RULE_NAMES`.
`StatsSettings.best_guess_rules` (default: all names) gates ANSWER rules; `[]` must stay
byte-identical to the strict analysis (pinned tests).

## Measured impact on the Nyamsuren run (post-review record, today)

Held rows off the line: LA 10, AE 0. Of the 10: 8 have es=NONE (one_group_only ×4,
contradicted+no-es ×2, orientation+no-es ×2 — Striemer LA also lacks dispersion) — these are
"literally cannot" under the user's rule. Exactly 2 rows carry a live es blocked only by a
contradiction flag: 71d8cd6a7200:d1 LA (es +0.386, series_identity_conflict) and
e6d86c2fefc2:d2 LA (es +0.098, locator_reads_conflict). On the PRE-review virgin record the
same veto classes blocked many more (Bagesteiro sign_mismatch ×4 AE cells with live values,
Scarpina series conflicts, Redding two-places conflict) — rows the manual review later admitted
at (mostly) the tool's own values. Narrowing vetoes is what would have pre-empted most of the
manual prompting.

## Proposed change (for the panel to interrogate)

A. **Narrow the vetoes to the "literally cannot" set.** Keep: `row_refusal`,
   `one_group_only`/`no_variance`, `orientation_unresolvable` (no sign = no row).
   Convert to ADMIT-with-loud-rule: `contradicted_value` (both flag-driven and sign_mismatch) —
   new row rule name, e.g. `disputed_reading_guess`: "the evidence disputes what this number is
   (…flags…); the best-guess line takes the resolved value as the tool's best answer; the
   question is open in human review". Evidence dict carries the contradicting flags verbatim.
B. **unit_mismatch becomes a row refusal.** Add to ROW_REFUSAL_CODES; resolver calls
   `_add_row_refusal(record, "unit_mismatch")` when the RESOLVED pair's units disagree
   (arm A unit_key != arm B unit_key, both non-empty, and not the unit_other_expression case).
   Note: checks.py flags candidate-level unit disagreement; the refusal must be about the
   RESOLVED pair (the two numbers actually divided), computed in resolve._finish where both
   inputs are known. A refused row is a "cannot" — Sainburg d2 would have shown NO number
   instead of +2.07 (correct: the guess about mm-vs-ratio is not a guess about the contrast).
C. **Rename**: user-facing strings in the web UI (server HTML/templates) Canopy → AutoMeta.
   Package/module names, run dirs, CLI command stay `canopy`.

## Open design questions for the panel

1. Should sign_mismatch admit at the resolved means' sign (trust-the-numbers, as Bagesteiro
   vindicated) or stay vetoed? User wants a guess; the Bagesteiro case shows text-claims can be
   the wrong side. But is there a case in the record where the MEANS were the wrong side?
2. Row-rule namespace vs `best_guess_rules` knob: row rules are NOT gated by the protocol knob
   today (only answer rules are). Does the new rule need a knob? ([] byte-identity only touches
   answer rules — verify.) Existing protocols must not silently change primary results (they
   won't — best-guess only), but the BEST-GUESS line of existing runs will grow rows on repool.
   Acceptable? (User: yes, explicitly.)
3. unit_mismatch refusal vs "pick the corroborated same-unit pair" (what the human did for
   Sainburg d2): pair re-selection in resolve is the better guess but a bigger, riskier change.
   Refuse now + follow-up feature, or in scope?
4. Orientation: keep the veto (no sign = no row), or guess via majority/sibling when a ballot
   exists? (Note verdicts already carry majority orientations; hib=None means truly nothing.)
5. Anything that breaks: DECISION B invariants (questions stay open; human answers supersede —
   supersede is structural, unaffected?), k_bg ≥ k_strict, marks in report/theme
   (best_guess_caveat counts by rule — new rule must appear there), JT figure generator reads
   best_guess_rule strings, tests pinned on VETOES/RULES tuples.

## Constraints

- No LLM calls in bestguess (pure function) — stays true. No new spend at repool.
- Never read .env; no CANOPY_LIVE in design/tests; nothing tuned to this corpus.
- Suite must stay green (2498 tests; 3 known ordering-dependent flakes pass chunked).
- Out of scope (recorded as follow-up): run-time LLM "provisional answer" stage where the
  adjudicator drafts a guess per open review question (would cover the 8 no-es rows when the
  paper does contain the data in a form the extractors missed).
