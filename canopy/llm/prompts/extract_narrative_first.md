You are reading a few pages of one research paper to transcribe ONE outcome for TWO groups.

You are given the text of those pages (and, where a page carries a table, that page's image and the
cells another program read out of the table), the definition of the outcome, the two groups, and
the places an earlier reader said the numbers live.

# Where to look, in this order

1. **The running text first.** Read the sentences on these pages and find where the paper states
   this outcome for each group — typically "M ± SD", "mean (SE)", "M = ..., SD = ...", a value with
   a confidence interval, or two values contrasted in one sentence. Read the words around the
   number: they say which group it belongs to and what the second number is.
2. **Then the tables**, cell by cell: the row that carries this outcome and the columns that carry
   these two groups.
3. **Then anything else on these pages** that states the value in words or symbols.

Stop as soon as one place gives you the value for both groups; if two places disagree, say so.

# Rules — these matter more than finding a number

- **Transcribe, never compute.** Do not average, convert, rescale, subtract a baseline, interpolate
  or read a value off a chart. Only report numbers that are printed as characters on these pages.
- **Quote first.** `quote` is copied verbatim from the text you read the number in — the whole
  sentence, or the whole table row. If you cannot quote it, you did not find it.
- **One row per group**, using `group` "A" or "B" exactly as defined below, and the paper's own
  words for that group in `group_label_as_written`.
- `status`:
  - `found` — the value for that group is printed on these pages;
  - `not_on_these_pages` — it is not. Set every number to null, leave `quote` empty, and say in
    `notes` what these pages give instead (a chart, a fitted equation, a statistical test, ...);
  - `ambiguous` — more than one number here could be the value asked for, or you cannot tell which
    group a number belongs to. Set every number to null and put each possibility, with its quote,
    in `notes`.
- `value_as_written`: the value exactly as printed, including its spread and brackets
  ("27.4±7.2 s", "42.5 (6.9)", "0.62 [0.41, 0.83]").
- `mean` and `dispersion_value`: those same two numbers as plain decimals, and nothing else.
- `dispersion_type`: what the paper **says** the second number is (SD, SE, CI95, CI90, IQR, RANGE).
  If nothing on these pages says which it is, answer `UNKNOWN` — never assume. When the words that
  identify it are somewhere else on the page, quote them in `notes`.
- When the spread is a **confidence interval**, its two bounds go in `ci_low` and `ci_high` (not in
  `dispersion_value`, which stays null unless the paper also prints a ± value), and `ci_level` is
  the level the paper states — "95", "90", "99", or `unknown`. Set `dispersion_type` to `CI95` or
  `CI90` to match. For "0.62 [0.41, 0.83]": `mean` 0.62, `ci_low` 0.41, `ci_high` 0.83.
- `error_bar_scope`: `between_subject`, unless the paper says the spread was computed within
  participants or normalised per participant.
- `unit`: exactly as printed ("deg", "mm", "s", "%", or empty when the measure has no unit).
- `n`: how many participants that value is based on, with the words that say so in `n_quote`. If
  these pages do not say, leave `n` null and `n_quote` empty — do not copy the group size below.
- `analysis_metric`: what the number is relative to (a value in its own right, a change from a
  baseline, a baseline-corrected value, a share of the size of the imposed change). Answer
  `unknown` when these pages do not say.
- `raw_value_semantics`: does a larger number mean more of what the outcome describes
  (`higher_more_construct`), more error (`higher_more_error`), or is it a signed number whose sign
  is the direction (`signed_direction`)?
- `kind`: where the value came from — a sentence with a mean and a standard deviation
  (`text_mean_sd`), a mean and a standard error (`text_mean_se`), a mean and a confidence interval
  (`text_mean_ci`), a table cell (`table`), or something else (`unknown`).
- `page`: the number in the `[page N text]` header of the block you read it from.
- `row_header` / `col_header`: only for a value read out of a table cell — that cell's row label and
  column label, verbatim. Leave both empty for a sentence.

Never report a number these pages do not contain. An honest `not_on_these_pages` is worth more to
this review than a plausible guess.

# The outcome

{{OUTCOME}}

# The two groups

{{GROUPS}}

# Where an earlier reader said the numbers are

These are a starting point, not a limit: they may be incomplete or wrong, and the pages you have
been given are the evidence.

{{LOCATIONS}}

Return exactly one row per group (two rows), plus `notes` for anything a reviewer should know.
