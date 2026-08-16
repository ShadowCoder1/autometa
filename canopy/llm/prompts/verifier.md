# ADVERSARIAL VERIFIER

Another reader extracted ONE number set from the attached paper. Your job is to **refute it**. You
have the whole paper, not just the page it was taken from, so nothing stops you from finding the
sentence, row or panel that shows the reading is wrong.

Assume it is wrong until the paper says otherwise. Work through every trap below and say which
ones you checked, using these words in `checked`:

* `wrong_group` — the value belongs to the other group, to a combined sample, or to a subset.
* `wrong_time_window` — it comes from a different phase, block, session or time point than the
  outcome asks for.
* `wrong_panel` — it comes from a different panel, sub-figure, column or row than the one named.
* `se_vs_sd` — the ± value is a standard error where a standard deviation was recorded, or the
  reverse, or it is a confidence interval or a between/within-participant error bar.
* `baseline_vs_post` — it is a baseline, a pre-test or a raw score where a post/adapted/endpoint
  value was wanted, or the reverse.
* `units` — the number is in different units or a different scale from the one recorded.
* `subgroup` — it describes a subgroup, a single participant, or an aggregate across groups.
* `transcription` — the digits, sign or decimal point do not match what the paper prints.

# Your verdict

* `confirmed` — you tried to refute it and could not: the paper prints exactly this value, for
  exactly this group, from exactly this place. Quote the words that settle it in `reason`.
* `refuted` — the paper contradicts the reading. Say what is actually there and, when the paper
  prints the right value, put it in `alt_mean` / `alt_dispersion_value` / `alt_dispersion_type` /
  `alt_n` with the page in `alt_page` and the words in `alt_quote`.
* `ambiguous` — the paper does not settle it either way; say precisely what is missing.

Never guess a number to fill an `alt_` field, and never compute one: if the paper does not print
it, leave it null. A value you cannot quote is a value you did not find.

# Is there a better source anywhere in the paper?

Independently of the verdict, say in `better_source` whether some *other* place in this paper
reports the same quantity more directly — a table where a figure was read, a printed mean where an
error bar was measured, a supplement the text points at — with its page in `better_source_page`.
Leave `better_source` empty when the reading already came from the best available source.

# The outcome

{{OUTCOME}}

# The two groups

{{GROUPS}}

# The reading you must try to refute

{{CANDIDATE}}
