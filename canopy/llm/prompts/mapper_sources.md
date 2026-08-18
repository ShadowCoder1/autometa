# STUDY MAPPER — SOURCE MAP

Second pass over the same paper (attached in full). The comparisons it supports are already
established and listed below. Your job now is to record, for each of those datasets and each
protocol outcome, **every place in the paper where a usable number lives**.

Later agents read the numbers, and they can only look where you point them: a location you omit is
lost to the review, and a wrong page number sends them to the wrong page. Exhaustiveness and exact
page numbers matter more than brevity. You do not extract values and you do not compute anything —
never read a value off a figure, never estimate, average or convert.

## PROTOCOL

{{PROTOCOL}}

## DATASETS ESTABLISHED IN THE FIRST PASS

Attach every outcome to one of these by its `dataset_index`.

{{DATASETS}}

## FIGURES AND TABLES DETECTED IN THIS PDF

Use these ids verbatim in `figure_id` / `table_id` (leave them empty for text sources).

{{ROSTER}}

## RULES

### Page numbers
`page` is the 1-based position of the sheet inside the attached PDF: the first sheet is page 1,
whatever number is printed on it. Journals often print a different page number (an article may
start at printed page 1057) — ignore the printed numbers, count sheets. Check every page number you
report; a source on the wrong page is worse than no source.

### Outcomes
Cover every protocol outcome each dataset can support, using the protocol's outcome `key`
verbatim. If the paper reports nothing for an outcome in a dataset, leave that pair out.

* `measure_name` / `units`: the quantity the paper actually reports for this outcome, in its own
  words and units.
* `operationalization`: exactly which trials, blocks, epochs or timepoints make up the number, as
  the paper defines them.
* `higher_is_better`: whether a larger value means more of the construct the protocol asks about,
  with a verbatim quote in `higher_is_better_evidence`. Decide it from the paper's own wording, not
  from the field's conventions. Answer `unknown` if the paper's wording does not settle it.

### Sources — the important part
`sources` lists EVERY location in the paper where a usable number for that dataset and outcome
lives, even when several locations report the same thing. Include:

* sentences in the text that print group values (mean ± SD, ± SE, ± CI, medians, IQRs);
* table cells (give `table_id` and the row/column in `locator`);
* figure panels that plot the value (give `figure_id`, and name the panel and the exact
  point/bar/box in `locator`, e.g. "Fig 2B, last block, filled squares");
* test statistics (t, F, χ², exact p) that compare the two groups on that outcome;
* effect sizes the paper itself reports;
* statements that the data are available elsewhere (repository, supplement, "available from the
  authors") — kind `author_data`.

Only list a location that actually prints numbers, or a plotted point/bar/box whose value could be
read off the axis. A sentence that merely describes a result in words ("the after-effect was similar
in both groups") is not a source — it belongs in `operationalization` or in the outcome's notes.

For every source give `page` (see above), `locator`, and a short verbatim `quote` (≤ 300
characters) that a human can search for. Put values in `values_in_text` only when they are printed
as text or in a table — never a value you read off a plot. Use the closest `kind`. `unknown` is for a location that does print
numbers of a sort no other kind describes (fitted-model parameters, for example) — a human will
route it, so quote it well.

`analysis_metric` says what the numbers at that location are: raw endpoint values, changes from
baseline, baseline-corrected values, values expressed as a percentage of the manipulation, or
unknown. Locations can differ within one outcome — record what each one shows. Answer it for every
location you list: a location that does not say which of these it reports is treated as a second,
unresolved measure and stops the outcome being read until a person settles it.

`sample` says WHOSE numbers are at that location — which people the value describes:

* `both_groups` — the two groups of this dataset, reported separately, so a number for each;
* `one_group` — only one of the two (one arm's own analysis, one group's post-hoc test);
* `pooled` — the two groups combined, or a wider or narrower sample: everyone in the study taken
  together, an analysis that collapses the grouping factor, a subgroup of one group;
* `other` — a different set of people again (another experiment's participants, a control sample
  this dataset does not compare);
* `unknown` — the sentence, caption or table does not say whose numbers these are.

Put the words that told you in `sample_note`. Answer from the paper's own sentence, never from
what would be convenient: a location whose sample is not this dataset's two groups is still worth
listing — it is evidence about the paper — but it is NOT read for this contrast's value, because
a number computed over different people is a wrong number here, not an imprecise one.

`role` says what the location IS for this outcome, and it decides whether a number is read there:

* `value` — the outcome's own number for the two groups can be read at this location. A **test
  statistic** is a `value` source only when its contrast is **exactly group A versus group B on
  this outcome, at this measurement window**, and you can say so from the paper's own sentence.
  Record the statistic's `design`, its numerator and denominator degrees of freedom, the model's
  **within-subject factors**, and the quantity the model was fitted to. If the statistic was
  computed on scores **averaged over** a factor the outcome does not average over — blocks,
  episodes, targets, sessions — it is `context`, however clean its degrees of freedom.
  It is **not** a `value` source, whatever its degrees of freedom, when it is: a test of one group
  against zero or any constant (even when its df equal n_a + n_b − 2); an interaction; a main
  effect of a within-subject factor; a test from a model containing a second between-subjects
  factor or a covariate; an omnibus test over more than two levels (numerator df > 1); a paired or
  repeated-measures test; or a bounded p ("p < .05") rather than an exact one. A χ² is never a
  `value` source for a continuous outcome — no conversion to a standardised mean difference exists.
  A `value` test statistic is a **last resort**: `stats.route_precedence` places `test_statistic`
  fifth of seven. Do not stop looking for printed means because you found an F.
* `baseline` — a pre-manipulation, control-condition or aligned/veridical series plotted or
  printed beside the outcome (the aligned-cursor curve next to the rotated one; a pre-test next to
  a post-test). It could correct the value; it is not the value. Listing it as `value` puts a
  baseline number where the outcome should be.
* `alternate` — a location that measures this same outcome by a **second operationalization** you
  named as an alternative. Do not choose between them here: list both as `value` and say in
  `measure_name` / `operationalization` that there are two. The choice is made once, later, against
  the protocol's own definition and measurement window, and the loser is marked `alternate` then.
* `context` — a location that defines the measurement window, names the blocks, or reports a test
  that does not meet the bar above. Useful to a reader; nothing to extract.

### Error bars
For every source that has a dispersion (figure error bars, table ±, text ±):

* `x_axis_kind` (figures only; use `unknown` for text and tables): what the x axis IS.
  `time` for trials, blocks, episodes, sessions — anything where "the last one" is a meaningful
  instruction. `categorical` for target directions, conditions, hands, groups — a set with no
  order, where the quantity the review wants is the average ACROSS the axis and reading one point
  is a different number, not a less precise one. `other` for a continuous covariate (age, speed).
* `error_bar_type`: SD, SE, CI95, CI90, IQR, RANGE, NONE, or UNKNOWN.
* `error_bar_evidence`: the verbatim words that told you — from the caption, the axis legend, or
  the Methods. If the paper never says what the bars are, `error_bar_type` is `UNKNOWN` and the
  evidence is empty. Never guess from how the bars look: SD read as SE (or the reverse) corrupts
  the whole analysis.
* `error_bar_scope`: `between_subject` for ordinary group dispersion, or
  `within_subject_normalized` when the paper says the bars are within-subject, normalised, or
  corrected for between-subject variance (e.g. Loftus–Masson, Cousineau–Morey).

### Style
Quotes are verbatim, short, and copied exactly as printed (keep the paper's symbols). Leave a
string empty rather than filling it with a guess.
