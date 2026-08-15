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
start at printed page 259) — ignore the printed numbers, count sheets. Check every page number you
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
as text or in a table — never a value you read off a plot. Use the closest `kind`; use `unknown`
only when nothing fits.

`analysis_metric` says what the numbers at that location are: raw endpoint values, changes from
baseline, baseline-corrected values, values expressed as a percentage of the manipulation, or
unknown. Locations can differ within one outcome — record what each one shows.

### Error bars
For every source that has a dispersion (figure error bars, table ±, text ±):

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
