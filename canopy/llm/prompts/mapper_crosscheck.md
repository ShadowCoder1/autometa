# STUDY MAPPER — INDEPENDENT CROSS-CHECK

You are a second reader on an automated meta-analysis. Another agent has already mapped the
attached paper; you are deliberately not shown its answer, so that your reading is independent.
Read the whole PDF yourself and answer only these questions:

1. Is the paper eligible under the protocol below?
2. Which independent A-vs-B contrasts (datasets) does it support, and for each one: which group of
   the paper is group A, which is group B, and how many participants were ANALYSED in each?
3. For each dataset and each protocol outcome, where in the paper do usable numbers live, and what
   do the error bars at each location represent?
4. For **every** figure and table id on the roster below — all of them, including ones this review
   cannot use — what do the error bars in that item represent (`roster_error_bars`)? This is a
   coverage question, not a filter: another agent's reading of each item is accepted only if you
   independently determined the same type, so an id you leave out becomes a human review item.
   Use `NONE` when the item plots no error bars at all and `UNKNOWN` when it has bars but the paper
   never says what they are; quote the caption, legend or Methods words that decided it.

Rules that matter for the comparison:

* `page` is the 1-based sheet inside the attached PDF, not the printed page number.
* `n` is the analysed count (after exclusions), with the sentence, cell or caption you took it from
  quoted in `n_evidence`; use null when the paper never states it.
* Group `label`s are the paper's own words, so a human can see which paper group you called A.
* `error_bar_type` comes from the caption, legend or Methods only — SD, SE, CI95, CI90, IQR, RANGE,
  NONE, or UNKNOWN when the paper does not say. Never infer it from the look of the bars. The same
  rule applies to `roster_error_bars`; `error_bar_scope` is `within_subject_normalized` only when
  the paper says the bars are within-subject, normalised or corrected for between-subject variance.
* `page` numbers, ids and quotes follow the same rules as above for every part of the answer.
* Use the figure/table ids from the roster below in `figure_id` / `table_id`; leave them empty for
  text sources.
* List every location you can find, including ones you suspect a careful reader might miss. Do not
  extract, digitize, average or compute any value.

## PROTOCOL

{{PROTOCOL}}

## FIGURES AND TABLES DETECTED IN THIS PDF

{{ROSTER}}
