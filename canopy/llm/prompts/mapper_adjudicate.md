# STUDY MAPPER — ADJUDICATION

Two independent agents mapped the attached paper and disagree. Settle it from the PDF itself, not
by splitting the difference and not by favouring either agent: read the paper and decide what it
really says.

Rule on:

* **Eligibility** under the protocol, with the criterion and a verbatim quote.
* **The datasets**, and for each one the two groups (the paper's own labels, which is group A and
  which is group B under the protocol's definitions) and the number of participants ANALYSED in
  each group — after exclusions and dropouts — quoting the sentence, cell or caption that says so.
  Set `primary_dataset_index` to the 1-based position of the matching dataset in MAP A, or null
  when MAP A has no such dataset.
* **The open error-bar questions listed below**, one ruling each: what the bars at that location
  really represent according to the caption, legend or Methods, with the words that say so. If the
  paper never says, rule `UNKNOWN`. Never infer the type from how the bars look.
* **Every dataset only ONE of the two agents proposed** (`dataset_inclusions`, one row each, listed
  under the disagreements). This is an INCLUSION question, and it is decided by the protocol, not by
  which agent found it. `verdict`:
  - `exclude` — only when a **named protocol rule** excludes it. Put that rule verbatim in `rule`
    and the paper's own sentence that makes the rule apply in `quote`. A rejection without both is
    not a rejection.
  - `include` — the protocol's eligibility and dataset rules cover it, quoting the rule it meets.
  - `unknown` — the paper does not settle it. Say what is missing; a person will answer.
  A dataset you exclude is recorded and never read. Do not exclude a dataset because it is
  inconvenient, small, or reported only in a figure.
* **Every outcome for which the map names TWO measures** (`measure_rulings`, one row each, listed
  under the disagreements). One outcome carries one measure. Decide **only** against the protocol's
  own `definition` and `measurement_window` for that outcome:
  - `winner` — one measure is what the protocol asks for. Give `winning_analysis_metric` (the
    `analysis_metric` of the locations that carry it), `winning_location` (the locator or figure
    id of the location the winner is read at, copied from MAP A), `winner_quote` (the paper's
    words for the measure that wins) and `loser_quote` (the paper's words for the one that loses).
    Both quotes are required; the loser is kept on the record, marked, and never read for a value.
    **Name `winning_location` whenever the two candidate measures carry the SAME
    `analysis_metric`** — the metric alone then names both of them, settles nothing, and the
    question goes to a person instead. You may also list the locations that lose in
    `losing_locations`.
  - `toss_up` — the window fits both, or the paper's wording does not separate them. A person will
    answer, and nothing is read for that outcome until they do. Choosing at random is worse than
    asking: say `toss_up` rather than guessing.
  A location that records no `analysis_metric` at all is one of the candidates: it cannot say
  which of the two measures it reads, so it cannot be read as the winner's number. Rule for the
  measure the protocol asks for, quote the location that reads it as `winner_quote` and the
  unresolved location's own sentence as `loser_quote`; it is then kept on the record and not read.
  Two measures means two different QUANTITIES. The SAME plotted element read against its other
  axis — one pair of bars with a degree axis on the left and a percent axis on the right — is one
  measure written twice: name the measure the protocol asks for and expect both expressions of
  those same marks to stay readable. A percentage printed somewhere else in the paper is not that:
  it may be computed over a different sample or against a different denominator, so treat it as
  its own candidate measure and rule on it.

Where the two maps agree, keep the agreed answer unless the paper contradicts it. Do not extract,
digitize or compute any value.

`page` is the 1-based position of the sheet inside the attached PDF: the first sheet is page 1,
whatever number is printed on it (an article may start at printed page 1057). Ignore the printed
numbers, count sheets — including in the error-bar rulings, whose page and ids must match the
location being ruled on.

## PROTOCOL

{{PROTOCOL}}

## MAP A (first agent)

{{MAP_A}}

## MAP B (second agent)

{{MAP_B}}

## DISAGREEMENTS TO SETTLE

{{DISAGREEMENTS}}
