You are reading a few pages of one research paper to transcribe the STATISTICS it prints about ONE
outcome for TWO groups.

You are given the text of those pages (and the cells of any table on them), the definition of the
outcome, the two groups, and the places an earlier reader said the statistics live.

Two kinds of answer, one row each — report every one you find, do not choose between them:

- `test_statistic` — a statistical test the paper reports on this outcome (a t, an F, a
  chi-square, or a bare p value);
- `reported_effect_size` — a standardised effect size the paper itself prints for this outcome,
  with its scale and, when given, its confidence interval.

# What to look for, in this order

1. A test that compares these two groups on this outcome **at the phase or time point the outcome
   definition names** — a contrast for one phase or one time point is worth more here than an
   overall test across the whole session.
2. The effect of the grouping factor on this outcome in the paper's overall analysis.
3. An effect size the paper prints for this outcome.

# Rules — these matter more than finding a number

- **Transcribe, never compute.** Copy what is printed. Do not convert one statistic into another,
  do not derive a p from a statistic or a statistic from a p, and do not work out an effect size:
  that happens later, in code, from exactly what you write down here.
- **Quote first.** `quote` is the verbatim sentence — or table row — that carries the statistic. If
  you cannot quote it, you did not find it.
- `effect_as_written`: the paper's own name for what was tested ("Age", "Group", "Group ×
  Session", "post hoc comparison at block 5").
- `compares_the_two_groups`: `yes` only when the test contrasts the two groups described below on
  the outcome described below; `no` when it tests anything else (a within-participant factor,
  another measure, a test against zero); `unknown` when you cannot tell from these pages.
- `design`, what kind of test it is:
  - `independent_t` — two independent groups compared by a t test;
  - `one_way_between` — an analysis of variance whose only factor is the grouping;
  - `mixed_main_effect` — the effect of the grouping factor in an analysis that *also* has a
    within-participant factor;
  - `interaction` — an interaction term;
  - `ancova` — an analysis adjusted for one or more covariates;
  - `paired` — a comparison within the same participants;
  - `welch` — a t test that does not assume equal variances;
  - `unknown` — the paper does not say.
- `stat_type` and `stat_value`: as printed. For an F, `df1` is the first number in F(df1, df2) and
  `df2` the second; for a t or a chi-square, put its degrees of freedom in `df`. When only a p
  value is printed, `stat_type` is `p` and `stat_value` is null.
- `tails`: 1 or 2 only when the paper says so; null otherwise. Never assume.
- `p_kind`: `exact` for "p = .03", `less_than` for "p < .05", `greater_than` for "p > .10", `ns`
  when the paper only says the effect was not significant, `unknown` when no p is given at all.
  `p_value` holds the number printed next to it.
- `direction`: which of the two groups scored higher **on this outcome**, according to the paper's
  own words — `a_greater`, `b_greater`, or `unknown`. Put the words that establish it in
  `direction_quote`. A significant test with no stated direction is `unknown`: a p value does not
  carry a sign, and guessing one would invert the result.
- For a printed effect size: `reported_value` as printed, `reported_scale` (which effect size it
  is), `standardizer` (what it was divided by, when the paper says), `reported_ci_low` and
  `reported_ci_high` when an interval is given, and `positive_means` — which group a positive value
  favours, per the paper's own wording.
- `admissible`: `true` only when this statistic is a comparison **between participants** of exactly
  these two groups on exactly this outcome, so that it could stand in for their means. Everything
  else is `false` with a short reason in `admissible_reason` (an interaction, a within-participant
  effect, a covariate-adjusted analysis, a different measure, a different phase). Your answer is
  recorded, not obeyed: the rule is applied again in code.
- `status`: `found` when the statistic is printed on these pages; `not_on_these_pages` when nothing
  of this kind is here (leave every number null and the quote empty, and say in `notes` what these
  pages do report); `ambiguous` when two readings are possible (numbers null, both readings with
  their quotes in `notes`).

Never report a statistic these pages do not contain, and never report a statistic about a different
measure as if it were this outcome.

# The outcome

{{OUTCOME}}

# The two groups

{{GROUPS}}

# Where an earlier reader said the statistics are

These are a starting point, not a limit: they may be incomplete or wrong, and the pages you have
been given are the evidence.

{{LOCATIONS}}

Return one row per statistic you found — several rows are normal — plus `notes` for anything a
reviewer should know.
