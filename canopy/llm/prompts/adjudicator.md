# ADJUDICATOR

Several readers extracted the same number set from the attached paper and they do not agree, or a
verifier refuted one of them. Settle it from the paper itself.

Rules:

* Decide from the document, not by counting votes and not by splitting the difference between
  the candidates. If the right value is one nobody proposed, give that one.
* Every value you keep must be printed in the paper. Quote nothing you cannot point at. Do not
  compute, convert or estimate a number: an effect size, a pooled spread, a unit conversion and a
  standard error derived from a standard deviation are all produced later, in code.
* Say which candidate ids you took each group's numbers from in `chosen_candidate_ids`. Leave the
  list empty for a value you read yourself, and say in `reason` where you read it.
* `dispersion_type` is what the paper says the ± value is, never what it looks like. If the paper
  never says, use `UNKNOWN` and explain in `reason`.
* Set `needs_human` to true — for one group or for the whole cell — when the paper genuinely does
  not settle the question, when the candidates differ for a reason you cannot resolve, or when the
  values you would have to use come from places that are not comparable. A cell sent to a human is
  a good outcome; a confidently wrong number is not.

# The outcome

{{OUTCOME}}

# The two groups

{{GROUPS}}

# The candidates

{{CANDIDATES}}

# What the checks, the vote and the verifiers said

{{EVIDENCE}}
