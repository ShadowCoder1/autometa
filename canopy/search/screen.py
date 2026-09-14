"""Title/abstract screening: which of the papers an index proposed are worth reading in full.

SECOND GENERATION (design 03 §4, prompts/screen_v2.txt). The screener now sees the PROTOCOL —
both groups with their definitions and synonyms, the outcomes, the hard rules an abstract may
fail and the soft rules it may only meet — and answers six questions per record with a quote
before it decides. Only a failing HARD question (language, population, task, primary study) or a
design that is single-group by construction may exclude; a related design (one that transfers,
generalizes, compares or counterbalances across the dimension separating the groups) is `unknown`
and goes to full text. A code guard enforces that after parsing, and a bounded second look
(`audit_excludes`) re-asks the excludes that still mention a groups word. The first real search
excluded three of the answer key's papers on "no comparison" framing an abstract cannot prove
(design 01); the walk in design 04 §C sends all three to full text under this rubric.

This is the one stage of a search that spends money per record, and the one whose mistakes are
invisible afterwards. A paper wrongly EXCLUDED here is never seen again — it does not appear in the
run, in the forest plot, or in the funnel; nothing downstream can audit it back. A paper wrongly
INCLUDED costs one PDF and one row a human unticks. The whole module is built around that
asymmetry: it is deliberately over-inclusive, it never defaults to `exclude`, and every decision it
writes carries the screener's own sentence beside it (`Candidate.screen_reason`, models.py:92).

Four rules, each of which is a test:

* **The reason is the model's sentence, verbatim.** Never paraphrased, never summarised, never
  dropped. It is the audit trail a reviewer defends in front of an editor. Where this module has
  something of its own to say — the abstract was missing, the word the model used was not one of
  ours — it *appends* a parenthetical and leaves the model's words untouched. Annotate, never
  subtract.
* **`unknown`, not `unsure`.** `assert_valid_output_schema` requires every string enum to offer an
  escape hatch from `UNKNOWN_ENUM_MEMBERS = {"unknown", "not_reported", "none", "ambiguous"}`
  (llm/schemas.py:88, enforced at :117-122) and `client.structured` runs that audit before a byte is
  sent (llm/client.py:368-370). The original design's `["include", "exclude", "unsure"]` therefore
  raised `AssertionError` on every call — screening could not run at all (review §B-BLK1). The
  schema says `unknown`; the page shows the word "unsure"; `counts_of` already maps
  `screen_decision == "unknown"` to the `unsure` count (models.py:233).
* **Nobody is dropped in silence.** A record the model returned no verdict for, a record the cost
  cap was reached before, a record screened with no model at all — each keeps `state
  "not_screened"`, an empty `screen_decision` (so `counts_of` does not count it as screened) and a
  reason saying, in words, that nobody read it.
* **The batch composition is pinned.** Candidates are sorted by `key` and cut into fixed batches, so
  the same paper set always builds byte-identical prompts and hits the disk cache at $0
  (client.py:511-513) rather than re-batching in a different order and paying a second full bill
  (review §D3). Two honest limits on that, because this docstring used to claim more than the code
  did: there is **no resume endpoint** — a search is screened once, in `_start` — and the cache
  lives inside the search's own directory (`jobs.py:75`), so pressing **"Search again" pays the
  full screening bill again** even for byte-identical prompts. The batch map is written into
  `search.json` (`SearchRecord.batches`) as an audit trail, not as a resume key.

**Model role.** `MODELS["screener"]` (claude-sonnet-5) — a named role so a user can override the
screener without moving the verifier. Callers may still pass `MODELS["secondary"]`, which *is*
`claude-sonnet-5`, the same model the design chose. Adding the named role is a follow-up, not a
behaviour change. Not haiku: it saves ~$0.12 on a 200-record search and it is the one model that
rejects `output_config.effort` (llm/providers.py:19-24), so the cheap model is also the one that
cannot be dialled down — a poor trade against a permanent, unauditable `exclude`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.costs import estimate_request_cost
from ..llm.errors import BudgetExceeded, LLMError
from .models import Candidate

__all__ = ["SCREEN_SCHEMA", "SCREEN_DECISIONS", "PROMPT_VERSION", "SYSTEM", "PREAMBLE", "BATCH",
           "MAX_ABSTRACT_CHARS", "MAX_TOKENS", "EFFORT", "QUESTIONS", "ANSWERS", "HARD_QUESTIONS",
           "AUDIT_SHARE", "ScreenBatch", "ScreenOutcome", "screen_candidates", "audit_excludes",
           "preamble_for", "prompt_text", "related_term_in", "NO_MODEL_REASON",
           "NO_VERDICT_REASON", "TITLE_ONLY_NOTE", "NO_ABSTRACT_MARKER", "TRUNCATION_MARKER",
           "cap_reason", "batches_of"]

PROMPT_VERSION = "search-screen-2"

#: the three answers, and the only three. `unknown` is the schema's word for what the page calls
#: "unsure" — see the module docstring and review §B-BLK1.
SCREEN_DECISIONS: tuple[str, ...] = ("include", "exclude", "unknown")

#: records per call. Big enough that the criteria preamble (~700 tokens) is amortised twenty ways;
#: small enough that one refusal or one truncation costs twenty decisions and not two hundred, and
#: that the cost cap is re-checked ten times over a 200-record search.
BATCH = 20

#: characters of abstract sent per record. Publisher abstracts routinely run past this (the Europe
#: PMC record sampled while designing this had 2,140 characters), and the tail of an abstract is
#: methods detail that a title/abstract screen does not turn on. This limit is *load-bearing for the
#: cost estimate* — see the cost note on `screen_candidates`.
MAX_ABSTRACT_CHARS = 1800

#: output budget per batch. Twenty verdicts of six answers, a quote and a sentence each is ~2,000
#: tokens; the headroom is for a model that writes longer reasons rather than for one that writes
#: more of them.
MAX_TOKENS = 4000

#: the six answers and their vocabularies (screen_v2.txt); `q4` can never be "no": an abstract
#: that does not mention an outcome is `unknown` on it, never a failure
QUESTIONS: tuple[str, ...] = ("q1", "q2", "q3", "q4", "q5", "q6")
ANSWERS: dict[str, tuple[str, ...]] = {
    "q1": ("yes", "no", "unknown"), "q2": ("yes", "no", "unknown"), "q3": ("yes", "no", "unknown"),
    "q4": ("yes", "unknown"), "q5": ("named", "possible", "no", "unknown"),
    "q6": ("yes", "no", "unknown")}
#: the questions a "no" to which may exclude on its own — q5 is handled by the guard's own rule
HARD_QUESTIONS: tuple[str, ...] = ("q1", "q2", "q3", "q6")
#: rubric kinds an abstract may definitively fail (the HARD list); the rest are SOFT
HARD_KINDS: frozenset[str] = frozenset({"language", "population", "task", "date"})
PRIMARY_RULE = ("It is a primary study with results, not a review, editorial, protocol or "
                "commentary.")
#: the second look at excludes (`audit_excludes`): at most this share of a round's records
AUDIT_SHARE = 0.10
MAX_SYNONYMS = 12
MAX_DEFINITION_CHARS = 400
MAX_OUTCOME_CHARS = 240

#: reading a title and an abstract against written criteria is a reading task, not a reasoning task.
EFFORT = "low"

#: what this module says when it, and not the model, is the author of a reason.
NO_MODEL_REASON = "no model was available to screen this record"
NO_VERDICT_REASON = ("the screener returned no verdict for this record — nobody read it, so it is "
                     "listed unscreened rather than excluded")
TITLE_ONLY_NOTE = "(screened on the title alone: this record had no abstract.)"
NO_ABSTRACT_MARKER = "[no abstract was available for this record — screen it on its title alone]"
TRUNCATION_MARKER = " […abstract truncated]"

#: `state` after a verdict, when no PDF is on disk yet. An `include` becomes `wanted`, NOT
#: `paywalled`: the fetch stage has not run, so nothing has established that a paywall exists,
#: and telling a user their paper sits behind one is a claim about a publisher nobody contacted.
#: `fetch` overwrites this with `fetched`, or with the `paywalled` it actually measured — and a
#: search that stops before fetching leaves rows reading "wanted", which is what is true.
_STATE_FOR: dict[str, str] = {"include": "wanted", "exclude": "excluded", "unknown": "unsure"}

#: `include` and `unknown` default to ON, `exclude` to OFF (design §2.6). Screening is
#: over-inclusive by construction: the run's own mapper decides eligibility per paper and records
#: it, so an `unknown` that reaches the run is corrected there, while an `exclude` that should not
#: have been is corrected nowhere.
_KEEP_FOR: dict[str, bool] = {"include": True, "unknown": True, "exclude": False}


# --------------------------------------------------------------------------------- the schema
#: One batch verdict. Strict objects, every property required, no derived statistic anywhere — the
#: house rules `assert_valid_output_schema` / `assert_no_derived_stats` enforce (llm/schemas.py).
#: `tests/test_search_screen.py` runs both audits against this constant, because a schema that the
#: client rejects is a stage that cannot run at all.
SCREEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "description": "One entry per record you were shown, in any order.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "q1", "q2", "q3", "q4", "q5", "q6", "decision", "quote",
                             "reason"],
                "properties": {
                    "ref": {"type": "string",
                            "description": "The record's number, exactly as it was given to you."},
                    "q1": {"type": "string", "enum": list(ANSWERS["q1"]),
                           "description": "language — does the record satisfy the language rule?"},
                    "q2": {"type": "string", "enum": list(ANSWERS["q2"]),
                           "description": "population — could the study include participants the "
                                          "population rules allow?"},
                    "q3": {"type": "string", "enum": list(ANSWERS["q3"]),
                           "description": "task — is the task or design the hard rules require "
                                          "present? \"no\" only when definitively absent."},
                    "q4": {"type": "string", "enum": list(ANSWERS["q4"]),
                           "description": "outcome — is any protocol outcome reported? Never "
                                          "\"no\" from an abstract."},
                    "q5": {"type": "string", "enum": list(ANSWERS["q5"]),
                           "description": "groups — does the DESIGN contain both GROUP A and "
                                          "GROUP B? named / possible / no / unknown."},
                    "q6": {"type": "string", "enum": list(ANSWERS["q6"]),
                           "description": "primary — is this a primary study reporting its own "
                                          "results?"},
                    "decision": {
                        "type": "string",
                        "enum": list(SCREEN_DECISIONS),
                        "description": "exclude only if q1 = no, q2 = no, q3 = no, q6 = no, or "
                                       "q5 = no; include if q1, q2, q3 and q6 = yes and q5 = "
                                       "named; unknown otherwise.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "The record's own words — at most 25 of them — that settle "
                                       "the decisive question.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence naming the rule met or failed. Never empty: "
                                       "this sentence is shown to the reviewer as the "
                                       "justification for the decision.",
                    },
                },
            },
        },
    },
}

SYSTEM = """You screen titles and abstracts for a systematic review. For each numbered record you decide
whether it should be read in full. You never invent a fact about a record you were not shown, and
you never decide what a paper found.

The protocol names two groups and the outcomes it will compare them on. You are screening for the
presence of a DESIGN, not for a paper's topic: a paper whose stated question is something else
entirely still belongs in the review if its design contains both groups and the task the protocol
requires. Abstracts describe what a paper is about, not everything its methods contain, so the
only safe exclusions from an abstract are the ones an abstract can prove.

Three answers. `include`: the hard rules are met and the abstract NAMES both groups, or the
dimension that separates them, as part of the design. `exclude`: the record definitively fails a
hard rule — wrong language, no eligible population, the required task definitively absent, or not
a primary study with results (a review, meta-analysis, editorial, commentary, protocol or
abstract-only item) — or its design is single-group by construction with no second experiment
mentioned. `unknown`: everything else, and in particular every record whose stated question is a
RELATED design: one that transfers, generalizes, compares or counterbalances across the dimension
that separates the two groups; a manipulation replicated in both groups with a control condition;
a clinical study with control groups drawn from both; a modelling or mechanism study that must
have collected data from both. What those look like depends on the field — for example a
stimulation or feedback study, a dose-arm or placebo-arm trial, a wait-list comparison, an
age-matched control group, a training-then-test design. Those go to full-text screening. They are
never excluded here.

Outcomes are OR'd: one reported outcome is enough, and an abstract that does not mention an
outcome is `unknown` on that question, never a failure.

A paper you wrongly exclude here is never looked at again; a paper you wrongly send to full text
costs one download. Every decision carries a quote from the record itself and one sentence naming
the rule it meets or fails.

If a record has no abstract, say so in your reason and answer `unknown` unless the title alone
settles a hard rule."""

PREAMBLE = """The review asks:

{QUESTION}

THE PROTOCOL

GROUP A — {A_LABEL}: {A_DEFINITION}
  also called: {A_SYNONYMS}
GROUP B — {B_LABEL}: {B_DEFINITION}
  also called: {B_SYNONYMS}

OUTCOMES (any one suffices):
{OUTCOMES}

HARD RULES (an abstract may fail these):
{HARD_RULES}

SOFT RULES (an abstract may meet these, never fail them):
{SOFT_RULES}

For each record answer, in order:
  q1 language — does the record satisfy the language rule?   yes / no / unknown
     (answer yes when there is no language rule)
  q2 population — could the study include participants the population rules allow?
                                                                yes / no / unknown
  q3 task — is the task or design the hard rules require present?   yes / no / unknown
     ("no" only when it is definitively absent)
  q4 outcome — is any protocol outcome reported?                yes / unknown
     (never "no" from an abstract)
  q5 groups — does the DESIGN contain both GROUP A and GROUP B?
     named    the abstract names both, or names the dimension separating them, as part of the design
     possible the design is a related design (see your instructions) or could otherwise hold data
              from both groups
     no       single-group or single-condition by construction, and no second experiment mentioned
     unknown  the record does not say
  q6 primary — is this a primary study reporting its own results?   yes / no / unknown
     ("no" for a review, meta-analysis, editorial, commentary, letter, protocol without results,
     or an abstract-only conference item that reports none)

Then decide:
  exclude   only if q1 = no, q2 = no, q3 = no, q6 = no, or q5 = no.
  include   if q1, q2, q3 and q6 = yes and q5 = named.
  unknown   otherwise.

`quote` is the record's own words — at most 25 of them — that settle the decisive question.
`reason` is one sentence naming the rule met or failed."""

RECORDS_HEAD = "Screen the {N} records below. Answer once per record, using its number as `ref`."

#: the one line the exclude audit prefixes to a record (screen_v2.txt, EXCLUDE AUDIT)
AUDIT_LINE = "This record contains the words: {TERMS}. Answer q5 again with that in view."


# --------------------------------------------------------------------------------- outcome shapes
@dataclass
class ScreenBatch:
    """One batch, sent or not — the pinned composition plus what it cost.

    `keys` is the audit trail: `run.py` writes these rows into `search.json` so a reader can price
    screening per batch and see which twenty records one failed call cost. It is NOT a resume key —
    there is no resume, and the module docstring says what "Search again" actually costs.
    """

    index: int
    keys: list[str]
    sent: bool = False
    #: what the cap was checked against, before the call — the WORST case, which is this call plus
    #: the one `client.structured` makes if the model runs out of output room. See
    #: `screen_candidates`; reserving only the first call let one batch bill 2.6× its reservation.
    estimated_usd: float = 0.0
    cost_usd: float = 0.0            # what it actually cost
    n_verdicts: int = 0              # verdicts that landed on a candidate in this batch
    error: str = ""                  # "" unless the call failed; the failure's own words
    audit: bool = False              # the exclude audit's second look, not a screening batch


@dataclass
class ScreenOutcome:
    """What screening did. The candidates themselves carry every decision; this is the summary.

    `stopped_because` mirrors `SearchRecord.stopped_because` (models.py:207): `""` when screening
    ran to the end of the list, `"budget"` when the cap stopped it. A capped search still finishes
    `done`, so a page keying its banner on the job status alone would never tell the user their
    search was cut short.
    """

    batches: list[ScreenBatch] = field(default_factory=list)
    n_screened: int = 0              # candidates a model actually returned a verdict for
    n_audited: int = 0               # excludes asked again by `audit_excludes`
    n_reversed: int = 0              # …of which changed their mind, to `unknown`
    cost_usd: float = 0.0
    stopped_because: str = ""        # "" | "budget"
    notes: list[str] = field(default_factory=list)


def _money(usd: float) -> str:
    """`$1.00`, but `$0.0010` for a cap small enough that two decimal places would print `$0.00`."""
    return f"${usd:.2f}" if usd >= 0.01 else f"${usd:.4f}"


def cap_reason(budget_usd: float | None) -> str:
    """The sentence a record gets when the cap stopped the search before anyone read it.

    It names the cap that actually fired — the caller's `budget_usd` when this module's own
    between-batch check stopped things, the client's when its reservation refused the call — because
    a reader asking "why was this paper never screened?" needs the number they can change.
    """
    if budget_usd is None:
        return "the search's cost cap was reached before this record was screened"
    return f"the search's {_money(budget_usd)} cost cap was reached before this record was screened"


# --------------------------------------------------------------------------------- small helpers
def batches_of(candidates: Sequence[Candidate], size: int = BATCH) -> list[list[Candidate]]:
    """The pinned composition: sorted by `key`, then cut into fixed runs of `size`.

    Sorted rather than left in arrival order because arrival order depends on which index answered
    first, which is a network fact, not a property of the paper set — and a resumed search that
    re-batches in a different order misses every cache entry and bills the user a second time with
    no warning (review §D3).
    """
    if size < 1:
        raise ValueError(f"a screening batch holds at least one record, not {size}")
    ordered = sorted(candidates, key=lambda c: c.key)
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def _truncate_abstract(abstract: str, limit: int = MAX_ABSTRACT_CHARS) -> str:
    """Whitespace-collapsed abstract, cut to `limit` characters with a visible marker.

    The marker matters: a model shown a sentence that stops mid-clause with no explanation may treat
    the missing half as absent from the paper, which is exactly the invented fact the system prompt
    forbids.
    """
    text = " ".join((abstract or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + TRUNCATION_MARKER


def _record_block(index: int, candidate: Candidate) -> str:
    """One numbered record, exactly as the model sees it."""
    head = f"{index}. {candidate.title.strip() or '[no title]'}"
    meta = ", ".join(x for x in (str(candidate.year) if candidate.year else "",
                                 candidate.venue.strip()) if x)
    if meta:
        head = f"{head} ({meta})"
    body = _truncate_abstract(candidate.abstract) or NO_ABSTRACT_MARKER
    return f"{head}\n{body}"


def _group_lines(protocol: Any, key: str, letter: str) -> tuple[str, str, str]:
    group = getattr(protocol, key, None) if protocol is not None else None
    if group is None:
        return (f"(the protocol names no group {letter})", "", "(none)")
    label = " ".join(str(getattr(group, "label", "") or "").split()) or f"group {letter}"
    definition = " ".join(str(getattr(group, "definition", "") or "").split())
    synonyms = [str(x).strip() for x in (getattr(group, "synonyms", None) or []) if str(x).strip()]
    return (label, definition[:MAX_DEFINITION_CHARS],
            ", ".join(synonyms[:MAX_SYNONYMS]) or "(none)")


def _rule_lists(rubric: Sequence[Mapping[str, Any]], protocol: Any,
                criteria: Sequence[str]) -> tuple[list[str], list[str]]:
    """HARD (an abstract may fail) and SOFT (may only meet) rules, from the plan's rubric; with
    no rubric, the protocol's eligibility list verbatim under SOFT and the primary-study rule
    alone under HARD. The primary-study rule is always the last HARD rule."""
    hard: list[str] = []
    soft: list[str] = []
    rows = [r for r in (rubric or ()) if isinstance(r, Mapping) and str(r.get("rule") or "").strip()]
    for row in rows:
        text = " ".join(str(row["rule"]).split())
        kind = str(row.get("kind") or "").strip().lower()
        if bool(row.get("abstract_can_fail")) and kind in HARD_KINDS:
            hard.append(text)
        else:
            soft.append(text)
    if not rows:
        source = list(getattr(protocol, "eligibility", None) or []) if protocol is not None \
            else list(criteria or [])
        soft = [" ".join(str(x).split()) for x in source if str(x).strip()]
    if not any(PRIMARY_RULE.lower()[:30] in h.lower() or "primary study" in h.lower()
               for h in hard):
        hard.append(PRIMARY_RULE)
    return hard, soft


def _numbered(rules: Sequence[str]) -> str:
    return "\n".join(f"  {i}. {rule}" for i, rule in enumerate(rules, start=1)) or "  (none)"


def preamble_for(*, question: str, protocol: Any = None,
                 rubric: Sequence[Mapping[str, Any]] = (),
                 criteria: Sequence[str] = ()) -> str:
    """Everything before the records — byte-identical across batches and rounds, which is what
    makes it a cache prefix (`_prompt_for`)."""
    a_label, a_def, a_syn = _group_lines(protocol, "group_a", "A")
    b_label, b_def, b_syn = _group_lines(protocol, "group_b", "B")
    outcomes = []
    for outcome in (getattr(protocol, "outcomes", None) or []) if protocol is not None else []:
        label = " ".join(str(getattr(outcome, "label", "") or "").split())
        definition = " ".join(str(getattr(outcome, "definition", "") or "").split())
        outcomes.append(f"  - {label}: {definition[:MAX_OUTCOME_CHARS]}")
    if not outcomes:
        outcomes = ["  - (the outcome named in the question)"]
    hard, soft = _rule_lists(rubric, protocol, criteria)
    return PREAMBLE.format(QUESTION=question.strip() or "(no question was given)",
                           A_LABEL=a_label, A_DEFINITION=a_def, A_SYNONYMS=a_syn,
                           B_LABEL=b_label, B_DEFINITION=b_def, B_SYNONYMS=b_syn,
                           OUTCOMES="\n".join(outcomes), HARD_RULES=_numbered(hard),
                           SOFT_RULES=_numbered(soft))


def _records_block(batch: Sequence[Candidate], prefix_lines: Mapping[str, str] | None = None
                   ) -> str:
    """The numbered records, one block. `prefix_lines` (key → one line) is the audit's."""
    records = []
    for i, candidate in enumerate(batch, start=1):
        block = _record_block(i, candidate)
        line = (prefix_lines or {}).get(candidate.key)
        records.append(f"{line}\n{block}" if line else block)
    return RECORDS_HEAD.format(N=len(batch)) + "\n\n" + "\n\n".join(records)


def _prompt_for(batch: Sequence[Candidate], *, question: str, protocol: Any = None,
                rubric: Sequence[Mapping[str, Any]] = (), criteria: Sequence[str] = (),
                preamble: str | None = None,
                prefix_lines: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """The user message for one batch, as TWO text blocks: the preamble, marked as a cache
    breakpoint, then the records. Deterministic, because the disk cache keys on it.

    The preamble (≈ 1,000 tokens) plus the system prompt (≈ 380) clears the provider's
    1,024-token cache minimum, so every batch after the first within its five-minute window
    reads the preamble from cache — ≈ 8 % of input, ≈ $0.002 a batch (design 03 §4). The v1
    screener passed one plain string and `cache_creation_input_tokens` was 0 on every call.
    """
    from ..llm.client import mark_cacheable

    text = preamble if preamble is not None else preamble_for(
        question=question, protocol=protocol, rubric=rubric, criteria=criteria)
    return (mark_cacheable([{"type": "text", "text": text}])
            + [{"type": "text", "text": _records_block(batch, prefix_lines)}])


def prompt_text(content: Sequence[Any]) -> str:
    """The user message as one string (for a test, a bench replay, a log)."""
    return "\n\n".join(str(b.get("text") or "") for b in content
                       if isinstance(b, Mapping) and b.get("type") == "text")


def _normalise(decision: Any) -> tuple[str, str]:
    """`(decision, note)` — one of `SCREEN_DECISIONS`, plus what to append when it was not.

    A strict schema should make this dead code, but a model that answers "unsure" (the word the page
    shows) or a provider that relaxes the enum must not silently become an `exclude`, and must not
    land a word `counts_of` cannot count. Anything unrecognised becomes `unknown` — the answer that
    keeps the paper visible — and says so.
    """
    word = str(decision or "").strip().lower()
    if word in SCREEN_DECISIONS:
        return word, ""
    return "unknown", (f'(the screener answered "{decision}", which is not one of '
                       f'{"/".join(SCREEN_DECISIONS)}; it is recorded as unknown.)')


def _answers_of(entry: Mapping[str, Any]) -> dict[str, str]:
    """`q1`…`q6` as the model wrote them, each forced onto its own vocabulary (an off-enum word
    is `unknown`), plus the quote."""
    out: dict[str, str] = {}
    for question in QUESTIONS:
        word = str(entry.get(question) or "").strip().lower()
        out[question] = word if word in ANSWERS[question] else "unknown"
    out["quote"] = " ".join(str(entry.get("quote") or "").split())[:400]
    return out


def related_term_in(candidate: Candidate, plan: Mapping[str, Any] | None) -> str:
    """The first `groups` or `related_designs` term literally present in the record's title and
    abstract (case-folded, hyphens as spaces, whole-word), or `""`. The same literal test
    `rank.coverage` makes, on the same block terms."""
    if not plan:
        return ""
    from .blocks import block_terms
    from .rank import coverage

    text = f"{candidate.title or ''} {candidate.abstract or ''}"
    for block in plan.get("blocks") or []:
        if block.get("name") not in ("groups", "related_designs"):
            continue
        for term in block_terms(block):
            if coverage(text, [term]):
                return str(term)
    return ""


def _guard(decision: str, answers: Mapping[str, str], candidate: Candidate,
           plan: Mapping[str, Any] | None) -> tuple[str, str]:
    """The decision rules of the rubric, enforced in code (screen_v2.txt, DECISION GUARD).

    The model's words are never changed; when the guard overrides, the decision becomes
    `unknown` and a parenthetical says why. Three rules: an exclude with no failing hard
    question and q5 not "no"; an exclude where q5 is the ONLY "no" (q1, q2, q3, q6 each yes or
    unknown) and the record mentions a groups or related-designs word — the transfer abstract
    that never names the perturbation answers q3 = unknown, and that must not let the exclude
    through (design 04 MAJOR-3); an include that did not name both groups on a fully met rubric.
    """
    hard_no = {q for q in HARD_QUESTIONS if answers.get(q) == "no"}
    if decision == "exclude" and not hard_no and answers.get("q5") != "no":
        return "unknown", ("(the screener excluded without a failing hard question; recorded as "
                           "unknown.)")
    if decision == "exclude" and answers.get("q5") == "no" and not hard_no:
        term = related_term_in(candidate, plan)
        if term:
            return "unknown", (f"(the record mentions \"{term}\" from the groups or "
                               f"related-designs block; sent to full text.)")
    if decision == "include" and (answers.get("q5") != "named" or hard_no
                                  or any(answers.get(q) == "unknown" for q in HARD_QUESTIONS)):
        return "unknown", ("(included without naming both groups on a fully met rubric; recorded "
                           "as unknown.)")
    return decision, ""


def _reason_for(model_reason: Any, *, title_only: bool, note: str) -> str:
    """The model's sentence verbatim, plus our own parentheticals — never instead of it."""
    parts = [str(model_reason or "").strip()]
    if not parts[0]:
        parts = ["the screener gave no reason for this decision"]
    if title_only:
        parts.append(TITLE_ONLY_NOTE)
    if note:
        parts.append(note)
    return " ".join(parts)


def _write_verdict(candidate: Candidate, *, decision: str, reason: str, title_only: bool,
                   answers: Mapping[str, Any] | None = None) -> None:
    """Put one verdict on one candidate."""
    candidate.screen_decision = decision
    candidate.screen_reason = reason
    candidate.screened_on_title_only = title_only
    candidate.keep = _KEEP_FOR[decision]
    if answers is not None:
        candidate.screen_answers = dict(answers)
    # A candidate that already has a PDF on disk keeps the state the file gave it: `fetched`,
    # `uploaded` and `extra` are facts about the filesystem, and a screener's opinion does not
    # un-fetch a paper. The decision, the reason and `keep` are still recorded, so an excluded
    # paper whose PDF a human attached still shows why the screener did not want it.
    if not candidate.pdf_path:
        candidate.state = _STATE_FOR[decision]


def _write_unscreened(candidate: Candidate, reason: str) -> None:
    """Mark a candidate nobody read — with the reason nobody read it.

    `screen_decision` stays `""` on purpose: `counts_of` counts a candidate as *screened* exactly
    when that field is non-empty (models.py:226), so an unread record must never carry a decision,
    and `keep` is False because the tool has no grounds to put an unread paper into a review.
    """
    candidate.screen_decision = ""
    candidate.screen_reason = reason
    candidate.keep = False
    if not candidate.pdf_path:
        candidate.state = "not_screened"


def _client_cost(client: Any) -> float:
    """What the client has spent in total, or 0.0 for a client that does not keep score."""
    total = getattr(client, "total_cost", None)
    try:
        return float(total()) if callable(total) else 0.0
    except Exception:                                # pragma: no cover - a stand-in without a ledger
        return 0.0


def _client_budget(client: Any, default: float | None = None) -> float | None:
    """The client's own USD budget, when it has one — the cap that raised `BudgetExceeded`."""
    budget = getattr(client, "budget_usd", None)
    return float(budget) if isinstance(budget, (int, float)) else default


def _emit(on_progress: Callable[[Mapping[str, Any]], None] | None, **event: Any) -> None:
    """Progress, straight to the caller's callback.

    Not through `pipeline.state.emit`: that builds a fixed five-key dict and takes no extra keys
    (state.py:256-261), so `n`/`total` — the two the bar needs — cannot travel through it.
    """
    if on_progress is None:
        return
    on_progress(dict(event))


# --------------------------------------------------------------------------------- the stage
def screen_candidates(client: Any | None, candidates: Sequence[Candidate], *,
                      question: str, model: str, budget_usd: float | None, cell_key_prefix: str,
                      protocol: Any = None, rubric: Sequence[Mapping[str, Any]] = (),
                      plan: Mapping[str, Any] | None = None, criteria: Sequence[str] = (),
                      round_index: int = 0, audit: bool = True,
                      batch_size: int = BATCH, max_tokens: int = MAX_TOKENS,
                      on_progress: Callable[[Mapping[str, Any]], None] | None = None,
                      ) -> ScreenOutcome:
    """Screen `candidates` on title + abstract, writing every verdict onto the candidate itself.

    Mutates each `Candidate` in place — `screen_decision`, `screen_reason`, `screen_answers`,
    `screened_on_title_only`, `state`, `keep` — and returns the summary. Sequential by design: the
    cost cap is checked *between* batches, and a batch already paid for is never thrown away.

    The screener sees the PROTOCOL (`protocol`), the plan's `rubric` split into hard and soft
    rules, and answers q1–q6 with a quote per record (`prompts/screen_v2.txt`); `_guard` then
    enforces the decision rules in code, and `audit_excludes` re-asks the excludes that still
    mention a groups word (`audit=True`, ≤ `AUDIT_SHARE` of the round). `criteria` is the
    fallback for the soft rules when there is no rubric and no protocol. `round_index` names the
    citation-chasing round in the call key (`<prefix>:r<round>:<batch>`).

    `client is None` means no model is available (no API key). That is not an error: every record
    comes back `not_screened` with the reason "no model was available to screen this record",
    `keep=False` and `stopped_because=""`, and the user reads the abstracts themselves. Degrade
    honestly, never die.

    **The cost cap is hard, and its worst case is one batch — INCLUDING that batch's retry.**
    Before each batch the next call's reservation price (`estimate_request_cost`, costs.py:217-221)
    is added to what screening has already spent; if that exceeds `budget_usd` the stage stops
    cleanly, every remaining record is `not_screened` with a reason naming the cap, and
    `stopped_because == "budget"`. The reservation is **two** calls, not one: when the model runs
    out of output room `client.structured` silently retries at double `max_tokens`
    (client.py:539-546), and that retry is a second billed call. Reserving only the first one let a
    measured batch of 20 bill **$0.134 against a $0.051 reservation** — the retry, not tokeniser
    variance, was the dominant term, and it is deterministic rather than occasional (review §M8).
    `LLMClient` reserves its own money before each call (client.py:525) and raises `BudgetExceeded`
    if its budget would not take it — that is caught here, charged to the ledger and turned into the
    same clean stop, never a lost batch. The residual overshoot is now only one batch's *actual*
    cost above its *reservation*, which is positive when the real tokeniser beats the `chars/3.5`
    estimate (costs.py:143): **≈ $0.05 on ordinary English abstracts and ≈ $0.15 on a batch of dense
    non-Latin abstracts**, where CJK text runs nearer one character per token. The client logs real
    cost, so the next check sees it and the cap self-corrects; the overshoot cannot compound across
    batches (review §D2).

    **What it costs.** Real v1 batches billed $0.393834 / 200 = $0.00197 a record (10,193–11,721
    input and 1,148–2,386 output tokens a batch); the six answers and a 25-word quote add ≈ 70
    output tokens a record → `cost.COST_PER_RECORD` = $0.003, ≈ $0.06 a batch of 20. The
    *reservation* is larger than the actual — it prices `max_tokens` of output for two calls,
    ≈ $0.13 a batch — and the between-batch check compares actual spend plus that one estimate
    (the client releases each reservation after its call), so a $3.50 cap stops near batch 87,
    not 26. That is the price of a cap that cannot be walked past: a cap the retry could
    overshoot is not a cap, and a user who wants the other records raises a number they can see.

    `cell_key_prefix` is the call log's handle on this stage — pass `f"screen:{search_id}"`; each
    batch is logged as `<prefix>:r<round>:<index>` so a reader can price screening apart from
    the rest.
    """
    outcome = ScreenOutcome()
    records = list(candidates)
    total = len(records)
    if not records:
        return outcome

    # ---- no model: say so on every record, and change nothing else about the search.
    if client is None:
        for candidate in records:
            _write_unscreened(candidate, NO_MODEL_REASON)
        outcome.notes.append(
            f"no model was configured, so none of the {total} records were screened — every one of "
            f"them is listed for you to read")
        _emit(on_progress, stage="screen", status="skipped", n=0, total=total, cost_so_far=0.0,
              message=NO_MODEL_REASON)
        return outcome

    preamble = preamble_for(question=question, protocol=protocol, rubric=rubric,
                            criteria=criteria)
    state = _PassState(client=client, model=model, budget_usd=budget_usd, max_tokens=max_tokens,
                       preamble=preamble, plan=plan, outcome=outcome, total=total,
                       on_progress=on_progress)
    for index, batch in enumerate(batches_of(records, batch_size)):
        state.run_batch(batch, index=index, cell_key=f"{cell_key_prefix}:r{round_index}:{index}")

    if audit and not state.stopped_reason:
        audit_excludes(state, records, cell_key_prefix=f"{cell_key_prefix}:r{round_index}",
                       batch_size=batch_size)

    outcome.cost_usd = round(state.spent, 6)
    return outcome


@dataclass
class _PassState:
    """One screening pass's ledger: what was spent, whether the cap fired, and the one sentence
    every unread record gets once it has."""

    client: Any
    model: str
    budget_usd: float | None
    max_tokens: int
    preamble: str
    plan: Mapping[str, Any] | None
    outcome: ScreenOutcome
    total: int
    on_progress: Callable[[Mapping[str, Any]], None] | None = None
    spent: float = 0.0
    done: int = 0
    #: set once the cap fires — the exact sentence every remaining record gets, so the whole tail
    #: of the search names one cap rather than each batch inventing its own wording.
    stopped_reason: str = ""

    def run_batch(self, batch: Sequence[Candidate], *, index: int, cell_key: str,
                  prefix_lines: Mapping[str, str] | None = None,
                  audit_of: Mapping[str, str] | None = None) -> ScreenBatch:
        """One call for one batch, with the cap checked first and every failure recorded.

        `audit_of` (key → the decision being re-asked) makes this an audit batch: a changed
        verdict becomes `unknown`, an unchanged one stays, and neither touches the batch count."""
        record = ScreenBatch(index=index, keys=[c.key for c in batch],
                             audit=audit_of is not None)
        self.outcome.batches.append(record)
        client, outcome = self.client, self.outcome
        if self.stopped_reason:                      # the cap already fired; nobody reads these
            if audit_of is None:
                for candidate in batch:
                    _write_unscreened(candidate, self.stopped_reason)
            return record

        messages = [{"role": "user", "content": _prompt_for(
            batch, question="", preamble=self.preamble, prefix_lines=prefix_lines)}]
        # this call AND the retry it may provoke — see the money paragraph in the docstring. Two
        # separate estimates rather than one at `2 * max_tokens`, because the retry re-sends the
        # whole prompt too: the worst case is 2× input, not 2× output.
        record.estimated_usd = (
            estimate_request_cost(self.model, SYSTEM, messages, self.max_tokens)
            + estimate_request_cost(self.model, SYSTEM, messages, self.max_tokens * 2))

        # The cap, checked BETWEEN batches: the work already paid for is kept, and the batch that
        # would not fit is never sent.
        if self.budget_usd is not None and self.spent + record.estimated_usd > self.budget_usd:
            self.stopped_reason = cap_reason(self.budget_usd)
            outcome.stopped_because = "budget"
            outcome.notes.append(
                f"the {_money(self.budget_usd)} cost cap stopped screening after {self.done} of "
                f"{self.total} records; the rest are listed unscreened")
            if audit_of is None:
                for candidate in batch:
                    _write_unscreened(candidate, self.stopped_reason)
            _emit(self.on_progress, stage="screen", status="skipped", n=self.done,
                  total=self.total, cost_so_far=round(self.spent, 6), message=outcome.notes[-1])
            return record

        billed_before = _client_cost(client)
        try:
            result = client.structured(
                model=self.model, system=SYSTEM, schema=SCREEN_SCHEMA, effort=EFFORT,
                max_tokens=self.max_tokens, prompt_version=PROMPT_VERSION, cell_key=cell_key,
                messages=messages)
        except BudgetExceeded as exc:
            # The client's own reservation refused it (client.py:331-340). Same clean stop, same
            # wording: the user is told about the cap, not about an exception. The cap NAMED is the
            # client's, because that is the one that fired — it can be lower than `budget_usd`, and
            # a reason quoting the wrong number sends the user to change the wrong setting.
            self.stopped_reason = cap_reason(_client_budget(client, default=self.budget_usd))
            outcome.stopped_because = "budget"
            record.error = str(exc)
            # A `BudgetExceeded` raised AFTER the provider answered — a retry the reservation
            # would not take — was still billed. This branch used to leave `record.cost_usd` at
            # zero, so a batch that really cost $0.052 was reported to the user as $0.00 and the
            # cap's own arithmetic never saw it (review §M9). Same ledger, same rule as `LLMError`.
            record.cost_usd = max(0.0, _client_cost(client) - billed_before)
            self.spent += record.cost_usd
            outcome.notes.append(
                f"the cost cap stopped screening after {self.done} of {self.total} records; the "
                f"rest are listed unscreened")
            if audit_of is None:
                for candidate in batch:
                    _write_unscreened(candidate, self.stopped_reason)
            _emit(self.on_progress, stage="screen", status="skipped", n=self.done,
                  total=self.total, cost_so_far=round(self.spent, 6), message=outcome.notes[-1])
            return record
        except LLMError as exc:
            # A refusal, a truncation or unparseable JSON costs THIS batch and no other — the
            # reason batches are twenty records and not two hundred. The failure is recorded on
            # every record it touched, in the failure's own words.
            record.error = f"{type(exc).__name__}: {exc}"
            # A call that failed AFTER the provider answered was still billed. Charging it to the
            # cap keeps a run of failing batches from spending past the number the user set.
            record.cost_usd = max(0.0, _client_cost(client) - billed_before)
            self.spent += record.cost_usd
            if audit_of is None:
                outcome.notes.append(f"batch {index + 1} failed ({record.error}); its "
                                     f"{len(batch)} records are listed unscreened")
                for candidate in batch:
                    _write_unscreened(candidate, f"the screener failed on this batch and nobody "
                                                 f"read this record ({record.error})")
            else:
                outcome.notes.append(f"the exclude audit's batch {index + 1} failed "
                                     f"({record.error}); its {len(batch)} excludes stand")
            _emit(self.on_progress, stage="screen", status="error", n=self.done,
                  total=self.total, cost_so_far=round(self.spent, 6), message=outcome.notes[-1])
            return record

        record.sent = True
        record.cost_usd = float(getattr(result, "cost_usd", 0.0) or 0.0)
        self.spent += record.cost_usd
        parsed = getattr(result, "parsed", None)
        if audit_of is None:
            n_verdicts = _apply(batch, parsed, self.plan)
            record.n_verdicts = n_verdicts
            self.done += n_verdicts
            outcome.n_screened += n_verdicts
            _emit(self.on_progress, stage="screen", status="progress", n=self.done,
                  total=self.total, cost_so_far=round(self.spent, 6),
                  message=f"screened {self.done} of {self.total} records")
        else:
            record.n_verdicts = _apply_audit(batch, parsed, self.plan)
            outcome.n_audited += len(batch)
            outcome.n_reversed += record.n_verdicts
        return record


def audit_excludes(state: _PassState, records: Sequence[Candidate], *, cell_key_prefix: str,
                   batch_size: int = BATCH) -> int:
    """The bounded second look (screen_v2.txt, EXCLUDE AUDIT).

    Population: the excludes whose only "no" was q5 and whose title + abstract still contains a
    `groups` or `related_designs` word — the guard already sent those to `unknown` when the
    hard questions were all yes/unknown, so what is left here are the excludes with a failing
    hard question that ALSO mention a groups word, up to `AUDIT_SHARE` of the round, most
    relevant first. Each is re-asked with one line naming the words; a verdict that changes
    becomes `unknown`, one that repeats `exclude` stands, and `screen_answers.audit` says which.
    Returns how many were reversed.
    """
    pool = []
    for candidate in records:
        if candidate.screen_decision != "exclude" or candidate.screen_answers.get("q5") != "no":
            continue
        term = related_term_in(candidate, state.plan)
        if term:
            pool.append((candidate, term))
    if not pool:
        return 0
    pool.sort(key=lambda pair: (-(pair[0].relevance or 0.0), pair[0].key))
    cap = max(1, int(len(records) * AUDIT_SHARE))
    chosen = pool[:cap]
    lines = {c.key: AUDIT_LINE.format(TERMS=f'"{term}"') for c, term in chosen}
    before = state.outcome.n_reversed
    for index, batch in enumerate(batches_of([c for c, _ in chosen], batch_size)):
        state.run_batch(batch, index=index, cell_key=f"{cell_key_prefix}:audit:{index}",
                        prefix_lines=lines, audit_of={c.key: "exclude" for c in batch})
        if state.stopped_reason:
            break
    reversed_now = state.outcome.n_reversed - before
    if chosen:
        state.outcome.notes.append(
            f"{len(chosen)} exclude(s) that still mentioned a groups word were asked again "
            f"({len(pool)} qualified, at most {AUDIT_SHARE:.0%} of the round): {reversed_now} "
            f"changed to unsure, {len(chosen) - reversed_now} stood")
    return reversed_now


def _by_ref(batch: Sequence[Candidate]) -> dict[str, Candidate]:
    by_ref: dict[str, Candidate] = {}
    for position, candidate in enumerate(batch, start=1):
        by_ref[str(position)] = candidate
        by_ref[candidate.key.lower()] = candidate     # a model that echoes the key instead
    return by_ref


def _apply(batch: Sequence[Candidate], parsed: Any, plan: Mapping[str, Any] | None = None) -> int:
    """Write one batch's verdicts onto its candidates; return how many landed.

    A record the model did not answer for keeps `not_screened` and is told so. It is never defaulted
    to `exclude`: "nobody read it" and "someone read it and said no" are different facts, and the
    page shows them in different buckets (models.py:41). The guard runs on every verdict.
    """
    by_ref = _by_ref(batch)
    answered: set[str] = set()
    for entry in (parsed or {}).get("decisions", []) or []:
        if not isinstance(entry, Mapping):
            continue
        ref = str(entry.get("ref", "")).strip().strip(".)").lower()
        candidate = by_ref.get(ref)
        if candidate is None or candidate.key in answered:
            continue                                  # a ref for nobody, or a second answer
        decision, note = _normalise(entry.get("decision"))
        answers = _answers_of(entry)
        decision, guard_note = _guard(decision, answers, candidate, plan)
        note = " ".join(n for n in (note, guard_note) if n)
        title_only = not (candidate.abstract or "").strip()
        _write_verdict(candidate, decision=decision,
                       reason=_reason_for(entry.get("reason"), title_only=title_only, note=note),
                       title_only=title_only, answers=answers)
        answered.add(candidate.key)

    for candidate in batch:
        if candidate.key not in answered:
            _write_unscreened(candidate, NO_VERDICT_REASON)
    return len(answered)


def _apply_audit(batch: Sequence[Candidate], parsed: Any,
                 plan: Mapping[str, Any] | None = None) -> int:
    """The audit's answers: a changed verdict → `unknown` (reversed), a repeated `exclude` stands
    (confirmed), no answer → confirmed by default. Returns how many were reversed."""
    by_ref = _by_ref(batch)
    seen: set[str] = set()
    reversed_n = 0
    for entry in (parsed or {}).get("decisions", []) or []:
        if not isinstance(entry, Mapping):
            continue
        ref = str(entry.get("ref", "")).strip().strip(".)").lower()
        candidate = by_ref.get(ref)
        if candidate is None or candidate.key in seen:
            continue
        seen.add(candidate.key)
        decision, _note = _normalise(entry.get("decision"))
        answers = _answers_of(entry)
        decision, _guard_note = _guard(decision, answers, candidate, plan)
        candidate.screen_answers = {**candidate.screen_answers,
                                    "audit_q5": answers.get("q5", "unknown"),
                                    "audit_quote": answers.get("quote", "")}
        if decision != "exclude":
            reversed_n += 1
            candidate.screen_answers["audit"] = "reversed"
            reason = (f"{candidate.screen_reason} (asked again with the groups words in view, "
                      f"the screener answered: {str(entry.get('reason') or '').strip()}; "
                      f"sent to full text.)")
            _write_verdict(candidate, decision="unknown", reason=reason,
                           title_only=candidate.screened_on_title_only,
                           answers=candidate.screen_answers)
        else:
            candidate.screen_answers["audit"] = "confirmed"
    for candidate in batch:
        if candidate.key not in seen:
            candidate.screen_answers = {**candidate.screen_answers, "audit": "confirmed"}
    return reversed_n
