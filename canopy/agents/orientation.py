"""Orientation (spec §3.3(5), amendment G) — what does a LARGER raw value on this measure mean?

This is the question that decides the SIGN of every effect size built from a measure, and getting
it wrong inverts a result rather than blurring it. So it is answered once per (outcome, measure),
by two agents that must agree independently, from the measure's own definition and the paper's
words — never from what the measure is usually called. Code then applies `orient()`; nothing here
computes anything.

The paper's own statement of which group came out higher travels with the verdict
(`direction_stated_in_text`), so `canopy.verify.checks.sign_check` can compare it with the sign the
extracted numbers imply and flag `sign_mismatch` when they disagree.

Three ceiling rules live in `combine_orientation`, and all three are about who is allowed to decide:

* **C3 — the deterministic check is DISCARD-ONLY.** It may throw out a reader whose own stated
  direction contradicts the resolved raw means, or it may abstain. It may never *choose* a
  direction, never branch on `raw_value_semantics` (which encodes direction, not desirability, so a
  check keyed on it is circular), never read a ballot C12 has already called a non-reply, and never
  fire while which-series-is-which is itself disputed. A discard can only ever remove the reader
  that committed to a checkable claim, so what it removes is that reader's VOTE and nothing else:
  the readers that made no checkable claim may not settle a direction the discarded one
  contradicted, at any n, and a discard that leaves fewer than two agreeing readers abstains. A
  polarity every decided reader named, the discarded one included, is still recorded — the question
  a discard leaves behind is about the values, and the `error` it raises holds the cell.
* **C12 — a degenerate reply is `not_run`, not an answer.** It is re-issued once past the cache;
  only if the re-issue is also degenerate does the measure fall back, and then the ONE coherent
  ORIGINAL reader left may set the direction as a flagged single witness. This is the only route to
  a one-reader verdict: money never buys one.
* **P-A residue — an abstention is not a dissent.** A reader that answered "unknown" has not voted
  against anything; it just did not vote. A third read (different model family, off by default,
  budget-gated) may be bought only where C3 abstains; everything it settles goes through the
  majority branch — same raw scale, a real majority, flagged and capped — and never through the
  row where two readers agreed on their own.
"""
from __future__ import annotations

import re
from typing import Any, Sequence, get_args

from ..config import MODELS
from ..ingest.pdf import PaperRecord
from ..llm.client import REISSUE_CACHE_KEY, BudgetExceeded, LLMClient, degenerate_reply
from ..llm.context import text_block
from ..models import (DatasetSpec, Direction, OrientationRun, OrientationVerdict, OutcomeDef,
                      OutcomeSources, Protocol, RawValueSemantics)
from ..verify.checks import DISPUTED_MEANS_FLAGS, orientation_note, sign_check
from ..verify.vote import model_family
from . import render_prompt
from .verify_common import (SYSTEM, clip, enum_schema, enum_value, groups_prompt, measure_prompt,
                            outcome_prompt, prompt_fingerprint, whole_paper)

__all__ = ["orientation", "orientation_run", "combine_orientation", "ORIENTATION_SCHEMA",
           "PROMPT_VERSION", "PROMPT_FILES", "DEFAULT_MODELS", "TIEBREAK_MODEL",
           "TIEBREAK_EFFORT", "MEANS_CHECK_NOTE", "tiebreak_ballot", "TIEBREAK_PROMPT_FILES",
           "TIEBREAK_PROMPT_VERSION", "ORIENTATION_SOURCES"]

PROMPT_FILES = ("orientation",)
PROMPT_VERSION = f"orientation/1@{prompt_fingerprint(PROMPT_FILES)}"

#: C2's ballot asks a DIFFERENT question — the two earlier ballots and this cell's raw group means
#: are in front of it — so it has a prompt of its own and a version of its own. Deliberately NOT in
#: `PROMPT_FILES`: that fingerprint is the ordinary readers' `prompt_version`, every recorded
#: orientation fixture is keyed on it, and adding a file to it would re-key reads whose prompt has
#: not changed by one character. The version is recorded for PROVENANCE (the cache key does not
#: contain it), so a reader of `verify.json` can tell which question a ballot answered.
TIEBREAK_PROMPT_FILES = ("orientation_tiebreak",)
TIEBREAK_PROMPT_VERSION = f"orientation_tiebreak/1@{prompt_fingerprint(TIEBREAK_PROMPT_FILES)}"

#: HOW a direction was settled, for `OrientationVerdict.orientation_source` and the row that copies
#: it. Empty means nothing settled it and the measure is still a question.
ORIENTATION_SOURCES: tuple[str, ...] = ("agreed", "single_witness", "tiebreak_ballot", "human")

MAX_TOKENS = 4000
#: two agents, and they must differ — one strong reader and one that fails differently
DEFAULT_MODELS: tuple[str, str] = (MODELS["primary"], MODELS["secondary"])
_EFFORT = {MODELS["primary"]: "high"}
_DEFAULT_EFFORT = "medium"

#: P-A residue. The third read is the LAST step and it is **off by default**: `tiebreak=None`
#: reproduces HEAD exactly, no call, no cost. A caller that switches it on buys at most one extra
#: ballot per (outcome, measure), only where the free deterministic check already abstained, from a
#: model family that differs from at least one of the first two readers.
TIEBREAK_MODEL = MODELS["adjudicator_max"]
TIEBREAK_EFFORT = "xhigh"

#: How every verdict says whether the deterministic check could run at all — `ran`, `no_means` or
#: `disputed`. Without it an `agreed` verdict reads identically whether the filter ran and found
#: nothing, the means were missing, or which series is which was itself in dispute, and those are
#: three different claims about how well this direction was checked (fix round F11).
MEANS_CHECK_NOTE = "means check"

_HIGHER = {"higher": True, "lower": False, "unknown": None}

#: 5 leaf properties — one measure, one question
ORIENTATION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["raw_value_semantics", "higher_is_better", "direction_stated_in_text", "quotes",
                 "reason"],
    "properties": {
        "raw_value_semantics": enum_schema(get_args(RawValueSemantics)),
        "higher_is_better": enum_schema(list(_HIGHER)),
        "direction_stated_in_text": enum_schema(get_args(Direction)),
        "quotes": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}


def _quotes(raw: Any) -> list[str]:
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        text = clip(item, 400) if isinstance(item, str) else ""
        if text and text not in out:
            out.append(text)
    return out


def orientation_run(client: LLMClient, paper: PaperRecord, dataset: DatasetSpec | None,
                    outcome_sources: OutcomeSources, model: str, *,
                    protocol: Protocol | None = None, outcome: OutcomeDef | None = None,
                    pdf_file_id: str | None = None, effort: str | None = None,
                    cache_key_extra: str = "") -> OrientationRun:
    """One agent's answer about one measure's direction.

    `cache_key_extra` is C12's re-issue lever: without it the disk cache is content-addressed on
    the request, so asking the same model the same question again returns the same malformed reply
    and the re-issue is a no-op that looks like a confirmation.
    """
    document, betas = whole_paper(client, paper, pdf_file_id)
    content = [document, text_block(render_prompt(
        "orientation",
        OUTCOME=outcome_prompt(outcome_sources.outcome_key, protocol=protocol, dataset=None,
                               outcome=outcome),
        MEASURE=measure_prompt(outcome_sources),
        GROUPS=groups_prompt(dataset)))]

    result = client.structured(
        model=model, system=SYSTEM, schema=ORIENTATION_SCHEMA,
        effort=effort or _EFFORT.get(model, _DEFAULT_EFFORT), max_tokens=MAX_TOKENS, betas=betas,
        prompt_version=PROMPT_VERSION, cache_key_extra=cache_key_extra,
        cell_key=f"orientation:{outcome_sources.outcome_key}:"
                 f"{outcome_sources.measure_name or 'measure'}",
        messages=[{"role": "user", "content": content}])

    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    return OrientationRun(
        higher_is_better=_HIGHER[enum_value(parsed.get("higher_is_better"), list(_HIGHER),
                                             "unknown")],
        raw_value_semantics=enum_value(parsed.get("raw_value_semantics"),
                                        get_args(RawValueSemantics), "unknown"),
        direction_stated_in_text=enum_value(parsed.get("direction_stated_in_text"),
                                             get_args(Direction), "unknown"),
        quotes=_quotes(parsed.get("quotes")),
        reason=(parsed.get("reason") or "").strip(),
        model=model, prompt_version=PROMPT_VERSION, llm_call_id=result.call_id)


def _means_prompt(group_a_label: str, group_b_label: str, mean_a: float | None,
                  mean_b: float | None, n_a: int | None, n_b: int | None, unit: str) -> str:
    """The two groups' raw values, un-orientated, in the paper's own units.

    Un-orientated on purpose: `higher_is_better` is applied in `resolve_effect`, far downstream,
    and showing a reader a signed number would be showing it this function's guess at the answer
    it is being asked for.
    """
    lines = []
    for key, label, mean, n in (("A", group_a_label, mean_a, n_a),
                                ("B", group_b_label, mean_b, n_b)):
        lines.append(f"GROUP {key} ({label or 'unnamed group'}): "
                     f"{'not resolved' if mean is None else mean}"
                     f"{' ' + unit if unit and mean is not None else ''}"
                     f"{f', n = {n}' if n else ''}")
    return "\n".join(lines)


#: how much of an earlier reader's reason the tiebreak prompt shows. A reply that came back
#: garbled — `degenerate_reply`'s `stuttered_tail` — runs to a kilobyte and a half of mid-word
#: sentence tails, and all of it used to go verbatim into the prompt of the reader that has to
#: break the tie (whole-branch review, MAJOR 3). A witness C12 will not let vote must not
#: dominate the context of the vote that replaces it. Both limits, whichever ends first: the
#: opening is where a reader states its case, and a bad reply's first run-on sentence never ends.
_REASON_SENTENCES = 3
_REASON_CHARS = 600
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _reason_opening(text: str) -> str:
    """The first `_REASON_SENTENCES` sentences of a reason, or its first `_REASON_CHARS`
    characters, whichever comes first — marked with an ellipsis when anything was left out."""
    whole = re.sub(r"\s+", " ", str(text or "")).strip()
    head = clip(" ".join(_SENTENCE_END.split(whole)[:_REASON_SENTENCES]), _REASON_CHARS)
    if head != whole and not head.endswith("…"):
        head += " …"
    return head


def _prior_readers_prompt(runs: Sequence[OrientationRun]) -> str:
    """Every earlier ballot: who, what it voted, why (its opening), and the words it rested on."""
    if not runs:
        return "(no earlier ballot was recorded)"
    blocks = []
    for index, run in enumerate(runs, start=1):
        direction = {True: "higher", False: "lower"}.get(run.higher_is_better, "unknown")
        lines = [f"READER {index}: {run.model or 'unnamed reader'}",
                 f"  higher_is_better: {direction}",
                 f"  raw_value_semantics: {run.raw_value_semantics}",
                 f"  direction_stated_in_text: {run.direction_stated_in_text}",
                 f"  reason: {_reason_opening(run.reason) or '(none given)'}"]
        lines += [f"  quote: {clip(quote, 400)}" for quote in run.quotes[:4]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def tiebreak_ballot(client: LLMClient, paper: PaperRecord, dataset: DatasetSpec | None,
                    outcome_sources: OutcomeSources, *,
                    protocol: Protocol | None = None, outcome: OutcomeDef | None = None,
                    pdf_file_id: str | None = None,
                    mean_a: float | None = None, mean_b: float | None = None,
                    group_a_label: str = "", group_b_label: str = "",
                    n_a: int | None = None, n_b: int | None = None, unit: str = "",
                    prior_runs: Sequence[OrientationRun] = (),
                    model: str = TIEBREAK_MODEL,
                    effort: str = TIEBREAK_EFFORT) -> OrientationRun:
    """C2: ONE more ballot on a direction the two readers and the free check could not settle.

    It differs from `orientation_run` in what it is shown and in nothing else. Same schema, same
    parsing, same `OrientationRun` — so `combine_orientation` weighs it exactly like any other
    ballot and it can never settle a direction on its own. What it is shown is:

    * the outcome definition and the measure, as every reader gets them;
    * this cell's RAW group means, un-orientated, with their labels, sizes and unit — the numbers
      only exist after the vote, which is why this call happens in the verify loop rather than
      where the first two ballots are bought;
    * both earlier ballots verbatim, including each reader's `direction_stated_in_text`, so that a
      claim about which group came out higher can be checked against the values rather than
      re-remembered.

    What it is never shown is what its answer would DO — nothing about pooling, about the size of
    the effect, or about which answer keeps the row in the analysis. A reader told the consequence
    is being asked a different question, and the answer would no longer be about the paper.
    """
    document, betas = whole_paper(client, paper, pdf_file_id)
    content = [document, text_block(render_prompt(
        "orientation_tiebreak",
        OUTCOME=outcome_prompt(outcome_sources.outcome_key, protocol=protocol, dataset=None,
                               outcome=outcome),
        MEASURE=measure_prompt(outcome_sources),
        RAW_MEANS=_means_prompt(group_a_label, group_b_label, mean_a, mean_b, n_a, n_b, unit),
        PRIOR_READERS=_prior_readers_prompt(prior_runs)))]

    result = client.structured(
        model=model, system=SYSTEM, schema=ORIENTATION_SCHEMA, effort=effort,
        max_tokens=MAX_TOKENS, betas=betas, prompt_version=TIEBREAK_PROMPT_VERSION,
        cell_key=f"orientation_tiebreak:{outcome_sources.outcome_key}:"
                 f"{outcome_sources.measure_name or 'measure'}",
        messages=[{"role": "user", "content": content}])

    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    return OrientationRun(
        higher_is_better=_HIGHER[enum_value(parsed.get("higher_is_better"), list(_HIGHER),
                                             "unknown")],
        raw_value_semantics=enum_value(parsed.get("raw_value_semantics"),
                                        get_args(RawValueSemantics), "unknown"),
        direction_stated_in_text=enum_value(parsed.get("direction_stated_in_text"),
                                             get_args(Direction), "unknown"),
        quotes=_quotes(parsed.get("quotes")),
        reason=(parsed.get("reason") or "").strip(),
        model=model, prompt_version=TIEBREAK_PROMPT_VERSION, llm_call_id=result.call_id)


class _Ballot:
    """One reader's run, plus what the deterministic check decided about it."""

    __slots__ = ("run", "not_run", "discarded", "why")

    def __init__(self, run: OrientationRun) -> None:
        self.run = run
        # C12: this reply did not happen — because the detector says so, OR because the record
        # already does. The record can only ever ADD to this, never take it away, so a ballot a
        # live pass classified as a non-reply stays one on every replay and a detector tuned
        # afterwards cannot promote it back into a witness (F13). `models.py` says the field is
        # the authority; this is the line that makes that true.
        self.not_run: list[str] = degenerate_reply(run.reason) or (
            ["recorded_not_run"] if run.not_run else [])
        self.discarded = False                                   # C3 row 1
        self.why = ""

    @property
    def replied(self) -> bool:
        """This reader ANSWERED. A discarded ballot still answered: the filter takes away its vote
        on the direction, not the fact that it read the paper and said what it says."""
        return not self.not_run

    @property
    def counts(self) -> bool:
        """A ballot: a coherent reply that the discard filter did not throw out."""
        return self.replied and not self.discarded

    @property
    def decided(self) -> bool:
        """A ballot that actually named a direction. `None` is an abstention, not a dissent."""
        return self.counts and self.run.higher_is_better is not None


def _discard_check(ballots: list[_Ballot], mean_a: float | None, mean_b: float | None,
                   open_flags: Sequence[str]) -> tuple[str, str]:
    """C3 row 1 — the ONLY mechanical contradiction there is: a reader against the raw means.

    A reader that wrote `direction_stated_in_text` made a checkable claim about which group is
    numerically higher, and `sign_check` is exactly the comparison that tests it. Nothing else here
    may discard anybody: a reader's `raw_value_semantics` restates its own answer, so branching on
    it is circular, and no arithmetic can tell whether "more of the construct" is more adaptation.

    Three conditions have to hold before a discard is even attempted:

    * the means must exist — with nothing to contradict, there is no contradiction;
    * which series is which must not itself be in dispute. Under any of `DISPUTED_MEANS_FLAGS` the
      "resolved" means may be the other group's, and a discard run against swapped means throws
      out the reader that was RIGHT about the paper;
    * the ballot must be a reply. C12 has already ruled that a degenerate ballot did not happen,
      so the fields parsed out of it are not a claim anybody made: checking a stub's
      `direction_stated_in_text` against the means unpools a correct cell and puts a reader that
      never answered on the record as having contradicted the paper (fix round F4).

    Returns `(state, sentence)`: `ran`, `no_means` or `disputed`, and the words that say so. Every
    verdict records it, because "checked, and nothing was wrong" and "could not be checked" are
    not the same claim about a direction (F11).
    """
    if mean_a is None or mean_b is None:
        return "no_means", ("the raw means were not available, so no reader could be checked "
                            "against them")
    disputed = sorted(set(open_flags) & DISPUTED_MEANS_FLAGS)
    if disputed:
        return "disputed", (f"which series is which is disputed ({', '.join(disputed)}), so no "
                            f"reader was checked against means that may be the other group's")
    for ballot in ballots:
        if not ballot.replied:
            continue
        if sign_check(ballot.run.direction_stated_in_text, mean_a, mean_b) is not None:
            ballot.discarded = True
            ballot.why = (f"{ballot.run.model} states "
                          f"{ballot.run.direction_stated_in_text.replace('_', ' ')} on this "
                          f"measure, but the resolved raw means say A = {mean_a} and B = {mean_b}")
    return "ran", (f"every reader that stated a direction was checked against the resolved raw "
                   f"means A = {mean_a} and B = {mean_b}")


def _dissenting_not_run(ballots: Sequence["_Ballot"], survivor: "_Ballot") -> list["_Ballot"]:
    """`not_run` ballots that named the OPPOSITE direction to the one reader left standing.

    C12's detector reads the justification PROSE, and it was deliberately allowed to over-fire so
    that a garbled reply is re-issued rather than trusted. What it must never do is delete an
    ANSWER: `higher_is_better` is a parsed enum, not prose, and a reader that named the other
    direction disagreed about the sign whatever its sentences looked like. Row 5 hands a measure
    to a single witness, which pools — so it may not be reached over one of these (whole-diff M2).
    """
    direction = survivor.run.higher_is_better
    return [b for b in ballots
            if b.not_run and b.run.higher_is_better is not None
            and bool(b.run.higher_is_better) != bool(direction)]


def combine_orientation(runs: Sequence[OrientationRun], outcome_key: str,
                        measure_name: str = "", *, mean_a: float | None = None,
                        mean_b: float | None = None, open_flags: Sequence[str] = (),
                        third_read: bool = False, dataset_id: str = "") -> OrientationVerdict:
    """Independent answers into one verdict — or, far more often than before, into a question.

    The whole of C3's decision table is here, and every row of it either keeps today's behaviour or
    ABSTAINS. Nothing in this function can invent a direction: `higher_is_better` is only ever
    copied from a reader that survived the discard filter and named one.

    The rows are tried in this order, and the order is itself a rule:

    1. the discard filter, which costs nothing, removes witnesses only, and records whether it
       could run at all;
    2. **a bought third read**, if there was one — it settles by majority, on one shared raw scale,
       flagged and capped. It is asked FIRST so that a verdict money paid for can never come out of
       the row below looking like two readers agreeing on their own;
    3. two or more surviving readers naming the same direction, which no discarded reader named the
       other way;
    4. C12's single witness: one ORIGINAL reader decided and its partner's reply did not happen;
    5. everything else abstains, with the reason named.

    `mean_a`/`mean_b` are the RESOLVED raw group means (not orientated) and `open_flags` the check
    codes already open on the cell. Both default to "unknown", which disables the discard filter —
    the safe default, since the filter's only power is to remove a witness.
    """
    ballots = [_Ballot(run) for run in runs]
    # C12 on the record rather than in the prose: each run keeps its own classification, so a
    # replayed `verify.json` says which replies were not replies without re-running the detector.
    # Copies, never mutation — `combine_orientation` is called twice on the same runs (once when
    # the reads are bought, once in `_verify_cell` with the resolved means) and must stay pure.
    scored = [b.run.model_copy(update={"not_run": bool(b.not_run)}) for b in ballots]
    for ballot, run in zip(ballots, scored):
        ballot.run = run
    verdict = OrientationVerdict(outcome_key=outcome_key, measure_name=measure_name,
                                 dataset_id=dataset_id, third_read=third_read, runs=scored,
                                 llm_call_ids=[r.llm_call_id for r in runs if r.llm_call_id])
    quotes: list[str] = []
    for run in runs:
        quotes += [q for q in run.quotes if q not in quotes]
    verdict.quotes = quotes
    verdict.reason = " | ".join(f"{r.model}: {r.reason}" for r in runs if r.reason)

    notes: list[str] = []
    for ballot in ballots:                                       # C12: say which replies were lost
        if ballot.not_run:
            notes.append(f"{ballot.run.model}'s reply was not a reply "
                         f"({', '.join(ballot.not_run)}) and is recorded as not_run")
    state, said = _discard_check(ballots, mean_a, mean_b, open_flags)
    discarded = [b for b in ballots if b.discarded]
    for ballot in ballots:            # on the RECORD, per reader, like `not_run` above (L3)
        ballot.run.discarded = ballot.discarded
    for ballot in discarded:
        notes.append(orientation_note("orientation_reader_contradicts_values", ballot.why))
    notes.append(f"{MEANS_CHECK_NOTE}: {state} — {said}")

    live = [b for b in ballots if b.counts]
    decided = [b for b in live if b.decided]
    answers = {b.run.higher_is_better for b in decided}
    one_answer = next(iter(answers)) if len(answers) == 1 else None
    #: what the discarded readers said about the POLARITY. A discard says a reader's claim about
    #: which group is numerically higher is contradicted by the numbers; it never establishes that
    #: the reader was wrong about which way the measure POINTS, and those are two different
    #: sentences read in two different places. So the direction it named may still be recorded
    #: when every other decided reader named it too — and may never be overruled by readers that
    #: made no checkable claim at all, because the filter can only ever remove the reader that
    #: committed to evidence, and letting the silent ones settle what it contradicted prefers
    #: silence to evidence at every n (controller ruling, fix round F1).
    contradicted = {b.run.higher_is_better for b in discarded
                    if b.run.higher_is_better is not None}

    def settled_by(value: bool) -> bool:
        """May this direction be recorded, given what the discarded readers said?"""
        return not contradicted - {value}

    def overruled() -> str:
        more = "more" if one_answer else "less"
        return (f"the readers left after the check agree that a larger raw value means {more} of "
                f"the construct, but " + ", ".join(sorted(b.run.model for b in discarded))
                + " — whose stated direction the resolved raw means contradict — read this "
                  "measure the other way, and readers that made no checkable claim do not settle "
                  "a direction the one reader the numbers could check contradicted")

    if third_read and len(decided) >= 2:
        # P-A: a bought third read settles by MAJORITY, and only if all three readers were talking
        # about the same kind of number — a majority built out of two different readings of
        # the raw scale is a coincidence, not a corroboration. This gate is BEFORE row 4 (fix
        # round F3): a verdict that exists only because money was spent must never come out
        # looking like two readers agreeing independently, whatever the tally, so every third-read
        # verdict goes through the shared-scale condition, the flag and the cap.
        tally = {value: [b for b in decided if b.run.higher_is_better is value]
                 for value in answers}
        top = max(tally.values(), key=len)
        # …asked of the MAJORITY, not of every decided reader (whole-branch review, MINOR 4).
        # Scoping it to `decided` let the outvoted reader break the majority by disagreeing about
        # the scale, which made the ballot a no-op in the case it was bought for: Bock's
        # aftereffect, where opus and fable both read `signed_direction` and both said False, and
        # sonnet's `higher_more_error` alone kept the cell with a human. What has to be one number
        # is the number the agreeing readers agreed about. `top` is never a single reader — the
        # even-split gate below runs first — so "never settle alone" is untouched.
        shared = {b.run.raw_value_semantics for b in top}
        if len(top) * 2 <= len(decided):
            notes.append("the readers split evenly on the direction of this measure, and a tie is "
                         "not a majority")
        elif len(shared) != 1:
            notes.append("the readers do not agree on what the raw scale IS "
                         f"({', '.join(sorted(shared))}), so their majority on the direction is "
                         "not evidence about the same number")
        elif not settled_by(top[0].run.higher_is_better):
            notes.append(overruled())
        else:
            verdict.higher_is_better = top[0].run.higher_is_better
            verdict.needs_human = False
            verdict.orientation_source = "tiebreak_ballot"
            notes.append(orientation_note(
                "orientation_by_majority",
                f"the direction was settled by a majority {len(top)} of {len(ballots)} after a "
                f"third read, not by two readers agreeing independently "
                f"({len(top)}-{len(decided) - len(top)} among the readers that named one)"))
    elif len(decided) >= 2 and one_answer is not None and settled_by(one_answer):
        # row 4 — two or more readers named the same direction and nothing contradicted them. A
        # discarded reader that named the SAME direction does not stop this: the question it
        # leaves behind is about the values, not about the polarity, and the `error` its own flag
        # carries holds the cell while that question is answered.
        verdict.higher_is_better = one_answer
        verdict.agreed = True
        verdict.needs_human = False
        verdict.orientation_source = "agreed"
    elif len(decided) >= 2 and one_answer is not None:
        notes.append(overruled())
    elif (len(decided) == 1 and any(b.not_run for b in ballots)
          and not discarded and not third_read and not _dissenting_not_run(ballots, decided[0])):
        # row 5 — reachable ONLY from C12's not_run: one ORIGINAL reader decided and its partner's
        # reply did not happen. Never from a discard, because the filter can only ever remove the
        # reader that made a checkable claim, so handing the measure to the reader that stayed
        # silent would systematically prefer silence to evidence; and never from a bought read,
        # because `orientation_single_witness` caps a cell at `accept_with_note`, which is a
        # POOLING bucket — one model's single ballot behind the sign of a pooled effect is exactly
        # what "a single reader never decides direction" forbids (fix round F2).
        #
        # …and never over a DISSENT. `degenerate_reply` was deliberately tuned to over-fire — a
        # doubled word, a stray `\theta`, one unbalanced brace — and it is applied to the prose,
        # not to the answer. A ballot that named the OPPOSITE direction is a reader disagreeing
        # about the sign whatever its prose looks like, so treating it as a non-reply handed the
        # sign to the other reader alone at a pooling bucket (whole-diff M2).
        verdict.higher_is_better = decided[0].run.higher_is_better
        verdict.needs_human = False
        verdict.orientation_source = "single_witness"
        notes.append(orientation_note(
            "orientation_single_witness",
            f"only {decided[0].run.model} returned a real reply about this measure, so one reader "
            f"set its direction where two normally must agree"))
    elif len(decided) == 1 and _dissenting_not_run(ballots, decided[0]) and not third_read:
        notes.append(
            "the other reader's reply was recorded as not a reply, but it named the OPPOSITE "
            "direction (" + ", ".join(sorted(b.run.model for b in
                                             _dissenting_not_run(ballots, decided[0])))
            + "), and a garbled dissent is still a dissent about the sign: one reader never "
              "decides the direction of a measure over another reader's disagreement")
    elif len(decided) == 1 and third_read:
        # the bought read did not produce a majority — one ballot is one ballot, whoever paid for
        # it and whichever reader it belongs to (fix round F2)
        notes.append(f"only {decided[0].run.model} named a direction on this measure and the "
                     f"third read did not change that; one reader never decides the direction of "
                     f"a measure on its own, and what a third read settles it settles by majority")
    elif not live:
        notes.append("no reader survived on this measure" if discarded else
                     "no reader returned a usable reply about this measure")
    elif len(live) == 1 and len(ballots) == 1:
        notes.append("only one agent ruled on this measure; the direction of a measure needs two "
                     "independent answers")
    elif len(live) == 1:
        notes.append("only one reader is left on this measure, and one reader never decides the "
                     "direction of a measure on its own")
    elif not decided:
        notes.append("neither agent could tell what a larger raw value means")
    elif len(decided) == 1:
        # P-A residue (a): an abstention is not a dissent. Nobody contradicted this reader — the
        # others simply did not vote, and one reader never decides the direction of a measure.
        notes.append(f"only {decided[0].run.model} named a direction and the other reader "
                     f"abstained; one reader never decides the direction of a measure on its own")
    else:
        notes.append("the agents disagree about the direction of this measure: "
                     + ", ".join(f"{b.run.model}={b.run.higher_is_better}" for b in decided))

    # Everything below is a summary of what the readers SAID, so it is built from the ballots that
    # were replies — discarded or not (a discard removes a vote, not a sentence) — and never from
    # the ones C12 says did not happen (fix round F7, F10).
    replied = [b for b in ballots if b.replied]

    # C3 rule 5: a disagreement about what the raw scale is, is information a reviewer needs. The
    # summary field cannot hold two answers, so it says "unknown" — but it no longer eats them,
    # and a reply that did not happen no longer gets a vote in what the scale is.
    semantics = {b.run.raw_value_semantics for b in replied}
    verdict.raw_value_semantics = semantics.pop() if len(semantics) == 1 else "unknown"
    # one entry per READER, so a re-issued ballot does not overwrite the coherent one it replaced
    for ballot in ballots:
        model = ballot.run.model or "unnamed"
        if model not in verdict.raw_value_semantics_by_model or ballot.replied:
            verdict.raw_value_semantics_by_model[model] = str(ballot.run.raw_value_semantics)
    if verdict.raw_value_semantics == "unknown" and len(replied) > 1:
        notes.append("raw_value_semantics per reader: "
                     + ", ".join(f"{b.run.model}={b.run.raw_value_semantics}" for b in replied))

    # C3 rule 6: two readers reading the paper's own sentence in opposite directions is a
    # contradiction about the paper, not a note — and today it silently disables `sign_check` too.
    # On the production path the means are always there, so the reader whose stated direction they
    # contradict is discarded FIRST; taking these from the survivors alone made the conflict
    # disappear from the record exactly where it matters most (fix round F7).
    directions = {b.run.direction_stated_in_text for b in replied
                  if b.run.direction_stated_in_text != "unknown"}
    if len(directions) == 1:
        verdict.direction_stated_in_text = directions.pop()
    elif directions:
        notes.append(orientation_note(
            "orientation_direction_conflict",
            f"the agents read the paper's stated direction differently ({sorted(directions)}), so "
            f"nothing is left for the sign check to compare the extracted numbers with"))
    verdict.notes = "; ".join(notes)
    return verdict


def orientation(client: LLMClient, paper: PaperRecord, dataset: DatasetSpec | None,
                outcome_sources: OutcomeSources,
                models: str | Sequence[str] = DEFAULT_MODELS, *,
                protocol: Protocol | None = None, outcome: OutcomeDef | None = None,
                pdf_file_id: str | None = None, mean_a: float | None = None,
                mean_b: float | None = None, open_flags: Sequence[str] = (),
                tiebreak: str | None = None) -> OrientationVerdict:
    """Decide the direction of ONE measure with two independent agents (spec §3.3(5)).

    Runs once per (outcome, measure), not once per value: the answer is a property of the measure.
    Three things can happen beyond the two ordinary reads, in this order and no other:

    1. **C12** — a reader whose justification is degenerate is asked again, exactly once, with a
       cache key that cannot return the same malformed reply. Free reads are retried before
       anything is inferred from their absence.
    2. **C3** — the deterministic discard filter runs. It costs nothing, so it runs before any
       paid step, and it can only remove a witness or abstain.
    3. **P-A** — only where C3 abstained, and only when the caller passed `tiebreak`, one extra
       read is bought. `tiebreak=None` (the default) means this never happens.

    Disagreement, two "unknown"s, a discard, or a tie leaves `needs_human` set and
    `higher_is_better` None, and Task 9 refuses to sign an effect size without it.
    """
    names = [models] if isinstance(models, str) else list(models)
    if not names:
        raise ValueError("orientation needs at least one model")

    def read(model: str, *, effort: str | None = None, cache_key_extra: str = "") -> OrientationRun:
        return orientation_run(client, paper, dataset, outcome_sources, model, protocol=protocol,
                               outcome=outcome, pdf_file_id=pdf_file_id, effort=effort,
                               cache_key_extra=cache_key_extra)

    def ballots_from(model: str, *, effort: str | None = None) -> list[OrientationRun]:
        """One reader's ballots: its reply, and — C12 rule 2 — EXACTLY one re-issue past the cache
        if that reply was not a reply. A false positive costs this one call; treating a stub as a
        witness costs a pooled cell.

        Both raw replies stay on the record when the re-issue is degenerate too, so `verify.json`
        shows what was asked and what came back. A re-issue that WORKED replaces the stub instead
        of joining it: a `not_run` ballot sitting beside a coherent one would open C12's
        single-witness row on a measure that has two real readers. Every read goes through here,
        the bought third one included — it used to have its first reply overwritten and its
        re-issue never re-checked, so a doubly-degenerate third read was appended as a silent
        witness (fix round F12).
        """
        first = read(model, effort=effort)
        if not degenerate_reply(first.reason):
            return [first]
        again = read(model, effort=effort, cache_key_extra=REISSUE_CACHE_KEY)
        return [first, again] if degenerate_reply(again.reason) else [again]

    runs: list[OrientationRun] = []
    for name in names:
        runs += ballots_from(name)

    dataset_id = dataset.dataset_id if dataset is not None else ""
    verdict = combine_orientation(runs, outcome_sources.outcome_key,
                                  outcome_sources.measure_name, mean_a=mean_a, mean_b=mean_b,
                                  open_flags=open_flags, dataset_id=dataset_id)
    if not verdict.needs_human or not tiebreak:
        return verdict

    # P-A residue: the third read. Its ballot is worth nothing unless it fails differently from
    # what came before, so it must not be from the same family as BOTH of the first two readers.
    families = {model_family(r.model) for r in runs}
    if families == {model_family(tiebreak)}:
        verdict.notes = "; ".join(filter(None, [
            verdict.notes, f"no third read was bought: {tiebreak} is the same model family as "
                           f"every reader that already answered"]))
        return verdict
    try:
        third = ballots_from(tiebreak, effort=TIEBREAK_EFFORT)
    except BudgetExceeded as exc:
        verdict.notes = "; ".join(filter(None, [
            verdict.notes, f"no third read was bought: {exc}"]))
        return verdict
    return combine_orientation(runs + third, outcome_sources.outcome_key,
                               outcome_sources.measure_name, mean_a=mean_a, mean_b=mean_b,
                               open_flags=open_flags, third_read=True, dataset_id=dataset_id)
