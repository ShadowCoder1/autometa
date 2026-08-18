"""Turn every held cell into a question a person can answer from a screenshot.

The review queue says *that* a cell is held and lists its candidates. A reviewer does not want a
list; they want to be shown the picture and asked the one thing the tool could not settle:
"which of these two numbers is the older group's bar?", "is this the left or the right axis?",
"what do the error bars show?". Every question carries the screenshot the tool itself read, the
answers it is choosing between, and the override its answer becomes — so answering it is a
recorded decision, not a note in the margin.

Nothing here calls a model. It reads the run's own stage files and writes `questions.json` and
`questions.md` beside them; the server serves the same list and accepts answers.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from ..pipeline.overrides import (MAP_KINDS, ROW_REFUSALS, codes_cleared_by_value, consumed_seqs)
from ..pipeline.rows import converted_route
from ..pipeline.state import paper_dir, read_json

__all__ = ["Question", "questions_for_run", "write_questions", "answer_to_override",
           "QUESTION_KINDS"]

#: what a question is about; the UI groups and phrases by kind
QUESTION_KINDS: tuple[str, ...] = (
    "which_value",        # candidates disagree — pick the right number, or "none of these"
    "confirm_value",      # one uncorroborated number — is it right?
    "which_axis",         # the readers and the ladder disagree about which axis the value is on
    "which_series",       # the marker/series for a group is disputed or may be swapped
    "error_bar_type",     # nothing settled what the error bars show
    "orientation",        # nothing settled which direction of this measure is more adaptation
    "group_mapping",      # which printed group is A and which is B
    "verifier_refuted",   # a verifier says the value is wrong; the tool could not settle it
    "no_value",           # nothing usable was found — where is it, if anywhere?
    "converted_statistic",  # neither group's numbers are printed; the row converts a t/F/p/d
    "quote_not_found",    # the quote this number rests on is not printed in the paper
    "number_unusable",    # a number the conversion needs is missing or cannot be right (n, SD)
    "dispersion_doubt",   # the spread the row was divided by cannot be the one the authors used
    "needs_group_values",  # the row rests on a printed statistic whose estimand is unestablished
    "reader_contradicts_values",   # a reader's stated direction contradicts this cell's own means
    "other",              # a held cell that fits none of the above: show the reason
    # ---- decided at the MAP stage, before anything was read: these block an extraction rather
    # than hold a value, so they never appear in the review queue (C6, C7)
    "include_dataset",    # only one mapping agent proposed this dataset — is it in the review?
    "which_measure",      # one outcome, two measures, and the window fits both
)

#: flag codes → the question they raise. NOT "first match wins by position": `_kind` prefers the
#: code that is actually holding the cell (an error, a contradiction or a cap) over one that only
#: cost the score a few points — see `_kind`.
#:
#: §C4: "A hold reason no question kind can express is a bug in the review layer." So EVERY
#: `error`-severity code in `canopy.verify.checks.CHECK_SEVERITY` and every `CONTRADICTING_FLAGS`
#: warn appears here — `tests/test_questions.py` asserts the subset, because a forcing finding with
#: no entry falls through to `confirm_value`, whose answer is a note that releases the cell.
#: Within a severity the order is "how badly is the cell's identity in doubt": is this the right
#: comparison at all, then the right series/ladder, then the right number, then the right
#: bookkeeping.
_FLAG_TO_KIND: tuple[tuple[str, str], ...] = (
    ("orientation_reader_contradicts_values", "reader_contradicts_values"),
    ("quote_row_only", "group_mapping"),
    ("group_label_swapped", "group_mapping"),
    ("series_transposed", "which_series"),
    ("series_identity_conflict", "which_series"),
    ("axis_conflict", "which_axis"),
    ("calibration_disputed", "which_axis"),
    ("calibration_refuted", "which_axis"),
    ("value_outside_axis", "which_value"),
    ("sign_mismatch", "which_value"),
    ("quote_not_grounded", "quote_not_found"),
    ("series_marker_mismatch", "which_series"),
    ("dispersion_type_from_legend", "error_bar_type"),
    ("figure_error_bar_unknown", "error_bar_type"),
    ("dispersion_type_conflict", "error_bar_type"),
    ("df_missing", "needs_group_values"),
    ("df_shortfall_unexplained", "needs_group_values"),
    ("test_stat_missing_df", "needs_group_values"),
    ("n_not_integer", "number_unusable"),
    ("n_too_small", "number_unusable"),
    ("sd_nonpositive", "number_unusable"),
    ("implausible_dispersion", "dispersion_doubt"),
    # last, deliberately: these codes can be on a cell whose direction *was* settled (C3 records
    # a single-witness or majority orientation with its own flag). When the direction is genuinely
    # unresolved `_kind` has already returned `orientation` above, before any of this list is read.
    # (`orientation_unresolved` and `multi_group_closest_to_definition` used to sit here; both are
    # `EffectSizeRecord.flags` strings, never `CheckFlag` codes, so nothing this reads could ever
    # carry them. `_ORIENTATION_UNRESOLVED` still names the first, because that set is matched
    # against the record's own flags rather than against a verdict's.)
    ("orientation_direction_conflict", "orientation"),
    ("orientation_unknown", "orientation"),
)

_MAX_OPTIONS = 6

#: a question is OPEN, ANSWERED, or answered-and-waiting: the decision is on the record and its
#: consequence has not happened, because the thing it decides is a model call nobody has bought
#: yet (a map answer decides an extraction; a re-extraction decides a read). Calling that
#: "answered" is the same overclaim C4 took out of the answers themselves — a green tick on a cell
#: nothing has changed — so it gets its own state, and the reason travels with it.
PENDING_RERUN = "pending_rerun"


def _pending_seqs(run: Path) -> dict[int, str]:
    """`{seq: why}` for every answer the last re-pool could not act on — its own record of it."""
    summary = _json_if_present(run / "overrides_applied.json") or {}
    return {entry["seq"]: str(entry.get("why") or "")
            for entry in (summary.get("pending") or [])
            if isinstance(entry, dict) and isinstance(entry.get("seq"), int)}


def _needs_a_rerun(override: Mapping[str, Any]) -> str:
    """Why this answer waits for `--resume`, or "" when re-pooling can act on it alone.

    The rule, not a copy of the strings: the two constants live in `pipeline.overrides` beside the
    branches that produce them, so the page and the log can never drift apart about what waits.
    """
    from ..pipeline.overrides import MAP_PENDING, RE_EXTRACT_PENDING

    kind = str(override.get("kind") or "")
    if kind == "re_extract":
        return RE_EXTRACT_PENDING
    if kind == "which_measure":
        return MAP_PENDING
    if kind == "include_dataset" and override.get("decision") != "exclude":
        return MAP_PENDING
    return ""


def _answer_status(already: Sequence[Mapping[str, Any]], pending: Mapping[int, str],
                   consumed: Collection[int] = ()) -> tuple[str, str]:
    """`(status, why)` for a question, from its latest answer. The run's own `pending` record wins
    over the rule, because it is what actually happened the last time the log was applied.

    `consumed` is what the pipeline says it has since acted on (`consumed_override_seqs` in the
    stage files): an answer that has been bought is not waiting for anything, and without that the
    third state was permanent — "not applied yet" for ever, on a decision the resume had applied.
    """
    if not already:
        return "open", ""
    last = already[-1]
    seq = last.get("seq")
    if isinstance(seq, int) and seq in consumed:
        return "answered", ""
    if isinstance(seq, int) and seq in pending:
        return PENDING_RERUN, pending[seq]
    why = _needs_a_rerun(last)
    return (PENDING_RERUN, why) if why else ("answered", "")


class Question(dict):
    """A plain dict with a stable shape; subclassed only so the intent is visible in signatures."""


# ----------------------------------------------------------------------------- building
def questions_for_run(run_dir: str | Path) -> list[Question]:
    """Every held cell of a finished run, as questions, worst first (biggest |Δ pooled| on top)."""
    run = Path(run_dir)
    manifest = _json_if_present(run / "manifest.json") or {}
    queue = list(manifest.get("human_review_queue") or [])
    if not queue:                       # a run that held nothing writes no queue file at all
        queue = list(_json_if_present(run / "human_review_queue.json") or [])
    overrides = _overrides(run)
    pending = _pending_seqs(run)
    consumed = consumed_seqs(run)
    rows = _rows_by_cell(run)
    provenance = _json_if_present(run / "provenance" / "provenance.json") or {}
    out: list[Question] = []
    for entry in queue:
        paper_id = str(entry.get("paper_id") or "")
        dataset_id = str(entry.get("dataset_id") or "")
        outcome_key = str(entry.get("outcome_key") or "")
        group = entry.get("group") if entry.get("group") in ("A", "B") else None
        verdict = _verdict(run, paper_id, dataset_id, outcome_key, group)
        candidates = _candidates(run, paper_id, dataset_id, outcome_key, group)
        study, dataset = _dataset(run, paper_id, dataset_id)
        already = [o for o in overrides if o.get("dataset_id") == dataset_id
                   and o.get("outcome_key") in ("", outcome_key)
                   # an override that names a group answers THAT group and no other: a hint given
                   # for group A is not an answer about group B. The kinds that carry no group
                   # (a direction, a map decision, a whole-cell exclusion) match both, which is
                   # what `group is None` already says — reading "not a value override" as
                   # "group-less" made one decision tick off two cells.
                   and o.get("group") in (None, group)
                   # a decision taken at the MAP stage is not an answer to a cell that was READ:
                   # "this dataset belongs in the review" says nothing about what its error bars
                   # are. The one exception is an exclusion, which takes the cell out entirely.
                   and (o.get("kind") not in MAP_KINDS or o.get("decision") == "exclude")]
        # what the log has already settled is not still to be asked: the cell moves on to whatever
        # else is holding it, instead of being asked the same question again. The stage files a
        # re-pool reads are never rewritten, so the log is the only record of it — and the codes a
        # value answer retires come from `codes_cleared_by_value`, the same rule the re-pool
        # applies, so the page and the analysis can never disagree about what is still open.
        settled = _answered_orientation(overrides, entry, dataset, outcome_key)
        retired = _codes_answered(already)
        # what the log has already overruled: a refutation or an adjudication a reviewer has
        # addressed on the record is not a reason to ask them again, exactly as a recorded
        # direction is not. Without this the refutation question is asked for ever, because the
        # verify stage file a re-pool reads still says "refuted" and always will.
        overruled = {str(name) for o in already for name in (o.get("overrules") or [])}
        answered_value = any(o.get("kind") == "value" for o in already)
        if verdict and (settled is not None or retired):
            flags = [f for f in verdict.get("flags") or []
                     if str(f.get("code") or "") not in retired
                     and (settled is None
                          or str(f.get("code") or "") not in _ORIENTATION_UNRESOLVED)]
            verdict = {**verdict, "flags": flags}
            if settled is not None:
                verdict["higher_is_better"] = settled
        out.append(_question(entry, verdict, candidates, study, dataset, provenance,
                             run, already, pending, answered_value, consumed, overruled,
                             rows.get((dataset_id, outcome_key)) or {}))
    out.extend(_excluded_questions(run, overrides, out))
    out.extend(_map_questions(run, overrides, pending, consumed))
    out.sort(key=lambda q: (q.get("answered", False),
                            -(q.get("impact") if isinstance(q.get("impact"), (int, float))
                              else -1.0)))
    for i, q in enumerate(out, 1):
        q["number"] = i
    return out


def _answered_orientation(overrides: Sequence[Mapping[str, Any]], entry: Mapping[str, Any],
                         dataset: Mapping[str, Any], outcome_key: str) -> bool | None:
    """The direction a reviewer has already recorded for this cell's measure, if any."""
    measure = _measure(dataset, outcome_key)
    dataset_id = str(entry.get("dataset_id") or "")
    paper_id = str(entry.get("paper_id") or "")
    for override in overrides:
        if override.get("kind") != "orientation" or override.get("outcome_key") != outcome_key:
            continue
        if not isinstance(override.get("higher_is_better"), bool):
            continue
        named = str(override.get("measure_name") or "")
        if override.get("dataset_id") == dataset_id or (
                paper_id and override.get("paper_id") == paper_id
                and (not named or _same_text(named, measure))):
            return bool(override["higher_is_better"])
    return None


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def _codes_answered(already: Sequence[Mapping[str, Any]]) -> set[str]:
    """The check codes this cell's recorded answers have retired.

    One rule, written once in `pipeline.overrides.codes_cleared_by_value` and applied both when the
    analysis is re-pooled and when the page decides what to ask next. Without the second half, a
    cell whose axis question has been answered is asked the same question for ever, because the
    verify stage file it is read from is never rewritten.
    """
    codes: set[str] = set()
    for override in already:
        if override.get("kind") == "value":
            codes |= codes_cleared_by_value(override)
        elif override.get("kind") == "mark_reviewed":
            codes |= {str(code) for code in override.get("clears") or []}
    return codes


def _with_ballots(why: str, run: Path, paper_id: str, outcome_key: str, measure: str) -> str:
    """Both readers' answers, verbatim, under a direction question (ADVERSARIAL Round 2, C3).

    "Nobody established the direction" is not something a person can act on. What they can act on
    is what the two readers actually said and which of them read the paper right — so the ballots
    are quoted here, each with the direction it voted for. This reads the decision the verify
    stage recorded; it never re-derives one.
    """
    decision = _orientation_decision(run, paper_id, outcome_key, measure)
    lines: list[str] = []
    for ballot in decision.get("runs") or []:
        if ballot.get("not_run"):
            continue
        direction = ballot.get("higher_is_better")
        said = ("a larger value is MORE of what this review scores" if direction is True
                else "a larger value is LESS of what this review scores" if direction is False
                else "could not say which way round it goes")
        quotes = "; ".join(f"“{_short(q, 200)}”" for q in (ballot.get("quotes") or [])[:2])
        lines.append(f"{ballot.get('model') or 'a reader'} — {said}: "
                     f"{_short(ballot.get('reason'), 700)}"
                     + (f" Quoting: {quotes}" if quotes else ""))
    if not lines:
        return why
    return why + " || The two readers' ballots, which is what this question is decided from: " \
        + " || ".join(lines)


def _orientation_decision(run: Path, paper_id: str, outcome_key: str,
                          measure: str) -> dict[str, Any]:
    """The orientation record for this cell's measure, as the verify stage wrote it."""
    decisions = _stage(run, paper_id, "verify").get("orientation") or {}
    if not isinstance(decisions, dict):
        return {}
    exact = decisions.get(f"{outcome_key}|{measure}")
    if isinstance(exact, dict):
        return exact
    for key, value in decisions.items():
        if isinstance(value, dict) and str(key).split("|", 1)[0] == outcome_key \
                and (not measure or _same_text(str(value.get("measure_name") or ""), measure)):
            return value
    return {}


def _question(entry: Mapping[str, Any], verdict: Mapping[str, Any],
              candidates: Sequence[Mapping[str, Any]],
              study: Mapping[str, Any], dataset: Mapping[str, Any],
              provenance: Mapping[str, Any], run: Path,
              already: Sequence[Mapping[str, Any]],
              pending: Mapping[int, str], answered_value: bool = False,
              consumed: Collection[int] = (),
              overruled: Collection[str] = (),
              row: Mapping[str, Any] | None = None) -> Question:
    group = entry.get("group") if entry.get("group") in ("A", "B") else None
    outcome_key = str(entry.get("outcome_key") or "")
    # a cell is held by what its ROW carries too: C9's screen on the resolved |d| is a refusal no
    # cell can carry, and a cell queued for it has no flag of its own to ask about.
    row = dict(row or {})
    # only the row's REFUSALS: its other flags are echoes of findings the cells already carry, and
    # counting them twice would re-ask a question the cell has answered.
    on_the_row = [str(code) for code in row.get("flags") or [] if str(code) in ROW_REFUSALS]
    flags = [str(f.get("code") or "") for f in (verdict.get("flags") or [])] + on_the_row
    holding = _holding_codes(verdict) | set(on_the_row)
    valued = [c for c in candidates if c.get("mean") is not None]
    # the whole cell, not this group's share of it: a printed t or d belongs to the CONTRAST, so
    # the extractor writes it with no group and the group filter above drops it.
    whole_cell = _candidates(run, str(entry.get("paper_id") or ""),
                             str(entry.get("dataset_id") or ""), outcome_key, None)
    statistic = _statistic_of(whole_cell, str(row.get("route") or ""))
    kind = _kind(verdict, flags, valued, holding, answered_value, overruled, row, statistic)
    measure = _measure(dataset, outcome_key)
    # an answer answers the question it was given to, and no other. A cell is asked one thing at
    # a time: when the answer retires that blocker the cell moves on to the next one, and the new
    # question is OPEN — not an "answered" tick on something nobody has settled. Answers from the
    # page name their question; a decision taken elsewhere (the manual form, an older log) is
    # matched by what this question writes, which is the best the record can say.
    question_id = "|".join([str(entry.get("dataset_id") or ""), outcome_key, group or "", kind])
    writes = _answer_kind(kind)

    def answers_this(override: Mapping[str, Any]) -> bool:
        if override.get("kind") == "exclude_dataset":
            return True                     # the cell has left the analysis: nothing is still open
        if override.get("kind") in MAP_KINDS:
            return override.get("decision") == "exclude"
        if override.get("question_id"):
            if override["question_id"] != question_id:
                return False
            if kind == "verifier_refuted":
                # a number does not retire a refutation: the verifier said the value is wrong, and
                # a different value is not an answer to that. Only overruling it on the record —
                # or removing the cell, handled above — settles this question.
                return "verifier_refuted" in (override.get("overrules") or [])
            # …and answering "no, the right number is this one" is not answering "is this right?":
            # the question is settled by the decision it advertises, or by one that names what it
            # retires. Anything else changed the cell and left the question standing.
            return (override.get("kind") == writes or bool(override.get("overrules"))
                    or bool(override.get("clears")))
        # a record written outside the questions page — the manual override form, a log older than
        # `question_id` — cannot say which question it answered, and the cell's question changes as
        # answers land. Matching it by kind marked a question the reviewer had never been shown as
        # answered, on the strength of an older decision of the same kind. It settles nothing here;
        # what it settles is on the record, and the cell asks until the page is answered.
        return False

    already = [o for o in already if answers_this(o)]
    status, pending_why = _answer_status(already, pending, consumed)
    label = _group_label(dataset, group)
    citation = study.get("citation") or {}
    paper = f"{citation.get('first_author') or citation.get('authors') or '?'} {citation.get('year') or ''}".strip()
    where = _where(candidates)
    unit = _unit(candidates, dataset, outcome_key)
    x_hint = _x_hint(candidates)
    options = _stamped(_options(kind, valued, verdict, flags, unit, overruled))
    if kind == "dispersion_doubt":
        options = _stamped(_dispersion_options(flags))
    if kind == "converted_statistic":
        options = _stamped(_converted_options(verdict, overruled))
    image = _image(run, candidates, provenance)
    prompt = _prompt(kind, label, outcome_key, where, unit, x_hint, options, verdict,
                     measure, overruled)
    if kind == "dispersion_doubt":
        prompt = _dispersion_prompt(who=label, outcome_key=outcome_key, row=row)
    if kind == "converted_statistic":
        prompt = _converted_prompt(who=label, outcome_key=outcome_key, row=row,
                                   statistic=statistic)
    if kind == "needs_group_values":
        prompt = _needs_values_prompt(who=label, prompt=prompt, statistic=statistic, row=row)
    if kind == "reader_contradicts_values":
        prompt = _contradiction_prompt(prompt, run, entry, dataset, outcome_key, measure)
    why = _why(entry, verdict)
    if kind == "orientation":
        why = _with_ballots(why, run, str(entry.get("paper_id") or ""), outcome_key, measure)
    return Question(OrderedDict([
        ("id", question_id),
        ("kind", kind),
        ("prompt", prompt),
        ("paper", paper),
        ("paper_id", str(entry.get("paper_id") or "")),
        ("dataset_id", str(entry.get("dataset_id") or "")),
        ("dataset_label", str(dataset.get("label") or dataset.get("experiment") or "")),
        ("outcome_key", outcome_key),
        ("measure_name", measure),
        ("group", group),
        ("group_label", label),
        ("where", where),
        ("unit", unit),
        ("image", image),
        ("options", options),
        ("free_text", True),                       # a reviewer may always type the answer
        ("answer_writes", _answer_kind(kind)),
        ("why", why),
        ("confidence", entry.get("confidence")),
        ("route", entry.get("route")),
        ("impact", entry.get("impact_abs_delta_pooled")),
        ("answered", bool(already)),
        ("status", status),
        ("pending_why", pending_why),
        ("answers", [{"kind": o.get("kind"), "justification": o.get("justification"),
                      "mean": o.get("mean"), "at": o.get("at") or o.get("timestamp")}
                     for o in already]),
    ]))


def _holding_codes(verdict: Mapping[str, Any]) -> set[str]:
    """The flag codes on this cell that are actually withholding it, not merely costing it score.

    An `error`, a "this may be a different quantity" contradiction, or a cap: those are the three
    kinds of finding a score cannot settle. A plain warning ranks a cell in the queue and, by
    construction (`confidence.py`'s floor), can never withhold it — so a warning must not be what
    picks the question either.
    """
    from ..verify.confidence import CAPPING_FLAGS, CONTRADICTING_FLAGS

    codes: set[str] = set()
    for flag in verdict.get("flags") or []:
        code = str(flag.get("code") or "")
        if not code:
            continue
        if flag.get("severity") == "error" or code in CONTRADICTING_FLAGS or code in CAPPING_FLAGS:
            codes.add(code)
    return codes


#: C3 abstains with one of these when no ballot may decide the direction; C12 re-issues first.
#: `orientation_direction_conflict` is deliberately NOT here — it marks readers who contradicted
#: each other about what the paper *says*, on a cell whose direction was still decided, so it is a
#: reason to ask, not a reason to outrank a cell's other blockers (it sits in `_FLAG_TO_KIND`).
_ORIENTATION_UNRESOLVED: frozenset[str] = frozenset({
    "orientation_unknown", "orientation_unresolved"})


def _orientation_unresolved(verdict: Mapping[str, Any], flags: Sequence[str]) -> bool:
    """`higher_is_better` is the one thing a score cannot supply and a reader can (C4/C3 row 6).

    A null direction is the test: C3's abstention leaves it null, and C3's *decisions* — a single
    surviving witness, a majority — set it and carry their own flag, so those cells are not asked
    this question. The flag codes are a second reading of the same fact, kept because a cell may
    carry the abstention's code while some other stage has copied a direction onto it.
    """
    if not verdict:
        return False
    return (verdict.get("higher_is_better") is None
            or bool(_ORIENTATION_UNRESOLVED & {str(f) for f in flags}))


def _converted_row(row: Mapping[str, Any] | None) -> bool:
    """Did this cell's ROW get an effect size from something other than the two cells' numbers?

    A printed `t`, `F`, `p` or `d` is a value for the ROW, and the review layer must not ask a
    cell of such a row where its number is: there is no number to find, the paper prints none, and
    the only honest question is whether the reviewer accepts the conversion (whole-diff H2). The
    test itself lives with the rows — the analysis and the page must not be able to disagree about
    which rows are converted.
    """
    return converted_route(str((row or {}).get("route") or ""))


def _kind(verdict: Mapping[str, Any], flags: Sequence[str],
          valued: Sequence[Mapping[str, Any]], holding: set[str] | None = None,
          answered_value: bool = False, overruled: Collection[str] = (),
          row: Mapping[str, Any] | None = None,
          statistic: Mapping[str, Any] | None = None) -> str:
    """The question a held cell asks is the one whose answer removes what is holding it.

    The order below is *why the cell is held*, strongest reason first — not the order codes happen
    to sit in `_FLAG_TO_KIND`. A **forcing** finding (nothing was resolved, a verifier refuted it,
    the direction of the measure is unknown) outranks a capping one, because only a forcing finding
    is something no amount of corroboration can settle. Among the capping codes, one that is
    actually withholding the cell outranks one that merely cost it score.

    Before C4 this scanned `_FLAG_TO_KIND` positionally and never looked at the direction at all,
    so Bock's aftereffect — held because `higher_is_better` is null — was asked `which_series`
    (because `series_marker_mismatch` happened to be in the list) and answering it correctly left
    the cell exactly where it was.
    """
    converted = _converted_row(row)
    #: "no usable value was found — where is it?" is a true question only when the paper printed
    #: NOTHING for this contrast. A cell whose paper printed a t or an F that the conversion gate
    #: then refused has a value in it; what it lacks is the provenance of that statistic to these
    #: two groups, and the only thing that replaces it is both groups' own numbers. Asking such a
    #: cell "where is it, or is it not reported?" offers one answer, and that answer is false
    #: (whole-diff re-review N3).
    printed = bool(statistic)
    if not valued and not converted and not printed:
        return "no_value"
    if verdict.get("verifier_verdict") == "refuted" and "verifier_refuted" not in overruled:
        # asked until an answer NAMES it. Suppressing it once any value answer existed is what let
        # the terminal confirmation release a refuted cell without anyone addressing the refutation.
        return "verifier_refuted"
    if _orientation_unresolved(verdict, flags):
        return "orientation"
    matched = [(code, kind) for code, kind in _FLAG_TO_KIND if code in flags]
    withholding = [pair for pair in matched if pair[0] in (holding or set())]
    if withholding or matched:
        return (withholding or matched)[0][1]
    if not valued:
        # a converted row with nothing else wrong: the cell has no number and needs none, so the
        # question is about the CONVERSION and not about a value nobody printed — unless the
        # reviewer has since typed the paper's own group values, which is one of the answers this
        # question offers. Then the row has a number a person put there and the cell is at the
        # ordinary terminus: "is it right?".
        if answered_value:
            return "confirm_value"
        return "converted_statistic" if converted else "needs_group_values"
    # the number is on the record because a person put it there, so "which number is it" is not
    # the question any more. What is left is what a score measures — corroboration — and the one
    # thing that can supply it now is the person: `confirm_value` writes `mark_reviewed`, which is
    # the terminus every held cell needs or the page would ask the same question for ever.
    if answered_value:
        return "confirm_value"
    distinct = _distinct_values(valued)
    if len(distinct) >= 2:
        return "which_value"
    if len(distinct) == 1:
        return "confirm_value"
    return "other"


#: how each finding that is NOT a flag code reads in a sentence (`overrides.OVERRULABLE`)
_SAID: dict[str, str] = {
    "verifier_refuted": "the verifier's refutation",
    "adjudicated": "the adjudicator's ruling",
    "low_score": "a confidence score below the acceptance line",
    "no_group_values": "that neither group's own numbers were resolved for this cell"}


def _overrulable(verdict: Mapping[str, Any], overruled: Collection[str] = ()) -> list[str]:
    """What is holding this cell that no flag code names, minus what the log has already settled."""
    from ..verify.confidence import ACCEPT_WITH_NOTE

    out: list[str] = []
    if verdict.get("verifier_verdict") == "refuted":
        out.append("verifier_refuted")
    if verdict.get("adjudicated"):
        out.append("adjudicated")
    if (verdict.get("confidence_score") or 0.0) < ACCEPT_WITH_NOTE:
        out.append("low_score")
    return [name for name in out if name not in set(overruled)]


def _rows_by_cell(run: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Every row of the run's own table, indexed by the cell it belongs to."""
    rows = _json_if_present(run / "results" / "extraction_table_all.json") or []
    return {(str(row.get("dataset_id") or ""), str(row.get("outcome_key") or "")): row
            for row in rows if isinstance(row, dict)}


def _dispersion_options(flags: Sequence[str]) -> list[dict[str, Any]]:
    """The two things a doubted denominator can turn out to be, each naming the dispersion.

    C9's screen says the spread the row was divided by cannot be the one the authors used — two
    means eight standard deviations apart are almost always two means divided by something else.
    So the answers name what that something else is; neither asserts the screen away, because the
    screen is arithmetic and is re-derived from whatever the cell holds afterwards.

    There used to be a third — "the value and its spread are both right as printed" — and its own
    label said it changed nothing. Recording it counted the question as ANSWERED (`answers_this`
    reads a `mark_reviewed` that names what it clears), so a row C9 still refuses ended up with
    every question on it ticked and nothing open anywhere: precisely the state C4 exists to
    remove. C9's screen is not human-overrulable in this build, so the honest form of "I have
    checked and they are both right" is that the row STAYS a visible question (whole-diff M3).
    """
    clears = _present(flags, "implausible_dispersion")
    return [
        {"key": "se_not_sd", "dispersion_type": "SE", "clears": clears,
         "label": "the error bars are standard errors, not standard deviations — convert them"},
        {"key": "within_subject",
         "label": "this is a within-subject error bar, not a between-group SD — the paper does "
                  "not print a usable denominator, so the cell leaves the analysis"},
    ]


def _converted_options(verdict: Mapping[str, Any],
                       overruled: Collection[str] = ()) -> list[dict[str, Any]]:
    """The two answers that can be PICKED for a row built from a printed statistic.

    Accepting it is not a number, so — exactly as `confirm_value` — it is not shaped like one: it
    names what it OVERRULES, because "neither group's numbers were resolved" is not a flag code
    and no `clears` could ever say a person had read the statistic and stood behind it. It names
    every OTHER finding still holding the cell as well: on a cell with no group value the score is
    0.0, so an answer naming only `no_group_values` would leave the cell exactly where it was —
    which is the defect `both_right` was dropped for.

    The third answer is the paper's own group values, typed into the same free-text form every
    value question offers, so it is not an option here.
    """
    overrules = ["no_group_values", *_overrulable(verdict, overruled)]
    said = ", ".join(_SAID[name] for name in overrules if name in _SAID)
    return [
        {"key": "accept", "overrules": overrules,
         "label": "accept the converted effect — I have read the printed statistic and it is the "
                  "comparison this row pools"
                  + (f"; this overrules {said}" if said else "")},
        {"key": "not_usable",
         "label": "the statistic is not usable for this contrast, and the paper prints no group "
                  "values — the cell leaves the analysis"},
    ]


def _statistic_of(candidates: Sequence[Mapping[str, Any]],
                  route: str = "") -> Mapping[str, Any]:
    """The printed statistic or printed effect size this row was converted FROM.

    Chosen by the RESOLVER's own rule (`pipeline.rows.converting_candidate`, which is
    `checks.best_statistic` for a statistic route), never "the first one in the stage file". A
    paper prints several statistics near an outcome and the row uses exactly one of them: taking
    the first listed named an inadmissible `F(1,22)` above the `t(22)` the row converted from, so
    the question showed a statistic that had nothing to do with the effect size beside it, and
    asked the reviewer to accept it (whole-diff re-review N1).

    The stage files hold dicts; they are the same models that wrote them, so they validate. One
    that does not is skipped rather than allowed to take the whole question down.
    """
    from ..models import Candidate
    from ..pipeline.rows import converting_candidate

    parsed: list[Candidate] = []
    for cand in candidates:
        try:
            parsed.append(Candidate.model_validate(cand))
        except Exception:                              # pragma: no cover - defensive
            continue
    chosen = converting_candidate(parsed, route)
    if chosen is None:
        return {}
    return next((c for c in candidates if c.get("candidate_id") == chosen.candidate_id), {})


def _num(value: Any) -> str:
    """A number as the paper would print it — `22`, not `22.0`; `""` for nothing at all."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _statistic_sentence(cand: Mapping[str, Any]) -> str:
    """`t(22) = 2.50, p = .02 — an independent_t contrasting groups`, from the record's own fields.

    Never a paraphrase of the paper: every part is a field the extractor filled in, and a field it
    could not fill in is left out rather than guessed at.
    """
    if not cand:
        return "the printed statistic is not on the record"
    if cand.get("kind") == "reported_d":
        head = f"a printed effect size of {_num(cand.get('reported_value'))}"
        scale = str(cand.get("reported_scale") or "")
        if scale and scale != "unknown":
            head += f" on the {scale} scale"
    else:
        stat = str(cand.get("stat_type") or "?")
        df = ", ".join(_num(x) for x in (cand.get("df1"), cand.get("df2")) if x is not None) \
            or _num(cand.get("df"))
        head = f"{stat}({df}) = {_num(cand.get('stat_value'))}" if df else \
            f"{stat} = {_num(cand.get('stat_value'))}"
        if cand.get("df") is None and cand.get("df1") is None:
            head += " with NO degrees of freedom printed"
        if cand.get("p_value") is not None:
            head += f", p = {_num(cand['p_value'])}"
    parts = [head]
    design = str(cand.get("design") or "")
    contrast = str(cand.get("contrast_kind") or "")
    if design and design != "unknown":
        parts.append(f"design {design}")
    parts.append(f"contrast {contrast or 'unknown'}")
    quote = str(cand.get("quote") or "")
    if quote:
        parts.append(f"quoting \u201c{_short(quote, 200)}\u201d")
    return " — ".join(parts)


def _converted_prompt(*, who: str, outcome_key: str, row: Mapping[str, Any],
                      statistic: Mapping[str, Any]) -> str:
    """The statistic, what the conversion made of it, and what the gates said — in the question.

    A reviewer asked "where is this value?" about a row whose value is a printed t cannot answer:
    the paper prints no group means, and the only decision there is is whether the conversion may
    stand. So the question shows the statistic as extracted, the effect size it produced, and the
    resolver's own words about it (whole-diff H2).
    """
    stat = statistic
    effect = (f"{_fmt(float(row['es']))}" if isinstance(row.get("es"), (int, float))
              else "no effect size")
    # the resolver's own words: the step that applied the formula, and anything the gate said
    # about the degrees of freedom or the contrast. Never a paraphrase of either.
    steps = [part for part in str(row.get("conversion_chain") or "").split("; ") if part.strip()]
    said = steps[:1] + [part for part in steps[1:]
                        if "degrees of freedom" in part or "df_" in part
                        or "contrast" in part][:1]
    group = f"the {who} group" if who else "this group"
    tail = (" The resolver recorded: " + "; ".join(_short(x, 220) for x in said)) if said else ""
    return (f"This {outcome_key.replace('_', ' ')} row prints no mean, spread or n for "
            f"{group} — its effect size was converted from what the paper DID print: "
            f"{_statistic_sentence(stat)}. That conversion gives d = {effect}. Does the "
            f"statistic contrast these two groups, so the converted effect may be pooled? "
            f"Answer for the other group too, or type this group's own mean, spread and n "
            f"below if the paper prints them after all.{tail}")


def _needs_values_prompt(*, who: str, prompt: str, statistic: Mapping[str, Any],
                         row: Mapping[str, Any]) -> str:
    """…and WHICH statistic, when the row rests on one the gate refused.

    `_number_trouble` reads the codes that raise `number_unusable` (a bad n, a bad SD), so on a
    cell held for the degrees of freedom of a printed t it said "the check did not say which". The
    statistic and the resolver's own refusal are what the reviewer needs to see before deciding
    whether to type the paper's group values or take the cell out (whole-diff re-review N3).
    """
    if not statistic:
        return prompt
    said = next((part for part in str(row.get("not_convertible_reason") or "").split("; ")
                 if "degrees of freedom" in part or "estimand" in part or "contrast" in part), "")
    return (f"{prompt} The statistic is {_statistic_sentence(statistic)}."
            + (f" The resolver refused it: {_short(said, 320)}" if said else ""))


def _dispersion_prompt(*, who: str, outcome_key: str, row: Mapping[str, Any]) -> str:
    """The resolver's own words about the denominator it divided by, in the question."""
    said = [step for step in row.get("conversion_steps") or []
            if "implausible_dispersion" in str(step)]
    detail = _short(said[0], 320) if said else (
        f"the resolved values imply |d| = {abs(float(row['es'])):.2f}"
        if isinstance(row.get("es"), (int, float)) else "the resolved values imply an impossible "
                                                        "effect")
    group = f"the {who} group" if who else "this group"
    return (f"The spread this {outcome_key.replace('_', ' ')} row was divided by cannot be the one "
            f"the authors used: {detail}. What is {group}'s spread really?")


def _present_holds(verdict: Mapping[str, Any], *names: str,
                   overruled: Collection[str] = ()) -> list[str]:
    """Of the non-code findings this answer would overrule, the ones the cell actually has."""
    holding = set(_overrulable(verdict, overruled))
    return [name for name in names if name in holding]


def _present(flags: Sequence[str], *codes: str) -> list[str]:
    """Of the findings this answer would settle, the ones the cell actually carries.

    An option may only name what is on the record: `clears` retires evidence, and the log refuses
    a record naming a finding its cell never raised. So the option is built from the cell rather
    than from the kind — "yes, this series is this group" answers the marker mismatch when that is
    what was found, and says nothing about a transposition nobody detected.
    """
    on_the_cell = set(flags)
    return [code for code in codes if code in on_the_cell]


def _options(kind: str, valued: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
             flags: Sequence[str], unit: str,
             overruled: Collection[str] = ()) -> list[dict[str, Any]]:
    """The answers on offer. Values are the candidates' own numbers; never invented."""
    if kind == "error_bar_type":
        return [{"key": k, "label": lbl, "dispersion_type": k}
                for k, lbl in (("SD", "standard deviation"), ("SE", "standard error"),
                               ("CI95", "95% confidence interval"), ("IQR", "interquartile range"),
                               ("RANGE", "range"))]
    if kind == "group_mapping":
        return [{"key": "as_mapped", "label": "the mapping is right",
                 "clears": _present(flags, "quote_row_only", "group_label_swapped")},
                {"key": "swapped", "label": "the two groups are swapped"}]
    if kind == "orientation":
        # the only two answers there are, and both change the cell: `higher_is_better` is what
        # `canopy.stats.orient` multiplies the effect size by.
        return [{"key": "higher_is_more", "higher_is_better": True,
                 "label": "a LARGER raw value means more of what the protocol scores "
                          "(so a larger value is the better score)"},
                {"key": "lower_is_more", "higher_is_better": False,
                 "label": "a SMALLER raw value means more of what the protocol scores "
                          "(so this is an error-type measure)"}]
    if kind == "which_axis":
        # an axis is only answerable if choosing it says which NUMBER the cell should hold, so
        # every option carries a candidate's own value and names the ladder it was read against.
        # Before C4 these were three prose descriptions of a ladder whose answer wrote nothing.
        return _value_options(valued, verdict, unit, axis=True)
    if kind == "which_series":
        # "yes, this series is this group" is a decision about a NUMBER, so it writes the number
        # it confirms. Before C4 it wrote `mark_reviewed / needs_human` — the answer and the
        # question in the same state.
        resolved = _value_options(valued, verdict, unit)[:1]
        out = []
        if resolved and resolved[0].get("mean") is not None:
            out.append({**resolved[0], "key": "as_read",
                        # confirming the identity is the whole answer to the identity findings:
                        # without saying so, the cell keeps asking a question it has answered.
                        "clears": _present(flags, "series_marker_mismatch",
                                           "series_identity_conflict", "series_transposed"),
                        "label": f"this series is this group — {resolved[0]['label']} is right"})
        out.append({"key": "other_series", "label": "the value belongs to the other group"})
        return out
    if kind == "reader_contradicts_values":
        # three answers, because there are exactly three ways the contradiction can resolve: the
        # sentence was misread, the two series are the wrong way round, or one of the numbers is
        # wrong — and the third is answered by saying which number is right.
        return [{"key": "numbers_right",
                 "label": "the numbers are right — the reader misread the sentence",
                 "clears": _present(flags, "orientation_reader_contradicts_values")},
                {"key": "groups_swapped",
                 "label": "the two groups are swapped — this number is the other group's"},
                *_value_options(valued, verdict, unit)]
    if kind == "quote_not_found":
        # the quote nobody could find was the number's whole provenance, so the answer has to be a
        # number a person will stand behind — or the statement that the paper does not print one.
        # Standing behind it IS the answer to the grounding, and the option says so.
        return [*({**option, "clears": _present(flags, "quote_not_grounded")}
                  for option in _value_options(valued, verdict, unit)),
                {"key": "not_reported", "label": "the paper does not print this value"}]
    if kind == "number_unusable":
        # nothing here can be picked off the page: the missing n / df / spread has to be typed, or
        # the cell has to leave. Offering a candidate's own number would be offering the number
        # the check already refused.
        return [{"key": "not_reported", "label": "it is genuinely not reported in the paper"}]
    if kind == "confirm_value":
        # "yes" is not a number, so it must not be shaped like one: picking it writes the
        # `mark_reviewed` this kind advertises, and the alternatives below still write a value.
        # It carries what it OVERRULES, because the findings that hold such a cell — a refutation,
        # an adjudication, a score below the line — are not flag codes, so no `clears` can name
        # them and nothing else in the record would say the reviewer had addressed them.
        settled = _value_options(valued, verdict, unit)
        overrules = _overrulable(verdict, overruled)
        said = (" — this overrules " + ", ".join(_SAID[name] for name in overrules)
                if overrules else "")
        head = {"key": "yes", "overrules": overrules,
                "label": (f"yes — {settled[0]['label']} is right, I have checked it{said}"
                          if settled else f"yes — this value is right, I have checked it{said}")}
        return [head, *settled[1:]]
    if kind == "needs_group_values":
        # §C9: the degrees of freedom of a printed statistic are answered by not needing the
        # statistic. Nothing on the page can be picked for that — both groups' own numbers have to
        # be typed — so the only option is the other answer: the paper does not print them.
        return [{"key": "not_reported",
                 "label": "the paper does not print both groups' means, spreads and sizes"}]
    if kind == "verifier_refuted":
        # a refutation is not a flag code, so no number retires it and no `clears` can name it: a
        # cell a verifier refused stays refused until a person says, on the record, that they have
        # read the objection and disagree. Without this option the question is unanswerable — the
        # numbers below change the value and leave the refutation exactly where it was.
        return [{"key": "stands",
                 "overrules": _present_holds(verdict, "verifier_refuted", overruled=overruled),
                 "label": "the value is right anyway — I have read the verifier's objection and "
                          "disagree with it"},
                *({**option, "clears": _present(flags, "value_outside_axis", "sign_mismatch")}
                  for option in _value_options(valued, verdict, unit))]
    if kind == "which_value":
        # naming the right number is the answer to "that number is off the ladder" and to "the
        # paper says the other group was higher" — the two findings that raise this question.
        return [{**option, "clears": _present(flags, "value_outside_axis", "sign_mismatch")}
                for option in _value_options(valued, verdict, unit)]
    if kind == "no_value":
        # the free-text answer ("it is on p. 5, Table 2") is a re-extraction; the one thing a
        # reviewer can settle without a model call is that there is nothing to read.
        return [{"key": "not_reported", "label": "it is genuinely not reported in the paper"}]
    return []


def _value_options(valued: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any], unit: str,
                   *, axis: bool = False) -> list[dict[str, Any]]:
    """The candidates' own numbers as answers, most-backed first. Nothing here is invented.

    When `axis` is set, each option also names the ladder its backers said they read it against,
    so "which axis?" is answered by picking the number that axis implies. The value the run
    actually resolved is always on offer even when no candidate carries it (an adjudicated cell
    holds a number no single reader wrote), because "the one you have is right" has to be a
    sayable answer — and because it has to be a *recorded* one, not a shrug.
    """
    out: list[dict[str, Any]] = []
    for value, backers in _distinct_values(valued).items():
        best = backers[0]
        option = {
            "key": f"v{len(out) + 1}",
            "label": f"{_fmt(value)}{(' ' + unit) if unit else ''}",
            "mean": value,
            "dispersion_value": best.get("dispersion_value"),
            "dispersion_type": _enum(best.get("dispersion_type")),
            "n": best.get("n"),
            "unit": unit,
            "backed_by": sorted({f"{_route_name(b)}" for b in backers}),
            "n_backers": len(backers),
            "quote": next((b.get("quote") for b in backers if b.get("quote")), ""),
            "page": best.get("page"),
        }
        if axis:
            option["axis"] = _axis_of(backers)
            if option["axis"]:
                option["label"] += f" — read against {_short(option['axis'], 90)}"
        out.append(option)
    out.sort(key=lambda o: -o["n_backers"])
    resolved = verdict.get("mean")
    if resolved is not None and not any(
            abs(float(o["mean"]) - float(resolved)) <= max(abs(float(resolved)) * 0.005, 1e-9)
            for o in out):
        out.insert(0, {
            "key": "resolved", "mean": float(resolved),
            "label": f"{_fmt(float(resolved))}{(' ' + unit) if unit else ''} — the value this run "
                     f"resolved is right",
            "dispersion_value": verdict.get("dispersion_value"),
            "dispersion_type": _enum(verdict.get("dispersion_type")),
            "n": verdict.get("n"), "unit": unit, "backed_by": [], "n_backers": 0,
            "quote": "", "page": None,
        })
    for i, option in enumerate(out, 1):
        if option["key"] != "resolved":
            option["key"] = f"v{i}"
    return out[:_MAX_OPTIONS]


def _axis_of(backers: Sequence[Mapping[str, Any]]) -> str:
    """The value ladder a reading was taken against, as its own reader described it."""
    for backer in backers:
        pp = backer.get("pixel_provenance") or {}
        reads = pp.get("axis_reads") or ([pp.get("axis_read")] if pp.get("axis_read") else [])
        for read in reads:
            if str(read or "").strip():
                return str(read).strip()
    return ""


def fingerprint(option: Mapping[str, Any]) -> str:
    """A short hash of what an option MEANS, so an answer can echo the thing it was shown.

    The question `id` guards which question is being answered; it cannot guard which option,
    because `v1…vN` are positional and the list is rebuilt from the cell's current numbers — after
    any value answer the same key under the same id can denote a different number. This is what
    the answer echoes and the server compares.
    """
    payload = json.dumps({key: option.get(key) for key in
                          ("key", "label", "mean", "dispersion_value", "dispersion_type", "n",
                           "analysis_metric", "location", "decision", "rule", "higher_is_better",
                           "clears", "overrules")},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _stamped(options: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**option, "fingerprint": fingerprint(option)} for option in options]


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rsplit(" ", 1)[0] + "…"


def _distinct_values(valued: Sequence[Mapping[str, Any]]) -> "OrderedDict[float, list]":
    """Candidate values grouped by rounded mean, ensembles first, most-backed first."""
    order = sorted(valued, key=lambda c: (not str(c.get("extractor_id") or "").endswith("ensemble"),
                                          str(c.get("candidate_id") or "")))
    groups: "OrderedDict[float, list]" = OrderedDict()
    for c in order:
        value = float(c["mean"])
        key = next((k for k in groups if abs(k - value) <= max(abs(k) * 0.005, 1e-9)), None)
        groups.setdefault(value if key is None else key, []).append(c)
    return groups


def _prompt(kind: str, label: str, outcome_key: str, where: str, unit: str, x_hint: str,
            options: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
            measure: str = "", overruled: Collection[str] = ()) -> str:
    who = f"the {label} group" if label else "this group"
    at = f" at {x_hint}" if x_hint else ""
    u = f" ({unit})" if unit else ""
    src = f" in {where}" if where else ""
    outcome = outcome_key.replace("_", " ")
    if kind == "which_value":
        return (f"Which of these is {who}'s {outcome}{at}{src}{u}? The routes that read it "
                f"disagree.")
    if kind == "confirm_value":
        value = next((o["label"] for o in options if o.get("mean") is not None),
                     (_fmt(float(verdict["mean"])) + u) if verdict.get("mean") is not None
                     else "this value")
        return (f"Is {value} {who}'s {outcome}{at}{src}? This cell is held because "
                f"{_held_because(verdict, overruled)}. Answering yes records that you have "
                f"checked it, and "
                f"overrules exactly that.")
    if kind == "which_axis":
        return (f"Which of these is {who}'s {outcome}{at}{src}{u}? The readers and the axis "
                f"ladder disagree about the scale, so each answer names the ladder it was read "
                f"against — pick the number the right ladder gives.")
    if kind == "which_series":
        return (f"In {where or 'this figure'}, which plotted series is {who}? The marker the "
                f"reader described and the one the pixel pass found do not match, so the number "
                f"below may belong to the other group.")
    if kind == "orientation":
        return (f"For “{_short(measure or outcome, 160)}”{u}: does a LARGER raw value mean more "
                f"of what this review scores, or less? The readers disagreed, so no effect size "
                f"built on this measure can be signed — and the answer settles it for every cell "
                f"of this measure in this paper.")
    if kind == "error_bar_type":
        return (f"What do the error bars{src} show? Nothing in the paper's text settled it, so "
                f"the spread cannot be converted with confidence.")
    if kind == "group_mapping":
        return (f"Is the group mapping right for this dataset — is {label or 'group A'} the "
                f"group the protocol calls A?")
    if kind == "verifier_refuted":
        return (f"A verifier reading the whole paper says this {outcome} value for {who} is "
                f"wrong: “{str(verdict.get('verifier_reason') or '')[:300]}”. Is it right "
                f"anyway, or should the cell be excluded?")
    if kind == "no_value":
        return (f"No usable {outcome} value was found for {who}{src}. Where in the paper is it "
                f"— page, figure or table — or is it genuinely not reported?")
    if kind == "quote_not_found":
        return (f"The sentence this {outcome} value for {who} rests on could not be found in the "
                f"paper, so nothing shows where the number came from. Which of these is the "
                f"right number{src}{u} — or does the paper not print one?")
    if kind == "needs_group_values":
        return (f"This row was converted from a printed statistic, and "
                f"{_short(_number_trouble(verdict), 220)}. Nothing about the statistic itself can "
                f"settle that: the only thing that replaces it is both groups' own numbers. Type "
                f"{who}'s mean, spread and n below — and answer the other group's question too — "
                f"or say the paper does not print them.")
    if kind == "number_unusable":
        return (f"A number this cell's conversion needs is missing or cannot be right "
                f"({_short(_number_trouble(verdict), 200)}). What is it for {who} — type it "
                f"below — or is it not reported?")
    if kind == "reader_contradicts_values":
        return (f"A reader's own words about which group came out higher contradict this cell's "
                f"numbers. Which is wrong?")
    return f"This {outcome} cell for {who} was held for review. Why is given below."


def _held_because(verdict: Mapping[str, Any], overruled: Collection[str] = ()) -> str:
    """Why the RECORD says this cell is held — never a hard-coded sentence.

    `confirm_value` is the terminus, so its prompt is the last thing a reviewer reads before a cell
    enters the analysis. It used to say "only one route produced it" whatever the cell's actual
    reason was, including on a cell a verifier had refuted.
    """
    said = {"verifier_refuted": "a verifier reading the whole paper says this value is wrong",
            "adjudicated": "an adjudicator had to settle it, so no vote decided it",
            "low_score": "its confidence score never reached the acceptance line"}
    reasons = [said[name] for name in _overrulable(verdict, overruled)]
    if verdict.get("agreement") == "single":
        reasons.append("only one route produced it, so nothing independent confirms it")
    score = verdict.get("confidence_score")
    tail = f" (score {float(score):.2f})" if isinstance(score, (int, float)) else ""
    return (" and ".join(reasons) or "the tool could not settle it") + tail


def _number_trouble(verdict: Mapping[str, Any]) -> str:
    """The check's own words about the number it refused, so the question names what is wrong."""
    wanted = {code for code, kind in _FLAG_TO_KIND if kind == "number_unusable"}
    said = [str(flag.get("message") or flag.get("code") or "")
            for flag in verdict.get("flags") or [] if str(flag.get("code") or "") in wanted]
    return "; ".join(dict.fromkeys(x for x in said if x)) or "the check did not say which"


def _contradiction_prompt(prompt: str, run: Path, entry: Mapping[str, Any],
                          dataset: Mapping[str, Any], outcome_key: str, measure: str) -> str:
    """The reader's own sentence and THIS cell's two means, in the question itself.

    The check fires when a reader said "the older group was higher" and the resolved means say the
    opposite. A reviewer cannot arbitrate that from a code — they need the sentence and the two
    numbers side by side, which is the whole content of the disagreement.
    """
    paper_id = str(entry.get("paper_id") or "")
    dataset_id = str(entry.get("dataset_id") or "")
    decision = _orientation_decision(run, paper_id, outcome_key, measure)
    # the DISCARDED reader's own words, not the verdict's summary. The check fires when a reader's
    # stated direction is contradicted by the means, and the summary field collapses to "unknown"
    # in exactly that case — two readers stated opposite directions, one of them was discarded —
    # so the reviewer was told 'The reader said "unknown"' about the only question they were being
    # asked to arbitrate (whole-diff L3).
    reader = next((r for r in (decision.get("runs") or [])
                   if isinstance(r, dict) and r.get("discarded")), {})
    stated = str(reader.get("direction_stated_in_text")
                 or decision.get("direction_stated_in_text") or "unknown")
    who = str(reader.get("model") or "")
    said_quotes = list(reader.get("quotes") or decision.get("quotes") or [])
    quotes = "; ".join(f"“{_short(q, 220)}”" for q in said_quotes[:2])
    means = []
    for group in ("A", "B"):
        cell = _verdict(run, paper_id, dataset_id, outcome_key, group)
        label = _group_label(dataset, group) or f"group {group}"
        value = cell.get("mean")
        means.append(f"{label} = {_fmt(float(value)) if value is not None else '—'}")
    said = {"a_greater": "group A came out higher", "b_greater": "group B came out higher",
            "equal": "the groups came out the same"}.get(stated, f"“{stated}”")
    name = f"The reader {who} said" if who else "The reader said"
    return (f"{prompt} {name} {said}; this cell resolved to {' and '.join(means)}."
            + (f" The reader was quoting: {quotes}" if quotes else ""))


# ------------------------------------------------------- the questions the MAP could not settle
def _excluded_questions(run: Path, overrides: Sequence[Mapping[str, Any]],
                        asked: Sequence[Mapping[str, Any]]) -> list[Question]:
    """One answered question per cell a reviewer took out of the analysis.

    An excluded cell leaves the review queue: it is settled, and a settled cell counted as
    outstanding for ever is what made the queue disagree with the exclusion table. The decision
    still has to be visible where it was taken, so the page keeps ONE entry for it — answered, with
    the reason — instead of the two (one per group) an excluded cell used to show.
    """
    seen = {(q["dataset_id"], q["outcome_key"]) for q in asked}
    out: list[Question] = []
    for override in overrides:
        if override.get("kind") == "exclude_dataset":
            pass
        elif override.get("kind") in MAP_KINDS and override.get("decision") == "exclude":
            pass
        else:
            continue
        dataset_id = str(override.get("dataset_id") or "")
        outcome_key = str(override.get("outcome_key") or "")
        key = (dataset_id, outcome_key)
        if not dataset_id or key in seen:
            continue
        seen.add(key)
        paper_id = str(override.get("paper_id") or "")
        study, dataset = _dataset(run, paper_id, dataset_id)
        citation = study.get("citation") or {}
        kind = (str(override.get("question_id") or "").split("|")[3:4] or ["other"])[0]
        out.append(Question(OrderedDict([
            ("id", "|".join([dataset_id, outcome_key, "", kind or "other"])),
            ("kind", kind if kind in QUESTION_KINDS else "other"),
            ("prompt", "This cell was excluded from the analysis, so nothing about it is still "
                       "open."),
            ("paper", f"{citation.get('first_author') or citation.get('authors') or '?'} "
                      f"{citation.get('year') or ''}".strip()),
            ("paper_id", paper_id),
            ("dataset_id", dataset_id),
            ("dataset_label", str(dataset.get("label") or dataset.get("experiment") or "")),
            ("outcome_key", outcome_key),
            ("measure_name", _measure(dataset, outcome_key)),
            ("group", None), ("group_label", ""), ("where", ""),
            ("unit", _unit([], dataset, outcome_key)),
            ("image", {}), ("options", []), ("free_text", False),
            ("answer_writes", "exclude_dataset"),
            ("why", str(override.get("justification") or "")),
            ("confidence", "excluded"), ("route", ""), ("impact", None),
            ("answered", True), ("status", "answered"), ("pending_why", ""),
            ("answers", [{"kind": override.get("kind"),
                          "justification": override.get("justification"), "mean": None,
                          "at": override.get("at")}]),
        ])))
    return out


def _map_questions(run: Path, overrides: Sequence[Mapping[str, Any]],
                   pending: Mapping[int, str],
                   consumed: Collection[int] = ()) -> list[Question]:
    """C6/C7: the map's own open questions, as questions a reviewer can answer.

    These are not held cells. A held cell was read and could not be trusted; a map question is a
    cell nobody was allowed to read, because buying an extraction against an unsettled question is
    how a rejected dataset produced a fully signed effect size. So they are absent from the review
    queue by construction, and before this they reached no page in the tool at all — the mapper
    wrote them and nothing asked them.

    Read from each paper's own map stage file. The review package deliberately does not import the
    agent package: the recorded JSON is the whole interface, which is also what makes an old run's
    questions readable by a newer tool.
    """
    rules = _dataset_rules(run)
    out: list[Question] = []
    for paper_id, study in _studies(run):
        for raw in study.get("open_questions") or []:
            if str(raw.get("kind") or "") in MAP_KINDS:
                out.append(_map_question(run, paper_id, study, raw, rules, overrides, pending,
                                         consumed))
    return out


def _studies(run: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every paper's study map, in the run's own order, with the paper id it was written under."""
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    manifest = _json_if_present(run / "manifest.json") or {}
    for paper in manifest.get("papers") or []:
        paper_id = str(paper.get("paper_id") or "")
        study = _stage(run, paper_id, "map").get("study") or {}
        if study:
            out.append((paper_id, study))
            seen.add(paper_id)
    for path in sorted((run / "papers").glob("*/map.json")):     # a paper the manifest never named
        study = (_json_if_present(path) or {}).get("study") or {}
        paper_id = str(study.get("paper_id") or "")
        if study and paper_id not in seen:
            out.append((paper_id, study))
            seen.add(paper_id)
    return out


def _dataset_rules(run: Path) -> list[str]:
    """The review's own dataset rules — the only reasons an inclusion question may be answered
    'exclude' on (C7: the adjudicator may reject only on a NAMED protocol rule, and so may a
    human). Read from the protocol the run was made with."""
    from ..protocol import load_protocol

    for name in ("protocol.yaml", "protocol.staged.yaml"):
        path = run / name
        if path.exists():
            try:
                protocol = load_protocol(path)
            except (OSError, ValueError):                       # a damaged protocol is not a crash
                continue
            return [str(rule) for rule in protocol.dataset_rules if str(rule).strip()]
    return []


def _map_question(run: Path, paper_id: str, study: Mapping[str, Any], raw: Mapping[str, Any],
                  rules: Sequence[str], overrides: Sequence[Mapping[str, Any]],
                  pending: Mapping[int, str], consumed: Collection[int] = ()) -> Question:
    kind = str(raw.get("kind") or "")
    dataset_id = str(raw.get("dataset_id") or "")
    outcome_key = str(raw.get("outcome_key") or "")
    dataset = next((d for d in study.get("datasets") or []
                    if d.get("dataset_id") == dataset_id), {})
    label = str(dataset.get("label") or dataset.get("experiment") or dataset_id)
    citation = study.get("citation") or {}
    paper = f"{citation.get('first_author') or citation.get('authors') or '?'} " \
            f"{citation.get('year') or ''}".strip()
    sources = _map_sources(dataset, outcome_key)
    options = _stamped(_measure_options(sources, raw) if kind == "which_measure"
                       else _inclusion_options(rules))
    image = _image(run, [{"paper_id": paper_id,
                          "pixel_provenance": {"figure_id": source.get("figure_id")}}
                         for source in sources if source.get("figure_id")], {})
    already = [o for o in overrides if o.get("kind") == kind
               and o.get("dataset_id") == dataset_id
               and (not outcome_key or o.get("outcome_key") in ("", outcome_key))]
    where = _short(str((sources[0].get("locator") if sources else "") or ""), 70)
    # a map answer is one answer per question, and the later one is the live one — the same rule
    # `map_answers` and the re-pool apply, so all three read the log the same way.
    already = already[-1:]
    status, pending_why = _answer_status(already, pending, consumed)
    return Question(OrderedDict([
        ("id", "|".join([dataset_id, outcome_key, "", kind])),
        ("kind", kind),
        ("prompt", str(raw.get("question") or "") or _map_prompt(kind, label, outcome_key)),
        ("paper", paper),
        ("paper_id", paper_id),
        ("dataset_id", dataset_id),
        ("dataset_label", label),
        ("outcome_key", outcome_key),
        ("measure_name", _measure(dataset, outcome_key)),
        ("group", None),
        ("group_label", ""),
        ("where", where),
        ("unit", _unit([], dataset, outcome_key)),
        ("image", image),
        ("options", options),
        ("free_text", True),
        ("answer_writes", _answer_kind(kind)),
        ("why", _map_why(kind, raw)),
        ("confidence", "needs_human"),
        ("route", "map"),
        ("impact", None),
        ("answered", bool(already)),
        ("status", status),
        ("pending_why", pending_why),
        ("answers", [{"kind": o.get("kind"), "justification": o.get("justification"),
                      "mean": None, "at": o.get("at") or o.get("timestamp")}
                     for o in already]),
    ]))


def _map_sources(dataset: Mapping[str, Any], outcome_key: str) -> list[dict[str, Any]]:
    """The map's own `value` locations: for one outcome when the question names one, else the
    dataset's. A location the map demoted (`context`, `baseline`, `alternate`) is not an answer to
    "where does this outcome's number come from" and is not offered as one."""
    out: list[dict[str, Any]] = []
    for outcome in dataset.get("outcomes") or []:
        if outcome_key and outcome.get("outcome_key") != outcome_key:
            continue
        for source in outcome.get("sources") or []:
            if str(source.get("role") or "value") == "value":
                out.append(dict(source))
    return out


def _measure_options(sources: Sequence[Mapping[str, Any]],
                     raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One option per measure the map's own value locations carry — the metric, where it is
    printed, and its own words. Choosing one names the winner; nothing here is invented."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        metric = str(source.get("analysis_metric") or "unknown")
        where = str(source.get("locator") or source.get("figure_id")
                    or source.get("table_id") or "").strip()
        if metric in ("", "unknown"):
            # the map withholds a location whose metric it never determined, and `apply_map_answers`
            # refuses an answer naming one — so offering it is offering an answer that settles
            # nothing, which is exactly the failure C4 exists to remove.
            continue
        if (metric, where) in seen:
            continue
        seen.add((metric, where))
        page = source.get("page")
        out.append({
            "key": f"m{len(out) + 1}",
            "label": f"{metric.replace('_', ' ')} — {_short(where, 90)}"
                     + (f" (p. {page})" if page else ""),
            "analysis_metric": metric,
            "location": where,
            "page": page,
            "figure_id": str(source.get("figure_id") or ""),
            "quote": _short(source.get("quote"), 220),
        })
    if out:
        return out[:_MAX_OPTIONS]
    # a map with no value location left to offer: the mapper's own words are still an answer
    return [{"key": f"m{i}", "label": _short(text, 160), "analysis_metric": "", "location":
             str(text), "page": None, "figure_id": "", "quote": ""}
            for i, text in enumerate(raw.get("options") or [], 1)][:_MAX_OPTIONS]


def _inclusion_options(rules: Sequence[str]) -> list[dict[str, Any]]:
    """Include, or exclude on a NAMED rule of this review's protocol (C7). The rules are the menu
    because "excluded because it looked wrong" is not a reason a reader of the review can check."""
    out: list[dict[str, Any]] = [
        {"key": "include", "decision": "include", "rule": "",
         "label": "include it — no rule in this review's protocol excludes it"}]
    for i, rule in enumerate(list(rules)[:max(_MAX_OPTIONS - 2, 0)], 1):
        out.append({"key": f"exclude_r{i}", "decision": "exclude", "rule": str(rule),
                    "label": f"exclude it — {_short(rule, 160)}"})
    out.append({"key": "exclude", "decision": "exclude", "rule": "",
                "label": "exclude it — for the reason in the note"})
    return out


def _map_prompt(kind: str, label: str, outcome_key: str) -> str:
    if kind == "include_dataset":
        return (f"Only one of the two mapping agents proposed {label!r}, and no protocol rule was "
                f"cited to exclude it. Does this review include it?")
    return (f"Two measures are named for {outcome_key.replace('_', ' ')} in {label!r}. Which one "
            f"does this review's definition and measurement window ask for?")


def _map_why(kind: str, raw: Mapping[str, Any]) -> str:
    what = ("only one of the two mapping agents proposed this dataset and no protocol rule was "
            "cited to exclude it" if kind == "include_dataset"
            else "this outcome's map names two measures and the review's window fits both")
    quotes = "; ".join(f"“{_short(q, 300)}”" for q in (raw.get("quotes") or []) if str(q).strip())
    return (f"Decided at the map stage, before anything was read: {what}, so no extraction was "
            f"bought for it — a cell read against an unsettled question is a number nobody asked "
            f"for." + (f" The map's own words: {quotes}" if quotes else ""))


# ----------------------------------------------------------------------------- answers
def _answer_kind(kind: str) -> str:
    """Which override an answer to this kind of question becomes.

    This is a contract the UI shows and `answer_to_override` must honour: before C4 it advertised
    `value` for five kinds that in fact wrote `mark_reviewed / needs_human`, which is the record
    saying "answered" about a cell nothing had changed.
    """
    return {"which_value": "value", "confirm_value": "mark_reviewed", "verifier_refuted": "value",
            "which_axis": "value", "which_series": "value", "error_bar_type": "value",
            "orientation": "orientation",
            "include_dataset": "include_dataset", "which_measure": "which_measure",
            "quote_not_found": "value", "number_unusable": "value",
            "dispersion_doubt": "value",
            "needs_group_values": "value",
            "converted_statistic": "mark_reviewed",
            "reader_contradicts_values": "value",
            "group_mapping": "mark_reviewed", "no_value": "re_extract"}.get(kind, "mark_reviewed")


def answer_to_override(question: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an answer into the override payload `append_override` validates.

    `answer` carries either `option` (a key from the question's options) or the free-text
    fields (`mean`, `dispersion_value`, `dispersion_type`, `n`, `hint`, `exclude`), plus an
    optional `note`. Whatever the reviewer chose, the justification names the question so the
    log reads as a decision about a stated uncertainty, not a bare number.
    """
    kind = str(question.get("kind") or "other")
    base = {"paper_id": question.get("paper_id", ""), "dataset_id": question.get("dataset_id", ""),
            "outcome_key": question.get("outcome_key", ""), "group": question.get("group"),
            "question_id": question.get("id", "")}
    note = str(answer.get("note") or "").strip()
    stem = f"answered question #{question.get('number', '?')} ({kind})"
    just = f"{stem}: {note}" if note else stem
    option = next((o for o in question.get("options") or []
                   if o.get("key") == answer.get("option")), None)

    if kind == "include_dataset":
        # C7's answer is a decision about the REVIEW's protocol, not about a number: it names
        # include or exclude, and an exclusion names the rule it is made under. `exclude` is
        # checked here rather than in the generic branch below so the "exclude this" button on a
        # map question writes the map answer the extract stage reads, not a cell exclusion.
        decision = str((option or {}).get("decision") or answer.get("decision")
                       or ("exclude" if answer.get("exclude") else "")).strip().lower()
        if decision not in ("include", "exclude"):
            return {**base, "kind": "mark_reviewed", "confidence": "needs_human",
                    "justification": f"{just} — no decision given"}
        rule = str((option or {}).get("rule") or answer.get("rule") or "")
        return {"kind": "include_dataset", "paper_id": question.get("paper_id", ""),
                "dataset_id": question.get("dataset_id", ""),
                "question_id": question.get("id", ""), "decision": decision,
                "rule": rule, "quote": str(answer.get("quote") or ""), "note": note,
                "justification": f"{just} — {decision}" + (f", under {_short(rule, 200)!r}"
                                                           if rule else "")}
    if answer.get("exclude"):
        return {**base, "kind": "exclude_dataset", "justification": f"{just} — excluded"}
    if kind == "which_measure":
        # C6's answer names the winner the way the map named it: the metric a value location
        # carries, and where that location is. Both travel, because a map may carry the same
        # metric at two locations and the same location under two metrics.
        metric = str((option or {}).get("analysis_metric")
                     or answer.get("winning_analysis_metric") or "")
        location = str((option or {}).get("location") or answer.get("winning_location") or "")
        if not metric and not location:
            return {**base, "kind": "mark_reviewed", "confidence": "needs_human",
                    "justification": f"{just} — no measure chosen"}
        return {"kind": "which_measure", "paper_id": question.get("paper_id", ""),
                "dataset_id": question.get("dataset_id", ""),
                "question_id": question.get("id", ""),
                "outcome_key": question.get("outcome_key", ""),
                "winning_analysis_metric": metric, "winning_location": location, "note": note,
                "justification": f"{just} — {_short(metric or location, 200)}"}
    if kind == "orientation":
        # scoped to the MEASURE, never to one group's cell: `higher_is_better` is decided once per
        # (paper, outcome, measure) and copied onto every cell that used it.
        direction = option.get("higher_is_better") if option is not None \
            else answer.get("higher_is_better")
        if isinstance(direction, str):
            direction = direction.strip().lower() in ("true", "yes", "higher", "1")
        if direction is None:
            return {**base, "kind": "mark_reviewed", "confidence": "needs_human",
                    "justification": f"{just} — no direction given"}
        return {**{k: v for k, v in base.items() if k != "group"}, "group": None,
                "kind": "orientation", "higher_is_better": bool(direction),
                "measure_name": question.get("measure_name") or "",
                "quote": note or str((option or {}).get("label") or ""),
                "justification": f"{just} — higher_is_better = {bool(direction)}"
                                 f"{(': ' + (option or {}).get('label', '')) if option else ''}"}
    # "it is not reported" is a decision wherever it is offered, and its consequence is that the
    # cell leaves the analysis with a stated reason — not a note saying a human looked.
    if option is not None and option.get("key") == "not_reported":
        return {**base, "kind": "exclude_dataset",
                "justification": f"{just} — not reported in the paper; excluded"}
    if kind == "converted_statistic" and option is not None \
            and option.get("key") == "not_usable":
        return {**base, "kind": "exclude_dataset",
                "justification": f"{just} — the printed statistic does not contrast these two "
                                 f"groups and the paper prints no group values; excluded"}
    if kind == "dispersion_doubt" and option is not None \
            and option.get("key") == "within_subject":
        return {**base, "kind": "exclude_dataset",
                "justification": f"{just} — a within-subject error bar is not a between-group "
                                 f"denominator, and the paper prints no other; excluded"}
    if option is not None and option.get("dispersion_type") and option.get("mean") is None:
        # naming what the error bars ARE is a decision about the spread's type and nothing else:
        # `resolve_effect` converts it and the row is re-derived, which is what decides whether the
        # screen that raised the question still stands.
        return {**base, "kind": "value", "dispersion_type": option["dispersion_type"],
                "clears": list(option.get("clears") or []),
                "mean": None, "dispersion_value": None, "n": None,
                "justification": f"{just} — {option['label']}"}
    if option is not None and option.get("mean") is None \
            and (option.get("clears") or option.get("overrules")):
        # a decision that names what it answers and changes no number: it retires those findings
        # and nothing else, and the bucket is DERIVED from what is left rather than stamped (§C4).
        return {**base, "kind": "mark_reviewed", "clears": list(option.get("clears") or []),
                "overrules": list(option.get("overrules") or []),
                "justification": f"{just} — {option['label']}"}
    if kind == "reader_contradicts_values" and option is not None \
            and option.get("key") == "groups_swapped":
        return {**base, "kind": "exclude_dataset",
                "justification": f"{just} — the two groups are swapped; excluded pending "
                                 f"re-mapping"}
    if kind == "no_value" or answer.get("hint"):
        hint = str(answer.get("hint") or note or "").strip()
        if hint:
            return {**base, "kind": "re_extract", "hint": hint, "justification": just}
        return {**base, "kind": "mark_reviewed", "confidence": "needs_human",
                "justification": f"{just} — not reported"}
    if kind == "group_mapping":
        if option is not None and option.get("key") == "swapped":
            return {**base, "kind": "exclude_dataset",
                    "justification": f"{just} — groups swapped; excluded pending re-mapping"}
        return {**base, "kind": "mark_reviewed", "confidence": "accept_with_note",
                "justification": f"{just} — mapping confirmed"}
    if kind == "which_series" and option is not None and option.get("key") == "other_series":
        return {**base, "kind": "exclude_dataset",
                "justification": f"{just} — value belongs to the other series; excluded "
                                 f"pending re-extraction"}
    # value-shaped answers: a chosen option's number, or a typed one. `which_axis` and
    # `which_series` land here too — their options carry the number the answer implies, so
    # choosing a ladder or confirming a series writes that number rather than a bare note.
    mean = option.get("mean") if option is not None else answer.get("mean")
    if mean is not None or answer.get("dispersion_value") is not None or answer.get("n"):
        if option is None and not note:
            # a picked option carries the candidate's backers, page and quote; a typed number
            # carries nothing at all, so the reviewer's own words are its only provenance and the
            # log refuses it without them.
            from ..pipeline.overrides import OverrideRejected

            raise OverrideRejected("a typed value needs a note saying where the number comes "
                                   "from — that note is the only provenance it will ever have")
        payload = {**base, "kind": "value",
                   "clears": list((option or {}).get("clears") or []),
                   "overrules": list((option or {}).get("overrules") or []),
                   "mean": mean,
                   "dispersion_value": (option or {}).get("dispersion_value")
                   if option is not None else answer.get("dispersion_value"),
                   "dispersion_type": (option or {}).get("dispersion_type")
                   if option is not None else answer.get("dispersion_type"),
                   "n": (option or {}).get("n") if option is not None else answer.get("n"),
                   "unit": question.get("unit") or "",
                   "justification": f"{just} — {(option or {}).get('label') or 'typed value'}"}
        return payload
    if kind == "confirm_value" and answer.get("option") == "yes":
        return {**base, "kind": "mark_reviewed", "confidence": "accept_with_note",
                "justification": f"{just} — confirmed"}
    return {**base, "kind": "mark_reviewed", "confidence": "needs_human", "justification": just}


# ----------------------------------------------------------------------------- output
def write_questions(run_dir: str | Path, questions: Sequence[Mapping[str, Any]] | None = None
                    ) -> dict[str, Path]:
    """`questions.json` and a readable `questions.md` in the run directory."""
    run = Path(run_dir)
    qs = list(questions) if questions is not None else questions_for_run(run)
    json_path = run / "questions.json"
    json_path.write_text(json.dumps(qs, ensure_ascii=False, indent=1, default=str),
                         encoding="utf-8")
    md = ["# Questions for the reviewer", "",
          f"{len(qs)} cell(s) need a decision. Each shows the picture the tool read, the answers "
          f"it is choosing between, and why it could not decide. Answer in the review tab, or by "
          f"appending to `overrides.jsonl` (`canopy validate` re-pools).", ""]
    for q in qs:
        head = f"## {q['number']}. {_md(q['paper'])} — {_md(q['dataset_label'] or q['dataset_id'])}"
        if q.get("group_label"):
            head += f" — {_md(q['group_label'])}"
        md += [head, "", f"**{_md(q['prompt'])}**", ""]
        if q.get("image", {}).get("path"):
            md += [f"![{q['kind']}]({q['image']['path']})", ""]
        for o in q.get("options") or []:
            extra = f" — backed by {_md(', '.join(o['backed_by']))}" if o.get("backed_by") else ""
            md.append(f"- **{_md(o['label'])}**{extra}")
        if q.get("free_text"):
            md.append("- _(or type the value / where to find it)_")
        md += ["", f"<details><summary>why the tool could not decide</summary>", "",
               f"{_md(q['why'])}", "", "</details>", ""]
        if q.get("answered"):
            waiting = (f" — **not applied yet**: {_md(str(q.get('pending_why') or ''))}"
                       if q.get("status") == PENDING_RERUN else "")
            md += [f"_Answered: {_md(q['answers'][-1].get('justification', ''))}{waiting}_", ""]
    (run / "questions.md").write_text("\n".join(md), encoding="utf-8")
    return {"json": json_path, "md": run / "questions.md"}


def _md(text: Any) -> str:
    """Model-written prose, neutralised for markdown.

    Everything on this page came out of a paper or a model: a reason containing `](…)` or a raw
    tag renders as a link or as markup in whatever reads `questions.md`, which is a file people
    share. The SPA never renders markdown, so this is not a page vector — it is the artefact
    being what it says it is.
    """
    out = str(text or "")
    for bad, good in (("\\", "\\\\"), ("[", "\\["), ("]", "\\]"), ("`", "\\`"),
                      ("<", "&lt;"), (">", "&gt;")):
        out = out.replace(bad, good)
    return out


# ----------------------------------------------------------------------------- readers
def _overrides(run: Path) -> list[dict[str, Any]]:
    path = run / "overrides.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _json_if_present(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def _stage(run: Path, paper_id: str, stage: str) -> dict[str, Any]:
    if not paper_id:
        return {}
    return _json_if_present(paper_dir(run, paper_id) / f"{stage}.json") or {}


def _verdict(run: Path, paper_id: str, dataset_id: str, outcome_key: str,
             group: str | None) -> dict[str, Any]:
    for v in _stage(run, paper_id, "verify").get("verdicts") or []:
        if (v.get("dataset_id") == dataset_id and v.get("outcome_key") == outcome_key
                and (group is None or v.get("group") in (group, None))):
            return v
    return {}


def _candidates(run: Path, paper_id: str, dataset_id: str, outcome_key: str,
                group: str | None) -> list[dict[str, Any]]:
    extract = _stage(run, paper_id, "extract")
    verify = _stage(run, paper_id, "verify")
    every = [*(extract.get("candidates") or []), *(verify.get("extra_candidates") or [])]
    return [c for c in every if c.get("dataset_id") == dataset_id
            and c.get("outcome_key") == outcome_key
            and (group is None or c.get("group") == group)]


def _dataset(run: Path, paper_id: str, dataset_id: str
             ) -> tuple[dict[str, Any], dict[str, Any]]:
    study = (_stage(run, paper_id, "map").get("study") or {})
    dataset = next((d for d in study.get("datasets") or [] if d.get("dataset_id") == dataset_id),
                   {})
    return study, dataset


def _measure(dataset: Mapping[str, Any], outcome_key: str) -> str:
    """What the mapper called the thing this outcome measures, as written in the study map."""
    for outcome in dataset.get("outcomes") or []:
        if outcome.get("outcome_key") == outcome_key:
            return str(outcome.get("measure_name") or "")
    return ""


def _group_label(dataset: Mapping[str, Any], group: str | None) -> str:
    if group is None:
        return ""
    return str(((dataset.get("group_a" if group == "A" else "group_b") or {}).get("label")) or "")


def _where(candidates: Sequence[Mapping[str, Any]]) -> str:
    for c in candidates:
        loc = c.get("locator") or (c.get("pixel_provenance") or {}).get("figure_id")
        if loc:
            page = c.get("page")
            loc = str(loc)
            if len(loc) > 70:
                loc = loc[:67].rsplit(" ", 1)[0] + "…"
            return f"{loc}" + (f" (p. {page})" if page else "")
    return ""


def _unit(candidates: Sequence[Mapping[str, Any]], dataset: Mapping[str, Any],
          outcome_key: str) -> str:
    def short(unit: Any) -> str:
        text = str(unit or "").strip()
        for sep in (";", " (", ","):                     # "degrees (CCW…); also percentage…"
            text = text.split(sep, 1)[0].strip()
        return text[:24]

    for c in candidates:
        if c.get("unit"):
            return short(c["unit"])
    for o in dataset.get("outcomes") or []:
        if o.get("outcome_key") == outcome_key and o.get("units"):
            return short(o["units"])
    return ""


def _x_hint(candidates: Sequence[Mapping[str, Any]]) -> str:
    def first(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            value = next((v for v in value if str(v or "").strip()), "")
        return str(value or "").strip()[:60]

    for c in candidates:
        pp = c.get("pixel_provenance") or {}
        for key in ("x_read", "late_window_x_read"):
            if first(pp.get(key)):
                return first(pp[key])
        rs = (pp.get("route_sample") or {}).get("extra") or {}
        if first(rs.get("x_read")):
            return first(rs["x_read"])
    return ""


def _image(run: Path, candidates: Sequence[Mapping[str, Any]],
           provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The picture the reviewer needs: the overlay if there is one (its marks show where every
    route landed), else the figure crop, else the page crop behind a text quote. Paths in the
    records are written relative to the working directory of the run, or absolute; the result is
    made relative to the run directory so the server can serve it and the markdown can link it.

    THIS run's copy of the artefact wins, always. A run directory that has been moved or copied —
    which is what serving a finished run from anywhere but the machine that made it means — still
    holds the paths the original was written with, and those may well still resolve, to the
    *other* run's files. Showing a reviewer a picture from another directory is showing them
    evidence that is not this run's; and the server refuses to serve anything outside the run, so
    it renders as no picture at all.
    """
    best: dict[str, Any] = {}

    def here(raw: Any) -> Path | None:
        """The longest tail of a recorded path that exists inside this run directory."""
        parts = Path(str(raw)).parts
        for start in range(len(parts)):
            candidate = run.joinpath(*parts[start:])
            try:
                if candidate.exists() and candidate.resolve().is_relative_to(run.resolve()):
                    return candidate
            except OSError:                                    # pragma: no cover - defensive
                continue
        return None

    def offer(raw: Any, rank: int, kind: str) -> None:
        nonlocal best
        if not raw or (best and best["rank"] <= rank):
            return
        path = Path(str(raw))
        found = here(raw) or next((c for c in (path, Path.cwd() / path) if c.exists()), None)
        if found is not None:
            best = {"rank": rank, "path": _rel(found, run), "kind": kind}

    for c in candidates:
        offer(c.get("overlay_path"), 0, "overlay")
        offer((c.get("pixel_provenance") or {}).get("overlay_path"), 0, "overlay")
        offer(c.get("crop_path"), 1, "crop")
        fig_id = (c.get("pixel_provenance") or {}).get("figure_id")
        if fig_id:
            offer(paper_dir(run, str(c.get("paper_id") or "")) / "ingest" / "figures"
                  / f"{fig_id}.png", 2, "figure")
        entry = provenance.get(c.get("candidate_id") or "", {}) or {}
        offer(entry.get("crop"), 3, "page")
    if best:
        best.pop("rank", None)
    return best


def _rel(path: Path, run: Path) -> str:
    try:
        return str(path.resolve().relative_to(run.resolve()))
    except ValueError:
        return str(path)


def _why(entry: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    reason = str(entry.get("reason") or "").strip()
    return reason or "; ".join(verdict.get("confidence_reasons") or []) or "held for review"


def _route_name(c: Mapping[str, Any]) -> str:
    cid = str(c.get("candidate_id") or "")
    if ":digitize:" in cid:
        tail = cid.split(":digitize:", 1)[1]
        return {"ensemble": "figure (ensemble)", "raster_cv": "figure (pixels)",
                "vlm_coords": "figure (model coordinates)"}.get(
            tail.split(":")[0], f"figure ({tail.split(':')[0]}"
                                f"{', ' + tail.split(':')[1] if ':' in tail else ''})")
    return str(c.get("route") or "text") + (f" ({c.get('model')})" if c.get("model") else "")


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value) or "").upper()


def _fmt(value: float) -> str:
    return f"{value:.4g}" if abs(value) < 1000 else f"{value:.0f}"
