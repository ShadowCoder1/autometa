"""Turn every held cell into a question a person can answer from a screenshot.

The review queue says *that* a cell is held and lists its candidates. A reviewer does not want a
list; they want to be shown the picture and asked the one thing the tool could not settle:
"which of these two numbers is the older group's bar?", "is this the left or the right axis?",
"what do the error bars show?". Every question carries the screenshot the tool itself read, the
answers it is choosing between, and the override its answer becomes — so answering it is a
recorded decision, not a note in the margin.

One decision, one card (DECISION §C1). The questions are built per CELL — that is where the
evidence is — and then folded into the decision a person actually takes: both groups of a dataset
are read off one picture, the direction of a measure is settled once for the paper, a whole paper's
eligibility is one answer. A card keeps its members, its slots and their options, so a fold can
never hide a hold; `questions_for_run(..., fold=False)` is the unfolded path underneath it.

Nothing here calls a model. It reads the run's own stage files and writes `questions.json` and
`questions.md` beside them; the server serves the same list and accepts answers. What a card says
an answer WOULD do — the effect size a combination implies, whether it moves the pooled estimate —
is produced by the run's own row builder over copied verdicts, never by arithmetic invented here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from ..pipeline.overrides import (GROUP_STATISTICS, MAP_KINDS, ORIENTATION_ANSWERED, ROW_REFUSALS,
                                  codes_cleared_by_value, consumed_seqs)
from ..pipeline.rows import converted_route
from ..pipeline.state import paper_dir, read_json, sha12

__all__ = ["Question", "questions_for_run", "write_questions", "answer_to_override",
           "answers_to_overrides", "QUESTION_KINDS", "PAIRABLE", "fingerprint"]

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
    # ---- §C1: the CARDS. A card is one decision, and the decision is not always a cell: a
    # dataset's two groups are read off one picture, a direction belongs to a measure, an
    # eligibility to a paper, an analysed n to two arms. The per-cell kinds above are still what
    # the page is built from — `_consolidate` folds them, and every card keeps its members.
    "pair",               # both groups of one dataset/outcome, decided together
    "cell",               # everything still open on one cell, one slot per question, answered
                          # one at a time (the second fold — what a pair could not combine)
    "precedence_override",  # D1: this row was built from a candidate pair, not the printed value
    "analysed_n",         # D4-lite: the n the row was divided by is a recruited count
    "include_paper",      # C3: a paper the MAPPER excluded — does this review include it?
)

#: the per-cell kinds a dataset's two groups may be decided together in (§C1). Every one of them
#: is answered by naming a NUMBER (or by confirming the one in hand), which is what makes a
#: combination of the two an answerable thing. `verifier_refuted` is deliberately absent: its
#: contract has no per-slot objection option, so folding it would hide a hold (ruling e).
PAIRABLE: tuple[str, ...] = ("which_value", "which_axis", "confirm_value", "which_series",
                             "error_bar_type")

#: how many combinations a pair card may offer before it stops being one screen a person reads
_MAX_COMBINATIONS = 9

#: §C4's low-impact band: every option moves the ROW's d by less than this…
LOW_IMPACT_D = 0.10
#: …and the POOLED estimate by less than this. Both, because either alone is half the question:
#: a row whose d barely moves can still be the row that decides a k = 2 pool, and a big swing on
#: a row with almost no weight is not a reason to put a reviewer's time there.
LOW_IMPACT_POOLED = 0.05

#: D1's own flag, `pipeline.resolve.PRECEDENCE_OVERRIDE`: this row was built from a candidate
#: pair because the value the precedence list preferred converts to nothing. It is a RECORD flag,
#: never a `CheckFlag` code, so it reaches this module through the row and not through a cell.
PRECEDENCE_OVERRIDE_FLAG = "precedence_override"
#: the code `verify.checks.n_before_exclusions` raises on a candidate whose n is a recruited count
N_BEFORE_EXCLUSIONS = "n_before_exclusions"

#: the exclusion reasons `pipeline.run` writes when the MAPPER — not a person, and not the
#: resolver — is why a paper contributes nothing. Those are the two decisions §C3 turns back into
#: a question; every other exclusion in the table was made by someone or something a card cannot
#: overrule.
MAPPER_EXCLUSIONS: tuple[str, ...] = ("not_eligible", "no_usable_data:no_datasets_mapped")

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
    #: D2: every reading of this cell came off a panel the caption gives to another group, and
    #: this group has none of its own. That is the same doubt as a transposed series — which thing
    #: in this picture is this group? — and it has the same two answers, so it asks the same
    #: question. `confirm_value` would be the wrong terminus: its answer is a note, and a note
    #: does not settle whose number this is.
    ("locator_panel_mismatch", "which_series"),
    ("axis_conflict", "which_axis"),
    ("calibration_disputed", "which_axis"),
    ("calibration_refuted", "which_axis"),
    ("value_outside_axis", "which_value"),
    ("sign_mismatch", "which_value"),
    #: D2: two places in one figure were read and they gave two numbers, so the vote refused to
    #: average them. Naming the right one is the whole answer.
    ("locator_reads_conflict", "which_value"),
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


def _freshest_queue(run: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The review queue as most recently written — the queue file or the manifest's copy."""
    from_manifest = list(manifest.get("human_review_queue") or [])
    queue_file = run / "human_review_queue.json"
    manifest_file = run / "manifest.json"
    if queue_file.exists():
        newer = (not manifest_file.exists()
                 or queue_file.stat().st_mtime >= manifest_file.stat().st_mtime)
        if newer or not from_manifest:
            return list(_json_if_present(queue_file) or [])
    return from_manifest


# ----------------------------------------------------------------------------- building
def questions_for_run(run_dir: str | Path, *,
                      queue: Sequence[Mapping[str, Any]] | None = None,
                      fold: bool = True) -> list[Question]:
    """Every open decision of a finished run, as cards, worst first (biggest |Δ pooled| on top).

    `queue` is the review queue to build from. The pipeline passes the one it has just built:
    while a run's outputs are being written, `manifest.json` on disk is still the PREVIOUS
    pass's manifest, so reading the queue off it wrote a resumed run's questions from the queue
    of the run before it (three Bock questions for a nine-cell queue). Without `queue`, the
    freshest record on disk is used: the queue file when it is at least as new as the manifest,
    else the manifest's copy.

    `fold` is §C1. A reviewer reads a figure once and settles the row; a direction is decided once
    per measure. So the per-cell questions below are what the page is BUILT from and `_consolidate`
    is what it SHOWS — one card per decision, each keeping its members, its slots and their
    options. `fold=False` is the unfolded path: the cells themselves, which is what a test about a
    cell's own question, and anything that indexes by group, is actually about.
    """
    run = Path(run_dir)
    manifest = _json_if_present(run / "manifest.json") or {}
    if queue is None:
        queue = _freshest_queue(run, manifest)
    queue = list(queue)
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
                          or str(f.get("code") or "") not in _ORIENTATION_ASKING)]
            verdict = {**verdict, "flags": flags}
            if settled is not None:
                verdict["higher_is_better"] = settled
        row = rows.get((dataset_id, outcome_key)) or {}
        question = _question(entry, verdict, candidates, study, dataset, provenance,
                             run, already, pending, answered_value, consumed, overruled,
                             row, settled)
        # what `_question` itself counted as holding this cell: the verdict's codes AND the row's
        # own refusals (line ~479). Read the same two halves here. Reading only the verdict's
        # meant the two ends of one function disagreed about what was still open, and the half
        # dropped was the row's — so a cell whose ROW the resolver had refused could have its last
        # card suppressed as settled, leaving the row in neither analysis line with nothing to
        # answer anywhere.
        row_holds = {str(code) for code in row.get("flags") or [] if str(code) in ROW_REFUSALS}
        if question["kind"] in _ASKED_ONCE and _value_settled(already) \
                and not _overrulable(verdict or {}, overruled) \
                and not _holding_codes(verdict or {}) \
                and not row_holds:
            # §C4's terminus, enforced: a number a person typed and then confirmed is not asked
            # about again, and the card does not stay on the page as a settled one either.
            # Everything else this cell may be held by is a DIFFERENT question and still asked —
            # a confirmation names the value and never the finding beside it — which is why the
            # suppression waits until nothing nameable is left unaddressed on the cell — neither
            # a finding no flag code names (`_overrulable`) nor one that does (`_holding_codes`:
            # an error, a contradiction, a cap). A confirmation names the number and never the
            # `sign_mismatch` beside it, so a cell held by a code keeps its question. A repeated
            # question is visible and a vanished one is not, which makes "held with nothing to
            # answer" the worse of the two failures, not the safer one.
            continue
        out.append(question)
    out.extend(_excluded_questions(run, overrides, out))
    out.extend(_map_questions(run, overrides, pending, consumed))
    # ONE read of the run's rows for the whole page. Every card asks its row where it stands (the
    # best-guess line's rule or veto, D1's flag), and reading that per card re-globbed every
    # `resolve.json` and the whole table once per card — 25 full scans and 432 file reads for one
    # nine-paper page, growing as O(cards x papers). Threaded rather than memoised on the module,
    # because the map has to be re-read after a re-pool and a cache keyed on mtimes is a harder
    # thing to get right than an argument.
    row_map = _rows_of(run)
    cards = _consolidate(out, run, overrides, pending, consumed, row_map) if fold else \
        [_as_cell_card(q, run, overrides, row_map) for q in out]
    cards.sort(key=_rank)
    for i, q in enumerate(cards, 1):
        q["number"] = i
    return cards


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


#: the kinds that change the NUMBER a row carries, so a confirmation made before one of them was
#: a confirmation of a different number: this cell's own value, the analysed group sizes the row is
#: divided by, and the direction it is signed with. Anything else — a note, a `mark_reviewed` that
#: names a finding, a decision about a sibling — leaves the confirmed value exactly as it was.
_CHANGES_THE_VALUE: frozenset[str] = frozenset({"value", "group_n", "orientation"})


#: the kinds that ask "is this the number?" — the ones a confirmation has already answered. A
#: refutation, an adjudication or an unresolved direction is not among them: those are findings a
#: confirmation did not name, and §C4 is explicit that an answer settles only what it names.
_ASKED_ONCE: frozenset[str] = frozenset({"confirm_value", "which_value", "needs_group_values",
                                         "converted_statistic", "no_value"})


def _value_settled(already: Sequence[Mapping[str, Any]]) -> bool:
    """Has a person typed this cell's value AND confirmed it, with nothing changing it since?

    §C4's terminus, enforced rather than merely offered. `confirm_value` is the last question a
    held cell asks and its answer is a `mark_reviewed` that accepts the number — but every re-pool
    rebuilds the row from the cells, so the same rule met the same row again and raised the same
    card. Langan's late-adaptation cell and its aftereffect, and Heuer 2008's aftereffect pair,
    came back as fresh "are you sure?" questions after every answer round in the same run: a
    reviewer who answers them is answering for ever, and a page that keeps asking a settled
    question is indistinguishable from one that lost the answer.

    A LATER record that changes the number re-opens it (`_CHANGES_THE_VALUE`), because the
    confirmation was about the value that stood when it was made and a new one has been confirmed
    by nobody. Read from the LOG, which is the only durable record of it: the stage files a
    re-pool reads are never rewritten and always say what the run decided.
    """
    typed = confirmed = False
    for override in already:
        kind = str(override.get("kind") or "")
        if kind == "mark_reviewed" and override.get("confidence") == "accept_with_note":
            confirmed = confirmed or typed
        elif kind in _CHANGES_THE_VALUE:
            typed, confirmed = kind == "value" or typed, False
    return confirmed


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
              row: Mapping[str, Any] | None = None,
              recorded_direction: bool | None = None) -> Question:
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
    # …and the ids of the CARDS this cell can be folded into (§C1). An answer given on the page is
    # recorded against the card the reviewer was shown, and a cell that compared only against its
    # own id saw none of them — so a card answer left no trace on the cell it settled and the
    # cell's history read as though nobody had ever been here.
    #
    # A card id is KIND-FREE, which is the whole difference: this cell's own id carries the kind it
    # was asked (`…|which_axis`), so it can never match a later question; `…||pair` matches every
    # question this cell will ever ask. So a card id is admitted on the STRICTER test below — the
    # record must NAME what it settled — and never on "it wrote the kind this question writes",
    # which would tick off the next question with the answer to the last one.
    fold_ids = {f"{entry.get('dataset_id') or ''}|{outcome_key}||pair",
                f"{entry.get('dataset_id') or ''}|{outcome_key}||precedence_override",
                _measure_id(str(entry.get("paper_id") or ""), outcome_key, measure)}
    # (a `cell` card's slot answers name the SLOT's own question, never the card — see
    # `_slot_record` — so the card id is not one a cell accepts answers under)
    writes = _answer_kind(kind)
    # which of a group's own statistics the log's value records have put on THIS cell, taken
    # together: the FIELDS, not the numbers, because what the tail of `answers_this` has to know is
    # whether a person has supplied the whole statistic — not in which submit they supplied it.
    typed = {field for o in already if o.get("kind") == "value"
             for field in GROUP_STATISTICS if o.get(field) is not None}

    def answers_this(override: Mapping[str, Any]) -> bool:
        if override.get("kind") == "exclude_dataset":
            return True                     # the cell has left the analysis: nothing is still open
        if override.get("kind") in MAP_KINDS:
            return override.get("decision") == "exclude"
        if kind == "no_value" and override.get("kind") == "re_extract":
            # a hint answers "where is it?" only until the reading it asks for has been bought.
            # Once the extract stage has re-read the cell with it and the cell STILL has no usable
            # value, the question is open again: the run has now paid to look where the reviewer
            # pointed and come back with nothing, which is an absence and not an answer. Ticking
            # the card on it would be the overclaim §C4 exists to remove — a settled question on a
            # cell nothing changed about — and it would leave a `needs_human` row with no open
            # question anywhere on the page.
            return override.get("seq") not in consumed
        if override.get("question_id"):
            on_a_card = override["question_id"] in fold_ids
            if override["question_id"] != question_id and not on_a_card:
                return False
            if kind == "verifier_refuted":
                # a number does not retire a refutation: the verifier said the value is wrong, and
                # a different value is not an answer to that. Only overruling it on the record —
                # or removing the cell, handled above — settles this question.
                return "verifier_refuted" in (override.get("overrules") or [])
            named = bool(override.get("overrules")) or bool(override.get("clears"))
            if on_a_card:
                return named
            # …and answering "no, the right number is this one" is not answering "is this right?":
            # the question is settled by the decision it advertises, or by one that names what it
            # retires. Anything else changed the cell and left the question standing.
            return override.get("kind") == writes or named
        # a record written outside the questions page — the manual override form, a log older than
        # `question_id` — cannot say which question it answered, and the cell's question changes as
        # answers land. Matching it by kind marked a question the reviewer had never been shown as
        # answered, on the strength of an older decision of the same kind. It settles nothing here;
        # what it settles is on the record, and the cell asks until the page is answered.
        #
        # ONE kind is different, and only for records that name nothing — which is why this sits
        # here rather than ahead of the branch above, where it settled cards whose reviewer had
        # been shown another question entirely. `no_value` asks "where is this group's value, if it
        # is reported at all?", and a person who has typed the group's own statistics onto the cell
        # has answered exactly that, however they recorded it. The whole trio is required, and it
        # is read off the CELL rather than one record: a mean alone builds no row, so ticking the
        # card on one left a `needs_human` row with no open question anywhere on the page — the
        # state §C4 exists to remove, and worse than the question it silenced. Reading the cell
        # rather than the record is what stops the mirror image of that: a reviewer who types the
        # spread in a second submit would otherwise be asked for ever where a number they had
        # already given was. (`already` is narrowed to this cell and this group at the call site,
        # and `_validate` refuses a `value` with no group, so nothing from another arm is in it.)
        return (kind == "no_value" and override.get("kind") == "value"
                and set(GROUP_STATISTICS) <= typed)

    # the hints this cell's re-reading was actually bought for, before `already` is narrowed to the
    # answers that SETTLE the question — a consumed hint no longer settles a `no_value` (above),
    # and the reviewer still has to be told what their hint bought.
    bought = [o for o in already
              if o.get("kind") == "re_extract" and o.get("seq") in consumed]
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
                     measure, overruled, flags)
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
    if kind == "no_value" and bought:
        why = _reread_why(bought, why)
    if kind == "orientation":
        why = _with_ballots(why, run, str(entry.get("paper_id") or ""), outcome_key, measure)
    if kind == "reader_contradicts_values" and recorded_direction is not None:
        # first, because it is the whole reason this cell is being asked anything about direction
        # again: a direction IS on the record, and this cell still carries C3's mechanical
        # contradiction — the only one there is — between a stated direction and its own resolved
        # raw means. That is new information about the answer, not the direction question again,
        # and it is asked as its own kind with its own three answers.
        why = (f"NEW, since the direction was recorded: a reviewer has already answered the "
               f"direction of this measure (higher_is_better = {bool(recorded_direction)}), and "
               f"the run's means check contradicts the direction stated for this cell — the "
               f"recorded answer is contradicted by this cell's own resolved raw means. That is "
               f"why this is asked again, and it is a different question. || {why}")
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
        # what has been recorded on this question — including the hint a re-opened `no_value` is
        # re-opening FROM, which `answers_this` no longer counts as settling it. The two lists are
        # deliberately different: `already` decides the STATUS, this decides what the reviewer is
        # shown, and a question that came back open after a re-reading is not a question nobody
        # has ever been here for.
        ("answers", [{"kind": o.get("kind"), "justification": o.get("justification"),
                      "mean": o.get("mean"), "at": o.get("at") or o.get("timestamp")}
                     for o in _history(already, bought)]),
    ]))


def _history(already: Sequence[Mapping[str, Any]],
             bought: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The records to SHOW on a card: what settles it, plus the acted-on hints that no longer do.

    In log order, and never twice — a record can be in both lists on a question a hint does settle.
    """
    seen = {record.get("seq") for record in already if isinstance(record.get("seq"), int)}
    extra = [record for record in bought if record.get("seq") not in seen]
    return sorted([*already, *extra],
                  key=lambda record: record.get("seq") if isinstance(record.get("seq"), int)
                  else 0)


def _reread_why(bought: Sequence[Mapping[str, Any]], why: str) -> str:
    """What a reviewer's hint bought, on the question it did not settle.

    The C8 rule, one layer up from the truncated read-out: a reading that was PAID FOR and came
    back with nothing is an absence, not evidence that the paper prints no value. So the card says
    both halves — the hint was acted on, and this is what it returned — rather than reappearing
    identical, which reads as though nobody had ever answered it.
    """
    hints = "; ".join(dict.fromkeys(
        " ".join(str(record.get("hint") or "").split()) for record in bought
        if str(record.get("hint") or "").strip()))
    return (f"NEW, since the hint was recorded: the extract stage bought a re-reading of this "
            f"cell with the reviewer's hint (“{hints[:300]}”) and the readers came back with no "
            f"usable value from it. That is an absence, not evidence that the paper prints none — "
            f"the hinted location was read and returned nothing. The question is open again "
            f"because the cell still has no number, not because the answer was ignored. || {why}")


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

#: every code whose QUESTION is the direction, so that a recorded direction retires all of them and
#: `_kind` stops returning `orientation` on a cell whose direction is on the record. It is the
#: ANALYSIS's own set (`pipeline.overrides.ORIENTATION_ANSWERED`) rather than a second list here:
#: `_apply_orientation` strips exactly these from the cell, and a page that retired a different set
#: would ask about a code the re-pool had removed, or tick off one it had not.
#: `tests/test_questions_consolidation.py` pins that every `_FLAG_TO_KIND` code asking for a
#: direction is in it — a new one that is not would be asked for ever, which is the bug this fixes.
_ORIENTATION_ASKING: frozenset[str] = ORIENTATION_ANSWERED


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
                                           "series_identity_conflict", "series_transposed",
                                           "locator_panel_mismatch"),
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
                *({**option, "clears": _present(flags, "value_outside_axis", "sign_mismatch",
                                                 "locator_reads_conflict")}
                  for option in _value_options(valued, verdict, unit))]
    if kind == "which_value":
        # naming the right number is the answer to "that number is off the ladder", to "the paper
        # says the other group was higher", and to "two places in this figure were read and they
        # do not agree" — the findings that raise this question.
        return [{**option, "clears": _present(flags, "value_outside_axis", "sign_mismatch",
                                              "locator_reads_conflict")}
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
                           "clears", "overrules",
                           # §C1: a FOLDED option's key is positional twice over (`a{i}|b{j}`),
                           # so the numbers each slot contributes are part of what the option
                           # means. Without them the same key under the same card id could denote
                           # a different pair of numbers after any answer, and the server's echo
                           # check — the whole point of this hash — would wave the stale one
                           # through. `n_a`/`n_b` do the same for an analysed-n card.
                           "slots", "n_a", "n_b")},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _stamped(options: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**option, "fingerprint": fingerprint(option)} for option in options]


#: markup a model sometimes emits inside a free-text field — closing tags, tool-call fragments,
#: stray parameter wrappers. It is not part of what the model meant to say, and a reviewer being
#: asked "which ladder was this read against?" should not be shown `</antml_parameter> <parameter
#: name="axis_read">` in the middle of the answer (nine-paper run, question #15).
_MARKUP = re.compile(r"</?[A-Za-z_][^>]{0,200}>")


def _short(text: str, limit: int) -> str:
    text = " ".join(_MARKUP.sub(" ", str(text or "")).split())
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



def _confirmed_value(options: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
                     unit_suffix: str) -> str:
    """What a `confirm_value` question is asking about: the value its `yes` option confirms.

    The head option carries it inside its label ("yes — 30.2 degrees is right, …"), which is
    where `_options` put the run's own resolved reading; the fallbacks are the verdict's mean and
    then any candidate that named one, so a question is never posed about a number no option
    offers.
    """
    head = next((o for o in options if o.get("key") == "yes"), None)
    if head is not None:
        label = str(head.get("label") or "")
        if label.startswith("yes — ") and " is right" in label:
            spoken = label[len("yes — "):label.index(" is right")].strip()
            if spoken and spoken != "this value":
                return spoken
    if verdict.get("mean") is not None:
        return _fmt(float(verdict["mean"])) + unit_suffix
    return next((str(o["label"]) for o in options if o.get("mean") is not None), "this value")


def _prompt(kind: str, label: str, outcome_key: str, where: str, unit: str, x_hint: str,
            options: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
            measure: str = "", overruled: Collection[str] = (),
            flags: Sequence[str] = ()) -> str:
    who = f"the {label} group" if label else "this group"
    at = f" at {x_hint}" if x_hint else ""
    u = f" ({unit})" if unit else ""
    src = f" in {where}" if where else ""
    outcome = outcome_key.replace("_", " ")
    raised = set(flags)
    if kind == "which_value":
        # a question states the finding that raised it, or a reviewer is told the routes disagree
        # about a cell where one route read two places and the routes never met (D2).
        if "locator_reads_conflict" in raised:
            return (f"Which of these is {who}'s {outcome}{at}{src}{u}? Two places in the same "
                    f"figure were read and they give different numbers, so neither corroborates "
                    f"the other and nothing was averaged.")
        return (f"Which of these is {who}'s {outcome}{at}{src}{u}? The routes that read it "
                f"disagree.")
    if kind == "confirm_value":
        # the number the question asks about is the one the `yes` option confirms — the value
        # this run RESOLVED — never a rival candidate and never the bare words "this value".
        # `_options` builds the head from `_value_options(...)[0]`; reading it back from the head
        # keeps the two in step, so the prompt cannot ask "is 30.57 right?" beside a button that
        # says "yes, 30.2 is right" (nine-paper run, questions #1 and #21).
        value = _confirmed_value(options, verdict, u)
        return (f"Is {value} {who}'s {outcome}{at}{src}? This cell is held because "
                f"{_held_because(verdict, overruled)}. Answering yes records that you have "
                f"checked it, and "
                f"overrules exactly that.")
    if kind == "which_axis":
        return (f"Which of these is {who}'s {outcome}{at}{src}{u}? The readers and the axis "
                f"ladder disagree about the scale, so each answer names the ladder it was read "
                f"against — pick the number the right ladder gives.")
    if kind == "which_series":
        # same question, same two answers; what differs is WHY the identity is in doubt — a
        # marker the pixel pass could not match, or a panel the caption gives to another group.
        if "locator_panel_mismatch" in raised:
            return (f"In {where or 'this figure'}, which plotted series is {who}? Every reading "
                    f"of this value was taken off a panel the caption gives to another group, and "
                    f"this group has no reading from its own panel, so the number below may "
                    f"belong to the other group.")
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
            "group_mapping": "mark_reviewed", "no_value": "re_extract",
            # §C1's cards. `pair` is decided by its slots, so what it writes is what they write
            # (`_pair_card` narrows it when they agree); the other three write one kind each.
            "pair": "value", "precedence_override": "value",
            "analysed_n": "group_n", "include_paper": "eligibility",
            # the second fold's card writes what its slots write; each slot carries its own
            "cell": "value",
            }.get(kind, "mark_reviewed")


def _single_override(question: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an answer to ONE cell's question into the payload `append_override` validates.

    `answers_to_overrides` is the public door: a card that folds two cells is answered by two of
    these, one per group, and each carries only its own option's `clears`.

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


# ============================================================== §C1/C3/C4: one decision, one card
#: every key a card carries, in the order the page reads them. The first block is what a per-cell
#: question has always carried (so `write_questions`, the SPA and every reader keep working); the
#: second is §C1's — what this card IS, what it is made of, and what answering it is worth.
_CARD_KEYS: tuple[str, ...] = (
    "id", "kind", "prompt", "paper", "paper_id", "dataset_id", "dataset_label", "outcome_key",
    "measure_name", "group", "group_label", "where", "unit", "image", "options", "free_text",
    "answer_writes", "why", "confidence", "route", "impact", "answered", "status", "pending_why",
    "answers",
    "scope", "member_ids", "cells", "slots", "settled", "impact_basis", "impact_rank",
    "impact_band", "blocking_rank", "status_line", "best_guess", "slot_answers")

_CARD_DEFAULTS: dict[str, Any] = {
    "id": "", "kind": "other", "prompt": "", "paper": "", "paper_id": "", "dataset_id": "",
    "dataset_label": "", "outcome_key": "", "measure_name": "", "group": None, "group_label": "",
    "where": "", "unit": "", "image": {}, "options": [], "free_text": True,
    "answer_writes": "mark_reviewed", "why": "", "confidence": None, "route": "", "impact": None,
    "answered": False, "status": "open", "pending_why": "", "answers": [],
    "scope": "cell", "member_ids": [], "cells": [], "slots": [], "settled": [],
    "impact_basis": "unknown", "impact_rank": -1.0, "impact_band": "high", "blocking_rank": 0,
    "status_line": "", "best_guess": {},
    # §C1, second fold: a card whose `slots` are answered one at a time — each answerable slot
    # carries its own options and writes its own record, and the card stays open until every
    # slot it names is settled. `pair` cards are NOT this: their answer is one combination.
    "slot_answers": False,
}


def _blank(**fields: Any) -> Question:
    card = OrderedDict((key, copy.deepcopy(_CARD_DEFAULTS[key])) for key in _CARD_KEYS)
    card.update(fields)
    return Question(card)


def _rank(card: Mapping[str, Any]) -> tuple[bool, float, int, str]:
    """§C4's order: unanswered first, then what answering is worth, then how much it is holding.

    `impact_rank` is negative for a card nothing could measure, so an unknown can never sort above
    a known — a reviewer working down the page would otherwise spend the top of their attention on
    the cards the tool has the least to say about.
    """
    return (bool(card.get("answered")), -float(card.get("impact_rank") or -1.0),
            -int(card.get("blocking_rank") or 0), str(card.get("id") or ""))


def _consolidate(out: Sequence[Question], run: Path, overrides: Sequence[Mapping[str, Any]],
                 pending: Mapping[int, str], consumed: Collection[int] = (),
                 rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
                 ) -> list[Question]:
    """The per-cell questions, folded into the decisions they are (§C1).

    Folded AFTER building, never instead of building: every builder above is untouched, every card
    keeps its `member_ids`, and a fold that would hide a hold does not happen (a non-`PAIRABLE`
    kind, more combinations than a screen holds, a cell whose partner is not held). What changes is
    what a reviewer is shown — one decision at a time instead of one cell at a time.
    """
    preview = _Preview(run)
    rows = _rows_of(run) if rows is None else rows
    cards: list[Question] = []
    rest = list(out)

    # a paper nobody read, and a dataset's analysed size: neither is a held cell, so neither can
    # come out of the queue — they are read from the run's own exclusion table and verify stages.
    cards += _excluded_paper_questions(run, overrides, pending, consumed)
    cards += _analysed_n_questions(run, overrides, pending, consumed, rows)

    # D1 first, because its card REPLACES the value-picking cards of the cell it is about: the row
    # was built from a candidate pair, so "which of these numbers is group A's" is no longer the
    # decision — "which pair is this row" is.
    precedence, rest = _precedence_override_questions(run, rest, overrides, pending, consumed,
                                                      preview, rows)
    # …and whatever else is still asked about the row a precedence card decides (a refutation,
    # a series identity) rides on that card as its own slot, so the row is one place on the page.
    rest = _attach_to_precedence(precedence, rest, run, overrides)
    cards += precedence
    orientation, rest = _fold_orientation(rest, run, overrides, rows)
    cards += orientation
    pairs, rest = _fold_pairs(rest, run, overrides, preview, rows)
    cards += pairs
    # the second fold (§C1): everything still open on ONE cell, on one card — a refutation beside
    # the value question it objects to, both groups of a cell a pair could not combine. Each slot
    # keeps its own options and writes its own record; nothing is merged into a combination.
    cells, rest = _fold_cells(rest, run, overrides, rows)
    cards += cells
    cards += [_as_cell_card(q, run, overrides, rows) for q in rest]
    return cards


# ------------------------------------------------------------------ the cell fold
def _cell_key(question: Mapping[str, Any]) -> tuple[str, str]:
    return (str(question.get("dataset_id") or ""), str(question.get("outcome_key") or ""))


def _answerable_slot(question: Question, run: Path,
                     overrides: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {**_slot_of(question, run, overrides), "answerable": True}


def _attach_to_precedence(precedence: Sequence[Question], rest: Sequence[Question], run: Path,
                          overrides: Sequence[Mapping[str, Any]]) -> list[Question]:
    """The questions D1 does not answer but that are about D1's row — carried on D1's card.

    The card's own three answers stay exactly what they are; each attached question is a slot of
    its own, answered through `slots` in the POST, and the card is open while any of them is. A
    reviewer deciding "which pair is this row" sees the verifier's objection to that pair on the
    same card, and the objection still has to be overruled BY NAME (the standing rule) — an
    answer to the precedence decision does not quietly retire it.
    """
    keep = list(rest)
    for card in precedence:
        if str(card.get("status") or "open") != "open":
            # the decision is recorded: whatever the row's cells still ask goes back to its own
            # card (a pair, a cell card), as `_precedence_override_questions` says it does
            continue
        # the same kinds the cell fold takes, for the same reasons: a direction is the measure
        # card's (one place, one answer), a `no_value` is a typed hint, a map question is not a
        # cell's; and a slot with nothing to pick is not a slot
        mine = [q for q in keep
                if _cell_key(q) == (str(card.get("dataset_id") or ""),
                                    str(card.get("outcome_key") or ""))
                and str(q.get("kind") or "") not in _NOT_CELL_FOLDABLE
                and (q.get("options") or [])]
        if not mine:
            continue
        keep = [q for q in keep if q not in mine]
        card["slots"] = [*(card.get("slots") or []),
                         *(_answerable_slot(q, run, overrides) for q in mine)]
        card["member_ids"] = [*(card.get("member_ids") or []), *(str(q["id"]) for q in mine)]
        card["cells"] = [*(card.get("cells") or []), *(_cell_of(q) for q in mine)]
        card["settled"] = [*(card.get("settled") or []),
                           *(s for q in mine
                             for s in _settled(overrides, str(q.get("dataset_id") or ""),
                                               str(q.get("outcome_key") or ""), q.get("group")))]
        card["slot_answers"] = True
        # the card was open (checked above) and stays open: the attached questions are the
        # row's own open questions, and their history travels with them
        card["answers"] = [*(card.get("answers") or []),
                           *(a for q in mine for a in (q.get("answers") or []))]
        card["prompt"] = (f"{card.get('prompt') or ''} This row also has {len(mine)} other open "
                          f"question(s), below — each is answered on its own.").strip()
    return keep


def _fold_cells(rest: Sequence[Question], run: Path, overrides: Sequence[Mapping[str, Any]],
                rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
                ) -> tuple[list[Question], list[Question]]:
    """Every cell still asked about more than once, asked about once (§C1, second fold).

    A `pair` is the stronger fold — one combination, with the effect size each one implies —
    and it has already taken every cell it honestly could. What reaches here is the rest: a
    refutation beside a value question, two groups with more options than a combination screen
    holds, a series question beside an axis question. Those are still ONE cell a reviewer reads
    once, so they go on one card — as separate slots, each with its own options and its own
    record, never combined. A fold of slots hides nothing a fold of combinations would show: the
    slot IS the question, unchanged.
    """
    cells: "OrderedDict[tuple[str, str], list[Question]]" = OrderedDict()
    cards: list[Question] = []
    keep: list[Question] = []
    for question in rest:
        # a map question is not a cell's question: the dataset has nothing extracted yet, and its
        # answer is a decision about the review's protocol (§C6/C7) that a resume acts on
        if str(question.get("kind") or "") in _NOT_CELL_FOLDABLE:
            keep.append(question)
            continue
        cells.setdefault(_cell_key(question), []).append(question)
    for (dataset_id, outcome_key), members in cells.items():
        if len(members) < 2 or not dataset_id or not outcome_key \
                or any(not (q.get("options") or []) for q in members):
            keep.extend(members)
            continue
        cards.append(_cell_card(dataset_id, outcome_key, members, run, overrides, rows))
    return cards, keep


#: kinds the cell fold leaves alone: the map's own questions; a direction (folded per measure);
#: `no_value`, whose answer is a typed hint that buys a re-extraction, not a pick from a list
_NOT_CELL_FOLDABLE: frozenset[str] = frozenset(MAP_KINDS) | {"include_dataset", "which_measure",
                                                              "orientation", "no_value"}


def _cell_card(dataset_id: str, outcome_key: str, members: Sequence[Question], run: Path,
               overrides: Sequence[Mapping[str, Any]],
               rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> Question:
    card_id = "|".join([dataset_id, outcome_key, "", "cell"])
    # what the card writes is what its slots write, when they agree (`_pair_card` does the same)
    writes = {str(m.get("answer_writes") or "") for m in members}
    card = _folded(card_id, "cell", "dataset", list(members), run, overrides, rows,
                   options=[], prompt=_cell_prompt(members),
                   why=" || ".join(dict.fromkeys(str(m.get("why") or "") for m in members
                                                 if m.get("why"))),
                   group=None, group_label="",
                   answer_writes=(writes.pop() if len(writes) == 1 else "value"),
                   slot_answers=True)
    card["slots"] = [{**slot, "answerable": True} for slot in card["slots"]]
    return card


def _cell_prompt(members: Sequence[Mapping[str, Any]]) -> str:
    """What is still open on this cell, one clause per question, each named by its group. The
    full question travels in its slot (`slot.prompt`), which is where the page asks it."""
    said = "; ".join(
        f"{str(m.get('group_label') or ('group ' + str(m.get('group') or '?')))} — "
        f"{str(m.get('kind') or '').replace('_', ' ')}"
        for m in members)
    return (f"{len(members)} questions are still open on this cell, answered one at a time "
            f"below: {said}.")


# ------------------------------------------------------------------ the cell, as a card of one
def _as_cell_card(question: Question, run: Path,
                  overrides: Sequence[Mapping[str, Any]] = (),
                  rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> Question:
    """A question nothing folded, wearing the same shape as everything else on the page."""
    dataset_id = str(question.get("dataset_id") or "")
    outcome_key = str(question.get("outcome_key") or "")
    row = _row_of_cell(rows, run, dataset_id, outcome_key)
    card = _blank(**{key: question[key] for key in question if key in _CARD_DEFAULTS})
    card.update({
        "scope": "cell",
        "member_ids": [question["id"]],
        "cells": [_cell_of(question)],
        "slots": [_slot_of(question, run, overrides)],
        "settled": _settled(overrides, dataset_id, outcome_key, question.get("group")),
        "status_line": _status_line(row),
        "best_guess": _best_guess(row),
    })
    _stamp_impact(card, [question], row_swings=None)
    return card


def _cell_of(question: Mapping[str, Any]) -> dict[str, Any]:
    """One cell a card settles, as the page lists it under the decision."""
    return {"dataset_id": str(question.get("dataset_id") or ""),
            "outcome_key": str(question.get("outcome_key") or ""),
            "group": question.get("group"),
            "group_label": str(question.get("group_label") or ""),
            "dataset_label": str(question.get("dataset_label") or ""),
            "paper": str(question.get("paper") or ""),
            "measure_name": str(question.get("measure_name") or ""),
            "question_id": str(question.get("id") or "")}


def _slot_of(question: Mapping[str, Any], run: Path,
             overrides: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """One group's own question inside a card: its kind, its options and why it is being asked.

    The slot is the whole of the guarantee that a fold hides nothing — the kind that was folded,
    the answers that kind offered, and the reason the cell is held all travel into the card, so a
    reviewer can always see the two questions the one decision is made of.
    """
    return {"group": question.get("group"),
            "group_label": str(question.get("group_label") or ""),
            "kind": str(question.get("kind") or ""),
            "member_id": str(question.get("id") or ""),
            "dataset_id": str(question.get("dataset_id") or ""),
            "outcome_key": str(question.get("outcome_key") or ""),
            "prompt": str(question.get("prompt") or ""),
            "why": str(question.get("why") or ""),
            "answer_writes": str(question.get("answer_writes") or ""),
            "options": list(question.get("options") or []),
            "image": dict(question.get("image") or {}),
            "where": str(question.get("where") or ""),
            "unit": str(question.get("unit") or ""),
            # whether the page may answer THIS slot on its own (a `cell` card, a refutation
            # carried on a precedence card, a paper's analysed-n slots). A pair's slots are not:
            # their answer is the combination the card offers.
            "answerable": False,
            "settled": _settled(overrides, str(question.get("dataset_id") or ""),
                                str(question.get("outcome_key") or ""), question.get("group"))}


def _settled(overrides: Sequence[Mapping[str, Any]], dataset_id: str, outcome_key: str,
             group: str | None) -> list[dict[str, Any]]:
    """What this group's recorded answers have already cleared — history, with its dates.

    Never an "answered" tick: a cell whose axis question was settled last week is asked about the
    series identity today, and the card it sits in is OPEN. What the reviewer needs to see is that
    somebody has already been here and what they decided, which is exactly this list.
    """
    out: list[dict[str, Any]] = []
    for override in overrides:
        if str(override.get("dataset_id") or "") != dataset_id:
            continue
        if str(override.get("outcome_key") or "") not in ("", outcome_key):
            continue
        if override.get("group") not in (None, group):
            continue
        cleared = [str(code) for code in (override.get("clears") or []) if str(code)]
        overruled = [str(name) for name in (override.get("overrules") or []) if str(name)]
        if not cleared and not overruled:
            continue
        out.append({"clears": cleared, "overrules": overruled,
                    "kind": str(override.get("kind") or ""),
                    "at": str(override.get("at") or override.get("timestamp") or ""),
                    "actor": str(override.get("actor") or ""),
                    "justification": _short(override.get("justification"), 300)})
    return out


# ------------------------------------------------------------------ the folds
def _measure_id(paper_id: str, outcome_key: str, measure: str) -> str:
    digest = hashlib.sha1(" ".join(str(measure or "").split()).casefold().encode("utf-8"))
    return "|".join([f"{sha12(paper_id)}:m{digest.hexdigest()[:8]}", outcome_key, "", "orientation"])


def _fold_orientation(rest: Sequence[Question], run: Path,
                      overrides: Sequence[Mapping[str, Any]],
                      rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
                      ) -> tuple[list[Question], list[Question]]:
    """One direction card per (paper, outcome, normalised measure) — the override's own scope.

    `_apply_orientation` settles every dataset of a paper's outcome that measures the same thing,
    so asking the question once per cell asked one question eight times and offered eight chances
    to answer it eight different ways. The card lists every cell its one answer settles.
    """
    groups: "OrderedDict[str, list[Question]]" = OrderedDict()
    keep: list[Question] = []
    for question in rest:
        if question.get("kind") != "orientation" or not question.get("paper_id"):
            keep.append(question)
            continue
        groups.setdefault(_measure_id(str(question["paper_id"]), str(question["outcome_key"]),
                                      str(question.get("measure_name") or "")), []).append(question)
    cards: list[Question] = []
    for card_id, members in groups.items():
        head = members[0]
        cards.append(_folded(card_id, "orientation", "measure", members, run, overrides, rows,
                             options=_stamped(list(head.get("options") or [])),
                             prompt=_measure_prompt(head, members, run),
                             why=str(head.get("why") or ""),
                             group=None, group_label="",
                             answer_writes="orientation"))
    return cards, keep


def _measure_prompt(head: Mapping[str, Any], members: Sequence[Mapping[str, Any]],
                    run: Path) -> str:
    """The measure's question, the outcome as the PROTOCOL defines it, and every cell the one
    answer signs — with its raw means, because a direction is decided against the numbers.

    The ballots are already under `why` (`_with_ballots`); §C asks the card to show these three
    as well, and they are the whole of what a reviewer arbitrates a direction from: what this
    review counts as more of the construct, and which way this paper's numbers actually run.
    """
    said = "; ".join(dict.fromkeys(
        f"{m.get('dataset_label') or m.get('dataset_id')} "
        f"({m.get('group_label') or m.get('group')} = "
        f"{_raw_mean(run, str(m.get('paper_id') or ''), str(m.get('dataset_id') or ''),
                    str(m.get('outcome_key') or ''), m.get('group'))})"
        for m in members))
    outcome = _outcome_definition(run, str(head.get("outcome_key") or ""))
    return (f"{head.get('prompt') or ''} This review counts as "
            f"{str(head.get('outcome_key') or '').replace('_', ' ')}: {outcome} This one answer "
            f"settles the direction of that measure for every cell of this paper that uses it, "
            f"with the raw means each of them resolved: {said}.")


def _raw_mean(run: Path, paper_id: str, dataset_id: str, outcome_key: str,
              group: Any) -> str:
    """One cell's own resolved mean, as the verify stage recorded it."""
    verdict = _verdict(run, paper_id, dataset_id, outcome_key,
                       group if group in ("A", "B") else None)
    value = verdict.get("mean")
    return _fmt(float(value)) if isinstance(value, (int, float)) else "—"


def _outcome_definition(run: Path, outcome_key: str) -> str:
    """What the REVIEW's protocol says this outcome is. A direction cannot be decided against a
    key: "late_adaptation" is a name, and the definition is the thing being scored."""
    from ..protocol import load_protocol

    for name in ("protocol.yaml", "protocol.staged.yaml"):
        path = run / name
        if not path.exists():
            continue
        try:
            outcome = load_protocol(path).outcome(outcome_key)
        except (OSError, ValueError, KeyError):
            continue
        said = " ".join(x for x in (outcome.definition, outcome.measurement_window) if x)
        if said:
            return _short(said, 400)
    return "the review's protocol records no definition for it."


def _fold_pairs(rest: Sequence[Question], run: Path, overrides: Sequence[Mapping[str, Any]],
                preview: "_Preview",
                rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
                ) -> tuple[list[Question], list[Question]]:
    """Both groups of one dataset/outcome, decided together — when that is honest (§C1).

    Three conditions, and each of them is a way the fold could hide a hold. Both groups must be
    held (a card that folded one held cell with one released one would ask about a cell nobody is
    asking about); every kind must be `PAIRABLE` (a refutation has no per-slot answer, so its cell
    keeps its own card); and the combinations must fit on a screen, which is what the nine is.
    """
    cells: "OrderedDict[tuple[str, str], list[Question]]" = OrderedDict()
    for question in rest:
        cells.setdefault((str(question.get("dataset_id") or ""),
                          str(question.get("outcome_key") or "")), []).append(question)
    cards: list[Question] = []
    keep: list[Question] = []
    for (dataset_id, outcome_key), members in cells.items():
        by_group = {q.get("group"): q for q in members}
        if len(members) != 2 or set(by_group) != {"A", "B"} \
                or any(q.get("kind") not in PAIRABLE for q in members) \
                or not dataset_id or not outcome_key:
            keep.extend(members)
            continue
        a, b = by_group["A"], by_group["B"]
        options = list(a.get("options") or []), list(b.get("options") or [])
        if not options[0] or not options[1] \
                or len(options[0]) * len(options[1]) > _MAX_COMBINATIONS:
            keep.extend(members)
            continue
        cards.append(_pair_card(dataset_id, outcome_key, a, b, run, overrides, preview, rows))
    return cards, keep


def _pair_card(dataset_id: str, outcome_key: str, a: Question, b: Question, run: Path,
               overrides: Sequence[Mapping[str, Any]], preview: "_Preview",
               rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> Question:
    card_id = "|".join([dataset_id, outcome_key, "", "pair"])
    combinations: list[dict[str, Any]] = []
    patch_sets: list[dict[str, dict[str, Any]]] = []
    for i, option_a in enumerate(a.get("options") or [], 1):
        for j, option_b in enumerate(b.get("options") or [], 1):
            patches = {"A": _patch(a, option_a), "B": _patch(b, option_b)}
            patch_sets.append(patches)
            combinations.append(_combination(f"a{i}|b{j}", (a, option_a), (b, option_b),
                                             dataset_id, outcome_key, preview, patches))
    writes = {str(a.get("answer_writes") or ""), str(b.get("answer_writes") or "")}
    card = _folded(card_id, "pair", "dataset", [a, b], run, overrides, rows,
                   options=_stamped(combinations),
                   prompt=_pair_prompt(a, b),
                   why=" || ".join(dict.fromkeys(x for x in (str(a.get("why") or ""),
                                                             str(b.get("why") or "")) if x)),
                   group=None, group_label="",
                   # what the card writes is what its slots write; when they differ it is still a
                   # number that lands, because every `PAIRABLE` kind's picked option carries one.
                   answer_writes=(writes.pop() if len(writes) == 1 else "value"))
    _stamp_impact(card, [a, b], row_swings=[o.get("implied_d") for o in combinations])
    card["impact_band"] = preview.band(dataset_id, outcome_key, patch_sets)
    return card


def _pair_prompt(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    """Both groups' questions in one sentence, each named by the group it is about."""
    return " ".join(f"{who}: {str(q.get('prompt') or '').strip()}"
                    for q, who in ((a, str(a.get("group_label") or "group A")),
                                   (b, str(b.get("group_label") or "group B"))) if q.get("prompt"))


def _combination(key: str, left: tuple[Mapping[str, Any], Mapping[str, Any]],
                 right: tuple[Mapping[str, Any], Mapping[str, Any]],
                 dataset_id: str, outcome_key: str, preview: "_Preview",
                 patches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One answer to a pair: this option for group A, that one for group B, and what it implies.

    `implied_d` is the resolver's own answer to "what would this put in the forest plot", got by
    running the run's own row builder over copied verdicts — never arithmetic this layer invented,
    because a page that computed effect sizes its own way would be a second resolver nobody tests
    against the first.
    """
    (question_a, option_a), (question_b, option_b) = left, right
    slots = [_option_slot("A", question_a, option_a), _option_slot("B", question_b, option_b)]
    implied, note = preview.implied(dataset_id, outcome_key, patches)
    label = " · ".join(
        f"{str(q.get('group_label') or ('group ' + str(q.get('group'))))}: {o.get('label')}"
        for q, o in (left, right))
    if implied is not None:
        label += f" — implies d = {_fmt(implied)}"
    elif note:
        label += f" — {note}"
    return {"key": key, "label": label, "a": option_a.get("key"), "b": option_b.get("key"),
            "slots": slots, "implied_d": implied, "implied_note": note}


def _option_slot(group: str, question: Mapping[str, Any],
                 option: Mapping[str, Any]) -> dict[str, Any]:
    """What one slot's chosen option contributes, in the shape the fingerprint hashes."""
    return {"group": group, "option": option.get("key"), "label": option.get("label"),
            "mean": option.get("mean"), "dispersion_value": option.get("dispersion_value"),
            "dispersion_type": option.get("dispersion_type"), "n": option.get("n"),
            "analysis_metric": option.get("analysis_metric"), "location": option.get("location")}


def _patch(question: Mapping[str, Any], option: Mapping[str, Any]) -> dict[str, Any]:
    """What choosing this option would DO to the cell, decided by the override it writes.

    Asked of `_single_override` rather than guessed from the option's shape, so the preview and
    the answer can never disagree about what an answer means: "it is not reported" takes the cell
    out, a dispersion type converts the spread, a number replaces it.
    """
    try:
        record = _single_override(question, {"option": option.get("key"),
                                             "note": "what this option would do"})
    except Exception:                                      # pragma: no cover - defensive
        return {}
    if record.get("kind") == "exclude_dataset":
        return {"exclude": True}
    if record.get("kind") != "value":
        return {}
    return {key: record.get(key) for key in ("mean", "dispersion_value", "dispersion_type", "n")}


def _folded(card_id: str, kind: str, scope: str, members: Sequence[Question], run: Path,
            overrides: Sequence[Mapping[str, Any]],
            rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
            **fields: Any) -> Question:
    """The common shape of a card built out of per-cell questions."""
    head = members[0]
    dataset_id = str(head.get("dataset_id") or "")
    outcome_key = str(head.get("outcome_key") or "")
    row = _row_of_cell(rows, run, dataset_id, outcome_key)
    card = _blank(
        id=card_id, kind=kind, scope=scope,
        paper=str(head.get("paper") or ""), paper_id=str(head.get("paper_id") or ""),
        dataset_id=dataset_id if scope in ("cell", "dataset") else "",
        dataset_label=str(head.get("dataset_label") or ""),
        outcome_key=outcome_key, measure_name=str(head.get("measure_name") or ""),
        where=str(head.get("where") or ""), unit=str(head.get("unit") or ""),
        image=dict(head.get("image") or {}),
        confidence=head.get("confidence"), route=head.get("route"),
        member_ids=[str(m["id"]) for m in members],
        cells=[_cell_of(m) for m in members],
        slots=[_slot_of(m, run, overrides) for m in members],
        settled=[s for m in members
                 for s in _settled(overrides, str(m.get("dataset_id") or ""),
                                   str(m.get("outcome_key") or ""), m.get("group"))],
        status_line=_status_line(row), best_guess=_best_guess(row))
    card.update(fields)
    _answered_from(card, members)
    _stamp_impact(card, members, row_swings=None)
    return card


def _answered_from(card: Question, members: Sequence[Mapping[str, Any]]) -> None:
    """A card is open while any cell it settles is: the answer has not been given until every
    slot it names has one. Its `answers` are its members', so the history is not lost."""
    statuses = [str(m.get("status") or "open") for m in members]
    card["status"] = ("open" if "open" in statuses
                      else PENDING_RERUN if PENDING_RERUN in statuses else "answered")
    card["answered"] = card["status"] != "open"
    card["pending_why"] = next((str(m.get("pending_why") or "") for m in members
                                if m.get("status") == PENDING_RERUN), "")
    card["answers"] = [answer for m in members for answer in (m.get("answers") or [])]


def _stamp_impact(card: Question, members: Sequence[Mapping[str, Any]],
                  row_swings: Sequence[float | None] | None = None) -> None:
    """What answering this card is worth, and on what evidence (§C4).

    `pooled` is the run's own number — what admitting this row would do to the estimate, computed
    by `state.pooled_impact` when the queue was built. `row` is the spread of the effect sizes the
    card's own options imply, which is the only measure a card has when the row carries no effect
    size to leave one out of. `unknown` is neither, and it sorts last rather than pretending to
    be harmless.
    """
    pooled = next((float(m["impact"]) for m in members
                   if isinstance(m.get("impact"), (int, float))), None)
    known = [float(x) for x in (row_swings or []) if isinstance(x, (int, float))]
    spread = (max(known) - min(known)) if len(known) >= 2 else (0.0 if known else None)
    if pooled is not None:
        card["impact"], card["impact_basis"], card["impact_rank"] = pooled, "pooled", abs(pooled)
    elif spread is not None:
        card["impact"], card["impact_basis"], card["impact_rank"] = spread, "row", abs(spread)
    else:
        card["impact"], card["impact_basis"], card["impact_rank"] = None, "unknown", -1.0
    card["blocking_rank"] = len({(c.get("dataset_id"), c.get("outcome_key"))
                                 for c in card.get("cells") or []})


# ------------------------------------------------------------------ D1: the precedence override
def _precedence_override_questions(run: Path, rest: Sequence[Question],
                                   overrides: Sequence[Mapping[str, Any]],
                                   pending: Mapping[int, str], consumed: Collection[int],
                                   preview: "_Preview",
                                   rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
                                   ) -> tuple[list[Question], list[Question]]:
    """One card per row the resolver built from a candidate pair instead of the printed value.

    It REPLACES the cell cards that pick a value for that row, because after D1 those are not the
    decision any more: the row already has a number, and what a reviewer has to settle is whether
    it may be the one the printed value could not give. The cards D1 does not answer — a
    refutation, a direction — stay exactly where they are.
    """
    keep = list(rest)
    cards: list[Question] = []
    for (dataset_id, outcome_key), row in (_rows_of(run) if rows is None else rows).items():
        if PRECEDENCE_OVERRIDE_FLAG not in set(row.get("flags") or []):
            continue
        card_id = "|".join([dataset_id, outcome_key, "", "precedence_override"])
        already = _live_answers(overrides, card_id, dataset_id, outcome_key)
        members: list[Question] = []
        if not already:
            # while the decision is OPEN it is the only value question this row has: picking a
            # number for one group is not an answer to "which pair is this row built from". Once
            # it is answered the cells go back to asking whatever is still holding them — a fold
            # that kept them absorbed would leave a held row with nothing open anywhere.
            members = [q for q in keep if q.get("dataset_id") == dataset_id
                       and q.get("outcome_key") == outcome_key and q.get("kind") in PAIRABLE]
            keep = [q for q in keep if q not in members]
        cards.append(_precedence_card(run, dataset_id, outcome_key, row, members, overrides,
                                      already, pending, consumed, preview))
    return cards, keep


def _live_answers(overrides: Sequence[Mapping[str, Any]], card_id: str, dataset_id: str,
                  outcome_key: str) -> list[dict[str, Any]]:
    """The answers naming this card that a LATER decision about the row's values has not overtaken.

    D1's card is not settled once and for ever: the resolver re-raises the override every time it
    rebuilds the row, so a reviewer who answered "use the candidate pair" and then typed the
    printed value back has a row that is overridden again — and a card ticked "answered" on a
    decision the log itself has superseded is the tick §C4 exists to remove.
    """
    already = _named(overrides, card_id)
    if not already:
        return []
    last = max((int(o["seq"]) for o in already if isinstance(o.get("seq"), int)), default=-1)
    for override in overrides:
        if override.get("kind") != "value" or str(override.get("question_id") or "") == card_id:
            continue
        if str(override.get("dataset_id") or "") != dataset_id \
                or str(override.get("outcome_key") or "") != outcome_key:
            continue
        if isinstance(override.get("seq"), int) and override["seq"] > last:
            return []
    return already


def _precedence_card(run: Path, dataset_id: str, outcome_key: str, row: Mapping[str, Any],
                     members: Sequence[Question], overrides: Sequence[Mapping[str, Any]],
                     already: Sequence[Mapping[str, Any]], pending: Mapping[int, str],
                     consumed: Collection[int], preview: "_Preview") -> Question:
    card_id = "|".join([dataset_id, outcome_key, "", "precedence_override"])
    paper_id = str(row.get("paper_id") or "")
    study, dataset = _dataset(run, paper_id, dataset_id)
    citation = study.get("citation") or {}
    options = [*_candidate_options(preview, dataset_id, outcome_key),
               {"key": "keep_printed",
                "label": "keep the printed value — the paper's own number is the one this row "
                         "should use, even though it converts to no effect size"},
               {"key": "exclude",
                "label": "take this dataset out of the analysis — neither the printed value nor "
                         "the reading is usable here"}]
    reason = _short(row.get("precedence_override_reason"), 900)
    printed = str(row.get("route_overridden_from") or "the printed value")
    prompt = (f"This {outcome_key.replace('_', ' ')} row is not built from {printed}, which the "
              f"protocol prefers: that value converts to no effect size, so the resolver used a "
              f"same-place candidate pair instead and is holding the row until a person decides. "
              f"{reason} Which of these is this row?")
    status, pending_why = _answer_status(already, pending, consumed)
    card = _blank(
        id=card_id, kind="precedence_override", scope="dataset",
        paper=f"{citation.get('first_author') or citation.get('authors') or '?'} "
              f"{citation.get('year') or ''}".strip(),
        paper_id=paper_id, dataset_id=dataset_id,
        dataset_label=str(dataset.get("label") or dataset.get("experiment") or ""),
        outcome_key=outcome_key, measure_name=_measure(dataset, outcome_key),
        options=_stamped(options), prompt=prompt,
        answer_writes="value", why=reason or "the resolver recorded no reason",
        confidence=row.get("confidence"), route=row.get("route"),
        member_ids=[card_id, *(str(m["id"]) for m in members)],
        cells=[{"dataset_id": dataset_id, "outcome_key": outcome_key, "group": group,
                "group_label": _group_label(dataset, group),
                "dataset_label": str(dataset.get("label") or ""),
                "paper": "", "measure_name": "", "question_id": card_id} for group in ("A", "B")],
        slots=[_slot_of(m, run, overrides) for m in members],
        settled=_settled(overrides, dataset_id, outcome_key, None),
        status=status, pending_why=pending_why, answered=bool(already),
        answers=[{"kind": o.get("kind"), "justification": o.get("justification"),
                  "mean": o.get("mean"), "at": o.get("at") or o.get("timestamp")}
                 for o in already],
        status_line=_status_line(row), best_guess=_best_guess(row))
    _stamp_impact(card, members or [{"impact": None}],
                  row_swings=[o.get("implied_d") for o in options if "implied_d" in o])
    return card


def _candidate_options(preview: "_Preview", dataset_id: str,
                       outcome_key: str) -> list[dict[str, Any]]:
    """`use_candidates`, one per PLACE the row could have been read (Task 4 review, finding 5).

    When a cell was read in two places the resolver's tie-break is route precedence and then
    reading order — a choice between two pictures that nothing on the record justifies. So the
    card offers each place as its own answer whenever there is more than one; with a single place
    the key stays the bare `use_candidates`, because there is nothing to choose between.
    """
    pairs = preview.alternatives(dataset_id, outcome_key)
    out: list[dict[str, Any]] = []
    for values, locator, key in pairs:
        implied, note = preview.implied(dataset_id, outcome_key, {
            "A": _group_patch(values.group_a), "B": _group_patch(values.group_b)})
        shown = " / ".join(_group_shown(g) for g in (values.group_a, values.group_b))
        label = (f"use the pair read under {locator!r}: {shown}" if locator
                 else f"use the candidate pair: {shown}")
        if implied is not None:
            label += f" — implies d = {_fmt(implied)}"
        elif note:
            label += f" — {note}"
        out.append({"key": "use_candidates" if len(pairs) == 1 else f"use_candidates@{key}",
                    "label": label, "locator": locator,
                    "slots": [{**_group_patch(values.group_a), "group": "A"},
                              {**_group_patch(values.group_b), "group": "B"}],
                    "implied_d": implied, "implied_note": note})
    if not out:
        out.append({"key": "use_candidates", "locator": "", "slots": [],
                    "implied_d": None, "implied_note": "",
                    "label": "use the candidate pair this row was built from — the readings are "
                             "no longer in this run's stage files, so answer with the numbers "
                             "below"})
    return out


def _group_patch(group: Any) -> dict[str, Any]:
    """One group of a candidate pair as an answer's numbers — `{}` when there is no such group."""
    if group is None:
        return {}
    kind = getattr(group.dispersion_type, "value", group.dispersion_type)
    return {"mean": group.mean, "dispersion_value": group.dispersion_value,
            "dispersion_type": str(kind or ""), "n": group.n}


def _group_shown(group: Any) -> str:
    patch = _group_patch(group)
    if not patch or patch.get("mean") is None:
        return "no value"
    out = _fmt(float(patch["mean"]))
    if patch.get("dispersion_value") is not None:
        out += f" ± {_fmt(float(patch['dispersion_value']))}"
        if patch.get("dispersion_type"):
            out += f" ({patch['dispersion_type']})"
    if patch.get("n"):
        out += f", n = {patch['n']}"
    return out


# ------------------------------------------------------------------ D4-lite: the analysed n
def _analysed_n_questions(run: Path, overrides: Sequence[Mapping[str, Any]],
                          pending: Mapping[int, str], consumed: Collection[int] = (),
                          rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
                          ) -> list[Question]:
    """One card per DATASET whose n may be a recruited count, never one per cell (D4-lite).

    How many people were analysed is a fact about the two arms: it holds for every outcome
    measured on those people, so a size answered for late adaptation that left the aftereffect on
    the recruited count would put two different denominators behind one pair of groups. The card
    is scoped where the `group_n` override is.
    """
    found: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for paper_id, verdicts in _verdicts_by_paper(run):
        for verdict in verdicts:
            flags = [f for f in verdict.get("flags") or []
                     if str(f.get("code") or "") == N_BEFORE_EXCLUSIONS]
            if not flags:
                continue
            dataset_id = str(verdict.get("dataset_id") or "")
            group = verdict.get("group")
            if not dataset_id or group not in ("A", "B"):
                continue
            entry = found.setdefault(dataset_id, {"paper_id": paper_id, "groups": {},
                                                  "outcomes": []})
            entry["outcomes"].append(str(verdict.get("outcome_key") or ""))
            detail = flags[0].get("detail") or {}
            said = {"recruited": detail.get("recruited"), "excluded": detail.get("excluded"),
                    "quote": str(detail.get("quote") or ""),
                    "message": str(flags[0].get("message") or ""), "n": verdict.get("n")}
            was = entry["groups"].get(group)
            # the same arm can be flagged on two outcomes and the check may have finished reading
            # on only one of them. The one that PARSED both counts is the one the card can offer a
            # subtraction from, so it wins whichever outcome it came from.
            if was is None or (not isinstance(was.get("excluded"), int)
                               and isinstance(said["excluded"], int)):
                entry["groups"][group] = said
    cards = [_analysed_n_card(run, dataset_id, entry, overrides, pending, consumed, rows)
             for dataset_id, entry in found.items()]
    return _fold_analysed_n_by_paper(cards)


def _fold_analysed_n_by_paper(cards: Sequence[Question]) -> list[Question]:
    """A paper's analysed sizes, asked once per paper when it has more than one flagged dataset.

    The fact is still per dataset — each dataset's arms are its own people, and each slot writes
    its own `group_n` record naming its own dataset — but the exclusion sentence is one passage
    of one paper, and a reviewer who has found it answers every dataset from it. One card, one
    slot per dataset, answered one at a time.
    """
    by_paper: "OrderedDict[str, list[Question]]" = OrderedDict()
    for card in cards:
        by_paper.setdefault(str(card.get("paper_id") or ""), []).append(card)
    out: list[Question] = []
    for paper_id, mine in by_paper.items():
        if len(mine) < 2 or not paper_id:
            out.extend(mine)
            continue
        out.append(_analysed_n_paper_card(paper_id, mine))
    return out


def _analysed_n_paper_card(paper_id: str, members: Sequence[Question]) -> Question:
    card_id = "|".join([sha12(paper_id), "", "", "analysed_n"])
    head = members[0]
    slots = [{"group": None,
              "group_label": str(m.get("dataset_label") or m.get("dataset_id") or ""),
              "kind": "analysed_n", "member_id": str(m.get("id") or ""),
              "dataset_id": str(m.get("dataset_id") or ""), "outcome_key": "",
              "prompt": str(m.get("prompt") or ""), "why": str(m.get("why") or ""),
              "answer_writes": "group_n", "options": list(m.get("options") or []),
              "image": {}, "where": "", "unit": "",
              # a dataset already answered is history on the card, not a second chance to write
              # a second `group_n` for the same arms; its record is what the slot shows
              "answerable": str(m.get("status") or "open") == "open",
              "settled": [{"clears": [], "overrules": [], "kind": "group_n",
                           "at": str(a.get("at") or ""), "actor": "",
                           "justification": _short(a.get("justification"), 300)}
                          for a in (m.get("answers") or [])]}
             for m in members]
    card = _blank(
        id=card_id, kind="analysed_n", scope="paper",
        paper=str(head.get("paper") or ""), paper_id=paper_id,
        options=[], answer_writes="group_n", slot_answers=True,
        prompt=(f"{len(members)} datasets of this paper are divided by group sizes that look "
                f"like the numbers the paper RECRUITED, not the numbers it analysed. How many "
                f"people are in each group's analysis? Each dataset is answered on its own, "
                f"below."),
        why=str(head.get("why") or ""), confidence="needs_human", route="",
        member_ids=[str(m.get("id") or "") for m in members],
        cells=[dict(c) for m in members for c in (m.get("cells") or [])],
        slots=slots)
    _answered_from(card, members)
    card["blocking_rank"] = len(card["cells"])
    return card


def _analysed_n_card(run: Path, dataset_id: str, entry: Mapping[str, Any],
                     overrides: Sequence[Mapping[str, Any]], pending: Mapping[int, str],
                     consumed: Collection[int],
                     rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> Question:
    card_id = "|".join([dataset_id, "", "", "analysed_n"])
    paper_id = str(entry.get("paper_id") or "")
    study, dataset = _dataset(run, paper_id, dataset_id)
    citation = study.get("citation") or {}
    groups = dict(entry.get("groups") or {})
    recorded = {group: _recorded_n(run, dataset_id, group, groups, rows)
                for group in ("A", "B")}
    options: list[dict[str, Any]] = [
        {"key": "recorded", "n_a": recorded["A"], "n_b": recorded["B"],
         "label": f"the sizes the run used are the ANALYSED sizes "
                  f"({_num(recorded['A'])}/{_num(recorded['B'])}) — the check read the wrong "
                  f"sentence"}]
    minus = _minus_option(groups, recorded)
    if minus is not None:
        options.append(minus)
    # the one option on the page that is a FORM rather than an answer: it carries no numbers, and
    # `needs_input` says which ones it is waiting for — so a client can tell "pick this" from
    # "pick this and fill these in", and the log's refusal of an empty one is a bug report about
    # the client rather than about the card.
    options.append({"key": "typed", "n_a": None, "n_b": None, "needs_input": ["n_a", "n_b"],
                    "label": "type the analysed size of each group below"})
    quoted = "; ".join(dict.fromkeys(
        f"{group}: “{_short(said.get('quote') or said.get('message'), 240)}”"
        for group, said in sorted(groups.items()) if said.get("quote") or said.get("message")))
    prompt = (f"The group sizes this dataset's rows are divided by look like the numbers the "
              f"paper RECRUITED, not the numbers it analysed: the check found an exclusion "
              f"beside them. How many people are in each group's analysis? {quoted}")
    already = _named(overrides, card_id)
    status, pending_why = _answer_status(already, pending, consumed)
    card = _blank(
        id=card_id, kind="analysed_n", scope="dataset",
        paper=f"{citation.get('first_author') or citation.get('authors') or '?'} "
              f"{citation.get('year') or ''}".strip(),
        paper_id=paper_id, dataset_id=dataset_id,
        dataset_label=str(dataset.get("label") or dataset.get("experiment") or ""),
        options=_stamped(options), prompt=prompt, answer_writes="group_n",
        why=f"the check `{N_BEFORE_EXCLUSIONS}` fired on "
            f"{len(set(entry.get('outcomes') or []))} outcome(s) of this dataset; an n that "
            f"counts people the analysis dropped makes every effect size on those arms too "
            f"precise",
        confidence="needs_human", route="",
        cells=[{"dataset_id": dataset_id, "outcome_key": key, "group": None,
                "group_label": "", "dataset_label": "", "paper": "", "measure_name": "",
                "question_id": card_id} for key in dict.fromkeys(entry.get("outcomes") or [])],
        member_ids=[card_id], status=status, pending_why=pending_why, answered=bool(already),
        answers=[{"kind": o.get("kind"), "justification": o.get("justification"),
                  "mean": None, "at": o.get("at") or o.get("timestamp")} for o in already])
    card["blocking_rank"] = len(card["cells"])
    return card


def _recorded_n(run: Path, dataset_id: str, group: str,
                groups: Mapping[str, Mapping[str, Any]],
                rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> int | None:
    said = groups.get(group) or {}
    if said.get("n"):
        return int(said["n"])
    for (ds, _outcome), row in (_rows_of(run) if rows is None else rows).items():
        if ds == dataset_id and row.get(f"n_{group.lower()}"):
            return int(row[f"n_{group.lower()}"])
    return None


def _minus_option(groups: Mapping[str, Mapping[str, Any]],
                  recorded: Mapping[str, int | None]) -> dict[str, Any] | None:
    """`recruited − excluded`, offered only where the check parsed BOTH numbers.

    A group the check never flagged keeps the size the run used — its n was never in doubt. A
    group it flagged but could not finish reading has no subtraction to offer, and offering one
    anyway would be offering a number the tool made up.
    """
    sizes: dict[str, int | None] = {}
    for group in ("A", "B"):
        said = groups.get(group)
        if said is None:
            sizes[group] = recorded.get(group)
            continue
        recruited, excluded = said.get("recruited"), said.get("excluded")
        if not isinstance(recruited, int) or not isinstance(excluded, int):
            return None
        sizes[group] = recruited - excluded
    if not all(isinstance(sizes[g], int) and sizes[g] >= 1 for g in ("A", "B")):
        return None
    quote = "; ".join(dict.fromkeys(str((groups.get(g) or {}).get("quote") or "")
                                    for g in ("A", "B") if (groups.get(g) or {}).get("quote")))
    return {"key": "recruited_minus_excluded", "n_a": sizes["A"], "n_b": sizes["B"],
            "quote": _short(quote, 900),
            "label": f"recruited minus excluded ({_num(sizes['A'])}/{_num(sizes['B'])}) — the "
                     f"counts the check read out of the paper"}


# ------------------------------------------------------------------ C3: include_paper
def _excluded_paper_questions(run: Path, overrides: Sequence[Mapping[str, Any]],
                              pending: Mapping[int, str],
                              consumed: Collection[int] = ()) -> list[Question]:
    """One card per paper the MAPPER excluded — the only exclusion a card may reopen (§C3).

    A paper the mapper called ineligible, or eligible with nothing to read, left the review with
    a model's reasoning and nobody's decision. That is a question, and it is the biggest one on the
    page: a whole paper. An exclusion a PERSON made is not — it is the answer.
    """
    cards: list[Question] = []
    seen: set[str] = set()
    for row in _json_if_present(run / "exclusions.json") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("stage") or "") != "map" or str(row.get("decider") or "") != "mapper":
            continue
        if str(row.get("reason") or "") not in MAPPER_EXCLUSIONS:
            continue
        paper_id = str(row.get("paper_id") or "")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        cards.append(_include_paper_card(run, paper_id, row, overrides, pending, consumed))
    return cards


def _include_paper_card(run: Path, paper_id: str, row: Mapping[str, Any],
                        overrides: Sequence[Mapping[str, Any]], pending: Mapping[int, str],
                        consumed: Collection[int]) -> Question:
    card_id = "|".join([sha12(paper_id), "", "", "include_paper"])
    study = _stage(run, paper_id, "map").get("study") or {}
    citation = study.get("citation") or {}
    name = f"{citation.get('first_author') or citation.get('authors') or ''} " \
           f"{citation.get('year') or ''}".strip() or str(row.get("filename") or "this paper")
    said = _short(row.get("detail"), 700)
    quote = _short(row.get("quote"), 700)
    mapped = len(study.get("datasets") or [])
    tail = ("" if mapped else " Nothing was mapped in it, so including it buys a fresh map as "
                              "well as an extraction on the next `--resume`.")
    prompt = (f"The mapper took {name} out of this review and no person has been asked about it. "
              f"Its reason: {said or 'none recorded'}. It was reading: “{quote}”. Does this "
              f"review include this paper?{tail}")
    already = _named(overrides, card_id)
    status, pending_why = _answer_status(already, pending, consumed)
    return _blank(
        id=card_id, kind="include_paper", scope="paper", paper=name, paper_id=paper_id,
        options=_stamped([
            {"key": "include", "decision": "include",
             "label": "include it — the mapper's reading of the criterion is wrong, and this "
                      "paper belongs in the review"},
            {"key": "keep_out", "decision": "exclude",
             "label": "keep it out — I have read the criterion and the mapper is right"}]),
        prompt=prompt, answer_writes="eligibility",
        why=f"decided at the map stage by the mapper alone ({row.get('reason')}), before "
            f"anything in this paper was read: {said}",
        confidence="excluded", route="map",
        member_ids=[card_id], cells=[], status=status, pending_why=pending_why,
        answered=bool(already),
        answers=[{"kind": o.get("kind"), "justification": o.get("justification"),
                  "mean": None, "at": o.get("at") or o.get("timestamp")} for o in already])


def _named(overrides: Sequence[Mapping[str, Any]], card_id: str) -> list[dict[str, Any]]:
    """Every recorded answer that names THIS card — the only thing that settles a card whose
    evidence lives in a stage file a re-pool never rewrites."""
    return [dict(o) for o in overrides if str(o.get("question_id") or "") == card_id]


# ------------------------------------------------------------------ what the row says about itself
def _rows_of(run: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Every row of the run, as the resolver wrote it AND as the last re-pool derived it.

    Both, and unioned on the flags, for `overrides.row_flags`' reason: the stage file is the fixed
    reference a re-pool never rewrites, and the table is where a finding a REVIEWER's own numbers
    produced can be seen. A card built from either alone would miss half the run.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((run / "papers").glob("*/resolve.json")):
        for record in (_json_if_present(path) or {}).get("records") or []:
            if isinstance(record, dict):
                out[(str(record.get("dataset_id") or ""),
                     str(record.get("outcome_key") or ""))] = dict(record)
    for row in _json_if_present(run / "results" / "extraction_table_all.json") or []:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("dataset_id") or ""), str(row.get("outcome_key") or ""))
        was = out.get(key, {})
        merged = {**was, **{k: v for k, v in row.items() if v is not None}}
        merged["flags"] = sorted({*(was.get("flags") or []), *(row.get("flags") or [])})
        out[key] = merged
    return out


def _row_of_cell(rows: Mapping[tuple[str, str], Mapping[str, Any]] | None, run: Path,
                 dataset_id: str, outcome_key: str) -> Mapping[str, Any]:
    """One cell's row, out of the map the page read once. `run` is the fallback for a caller that
    has no map — there is none inside `questions_for_run`, and there should not be one."""
    if rows is None:
        rows = _rows_of(run)
    return rows.get((dataset_id, outcome_key)) or {}


def _status_line(row: Mapping[str, Any]) -> str:
    """Where this row stands WHILE the question is open — the best-guess line's own words (§A).

    A held row is not simply missing: the best-guess line either admits it under a named rule or
    vetoes it, and a reviewer deciding what to spend their attention on needs to know which. Read
    off the row's own columns; a run written before they existed says what it can.
    """
    if not row:
        return ""
    rule = str(row.get("best_guess_rule") or "")
    reason = _short(row.get("best_guess_reason"), 300)
    in_line = row.get("in_best_guess")
    if in_line is True:
        head = (f"in the best-guess line at d = {_fmt(float(row['best_guess_es']))}"
                if isinstance(row.get("best_guess_es"), (int, float))
                else "in the best-guess line")
        return f"Held, {head}" + (f" by rule `{rule}`" if rule else "") \
            + (f": {reason}" if reason else "") + "."
    if in_line is False:
        return "Held, and in NEITHER analysis line" + (f": {reason}" if reason else "") + "."
    return (f"Held ({row.get('confidence') or 'needs_human'}); this run predates the best-guess "
            f"line, so nothing says whether it would be admitted.")


def _best_guess(row: Mapping[str, Any]) -> dict[str, Any]:
    """The row's own best-guess verdict — the rule that admitted it, or the veto that did not."""
    if not row:
        return {}
    reason = str(row.get("best_guess_reason") or "")
    veto = reason.split(":", 1)[0] if row.get("in_best_guess") is False and ":" in reason else ""
    return {"in": row.get("in_best_guess"), "rule": str(row.get("best_guess_rule") or ""),
            "veto": veto, "reason": _short(reason, 500),
            "es": row.get("best_guess_es"), "se": row.get("best_guess_se")}


def _verdicts_by_paper(run: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for path in sorted((run / "papers").glob("*/verify.json")):
        payload = _json_if_present(path) or {}
        verdicts = [v for v in payload.get("verdicts") or [] if isinstance(v, dict)]
        paper_id = next((str(v.get("paper_id") or "") for v in verdicts if v.get("paper_id")), "")
        if not paper_id:
            study = (_json_if_present(path.parent / "map.json") or {}).get("study") or {}
            paper_id = str(study.get("paper_id") or "")
        out.append((paper_id, verdicts))
    return out


# ------------------------------------------------------------------ what an answer would do
#: what `_Preview._resolve` returns for an answer that takes the cell OUT of the analysis: there
#: is no record to show, and `None` already means "this run could not be resolved from".
_EXCLUDED = object()


class _Preview:
    """The run's OWN row builder, borrowed so a card can say what an answer would do.

    Every number a card shows about a combination — the effect size it implies, whether it moves
    the pooled estimate enough to be worth a reviewer's attention — is produced by
    `pipeline.rows.prepare_rows` and `resolve.resolve_effect` over COPIES of the cells' verdicts:
    the same path the run and the re-pool take, including the shared-control split and the row's
    own policy flags. A review layer that did this arithmetic itself would be a second resolver,
    and the first thing a second resolver does is disagree with the first.

    Everything is lazy and memoised. A page with no combination question on it never loads a stage
    file through here; one with several loads them once and resolves each distinct combination
    once, which is what keeps `questions_for_run` a read of the run rather than a re-run of it.
    """

    def __init__(self, run: Path) -> None:
        self.run = Path(run)
        self._loaded = False
        self._state: Any = None
        self._protocol: Any = None
        self._resolved: dict[str, Any] = {}
        self._alternatives: dict[tuple[str, str], list[tuple[Any, str, str]]] = {}

    # ---- the run's own state, loaded once
    def _load(self) -> tuple[Any, Any]:
        if not self._loaded:
            self._loaded = True
            try:
                from ..pipeline.overrides import _RunState
                from ..pipeline.state import load_manifest
                from ..protocol import load_protocol

                manifest = load_manifest(self.run)
                for name in (self.run / "protocol.yaml", self.run / "protocol.staged.yaml",
                             Path(str(manifest.protocol_path or ""))):
                    if str(name) and name.exists():
                        self._protocol = load_protocol(name)
                        break
                if self._protocol is not None:
                    self._state = _RunState(self.run, manifest)
            except Exception:            # a run this cannot be read from still gets its questions
                self._state = self._protocol = None
        return self._state, self._protocol

    def alternatives(self, dataset_id: str, outcome_key: str) -> list[tuple[Any, str, str]]:
        """D1's own candidate pairs for this cell — `(values, locator, place key)`, in its order."""
        key = (dataset_id, outcome_key)
        if key in self._alternatives:
            return self._alternatives[key]
        out: list[tuple[Any, str, str]] = []
        prepared = self._prepared(dataset_id, outcome_key, {})
        for values in getattr(prepared, "alternatives", None) or []:
            locator = next((g.locator for g in (values.group_a, values.group_b)
                            if g is not None and g.locator), "")
            digest = hashlib.sha1(" ".join(locator.split()).casefold().encode("utf-8"))
            out.append((values, _short(locator, 90), digest.hexdigest()[:8]))
        self._alternatives[key] = out
        return out

    def implied(self, dataset_id: str, outcome_key: str,
                patches: Mapping[str, Mapping[str, Any]]) -> tuple[float | None, str]:
        """The effect size this answer would put in the plot, and why there is none when there is
        none. `(None, "")` when this run cannot be resolved from at all — a card still asks."""
        record = self._resolve(dataset_id, outcome_key, patches)
        if record is _EXCLUDED:
            return None, "this answer takes the cell out of the analysis"
        if record is None:
            return None, ""
        if record.es is None:
            return None, _short(record.not_convertible_reason, 200) or "no effect size converts"
        return float(record.es), ""

    def band(self, dataset_id: str, outcome_key: str,
             patch_sets: Sequence[Mapping[str, Mapping[str, Any]]]) -> str:
        """§C4's band: `low` when every answer on offer barely moves the row AND barely moves the
        pool. Both halves, because either alone is half the question — a row whose d hardly moves
        can still be the row that decides a k = 2 pool, and a big swing on a row with almost no
        weight is not where a reviewer's attention belongs. `high` whenever it cannot be priced:
        "we could not tell" is not "it does not matter"."""
        records = [self._resolve(dataset_id, outcome_key, patches) for patches in patch_sets]
        if not records or any(r is None or r is _EXCLUDED or r.es is None for r in records):
            return "high"
        spread = max(r.es for r in records) - min(r.es for r in records)
        if spread >= LOW_IMPACT_D:
            return "high"
        pooled = [self._pooled(dataset_id, outcome_key, record) for record in records]
        if any(value is None for value in pooled):
            return "high"
        return "low" if max(pooled) - min(pooled) < LOW_IMPACT_POOLED else "high"

    # ---- internals
    def _resolve(self, dataset_id: str, outcome_key: str,
                 patches: Mapping[str, Mapping[str, Any]]) -> Any:
        key = json.dumps([dataset_id, outcome_key, patches], sort_keys=True, default=str)
        if key in self._resolved:
            return self._resolved[key]
        out: Any = None
        if any((patches.get(group) or {}).get("exclude") for group in ("A", "B")):
            out = _EXCLUDED
        else:
            prepared = self._prepared(dataset_id, outcome_key, patches)
            _state, protocol = self._load()
            if prepared is not None and protocol is not None:
                try:
                    from ..pipeline.resolve import resolve_effect

                    out = resolve_effect(self._state.datasets[dataset_id],
                                         protocol.outcome(outcome_key), prepared.values,
                                         protocol.stats)
                except Exception:                          # pragma: no cover - defensive
                    out = None
        self._resolved[key] = out
        return out

    def _prepared(self, dataset_id: str, outcome_key: str,
                  patches: Mapping[str, Mapping[str, Any]]) -> Any:
        state, protocol = self._load()
        if state is None or protocol is None or dataset_id not in state.datasets:
            return None
        cells = {group: state.verdict(dataset_id, outcome_key, group) for group in ("A", "B")}
        if any(cell is None for cell in cells.values()):
            return None
        for group, cell in cells.items():
            cells[group] = _patched(cell, patches.get(group) or {})
        try:
            from ..pipeline.overrides import _prepare

            return _prepare(state.datasets[dataset_id], outcome_key, cells["A"], cells["B"],
                            protocol, state=state)
        except Exception:                                  # pragma: no cover - defensive
            return None

    def _pooled(self, dataset_id: str, outcome_key: str, record: Any) -> float | None:
        """The pooled estimate this row would sit in, at this answer's number."""
        state, protocol = self._load()
        if state is None or protocol is None or record is None or record.es is None:
            return None
        primary = [r for r in state.records if r.outcome_key == outcome_key
                   and r.dataset_id != dataset_id and r.es is not None
                   and r.confidence != "needs_human"]
        try:
            from ..pipeline.state import _pool

            return _pool([*primary, record], protocol.stats)
        except Exception:                                  # pragma: no cover - defensive
            return None


def _patched(verdict: Any, patch: Mapping[str, Any]) -> Any:
    """One cell as an answer would leave it — the same fields `overrides._apply_value` writes."""
    from ..models import DispersionType

    out = verdict.model_copy(deep=True)
    if patch.get("mean") is not None:
        out.mean = float(patch["mean"])
    if patch.get("dispersion_value") is not None:
        out.dispersion_value = float(patch["dispersion_value"])
    if patch.get("dispersion_type"):
        try:
            out.dispersion_type = DispersionType(patch["dispersion_type"])
        except ValueError:                                 # an option naming no known type
            pass
    if patch.get("n") is not None:
        out.n = int(patch["n"])
    return out


# ------------------------------------------------------------------ answers, one per cell named
def answers_to_overrides(question: Mapping[str, Any],
                         answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One answer, one override per CELL the answer names (§C1).

    A card can be a decision about two cells at once — a dataset's pair, a precedence override —
    and the record of it must still be one record per cell, each naming its own group and carrying
    only its own option's `clears`. A single merged record would be an answer that cleared findings
    on a cell nobody said anything about, which is the standing ruling this list exists to keep:
    answers clear only what they name.
    """
    kind = str(question.get("kind") or "")
    slot_answers = answer.get("slots")
    if isinstance(slot_answers, list) and slot_answers:
        # §C1, second fold: the card's own answer (if one was given) and then one record per slot
        # answered — each through the slot's OWN question, so a refutation is overruled by name
        # and a value lands on the group it names. An answer naming a slot the card does not
        # carry, one the card does not let the page answer alone, one that names a slot twice,
        # or one with nothing in it, is refused whole: half a decision is not recorded as one.
        named = [str((e or {}).get("slot") or "") for e in slot_answers]
        if len(set(named)) != len(named):
            from ..pipeline.overrides import OverrideRejected

            raise OverrideRejected("a slot is named twice in one answer; one answer per slot")
        rest = {key: value for key, value in answer.items() if key != "slots"}
        out = answers_to_overrides(question, rest) if _has_card_answer(rest) else []
        return [*out, *(_slot_record(question, entry) for entry in slot_answers)]
    if kind == "pair":
        return _pair_answers(question, answer)
    if kind == "cell":
        return _cell_answers(question, answer)
    if kind == "precedence_override":
        return _precedence_answers(question, answer)
    if kind == "analysed_n":
        if question.get("slot_answers"):
            from ..pipeline.overrides import OverrideRejected

            raise OverrideRejected("this card asks one dataset per slot: answer it through "
                                   "`slots`, naming the dataset each size is for")
        return [_analysed_n_answer(question, answer)]
    if kind == "include_paper":
        return [_eligibility_answer(question, answer)]
    return [_single_override(question, answer)]


#: the payload fields that are an answer to the CARD (as opposed to `slots`, `note`, `id`).
#: `group` and `dispersion_type` are not here: the one only says WHICH slot a typed value is
#: for, the other only says what a typed spread is — neither is an answer on its own.
_CARD_ANSWER_FIELDS = ("option", "exclude", "mean", "dispersion_value", "n", "hint",
                       "decision", "n_a", "n_b", "rule", "quote")
#: the fields that make a slot entry an answer (an option, a typed number, a hint, an exclusion)
_SLOT_ANSWER_FIELDS = ("option", "mean", "dispersion_value", "n", "hint", "n_a", "n_b", "exclude")


def _has_card_answer(answer: Mapping[str, Any]) -> bool:
    return any(answer.get(field) not in (None, "", False) for field in _CARD_ANSWER_FIELDS)


def _cell_answers(card: Mapping[str, Any], answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A card-level answer on a `cell` card: the row's exclusion, or a typed value that names its
    group — which is that slot's own question, answered by hand. Anything else is refused: the
    card offers no options of its own, so an `option` here is one the card never showed, and a
    fall-through that recorded it as "a human looked at it" would be the overclaim §C4 removed.
    """
    from ..pipeline.overrides import OverrideRejected

    if answer.get("exclude"):
        return [_single_override(card, answer)]
    slots = {str(s.get("group") or ""): s for s in card.get("slots") or [] if s.get("answerable")}
    if answer.get("option"):
        raise OverrideRejected(f"{str(answer.get('option'))!r} is not one of this card's answers: "
                               f"answer a slot by name through `slots`")
    named = str(answer.get("group") or "")
    if named not in slots:
        raise OverrideRejected("a typed value on this card has to say which group it is for "
                               f"({', '.join(sorted(slots)) or 'no slot is open'})")
    if not any(answer.get(f) not in (None, "", False) for f in _SLOT_ANSWER_FIELDS):
        raise OverrideRejected("nothing was typed; an answer that decides nothing is not "
                               "recorded as one that does")
    slot = slots[named]
    return [_slot_answer(card, slot, {**answer, "option": None},
                         name=str(slot.get("member_id") or "") or None)]


def _slot_record(card: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """One slot's answer, as the record its own question writes."""
    from ..pipeline.overrides import OverrideRejected

    wanted = str((entry or {}).get("slot") or "")
    slot = next((s for s in card.get("slots") or []
                 if str(s.get("member_id") or "") == wanted and s.get("answerable")), None)
    if slot is None:
        raise OverrideRejected(f"this card has no slot {wanted or 'unnamed'!r} that can be "
                               f"answered on its own; reload the questions")
    if not any((entry or {}).get(f) not in (None, "", False) for f in _SLOT_ANSWER_FIELDS):
        raise OverrideRejected(f"slot {wanted!r} was sent with no answer in it; an answer that "
                               f"decides nothing is not recorded as one that does")
    if str(slot.get("kind") or "") == "analysed_n":
        # the record names the DATASET's own card, which is what `_named` settles it by
        view = {"id": slot.get("member_id"), "number": card.get("number"),
                "kind": "analysed_n", "paper_id": card.get("paper_id", ""),
                "dataset_id": slot.get("dataset_id") or "", "options": list(slot.get("options") or [])}
        return _analysed_n_answer(view, entry)
    return _slot_answer(card, slot, entry, name=str(slot.get("member_id") or "") or None)


def answer_to_override(question: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """The FIRST record an answer becomes — the whole of it for every one-cell question.

    Kept because it is the older contract and most callers answer a cell. Anything that records a
    decision must use `answers_to_overrides`: on a folded card this returns group A's record and
    silently leaves group B unanswered.
    """
    return answers_to_overrides(question, answer)[0]


def _stem(question: Mapping[str, Any], answer: Mapping[str, Any]) -> tuple[str, str]:
    note = str(answer.get("note") or "").strip()
    stem = f"answered question #{question.get('number', '?')} ({question.get('kind') or 'other'})"
    return (f"{stem}: {note}" if note else stem), note


def _base(question: Mapping[str, Any], **over: Any) -> dict[str, Any]:
    out = {"paper_id": question.get("paper_id", ""), "dataset_id": question.get("dataset_id", ""),
           "outcome_key": question.get("outcome_key", ""), "group": question.get("group"),
           "question_id": question.get("id", "")}
    out.update(over)
    return out


def _pair_answers(card: Mapping[str, Any],
                  answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The combination, translated one slot at a time through each slot's OWN question."""
    slots = {str(s.get("group")): s for s in card.get("slots") or []}
    option = next((o for o in card.get("options") or []
                   if o.get("key") == answer.get("option")), None)
    if option is None:
        # a typed answer, or a whole-cell decision: it names its group, or it is about the row
        named = str(answer.get("group") or "")
        if named in slots:
            return [_slot_answer(card, slots[named], {**answer, "option": None})]
        return [_single_override(card, answer)]
    out: list[dict[str, Any]] = []
    for group, chosen in (("A", option.get("a")), ("B", option.get("b"))):
        slot = slots.get(group)
        if slot is not None:
            out.append(_slot_answer(card, slot, {**answer, "option": chosen}))
    return out or [_single_override(card, answer)]


def _slot_answer(card: Mapping[str, Any], slot: Mapping[str, Any],
                 answer: Mapping[str, Any], *, name: str | None = None) -> dict[str, Any]:
    """One slot's own question, answered the way it would have been answered on its own card.

    The record names the CARD for a pair (one combination was the question the reviewer was
    shown), and the SLOT's own question when `name` says so — a slot answered on its own (§C1,
    second fold) is that question, and a record naming the card would be read by every later
    question the same cell asks as an answer to it too (the `on_a_card` rule in `_question`)."""
    view = {"id": name or card.get("id"), "number": card.get("number"),
            "kind": slot.get("kind"), "group": slot.get("group"),
            "paper_id": card.get("paper_id", ""),
            "dataset_id": slot.get("dataset_id") or card.get("dataset_id", ""),
            "outcome_key": slot.get("outcome_key") or card.get("outcome_key", ""),
            "measure_name": card.get("measure_name", ""),
            "unit": slot.get("unit") or card.get("unit", ""),
            "options": list(slot.get("options") or [])}
    return _single_override(view, answer)


def _precedence_answers(card: Mapping[str, Any],
                        answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """D1's three answers: take the pair, keep the printed value, or take the dataset out."""
    just, note = _stem(card, answer)
    option = next((o for o in card.get("options") or []
                   if o.get("key") == answer.get("option")), None)
    key = str((option or {}).get("key") or answer.get("option") or "")
    label = _short((option or {}).get("label"), 300)
    if key.startswith("use_candidates"):
        out = [
            {**_base(card, group=str(slot.get("group") or ""), kind="value",
                     mean=slot.get("mean"), dispersion_value=slot.get("dispersion_value"),
                     dispersion_type=slot.get("dispersion_type") or "", n=slot.get("n"),
                     unit=card.get("unit") or "", clears=[], overrules=[],
                     justification=f"{just} — {label or 'the candidate pair'}")}
            for slot in (option or {}).get("slots") or []
            if str(slot.get("group") or "") in ("A", "B") and slot.get("mean") is not None]
        if out:
            return out
        return [{**_base(card, kind="mark_reviewed", group=None, confidence="needs_human",
                         justification=f"{just} — the candidate pair is not on the record; "
                                       f"nothing could be written")}]
    # …and the page's own "Exclude these cells" button, which posts `{exclude: true}` and no
    # option at all. Reading only the option key sent that answer through the fallthrough below
    # and recorded the OPPOSITE decision — the card ticked "the printed value stands" and nothing
    # was excluded (review finding 11).
    if key == "exclude" or answer.get("exclude"):
        return [{**_base(card, group=None, kind="exclude_dataset",
                         justification=f"{just} — neither the printed value nor the reading is "
                                       f"usable; excluded")}]
    if key == "keep_printed":
        # a review decision on both cells that changes no number, and one the REBUILD has to see:
        # every release path re-resolves the row through `resolve_effect_with_fallback`, which
        # would raise the override again out from under the reviewer who just declined it. The
        # decision travels on the record (`keep_printed`), `overrides._prepare` offers that row no
        # alternatives, and the row resolves `not_convertible` — which is what "the printed value
        # stands" MEANS for a printed value with no spread (review finding 12).
        return [{**_base(card, group=group, kind="mark_reviewed", clears=[], overrules=[],
                         keep_printed=True, confidence="needs_human",
                         justification=f"{just} — the printed value stands; this row keeps no "
                                       f"effect size from it")}
                for group in ("A", "B")]
    from ..pipeline.overrides import OverrideRejected

    raise OverrideRejected(
        "this card offers three decisions — take the candidate pair, keep the printed value, or "
        f"exclude the dataset — and {key or 'nothing'!r} is none of them; an answer that decides "
        "nothing is not recorded as one that does")


def _analysed_n_answer(card: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """D4-lite's one record: the two arms' analysed sizes, for every outcome of the dataset."""
    just, note = _stem(card, answer)
    option = next((o for o in card.get("options") or []
                   if o.get("key") == answer.get("option")), None)
    sizes = {}
    for field in ("n_a", "n_b"):
        typed = answer.get(field)
        sizes[field] = typed if typed not in (None, "") else (option or {}).get(field)
    return {"kind": "group_n", "paper_id": card.get("paper_id", ""),
            "dataset_id": card.get("dataset_id", ""), "outcome_key": "", "group": None,
            "question_id": card.get("id", ""), "n_a": sizes["n_a"], "n_b": sizes["n_b"],
            "quote": str((option or {}).get("quote") or note or ""),
            "justification": f"{just} — analysed n {_num(sizes['n_a'])}/{_num(sizes['n_b'])}"
                             + (f": {_short((option or {}).get('label'), 200)}" if option else "")}


def _eligibility_answer(card: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """§C3's record: this review includes the paper, or it agrees that it does not."""
    just, note = _stem(card, answer)
    option = next((o for o in card.get("options") or []
                   if o.get("key") == answer.get("option")), None)
    decision = str((option or {}).get("decision") or answer.get("decision")
                   or ("exclude" if answer.get("exclude") else "")).strip().lower()
    if decision not in ("include", "exclude"):
        # INCLUDE was the default of every malformed answer — a note-only submit, a typo'd option
        # key, a decision spelled "excluded" or "no" — and an included paper buys a map and an
        # extraction at the next `--resume`. A paper the screen left out stays out until a
        # reviewer says the word, the way `include_dataset` has always required it (finding 13).
        from ..pipeline.overrides import OverrideRejected

        raise OverrideRejected(
            "an eligibility answer has to say include or exclude; "
            f"{decision or 'nothing'!r} decides neither, and the paper stays as the screen left it")
    return {"kind": "eligibility", "paper_id": card.get("paper_id", ""), "dataset_id": "",
            "outcome_key": "", "group": None, "question_id": card.get("id", ""),
            "eligible": decision == "include",
            "rule": str(answer.get("rule") or ""),
            "quote": str(answer.get("quote") or note or ""),
            "justification": f"{just} — "
                             f"{'included' if decision == 'include' else 'excluded'} by the "
                             f"reviewer"}


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
          f"{len(qs)} decision(s) need an answer. Each shows the picture the tool read, the "
          f"answers it is choosing between, and why it could not decide — one card per decision, "
          f"which may settle more than one cell. Answer in the review tab, or by appending to "
          f"`overrides.jsonl` (`canopy validate` re-pools).", ""]
    for q in qs:
        head = f"## {q['number']}. {_md(q['paper'])} — {_md(q['dataset_label'] or q['dataset_id'])}"
        if q.get("group_label"):
            head += f" — {_md(q['group_label'])}"
        md += [head, "", f"**{_md(q['prompt'])}**", ""]
        if q.get("status_line"):
            md += [f"_{_md(q['status_line'])}_", ""]
        if q.get("image", {}).get("path"):
            md += [f"![{q['kind']}]({q['image']['path']})", ""]
        for o in q.get("options") or []:
            extra = f" — backed by {_md(', '.join(o['backed_by']))}" if o.get("backed_by") else ""
            md.append(f"- **{_md(o['label'])}**{extra}")
        if q.get("free_text"):
            md.append("- _(or type the value / where to find it)_")
        if len(q.get("cells") or []) > 1 or q.get("scope") not in ("cell", "paper"):
            named = ", ".join(dict.fromkeys(
                f"{_md(c.get('dataset_id'))}/{_md(c.get('outcome_key'))}"
                + (f" ({_md(c.get('group_label') or c.get('group'))})" if c.get("group") else "")
                for c in q.get("cells") or []))
            md += ["", f"_This answer settles {len(q.get('cells') or [])} cell(s): {named}._"]
        for slot in q.get("slots") or []:
            if len(q.get("slots") or []) < 2 and not slot.get("answerable"):
                continue
            md += ["", f"_{_md(slot.get('group_label') or slot.get('group'))} was asked "
                       f"`{_md(slot.get('kind'))}`: {_md(slot.get('prompt'))}_"]
            # a slot answered on its own carries its own answers (§C1, second fold): the file
            # has to print them, or it prints a question with nothing to answer it by
            if slot.get("answerable"):
                for o in slot.get("options") or []:
                    extra = (f" — backed by {_md(', '.join(o['backed_by']))}"
                             if o.get("backed_by") else "")
                    md.append(f"  - **{_md(o['label'])}**{extra}")
            for was in slot.get("settled") or []:
                md.append(f"  - _already recorded ({_md(was.get('at'))}): "
                          f"{_md(was.get('justification'))}_")
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
