"""The review workflow: an append-only log of human decisions, and re-pooling under them.

A reviewer disagreeing with the tool is the point of the tool, so a disagreement is **evidence**,
not a correction: every decision is appended to `<run>/overrides.jsonl` with a justification, a
sequence number and a timestamp, and nothing is ever rewritten or deleted. The stage files the
pipeline wrote are never touched either — `repool` reads them, applies the log on top and rewrites
only the derived artefacts. That is what makes an override survive `--resume`: a resumed run
rebuilds the outputs from the same stage files, and re-applying the same log reproduces the same
reviewed analysis.

Six decisions (amendment I, plus C4's `orientation`):

| kind | what it does |
|---|---|
| `value` | replaces one group's mean / dispersion / spread type / n for one cell, and re-derives that row's effect size |
| `orientation` | settles which direction of one measure is more of the construct, for every cell of that measure |
| `mark_reviewed` | a human has checked the cell: it moves into the primary analysis at `accept_with_note` |
| `exclude_dataset` | that dataset leaves the analysis, and appears in the exclusion table |
| `eligibility` | a paper is in or out, whatever the mapper decided |
| `re_extract` | "read it again, with this hint" — needs a model, so it waits for the `--resume` that buys the reading |
| `include_dataset` | C7's map question answered: this dataset is in the review, or out under a named rule |
| `which_measure` | C6's map question answered: which measure this outcome carries |

`orientation` exists because the direction of a measure is not a cell value: it is decided once
per (paper, outcome, measure) and copied onto every cell that used it, so before C4 there was no
slot a human could write it into and `resolve` refused every such row with `orientation_unresolved`.
It is deliberately the *narrowest* decision in the log: it sets `higher_is_better` and nothing
else, it caps every cell it touches at `accept_with_note` so the sign a human supplied is visible
in the review table, and — unlike `mark_reviewed` — it does **not** stamp a bucket. A cell that was
held for orientation *and* something else is still held after it (see `_bucket_after_orientation`).

`repool` makes **no** model calls. It computes nothing itself either: values are re-derived by
`canopy.pipeline.resolve.resolve_effect` and pooled by `canopy.stats`, exactly as the run did, and
every artefact is written by the same writers `run.py` uses.

This lives beside the orchestrator rather than in the server because **both** entry points must
apply the log the same way: `run_pipeline` re-applies it at the end of every run (so an override
survives `canopy run --resume` on the command line, with no server involved), and the UI calls it
after each decision. `canopy/server/overrides.py` is a thin adapter over this module.
"""
from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from ..models import (Candidate, DatasetSpec, DispersionType, EffectSizeRecord, Protocol,
                      RunManifest, StudyMap, Verdict)
from ..verify.confidence import ROW_REFUSAL_CODES
from ..protocol import load_protocol
from ..stats.conversions import split_control
from ..report import (dump_json, exclusions_table, extraction_table, pool_rows, prisma_flow,
                      write_html_report, write_outcome_outputs, write_rows)
from .resolve import SHARED_CONTROL_ARM, ResolvedValues, resolve_effect_with_fallback
from .rows import (PreparedRow, converted_route, house_spread_type, prepare_rows,
                   shared_control_siblings)
from .state import (load_manifest, read_stage, review_entry, save_manifest, sha12,
                    sort_review_queue, stage_done)

__all__ = ["GROUP_STATISTICS", "HUMAN_OVERRIDE", "KINDS", "MAP_KINDS", "MAP_PENDING", "ORIENTATION_ANSWERED",
           "OVERRIDES_FILE", "OverrideRejected",
           "OVERRULABLE", "RE_EXTRACT_PENDING", "codes_cleared_by_value", "consumed_seqs",
           "recorded_flags", "recorded_holds", "row_flags", "append_override", "append_overrides",
           "read_overrides", "apply_overrides_and_repool", "map_answers", "eligibility_answers",
           "re_extract_answers", "override_summary", "repool_lock"]

OVERRIDES_FILE = "overrides.jsonl"
SUMMARY_FILE = "overrides_applied.json"
KINDS: tuple[str, ...] = ("value", "mark_reviewed", "exclude_dataset", "eligibility",
                          "re_extract", "orientation", "include_dataset", "which_measure",
                          "group_n")
#: the two kinds answered at the MAP stage: they decide what may be EXTRACTED, so — except for an
#: exclusion, which needs no reading — the pipeline applies them on the next `--resume`, not here.
MAP_KINDS: tuple[str, ...] = ("include_dataset", "which_measure")
MAP_PENDING = "extraction was never bought for this; re-run with --resume to extract it"
#: the other answer whose consequence is a model call. `run._extract` reads the log on `--resume`,
#: re-enters the extract stage for exactly the hinted cells and hands the reviewer's words to that
#: cell's readers — so the honest thing to tell a reviewer is what will happen, and when. It says
#: "both groups" because the extractors answer both arms in one call: a hint recorded against one
#: group re-reads the whole cell, and a reviewer is entitled to know that before they type it.
#: The questions page prints this text (DECISION §C4).
RE_EXTRACT_PENDING = ("recorded; the next --resume re-reads this cell — both groups — with the "
                      "hint")
_MAX_TEXT = 4000
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class OverrideRejected(ValueError):
    """An override Canopy will not record (no justification, no target, unknown kind)."""


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def repool_lock(run_dir: str | Path) -> threading.Lock:
    """One re-pool at a time per run: it rewrites every artefact the run owns."""
    return _lock_for(Path(run_dir) / "repool")


def _text(value: Any, limit: int = _MAX_TEXT) -> str:
    return str(value or "").strip()[:limit]


def _number(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise OverrideRejected(f"{field} must be a number, not {value!r}") from None


def _count(value: Any, field: str) -> int | None:
    """A number of PEOPLE — whole, at least one, and never quietly rounded.

    `int(payload["n"])` was three defects in one call (review finding 16). `ValueError` is not
    `OverrideRejected`, so "abc" — or the "1e3" an `<input type=number>` is entitled to emit —
    left the endpoint returning 500 where the contract says 422. `int(12.7)` silently became 12,
    a group size nobody stated standing in a record whose whole purpose is to say what a human
    said. And `bool` is an `int` in Python, so `True` was one participant.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise OverrideRejected(f"{field} must be a whole number of people, not {value!r}")
    number = _number(value, field)
    if number is None:
        return None
    if not math.isfinite(number) or number != int(number):
        raise OverrideRejected(f"{field} must be a whole number of people, not {value!r} — "
                               f"a size the tool rounded is a size nobody stated")
    if int(number) < 1:
        raise OverrideRejected(f"{field} must be at least 1; a group of nobody is an exclusion, "
                               f"not a size")
    return int(number)


# ----------------------------------------------------------------------------- the log
def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    kind = _text(payload.get("kind"), 40)
    if kind not in KINDS:
        raise OverrideRejected(f"unknown override kind {kind!r} (expected one of {list(KINDS)})")
    justification = _text(payload.get("justification"))
    if len(justification) < 3 and kind in MAP_KINDS:
        # a map answer's own record (below) carries `note`, not `justification`: the reviewer's
        # words are the justification, so the log takes them rather than refusing the record shape
        # the extract stage reads.
        justification = _text(payload.get("note")) or _text(payload.get("rule"))
    if len(justification) < 3:
        raise OverrideRejected("every override needs a justification — that is the record a "
                               "reader of the review will check")

    record: dict[str, Any] = {
        "kind": kind,
        "paper_id": _text(payload.get("paper_id"), 120),
        "dataset_id": _text(payload.get("dataset_id"), 200),
        "outcome_key": _text(payload.get("outcome_key"), 200),
        "group": payload.get("group") if payload.get("group") in ("A", "B") else None,
        # which question this answers, when it answers one. A cell is asked one thing at a time
        # and the next question may write the same KIND of override as the last (a `which_axis`
        # answer and a `quote_not_found` answer are both `value`), so the kind cannot say which
        # question has been answered — only the id can. Empty for a decision taken outside the
        # questions page.
        "question_id": _text(payload.get("question_id"), 200),
        "justification": justification,
    }
    if kind in ("value", "mark_reviewed", "exclude_dataset", "re_extract", "group_n") \
            and not record["dataset_id"]:
        raise OverrideRejected(f"a {kind} override needs a dataset_id")
    if kind in ("value", "mark_reviewed", "re_extract") and not record["outcome_key"]:
        raise OverrideRejected(f"a {kind} override needs an outcome_key")

    if kind == "value":
        if record["group"] is None:
            raise OverrideRejected("a value override needs group 'A' or 'B'")
        record.update({
            "mean": _number(payload.get("mean"), "mean"),
            "dispersion_value": _number(payload.get("dispersion_value"), "dispersion_value"),
            "dispersion_type": _text(payload.get("dispersion_type"), 20).upper() or "",
            "n": _count(payload.get("n"), "n"),
            "unit": _text(payload.get("unit"), 40),
            # the findings THIS answer settles, named by the question that offered it — "yes, this
            # series is this group" answers the series identity, which no field of the record
            # could otherwise say. Never a licence to clear more than the option named.
            "clears": [_text(code, 60) for code in (payload.get("clears") or [])
                       if _text(code, 60)][:12],
            "overrules": [name for name in (payload.get("overrules") or [])
                          if _text(name, 40) in OVERRULABLE][:len(OVERRULABLE)],
        })
        # C4: "the error bars are standard errors, not standard deviations" is a complete
        # answer to a complete question — it names the spread's TYPE and changes the effect size
        # (`_apply_value` re-derives the row through `resolve_effect`, which converts SE to SD
        # with n). Before this, the only question in the run carrying an `impact` produced an
        # override the log refused, so the question could not be answered at all.
        if all(record.get(k) is None for k in ("mean", "dispersion_value", "n")) \
                and not record["dispersion_type"]:
            raise OverrideRejected("a value override must set at least one of mean, "
                                   "dispersion_value, n or dispersion_type")
    if kind == "mark_reviewed":
        # C4: a review decision may name the findings it answers. When it does, it clears those
        # and the bucket is DERIVED from what is left — the reviewer is settling one thing, not
        # declaring the cell fit. A plain `mark_reviewed` (no `clears`) is the older, wider
        # decision: "I have checked this cell", and it stamps the bucket the reviewer chose.
        record["clears"] = [_text(code, 60) for code in (payload.get("clears") or [])
                            if _text(code, 60)][:12]
        # the findings that are not flag codes — a refutation, an adjudication, a score that never
        # cleared the line. A reviewer may disagree with each, and must say which: an answer that
        # names none of them retires none of them.
        # capped at how many overrulable findings there ARE, not at a literal: with four of them a
        # `3` silently drops one, and a cell held by the finding that was dropped stays held with
        # its question ticked (whole-diff H2 added the fourth).
        record["overrules"] = [name for name in (payload.get("overrules") or [])
                               if _text(name, 40) in OVERRULABLE][:len(OVERRULABLE)]
        bucket = _text(payload.get("confidence"), 40) or "accept_with_note"
        if bucket not in ("auto_accept", "accept_with_note", "needs_human"):
            raise OverrideRejected(f"unknown confidence bucket {bucket!r}")
        record["confidence"] = bucket
        # D1's third answer: "keep the printed value". It is a decision about how the ROW is
        # built, not about either cell's number, so it cannot be expressed as a clear or a bucket
        # — and without it every rebuild raised the precedence override again out from under the
        # reviewer who had just declined it (review finding 12).
        record["keep_printed"] = bool(payload.get("keep_printed"))
    if kind == "eligibility":
        if not record["paper_id"]:
            raise OverrideRejected("an eligibility override needs a paper_id")
        if not isinstance(payload.get("eligible"), bool):
            raise OverrideRejected("an eligibility override needs eligible: true or false")
        record["eligible"] = bool(payload["eligible"])
        # §C3: the criterion the decision was made under, and the paper's own words it rests on —
        # the same two fields an `include_dataset` answer carries, and for the same reason. A
        # reader of the review must be able to check an inclusion against the protocol, not merely
        # see that somebody made one.
        record["rule"] = _text(payload.get("rule"), 1000)
        record["quote"] = _text(payload.get("quote"), 1000)
    if kind == "re_extract":
        record["hint"] = _text(payload.get("hint"), 1000)
        if not record["hint"]:
            raise OverrideRejected("a re-extraction request needs a hint saying what to read")
        # a `categorical_axis_kind` answer travels structurally, not as prose: the re-read's
        # TargetSpec consumes it as a caller statement (resolver rule 1), which free text in the
        # reviewer-hint line can never do. "groups" = the x categories are the comparison arms;
        # "conditions" = the outcome is their average, so the re-read collapses this one cell.
        categorical = _text(payload.get("categorical_x"), 20)
        if categorical and categorical not in ("groups", "conditions"):
            raise OverrideRejected("categorical_x must be 'groups' or 'conditions' — what the "
                                   "figure's x categories are")
        record["categorical_x"] = categorical
    if kind == "orientation":
        # the direction of a measure is not a property of one group's cell, so an orientation
        # override never carries a group: it is scoped to (paper, outcome, measure) and applies
        # to both groups of every dataset in that paper measuring the same thing.
        record["group"] = None
        if not record["paper_id"]:
            raise OverrideRejected("an orientation override needs a paper_id: direction is "
                                   "decided once per paper's measure, not per cell")
        if not record["outcome_key"]:
            raise OverrideRejected("an orientation override needs an outcome_key")
        if not isinstance(payload.get("higher_is_better"), bool):
            raise OverrideRejected("an orientation override needs higher_is_better: true or "
                                   "false — whether a larger raw value on this measure means "
                                   "more of the construct")
        record["higher_is_better"] = bool(payload["higher_is_better"])
        # `measure_name` may be blank — an outcome whose map never named a measure has one, and
        # requiring the field would leave its direction question with no acceptable answer at all.
        # The scope is enforced where it can be seen instead: `_apply_orientation` refuses when the
        # rows it matched carry MORE THAN ONE distinct measure name, which is the case that
        # re-signed Tracking RMSE from a decision about Angular pointing error.
        # 1000, not 300: a measure's name is whatever the map wrote, and `_apply_orientation`
        # matches it by exact (normalised) equality — a cap shorter than the map's own field
        # would leave long-named measures with no acceptable answer at all.
        record["measure_name"] = _text(payload.get("measure_name"), 1000)
        record["quote"] = _text(payload.get("quote"), 1000)
    if kind == "group_n":
        # D4-lite. How many people were ANALYSED is a fact about the two arms, so it is scoped to
        # the dataset and to nothing narrower: it holds for every outcome measured on those
        # people, and a size answered for late adaptation that left the aftereffect on the
        # recruited count would put two different denominators behind one pair of groups.
        record["group"] = None
        record["outcome_key"] = ""
        for field in ("n_a", "n_b"):
            value = payload.get(field)
            size = _count(value, f"an analysed-n answer's {field}")
            if size is None:
                raise OverrideRejected(f"an analysed-n answer needs {field} as a whole number, "
                                       f"not {value!r}")
            record[field] = size
        record["quote"] = _text(payload.get("quote"), 1000)
    if kind in MAP_KINDS:
        # a map answer is about a dataset nobody has read yet, so it is scoped by ids alone —
        # there is no cell, no group and no value to name.
        record["group"] = None
        if not record["paper_id"]:
            raise OverrideRejected(f"a {kind} answer needs the paper_id of the map it answers")
        if not record["dataset_id"]:
            raise OverrideRejected(f"a {kind} answer needs a dataset_id")
        record["note"] = _text(payload.get("note"), 1000)
    if kind == "include_dataset":
        decision = _text(payload.get("decision"), 20).lower()
        if decision not in ("include", "exclude"):
            raise OverrideRejected("an include_dataset answer needs decision: 'include' or "
                                   "'exclude' — whether this review includes the dataset")
        record["decision"] = decision
        # C7: an exclusion may only be made on a NAMED rule, but the rule may be typed rather than
        # picked, and the note carries it either way — so this records what was given, and the
        # exclusion table prints it beside the dataset.
        record["rule"] = _text(payload.get("rule"), 1000)
        record["quote"] = _text(payload.get("quote"), 1000)
    if kind == "which_measure":
        if not record["outcome_key"]:
            raise OverrideRejected("a which_measure answer needs an outcome_key")
        metric = _text(payload.get("winning_analysis_metric"), 100)
        location = _text(payload.get("winning_location"), 300)
        if not metric and not location:
            raise OverrideRejected("a which_measure answer must name the winner: either "
                                   "winning_analysis_metric or winning_location")
        record["winning_analysis_metric"] = metric
        record["winning_location"] = location
        # …and which readings the answer is choosing AGAINST. `read_measure_answer` has always
        # accepted these — it is how a split whose two sides share one `analysis_metric` is settled
        # at all — and validation dropped them, so the one path that needed them most could not say
        # them: a reviewer switching an outcome printed one panel per group could name only one
        # panel as the winner, and every other reading, including the other group's, was set aside
        # with the measure they rejected.
        record["losing_locations"] = [_text(x, 300) for x in
                                      (payload.get("losing_locations") or [])[:12]
                                      if _text(x, 300)]
    return record


def _recorded_verdict(run_dir: str | Path, record: Mapping[str, Any]) -> dict[str, Any] | None:
    """This cell as the RUN recorded it, read from the verify stage file.

    The stage file rather than the re-pool's working copies, because that is the only fixed
    reference: replaying a log mutates the copies, so an answer that legitimately named a finding
    would be refused the second time it was applied.
    """
    dataset_id = str(record.get("dataset_id") or "")
    outcome_key = str(record.get("outcome_key") or "")
    if not dataset_id or not outcome_key:
        return None
    group = record.get("group")
    paper_id = str(record.get("paper_id") or "")
    root = Path(run_dir) / "papers"
    files = [root / sha12(paper_id) / "verify.json"] if paper_id else sorted(
        root.glob("*/verify.json"))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for verdict in payload.get("verdicts") or []:
            if (verdict.get("dataset_id") == dataset_id
                    and verdict.get("outcome_key") == outcome_key
                    and (group is None or verdict.get("group") in (group, None))):
                return verdict
    return None


def row_flags(run_dir: str | Path, record: Mapping[str, Any]) -> list[str]:
    """The flags on the ROW this cell belongs to — the run's own record, plus the re-pool's.

    The row is where the resolver's refusals live (C9's |d| screen), and a cell is held by them
    without carrying one itself — so both the questions page and the validator have to be able to
    see them.

    BOTH references, deliberately. The stage file is the fixed one: `results/…` is an artefact a
    re-pool rewrites, so reading only that made an answer's admissibility depend on the previous
    re-pool's output, and a log replayed onto fresh stage files with no table at all fell back to
    the record's own `carried` list (whole-diff L7). But a refusal can also be one a REVIEWER's
    own numbers produced — a value answer whose |d| the screen then refuses — which exists only
    in the re-pooled table, and an answer must be able to name that too.
    """
    return sorted({str(flag) for row in (_row_of(run_dir, record), _table_row_of(run_dir, record))
                   for flag in row.get("flags") or []})


def _row_of(run_dir: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """This cell's row as the RUN resolved it, read from the resolve stage file.

    The stage file, for the same reason `_recorded_verdict` reads one and not the working copies:
    it is the only fixed reference. `results/extraction_table_all.json` is an artefact a re-pool
    REWRITES, so validating an answer against it made the answer's admissibility depend on the
    previous re-pool's output — a log replayed onto fresh stage files with no table at all fell
    back to the record's own `carried` list alone (whole-diff L7). The table is still the fallback
    for a run directory whose resolve stage was never written.
    """
    dataset_id = str(record.get("dataset_id") or "")
    outcome_key = str(record.get("outcome_key") or "")
    if not dataset_id or not outcome_key:
        return {}
    paper_id = str(record.get("paper_id") or "")
    root = Path(run_dir) / "papers"
    files = [root / sha12(paper_id) / "resolve.json"] if paper_id else sorted(
        root.glob("*/resolve.json"))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in payload.get("records") or []:
            if isinstance(row, dict) and row.get("dataset_id") == dataset_id \
                    and row.get("outcome_key") == outcome_key:
                return row
    return _table_row_of(run_dir, record)


def _table_row_of(run_dir: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """This cell's row in the run's own table — what the LAST re-pool derived (`{}` if none)."""
    dataset_id = str(record.get("dataset_id") or "")
    outcome_key = str(record.get("outcome_key") or "")
    try:
        rows = json.loads((Path(run_dir) / "results" / "extraction_table_all.json"
                           ).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("dataset_id") == dataset_id \
                and row.get("outcome_key") == outcome_key:
            return row
    return {}


def recorded_flags(run_dir: str | Path, record: Mapping[str, Any]) -> set[str] | None:
    """The check codes the run recorded for this cell, or `None` when it has no such cell."""
    verdict = _recorded_verdict(run_dir, record)
    if verdict is None:
        return None
    return {str(flag.get("code") or "") for flag in verdict.get("flags") or []}


def recorded_holds(run_dir: str | Path, record: Mapping[str, Any]) -> set[str] | None:
    """EVERYTHING holding this cell that an answer could name: the flag codes, plus the three
    findings that are not codes at all — a verifier's refutation, an adjudication, and a score
    that never reached the acceptance line. One set, so one rule validates `clears` and
    `overrules` alike, and one list on the record (`carried`) is the whole reference."""
    from ..verify.confidence import ACCEPT_WITH_NOTE

    verdict = _recorded_verdict(run_dir, record)
    if verdict is None:
        return None
    holds = {str(flag.get("code") or "") for flag in verdict.get("flags") or []}
    # …and what the ROW carries, which is holding this cell just as surely: C9's screen on the
    # resolved |d| is a finding no cell can carry, and an answer that addresses it must be able to
    # name it.
    holds |= set(row_flags(run_dir, record)) & ROW_REFUSALS
    if verdict.get("verifier_verdict") == "refuted":
        holds.add("verifier_refuted")
    if verdict.get("adjudicated"):
        holds.add("adjudicated")
    if (verdict.get("confidence_score") or 0.0) < ACCEPT_WITH_NOTE:
        holds.add("low_score")
    if verdict.get("mean") is None \
            and converted_route(str(_row_of(run_dir, record).get("route") or "")):
        # the fourth non-code finding (whole-diff H2): this cell has no number of its own, its row
        # took a route that did not need one, and the only answer there is is a person accepting
        # the conversion. Named here so `overrules: ["no_group_values"]` is validated exactly like
        # a refutation or an adjudication — against what the cell was actually holding.
        holds.add("no_group_values")
    return holds


def _check_clears(run_dir: str | Path, record: Mapping[str, Any]) -> str:
    """"" if this answer names only findings its cell carries, else why it does not.

    `clears` is the one field of an answer that RETIRES evidence, so it may name only what is
    actually on the cell. Without this a record — hand-written, or an option built from a stale
    question — could retire a finding that was never raised, and the record would read as though a
    reviewer had settled something nobody had found.
    """
    named = [str(x) for x in (record.get("clears") or []) if str(x)]
    named += [str(x) for x in (record.get("overrules") or []) if str(x)]
    if not named:
        return ""
    # what the reviewer was shown (`carried`, written at append time) OR what the cell holds now.
    # The union is deliberate, and it is all the guarantee there is: a record that supplies its own
    # `carried` is trusted about it, because nothing can prove whether a code missing from the
    # stage file went legitimately (a re-extraction found the quote) or was never there. What the
    # check does buy is that a code in NEITHER — a finding no version of this cell ever had — is
    # refused. Clearing one the cell does not carry is a no-op on the flags either way.
    now = recorded_holds(run_dir, record)
    stored = {str(code) for code in record.get("carried") or []}
    if now is None and not stored:      # no such cell in this run: other checks report that
        return ""
    carried = stored | (now or set())
    unknown = sorted({code for code in named if code not in carried})
    if not unknown:
        return ""
    return (f"this answer names a finding this cell does not carry: {', '.join(unknown)}"
            + (f" (it carries {', '.join(sorted(carried))})" if carried
               else " (the cell carries none)"))


def append_override(run_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and append one decision. Returns the record as it was written."""
    return append_overrides(run_dir, [payload])[0]


def append_overrides(run_dir: str | Path,
                     payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Append several decisions as ONE act: all are validated before any of them is written.

    One answer is one record per CELL it names (§C1), and appending them one at a time meant a
    refusal on the second left the first in the log — half a decision on the record, with no
    re-pool behind it and nothing in the response to say so (review MINOR 29). They are also
    written under one lock, so no other writer can land between the halves of one answer.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OVERRIDES_FILE
    checked = [(_checked(directory, payload), payload) for payload in payloads]
    with _lock_for(path):
        return [_write(directory, path, record, payload) for record, payload in checked]


def _checked(directory: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The record this payload would be written as — or `OverrideRejected`. Writes nothing."""
    record = _validate(payload)
    if record.get("clears") or record.get("overrules"):
        # what the cell was holding when the reviewer answered, on the record. A `--resume`
        # rewrites the verify stage file, and without this the same answer would be re-validated
        # against a different cell later and silently fall out of the analysis.
        holds = recorded_holds(directory, record)
        if holds is not None:
            record["carried"] = sorted(holds)
    refused = _check_clears(directory, record)
    if refused:
        raise OverrideRejected(refused)
    return record


def _write(directory: Path, path: Path, record: dict[str, Any],
           payload: Mapping[str, Any]) -> dict[str, Any]:
    """One validated record onto the end of the log. The caller holds the lock."""
    record["seq"] = len(read_overrides(directory)) + 1
    record["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["actor"] = _text(payload.get("actor"), 80) or "local reviewer"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def read_overrides(run_dir: str | Path) -> list[dict[str, Any]]:
    """Every decision, in the order it was made. A damaged line is reported, never skipped."""
    path = Path(run_dir) / OVERRIDES_FILE
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            out.append({"kind": "unreadable", "seq": number, "justification": "",
                        "error": f"line {number} of {OVERRIDES_FILE} is not JSON"})
            continue
        record.setdefault("seq", number)
        out.append(record)
    return out


def override_summary(run_dir: str | Path) -> dict[str, Any]:
    """What the last `repool` did, or an empty summary when it has never run."""
    path = Path(run_dir) / SUMMARY_FILE
    if not path.exists():
        return {"applied": 0, "pending": [], "excluded": [], "outcomes": {}, "at": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:                                     # pragma: no cover - defensive
        return {"applied": 0, "pending": [], "excluded": [], "outcomes": {}, "at": ""}


def _live_overrides(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The log with the superseded MAP answers dropped — the same last-wins rule `map_answers`
    applies, so the analysis and the page can never act on different answers to one question.

    Only the map kinds are deduplicated. Every other kind is cumulative by design: two `value`
    answers on the same cell are a correction *and* its history, and both are meant to be applied
    in order. A map answer is not cumulative — "include it" then "exclude it" is one decision that
    changed its mind, and applying both means excluding a dataset the reviewer has since kept.
    """
    live: list[dict[str, Any]] = []
    latest: dict[tuple[str, str, str, str], int] = {}
    for record in records:
        if record.get("kind") not in KINDS:
            continue
        if record.get("kind") in MAP_KINDS:
            key = (str(record.get("kind")), str(record.get("paper_id") or ""),
                   str(record.get("dataset_id") or ""), str(record.get("outcome_key") or ""))
            if key in latest:
                live[latest[key]] = None            # type: ignore[assignment]
            latest[key] = len(live)
        live.append(dict(record))
    return [record for record in live if record is not None]


def consumed_seqs(run_dir: str | Path) -> set[int]:
    """The `seq` of every override a pipeline stage has already acted on.

    An answer that needs a model call — a re-extraction, an inclusion, a choice of measure — is
    "pending" until the run that buys it says it consumed it. The stages record that themselves:
    `map.json` and `extract.json` each carry `consumed_override_seqs: list[int]`. Reading it here
    is what lets a pending answer stop being pending; without it "not applied yet" is permanent
    and a reviewer cannot tell which decisions are still outstanding.
    """
    out: set[int] = set()
    for stage in ("map", "extract"):
        for path in sorted(Path(run_dir).glob(f"papers/*/{stage}.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict):
                out |= {int(seq) for seq in payload.get("consumed_override_seqs") or []
                        if isinstance(seq, int)}
    return out


def map_answers(run_dir: str | Path, paper_id: str) -> list[dict[str, Any]]:
    """Validated override records of kinds include_dataset / which_measure for this paper, in
    log order (later answers win). Empty list when there is no overrides file.

    This is the whole interface between the review page and the map stage: `run.py` calls it
    before mapping a paper on `--resume` and hands the answers to `apply_map_answers`, so a
    question answered in the UI is applied by the pipeline itself rather than by the UI reaching
    into a stage file. One answer per question survives — a reviewer who changes their mind
    changes the review, and the log still holds both, which is what makes the change auditable.
    """
    wanted = sha12(str(paper_id or ""))
    if not wanted:                    # "" once meant "every paper's answers", which is not a paper
        raise ValueError("map_answers needs a paper_id; '' is not a paper")
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in read_overrides(run_dir):
        if raw.get("kind") not in MAP_KINDS:
            continue
        if sha12(str(raw.get("paper_id") or "")) != wanted:
            continue
        try:
            record = _validate(raw)
        except OverrideRejected:            # a record this log would not accept is not an answer
            continue
        record.update({key: raw[key] for key in ("seq", "at", "actor") if key in raw})
        key = (record["kind"], record["dataset_id"], record["outcome_key"])
        latest.pop(key, None)               # the later answer wins, and stands at its own place
        latest[key] = record
    return list(latest.values())


def re_extract_answers(run_dir: str | Path, paper_id: str) -> list[dict[str, Any]]:
    """Validated `re_extract` records that could name a cell of this paper, in log order.

    The sibling of `map_answers`, for the answer whose consequence is a READING: `run._extract`
    calls it on `--resume`, re-enters the extract stage for exactly the cells these records name
    and hands their hints to that cell's readers.

    Cumulative, not last-wins. A map answer is one decision that may change its mind, so applying
    a superseded one excludes a dataset the reviewer has since kept; two hints on one cell are two
    places a person looked, and a reader given both looks in both. That is also what keeps the
    record honest: every hint on a cell is consumed by the reading that was bought for it, so none
    is left saying "not applied yet" about a re-read that happened.

    A record with no `paper_id` is offered to every paper — the manual override form records one
    (`test_server`'s does), and the caller matches on `dataset_id`, which is minted per paper and
    so belongs to exactly one map.
    """
    wanted = sha12(str(paper_id or ""))
    if not wanted:
        raise ValueError("re_extract_answers needs a paper_id; '' is not a paper")
    out: list[dict[str, Any]] = []
    for raw in read_overrides(run_dir):
        if raw.get("kind") != "re_extract":
            continue
        named = str(raw.get("paper_id") or "")
        if named and sha12(named) != wanted:
            continue
        try:
            record = _validate(raw)
        except OverrideRejected:            # a record this log would not accept is not an answer
            continue
        record.update({key: raw[key] for key in ("seq", "at", "actor") if key in raw})
        out.append(record)
    return out


def eligibility_answers(run_dir: str | Path, paper_id: str) -> list[dict[str, Any]]:
    """Validated `eligibility` records for this paper, in log order — the later answer wins.

    The sibling of `map_answers`, and the whole interface between §C3's `include_paper` card and
    the run: `run._answered_eligibility` calls it before the mapper's own verdict is acted on, so
    a paper a reviewer put back into the review is mapped, extracted and resolved by the pipeline
    itself rather than by anyone editing a stage file.
    """
    wanted = sha12(str(paper_id or ""))
    if not wanted:
        raise ValueError("eligibility_answers needs a paper_id; '' is not a paper")
    out: list[dict[str, Any]] = []
    for raw in read_overrides(run_dir):
        if raw.get("kind") != "eligibility":
            continue
        if sha12(str(raw.get("paper_id") or "")) != wanted:
            continue
        try:
            record = _validate(raw)
        except OverrideRejected:            # a record this log would not accept is not an answer
            continue
        record.update({key: raw[key] for key in ("seq", "at", "actor") if key in raw})
        out.append(record)
    return out


# ----------------------------------------------------------------------------- run state
class _RunState:
    """Everything the stage files hold, indexed the way re-pooling needs it."""

    def __init__(self, run_dir: Path, manifest: RunManifest):
        #: where this state was read from. Held so the rules that need the map AS THE REVIEW HAS
        #: IT — the map answers applied — can ask for it without every caller threading the path
        #: down. `_out_of_reach` is the one that does.
        self.run_dir = Path(run_dir)
        self.records: list[EffectSizeRecord] = []
        self.verdicts: list[Verdict] = []
        self.candidates: list[Candidate] = []
        self.datasets: dict[str, DatasetSpec] = {}
        self.studies: dict[str, StudyMap] = {}
        self.paper_of: dict[str, str] = {}
        #: D4-lite: the analysed group sizes a reviewer answered, per dataset. Kept on the STATE
        #: rather than passed down each applier, because it must reach every later rebuild of
        #: every row of that dataset — an n answered before a value answer must still be the n the
        #: value answer's row is built with, whatever order the log happens to be in.
        self.group_n: dict[str, tuple[int, int]] = {}
        #: WHEN each answer about a group size was made (`seq` in the log). The dataset-level
        #: analysed n has to reach every later rebuild of every row — and must not overwrite a
        #: per-cell `n` a reviewer answered AFTER it, which is how the table came to show one
        #: size while the row was divided by another (review MINOR 30).
        self.group_n_seq: dict[str, int] = {}
        self.n_answered: dict[tuple[str, str, str], int] = {}
        #: D1's "keep the printed value" answers, as `(dataset_id, outcome_key)`. Held here for
        #: the same reason `group_n` is: it must reach EVERY later rebuild of that row, whatever
        #: else is answered afterwards and in whatever order the log happens to be in.
        self.keep_printed: set[tuple[str, str]] = set()
        for status in manifest.papers:
            paper_id = status.paper_id
            if stage_done(run_dir, paper_id, "map"):
                study = StudyMap.model_validate(read_stage(run_dir, paper_id, "map")["study"])
                self.studies[paper_id] = study
                for dataset in study.datasets:
                    self.datasets[dataset.dataset_id] = dataset
                    self.paper_of[dataset.dataset_id] = paper_id
            if stage_done(run_dir, paper_id, "extract"):
                self.candidates.extend(
                    Candidate.model_validate(c)
                    for c in read_stage(run_dir, paper_id, "extract")["candidates"])
            if stage_done(run_dir, paper_id, "verify"):
                payload = read_stage(run_dir, paper_id, "verify")
                self.verdicts.extend(Verdict.model_validate(v) for v in payload["verdicts"])
                self.candidates.extend(Candidate.model_validate(c)
                                       for c in payload.get("extra_candidates", []))
            if stage_done(run_dir, paper_id, "resolve"):
                self.records.extend(
                    EffectSizeRecord.model_validate(r)
                    for r in read_stage(run_dir, paper_id, "resolve")["records"])

    def verdict(self, dataset_id: str, outcome_key: str, group: str) -> Verdict | None:
        for verdict in self.verdicts:
            if (verdict.dataset_id == dataset_id and verdict.outcome_key == outcome_key
                    and verdict.group == group):
                return verdict
        return None


# ----------------------------------------------------------------------------- repool
def apply_overrides_and_repool(run_dir: str | Path, *, protocol_path: str | Path | None = None
                               ) -> dict[str, Any]:
    """Re-pool a finished run with the override log applied, and rewrite its artefacts.

    Makes no model calls. Returns `{"applied", "pending", "excluded", "outcomes", "at"}` and
    writes the same summary to `overrides_applied.json`, which the results endpoint serves so a
    reader can see which numbers a human changed.
    """
    out = Path(run_dir)
    manifest = load_manifest(out)
    protocol = _protocol_for(out, manifest, protocol_path)
    settings = protocol.stats
    overrides = _live_overrides(read_overrides(out))
    consumed = consumed_seqs(out)
    state = _RunState(out, manifest)

    applied: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    records = {(r.dataset_id, r.outcome_key): r.model_copy(deep=True) for r in state.records}
    verdicts = {(v.dataset_id, v.outcome_key, v.group): v.model_copy(deep=True)
                for v in state.verdicts}
    dropped: set[tuple[str, str]] = set()
    # what the log has overruled on each cell so far. A refutation a reviewer addressed in one
    # answer is addressed for the whole log: the next answer on that cell must not be judged as
    # though nobody had ever read the objection, or the cell can never be released and the page
    # (which reads the log the same way) would show an answered question on a held cell.
    overruled: dict[tuple[str, str, str], set[str]] = {}
    #: cells whose number a human has supplied. "The readers disagree" is answered by a mean, and
    #: it stays answered: a later decision about the same cell must not be judged as though the
    #: disagreement were still open.
    means_answered: set[tuple[str, str, str]] = set()
    #: rows the log touched. The consistency pass below only speaks about these: a row nobody
    #: answered keeps exactly what the run decided about it, whatever this module can or cannot
    #: see of the reason.
    touched_rows: set[tuple[str, str]] = set()

    for override in overrides:
        kind = override["kind"]
        dataset_id = override.get("dataset_id", "")
        outcome_key = override.get("outcome_key", "")
        if kind == "re_extract":
            if override.get("seq") in consumed:
                applied.append(override)
                continue
            # …and when no resume can act on it, the record says what is in the way instead of
            # promising a reading nothing will ever buy (C4). The rule is the extract stage's own
            # (`mapper.unreadable_cell`), read off the same map, so the page and the run cannot
            # disagree about whether a hint is going anywhere.
            blocked = _hint_out_of_reach(out, override, state, protocol)
            pending.append({**override, "why": blocked or RE_EXTRACT_PENDING})
            continue
        if kind == "eligibility":
            hit = _apply_eligibility(override, records, dropped, excluded)
            (applied if hit else pending).append(override if hit else {
                **override, "why": "this paper has no extracted data in the run directory, so "
                                   "including it needs a model call (`canopy run --resume`)"})
            continue
        if kind == "exclude_dataset":
            hit = _apply_exclude(override, records, dropped, excluded, state)
            (applied if hit else pending).append(override if hit else {
                **override, "why": f"no row for dataset {dataset_id!r} in this run"})
            continue
        if kind == "include_dataset":
            # "exclude it" is the one map answer that needs no reading: the dataset leaves the
            # analysis exactly as an `exclude_dataset` decision does, with the rule it was
            # excluded under beside it. "include it" is a decision about what to EXTRACT, and
            # extraction is a model call, so it waits for the resume that buys it.
            if override.get("decision") == "exclude":
                hit = _apply_exclude(override, records, dropped, excluded, state, always=True)
                (applied if hit else pending).append(override if hit else {
                    **override, "why": f"no such dataset in this run ({dataset_id!r})"})
            elif override.get("seq") in consumed:
                applied.append(override)
            else:
                # …and WHY it waits. `MAP_PENDING` says "extraction was never bought for this",
                # which is true of an answer to a question that blocked its cell and false of one
                # that overrules a measure the map settled for itself: that cell WAS read — against
                # the reading the answer rejects — and what the resume owes it is a re-reading, not
                # a first one. The audit file and the review page must not disagree about which.
                pending.append({**override, "why": _map_pending_why(override, state)})
            continue
        if kind == "which_measure":
            (applied if override.get("seq") in consumed
             else pending).append(override if override.get("seq") in consumed
                                  else {**override, "why": MAP_PENDING})
            continue
        if kind == "mark_reviewed":
            if override.get("keep_printed"):
                # before anything is rebuilt in this branch: the probe below goes through
                # `_prepare` too, and a row prepared with the alternatives still on it would
                # answer the row's own findings as the overridden row rather than the held one.
                state.keep_printed.add((dataset_id, outcome_key))
            record = records.get((dataset_id, outcome_key))
            if record is None:
                pending.append({**override, "why": f"no row for {dataset_id}/{outcome_key}"})
                continue
            refused = _check_clears(out, override)
            if refused:
                pending.append({**override, "why": refused})
                continue
            clears = _cleared_on(override, {str(code) for code in override.get("clears") or []})
            named = override.get("group")
            asked = override.get("confidence", "accept_with_note")
            buckets: list[str] = []
            for group in ("A", "B"):                       # the clearing, before the row is rebuilt
                verdict = verdicts.get((dataset_id, outcome_key, group))
                if verdict is not None and clears and named in (None, group):
                    verdict.flags = [f for f in verdict.flags if f.code not in clears]
            # …and the ROW, through the same resolver every other release path uses. Marking cells
            # reviewed used to set the row's bucket from the two cells' and never re-derive the
            # rules that live on the row, so "I opened the figure and this value is what it shows"
            # — true of both cells — pooled a row whose DENOMINATOR the screen had refused.
            dataset = state.datasets.get(dataset_id)
            cell_a = verdicts.get((dataset_id, outcome_key, "A"))
            cell_b = verdicts.get((dataset_id, outcome_key, "B"))
            row_flags: list[str] = list(record.flags)
            row_route = str(record.route or "")
            rebuildable = dataset is not None and cell_a is not None and cell_b is not None
            if rebuildable:                                # the PROBE, for the row's own findings
                probe = _rebuild_row(record, dataset, cell_a, cell_b, protocol,
                                     outcome_key, override["justification"],
                                     state=state, verdicts=verdicts)
                row_flags, row_route = list(probe.flags), str(probe.route or "")
            for group in ("A", "B"):
                verdict = verdicts.get((dataset_id, outcome_key, group))
                if verdict is None:
                    continue
                if named not in (None, group):
                    # a review decision about group A is not one about group B: it leaves the
                    # other cell exactly as it was, and the ROW stays held while that cell is.
                    buckets.append(verdict.confidence)
                    continue
                # ALWAYS derived, and never above `accept_with_note`. "A human has checked this
                # cell" is a reason to show the cell in the review table, not authority over a
                # refutation or an adjudication the reviewer did not mention: naming nothing
                # retires nothing. `needs_human` is still honoured — a reviewer may say "keep it".
                other = verdicts.get((dataset_id, outcome_key, "B" if group == "A" else "A"))
                names = overruled.setdefault((dataset_id, outcome_key, group), set())
                names |= {str(x) for x in override.get("overrules") or []}
                cell = "needs_human" if asked == "needs_human" else _derived_bucket(
                    verdict, overrules=names, other=other, row_flags=row_flags,
                    row_route=row_route,
                    mean_answered=(dataset_id, outcome_key, group) in means_answered)
                verdict.confidence = cell
                verdict.needs_human = cell == "needs_human"
                buckets.append(cell)
            # …and the row LAST, from the cells as they now stand, so it can never carry the
            # bucket of the answer it has just been given.
            if rebuildable:
                record = _rebuild_row(record, dataset, cell_a, cell_b, protocol, outcome_key,
                                      override["justification"], state=state, verdicts=verdicts)
            # the STRICTER of the two, never just the cells'. The resolver puts rules on the row
            # that no cell carries — the conversion gate raises `df_missing` while the effect size
            # is being built, on the ROW only — and overwriting the row's bucket with the cells'
            # minimum discarded them. The other two appliers keep the resolver's by not touching
            # it; this one has cell buckets of its own to combine, so it says so.
            from_cells = ("needs_human" if "needs_human" in buckets
                          else buckets[0] if buckets
                          else override.get("confidence", "accept_with_note"))
            record.confidence = _stricter(record.confidence, from_cells)
            record.flags = sorted({*record.flags, "human_reviewed"})
            records[(dataset_id, outcome_key)] = record
            touched_rows.add((dataset_id, outcome_key))
            applied.append(override)
            continue
        if kind == "orientation":
            ok, why = _apply_orientation(override, records, verdicts, state, protocol,
                                         touched_rows, overruled)
            (applied if ok else pending).append(override if ok else {**override, "why": why})
            continue
        if kind == "group_n":
            ok, why = _apply_analysed_n(override, records, verdicts, state, protocol,
                                        touched_rows)
            (applied if ok else pending).append(override if ok else {**override, "why": why})
            continue
        if kind == "value":
            refused = _check_clears(out, override)
            if refused:
                pending.append({**override, "why": refused})
                continue
            names = overruled.setdefault(
                (dataset_id, outcome_key, str(override.get("group") or "")), set())
            names |= {str(x) for x in override.get("overrules") or []}
            ok, why = _apply_value(override, records, verdicts, state, protocol, names)
            if ok:
                touched_rows.add((dataset_id, outcome_key))
            if ok and override.get("mean") is not None:
                means_answered.add((dataset_id, outcome_key, str(override.get("group") or "")))
            (applied if ok else pending).append(override if ok else {**override, "why": why})

    _reconcile(records, verdicts, touched_rows, dropped)
    kept = [record for key, record in records.items() if key not in dropped]
    outcomes = _rewrite(out, manifest, protocol, kept, state, verdicts, excluded)
    summary = {"applied": len(applied), "pending": pending, "excluded": excluded,
               "outcomes": outcomes, "overrides": applied,
               "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    dump_json(summary, out / SUMMARY_FILE)
    return summary


def _out_of_reach(state: "_RunState", dataset_id: str, outcome_key: str, protocol: Protocol,
                  paper_id: str = "") -> str:
    """Why this run's map puts this cell beyond any answer about its numbers — `""` when it does not.

    ONE rule for BOTH answers whose subject is a cell of the map: the `re_extract` hint that asks
    for it to be READ, and the `value` that supplies its numbers directly. They must refuse the
    same cells for the same reason, or the review says two opposite things about one dataset —
    which is what it did: the hint on Roller's E1a was refused because the map excluded the
    dataset under a protocol rule, while a typed value on the same cell built the row, put its
    cells back in the queue and let the ordinary confirmation pool it. An answer that never named
    the exclusion may not undo it (§C4), and the way to keep the dataset is to say so:
    `include_dataset`, which `apply_map_answers` now honours over a map's own exclusion.

    The map as the REVIEW has it — the stage file with the log's map answers applied — which is
    the same map `run._extract` reads.
    """
    from ..agents.mapper import apply_map_answers, unreadable_cell

    paper = state.paper_of.get(dataset_id) or paper_id
    study = state.studies.get(paper)
    if study is None:
        return f"no map for {dataset_id!r} in this run, so no reader can be sent to it"
    answered = apply_map_answers(study, map_answers(state.run_dir, paper))
    return unreadable_cell(answered, dataset_id, outcome_key, {o.key for o in protocol.outcomes})


#: what a map answer waits for when the cell it names was ALREADY read — against the very reading
#: the answer rejects. The other half of `MAP_PENDING`, which promises a first extraction.
MEASURE_PENDING = ("recorded; this cell was read from the measure you rejected, so the next "
                   "--resume reads it again from the one you chose — the readings taken against "
                   "the other do not stand as a fallback")


def _map_pending_why(override: Mapping[str, Any], state: "_RunState") -> str:
    """Which of the two things a map answer is waiting for. Read from the MAP, not from the kind.

    A `which_measure` answer to an open question un-blocks a cell nobody read; one that overrules a
    ruling the map made for itself lands on a cell the run already read. Telling a reviewer
    "extraction was never bought for this" about the second is false about the thing they can check.
    """
    from ..agents.mapper import settled_measure

    if str(override.get("kind") or "") != "which_measure":
        return MAP_PENDING
    paper = str(override.get("paper_id") or "")
    dataset_id = str(override.get("dataset_id") or "")
    study = state.studies.get(state.paper_of.get(dataset_id) or paper)
    if study is None:
        return MAP_PENDING
    return (MEASURE_PENDING
            if settled_measure(study, dataset_id, str(override.get("outcome_key") or ""))
            else MAP_PENDING)


def _hint_out_of_reach(out: Path, override: Mapping[str, Any], state: "_RunState",
                       protocol: Protocol) -> str:
    """`_out_of_reach` for a `re_extract` record, plus the answer that would lift the block.

    The refusal is not a dead end: a dataset the map excluded is one an `include_dataset` ruling
    puts back, and a message that says what is in the way without saying what to do about it
    leaves the reviewer exactly where the false promise did.
    """
    why = _out_of_reach(state, str(override.get("dataset_id") or ""),
                        str(override.get("outcome_key") or ""), protocol,
                        str(override.get("paper_id") or ""))
    if why and "excluded this dataset" in why:
        why += ("; answer the dataset's inclusion with `include_dataset` / include if this review "
                "keeps it, and the next --resume reads the cell with this hint")
    return why


def _protocol_for(out: Path, manifest: RunManifest,
                  protocol_path: str | Path | None) -> Protocol:
    for candidate in (protocol_path, out / "protocol.yaml", manifest.protocol_path):
        if candidate and Path(candidate).exists():
            return load_protocol(candidate)
    raise FileNotFoundError(f"no protocol to re-pool {out} with")


def _apply_exclude(override: Mapping[str, Any], records: dict[tuple[str, str], EffectSizeRecord],
                   dropped: set[tuple[str, str]], excluded: list[dict[str, Any]],
                   state: "_RunState", *, always: bool = False) -> bool:
    """Take a dataset (or one of its cells) out of the analysis, and record why it left.

    `always` is for a decision taken at the MAP stage: nothing was extracted, so there is no row
    to drop — and the decision is still the analysis-level fact that the dataset is out, which a
    reader of the review must be able to see in the exclusion table rather than infer from an
    absence.
    """
    dataset_id = str(override.get("dataset_id") or "")
    outcome_key = str(override.get("outcome_key") or "")
    detail = "; ".join(x for x in (str(override.get("rule") or ""),
                                   override["justification"]) if x)

    def entry(d_id: str, o_key: str) -> dict[str, Any]:
        return {"paper_id": state.paper_of.get(dataset_id, "") or str(override.get("paper_id")
                                                                      or ""),
                "filename": "", "dataset_id": d_id, "outcome_key": o_key, "stage": "review",
                "reason": "human_override", "quote": str(override.get("quote") or ""),
                "decider": override.get("actor", "human"), "detail": detail}

    targets = [key for key in records
               if key[0] == dataset_id and (not outcome_key or key[1] == outcome_key)]
    for key in targets:
        dropped.add(key)
        excluded.append(entry(key[0], key[1]))
    if targets:
        return True
    # nothing was extracted for it: that is the normal state of a dataset a map question blocked,
    # and the decision still belongs in the exclusion table — but ONLY if this run's map actually
    # has the dataset. An answer naming an id nobody mapped is a mistake, not an exclusion, and
    # writing a row for it would put a dataset that never existed into the PRISMA count.
    if always and dataset_id in state.datasets:
        excluded.append(entry(dataset_id, outcome_key))
        return True
    return False


def _apply_eligibility(override: Mapping[str, Any],
                       records: dict[tuple[str, str], EffectSizeRecord],
                       dropped: set[tuple[str, str]],
                       excluded: list[dict[str, Any]]) -> bool:
    paper_id = override.get("paper_id", "")
    mine = [key for key, record in records.items()
            if record.paper_id == paper_id or sha12(record.paper_id) == sha12(paper_id)]
    if override.get("eligible"):
        return bool(mine)                                  # already in; nothing to do
    if not mine:
        # keeping a paper OUT needs nothing bought and nothing dropped — the run already has no
        # row for it. Reporting that as "pending a model call" (the message for an INCLUSION) told
        # a reviewer their decision had not happened when it was the state of the world.
        return True
    for key in mine:
        dropped.add(key)
        excluded.append({"paper_id": paper_id, "filename": "", "dataset_id": key[0],
                         "outcome_key": key[1], "stage": "review", "reason": "human_override",
                         "quote": "", "decider": override.get("actor", "human"),
                         "detail": override["justification"]})
    return bool(mine)


def _measure_name(state: "_RunState", dataset_id: str, outcome_key: str) -> str:
    """What the mapper called the thing this cell measures, as written in the study map."""
    dataset = state.datasets.get(dataset_id)
    for outcome in getattr(dataset, "outcomes", None) or []:
        if outcome.outcome_key == outcome_key:
            return str(getattr(outcome, "measure_name", "") or "")
    return ""


def _same_measure(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def _bucket_after_orientation(verdict: Verdict, row_flags: Collection[str] = (),
                              row_route: str = "", overruled: Collection[str] = ()) -> str:
    """The bucket a cell holds once its direction — and *only* its direction — is settled.

    The run's own scorer is deliberately not re-run here. Its inputs (the vote's routes, the
    adjudicator's ruling) are not in the stage files a re-pool reads, so reconstructing them would
    mean guessing at the very arithmetic this answer is supposed to leave alone. What an override
    may honestly ask is the narrow question: *was direction the only thing holding this cell?*

    So this returns `accept_with_note` only when every blocker it can see is absent — the cell has
    a value, the readers did not disagree, no verifier refuted it, no consistency error or
    "this may be a different quantity" flag stands, no adjudicator had to settle it, and the score
    the run itself recorded already clears the acceptance threshold on its own. Anything else stays
    `needs_human`, with the other reason still on the record.

    The rule is one-sided on purpose: it can only keep holding a cell that might have cleared, and
    can never release one that should not. A cell that was already accepted keeps its bucket but is
    capped at `accept_with_note`, because a sign a human supplied belongs in the review table.
    """
    from ..verify.confidence import ACCEPT_WITH_NOTE, CONTRADICTING_FLAGS

    if set(row_flags) & ROW_REFUSALS:                      # a refusal no cell answer can name
        return "needs_human"
    if verdict.confidence != "needs_human":                # a healthy cell of the same measure
        return "accept_with_note"                          # capped, never promoted
    if verdict.mean is None and not (converted_route(row_route)
                                     and "no_group_values" in set(overruled)):
        return "needs_human"                               # nothing was resolved to sign
    if verdict.agreement == "disagree":
        return "needs_human"
    if verdict.verifier_verdict == "refuted":
        return "needs_human"
    if verdict.adjudicated:
        # an adjudicated cell is one a model had to settle, and whether that model asked for a
        # human is not recorded separately in the stage files. Direction is not the adjudication,
        # so answering it does not release the cell.
        return "needs_human"
    if any(flag.severity == "error" for flag in verdict.flags):
        return "needs_human"
    if {flag.code for flag in verdict.flags} & CONTRADICTING_FLAGS:
        return "needs_human"
    if (verdict.confidence_score or 0.0) < ACCEPT_WITH_NOTE:
        return "needs_human"
    return "accept_with_note"


def _apply_orientation(override: Mapping[str, Any],
                       records: dict[tuple[str, str], EffectSizeRecord],
                       verdicts: dict[tuple[str, str, str], Verdict], state: "_RunState",
                       protocol: Protocol,
                       touched_rows: set[tuple[str, str]] | None = None,
                       overruled: Mapping[tuple[str, str, str], set[str]] | None = None
                       ) -> tuple[bool, str]:
    """Settle the direction of one measure, and re-derive every cell that used it.

    Scope is (paper, outcome_key, measure_name) because that is where the pipeline decides it: an
    empty `measure_name` means every dataset of that paper measuring that outcome. The answer sets
    `higher_is_better`, retires the codes it answers (`ORIENTATION_ANSWERED`), records the
    reviewer's quote as the cell's orientation evidence, and re-derives the row through the
    ordinary resolution path — so the sign appears in the effect size and nowhere else.
    """
    paper_id = str(override.get("paper_id") or "")
    outcome_key = str(override.get("outcome_key") or "")
    measure = str(override.get("measure_name") or "")
    higher_is_better = bool(override.get("higher_is_better"))
    evidence = str(override.get("quote") or "").strip() or override["justification"]

    targets = [key for key, record in records.items()
               if key[1] == outcome_key
               and (record.paper_id == paper_id or sha12(record.paper_id) == sha12(paper_id)
                    or state.paper_of.get(key[0], "") == paper_id)
               and (not measure or _same_measure(_measure_name(state, key[0], outcome_key),
                                                 measure))]
    if not targets:
        return False, (f"no row for {outcome_key!r}"
                       + (f" measuring {measure!r}" if measure else "")
                       + f" in paper {paper_id!r} in this run")
    # belt and braces behind the validator: whatever the scope resolved to, it must be ONE measure.
    # A direction that lands on two different measures is not a direction, it is a guess applied
    # twice — and one of the two is a quantity nobody asked the reviewer about.
    named = sorted({_measure_name(state, key[0], outcome_key) for key in targets})
    if len(named) > 1:
        return False, (f"this direction would land on {len(named)} different measures "
                       f"({', '.join(repr(n) for n in named)}); answer it once per measure")

    touched = 0
    for key in targets:
        dataset_id = key[0]
        dataset = state.datasets.get(dataset_id)
        verdict_a = verdicts.get((dataset_id, outcome_key, "A"))
        verdict_b = verdicts.get((dataset_id, outcome_key, "B"))
        if dataset is None or verdict_a is None or verdict_b is None:
            continue
        for verdict in (verdict_a, verdict_b):
            verdict.higher_is_better = higher_is_better
            verdict.flags = [f for f in verdict.flags if f.code not in ORIENTATION_ANSWERED]
            # C2: a person decided this one. "older adults adapted less" read off a majority of
            # three machines is a different claim from one somebody made, and the extraction table
            # cannot tell them apart from the sign alone.
            verdict.orientation_source = "human"
            verdict.orientation_evidence = evidence
            verdict.overridden_by_human = True
            verdict.override_justification = override["justification"]

        record = records[key]
        probe = _rebuild_row(record, dataset, verdict_a, verdict_b, protocol, outcome_key,
                             override["justification"], higher_is_better=higher_is_better,
                             state=state, verdicts=verdicts)
        for cell in (verdict_a, verdict_b):
            names = (overruled or {}).get((dataset_id, outcome_key, cell.group), set())
            bucket = _bucket_after_orientation(cell, row_flags=probe.flags,
                                               row_route=str(probe.route or ""), overruled=names)
            cell.confidence = bucket
            cell.needs_human = bucket == "needs_human"
        records[key] = _rebuild_row(record, dataset, verdict_a, verdict_b, protocol, outcome_key,
                                    override["justification"],
                                    higher_is_better=higher_is_better,
                                    state=state, verdicts=verdicts)
        if touched_rows is not None:
            touched_rows.add(key)
        touched += 1
    if not touched:
        return False, (f"no verified cell of {outcome_key!r} in paper {paper_id!r} to give a "
                       f"direction to")
    return True, ""


def _apply_analysed_n(override: Mapping[str, Any],
                      records: dict[tuple[str, str], EffectSizeRecord],
                      verdicts: dict[tuple[str, str, str], Verdict], state: "_RunState",
                      protocol: Protocol,
                      touched_rows: set[tuple[str, str]] | None = None) -> tuple[bool, str]:
    """D4-lite: record the analysed group sizes and rebuild EVERY row of the dataset with them.

    Every row, because how many people were in each arm is not a property of an outcome: the same
    two groups produced the late-adaptation number and the aftereffect number, and a denominator
    corrected for one of them and not the other says the study had two different sample sizes.
    That is also why the sizes go on `_RunState` before the rebuild — a later answer about any
    cell of this dataset rebuilds its row through `_prepare`, and must build it with the sizes a
    human has already supplied rather than with the recruited ones the paper printed.

    Nothing here computes the sizes. `checks.n_before_exclusions` says the row's n may be a
    recruited count and the card offers `recruited − excluded`; which number is the analysed one
    is the reviewer's answer, on the record with their quote.
    """
    dataset_id = str(override.get("dataset_id") or "")
    dataset = state.datasets.get(dataset_id)
    if dataset is None:
        return False, f"no dataset {dataset_id!r} in this run"
    state.group_n[dataset_id] = (int(override["n_a"]), int(override["n_b"]))
    state.group_n_seq[dataset_id] = int(override.get("seq") or 0)

    sizes = {"A": int(override["n_a"]), "B": int(override["n_b"])}
    touched = 0
    for key in [k for k in records if k[0] == dataset_id]:
        outcome_key = key[1]
        verdict_a = verdicts.get((dataset_id, outcome_key, "A"))
        verdict_b = verdicts.get((dataset_id, outcome_key, "B"))
        if verdict_a is None or verdict_b is None:
            continue
        # on the CELLS as well as on the row: the extraction table prints each group's n from its
        # verdict, and a reviewer who has just supplied the analysed sizes must not be shown the
        # recruited ones beside a row that no longer uses them. It settles nothing else — the
        # bucket is untouched, because a size is not an answer to whatever is holding the cell.
        for verdict in (verdict_a, verdict_b):
            verdict.n = sizes[str(verdict.group)]
            # …and NOT `overridden_by_human`: the only reader of that field is the review page,
            # which shows it as "a human overrode this cell", and nobody overrode this cell's
            # value, its spread or its direction — a group size was supplied. The justification
            # still travels, so the record says who supplied it and why (review finding 4).
            verdict.override_justification = override["justification"]
        records[key] = _rebuild_row(records[key], dataset, verdict_a, verdict_b, protocol,
                                    outcome_key, override["justification"],
                                    state=state, verdicts=verdicts)
        if touched_rows is not None:
            touched_rows.add(key)
        touched += 1
    if not touched:
        return False, (f"no verified row of dataset {dataset_id!r} in this run to give analysed "
                       f"group sizes to")
    return True, ""


#: §C4, mandatory: "an override clears ONLY the named blocker". These are the findings a value
#: answer names, and nothing else is ever retired by one.
#:
#: A **mean** answers "which number, off which ladder": the readers' axis dispute, the ladder that
#: could not be corroborated, and a value the ladder cannot draw. It does NOT answer where the
#: number came from — an ungrounded quote, a row-only match, a sign the paper contradicts are
#: findings about provenance, and only the question that names them can retire them (its option
#: carries `clears`). A **spread type** answers what the error bars are, and only that. An `n` or
#: a spread VALUE answers that the number was missing.
#: the check codes a recorded DIRECTION answers: the abstention's own, and the readers' quarrel
#: about which way the measure points — all of them ask for a direction, so a direction retires all
#: of them. ONE rule, here beside the applier that strips them and read by `canopy.review.questions`
#: to decide what a cell still asks, for the same reason `codes_cleared_by_value` lives here: the
#: page and the analysis may not disagree about what an answer has settled. Retiring only
#: `orientation_unknown` left `orientation_direction_conflict` standing on a cell whose direction
#: was on the record, and the page asked the answered question for ever — 879 identical answers to
#: one measure card in one run.
#:
#: `orientation_reader_contradicts_values` is deliberately absent: that is C3's contradiction
#: between a stated direction and this cell's own resolved means, and no direction answers it.
ORIENTATION_ANSWERED: frozenset[str] = frozenset({
    "orientation_unknown", "orientation_unresolved", "orientation_direction_conflict"})

#: what a group's OWN statistics are made of — the three fields `_apply_value` calls a cell
#: `complete` on, and the least a typed answer must carry before a row can be built from the two
#: cells rather than from a printed statistic. Named here, beside the clearing rules, because
#: `canopy.review.questions` reads it to decide whether a typed answer is the whole of what "where
#: is this value?" asked for: a mean alone builds no row, and a question ticked off by one leaves a
#: held row with nothing open anywhere.
GROUP_STATISTICS: tuple[str, ...] = ("mean", "dispersion_value", "n")

#: on a ROW a reviewer's own answer rebuilt. It is the resolver's only way to tell "this paper did
#: not report enough" from "a person answered this row and it still converts to nothing" — the
#: first is an exclusion, the second is a question, and `run.cells_for_review` keeps the cells of
#: the second in the queue on the strength of this flag.
HUMAN_OVERRIDE = "human_override"

VALUE_CLEARS_MEAN: frozenset[str] = frozenset({
    "axis_conflict", "calibration_disputed", "calibration_refuted", "calibration_single_witness",
    "calibration_two_point",
    "calibration_missing", "value_outside_axis"})
VALUE_CLEARS_SPREAD_TYPE: frozenset[str] = frozenset({
    "dispersion_type_from_legend", "figure_error_bar_unknown", "dispersion_type_conflict"})
#: the spread types a typed answer can build a row from — the ONLY ones that retire the error-bar
#: question above. A `value` override writes a mean, a spread value, an `n` and a type and nothing
#: else, and `resolve._has_spread` accepts exactly these five from that much: RANGE wants a minimum
#: and a maximum, and NONE and UNKNOWN are not spreads at all. Naming a type the row still cannot
#: divide by is not an answer to "what do these error bars show?", and treating it as one is what
#: took the last question off a cell whose row then converted to nothing.
#: `tests/test_questions.py` pins this set against `_has_spread` itself so the two cannot drift.
SPREAD_TYPES_A_VALUE_CONVERTS: frozenset[str] = frozenset({"SD", "SE", "CI95", "CI90", "IQR"})
VALUE_CLEARS_N: frozenset[str] = frozenset({"n_missing", "n_not_integer", "n_too_small",
                                            "n_mismatch"})
VALUE_CLEARS_DISPERSION: frozenset[str] = frozenset({
    "dispersion_missing", "sd_nonpositive", "sd_near_zero", "se_sd_inconsistent"})
#: What a printed statistic's degrees of freedom are evidence ABOUT is the estimand — whether the
#: test is the contrast between these two groups at all (§C9). A group size is not evidence about
#: that, and neither is a spread: the only thing that answers it is not needing the statistic any
#: more, which means a complete group-value pair for BOTH groups. So these are retired by that and
#: by nothing else — see `_apply_value`, which is the only place that can see both groups.
VALUE_CLEARS_DF: frozenset[str] = frozenset({"df_missing", "df_shortfall_unexplained",
                                             "test_stat_missing_df"})
#: the same family, written as a prefix by `checks.CHECK_SEVERITY_PREFIXES` (`df_off_by_3`, …). A
#: prefix cannot live in a frozenset, so it is expanded against the codes the cell actually carries
#: — and it follows the same rule as its siblings: only a complete group-value pair retires it.
VALUE_CLEARS_DF_PREFIXES: tuple[str, ...] = ("df_off_by_",)
#: `implausible_dispersion` is deliberately in NO cleared family. It says "the denominator this
#: effect was divided by cannot be the one the authors used", and supplying a denominator is not
#: an argument that the new one is plausible — the screen has to be RE-DERIVED from whatever
#: numbers the cell now holds — which is what the rebuild at the end of every apply does, through
#: `resolve._finish`'s own screen rather than through arithmetic this module keeps a copy of.


def codes_cleared_by_value(override: Mapping[str, Any], *,
                           both_groups_complete: bool = False,
                           carried: Collection[str] = ()) -> frozenset[str]:
    """The check codes THIS value answer answers — the one place the rule is written.

    Read by `canopy.review.questions` too, so the page and the log cannot disagree about which
    blocker a recorded answer has retired and which one the cell should be asked about next.

    `both_groups_complete` is the df family's condition: both groups of this cell now carry a
    mean, a spread and an n of their own, so the row no longer converts from the printed statistic
    whose estimand was in doubt. Only `_apply_value` can see that, so the page passes `False` and
    asks the question again until the analysis says otherwise.

    Whatever this returns, nothing outside the codes the cell CARRIED when the answer was recorded
    is ever retired — `_cleared_on(record, ...)` intersects it with the record's own `carried`.
    """
    cleared = {str(code) for code in override.get("clears") or []}
    if override.get("mean") is not None:
        cleared |= VALUE_CLEARS_MEAN
    # …and only a type this answer can actually BUILD A ROW FROM retires the error-bar question.
    # The test used to be the field's truthiness, and every type string is truthy, so answering
    # "the bars are unlabelled" — or "they are a range" — retired `figure_error_bar_unknown`, the
    # very finding that asks. The cell then had nothing left to ask while `resolve._has_spread`
    # refused the spread, so the row converted to nothing and stood in neither analysis line with
    # no question anywhere: Langan's four rows left the forest that way. RANGE and NONE are here
    # for the same reason as UNKNOWN and not as an afterthought — a value override writes a mean,
    # a spread, an `n` and a type, and RANGE needs a minimum and a maximum it cannot write, so
    # answering it names a spread the row still cannot divide by. `.strip().upper()` because this
    # rule is public and the page calls it on raw answers, not only on validated records.
    if str(override.get("dispersion_type") or "").strip().upper() in SPREAD_TYPES_A_VALUE_CONVERTS:
        cleared |= VALUE_CLEARS_SPREAD_TYPE
    if override.get("n") is not None:
        cleared |= VALUE_CLEARS_N
    if override.get("dispersion_value") is not None:
        cleared |= VALUE_CLEARS_DISPERSION
    if both_groups_complete:
        cleared |= VALUE_CLEARS_DF
        cleared |= {str(code) for code in (carried or override.get("carried") or ())
                    if str(code).startswith(VALUE_CLEARS_DF_PREFIXES)}
    return frozenset(cleared)


def _cleared_on(override: Mapping[str, Any], codes: Iterable[str]) -> set[str]:
    """`codes`, narrowed to what this record says its cell carried when it was written.

    The record carries its own `carried` list (written by `append_override`), so a `--resume` that
    rewrites the verify stage file cannot retroactively un-apply an answer, and an answer can
    never retire a finding that was not on the cell in front of the reviewer.
    """
    carried = {str(code) for code in override.get("carried") or []}
    if not override.get("carried"):
        return set(codes)            # a record written before this field: the old behaviour
    return {code for code in codes if code in carried}


#: the findings that hold a cell and are NOT flag codes, so no `clears` can name them. A
#: reviewer may overrule each — that is what `confirm_value` is — but only by saying which.
#:
#: `no_group_values` is the fourth, and it is a decision about the ROW taken on a cell: this
#: paper prints no mean, spread or n for either group, the row's effect size came from a printed
#: statistic or a printed d instead, and the reviewer has read that statistic and accepts it. It
#: is separate from the other three because it is the only one that lets a cell WITHOUT A NUMBER
#: be released, and nothing else in the record could say a person had looked. The C5/C9 gates are
#: untouched by it: they are re-derived by the resolver on every rebuild, so a converted row whose
#: contrast is unestablished stays held however many cells are released (whole-diff H2).
OVERRULABLE: tuple[str, ...] = ("verifier_refuted", "adjudicated", "low_score",
                                "no_group_values")

#: findings `resolve._finish` puts on the ROW rather than on either cell, because they are about
#: the number the conversion produced. No `clears` can name one and no cell answer retires one:
#: each rebuild re-derives it from the values the row now holds, and the cells are held while it
#: stands. The set itself is the RESOLVER's, imported rather than restated: a second copy here
#: is one the resolver can grow past without anyone noticing, and the first code that reached the
#: row and not this list would release cells under a refused row (integration fix round,
#: consistency item 1). `resolve._add_row_refusal` refuses any code that is not in it.
ROW_REFUSALS: frozenset[str] = ROW_REFUSAL_CODES


def _derived_bucket(verdict: Verdict, *, mean_answered: bool = False,
                    overrules: Iterable[str] = (), other: Verdict | None = None,
                    row_flags: Collection[str] = (), row_route: str = "") -> str:
    """What the record still says about a cell once the codes an answer named are gone.

    Derived from the verdict as it now stands — never assigned from the answer. Every forcing
    finding a score cannot settle keeps holding: a refutation, an adjudication, an unresolved
    direction, an unanswered disagreement about the number, any surviving `error`, any "this may
    be a different quantity" contradiction, and a score that never cleared the acceptance line.

    `overrules` is the reviewer saying, in the record, which of the three findings that are not
    flag codes they have looked at and disagree with (`OVERRULABLE`). Nothing is implicit: an
    answer that names none of them cannot release a refuted or adjudicated cell, which is what a
    plain `mark_reviewed` used to do by assigning its own bucket.
    """
    from ..verify.confidence import ACCEPT_WITH_NOTE, CONTRADICTING_FLAGS

    overruled = {str(name) for name in overrules}
    if set(row_flags) & ROW_REFUSALS:
        # FIRST, and before the healthy-cell shortcut: a finding on the ROW is one no cell answer
        # can name and no cell can look healthy past. The denominator the conversion actually
        # divided by cannot be the one the authors used — which is exactly the case where both
        # cells read as fine. It is re-derived by the resolver on every rebuild, so the reviewer's
        # own numbers are what it judges, and while it stands the cells stay in the queue: a row
        # held with no question anywhere is the failure C4 exists to remove.
        return "needs_human"
    if verdict.confidence != "needs_human":                # a healthy cell: capped, not promoted
        return "accept_with_note"
    if verdict.mean is None and not (converted_route(row_route) and "no_group_values" in overruled):
        # …unless the ROW does not need one. A row converted from a printed t, F, p or d has an
        # effect size that no cell value would improve, and the paper prints no group values to
        # resolve — so "no value was resolved for this cell" is a true statement about the cell
        # and not a question anybody can answer. It is released only by a reviewer saying, on the
        # record, that they have read the statistic and accept the converted effect; the gates
        # that judge the CONVERSION (C5's contrast, C9's degrees of freedom and resolved-|d|
        # screen) are re-derived by the resolver afterwards and are not overruled by this.
        return "needs_human"
    if verdict.higher_is_better is None:
        # `confidence.py` forces a human here and so must this: an effect size built on a measure
        # whose direction nobody has established has an undecided SIGN, and no number answers that.
        return "needs_human"
    if verdict.verifier_verdict == "refuted" and "verifier_refuted" not in overruled:
        return "needs_human"
    if verdict.adjudicated and "adjudicated" not in overruled:
        return "needs_human"
    if verdict.agreement == "disagree" and not mean_answered:
        # a mean IS the answer to "the readers disagree"; anything else leaves that dispute open
        return "needs_human"
    if any(flag.severity == "error" for flag in verdict.flags):
        return "needs_human"
    if {flag.code for flag in verdict.flags} & CONTRADICTING_FLAGS:
        return "needs_human"
    if (verdict.confidence_score or 0.0) < ACCEPT_WITH_NOTE \
            and "low_score" not in overruled:
        return "needs_human"
    return "accept_with_note"


def _apply_group_n(row: PreparedRow, state: "_RunState" | None, siblings: int = 1) -> None:
    """D4-lite: put the ANALYSED group sizes a reviewer answered on a prepared row.

    Here, rather than in the loop that reads the answer, because `_prepare` is the one funnel
    every rebuild goes through — a value answer, a direction, a cell marked reviewed — and the analysed
    n has to be the n each of those rows is built with too. It is applied to the row's values
    (`prepare_row_values`) and to nothing else: the verdicts keep the n the extractors read, which
    is what they saw, and the record keeps the size the row was actually divided by.

    AFTER the shared-control adjustment, so the arm a reviewer answered for is re-split the way
    the run split it (Cochrane 16.5.4). Overwriting a split arm with the whole answered size would
    hand a control shared between two comparisons its full n back, which shrinks the row's
    variance and raises its weight — the same un-splitting `pipeline.rows` exists to prevent.

    Which arm was shared is `resolve.SHARED_CONTROL_ARM` — the same name `rows.prepare_rows` hands
    to `apply_shared_control` — and not a literal "B" written here, because an assumption about
    another module's default that is true today is a wrong number tomorrow. And under
    `combine_arms` the OTHER arm of the surviving row is two arms added together: a per-arm size a
    reviewer answered is not that number, so that arm is left exactly as the strategy built it
    (review finding 3).
    """
    sizes = (state.group_n if state is not None else {}).get(row.dataset.dataset_id)
    if sizes is None:
        return
    answered = {"A": sizes[0], "B": sizes[1]}
    # …except an arm a reviewer answered LATER, cell by cell. The dataset-level size is the
    # reviewer's word about both arms of every outcome, and a per-cell `n` answered after it is
    # their word about this one: re-applying the older number left the review table showing the
    # newer size beside a row divided by the older (review MINOR 30).
    when = (state.group_n_seq if state is not None else {}).get(row.dataset.dataset_id, 0)
    for key in ("A", "B"):
        if (state.n_answered if state is not None else {}).get(
                (row.dataset.dataset_id, row.outcome_key, key), -1) > when:
            answered.pop(key, None)
    flags = set(row.values.flags)
    shared = SHARED_CONTROL_ARM
    merged = "A" if shared == "B" else "B"
    for values in (row.values, *row.alternatives):
        for key, size in answered.items():
            group = values.group(key)
            if group is None:
                continue
            if key == merged and "shared_control_combined" in flags:
                continue                    # two arms added together is not one arm's answered n
            if key == shared and "shared_control_split" in flags and siblings > 1:
                group.n = int(round(split_control(size, siblings)))
            else:
                group.n = size


def _prepare(dataset: DatasetSpec, outcome_key: str, verdict_a: Verdict, verdict_b: Verdict,
             protocol: Protocol, *, higher_is_better: bool | None = None,
             state: "_RunState" | None = None,
             verdicts: Mapping[tuple[str, str, str], Verdict] | None = None) -> PreparedRow:
    """This row's inputs, prepared by the RUN's own function over the cluster it belongs to.

    The cluster matters for exactly one reason and it is not cosmetic: `apply_shared_control`
    divides a control arm's n by how many comparisons use it, so a row prepared on its own is a
    row whose control was never split. The membership test here is the run's — same cluster, same
    outcome, `shared_control` set, both cells verified — so an untouched row rebuilds to the
    record the run wrote, byte for byte.
    """
    live = dict(verdicts or {})
    live[(dataset.dataset_id, outcome_key, "A")] = verdict_a
    live[(dataset.dataset_id, outcome_key, "B")] = verdict_b
    known = list((state.datasets if state is not None else {}).values())
    # the row being rebuilt is always in the list, whatever the state holds: a cluster assembled
    # without it would prepare its siblings and not it.
    datasets = ([dataset] if all(d.dataset_id != dataset.dataset_id for d in known) else []) + known

    def cluster_of(spec: DatasetSpec) -> str:
        paper = (state.paper_of.get(spec.dataset_id, "") if state is not None else "")
        return spec.cluster_id or paper or spec.dataset_id

    def has_cell(dataset_id: str, key: str) -> bool:
        return all((dataset_id, key, group) in live for group in ("A", "B"))

    siblings = shared_control_siblings(dataset, outcome_key, datasets, cluster_of, has_cell)
    cells = [(spec, key, live[(spec.dataset_id, key, "A")], live[(spec.dataset_id, key, "B")])
             for spec, key in siblings]
    candidates = list(state.candidates) if state is not None else []
    directions = ({(dataset.dataset_id, outcome_key): higher_is_better}
                  if higher_is_better is not None else None)
    # Rule A's premise, from THIS PAPER's map — `state.datasets` is run-wide, and a house style
    # computed across papers would let one paper's captions type another paper's bars. Scoped via
    # the same `paper_of`/`studies` the rest of the re-pool uses; absent a study (never, in a run
    # this tool wrote), no inference. BOTH calls below carry it: the sibling-cluster build and the
    # lone-row fallback must agree or an answered row would rebuild differently from an untouched
    # one (byte-identity with the run, which computes the same premise from the same map).
    _study = (state.studies.get(state.paper_of.get(dataset.dataset_id, ""))
              if state is not None else None)
    house = house_spread_type(_study.datasets) if _study is not None else None
    prepared = prepare_rows(cells, candidates, protocol.stats, cluster_of=cluster_of,
                            directions=directions, house_spread=house)
    mine = [row for row in prepared if row.key == (dataset.dataset_id, outcome_key)]
    row = mine[0] if mine else prepare_rows(
        [(dataset, outcome_key, verdict_a, verdict_b)], candidates, protocol.stats,
        cluster_of=cluster_of, directions=directions, house_spread=house)[0]
    _apply_group_n(row, state, len(cells))
    if state is not None and (dataset.dataset_id, outcome_key) in state.keep_printed:
        # D1's fallback is offered alternatives or it is not; there is no third state. A reviewer
        # who kept the printed value declined the candidate pair, and `resolve_effect_with_fallback`
        # has no other way to be told (review finding 12).
        row.alternatives = []
    return row


def _rebuild_row(record: EffectSizeRecord, dataset: DatasetSpec, verdict_a: Verdict,
                 verdict_b: Verdict, protocol: Protocol, outcome_key: str, justification: str,
                 *, higher_is_better: bool | None = None, state: "_RunState" | None = None,
                 verdicts: Mapping[tuple[str, str, str], Verdict] | None = None
                 ) -> EffectSizeRecord:
    """This row, re-derived from the cells as they now stand, through the ordinary resolver.

    EVERY release path goes through here — a value, a direction, a cell marked reviewed — because
    the rules that refuse a ROW live in `resolve._finish` and nowhere a cell can see: C9's screen on
    the resolved |d| is computed from the number the conversion produced, post-vote, post-conversion.
    A path that skipped the rebuild released the row on the cells' word alone; that is how marking
    two cells reviewed pooled `d = −28.8`.

    And it is re-derived through `pipeline.rows.prepare_rows`, which is the run's OWN preparation
    — not `from_verdicts` alone. The whole-diff review measured what the shortcut cost: the
    printed statistic and printed d the row converts from were dropped (so a converted row became
    `not_convertible` the moment a human answered anything about it), the row's policy flags were
    dropped, and a control shared between two comparisons was silently un-split, which shrinks the
    row's variance by a quarter and raises its weight in the pooled estimate. Every one of those
    is invisible from either cell, so no amount of care inside this function could have caught it;
    the only fix is that both callers build the row the same way (whole-diff H1/H2).

    `state`/`verdicts` are how the siblings are found. Without them the row is prepared alone,
    which is right for a row that shares no control and is the old behaviour for one that does —
    callers inside `apply_overrides_and_repool` always pass both.
    """
    prepared = _prepare(dataset, outcome_key, verdict_a, verdict_b, protocol,
                        higher_is_better=higher_is_better, state=state, verdicts=verdicts)
    values = prepared.values
    values.flags = sorted({*values.flags, HUMAN_OVERRIDE})
    # …and through `resolve_effect_with_fallback`, which is what the run calls (D1). A row whose
    # printed values convert to nothing is built from the same-locator candidate pair `_prepare`
    # found; were this `resolve_effect` alone, answering anything about such a cell would take the
    # row's effect size away again — the one-row-path rule, on the newest branch of the resolver.
    rebuilt = resolve_effect_with_fallback(dataset, protocol.outcome(outcome_key), values,
                                           prepared.alternatives, protocol.stats)
    rebuilt.paper_id = record.paper_id
    rebuilt.sample_id = record.sample_id
    rebuilt.cluster_id = record.cluster_id
    rebuilt.citation = record.citation
    rebuilt.label = record.label
    rebuilt.moderators = record.moderators
    rebuilt.analysis_metric = record.analysis_metric
    # C2, through `_prepare` like everything else here: the row says how its direction was settled,
    # and a rebuild that dropped it would leave a human's decision reading as two agreeing models.
    rebuilt.orientation_source = prepared.orientation_source or record.orientation_source
    rebuilt.flags = sorted({*rebuilt.flags, HUMAN_OVERRIDE})
    rebuilt.notes = "; ".join(x for x in (record.notes,
                                          f"human override: {justification}") if x)
    return rebuilt


def _bucket_after_value(verdict: Verdict, override: Mapping[str, Any],
                        other: Verdict | None = None,
                        overruled: Collection[str] = (), mean_settled: bool = False,
                        row_flags: Collection[str] = (), row_route: str = "") -> str:
    """The bucket a cell holds once a reviewer's numbers are on it — DERIVED, never stamped.

    The sibling of `_bucket_after_orientation`, and it exists for the same clause of §C4: an
    override clears only the blocker it names and the cell is re-scored, not re-stamped. Before
    this, `_apply_value` set `needs_human → accept_with_note` unconditionally, so answering "which
    axis is this read off?" on Heuer d1 late adaptation released a cell that also carried
    `quote_not_grounded` (its quote is not printed in the paper) and an adjudication, and pooled
    it. Nobody had been asked about the quote.

    One-sided in the same direction: it can only keep holding a cell that might have cleared.
    """
    return _derived_bucket(verdict,
                           mean_answered=override.get("mean") is not None or mean_settled,
                           overrules={*(override.get("overrules") or []), *overruled},
                           other=other, row_flags=row_flags, row_route=row_route)


#: what a cell built from the log alone says about itself. Not a reading and never mistaken for
#: one: no candidate id, no verifier, no score — `needs_human` until a person confirms it, which
#: is what every other cell a human typed into does too.
FROM_THE_LOG = ("this cell was never read by the run: the map asks for it and the extract stage "
                "never reached it. It exists because a reviewer typed its numbers.")


def _cell_the_run_never_read(dataset: DatasetSpec, outcome_key: str, state: "_RunState",
                             records: dict[tuple[str, str], EffectSizeRecord],
                             verdicts: dict[tuple[str, str, str], Verdict],
                             protocol: Protocol) -> str:
    """Make the two verdicts and the row a typed value needs. `""` on success, else why not.

    Idempotent and derived from the map, so every re-pool of the same log builds the same cell:
    the failure this exists to remove is a row that appeared when its pair was typed and was gone
    after the next round of unrelated answers, because nothing recreated what the values had made.

    The direction is not invented. It is taken from a sibling cell of the SAME paper and outcome
    that the run did orient — the orientation is decided once per (paper, outcome, measure), so a
    sibling's answer is this cell's answer — and left `None` when there is none, which holds the
    row for the direction question exactly as an unoriented cell always is.
    """
    dataset_id = dataset.dataset_id
    blocked = _out_of_reach(state, dataset_id, outcome_key, protocol)
    if blocked:
        # the SAME sentence a hint on this cell gets, for the same reason: the map put it out of
        # reach, and a value answer is not a decision about the map (MAJOR 1).
        return blocked
    paper_id = state.paper_of.get(dataset_id, "")
    # the direction is decided once per (paper, outcome, MEASURE), so the sibling must be
    # measuring the same thing. Two datasets of one paper can carry different measures under one
    # outcome — an error measure and a magnitude measure have opposite directions — and
    # `_apply_orientation` refuses a direction that would land on more than one for exactly this
    # reason. Borrowing across them would sign a row nobody read backwards, in silence.
    mine = _measure_name(state, dataset_id, outcome_key)
    oriented = next((v for v in verdicts.values()
                     if v.outcome_key == outcome_key and v.higher_is_better is not None
                     and state.paper_of.get(v.dataset_id, "") == paper_id
                     and _same_measure(_measure_name(state, v.dataset_id, outcome_key), mine)),
                    None)
    for group, size in (("A", dataset.group_a.n), ("B", dataset.group_b.n)):
        if (dataset_id, outcome_key, group) in verdicts:
            continue
        verdicts[(dataset_id, outcome_key, group)] = Verdict(
            dataset_id=dataset_id, outcome_key=outcome_key, group=group,
            n=size, confidence="needs_human", needs_human=True, route="human",
            higher_is_better=(oriented.higher_is_better if oriented is not None else None),
            orientation_source=(oriented.orientation_source if oriented is not None else ""),
            verifier_reason=FROM_THE_LOG)
    if (dataset_id, outcome_key) not in records:
        from .run import sample_key                        # `run` imports this module, not vice

        study = state.studies.get(paper_id)
        records[(dataset_id, outcome_key)] = EffectSizeRecord(
            paper_id=paper_id, dataset_id=dataset_id, outcome_key=outcome_key,
            cluster_id=dataset.cluster_id or paper_id,
            sample_id=sample_key(dataset, paper_id),
            citation=(study.citation if study is not None else None),
            label=dataset.label or dataset_id,
            confidence="needs_human", notes=FROM_THE_LOG)
    return ""


def _apply_value(override: Mapping[str, Any], records: dict[tuple[str, str], EffectSizeRecord],
                 verdicts: dict[tuple[str, str, str], Verdict], state: _RunState,
                 protocol: Protocol, overruled: Collection[str] = ()) -> tuple[bool, str]:
    """Replace one group's numbers and re-derive that row through the ordinary resolution path."""
    dataset_id, outcome_key = override["dataset_id"], override["outcome_key"]
    group = override["group"]
    key = (dataset_id, outcome_key)
    dataset = state.datasets.get(dataset_id)
    if dataset is None:
        return False, f"no dataset {dataset_id!r} in this run"
    # §C4: a mapped cell the run never read is still a cell, and a human's numbers are the whole
    # of one. Roller's E1a and E3 and Panouillères's aftereffect are mapped cells with no
    # candidate, no verdict and no row — the extract stage never reached them — and four complete
    # answers came back "no row for this in this run", pending for ever against a reading nobody
    # was going to buy. The cell is made HERE, from the map, on every re-pool: it is derived from
    # the log and the stage files, so it cannot be built once and lost by the next repool.
    missing = _cell_the_run_never_read(dataset, outcome_key, state, records, verdicts, protocol)
    if missing:
        return False, missing
    record = records.get(key)
    verdict = verdicts.get((dataset_id, outcome_key, group))
    if record is None or verdict is None:                  # unreachable; kept as a loud refusal
        return False, f"no row for {dataset_id}/{outcome_key} in this run"

    if override.get("mean") is not None:
        verdict.mean = float(override["mean"])
    # the spread this cell held BEFORE this record: a type is a statement about the number it was
    # stated for, so whether the standing type may survive is decided against the standing spread
    # and has to be read before that spread is replaced.
    held_spread = verdict.dispersion_value
    stated_spread = override.get("dispersion_value") is not None
    #: a record that RESTATES the standing spread says nothing new about that number; a record
    #: stating a different spread is a different number, and the standing label described the
    #: number it replaced (the same rule the UNKNOWN branch below applies)
    restated = (not stated_spread
                or (held_spread is not None
                    and float(override["dispersion_value"]) == float(held_spread)))
    if stated_spread:
        verdict.dispersion_value = float(override["dispersion_value"])
    if not override.get("dispersion_type") and not restated:
        # a new spread with no stated label has no label — carrying the old one across would
        # divide a number nobody typed a type for by the replaced number's semantics
        verdict.dispersion_type = DispersionType.UNKNOWN
    if override.get("dispersion_type"):
        try:
            named = DispersionType(override["dispersion_type"])
        except ValueError:
            return False, (f"unknown dispersion type {override['dispersion_type']!r} "
                           f"(expected one of {[d.value for d in DispersionType]})")
        # `UNKNOWN` is what a candidate carries when nobody could identify the error bars, and the
        # page copies the candidate's fields onto every value answer read off it — so a later
        # "this series is this group" arrives saying UNKNOWN about a spread somebody has already
        # typed the type of. When it RESTATES that same spread it has said nothing new about it and
        # the type stands; the row went `not_convertible` on a cell holding both typed numbers, and
        # its best-guess line read `one_group_only`.
        #
        # When it states a DIFFERENT spread it is a different number, and a label stated about the
        # number it replaced is not a label about this one. Then the cell goes back to unknown-type
        # — held, visible, in the exclusion table — which is where an untyped spread has always
        # left it. Carrying the label across would divide a new spread by an old one's semantics
        # and put a number nobody stated into the pooled effect size.
        if named is not DispersionType.UNKNOWN or not restated:
            verdict.dispersion_type = named
    if override.get("n") is not None:
        verdict.n = int(override["n"])
        # before the rebuild below, which reads it: this arm's size now comes from an answer of
        # its own, and a dataset-level analysed n answered EARLIER may not put its number back.
        state.n_answered[(dataset_id, outcome_key, group)] = int(override.get("seq") or 0)
    if override.get("unit"):
        verdict.unit = override["unit"]
    verdict.overridden_by_human = True
    verdict.override_justification = override["justification"]

    other = verdicts.get((dataset_id, outcome_key, "B" if group == "A" else "A"))
    if other is None:
        return False, f"the other group of {dataset_id}/{outcome_key} was never verified"
    # §C9: a printed statistic's degrees of freedom are answered by not needing the statistic —
    # which is true only once BOTH groups carry a mean, a spread and an n of their own.
    complete = all(cell.mean is not None and cell.dispersion_value is not None and cell.n
                   for cell in (verdict, other))
    cleared = _cleared_on(override, codes_cleared_by_value(
        override, both_groups_complete=complete,
        carried={flag.code for cell in (verdict, other) for flag in cell.flags}))
    verdict.flags = [flag for flag in verdict.flags if flag.code not in cleared]
    if complete:                       # the df findings belong to the row, not to one group
        other.flags = [flag for flag in other.flags if flag.code not in (cleared & VALUE_CLEARS_DF)]
    verdict_a, verdict_b = (verdict, other) if group == "A" else (other, verdict)
    # Twice, in this order, and the reason is that each rebuild reads the other's inputs. The first
    # is a PROBE: C9's screen on the resolved |d| only exists once the conversion has run, so a
    # bucket derived before a rebuild is derived without it. The second is the RECORD, and it is
    # the last thing every apply does: `ResolvedValues.from_verdicts` takes the weaker of the two
    # cells' buckets, so a row built before the answering cell's new bucket was assigned carries
    # the bucket the answer replaced — one answer behind, for ever, if nothing later mops it up.
    probe = _rebuild_row(record, dataset, verdict_a, verdict_b, protocol, outcome_key,
                         override["justification"], state=state, verdicts=verdicts)
    bucket = _bucket_after_value(verdict, override, other, overruled, row_flags=probe.flags,
                                 row_route=probe.route)
    verdict.confidence = bucket
    verdict.needs_human = bucket == "needs_human"
    records[key] = _rebuild_row(record, dataset, verdict_a, verdict_b, protocol, outcome_key,
                                override["justification"], state=state, verdicts=verdicts)
    return True, ""


def _stricter(*buckets: str) -> str:
    """The most conservative of these buckets, by the resolver's own order."""
    from .resolve import _BUCKET_ORDER

    return min(buckets, key=lambda bucket: _BUCKET_ORDER.get(bucket, 0))


def _reconcile(records: dict[tuple[str, str], EffectSizeRecord],
               verdicts: dict[tuple[str, str, str], Verdict],
               touched_rows: Collection[tuple[str, str]],
               dropped: Collection[tuple[str, str]]) -> None:
    """After the whole log: no row may be held while every one of its cells is released.

    A row and its cells are decided by different rules and answered by different records, so a
    sequence of answers can leave them disagreeing — which is not a cosmetic disagreement: a held
    row whose cells are all released is a row with no open question anywhere, and C4 exists to
    remove exactly that. The repair only ever goes one way: the CELLS follow the row back into the
    queue and keep asking. Since the rebuild is the last step of every apply, a row that is still
    held is one the RESOLVER held — a `ROW_REFUSALS` code, the conversion gate, no effect size at
    all — and this module cannot name the rule that did it, let alone answer it. Releasing the row
    on the cells' word would be the wrong direction, and a worse bug than the one this prevents.

    Only rows the log touched. A row nobody answered keeps exactly what the run decided about it.
    """
    for key in sorted(touched_rows):
        record = records.get(key)
        if record is None or key in dropped or record.confidence != "needs_human":
            continue
        cells = [verdicts[(key[0], key[1], group)] for group in ("A", "B")
                 if (key[0], key[1], group) in verdicts]
        if not cells or any(cell.needs_human for cell in cells):
            continue
        for cell in cells:
            cell.confidence = "needs_human"
            cell.needs_human = True


def _rewrite(out: Path, manifest: RunManifest, protocol: Protocol,
             records: Sequence[EffectSizeRecord], state: _RunState,
             verdicts: Mapping[tuple[str, str, str], Verdict],
             excluded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write every derived artefact again from the reviewed rows. No stage file is touched.

    Every writer here is the one `run.py` calls, with the same arguments, so a re-pooled run
    directory is the same *kind* of thing a finished run is: the per-outcome tables carry the rows
    an aggregation replaced (`all_rows`), the run-wide table and the PRISMA flow are rebuilt from
    the reviewed rows, and the report keeps its provenance table.
    """
    # the run's own primary/held/every split, and the run's own column list: a second list here
    # is a second place for a column to be forgotten, and `csv.DictWriter(extrasaction="ignore")`
    # forgets in silence — which is how C11's three columns survived a re-pool in the JSON and
    # vanished from the CSV a reviewer opens.
    from ..agents.mapper import HUMAN_DECIDER
    from .run import REVIEW_QUEUE_COLUMNS, _split_rows, cells_for_review

    live = list(verdicts.values())
    outcomes: dict[str, Any] = {}
    per_outcome: dict[str, dict[str, Any]] = {}
    review: list[dict[str, Any]] = []
    every_row: list[EffectSizeRecord] = []
    primary_rows: list[EffectSizeRecord] = []
    #: the same sink the run path uses: one best-guess decision per cell, shown by both tables
    best_guess_cells: dict[tuple[str, str], dict] = {}
    aggregated_out: list[dict[str, Any]] = []
    by_cell = {(r.dataset_id, r.outcome_key): r for r in records}
    # `cells_for_review` recognises a person's exclusion by its `decider`; a re-pool writes the
    # reviewer's own actor there, which is the more useful record and the reason the two spellings
    # differ. `reason == "human_override"` is this module's mark for the same fact, so the rows are
    # relabelled for the predicate and written unchanged.
    human_exclusions = [{**entry, "decider": HUMAN_DECIDER}
                        if entry.get("reason") == "human_override" else entry
                        for entry in excluded]

    for outcome in protocol.outcomes:
        mine = [r for r in records if r.outcome_key == outcome.key]
        split = _split_rows(mine, protocol.stats)
        primary, held = split.primary, split.held
        every_row.extend(split.every)
        primary_rows.extend(primary)
        aggregated_out.extend(split.exclusions)
        pooled = pool_rows(primary, protocol.stats)
        artefacts = write_outcome_outputs(out, outcome, primary, pooled, protocol.stats,
                                          needs_human_rows=held, verdicts=live,
                                          candidates=state.candidates,
                                          moderators=protocol.moderators or None,
                                          all_rows=split.every,
                                          primary_pre_agg=split.primary_pre_agg,
                                          best_guess_cells=best_guess_cells,
                                          protocol=protocol,
                                          warnings=manifest.warnings)
        _drop_stale_forest(out, outcome.key, artefacts, manifest)
        manifest.outputs.update({f"{outcome.key}.{k}": str(Path(v).relative_to(out))
                                 for k, v in artefacts.items()})
        per_outcome[outcome.key] = {"pooled": pooled, "outputs": artefacts, "rows": primary,
                                    "needs_human_rows": held}
        outcomes[outcome.key] = {
            "k": 0 if pooled is None else pooled.k, "n_needs_human": len(held),
            "estimate": None if pooled is None else pooled.estimate,
            "ci_low": None if pooled is None else pooled.ci_low,
            "ci_high": None if pooled is None else pooled.ci_high}
        # `run.py` owns the queue's rules and this calls them, rather than restating them: a
        # queue built one way by a fresh run and another way by a re-pool of it is a queue that
        # changes when a reviewer answers something unrelated. All three rules — the cell's own
        # bucket, a row held by a rule no cell carries, and BOTH cells of a row the resolver
        # refused — plus the subtraction of the cells a person has excluded, live there.
        for verdict in cells_for_review([v for v in live if v.outcome_key == outcome.key],
                                        held, human_exclusions):
            record = by_cell.get((verdict.dataset_id, verdict.outcome_key))
            review.append(review_entry(verdict, paper_id=record.paper_id if record else "",
                                       candidates=state.candidates, record=record,
                                       primary=primary, settings=protocol.stats))

    # the run-wide table: every row of every outcome, marked with whether it was pooled
    manifest.outputs.update({f"extraction_table_all.{k}": str(Path(v).relative_to(out))
                             for k, v in extraction_table(
                                 every_row, out / "results" / "extraction_table_all",
                                 verdicts=live, candidates=state.candidates,
                                 primary=primary_rows,
                                 best_guess=best_guess_cells).items()})

    manifest.human_review_queue = sort_review_queue(review)
    write_rows(manifest.human_review_queue, out / "human_review_queue",
               REVIEW_QUEUE_COLUMNS, formats=("csv", "json"))
    exclusions = _rewrite_exclusions(out, [*excluded, *aggregated_out])
    _rewrite_prisma(out, manifest, state, records, exclusions)

    provenance = _read_json(out / "provenance" / "provenance.json")
    run_outputs = {name: path for name, path in (
        ("methods_fig_png", out / "methods_routes.png"), ("prisma_png", out / "prisma.png"),
        ("provenance_json", out / "provenance" / "provenance.json")) if path.exists()}
    write_html_report(out, manifest, protocol, results=per_outcome,
                      review_queue=manifest.human_review_queue, exclusions=exclusions,
                      provenance=provenance if isinstance(provenance, dict) else None,
                      run_outputs=run_outputs)
    save_manifest(out, manifest)
    # the questions file is what a CLI reviewer reads next; a re-pool has just changed which
    # cells are held, so it is rewritten from the queue this re-pool built (never from a stale
    # manifest — `questions_for_run` is given the queue explicitly)
    try:
        from ..review.questions import questions_for_run, write_questions
        write_questions(out, questions_for_run(out, queue=manifest.human_review_queue))
    except Exception:                                    # pragma: no cover - never fail a re-pool on it
        pass
    return outcomes


def _rewrite_prisma(out: Path, manifest: RunManifest, state: "_RunState",
                    records: Sequence[EffectSizeRecord],
                    exclusions: Sequence[Mapping[str, Any]]) -> None:
    """Re-count the PRISMA flow after a review, with the run's own counter.

    Screening cannot change under a re-pool — no file appears or disappears — so the head of the
    chain (`files`, `duplicates_removed`, `unique_papers`, `not_processed`) is kept from the flow
    the run wrote, and everything downstream of eligibility is recounted from the reviewed rows.
    """
    from .run import PaperResult, _prisma_counts

    previous = _read_json(out / "prisma.json")
    results = [PaperResult(status=status, study=state.studies.get(status.paper_id))
               for status in manifest.papers]
    counts = _prisma_counts(results, [], 0, records, exclusions)
    for key in ("files", "duplicates_removed", "unique_papers", "not_processed"):
        if isinstance(previous, dict) and key in previous:
            counts[key] = previous[key]
    manifest.outputs.update({f"prisma.{k}": str(Path(v).relative_to(out))
                             for k, v in prisma_flow(counts, out / "prisma").items()})


def _drop_stale_forest(out: Path, outcome_key: str, artefacts: Mapping[str, Any],
                       manifest: RunManifest) -> None:
    """A plot of an analysis that no longer exists is worse than no plot.

    `write_outcome_outputs` writes no forest below k = 2, so an exclusion that takes an outcome
    under that leaves the previous run's plot on disk — still showing the row a reviewer just
    removed. Delete it, and stop the manifest promising it. The best-guess forest has the same
    rule and the same failure, so it is swept with the same broom.
    """
    for stem in ("forest", "forest_best_guess"):
        for suffix in ("png", "svg", "pdf"):
            if f"{stem}_{suffix}" in artefacts:
                continue
            (out / "results" / outcome_key / f"{stem}.{suffix}").unlink(missing_ok=True)
            manifest.outputs.pop(f"{outcome_key}.{stem}_{suffix}", None)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:                                     # pragma: no cover - defensive
        return None


def _all_exclusions(out: Path) -> list[dict[str, Any]]:
    rows = _read_json(out / "exclusions.json")
    return [r for r in (rows or []) if isinstance(r, dict)]


def _rewrite_exclusions(out: Path, extra: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the run's own exclusions and add the reviewer's, without duplicating a row.

    One decision, one row. A dataset a reviewer excluded is recorded twice by two writers that
    cannot see each other: the run's C7 loop reads the ANSWERED map and writes
    `map_adjudication:dataset_rule` with `decider = a human reviewer`, and `_apply_exclude` here
    writes `human_override` for the same answer. De-duping on the REASON kept both, so a resumed
    run listed the dataset twice in the exclusions table and counted it twice in the PRISMA
    reasons (whole-diff L5). What identifies the decision is the cell it removed and the person
    who removed it, so that is the key — for human deciders only: a dataset the CODE dropped for
    one reason and a person removed for another is two different facts about it.
    """
    rows = _all_exclusions(out)
    seen = {(r.get("dataset_id"), r.get("outcome_key"), r.get("reason")) for r in rows}
    by_human = {(r.get("dataset_id"), r.get("outcome_key")) for r in rows if _human_decision(r)}
    added = [dict(e) for e in extra
             if (e.get("dataset_id"), e.get("outcome_key"), e.get("reason")) not in seen
             and not (_human_decision(e) and (e.get("dataset_id"), e.get("outcome_key")) in by_human)]
    if added:
        rows = [*rows, *added]
        exclusions_table(rows, out / "exclusions")
    return rows


def _human_decision(row: Mapping[str, Any]) -> bool:
    """Was this exclusion made by a person? Both spellings, because both writers are honest.

    The run names the ROLE (`mapper.HUMAN_DECIDER`) because it is writing about an answer in the
    log; the re-pool names the ACTOR the reviewer signed with, which is the more useful record.
    `human_override` is this module's own mark for the same fact and is what makes the second
    spelling recognisable without a list of usernames.
    """
    from ..agents.mapper import HUMAN_DECIDER

    return (row.get("reason") == "human_override"
            or row.get("decider") == HUMAN_DECIDER)
