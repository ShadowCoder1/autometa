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
| `re_extract` | "read it again, with this hint" — needs a model, so `repool` reports it as pending |
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
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from ..models import (Candidate, DatasetSpec, DispersionType, EffectSizeRecord, Protocol,
                      RunManifest, StudyMap, Verdict)
from ..verify.confidence import ROW_REFUSAL_CODES
from ..protocol import load_protocol
from ..report import (dump_json, exclusions_table, extraction_table, pool_rows, prisma_flow,
                      write_html_report, write_outcome_outputs, write_rows)
from .resolve import ResolvedValues, resolve_effect
from .rows import (PreparedRow, converted_route, prepare_rows,
                   shared_control_siblings)
from .state import (load_manifest, read_stage, review_entry, save_manifest, sha12,
                    sort_review_queue, stage_done)

__all__ = ["KINDS", "MAP_KINDS", "MAP_PENDING", "OVERRIDES_FILE", "OverrideRejected",
           "OVERRULABLE", "RE_EXTRACT_PENDING", "codes_cleared_by_value", "consumed_seqs",
           "recorded_flags", "recorded_holds", "row_flags", "append_override",
           "read_overrides", "apply_overrides_and_repool", "map_answers", "override_summary",
           "repool_lock"]

OVERRIDES_FILE = "overrides.jsonl"
SUMMARY_FILE = "overrides_applied.json"
KINDS: tuple[str, ...] = ("value", "mark_reviewed", "exclude_dataset", "eligibility",
                          "re_extract", "orientation", "include_dataset", "which_measure")
#: the two kinds answered at the MAP stage: they decide what may be EXTRACTED, so — except for an
#: exclusion, which needs no reading — the pipeline applies them on the next `--resume`, not here.
MAP_KINDS: tuple[str, ...] = ("include_dataset", "which_measure")
MAP_PENDING = "extraction was never bought for this; re-run with --resume to extract it"
#: the other answer whose consequence is a model call: the questions page reads both, so a
#: reviewer is told that a recorded decision has not happened yet rather than left to assume it has
RE_EXTRACT_PENDING = ("a re-extraction needs a model call: run `canopy run --resume` with this "
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
    if kind in ("value", "mark_reviewed", "exclude_dataset", "re_extract") \
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
            "n": None if payload.get("n") in (None, "") else int(payload["n"]),
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
    if kind == "eligibility":
        if not record["paper_id"]:
            raise OverrideRejected("an eligibility override needs a paper_id")
        if not isinstance(payload.get("eligible"), bool):
            raise OverrideRejected("an eligibility override needs eligible: true or false")
        record["eligible"] = bool(payload["eligible"])
    if kind == "re_extract":
        record["hint"] = _text(payload.get("hint"), 1000)
        if not record["hint"]:
            raise OverrideRejected("a re-extraction request needs a hint saying what to read")
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
        record["measure_name"] = _text(payload.get("measure_name"), 300)
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
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OVERRIDES_FILE
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
    with _lock_for(path):
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


# ----------------------------------------------------------------------------- run state
class _RunState:
    """Everything the stage files hold, indexed the way re-pooling needs it."""

    def __init__(self, run_dir: Path, manifest: RunManifest):
        self.records: list[EffectSizeRecord] = []
        self.verdicts: list[Verdict] = []
        self.candidates: list[Candidate] = []
        self.datasets: dict[str, DatasetSpec] = {}
        self.studies: dict[str, StudyMap] = {}
        self.paper_of: dict[str, str] = {}
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
            (applied if override.get("seq") in consumed
             else pending).append(override if override.get("seq") in consumed
                                  else {**override, "why": RE_EXTRACT_PENDING})
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
                pending.append({**override, "why": MAP_PENDING})
            continue
        if kind == "which_measure":
            (applied if override.get("seq") in consumed
             else pending).append(override if override.get("seq") in consumed
                                  else {**override, "why": MAP_PENDING})
            continue
        if kind == "mark_reviewed":
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
    `higher_is_better`, retires the `orientation_unknown` flag it answers, records the reviewer's
    quote as the cell's orientation evidence, and re-derives the row through the ordinary
    resolution path — so the sign appears in the effect size and nowhere else.
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
            verdict.flags = [f for f in verdict.flags if f.code != "orientation_unknown"]
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


#: §C4, mandatory: "an override clears ONLY the named blocker". These are the findings a value
#: answer names, and nothing else is ever retired by one.
#:
#: A **mean** answers "which number, off which ladder": the readers' axis dispute, the ladder that
#: could not be corroborated, and a value the ladder cannot draw. It does NOT answer where the
#: number came from — an ungrounded quote, a row-only match, a sign the paper contradicts are
#: findings about provenance, and only the question that names them can retire them (its option
#: carries `clears`). A **spread type** answers what the error bars are, and only that. An `n` or
#: a spread VALUE answers that the number was missing.
VALUE_CLEARS_MEAN: frozenset[str] = frozenset({
    "axis_conflict", "calibration_disputed", "calibration_refuted", "calibration_single_witness",
    "calibration_missing", "value_outside_axis"})
VALUE_CLEARS_SPREAD_TYPE: frozenset[str] = frozenset({
    "dispersion_type_from_legend", "figure_error_bar_unknown", "dispersion_type_conflict"})
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
    if override.get("dispersion_type"):
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
    prepared = prepare_rows(cells, candidates, protocol.stats, cluster_of=cluster_of,
                            directions=directions)
    mine = [row for row in prepared if row.key == (dataset.dataset_id, outcome_key)]
    if mine:
        return mine[0]
    return prepare_rows([(dataset, outcome_key, verdict_a, verdict_b)], candidates,
                        protocol.stats, cluster_of=cluster_of, directions=directions)[0]


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
    values.flags = sorted({*values.flags, "human_override"})
    rebuilt = resolve_effect(dataset, protocol.outcome(outcome_key), values, protocol.stats)
    rebuilt.paper_id = record.paper_id
    rebuilt.sample_id = record.sample_id
    rebuilt.cluster_id = record.cluster_id
    rebuilt.citation = record.citation
    rebuilt.label = record.label
    rebuilt.moderators = record.moderators
    rebuilt.analysis_metric = record.analysis_metric
    rebuilt.flags = sorted({*rebuilt.flags, "human_override"})
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


def _apply_value(override: Mapping[str, Any], records: dict[tuple[str, str], EffectSizeRecord],
                 verdicts: dict[tuple[str, str, str], Verdict], state: _RunState,
                 protocol: Protocol, overruled: Collection[str] = ()) -> tuple[bool, str]:
    """Replace one group's numbers and re-derive that row through the ordinary resolution path."""
    dataset_id, outcome_key = override["dataset_id"], override["outcome_key"]
    group = override["group"]
    key = (dataset_id, outcome_key)
    record = records.get(key)
    dataset = state.datasets.get(dataset_id)
    if record is None or dataset is None:
        return False, f"no row for {dataset_id}/{outcome_key} in this run"
    verdict = verdicts.get((dataset_id, outcome_key, group))
    if verdict is None:
        return False, f"no verified cell for group {group} of {dataset_id}/{outcome_key}"

    if override.get("mean") is not None:
        verdict.mean = float(override["mean"])
    if override.get("dispersion_value") is not None:
        verdict.dispersion_value = float(override["dispersion_value"])
    if override.get("dispersion_type"):
        try:
            verdict.dispersion_type = DispersionType(override["dispersion_type"])
        except ValueError:
            return False, (f"unknown dispersion type {override['dispersion_type']!r} "
                           f"(expected one of {[d.value for d in DispersionType]})")
    if override.get("n") is not None:
        verdict.n = int(override["n"])
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
                                          all_rows=split.every)
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
                                 primary=primary_rows).items()})

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
    removed. Delete it, and stop the manifest promising it.
    """
    for suffix in ("png", "svg", "pdf"):
        if f"forest_{suffix}" in artefacts:
            continue
        stale = out / "results" / outcome_key / f"forest.{suffix}"
        stale.unlink(missing_ok=True)
        manifest.outputs.pop(f"{outcome_key}.forest_{suffix}", None)


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
