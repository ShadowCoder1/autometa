"""The review workflow: an append-only log of human decisions, and re-pooling under them.

A reviewer disagreeing with the tool is the point of the tool, so a disagreement is **evidence**,
not a correction: every decision is appended to `<run>/overrides.jsonl` with a justification, a
sequence number and a timestamp, and nothing is ever rewritten or deleted. The stage files the
pipeline wrote are never touched either — `repool` reads them, applies the log on top and rewrites
only the derived artefacts. That is what makes an override survive `--resume`: a resumed run
rebuilds the outputs from the same stage files, and re-applying the same log reproduces the same
reviewed analysis.

Five decisions (amendment I):

| kind | what it does |
|---|---|
| `value` | replaces one group's mean / dispersion / n for one cell, and re-derives that row's effect size |
| `mark_reviewed` | a human has checked the cell: it moves into the primary analysis at `accept_with_note` |
| `exclude_dataset` | that dataset leaves the analysis, and appears in the exclusion table |
| `eligibility` | a paper is in or out, whatever the mapper decided |
| `re_extract` | "read it again, with this hint" — needs a model, so `repool` reports it as pending |

`repool` makes **no** model calls. It computes nothing itself either: values are re-derived by
`canopy.pipeline.resolve.resolve_effect` and pooled by `canopy.stats`, exactly as the run did.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import (Candidate, DatasetSpec, DispersionType, EffectSizeRecord, Protocol,
                      RunManifest, StudyMap, Verdict)
from ..pipeline.resolve import ResolvedValues, resolve_effect
from ..pipeline.state import (load_manifest, read_stage, review_entry, save_manifest, sha12,
                              sort_review_queue, stage_done)
from ..protocol import load_protocol
from ..report import (dump_json, exclusions_table, pool_rows, write_html_report,
                      write_outcome_outputs, write_rows)

__all__ = ["KINDS", "OVERRIDES_FILE", "OverrideRejected", "append_override", "read_overrides",
           "apply_overrides_and_repool", "override_summary", "repool_lock"]

OVERRIDES_FILE = "overrides.jsonl"
SUMMARY_FILE = "overrides_applied.json"
KINDS: tuple[str, ...] = ("value", "mark_reviewed", "exclude_dataset", "eligibility",
                          "re_extract")
_MAX_TEXT = 4000
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class OverrideRejected(ValueError):
    """An override the server will not record (no justification, no target, unknown kind)."""


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
    if len(justification) < 3:
        raise OverrideRejected("every override needs a justification — that is the record a "
                               "reader of the review will check")

    record: dict[str, Any] = {
        "kind": kind,
        "paper_id": _text(payload.get("paper_id"), 120),
        "dataset_id": _text(payload.get("dataset_id"), 200),
        "outcome_key": _text(payload.get("outcome_key"), 200),
        "group": payload.get("group") if payload.get("group") in ("A", "B") else None,
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
        })
        if all(record.get(k) is None for k in ("mean", "dispersion_value", "n")):
            raise OverrideRejected("a value override must set at least one of mean, "
                                   "dispersion_value or n")
    if kind == "mark_reviewed":
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
    return record


def append_override(run_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and append one decision. Returns the record as it was written."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OVERRIDES_FILE
    record = _validate(payload)
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
    overrides = [o for o in read_overrides(out) if o.get("kind") in KINDS]
    state = _RunState(out, manifest)

    applied: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    records = {(r.dataset_id, r.outcome_key): r.model_copy(deep=True) for r in state.records}
    verdicts = {(v.dataset_id, v.outcome_key, v.group): v.model_copy(deep=True)
                for v in state.verdicts}
    dropped: set[tuple[str, str]] = set()

    for override in overrides:
        kind = override["kind"]
        dataset_id = override.get("dataset_id", "")
        outcome_key = override.get("outcome_key", "")
        if kind == "re_extract":
            pending.append({**override, "why": "a re-extraction needs a model call: run "
                                               "`canopy run --resume` with this hint"})
            continue
        if kind == "eligibility":
            hit = _apply_eligibility(override, records, dropped, excluded, state)
            (applied if hit else pending).append(override if hit else {
                **override, "why": "this paper has no extracted data in the run directory, so "
                                   "including it needs a model call (`canopy run --resume`)"})
            continue
        if kind == "exclude_dataset":
            targets = [key for key in records
                       if key[0] == dataset_id and (not outcome_key or key[1] == outcome_key)]
            for key in targets:
                dropped.add(key)
                excluded.append({
                    "paper_id": state.paper_of.get(dataset_id, ""), "filename": "",
                    "dataset_id": key[0], "outcome_key": key[1], "stage": "review",
                    "reason": "human_override", "quote": "",
                    "decider": override.get("actor", "human"),
                    "detail": override["justification"]})
            (applied if targets else pending).append(override if targets else {
                **override, "why": f"no row for dataset {dataset_id!r} in this run"})
            continue
        if kind == "mark_reviewed":
            record = records.get((dataset_id, outcome_key))
            if record is None:
                pending.append({**override, "why": f"no row for {dataset_id}/{outcome_key}"})
                continue
            bucket = override.get("confidence", "accept_with_note")
            record.confidence = bucket
            record.flags = sorted({*record.flags, "human_reviewed"})
            for group in ("A", "B"):
                verdict = verdicts.get((dataset_id, outcome_key, group))
                if verdict is not None:
                    verdict.confidence = bucket
                    verdict.needs_human = bucket == "needs_human"
            applied.append(override)
            continue
        if kind == "value":
            ok, why = _apply_value(override, records, verdicts, state, protocol)
            (applied if ok else pending).append(override if ok else {**override, "why": why})

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


def _apply_eligibility(override: Mapping[str, Any],
                       records: dict[tuple[str, str], EffectSizeRecord],
                       dropped: set[tuple[str, str]], excluded: list[dict[str, Any]],
                       state: _RunState) -> bool:
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


def _apply_value(override: Mapping[str, Any], records: dict[tuple[str, str], EffectSizeRecord],
                 verdicts: dict[tuple[str, str, str], Verdict], state: _RunState,
                 protocol: Protocol) -> tuple[bool, str]:
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
    verdict.needs_human = False
    if verdict.confidence == "needs_human":
        verdict.confidence = "accept_with_note"

    other = verdicts.get((dataset_id, outcome_key, "B" if group == "A" else "A"))
    if other is None:
        return False, f"the other group of {dataset_id}/{outcome_key} was never verified"
    verdict_a, verdict_b = (verdict, other) if group == "A" else (other, verdict)

    values = ResolvedValues.from_verdicts(verdict_a, verdict_b)
    values.flags = sorted({*values.flags, "human_override"})
    rebuilt = resolve_effect(dataset, protocol.outcome(outcome_key), values, protocol.stats)
    rebuilt.paper_id = record.paper_id
    rebuilt.cluster_id = record.cluster_id
    rebuilt.citation = record.citation
    rebuilt.label = record.label
    rebuilt.moderators = record.moderators
    rebuilt.analysis_metric = record.analysis_metric
    rebuilt.flags = sorted({*rebuilt.flags, "human_override"})
    rebuilt.notes = "; ".join(x for x in (record.notes, f"human override: "
                                                        f"{override['justification']}") if x)
    records[key] = rebuilt
    return True, ""


def _rewrite(out: Path, manifest: RunManifest, protocol: Protocol,
             records: Sequence[EffectSizeRecord], state: _RunState,
             verdicts: Mapping[tuple[str, str, str], Verdict],
             excluded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write every derived artefact again from the reviewed rows. No stage file is touched."""
    from ..pipeline.run import _primary_rows            # the run's own primary/held-back split

    live = list(verdicts.values())
    outcomes: dict[str, Any] = {}
    per_outcome: dict[str, dict[str, Any]] = {}
    review: list[dict[str, Any]] = []
    by_cell = {(r.dataset_id, r.outcome_key): r for r in records}

    for outcome in protocol.outcomes:
        mine = [r for r in records if r.outcome_key == outcome.key]
        primary, held, _ = _primary_rows(mine, protocol.stats)
        pooled = pool_rows(primary, protocol.stats)
        artefacts = write_outcome_outputs(out, outcome, primary, pooled, protocol.stats,
                                          needs_human_rows=held, verdicts=live,
                                          candidates=state.candidates,
                                          moderators=protocol.moderators or None)
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
        for verdict in live:
            if verdict.outcome_key != outcome.key or not verdict.needs_human:
                continue
            record = by_cell.get((verdict.dataset_id, verdict.outcome_key))
            review.append(review_entry(verdict, paper_id=record.paper_id if record else "",
                                       candidates=state.candidates, record=record,
                                       primary=primary, settings=protocol.stats))

    manifest.human_review_queue = sort_review_queue(review)
    write_rows(manifest.human_review_queue, out / "human_review_queue",
               ["paper_id", "dataset_id", "outcome_key", "group", "confidence", "route", "reason",
                "impact_abs_delta_pooled", "candidates"], formats=("csv", "json"))
    _rewrite_exclusions(out, excluded)
    run_outputs = {name: path for name, path in (
        ("methods_fig_png", out / "methods_routes.png"), ("prisma_png", out / "prisma.png"),
        ("provenance_json", out / "provenance" / "provenance.json")) if path.exists()}
    write_html_report(out, manifest, protocol, results=per_outcome,
                      review_queue=manifest.human_review_queue,
                      exclusions=_all_exclusions(out), run_outputs=run_outputs)
    save_manifest(out, manifest)
    return outcomes


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


def _all_exclusions(out: Path) -> list[dict[str, Any]]:
    path = out / "exclusions.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:                                     # pragma: no cover - defensive
        return []
    return [r for r in rows if isinstance(r, dict)]


def _rewrite_exclusions(out: Path, extra: Sequence[Mapping[str, Any]]) -> None:
    """Keep the run's own exclusions and add the reviewer's, without duplicating a row."""
    rows = _all_exclusions(out)
    seen = {(r.get("dataset_id"), r.get("outcome_key"), r.get("reason")) for r in rows}
    added = [dict(e) for e in extra
             if (e.get("dataset_id"), e.get("outcome_key"), e.get("reason")) not in seen]
    if added:
        exclusions_table([*rows, *added], out / "exclusions")
