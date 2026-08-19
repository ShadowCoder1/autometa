"""The `runs/nine` fixture: real records from a real run, read as models.

`tests/fixtures/runs/nine` is a copy of the nine-paper run's stage files — the study maps the
mapper wrote, the candidates the extractors found, the verdicts the verifiers reached, the twenty
records the resolver produced, and the pooled results the report was built from. Tests read it
through this module instead of hand-writing a record, because a hand-written record only ever
contains the fields whoever wrote it was thinking about: it agrees with any change made to the
models, which is exactly the property a fixture must not have. These files were written by the
pipeline and are never edited to suit a test — when one of them stops loading, the run that
produced it would have stopped loading too.

The copy is deliberate. `runs/` is git-ignored working output that a re-run overwrites; the
fixture is committed, so a test's meaning cannot change under it. Read it read-only: a test that
needs to write into a run copies the tree first (`copy_to(tmp_path)`).
"""
from __future__ import annotations

import copy
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from canopy.models import (
    Candidate,
    DatasetSpec,
    EffectSizeRecord,
    Protocol,
    StudyMap,
    Verdict,
)

#: the run directory itself — `nine.NINE / "manifest.json"` and friends
NINE = Path(__file__).resolve().parents[1] / "fixtures" / "runs" / "nine"

#: the six papers whose stage files were copied (the run had nine; three were excluded before
#: they resolved, so they carry no records worth pinning a test to)
PAPERS = ("3570e4ce2a9c", "5039533c85ef", "592b3b55a318", "b511dbb76fa6", "b7523a41b03a",
          "d1f2946e7e81")


@lru_cache(maxsize=None)
def _payload(path: Path) -> dict[str, Any]:
    """Parsed stage file, cached — `extract.json` runs to two megabytes and tests read it often.

    Callers never receive this dict: every accessor below either validates a model out of it
    (pydantic builds fresh containers, so no test can reach into another test's fixture) or
    hands back a deep copy.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(paper12: str, stage: str) -> dict[str, Any]:
    return _payload(NINE / "papers" / paper12 / f"{stage}.json")


def protocol() -> Protocol:
    """The protocol the run was made with (profile `cisneros2024`), profile resolved."""
    from canopy.protocol import load_protocol

    return load_protocol(NINE / "protocol.yaml")


def records(outcome_key: str | None = None) -> list[EffectSizeRecord]:
    """Every resolved row of the run, in paper order; one outcome's rows when named."""
    out: list[EffectSizeRecord] = []
    for path in sorted((NINE / "papers").glob("*/resolve.json")):
        out.extend(EffectSizeRecord.model_validate(r) for r in _payload(path)["records"])
    if outcome_key is None:
        return out
    return [r for r in out if r.outcome_key == outcome_key]


def record(dataset_id: str, outcome_key: str) -> EffectSizeRecord:
    """One cell's row. Raises rather than returning None: a test naming a cell means it."""
    for row in records(outcome_key):
        if row.dataset_id == dataset_id:
            return row
    raise KeyError(f"no record for {dataset_id}/{outcome_key} in {NINE}")


def verdicts(paper12: str) -> list[Verdict]:
    """The verifier's cell verdicts for one paper."""
    return [Verdict.model_validate(v) for v in _stage(paper12, "verify")["verdicts"]]


def candidates(paper12: str) -> list[Candidate]:
    """The extractors' candidates for one paper (extract stage only — see `verify.json`'s
    `extra_candidates` for the ones the verifier added)."""
    return [Candidate.model_validate(c) for c in _stage(paper12, "extract")["candidates"]]


def study(paper12: str) -> StudyMap:
    """The mapper's study map for one paper."""
    return StudyMap.model_validate(_stage(paper12, "map")["study"])


def dataset(paper12: str, dataset_id: str) -> DatasetSpec:
    return next(d for d in study(paper12).datasets if d.dataset_id == dataset_id)


def paper_json(paper12: str) -> dict[str, Any]:
    """The ingested paper record, as JSON (only two papers carry one — see the fixture tree)."""
    return copy.deepcopy(_payload(NINE / "papers" / paper12 / "ingest" / "paper.json"))


def copy_to(tmp_path: Path) -> Path:
    """A writable copy of the whole run under `tmp_path`, returned as its run directory."""
    dest = Path(tmp_path) / "nine"
    shutil.copytree(NINE, dest)
    return dest


def page_texts(paper12: str) -> list[str]:
    """The ingested page text of one paper, page 1 first.

    The stage files record where each page's text was written (`pages/pNNN.txt`), and the fixture
    carries those files for the two papers that have an `ingest/` record — because a check that
    reads the paper's own prose (`checks.n_before_exclusions`) can only be tested against prose
    the paper actually printed. Hand-written pages would agree with whatever pattern the check
    happens to use, which is the property this fixture exists to refuse.
    """
    record = _payload(NINE / "papers" / paper12 / "ingest" / "paper.json")
    base = NINE / "papers" / paper12 / "ingest"
    return [(base / page["text_file"]).read_text(encoding="utf-8") for page in record["pages"]]


# ------------------------------------------------------------- injecting what the fixture predates
# The fixture is the run as it stood before Tasks 2–5: its verdicts carry no `n_before_exclusions`
# and its records no `precedence_override`, because neither check nor the resolver's fallback
# existed when it was written. Re-recording it to suit a test is exactly what this module refuses
# (a fixture that agrees with every change is not evidence), so a test that needs one of those
# findings puts it on a COPY — the shape the producing code writes, on the cell the run really has.
def with_flags(tmp_run: Path, dataset_id: str, outcome_key: str, group: str,
               codes: list[str], *, detail: dict[str, Any] | None = None) -> Path:
    """Append `CheckFlag`-shaped entries to one verdict of the copied run's `verify.json`."""
    from canopy.verify.checks import severity_of

    path = Path(tmp_run) / "papers" / dataset_id.split(":")[0] / "verify.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for verdict in payload.get("verdicts") or []:
        if (verdict.get("dataset_id") == dataset_id and verdict.get("outcome_key") == outcome_key
                and verdict.get("group") == group):
            verdict["flags"] = [*(verdict.get("flags") or []),
                                *({"code": code, "severity": severity_of(code),
                                   "message": f"{code} on {dataset_id}/{outcome_key} {group}",
                                   "candidate_ids": [], "detail": dict(detail or {})}
                                  for code in codes)]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def with_record(tmp_run: Path, dataset_id: str, outcome_key: str, **fields: Any) -> Path:
    """Patch one resolved record of the copied run's `resolve.json` with the given fields."""
    path = Path(tmp_run) / "papers" / dataset_id.split(":")[0] / "resolve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload.get("records") or []:
        if record.get("dataset_id") == dataset_id and record.get("outcome_key") == outcome_key:
            record.update(fields)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path
