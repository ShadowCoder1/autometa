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
