"""Run state: stage files, atomic writes, per-paper accounting, and the human-review queue.

The orchestrator is resumable because every stage of every paper is a file. `run.py` asks
`stage_done` before it spends anything, and `read_stage` gives back exactly what the previous run
produced. Two properties matter and are tested:

* **Writes are atomic.** A run interrupted mid-write must not leave a half-written stage file that
  a later `--resume` reads as complete, so every write goes to a temporary file in the same
  directory and is renamed into place.
* **Cost is attributed to the paper that spent it.** The `LLMClient` is shared (one cache, one
  budget, one call log) and papers run concurrently, so a before/after reading of the client's
  total would attribute another thread's spending. `PaperClient` is a thin per-paper view that
  meters the calls it makes and enforces `max_usd_per_paper` itself.

The review queue is sorted by how far the pooled estimate would move if the value changed, which
is the only ordering that answers "where is a reviewer's next hour worth most?".
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..models import (Candidate, EffectSizeRecord, RunManifest, StatsSettings, Verdict)
from ..report.tables import pool_rows

__all__ = ["STAGES", "atomic_write", "write_json", "read_json", "paper_dir", "stage_path",
           "stage_done", "write_stage", "read_stage", "manifest_path", "save_manifest",
           "load_manifest", "PaperClient", "PaperBudgetExceeded", "review_entry",
           "sort_review_queue", "pooled_impact", "sha12", "emit"]

#: the per-paper stages, in the order they run; each writes `<out>/papers/<sha12>/<stage>.json`
STAGES: tuple[str, ...] = ("ingest", "map", "extract", "verify", "resolve")


def sha12(paper_id: str) -> str:
    return str(paper_id)[:12]


# ----------------------------------------------------------------------------- files
def atomic_write(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Write via a temporary file in the same directory, then rename — never a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.",
                                   suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=1, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paper_dir(out_dir: str | Path, paper_id: str) -> Path:
    return Path(out_dir) / "papers" / sha12(paper_id)


def stage_path(out_dir: str | Path, paper_id: str, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} (have {list(STAGES)})")
    return paper_dir(out_dir, paper_id) / f"{stage}.json"


def stage_done(out_dir: str | Path, paper_id: str, stage: str) -> bool:
    """Has this stage finished? A file that exists is not the same as a stage that completed.

    Size alone used to answer this, which is exactly wrong once a stage writes as it goes: a paper
    that died on its budget half way through extraction leaves a real, non-empty `extract.json`
    holding the rows it managed, and `--resume` read that as "done" and truncated the paper for
    good. A stage that writes incrementally marks its file `"complete": false` until it finishes,
    and this refuses to call that done.
    """
    path = stage_path(out_dir, paper_id, stage)
    if not (path.exists() and path.stat().st_size > 0):
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError):                    # a truncated or unreadable file is not done
        return False
    if isinstance(payload, dict) and payload.get("complete") is False:
        return False
    return True


def write_stage(out_dir: str | Path, paper_id: str, stage: str, payload: Any) -> Path:
    return write_json(stage_path(out_dir, paper_id, stage), payload)


def read_stage(out_dir: str | Path, paper_id: str, stage: str) -> Any:
    return read_json(stage_path(out_dir, paper_id, stage))


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "manifest.json"


def save_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    return atomic_write(manifest_path(run_dir),
                        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=1))


def load_manifest(run_dir: str | Path) -> RunManifest:
    return RunManifest.model_validate(read_json(manifest_path(run_dir)))


# ----------------------------------------------------------------------------- per-paper client
class PaperBudgetExceeded(RuntimeError):
    """One paper hit `max_usd_per_paper`. The run continues; this paper stops."""


class PaperClient:
    """A per-paper view of the shared `LLMClient`.

    Same cache, same fixtures, same global budget — but its own cost meter, its own call list and
    its own cap. Everything an agent might reach for (`pdf_block`, `live`, `count_tokens`, …) is
    delegated untouched; only the three call entry points are metered.
    """

    def __init__(self, client: Any, paper_id: str = "", max_usd: float | None = None):
        self._client = client
        self.paper_id = paper_id
        self.max_usd = max_usd
        self.cost_usd = 0.0
        self.call_ids: list[str] = []

    # the three ways a call reaches the provider -----------------------------
    def structured(self, **kwargs: Any) -> Any:
        return self._meter(self._client.structured, kwargs)

    def text(self, **kwargs: Any) -> Any:
        return self._meter(self._client.text, kwargs)

    def tool_loop(self, **kwargs: Any) -> Any:
        return self._meter(self._client.tool_loop, kwargs)

    def _meter(self, call: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        # the guard runs BEFORE the call and only before it. Guarding after the call as well meant
        # the exception was raised holding a result that had already been paid for, and the unwind
        # threw it away along with every candidate the stage had built (F7). Checked first, the
        # cap stops the NEXT call and nothing bought is ever lost.
        self._guard()
        result = call(**kwargs)
        self.cost_usd += float(getattr(result, "cost_usd", 0.0) or 0.0)
        ids = getattr(result, "call_ids", None)
        if ids:
            self.call_ids.extend(ids)
        elif getattr(result, "call_id", ""):
            self.call_ids.append(result.call_id)
        return result

    def _guard(self) -> None:
        """Raise before spending anything more, when the allowance is already gone."""
        if self.max_usd is not None and self.cost_usd >= self.max_usd:
            raise PaperBudgetExceeded(
                f"paper {sha12(self.paper_id)} has spent ${self.cost_usd:.4f} of its "
                f"${self.max_usd:.4f} allowance")

    @property
    def n_calls(self) -> int:
        return len(self.call_ids)

    def __getattr__(self, name: str) -> Any:            # everything else is the shared client's
        return getattr(self._client, name)


# ----------------------------------------------------------------------------- review queue
def _pool(rows: Sequence[EffectSizeRecord], settings: StatsSettings) -> float | None:
    """The pooled estimate, or `None` below k = 2. One pooler for the whole codebase."""
    result = pool_rows(rows, settings)
    return None if result is None else float(result.estimate)


def pooled_impact(primary: Sequence[EffectSizeRecord], record: EffectSizeRecord | None,
                  settings: StatsSettings) -> float | None:
    """|Δ pooled| if this row were added to (or removed from) the primary analysis.

    A row held for review is a question the tool is asking the reviewer; this says how much the
    answer could matter. `None` when the row carries no effect size, so it sorts last rather than
    pretending to be harmless.
    """
    if record is None or record.es is None or not record.var or record.var <= 0:
        return None
    base = _pool(primary, settings)
    if base is None:
        return None
    ids = {r.dataset_id for r in primary}
    if record.dataset_id in ids:
        other = _pool([r for r in primary if r.dataset_id != record.dataset_id], settings)
    else:
        other = _pool([*primary, record], settings)
    if other is None:
        return None
    return abs(base - other)


def review_entry(verdict: Verdict, *, paper_id: str, candidates: Sequence[Candidate] = (),
                 record: EffectSizeRecord | None = None, primary: Sequence[EffectSizeRecord] = (),
                 settings: StatsSettings | None = None) -> dict[str, Any]:
    """One queue row: what is uncertain, both candidate values, and what it would cost to be wrong."""
    reasons = list(verdict.confidence_reasons)
    reasons += [f"{flag.code}: {flag.message}" for flag in verdict.flags
                if flag.severity in ("warn", "error")]
    if verdict.verifier_verdict == "refuted":
        reasons.append(f"verifier refuted: {verdict.verifier_reason}")
    mine = [c for c in candidates
            if c.dataset_id == verdict.dataset_id and c.outcome_key == verdict.outcome_key
            and (verdict.group is None or c.group == verdict.group)]
    return {
        "paper_id": paper_id, "dataset_id": verdict.dataset_id,
        "outcome_key": verdict.outcome_key, "group": verdict.group,
        "confidence": verdict.confidence, "route": verdict.route,
        "reason": "; ".join(r for r in reasons if r) or "held for human review",
        "candidates": [{"candidate_id": c.candidate_id, "value": c.mean,
                        "dispersion_value": c.dispersion_value,
                        "dispersion_type": getattr(c.dispersion_type, "value", c.dispersion_type),
                        "n": c.n, "route": c.route, "model": c.model, "page": c.page,
                        "quote": c.quote, "status": c.status} for c in mine],
        "impact_abs_delta_pooled": (None if settings is None
                                    else pooled_impact(primary, record, settings)),
    }


def sort_review_queue(queue: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Worst first: the biggest |Δ pooled| at the top, unknown impact last."""
    def key(entry: Mapping[str, Any]) -> tuple[int, float]:
        value = entry.get("impact_abs_delta_pooled")
        try:
            return (0, -abs(float(value)))
        except (TypeError, ValueError):
            return (1, 0.0)
    return [dict(e) for e in sorted(queue, key=key)]


# ----------------------------------------------------------------------------- progress
def emit(progress: Callable[[dict[str, Any]], None] | None, stage: str, paper: str = "",
         status: str = "started", *, cost_so_far: float = 0.0, message: str = "") -> None:
    if progress is None:
        return
    progress({"stage": stage, "paper": paper, "status": status,
              "cost_so_far": round(float(cost_so_far), 6), "message": message})
