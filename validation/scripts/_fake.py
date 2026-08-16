"""The offline fake used by `run_cisneros.py --demo` — one seam, one place to change.

There is exactly one FakeProvider router in this repository and it lives in the pipeline's own
offline test (`tests/test_pipeline_offline.py`), where it is maintained alongside the pipeline it
fakes.  `--demo` reuses it rather than keeping a second, silently-drifting copy alive.

That test **defines** `FakeSpec` and `fake_router` inline, so they cannot be moved here without
editing the test — which this dispatch may not do (Task 15 owns `tests/test_*.py`).  This module
is therefore the seam rather than the source: everything that wants the fake imports it from
here, so when the fake moves to a shared home (`canopy/testing/fake_pipeline.py` is the obvious
one) exactly one file changes.

Nothing here is imported by the pipeline, and nothing here runs during a live run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

#: the two fixture papers the demo corpus is built from, in a stable order
DEMO_MEANS: tuple[tuple[float, float], ...] = ((44.6, 30.2), (21.5, 17.9))


class FakeUnavailable(RuntimeError):
    """The offline fake lives in the test package, which is not installed with `canopy`."""


def _load() -> tuple[Any, Any, Sequence[Path]]:
    try:
        from tests.test_pipeline_offline import PDFS, FakeSpec, fake_router
    except ImportError as exc:                             # pragma: no cover - packaged install
        raise FakeUnavailable(
            "`--demo` needs the repository's test package (tests/test_pipeline_offline.py), which "
            "holds the pipeline's own offline fake. Run it from a source checkout, or use "
            "`--replay tests/fixtures/llm` once the pipeline cassettes are recorded."
        ) from exc
    return FakeSpec, fake_router, PDFS


def demo_specs(work_dir: str | Path) -> list[Any]:
    """Really ingest both fixture PDFs and wrap each in the pipeline's own `FakeSpec`."""
    from canopy.ingest.pdf import ingest_pdf

    FakeSpec, _, pdfs = _load()
    root = Path(work_dir)
    return [FakeSpec(ingest_pdf(path, root / path.stem), a, b)
            for path, (a, b) in zip(pdfs, DEMO_MEANS)]


def demo_router(work_dir: str | Path) -> Any:
    """A router that answers every agent from the real fixture PDFs' own text."""
    _, fake_router, _ = _load()
    return fake_router(demo_specs(work_dir))


def demo_pdfs() -> list[Path]:
    return [Path(p) for p in _load()[2]]
