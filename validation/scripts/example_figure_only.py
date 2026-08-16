#!/usr/bin/env python
"""Example 2 of 3 — a paper whose numbers exist ONLY in a figure.

    .venv/bin/python validation/scripts/example_figure_only.py \
        --run validation/out/run_cisneros_dev --paper "tests/fixtures/pdfs/bock2005.pdf"

Bock (2005) is the canonical case: the group means and error bars for the adaptation block are
plotted and never printed, so the only way to a Cohen's d is to measure the picture.  Four
independent routes do that (amendment F) and they vote:

    A  vector      the PDF's own drawing operators — exact when the figure is not a raster
    B  raster CV   colour-segmented bars/markers, cap ends found by edge detection
    C  VLM coords  a model points at each mark, the pixel is then SNAPPED to a detected edge
    D  read-out    a model reads the values off the axis, twice per model family

The script prints every route's sample, the ensemble's median and robust spread, the axis
calibration it used (OCR'd tick labels, least-squares fit), the digitisation sigma that becomes
part of the variance, and the conversion to d.  The figure it writes shows the digitiser's own
overlay — its marks drawn on the plot — beside the chain and the resulting effect size, so you
can see exactly which pixel became which number.

No model is called: everything comes from the run's stored provenance.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._common import banner  # noqa: E402
from validation.scripts._example import Cell, build_parser, run_example  # noqa: E402

HINT = ("    .venv/bin/python validation/scripts/run_cisneros.py --split dev "
        "--out validation/out/run_cisneros_dev\n"
        "  offline alternative (no credit, recorded cassettes):\n"
        "    .venv/bin/python validation/scripts/run_cisneros.py --replay tests/fixtures/llm "
        "--papers tests/fixtures/pdfs --out validation/out/run_bock_replay")


def has_figure_reading(cell: Cell) -> bool:
    return any(c.route == "figure" and c.mean is not None for c in cell.candidates)


def wants(cell: Cell) -> bool:
    """First choice: the figure route actually won the cell."""
    return cell.record.route == "figure" and has_figure_reading(cell)


def wants_fallback(cell: Cell) -> bool:
    """Fallback: the digitiser read the figure, but the cell was held back anyway."""
    return has_figure_reading(cell)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args(argv)

    code = run_example(args, wants=wants, stem="example_figure_only",
                       subtitle="route: figure — four digitiser routes vote, the ensemble wins",
                       missing_hint=HINT)
    if code == 0:
        return 0

    print(banner("no cell was RESOLVED from a figure — falling back to one that was HELD BACK"))
    print("  A digitised cell that did not survive verification is worth looking at, not hiding:\n"
          "  the routes' samples, the ensemble's tolerance test and the verifier's objection are\n"
          "  exactly what a human needs in order to decide the cell. That is what follows.")
    return run_example(args, wants=wants_fallback, stem="example_figure_only",
                       subtitle="route: figure — read by the digitiser, then HELD for a human",
                       missing_hint=HINT)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
