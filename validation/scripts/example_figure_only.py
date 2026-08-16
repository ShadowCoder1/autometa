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

from validation.scripts._example import Cell, build_parser, run_example  # noqa: E402

HINT = ("    .venv/bin/python validation/scripts/run_cisneros.py --split dev "
        "--out validation/out/run_cisneros_dev\n"
        "  offline alternative (no credit, recorded cassettes):\n"
        "    .venv/bin/python validation/scripts/run_cisneros.py --replay tests/fixtures/llm "
        "--papers tests/fixtures/pdfs --out validation/out/run_bock_replay")


def wants(cell: Cell) -> bool:
    """A cell whose winning route was the figure."""
    if cell.record.route != "figure":
        return False
    return any(c.route == "figure" and c.mean is not None for c in cell.candidates)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args(argv)
    return run_example(args, wants=wants, stem="example_figure_only",
                       subtitle="route: figure — four digitiser routes vote, the ensemble wins",
                       missing_hint=HINT)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
