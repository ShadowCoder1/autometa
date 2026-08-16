#!/usr/bin/env python
"""Example 1 of 3 — a paper that PRINTS the means and SDs, in the text or a table.

    .venv/bin/python validation/scripts/example_text_ms.py \
        --run validation/out/run_cisneros_dev

Prints the whole chain for one such cell — the sentence the extractor quoted, the number each
model read out of it, what verification made of the disagreement (if any), and the arithmetic
that turned "44.6 ± 6.0 deg (n = 12)" into a Cohen's d with a confidence interval — and writes
`validation/out/example_text_ms.png|svg`, which shows the page with that sentence highlighted
next to the chain and the resulting effect size.

This is the easy route, and it is the one the tool should never get wrong: the number is written
down.  If the auto value differs from the human's here, the disagreement is about *which* number
belongs in the meta-analysis, not about reading it.

No model is called.  `--run` must be a finished run; add `--paper` to insist on one paper.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._example import Cell, build_parser, run_example  # noqa: E402

HINT = ("    .venv/bin/python validation/scripts/run_cisneros.py --split dev "
        "--out validation/out/run_cisneros_dev\n"
        "  (or point --run at a run that already contains it — the run's cache means an "
        "already-answered question is never paid for twice)")


def wants(cell: Cell) -> bool:
    """A cell resolved from printed means and dispersions — text or table, SD, SE or CI."""
    if cell.record.route not in ("text_mean_sd", "table", "text_mean_se_ci"):
        return False
    # and the winning candidates really are printed readings, not a figure read-out
    # (`Candidate.kind` is what was extracted, `Candidate.route` is where it was read from)
    return any(c.kind == "group_stats" and c.route == "text" and c.mean is not None
               for c in cell.candidates)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args(argv)
    return run_example(args, wants=wants, stem="example_text_ms",
                       subtitle="route: printed means ± dispersion (text or table)",
                       missing_hint=HINT)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
