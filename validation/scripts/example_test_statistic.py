#!/usr/bin/env python
"""Example 3 of 3 — a paper that reports only a test statistic (t or F), no means.

    .venv/bin/python validation/scripts/example_test_statistic.py \
        --run validation/out/run_cisneros_dev

When a paper prints "t(22) = 3.14, p = .005" and no group means, an effect size can still be
recovered — but only under conditions the code checks before it will do the arithmetic
(amendment C, `canopy.stats.effect_sizes.smd_from_t` / `smd_from_f`):

* the design must be `independent_t` or `one_way_between` — a paired t or an interaction term
  does NOT convert to a between-groups d, and the code raises `NotConvertible` rather than
  pretending;
* the degrees of freedom must match the group sizes (df ≈ nA + nB − 2, within ±2), or the record
  is flagged;
* the direction must be known, because a t statistic has no sign the way a mean difference does.

So this script prints two things: the statistic the extractor found (with its design, df, tails
and admissibility ruling) and what the resolver decided to do with it.  **Two outcomes are
normal and both are informative** — either the statistic route won (and you see the conversion),
or a printed/plotted route outranked it (`route_precedence` puts test statistics fifth) and the
statistic is shown as the *fallback* it would have been.  The script says which case you are
looking at instead of hiding one of them.

No model is called.
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
        "  If no paper in the corpus reports a usable t/F for these outcomes, that is itself a "
        "finding: say so in the validation report rather than inventing one.")


def has_statistic(cell: Cell) -> bool:
    return any(c.kind == "test_statistic" and c.stat_value is not None for c in cell.candidates)


def wants(cell: Cell) -> bool:
    """First choice: the statistic route actually won.  Fallback: a statistic was available."""
    return cell.record.route in ("test_statistic", "p_value") and has_statistic(cell)


def wants_fallback(cell: Cell) -> bool:
    return has_statistic(cell)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args(argv)

    code = run_example(args, wants=wants, stem="example_test_statistic",
                       subtitle="route: test statistic (t / F) converted to d",
                       missing_hint=HINT)
    if code == 0:
        return 0

    print(banner("no cell was RESOLVED from a test statistic — falling back"))
    print("  `route_precedence` prefers printed means, then tables, then figures, so a paper that\n"
          "  reports both a t and a mean is resolved from the mean.  The cell below is the one\n"
          "  whose t/F WOULD have been used if the earlier routes had failed; the conversion the\n"
          "  resolver would have run is shown under `routes_rejected` when it refused it.")
    return run_example(args, wants=wants_fallback, stem="example_test_statistic",
                       subtitle="route: test statistic available as a FALLBACK (another route won)",
                       missing_hint=HINT)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
