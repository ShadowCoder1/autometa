#!/usr/bin/env python
"""The auto forest beside the manual one — same rows, same sort, same conventions.

    .venv/bin/python validation/scripts/replot_forest.py \
        --run validation/out/run_cisneros_dev --outcome late_adaptation

Writes into `validation/out/`:

* `forest_manual_<outcome>.png|svg`  — the human's TE/seTE, drawn by `canopy.report.forest`;
* `forest_auto_<outcome>.png|svg`    — the run's own rows, drawn by the same code;
* `forest_side_by_side_<outcome>.png|svg` — the two, stitched, sorted identically so a reader's
  eye can travel across a row;
* `forest_comparison_<outcome>.json` — k, pooled d, CI, τ², I² and the prediction interval for
  both sides.

**Where the weights come from.** Both sides are re-pooled here by
`canopy.stats.meta.random_effects`, and every study's random-effects weight is
`1/(vᵢ + τ²)` — precision, which is driven by sample size *and* by how noisy that study was.
Neither the manual spreadsheet's weights nor the run's stored ones are copied: they are
recomputed from the same estimator so the two plots are comparable by construction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._common import (GOLD_DIR, GOLD_FOR_OUTCOME, OUT_DIR, fmt,  # noqa: E402
                                        gold_records, load_gold, load_run, write_json)


def _pool(records: Sequence[Any], settings: Any) -> Any:
    from canopy.report.tables import pool_rows

    return pool_rows(records, settings)


def _summary(pooled: Any, records: Sequence[Any], settings: Any) -> dict[str, Any]:
    if pooled is None:
        return {"k": 0, "k_papers": len({r.cluster_id or r.paper_id for r in records})}
    from canopy.stats.meta import prediction_interval

    try:
        pi_low, pi_high, pi_df = prediction_interval(pooled, str(settings.pi_method))
    except ValueError:
        pi_low = pi_high = pi_df = None
    return {"k": pooled.k, "k_papers": len({r.cluster_id or r.paper_id for r in records}),
            "estimate": pooled.estimate, "ci_low": pooled.ci_low, "ci_high": pooled.ci_high,
            "se": pooled.se, "tau2": pooled.tau2, "i2_pct": 100.0 * pooled.I2, "q": pooled.Q,
            "p": pooled.p, "pi_low": pi_low, "pi_high": pi_high, "pi_df": pi_df,
            "weights_pct": [float(w) for w in pooled.weights_pct]}


def _order(records: Sequence[Any]) -> list[Any]:
    """`forest_layout` sorts by effect; do the same here so both panels agree on row order."""
    return sorted(records, key=lambda r: (r.es is None, r.es if r.es is not None else 0.0))


def side_by_side(left_png: Path, right_png: Path, out_stem: Path,
                 formats: Sequence[str] = ("png", "svg")) -> dict[str, Path]:
    """Stitch two rendered forests into one figure (no re-drawing: the panels stay identical)."""
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg
    from canopy.report.theme import figure_style, save_figure

    with figure_style():
        images = [mpimg.imread(left_png), mpimg.imread(right_png)]
        heights = [im.shape[0] / im.shape[1] for im in images]
        width = 15.0
        fig = plt.figure(figsize=(width, width / 2 * max(heights) + 0.25))
        for index, (image, title) in enumerate(zip(images, ("manual (human extraction)",
                                                            "automatic (Canopy)"))):
            ax = fig.add_subplot(1, 2, index + 1)
            ax.imshow(image)
            ax.set_title(title, fontsize=11, pad=6)
            ax.axis("off")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005, wspace=0.02)
        return save_figure(fig, out_stem, formats=formats, bbox_inches=None)


def replot(run_dir: str | Path, outcome_key: str, *, out_dir: str | Path = OUT_DIR,
           gold_dir: str | Path = GOLD_DIR,
           formats: Sequence[str] = ("png", "svg")) -> dict[str, Any]:
    from canopy.report.forest import forest_plot

    run = load_run(run_dir, papers=False)
    settings = run.settings
    outcome = run.protocol.outcome(outcome_key)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gold = load_gold(outcome_key, gold_dir)
    manual_rows = _order(gold_records(gold, settings))
    manual_pooled = _pool(manual_rows, settings)

    auto_all = run.rows_for(outcome_key)
    auto_rows = _order([r for r in auto_all
                        if r.confidence in settings.primary_analysis_includes
                        and r.es is not None and r.var not in (None, 0)])
    held = [r for r in auto_all if r not in auto_rows]
    auto_pooled = _pool(auto_rows, settings)

    payload: dict[str, Any] = {
        "outcome_key": outcome_key, "run_dir": str(run_dir),
        "gold_file": GOLD_FOR_OUTCOME[outcome_key],
        "settings": {"estimator": settings.estimator, "variance": settings.variance,
                     "tau2_method": settings.tau2_method, "hakn": settings.hakn,
                     "pi_method": settings.pi_method,
                     "weights": "random-effects 1/(v_i + tau^2), recomputed for both sides"},
        "manual": _summary(manual_pooled, manual_rows, settings),
        "auto": _summary(auto_pooled, auto_rows, settings),
        "n_auto_held_for_review": len(held),
        "files": {},
    }

    if manual_pooled is not None:
        files = forest_plot(manual_rows, manual_pooled, outcome, settings,
                            out / f"forest_manual_{outcome_key}", formats=formats,
                            title=f"{outcome.label} — manual extraction (Cisneros et al. 2024)",
                            subtitle="human-extracted d, seTE from the review's own spreadsheet")
        payload["files"].update({f"manual_{k}": str(v) for k, v in files.items()})
    if auto_pooled is not None:
        files = forest_plot(auto_rows, auto_pooled, outcome, settings,
                            out / f"forest_auto_{outcome_key}", needs_human_rows=held,
                            moderators=run.protocol.moderators or None, formats=formats,
                            title=f"{outcome.label} — automatic extraction (Canopy)",
                            subtitle="agent-extracted values; weights recomputed by "
                                     "canopy.stats.meta.random_effects")
        payload["files"].update({f"auto_{k}": str(v) for k, v in files.items()})

    left = payload["files"].get("manual_png")
    right = payload["files"].get("auto_png")
    if left and right:
        files = side_by_side(Path(left), Path(right),
                             out / f"forest_side_by_side_{outcome_key}", formats=formats)
        payload["files"].update({f"side_by_side_{k}": str(v) for k, v in files.items()})

    payload["files"]["comparison_json"] = str(
        write_json(payload, out / f"forest_comparison_{outcome_key}.json"))
    return payload


def report(payload: dict[str, Any]) -> str:
    lines = [f"\n{payload['outcome_key']}", "─" * 72,
             f"{'':<10}{'k':>4} {'k papers':>9} {'d':>8} {'95% CI':>20} {'tau2':>8} "
             f"{'I2':>7} {'PI':>20}"]
    for side in ("manual", "auto"):
        s = payload[side]
        ci = f"[{fmt(s.get('ci_low'), 2)}, {fmt(s.get('ci_high'), 2)}]"
        pi = f"[{fmt(s.get('pi_low'), 2)}, {fmt(s.get('pi_high'), 2)}]"
        lines.append(f"{side:<10}{s.get('k', 0):>4} {s.get('k_papers', 0):>9} "
                     f"{fmt(s.get('estimate'), 3):>8} {ci:>20} {fmt(s.get('tau2'), 3):>8} "
                     f"{fmt(s.get('i2_pct'), 1):>7} {pi:>20}")
    lines.append(f"  held for review (auto): {payload['n_auto_held_for_review']}")
    for name, path in payload["files"].items():
        lines.append(f"  wrote {name:<20} {path}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outcome", action="append", default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--gold", type=Path, default=GOLD_DIR)
    args = parser.parse_args(argv)

    for key in (args.outcome or list(GOLD_FOR_OUTCOME)):
        print(report(replot(args.run, key, out_dir=args.out, gold_dir=args.gold)))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
