#!/usr/bin/env python
"""Backward validation: the human's Cohen's d against Canopy's, one point per dataset.

    .venv/bin/python validation/scripts/compare_manual_vs_auto.py \
        --run validation/out/run_cisneros_dev --outcome late_adaptation

Writes, per outcome, into `validation/out/`:

* `manual_vs_auto_<outcome>.png|svg` — the scatter the user asked for: manual d on x, auto d on
  y, **95 % CI bars on both axes**, the identity line, one marker shape per extraction route,
  hollow markers for rows the tool held back for a human, and every point that missed tolerance
  labelled so it can be looked up;
* `discrepancies_<outcome>.csv` — every pair, with the columns a person fills in while
  adjudicating (`classification`, `adjudicated_truth`, `adjudicator_note`) and everything they
  need to do it without opening the run (route, page, quote, conversion chain, both CIs);
* `agreement_<outcome>.json` — CCC, MAE, RMSE, sign agreement, Bland–Altman bias and limits,
  auto-accept precision, per-bucket calibration, and the pooled-estimate comparison.

Nothing here recomputes a statistic by hand: pooling is `canopy.stats.meta.random_effects` on both
sides, and the auto CIs are the ones the run wrote.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._common import (GOLD_DIR, GOLD_FOR_OUTCOME, OUT_DIR, TOLERANCE_D,  # noqa: E402
                                        Pair, bland_altman, ci_of, fmt, join_rows, lins_ccc,
                                        load_gold, load_overrides, load_run, mae, pool_gold,
                                        rmse, sign_agreement, write_csv, write_json)

OVERRIDES = GOLD_DIR / "join_overrides.csv"

#: one marker per route bucket — shape, never hue alone, so the plot survives being printed
ROUTE_MARKERS = {"text": "o", "table": "s", "figure": "^", "test_statistic": "D",
                 "reported_d": "v", "not_convertible": "x", "other": "P", "manual": "o"}


# ----------------------------------------------------------------------------- the numbers
def agreement(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Every headline agreement statistic, over the pairs that have both numbers."""
    matched = [p for p in pairs if p.auto is not None and p.gold is not None
               and p.auto_d is not None and p.manual_d is not None]
    manual = [p.manual_d for p in matched]
    auto = [p.auto_d for p in matched]
    ba = bland_altman(manual, auto)
    out: dict[str, Any] = {
        "n_pairs": len(matched),
        "n_unmatched_auto": sum(1 for p in pairs if p.how == "unmatched_auto"),
        "n_unmatched_gold": sum(1 for p in pairs if p.how == "unmatched_gold"),
        "n_auto_without_effect": sum(1 for p in pairs if p.auto is not None
                                     and p.auto_d is None),
        "join_rules": {rule: sum(1 for p in pairs if p.how == rule)
                       for rule in sorted({p.how for p in pairs})},
        "tolerance_d": TOLERANCE_D,
        "lins_ccc": lins_ccc(manual, auto),
        "mae": mae(manual, auto),
        "rmse": rmse(manual, auto),
        "sign_agreement": sign_agreement(manual, auto),
        "within_tolerance": (sum(1 for p in matched if p.within_tolerance) / len(matched)
                             if matched else None),
        "bland_altman": ba,
        "max_abs_delta": max((abs(p.delta) for p in matched if p.delta is not None), default=None),
    }
    # amendment J: is a bucket's name honest?  `auto_accept` must mean "you can use this".
    buckets: dict[str, dict[str, Any]] = {}
    for pair in matched:
        entry = buckets.setdefault(pair.auto.confidence,
                                   {"n": 0, "within_tolerance": 0, "abs_deltas": []})
        entry["n"] += 1
        entry["within_tolerance"] += int(pair.within_tolerance)
        entry["abs_deltas"].append(abs(pair.delta))
    for entry in buckets.values():
        deltas = entry.pop("abs_deltas")
        entry["precision"] = entry["within_tolerance"] / entry["n"] if entry["n"] else None
        entry["mean_abs_delta"] = sum(deltas) / len(deltas) if deltas else None
        entry["max_abs_delta"] = max(deltas) if deltas else None
    out["calibration_by_confidence"] = buckets
    out["auto_accept_precision"] = buckets.get("auto_accept", {}).get("precision")

    by_route: dict[str, dict[str, Any]] = {}
    from canopy.report.methods_fig import route_group

    for pair in matched:
        entry = by_route.setdefault(route_group(pair.auto.route),
                                    {"n": 0, "within_tolerance": 0, "sum_abs": 0.0})
        entry["n"] += 1
        entry["within_tolerance"] += int(pair.within_tolerance)
        entry["sum_abs"] += abs(pair.delta)
    for entry in by_route.values():
        entry["mean_abs_delta"] = entry.pop("sum_abs") / entry["n"]
    out["by_route"] = by_route
    return out


def pooled_comparison(run: Any, outcome_key: str, gold: Sequence[Any]) -> dict[str, Any]:
    """Auto vs manual pooled estimate, CI, τ² and I² — both through `canopy.stats.meta`.

    The manual side is `metagen`'s own inputs (the spreadsheet's TE and seTE) pooled under the
    protocol's settings, so any difference is a difference in the *data*, not in the estimator.
    """
    settings = run.settings
    manual = pool_gold(gold, settings)
    auto = run.pooled.get(outcome_key, {})

    def side(res: Any) -> dict[str, Any]:
        if res is None:
            return {"k": 0}
        return {"k": res.k, "estimate": res.estimate, "ci_low": res.ci_low,
                "ci_high": res.ci_high, "se": res.se, "tau2": res.tau2,
                "i2_pct": 100.0 * res.I2, "q": res.Q, "p": res.p}

    manual_side = side(manual)
    i2 = auto.get("I2")
    auto_side = {"k": auto.get("k", 0), "estimate": auto.get("estimate"),
                 "ci_low": auto.get("ci_low"), "ci_high": auto.get("ci_high"),
                 "se": auto.get("se"), "tau2": auto.get("tau2"),
                 "i2_pct": None if i2 is None else 100.0 * i2,
                 "q": auto.get("Q"), "p": auto.get("p")}
    delta = (None if auto_side["estimate"] is None or manual_side.get("estimate") is None
             else auto_side["estimate"] - manual_side["estimate"])
    overlap = None
    if all(v is not None for v in (auto_side["ci_low"], auto_side["ci_high"],
                                   manual_side.get("ci_low"), manual_side.get("ci_high"))):
        overlap = (auto_side["ci_low"] <= manual_side["ci_high"]
                   and manual_side["ci_low"] <= auto_side["ci_high"])
    return {"manual": manual_side, "auto": auto_side, "delta_estimate": delta,
            "cis_overlap": overlap,
            "settings": {"estimator": settings.estimator, "variance": settings.variance,
                         "tau2_method": settings.tau2_method, "hakn": settings.hakn,
                         "pi_method": settings.pi_method},
            "note": ("both sides pooled by canopy.stats.meta.random_effects under the protocol's "
                     "own settings; the manual side uses the spreadsheet's TE and seTE unchanged")}


# ----------------------------------------------------------------------------- the table
def discrepancy_rows(pairs: Sequence[Pair], run: Any) -> list[dict[str, Any]]:
    """One row per pair, with everything an adjudicator needs and the columns they fill in."""
    from canopy.report.methods_fig import route_group

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        auto, gold = pair.auto, pair.gold
        auto_low, auto_high = ci_of(auto) if auto is not None else (None, None)
        quote = page = ""
        if auto is not None:
            chosen = [c for c in run.candidates_for(auto.dataset_id, auto.outcome_key)
                      if c.quote]
            if chosen:
                quote, page = chosen[0].quote[:220], chosen[0].page
        rows.append({
            "outcome_key": pair.outcome_key,
            "study": pair.label,
            "join_rule": pair.how,
            "join_note": pair.note,
            "gold_row_index": gold.index if gold else None,
            "manual_d": gold.te if gold else None,
            "manual_ci_low": gold.ci_low if gold else None,
            "manual_ci_high": gold.ci_high if gold else None,
            "manual_se": gold.se if gold else None,
            "manual_n_old": gold.n_old if gold else None,
            "manual_n_young": gold.n_young if gold else None,
            "manual_measure": gold.measure if gold else "",
            "manual_figure": gold.figure if gold else "",
            "auto_dataset_id": auto.dataset_id if auto else "",
            "auto_paper_id": (auto.paper_id[:12] if auto else ""),
            "auto_d": auto.es if auto else None,
            "auto_ci_low": auto_low, "auto_ci_high": auto_high,
            "auto_se": auto.se if auto else None,
            "auto_n_old": auto.n_a if auto else None,
            "auto_n_young": auto.n_b if auto else None,
            "auto_route": auto.route if auto else "",
            "auto_route_bucket": route_group(auto.route) if auto else "",
            "auto_confidence": auto.confidence if auto else "",
            "auto_flags": ";".join(auto.flags) if auto else "",
            "auto_conversion_chain": auto.conversion_chain if auto else "",
            "auto_page": page, "auto_quote": quote,
            "delta_auto_minus_manual": pair.delta,
            "abs_delta": None if pair.delta is None else abs(pair.delta),
            "within_tolerance": ("" if pair.delta is None else str(pair.within_tolerance).lower()),
            # ---- filled in by hand while adjudicating (amendment J) ----
            "classification": "",
            "adjudicated_truth": "",
            "adjudicator_note": "",
        })
    rows.sort(key=lambda r: (-(r["abs_delta"] or -1), r["study"]))
    return rows


# ----------------------------------------------------------------------------- the figure
def scatter(pairs: Sequence[Pair], outcome: Any, out_stem: str | Path, *,
            summary: dict[str, Any], formats: Sequence[str] = ("png", "svg")) -> dict[str, Path]:
    """Manual d (x) against auto d (y), with CI bars on both axes."""
    import matplotlib.pyplot as plt
    from canopy.report.methods_fig import route_group
    from canopy.report.theme import ACCENT, GRID, INK, INK_SECONDARY, MUTED, figure_style, save_figure

    matched = [p for p in pairs if p.auto is not None and p.gold is not None
               and p.auto_d is not None and p.manual_d is not None]

    with figure_style():
        fig, ax = plt.subplots(figsize=(7.4, 7.0))
        values: list[float] = []
        for pair in matched:
            values += [pair.manual_d, pair.auto_d]
            low, high = ci_of(pair.auto)
            values += [v for v in (low, high, pair.gold.ci_low, pair.gold.ci_high)
                       if v is not None]
        if not values:
            values = [-1.0, 1.0]
        lo, hi = min(values), max(values)
        pad = 0.08 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, lw=1.0, zorder=1,
                label="identity (auto = manual)")
        ax.fill_between([lo, hi], [lo - TOLERANCE_D, hi - TOLERANCE_D],
                        [lo + TOLERANCE_D, hi + TOLERANCE_D], color=ACCENT, alpha=0.07,
                        lw=0, zorder=0, label=f"±{TOLERANCE_D} agreement band")
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)

        seen: set[str] = set()
        for pair in matched:
            bucket = route_group(pair.auto.route)
            marker = ROUTE_MARKERS.get(bucket, "P")
            held = pair.auto.confidence == "needs_human"
            low, high = ci_of(pair.auto)
            ax.errorbar(pair.manual_d, pair.auto_d,
                        yerr=[[pair.auto_d - low], [high - pair.auto_d]] if None not in (low, high)
                        else None,
                        xerr=[[pair.manual_d - pair.gold.ci_low], [pair.gold.ci_high - pair.manual_d]]
                        if None not in (pair.gold.ci_low, pair.gold.ci_high) else None,
                        fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=0, zorder=2, alpha=0.75)
            label = None
            if bucket not in seen:
                seen.add(bucket)
                label = bucket.replace("_", " ")
            ax.scatter([pair.manual_d], [pair.auto_d], marker=marker, s=46,
                       facecolors="none" if held else ACCENT,
                       edgecolors=ACCENT if held else "white", linewidths=1.3, zorder=3,
                       label=label)
            if not pair.within_tolerance:
                ax.annotate(pair.label, (pair.manual_d, pair.auto_d), fontsize=6.6,
                            color=INK_SECONDARY, xytext=(4, 4), textcoords="offset points")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("manual Cohen's d  (Cisneros et al. 2024 spreadsheet)", fontsize=9)
        ax.set_ylabel("Canopy Cohen's d", fontsize=9)
        ax.set_title(f"{outcome.label}: every dataset the human and the tool both extracted",
                     fontsize=10.5, color=INK, pad=10)
        ax.legend(loc="upper left", fontsize=7.4, frameon=False, handletextpad=0.5,
                  borderaxespad=0.4)

        text = (f"k = {summary['n_pairs']}   CCC = {fmt(summary['lins_ccc'])}   "
                f"MAE = {fmt(summary['mae'])}   sign agreement = "
                f"{fmt(summary['sign_agreement'], 2)}\n"
                f"within ±{TOLERANCE_D}: {fmt(summary['within_tolerance'], 2)}   "
                f"bias (auto − manual) = {fmt(summary['bland_altman']['bias'])}   "
                f"hollow = held for human review\n"
                f"unmatched: {summary['n_unmatched_auto']} auto, "
                f"{summary['n_unmatched_gold']} manual")
        fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=7.2, color=MUTED,
                 linespacing=1.5)
        fig.subplots_adjust(bottom=0.16, top=0.94, left=0.11, right=0.97)
        return save_figure(fig, out_stem, formats=formats)


# ----------------------------------------------------------------------------- main
def compare(run_dir: str | Path, outcome_key: str, *, out_dir: str | Path = OUT_DIR,
            gold_dir: str | Path = GOLD_DIR,
            overrides_path: str | Path = OVERRIDES) -> dict[str, Any]:
    """Everything for one outcome: join, statistics, table, figure.  Returns the summary."""
    run = load_run(run_dir, papers=False)
    outcome = run.protocol.outcome(outcome_key)
    gold = load_gold(outcome_key, gold_dir)
    records = run.rows_for(outcome_key)
    pairs = join_rows(records, gold, outcome_key,
                      overrides=load_overrides(overrides_path), studies=run.studies)

    summary = agreement(pairs)
    summary["outcome_key"] = outcome_key
    summary["run_dir"] = str(run_dir)
    summary["gold_file"] = GOLD_FOR_OUTCOME[outcome_key]
    summary["n_gold_rows"] = len(gold)
    summary["n_auto_rows"] = len(records)
    summary["pooled"] = pooled_comparison(run, outcome_key, gold)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = discrepancy_rows(pairs, run)
    summary["files"] = {
        "discrepancies": str(write_csv(rows, out / f"discrepancies_{outcome_key}.csv")),
    }
    if summary["n_pairs"]:
        figures = scatter(pairs, outcome, out / f"manual_vs_auto_{outcome_key}",
                          summary=summary)
        summary["files"].update({k: str(v) for k, v in figures.items()})
    summary["files"]["agreement"] = str(
        write_json(summary, out / f"agreement_{outcome_key}.json"))
    return summary


def report(summary: dict[str, Any]) -> str:
    pooled = summary["pooled"]
    lines = [
        f"\n{summary['outcome_key']}  ({summary['n_auto_rows']} auto rows, "
        f"{summary['n_gold_rows']} manual rows)",
        "─" * 72,
        f"  matched pairs        {summary['n_pairs']}   "
        f"(rules: {summary['join_rules']})",
        f"  unmatched            {summary['n_unmatched_auto']} auto, "
        f"{summary['n_unmatched_gold']} manual",
        f"  Lin's CCC            {fmt(summary['lins_ccc'])}",
        f"  MAE / RMSE           {fmt(summary['mae'])} / {fmt(summary['rmse'])}",
        f"  sign agreement       {fmt(summary['sign_agreement'], 2)}",
        f"  within ±{summary['tolerance_d']}          {fmt(summary['within_tolerance'], 2)}",
        f"  bias (auto−manual)   {fmt(summary['bland_altman']['bias'])}  "
        f"[LoA {fmt(summary['bland_altman']['loa_low'])}, "
        f"{fmt(summary['bland_altman']['loa_high'])}]",
        f"  auto_accept precision {fmt(summary['auto_accept_precision'], 2)}",
        "  pooled  manual  " + f"d = {fmt(pooled['manual'].get('estimate'))} "
        f"[{fmt(pooled['manual'].get('ci_low'))}, {fmt(pooled['manual'].get('ci_high'))}]  "
        f"k = {pooled['manual'].get('k')}  I² = {fmt(pooled['manual'].get('i2_pct'), 1)}%",
        "  pooled  auto    " + f"d = {fmt(pooled['auto'].get('estimate'))} "
        f"[{fmt(pooled['auto'].get('ci_low'))}, {fmt(pooled['auto'].get('ci_high'))}]  "
        f"k = {pooled['auto'].get('k')}  I² = {fmt(pooled['auto'].get('i2_pct'), 1)}%",
    ]
    for name, path in summary["files"].items():
        lines.append(f"  wrote {name:<14} {path}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="a finished run directory")
    parser.add_argument("--outcome", action="append", default=None,
                        help="outcome key (repeatable; default: every outcome with gold data)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--gold", type=Path, default=GOLD_DIR)
    args = parser.parse_args(argv)

    outcomes = args.outcome or list(GOLD_FOR_OUTCOME)
    for key in outcomes:
        print(report(compare(args.run, key, out_dir=args.out, gold_dir=args.gold)))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
