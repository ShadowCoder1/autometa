#!/usr/bin/env python
"""Where every number came from: the share of datasets per extraction route, with real examples.

    .venv/bin/python validation/scripts/extraction_routes_figure.py \
        --run validation/out/run_cisneros_dev

Writes into `validation/out/`:

* `extraction_routes.png|svg` — donut + bars of the route mix over **all** outcomes, with up to
  three real crops per route underneath: a page crop with the extractor's own quote highlighted
  for the printed routes, and the digitiser's overlay (its marks on the figure) for the plotted
  ones.  The examples are pulled out of the run's own provenance, so every thumbnail is a picture
  of a number that is actually in the results;
* `extraction_routes_<outcome>.png|svg` — the same, per outcome;
* `extraction_routes.csv` / `.json` — the counts behind the figure, plus one row per dataset
  naming its route, its confidence and the page it was read from.

This is `canopy.report.methods_fig` used as a library — the report already draws this figure for
a run, and the validation copy exists so the mix can be quoted, filed and compared across runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._common import OUT_DIR, load_run, write_csv, write_json  # noqa: E402


def route_rows(run: Any, records: Sequence[Any]) -> list[dict[str, Any]]:
    """One row per dataset × outcome: which route won, and where it read from."""
    from canopy.report.methods_fig import route_group
    from canopy.report.theme import study_label

    by_cell = {(v.dataset_id, v.outcome_key, v.group): v for v in run.verdicts}
    rows: list[dict[str, Any]] = []
    for record in records:
        pages, locators = [], []
        for group in ("A", "B"):
            verdict = by_cell.get((record.dataset_id, record.outcome_key, group))
            if verdict is None:
                continue
            for candidate in run.candidates_for(record.dataset_id, record.outcome_key):
                if candidate.candidate_id in verdict.candidate_ids:
                    if candidate.page:
                        pages.append(str(candidate.page))
                    if candidate.locator:
                        locators.append(candidate.locator)
        rows.append({
            "study": study_label(record),
            "paper_id": record.paper_id[:12],
            "dataset_id": record.dataset_id,
            "outcome_key": record.outcome_key,
            "route": record.route,
            "route_bucket": route_group(record.route),
            "confidence": record.confidence,
            "pages": ";".join(dict.fromkeys(pages)),
            "locators": ";".join(dict.fromkeys(locators)),
            "conversion_chain": record.conversion_chain,
            "routes_available": ";".join(record.routes_available),
            "d": record.es,
        })
    rows.sort(key=lambda r: (r["outcome_key"], r["route_bucket"], r["study"]))
    return rows


def attempted_routes(run: Any, records: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Which routes the tool actually got a reading from — regardless of which one won.

    `route_counts` answers "where did the number come from", which is blank when a cell was held
    back: every such row is `not_convertible` and the figure says only that. This answers the
    other half of the question the validation asks — *where did the tool look, and what did it
    find there* — so a run that resolved nothing is still informative.
    """
    from canopy.report.methods_fig import route_group

    total = len(records) or 1
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        buckets: set[str] = set()
        for candidate in run.candidates_for(record.dataset_id, record.outcome_key):
            if candidate.status != "found":
                continue
            if candidate.mean is None and candidate.stat_value is None \
                    and candidate.reported_value is None:
                continue
            buckets.add(route_group(candidate.route))
        for name in buckets:
            entry = out.setdefault(name, {"n": 0, "pct": 0.0, "datasets": []})
            entry["n"] += 1
            entry["datasets"].append(f"{record.dataset_id}/{record.outcome_key}")
    for entry in out.values():
        entry["pct"] = 100.0 * entry["n"] / total
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def attempted_figure(counts: Mapping[str, Any], n_datasets: int, out_stem: Path,
                     formats: Sequence[str] = ("png", "svg")) -> dict[str, Path]:
    """A companion bar chart: the share of datasets each route produced a reading for."""
    import matplotlib.pyplot as plt
    from canopy.report.methods_fig import ROUTE_TITLES
    from canopy.report.theme import ACCENT, GRID, INK, MUTED, figure_style, save_figure

    with figure_style():
        names = list(counts) or ["(none)"]
        fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(names) + 1.9))
        values = [counts[n]["pct"] if n in counts else 0.0 for n in names]
        ax.barh(range(len(names)), values, height=0.58, color=ACCENT, zorder=2)
        for index, name in enumerate(names):
            entry = counts.get(name, {"n": 0, "pct": 0.0})
            ax.text(entry["pct"] + 1.5, index, f"{entry['pct']:.0f}%  (n = {entry['n']})",
                    fontsize=8, color=MUTED, ha="left", va="center")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([ROUTE_TITLES.get(n, n) for n in names], fontsize=8.4)
        ax.set_ylim(len(names) - 0.5, -0.6)
        ax.set_xlim(0, 122)
        ax.set_xlabel("share of the datasets this route produced a reading for", fontsize=8.4)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"Where the tool looked, and what it found there  "
                     f"({n_datasets} dataset x outcome cells)", fontsize=10.5, color=INK,
                     loc="left", pad=10)
        fig.text(0.01, 0.012, "A cell can appear in several rows: the routes are alternatives, "
                              "and route precedence decides which one is USED.",
                 fontsize=7.2, color=MUTED)
        fig.subplots_adjust(left=0.26, right=0.98, top=0.86, bottom=0.20)
        return save_figure(fig, out_stem, formats=formats, bbox_inches="tight")


def build(run_dir: str | Path, *, out_dir: str | Path = OUT_DIR, per_outcome: bool = True,
          max_examples: int = 3, formats: Sequence[str] = ("png", "svg")) -> dict[str, Any]:
    from canopy.report.methods_fig import methods_figure, route_counts, route_examples

    run = load_run(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    thumbs = out / "route_examples"
    thumbs.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"run_dir": str(run_dir), "files": {}, "outcomes": {}}

    def one(records: Sequence[Any], stem: str, title: str) -> dict[str, Any]:
        counts = route_counts(records)
        examples = route_examples(records, candidates=run.candidates, verdicts=run.verdicts,
                                  papers=run.papers.values(), out_dir=thumbs,
                                  max_examples=max_examples)
        files: dict[str, Any] = {}
        if records:
            files = methods_figure(records, out / stem, examples=examples, formats=formats,
                                   title=title)
        return {"n_datasets": len(records), "counts": counts,
                "n_examples": {k: len(v) for k, v in examples.items()},
                "files": {k: str(v) for k, v in files.items()}}

    overall = one(run.records, "extraction_routes",
                  "How every value in this review was obtained")
    payload["overall"] = overall
    payload["files"].update(overall["files"])

    if per_outcome:
        for outcome in run.protocol.outcomes:
            records = run.rows_for(outcome.key)
            if not records:
                continue
            entry = one(records, f"extraction_routes_{outcome.key}",
                        f"{outcome.label}: how every value was obtained")
            payload["outcomes"][outcome.key] = entry
            payload["files"].update({f"{outcome.key}_{k}": v for k, v in entry["files"].items()})

    payload["attempted"] = attempted_routes(run, run.records)
    files = attempted_figure(payload["attempted"], len(run.records),
                             out / "extraction_routes_attempted", formats=formats)
    payload["files"].update({f"attempted_{k}": str(v) for k, v in files.items()})

    rows = route_rows(run, run.records)
    for row in rows:
        row["routes_with_a_reading"] = ";".join(
            sorted(n for n, e in payload["attempted"].items()
                   if f"{row['dataset_id']}/{row['outcome_key']}" in e["datasets"]))
    payload["files"]["csv"] = str(write_csv(rows, out / "extraction_routes.csv"))
    payload["files"]["json"] = str(write_json(payload, out / "extraction_routes.json"))
    return payload


def report(payload: dict[str, Any]) -> str:
    lines = ["\nextraction routes", "─" * 72]
    counts = payload["overall"]["counts"]
    for name, entry in counts.items():
        examples = payload["overall"]["n_examples"].get(name, 0)
        lines.append(f"  {name:<16} {entry['n']:>3} datasets  {entry['pct']:>5.1f}%   "
                     f"{examples} example crop(s)   {', '.join(entry['routes'][:3])}")
    if not counts:
        lines.append("  (no rows in this run)")
    lines.append("  routes that produced a reading (regardless of which one was used):")
    for name, entry in payload.get("attempted", {}).items():
        lines.append(f"      {name:<16} {entry['n']:>3} cells  {entry['pct']:>5.1f}%")
    for key, entry in payload["outcomes"].items():
        mix = ", ".join(f"{n} {v['pct']:.0f}%" for n, v in entry["counts"].items())
        lines.append(f"  {key}: {entry['n_datasets']} datasets — {mix}")
    for name, path in payload["files"].items():
        lines.append(f"  wrote {name:<24} {path}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--no-per-outcome", action="store_true")
    args = parser.parse_args(argv)

    print(report(build(args.run, out_dir=args.out, per_outcome=not args.no_per_outcome,
                       max_examples=args.max_examples)))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
