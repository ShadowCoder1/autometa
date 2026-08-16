#!/usr/bin/env python
"""A figure corpus whose truth is known exactly, and what the digitiser makes of it.

    .venv/bin/python validation/scripts/synthetic_figures.py --out validation/synthetic

Amendment J asks for a synthetic corpus because real papers cannot answer the one question that
matters about a digitiser: *how wrong is it?*  A published figure has no ground truth — the best
anyone can do is compare two humans.  A matplotlib figure has ground truth by construction: the
values are the ones that were plotted.

The corpus varies what actually breaks pixel read-outs — chart form (bars, points, a line series),
dispersion (SD, SE, CI), tick density, font family and size, figure DPI, a log axis, a negative
range, bars narrower than the 8-px flag, and a deliberately noisy/anti-aliased render — and each
case is written as **both** a PNG (the raster the CV routes see) and a one-page PDF (the vector
route's input), so the two deterministic paths are measured on identical geometry.

What is measured here runs with **no model and no credit**:

* route A (vector)   — `canopy.digitize.vector` on the PDF: drawing operators, tick labels from
                       the text layer, whisker ends from the path family;
* route B (raster CV)— `canopy.digitize.cv` on the PNG: axis lines, tick marks, tesseract on the
                       tick labels, least-squares calibration, colour-segmented bars/markers,
                       edge-refined error-bar caps.

Routes C and D (the VLM ones) need a model, so they are **not** measured here; the report says so
rather than quietly reporting a two-route number as if it were the digitiser's accuracy.

Outputs, under `--out` (default `validation/synthetic/`):

    corpus/<case>.png / .pdf / .json     the figures and their exact truth
    accuracy.csv                         one row per (case, route, series): truth, read, error
    accuracy.json                        the per-route summary table
    accuracy.png|svg                     error vs. case, and the sigma-coverage bar
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._common import VALIDATION, fmt, write_csv, write_json  # noqa: E402

DEFAULT_OUT = VALIDATION / "synthetic"


# ============================================================================ the corpus
@dataclass
class Series:
    """One plotted group: its true value and its true error-bar half-length, in data units."""

    label: str
    x: float
    value: float
    error: float
    colour: str


@dataclass
class Case:
    """One synthetic figure and everything that is true about it."""

    name: str
    kind: str                       # bar | point | line
    series: list[Series]
    y_lo: float
    y_hi: float
    ticks: list[float]
    dispersion: str                 # SD | SE | CI
    dpi: int = 200
    font: str = "DejaVu Sans"
    fontsize: float = 10.0
    bar_width: float = 0.55
    log_y: bool = False
    noise: float = 0.0              # gaussian speckle added to the raster only
    note: str = ""
    files: dict[str, str] = field(default_factory=dict)

    @property
    def axis_range(self) -> float:
        return abs(self.y_hi - self.y_lo)


def _series(values: Sequence[tuple[str, float, float]], colours: Sequence[str] | None = None
            ) -> list[Series]:
    palette = list(colours or ("#3b6fb0", "#c1662f", "#4f8a5b", "#8b5aa8"))
    return [Series(label=name, x=float(index), value=value, error=error,
                   colour=palette[index % len(palette)])
            for index, (name, value, error) in enumerate(values)]


def corpus() -> list[Case]:
    """Twelve cases, each chosen because it breaks a different part of a pixel read-out."""
    cases: list[Case] = [
        Case("bars_sd_plain", "bar",
             _series([("young", 12.0, 2.5), ("older", 21.0, 4.0)]),
             0.0, 30.0, [0, 5, 10, 15, 20, 25, 30], "SD",
             note="the ordinary case: two bars, SD whiskers, dense ticks"),
        Case("bars_se_sparse_ticks", "bar",
             _series([("young", 7.5, 0.9), ("older", 11.25, 1.4)]),
             0.0, 15.0, [0, 5, 10, 15], "SE",
             note="four ticks only — the least-squares fit has little to hold on to"),
        Case("bars_ci_four_groups", "bar",
             _series([("y1", 18.0, 3.0), ("y2", 22.5, 3.6), ("o1", 31.0, 5.2),
                      ("o2", 27.75, 4.1)]),
             0.0, 40.0, [0, 10, 20, 30, 40], "CI",
             note="four bars: the nearest-bar match has to pick the right one"),
        Case("bars_negative_range", "bar",
             _series([("young", -6.5, 1.8), ("older", -13.0, 2.9)]),
             -20.0, 5.0, [-20, -15, -10, -5, 0, 5], "SD",
             note="values below zero — a sign error shows up here first"),
        Case("bars_narrow", "bar",
             _series([("a", 9.0, 1.1), ("b", 14.0, 1.6), ("c", 11.0, 1.3), ("d", 16.5, 2.0),
                      ("e", 12.5, 1.4), ("f", 19.0, 2.2)]),
             0.0, 25.0, [0, 5, 10, 15, 20, 25], "SD", bar_width=0.16,
             note="bars under the 8-px flag: the digitiser should say so, not guess"),
        Case("points_sd", "point",
             _series([("young", 15.5, 2.2), ("older", 24.5, 3.4)]),
             0.0, 35.0, [0, 5, 10, 15, 20, 25, 30, 35], "SD",
             note="markers instead of bars — detection is by colour blob, not by rectangle"),
        Case("points_small_font", "point",
             _series([("young", 4.2, 0.6), ("older", 6.9, 0.9)]),
             0.0, 10.0, [0, 2, 4, 6, 8, 10], "SE", fontsize=6.5,
             note="6.5 pt tick labels — the OCR's hardest input"),
        Case("points_serif_font", "point",
             _series([("young", 33.0, 4.5), ("older", 47.0, 6.1)]),
             0.0, 60.0, [0, 10, 20, 30, 40, 50, 60], "CI", font="DejaVu Serif",
             note="a serif face, because tick OCR is font-sensitive"),
        Case("points_low_dpi", "point",
             _series([("young", 2.4, 0.35), ("older", 3.8, 0.5)]),
             0.0, 5.0, [0, 1, 2, 3, 4, 5], "SD", dpi=96,
             note="96 dpi: every pixel is worth more data units"),
        Case("points_high_dpi", "point",
             _series([("young", 2.4, 0.35), ("older", 3.8, 0.5)]),
             0.0, 5.0, [0, 1, 2, 3, 4, 5], "SD", dpi=400,
             note="400 dpi: the accuracy ceiling of the same figure"),
        Case("points_noisy_scan", "point",
             _series([("young", 19.0, 2.6), ("older", 28.5, 3.9)]),
             0.0, 40.0, [0, 10, 20, 30, 40], "SD", noise=9.0,
             note="speckle over the raster, the way a scanned page arrives"),
        Case("line_timeseries_endpoint", "line",
             _series([("young", 8.0, 1.2), ("older", 15.0, 2.3)], colours=("#3b6fb0", "#c1662f")),
             0.0, 25.0, [0, 5, 10, 15, 20, 25], "SD",
             note="a time series whose LAST point is the outcome (the late-window rule)"),
    ]
    for case in cases:                       # the line case needs its series far apart in x
        if case.kind == "line":
            for index, item in enumerate(case.series):
                item.x = float(index * 3)
    return cases


def select_marks(marks: Sequence[Any], case: Case, x_of: Any) -> tuple[list[Any], str]:
    """Pair detected marks to the plotted series, or say why that could not be done.

    Detection failure and read-out error are different things, and mixing them makes both
    numbers meaningless: a harness that pairs two series to one detected blob reports a huge
    "read-out error" that is really a missed detection.  So a case whose mark count does not
    match the plot is reported as `mark_count_mismatch` and contributes to the read RATE, never
    to the read-out MAE.

    For the time-series case the marks are split into one cluster per series at the largest
    gaps in x, and the outcome is each cluster's RIGHTMOST mark — the late-window rule.
    """
    ordered = sorted(marks, key=x_of)
    n = len(case.series)
    if case.kind != "line":
        if len(ordered) != n:
            return [], f"mark_count_mismatch:{len(ordered)}_detected_{n}_plotted"
        return ordered, "read"
    if len(ordered) < n:
        return [], f"mark_count_mismatch:{len(ordered)}_detected_at_least_{n}_needed"
    gaps = sorted(range(1, len(ordered)),
                  key=lambda i: x_of(ordered[i]) - x_of(ordered[i - 1]), reverse=True)
    cuts = sorted(gaps[:n - 1])
    groups, start = [], 0
    for cut in [*cuts, len(ordered)]:
        groups.append(ordered[start:cut])
        start = cut
    if len(groups) != n or any(not g for g in groups):
        return [], f"mark_count_mismatch:{len(ordered)}_detected_could_not_cluster"
    return [group[-1] for group in groups], "read"


def draw(case: Case, out_dir: Path) -> Case:
    """Render one case to PNG and PDF and write its truth beside them."""
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"font.family": case.font, "font.size": case.fontsize,
                                "axes.linewidth": 1.0, "figure.facecolor": "white",
                                "savefig.facecolor": "white", "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        if case.kind == "bar":
            for item in case.series:
                ax.bar(item.x, item.value, width=case.bar_width, color=item.colour,
                       edgecolor="none")
                ax.errorbar(item.x, item.value, yerr=item.error, fmt="none", ecolor="#222222",
                            elinewidth=1.1, capsize=4.5, capthick=1.1)
        elif case.kind == "point":
            for item in case.series:
                ax.errorbar(item.x, item.value, yerr=item.error, fmt="o", color=item.colour,
                            markersize=7.5, ecolor="#222222", elinewidth=1.1, capsize=4.5,
                            capthick=1.1)
        else:                                       # a time series: the endpoint is the outcome
            trials = np.arange(0, 11)
            for item in case.series:
                start = item.value * 2.4
                curve = item.value + (start - item.value) * np.exp(-trials / 2.6)
                curve[-1] = item.value              # the last point IS the truth, exactly
                xs = item.x + trials * 0.1
                ax.plot(xs, curve, "-o", color=item.colour, markersize=4.5, lw=1.3)
                ax.errorbar(xs[-1], item.value, yerr=item.error, fmt="none", ecolor="#222222",
                            elinewidth=1.1, capsize=4.5, capthick=1.1)
        ax.set_ylim(case.y_lo, case.y_hi)
        ax.set_yticks(case.ticks)
        right = max(s.x for s in case.series) + (1.4 if case.kind == "line" else 0.95)
        ax.set_xlim(-0.75, right)
        ax.set_xticks([s.x for s in case.series])
        ax.set_xticklabels([s.label for s in case.series])
        ax.set_ylabel("outcome (deg)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        png = out_dir / f"{case.name}.png"
        pdf = out_dir / f"{case.name}.pdf"
        fig.savefig(png, dpi=case.dpi)
        fig.savefig(pdf)                            # vector, for route A
        plt.close(fig)

    if case.noise:
        from PIL import Image

        rng = np.random.default_rng(11)
        image = np.asarray(Image.open(png).convert("RGB")).astype(np.float32)
        image += rng.normal(0.0, case.noise, image.shape)
        Image.fromarray(np.clip(image, 0, 255).astype("uint8")).save(png)

    case.files = {"png": str(png), "pdf": str(pdf)}
    (out_dir / f"{case.name}.json").write_text(json.dumps(asdict(case), indent=2) + "\n")
    return case


# ============================================================================ route B: raster CV
def read_raster(case: Case) -> list[dict[str, Any]]:
    """Route B, end to end, with no model: axes → ticks → OCR → fit → marks → caps → values."""
    from canopy.digitize.calibrate import fit_axis, pair_ticks, pixel_resolution, px_to_value
    from canopy.digitize.cv import (detect_bars, detect_markers, find_axes, find_cap_ends,
                                    find_tick_marks, load_color, load_gray, ocr_tick_labels)

    png = Path(case.files["png"])
    gray, colour = load_gray(png), load_color(png)
    axes = find_axes(gray)
    rows = find_tick_marks(gray, axes).get("left", []) if axes.y_axis_x is not None else []
    labels = ocr_tick_labels(gray, axes, side="left", ticks=rows or None)
    pairs = pair_ticks(list(labels), rows, axis="y")
    out: list[dict[str, Any]] = []
    base = {"case": case.name, "route": "B_raster_cv", "kind": case.kind,
            "axis_range": case.axis_range, "dpi": case.dpi,
            "n_ticks_detected": len(rows), "n_ticks_ocr": len(list(labels)),
            "ocr_status": str(getattr(labels, "status", "missing")),
            "n_ticks_paired": len(pairs)}
    if len(pairs) < 2:
        return [{**base, "series": s.label, "status": "no_calibration",
                 "truth": s.value, "read": None, "error": None} for s in case.series]
    cal = fit_axis(pairs, axis="y")
    # a least-squares fit through mutually contradictory ticks still returns a calibration; its
    # residual is the only thing that says so, and nothing downstream looks at it
    base.update({"cal_rmse_px": cal.rmse, "cal_ticks": [v for _, v in cal.ticks],
                 "ocr_texts": [l.text for l in labels],
                 "implausible_calibration": bool(cal.rmse > 2.0)})

    # `digitizer._cv_core` tries bars first and only falls back to markers when it finds none.
    # Here the chart form is KNOWN, so the right detector is used and the precedence hazard is
    # recorded separately (`n_bars_detected` on a point chart is a spurious bar that would stop
    # production ever reaching `detect_markers` — see validation/README.md).
    bars = detect_bars(colour, axes)
    if case.kind == "bar":
        marks: list[Any] = bars
        kind = "bar"
    else:
        marks = detect_markers(colour, axes)
        kind = "marker"
    span = abs(axes.plot_bbox[3] - axes.plot_bbox[1]) or float(gray.shape[0])
    base["n_marks_detected"] = len(marks)
    base["n_bars_detected"] = len(bars)
    base["cv_core_would_use"] = "bars" if bars else "markers"

    ordered, status = select_marks(marks, case,
                                   (lambda m: m.x_center) if kind == "bar" else (lambda m: m.x))
    if status != "read":
        return [{**base, "series": s.label, "status": status, "truth": s.value,
                 "truth_error": s.error, "read": None, "error": None} for s in case.series]

    for index, item in enumerate(case.series):
        row = {**base, "series": item.label, "truth": item.value, "truth_error": item.error}
        mark = ordered[index]
        x = mark.x_center if kind == "bar" else mark.x
        y = mark.top_y if kind == "bar" else mark.y
        value = px_to_value(cal, y)
        top, bottom = find_cap_ends(gray, x, y, max_len_px=span)
        half = None
        if top is not None and bottom is not None:
            half = abs(px_to_value(cal, top) - px_to_value(cal, bottom)) / 2.0
        elif top is not None:
            half = abs(px_to_value(cal, top) - value)
        sigma = pixel_resolution(cal, y)
        out.append({**row, "status": "read", "read": value, "error": value - item.value,
                    "abs_pct_of_range": 100.0 * abs(value - item.value) / case.axis_range,
                    "read_error_bar": half,
                    "error_bar_error": None if half is None else half - item.error,
                    "sigma": sigma,
                    "covered": None if sigma in (None, 0)
                    else bool(abs(value - item.value) <= 1.96 * sigma),
                    "narrow_bar": bool(getattr(mark, "narrow", False))})
    return out


# ============================================================================ route A: vector
def read_vector(case: Case, work_dir: Path) -> list[dict[str, Any]]:
    """Route A: the PDF's own drawing operators, through `canopy.digitize.vector`."""
    from canopy.digitize.calibrate import px_to_value
    from canopy.digitize.vector import calibrate_from_scene, vector_candidates, whisker_ends
    from canopy.ingest.pdf import ingest_pdf

    base = {"case": case.name, "route": "A_vector", "kind": case.kind,
            "axis_range": case.axis_range, "dpi": case.dpi}
    try:
        paper = ingest_pdf(Path(case.files["pdf"]), work_dir / case.name)
    except Exception as exc:                          # pragma: no cover - defensive
        return [{**base, "series": s.label, "status": f"ingest_failed: {exc}",
                 "truth": s.value, "read": None, "error": None} for s in case.series]
    if not paper.figures:
        return [{**base, "series": s.label, "status": "no_figure_region",
                 "truth": s.value, "read": None, "error": None} for s in case.series]

    figure = paper.figures[0]
    scene = vector_candidates(paper, figure)
    cal = calibrate_from_scene(scene, axis="y")
    base.update({"n_marks": len(scene.marks), "n_whiskers": len(scene.whiskers),
                 "n_tick_lines": len(scene.tick_lines), "n_warnings": len(scene.warnings)})
    if cal is None:
        return [{**base, "series": s.label, "status": "no_calibration",
                 "truth": s.value, "read": None, "error": None} for s in case.series]

    marks, status = select_marks(scene.marks, case, lambda m: m.x)
    if status != "read":
        return [{**base, "series": s.label, "status": status, "truth": s.value,
                 "truth_error": s.error, "read": None, "error": None} for s in case.series]

    out: list[dict[str, Any]] = []
    for index, item in enumerate(case.series):
        row = {**base, "series": item.label, "truth": item.value, "truth_error": item.error}
        mark = marks[index]
        # a `Mark` is the shape's CENTRE; a bar's datum is its free end.  This corpus draws bars
        # from zero, so the free end is whichever edge is farther from zero.
        y_px = mark.y
        if str(mark.kind) == "rect" and case.kind == "bar":
            edges = [mark.y - mark.h / 2.0, mark.y + mark.h / 2.0]
            y_px = max(edges, key=lambda p: abs(px_to_value(cal, p)))
        value = px_to_value(cal, y_px)
        ends = whisker_ends(scene, mark, axis="y")
        half = None
        if ends and all(e is not None for e in ends):
            half = abs(px_to_value(cal, ends[0]) - px_to_value(cal, ends[1])) / 2.0
        out.append({**row, "status": "read", "read": value, "error": value - item.value,
                    "abs_pct_of_range": 100.0 * abs(value - item.value) / case.axis_range,
                    "read_error_bar": half,
                    "error_bar_error": None if half is None else half - item.error,
                    "sigma": None, "covered": None})
    return out


# ============================================================================ the summary
def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route in sorted({r["route"] for r in rows}):
        mine = [r for r in rows if r["route"] == route]
        read = [r for r in mine if r.get("read") is not None]
        pct = [r["abs_pct_of_range"] for r in read if r.get("abs_pct_of_range") is not None]
        bars = [abs(r["error_bar_error"]) / r["axis_range"] * 100.0 for r in read
                if r.get("error_bar_error") is not None]
        covered = [r["covered"] for r in read if r.get("covered") is not None]
        out[route] = {
            "n_series": len(mine),
            "n_read": len(read),
            "read_rate": len(read) / len(mine) if mine else None,
            "mae_pct_of_axis_range": sum(pct) / len(pct) if pct else None,
            "median_pct_of_axis_range": sorted(pct)[len(pct) // 2] if pct else None,
            "max_pct_of_axis_range": max(pct) if pct else None,
            "within_2pct_of_range": sum(1 for p in pct if p <= 2.0) / len(pct) if pct else None,
            "error_bar_mae_pct_of_range": sum(bars) / len(bars) if bars else None,
            "n_error_bars_read": len(bars),
            "sigma_coverage_95": (sum(1 for c in covered if c) / len(covered)
                                  if covered else None),
            "n_sigma_checked": len(covered),
            "failures": sorted({r["status"] for r in mine if r.get("read") is None}),
            "cases_with_implausible_calibration": sorted(
                {r["case"] for r in mine if r.get("implausible_calibration")}),
            "cases_where_cv_core_would_pick_bars_on_a_point_chart": sorted(
                {r["case"] for r in mine
                 if r.get("kind") in ("point", "line") and r.get("n_bars_detected")}),
        }
    out["_note"] = ("routes C (VLM coordinates) and D (VLM read-out) are NOT measured here — "
                    "they need a model.  Run them on this corpus once credit allows; see "
                    "validation/README.md.")
    return out


def plot(rows: Sequence[dict[str, Any]], summary: dict[str, Any], out_stem: Path,
         formats: Sequence[str] = ("png", "svg")) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    from canopy.report.theme import (ACCENT, GRID, INK, INK_SECONDARY, MUTED, figure_style,
                                     save_figure)

    routes = sorted({r["route"] for r in rows})
    cases = list(dict.fromkeys(r["case"] for r in rows))
    with figure_style():
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.0, 7.4),
                                      gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.55})
        markers = {"A_vector": "s", "B_raster_cv": "o"}
        for route in routes:
            xs, ys = [], []
            for index, case in enumerate(cases):
                for row in rows:
                    if row["case"] != case or row["route"] != route:
                        continue
                    if row.get("abs_pct_of_range") is None:
                        continue
                    xs.append(index + (0.16 if route == "B_raster_cv" else -0.16))
                    ys.append(row["abs_pct_of_range"])
            ax.scatter(xs, ys, s=34, marker=markers.get(route, "P"),
                       facecolors="none" if route == "A_vector" else ACCENT,
                       edgecolors=ACCENT, linewidths=1.2, label=route, zorder=3)
        ax.axhline(2.0, color=INK_SECONDARY, lw=1.0, ls="--", zorder=2)
        ax.text(len(cases) - 0.4, 2.15, "2 % of axis range (the vote tolerance)", fontsize=7.4,
                color=INK_SECONDARY, ha="right")
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=7.4)
        ax.set_ylabel("absolute read-out error\n(% of axis range)", fontsize=8.6)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(bottom=0)                      # an ABSOLUTE error has no negative half
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.6, frameon=False, loc="upper left")
        ax.set_title("Digitiser accuracy on figures whose values are known exactly",
                     fontsize=11, color=INK, pad=8)

        names, values = [], []
        for route in routes:
            entry = summary[route]
            names += [f"{route}\nread rate", f"{route}\nwithin 2 %"]
            values += [100.0 * (entry["read_rate"] or 0.0),
                       100.0 * (entry["within_2pct_of_range"] or 0.0)]
            if entry.get("sigma_coverage_95") is not None:
                names.append(f"{route}\nsigma covers truth")
                values.append(100.0 * entry["sigma_coverage_95"])
        ax2.bar(range(len(values)), values, color=ACCENT, width=0.55)
        for index, value in enumerate(values):
            ax2.text(index, value + 2, f"{value:.0f}%", ha="center", fontsize=7.6, color=MUTED)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, fontsize=7.0)
        ax2.set_ylim(0, 112)
        ax2.set_ylabel("%", fontsize=8.6)
        ax2.grid(axis="y", color=GRID, lw=0.6)
        ax2.set_axisbelow(True)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.14)
        return save_figure(fig, out_stem, formats=formats, bbox_inches="tight")


def build(out_dir: str | Path = DEFAULT_OUT, *, only: Sequence[str] = (),
          routes: Sequence[str] = ("A", "B")) -> dict[str, Any]:
    out = Path(out_dir)
    corpus_dir = out / "corpus"
    work = out / "_ingest"
    cases = [c for c in corpus() if not only or c.name in only]
    rows: list[dict[str, Any]] = []
    for case in cases:
        draw(case, corpus_dir)
        if "B" in routes:
            rows.extend(read_raster(case))
        if "A" in routes:
            rows.extend(read_vector(case, work))

    summary = summarise(rows)
    summary["n_cases"] = len(cases)
    summary["corpus_dir"] = str(corpus_dir)
    files = {"csv": str(write_csv(rows, out / "accuracy.csv"))}
    if any(r.get("abs_pct_of_range") is not None for r in rows):
        files.update({k: str(v) for k, v in plot(rows, summary, out / "accuracy").items()})
    summary["files"] = files
    files["json"] = str(write_json(summary, out / "accuracy.json"))
    return {"summary": summary, "rows": rows}


def report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = ["\nsynthetic digitiser accuracy", "─" * 72,
             f"  {summary['n_cases']} case(s), {len(result['rows'])} series read-outs"]
    for route, entry in summary.items():
        if not isinstance(entry, dict) or "read_rate" not in entry:
            continue
        lines.append(f"  {route}")
        lines.append(f"      read rate            {entry['n_read']}/{entry['n_series']} "
                     f"({fmt(entry['read_rate'], 2)})")
        lines.append(f"      MAE (% axis range)   {fmt(entry['mae_pct_of_axis_range'], 2)}   "
                     f"median {fmt(entry['median_pct_of_axis_range'], 2)}   "
                     f"max {fmt(entry['max_pct_of_axis_range'], 2)}")
        lines.append(f"      within 2 % of range  {fmt(entry['within_2pct_of_range'], 2)}")
        lines.append(f"      error-bar MAE (%)    {fmt(entry['error_bar_mae_pct_of_range'], 2)} "
                     f"({entry['n_error_bars_read']} read)")
        if entry["sigma_coverage_95"] is not None:
            lines.append(f"      sigma covers truth   {fmt(entry['sigma_coverage_95'], 2)} "
                         f"({entry['n_sigma_checked']} checked)")
        if entry["failures"]:
            lines.append(f"      failures             {', '.join(entry['failures'])}")
    lines.append(f"  {summary['_note']}")
    for name, path in summary["files"].items():
        lines.append(f"  wrote {name:<6} {path}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case", action="append", default=None, help="only these cases")
    parser.add_argument("--routes", default="A,B", help="which offline routes to measure")
    args = parser.parse_args(argv)

    print(report(build(args.out, only=args.case or (),
                       routes=tuple(r.strip() for r in args.routes.split(",") if r.strip()))))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
