"""The forest plot (spec §3.5, amendment H).

What this file is careful about, beyond drawing:

* **What is pooled and what is not is visible.** Rows whose confidence is outside
  `settings.primary_analysis_includes` are drawn **hollow**, sit in their own block below the
  pooled diamond, and are counted in the footer. They are never in `pooled`.
* **Where every number came from is on the plot.** Each row carries a route glyph (text, table,
  figure, test statistic, reported) and, when a human replaced a value, an override marker.
* **The conventions are printed.** Estimator, variance formula, τ² method, Hartung–Knapp, the
  prediction-interval method and df, which I² is shown, k datasets from k papers, and how many
  rows were held back — a reader can reproduce the pooled row from the footer alone.
* **Nothing is computed here.** Weights are `1/(vᵢ + τ²)` from the `MetaResult` the caller
  already computed; the pooled estimate, CI and prediction interval are read off it.
* **The square area is AFFINE in the weight, not proportional to it**:
  `MIN_SQUARE + (MAX_SQUARE − MIN_SQUARE)·wᵢ/w_max`. A strictly proportional area makes a
  0.2 %-weight study invisible, which is a worse lie than a floor; the printed `Weight` column
  carries the exact percentage, and this is said here rather than being called proportional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats as sps

from ..models import EffectSizeRecord, OutcomeDef, StatsSettings
from ..stats.meta import MetaResult, prediction_interval
from . import labels, theme
from .theme import (ACCENT, ACCENT_SOFT, AXIS, GRID, INK, INK_SECONDARY, MARK, MUTED,
                    estimator_label, figure_style, fmt, fmt_ci, is_overridden, route_glyph,
                    save_figure, study_label)

__all__ = ["forest_plot", "forest_layout", "ForestLayout", "ForestRow", "MAX_SQUARE", "MIN_SQUARE"]

MAX_SQUARE = 240.0          # marker *area* (pt²) of the heaviest study
MIN_SQUARE = 14.0           # …and the floor, so a 0.2%-weight study is still visible
ROW_HEIGHT = 0.26           # inches per row
_PAD = 0.06                 # fraction of the x-range added at each end


# ----------------------------------------------------------------------------- layout
@dataclass
class ForestRow:
    record: EffectSizeRecord
    y: float = 0.0
    es: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    weight: float = 0.0
    weight_pct: float = 0.0
    glyph: str = ""
    overridden: bool = False
    hollow: bool = False
    #: admitted by the best-guess line rather than by the primary analysis — drawn as a circle,
    #: the same distinction `type.study` makes on the R forest
    guessed: bool = False


@dataclass
class ForestLayout:
    rows: list[ForestRow] = field(default_factory=list)
    excluded: list[ForestRow] = field(default_factory=list)
    pooled: MetaResult | None = None
    pooled_y: float = 0.0
    pi_y: float = 0.0
    pi_low: float = float("nan")
    pi_high: float = float("nan")
    pi_df: float = float("nan")
    xlim: tuple[float, float] = (-1.0, 1.0)
    moderators: list[str] = field(default_factory=list)
    k_papers: int = 0
    k_datasets: int = 0

    @property
    def all_rows(self) -> list[ForestRow]:
        return [*self.rows, *self.excluded]


def _ci(record: EffectSizeRecord, level: float) -> tuple[float | None, float | None]:
    if record.ci_low is not None and record.ci_high is not None:
        return record.ci_low, record.ci_high
    if record.es is None:
        return None, None
    se = record.se
    if se is None and record.var is not None and record.var > 0:
        se = float(np.sqrt(record.var))
    if se is None:
        return None, None
    q = float(sps.norm.ppf(1 - (1 - level) / 2))
    return record.es - q * se, record.es + q * se


def _moderator_names(rows: Sequence[EffectSizeRecord],
                     moderators: Sequence[str] | None) -> list[str]:
    """The protocol's moderator columns, or — when the caller names none — the ones the rows carry.

    First-seen order, never alphabetical: the protocol lists moderators in the order the reviewer
    thinks about them, and the mapper copies that order onto every row.
    """
    if moderators is not None:
        return [str(m) for m in moderators]
    seen: list[str] = []
    for record in rows:
        for key in record.moderators:
            if key not in seen:
                seen.append(key)
    return seen


def forest_layout(rows: Sequence[EffectSizeRecord], pooled: MetaResult, settings: StatsSettings, *,
                  needs_human_rows: Sequence[EffectSizeRecord] = (),
                  moderators: Sequence[str] | None = None,
                  pi: str | None = None) -> ForestLayout:
    """Positions, weights and intervals for every row — the drawing-free half of the plot."""
    level = float(settings.ci_level)
    tau2 = float(pooled.tau2)
    layout = ForestLayout(pooled=pooled,
                          moderators=_moderator_names([*rows, *needs_human_rows], moderators))

    # the printed weight is the POOLER'S weight, keyed by dataset_id over the same predicate the
    # pooler used (`poolable_rows`) — never a formula recomputed here. Under the ordinary pool
    # `weights_raw` IS 1/(vᵢ + τ²), so nothing changes; under a cluster-robust pool it is the
    # CORR working weight, and recomputing 1/(v+τ²) would print weights the estimator never used.
    from .tables import poolable_rows

    weight_of = {record.dataset_id: float(weight)
                 for record, weight in zip(poolable_rows(rows), pooled.weights_raw)}
    prepared: list[ForestRow] = []
    for record in rows:
        if record.es is None or not record.var or record.var <= 0:
            raise ValueError(
                f"row {record.dataset_id!r} has no effect size or no positive variance and so "
                f"cannot have been pooled — pass it in `needs_human_rows` instead")
        low, high = _ci(record, level)
        prepared.append(ForestRow(record=record, es=record.es, ci_low=low, ci_high=high,
                                  weight=weight_of.get(record.dataset_id,
                                                       1.0 / (record.var + tau2)),
                                  glyph=route_glyph(record.route),
                                  overridden=is_overridden(record)))
    total = sum(row.weight for row in prepared) or 1.0
    for row in prepared:
        row.weight_pct = 100.0 * row.weight / total
    prepared.sort(key=lambda r: (r.es if r.es is not None else float("inf"), study_label(r.record)))

    for index, row in enumerate(prepared):
        row.y = -float(index)
    layout.rows = prepared

    y = -(len(prepared) + 0.8)
    layout.pooled_y = y
    method = pi if pi is not None else settings.pi_method
    # under a cluster-robust pool the PI exists from 3 CLUSTERS, not 3 rows
    pi_units = pooled.n_clusters if getattr(pooled, "robust", False) else pooled.k
    if method and pi_units >= 3:
        try:
            low, high, df = prediction_interval(pooled, str(method))
        except ValueError:
            low = high = float("nan")
            df = float("nan")
        layout.pi_low, layout.pi_high, layout.pi_df = low, high, df
        layout.pi_y = y - 0.75

    excluded_top = (layout.pi_y or layout.pooled_y) - 2.0
    for index, record in enumerate(needs_human_rows):
        low, high = _ci(record, level)
        layout.excluded.append(ForestRow(
            record=record, y=excluded_top - index, es=record.es, ci_low=low, ci_high=high,
            weight=0.0, weight_pct=0.0, glyph=route_glyph(record.route),
            overridden=is_overridden(record), hollow=True))

    values: list[float] = []
    for row in layout.all_rows:
        values.extend(v for v in (row.es, row.ci_low, row.ci_high) if v is not None)
    for value in (pooled.ci_low, pooled.ci_high, layout.pi_low, layout.pi_high):
        if value is not None and value == value:
            values.append(float(value))
    lo, hi = (min(values), max(values)) if values else (-1.0, 1.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    layout.xlim = (lo - _PAD * span, hi + _PAD * span)

    everything = [*rows, *needs_human_rows]
    layout.k_datasets = len(rows)
    layout.k_papers = len({r.cluster_id or r.paper_id or r.dataset_id for r in rows})
    if not everything:                                    # pragma: no cover - defensive
        layout.k_papers = 0
    return layout


# ----------------------------------------------------------------------------- drawing
CHAR_IN = 0.055             # inches per character at the row font size (DejaVu Sans 7.6pt)
GAP_CH = 2.2                # blank characters between two columns
BOLD_CH = 1.14              # a bold header character is wider than a plain cell character
MAX_LABEL_CH = labels.MAX_LABEL_CH     # an author string longer than this is elided
MAX_MOD_CH = labels.MAX_MOD_CH         # …and so is a moderator value


@dataclass
class _Column:
    key: str
    header: str
    cells: list[str]
    pad_ch: float = 0.0      # extra room a bold/larger row in this column needs
    width_ch: float = 0.0
    x: float = 0.0           # left edge, as a fraction of its axes


def _wrap_header(name: str, width_ch: float) -> str:
    """`perturbation_size_deg` over two lines rather than across its neighbour's column."""
    words = str(name).replace("_", " ").split()
    lines: list[str] = []
    for word in words:
        if lines and len(lines[-1]) + 1 + len(word) <= max(6, int(width_ch)):
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return "\n".join(lines)


def _measure(columns: list[_Column]) -> float:
    """Give every column the width its widest entry needs; returns the total in characters."""
    total = 0.0
    for column in columns:
        content = max((len(c) for c in column.cells), default=0)
        header = max((len(part) for part in column.header.split("\n")), default=0)
        column.width_ch = float(max(content, header * BOLD_CH)) + GAP_CH + column.pad_ch
        total += column.width_ch
    offset = 0.0
    for column in columns:
        column.x = offset / total if total else 0.0
        offset += column.width_ch
    return total


def _left_labels(layout: ForestLayout, columns: labels.ForestColumns | None) -> list[str]:
    """`[Author, Year, …moderators…, N (A/B)]` — the protocol's, when a caller passed them.

    Without a protocol there are no group labels to take the N column's initials from and no
    prettified moderator names to use, so the plot falls back to the row's own field names. Both
    renderers read the SAME `ForestColumns` whenever one exists.
    """
    default = ["Author", "Year", *(str(m).replace("_", " ") for m in layout.moderators),
               "N (A/B)"]
    if columns is not None and columns.moderators == [str(m) for m in layout.moderators]:
        return list(columns.leftlabs)
    return default


def _left_columns(layout: ForestLayout,
                  spec: labels.ForestColumns | None = None) -> list[_Column]:
    rows = layout.all_rows
    headings = _left_labels(layout, spec)
    columns = [
        _Column("author", headings[0],
                # the marker goes on after the cut (`elide_marked`): appended before it, a label
                # over MAX_LABEL_CH − 2 characters long lost the △ to the ellipsis, which hid the
                # override on precisely the rows with the longest author strings (review finding)
                [labels.elide_marked(study_label(r.record), MAX_LABEL_CH,
                                     theme.OVERRIDE_MARK if r.overridden else "")
                 for r in rows]),
        _Column("year", headings[1], [str(r.record.citation.year or "") for r in rows]),
    ]
    for index, name in enumerate(layout.moderators):
        cells = [labels.elide(str(r.record.moderators.get(name, "") or "\u2014"), MAX_MOD_CH)
                 for r in rows]
        heading = headings[2 + index]
        column = _Column(f"mod:{name}", heading, cells)
        width = max((len(c) for c in cells), default=0)
        column.header = _wrap_header(heading, min(14, max(width, 10)))
        columns.append(column)
    columns.append(_Column("n", headings[-1],
                           [f"{r.record.n_a or '\u2014'}/{r.record.n_b or '\u2014'}" for r in rows]))
    columns.append(_Column("src", "Src", [r.glyph for r in rows]))
    return columns


def _right_columns(layout: ForestLayout, outcome: OutcomeDef, settings: StatsSettings,
                   spec: labels.ForestColumns | None = None) -> list[_Column]:
    """The numbers, in the order a meta-analysis reader expects: estimate, interval, weight.

    The estimate gets its own column headed by what it IS \u2014 `Cohen's d`, `Hedges' g` \u2014 rather
    than by the outcome, which the title already names. One column holding "\u22121.66 [\u22122.60, \u22120.72]"
    reads as a single quantity; the convention every published forest follows (and the one the
    reference review uses) is to print the point estimate and its interval side by side, so a
    reader can scan a column of effect sizes without parsing brackets out of it.
    """
    rows = layout.all_rows
    heads = list(spec.rightlabs) if spec is not None and len(spec.rightlabs) == 3 else [
        estimator_label(settings), f"{settings.ci_level * 100:g}% CI", "Weight"]
    estimate = _Column("effect", heads[0],
                       [fmt(r.es) if r.es is not None else "\u2014" for r in rows]
                       + [fmt(layout.pooled.estimate)],
                       pad_ch=2.5)        # the pooled row is set bold and one point larger
    interval = _Column("ci", heads[1],
                       [fmt_ci(r.ci_low, r.ci_high) if r.es is not None else "\u2014"
                        for r in rows]
                       + [fmt_ci(layout.pooled.ci_low, layout.pooled.ci_high)],
                       pad_ch=2.5)
    weight = _Column("weight", heads[2],
                     [f"{r.weight_pct:.1f}%" for r in layout.rows] + ["100%"])
    return [estimate, interval, weight]


def _draw_row(ax, row: ForestRow, xlim: tuple[float, float], max_weight: float) -> None:
    """One study's interval and mark. A best-guess row is a CIRCLE, as it is on the R forest."""
    lo, hi = xlim
    if row.es is None:
        return
    if row.ci_low is not None and row.ci_high is not None:
        left, right = max(row.ci_low, lo), min(row.ci_high, hi)
        ax.plot([left, right], [row.y, row.y], color=MARK, lw=1.0, solid_capstyle="butt",
                zorder=2)
        for value, drawn, direction in ((row.ci_low, left, -1), (row.ci_high, right, 1)):
            if value is not None and lo <= value <= hi:
                ax.plot([drawn, drawn], [row.y - 0.16, row.y + 0.16], color=MARK, lw=1.0, zorder=2)
            else:                                          # the interval runs off the plot
                ax.annotate("", xy=(drawn + direction * 0.02 * (hi - lo), row.y),
                            xytext=(drawn, row.y),
                            arrowprops=dict(arrowstyle="-|>", color=MARK, lw=1.0,
                                            shrinkA=0, shrinkB=0), zorder=2)
    marker = "o" if row.guessed else "s"
    if row.hollow:
        ax.scatter([row.es], [row.y], s=MIN_SQUARE * 3, marker=marker, facecolors="none",
                   edgecolors=MARK, linewidths=1.0, zorder=3)
        return
    # affine in the weight, not proportional to it: the floor keeps a near-zero-weight study
    # visible, and the `Weight` column prints the exact percentage next to it
    area = MIN_SQUARE + (MAX_SQUARE - MIN_SQUARE) * (row.weight / max_weight if max_weight else 0)
    colour = MUTED if row.guessed else MARK
    ax.scatter([row.es], [row.y], s=area, marker=marker, facecolors=colour, edgecolors=colour,
               linewidths=0.0, zorder=3)


def _draw_diamond(ax, x: float, low: float, high: float, y: float, half_height: float = 0.42,
                  colour: str = ACCENT) -> None:
    ax.fill([low, x, high, x], [y, y + half_height, y, y - half_height], color=colour,
            edgecolor=colour, lw=0.8, zorder=4)


def _text_column(ax, column: _Column, ys, values, *, size: float, colours) -> None:
    for y, text, colour in zip(ys, values, colours):
        ax.text(column.x, y, text, fontsize=size, color=colour, ha="left", va="center",
                transform=ax.get_yaxis_transform())


def forest_plot(rows: Sequence[EffectSizeRecord], pooled: MetaResult, outcome: OutcomeDef,
                settings: StatsSettings, out_stem: str | Path, *,
                needs_human_rows: Sequence[EffectSizeRecord] = (),
                pi: str | None = None, moderators: Sequence[str] | None = None,
                title: str | None = None, formats: Sequence[str] = ("png", "svg", "pdf"),
                subtitle: str = "", columns: labels.ForestColumns | None = None,
                best_guess_ids: Sequence[str] = ()) -> dict[str, Path]:
    """Draw the primary forest plot for one outcome; returns `{format: path}`.

    `rows` are the rows that were pooled (primary analysis); `needs_human_rows` are shown hollow,
    below the pooled diamond, and excluded from it. Direction labels, moderator columns and the
    outcome label all come from the protocol — nothing here knows what is being reviewed. A
    square's AREA is affine in its random-effects weight (see the module docstring); the exact
    weight is printed beside it.

    `columns` (`report.labels.forest_columns`) is the SAME column spec the R renderer draws from,
    so the two plots head their columns identically; without it the headings fall back to the
    rows' own field names. `best_guess_ids` names the rows the second analysis line admitted:
    they are drawn as circles, the distinction `type.study` makes on the R forest.
    """
    layout = forest_layout(rows, pooled, settings, needs_human_rows=needs_human_rows,
                           moderators=moderators, pi=pi)
    all_rows = layout.all_rows
    guessed = set(best_guess_ids)
    for row in all_rows:
        row.guessed = row.record.dataset_id in guessed
    left = _left_columns(layout, columns)
    right = _right_columns(layout, outcome, settings, columns)
    left_ch, right_ch = _measure(left), _measure(right)
    header_lines = max([1] + [c.header.count("\n") + 1 for c in (*left, *right)])

    not_convertible = sum(1 for r in layout.excluded
                          if r.record.route == "not_convertible" or r.record.not_convertible_reason)
    footer = theme.conventions_footer(settings, pooled, k_papers=layout.k_papers,
                                      k_datasets=layout.k_datasets,
                                      n_excluded=len(layout.excluded),
                                      n_not_convertible=not_convertible)
    footer.append(theme.GLYPH_LEGEND([r.record.route for r in all_rows],
                                     overridden=any(r.overridden for r in all_rows)))

    # --- geometry, in inches, so nothing has to fit by luck
    left_in, right_in = left_ch * CHAR_IN, right_ch * CHAR_IN
    plot_in = 3.2
    width = left_in + plot_in + right_in + 0.3
    footer_in = 0.12 + 0.125 * len(footer)
    axis_in = 0.26                                     # x tick labels
    direction_in = 0.24 if (outcome.positive_direction_label
                            or outcome.negative_direction_label) else 0.0
    top_in = 0.34 + (0.16 if subtitle else 0.0) + 0.16 * header_lines
    body_in = ROW_HEIGHT * (len(all_rows) + 4.2)
    height = body_in + top_in + axis_in + direction_in + footer_in

    with figure_style():
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(1, 3, width_ratios=[left_in, plot_in, right_in], wspace=0.0,
                              left=0.012, right=0.995, top=1 - top_in / height,
                              bottom=(footer_in + direction_in + axis_in) / height)
        ax_left = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[0, 2])

        top = 0.9
        bottom = min([r.y for r in all_rows] + [layout.pi_y or layout.pooled_y,
                                                layout.pooled_y]) - 0.9
        for side in (ax_left, ax, ax_right):
            side.set_ylim(bottom, top)
            side.set_yticks([])
        for side in (ax_left, ax_right):
            side.set_xlim(0, 1)
            side.set_xticks([])
            for spine in side.spines.values():
                spine.set_visible(False)
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlim(*layout.xlim)
        ax.tick_params(axis="x", length=3, width=0.8, colors=MUTED, labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

        # --- the plot itself
        ax.axvline(0.0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(pooled.estimate, color=ACCENT, lw=0.8, ls=(0, (4, 3)), alpha=0.6, zorder=1)
        max_weight = max((r.weight for r in layout.rows), default=1.0)
        for row in all_rows:
            _draw_row(ax, row, layout.xlim, max_weight)
        _draw_diamond(ax, pooled.estimate, pooled.ci_low, pooled.ci_high, layout.pooled_y)
        if layout.pi_y and layout.pi_low == layout.pi_low:
            lo, hi = layout.xlim
            pi_left, pi_right = max(layout.pi_low, lo), min(layout.pi_high, hi)
            ax.plot([pi_left, pi_right], [layout.pi_y, layout.pi_y], color=ACCENT_SOFT, lw=2.4,
                    solid_capstyle="butt", zorder=3)
            for edge in (pi_left, pi_right):
                ax.plot([edge, edge], [layout.pi_y - 0.22, layout.pi_y + 0.22], color=ACCENT_SOFT,
                        lw=1.4, zorder=3)

        # --- column headers and cells
        ys = [r.y for r in all_rows]
        colours = [MUTED if r.hollow else INK_SECONDARY for r in all_rows]
        for side, columns in ((ax_left, left), (ax_right, right)):
            for column in columns:
                side.text(column.x, top + 0.12, column.header, fontsize=8.2, fontweight="bold",
                          color=INK, ha="left", va="bottom",
                          transform=side.get_yaxis_transform())
        for column in left:
            _text_column(ax_left, column, ys, column.cells, size=7.6, colours=colours)
        by_key = {column.key: column for column in right}
        for column in right:
            # the estimate and its interval have a cell for every row, held ones included; a
            # weight belongs only to a row that was pooled
            values = (column.cells[:len(all_rows)] if column.key in ("effect", "ci")
                      else column.cells[:len(layout.rows)] + ["\u2014"] * len(layout.excluded))
            _text_column(ax_right, column, ys, values, size=7.6, colours=colours)

        pooled_label = f"Random-effects model (k = {pooled.k})"
        ax_left.text(left[0].x, layout.pooled_y, pooled_label, fontsize=8.2, fontweight="bold",
                     color=INK, ha="left", va="center", transform=ax_left.get_yaxis_transform())
        for key, text in (("effect", fmt(pooled.estimate)),
                          ("ci", fmt_ci(pooled.ci_low, pooled.ci_high)), ("weight", "100%")):
            ax_right.text(by_key[key].x, layout.pooled_y, text, fontsize=8.2, fontweight="bold",
                          color=INK, ha="left", va="center",
                          transform=ax_right.get_yaxis_transform())
        if layout.pi_y:
            ax_left.text(left[0].x, layout.pi_y, "Prediction interval", fontsize=7.6, color=MUTED,
                         ha="left", va="center", transform=ax_left.get_yaxis_transform())
            if layout.pi_low == layout.pi_low:
                # an interval with no point estimate of its own: it goes under the CI column
                ax_right.text(by_key["ci"].x, layout.pi_y, fmt_ci(layout.pi_low, layout.pi_high),
                              fontsize=7.6, color=MUTED, ha="left", va="center",
                              transform=ax_right.get_yaxis_transform())
        if layout.excluded:
            ax_left.text(left[0].x, layout.excluded[0].y + 1.0,
                         f"Held for human review \u2014 not pooled ({len(layout.excluded)})",
                         fontsize=7.8, fontweight="bold", color=MUTED, ha="left", va="center",
                         transform=ax_left.get_yaxis_transform())

        # --- direction labels, from the protocol
        if direction_in:
            axes_in = height * (1 - (footer_in + direction_in + axis_in) / height
                                - top_in / height)
            y = -(axis_in + direction_in * 0.55) / axes_in
            left = f"\u2190 {outcome.negative_direction_label}" \
                if outcome.negative_direction_label else ""
            right = f"{outcome.positive_direction_label} \u2192" \
                if outcome.positive_direction_label else ""
            # These are anchored at the two ends of a 3.2in axis, so a protocol whose direction
            # labels are at all wordy runs them into each other in the middle \u2014 "Reduced under
            # divided attention" against "Greater under divided attention" overprinted to
            # "Gaeatetunder", which is worse than either label alone because it is unreadable
            # rather than merely truncated. Width is estimated rather than measured: a real
            # measurement needs the renderer, which does not exist until draw time, and the
            # estimate only has to decide between three layouts. 0.5em per character is the usual
            # rule for this sans face and errs wide, which is the safe direction here.
            size = 8.0
            while size > 6.0 and (len(left) + len(right) + 2) * 0.5 * size / 72.0 > plot_in:
                size -= 0.5
            stacked = (len(left) + len(right) + 2) * 0.5 * size / 72.0 > plot_in
            # Still colliding at the floor: give each its own line rather than shrink into
            # illegibility. The band already has the height, and a reader loses nothing \u2014 the
            # arrow still points the way the label means.
            line = (size / 72.0) / axes_in if stacked else 0.0
            if left:
                ax.text(0.0, y + line * 0.5, left, fontsize=size,
                        color=MUTED, ha="left", va="center", transform=ax.transAxes)
            if right:
                ax.text(1.0, y - line * 0.5, right, fontsize=size,
                        color=MUTED, ha="right", va="center", transform=ax.transAxes)

        heading = title if title is not None else (outcome.label or outcome.key)
        fig.text(0.012, 1 - 0.16 / height, heading, fontsize=11, fontweight="bold", color=INK,
                 ha="left", va="top")
        if subtitle:
            fig.text(0.012, 1 - 0.34 / height, subtitle, fontsize=8, color=MUTED, ha="left",
                     va="top")
        for i, line in enumerate(footer):
            fig.text(0.012, (0.08 + 0.125 * (len(footer) - 1 - i)) / height, line, fontsize=6.8,
                     color=MUTED, ha="left", va="bottom")

        return {k: Path(v) for k, v in save_figure(fig, out_stem, formats,
                                                   bbox_inches=None).items()}
