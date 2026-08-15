"""Where the numbers came from — the methods-breakdown figure.

A reader's first question about an automated review is "did it read this out of the text, or did
it measure a picture?", and the honest answer is a distribution, not a sentence. This figure gives
the share of datasets per extraction route and then shows *actual examples*: the page crop with the
quote highlighted, or the digitiser's overlay on the figure it measured. Nothing is illustrative —
every thumbnail is a value that is in the forest plot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..ingest.pdf import PaperRecord
from ..models import Candidate, EffectSizeRecord, Verdict
from . import theme
from .provenance import figure_provenance, quote_crop
from .theme import ACCENT, AXIS, GRID, INK, INK_SECONDARY, MARK, MUTED, figure_style

__all__ = ["methods_figure", "route_counts", "route_examples", "route_group", "ROUTE_ORDER"]

#: the buckets the figure reports, in the order a reader should read them (best evidence first)
ROUTE_ORDER: tuple[str, ...] = ("text", "table", "figure", "test_statistic", "reported_d",
                                "not_convertible", "other")
ROUTE_TITLES: dict[str, str] = {
    "text": "Text (mean ± SD/SE/CI)", "table": "Table cell", "figure": "Figure (digitised)",
    "test_statistic": "Test statistic (t/F/p)", "reported_d": "Effect size as reported",
    "not_convertible": "No usable route", "other": "Other",
}
THUMB_PX = 460


def route_group(route: str) -> str:
    """Which bucket an `EffectSizeRecord.route` belongs to."""
    head = (route or "").split(":", 1)[0]
    if head.startswith(("figure", "digitize")):
        return "figure"
    if head == "table":
        return "table"
    if head.startswith("text") or head == "adjudicated":
        return "text"
    if head in ("test_statistic", "p_value"):
        return "test_statistic"
    if head == "reported_d":
        return "reported_d"
    if head == "not_convertible":
        return "not_convertible"
    return "other"


def route_counts(rows: Sequence[EffectSizeRecord]) -> dict[str, dict[str, Any]]:
    """`{bucket: {n, pct, routes}}` over datasets — buckets with nothing in them are left out."""
    total = len(rows) or 1
    counts: dict[str, dict[str, Any]] = {}
    for record in rows:
        bucket = route_group(record.route)
        entry = counts.setdefault(bucket, {"n": 0, "pct": 0.0, "routes": []})
        entry["n"] += 1
        if record.route and record.route not in entry["routes"]:
            entry["routes"].append(record.route)
    for entry in counts.values():
        entry["pct"] = 100.0 * entry["n"] / total
    return {name: counts[name] for name in ROUTE_ORDER if name in counts}


def _thumbnail(source: Path, target: Path) -> Path | None:
    from PIL import Image

    try:
        image = Image.open(source).convert("RGB")
    except Exception:                                      # pragma: no cover - unreadable asset
        return None
    if image.width > THUMB_PX:
        height = max(1, round(image.height * THUMB_PX / image.width))
        image = image.resize((THUMB_PX, height), Image.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    return target


def route_examples(rows: Sequence[EffectSizeRecord], *, candidates: Sequence[Candidate] = (),
                   verdicts: Sequence[Verdict] = (), papers: Iterable[PaperRecord] = (),
                   out_dir: str | Path = ".", max_examples: int = 3
                   ) -> dict[str, list[Path]]:
    """Up to `max_examples` real thumbnails per route bucket, built from the run's own provenance."""
    by_sha = {p.sha256: p for p in papers}
    chosen_ids = {(v.dataset_id, v.outcome_key): list(v.candidate_ids) for v in verdicts}
    by_cell: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in candidates:
        by_cell.setdefault((candidate.dataset_id, candidate.outcome_key), []).append(candidate)

    directory = Path(out_dir)
    out: dict[str, list[Path]] = {}
    for record in rows:
        bucket = route_group(record.route)
        picked = out.setdefault(bucket, [])
        if len(picked) >= max_examples:
            continue
        cell = (record.dataset_id, record.outcome_key)
        pool = by_cell.get(cell, [])
        preferred = set(chosen_ids.get(cell, []))
        pool = [c for c in pool if c.candidate_id in preferred] or pool
        paper = by_sha.get(record.paper_id) or (next(iter(by_sha.values())) if len(by_sha) == 1
                                                else None)
        if paper is None:
            continue
        for candidate in pool:
            stem = directory / f"{bucket}_{record.dataset_id}_{candidate.candidate_id or 'c'}.png"
            source: Path | None = None
            existing = candidate.overlay_path or candidate.crop_path
            figure_id = str((candidate.pixel_provenance or {}).get("figure_id") or "")
            if existing and (Path(paper.out_dir) / existing).exists():
                source = Path(paper.out_dir) / existing
            elif figure_id:
                result = figure_provenance(paper, figure_id, stem)
                source = Path(result["path"]) if result["matched"] else None
            elif candidate.quote and candidate.page:
                result = quote_crop(paper, int(candidate.page), candidate.quote, stem)
                source = Path(result["path"]) if result["matched"] else None
            if source is None:
                continue
            thumb = _thumbnail(source, directory / f"thumb_{stem.name}")
            if thumb is not None:
                picked.append(thumb)
                break
    return {k: v for k, v in out.items() if v}


def _row_heights(thumb_rows: Sequence[str], examples: Mapping[str, Sequence[Path]],
                 cell_in: float) -> list[float]:
    """Give each thumbnail row the height its own images need, so nothing letterboxes."""
    from PIL import Image

    heights: list[float] = []
    for name in thumb_rows:
        aspect = 0.35
        for path in list(examples.get(name, []))[:3]:
            try:
                with Image.open(path) as image:
                    aspect = max(aspect, image.height / max(image.width, 1))
            except Exception:                              # pragma: no cover - unreadable asset
                continue
        heights.append(min(1.75, max(0.55, aspect * cell_in)) + 0.18)
    return heights


def methods_figure(rows: Sequence[EffectSizeRecord], out_stem: str | Path, *,
                   examples: Mapping[str, Sequence[Path]] | None = None,
                   formats: Sequence[str] = ("png", "svg"),
                   title: str = "How each value was obtained") -> dict[str, Path]:
    """Donut + bars of the route mix, with up to three real examples per route underneath."""
    counts = route_counts(rows)
    examples = {k: [Path(p) for p in v] for k, v in (examples or {}).items() if v}
    if not counts:
        counts = {"other": {"n": 0, "pct": 0.0, "routes": []}}
    thumb_rows = [n for n in counts if examples.get(n)]

    width = 9.0
    label_in = 1.5                       # room for the thumbnail-row labels
    cell_in = (width * 0.96 - label_in) / 3
    row_heights = _row_heights(thumb_rows, examples, cell_in)
    top_in = 3.3
    strip_in = sum(row_heights) + 0.22 * max(0, len(thumb_rows) - 1)
    foot_in = 0.34
    height = top_in + strip_in + foot_in

    with figure_style():
        import matplotlib.pyplot as plt
        from matplotlib import image as mpimg

        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 2.3], wspace=0.30, left=0.02,
                              right=0.985, top=1 - 0.44 / height,
                              bottom=1 - (top_in - 0.55) / height)
        ax_donut = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])

        # a single accent stepped by lightness: the buckets are ordered evidence, not categories
        shades = ["#0d366b", "#1c5cab", ACCENT, "#5598e7", "#86b6ef", "#b7d3f6", "#cde2fb"]
        colours = [shades[min(i, len(shades) - 1)] for i in range(len(counts))]
        values = [counts[n]["n"] for n in counts]
        if sum(values) > 0:
            ax_donut.pie(values, colors=colours, startangle=90, counterclock=False,
                         wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
        ax_donut.text(0, 0, f"{len(rows)}\ndatasets", ha="center", va="center", fontsize=10,
                      color=INK, linespacing=1.3)
        ax_donut.set_aspect("equal")

        for y, (name, colour) in enumerate(zip(counts, colours)):
            entry = counts[name]
            ax_bar.barh(y, entry["pct"], height=0.58, color=colour, edgecolor="none", zorder=2)
            ax_bar.text(entry["pct"] + 1.5, y, f"{entry['pct']:.0f}%  (n = {entry['n']})",
                        fontsize=8, color=MUTED, ha="left", va="center")
        ax_bar.set_yticks(range(len(counts)))
        ax_bar.set_yticklabels([ROUTE_TITLES.get(n, n) for n in counts], fontsize=8.4)
        ax_bar.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
        ax_bar.set_ylim(len(counts) - 0.5, -0.6)
        ax_bar.set_xlim(0, 122)
        ax_bar.set_xticks([0, 25, 50, 75, 100])
        ax_bar.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7.5)
        for spine in ("top", "right", "left"):
            ax_bar.spines[spine].set_visible(False)
        ax_bar.spines["bottom"].set_color(AXIS)
        ax_bar.tick_params(axis="x", length=3, width=0.8, colors=MUTED)
        ax_bar.grid(axis="x", color=GRID, lw=0.6)
        ax_bar.set_axisbelow(True)
        ax_bar.set_xlabel("share of datasets", fontsize=8, color=MUTED, labelpad=2)

        if thumb_rows:
            strip = fig.add_gridspec(len(thumb_rows), 3, left=label_in / width, right=0.985,
                                     top=1 - top_in / height, bottom=foot_in / height,
                                     hspace=0.22 / max(row_heights), wspace=0.05,
                                     height_ratios=row_heights)
            for r, name in enumerate(thumb_rows):
                paths = list(examples.get(name, []))[:3]
                for c in range(3):
                    ax = fig.add_subplot(strip[r, c])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if c < len(paths):
                        for spine in ax.spines.values():
                            spine.set_color(GRID)
                            spine.set_linewidth(0.8)
                        try:
                            ax.imshow(mpimg.imread(paths[c]), aspect="auto")
                        except Exception:                  # pragma: no cover - unreadable asset
                            ax.text(0.5, 0.5, "unavailable", fontsize=7, color=MUTED,
                                    ha="center", va="center", transform=ax.transAxes)
                    else:
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                    if c == 0:
                        ax.set_ylabel(ROUTE_TITLES.get(name, name), fontsize=7.8,
                                      color=INK_SECONDARY, rotation=0, ha="right", va="center",
                                      labelpad=10)

        fig.text(0.02, 1 - 0.14 / height, title, fontsize=11, fontweight="bold", color=INK,
                 ha="left", va="top")
        note = ("Each thumbnail is a value that is in the results: a page crop with the extracted "
                "quote highlighted, or the digitiser's overlay on the figure it measured."
                if thumb_rows else
                "No provenance thumbnails were available for this run.")
        fig.text(0.02, 0.05 / height, note, fontsize=6.8, color=MUTED, ha="left", va="bottom")
        return {k: Path(v) for k, v in theme.save_figure(fig, out_stem, formats,
                                                         bbox_inches=None).items()}
