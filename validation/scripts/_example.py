"""The engine behind the three `example_*.py` scripts.

Each of those scripts answers one question — *where exactly did this number come from?* — for one
extraction route, on one real paper, from a finished run.  They make **no model call**: a run
directory already holds every candidate, verdict, quote, crop and conversion chain, so the
explanation is a read, not a re-extraction.  If the paper is not in the run, the script says so
and prints the exact command that would put it there.

What every example prints:

    the mapper's located source        ("Fig 2A, page 4", with the caption or the quote)
    every candidate, by extractor      (model, variant, the value it read, its own quote)
    the verification verdict           (vote, verifier, adjudication, confidence bucket)
    the resolved values                (mean ± dispersion, n, unit, per group)
    the conversion chain               (every step, from what was printed to Cohen's d)
    the effect size                    (d, variance, SE and CI — computed by canopy.stats)

…and the figure it writes shows the same thing: the page crop with the extractor's own quote
highlighted (or the digitiser's marks on the figure), the chain, and the resulting d with its CI.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._common import OUT_DIR, banner, ci_of, fmt, load_run, write_json  # noqa: E402


# ----------------------------------------------------------------------------- selection
@dataclass
class Cell:
    """One (paper, dataset, outcome) the example explains."""

    record: Any                                   # EffectSizeRecord
    paper: Any                                    # PaperRecord | None
    study: Any                                    # StudyMap | None
    candidates: list[Any] = field(default_factory=list)
    verdicts: list[Any] = field(default_factory=list)

    @property
    def winner(self) -> dict[str, Any]:
        chosen = {v.group: v for v in self.verdicts}
        return {g: chosen.get(g) for g in ("A", "B")}


def paper_sha(pdf: str | Path) -> str:
    from canopy.ingest.dedupe import sha256_of

    return sha256_of(Path(pdf))


def find_cells(run: Any, *, sha256: str = "", outcome_key: str = "") -> list[Cell]:
    cells: list[Cell] = []
    for record in run.records:
        if sha256 and record.paper_id != sha256:
            continue
        if outcome_key and record.outcome_key != outcome_key:
            continue
        cells.append(Cell(record=record, paper=run.papers.get(record.paper_id),
                          study=run.studies.get(record.paper_id),
                          candidates=run.candidates_for(record.dataset_id, record.outcome_key),
                          verdicts=run.verdicts_for(record.dataset_id, record.outcome_key)))
    return cells


def pick(cells: Sequence[Cell], wants: Callable[[Cell], bool]) -> Cell | None:
    """The first cell this example can actually explain, preferring one that was pooled."""
    matches = [c for c in cells if wants(c)]
    matches.sort(key=lambda c: (c.record.confidence == "needs_human", c.record.dataset_id))
    return matches[0] if matches else None


# ----------------------------------------------------------------------------- the printing
def disp(value: Any) -> str:
    """`DispersionType.SD` prints as `DispersionType.SD`; a reader wants `SD`."""
    return str(getattr(value, "value", None) or getattr(value, "name", None) or value)


def _candidate_line(candidate: Any) -> str:
    bits = [f"    {candidate.candidate_id}"]
    if candidate.status != "found":
        bits.append(f"      status: {candidate.status}")
    if candidate.mean is not None:
        bits.append(f"      value:  {candidate.mean} {candidate.unit} "
                    f"({disp(candidate.dispersion_type)} = {candidate.dispersion_value}, "
                    f"n = {candidate.n})")
    if candidate.stat_value is not None:
        bits.append(f"      stat:   {candidate.stat_type} = {candidate.stat_value}, "
                    f"df = {candidate.df or (candidate.df1, candidate.df2)}, "
                    f"design = {candidate.design}, admissible = {candidate.admissible}"
                    + (f" ({candidate.admissible_reason})" if not candidate.admissible else ""))
    if candidate.reported_value is not None:
        bits.append(f"      reported effect: {candidate.reported_scale} = "
                    f"{candidate.reported_value} (standardizer {candidate.standardizer})")
    where = candidate.locator or (f"page {candidate.page}" if candidate.page else "")
    if where:
        bits.append(f"      where:  {where}"
                    + (f" (page {candidate.page})" if candidate.locator and candidate.page else "")
                    + (" [page corrected ±1]" if candidate.page_corrected else ""))
    if candidate.quote:
        bits.append(f"      quote:  “{candidate.quote.strip()[:260]}”")
    if candidate.grounded is not None:
        bits.append(f"      grounded in the PDF text: {candidate.grounded} "
                    f"(similarity {fmt(candidate.grounding_similarity, 2)})")
    if candidate.sigma is not None:
        bits.append(f"      digitisation sigma: {fmt(candidate.sigma)} {candidate.unit}")
    if candidate.notes:
        bits.append(f"      notes:  {candidate.notes[:200]}")
    return "\n".join(bits)


def explain(cell: Cell, run: Any) -> str:
    """The whole story of one cell, as text."""
    record = cell.record
    study = cell.study
    outcome = run.protocol.outcome(record.outcome_key)
    dataset = None
    if study is not None:
        dataset = next((d for d in study.datasets if d.dataset_id == record.dataset_id), None)

    lines: list[str] = []
    citation = record.citation
    lines.append(banner(f"{citation.authors or citation.first_author} ({citation.year}) — "
                        f"{record.dataset_id} · {outcome.label}"))
    if study is not None:
        lines.append(f"  paper      {study.citation.title[:96]}")
    lines.append(f"  file       {record.paper_id[:12]}   route {record.route}   "
                 f"confidence {record.confidence}")

    if dataset is not None:
        lines.append(banner("1 · what the mapper found"))
        lines.append(f"  groups     A = {dataset.group_a.label or 'group A'} "
                     f"(n = {dataset.group_a.n}), B = "
                     f"{dataset.group_b.label or 'group B'} (n = {dataset.group_b.n})")
        if dataset.chosen_pair_rationale:
            lines.append(f"  pair       {dataset.chosen_pair_rationale[:200]}")
        for sources in dataset.outcomes:
            if sources.outcome_key != record.outcome_key:
                continue
            for source in sources.sources:
                lines.append(f"  source     {source.kind}: {source.locator or '—'} "
                             f"(page {source.page})")
                if source.quote:
                    lines.append(f"             “{source.quote.strip()[:200]}”")

    lines.append(banner("2 · what every extractor read"))
    for candidate in sorted(cell.candidates, key=lambda c: (c.group or "", c.candidate_id)):
        lines.append(f"  group {candidate.group} · {candidate.kind} · {candidate.model or 'code'}")
        lines.append(_candidate_line(candidate))

    lines.append(banner("3 · what verification decided"))
    for verdict in sorted(cell.verdicts, key=lambda v: v.group or ""):
        lines.append(f"  group {verdict.group}: vote {verdict.agreement} "
                     f"({verdict.vote_method}, tolerance {fmt(verdict.vote_tolerance, 3)}) · "
                     f"verifier {verdict.verifier_verdict} · "
                     f"{'adjudicated · ' if verdict.adjudicated else ''}"
                     f"confidence {verdict.confidence}")
        if verdict.verifier_reason:
            lines.append(f"      verifier: {verdict.verifier_reason[:220]}")
        if verdict.flags:
            lines.append(f"      flags:    {', '.join(str(f) for f in verdict.flags)}")
        lines.append(f"      resolved: mean {verdict.mean} {verdict.unit}, "
                     f"{disp(verdict.dispersion_type)} {verdict.dispersion_value}, "
                     f"n {verdict.n}")
        if verdict.higher_is_better is not None:
            lines.append(f"      direction: higher is "
                         f"{'better' if verdict.higher_is_better else 'worse'} — "
                         f"{verdict.orientation_evidence[:160]}")

    lines.append(banner("4 · the arithmetic (canopy.stats — no model does this)"))
    for key, value in record.inputs.items():
        lines.append(f"  {key:<12} {value}")
    for step in (record.conversion_steps or [record.conversion_chain]):
        if step:
            lines.append(f"  → {step}")
    if record.routes_rejected:
        for route, why in record.routes_rejected.items():
            lines.append(f"  (route {route} not used: {why})")
    low, high = ci_of(record)
    lines.append(banner("5 · the effect size"))
    lines.append(f"  {record.estimator} = {fmt(record.es)}   var = {fmt(record.var, 4)}   "
                 f"SE = {fmt(record.se)}   95% CI [{fmt(low)}, {fmt(high)}]")
    lines.append(f"  n = {record.n_a} (A, older) vs {record.n_b} (B, younger)   "
                 f"orientation applied: {record.orientation_applied}")
    if record.flags:
        lines.append(f"  flags: {', '.join(record.flags)}")
    if record.not_convertible_reason:
        lines.append(f"  NOT CONVERTIBLE: {record.not_convertible_reason}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the figure
def provenance_images(cell: Cell, out_dir: Path) -> list[tuple[str, Path]]:
    """The pictures that prove it: a highlighted page crop, or the digitiser's overlay."""
    from canopy.report.naming import safe_name
    from canopy.report.provenance import figure_provenance, quote_crop

    paper = cell.paper
    images: list[tuple[str, Path, str]] = []          # caption, file, identity of the SOURCE
    if paper is None:
        return []
    seen: set[str] = set()
    chosen_ids = {i for v in cell.verdicts for i in v.candidate_ids}
    # prefer the candidate the verdict accepted, then one that carries the digitiser's OVERLAY
    # (its marks drawn on the figure) over a bare crop — the overlay is the thing worth showing
    ordered = sorted(cell.candidates,
                     key=lambda c: (c.candidate_id not in chosen_ids, not c.overlay_path,
                                    c.group or "", c.candidate_id))
    for candidate in ordered:
        if candidate.group in seen or candidate.group is None:
            continue
        stem = out_dir / f"{safe_name(candidate.candidate_id)}.png"
        existing = candidate.overlay_path or candidate.crop_path
        figure_id = str((candidate.pixel_provenance or {}).get("figure_id") or "")
        if existing and (Path(paper.out_dir) / existing).exists():
            seen.add(candidate.group)
            images.append((f"group {candidate.group}: {candidate.route or candidate.kind}",
                           Path(paper.out_dir) / existing, str(existing)))
        elif figure_id:
            marks = (candidate.pixel_provenance or {}).get("marks") or []
            result = figure_provenance(paper, figure_id, stem, marks=marks)
            if result["matched"]:
                seen.add(candidate.group)
                images.append((f"group {candidate.group}: {figure_id} "
                               f"(page {result.get('page')})", Path(result["path"]),
                               f"fig:{figure_id}:{bool(marks)}"))
        elif candidate.quote and candidate.page:
            result = quote_crop(paper, int(candidate.page), candidate.quote, stem)
            if result["matched"]:
                seen.add(candidate.group)
                images.append((f"group {candidate.group}: page {candidate.page}, quote "
                               f"highlighted", Path(result["path"]),
                               f"quote:{candidate.page}:{candidate.quote[:80]}"))

    # both groups usually live in ONE figure (or one sentence); showing the same picture twice
    # wastes half the page, so identical sources are collapsed and the caption names both groups
    merged: dict[str, tuple[str, Path]] = {}
    for caption, path, key in images:
        if key in merged:
            existing, kept = merged[key]
            merged[key] = (f"{existing} · {caption.split(':', 1)[0]}", kept)
        else:
            merged[key] = (caption, path)
    return list(merged.values())


def figure(cell: Cell, run: Any, out_stem: Path, *, subtitle: str = "",
           formats: Sequence[str] = ("png", "svg")) -> dict[str, Path]:
    """One picture: where the numbers are printed, the chain, and the resulting d with its CI."""
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg
    from canopy.report.theme import (ACCENT, INK, INK_SECONDARY, MARK, MUTED, figure_style,
                                     save_figure, study_label)

    record = cell.record
    outcome = run.protocol.outcome(record.outcome_key)
    images = provenance_images(cell, out_stem.parent / "example_crops")
    low, high = ci_of(record)

    # the geometry below is computed in inches; `theme` defaults savefig.bbox to "tight", which
    # would re-crop it (and, before the text was wrapped, stretched the page to several metres)
    with figure_style(**{"savefig.bbox": "standard"}):
        n_images = max(1, len(images))
        fig = plt.figure(figsize=(11.0, 2.6 + 2.5 * n_images))
        has_effect = record.es is not None and None not in (low, high)
        # the header is positioned in INCHES: the figure's height depends on how many crops there
        # are, and a fractional y put the subtitle through the title on a short one
        height_in = fig.get_size_inches()[1]
        gs = fig.add_gridspec(n_images + 1, 2, width_ratios=[1.35, 1.0],
                              height_ratios=[*([1.0] * n_images), 0.62 if has_effect else 0.22],
                              left=0.035, right=0.975, top=1 - 0.95 / height_in, bottom=0.05,
                              hspace=0.28, wspace=0.10)

        fig.text(0.035, 1 - 0.30 / height_in, f"{study_label(record)} — {outcome.label}",
                 fontsize=12.5, color=INK, ha="left", va="center")
        fig.text(0.035, 1 - 0.62 / height_in, subtitle or f"route: {record.route}",
                 fontsize=8.6, color=MUTED, ha="left", va="center")

        for index in range(n_images):
            ax = fig.add_subplot(gs[index, 0])
            ax.axis("off")
            if index < len(images):
                caption, path = images[index]
                ax.imshow(mpimg.imread(path))
                ax.set_title(caption, fontsize=8, color=INK_SECONDARY, loc="left", pad=4)
            else:
                ax.text(0.5, 0.5, "no crop could be made for this cell", ha="center",
                        va="center", fontsize=9, color=MUTED)

        ax_text = fig.add_subplot(gs[0:n_images, 1])
        ax_text.axis("off")
        # `theme` sets savefig.bbox="tight", so ONE unwrapped line (a not-convertible chain runs
        # to ~900 characters) would stretch the saved figure to several metres. Wrap everything.
        def wrapped(text: str, indent: str = "      ") -> list[str]:
            return textwrap.wrap(text, width=62, subsequent_indent=indent) or [""]

        body: list[str] = ["what was read"]
        for verdict in sorted(cell.verdicts, key=lambda v: v.group or ""):
            name = "older (A)" if verdict.group == "A" else "younger (B)"
            body.extend(wrapped(f"  {name}: {verdict.mean} {verdict.unit}"
                                f"  {disp(verdict.dispersion_type)} {verdict.dispersion_value}"
                                f"  n = {verdict.n}"))
        statistic = next((c for c in cell.candidates if c.stat_value is not None), None)
        if statistic is not None:
            body.extend(wrapped(f"  test statistic: {statistic.stat_type} = "
                                f"{statistic.stat_value} "
                                f"(df {statistic.df or f'{statistic.df1},{statistic.df2}'}), "
                                f"{statistic.design}"))
        body.append("")
        body.append("how it became an effect size")
        for step in (record.conversion_steps or [record.conversion_chain]):
            if step:
                body.extend(wrapped(f"  → {step}"))
        body.append("")
        body.extend(wrapped(f"confidence: {record.confidence}"
                            + (f"   flags: {', '.join(record.flags)}" if record.flags else "")))
        ax_text.text(0, 1, "\n".join(body), va="top", ha="left", fontsize=8.2, color=INK,
                     family="monospace", linespacing=1.55, transform=ax_text.transAxes)

        ax_es = fig.add_subplot(gs[n_images, :])
        span = max(abs(low or 0), abs(high or 0), 1.0) * 1.15
        ax_es.set_xlim(-span, span)
        ax_es.set_ylim(-1, 1)
        ax_es.axvline(0, color=MARK, lw=0.9)
        if record.es is not None and low is not None and high is not None:
            ax_es.plot([low, high], [0, 0], color=MARK, lw=1.4, solid_capstyle="butt")
            ax_es.plot([low, low], [-0.18, 0.18], color=MARK, lw=1.2)
            ax_es.plot([high, high], [-0.18, 0.18], color=MARK, lw=1.2)
            ax_es.scatter([record.es], [0], s=110, marker="s", color=ACCENT, zorder=3)
            ax_es.text(record.es, 0.42,
                       f"{record.estimator} = {fmt(record.es)}  [{fmt(low)}, {fmt(high)}]",
                       ha="center", fontsize=8.6, color=INK)
        else:
            ax_es.text(0, 0, "no effect size could be computed for this cell", ha="center",
                       va="center", fontsize=9, color=MUTED)
        ax_es.set_yticks([])
        ax_es.tick_params(axis="x", labelsize=7.6, colors=MUTED)
        for side in ("left", "right", "top"):
            ax_es.spines[side].set_visible(False)
        ax_es.set_xlabel(f"← {outcome.negative_direction_label}      "
                         f"{outcome.positive_direction_label} →", fontsize=8, color=MUTED)
        return save_figure(fig, out_stem, formats=formats, bbox_inches=None)


# ----------------------------------------------------------------------------- the CLI shell
def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True,
                        help="a finished run directory (nothing here calls a model)")
    parser.add_argument("--paper", type=Path, default=None,
                        help="the PDF to explain (defaults to whichever paper in the run took "
                             "this route)")
    parser.add_argument("--outcome", default=None, help="outcome key (default: any)")
    parser.add_argument("--dataset", default=None, help="dataset id, when a paper has several")
    parser.add_argument("--protocol", type=Path, default=None,
                        help="a protocol to read the outcome labels and direction labels from; "
                             "by default the run's own `protocol.yaml`, which is the one its "
                             "numbers were produced under")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="where to write the figure")
    parser.add_argument("--name", default=None, help="output file stem")
    return parser


def run_example(args: argparse.Namespace, *, wants: Callable[[Cell], bool], stem: str,
                subtitle: str, missing_hint: str) -> int:
    """Load the run, pick the cell this example is about, print it and draw it."""
    run = load_run(args.run)
    if getattr(args, "protocol", None):
        from canopy.protocol import load_protocol

        run.protocol = load_protocol(args.protocol)
        print(f"(labels and directions read from {args.protocol}, not the run's own protocol)")
    sha = paper_sha(args.paper) if args.paper else ""
    if args.paper and sha not in {s.paper_id for s in run.manifest.papers}:
        print(f"{args.paper.name} is not in {args.run}.\n"
              f"Put it there first — the example never calls a model itself:\n  {missing_hint}")
        return 2

    cells = find_cells(run, sha256=sha, outcome_key=args.outcome or "")
    if args.dataset:
        cells = [c for c in cells if c.record.dataset_id == args.dataset]
    cell = pick(cells, wants)
    if cell is None:
        print(f"no cell in {args.run} took this route"
              + (f" for {args.paper.name}" if args.paper else "")
              + ".\nThe run holds these routes:")
        for record in run.records:
            print(f"  {record.paper_id[:12]}  {record.dataset_id:<6} "
                  f"{record.outcome_key:<16} {record.route}")
        print(f"\n{missing_hint}")
        return 2

    print(explain(cell, run))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = figure(cell, run, out / (args.name or stem), subtitle=subtitle)
    payload = {"run_dir": str(args.run), "paper_id": cell.record.paper_id,
               "dataset_id": cell.record.dataset_id, "outcome_key": cell.record.outcome_key,
               "route": cell.record.route, "confidence": cell.record.confidence,
               "es": cell.record.es, "se": cell.record.se, "ci_low": cell.record.ci_low,
               "ci_high": cell.record.ci_high,
               "conversion_chain": cell.record.conversion_chain,
               "files": {k: str(v) for k, v in files.items()}}
    payload["files"]["json"] = str(write_json(payload, out / f"{args.name or stem}.json"))
    print(banner("6 · files written"))
    for name, path in payload["files"].items():
        print(f"  {name:<6} {path}")
    return 0
