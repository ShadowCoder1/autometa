"""The static HTML report and the templated methods paragraph.

Two rules shaped this file.

**No external requests.** The page carries its own CSS and links only to files inside the run
directory, so a run can be zipped, e-mailed, opened offline, or served by the Task-12 server
without any of them behaving differently.

**Everything from a PDF or a model is escaped.** Quotes, titles, filenames and model rationales
all reach this page as text a stranger wrote; `html.escape` runs on every one of them, and links
are built from the run's own relative paths rather than from anything an agent produced.

`methods_paragraph` is a template, not a generation: every number in it is read out of the
manifest, the protocol or a `MetaResult`. No model writes a number a reader might quote.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..models import EffectSizeRecord, Protocol, RunManifest
from ..stats.meta import MetaResult, prediction_interval
from . import theme
from .theme import estimator_label, pi_label, variance_label

__all__ = ["write_html_report", "methods_paragraph", "human_review_table",
           "provenance_table", "REPORT_CSS"]

REPORT_CSS = """
:root { color-scheme: light dark;
  --surface: #ffffff; --panel: #fafaf9; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --accent: #2a78d6; --warn: #ec835a; }
@media (prefers-color-scheme: dark) { :root {
  --surface: #1a1a19; --panel: #232322; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --accent: #3987e5; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1.25rem 4rem; background: var(--surface); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 62rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.4rem 0 .6rem; padding-bottom: .3rem;
  border-bottom: 1px solid var(--grid); }
h3 { font-size: .98rem; margin: 1.4rem 0 .4rem; }
p, li { color: var(--ink-2); }
a { color: var(--accent); }
.sub { color: var(--muted); font-size: .85rem; margin: 0 0 1.5rem; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0 0; }
.card { flex: 1 1 8rem; border: 1px solid var(--grid); border-radius: .5rem; padding: .7rem .85rem;
  background: var(--panel); }
.card .k { display: block; font-size: 1.35rem; font-weight: 650; color: var(--ink); }
.card .l { font-size: .76rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .04em; }
figure { margin: 1rem 0; }
img { max-width: 100%; height: auto; border: 1px solid var(--grid); border-radius: .4rem;
  background: #fff; }
figcaption { font-size: .8rem; color: var(--muted); margin-top: .4rem; }
.files { font-size: .85rem; color: var(--muted); }
.files a { margin-right: .9rem; white-space: nowrap; }
.tablewrap { overflow-x: auto; border: 1px solid var(--grid); border-radius: .5rem; }
table { border-collapse: collapse; width: 100%; font-size: .84rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: .72rem;
  letter-spacing: .04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem; }
pre { background: var(--panel); border: 1px solid var(--grid); border-radius: .5rem;
  padding: .8rem; overflow-x: auto; color: var(--ink-2); white-space: pre-wrap; }
.warn { color: var(--warn); }
"""


def _e(value: Any) -> str:
    """Escape anything that reached us from a PDF, a model or a file name."""
    return _html.escape("" if value is None else str(value), quote=True)


def _rel(path: Any, run_dir: Path) -> str:
    try:
        return Path(path).resolve().relative_to(run_dir.resolve()).as_posix()
    except (ValueError, TypeError):                        # outside the run: link by name only
        return Path(str(path)).name


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _e(value)


# ----------------------------------------------------------------------------- fragments
def _cards(pairs: Sequence[tuple[str, Any]]) -> str:
    cells = "".join(f'<div class="card"><span class="k">{_e(v)}</span>'
                    f'<span class="l">{_e(k)}</span></div>' for k, v in pairs)
    return f'<div class="cards">{cells}</div>'


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]],
           numeric: Sequence[int] = ()) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{_e(v)}</td>' if i in numeric else f"<td>{_e(v)}</td>"
            for i, v in enumerate(row))
        body.append(f"<tr>{cells}</tr>")
    if not body:
        body.append(f'<tr><td colspan="{len(headers)}">nothing to show</td></tr>')
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>")


def human_review_table(queue: Sequence[Mapping[str, Any]]) -> str:
    """The review queue, worst first — sorted by how far the pooled estimate would move."""
    def impact(entry: Mapping[str, Any]) -> float:
        value = entry.get("impact_abs_delta_pooled")
        try:
            return -abs(float(value))
        except (TypeError, ValueError):
            return 1.0                                     # unknown impact sorts last
    ordered = sorted(queue, key=impact)
    rows = []
    for entry in ordered:
        candidates = entry.get("candidates") or []
        summary = "; ".join(
            f"{c.get('value', c.get('mean', '?'))}"
            f"{' (' + str(c.get('route')) + ')' if c.get('route') else ''}"
            for c in candidates if isinstance(c, Mapping)) or "—"
        value = entry.get("impact_abs_delta_pooled")
        rows.append([entry.get("paper_id", ""), entry.get("dataset_id", ""),
                     entry.get("outcome_key", ""), entry.get("group", ""),
                     entry.get("reason", ""), summary,
                     "—" if value is None else _num(value, 3)])
    return _table(["Paper", "Dataset", "Outcome", "Group", "Why", "Candidates", "|Δ pooled|"],
                  rows, numeric=(6,))


def _links(outputs: Mapping[str, Any], run_dir: Path, keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        path = outputs.get(key)
        if path is None:
            continue
        parts.append(f'<a href="{_e(_rel(path, run_dir))}">{_e(key.replace("_", " "))}</a>')
    return f'<p class="files">{"".join(parts)}</p>' if parts else ""


# ----------------------------------------------------------------------------- methods paragraph
def methods_paragraph(manifest: RunManifest, protocol: Protocol,
                      results: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """A Methods section a reader could paste into a paper — every number from the manifest."""
    settings = manifest.settings
    results = results or {}
    papers = manifest.papers
    eligible = [p for p in papers if p.eligible]
    excluded = [p for p in papers if p.eligible is False]
    models = ", ".join(f"{role}: {name}" for role, name in sorted(manifest.models.items())) or "—"

    lines = [
        f"## Methods — {protocol.title}",
        "",
        f"Records were screened and extracted by Canopy {manifest.canopy_version or '(dev)'} "
        f"(commit `{manifest.git_commit or 'unknown'}`) under protocol "
        f"`{Path(manifest.protocol_path).name or 'protocol.yaml'}` "
        f"(sha256 `{manifest.protocol_hash[:12]}`), run `{manifest.run_id}` on "
        f"{manifest.created_at}. Models: {models}.",
        "",
        f"{len(papers)} unique papers entered the pipeline; {len(eligible)} were judged eligible "
        f"against the protocol's criteria and {len(excluded)} were excluded (see "
        f"`exclusions.csv` for the reason and the quote behind each decision). "
        f"{manifest.n_llm_calls} model calls were made at a cost of "
        f"${manifest.cost_usd:.2f}.",
        "",
        "Every reported value was located by one agent, transcribed by two independent "
        "extractors, checked against the paper's own text by an adversarial verifier running a "
        "different model, and adjudicated by a third model only where those disagreed. No model "
        "computed a statistic: all effect sizes, conversions and pooled estimates were calculated "
        "in code from the transcribed values, and every conversion chain is printed in "
        "`extraction_table.csv`.",
        "",
        f"Effect sizes are standardised mean differences ({estimator_label(settings)}) with "
        f"variance by the {variance_label(settings)} convention. Where a paper reported a "
        f"standard error or confidence interval instead of a standard deviation, the SD was "
        f"recovered analytically (ci_to_sd_dist = `{settings.ci_to_sd_dist}`); where it reported "
        f"only a test statistic, conversion was attempted and refused unless the design carried "
        f"the between-group contrast. Route precedence was "
        f"{' > '.join(settings.route_precedence)}.",
        "",
        f"Studies were combined with a random-effects model, τ² estimated by "
        f"{settings.tau2_method}"
        f"{', with the Hartung–Knapp adjustment' if settings.hakn else ' (no Hartung–Knapp '
                                                                      'adjustment)'}, "
        f"and {settings.ci_level * 100:g}% confidence intervals; prediction intervals use the "
        f"{pi_label(settings)} convention. Heterogeneity is reported as Q, I² and τ². "
        f"Rows whose verification confidence was not "
        f"{' or '.join(settings.primary_analysis_includes)} were held for human review and "
        f"excluded from the "
        f"primary analysis; they appear hollow on the forest plots and are listed in "
        f"`human_review_queue.csv`.",
        "",
    ]
    for key, payload in results.items():
        pooled: MetaResult | None = payload.get("pooled")
        rows: Sequence[EffectSizeRecord] = payload.get("rows") or []
        held = payload.get("needs_human_rows") or []
        outcome = protocol.outcome(key) if any(o.key == key for o in protocol.outcomes) else None
        label = outcome.label if outcome is not None else key
        if pooled is None:
            lines.append(f"For **{label}**, no outcome could be pooled "
                         f"({len(held)} rows were held for human review).")
            lines.append("")
            continue
        try:
            pi_low, pi_high, pi_df = prediction_interval(pooled, str(settings.pi_method))
        except ValueError:                                 # pragma: no cover - guarded upstream
            pi_low = pi_high = float("nan")
            pi_df = 0
        papers_k = len({r.cluster_id or r.paper_id or r.dataset_id for r in rows})
        lines.append(
            f"For **{label}**, k = {pooled.k} datasets from {papers_k} papers were pooled "
            f"(a further {len(held)} were held for human review). The pooled "
            f"{estimator_label(settings)} was {pooled.estimate:.2f} "
            f"({settings.ci_level * 100:g}% CI {pooled.ci_low:.2f} to {pooled.ci_high:.2f}; "
            f"{'t' if pooled.hakn else 'z'} = {pooled.z:.2f}, p {theme.fmt_p(pooled.p)}), with "
            f"τ² = {pooled.tau2:.3f}, I² = {100 * pooled.I2:.1f}% and a "
            f"{settings.ci_level * 100:g}% prediction interval of {pi_low:.2f} to {pi_high:.2f}"
            f"{f' (df = {pi_df})' if isinstance(pi_df, int) else ''}. "
            f"Positive values mean “{outcome.positive_direction_label}” and negative values "
            f"“{outcome.negative_direction_label}”." if outcome is not None else "")
        lines.append("")
    lines.append("Sensitivity analyses re-pooled each outcome excluding figure-derived rows, "
                 "excluding rows converted from test statistics, including the rows held for "
                 "review, with and without the Hartung–Knapp adjustment, with and without "
                 "digitisation variance, split by analysis metric, and with one row per paper; "
                 "all are reported in `sensitivity.json`. Small-study effects were examined with "
                 "a funnel plot and Egger's test.")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the report
def provenance_table(entries: Mapping[str, Mapping[str, Any]], run_dir: Path) -> str:
    """One row per value that reached the analysis, linking the image its evidence lives in."""
    rows = []
    for entry in sorted(entries.values(), key=lambda e: (str(e.get("dataset_id", "")),
                                                         str(e.get("outcome_key", "")),
                                                         str(e.get("group", "")))):
        crop = entry.get("crop") or ""
        link = (f'<a href="{_e(_rel(crop, run_dir))}">evidence</a>' if crop
                else _e(entry.get("note", "") or "—"))
        value = entry.get("mean")
        spread = entry.get("dispersion_value")
        printed = "—" if value is None else (
            f"{value}" + (f" ± {spread} {entry.get('dispersion_type', '')}"
                          if spread is not None else ""))
        rows.append((entry.get("dataset_id", ""), entry.get("outcome_key", ""),
                     entry.get("group", ""), printed, entry.get("route", ""),
                     entry.get("page", ""), entry.get("quote", ""), link))
    head = "".join(f"<th>{_e(h)}</th>" for h in
                   ("Dataset", "Outcome", "Group", "Value", "Route", "Page", "Quote", ""))
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row[:-1]) + f"<td>{row[-1]}</td></tr>"
        for row in rows) or '<tr><td colspan="8">nothing to show</td></tr>'
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def write_html_report(run_dir: str | Path, manifest: RunManifest, protocol: Protocol, *,
                      results: Mapping[str, Mapping[str, Any]] | None = None,
                      review_queue: Sequence[Mapping[str, Any]] = (),
                      exclusions: Sequence[Mapping[str, Any]] = (),
                      provenance: Mapping[str, Mapping[str, Any]] | None = None,
                      run_outputs: Mapping[str, Any] | None = None,
                      filename: str = "report.html") -> dict[str, Path]:
    """Write `report.html` and `methods.md` into the run directory; returns both paths."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = results or {}
    run_outputs = dict(run_outputs or {})
    settings = manifest.settings

    methods = methods_paragraph(manifest, protocol, results)
    methods_path = directory / "methods.md"
    methods_path.write_text(methods, encoding="utf-8")

    parts: list[str] = []
    parts.append(f"<h1>{_e(protocol.title)}</h1>")
    parts.append(f'<p class="sub">{_e(protocol.research_question)}</p>')
    parts.append(_cards([
        ("papers", len(manifest.papers)),
        ("eligible", sum(1 for p in manifest.papers if p.eligible)),
        ("outcomes", len(results)),
        ("model calls", manifest.n_llm_calls),
        ("cost", f"${manifest.cost_usd:.2f}"),
        ("held for review", len(review_queue)),
    ]))
    parts.append(f'<p class="files">run <code>{_e(manifest.run_id)}</code> · '
                 f'{_e(manifest.created_at)} · protocol sha256 '
                 f'<code>{_e(manifest.protocol_hash[:12])}</code> · canopy '
                 f'{_e(manifest.canopy_version or "dev")} '
                 f'(<code>{_e(manifest.git_commit or "unknown")}</code>) · profile '
                 f'<code>{_e(settings.profile)}</code></p>')

    for key, payload in results.items():
        outputs = dict(payload.get("outputs") or {})
        pooled: MetaResult | None = payload.get("pooled")
        rows = payload.get("rows") or []
        held = payload.get("needs_human_rows") or []
        outcome = next((o for o in protocol.outcomes if o.key == key), None)
        parts.append(f"<h2>{_e(outcome.label if outcome else key)}</h2>")
        if pooled is not None:
            parts.append(_cards([
                (f"pooled {estimator_label(settings)}", f"{pooled.estimate:.2f}"),
                (f"{settings.ci_level * 100:g}% CI",
                 f"{pooled.ci_low:.2f} to {pooled.ci_high:.2f}"),
                ("k datasets", pooled.k),
                ("k papers", len({r.cluster_id or r.paper_id or r.dataset_id for r in rows})),
                ("I²", f"{100 * pooled.I2:.0f}%"),
                ("τ²", f"{pooled.tau2:.3f}"),
                ("held for review", len(held)),
            ]))
        forest = outputs.get("forest_png")
        if forest is not None:
            parts.append(
                f'<figure><img src="{_e(_rel(forest, directory))}" '
                f'alt="Forest plot for {_e(outcome.label if outcome else key)}">'
                f"<figcaption>Squares are individual datasets, area proportional to their "
                f"random-effects weight; the diamond is the pooled estimate and the bar beneath "
                f"it the prediction interval. Hollow squares were held for human review and are "
                f"not pooled.</figcaption></figure>")
        for name, caption in (("sensitivity_png",
                               "Each panel re-pools this outcome under one changed choice."),
                              ("funnel_png",
                               "Funnel plot with pseudo-confidence contours; Egger's test is in "
                               "the accompanying JSON.")):
            path = outputs.get(name)
            if path is not None:
                parts.append(f'<figure><img src="{_e(_rel(path, directory))}" alt="{_e(name)}">'
                             f"<figcaption>{_e(caption)}</figcaption></figure>")
        parts.append(_links(outputs, directory, [
            "forest_png", "forest_svg", "forest_pdf", "extraction_csv", "extraction_json",
            "extraction_xlsx", "leave_one_out_csv", "sensitivity_json", "sensitivity_png",
            "funnel_png", "funnel_json", "pooled_json"]))

    methods_fig = run_outputs.get("methods_fig_png")
    if methods_fig is not None:
        parts.append("<h2>How each value was obtained</h2>")
        parts.append(f'<figure><img src="{_e(_rel(methods_fig, directory))}" '
                     f'alt="Extraction routes"><figcaption>Share of datasets by extraction '
                     f"route, with real examples: page crops with the extracted quote "
                     f"highlighted, and the digitiser's overlays.</figcaption></figure>")
    prisma = run_outputs.get("prisma_png")
    if prisma is not None:
        parts.append("<h2>Flow of records</h2>")
        parts.append(f'<figure><img src="{_e(_rel(prisma, directory))}" alt="PRISMA-style flow">'
                     f"<figcaption>Files found → unique papers → eligible papers → datasets → "
                     f"datasets included.</figcaption></figure>")

    parts.append("<h2>Held for human review</h2>")
    parts.append("<p>Sorted by how far the pooled estimate would move if the value changed, so "
                 "the top of this list is where a reviewer's time is worth most.</p>")
    parts.append(human_review_table(review_queue))

    if provenance:
        parts.append("<h2>Provenance</h2>")
        parts.append("<p>Every value that reached the analysis, with the page crop its quote is "
                     "highlighted on or the overlay the digitiser measured.</p>")
        parts.append(provenance_table(provenance, directory))
        prov_json = run_outputs.get("provenance_json")
        if prov_json is not None:
            parts.append(f'<p class="files"><a href="{_e(_rel(prov_json, directory))}">'
                         f"provenance.json</a></p>")

    parts.append("<h2>Exclusions</h2>")
    parts.append(_table(
        ["Paper", "File", "Stage", "Reason", "Quote", "Decided by"],
        [[e.get("paper_id", ""), e.get("filename", ""), e.get("stage", ""), e.get("reason", ""),
          e.get("quote", ""), e.get("decider", "")] for e in exclusions]))

    parts.append("<h2>Papers</h2>")
    parts.append(_table(
        ["Paper", "File", "Status", "Eligible", "Cost", "Seconds", "Warnings"],
        [[p.paper_id[:12], p.filename, p.status,
          "—" if p.eligible is None else ("yes" if p.eligible else "no"),
          f"${p.cost_usd:.2f}", f"{p.seconds:.0f}", "; ".join(p.warnings) or "—"]
         for p in manifest.papers], numeric=(4, 5)))

    if manifest.warnings:
        parts.append("<h2>Warnings</h2>")
        parts.append("<ul>" + "".join(f'<li class="warn">{_e(w)}</li>'
                                      for w in manifest.warnings) + "</ul>")

    parts.append("<h2>Methods</h2>")
    parts.append(f'<p class="files"><a href="methods.md">methods.md</a>'
                 f'<a href="manifest.json">manifest.json</a></p>')
    parts.append(f"<pre>{_e(methods)}</pre>")

    page = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{_e(protocol.title)} — Canopy report</title>"
            f"<style>{REPORT_CSS}</style></head><body><main>"
            + "\n".join(parts) +
            "</main></body></html>\n")
    path = directory / filename
    path.write_text(page, encoding="utf-8")
    return {"html": path, "methods": methods_path}
