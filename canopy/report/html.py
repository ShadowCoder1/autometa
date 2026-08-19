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
from .conclusion import Conclusion, conclusion_from_payload, overall_conclusion, render_text
from .forest_render import caption_line, methods_line
from .naming import url_path
from .theme import estimator_label, pi_label, variance_label

__all__ = ["write_html_report", "methods_paragraph", "human_review_table",
           "provenance_table", "forest_caption", "REPORT_CSS"]

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
.conclusion { border-left: 2px solid var(--grid); padding-left: .8rem; }
.callout { border-left: 3px solid #c0392b; background: rgba(192, 57, 43, .07); color: #c0392b;
  padding: .6rem .8rem; border-radius: .3rem; font-size: .85rem; margin: .8rem 0; }
"""


def _e(value: Any) -> str:
    """Escape anything that reached us from a PDF, a model or a file name."""
    return _html.escape("" if value is None else str(value), quote=True)


def _rel(path: Any, run_dir: Path) -> str:
    try:
        return Path(path).resolve().relative_to(run_dir.resolve()).as_posix()
    except (ValueError, TypeError):                        # outside the run: link by name only
        return Path(str(path)).name


def _link(path: Any, run_dir: Path) -> str:
    """A run-relative path as an `href`/`src` value: percent-encoded, then HTML-escaped.

    Both steps are needed and they are not the same step. Encoding is what makes a name that
    contains `#`, `%` or a space reach its file at all (a browser truncates a URL at `#`);
    escaping is what stops a file name from closing the attribute.
    """
    return _e(url_path(_rel(path, run_dir)))


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
        margin = entry.get("confidence_margin")
        boundary = entry.get("nearest_boundary") or ""
        rows.append([entry.get("paper_id", ""), entry.get("dataset_id", ""),
                     entry.get("outcome_key", ""), entry.get("group", ""),
                     entry.get("reason", ""), summary,
                     # C11: "0.0000 from accept_with_note" is the difference between a cell the
                     # evidence held back and one the rounding did.
                     "—" if margin is None
                     else f"{_num(margin, 4)}{' from ' + boundary if boundary else ''}",
                     "—" if value is None else _num(value, 3)])
    return _table(["Paper", "Dataset", "Outcome", "Group", "Why", "Candidates", "Margin",
                   "|Δ pooled|"], rows, numeric=(7,))


def _links(outputs: Mapping[str, Any], run_dir: Path, keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        path = outputs.get(key)
        if path is None:
            continue
        parts.append(f'<a href="{_link(path, run_dir)}">{_e(key.replace("_", " "))}</a>')
    return f'<p class="files">{"".join(parts)}</p>' if parts else ""


# ----------------------------------------------------------------------------- methods paragraph
def methods_paragraph(manifest: RunManifest, protocol: Protocol,
                      results: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """A Methods section a reader could paste into a paper — every number from the manifest."""
    settings = manifest.settings
    results = results or {}
    papers = manifest.papers
    eligible = [p for p in papers if p.eligible]
    excluded = [p for p in papers if not p.eligible]       # `None` (never mapped) is not eligible
    # a paper is "contributing" when at least one of its rows — pooled or held — exists; an
    # eligible paper with none is a paper the review lost, and saying "0 were excluded" beside it
    # is what `runs/proof/methods.md` did while a whole paper was missing from every table
    with_rows = {r.paper_id for payload in results.values()
                 for r in [*(payload.get("rows") or []), *(payload.get("needs_human_rows") or [])]}
    lost = [p for p in eligible if p.paper_id not in with_rows]
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
        f"against the protocol's criteria and {len(excluded)} were excluded"
        + (f"; a further {len(lost)} eligible paper{'s' if len(lost) != 1 else ''} yielded no "
           f"usable row and contribute{'s' if len(lost) == 1 else ''} nothing to the pooled "
           f"estimate" if lost else "")
        + f" (see `exclusions.csv` for the reason and the quote behind each decision). "
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
    # DECISION F: which renderer actually drew the figures, and the version of everything in it
    drawn = methods_line(next(iter(_renderers(results).values()), None))
    if drawn:
        lines.extend([drawn, ""])
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
        # the pooled sentence is written whether or not the protocol still carries this outcome
        # key (a run can be re-pooled against an edited protocol); only the direction sentence
        # depends on the outcome definition, so only that one is conditional
        sentence = (
            f"For **{label}**, k = {pooled.k} datasets from {papers_k} papers were pooled "
            f"(a further {len(held)} were held for human review). The pooled "
            f"{estimator_label(settings)} was {pooled.estimate:.2f} "
            f"({settings.ci_level * 100:g}% CI {pooled.ci_low:.2f} to {pooled.ci_high:.2f}; "
            f"{'t' if pooled.hakn and not pooled.hakn_fallback else 'z'} = {pooled.z:.2f}, "
            f"p {theme.fmt_p(pooled.p)}"
            + (f"; the Hartung–Knapp adjustment was requested but had nothing to adjust: "
               f"{pooled.hakn_fallback}" if pooled.hakn_fallback else "")
            + f"), with "
            f"τ² = {pooled.tau2:.3f}, I² = {100 * pooled.I2:.1f}% and a "
            f"{settings.ci_level * 100:g}% prediction interval of {pi_low:.2f} to {pi_high:.2f}"
            f"{f' (df = {pi_df})' if isinstance(pi_df, int) else ''}.")
        if outcome is not None and (outcome.positive_direction_label
                                    or outcome.negative_direction_label):
            sentence += (f" Positive values mean “{outcome.positive_direction_label}” and "
                         f"negative values “{outcome.negative_direction_label}”.")
        lines.append(sentence)
        lines.append("")
    lines.append("Sensitivity analyses re-pooled each outcome excluding figure-derived rows, "
                 "excluding rows converted from test statistics, including the rows held for "
                 "review, with and without the Hartung–Knapp adjustment, with and without "
                 "digitisation variance, split by analysis metric, and with one row per paper; "
                 "all are reported in `sensitivity.json`. Small-study effects were examined with "
                 "a funnel plot and Egger's test.")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- conclusion
def _conclusions(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Conclusion]:
    """Each outcome's conclusion as `write_outcome_outputs` wrote it into its `pooled.json`.

    Read rather than re-derived, deliberately (DECISION B): the paragraph in the report, the
    block in `pooled.json` and the card the SPA draws are then the same sentences, and a run
    whose numbers were re-pooled cannot end up with a report that disagrees with its own file.
    """
    out: dict[str, Conclusion] = {}
    for key, payload in results.items():
        path = (payload.get("outputs") or {}).get("pooled_json")
        if path is None:
            continue
        try:
            block = json.loads(Path(path).read_text(encoding="utf-8")).get("conclusion")
        except (OSError, ValueError):                      # a report never dies over one file
            continue
        if block:
            out[key] = conclusion_from_payload(block)
    return out


def _renderers(results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each outcome's `renderer` block (DECISION F), read out of its own `pooled.json`.

    Same rule as the conclusion above: what the report says about a figure is what the file
    beside the figure says, so a re-pool cannot leave the two disagreeing.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, payload in results.items():
        path = (payload.get("outputs") or {}).get("pooled_json")
        if path is None:
            continue
        try:
            block = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):                      # a report never dies over one file
            continue
        for name, suffix in (("renderer", ""), ("renderer_best_guess", "_best_guess")):
            if isinstance(block.get(name), dict):
                out[f"{key}{suffix}"] = block[name]
    return out


def _crosscheck_failed(renderer: Mapping[str, Any] | None) -> bool:
    check = (renderer or {}).get("crosscheck")
    return isinstance(check, Mapping) and check.get("ok") is False


#: what a reader has to know to read the plot, per renderer. `meta::forest.meta` draws only what
#: was pooled and sizes its squares its own way; ours draws the held rows hollow beside them.
_CAPTIONS = {
    "R meta::forest.meta": (
        "Squares are individual datasets, sized by their random-effects weight; the diamond is "
        "the pooled estimate and the bar beneath it the prediction interval. Only the rows in "
        "this analysis line are drawn \u2014 rows held for human review are listed below."),
    "": ("Squares are individual datasets; a square's area grows with its random-effects weight "
         "(from a floor, so a near-zero weight is still visible \u2014 the exact weight is "
         "printed beside it). The diamond is the pooled estimate and the bar beneath it the "
         "prediction interval. Hollow squares were held for human review and are not pooled."),
}


def forest_caption(renderer: Mapping[str, Any] | None) -> str:
    """The sentences under a forest: how to read it, then who drew it and why (DECISION F)."""
    name = str((renderer or {}).get("renderer") or "")
    text = _CAPTIONS.get(name, _CAPTIONS[""])
    drawn = caption_line(renderer) if renderer else ""
    return f"{text} {drawn}".strip() if drawn else text


def _renderer_callout(renderer: Mapping[str, Any] | None) -> str:
    """The red box a reader must not be able to miss: R and canopy did not agree."""
    if not _crosscheck_failed(renderer):
        return ""
    check = renderer["crosscheck"]
    named = ", ".join(str(q) for q in (check.get("failed") or [])) or "the pooled result"
    detail = "; ".join(str(w) for w in (check.get("warnings") or []))
    return (f'<p class="callout">R <code>meta</code> and canopy disagreed on {_e(named)}, so '
            f'this plot was NOT drawn by R: it is canopy\u2019s own, from the numbers in the '
            f'tables below. <code>canopy validate</code> exits non-zero on this run. '
            f'{_e(renderer.get("reason") or detail)}</p>')


def _conclusion_markdown(protocol: Protocol, conclusions: Mapping[str, Conclusion]) -> str:
    """`conclusion.md`, beside `methods.md` — the same text, in the form a reader can paste."""
    lines = [f"## Conclusion — {protocol.title}", ""]
    if conclusions:
        lines.extend([overall_conclusion(protocol, conclusions), ""])
    else:
        lines.extend(["No outcome produced a conclusion: nothing was pooled and nothing was "
                      "held.", ""])
    for key, conclusion in conclusions.items():
        lines.extend([f"### {conclusion.facts.get('outcome_label') or key}", "",
                      render_text(conclusion), ""])
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the report
def provenance_table(entries: Mapping[str, Mapping[str, Any]], run_dir: Path) -> str:
    """One row per value that reached the analysis, linking the image its evidence lives in."""
    rows = []
    for entry in sorted(entries.values(), key=lambda e: (str(e.get("dataset_id", "")),
                                                         str(e.get("outcome_key", "")),
                                                         str(e.get("group", "")))):
        crop = entry.get("crop") or ""
        link = (f'<a href="{_link(crop, run_dir)}">evidence</a>' if crop
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
    """Write `report.html`, `methods.md` and `conclusion.md` into the run directory."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = results or {}
    run_outputs = dict(run_outputs or {})
    settings = manifest.settings

    methods = methods_paragraph(manifest, protocol, results)
    methods_path = directory / "methods.md"
    methods_path.write_text(methods, encoding="utf-8")

    conclusions = _conclusions(results)
    renderers = _renderers(results)
    conclusion_path = directory / "conclusion.md"
    conclusion_path.write_text(_conclusion_markdown(protocol, conclusions), encoding="utf-8")

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

    # DECISION B: the conclusion sits immediately after the header cards, because it is the part
    # of this page a reader quotes — and every sentence of it was assembled from the numbers
    # below by `report.conclusion`, with no model in the path
    if conclusions:
        parts.append("<h2>Conclusion</h2>")
        parts.extend(f"<p>{_e(line)}</p>"
                     for line in overall_conclusion(protocol, conclusions).split("\n")
                     if line.strip())
        parts.extend(f'<p class="conclusion">{_e(render_text(c))}</p>'
                     for c in conclusions.values())
        parts.append('<p class="files"><a href="conclusion.md">conclusion.md</a></p>')

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
        conclusion = conclusions.get(key)
        if conclusion is not None:                         # the same paragraph, above the forest
            parts.append(f'<p class="conclusion">{_e(render_text(conclusion))}</p>')
        forest = outputs.get("forest_png")
        if forest is not None:
            renderer = renderers.get(key)
            parts.append(_renderer_callout(renderer))
            parts.append(
                f'<figure><img src="{_link(forest, directory)}" '
                f'alt="Forest plot for {_e(outcome.label if outcome else key)}">'
                f"<figcaption>{_e(forest_caption(renderer))}</figcaption></figure>")
        for name, caption in (("sensitivity_png",
                               "Each panel re-pools this outcome under one changed choice."),
                              ("funnel_png",
                               "Funnel plot with pseudo-confidence contours; Egger's test is in "
                               "the accompanying JSON.")):
            path = outputs.get(name)
            if path is not None:
                parts.append(f'<figure><img src="{_link(path, directory)}" alt="{_e(name)}">'
                             f"<figcaption>{_e(caption)}</figcaption></figure>")
        parts.append(_links(outputs, directory, [
            "forest_png", "forest_svg", "forest_pdf", "extraction_csv", "extraction_json",
            "extraction_xlsx", "leave_one_out_csv", "sensitivity_json", "sensitivity_png",
            "funnel_png", "funnel_json", "pooled_json"]))

    methods_fig = run_outputs.get("methods_fig_png")
    if methods_fig is not None:
        parts.append("<h2>How each value was obtained</h2>")
        parts.append(f'<figure><img src="{_link(methods_fig, directory)}" '
                     f'alt="Extraction routes"><figcaption>Share of datasets by extraction '
                     f"route, with real examples: page crops with the extracted quote "
                     f"highlighted, and the digitiser's overlays.</figcaption></figure>")
    prisma = run_outputs.get("prisma_png")
    if prisma is not None:
        parts.append("<h2>Flow of records</h2>")
        parts.append(f'<figure><img src="{_link(prisma, directory)}" alt="PRISMA-style flow">'
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
            parts.append(f'<p class="files"><a href="{_link(prov_json, directory)}">'
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
    return {"html": path, "methods": methods_path, "conclusion": conclusion_path}
