"""One look for every figure Canopy draws, and the conventions footer they all carry.

Colour follows the `dataviz` skill: the marks are **neutral greys** and exactly **one accent**
carries meaning (the pooled diamond and everything that belongs to it — its whiskers, its
prediction bar, the reference line at the pooled estimate). Nothing else is coloured, so a reader
never has to decode a palette, and nothing is distinguishable by hue alone: a `needs_human` row is
hollow, a route is a glyph, an override is a marker. Validated with the skill's own checker against
a white surface (`node scripts/validate_palette.js "#2a78d6,#52514e,#898781" --mode light
--surface "#ffffff" --pairs all`): accent-vs-grey CVD ΔE 15.9, normal-vision ΔE 17.8, every colour
≥ 3:1 on the surface. The checker's chroma floor is reported as a FAIL for the two greys, which is
the intended result — they are chart chrome, not a categorical series.

The footer is not decoration. A forest plot without its conventions cannot be reproduced or
compared with a published one, so `conventions_footer` enumerates every choice that moved a number:
estimator, variance formula, τ² method, Hartung–Knapp, the prediction-interval method and its df,
which I² is printed, how many datasets and how many papers they came from, and how many rows were
held back for a human (amendment A/H).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")                      # never open a window; every output is a file
import matplotlib.pyplot as plt            # noqa: E402

from ..models import EffectSizeRecord, StatsSettings   # noqa: E402
from ..stats.meta import MetaResult                    # noqa: E402

__all__ = ["SURFACE", "INK", "INK_SECONDARY", "MUTED", "GRID", "AXIS", "MARK", "ACCENT",
           "ACCENT_SOFT", "HOLLOW_FACE", "ROUTE_GLYPHS", "ROUTE_LABELS", "GLYPH_LEGEND",
           "OVERRIDE_FLAG", "OVERRIDE_MARK", "BEST_GUESS_CAVEAT", "DPI", "route_glyph",
           "is_overridden",
           "conventions_footer", "estimator_label", "variance_label", "pi_label", "fmt_p",
           "figure_style", "save_figure", "study_label", "fmt", "fmt_ci"]

# --------------------------------------------------------------------------- palette
SURFACE = "#ffffff"          # printed figures live on white paper
INK = "#0b0b0b"              # primary text
INK_SECONDARY = "#52514e"    # study marks and secondary text
MUTED = "#898781"            # axis labels, footers
GRID = "#e1e0d9"             # hairline grid
AXIS = "#c3c2b7"             # baseline / spines
MARK = INK_SECONDARY         # study squares and whiskers (neutral grey)
ACCENT = "#2a78d6"           # THE accent: the pooled summary and nothing else
ACCENT_SOFT = "#86b6ef"      # the same hue, lighter — used only inside the pooled summary
HOLLOW_FACE = "none"         # a needs_human row is drawn hollow
DPI = 300

#: one glyph per extraction route, so the source of every row is on the plot itself
ROUTE_GLYPHS: dict[str, str] = {
    "text_mean_sd": "¶",
    "text_mean_se_ci": "¶",
    "text": "¶",
    "table": "▦",
    "figure": "▤",
    "test_statistic": "ƒ",
    "p_value": "ƒ",
    "reported_d": "†",
    "adjudicated": "¶",
    "composite": "Σ",
    "not_convertible": "⌀",
    "": "·",
}
ROUTE_LABELS: dict[str, str] = {
    "¶": "text", "▦": "table", "▤": "figure", "ƒ": "test statistic",
    "†": "reported effect size", "Σ": "this paper's rows combined", "⌀": "not convertible",
    "·": "unknown",
}
#: a value a human replaced carries this flag (the review workflow writes it) and this marker
OVERRIDE_FLAG = "human_override"
OVERRIDE_MARK = "△"

#: DECISION A. Printed on every best-guess artefact — the forest's subtitle, the report's cards,
#: the SPA's second tab — because the one real risk of a second line is that it becomes the
#: quoted number. It says what the line is, in one sentence, wherever the line is shown.
BEST_GUESS_CAVEAT = ("Best guess, not the primary analysis: this line adds rows held for human "
                     "review, each taken at the value the tool read, and each still unconfirmed.")


def best_guess_caveat(*, fired: bool = False, crossings: bool = False,
                      borrowed: bool = False,
                      by_rule_totals: dict[str, int] | None = None,
                      disputed: int = 0) -> str:
    """DECISION B: the caveat ASSEMBLED, when the answer tier fired — else the constant above.

    Fire-gated on purpose (rules §7/M3): an outcome where no cell rule fired keeps the exact
    string it printed before the tier existed, byte for byte. When one did, the base sentence
    changes to say what the line now does — enters values already in the record under named
    rules — and each clause below is added only when it is true of THIS outcome. The `disputed`
    clause is the exception, added fired or not: a `disputed_reading_guess` admission is a
    DECISION A row rule, and the line must announce it wherever the caveat is shown.
    """
    if fired:
        text = ("Best guess, not the primary analysis: this line adds rows held for human "
                "review, each entered at values already in the run's own record under named "
                "per-row rules, and each still unconfirmed.")
        if crossings:
            text += (" Some rows answer still-open review questions past standing objections; "
                     "every crossing is listed on the row.")
        if borrowed:
            text += (" Where a settled mean lacked a printed spread, the spread was borrowed "
                     "from a same-cell reading the record designates — borrowing can "
                     "under-weight a row and pull its effect toward null.")
        if by_rule_totals:
            text += (" Best-guess rows by rule: "
                     + ", ".join(f"{rule} ×{count}"
                                 for rule, count in sorted(by_rule_totals.items())) + ".")
    else:
        text = BEST_GUESS_CAVEAT
    if disputed > 0:
        text += (f" {disputed} row(s) enter over a standing dispute about what the number is "
                 f"or which way it points; each names its dispute.")
    return text


def route_glyph(route: str) -> str:
    """The glyph for a route name, tolerant of the digitiser's `figure:*` sub-routes."""
    name = (route or "").strip()
    if name in ROUTE_GLYPHS:
        return ROUTE_GLYPHS[name]
    head = name.split(":", 1)[0]
    if head in ROUTE_GLYPHS:
        return ROUTE_GLYPHS[head]
    if head.startswith(("figure", "digitize")):
        return ROUTE_GLYPHS["figure"]
    if head.startswith("text"):
        return ROUTE_GLYPHS["text"]
    return ROUTE_GLYPHS[""]


def is_overridden(record: EffectSizeRecord) -> bool:
    return OVERRIDE_FLAG in (record.flags or ()) or record.route == OVERRIDE_FLAG


def GLYPH_LEGEND(routes: Iterable[str], overridden: bool = False) -> str:
    """`Source: ¶ text · ▦ table …` for exactly the routes on this plot."""
    seen: list[str] = []
    for route in routes:
        glyph = route_glyph(route)
        if glyph not in seen:
            seen.append(glyph)
    parts = [f"{g} {ROUTE_LABELS.get(g, 'unknown')}" for g in seen]
    if overridden:
        parts.append(f"{OVERRIDE_MARK} value overridden by a human")
    return "Source:  " + "   ".join(parts) if parts else ""


# --------------------------------------------------------------------------- number formatting
def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return "—"
    return f"{value:.{digits}f}"


def fmt_p(p: float | None, digits: int = 4) -> str:
    """`< 0.0001` rather than a rounded `= 0.0000`, which would read as an exact zero."""
    if p is None or p != p:
        return "= —"
    floor = 10.0 ** -digits
    return f"< {floor:.{digits}f}" if p < floor else f"= {p:.{digits}f}"


def fmt_ci(low: float | None, high: float | None, digits: int = 2) -> str:
    if low is None or high is None or low != low or high != high:
        return "—"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def study_label(record: EffectSizeRecord) -> str:
    """`First author` — the citation when the mapper found one, else whatever names the row."""
    cite = record.citation
    name = (cite.first_author or "").strip() or (cite.authors or "").strip()
    if not name:
        name = (record.label or record.dataset_id or record.paper_id or "").strip()
    return name or "—"


# --------------------------------------------------------------------------- footer
_ESTIMATORS = {"cohen": "Cohen's d", "hedges": "Hedges' g"}
_VARIANCES = {
    "borenstein": "Borenstein large-sample",
    "hedges_olkin_df": "Hedges–Olkin (df form)",
    "meta_exact": "meta::metacont exact",
    "meta_exact_g": "meta::metacont exact (g)",
    "meta_hedges_approx": "meta Hedges approximation",
}
_PI = {"V": ("V", "t(k−1)"), "HTS": ("HTS", "t(k−2)"), "Z": ("z", "normal"), "S": ("z", "normal")}


def estimator_label(settings: StatsSettings) -> str:
    return _ESTIMATORS.get(str(settings.estimator), str(settings.estimator))


def variance_label(settings: StatsSettings) -> str:
    return _VARIANCES.get(str(settings.variance), str(settings.variance))


def pi_label(settings: StatsSettings, pooled: MetaResult | None = None) -> str:
    name, dist = _PI.get(str(settings.pi_method).upper(), (str(settings.pi_method), "t"))
    if pooled is None:
        return f"{name}, {dist}"
    # under a cluster-robust pool the PI was computed over CLUSTERS — the label must count what
    # the estimator counted, and adds the caveat the method owes (robumeta reports no PI at all)
    base = pooled.n_clusters if getattr(pooled, "robust", False) else pooled.k
    df = {"V": base - 1, "HTS": base - 2}.get(str(settings.pi_method).upper())
    label = f"{name}, {dist}" + (f", df = {df}" if df is not None else "")
    if getattr(pooled, "robust", False):
        label += " over clusters (working-model quantity; robumeta reports none)"
    return label


def ci_method_label(pooled: MetaResult) -> str:
    """What the interval on THIS pool was actually built from — not what was asked for.

    `MetaResult.hakn` records the request, deliberately (R2), and the R2 guard sets
    `hakn_fallback` when the adjustment had nothing to adjust and the ordinary random-effects SE
    and the normal `z` quantile were used instead. Reading `hakn` alone printed "CI from t(k−1),
    Hartung–Knapp" over a normal-z interval — a false statement about the method, sitting next to
    an honest note nothing read (review M4).
    """
    if getattr(pooled, "robust_fallback", ""):
        return (f"z (cluster-robust variance was requested but could not be applied: "
                f"{pooled.robust_fallback})")
    if getattr(pooled, "robust", False):
        return f"t(df = {pooled.df_robust:.2f}, Satterthwaite), cluster-robust"
    if getattr(pooled, "hakn_fallback", ""):
        return (f"z (the Hartung–Knapp adjustment was requested but had nothing to adjust: "
                f"{pooled.hakn_fallback})")
    return "t(k−1), Hartung–Knapp" if pooled.hakn else "z"


def conventions_footer(settings: StatsSettings, pooled: MetaResult, *, k_papers: int,
                       k_datasets: int, n_excluded: int = 0, n_not_convertible: int = 0,
                       level: float | None = None, extra: Sequence[str] = ()) -> list[str]:
    """Every statistical convention that moved a number on this plot, as footer lines.

    `n_excluded` is every row held back from the pool and `n_not_convertible` how many of those
    were held because no route could produce an effect size at all. The two are different
    findings — one is a question for a reviewer, the other is a paper that did not report enough —
    so the footer never folds the second into the first.
    """
    level = float(level if level is not None else settings.ci_level)
    pct = f"{level * 100:g}%"
    robust = bool(getattr(pooled, "robust", False))
    if robust:
        # every fragment names what the CORR fit actually computed: MoM τ², robumeta's Q-based
        # I² with its non-integer df and NO p (robumeta defines none), and no I²τ/H² — those are
        # single-level constructs the working model does not produce. Printing "τ² by REML" or
        # "I²τ = nan%" here would be a wrong label on a real number.
        tau_line = (f"Random-effects model · {estimator_label(settings)} "
                    f"(variance: {variance_label(settings)}) · τ² by CORR method of moments "
                    f"(RVE) (τ² = {pooled.tau2:.4f}, τ = {pooled.tau:.3f})")
        q_line = (f"Q = {pooled.Q:.2f} (df = {pooled.Q_df:.2f}; no p: heterogeneity test not "
                  f"defined under RVE) · I² = {100 * pooled.I2:.1f}% [(Q−df)/Q, robumeta]")
        method_slot = "(dependency: cluster-robust RVE)"
    else:
        tau_line = (f"Random-effects model · {estimator_label(settings)} "
                    f"(variance: {variance_label(settings)}) · τ² by {settings.tau2_method} "
                    f"(τ² = {pooled.tau2:.4f}, τ = {pooled.tau:.3f})")
        q_line = (f"Q = {pooled.Q:.2f} (df = {pooled.Q_df}, p {fmt_p(pooled.Q_p)}) · "
                  f"I² = {100 * pooled.I2:.1f}% [(Q−df)/Q, meta] · "
                  f"I²τ = {100 * pooled.I2_tau:.1f}% [τ²/(τ²+s²), metafor] · "
                  f"H² = {pooled.H2:.2f}")
        method_slot = f"(Hartung–Knapp: {'on' if settings.hakn else 'off'})"
    lines = [
        tau_line,
        q_line,
        f"{pct} CI from {ci_method_label(pooled)} {method_slot} · "
        + (f"{pct} prediction interval: {pi_label(settings, pooled)}"
           if (pooled.n_clusters if robust else pooled.k) >= 3 else
           f"no prediction interval: it needs k ≥ 3 and k = "
           f"{pooled.n_clusters if robust else pooled.k}"),
        f"k = {k_datasets} datasets from {k_papers} papers · "
        f"{n_excluded} rows excluded (needs_human {max(0, n_excluded - n_not_convertible)}, "
        f"not convertible {n_not_convertible}): drawn hollow, not pooled",
    ]
    if robust:
        line = (f"Cluster-robust SE (robumeta-style CORR working model, ρ = {pooled.rho:g}): "
                f"m = {pooled.n_clusters} clusters of {pooled.k} rows; rows of one paper share "
                f"a cluster · Satterthwaite df = {pooled.df_robust:.2f}")
        if getattr(pooled, "robust_small_sample", False):
            line += " · df < 4: the p-value is not trustworthy at this few clusters (Tipton 2015)"
        if getattr(pooled, "robust_note", ""):
            line += f" · {pooled.robust_note}"
        lines.insert(3, line)
    elif getattr(pooled, "robust_fallback", ""):
        lines.insert(3, f"Cluster-robust SE requested but not applied: {pooled.robust_fallback}")
    lines.extend(str(line) for line in extra if line)
    return lines


# --------------------------------------------------------------------------- matplotlib
_RC: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",            # keep text as text, so a test (and a reader) can find it
    "pdf.fonttype": 42,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
}


@contextmanager
def figure_style(**overrides: Any) -> Iterator[None]:
    with plt.rc_context({**_RC, **overrides}):
        yield


def save_figure(fig, out_stem, formats: Sequence[str] = ("png", "svg", "pdf"),
                bbox_inches: Any = "tight") -> dict[str, Any]:
    """Write one figure in every requested format; returns `{format: path}`.

    `bbox_inches=None` keeps the figure's own geometry — pass it when the layout was computed in
    inches and a tight box would silently re-crop it.
    """
    from pathlib import Path

    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, format=suffix, dpi=DPI if suffix == "png" else None,
                    bbox_inches=bbox_inches)
        out[suffix] = path
    plt.close(fig)
    return out
