"""The JT deliverable: two bands (late adaptation, aftereffect), each with two forest panels —
Canopy's run on the left, Nyamsuren & Tsay's own values (same papers only) on the right — in the
same format as the Cisneros/Elizabeth comparison figure.

Every number is read from the run's artefacts or her transcribed Fig 4 rows; both sides are
pooled by canopy.stats.meta.cluster_robust (left: clustered by paper, the run's setting; right:
per dataset, her own convention, established against her printed weights).
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, "/Users/sritejpadmanabhan/Projects/canopy")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from canopy.stats.meta import cluster_robust

RUN = Path("/Users/sritejpadmanabhan/Projects/canopy/runs/"
           "20260827-080330-handdominanceandupper-limbsensorimotorad")
GOLD = Path("/Users/sritejpadmanabhan/Projects/canopy/validation/out/nyamsuren_tsay")
OUT = GOLD.parent / "jt_indranil_vs_canopy.png"

NAVY = "#25344a"
INK = "#1b1f24"
MUTED = "#6a7280"
SQ = "#8a8f98"
RED = "#c0392b"

ROUTE_NAME = {"figure": "Fig", "text_mean_sd": "Text", "text_mean_se_ci": "Text",
              "table": "Table", "test_statistic": "Text", "reported_d": "Text"}

# ---- the mark system (Cisneros format), every mark from the run record ------------------------
# ★/☆ = the row's plotted values were entered by the reviewer (both groups / one group), read
# straight off the overrides log via canopy's own registry (human_landed_values). The reason
# glyph compares the human's number to the tool's own PRE-answer read for that cell: the
# verdict-chosen candidates (papers/<c>/verify.json) or, for cells the tool re-read after an
# answer, only the superseded (displaced original) candidates — so an answer-informed re-read
# can never masquerade as "the tool had already read it".
from canopy.pipeline.overrides import human_landed_values

_POOLS, _LOCATED, _CHOSEN, _REREAD = {}, {}, {}, {}


def _load_candidates():
    import json
    for pdir in sorted((RUN / "papers").iterdir()):
        if not (pdir / "extract.json").exists():
            continue
        ex = json.load(open(pdir / "extract.json"))
        ver = json.load(open(pdir / "verify.json")) if (pdir / "verify.json").exists() else {}
        cluster = pdir.name
        rr = {tuple(cell.split("/")) for cell in ex.get("cells_reread_for_override") or []}
        _REREAD[cluster] = rr
        cur = list(ex.get("candidates") or []) + list(ver.get("extra_candidates") or [])
        sup = list(ex.get("superseded_candidates") or [])
        for c in cur + sup:
            key = (c.get("dataset_id"), c.get("outcome_key"), c.get("group"))
            _LOCATED.setdefault(cluster, {}).setdefault(key, []).append(c)
            if (key[0], key[1]) in rr and c not in sup:
                continue                       # post-answer re-read: not a pre-answer read
            _POOLS.setdefault(cluster, {}).setdefault(key, []).append(c)
        by_id = {c.get("candidate_id"): c for c in cur + sup}
        for v in ver.get("verdicts") or []:
            key = (v.get("dataset_id"), v.get("outcome_key"), v.get("group"))
            if (key[0], key[1]) in rr:
                continue                       # verdict may postdate the answer there
            picked = [by_id[i] for i in v.get("agreeing_ids") or [] if i in by_id]
            if picked:
                _CHOSEN.setdefault(cluster, {}).setdefault(key, picked)


def _tool_reads(cluster, ds, oc, g, field):
    cands = _CHOSEN.get(cluster, {}).get((ds, oc, g)) or _POOLS.get(cluster, {}).get((ds, oc, g), [])
    vals = [float(c[field]) for c in cands
            if c.get("status") == "found" and c.get(field) is not None]
    if not vals:
        vals = [float(c[field]) for c in _POOLS.get(cluster, {}).get((ds, oc, g), [])
                if c.get("status") == "found" and c.get(field) is not None]
    return vals


def _same(tool, human, tol=0.03):
    return abs(tool - human) <= max(abs(human) * tol, 1e-9)


def _classify_cell(cluster, ds, oc, g, human):
    """None if no value human-stated; else 'clar' (◆), 'none' (▲) or 'diff' (●)."""
    fields = human["field_seqs"]
    for field in ("mean", "dispersion_value"):
        if human.get(field) is not None and field in fields:
            tool = _tool_reads(cluster, ds, oc, g, field)
            if not tool:
                return "none"
            return "clar" if any(_same(t, float(human[field])) for t in tool) else "diff"
    if "n" in fields or human.get("n") is not None or "dispersion_type" in fields:
        return "clar"                     # size/typing pinned; the numbers are the tool's
    return None


def _read_from(r, cluster, ds, oc, g):
    """The source the tool itself located for the cell, in the paper's own numbering."""
    cands = _LOCATED.get(cluster, {}).get((ds, oc, g), [])
    if not any(c.get("status") == "found" for c in cands):
        return ""                          # the tool located nothing
    route = (r.get(f"route_{g.lower()}") or r["route"] or "").split(":")[0]
    page = (r.get(f"page_{g.lower()}") or "").strip()
    mean = r.get(f"mean_{g.lower()}")
    best, best_score = None, -1
    for c in cands:
        score = (str(c.get("page") or "") == page) + ((c.get("route") or "") == route) \
            + bool(c.get("locator"))
        try:
            if mean not in (None, "") and c.get("mean") is not None and \
                    _same(float(c["mean"]), float(mean)):
                score += 2
        except ValueError:
            pass
        if score > best_score:
            best, best_score = c, score
    label = ""
    if best is not None:
        text = " ".join(str(best.get(k) or "") for k in ("locator", "row_header",
                                                         "col_header", "quote"))
        fig = re.search(r"\b(?:Fig(?:ure)?\.?)\s*(S?\d+[A-Za-z]?)", text)
        tab = re.search(r"\bTable\s*(S?\d+[A-Za-z]?)", text, re.I)
        if route == "figure" and fig:
            label = f"Fig {fig.group(1)}"
        elif route == "table" and tab:
            label = f"Table {tab.group(1)}"
    if not label:
        label = {"figure": "Fig", "table": "Table"}.get(route, "Text" if route else "")
    if not page and best is not None and best.get("page"):
        page = str(best["page"])
    return f"{label}, p.{page}" if label and page else label


def canopy_rows(outcome):
    if not _POOLS:
        _load_candidates()
    landed = human_landed_values(RUN)
    rows = [r for r in csv.DictReader(open(RUN / "results" / outcome / "extraction_table.csv"))
            if r["in_best_guess"] == "true"]
    out = []
    for r in rows:
        es, se = float(r["best_guess_es"]), float(r["best_guess_se"])
        pert = (r.get("mod_perturbation_size_deg") or "").strip()
        m = re.search(r"(\d+(?:\.\d+)?)", pert)
        pert = m.group(1) if m else ""
        ds, cluster = r["dataset_id"], r["cluster_id"]
        klass, starred = [], []
        for g in ("A", "B"):
            human = landed.for_cell(ds, outcome, g)
            if human is None or (not human["field_seqs"] and human.get("n") is None):
                continue
            c = _classify_cell(cluster, ds, outcome, g, human)
            if c:
                klass.append(c)
                starred.append(g)
        if not klass:
            mark = ""
        else:
            star = "★" if len(starred) == 2 else "☆"
            glyph = "●" if "diff" in klass else "▲" if "none" in klass else "◆"
            mark = star + glyph
        src = _read_from(r, cluster, ds, outcome, "A") or _read_from(r, cluster, ds, outcome, "B")
        out.append(dict(
            author=r["first_author"], year=r["year"], task=r["mod_task_type"],
            pert=(pert + "°") if pert and pert.replace(".", "").isdigit() else (pert or "—"),
            n=f"{r['n_a'] or '—'}/{r['n_b'] or '—'}",
            src=src,
            es=es, lo=es - 1.959964 * se, hi=es + 1.959964 * se, var=se * se,
            cluster=r["cluster_id"], mark=mark,
            primary=r["primary_row"] == "true"))
    out.sort(key=lambda d: d["es"])
    return out


def _skey(name):
    """First surname, accent-stripped, for matching her citation to the run's paper."""
    import unicodedata
    s = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()
    return s.replace("&", " ").split()[0].strip(".,").lower()


def her_rows(fname, plotted=None):
    """Her transcribed Fig 4 rows; with `plotted` (the (surname, year) keys of the rows the
    Canopy panel shows), only rows for those same papers are kept — her citation year may sit
    one off the run's (e.g. Carroll 2015/2016, Addison 2023/2024), so years match within ±1.
    (Solo Sainburg 2002 and Sainburg & Wang 2002 share a surname-year key; both papers are in
    the run since 2026-09-07, so both her rows stay.)"""
    out = []
    for r in csv.DictReader(open(GOLD / fname)):
        author, year = r["Author"].strip(), r["Year"].strip()
        if plotted is not None and not any(
                _skey(author) == sk and abs(int(year) - yr) <= 1 for sk, yr in plotted):
            continue
        exp = (r.get("Experiment") or "").strip()
        out.append(dict(author=author, year=year, task=r["Task"].strip(), pert=exp or "—",
                        es=float(r["TE"]), lo=float(r["CI_low"]), hi=float(r["CI_high"]),
                        var=float(r["seTE"]) ** 2, cluster=None, mark="", n="", src=""))
    out.sort(key=lambda d: d["es"])
    return out


def pool(rows, per_dataset):
    y = [r["es"] for r in rows]
    v = [r["var"] for r in rows]
    cl = [str(i) for i in range(len(rows))] if per_dataset else [r["cluster"] for r in rows]
    res = cluster_robust(y, v, cl, rho=0.8)
    for r, w in zip(rows, res.weights_pct):
        r["w"] = float(w)
    return res


def fmt(v, d=1):
    s = f"{v:.{d}f}"
    return "-0.0" if s == "-0.0" and d == 1 else s


# ---- panel geometry: everything in axes coordinates, one row per unit of y -------------------
COLS_L = [("Author", 0.000, "left"), ("Year", 0.118, "left"), ("Task", 0.158, "left"),
          ("Perturb.", 0.235, "left"), ("N (D/N)", 0.300, "left"), ("Read from", 0.362, "left"),
          ("Hedges' g", 0.795, "right"), ("95% CI", 0.910, "right"), ("Weight", 0.985, "right")]
COLS_R = [("Author", 0.000, "left"), ("Year", 0.150, "left"), ("Task", 0.196, "left"),
          ("Exp.", 0.268, "left"),
          ("Hedges' g", 0.795, "right"), ("95% CI", 0.910, "right"), ("Weight", 0.985, "right")]
PLOT_L, PLOT_R = 0.435, 0.700          # the forest strip inside the panel


def draw_panel(ax, rows, pooled, title, sub1, sub2, xlim, cols, direction, n_max=None,
               name_chars=12):
    n = len(rows)
    top = n if n_max is None else max(n, n_max)
    ax.set_xlim(0, 1)
    ax.set_ylim(-8.2, top + 4.6)
    ax.axis("off")

    def x_of(v):
        return PLOT_L + (v - xlim[0]) / (xlim[1] - xlim[0]) * (PLOT_R - PLOT_L)

    ax.text(0, top + 4.1, title, fontsize=11.5, fontweight="bold", color=INK, va="top")
    ax.text(0, top + 2.9, sub1, fontsize=8.6, color=INK, va="top")
    ax.text(0, top + 1.9, sub2, fontsize=7.4, color=MUTED, va="top")
    for name, x, alignment in cols:
        ax.text(x, top + 0.55, name, fontsize=7.6, fontweight="bold", color=INK,
                ha=alignment, va="center")
    ax.plot([0, 1], [top + 0.05, top + 0.05], lw=0.8, color=INK)

    # zero and pooled guide lines through the rows
    row_bottom = top - n - 0.4
    ax.plot([x_of(0)] * 2, [row_bottom - 0.9, top - 0.1], ls=(0, (1, 2)), lw=0.8, color="#9aa0a8", zorder=1)
    ax.plot([x_of(pooled.estimate)] * 2, [row_bottom - 0.9, top - 0.1], ls=(0, (4, 3)), lw=0.7,
            color="#c3c7cd", zorder=1)

    wmax = max(r["w"] for r in rows)
    for i, r in enumerate(rows):
        y = top - 1 - i + 0.5
        name = r["author"][:name_chars] + ("…" if len(r["author"]) > name_chars else "")
        ax.text(0.000, y, f"{name} {r['mark']}".strip(), fontsize=7.4, color=INK, va="center")
        ax.text(cols[1][1], y, r["year"], fontsize=7.4, color=INK, va="center")
        ax.text(cols[2][1], y, r["task"].replace("force_field", "force field")[:11], fontsize=7.4, color=INK, va="center")
        ax.text(cols[3][1], y, str(r["pert"])[:7], fontsize=7.4, color=INK, va="center")
        if r["n"]:
            ax.text(0.300, y, r["n"], fontsize=7.4, color=INK, va="center")
        if r["src"]:
            ax.text(0.362, y, r["src"][:12], fontsize=7.4, color=INK, va="center")
        lo, hi = max(r["lo"], xlim[0]), min(r["hi"], xlim[1])
        ax.plot([x_of(lo), x_of(hi)], [y, y], lw=0.9, color=INK, zorder=2,
                solid_capstyle="butt")
        for endpoint, val in ((r["lo"], lo), (r["hi"], hi)):
            if endpoint == val:                              # not clipped: draw the serif
                ax.plot([x_of(val)] * 2, [y - 0.16, y + 0.16], lw=0.9, color=INK, zorder=2)
        side = 0.006 + 0.016 * (r["w"] / wmax) ** 0.5
        ax.add_patch(Rectangle((x_of(r["es"]) - side / 2, y - side / 2 * 16), side, side * 16,
                               facecolor=SQ, edgecolor="none", zorder=3))
        ax.text(0.795, y, fmt(r["es"]), fontsize=7.4, color=INK, va="center", ha="right")
        ax.text(0.910, y, f"[{fmt(r['lo'])}; {fmt(r['hi'])}]", fontsize=7.4, color=INK,
                va="center", ha="right")
        ax.text(0.985, y, f"{r['w']:.1f}%", fontsize=7.4, color=INK, va="center", ha="right")

    # pooled diamond + prediction interval + heterogeneity
    y0 = top - n - 1.6
    ax.text(0.000, y0, "Random effects model (cluster-robust)", fontsize=7.6,
            fontweight="bold", color=INK, va="center")
    dx0, dx1, dxm = x_of(pooled.ci_low), x_of(pooled.ci_high), x_of(pooled.estimate)
    ax.add_patch(Polygon([(dx0, y0), (dxm, y0 + 0.42), (dx1, y0), (dxm, y0 - 0.42)],
                         facecolor="#5b6470", edgecolor=INK, lw=0.5, zorder=3))
    ax.text(0.795, y0, fmt(pooled.estimate), fontsize=7.6, fontweight="bold", color=INK,
            va="center", ha="right")
    ax.text(0.910, y0, f"[{fmt(pooled.ci_low)}; {fmt(pooled.ci_high)}]", fontsize=7.6,
            fontweight="bold", color=INK, va="center", ha="right")
    ax.text(0.985, y0, "100.0%", fontsize=7.6, fontweight="bold", color=INK, va="center",
            ha="right")
    y1 = y0 - 1.0
    ax.text(0.000, y1, "Prediction interval", fontsize=7.6, color=INK, va="center")
    if pooled.pi_low == pooled.pi_low:                     # not NaN
        ax.plot([x_of(max(pooled.pi_low, xlim[0])), x_of(min(pooled.pi_high, xlim[1]))],
                [y1, y1], lw=2.4, color=RED, zorder=3, solid_capstyle="butt")
        ax.text(0.910, y1, f"[{fmt(pooled.pi_low)}; {fmt(pooled.pi_high)}]", fontsize=7.6,
                color=INK, va="center", ha="right")
    het = (f"Heterogeneity: $I^2$ = {100 * pooled.I2:.1f}% · τ² = "
           f"{pooled.tau2:.2f} (CORR MoM; no p under RVE) · Satterthwaite df = "
           f"{pooled.df_robust:.2f}")
    ax.text(0.000, y1 - 1.0, het, fontsize=7.0, color=MUTED, va="center")

    # axis
    ya = -5.2
    ax.plot([x_of(xlim[0]), x_of(xlim[1])], [ya, ya], lw=0.8, color=INK)
    tick = int(xlim[0])
    while tick <= xlim[1]:
        ax.plot([x_of(tick)] * 2, [ya, ya - 0.22], lw=0.8, color=INK)
        ax.text(x_of(tick), ya - 0.55, str(tick), fontsize=7.0, color=INK, ha="center",
                va="top")
        tick += 2
    left_lab, right_lab = [s.strip() for s in direction.split("|")]
    ax.text(x_of(0) - 0.006, ya - 1.75, left_lab, fontsize=7.2, color=INK, ha="right")
    ax.text(x_of(0) + 0.006, ya - 1.75, right_lab, fontsize=7.2, color=INK, ha="left")


def band(fig, y_top, height, title, left, right):
    bar = fig.add_axes([0.015, y_top - 0.020, 0.970, 0.020])
    bar.set_facecolor(NAVY)
    bar.set_xticks([]); bar.set_yticks([])
    for s in bar.spines.values():
        s.set_visible(False)
    bar.text(0.006, 0.5, title, color="white", fontsize=12.5, fontweight="bold",
             va="center", transform=bar.transAxes)
    axl = fig.add_axes([0.015, y_top - height, 0.470, height - 0.024])
    axr = fig.add_axes([0.515, y_top - height, 0.470, height - 0.024])
    n_max = max(len(left[0]), len(right[0]))
    draw_panel(axl, *left, n_max=n_max, name_chars=12)
    draw_panel(axr, *right, n_max=n_max, name_chars=17)


def main():
    direction = "Greater in non-dominant limb | Greater in dominant limb"
    bands = []
    for outcome, gold_file, name in (("late_adaptation", "late_gsheet.csv", "LATE ADAPTATION"),
                                     ("aftereffect", "aft_gsheet.csv", "AFTEREFFECT")):
        c_rows = canopy_rows(outcome)
        c_pool = pool(c_rows, per_dataset=False)
        plotted = {(_skey(r["author"]), int(r["year"])) for r in c_rows}
        h_full = her_rows(gold_file)
        h_pool_full = pool(h_full, per_dataset=True)
        h_rows = her_rows(gold_file, plotted=plotted)
        h_pool = pool(h_rows, per_dataset=True)
        dropped = [f"{r['author']} {r['year']}" for r in h_full
                   if not any(_skey(r["author"]) == sk and abs(int(r["year"]) - yr) <= 1
                              for sk, yr in plotted)]
        print(f"{outcome}: right panel keeps {len(h_rows)} of {len(h_full)} of her rows; "
              f"dropped: {sorted(set(dropped))}")
        n_star = sum(1 for r in c_rows if r["mark"])
        n_clar = sum(1 for r in c_rows if r["mark"].endswith("◆"))
        n_none = sum(1 for r in c_rows if r["mark"].endswith("▲"))
        n_diff = sum(1 for r in c_rows if r["mark"].endswith("●"))
        n_all = len(list(csv.DictReader(open(RUN / "results" / outcome /
                                             "extraction_table.csv"))))
        prim = [r for r in c_rows if r["primary"]]
        p_pool = pool_primary = cluster_robust([r["es"] for r in prim],
                                               [r["var"] for r in prim],
                                               [r["cluster"] for r in prim], rho=0.8)
        left = (c_rows, c_pool,
                "AutoMeta  (this run, best-guess line: primary + rows still held for review)",
                f"AutoMeta: g = {fmt(c_pool.estimate, 2)} [{fmt(c_pool.ci_low, 2)}; "
                f"{fmt(c_pool.ci_high, 2)}], k={c_pool.k}   ·   primary line alone: "
                f"g = {fmt(p_pool.estimate, 2)} [{fmt(p_pool.ci_low, 2)}; "
                f"{fmt(p_pool.ci_high, 2)}], k={p_pool.k}",
                f"★ {n_star} of {len(c_rows)} rows human-entered:   "
                f"◆ {n_clar} clarification only   ▲ {n_none} tool read nothing   "
                f"● {n_diff} tool's read corrected   ·   "
                f"{n_all - len(c_rows)} rows still held for review are not shown",
                (-4, 4), COLS_L, direction)
        right = (h_rows, h_pool,
                 "Nyamsuren & Tsay  (their values, same papers only)",
                 f"Same papers as left: g = {fmt(h_pool.estimate, 2)} [{fmt(h_pool.ci_low, 2)}; "
                 f"{fmt(h_pool.ci_high, 2)}], k={h_pool.k}   ·   their full Fig 4 panel: "
                 f"g = {fmt(h_pool_full.estimate, 2)} [{fmt(h_pool_full.ci_low, 2)}; "
                 f"{fmt(h_pool_full.ci_high, 2)}], k={h_pool_full.k}",
                 "their published per-dataset values, transcribed; rows for papers absent from "
                 "the left panel are dropped — the foot says why each paper is absent",
                 (-4, 4), COLS_R, direction)
        bands.append((name, left, right, len(c_rows), len(h_rows)))

    rows_max = max(max(b[3], b[4]) for b in bands)
    fig = plt.figure(figsize=(17.4, 15.2), dpi=200)
    fig.patch.set_facecolor("white")
    heights = []
    total_rows = sum(max(b[3], b[4]) + 14 for b in bands)
    y = 0.995
    for name, left, right, nl, nr in bands:
        h = (max(nl, nr) + 14) / total_rows * 0.865
        band(fig, y, h, name, left, right)
        y -= h + 0.012
    foot = (
        "★ = the row's values were entered by the reviewer during the run's logged review, from the papers themselves.   Why the tool needed a human there:\n"
        "      ◆ it had ALREADY read the same numbers (within ~3%) — the question was only a clarification: which series is which arm, which direction the measure points, whether an\n"
        "         unlabelled error bar is SD or SE, or the analysed group size.      ▲ it could not get a number off the paper at all (no readable series or passage).\n"
        "      ● it read a real number, but from a different series, epoch, panel or column than the paper supports; the reviewer re-read it.\n"
        "☆ = one of the two groups was entered.      no mark = AutoMeta's own extraction — a reviewer may have confirmed or re-signed it, but the numbers on the row are the tool's.\n"
        "\"Read from\" = the figure, table or passage the tool itself located for this cell, in the paper's own numbering, recorded during the run.  On ★ rows the tool identified this\n"
        "source but the plotted value was typed in by the reviewer; blank = the tool located nothing.\n"
        "Right panels show her rows only for papers plotted at left.  Absent from late adaptation: Jones 2020 (uploaded PDF reviewer-confirmed ineligible — it measures steady\n"
        "performance under non-standard mappings, with no adaptation phase), Striemer & Morrill 2023 (its late-adaptation cells are honestly held: near-zero signed last-5 errors\n"
        "whose sign convention flips with prism direction, and no dispersion printed), Schabowsky 2007 (paper prints no late-phase dispersion — recorded exclusion).\n"
        "Absent from aftereffect: Jones 2020 (same) and Kumar & Mutha 2023 (the run mapped no aftereffect outcome for it).\n"
        "Both sides are pooled cluster-robust (robumeta-style CORR fit, ρ = 0.8, CR2 small-sample SE, t on Satterthwaite df): the AutoMeta panels cluster rows by paper (the tool's\n"
        "conservative default); the right panels use the preprint's own per-dataset convention, established against its printed weights.\n"
        "The best-guess line is not the primary analysis: it adds rows still held for open review questions, each at its current recorded value (the marks above say whose read that is).")
    fig.text(0.015, 0.004, foot, fontsize=7.0, color=MUTED, va="bottom")
    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
