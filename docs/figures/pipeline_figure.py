"""The AutoMeta pipeline, as one publication figure.

Regenerate with:  .venv/bin/python docs/figures/pipeline_figure.py
Writes autometa_pipeline.png (600 dpi) and autometa_pipeline.pdf beside this file.

Reading order for someone who knows nothing: you provide two things (left), every
paper walks five stages (middle), the studies are pooled (right), and a human sits
under the whole middle band answering the machine's open questions, which re-pool
the plot. The mini forest in the result panel shows the two analysis lines.
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

# ── palette: the product's own tokens ────────────────────────────────────────
NAVY = "#25344a"
INK = "#1a2332"
SOFT = "#3e4b62"
MUTED = "#5b6980"
RULE = "#c4cdd8"
ACCENT = "#2a78d6"
AMBER = "#8d5c07"
AMBER_EDGE = "#d9bc7a"
AMBER_WASH = "#f9f0d9"
GROUP_WASH = "#f4f6f9"
SUNK = "#e9edf2"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

W, H = 114.0, 66.0
fig, ax = plt.subplots(figsize=(11.4, 6.6))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def stage_box(x, y, w, h, number, name, body, footer=None, wrap=20,
              head_fill=NAVY, head_text="white"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=RULE,
                           linewidth=0.9, zorder=3))
    head_h = 4.6
    ax.add_patch(Rectangle((x, y + h - head_h), w, head_h, facecolor=head_fill,
                           edgecolor="none", zorder=4))
    label = f"{number}  ·  {name}" if number else name
    ax.text(x + w / 2, y + h - head_h / 2, label, ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=head_text, zorder=5)
    ax.text(x + w / 2, y + h - head_h - 1.6, textwrap.fill(body, wrap),
            ha="center", va="top", fontsize=6.9, color=SOFT, linespacing=1.45,
            zorder=5)
    if footer:
        ax.text(x + w / 2, y + 1.7, textwrap.fill(footer, wrap), ha="center",
                va="bottom", fontsize=6.2, color=MUTED, style="italic",
                linespacing=1.3, zorder=5)


def arrow(p, q, color=MUTED, lw=1.3, rad=0.0, z=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


# ── title ────────────────────────────────────────────────────────────────────
ax.text(3, 63.4, "The AutoMeta pipeline", fontsize=15, fontweight="bold",
        color=NAVY, va="center")
ax.text(3, 60.4, "From a folder of PDFs to an auditable meta-analysis — "
        "with a human over every doubtful number.", fontsize=9.2, color=MUTED,
        va="center")

# ── inputs ───────────────────────────────────────────────────────────────────
ax.text(3, 54.6, "YOU PROVIDE", fontsize=7.2, fontweight="bold", color=MUTED,
        va="center")
stage_box(3, 41.5, 16.5, 11.5, "", "THE PAPERS",
          "A folder of PDFs — the studies to be reviewed.",
          wrap=24, head_fill=SUNK, head_text=NAVY)
stage_box(3, 27.0, 16.5, 12.5, "", "THE PROTOCOL",
          "One page of rules: the two groups compared, the outcomes, "
          "who is eligible.",
          wrap=24, head_fill=SUNK, head_text=NAVY)

# ── the per-paper band ───────────────────────────────────────────────────────
BX, BW = 23.5, 60.0   # container
ax.add_patch(Rectangle((BX, 24.0), BW, 31.0, facecolor=GROUP_WASH,
                       edgecolor=RULE, linewidth=1.0, linestyle=(0, (4, 3)),
                       zorder=2))
ax.text(BX + 1.6, 52.9, "AUTOMETA READS EVERY PAPER — FIVE STAGES, IN ORDER",
        fontsize=7.2, fontweight="bold", color=NAVY, va="center", zorder=5)

stages = [
    ("1", "INGEST",
     "The PDF is split into pages, figures and tables, each fingerprinted "
     "so it can be traced.", None),
    ("2", "MAP",
     "Agents catalogue what the paper contains and apply the protocol's "
     "eligibility rules.", None),
    ("3", "EXTRACT",
     "The numbers are read from the best source: text, then tables, then "
     "figures, then statistics.", "7 reading routes, best first"),
    ("4", "VERIFY",
     "A second model reads independently, and a checker tries to refute "
     "every value.", "disagreement becomes a question"),
    ("5", "RESOLVE",
     "A deterministic statistics engine converts what was read into one "
     "standard effect size.", "Hedges' g ± its precision"),
]
inner_x, inner_w, gap = BX + 1.5, BW - 3.0, 1.2
box_w = (inner_w - gap * 4) / 5
by, bh = 26.0, 24.5
for i, (num, name, body, foot) in enumerate(stages):
    x = inner_x + i * (box_w + gap)
    stage_box(x, by, box_w, bh, num, name, body, foot, wrap=19)
    if i:
        arrow((x - gap - 0.05, by + bh / 2), (x + 0.05, by + bh / 2),
              color=NAVY, lw=1.1)

# inputs → band
arrow((19.5, 47.2), (BX, 47.2), color=NAVY)
arrow((19.5, 33.2), (BX, 33.2), color=NAVY)

# ── pool ─────────────────────────────────────────────────────────────────────
stage_box(88, 42.5, 23, 12.5, "6", "POOL",
          "Random-effects meta-analysis combines every study; larger, more "
          "precise studies weigh more.", wrap=34)
arrow((BX + BW, 48.7), (88, 48.7), color=NAVY)

# ── result panel with a miniature forest plot ────────────────────────────────
RX, RY, RW, RH = 88, 5.0, 23, 33.5
ax.add_patch(Rectangle((RX, RY), RW, RH, facecolor="white", edgecolor=RULE,
                       linewidth=0.9, zorder=3))
ax.add_patch(Rectangle((RX, RY + RH - 4.6), RW, 4.6, facecolor=NAVY,
                       edgecolor="none", zorder=4))
ax.text(RX + RW / 2, RY + RH - 2.3, "RESULT", ha="center", va="center",
        fontsize=7.6, fontweight="bold", color="white", zorder=5)
arrow((99.5, 42.5), (99.5, RY + RH + 0.05), color=NAVY)

# mini forest: g in [-1, 1] mapped onto x ∈ [96.5, 109]
fx0, fx1 = 96.5, 109.0
def gx(g):
    return fx0 + (g + 1) / 2 * (fx1 - fx0)

axis_y = 13.2
ax.plot([gx(0), gx(0)], [axis_y, 32.3], color=RULE, lw=0.8, ls=(0, (2, 2)),
        zorder=4)
rows = [("Paper 1", -0.62, 0.34), ("Paper 2", 0.18, 0.30),
        ("Paper 3", -0.38, 0.44), ("Paper 4", -0.12, 0.24)]
for i, (label, g, half) in enumerate(rows):
    ry = 30.6 - i * 3.1
    ax.text(90.0, ry, label, fontsize=5.6, color=MUTED, va="center", zorder=5)
    ax.plot([gx(g - half), gx(g + half)], [ry, ry], color=SOFT, lw=0.9,
            zorder=5)
    ax.add_patch(Rectangle((gx(g) - 0.55, ry - 0.55), 1.1, 1.1,
                           facecolor=ACCENT, edgecolor="none", zorder=6))

def diamond(cy, cg, half, **kw):
    pts = [(gx(cg - half), cy), (gx(cg), cy + 0.85),
           (gx(cg + half), cy), (gx(cg), cy - 0.85)]
    ax.add_patch(Polygon(pts, closed=True, zorder=6, **kw))

diamond(17.6, -0.20, 0.28, facecolor=NAVY, edgecolor="none")
ax.text(90.0, 17.6, "strict", fontsize=5.6, color=NAVY, va="center",
        fontweight="bold", zorder=5)
diamond(15.0, -0.14, 0.25, facecolor="none", edgecolor=AMBER, linewidth=0.9,
        linestyle=(0, (2.2, 1.4)))
ax.text(90.0, 15.0, "best-guess", fontsize=5.6, color=AMBER, va="center",
        zorder=5)

ax.plot([fx0, fx1], [axis_y, axis_y], color=MUTED, lw=0.8, zorder=5)
for g, t in [(-1, "−1"), (0, "0"), (1, "1")]:
    ax.plot([gx(g), gx(g)], [axis_y - 0.45, axis_y], color=MUTED, lw=0.8,
            zorder=5)
    ax.text(gx(g), axis_y - 1.0, t, ha="center", va="top", fontsize=5.4,
            color=MUTED, zorder=5)
ax.text((fx0 + fx1) / 2, axis_y - 3.0, "effect size (g)", ha="center",
        va="top", fontsize=5.8, color=MUTED, zorder=5)

ax.text(RX + RW / 2, 7.2, textwrap.fill(
    "A forest plot, a written conclusion, and a full audit trail.", 40),
    ha="center", va="center", fontsize=6.7, color=SOFT, linespacing=1.4,
    zorder=5)

# ── the human loop ───────────────────────────────────────────────────────────
HX, HY, HW, HH = BX, 5.0, BW, 14.0
ax.add_patch(Rectangle((HX, HY), HW, HH, facecolor=AMBER_WASH,
                       edgecolor=AMBER_EDGE, linewidth=1.0, zorder=3))
ax.text(HX + 2.0, HY + HH - 2.6, "HUMAN REVIEW — THE MACHINE NEVER HAS THE "
        "LAST WORD", fontsize=7.4, fontweight="bold", color=AMBER,
        va="center", zorder=5)
ax.text(HX + 2.0, HY + HH - 5.2, textwrap.fill(
    "Anything uncertain becomes a question card: the page image beside every "
    "reading the agents weighed, and why they could not decide. A person's "
    "answer overrides the machine, is recorded permanently, and the plot "
    "re-pools at once.", 92),
    ha="left", va="top", fontsize=6.9, color=SOFT, linespacing=1.5, zorder=5)

arrow((66.5, by - 0.05), (66.5, HY + HH + 0.05), color=AMBER)
ax.text(67.6, (by + HY + HH) / 2, "open questions", fontsize=6.0, color=AMBER,
        va="center", ha="left")
arrow((HX + HW, 9.0), (RX, 9.0), color=AMBER)
ax.text((HX + HW + RX) / 2, 10.0, "answers\nre-pool", ha="center", va="bottom",
        fontsize=5.6, color=AMBER, linespacing=1.2)

# ── footnote ─────────────────────────────────────────────────────────────────
ax.text(3, 2.0, "Every value on the plot traces to the exact sentence, table "
        "cell or figure pixel it was read from; every decision, human or "
        "machine, is kept in an append-only log.",
        fontsize=6.6, color=MUTED, va="center", style="italic")

out = Path(__file__).resolve().parent
fig.savefig(out / "autometa_pipeline.png", dpi=600, facecolor="white")
fig.savefig(out / "autometa_pipeline.pdf", facecolor="white")
print("wrote", out / "autometa_pipeline.png")
