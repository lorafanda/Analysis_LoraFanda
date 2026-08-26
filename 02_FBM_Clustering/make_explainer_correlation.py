"""Spearman vs Pearson on skewed data - why this project defaults to Spearman.

Every number on the figure is computed, not typed. The two panels are built so that
the answer is obvious by eye before either coefficient is read.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, rankdata

OUT = (Path(__file__).resolve().parent / "outputs" / "clustering" / "explainers"
       / "E7_spearman_vs_pearson.png")
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
BLUE, RED, GREEN, ORANGE, PURPLE = "#4a6fa5", "#c1121f", "#1b7837", "#e08214", "#5b2c83"
MONO = {"family": "DejaVu Sans Mono"}

rng = np.random.default_rng(7)

# ── A: one leverage point ───────────────────────────────────────────────────
# Twelve points with NO relationship, plus one far out along both axes. That single
# point is enough to make Pearson report a strong correlation.
xa = np.concatenate([rng.uniform(0.5, 3.0, 12), [12.0]])
ya = np.concatenate([rng.uniform(0.5, 3.0, 12), [11.0]])
ra_p = pearsonr(xa, ya)[0]
ra_s = spearmanr(xa, ya)[0]
ra_p_drop = pearsonr(xa[:-1], ya[:-1])[0]
ra_s_drop = spearmanr(xa[:-1], ya[:-1])[0]

# ── B: monotone but curved ──────────────────────────────────────────────────
# y rises with x at EVERY step - a perfect monotone relationship - but the shape is
# exponential, so Pearson, which only measures how close the cloud is to a STRAIGHT
# line, cannot report 1.0.
xb = np.linspace(0.2, 3.4, 14)
yb = np.exp(xb)
rb_p = pearsonr(xb, yb)[0]
rb_s = spearmanr(xb, yb)[0]

# ── C: what ranking actually does ───────────────────────────────────────────
vals = np.array([0.01, 0.02, 0.04, 0.09, 0.31, 8.70])
rks = rankdata(vals)

# TALL ON PURPOSE. At 10.6in panel D's box ran into the notes under A and B, and its
# own last row ran into the rule of thumb underneath it.
fig = plt.figure(figsize=(15.4, 12.8), dpi=170)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 15.4)
ax.set_ylim(0, 12.8)
ax.axis("off")

fig.text(0.032, 0.975, "Spearman or Pearson, on skewed data", fontsize=21, color=INK,
         va="top")
fig.text(0.032, 0.936,
         "Both answer \"do these two go up together?\" - but they read different things. "
         "PEARSON works on the VALUES and measures how close the cloud is to a straight "
         "line. SPEARMAN throws the values away,\nreplaces each with its RANK, and runs "
         "Pearson on those. So Spearman asks only \"does y go up when x goes up\", never "
         "\"by how much\" - and that is the whole difference.",
         fontsize=9.6, color=MUTED, va="top", linespacing=1.6)

# ── panel A ─────────────────────────────────────────────────────────────────
axA = fig.add_axes([0.055, 0.505, 0.265, 0.325])
axA.scatter(xa[:-1], ya[:-1], s=44, color=BLUE, zorder=3, ec="white", lw=0.8)
axA.scatter(xa[-1:], ya[-1:], s=110, color=RED, zorder=4, ec="white", lw=1.2)
m, b = np.polyfit(xa, ya, 1)
xs = np.array([0, 13.2])
axA.plot(xs, m * xs + b, color=RED, lw=1.7, ls="-", zorder=2,
         label="Pearson's straight line")
m2, b2 = np.polyfit(xa[:-1], ya[:-1], 1)
axA.plot(np.array([0, 4]), m2 * np.array([0, 4]) + b2, color=GREY, lw=1.6, ls="--",
         zorder=2)
axA.annotate("this ONE point", xy=(12.0, 11.0), xytext=(-14, -30),
             textcoords="offset points", ha="right", fontsize=8.4, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
axA.text(0.6, 4.4, "the other twelve\nhave no relationship\nat all", fontsize=8.2,
         color=MUTED, linespacing=1.45)
axA.set_xlim(0, 13.2)
axA.set_ylim(0, 12.4)
axA.set_title("A · one point off on its own", fontsize=11.4, loc="left", color=INK,
              pad=6)
axA.tick_params(labelsize=7.4, colors=MUTED)
axA.spines[["top", "right"]].set_visible(False)

# ── panel B ─────────────────────────────────────────────────────────────────
axB = fig.add_axes([0.385, 0.505, 0.265, 0.325])
axB.plot(xb, yb, "-o", ms=5.2, color=PURPLE, lw=1.6, zorder=3)
mb, bb = np.polyfit(xb, yb, 1)
axB.plot(xb, mb * xb + bb, color=RED, lw=1.7, zorder=2)
axB.annotate("every step goes UP,\nbut not by the same amount",
             xy=(2.9, np.exp(2.9)), xytext=(-6, -46), textcoords="offset points",
             ha="right", fontsize=8.4, color=PURPLE, linespacing=1.45,
             arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.1))
axB.set_title("B · a perfect relationship that is not a line", fontsize=11.4,
              loc="left", color=INK, pad=6)
axB.tick_params(labelsize=7.4, colors=MUTED)
axB.spines[["top", "right"]].set_visible(False)

# the two verdicts, side by side under their panels
for x0, rp, rs, note in (
        (0.055, ra_p, ra_s,
         f"Pearson is fooled: it reports a strong relationship that exists only because\n"
         f"of the red point. Delete that one point and Pearson drops to "
         f"{ra_p_drop:+.2f}.\nSpearman barely moves ({ra_s:+.2f} to {ra_s_drop:+.2f}) - "
         f"the red point is just\n\"the largest\" on both axes, and being largest is "
         f"worth one rank, not twelve."),
        (0.385, rb_p, rb_s,
         "Spearman says 1.00, and it is RIGHT: y really does rise at every single\n"
         "step. Pearson is asked a different question - \"how close to a straight\n"
         "line?\" - and honestly answers a bit less than 1. Neither is wrong; they\n"
         "are not measuring the same thing.")):
    fig.text(x0, 0.468, f"Pearson  r = {rp:+.2f}", fontsize=11.6,
             color=RED if abs(rp - rs) > 0.15 else INK, **MONO)
    fig.text(x0 + 0.135, 0.468, f"Spearman  ρ = {rs:+.2f}", fontsize=11.6,
             color=GREEN, **MONO)
    fig.text(x0, 0.442, note, fontsize=8.5, color=MUTED, va="top", linespacing=1.55)

# ── panel C: the transform itself ───────────────────────────────────────────
cx, cy = 10.85, 8.98
ax.text(cx, cy + 1.62, "C · what ranking does to a skewed variable", fontsize=11.4,
        color=INK)
ax.text(cx, cy + 1.30, "six values of the kind an atlas probability map is full of — "
        "mostly near zero,", fontsize=8.5, color=MUTED)
ax.text(cx, cy + 1.06, "one far out. Ranking keeps the ORDER and discards the SPACING.",
        fontsize=8.5, color=MUTED)
for i, (v, r) in enumerate(zip(vals, rks)):
    y = cy + 0.60 - i * 0.42
    ax.add_patch(Rectangle((cx, y - 0.15), 1.05, 0.33,
                           fc=plt.cm.Purples(0.10 + 0.72 * v / vals.max()), ec="white",
                           lw=1.3))
    ax.text(cx + 0.52, y, f"{v:.2f}", ha="center", va="center", fontsize=8.4,
            color="white" if v > 3 else INK, **MONO)
    ax.annotate("", xy=(cx + 1.72, y), xytext=(cx + 1.14, y),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.add_patch(Rectangle((cx + 1.80, y - 0.15), 1.05, 0.33,
                           fc=plt.cm.Greens(0.10 + 0.62 * r / rks.max()), ec="white",
                           lw=1.3))
    ax.text(cx + 2.32, y, f"{int(r)}", ha="center", va="center", fontsize=8.4,
            color="white" if r > 4 else INK, **MONO)
ax.text(cx + 0.52, cy + 0.92, "value", ha="center", fontsize=8.6, color=INK)
ax.text(cx + 2.32, cy + 0.92, "rank", ha="center", fontsize=8.6, color=INK)
ax.text(cx, cy - 2.16, f"8.70 is {vals[-1] / vals[-2]:.0f}x the next value.\n"
                       f"As a rank it is simply 6, one step\nabove 5. That is the whole "
                       f"of it.", fontsize=8.5, color=RED, va="top", linespacing=1.5)

# ── panel D: where it lands in this project ─────────────────────────────────
ax.add_patch(Rectangle((0.52, 0.42), 14.4, 4.02, fc="#f6f7f9", ec=GREY, lw=1.1))
ax.text(0.85, 4.10, "D · which one this project uses, and where", fontsize=11.4,
        color=INK)
rows = [
    ("lf_atlas_corr.tf_correlation_map", "spearman", GREEN,
     "the DEFAULT, and the docstring gives the reason: the atlas probability map is "
     "heavily zero-inflated"),
    ("", "", GREEN,
     "and skewed - most voxels are near zero and a few are high, which is panel C - "
     "and ERSP has outliers."),
    ("make_fedorenko_corr.py", "spearman", GREEN,
     "passes method=\"spearman\" explicitly, then Benjamini-Hochberg FDR across the "
     "whole TF plane."),
    ("lf_cluster_timing (size vs onset)", "spearman", GREEN,
     "cluster SIZE against onset - a rank question, and n is small enough that one big "
     "cluster would drag a Pearson."),
    ("lf_cluster_timing (cross-correlation)", "pearson", BLUE,
     "CORRECT here: lag is found by sliding two time courses, and the shape of the "
     "waveform is the signal, not a nuisance."),
    ("make_reliability_gate (split-half)", "pearson", BLUE,
     "correct here too - Spearman-BROWN is a different thing entirely, a length "
     "correction, not a rank correlation."),
]
for i, (fn, meth, col, why) in enumerate(rows):
    y = 3.66 - i * 0.42
    if fn:
        ax.text(0.85, y, fn, fontsize=8.4, color=INK, **MONO)
        ax.text(5.55, y, meth, fontsize=8.4, color=col, fontweight="bold", **MONO)
    ax.text(6.85, y, why, fontsize=8.4, color=MUTED)

ax.text(0.85, 0.78, "RULE OF THUMB: if the QUESTION is \"do they go up together\", use "
        "Spearman. If the SHAPE or the SIZE of the change is the thing you are "
        "measuring - a waveform, a lag, a slope - use Pearson.",
        fontsize=9.0, color=INK)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"A  pearson {ra_p:+.3f}  spearman {ra_s:+.3f}   "
      f"(drop the outlier: {ra_p_drop:+.3f} / {ra_s_drop:+.3f})")
print(f"B  pearson {rb_p:+.3f}  spearman {rb_s:+.3f}")
print(f"-> {OUT}")
