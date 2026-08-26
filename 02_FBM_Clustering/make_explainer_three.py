"""argmax vs threshold, convex NMF vs semi-NMF, and graded vs convex."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path

OUT = (Path(__file__).resolve().parent / "outputs" / "clustering"
       / "explainers" / "E6_three_distinctions.png")
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
BLUE, RED, GREEN, ORANGE, PURPLE = "#4a6fa5", "#c1121f", "#1b7837", "#e08214", "#5b2c83"
COMP = [PURPLE, BLUE, GREEN]
MONO = {"family": "DejaVu Sans Mono"}
TAU = 0.25

G = np.array([[0.86, 0.10, 0.04],
              [0.52, 0.44, 0.04],
              [0.41, 0.33, 0.26],
              [0.30, 0.62, 0.08],
              [0.05, 0.28, 0.67],
              [0.34, 0.31, 0.35]])
sizes_a = [int(v) for v in np.bincount(G.argmax(1), minlength=3)]
sizes_t = [int(v) for v in (G >= TAU).sum(0)]

fig = plt.figure(figsize=(16.4, 14.6), dpi=165)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16.4); ax.set_ylim(0, 15.6); ax.axis("off")
fig.text(0.032, 0.988, "Three distinctions that are easy to run together",
         fontsize=21, color=INK, va="top")
fig.text(0.032, 0.968,
         "All three concern the same object — the LOADING MATRIX a graded model returns "
         "— but they are independent questions.",
         fontsize=10.2, color=MUTED, va="top")

# ══════════════════════════ 1 · argmax vs threshold ══════════════════════════
ax.text(0.4, 14.55, "1 · ARGMAX vs THRESHOLD  —  the same loadings, read two ways",
        fontsize=14, color=INK)
ax.text(0.4, 14.22, "The model gives every electrode a weight on every component. "
        "Neither reading changes the model — they change what you COUNT.",
        fontsize=9.4, color=MUTED)

y0, ch, cw, x0 = 11.30, 0.36, 0.95, 1.15
top = y0 + 6 * ch                                   # 13.46
ax.text(x0 - 0.20, top + 0.40, "loadings (rows sum to 1)", fontsize=9, color=INK)
for j in range(3):
    ax.text(x0 + j * cw + cw / 2, top + 0.10, f"c{j}", ha="center", fontsize=9,
            color=COMP[j], fontweight="bold")
for i in range(6):
    yy = y0 + (5 - i) * ch
    ax.text(x0 - 0.14, yy + ch / 2, f"e{i}", ha="right", va="center", fontsize=8.6, color=INK)
    for j in range(3):
        v = G[i, j]
        ax.add_patch(Rectangle((x0 + j * cw, yy), cw, ch, fc=COMP[j],
                               alpha=0.10 + 0.75 * v, ec="white", lw=1.4))
        ax.text(x0 + j * cw + cw / 2, yy + ch / 2, f"{v:.2f}", ha="center", va="center",
                fontsize=8.2, color="white" if v > 0.5 else INK)

ax.text(4.55, top + 0.40, "ARGMAX", fontsize=10.5, color=INK, fontweight="bold")
ax.text(4.55, top + 0.13, "keep the largest, discard the rest", fontsize=8.4, color=MUTED)
for i in range(6):
    yy = y0 + (5 - i) * ch
    j = int(G[i].argmax())
    ax.add_patch(Rectangle((4.55, yy + 0.04), 0.75, ch - 0.08, fc=COMP[j], ec="none"))
    ax.text(4.92, yy + ch / 2, f"c{j}", ha="center", va="center", fontsize=8.4,
            color="white", fontweight="bold")
    if G[i].max() < 0.5:
        ax.text(5.42, yy + ch / 2, f"won on {G[i].max():.2f}", va="center",
                fontsize=7.6, color=RED)

ax.text(7.45, top + 0.40, f"THRESHOLD  ≥ {TAU}", fontsize=10.5, color=INK, fontweight="bold")
ax.text(7.45, top + 0.13, "keep every component it really carries", fontsize=8.4, color=MUTED)
for i in range(6):
    yy = y0 + (5 - i) * ch
    hits = [j for j in range(3) if G[i, j] >= TAU]
    for n, j in enumerate(hits):
        ax.add_patch(Rectangle((7.45 + n * 0.82, yy + 0.04), 0.75, ch - 0.08,
                               fc=COMP[j], ec="none"))
        ax.text(7.45 + n * 0.82 + 0.37, yy + ch / 2, f"c{j}", ha="center", va="center",
                fontsize=8.4, color="white", fontweight="bold")
    if len(hits) > 1:
        ax.text(7.45 + len(hits) * 0.82 + 0.08, yy + ch / 2, f"{len(hits)} memberships",
                va="center", fontsize=7.6, color=GREEN)

ax.text(11.15, top + 0.40, "what you end up counting", fontsize=10, color=INK)
ax.text(11.30, top - 0.15, f"argmax     sizes {sizes_a}   total {sum(sizes_a)} = n",
        fontsize=8.8, color=INK, **MONO)
ax.text(11.30, top - 0.48, f"threshold  sizes {sizes_t}   total {sum(sizes_t)} > n",
        fontsize=8.8, color=GREEN, **MONO)
ax.text(11.30, top - 0.92,
        "Argmax PARTITIONS: every electrode once, groups\n"
        "mutually exclusive. That is what makes ARI, silhouette\n"
        "and a contingency table computable at all.",
        fontsize=8.5, color=MUTED, va="top", linespacing=1.5)
ax.text(11.30, top - 1.72,
        "Threshold does NOT partition. Sizes overrun n, and\n"
        "ARI / silhouette stop applying — overlap-aware\n"
        "measures are needed instead.",
        fontsize=8.5, color=RED, va="top", linespacing=1.5)

ax.text(0.4, 10.90, "ON YOUR DATA this is not a small difference.", fontsize=10.4, color=RED)
ax.text(0.4, 10.60,
        "Median top weight 0.43; only 34% of electrodes have a majority; 19% sit within "
        "0.05 of a tie. Component 2 held 97 electrodes by argmax and 583 above a 0.08 "
        "loading. That gap IS the argmax's cost — a taxonomy",
        fontsize=9, color=INK)
ax.text(0.4, 10.32,
        "reports the first number and discards the second. Neither is wrong: argmax asks "
        "\"which component leads here\", threshold asks \"which components does this "
        "electrode carry\".", fontsize=9, color=INK)
ax.plot([0.3, 16.1], [10.02, 10.02], color=GREY, lw=1.2)

# ══════════════════════════ 2 · convex vs semi ══════════════════════════
ax.text(0.4, 9.62, "2 · CONVEX NMF vs SEMI-NMF  —  where the components are allowed to sit",
        fontsize=14, color=INK)
ax.text(0.4, 9.30, "Ding, Li & Jordan (2010) introduce both in one paper. They differ in "
        "exactly one constraint.", fontsize=9.4, color=MUTED)

rng = np.random.default_rng(4)
P = np.vstack([rng.normal([2.6, 2.4], 0.52, (26, 2)),
               rng.normal([5.6, 3.1], 0.52, (26, 2)),
               rng.normal([4.0, 5.8], 0.52, (26, 2))])
SC, BY = 0.40, 5.62                       # scatter occupies y 5.9 .. 8.6

for panel, (px, title, col) in enumerate([
        (0.6, "SEMI-NMF     X ≈ F G'      G ≥ 0,  F FREE", ORANGE),
        (8.5, "CONVEX NMF   X ≈ (XW) G'   G ≥ 0,  W ≥ 0", PURPLE)]):
    bx = px + 0.5
    ax.text(px, 8.92, title, fontsize=10.4, color=col, **MONO)
    ax.scatter(bx + P[:, 0] * SC, BY + P[:, 1] * SC, s=11, color=GREY, zorder=2)
    if panel:
        hull = np.array([[1.6, 1.4], [6.7, 2.0], [5.2, 6.8], [2.5, 5.5]])
        ax.add_patch(Polygon(np.c_[bx + hull[:, 0] * SC, BY + hull[:, 1] * SC],
                             closed=True, fill=False, ec=PURPLE, lw=1.3, ls="--", zorder=1))
        ax.text(bx + 1.4 * SC, BY + 7.25 * SC, "convex hull of the electrodes",
                fontsize=7.8, color=PURPLE)
        for j, c in enumerate(np.array([[2.7, 2.5], [5.5, 3.0], [4.0, 5.6]])):
            for m in P[np.argsort(((P - c) ** 2).sum(1))[:7]]:
                ax.plot([bx + c[0] * SC, bx + m[0] * SC], [BY + c[1] * SC, BY + m[1] * SC],
                        color=COMP[j], lw=0.7, alpha=0.45, zorder=2)
            ax.scatter([bx + c[0] * SC], [BY + c[1] * SC], s=140, marker="*",
                       color=COMP[j], ec="white", lw=1.1, zorder=4)
        ax.text(px, 5.62,
                "Each component is forced to be a WEIGHTED AVERAGE of real electrodes\n"
                "(the thin lines). It cannot leave the hull, so it is itself a response\n"
                "profile you could have recorded — which is why FIG C.3 B1 is in real dB.",
                fontsize=8.6, color=INK, va="top", linespacing=1.5)
        ax.text(px, 4.68,
                "Cost: components are pulled toward dense regions and cannot express an\n"
                "extreme that no electrode occupies.",
                fontsize=8.6, color=MUTED, va="top", linespacing=1.5)
    else:
        for j, c in enumerate(np.array([[0.6, 4.9], [7.4, 1.0], [4.0, 5.6]])):
            ax.scatter([bx + c[0] * SC], [BY + c[1] * SC], s=140, marker="*",
                       color=COMP[j], ec="white", lw=1.1, zorder=4)
        ax.text(bx + 0.6 * SC, BY + 5.6 * SC, "outside the data", fontsize=7.6,
                color=RED, ha="center")
        ax.text(px, 5.62,
                "F is unconstrained in sign and position. A component can sit anywhere in\n"
                "feature space, including where no electrode is — it is a direction, not a\n"
                "recordable response.",
                fontsize=8.6, color=INK, va="top", linespacing=1.5)
        ax.text(px, 4.68,
                "Gain: handles signed dB with no A+/A− split, and can express a pattern\n"
                "lying outside the observed cloud.",
                fontsize=8.6, color=MUTED, va="top", linespacing=1.5)

ax.text(0.4, 4.16, "In BOTH, the loadings G are non-negative — so both are graded, and an "
        "electrode still carries a weight on every component. Only the components' freedom "
        "differs.", fontsize=9.2, color=INK)
ax.plot([0.3, 16.1], [3.86, 3.86], color=GREY, lw=1.2)

# ══════════════════════════ 3 · graded vs convex ══════════════════════════
ax.text(0.4, 3.62, "3 · GRADED vs CONVEX  —  two independent axes, not one scale",
        fontsize=14, color=INK)
ax.text(0.4, 3.34, "GRADED is about the ELECTRODE: one label, or a weight on every "
        "component?    CONVEX is about the COMPONENT: a weighted average of real "
        "electrodes, or any vector?", fontsize=9.4, color=MUTED)

cx, cy, cw2, ch2 = 3.5, 0.98, 5.6, 0.92
ax.text(cx + cw2 * 0.5, 2.99, "HARD membership\none label per electrode",
        ha="center", fontsize=9.2, color=INK, linespacing=1.4)
ax.text(cx + cw2 * 1.5, 2.99, "GRADED membership\na weight on every component",
        ha="center", fontsize=9.2, color=INK, linespacing=1.4)
cells = [[("k-means, Ward", "the centroid IS the mean of its members,\nso it already lies inside the hull", GREEN),
          ("CONVEX NMF, fuzzy c-means, archetypal analysis", "your convex NMF is this box", PURPLE)],
         [("(rare — little reason to want it)", "", GREY),
          ("semi-NMF, PCA, ICA", "free directions; PCA loadings can be NEGATIVE,\nso they are not memberships at all", ORANGE)]]
for r, rowlab in enumerate(["YES\nconvex", "NO\nfree"]):
    ax.text(cx - 0.18, cy + (1 - r) * ch2 + ch2 / 2, rowlab, ha="right", va="center",
            fontsize=9, color=INK, linespacing=1.4)
    for c in range(2):
        name, note, col = cells[r][c]
        yy = cy + (1 - r) * ch2
        ax.add_patch(Rectangle((cx + c * cw2, yy), cw2, ch2, fc="white", ec=col, lw=1.8))
        ax.text(cx + c * cw2 + 0.14, yy + ch2 - 0.16, name, fontsize=9, color=col,
                va="top", fontweight="bold")
        if note:
            ax.text(cx + c * cw2 + 0.14, yy + ch2 - 0.42, note, fontsize=7.8,
                    color=MUTED, va="top", linespacing=1.4)
ax.text(cx - 0.18, 2.99, "component is a weighted\naverage of real electrodes",
        ha="right", fontsize=8.8, color=INK, va="center", linespacing=1.4)

ax.text(0.4, 0.58, "WHY THIS MATTERS: k-means is ALREADY convex — its centroid is the mean "
        "of its members. So convex NMF's advantage over k-means is not convexity, it is "
        "GRADEDNESS.", fontsize=9.2, color=INK)
ax.text(0.4, 0.28, "And semi-NMF is graded WITHOUT being convex — which makes it the clean "
        "test of whether the convex constraint buys anything beyond the graded loadings.",
        fontsize=9.2, color=PURPLE)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=165, bbox_inches="tight", facecolor="white")
print(f"argmax sizes {sizes_a} total {sum(sizes_a)}   threshold sizes {sizes_t} total {sum(sizes_t)}")
print(f"-> {OUT}")
