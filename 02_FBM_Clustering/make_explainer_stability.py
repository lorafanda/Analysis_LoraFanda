"""How sweep_stability computes a number, on six electrodes you can count by hand."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "E5_sweep_stability.png"
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
BLUE, RED, GREEN, ORANGE = "#4a6fa5", "#c1121f", "#1b7837", "#e08214"
MONO = {"family": "DejaVu Sans Mono"}

# six electrodes, four resampling rounds. Each round draws 80% (so some sit out),
# refits k-means, and records who landed together.
N, RUNS = 6, 4
draws = [                      # who was SAMPLED in each round
    [0, 1, 2, 3, 4],
    [0, 1, 2, 4, 5],
    [0, 1, 3, 4, 5],
    [0, 2, 3, 4, 5],
]
labs = [                       # the refit label, for the sampled ones only
    {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
    {0: 0, 1: 0, 2: 1, 4: 1, 5: 1},
    {0: 0, 1: 0, 3: 1, 4: 1, 5: 1},
    {0: 0, 2: 0, 3: 1, 4: 1, 5: 1},
]
FULL = [0, 0, 0, 1, 1, 1]      # the run's own published labels

tog = np.zeros((N, N)); both = np.zeros((N, N))
for d, l in zip(draws, labs):
    for i in d:
        for j in d:
            if i == j:
                continue
            both[i, j] += 1
            if l[i] == l[j]:
                tog[i, j] += 1
C = np.divide(tog, both, out=np.zeros_like(tog), where=both > 0)
np.fill_diagonal(C, 1.0)

jac = {}
for c in (0, 1):
    idx = [i for i in range(N) if FULL[i] == c]
    blk = C[np.ix_(idx, idx)]
    m = ~np.eye(len(idx), dtype=bool)
    jac[c] = float(blk[m].mean())

fig = plt.figure(figsize=(15.4, 9.7), dpi=170)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(-1.25, 9); ax.axis("off")

fig.text(0.035, 0.975, "What sweep_stability.py measures", fontsize=20, color=INK, va="top")
fig.text(0.035, 0.938,
         "Not 'is the fit good' but 'would I get the same clusters again'. It refits the "
         "SAME K many times on random 80% subsamples and asks how often each pair of "
         "electrodes lands together.\n"
         "The run already carries one such number, at its published K. This computes it "
         "at EVERY K in the sweep, because the visualizer lets you re-cut a run and a "
         "stability number quoted under the wrong K is worse than none.",
         fontsize=9.4, color=MUTED, va="top", linespacing=1.6)

# ── A: the rounds ─────────────────────────────────────────────────────────
ax.text(0.4, 7.6, "A · four rounds (the real default is 50, on 80% of electrodes)",
        fontsize=11.6, color=INK)
for r, (d, l) in enumerate(zip(draws, labs)):
    y = 7.05 - r * 0.52
    ax.text(0.4, y, f"round {r+1}", fontsize=8.8, color=MUTED)
    for i in range(N):
        x = 1.85 + i * 0.62
        if i not in d:
            ax.add_patch(Rectangle((x, y - 0.16), 0.48, 0.34, fc="white", ec=GREY,
                                   lw=1.2, ls=":"))
            ax.text(x + 0.24, y, "—", ha="center", va="center", fontsize=9, color=GREY)
        else:
            col = BLUE if l[i] == 0 else ORANGE
            ax.add_patch(Rectangle((x, y - 0.16), 0.48, 0.34, fc=col, ec="none"))
            ax.text(x + 0.24, y, "AB"[l[i]], ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
for i in range(N):
    ax.text(1.85 + i * 0.62 + 0.24, 7.42, f"e{i}", ha="center", fontsize=8.4, color=INK)
ax.text(5.75, 7.05, "dotted = not drawn this round.\nA/B = which cluster it landed in\n"
                    "after refitting.", fontsize=8.6, color=MUTED, va="top", linespacing=1.5)
ax.text(5.75, 6.05, "A pair only counts in a round where BOTH\nwere drawn — otherwise "
                    "sitting out would\nlook like disagreement.",
        fontsize=8.6, color=RED, va="top", linespacing=1.5)

# ── B: the consensus matrix ───────────────────────────────────────────────
ax.text(0.4, 4.75, "B · consensus = times together ÷ times both drawn",
        fontsize=11.6, color=INK)
x0, y0, cell = 1.5, 1.35, 0.46
for i in range(N):
    ax.text(x0 - 0.16, y0 + (N - 1 - i) * cell + cell / 2, f"e{i}", ha="right",
            va="center", fontsize=8.2, color=INK)
    ax.text(x0 + i * cell + cell / 2, y0 + N * cell + 0.10, f"e{i}", ha="center",
            fontsize=8.2, color=INK)
    for j in range(N):
        v = C[i, j]
        yy = y0 + (N - 1 - i) * cell
        ax.add_patch(Rectangle((x0 + j * cell, yy), cell, cell,
                               fc=plt.cm.Blues(0.12 + 0.72 * v), ec="white", lw=1.4))
        ax.text(x0 + j * cell + cell / 2, yy + cell / 2, f"{v:.2f}", ha="center",
                va="center", fontsize=7.2,
                color="white" if v > 0.55 else INK)
for idx, col in ((range(0, 3), BLUE), (range(3, 6), ORANGE)):
    i0 = min(idx); n = len(list(idx))
    ax.add_patch(Rectangle((x0 + i0 * cell, y0 + (N - i0 - n) * cell), n * cell,
                           n * cell, fill=False, ec=col, lw=2.6))

ax.text(4.65, 4.30, "The run's own labels say  e0 e1 e2 = cluster A,  e3 e4 e5 = cluster B.",
        fontsize=9.2, color=INK)
ax.text(4.65, 3.98, "Each cluster's score is the MEAN of its own block, diagonal excluded:",
        fontsize=9.2, color=INK)
ax.text(4.9, 3.58, f"cluster A   mean of the 6 off-diagonal cells in the blue box   "
                   f"=  {jac[0]:.2f}", fontsize=9.4, color=BLUE, **MONO)
ax.text(4.9, 3.24, f"cluster B   mean of the 6 off-diagonal cells in the orange box  "
                   f"=  {jac[1]:.2f}", fontsize=9.4, color=ORANGE, **MONO)
ax.text(4.9, 2.82, f"mean_jaccard = {np.mean(list(jac.values())):.2f}      "
                   f"min_jaccard = {min(jac.values()):.2f}   ← the one to read",
        fontsize=9.6, color=INK, **MONO)

ax.text(4.65, 2.32, "1.0 = every pair together in every round it could have been.\n"
                    "0.5 = together half the time; the cluster is not a stable object.",
        fontsize=9, color=MUTED, va="top", linespacing=1.5)

ax.text(4.65, 1.55, "WHAT IT WRITES, per run:", fontsize=10, color=INK)
for i, (f, d_) in enumerate([
        ("stability_by_k.csv", "one row per K — mean / min / max jaccard"),
        ("stability_by_k/k_11/stability_summary.json", "the same shape the run already had"),
        ("stability_by_k/k_11/per_cluster_stability.csv", "per cluster, so one bad cluster is visible"),
        ("stability_by_k/k_11/consensus_heatmap.png", "the matrix in panel B, full size")]):
    ax.text(4.9, 1.22 - i * 0.28, f, fontsize=8.2, color=INK, **MONO)
    ax.text(10.6, 1.22 - i * 0.28, d_, fontsize=8.4, color=MUTED)

ax.text(0.4, -0.45, "ONE CAVEAT, from the script's own docstring: the resampler always "
        "refits with k-means. For a k-means or Ward run that is exact. On a convex-NMF "
        "run it measures how reproducibly",
        fontsize=8.8, color=RED)
ax.text(0.4, -0.73, "K-MEANS partitions the space the decomposition was fitted in — not "
        "how reproducible the decomposition is. It is kept that way so K=7 keeps "
        "agreeing with the number already published;",
        fontsize=8.8, color=RED)
ax.text(0.4, -1.01, "measure_cluster_stability.py refits the decomposition itself, which "
        "is a different statistic and belongs in its own column.",
        fontsize=8.8, color=RED)

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"cluster A {jac[0]:.3f}   cluster B {jac[1]:.3f}")
print(f"-> {OUT}")
