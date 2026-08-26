"""What the _half1 / _half2 files are, drawn."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path(__file__).resolve().parent / "E4_what_the_halves_are.png"
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
BLUE, RED, GREEN, ORANGE = "#4a6fa5", "#c1121f", "#1b7837", "#e08214"
MONO = {"family": "DejaVu Sans Mono"}
rng = np.random.default_rng(7)


def ersp_patch(ax, x, y, w, h, seed, noise=0.0, label=None, col=INK, lw=1.4):
    """A little time-frequency tile that looks like an ERSP."""
    r = np.random.default_rng(seed)
    nf, nt = 22, 34
    t = np.linspace(0, 1, nt)
    f = np.linspace(0, 1, nf)
    blob = np.exp(-((t - 0.45) ** 2) / 0.02)[None, :] * np.exp(-((f - 0.72) ** 2) / 0.05)[:, None]
    blob -= 0.6 * np.exp(-((t - 0.72) ** 2) / 0.03)[None, :] * np.exp(-((f - 0.22) ** 2) / 0.04)[:, None]
    img = blob * 2.4 + r.normal(0, 0.35 + noise, (nf, nt))
    ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto", cmap="bwr",
              vmin=-2.5, vmax=2.5, zorder=2, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=lw, zorder=3))
    if label:
        ax.text(x + w / 2, y - 0.11, label, ha="center", va="top", fontsize=7.6,
                color=col, zorder=4)


def arrow(ax, x0, y0, x1, y1, col=MUTED, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, lw=lw, color=col, zorder=5))


def trials(ax, x, y, w, n, picks, hi=None, lab=None):
    """n little trial rows; `picks` indexes get highlighted."""
    gap = 0.052
    for i in range(n):
        yy = y - i * gap
        c = hi if (hi and i in picks) else GREY
        ax.plot([x, x + w], [yy, yy], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        if i in picks and hi:
            ax.text(x - 0.045, yy, f"{i+1}", fontsize=6, color=c, ha="right", va="center")
    if lab:
        ax.text(x + w / 2, y + 0.075, lab, ha="center", fontsize=7.8, color=INK)


fig = plt.figure(figsize=(15.6, 11.6), dpi=170)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")

fig.text(0.035, 0.972, "What the _half1 / _half2 files are",
         fontsize=21, color=INK, va="top")
fig.text(0.035, 0.941,
         "One recording session gives you many TRIALS of the same condition. An ERSP is what you get "
         "after averaging them.\nThe halves are the same averaging done twice, on two independent "
         "sets of trials, so the result can be checked against itself.",
         fontsize=10.4, color=MUTED, va="top", linespacing=1.55)

# ══════════════ A — the old way ══════════════
ax.text(0.5, 10.35, "A · THE OLD WAY — one average, one file", fontsize=12.5, color=INK)
ax.text(0.5, 10.05, "every trial goes into one pot", fontsize=9, color=MUTED)
trials(ax, 0.7, 9.6, 1.5, 10, set(), lab="all trials")
ax.text(1.45, 9.05, "…", ha="center", fontsize=13, color=GREY)
arrow(ax, 2.45, 9.35, 3.3, 9.35)
ax.text(2.87, 9.5, "mean", ha="center", fontsize=8.4, color=MUTED)
ersp_patch(ax, 3.5, 8.85, 1.5, 1.05, seed=1, label=None, col=INK, lw=1.8)
ax.text(4.25, 8.66, "the ERSP cube   (129 freq × 300 time)", ha="center", fontsize=8.2, color=INK)
ax.text(4.25, 8.40, "EL033_audio_WM_ERSP_A_L10_TN.npy", ha="center", fontsize=8, color=INK, **MONO)
ax.text(0.5, 8.02, "This is the only file that existed before. It is what every clustering run has always read.",
        fontsize=9, color=MUTED)

ax.plot([0.3, 15.7], [7.72, 7.72], color=GREY, lw=1)

# ══════════════ B — what was added ══════════════
ax.text(0.5, 7.42, "B · WHAT I ADDED — the same trials, split odd / even, averaged twice more",
        fontsize=12.5, color=INK)
ax.text(0.5, 7.14,
        "The per-trial stack already existed inside compute_ersp; it was collapsed to the mean\n"
        "and thrown away, so no split-half estimate was possible from anything on disk.",
        fontsize=9, color=MUTED, va="top", linespacing=1.5)

trials(ax, 0.7, 6.65, 1.5, 10, {0, 2, 4, 6, 8}, hi=ORANGE, lab="odd trials  1,3,5…")
trials(ax, 2.9, 6.65, 1.5, 10, {1, 3, 5, 7, 9}, hi=GREEN, lab="even trials  2,4,6…")

arrow(ax, 2.35, 6.4, 2.75, 6.4, col=ORANGE)
arrow(ax, 4.55, 6.4, 4.95, 6.4, col=GREEN)

ersp_patch(ax, 5.15, 6.4, 1.35, 0.95, seed=2, noise=0.55, col=ORANGE, lw=1.8)
ax.text(5.82, 6.22, "half1", ha="center", fontsize=8.6, color=ORANGE)
ax.text(5.82, 5.99, "…_A_L10_TN_half1.npy", ha="center", fontsize=7.4, color=ORANGE, **MONO)

ersp_patch(ax, 6.95, 6.4, 1.35, 0.95, seed=3, noise=0.55, col=GREEN, lw=1.8)
ax.text(7.62, 6.22, "half2", ha="center", fontsize=8.6, color=GREEN)
ax.text(7.62, 5.99, "…_A_L10_TN_half2.npy", ha="center", fontsize=7.4, color=GREEN, **MONO)

ax.text(8.75, 7.16, "ODD / EVEN, not first-half / second-half.", fontsize=9.6,
        color=INK, va="top")
ax.text(8.75, 6.94,
        "Drift, fatigue and block structure spread evenly across the two\n"
        "halves instead of landing entirely in one of them. Splitting\n"
        "1–25 vs 26–50 would confound 'reproducible' with 'the patient\n"
        "got tired'. Norman-Haignere split odd/even runs for the same reason.",
        fontsize=8.6, color=MUTED, linespacing=1.5, va="top")

ax.text(8.75, 5.72, "Each half averages HALF the trials, so each is noisier than the full cube — "
        "that matters in D.", fontsize=8.8, color=RED)

ax.plot([0.3, 15.7], [5.46, 5.46], color=GREY, lw=1)

# ══════════════ C — what they are FOR ══════════════
ax.text(0.5, 5.16, "C · WHAT THEY ARE FOR — a reliability gate instead of a loudness gate",
        fontsize=12.5, color=INK)

ersp_patch(ax, 0.8, 4.05, 1.25, 0.88, seed=2, noise=0.55, col=ORANGE, lw=1.6)
ersp_patch(ax, 2.45, 4.05, 1.25, 0.88, seed=3, noise=0.55, col=GREEN, lw=1.6)
ax.text(2.25, 4.49, "vs", ha="center", fontsize=11, color=INK)
ax.text(2.25, 3.86, "correlate the two halves,  bin by bin", ha="center", fontsize=8.6, color=INK)

ax.text(4.3, 4.82, "r high", fontsize=9.6, color=GREEN)
ax.text(4.95, 4.82, "→  the response repeats on trials it has never seen  →  KEEP", fontsize=9.2, color=INK)
ax.text(4.3, 4.50, "r low", fontsize=9.6, color=RED)
ax.text(4.95, 4.50, "→  it was noise  →  DROP", fontsize=9.2, color=INK)
ax.text(4.3, 4.16,
        "The gate in use today keeps an electrode if enough of its time-frequency plane exceeds a\n"
        "threshold — that selects LOUD electrodes. This selects REPRODUCIBLE ones, which is the\n"
        "question you actually want answered. Validated on planted data: a repeatable response\n"
        "scores r = +0.908, a pure-noise channel r = +0.037.",
        fontsize=8.6, color=MUTED, linespacing=1.5, va="top")

ax.plot([0.3, 15.7], [3.42, 3.42], color=GREY, lw=1)

# ══════════════ D — the bug ══════════════
ax.text(0.5, 3.12, "D · WHY THIS BROKE THE COHORT COUNT", fontsize=12.5, color=RED)
ax.text(0.5, 2.84, "All three files sit in the SAME folder, and the dataset builder globs *.npy:",
        fontsize=9.4, color=INK)

fold = [("EL033_audio_WM_ERSP_A_L10_TN.npy", INK, "the real cube"),
        ("EL033_audio_WM_ERSP_A_L10_TN_half1.npy", ORANGE, "read as a SEPARATE electrode"),
        ("EL033_audio_WM_ERSP_A_L10_TN_half2.npy", GREEN, "read as a SEPARATE electrode")]
ax.add_patch(Rectangle((0.72, 1.86), 6.4, 0.86, fill=False, ec=GREY, lw=1.2))
ax.text(0.85, 2.63, "ERSP_matrix/audio/", fontsize=8, color=MUTED, **MONO)
for i, (fn, c, note) in enumerate(fold):
    y = 2.42 - i * 0.19
    ax.text(0.95, y, fn, fontsize=7.6, color=c, **MONO)
    ax.text(7.3, y, note, fontsize=8, color=c)

ax.text(0.5, 1.60, "lf_dataset.py:279     npy_files = sorted(cond_dir.glob(\"*.npy\"))",
        fontsize=8.6, color=RED, **MONO)
ax.text(0.5, 1.42, "So one real electrode became THREE rows. And because a half-cube averages half the "
        "trials it is noisier, so it trips the\nloudness threshold far more easily — which is why the "
        "gate suddenly passed almost everything:",
        fontsize=9.2, color=INK, linespacing=1.5, va="top")

tab = [("", "real cubes", "half-cubes"),
       ("rows in the v3 cache", "9,342", "19,380   (67.5%)"),
       ("unique contacts", "3,296", "6,820"),
       ("passed the loudness gate", "35.1%", "96.9%"),
       ("prop_above_pos, median", "0.0124", "0.0480")]
for i, row in enumerate(tab):
    y = 0.86 - i * 0.175
    for j, (txt, xx) in enumerate(zip(row, (0.9, 5.0, 7.0))):
        c = INK if j == 0 else (INK if j == 1 else RED)
        w = "bold" if i == 0 else "normal"
        ax.text(xx, y, txt, fontsize=8.6, color=(MUTED if i == 0 else c), fontweight=w)

ax.text(9.6, 0.86, "3,296 real contacts is essentially the 3,267 the previous cache had.",
        fontsize=9, color=INK)
ax.text(9.6, 0.68, "The cohort did not really triple. Two thirds of it is the same\n"
                   "electrodes counted twice more, under names ending _half1 and _half2.",
        fontsize=9, color=RED, linespacing=1.5, va="top")
ax.text(9.6, 0.16, "EL033 checks out exactly: 74 real channels × 3 conditions × 3 files = 666 .npy",
        fontsize=8.4, color=MUTED)

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"-> {OUT}")
