"""What the late broadband feature in PAT_6684 actually is.

Three questions decide between noise, epilepsy and physiology, and each one is a
measurement rather than an opinion:

  A  WHEN. Average the broadband profile over every channel. A focal discharge would
     not survive averaging 138 contacts; a whole-montage event would.
  B  WHAT SHAPE IN FREQUENCY. An interictal spike is a low-frequency event with a
     high-frequency ripple on top, so it lifts the LOW end. Muscle is the opposite:
     broadband from ~50 Hz upward with the low frequencies unchanged or suppressed.
  C  WHERE. Per shaft. Muscle contaminates the contacts nearest the temporalis, jaw and
     orbit; an epileptic focus sits on one or two shafts and does not care about the
     skull.
"""
import glob
import os
import re

PATIENT = "PAT_6684"

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "outputs" / "04_ersp_LM_RAWONLY" / PATIENT / "LM" / "ERSP_matrix"
OUT = Path(__file__).resolve().parent / "outputs" / "_status_png" / "late_broadband_band.png"
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
COND_C = {"audio": "#4a6fa5", "picture": "#c1121f", "reading": "#1b7837"}


def shaft(n):
    m = re.search(r"ERSP_([A-Za-z]+)\d+_TN", n)
    return m.group(1) if m else "?"


data = {}
for cond in ("audio", "picture", "reading"):
    cubes, names = [], []
    for f in sorted(glob.glob(str(ROOT / cond / "*.npy"))):
        cubes.append(np.load(f))
        names.append(os.path.basename(f))
    data[cond] = (np.stack(cubes), names)

nf, nt = data["audio"][0].shape[1:]
freq = np.linspace(0, 500, nf)
x = 100.0 * np.arange(nt) / nt
LATE = slice(int(0.80 * nt), int(0.95 * nt))
EARLY = slice(int(0.05 * nt), int(0.35 * nt))

fig = plt.figure(figsize=(12.4, 4.3), dpi=170)
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.35], wspace=0.32,
                      left=0.055, right=0.985, top=0.80, bottom=0.16)

# --- A. when -------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
for cond, (C, _) in data.items():
    prof = np.nanmean(np.nanmean(C, 1), 0)          # channels x freq -> one curve
    ax.plot(x, prof, lw=1.5, color=COND_C[cond], label=cond)
ax.axvline(50, color=INK, lw=1.0)
ax.axvspan(80, 95, color="#e08214", alpha=0.13, lw=0)
ax.text(50, ax.get_ylim()[1], " GO", fontsize=7, color=INK, va="top")
ax.set_xlabel("normalised trial time (%)", fontsize=8, color=MUTED)
ax.set_ylabel("mean dB, all 129 frequencies, all 138 channels", fontsize=8, color=MUTED)
ax.set_title("A   it is in the average of the whole montage", fontsize=9, color=INK)
ax.legend(fontsize=7, frameon=False)

# --- B. what shape -------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for cond, (C, _) in data.items():
    late = np.nanmean(np.nanmean(C[:, :, LATE], 2), 0)
    early = np.nanmean(np.nanmean(C[:, :, EARLY], 2), 0)
    ax.plot(late - early, freq, lw=1.5, color=COND_C[cond])
ax.axvline(0, color=GREY, lw=0.9)
ax.axhspan(70, 150, color=GREY, alpha=0.25, lw=0)
ax.text(ax.get_xlim()[1], 110, "HFA ", fontsize=7, color=MUTED, ha="right", va="center")
ax.set_xlabel("late (80-95%) minus early (5-35%), dB", fontsize=8, color=MUTED)
ax.set_ylabel("frequency (Hz)", fontsize=8, color=MUTED)
ax.set_title("B   broadband above ~50 Hz,\nnot a spike's low-frequency shape",
             fontsize=9, color=INK)

# --- C. where ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
C, names = data["audio"]
lift = np.nanmean(np.nanmean(C[:, :, LATE], 2), 1) - np.nanmean(np.nanmean(C[:, :, EARLY], 2), 1)
by = {}
for v, n in zip(lift, names):
    by.setdefault(shaft(n), []).append(float(v))
order = sorted(by, key=lambda s: -np.mean(by[s]))
for i, sh in enumerate(order):
    v = by[sh]
    ax.plot(v, [i] * len(v), "o", ms=2.6, color="#4a6fa5", alpha=0.55)
    ax.plot(np.mean(v), i, "|", ms=11, color=INK, mew=1.6)
ax.axvline(0, color=GREY, lw=0.9)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{s} ({len(by[s])})" for s in order], fontsize=6.2)
ax.invert_yaxis()
ax.set_xlabel("late lift, dB   (audio, one dot per contact)", fontsize=8, color=MUTED)
ax.set_title("C   every shaft, not one focus", fontsize=9, color=INK)

for a in fig.axes:
    a.tick_params(labelsize=7, colors=MUTED, length=2)
    for sp in a.spines.values():
        sp.set_color(GREY)

fig.suptitle("PAT_6684 (G-05) - the late broadband band in the time-warped ERSP",
             fontsize=10.5, color=INK, y=0.955)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)

# --- the numbers the figure is claiming ----------------------------------------
for cond, (C, names) in data.items():
    lift = np.nanmean(np.nanmean(C[:, :, LATE], 2), 1) - np.nanmean(np.nanmean(C[:, :, EARLY], 2), 1)
    lo = np.nanmean(np.nanmean(C[:, freq < 30, :][:, :, LATE], 2), 1) - \
        np.nanmean(np.nanmean(C[:, freq < 30, :][:, :, EARLY], 2), 1)
    hi = np.nanmean(np.nanmean(C[:, freq > 200, :][:, :, LATE], 2), 1) - \
        np.nanmean(np.nanmean(C[:, freq > 200, :][:, :, EARLY], 2), 1)
    print(f"{cond:8s} late lift: broadband {np.mean(lift):+.2f} dB, "
          f"{int((lift > 0).sum())}/{len(lift)} channels positive | "
          f"<30 Hz {np.mean(lo):+.2f} dB | >200 Hz {np.mean(hi):+.2f} dB")
