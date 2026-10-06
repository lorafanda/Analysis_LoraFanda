#!/usr/bin/env python3
"""
make_cp_explainer.py - how a CP / PARAFAC tensor decomposition works, on a feature set small
enough to read.

The same twelve invented electrodes, each now recorded as FOUR bands x forty time points x
THREE conditions - the shape of a real cube, shrunk. Two components are planted:
  1  a high-gamma burst late in the trial, the same in every condition
  2  a beta drop early in the trial, in the picture condition only
Each electrode carries a known amount of each plus noise, so every recovered factor can be
checked against what was planted.

WHAT IT IS MEANT TO MAKE OBVIOUS

  * convex NMF never sees bands, time and conditions as separate axes: it unrolls each
    electrode's 4 x 40 x 3 record into one strip of 480 numbers and compares strips.
  * CP keeps the axes. ONE component is FOUR vectors - a band profile, a time course, a
    weight per condition, a loading per electrode - and their product is a full
    bands x time x condition record. The data is the sum of a few such products.
  * the assumption that buys this readability: inside one component the time course is the
    same in every band and the band profile the same at every time (separability). A real
    ERSP in which high gamma rises while beta falls at a different moment needs two
    components for what is one physiological event.

Fitted with the project's own run_counter_cp.cp_best, not a re-implementation.

    python make_cp_explainer.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from functions import lf_counters as LCn                                         # noqa: E402
_spec = importlib.util.spec_from_file_location("run_counter_cp", ROOT / "run_counter_cp.py")
RC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RC)

OUT = ROOT / "outputs" / "clustering" / "explainers"
N_ELEC, N_BAND, N_TIME, N_COND = 12, 4, 40, 3
BANDS = ["beta", "low γ", "HG", "HFA"]
CONDS = ["audio", "picture", "reading"]
CC = {"audio": "#e06c9f", "picture": "#4a6fa5", "reading": "#8c6d46"}
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
COL = ["#2a9d5c", "#5b2c83"]


def build():
    t = np.linspace(0, 1, N_TIME)
    nrm = lambda v: v / np.linalg.norm(v)
    p = [nrm(np.array([0.1, 0.3, 1.0, 0.8])), nrm(np.array([1.0, 0.4, 0.0, 0.0]))]            # band profiles
    tc = [nrm(np.exp(-((t - 0.75) ** 2) / 0.012)), nrm(-np.exp(-((t - 0.30) ** 2) / 0.020))]  # time courses
    cw = [nrm(np.array([1.0, 1.0, 1.0])), nrm(np.array([0.0, 1.0, 0.0]))]                     # condition weights
    a = [np.array([1.0, 0.9, 0.8, 0.6, 0.3, 0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.4]),
         np.array([0.0, 0.0, 0.2, 0.4, 0.9, 0.5, 1.0, 0.8, 0.7, 0.9, 0.2, 0.3])]
    amp = np.linspace(1.0, 2.2, N_ELEC)
    Y = np.zeros((N_ELEC, N_BAND, N_TIME, N_COND))
    for r in range(2):
        Y += np.einsum("i,j,k,l->ijkl", a[r] * amp, p[r], tc[r], cw[r])
    clean = Y.copy()
    rng = np.random.default_rng(3)
    Y = Y + rng.normal(0, 0.03, Y.shape)
    return t, p, tc, cw, a, clean, Y


def heat(ax, M, v, title=None):
    ax.imshow(M, cmap="RdBu_r", aspect="auto", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=6.6, color=INK, pad=2)


def main() -> int:
    t, p, tc, cw, a, clean, Y = build()
    Yu = LCn.unit_norm_slices(Y)                                  # what the analysis fits on: shape only
    fac, w, ev = RC.cp_best(Yu, 2, n_init=4, n_iter=300)
    A, Bf, Tm, Cn = fac
    # match the recovered components to the planted ones and fix the signs for the picture only:
    # a factor and its partner can flip together, so make each band profile mostly positive
    truth = [np.column_stack(a), np.column_stack(p), np.column_stack(tc), np.column_stack(cw)]
    _, perm = RC.similarity(truth, fac)
    A, Bf, Tm, Cn, w = A[:, perm], Bf[:, perm], Tm[:, perm], Cn[:, perm], w[perm]
    for r in range(2):
        if Bf[:, r].sum() < 0:
            Bf[:, r] *= -1; Tm[:, r] *= -1
        if np.dot(Tm[:, r], tc[r]) < 0:
            Tm[:, r] *= -1; A[:, r] *= -1
        if Cn[:, r].sum() < 0:
            Cn[:, r] *= -1; A[:, r] *= -1
    recon = np.einsum("ir,jr,kr,lr,r->ijkl", A, Bf, Tm, Cn, w)
    err = float(np.abs(Yu - recon).mean())
    pick = 5
    # planted loadings in the fitted space: each noise-free unit-normed slice regressed on the two planted patterns
    cu = LCn.unit_norm_slices(clean).reshape(N_ELEC, -1)
    P = np.column_stack([np.einsum("j,k,l->jkl", p[r], tc[r], cw[r]).ravel() for r in range(2)])
    Ap = np.linalg.lstsq(P, cu.T, rcond=None)[0].T
    Ap = Ap / np.linalg.norm(Ap, axis=0) * np.linalg.norm(A, axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 9.2), dpi=200)
    gs = GridSpec(2, 3, hspace=0.55, wspace=0.32, left=0.055, right=0.975, top=0.600, bottom=0.070)
    v = float(np.abs(Yu).max())

    # A - two electrodes' records: bands x time, one panel per condition
    subA = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 0], hspace=0.35, wspace=0.12)
    for row, e in enumerate((2, pick)):
        for c, cond in enumerate(CONDS):
            ax = fig.add_subplot(subA[row, c])
            heat(ax, Yu[e, :, :, c], v, title=(cond if row == 0 else None))
            if c == 0:
                ax.set_ylabel(f"e{e}", fontsize=7, color=MUTED)
                ax.set_yticks(range(N_BAND)); ax.set_yticklabels(BANDS, fontsize=5)
    fig.text(0.055, 0.600 + 0.010, "A · the data — each electrode is bands × time × condition\nshown for e2 and e5, unit-normed", fontsize=9.5, color=INK, va="bottom")

    # B - one component is four vectors
    subB = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1], hspace=0.6, wspace=0.45)
    r = 0
    axb = fig.add_subplot(subB[0, 0]); axb.bar(range(N_BAND), Bf[:, r], color=COL[r]); axb.set_xticks(range(N_BAND)); axb.set_xticklabels(BANDS, fontsize=5.5); axb.set_yticks([])
    axb.set_title("band profile", fontsize=7, color=INK, pad=2)
    axb = fig.add_subplot(subB[0, 1]); axb.plot(t, Tm[:, r], color=COL[r], lw=1.6); axb.set_xticks([]); axb.set_yticks([]); axb.axhline(0, color=GREY, lw=.6)
    axb.set_title("time course", fontsize=7, color=INK, pad=2)
    axb = fig.add_subplot(subB[1, 0]); axb.bar(range(N_COND), Cn[:, r], color=[CC[c] for c in CONDS]); axb.set_xticks(range(N_COND)); axb.set_xticklabels(CONDS, fontsize=5.5); axb.set_yticks([])
    axb.set_title("condition weights", fontsize=7, color=INK, pad=2)
    axb = fig.add_subplot(subB[1, 1]); axb.bar(range(N_ELEC), A[:, r], color=COL[r]); axb.set_xticks(range(N_ELEC)); axb.set_xticklabels([f"e{i}" for i in range(N_ELEC)], fontsize=4.5); axb.set_yticks([])
    axb.set_title("electrode loadings", fontsize=7, color=INK, pad=2)
    for axx in fig.axes[-4:]:
        axx.spines[["top", "right"]].set_visible(False)
    fig.text(0.055 + 0.332, 0.600 + 0.010, "B · one component = four vectors (component 1 shown)\ntheir product is a full bands × time × condition record", fontsize=9.5, color=INK, va="bottom")

    # C - recovered against planted: time courses and band profiles
    subC = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], wspace=0.35, width_ratios=[1.6, 1])
    axc = fig.add_subplot(subC[0])
    for r in range(2):
        axc.plot(t, tc[r] + r * 0.55, color=GREY, lw=2.6); axc.plot(t, Tm[:, r] + r * 0.55, color=COL[r], lw=1.5)
        axc.text(1.01, r * 0.55, f"c{r + 1}", color=COL[r], fontsize=8)
    axc.set_xticks([]); axc.set_yticks([]); axc.set_xlim(0, 1.12)
    for s in axc.spines.values():
        s.set_visible(False)
    axc2 = fig.add_subplot(subC[1])
    for r in range(2):
        axc2.bar(np.arange(N_BAND) + r * 0.4 - 0.2, Bf[:, r], width=0.38, color=COL[r])
        axc2.plot(np.arange(N_BAND) + r * 0.4 - 0.2, p[r], "o", color=GREY, ms=5, zorder=5)
    axc2.set_xticks(range(N_BAND)); axc2.set_xticklabels(BANDS, fontsize=5.5); axc2.set_yticks([])
    axc2.spines[["top", "right"]].set_visible(False)
    fig.text(0.055 + 0.664, 0.600 + 0.010, "C · the recovered time courses and band profiles\ngrey = what was planted", fontsize=9.5, color=INK, va="bottom")

    # D - the loadings against the planted mixture
    axd = fig.add_subplot(gs[1, 0])
    M = np.column_stack([A, Ap]); vv = float(np.abs(M).max())
    axd.imshow(M, cmap="RdBu_r", aspect="auto", vmin=-vv, vmax=vv)
    axd.set_xticks(range(4)); axd.set_xticklabels(["c1", "c2", "planted\nc1", "planted\nc2"], fontsize=7)
    axd.set_yticks(range(N_ELEC)); axd.set_yticklabels([f"e{i}" for i in range(N_ELEC)], fontsize=6.6)
    for i in range(N_ELEC):
        for j in range(4):
            axd.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.0, color="white" if abs(M[i, j]) > 0.6 * vv else INK)
    axd.axvline(1.5, color=INK, lw=0.8)
    axd.set_title("D · the electrode loadings — signed, one row per electrode\nleft: recovered · right: planted, in the same unit-normed space", fontsize=9.5, loc="left", color=INK, pad=5)
    axd.tick_params(length=0)

    # E - e5 rebuilt, picture condition: the two rank-1 pieces, their sum, the data, what is left
    subE = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, 1], wspace=0.15)
    c = 1
    pieces = [w[r] * A[pick, r] * np.outer(Bf[:, r], Tm[:, r]) * Cn[c, r] for r in range(2)]
    cells = [(pieces[0], f"{A[pick, 0]:+.2f} × c1"), (pieces[1], f"{A[pick, 1]:+.2f} × c2"), (pieces[0] + pieces[1], "sum"),
             (Yu[pick, :, :, c], "e5, data"), (Yu[pick, :, :, c] - pieces[0] - pieces[1], "left over")]
    for j, (Mx, ttl) in enumerate(cells):
        ax = fig.add_subplot(subE[j]); heat(ax, Mx, v, title=ttl)
        if j == 0:
            ax.set_yticks(range(N_BAND)); ax.set_yticklabels(BANDS, fontsize=5)
    fig.text(0.055 + 0.332, 0.290, "E · e5 rebuilt, picture condition\nloading × component, summed, against the data", fontsize=9.5, color=INK, va="bottom")

    # F - what convex NMF sees instead: the same record unrolled into one strip
    subF = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 2], height_ratios=[2.2, 0.7, 1.4], hspace=0.7)
    axf = fig.add_subplot(subF[0])
    heat(axf, np.concatenate([Yu[pick, :, :, c] for c in range(N_COND)], axis=1), v)
    axf.set_yticks(range(N_BAND)); axf.set_yticklabels(BANDS, fontsize=5)
    for c, cond in enumerate(CONDS):
        axf.text(N_TIME * c + N_TIME / 2, -0.7, cond, ha="center", fontsize=6, color=CC[cond])
    axf2 = fig.add_subplot(subF[1])
    strip = Yu[pick].reshape(1, -1)
    axf2.imshow(strip, cmap="RdBu_r", aspect="auto", vmin=-v, vmax=v, interpolation="nearest"); axf2.set_xticks([]); axf2.set_yticks([])
    axf2.set_title(f"the same record unrolled: {strip.size} numbers in a row — what convex NMF compares", fontsize=6.6, color=INK, pad=2)
    axf3 = fig.add_subplot(subF[2]); axf3.axis("off")
    axf3.text(0, 0.95, "convex NMF: a component is an average of such strips;\n"
                       "which band, when, and in which condition are mixed into one axis.\n"
                       "CP: a component is a band profile, a time course and condition\n"
                       "weights, each readable on its own - at the price of assuming the\n"
                       "time course is the same in every band of that component.",
              fontsize=6.8, color=INK, va="top", linespacing=1.45)
    fig.text(0.055 + 0.664, 0.290, "F · how this differs from convex NMF\ne5 as CP keeps it, and as cNMF flattens it", fontsize=9.5, color=INK, va="bottom")

    fig.suptitle("How a CP tensor decomposition keeps bands, time and conditions apart — a feature set small enough to read",
                 x=0.055, y=0.972, ha="left", fontsize=15, color=INK)
    body = [
        "The same twelve invented electrodes, each now a record of four bands × forty time points × three conditions - a real cube, "
        "shrunk. Two components are planted: a high-gamma burst late in the trial that is the same in every condition, and a beta drop "
        "early in the trial in the picture condition only. Every electrode carries a known amount of each plus noise. "
        "Fitted with the project's own run_counter_cp.cp_best.",
        "THE MODEL IS A SUM OF PRODUCTS: record ≈ Σ loading × band profile × time course × condition weight. One component is four "
        "vectors (B); the fit recovers both planted components (C, D) and rebuilds e5 from its two loadings "
        f"({A[pick, 0]:+.2f} × c1, {A[pick, 1]:+.2f} × c2; mean |error| {err:.3f}, {100 * ev:.0f} % of the variance at rank 2) (E). "
        f"The recovered condition weights say by themselves where each component lives: c1 {np.round(Cn[:, 0], 2).tolist()}, "
        f"c2 {np.round(Cn[:, 1], 2).tolist()} over audio / picture / reading.",
        "WHAT IT ADDS OVER CONVEX NMF (F): cNMF unrolls the record into one strip and averages strips, so a component mixes band, time and "
        "condition into one axis; CP keeps the three axes and the condition weights are where the labels enter. WHAT IT ASSUMES: inside a "
        "component the time course is the same in every band and the band profile the same at every moment. A real response in which "
        "high gamma rises while beta falls at another moment is two CP components for one event, and the fit can become unstable "
        "when many are needed - which is what split-half similarity is there to catch.",
    ]
    fig.text(0.055, 0.930, "\n".join(textwrap.fill(x, width=150) for x in body), fontsize=8.3, color=MUTED, va="top", linespacing=1.5)
    pth = OUT / "E16_cp_explained.png"
    fig.savefig(pth, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = dict(n_elec=N_ELEC, bands=BANDS, n_time=N_TIME, conditions=CONDS, rank=2, var_explained=float(ev), mean_abs_error=err,
                condition_weights=Cn.tolist(), loadings=A.tolist(), planted_loadings=Ap.tolist(), e5={"c1": float(A[pick, 0]), "c2": float(A[pick, 1])})
    (OUT / "E16_cp_explained.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"rank 2: variance explained {ev:.3f}; mean |error| {err:.4f}; condition weights c1 {np.round(Cn[:, 0], 2)} c2 {np.round(Cn[:, 1], 2)}")
    for i in range(N_ELEC):
        print(f"  e{i:<3} planted ({a[0][i]:.2f}, {a[1][i]:.2f})   recovered ({A[i, 0]:+.2f}, {A[i, 1]:+.2f})   planted in fit space ({Ap[i, 0]:+.2f}, {Ap[i, 1]:+.2f})")
    print(f"\n-> {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
