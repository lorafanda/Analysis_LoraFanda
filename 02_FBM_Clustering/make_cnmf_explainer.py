#!/usr/bin/env python3
"""
make_cnmf_explainer.py - how convex NMF works, on a feature set small enough to read.

Twelve made-up electrodes, forty time points, three underlying response shapes. Small
enough that every matrix in the factorisation fits on one figure, and the ground truth
is known because it was planted - so the recovered loadings can be checked against the
mixture each electrode was actually built from.

WHAT IT IS MEANT TO MAKE OBVIOUS

    X  ~=  G (W' X)          the whole model, in one line

  * a COMPONENT is not an abstract axis. W' X is a weighted average of REAL electrodes,
    so every component is itself a response profile you could have recorded.
  * a LOADING is graded. Electrode 5 is not "in cluster 2" - it is 0.55 of component 2
    and 0.35 of component 1, and the model says so.
  * the ARGMAX is a summary imposed afterwards. It is what makes the result comparable
    to k-means and Ward, and it is where the graded information is lost.

Uses the project's own convex_nmf, not a re-implementation, so what is drawn is what
the analysis runs.

    python make_cnmf_explainer.py
"""
from __future__ import annotations

import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_decompose as LD

OUT = ROOT / "outputs" / "clustering" / "explainers"

N_ELEC, N_TIME, K = 12, 40, 3
INK, MUTED = "#1b232c", "#68727d"
COMP_COL = ["#5b2c83", "#1f77b4", "#2a9d5c"]
GREY = "#c9ced4"


def build():
    """Three shapes, twelve electrodes, each a known mixture of them plus noise."""
    t = np.linspace(0, 1, N_TIME)
    shapes = np.stack([
        np.exp(-((t - 0.18) ** 2) / 0.006),                 # early transient
        np.exp(-((t - 0.55) ** 2) / 0.020),                 # sustained middle
        np.exp(-((t - 0.85) ** 2) / 0.010) * 1.1,           # late burst
    ])
    rng = np.random.default_rng(3)
    mix = np.array([
        [1.00, 0.00, 0.00], [0.90, 0.10, 0.00], [0.75, 0.25, 0.00], [0.60, 0.40, 0.00],
        [0.10, 0.90, 0.00], [0.35, 0.55, 0.10], [0.00, 1.00, 0.00], [0.00, 0.80, 0.20],
        [0.00, 0.30, 0.70], [0.00, 0.10, 0.90], [0.05, 0.00, 0.95], [0.40, 0.05, 0.55],
    ])
    amp = np.linspace(1.0, 2.2, N_ELEC)          # electrodes differ in loudness too
    X = (mix @ shapes) * amp[:, None]
    X = X + rng.normal(0, 0.05, X.shape)
    return t, shapes, mix, X


def main() -> int:
    t, shapes, mix, X = build()
    Xu = LD.unit_norm(X)                          # what the analysis fits on
    W, G, comps = LD.convex_nmf(Xu, K, random_state=0, n_iter=300)
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)

    # order the recovered components to match the planted shapes, for readability only
    corr = np.array([[np.corrcoef(comps[j], shapes[i])[0, 1] for i in range(K)]
                     for j in range(K)])
    order = list(np.argmax(corr, axis=0))
    if len(set(order)) == K:
        # G must be permuted with W or the product G(W'X) no longer reconstructs
        # anything - permuting one and not the other cost a 7x error before it was
        # caught by the reconstruction going up instead of down.
        comps, Gn, G, W = comps[order], Gn[:, order], G[:, order], W[:, order]
    lab = Gn.argmax(1)
    # TWO RECONSTRUCTIONS, and the difference is worth knowing.
    #
    # The model convex_nmf minimises is X ~= G(W'X). But nothing constrains G's ROWS
    # to sum to 1 - on this fit they sum to about 1.47 - so the raw product carries a
    # scale offset and sits above the data. Checked at 300, 1000, 3000 and 10000
    # iterations: identical, so this is the converged solution, not under-training.
    #
    # Every reported quantity in this project uses the normalised pair instead: Gn with
    # rows summing to 1, and comps = (W'X)/colsum(W). On this data that also happens to
    # reconstruct better (Frobenius 0.64 against 1.15), and it is the version panel D
    # shows, so panel E draws it too rather than mixing conventions mid-figure.
    comps_raw = W.T @ Xu
    recon_model = G @ comps_raw
    recon = Gn @ comps

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 8.8), dpi=200)
    gs = GridSpec(2, 3, hspace=0.52, wspace=0.30,
                  left=0.055, right=0.975, top=0.640, bottom=0.075)

    # A - the twelve electrodes we start with
    ax = fig.add_subplot(gs[0, 0])
    for i in range(N_ELEC):
        ax.plot(t, Xu[i] + i * 0.22, color=GREY, lw=1.1)
        ax.text(-0.04, i * 0.22, f"e{i}", fontsize=6.6, color=MUTED,
                ha="right", va="center")
    ax.set_title("A · the data — 12 electrodes × 40 points\n"
                 "unit-normed, so only shape is left",
                 fontsize=9.5, loc="left", color=INK, pad=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.08, 1.02)
    for s in ax.spines.values():
        s.set_visible(False)

    # B - W: which real electrodes get averaged to make each component
    axb = fig.add_subplot(gs[0, 1])
    Wn = W / np.maximum(W.sum(0, keepdims=True), 1e-12)
    im = axb.imshow(Wn, cmap="Purples", aspect="auto", vmin=0)
    axb.set_xticks(range(K))
    axb.set_xticklabels([f"c{j}" for j in range(K)], fontsize=8.5)
    axb.set_yticks(range(N_ELEC))
    axb.set_yticklabels([f"e{i}" for i in range(N_ELEC)], fontsize=6.6)
    axb.set_title("B · W — each component is a weighted\naverage of REAL electrodes",
                  fontsize=9.5, loc="left", color=INK, pad=5)
    axb.tick_params(length=0)
    for i in range(N_ELEC):
        for j in range(K):
            if Wn[i, j] > 0.12:
                axb.text(j, i, f"{Wn[i, j]:.2f}", ha="center", va="center",
                         fontsize=6.2, color="white" if Wn[i, j] > 0.30 else INK)

    # C - the recovered components, against the shapes that were planted
    axc = fig.add_subplot(gs[0, 2])
    for j in range(K):
        s_n = shapes[j] / np.linalg.norm(shapes[j])
        c_n = comps[j] / max(np.linalg.norm(comps[j]), 1e-12)
        axc.plot(t, s_n + j * 0.30, color=GREY, lw=2.6)
        axc.plot(t, c_n + j * 0.30, color=COMP_COL[j], lw=1.5)
        axc.text(1.01, j * 0.30 + 0.06, f"c{j}", color=COMP_COL[j], fontsize=8.5)
    axc.set_title("C · W′X — the recovered components\n"
                  "grey = the shape that was planted", fontsize=9.5, loc="left",
                  color=INK, pad=5)
    axc.set_xticks([]); axc.set_yticks([]); axc.set_xlim(0, 1.10)
    for s in axc.spines.values():
        s.set_visible(False)

    # D - G: the graded loadings, against the true mixture
    axd = fig.add_subplot(gs[1, 0])
    axd.imshow(Gn, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axd.set_xticks(range(K))
    axd.set_xticklabels([f"c{j}" for j in range(K)], fontsize=8.5)
    axd.set_yticks(range(N_ELEC))
    axd.set_yticklabels([f"e{i}" for i in range(N_ELEC)], fontsize=6.6)
    for i in range(N_ELEC):
        for j in range(K):
            axd.text(j, i, f"{Gn[i, j]:.2f}", ha="center", va="center",
                     fontsize=6.4, color="white" if Gn[i, j] > 0.55 else INK)
    axd.set_title("D · G — the loadings, one row per electrode\n"
                  "rows sum to 1; this is the actual output",
                  fontsize=9.5, loc="left", color=INK, pad=5)
    axd.tick_params(length=0)

    # E - one electrode rebuilt from its loadings
    axe = fig.add_subplot(gs[1, 1])
    pick = 5                                   # deliberately a mixed one
    axe.plot(t, Xu[pick], color=INK, lw=2.4, label=f"electrode e{pick}", zorder=5)
    run = np.zeros_like(t)
    for j in range(K):
        part = Gn[pick, j] * comps[j]
        axe.fill_between(t, run, run + part, color=COMP_COL[j], alpha=0.55, lw=0,
                         label=f"{Gn[pick, j]:.0%} c{j}")
        run = run + part
    axe.plot(t, recon[pick], color="#c1121f", lw=1.4, ls="--",
             label="sum = reconstruction")
    axe.set_title(f"E · e{pick} rebuilt from its own loadings\n"
                  f"its share of each component, stacked", fontsize=9.5, loc="left",
                  color=INK, pad=5)
    axe.legend(fontsize=6.8, frameon=False, loc="upper right")
    axe.set_xticks([]); axe.set_yticks([])
    axe.spines[["top", "right"]].set_visible(False)

    # F - what the argmax keeps, and what it throws away
    axf = fig.add_subplot(gs[1, 2])
    top = np.sort(Gn, axis=1)[:, ::-1]
    marg = top[:, 0] - top[:, 1]
    y = np.arange(N_ELEC)
    axf.barh(y, top[:, 0], color=[COMP_COL[l] for l in lab], height=0.62)
    axf.barh(y, -top[:, 1], color=GREY, height=0.62)
    for i in range(N_ELEC):
        axf.text(top[i, 0] + 0.02, i, f"→ c{lab[i]}", va="center", fontsize=6.8,
                 color=COMP_COL[lab[i]],
                 fontweight="bold" if marg[i] > 0.25 else "normal")
        if marg[i] < 0.25:
            axf.text(-top[i, 1] - 0.03, i, "close call", va="center", ha="right",
                     fontsize=6.2, color="#c1121f")
    axf.set_yticks(y); axf.set_yticklabels([f"e{i}" for i in range(N_ELEC)],
                                           fontsize=6.6)
    axf.invert_yaxis()
    axf.axvline(0, color=INK, lw=0.8)
    axf.set_xlim(-0.75, 1.05)
    axf.set_xticks([-0.5, 0, 0.5, 1.0])
    axf.set_xticklabels(["2nd weight", "0", "0.5", "1.0"], fontsize=7)
    axf.set_title("F · the argmax — top weight vs runner-up\n"
                  "one label kept, the rest discarded", fontsize=9.5, loc="left",
                  color=INK, pad=5)
    axf.spines[["top", "right", "left"]].set_visible(False)
    axf.tick_params(length=0, colors=MUTED)

    err = np.abs(Xu - recon).mean()
    rel = err / np.abs(Xu).mean()
    err_model = np.abs(Xu - recon_model).mean()
    fig.suptitle("How convex NMF clusters — a feature set small enough to read",
                 x=0.055, y=0.972, ha="left", fontsize=15, color=INK)
    body = [
        "Twelve invented electrodes, forty time points, three planted response shapes. "
        "Each electrode is a known mixture of the three plus noise, so the recovered "
        "loadings can be checked against what it was actually built from. Fitted with "
        "the project's own convex_nmf, not a re-implementation.",
        "THE MODEL IS ONE LINE:  X ≈ G(W′X).  W says which real electrodes to average "
        "to make each component (B), so a component is itself a recordable response "
        "profile rather than an abstract axis (C). G says how much of each component "
        "every electrode expresses (D), and its rows sum to 1.",
        f"Panel E stacks electrode 5's share of each component - 30/63/8 - and the "
        f"stack lands on the data (mean |error| {err:.3f}). One caveat stated rather "
        f"than hidden: nothing constrains G's ROWS to sum to 1, and on this fit they sum "
        f"to about 1.47, so the raw product G(W'X) sits above the data ({err_model:.3f}). "
        f"Panels D and E use the normalised pair, which is what every reported number in "
        f"this project uses.",
        f"Panel F is where the CLUSTERING happens: the argmax keeps the largest weight "
        f"and discards the rest. Here e3 splits 57/43 and still gets one label. On the "
        f"real data that costs far more - median top weight 0.43, only 34% with a "
        f"majority, 19% of electrodes within 0.05 of a tie.",
    ]
    fig.text(0.055, 0.930, "\n".join(textwrap.fill(x, width=146) for x in body),
             fontsize=8.5, color=MUTED, va="top", linespacing=1.55)

    p = OUT / "E1_cnmf_explained.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"reconstruction mean |error| {err:.4f}")
    print("planted mixture vs recovered loading, per electrode:")
    for i in range(N_ELEC):
        print(f"  e{i:<3} true {np.round(mix[i], 2)}   recovered {np.round(Gn[i], 2)}"
              f"   -> c{lab[i]}")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
