#!/usr/bin/env python3
"""
make_overview_options.py — two candidate opening figures for the paper/overview.

The same evidence can open two ways, and the choice is editorial rather than technical.
Both are generated from the same files so neither is flattered by fresher numbers.

    OPT-A  "Five response types"       leads with the partition, caveats it afterwards
    OPT-B  "A graded organisation"     leads with the negative result, then the
                                       decomposition it forces

They are deliberately the same size and weight so they can be compared side by side.

    python make_overview_options.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEC = ROOT / "outputs" / "clustering" / "decomposition"
RUN = ROOT / "outputs" / "clustering" / "kmeans" / "concat_hg" / "runs" / "20260803_175417"
OUT = ROOT / "outputs" / "clustering" / "story"
INK, MUTED, ACC, WARN, GOOD = "#1b232c", "#68727d", "#1f77b4", "#c1121f", "#2a9d8f"
CONDS = ["audio", "picture", "reading"]


def _load():
    cv = pd.read_csv(DEC / "cv_rank_curve.csv")
    C = np.load(DEC / "components.npy")
    G = np.load(DEC / "G_loadings.npy")
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    lab = pd.read_csv(RUN / "labels.csv")
    ccol = next(c for c in lab.columns
                if c.startswith("cluster_") and not c.endswith("_ranked"))
    X = np.load(RUN / "X_train.npy").astype(float)
    return cv, C, Gn, lab, ccol, X


def _pipeline_ari():
    """Agreement between the five preprocessing variants — the negative result."""
    import sys
    sys.path.insert(0, str(ROOT / "functions"))
    import lf_decompose as D
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score as ari
    lab = pd.read_csv(RUN / "labels.csv")
    pat = lab["patient_id"].astype(str).to_numpy()
    X = np.load(RUN / "X_train.npy").astype(float)
    var = {n: D.apply_pipeline(X, n, pat) for n in D.PIPELINES}
    var["pca20"] = PCA(20, random_state=0).fit_transform(X)
    fits = {k: KMeans(n_clusters=5, n_init=20, random_state=42).fit_predict(v)
            for k, v in var.items()}
    names = list(fits)
    M = np.array([[ari(fits[a], fits[b]) for b in names] for a in names])
    return names, M


def _clusters_panel(ax, X, y, ccol):
    """Mean HG per cluster, audio block only — the recognisable 'five types' view."""
    NT = X.shape[1] // 3
    t = np.linspace(0, 100, NT)
    for i, c in enumerate(sorted(np.unique(y))):
        m = y == c
        ax.plot(t, X[m].reshape(m.sum(), 3, NT)[:, 0, :].mean(0), lw=1.6,
                color=plt.get_cmap("tab10").colors[i], label=f"c{c} (n={m.sum()})")
    ax.axvline(50, color="0.7", lw=.8, ls=":")
    ax.axhline(0, color="0.85", lw=.8)
    ax.set_xlabel("% of trial (50 = GO)", fontsize=8.5)
    ax.set_ylabel("HGA (dB)", fontsize=8.5)
    ax.legend(fontsize=6.8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)


def _ari_panel(ax, names, M):
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if M[i, j] > .6 or M[i, j] < .2 else "black")
    return im


def _components_panel(ax, C, K):
    NT = C.shape[1] // 3
    t = np.linspace(0, 100, NT)
    off = 0.0
    for j in range(K):
        prof = C[j].reshape(3, NT)[0]
        ax.plot(t, prof + off, lw=1.3, color=plt.get_cmap("tab10").colors[j % 10])
        ax.text(101, off, f"c{j}", fontsize=7, color=plt.get_cmap("tab10").colors[j % 10],
                va="center")
        off += 0.055
    ax.axvline(50, color="0.7", lw=.8, ls=":")
    ax.set_xlabel("% of trial (50 = GO)", fontsize=8.5)
    ax.set_yticks([])
    ax.set_xlim(0, 108)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=7.5)


def _mixture_panel(ax, Gn):
    top = Gn.max(1)
    ax.hist(top, bins=34, color="#4a6fa5")
    ax.axvline(.5, color=WARN, ls="--", lw=1.1)
    ax.set_xlabel("largest component weight", fontsize=8.5)
    ax.set_ylabel("electrodes", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.text(.97, .93, f"{100*(top < .5).mean():.0f}% have\nno majority",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)


def option_a(cv, C, Gn, lab, ccol, X, names, M):
    """Lead with the partition. Caveat second."""
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=.34, left=.06, right=.97, top=.76, bottom=.17)
    y = lab[ccol].to_numpy()

    ax = fig.add_subplot(gs[0, 0])
    _clusters_panel(ax, X, y, ccol)
    ax.set_title("1 - Five response types", fontsize=10.5, loc="left", color=INK)

    ax = fig.add_subplot(gs[0, 1])
    _ari_panel(ax, names, M)
    ax.set_title("2 - ...but the partition is a preprocessing choice",
                 fontsize=10.5, loc="left", color=WARN)

    ax = fig.add_subplot(gs[0, 2])
    _mixture_panel(ax, Gn)
    ax.set_title("3 - ...and membership is graded", fontsize=10.5, loc="left", color=WARN)

    fig.suptitle("OPTION A - lead with the five types, then qualify them",
                 fontsize=13, x=.06, ha="left", y=.95, color=INK)
    fig.text(.06, .88, "Familiar and concrete, and it matches what the lab has already seen. "
                       "Risk: the reader forms the taxonomy first and the caveats arrive as "
                       "hedging.", fontsize=9, color=MUTED)
    p = OUT / "OPT_A_lead_with_types.png"
    fig.savefig(p, dpi=155, bbox_inches="tight", facecolor="white")
    print(f"  wrote {p.name}")


def option_b(cv, C, Gn, lab, ccol, X, names, M):
    """Lead with the negative result. The decomposition follows from it."""
    K = C.shape[0]
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=.34, left=.06, right=.97, top=.76, bottom=.17)

    ax = fig.add_subplot(gs[0, 0])
    _ari_panel(ax, names, M)
    ax.set_title("1 - No stable partition exists", fontsize=10.5, loc="left", color=INK)

    ax = fig.add_subplot(gs[0, 1])
    g = cv.groupby("k")["var_explained"].agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=4, lw=1.4,
                color=ACC, capsize=2.5)
    ax.axvline(K, color=GOOD, ls="--", lw=1.2)
    ax.text(K + .5, g["mean"].min() + .01, f"K={K}", color=GOOD, fontsize=9)
    ax.set_xlabel("components", fontsize=8.5)
    ax.set_ylabel("held-out variance explained", fontsize=8.5)
    ax.set_title("2 - and no natural K either", fontsize=10.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[0, 2])
    _components_panel(ax, C, K)
    ax.set_title(f"3 - so: {K} graded components", fontsize=10.5, loc="left", color=GOOD)

    fig.suptitle("OPTION B - lead with what does not hold, then what does",
                 fontsize=13, x=.06, ha="left", y=.95, color=INK)
    fig.text(.06, .88, "Harder opening, but the method change is then forced by evidence rather "
                       "than asserted. Risk: three panels before any positive result.",
             fontsize=9, color=MUTED)
    p = OUT / "OPT_B_lead_with_evidence.png"
    fig.savefig(p, dpi=155, bbox_inches="tight", facecolor="white")
    print(f"  wrote {p.name}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cv, C, Gn, lab, ccol, X = _load()
    names, M = _pipeline_ari()
    option_a(cv, C, Gn, lab, ccol, X, names, M)
    option_b(cv, C, Gn, lab, ccol, X, names, M)
    iu = np.triu_indices(len(names), 1)
    print(f"  lowest pipeline agreement: {M[iu].min():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
