#!/usr/bin/env python3
"""
make_correspondence_explainer.py - E.11 for the KISS tab: how FIG 4 decides that two
clusterings found the same clusters.

    python make_correspondence_explainer.py

WHY A TOY AND NOT THE REAL THING. The real figure has six comparisons, eight clusters
each and 1688 electrodes, which is the wrong size for explaining a procedure. This runs
the identical procedure - centroid correlation, greedy pairing, Jaccard, permutation
null - on 24 electrodes and three clusters, where every step can be seen at once and
counted by hand. Every number printed on the figure is computed here, so the explainer
cannot quietly disagree with the analysis it explains.

    A  two clusterings of the same 24 electrodes. They agree about most of them and
       disagree about four, which is the situation the whole test exists for.
    B  each cluster's mean response. This is what the pairing is done on.
    C  the correlation between every pair of centroids; the greedy rule takes the
       biggest cell, then the biggest of what is left, and so on.
    D  the pairing is tested on something it did not use: how many electrodes the two
       paired clusters actually share, against a null that shuffles the labels.

WRITES  outputs/clustering/explainers/E11_cluster_correspondence.png
        outputs/clustering/explainers/E11_cluster_correspondence.json  (its numbers,
        which make_kiss_tab.py reads so the words quote the picture)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "clustering" / "explainers"

INK, MUTED, GREY, PALE = "#1b1b1b", "#6b6b6b", "#b9b9b9", "#ededed"
COL = ["#1f5f8b", "#a4501f", "#3d7a4e"]        # the three toy clusters
N, K, NT = 24, 3, 60
SEED = 7


def toy():
    """Two clusterings of one set of electrodes, agreeing about most of it."""
    rng = np.random.default_rng(SEED)
    t = np.linspace(0, 1, NT)
    shapes = np.stack([
        np.exp(-((t - .25) ** 2) / .012),                       # early transient
        np.exp(-((t - .62) ** 2) / .030) * 0.9,                 # late, broader
        np.clip((t - .15) * 1.4, 0, None) * np.exp(-(t - .15))  # a ramp
    ])
    lab_a = np.repeat(np.arange(K), N // K)
    X = shapes[lab_a] + rng.normal(0, .16, (N, NT))
    lab_b = lab_a.copy()
    moved = np.array([2, 9, 15, 21])            # four electrodes the two disagree about
    lab_b[moved] = (lab_b[moved] + 1) % K
    return X, lab_a, lab_b, moved, t


def centroids(X, lab):
    Z = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-12)
    C = np.stack([Z[lab == c].mean(0) for c in range(K)])
    return C - C.mean(0, keepdims=True)          # the grand mean comes off, as in FIG 4


def corr(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def greedy(S):
    used1, used2, pairs = set(), set(), []
    for flat in np.argsort(-S, axis=None):
        i, j = divmod(int(flat), S.shape[1])
        if i in used1 or j in used2:
            continue
        pairs.append((i, j, float(S[i, j])))
        used1.add(i); used2.add(j)
    return pairs


def jaccard(a, b):
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def null(lab_a, lab_b, X, n_perm=2000):
    """The whole procedure under chance: relabel one side, re-pair, re-measure."""
    rng = np.random.default_rng(SEED + 1)
    Ca = centroids(X, lab_a)
    out = np.zeros((n_perm, K))
    for d in range(n_perm):
        lb = lab_b[rng.permutation(len(lab_b))]
        for i, j, _ in greedy(corr(Ca, centroids(X, lb))):
            out[d, i] = jaccard(lab_a == i, lb == j)
    return out


def pfmt(v, n_perm=2000):
    """A permutation p can be no smaller than 1/(n+1); do not print it as 0.000."""
    lo = 1.0 / (n_perm + 1)
    return f"< {lo:.4f}" if v <= lo + 1e-12 else f"= {v:.3f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X, lab_a, lab_b, moved, t = toy()
    Ca, Cb = centroids(X, lab_a), centroids(X, lab_b)
    S = corr(Ca, Cb)
    pairs = greedy(S)
    obs = np.array([jaccard(lab_a == i, lab_b == j) for i, j, _ in pairs])
    draws = null(lab_a, lab_b, X)
    j0 = np.array([draws[:, i].mean() for i, _, _ in pairs])
    p = np.array([(1 + (draws[:, i] >= v).sum()) / (len(draws) + 1)
                  for (i, _, _), v in zip(pairs, obs)])
    adj = (obs - j0) / (1 - j0)

    fig = plt.figure(figsize=(11.0, 6.6), facecolor="white")
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.0, 1.12], hspace=.55, wspace=1.5,
                  left=.06, right=.975, top=.86, bottom=.10)

    # ---- A  the two clusterings ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:7])
    for row, lab in ((1.25, lab_a), (0, lab_b)):
        for e in range(N):
            ax.add_patch(plt.Rectangle((e, row), .86, .78, color=COL[lab[e]],
                                       alpha=.92, lw=0))
    for e in moved:
        ax.annotate("", xy=(e + .43, 0.82), xytext=(e + .43, 1.23),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0,
                                    shrinkA=0, shrinkB=0))
    ax.set_xlim(-.4, N + .2); ax.set_ylim(-.45, 2.30)
    ax.set_yticks([1.64, .39])
    ax.set_yticklabels(["solution 1", "solution 2"], fontsize=8)
    ax.set_xticks([]); ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A   the same 24 electrodes, cut two ways", fontsize=9.4, color=INK,
                 loc="left", pad=6)
    ax.text(0, -.36, f"the two agree about {N - len(moved)} electrodes and disagree "
            f"about {len(moved)} (arrows)", fontsize=7.4, color=MUTED)

    # ---- B  the centroids ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 7:12])
    for c in range(K):
        ax.plot(t, Ca[c], color=COL[c], lw=1.6, label="solution 1" if c == 0 else None)
    for i, j, _ in pairs:
        ax.plot(t, Cb[j], color=COL[i], lw=1.1, ls=(0, (2.5, 1.5)),
                label="solution 2" if i == 0 else None)
    ax.set_title("B   each cluster's mean response", fontsize=9.4, color=INK,
                 loc="left", pad=6)
    ax.set_xlabel("time", fontsize=8, color=MUTED)
    ax.set_ylabel("z-scored, grand mean removed", fontsize=7.4, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=1.6, label="solution 1"),
                       Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 1.5)),
                              label="solution 2, in its partner's colour")],
              fontsize=7, frameon=False, loc="upper right", labelcolor=INK,
              handlelength=1.8)

    # ---- C  the correlation matrix and the greedy rule --------------------------
    ax = fig.add_subplot(gs[1, 0:4])
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center", fontsize=7.6,
                    color="white" if abs(S[i, j]) > .55 else INK)
    for rank, (i, j, _) in enumerate(pairs, start=1):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor=INK,
                                   lw=1.8))
        ax.text(j + .40, i - .40, str(rank), fontsize=6.6, color=INK, ha="right",
                va="top")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"s2 c{j}" for j in range(K)], fontsize=7.4)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"s1 c{i}" for i in range(K)], fontsize=7.4)
    ax.tick_params(length=2, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.set_title("C   pair them by correlation, biggest first", fontsize=9.4, color=INK,
                 loc="left", pad=6)
    cax = fig.add_axes([.075, .045, .155, .011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("centroid correlation", fontsize=7, color=MUTED, labelpad=3)
    cb.ax.tick_params(labelsize=6.4, colors=MUTED, length=2)
    cb.outline.set_edgecolor(GREY)

    # ---- D  the test ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 4:12])
    i0, j0_ = pairs[0][0], pairs[0][1]
    d0 = draws[:, i0]
    ax.hist(d0, bins=28, color=PALE, edgecolor=GREY, lw=.5)
    ax.axvline(obs[0], color=COL[i0], lw=2.0)
    ax.annotate(f"observed  {obs[0]:.2f}", xy=(obs[0], ax.get_ylim()[1] * .82),
                xytext=(obs[0] - .30, ax.get_ylim()[1] * .92), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=.9))
    ax.axvline(j0[0], color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.text(j0[0], ax.get_ylim()[1] * .55, f"  chance  {j0[0]:.2f}", fontsize=7.6,
            color=MUTED, va="center")
    ax.set_title("D   now test it on something the pairing did not use: "
                 "the electrodes the two clusters share", fontsize=9.4, color=INK,
                 loc="left", pad=6)
    ax.set_xlabel("overlap of s1 c%d with its partner  (Jaccard)" % i0, fontsize=8,
                  color=MUTED)
    ax.set_ylabel("permutations", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.text(.975, .44, f"p {pfmt(p[0])}\nadjusted overlap "
            f"= (obs - chance)/(1 - chance) = {adj[0]:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK, linespacing=1.6)

    fig.text(.06, .935, "How FIG 4 decides that two clusterings found the same clusters",
             fontsize=11.5, color=INK)
    fig.text(.06, .905, "the real analysis, run on 24 electrodes and three clusters so "
             "every step can be counted by hand", fontsize=8.4, color=MUTED)

    png = OUT / "E11_cluster_correspondence.png"
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)

    js = dict(n_electrodes=N, k=K, n_moved=int(len(moved)), n_perm=int(len(draws)),
              matched=[dict(rank=r + 1, s1=int(i), s2=int(j), r=round(float(rr), 3),
                            jaccard=round(float(obs[r]), 3),
                            chance=round(float(j0[r]), 3),
                            adjusted=round(float(adj[r]), 3),
                            p=round(float(p[r]), 4))
                       for r, (i, j, rr) in enumerate(pairs)])
    (OUT / "E11_cluster_correspondence.json").write_text(
        json.dumps(js, indent=2), encoding="utf-8")
    print(f"  {png}")
    for m in js["matched"]:
        print(f"    rank {m['rank']}  s1 c{m['s1']} - s2 c{m['s2']}  r {m['r']:+.2f}  "
              f"J {m['jaccard']:.2f}  chance {m['chance']:.2f}  adj {m['adjusted']:.2f}  "
              f"p {m['p']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
