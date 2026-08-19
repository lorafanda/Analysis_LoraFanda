#!/usr/bin/env python3
"""
make_separation_figure.py - is the K=7 partition a separation at all?

Writes outputs/clustering/comparison/: one figure and the JSON it was drawn from,
so the caption on the site cannot drift from the numbers.

The question this answers is not "which method wins". It is whether carving this
dataset into K=7 boxes produces anything a null model would not. Every method runs
on the SAME X_train, at the SAME K, and is compared against ITS OWN null - k-means
optimises compactness and convex NMF does not, so scoring cNMF against a k-means
null would be rigged against it.

THE NULL is a Gaussian with the data's own covariance: correlated features, smooth
time courses, one blob, no cluster structure. It is not white noise - white noise
would be a trivially easy null to beat and would prove nothing.

Two results that are both true and easy to confuse:

  * the DECOMPOSITION is real. Bi-cross-validation - held out in rows AND columns -
    rises from 0.386 at K=2 to a peak near K=8, so the components capture structure
    that generalises to unseen data.

  * the PARTITION taken from it is not. cNMF's argmax scores silhouette 0.046
    against 0.042 +/- 0.005 for the same procedure run on the one-blob null -
    z = +1.0, which is not a separation. k-means and Ward reach z = +15 and +16
    on the same data, so the null is beatable and cNMF's argmax does not beat it.

Those are consistent: the graded structure is real, and hard labels drawn from it
are not a separation.

    python make_separation_figure.py             # refits the nulls (~3 min)
    python make_separation_figure.py --cached    # redraw from stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "comparison"
K = 7
# The cNMF null moved from z = -0.1 to +0.7 between seeds at 3 reps - too few
# for a number that goes on the site. 12 reps puts the SD of the mean under
# a third of that spread.
N_NULL_KM, N_NULL_CNMF = 10, 12

RUNS = {
    "cNMF argmax": CLUST / "cnmf/concat_hg/runs/20260818_112939",
    "k-means":     CLUST / "kmeans/concat_hg/runs/20260817_171544",
    "Ward":        CLUST / "hierarchical/concat_hg/runs/20260817_171627",
}
COL = {"cNMF argmax": "#5b2c83", "k-means": "#1f77b4", "Ward": "#2a9d5c"}
INK, MUTED = "#1b232c", "#68727d"


def compute():
    """Silhouette for each method in BOTH spaces, each against its own matched null.

    THE SPACE IS PART OF THE METHOD. X_train.npy is raw dB; convex NMF unit-norms
    per electrode before fitting (see its manifest), k-means and Ward do not. An
    earlier version of this figure scored all three in dB and concluded cNMF did not
    separate - it had simply been scored in a space it never optimised in. Measured
    both ways, cNMF goes 0.046 in dB to 0.094 in unit-norm, and k-means goes 0.113
    in dB to -0.004 in unit-norm. Neither number alone is the answer.

    Unit-norm removes amplitude, so it asks about response SHAPE. Raw dB keeps
    amplitude, so part of what k-means separates is loud electrodes from quiet ones.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    import lf_decompose as LD

    hg = RUNS["cNMF argmax"]
    X = np.load(hg / "X_train.npy")
    G = np.load(hg / "G_loadings.npy")
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    labels = {
        "cNMF argmax": Gn.argmax(1),
        "k-means": pd.read_csv(RUNS["k-means"] / "cluster_labels_by_k.csv")[f"k_{K}"].to_numpy(),
        "Ward": pd.read_csv(RUNS["Ward"] / "cluster_labels_by_k.csv")[f"k_{K}"].to_numpy(),
    }
    HOME = {"cNMF argmax": "unit-norm", "k-means": "dB", "Ward": "dB"}

    def unit(A):
        return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

    spaces = {"dB": X, "unit-norm": unit(X)}

    Xc = X - X.mean(0)
    n = len(X)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    def surrogate(seed):
        Z = np.random.default_rng(seed).standard_normal((n, len(S)))
        return (Z * (S / np.sqrt(n - 1))) @ Vt

    def fit(alg, Y, seed):
        if alg == "k-means":
            return KMeans(K, n_init=10, random_state=seed).fit_predict(Y)
        if alg == "Ward":
            return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Y)
        Gy = LD.convex_nmf(Y, K, n_iter=300)[1]
        return (Gy / np.maximum(Gy.sum(1, keepdims=True), 1e-12)).argmax(1)

    nulls = {}
    for alg in labels:
        for sp in spaces:
            reps = N_NULL_CNMF if alg == "cNMF argmax" else N_NULL_KM
            vals = []
            for i in range(reps):
                Y = surrogate(400 + i)
                Y = unit(Y) if sp == "unit-norm" else Y
                vals.append(silhouette_score(Y, fit(alg, Y, i)))
            nulls[(alg, sp)] = vals
            print(f"      null {alg:<12} {sp:<10} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    stats = {"K": K, "n_electrodes": int(n), "n_features": int(X.shape[1]),
             "home_space": HOME, "methods": {}, "ari": {}, "bicv": {},
             "per_component": {}}
    for m, lab in labels.items():
        e = {"sizes": [int(v) for v in np.bincount(lab, minlength=K)],
             "largest_share": float(np.bincount(lab, minlength=K).max() / n),
             "home": HOME[m], "spaces": {}}
        for sp, A in spaces.items():
            nl = np.array(nulls[(m, sp)])
            sv = float(silhouette_score(A, lab))
            e["spaces"][sp] = dict(silhouette=sv, null_mean=float(nl.mean()),
                                   null_sd=float(nl.std()),
                                   z=float((sv - nl.mean()) / max(nl.std(), 1e-9)),
                                   n_null=len(nl))
        stats["methods"][m] = e

    names = list(labels)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            stats["ari"][f"{a} vs {b}"] = dict(
                ari=float(adjusted_rand_score(labels[a], labels[b])),
                nmi=float(normalized_mutual_info_score(labels[a], labels[b])))

    bic = pd.read_csv(CLUST / "decomposition/concat_hg/k_bicv.csv")
    g = bic.groupby("k")["var_explained"]
    stats["bicv"] = {"k": [int(v) for v in g.mean().index],
                     "mean": [float(v) for v in g.mean()],
                     "se": [float(v) for v in g.std() / np.sqrt(g.count())]}

    cn = labels["cNMF argmax"]
    shares, maxd = [], []
    for j in range(K):
        m_ = cn == j
        a, b = X[m_], X[~m_]
        sp_ = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1))
                      / (len(a) + len(b) - 2))
        d = np.abs((a.mean(0) - b.mean(0)) / np.maximum(sp_, 1e-9))
        shares.append(float((d > 0.8).mean() * 100))
        maxd.append(float(d.max()))
    stats["per_component"] = {"share_large_pct": shares, "max_abs_d": maxd,
                              "sizes": [int(v) for v in np.bincount(cn, minlength=K)]}
    return stats


def draw(stats, out_png):
    # The header is nine lines now, so it needs its own strip rather than being
    # squeezed above a top=0.775 grid - it was landing on the A and B titles.
    fig = plt.figure(figsize=(11.2, 8.1), dpi=200)
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                  wspace=0.30, hspace=0.60, left=0.075, right=0.975,
                  top=0.700, bottom=0.075)
    names = list(stats["methods"])

    # ---- A: silhouette in BOTH spaces, each against its own null
    ax = fig.add_subplot(gs[0, 0])
    SPACES = ["dB", "unit-norm"]
    w = 0.36
    for i, m in enumerate(names):
        for si, sp in enumerate(SPACES):
            d = stats["methods"][m]["spaces"][sp]
            home = stats["methods"][m]["home"] == sp
            x = i + (si - 0.5) * w
            ax.bar(x, max(d["silhouette"], 0), width=w * 0.9, color=COL[m],
                   alpha=1.0 if home else 0.32, zorder=3,
                   edgecolor=INK if home else "none", lw=1.1)
            lo = d["null_mean"] - d["null_sd"]
            ax.add_patch(plt.Rectangle((x - w * 0.45, lo), w * 0.9,
                                       max(2 * d["null_sd"], 1e-4),
                                       color="#c1121f", alpha=0.22, zorder=4))
            ax.plot([x - w * 0.45, x + w * 0.45], [d["null_mean"]] * 2,
                    color="#c1121f", lw=1.2, zorder=5)
            ax.text(x, max(d["silhouette"], 0) + 0.004, f"{d['z']:+.0f}",
                    ha="center", fontsize=7.6,
                    color=INK if d["z"] > 2 else "#c1121f",
                    fontweight="bold" if home else "normal")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{m}\n(fits in {stats['methods'][m]['home']})" for m in names],
                       fontsize=8)
    ax.set_ylabel("silhouette", fontsize=9)
    ax.set_ylim(0, 0.135)
    ax.tick_params(labelsize=7.5, length=2, colors=MUTED)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.set_title("A — each method separates, but only in its own space",
                 fontsize=10.5, loc="left", color=INK, pad=6)
    ax.text(0.0, -0.30,
            "left bar of each pair = dB, right = unit-norm; SOLID + outlined is the space that "
            "method actually fits in.\n"
            "Numbers are z against the SAME method run on one-blob data "
            "in that space (red line, ±1 SD).",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="top")

    # ---- B: the decomposition itself does generalise
    axb = fig.add_subplot(gs[0, 1])
    k = np.array(stats["bicv"]["k"]); mu = np.array(stats["bicv"]["mean"])
    se = np.array(stats["bicv"]["se"])
    axb.errorbar(k, mu, yerr=se, marker="o", ms=3.5, lw=1.4, capsize=2.5,
                 color=INK)
    axb.axvline(K, color="#5b2c83", lw=1.2, ls="--")
    axb.text(K, mu.min(), f" K={K}", fontsize=7.5, color="#5b2c83", va="bottom")
    axb.set_xlabel("components", fontsize=9)
    axb.set_ylabel("held-out variance explained", fontsize=9)
    axb.tick_params(labelsize=7.5, length=2, colors=MUTED)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.set_title("B — the decomposition, on data it never saw", fontsize=10.5,
                  loc="left", color=INK, pad=6)
    axb.text(0.0, -0.235, f"bi-cross-validation: rows AND columns held out. "
             f"{mu[0]:.2f} at K=2 → {mu.max():.2f} at the peak — real, generalising structure.",
             transform=axb.transAxes, fontsize=7, color=MUTED, va="top")

    # ---- C: do the methods agree, and how balanced are they
    axc = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(names))
    for j in range(K):
        vals = np.array([stats["methods"][m]["sizes"][j] for m in names], float)
        axc.bar(range(len(names)), vals, bottom=bottom, width=0.6,
                color=plt.get_cmap("tab20")(j % 20), edgecolor="white", lw=0.6)
        bottom += vals
    for i, m in enumerate(names):
        sh = stats["methods"][m]["largest_share"] * 100
        axc.text(i, bottom[i] + 14, f"largest cluster\n{sh:.0f}% of all",
                 ha="center", fontsize=7.2, color=INK)
    axc.set_xticks(range(len(names)))
    axc.set_xticklabels(names, fontsize=8.5)
    axc.set_ylabel("electrodes", fontsize=9)
    axc.set_ylim(0, stats["n_electrodes"] * 1.22)
    axc.tick_params(labelsize=7.5, length=2, colors=MUTED)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    axc.set_title("C — cluster sizes, and agreement between methods",
                  fontsize=10.5, loc="left", color=INK, pad=6)
    txt = "  ·  ".join(f"{p}: ARI {v['ari']:+.2f}" for p, v in stats["ari"].items())
    axc.text(0.0, -0.235, txt + "\n(1.0 = identical partitions, 0 = chance)",
             transform=axc.transAxes, fontsize=7, color=MUTED, va="top")

    # ---- D: which components are distinctive at all
    axd = fig.add_subplot(gs[1, 1])
    sh = np.array(stats["per_component"]["share_large_pct"])
    order = np.argsort(-sh)
    axd.bar(range(K), sh[order], width=0.62, color="#1b7837", zorder=3)
    axd.axhline(15, color="#c1121f", lw=1.0, ls="--")
    axd.text(K - 0.45, 17.0, "15%", fontsize=7, color="#c1121f", ha="right")
    for i, j in enumerate(order):
        axd.text(i, sh[j] + 1.4, f"{stats['per_component']['max_abs_d'][j]:.1f}",
                 ha="center", fontsize=6.6, color=MUTED)
    axd.set_xticks(range(K))
    axd.set_xticklabels([f"c{j}" for j in order], fontsize=8)
    axd.set_ylabel("% of features with |d| > 0.8", fontsize=9)
    axd.set_ylim(0, max(sh) * 1.25)
    axd.tick_params(labelsize=7.5, length=2, colors=MUTED)
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)
    axd.set_title("D — how distinctive each component is", fontsize=10.5,
                  loc="left", color=INK, pad=6)
    axd.text(0.0, -0.235, "number above each bar = that component's largest |d|.  "
             "Cohen: 0.8 is a large effect.",
             transform=axd.transAxes, fontsize=7, color=MUTED, va="top")

    fig.suptitle("Is the K=7 partition a separation?", x=0.075, y=0.965,
                 ha="left", fontsize=14, color=INK)
    cu = stats["methods"]["cNMF argmax"]["spaces"]["unit-norm"]
    cd = stats["methods"]["cNMF argmax"]["spaces"]["dB"]
    ku = stats["methods"]["k-means"]["spaces"]["unit-norm"]
    hdr = "".join([chr(10)]).join([
        "Yes — but which space you measure in decides the answer, and the space is "
        "part of the method. Convex NMF unit-norms each electrode before fitting;",
        "k-means and Ward do not.",
        "",
        f"In its OWN space cNMF scores {cu['silhouette']:.3f} against a null of "
        f"{cu['null_mean']:.3f} (z = {cu['z']:+.0f}). Measured in dB instead it scores "
        f"{cd['silhouette']:.3f}, and k-means goes the other way: {ku['silhouette']:+.3f} "
        "in unit-norm.",
        "",
        "Unit-norm removes amplitude, so it asks about response SHAPE; raw dB keeps it, "
        "so part of what k-means separates is loud electrodes from quiet ones. That is",
        "also why the two agree so weakly (panel C): they describe different structure, "
        "rather than disagreeing about the same structure.",
    ])
    fig.text(0.075, 0.925, hdr, fontsize=8.5, color=MUTED, va="top",
             linespacing=1.4)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true",
                    help="redraw from stats.json instead of refitting the nulls")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / "separation_stats.json"

    if a.cached and js.exists():
        stats = json.loads(js.read_text(encoding="utf-8"))
        print("  redrawing from cached stats")
    else:
        print("  fitting nulls (this refits convex NMF a few times) ...")
        stats = compute()
        js.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    for m, e in stats["methods"].items():
        for sp, d in e["spaces"].items():
            mark = "  <- its own space" if sp == e["home"] else ""
            print(f"    {m:<12} {sp:<10} silhouette {d['silhouette']:+.4f}  "
                  f"null {d['null_mean']:.4f}+/-{d['null_sd']:.4f}  "
                  f"z {d['z']:+6.1f}{mark}")
        print(f"    {'':<12} largest cluster {100*e['largest_share']:.0f}% of all")
    for p, v in stats["ari"].items():
        print(f"    {p:<28} ARI {v['ari']:+.3f}")
    draw(stats, OUT / "C7_separation.png")
    print(f"  -> {OUT / 'C7_separation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
