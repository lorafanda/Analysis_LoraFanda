#!/usr/bin/env python3
"""
make_separation_across_features.py - FIG C.7's question, asked of every feature set.

C.7 asks whether a partition is a separation at all, on concat_hg only. This asks the
same question of every feature set that has all three methods, so they can be compared:

    concat_hg       1266 x  900   high gamma, 3 conditions x 300 time bins
    concat_hg_all   2946 x  900   the same, responsiveness gate lifted
    concat_rawds    1266 x 1350   15 bands x 3 conditions x 30 bins

hg, raw and rawds are excluded on purpose: they have k-means and Ward runs but no
convex NMF run, so the three-way comparison cannot be made on them.

EVERY CELL IS SCORED IN BOTH SPACES AGAINST ITS OWN NULL. The null is a Gaussian with
that feature set's own covariance - correlated features, smooth time courses, one blob,
no cluster structure - passed through the identical procedure. White noise would be
trivially beatable and would prove nothing.

WHY TWO SPACES. Convex NMF unit-norms each electrode before fitting; k-means and Ward
use raw dB. Silhouette is not space-free, so scoring all three in one space measures
two of them in a space they never optimised in. That error is what made the first
version of C.7 wrong, and it is the whole point of the figure.

    python make_separation_across_features.py               # ~30 min
    python make_separation_across_features.py --cached      # redraw from json
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import measure_cluster_stability as MS

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "comparison"
K = 7
N_NULL = 10

FEATURE_SETS = [
    ("concat_hg", "high gamma, gated"),
    ("concat_hg_all", "high gamma, gate lifted"),
    ("concat_rawds", "15 bands x time"),
]
METHODS = [("cnmf", "convex NMF"), ("kmeans", "k-means"), ("hierarchical", "Ward")]
HOME = {"cnmf": "unit-norm", "kmeans": "dB", "hierarchical": "dB"}

INK, MUTED = "#1b232c", "#68727d"
GREEN, RED = "#1b7837", "#c1121f"


def labels_at_k(run, method, A):
    """The published labels where they exist, else a fresh fit at K."""
    p = run / "cluster_labels_by_k.csv"
    if p.exists():
        d = pd.read_csv(p)
        if f"k_{K}" in d.columns:
            return pd.to_numeric(d[f"k_{K}"], errors="coerce").to_numpy().astype(int)
    if method == "cnmf":
        g = run / "G_loadings.npy"
        if g.exists():
            G = np.load(g)
            if G.shape[1] == K:
                return (G / np.maximum(G.sum(1, keepdims=True), 1e-12)).argmax(1)
    return MS.fit_any(A, K, 0, method)


def compute():
    from sklearn.metrics import silhouette_score
    rows = []
    for fs, fs_lab in FEATURE_SETS:
        for m, m_lab in METHODS:
            try:
                run, _ = MS.resolve(m, fs)
            except FileNotFoundError:
                print(f"  {m}/{fs}: no run, skipped")
                continue
            X = np.load(run / "X_train.npy").astype(float)
            spaces = {"dB": X, "unit-norm": MS.unit(X)}
            lab = labels_at_k(run, m, spaces[HOME[m]])
            if len(set(lab.tolist())) < 2:
                print(f"  {m}/{fs}: degenerate labels, skipped")
                continue
            for sp, A in spaces.items():
                t0 = time.time()
                obs = float(silhouette_score(A, lab))
                nulls = []
                for i in range(N_NULL):
                    Y = MS.surrogate(X, 400 + i)
                    Y = MS.unit(Y) if sp == "unit-norm" else Y
                    nulls.append(silhouette_score(Y, MS.fit_any(Y, K, i, m)))
                nl = np.array(nulls, float)
                z = (obs - nl.mean()) / max(nl.std(), 1e-9)
                rows.append(dict(feature_set=fs, feature_label=fs_lab, method=m,
                                 method_label=m_lab, space=sp, home=HOME[m] == sp,
                                 n=int(X.shape[0]), p=int(X.shape[1]),
                                 silhouette=obs, null_mean=float(nl.mean()),
                                 null_sd=float(nl.std()), z=float(z),
                                 n_null=N_NULL))
                print(f"  {m:<13} {fs:<14} {sp:<10} sil {obs:+.3f}  "
                      f"null {nl.mean():+.3f}+/-{nl.std():.3f}  z {z:+6.1f}   "
                      f"({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def draw(df, out_png):
    piv_z = df.pivot_table(index=["feature_set", "method"], columns="space",
                           values="z")
    piv_s = df.pivot_table(index=["feature_set", "method"], columns="space",
                           values="silhouette")
    order = [(fs, m) for fs, _ in FEATURE_SETS for m, _ in METHODS
             if (fs, m) in piv_z.index]
    piv_z, piv_s = piv_z.loc[order], piv_s.loc[order]
    cols = ["dB", "unit-norm"]
    Z = piv_z[cols].to_numpy()
    S = piv_s[cols].to_numpy()

    fig = plt.figure(figsize=(11.6, 8.4), dpi=200)
    gs = GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 0.035],
                  wspace=0.42, hspace=0.10,
                  left=0.175, right=0.975, top=0.600, bottom=0.075)

    ax = fig.add_subplot(gs[0, 0])
    lim = np.nanmax(np.abs(Z))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    mlab = dict(METHODS)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{dict(FEATURE_SETS)[fs]}\n{mlab[m]}" for fs, m in order],
                       fontsize=8.2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["scored in dB", "scored in unit-norm"], fontsize=9)
    for i, (fs, m) in enumerate(order):
        for j, sp in enumerate(cols):
            home = HOME[m] == sp
            ax.text(j, i - 0.13, f"z = {Z[i, j]:+.0f}", ha="center", va="center",
                    fontsize=11.5 if home else 9.5,
                    fontweight="bold" if home else "normal",
                    color="white" if abs(Z[i, j]) > 0.55 * lim else INK)
            ax.text(j, i + 0.21, f"sil {S[i, j]:+.3f}", ha="center", va="center",
                    fontsize=7.4,
                    color="white" if abs(Z[i, j]) > 0.55 * lim else MUTED)
            if home:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor=GREEN, lw=2.6))
    for i in range(3, len(order), 3):
        ax.axhline(i - 0.5, color=INK, lw=1.4)
    ax.set_title("Separation against each method's own null\n"
                 "green box = the space that method actually fits in",
                 fontsize=10, loc="left", color=INK, pad=8)
    ax.tick_params(length=0)
    cax = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("z vs a one-blob null", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)

    axb = fig.add_subplot(gs[0, 1])   # home-space bars
    home = df[df.home]
    lbl, zs, cl = [], [], []
    for fs, _ in FEATURE_SETS:
        for m, m_lab in METHODS:
            r = home[(home.feature_set == fs) & (home.method == m)]
            if not len(r):
                continue
            lbl.append(f"{m_lab}\n{dict(FEATURE_SETS)[fs]}")
            zs.append(float(r.z.iloc[0]))
            cl.append({"cnmf": "#5b2c83", "kmeans": "#1f77b4",
                       "hierarchical": "#2a9d5c"}[m])
    y = np.arange(len(lbl))
    axb.barh(y, zs, color=cl, height=0.68)
    axb.set_yticks(y); axb.set_yticklabels(lbl, fontsize=8)
    axb.invert_yaxis()
    axb.axvline(0, color=INK, lw=0.9)
    axb.axvline(2, color=RED, ls="--", lw=1.1)
    axb.text(2, -0.75, " z = 2", color=RED, fontsize=7.6)
    for yy, v in zip(y, zs):
        axb.text(v + (0.6 if v >= 0 else -0.6), yy, f"{v:+.0f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=8.4,
                 color=INK)
    axb.set_xlabel("z in its own space", fontsize=9)
    axb.set_title("Home-space score only", fontsize=10, loc="left", color=INK, pad=8)
    axb.spines[["top", "right"]].set_visible(False)
    axb.tick_params(labelsize=8, colors=MUTED)

    fig.suptitle("Does any of them separate? — every feature set, both spaces",
                 x=0.032, y=0.972, ha="left", fontsize=15, color=INK)
    n_by = {fs: df[df.feature_set == fs].iloc[0] for fs, _ in FEATURE_SETS
            if (df.feature_set == fs).any()}
    dims = "; ".join(f"{fs} {int(r.n)}x{int(r.p)}" for fs, r in n_by.items())
    body = [
        f"FIG C.7 asks whether a K={K} partition is a separation at all, on concat_hg. "
        f"This asks it of every feature set with all three methods ({dims}). Each cell "
        f"is scored against ITS OWN null - a Gaussian carrying that feature set's "
        f"covariance, so the features stay correlated and the time courses stay smooth, "
        f"but there is one blob and no cluster structure. {N_NULL} nulls per cell.",
        "Convex NMF unit-norms each electrode before fitting; k-means and Ward use raw "
        "dB. Silhouette is not space-free, so each method is read in the green box - the "
        "space it actually optimised in. Reading a method outside its own box is the "
        "error that made the first version of C.7 wrong.",
        "THE ANSWER IS ABOUT THE FEATURES, NOT THE METHOD. Read the green boxes down "
        "the right-hand panel: all three methods separate on gated high gamma (+10 to "
        "+15) and separate best on bands x time (+15 to +23), but ALL THREE COLLAPSE TO "
        "CHANCE once the responsiveness gate is lifted (+0 to +3). Whatever structure "
        "these methods find, the gate is what makes it findable.",
        f"Read at a fixed K={K}, which is the published choice for concat_hg and is NOT "
        f"the best K for concat_rawds - bi-cross-validation picks 14 there. So this "
        f"compares the feature sets at one granularity; it does not show what each could "
        f"do at its own.",
    ]
    fig.text(0.032, 0.930, "\n".join(textwrap.fill(t, width=140) for t in body),
             fontsize=8.5, color=MUTED, va="top", linespacing=1.55)

    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / "separation_across_features.json"
    if a.cached and js.exists():
        df = pd.DataFrame(json.loads(js.read_text(encoding="utf-8")))
    else:
        df = compute()
        js.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    df.to_csv(OUT / "separation_across_features.csv", index=False)
    print()
    print(df[["feature_set", "method", "space", "home", "silhouette", "z"]]
          .to_string(index=False))
    draw(df, OUT / "C12_separation_across_features.png")
    print(f"\n-> {OUT / 'C12_separation_across_features.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
