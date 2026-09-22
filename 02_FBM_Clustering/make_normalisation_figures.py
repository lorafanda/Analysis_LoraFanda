#!/usr/bin/env python3
"""
make_normalisation_figures.py - what the normalisation choice does to a hard partition.

WHY THIS EXISTS. k-means and Ward minimise a sum of squares, so whatever carries the
variance carries the partition. Three separable things live in every concatenated ERSP
(Cronbach & Gleser 1953): ELEVATION (the mean level of the vector), SCATTER/amplitude
(its length) and SHAPE (where in frequency, condition and time the values sit). Only the
third is the result we want to report, and nothing in the pipeline decides between them
until a feature space is chosen. These figures measure which of the three is driving the
K=8 partition under each candidate transform, on the cohort the site publishes.

THE PRECEDENT THIS REPRODUCES. Hamilton, Edwards & Chang (2018, Curr Biol 28:1860-1871)
clustered 1,906 speech-responsive electrodes with convex NMF and reported that beyond
k=2 the extra clusters were the same two response types "mostly further subdivided
according to response magnitude" (p.1861, and Figure S2B-D), even though every electrode
had already been z-scored against its own session (STAR Methods e1). FIG N.3 runs that
same test on our cohort, in raw dB and after normalisation.

    python make_normalisation_figures.py            # all six figures
    python make_normalisation_figures.py --quick    # skip the bi-CV sweep (FIG N.4)

Outputs -> outputs/clustering/normalisation/N1..N6_*.png  + normalisation_scorecard.csv
The numbers quoted in the site block come from that CSV, so regenerate both together.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "normalisation"
# The v8 runs the site publishes. X_train.npy is raw dB in both; the run's own fit space
# is recorded in its manifest and is irrelevant here - we re-fit every space ourselves.
RUN_B5 = CLUST / "kmeans" / "concat_bands5" / "runs" / "20260915_004248"
RUN_HG = CLUST / "kmeans" / "concat_hg" / "runs" / "20260915_004134"

INK, MUTED, RED, GREEN, BLUE = "#1b232c", "#68727d", "#c1121f", "#1b7837", "#2471a3"
ACC = "#b5651d"
K_MAIN = 8
WRAP = 155        # characters per suptitle line; a long single line stretches the canvas
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.edgecolor": "#c9d1d9", "axes.linewidth": 0.8,
                     "figure.facecolor": "white"})


# ---------------------------------------------------------------- data + transforms
def wrap(text: str, width: int = WRAP) -> str:
    """Hard-wrap a caption. matplotlib will not wrap a suptitle, and bbox_inches='tight'
    then widens the whole canvas to fit one long line."""
    import textwrap as _tw
    nl = chr(10)
    return nl.join(nl.join(_tw.wrap(par, width)) if par.strip() else ""
                   for par in text.split(nl))


def load(run: Path):
    X = np.load(run / "X_train.npy").astype(float)
    D = pd.read_parquet(run / "df_keep_with_clusters.parquet")
    names = json.loads((run / "feature_schema.json").read_text())["feature_names"]
    bands = list(dict.fromkeys(n.split("|")[1] for n in names))
    conds = list(dict.fromkeys(n.split("|")[0] for n in names))
    return X, D, bands, conds


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def zrow(A):
    return (A - A.mean(1, keepdims=True)) / np.maximum(A.std(1, keepdims=True), 1e-12)


def soft(A, alpha=0.5):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12) ** alpha


def peaknorm(A):
    return A / np.maximum(np.abs(A).max(1, keepdims=True), 1e-12)


def make_bandz(NB, NC, NT):
    """cohort-level z per band - one mean and one SD over every electrode, condition and
    bin of that band. This is concat_bands5z_features."""
    def f(A):
        B = A.copy()
        for b in range(NB):
            sl = slice(b * NC * NT, (b + 1) * NC * NT)
            B[:, sl] = (A[:, sl] - A[:, sl].mean()) / max(A[:, sl].std(), 1e-12)
        return B
    return f


def make_colz():
    def f(A):
        return (A - A.mean(0)) / np.maximum(A.std(0), 1e-12)
    return f


def make_rowz_perband(NB, NC, NT):
    def f(A):
        B = A.copy()
        for b in range(NB):
            sl = slice(b * NC * NT, (b + 1) * NC * NT)
            blk = A[:, sl]
            B[:, sl] = (blk - blk.mean(1, keepdims=True)) / np.maximum(blk.std(1, keepdims=True), 1e-12)
        return B
    return f


# ---------------------------------------------------------------- diagnostics
def eta2(lab, v):
    """How much of the spread in v is between clusters. 1 = the partition IS a sort by v."""
    g = pd.Series(v).groupby(lab)
    return float(1 - g.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
                 / max(((v - v.mean()) ** 2).sum(), 1e-12))


def cramers_v(a, b):
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    N = ct.sum()
    exp = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / N
    chi2 = ((ct - exp) ** 2 / np.maximum(exp, 1e-12)).sum()
    return float(np.sqrt(chi2 / (N * (min(ct.shape) - 1))))


def patient_dominated(lab, pat):
    tot = 0
    for k in np.unique(lab):
        idx = lab == k
        if pd.Series(pat[idx]).value_counts().iloc[0] / idx.sum() > 0.5:
            tot += int(idx.sum())
    return tot / len(lab)


def stability(A, K=K_MAIN, n_pairs=10, frac=0.8, seed=0):
    """Two 80% subsamples, k-means on each, ARI on the electrodes they share."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    out = []
    for _ in range(n_pairs):
        i1 = rng.choice(n, int(frac * n), replace=False)
        i2 = rng.choice(n, int(frac * n), replace=False)
        both = np.intersect1d(i1, i2)
        l1 = KMeans(K, n_init=5, random_state=int(rng.integers(1e6))).fit(A[i1]).predict(A[both])
        l2 = KMeans(K, n_init=5, random_state=int(rng.integers(1e6))).fit(A[i2]).predict(A[both])
        out.append(adjusted_rand_score(l1, l2))
    return float(np.mean(out))


def dominant(Bz, NB, NC, NT):
    """Which band / condition / time bin carries an electrode's response, measured on the
    BAND-EQUALISED matrix so the answer is not itself a 1/f artefact."""
    be = np.stack([np.abs(Bz[:, b * NC * NT:(b + 1) * NC * NT]).mean(1) for b in range(NB)], 1)
    cc = [np.concatenate([np.arange(b * NC * NT + c * NT, b * NC * NT + c * NT + NT)
                          for b in range(NB)]) for c in range(NC)]
    ce = np.stack([np.abs(Bz[:, cols]).mean(1) for cols in cc], 1)
    db_, dc_ = be.argmax(1), ce.argmax(1)
    pt = np.array([np.abs(Bz[i, db_[i] * NC * NT + dc_[i] * NT: db_[i] * NC * NT + dc_[i] * NT + NT]).argmax()
                   for i in range(Bz.shape[0])])
    return db_, dc_, pt


# ---------------------------------------------------------------- figures
def fig_n1(X, r, elev, bands, NB, NC, NT, out):
    """What a concatenated cube's numbers are made of, before any choice is made."""
    fig = plt.figure(figsize=(11.5, 3.1), dpi=180)
    gs = GridSpec(1, 4, figure=fig, wspace=0.34, left=0.055, right=0.985, top=0.80, bottom=0.20)

    ax = fig.add_subplot(gs[0])
    ax.hist(r, bins=60, color=BLUE, alpha=0.85)
    for q, c in ((5, MUTED), (50, INK), (95, MUTED)):
        ax.axvline(np.percentile(r, q), color=c, lw=1.0, ls="--")
    ax.set_xlabel("‖x‖  (length of the 450-vector, dB)")
    ax.set_ylabel("electrodes")
    ax.set_title(f"A · amplitude spans {np.percentile(r,95)/np.percentile(r,5):.1f}×\n"
                 f"5% {np.percentile(r,5):.0f} · median {np.median(r):.0f} · 95% {np.percentile(r,95):.0f}",
                 loc="left")

    ax = fig.add_subplot(gs[1])
    ss = [float((X[:, b * NC * NT:(b + 1) * NC * NT] ** 2).sum()) for b in range(NB)]
    sh = 100 * np.array(ss) / sum(ss)
    ax.barh(range(NB), sh, color=[ACC if s == sh.max() else BLUE for s in sh])
    ax.set_yticks(range(NB))
    ax.set_yticklabels(bands)
    ax.invert_yaxis()
    for i, s in enumerate(sh):
        ax.text(s + 0.6, i, f"{s:.0f}%", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("share of the cohort's sum of squares")
    ax.set_xlim(0, max(sh) * 1.25)
    ax.set_title("B · Euclidean distance has no idea\n1/f exists", loc="left")

    ax = fig.add_subplot(gs[2])
    ax.hist(elev, bins=60, color=BLUE, alpha=0.85)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlabel("mean dB over all 450 features")
    ax.set_ylabel("electrodes")
    ax.set_title(f"C · elevation: a constant offset\n5–95%  {np.percentile(elev,5):+.2f} … {np.percentile(elev,95):+.2f} dB",
                 loc="left")

    ax = fig.add_subplot(gs[3])
    m = X.mean(0)
    m = m / np.linalg.norm(m)
    a = X @ m
    r1 = 1 - ((X - np.outer(a, m)) ** 2).sum() / (X ** 2).sum()
    ax.bar([0, 1], [100 * r1, 100 * (1 - r1)], color=[ACC, "#d6dde4"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["own amplitude ×\ncommon profile", "everything else\n(the shapes)"])
    ax.set_ylabel("% of the raw sum of squares")
    for i, v in enumerate([100 * r1, 100 * (1 - r1)]):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 100)
    ax.set_title("D · amplitude is real but not\nthe main thing", loc="left")

    fig.suptitle(wrap(
        "FIG N.1 · what the numbers in a concatenated cube are made of — concat_bands5, v8, 1680 electrodes × 450 features (raw dB)"),
        fontsize=10, x=0.055, ha="left", y=0.975)
    fig.savefig(out / "N1_what_the_numbers_are.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_n2(score, out):
    """The scorecard: one row per candidate transform, one column per diagnostic."""
    cols = [("amp", "sorts by\namplitude", False), ("elev", "sorts by\nelevation", False),
            ("vband", "tracks the\nresponse band", True), ("vcond", "tracks the\ncondition", True),
            ("bal", "cluster balance\nmin / max", True), ("pat", "1-patient\nclusters", False),
            ("stab", "stability\n(ARI)", True), ("ward_amp", "Ward: sorts\nby amplitude", False)]
    S = score.set_index("transform")
    rows = list(S.index)
    M = np.zeros((len(rows), len(cols)))
    for j, (k, _, good_high) in enumerate(cols):
        v = S[k].to_numpy(float)
        rng = np.ptp(v) or 1.0
        z = (v - v.min()) / rng
        M[:, j] = z if good_high else 1 - z

    fig, ax = plt.subplots(figsize=(11.6, 0.46 * len(rows) + 2.2), dpi=180)
    ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, alpha=0.75)
    for i, rname in enumerate(rows):
        for j, (k, _, _) in enumerate(cols):
            ax.text(j, i, f"{S.iloc[i][k]:.2f}", ha="center", va="center",
                    fontsize=8, color=INK,
                    fontweight="bold" if rname.startswith("band-z → unit") else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5)
    for i, rname in enumerate(rows):
        if rname.startswith("band-z → unit"):
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False,
                                       edgecolor=GREEN, lw=2.2))
        if "trap" in rname:
            # a trap can still score well on one column - hatch the row so a green cell
            # there is never read as an endorsement
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False,
                                       edgecolor=RED, lw=1.6, hatch="////", alpha=0.55))
            ax.get_yticklabels()[i].set_color(RED)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    ax.set_title("FIG N.2 · the normalisation scorecard — k-means K=8 on the same 1680 electrodes, concat_bands5 (v8)\n"
                 "Green = the behaviour we want, column by column: columns 1, 2, 6 and 8 LOW (the partition is not a sort by "
                 "size or by level, and no cluster is one patient),\ncolumns 3, 4, 5 and 7 HIGH (it tracks where the response "
                 "is, it is balanced, it reproduces). Green box = the space the site's runs already use.\n"
                 "Hatched rows destroy the information the clustering is for — a green cell there is an artefact of the "
                 "column, not an endorsement.",
                 loc="left", fontsize=9.5, pad=12)
    fig.savefig(out / "N2_scorecard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_n3(X, Xn, r, pat, out, ks=tuple(range(2, 15))):
    """Do extra clusters buy shape, or do they subdivide by magnitude?
    The direct test of Hamilton et al. 2018 (p.1861 / Fig S2B-D) on our cohort."""
    rec = []
    for name, A in (("raw dB", X), ("band-z → unit-norm", Xn)):
        for k in ks:
            lab = KMeans(k, n_init=10, random_state=42).fit_predict(A)
            mn = pd.Series(r).groupby(lab).mean().to_numpy()
            sz = np.bincount(lab)
            rec.append(dict(space=name, k=k, eta2=eta2(lab, r),
                            spread=mn.max() / mn.min(), bal=sz.min() / sz.max(),
                            pat=patient_dominated(lab, pat), means=mn))
    R = pd.DataFrame(rec)

    fig = plt.figure(figsize=(11.5, 4.3), dpi=180)
    gs = GridSpec(1, 3, figure=fig, wspace=0.28, left=0.06, right=0.985, top=0.74, bottom=0.13)
    col = {"raw dB": RED, "band-z → unit-norm": GREEN}

    ax = fig.add_subplot(gs[0])
    for nm, sub in R.groupby("space", sort=False):
        ax.plot(sub.k, sub.eta2, "o-", ms=4, lw=1.6, color=col[nm], label=nm)
    ax.axvline(K_MAIN, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("K"); ax.set_ylabel("η² of ‖x‖ across clusters")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("A · how much of the partition\nis a sort by amplitude", loc="left")

    ax = fig.add_subplot(gs[1])
    off = {"raw dB": -0.16, "band-z → unit-norm": +0.16}
    for nm, sub in R.groupby("space", sort=False):
        for _, row in sub.iterrows():
            x = row.k + off[nm]
            ax.plot([x, x], [row.means.min(), row.means.max()], color=col[nm], lw=1.1, alpha=0.6)
            ax.scatter([x] * len(row.means), row.means, s=11, color=col[nm],
                       alpha=0.9, edgecolors="none", zorder=3)
    ax.set_xlabel("K"); ax.set_ylabel("mean ‖x‖ of each cluster (dB)")
    hi = R[R.space == "raw dB"].spread.iloc[-1]
    lo = R[R.space != "raw dB"].spread.iloc[-1]
    ax.set_title(f"B · the amplitude ladder each K builds\nat K=14 the clusters span {hi:.1f}× in dB, {lo:.1f}× normalised",
                 loc="left")

    ax = fig.add_subplot(gs[2])
    for nm, sub in R.groupby("space", sort=False):
        ax.plot(sub.k, sub.bal, "o-", ms=4, lw=1.6, color=col[nm])
    ax.set_xlabel("K"); ax.set_ylabel("smallest / largest cluster")
    ax.set_ylim(0, 0.6)
    ax.set_title("C · and what it costs in balance", loc="left")

    fig.suptitle(wrap(
        "FIG N.3 · do extra clusters buy shape, or subdivide by magnitude? — the Hamilton et al. 2018 test on our cohort\n"
                 "They found that beyond k=2 the extra convex-NMF clusters were the same two response types, \"mostly further "
                 "subdivided according to response magnitude\" (Curr Biol 28:1861; Fig S2B–D), even after per-electrode session z-scoring."),
        fontsize=10, x=0.06, ha="left", y=0.99)
    fig.savefig(out / "N3_magnitude_splitting.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    R.drop(columns=["means"]).to_csv(OUT / "N3_magnitude_splitting.csv", index=False)
    return R


def fig_n4(X, Xn, out, ks=(5, 8, 12, 20, 30, 50, 80, 120, 200, 300),
           ks_cnmf=(5, 8, 12, 20, 30)):
    """Why the held-out curve never peaks for a hard partition - and what Hamilton's
    in-sample convention would have shown on the same data."""
    import make_heldout_variance as H
    rows = []
    for space, A in (("raw dB", X), ("band-z → unit-norm", Xn)):
        for m, kk in (("kmeans", ks), ("cnmf", ks_cnmf)):
            t0 = time.time()
            d = H.bicv(A, list(kk), m, n_row_folds=3, n_col_folds=3, seed=0, progress=False)
            d["space"], d["method"] = space, m
            rows.append(d)
            print(f"    bi-CV {m:6s} {space:20s} {time.time()-t0:.0f}s", flush=True)
    B = pd.concat(rows)
    g = B.groupby(["space", "method", "k"]).var_explained.agg(["mean", "std"]).reset_index()

    # in-sample R^2 of the k-means partition, Hamilton's Figure S2A convention
    ins = []
    for space, A in (("raw dB", X), ("band-z → unit-norm", Xn)):
        sst = ((A - A.mean(0)) ** 2).sum()
        for k in range(2, 33):
            km = KMeans(k, n_init=10, random_state=42).fit(A)
            ins.append(dict(space=space, k=k, r2=1 - km.inertia_ / sst))
    I = pd.DataFrame(ins)
    I["added"] = I.groupby("space").r2.diff()

    fig = plt.figure(figsize=(11.5, 4.2), dpi=180)
    gs = GridSpec(1, 3, figure=fig, wspace=0.27, left=0.06, right=0.985, top=0.72, bottom=0.14)
    style = {("raw dB", "kmeans"): (RED, "-", "k-means, raw dB"),
             ("band-z → unit-norm", "kmeans"): (GREEN, "-", "k-means, band-z → unit-norm"),
             ("raw dB", "cnmf"): (RED, "--", "convex NMF, raw dB"),
             ("band-z → unit-norm", "cnmf"): (GREEN, "--", "convex NMF, band-z → unit-norm")}

    ax = fig.add_subplot(gs[0])
    for (sp, me), sub in g.groupby(["space", "method"]):
        c, ls, lab = style[(sp, me)]
        ax.plot(sub.k, sub["mean"], ls, color=c, lw=1.5, marker="o", ms=3.2, label=lab)
        if me == "cnmf":
            i = sub["mean"].idxmax()
            ax.plot(sub.loc[i, "k"], sub.loc[i, "mean"], "v", color=c, ms=8)
    ax.set_xscale("log")
    ax.set_xticks(list(ks)); ax.set_xticklabels([str(k) for k in ks])
    ax.axvline(K_MAIN, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("K  (log)"); ax.set_ylabel("held-out variance explained")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("A · bi-cross-validated: the hard methods\nnever turn over, ▼ = an interior peak", loc="left")

    ax = fig.add_subplot(gs[1])
    for sp, sub in I.groupby("space"):
        ax.plot(sub.k, 100 * sub.r2, "o-", ms=3.2, lw=1.5,
                color=RED if sp == "raw dB" else GREEN, label=sp)
    ax.set_xlabel("K"); ax.set_ylabel("% variance explained (in sample)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("B · in-sample R², the convention\nHamilton et al. used (their Fig S2A)", loc="left")

    ax = fig.add_subplot(gs[2])
    for sp, sub in I.groupby("space"):
        ax.plot(sub.k, 100 * sub.added, "o-", ms=3.2, lw=1.5,
                color=RED if sp == "raw dB" else GREEN)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlabel("K"); ax.set_ylabel("% variance ADDED by cluster K")
    ax.set_title("C · the increment they read as an elbow:\nno optimum, only a decay", loc="left")

    fig.suptitle(wrap(
        "FIG N.4 · why 'variance explained vs K' cannot choose K for k-means or Ward — concat_bands5, v8\n"
                 "A held-out row is reconstructed by ONE centroid, so an extra cluster almost never hurts; convex NMF's k graded "
                 "loadings overfit sooner, which is why only its curve has an interior peak (Owen & Perry 2009)."),
        fontsize=10, x=0.06, ha="left", y=0.99)
    fig.savefig(out / "N4_why_no_peak.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    g.to_csv(OUT / "N4_heldout_to_K300.csv", index=False)
    I.to_csv(OUT / "N4_insample_r2.csv", index=False)
    return g, I


def fig_n5(X, Xn, r, nm, lab_db, lab_un, bands, NB, NC, NT, out):
    """The same electrodes, before and after - why the scorecard numbers look like that."""
    U = unit(X)
    C = U @ U.T
    np.fill_diagonal(C, -1)
    iu = np.triu_indices(len(r), 1)
    cos = C[iu]
    ratio = np.maximum(r[iu[0]], r[iu[1]]) / np.minimum(r[iu[0]], r[iu[1]])
    same_db = lab_db[iu[0]] == lab_db[iu[1]]
    same_un = lab_un[iu[0]] == lab_un[iu[1]]

    selA = np.where((cos >= 0.8) & (ratio >= 2.0) & ~same_db & same_un)[0]
    selA = selA[np.argsort(-cos[selA])][:3]
    selB = np.where((cos <= 0.3) & (ratio <= 1.1) & same_db & ~same_un)[0]
    selB = selB[np.argsort(cos[selB])][:3]
    pairs = [(iu[0][k], iu[1][k], "A") for k in selA] + [(iu[0][k], iu[1][k], "B") for k in selB]

    nA = int(((cos >= 0.8) & (ratio >= 2.0)).sum())
    nA_split = int(((cos >= 0.8) & (ratio >= 2.0) & ~same_db).sum())
    nA_split_u = int(((cos >= 0.8) & (ratio >= 2.0) & ~same_un).sum())
    nB = int(((cos <= 0.3) & (ratio <= 1.1)).sum())
    nB_join = int(((cos <= 0.3) & (ratio <= 1.1) & same_db).sum())
    nB_join_u = int(((cos <= 0.3) & (ratio <= 1.1) & same_un).sum())

    fig, axes = plt.subplots(len(pairs), 2, figsize=(11.5, 1.55 * len(pairs) + 1.1),
                             dpi=180, sharex=True)
    t = np.arange(X.shape[1])
    for row, (i, j, kind) in enumerate(pairs):
        for c, (M, lab) in enumerate([(X, "raw dB — what k-means sees without normalisation"),
                                      (Xn, "band-z → unit-norm — what it sees with it")]):
            ax = axes[row, c]
            ax.plot(t, M[i], lw=0.75, color=RED, label=f"{nm[i]}   ‖x‖ {r[i]:.0f}")
            ax.plot(t, M[j], lw=0.75, color=BLUE, label=f"{nm[j]}   ‖x‖ {r[j]:.0f}")
            for b in range(1, NB):
                ax.axvline(b * NC * NT - 0.5, color="#c9d1d9", lw=0.7)
            ax.legend(fontsize=6.5, loc="upper right", frameon=False, ncol=2)
            ax.tick_params(labelsize=6.5)
            ax.set_ylabel("dB" if c == 0 else "unit", fontsize=7.5)
            if row == 0:
                ax.set_title(lab, fontsize=9, loc="left")
            tag = (f"{kind}{row+1 if kind=='A' else row-len(selA)+1} · same shape (cos {C[i,j]:.2f}), "
                   f"{max(r[i],r[j])/min(r[i],r[j]):.1f}× apart in size"
                   if kind == "A" else
                   f"{kind}{row-len(selA)+1} · same size, opposite shape (cos {C[i,j]:+.2f})")
            ax.text(0.005, 0.90, tag, transform=ax.transAxes, fontsize=7.2,
                    color=INK if kind == "A" else MUTED, va="top")
    for ax in axes[-1]:
        ax.set_xticks([(b + 0.5) * NC * NT for b in range(NB)])
        ax.set_xticklabels(bands, fontsize=7)
        ax.set_xlabel("5 bands × (audio | picture | reading) × 30 time bins", fontsize=7.5)
    fig.suptitle(wrap(
        "FIG N.5 · the same electrode pairs before and after — concat_bands5, v8; cluster labels from k-means K=8 fitted in each space\n"
                 f"A: {nA} pairs in the cohort have cos ≥ 0.8 with ≥ 2× size difference — raw dB splits {nA_split} of them, "
                 f"the normalised space {nA_split_u}.   "
                 f"B: {nB:,} pairs have opposite shape at the same size — raw dB puts {nB_join:,} in one cluster, "
                 f"the normalised space {nB_join_u:,}."),
        fontsize=9.5, x=0.02, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out / "N5_pairs_before_after.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return dict(nA=nA, nA_split=nA_split, nA_split_u=nA_split_u,
                nB=nB, nB_join=nB_join, nB_join_u=nB_join_u)


def fig_n6(X, Xn, elev, r, pat, bands, NB, NC, NT, out):
    """The one thing unit-norm does NOT remove: a constant offset. Which cluster is it,
    is it physiology or a baseline problem, and what row-z would do instead."""
    lab_u = KMeans(K_MAIN, n_init=10, random_state=42).fit_predict(Xn)
    lab_z = KMeans(K_MAIN, n_init=10, random_state=42).fit_predict(zrow(Xn))
    means = pd.Series(elev).groupby(lab_u).mean()
    k_lo = int(means.idxmin())
    memb = lab_u == k_lo

    fig = plt.figure(figsize=(11.5, 4.1), dpi=180)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1, 1.15], wspace=0.3,
                  left=0.05, right=0.985, top=0.70, bottom=0.15)

    ax = fig.add_subplot(gs[0])
    C = X[memb].mean(0).reshape(NB, NC * NT)
    v = np.abs(C).max()
    ax.imshow(C, aspect="auto", cmap="bwr", vmin=-v, vmax=v, origin="lower", interpolation="none")
    for x in (NC * NT / 3 - 0.5, 2 * NC * NT / 3 - 0.5):
        ax.axvline(x, color="k", lw=0.6)
    ax.set_yticks(range(NB)); ax.set_yticklabels(bands, fontsize=7)
    ax.set_xticks([15, 45, 75]); ax.set_xticklabels(["audio", "picture", "reading"], fontsize=7.5)
    ax.set_title(f"A · the low-elevation cluster, mean raw dB\nn={int(memb.sum())} electrodes, "
                 f"{pd.Series(pat[memb]).nunique()} patients, ±{v:.1f} dB", loc="left")

    ax = fig.add_subplot(gs[1])
    ax.hist(elev[~memb], bins=45, color="#d6dde4", label="the other 7 clusters")
    ax.hist(elev[memb], bins=45, color=ACC, alpha=0.9, label="this cluster")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("mean dB over all features"); ax.set_ylabel("electrodes")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"B · it is defined by elevation\n(η² of elevation = {eta2(lab_u, elev):.2f} in this space)", loc="left")

    ax = fig.add_subplot(gs[2])
    cnt = pd.Series(lab_z[memb]).value_counts().sort_values(ascending=False)
    share = cnt.iloc[0] / cnt.sum()
    ax.bar(range(len(cnt)), cnt.values, color=[GREEN] + ["#c8d8cb"] * (len(cnt) - 1))
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels([f"c{int(i)}" for i in cnt.index], fontsize=7)
    ax.set_xlabel("cluster it lands in when elevation is removed (row z)")
    ax.set_ylabel("electrodes")
    ax.set_title(f"C · with the offset gone, {100*share:.0f}% of them\nstay together "
                 f"(elevation η² {eta2(lab_u, elev):.2f} → {eta2(lab_z, elev):.2f})", loc="left")

    fig.suptitle(wrap(
        "FIG N.6 · the decision unit-norm does not make for you — concat_bands5, v8, k-means K=8\n"
                 "Unit-norm removes size but keeps a constant offset: a vector that is negative everywhere keeps that as its direction. "
                 "Broadband high-frequency suppression is a documented response (Ossandón et al. 2011; Ramot et al. 2012), so this "
                 "group is only an artefact if the suppression is a baseline problem — check the single trials before choosing "
                 "between unit-norm (keeps elevation) and row z (removes it)."),
        fontsize=9.5, x=0.05, ha="left", y=0.99)
    fig.savefig(out / "N6_elevation_cluster.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return dict(n=int(memb.sum()), n_pat=int(pd.Series(pat[memb]).nunique()),
                eta_u=eta2(lab_u, elev), eta_z=eta2(lab_z, elev), n_after=len(cnt),
                kept_together=float(share))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the bi-CV sweep (FIG N.4)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    X, D, bands, conds = load(RUN_B5)
    n, p = X.shape
    NB, NC, NT = len(bands), len(conds), 30
    assert p == NB * NC * NT, (p, NB, NC, NT)
    pat = D.patient_id.astype(str).to_numpy()
    nm = (D.patient_id.astype(str) + " " + D.electrode.astype(str)).to_numpy()
    r = np.linalg.norm(X, axis=1)
    elev = X.mean(1)
    bandz = make_bandz(NB, NC, NT)
    Bz = bandz(X)
    Xn = unit(Bz)
    dom_band, dom_cond, peak_t = dominant(Bz, NB, NC, NT)
    print(f"[data] {n} electrodes x {p} features, bands {bands}, conditions {conds}")

    T = {
        "raw dB": X,
        "unit-norm": unit(X),
        "row z (correlation distance)": zrow(X),
        "peak-norm  x / max|x|": peaknorm(X),
        "soft norm  x / ‖x‖^0.5": soft(X),
        "band-z  (concat_bands5z)": Bz,
        "band-z → unit-norm": Xn,
        "band-z → row z": zrow(Bz),
        "band-z → soft norm": soft(Bz),
        "per-bin column z  (trap)": make_colz()(X),
        "per-electrode per-band z  (trap)": make_rowz_perband(NB, NC, NT)(X),
    }
    rows = []
    labs = {}
    for name, A in T.items():
        t0 = time.time()
        lab = KMeans(K_MAIN, n_init=10, random_state=42).fit_predict(A)
        wl = AgglomerativeClustering(K_MAIN, linkage="ward").fit_predict(A)
        sz = np.bincount(lab)
        labs[name] = lab
        rows.append(dict(transform=name, amp=eta2(lab, r), elev=eta2(lab, elev),
                         vband=cramers_v(lab, dom_band), vcond=cramers_v(lab, dom_cond),
                         peakt=eta2(lab, peak_t), bal=sz.min() / sz.max(),
                         pat=patient_dominated(lab, pat), stab=stability(A),
                         ward_amp=eta2(wl, r),
                         ward_bal=np.bincount(wl).min() / np.bincount(wl).max()))
        print(f"  {name:34s} amp {rows[-1]['amp']:.2f}  stab {rows[-1]['stab']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    score = pd.DataFrame(rows)
    score.to_csv(OUT / "normalisation_scorecard.csv", index=False)

    ari = pd.DataFrame({a_: {b_: adjusted_rand_score(labs[a_], labs[b_]) for b_ in T} for a_ in T})
    ari.round(3).to_csv(OUT / "normalisation_ari.csv")

    fig_n1(X, r, elev, bands, NB, NC, NT, OUT)
    fig_n2(score, OUT)
    R = fig_n3(X, Xn, r, pat, OUT)
    pairstats = fig_n5(X, Xn, r, nm, labs["raw dB"], labs["band-z → unit-norm"],
                       bands, NB, NC, NT, OUT)
    elstats = fig_n6(X, Xn, elev, r, pat, bands, NB, NC, NT, OUT)
    if not a.quick:
        fig_n4(X, Xn, OUT)

    summary = dict(cohort="v8", run=str(RUN_B5.relative_to(CLUST)), n=n, p=p,
                   bands=bands, conditions=conds, K=K_MAIN,
                   norm_p05=float(np.percentile(r, 5)), norm_med=float(np.median(r)),
                   norm_p95=float(np.percentile(r, 95)),
                   ari_db_vs_unit=float(adjusted_rand_score(labs["raw dB"], labs["band-z → unit-norm"])),
                   pairs=pairstats, elevation=elstats,
                   written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "normalisation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote -> {OUT}")
    print(score.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
