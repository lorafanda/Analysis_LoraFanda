#!/usr/bin/env python3
"""
00_paper2_figure4_correspondence.py - FIG 4, correspondence: do two clusterings find
the same clusters more often than chance?

    python 00_paper2_figure4_correspondence.py                  K = 8, all six pairs
    python 00_paper2_figure4_correspondence.py --k 12
    python 00_paper2_figure4_correspondence.py --n-perm 5000
    python 00_paper2_figure4_correspondence.py --pairs algo     just the three algorithms
    python 00_paper2_figure4_correspondence.py --k1 8 --k2 10   unequal K, one pair

WHY THIS IS NOT FIG 2. FIG 2 matches clusters by shared electrodes and then reports
shared electrodes, so the number it quotes is the number it optimised - a cluster pair
cannot help but overlap once overlap is what chose it. Here the two steps are
independent: clusters are PAIRED by the shape of their mean response, and the pairing is
TESTED on electrode overlap. Nothing about the shape correlation forces two clusters to
contain the same electrodes, so an overlap above chance is evidence and not arithmetic.

THE TWO AXES.

    A  same feature set, different algorithms   convex NMF / k-means / Ward on HFA,
                                                all three pairs
    B  same algorithm, different feature sets   convex NMF on HFA against 5-band,
                                                5-band z-scored and 15-band

ONE DESCRIPTION SPACE FOR EVERY CLUSTER: concat_bands5z. A clustering fitted on HFA and
one fitted on 15 bands live in feature spaces of different size, so their centroids
cannot be correlated as they stand. Every cluster is therefore described by the mean
concat_bands5z representation OF ITS MEMBER ELECTRODES - the cross-space view the
cluster visualizer draws, built here from the concat cache at full precision rather than
from the quantised web bundle. The clustering is whatever it is; only the yardstick is
shared.

TWO WEIGHTINGS, REPORTED SIDE BY SIDE.

    hard        every electrode counts once, in its argmax cluster
    weighted    every electrode counts as P(belonging), the convex NMF loading

The weighted overlap is the weighted Jaccard, sum(min(w1, w2)) / sum(max(w1, w2)). For
k-means and Ward the weights are 0/1 and it reduces to the plain Jaccard exactly, so the
two rows of the table are comparable rather than two different statistics.

WHY THE GRAND MEAN COMES OFF BEFORE MATCHING. Every cluster carries the same
task-evoked response, so raw centroid correlations sit high for every pair and the
greedy order is close to arbitrary. Subtracting the mean across a solution's own
clusters asks what distinguishes a cluster from the average cluster. This cannot bias
the test: matching is a SELECTION, the p-value comes from overlap, and the null redoes
the matching inside every permutation, so any sharpening of the selection is applied to
the null in the same measure. The plain matrix is written beside the centred one.

TWO NULLS, BOTH RE-MATCHED PER DRAW.

    plain           one side's electrodes are reassigned at random, cluster sizes kept
    within-patient  the same, but only within each patient, so a cluster keeps its
                    patient composition and the test asks whether correspondence
                    survives beyond "both solutions grouped the same patient together"

Reassigning changes that side's centroids too, so the greedy matching is redone from the
permuted centroids and the null is the whole procedure under chance, not the last step
of it. The null for a cluster is indexed by the SIDE-1 cluster, which is never permuted.

WHAT IT WRITES, into outputs/paper_figures/

    FIG4_correspondence_K08.png            the six correlation matrices, matches marked
    FIG4_correspondence_K08_caption.txt
    FIG4_matched_K08.csv                   one row per matched pair, both weightings
    FIG4_corr_<comparison>_K08.csv         the full k1 x k2 matrix, centred and plain
    FIG4_summary_K08.csv                   one row per comparison

IT REFUSES TO RUN on solutions fitted on different electrodes, and names the stale run
rather than comparing two cohorts and calling the difference a result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import lf_concat as CC          # noqa: E402
import lf_runs as LR            # noqa: E402

OUT = ROOT / "outputs" / "paper_figures"
SPACE_FSET = "concat_bands5z"           # the yardstick every cluster is described in
ALGO_FSET = "concat_hg"                 # axis A is run on HFA
METHOD_LABEL = {"cnmf": "convex NMF", "kmeans": "k-means", "hierarchical": "Ward"}
FS_LABEL = {"concat_hg": "HFA", "concat_rawds": "15 bands",
            "concat_bands5": "5 bands", "concat_bands5z": "5 bands z"}

INK, MUTED, GREY = "#1b1b1b", "#6b6b6b", "#c9c9c9"

# NOTE ON NAMING: this script says HFA where the runs say concat_hg. The feature set is
# the 70-150 Hz band; "HG" survives only in run ids and stored labels.


def norm(s) -> str:
    """'aH_R-1' -> 'AHR1'. The same rule the runs and the recon side join on."""
    return str(s).replace("_", "").replace("-", "").upper()


# ---- loading ------------------------------------------------------------------
def solution(method: str, fset: str, k: int):
    """One clustering at K: weights (n, k), hard labels, keys, run dir.

    Convex NMF returns its loadings renormalised to sum to 1 - P(belonging) - and the
    hard methods return a one-hot matrix, so everything downstream reads one object and
    the weighted statistics collapse to the hard ones for k-means and Ward by
    construction rather than by a branch.
    """
    try:
        run = LR.newest_run(method, fset)
    except Exception as e:
        raise SystemExit(f"no run for {method}/{fset}: {e}")

    W, lab = None, None
    f = run / "loadings_by_k" / f"G_k{k:02d}.npy"
    if f.exists():
        G = np.load(f).astype(float)
        W = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
        lab = W.argmax(1)
        graded = True
    else:
        lf = run / "cluster_labels_by_k.csv"
        if not lf.exists():
            raise SystemExit(f"{run.name}: neither loadings_by_k/G_k{k:02d}.npy nor "
                             "cluster_labels_by_k.csv - nothing to read at this K")
        tab = pd.read_csv(lf)
        col = f"k_{k}"
        if col not in tab.columns:
            raise SystemExit(f"{run.name}: {lf.name} has no column {col} - the sweep "
                             f"did not include K={k} for {fset}")
        lab = tab[col].to_numpy(int)
        W = np.zeros((len(lab), k))
        W[np.arange(len(lab)), lab] = 1.0
        graded = False

    if W.shape[1] != k:
        raise SystemExit(f"{run.name}: loadings have {W.shape[1]} columns, expected {k}")

    m = pd.read_csv(run / "labels.csv", usecols=["patient_id", "electrode"])
    keys = [f"{p}|{norm(e)}" for p, e in zip(m.patient_id, m.electrode)]
    if len(keys) != len(lab):
        raise SystemExit(f"{run.name}: labels.csv has {len(keys)} rows, the partition "
                         f"has {len(lab)}")
    return dict(W=W, lab=np.asarray(lab, int), keys=keys, run=run, graded=graded,
                method=method, fset=fset, patient=np.array([x.split("|")[0] for x in keys]))


def description_space():
    """The concat_bands5z features of the whole cohort, with their keys.

    Built from the concat cache rather than from any run's X_train, so a cluster found
    in ANY feature set can be described here. The cache pointer resolves to the newest
    concat_source_v<N> on its own.
    """
    cache = CC.DEFAULT_CONCAT_CACHE
    params = json.loads((cache / "params.json").read_text(encoding="utf-8"))
    df, X_concat = CC.build_concat_dataset(
        params["input_dir"], conditions=("audio", "picture", "reading"),
        require_high_activity=True, cache_dir=cache, verbose=False)
    Z = CC.concat_bands5z_features(X_concat, n_blocks=3, fmax_hz=500.0)
    keys = [f"{p}|{norm(e)}" for p, e in zip(df.patient_id, df.electrode)]
    return dict(Z=np.asarray(Z, float), keys=keys, cache=cache.name, n=len(keys))


def align(space, *sols):
    """Prove every solution describes the same electrodes in the same order.

    A comparison of two cohorts would look exactly like a comparison of two algorithms
    and read as a finding, so this refuses instead of reindexing.
    """
    for s in sols:
        if s["keys"] != space["keys"]:
            extra = len(set(s["keys"]) - set(space["keys"]))
            miss = len(set(space["keys"]) - set(s["keys"]))
            raise SystemExit(
                f"{s['method']}/{s['fset']} run {s['run'].name} was fitted on "
                f"{len(s['keys'])} electrodes, the cache ({space['cache']}) holds "
                f"{space['n']}; {extra} not in the cache, {miss} missing. Refit this "
                "track before comparing - the runs are on different cohorts.")


# ---- the measurements ---------------------------------------------------------
def electrode_z(Z):
    """Each electrode scaled to zero mean and unit sd across its own features.

    Shape, not amplitude: without this a centroid is pulled towards whichever of its
    members happens to be loudest, and two solutions would be judged on how they split
    the loud electrodes.
    """
    mu = Z.mean(1, keepdims=True)
    sd = np.maximum(Z.std(1, keepdims=True), 1e-12)
    return (Z - mu) / sd


def centroids(Zz, W):
    """(k, F) weighted mean of the member electrodes. W is P(belonging), or one-hot."""
    den = np.maximum(W.sum(0), 1e-12)[:, None]
    return (W.T @ Zz) / den


def corr_matrix(C1, C2, centre: bool):
    """Pearson r between every centroid of solution 1 and every centroid of solution 2.

    `centre` subtracts each solution's own mean centroid first - see the module
    docstring on why that cannot bias the test.
    """
    A = C1 - C1.mean(0, keepdims=True) if centre else C1
    B = C2 - C2.mean(0, keepdims=True) if centre else C2
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def greedy_match(S):
    """Pairs taken highest correlation first, each cluster used once.

    Returns [(i, j, r), ...] in the order they were taken. With k1 != k2 the surplus
    clusters on the larger side stay unmatched, which is the point: a solution is
    allowed to have found something the other one did not.
    """
    order = np.argsort(-S, axis=None, kind="stable")
    used1, used2, pairs = set(), set(), []
    for flat in order:
        i, j = divmod(int(flat), S.shape[1])
        if i in used1 or j in used2:
            continue
        pairs.append((i, j, float(S[i, j])))
        used1.add(i)
        used2.add(j)
        if len(pairs) == min(S.shape):
            break
    return pairs


def jaccard_hard(a, b):
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def jaccard_weighted(w1, w2):
    """Ruzicka: sum(min) / sum(max). Identical to the plain Jaccard for 0/1 weights."""
    den = float(np.maximum(w1, w2).sum())
    return float(np.minimum(w1, w2).sum()) / den if den else 0.0


def overlaps(s1, s2, pairs, weighted: bool):
    """Overlap of every matched pair, in the requested weighting."""
    out = []
    for i, j, r in pairs:
        if weighted:
            v = jaccard_weighted(s1["W"][:, i], s2["W"][:, j])
        else:
            v = jaccard_hard(s1["lab"] == i, s2["lab"] == j)
        out.append(v)
    return np.array(out, float)


def permutation_index(rng, n, patients=None):
    """A reassignment of electrodes: free, or within patient."""
    if patients is None:
        return rng.permutation(n)
    idx = np.arange(n)
    for p in np.unique(patients):
        sub = np.flatnonzero(patients == p)
        idx[sub] = rng.permutation(sub)
    return idx


def null_distribution(Zz, s1, s2, C1, n_perm, weighted, within, seed):
    """The whole procedure under chance, indexed by the SIDE-1 cluster.

    Side 2's electrodes are reassigned - which changes its centroids as well as its
    membership - then the greedy matching is redone and the matched pair's overlap is
    recorded against the side-1 cluster it fell to. Side 1 is never touched, so every
    draw has an entry for each of its clusters and the observed value has something to
    be compared with.
    """
    rng = np.random.default_rng(seed)
    k1 = C1.shape[0]
    draws = np.full((n_perm, k1), np.nan)
    pats = s2["patient"] if within else None
    n = len(s2["lab"])
    for d in range(n_perm):
        idx = permutation_index(rng, n, pats)
        Wp = s2["W"][idx]
        labp = s2["lab"][idx]
        C2p = centroids(Zz, Wp)
        S = corr_matrix(C1, C2p, centre=True)
        for i, j, _ in greedy_match(S):
            if weighted:
                draws[d, i] = jaccard_weighted(s1["W"][:, i], Wp[:, j])
            else:
                draws[d, i] = jaccard_hard(s1["lab"] == i, labp == j)
    return draws


def empirical(pairs, obs, draws, k1):
    """z and a one-sided p, INDEXED BY THE SIDE-1 CLUSTER, not by match rank.

    The null columns are side-1 clusters because side 1 is the one never permuted; the
    observations arrive in match order. Indexing the two the same way is the difference
    between a p-value and a shuffled p-value, so the mapping is explicit here rather
    than implied by a loop variable.
    """
    z = np.full(k1, np.nan)
    p = np.full(k1, np.nan)
    mu = np.full(k1, np.nan)
    sd = np.full(k1, np.nan)
    for (i, _j, _r), v in zip(pairs, obs):
        col = draws[:, i]
        col = col[~np.isnan(col)]
        if len(col) < 2 or not np.isfinite(v):
            continue
        mu[i], sd[i] = float(col.mean()), float(col.std(ddof=1))
        z[i] = (v - mu[i]) / sd[i] if sd[i] > 1e-12 else np.nan
        p[i] = (1.0 + float((col >= v).sum())) / (len(col) + 1.0)
    return z, p, mu, sd


# ---- one comparison -----------------------------------------------------------
def compare(space, Zz, spec, k1, k2, n_perm, seed, verbose=True):
    """Everything the spec asks for, for one pair of solutions."""
    (m1, f1), (m2, f2) = spec["a"], spec["b"]
    s1 = solution(m1, f1, k1)
    s2 = solution(m2, f2, k2)
    align(space, s1, s2)

    C1 = centroids(Zz, s1["W"])
    C2 = centroids(Zz, s2["W"])
    S_centred = corr_matrix(C1, C2, centre=True)
    S_plain = corr_matrix(C1, C2, centre=False)
    pairs = greedy_match(S_centred)

    rows = []
    for weighted in (False, True):
        obs = overlaps(s1, s2, pairs, weighted)
        t0 = time.time()
        d_plain = null_distribution(Zz, s1, s2, C1, n_perm, weighted, False, seed)
        d_within = null_distribution(Zz, s1, s2, C1, n_perm, weighted, True, seed + 1)
        zp, pp, mup, sdp = empirical(pairs, obs, d_plain, C1.shape[0])
        zw, pw, muw, sdw = empirical(pairs, obs, d_within, C1.shape[0])
        if verbose:
            print(f"    {'weighted' if weighted else 'hard    '} "
                  f"{n_perm} x 2 permutations in {time.time() - t0:.0f}s")
        for rank, ((i, j, r), v) in enumerate(zip(pairs, obs), start=1):
            n1 = float(s1["W"][:, i].sum()) if weighted else int((s1["lab"] == i).sum())
            n2 = float(s2["W"][:, j].sum()) if weighted else int((s2["lab"] == j).sum())
            rows.append(dict(
                comparison=spec["name"], axis=spec["axis"], rank=rank,
                weighting="weighted" if weighted else "hard",
                cluster_1=i, cluster_2=j,
                r_centred=r, r_plain=float(S_plain[i, j]),
                overlap=v, n_1=n1, n_2=n2,
                null_plain_mean=mup[i], null_plain_sd=sdp[i], z_plain=zp[i], p_plain=pp[i],
                null_within_mean=muw[i], null_within_sd=sdw[i],
                z_within=zw[i], p_within=pw[i]))

    matched = np.array([S_centred[i, j] for i, j, _ in pairs], float)
    off = S_centred.copy()
    for i, j, _ in pairs:
        off[i, j] = np.nan
    summary = dict(
        comparison=spec["name"], axis=spec["axis"],
        method_1=m1, feature_set_1=f1, run_1=s1["run"].name, k_1=k1, graded_1=s1["graded"],
        method_2=m2, feature_set_2=f2, run_2=s2["run"].name, k_2=k2, graded_2=s2["graded"],
        n_matched=len(pairs),
        r_matched_mean=float(matched.mean()), r_matched_min=float(matched.min()),
        r_unmatched_mean=float(np.nanmean(off)),
        r_gap=float(matched.mean() - np.nanmean(off)))
    df = pd.DataFrame(rows)
    for w in ("hard", "weighted"):
        sub = df[df.weighting == w]
        summary[f"overlap_{w}_mean"] = float(sub.overlap.mean())
        summary[f"n_sig_plain_{w}"] = int((sub.p_plain < 0.05).sum())
        summary[f"n_sig_within_{w}"] = int((sub.p_within < 0.05).sum())
    return dict(spec=spec, s1=s1, s2=s2, S=S_centred, S_plain=S_plain,
                pairs=pairs, rows=df, summary=summary)


# ---- the figure ---------------------------------------------------------------
def draw(results, k1, k2, path, n_perm):
    n = len(results)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow),
                             squeeze=False, facecolor="white")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, res in zip(axes.ravel(), results):
        ax.axis("on")
        S = res["S"]
        im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        hard = res["rows"][res["rows"].weighting == "hard"].set_index("rank")
        for rank, (i, j, r) in enumerate(res["pairs"], start=1):
            row = hard.loc[rank]
            star = "***" if row.p_plain < 0.001 else "**" if row.p_plain < 0.01 \
                else "*" if row.p_plain < 0.05 else ""
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor=INK, lw=1.6))
            ax.text(j, i, f"{row.overlap:.2f}{star}", ha="center", va="center",
                    fontsize=6.5, color=INK if abs(r) < .55 else "white")
        sp = res["spec"]
        ax.set_title(sp["title"], fontsize=9, color=INK, pad=6)
        ax.set_xlabel(sp["lab_b"], fontsize=8, color=MUTED)
        ax.set_ylabel(sp["lab_a"], fontsize=8, color=MUTED)
        ax.set_xticks(range(S.shape[1]))
        ax.set_yticks(range(S.shape[0]))
        ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
    cax = fig.add_axes([0.35, 0.035, 0.30, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("centroid correlation, grand mean removed", fontsize=7.5, color=MUTED)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    fig.suptitle(f"Cluster correspondence at K = {k1}"
                 + (f" vs {k2}" if k2 != k1 else "")
                 + f"   ·   cell = Jaccard, {n_perm} permutations",
                 fontsize=10.5, color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def caption(results, k1, k2, space, n_perm):
    L = [f"FIG 4  Cluster correspondence, K = {k1}"
         + (f" against K = {k2}" if k2 != k1 else "") + ".",
         "",
         "Clusters from two solutions are PAIRED by the correlation between their mean "
         f"{FS_LABEL[SPACE_FSET]} responses, each electrode z-scored across its own "
         "features first and the mean across a solution's clusters removed, and the "
         "pairing is TESTED on how many electrodes the paired clusters share. The two "
         "quantities are independent: nothing about a shape correlation forces two "
         "clusters to hold the same electrodes.",
         "",
         f"Every cluster is described in the same space ({SPACE_FSET}, from "
         f"{space['cache']}, {space['n']} electrodes) whatever feature set produced it, "
         "so a clustering fitted on HFA and one fitted on 15 bands can be compared at "
         "all. Cells give the Jaccard overlap of the matched pair; stars are the "
         f"permutation p against the plain null ({n_perm} draws, cluster sizes kept, "
         "the matching redone in every draw). * p<0.05, ** p<0.01, *** p<0.001.",
         ""]
    for res in results:
        s = res["summary"]
        hard = res["rows"][res["rows"].weighting == "hard"]
        L.append(f"{s['comparison']}: matched r {s['r_matched_mean']:+.2f} against "
                 f"{s['r_unmatched_mean']:+.2f} unmatched (gap {s['r_gap']:+.2f}); "
                 f"mean Jaccard {s['overlap_hard_mean']:.2f} hard, "
                 f"{s['overlap_weighted_mean']:.2f} weighted; "
                 f"{s['n_sig_plain_hard']} of {len(hard)} pairs above the plain null, "
                 f"{s['n_sig_within_hard']} above the within-patient null. "
                 f"Runs {s['run_1']} and {s['run_2']}.")
    L += ["",
          "The within-patient null keeps each cluster's patient composition, so a pair "
          "that clears it corresponds beyond the two solutions having grouped the same "
          "patient together. Weighted rows use P(belonging) as the weight and the "
          "weighted Jaccard, which is the plain Jaccard exactly for k-means and Ward.",
          f"Written {datetime.now():%Y-%m-%d %H:%M}."]
    return "\n".join(L)


# ---- writers ------------------------------------------------------------------
def write_verified(path: Path, text: str):
    """The share truncates files on a failed write, so read it back before moving on."""
    path.write_text(text, encoding="utf-8")
    if path.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{path} did not survive the write")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="K on both sides")
    ap.add_argument("--k1", type=int, default=None, help="K on side 1, if it differs")
    ap.add_argument("--k2", type=int, default=None, help="K on side 2, if it differs")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", choices=["all", "algo", "feature"], default="all")
    ap.add_argument("--algo-feature-set", default=ALGO_FSET)
    a = ap.parse_args()
    k1 = a.k1 or a.k
    k2 = a.k2 or a.k

    af = a.algo_feature_set
    specs = []
    if a.pairs in ("all", "algo"):
        for m1, m2 in (("cnmf", "kmeans"), ("cnmf", "hierarchical"),
                       ("kmeans", "hierarchical")):
            specs.append(dict(
                axis="A algorithms", a=(m1, af), b=(m2, af),
                name=f"{m1}_vs_{m2}_{af}",
                title=f"{METHOD_LABEL[m1]} vs {METHOD_LABEL[m2]}\non {FS_LABEL[af]}",
                lab_a=METHOD_LABEL[m1], lab_b=METHOD_LABEL[m2]))
    if a.pairs in ("all", "feature"):
        for f2 in ("concat_bands5", "concat_bands5z", "concat_rawds"):
            specs.append(dict(
                axis="B feature sets", a=("cnmf", "concat_hg"), b=("cnmf", f2),
                name=f"cnmf_concat_hg_vs_{f2}",
                title=f"convex NMF\n{FS_LABEL['concat_hg']} vs {FS_LABEL[f2]}",
                lab_a=FS_LABEL["concat_hg"], lab_b=FS_LABEL[f2]))

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  description space: {SPACE_FSET}")
    space = description_space()
    print(f"  cohort: {space['cache']}, {space['n']} electrodes, "
          f"{space['Z'].shape[1]} features")
    Zz = electrode_z(space["Z"])

    results = []
    for spec in specs:
        print(f"  {spec['name']}")
        results.append(compare(space, Zz, spec, k1, k2, a.n_perm, a.seed))

    tag = f"K{k1:02d}" + (f"_{k2:02d}" if k2 != k1 else "")
    allrows = pd.concat([r["rows"] for r in results], ignore_index=True)
    allrows.to_csv(OUT / f"FIG4_matched_{tag}.csv", index=False)
    pd.DataFrame([r["summary"] for r in results]).to_csv(
        OUT / f"FIG4_summary_{tag}.csv", index=False)
    for r in results:
        k_a, k_b = r["S"].shape
        pd.DataFrame(
            [dict(cluster_1=i, cluster_2=j, r_centred=float(r["S"][i, j]),
                  r_plain=float(r["S_plain"][i, j]),
                  matched=any(i == p[0] and j == p[1] for p in r["pairs"]))
             for i in range(k_a) for j in range(k_b)]
        ).to_csv(OUT / f"FIG4_corr_{r['spec']['name']}_{tag}.csv", index=False)

    png = OUT / f"FIG4_correspondence_{tag}.png"
    draw(results, k1, k2, png, a.n_perm)
    write_verified(OUT / f"FIG4_correspondence_{tag}_caption.txt",
                   caption(results, k1, k2, space, a.n_perm))

    print(f"\n  {png.name}")
    for r in results:
        s = r["summary"]
        print(f"    {s['comparison']:<38} matched r {s['r_matched_mean']:+.2f} vs "
              f"{s['r_unmatched_mean']:+.2f}   Jaccard {s['overlap_hard_mean']:.2f} "
              f"hard / {s['overlap_weighted_mean']:.2f} weighted   "
              f"{s['n_sig_plain_hard']} sig plain, {s['n_sig_within_hard']} within")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
