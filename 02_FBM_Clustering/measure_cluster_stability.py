#!/usr/bin/env python3
"""
measure_cluster_stability.py - how stable is each individual cluster, and under what?

"Is this cluster stable" is not one question. A cluster can survive resampling the
electrodes and still dissolve when you change K, or be reproducible from any random
start and still be an arbitrary slice of a continuum. This measures four things that
are usually conflated, per cluster, and writes them to JSON so a figure or caption
cannot drift from them.

    1. BOOTSTRAP JACCARD  (Hennig 2007, the standard for this question)
       Resample electrodes with replacement, refit, and ask how much of the original
       cluster is recovered by its best match. Hennig's rule of thumb: > 0.75 is
       stable, 0.60-0.75 is a pattern but not a firm one, < 0.60 is not a real
       cluster. This is the measure that answers "would I see this again in another
       sample of electrodes".

    2. INITIALISATION STABILITY
       Refit at the same K from different random starts. Norman-Haignere et al.
       (2019) kept components correlating > 0.9 across 1000 inits. A cluster that
       moves between seeds is an artefact of the optimiser, not of the data. This is
       a WEAKER test than the bootstrap: the data never changes.

    3. CROSS-K PERSISTENCE
       Best Jaccard against the clusters found at neighbouring K. Answers "does this
       cluster survive being asked for a different number of them" - the question
       that matters when K itself is not identified, which on this dataset it is not.

    4. PER-CLUSTER SILHOUETTE
       Compactness relative to the nearest other cluster, in the space the method
       fits in. Reported for completeness; it says nothing about reproducibility.

A cluster is only worth naming if it holds up on 1 and 3. 2 alone is cheap to pass.

CORRECTION 2026-08-21. Every bootstrap number this script produced before today was
biased down by a factor of 1 - 1/e, and the conclusion drawn from them - that none of
the seven K=7 clusters survives a bootstrap - was an artefact of the bug and is
withdrawn. See bootstrap_jaccard() for the mechanism and the verification.

Three things the corrected version adds, because a bare mean over 25 resamples was
never enough to hang a pass/fail on:

    * a CONFIDENCE INTERVAL on each mean, so a cluster is judged by its interval and
      not by which side of the threshold its point estimate happens to land;
    * a CALIBRATED NULL, because Hennig's 0.60 is a rule of thumb rather than a level
      derived for a given K, n and set of cluster sizes - a cluster holding half the
      data scores high by construction;
    * the full per-resample values, so it is possible to SEE whether the number of
      resamples was enough rather than assuming it.

    python measure_cluster_stability.py --k 7
    python measure_cluster_stability.py --k 7 --boot 30 --seeds 12 --null 3
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "comparison"
RUN = CLUST / "cnmf/concat_hg/runs/20260818_112939"


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def fit_labels(Xu, K, seed):
    import lf_decompose as LD
    G = LD.convex_nmf(Xu, K, n_iter=300, random_state=seed)[1]
    return (G / np.maximum(G.sum(1, keepdims=True), 1e-12)).argmax(1)


# Methods differ in the space they fit in, and silhouette is not space-free -
# scoring all of them in dB is the error that made the first separation figure wrong.
SPACE = {"cnmf": "unit-norm", "kmeans": "dB", "hierarchical": "dB"}


def resolve(method, feature_set):
    """(newest run dir, space) for a track, from index.json."""
    idx = json.loads((CLUST / "index.json").read_text(encoding="utf-8"))
    runs = idx["runs"] if isinstance(idx, dict) else idx
    best = None
    for r in runs:
        if r["method"] != method or r["feature_set"] != feature_set:
            continue
        rd = CLUST / method / feature_set / "runs" / r["run_id"]
        if (rd / "X_train.npy").exists() and (best is None or r["run_id"] > best[0]):
            best = (r["run_id"], rd)
    if best is None:
        raise FileNotFoundError(f"no run with X_train for {method}/{feature_set}")
    return best[1], SPACE[method]


def fit_any(A, K, seed, method):
    """Labels at K, by whichever method - so the same filters apply to all three.

    Ward is DETERMINISTIC: it has no random initialisation, so its init-stability
    score is 1.0 by construction and means nothing. The notebook says so rather than
    reporting it as if it had passed a test.
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(K, n_init=10, random_state=seed).fit_predict(A)
    if method == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(A)
    return fit_labels(A, K, seed)


def patient_share(labels, patients, Gn=None):
    """Largest single-patient share of each cluster.

    Convex NMF has loadings, so the share is of component WEIGHT, matching
    Norman-Haignere Fig 1F. A hard partition has no weights, so it is the share of
    MEMBERS instead. The two are not the same quantity and the threshold carries over
    only loosely.
    """
    ids = sorted(set(int(v) for v in labels))
    pats = sorted(set(patients))
    out = {}
    for j in ids:
        if Gn is not None:
            w = np.array([Gn[patients == p, j].sum() for p in pats])
        else:
            m = labels == j
            w = np.array([float((patients[m] == p).sum()) for p in pats])
        w = w / max(w.sum(), 1e-12)
        out[j] = (float(w.max()), pats[int(w.argmax())])
    return out


def jaccard(a_idx, b_idx):
    a, b = set(a_idx.tolist()), set(b_idx.tolist())
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def best_jaccard(base_members, other_labels, other_ids):
    """Best recovery of one original cluster by any cluster of another partition."""
    return max((jaccard(base_members, np.where(other_labels == j)[0])
                for j in other_ids), default=0.0)


# ==============================================================================
# 1. bootstrap
# ==============================================================================
def bootstrap_jaccard(A, members, K, method, n_boot=25, seed0=1000, progress=True):
    """Hennig 2007 bootstrap, scored on the rows the refit actually saw.

    Returns {cluster_id: np.ndarray of length n_boot} - every per-resample value, not
    a summary, so the caller can put an interval on it and check convergence.

    THE OUT-OF-BAG ROWS ARE WHY THIS FUNCTION EXISTS. A bootstrap of size n draws each
    row with probability 1 - (1 - 1/n)**n, which tends to 1 - 1/e = 0.632. The rows it
    never draws cannot possibly be recovered by the refit. If they are left in the
    original cluster while being absent from the refit's, they sit in the Jaccard
    DENOMINATOR and never in the numerator, so the statistic is capped at 0.632 even
    when recovery is perfect.

    Verified rather than argued: feeding this procedure an exact reproduction of the
    partition returns 0.6322 +/- 0.0084 over 200 resamples, against 1 - 1/e = 0.6321.
    An earlier version of this file did exactly that and compared the result to
    Hennig's 0.60 and 0.75, which are defined on a scale where 1.0 means perfect
    recovery. On the real K=7 cNMF run the bug cost 0.23-0.31 Jaccard per cluster and
    turned seven passes into none.

    Restricting the original cluster to the in-bag rows - below - is Hennig's own
    convention and returns 1.0 on perfect recovery.

    The alternative convention is to ASSIGN the held-out rows to the bootstrap
    solution and score on the full set, which measures out-of-sample prediction rather
    than agreement on shared rows. It is arguably the better question but needs a
    predict step per method (for convex NMF, projection onto the bootstrap's W), so it
    is not what runs here. Both return 1.0 on perfect recovery, which is the property
    that was missing.

    The refit uses a FIXED seed. Resampling variance is the thing being measured here;
    initialisation variance is filter B's job, and letting both move at once would
    confound them.
    """
    n = len(A)
    ids = sorted(members)
    out = {j: [] for j in ids}
    for b in range(n_boot):
        take = np.random.default_rng(seed0 + b).integers(0, n, n)
        lb = fit_any(A[take], K, 0, method)

        # One label per unique in-bag row. A row drawn three times votes three times,
        # so take the label it most often received.
        votes = {}
        for pos, orig in enumerate(take):
            votes.setdefault(orig, []).append(lb[pos])
        back = np.full(n, -1)
        for orig, v in votes.items():
            back[orig] = np.bincount(v).argmax()

        inbag = np.unique(take)
        seen = sorted(set(int(v) for v in back if v >= 0))
        for j in ids:
            out[j].append(best_jaccard(np.intersect1d(members[j], inbag), back, seen))
        if progress:
            print(f"    bootstrap {b + 1}/{n_boot}", end="\r")
    if progress:
        print(" " * 40, end="\r")
    return {j: np.asarray(v, float) for j, v in out.items()}


def ci95(v):
    """(mean, lo, hi) for the MEAN of v, t-based.

    The question is whether the cluster's TRUE mean Jaccard clears the threshold, so
    the interval is on the mean and narrows with more resamples - it is not a spread
    of the individual values. Judging a threshold by the point estimate alone treats
    0.59 +/- 0.02 and 0.59 +/- 0.15 as the same verdict, which they are not.
    """
    from scipy import stats
    v = np.asarray(v, float)
    m, k = float(v.mean()), len(v)
    if k < 2:
        return m, m, m
    h = float(stats.t.ppf(0.975, k - 1) * v.std(ddof=1) / np.sqrt(k))
    return m, m - h, m + h


# ==============================================================================
# 2. null calibration
# ==============================================================================
def surrogate(X, seed):
    """Gaussian with the data's own covariance: correlated features, smooth time
    courses, ONE blob, no cluster structure. White noise would be trivially beatable
    and would prove nothing. Same generator as make_separation_figure.py.
    """
    Xc = X - X.mean(0)
    n = len(X)
    S, Vt = np.linalg.svd(Xc, full_matrices=False)[1:]
    Z = np.random.default_rng(seed).standard_normal((n, len(S)))
    return (Z * (S / np.sqrt(n - 1))) @ Vt


def null_bootstrap(X, K, method, space, n_null=2, n_boot=15, seed0=7000,
                   progress=True):
    """Per-cluster bootstrap Jaccard on structureless data with the same covariance.

    WHY THIS IS NEEDED. Hennig's 0.60 and 0.75 are rules of thumb, not levels derived
    for a given K, n and set of cluster sizes. Jaccard has a floor that moves with all
    three: a partition into few big clusters reproduces itself far more easily than one
    into many small ones, because a large cluster overlaps its best match by a lot
    whatever the data. So "stability degrades as K rises" is partly arithmetic, and a
    fixed threshold is a different test at every K.

    Returns the pooled per-cluster values from n_null surrogate datasets. The level to
    beat is a high percentile of that pool: a real cluster has to be more reproducible
    than what one-blob data of the same shape produces at the same K.

    Cluster sizes on the null are not matched one-to-one to the observed ones, so this
    calibrates K and n but only approximately calibrates size. A cluster much larger
    than anything the null produced should be read with that in mind.
    """
    vals = []
    for i in range(n_null):
        Y = surrogate(X, seed0 + i)
        Y = unit(Y) if space == "unit-norm" else Y
        base_y = fit_any(Y, K, 0, method)
        mem_y = {j: np.where(base_y == j)[0]
                 for j in sorted(set(int(v) for v in base_y))}
        bj = bootstrap_jaccard(Y, mem_y, K, method, n_boot=n_boot,
                               seed0=seed0 + 100 * (i + 1), progress=False)
        vals.extend(float(v.mean()) for v in bj.values())
        if progress:
            print(f"    null {i + 1}/{n_null}, mean J so far {np.mean(vals):.3f}",
                  end="\r")
    if progress:
        print(" " * 50, end="\r")
    return np.asarray(vals, float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--boot", type=int, default=25, help="bootstrap resamples")
    ap.add_argument("--seeds", type=int, default=10, help="random restarts")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 6, 8, 9, 10])
    ap.add_argument("--null", type=int, default=2,
                    help="surrogate datasets for the null level (0 to skip)")
    a = ap.parse_args()

    from sklearn.metrics import silhouette_samples

    X = np.load(RUN / "X_train.npy").astype(float)
    Xu = unit(X)                      # the space convex NMF actually fits in
    n = len(Xu)
    K = a.k

    base = fit_labels(Xu, K, 0)
    ids = list(range(K))
    members = {j: np.where(base == j)[0] for j in ids}
    sizes = {j: int(len(members[j])) for j in ids}
    print(f"  reference fit K={K}: sizes {[sizes[j] for j in ids]}")

    # ---- 1. bootstrap Jaccard
    boot = bootstrap_jaccard(Xu, members, K, "cnmf", n_boot=a.boot)

    # ---- 2. initialisation stability
    init = {j: [] for j in ids}
    for s in range(1, a.seeds + 1):
        ls = fit_labels(Xu, K, s)
        for j in ids:
            init[j].append(best_jaccard(members[j], ls, ids))

    # ---- 3. cross-K persistence
    cross = {j: {} for j in ids}
    for k2 in a.ks:
        l2 = fit_labels(Xu, k2, 0)
        for j in ids:
            cross[j][k2] = best_jaccard(members[j], l2, list(range(k2)))

    # ---- 4. per-cluster silhouette, in the fitting space
    sil = silhouette_samples(Xu, base)
    sil_by = {j: float(sil[base == j].mean()) for j in ids}

    # ---- 5. what the same procedure gives on structureless data
    null_level = None
    if a.null > 0:
        nv = null_bootstrap(X, K, "cnmf", "unit-norm", n_null=a.null,
                            n_boot=max(8, a.boot // 2))
        null_level = float(np.percentile(nv, 95))
        print(f"  null: {len(nv)} clusters over {a.null} surrogates, "
              f"mean {nv.mean():.3f}, 95th pct {null_level:.3f}")

    stats = {"run": str(RUN.relative_to(CLUST)), "K": K, "n": int(n),
             "n_boot": a.boot, "n_seeds": a.seeds, "ks_compared": a.ks,
             "n_null": a.null, "null_p95": null_level,
             "bootstrap_scoring": "in-bag (Hennig); perfect recovery = 1.0",
             "clusters": {}}
    for j in ids:
        m, lo, hi = ci95(boot[j])
        stats["clusters"][str(j)] = dict(
            size=sizes[j],
            bootstrap_jaccard=float(m),
            bootstrap_ci=[float(lo), float(hi)],
            bootstrap_sd=float(boot[j].std(ddof=1)),
            bootstrap_values=[float(v) for v in boot[j]],
            init_jaccard=float(np.mean(init[j])),
            cross_k={str(k): float(v) for k, v in cross[j].items()},
            cross_k_mean=float(np.mean(list(cross[j].values()))),
            silhouette=sil_by[j])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cluster_stability_K{K}.json").write_text(json.dumps(stats, indent=2),
                                                      encoding="utf-8")

    print()
    kcols = "  ".join("%5s" % ("K%d" % k) for k in a.ks)
    print("  %3s %5s %7s %14s %7s %7s  %s  %6s  verdict"
          % ("cl", "n", "boot J", "95% CI", "init J", "crossK", kcols, "sil"))
    for j in ids:
        c = stats["clusters"][str(j)]
        lo, hi = c["bootstrap_ci"]
        # the verdict comes from the LOWER bound, so a cluster whose interval
        # straddles the threshold is not counted as having cleared it
        verdict = ("stable" if lo > 0.75 else
                   "pattern" if lo > 0.60 else "not a cluster")
        if null_level is not None and lo <= null_level:
            verdict += ", at null"
        row = "  ".join(f"{c['cross_k'][str(k)]:5.2f}" for k in a.ks)
        print(f"  c{j:>2} {c['size']:>5} {c['bootstrap_jaccard']:>7.2f} "
              f" [{lo:.2f}, {hi:.2f}] {c['init_jaccard']:>7.2f} "
              f"{c['cross_k_mean']:>7.2f}  {row}  {c['silhouette']:>6.3f}  {verdict}")
    print()
    print("  Hennig 2007: bootstrap Jaccard > 0.75 stable, 0.60-0.75 a pattern,")
    print("  < 0.60 not a real cluster - read off the CI lower bound, not the mean.")
    if null_level is not None:
        print(f"  Structureless data with this covariance reaches {null_level:.2f} at")
        print(f"  K={K}, so a cluster below that is at chance whatever Hennig says.")
    print("  Cross-K is the best Jaccard against the clusters found when a different")
    print("  number of them is requested.")
    print(f"  -> {OUT / f'cluster_stability_K{K}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
