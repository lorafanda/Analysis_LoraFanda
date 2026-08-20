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

    python measure_cluster_stability.py --k 7
    python measure_cluster_stability.py --k 7 --boot 30 --seeds 12
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


def jaccard(a_idx, b_idx):
    a, b = set(a_idx.tolist()), set(b_idx.tolist())
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def best_jaccard(base_members, other_labels, other_ids):
    """Best recovery of one original cluster by any cluster of another partition."""
    return max((jaccard(base_members, np.where(other_labels == j)[0])
                for j in other_ids), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--boot", type=int, default=25, help="bootstrap resamples")
    ap.add_argument("--seeds", type=int, default=10, help="random restarts")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 6, 8, 9, 10])
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
    # Resample WITH replacement, refit, map each original cluster to its best match.
    # Duplicated rows are fine: the question is whether the cluster reappears, and
    # Hennig's procedure is defined on exactly this resampling.
    boot = {j: [] for j in ids}
    for b in range(a.boot):
        rng = np.random.default_rng(1000 + b)
        take = rng.integers(0, n, n)
        lb = fit_labels(Xu[take], K, 0)
        # carry labels back to original row ids; a row drawn twice votes twice, so
        # take the label it most often received
        back = {}
        for pos, orig in enumerate(take):
            back.setdefault(orig, []).append(lb[pos])
        lab_back = np.full(n, -1)
        for orig, v in back.items():
            lab_back[orig] = np.bincount(v).argmax()
        seen = [j for j in ids if (lab_back == j).any()]
        for j in ids:
            boot[j].append(best_jaccard(members[j], lab_back, seen))
        print(f"    bootstrap {b + 1}/{a.boot}", end="\r")
    print(" " * 40, end="\r")

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

    stats = {"run": str(RUN.relative_to(CLUST)), "K": K, "n": int(n),
             "n_boot": a.boot, "n_seeds": a.seeds, "ks_compared": a.ks,
             "clusters": {}}
    for j in ids:
        stats["clusters"][str(j)] = dict(
            size=sizes[j],
            bootstrap_jaccard=float(np.mean(boot[j])),
            bootstrap_sd=float(np.std(boot[j])),
            init_jaccard=float(np.mean(init[j])),
            cross_k={str(k): float(v) for k, v in cross[j].items()},
            cross_k_mean=float(np.mean(list(cross[j].values()))),
            silhouette=sil_by[j])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cluster_stability_K{K}.json").write_text(json.dumps(stats, indent=2),
                                                      encoding="utf-8")

    print()
    kcols = "  ".join("%5s" % ("K%d" % k) for k in a.ks)
    print("  %3s %5s %8s %6s %7s %7s  %s  %6s  verdict"
          % ("cl", "n", "boot J", "+-sd", "init J", "crossK", kcols, "sil"))
    for j in ids:
        c = stats["clusters"][str(j)]
        bj = c["bootstrap_jaccard"]
        verdict = ("stable" if bj > 0.75 else
                   "pattern" if bj > 0.60 else "not a cluster")
        row = "  ".join(f"{c['cross_k'][str(k)]:5.2f}" for k in a.ks)
        print(f"  c{j:>2} {c['size']:>5} {bj:>8.2f} {c['bootstrap_sd']:>6.2f} "
              f"{c['init_jaccard']:>7.2f} {c['cross_k_mean']:>7.2f}  {row}  "
              f"{c['silhouette']:>6.3f}  {verdict}")
    print()
    print("  Hennig 2007: bootstrap Jaccard > 0.75 stable, 0.60-0.75 a pattern,")
    print("  < 0.60 not a real cluster. Cross-K is the best Jaccard against the")
    print("  clusters found when a different number of them is requested.")
    print(f"  -> {OUT / f'cluster_stability_K{K}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
