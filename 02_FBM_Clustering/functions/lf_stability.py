"""
lf_stability.py — Consensus-clustering stability analysis.

Implements the post-hoc validation analyses recommended for the FBM
clustering pipeline (Monti et al. 2003; Hennig 2007):

    Monti, S., Tamayo, P., Mesirov, J. & Golub, T. (2003).
        Consensus clustering: A resampling-based method for class discovery
        and visualization of gene expression microarray data.
        Machine Learning 52, 91–118.

    Hennig, C. (2007).
        Cluster-wise assessment of cluster stability.
        Computational Statistics & Data Analysis 52(1), 258–271.

The core question this answers: "If I re-run KMeans with a different seed,
how often do the same pairs of samples land in the same cluster?" Stable
clusters survive resampling; unstable ones are an artifact of one
particular seed and should be down-weighted in interpretation.

Public API:
    compute_consensus_matrix(X, k, n_runs=50, random_state=0)
        -> (n_samples, n_samples) float32 in [0, 1]; entry (i,j) = fraction
           of n_runs in which samples i and j were in the same cluster.

    per_cluster_jaccard(labels, consensus_matrix)
        -> dict {cluster_id: jaccard_stability_score in [0,1]}.
           Higher = cluster is reproducible across resamplings.

    save_consensus_artifacts(run_dir, X, labels, *, n_runs=50, random_state=0)
        -> writes:
           consensus_matrix.npy        (n×n float32; gitignored)
           per_cluster_stability.csv   (cluster_id, size, jaccard, sil)
           consensus_heatmap.png       (samples reordered by cluster)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ============================================================
# Core: consensus matrix
# ============================================================
def compute_consensus_matrix(
    X: np.ndarray,
    k: int,
    *,
    n_runs: int = 50,
    random_state: int = 0,
    n_init: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run KMeans n_runs times with different random_state offsets, accumulate
    the per-pair co-clustering frequency.

    For large n_samples (>~3000) this allocates an n×n float32 matrix which
    can be hundreds of MB. KMeans itself is the slow part — at K=20 and
    1500 samples, ~50 runs takes a couple of minutes.
    """
    n = X.shape[0]
    consensus = np.zeros((n, n), dtype=np.float32)

    for r in range(n_runs):
        seed = int(random_state + r)
        km = KMeans(n_clusters=int(k), random_state=seed, n_init=n_init).fit(X)
        lbl = km.labels_
        # Indicator: same cluster -> add 1 to (i,j) for all pairs in same group.
        # Vectorized: for each cluster c, get its sample indices, set their
        # outer product block to +=1. Faster than pairwise comparison.
        for c in np.unique(lbl):
            idx = np.where(lbl == c)[0]
            if idx.size > 1:
                # np.ix_ creates a row+col selector for a block update
                consensus[np.ix_(idx, idx)] += 1.0
        if verbose and ((r + 1) % 10 == 0):
            print(f"  consensus run {r+1}/{n_runs}")

    # Normalize: each pair was seen in n_runs runs (every sample assigned every run)
    consensus /= float(n_runs)
    return consensus


# ============================================================
# Per-cluster Jaccard stability (Hennig 2007 style)
# ============================================================
def per_cluster_jaccard(
    labels: np.ndarray,
    consensus_matrix: np.ndarray,
) -> Dict[int, float]:
    """
    For each cluster in `labels`, compute the average co-clustering
    frequency among its own members (intra-cluster mean of the consensus
    matrix). Equivalent to the simplest Hennig stability measure: mean
    pairwise co-occurrence within the labelled group.

    A value of 1.0 means every pair of samples in this cluster ended up
    together in every consensus run -> perfectly stable cluster.
    A value of 0.5 means pairs were together only half the time -> very
    unstable.
    """
    out: Dict[int, float] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size < 2:
            out[int(c)] = float("nan")
            continue
        block = consensus_matrix[np.ix_(idx, idx)]
        # Exclude the diagonal (which is always 1.0)
        mask = ~np.eye(block.shape[0], dtype=bool)
        out[int(c)] = float(block[mask].mean())
    return out


# ============================================================
# Save artifacts into a clustering-run directory
# ============================================================
def save_consensus_artifacts(
    run_dir,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_runs: int = 50,
    random_state: int = 0,
    verbose: bool = True,
):
    """
    Compute the consensus matrix + per-cluster Jaccard stability for an
    existing clustering run, write artifacts into the run dir:

        consensus_matrix.npy          (n×n float32, gitignored)
        per_cluster_stability.csv     (cluster_id, size, jaccard, n_runs)
        consensus_heatmap.png         (samples reordered by cluster — tight
                                        blocks on the diagonal = stable
                                        clusters; off-diagonal smear = unstable)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels).astype(np.int32)
    n = X.shape[0]
    assert labels.shape[0] == n, f"labels ({labels.shape[0]}) vs X ({n}) mismatch"

    k = int(len(np.unique(labels)))
    if verbose:
        print(f"[stability] consensus_matrix at K={k}, n_runs={n_runs}, n={n}")

    M = compute_consensus_matrix(X, k, n_runs=n_runs, random_state=random_state,
                                  verbose=verbose)
    np.save(run_dir / "consensus_matrix.npy", M)

    jacc = per_cluster_jaccard(labels, M)

    # Per-cluster summary CSV
    rows = []
    for c in sorted(jacc.keys()):
        idx = np.where(labels == c)[0]
        rows.append({
            "cluster_id": int(c),
            "size": int(idx.size),
            "jaccard_stability": float(jacc[c]),
            "n_runs": int(n_runs),
        })
    df_stab = pd.DataFrame(rows)
    df_stab.to_csv(run_dir / "per_cluster_stability.csv", index=False)
    if verbose:
        print(df_stab.to_string(index=False))

    # Heatmap: reorder rows/cols by cluster, draw consensus matrix
    order = np.argsort(labels, kind="stable")
    M_ord = M[np.ix_(order, order)]
    sorted_labels = labels[order]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M_ord, cmap="magma", aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title(f"Consensus matrix (K={k}, n_runs={n_runs})\n"
                 f"diagonal blocks = stable clusters")
    ax.set_xlabel("sample (reordered by cluster)")
    ax.set_ylabel("sample (reordered by cluster)")

    # Cluster boundary lines
    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=0.4, alpha=0.6)
        ax.axvline(b - 0.5, color="white", lw=0.4, alpha=0.6)

    plt.colorbar(im, ax=ax, fraction=0.046, label="co-clustering frequency")
    fig.tight_layout()
    fig.savefig(run_dir / "consensus_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Also write a JSON summary so MOBA can read it directly
    summary = {
        "n_runs": int(n_runs),
        "k": k,
        "n_samples": int(n),
        "mean_jaccard": float(np.nanmean(list(jacc.values()))),
        "min_jaccard": float(np.nanmin([v for v in jacc.values() if not np.isnan(v)] or [float("nan")])),
        "max_jaccard": float(np.nanmax([v for v in jacc.values() if not np.isnan(v)] or [float("nan")])),
        "per_cluster_jaccard": {str(k_): v for k_, v in jacc.items()},
    }
    (run_dir / "stability_summary.json").write_text(json.dumps(summary, indent=2))

    if verbose:
        print(f"[stability] wrote {run_dir / 'consensus_matrix.npy'}, "
              f"{run_dir / 'per_cluster_stability.csv'}, "
              f"{run_dir / 'consensus_heatmap.png'}, "
              f"{run_dir / 'stability_summary.json'}")

    return df_stab, summary
