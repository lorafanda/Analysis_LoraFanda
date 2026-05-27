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
    compute_consensus_matrix(X, k, n_runs=50, random_state=0, subsample_frac=1.0)
        -> (n_samples, n_samples) float32 in [0, 1]; entry (i,j) = fraction
           of n_runs in which samples i and j were in the same cluster
           (conditional on both being sampled in that run when subsampling).

    per_cluster_jaccard(labels, consensus_matrix)
        -> dict {cluster_id: jaccard_stability_score in [0,1]}.
           Higher = cluster is reproducible across resamplings.

    reassign_by_consensus_similarity(labels, consensus_matrix,
                                     invalid_ids, valid_ids, df_meta=None)
        -> new labels array. For each cluster id in `invalid_ids`, computes
           the mean co-clustering frequency from its members to every cluster
           in `valid_ids` and reassigns the whole invalid block to the
           highest-similarity valid cluster.

    cluster_center_coverage(labels, df_meta, *, bern_prefix='EL',
                            gva_prefix='PAT', patient_col='patient_id')
        -> dict keyed by cluster_id with {n_bern, n_gva, n_micro, n_patients,
           has_bern, has_gva, is_valid}. is_valid means cluster has at least
           one BERN (EL*) AND one GVA (PAT*) patient — used by the 213
           ranking notebook to decide which clusters to dissolve.

    save_consensus_artifacts(run_dir, X, labels, *, n_runs=50, random_state=0)
        -> writes:
           consensus_matrix.npy        (n×n float32; gitignored)
           per_cluster_stability.csv   (cluster_id, size, jaccard, sil)
           consensus_heatmap.png       (samples reordered by cluster)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    subsample_frac: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run KMeans n_runs times with different random_state offsets, accumulate
    the per-pair co-clustering frequency.

    Parameters
    ----------
    X : (n_samples, n_features)
    k : number of clusters per KMeans fit
    n_runs : number of resampling rounds (Monti et al. 2003 use 25-50; we
        default to 50)
    random_state : seed offset for KMeans + the per-run subsample drawer
    subsample_frac : in (0, 1]. If <1, each run draws floor(n*frac) samples
        without replacement and only those rows participate in that run's
        co-cluster updates. The normalization is then per-pair-conditional
        — entry (i,j) = (#runs i and j co-clustered) / (#runs both sampled),
        with 0 used when the pair never co-appeared. Default 1.0 reproduces
        the previous all-sample behavior.
    n_init : passed through to sklearn KMeans

    Returns
    -------
    consensus : (n_samples, n_samples) float32 in [0, 1]

    For large n_samples (>~3000) this allocates two n×n float32 matrices
    (consensus + indicator) which can be hundreds of MB. KMeans itself
    is the slow part — at K=20 and 1500 samples, ~50 runs takes a couple
    of minutes.
    """
    if not (0.0 < subsample_frac <= 1.0):
        raise ValueError(f"subsample_frac must be in (0, 1], got {subsample_frac}")

    n = X.shape[0]
    consensus = np.zeros((n, n), dtype=np.float32)

    # When subsampling, we need to track which pairs (i, j) were both sampled
    # in each run so the final normalization is conditional on co-appearance,
    # not on raw n_runs (which would underestimate co-cluster frequency for
    # pairs that happened to be picked together fewer times).
    use_subsample = subsample_frac < 1.0
    if use_subsample:
        indicator = np.zeros((n, n), dtype=np.float32)
        n_sub = max(2, int(round(n * float(subsample_frac))))
        # Master RNG draws per-run subsample indices reproducibly
        sub_rng = np.random.default_rng(int(random_state))
    else:
        indicator = None
        n_sub = n

    for r in range(n_runs):
        seed = int(random_state + r)
        if use_subsample:
            sub_idx = sub_rng.choice(n, size=n_sub, replace=False)
            X_run = X[sub_idx]
            # Record that every pair (i, j) with i, j in sub_idx was seen
            # this run (including diagonal)
            indicator[np.ix_(sub_idx, sub_idx)] += 1.0
        else:
            sub_idx = None
            X_run = X

        km = KMeans(n_clusters=int(k), random_state=seed, n_init=n_init).fit(X_run)
        lbl = km.labels_
        # Indicator: same cluster -> add 1 to (i,j) for all pairs in same group.
        # Vectorized: for each cluster c, get its sample indices, set their
        # outer product block to +=1. Faster than pairwise comparison.
        for c in np.unique(lbl):
            idx_in_run = np.where(lbl == c)[0]
            if idx_in_run.size > 1:
                if use_subsample:
                    # Map subsample-local indices back to original indices
                    orig_idx = sub_idx[idx_in_run]
                    consensus[np.ix_(orig_idx, orig_idx)] += 1.0
                else:
                    # np.ix_ creates a row+col selector for a block update
                    consensus[np.ix_(idx_in_run, idx_in_run)] += 1.0
        if verbose and ((r + 1) % 10 == 0):
            print(f"  consensus run {r+1}/{n_runs}")

    if use_subsample:
        # Per-pair conditional normalization: entry / (#runs both sampled).
        # Where a pair never co-appeared we leave the entry at 0 (it carries
        # no information rather than being a true zero).
        with np.errstate(invalid="ignore", divide="ignore"):
            consensus = np.where(indicator > 0,
                                  consensus / np.maximum(indicator, 1.0),
                                  0.0).astype(np.float32)
    else:
        # Every pair was seen in every run -> simple division
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
    subsample_frac: float = 1.0,
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

    `subsample_frac` is passed through to compute_consensus_matrix (default
    1.0 = use every sample every run, identical to the pre-Monti behavior).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels).astype(np.int32)
    n = X.shape[0]
    assert labels.shape[0] == n, f"labels ({labels.shape[0]}) vs X ({n}) mismatch"

    k = int(len(np.unique(labels)))
    if verbose:
        print(f"[stability] consensus_matrix at K={k}, n_runs={n_runs}, "
              f"subsample_frac={subsample_frac}, n={n}")

    M = compute_consensus_matrix(X, k, n_runs=n_runs, random_state=random_state,
                                  subsample_frac=subsample_frac, verbose=verbose)
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
        "subsample_frac": float(subsample_frac),
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


# ============================================================
# Center-coverage validity + consensus-driven reassignment
# (used by 213_cluster_ranking.ipynb to dissolve clusters that
#  don't bridge BERN ↔ GVA cohorts)
# ============================================================
def _patient_cohort_prefix(pid: str, *, bern_prefix: str, gva_prefix: str) -> str:
    """Map a patient_id to {'BERN', 'GVA', 'MICRO', 'OTHER'}."""
    s = str(pid).strip().upper()
    if s.startswith(bern_prefix.upper()):
        return "BERN"
    if s.startswith(gva_prefix.upper()):
        return "GVA"
    if s.startswith("MICRO") or s.startswith("G-") or s.startswith("B-"):
        return "MICRO"
    return "OTHER"


def cluster_center_coverage(
    labels: np.ndarray,
    df_meta: pd.DataFrame,
    *,
    bern_prefix: str = "EL",
    gva_prefix: str = "PAT",
    patient_col: str = "patient_id",
) -> Dict[int, Dict[str, Any]]:
    """
    For every cluster in `labels`, count how many distinct patients fall into
    the BERN cohort (patient_id starts with `bern_prefix`, e.g. EL*) versus
    the GVA cohort (`gva_prefix`, e.g. PAT*) versus MICRO (G-* / B-* / MICRO*).

    A cluster is considered "valid" iff it contains at least one BERN AND
    at least one GVA patient — i.e. the cluster bridges the two recording
    centers and isn't a single-site artifact. The 213 ranking notebook uses
    this to decide which clusters to dissolve and reassign.

    Returns
    -------
    Dict[int cluster_id, {
        "n_bern":      int   # unique BERN patients
        "n_gva":       int   # unique GVA patients
        "n_micro":     int   # unique MICRO patients
        "n_patients":  int   # total unique patients across all cohorts
        "has_bern":    bool
        "has_gva":     bool
        "is_valid":    bool  # has_bern AND has_gva
    }]
    """
    labels = np.asarray(labels)
    if len(df_meta) != len(labels):
        raise ValueError(
            f"df_meta ({len(df_meta)} rows) vs labels ({len(labels)}) mismatch"
        )
    if patient_col not in df_meta.columns:
        raise KeyError(f"df_meta missing '{patient_col}' column")

    pids = df_meta[patient_col].astype(str).to_numpy()

    out: Dict[int, Dict[str, Any]] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        # Map patient -> cohort (one cohort per patient — patient_ids don't
        # change cohort, so set + classify once)
        pids_in_cluster = set(pids[idx].tolist())
        cohort_by_pid = {
            p: _patient_cohort_prefix(p, bern_prefix=bern_prefix, gva_prefix=gva_prefix)
            for p in pids_in_cluster
        }
        n_bern  = sum(1 for v in cohort_by_pid.values() if v == "BERN")
        n_gva   = sum(1 for v in cohort_by_pid.values() if v == "GVA")
        n_micro = sum(1 for v in cohort_by_pid.values() if v == "MICRO")
        out[int(c)] = {
            "n_bern":     int(n_bern),
            "n_gva":      int(n_gva),
            "n_micro":    int(n_micro),
            "n_patients": int(len(pids_in_cluster)),
            "has_bern":   bool(n_bern > 0),
            "has_gva":    bool(n_gva > 0),
            "is_valid":   bool(n_bern > 0 and n_gva > 0),
        }
    return out


def reassign_by_consensus_similarity(
    labels: np.ndarray,
    consensus_matrix: np.ndarray,
    invalid_ids: Iterable[int],
    valid_ids: Iterable[int],
    df_meta: Optional[pd.DataFrame] = None,  # accepted for API stability; unused
) -> np.ndarray:
    """
    Reassign every member of each cluster in `invalid_ids` to the cluster
    in `valid_ids` with the highest mean consensus similarity to the
    invalid cluster as a whole.

    Similarity(I, V) = mean of consensus_matrix[i, j] over all (i ∈ I, j ∈ V).
    That's the average pairwise co-clustering frequency across the two
    cluster blocks. All members of an invalid cluster get reassigned to
    the SAME target valid cluster (the one with max mean similarity); we
    don't split an invalid cluster across multiple targets.

    Does NOT mutate `labels`. Returns a fresh array of the same shape.

    Parameters
    ----------
    labels : (n_samples,) int array. Current cluster assignment.
    consensus_matrix : (n_samples, n_samples) float in [0,1].
    invalid_ids : iterable of cluster ids to dissolve.
    valid_ids   : iterable of cluster ids to consider as merge targets.
    df_meta : accepted for forward-compat (e.g. if we ever want to bias
        reassignment by cohort identity); currently ignored.

    Returns
    -------
    new_labels : (n_samples,) int array. Members of invalid clusters now
        carry the id of their best-matching valid cluster. If `valid_ids`
        is empty, returns a copy of `labels` unchanged.
    """
    labels = np.asarray(labels)
    new_labels = labels.copy()

    invalid_list = [int(c) for c in invalid_ids]
    valid_list   = [int(c) for c in valid_ids]
    if not invalid_list or not valid_list:
        return new_labels

    # Pre-compute index sets for valid clusters once (constant across the loop)
    valid_idx_by_cluster = {
        int(c): np.where(labels == c)[0] for c in valid_list
    }

    for invalid_c in invalid_list:
        invalid_idx = np.where(labels == invalid_c)[0]
        if invalid_idx.size == 0:
            continue

        best_target = None
        best_sim = -np.inf
        for valid_c, valid_idx in valid_idx_by_cluster.items():
            if valid_idx.size == 0:
                continue
            block = consensus_matrix[np.ix_(invalid_idx, valid_idx)]
            sim = float(block.mean()) if block.size else -np.inf
            if sim > best_sim:
                best_sim = sim
                best_target = int(valid_c)

        if best_target is not None:
            new_labels[invalid_idx] = best_target

    return new_labels
