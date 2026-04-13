"""
lf_clustering_methods.py

Shared clustering logic for 230_blob_clustering and 231_minus101_clustering.

Covers:
  - KMeans with silhouette-based K selection
  - Hierarchical clustering (HC) with linkage, dendrogram, and distance-cut labels
  - Medoid computation
  - Prototype maps (mean -101 segmentation per cluster)
  - Dendrogram QC plots (plain / condition strip / patient strip)
  - Saving cluster assignments into df_keep_with_clusters
  - Run save helper (s71_save_run)
"""

from __future__ import annotations

import json
import os
import platform
import getpass
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# KMeans
# ============================================================

def s50_run_kmeans(
    Xw: np.ndarray,
    k_range: List[int],
    random_state: int = 42,
    n_init: int = 10,
    verbose: bool = True,
) -> Tuple[np.ndarray, int, Dict[int, float], Dict[int, np.ndarray]]:
    """
    Run KMeans over k_range, select best K by silhouette score.

    Parameters
    ----------
    Xw : (n_samples, n_features) weighted feature matrix
    k_range : list of K values to try
    random_state : random seed
    n_init : number of KMeans initialisations per K

    Returns
    -------
    labels_final : (n_samples,) int32 cluster labels at best K
    best_k : int
    sil_by_k : dict {k: silhouette_score}
    labels_by_k : dict {k: labels array}
    """
    sil_by_k: Dict[int, float] = {}
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in k_range:
        km = KMeans(n_clusters=int(k), random_state=random_state, n_init=n_init)
        lab = km.fit_predict(Xw)
        labels_by_k[int(k)] = lab.astype(np.int32)
        if len(np.unique(lab)) > 1:
            sil_by_k[int(k)] = float(silhouette_score(Xw, lab))
        else:
            sil_by_k[int(k)] = float("nan")
        if verbose:
            print(f"  K={k:3d}  sil={sil_by_k[int(k)]:.4f}")

    best_k = max(sil_by_k, key=lambda k: np.nan_to_num(sil_by_k[k], nan=-1e9))
    labels_final = labels_by_k[best_k]

    if verbose:
        print(f"\n[KMeans] Best K={best_k}  silhouette={sil_by_k[best_k]:.4f}")

    return labels_final, best_k, sil_by_k, labels_by_k


def q50_plot_silhouette_curve(
    sil_by_k: Dict[int, float],
    best_k: int,
    *,
    title: str = "KMeans silhouette vs K",
    save_path: Optional[Path] = None,
):
    """Plot silhouette score vs K and mark the best K."""
    ks = sorted(sil_by_k.keys())
    sils = [sil_by_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(ks, sils, marker="o", lw=1.5)
    ax.axvline(best_k, color="red", ls="--", lw=1.2, label=f"best K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print("[saved]", save_path)
    plt.show()
    plt.close(fig)


# ============================================================
# Hierarchical clustering
# ============================================================

def s51_run_hc(
    Xw: np.ndarray,
    *,
    method: str = "average",
    metric: str = "euclidean",
    precomputed_D: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build hierarchical linkage from Xw or a precomputed distance matrix.

    Parameters
    ----------
    Xw : (n_samples, n_features)
        Feature matrix. Used only when precomputed_D is None.
    method : linkage method ('average', 'complete', 'ward')
        Ward requires Euclidean metric — only valid when precomputed_D is None.
    metric : distance metric when computing from Xw ('euclidean', 'cosine', …)
    precomputed_D : (n_samples, n_samples) optional precomputed distance matrix.
        If provided, Xw is ignored and method must NOT be 'ward'.

    Returns
    -------
    Z : (n-1, 4) linkage matrix
    D : (n_samples, n_samples) distance matrix (symmetric, float32)
    """
    n = Xw.shape[0] if precomputed_D is None else precomputed_D.shape[0]

    if precomputed_D is not None:
        D = np.asarray(precomputed_D, dtype=np.float32)
        if method == "ward":
            raise ValueError("Ward linkage requires Euclidean distances from raw features, "
                             "not a precomputed matrix. Use method='average' or 'complete'.")
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method=method)
    else:
        from scipy.spatial.distance import pdist
        condensed = pdist(Xw, metric=metric)
        D = squareform(condensed).astype(np.float32)
        Z = linkage(condensed, method=method)

    if verbose:
        coph, _ = cophenet(Z, squareform(D, checks=False))
        print(f"[HC] method={method}  metric={metric if precomputed_D is None else 'precomputed'}"
              f"  n={n}  cophenetic_r={coph:.3f}")

    return Z, D


def s52_cut_hc(
    Z: np.ndarray,
    *,
    n_clusters: Optional[int] = None,
    cut_height: Optional[float] = None,
) -> np.ndarray:
    """
    Cut a linkage matrix into flat cluster labels.

    Exactly one of n_clusters or cut_height must be provided.

    Returns
    -------
    labels : (n_samples,) int32 cluster labels (1-based from scipy fcluster)
    """
    if (n_clusters is None) == (cut_height is None):
        raise ValueError("Provide exactly one of n_clusters or cut_height.")

    if n_clusters is not None:
        labels = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    else:
        labels = fcluster(Z, t=float(cut_height), criterion="distance")

    return labels.astype(np.int32)


def s53_compute_medoids(
    D: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, int]:
    """
    For each cluster, find the medoid: sample minimising mean intra-cluster distance.

    Parameters
    ----------
    D : (n, n) pairwise distance matrix
    labels : (n,) cluster labels

    Returns
    -------
    medoid_by_cluster : dict {cluster_id: sample_index}
    """
    medoid_by_cluster: Dict[int, int] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        Dc = D[np.ix_(idx, idx)]
        mean_dist = Dc.mean(axis=1)
        medoid_by_cluster[int(c)] = int(idx[np.argmin(mean_dist)])
    return medoid_by_cluster


# ============================================================
# Prototype maps
# ============================================================

def s54_compute_prototypes(
    ersp_list: List[np.ndarray],
    labels: np.ndarray,
    *,
    thr_pos: float,
    thr_neg: float,
    use_valley_blobs: bool = True,
    valley_params: Optional[Dict] = None,
    hard_thr: float = 0.2,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute mean -101 segmentation prototype and hard prototype per cluster.

    If use_valley_blobs=True, blob masks (from s21_segment_valley_blobs) are used
    to paint the -101 map — consistent with the feature extraction.
    If use_valley_blobs=False, simple threshold is applied directly.

    Parameters
    ----------
    ersp_list : list of (nF, nT) ERSP arrays aligned with labels
    labels : (n,) cluster labels
    thr_pos, thr_neg : thresholds for segmentation
    use_valley_blobs : whether to use valley blob masks (recommended: True)
    valley_params : dict with segmentation params (required if use_valley_blobs=True)
    hard_thr : consensus threshold for hard prototype (default 0.2)

    Returns
    -------
    proto_mean : dict {cluster_id: float array in [-1, 1]}
    proto_hard : dict {cluster_id: int8 array in {-1, 0, +1}}
    """
    def _to_minus101(ersp):
        if use_valley_blobs:
            from functions.lf_blob_metrics import s21_segment_valley_blobs
            vp = valley_params or {}
            blobs = s21_segment_valley_blobs(
                ersp,
                thr_pos=thr_pos,
                thr_neg=thr_neg,
                delta_valley=float(vp.get("delta_valley", 1.5)),
                min_mean_pos=float(vp.get("min_mean_pos", 0.0)),
                max_mean_neg=float(vp.get("max_mean_neg", 0.0)),
                max_blobs=int(vp.get("max_blobs", 6)),
                sign_mode=str(vp.get("sign_mode", "both")),
            )
            seg = np.zeros_like(ersp, dtype=np.int8)
            for b in blobs:
                m = b["mask"]
                s = b.get("sign", "pos")
                if isinstance(s, (int, float, np.integer, np.floating)):
                    seg[m] = 1 if float(s) > 0 else -1
                else:
                    ss = str(s).strip().lower()
                    seg[m] = 1 if ss in {"pos", "+", "positive", "p", "1", "plus"} else -1
            return seg
        else:
            seg = np.zeros_like(ersp, dtype=np.int8)
            seg[ersp >= thr_pos] = 1
            seg[ersp <= thr_neg] = -1
            return seg

    proto_mean: Dict[int, np.ndarray] = {}
    proto_hard: Dict[int, np.ndarray] = {}

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        segs = np.stack([_to_minus101(ersp_list[i]) for i in idx], axis=0)
        mean_map = segs.mean(axis=0).astype(np.float32)
        proto_mean[int(c)] = mean_map
        hard = np.zeros_like(mean_map, dtype=np.int8)
        hard[mean_map >= hard_thr] = 1
        hard[mean_map <= -hard_thr] = -1
        proto_hard[int(c)] = hard

    return proto_mean, proto_hard


# ============================================================
# QC: dendrogram plots
# ============================================================

def q51_plot_dendrogram(
    Z: np.ndarray,
    df_leaf: pd.DataFrame,
    *,
    strip_col: Optional[str] = None,
    title: str = "Dendrogram",
    cut_height: Optional[float] = None,
    save_path: Optional[Path] = None,
    dpi: int = 200,
):
    """
    Plot a dendrogram with an optional categorical colour strip beneath it.

    Parameters
    ----------
    Z : linkage matrix
    df_leaf : DataFrame in leaf order (must be pre-sorted by dendrogram leaf order).
              Must have a 'leaf_label' column for x-axis labels.
    strip_col : column name in df_leaf to colour the strip (e.g. 'condition', 'patient_id').
                If None, only the plain dendrogram is drawn.
    cut_height : if provided, draw a horizontal dashed line at this height.
    save_path : if provided, save the figure here.
    """
    n = len(df_leaf)
    fig_h = min(0.18 * n + (6 if strip_col else 4), 40)

    if strip_col and strip_col in df_leaf.columns:
        fig = plt.figure(figsize=(40, fig_h))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1], hspace=0.05)
        ax_tree = fig.add_subplot(gs[0, 0])
        ax_strip = fig.add_subplot(gs[1, 0])
    else:
        fig, ax_tree = plt.subplots(figsize=(40, fig_h))
        ax_strip = None

    labels = df_leaf["leaf_label"].tolist() if "leaf_label" in df_leaf.columns else None
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=7, ax=ax_tree)
    ax_tree.set_title(title)
    ax_tree.set_ylabel("Distance")

    if cut_height is not None:
        ax_tree.axhline(cut_height, color="black", lw=1.5, ls="--")

    if ax_strip is not None:
        # Build colour map for this column
        vals = df_leaf[strip_col].astype(str).fillna("NA").tolist()
        uniq = sorted(set(vals))
        cmap = plt.get_cmap("tab20", max(3, len(uniq)))
        lut = {u: cmap(i % cmap.N) for i, u in enumerate(uniq)}
        img = np.array([lut[v] for v in vals], dtype=float).reshape(1, n, 4)[:, :, :3]
        ax_strip.imshow(img, aspect="auto")
        ax_strip.set_yticks([])
        ax_strip.set_xticks([])
        ax_strip.set_ylabel(strip_col, rotation=0, labelpad=30, va="center")

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        print("[saved]", save_path)

    plt.show()
    plt.close(fig)


def q52_plot_prototype_grid(
    proto_mean: Dict[int, np.ndarray],
    labels: np.ndarray,
    *,
    title: str = "Cluster prototypes — mean -101 map",
    n_cols: int = 5,
    save_path: Optional[Path] = None,
    dpi: int = 150,
):
    """
    Grid of mean -101 prototype maps per cluster, sorted by cluster size.
    """
    cluster_sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    order = sorted(cluster_sizes, key=lambda c: -cluster_sizes[c])

    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=dpi)
    axes = np.atleast_2d(axes)

    for k, c in enumerate(order):
        r, col = divmod(k, n_cols)
        ax = axes[r, col]
        ax.imshow(proto_mean[c], aspect="auto", origin="lower",
                  cmap="bwr", vmin=-1, vmax=1)
        ax.set_title(f"Cluster {c} (n={cluster_sizes[c]})", fontsize=9)
        ax.set_xlabel("time"); ax.set_ylabel("freq")

    for k in range(len(order), n_rows * n_cols):
        r, col = divmod(k, n_cols)
        axes[r, col].axis("off")

    plt.suptitle(title, y=1.01)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print("[saved]", save_path)

    plt.show()
    plt.close(fig)


# ============================================================
# Build df_keep_with_clusters
# ============================================================

def s55_build_df_with_clusters(
    df_keep: pd.DataFrame,
    labels: np.ndarray,
    *,
    algo_tag: str,
    param_tag: str,
    keep_idx: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Attach cluster labels to df_keep and return df_keep_with_clusters.

    Parameters
    ----------
    df_keep : sample-level metadata DataFrame
    labels : (n_samples,) cluster labels aligned with df_keep rows
    algo_tag : algorithm identifier (e.g. 'kmeans_blob', 'hc_minus101')
    param_tag : parameter string (e.g. 'k23_q0p9')
    keep_idx : optional original sample indices (added as 'sample_idx' if not present)

    Returns
    -------
    df_keep_with_clusters : pd.DataFrame with cluster column added
    """
    df = df_keep.copy()

    # Ensure recommended columns exist
    for col in ["patient_id", "electrode", "condition", "task"]:
        if col not in df.columns:
            df[col] = pd.NA

    if "sample_idx" not in df.columns:
        if keep_idx is not None:
            df["sample_idx"] = np.asarray(keep_idx, dtype=np.int64)
        else:
            df["sample_idx"] = pd.NA

    if "file_path" not in df.columns:
        df["file_path"] = pd.NA

    # Attach labels
    cluster_col = f"cluster_{algo_tag}_{param_tag}"
    lab = np.asarray(labels)
    assert lab.ndim == 1 and len(lab) == len(df), \
        f"labels length {len(lab)} must match df_keep rows {len(df)}"
    df[cluster_col] = lab.astype(np.int32)

    # Sanity: no NaN in cluster column
    if df[cluster_col].isna().any():
        raise ValueError(f"{cluster_col} contains NaN — every row must have a cluster label.")

    return df


# ============================================================
# Save run
# ============================================================

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return str(o)


def _save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _safe_git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode().strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.STDOUT
        ).decode().strip()
        dirty = bool(status)
    except Exception:
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def s71_save_run(
    *,
    run_dir: Path,
    df_keep_with_clusters: pd.DataFrame,
    labels: np.ndarray,
    algo_tag: str,
    param_tag: str,
    # Optional artifacts
    Z: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    sil_by_k: Optional[Dict] = None,
    labels_by_k: Optional[Dict] = None,
    extra_meta: Optional[dict] = None,
    verbose: bool = True,
):
    """
    Save clustering outputs into run_dir.

    Saves:
      - df_keep_with_clusters.parquet
      - labels_<algo_tag>_<param_tag>.npy
      - linkage_Z.npy (optional, HC)
      - distance_D.npy (optional, HC)
      - silhouette_by_k.json (optional, KMeans)
      - run_metadata.json

    Parameters
    ----------
    run_dir : output directory (must already exist, e.g. from s71_save_gated_blob_run)
    df_keep_with_clusters : output of s55_build_df_with_clusters
    labels : (n,) final cluster labels
    algo_tag, param_tag : used to name the labels file
    Z : linkage matrix (HC only)
    D : distance matrix (HC only)
    sil_by_k : silhouette scores by K (KMeans only)
    labels_by_k : all K-level labels (KMeans only)
    extra_meta : additional fields to write into run_metadata.json
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Core
    out_parquet = run_dir / "df_keep_with_clusters.parquet"
    df_keep_with_clusters.to_parquet(out_parquet, index=False)

    out_labels = run_dir / f"labels_{algo_tag}_{param_tag}.npy"
    np.save(out_labels, np.asarray(labels, dtype=np.int32))

    # HC artifacts
    if Z is not None:
        np.save(run_dir / "linkage_Z.npy", Z)
    if D is not None:
        np.save(run_dir / "distance_D.npy", D.astype(np.float32))

    # KMeans artifacts
    if sil_by_k is not None:
        _save_json(run_dir / "silhouette_by_k.json",
                   {str(k): v for k, v in sil_by_k.items()})
    if labels_by_k is not None:
        for k, lab in labels_by_k.items():
            np.save(run_dir / f"labels_k{k}.npy", np.asarray(lab, dtype=np.int32))

    # Metadata
    meta = {
        "algo_tag": algo_tag,
        "param_tag": param_tag,
        "n_samples": int(len(labels)),
        "n_clusters": int(len(np.unique(labels))),
        "cluster_sizes": {
            str(c): int((np.asarray(labels) == c).sum())
            for c in np.unique(labels)
        },
        "user": getpass.getuser(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        **_safe_git_info(),
    }
    if extra_meta:
        meta.update(extra_meta)

    _save_json(run_dir / "run_metadata_clustering.json", meta)

    if verbose:
        n_files = len(list(run_dir.glob("*")))
        print(f"[s71_save_run] Saved to: {run_dir}")
        print(f"  algo={algo_tag}  param={param_tag}  n_clusters={meta['n_clusters']}")
        print(f"  files in run_dir: {n_files}")

    return run_dir