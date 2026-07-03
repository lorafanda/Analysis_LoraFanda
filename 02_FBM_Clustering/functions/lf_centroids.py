"""
lf_centroids.py — single source of truth for per-cluster centroid PNGs.

Every clustering notebook (210 raw, 212 rawds, 231 minus101, 232 hg, ...)
used to roll its own centroid-rendering + path logic, which drifted: some
wrote only the flat best-K centroid, so the MOBA K-slider would fall back to
the best-K thumbnail at EVERY slider position (leakage across K). This module
centralises it:

  * Per-K centroids are written to  cluster_centroids/k_{K}/cluster_NN.png
    for EVERY K found in cluster_labels_by_k.csv, so the MOBA slider updates
    with no leakage — each K/cluster thumbnail is the true mean of that exact
    (K, cluster) membership.
  * The run's default (best-K) labels are also written to the flat path
    cluster_centroids/cluster_NN.png for backward compatibility.

Rendering, switched on feature_set:
  * 'hg'  → 1D high-gamma envelope. BLACK mean line + SEM shaded band
            (mean ± standard error of the mean across the cluster's samples).
            No directional red/blue fill.
  * else  → 2D heatmap, reshaped via `centroid_shape`, cmap='bwr', symmetric
            vlim. Matches the ERSP_clean convention.

Public API
----------
save_per_cluster_centroids(run_dir, X, feature_set, method, ...) -> dict
    Writes all centroid PNGs for a run. Returns {'k_<K>': n, 'flat': n, ...}.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Fixed y-limits for HG line plots — matches cfg.hg_vmin/vmax (±6.5 dB) so
# every HG centroid shares one amplitude scale across clusters, K cuts, runs.
HG_YLIM_DEFAULT = (-6.5, 6.5)
FIGSIZE_DEFAULT = (2.4, 1.7)


# ============================================================
# Rendering primitives
# ============================================================
def render_hg_centroid(ax, cluster_samples: np.ndarray, *,
                       ylim: Tuple[float, float] = HG_YLIM_DEFAULT,
                       line_color: str = "black",
                       sem_color: str = "#888888",
                       show_axes: bool = False):
    """
    Draw a high-gamma cluster centroid as mean line + SEM shaded band.

    Parameters
    ----------
    cluster_samples : (n_samples, n_time) — the HG time series of every
        sample assigned to this cluster. Mean and SEM are computed across
        the sample axis (axis 0).

    The SEM band = mean ± std / sqrt(n). For a singleton cluster (n==1) the
    SEM is zero, so only the line shows.
    """
    M = np.asarray(cluster_samples, dtype=float)
    if M.ndim == 1:
        M = M[None, :]                    # tolerate a pre-averaged 1D vector
    n, T = M.shape
    x = np.arange(T)
    mean = M.mean(axis=0)
    sem = M.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(T)

    ax.axhline(0, color="#999", lw=0.35, alpha=0.7, zorder=1)
    ax.fill_between(x, mean - sem, mean + sem,
                    color=sem_color, alpha=0.35, lw=0, zorder=2)
    ax.plot(x, mean, color=line_color, lw=1.1, zorder=3)

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(0, max(1, T - 1))
    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def render_heatmap_centroid(ax, mean_vec: np.ndarray, centroid_shape, *,
                            vlim: float = 5.0, cmap: str = "bwr"):
    """Draw a 2D ERSP centroid heatmap. Falls back to a 1×N strip if the
    flat vector doesn't match centroid_shape."""
    v = np.asarray(mean_vec, dtype=float).ravel()
    if centroid_shape is not None and centroid_shape[0] * centroid_shape[1] == v.size:
        img = v.reshape(centroid_shape)
    else:
        img = v.reshape(1, -1)
    ax.imshow(img, aspect="auto", origin="lower", cmap=cmap,
              vmin=-abs(vlim), vmax=abs(vlim), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ============================================================
# One-cluster writer
# ============================================================
def _write_one(out_path: Path, X_cluster: np.ndarray, feature_set: str, *,
               centroid_shape, hg_ylim, vlim, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    if feature_set == "hg":
        render_hg_centroid(ax, X_cluster, ylim=hg_ylim)
    else:
        render_heatmap_centroid(ax, np.asarray(X_cluster, float).mean(axis=0),
                                centroid_shape, vlim=vlim)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _write_group(out_dir: Path, X: np.ndarray, labels: np.ndarray, feature_set: str, *,
                 centroid_shape, hg_ylim, vlim, figsize) -> int:
    labels = np.asarray(labels)
    uniq = sorted(int(c) for c in np.unique(labels))
    for c in uniq:
        idx = np.where(labels == c)[0]
        _write_one(out_dir / f"cluster_{c:02d}.png", X[idx], feature_set,
                   centroid_shape=centroid_shape, hg_ylim=hg_ylim, vlim=vlim, figsize=figsize)
    return len(uniq)


# ============================================================
# Public: full per-run writer (per-K + flat)
# ============================================================
def save_per_cluster_centroids(
    run_dir,
    X: np.ndarray,
    feature_set: str,
    method: str,
    *,
    centroid_shape: Optional[Tuple[int, int]] = None,
    hg_ylim: Tuple[float, float] = HG_YLIM_DEFAULT,
    vlim: Optional[float] = None,
    figsize: Tuple[float, float] = FIGSIZE_DEFAULT,
    write_per_k: bool = True,
    write_flat: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Write all per-cluster centroid PNGs for one clustering run.

    Parameters
    ----------
    run_dir : the run directory (contains labels.csv and, for K-sweep runs,
        cluster_labels_by_k.csv).
    X : (n_samples, n_features) feature matrix the clustering was fit on.
        MUST be row-aligned with labels.csv / cluster_labels_by_k.csv.
        For 'hg' this is the (n_samples, n_time) HG matrix; centroid = mean
        line + SEM band. For heatmap feature sets it's the flat 2D-reshapeable
        vector per sample.
    feature_set : 'hg' -> line+SEM; anything else -> heatmap.
    method : e.g. 'kmeans' / 'hierarchical' (only used to locate the default
        cluster column in labels.csv).
    centroid_shape : (n_freq, n_time) reshape for heatmap feature sets.
    vlim : symmetric colour limit for heatmaps (auto = 99th pct of |X|).

    Returns
    -------
    dict mapping written groups -> n clusters, e.g. {'k_6': 6, ..., 'flat': 7}.
    """
    run_dir = Path(run_dir)
    root = run_dir / "cluster_centroids"
    written: Dict[str, int] = {}

    if vlim is None and feature_set != "hg":
        vlim = float(np.percentile(np.abs(X), 99)) or 1.0

    # ── Per-K centroids from cluster_labels_by_k.csv ──
    by_k_path = run_dir / "cluster_labels_by_k.csv"
    if write_per_k and by_k_path.exists():
        by_k = pd.read_csv(by_k_path)
        k_cols = [c for c in by_k.columns if c.startswith("k_")]
        for kc in k_cols:
            labels_k = by_k[kc].to_numpy()
            if len(labels_k) != X.shape[0]:
                if verbose:
                    print(f"  [skip] {run_dir.name} {kc}: labels ({len(labels_k)}) "
                          f"vs X ({X.shape[0]}) mismatch")
                continue
            K = kc.split("_", 1)[1]
            n = _write_group(root / f"k_{K}", X, labels_k, feature_set,
                             centroid_shape=centroid_shape, hg_ylim=hg_ylim,
                             vlim=vlim, figsize=figsize)
            written[kc] = n

    # ── Flat (best-K) centroids from labels.csv default column ──
    if write_flat:
        labels_path = run_dir / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            col = f"cluster_{method}_{feature_set}"
            if col not in df.columns:
                cands = [c for c in df.columns if c.startswith("cluster_")]
                col = cands[0] if cands else None
            if col is not None:
                labels = df[col].to_numpy()
                if len(labels) == X.shape[0]:
                    n = _write_group(root, X, labels, feature_set,
                                     centroid_shape=centroid_shape, hg_ylim=hg_ylim,
                                     vlim=vlim, figsize=figsize)
                    written["flat"] = n
                elif verbose:
                    print(f"  [skip] {run_dir.name} flat: labels ({len(labels)}) "
                          f"vs X ({X.shape[0]}) mismatch")

    if verbose:
        summary = ", ".join(f"{k}={v}" for k, v in written.items())
        print(f"  [{method}/{feature_set}] {run_dir.name}: {summary or 'nothing written'}")
    return written
