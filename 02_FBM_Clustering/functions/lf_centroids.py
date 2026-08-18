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
# Bigger than the old (2.4, 1.7): these are shown large in the run report and the
# axes need room. dpi 220 rather than 90 for the same reason.
FIGSIZE_DEFAULT = (3.4, 2.1)



def shared_hg_ylim(X: np.ndarray, labels: np.ndarray, *, pad: float = 1.08) -> tuple:
    """One symmetric y range for every cluster of a run, fitted to the data.

    The fixed +/-6.5 default was set for the per-condition 'hg' features. On the
    concatenated set the centroids only reach 2.1 (cnmf) to 3.1 (k-means), so half
    the axis was empty and every centroid looked flat.

    Fitted PER RUN and shared across its clusters, so clusters stay comparable with
    each other; the limit is printed on the axis, so comparison ACROSS runs is done
    by reading the number rather than by assuming a constant.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    hi = 0.0
    for k in np.unique(labels):
        M = X[labels == k]
        if not len(M):
            continue
        m = M.mean(axis=0)
        sd = M.std(axis=0, ddof=1) if len(M) > 1 else 0.0
        hi = max(hi, float(np.abs(m + sd).max()), float(np.abs(m - sd).max()))
    if not np.isfinite(hi) or hi <= 0:
        return HG_YLIM_DEFAULT
    hi *= pad
    step = 0.5 if hi < 5 else 1.0
    hi = float(np.ceil(hi / step) * step)
    return (-hi, hi)

# ============================================================
# Rendering primitives
# ============================================================
def render_hg_centroid(ax, cluster_samples: np.ndarray, *,
                       ylim: Tuple[float, float] = HG_YLIM_DEFAULT,
                       line_color: str = "black",
                       sem_color: str = "#4a6fa5",
                       show_axes: bool = False,
                       n_blocks: int = 1,
                       cond_names=("audio", "picture", "reading")):
    """
    Draw a high-gamma cluster centroid as mean line + SEM shaded band.

    Parameters
    ----------
    cluster_samples : (n_samples, n_time) — the HG time series of every
        sample assigned to this cluster. Mean and SEM are computed across
        the sample axis (axis 0).

    The band is the STANDARD DEVIATION across electrodes, mean ± SD — how much the
    electrodes in this cluster differ from each other. Not the SEM: with 139-297
    electrodes per cluster the SEM is a hairline and says only that the mean is well
    estimated, which is never in doubt here. SD answers the question the plot is
    actually for — is this cluster a tight family or a loose one.

    For a singleton cluster (n==1) the SD is zero, so only the line shows.
    """
    M = np.asarray(cluster_samples, dtype=float)
    if M.ndim == 1:
        M = M[None, :]                    # tolerate a pre-averaged 1D vector
    n, T = M.shape
    x = np.arange(T)
    mean = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1) if n > 1 else np.zeros(T)

    ax.axhline(0, color="#999", lw=0.5, alpha=0.8, zorder=1)

    # GO CUE, one per condition. Each block is one condition warped to 0-100% of its
    # trial, and 50% IS the GO cue (cfg.proportions = (0.0, 0.5, 0.5)), so the marker
    # belongs at the midpoint of every block rather than once in the middle of the row.
    per = T / max(n_blocks, 1)
    for b in range(max(n_blocks, 1)):
        ax.axvline((b + 0.5) * per, color="#9aa3ab", lw=0.9, ls=(0, (4, 3)), zorder=2)

    # +/-1 SD across electrodes. For a singleton cluster this is 0 and only the mean
    # line shows - a statement about n, not a rendering failure.
    ax.fill_between(x, mean - sd, mean + sd,
                    color=sem_color, alpha=0.22, lw=0, zorder=3)
    ax.plot(x, mean - sd, color=sem_color, lw=0.45, alpha=0.7, zorder=4)
    ax.plot(x, mean + sd, color=sem_color, lw=0.45, alpha=0.7, zorder=4)
    ax.plot(x, mean, color=line_color, lw=1.3, zorder=5)

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(0, max(1, T - 1))

    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return

    # Minimal axes, and the y range is the SHARED one (HG_YLIM_DEFAULT) so a centroid
    # can be compared against any other cluster, K cut or run by eye.
    lo, hi = ax.get_ylim()
    ax.set_yticks([lo, 0, hi])
    ax.set_yticklabels([f"{lo:g}", "0", f"{hi:g}"], fontsize=6.5)
    ax.set_ylabel("HG (dB)", fontsize=7, labelpad=1)
    ax.set_xticks([(b + 0.5) * per for b in range(max(n_blocks, 1))])
    ax.set_xticklabels(list(cond_names)[:max(n_blocks, 1)] if n_blocks > 1 else ["trial"],
                       fontsize=6.5)
    ax.tick_params(length=2, width=0.6, pad=1.5, colors="#68727d")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#c8cfd6")
    # the label sits ON the dashed line, so say once what the line is
    ax.text(0.5, -0.30, "dashed = GO cue (50% of each condition)  ·  band = ±1 SD across electrodes",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.8, color="#9aa3ab")


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
def _labels_for(run_dir, method):
    """The run's own cluster column, for fitting one y range across its clusters."""
    import pandas as _pd
    lab = _pd.read_csv(Path(run_dir) / "labels.csv")
    col = next((c for c in lab.columns
                if c.startswith("cluster_") and not c.endswith("_ranked")), None)
    return lab[col].to_numpy() if col else np.zeros(len(lab), dtype=int)


def _is_line_feature_set(feature_set: str) -> bool:
    """Feature sets whose centroid is a 1-D time course (line + SEM) rather than a
    2-D ERSP heatmap: the per-condition HG track and its concatenated counterpart."""
    return feature_set == "hg" or feature_set == "concat_hg"


def _n_condition_blocks(feature_set: str) -> int:
    """Concatenated feature sets stitch 3 conditions in time; draw the seams so the
    [audio | picture | reading] structure is readable on the centroid chip."""
    return 3 if str(feature_set).startswith("concat_") else 1


def _draw_block_seams(ax, n_blocks: int, n_x: int, *, heatmap: bool):
    if n_blocks <= 1:
        return
    for b in range(1, n_blocks):
        x = b * (n_x / n_blocks)
        ax.axvline(x - (0.5 if heatmap else 0), color="k", lw=1.0, zorder=5)


def _write_one(out_path: Path, X_cluster: np.ndarray, feature_set: str, *,
               centroid_shape, hg_ylim, vlim, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    nb = _n_condition_blocks(feature_set)
    if _is_line_feature_set(feature_set):
        render_hg_centroid(ax, X_cluster, ylim=hg_ylim, show_axes=True, n_blocks=nb)
        _draw_block_seams(ax, nb, np.asarray(X_cluster).shape[1], heatmap=False)
    else:
        render_heatmap_centroid(ax, np.asarray(X_cluster, float).mean(axis=0),
                                centroid_shape, vlim=vlim)
        if centroid_shape is not None:
            _draw_block_seams(ax, nb, centroid_shape[1], heatmap=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
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
    hg_ylim="auto",
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

    if isinstance(hg_ylim, str) and hg_ylim == "auto":
        hg_ylim = (shared_hg_ylim(X, _labels_for(run_dir, method))
                   if _is_line_feature_set(feature_set) else HG_YLIM_DEFAULT)
        if verbose:
            print(f"  [{feature_set}] shared y range fitted to the run: "
                  f"{hg_ylim[0]:g} to {hg_ylim[1]:g} dB")
    if vlim is None and not _is_line_feature_set(feature_set):
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
