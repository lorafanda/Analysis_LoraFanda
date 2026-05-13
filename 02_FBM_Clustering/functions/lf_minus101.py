"""
lf_minus101.py

-101 feature extraction for 231_minus101_clustering.

The -101 representation is derived from the SAME valley blob objects produced by
s21_segment_valley_blobs (shared with the blob feature path in 230). Each blob
mask is painted as +1 (positive blob) or -1 (negative blob) on a zero map.
The full painted map is then downsampled to 15% of the original size using
skimage.transform.resize with anti-aliasing.

Output per sample: a float32 vector in [-1, 1] of shape (n_freq_ds * n_time_ds,)
where n_freq_ds = round(n_freq * scale), n_time_ds = round(n_time * scale).

Functions
---------
paint_minus101_map
    Paint blob masks onto a zero map -> (n_freq, n_time) int8 {-1, 0, +1}.

downsample_minus101_map
    Resize a painted map to target shape using skimage anti-aliased resize.

s23_build_minus101_feature_matrix
    Build downsampled feature matrix X_101 for all samples.

q60_plot_minus101_overlays
    QC grid: ERSP + painted map + downsampled map per sample.

q61_plot_minus101_prototype
    Mean -101 map for a single cluster (in downsampled space).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize as sk_resize


# ============================================================
# Default downsampling scale
# ============================================================

DEFAULT_SCALE: float = 0.10


# ============================================================
# Paint -101 map from blob objects
# ============================================================

def paint_minus101_map(
    ersp: np.ndarray,
    blobs: List[dict],
    *,
    score_min: Optional[float] = None,
) -> np.ndarray:
    """
    Paint a -101 segmentation map from blob masks.

    Positive blobs -> +1, negative blobs -> -1, rest -> 0.
    Intensity information is discarded — only mask and sign are used.

    Parameters
    ----------
    ersp : (n_freq, n_time) — used only for shape
    blobs : list of blob dicts from s21_segment_valley_blobs
    score_min : optional score gate — blobs below this score are skipped

    Returns
    -------
    seg : (n_freq, n_time) int8 with values in {-1, 0, +1}
    """
    seg = np.zeros_like(np.asarray(ersp), dtype=np.int8)

    for b in blobs:
        if score_min is not None and float(b.get("score", 0.0)) < score_min:
            continue

        mask = b["mask"]
        sign_raw = b.get("sign", 1)

        if isinstance(sign_raw, (int, float, np.integer, np.floating)):
            sign_val = np.int8(1 if float(sign_raw) > 0 else -1)
        else:
            ss = str(sign_raw).strip().lower()
            sign_val = np.int8(1 if ss in {"pos", "+", "positive", "p", "1", "plus"} else -1)

        seg[mask] = sign_val

    return seg


# ============================================================
# Downsample a painted map
# ============================================================

def downsample_minus101_map(
    seg: np.ndarray,
    *,
    scale: float = DEFAULT_SCALE,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Resize a painted -101 map using skimage anti-aliased resize.

    Output values are floats in [-1, 1]. Partial blob coverage of an output
    pixel appears as an intermediate value (e.g. +0.3 means ~30% of the
    contributing area was a positive blob).

    Parameters
    ----------
    seg : (n_freq, n_time) painted map
    scale : fraction of original size (default 0.15 = 15%).
            Ignored if target_shape is provided.
    target_shape : (n_freq_out, n_time_out) explicit output shape.

    Returns
    -------
    seg_ds : (n_freq_out, n_time_out) float32
    """
    seg = np.asarray(seg, dtype=np.float32)
    n_freq, n_time = seg.shape

    if target_shape is None:
        target_shape = (
            max(1, int(round(n_freq * scale))),
            max(1, int(round(n_time * scale))),
        )

    seg_ds = sk_resize(
        seg,
        target_shape,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    return seg_ds


# ============================================================
# Core: build feature matrix
# ============================================================

def s23_build_minus101_feature_matrix(
    ersp_list: List[np.ndarray],
    blobs_per_sample: List[List[dict]],
    *,
    scale: float = DEFAULT_SCALE,
    target_shape: Optional[Tuple[int, int]] = None,
    score_min: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Build the downsampled -101 feature matrix X_101.

    For each sample:
      1. Paint blob masks onto a zero map -> (n_freq, n_time) {-1, 0, +1}
      2. Resize to target_shape using skimage anti-aliased resize
      3. Flatten to 1D

    All samples share the same target_shape, inferred from the first ERSP
    or provided explicitly.

    Parameters
    ----------
    ersp_list : list of (n_freq, n_time) ERSP arrays
        Used for shape only — intensity not used.
    blobs_per_sample : list of list of blob dicts
        Aligned with ersp_list. Shared output from s22_build_blob_feature_matrix.
    scale : downsample fraction (default 0.15)
    target_shape : explicit (n_freq_out, n_time_out). If None, inferred from scale.
    score_min : optional per-blob score gate

    Returns
    -------
    X_101 : (n_samples, n_freq_out * n_time_out) float32
    ds_shape : (n_freq_out, n_time_out)
    """
    n_samples = len(ersp_list)
    n_freq_orig, n_time_orig = np.asarray(ersp_list[0]).shape

    if target_shape is None:
        target_shape = (
            max(1, int(round(n_freq_orig * scale))),
            max(1, int(round(n_time_orig * scale))),
        )

    feat_dim = target_shape[0] * target_shape[1]
    X_101 = np.zeros((n_samples, feat_dim), dtype=np.float32)

    for i, (ersp, blobs) in enumerate(zip(ersp_list, blobs_per_sample)):
        seg = paint_minus101_map(ersp, blobs, score_min=score_min)
        seg_ds = downsample_minus101_map(seg, target_shape=target_shape)
        X_101[i, :] = seg_ds.ravel()

    if verbose:
        n_nonzero = int((np.abs(X_101) > 0.01).any(axis=1).sum())
        print(f"[s23_build_minus101_feature_matrix]")
        print(f"  Original map : {n_freq_orig}×{n_time_orig} = {n_freq_orig*n_time_orig} bins")
        print(f"  Downsampled  : {target_shape[0]}×{target_shape[1]} = {feat_dim} features  "
              f"(scale={scale:.0%})")
        print(f"  X_101.shape  : {X_101.shape}")
        print(f"  Samples with any blob activity: {n_nonzero}/{n_samples}")

    return X_101, target_shape


# ============================================================
# Per-sample / per-cluster -101 PNG render
# ============================================================

def render_minus101(
    ax,
    ersp,
    blobs,
    *,
    score_min=None,
    show_axes: bool = False,
):
    """
    Render a -101 painted map onto an existing matplotlib axis.

    Positive-blob pixels → red, negative-blob pixels → blue, rest → white.
    Used both by per-sample PNG export and by cluster-centroid chip rendering.
    """
    seg = paint_minus101_map(ersp, blobs, score_min=score_min)
    n_freq, n_time = seg.shape
    ax.imshow(
        seg,
        aspect="auto", origin="lower",
        cmap="bwr", vmin=-1, vmax=1,
        interpolation="nearest",
    )
    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(-0.5, n_time - 0.5)
    ax.set_ylim(-0.5, n_freq - 0.5)


def save_sample_minus101_png(
    ersp,
    blobs,
    path,
    *,
    figsize=(3, 1.5),
    dpi: int = 100,
    score_min=None,
):
    """Write a single-sample painted -101 PNG (no axes, no title)."""
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    render_minus101(ax, ersp, blobs, score_min=score_min)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# QC visualisation
# ============================================================

def q60_plot_minus101_overlays(
    indices: List[int],
    ersp_list: List[np.ndarray],
    blobs_per_sample: List[List[dict]],
    df_meta,
    *,
    scale: float = DEFAULT_SCALE,
    score_min: Optional[float] = None,
    n_cols: int = 6,
    title_prefix: str = "-101 maps",
    fmax_hz: float = 500.0,
    save_path: Optional[Path] = None,
    dpi: int = 150,
):
    """
    QC grid: three panels per sample — ERSP | painted -101 | downsampled -101.

    Parameters
    ----------
    indices : sample indices to plot
    ersp_list, blobs_per_sample : aligned lists from the pipeline
    df_meta : DataFrame with patient_id, electrode, condition columns
    scale : downsample scale for the third panel
    score_min : optional score gate for blobs
    n_cols : number of samples per row
    """
    if not indices:
        print("No indices provided.")
        return

    n = len(indices)
    n_rows = int(np.ceil(n / n_cols))
    extent = (0.0, 100.0, 0.0, float(fmax_hz))

    fig, axes = plt.subplots(
        n_rows * 3, n_cols,
        figsize=(n_cols * 2.0, n_rows * 4.5),
        squeeze=False,
    )

    for pos, idx in enumerate(indices):
        row = pos // n_cols
        col = pos % n_cols

        ax_ersp = axes[row * 3,     col]
        ax_seg  = axes[row * 3 + 1, col]
        ax_ds   = axes[row * 3 + 2, col]

        ersp = np.asarray(ersp_list[idx])
        blobs = blobs_per_sample[idx]
        seg = paint_minus101_map(ersp, blobs, score_min=score_min)
        seg_ds = downsample_minus101_map(seg, scale=scale)

        meta = df_meta.iloc[idx]
        label = (f"{meta.get('patient_id', '')} "
                 f"{meta.get('electrode', '')} "
                 f"{meta.get('condition', '')}")

        ax_ersp.imshow(ersp, aspect="auto", origin="lower",
                       cmap="bwr", vmin=-6, vmax=6,
                       interpolation="nearest", extent=extent)
        ax_ersp.set_title(label, fontsize=6)
        ax_ersp.axis("off")

        ax_seg.imshow(seg, aspect="auto", origin="lower",
                      cmap="bwr", vmin=-1, vmax=1,
                      interpolation="nearest", extent=extent)
        ax_seg.set_title("-101 (full)", fontsize=6)
        ax_seg.axis("off")

        ax_ds.imshow(seg_ds, aspect="auto", origin="lower",
                     cmap="bwr", vmin=-1, vmax=1,
                     interpolation="nearest", extent=extent)
        ax_ds.set_title(f"-101 ({scale:.0%})", fontsize=6)
        ax_ds.axis("off")

    for pos in range(len(indices), n_rows * n_cols):
        row = pos // n_cols
        col = pos % n_cols
        for sub in range(3):
            axes[row * 3 + sub, col].axis("off")

    fig.suptitle(f"{title_prefix}  (score_min={score_min}  scale={scale:.0%})", fontsize=9)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print("[saved]", save_path)

    plt.show()
    plt.close(fig)


def q61_plot_minus101_prototype(
    proto_mean: np.ndarray,
    cluster_id: int,
    cluster_size: int,
    *,
    fmax_hz: float = 500.0,
    save_path: Optional[Path] = None,
    dpi: int = 150,
):
    """
    Plot the mean -101 map for a single cluster (in downsampled space).

    Parameters
    ----------
    proto_mean : (n_freq_ds, n_time_ds) float array in [-1, 1]
    cluster_id : cluster label (for title)
    cluster_size : number of samples in cluster (for title)
    """
    extent = (0.0, 100.0, 0.0, float(fmax_hz))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(proto_mean, aspect="auto", origin="lower",
              cmap="bwr", vmin=-1, vmax=1,
              interpolation="nearest", extent=extent)
    ax.set_title(f"Cluster {cluster_id}  (n={cluster_size})\nmean -101 map (downsampled)")
    ax.set_xlabel("% time")
    ax.set_ylabel("Hz")
    ax.axvline(50.0, color="lightgray", lw=1.0)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print("[saved]", save_path)

    plt.show()
    plt.close(fig)