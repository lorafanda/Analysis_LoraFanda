# blob_clustering.py

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from functions.lf_blob_metrics import (
    s22_build_blob_feature_matrix,
    s21_segment_valley_blobs,
    q34_plot_valley_blob_overlays,
)


# ------------------------------------------------------------
# 1. Build valley-blob feature matrix
# ------------------------------------------------------------

def s20_build_valley_features(
    ersp_list: List[np.ndarray],
    valley_params: Dict[str, float | int | str],
) -> np.ndarray:
    """
    Compute X_blob for all ERSPs using a single valley-parameter dict.

    Returns
    -------
    X_blob : (n_samples, max_blobs * 8) float32
        Concatenated [t_start, f_peak, t_peak, sf, st, cov, mean, area_norm]
        for up to max_blobs blobs per ERSP (zero padded if fewer blobs).
    """
    return s22_build_blob_feature_matrix(
        ersp_list=ersp_list,
        max_blobs=int(valley_params["max_blobs"]),
        thr_pos=float(valley_params["thr_pos"]),
        thr_neg=float(valley_params["thr_neg"]),
        delta_valley=float(valley_params["delta_valley"]),
        min_mean_pos=float(valley_params["min_mean_pos"]),
        max_mean_neg=float(valley_params["max_mean_neg"]),
        sign_mode=str(valley_params["sign_mode"]),
    )


# ------------------------------------------------------------
# 2. Per-ERSP max blob score
# ------------------------------------------------------------

def s25_compute_max_blob_scores(
    ersp_list: List[np.ndarray],
    valley_params: Dict[str, float | int | str],
) -> np.ndarray:
    """
    For each ERSP, segment blobs and return the max score across blobs.

    Score is the same score used in s21_segment_valley_blobs:
        score = |mean| * area
    """
    thr_pos      = float(valley_params["thr_pos"])
    thr_neg      = float(valley_params["thr_neg"])
    delta_valley = float(valley_params["delta_valley"])
    min_mean_pos = float(valley_params["min_mean_pos"])
    max_mean_neg = float(valley_params["max_mean_neg"])
    max_blobs    = int(valley_params["max_blobs"])
    sign_mode    = str(valley_params["sign_mode"])

    n_samples = len(ersp_list)
    max_scores = np.zeros(n_samples, dtype=np.float32)

    for i, ersp in enumerate(ersp_list):
        blobs = s21_segment_valley_blobs(
            ersp,
            thr_pos=thr_pos,
            thr_neg=thr_neg,
            delta_valley=delta_valley,
            min_mean_pos=min_mean_pos,
            max_mean_neg=max_mean_neg,
            max_blobs=max_blobs,
            sign_mode=sign_mode,
        )
        if blobs:
            max_scores[i] = max(b["score"] for b in blobs)
        else:
            max_scores[i] = 0.0

    return max_scores


# ------------------------------------------------------------
# 3. Score gating
# ------------------------------------------------------------

def s30_gate_by_blob_score(
    df_meta,
    ersp_list: List[np.ndarray],
    X_blob: np.ndarray,
    max_scores: np.ndarray,
    quantile: float,
    manual_threshold: float | None = None,
) -> Tuple:
    """
    Apply electrode-level gating based on max blob score.

    Parameters
    ----------
    df_meta : DataFrame
        Metadata for all ERSPs (pre-gating).
    ersp_list : list[np.ndarray]
        ERSPs (pre-gating).
    X_blob : np.ndarray
        Feature matrix (pre-gating).
    max_scores : np.ndarray
        Max blob score per ERSP.
    quantile : float
        Quantile to propose a score threshold.
    manual_threshold : float or None
        If not None, overrides the quantile-based threshold.

    Returns
    -------
    df_meta_f : DataFrame
    ersp_list_f : list[np.ndarray]
    X_blob_f : np.ndarray
    keep_idx : np.ndarray[int]
    score_thresh : float
    keep_mask : np.ndarray[bool]
    drop_idx : np.ndarray[int]
    """
    assert len(max_scores) == len(df_meta) == len(ersp_list) == X_blob.shape[0]

    # Proposed threshold from quantile
    q_thresh = float(np.quantile(max_scores, quantile))
    score_thresh = float(manual_threshold) if manual_threshold is not None else q_thresh

    print(f"Score gating:")
    print(f"  quantile={quantile:.2f} → q_thresh={q_thresh:.3f}")
    print(f"  using score_thresh={score_thresh:.3f} "
          f"({'manual' if manual_threshold is not None else 'from quantile'})")

    keep_mask = max_scores >= score_thresh
    drop_mask = ~keep_mask
    keep_idx = np.where(keep_mask)[0]
    drop_idx = np.where(drop_mask)[0]

    df_meta["keep_for_clustering"] = keep_mask

    print(f"  kept   : {keep_mask.sum()} ERSPs")
    print(f"  dropped: {drop_mask.sum()} ERSPs")

    # Apply gating
    df_meta_f = df_meta.iloc[keep_idx].reset_index(drop=True)
    ersp_list_f = [ersp_list[i] for i in keep_idx]
    X_blob_f = X_blob[keep_mask]

    return df_meta_f, ersp_list_f, X_blob_f, keep_idx, score_thresh, keep_mask, drop_idx


# ------------------------------------------------------------
# 4. 50 vs 50 QC panel (kept vs dropped)
# ------------------------------------------------------------

def q32_plot_score_gating_grid(
    ersp_list_raw: List[np.ndarray],
    df_meta_raw,
    keep_idx: np.ndarray,
    drop_idx: np.ndarray,
    score_thresh: float,
    n_show: int = 50,
    rng_seed: int = 42,
):
    """
    Plot small ERSP thumbnails for kept vs dropped electrodes
    (left half = kept, right half = dropped).
    """
    rng = np.random.default_rng(rng_seed)

    n_kept = len(keep_idx)
    n_dropped = len(drop_idx)
    n_show = min(n_show, n_kept, n_dropped)

    if n_show == 0:
        print("Not enough both kept and dropped samples to make a 50/50 panel — skipping.")
        return

    kept_sample_idx = rng.choice(keep_idx, size=n_show, replace=False)
    dropped_sample_idx = rng.choice(drop_idx, size=n_show, replace=False)

    n_cols_each = 15
    n_rows = int(np.ceil(n_show / n_cols_each))

    fig, axes = plt.subplots(
        n_rows,
        2 * n_cols_each,
        figsize=(2 * n_cols_each * 0.7, n_rows * 0.7),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i in range(n_show):
        row = i // n_cols_each
        col = i % n_cols_each

        # Left half: kept
        ax_keep = axes[row, col]
        idx_keep_global = int(kept_sample_idx[i])
        ersp_k = ersp_list_raw[idx_keep_global]
        meta_k = df_meta_raw.iloc[idx_keep_global]

        ax_keep.imshow(
            ersp_k,
            aspect="auto",
            origin="lower",
            cmap="bwr",
            vmin=-6,
            vmax=6,
            interpolation="nearest",
        )
        ax_keep.set_title(
            f"K {meta_k['patient_id']} {meta_k['electrode']}",
            fontsize=5,
        )
        ax_keep.axis("off")

        # Right half: dropped
        ax_drop = axes[row, col + n_cols_each]
        idx_drop_global = int(dropped_sample_idx[i])
        ersp_d = ersp_list_raw[idx_drop_global]
        meta_d = df_meta_raw.iloc[idx_drop_global]

        ax_drop.imshow(
            ersp_d,
            aspect="auto",
            origin="lower",
            cmap="bwr",
            vmin=-6,
            vmax=6,
            interpolation="nearest",
        )
        ax_drop.set_title(
            f"D {meta_d['patient_id']} {meta_d['electrode']}",
            fontsize=5,
        )
        ax_drop.axis("off")

    # Turn off unused axes
    for j in range(2 * n_cols_each * n_rows):
        if j >= 2 * n_show:
            axes_flat[j].axis("off")

    fig.suptitle(
        f"Score gating visual check — left: kept (score ≥ {score_thresh:.2f}), "
        f"right: dropped (score < {score_thresh:.2f})",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ------------------------------------------------------------
# 5. Histogram / CDF of max scores
# ------------------------------------------------------------

def q31_plot_score_histograms(
    max_scores: np.ndarray,
    score_thresh: float,
    quantile: float,
):
    """
    Plot:
      - Histogram of max blob scores (log-x if needed)
      - Cumulative distribution with threshold and quantile line
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 5.1 Plain histogram (log-scale on x)
    ax = axes[0]
    ax.hist(
        max_scores,
        bins=2000,
        alpha=0.8,
        edgecolor="none",
    )
    ax.axvline(score_thresh, color="red", linestyle="--", linewidth=2,
               label=f"threshold = {score_thresh:.2f}")

    ax.set_xlabel("Max blob score per ERSP")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of max blob scores")
    ax.legend(loc="upper right", fontsize=8)
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([0,4000])
    # 5.2 Cumulativ distribution
    ax2 = axes[1]
    sorted_scores = np.sort(max_scores)
    cum = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    ax2.plot(sorted_scores, cum, lw=2)
    ax2.axvline(score_thresh, color="red", linestyle="--", linewidth=2)
    ax2.axhline(quantile, color="gray", linestyle=":", linewidth=1)

    ax2.set_xlabel("Max blob score per ERSP")
    ax2.set_ylabel("Cumulative proportion")
    ax2.set_title("Cumulative distribution of max blob scores")

    ax2.text(
        0.02, 0.05,
        f"Quantile = {quantile:.2f}\n"
        f"Score thresh = {score_thresh:.2f}",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. Overlay grid of blobs (post score-gating)
# ------------------------------------------------------------

def q33_plot_valley_overlay_grid(
    df_meta,
    ersp_list: List[np.ndarray],
    valley_params: Dict[str, float | int | str],
    score_min: float | None = None,
    max_samples: int = 500,
    n_cols: int = 8,
    title_prefix: str = "Valley-based blobs used for features (post score-gating)",
):
    """
    Wrapper around q34_plot_valley_blob_overlays that:
      - picks up to `max_samples` evenly spaced indices
      - forwards valley_params + optional score_min.
    """
    if len(df_meta) == 0:
        print("No samples available for overlays — skipping.")
        return

    step = max(1, len(df_meta) // max_samples)
    sample_indices = list(range(0, len(df_meta), step))[:max_samples]

    q34_plot_valley_blob_overlays(
        indices=sample_indices,
        df_meta=df_meta,
        ersp_list=ersp_list,
        thr_pos=float(valley_params["thr_pos"]),
        thr_neg=float(valley_params["thr_neg"]),
        delta_valley=float(valley_params["delta_valley"]),
        min_mean_pos=float(valley_params["min_mean_pos"]),
        max_mean_neg=float(valley_params["max_mean_neg"]),
        max_blobs=int(valley_params["max_blobs"]),
        sign_mode=str(valley_params["sign_mode"]),
        n_cols=n_cols,
        title_prefix=title_prefix,
        # assuming your updated q34_plot_valley_blob_overlays supports this:
        score_min=score_min,
    )
