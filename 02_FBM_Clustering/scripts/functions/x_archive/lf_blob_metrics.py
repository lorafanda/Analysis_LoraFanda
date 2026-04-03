"""
lf_blob_metrics.py

Utilities for:
- valley-based multi-blob segmentation
- amplitude-based multi-blob feature extraction
- distance metric in blob-feature space (for UMAP / clustering)
- visualization + debug helpers

functions: 
Here you go, one functional-style sentence per function:

1. **`segment_valley_blobs`**
   Computes individual valley-based “mountain” blobs in a single ERSP using thresholds and valley-growth rules, and returns a list of blob dictionaries containing masks, geometry, mean, area, score, and index coordinates.

2. **`compute_valley_blob_features`**
   Computes a fixed-length feature vector for a single ERSP by extracting up to `max_blobs` score-gated blobs and returns a 1D array of normalized blob features (t_start, f_peak, t_peak, sf, st, cov, mean, area_norm).

3. **`build_valley_blob_feature_matrix`**
   Computes valley-based blob feature vectors for all ERSPs in `ersp_list` using the same parameters and score gate, and returns a 2D matrix `X_blob` with shape `(n_samples, max_blobs * 8)`.

4. **`find_valley_merged_blobs`**
   Computes valley-based blobs for a single ERSP with the same logic as `segment_valley_blobs` and, optionally, returns these blobs while also writing a debug GIF that visualizes candidate components, cumulative merges, and final ranked blobs.

5. **`plot_valley_blob_overlays`**
   Computes valley-based blobs for selected ERSP indices using the same parameters and score gate, and returns nothing while plotting ERSP images with blob outlines and simple geometry overlays in a grid.

6. **`init_blob_metric`**
   Computes and stores a global per-dimension weight vector for the blob-feature space based on `max_blobs`, `features_per_blob`, and `feature_type_weights`, and returns nothing (it just configures the global metric state).

7. **`ersp_blob_metric`**
   Computes a weighted Euclidean distance between two blob-feature vectors `a` and `b` using the global weight vector from `init_blob_metric`, and returns a single float distance value.


"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ============================================================
# 1. Valley-based blob segmentation (core logic)
# ============================================================

def segment_valley_blobs(
    ersp: np.ndarray,
    thr_pos: float = 2.0,
    thr_neg: float = -4.0,
    delta_valley: float = 1.0,
    min_mean_pos: float = 2.0,
    max_mean_neg: float = -2.0,
    max_blobs: int = 4,
    sign_mode: str = "both",  # "both", "pos", "neg"
) -> List[Dict[str, Any]]:
    """
    PART 1: Segment 'mountain-range' blobs from a single ERSP (freq x time).

    Positive side:
      core_pos  = ersp >= thr_pos
      merge_pos = ersp >= (thr_pos - delta_valley)

    Negative side:
      core_neg  = ersp <= thr_neg
      merge_neg = ersp <= (thr_neg + delta_valley)

    For each 8-connected component in merge_pos / merge_neg:
      - must contain at least one core pixel
      - must have mean >= min_mean_pos (pos) or <= max_mean_neg (neg)
      - weights = positive amplitude (pos) or |negative amplitude| (neg)
      - score_raw      = |mean| * area
      - score          = score_raw / (f_center_log + eps)
                         where f_center_log is the log-frequency center of mass.

    Returns
    -------
    blobs : list[dict]
        Each dict contains:
          'sign'          : +1 or -1
          'mask'          : 2D bool array, same shape as ersp
          'vals'          : 1D array of ERSP values in this blob
          'weights'       : 1D array of non-negative weights
          'mean'          : float, mean ERSP in blob
          'area'          : int, number of pixels
          'score_raw'     : float, |mean| * area
          'score'         : float, frequency-normalized score
          'freqs_idx'     : 1D int array of row indices
          'times_idx'     : 1D int array of col indices
          'f_center_idx'  : float, center-of-mass row index
          'f_center_log'  : float, log-frequency center used in score

        Blobs are sorted by descending 'score' and truncated to max_blobs.
    """
    n_freq, n_time = ersp.shape
    structure = np.ones((3, 3), dtype=int)  # 8-connected
    blobs: List[Dict[str, Any]] = []

    # --- build a simple "frequency" axis and its log ---
    # For now we treat row index+1 as frequency and work in natural log.
    freq_axis = np.arange(n_freq, dtype=np.float32) + 1.0
    log_freq_axis = np.log(freq_axis)
    eps = 1e-6  # to avoid division by zero

    # -------- positive mountains --------
    if sign_mode in ("both", "pos"):
        core_pos = ersp >= thr_pos
        merge_pos = ersp >= (thr_pos - delta_valley)

        if merge_pos.any():
            labeled, n_labels = ndimage.label(merge_pos, structure=structure)

            for lab in range(1, n_labels + 1):
                mask = (labeled == lab)
                if not mask.any():
                    continue

                vals = ersp[mask]
                mean_val = float(vals.mean())
                if mean_val < min_mean_pos:
                    continue

                weights = vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())
                if total <= 0.0:
                    continue

                has_core = bool((mask & core_pos).any())
                if not has_core:
                    continue

                area = int(mask.sum())
                freqs_idx, times_idx = np.where(mask)

                # ----- frequency-normalized score -----
                # Center of mass in frequency index space (0..n_freq-1)
                f_center_idx = float((freqs_idx * weights).sum() / total)
                # Map to log-frequency via precomputed axis
                f_center_log = float(log_freq_axis[int(round(f_center_idx))])

                score_raw = float(abs(mean_val) * area)
                score = score_raw / (f_center_log + eps)

                blobs.append(
                    {
                        "sign": +1,
                        "mask": mask,
                        "vals": vals,
                        "weights": weights,
                        "mean": mean_val,
                        "area": area,
                        "score_raw": score_raw,
                        "score": score,
                        "freqs_idx": freqs_idx,
                        "times_idx": times_idx,
                        "f_center_idx": f_center_idx,
                        "f_center_log": f_center_log,
                    }
                )

    # -------- negative mountains --------
    if sign_mode in ("both", "neg"):
        core_neg = ersp <= thr_neg
        merge_neg = ersp <= (thr_neg + delta_valley)

        if merge_neg.any():
            labeled, n_labels = ndimage.label(merge_neg, structure=structure)

            for lab in range(1, n_labels + 1):
                mask = (labeled == lab)
                if not mask.any():
                    continue

                vals = ersp[mask]
                mean_val = float(vals.mean())
                if mean_val > max_mean_neg:
                    continue

                weights = -vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())
                if total <= 0.0:
                    continue

                has_core = bool((mask & core_neg).any())
                if not has_core:
                    continue

                area = int(mask.sum())
                freqs_idx, times_idx = np.where(mask)

                # ----- frequency-normalized score -----
                f_center_idx = float((freqs_idx * weights).sum() / total)
                f_center_log = float(log_freq_axis[int(round(f_center_idx))])

                score_raw = float(abs(mean_val) * area)
                score = score_raw / (f_center_log + eps)

                blobs.append(
                    {
                        "sign": -1,
                        "mask": mask,
                        "vals": vals,
                        "weights": weights,
                        "mean": mean_val,
                        "area": area,
                        "score_raw": score_raw,
                        "score": score,
                        "freqs_idx": freqs_idx,
                        "times_idx": times_idx,
                        "f_center_idx": f_center_idx,
                        "f_center_log": f_center_log,
                    }
                )

    if not blobs:
        return []

    # Sort by the new frequency-normalized score
    blobs.sort(key=lambda b: b["score"], reverse=True)
    return blobs[:max_blobs]

# ============================================================
# 2. Feature extraction with score_min gating
# ============================================================

def compute_valley_blob_features(
    ersp: np.ndarray,
    thr_pos: float = 2.0,
    thr_neg: float = -4.0,
    delta_valley: float = 1.0,
    min_mean_pos: float = 2.0,
    max_mean_neg: float = -2.0,
    max_blobs: int = 4,
    sign_mode: str = "both",
    score_min: float = 0.0,
) -> np.ndarray:
    """
    PART 2 (single ERSP): turn valley-blobs into a fixed feature vector.

    Steps:
      1. Segment blobs via `segment_valley_blobs` (geometry + mean-based filtering).
      2. Keep only blobs with `score >= score_min`.
      3. For each remaining blob (in score order) compute, in NORMALIZED coords:

         - t_start_norm : earliest time in blob
         - f_peak_norm  : frequency at max |ERSPS|
         - t_peak_norm  : time at max |ERSPS|
         - sf_norm      : freq spread (std) around CoM
         - st_norm      : time spread (std) around CoM
         - cov_norm     : freq-time covariance around CoM
         - mean         : signed mean ERSP in the blob
         - area_norm    : blob area / (n_freq * n_time)

      4. Pad with zeros up to `max_blobs` blobs.

    Returns
    -------
    feat : np.ndarray, shape (max_blobs * 8,)
    """
    n_freq, n_time = ersp.shape
    blobs_all = segment_valley_blobs(
        ersp,
        thr_pos=thr_pos,
        thr_neg=thr_neg,
        delta_valley=delta_valley,
        min_mean_pos=min_mean_pos,
        max_mean_neg=max_mean_neg,
        max_blobs=max_blobs,
        sign_mode=sign_mode,
    )

    # --- score gating at blob level ---
    if score_min > 0.0:
        blobs = [b for b in blobs_all if b["score"] >= score_min]
    else:
        blobs = blobs_all

    features_per_blob = 8  # t_start, f_peak, t_peak, sf, st, cov, mean, area_norm
    feat = np.zeros(max_blobs * features_per_blob, dtype=np.float32)
    if not blobs:
        return feat

    # normalized axes
    if n_freq > 1:
        freq_norm_axis = np.linspace(0.0, 1.0, n_freq, dtype=np.float32)
    else:
        freq_norm_axis = np.array([0.0], dtype=np.float32)

    if n_time > 1:
        time_norm_axis = np.linspace(0.0, 1.0, n_time, dtype=np.float32)
    else:
        time_norm_axis = np.array([0.0], dtype=np.float32)

    total_bins = float(n_freq * n_time)

    for k, b in enumerate(blobs[:max_blobs]):
        freqs_idx = b["freqs_idx"]
        times_idx = b["times_idx"]
        vals      = b["vals"]
        w         = b["weights"]
        mean_val  = float(b["mean"])
        area      = int(b["area"])

        f = freq_norm_axis[freqs_idx]
        t = time_norm_axis[times_idx]

        total_w = float(w.sum())
        if total_w <= 0.0:
            continue

        # t_start
        t_start_norm = float(t.min())

        # peak |ERSPS|
        abs_vals = np.abs(vals)
        peak_idx = int(np.argmax(abs_vals))
        f_peak_norm = float(f[peak_idx])
        t_peak_norm = float(t[peak_idx])

        # center of mass in normalized coords
        cf = float((f * w).sum() / total_w)
        ct = float((t * w).sum() / total_w)

        df = f - cf
        dt = t - ct

        sf_norm  = float(np.sqrt((w * df * df).sum() / total_w))
        st_norm  = float(np.sqrt((w * dt * dt).sum() / total_w))
        cov_norm = float((w * df * dt).sum() / total_w)

        area_norm = float(area) / total_bins if total_bins > 0 else 0.0

        base = k * features_per_blob
        feat[base + 0] = t_start_norm
        feat[base + 1] = f_peak_norm
        feat[base + 2] = t_peak_norm
        feat[base + 3] = sf_norm
        feat[base + 4] = st_norm
        feat[base + 5] = cov_norm
        feat[base + 6] = mean_val
        feat[base + 7] = area_norm

    return feat


def build_valley_blob_feature_matrix(
    ersp_list: List[np.ndarray],
    score_min: float | None = None,
    max_blobs: int = 4,
    thr_pos: float = 2.0,
    thr_neg: float = -4.0,
    delta_valley: float = 1.0,
    min_mean_pos: float = 2.0,
    max_mean_neg: float = -2.0,
    sign_mode: str = "both",
) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:

    """
    PART 2 (all ERSPs): in ONE pass, for each ERSP:
      - run segment_valley_blobs (with VALLEY_PARAMS)
      - compute max blob score for this ERSP (before score_min gating)
      - optionally drop blobs with score < score_min
      - convert kept blobs into a fixed-length feature vector

    Returns
    -------
    X_blob : np.ndarray, shape (n_samples, max_blobs * 8)
        Valley-blob feature matrix (zero-padded when < max_blobs blobs).
    max_scores : np.ndarray, shape (n_samples,)
        Max blob score per ERSP, BEFORE any score_min gating.
    blobs_per_sample : list[list[dict]]
        List of blobs for each ERSP AFTER score_min gating.
        These blob dicts are exactly what plotting functions should re-use.
    """
    n_samples = len(ersp_list)
    features_per_blob = 8   # t_start, f_peak, t_peak, sf, st, cov, mean, area_norm
    feat_dim = max_blobs * features_per_blob

    X_blob = np.zeros((n_samples, feat_dim), dtype=np.float32)
    max_scores = np.zeros(n_samples, dtype=np.float32)
    blobs_per_sample: List[List[Dict[str, Any]]] = []

    for i, ersp in enumerate(ersp_list):
        n_freq, n_time = ersp.shape
        total_bins = float(n_freq * n_time)

        # --- segment all blobs for this ERSP ---
        blobs = segment_valley_blobs(
            ersp,
            thr_pos=thr_pos,
            thr_neg=thr_neg,
            delta_valley=delta_valley,
            min_mean_pos=min_mean_pos,
            max_mean_neg=max_mean_neg,
            max_blobs=max_blobs,      # applies AFTER sorting by score
            sign_mode=sign_mode,
        )

        # max_score is based on *all* candidate blobs (before score_min)
        if blobs:
            max_scores[i] = max(b["score"] for b in blobs)
        else:
            max_scores[i] = 0.0

        # --- optional per-blob gating by score_min (for features & plots) ---
        if (score_min is not None) and (score_min > 0.0):
            blobs = [b for b in blobs if b["score"] >= score_min]

        blobs_per_sample.append(blobs)

        if not blobs:
            # leave feature row as zeros
            continue

        # Normalised axes
        if n_freq > 1:
            freq_norm_axis = np.linspace(0.0, 1.0, n_freq, dtype=np.float32)
        else:
            freq_norm_axis = np.array([0.0], dtype=np.float32)

        if n_time > 1:
            time_norm_axis = np.linspace(0.0, 1.0, n_time, dtype=np.float32)
        else:
            time_norm_axis = np.array([0.0], dtype=np.float32)

        feat = np.zeros(feat_dim, dtype=np.float32)

        for k, b in enumerate(blobs[:max_blobs]):
            freqs_idx = b["freqs_idx"]
            times_idx = b["times_idx"]
            vals      = b["vals"]
            w         = b["weights"]
            mean_val  = float(b["mean"])
            area      = int(b["area"])

            total_w = float(w.sum())
            if total_w <= 0.0:
                continue

            f = freq_norm_axis[freqs_idx]
            t = time_norm_axis[times_idx]

            # t_start (earliest time in blob)
            t_start_norm = float(t.min())

            # peak |ERSPS|
            abs_vals = np.abs(vals)
            peak_idx = int(np.argmax(abs_vals))
            f_peak_norm = float(f[peak_idx])
            t_peak_norm = float(t[peak_idx])

            # center of mass
            cf = float((f * w).sum() / total_w)
            ct = float((t * w).sum() / total_w)

            df = f - cf
            dt = t - ct

            sf_norm = float(np.sqrt((w * df * df).sum() / total_w))
            st_norm = float(np.sqrt((w * dt * dt).sum() / total_w))
            cov_norm = float((w * df * dt).sum() / total_w)

            # area as fraction of TF plane
            area_norm = float(area) / total_bins if total_bins > 0 else 0.0

            base = k * features_per_blob
            feat[base + 0] = t_start_norm
            feat[base + 1] = f_peak_norm
            feat[base + 2] = t_peak_norm
            feat[base + 3] = sf_norm
            feat[base + 4] = st_norm
            feat[base + 5] = cov_norm
            feat[base + 6] = mean_val
            feat[base + 7] = area_norm

        X_blob[i, :] = feat

    print(
        f"Valley-blob feature matrix: X_blob.shape={X_blob.shape} "
        f"(max_blobs={max_blobs}, features_per_blob={features_per_blob})"
    )
    return X_blob, max_scores, blobs_per_sample


# ============================================================
# 3. GIF debug of valley segmentation (same logic as segment_valley_blobs)
# ============================================================

def find_valley_merged_blobs(
    ersp: np.ndarray,
    thr_pos: float = 2.0,
    thr_neg: float = -4.0,
    delta_valley: float = 1.0,
    min_mean_pos: float = 2.0,
    max_mean_neg: float = -2.0,
    max_blobs: int = 4,
    sign_mode: str = "both",
    # ---- DEBUG OPTIONS ----
    debug: bool = False,
    debug_dir: str | Path | None = None,
    sample_tag: str | None = None,
    max_debug_frames: int = 30,
) -> List[Dict[str, Any]]:
    """
    Debug-friendly version of valley segmentation.

    Uses the same rules as `segment_valley_blobs`, but also:
      - builds animated GIF frames showing:
          * candidate components (ACCEPT / REJECT),
          * cumulative accepted blobs,
          * final merged blobs with rank numbers (1,2,3,... by score).
    """
    blobs: List[Dict[str, Any]] = []
    n_freq, n_time = ersp.shape
    structure = np.ones((3, 3), dtype=int)   # 8-connected
    
    # --- simple frequency + log-frequency axis for scoring ---
    freq_axis = np.arange(n_freq, dtype=np.float32) + 1.0
    log_freq_axis = np.log(freq_axis)
    eps = 1e-6
    debug_frames: List[np.ndarray] = []
    
    if debug and debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    merged_pos = np.zeros_like(ersp, dtype=bool)
    merged_neg = np.zeros_like(ersp, dtype=bool)
    cumulative_step = 0

    def _debug_snapshot(mask, sign_char, mean_val, area, score, has_core, accepted, stage_label):
        nonlocal debug_frames
        if not debug:
            return
        if len(debug_frames) >= max_debug_frames:
            return

        color = "yellow" if (accepted and sign_char == "+") else \
                "cyan"   if (accepted and sign_char == "-") else \
                "red"

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(
            ersp,
            aspect="auto",
            origin="lower",
            cmap="bwr",
            vmin=-6,
            vmax=6,
            interpolation="nearest",
        )

        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=color,
            linewidths=1.0,
            alpha=0.9,
            origin="lower",
        )

        title_tag = sample_tag if sample_tag is not None else "sample"
        ax.set_title(
            f"{title_tag} — {stage_label} {sign_char}-component\n"
            f"{'ACCEPT' if accepted else 'REJECT'}",
            fontsize=8,
        )

        txt = (
            f"mean={mean_val:.2f}, area={area}, score={score:.1f}\n"
            f"has_core={has_core}"
        )
        ax.text(
            0.01, 0.99,
            txt,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel("Time (bins)", fontsize=7)
        ax.set_ylabel("Freq (bins)", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(h, w, 3)
        debug_frames.append(frame)
        plt.close(fig)

    def _debug_cumulative(merged_pos_mask, merged_neg_mask, step_idx):
        nonlocal debug_frames
        if not debug:
            return
        if len(debug_frames) >= max_debug_frames:
            return

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(
            ersp,
            aspect="auto",
            origin="lower",
            cmap="bwr",
            vmin=-6,
            vmax=6,
            interpolation="nearest",
        )

        if merged_pos_mask.any():
            ax.contour(
                merged_pos_mask.astype(float),
                levels=[0.5],
                colors="yellow",
                linewidths=1.2,
                alpha=0.9,
                origin="lower",
            )
        if merged_neg_mask.any():
            ax.contour(
                merged_neg_mask.astype(float),
                levels=[0.5],
                colors="cyan",
                linewidths=1.2,
                alpha=0.9,
                origin="lower",
            )

        title_tag = sample_tag if sample_tag is not None else "sample"
        ax.set_title(f"{title_tag} — cumulative blobs (step {step_idx})",
                     fontsize=9)

        ax.set_xlabel("Time (bins)", fontsize=7)
        ax.set_ylabel("Freq (bins)", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(h, w, 3)
        debug_frames.append(frame)
        plt.close(fig)

    def _debug_final(blobs_kept):
        nonlocal debug_frames
        if not debug or not blobs_kept:
            return

        final_pos = np.zeros_like(ersp, dtype=bool)
        final_neg = np.zeros_like(ersp, dtype=bool)
        for b in blobs_kept:
            if b["sign"] > 0:
                final_pos |= b["mask"]
            else:
                final_neg |= b["mask"]

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(
            ersp,
            aspect="auto",
            origin="lower",
            cmap="bwr",
            vmin=-6,
            vmax=6,
            interpolation="nearest",
        )

        if final_pos.any():
            ax.contour(
                final_pos.astype(float),
                levels=[0.5],
                colors="yellow",
                linewidths=1.2,
                alpha=0.9,
                origin="lower",
            )
        if final_neg.any():
            ax.contour(
                final_neg.astype(float),
                levels=[0.5],
                colors="cyan",
                linewidths=1.2,
                alpha=0.9,
                origin="lower",
            )

        # number blobs by score rank (1,2,3,...)
        for rank, b in enumerate(blobs_kept, start=1):
            w = b["weights"]
            freqs = b["freqs"]
            times = b["times"]
            total_w = float(w.sum())
            if total_w <= 0.0:
                continue

            cf_idx = float((freqs * w).sum() / total_w)
            ct_idx = float((times * w).sum() / total_w)

            ax.text(
                ct_idx,
                cf_idx,
                str(rank),
                color="black",
                fontsize=8,
                ha="center",
                va="center",
                weight="bold",
                zorder=4,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        title_tag = sample_tag if sample_tag is not None else "sample"
        ax.set_title(f"{title_tag} — FINAL merged blobs (ranked)", fontsize=9)

        n_pos = sum(1 for b in blobs_kept if b["sign"] > 0)
        n_neg = sum(1 for b in blobs_kept if b["sign"] < 0)
        txt = f"Kept blobs: {len(blobs_kept)} (pos={n_pos}, neg={n_neg})"
        ax.text(
            0.01, 0.99,
            txt,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel("Time (bins)", fontsize=7)
        ax.set_ylabel("Freq (bins)", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(h, w, 3)
        debug_frames.append(frame)
        plt.close(fig)

    # positive / negative loops – same logic as segment_valley_blobs
    if sign_mode in ("both", "pos"):
        core_pos  = ersp >= thr_pos
        merge_pos = ersp >= (thr_pos - delta_valley)

        if merge_pos.any():
            labeled, n_labels = ndimage.label(merge_pos, structure=structure)
            for lab in range(1, n_labels + 1):
                mask = (labeled == lab)
                if not mask.any():
                    continue

                vals = ersp[mask]
                mean_val = float(vals.mean())
                area = int(mask.sum())
                has_core = bool((mask & core_pos).any())

                weights = vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())
                
                # frequency-normalized score (mirror of segment_valley_blobs)
                if area > 0 and total > 0.0:
                    freqs, times = np.where(mask)
                    f_center_idx = float((freqs * weights).sum() / total)
                    f_center_log = float(log_freq_axis[int(round(f_center_idx))])
                    score_raw = float(abs(mean_val) * area)
                    score = score_raw / (f_center_log + eps)
                else:
                    score_raw = 0.0
                    score = 0.0

                accepted = (
                    has_core
                    and (mean_val >= min_mean_pos)
                    and (total > 0.0)
                )

                _debug_snapshot(mask, "+", mean_val, area, score, has_core, accepted, "pos")
                if not accepted:
                    continue

                merged_pos |= mask
                cumulative_step += 1
                _debug_cumulative(merged_pos, merged_neg, cumulative_step)

                freqs, times = np.where(mask)
                blobs.append({
                    "sign": +1,
                    "mask": mask,
                    "vals": vals,
                    "weights": weights,
                    "mean": mean_val,
                    "area": area,
                    "score": score,
                    "freqs": freqs,
                    "times": times,
                })

    if sign_mode in ("both", "neg"):
        core_neg  = ersp <= thr_neg
        merge_neg = ersp <= (thr_neg + delta_valley)

        if merge_neg.any():
            labeled, n_labels = ndimage.label(merge_neg, structure=structure)
            for lab in range(1, n_labels + 1):
                mask = (labeled == lab)
                if not mask.any():
                    continue

                vals = ersp[mask]
                mean_val = float(vals.mean())
                area = int(mask.sum())
                has_core = bool((mask & core_neg).any())

                weights = -vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())
                
                # frequency-normalized score (mirror of segment_valley_blobs)
                if area > 0 and total > 0.0:
                    freqs, times = np.where(mask)
                    f_center_idx = float((freqs * weights).sum() / total)
                    f_center_log = float(log_freq_axis[int(round(f_center_idx))])
                    score_raw = float(abs(mean_val) * area)
                    score = score_raw / (f_center_log + eps)
                else:
                    score_raw = 0.0
                    score = 0.0

                accepted = (
                    has_core
                    and (mean_val <= max_mean_neg)
                    and (total > 0.0)
                )

                _debug_snapshot(mask, "-", mean_val, area, score, has_core, accepted, "neg")
                if not accepted:
                    continue

                merged_neg |= mask
                cumulative_step += 1
                _debug_cumulative(merged_pos, merged_neg, cumulative_step)

                freqs, times = np.where(mask)
                blobs.append({
                    "sign": -1,
                    "mask": mask,
                    "vals": vals,
                    "weights": weights,
                    "mean": mean_val,
                    "area": area,
                    "score": score,
                    "freqs": freqs,
                    "times": times,
                })

    if not blobs:
        if debug and debug_frames and debug_dir is not None:
            gif_path = Path(debug_dir) / f"{sample_tag or 'sample'}_valley_debug.gif"
            imageio.mimsave(gif_path, debug_frames, duration=0.7)
            print("[DEBUG] Saved valley-debug GIF (no accepted blobs) to:", gif_path)
        return []

    blobs.sort(key=lambda b: b["score"], reverse=True)
    blobs = blobs[:max_blobs]

    _debug_final(blobs)

    if debug and debug_frames and debug_dir is not None:
        gif_path = Path(debug_dir) / f"{sample_tag or 'sample'}_valley_debug.gif"
        imageio.mimsave(gif_path, debug_frames, duration=0.7)
        print("[DEBUG] Saved valley-debug GIF to:", gif_path)

    return blobs


# ============================================================
# 4. Overlay visualization (now also score-gated)
# ============================================================

def plot_valley_blob_overlays(
    indices,
    df_meta,
    ersp_list,
    thr_pos: float = 2.0,
    thr_neg: float = -4.0,
    delta_valley: float = 1.0,
    min_mean_pos: float = 2.0,
    max_mean_neg: float = -2.0,
    max_blobs: int = 4,
    sign_mode: str = "both",
    score_min: float = 0.0,
    blobs_per_sample: list | None = None,
    n_cols: int = 4,
    title_prefix: str = "Valley-based blob overlays",
):
    """
    Visualize valley-blobs directly on ERSPs.

    If blobs_per_sample is provided, we REUSE those blobs (already segmented
    and optionally score-gated). Otherwise we call segment_valley_blobs
    internally with the supplied parameters.
    """
    if len(indices) == 0:
        print("No indices provided to plot_valley_blob_overlays.")
        return

    n = len(indices)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax, idx in zip(axes_flat, indices):
        if idx >= len(ersp_list):
            ax.axis("off")
            continue

        ersp = ersp_list[idx]
        meta = df_meta.iloc[idx]
        n_freq, n_time = ersp.shape

        ax.imshow(
            ersp,
            aspect="auto",
            origin="lower",
            cmap="bwr",
            vmin=-6,
            vmax=6,
            interpolation="nearest",
        )

        ax.set_title(
            f"{meta['patient_id']} {meta['electrode']} {meta['condition']}",
            fontsize=7,
        )
        ax.set_xlabel("Time (bins)", fontsize=6)
        ax.set_ylabel("Freq (bins)", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)

        # --- get blobs for this sample ---
        if blobs_per_sample is not None:
            blobs = blobs_per_sample[idx]
        else:
            blobs = segment_valley_blobs(
                ersp,
                thr_pos=thr_pos,
                thr_neg=thr_neg,
                delta_valley=delta_valley,
                min_mean_pos=min_mean_pos,
                max_mean_neg=max_mean_neg,
                max_blobs=max_blobs,
                sign_mode=sign_mode,
            )
            if score_min > 0.0:
                blobs = [b for b in blobs if b["score"] >= score_min]

        if not blobs:
            continue

        for b in blobs:
            sign = b["sign"]
            mask = b["mask"]
            w = b["weights"]
            freqs_idx = b["freqs_idx"]
            times_idx = b["times_idx"]

            total_w = float(w.sum())
            if total_w <= 0.0:
                continue

            color = "yellow" if sign > 0 else "cyan"

            # center of mass in index space
            cf_idx = float((freqs_idx * w).sum() / total_w)
            ct_idx = float((times_idx * w).sum() / total_w)

            df_idx = freqs_idx - cf_idx
            dt_idx = times_idx - ct_idx
            sf_idx = float(np.sqrt((w * df_idx * df_idx).sum() / total_w))
            st_idx = float(np.sqrt((w * dt_idx * dt_idx).sum() / total_w))
            cov_idx = float((w * df_idx * dt_idx).sum() / total_w)

            # outline
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                colors=color,
                linewidths=0.7,
                alpha=0.9,
                origin="lower",
            )

            # center marker
            ax.scatter(
                ct_idx, cf_idx,
                marker="o",
                s=20,
                c=color,
                edgecolor="black",
                linewidths=0.5,
                zorder=3,
            )

            # vertical spread
            f_low = max(0, cf_idx - 2 * sf_idx)
            f_high = min(n_freq - 1, cf_idx + 2 * sf_idx)
            ax.plot(
                [ct_idx, ct_idx],
                [f_low, f_high],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )

            # horizontal spread
            t_low = max(0, ct_idx - 2 * st_idx)
            t_high = min(n_time - 1, ct_idx + 2 * st_idx)
            ax.plot(
                [t_low, t_high],
                [cf_idx, cf_idx],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )

            # diagonal line for covariance sign
            if cov_idx != 0.0:
                slope = np.sign(cov_idx)
                half_len_t = max(st_idx, 1.0)

                t1 = max(0, ct_idx - half_len_t)
                t2 = min(n_time - 1, ct_idx + half_len_t)

                f1 = cf_idx - slope * sf_idx
                f2 = cf_idx + slope * sf_idx

                f1 = np.clip(f1, 0, n_freq - 1)
                f2 = np.clip(f2, 0, n_freq - 1)

                ax.plot(
                    [t1, t2],
                    [f1, f2],
                    color=color,
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=2,
                )

    # turn off unused axes
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{title_prefix}\n(thr_pos={thr_pos}, thr_neg={thr_neg}, Δvalley={delta_valley}, score_min={score_min})",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



# ============================================================
# 5. Metric in blob-feature space (with per-feature-type weights)
# ============================================================

BLOB_FEATURES_PER = 8   # default; overwritten by init_blob_metric if needed
BLOB_MAX_COUNT    = 4
BLOB_VEC_DIM      = BLOB_FEATURES_PER * BLOB_MAX_COUNT
BLOB_WEIGHT_VEC   = None  # 1D vector of length BLOB_VEC_DIM

# Canonical feature order inside each blob
FEATURE_NAME_ORDER = [
    "t_start",
    "f_peak",
    "t_peak",
    "sf",
    "st",
    "cov",
    "mean",
    "area_norm",
]


def init_blob_metric(
    n_freq=None,
    n_time=None,
    max_blobs: int = 4,
    features_per_blob: int = 8,
    feature_type_weights: dict | None = None,
):
    """
    Configure global dimensions + per-feature weights for the blob-feature metric.

    Parameters
    ----------
    max_blobs : int
        How many blobs per electrode are encoded.
    features_per_blob : int
        Number of features per blob (first `features_per_blob` entries of FEATURE_NAME_ORDER).
    feature_type_weights : dict or None
        Mapping from feature name -> scalar weight, e.g.:
          {
            "t_start": 1.5,
            "f_peak":  1.5,
            "t_peak":  1.5,
            "sf":      1.2,
            "st":      1.2,
            "cov":     1.0,
            "mean":    0.5,
            "area_norm": 0.1,
          }
        Any feature not in the dict defaults to weight 1.0.
    """
    global BLOB_FEATURES_PER, BLOB_MAX_COUNT, BLOB_VEC_DIM, BLOB_WEIGHT_VEC

    BLOB_FEATURES_PER = int(features_per_blob)
    BLOB_MAX_COUNT    = int(max_blobs)
    BLOB_VEC_DIM      = BLOB_FEATURES_PER * BLOB_MAX_COUNT

    if feature_type_weights is None:
        feature_type_weights = {}

    per_blob_weights = []
    for j in range(BLOB_FEATURES_PER):
        feat_name = FEATURE_NAME_ORDER[j]
        w = float(feature_type_weights.get(feat_name, 1.0))
        per_blob_weights.append(w)
    per_blob_weights = np.array(per_blob_weights, dtype=np.float32)

    BLOB_WEIGHT_VEC = np.tile(per_blob_weights, BLOB_MAX_COUNT).astype(np.float32)

    print(
        f"[init_blob_metric] {BLOB_MAX_COUNT} blobs × {BLOB_FEATURES_PER} features "
        f"= {BLOB_VEC_DIM}D; weight_vec.shape={BLOB_WEIGHT_VEC.shape}"
    )


def ersp_blob_metric(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance between two blob-feature vectors a, b in R^{BLOB_VEC_DIM},
    with per-feature-type weights stored in BLOB_WEIGHT_VEC.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    diff = a - b

    if BLOB_WEIGHT_VEC is not None:
        if diff.shape != BLOB_WEIGHT_VEC.shape:
            raise ValueError(
                f"ersp_blob_metric: diff.shape={diff.shape} != "
                f"BLOB_WEIGHT_VEC.shape={BLOB_WEIGHT_VEC.shape}"
            )
        diff = diff * BLOB_WEIGHT_VEC

    return float(np.sqrt(np.dot(diff, diff)))
