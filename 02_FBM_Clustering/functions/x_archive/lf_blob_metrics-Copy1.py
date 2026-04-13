"""
lf_blob_metrics.py

Utilities for:
- high-activity electrode screening in ERSP maps
- amplitude-based multi-blob feature extraction
- distance metric in blob-feature space (for UMAP)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from typing import List, Tuple


# ------------------------------------------------------------
# 1. High-activity screening
# ------------------------------------------------------------

def high_activity_screening(
    ersp_list: List[np.ndarray],
    thr_pos: float = 2.5,
    min_prop_pos: float = 0.03,
    thr_neg: float = -4.0,
    min_prop_neg: float = 0.08,
    round_decimals: int = 1,
):
    """
    For each ERSP (freq x time):

      1. Replace NaNs with 0 and round to `round_decimals` in-place.
      2. Compute proportion of bins > thr_pos and < thr_neg.
      3. Mark electrode 'high_activity' if:
            prop(>thr_pos) >= min_prop_pos  OR
            prop(<thr_neg) >= min_prop_neg
      4. Build a sign-agnostic shape map:
            shape[i,j] = |ERSPSample[i,j]| if it crosses either threshold
                          0 otherwise.

    Returns
    -------
    prop_pos : list[float]
        Proportion of bins > thr_pos per ERSP.

    prop_neg : list[float]
        Proportion of bins < thr_neg per ERSP.

    high_flags : list[bool]
        High-activity boolean flag per ERSP.

    shape_list : list[np.ndarray]
        List of shape maps (same shape as ERSP, dtype=float32).
    """
    prop_pos: List[float] = []
    prop_neg: List[float] = []
    high_flags: List[bool] = []
    shape_list: List[np.ndarray] = []

    for arr in ersp_list:
        # ensure finite + round in-place (memory-conscious)
        np.nan_to_num(arr, nan=0.0, copy=False)
        np.round(arr, round_decimals, out=arr)

        mask_pos = arr > thr_pos
        mask_neg = arr < thr_neg
        mask_any = mask_pos | mask_neg

        p_pos = mask_pos.mean()
        p_neg = mask_neg.mean()

        prop_pos.append(float(p_pos))
        prop_neg.append(float(p_neg))

        is_high = (p_pos >= min_prop_pos) or (p_neg >= min_prop_neg)
        high_flags.append(bool(is_high))

        # amplitude-only shape map
        shape = np.zeros_like(arr, dtype=np.float32)
        shape[mask_any] = np.abs(arr[mask_any]).astype(np.float32)
        shape_list.append(shape)

    return prop_pos, prop_neg, high_flags, shape_list


# ------------------------------------------------------------
# 2. Filter high-activity electrodes
# ------------------------------------------------------------

def filter_high_activity(
    df_meta,
    ersp_list: List[np.ndarray],
    shape_list: List[np.ndarray],
    high_flags: List[bool],
):
    """
    Keep only electrodes with high_activity=True.

    Parameters
    ----------
    df_meta : pandas.DataFrame
        Metadata with one row per ERSP sample.
    ersp_list : list of np.ndarray
    shape_list : list of np.ndarray
    high_flags : list of bool

    Returns
    -------
    df_meta_f, ersp_list_f, shape_list_f
        Filtered versions containing only high-activity entries.
    """
    keep_idx = [i for i, f in enumerate(high_flags) if f]
    drop_idx = [i for i, f in enumerate(high_flags) if not f]

    print("====== High-Activity Electrode Filtering ======")
    print(f"Total electrodes before filtering: {len(df_meta)}")
    print(f"High-activity electrodes kept:    {len(keep_idx)}")
    print(f"Low-activity electrodes removed:  {len(drop_idx)}")
    print("===============================================")

    df_meta_f = df_meta.iloc[keep_idx].reset_index(drop=True)
    ersp_list_f = [ersp_list[i] for i in keep_idx]
    shape_list_f = [shape_list[i] for i in keep_idx]

    df_meta_f["sample_idx"] = np.arange(len(df_meta_f))

    print("\nAfter filtering:")
    print("  df_meta rows:", len(df_meta_f))
    print("  ersp_list len:", len(ersp_list_f))
    print("  shape_list len:", len(shape_list_f))

    return df_meta_f, ersp_list_f, shape_list_f


# ------------------------------------------------------------
# 3. Blob features (sign-agnostic, amplitude-based)
# ------------------------------------------------------------

# in functions/lf_blob_metrics.py

from scipy import ndimage
import numpy as np

# ============================================================
# Valley-based blob segmentation + feature extraction
# ============================================================

from typing import List, Dict, Any
import numpy as np
from scipy import ndimage


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
      - score = |mean| * area

    Returns
    -------
    blobs : list[dict]
        Each dict contains:
          'sign'      : +1 or -1
          'mask'      : 2D bool array, same shape as ersp
          'vals'      : 1D array of ERSP values in this blob
          'weights'   : 1D array of non-negative weights
          'mean'      : float, mean ERSP in blob
          'area'      : int, number of pixels
          'score'     : float, |mean| * area
          'freqs_idx' : 1D int array of row indices
          'times_idx' : 1D int array of col indices

        Blobs are sorted by descending 'score' and truncated to max_blobs.
    """
    n_freq, n_time = ersp.shape
    structure = np.ones((3, 3), dtype=int)  # 8-connected
    blobs: List[Dict[str, Any]] = []

    # -------- positive mountains --------
    if sign_mode in ("both", "pos"):
        core_pos = ersp >= thr_pos
        merge_pos = ersp >= (thr_pos - delta_valley)

        if merge_pos.any():
            labeled, n_labels = ndimage.label(merge_pos, structure=structure)

            for lab in range(1, n_labels + 1):
                mask = (labeled == lab)
                # must contain core pixels
                if not (mask & core_pos).any():
                    continue

                vals = ersp[mask]
                mean_val = float(vals.mean())
                if mean_val < min_mean_pos:
                    continue

                # weights: positive amplitude
                weights = vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())
                if total <= 0.0:
                    continue

                freqs_idx, times_idx = np.where(mask)
                blobs.append(
                    {
                        "sign": +1,
                        "mask": mask,
                        "vals": vals,
                        "weights": weights,
                        "mean": mean_val,
                        "area": int(mask.sum()),
                        "score": float(abs(mean_val) * mask.sum()),
                        "freqs_idx": freqs_idx,
                        "times_idx": times_idx,
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
                if not (mask & core_neg).any():
                    continue

                vals = ersp[mask]
                mean_val = float(vals.mean())
                if mean_val > max_mean_neg:
                    continue

                # weights: magnitude of negativity
                weights = -vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())
                if total <= 0.0:
                    continue

                freqs_idx, times_idx = np.where(mask)
                blobs.append(
                    {
                        "sign": -1,
                        "mask": mask,
                        "vals": vals,
                        "weights": weights,
                        "mean": mean_val,
                        "area": int(mask.sum()),
                        "score": float(abs(mean_val) * mask.sum()),
                        "freqs_idx": freqs_idx,
                        "times_idx": times_idx,
                    }
                )

    if not blobs:
        return []

    # sort strongest first
    blobs.sort(key=lambda b: b["score"], reverse=True)
    return blobs[:max_blobs]


def compute_valley_blob_features(
    ersp: np.ndarray,
    thr_pos: float = 2.0,
    thr_neg: float = -4.0,
    delta_valley: float = 1.0,
    min_mean_pos: float = 2.0,
    max_mean_neg: float = -2.0,
    max_blobs: int = 4,
    sign_mode: str = "both",
) -> np.ndarray:
    """
    PART 2 (single ERSP): turn valley-blobs into a fixed feature vector.

    For each blob we compute in NORMALIZED coordinates (0..1):
      - t_start_norm : earliest time in blob
      - f_peak_norm  : frequency at max |ERSPS|
      - t_peak_norm  : time at max |ERSPS|
      - sf_norm      : freq spread (std) around CoM
      - st_norm      : time spread (std) around CoM
      - cov_norm     : freq-time covariance around CoM
      - mean         : signed mean ERSP in the blob
      - area_norm    : blob area / (n_freq * n_time)

    Blobs are sorted by descending score (same as segment_valley_blobs)
    and we keep up to max_blobs. Missing blobs are padded with zeros.

    Returns
    -------
    feat : np.ndarray, shape (max_blobs * 8,)
    """
    n_freq, n_time = ersp.shape
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

    features_per_blob = 8  # now include mean + area_norm
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

        sf_norm = float(np.sqrt((w * df * df).sum() / total_w))
        st_norm = float(np.sqrt((w * dt * dt).sum() / total_w))
        cov_norm = float((w * df * dt).sum() / total_w)

        # area as fraction of the full TF plane
        area_norm = float(area) / total_bins if total_bins > 0 else 0.0

        base = k * features_per_blob
        feat[base + 0] = t_start_norm
        feat[base + 1] = f_peak_norm
        feat[base + 2] = t_peak_norm
        feat[base + 3] = sf_norm
        feat[base + 4] = st_norm
        feat[base + 5] = cov_norm
        feat[base + 6] = mean_val      # signed mean dB
        feat[base + 7] = area_norm     # normalized area

    return feat




from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as imageio  # ensure imageio is installed


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as imageio


def find_valley_merged_blobs(
    ersp,
    thr_pos=2.0,
    thr_neg=-4.0,
    delta_valley=1.0,      # how far below threshold valleys may go and still be "same mountain"
    min_mean_pos=2.0,      # discard pos blobs with mean < this
    max_mean_neg=-2.0,     # discard neg blobs with mean > this
    max_blobs=4,
    sign_mode="both",      # "both", "pos", "neg"
    # ---- DEBUG OPTIONS ----
    debug=False,
    debug_dir=None,
    sample_tag=None,
    max_debug_frames=30,
):
    """
    Return up to `max_blobs` strongest blobs in this ERSP.

    Strategy:
      - Positive side:
          core_pos  = ersp >= thr_pos
          merge_pos = ersp >= (thr_pos - delta_valley)
        → connected components on merge_pos define "mountain ranges",
          but we keep only components that contain some core_pos pixels
          and have mean >= min_mean_pos.

      - Negative side:
          core_neg  = ersp <= thr_neg
          merge_neg = ersp <= (thr_neg + delta_valley)
        → same logic, mean <= max_mean_neg.

      - Each blob gets:
          sign (+1 / -1), mask, mean value, area, score (sorting key).

      - We then:
          - optionally filter by sign_mode,
          - sort by score descending,
          - return up to max_blobs blobs.

    DEBUG MODE:
      If debug=True, we:
        * create a frame per candidate component (ACCEPT / REJECT),
        * create cumulative frames as accepted blobs are added,
        * add a final frame with **all accepted blobs merged** and
          numbered (1,2,3,...) by descending score,
        * write a GIF to `debug_dir` named with `sample_tag`.
    """
    blobs = []
    n_freq, n_time = ersp.shape
    structure = np.ones((3, 3), dtype=int)   # 8-connected

    # Collect frames for GIF if debug is on
    debug_frames = []
    if debug and debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Cumulative accepted blobs (for “level-by-level” visualization)
    merged_pos = np.zeros_like(ersp, dtype=bool)
    merged_neg = np.zeros_like(ersp, dtype=bool)
    cumulative_step = 0

    # ---------- helpers for debug snapshots ----------

    def _debug_snapshot(mask, sign_char, mean_val, area, score, has_core, accepted, stage_label):
        """One frame per candidate component."""
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
        """Frame showing how accepted blobs accumulate over time."""
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
        """
        Final frame with all accepted blobs merged AND numbered by rank.

        Rank = order in `blobs_kept` (already sorted by descending score).
        """
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

        # Number each blob by rank (1,2,3,...) at its weighted center
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

    # ---------- positive mountains ----------
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

                # weights for geometry: use positive amplitude
                weights = vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())

                score = float(abs(mean_val) * area)

                accepted = (
                    has_core
                    and (mean_val >= min_mean_pos)
                    and (total > 0.0)
                )

                _debug_snapshot(
                    mask=mask,
                    sign_char="+",
                    mean_val=mean_val,
                    area=area,
                    score=score,
                    has_core=has_core,
                    accepted=accepted,
                    stage_label="pos",
                )

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

    # ---------- negative mountains ----------
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

                # weights: magnitude of negativity
                weights = -vals.copy()
                weights[weights < 0] = 0.0
                total = float(weights.sum())

                score = float(abs(mean_val) * area)

                accepted = (
                    has_core
                    and (mean_val <= max_mean_neg)
                    and (total > 0.0)
                )

                _debug_snapshot(
                    mask=mask,
                    sign_char="-",
                    mean_val=mean_val,
                    area=area,
                    score=score,
                    has_core=has_core,
                    accepted=accepted,
                    stage_label="neg",
                )

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
            gif_path = debug_dir / f"{sample_tag or 'sample'}_valley_debug.gif"
            imageio.mimsave(gif_path, debug_frames, duration=0.7)
            print("[DEBUG] Saved valley-debug GIF (no accepted blobs) to:", gif_path)
        return []

    # sort strongest first and keep top N
    blobs.sort(key=lambda b: b["score"], reverse=True)
    blobs = blobs[:max_blobs]

    # final ranked frame
    _debug_final(blobs)

    # save GIF
    if debug and debug_frames and debug_dir is not None:
        gif_path = debug_dir / f"{sample_tag or 'sample'}_valley_debug.gif"
        imageio.mimsave(gif_path, debug_frames, duration=0.7)
        print("[DEBUG] Saved valley-debug GIF to:", gif_path)

    return blobs



# def build_valley_blob_feature_matrix(
#     ersp_list: List[np.ndarray],
#     max_blobs: int = 4,
#     thr_pos: float = 2.0,
#     thr_neg: float = -4.0,
#     delta_valley: float = 1.0,
#     min_mean_pos: float = 2.0,
#     max_mean_neg: float = -2.0,
#     sign_mode: str = "both",
# ) -> np.ndarray:
#     """
#     PART 2 (all ERSPs): loop compute_valley_blob_features over electrodes.

#     Returns
#     -------
#     X_blob : np.ndarray, shape (n_samples, max_blobs * 7)
#     """
#     n_samples = len(ersp_list)
#     features_per_blob = 7
#     feat_dim = max_blobs * features_per_blob

#     X_blob = np.zeros((n_samples, feat_dim), dtype=np.float32)

#     for i, ersp in enumerate(ersp_list):
#         X_blob[i, :] = compute_valley_blob_features(
#             ersp,
#             thr_pos=thr_pos,
#             thr_neg=thr_neg,
#             delta_valley=delta_valley,
#             min_mean_pos=min_mean_pos,
#             max_mean_neg=max_mean_neg,
#             max_blobs=max_blobs,
#             sign_mode=sign_mode,
#         )

#     print(
#         f"Valley-blob feature matrix: X_blob.shape={X_blob.shape} "
#         f"(max_blobs={max_blobs}, features_per_blob={features_per_blob})"
#     )
#     return X_blob

# import matplotlib.pyplot as plt

def build_valley_blob_feature_matrix(
    ersp_list: List[np.ndarray],
    max_blobs: int = 4,
    thr_pos: float = 2.0,
    thr_neg: float = -4.0,
    delta_valley: float = 1.0,
    min_mean_pos: float = 2.0,
    max_mean_neg: float = -2.0,
    sign_mode: str = "both",
) -> np.ndarray:
    """
    PART 2 (all ERSPs): loop compute_valley_blob_features over electrodes.

    Returns
    -------
    X_blob : np.ndarray, shape (n_samples, max_blobs * 8)
    """
    n_samples = len(ersp_list)
    features_per_blob = 8   # was 7
    feat_dim = max_blobs * features_per_blob

    X_blob = np.zeros((n_samples, feat_dim), dtype=np.float32)

    for i, ersp in enumerate(ersp_list):
        X_blob[i, :] = compute_valley_blob_features(
            ersp,
            thr_pos=thr_pos,
            thr_neg=thr_neg,
            delta_valley=delta_valley,
            min_mean_pos=min_mean_pos,
            max_mean_neg=max_mean_neg,
            max_blobs=max_blobs,
            sign_mode=sign_mode,
        )

    print(
        f"Valley-blob feature matrix: X_blob.shape={X_blob.shape} "
        f"(max_blobs={max_blobs}, features_per_blob={features_per_blob})"
    )
    return X_blob



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
    n_cols: int = 4,
    title_prefix: str = "Valley-based blob overlays",
):
    """
    Visualize valley-blobs directly on ERSPs.

    Uses the SAME segmentation (segment_valley_blobs) as the features.
    Everything is plotted in index units (freq/time bins), so it lines up
    with imshow(..., origin='lower').
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
            vmin=-5,
            vmax=5,
            interpolation="nearest",
        )

        ax.set_title(
            f"{meta['patient_id']} {meta['electrode']} {meta['condition']}",
            fontsize=7,
        )
        ax.set_xlabel("Time (bins)", fontsize=6)
        ax.set_ylabel("Freq (bins)", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)

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
        f"{title_prefix}\n(thr_pos={thr_pos}, thr_neg={thr_neg}, Δvalley={delta_valley})",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ------------------------------------------------------------
# 4. Metric in blob-feature space (with per-feature-type weights)
# ------------------------------------------------------------

BLOB_FEATURES_PER = 8   # default; overwritten by init_blob_metric if needed
BLOB_MAX_COUNT    = 4
BLOB_VEC_DIM      = BLOB_FEATURES_PER * BLOB_MAX_COUNT
BLOB_WEIGHT_VEC   = None  # 1D vector of length BLOB_VEC_DIM

# Canonical feature order inside each blob
# (we now have 8: includes mean and area_norm)
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
    features_per_blob: int = 7,
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
        e.g. 7  -> [t_start, f_peak, t_peak, sf, st, cov, mean_signed]
             6  -> [t_start, f_peak, t_peak, sf, st, cov]  (no mean_signed)
    feature_type_weights : dict or None
        Mapping from feature name -> scalar weight, e.g.:
          {
            "t_start": 1.5,
            "f_peak":  1.5,
            "t_peak":  1.5,
            "sf":      1.2,
            "st":      1.2,
            "cov":     1.0,
            "mean_signed": 0.6,
          }
        Any feature not in the dict defaults to weight 1.0.
    """
    global BLOB_FEATURES_PER, BLOB_MAX_COUNT, BLOB_VEC_DIM, BLOB_WEIGHT_VEC

    BLOB_FEATURES_PER = int(features_per_blob)
    BLOB_MAX_COUNT    = int(max_blobs)
    BLOB_VEC_DIM      = BLOB_FEATURES_PER * BLOB_MAX_COUNT

    if feature_type_weights is None:
        feature_type_weights = {}

    # Build weights for ONE blob
    per_blob_weights = []
    for j in range(BLOB_FEATURES_PER):
        feat_name = FEATURE_NAME_ORDER[j]  # only first features_per_blob entries used
        w = float(feature_type_weights.get(feat_name, 1.0))
        per_blob_weights.append(w)
    per_blob_weights = np.array(per_blob_weights, dtype=np.float32)

    # Tile across all blobs -> final weight vector
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
        # Apply weights elementwise
        if diff.shape != BLOB_WEIGHT_VEC.shape:
            raise ValueError(
                f"ersp_blob_metric: diff.shape={diff.shape} != "
                f"BLOB_WEIGHT_VEC.shape={BLOB_WEIGHT_VEC.shape}"
            )
        diff = diff * BLOB_WEIGHT_VEC

    return float(np.sqrt(np.dot(diff, diff)))
