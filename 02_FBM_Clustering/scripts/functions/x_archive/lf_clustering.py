



"""
lf_clustering.py

Utilities for:
- Loading ERSPs + metadata
- UMAP embeddings with patient/condition styling + optional ERSP thumbnails
- Cluster-card plots (condition stripes + mean ERSP)
- Cluster cards with blob-feature overlays
- Cluster membership saving
- Small helpers for selecting best K by silhouette
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import umap
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# 0. Loading ERSPs
# ------------------------------------------------------------

def parse_electrode_from_filename(fname: str) -> str:
    """
    Extract electrode name from ERSP .npy filename.

    Examples
    --------
    PAT_3301_picture_None_ERSP_AG2_TN.npy   -> 'AG2'
    EL035_reading_WM_ERSP_A_R10_TN.npy      -> 'A_R10'
    EL035_reading_WM_ERSP_Fp2_TN.npy        -> 'Fp2'
    EL030_audio_WM_ERSP_aH_L1_TN.npy        -> 'aH_L1'

    Assumes pattern: ..._ERSP_<electrode>_<suffix>.npy
    where <electrode> may contain underscores or dashes.
    """
    from pathlib import Path

    name = Path(fname).name
    if "_ERSP_" in name:
        _, right = name.split("_ERSP_", 1)
        # right like "A_R10_TN.npy" or "AG2_TN.npy"
        right_no_ext = right.rsplit(".", 1)[0]
        parts = right_no_ext.split("_")

        if len(parts) == 1:
            # No suffix, take whole thing as electrode
            electrode = right_no_ext
        else:
            # Last part is suffix (e.g. TN), rest is electrode name
            electrode = "_".join(parts[:-1])
        return electrode
    else:
        # fallback: just strip extension
        return name.rsplit(".", 1)[0]


def load_ersp_all(
    input_dir: Path,
    task: str = "LM",
    allowed_conditions=("audio", "picture", "reading"),
    n_freq: int = 129,
    n_time: int = 300,
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Load ERSP matrices + metadata for ALL patients and ALL allowed conditions.

    Folder structure expected:
        input_dir/
            PAT_3301/
                LM/
                    ERSP_matrix/
                        audio/*.npy
                        picture/*.npy
                        reading/*.npy
            EL030/
                LM/
                    ERSP_matrix/
                        ...

    Parameters
    ----------
    input_dir : Path
        Root folder of '04_ersp_LM_RAWONLY' outputs (contains patient folders).
    task : str
        Task name ('LM', etc.) used in the path under each patient.
    allowed_conditions : iterable of str
        Which condition folder names to include (e.g. audio/picture/reading).
    n_freq : int
        Expected frequency dimension of ERSP arrays.
    n_time : int
        Expected time dimension of ERSP arrays.

    Returns
    -------
    df_meta : pd.DataFrame
        One row per ERSP sample, with columns:
        ['sample_idx', 'patient_id', 'condition', 'task', 'electrode', 'file_path']
    ersp_list : list of np.ndarray
        Each array shape (n_freq, n_time), aligned with df_meta rows.
    """
    input_dir = Path(input_dir)
    allowed_conditions = set(allowed_conditions)

    # Auto-detect patient IDs as subfolders
    patient_ids = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])

    print("Detected patient folders:")
    for p in patient_ids:
        print("  -", p)
    print("\nTotal patients detected:", len(patient_ids))

    ersp_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []

    for pat in patient_ids:
        patient_dir = input_dir / pat / task / "ERSP_matrix"
        if not patient_dir.exists():
            print(f"[WARN] Missing ERSP_matrix for {pat}: {patient_dir}")
            continue

        # condition subfolders under ERSP_matrix
        condition_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        if not condition_dirs:
            print(f"[WARN] No condition subfolders for {pat} in {patient_dir}")
            continue

        cond_names = [d.name for d in condition_dirs]
        print(f"\nPatient {pat} — conditions found:", cond_names)

        for cond_dir in condition_dirs:
            cond_name = cond_dir.name
            if cond_name not in allowed_conditions:
                continue

            npy_files = sorted(cond_dir.glob("*.npy"))
            if not npy_files:
                print(f"[WARN] No .npy files in {cond_dir}")
                continue

            print(f"  Loading {len(npy_files)} files from condition '{cond_name}'")

            for fpath in npy_files:
                arr = np.load(fpath)

                if arr.shape != (n_freq, n_time):
                    print(f"  [WARN] Incorrect shape {arr.shape} in {fpath}, skipping.")
                    continue

                ersp_list.append(arr)

                meta_rows.append(
                    {
                        "patient_id": pat,
                        "condition": cond_name,
                        "task": task,
                        "electrode": parse_electrode_from_filename(fpath.name),
                        "file_path": str(fpath),
                    }
                )

    df_meta = pd.DataFrame(meta_rows)
    df_meta.index.name = "sample_idx"
    df_meta = df_meta.reset_index(drop=False)

    print("\n=== Finished Loading ===")
    print("Total ERSP samples loaded:", len(ersp_list))
    print("  df_meta rows:", len(df_meta))
    print("  ersp_list len:", len(ersp_list))

    return df_meta, ersp_list


# ------------------------------------------------------------
# 1. ERSP thumbnail maker
# ------------------------------------------------------------

def make_ersp_thumbnail(
    arr: np.ndarray,
    vmin: float = -10.0,
    vmax: float = 10.0,
) -> np.ndarray:
    """
    Simple clipping-based thumbnail for ERSPs.
    """
    arr = np.nan_to_num(np.array(arr, copy=True), nan=0.0)
    arr_clipped = np.clip(arr, vmin, vmax)
    return arr_clipped


# ------------------------------------------------------------
# 2. UMAP embedding + scatter with patient/condition styling
# ------------------------------------------------------------

def run_umap_and_plot(
    X: np.ndarray,
    df_meta: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric="euclidean",
    random_state: int = 42,
    title_prefix: str = "UMAP",
    ersp_list: Optional[List[np.ndarray]] = None,
    n_step: Optional[int] = 10,
    max_thumbs: int = 200,
    thumb_zoom: float = 0.15,
    save_path: Optional[Path] = None,
):
    """
    Run UMAP on X and create a patient/condition-colored scatter plot.

    - Points are colored by 'patient_id'.
    - Edgecolor encodes 'condition' (audio/picture/reading).
    - Optionally overlays ERSP thumbnails every `n_step` points.

    Returns
    -------
    embedding : np.ndarray
        Low-dimensional embedding (n_samples, n_components).
    umap_model : umap.UMAP
        Fitted UMAP model.
    """
    # Factorize patient + condition
    patient_codes, patient_uniques = pd.factorize(df_meta["patient_id"])
    cond_codes, cond_uniques = pd.factorize(df_meta["condition"])

    cond_edge_map = {
        "audio": "black",
        "picture": "white",
        "reading": "gray",
    }
    edgecolors = df_meta["condition"].map(cond_edge_map).fillna("black").to_numpy()

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )

    embedding = umap_model.fit_transform(X)
    print(
        f"UMAP embedding done: shape={embedding.shape}, "
        f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    )

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(8, 8))

        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=patient_codes,
            s=30,
            alpha=0.8,
            cmap="tab20",
            edgecolors=edgecolors,
            linewidths=0.7,
        )

        ax.set_title(
            f"{title_prefix}\n"
            f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}",
            fontsize=12,
        )
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)

        # --- Patient legend (color) ---
        cmap_obj = sc.cmap
        norm = plt.Normalize(vmin=patient_codes.min(), vmax=patient_codes.max())

        patient_handles = []
        patient_labels = []
        for code, pid in enumerate(patient_uniques):
            patient_handles.append(
                plt.Line2D(
                    [], [],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=cmap_obj(norm(code)),
                    markeredgecolor="black",
                    linewidth=0,
                )
            )
            patient_labels.append(pid)

        leg_pat = ax.legend(
            patient_handles,
            patient_labels,
            title="patient_id",
            fontsize=8,
            loc="best",
        )

        # --- Condition legend (edgecolor) ---
        cond_handles = []
        cond_labels = []
        for cond_name in cond_uniques:
            cond_handles.append(
                plt.Line2D(
                    [], [],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor="lightgray",
                    markeredgecolor=cond_edge_map.get(cond_name, "black"),
                    linewidth=1.0,
                )
            )
            cond_labels.append(cond_name)

        leg_cond = ax.legend(
            cond_handles,
            cond_labels,
            title="condition (edge)",
            fontsize=8,
            loc="upper right",
        )
        ax.add_artist(leg_pat)

        # --- Optional ERSP thumbnails ---
        if ersp_list is not None and n_step is not None and n_step > 0:
            idxs = list(range(0, len(embedding), n_step))
            if len(idxs) > max_thumbs:
                idxs = idxs[:max_thumbs]

            for i in idxs:
                img = make_ersp_thumbnail(ersp_list[i])
                oi = OffsetImage(img, cmap="bwr", zoom=thumb_zoom)
                ab = AnnotationBbox(
                    oi,
                    (embedding[i, 0], embedding[i, 1]),
                    frameon=False,
                    pad=0.0,
                )
                ax.add_artist(ab)

        plt.tight_layout()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print("Saved UMAP figure to:", save_path)
        plt.show()

    return embedding, umap_model


# ------------------------------------------------------------
# 3. Cluster cards (condition stripe + mean ERSP)
# ------------------------------------------------------------

def save_cluster_cards_generic(
    df_meta: pd.DataFrame,
    labels: np.ndarray,
    ersp_list: List[np.ndarray],
    algo_tag: str,
    param_tag: str,
    out_dir: Path,
    n_freq: int,
    n_time: int,
):
    """
    Make 'cluster cards':
      - Top: horizontal stripe with condition composition
      - Bottom: mean ERSP per cluster
    """
    if len(df_meta) != len(labels):
        raise ValueError(f"df_meta ({len(df_meta)}) != labels ({len(labels)})")
    if len(ersp_list) != len(labels):
        raise ValueError(f"ersp_list ({len(ersp_list)}) != labels ({len(labels)})")

    df = df_meta.copy()
    df["cluster_tmp"] = labels
    cluster_ids = sorted(np.unique(labels))
    if not cluster_ids:
        print(f"[{algo_tag} {param_tag}] No clusters to summarize.")
        return

    # Conditions + colors
    condition_order = sorted(df["condition"].unique())
    default_colors = ["#1f77b4","#ff7f0e","#2ca02c",
                      "#d62728","#9467bd","#8c564b"]
    condition_colors = {
        c: default_colors[i % len(default_colors)]
        for i, c in enumerate(condition_order)
    }

    # Stats + condition counts + mean ERSP
    cluster_sizes, el_counts, pat_counts = [], [], []
    cluster_cond_counts, cluster_means = {}, []

    for c in cluster_ids:
        df_c = df[df["cluster_tmp"] == c]
        cluster_sizes.append(len(df_c))

        patients = df_c["patient_id"].unique()
        el_counts.append(sum(str(p).startswith("EL")  for p in patients))
        pat_counts.append(sum(str(p).startswith("PAT") for p in patients))

        cond_counts = (
            df_c.groupby("condition")
               .size()
               .reindex(condition_order, fill_value=0)
        )
        cluster_cond_counts[c] = cond_counts

        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0:
            cluster_means.append(None)
        else:
            arr_c = np.stack([ersp_list[i] for i in idx_c], axis=0)
            cluster_means.append(arr_c.mean(axis=0))

    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0

    # Color scale across all means
    vals = [mc.ravel() for mc in cluster_means if mc is not None]
    if vals:
        vals = np.concatenate(vals)
        vmax = float(np.nanmax(np.abs(vals)))
        vmin = -vmax
    else:
        vmin = vmax = None

    # Layout
    n_clusters = len(cluster_ids)
    n_cols = min(4, n_clusters)
    n_card_rows = int(np.ceil(n_clusters / n_cols))
    n_grid_rows = n_card_rows * 2  # stripe + ERSP

    height_ratios = []
    for _ in range(n_card_rows):
        height_ratios.extend([0.25, 0.75])

    fig_width  = 1.4 * n_cols + 3.0
    fig_height = 2.0 * n_card_rows + 1.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=n_grid_rows,
        ncols=n_cols,
        height_ratios=height_ratios,
        figure=fig,
    )

    legends_done = False

    for idx, (c, mean_c, size_c, n_el, n_pat) in enumerate(
        zip(cluster_ids, cluster_means, cluster_sizes, el_counts, pat_counts)
    ):
        card_row  = idx // n_cols
        col       = idx % n_cols
        stripe_row = 2 * card_row
        ersp_row   = 2 * card_row + 1

        ax_stripe = fig.add_subplot(gs[stripe_row, col])
        ax_ersp   = fig.add_subplot(gs[ersp_row, col])

        # --- condition stripe ---
        cond_counts = cluster_cond_counts[c]
        total_c = cond_counts.sum()
        fractions = cond_counts / total_c if total_c > 0 else cond_counts * 0.0

        left = 0.0
        for cond in condition_order:
            frac = fractions.loc[cond]
            if frac <= 0:
                continue
            ax_stripe.barh(
                0,
                frac,
                left=left,
                color=condition_colors.get(cond, "gray"),
                edgecolor="none",
                height=1.0,
            )
            left += frac

        ax_stripe.set_xlim(0, 1)
        ax_stripe.set_ylim(-0.6, 0.6)
        ax_stripe.axis("off")

        # --- mean ERSP ---
        if mean_c is None:
            ax_ersp.set_title(f"C{c} (empty)", fontsize=7)
            ax_ersp.axis("off")
        else:
            ax_ersp.imshow(
                mean_c,
                aspect="auto",
                origin="lower",
                cmap="bwr",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax_ersp.set_title(
                f"C{c} (n={size_c}, max={max_cluster_size})\nEL:{n_el} | PAT:{n_pat}",
                fontsize=7,
            )
            ax_ersp.set_xlabel("Time", fontsize=6)
            ax_ersp.set_ylabel("Freq", fontsize=6)
            ax_ersp.tick_params(axis="both", labelsize=5)

        # one legend for conditions
        if not legends_done:
            handles, labels_ = [], []
            for cond in condition_order:
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="s",
                        linestyle="",
                        markersize=5,
                        color=condition_colors.get(cond, "gray"),
                    )
                )
                labels_.append(cond)
            ax_ersp.legend(
                handles, labels_,
                title="Condition",
                fontsize=5,
                loc="upper right",
            )
            legends_done = True

    fig.suptitle(f"{algo_tag} cluster cards – {param_tag}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{algo_tag}_{param_tag}_cluster_cards.png"
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{algo_tag} {param_tag}] Saved cluster cards to:", out_path)


# ------------------------------------------------------------
# 4. Cluster cards WITH blob-feature overlays + membership .npy
# ------------------------------------------------------------

def save_cluster_members_npy(
    df_meta: pd.DataFrame,
    labels: np.ndarray,
    out_path: Path,
):
    """
    Save, for each cluster, the list of original ERSP entries.

    Structure:
        {
          cluster_id_0: [
             {
               "sample_idx": int,
               "patient_id": str,
               "electrode": str,
               "condition": str,
               "file_path": str,
             },
             ...
          ],
          ...
        }
    """
    cluster_ids = np.unique(labels)
    cluster_members = {}

    for c in cluster_ids:
        mask = (labels == c)
        subset = df_meta.loc[mask, ["sample_idx", "patient_id",
                                    "electrode", "condition", "file_path"]]
        cluster_members[int(c)] = subset.to_dict(orient="records")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cluster_members)
    print(f"[cluster_members] Saved membership .npy to: {out_path}")


# ------------------------------------------------------------
# 5. Small helpers (best-K selection)
# ------------------------------------------------------------

def select_best_k_by_silhouette(
    results_list: List[dict],
    k_key: str = "K",
    sil_key: str = "silhouette",
    k_min: int = 9,
    k_max: int = 19,
):
    """
    Given a list of dicts with keys [k_key, sil_key], returns
    (best_K, best_sil) restricted to k_min <= K <= k_max.
    """
    candidates = [
        (r[k_key], r.get(sil_key, np.nan))
        for r in results_list
        if (k_min <= r[k_key] <= k_max and not np.isnan(r.get(sil_key, np.nan)))
    ]
    if not candidates:
        return None, None
    best_K, best_sil = max(candidates, key=lambda t: t[1])
    return best_K, best_sil





# ------------------------------------------------------------
# 5. Run metadata helper (JSON per result folder)
# ------------------------------------------------------------

import json
from datetime import datetime

def save_run_metadata(
    out_dir: Path,
    script_name: str,
    algo_tag: str,
    run_id: str,
    extra: dict | None = None,
):
    """
    Save a small JSON file in `out_dir` describing this run.

    Parameters
    ----------
    out_dir : Path
        Folder where results (figures, npy, etc.) are stored.
    script_name : str
        Name of the notebook/script (e.g. '210_clustering.ipynb').
    algo_tag : str
        Short label for the algorithm / configuration.
    run_id : str
        Timestamp-like ID used in filenames.
    extra : dict, optional
        Any extra metadata you want (thresholds, K range, feature config, etc.).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "script_name": script_name,
        "algo_tag": algo_tag,
        "run_id": run_id,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra is not None:
        # Shallow merge; driver decides what goes into `extra`
        meta.update(extra)

    meta_path = out_dir / f"run_metadata_{algo_tag}_{run_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved run metadata to:", meta_path)


    
    
import numpy as np
import matplotlib.pyplot as plt

def plot_blob_features_from_X(
    ersp_list,
    df_meta,
    X_blob,
    indices,
    max_blobs=4,
    features_per_blob=7,
    vmin=-5,
    vmax=5,
    title_prefix="Blob feature overlays from X_blob (bin axes)",
):
    """
    Overlay precomputed blob features (from X_blob_full) on top of ERSPs.

    Assumes X_blob rows encode, per sample:

        [t_start, f_peak, t_peak, sf, st, cov, mean_signed] × max_blobs

    where all geometric features are in NORMALIZED [0..1] units,
    and ERSPs are plotted in *bin* coordinates:

        x-axis: time bins   -> 0 ... n_time-1
        y-axis: freq bins   -> 0 ... n_freq-1
    """

    if len(indices) == 0:
        print("No indices provided to plot_blob_features_from_X.")
        return
    print("this didnt print")
    n_samples = len(indices)
    n_cols = min(4, n_samples)
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, idx in zip(axes_flat, indices):
        arr = ersp_list[idx]
        meta = df_meta.iloc[idx]

        n_freq, n_time = arr.shape

        # --- Base ERSP in BIN space: x=0..n_time, y=0..n_freq ---
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            cmap="bwr",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=[0, n_time, 0, n_freq],  # explicit bin extents
        )

        ax.set_title(
            f"{meta['patient_id']} {meta['electrode']} {meta['condition']}",
            fontsize=7,
        )
        ax.set_xlabel("Time (bins)", fontsize=6)
        ax.set_ylabel("Freq (bins)", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)

        # --- Features for this sample ---
        row = X_blob[idx]
        expected_len = max_blobs * features_per_blob
        if row.size != expected_len:
            print(
                f"[WARN] X_blob row {idx} has size {row.size}, "
                f"expected {expected_len}. Skipping overlays."
            )
            continue

        blob_feats = row.reshape(max_blobs, features_per_blob)

        # Conversion: normalized [0,1] -> bin indices
        def t_norm_to_bin(t_norm):
            return float(t_norm * (n_time - 1))

        def f_norm_to_bin(f_norm):
            return float(f_norm * (n_freq - 1))

        for b in range(max_blobs):
            t_start_norm = blob_feats[b, 0]
            f_peak_norm  = blob_feats[b, 1]
            t_peak_norm  = blob_feats[b, 2]
            sf_norm      = blob_feats[b, 3]
            st_norm      = blob_feats[b, 4]
            cov_val      = blob_feats[b, 5]
            mean_signed  = blob_feats[b, 6] if features_per_blob >= 7 else 0.0

            # If blob slot is all zeros, skip
            if np.allclose(blob_feats[b, :], 0.0):
                continue

            # --- Convert to BIN indices ---
            t_start_bin = t_norm_to_bin(t_start_norm)
            t_peak_bin  = t_norm_to_bin(t_peak_norm)
            f_peak_bin  = f_norm_to_bin(f_peak_norm)

            sf_bin = sf_norm * (n_freq - 1)
            st_bin = st_norm * (n_time - 1)

            # Choose color based on sign
            if mean_signed > 0:
                color = "yellow"
            elif mean_signed < 0:
                color = "cyan"
            else:
                color = "white"

            # --- Peak marker (in bins) ---
            ax.scatter(
                t_peak_bin,
                f_peak_bin,
                marker="o",
                s=20,
                c=color,
                edgecolor="black",
                linewidths=0.5,
                zorder=3,
            )

            # --- Onset (t_start) vertical line (bin) ---
            ax.axvline(
                x=t_start_bin,
                color=color,
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
            )

            # --- Vertical extent: freq bins [f_peak ± 2*sf_bin] ---
            f_low = max(0, f_peak_bin - 2 * sf_bin)
            f_high = min(n_freq - 1, f_peak_bin + 2 * sf_bin)
            ax.plot(
                [t_peak_bin, t_peak_bin],
                [f_low, f_high],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )

            # --- Horizontal extent: time bins [t_peak ± 2*st_bin] ---
            t_low = max(0, t_peak_bin - 2 * st_bin)
            t_high = min(n_time - 1, t_peak_bin + 2 * st_bin)
            ax.plot(
                [t_low, t_high],
                [f_peak_bin, f_peak_bin],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )
    print("\nt_start_bin ",t_start_bin," - "," t_peak_bin",t_peak_bin," - ","st_bin ",st_bin)
    # Turn off unused axes
    for ax in axes_flat[n_samples:]:
        ax.axis("off")

    fig.suptitle(title_prefix, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def save_cluster_cards_with_blob_features(
    df_meta: pd.DataFrame,
    labels: np.ndarray,
    ersp_list: List[np.ndarray],
    X_blob: np.ndarray,
    algo_tag: str,
    param_tag: str,
    out_dir: Path,
    n_freq: int,
    n_time: int,
    max_blobs: int,
    features_per_blob: int,
):
    """
    Cluster cards = condition stripe + mean ERSP, plus:
    overlays of *average blob peak positions* per cluster.

    X_blob is interpreted as
      [t_start, f_peak, t_peak, sf, st, cov, mean_signed] × max_blobs,
    where t_start, f_peak, t_peak are in [0, 1] (normalized).
    """
    if len(df_meta) != len(labels):
        raise ValueError(f"df_meta ({len(df_meta)}) != labels ({len(labels)})")
    if len(ersp_list) != len(labels):
        raise ValueError(f"ersp_list ({len(ersp_list)}) != labels ({len(labels)})")
    if len(X_blob) != len(labels):
        raise ValueError(f"X_blob ({len(X_blob)}) != labels ({len(labels)})")

    df = df_meta.copy()
    df["cluster_tmp"] = labels
    cluster_ids = sorted(np.unique(labels))
    if not cluster_ids:
        print(f"[{algo_tag} {param_tag}] No clusters to summarize.")
        return

    condition_order = sorted(df["condition"].unique())
    default_colors = ["#1f77b4","#ff7f0e","#2ca02c",
                      "#d62728","#9467bd","#8c564b"]
    condition_colors = {
        c: default_colors[i % len(default_colors)]
        for i, c in enumerate(condition_order)
    }

    cluster_sizes, el_counts, pat_counts = [], [], []
    cluster_cond_counts, cluster_means = {}, []
    cluster_blob_means = {}

    for c in cluster_ids:
        mask_c = (labels == c)
        df_c   = df.loc[mask_c]
        X_c    = X_blob[mask_c, :]

        cluster_sizes.append(len(df_c))

        patients = df_c["patient_id"].unique()
        el_counts.append(sum(str(p).startswith("EL")  for p in patients))
        pat_counts.append(sum(str(p).startswith("PAT") for p in patients))

        cond_counts = (
            df_c.groupby("condition")
               .size()
               .reindex(condition_order, fill_value=0)
        )
        cluster_cond_counts[c] = cond_counts

        idx_c = np.where(mask_c)[0]
        if len(idx_c) == 0:
            cluster_means.append(None)
        else:
            arr_c = np.stack([ersp_list[i] for i in idx_c], axis=0)
            cluster_means.append(arr_c.mean(axis=0))

        if X_c.size > 0:
            mean_feat = X_c.mean(axis=0)
            # reshape to (max_blobs, features_per_blob)
            cluster_blob_means[c] = mean_feat.reshape(max_blobs, features_per_blob)
        else:
            cluster_blob_means[c] = np.zeros((max_blobs, features_per_blob), dtype=np.float32)

    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0

    vals = [mc.ravel() for mc in cluster_means if mc is not None]
    if vals:
        vals = np.concatenate(vals)
        vmax = float(np.nanmax(np.abs(vals)))
        vmin = -vmax
    else:
        vmin = vmax = None

    n_clusters = len(cluster_ids)
    n_cols = min(4, n_clusters)
    n_card_rows = int(np.ceil(n_clusters / n_cols))
    n_grid_rows = n_card_rows * 2

    height_ratios = []
    for _ in range(n_card_rows):
        height_ratios.extend([0.25, 0.75])

    fig_width  = 1.4 * n_cols + 3.0
    fig_height = 2.0 * n_card_rows + 1.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=n_grid_rows,
        ncols=n_cols,
        height_ratios=height_ratios,
        figure=fig,
    )

    legends_done = False

    for idx, (c, mean_c, size_c, n_el, n_pat) in enumerate(
        zip(cluster_ids, cluster_means, cluster_sizes, el_counts, pat_counts)
    ):
        card_row  = idx // n_cols
        col       = idx % n_cols
        stripe_row = 2 * card_row
        ersp_row   = 2 * card_row + 1

        ax_stripe = fig.add_subplot(gs[stripe_row, col])
        ax_ersp   = fig.add_subplot(gs[ersp_row, col])

        # --- condition stripe ---
        cond_counts = cluster_cond_counts[c]
        total_c = cond_counts.sum()
        fractions = cond_counts / total_c if total_c > 0 else cond_counts * 0.0

        left = 0.0
        for cond in condition_order:
            frac = fractions.loc[cond]
            if frac <= 0:
                continue
            ax_stripe.barh(
                0,
                frac,
                left=left,
                color=condition_colors.get(cond, "gray"),
                edgecolor="none",
                height=1.0,
            )
            left += frac

        ax_stripe.set_xlim(0, 1)
        ax_stripe.set_ylim(-0.6, 0.6)
        ax_stripe.axis("off")

        # --- mean ERSP + blob overlays ---
        if mean_c is None:
            ax_ersp.set_title(f"C{c} (empty)", fontsize=7)
            ax_ersp.axis("off")
        else:
            ax_ersp.imshow(
                mean_c,
                aspect="auto",
                origin="lower",
                cmap="bwr",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax_ersp.set_title(
                f"C{c} (n={size_c}, max={max_cluster_size})\nEL:{n_el} | PAT:{n_pat}",
                fontsize=7,
            )
            ax_ersp.set_xlabel("Time (samples)", fontsize=6)
            ax_ersp.set_ylabel("Freq (samples)", fontsize=6)
            ax_ersp.tick_params(axis="both", labelsize=5)

            blob_means = cluster_blob_means[c]

            # blob_means[b] = [t_start, f_peak, t_peak, sf, st, cov, mean_signed]
            for b in range(max_blobs):
                t_start, f_peak, t_peak, sf, st, cov, mean_signed = blob_means[b, :]

                # skip empty blob slots (all zeros)
                if np.allclose(blob_means[b, :], 0.0):
                    continue

                # ---- convert normalized coords -> index space ----
                # t_peak, f_peak are in [0, 1]
                t_idx = t_peak * (n_time - 1)
                f_idx = f_peak * (n_freq - 1)

                # (optionally also t_start -> index for vertical markers)
                # t_start_idx = t_start * (n_time - 1)

                if mean_signed > 0:
                    color = "yellow"
                elif mean_signed < 0:
                    color = "cyan"
                else:
                    color = "white"

                size = 20 + 40 * min(1.0, abs(mean_signed) / 5.0)

                ax_ersp.scatter(
                    t_idx, f_idx,
                    marker="s",
                    s=size,
                    c=color,
                    edgecolor="black",
                    linewidths=0.5,
                    zorder=3,
                )

        if not legends_done:
            handles, labels_ = [], []
            for cond in condition_order:
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="s",
                        linestyle="",
                        markersize=5,
                        color=condition_colors.get(cond, "gray"),
                    )
                )
                labels_.append(cond)
            ax_ersp.legend(
                handles, labels_,
                title="Condition",
                fontsize=5,
                loc="upper right",
            )
            legends_done = True

    fig.suptitle(f"{algo_tag} cluster cards – {param_tag}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{algo_tag}_{param_tag}_cluster_cards.png"
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{algo_tag} {param_tag}] Saved cluster cards to:", out_path)


    


