



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

def s71_save_gated_blob_run(
    *,
    base_runs_dir,
    X_blob_keep,
    df_keep,
    keep_idx,
    drop_idx,
    score_thr,
    blob_score_quantile,
    valley_params,
    # Optional:
    ersp_keep=None,
    max_scores=None,
    feature_schema=None,
    run_metadata_extra=None,
    run_id=None,
    verbose: bool = True,
):
    """
    Save a gated blob run to a unique run folder.

    Saves:
      - X_blob_keep.npy
      - df_keep.parquet
      - keep_idx.npy, drop_idx.npy
      - gating_info.json
      - config_blob.json
      - feature_schema.json (optional)
      - max_scores_full.npy, max_scores_keep.npy (optional)
      - ersp_keep.npy (preferred, stacked) or ersp_keep.npz (fallback) (optional)
      - run_metadata.json

    Returns:
      run_dir (Path)
    """
    import os, json, platform, getpass, subprocess
    from pathlib import Path
    from datetime import datetime
    import numpy as np
    import pandas as pd

    base_runs_dir = Path(base_runs_dir)

    # Create a unique run directory automatically
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_runs_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    # -------------------------
    # Helpers
    # -------------------------
    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    def _safe_git_info():
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            commit = None
        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.STDOUT).decode().strip()
            dirty = bool(status)
        except Exception:
            dirty = None
        return {"git_commit": commit, "git_dirty": dirty}

    def _save_json(path: Path, obj: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_json_default)

    def _try_stack_ersps(ersp_list):
        """
        Try to stack list of (nF,nT) into (n,nF,nT). If shapes differ, return None.
        """
        if not isinstance(ersp_list, (list, tuple)) or len(ersp_list) == 0:
            return None
        shapes = [np.asarray(e).shape for e in ersp_list]
        if len(set(shapes)) != 1:
            return None
        return np.stack([np.asarray(e, dtype=np.float32) for e in ersp_list], axis=0)

    # -------------------------
    # Core filenames
    # -------------------------
    f_X_keep       = run_dir / "X_blob_keep.npy"
    f_df_keep      = run_dir / "df_keep.parquet"
    f_keep_idx     = run_dir / "keep_idx.npy"
    f_drop_idx     = run_dir / "drop_idx.npy"
    f_gate_info    = run_dir / "gating_info.json"
    f_config       = run_dir / "config_blob.json"
    f_schema       = run_dir / "feature_schema.json"        # optional
    f_scores_full  = run_dir / "max_scores_full.npy"        # optional
    f_scores_keep  = run_dir / "max_scores_keep.npy"        # optional
    f_run_metadata = run_dir / "run_metadata.json"

    # NEW: ERSP artifacts
    f_ersp_keep_npy = run_dir / "ersp_keep.npy"             # preferred: (n,nF,nT)
    f_ersp_keep_npz = run_dir / "ersp_keep.npz"             # fallback

    # -------------------------
    # Save core artifacts
    # -------------------------
    np.save(f_X_keep, np.asarray(X_blob_keep))
    df_keep.to_parquet(f_df_keep, index=False)

    keep_idx = np.asarray(keep_idx, dtype=int)
    drop_idx = np.asarray(drop_idx, dtype=int)
    np.save(f_keep_idx, keep_idx)
    np.save(f_drop_idx, drop_idx)

    gating_info = {
        "quantile": float(blob_score_quantile),
        "score_threshold": float(score_thr),
        "n_kept": int(len(keep_idx)),
        "n_dropped": int(len(drop_idx)),
    }
    _save_json(f_gate_info, gating_info)

    # Snapshot of the blob config used
    _save_json(f_config, dict(valley_params))

    # Optional: feature schema
    if feature_schema is not None:
        _save_json(f_schema, feature_schema)

    # Optional: max_scores (full + kept)
    if max_scores is not None:
        max_scores = np.asarray(max_scores)
        np.save(f_scores_full, max_scores)
        np.save(f_scores_keep, max_scores[keep_idx])

    # -------------------------
    # Optional: Save ersp_keep (aligned with df_keep / X_blob_keep)
    # -------------------------
    ersp_saved = False
    ersp_format = None
    ersp_shape = None

    if ersp_keep is not None:
        try:
            stacked = _try_stack_ersps(ersp_keep)
            if stacked is not None:
                np.save(f_ersp_keep_npy, stacked.astype(np.float32, copy=False))
                ersp_saved = True
                ersp_format = "npy_stacked"
                ersp_shape = list(map(int, stacked.shape))
            else:
                arr_dict = {f"ersp_{i:06d}": np.asarray(e, dtype=np.float32) for i, e in enumerate(ersp_keep)}
                np.savez_compressed(f_ersp_keep_npz, **arr_dict)
                ersp_saved = True
                ersp_format = "npz_per_sample"
                ersp_shape = {"n_samples": int(len(ersp_keep))}
        except Exception as e:
            if verbose:
                print("[WARN] Failed to save ersp_keep:", repr(e))

    # -------------------------
    # Run provenance metadata
    # -------------------------
    run_metadata = {
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "user": getpass.getuser(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),

        "X_blob_keep_shape": list(map(int, np.asarray(X_blob_keep).shape)),
        "df_keep_rows": int(len(df_keep)),
        "df_keep_cols": int(df_keep.shape[1]),

        "gating": gating_info,
        "valley_params": dict(valley_params),

        "ersp_keep_saved": bool(ersp_saved),
        "ersp_keep_format": ersp_format,
        "ersp_keep_shape": ersp_shape,

        **_safe_git_info(),
    }

    if run_metadata_extra is not None:
        run_metadata.update(run_metadata_extra)

    _save_json(f_run_metadata, run_metadata)

    if verbose:
        n_files = len([p for p in run_dir.glob("*") if p.is_file()])
        print("Saved gated blob run to:")
        print(run_dir)
        print("Files:", n_files)
        if ersp_saved:
            print("ERSP saved as:", ersp_format, "| shape:", ersp_shape)
        else:
            print("ERSP not saved (ersp_keep missing or save failed).")

    return run_dir


# ------------------------------------------------------------
# Poster plotting utilities (transparent + consistent axes/fonts)
# ------------------------------------------------------------


POSTER_CMAP = "bwr"
POSTER_VMIN = -6.0
POSTER_VMAX = 6.0
POSTER_FMAX_HZ = 500.0
POSTER_VLINE_PCT = 50.0

def set_poster_style(
    *,
    font_family: str = "DejaVu Sans",
    base_fontsize: int = 10,
    title_fontsize: int = 11,
    label_fontsize: int = 10,
    tick_fontsize: int = 9,
    linewidth: float = 1.0,
):
    """
    Call once at notebook start.
    Enforces consistent typography and transparent backgrounds for all saves.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        # Typography
        "font.family": font_family,
        "font.size": base_fontsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": tick_fontsize,
        "lines.linewidth": linewidth,

        # Transparent by default
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
        "savefig.transparent": True,

        # Clean layout defaults
        "figure.autolayout": False,
    })


def _poster_extent(n_freq: int, n_time: int, fmax_hz: float) -> list[float]:
    # Map pixel index space -> [% time, Hz]
    # x: 0..100 (%), y: 0..fmax_hz
    return [0.0, 100.0, 0.0, float(fmax_hz)]


def _poster_add_vline(ax, vline_pct: float = POSTER_VLINE_PCT):
    ax.axvline(float(vline_pct), color="gray", linewidth=1.0, alpha=0.8)


def plot_ersp_poster(
    ax,
    ersp,
    *,
    fmax_hz: float = POSTER_FMAX_HZ,
    vmin: float = POSTER_VMIN,
    vmax: float = POSTER_VMAX,
    cmap: str = POSTER_CMAP,
    add_vline: bool = True,
    vline_pct: float = POSTER_VLINE_PCT,
    title: str | None = None,
    show_axes: bool = True,
    interpolation: str = "nearest",
):
    """
    Standard ERSP imshow: x in % time, y in Hz, fixed bwr, fixed vmin/vmax, optional 50% line.
    """
    import numpy as np

    ersp = np.asarray(ersp)
    n_freq, n_time = ersp.shape
    extent = _poster_extent(n_freq, n_time, fmax_hz)

    im = ax.imshow(
        ersp,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        extent=extent,
    )

    if title is not None:
        ax.set_title(title)

    if show_axes:
        ax.set_xlabel("Time (% trial)")
        ax.set_ylabel("Frequency (Hz)")
    else:
        ax.set_axis_off()

    if add_vline:
        _poster_add_vline(ax, vline_pct=vline_pct)

    return im


def plot_map101_poster(
    ax,
    map101,
    *,
    fmax_hz: float = POSTER_FMAX_HZ,
    scale_to_ersp_range: bool = True,
    vmin: float = POSTER_VMIN,
    vmax: float = POSTER_VMAX,
    cmap: str = POSTER_CMAP,
    add_vline: bool = True,
    vline_pct: float = POSTER_VLINE_PCT,
    title: str | None = None,
    show_axes: bool = True,
):
    """
    -101 map plot, optionally scaled by 6 so it displays with the SAME vmin/vmax (-6..6).
    """
    import numpy as np
    arr = np.asarray(map101, dtype=float)
    if scale_to_ersp_range:
        arr = arr * (abs(vmax) if vmax is not None else 6.0)
    return plot_ersp_poster(
        ax, arr,
        fmax_hz=fmax_hz, vmin=vmin, vmax=vmax, cmap=cmap,
        add_vline=add_vline, vline_pct=vline_pct,
        title=title, show_axes=show_axes
    )


def savefig_poster(fig, out_path, *, dpi: int = 300):
    """
    Standard save: transparent background, tight bbox.
    """
    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)

# ------------------------------------------------------------
# 0. Loading ERSPs
# ------------------------------------------------------------

def _h_parse_electrode_from_filename(fname: str) -> str:
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


def s10_load_ersps(
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
                        "electrode": _h_parse_electrode_from_filename(fpath.name),
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

def _h_make_ersp_thumbnail(
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

def s40_umap_embed_and_plot(
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
        fig, ax = plt.subplots(figsize=(28, 28))

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
                    markersize=15,
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
                img = _h_make_ersp_thumbnail(ersp_list[i])
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

def _h_save_cluster_cards_generic(
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
    cluster_ids = sorted(np.unique(labels).tolist(), key=lambda c: (-np.sum(labels==c), int(c)))
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

    fig_width  = 1 * n_cols + 3.0
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
                        markersize=15,
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

def s70_save_cluster_members(
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

def s55_select_best_by_silhouette(
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

def s71_save_run_metadata(
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

def q61_plot_blob_features_from_X(
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
        print("No indices provided to q61_plot_blob_features_from_X.")
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


# def s60_save_cluster_cards_with_blob_features(
#     df_meta: pd.DataFrame,
#     labels: np.ndarray,
#     ersp_list: List[np.ndarray],
#     X_blob: np.ndarray,
#     algo_tag: str,
#     param_tag: str,
#     out_dir: Path,
#     n_freq: int,
#     n_time: int,
#     max_blobs: int,
#     features_per_blob: int,
# ):
#     """
#     Cluster cards = condition stripe + mean ERSP, plus:
#     overlays of *average blob peak positions* per cluster.

#     X_blob is interpreted as
#       [t_start, f_peak, t_peak, sf, st, cov, mean_signed] × max_blobs,
#     where t_start, f_peak, t_peak are in [0, 1] (normalized).
#     """
#     if len(df_meta) != len(labels):
#         raise ValueError(f"df_meta ({len(df_meta)}) != labels ({len(labels)})")
#     if len(ersp_list) != len(labels):
#         raise ValueError(f"ersp_list ({len(ersp_list)}) != labels ({len(labels)})")
#     if len(X_blob) != len(labels):
#         raise ValueError(f"X_blob ({len(X_blob)}) != labels ({len(labels)})")

#     df = df_meta.copy()
#     df["cluster_tmp"] = labels
#     cluster_ids = sorted(np.unique(labels))
#     if not cluster_ids:
#         print(f"[{algo_tag} {param_tag}] No clusters to summarize.")
#         return

#     condition_order = sorted(df["condition"].unique())
#     default_colors = ["#1f77b4","#ff7f0e","#2ca02c",
#                       "#d62728","#9467bd","#8c564b"]
#     condition_colors = {
#         c: default_colors[i % len(default_colors)]
#         for i, c in enumerate(condition_order)
#     }

#     cluster_sizes, el_counts, pat_counts = [], [], []
#     cluster_cond_counts, cluster_means = {}, []
#     cluster_blob_means = {}

#     for c in cluster_ids:
#         mask_c = (labels == c)
#         df_c   = df.loc[mask_c]
#         X_c    = X_blob[mask_c, :]

#         cluster_sizes.append(len(df_c))

#         patients = df_c["patient_id"].unique()
#         el_counts.append(sum(str(p).startswith("EL")  for p in patients))
#         pat_counts.append(sum(str(p).startswith("PAT") for p in patients))

#         cond_counts = (
#             df_c.groupby("condition")
#                .size()
#                .reindex(condition_order, fill_value=0)
#         )
#         cluster_cond_counts[c] = cond_counts

#         idx_c = np.where(mask_c)[0]
#         if len(idx_c) == 0:
#             cluster_means.append(None)
#         else:
#             arr_c = np.stack([ersp_list[i] for i in idx_c], axis=0)
#             cluster_means.append(arr_c.mean(axis=0))

#         if X_c.size > 0:
#             mean_feat = X_c.mean(axis=0)
#             # reshape to (max_blobs, features_per_blob)
#             cluster_blob_means[c] = mean_feat.reshape(max_blobs, features_per_blob)
#         else:
#             cluster_blob_means[c] = np.zeros((max_blobs, features_per_blob), dtype=np.float32)

#     max_cluster_size = max(cluster_sizes) if cluster_sizes else 0

#     vals = [mc.ravel() for mc in cluster_means if mc is not None]
#     if vals:
#         vals = np.concatenate(vals)
#         vmax = float(np.nanmax(np.abs(vals)))
#         vmin = -vmax
#     else:
#         vmin = vmax = None

#     n_clusters = len(cluster_ids)
#     n_cols = min(4, n_clusters)
#     n_card_rows = int(np.ceil(n_clusters / n_cols))
#     n_grid_rows = n_card_rows * 2

#     height_ratios = []
#     for _ in range(n_card_rows):
#         height_ratios.extend([0.25, 0.75])

#     fig_width  = 1.4 * n_cols + 3.0
#     fig_height = 2.0 * n_card_rows + 1.5

#     fig = plt.figure(figsize=(fig_width, fig_height))
#     gs = GridSpec(
#         nrows=n_grid_rows,
#         ncols=n_cols,
#         height_ratios=height_ratios,
#         figure=fig,
#     )

#     legends_done = False

#     for idx, (c, mean_c, size_c, n_el, n_pat) in enumerate(
#         zip(cluster_ids, cluster_means, cluster_sizes, el_counts, pat_counts)
#     ):
#         card_row  = idx // n_cols
#         col       = idx % n_cols
#         stripe_row = 2 * card_row
#         ersp_row   = 2 * card_row + 1

#         ax_stripe = fig.add_subplot(gs[stripe_row, col])
#         ax_ersp   = fig.add_subplot(gs[ersp_row, col])

#         # --- condition stripe ---
#         cond_counts = cluster_cond_counts[c]
#         total_c = cond_counts.sum()
#         fractions = cond_counts / total_c if total_c > 0 else cond_counts * 0.0

#         left = 0.0
#         for cond in condition_order:
#             frac = fractions.loc[cond]
#             if frac <= 0:
#                 continue
#             ax_stripe.barh(
#                 0,
#                 frac,
#                 left=left,
#                 color=condition_colors.get(cond, "gray"),
#                 edgecolor="none",
#                 height=1.0,
#             )
#             left += frac

#         ax_stripe.set_xlim(0, 1)
#         ax_stripe.set_ylim(-0.6, 0.6)
#         ax_stripe.axis("off")

#         # --- mean ERSP + blob overlays ---
#         if mean_c is None:
#             ax_ersp.set_title(f"C{c} (empty)", fontsize=7)
#             ax_ersp.axis("off")
#         else:
#             ax_ersp.imshow(
#                 mean_c,
#                 aspect="auto",
#                 origin="lower",
#                 cmap="bwr",
#                 vmin=vmin,
#                 vmax=vmax,
#                 interpolation="nearest",
#             )
#             ax_ersp.set_title(
#                 f"C{c} (n={size_c}, max={max_cluster_size})\nEL:{n_el} | PAT:{n_pat}",
#                 fontsize=7,
#             )
#             ax_ersp.set_xlabel("Time (samples)", fontsize=6)
#             ax_ersp.set_ylabel("Freq (samples)", fontsize=6)
#             ax_ersp.tick_params(axis="both", labelsize=5)

#             blob_means = cluster_blob_means[c]

#             # blob_means[b] = [t_start, f_peak, t_peak, sf, st, cov, mean_signed]
#             for b in range(max_blobs):
#                 t_start, f_peak, t_peak, sf, st, cov, mean_signed = blob_means[b, :]

#                 # skip empty blob slots (all zeros)
#                 if np.allclose(blob_means[b, :], 0.0):
#                     continue

#                 # ---- convert normalized coords -> index space ----
#                 # t_peak, f_peak are in [0, 1]
#                 t_idx = t_peak * (n_time - 1)
#                 f_idx = f_peak * (n_freq - 1)

#                 # (optionally also t_start -> index for vertical markers)
#                 # t_start_idx = t_start * (n_time - 1)

#                 if mean_signed > 0:
#                     color = "yellow"
#                 elif mean_signed < 0:
#                     color = "cyan"
#                 else:
#                     color = "white"

#                 size = 20 + 40 * min(1.0, abs(mean_signed) / 5.0)

#                 ax_ersp.scatter(
#                     t_idx, f_idx,
#                     marker="s",
#                     s=size,
#                     c=color,
#                     edgecolor="black",
#                     linewidths=0.5,
#                     zorder=3,
#                 )

#         if not legends_done:
#             handles, labels_ = [], []
#             for cond in condition_order:
#                 handles.append(
#                     plt.Line2D(
#                         [0], [0],
#                         marker="s",
#                         linestyle="",
#                         markersize=5,
#                         color=condition_colors.get(cond, "gray"),
#                     )
#                 )
#                 labels_.append(cond)
#             ax_ersp.legend(
#                 handles, labels_,
#                 title="Condition",
#                 fontsize=5,
#                 loc="upper right",
#             )
#             legends_done = True

#     fig.suptitle(f"{algo_tag} cluster cards – {param_tag}", fontsize=10)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])

#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_name = f"{algo_tag}_{param_tag}_cluster_cards.png"
#     out_path = out_dir / out_name
#     plt.savefig(out_path, dpi=150)
#     plt.close(fig)
#     print(f"[{algo_tag} {param_tag}] Saved cluster cards to:", out_path)

# =========================
# REPLACE your existing s60_save_cluster_cards_with_blob_features in lf_clustering.py
# =========================
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

def s60_save_cluster_cards_with_blob_features(
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
    *,
    fmax_hz: float = 500.0,
    vmin: float = -6.0,
    vmax: float = 6.0,
    dpi: int = 300,
    cluster_stat: str = "mean",          # "mean" or "median"
    show_blob_overlay: bool = True,
    overlay_top_n: Optional[int] = None,
    do_cond: bool = True,
    do_pat: bool = True,
    do_roi: bool = True,
    roi_col: str = "brain_region",
    pat_col: str = "patient_id",
    cond_col: str = "condition",
    atlas_dir: Optional[Path] = None,
    atlas_pattern: str = "cluster_{cid}.png",   # expects atlas_dir / atlas_pattern.format(cid=cluster_id)
):
    if len(df_meta) != len(labels):
        raise ValueError(f"df_meta ({len(df_meta)}) != labels ({len(labels)})")
    if len(ersp_list) != len(labels):
        raise ValueError(f"ersp_list ({len(ersp_list)}) != labels ({len(labels)})")
    if len(X_blob) != len(labels):
        raise ValueError(f"X_blob ({len(X_blob)}) != labels ({len(labels)})")

    df = df_meta.copy().reset_index(drop=True)
    labels = np.asarray(labels)

    cluster_ids = sorted(np.unique(labels).tolist(), key=lambda c: (-np.sum(labels==c), int(c)))
    if not cluster_ids:
        print(f"[{algo_tag} {param_tag}] No clusters.")
        return

    # Color maps from recon config
    if do_pat and pat_col in df.columns:
        pat_colors = _build_patient_color_map_css4(df[pat_col].astype(str).unique().tolist())
    else:
        pat_colors = {}

    if do_cond and cond_col in df.columns:
        cond_colors = _build_condition_color_map_css4(df[cond_col].astype(str).unique().tolist())
    else:
        cond_colors = {}

    if do_roi and (roi_col in df.columns):
        roi_colors = _roi_color_map(df[roi_col].astype(str).fillna("NA"))
    else:
        roi_colors = {}

    # Stats function
    cluster_stat = str(cluster_stat).strip().lower()
    if cluster_stat not in {"mean", "median"}:
        raise ValueError("cluster_stat must be 'mean' or 'median'")

    def _reduce_stack(arr3):
        return arr3.mean(axis=0) if cluster_stat == "mean" else np.median(arr3, axis=0)

    # Precompute cluster summaries
    cluster_sizes = {}
    cluster_ersp = {}
    cluster_blob_means = {}
    cluster_bar_counts = {}

    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        cluster_sizes[c] = int(len(idx))
        if len(idx) == 0:
            cluster_ersp[c] = None
            cluster_blob_means[c] = np.zeros((max_blobs, features_per_blob), dtype=np.float32)
            cluster_bar_counts[c] = {}
            continue

        arr = np.stack([ersp_list[i] for i in idx], axis=0)  # (n, F, T)
        cluster_ersp[c] = _reduce_stack(arr)

        mean_feat = X_blob[idx, :].mean(axis=0)
        cluster_blob_means[c] = mean_feat.reshape(max_blobs, features_per_blob)

        # bar distributions (within cluster)
        counts = {}
        if do_cond and cond_col in df.columns:
            cc = df.loc[idx, cond_col].astype(str).fillna("NA").value_counts().to_dict()
            counts["cond"] = cc
        if do_pat and pat_col in df.columns:
            pc = df.loc[idx, pat_col].astype(str).fillna("NA").value_counts().to_dict()
            counts["pat"] = pc
        if do_roi and roi_col in df.columns:
            rc = df.loc[idx, roi_col].astype(str).fillna("NA").value_counts().to_dict()
            counts["roi"] = rc
        cluster_bar_counts[c] = counts

    # Layout
    n_clusters = len(cluster_ids)
    n_cols = min(4, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))

    # bars: ~3x thinner than before
    bar_h = 0.08  # thinner
    bar_rows = int(do_cond) + int(do_pat) + int(do_roi)
    if bar_rows == 0:
        bar_rows = 0

    # height ratios: bars then ERSP
    ratios_per_card = ([bar_h] * bar_rows) + [1.0]
    rows_per_card = len(ratios_per_card)

    fig_w = 4.0 * n_cols
    fig_h = 3.0 * n_rows
    hspace = 0.18*1.5
    wspace = 0.12*1.5
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    _poster_set_transparent(fig, [])

    gs = GridSpec(
        nrows=rows_per_card * n_rows,
        ncols=n_cols,
        height_ratios=ratios_per_card * n_rows,
        hspace=hspace,
        wspace=wspace,
        figure=fig,
    )

    # Build list of which bars to draw (in order)
    bar_order = []
    if do_cond and cond_col in df.columns: bar_order.append(("cond", cond_colors))
    if do_pat and pat_col in df.columns:   bar_order.append(("pat", {k: plt.matplotlib.colors.to_rgba(v) for k, v in pat_colors.items()}))
    if do_roi and roi_col in df.columns:   bar_order.append(("roi", roi_colors))

    for k, c in enumerate(cluster_ids):
        r = k // n_cols
        col = k % n_cols

        # bar axes
        for bi, (bname, bcolors) in enumerate(bar_order):
            axb = fig.add_subplot(gs[rows_per_card * r + bi, col])
            axb.set_facecolor("none")
            counts = cluster_bar_counts[c].get(bname, {})
            _draw_thin_bar(axb, counts, bcolors)

        # ERSP axis
        ax = fig.add_subplot(gs[rows_per_card * r + (rows_per_card - 1), col])
        ax.set_facecolor("none")

        mean_c = cluster_ersp[c]
        if mean_c is None:
            ax.axis("off")
            continue

        title = f"C{c} (n={cluster_sizes[c]}) {cluster_stat}"
        _poster_plot_ersp(ax, mean_c, fmax_hz=fmax_hz, vmin=vmin, vmax=vmax, title=title)
        # hide axes (no ticks/labels)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        # keep a black outline (frame)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("black")
            s.set_linewidth(0.9)
            
        # Atlas thumbnail
        if atlas_dir is not None:
            atlas_path = Path(atlas_dir) / atlas_pattern.format(cid=c)
            _try_add_atlas_thumbnail(ax, atlas_path)

        # Blob overlay (rank grayscale + sign marker)
        if show_blob_overlay:
            blob_means = cluster_blob_means[c]
            B = max_blobs if overlay_top_n is None else int(min(max_blobs, overlay_top_n))

            for b in range(B):
                feat = blob_means[b, :]
                if np.allclose(feat, 0.0):
                    continue

                # expected: [t_start, f_peak, t_peak, sf, st, cov, mean_signed]
                f_peak = float(feat[1])
                t_peak = float(feat[2])
                mean_signed = float(feat[6]) if feat.shape[0] >= 7 else 0.0

                x_pct = t_peak * 100.0
                y_hz  = f_peak * float(fmax_hz)

                col_rgba = _poster_blob_rank_gray(b, max_blobs)
                mk = "o" if mean_signed >= 0 else "s"

                ax.scatter(
                    x_pct, y_hz,
                    s=50,
                    marker=mk,
                    facecolor=col_rgba,
                    edgecolor="b" if mk == "s" else "red",
                    linewidths=0.9,
                    zorder=5,
                )

    fig.suptitle(f"{algo_tag} cluster cards – {param_tag}", y=0.995)
    # fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo_tag}_{param_tag}_cluster_cards.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    print(f"[{algo_tag} {param_tag}] Saved:", out_path)



# =========================
# ADD s62_save_cluster_cards_mean_ersp (for -101) to lf_clustering.py
# =========================
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

def s62_save_cluster_cards_mean_ersp(
    df_meta: pd.DataFrame,
    labels: np.ndarray,
    ersp_list: List[np.ndarray],
    algo_tag: str,
    param_tag: str,
    out_dir: Path,
    *,
    fmax_hz: float = 500.0,
    vmin: float = -6.0,
    vmax: float = 6.0,
    dpi: int = 300,
    cluster_stat: str = "mean",          # "mean" or "median" for the MAIN ERSP panel
    X101: Optional[np.ndarray] = None,   # optional: 2D flattened or 3D
    n_freq: Optional[int] = None,
    n_time: Optional[int] = None,

    # ---- UPDATED: second-panel mode ----
    show_panel2: bool = True,
    panel2_mode: str = "101",            # "101" or "ersp_medoid" or "none"
    x101_discrete: bool = True,          # if True: force {-1,0,1} map for the 101 panel (blue/white/red)
    x101_eps: float = 0.0,               # if >0: treat |mean|<=eps as 0 in discretization

    do_cond: bool = True,
    do_pat: bool = True,
    do_roi: bool = True,
    roi_col: str = "brain_region",
    pat_col: str = "patient_id",
    cond_col: str = "condition",
    atlas_dir: Optional[Path] = None,
    atlas_pattern: str = "cluster_{cid}.png",
):
    if len(df_meta) != len(labels) or len(ersp_list) != len(labels):
        raise ValueError("df_meta/ersp_list/labels must align length-wise")

    df = df_meta.copy().reset_index(drop=True)
    labels = np.asarray(labels)

    cluster_ids = sorted(np.unique(labels).tolist(), key=lambda c: (-np.sum(labels == c), int(c)))
    if not cluster_ids:
        print(f"[{algo_tag} {param_tag}] No clusters.")
        return

    # Colors
    if do_pat and pat_col in df.columns:
        pat_colors = _build_patient_color_map_css4(df[pat_col].astype(str).unique().tolist())
        pat_colors = {k: plt.matplotlib.colors.to_rgba(v) for k, v in pat_colors.items()}
    else:
        pat_colors = {}

    if do_cond and cond_col in df.columns:
        cond_colors = _build_condition_color_map_css4(df[cond_col].astype(str).unique().tolist())
        cond_colors = {k: plt.matplotlib.colors.to_rgba(v) for k, v in cond_colors.items()}
    else:
        cond_colors = {}

    if do_roi and (roi_col in df.columns):
        roi_colors = _roi_color_map(df[roi_col].astype(str).fillna("NA"))
    else:
        roi_colors = {}

    # stat
    cluster_stat = str(cluster_stat).strip().lower()
    if cluster_stat not in {"mean", "median"}:
        raise ValueError("cluster_stat must be 'mean' or 'median'")

    def _reduce_stack(arr3):
        return arr3.mean(axis=0) if cluster_stat == "mean" else np.median(arr3, axis=0)

    # Panel2 mode
    panel2_mode = str(panel2_mode).strip().lower()
    if panel2_mode not in {"101", "ersp_medoid", "none"}:
        raise ValueError("panel2_mode must be one of: '101', 'ersp_medoid', 'none'")

    # Handle X101 shape if we might use it
    if (panel2_mode == "101") and show_panel2:
        if X101 is None:
            raise ValueError("panel2_mode='101' requires X101 to be provided.")
        if X101.ndim == 2:
            if n_freq is None or n_time is None:
                n_freq, n_time = ersp_list[0].shape
            expected = int(n_freq) * int(n_time)
            if X101.shape[1] != expected:
                raise ValueError(f"X101 has {X101.shape[1]} features, expected {expected}")
        elif X101.ndim == 3:
            n_freq, n_time = X101.shape[1], X101.shape[2]
        else:
            raise ValueError("X101 must be 2D or 3D")

    # Layout
    n_clusters = len(cluster_ids)
    n_cols = min(4, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))

    bar_h = 0.06
    bar_order = []
    if do_cond and cond_col in df.columns: bar_order.append(("cond", cond_colors))
    if do_pat and pat_col in df.columns:   bar_order.append(("pat", pat_colors))
    if do_roi and roi_col in df.columns:   bar_order.append(("roi", roi_colors))
    bar_rows = len(bar_order)

    use_panel2 = show_panel2 and (panel2_mode != "none")
    panel_rows = 1 + int(use_panel2)

    rows_per_card = bar_rows + panel_rows
    ratios_per_card = ([bar_h] * bar_rows) + ([1.0] * panel_rows)

    fig_w = 3 * n_cols
    fig_h = 4 * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    _poster_set_transparent(fig, [])

    gs = GridSpec(
        nrows=rows_per_card * n_rows,
        ncols=n_cols,
        height_ratios=ratios_per_card * n_rows,
        hspace=0.18 * 1.5,
        wspace=0.12 * 1.3,
        figure=fig,
    )

    for k, c in enumerate(cluster_ids):
        r = k // n_cols
        col = k % n_cols
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        # bars
        for bi, (bname, bcolors) in enumerate(bar_order):
            axb = fig.add_subplot(gs[rows_per_card * r + bi, col])
            axb.set_facecolor("none")
            counts = (
                df.loc[idx, {"cond": cond_col, "pat": pat_col, "roi": roi_col}[bname]]
                .astype(str).fillna("NA").value_counts().to_dict()
            )
            _draw_thin_bar(axb, counts, bcolors)

        # MAIN ERSP panel (mean/median)
        ax1 = fig.add_subplot(gs[rows_per_card * r + bar_rows + 0, col])
        ax1.set_facecolor("none")
        mean_ersp = _reduce_stack(np.stack([ersp_list[i] for i in idx], axis=0))
        _poster_plot_ersp(
            ax1, mean_ersp, fmax_hz=fmax_hz, vmin=vmin, vmax=vmax,
            title=f"C{c} (n={len(idx)}) {cluster_stat} ERSP"
        )

        # Atlas thumbnail on main ERSP panel
        if atlas_dir is not None:
            atlas_path = Path(atlas_dir) / atlas_pattern.format(cid=c)
            _try_add_atlas_thumbnail(ax1, atlas_path)

        # SECOND PANEL: either ERSP medoid OR discrete -101 extraction
        if use_panel2:
            ax2 = fig.add_subplot(gs[rows_per_card * r + bar_rows + 1, col])
            ax2.set_facecolor("none")

            if panel2_mode == "ersp_medoid":
                # "medoid-like" representative: sample whose ERSP is closest to the cluster mean ERSP (L2)
                stack = np.stack([np.asarray(ersp_list[i], dtype=np.float32) for i in idx], axis=0)
                mu = stack.mean(axis=0, keepdims=False)
                d2 = ((stack - mu) ** 2).mean(axis=(1, 2))
                i_best = int(idx[int(np.argmin(d2))])
                E_med = np.asarray(ersp_list[i_best], dtype=np.float32)

                _poster_plot_ersp(
                    ax2, E_med, fmax_hz=fmax_hz, vmin=vmin, vmax=vmax,
                    title=f"C{c} ERSP medoid"
                )

            elif panel2_mode == "101":
                # Mean -101 then DISCRETIZE to {-1,0,1} for clean blue/white/red
                if X101.ndim == 2:
                    Xm = (
                        X101[idx, :].astype(np.float32)
                        .mean(axis=0)
                        .reshape(int(n_freq), int(n_time))
                    )
                else:
                    Xm = X101[idx, :, :].astype(np.float32).mean(axis=0)

                Xm = np.clip(Xm, -1.0, 1.0)

                if x101_discrete:
                    eps = float(x101_eps)
                    Xm_disc = np.zeros_like(Xm, dtype=np.int8)
                    Xm_disc[Xm > +eps] = 1
                    Xm_disc[Xm < -eps] = -1
                    Xm_to_plot = Xm_disc.astype(np.float32)  # keep plotting functions happy
                    v2min, v2max = -1.0, 1.0
                    title2 = f"C{c} -101 (discrete)"
                else:
                    Xm_to_plot = Xm
                    v2min, v2max = -1.0, 1.0
                    title2 = f"C{c} mean -101 occupancy"

                _poster_plot_ersp(
                    ax2, Xm_to_plot, fmax_hz=fmax_hz, vmin=v2min, vmax=v2max,
                    title=title2
                )

            # keep your axis-off + black border policy for panel2
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_xlabel(""); ax2.set_ylabel("")
            ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            for s in ax2.spines.values():
                s.set_visible(True); s.set_color("black"); s.set_linewidth(0.9)

    fig.suptitle(f"{algo_tag} cluster cards – {param_tag}", y=0.995)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo_tag}_{param_tag}_cluster_cards.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08, transparent=True)
    plt.close(fig)
    print(f"[{algo_tag} {param_tag}] Saved:", out_path)


# def s62_save_cluster_cards_mean_ersp(
#     df_meta: pd.DataFrame,
#     labels: np.ndarray,
#     ersp_list: List[np.ndarray],
#     algo_tag: str,
#     param_tag: str,
#     out_dir: Path,
#     *,
#     fmax_hz: float = 500.0,
#     vmin: float = -6.0,
#     vmax: float = 6.0,
#     dpi: int = 300,
#     cluster_stat: str = "mean",          # "mean" or "median"
#     X101: Optional[np.ndarray] = None,   # optional: 2D flattened or 3D
#     n_freq: Optional[int] = None,
#     n_time: Optional[int] = None,
#     show_101: bool = True,              # NEW: toggle -101 panel on/off
#     do_cond: bool = True,
#     do_pat: bool = True,
#     do_roi: bool = True,
#     roi_col: str = "brain_region",
#     pat_col: str = "patient_id",
#     cond_col: str = "condition",
#     atlas_dir: Optional[Path] = None,
#     atlas_pattern: str = "cluster_{cid}.png",
# ):
#     if len(df_meta) != len(labels) or len(ersp_list) != len(labels):
#         raise ValueError("df_meta/ersp_list/labels must align length-wise")

#     df = df_meta.copy().reset_index(drop=True)
#     labels = np.asarray(labels)

#     cluster_ids = sorted(np.unique(labels).tolist(), key=lambda c: (-np.sum(labels==c), int(c)))
#     if not cluster_ids:
#         print(f"[{algo_tag} {param_tag}] No clusters.")
#         return

#     # Colors
#     if do_pat and pat_col in df.columns:
#         pat_colors = _build_patient_color_map_css4(df[pat_col].astype(str).unique().tolist())
#         pat_colors = {k: plt.matplotlib.colors.to_rgba(v) for k, v in pat_colors.items()}
#     else:
#         pat_colors = {}

#     if do_cond and cond_col in df.columns:
#         cond_colors = _build_condition_color_map_css4(df[cond_col].astype(str).unique().tolist())
#         cond_colors = {k: plt.matplotlib.colors.to_rgba(v) for k, v in cond_colors.items()}
#     else:
#         cond_colors = {}

#     if do_roi and (roi_col in df.columns):
#         roi_colors = _roi_color_map(df[roi_col].astype(str).fillna("NA"))
#     else:
#         roi_colors = {}

#     # stat
#     cluster_stat = str(cluster_stat).strip().lower()
#     if cluster_stat not in {"mean", "median"}:
#         raise ValueError("cluster_stat must be 'mean' or 'median'")

#     def _reduce_stack(arr3):
#         return arr3.mean(axis=0) if cluster_stat == "mean" else np.median(arr3, axis=0)

#     # Handle X101 shape if provided (only if we will show it)
#     if (X101 is not None) and show_101:
#         if X101.ndim == 2:
#             if n_freq is None or n_time is None:
#                 n_freq, n_time = ersp_list[0].shape
#             expected = int(n_freq) * int(n_time)
#             if X101.shape[1] != expected:
#                 raise ValueError(f"X101 has {X101.shape[1]} features, expected {expected}")
#         elif X101.ndim == 3:
#             n_freq, n_time = X101.shape[1], X101.shape[2]
#         else:
#             raise ValueError("X101 must be 2D or 3D")

#     # Layout
#     n_clusters = len(cluster_ids)
#     n_cols = min(4, n_clusters)
#     n_rows = int(np.ceil(n_clusters / n_cols))

#     bar_h = 0.06
#     bar_order = []
#     if do_cond and cond_col in df.columns: bar_order.append(("cond", cond_colors))
#     if do_pat and pat_col in df.columns:   bar_order.append(("pat", pat_colors))
#     if do_roi and roi_col in df.columns:   bar_order.append(("roi", roi_colors))
#     bar_rows = len(bar_order)

#     # NEW: panels per cluster (ERSP always, -101 only if enabled)
#     use_101 = (X101 is not None) and show_101
#     panel_rows = 1 + int(use_101)

#     rows_per_card = bar_rows + panel_rows
#     ratios_per_card = ([bar_h] * bar_rows) + ([1.0] * panel_rows)

#     fig_w = 5 * n_cols
#     # NEW: give more vertical space if second panel is shown
#     fig_h = (2.6 if not use_101 else 4.2) * n_rows

#     fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
#     _poster_set_transparent(fig, [])

#     gs = GridSpec(
#         nrows=rows_per_card * n_rows,
#         ncols=n_cols,
#         height_ratios=ratios_per_card * n_rows,
#         hspace=0.18*1.5,
#         wspace=0.12*1.2,
#         figure=fig,
#     )

#     for k, c in enumerate(cluster_ids):
#         r = k // n_cols
#         col = k % n_cols
#         idx = np.where(labels == c)[0]
#         if len(idx) == 0:
#             continue

#         # bars
#         for bi, (bname, bcolors) in enumerate(bar_order):
#             axb = fig.add_subplot(gs[rows_per_card * r + bi, col])
#             axb.set_facecolor("none")
#             counts = (
#                 df.loc[idx, {"cond": cond_col, "pat": pat_col, "roi": roi_col}[bname]]
#                 .astype(str).fillna("NA").value_counts().to_dict()
#             )
#             _draw_thin_bar(axb, counts, bcolors)

#         # ERSP panel
#         ax1 = fig.add_subplot(gs[rows_per_card * r + bar_rows + 0, col])
#         ax1.set_facecolor("none")
#         mean_ersp = _reduce_stack(np.stack([ersp_list[i] for i in idx], axis=0))
#         _poster_plot_ersp(
#             ax1, mean_ersp, fmax_hz=fmax_hz, vmin=vmin, vmax=vmax,
#             title=f"C{c} (n={len(idx)}) {cluster_stat} ERSP"
#         )

#         # Atlas thumbnail on ERSP panel
#         if atlas_dir is not None:
#             atlas_path = Path(atlas_dir) / atlas_pattern.format(cid=c)
#             _try_add_atlas_thumbnail(ax1, atlas_path)

#         # -101 occupancy panel (optional, NEW toggle + float cast + clip)
#         if use_101:
#             ax2 = fig.add_subplot(gs[rows_per_card * r + bar_rows + 1, col])
#             ax2.set_facecolor("none")

#             if X101.ndim == 2:
#                 Xm = (
#                     X101[idx, :].astype(np.float32)
#                     .mean(axis=0)
#                     .reshape(int(n_freq), int(n_time))
#                 )
#             else:
#                 Xm = X101[idx, :, :].astype(np.float32).mean(axis=0)

#             Xm = np.clip(Xm, -1.0, 1.0)

#             _poster_plot_ersp(
#                 ax2, Xm, fmax_hz=fmax_hz, vmin=-1.0, vmax=1.0,
#                 title=f"C{c} mean -101 occupancy"
#             )
#             ax2.set_xticks([]); ax2.set_yticks([])
#             ax2.set_xlabel(""); ax2.set_ylabel("")
#             ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
#             for s in ax2.spines.values():
#                 s.set_visible(True); s.set_color("black"); s.set_linewidth(0.9)
            

#     fig.suptitle(f"{algo_tag} cluster cards – {param_tag}", y=0.995)

#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / f"{algo_tag}_{param_tag}_cluster_cards.png"
#     fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08, transparent=True)
#     plt.close(fig)
#     print(f"[{algo_tag} {param_tag}] Saved:", out_path)

def s23_build_minus101_feature_matrix(
    ersp_list: list[np.ndarray],
    *,
    thr_pos: float,
    thr_neg: float,
    # If you want blob-consistent segmentation, set use_valley_blobs=True and pass valley params:
    use_valley_blobs: bool = True,
    delta_valley: float = 0.5,
    min_mean_pos: float = 0.0,
    max_mean_neg: float = 0.0,
    max_blobs: int = 6,
    sign_mode: str = "both",
    score_min: float | None = None,
    fmax: float = POSTER_FMAX_HZ,
    flatten: bool = True,
) -> np.ndarray:
    """
    Build the -101 feature set.

    If use_valley_blobs=True:
      - uses the same valley segmentation logic as blob extraction,
      - then writes +1/-1 into the blob masks (intensity discarded).
    If use_valley_blobs=False:
      - simple thresholding: +1 if ersp>=thr_pos, -1 if ersp<=thr_neg, else 0.

    Returns:
      X_101: (n_samples, n_freq*n_time) if flatten else (n_samples, n_freq, n_time)
    """
    import numpy as np

    n = len(ersp_list)
    n_freq, n_time = ersp_list[0].shape

    if flatten:
        X = np.zeros((n, n_freq * n_time), dtype=np.int8)
    else:
        X = np.zeros((n, n_freq, n_time), dtype=np.int8)

    if use_valley_blobs:
        from functions.lf_blob_metrics import s21_segment_valley_blobs

    for i, ersp in enumerate(ersp_list):
        ersp = np.asarray(ersp)

        if use_valley_blobs:
            blobs = s21_segment_valley_blobs(
                ersp,
                thr_pos=thr_pos,
                thr_neg=thr_neg,
                delta_valley=delta_valley,
                min_mean_pos=min_mean_pos,
                max_mean_neg=max_mean_neg,
                max_blobs=max_blobs,
                sign_mode=sign_mode,
                fmax=float(fmax),
            )
            if score_min is not None:
                blobs = [b for b in blobs if float(b.get("score", 0.0)) >= float(score_min)]

            seg = np.zeros_like(ersp, dtype=np.int8)
            for b in blobs:
                m = b["mask"]
                s = b.get("sign", "pos")

                # robust decode
                if isinstance(s, (int, np.integer, float, np.floating)):
                    seg[m] = 1 if float(s) > 0 else -1
                else:
                    ss = str(s).strip().lower()
                    if ss in {"pos", "+", "positive", "p", "1", "plus"}:
                        seg[m] = 1
                    elif ss in {"neg", "-", "negative", "n", "-1", "minus"}:
                        seg[m] = -1
                    else:
                        # last resort: infer from blob mean if present
                        mu = float(b.get("mean", b.get("mean_signed", 0.0)))
                        seg[m] = 1 if mu >= 0 else -1

        else:
            seg = np.zeros_like(ersp, dtype=np.int8)
            seg[ersp >= thr_pos] = 1
            seg[ersp <= thr_neg] = -1

        if flatten:
            X[i, :] = seg.reshape(-1)
        else:
            X[i, :, :] = seg

    return X

# =========================
# Poster helpers (ADD to lf_clustering.py)
# =========================
import numpy as np
import matplotlib.pyplot as plt

def _poster_set_transparent(fig, axes):
    fig.patch.set_alpha(0.0)
    for ax in np.ravel(axes):
        ax.set_facecolor("none")

def _poster_plot_ersp(
    ax,
    ersp,
    *,
    fmax_hz=500.0,
    vmin=-6.0,
    vmax=6.0,
    cmap="bwr",
    midline_pct=50.0,
    xlabel="Time (%)",
    ylabel="Frequency (Hz)",
    title=None,
):
    # Force consistent units: x in % time, y in Hz
    ax.imshow(
        ersp,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0.0, 100.0, 0.0, float(fmax_hz)),
        interpolation="nearest",
    )
    ax.axvline(float(midline_pct), color="lightgray", linewidth=1.0, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_axis_off()
    if title:
        ax.set_title(title)
    
    # hide axes (no ticks/labels)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # keep a black outline (frame)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.9)


def _poster_blob_rank_gray(b, max_blobs, dark=0.15, light=0.85):
    # blob1 darkest, blobN lightest
    if max_blobs <= 1:
        g = dark
    else:
        t = b / (max_blobs - 1)
        g = dark + t * (light - dark)
    return (g, g, g, 1.0)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _poster_set_transparent(fig, axes):
    fig.patch.set_alpha(0.0)
    for ax in np.ravel(axes):
        ax.set_facecolor("none")

# def _poster_plot_ersp(
#     ax,
#     ersp,
#     *,
#     fmax_hz=500.0,
#     vmin=-6.0,
#     vmax=6.0,
#     cmap="bwr",
#     midline_pct=50.0,
#     xlabel="Time (%)",
#     ylabel="Frequency (Hz)",
#     title=None,
# ):
#     ax.imshow(
#         ersp,
#         origin="lower",
#         aspect="auto",
#         cmap=cmap,
#         vmin=vmin,
#         vmax=vmax,
#         extent=(0.0, 100.0, 0.0, float(fmax_hz)),
#         interpolation="nearest",
#     )
#     ax.axvline(float(midline_pct), color="gray", linewidth=1.0, alpha=0.9)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_axis_off()
#     if title:
#         ax.set_title(title)

def _poster_blob_rank_gray(b, max_blobs, dark=0.15, light=0.85):
    if max_blobs <= 1:
        g = dark
    else:
        t = b / (max_blobs - 1)
        g = dark + t * (light - dark)
    return (g, g, g, 1.0)

def _cohort(pid: str) -> str:
    p = str(pid).strip().upper()
    if p.startswith("EL"): return "EL"
    if p.startswith("PAT_"): return "PAT"
    if p.startswith("MICROEPI"): return "MICRO"
    return "OTHER"

def _build_patient_color_map_css4(patients):
    patients = sorted(set(map(str, patients)))
    try:
        from functions import lf_blob_recon_config as C
        el_names = list(C.EL_COLOR_NAMES)
        pat_names = list(C.PAT_COLOR_NAMES)
        mic_names = list(C.MICRO_COLOR_NAMES)
    except Exception:
        el_names = ["blue","deepskyblue","cyan","teal","green","lime","powderblue","forestgreen","lightcyan"]
        pat_names = ["blueviolet","fuchsia","deeppink","crimson","pink","red","yellow","chocolate","gold","purple","saddlebrown","lemonchiffon","lavenderblush"]
        mic_names = ["navy","darkslategray","black","darkred","darkolivegreen"]

    out = {}
    el  = [p for p in patients if _cohort(p) == "EL"]
    pat = [p for p in patients if _cohort(p) == "PAT"]
    mic = [p for p in patients if _cohort(p) == "MICRO"]
    oth = [p for p in patients if _cohort(p) == "OTHER"]

    for i, p in enumerate(el):  out[p] = el_names[i % len(el_names)]
    for i, p in enumerate(pat): out[p] = pat_names[i % len(pat_names)]
    for i, p in enumerate(mic): out[p] = mic_names[i % len(mic_names)]
    fb = ["gray", "dimgray", "slategray"]
    for i, p in enumerate(oth): out[p] = fb[i % len(fb)]
    return out

def _build_condition_color_map_css4(conditions):
    conds = sorted(set(map(str, conditions)))
    try:
        from functions import lf_blob_recon_config as C
        names = list(C.CONDITION_COLOR_NAMES)
    except Exception:
        names = ["red", "blueviolet", "deeppink"]
    return {c: names[i % len(names)] for i, c in enumerate(conds)}

def _cat_to_codes(series):
    s = series.astype(str).fillna("NA")
    cats = sorted(s.unique().tolist())
    mapping = {c: i for i, c in enumerate(cats)}
    codes = s.map(mapping).to_numpy(dtype=np.int32)
    return codes, cats, mapping

def _roi_color_map(series):
    # ROI palette is not defined in recon_config; use a stable discrete colormap.
    codes, cats, mapping = _cat_to_codes(series)
    cmap = plt.get_cmap("tab20", max(3, len(cats)))
    colors = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
    return colors

def _draw_thin_bar(ax, counts_dict, color_map, *, total=None):
    ax.axis("off")
    if total is None:
        total = float(sum(counts_dict.values()))
    if total <= 0:
        return
    left = 0.0
    for key, cnt in counts_dict.items():
        frac = float(cnt) / total if total > 0 else 0.0
        if frac <= 0:
            continue
        ax.barh(0, frac, left=left, height=1.0, color=color_map.get(key, "gray"), edgecolor="none")
        left += frac
    ax.set_xlim(0, 1)

def _try_add_atlas_thumbnail(ax, atlas_img_path: Path):
    if atlas_img_path is None:
        return False
    atlas_img_path = Path(atlas_img_path)
    if not atlas_img_path.exists():
        return False
    try:
        img = plt.imread(str(atlas_img_path))
    except Exception:
        return False

    # Inset on the right side of the ERSP axis
    ins = ax.inset_axes([0.73, 0.06, 0.25, 0.25])  # [x0,y0,w,h] in axis fraction
    ins.set_facecolor("none")
    ins.imshow(img)
    ins.axis("off")
    return True
