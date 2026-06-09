"""
lf_pool.py — Confirmatory power-pooling engine for 04_FBM_Pooling.

Fourth stage of the FBM pipeline. Where 02 clustered ERSPs and 03 decoded them,
this stage runs a CONFIRMATORY test: do contacts show a task-related response in
*a-priori* time zones of the warped trial timeline, and do the responsive
contacts cluster anatomically?

It is built on the SAME canonical samples the clustering pipeline uses
(02_FBM_Clustering/functions/lf_dataset.prepare_dataset): one sample = the
trial-AVERAGED ERSP per electrode x condition, shape (129 freq x 300 time).
The time axis is warped 50% stimulus / 50% response, so bin 150 == response
onset. ERSP values are already baseline-relative (normalized to a -0.6..-0.1 s
pre-stim baseline), so "compare against baseline" is baked in — significance is
a WINDOWED version of clustering's high-activity sigma/proportion gate.

Pipeline (notebooks):
  410  zone discovery  — overlay every contact's blobs as red/blue circle
       outlines per condition, read the dense time regions, hand-define the
       boxcar + gaussian windows for three zones (perception, pre_articulation,
       audio) -> outputs/pooling/window_config.json
  420  pool & qualify  — pool ERSP power as a time-weighted average inside each
       window, for TWO feature sets (HG line; 15 bands separately) x both window
       shapes, qualify each contact with the windowed clustering-comparable gate
       (both signs kept) -> a tidy per-(contact,condition,zone,shape,feature) table
  430  anatomy         — map qualifiers to Yeo-7/17 + Desikan-Killiany gyri,
       summarize purity/compactness per zone x shape, render on fsaverage
  490  results         — narrative + boxcar-vs-gaussian robustness comparison

Design (locked with Lora, 2026-06):
  - Samples / gate: identical to clustering (thr_pos=2.2 min_prop=0.02 OR
    thr_neg=-3.0 min_prop=0.04), computed WITHIN each pooled time window.
  - Feature sets, run identically: 'hg' (70-150 Hz line) and 'bands15' (each of
    the 15 canonical bands separately, on the native 300-bin time axis).
  - Windows: 'boxcar' (primary) and 'gaussian' (robustness vs edge effects /
    latency jitter). Gaussian gate support = +/-2 sigma (95% mass).
  - Sign: keep BOTH; 410 draws each blob as a red(+)/blue(-) ellipse outline
    whose shade scales with the blob's mean dB.
  - Anatomy: Yeo-7 & Yeo-17 (precomputed in the coords CSVs), Desikan-Killiany
    gyri (built once via lf_anatomy.build_aparc_cache), and fsaverage renders.

Reuse-first: feature/blob/dataset/anatomy logic is imported from
02_FBM_Clustering/functions via an importlib path trick (same idea as
03_FBM_Classifying/lf_classify._load_clustering_module). The contact<->coords
join helpers mirror lf_classify/lf_io_utils exactly and are re-stated here so
this module stays self-contained (no dependency on the classification stack).
"""
from __future__ import annotations

import datetime as _dt
import getpass
import importlib.util
import json
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0 — Reuse the clustering modules (single source of truth)
# ============================================================
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]                          # functions -> 04_.. -> repo
_CLUST_FUNCS = _REPO_ROOT / "02_FBM_Clustering" / "functions"


def _load_module(funcs_dir: Path, mod_name: str):
    """Import a module by absolute path, avoiding the sibling-`functions/`
    package collision (02's vs 04's). Mirrors lf_classify._load_clustering_module."""
    path = Path(funcs_dir) / f"{mod_name}.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location(f"_pool_{mod_name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)                # type: ignore[union-attr]
        return module
    if str(funcs_dir) not in sys.path:
        sys.path.insert(0, str(funcs_dir))
    return importlib.import_module(mod_name)


# Eager: light deps only (numpy/pandas/skimage-optional). lf_features's skimage
# import is guarded, and we only call downsample with time_bins_out == n_time
# (the no-resize short-circuit), so skimage is never actually required here.
_lf_dataset = _load_module(_CLUST_FUNCS, "lf_dataset")
_lf_hg = _load_module(_CLUST_FUNCS, "lf_hg")
_lf_features = _load_module(_CLUST_FUNCS, "lf_features")
_lf_anatomy = _load_module(_CLUST_FUNCS, "lf_anatomy")

# Lazy: heavier / optional. Blob segmentation pulls scipy.ndimage; the recon
# config is only needed for the 430 brain renders.
_lf_blob = None
_recon_cfg = None


def _ensure_blob():
    global _lf_blob
    if _lf_blob is None:
        _lf_blob = _load_module(_CLUST_FUNCS, "lf_blob_metrics")
    return _lf_blob


def _ensure_recon_cfg():
    global _recon_cfg
    if _recon_cfg is None:
        _recon_cfg = _load_module(_CLUST_FUNCS, "lf_recon_shared_config")
    return _recon_cfg


# ============================================================
# 1 — Constants (= clustering's, so the gate stays comparable)
# ============================================================
CONDITIONS = ("audio", "picture", "reading")
TASK = "LM"
N_FREQ = 129
N_TIME = 300
FMAX_HZ = 500.0
HG_BAND = (70.0, 150.0)
STIM_FRAC = 0.5                       # bin int(STIM_FRAC*N_TIME)=150 == response onset

# Windowed high-activity gate — identical thresholds to lf_dataset defaults.
THR_POS, MIN_PROP_POS = 2.2, 0.02
THR_NEG, MIN_PROP_NEG = -3.0, 0.04

FREQ_BANDS = list(_lf_features.FREQ_BANDS_15_TO_400HZ)   # 15 (lo, hi) Hz tuples
FEATURE_SETS = ("hg", "bands15")
WINDOW_SHAPES = ("boxcar", "gaussian")
DEFAULT_ZONES = ("perception", "pre_articulation", "audio")
GAUSS_SUPPORT_SIGMA = 2.0             # gate support = center +/- this many sigma

# 410 overlay blob-score gate: drop blobs below this percentile of the dataset's
# blob scores (same idea as clustering/classification's M101_SCORE_PCT). Tunable.
SCORE_PCT = 33.0
# 'ds' (downsampled) grid — the clustering "rawds" representation: 15 freq bands
# x DS_TIME_BINS time bins, via lf_features.build_X_3d_downsampled.
DS_TIME_BINS = 30

COORDS_DIR = (_REPO_ROOT / "02_FBM_Clustering" / "outputs" / "250_recon"
              / "fsaverage" / "coords")
OUTPUTS_ROOT = _HERE.parents[1] / "outputs" / "pooling"
DATASET_CACHE = _HERE.parents[1] / "outputs" / "_dataset" / "pooling"
APARC_CACHE = _HERE.parents[1] / "outputs" / "_anatomy" / "aparc_lookup.csv"

SCHEMA_VERSION = 1

# Non-network Yeo labels dropped from anatomy summaries (mirrors lf_classify).
_YEO_NON_NETWORK = {
    "FreeSurfer_Defined_Medial_Wall", "Medial_Wall", "medial_wall",
    "WhiteMatter", "Unknown", "unknown", "None", "", "nan",
}


# ============================================================
# 2 — Electrode <-> fsaverage contact-name join (mirrors lf_classify / 252)
# ============================================================
_RX_BERN = re.compile(r"_ERSP_([A-Za-z]+)_([LR])(\d+)_TN", re.IGNORECASE)


def normalize_label(s) -> str:
    """Strip '_' and '-', uppercase ('aH_R-1' -> 'AHR1'). Same rule the coords
    side and the clustering/classification stages use, so joins line up."""
    if s is None:
        return ""
    return str(s).replace("_", "").replace("-", "").upper()


def contact_from_row(patient_id: str, electrode: str, file_path: str) -> Optional[str]:
    """Normalized contact key joining an ERSP sample to a coords row.

    BERN/EL filenames look like <pat>_<cond>_WM_ERSP_<electrode>_<L|R><num>_TN
      -> contact = electrode + side + num ('A_R10' -> 'AR10').
    GVA/PAT carry the contact directly in the electrode column.
    """
    base = Path(str(file_path).replace("\\", "/")).name.rsplit(".", 1)[0]
    m = _RX_BERN.search(base)
    if m:
        return normalize_label(f"{m.group(1)}{m.group(2)}{m.group(3)}")
    if isinstance(electrode, str) and electrode.strip():
        return normalize_label(electrode)
    return None


def is_real_network(label) -> bool:
    return str(label) not in _YEO_NON_NETWORK and not str(label).lower().startswith("nan")


# ============================================================
# 3 — Dataset preparation (the SAME canonical samples as clustering)
# ============================================================
def prepare_pooling_dataset(input_dir, cache_dir: Path = DATASET_CACHE,
                            verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load the canonical (df_meta, X_3d), UNGATED, and add the `contact_norm`
    join key. Windowed gating is applied later (per zone), so high-activity is
    NOT pre-applied here. X_3d is (N, 129, 300). The heavy ERSP walk is cached
    by lf_dataset.prepare_dataset.
    """
    df_meta, _ersp_list, X_3d = _lf_dataset.prepare_dataset(
        input_dir,
        task=TASK, conditions=CONDITIONS, n_freq=N_FREQ, n_time=N_TIME,
        thr_pos=THR_POS, min_prop_pos=MIN_PROP_POS,
        thr_neg=THR_NEG, min_prop_neg=MIN_PROP_NEG,
        apply_high_activity=False,
        cache_dir=Path(cache_dir) / "_raw_ungated",
        verbose=verbose,
    )
    if X_3d.shape[1:] != (N_FREQ, N_TIME):
        raise ValueError(f"expected ERSP (N,{N_FREQ},{N_TIME}); got {X_3d.shape}")
    df_meta = df_meta.copy()
    df_meta["contact_norm"] = [
        contact_from_row(r.patient_id, r.electrode, r.file_path)
        for r in df_meta.itertuples()
    ]
    if verbose:
        n_match = int(df_meta["contact_norm"].notna().sum())
        print(f"[lf_pool] {len(df_meta)} samples, {n_match} with a contact key, "
              f"X_3d={X_3d.shape}")
    return df_meta, X_3d


def downsample_dataset(X_3d: np.ndarray, *, time_bins: int = DS_TIME_BINS,
                       verbose: bool = True) -> np.ndarray:
    """Band-aware downsample the full (N,129,300) ERSPs to the clustering
    'rawds' grid (N, 15 bands, `time_bins`). Reuses
    lf_features.build_X_3d_downsampled with the canonical 15-band edges. Used
    for the preliminary `USE_DS` mode (faster; matches 212's representation)."""
    ds = _lf_features.build_X_3d_downsampled(
        list(X_3d), freq_band_edges=FREQ_BANDS, fmax_hz=FMAX_HZ,
        time_bins_out=int(time_bins), verbose=verbose)
    return ds


# ============================================================
# 4 — Window configuration (defined by hand in 410, read by 420)
# ============================================================
def default_window_config() -> dict:
    """Seed config — placeholder windows for the three zones. Lora overwrites
    the numbers in 410 after looking at the overlays. Percentages are of the
    0..N_TIME warped axis (50% == response onset)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "n_time": N_TIME,
        "stim_frac": STIM_FRAC,
        "axis_units": "percent_of_300_warped_axis",
        "zones": {
            "perception":       {"boxcar": {"t_lo_pct": 0.0,  "t_hi_pct": 50.0},
                                 "gaussian": {"center_pct": 25.0, "sigma_pct": 10.0}},
            "pre_articulation": {"boxcar": {"t_lo_pct": 40.0, "t_hi_pct": 60.0},
                                 "gaussian": {"center_pct": 50.0, "sigma_pct": 8.0}},
            "audio":            {"boxcar": {"t_lo_pct": 50.0, "t_hi_pct": 100.0},
                                 "gaussian": {"center_pct": 75.0, "sigma_pct": 12.0}},
        },
    }


def validate_window_config(cfg: dict) -> None:
    """Raise if the config is malformed (ranges out of 0..100, sigma<=0, missing
    shapes). Called before saving in 410 and after loading in 420."""
    zones = cfg.get("zones", {})
    if not zones:
        raise ValueError("config has no 'zones'")
    for name, spec in zones.items():
        if "boxcar" not in spec or "gaussian" not in spec:
            raise ValueError(f"zone '{name}' must define both 'boxcar' and 'gaussian'")
        b = spec["boxcar"]
        lo, hi = float(b["t_lo_pct"]), float(b["t_hi_pct"])
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(f"zone '{name}' boxcar needs 0<=t_lo_pct<t_hi_pct<=100")
        g = spec["gaussian"]
        c, s = float(g["center_pct"]), float(g["sigma_pct"])
        if not (0.0 <= c <= 100.0) or s <= 0.0:
            raise ValueError(f"zone '{name}' gaussian needs 0<=center_pct<=100, sigma_pct>0")


def save_window_config(cfg: dict, path) -> Path:
    validate_window_config(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(cfg, indent=2))
    return path


def load_window_config(path) -> dict:
    cfg = json.loads(Path(path).read_text())
    validate_window_config(cfg)
    return cfg


def make_window_weights(zone_spec: dict, shape: str, n_time: int = N_TIME) -> np.ndarray:
    """Normalized (sum=1) time-weight vector (n_time,) for one (zone, shape).
    boxcar: 1 inside [t_lo_pct, t_hi_pct), 0 outside. gaussian: exp(-(t-c)^2/2s^2).
    """
    t = np.arange(n_time, dtype=float)
    if shape == "boxcar":
        b = zone_spec["boxcar"]
        lo = float(b["t_lo_pct"]) / 100.0 * n_time
        hi = float(b["t_hi_pct"]) / 100.0 * n_time
        w = ((t >= lo) & (t < hi)).astype(float)
    elif shape == "gaussian":
        g = zone_spec["gaussian"]
        c = float(g["center_pct"]) / 100.0 * n_time
        s = float(g["sigma_pct"]) / 100.0 * n_time
        w = np.exp(-0.5 * ((t - c) / s) ** 2)
    else:
        raise ValueError(f"unknown window shape: {shape}")
    tot = float(w.sum())
    return w / tot if tot > 0 else w


def window_support(zone_spec: dict, shape: str, n_time: int = N_TIME) -> np.ndarray:
    """Boolean (n_time,) mask of the time columns the gate is evaluated over.
    boxcar -> the box; gaussian -> center +/- GAUSS_SUPPORT_SIGMA (95% mass)."""
    t = np.arange(n_time)
    if shape == "boxcar":
        b = zone_spec["boxcar"]
        lo = float(b["t_lo_pct"]) / 100.0 * n_time
        hi = float(b["t_hi_pct"]) / 100.0 * n_time
        return (t >= lo) & (t < hi)
    g = zone_spec["gaussian"]
    c = float(g["center_pct"]) / 100.0 * n_time
    s = float(g["sigma_pct"]) / 100.0 * n_time
    return np.abs(t - c) <= GAUSS_SUPPORT_SIGMA * s


# ============================================================
# 5 — Blob segmentation + ellipse geometry (410 overlays)
# ============================================================
def segment_contact_blobs(ersp: np.ndarray, score_min: Optional[float] = None) -> List[dict]:
    """Score-sorted valley blobs (both signs) for one ERSP. Reuses
    lf_blob_metrics.s21_segment_valley_blobs — the SAME segmentation the
    clustering -1/0/+1 path uses. If `score_min` is given, blobs below that
    score are dropped (thins the 410 overlays)."""
    blob = _ensure_blob()
    blobs = blob.s21_segment_valley_blobs(ersp, sign_mode="both", fmax=FMAX_HZ)
    if score_min is not None:
        blobs = [b for b in blobs if float(b.get("score", 0.0)) >= score_min]
    return blobs


def resolve_score_gate(X_3d: np.ndarray, *, pct: float = SCORE_PCT,
                       sample_cap: int = 3000, verbose: bool = True) -> float:
    """Blob-score gate = the `pct`-th percentile of blob scores over a subsample
    of the (full-res) dataset. Pass the result as `score_min` to the 410 plots so
    only the stronger blobs are drawn. Mirrors lf_classify.resolve_m101_score_min."""
    n = int(X_3d.shape[0])
    if n == 0:
        return 0.0
    step = max(1, n // int(sample_cap))
    scores: List[float] = []
    n_sampled = 0
    for i in range(0, n, step):
        scores += [float(b["score"]) for b in segment_contact_blobs(X_3d[i])]
        n_sampled += 1
    gate = float(np.percentile(scores, pct)) if scores else 0.0
    if verbose:
        print(f"[lf_pool] blob-score gate = {pct:.0f}th pct = {gate:.4g} "
              f"({len(scores)} blobs over {n_sampled} sampled contacts)")
    return gate


def blob_ellipse(blob: dict) -> Tuple[float, float, float, float, int, float]:
    """Moment-matched ellipse for one blob, in (time-bin, freq-Hz) coordinates.

    Returns (t_center, f_center_hz, t_half, f_half_hz, sign, mean_db).
    Center = centroid of the mask; half-extents = 2*std (≈95% under a Gaussian
    approx), floored so single-pixel blobs stay visible. Blobs are arbitrary
    shapes — this ellipse is an approximation, not the true outline.
    """
    ti = np.asarray(blob["times_idx"], dtype=float)
    fi = np.asarray(blob["freqs_idx"], dtype=float)
    t_center = float(ti.mean())
    f_center = float(fi.mean())
    t_half = max(2.0 * float(ti.std()), 1.5)
    f_half = max(2.0 * float(fi.std()), 1.5)
    hz = FMAX_HZ / (N_FREQ - 1)
    return (t_center, f_center * hz, t_half, f_half * hz,
            int(blob["sign"]), float(blob["mean"]))


def _condition_indices(df_meta: pd.DataFrame, condition: str) -> np.ndarray:
    return df_meta.index[df_meta["condition"] == condition].to_numpy()


def plot_blob_overlay(X_3d: np.ndarray, df_meta: pd.DataFrame, condition: str,
                      out_png, *, score_min: Optional[float] = None,
                      db_norm=(2.0, 8.0), alpha: float = 0.18,
                      dpi: int = 150, blobs_per_sample: Optional[List] = None) -> Path:
    """Overlay every contact's blobs (this condition) as RED(+)/BLUE(-) ellipse
    OUTLINES in the freq x time plane; outline shade scales with |mean dB|.
    Dense time regions reveal candidate pooling zones. Vertical line at the
    50% mark = response onset. `score_min` drops weak blobs (see resolve_score_gate).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection

    idx = _condition_indices(df_meta, condition)
    norm = mcolors.Normalize(vmin=db_norm[0], vmax=db_norm[1])
    reds = plt.get_cmap("Reds")
    blues = plt.get_cmap("Blues")

    patches, edge_colors = [], []
    n_blobs = 0
    for i in idx:
        bl = blobs_per_sample[i] if blobs_per_sample is not None else segment_contact_blobs(X_3d[i])
        if score_min is not None:
            bl = [b for b in bl if float(b.get("score", 0.0)) >= score_min]
        for b in bl:
            tc, fc, th, fh, sign, mean = blob_ellipse(b)
            patches.append(Ellipse((tc, fc), width=2 * th, height=2 * fh))
            shade = float(norm(abs(mean)))
            cmap = reds if sign > 0 else blues
            edge_colors.append(cmap(0.35 + 0.6 * min(max(shade, 0.0), 1.0)))
            n_blobs += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    if patches:
        pc = PatchCollection(patches, facecolors="none",
                             edgecolors=edge_colors, linewidths=0.8, alpha=alpha)
        ax.add_collection(pc)
    ax.axvline(STIM_FRAC * N_TIME, color="0.35", lw=1.2, ls="--", zorder=5)
    ax.text(STIM_FRAC * N_TIME + 3, FMAX_HZ * 0.96, "response onset",
            fontsize=8, color="0.35")
    ax.set_xlim(0, N_TIME)
    ax.set_ylim(0, FMAX_HZ)
    ax.set_xlabel("warped time bin  (0–300; 150 = 50% = response onset)")
    ax.set_ylabel("frequency (Hz)")
    gate_txt = "" if score_min is None else f" · score≥{score_min:.3g}"
    ax.set_title(f"{condition} — blob overlay  "
                 f"(n={len(idx)} contacts · {n_blobs} blobs{gate_txt} · red=+ blue=−, shade=|dB|)")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def compute_time_marginal(X_3d: np.ndarray, df_meta: pd.DataFrame, condition: str,
                          *, score_min: Optional[float] = None,
                          blobs_per_sample: Optional[List] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-time-bin count of active blobs (this condition), split by sign.
    Returns (pos_counts, neg_counts), each shape (N_TIME,). `score_min` drops weak blobs."""
    idx = _condition_indices(df_meta, condition)
    pos = np.zeros(N_TIME, dtype=float)
    neg = np.zeros(N_TIME, dtype=float)
    for i in idx:
        bl = blobs_per_sample[i] if blobs_per_sample is not None else segment_contact_blobs(X_3d[i])
        if score_min is not None:
            bl = [b for b in bl if float(b.get("score", 0.0)) >= score_min]
        for b in bl:
            tcols = np.unique(np.asarray(b["times_idx"], dtype=int))
            tcols = tcols[(tcols >= 0) & (tcols < N_TIME)]
            if int(b["sign"]) > 0:
                pos[tcols] += 1
            else:
                neg[tcols] += 1
    return pos, neg


def plot_time_marginal(X_3d: np.ndarray, df_meta: pd.DataFrame, condition: str,
                       out_png, *, score_min: Optional[float] = None, dpi: int = 150,
                       blobs_per_sample: Optional[List] = None) -> Path:
    """Time-marginal blob density (the quantitative aid for choosing windows):
    contacts with a positive (red) / negative (blue) blob active at each bin."""
    import matplotlib.pyplot as plt
    pos, neg = compute_time_marginal(X_3d, df_meta, condition, score_min=score_min,
                                     blobs_per_sample=blobs_per_sample)
    t = np.arange(N_TIME)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(t, 0, pos, color="#cc0033", alpha=0.5, label="positive (+)")
    ax.fill_between(t, 0, -neg, color="#0033cc", alpha=0.5, label="negative (−)")
    ax.axvline(STIM_FRAC * N_TIME, color="0.35", lw=1.2, ls="--")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(0, N_TIME)
    ax.set_xlabel("warped time bin  (150 = response onset)")
    ax.set_ylabel("# contacts active  (+ up / − down)")
    ax.set_title(f"{condition} — time-marginal blob density")
    ax.legend(loc="upper right", fontsize=8)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _band_labels() -> List[str]:
    return [f"{int(lo)}-{int(hi)}" for lo, hi in FREQ_BANDS]


def plot_ds_heatmap(X_ds: np.ndarray, df_meta: pd.DataFrame, condition: str,
                    out_png, *, dpi: int = 150, vlim: Optional[Tuple[float, float]] = None) -> Path:
    """`USE_DS` mode 410 view: per-condition MEAN downsampled ERSP (15 bands x
    n_time_ds) as a band×time heatmap. Dense / strong patches reveal candidate
    pooling zones (blob outlines don't survive downsampling, so this replaces
    them). Vertical line at the 50% mark = response onset."""
    import matplotlib.pyplot as plt
    idx = _condition_indices(df_meta, condition)
    M = X_ds[idx].mean(axis=0)                       # (15, n_time_ds)
    n_band, n_t = M.shape
    if vlim is None:
        a = float(np.nanpercentile(np.abs(M), 99)) or 1.0
        vlim = (-a, a)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=vlim[0], vmax=vlim[1], interpolation="nearest",
                   extent=[0, 100, 0, n_band])
    ax.axvline(STIM_FRAC * 100, color="0.2", lw=1.2, ls="--")
    ax.set_yticks(np.arange(n_band) + 0.5)
    ax.set_yticklabels(_band_labels(), fontsize=7)
    ax.set_xlabel("warped time (%)  (50 = response onset)")
    ax.set_ylabel("frequency band (Hz)")
    ax.set_title(f"{condition} — mean downsampled ERSP  (n={len(idx)} contacts)")
    fig.colorbar(im, ax=ax, label="mean power (dB)")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_ds_time_marginal(X_ds: np.ndarray, df_meta: pd.DataFrame, condition: str,
                          out_png, *, dpi: int = 150) -> Path:
    """`USE_DS` mode time-marginal: mean positive / negative band power across
    contacts × bands at each time bin — the read-off for choosing windows."""
    import matplotlib.pyplot as plt
    idx = _condition_indices(df_meta, condition)
    M = X_ds[idx]                                    # (n, 15, n_time_ds)
    pos = np.clip(M, 0, None).mean(axis=(0, 1))
    neg = np.clip(M, None, 0).mean(axis=(0, 1))
    t_pct = np.linspace(0, 100, M.shape[2])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.fill_between(t_pct, 0, pos, color="#cc0033", alpha=0.5, label="mean +power")
    ax.fill_between(t_pct, 0, neg, color="#0033cc", alpha=0.5, label="mean −power")
    ax.axvline(STIM_FRAC * 100, color="0.2", lw=1.2, ls="--")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("warped time (%)  (50 = response onset)")
    ax.set_ylabel("mean band power (dB)")
    ax.set_title(f"{condition} — ds time-marginal (mean ± power)")
    ax.legend(loc="upper right", fontsize=8)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_window_preview(cfg: dict, out_png=None, *, dpi: int = 150):
    """Plot the boxcar + gaussian weight vectors for every zone (sanity check
    after defining the windows in 410)."""
    import matplotlib.pyplot as plt
    zones = cfg["zones"]
    fig, axes = plt.subplots(len(zones), 1, figsize=(9, 2.0 * len(zones)),
                             squeeze=False, sharex=True)
    t = np.arange(N_TIME)
    for ax, (name, spec) in zip(axes[:, 0], zones.items()):
        ax.plot(t, make_window_weights(spec, "boxcar"), color="#444", label="boxcar")
        ax.plot(t, make_window_weights(spec, "gaussian"), color="#cc0033", label="gaussian")
        ax.axvline(STIM_FRAC * N_TIME, color="0.6", lw=1.0, ls="--")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("warped time bin  (150 = response onset)")
    fig.suptitle("Window weights per zone")
    fig.tight_layout()
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    return fig


# ============================================================
# 6 — Pooling + windowed qualification (420)
# ============================================================
def feature_label(feature_set: str, i: int) -> Tuple[str, int, float, float]:
    """(label, band_idx, band_lo, band_hi) for one feature column."""
    if feature_set == "hg":
        return ("HG_70-150Hz", -1, HG_BAND[0], HG_BAND[1])
    lo, hi = FREQ_BANDS[i]
    return (f"{int(lo)}-{int(hi)}Hz", i, float(lo), float(hi))


def _hg_ds_rows() -> List[int]:
    """Indices of the 15 downsampled bands overlapping the HG band (≈70–170 Hz)."""
    return [i for i, (lo, hi) in enumerate(FREQ_BANDS)
            if hi > HG_BAND[0] and lo < HG_BAND[1]]


def feature_vector(ersp: np.ndarray, feature_set: str, grid: str = "full") -> np.ndarray:
    """The per-time feature representation for one ERSP.
    'hg'      -> (1, n_time) single HG band-mean line.
    'bands15' -> (15, n_time) band-means.
    grid='full' (129 freq x n_time): collapse the raw freq axis to bands / HG line.
    grid='ds'   (15 bands x n_time): rows ARE the bands already, so use directly;
                HG = mean of the bands overlapping 70–150 Hz (≈70–170)."""
    if grid == "ds":
        if feature_set == "hg":
            return np.asarray(ersp, dtype=np.float32)[_hg_ds_rows(), :].mean(axis=0, keepdims=True)
        if feature_set == "bands15":
            return np.asarray(ersp, dtype=np.float32)
        raise ValueError(f"unknown feature_set: {feature_set}")
    if feature_set == "hg":
        ts = _lf_hg.extract_hg_time_series(ersp, hg_band=HG_BAND, fmax=FMAX_HZ)
        return ts[None, :]
    if feature_set == "bands15":
        # time_bins_out == n_time triggers lf_features' no-resize short-circuit,
        # so this stays on the native time axis (no skimage needed).
        return _lf_features.downsample_ersp_to_bands(
            ersp, FREQ_BANDS, fmax_hz=FMAX_HZ, time_bins_out=ersp.shape[1])
    raise ValueError(f"unknown feature_set: {feature_set}")


def pooled_power(ersp: np.ndarray, weights: np.ndarray, feature_set: str,
                 grid: str = "full") -> np.ndarray:
    """Time-weighted mean power inside the window, per feature.
    Returns (1,) for 'hg', (15,) for 'bands15'."""
    feat = feature_vector(ersp, feature_set, grid=grid)   # (n_feat, n_time)
    w = weights / (weights.sum() + 1e-12)
    return (feat * w[None, :]).sum(axis=1)


def windowed_gate(ersp: np.ndarray, support: np.ndarray
                  ) -> Tuple[float, float, bool, bool]:
    """Clustering's high-activity gate, restricted to the window columns.
    Returns (prop_above_pos_w, prop_below_neg_w, qual_pos, qual_neg)."""
    sub = ersp[:, support]
    if sub.size == 0:
        return 0.0, 0.0, False, False
    prop_pos = float((sub > THR_POS).mean())
    prop_neg = float((sub < THR_NEG).mean())
    return prop_pos, prop_neg, prop_pos >= MIN_PROP_POS, prop_neg >= MIN_PROP_NEG


def _sign_of(prop_pos, prop_neg, qpos, qneg) -> int:
    if qpos and qneg:
        return 1 if prop_pos >= prop_neg else -1
    if qpos:
        return 1
    if qneg:
        return -1
    return 0


def temporal_null_p(ersp: np.ndarray, weights: np.ndarray, feature_set: str,
                    observed_vec: np.ndarray, *, grid: str = "full",
                    n_perm: int = 500, seed: int = 0) -> np.ndarray:
    """Per-feature circular-time-shift null p-value (secondary robustness).
    p = P(|pooled(rolled)| >= |observed|), +1 smoothing. NaN when n_perm<=0."""
    if n_perm <= 0:
        return np.full(observed_vec.shape, np.nan)
    rng = np.random.default_rng(seed)
    n_time = ersp.shape[1]
    obs = np.abs(observed_vec)
    ge = np.zeros(observed_vec.shape, dtype=float)
    for _ in range(n_perm):
        shift = int(rng.integers(1, n_time))
        null_vec = pooled_power(np.roll(ersp, shift, axis=1), weights, feature_set, grid=grid)
        ge += (np.abs(null_vec) >= obs).astype(float)
    return (ge + 1.0) / (n_perm + 1.0)


def build_pool_table(df_meta: pd.DataFrame, X_3d: np.ndarray, cfg: dict, *,
                     grid: str = "full",
                     feature_sets: Sequence[str] = FEATURE_SETS,
                     window_shapes: Sequence[str] = WINDOW_SHAPES,
                     n_perm: int = 0, seed: int = 0,
                     cache: bool = True, write_run: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """Pool power + qualify every contact, for each condition x zone x window
    shape x feature set x feature. Returns the tidy table (one row per
    contact·condition·zone·shape·feature) and caches it to parquet.

    grid='full' -> 129x300 ERSPs (the real run). grid='ds' -> 15x30 band-
    downsampled (preliminary/fast). The window weights + gate are built on the
    array's own time axis (300 or 30). NOTE: on 'ds' the sigma/proportion gate
    runs on the smoothed band-mean map — a coarse proxy for the full-res
    clustering gate; use grid='full' for final qualification.
    """
    zones = cfg["zones"]
    n_time = int(X_3d.shape[2])
    # Precompute weights + support per (zone, shape) on THIS grid's time axis.
    weights = {(z, s): make_window_weights(spec, s, n_time) for z, spec in zones.items()
               for s in window_shapes}
    support = {(z, s): window_support(spec, s, n_time) for z, spec in zones.items()
               for s in window_shapes}

    rows: List[dict] = []
    n = len(df_meta)
    for pos_i, row in enumerate(df_meta.itertuples()):
        ersp = X_3d[row.Index]
        feats = {fs: feature_vector(ersp, fs, grid=grid) for fs in feature_sets}  # once per contact
        for z in zones:
            for shape in window_shapes:
                w = weights[(z, shape)]
                wn = w / (w.sum() + 1e-12)
                prop_pos, prop_neg, qpos, qneg = windowed_gate(ersp, support[(z, shape)])
                sign = _sign_of(prop_pos, prop_neg, qpos, qneg)
                qualifies = bool(qpos or qneg)
                for fs in feature_sets:
                    vec = (feats[fs] * wn[None, :]).sum(axis=1)
                    pvals = temporal_null_p(ersp, w, fs, vec, grid=grid, n_perm=n_perm, seed=seed)
                    for k in range(vec.shape[0]):
                        lbl, bidx, blo, bhi = feature_label(fs, k)
                        rows.append({
                            "patient_id": row.patient_id,
                            "electrode": row.electrode,
                            "contact_norm": row.contact_norm,
                            "file_path": row.file_path,
                            "condition": row.condition,
                            "grid": grid,
                            "zone": z,
                            "window_shape": shape,
                            "feature_set": fs,
                            "feature": lbl,
                            "band_idx": bidx,
                            "band_lo": blo,
                            "band_hi": bhi,
                            "pooled_db": float(vec[k]),
                            "prop_above_pos_w": prop_pos,
                            "prop_below_neg_w": prop_neg,
                            "qual_pos": bool(qpos),
                            "qual_neg": bool(qneg),
                            "qualifies": qualifies,
                            "sign": sign,
                            "temporal_null_p": float(pvals[k]),
                        })
        if verbose and (pos_i + 1) % 200 == 0:
            print(f"  pooled {pos_i + 1}/{n} contacts")

    df_pool = pd.DataFrame(rows)
    if cache:
        DATASET_CACHE.mkdir(parents=True, exist_ok=True)
        df_pool.to_parquet(DATASET_CACHE / f"pool_table_{grid}.parquet", index=False)
    if write_run:
        run_dir = new_run_dir("pool", grid)
        df_pool.to_parquet(run_dir / "pool_table.parquet", index=False)
        summ = qualifier_summary(df_pool)
        summ.to_csv(run_dir / "qualifier_summary.csv", index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION, "stage": "pool", "grid": grid,
            "run_id": run_dir.name,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "user": _safe_user(),
            "n_rows": int(len(df_pool)), "n_time": n_time,
            "n_contacts": int(df_meta.shape[0]),
            "feature_sets": list(feature_sets), "window_shapes": list(window_shapes),
            "zones": list(zones), "n_perm": int(n_perm),
            "path": str(run_dir.relative_to(OUTPUTS_ROOT)),
        }
        write_json(run_dir / "manifest.json", manifest)
        update_index(manifest)
        if verbose:
            print(f"[lf_pool] pool run -> {run_dir}")
    if verbose:
        print(f"[lf_pool] pool table: {df_pool.shape}")
    return df_pool


def qualifier_summary(df_pool: pd.DataFrame) -> pd.DataFrame:
    """Counts of QUALIFYING distinct contacts per (condition, zone, window_shape,
    sign). The gate is feature-independent, so collapse over features first."""
    g = (df_pool[df_pool["qualifies"]]
         .drop_duplicates(["patient_id", "contact_norm", "condition", "zone",
                           "window_shape"]))
    out = (g.groupby(["condition", "zone", "window_shape", "sign"])
           .size().reset_index(name="n_contacts"))
    return out.sort_values(["condition", "zone", "window_shape", "sign"]).reset_index(drop=True)


def load_pool_table(grid: str = "full", path: Optional[Path] = None) -> pd.DataFrame:
    """Read the cached pool table for a grid ('full' or 'ds'), or an explicit path."""
    p = Path(path) if path is not None else DATASET_CACHE / f"pool_table_{grid}.parquet"
    return pd.read_parquet(p)


# ============================================================
# 7 — Anatomy mapping (430)
# ============================================================
def ensure_aparc_cache(coords_dir: Path = COORDS_DIR, out_csv: Path = APARC_CACHE,
                       verbose: bool = True) -> pd.DataFrame:
    """Build the Desikan-Killiany (aparc) lookup once (needs MNE + fsaverage,
    ~30 s), else load the cached CSV. Columns: patient, electrode, hemi,
    aparc_label, vertex_idx, x, y, z."""
    out_csv = Path(out_csv)
    if out_csv.exists():
        if verbose:
            print(f"[lf_pool] aparc cache hit: {out_csv}")
        return pd.read_csv(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    glob = str(Path(coords_dir) / "*_contacts_fsaverage.csv")
    return _lf_anatomy.build_aparc_cache(glob, out_csv, verbose=verbose)


def load_coords(coords_dir: Path = COORDS_DIR) -> pd.DataFrame:
    """Per-patient coords (excluding ALL_PATIENTS*) with a `contact_norm` key.
    Columns: patient_id, contact_norm, name, hemi, x, y, z, is_wm, is_cortical,
    yeo7_network, yeo17_network."""
    coords_dir = Path(coords_dir)
    files = sorted(coords_dir.glob("*_contacts_fsaverage.csv"))
    if not files:
        raise FileNotFoundError(f"No coords CSVs under {coords_dir}")
    frames = []
    for f in files:
        if "ALL_PATIENTS" in f.name.upper():
            continue
        df = pd.read_csv(f)
        frames.append(df)
    coords = pd.concat(frames, ignore_index=True)
    coords["patient_id"] = coords["patient"].astype(str)
    coords["contact_norm"] = coords["name"].map(normalize_label)
    return coords


def attach_anatomy(df_qual: pd.DataFrame, coords: pd.DataFrame, aparc: pd.DataFrame,
                   *, n_networks: int = 7, verbose: bool = True) -> pd.DataFrame:
    """Enrich qualifying rows with Yeo-{n} + DK gyrus + fsaverage xyz, joined on
    (patient_id, contact_norm). Reports the unmatched count."""
    yeo_col = "yeo7_network" if int(n_networks) == 7 else "yeo17_network"
    c = (coords[["patient_id", "contact_norm", "hemi", "x", "y", "z",
                 "is_wm", "is_cortical", yeo_col]]
         .rename(columns={yeo_col: "yeo_label"})
         .drop_duplicates(["patient_id", "contact_norm"]))
    out = df_qual.merge(c, on=["patient_id", "contact_norm"], how="left")

    a = aparc.copy()
    a["patient_id"] = a["patient"].astype(str)
    a["contact_norm"] = a["electrode"].map(normalize_label)
    a = a[["patient_id", "contact_norm", "aparc_label"]].drop_duplicates(
        ["patient_id", "contact_norm"])
    out = out.merge(a, on=["patient_id", "contact_norm"], how="left")

    out["yeo_real"] = out["yeo_label"].map(is_real_network)
    if verbose:
        n_contacts = out.drop_duplicates(["patient_id", "contact_norm"]).shape[0]
        n_no_coord = out[out["x"].isna()].drop_duplicates(
            ["patient_id", "contact_norm"]).shape[0]
        n_no_aparc = out[out["aparc_label"].isna()].drop_duplicates(
            ["patient_id", "contact_norm"]).shape[0]
        print(f"[lf_pool] anatomy join (Yeo-{n_networks}): {n_contacts} contacts, "
              f"{n_no_coord} without coords, {n_no_aparc} without aparc")
    return out


def zone_to_cluster_id(df: pd.DataFrame, by: str = "zone"
                       ) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Map zone (or zone×sign) -> a contiguous integer `cluster` column (the
    anatomy summary fns require integer ids). Returns (df_with_cluster, id2name)."""
    df = df.copy()
    if by == "zone":
        key = df["zone"].astype(str)
    elif by == "zone_sign":
        key = df["zone"].astype(str) + "_" + df["sign"].map(
            {1: "pos", -1: "neg", 0: "none"}).astype(str)
    else:
        raise ValueError("by must be 'zone' or 'zone_sign'")
    names = sorted(key.dropna().unique())
    name2id = {k: i for i, k in enumerate(names)}
    df["cluster"] = key.map(name2id).astype(int)
    return df, {i: k for k, i in name2id.items()}


def summarize_anatomy(df_qual: pd.DataFrame, aparc: pd.DataFrame, coords: pd.DataFrame,
                      run_dir, *, by: str = "zone", n_perm: int = 0,
                      verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """Per-zone anatomy purity + spatial compactness for the qualifiers. Each
    zone is one 'cluster'. Reuses lf_anatomy.save_anatomy_artifacts +
    save_spatial_compactness_artifacts. Returns (anatomy_df, compactness_df, id2name)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df_lab, id2name = zone_to_cluster_id(df_qual.dropna(subset=["zone"]), by=by)
    # One row per (patient, contact, cluster) — collapse the per-feature rows.
    df_lab = df_lab.drop_duplicates(["patient_id", "contact_norm", "cluster"])
    # Build minimal frames with NO duplicate column labels. df_qual already carries
    # its own 'electrode' column (and coords its own 'patient'), so renaming
    # contact_norm->electrode / patient_id->patient in place would create DUPLICATE
    # labels -> df[col] then returns a DataFrame (not a Series), which breaks the
    # anatomy helpers. The anatomy fns default to patient_col='patient_id',
    # electrode_col='electrode'; the contact key is the normalized contact name.
    labels = pd.DataFrame({
        "patient_id": df_lab["patient_id"].astype(str),
        "electrode":  df_lab["contact_norm"].astype(str),
        "cluster":    df_lab["cluster"].astype(int),
    })
    ap = pd.DataFrame({
        "patient":     aparc["patient"].astype(str),
        "electrode":   aparc["electrode"].map(normalize_label),
        "aparc_label": aparc["aparc_label"].astype(str),
    })
    co = pd.DataFrame({
        "patient":   coords["patient_id"].astype(str),
        "electrode": coords["contact_norm"].astype(str),
        "x": coords["x"], "y": coords["y"], "z": coords["z"],
    })

    anat_df, _ = _lf_anatomy.save_anatomy_artifacts(
        run_dir, labels, ap, cluster_col="cluster", n_perm=n_perm, verbose=verbose)
    comp_df, _ = _lf_anatomy.save_spatial_compactness_artifacts(
        run_dir, labels, co, cluster_col="cluster", verbose=verbose)

    # Attach the readable zone name back onto the integer cluster_id.
    anat_df = anat_df.copy(); anat_df["zone"] = anat_df["cluster_id"].map(id2name)
    comp_df = comp_df.copy(); comp_df["zone"] = comp_df["cluster_id"].map(id2name)
    anat_df.to_csv(run_dir / "per_cluster_anatomy.csv", index=False)
    comp_df.to_csv(run_dir / "per_cluster_spatial_compactness.csv", index=False)
    write_json(run_dir / "cluster_id_map.json", id2name)
    return anat_df, comp_df, id2name


def render_zone_brains(df_qual: pd.DataFrame, run_dir, *, condition: str, zone: str,
                       window_shape: str = "boxcar", dpi_scale: int = 2) -> List[Path]:
    """Render the qualifying contacts of one (condition, zone, window_shape) on
    the fsaverage pial surface, coloured red(+) / blue(−). Self-contained
    PyVista render (mirrors lf_recon_shared's load_fsaverage_meshes /
    compute_cameras / add_brain_mesh / add_electrodes primitives) so it doesn't
    depend on the notebook-scoped render_views wrapper. Requires pyvista +
    nibabel + the fsaverage pial surfaces; set PYVISTA_OFF_SCREEN before import.
    """
    import pyvista as pv
    from nibabel.freesurfer.io import read_geometry

    C = _ensure_recon_cfg()
    sub = df_qual[(df_qual["condition"] == condition) & (df_qual["zone"] == zone)
                  & (df_qual["window_shape"] == window_shape) & (df_qual["qualifies"])]
    sub = sub.dropna(subset=["x", "y", "z"]).drop_duplicates(
        ["patient_id", "contact_norm"])
    out_dir = Path(run_dir) / "renders" / condition / f"{zone}_{window_shape}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if sub.empty:
        print(f"[lf_pool] no coord-resolved qualifiers for {condition}/{zone}/{window_shape}")
        return []

    def _to_pv(path):
        v, f = read_geometry(str(path))
        faces = np.hstack([np.full((f.shape[0], 1), 3, dtype=np.int64), f]).ravel()
        m = pv.PolyData(v, faces)
        m.compute_normals(inplace=True)
        return m

    lh = _to_pv(C.FSAVERAGE_DIR / "surf" / "lh.pial")
    rh = _to_pv(C.FSAVERAGE_DIR / "surf" / "rh.pial")
    bounds = (min(lh.bounds[0], rh.bounds[0]), max(lh.bounds[1], rh.bounds[1]),
              min(lh.bounds[2], rh.bounds[2]), max(lh.bounds[3], rh.bounds[3]),
              min(lh.bounds[4], rh.bounds[4]), max(lh.bounds[5], rh.bounds[5]))
    cx, cy, cz = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2)
    d = 2.4 * max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    cams = {
        "left": ((cx - d, cy, cz), (cx, cy, cz), (0, 0, 1)),
        "right": ((cx + d, cy, cz), (cx, cy, cz), (0, 0, 1)),
        "frontal": ((cx, cy + d, cz), (cx, cy, cz), (0, 0, 1)),
        "posterior": ((cx, cy - d, cz), (cx, cy, cz), (0, 0, 1)),
        "dorsal": ((cx, cy, cz + d), (cx, cy, cz), (0, 1, 0)),
        "ventral": ((cx, cy, cz - d), (cx, cy, cz), (0, 1, 0)),
    }
    colors = {1: (0.80, 0.0, 0.20), -1: (0.0, 0.20, 0.80), 0: (0.5, 0.5, 0.5)}

    written: List[Path] = []
    for view, cam in cams.items():
        pl = pv.Plotter(off_screen=True, window_size=getattr(C, "WINDOW_SIZE", (1200, 1000)))
        for m in (lh, rh):
            pl.add_mesh(m, color=getattr(C, "BRAIN_COLOR", "#ead6db"),
                        opacity=getattr(C, "BRAIN_OPACITY_CLEAN", 0.25),
                        smooth_shading=True)
        for _, r in sub.iterrows():
            rgb = colors.get(int(r["sign"]), colors[0])
            sph = pv.Sphere(radius=getattr(C, "DEPTH_RADIUS", 1.5),
                            center=(float(r["x"]), float(r["y"]), float(r["z"])),
                            theta_resolution=18, phi_resolution=18)
            pl.add_mesh(sph, color=rgb, opacity=1.0)
        pl.camera_position = cam
        pl.reset_camera_clipping_range()
        out_png = out_dir / f"{view}.png"
        pl.screenshot(str(out_png), transparent_background=True, scale=dpi_scale)
        pl.close()
        written.append(out_png)
    print(f"[lf_pool] rendered {len(written)} views for {condition}/{zone}/{window_shape} "
          f"({len(sub)} contacts)")
    return written


# ============================================================
# 8 — Run-dir / index plumbing (mirrors lf_classify)
# ============================================================
def _now_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def new_run_dir(*parts: str) -> Path:
    rd = OUTPUTS_ROOT.joinpath(*parts, "runs", _now_id())
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def update_index(manifest: dict) -> None:
    idx_path = OUTPUTS_ROOT / "index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    else:
        idx = {"schema_version": SCHEMA_VERSION, "runs": []}
    keep = ("stage", "run_id", "created_at", "path")
    row = {k: manifest.get(k) for k in keep}
    for extra in ("n_rows", "n_contacts", "n_perm"):
        if extra in manifest:
            row[extra] = manifest[extra]
    idx["runs"] = [r for r in idx["runs"] if r.get("path") != row["path"]] + [row]
    write_json(idx_path, idx)


def list_runs() -> pd.DataFrame:
    idx_path = OUTPUTS_ROOT / "index.json"
    if not idx_path.exists():
        return pd.DataFrame()
    runs = json.loads(idx_path.read_text()).get("runs", [])
    df = pd.DataFrame(runs)
    if len(df):
        df = df.sort_values("created_at").reset_index(drop=True)
    return df


def latest_run(*parts: str) -> Optional[Path]:
    base = OUTPUTS_ROOT.joinpath(*parts, "runs")
    if not base.exists():
        return None
    runs = sorted([d for d in base.iterdir() if d.is_dir()])
    return runs[-1] if runs else None
