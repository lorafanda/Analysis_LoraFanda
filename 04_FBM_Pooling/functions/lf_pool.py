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
_lf_blob_cfg = _load_module(_CLUST_FUNCS, "lf_blob_clustering_config")  # VALLEY_PARAMS source

# Lazy: heavier / optional. Blob segmentation pulls scipy.ndimage; the recon
# config is only needed for the 430 brain renders.
_lf_blob = None
_lf_minus101 = None
_recon_cfg = None


def _ensure_blob():
    global _lf_blob
    if _lf_blob is None:
        _lf_blob = _load_module(_CLUST_FUNCS, "lf_blob_metrics")
    return _lf_blob


def _ensure_minus101():
    global _lf_minus101
    if _lf_minus101 is None:
        _lf_minus101 = _load_module(_CLUST_FUNCS, "lf_minus101")
    return _lf_minus101


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

# Role-box "on" gate: a role box counts as expressing its sign only if AT LEAST
# this fraction of its discretized cells are the expected sign (+1 for pos, -1 for
# neg). Far stricter than the clustering presence gate above (a handful of cells is
# not enough). "Silent"/zero boxes still use the strict MIN_PROP_* presence gate.
ROLE_BOX_MIN_PROP = 0.3

FREQ_BANDS = list(_lf_features.FREQ_BANDS_15_TO_400HZ)   # 15 (lo, hi) Hz tuples
FEATURE_SETS = ("hg", "bands15")
WINDOW_SHAPES = ("boxcar", "gaussian")
DEFAULT_ZONES = ("perception", "pre_articulation", "audio")
GAUSS_SUPPORT_SIGMA = 2.0             # gate support = center +/- this many sigma

# 410 overlay blob-score gate: drop blobs below this percentile of the dataset's
# blob scores (same idea as clustering/classification's M101_SCORE_PCT). Tunable.
SCORE_PCT = 33.0
# EXACT clustering discretization (02's 231 minus101 path): segment with the SAME
# VALLEY_PARAMS the clustering used, and NO score gate (231 ran score_min=None).
# Sourced live from lf_blob_clustering_config so it stays in sync with 02. sign_mode
# is dropped here because discretize_ersp passes sign_mode="both" explicitly.
# => thr_pos 2.5, thr_neg -3.5, delta_valley 1.5, min_mean_pos 1.75 (0.7*thr),
#    max_mean_neg -2.8 (0.8*thr), max_blobs 6.
CLUSTERING_SEG_KWARGS = {k: v for k, v in dict(_lf_blob_cfg.VALLEY_PARAMS).items()
                         if k != "sign_mode"}
# Default for BOTH 460 matching and 470 viz = the clustering procedure, so the pooled
# roles are judged on a clustering-comparable -1/0/+1 map. Override per-call via
# seg_kwargs (merged over this).
DEFAULT_SEG_KWARGS = dict(CLUSTERING_SEG_KWARGS)
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
# 7b — Region x time cascade raster (thesis figure)
# ============================================================
def region_labels(df_meta: pd.DataFrame, coords: pd.DataFrame,
                  aparc: Optional[pd.DataFrame] = None, *, scheme: str = "yeo7"
                  ) -> pd.Series:
    """Per-contact region label, index-aligned to df_meta. scheme:
    'yeo7' | 'yeo17' (from the coords CSVs) | 'aparc' (Desikan-Killiany gyrus,
    from the aparc cache). Non-real networks / unknown / unmatched -> NaN
    (dropped from the cascade)."""
    key = ["patient_id", "contact_norm"]
    if scheme in ("yeo7", "yeo17"):
        col = "yeo7_network" if scheme == "yeo7" else "yeo17_network"
        lut = (coords[key + [col]].drop_duplicates(key).rename(columns={col: "region"}))
        lut["region"] = lut["region"].where(lut["region"].map(is_real_network))
    elif scheme == "aparc":
        if aparc is None:
            raise ValueError("scheme='aparc' needs the aparc cache (ensure_aparc_cache)")
        lut = pd.DataFrame({
            "patient_id": aparc["patient"].astype(str),
            "contact_norm": aparc["electrode"].map(normalize_label),
            "region": aparc["aparc_label"].astype(str),
        }).drop_duplicates(key)
        bad = {"unknown", "nan", "", "none"}
        lut["region"] = lut["region"].where(~lut["region"].str.lower().isin(bad))
    else:
        raise ValueError("scheme must be 'yeo7', 'yeo17', or 'aparc'")
    merged = df_meta[key].merge(lut, on=key, how="left")
    return merged["region"]


def responsive_contacts(df_pool: pd.DataFrame) -> set:
    """Set of (patient_id, contact_norm) qualifying in >=1 zone (any shape)."""
    q = df_pool[df_pool["qualifies"]]
    return set(zip(q["patient_id"].astype(str), q["contact_norm"].astype(str)))


def build_cascade(df_meta: pd.DataFrame, X: np.ndarray, region_series: pd.Series, *,
                  grid: str = "full", feature: str = "hg",
                  conditions: Sequence[str] = CONDITIONS, min_contacts: int = 5,
                  qualifies: Optional[set] = None, sort_by: Optional[str] = None) -> dict:
    """Mean `feature` time-course per anatomical region (rows) x warped time (cols),
    one matrix per condition, with rows ordered by peak latency so the cascade
    reads top (early) -> bottom (late). No spatial interpolation — each row is a
    real average over that region's contacts. `feature`='hg' (the HG line) by
    default; `qualifies` restricts to the responsive set."""
    n_time = int(X.shape[2])
    time_pct = np.linspace(0, 100, n_time)
    reg = region_series.to_numpy()
    cond_arr = df_meta["condition"].to_numpy()
    pid = df_meta["patient_id"].astype(str).to_numpy()
    con = df_meta["contact_norm"].astype(str).to_numpy()

    acc: Dict[str, Dict[str, list]] = {c: {} for c in conditions}
    seen: Dict[str, set] = {}                       # region -> unique (pid, contact)
    for i in range(len(df_meta)):
        r = reg[i]
        if r is None or (isinstance(r, float) and np.isnan(r)):
            continue
        c = cond_arr[i]
        if c not in acc:
            continue
        if qualifies is not None and (pid[i], con[i]) not in qualifies:
            continue
        trace = feature_vector(X[i], feature, grid=grid)[0]          # (n_time,)
        d = acc[c].setdefault(r, [np.zeros(n_time), 0])
        d[0] += trace
        d[1] += 1
        seen.setdefault(r, set()).add((pid[i], con[i]))

    region_ok = sorted({r for c in conditions for r, (s, n) in acc[c].items()
                        if n >= min_contacts})
    matrices, counts = {}, {}
    for c in conditions:
        M = np.full((len(region_ok), n_time), np.nan)
        for j, r in enumerate(region_ok):
            if r in acc[c] and acc[c][r][1] >= min_contacts:
                M[j] = acc[c][r][0] / acc[c][r][1]
        matrices[c] = M
        counts[c] = {r: (acc[c][r][1] if r in acc[c] else 0) for r in region_ok}

    # order rows by peak latency (reference condition if given, else mean across conds)
    lat = []
    for j in range(len(region_ok)):
        tr = (matrices[sort_by][j] if sort_by in conditions
              else np.nanmean(np.vstack([matrices[c][j] for c in conditions]), axis=0))
        lat.append(np.inf if np.all(np.isnan(tr))
                   else time_pct[int(np.nanargmax(np.where(np.isnan(tr), -np.inf, tr)))])
    order = np.argsort(lat)
    region_order = [region_ok[k] for k in order]
    return {
        "conditions": list(conditions),
        "time_pct": time_pct,
        "regions": region_order,
        "peak_latency_pct": [float(lat[k]) for k in order],
        "n_total": {r: len(seen.get(r, ())) for r in region_order},
        "matrices": {c: matrices[c][order] for c in conditions},
        "counts": {c: counts[c] for c in conditions},
        "grid": grid, "feature": feature,
    }


def cascade_table(cascade: dict) -> pd.DataFrame:
    """Tidy companion table: region, n_total, peak-latency (%) per condition."""
    rows = []
    for j, r in enumerate(cascade["regions"]):
        rec = {"region": r, "n_total": cascade["n_total"][r],
               "peak_latency_pct": round(cascade["peak_latency_pct"][j], 1)}
        for c in cascade["conditions"]:
            tr = cascade["matrices"][c][j]
            rec[f"peak_{c}_pct"] = (float("nan") if np.all(np.isnan(tr))
                                    else round(float(cascade["time_pct"][
                                        int(np.nanargmax(np.where(np.isnan(tr), -np.inf, tr)))]), 1))
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_cascade(cascade: dict, out_png, *, cfg: Optional[dict] = None,
                 vlim: Optional[Tuple[float, float]] = None, dpi: int = 150) -> Path:
    """Render the region x time cascade: one heatmap panel per condition, rows
    shared + sorted by peak latency, RdBu_r centred at 0, response-onset line at
    50%, and (if cfg given) the a-priori zones shaded behind it."""
    import matplotlib.pyplot as plt
    conds, regions, t = cascade["conditions"], cascade["regions"], cascade["time_pct"]
    mats, n_total = cascade["matrices"], cascade["n_total"]
    if not regions:
        raise ValueError("no regions met min_contacts — lower min_contacts or check the join")
    allv = np.concatenate([m[~np.isnan(m)].ravel() for m in mats.values() if m.size]) \
        if any(m.size for m in mats.values()) else np.array([1.0])
    if vlim is None:
        a = float(np.nanpercentile(np.abs(allv), 99)) or 1.0
        vlim = (-a, a)
    from matplotlib.transforms import blended_transform_factory
    fig, axes = plt.subplots(1, len(conds), squeeze=False, sharey=True,
                             figsize=(3.8 * len(conds) + 0.6, 0.34 * len(regions) + 1.9),
                             constrained_layout=True)   # lets suptitle/titles self-space
    im = None
    for ax, c in zip(axes[0], conds):
        btf = blended_transform_factory(ax.transData, ax.transAxes)
        if cfg:
            for zname, spec in cfg["zones"].items():
                b = spec["boxcar"]
                ax.axvspan(b["t_lo_pct"], b["t_hi_pct"], color="0.5", alpha=0.10, zorder=0)
                ax.text((b["t_lo_pct"] + b["t_hi_pct"]) / 2, 0.995, zname, transform=btf,
                        ha="center", va="top", fontsize=7, color="0.2", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.6))
        im = ax.imshow(mats[c], aspect="auto", origin="upper", cmap="RdBu_r",
                       vmin=vlim[0], vmax=vlim[1], extent=[0, 100, len(regions), 0],
                       interpolation="nearest", zorder=1)
        ax.axvline(STIM_FRAC * 100, color="k", lw=1.0, ls="--", zorder=2)
        # per-region peak-latency marker -> makes the top->bottom cascade explicit
        m = mats[c]
        for i in range(len(regions)):
            row = m[i]
            if not row.size or np.all(np.isnan(row)):
                continue
            j = int(np.nanargmax(np.abs(row)))
            ax.plot((j + 0.5) / row.size * 100.0, i + 0.5, marker="o", ms=3,
                    mfc="k", mec="white", mew=0.5, zorder=3)
        ax.set_title(c)
        ax.set_xlabel("warped time (%)  (50 = response onset)")
    axes[0][0].set_yticks(np.arange(len(regions)) + 0.5)
    axes[0][0].set_yticklabels([f"{r}  (n={n_total[r]})" for r in regions], fontsize=9)
    axes[0][0].tick_params(axis="y", pad=4)
    fig.colorbar(im, ax=axes[0].tolist(), label=f"mean {cascade['feature']} power (dB)",
                 fraction=0.025, pad=0.02)
    fig.suptitle(f"Region × time cascade — {cascade['feature']} "
                 f"(rows sorted by peak latency · dot = peak · grid={cascade['grid']})")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)          # constrained_layout handles spacing (no bbox_inches)
    plt.close(fig)
    return out_png


# ============================================================
# 7c — Predefined time-frequency ROI pooling (cluster/condition-blind)
# ============================================================
# The same fixed ROI library is applied to EVERY electrode x condition ERSP,
# independent of the 02 clusters and the condition label. Each ROI is a 2-D box
# (1-based, inclusive f_rows x t_bins) + a sign hypothesis. Pooling = mean dB in
# the box; qualification = the box clears the clustering sigma/proportion gate in
# the ROI's OWN sign. Option A: a contact "expresses" an ROI if it qualifies in
# >=1 condition. See functions/roi_config.py.
def _load_roi_config():
    path = _HERE.parent / "roi_config.py"
    spec = importlib.util.spec_from_file_location("_pool_roi_config", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)               # type: ignore[union-attr]
    return m.ERSP_POOLING_PARAMS


ROI_PARAMS = _load_roi_config()
_ROI_SIGN_COLOR = {"pos": "#cc0033", "neg": "#1f4fd8", "both": "#7a1fa2"}


def roi_tag(i: int) -> str:
    return f"({chr(97 + i)})"                 # 0 -> (a), 1 -> (b), ...


def validate_roi_config(roi_params: Optional[dict] = None) -> pd.DataFrame:
    """Check ds<->full Hz / time consistency for each ROI and flag mismatches.
    The ds/full numbers are YOUR predeterminants — this only surfaces where the
    two grids disagree (e.g. alpha_beta, broadband) so you can reconcile them."""
    rp = roi_params or ROI_PARAMS
    ds = {r["label"]: r for r in rp["ds"]}
    full = {r["label"]: r for r in rp["full"]}
    rows = []
    for lab, d in ds.items():
        d_lo = FREQ_BANDS[d["f_rows"][0] - 1][0]
        d_hi = FREQ_BANDS[d["f_rows"][1] - 1][1]
        rec = {"label": lab, "ds_hz": (d_lo, d_hi),
               "ds_t_pct": (round((d["t_bins"][0] - 1) / 30 * 100), round(d["t_bins"][1] / 30 * 100)),
               "ds_sign": d["sign"]}
        f = full.get(lab)
        if f is not None:
            f_lo = (f["f_rows"][0] - 1) / (N_FREQ - 1) * FMAX_HZ
            f_hi = (f["f_rows"][1] - 1) / (N_FREQ - 1) * FMAX_HZ
            rec.update({"full_hz": (round(f_lo), round(f_hi)),
                        "full_t_pct": (round((f["t_bins"][0] - 1) / N_TIME * 100),
                                       round(f["t_bins"][1] / N_TIME * 100)),
                        "full_sign": f["sign"],
                        "hz_mismatch": (abs(d_lo - f_lo) > 8) or (abs(d_hi - f_hi) > 8),
                        "sign_mismatch": d["sign"] != f["sign"]})
        rows.append(rec)
    return pd.DataFrame(rows)


def _roi_box(ersp: np.ndarray, roi: dict) -> np.ndarray:
    """The (f_rows x t_bins) sub-block. 1-based inclusive -> 0-based slice."""
    f0, f1 = roi["f_rows"]
    t0, t1 = roi["t_bins"]
    return ersp[f0 - 1:f1, t0 - 1:t1]


def pool_roi(ersp: np.ndarray, roi: dict) -> float:
    """Mean dB inside the ROI box."""
    box = _roi_box(ersp, roi)
    return float(box.mean()) if box.size else float("nan")


def roi_gate(ersp: np.ndarray, roi: dict) -> Tuple[float, float, bool]:
    """Sign-directional windowed gate over the box. Returns
    (prop_pos, prop_neg, qualifies) using the SAME sigma thresholds as clustering."""
    box = _roi_box(ersp, roi)
    if box.size == 0:
        return 0.0, 0.0, False
    prop_pos = float((box > THR_POS).mean())
    prop_neg = float((box < THR_NEG).mean())
    s = roi.get("sign", "both")
    if s == "pos":
        q = prop_pos >= MIN_PROP_POS
    elif s == "neg":
        q = prop_neg >= MIN_PROP_NEG
    else:
        q = (prop_pos >= MIN_PROP_POS) or (prop_neg >= MIN_PROP_NEG)
    return prop_pos, prop_neg, bool(q)


def build_roi_table(df_meta: pd.DataFrame, X: np.ndarray, *, grid: str = "full",
                    roi_params: Optional[dict] = None, cache: bool = True,
                    write_run: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Pool + sign-gate every electrode x condition ERSP against every ROI
    (condition-blind, cluster-blind). One row per (contact, condition, ROI)."""
    rp = roi_params or ROI_PARAMS
    rois = rp[grid]
    rows = []
    for row in df_meta.itertuples():
        ersp = X[row.Index]
        for i, roi in enumerate(rois):
            pp, pn, q = roi_gate(ersp, roi)
            rows.append({
                "patient_id": row.patient_id, "electrode": row.electrode,
                "contact_norm": row.contact_norm, "condition": row.condition,
                "grid": grid, "roi_tag": roi_tag(i), "roi_label": roi["label"],
                "sign": roi["sign"],
                "f_lo_row": roi["f_rows"][0], "f_hi_row": roi["f_rows"][1],
                "t_lo_bin": roi["t_bins"][0], "t_hi_bin": roi["t_bins"][1],
                "pooled_db": pool_roi(ersp, roi),
                "prop_pos": pp, "prop_neg": pn, "qualifies": q,
            })
    df = pd.DataFrame(rows)
    if cache:
        DATASET_CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATASET_CACHE / f"roi_table_{grid}.parquet", index=False)
    if write_run:
        run_dir = new_run_dir("roi", grid)
        df.to_parquet(run_dir / "roi_table.parquet", index=False)
        roi_counts(df).to_csv(run_dir / "roi_counts.csv", index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION, "stage": "roi", "grid": grid,
            "run_id": run_dir.name,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "user": _safe_user(),
            "n_rows": int(len(df)), "n_contacts": int(df_meta.shape[0]),
            "n_rois": len(rois), "path": str(run_dir.relative_to(OUTPUTS_ROOT)),
        }
        write_json(run_dir / "manifest.json", manifest)
        update_index(manifest)
        if verbose:
            print(f"[lf_pool] roi run -> {run_dir}")
    if verbose:
        print(f"[lf_pool] roi table: {df.shape}  ({len(rois)} ROIs x {df_meta.shape[0]} samples)")
    return df


def load_roi_table(grid: str = "full", path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DATASET_CACHE / f"roi_table_{grid}.parquet"
    return pd.read_parquet(p)


def roi_contact_summary(df_roi: pd.DataFrame) -> pd.DataFrame:
    """Option A: a contact 'expresses' an ROI if it qualifies in >=1 condition.
    One row per (contact, ROI)."""
    return (df_roi.groupby(["patient_id", "contact_norm", "roi_tag", "roi_label", "sign"],
                           as_index=False)
            .agg(expresses=("qualifies", "any"),
                 n_cond_qual=("qualifies", "sum"),
                 pooled_db_mean=("pooled_db", "mean")))


def roi_counts(df_roi: pd.DataFrame) -> pd.DataFrame:
    """Distinct contacts expressing each ROI (option A)."""
    s = roi_contact_summary(df_roi)
    s = s[s["expresses"]]
    out = (s.groupby(["roi_tag", "roi_label", "sign"]).size()
           .reset_index(name="n_contacts"))
    return out.sort_values("roi_tag").reset_index(drop=True)


def roi_region_crosstab(df_roi: pd.DataFrame, coords: pd.DataFrame, *,
                        scheme: str = "yeo7") -> pd.DataFrame:
    """ROI (rows) x anatomical region (cols) counts of expressing contacts."""
    s = roi_contact_summary(df_roi)
    s = s[s["expresses"]]
    col = "yeo7_network" if scheme == "yeo7" else "yeo17_network"
    lut = (coords[["patient_id", "contact_norm", col]]
           .drop_duplicates(["patient_id", "contact_norm"]).rename(columns={col: "region"}))
    lut = lut[lut["region"].map(is_real_network)]
    m = s.merge(lut, on=["patient_id", "contact_norm"], how="left").dropna(subset=["region"])
    m["roi"] = m["roi_tag"] + " " + m["roi_label"]
    return pd.crosstab(m["roi"], m["region"])


def roi_legend(roi_params: Optional[dict] = None, *, grid: str = "ds") -> pd.DataFrame:
    """Letter -> ROI legend table for the figure."""
    rp = roi_params or ROI_PARAMS
    return pd.DataFrame([
        {"tag": roi_tag(i), "label": r["label"], "sign": r["sign"],
         "t_bins": tuple(r["t_bins"]), "f_rows": tuple(r["f_rows"]),
         "description": r["description"]}
        for i, r in enumerate(rp[grid])])


def plot_roi_map(roi_params: Optional[dict] = None, *, grid: str = "ds",
                 background: Optional[np.ndarray] = None,
                 signs: Sequence[str] = ("pos", "neg", "both"),
                 out_png=None, dpi: int = 150, ax=None):
    """Draw each ROI as a labelled (a)/(b)/... box on the time-frequency plane,
    coloured by sign (red=pos, blue=neg, purple=both). Optional `background` =
    a (n_freq, n_time) mean ERSP to lay the boxes over. Plot pos / neg on separate
    panels (via `signs`) to separate same-box opposite-sign ROIs."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rp = roi_params or ROI_PARAMS
    rois = rp[grid]
    n_freq, n_time, stim = (15, 30, 15) if grid == "ds" else (N_FREQ, N_TIME, int(STIM_FRAC * N_TIME))
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    if background is not None:
        a = float(np.nanpercentile(np.abs(background), 99)) or 1.0
        ax.imshow(background, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-a, vmax=a,
                  extent=[0.5, n_time + 0.5, 0.5, n_freq + 0.5], alpha=0.55)
    for i, r in enumerate(rois):
        if r["sign"] not in signs:
            continue
        t0, t1 = r["t_bins"]; f0, f1 = r["f_rows"]; c = _ROI_SIGN_COLOR.get(r["sign"], "#444")
        ax.add_patch(Rectangle((t0 - 0.5, f0 - 0.5), (t1 - t0) + 1, (f1 - f0) + 1,
                     fill=True, facecolor=c, alpha=0.15, edgecolor=c, lw=1.8))
        ax.text((t0 + t1) / 2, (f0 + f1) / 2, roi_tag(i), ha="center", va="center",
                fontsize=11, fontweight="bold", color=c)
    ax.axvline(stim + 0.5, color="k", ls="--", lw=1.0)
    ax.set_xlim(0.5, n_time + 0.5); ax.set_ylim(0.5, n_freq + 0.5)
    ax.set_xlabel(f"time bin (1..{n_time};  {stim} = response onset)")
    if grid == "ds":
        ax.set_yticks(np.arange(1, n_freq + 1))
        ax.set_yticklabels([f"{int(lo)}-{int(hi)}" for lo, hi in FREQ_BANDS], fontsize=6)
        ax.set_ylabel("freq band (Hz)")
    else:
        ax.set_ylabel("freq row (1..129, 0-500 Hz)")
    ax.set_title(f"Predefined time-frequency ROIs — {grid}   (red=pos · blue=neg · purple=both)")
    if out_png is not None:
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(out_png, dpi=dpi, bbox_inches="tight")
    return ax


# ---- ROI boxes over ONE real ERSP (validation / figure for named exemplars) ----
_RX_ERSP_NAME = re.compile(r"^(?P<pid>.+?)_(?P<cond>audio|picture|reading)_", re.IGNORECASE)


def resolve_ersp_npy(input_dir, name: str, *, task: str = TASK) -> Optional[Path]:
    """Resolve a contact's ERSP .npy from a PNG/contact name like
    'EL045_reading_WM_ERSP_pH_L11_TN_CLEAN.png' ->
    <input_dir>/EL045/LM/ERSP_matrix/reading/EL045_reading_WM_ERSP_pH_L11_TN.npy.
    Falls back to a glob on the electrode tail. Returns Path or None."""
    base = Path(str(name)).name
    for suf in ("_CLEAN.png", ".png", "_CLEAN.npy", ".npy"):
        if base.lower().endswith(suf.lower()):
            base = base[:-len(suf)]
            break
    m = _RX_ERSP_NAME.match(base)
    if not m:
        return None
    pid, cond = m.group("pid"), m.group("cond").lower()
    d = Path(input_dir) / pid / task / "ERSP_matrix" / cond
    cand = d / f"{base}.npy"
    if cand.exists():
        return cand
    if d.exists():
        for pat in (f"{base}*.npy", f"*{base.split('_ERSP_')[-1]}*.npy"):
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
    return None


def load_named_ersp(input_dir, name: str, *, task: str = TASK) -> Tuple[np.ndarray, Path]:
    """Load the (n_freq, n_time) ERSP for a named exemplar contact."""
    p = resolve_ersp_npy(input_dir, name, task=task)
    if p is None:
        raise FileNotFoundError(f"no ERSP .npy resolved for '{name}' under {input_dir}")
    return np.load(p), p


ERSP_VLIM = 6.5      # dB display clamp — matches the ERSP_clean PNGs (cmap='bwr', +/-6.5)
ERSP_CMAP = "bwr"    # blue = negative, red = positive (same as 01's save_clean_png)


def _set_hz_yaxis(ax, n_freq, fmax=FMAX_HZ):
    """Label the y-axis in Hz while the data stays in row coords (boxes stay aligned)."""
    hzs = [h for h in (0, 100, 200, 300, 400, 500) if h <= fmax]
    ax.set_yticks([h / fmax * (n_freq - 1) + 1 for h in hzs])
    ax.set_yticklabels([str(h) for h in hzs])
    ax.set_ylabel("frequency (Hz)")


def _set_pct_xaxis(ax, nt_block, n_blocks=1):
    """Label the x-axis as % time per block (0–100), with boundaries deduped."""
    ticks, labels = [], []
    for i in range(n_blocks):
        for pct in ((0, 50, 100) if i == 0 else (50, 100)):
            ticks.append(i * nt_block + pct / 100 * nt_block); labels.append(str(pct))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("% time" + ("  ([audio | picture | reading];  50 = response onset)"
                              if n_blocks > 1 else "  (50 = response onset)"))


def plot_roi_on_ersp(ersp: np.ndarray, *, grid: str = "full", roi_params: Optional[dict] = None,
                     signs: Sequence[str] = ("pos", "neg", "both"), label: str = "",
                     vlim: Optional[float] = None, out_png=None, dpi: int = 150, ax=None):
    """Overlay the roi_config 2-D ROI boxes on ONE real ERSP — the ERSP at full
    contrast, ROIs as labelled (a)/(b)/... outlines coloured by sign. Validates
    that each box lands on real activity. grid='full' overlays a 129x300 .npy
    directly; 'ds' expects a 15x30 band-downsampled map."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rp = roi_params or ROI_PARAMS
    rois = rp[grid]
    ersp = np.asarray(ersp)
    if grid == "ds":
        n_freq, n_time, stim = 15, 30, 15
    else:
        n_freq, n_time = ersp.shape
        stim = int(STIM_FRAC * n_time)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    a = float(vlim) if vlim is not None else ERSP_VLIM
    ax.imshow(ersp, aspect="auto", origin="lower", cmap=ERSP_CMAP, vmin=-a, vmax=a,
              extent=[0.5, n_time + 0.5, 0.5, n_freq + 0.5])
    for i, r in enumerate(rois):
        if r["sign"] not in signs:
            continue
        t0, t1 = r["t_bins"]; f0, f1 = r["f_rows"]; c = _ROI_SIGN_COLOR.get(r["sign"], "#444")
        ax.add_patch(Rectangle((t0 - 0.5, f0 - 0.5), (t1 - t0) + 1, (f1 - f0) + 1,
                     fill=False, edgecolor=c, lw=2.0))
        ax.text(t0 - 0.5 + 1, f1 + 0.5 - 1, roi_tag(i), ha="left", va="top",
                fontsize=9, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.12", fc=c, ec="none", alpha=0.9))
    ax.axvline(stim + 0.5, color="k", ls="--", lw=1.0)
    ax.set_xlim(0.5, n_time + 0.5); ax.set_ylim(0.5, n_freq + 0.5)
    _set_pct_xaxis(ax, n_time, 1); _set_hz_yaxis(ax, n_freq)
    ax.set_title(f"{label}   ROIs on ERSP  (red=pos · blue=neg · purple=both)")
    if out_png is not None:
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(out_png, dpi=dpi, bbox_inches="tight")
    return ax


# ---- CONCATENATED [audio|picture|reading] ERSP + the concatenated role ROIs ----
def _contact_id_from_name(name: str) -> Optional[Tuple[str, str]]:
    """(pid, mid) from a name like 'EL045_reading_WM_ERSP_pH_L11_TN_CLEAN.png'
    -> ('EL045', 'WM_ERSP_pH_L11_TN'); the mid is condition-independent."""
    base = Path(str(name)).name
    for suf in ("_CLEAN.png", ".png", "_CLEAN.npy", ".npy"):
        if base.lower().endswith(suf.lower()):
            base = base[:-len(suf)]
            break
    m = re.match(r"^(?P<pid>.+?)_(?P<cond>audio|picture|reading)_(?P<mid>.+)$", base, re.IGNORECASE)
    return (m.group("pid"), m.group("mid")) if m else None


def _concat_block_paths(input_dir, name: str, *, conditions: Sequence[str] = CONDITIONS,
                        task: str = TASK) -> list:
    """The .npy path for each condition of the contact in `name` (all required)."""
    parsed = _contact_id_from_name(name)
    if parsed is None:
        raise ValueError(f"can't parse a contact from '{name}'")
    pid, mid = parsed
    paths = []
    for cond in conditions:
        d = Path(input_dir) / pid / task / "ERSP_matrix" / cond
        p = d / f"{pid}_{cond}_{mid}.npy"
        if not p.exists() and d.exists():
            hits = sorted(d.glob(f"*{mid.split('_ERSP_')[-1]}*.npy"))
            p = hits[0] if hits else p
        if not p.exists():
            raise FileNotFoundError(f"{pid} · {cond} · {mid}: no .npy")
        paths.append(p)
    return paths


def load_concat_ersp(input_dir, name: str, *, conditions: Sequence[str] = CONDITIONS,
                     task: str = TASK) -> np.ndarray:
    """Concatenated [audio|picture|reading] ERSP (n_freq, 3*n_time) for the contact in
    `name`. Requires ALL conditions to exist for that contact (else FileNotFoundError)."""
    paths = _concat_block_paths(input_dir, name, conditions=conditions, task=task)
    return np.concatenate([np.load(p) for p in paths], axis=1)


def load_concat_discretized(input_dir, name: str, *, score_min: Optional[float] = None,
                            grid: str = "full", conditions: Sequence[str] = CONDITIONS,
                            task: str = TASK, seg_kwargs: Optional[dict] = None) -> np.ndarray:
    """Concatenated score-gated -1/0/+1 map for a named contact — exactly what the role
    matching operates on. `seg_kwargs` loosens/tightens the blob segmentation
    (e.g. {'thr_pos':1.5,'thr_neg':-3.0,'max_blobs':8})."""
    paths = _concat_block_paths(input_dir, name, conditions=conditions, task=task)
    blocks = [discretize_ersp(np.load(p), score_min=score_min, grid=grid, seg_kwargs=seg_kwargs)
              for p in paths]
    return np.concatenate(blocks, axis=1).astype(np.int8)


def plot_discretized_on_concat(disc: np.ndarray, *, grid: str = "full",
                               role_params: Optional[dict] = None, label: str = "",
                               out_png=None, dpi: int = 150, ax=None):
    """Show the score-gated -1/0/+1 concatenated map (blue=-1 · white=0 · red=+1) — exactly
    what the role matching sees. Block dividers + response lines + Hz/% axes; the title
    reports how many cells survived as +1 / -1 (a quick read on how strict the gate is)."""
    import matplotlib.pyplot as plt
    rp = role_params or ROLE_PARAMS
    blocks = rp["block_order"]; nt = rp[grid]["n_time_per_block"]; stim_end = rp[grid]["stim_bins"][1]
    disc = np.asarray(disc); n_freq, n_time = disc.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(disc, aspect="auto", origin="lower", cmap=ERSP_CMAP, vmin=-1, vmax=1,
              extent=[0.5, n_time + 0.5, 0.5, n_freq + 0.5], interpolation="nearest")
    for i, blk in enumerate(blocks):
        off = i * nt
        if i > 0:
            ax.axvline(off + 0.5, color="k", lw=1.6)
        ax.axvline(off + stim_end + 0.5, color="0.3", ls="--", lw=1.0)
        ax.text(off + nt / 2, n_freq + 0.5, blk, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlim(0.5, n_time + 0.5); ax.set_ylim(0.5, n_freq + 0.5)
    _set_pct_xaxis(ax, nt, len(blocks)); _set_hz_yaxis(ax, n_freq)
    npos = int((disc == 1).sum()); nneg = int((disc == -1).sum())
    ax.set_title(f"{label}   score-gated -1/0/+1  (blue=-1 · red=+1 · {npos} pos / {nneg} neg cells)")
    if out_png is not None:
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(out_png, dpi=dpi, bbox_inches="tight")
    return ax


_BAND_COLOR = {"HGA": "#cc0033", "alpha_beta": "#1f4fd8", "theta": "#2ca02c", "narrow_gamma": "#9467bd"}


def _hz_to_rows(f_hz, grid: str) -> Tuple[int, int]:
    """(f_lo_row, f_hi_row), 1-based inclusive, for a Hz range on the given grid.
    full: linear 0..FMAX over N_FREQ rows. ds: the 15 bands overlapping the range."""
    lo_hz, hi_hz = float(f_hz[0]), float(f_hz[1])
    if grid == "ds":
        # inclusive boundaries: match JS boxPass `bd[1] >= lo && bd[0] <= hi`
        idx = [i + 1 for i, (blo, bhi) in enumerate(FREQ_BANDS) if bhi >= lo_hz and blo <= hi_hz]
        return (min(idx), max(idx)) if idx else (1, 1)
    lo = max(1, int(round(lo_hz / FMAX_HZ * (N_FREQ - 1))) + 1)
    hi = min(N_FREQ, int(round(hi_hz / FMAX_HZ * (N_FREQ - 1))) + 1)
    return (lo, hi)


def _box_rows(box: dict, grid: str) -> Tuple[int, int]:
    """1-based inclusive (f_lo, f_hi) ROWS for a box — accepts 'f_hz' (Hz) or legacy 'f_rows'."""
    if "f_hz" in box:
        return _hz_to_rows(box["f_hz"], grid)
    return tuple(box["f_rows"])


def _box_hz(box: dict, grid: str) -> Tuple[float, float]:
    """(f_lo, f_hi) in Hz for a box — from 'f_hz' directly, or converting legacy 'f_rows'."""
    if "f_hz" in box:
        return tuple(box["f_hz"])
    r0, r1 = box["f_rows"]
    if grid == "ds":
        return (FREQ_BANDS[r0 - 1][0], FREQ_BANDS[r1 - 1][1])
    return ((r0 - 1) / (N_FREQ - 1) * FMAX_HZ, (r1 - 1) / (N_FREQ - 1) * FMAX_HZ)


def _band_of(f_hz) -> str:
    lo, hi = float(f_hz[0]), float(f_hz[1])
    if hi <= 8:   return "theta"
    if lo >= 50:  return "HGA"
    if lo >= 30:  return "narrow_gamma"
    return "alpha_beta"


def plot_roles_on_concat(concat: np.ndarray, *, grid: str = "full", role_params: Optional[dict] = None,
                         label: str = "", out_png=None, dpi: int = 150, ax=None):
    """Overlay the CONCATENATED role ROIs (roi_config_concatenated) on one contact's
    [audio|picture|reading] triptych ERSP. The unique (block × time × freq) box regions
    are drawn as outlines coloured by frequency band, with block dividers + per-block
    response-onset lines."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    rp = role_params or ROLE_PARAMS
    blocks = rp["block_order"]; nt = rp[grid]["n_time_per_block"]; stim_end = rp[grid]["stim_bins"][1]
    concat = np.asarray(concat); n_freq, n_time = concat.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    a = ERSP_VLIM
    ax.imshow(concat, aspect="auto", origin="lower", cmap=ERSP_CMAP, vmin=-a, vmax=a,
              extent=[0.5, n_time + 0.5, 0.5, n_freq + 0.5])
    seen = set()
    for role in rp[grid]["roles"]:
        for b in role["boxes"]:
            key = (b["block"], tuple(b["t_bins"]), tuple(b.get("f_hz", b.get("f_rows"))))
            if key in seen:
                continue
            seen.add(key)
            off = blocks.index(b["block"]) * nt
            t0, t1 = b["t_bins"]; f0, f1 = _box_rows(b, grid)
            c = _BAND_COLOR.get(_band_of(_box_hz(b, grid)), "#333")
            ax.add_patch(Rectangle((off + t0 - 0.5, f0 - 0.5), (t1 - t0) + 1, (f1 - f0) + 1,
                         fill=False, edgecolor=c, lw=1.6))
    for i, blk in enumerate(blocks):
        off = i * nt
        if i > 0:
            ax.axvline(off + 0.5, color="k", lw=1.6)
        ax.axvline(off + stim_end + 0.5, color="0.3", ls="--", lw=1.0)
        ax.text(off + nt / 2, n_freq + 0.5, blk, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlim(0.5, n_time + 0.5); ax.set_ylim(0.5, n_freq + 0.5)
    _set_pct_xaxis(ax, nt, len(blocks)); _set_hz_yaxis(ax, n_freq)
    ax.legend(handles=[Patch(edgecolor=v, facecolor="none", label=k) for k, v in _BAND_COLOR.items()],
              loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"{label}   concatenated ROIs on [audio|picture|reading] ERSP")
    if out_png is not None:
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(out_png, dpi=dpi, bbox_inches="tight")
    return ax


_SIGN3 = {"pos": "#cc0033", "neg": "#1f4fd8", "zero": "#777777", "nonpos": "#1f9e89", "both": "#7a1fa2"}


def plot_role_on_concat(concat: np.ndarray, role: dict, *, grid: str = "full",
                        role_params: Optional[dict] = None, label: str = "",
                        out_png=None, dpi: int = 150, ax=None, show_gaussian: bool = False):
    """Overlay ONE role's template on a contact's [audio|picture|reading] triptych:
    pos boxes red (filled), neg blue (filled), zero grey (dashed = 'must be silent').
    With show_gaussian=True, also draw each box's gaussian window as a dotted ±1σ
    ellipse (same colour) so the box (hard edge) and gaussian (soft) gates show on the
    SAME plot. Used to show a representative ERSP that fits each role."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch, Ellipse
    rp = role_params or ROLE_PARAMS
    blocks = rp["block_order"]; nt = rp[grid]["n_time_per_block"]; stim_end = rp[grid]["stim_bins"][1]
    concat = np.asarray(concat); n_freq, n_time = concat.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.6))
    a = ERSP_VLIM
    ax.imshow(concat, aspect="auto", origin="lower", cmap=ERSP_CMAP, vmin=-a, vmax=a,
              extent=[0.5, n_time + 0.5, 0.5, n_freq + 0.5])
    for b in role["boxes"]:
        off = blocks.index(b["block"]) * nt
        f0, f1 = _box_rows(b, grid)
        if "t_pct" in b:
            t0 = round(b["t_pct"][0] / 100.0 * nt) + 1
            t1 = round(b["t_pct"][1] / 100.0 * nt)
        else:
            t0, t1 = b["t_bins"]
        s = b.get("sign", "pos"); c = _SIGN3.get(s, "#333")
        active = s in ("pos", "neg")        # active=solid fill; zero/nonpos=translucent gray dashed
        ax.add_patch(Rectangle((off + t0 - 0.5, f0 - 0.5), (t1 - t0) + 1, (f1 - f0) + 1,
                     fill=True, facecolor=c, alpha=(0.16 if active else 0.12),
                     edgecolor=c, lw=2.0, ls=("-" if active else ":")))
        if show_gaussian:                    # dotted ±1σ ellipse = the gaussian window (same box)
            wt = (t1 - t0 + 1); hf = (f1 - f0 + 1)
            ax.add_patch(Ellipse((off + (t0 + t1) / 2.0, (f0 + f1) / 2.0),
                         width=wt / GAUSS_SUPPORT_SIGMA, height=hf / GAUSS_SUPPORT_SIGMA,
                         fill=False, edgecolor=c, lw=1.3, ls=(0, (1, 1.2))))
    # block labels sit a FIXED 6-pt gap above the axes (offset points = dpi-independent)
    # so they never collide with the data, the title, or each other.
    for i, blk in enumerate(blocks):
        off = i * nt
        if i > 0:
            ax.axvline(off + 0.5, color="k", lw=1.6)
        ax.axvline(off + stim_end + 0.5, color="0.3", ls="--", lw=1.0)
        ax.annotate(blk, xy=(off + nt / 2, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", annotation_clip=False)
    # name the response-onset line once (first block), tight to the line just inside the top
    ax.annotate("resp", xy=(stim_end + 0.5, 1.0), xycoords=("data", "axes fraction"),
                xytext=(2, -2), textcoords="offset points", ha="left", va="top",
                fontsize=8, color="0.3", annotation_clip=False)
    ax.set_xlim(0.5, n_time + 0.5); ax.set_ylim(0.5, n_freq + 0.5)
    _set_pct_xaxis(ax, nt, len(blocks)); _set_hz_yaxis(ax, n_freq)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(11); ax.yaxis.label.set_size(11)
    _SIGN_LABEL = {"pos": "activation", "neg": "suppression", "zero": "silent", "nonpos": "not active (≤0)"}
    signs_used = [s for s in ("pos", "neg", "zero", "nonpos") if any(b.get("sign") == s for b in role["boxes"])]
    ax.legend(handles=[Patch(facecolor=_SIGN3.get(s, "#333"), edgecolor=_SIGN3.get(s, "#333"), alpha=0.5,
                             label=_SIGN_LABEL.get(s, s)) for s in signs_used],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, framealpha=0.9)
    ax.set_title(f"{label} — role '{role['role']}'", fontsize=13, pad=26)
    if out_png is not None:
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        fig = ax.get_figure()
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")                       # 150-dpi (slides)
        out300 = out_png.with_name(out_png.stem + "_300dpi" + out_png.suffix)
        fig.savefig(out300, dpi=300, bbox_inches="tight")                        # 300-dpi (print)
    return ax


def plot_role_exemplars(input_dir, df_role: pd.DataFrame, *, grid: str = "full",
                        role_params: Optional[dict] = None, n_per_role: int = 2,
                        run_dir=None, verbose: bool = True) -> Dict[str, list]:
    """For each role, pick `n_per_role` contacts ASSIGNED that role (from a 460 role
    table) and plot their concatenated ERSP with that role's template. One
    representative ERSP per role. Returns {role: [png paths]}."""
    rp = role_params or ROLE_PARAMS
    out: Dict[str, list] = {}
    for role in rp[grid]["roles"]:
        rn = role["role"]
        sub = (df_role[df_role["role"] == rn]
               .drop_duplicates(["patient_id", "contact_norm"]).head(int(n_per_role)))
        paths = []
        for r in sub.itertuples():
            name = f"{r.patient_id}_reading_WM_ERSP_{r.electrode}_TN.npy"   # cond-independent mid
            try:
                concat = load_concat_ersp(input_dir, name)
            except (FileNotFoundError, ValueError) as e:
                if verbose:
                    print(f"    [skip] {rn} · {r.patient_id} {r.electrode}: {e}")
                continue
            png = None
            if run_dir is not None:
                png = Path(run_dir) / ("exemplar_%s_%s_%s.png" % (rn, r.patient_id, r.electrode)).replace("/", "-")
            plot_role_on_concat(concat, role, grid=grid,
                                label=f"{r.patient_id} {r.electrode}", out_png=png)
            paths.append(png)
        out[rn] = paths
        if verbose:
            print(f"  {rn}: {len(paths)} exemplar(s)")
    return out


def plot_role_hga_timeseries(
        df_contacts: pd.DataFrame,
        X_raw: np.ndarray,
        df_pool: pd.DataFrame,
        role_info: Dict[str, dict],
        *,
        roles: Optional[list] = None,
        grid: str = "ds",
        hga_hz: tuple = (70.0, 150.0),
        ci_mult: float = 1.0,
        onset_thr_db: float = 0.5,
        out_png=None,
        dpi: int = 150,
) -> tuple:
    """Average HGA time series per role, one subplot per condition (audio / picture / reading).

    X_raw  : (n_contacts, n_freq, n_time_total) raw dB.
              For grid='ds' pass the output of build_concat_raw(df_meta, _ds_cube(X_3d)).
    Returns (fig, onset_dict) where onset_dict = {role: onset_t_pct}.
    Patch onset_dict into role_info.json to enable website sidebar sorting.
    """
    import matplotlib.pyplot as plt

    nt = 30 if grid == "ds" else N_TIME
    nf = X_raw.shape[1]

    # HGA row selection
    if grid == "ds":
        hga_rows = np.array([i for i, (lo, hi) in enumerate(FREQ_BANDS)
                             if lo < hga_hz[1] and hi > hga_hz[0]])
    else:
        hz_per_row = FMAX_HZ / nf
        hga_rows = np.array([i for i in range(nf)
                             if i * hz_per_row < hga_hz[1] and (i + 1) * hz_per_row > hga_hz[0]])
    if not len(hga_rows):
        raise ValueError(f"No freq rows for HGA {hga_hz} Hz in grid='{grid}' (n_freq={nf})")

    hga_all = X_raw[:, hga_rows, :].mean(axis=1)   # (n_contacts, n_time_total)

    p_col  = "patient_id"   if "patient_id"   in df_contacts.columns else "patient"
    c_col  = "contact_norm" if "contact_norm"  in df_contacts.columns else "contact"
    dp_col = "patient_id"   if "patient_id"   in df_pool.columns     else "patient"
    dc_col = "contact_norm" if "contact_norm"  in df_pool.columns     else "contact"
    key_to_idx: dict = {(row[p_col], row[c_col]): i for i, row in df_contacts.iterrows()}

    if roles is not None:
        roles = [r for r in roles if r in role_info]
    else:
        roles = [r for r in df_pool["role"].dropna().unique()
                 if r not in ("none", "") and r in role_info]

    CONDS = ["audio", "picture", "reading"]
    t_pct = np.linspace(0, 100, nt, endpoint=False) + 50.0 / nt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ci, (cond, ax) in enumerate(zip(CONDS, axes)):
        sl = slice(ci * nt, (ci + 1) * nt)
        ax.axhline(0, color="0.75", lw=0.8, zorder=0)
        ax.axvline(50, color="0.4", lw=1.0, ls="--", zorder=0)
        ax.text(51.5, 0, "GO", fontsize=7, color="0.4", va="bottom")
        for role_name in roles:
            color = role_info[role_name].get("color", "#888")
            rows_r = df_pool[df_pool["role"] == role_name]
            mats = []
            for _, row in rows_r.iterrows():
                idx = key_to_idx.get((row[dp_col], row[dc_col]))
                if idx is not None:
                    mats.append(hga_all[idx, sl])
            if not mats:
                continue
            mat  = np.stack(mats)
            mean = mat.mean(0)
            sem  = mat.std(0) / np.sqrt(len(mats)) * ci_mult
            ax.plot(t_pct, mean, color=color, lw=1.5,
                    label=f"{role_name} (n={len(mats)})" if ci == 0 else None, zorder=2)
            ax.fill_between(t_pct, mean - sem, mean + sem, color=color, alpha=0.15, zorder=1)
        ax.set_title(cond.capitalize(), fontsize=11)
        ax.set_xlabel("Trial time (%)")
        if ci == 0:
            ax.set_ylabel("HGA (dB)")
            ax.legend(fontsize=6.5, loc="upper left", ncol=1, framealpha=0.85)
        ax.set_xlim(0, 100)

    fig.suptitle("Average HGA time series per role  (±1 SEM)", fontsize=12, y=1.01)
    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    # Onset time: first bin where |grand mean across all conditions| > onset_thr_db
    onset_dict: Dict[str, float] = {}
    for role_name in roles:
        rows_r = df_pool[df_pool["role"] == role_name]
        grand = []
        for ci_b in range(3):
            sl_b = slice(ci_b * nt, (ci_b + 1) * nt)
            for _, row in rows_r.iterrows():
                idx = key_to_idx.get((row[dp_col], row[dc_col]))
                if idx is not None:
                    grand.append(hga_all[idx, sl_b])
        if grand:
            gm = np.stack(grand).mean(0)
            hits = np.where(np.abs(gm) > onset_thr_db)[0]
            onset_dict[role_name] = float(t_pct[hits[0]]) if len(hits) else 100.0

    return fig, onset_dict


def plot_region_hga_timeseries(
        df_contacts: pd.DataFrame,
        X_raw: np.ndarray,
        ns_labels: pd.DataFrame,
        region_info: Dict[str, dict],
        *,
        regions: Optional[list] = None,
        region_col: str = "neurosynth_primary",
        grid: str = "full",
        hga_hz: tuple = (70.0, 150.0),
        ci_mult: float = 1.0,
        onset_thr_db: float = 0.5,
        out_png=None,
        dpi: int = 150,
) -> tuple:
    """Average HGA time series per neurosynth region, one subplot per condition.

    ns_labels : DataFrame with columns (patient|patient_id), (contact|contact_norm),
                and `region_col` (default: neurosynth_primary).
    Returns (fig, onset_dict) where onset_dict = {region: onset_t_pct}.
    """
    import matplotlib.pyplot as plt

    nt = 30 if grid == "ds" else N_TIME
    nf = X_raw.shape[1]

    if grid == "ds":
        hga_rows = np.array([i for i, (lo, hi) in enumerate(FREQ_BANDS)
                             if lo < hga_hz[1] and hi > hga_hz[0]])
    else:
        hz_per_row = FMAX_HZ / nf
        hga_rows = np.array([i for i in range(nf)
                             if i * hz_per_row < hga_hz[1] and (i + 1) * hz_per_row > hga_hz[0]])
    if not len(hga_rows):
        raise ValueError(f"No freq rows for HGA {hga_hz} Hz in grid='{grid}' (n_freq={nf})")

    hga_all = X_raw[:, hga_rows, :].mean(axis=1)   # (n_contacts, n_time_total)

    # normalise column names: df_contacts side
    p_col = "patient_id" if "patient_id" in df_contacts.columns else "patient"
    c_col = "contact_norm" if "contact_norm" in df_contacts.columns else "contact"
    key_to_idx: dict = {(row[p_col], row[c_col]): i for i, row in df_contacts.iterrows()}

    # normalise column names: ns_labels side
    np_col = "patient_id" if "patient_id" in ns_labels.columns else "patient"
    nc_col = "contact_norm" if "contact_norm" in ns_labels.columns else "contact"

    # drop rows with no region assignment
    ns_valid = ns_labels[ns_labels[region_col].notna() & (ns_labels[region_col] != "")]

    if regions is not None:
        region_list = [r for r in regions if r in region_info]
    else:
        region_list = [r for r in ns_valid[region_col].unique() if r in region_info]

    CONDS = ["audio", "picture", "reading"]
    t_pct = np.linspace(0, 100, nt, endpoint=False) + 50.0 / nt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ci, (cond, ax) in enumerate(zip(CONDS, axes)):
        sl = slice(ci * nt, (ci + 1) * nt)
        ax.axhline(0, color="0.75", lw=0.8, zorder=0)
        ax.axvline(50, color="0.4", lw=1.0, ls="--", zorder=0)
        ax.text(51.5, 0, "GO", fontsize=7, color="0.4", va="bottom")
        for region_name in region_list:
            color = region_info[region_name].get("color", "#888")
            rows_r = ns_valid[ns_valid[region_col] == region_name]
            mats = []
            for _, row in rows_r.iterrows():
                idx = key_to_idx.get((row[np_col], row[nc_col]))
                if idx is not None:
                    mats.append(hga_all[idx, sl])
            if not mats:
                continue
            mat  = np.stack(mats)
            mean = mat.mean(0)
            sem  = mat.std(0) / np.sqrt(len(mats)) * ci_mult
            ax.plot(t_pct, mean, color=color, lw=1.5,
                    label=f"{region_name} (n={len(mats)})" if ci == 0 else None, zorder=2)
            ax.fill_between(t_pct, mean - sem, mean + sem, color=color, alpha=0.15, zorder=1)
        ax.set_title(cond.capitalize(), fontsize=11)
        ax.set_xlabel("Trial time (%)")
        if ci == 0:
            ax.set_ylabel("HGA (dB)")
            ax.legend(fontsize=6.5, loc="upper left", ncol=1, framealpha=0.85)
        ax.set_xlim(0, 100)

    fig.suptitle("Average HGA time series per neurosynth region  (±1 SEM)", fontsize=12, y=1.01)
    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    # Onset time: first bin where |grand mean across all conditions| > onset_thr_db
    onset_dict: Dict[str, float] = {}
    for region_name in region_list:
        rows_r = ns_valid[ns_valid[region_col] == region_name]
        grand = []
        for ci_b in range(3):
            sl_b = slice(ci_b * nt, (ci_b + 1) * nt)
            for _, row in rows_r.iterrows():
                idx = key_to_idx.get((row[np_col], row[nc_col]))
                if idx is not None:
                    grand.append(hga_all[idx, sl_b])
        if grand:
            gm = np.stack(grand).mean(0)
            hits = np.where(np.abs(gm) > onset_thr_db)[0]
            onset_dict[region_name] = float(t_pct[hits[0]]) if len(hits) else 100.0

    return fig, onset_dict


def export_role_info(input_dir, df_role: pd.DataFrame, out_dir, *, grid: str = "full",
                     role_params: Optional[dict] = None, verbose: bool = True) -> Path:
    """Write the per-role 'info card' assets the POOL web page shows in its info panel:
      - role_cards/<role>.png : a representative ASSIGNED contact's [a|p|r] ERSP with
        the role's box (solid) + gaussian (dotted ±1σ ellipse) windows on the SAME plot.
      - role_info.json : {role: {description (citations stripped), color, img, n_primary}}.
    One image per role; img paths are relative to out_dir (the page prepends the pool dir)."""
    import matplotlib.pyplot as plt
    rp = role_params or ROLE_PARAMS
    out_dir = Path(out_dir); cards = out_dir / "role_cards"; cards.mkdir(parents=True, exist_ok=True)
    colors = role_colors(grid)
    info: Dict[str, dict] = {}
    for role in rp[grid]["roles"]:
        rn = role["role"]
        desc = str(role.get("description", "")).split("Citation:")[0].strip()
        entry = {"description": desc, "color": colors.get(rn, "#888888"),
                 "img": None, "n_primary": int((df_role["role"] == rn).sum()),
                 "layer": role.get("layer", 1),
                 "boxes": role.get("boxes", []),
                 "thr": role.get("thr", 2.0),
                 "frac": role.get("frac", 0.30),
                 "match": role.get("match", "all")}
        sub = df_role[df_role["role"] == rn].drop_duplicates(["patient_id", "contact_norm"])
        for r in sub.itertuples():                         # first contact whose ERSP loads
            name = f"{r.patient_id}_reading_WM_ERSP_{r.electrode}_TN.npy"
            try:
                concat = load_concat_ersp(input_dir, name)
            except (FileNotFoundError, ValueError):
                continue
            fig, ax = plt.subplots(figsize=(11, 4))
            plot_role_on_concat(concat, role, grid=grid, ax=ax, show_gaussian=True,
                                label=f"{r.patient_id} {r.electrode}")
            fig.savefig(cards / f"{rn}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
            entry["img"] = f"role_cards/{rn}.png"
            break
        info[rn] = entry
    write_json(out_dir / "role_info.json", info)
    if verbose:
        n_img = sum(1 for v in info.values() if v["img"])
        print(f"[lf_pool] role_info -> {out_dir / 'role_info.json'}  ({len(info)} roles, {n_img} cards)")
    return out_dir / "role_info.json"


# ============================================================
# 7c-bis — Neurosynth meta-analytic region assignment (electrode-anatomy based)
# ============================================================
# Sample each Neurosynth association-test z-map (MNI152 volume) at every electrode's
# fsaverage (MNI305) location -> MULTI-LABEL region membership. Region = subfolder
# name under 04_FBM_Pooling/neurosynth/. Anatomy-based, so independent of the
# box/gaussian role gate (parallels the Yeo electrode colouring).
_NEUROSYNTH_DIR = _HERE.parent.parent / "neurosynth"

# FreeSurfer MNI305 (fsaverage) -> MNI152 linear transform (standard constant).
_MNI305_TO_152 = np.array([
    [ 0.9975, -0.0073,  0.0176, -0.0429],
    [ 0.0146,  1.0009, -0.0024,  1.5496],
    [-0.0130, -0.0093,  0.9971,  1.1840],
    [ 0.0,     0.0,     0.0,     1.0   ]])

NEUROSYNTH_COLORS = {
    "auditory":      "#1f77b4",
    "visual":        "#2ca02c",
    "motor control": "#d62728",
    "phonological":  "#9467bd",
    "lexical":       "#ff7f0e",
    "semantic":      "#8c564b",
}


def _load_neurosynth_maps(ns_dir=None):
    """{region: (z_volume, inv_affine)} for each subfolder's *association-test_z_FDR*.nii.gz."""
    import nibabel as nib
    ns_dir = Path(ns_dir or _NEUROSYNTH_DIR)
    maps = {}
    for sub in sorted(p for p in ns_dir.iterdir() if p.is_dir()):
        zfs = sorted(sub.glob("*association-test_z_FDR*.nii.gz"))
        if not zfs:
            continue
        img = nib.load(str(zfs[0]))
        maps[sub.name] = (np.asarray(img.dataobj, dtype=float), np.linalg.inv(img.affine))
    return maps


def assign_neurosynth_regions(coords: pd.DataFrame, *, ns_dir=None, z_thr: float = 0.0,
                              verbose: bool = True) -> pd.DataFrame:
    """Multi-label Neurosynth region membership per contact. Each contact's fsaverage
    (MNI305) xyz -> MNI152 -> sample every region z-map; tag the region when z > z_thr.
    Returns patient, contact, name, neurosynth (';'-joined), neurosynth_primary (max-z)."""
    maps = _load_neurosynth_maps(ns_dir)
    if verbose:
        print(f"[lf_pool] neurosynth maps: {list(maps)}")
    xyz = coords[["x", "y", "z"]].to_numpy(dtype=float)
    mni152 = (np.c_[xyz, np.ones(len(xyz))] @ _MNI305_TO_152.T)[:, :3]
    pat = coords["patient_id"].astype(str).to_numpy()
    con = coords["contact_norm"].astype(str).to_numpy()
    nm = coords["name"].astype(str).to_numpy()
    rows = []
    for i in range(len(coords)):
        p = np.array([mni152[i, 0], mni152[i, 1], mni152[i, 2], 1.0])
        hits = []
        for name, (vol, inv_aff) in maps.items():
            ijk = np.round((inv_aff @ p)[:3]).astype(int)
            if (ijk >= 0).all() and (ijk < np.array(vol.shape[:3])).all():
                z = float(vol[ijk[0], ijk[1], ijk[2]])
                if np.isfinite(z) and z > z_thr:
                    hits.append((z, name))
        hits.sort(reverse=True)
        rows.append({"patient": pat[i], "contact": con[i], "name": nm[i],
                     "neurosynth": ";".join(n for _, n in hits),
                     "neurosynth_primary": hits[0][1] if hits else ""})
    df = pd.DataFrame(rows)
    if verbose:
        print(f"[lf_pool] neurosynth: {len(df)} contacts, "
              f"{int((df['neurosynth_primary'] != '').sum())} in >=1 region (z>{z_thr})")
    return df


def _plot_concat_mean(concat: np.ndarray, label, out_png, *, grid="full",
                      role_params=None, dpi: int = 130):
    """Plot a mean concatenated [audio|picture|reading] ERSP (no boxes) for a region card."""
    import matplotlib.pyplot as plt
    rp = role_params or ROLE_PARAMS
    blocks = rp["block_order"]; nt = rp[grid]["n_time_per_block"]; stim_end = rp[grid]["stim_bins"][1]
    concat = np.asarray(concat); n_freq, n_time = concat.shape
    fig, ax = plt.subplots(figsize=(11, 4))
    a = ERSP_VLIM
    ax.imshow(concat, aspect="auto", origin="lower", cmap=ERSP_CMAP, vmin=-a, vmax=a,
              extent=[0.5, n_time + 0.5, 0.5, n_freq + 0.5])
    for i, blk in enumerate(blocks):
        off = i * nt
        if i > 0:
            ax.axvline(off + 0.5, color="k", lw=1.6)
        ax.axvline(off + stim_end + 0.5, color="0.3", ls="--", lw=1.0)
        ax.annotate(blk, xy=(off + nt / 2, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", annotation_clip=False)
    ax.set_xlim(0.5, n_time + 0.5); ax.set_ylim(0.5, n_freq + 0.5)
    _set_pct_xaxis(ax, nt, len(blocks)); _set_hz_yaxis(ax, n_freq)
    ax.tick_params(labelsize=11); ax.set_title(label, fontsize=13, pad=26)
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def export_neurosynth(input_dir, coords: pd.DataFrame, out_dir, *, grid: str = "full",
                      ns_dir=None, z_thr: float = 0.0, pool_csv=None, verbose: bool = True) -> Path:
    """Assign electrodes to Neurosynth regions (multi-label), average each region's
    member ERSPs, and write what the POOL web page reads:
      - neurosynth_labels.csv : patient, contact, neurosynth, neurosynth_primary
      - neurosynth_info.json  : {region: {color, img, n, n_members}}
      - region_cards/<region>.png : mean concatenated ERSP of that region's members.
    If `pool_csv` (a contacts_pool.csv) is given, restrict to those POOLED contacts so the
    counts match the analysis (not all 5k recon contacts). `n` = contacts actually averaged
    into the card (== what the card claims); `n_members` = total tagged to the region."""
    if pool_csv is not None and Path(pool_csv).exists():
        pool = pd.read_csv(pool_csv)
        keep = set(zip(pool["patient"].astype(str), pool["contact"].astype(str)))
        mask = [(str(p), str(c)) in keep for p, c in zip(coords["patient_id"], coords["contact_norm"])]
        coords = coords[mask].reset_index(drop=True)
        if verbose:
            print(f"[lf_pool] neurosynth: restricted to {len(coords)} pooled contacts")
    df = assign_neurosynth_regions(coords, ns_dir=ns_dir, z_thr=z_thr, verbose=verbose)
    out_dir = Path(out_dir); cards = out_dir / "region_cards"; cards.mkdir(parents=True, exist_ok=True)
    sets = df["neurosynth"].apply(lambda s: [x for x in str(s).split(";") if x])
    regions = list(NEUROSYNTH_COLORS) + sorted({r for s in sets for r in s if r not in NEUROSYNTH_COLORS})
    info = {}
    for region in regions:
        members = df[[region in s for s in sets]]
        entry = {"color": NEUROSYNTH_COLORS.get(region, "#888888"), "img": None,
                 "n": 0, "n_members": int(len(members))}
        acc, k = None, 0
        for r in members.itertuples():
            try:
                c = load_concat_ersp(input_dir, f"{r.patient}_reading_WM_ERSP_{r.name}_TN.npy")
            except (FileNotFoundError, ValueError):
                continue
            acc = c.astype(float) if acc is None else acc + c.astype(float)
            k += 1
        entry["n"] = k                                    # contacts actually averaged (what the card claims)
        if k:
            fn = region.replace(" ", "_") + ".png"
            _plot_concat_mean(acc / k, f"neurosynth · {region}  (mean of {k} contacts)",
                              cards / fn, grid=grid)
            entry["img"] = f"region_cards/{fn}"
        info[region] = entry
    df[["patient", "contact", "neurosynth", "neurosynth_primary"]].to_csv(
        out_dir / "neurosynth_labels.csv", index=False)
    write_json(out_dir / "neurosynth_info.json", info)
    if verbose:
        print(f"[lf_pool] neurosynth -> {out_dir}  "
              f"({sum(1 for v in info.values() if v['img'])} region cards)")
    return out_dir


# ============================================================
# 7c-ter — Downsampled raw-ERSP cube for the poolv2 interactive role designer
# ============================================================
def export_pool_cube(input_dir, contacts: pd.DataFrame, out_dir, *, scale: float = 4.0,
                     verbose: bool = True) -> Path:
    """Downsampled RAW ERSP cube the poolv2 web page reads to re-match boxes live
    (fraction-above-threshold gate, no discretization). For each pooled contact, load
    its concatenated [audio|picture|reading] ERSP (129 x 900), average down to the ds
    grid (15 bands x 90 time = 30 bins x 3 blocks), quantize dB to int8 (round(dB*scale)),
    and stack -> cube_i8.bin (int8, N x 15 x 90, C-order) + cube_manifest.json.
    `contacts` needs patient(_id) / contact(_norm) / name|electrode columns."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bands = [tuple(b) for b in FREQ_BANDS]                      # 15 (lo,hi) Hz
    nF, nT = len(bands), 90
    full_hz = np.arange(129) / 128.0 * FMAX_HZ                  # full-grid row -> Hz
    band_rows = [np.where((full_hz >= lo) & (full_hz <= hi))[0] for (lo, hi) in bands]
    keys, mats = [], []
    for r in contacts.drop_duplicates(["patient_id", "contact_norm"]).itertuples():
        pid, con = str(r.patient_id), str(r.contact_norm)
        ele = str(getattr(r, "electrode", getattr(r, "name", con)))
        try:
            ersp = np.asarray(load_concat_ersp(input_dir, f"{pid}_reading_WM_ERSP_{ele}_TN.npy"), float)
        except (FileNotFoundError, ValueError):
            continue
        fac = ersp.shape[1] // nT                               # 900 -> 90 (avg every 10)
        t_ds = ersp[:, :fac * nT].reshape(ersp.shape[0], nT, fac).mean(axis=2)
        f_ds = np.vstack([t_ds[rws].mean(axis=0) if len(rws) else np.zeros(nT) for rws in band_rows])
        keys.append(f"{pid}|{con}"); mats.append(f_ds.astype(np.float32))
    cube = np.stack(mats) if mats else np.zeros((0, nF, nT), np.float32)
    q = np.clip(np.round(cube * scale), -127, 127).astype(np.int8)
    (out_dir / "cube_i8.bin").write_bytes(q.tobytes(order="C"))
    manifest = {"n": len(keys), "n_freq": nF, "n_time": nT, "n_blocks": 3,
                "bins_per_block": nT // 3, "stim_bins": (nT // 3) // 2, "scale": scale,
                "fmax_hz": FMAX_HZ, "band_hz": [list(b) for b in bands], "contacts": keys}
    write_json(out_dir / "cube_manifest.json", manifest)
    if verbose:
        print(f"[lf_pool] pool cube -> {out_dir}  ({len(keys)} contacts, {q.nbytes/1e6:.1f} MB int8)")
    return out_dir


# ============================================================
# 7d — Concatenated [audio|picture|reading] functional-role matching
# ============================================================
# Condition-STRUCTURED extension: stitch the three conditions per contact, score-
# gate-discretize to -1/0/+1, and assign a functional role only if the contact
# expresses EVERY box of that role's conjunction template (strict AND). A box
# "expresses" its sign via the clustering proportion gate on the -1/+1 cells.
# See functions/roi_config_concatenated.py.
def _load_roles_config():
    path = _HERE.parent / "roi_config_concatenated.py"
    spec = importlib.util.spec_from_file_location("_pool_roles", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                # type: ignore[union-attr]
    return m.ERSP_POOLING_ROLES


ROLE_PARAMS = _load_roles_config()


def discretize_ersp(ersp: np.ndarray, *, score_min: Optional[float] = None,
                    grid: str = "full", n_time_ds: int = DS_TIME_BINS,
                    seg_kwargs: Optional[dict] = None) -> np.ndarray:
    """Score-gated -1/0/+1 map (the m101 representation, same machinery as 01's
    ERSP_minus101). Blobs segmented at full resolution; `score_min` drops low-score
    blobs (pass resolve_score_gate(...)). `seg_kwargs` overrides the segmentation
    thresholds to loosen/tighten it, e.g. {'thr_pos':1.5,'thr_neg':-3.0,'max_blobs':8}.
    grid='full' -> 129x300 int8; 'ds' -> painted map downsampled to 15 x n_time_ds."""
    blob = _ensure_blob()
    m101 = _ensure_minus101()
    kw = {**DEFAULT_SEG_KWARGS, **(seg_kwargs or {})}   # default discretization, caller overrides
    blobs = blob.s21_segment_valley_blobs(ersp, sign_mode="both", fmax=FMAX_HZ, **kw)
    painted = m101.paint_minus101_map(ersp, blobs, score_min=score_min)   # (129,300) int8
    if grid == "full":
        return painted.astype(np.int8)
    ds = m101.downsample_minus101_map(painted, target_shape=(15, int(n_time_ds)))
    out = np.zeros_like(ds, dtype=np.int8)
    out[ds > 0.34] = 1
    out[ds < -0.34] = -1
    return out


def build_concat_discretized(df_meta: pd.DataFrame, X_3d: np.ndarray, *,
                             score_min: Optional[float] = None, grid: str = "full",
                             conditions: Sequence[str] = CONDITIONS,
                             n_time_ds: int = DS_TIME_BINS, seg_kwargs: Optional[dict] = None,
                             verbose: bool = True
                             ) -> Tuple[pd.DataFrame, np.ndarray]:
    """One score-gated discretized [audio|picture|reading] map per contact that
    has ALL conditions. Returns (df_contacts, X_disc) with X_disc int8
    (n_contacts, n_freq, len(conditions)*n_time_block). `seg_kwargs` overrides the
    blob segmentation (defaults to DEFAULT_SEG_KWARGS) so 460 matching uses the same
    discretization tuned in 470."""
    by_contact: Dict[tuple, dict] = {}
    for row in df_meta.itertuples():
        if row.contact_norm is None:
            continue
        by_contact.setdefault((row.patient_id, row.contact_norm), {})[row.condition] = \
            (row.Index, row.electrode)
    rows, disc_maps = [], []
    cond = list(conditions)
    for (pid, contact), cm in by_contact.items():
        if not all(c in cm for c in cond):
            continue
        blocks = [discretize_ersp(X_3d[cm[c][0]], score_min=score_min, grid=grid,
                                  n_time_ds=n_time_ds, seg_kwargs=seg_kwargs) for c in cond]
        rows.append({"patient_id": pid, "contact_norm": contact, "electrode": cm[cond[0]][1]})
        disc_maps.append(np.concatenate(blocks, axis=1).astype(np.int8))
    df_contacts = pd.DataFrame(rows).reset_index(drop=True)
    X_disc = np.stack(disc_maps, 0) if disc_maps else np.zeros((0, 0, 0), np.int8)
    if verbose:
        print(f"[lf_pool] concat-discretized: {len(df_contacts)} contacts with all "
              f"{len(cond)} conditions · X_disc={X_disc.shape}")
    return df_contacts, X_disc


def _box_slices(box: dict, grid: str) -> Tuple[slice, slice]:
    nt = ROLE_PARAMS[grid]["n_time_per_block"]
    bi = ROLE_PARAMS["block_order"].index(box["block"])
    f0, f1 = _box_rows(box, grid)
    if "t_pct" in box:          # designer export format: 0-100% percentages
        t_start = round(box["t_pct"][0] / 100.0 * nt)
        t_end   = round(box["t_pct"][1] / 100.0 * nt)
    else:                        # legacy t_bins: 1-indexed absolute bin indices
        t0, t1 = box["t_bins"]; t_start, t_end = t0 - 1, t1
    return slice(f0 - 1, f1), slice(bi * nt + t_start, bi * nt + t_end)


def _box_gaussian_weights(nf: int, nt: int) -> np.ndarray:
    """Normalized 2-D gaussian over an (nf x nt) box: centred on the box, sigma =
    half-extent / GAUSS_SUPPORT_SIGMA on each axis (so +/-GAUSS_SUPPORT_SIGMA sigma
    spans the box). Centre cells count fully, edges taper -> a soft window that
    tolerates latency/frequency jitter instead of the boxcar's hard edge."""
    fr = np.arange(nf); tc = np.arange(nt)
    cf, ct = (nf - 1) / 2.0, (nt - 1) / 2.0
    sf = max((nf / 2.0) / GAUSS_SUPPORT_SIGMA, 1e-6)
    st = max((nt / 2.0) / GAUSS_SUPPORT_SIGMA, 1e-6)
    gf = np.exp(-0.5 * ((fr - cf) / sf) ** 2)
    gt = np.exp(-0.5 * ((tc - ct) / st) ** 2)
    w = np.outer(gf, gt)
    return w / (w.sum() + 1e-12)


def box_expresses(disc_concat: np.ndarray, box: dict, grid: str, *, shape: str = "box") -> bool:
    """Proportion gate on the score-gated -1/+1 cells of one box. An "on" box
    (pos/neg/both) must have at least `ROLE_BOX_MIN_PROP` (default 0.5 → half) of its
    cells at the expected sign — a handful of cells is not enough. "Silent"
    constraints (zero/nonpos) stay strict on the clustering presence gate (MIN_PROP_*),
    so they still demand near-total silence. Per-box override via box['min_prop'].

    shape='box'      -> every cell counts equally (boxcar; the default/reference).
    shape='gaussian' -> cells are weighted by a 2-D gaussian centred on the box
                        (time x freq), so the proportion is a soft, jitter-tolerant
                        weighted fraction. Threshold semantics are unchanged.
      'pos'    -> >=half +1 cells (activation),
      'neg'    -> >=half -1 cells (suppression),
      'zero'   -> SILENT: NEITHER +1 nor -1 reaches the strict presence gate (the
                  negative constraint that makes a role discriminating),
      'nonpos' -> must NOT activate: +1 below the strict presence gate (suppression
                  or silence both OK),
      'both'   -> either +1 or -1 reaches the >=half "on" gate."""
    fs, ts = _box_slices(box, grid)
    sub = disc_concat[fs, ts]
    if sub.size == 0:
        return False
    if shape == "gaussian":
        w = _box_gaussian_weights(*sub.shape)            # sums to 1
        pp = float((w * (sub == 1)).sum()); pn = float((w * (sub == -1)).sum())
    else:
        pp = float((sub == 1).mean()); pn = float((sub == -1).mean())
    s = box.get("sign", "both")
    on = float(box.get("min_prop", ROLE_BOX_MIN_PROP))   # "on" gate: >=half by default
    if s == "pos":
        return pp >= on
    if s == "neg":
        return pn >= on
    if s == "zero":            # strict silence: below the clustering presence gate
        return (pp < MIN_PROP_POS) and (pn < MIN_PROP_NEG)
    if s == "nonpos":          # "at most zero": must NOT activate (suppression or silence both OK)
        return pp < MIN_PROP_POS
    return (pp >= on) or (pn >= on)


def role_matches(disc_concat: np.ndarray, role: dict, grid: str, *, shape: str = "box") -> bool:
    """Box reducer. Default `match`='all' -> strict conjunction (every box must
    express its sign — the selective roles). `match`='any' -> disjunction: the role
    matches if AT LEAST ONE box expresses (umbrella/general tags like
    stimulus_responsive: 'HGA in any one stim window'), so it co-occurs with the
    specific roles instead of excluding them. `shape` selects the box gate
    ('box' boxcar or 'gaussian' soft window)."""
    reducer = any if role.get("match") == "any" else all
    return reducer(box_expresses(disc_concat, b, grid, shape=shape) for b in role["boxes"])


def role_colors(grid: str = "full", role_params: Optional[dict] = None) -> Dict[str, str]:
    rp = role_params or ROLE_PARAMS
    d = {r["role"]: r["color"] for r in rp[grid]["roles"]}
    d["none"] = "#cccccc"
    return d


def build_role_table(df_contacts: pd.DataFrame, X_disc: np.ndarray, *, grid: str = "full",
                     role_params: Optional[dict] = None, shape: str = "box",
                     cache: bool = True, write_run: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Assign each contact its matching role(s). `role` = the MOST SPECIFIC match
    (most boxes); `roles_matched` lists all. One row per contact. `shape` selects the
    box gate: 'box' (boxcar, the default/reference) or 'gaussian' (2-D gaussian-
    weighted window). Non-box shapes get a '_<shape>' suffix on cache + run paths so
    they never clobber the boxcar artifacts (470 etc. read the un-suffixed table)."""
    rp = role_params or ROLE_PARAMS
    roles = rp[grid]["roles"]
    tag = "" if shape == "box" else f"_{shape}"
    rows = []
    for k, row in enumerate(df_contacts.itertuples()):
        disc = X_disc[k]
        matched = [r for r in roles if role_matches(disc, r, grid, shape=shape)]
        best = max(matched, key=lambda r: len(r["boxes"]))["role"] if matched else "none"
        rows.append({"patient_id": row.patient_id, "contact_norm": row.contact_norm,
                     "electrode": row.electrode, "grid": grid, "role": best,
                     "roles_matched": ";".join(r["role"] for r in matched),
                     "n_roles": len(matched)})
    df = pd.DataFrame(rows)
    if cache:
        DATASET_CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATASET_CACHE / f"role_table_{grid}{tag}.parquet", index=False)
    if write_run:
        run_dir = new_run_dir(f"roles{tag}", grid)
        df.to_parquet(run_dir / "role_table.parquet", index=False)
        role_counts(df).to_csv(run_dir / "role_counts.csv", index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION, "stage": f"roles{tag}", "grid": grid,
            "shape": shape, "run_id": run_dir.name,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "user": _safe_user(),
            "n_contacts": int(len(df)), "n_roles": len(roles),
            "path": str(run_dir.relative_to(OUTPUTS_ROOT)),
        }
        write_json(run_dir / "manifest.json", manifest)
        update_index(manifest)
        if verbose:
            print(f"[lf_pool] roles run -> {run_dir}")
    if verbose:
        print(f"[lf_pool] role table: {df.shape}")
    return df


def build_concat_raw(df_meta: pd.DataFrame, X_3d: np.ndarray, *,
                     conditions: Sequence[str] = CONDITIONS,
                     verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Concatenated [audio|picture|reading] raw dB ERSP per contact that has ALL conditions.
    Returns (df_contacts, X_raw) float32 (n_contacts, n_freq, 3*n_time_per_block).
    No discretization — preserves raw amplitude for web-style threshold matching."""
    by_contact: Dict[tuple, dict] = {}
    for row in df_meta.itertuples():
        if row.contact_norm is None:
            continue
        by_contact.setdefault((row.patient_id, row.contact_norm), {})[row.condition] = \
            (row.Index, row.electrode)
    rows, raw_maps = [], []
    cond = list(conditions)
    for (pid, contact), cm in by_contact.items():
        if not all(c in cm for c in cond):
            continue
        blocks = [X_3d[cm[c][0]].astype(np.float32) for c in cond]
        rows.append({"patient_id": pid, "contact_norm": contact, "electrode": cm[cond[0]][1]})
        raw_maps.append(np.concatenate(blocks, axis=1).astype(np.float32))
    df_contacts = pd.DataFrame(rows).reset_index(drop=True)
    X_raw = np.stack(raw_maps, 0) if raw_maps else np.zeros((0, 0, 0), np.float32)
    if verbose:
        print(f"[lf_pool] concat-raw: {len(df_contacts)} contacts, all {len(cond)} conditions "
              f"· X_raw={X_raw.shape}")
    return df_contacts, X_raw


def box_expresses_raw(ersp_raw: np.ndarray, box: dict, grid: str, *,
                       thr: float, min_prop: float) -> bool:
    """Raw-dB fraction gate — mirrors poolv2's boxPass() exactly.
    ersp_raw: (n_freq, n_time_total) float32 raw dB [audio|picture|reading].
    'pos'  -> fraction of cells > thr >= min_prop.
    'neg'  -> fraction of cells < -thr >= min_prop.
    'zero' -> silent gate: fraction of cells with |dB| > thr must be < min_prop."""
    fs, ts = _box_slices(box, grid)
    patch = ersp_raw[fs, ts]
    if patch.size == 0:
        return box.get("sign", "pos") in ("zero", "nonpos")
    sign = box.get("sign", "pos")
    if sign == "pos":
        return float((patch > thr).mean()) >= min_prop
    if sign == "neg":
        return float((patch < -thr).mean()) >= min_prop
    if sign == "nonpos":
        return float((patch > thr).mean()) < min_prop       # not-activating gate
    return float((np.abs(patch) > thr).mean()) < min_prop   # zero / silent


def role_matches_raw(ersp_raw: np.ndarray, role: dict, grid: str) -> bool:
    """Raw-dB conjunction/disjunction — mirrors poolv2's Apply button.
    thr and frac come from the role definition in roi_config_concatenated.py."""
    thr      = float(role.get("thr",  2.0))
    min_prop = float(role.get("frac", ROLE_BOX_MIN_PROP))
    reducer  = any if role.get("match") == "any" else all
    return reducer(box_expresses_raw(ersp_raw, b, grid, thr=thr, min_prop=min_prop)
                   for b in role["boxes"])


def build_role_table_raw(df_contacts: pd.DataFrame, X_raw: np.ndarray, *, grid: str = "full",
                          role_params: Optional[dict] = None,
                          cache: bool = True, write_run: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Role assignment via raw-dB matching (identical to poolv2's Apply button).
    X_raw: (n_contacts, n_freq, n_time_total) float32 — output of build_concat_raw().
    Each role's thr and frac come from roi_config_concatenated.py."""
    rp    = role_params or ROLE_PARAMS
    roles = rp[grid]["roles"]
    rows  = []
    for k, row in enumerate(df_contacts.itertuples()):
        ersp    = X_raw[k].astype(np.float32)
        matched = [r for r in roles if role_matches_raw(ersp, r, grid)]
        best    = max(matched, key=lambda r: len(r["boxes"]))["role"] if matched else "none"
        rows.append({"patient_id": row.patient_id, "contact_norm": row.contact_norm,
                     "electrode": row.electrode, "grid": grid, "role": best,
                     "roles_matched": ";".join(r["role"] for r in matched),
                     "n_roles": len(matched)})
    df = pd.DataFrame(rows)
    if cache:
        DATASET_CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATASET_CACHE / "role_table_raw.parquet", index=False)
    if write_run:
        run_dir = new_run_dir("roles_raw", grid)
        df.to_parquet(run_dir / "role_table.parquet", index=False)
        role_counts(df).to_csv(run_dir / "role_counts.csv", index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION, "stage": "roles_raw", "grid": grid,
            "shape": "raw_db", "run_id": run_dir.name,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "user": _safe_user(),
            "n_contacts": int(len(df)), "n_roles": len(roles),
            "path": str(run_dir.relative_to(OUTPUTS_ROOT)),
        }
        write_json(run_dir / "manifest.json", manifest)
        update_index(manifest)
        if verbose:
            print(f"[lf_pool] roles_raw run -> {run_dir}")
    if verbose:
        print(f"[lf_pool] role table (raw dB): {df.shape}")
    return df


def _matched_sets(df_role: pd.DataFrame) -> pd.Series:
    """roles_matched ';'-string -> set of role names, per contact (multi-label)."""
    return df_role["roles_matched"].fillna("").apply(
        lambda s: {x.strip() for x in str(s).split(";") if x.strip()})


def role_counts(df_role: pd.DataFrame) -> pd.DataFrame:
    """MULTI-LABEL role counts. `n_matched` = contacts where the role is in
    roles_matched (a contact counts toward EVERY role it matches, so the column can
    sum to more than the contact count); `n_primary` = contacts where it is the
    most-specific `role`. This is the scientifically standard 'response profile'
    view — a contact is reported by all the responses it shows, not one winner."""
    sets = _matched_sets(df_role)
    prim = df_role.groupby("role").size()
    matched = pd.Series([r for s in sets for r in s], dtype="object").value_counts()
    roles = sorted(set(prim.index) | set(matched.index),
                   key=lambda r: (-int(matched.get(r, 0)), r))
    out = pd.DataFrame({"role": roles,
                        "n_matched": [int(matched.get(r, 0)) for r in roles],
                        "n_primary": [int(prim.get(r, 0)) for r in roles]})
    return out.sort_values("n_matched", ascending=False).reset_index(drop=True)


def role_membership(df_role: pd.DataFrame, roles: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One-hot multi-label membership: one row per contact, one bool column per role
    (from roles_matched). A contact can be True for several roles."""
    sets = _matched_sets(df_role)
    all_roles = list(roles) if roles is not None else sorted({r for s in sets for r in s})
    M = pd.DataFrame({r: sets.apply(lambda s, r=r: r in s) for r in all_roles})
    base = df_role[["patient_id", "contact_norm"]].reset_index(drop=True)
    return pd.concat([base, M.reset_index(drop=True)], axis=1)


def role_cooccurrence(df_role: pd.DataFrame) -> pd.DataFrame:
    """Symmetric role x role matrix: how many contacts match BOTH roles (diagonal =
    total contacts matching that role). Directly answers 'which roles share contacts'."""
    mem = role_membership(df_role)
    role_cols = [c for c in mem.columns if c not in ("patient_id", "contact_norm")]
    M = mem[role_cols].to_numpy(dtype=int)
    return pd.DataFrame(M.T @ M, index=role_cols, columns=role_cols)


def load_role_table(grid: str = "full", path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DATASET_CACHE / f"role_table_{grid}.parquet"
    return pd.read_parquet(p)


def _ersp_png_rel(file_path) -> Optional[str]:
    """Source ERSP .npy path -> repo-relative ERSP_clean PNG web path, the same
    swap MOBA's _filepathToSampleViewUrl does (ERSP_matrix->ERSP_clean, .npy->
    _CLEAN.png), starting at '01_FBM_Analysis/'. Returns None if unmappable."""
    p = str(file_path).replace("\\", "/")
    i = p.find("01_FBM_Analysis/")
    if i < 0:
        return None
    p = p[i:].replace("/ERSP_matrix/", "/ERSP_clean/")
    return re.sub(r"\.npy$", "_CLEAN.png", p, flags=re.IGNORECASE)


def export_pool_web(df_role: pd.DataFrame, coords: pd.DataFrame, run_dir, *,
                    grid: str = "full", df_meta: pd.DataFrame = None) -> Path:
    """Write the data the POOL web page reads: contacts_pool.csv (xyz + role +
    colour) + pool_index.json. If df_meta is given, also write pool_samples.json
    mapping 'patient|contact' -> {condition: ERSP_clean PNG path} so the page can
    show each pooled contact's ERSPs. Reuses the MOBA fsaverage meshes."""
    colors = role_colors(grid)
    cols = ["patient_id", "contact_norm", "name", "hemi", "x", "y", "z"]
    if "is_cortical" in coords.columns:
        cols.append("is_cortical")
    for yc in ("yeo7_network", "yeo17_network"):   # per-electrode Yeo labels (for the viewer's Yeo colour mode)
        if yc in coords.columns:
            cols.append(yc)
    c = coords[cols].drop_duplicates(["patient_id", "contact_norm"])
    m = df_role.merge(c, on=["patient_id", "contact_norm"], how="left").dropna(subset=["x", "y", "z"])
    m["color"] = m["role"].map(colors).fillna("#cccccc")
    if "is_cortical" not in m.columns:
        m["is_cortical"] = 1                       # cortical/depth radius split (page)
    for yc in ("yeo7_network", "yeo17_network"):
        m[yc] = m[yc].fillna("") if yc in m.columns else ""
    if "n_roles" not in m.columns:
        m["n_roles"] = m["roles_matched"].fillna("").apply(
            lambda s: len([x for x in str(s).split(";") if x.strip()]))
    out = (m[["patient_id", "contact_norm", "name", "hemi", "x", "y", "z",
              "is_cortical", "role", "roles_matched", "n_roles",
              "yeo7_network", "yeo17_network", "color"]]
           .rename(columns={"patient_id": "patient", "contact_norm": "contact"}))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(run_dir / "contacts_pool.csv", index=False)
    # MULTI-LABEL counts: a contact counts toward EVERY role in roles_matched.
    msets = out["roles_matched"].fillna("").apply(
        lambda s: {x.strip() for x in str(s).split(";") if x.strip()})
    mcount = pd.Series([r for s in msets for r in s], dtype="object").value_counts()
    idx = {"grid": grid, "n_contacts": int(len(out)), "csv": "contacts_pool.csv",
           "multilabel": True,
           "roles": [{"role": r, "color": colors[r],
                      "n": int(mcount.get(r, 0)),                    # any-match (multi-label)
                      "n_primary": int((out["role"] == r).sum())}    # most-specific
                     for r in colors]}
    write_json(run_dir / "pool_index.json", idx)

    # Per-contact ERSP images for the web page's samples strip (optional).
    if df_meta is not None and {"patient_id", "contact_norm", "condition",
                                "file_path"}.issubset(df_meta.columns):
        keep = set(zip(out["patient"], out["contact"]))
        samples: dict = {}
        for r in df_meta.itertuples(index=False):
            if (r.patient_id, r.contact_norm) not in keep:
                continue
            png = _ersp_png_rel(r.file_path)
            if png:
                samples.setdefault(f"{r.patient_id}|{r.contact_norm}", {})[r.condition] = png
        write_json(run_dir / "pool_samples.json", samples)
        print(f"[lf_pool] POOL samples -> pool_samples.json  ({len(samples)} contacts)")

    print(f"[lf_pool] POOL web export -> {run_dir}  ({len(out)} contacts)")
    return run_dir


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
