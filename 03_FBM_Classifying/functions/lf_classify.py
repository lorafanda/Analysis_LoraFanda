"""
lf_classify.py — Classification engine for 03_FBM_Classifying.

Sister module to 02_FBM_Clustering's lf_cluster_run. Where clustering asked
"what unsupervised structure is in the ERSPs?", classification asks two
*supervised* questions:

  (A) CONDITION decoding  — given one electrode's spectro-temporal response,
      can we tell whether the trial was AUDIO, PICTURE, or READING? (3 classes,
      one sample per electrode x condition, high-activity-gated.)

  (B) PARCELLATION decoding — given an electrode's response *profile across all
      three conditions* (concatenated), can we tell which Yeo functional
      network it sits in? (7- or 17-class, one sample per electrode.)

Everything is built on the SAME upstream ERSP .npy files the clustering
pipeline uses (01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY/...), reusing the
canonical loader (lf_dataset.prepare_dataset) and the feature builders
(lf_features.downsample_ersp_to_bands, lf_hg.extract_hg_time_series) from
02_FBM_Clustering/functions so the feature definitions stay identical across
the two stages.

Feature variants form a NESTED, TIME-MATCHED family so "full spectrum vs high-
gamma" is a fair test (HG and FULL share a time grid; only frequency changes):
  - 'full_300' : 15 band-aware ERSP rows x 300 time  (the full spectrum)
  - 'hg_300'   : single 70-150 Hz band-mean line x 300 time
  - 'full_30'  : 15 bands x 30 time
  - 'hg_30'    : single HG line x 30 time
Valid contrasts are the matched pairs (full_300 vs hg_300) and (full_30 vs
hg_30); full_300 vs full_30 isolates time resolution. NOTE: the STFT couples
time and frequency resolution, so even matched grids aren't a perfectly clean
separation -- stated as a caveat, not hidden.

Design decisions (locked with Lora, 2026-06):
  - Patients: only those with ALL THREE conditions computed enter either task.
  - Condition task: high-activity-gated electrode x condition samples.
  - Parcellation task: electrode included if high-activity in >=1 condition;
    its features are the concatenation of all three conditions' (ungated)
    feature vectors in fixed order [audio, picture, reading].
  - Yeo labels: medial-wall / white-matter / unknown contacts are dropped;
    only the true 7 (or 17) networks are classified.
  - Classifiers: logistic regression (multinomial) AND random forest, both
    with class_weight='balanced'.
  - Validation: NESTED GroupKFold by patient — outer fold = held-out test
    patients, inner fold = validation for hyper-parameter tuning. No patient
    ever appears in both train and test.
  - Above-chance significance: grouped label-permutation null (overall +
    per-class), empirical p-values, FDR-corrected per class.

Public API (used by the notebooks):
  prepare_full_dataset(input_dir, cache_dir)       -> df_meta, X_3d
  build_condition_arrays(df_meta, X_3d, variant)   -> X, y, groups, meta, cols
  build_parcellation_arrays(df_meta, X_3d, coords_dir, n_networks, variant)
  run_experiment(...)                              -> manifest dict (+ saved run dir)

  Plot/IO helpers: plot_confusion, plot_per_class_strength,
  plot_permutation_null, plot_feature_importance, load_run, list_runs.
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
# 0 — Reuse the clustering feature builders (single source of truth)
# ============================================================
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]                       # functions -> 03_.. -> repo
_CLUST_FUNCS = _REPO_ROOT / "02_FBM_Clustering" / "functions"


def _load_clustering_module(mod_name: str):
    """Import a module from 02_FBM_Clustering/functions by absolute path.

    Avoids a sys.path collision between the two sibling `functions/` packages
    (02's and 03's). Falls back to a plain `import` if the file isn't found.
    """
    path = _CLUST_FUNCS / f"{mod_name}.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location(f"_clust_{mod_name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)      # type: ignore[union-attr]
        return module
    # Fallback: maybe 02/functions is already importable on sys.path
    if str(_CLUST_FUNCS) not in sys.path:
        sys.path.insert(0, str(_CLUST_FUNCS))
    return importlib.import_module(mod_name)


_lf_features = _load_clustering_module("lf_features")
_lf_hg = _load_clustering_module("lf_hg")
_lf_dataset = _load_clustering_module("lf_dataset")

# Blob segmentation + -1/0/+1 painting are only needed for the discretized
# ('m101') variants, and pull heavier deps (scipy.ndimage, imageio), so load
# them lazily on first use.
_lf_blob = None
_lf_minus101 = None


def _ensure_blob_modules():
    global _lf_blob, _lf_minus101
    if _lf_blob is None:
        _lf_blob = _load_clustering_module("lf_blob_metrics")
    if _lf_minus101 is None:
        _lf_minus101 = _load_clustering_module("lf_minus101")
    return _lf_blob, _lf_minus101

# Re-export the canonical band edges so notebooks + cache params can show them.
FREQ_BANDS = list(_lf_features.FREQ_BANDS_15_TO_400HZ)
prepare_dataset = _lf_dataset.prepare_dataset

try:
    from skimage.transform import resize as _sk_resize
except Exception:  # pragma: no cover
    _sk_resize = None


# ============================================================
# 1 — Constants
# ============================================================
CONDITIONS = ("audio", "picture", "reading")     # fixed concat order for task B
TASK = "LM"
N_FREQ = 129
N_TIME = 300
FMAX_HZ = 500.0                                  # original ERSP freq ceiling
HG_BAND = (70.0, 150.0)
# ── Feature variants: a NESTED, TIME-MATCHED family ─────────────────────────
# The scientific question is "does the full spectrum beat high-gamma alone?".
# For that to be a fair test, HG and FULL must sit on the SAME time grid, so
# the only thing that changes between a matched pair is the frequency content.
# Hence two matched pairs: (full_300 vs hg_300) and (full_30 vs hg_30).
#   - 'full' = the 15 band-aware ERSP rows (1..400 Hz)   -> a band x time map
#   - 'hg'   = the single 70-150 Hz band-mean line        -> a 1-D time series
# CAVEAT (frequency and time go hand in hand): the underlying STFT has a fixed
# time-frequency resolution trade-off, so HG-with-fine-time vs full-with-coarse-
# time can never be a perfectly clean separation. Matching the time grid removes
# the gross confound; the residual coupling is stated in the narrative.
STIM_FRAC = 0.5     # ERSPs are warped 50% stimulus (sensing) / 50% response.

# ── The AMPLITUDE-CONFOUND triad ────────────────────────────────────────────
# Per-feature z-scoring (StandardScaler) makes each band x time CELL comparable
# across electrodes, but it cannot remove a sample's overall MAGNITUDE: a strong
# visually-evoked electrode is "high across many cells at once", so the model can
# decode on "how strong" instead of "what shape". To bracket that confound we run
# three representations of the same full-spectrum grid:
#   - continuous  (full_*)      : the raw band x time values  (amplitude intact)
#   - row-normed  (full_*_rn)   : each sample L2-normalised   (magnitude removed,
#                                  graded shape kept)
#   - discretized (m101_*)      : score-gated -1/0/+1 blob map (amplitude fully
#                                  discarded; ONLY significant, high-score
#                                  segments are painted — low-score blobs stay 0)
# Each at matched time grids {300, 30}. HG line is the continuous-only baseline.
VARIANT_SPEC = {
    "full_300":    {"bands": "full", "n_time": 300, "disc": False, "rownorm": False},
    "hg_300":      {"bands": "hg",   "n_time": 300, "disc": False, "rownorm": False},
    "full_30":     {"bands": "full", "n_time": 30,  "disc": False, "rownorm": False},
    "hg_30":       {"bands": "hg",   "n_time": 30,  "disc": False, "rownorm": False},
    "m101_300":    {"bands": "full", "n_time": 300, "disc": True,  "rownorm": False},
    "m101_30":     {"bands": "full", "n_time": 30,  "disc": True,  "rownorm": False},
    "full_300_rn": {"bands": "full", "n_time": 300, "disc": False, "rownorm": True},
    "full_30_rn":  {"bands": "full", "n_time": 30,  "disc": False, "rownorm": True},
}
VARIANTS = tuple(VARIANT_SPEC.keys())

VARIANT_LABELS = {
    "full_300":    "Full spectrum (15 bands x 300 time)",
    "hg_300":      "High-gamma line (70-150 Hz, 300 time)",
    "full_30":     "Full spectrum (15 bands x 30 time)",
    "hg_30":       "High-gamma line (70-150 Hz, 30 time)",
    "m101_300":    "Discretized -1/0/+1 (score-gated, 15 x 300)",
    "m101_30":     "Discretized -1/0/+1 (score-gated, 15 x 30)",
    "full_300_rn": "Full spectrum, row-normalised (15 x 300)",
    "full_30_rn":  "Full spectrum, row-normalised (15 x 30)",
}
# Amplitude triad per time grid — continuous vs row-normed vs discretized.
AMPLITUDE_TRIADS = (
    ("full_300", "full_300_rn", "m101_300"),
    ("full_30",  "full_30_rn",  "m101_30"),
)
# Full-vs-HG matched pairs (time held fixed) — the frequency-content contrast.
MATCHED_PAIRS = (("full_300", "hg_300"), ("full_30", "hg_30"))

# ── Discretization (-1/0/+1) parameters ─────────────────────────────────────
M101_MAX_BLOBS = 6          # top-N blobs per ERSP (by score) considered
M101_SIGN_MODE = "both"     # paint both positive (+1) and negative (-1) blobs
M101_SCORE_PCT = 33.0       # GATE: drop blobs below this global score percentile
M101_DEADZONE = 0.15        # |band-mean of painted map| below this -> 0 (mixed/weak)
M101_SAMPLE_CAP = 3000      # subsample size for estimating the score percentile
_M101_SCORE_MIN = None      # resolved global score gate (set by resolve_*)

# Anything in this set is NOT a real Yeo network -> dropped from task B.
_YEO_NON_NETWORK = {
    "FreeSurfer_Defined_Medial_Wall", "Medial_Wall", "medial_wall",
    "WhiteMatter", "Unknown", "unknown", "None", "", "nan",
}

OUTPUTS_ROOT = (_HERE.parents[1] / "outputs" / "classification")
DATASET_CACHE = (_HERE.parents[1] / "outputs" / "_dataset" / "classification")
COORDS_DIR = (_REPO_ROOT / "02_FBM_Clustering" / "outputs" / "250_recon"
              / "fsaverage" / "coords")

SCHEMA_VERSION = 1


# ============================================================
# 2 — Electrode <-> fsaverage contact name normalization
#     (identical rule to 02_FBM_Clustering/252_clustering_recon)
# ============================================================
_RX_BERN = re.compile(r"_ERSP_([A-Za-z]+)_([LR])(\d+)_TN", re.IGNORECASE)


def normalize_label(s) -> str:
    """Strip '_' and '-', uppercase. Matches lf_io_utils.normalize_label so
    the ERSP side and the coords side join cleanly ('aH_R-1' -> 'AHR1')."""
    if s is None:
        return ""
    return str(s).replace("_", "").replace("-", "").upper()


def contact_from_row(patient_id: str, electrode: str, file_path: str) -> Optional[str]:
    """Normalized contact key for joining an ERSP sample to a coords row.

    BERN/EL filenames look like  <pat>_<cond>_WM_ERSP_<electrode>_<L|R><num>_TN
      -> contact = electrode + side + num   (e.g. 'A_R10' -> 'AR10')
    GVA/PAT filenames carry the contact directly in the electrode column.
    """
    base = Path(str(file_path).replace("\\", "/")).name.rsplit(".", 1)[0]
    m = _RX_BERN.search(base)
    if m:
        return normalize_label(f"{m.group(1)}{m.group(2)}{m.group(3)}")
    if isinstance(electrode, str) and electrode.strip():
        return normalize_label(electrode)
    return None


# ============================================================
# 3 — Per-sample feature builders + column metadata
# ============================================================
def _freqs_axis(n_freq: int = N_FREQ, fmax: float = FMAX_HZ) -> np.ndarray:
    return np.linspace(0.0, float(fmax), n_freq)


def _resize_time(arr2d: np.ndarray, n_time: int) -> np.ndarray:
    """Anti-aliased resize of a (n_rows, n_time_orig) array to n_time columns."""
    if arr2d.shape[1] == n_time:
        return arr2d
    if _sk_resize is None:
        raise ImportError("scikit-image required for time downsampling.")
    return _sk_resize(arr2d, (arr2d.shape[0], int(n_time)),
                      anti_aliasing=True, preserve_range=True).astype(np.float32)


def _segment_blobs(ersp: np.ndarray):
    """Score-sorted blobs for one ERSP (top M101_MAX_BLOBS)."""
    blob, _ = _ensure_blob_modules()
    return blob.s21_segment_valley_blobs(
        ersp, max_blobs=M101_MAX_BLOBS, sign_mode=M101_SIGN_MODE, fmax=FMAX_HZ)


def resolve_m101_score_min(X_3d: np.ndarray, *, pct: float = M101_SCORE_PCT,
                           sample_cap: int = M101_SAMPLE_CAP, verbose: bool = True
                           ) -> float:
    """Estimate + set the global blob-score gate as the `pct`-th percentile of
    blob scores over (a subsample of) the dataset. Low-score blobs (< this) are
    never painted, so noise/weak segments don't get discretized into +/-1."""
    global _M101_SCORE_MIN
    n = X_3d.shape[0]
    if n == 0:
        _M101_SCORE_MIN = 0.0
        return 0.0
    step = max(1, n // int(sample_cap))
    scores = []
    for i in range(0, n, step):
        scores += [float(b["score"]) for b in _segment_blobs(X_3d[i])]
    _M101_SCORE_MIN = float(np.percentile(scores, pct)) if scores else 0.0
    if verbose:
        print(f"[m101] score gate = {pct:.0f}th pct = {_M101_SCORE_MIN:.4g} "
              f"({len(scores)} blobs over {len(range(0, n, step))} sampled ERSPs)")
    return _M101_SCORE_MIN


def ersp_to_minus101(ersp: np.ndarray, n_time: int) -> np.ndarray:
    """Score-gated -1/0/+1 map on the 15-band x n_time grid (amplitude-free).

    Paints only high-score blobs (>= the resolved global gate), band-averages
    onto the same grid as the 'full' variant, then re-discretizes with a
    deadzone so weak/mixed cells stay 0.
    """
    _, m101 = _ensure_blob_modules()
    score_min = _M101_SCORE_MIN if _M101_SCORE_MIN is not None else 0.0
    blobs = _segment_blobs(ersp)
    painted = m101.paint_minus101_map(ersp, blobs, score_min=score_min)   # 129x300, {-1,0,1}
    grid = _lf_features.downsample_ersp_to_bands(
        painted.astype(np.float32), FREQ_BANDS, fmax_hz=FMAX_HZ, time_bins_out=n_time)
    out = np.zeros_like(grid, dtype=np.float32)
    out[grid > M101_DEADZONE] = 1.0
    out[grid < -M101_DEADZONE] = -1.0
    return out.ravel().astype(np.float32)             # 15 * n_time, in {-1,0,1}


def ersp_to_feature(ersp: np.ndarray, variant: str) -> np.ndarray:
    """Map one (n_freq, n_time) ERSP to a flat 1-D feature vector for `variant`.

    'full' -> 15 band rows x n_time (band x time map, row-major = band outer,
    time inner). 'hg' -> single 70-150 Hz band-mean line at n_time. 'disc' ->
    score-gated -1/0/+1 map on the full grid. Row-normalisation is applied to
    the assembled matrix later (build_*_arrays), not per-ERSP.
    """
    spec = VARIANT_SPEC.get(variant)
    if spec is None:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    n_time = int(spec["n_time"])
    if spec.get("disc"):
        return ersp_to_minus101(ersp, n_time)         # 15 * n_time, {-1,0,1}
    if spec["bands"] == "full":
        ds = _lf_features.downsample_ersp_to_bands(
            ersp, FREQ_BANDS, fmax_hz=FMAX_HZ, time_bins_out=n_time)
        return ds.ravel().astype(np.float32)          # 15 * n_time
    # single HG band-mean line, resized to the variant's time grid
    hg = _lf_hg.extract_hg_time_series(ersp, hg_band=HG_BAND, fmax=FMAX_HZ)
    hg = _resize_time(hg.reshape(1, -1), n_time)
    return hg.ravel().astype(np.float32)              # n_time


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Unit-L2-normalise each row (sample). Removes overall magnitude, keeps
    graded shape. Zero rows are left as zeros."""
    X = np.asarray(X, dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (X / norm).astype(np.float32)


def build_feature_matrix(X_3d: np.ndarray, variant: str, verbose: bool = True
                         ) -> np.ndarray:
    """Stack ersp_to_feature over a (n, n_freq, n_time) array -> (n, d)."""
    n = X_3d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    first = ersp_to_feature(X_3d[0], variant)
    X = np.zeros((n, first.size), dtype=np.float32)
    X[0] = first
    for i in range(1, n):
        X[i] = ersp_to_feature(X_3d[i], variant)
    if verbose:
        print(f"[build_feature_matrix] variant={variant} -> X.shape={X.shape}")
    return X


def feature_columns(variant: str, condition: Optional[str] = None) -> pd.DataFrame:
    """One row per feature column, describing what it means. Powers the
    feature-importance interpretation (which condition / band / time drove it)."""
    spec = VARIANT_SPEC.get(variant)
    if spec is None:
        raise ValueError(variant)
    n_time = int(spec["n_time"])
    rows = []
    if spec["bands"] == "full":
        for bi, (lo, hi) in enumerate(FREQ_BANDS):
            for ti in range(n_time):
                rows.append({"band_idx": bi, "band_lo": lo, "band_hi": hi,
                             "band": f"{lo:.0f}-{hi:.0f}Hz", "time_bin": ti,
                             "kind": "ersp"})
    else:   # single HG line
        for ti in range(n_time):
            rows.append({"band_idx": -1, "band_lo": HG_BAND[0], "band_hi": HG_BAND[1],
                         "band": "HG 70-150Hz", "time_bin": ti, "kind": "hg"})
    df = pd.DataFrame(rows)
    if condition is not None:
        df.insert(0, "condition", condition)
    return df


def concat_feature_columns(variant: str) -> pd.DataFrame:
    """Column metadata for the parcellation task = the single-condition columns
    repeated for [audio, picture, reading] in that fixed order."""
    return pd.concat([feature_columns(variant, c) for c in CONDITIONS],
                     ignore_index=True)


# ============================================================
# 4 — Yeo labels per electrode (read straight from the coords CSVs)
# ============================================================
def load_yeo_lookup(coords_dir: Path = COORDS_DIR, n_networks: int = 7
                    ) -> pd.DataFrame:
    """Read every <pid>_contacts_fsaverage.csv and return a tidy lookup:

        patient_id, contact_norm, yeo_label

    `contact_norm` is normalize_label(name) so it joins to contact_from_row().
    `yeo_label` is the raw network string ('7Networks_3' etc.); non-network
    labels are kept here and filtered downstream so callers can audit them.
    """
    col = "yeo7_network" if int(n_networks) == 7 else "yeo17_network"
    coords_dir = Path(coords_dir)
    files = sorted(coords_dir.glob("*_contacts_fsaverage.csv"))
    if not files:
        raise FileNotFoundError(f"No coords CSVs under {coords_dir}")
    rows = []
    for f in files:
        if "ALL_PATIENTS" in f.name.upper():
            continue
        df = pd.read_csv(f)
        if not {"patient", "name", col}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            rows.append({
                "patient_id":   str(r["patient"]),
                "contact_norm": normalize_label(r["name"]),
                "yeo_label":    str(r[col]),
            })
    out = pd.DataFrame(rows).drop_duplicates(subset=["patient_id", "contact_norm"])
    return out


def is_real_network(label) -> bool:
    return str(label) not in _YEO_NON_NETWORK and not str(label).lower().startswith("nan")


# ============================================================
# 5 — Dataset assembly
# ============================================================
def prepare_full_dataset(input_dir, cache_dir: Path = DATASET_CACHE,
                         verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load EVERY electrode x condition ERSP (ungated) for patients that have
    all three conditions. The high-activity flag is still computed per sample
    (prepare_dataset always returns it) so both tasks can be derived from this
    single object.

    Returns df_meta (+ 'contact_norm', 'has_all3' helper cols) and X_3d.
    """
    df_meta, _ersp, X_3d = prepare_dataset(
        input_dir, task=TASK, conditions=CONDITIONS,
        n_freq=N_FREQ, n_time=N_TIME,
        apply_high_activity=False,                 # keep everything; gate later
        cache_dir=str(cache_dir / "_raw_ungated"),
        verbose=verbose,
    )
    # Restrict to patients that actually have all three conditions present.
    present = (df_meta.groupby("patient_id")["condition"]
               .apply(lambda s: set(s.unique())))
    full = {p for p, conds in present.items() if set(CONDITIONS).issubset(conds)}
    keep = df_meta["patient_id"].isin(full).to_numpy()
    df_meta = df_meta[keep].reset_index(drop=True)
    X_3d = X_3d[keep]
    df_meta["contact_norm"] = [
        contact_from_row(p, e, fp)
        for p, e, fp in zip(df_meta["patient_id"], df_meta["electrode"],
                            df_meta["file_path"])
    ]
    if verbose:
        dropped = sorted(set(present.index) - full)
        print(f"[prepare_full_dataset] kept {len(full)} patients with all 3 "
              f"conditions; dropped {len(dropped)}: {dropped}")
        print(f"  {len(df_meta)} electrode x condition samples · X_3d={X_3d.shape}")
    return df_meta, X_3d


def _ensure_m101_ready(X_3d: np.ndarray, variant: str):
    """Resolve the global blob-score gate before building any 'm101' variant."""
    if VARIANT_SPEC[variant].get("disc") and _M101_SCORE_MIN is None:
        resolve_m101_score_min(X_3d)


def _apply_rownorm(X: np.ndarray, variant: str) -> np.ndarray:
    """L2-normalise each (possibly concatenated) sample for '_rn' variants."""
    return _l2_normalize_rows(X) if VARIANT_SPEC[variant].get("rownorm") else X


def build_condition_arrays(df_meta: pd.DataFrame, X_3d: np.ndarray, variant: str,
                           verbose: bool = True):
    """Task A arrays. One sample per high-activity electrode x condition.

    Returns X (n,d), y (condition strings), groups (patient_id), meta DataFrame,
    cols (feature_columns metadata).
    """
    _ensure_m101_ready(X_3d, variant)
    mask = df_meta["high_activity"].to_numpy().astype(bool)
    meta = df_meta[mask].reset_index(drop=True)
    X = build_feature_matrix(X_3d[mask], variant, verbose=verbose)
    X = _apply_rownorm(X, variant)
    y = meta["condition"].to_numpy()
    groups = meta["patient_id"].to_numpy()
    cols = feature_columns(variant)
    if verbose:
        print(f"[condition:{variant}] X={X.shape}  classes="
              f"{dict(pd.Series(y).value_counts())}  patients={len(set(groups))}")
    return X, y, groups, meta, cols


def build_parcellation_arrays(df_meta: pd.DataFrame, X_3d: np.ndarray,
                              variant: str, n_networks: int = 7,
                              coords_dir: Path = COORDS_DIR, verbose: bool = True):
    """Task B arrays. One sample per electrode; features = concat over the three
    conditions (fixed order audio, picture, reading) of the UNGATED feature
    vectors. Electrode kept iff: has all 3 conditions present, is high-activity
    in >=1 condition, and has a real Yeo network label.

    Returns X (n, 3*d), y (yeo labels), groups (patient_id), meta DataFrame,
    cols (concat_feature_columns metadata).
    """
    _ensure_m101_ready(X_3d, variant)
    df = df_meta.copy()
    df["sample_pos"] = np.arange(len(df))          # row index into X_3d

    # Per-electrode pivot of condition -> X_3d row.
    grp = df.groupby(["patient_id", "contact_norm"])
    yeo = load_yeo_lookup(coords_dir, n_networks=n_networks)
    yeo_map = {(r.patient_id, r.contact_norm): r.yeo_label
               for r in yeo.itertuples(index=False)}

    feats, labels, groups, meta_rows = [], [], [], []
    n_no_label, n_not_real, n_missing_cond, n_no_activity = 0, 0, 0, 0
    for (pid, contact), sub in grp:
        cond_to_pos = dict(zip(sub["condition"], sub["sample_pos"]))
        if not set(CONDITIONS).issubset(cond_to_pos):
            n_missing_cond += 1
            continue
        if not bool(sub["high_activity"].any()):
            n_no_activity += 1
            continue
        lab = yeo_map.get((pid, contact))
        if lab is None:
            n_no_label += 1
            continue
        if not is_real_network(lab):
            n_not_real += 1
            continue
        vec = np.concatenate([
            ersp_to_feature(X_3d[cond_to_pos[c]], variant) for c in CONDITIONS
        ])
        feats.append(vec.astype(np.float32))
        labels.append(lab)
        groups.append(pid)
        meta_rows.append({"patient_id": pid, "contact_norm": contact,
                          "yeo_label": lab})

    X = np.vstack(feats) if feats else np.zeros((0, 0), np.float32)
    X = _apply_rownorm(X, variant)          # normalise the CONCATENATED vector
    y = np.asarray(labels)
    groups = np.asarray(groups)
    meta = pd.DataFrame(meta_rows)
    cols = concat_feature_columns(variant)
    if verbose:
        print(f"[parcel:yeo{n_networks}:{variant}] X={X.shape}  "
              f"classes={meta['yeo_label'].nunique() if len(meta) else 0}  "
              f"patients={len(set(groups))}")
        print(f"  dropped: missing_cond={n_missing_cond} no_activity={n_no_activity} "
              f"no_label={n_no_label} non_network={n_not_real}")
        if len(meta):
            print("  class counts:", dict(pd.Series(y).value_counts()))
    return X, y, groups, meta, cols


# ============================================================
# 6 — Estimators
# ============================================================
def make_estimator(name: str, random_state: int = 42):
    """Return (pipeline, param_grid) for 'logreg' or 'rf'. Scaler is folded
    into the pipeline so it is fit on TRAIN ONLY inside each CV split."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=2000,
                class_weight="balanced", multi_class="multinomial",
                random_state=random_state)),
        ])
        grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
        return pipe, grid
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", RandomForestClassifier(
                n_estimators=400, class_weight="balanced_subsample",
                n_jobs=-1, random_state=random_state)),
        ])
        grid = {"clf__max_depth": [None, 8, 16],
                "clf__min_samples_leaf": [1, 3]}
        return pipe, grid
    raise ValueError(f"unknown estimator {name!r}")


CLASSIFIER_LABELS = {"logreg": "Logistic Regression (multinomial)",
                     "rf": "Random Forest"}


# ============================================================
# 7 — Nested GroupKFold + metrics + permutation null
# ============================================================
def _per_class_recall(y_true, y_pred, classes) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.diag(cm) / cm.sum(axis=1)
    return np.nan_to_num(rec)


def nested_cv_predict(pipe, grid, X, y, groups, *, outer_splits=5, inner_splits=3,
                      scoring="balanced_accuracy", verbose=True):
    """Run nested GroupKFold. Returns out-of-fold predictions for every sample
    (each predicted exactly once, by a model that never saw its patient), plus
    the per-fold chosen hyper-parameters.

    Returns dict: y_true, y_pred, y_proba, classes, fold, best_params (list).
    """
    from sklearn.model_selection import GroupKFold, GridSearchCV

    classes = np.array(sorted(np.unique(y)))
    n_groups = len(np.unique(groups))
    outer_splits = int(min(outer_splits, n_groups))
    inner_splits = int(min(inner_splits, max(2, n_groups - 1)))

    oof_pred = np.empty(len(y), dtype=object)
    oof_proba = np.zeros((len(y), len(classes)), dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    best_params: List[dict] = []

    outer = GroupKFold(n_splits=outer_splits)
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        inner = GroupKFold(n_splits=int(min(inner_splits,
                                             len(np.unique(groups[tr])))))
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=-1,
                          refit=True)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        best_params.append(gs.best_params_)
        oof_pred[te] = gs.predict(X[te])
        # align proba columns to `classes`; a rare class can be absent from a
        # training fold, so map column-by-column with a zero fallback.
        proba = gs.predict_proba(X[te])
        present = list(gs.classes_)
        proba_full = np.zeros((len(te), len(classes)))
        for j, c in enumerate(classes):
            if c in present:
                proba_full[:, j] = proba[:, present.index(c)]
        oof_proba[te] = proba_full
        oof_fold[te] = k
        if verbose:
            print(f"  outer fold {k+1}/{outer_splits}: best={gs.best_params_}")
    return {"y_true": np.asarray(y), "y_pred": oof_pred, "y_proba": oof_proba,
            "classes": classes, "fold": oof_fold, "best_params": best_params}


def compute_metrics(y_true, y_pred, y_proba, classes, groups, *, n_boot=1000,
                    random_state=42):
    """Confusion matrix, overall scores, and per-class strength with bootstrap
    CIs (resampling whole PATIENTS so CIs respect the grouped design)."""
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                  accuracy_score, confusion_matrix,
                                  roc_auc_score, precision_recall_fscore_support)
    classes = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro",
                        zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)

    # one-vs-rest ROC AUC per class
    y_bin = np.array([[1 if t == c else 0 for c in classes] for t in y_true])
    aucs = []
    for ci in range(len(classes)):
        try:
            aucs.append(roc_auc_score(y_bin[:, ci], y_proba[:, ci]))
        except Exception:
            aucs.append(float("nan"))

    # Bootstrap over patients for per-class recall + balanced accuracy CIs.
    rng = np.random.RandomState(random_state)
    uniq = np.unique(groups)
    boot_rec = np.zeros((n_boot, len(classes)))
    boot_bal = np.zeros(n_boot)
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        boot_rec[b] = _per_class_recall(yt[idx], yp[idx], classes)
        try:
            boot_bal[b] = balanced_accuracy_score(yt[idx], yp[idx])
        except Exception:
            boot_bal[b] = np.nan
    rec_lo, rec_hi = np.nanpercentile(boot_rec, [2.5, 97.5], axis=0)
    bal_lo, bal_hi = np.nanpercentile(boot_bal, [2.5, 97.5])

    per_class = pd.DataFrame({
        "class": classes, "support": support,
        "precision": prec, "recall": rec, "f1": f1,
        "recall_ci_lo": rec_lo, "recall_ci_hi": rec_hi,
        "roc_auc_ovr": aucs,
    })
    overall = {
        "balanced_accuracy": float(bal_acc),
        "balanced_accuracy_ci": [float(bal_lo), float(bal_hi)],
        "macro_f1": float(macro_f1), "accuracy": float(acc),
        "chance_level": float(1.0 / len(classes)),
        "n_samples": int(len(y_true)), "n_classes": len(classes),
        "n_patients": int(len(uniq)),
    }
    return cm, overall, per_class


def permutation_test(pipe, grid, X, y, groups, classes, observed_bal,
                     observed_rec, *, n_perm=200, outer_splits=5,
                     scoring="balanced_accuracy", random_state=42, verbose=True):
    """Grouped label-permutation null for overall balanced accuracy AND each
    class's recall. To keep it tractable we DON'T re-tune inside the null: we
    refit the pipeline at its default params under a single GroupKFold. Returns
    overall p, per-class p (FDR-corrected), and the null distributions.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score

    if n_perm <= 0:
        return {"overall_p": None, "per_class_p": None, "per_class_p_fdr": None,
                "null_bal": [], "null_rec": []}

    rng = np.random.RandomState(random_state)
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=int(min(outer_splits, n_groups)))
    null_bal = np.zeros(n_perm)
    null_rec = np.zeros((n_perm, len(classes)))
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        pred = cross_val_predict(pipe, X, y_perm, groups=groups, cv=cv, n_jobs=-1)
        null_bal[i] = balanced_accuracy_score(y_perm, pred)
        null_rec[i] = _per_class_recall(y_perm, pred, classes)
        if verbose and (i + 1) % max(1, n_perm // 5) == 0:
            print(f"  permutation {i+1}/{n_perm}")
    overall_p = (1 + int(np.sum(null_bal >= observed_bal))) / (n_perm + 1)
    per_class_p = np.array([
        (1 + int(np.sum(null_rec[:, ci] >= observed_rec[ci]))) / (n_perm + 1)
        for ci in range(len(classes))
    ])
    per_class_p_fdr = _fdr_bh(per_class_p)
    return {"overall_p": float(overall_p),
            "per_class_p": per_class_p.tolist(),
            "per_class_p_fdr": per_class_p_fdr.tolist(),
            "null_bal": null_bal.tolist(), "null_rec": null_rec.tolist()}


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1)
    return out


# ============================================================
# 8 — Feature importance ("why" each class separates)
# ============================================================
def feature_importance(estimator_name, X, y, groups, cols, classes, *,
                       random_state=42, n_repeats=10, verbose=True):
    """Fit the chosen estimator on ALL data (after grouped tuning) and extract
    importances, mapped back to (condition / band / time). For logreg we return
    signed per-class standardized coefficients; for rf, impurity + permutation
    importance. Aggregates by band and by condition for the narrative plots.

    Returns dict with 'per_feature' (DataFrame) and 'aggregated' (dict of
    DataFrames keyed 'by_band', 'by_condition', 'by_time').
    """
    from sklearn.model_selection import GroupKFold, GridSearchCV
    from sklearn.inspection import permutation_importance

    pipe, grid = make_estimator(estimator_name, random_state)
    inner = GroupKFold(n_splits=int(min(3, len(np.unique(groups)))))
    gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=inner,
                      n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    best = gs.best_estimator_
    clf = best.named_steps["clf"]

    pf = cols.copy().reset_index(drop=True)
    if estimator_name == "logreg":
        coef = clf.coef_                       # (n_classes, n_features) or (1, n)
        if coef.shape[0] == 1:                 # binary edge-case
            coef = np.vstack([-coef[0], coef[0]])
        for ci, c in enumerate(clf.classes_):
            pf[f"coef[{c}]"] = coef[ci]
        pf["importance"] = np.abs(coef).mean(axis=0)
    else:
        pf["impurity_importance"] = clf.feature_importances_
        # Per-feature permutation importance is O(n_features * n_repeats) model
        # evals — intractable for the high-dim 'full_*' variants (4500-13500
        # features). Run it only when the feature count is modest; otherwise fall
        # back to impurity importance (still mapped to band/condition/time).
        PERM_IMP_MAX_FEATURES = 1200
        if X.shape[1] <= PERM_IMP_MAX_FEATURES:
            perm = permutation_importance(best, X, y, n_repeats=n_repeats,
                                          random_state=random_state, n_jobs=-1,
                                          scoring="balanced_accuracy")
            pf["perm_importance"] = perm.importances_mean
            pf["perm_importance_std"] = perm.importances_std
            pf["importance"] = pf["perm_importance"].clip(lower=0)
        else:
            if verbose:
                print(f"  [feature_importance] {X.shape[1]} features > "
                      f"{PERM_IMP_MAX_FEATURES}: using impurity importance "
                      f"(skipping slow per-feature permutation importance)")
            pf["importance"] = pf["impurity_importance"]

    agg = {}
    if "band" in pf.columns:
        agg["by_band"] = (pf.groupby("band", sort=False)["importance"].sum()
                          .reset_index())
    if "condition" in pf.columns:
        agg["by_condition"] = (pf.groupby("condition", sort=False)["importance"]
                               .sum().reset_index())
    if "time_bin" in pf.columns:
        agg["by_time"] = (pf.groupby("time_bin")["importance"].sum()
                          .reset_index())
    if verbose:
        print(f"[feature_importance:{estimator_name}] best={gs.best_params_}")
    return {"per_feature": pf, "aggregated": agg, "best_params": gs.best_params_}


# ============================================================
# 9 — Plot helpers (write PNGs into a run dir)
# ============================================================
def plot_confusion(cm, classes, out_png, title="", normalize=True):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    disp = cm.copy()
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            disp = cm / cm.sum(axis=1, keepdims=True)
        disp = np.nan_to_num(disp)
    fig, ax = plt.subplots(figsize=(1.3 + 0.6 * len(classes),
                                    1.2 + 0.6 * len(classes)))
    im = ax.imshow(disp, cmap="magma", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thr = (disp.max() if disp.size else 1) * 0.6
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = disp[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if val < thr else "black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_strength(per_class, overall, out_png, perm=None, title=""):
    """Per-class recall bars with bootstrap CIs, chance line, and significance
    stars from the permutation FDR p-values."""
    import matplotlib.pyplot as plt
    pc = per_class.reset_index(drop=True)
    classes = pc["class"].tolist()
    rec = pc["recall"].to_numpy()
    lo = pc["recall_ci_lo"].to_numpy(); hi = pc["recall_ci_hi"].to_numpy()
    err = np.vstack([rec - lo, hi - rec])
    fig, ax = plt.subplots(figsize=(1.5 + 0.55 * len(classes), 3.2))
    bars = ax.bar(range(len(classes)), rec, yerr=err, capsize=3,
                  color="#4363d8", alpha=0.85)
    ax.axhline(overall["chance_level"], ls="--", color="#888",
               label=f"chance = {overall['chance_level']:.2f}")
    if perm and perm.get("per_class_p_fdr"):
        for i, p in enumerate(perm["per_class_p_fdr"]):
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            ax.text(i, min(1.0, hi[i] + 0.03), star, ha="center", fontsize=8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Recall (sensitivity)")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_null(perm, observed_bal, out_png, title=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    null = np.asarray(perm.get("null_bal", []))
    if null.size:
        ax.hist(null, bins=30, color="#bbb", edgecolor="white",
                label="permutation null")
    ax.axvline(observed_bal, color="#cc0033", lw=2,
               label=f"observed = {observed_bal:.3f}")
    p = perm.get("overall_p")
    sub = f"  p = {p:.4f}" if p is not None else ""
    ax.set_xlabel("Balanced accuracy"); ax.set_ylabel("count")
    ax.set_title(title + sub, fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(imp, out_png, title=""):
    """Bar charts of aggregated importance by band, condition (if present),
    and time."""
    import matplotlib.pyplot as plt
    agg = imp["aggregated"]
    panels = [k for k in ("by_condition", "by_band", "by_time") if k in agg]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, key in zip(axes, panels):
        d = agg[key]
        xcol = d.columns[0]
        ax.bar(d[xcol].astype(str), d["importance"], color="#27ae60", alpha=0.85)
        ax.set_title(key.replace("by_", "by "), fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9b — Heatmap interpretations ("values x on regions y")
# ============================================================
def _zscore_cols(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _col_group_labels(cols: pd.DataFrame, variant: str, n_time_buckets: int = 30
                      ) -> List[str]:
    """Compact, ordered column-group label per feature column for the
    class x feature heatmap.
      full_* -> (cond|)band            (15 or 45 groups)
      hg_*   -> (cond|)t{bucket}        (<=30 or <=90 groups)
    """
    has_cond = "condition" in cols.columns
    is_full = VARIANT_SPEC.get(variant, {}).get("bands") == "full"
    if is_full:
        keyer = lambda r: str(r["band"])
    else:
        tb = cols["time_bin"].to_numpy()
        tmax = int(tb.max()) + 1 if len(tb) else 1
        keyer = lambda r: f"t{int(int(r['time_bin']) * n_time_buckets / tmax):02d}"
    labels = []
    for _, r in cols.iterrows():
        cond = (f"{str(r['condition'])[:3]}|" if has_cond else "")
        labels.append(cond + keyer(r))
    return labels


def class_feature_heatmap_matrix(X, y, cols, classes, variant) -> pd.DataFrame:
    """rows = classes, cols = grouped features, value = per-class mean of the
    z-scored feature (averaged within each column group). Classifier-agnostic —
    this is the data's own per-class signature."""
    Xz = _zscore_cols(X)
    labels = np.asarray(_col_group_labels(cols, variant))
    order = list(dict.fromkeys(labels.tolist()))
    by_group = np.zeros((Xz.shape[0], len(order)), dtype=float)
    for j, g in enumerate(order):
        idx = np.where(labels == g)[0]
        by_group[:, j] = Xz[:, idx].mean(axis=1)
    y = np.asarray(y)
    mat = np.full((len(classes), len(order)), np.nan)
    for i, c in enumerate(classes):
        m = y == c
        if m.any():
            mat[i] = by_group[m].mean(axis=0)
    return pd.DataFrame(mat, index=[str(c) for c in classes], columns=order)


def coef_heatmap_matrix(imp, cols, variant) -> Optional[pd.DataFrame]:
    """rows = classes, cols = grouped features, value = mean signed LR
    coefficient per group. Returns None for estimators without per-class coefs."""
    pf = imp["per_feature"]
    coef_cols = [c for c in pf.columns if c.startswith("coef[")]
    if not coef_cols:
        return None
    labels = np.asarray(_col_group_labels(cols, variant))
    order = list(dict.fromkeys(labels.tolist()))
    rows, idx = [], []
    for cc in coef_cols:
        vals = pf[cc].to_numpy()
        rows.append([np.nanmean(vals[labels == g]) for g in order])
        idx.append(cc[len("coef["):-1])
    return pd.DataFrame(rows, index=idx, columns=order)


def plot_class_heatmap(mat: pd.DataFrame, out_png, title="", cmap="RdBu_r",
                       cbar_label="mean z-scored response"):
    import matplotlib.pyplot as plt
    M = mat.to_numpy(dtype=float)
    if M.size == 0:
        return
    vmax = np.nanmax(np.abs(M)) or 1.0
    nrow, ncol = M.shape
    fig, ax = plt.subplots(figsize=(min(22, max(4, 0.30 * ncol + 2.5)),
                                    max(2.4, 0.42 * nrow + 1.2)))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
    ax.set_xlabel("feature  (frequency band / time, per condition)")
    ax.set_ylabel("class (region / condition)")
    ax.set_title(title, fontsize=9)
    if ncol <= 20 and nrow <= 12:
        for i in range(nrow):
            for j in range(ncol):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                            fontsize=6,
                            color="black" if abs(M[i, j]) < 0.6 * vmax else "white")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_class_ersp_profiles(X_full, y, classes, out_png, *, n_time, n_cond,
                             cond_names=None, title="", zscore=False,
                             stim_frac=STIM_FRAC):
    """Per-class mean FULL-spectrum ERSP (15 bands x n_time), with the n_cond
    condition blocks concatenated along time.

    This is the per-class *response profile* the comparison actually needs: a
    real spectro-temporal map (not a collapsed line). A dashed gray line marks
    the stimulus->response boundary (`stim_frac`) inside EACH condition block --
    first half = stimulus sensing (viewing / hearing), second half = response.
    Solid lines separate the conditions.

    X_full must be the 'full' feature matrix: 15*n_time per condition block,
    concatenated in CONDITIONS order (n_cond=1 for the condition task, where each
    class IS a condition; n_cond=3 for the parcellation task).
    """
    import matplotlib.pyplot as plt
    if X_full is None or len(X_full) == 0:
        return
    X = np.asarray(X_full, dtype=float)
    nb = len(FREQ_BANDS)
    if X.shape[1] != n_cond * nb * n_time:
        return  # not the full-spectrum layout — skip rather than mislabel
    if zscore:
        X = _zscore_cols(X)
    band_lbls = [f"{lo:.0f}-{hi:.0f}" for lo, hi in FREQ_BANDS]
    y = np.asarray(y)
    imgs, vmax = [], 0.0
    for c in classes:
        m = y == c
        v = X[m].mean(axis=0) if m.any() else np.zeros(X.shape[1])
        blocks = v.reshape(n_cond, nb, n_time)
        wide = np.concatenate([blocks[ci] for ci in range(n_cond)], axis=1)
        imgs.append(wide); vmax = max(vmax, float(np.nanmax(np.abs(wide))))
    vmax = vmax or 1.0
    n = len(classes)
    sep = n_cond * n_time
    fig, axes = plt.subplots(n, 1, squeeze=False,
                             figsize=(max(5.0, 1.9 * n_cond + 1.6),
                                      max(2.0, 0.85 * n + 1.0)))
    im = None
    for k, (c, wide) in enumerate(zip(classes, imgs)):
        ax = axes[k][0]
        im = ax.imshow(wide, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=[0, sep, 0, nb])
        for ci in range(n_cond):
            x0 = ci * n_time
            ax.axvline(x0 + stim_frac * n_time, color="0.35", lw=1.0, ls="--")
            if ci > 0:
                ax.axvline(x0, color="black", lw=1.4)
        ax.set_title(str(c), fontsize=7, loc="left", pad=1)
        ax.set_yticks(np.arange(nb) + 0.5)
        ax.set_yticklabels(band_lbls, fontsize=3.4)
        if k < n - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks([(ci + 0.5) * n_time for ci in range(n_cond)])
            ax.set_xticklabels(list(cond_names) if cond_names else [""] * n_cond,
                               fontsize=8)
    fig.suptitle(title + "  — per-class ERSP   (dashed grey = stim→response, "
                 "left half = sensing · right half = response)", fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02,
                     label=("z-scored" if zscore else "mean dB") + " response")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10 — Orchestrator: run_experiment + save + index
# ============================================================
def _now_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def run_experiment(task, variant, classifier, X, y, groups, cols, meta, *,
                   n_networks=None, outputs_root: Path = OUTPUTS_ROOT,
                   outer_splits=5, inner_splits=3, n_perm=200, n_boot=1000,
                   random_state=42, do_importance=True, profile=None,
                   verbose=True) -> dict:
    """Run ONE experiment end-to-end and persist a run dir mirroring the
    clustering layout: outputs/classification/<task>/<variant>/<classifier>/runs/<id>/.

    task        : 'condition' | 'parcellation_yeo7' | 'parcellation_yeo17'
    classifier  : 'logreg' | 'rf'
    Returns the manifest dict.
    """
    if len(np.unique(groups)) < 2 or len(X) == 0:
        raise ValueError(f"Not enough data/patients for {task}/{variant}/{classifier}")

    run_id = _now_id()
    run_dir = Path(outputs_root) / task / variant / classifier / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n=== {task} | {variant} | {classifier} -> {run_dir} ===")

    pipe, grid = make_estimator(classifier, random_state)

    # --- nested CV (the headline estimate) ---
    cv = nested_cv_predict(pipe, grid, X, y, groups,
                           outer_splits=outer_splits, inner_splits=inner_splits,
                           verbose=verbose)
    classes = cv["classes"]
    cm, overall, per_class = compute_metrics(
        cv["y_true"], cv["y_pred"], cv["y_proba"], classes, groups,
        n_boot=n_boot, random_state=random_state)
    observed_rec = per_class["recall"].to_numpy()

    # --- permutation null (significance) ---
    perm = permutation_test(pipe, grid, X, y, groups, classes,
                            overall["balanced_accuracy"], observed_rec,
                            n_perm=n_perm, outer_splits=outer_splits,
                            random_state=random_state, verbose=verbose)
    overall["permutation_p"] = perm.get("overall_p")
    per_class["perm_p"] = perm.get("per_class_p")
    per_class["perm_p_fdr"] = perm.get("per_class_p_fdr")

    # --- feature importance ("why") ---
    imp = None
    if do_importance:
        imp = feature_importance(classifier, X, y, groups, cols, classes,
                                 random_state=random_state, verbose=verbose)

    # --- save artifacts ---
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(run_dir / "confusion_matrix.csv")
    per_class.to_csv(run_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame({"patient_id": groups, "y_true": cv["y_true"],
                  "y_pred": cv["y_pred"], "fold": cv["fold"]}
                 ).to_csv(run_dir / "predictions.csv", index=False)
    _write_json(run_dir / "metrics.json",
                {"overall": overall,
                 "per_class": per_class.to_dict(orient="records")})
    _write_json(run_dir / "permutation_null.json",
                {"overall_p": perm.get("overall_p"),
                 "null_bal": perm.get("null_bal"),
                 "per_class_p": perm.get("per_class_p"),
                 "per_class_p_fdr": perm.get("per_class_p_fdr")})
    if imp is not None:
        imp["per_feature"].to_csv(run_dir / "feature_importance.csv", index=False)
        for k, d in imp["aggregated"].items():
            d.to_csv(run_dir / f"feature_importance_{k}.csv", index=False)

    # --- figures ---
    title = f"{task} · {VARIANT_LABELS[variant]} · {CLASSIFIER_LABELS[classifier]}"
    plot_confusion(cm, classes, run_dir / "confusion_matrix.png",
                   title=title, normalize=True)
    plot_per_class_strength(per_class, overall, run_dir / "per_class_strength.png",
                            perm=perm, title=title)
    if perm.get("null_bal"):
        plot_permutation_null(perm, overall["balanced_accuracy"],
                              run_dir / "permutation_null.png", title=title)
    if imp is not None:
        plot_feature_importance(imp, run_dir / "feature_importance.png", title=title)

    # --- heatmap interpretations (values x on regions y) ---
    try:
        cfmat = class_feature_heatmap_matrix(X, y, cols, classes, variant)
        cfmat.to_csv(run_dir / "class_feature_heatmap.csv")
        plot_class_heatmap(cfmat, run_dir / "class_feature_heatmap.png",
                           title=title + " — per-class response profile")
        # full-spectrum, concatenated, stim-aware per-class ERSP profile
        if profile is not None:
            plot_class_ersp_profiles(
                profile["X"], y, classes, run_dir / "class_ersp_profile.png",
                n_time=profile["n_time"], n_cond=profile["n_cond"],
                cond_names=profile.get("cond_names"), title=title)
        if imp is not None:
            cmat = coef_heatmap_matrix(imp, cols, variant)
            if cmat is not None:
                cmat.to_csv(run_dir / "coef_heatmap.csv")
                plot_class_heatmap(cmat, run_dir / "coef_heatmap.png",
                                   title=title + " — LR coefficients (class x band)",
                                   cbar_label="mean signed coefficient")
    except Exception as e:                                # never kill a run on a plot
        print("  [warn] heatmap step skipped:", repr(e))

    # --- manifest + index ---
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task, "variant": variant, "classifier": classifier,
        "n_networks": n_networks, "run_id": run_id,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "user": _safe_user(),
        "feature_dim": int(X.shape[1]), "classes": list(map(str, classes)),
        "cv": {"scheme": "nested GroupKFold (by patient)",
               "outer_splits": int(outer_splits), "inner_splits": int(inner_splits),
               "best_params_per_fold": cv["best_params"]},
        "n_permutations": int(n_perm), "n_bootstrap": int(n_boot),
        "summary": overall,
        "path": str(run_dir.relative_to(Path(outputs_root))),
    }
    _write_json(run_dir / "manifest.json", manifest)
    _update_index(Path(outputs_root), manifest)
    if verbose:
        p = overall.get("permutation_p")
        print(f"  bal_acc={overall['balanced_accuracy']:.3f} "
              f"(chance {overall['chance_level']:.3f})  macro_F1={overall['macro_f1']:.3f}"
              + (f"  p={p:.4f}" if p is not None else ""))
    return manifest


def _safe_user():
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _update_index(outputs_root: Path, manifest: dict):
    idx_path = outputs_root / "index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    else:
        idx = {"schema_version": SCHEMA_VERSION, "runs": []}
    row = {k: manifest[k] for k in ("task", "variant", "classifier",
                                    "n_networks", "run_id", "created_at", "path")}
    row["balanced_accuracy"] = manifest["summary"]["balanced_accuracy"]
    row["macro_f1"] = manifest["summary"]["macro_f1"]
    row["chance_level"] = manifest["summary"]["chance_level"]
    row["permutation_p"] = manifest["summary"].get("permutation_p")
    idx["runs"] = [r for r in idx["runs"]
                   if not (r.get("path") == row["path"])] + [row]
    _write_json(idx_path, idx)


# ============================================================
# 11 — Loading helpers for the results / narrative notebook
# ============================================================
def list_runs(outputs_root: Path = OUTPUTS_ROOT) -> pd.DataFrame:
    """Tidy table of every run in index.json (latest per combo first)."""
    idx_path = Path(outputs_root) / "index.json"
    if not idx_path.exists():
        return pd.DataFrame()
    runs = json.loads(idx_path.read_text()).get("runs", [])
    df = pd.DataFrame(runs)
    if len(df):
        df = df.sort_values("created_at").reset_index(drop=True)
    return df


def latest_run_dir(task, variant, classifier, outputs_root: Path = OUTPUTS_ROOT
                   ) -> Optional[Path]:
    base = Path(outputs_root) / task / variant / classifier / "runs"
    if not base.exists():
        return None
    runs = sorted([d for d in base.iterdir() if d.is_dir()])
    return runs[-1] if runs else None


# ============================================================
# 12 — Feature-cache I/O (written by 310, read by 320 / 330)
# ============================================================
def _cache_dir_for(kind: str, key: Optional[str], variant: str) -> Path:
    """kind='condition' (key=None) | 'parcellation' (key='yeo7'/'yeo17')."""
    if key is None:
        return DATASET_CACHE / kind / variant
    return DATASET_CACHE / kind / key / variant


def save_arrays(kind, key, variant, X, y, groups, meta, cols) -> Path:
    d = _cache_dir_for(kind, key, variant)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", np.asarray(X, dtype=np.float32))
    np.save(d / "y.npy", np.asarray(y))
    np.save(d / "groups.npy", np.asarray(groups))
    meta.to_parquet(d / "meta.parquet", index=False)
    cols.to_parquet(d / "cols.parquet", index=False)
    return d


def load_arrays(kind, key, variant):
    d = _cache_dir_for(kind, key, variant)
    X = np.load(d / "X.npy", allow_pickle=False)
    y = np.load(d / "y.npy", allow_pickle=True)
    groups = np.load(d / "groups.npy", allow_pickle=True)
    meta = pd.read_parquet(d / "meta.parquet")
    cols = pd.read_parquet(d / "cols.parquet")
    return X, y, groups, meta, cols


def load_run(task, variant, classifier, outputs_root: Path = OUTPUTS_ROOT,
             run_id: str = "latest") -> Optional[dict]:
    """Load a run's manifest + metrics + per-class table + figure paths for
    display in the narrative notebook."""
    base = Path(outputs_root) / task / variant / classifier / "runs"
    if run_id == "latest":
        rd = latest_run_dir(task, variant, classifier, outputs_root)
    else:
        rd = base / run_id
    if rd is None or not rd.exists():
        return None
    out = {"run_dir": rd,
           "manifest": json.loads((rd / "manifest.json").read_text()),
           "metrics": json.loads((rd / "metrics.json").read_text()),
           "per_class": pd.read_csv(rd / "per_class_metrics.csv"),
           "confusion": pd.read_csv(rd / "confusion_matrix.csv", index_col=0),
           "figures": {p.stem: p for p in rd.glob("*.png")}}
    return out
