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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Benign + very chatty under imbalanced Yeo classes with grouped CV / bootstrap:
# a resample/fold's y_true may lack a rare class the model still predicts. The
# metric just averages over present classes, so the numbers are correct — we
# silence only this one specific message (fires up to n_boot times per run).
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


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
# VARIANT_SPEC = {
#     "full_300":    {"bands": "full", "n_time": 300, "disc": False, "rownorm": False},
#     "hg_300":      {"bands": "hg",   "n_time": 300, "disc": False, "rownorm": False},
#     "full_30":     {"bands": "full", "n_time": 30,  "disc": False, "rownorm": False},
#     "hg_30":       {"bands": "hg",   "n_time": 30,  "disc": False, "rownorm": False},
#     "m101_300":    {"bands": "full", "n_time": 300, "disc": True,  "rownorm": False},
#     "m101_30":     {"bands": "full", "n_time": 30,  "disc": True,  "rownorm": False},
#     "full_300_rn": {"bands": "full", "n_time": 300, "disc": False, "rownorm": True},
#     "full_30_rn":  {"bands": "full", "n_time": 30,  "disc": False, "rownorm": True},
# }

VARIANT_SPEC = {
    "full_300":    {"bands": "full", "n_time": 300, "disc": False, "rownorm": False},
    "hg_300":      {"bands": "hg",   "n_time": 300, "disc": False, "rownorm": False},
    "full_300_rn": {"bands": "full", "n_time": 300, "disc": False, "rownorm": True},
    "hg_300_rn":   {"bands": "hg",   "n_time": 300, "disc": False, "rownorm": True},
    "full_30":     {"bands": "full", "n_time": 30,  "disc": False, "rownorm": False},
    "hg_30":       {"bands": "hg",   "n_time": 30,  "disc": False, "rownorm": False},
    "full_30_rn":  {"bands": "full", "n_time": 30,  "disc": False, "rownorm": True},
    "hg_30_rn":    {"bands": "hg",   "n_time": 30,  "disc": False, "rownorm": True},
}

VARIANTS = tuple(VARIANT_SPEC.keys())

VARIANT_LABELS = {
    "full_300":    "Full spectrum 300 time",
    "hg_300":      "High-gamma line 300 time",
    "full_300_rn": "Full spectrum, row-normalised 300 time",
    "hg_300_rn":   "High-gamma line, row-normalised 300 time",
    "full_30":     "Full spectrum 30 time",
    "hg_30":       "High-gamma line 30 time",
    "full_30_rn":  "Full spectrum, row-normalised 30 time",
    "hg_30_rn":    "High-gamma line, row-normalised 30 time",
}

# Amplitude triad per time grid — continuous vs row-normed vs discretized.
AMPLITUDE_TRIADS = (
    ("full_300", "full_300_rn", "hg_300", "hg_300_rn"),
    ("full_30",  "full_30_rn",  "hg_30",  "hg_30_rn"),
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
                # multinomial is the default for multiclass + lbfgs (sklearn >=1.5);
                # passing multi_class explicitly is deprecated, so we leave it off.
                penalty="l2", solver="lbfgs", max_iter=2000,
                class_weight="balanced", random_state=random_state)),
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


def plot_coef_ersp_maps(imp, classes, out_png, *, n_time, n_cond,
                        cond_names=None, title="", stim_frac=STIM_FRAC):
    """Per-class LR coefficients laid out as the FULL band x time ERSP map
    (n_cond condition blocks concatenated along time) — same layout as the
    response profile, but showing WHAT THE MODEL USES rather than what the class
    looks like. Red = this time-frequency cell pushes an electrode TOWARD the
    class, blue = away. Dashed grey = stim->response boundary inside each block.

    LR only (RF has no per-class coefficients); 'full'/'m101' band-grid variants
    only (skips the single-line HG variants).
    """
    import matplotlib.pyplot as plt
    pf = imp.get("per_feature") if isinstance(imp, dict) else None
    if pf is None:
        return
    coef_cols = {c[len("coef["):-1]: c for c in pf.columns if c.startswith("coef[")}
    if not coef_cols:
        return                                   # not an LR run
    nb = len(FREQ_BANDS)
    if pf.shape[0] != n_cond * nb * n_time:
        return                                   # not the band x time layout
    band_lbls = [f"{lo:.0f}-{hi:.0f}" for lo, hi in FREQ_BANDS]
    use = [c for c in classes if str(c) in coef_cols] or list(coef_cols)
    imgs, vmax = [], 0.0
    for c in use:
        v = pf[coef_cols[str(c)]].to_numpy(dtype=float)
        blocks = v.reshape(n_cond, nb, n_time)          # [cond][band][time]
        wide = np.concatenate([blocks[ci] for ci in range(n_cond)], axis=1)
        imgs.append(wide); vmax = max(vmax, float(np.nanmax(np.abs(wide))))
    vmax = vmax or 1.0
    n = len(use); sep = n_cond * n_time
    fig, axes = plt.subplots(n, 1, squeeze=False,
                             figsize=(max(5.0, 1.9 * n_cond + 1.6),
                                      max(2.0, 0.85 * n + 1.0)))
    im = None
    for k, (c, wide) in enumerate(zip(use, imgs)):
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
    fig.suptitle(title + "  — LR coefficients as band×time   (red → toward class · "
                 "dashed grey = stim→response)", fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02,
                     label="signed LR coefficient")
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
    # Standardized distance from the permutation null (useful once the empirical
    # p floors at 1/(n_perm+1)): z = (observed - null_mean) / null_std.
    _nb = np.asarray(perm.get("null_bal") or [], dtype=float)
    overall["perm_z"] = (float((overall["balanced_accuracy"] - _nb.mean()) / _nb.std())
                         if _nb.size and _nb.std() > 0 else None)
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
            # Full band x time coefficient maps (which time/freq cells lead each class)
            _spec = VARIANT_SPEC.get(variant, {})
            if _spec.get("bands") == "full":
                _vt = int(_spec["n_time"]); _cell = len(FREQ_BANDS) * _vt
                _nfeat = imp["per_feature"].shape[0]
                if _cell and _nfeat % _cell == 0:
                    _ncond = _nfeat // _cell
                    plot_coef_ersp_maps(
                        imp, classes, run_dir / "coef_ersp_maps.png",
                        n_time=_vt, n_cond=_ncond,
                        cond_names=(profile or {}).get("cond_names")
                                   or (list(CONDITIONS) if _ncond > 1 else None),
                        title=title)
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
    # Newest first, but SKIP incomplete runs: an interrupted/crashed experiment
    # leaves an empty (or partial) run dir with no manifest.json. Returning it
    # would make load_run() raise FileNotFoundError, so walk back to the newest
    # run that actually finished.
    for d in sorted((d for d in base.iterdir() if d.is_dir()), reverse=True):
        if (d / "manifest.json").exists():
            return d
    return None


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
    if rd is None or not rd.exists() or not (rd / "manifest.json").exists():
        return None                          # missing/incomplete run -> caller skips it
    out = {"run_dir": rd,
           "manifest": json.loads((rd / "manifest.json").read_text()),
           "metrics": json.loads((rd / "metrics.json").read_text()),
           "per_class": pd.read_csv(rd / "per_class_metrics.csv"),
           "confusion": pd.read_csv(rd / "confusion_matrix.csv", index_col=0),
           "figures": {p.stem: p for p in rd.glob("*.png")}}
    return out


# ============================================================
# 13 — Cross-run comparison figures (compare features x tasks)
# ============================================================
# Every comparison below is CHANCE-NORMALIZED: chance differs by task (0.33 /
# 0.14 / 0.06), so raw balanced accuracy is never shared across tasks on one
# axis. We plot above_chance = (BA - chance) / (1 - chance), or one panel per
# task each with its own chance line.
VARIANT_DISPLAY_ORDER = ("full_300", "full_30", "full_300_rn", "full_30_rn",
                         "m101_300", "m101_30", "hg_300", "hg_30")
TASK_ORDER = ("condition", "parcellation_yeo7", "parcellation_yeo17")
TASK_SHORT = {"condition": "Condition", "parcellation_yeo7": "Yeo-7",
              "parcellation_yeo17": "Yeo-17"}
FAMILY_COLOR = {"full": "#4363d8", "rn": "#8e44ad", "m101": "#27ae60", "hg": "#16a085"}
# Task colour + classifier marker — used by the condensed z-score figures so the
# three tasks can share ONE axis (z is chance-normalized, balanced accuracy isn't).
TASK_COLOR = {"condition": "#d1495b", "parcellation_yeo7": "#2e86ab",
              "parcellation_yeo17": "#e08e0b"}
CLF_MARKER = {"logreg": "o", "rf": "s"}


def _variant_family(v: str) -> str:
    if v.endswith("_rn"):     return "rn"
    if v.startswith("m101"):  return "m101"
    if v.startswith("hg"):    return "hg"
    return "full"


def _pretty():
    """Seaborn aesthetics when it's installed, otherwise matplotlib's bundled
    seaborn stylesheet — so the figures look the same with or without seaborn as
    a hard dependency. Returns (style_for_plt_style_context, seaborn_or_None)."""
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        return sns.axes_style("whitegrid"), sns          # a dict of rcParams
    except Exception:
        name = next((s for s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid")
                     if s in plt.style.available), "default")
        return name, None


def _above_chance(ba, ch):
    try:
        return (float(ba) - float(ch)) / (1.0 - float(ch))
    except Exception:
        return float("nan")


def _read_null_bal(run_dir: Path) -> np.ndarray:
    """The permutation null distribution of balanced accuracy for a run."""
    p = Path(run_dir) / "permutation_null.json"
    if not p.exists():
        return np.array([])
    try:
        return np.asarray(json.loads(p.read_text()).get("null_bal", []), float)
    except Exception:
        return np.array([])


def _zscore_vs_null(value, null_bal) -> float:
    """How many sigma `value` sits above the permutation null. CHANCE-NORMALIZED
    by construction (null mean ~ chance, null spread = sampling noise at chance),
    so z is directly comparable across tasks with different chance levels."""
    null = np.asarray(null_bal, float)
    if null.size < 2:
        return float("nan")
    sd = float(null.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return (float(value) - float(null.mean())) / sd


_Z975 = 1.959964   # two-sided 95% normal quantile


def compare_table(outputs_root: Path = OUTPUTS_ROOT) -> pd.DataFrame:
    """One tidy row per (task, variant, classifier) — latest run each — with
    balanced accuracy, chance, permutation p, bootstrap CI, and comparable
    effect sizes that work across tasks with different chance levels:

      above_chance (+ ac_lo / ac_hi) : (BA − chance)/(1 − chance), bounded 0–1.
                The axis for ALL comparison figures — comparable across tasks AND
                it carries a real, varying bootstrap 95% CI (ac_lo..ac_hi), so the
                forest/paired panels can show honest error bars. (A z-score can't:
                its CI is ±1.96 by construction.)
      z_score : Wald z vs chance = (BA − chance) / SE, SE from the bootstrap CI.
                "Standard errors above chance" — kept as a table column.
      z_perm  : (BA − null_mean) / null_std vs the permutation null. Reference
                only — DEGENERATE for random-forest on the imbalanced parcellation
                tasks (permuted-label null collapses → null_std ≈ 0 → z in the
                hundreds), which is why it does not drive any figure."""
    runs = list_runs(outputs_root)
    if not len(runs):
        return pd.DataFrame()
    runs = (runs.sort_values("created_at")
            .drop_duplicates(["task", "variant", "classifier"], keep="last"))
    rows = []
    for r in runs.itertuples(index=False):
        ci = [float("nan"), float("nan")]
        null_bal = np.array([])
        rd = load_run(r.task, r.variant, r.classifier, outputs_root)
        if rd:
            ci = rd["metrics"]["overall"].get("balanced_accuracy_ci", ci) or ci
            null_bal = _read_null_bal(rd["run_dir"])
        p = getattr(r, "permutation_p", None)
        ci_lo = float(ci[0]) if ci[0] is not None else float("nan")
        ci_hi = float(ci[1]) if ci[1] is not None else float("nan")
        ba = float(r.balanced_accuracy); chance = float(r.chance_level)
        se = (ci_hi - ci_lo) / (2 * _Z975) if np.isfinite(ci_lo) and np.isfinite(ci_hi) else float("nan")
        z_wald = (ba - chance) / se if np.isfinite(se) and se > 0 else float("nan")
        rows.append({
            "task": r.task, "variant": r.variant, "classifier": r.classifier,
            "balanced_accuracy": ba, "chance_level": chance,
            "permutation_p": (None if p is None else float(p)),
            "macro_f1": float(getattr(r, "macro_f1", float("nan"))),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "above_chance": _above_chance(ba, chance),
            "ac_lo": _above_chance(ci_lo, chance), "ac_hi": _above_chance(ci_hi, chance),
            "boot_se": se,
            "z_score": z_wald,                                  # PRIMARY (Wald vs chance)
            "z_perm": _zscore_vs_null(ba, null_bal),            # reference (vs perm null)
            "null_mean": float(null_bal.mean()) if null_bal.size else float("nan"),
            "null_std": float(null_bal.std(ddof=1)) if null_bal.size > 1 else float("nan"),
            "sig": (p is not None and not pd.isna(p) and float(p) < 0.05),
        })
    return pd.DataFrame(rows)


def _ordered(comp, attr, order):
    present = set(comp[attr])
    return [x for x in order if x in present]


def plot_compare_heatmap(comp, out_png=None):
    """FIG 1 — variant x (task x classifier) heatmap. Cell colour = fraction
    above chance; annotation = balanced accuracy (+ '*' if p<.05)."""
    import matplotlib.pyplot as plt
    if not len(comp):
        return None
    variants = _ordered(comp, "variant", VARIANT_DISPLAY_ORDER)
    tasks = _ordered(comp, "task", TASK_ORDER)
    clfs = sorted(comp["classifier"].unique())
    cols = [(t, c) for t in tasks for c in clfs]
    M = np.full((len(variants), len(cols)), np.nan)
    ann = [["" for _ in cols] for _ in variants]
    for i, v in enumerate(variants):
        for j, (t, c) in enumerate(cols):
            sub = comp[(comp.variant == v) & (comp.task == t) & (comp.classifier == c)]
            if len(sub):
                s = sub.iloc[0]
                M[i, j] = max(0.0, s.above_chance) if np.isfinite(s.above_chance) else np.nan
                ann[i][j] = f"{s.balanced_accuracy:.2f}" + ("*" if s.sig else "")
    vmax = np.nanmax(M) if np.isfinite(np.nanmax(M)) else 0.3
    vmax = max(0.3, vmax)
    fig, ax = plt.subplots(figsize=(1.5 + 1.05 * len(cols), 1.2 + 0.46 * len(variants)))
    im = ax.imshow(M, cmap="YlGn", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{TASK_SHORT.get(t, t)}\n{c}" for t, c in cols], fontsize=7)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=8)
    for i in range(len(variants)):
        for j in range(len(cols)):
            if ann[i][j]:
                light = np.isfinite(M[i, j]) and M[i, j] > 0.55 * vmax
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=7,
                        color="white" if light else "#3a2e2a")
    for k in range(1, len(tasks)):
        ax.axvline(k * len(clfs) - 0.5, color="white", lw=2.5)
    ax.set_title("Above-chance decoding — variant × task   (cell = balanced acc · * p<.05)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="fraction above chance")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig


def plot_compare_forest(comp, out_png=None):
    """FIG 2 — ONE condensed panel; all three tasks on a shared axis.

    x = fraction above chance = (BA − chance)/(1 − chance), bounded 0–1 and
    comparable across tasks (chance 0.33 / 0.14 / 0.06) — what raw balanced
    accuracy could never share. Error bar = bootstrap 95% CI.
      y = variant band · colour = task · marker = classifier (o logreg / s rf)
      · filled = p<.05, hollow = ns.  Vertical line at 0 = chance."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not len(comp):
        return None
    style, sns = _pretty()
    tasks = _ordered(comp, "task", TASK_ORDER)
    clfs = _ordered(comp, "classifier", tuple(CLF_MARKER)) or sorted(comp.classifier.unique())
    variants = _ordered(comp, "variant", VARIANT_DISPLAY_ORDER)
    n_t = len(tasks)
    band = n_t + 1.2                                   # vertical room per variant
    t_off = {t: (i - (n_t - 1) / 2) * 0.95 for i, t in enumerate(tasks)}
    c_nudge = {c: (i - (len(clfs) - 1) / 2) * 0.26 for i, c in enumerate(clfs)}

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(8.4, 0.62 * band * len(variants) / n_t + 1.4))
        for vi, v in enumerate(variants):
            base = vi * band
            for t in tasks:
                col = TASK_COLOR.get(t, "#555")
                for c in clfs:
                    sub = comp[(comp.variant == v) & (comp.task == t) & (comp.classifier == c)]
                    if not len(sub) or not np.isfinite(sub.iloc[0].above_chance):
                        continue
                    s = sub.iloc[0]
                    y = base + t_off[t] + c_nudge[c]
                    lo = s.ac_lo if np.isfinite(s.ac_lo) else s.above_chance
                    hi = s.ac_hi if np.isfinite(s.ac_hi) else s.above_chance
                    xerr = [[max(0, s.above_chance - lo)], [max(0, hi - s.above_chance)]]
                    ax.errorbar(s.above_chance, y, xerr=xerr, fmt=CLF_MARKER.get(c, "o"),
                                color=col, mfc=col if s.sig else "white", mec=col,
                                ms=6, capsize=2, elinewidth=1, zorder=3)
        ax.axvline(0, color="#888", lw=1.2)            # chance
        ax.set_yticks([vi * band for vi in range(len(variants))])
        ax.set_yticklabels(variants)
        ax.set_ylim(-band / 2, (len(variants) - 1) * band + band / 2)
        ax.invert_yaxis()
        ax.set_xlabel("fraction above chance   (BA − chance)/(1 − chance),  95% CI")
        ax.margins(x=0.04)
        task_h = [Line2D([0], [0], marker="o", ls="", mfc=TASK_COLOR[t], mec=TASK_COLOR[t],
                         ms=7, label=TASK_SHORT.get(t, t)) for t in tasks]
        clf_h = [Line2D([0], [0], marker=CLF_MARKER.get(c, "o"), ls="", color="#555",
                        ms=7, label=c) for c in clfs]
        sig_h = [Line2D([0], [0], marker="o", ls="", color="#555", mfc="#555", ms=7, label="p<.05"),
                 Line2D([0], [0], marker="o", ls="", color="#555", mfc="white", ms=7, label="ns")]
        ax.legend(handles=task_h + clf_h + sig_h, fontsize=7, ncol=3,
                  loc="lower right", framealpha=0.9)
        ax.set_title("Decoding strength — fraction above chance (95% CI), all tasks on one axis",
                     fontsize=10)
        if sns:
            sns.despine(fig=fig, left=True)
        fig.tight_layout()
        if out_png:
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig


def plot_paired_contrasts(comp, classifier="logreg", out_png=None):
    """FIG 3 — paired contrasts, two panels sharing one axis (task = colour).
      Left  — amplitude triad (continuous → row-norm → discretized): a line per
              task×grid (solid 300, dashed 30). A drop left→right = the decode
              was riding on power, not pattern.
      Right — time resolution: 300 (filled) vs 30 (hollow) for the full & HG
              families, one paired marker per task. Read the direction.
    y = fraction above chance with bootstrap 95% CI — comparable across tasks."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not len(comp):
        return None
    style, sns = _pretty()
    tasks = _ordered(comp, "task", TASK_ORDER)
    d = comp[comp.classifier == classifier]

    def ac(v, t):
        """(point, err_low, err_high) of fraction-above-chance, NaNs if missing."""
        sub = d[(d.variant == v) & (d.task == t)]
        if not len(sub):
            return float("nan"), 0.0, 0.0
        s = sub.iloc[0]
        m = s.above_chance
        lo = m - s.ac_lo if np.isfinite(s.ac_lo) else 0.0
        hi = s.ac_hi - m if np.isfinite(s.ac_hi) else 0.0
        return m, max(0.0, lo), max(0.0, hi)

    triad = {300: ["full_300", "full_300_rn", "m101_300"],
             30:  ["full_30", "full_30_rn", "m101_30"]}
    with plt.style.context(style):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

        for t in tasks:
            col = TASK_COLOR.get(t, "#555")
            for grid, vs in triad.items():
                pts = [ac(v, t) for v in vs]
                axA.errorbar([0, 1, 2], [p[0] for p in pts],
                             yerr=[[p[1] for p in pts], [p[2] for p in pts]],
                             ls="-" if grid == 300 else "--", marker="o", color=col,
                             ms=5, lw=1.4, capsize=2, elinewidth=0.9)
        axA.set_xticks([0, 1, 2])
        axA.set_xticklabels(["continuous", "row-norm", "discretized"], fontsize=8)
        axA.axhline(0, color="#888", lw=1.0)
        axA.set_title("Amplitude triad  (solid = 300-time · dashed = 30-time)", fontsize=10)
        axA.set_ylabel("fraction above chance   (95% CI)")

        fams = [("full", ("full_300", "full_30")), ("hg", ("hg_300", "hg_30"))]
        n_t = len(tasks)
        for xi, (fam, (v3, v30)) in enumerate(fams):
            for ti, t in enumerate(tasks):
                col = TASK_COLOR.get(t, "#555")
                x = xi + (ti - (n_t - 1) / 2) * 0.18
                m3, l3, h3 = ac(v3, t); m30, l30, h30 = ac(v30, t)
                axB.plot([x, x], [m3, m30], color=col, lw=1.4, zorder=1)
                axB.errorbar(x, m3, yerr=[[l3], [h3]], fmt="o", color=col,
                             ms=7, capsize=2, elinewidth=0.9)               # 300 filled
                axB.errorbar(x, m30, yerr=[[l30], [h30]], fmt="o", color=col,
                             mfc="white", mec=col, ms=7, capsize=2, elinewidth=0.9)  # 30 hollow
        axB.set_xticks([0, 1]); axB.set_xticklabels(["full", "hg"], fontsize=8)
        axB.set_xlim(-0.5, 1.5); axB.axhline(0, color="#888", lw=1.0)
        axB.set_title("Time resolution  (●300  ○30)", fontsize=10)

        task_h = [Line2D([0], [0], marker="o", ls="", mfc=TASK_COLOR[t], mec=TASK_COLOR[t],
                         ms=7, label=TASK_SHORT.get(t, t)) for t in tasks]
        axA.legend(handles=task_h, fontsize=7, loc="upper right")
        fig.suptitle(f"Paired contrasts ({classifier}) — amplitude (left) · time resolution (right)",
                     fontsize=11)
        if sns:
            sns.despine(fig=fig)
        fig.tight_layout()
        if out_png:
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig


def plot_compare_clf_scatter(comp, out_png=None):
    """FIG 4 — classifier-agreement robustness check. Each point is one
    (task, variant): x = above-chance(logreg), y = above-chance(rf). Points near
    the diagonal mean the two classifiers reach the same decoding strength; the
    off-diagonal ones (labelled) flag a feature whose decodability depends on the
    model. Same fraction-above-chance metric as Figs 2–3."""
    import matplotlib.pyplot as plt
    if not len(comp) or comp.classifier.nunique() < 2:
        return None
    style, sns = _pretty()
    wide = comp.pivot_table(index=["task", "variant"], columns="classifier",
                            values="above_chance").reset_index()
    if "logreg" not in wide or "rf" not in wide:
        return None
    wide = wide.dropna(subset=["logreg", "rf"])
    if not len(wide):
        return None
    lab_thr = 0.05          # label only points that disagree by >5 pts of above-chance
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(5.8, 5.6))
        lim = float(np.nanmax([wide.logreg.max(), wide.rf.max()])) * 1.10
        ax.plot([0, lim], [0, lim], ls="--", color="#aaa", lw=1.0, zorder=1)   # y=x
        for t in _ordered(comp, "task", TASK_ORDER):
            sub = wide[wide.task == t]
            ax.scatter(sub.logreg, sub.rf, color=TASK_COLOR.get(t, "#555"), s=46,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=TASK_SHORT.get(t, t))
            for _, row in sub.iterrows():
                if abs(row.logreg - row.rf) >= lab_thr:
                    ax.annotate(row.variant, (row.logreg, row.rf), fontsize=6,
                                xytext=(4, 2), textcoords="offset points", color="#333")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel("fraction above chance  (logreg)")
        ax.set_ylabel("fraction above chance  (rf)")
        ax.legend(fontsize=7, loc="lower right", title="task", title_fontsize=7)
        ax.set_title("Classifier agreement — logreg vs rf\n(near diagonal = robust to model choice)",
                     fontsize=9)
        if sns:
            sns.despine(fig=fig)
        fig.tight_layout()
        if out_png:
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig


# ----- confusion-matrix comparison (per task — class sets differ across tasks) -----
def _short_class(x) -> str:
    """'7Networks_3' -> '3', '17Networks_12' -> '12'; condition names unchanged."""
    s = str(x)
    return s.split("Networks_")[-1] if "Networks_" in s else s


def _class_order(labels):
    """Canonical class order: numeric for the Yeo networks (1,2,…,17 — not the
    lexicographic 1,10,11,…,2), original order otherwise (condition names)."""
    short = [_short_class(x) for x in labels]
    if all(s.isdigit() for s in short):
        return [x for _, x in sorted(zip([int(s) for s in short], labels))]
    return list(labels)


_CLF_ABBR = {"logreg": "LR", "rf": "RF"}


def _task_runs(task, outputs_root: Path = OUTPUTS_ROOT):
    """Latest run of every variant×classifier under one task, display order."""
    for v in VARIANT_DISPLAY_ORDER:
        for c in CLF_MARKER:
            rd = load_run(task, v, c, outputs_root)
            if rd is not None:
                yield v, c, rd


def plot_confusion_consensus(out_png=None, outputs_root: Path = OUTPUTS_ROOT):
    """FIG 5 — combined confusion. One matrix per task: the MEAN row-normalized
    confusion across every variant×classifier run. Diagonal = mean recall;
    off-diagonal = the confusions that are stable across all features (which
    class gets mistaken for which). Confusion matrices are only comparable WITHIN
    a task — the three tasks have different class sets, so they get their own
    matrix rather than being merged."""
    import matplotlib.pyplot as plt
    style, sns = _pretty()
    mats = {}
    for t in TASK_ORDER:
        acc, n, order = None, 0, None
        for v, c, rd in _task_runs(t, outputs_root):
            cm = rd["confusion"]
            if order is None:
                order = _class_order(list(cm.columns))
            cm = cm.reindex(index=order, columns=order)   # canonical (numeric) order
            M = cm.values.astype(float)
            rs = M.sum(1, keepdims=True); rs[rs == 0] = 1.0
            M = M / rs                                    # row-normalize -> rates
            acc = M if acc is None else acc + M
            n += 1
        if acc is not None and n:
            mats[t] = (acc / n, [_short_class(x) for x in order], n)
    tasks = [t for t in TASK_ORDER if t in mats]
    if not tasks:
        return None
    ncs = [mats[t][0].shape[0] for t in tasks]
    with plt.style.context(style):
        fig, axes = plt.subplots(1, len(tasks), squeeze=False,
                                 figsize=(sum(ncs) * 0.42 + 3.0, max(ncs) * 0.40 + 1.8),
                                 gridspec_kw={"width_ratios": ncs})
        axes = axes[0]
        im = None
        for ax, t in zip(axes, tasks):
            M, classes, n = mats[t]
            im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=6)
            ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=6)
            ax.set_title(f"{TASK_SHORT.get(t, t)}  (mean of {n} runs)", fontsize=9)
            ax.set_xlabel("predicted", fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("true", fontsize=8)
            if M.shape[0] <= 7:                          # annotate only the small ones
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if M[i, j] < 0.5 else "#222")
        fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02,
                     label="mean row-normalized rate")
        fig.suptitle("Combined confusion — mean row-normalized confusion per task "
                     "(diag = recall · off-diag = stable confusions)", fontsize=10)
        if out_png:
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig


def plot_class_recall_compare(out_png=None, outputs_root: Path = OUTPUTS_ROOT):
    """FIG 6 — compare the confusion DIAGONALS across all runs. One stacked
    heatmap per task: rows = class, cols = variant·classifier, cell = per-class
    recall, `*` = permutation-FDR p<.05. Reads which classes a feature decodes
    and which stay chronically confused (a row that's pale everywhere)."""
    import matplotlib.pyplot as plt
    style, sns = _pretty()
    data = {}                                            # task -> (recall df, sig df, cols)
    for t in TASK_ORDER:
        cols, rec, sig, classes = [], {}, {}, None
        for v, c, rd in _task_runs(t, outputs_root):
            key = f"{v}·{_CLF_ABBR.get(c, c)}"
            cols.append(key)
            pc = rd["per_class"].set_index("class")
            if classes is None:
                classes = _class_order(list(pc.index))   # canonical (numeric) order
            rec[key] = pc.reindex(classes)["recall"].values
            sig[key] = (pc.reindex(classes)["perm_p_fdr"].values < 0.05)
        if cols:
            data[t] = (pd.DataFrame(rec, index=[_short_class(x) for x in classes]),
                       pd.DataFrame(sig, index=[_short_class(x) for x in classes]), cols)
    tasks = [t for t in TASK_ORDER if t in data]
    if not tasks:
        return None
    nrows = [data[t][0].shape[0] for t in tasks]
    ncols = max(data[t][0].shape[1] for t in tasks)
    with plt.style.context(style):
        fig, axes = plt.subplots(len(tasks), 1, squeeze=False,
                                 figsize=(ncols * 0.42 + 2.5, sum(nrows) * 0.30 + 2.2),
                                 gridspec_kw={"height_ratios": nrows})
        axes = axes[:, 0]
        im = None
        for ax, t in zip(axes, tasks):
            rec, sig, cols = data[t]
            im = ax.imshow(rec.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_yticks(range(rec.shape[0])); ax.set_yticklabels(rec.index, fontsize=6)
            ax.set_xticks(range(rec.shape[1]))
            ax.set_xticklabels(cols if ax is axes[-1] else [], rotation=90, fontsize=6)
            ax.set_title(TASK_SHORT.get(t, t), fontsize=9, loc="left")
            for i in range(rec.shape[0]):
                for j in range(rec.shape[1]):
                    if sig.values[i, j]:
                        ax.text(j, i, "*", ha="center", va="center", fontsize=8,
                                color="white" if rec.values[i, j] > 0.5 else "#222")
        fig.colorbar(im, ax=list(axes), fraction=0.015, pad=0.02, label="per-class recall")
        fig.suptitle("Per-class recall — confusion diagonals across variants×classifiers  "
                     "(* = FDR p<.05)", fontsize=10)
        if out_png:
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig


# ----- HG vs FULL: does single-band high-gamma match the full spectrum? -----
_BAND_GRIDS = ((300, False), (30, False), (300, True), (30, True))   # n_time, rownorm


def _band_variant(band: str, n_time: int, rownorm: bool) -> str:
    return f"{band}_{n_time}" + ("_rn" if rownorm else "")


def hg_vs_full_stats(comp, metric: str = "above_chance") -> pd.DataFrame:
    """Per-task test of whether HIGH-GAMMA (single 70–150 Hz band) decodes as well
    as the FULL 15-band spectrum. The variants are matched pairs — same time grid,
    rownorm, and classifier — so this is a PAIRED comparison (the only fair one;
    HG and FULL share a time grid by design). Reports, per task:
      n_pairs, group means, median paired Δ (HG−FULL), how many pairs favour HG,
      a sign-test and Wilcoxon signed-rank p, and the single best HG vs best FULL
      run (with whether their 95% CIs overlap) — which directly answers
      'is HG of any type better than FULL of any type?'."""
    from scipy import stats
    clfs = _ordered(comp, "classifier", tuple(CLF_MARKER)) or sorted(comp["classifier"].unique())
    out = []
    for t in TASK_ORDER:
        d = comp[comp.task == t]
        if not len(d):
            continue
        hg_v, full_v = [], []
        for c in clfs:
            for nt, rn in _BAND_GRIDS:
                hr = d[(d.variant == _band_variant("hg", nt, rn)) & (d.classifier == c)]
                fr = d[(d.variant == _band_variant("full", nt, rn)) & (d.classifier == c)]
                if len(hr) and len(fr):
                    hg_v.append(float(hr.iloc[0][metric]))
                    full_v.append(float(fr.iloc[0][metric]))
        if not hg_v:
            continue
        hg_v, full_v = np.array(hg_v), np.array(full_v)
        delta = hg_v - full_v
        n, k = len(delta), int((delta > 0).sum())
        sign_p = float(stats.binomtest(k, n, 0.5).pvalue)
        try:
            wil_p = float(stats.wilcoxon(hg_v, full_v).pvalue)
        except Exception:
            wil_p = float("nan")
        hg_all = d[d.variant.str.startswith("hg")]
        full_all = d[d.variant.str.startswith("full")]
        bh = hg_all.loc[hg_all[metric].idxmax()]
        bf = full_all.loc[full_all[metric].idxmax()]
        overlap = bool((bh.ac_lo <= bf.ac_hi) and (bf.ac_lo <= bh.ac_hi))
        out.append({
            "task": TASK_SHORT.get(t, t), "n_pairs": n,
            "hg_mean": round(float(hg_v.mean()), 3),
            "full_mean": round(float(full_v.mean()), 3),
            "median_d(hg-full)": round(float(np.median(delta)), 3),
            "hg_better": f"{k}/{n}",
            "sign_p": round(sign_p, 3),
            "wilcoxon_p": (round(wil_p, 3) if wil_p == wil_p else None),
            "best_hg": f"{bh.variant}·{bh.classifier}={bh[metric]:.3f}",
            "best_full": f"{bf.variant}·{bf.classifier}={bf[metric]:.3f}",
            "best_d": round(float(bh[metric] - bf[metric]), 3),
            "best_CIs_overlap": overlap,
        })
    return pd.DataFrame(out)


def plot_hg_vs_full(comp, metric: str = "above_chance", out_png=None):
    """FIG 7 — HG vs FULL as matched-pair slopegraphs, one panel per task. Each
    line connects a FULL variant to its HG twin (same grid×rownorm×classifier);
    green rising = HG higher, red falling = FULL higher. Shared y = fraction above
    chance, so panels are comparable. Title shows how many pairs favour HG."""
    import matplotlib.pyplot as plt
    style, sns = _pretty()
    clfs = _ordered(comp, "classifier", tuple(CLF_MARKER)) or sorted(comp["classifier"].unique())
    tasks = _ordered(comp, "task", TASK_ORDER)
    with plt.style.context(style):
        fig, axes = plt.subplots(1, len(tasks), figsize=(3.3 * len(tasks), 4.4),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, t in zip(axes, tasks):
            d = comp[comp.task == t]
            nb = up = 0
            for c in clfs:
                for nt, rn in _BAND_GRIDS:
                    hr = d[(d.variant == _band_variant("hg", nt, rn)) & (d.classifier == c)]
                    fr = d[(d.variant == _band_variant("full", nt, rn)) & (d.classifier == c)]
                    if not (len(hr) and len(fr)):
                        continue
                    h, f = float(hr.iloc[0][metric]), float(fr.iloc[0][metric])
                    nb += 1; up += int(h > f)
                    ax.plot([0, 1], [f, h], "-o", ms=4, lw=1.1, alpha=0.85,
                            color="#2e7d32" if h > f else "#c0392b")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["FULL", "HG"])
            ax.set_xlim(-0.3, 1.3)
            ax.set_title(f"{TASK_SHORT.get(t, t)}\nHG > FULL in {up}/{nb} pairs", fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel("fraction above chance")
        fig.suptitle("HG vs FULL — matched pairs (grid × rownorm × classifier) · "
                     "green = HG higher", fontsize=10)
        if sns:
            sns.despine(fig=fig)
        fig.tight_layout()
        if out_png:
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig


# ============================================================
# 14 — Paired feature comparison (patient-grouped, from predictions.csv)
# ============================================================
# Every feature is scored on the SAME electrodes with the SAME GroupKFold, so
# predictions are aligned electrode-by-electrode and per-patient balanced
# accuracies are PAIRED across features. That lets us test "is feature A better
# than B" honestly (Wilcoxon over patients), and diff their confusion matrices.
def per_patient_balanced_acc(task, variant, classifier,
                             outputs_root: Path = OUTPUTS_ROOT) -> dict:
    """{patient_id: balanced accuracy} from a run's out-of-fold predictions.csv.
    Each patient's rows are the fold in which that patient was held out, so this
    is a per-patient generalisation estimate."""
    from sklearn.metrics import balanced_accuracy_score
    rd = latest_run_dir(task, variant, classifier, outputs_root)
    if rd is None or not (rd / "predictions.csv").exists():
        return {}
    df = pd.read_csv(rd / "predictions.csv")
    out = {}
    for pid, g in df.groupby("patient_id"):
        try:
            out[str(pid)] = float(balanced_accuracy_score(g["y_true"], g["y_pred"]))
        except Exception:
            pass
    return out


def paired_feature_wilcoxon(task, classifier, baseline="full_300", variants=None,
                            outputs_root: Path = OUTPUTS_ROOT) -> pd.DataFrame:
    """Per-patient paired Wilcoxon signed-rank of each feature vs `baseline`,
    respecting the patient-grouped design (the honest 'is A better than B').

    Columns: variant, n_patients, median_delta (feature − baseline balanced acc),
    wilcoxon_p. NOTE: ~N_patients paired samples → limited power; per-patient
    balanced accuracy is noisy for patients with few electrodes. Screen, not proof."""
    from scipy.stats import wilcoxon
    if variants is None:
        variants = [v for v in VARIANT_DISPLAY_ORDER
                    if latest_run_dir(task, v, classifier, outputs_root) is not None]
    base = per_patient_balanced_acc(task, baseline, classifier, outputs_root)
    rows = []
    for v in variants:
        if v == baseline:
            continue
        cur = per_patient_balanced_acc(task, v, classifier, outputs_root)
        pats = sorted(set(base) & set(cur))
        d = np.array([cur[p] - base[p] for p in pats], dtype=float)
        pval = float("nan")
        if len(pats) >= 3 and np.any(d != 0):
            try:
                pval = float(wilcoxon(d).pvalue)
            except Exception:
                pval = float("nan")
        rows.append({"variant": v, "n_patients": len(pats),
                     "median_delta": float(np.median(d)) if len(d) else float("nan"),
                     "wilcoxon_p": pval})
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "vs_baseline", baseline)
    return out


def plot_delta_confusion(task, classifier, baseline="full_300", variants=None,
                         outputs_root: Path = OUTPUTS_ROOT, out_png=None):
    """Row-normalised confusion(feature) − confusion(baseline), one diverging
    panel per feature. Red = the feature sends MORE of a true class to that
    predicted class than the baseline does (blue = less). Isolates the change."""
    import matplotlib.pyplot as plt

    def rn_conf(v):
        rd = latest_run_dir(task, v, classifier, outputs_root)
        if rd is None or not (rd / "confusion_matrix.csv").exists():
            return None, None
        df = pd.read_csv(rd / "confusion_matrix.csv", index_col=0)
        cm = df.to_numpy(dtype=float)
        rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        return list(df.index), cm / rs

    labels, base = rn_conf(baseline)
    if base is None:
        return None
    if variants is None:
        variants = [v for v in VARIANT_DISPLAY_ORDER
                    if v != baseline and latest_run_dir(task, v, classifier, outputs_root)]
    diffs = []
    for v in variants:
        _, cm = rn_conf(v)
        if cm is not None and cm.shape == base.shape:
            diffs.append((v, cm - base))
    if not diffs:
        return None
    vmax = max(float(np.nanmax(np.abs(d))) for _, d in diffs) or 1.0
    short = [str(l).replace("Networks_", "N") for l in labels]
    n = len(diffs); ncols = min(4, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows),
                             squeeze=False)
    im = None
    for k, (v, d) in enumerate(diffs):
        ax = axes[k // ncols][k % ncols]
        im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{v} − {baseline}", fontsize=8)
        ax.set_xticks(range(len(short))); ax.set_xticklabels(short, fontsize=5, rotation=90)
        ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=5)
        ax.set_xlabel("predicted", fontsize=6); ax.set_ylabel("true", fontsize=6)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Δ confusion vs {baseline} — {TASK_SHORT.get(task, task)} · {classifier}"
                 "   (red = feature sends more mass here)", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                     label="Δ row-fraction")
    if out_png:
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
    return fig
