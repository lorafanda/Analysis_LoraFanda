"""
lf_dataset.py — Canonical dataset preparation for the clustering notebooks.

THE ONE place that produces the (df_meta, ersp_list, X_3d) used by 210, 230,
231 and 232. Every clustering notebook starts with the same call:

    from functions.lf_dataset import prepare_dataset
    df_meta, ersp_list, X_3d = prepare_dataset(INPUT_DIR)

Guarantees that cross-feature-set and cross-method comparisons (raw vs
blob vs minus101 vs hg) operate on the IDENTICAL sample set. Previously
each notebook applied its own filter (210 used high-activity, 230 added
blob-score gating on top, 232 used no filter at all) and the sample
counts diverged across runs (1538 vs 727 vs 7268) — making any cross-
comparison invalid.

Filter order, identical for every caller:
    1. Walk INPUT_DIR/<patient>/<task>/ERSP_matrix/<condition>/*.npy
    2. Drop non-neural channels (name pattern: PHOTO, MRK, MKR, ECG, AUDIO,
       contains '+' or '-', X / X1..X8 / E1..E8)
    3. Compute per-sample (prop>thr_pos, prop<thr_neg)
    4. Gate: keep iff (prop>thr_pos >= min_prop_pos) OR
                       (prop<thr_neg >= min_prop_neg)

Caching:
    If `cache_dir` is given, the (df_meta, X_3d) pair is stored there with
    a small params.json. Subsequent calls with matching params load
    instantly from cache (~hundreds of MB → milliseconds). Cache misses
    rebuild + overwrite. Defaults: no caching (call always rebuilds fresh).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Defaults — single source of truth for the canonical filter
# ============================================================
DEFAULT_TASK             = "LM"
DEFAULT_CONDITIONS       = ("audio", "picture", "reading")
DEFAULT_N_FREQ           = 129
DEFAULT_N_TIME           = 300
DEFAULT_THR_POS          = 2.2
DEFAULT_MIN_PROP_POS     = 0.02
DEFAULT_THR_NEG          = -3.0
DEFAULT_MIN_PROP_NEG     = 0.04


# ============================================================
# Filename → electrode helpers
# ============================================================
def parse_electrode_from_filename(fname: str) -> str:
    """Pull the electrode name out of an ERSP .npy filename.

    Examples:
        PAT_3301_picture_None_ERSP_AG2_TN.npy       -> 'AG2'
        EL035_reading_WM_ERSP_A_R10_TN.npy          -> 'A_R10'
        EL030_audio_WM_ERSP_aH_L1_TN.npy            -> 'aH_L1'
    """
    name = Path(fname).name
    if "_ERSP_" in name:
        right = name.split("_ERSP_", 1)[1]
        right_no_ext = right.rsplit(".", 1)[0]
        # Strip trailing "_TN" or similar tags
        parts = right_no_ext.split("_")
        # Heuristic: drop trailing TN/CLEAN tokens
        while parts and parts[-1].upper() in ("TN", "CLEAN", "TF"):
            parts.pop()
        return "_".join(parts) if parts else right_no_ext
    return name.rsplit(".", 1)[0]


def is_non_neural_electrode(label: str) -> bool:
    """Match non-neural channel name patterns: PHOTO, MRK, ECG, AUDIO, AINP
    (Blackrock analog input — photodiode / mic / trigger), EKG/EMG,
    X / X1..X8 / E1..E8, or anything with '+' or '-' in the label."""
    if label is None:
        return False
    s = str(label).strip().upper()
    if "+" in s or "-" in s:
        return True
    # AINP1/2/3... are the Blackrock analog inputs (photodiode, microphone, trigger).
    # They were slipping through and being clustered as if they were brain channels.
    if re.fullmatch(r"AINP\d*", s):
        return True
    if any(tag in s for tag in ("PHOTO", "MRK", "MKR", "ECG", "EKG", "EMG", "AUDIO")):
        return True
    if s == "X":
        return True
    if re.fullmatch(r"[XE][1-8]", s):
        return True
    return False


# ============================================================
# Public: prepare_dataset
# ============================================================
def prepare_dataset(
    input_dir,
    *,
    task: str = DEFAULT_TASK,
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
    n_freq: int = DEFAULT_N_FREQ,
    n_time: int = DEFAULT_N_TIME,
    thr_pos: float = DEFAULT_THR_POS,
    min_prop_pos: float = DEFAULT_MIN_PROP_POS,
    thr_neg: float = DEFAULT_THR_NEG,
    min_prop_neg: float = DEFAULT_MIN_PROP_NEG,
    apply_high_activity: bool = True,
    cache_dir = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[np.ndarray], np.ndarray]:
    """
    Build the canonical clustering dataset.

    Returns
    -------
    df_meta : DataFrame, columns
        sample_idx, patient_id, condition, task, electrode, file_path,
        prop_above_pos, prop_below_neg, high_activity
    ersp_list : list of (n_freq, n_time) ndarrays
    X_3d : ndarray of shape (n_samples, n_freq, n_time) — np.stack(ersp_list)

    Cache layout (when `cache_dir` is provided):
        <cache_dir>/params.json
        <cache_dir>/df_meta.parquet
        <cache_dir>/X_3d.npy
    """
    input_dir = Path(input_dir)
    params = {
        "input_dir": str(input_dir),
        "task": task,
        "conditions": list(conditions),
        "n_freq": int(n_freq),
        "n_time": int(n_time),
        "thr_pos": float(thr_pos),
        "min_prop_pos": float(min_prop_pos),
        "thr_neg": float(thr_neg),
        "min_prop_neg": float(min_prop_neg),
        "apply_high_activity": bool(apply_high_activity),
        # Version tag: bump if filter logic changes incompatibly so old caches invalidate
        "schema": 1,
    }

    # ---- Cache hit? ----
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        params_path = cache_dir / "params.json"
        meta_path   = cache_dir / "df_meta.parquet"
        x_path      = cache_dir / "X_3d.npy"
        if params_path.exists() and meta_path.exists() and x_path.exists():
            cached_params = json.loads(params_path.read_text())
            if cached_params == params:
                df_meta = pd.read_parquet(meta_path)
                X_3d    = np.load(x_path)
                ersp_list = [X_3d[i] for i in range(X_3d.shape[0])]
                if verbose:
                    print(f"[lf_dataset cache hit] {cache_dir}")
                    print(f"  {len(df_meta)} samples · X_3d.shape={X_3d.shape}")
                return df_meta, ersp_list, X_3d
            elif verbose:
                print(f"[lf_dataset cache miss] params differ — rebuilding")

    # ---- Build fresh ----
    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {input_dir}")

    patient_ids = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    if verbose:
        print(f"[lf_dataset] detected {len(patient_ids)} patient folders under {input_dir}")

    cond_set = set(conditions)
    ersp_list: List[np.ndarray] = []
    meta_rows = []

    for pat in patient_ids:
        patient_dir = input_dir / pat / task / "ERSP_matrix"
        if not patient_dir.exists():
            if verbose:
                print(f"  [WARN] missing ERSP_matrix for {pat}")
            continue
        condition_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name in cond_set]
        if not condition_dirs:
            if verbose:
                print(f"  [WARN] no allowed conditions in {patient_dir}")
            continue
        for cond_dir in condition_dirs:
            npy_files = sorted(cond_dir.glob("*.npy"))
            for fpath in npy_files:
                arr = np.load(fpath)
                if arr.shape != (n_freq, n_time):
                    if verbose:
                        print(f"  [WARN] {fpath.name} shape={arr.shape} != ({n_freq},{n_time}) — skipping")
                    continue
                ersp_list.append(arr)
                meta_rows.append({
                    "patient_id": pat,
                    "condition":  cond_dir.name,
                    "task":       task,
                    "electrode":  parse_electrode_from_filename(fpath.name),
                    "file_path":  str(fpath),
                })

    df_meta = pd.DataFrame(meta_rows)
    n_loaded = len(df_meta)
    if verbose:
        print(f"  loaded {n_loaded} samples")

    # ---- Non-neural filter ----
    non_neural_mask = df_meta["electrode"].apply(is_non_neural_electrode) if n_loaded else pd.Series(dtype=bool)
    n_non_neural = int(non_neural_mask.sum()) if n_loaded else 0
    if n_non_neural:
        keep = ~non_neural_mask
        df_meta = df_meta[keep].reset_index(drop=True)
        ersp_list = [ersp_list[i] for i in np.where(keep.to_numpy())[0]]
        if verbose:
            print(f"  excluded {n_non_neural} non-neural channels → {len(df_meta)} samples")

    # ---- High-activity computation (always computed, optionally filtered) ----
    prop_pos = []
    prop_neg = []
    for arr in ersp_list:
        prop_pos.append(float((arr > thr_pos).mean()))
        prop_neg.append(float((arr < thr_neg).mean()))
    df_meta["prop_above_pos"] = prop_pos
    df_meta["prop_below_neg"] = prop_neg
    df_meta["high_activity"] = [
        (p >= min_prop_pos) or (n >= min_prop_neg)
        for p, n in zip(prop_pos, prop_neg)
    ]

    if apply_high_activity:
        keep = df_meta["high_activity"].to_numpy()
        n_high = int(keep.sum())
        n_low = len(df_meta) - n_high
        df_meta = df_meta[keep].reset_index(drop=True)
        ersp_list = [ersp_list[i] for i in np.where(keep)[0]]
        if verbose:
            print(f"  high-activity gate (thr_pos>{thr_pos} prop≥{min_prop_pos}, "
                  f"thr_neg<{thr_neg} prop≥{min_prop_neg}): kept {n_high}, dropped {n_low}")

    # ---- Rebuild sample_idx + stack ----
    df_meta = df_meta.reset_index(drop=True)
    df_meta["sample_idx"] = np.arange(len(df_meta), dtype=np.int64)

    if ersp_list:
        X_3d = np.stack(ersp_list, axis=0).astype(np.float32)
    else:
        X_3d = np.zeros((0, n_freq, n_time), dtype=np.float32)

    if verbose:
        print(f"[lf_dataset] canonical dataset ready: {len(df_meta)} samples · X_3d.shape={X_3d.shape}")

    # ---- Write cache ----
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df_meta.to_parquet(cache_dir / "df_meta.parquet", index=False)
        np.save(cache_dir / "X_3d.npy", X_3d)
        (cache_dir / "params.json").write_text(json.dumps(params, indent=2))
        if verbose:
            print(f"  cached to {cache_dir}")

    return df_meta, ersp_list, X_3d


# ============================================================
# Convenience: default cache location
# ============================================================
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent / "outputs" / "_dataset" / "canonical"
)


def load_canonical(verbose: bool = True) -> Tuple[pd.DataFrame, List[np.ndarray], np.ndarray]:
    """
    Load the canonical dataset from the default cache location, building it
    from scratch if needed using the default INPUT_DIR from the notebook
    conventions.
    """
    input_dir = Path("../01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY").resolve()
    if not input_dir.exists():
        # Try absolute UNC path used on Lora's server
        input_dir = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\01_FBM_Analysis\outputs\04_ersp_LM_RAWONLY")
    return prepare_dataset(input_dir, cache_dir=DEFAULT_CACHE_DIR, verbose=verbose)
