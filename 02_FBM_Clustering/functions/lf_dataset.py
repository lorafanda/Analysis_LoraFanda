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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Defaults — single source of truth for the canonical filter
# ============================================================
DEFAULT_TASK             = "LM"
DEFAULT_CONDITIONS       = ("audio", "picture", "reading")
DEFAULT_N_FREQ           = 103   # 0-400 Hz cube since 2026-09-18 (was 129 = 0-500 Hz)
DEFAULT_N_TIME           = 300
DEFAULT_THR_POS          = 2.2
DEFAULT_MIN_PROP_POS     = 0.02
DEFAULT_THR_NEG          = -3.0
DEFAULT_MIN_PROP_NEG     = 0.04
DEFAULT_NOISE_K          = 2.0   # a bin counts only beyond k x the cube's split-half noise (2026-09-23); 0 = fixed thresholds


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
    X / X1..X8 / E1..E8, or anything with '+' in the label."""
    if label is None:
        return False
    # Treat '-' like '_' before the checks, matching lf_io_utils._is_non_neural.
    # Dash-separated names (Cing-L1, OFG-L15, Fp_L-5, ...) are legitimate neural
    # channels in the dykstra-style recon naming, not bipolar markers. Testing
    # for '-' silently discarded ALL 58 of EL034's contacts (losing the patient
    # outright) and 4 of EL046's; measured across the cohort, 62 of 3089 contacts
    # contain '-' and not one of them is a bipolar pair.
    s = str(label).strip().upper().replace("-", "_")
    if "+" in s:
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


# GVA (PAT_*) MICROELECTRODE contacts: the MicroEPI exports name a microwire bundle with a
# lower-case "m" between the macro shaft's stem and the contact number - ADm1, FODm12,
# HAGm3, PHDm7 - and the Micromed TRC of G-05 wrote its bundle as FOM; X1M/X2M are the
# discarded analog families. They are neural, just a different scale: no macro recon
# contact, so they do not belong in a macro-ERSP analysis, where they would cluster and
# pool but never appear on the brain.
#
# CASE MATTERS (2026-09-22). The rule used to be "the shaft, upper-cased, ends in M",
# which also took IDM and POM of PAT_2868 - DIXI depth shafts in the insula and the
# parietal lobe (IDM2, IDM5, POM2-4 have cubes) - out of every dataset as if they were
# microwires. Lora wants only stereo contacts in the v9 cohort: the microwires out, those
# five back in. The lower-case m is the recording's own convention and no macro contact
# of any cohort carries one before its number (checked over all 4796 recorded contacts).
_RX_MICRO_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*m[_-]?\d+$")
_MICRO_SHAFTS_UPPER = {"FOM", "X1M", "X2M"}        # TRC-export spellings, no lower-case m


def is_micro_electrode(label, patient_id=None) -> bool:
    """True for a GVA microelectrode contact. `patient_id` restricts the rule to the
    GVA cohort (BERN naming does not use this convention); pass None to apply it to
    any label."""
    if label is None:
        return False
    if patient_id is not None and not str(patient_id).upper().startswith("PAT"):
        return False
    s = str(label).strip().replace(" ", "")
    if _RX_MICRO_LABEL.fullmatch(s):
        return True
    shaft = re.sub(r"\d+$", "", s.replace("_", "").replace("-", "")).upper()
    return shaft in _MICRO_SHAFTS_UPPER


# Subdural GRID / strip shafts, per patient. Listed EXPLICITLY rather than matched by
# pattern: a regex like ^G[A-Z]$ happens to fit PAT_3415 today, but would silently
# swallow a future patient's depth shaft that shares the naming.
#
# The exclusion is at CONTACT level, not patient level, because the sample unit is one
# electrode. A depth contact in a mixed-implant patient is exactly as comparable to
# other depth contacts as anyone else's; what differs physically is the subdural grid
# contact — larger surface, cortical surface potential rather than intraparenchymal —
# so that is what should go.
#
#   PAT_3415   GA..GH = 8 shafts x 8 = a 64-contact 8x8 grid, alongside 57 depth
#              contacts (IMG, IPG, OI, OS, TA, TM, TP) which are KEPT.
#   EL044      not listed here: it is ECoG throughout (Pa 51, T 46, P 6, postP 5) with
#              no depth contacts to keep, so it stays a whole-patient exclusion.
GRID_SHAFTS: dict = {
    "PAT_3415": ("GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH"),
}


def is_grid_electrode(label, patient_id=None) -> bool:
    """True for a subdural grid/strip contact listed in GRID_SHAFTS for that patient."""
    if label is None or patient_id is None:
        return False
    shafts = GRID_SHAFTS.get(str(patient_id))
    if not shafts:
        return False
    s = str(label).replace("_", "").replace("-", "").upper()
    return re.sub(r"\d+$", "", s) in {x.upper() for x in shafts}


# Shafts excluded for SIGNAL QUALITY rather than geometry — heavy noise contamination
# judged from the raw traces. Distinct from GRID_SHAFTS because the reason is different:
# a grid contact is a valid recording of the wrong kind, whereas these are unusable.
# Applied in prepare_dataset, so EVERY track (per-condition and concatenated) drops them.
#
#   PAT_3415  IMG, TA, IPG — leaving OI, OS, TM, TP as its usable depth coverage.
NOISY_SHAFTS: dict = {
    "PAT_3415": ("IMG", "TA", "IPG"),
}


def is_noisy_electrode(label, patient_id=None) -> bool:
    """True for a contact on a shaft listed in NOISY_SHAFTS for that patient."""
    if label is None or patient_id is None:
        return False
    shafts = NOISY_SHAFTS.get(str(patient_id))
    if not shafts:
        return False
    s = str(label).replace("_", "").replace("-", "").upper()
    return re.sub(r"\d+$", "", s) in {x.upper() for x in shafts}


# ============================================================
# Public: prepare_dataset
# ============================================================
def _halves_noise(cube_paths, n_threads: int = 8) -> List[float]:
    """Noise of each trial-averaged cube, from its two halves in ERSP_halves:
    sd(half1 - half2) / 2 in dB (var(h1 - h2) = 2 sigma^2 / (N/2) = 4 sigma^2 / N, and the
    full mean has sigma^2 / N). NaN where a half is missing or the shapes differ."""
    def one(p):
        p = str(p)
        h1 = p.replace("ERSP_matrix", "ERSP_halves").replace("_TN.npy", "_TN_half1.npy")
        h2 = h1.replace("_half1.npy", "_half2.npy")
        try:
            a, b = np.load(h1), np.load(h2)
        except (OSError, ValueError):
            return float("nan")
        if a.shape != b.shape:
            return float("nan")
        d = (a - b).astype(np.float32)
        if np.isfinite(d).sum() < 100:
            return float("nan")
        return float(np.nanstd(d)) / 2.0
    with ThreadPoolExecutor(n_threads) as ex:
        return list(ex.map(one, cube_paths))


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
    noise_k: float = DEFAULT_NOISE_K,
    apply_high_activity: bool = True,
    exclude_micro: bool = True,
    cache_dir = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[np.ndarray], np.ndarray]:
    """
    Build the canonical clustering dataset.

    Returns
    -------
    df_meta : DataFrame, columns
        sample_idx, patient_id, condition, task, electrode, file_path,
        noise_db, thr_pos_used, thr_neg_used, prop_above_pos, prop_below_neg, high_activity
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
        "exclude_micro": bool(exclude_micro),   # part of the key: a cache built without
                                                # the micro filter must not be reused
        # Same reasoning: a cache built before a shaft was blacklisted still contains it.
        "noisy_shafts": {k: sorted(v) for k, v in sorted(NOISY_SHAFTS.items())},
        "conditions": list(conditions),
        "n_freq": int(n_freq),
        "n_time": int(n_time),
        "thr_pos": float(thr_pos),
        "min_prop_pos": float(min_prop_pos),
        "thr_neg": float(thr_neg),
        "min_prop_neg": float(min_prop_neg),
        "noise_k": float(noise_k),
        "apply_high_activity": bool(apply_high_activity),
        # Version tag: bump if filter logic changes incompatibly so old caches invalidate
        # 2 (2026-09-22): the micro rule is the lower-case-m spelling; IDM / POM of PAT_2868 are data
        # 3 (2026-09-23): gate thresholds scale with the split-half noise (noise_k); noise_db,
        #                 thr_pos_used, thr_neg_used columns
        "schema": 3,
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
            else:
                diff = sorted(k for k in set(params) | set(cached_params) if params.get(k) != cached_params.get(k))
                raise RuntimeError(
                    f"[lf_dataset] {cache_dir} was built with other params ({', '.join(diff)}) and is not "
                    f"rebuilt in place (that would silently replace the cohort every run was fitted on): "
                    f"build the next version with rebuild_concat_cache.py --version N --apply")

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

    # ---- Microelectrode filter (GVA "*M" shafts) ----
    # These have no macro recon contact, so they cluster/pool but never render on the
    # brain. Set exclude_micro=False to keep them (e.g. for a micro-specific analysis).
    if exclude_micro and len(df_meta):
        micro_mask = df_meta.apply(
            lambda r: is_micro_electrode(r["electrode"], r["patient_id"]), axis=1)
        n_micro = int(micro_mask.sum())
        if n_micro:
            keep = ~micro_mask
            shafts = sorted({re.sub(r"\d+$", "", str(e).replace("_", "").upper())
                             for e in df_meta.loc[micro_mask, "electrode"]})
            df_meta = df_meta[keep].reset_index(drop=True)
            ersp_list = [ersp_list[i] for i in np.where(keep.to_numpy())[0]]
            if verbose:
                print(f"  excluded {n_micro} microelectrode channels ({', '.join(shafts)}) "
                      f"→ {len(df_meta)} samples")

    # ---- Noise-contaminated shafts (per patient, see NOISY_SHAFTS) ----
    # Excluded for signal quality, not geometry: unlike a grid contact these are not a
    # valid recording of a different kind, they are unusable. Applied here rather than in
    # lf_concat so the per-condition tracks drop them too.
    if len(df_meta):
        noisy_mask = df_meta.apply(
            lambda r: is_noisy_electrode(r["electrode"], r["patient_id"]), axis=1)
        n_noisy = int(noisy_mask.sum())
        if n_noisy:
            keep = ~noisy_mask
            by = (df_meta.loc[noisy_mask].groupby("patient_id")["electrode"]
                  .apply(lambda s: sorted({re.sub(r"\d+$", "", str(e).replace("_", "").upper())
                                           for e in s})).to_dict())
            df_meta = df_meta[keep].reset_index(drop=True)
            ersp_list = [ersp_list[i] for i in np.where(keep.to_numpy())[0]]
            if verbose:
                print(f"  excluded {n_noisy} noise-contaminated channels {by} "
                      f"→ {len(df_meta)} samples")

    # ---- High-activity computation (always computed, optionally filtered) ----
    # Since 2026-09-23 the thresholds follow the electrode's own noise. The trial-averaged
    # cube's noise is sd(half1 - half2) / 2 (ERSP_halves), ~ 1/sqrt(N trials): 2.1 dB at 8
    # trials, 0.9 dB at 45 - so a fixed +2.2 dB let the low-trial patients gate on noise
    # (EL033, EL038, EL043, EL046, PAT_6619 were 100 % gated in v9). A bin now counts only
    # above max(thr_pos, noise_k * noise) and below min(thr_neg, -noise_k * noise); an
    # electrode without halves keeps the fixed thresholds. noise_k = 0 is the old rule.
    noise_db = _halves_noise(df_meta["file_path"].tolist()) if (noise_k and len(df_meta)) else [float("nan")] * len(df_meta)
    thr_up = [max(thr_pos, noise_k * n) if np.isfinite(n) else thr_pos for n in noise_db]
    thr_dn = [min(thr_neg, -noise_k * n) if np.isfinite(n) else thr_neg for n in noise_db]
    prop_pos = [float((arr > tu).mean()) for arr, tu in zip(ersp_list, thr_up)]
    prop_neg = [float((arr < td).mean()) for arr, td in zip(ersp_list, thr_dn)]
    df_meta["noise_db"] = noise_db
    df_meta["thr_pos_used"] = thr_up
    df_meta["thr_neg_used"] = thr_dn
    if verbose and noise_k and len(df_meta):
        nd = np.asarray(noise_db, dtype=float)
        raised = int(sum(1 for t in thr_up if t > thr_pos))
        print(f"  noise-scaled gate (k={noise_k:g}): halves for {int(np.isfinite(nd).sum())} of {len(nd)} cubes, "
              f"noise median {np.nanmedian(nd):.2f} dB, threshold raised above {thr_pos} dB for {raised}")
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
            print(f"  high-activity gate (above max({thr_pos}, {noise_k:g}·noise) prop≥{min_prop_pos}, "
                  f"below min({thr_neg}, -{noise_k:g}·noise) prop≥{min_prop_neg}): kept {n_high}, dropped {n_low}")

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
