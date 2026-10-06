#!/usr/bin/env python3
"""
lf_counters.py - what the two counters to convex NMF share.

run_counter_dpca.py (demixed PCA) and run_counter_cp.py (CP / PARAFAC tensor decomposition)
are fitted on the SAME electrodes, in the SAME order, from the SAME cache as every published
k-means / Ward / cNMF run, so their numbers can be put next to those. This module holds the
pieces both need and nothing else:

  load_cohort      the gated cohort of the newest cache (or LF_CONCAT_CACHE), through
                   build_concat_dataset with the cache's own input_dir, so the cohort tag
                   travels with the frame (df.attrs["cache"]) like everywhere else
  load_halves      the odd / even trial halves of every cube of that cohort (ERSP_halves
                   beside the cubes), cached once per cache version
  coords_for       fsaverage coordinates joined the way run_decomposition.py joins them
  band_tensor      (n, bands, time, condition) from X_concat (n, 103, 3 * 300)
  new_run_dir      outputs/clustering/<method>/<feature_set>/runs/<run_id>/ and a manifest
                   that carries the cohort tag, so lf_runs can tell a stale run from a
                   current one. The runs are NOT registered in outputs/clustering/index.json:
                   the site's visualizers expect a hard label per electrode and these
                   decompositions have none.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLUST = ROOT / "outputs" / "clustering"
COUNTER_CACHE = CLUST / "_counters"
COORDS = ROOT / "outputs" / "250_recon" / "fsaverage" / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv"
CONDS = ("audio", "picture", "reading")
DF_HZ = 1000.0 / 256.0                   # the cube's row step: 1 kHz / nfft 256
FREQ = np.arange(103) * DF_HZ
N_TIME = 300


def nz(s) -> str:
    return str(s).replace("_", "").replace("-", "").upper()


# ---------------------------------------------------------------------------------------
def load_cohort(verbose: bool = True):
    """(df_contacts, X_concat, cache_tag) of the newest cache, gated as the published runs are.

    build_concat_dataset refuses an input_dir other than the one the cache was built with
    (that is how it protects the cohort), so the input_dir is read from the cache's own
    params.json rather than guessed.
    """
    from functions.lf_concat import DEFAULT_CONCAT_CACHE, build_concat_dataset, cache_tag
    cache = Path(DEFAULT_CONCAT_CACHE)
    params = json.loads((cache / "params.json").read_text(encoding="utf-8"))
    df, X = build_concat_dataset(params["input_dir"], cache_dir=cache, verbose=verbose)
    df = df.reset_index(drop=True)
    return df, X, cache_tag(cache)


def _half_path(cube_path: str, which: int) -> str:
    return str(cube_path).replace("ERSP_matrix", "ERSP_halves").replace("_TN.npy", f"_TN_half{which}.npy")


def load_halves(df: pd.DataFrame, cache_name: str, *, conds: Sequence[str] = CONDS, verbose: bool = True):
    """X1, X2: (n, 103, 3 * 300) built from the odd / even trial halves of every cube.

    Read once per cache version and kept under outputs/clustering/_counters/, keyed by the
    cache name and the electrode list (a different cohort never reads another's file).
    """
    COUNTER_CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{cache_name}_{len(df)}"
    f = COUNTER_CACHE / f"halves_{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["keys"]) == list(df["patient_id"].astype(str) + "|" + df["contact_norm"]):
            if verbose:
                print(f"[counters] halves from {f.name}")
            return z["X1"], z["X2"]
    n = len(df)
    X1 = np.zeros((n, 103, N_TIME * len(conds)), np.float32)
    X2 = np.zeros_like(X1)
    missing = 0
    for i, r in df.iterrows():
        for b, c in enumerate(conds):
            p = r[f"file_path_{c}"]
            h1, h2 = _half_path(p, 1), _half_path(p, 2)
            if not (os.path.exists(h1) and os.path.exists(h2)):
                missing += 1
                X1[i, :, b * N_TIME:(b + 1) * N_TIME] = np.nan
                X2[i, :, b * N_TIME:(b + 1) * N_TIME] = np.nan
                continue
            X1[i, :, b * N_TIME:(b + 1) * N_TIME] = np.load(h1)
            X2[i, :, b * N_TIME:(b + 1) * N_TIME] = np.load(h2)
        if verbose and i % 200 == 0:
            print(f"[counters] halves {i}/{n}", flush=True)
    if missing:
        print(f"[counters] WARNING: {missing} cube halves missing; those entries are NaN")
    np.savez(f, X1=X1, X2=X2, keys=np.array(list(df["patient_id"].astype(str) + "|" + df["contact_norm"])))
    if verbose:
        print(f"[counters] halves -> {f}")
    return X1, X2


def coords_for(df: pd.DataFrame) -> np.ndarray:
    """(n, 3) fsaverage x, y, z per electrode; NaN where the recon has no contact."""
    if not COORDS.exists():
        return np.full((len(df), 3), np.nan)
    co = pd.read_csv(COORDS)
    co["key"] = [f"{p}|{nz(x)}" for p, x in zip(co["patient"], co["name"])]
    co = co.drop_duplicates("key")
    keys = pd.DataFrame({"key": [f"{p}|{nz(e)}" for p, e in zip(df["patient_id"], df["electrode"])]})
    return keys.merge(co[["key", "x", "y", "z"]], on="key", how="left")[["x", "y", "z"]].to_numpy(float)


# ---------------------------------------------------------------------------------------
def band_rows(bands) -> list:
    """Row indices of each (lo, hi) band in the 103-row cube: lo <= f < hi, like lf_features."""
    return [[r for r in range(103) if lo <= FREQ[r] < hi] for lo, hi in bands]


def band_tensor(X_concat: np.ndarray, bands, *, time_bins: int = 100, n_blocks: int = 3) -> np.ndarray:
    """(n, len(bands), time_bins, n_blocks) from X_concat (n, 103, n_blocks * 300).

    Band = mean of its rows (the mains rows are kept, exactly as the pipeline's own band
    features keep them); time = mean over 300 / time_bins consecutive bins per block.
    """
    n = X_concat.shape[0]
    rows = band_rows(bands)
    step = N_TIME // time_bins
    assert step * time_bins == N_TIME, f"time_bins must divide {N_TIME}"
    out = np.empty((n, len(bands), time_bins, n_blocks), np.float32)
    for b in range(n_blocks):
        blk = X_concat[:, :, b * N_TIME:(b + 1) * N_TIME]
        for j, rr in enumerate(rows):
            x = blk[:, rr, :].mean(1)                                   # (n, 300)
            out[:, j, :, b] = x.reshape(n, time_bins, step).mean(2)
    return out


def unit_norm_slices(Y: np.ndarray) -> np.ndarray:
    """Each electrode's slice scaled to unit Frobenius norm: shape only, like D.unit_norm."""
    flat = Y.reshape(Y.shape[0], -1)
    nrm = np.linalg.norm(flat, axis=1)
    nrm[nrm == 0] = 1.0
    return (flat / nrm[:, None]).reshape(Y.shape)


# ---------------------------------------------------------------------------------------
def new_run_dir(method: str, feature_set: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = CLUST / method / feature_set / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(run_dir: Path, *, method: str, method_label: str, feature_set: str,
                   feature_set_label: str, params: dict, summary: dict, artifacts: dict,
                   cache: dict, note: str = "") -> dict:
    man = dict(
        schema_version=1, method=method, method_label=method_label,
        feature_set=feature_set, feature_set_label=feature_set_label,
        run_id=run_dir.name, created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        params=params, summary=summary, artifacts=artifacts, cache=cache, note=note,
        predictor_type="loadings",
    )
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=2, default=_json_default), encoding="utf-8")
    return man


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def newest_cnmf(feature_set: str):
    """The newest cnmf run of a feature set (its G, components and electrode keys), or None."""
    d = CLUST / "cnmf" / feature_set / "runs"
    if not d.is_dir():
        return None
    for r in sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.name, reverse=True):
        if (r / "G_loadings.npy").exists() and (r / "components.npy").exists() and (r / "labels.csv").exists():
            lab = pd.read_csv(r / "labels.csv")
            return dict(run=r, G=np.load(r / "G_loadings.npy"), C=np.load(r / "components.npy"),
                        keys=list(lab["patient_id"].astype(str) + "|" + lab["contact_norm"].astype(str)))
    return None
