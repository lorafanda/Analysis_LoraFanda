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


# ---------------------------------------------------------------------------------------
# Shared by the partition counters of 2026-10-07 (run_crosshalf_clustering.py,
# run_kmedoids.py, run_consensus_halves.py, run_tight_peeling.py): the 13-170 Hz feature
# set, the muscle flag, a k-medoids, the cross-half distance, subsample stability, cluster
# fingerprints and the two figures every one of them draws.
# ---------------------------------------------------------------------------------------
HARM_ROWS: set = set()
for _h in (50, 100, 150, 200, 250, 300, 350, 400):
    HARM_ROWS |= {int(round(_h / DF_HZ)) + _d for _d in (-1, 0, 1)}


def band_features(X_concat: np.ndarray, lo: float = 13.0, hi: float = 170.0, bins: int = 30):
    """(n, n_bands * 3 * bins) and the bands used: the 15-band edges that fall inside
    [lo, hi), `bins` per condition, condition-major like the pipeline's concat features.
    13-170 Hz is the set the split-half measurement of 2026-10-07 found most dependable:
    no rows the 128-ms window cannot resolve, no copies of HG above 170 Hz."""
    from functions.lf_features import FREQ_BANDS_15_TO_400HZ as B15
    bands = [tuple(b) for b in B15 if b[0] >= lo and b[1] <= hi]
    Y = band_tensor(X_concat, bands, time_bins=bins)                    # (n, B, T, C)
    return np.transpose(Y, (0, 3, 1, 2)).reshape(Y.shape[0], -1).astype(np.float64), bands


def muscle_flags(X_concat: np.ndarray) -> np.ndarray:
    """The speech-muscle signature of the 2026-10-05 audit, per electrode: in some condition
    the response-half mean at 70-150 Hz is >= 1.5 dB and the one at 250-400 Hz is >= 0.75 dB
    and >= 40 % of it. Mains rows are left out of both bands."""
    hg = [r for r in range(103) if 70 <= FREQ[r] < 150 and r not in HARM_ROWS]
    hi = [r for r in range(103) if 250 <= FREQ[r] < 400 and r not in HARM_ROWS]
    flag = np.zeros(X_concat.shape[0], bool)
    for b in range(3):
        blk = X_concat[:, :, b * N_TIME + 150:(b + 1) * N_TIME]
        g, h = blk[:, hg, :].mean((1, 2)), blk[:, hi, :].mean((1, 2))
        flag |= (g >= 1.5) & (h >= 0.75) & (h >= 0.4 * g)
    return flag


def unit_norm(F: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return F / nrm


def corr_distance(F: np.ndarray) -> np.ndarray:
    """1 - Pearson r between electrodes (shape only), zero diagonal."""
    Z = F - F.mean(1, keepdims=True)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    D = 1.0 - Z @ Z.T
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 2.0)


def cross_half_distance(F1: np.ndarray, F2: np.ndarray) -> np.ndarray:
    """Euclidean distance between electrode i in half 1 and electrode j in half 2 (unit-normed),
    symmetrised. An electrode's own trial noise sits only on the diagonal, which is zeroed."""
    from scipy.spatial.distance import cdist
    D12 = cdist(unit_norm(F1), unit_norm(F2))
    D = 0.5 * (D12 + D12.T)
    np.fill_diagonal(D, 0.0)
    return D


def ward_on_distance(D: np.ndarray, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method="ward")
    return fcluster(Z, t=k, criterion="maxclust") - 1


def kmedoids(D: np.ndarray, k: int, *, n_init: int = 10, max_iter: int = 100, seed: int = 0):
    """Alternating k-medoids on a precomputed distance matrix: assign to the nearest medoid,
    move each medoid to the member with the smallest summed distance, until nothing moves.
    The best of n_init random starts. Returns (labels, medoids, cost)."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    best = None
    for _ in range(n_init):
        med = rng.choice(n, k, replace=False)
        for _it in range(max_iter):
            lab = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for j in range(k):
                m = np.flatnonzero(lab == j)
                if m.size:
                    new[j] = m[np.argmin(D[np.ix_(m, m)].sum(1))]
            if np.array_equal(new, med):
                break
            med = new
        lab = np.argmin(D[:, med], axis=1)
        cost = float(D[np.arange(n), med[lab]].sum())
        if best is None or cost < best[2]:
            best = (lab, med, cost)
    return best


def match_labels(a: np.ndarray, b: np.ndarray):
    """Relabel b onto a by the Hungarian method on the overlap table; (b_matched, agreement)."""
    from scipy.optimize import linear_sum_assignment
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    M = np.zeros((ka, kb))
    for i, j in zip(a, b):
        M[i, j] += 1
    r, c = linear_sum_assignment(-M)
    remap = {int(cj): int(ri) for ri, cj in zip(r, c)}
    bm = np.array([remap.get(int(x), ka + int(x)) for x in b])
    return bm, float(M[r, c].sum() / len(a))


def subsample_stability(cluster_fn, n: int, labels: np.ndarray, *, n_rounds: int = 20, frac: float = 0.8, seed: int = 0):
    """Co-association over random subsamples (a pair counts only in rounds where both were
    drawn); per cluster of `labels`, the mean co-association among its members.
    cluster_fn(idx) must return labels for the electrodes idx. Returns (per_cluster, C)."""
    rng = np.random.default_rng(seed)
    C = np.zeros((n, n)); N = np.zeros((n, n))
    for _ in range(n_rounds):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        lab = np.asarray(cluster_fn(idx))
        C[np.ix_(idx, idx)] += (lab[:, None] == lab[None, :]); N[np.ix_(idx, idx)] += 1
    with np.errstate(invalid="ignore"):
        C = np.where(N > 0, C / N, np.nan)
    per = {}
    for k in np.unique(labels):
        m = np.flatnonzero(labels == k)
        per[int(k)] = float(np.nanmean(C[np.ix_(m, m)][np.triu_indices(m.size, 1)])) if m.size > 1 else float("nan")
    return per, C


def fingerprints(X_concat: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Per cluster: size, HG (70-150 Hz) in the stimulus and the response half per condition,
    beta (13-30 Hz) per condition. Means in dB over the cluster's electrodes."""
    hg = [r for r in range(103) if 70 <= FREQ[r] < 150 and r not in HARM_ROWS]
    beta = [r for r in range(103) if 13 <= FREQ[r] < 30 and r not in HARM_ROWS]
    rows = []
    for k in np.unique(labels):
        m = labels == k
        rec = dict(cluster=int(k), n=int(m.sum()))
        for b, c in enumerate(CONDS):
            blk = X_concat[m][:, :, b * N_TIME:(b + 1) * N_TIME]
            rec[f"hg_stim_{c}"] = float(blk[:, hg, :150].mean()); rec[f"hg_resp_{c}"] = float(blk[:, hg, 150:].mean())
            rec[f"beta_{c}"] = float(blk[:, beta, :].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def cluster_maps(labels: np.ndarray, xyz: np.ndarray, out: Path, title: str):
    """Three projections of fsaverage, one colour per cluster; labels < 0 in grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    ok = ~np.isnan(xyz).any(1)
    cols = plt.get_cmap("tab10")(np.clip(labels, 0, None) % 10)
    cols[labels < 0] = (0.78, 0.78, 0.78, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, (p, q, ti) in zip(axes, [(0, 1, "axial"), (0, 2, "sagittal"), (1, 2, "coronal")]):
        a.scatter(xyz[ok, p], xyz[ok, q], c=cols[ok], s=12, lw=0); a.set_aspect("equal"); a.axis("off"); a.set_title(ti, fontsize=9)
    fig.suptitle(title, x=.02, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def cluster_courses(X_concat: np.ndarray, labels: np.ndarray, out: Path, title: str):
    """Mean HG (70-150 Hz) time course per cluster and condition, from the 300-bin cubes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    hg = [r for r in range(103) if 70 <= FREQ[r] < 150 and r not in HARM_ROWS]
    ks = np.unique(labels)
    cc = {"audio": "#e06c9f", "picture": "#4a6fa5", "reading": "#8c6d46"}
    fig, axes = plt.subplots(len(ks), 1, figsize=(7.5, 1.4 * len(ks) + 0.6), sharex=True, squeeze=False)
    t = np.linspace(0, 100, N_TIME, endpoint=False)
    for i, k in enumerate(ks):
        a = axes[i][0]; m = labels == k
        for b, c in enumerate(CONDS):
            a.plot(t, X_concat[m][:, hg, b * N_TIME:(b + 1) * N_TIME].mean((0, 1)), color=cc[c], lw=1.4, label=c if i == 0 else None)
        a.axvline(50, color="0.7", lw=.8, ls=":"); a.axhline(0, color="0.85", lw=.8)
        a.set_ylabel(("scattered" if k < 0 else f"cluster {k}") + f"\nn = {int(m.sum())}", fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    axes[-1][0].set_xlabel("% of the warped trial (50 = GO); mean HG, dB")
    fig.suptitle(title, x=.02, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
