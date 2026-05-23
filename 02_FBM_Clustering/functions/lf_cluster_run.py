"""
lf_cluster_run.py — One-stop orchestrator for clustering runs.

Wraps existing primitives in `lf_clustering_methods` and `lf_clustering` and
adds the run lifecycle pieces:

  - fitted model persistence (joblib)
  - 1-NN predictor (so any method can score new samples)
  - manifest.json + index.json (powers the website)
  - within / between similarity metrics + figures
  - layout: outputs/clustering/<method>/<feature_set>/runs/<timestamp>/

Three public functions are the whole API:

  fit_and_save(X, df_keep, method, feature_set, params, ...) -> manifest dict
  load_run(method, feature_set, run_id="latest", outputs_root=None) -> bundle
  assign_new(bundle, X_new) -> {labels, confidence, predictor_type, ...}

Older save helpers (s71_save_run, s71_save_gated_blob_run) are left untouched
so legacy notebooks keep working.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform

# Existing primitives (do not duplicate)
from functions.lf_clustering_methods import (
    s50_run_kmeans,
    s51_run_hc,
    s52_cut_hc,
)


# ============================================================
# Constants
# ============================================================

SCHEMA_VERSION = 1

# Anchor the default outputs root to this module's location so the runs land at
# `<repo>/02_FBM_Clustering/outputs/clustering/` regardless of the notebook's
# working directory. (Previously this was a relative path which double-nested
# when the notebook ran from inside 02_FBM_Clustering/.)
DEFAULT_OUTPUTS_ROOT = (Path(__file__).resolve().parent.parent / "outputs" / "clustering")

METHOD_LABELS = {
    "kmeans": "K-Means",
    "hierarchical": "Hierarchical",
}

FEATURE_SET_LABELS = {
    "raw": "Raw ERSP",
    "blob": "Blob Features",
    "minus101": "−1 / 0 / +1 Segmentation",
    "hg": "High-Gamma Time Series",
}


# ============================================================
# Public API: fit_and_save
# ============================================================

def fit_and_save(
    X: np.ndarray,
    *,
    df_keep: Optional[pd.DataFrame] = None,
    method: str,
    feature_set: str,
    params: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    scaler: Optional[Any] = None,
    method_label: Optional[str] = None,
    feature_set_label: Optional[str] = None,
    notebook: Optional[str] = None,
    outputs_root: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    compute_gap: bool = False,
    gap_n_refs: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fit a clustering model, persist everything needed to reuse it on new data,
    and write a manifest the website can render.

    Parameters
    ----------
    X : (n_samples, n_features) feature matrix
    df_keep : optional sample-level metadata (n_samples rows). If provided,
        cluster labels are attached and saved as df_keep_with_clusters.parquet.
    method : "kmeans" or "hierarchical"
    feature_set : "raw", "blob", "minus101", or any other id
    params : method-specific parameters. See `_fit_kmeans` / `_fit_hierarchical`
        for the keys each method accepts.
    feature_names : optional column names for X (saved into feature_schema.json)
    scaler : optional fitted preprocessing transform (saved as scaler.joblib)
    method_label, feature_set_label : optional human-readable names. If omitted,
        defaults from METHOD_LABELS / FEATURE_SET_LABELS are used.
    notebook : path of the calling notebook (for traceability in manifest)
    outputs_root : root for clustering outputs. Defaults to
        02_FBM_Clustering/outputs/clustering.
    run_id : override the timestamp run id (mostly for tests / replays).

    Returns
    -------
    manifest : dict matching the on-disk manifest.json
    """
    method = method.lower()
    if method not in ("kmeans", "hierarchical"):
        raise ValueError(f"method must be 'kmeans' or 'hierarchical', got {method!r}")

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")

    params = dict(params or {})
    outputs_root = Path(outputs_root) if outputs_root else DEFAULT_OUTPUTS_ROOT
    if run_id is None:
        run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = outputs_root / method / feature_set / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- fit ----
    if method == "kmeans":
        fit = _fit_kmeans(X, params, verbose=verbose)
    else:
        fit = _fit_hierarchical(X, params, verbose=verbose)

    labels = fit["labels"]
    model = fit["model"]
    extras = fit.get("extras", {})  # method-specific objects to persist

    # ---- 1-NN predictor (uniform inference path for every method) ----
    predictor = KNeighborsClassifier(n_neighbors=1).fit(X, labels)
    predictor_type = "centroid" if method == "kmeans" else "1-NN"

    # ---- metrics + figures ----
    # Compute per-sample silhouette once and reuse for both metrics and labels.csv.
    if len(np.unique(labels)) > 1:
        sil_per_point = silhouette_samples(X, labels)
    else:
        sil_per_point = np.full(len(labels), float("nan"))
    metrics = _compute_metrics(X, labels, df_keep=df_keep, sil_per_point=sil_per_point)
    figures = _save_figures(
        X, labels, metrics, run_dir,
        method=method, feature_set=feature_set,
        method_label=method_label or METHOD_LABELS.get(method, method),
        feature_set_label=feature_set_label or FEATURE_SET_LABELS.get(feature_set, feature_set),
        sweep=fit.get("sweep"),
    )

    # ---- persist artifacts ----
    artifacts: Dict[str, str] = {}

    joblib.dump(model, run_dir / "model.joblib")
    artifacts["model"] = "model.joblib"

    joblib.dump(predictor, run_dir / "predictor.joblib")
    artifacts["predictor"] = "predictor.joblib"

    if scaler is not None:
        joblib.dump(scaler, run_dir / "scaler.joblib")
        artifacts["scaler"] = "scaler.joblib"

    np.save(run_dir / "X_train.npy", X.astype(np.float32))
    artifacts["X_train"] = "X_train.npy"

    feature_schema = {
        "n_features": int(X.shape[1]),
        "dtype": str(X.dtype),
        "feature_names": list(feature_names) if feature_names is not None else None,
    }
    _write_json(run_dir / "feature_schema.json", feature_schema)
    artifacts["feature_schema"] = "feature_schema.json"

    # `labels.csv` is the website's primary data source — when df_keep is
    # provided, embed all sample-level metadata so the per-cluster patient /
    # condition histograms can render without a parquet reader in the browser.
    cluster_col = f"cluster_{method}_{feature_set}"
    if df_keep is not None:
        df_out = _attach_labels(df_keep, labels, method=method, feature_set=feature_set)
        if "sample_idx" not in df_out.columns:
            df_out.insert(0, "sample_idx", np.arange(len(df_out), dtype=np.int64))
        # P2: per-sample silhouette in labels.csv — the website sorts cluster
        # thumbnails by this and shows the value as a confidence badge.
        df_out["silhouette"] = sil_per_point.astype(np.float32)
        df_out.to_csv(run_dir / "labels.csv", index=False)
        df_out.to_parquet(run_dir / "df_keep_with_clusters.parquet", index=False)
        artifacts["df_keep_with_clusters"] = "df_keep_with_clusters.parquet"
    else:
        pd.DataFrame({
            "sample_idx":   np.arange(len(labels)),
            cluster_col:    labels.astype(np.int32),
            "silhouette":   sil_per_point.astype(np.float32),
        }).to_csv(run_dir / "labels.csv", index=False)
    artifacts["labels"] = "labels.csv"

    _write_json(run_dir / "metrics.json", metrics)
    artifacts["metrics"] = "metrics.json"

    artifacts.update(figures)

    # Hierarchical: also save linkage + distance matrices (handy for re-cuts)
    if method == "hierarchical":
        if "Z" in extras:
            np.save(run_dir / "linkage_Z.npy", extras["Z"])
            artifacts["linkage_Z"] = "linkage_Z.npy"
        if "D" in extras:
            np.save(run_dir / "distance_D.npy", extras["D"].astype(np.float32))
            artifacts["distance_D"] = "distance_D.npy"

    # K-sweep: silhouette by k (kmeans + hierarchical when k_range was supplied)
    if "sweep" in fit and fit["sweep"]:
        _write_json(run_dir / "silhouette_by_k.json",
                    {str(k): v for k, v in fit["sweep"]["sil_by_k"].items()})
        artifacts["silhouette_sweep"] = "silhouette_by_k.json"

    # K-sweep: Tibshirani 2001 gap statistic (optional — slow because it
    # fits KMeans gap_n_refs extra times per K on uniform-bounding-box null
    # data). Only computed when compute_gap=True AND a k_range was used.
    if compute_gap and "sweep" in fit and fit["sweep"]:
        ks = sorted(int(k) for k in fit["sweep"]["sil_by_k"].keys())
        if verbose:
            print(f"[fit_and_save] computing gap statistic over K={ks} with {gap_n_refs} null refs each")
        gap_by_k = _compute_gap_statistic(X, ks, n_refs=int(gap_n_refs), verbose=verbose)
        _write_json(run_dir / "gap_by_k.json",
                    {str(k): v for k, v in gap_by_k.items()})
        artifacts["gap_sweep"] = "gap_by_k.json"
        _save_gap_curve(run_dir / "gap_by_k.png", gap_by_k,
                        title_prefix=f"{method_label or method} · {feature_set_label or feature_set}")
        artifacts["gap_curve_figure"] = "gap_by_k.png"

    # Per-K labels: when present (k_range was used), write a wide-format CSV
    # so MOBA's K slider can scrub across cuts without re-fitting.
    # Columns: sample_idx, k_5, k_6, ..., k_15
    if extras.get("labels_by_k"):
        lbk = extras["labels_by_k"]
        ks_sorted = sorted(int(k) for k in lbk.keys())
        n = int(X.shape[0])
        labels_by_k_df = pd.DataFrame({"sample_idx": np.arange(n, dtype=np.int64)})
        for k in ks_sorted:
            labels_by_k_df[f"k_{k}"] = np.asarray(lbk[k]).astype(np.int32)
        labels_by_k_df.to_csv(run_dir / "cluster_labels_by_k.csv", index=False)
        artifacts["labels_by_k"] = "cluster_labels_by_k.csv"

    # ---- manifest ----
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "method_label": method_label or METHOD_LABELS.get(method, method),
        "feature_set": feature_set,
        "feature_set_label": feature_set_label or FEATURE_SET_LABELS.get(feature_set, feature_set),
        "run_id": run_id,
        "created_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "params": params,
        "summary": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_clusters": int(metrics["n_clusters"]),
            "silhouette_overall": metrics["silhouette_overall"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "davies_bouldin": metrics["davies_bouldin"],
            **({"best_k": fit["sweep"]["best_k"]} if fit.get("sweep") else {}),
            **({"k_range": fit["sweep"]["k_range"]} if fit.get("sweep") else {}),
        },
        "predictor_type": predictor_type,
        "artifacts": artifacts,
        "notebook": notebook,
        **_safe_git_info(),
        "user": getpass.getuser(),
        "host": platform.node(),
        "python_version": platform.python_version(),
    }
    _write_json(run_dir / "manifest.json", manifest)

    # ---- update index ----
    _update_index(outputs_root, manifest)

    if verbose:
        s = manifest["summary"]
        print(f"[fit_and_save] {method}/{feature_set}/{run_id}")
        print(f"  n_samples={s['n_samples']} n_clusters={s['n_clusters']} "
              f"silhouette={s['silhouette_overall']:.3f}")
        print(f"  -> {run_dir}")

    return manifest


# ============================================================
# Public API: load_run
# ============================================================

def load_run(
    method: str,
    feature_set: str,
    run_id: str = "latest",
    outputs_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load a saved run into a bundle dict ready for `assign_new`.

    Returns a dict with: manifest, model, predictor, scaler (or None),
    X_train, labels, run_dir.
    """
    outputs_root = Path(outputs_root) if outputs_root else DEFAULT_OUTPUTS_ROOT
    method_dir = outputs_root / method / feature_set / "runs"
    if not method_dir.exists():
        raise FileNotFoundError(f"No runs found at {method_dir}")

    if run_id == "latest":
        run_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"No runs in {method_dir}")
        run_dir = run_dirs[-1]
    else:
        run_dir = method_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_dir}")

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)

    bundle: Dict[str, Any] = {
        "manifest": manifest,
        "run_dir": run_dir,
        "model": joblib.load(run_dir / "model.joblib"),
        "predictor": joblib.load(run_dir / "predictor.joblib"),
        "scaler": joblib.load(run_dir / "scaler.joblib") if (run_dir / "scaler.joblib").exists() else None,
        "X_train": np.load(run_dir / "X_train.npy"),
        "labels": pd.read_csv(run_dir / "labels.csv")["cluster"].to_numpy(dtype=np.int32),
    }
    return bundle


# ============================================================
# Public API: assign_new
# ============================================================

def assign_new(
    bundle: Dict[str, Any],
    X_new: np.ndarray,
    *,
    use_native_predict: bool = True,
) -> Dict[str, Any]:
    """
    Assign new samples to clusters and score how well they fit.

    Parameters
    ----------
    bundle : output of `load_run`
    X_new : (m, n_features) feature matrix in the same space as X_train
        (apply the same scaler before calling — the bundle's scaler is exposed
        but not auto-applied since callers usually have their own pipeline).
    use_native_predict : if True (default), KMeans uses model.predict().
        If False, the 1-NN predictor is used uniformly for all methods.

    Returns
    -------
    {
      "labels":          (m,) int cluster ids
      "confidence":      (m,) per-point silhouette against training labels
      "predictor_type":  "centroid" | "1-NN"
    }

    Confidence is the silhouette of each new point against the bundled
    training labels. Range [-1, 1]; closer to 1 means the point fits its
    assigned cluster much better than any other cluster.
    """
    X_new = np.asarray(X_new)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    manifest = bundle["manifest"]
    method = manifest["method"]
    X_train = bundle["X_train"]
    labels_train = bundle["labels"]

    if X_new.shape[1] != X_train.shape[1]:
        raise ValueError(
            f"X_new has {X_new.shape[1]} features but training X has {X_train.shape[1]}."
        )

    if method == "kmeans" and use_native_predict:
        labels_new = bundle["model"].predict(X_new).astype(np.int32)
        predictor_type = "centroid"
    else:
        labels_new = bundle["predictor"].predict(X_new).astype(np.int32)
        predictor_type = "1-NN"

    confidence = _silhouette_of_new_points(X_new, labels_new, X_train, labels_train)

    return {
        "labels": labels_new,
        "confidence": confidence,
        "predictor_type": predictor_type,
    }


# ============================================================
# Internals: fitting
# ============================================================

def _fit_kmeans(X: np.ndarray, params: Dict[str, Any], *, verbose: bool) -> Dict[str, Any]:
    """
    Accepts either:
      - {"k": int, "random_state": int, "n_init": int}             single fit
      - {"k_range": [..], "random_state": int, "n_init": int}      sweep + best K

    Returns dict with: model, labels, sweep (or None), extras={labels_by_k} when sweeping.
    """
    rs = int(params.get("random_state", 42))
    n_init = int(params.get("n_init", 10))

    if "k_range" in params and params["k_range"]:
        k_range = list(params["k_range"])
        labels, best_k, sil_by_k, labels_by_k = s50_run_kmeans(
            X, k_range=k_range, random_state=rs, n_init=n_init, verbose=verbose
        )
        # refit at best_k to get a fitted estimator (s50 returns labels only)
        model = KMeans(n_clusters=int(best_k), random_state=rs, n_init=n_init).fit(X)
        # Normalize labels_by_k so int-keys land alongside the labels.csv labels.
        labels_by_k_norm = {int(k): np.asarray(v).astype(np.int32)
                            for k, v in (labels_by_k or {}).items()}
        return {
            "model": model,
            "labels": labels.astype(np.int32),
            "sweep": {"best_k": int(best_k), "sil_by_k": sil_by_k, "k_range": k_range},
            "extras": {"labels_by_k": labels_by_k_norm},
        }
    else:
        k = int(params["k"])
        model = KMeans(n_clusters=k, random_state=rs, n_init=n_init).fit(X)
        return {"model": model, "labels": model.labels_.astype(np.int32)}


def _fit_hierarchical(X: np.ndarray, params: Dict[str, Any], *, verbose: bool) -> Dict[str, Any]:
    """
    Accepts:
      - {"linkage": "ward"|"average"|"complete", "metric": "euclidean"|...,
         "n_clusters": int  OR  "cut_height": float}     single cut
      - {"linkage": ..., "metric": ..., "k_range": [..]}  K-sweep: linkage computed
         once, then cut at every K. Best K picked by silhouette. Labels at every K
         are persisted to cluster_labels_by_k.csv so the website can scrub.

    Returns dict with: model (the linkage matrix Z packaged with metadata),
    labels (best K when sweeping, else the single cut), sweep (or None when
    a single n_clusters was requested), extras={Z, D, labels_by_k? }.
    """
    linkage_method = str(params.get("linkage", "ward"))
    metric = str(params.get("metric", "euclidean"))
    cut_height = params.get("cut_height")
    n_clusters = params.get("n_clusters")
    k_range = params.get("k_range")

    Z, D = s51_run_hc(X, method=linkage_method, metric=metric, verbose=verbose)

    if k_range:
        # K-sweep: cut at every K, score silhouette, pick best.
        k_range = [int(k) for k in k_range]
        sil_by_k: Dict[int, float] = {}
        labels_by_k: Dict[int, np.ndarray] = {}
        for k in k_range:
            lbl = s52_cut_hc(Z, n_clusters=k)
            lbl = (lbl - lbl.min()).astype(np.int32)
            labels_by_k[int(k)] = lbl
            if len(np.unique(lbl)) > 1:
                sil_by_k[int(k)] = float(silhouette_score(X, lbl))
            else:
                sil_by_k[int(k)] = float("nan")
            if verbose:
                print(f"  HC sweep k={k}: silhouette={sil_by_k[int(k)]:.3f}")
        # Best K = max silhouette (ignore NaN)
        finite = {k: v for k, v in sil_by_k.items() if not np.isnan(v)}
        best_k = max(finite, key=finite.get) if finite else int(k_range[0])
        labels = labels_by_k[int(best_k)]
        sweep = {"best_k": int(best_k), "sil_by_k": sil_by_k, "k_range": k_range}
    else:
        labels = s52_cut_hc(Z, n_clusters=n_clusters, cut_height=cut_height)
        labels = (labels - labels.min()).astype(np.int32)
        sweep = None
        labels_by_k = None

    model = {
        "kind": "hierarchical",
        "linkage": linkage_method,
        "metric": metric,
        "n_clusters": int(len(np.unique(labels))),
        "Z": Z,
    }
    extras: Dict[str, Any] = {"Z": Z, "D": D}
    if labels_by_k is not None:
        extras["labels_by_k"] = labels_by_k
    return {
        "model": model,
        "labels": labels,
        "sweep": sweep,
        "extras": extras,
    }


# ============================================================
# Internals: metrics
# ============================================================

def _patient_cohort(pid: Any) -> str:
    """Classify a patient ID into a recording center / cohort."""
    s = str(pid).strip().upper()
    if s.startswith("EL"):                                    return "EL"
    if s.startswith("PAT"):                                   return "PAT"
    if s.startswith("MICRO") or s.startswith(("G-", "B-")):   return "MICRO"
    return "OTHER"


def _compute_metrics(X: np.ndarray, labels: np.ndarray,
                      df_keep: Optional[pd.DataFrame] = None,
                      sil_per_point: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Per-run metrics: overall + per-cluster cohesion / separation / diversity.

    Within-cluster:  mean / std pairwise distance among members (cohesion).
    Between-cluster: pairwise centroid distance matrix (separation).
    Diversity:       n_patients, n_centers per cluster (when df_keep is given) —
                     a single-patient cluster of 30 is far more suspect than a
                     6-patient cluster of 6, so this is the headline diagnostic
                     for "is this cluster real or a patient artifact?"
    """
    labels = np.asarray(labels).astype(np.int32)
    uniq = np.unique(labels)
    n_clusters = int(len(uniq))

    # Overall scores
    if n_clusters > 1:
        sil = float(silhouette_score(X, labels))
        if sil_per_point is None:
            sil_per_point = silhouette_samples(X, labels)
        ch = float(calinski_harabasz_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
    else:
        sil = float("nan")
        sil_per_point = np.full(len(labels), float("nan"))
        ch = float("nan")
        db = float("nan")

    # Centroids (mean of members) — works for both KMeans and Hierarchical
    centroids = np.stack([X[labels == c].mean(axis=0) for c in uniq], axis=0)
    centroid_dist = squareform(pdist(centroids, metric="euclidean"))

    # Per-cluster patient diversity (when metadata is available)
    have_pid = df_keep is not None and "patient_id" in df_keep.columns and len(df_keep) == len(labels)
    pid_arr = df_keep["patient_id"].astype(str).to_numpy() if have_pid else None
    cohort_arr = (np.array([_patient_cohort(p) for p in pid_arr])
                  if pid_arr is not None else None)

    per_cluster: Dict[str, Dict[str, float]] = {}
    for c in uniq:
        idx = np.where(labels == c)[0]
        size = int(len(idx))
        if size > 1:
            within = pdist(X[idx], metric="euclidean")
            within_mean = float(within.mean())
            within_std = float(within.std())
        else:
            within_mean = 0.0
            within_std = 0.0
        sil_c = float(np.nanmean(sil_per_point[idx])) if size and n_clusters > 1 else float("nan")
        entry: Dict[str, Any] = {
            "size": size,
            "silhouette_mean": sil_c,
            "within_dist_mean": within_mean,
            "within_dist_std": within_std,
        }
        if pid_arr is not None:
            pids_in_cluster = pid_arr[idx]
            entry["n_patients"] = int(len(set(pids_in_cluster)))
            entry["n_centers"]  = int(len(set(cohort_arr[idx])))
            entry["patients"]   = sorted(set(pids_in_cluster.tolist()))
        per_cluster[str(int(c))] = entry

    return {
        "n_samples": int(len(labels)),
        "n_clusters": n_clusters,
        "silhouette_overall": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "per_cluster": per_cluster,
        "centroid_distance_matrix": centroid_dist.tolist(),
        "cluster_ids": [int(c) for c in uniq],
    }


def _silhouette_of_new_points(
    X_new: np.ndarray,
    labels_new: np.ndarray,
    X_train: np.ndarray,
    labels_train: np.ndarray,
) -> np.ndarray:
    """
    Silhouette of each new point against the training set's existing clusters.

    s = (b - a) / max(a, b)
      a = mean distance from new point to assigned cluster's training points
      b = min over other clusters of mean distance to that cluster's training points
    """
    out = np.zeros(len(X_new), dtype=np.float32)
    cluster_ids = np.unique(labels_train)
    cluster_idx = {int(c): np.where(labels_train == c)[0] for c in cluster_ids}

    for i, x in enumerate(X_new):
        c_assigned = int(labels_new[i])
        if c_assigned not in cluster_idx:
            out[i] = float("nan")
            continue
        own = cluster_idx[c_assigned]
        if len(own) == 0:
            out[i] = float("nan")
            continue
        a = float(np.linalg.norm(X_train[own] - x, axis=1).mean())
        bs = []
        for c, idx in cluster_idx.items():
            if c == c_assigned or len(idx) == 0:
                continue
            bs.append(float(np.linalg.norm(X_train[idx] - x, axis=1).mean()))
        if not bs:
            out[i] = 0.0
            continue
        b = min(bs)
        denom = max(a, b)
        out[i] = 0.0 if denom == 0 else (b - a) / denom
    return out


# ============================================================
# Internals: figures
# ============================================================

def _save_figures(
    X: np.ndarray,
    labels: np.ndarray,
    metrics: Dict[str, Any],
    run_dir: Path,
    *,
    method: str,
    feature_set: str,
    method_label: str,
    feature_set_label: str,
    sweep: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate the standard figure set:

      centroids.png                    — cluster × feature heatmap (default view)
      silhouette_per_cluster.png       — bar chart, sorted; tells you which clusters are tight
      similarity_heatmap.png           — sample × sample distance matrix reordered by cluster
      centroid_distance_heatmap.png    — k × k inter-centroid distances; cluster separation
      silhouette_by_k.png              — sweep curve (KMeans with k_range only)
    """
    figs: Dict[str, str] = {}
    title_prefix = f"{method_label} · {feature_set_label}"

    uniq = np.unique(labels)
    centroids = np.stack([X[labels == c].mean(axis=0) for c in uniq], axis=0)

    # 1) centroids heatmap (cluster × feature)
    p = run_dir / "centroids.png"
    fig, ax = plt.subplots(figsize=(min(0.05 * X.shape[1] + 4, 18),
                                    min(0.25 * len(uniq) + 2, 12)))
    vmax = float(np.max(np.abs(centroids))) or 1.0
    im = ax.imshow(centroids, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("feature")
    ax.set_ylabel("cluster")
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels([str(int(c)) for c in uniq])
    ax.set_title(f"{title_prefix} — centroids")
    plt.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs["centroids_figure"] = "centroids.png"

    # 2) silhouette per cluster
    pc = metrics["per_cluster"]
    cluster_ids = sorted(pc.keys(), key=lambda k: -pc[k]["silhouette_mean"]
                         if not np.isnan(pc[k]["silhouette_mean"]) else 0)
    sils = [pc[c]["silhouette_mean"] for c in cluster_ids]
    sizes = [pc[c]["size"] for c in cluster_ids]

    p = run_dir / "silhouette_per_cluster.png"
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(cluster_ids) + 4), 4))
    bars = ax.bar(range(len(cluster_ids)), sils,
                  color=["#7a8c6e" if s >= 0 else "#b85c6e" for s in sils])
    ax.axhline(metrics["silhouette_overall"], color="black", lw=1, ls="--",
               label=f"overall = {metrics['silhouette_overall']:.3f}")
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f"{c}\n(n={sizes[i]})" for i, c in enumerate(cluster_ids)],
                       fontsize=8)
    ax.set_ylabel("mean silhouette")
    ax.set_title(f"{title_prefix} — silhouette by cluster (sorted)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs["silhouette_per_cluster_figure"] = "silhouette_per_cluster.png"

    # 3) similarity heatmap (sample × sample distance, reordered by cluster)
    # Subsample for very large sets to keep the figure readable & fast.
    n = X.shape[0]
    max_n = 800
    if n > max_n:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, size=max_n, replace=False)
        Xs = X[idx]
        ls = labels[idx]
    else:
        Xs = X
        ls = labels
    order = np.argsort(ls, kind="stable")
    D = squareform(pdist(Xs[order], metric="euclidean"))

    p = run_dir / "similarity_heatmap.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(D, cmap="magma_r", aspect="auto")
    ax.set_xlabel("sample (reordered by cluster)")
    ax.set_ylabel("sample (reordered by cluster)")
    ax.set_title(f"{title_prefix} — pairwise distance matrix\n(block-diagonal = tight clusters)")
    # cluster boundaries
    sorted_labels = ls[order]
    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 1
    for bnd in boundaries:
        ax.axhline(bnd - 0.5, color="white", lw=0.4, alpha=0.6)
        ax.axvline(bnd - 0.5, color="white", lw=0.4, alpha=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs["similarity_heatmap_figure"] = "similarity_heatmap.png"

    # 4) centroid distance heatmap (k × k)
    cd = np.array(metrics["centroid_distance_matrix"])
    p = run_dir / "centroid_distance_heatmap.png"
    fig, ax = plt.subplots(figsize=(max(5, 0.3 * len(uniq) + 3),
                                    max(4, 0.3 * len(uniq) + 2)))
    im = ax.imshow(cd, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(uniq))); ax.set_yticks(range(len(uniq)))
    ax.set_xticklabels([str(int(c)) for c in uniq], fontsize=8)
    ax.set_yticklabels([str(int(c)) for c in uniq], fontsize=8)
    ax.set_xlabel("cluster"); ax.set_ylabel("cluster")
    ax.set_title(f"{title_prefix} — inter-centroid distance")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs["centroid_distance_heatmap_figure"] = "centroid_distance_heatmap.png"

    # 5) silhouette-by-k curve (only when KMeans swept K)
    if sweep:
        ks = sorted(int(k) for k in sweep["sil_by_k"].keys())
        ys = [sweep["sil_by_k"][k] for k in ks]
        p = run_dir / "silhouette_by_k.png"
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(ks, ys, marker="o", lw=1.5, color="#b85c6e")
        ax.axvline(sweep["best_k"], color="black", ls="--", lw=1,
                   label=f"best K = {sweep['best_k']}")
        ax.set_xlabel("K"); ax.set_ylabel("silhouette")
        ax.set_title(f"{title_prefix} — silhouette vs K")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figs["silhouette_sweep_figure"] = "silhouette_by_k.png"

    return figs


# ============================================================
# Internals: misc
# ============================================================

def _attach_labels(df_keep: pd.DataFrame, labels: np.ndarray,
                   *, method: str, feature_set: str) -> pd.DataFrame:
    """Attach a `cluster_<method>_<feature_set>` column to df_keep."""
    if len(df_keep) != len(labels):
        raise ValueError(f"df_keep has {len(df_keep)} rows but labels has {len(labels)}.")
    df = df_keep.copy()
    col = f"cluster_{method}_{feature_set}"
    df[col] = np.asarray(labels, dtype=np.int32)
    return df


def _write_json(path: Path, obj: Any) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def _safe_git_info() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode().strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.STDOUT
        ).decode().strip()
        dirty = bool(status)
    except Exception:
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _update_index(outputs_root: Path, manifest: Dict[str, Any]) -> None:
    """
    Append/update the run in outputs/clustering/index.json (idempotent).

    The website fetches this single file to discover all available runs.
    """
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    index_path = outputs_root / "index.json"

    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
    else:
        idx = {
            "schema_version": SCHEMA_VERSION,
            "methods": [],
            "feature_sets": [],
            "runs": [],
            "latest": {},
        }

    # Ensure method / feature_set entries exist
    methods = {m["id"]: m for m in idx.get("methods", [])}
    if manifest["method"] not in methods:
        idx.setdefault("methods", []).append({
            "id": manifest["method"], "label": manifest["method_label"],
        })
    feats = {f["id"]: f for f in idx.get("feature_sets", [])}
    if manifest["feature_set"] not in feats:
        idx.setdefault("feature_sets", []).append({
            "id": manifest["feature_set"], "label": manifest["feature_set_label"],
        })

    # Replace any existing entry with the same (method, feature_set, run_id)
    runs = [r for r in idx.get("runs", [])
            if not (r.get("method") == manifest["method"]
                    and r.get("feature_set") == manifest["feature_set"]
                    and r.get("run_id") == manifest["run_id"])]
    runs.append({
        "method": manifest["method"],
        "feature_set": manifest["feature_set"],
        "run_id": manifest["run_id"],
        "created_at": manifest["created_at"],
        "n_samples": manifest["summary"]["n_samples"],
        "n_clusters": manifest["summary"]["n_clusters"],
        "silhouette": manifest["summary"]["silhouette_overall"],
        "path": f"{manifest['method']}/{manifest['feature_set']}/runs/{manifest['run_id']}",
    })
    idx["runs"] = sorted(runs, key=lambda r: (r["method"], r["feature_set"], r["run_id"]))

    # Latest pointer per (method, feature_set)
    latest = idx.get("latest", {})
    latest.setdefault(manifest["method"], {})
    latest[manifest["method"]][manifest["feature_set"]] = manifest["run_id"]
    idx["latest"] = latest

    idx["updated_at"] = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    _write_json(index_path, idx)


# ============================================================
# Gap statistic (Tibshirani 2001) — optional K-selection diagnostic
# ============================================================
# References:
#   Tibshirani, R., Walther, G. & Hastie, T. (2001).
#       Estimating the number of clusters in a data set via the gap statistic.
#       JRSSB 63(2), 411–423.
#
# Idea: compare the observed within-cluster dispersion W_k (sum of pairwise
# distances within clusters / 2N_c, or equivalently KMeans inertia / 2N_c)
# to the expected W_k under a null distribution of uniform samples in the
# data's bounding box. The gap is log(E[W_k*]) − log(W_k_obs). Best K is
# the smallest K such that Gap(K) >= Gap(K+1) − s_{K+1}.

def _compute_gap_statistic(
    X: np.ndarray,
    k_range,
    *,
    n_refs: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Compute Tibshirani gap statistic across k_range.

    Returns
    -------
    Dict[k -> {gap, sk, w_obs, w_ref_mean}]
        gap     : log(E[W_k_ref]) − log(W_k_obs)
        sk      : std of log(W_k_ref) × sqrt(1 + 1/n_refs)
        w_obs   : observed KMeans inertia at k (proxy for W_k)
        w_ref_mean : geometric mean of reference inertias
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    mins = X.min(axis=0)
    spans = X.max(axis=0) - mins

    out: Dict[int, Dict[str, float]] = {}
    for k in k_range:
        k = int(k)
        # Observed
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)
        w_obs = float(max(km.inertia_, 1e-12))

        # Reference (uniform over bounding box, same shape)
        log_w_refs = np.empty(n_refs, dtype=np.float64)
        for b in range(n_refs):
            X_ref = mins + rng.uniform(size=(n, d)) * spans
            km_ref = KMeans(n_clusters=k, random_state=random_state + b + 1, n_init=max(3, n_init // 2)).fit(X_ref)
            log_w_refs[b] = np.log(max(km_ref.inertia_, 1e-12))

        gap = float(log_w_refs.mean() - np.log(w_obs))
        sk = float(log_w_refs.std() * np.sqrt(1.0 + 1.0 / n_refs))

        out[k] = {
            "gap":        gap,
            "sk":         sk,
            "w_obs":      float(w_obs),
            "w_ref_mean": float(np.exp(log_w_refs.mean())),
        }
        if verbose:
            print(f"  gap stat k={k:>3d}: gap={gap:+.3f}  sk={sk:.3f}")

    return out


def _save_gap_curve(path: Path, gap_by_k: Dict[int, Dict[str, float]],
                     *, title_prefix: str = ""):
    """Plot Gap(K) ± s_k as a line with error bars. The Tibshirani rule:
    best K = smallest K such that Gap(K) >= Gap(K+1) − s_{K+1}."""
    ks  = sorted(int(k) for k in gap_by_k.keys())
    gap = np.array([gap_by_k[k]["gap"] for k in ks])
    sk  = np.array([gap_by_k[k]["sk"]  for k in ks])

    # Best K by Tibshirani rule
    best_k = ks[-1]
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            best_k = ks[i]
            break

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(ks) + 4), 4))
    ax.errorbar(ks, gap, yerr=sk, fmt="o-", color="#3a72b8", ecolor="#9ab8d6",
                capsize=3, lw=1.5, ms=6, label="Gap(k) ± s_k")
    ax.axvline(best_k, color="black", ls="--", lw=1,
               label=f"best k = {best_k} (Tibshirani rule)")
    ax.set_xlabel("k")
    ax.set_ylabel("Gap statistic (higher = better)")
    ax.set_title(f"{title_prefix} — gap statistic" if title_prefix else "Gap statistic")
    ax.set_xticks(ks)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
