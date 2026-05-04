"""
Smoke test for lf_cluster_run.

Run from the 02_FBM_Clustering directory:
    python _smoke_test_cluster_run.py

Generates synthetic well-separated clusters, exercises every public function,
and confirms all expected artifacts land on disk. Doesn't touch real data.

Delete this file once you're confident the module works.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ensure functions/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

from functions import lf_cluster_run as R  # noqa: E402


def _check(cond, msg):
    if cond:
        print(f"  ✓ {msg}")
    else:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)


def main() -> None:
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.normal(0, 1, (60, 8)),
        rng.normal(6, 1, (60, 8)),
        rng.normal(-6, 1, (60, 8)),
    ])
    df_keep = pd.DataFrame({
        "sample_idx": np.arange(len(X)),
        "patient_id": ["EL030"] * 60 + ["PAT_2868"] * 60 + ["MicroEPI-G-06"] * 60,
        "condition":  ["audio", "picture", "reading"] * 60,
    })

    tmp = Path(tempfile.mkdtemp(prefix="lf_cluster_test_"))
    print(f"[smoke test] tmp dir: {tmp}\n")

    try:
        # ------------------------------------------------------------ KMeans
        print("[1/4] fit_and_save — KMeans with k_range sweep")
        m1 = R.fit_and_save(
            X, df_keep=df_keep,
            method="kmeans", feature_set="raw",
            params={"k_range": [2, 3, 4, 5], "random_state": 42},
            outputs_root=tmp, verbose=False,
        )
        run1 = tmp / m1["artifacts"]["model"].rsplit("/", 1)[0] if "/" in m1["artifacts"]["model"] \
               else tmp / "kmeans" / "raw" / "runs" / m1["run_id"]
        _check(run1.exists(), f"run dir created: {run1}")
        for k in ["model", "predictor", "X_train", "feature_schema",
                  "labels", "metrics", "df_keep_with_clusters",
                  "centroids_figure", "silhouette_per_cluster_figure",
                  "similarity_heatmap_figure", "centroid_distance_heatmap_figure",
                  "silhouette_sweep", "silhouette_sweep_figure"]:
            _check(k in m1["artifacts"], f"manifest has artifact key '{k}'")
            _check((run1 / m1["artifacts"][k]).exists(),
                   f"file exists: {m1['artifacts'][k]}")
        _check(m1["summary"]["n_clusters"] == 3,
               f"correct cluster count detected (got {m1['summary']['n_clusters']})")
        _check(m1["summary"]["silhouette_overall"] > 0.5,
               f"silhouette > 0.5 on well-separated synthetic data "
               f"(got {m1['summary']['silhouette_overall']:.3f})")
        _check(m1["predictor_type"] == "centroid",
               "kmeans predictor_type = centroid")

        # ----------------------------------------------------- Hierarchical
        print("\n[2/4] fit_and_save — Hierarchical (Ward)")
        m2 = R.fit_and_save(
            X, df_keep=df_keep,
            method="hierarchical", feature_set="raw",
            params={"linkage": "ward", "n_clusters": 3},
            outputs_root=tmp, verbose=False,
        )
        run2 = tmp / "hierarchical" / "raw" / "runs" / m2["run_id"]
        _check(run2.exists(), f"run dir created: {run2}")
        for k in ["model", "predictor", "X_train", "linkage_Z", "distance_D"]:
            _check(k in m2["artifacts"], f"manifest has artifact key '{k}'")
            _check((run2 / m2["artifacts"][k]).exists(),
                   f"file exists: {m2['artifacts'][k]}")
        _check(m2["predictor_type"] == "1-NN",
               "hierarchical predictor_type = 1-NN")

        # ----------------------------------------------------- index.json
        print("\n[3/4] index.json updated correctly")
        import json
        with open(tmp / "index.json") as f:
            idx = json.load(f)
        _check(len(idx["runs"]) == 2, f"index has 2 runs (got {len(idx['runs'])})")
        _check(idx["latest"]["kmeans"]["raw"] == m1["run_id"],
               "kmeans/raw latest pointer correct")
        _check(idx["latest"]["hierarchical"]["raw"] == m2["run_id"],
               "hierarchical/raw latest pointer correct")
        _check({m["id"] for m in idx["methods"]} == {"kmeans", "hierarchical"},
               "both methods registered")
        _check({f["id"] for f in idx["feature_sets"]} == {"raw"},
               "feature_set 'raw' registered")

        # ------------------------------------------------- load + assign_new
        print("\n[4/4] load_run + assign_new (KMeans + Hierarchical)")
        b1 = R.load_run("kmeans", "raw", outputs_root=tmp)
        _check(set(b1.keys()) >= {"manifest", "model", "predictor",
                                  "X_train", "labels", "run_dir"},
               "kmeans bundle has expected keys")

        # New points near each centroid: should land in 3 distinct clusters
        X_new = np.vstack([
            rng.normal(0, 0.5, (3, 8)),
            rng.normal(6, 0.5, (3, 8)),
            rng.normal(-6, 0.5, (3, 8)),
        ])
        out1 = R.assign_new(b1, X_new)
        _check(out1["labels"].shape == (9,), "9 new labels returned")
        _check(len(np.unique(out1["labels"])) == 3,
               "new points partition into 3 clusters")
        _check(np.all(out1["confidence"] > 0.3),
               f"confidence > 0.3 for clean new points "
               f"(min {out1['confidence'].min():.3f})")
        _check(out1["predictor_type"] == "centroid",
               "kmeans assign_new uses native predict")

        b2 = R.load_run("hierarchical", "raw", outputs_root=tmp)
        out2 = R.assign_new(b2, X_new)
        _check(len(np.unique(out2["labels"])) == 3,
               "hierarchical new points partition into 3 clusters")
        _check(out2["predictor_type"] == "1-NN",
               "hierarchical assign_new uses 1-NN")

        # latest=run_id resolution
        b1b = R.load_run("kmeans", "raw", run_id="latest", outputs_root=tmp)
        _check(b1b["manifest"]["run_id"] == m1["run_id"],
               "load_run(run_id='latest') resolves correctly")

        print("\n[smoke test] ALL CHECKS PASSED ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
