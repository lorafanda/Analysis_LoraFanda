#!/usr/bin/env python3
"""
publish_decomposition.py — write the K=7 convex-NMF decomposition as a run both tools read.

MOBA and clustering_visualizer both expect a HARD LABEL per electrode. The decomposition
does not have one: 59% of electrodes have no component above half their total weight. So
the published label is the argmax, and that is a lossy summary of the thing it summarises.
The full loadings ride along in labels.csv as w0..w6 plus `top_weight` and `margin`, so
nothing is thrown away and the uncertainty stays inspectable.

Read the argmax map as "which component leads here", never as "which type this electrode
is". For most electrodes the lead is narrow.

    python publish_decomposition.py            # write the run + register it
    python publish_decomposition.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEC = ROOT / "outputs" / "clustering" / "decomposition"
CLUST = ROOT / "outputs" / "clustering"
METHOD = "cnmf"
METHOD_LABEL = "Convex NMF (graded)"
FSET_LABELS = {
    "concat_hg": "Concatenated HG [a|p|r]",
    "concat_rawds": "Concatenated raw-ds [a|p|r]",
    # --feature-set is argparse `choices=sorted(FSET_LABELS)`, so a set missing from
    # this dict is REJECTED at the command line rather than defaulted - which is the
    # right behaviour, and the reason this entry has to exist.
    "concat_bands5": "Concatenated 5 bands [a|p|r]",
    "concat_hg_all": "Concatenated HG, ungated [a|p|r]",
}
# SRC is no longer hard-coded. It is read from the decomposition's own meta.json,
# because pinning it to one run is what let the loadings and the labels drift onto
# different cohorts (1266 weights against a 1027-row labels.csv).


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--feature-set", default="concat_hg", choices=sorted(FSET_LABELS))
    a = ap.parse_args()

    FSET = a.feature_set
    FSET_LABEL = FSET_LABELS[FSET]
    dec = DEC / FSET
    if not (dec / "meta.json").exists():
        print(f"  !! no decomposition at {dec} - run run_decomposition.py first",
              file=sys.stderr)
        return 1
    meta_in = json.loads((dec / "meta.json").read_text(encoding="utf-8"))
    SRC = ROOT / meta_in["source_run"]

    G = np.load(dec / "G_loadings.npy")
    C = np.load(dec / "components.npy")
    K = G.shape[1]
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    lab = pd.read_csv(SRC / "labels.csv")
    if len(lab) != len(Gn):
        print(f"  !! {len(lab)} label rows vs {len(Gn)} loadings — refusing", file=sys.stderr)
        return 1

    srt = np.sort(Gn, axis=1)
    top, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    ccol = f"cluster_{METHOD}_{FSET}"

    out = lab.drop(columns=[c for c in lab.columns if c.startswith("cluster_")
                            or c == "silhouette"], errors="ignore").copy()
    out[ccol] = Gn.argmax(1).astype(int)
    for j in range(K):
        out[f"w{j}"] = Gn[:, j]
    out["top_weight"] = top
    out["margin"] = margin

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rd = CLUST / METHOD / FSET / "runs" / run_id
    sizes = np.bincount(out[ccol], minlength=K)
    print(f"  run {METHOD}/{FSET}/{run_id}")
    print(f"  K={K}  sizes={list(sizes)}")
    print(f"  argmax is a NARROW lead for most electrodes: "
          f"{100*(top < 0.5).mean():.0f}% have no majority, median top weight {np.median(top):.2f}")
    if a.dry_run:
        print("  (dry run — nothing written)")
        return 0

    rd.mkdir(parents=True, exist_ok=True)
    out.to_csv(rd / "labels.csv", index=False)
    np.save(rd / "X_train.npy", np.load(SRC / "X_train.npy"))
    np.save(rd / "components.npy", C)
    np.save(rd / "G_loadings.npy", G)
    for f in ("feature_schema.json",):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, rd / f)

    manifest = {
        "schema_version": 1,
        "method": METHOD, "method_label": METHOD_LABEL,
        "feature_set": FSET, "feature_set_label": FSET_LABEL,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {"k": K, "n_iter": 300, "random_state": 0,
                   "preprocessing": "unit-norm per electrode",
                   "algorithm": "convex NMF (Ding, Li & Jordan 2010)"},
        "summary": {
            "n_samples": int(len(out)), "n_features": int(C.shape[1]),
            "n_clusters": K, "best_k": K,
            "silhouette_overall": None,
            "held_out_variance_explained": meta_in.get("in_sample_var_explained"),
            "frac_no_majority": float((top < 0.5).mean()),
            "frac_dominant": float((top >= 0.8).mean()),
            "median_top_weight": float(np.median(top)),
        },
        "predictor_type": "loadings",
        "artifacts": {"labels": "labels.csv", "X_train": "X_train.npy",
                      "components": "components.npy", "loadings": "G_loadings.npy"},
        "notebook": "235_concat_decomposition.ipynb",
        "note": ("Graded decomposition. The cluster column is the ARGMAX of the loadings and is "
                 "a lossy summary: most electrodes have no majority component. Use w0..w{} plus "
                 "top_weight and margin for the real membership.".format(K - 1)),
        "derived_from": str(SRC.relative_to(ROOT)).replace("\\", "/"),
    }
    (rd / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ── register in index.json, the file MOBA reads
    ip = CLUST / "index.json"
    idx = json.loads(ip.read_text(encoding="utf-8"))
    if not any(m["id"] == METHOD for m in idx["methods"]):
        idx["methods"].append({"id": METHOD, "label": METHOD_LABEL})
        print(f"  registered method '{METHOD}'")
    if not any(f["id"] == FSET for f in idx["feature_sets"]):
        idx["feature_sets"].append({"id": FSET, "label": FSET_LABEL})
    idx["runs"] = [r for r in idx["runs"]
                   if not (r.get("method") == METHOD and r.get("feature_set") == FSET)]
    idx["runs"].append({
        "method": METHOD, "feature_set": FSET, "run_id": run_id,
        "created_at": manifest["created_at"],
        "n_samples": int(len(out)), "n_clusters": K, "silhouette": None,
        "path": f"{METHOD}/{FSET}/runs/{run_id}", "has_ranking": False,
    })
    idx.setdefault("latest", {}).setdefault(METHOD, {})[FSET] = run_id
    idx["updated_at"] = manifest["created_at"]
    ip.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    print(f"  index.json now lists {len(idx['runs'])} runs")

    print(f"\n  -> {rd}")
    print("  STILL REQUIRED, and silent if skipped:")
    print(f"    1. notebook 252 with RUN_FILTER = "
          f"[{{'feature_set': '{FSET}', 'method': 'cnmf'}}]")
    print("       — fit_and_save never writes recon/, so MOBA's 3-D brain stays empty without it")
    print(f"    2. append (\"{METHOD}/{FSET}\", \"Convex NMF - graded\", "
          f"\"{len(out)}-electrode concat cohort\") to TRACKS in make_coverage_bundle.py")
    print("       then regenerate the whole bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
