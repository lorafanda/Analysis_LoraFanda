#!/usr/bin/env python3
"""
make_coverage_bundle.py — per-vertex sampling counts for clustering_visualizer.html.

The question the page answers: at this point on the cortex, HOW WELL DID WE SAMPLE, and
of the electrodes that are here, what fraction belong to a given cluster? Every brain
figure in this project carries the caveat "dot density reflects where electrodes were
implanted, not where language is" — this makes that checkable instead of a footnote.

Why not reuse the activity-visualizer projection: that one keeps the k=4 NEAREST
contacts per vertex, so a patient count taken from it can never exceed 4 of 35 and is
useless as a probability. Here every contact within RADIUS_MM of a vertex is counted,
which is what makes "fraction of patients sampled here" meaningful.

Per vertex, per hemisphere, this writes:

    n_contacts   how many contacts sit within RADIUS_MM
    n_patients   how many DISTINCT patients contributed one       <- the honest denominator
    n_cluster_k  how many of those contacts belong to cluster k

so the page can compute both maps as plain ratios client-side:

    coverage   P(sampled)          = n_patients / n_patients_total
    cluster k  P(cluster k | here) = n_cluster_k / n_contacts

Counts, not probabilities, are shipped on purpose: the ratio you want depends on the
question, and a stored ratio hides its own denominator.

    python make_coverage_bundle.py
    python make_coverage_bundle.py --radius 8 --run <clustering run dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FS = ROOT / "outputs" / "250_recon" / "fsaverage"
MESHES = FS / "meshes"
COORDS = FS / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv"
OUT = FS / "coverage_viz"
DEFAULT_TRACK = ROOT / "outputs" / "clustering" / "kmeans" / "concat_hg" / "runs"
RADIUS_MM = 10.0


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=RADIUS_MM)
    ap.add_argument("--run", default=None, help="clustering run dir (default: newest concat_hg)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import nibabel as nib
    from scipy.spatial import cKDTree

    run = Path(a.run) if a.run else sorted(
        (d for d in DEFAULT_TRACK.iterdir() if d.is_dir()), key=lambda d: d.name)[-1]
    lab = pd.read_csv(run / "labels.csv")
    ccol = next(c for c in lab.columns
                if c.startswith("cluster_") and not c.endswith("_ranked"))
    lab["key"] = [f"{p}|{norm(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
    clu_of = dict(zip(lab["key"], lab[ccol]))
    clusters = sorted(int(c) for c in lab[ccol].unique())
    print(f"  run {run.name}: {len(lab)} electrodes, K={len(clusters)}")

    co = pd.read_csv(COORDS).dropna(subset=["x", "y", "z"])
    co["key"] = [f"{p}|{norm(n)}" for p, n in zip(co["patient"], co["name"])]
    co["cluster"] = co["key"].map(clu_of)
    n_pat_total = int(co["patient"].nunique())
    matched = int(co["cluster"].notna().sum())
    print(f"  coords {len(co)} contacts / {n_pat_total} patients; "
          f"{matched} carry a cluster label from this run")

    pat_codes = {p: i for i, p in enumerate(sorted(co["patient"].unique()))}
    manifest = {
        "generated_from": {"run": run.name, "coords": COORDS.name},
        "radius_mm": a.radius,
        "n_patients_total": n_pat_total,
        "n_contacts_total": int(len(co)),
        "n_contacts_with_cluster": matched,
        "clusters": clusters,
        "cluster_sizes": {str(k): int((lab[ccol] == k).sum()) for k in clusters},
        "fields": ["n_contacts", "n_patients"] + [f"n_cluster_{k}" for k in clusters],
        "dtype": "uint16",
        "hemis": {},
        "note": ("counts within radius_mm of each vertex; ratios are computed client-side "
                 "so the denominator stays visible"),
    }

    n_fields = 2 + len(clusters)
    for h, tag in (("lh", "L"), ("rh", "R")):
        verts = nib.load(str(MESHES / f"fsaverage_{h}.gii")).darrays[0].data.astype(float)
        sub = co[co["hemi"].astype(str).str.upper().str[0] == tag].reset_index(drop=True)
        tree = cKDTree(sub[["x", "y", "z"]].to_numpy())
        near = tree.query_ball_point(verts, r=a.radius)

        arr = np.zeros((len(verts), n_fields), dtype=np.uint16)
        pcodes = sub["patient"].map(pat_codes).to_numpy()
        cl = sub["cluster"].to_numpy()
        cidx = {k: i for i, k in enumerate(clusters)}
        for v, hits in enumerate(near):
            if not hits:
                continue
            arr[v, 0] = len(hits)
            arr[v, 1] = len(set(pcodes[hits]))
            for i in hits:
                c = cl[i]
                if c == c:                       # not NaN
                    arr[v, 2 + cidx[int(c)]] += 1
        f = OUT / f"counts_{h}_u16.bin"
        arr.tofile(f)
        cov = 100 * float((arr[:, 1] > 0).mean())
        manifest["hemis"][h] = {
            "nvert": int(len(verts)), "file": f.name,
            "n_contacts_hemi": int(len(sub)),
            "pct_vertices_sampled": round(cov, 1),
            "max_patients_at_a_vertex": int(arr[:, 1].max()),
        }
        print(f"  {h}: {len(verts)} verts, {len(sub)} contacts | "
              f"{cov:.1f}% of vertices within {a.radius:g} mm of a contact | "
              f"max {arr[:, 1].max()} patients at one vertex | {f.stat().st_size/1e6:.1f} MB")

    # contact list for the electrode overlay (small: xyz + hemi + cluster)
    cj = [{"x": round(float(r.x), 2), "y": round(float(r.y), 2), "z": round(float(r.z), 2),
           "hemi": "lh" if str(r.hemi).upper().startswith("L") else "rh",
           "cluster": (None if r.cluster != r.cluster else int(r.cluster))}
          for r in co.itertuples()]
    (OUT / "contacts.json").write_text(json.dumps(cj), encoding="utf-8")
    manifest["contacts_file"] = "contacts.json"
    manifest["meshes"] = {"lh": "../meshes/fsaverage_lh.gii",
                          "rh": "../meshes/fsaverage_rh.gii",
                          "lh_inflated": "../meshes/fsaverage_lh.inflated.gii",
                          "rh_inflated": "../meshes/fsaverage_rh.inflated.gii"}
    print(f"  contacts.json: {len(cj)} contacts")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
