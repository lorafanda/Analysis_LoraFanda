#!/usr/bin/env python3
"""
make_centroid_bundle.py - what cluster_visualizer.html needs to draw a LIVE centroid.

    python make_centroid_bundle.py

The page's bottom-left panel shows the centroid of the selected cluster at the selected
K, recomputed in the browser from the electrodes whose loading on that cluster is at or
above the minimum-loading slider, weighted by that loading. Nothing is precomputed per
threshold: the page gets the ingredients and does the arithmetic (a few milliseconds).

WHAT IS SHIPPED, to outputs/clustering/paper_web/centroids/:

  x_<feature set>.bin      the feature matrix, int16, value x 1000 (dB), laid out as
                           (n, ncond, nband, nt) - already in the figures' cube order,
                           so the page never has to know the stored axis is band-major.
                           ONE PER FEATURE SET: every method's run on a feature set was
                           fitted to the same matrix, which is checked byte for byte
                           here and refused if it ever stops being true.
  <run id>.w.bin           graded runs only: the loadings at every K of the sweep,
                           uint16 x 65535, one (n, K) block per K, offsets in the json.
                           Convex NMF's G row-normalised to sum to 1 - the same
                           normalisation FIG 1 and FIG 2 use.
  <run id>.json            shape, names, scales, offsets, and the hard labels at every
                           K (argmax for graded runs, the run's own cut for k-means and
                           Ward), so the panel can also show the paper's definition -
                           the plain mean over argmax members - and count members.

Every file is read back after writing and compared byte for byte; the share has
truncated writes before.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLUST = ROOT / "outputs" / "clustering"
MANIFEST = ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz" / "manifest.json"
OUT = CLUST / "paper_web" / "centroids"

FSETS = ["concat_bands5z", "concat_hg", "concat_rawds", "concat_bands5"]
METHODS = ["cnmf", "kmeans", "hierarchical"]          # archetypal analysis dropped 2026-09-06
COHORT = "cohort1_n27"
X_SCALE = 0.001          # int16 -> dB
W_SCALE = 1.0 / 65535    # uint16 -> loading
# what a value IS, per feature set - the page labels its colour bar and y axis with it.
# bands5z is a per-band z-score, so its values are standard deviations, not decibels.
UNIT = {"concat_hg": "dB", "concat_rawds": "dB", "concat_bands5": "dB",
        "concat_bands5z": "z"}
UNIT_LONG = {"concat_hg": "dB vs baseline", "concat_rawds": "dB vs baseline",
             "concat_bands5": "dB vs baseline", "concat_bands5z": "SD (per-band z-score)"}


def verified_write(path: Path, data: bytes, tries: int = 6):
    path.parent.mkdir(parents=True, exist_ok=True)
    want = hashlib.sha256(data).hexdigest()
    for i in range(tries):
        try:
            path.write_bytes(data)
            if hashlib.sha256(path.read_bytes()).hexdigest() == want:
                return
        except OSError as e:
            print(f"    write failed ({e}); retrying")
        time.sleep(1.5 * (i + 1))
    raise SystemExit(f"could not write {path} intact after {tries} tries")


def paper_runs(manifest):
    by = {}
    for r in manifest["runs"]:
        parts = r["id"].split("__")
        if len(parts) != 3 or r.get("cohort_id") != COHORT:
            continue
        method, fset, stamp = parts
        if method in METHODS and fset in FSETS:
            by[(method, fset)] = (r["id"], stamp)
    return [(m, f) + by[(m, f)] for m in METHODS for f in FSETS if (m, f) in by]


def schema(run: Path):
    feats = json.loads((run / "feature_schema.json").read_text())["feature_names"]
    parts = [f.split("|") for f in feats]
    conds = list(dict.fromkeys(p[0] for p in parts))
    bands = list(dict.fromkeys(p[1] for p in parts))
    nt = len(feats) // (len(conds) * len(bands))
    # the stored axis is band-major (band, cond, time) - assert it, as load_run() does
    exp = [f"{c}|{b}|" for b in bands for c in conds]
    got = [f"{p[0]}|{p[1]}|" for p in parts[::nt]]
    if exp != got:
        raise SystemExit(f"{run}: feature order is not band x cond x time")
    return conds, bands, nt


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runs = paper_runs(manifest)
    ks = [int(k) for k in manifest.get("sweep_ks", range(5, 13))]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- one feature matrix per feature set, proven identical across its runs -----
    xhash = {}
    for fset in FSETS:
        dirs = [CLUST / m / fset / "runs" / stamp for m, f, rid, stamp in runs if f == fset]
        if not dirs:
            continue
        X0 = np.load(dirs[0] / "X_train.npy")
        for d in dirs[1:]:
            X = np.load(d / "X_train.npy")
            if X.shape != X0.shape or not np.array_equal(X, X0):
                raise SystemExit(f"{fset}: {d.name} was not fitted to the same matrix "
                                 f"as {dirs[0].name} - one file per run would be needed")
        conds, bands, nt = schema(dirs[0])
        n = len(X0)
        cube = X0.reshape(n, len(bands), len(conds), nt).transpose(0, 2, 1, 3)
        if np.abs(cube).max() * (1 / X_SCALE) > 32000:
            raise SystemExit(f"{fset}: values exceed the int16 range at scale {X_SCALE}")
        q = np.round(cube / X_SCALE).astype("<i2")
        verified_write(OUT / f"x_{fset}.bin", q.tobytes())
        xhash[fset] = dict(n=n, conds=conds, bands=bands, nt=nt,
                           unit=UNIT.get(fset, "dB"), unit_long=UNIT_LONG.get(fset, ""),
                           sha=hashlib.sha256(q.tobytes()).hexdigest()[:12])
        print(f"  x_{fset}.bin  {q.nbytes/1e6:.1f} MB  (n={n}, {len(conds)} conds x "
              f"{len(bands)} bands x {nt} bins; max |err| {X_SCALE/2:.4f} dB)")

    # ---- per run: loadings (graded) and labels (all) at every K ---------------------
    index = []
    for method, fset, rid, stamp in runs:
        run = CLUST / method / fset / "runs" / stamp
        meta = xhash[fset]
        n = meta["n"]
        graded = method == "cnmf"
        labels, offsets, blobs, have = {}, {}, [], []
        off = 0
        lab_csv = run / "cluster_labels_by_k.csv"
        hard = pd.read_csv(lab_csv) if lab_csv.exists() else None
        for k in ks:
            if graded:
                f = run / "loadings_by_k" / (f"G_k{k:02d}.npy" if method == "cnmf"
                                             else f"A_k{k:02d}.npy")
                if not f.exists():
                    continue
                W = np.load(f).astype(float)
                if W.shape != (n, k):
                    raise SystemExit(f"{rid}: {f.name} is {W.shape}, expected {(n, k)}")
                if method == "cnmf":
                    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
                q = np.round(np.clip(W, 0, 1) / W_SCALE).astype("<u2")
                blobs.append(q.tobytes())
                offsets[str(k)] = off
                off += q.size
                labels[str(k)] = [int(x) for x in W.argmax(1)]
            else:
                if hard is None or f"k_{k}" not in hard.columns:
                    continue
                lab = hard[f"k_{k}"].to_numpy().astype(int)
                if len(lab) != n:
                    raise SystemExit(f"{rid}: {len(lab)} labels, expected {n}")
                labels[str(k)] = [int(x) for x in lab]
            have.append(k)
        if not have:
            print(f"  {rid}: no cut in {ks}; skipped")
            continue
        if graded:
            verified_write(OUT / f"{rid}.w.bin", b"".join(blobs))
        j = dict(id=rid, method=method, feature_set=fset, run=stamp, kind="graded" if graded
                 else "hard", n=n, ncond=len(meta["conds"]), nband=len(meta["bands"]),
                 nt=meta["nt"], conds=meta["conds"], bands=meta["bands"], ks=have,
                 x_file=f"x_{fset}.bin", x_scale=X_SCALE, x_sha=meta["sha"],
                 w_file=f"{rid}.w.bin" if graded else None, w_scale=W_SCALE,
                 w_offsets=offsets, labels=labels,
                 normalisation=("convex NMF G, row-normalised to sum to 1" if graded
                                else "hard partition; no loadings"),
                 generated=dt.date.today().isoformat())
        verified_write(OUT / f"{rid}.json", json.dumps(j).encode("utf-8"))
        index.append(dict(id=rid, kind=j["kind"], ks=have))
        print(f"  {rid:<46} {j['kind']:<6} K {have[0]}-{have[-1]}"
              + (f"  w.bin {off*2/1e6:.1f} MB" if graded else ""))

    # ---- ONE ELECTRODE ORDER ACROSS FEATURE SETS, proven, so a clustering found on one
    # representation can be averaged over another's matrix. The page offers the "shown
    # on" selector only if this says so.
    def norm(s):
        return str(s).replace("_", "").replace("-", "").upper()
    row_keys = {}
    for method, fset, rid, stamp in runs:
        lab = pd.read_csv(CLUST / method / fset / "runs" / stamp / "labels.csv")
        row_keys[rid] = [f"{p}|{norm(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
    ref_id = runs[0][2]
    aligned = all(v == row_keys[ref_id] for v in row_keys.values())
    key_sha = hashlib.sha256("\n".join(row_keys[ref_id]).encode()).hexdigest()[:12]
    print(f"  electrode order identical across all {len(row_keys)} runs: {aligned}  "
          f"(key sha {key_sha})")

    verified_write(OUT / "index.json", json.dumps(dict(
        generated=dt.date.today().isoformat(), x_scale=X_SCALE, w_scale=W_SCALE,
        rows_aligned=bool(aligned), row_key_sha=key_sha,
        feature_sets={f: {k: v for k, v in m.items() if k != "sha"} for f, m in xhash.items()},
        runs=index)).encode("utf-8"))
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"  -> {OUT}  ({len(index)} runs, {total/1e6:.1f} MB in all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
