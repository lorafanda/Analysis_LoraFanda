#!/usr/bin/env python3
"""
run_archetypes.py - archetypal analysis as a first-class track, next to cnmf.

Writes a run directory in exactly the layout the rest of the pipeline expects, so the
statistics, the figures and the visualizer can read an archetype run without any of them
learning a new format.

THE COHORT IS TAKEN, NOT REFITTED. X_train.npy is copied byte-for-byte from the run
named by --source (the k-means run the decomposition is derived from), so an archetype
run and the convex-NMF run it is compared against are guaranteed to hold the same
electrodes in the same order. Refitting the dataset here would have been one more place
for the cohort to drift.

    python run_archetypes.py --feature-set concat_hg
    python run_archetypes.py --feature-set concat_hg concat_rawds --ks 5 6 7 8 9 10 11 12
    python run_archetypes.py --feature-set concat_hg --k 11 --n-iter 2000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_archetypes as AA  # noqa: E402

CLUST = ROOT / "outputs" / "clustering"
FS_LABEL = {"concat_hg": "Concatenated HG [a|p|r]",
            "concat_rawds": "Concatenated 15 bands x time [a|p|r]",
            "concat_bands5": "Concatenated 5 bands x time [a|p|r]",
            "concat_hg_all": "Concatenated HG, gate lifted"}
DEFAULT_KS = list(range(5, 31))


def newest(method: str, fset: str):
    base = CLUST / method / fset / "runs"
    if not base.is_dir():
        return None
    runs = sorted(d for d in base.iterdir() if d.is_dir() and (d / "X_train.npy").exists())
    return runs[-1] if runs else None


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def fit_one(Xu, k, n_iter, seed):
    A, B, Z, info = AA.archetypal_analysis(Xu, k, n_iter=n_iter, random_state=seed)
    # A's rows ALREADY sum to 1 - that is the constraint, not a post-hoc
    # normalisation. Convex NMF's G does not, and has to be divided by its row sum
    # before an argmax means anything; here the weights are proportions as fitted.
    lab = A.argmax(1)
    srt = np.sort(A, axis=1)
    return A, B, Z, info, lab, srt[:, -1], srt[:, -1] - srt[:, -2]


def run(fset: str, ks, kpub: int, n_iter: int, seed: int, source: Path) -> Path:
    X = np.load(source / "X_train.npy").astype(float)
    Xu = unit(X)
    n, p = X.shape
    print(f"\n=== {fset}: {n} electrodes x {p} features, from {source.name}")

    rid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rd = CLUST / "archetypes" / fset / "runs" / rid
    (rd / "loadings_by_k").mkdir(parents=True, exist_ok=True)
    (rd / "components_by_k").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "X_train.npy", rd / "X_train.npy")
    for f in ("feature_schema.json",):
        if (source / f).exists():
            shutil.copy2(source / f, rd / f)

    rows, sweep, pub = {}, [], None
    for k in ks:
        t = time.time()
        A, B, Z, info, lab, top, marg = fit_one(Xu, k, n_iter, seed)
        np.save(rd / "loadings_by_k" / f"A_k{k:02d}.npy", A.astype(np.float32))
        np.save(rd / "components_by_k" / f"C_k{k:02d}.npy", Z.astype(np.float32))
        rows[f"k_{k}"] = lab
        es = np.asarray(info["effective_support"])
        sweep.append(dict(k=k, var_explained=info["var_explained"], rss=info["rss"],
                          n_iter_used=info["n_iter"], converged=info["converged"],
                          median_top_weight=float(np.median(top)),
                          frac_no_majority=float((top < 0.5).mean()),
                          min_effective_support=float(es.min()),
                          median_effective_support=float(np.median(es)),
                          seconds=time.time() - t))
        flag = "" if info["converged"] else "   <-- HIT THE ITERATION CAP"
        print(f"  K={k:<3d} var={info['var_explained']:.4f}  "
              f"it={info['n_iter']:<5d} support>={es.min():.1f}  "
              f"{time.time()-t:5.1f}s{flag}")
        if k == kpub:
            pub = (A, B, Z, info, lab, top, marg)

    if pub is None:
        raise SystemExit(f"published K={kpub} was not in --ks")
    A, B, Z, info, lab, top, marg = pub

    np.save(rd / "A_loadings.npy", A.astype(np.float32))
    np.save(rd / "B_weights.npy", B.astype(np.float32))
    np.save(rd / "components.npy", Z.astype(np.float32))
    pd.DataFrame(rows).to_csv(rd / "cluster_labels_by_k.csv", index=False)
    sw = pd.DataFrame(sweep)
    sw.to_csv(rd / "sweep_by_k.csv", index=False)

    # labels.csv keeps the source run's identifying columns so a row still knows which
    # patient and contact it is - the whole downstream chain joins on them.
    src_lab = pd.read_csv(source / "labels.csv")
    # "silhouette" is the SOURCE method's score. Carrying it into an archetype run
    # would leave a k-means number in a column nothing here computed, under a name
    # something downstream would eventually read as this run's own.
    keep = [c for c in src_lab.columns if not c.startswith(("cluster_", "w"))
            and c not in ("top_weight", "margin", "silhouette")]
    out = src_lab[keep].copy()
    out[f"cluster_archetypes_{fset}"] = lab
    for j in range(kpub):
        out[f"w{j}"] = A[:, j]
    out["top_weight"] = top
    out["margin"] = marg
    out.to_csv(rd / "labels.csv", index=False)

    man = dict(
        schema_version=1, method="archetypes",
        method_label="Archetypal analysis (graded, on the hull)",
        feature_set=fset, feature_set_label=FS_LABEL.get(fset, fset),
        run_id=rid, created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        params=dict(k=kpub, n_iter=n_iter, random_state=seed,
                    preprocessing="unit-norm per electrode",
                    algorithm="archetypal analysis (Cutler & Breiman 1994), "
                              "projected gradient with exact simplex projection "
                              "(Duchi et al. 2008), FurthestSum init "
                              "(Morup & Hansen 2012)"),
        summary=dict(n_samples=int(n), n_features=int(p), n_clusters=int(kpub),
                     best_k=int(kpub),
                     variance_explained=float(info["var_explained"]),
                     converged=bool(info["converged"]),
                     n_iter_used=int(info["n_iter"]),
                     frac_no_majority=float((top < 0.5).mean()),
                     median_top_weight=float(np.median(top)),
                     min_effective_support=float(np.min(info["effective_support"]))),
        predictor_type="loadings",
        artifacts=dict(labels="labels.csv", X_train="X_train.npy",
                       components="components.npy", loadings="A_loadings.npy",
                       archetype_weights="B_weights.npy"),
        notebook="243_cluster_archetypes.ipynb",
        note=("Graded decomposition on the CONVEX HULL. Each row of A sums to 1, so "
              "w0..wK-1 ARE proportions as fitted and need no renormalising - unlike "
              "convex NMF's G. The cluster column is the argmax and is a lossy summary; "
              "an archetype is an EXTREME response, not a cluster centre, so few "
              "electrodes should sit near one and a low median top_weight is expected "
              "rather than a fault."),
        derived_from=str(source.relative_to(CLUST)).replace("\\", "/"),
        sweep=dict(ks=[int(k) for k in ks],
                   labels="cluster_labels_by_k.csv", stats="sweep_by_k.csv",
                   loadings="loadings_by_k"),
    )
    (rd / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  -> {rd}")
    return rd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", nargs="+",
                    default=["concat_hg", "concat_rawds", "concat_bands5"])
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    ap.add_argument("--k", type=int, default=None,
                    help="the published K; default = this feature set's cNMF peak")
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default=None, help="run dir to take X_train from")
    a = ap.parse_args()

    peaks = {}
    pk = CLUST / "bsf_comparison" / "peak_k.json"
    if pk.exists():
        peaks = {k: int(v) for k, v in json.loads(pk.read_text()).items()}

    for fs in a.feature_set:
        src = Path(a.source) if a.source else newest("kmeans", fs)
        if src is None:
            print(f"  !! no source run for {fs}", file=sys.stderr)
            continue
        kpub = a.k or peaks.get(fs)
        if kpub is None:
            print(f"  !! no published K for {fs} (no peak_k.json, pass --k)",
                  file=sys.stderr)
            continue
        ks = sorted(set(list(a.ks) + [kpub]))
        run(fs, ks, kpub, a.n_iter, a.seed, src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
