#!/usr/bin/env python3
"""
sweep_stability.py - subsample stability for every K in a run's sweep.

WHY
stability_summary.json is written once, at the run's published K. The K control in the
visualizer and the report can re-cut a run to any K in its sweep, so a results table
that quotes stability would keep showing the published K's number under a different K's
heading - a number silently changing meaning, which is worse than no number.

This computes the same statistic at every K in cluster_labels_by_k.csv and writes it
per K, plus a roll-up the report reads directly.

WHAT IT WRITES, into <run>/

    stability_by_k/k_07/stability_summary.json     same shape as the run's own
    stability_by_k/k_07/per_cluster_stability.csv
    stability_by_k/k_07/consensus_heatmap.png
    stability_by_k.csv                             k, mean/min/max jaccard, n_runs

ON METHOD - read this before quoting the cnmf numbers.
lf_stability.compute_consensus_matrix resamples with KMeans whatever run it is pointed
at. That is exact for the k-means and Ward runs. For a convex-NMF run it measures how
reproducibly *k-means* partitions the space the decomposition was fitted in, which is
not the same thing as how reproducible the decomposition is. It is used here anyway,
deliberately: the stability_summary.json already published on the cnmf runs was
produced by that same function through 211, so matching it keeps K=7 agreeing with the
number already on the site. Changing the estimator would silently move a published
figure. If you want a genuinely cNMF-native stability, measure_cluster_stability.py
refits the decomposition itself - that is a different statistic and belongs in its own
column, not in place of this one.

    python sweep_stability.py --run outputs/clustering/cnmf/concat_hg/runs/20260818_112939
    python sweep_stability.py --new-concat            # the six 2026-08 concat runs
    python sweep_stability.py --new-concat --n-runs 20   # faster, coarser
    python sweep_stability.py --new-concat --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_stability as S  # noqa: E402

CLUST = ROOT / "outputs" / "clustering"

# The runs the K control is offered on. Resolved by track, newest run each, so a
# re-publish does not leave this pointing at a superseded id.
NEW_CONCAT_TRACKS = [
    ("cnmf", "concat_hg"), ("cnmf", "concat_rawds"),
    ("kmeans", "concat_hg"), ("kmeans", "concat_rawds"),
    ("hierarchical", "concat_hg"), ("hierarchical", "concat_rawds"),
]


def newest(method: str, fset: str):
    base = CLUST / method / fset / "runs"
    if not base.is_dir():
        return None
    runs = sorted(d for d in base.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def sweep_one(rd: Path, ks, n_runs: int, frac: float, dry: bool) -> int:
    lbk = rd / "cluster_labels_by_k.csv"
    xtr = rd / "X_train.npy"
    if not lbk.exists():
        print(f"  -- {rd.name}: no cluster_labels_by_k.csv, skipped")
        return 0
    if not xtr.exists():
        print(f"  -- {rd.name}: no X_train.npy, skipped")
        return 0

    df = pd.read_csv(lbk)
    have = {}
    for c in df.columns:
        if c.startswith("k_"):
            try:
                have[int(c[2:])] = c
            except ValueError:
                pass
    want = [k for k in ks if k in have]
    missing = [k for k in ks if k not in have]
    if missing:
        print(f"     (no column for K={missing} in this run's sweep)")
    if not want:
        print(f"  -- {rd.name}: none of K={ks} present, skipped")
        return 0

    X = np.load(xtr).astype(float)
    if len(X) != len(df):
        print(f"  !! {rd.name}: X has {len(X)} rows, sweep has {len(df)} - skipped")
        return 0

    # a --run given relative to cwd is not under the absolute CLUST
    try:
        shown = rd.resolve().relative_to(CLUST)
    except ValueError:
        shown = rd
    print(f"  {shown}  K={want}  n_runs={n_runs} frac={frac}")
    if dry:
        return 0

    rows = []
    for k in want:
        lab = df[have[k]].to_numpy()
        out = rd / "stability_by_k" / f"k_{k:02d}"
        out.mkdir(parents=True, exist_ok=True)
        S.save_consensus_artifacts(out, X, lab, n_runs=n_runs,
                                   subsample_frac=frac, verbose=False)
        js = json.loads((out / "stability_summary.json").read_text(encoding="utf-8"))
        rows.append(dict(k=k,
                         mean_jaccard=js.get("mean_jaccard"),
                         min_jaccard=js.get("min_jaccard"),
                         max_jaccard=js.get("max_jaccard"),
                         n_runs=js.get("n_runs"),
                         subsample_frac=js.get("subsample_frac")))
        print(f"    K={k:<3d} mean={js.get('mean_jaccard'):.3f} "
              f"worst={js.get('min_jaccard'):.3f}")

    pd.DataFrame(rows).to_csv(rd / "stability_by_k.csv", index=False)

    # Does the sweep agree with the run's own published number at its own K? If not,
    # the two were computed differently and the report must not present them together.
    own = rd / "stability_summary.json"
    if own.exists():
        pub = json.loads(own.read_text(encoding="utf-8"))
        pk = pub.get("k")
        m = [r for r in rows if r["k"] == pk]
        if m:
            d = abs((m[0]["mean_jaccard"] or 0) - (pub.get("mean_jaccard") or 0))
            tag = "matches" if d < 0.02 else f"DIFFERS by {d:.3f}"
            print(f"    published K={pk}: sweep {m[0]['mean_jaccard']:.3f} vs "
                  f"run {pub.get('mean_jaccard'):.3f}  -> {tag}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--new-concat", action="store_true",
                    help="the six 2026-08 concat runs the K control is offered on")
    ap.add_argument("--ks", type=int, nargs="+", default=list(range(5, 13)))
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--subsample-frac", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.run:
        targets = [Path(a.run).resolve()]
    elif a.new_concat:
        targets = [p for p in (newest(m, f) for m, f in NEW_CONCAT_TRACKS) if p]
    else:
        print("  give --run <dir> or --new-concat", file=sys.stderr)
        return 2

    total = 0
    for rd in targets:
        total += sweep_one(rd, a.ks, a.n_runs, a.subsample_frac, a.dry_run)
    print(f"\n  {total} (run, K) stability fits written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
