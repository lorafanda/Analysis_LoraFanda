#!/usr/bin/env python3
"""
make_missing_centroids.py - write cluster_centroids/ for any run that lacks them.

The clustering notebooks write centroids inside fit_and_save, so k-means and Ward
runs get them for free. The convex-NMF runs are published by
publish_decomposition.py, which never called that path - so cnmf runs have
labels.csv, components.npy and glassbrains but no centroid chips, and every
downstream figure that tiles centroids skips them silently.

This reuses the project's own lf_centroids renderer rather than drawing its own,
so a cnmf centroid chip looks identical to a k-means one and the two can sit side
by side in a figure.

    python make_missing_centroids.py                # every run missing them
    python make_missing_centroids.py --dry-run
    python make_missing_centroids.py --run <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_centroids as LC  # noqa: E402

CLUST = ROOT / "outputs" / "clustering"


def centroid_shape_for(run_dir: Path, feature_set: str):
    """Heatmap feature sets need (n_freq, n_time) to reshape a flat row.

    Derived from feature_schema.json rather than assumed: concat_rawds is
    3 conditions x 15 bands x 30 time bins, and reshaping it as if it were a time
    course is exactly the bug that produced a meaningless component figure.
    """
    if LC._is_line_feature_set(feature_set):
        return None
    fs = run_dir / "feature_schema.json"
    if not fs.exists():
        return None
    names = json.loads(fs.read_text(encoding="utf-8"))["feature_names"]
    bands, times, conds = [], set(), []
    for f in names:
        c, b, t = f.split("|")
        if b not in bands:
            bands.append(b)
        if c not in conds:
            conds.append(c)
        times.add(t)
    # rows = bands, columns = the 3 conditions laid end to end in time
    return (len(bands), len(conds) * len(times))


def needs(run_dir: Path) -> bool:
    d = run_dir / "cluster_centroids"
    return not (d.is_dir() and any(d.iterdir()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    idx = json.loads((CLUST / "index.json").read_text(encoding="utf-8"))
    runs = idx["runs"] if isinstance(idx, dict) else idx

    todo = []
    for r in runs:
        rd = CLUST / r["method"] / r["feature_set"] / "runs" / r["run_id"]
        if a.run and Path(a.run).resolve() != rd.resolve():
            continue
        if not rd.is_dir():
            continue
        if not (rd / "X_train.npy").exists() or not (rd / "labels.csv").exists():
            continue
        if needs(rd):
            todo.append((r["method"], r["feature_set"], rd))

    if not todo:
        print("  every run with X_train already has centroids")
        return 0

    print(f"  {len(todo)} run(s) missing cluster_centroids:")
    for m, f, rd in todo:
        print(f"    {m}/{f}/{rd.name}")
    if a.dry_run:
        print("  (dry run)")
        return 0

    for m, f, rd in todo:
        X = np.load(rd / "X_train.npy")
        shape = centroid_shape_for(rd, f)
        print(f"\n  {m}/{f}/{rd.name}: X {X.shape}, feature_set={f}, shape={shape}")
        try:
            out = LC.save_per_cluster_centroids(rd, X, f, m, centroid_shape=shape)
            print(f"    wrote {out}")
        except Exception as e:
            print(f"    !! failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
