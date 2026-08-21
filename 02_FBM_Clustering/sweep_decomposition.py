#!/usr/bin/env python3
"""
sweep_decomposition.py - fit convex NMF across a range of K and publish the sweep.

WHY THIS EXISTS
k-means and Ward runs carry cluster_labels_by_k.csv: one column per K, so any
downstream tool can re-cut the same run at a different number of clusters without
refitting. Convex NMF had no equivalent - run_decomposition fits exactly one K and
publish_decomposition writes exactly that - so the K control in the visualizer and the
report had nothing to switch to on a cnmf run.

This closes that gap by fitting the same decomposition at every K in the range and
writing the result into the PUBLISHED run directory, in the same shape the hard
clustering runs use.

WHAT IT WRITES, into cnmf/<fset>/runs/<id>/

    cluster_labels_by_k.csv      sample_idx, k_5 .. k_12   argmax component per K
    loadings_by_k/G_k05.npy ..   (n_electrodes, K) row-normalised loadings per K
    components_by_k/C_k05.npy    (K, n_features) component shapes per K
    sweep_by_k.csv               k, var_explained, frac_dominant, frac_no_majority,
                                 median_top_weight, sizes

The loadings matter as much as the labels. Convex NMF is a GRADED model: an electrode
carries a weight on every component, and argmax is a summary of that, not the result.
If the sweep saved only argmax then every K except the published one would render as a
hard partition - no loading gate, no argmax sizing, no grey ramp - and the graded model
would silently become a k-means with extra steps. G per K is (1266 x K) floats, so
keeping it costs almost nothing.

    python sweep_decomposition.py --feature-set concat_hg
    python sweep_decomposition.py --feature-set concat_hg --ks 5 6 7 8 9 10 11 12
    python sweep_decomposition.py --feature-set concat_hg --dry-run
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
import lf_decompose as D  # noqa: E402
import lf_runs as R  # noqa: E402

import run_decomposition as RD  # noqa: E402  (reuses load(); same X, same order)

CLUST = ROOT / "outputs" / "clustering"
DEFAULT_KS = list(range(5, 13))


def newest_cnmf_run(fset: str) -> Path:
    """The run the sweep belongs to.

    Resolved rather than pinned: publish_decomposition stamps a new run id every time
    it runs, and a hard-coded id here would attach the sweep to a superseded fit while
    reporting success.
    """
    base = CLUST / "cnmf" / fset / "runs"
    if not base.is_dir():
        raise SystemExit(f"  !! no published cnmf run for {fset} - run "
                         f"publish_decomposition.py --feature-set {fset} first")
    runs = sorted(d for d in base.iterdir() if d.is_dir())
    if not runs:
        raise SystemExit(f"  !! {base} is empty")
    return runs[-1]


def mixture_stats(Gn: np.ndarray) -> dict:
    """How graded the fit is at this K, on the same definitions meta.json uses."""
    srt = np.sort(Gn, axis=1)
    top = srt[:, -1]
    return dict(
        frac_dominant=float((top >= 0.5).mean()),
        frac_no_majority=float((top < 0.5).mean()),
        median_top_weight=float(np.median(top)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="concat_hg", choices=RD.FEATURE_SETS)
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--run", default=None, help="target run dir (default: newest)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fset = a.feature_set
    rd = Path(a.run) if a.run else newest_cnmf_run(fset)
    ks = sorted(set(int(k) for k in a.ks))

    src, X, lab, xyz, conds, bands, nt = RD.load(fset)
    Xs = D.unit_norm(X)
    print(f"  {fset}: {X.shape[0]} electrodes x {X.shape[1]} features")
    print(f"  target run: {rd.relative_to(ROOT)}")
    print(f"  K sweep: {ks}")

    # The sweep must describe the SAME electrodes the run published, in the same order,
    # or k_7 here would not be the run's own labels.csv. Checked rather than assumed.
    pub = pd.read_csv(rd / "labels.csv")
    if len(pub) != len(Xs):
        raise SystemExit(f"  !! run has {len(pub)} rows, X has {len(Xs)} - refusing")

    if a.dry_run:
        print("  (dry run - nothing written)")
        return 0

    (rd / "loadings_by_k").mkdir(parents=True, exist_ok=True)
    (rd / "components_by_k").mkdir(parents=True, exist_ok=True)

    cols = {"sample_idx": np.arange(len(Xs), dtype=int)}
    rows = []
    for k in ks:
        W, G, C = D.convex_nmf(Xs, k, random_state=0, n_iter=a.n_iter)
        ve = float(1 - ((Xs - D.reconstruct(Xs, W, G)) ** 2).sum() / (Xs ** 2).sum())
        Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
        lab_k = Gn.argmax(1).astype(int)
        sizes = np.bincount(lab_k, minlength=k)

        cols[f"k_{k}"] = lab_k
        np.save(rd / "loadings_by_k" / f"G_k{k:02d}.npy", Gn.astype(np.float32))
        np.save(rd / "components_by_k" / f"C_k{k:02d}.npy", C.astype(np.float32))

        m = mixture_stats(Gn)
        rows.append(dict(k=k, var_explained=ve, sizes=" ".join(str(int(s)) for s in sizes), **m))
        print(f"    K={k:<3d} VE={ve:.3f}  no-majority={100*m['frac_no_majority']:.0f}%"
              f"  median top={m['median_top_weight']:.2f}  sizes={[int(x) for x in sizes]}")

    pd.DataFrame(cols).to_csv(rd / "cluster_labels_by_k.csv", index=False)
    pd.DataFrame(rows).to_csv(rd / "sweep_by_k.csv", index=False)

    # Record the sweep on the run so a reader knows the columns are cNMF argmax rather
    # than a hard partition, and which K the run itself publishes.
    mp = rd / "manifest.json"
    if mp.exists():
        man = json.loads(mp.read_text(encoding="utf-8"))
        man["sweep"] = {
            "ks": ks,
            "labels": "cluster_labels_by_k.csv",
            "loadings": "loadings_by_k/G_k{k:02d}.npy",
            "components": "components_by_k/C_k{k:02d}.npy",
            "note": "argmax of a graded fit; loadings_by_k carries the weights the "
                    "argmax summarises",
        }
        mp.write_text(json.dumps(man, indent=2), encoding="utf-8")

    print(f"\n  wrote cluster_labels_by_k.csv ({len(ks)} cuts) + loadings_by_k/ "
          f"+ components_by_k/ into {rd.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
