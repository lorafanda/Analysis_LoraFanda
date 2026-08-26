#!/usr/bin/env python3
"""
compare_band_schemes.py - is 15 bands worth it, and which 5 would you keep?

Merges the 15 bands of concat_rawds into candidate schemes and scores each against the
same electrodes, so the only thing that varies is how the spectrum is divided.

TWO DIFFERENT AXES DECIDE WHETHER A MERGE IS FREE, and they are easy to conflate:

    REDUNDANCY    adjacent bands that correlate at r = 0.9 say the same thing, so
                  fusing them costs nothing.
    INFORMATIVENESS  eta2 - how much of a band's variance cluster identity explains.

A merge is safe if the bands it fuses are redundant OR uninformative. It is expensive
only when it fuses bands that are BOTH distinct and informative. 30-50 and 50-70 Hz are
the least redundant pair in the whole spectrum (r = 0.48) and also the two LEAST
informative bands (eta2 0.14 and 0.11), so fusing them is cheap despite the low
correlation - which the correlation alone would have told you not to do.

SCORED ON WHAT IS NOT GIVEN TO THE ALGORITHM. Silhouette and variance explained both
improve when features are removed, so neither can say a scheme is better - fewer
dimensions is an easier problem, not a truer one. Anatomical coherence is the honest
criterion here: no method is given a coordinate, so a scheme that lands electrodes near
their neighbours found something rather than fitted something.

    python compare_band_schemes.py
    python compare_band_schemes.py --k 12 --zscore
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
import lf_decompose as LD  # noqa: E402

CLUST = ROOT / "outputs" / "clustering"

# Candidate schemes, named by the ORIGINAL band labels they fuse. Every edge lands on a
# real band boundary - a scheme that cuts inside a band cannot be built by averaging.
SCHEMES = {
    "15 bands (baseline)": None,
    "user 4-band": [
        ["1-4Hz", "4-8Hz", "8-13Hz", "13-20Hz"],
        ["20-30Hz", "30-50Hz", "50-70Hz"],
        ["70-100Hz", "100-130Hz", "130-170Hz"],
        ["170-220Hz", "220-270Hz", "270-320Hz", "320-360Hz", "360-400Hz"],
    ],
    "user 5-band (+VHF split)": [
        ["1-4Hz", "4-8Hz", "8-13Hz", "13-20Hz"],
        ["20-30Hz", "30-50Hz", "50-70Hz"],
        ["70-100Hz", "100-130Hz", "130-170Hz"],
        ["170-220Hz", "220-270Hz"],
        ["270-320Hz", "320-360Hz", "360-400Hz"],
    ],
    # keeps 20-30 out of the low-gamma group, because 20-30 | 30-50 is the second least
    # redundant boundary in the spectrum
    "redundancy-aware 5-band": [
        ["1-4Hz", "4-8Hz", "8-13Hz", "13-20Hz"],
        ["20-30Hz"],
        ["30-50Hz", "50-70Hz"],
        ["70-100Hz", "100-130Hz", "130-170Hz"],
        ["170-220Hz", "220-270Hz", "270-320Hz", "320-360Hz", "360-400Hz"],
    ],
    # the two eta2 peaks kept apart from everything else
    "eta2-aware 5-band": [
        ["1-4Hz", "4-8Hz"],
        ["8-13Hz", "13-20Hz"],
        ["20-30Hz", "30-50Hz", "50-70Hz", "70-100Hz", "100-130Hz"],
        ["130-170Hz", "170-220Hz", "220-270Hz", "270-320Hz"],
        ["320-360Hz", "360-400Hz"],
    ],
    "HG only (70-170Hz)": [["70-100Hz", "100-130Hz", "130-170Hz"]],
    "VHF only (170-400Hz)": [["170-220Hz", "220-270Hz", "270-320Hz",
                              "320-360Hz", "360-400Hz"]],
    "low only (1-20Hz)": [["1-4Hz", "4-8Hz", "8-13Hz", "13-20Hz"]],
}


def newest(method, fset):
    d = CLUST / method / fset / "runs"
    r = sorted(d.glob("20260826_*")) or sorted(p for p in d.iterdir() if p.is_dir())
    return r[-1] if r else None


def merge(X, band, groups, zscore=False):
    """Average the dB values within each group.

    AVERAGING dB IS A GEOMETRIC MEAN IN POWER, which is the right thing here: the whole
    pipeline is in dB, the 15 bands arrive in dB, and converting to power to average and
    back would change what a merged band means relative to every other figure.

    Bands are averaged UNWEIGHTED. They are unequal width - 3 Hz at the bottom, 40 at the
    top - so a plain mean gives a narrow low band the same say as a wide high one. That
    is a real choice, not an oversight: the bands were defined as units of interest, not
    as samples of a continuum.
    """
    time_of = np.array([f.split("|")[0] + "|" + f.split("|")[2] for f in FEATS])
    cols = sorted(set(time_of))
    out = np.empty((X.shape[0], len(groups) * len(cols)))
    for gi, g in enumerate(groups):
        for ci, c in enumerate(cols):
            m = np.isin(band, g) & (time_of == c)
            out[:, gi * len(cols) + ci] = X[:, m].mean(1)
    if zscore:
        # equal weight per merged band, which is what the 15-band set does NOT do:
        # the four lowest bands hold 55% of its sum of squares
        n = len(cols)
        for gi in range(len(groups)):
            s = slice(gi * n, (gi + 1) * n)
            out[:, s] = (out[:, s] - out[:, s].mean()) / max(out[:, s].std(), 1e-12)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--zscore", action="store_true",
                    help="z-score each merged band to equal weight before clustering")
    ap.add_argument("--n-shuffle", type=int, default=20)
    a = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ari, silhouette_score

    global FEATS
    rd = newest("cnmf", "concat_rawds")
    X = np.load(rd / "X_train.npy").astype(float)
    FEATS = json.loads((rd / "feature_schema.json").read_text())["feature_names"]
    band = np.array([f.split("|")[1] for f in FEATS])
    bands = list(dict.fromkeys(band))
    K = a.k

    xyz = np.load(CLUST / "statistics" / f"concat_rawds_K{K}" / "stats_xyz.npy")
    ok = np.isfinite(xyz).all(1)
    ref = pd.read_csv(newest("kmeans", "concat_rawds") /
                      "cluster_labels_by_k.csv")[f"k_{K}"].to_numpy()

    print(f"n={len(X)} electrodes, K={K}, coords for {ok.sum()}"
          f"  ({'per-band z-scored' if a.zscore else 'raw dB, unnormalised'})\n")

    rows = []
    for name, groups in SCHEMES.items():
        g = [[b] for b in bands] if groups is None else groups
        flat = [b for grp in g for b in grp]
        unknown = [b for b in flat if b not in bands]
        if unknown:
            print(f"  !! {name}: unknown bands {unknown}"); continue
        Xs = merge(X, band, g, zscore=a.zscore)
        lab = KMeans(K, n_init=10, random_state=0).fit_predict(Xs)
        obs, over = LD.spatial_coherence(lab[ok], xyz[ok], n_shuffle=a.n_shuffle)
        rows.append(dict(scheme=name, n_bands=len(g), n_feat=Xs.shape[1],
                         coherence_over_chance=over,
                         silhouette=silhouette_score(Xs, lab),
                         ari_vs_15band=ari(lab, ref),
                         smallest_cluster=int(np.bincount(lab, minlength=K).min())))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(d.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    base = d[d.scheme.str.startswith("15 bands")].iloc[0]
    print(f"\n  ANATOMY IS THE CRITERION THAT IS NOT GIVEN TO THE ALGORITHM.")
    print(f"  15-band baseline: {base.coherence_over_chance:.3f}x chance\n")
    for _, r in d.sort_values("coherence_over_chance", ascending=False).iterrows():
        d_ = r.coherence_over_chance - base.coherence_over_chance
        tag = "  <-- BEATS the 15-band set" if d_ > 0.02 else ""
        print(f"   {r.coherence_over_chance:.3f}x  ({d_:+.3f})  "
              f"{r.n_feat:>4} feat  {r.scheme}{tag}")
    print("\n  silhouette rises as features are removed - fewer dimensions is an easier")
    print("  problem, not a truer one - so it is reported but not ranked on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
