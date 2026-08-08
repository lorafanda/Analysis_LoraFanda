#!/usr/bin/env python3
"""
make_xcorr_figures.py — FIG 2.18 and FIG 2.19, regenerated with a polarity-correct estimator.

WHY THIS SCRIPT EXISTS AT ALL. Three separate problems made the published pair
unreproducible, and none of them were visible from the figures:

  1. The estimator picked its peak with argmax on the SIGNED Pearson r, so a
     suppression cluster — an inverted copy of the activation reference — could
     never align at its true peak. Three rows were certified `interpretable` with
     fabricated lags. Fixed in lf_cluster_timing.cluster_xcorr.
  2. 215_concat_crosscorrelation.ipynb had REF_CLUSTER hard-coded to 1, while every
     published number had been computed against c4. The notebook could not
     reproduce its own figures. The reference is now chosen by PROFILE
     (lf_cluster_timing.pick_reference_cluster), which is what the write-up always
     claimed was happening.
  3. NOTHING IN THE REPOSITORY PRODUCED FIG 2.19. Searching every .py and .ipynb
     for the full-range figure returns nothing; the notebook only ever saved the
     cropped view. The full-range panel is reconstructed here. Its axis is +/-47%
     of the trial, which pins min_overlap to 0.06: max_lag = int(150 * 0.94) = 141
     bins of 300 = 47.0%. That is how the original was made.

Everything is written from the run on disk, so the two figures and the CSV cannot
drift apart from each other again.

    python make_xcorr_figures.py
    python make_xcorr_figures.py --run <run dir> --k 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

RUN = ROOT / "outputs" / "clustering" / "kmeans" / "concat_hg" / "runs" / "20260803_175417"
# +/-47% of the trial. 150 response bins, int(150 * (1 - 0.06)) = 141, 141/300 = 47.0%.
FULLRANGE_MIN_OVERLAP = 0.06


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--window", default="response")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import lf_cluster_timing as T

    rd = Path(a.run)
    X = np.load(rd / "X_train.npy").astype(float)
    lab = pd.read_csv(rd / "cluster_labels_by_k.csv")[f"k_{a.k}"].to_numpy()
    out = rd / "timing"
    out.mkdir(parents=True, exist_ok=True)
    print(f"  run {rd.name}  K={a.k}  X{X.shape}  window={a.window}")

    # ---- reference, by profile ------------------------------------------------
    timing = T.cluster_timing(X, lab)
    ref = T.pick_reference_cluster(timing)
    rt = timing[(timing.cluster == ref)].set_index("condition")
    sizes = pd.Series(lab).value_counts().to_dict()
    print(f"  reference: c{ref} (n={sizes[ref]}) chosen by profile")
    for cond in rt.index:
        r = rt.loc[cond]
        print(f"    {cond:8s} stim onset {r.onset_stim_pct!s:>8.8}  "
              f"peak {r.peak_db:+.2f} dB  polarity {r.polarity}")

    # ---- the table ------------------------------------------------------------
    tab, prof = T.cluster_xcorr(X, lab, ref_cluster=ref, window=a.window)
    csv = out / f"xcorr_K{a.k}_ref-c{ref}_{a.window}.csv"
    tab.to_csv(csv, index=False)

    n_anti = int(tab["anti"].sum())
    n_ok = int(tab["interpretable"].sum())
    print(f"\n  {len(tab)} rows · {n_anti} cross-polarity · {n_ok} interpretable")
    print("  " + "-" * 74)
    for _, r in tab.sort_values(["condition", "cluster"]).iterrows():
        flag = "SUPP vs ACT" if r["anti"] else "           "
        print(f"  {r['condition']:8s} c{int(r['cluster']):<2d} n={int(r['n']):<4d} {flag} "
              f"lag {r['peak_lag_pct']:+6.1f}%  r {r['peak_r']:+.3f}  "
              f"{'OK ' if r['interpretable'] else '   '}{r['reason']}")

    # A suppression's offset is not a lead or a lag, so the headline is computed
    # over same-polarity rows only and the anti rows are reported separately.
    aud = tab[(tab.condition == "audio") & tab.interpretable]
    same = aud[~aud["anti"]]
    anti = aud[aud["anti"]]
    print(f"\n  AUDIO headline: {len(same)} same-polarity clusters interpretable, "
          f"{int((same.peak_lag_pct < 0).sum())} lead c{ref} "
          f"({-same.peak_lag_pct.max():.1f} to {-same.peak_lag_pct.min():.1f}%), "
          f"{int((same.peak_lag_pct > 0).sum())} lag it")
    for _, r in anti.iterrows():
        print(f"  AUDIO anti-polarity: c{int(r['cluster'])} at {r['peak_lag_pct']:+.1f}% "
              f"r={r['peak_r']:+.3f} — anti-correlated alignment, NOT a lag")

    # ---- FIG 2.18, the trustworthy window ------------------------------------
    order = sorted(int(c) for c in np.unique(lab))
    p18 = out / f"xcorr_K{a.k}_ref-c{ref}_{a.window}.png"
    T.plot_xcorr_profiles(
        tab, prof, ref_cluster=ref, order=order,
        title=f"Lag against the auditory cluster c{ref} · K={a.k} · {a.window} window",
        out_png=p18, dpi=a.dpi)
    print(f"\n  wrote {p18.name}")

    # ---- FIG 2.19, the whole lag axis ----------------------------------------
    tab_f, prof_f = T.cluster_xcorr(X, lab, ref_cluster=ref, window=a.window,
                                    min_overlap=FULLRANGE_MIN_OVERLAP,
                                    n_boot=120, n_shuffle=60)
    span = max(abs(v) for lg, _ in prof_f.values() for v in (lg[0], lg[-1]))
    reliable = max(abs(v) for lg, _ in prof.values() for v in (lg[0], lg[-1]))
    p19 = out / f"xcorr_K{a.k}_ref-c{ref}_{a.window}_fullrange.png"
    T.plot_xcorr_profiles(
        tab_f, prof_f, ref_cluster=ref, order=order, ylim=(-1, 1),
        reliable_lag=reliable, mark_within=reliable,
        title=f"The whole lag axis (+/-{span:.0f}%) · shaded = the +/-{reliable:.0f}% "
              f"reported above · peaks outside it are drawn but not marked",
        out_png=p19, dpi=a.dpi)
    tab_f.to_csv(out / f"xcorr_K{a.k}_ref-c{ref}_{a.window}_fullrange.csv", index=False)
    print(f"  wrote {p19.name}   axis +/-{span:.0f}%, reported window +/-{reliable:.0f}%")
    print(f"  wrote {csv.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
