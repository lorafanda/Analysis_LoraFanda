#!/usr/bin/env python3
"""
run_kmedoids.py - counter 4: k-medoids with correlation distance, K by the elbow, odd / even
validation - the recipe of Regev, Casto & Fedorenko (2024, Nat Hum Behav 8:1924) on their
intracranial language data, run on this cohort as a published benchmark.

    python run_kmedoids.py                       bands 13-170 Hz x 30 bins, K 2..12
    python run_kmedoids.py --feature hg          the pipeline's concat_hg
    python run_kmedoids.py --keep-muscle
    python run_kmedoids.py --quick

WHY IT IS A COUNTER. A medoid is a real electrode (like a cNMF archetype), the distance is
1 - r (shape only, like unit-norm), and K is read from the elbow of the cost curve with the
odd / even trials as the check - nothing is shared with the k-means / Ward / cNMF runs
except the electrodes.

WHAT IS REPORTED
  the cost curve (summed correlation distance to the medoids) and its elbow
  odd / even: k-medoids on each half separately, clusters matched by medoid correlation,
  agreement and ARI per K
  robustness to removing electrodes (5 ... 25 % removed, refit, ARI with the full solution)
  at the chosen K: medoid electrodes, cluster sizes, patient spread, fingerprints, maps,
  anatomical coherence; the same at K = 7 for comparison with the published runs

OUTPUT  outputs/clustering/kmedoids/<feature>/runs/<run_id>/   labels.csv carries
        cluster_kmedoids_K<elbow> and cluster_kmedoids_K7. Not registered in index.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "functions"))
from functions import lf_counters as LCn                       # noqa: E402
from functions.lf_concat import concat_hg_features             # noqa: E402
import lf_decompose as D                                       # noqa: E402

METHOD = "kmedoids"


def features(X, name):
    if name == "hg":
        return concat_hg_features(X).astype(np.float64), "concat_hg"
    F, _ = LCn.band_features(X, 13.0, 170.0, 30)
    return F, "bands13_170_30bins"


def match_by_medoids(F1, m1, F2, m2):
    """Match the clusters of two fits by the correlation of their medoid electrodes' profiles."""
    from scipy.optimize import linear_sum_assignment
    Cm = np.corrcoef(F1[m1], F2[m2])[:len(m1), len(m1):]
    r, c = linear_sum_assignment(-Cm)
    return dict(zip(c.tolist(), r.tolist())), float(Cm[r, c].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["b13_170", "hg"], default="b13_170")
    ap.add_argument("--ks", default="2-12")
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--keep-muscle", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    df, X, cache = LCn.load_cohort(verbose=not a.quick)
    X1, X2 = LCn.load_halves(df, cache["name"], verbose=not a.quick)
    X1, X2 = np.nan_to_num(X1), np.nan_to_num(X2)
    muscle = LCn.muscle_flags(X)
    keep = np.ones(len(df), bool) if a.keep_muscle else ~muscle
    df, X, X1, X2 = df[keep].reset_index(drop=True), X[keep], X1[keep], X2[keep]
    F, fset = features(X, a.feature); F1, _ = features(X1, a.feature); F2, _ = features(X2, a.feature)
    n = len(df); pat = df["patient_id"].astype(str).to_numpy()
    n_init = 3 if a.quick else a.n_init
    print(f"\n=== k-medoids (1 - r) on {fset}: {n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), cache {cache['name']}")
    Dm, D1, D2 = LCn.corr_distance(F), LCn.corr_distance(F1), LCn.corr_distance(F2)
    lo, hi = (int(v) for v in a.ks.split("-"))
    ks = [3, 5, 7] if a.quick else list(range(lo, hi + 1))
    fits, rows = {}, []
    cost1 = float(Dm[:, np.argmin(Dm.sum(1))].sum())                 # K = 1: one medoid for everybody
    for k in ks:
        lab, med, cost = LCn.kmedoids(Dm, k, n_init=n_init)
        l1, m1, _ = LCn.kmedoids(D1, k, n_init=n_init); l2, m2, _ = LCn.kmedoids(D2, k, n_init=n_init)
        remap, r_med = match_by_medoids(F1, m1, F2, m2)
        l2m = np.array([remap[int(x)] for x in l2])
        agree = float((l1 == l2m).mean())
        fits[k] = (lab, med, cost)
        rows.append(dict(k=k, cost=cost, explained=1 - cost / cost1, oddeven_ari=adjusted_rand_score(l1, l2), oddeven_agreement=agree, medoid_r_across_halves=r_med))
        print(f"    K = {k:2d}: explained {1 - cost / cost1:.3f} · odd/even ARI {rows[-1]['oddeven_ari']:.3f}, agreement {100 * agree:.0f} %, matched medoids r {r_med:.2f}")
    curves = pd.DataFrame(rows)
    # the elbow: the point of the explained curve (K = 1 at 0) farthest above the chord joining its two ends
    kk = np.r_[1, curves.k.to_numpy()]; ex = np.r_[0.0, curves.explained.to_numpy()]
    x = (kk - kk[0]) / max(kk[-1] - kk[0], 1); y = (ex - ex[0]) / max(ex[-1] - ex[0], 1e-12)
    K_elbow = int(kk[np.argmax(y - x)])
    Ks = sorted({K_elbow, 7} & set(ks)) or [K_elbow]
    # robustness to removing electrodes, at the elbow
    lab_e, med_e, _ = fits[K_elbow] if K_elbow in fits else LCn.kmedoids(Dm, K_elbow, n_init=n_init)
    rng = np.random.default_rng(0); rob = []
    for frac in ([0.1] if a.quick else [0.05, 0.10, 0.15, 0.20, 0.25]):
        for s in range(2 if a.quick else 3):
            idx = np.sort(rng.choice(n, int((1 - frac) * n), replace=False))
            l, _, _ = LCn.kmedoids(Dm[np.ix_(idx, idx)], K_elbow, n_init=n_init, seed=s)
            rob.append(dict(removed=frac, seed=s, ari=adjusted_rand_score(lab_e[idx], l)))
    rob = pd.DataFrame(rob)
    xyz = LCn.coords_for(df)

    out = LCn.new_run_dir(METHOD, fset)
    lab_df = df.drop(columns=[c for c in df.columns if c.startswith("file_path_")], errors="ignore").copy()
    lab_df["muscle_flag"] = muscle[keep]
    res = {}
    for K in Ks:
        lab, med, cost = fits[K]
        lab_df[f"cluster_kmedoids_K{K}"] = lab
        fp = LCn.fingerprints(X, lab); fp["patients"] = [df.loc[lab == k, "patient_id"].nunique() for k in fp.cluster]
        fp["medoid"] = [f"{df.patient_id[med[k]]} {df.electrode[med[k]]}" for k in fp.cluster]
        fp.to_csv(out / f"clusters_K{K}.csv", index=False)
        coh = D.spatial_coherence(lab, xyz)
        res[K] = dict(fp=fp, coh=coh, lab=lab)
        LCn.cluster_courses(X, lab, out / f"M2_courses_K{K}.png", f"M2 - mean HG per cluster, k-medoids K = {K} ({fset})")
        LCn.cluster_maps(lab, xyz, out / f"M3_maps_K{K}.png", f"M3 - k-medoids clusters on fsaverage, K = {K}")
    lab_df.to_csv(out / "labels.csv", index=False); curves.to_csv(out / "k_curves.csv", index=False); rob.to_csv(out / "robustness.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(curves.k, curves.explained, marker="o"); axes[0].axvline(K_elbow, color="#c1121f", ls="--", lw=1); axes[0].set_xlabel("K"); axes[0].set_ylabel("1 - cost / cost(K=1)"); axes[0].set_title("elbow", loc="left", fontsize=9)
    axes[1].plot(curves.k, curves.oddeven_ari, marker="o", label="ARI"); axes[1].plot(curves.k, curves.oddeven_agreement, marker="s", label="agreement"); axes[1].set_xlabel("K"); axes[1].set_title("odd vs even trials", loc="left", fontsize=9); axes[1].legend(fontsize=7, frameon=False)
    g = rob.groupby("removed").ari.agg(["mean", "std"]); axes[2].errorbar(100 * g.index, g["mean"], yerr=g["std"], marker="o", capsize=3); axes[2].set_xlabel("% electrodes removed"); axes[2].set_ylabel("ARI with the full solution"); axes[2].set_title(f"robustness at K = {K_elbow}", loc="left", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"M1 - k-medoids with correlation distance ({fset})", x=.02, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(out / "M1_k_selection.png", dpi=150); plt.close(fig)

    summary = dict(n_electrodes=n, n_patients=int(len(np.unique(pat))), muscle_dropped=int(0 if a.keep_muscle else muscle.sum()), K_elbow=K_elbow, Ks_saved=Ks,
                   k_curves=curves.to_dict("records"), robustness=rob.groupby("removed").ari.mean().to_dict(),
                   spatial_coherence={int(K): list(map(float, res[K]["coh"])) for K in Ks}, sizes={int(K): [int((res[K]["lab"] == k).sum()) for k in range(K)] for K in Ks})
    LCn.write_manifest(out, method=METHOD, method_label="k-medoids, correlation distance (Regev et al. 2024 recipe)", feature_set=fset, feature_set_label=fset,
                       params=dict(feature=a.feature, ks=ks, n_init=n_init, keep_muscle=a.keep_muscle, quick=a.quick), summary=summary,
                       artifacts=dict(labels="labels.csv", curves="k_curves.csv", robustness="robustness.csv"), cache=cache,
                       note="Hard labels; medoids are real electrodes (clusters_K*.csv names them).")
    lines = [f"# k-medoids (1 - r) on {fset} - {out.name}", "",
             f"{n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), {len(np.unique(pat))} patients, cache {cache['name']}; {n_init} starts per fit.", "",
             f"- elbow: K = {K_elbow} (the point of the explained curve farthest above the chord from K = 1 to K = {ks[-1]})",
             "", "| K | explained | odd/even ARI | odd/even agreement | matched medoids r |", "|---|---|---|---|---|"]
    for _, r in curves.iterrows():
        lines.append(f"| {int(r.k)} | {r.explained:.3f} | {r.oddeven_ari:.3f} | {100 * r.oddeven_agreement:.0f} % | {r.medoid_r_across_halves:.2f} |")
    lines += ["", f"- robustness at K = {K_elbow}: ARI with the full solution " + ", ".join(f"{100 * k:.0f} % removed {v:.2f}" for k, v in rob.groupby('removed').ari.mean().items())]
    for K in Ks:
        fp, coh = res[K]["fp"], res[K]["coh"]
        lines += ["", f"## K = {K}  (anatomical coherence / chance {coh[1]:.2f})", "", "| cluster | n | patients | medoid | HG resp a / p / r | HG stim a / p / r | beta a / p / r |", "|---|---|---|---|---|---|---|"]
        for _, r in fp.iterrows():
            lines.append(f"| {int(r.cluster)} | {int(r.n)} | {int(r.patients)} | {r.medoid} | {r.hg_resp_audio:+.2f} / {r.hg_resp_picture:+.2f} / {r.hg_resp_reading:+.2f} | {r.hg_stim_audio:+.2f} / {r.hg_stim_picture:+.2f} / {r.hg_stim_reading:+.2f} | {r.beta_audio:+.2f} / {r.beta_picture:+.2f} / {r.beta_reading:+.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:6])); print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
