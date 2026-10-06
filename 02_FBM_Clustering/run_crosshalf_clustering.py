#!/usr/bin/env python3
"""
run_crosshalf_clustering.py - counter 3: cluster on the CROSS-HALF distance.

    python run_crosshalf_clustering.py                    bands 13-170 Hz x 30 bins, K = 7, sweep 2..12
    python run_crosshalf_clustering.py --feature hg       the pipeline's concat_hg instead
    python run_crosshalf_clustering.py --keep-muscle      do not drop the muscle-flagged electrodes
    python run_crosshalf_clustering.py --quick

THE IDEA. Every published run measures the distance between two electrodes on the SAME
trial average, so each electrode's own trial noise sits in every distance it has. Here the
distance between electrode i and electrode j is measured between i's odd trials and j's even
trials (and the other way round, averaged). Shared noise cannot survive that, and an
electrode's self-distance is no longer inflated by its noise. Two partitions are fitted on
that matrix, Ward and k-medoids, and compared with Ward on the ordinary full-data distance.

WHAT IS REPORTED
  agreement (ARI) between the cross-half partition and the full-data one, per K
  subsample stability per cluster (co-association over 80 % subsamples)
  anatomical coherence of the argmax labels against a label shuffle (lf_decompose)
  cluster sizes, patient spread, fingerprints (HG in the stimulus and response halves per
  condition, beta, a modality index), the mean HG course per cluster, maps

OUTPUT  outputs/clustering/crosshalf/<feature>/runs/<run_id>/  labels.csv carries
        cluster_crosshalf_ward, cluster_crosshalf_kmedoids, cluster_full_ward.
        Not registered in index.json; the cohort tag is in the manifest.
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
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "functions"))
from functions import lf_counters as LCn                       # noqa: E402
from functions.lf_concat import concat_hg_features             # noqa: E402
import lf_decompose as D                                       # noqa: E402

METHOD = "crosshalf"


def features(X, name):
    if name == "hg":
        return concat_hg_features(X).astype(np.float64), "concat_hg"
    F, bands = LCn.band_features(X, 13.0, 170.0, 30)
    return F, "bands13_170_30bins"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["b13_170", "hg"], default="b13_170")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--ks", default="2-12")
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
    print(f"\n=== cross-half clustering on {fset}: {n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), cache {cache['name']}")

    Dx = LCn.cross_half_distance(F1, F2)
    Df = squareform(pdist(LCn.unit_norm(F)))
    lo, hi = (int(v) for v in a.ks.split("-"))
    ks = [3, 5, 7] if a.quick else list(range(lo, hi + 1))
    rows = []
    for k in ks:
        lw = LCn.ward_on_distance(Dx, k); lf = LCn.ward_on_distance(Df, k)
        lm, _, cost = LCn.kmedoids(Dx, k, n_init=3 if a.quick else 10)
        rows.append(dict(k=k, ari_crossward_vs_fullward=adjusted_rand_score(lw, lf), ari_crosskmed_vs_fullward=adjusted_rand_score(lm, lf),
                         ari_crossward_vs_crosskmed=adjusted_rand_score(lw, lm), kmedoids_cost=cost))
        print(f"    K = {k:2d}: ARI cross-half Ward vs full Ward {rows[-1]['ari_crossward_vs_fullward']:.3f} · cross-half k-medoids vs full Ward {rows[-1]['ari_crosskmed_vs_fullward']:.3f}")
    curves = pd.DataFrame(rows)

    K = a.k
    lw = LCn.ward_on_distance(Dx, K); lf = LCn.ward_on_distance(Df, K); lm, med, _ = LCn.kmedoids(Dx, K, n_init=3 if a.quick else 10)
    lf_m, agree_f = LCn.match_labels(lw, lf); lm_m, agree_m = LCn.match_labels(lw, lm)
    per_w, C = LCn.subsample_stability(lambda idx: LCn.ward_on_distance(Dx[np.ix_(idx, idx)], K), n, lw, n_rounds=5 if a.quick else 25)
    per_f, _ = LCn.subsample_stability(lambda idx: LCn.ward_on_distance(Df[np.ix_(idx, idx)], K), n, lf, n_rounds=5 if a.quick else 25)
    xyz = LCn.coords_for(df)
    coh_w = D.spatial_coherence(lw, xyz); coh_f = D.spatial_coherence(lf, xyz); coh_m = D.spatial_coherence(lm, xyz)

    out = LCn.new_run_dir(METHOD, fset)
    lab = df.drop(columns=[c for c in df.columns if c.startswith("file_path_")], errors="ignore").copy()
    lab["cluster_crosshalf_ward"] = lw; lab["cluster_crosshalf_kmedoids"] = lm_m; lab["cluster_full_ward"] = lf_m
    lab["muscle_flag"] = muscle[keep]
    lab.to_csv(out / "labels.csv", index=False)
    curves.to_csv(out / "k_curves.csv", index=False)
    fp = LCn.fingerprints(X, lw); fp["stability"] = fp.cluster.map(per_w); fp["patients"] = [df.loc[lw == k, "patient_id"].nunique() for k in fp.cluster]
    fp.to_csv(out / "clusters_crosshalf_ward.csv", index=False)
    np.save(out / "coassociation_crosshalf_ward.npy", C.astype(np.float32))
    # figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(curves.k, curves.ari_crossward_vs_fullward, marker="o", label="cross-half Ward vs full Ward")
    axes[0].plot(curves.k, curves.ari_crosskmed_vs_fullward, marker="s", label="cross-half k-medoids vs full Ward")
    axes[0].plot(curves.k, curves.ari_crossward_vs_crosskmed, marker="^", label="cross-half Ward vs k-medoids")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("adjusted Rand index"); axes[0].legend(fontsize=7, frameon=False); axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].bar(np.arange(K) - 0.2, [per_w.get(k, np.nan) for k in range(K)], width=0.4, label="cross-half Ward")
    axes[1].bar(np.arange(K) + 0.2, [per_f.get(k, np.nan) for k in range(K)], width=0.4, label="full-data Ward")
    axes[1].set_xlabel(f"cluster (K = {K})"); axes[1].set_ylabel("subsample stability"); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7, frameon=False); axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"X1 - cross-half clustering on {fset}", x=.02, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(out / "X1_k_and_stability.png", dpi=150); plt.close(fig)
    LCn.cluster_courses(X, lw, out / "X2_courses_crosshalf_ward.png", f"X2 - mean HG per cluster, cross-half Ward, K = {K} ({fset})")
    LCn.cluster_maps(lw, xyz, out / "X3_maps_crosshalf_ward.png", f"X3 - cross-half Ward clusters on fsaverage (K = {K})")
    o = np.argsort(lw)
    fig, ax = plt.subplots(figsize=(5.2, 4.6)); im = ax.imshow(np.nan_to_num(C)[np.ix_(o, o)], cmap="magma", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"X4 - co-association over subsamples, electrodes sorted by cluster", loc="left", fontsize=9); fig.colorbar(im, label="fraction co-clustered"); fig.savefig(out / "X4_coassociation.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    summary = dict(n_electrodes=n, n_patients=int(len(np.unique(pat))), muscle_dropped=int(0 if a.keep_muscle else muscle.sum()), K=K,
                   ari_vs_full_ward=float(adjusted_rand_score(lw, lf)), agreement_vs_full_ward=agree_f, ari_vs_crosshalf_kmedoids=float(adjusted_rand_score(lw, lm)),
                   stability_crosshalf_ward=per_w, stability_full_ward=per_f,
                   spatial_coherence=dict(crosshalf_ward=list(map(float, coh_w)), full_ward=list(map(float, coh_f)), crosshalf_kmedoids=list(map(float, coh_m))),
                   sizes=[int((lw == k).sum()) for k in range(K)], k_curves=curves.to_dict("records"))
    LCn.write_manifest(out, method=METHOD, method_label="Ward / k-medoids on the cross-half distance", feature_set=fset, feature_set_label=fset,
                       params=dict(feature=a.feature, k=K, ks=ks, keep_muscle=a.keep_muscle, quick=a.quick), summary=summary,
                       artifacts=dict(labels="labels.csv", clusters="clusters_crosshalf_ward.csv", curves="k_curves.csv"), cache=cache,
                       note="Hard labels from the cross-half distance matrix. Compare cluster_crosshalf_ward with cluster_full_ward (matched).")
    lines = [f"# cross-half clustering on {fset} - {out.name}", "",
             f"{n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), {len(np.unique(pat))} patients, cache {cache['name']}; K = {K}.", "",
             f"- agreement with Ward on the ordinary full-data distance: ARI {adjusted_rand_score(lw, lf):.3f}, {100 * agree_f:.0f} % of electrodes in the matched cluster",
             f"- cross-half Ward vs cross-half k-medoids: ARI {adjusted_rand_score(lw, lm):.3f}",
             f"- subsample stability per cluster, cross-half Ward: " + ", ".join(f"c{k} {per_w.get(k, float('nan')):.2f}" for k in range(K)) + f" (full-data Ward: " + ", ".join(f"{per_f.get(k, float('nan')):.2f}" for k in range(K)) + ")",
             f"- anatomical coherence / chance: cross-half Ward {coh_w[1]:.2f}, full-data Ward {coh_f[1]:.2f}, cross-half k-medoids {coh_m[1]:.2f}",
             "", "| K | ARI cross-half Ward vs full Ward | ARI cross-half k-medoids vs full Ward | ARI Ward vs k-medoids (cross-half) |", "|---|---|---|---|"]
    for _, r in curves.iterrows():
        lines.append(f"| {int(r.k)} | {r.ari_crossward_vs_fullward:.3f} | {r.ari_crosskmed_vs_fullward:.3f} | {r.ari_crossward_vs_crosskmed:.3f} |")
    lines += ["", "| cluster | n | patients | stability | HG resp audio | picture | reading | HG stim audio | picture | reading | beta a / p / r |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in fp.iterrows():
        lines.append(f"| {int(r.cluster)} | {int(r.n)} | {int(r.patients)} | {r.stability:.2f} | {r.hg_resp_audio:+.2f} | {r.hg_resp_picture:+.2f} | {r.hg_resp_reading:+.2f} | {r.hg_stim_audio:+.2f} | {r.hg_stim_picture:+.2f} | {r.hg_stim_reading:+.2f} | {r.beta_audio:+.2f} / {r.beta_picture:+.2f} / {r.beta_reading:+.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:8])); print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
