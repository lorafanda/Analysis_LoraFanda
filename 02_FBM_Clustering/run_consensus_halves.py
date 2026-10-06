#!/usr/bin/env python3
"""
run_consensus_halves.py - counter 5: consensus over halves x feature sets x algorithms.
Only pairs of electrodes that stay together are reported as together.

    python run_consensus_halves.py                 K = 7
    python run_consensus_halves.py --k 5
    python run_consensus_halves.py --keep-muscle
    python run_consensus_halves.py --quick

WHAT IS POOLED. Partitions at the same K from every combination of
  data        the full trial average, the odd half, the even half
  features    bands 13-170 Hz x 30 bins, the pipeline's concat_hg, concat_bands5z
  algorithm   k-means (unit-norm, Euclidean), Ward (unit-norm), k-medoids (1 - r)
  seeds       several for the two randomised ones
into one co-association matrix: C[i, j] = the fraction of partitions in which i and j land
together. lf_decompose.pipeline_consensus does the same over preprocessing pipelines; this
one adds the halves and the feature sets, which is where the trial noise and the
feature-set choice enter.

WHAT IS REPORTED
  the share of electrode pairs that stay together in >= 80 % of the partitions, and the
  share never together in more than 20 %
  per electrode, its stable partners: how many others it stays with in >= 80 % of the
  partitions; an electrode with none belongs to nothing that survives the pooling, and is listed
  consensus clusters (average linkage on 1 - C, cut at K): sizes, mean co-association
  inside, patient spread, fingerprints, anatomical coherence, maps

OUTPUT  outputs/clustering/consensus/K<k>/runs/<run_id>/   labels.csv carries
        cluster_consensus and stable_partners_80. Not registered in index.json.
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
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "functions"))
from functions import lf_counters as LCn                                                     # noqa: E402
from functions.lf_concat import concat_hg_features, concat_bands5z_features                  # noqa: E402
import lf_decompose as D                                                                     # noqa: E402

METHOD = "consensus"


def feature_sets(X):
    F, _ = LCn.band_features(X, 13.0, 170.0, 30)
    return {"bands13_170": F, "concat_hg": concat_hg_features(X).astype(np.float64), "concat_bands5z": concat_bands5z_features(X).astype(np.float64)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--keep-muscle", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    K = a.k

    df, X, cache = LCn.load_cohort(verbose=not a.quick)
    X1, X2 = LCn.load_halves(df, cache["name"], verbose=not a.quick)
    X1, X2 = np.nan_to_num(X1), np.nan_to_num(X2)
    muscle = LCn.muscle_flags(X)
    keep = np.ones(len(df), bool) if a.keep_muscle else ~muscle
    df, X, X1, X2 = df[keep].reset_index(drop=True), X[keep], X1[keep], X2[keep]
    n = len(df); pat = df["patient_id"].astype(str).to_numpy()
    print(f"\n=== consensus at K = {K}: {n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), cache {cache['name']}")
    views = {"full": feature_sets(X), "half1": feature_sets(X1), "half2": feature_sets(X2)}
    if a.quick:
        views = {"full": {"bands13_170": views["full"]["bands13_170"]}, "half1": {"bands13_170": views["half1"]["bands13_170"]}, "half2": {"bands13_170": views["half2"]["bands13_170"]}}
    seeds = range(1 if a.quick else a.seeds)
    C = np.zeros((n, n)); runs = []
    for vname, fs in views.items():
        for fname, F in fs.items():
            Fu = LCn.unit_norm(F); Dc = LCn.corr_distance(F)
            parts = {}
            for s in seeds:
                parts[f"kmeans_s{s}"] = KMeans(n_clusters=K, n_init=10, random_state=s).fit_predict(Fu)
                parts[f"kmedoids_s{s}"] = LCn.kmedoids(Dc, K, n_init=3 if a.quick else 10, seed=s)[0]
            parts["ward"] = LCn.ward_on_distance(squareform(pdist(Fu)), K)
            for pname, lab in parts.items():
                C += (lab[:, None] == lab[None, :]); runs.append(f"{vname}/{fname}/{pname}")
            print(f"    {vname:<6} {fname:<15} {len(parts)} partitions", flush=True)
    C /= len(runs)
    np.fill_diagonal(C, 1.0)
    iu = np.triu_indices(n, 1)
    together = float((C[iu] >= 0.8).mean()); apart = float((C[iu] <= 0.2).mean())
    Coff = C.copy(); np.fill_diagonal(Coff, np.nan)
    n80 = (Coff >= 0.8).sum(1)                                   # stable partners: together in >= 80 % of the partitions
    lab = D.consensus_labels(C, K)
    inside = {}
    for k in range(K):
        m = np.flatnonzero(lab == k)
        inside[k] = float(np.nanmean(Coff[np.ix_(m, m)])) if m.size > 1 else float("nan")
    xyz = LCn.coords_for(df); coh = D.spatial_coherence(lab, xyz)
    fp = LCn.fingerprints(X, lab); fp["patients"] = [df.loc[lab == k, "patient_id"].nunique() for k in fp.cluster]; fp["inside_coassociation"] = fp.cluster.map(inside)

    out = LCn.new_run_dir(METHOD, f"K{K}")
    lab_df = df.drop(columns=[c for c in df.columns if c.startswith("file_path_")], errors="ignore").copy()
    lab_df["cluster_consensus"] = lab; lab_df["stable_partners_80"] = n80; lab_df["muscle_flag"] = muscle[keep]
    lab_df.to_csv(out / "labels.csv", index=False); fp.to_csv(out / "clusters_consensus.csv", index=False)
    np.save(out / "coassociation.npy", C.astype(np.float32)); (out / "partitions.txt").write_text("\n".join(runs), encoding="utf-8")
    o = np.argsort(lab)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    im = axes[0].imshow(C[np.ix_(o, o)], cmap="magma", vmin=0, vmax=1); axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].set_title(f"co-association over {len(runs)} partitions, sorted by consensus cluster", loc="left", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="fraction together")
    axes[1].hist(n80, bins=40, color="#4a6fa5"); axes[1].set_xlabel("stable partners: other electrodes it stays with in >= 80 % of the partitions"); axes[1].set_ylabel("electrodes"); axes[1].set_title(f"{int((n80 == 0).sum())} electrodes have none", loc="left", fontsize=9); axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"S1 - consensus over halves x feature sets x algorithms, K = {K}", x=.02, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(out / "S1_coassociation.png", dpi=150); plt.close(fig)
    LCn.cluster_courses(X, lab, out / "S2_courses.png", f"S2 - mean HG per consensus cluster, K = {K}")
    LCn.cluster_maps(lab, xyz, out / "S3_maps.png", f"S3 - consensus clusters on fsaverage, K = {K}")
    loose = lab_df.loc[n80 == 0, ["patient_id", "electrode", "cluster_consensus"]]
    loose.to_csv(out / "electrodes_without_a_stable_partner.csv", index=False)

    summary = dict(n_electrodes=n, n_patients=int(len(np.unique(pat))), muscle_dropped=int(0 if a.keep_muscle else muscle.sum()), K=K, n_partitions=len(runs),
                   pairs_together_80=together, pairs_apart_20=apart, electrodes_without_stable_partner=int((n80 == 0).sum()), median_stable_partners=float(np.median(n80)),
                   inside_coassociation=inside, sizes=[int((lab == k).sum()) for k in range(K)], spatial_coherence=list(map(float, coh)))
    LCn.write_manifest(out, method=METHOD, method_label="Consensus over halves x feature sets x algorithms", feature_set=f"K{K}", feature_set_label=f"consensus at K = {K}",
                       params=dict(k=K, seeds=len(list(seeds)), keep_muscle=a.keep_muscle, quick=a.quick, partitions=runs), summary=summary,
                       artifacts=dict(labels="labels.csv", clusters="clusters_consensus.csv", coassociation="coassociation.npy"), cache=cache,
                       note="cluster_consensus = average linkage on 1 - C cut at K; stable_partners_80 = how many other electrodes it stays with in >= 80 % of the partitions (0 = belongs to nothing that survives the pooling).")
    lines = [f"# consensus at K = {K} - {out.name}", "",
             f"{n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), {len(np.unique(pat))} patients, cache {cache['name']}; {len(runs)} partitions pooled.", "",
             f"- electrode pairs together in >= 80 % of partitions: {100 * together:.1f} %; apart in >= 80 %: {100 * apart:.1f} %; the rest ({100 * (1 - together - apart):.1f} %) depends on the choice of data, features or algorithm",
             f"- stable partners per electrode (others it stays with in >= 80 % of the partitions): median {int(np.median(n80))}; {int((n80 == 0).sum())} of {n} have none (electrodes_without_a_stable_partner.csv)",
             f"- anatomical coherence / chance of the consensus clusters: {coh[1]:.2f}",
             "", "| cluster | n | patients | co-association inside | HG resp a / p / r | HG stim a / p / r | beta a / p / r |", "|---|---|---|---|---|---|---|"]
    for _, r in fp.iterrows():
        lines.append(f"| {int(r.cluster)} | {int(r.n)} | {int(r.patients)} | {r.inside_coassociation:.2f} | {r.hg_resp_audio:+.2f} / {r.hg_resp_picture:+.2f} / {r.hg_resp_reading:+.2f} | {r.hg_stim_audio:+.2f} / {r.hg_stim_picture:+.2f} / {r.hg_stim_reading:+.2f} | {r.beta_audio:+.2f} / {r.beta_picture:+.2f} / {r.beta_reading:+.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:7])); print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
