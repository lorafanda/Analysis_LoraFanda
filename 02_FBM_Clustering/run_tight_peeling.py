#!/usr/bin/env python3
"""
run_tight_peeling.py - counter 6: tight clustering by peeling (Tseng & Wong 2005).
Fit, take out the one cluster whose members always land together, refit on the rest,
repeat. Electrodes that never join such a set stay unassigned ("scattered").

    python run_tight_peeling.py                      bands 13-170 Hz x 30 bins
    python run_tight_peeling.py --feature hg          the pipeline's concat_hg
    python run_tight_peeling.py --k0 7 --alpha 0.1 --beta 0.7 --min-size 10
    python run_tight_peeling.py --keep-muscle
    python run_tight_peeling.py --quick

THE RECIPE (Tseng & Wong, Biometrics 2005, 61:10-16)
  1. on the current pool, k-means with K = k0 on B random subsamples (70 % of the pool);
     D[i, j] = fraction of the subsamples holding both i and j in which they were together
  2. the tight sets at this K: groups in which EVERY pair has D >= 1 - alpha (complete
     linkage on 1 - D cut at alpha); the candidates are the q largest of them
  3. a candidate at K is accepted when one of the candidates at K + 1 is nearly the same
     set (Jaccard >= beta): a set that depends on K is not a cluster. Largest first; if
     none at K qualifies, K + 1 against K + 2 is tried; if still none, the largest set at K
     is taken and flagged
  4. the accepted set is the next tight cluster; its electrodes leave the pool; k0 drops by
     one; back to 1. It stops when the candidate is smaller than min_size, when K reaches 1,
     or when the pool is smaller than 2 * min_size
  5. the rest is scattered: electrodes that belong to no set that survives resampling

VALIDATION  the same peeling on the odd and on the even trials; every full-data tight
cluster is matched to the half's clusters by overlap (Jaccard), and that overlap is reported.

OUTPUT  outputs/clustering/tight/<feature>/runs/<run_id>/   labels.csv carries
        cluster_tight (0, 1, ... in peel order; -1 = scattered). Not registered in index.json.
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
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "functions"))
from functions import lf_counters as LCn                       # noqa: E402
from functions.lf_concat import concat_hg_features             # noqa: E402
import lf_decompose as D                                       # noqa: E402

METHOD = "tight"


def features(X, name):
    if name == "hg":
        return concat_hg_features(X).astype(np.float64), "concat_hg"
    F, _ = LCn.band_features(X, 13.0, 170.0, 30)
    return F, "bands13_170_30bins"


def comembership(F, k, *, B, frac, n_init, rng):
    """D[i, j]: of the subsamples that held both i and j, the fraction in which k-means put them together."""
    n = len(F); C = np.zeros((n, n)); N = np.zeros((n, n))
    for _ in range(B):
        idx = np.sort(rng.choice(n, max(int(frac * n), k + 1), replace=False))
        lab = KMeans(n_clusters=k, n_init=n_init, random_state=int(rng.integers(2**31 - 1))).fit_predict(F[idx])
        C[np.ix_(idx, idx)] += (lab[:, None] == lab[None, :]); N[np.ix_(idx, idx)] += 1
    with np.errstate(invalid="ignore"):
        Dm = np.where(N > 0, C / N, 0.0)
    np.fill_diagonal(Dm, 1.0)
    return Dm


def tight_sets(Dm, alpha, q):
    """The q largest groups in which every pair has D >= 1 - alpha, largest first."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dist = 1.0 - Dm; np.fill_diagonal(dist, 0.0)
    lab = fcluster(linkage(squareform(dist, checks=False), method="complete"), t=alpha, criterion="distance")
    ids, counts = np.unique(lab, return_counts=True)
    return [np.flatnonzero(lab == ids[i]) for i in np.argsort(-counts)[:q]]


def jaccard(a, b):
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / max(len(a | b), 1)


def peel(F, *, k0, alpha, beta, B, frac, min_size, n_init, seed, q=7, tag=""):
    rng = np.random.default_rng(seed)
    n = len(F); pool = np.arange(n); clusters = []; k = k0; D_first = None
    while k >= 2 and len(pool) >= 2 * min_size:
        Fp = F[pool]
        Ds = {kk: comembership(Fp, kk, B=B, frac=frac, n_init=n_init, rng=rng) for kk in (k, k + 1, k + 2)}
        if D_first is None:
            D_first = Ds[k]
        cands = {kk: tight_sets(Ds[kk], alpha, q) for kk in Ds}
        # Tseng & Wong: a candidate at K is stable when one of the top candidates at K + 1 is nearly the same set
        acc = None
        for kk in (k, k + 1):
            for c in cands[kk]:
                j = max([jaccard(c, c2) for c2 in cands[kk + 1]], default=0.0)
                if j >= beta and len(c) >= min_size:
                    acc = (kk, c, j, True); break
            if acc is not None:
                break
        if acc is None:
            c = cands[k][0]; acc = (k, c, max([jaccard(c, c2) for c2 in cands[k + 1]], default=0.0), False)
        kk, members, j, stable = acc
        if len(members) < min_size:
            print(f"    {tag}stop: the candidate has {len(members)} electrodes (< {min_size})", flush=True); break
        rest = np.setdiff1d(np.arange(len(pool)), members)
        outside = float(Ds[kk][np.ix_(members, rest)].mean()) if rest.size else float("nan")
        clusters.append(dict(order=len(clusters), k=int(kk), members=pool[members], jaccard_k_plus_1=j, stable=stable, outside=outside, pool=int(len(pool))))
        print(f"    {tag}tight cluster {len(clusters) - 1}: {len(members)} electrodes at K = {kk} (pool {len(pool)}), co-membership with the rest of the pool {outside:.2f}, overlap with the matching set at K + 1 {j:.2f}{'' if stable else ' (not stable: largest set taken)'}", flush=True)
        pool = np.setdiff1d(pool, pool[members]); k -= 1
    return clusters, pool, D_first


def labels_of(clusters, n):
    lab = np.full(n, -1)
    for c in clusters:
        lab[c["members"]] = c["order"]
    return lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["b13_170", "hg"], default="b13_170")
    ap.add_argument("--k0", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.1, help="a tight set: every pair together in >= 1 - alpha of the subsamples")
    ap.add_argument("--beta", type=float, default=0.7, help="accepted when the candidate at K overlaps the one at K + 1 by this Jaccard")
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--min-size", type=int, default=10)
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
    F, F1, F2 = LCn.unit_norm(F), LCn.unit_norm(F1), LCn.unit_norm(F2)
    n = len(df); pat = df["patient_id"].astype(str).to_numpy()
    B, n_init = (5, 1) if a.quick else (a.subsamples, 3)
    kw = dict(k0=a.k0, alpha=a.alpha, beta=a.beta, B=B, frac=0.7, min_size=a.min_size, n_init=n_init)
    print(f"\n=== tight clustering by peeling on {fset}: {n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), cache {cache['name']}; k0 = {a.k0}, alpha = {a.alpha}, beta = {a.beta}, {B} subsamples")
    print("  full data")
    clusters, scattered, D_first = peel(F, seed=0, tag="full: ", **kw)
    lab = labels_of(clusters, n)
    print("  odd trials"); c1, s1, _ = peel(F1, seed=1, tag="odd: ", **kw); lab1 = labels_of(c1, n)
    print("  even trials"); c2, s2, _ = peel(F2, seed=2, tag="even: ", **kw); lab2 = labels_of(c2, n)
    for c in clusters:
        c["jaccard_odd"] = max([jaccard(c["members"], h["members"]) for h in c1], default=0.0)
        c["jaccard_even"] = max([jaccard(c["members"], h["members"]) for h in c2], default=0.0)
    xyz = LCn.coords_for(df)
    assigned = lab >= 0
    coh = D.spatial_coherence(lab[assigned], xyz[assigned]) if assigned.sum() > 20 and len(np.unique(lab[assigned])) > 1 else (float("nan"), float("nan"))

    out = LCn.new_run_dir(METHOD, fset)
    lab_df = df.drop(columns=[c for c in df.columns if c.startswith("file_path_")], errors="ignore").copy()
    lab_df["cluster_tight"] = lab; lab_df["cluster_tight_odd"] = lab1; lab_df["cluster_tight_even"] = lab2; lab_df["muscle_flag"] = muscle[keep]
    lab_df.to_csv(out / "labels.csv", index=False)
    fp = LCn.fingerprints(X, lab)
    tab = pd.DataFrame([dict(cluster=c["order"], k=c["k"], pool=c["pool"], n=len(c["members"]), patients=int(len(np.unique(pat[c["members"]]))),
                             outside=c["outside"], jaccard_k_plus_1=c["jaccard_k_plus_1"], stable=c["stable"], jaccard_odd=c["jaccard_odd"], jaccard_even=c["jaccard_even"]) for c in clusters])
    if len(tab):
        tab = tab.merge(fp, on="cluster", how="left", suffixes=("", "_fp"))
    tab.to_csv(out / "clusters_tight.csv", index=False)
    np.save(out / "comembership_first_pass.npy", D_first.astype(np.float32))
    # figures
    o = np.lexsort((np.arange(n), np.where(lab < 0, 10**6, lab)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    im = axes[0].imshow(D_first[np.ix_(o, o)], cmap="magma", vmin=0, vmax=1); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title(f"first pass co-membership (K = {a.k0}), sorted by tight cluster, scattered last", loc="left", fontsize=9); fig.colorbar(im, ax=axes[0], shrink=0.8, label="fraction together")
    if len(tab):
        xx = np.arange(len(tab))
        axes[1].bar(xx - 0.25, tab.jaccard_k_plus_1, width=0.25, label="overlap with the matching set at K + 1"); axes[1].bar(xx, tab.jaccard_odd, width=0.25, label="overlap with an odd-trial cluster"); axes[1].bar(xx + 0.25, tab.jaccard_even, width=0.25, label="overlap with an even-trial cluster")
        axes[1].set_xticks(xx); axes[1].set_xticklabels([f"c{int(k)}\nn={int(m)}" for k, m in zip(tab.cluster, tab.n)], fontsize=8); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7, frameon=False); axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title(f"{len(tab)} tight clusters, {int((lab < 0).sum())} scattered of {n}", loc="left", fontsize=9)
    fig.suptitle(f"P1 - tight clustering by peeling on {fset}", x=.02, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(out / "P1_peeling.png", dpi=150); plt.close(fig)
    LCn.cluster_courses(X, lab, out / "P2_courses.png", f"P2 - mean HG per tight cluster (peel order) and the scattered rest ({fset})")
    LCn.cluster_maps(lab, xyz, out / "P3_maps.png", "P3 - tight clusters on fsaverage (grey = scattered)")

    summary = dict(n_electrodes=n, n_patients=int(len(np.unique(pat))), muscle_dropped=int(0 if a.keep_muscle else muscle.sum()), n_tight=len(clusters), n_scattered=int((lab < 0).sum()),
                   n_tight_odd=len(c1), n_tight_even=len(c2), clusters=tab.drop(columns=[c for c in tab.columns if c.startswith(("hg_", "beta_"))], errors="ignore").to_dict("records"),
                   spatial_coherence_assigned=list(map(float, coh)))
    LCn.write_manifest(out, method=METHOD, method_label="Tight clustering by peeling (Tseng & Wong 2005)", feature_set=fset, feature_set_label=fset,
                       params=dict(feature=a.feature, k0=a.k0, alpha=a.alpha, beta=a.beta, subsamples=B, frac=0.7, min_size=a.min_size, n_init=n_init, keep_muscle=a.keep_muscle, quick=a.quick),
                       summary=summary, artifacts=dict(labels="labels.csv", clusters="clusters_tight.csv", comembership="comembership_first_pass.npy"), cache=cache,
                       note="cluster_tight: peel order, -1 = scattered (in no set that survives resampling).")
    lines = [f"# tight clustering by peeling on {fset} - {out.name}", "",
             f"{n} electrodes ({int(muscle.sum())} muscle-flagged {'kept' if a.keep_muscle else 'dropped'}), {len(np.unique(pat))} patients, cache {cache['name']}; k0 = {a.k0}, alpha = {a.alpha}, beta = {a.beta}, {B} subsamples of 70 %.", "",
             f"- {len(clusters)} tight clusters hold {int(assigned.sum())} electrodes; {int((lab < 0).sum())} are scattered (odd trials alone: {len(c1)} clusters, {int((lab1 < 0).sum())} scattered; even: {len(c2)}, {int((lab2 < 0).sum())})",
             f"- anatomical coherence / chance of the assigned electrodes: {coh[1]:.2f}",
             "", "| cluster | K used | pool | n | patients | co-membership with the rest | overlap at K + 1 | overlap odd | overlap even | HG resp a / p / r | HG stim a / p / r |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {int(r.cluster)} | {int(r.k)} | {int(r.pool)} | {int(r.n)} | {int(r.patients)} | {r.outside:.2f} | {r.jaccard_k_plus_1:.2f}{'' if r.stable else ' *'} | {r.jaccard_odd:.2f} | {r.jaccard_even:.2f} | {r.hg_resp_audio:+.2f} / {r.hg_resp_picture:+.2f} / {r.hg_resp_reading:+.2f} | {r.hg_stim_audio:+.2f} / {r.hg_stim_picture:+.2f} / {r.hg_stim_reading:+.2f} |")
    lines += ["", "Every pair inside a tight cluster was together in >= 1 - alpha of the subsamples (by definition). "
              "`*` = no candidate at K matched one at K + 1 by at least beta; the largest set at K was taken anyway, so that cluster is not a Tseng-Wong cluster."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:6])); print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
