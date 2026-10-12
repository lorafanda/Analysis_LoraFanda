#!/usr/bin/env python3
"""
run_hg_timebins_test.py - does high gamma at coarser time, or with the band below it, change
what convex NMF finds?  (2026-10-12, Lora's question; convex NMF only)

THE SETS (per condition; the three conditions stitched; every electrode unit-normed before the
fit, as in 242)
    hg300                        today's concat_hg: 70-150 Hz mean per time bin, 300 bins   900 features
    hg150 / hg100 / hg60 / hg30  the same, averaged over 2 / 3 / 5 / 10 consecutive bins    450 / 300 / 180 / 90
    lowhg30                      13-70 Hz and 70-150 Hz, 30 bins each                        180
    lowhg60 / lowhg100 / lowhg150  on request (--sets)
hg300 is the newest cnmf/concat_hg run's own X_train (checked against the pipeline's
concat_hg_features), and every coarser set is a block mean of it - so "today" is in the test
exactly as it was fitted, and the only thing that changes between sets is the grid.

THE ELECTRODES: those of the newest cnmf/concat_hg run, in its order. The run must have been
fitted on the newest cache (lf_runs' rule); the script refuses otherwise. Muscle-flagged contacts
stay in, as the pipeline keeps them (--drop-muscle takes them out of every set).

THE FIT: lf_decompose.convex_nmf with the run's own K and n_iter (today K = 7, 300 iterations)
unless --k / --n-iter say otherwise. Hard label = argmax of the row-normalised loadings;
top_weight and margin as publish_decomposition.

MEASURED, per set
    vs today     ARI, NMI and Hungarian overlap of the hard labels against the run's PUBLISHED
                 labels (labels.csv) when K is the run's K, else against the sweep's refit at K
                 (cluster_labels_by_k.csv); mean correlation of the matched components on the
                 set's grid (HG part only for the lowhg sets). ari_fit_floor is the run's own
                 published fit against its sweep refit at the same K: two fits of the same
                 features, so no set can be expected to agree with today more than that
    split-half   fit the odd-trial half and the even-trial half separately; ARI between the
                 two labelings (the halves are read once per cache by lf_counters)
    stability    co-association over subsamples (80 % of the electrodes, --rounds rounds),
                 size-weighted mean over the full fit's clusters
    coherence    nearest-neighbour label agreement on fsaverage, divided by a shuffle null
    sharpness    median top_weight and the share of electrodes with top_weight > 0.5
    rank curve   (--rank-curve) held-out variance against K, to see whether the nominated K
                 moves with the grid; slow
OUTPUT  outputs/clustering/hg_timebins_test/<run_id>/
    summary.tsv              one row per set
    labels.csv               electrode keys, today's label, each set's label / top_weight / margin
    per_cluster_<set>.tsv    size, patients, stability, HG / beta fingerprints per cluster
    components_<set>.npy     (K, features)      G_<set>.npy  (n, K) row-normalised loadings
    F1_metrics.png           every metric against the number of bins
    F2_components.png        the components of every set matched to today's, HG course per condition
    F3_maps_<set>.png        the clusters on fsaverage
    manifest.json
Nothing is registered in index.json; 242 and its runs are untouched.

    python run_hg_timebins_test.py                              # the six sets, K and n_iter of today's run
    python run_hg_timebins_test.py --k 8 --sets hg300 hg60 hg30 lowhg30
    python run_hg_timebins_test.py --drop-muscle --rank-curve
    python run_hg_timebins_test.py --quick --n-sub 250          # smoke test on 250 electrodes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from functions import lf_concat as CC                                   # noqa: E402
from functions import lf_counters as LCn                                # noqa: E402
from functions import lf_decompose as D                                 # noqa: E402

OUT_ROOT = ROOT / "outputs" / "clustering" / "hg_timebins_test"
N_TIME, N_COND = 300, 3
LOW, HG = (13.0, 70.0), (70.0, 150.0)
ALL_SETS = ["hg300", "hg150", "hg100", "hg60", "hg30", "lowhg30", "lowhg60", "lowhg100", "lowhg150"]
DEFAULT_SETS = ["hg300", "hg150", "hg100", "hg60", "hg30", "lowhg30"]


# ---- features ------------------------------------------------------------------------------
def bins_of(name: str) -> int:
    return int(name.replace("lowhg", "").replace("hg", ""))


def hg_tensor(hg900: np.ndarray, bins: int) -> np.ndarray:
    """(n, 3, bins): the 900-bin HG vector (conditions stitched along time) block-averaged."""
    n = hg900.shape[0]
    step = N_TIME // bins
    assert step * bins == N_TIME, f"{bins} does not divide {N_TIME}"
    return hg900.reshape(n, N_COND, bins, step).mean(3)


def feature_tensor(name: str, X: np.ndarray, hg900: np.ndarray) -> np.ndarray:
    """(n, 3 conditions, n_bands, bins); band 0 = 13-70 Hz for the lowhg sets, HG last."""
    bins = bins_of(name)
    hg = hg_tensor(hg900, bins)[:, :, None, :]
    if not name.startswith("lowhg"):
        return hg
    low = LCn.band_tensor(X, [LOW], time_bins=bins)[:, 0]              # (n, bins, 3), mains rows kept
    low = np.transpose(low, (0, 2, 1))[:, :, None, :]                  # (n, 3, 1, bins)
    return np.concatenate([low, hg], axis=2)


def flat(T: np.ndarray) -> np.ndarray:
    return T.reshape(T.shape[0], -1).astype(np.float64)


# ---- fitting -------------------------------------------------------------------------------
def fit(F: np.ndarray, k: int, n_iter: int, prep: str, patient: np.ndarray, seed: int = 0):
    Fp = D.apply_pipeline(F, prep, patient) if prep else F
    with np.errstate(invalid="ignore"):          # lf_decompose line 139 divides inf by inf on its first iteration
        W, G, C = D.convex_nmf(Fp, k, random_state=seed, n_iter=n_iter)
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    srt = np.sort(Gn, axis=1)
    return Gn.argmax(1), Gn, C, srt[:, -1], srt[:, -1] - srt[:, -2]


def component_match(today: np.ndarray, lab: np.ndarray) -> dict:
    """{set component j: today's component i} from the Hungarian overlap of the hard labels."""
    bm, _ = LCn.match_labels(today, lab)
    out = {}
    for j in np.unique(lab):
        out[int(j)] = int(np.bincount(bm[lab == j]).argmax())
    return out


def hg_part(name: str, C: np.ndarray, k: int) -> np.ndarray:
    """(K, 3, bins): the HG band of a set's components, for the comparison with today's."""
    bins = bins_of(name)
    nb = 2 if name.startswith("lowhg") else 1
    return C.reshape(k, N_COND, nb, bins)[:, :, -1, :]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS, choices=ALL_SETS)
    ap.add_argument("--k", type=int, help="components (default: today's run)")
    ap.add_argument("--n-iter", type=int, help="iterations of the full fits (default: today's run)")
    ap.add_argument("--rounds", type=int, default=20, help="subsample rounds for the stability")
    ap.add_argument("--drop-muscle", action="store_true", help="take the speech-muscle contacts out")
    ap.add_argument("--rank-curve", action="store_true", help="held-out variance against K per set (slow)")
    ap.add_argument("--ks", default="5-12", help="K range of the rank curve, lo-hi")
    ap.add_argument("--n-sub", type=int, help="smoke test: a random subset of this many electrodes")
    ap.add_argument("--quick", action="store_true", help="few rounds, few iterations")
    ap.add_argument("--allow-stale", action="store_true", help="accept a run fitted on an older cache")
    ap.add_argument("--seed", type=int, default=0, help="random_state of the fits (today's run: 0)")
    a = ap.parse_args()
    t_start = time.time()

    # ---- today's run, the cohort in its order ------------------------------------------
    df, X, cache = LCn.load_cohort(verbose=not a.quick)
    run = LCn.newest_cnmf("concat_hg")
    if run is None:
        print("no cnmf/concat_hg run with G_loadings / components / labels.csv")
        return 1
    man = json.loads((run["run"] / "manifest.json").read_text(encoding="utf-8"))
    run_cache = (man.get("cache") or {}).get("name", "?")
    if run_cache != cache["name"] and not a.allow_stale:
        print(f"the newest run {run['run'].name} was fitted on {run_cache}, the newest cache is {cache['name']}: "
              f"refit 242 first, or --allow-stale")
        return 1
    keys = list(df["patient_id"].astype(str) + "|" + df["contact_norm"].astype(str))
    pos = {k: i for i, k in enumerate(keys)}
    missing = [k for k in run["keys"] if k not in pos]
    if missing:
        print(f"{len(missing)} electrodes of the run are not in the cohort (e.g. {missing[:3]}): stop")
        return 1
    order = np.array([pos[k] for k in run["keys"]])
    df, X = df.iloc[order].reset_index(drop=True), X[order]
    n_all = len(df)
    K = a.k or int(run["G"].shape[1])
    n_iter = a.n_iter or int((man.get("params") or {}).get("n_iter", 300))
    prep_txt = str((man.get("params") or {}).get("preprocessing", "unit_norm"))   # the run writes free text
    prep = prep_txt if prep_txt in ("raw", "unit_norm", "patient_z", "shape_patient_z") else ("unit_norm" if "unit" in prep_txt else "raw")
    n_iter_part = 60 if a.quick else max(100, n_iter // 2)
    rounds = 3 if a.quick else a.rounds

    from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
    # today's labels: the published fit (labels.csv) at the run's K, the sweep's refit otherwise
    pub = pd.read_csv(run["run"] / "labels.csv")
    by_k = pd.read_csv(run["run"] / "cluster_labels_by_k.csv")
    pub_col = [c for c in pub.columns if c.startswith("cluster_")]
    K_run = int(run["G"].shape[1])
    if K == K_run and pub_col:
        today, today_src = pub[pub_col[0]].to_numpy(int), f"labels.csv {pub_col[0]} (the published fit)"
    elif f"k_{K}" in by_k:
        today, today_src = by_k[f"k_{K}"].to_numpy(int), f"cluster_labels_by_k.csv k_{K} (the sweep's refit)"
    else:
        today, today_src = None, "none"
        print(f"today's run has no labels at K = {K}: the 'vs today' columns stay empty")
    floor = float(ARI(pub[pub_col[0]], by_k[f"k_{K_run}"])) if pub_col and f"k_{K_run}" in by_k else float("nan")
    print(f"[today] labels: {today_src}; fit-to-fit floor (published fit vs sweep refit at K = {K_run}): ARI {floor:.3f}")

    # ---- hg300 = the run's own features ----------------------------------------------------
    hg900 = CC.concat_hg_features(X).astype(np.float64)                 # (n, 900)
    xtr = run["run"] / "X_train.npy"
    if xtr.exists():
        Xt = np.load(xtr).astype(np.float64)
        if Xt.shape == hg900.shape and np.allclose(Xt, hg900, atol=1e-4):
            print("[check] concat_hg_features reproduces the run's X_train: hg300 is today's feature set")
        else:
            print(f"[check] WARNING: the run's X_train differs from concat_hg_features (max |d| = "
                  f"{np.abs(Xt - hg900).max() if Xt.shape == hg900.shape else 'shape'}); using X_train as hg300")
            if Xt.shape == hg900.shape:
                hg900 = Xt

    # ---- the halves, read for the whole run cohort so the cached file serves every later run ----
    X1, X2 = LCn.load_halves(df, cache["name"], verbose=not a.quick)

    # ---- electrode selection ------------------------------------------------------------------
    keep = np.ones(n_all, bool)
    muscle = LCn.muscle_flags(X)
    if a.drop_muscle:
        keep &= ~muscle
    if a.n_sub and a.n_sub < keep.sum():
        rng = np.random.default_rng(0)
        chosen = rng.choice(np.flatnonzero(keep), a.n_sub, replace=False)
        keep[:] = False
        keep[np.sort(chosen)] = True
    sel = np.flatnonzero(keep)
    df, X, hg900 = df.iloc[sel].reset_index(drop=True), X[sel], hg900[sel]
    X1, X2 = X1[sel], X2[sel]
    today = today[sel] if today is not None else None
    n = len(df)
    patient = df["patient_id"].astype(str).to_numpy()
    xyz = LCn.coords_for(df)
    print(f"[cohort] {cache['name']}: {n} electrodes of {n_all} ({int(muscle.sum())} muscle-flagged, "
          f"{'dropped' if a.drop_muscle else 'kept'}), run {run['run'].name}, K = {K}, n_iter = {n_iter}, prep = {prep}")

    # ---- the halves of the selected electrodes ------------------------------------------------
    ok_half = ~(np.isnan(X1).any((1, 2)) | np.isnan(X2).any((1, 2)))
    hg1 = CC.concat_hg_features(X1[ok_half]).astype(np.float64)
    hg2 = CC.concat_hg_features(X2[ok_half]).astype(np.float64)

    # ---- run -------------------------------------------------------------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    ks = list(range(int(a.ks.split("-")[0]), int(a.ks.split("-")[1]) + 1))
    labels = pd.DataFrame({"patient_id": df["patient_id"], "contact_norm": df["contact_norm"], "electrode": df["electrode"]})
    if today is not None:
        labels["today"] = today
    rows, comps, courses = [], {}, {}
    today_hg = run["C"][:, :].reshape(K, N_COND, N_TIME) if run["C"].shape[0] == K else None
    for name in a.sets:
        t0 = time.time()
        T = feature_tensor(name, X, hg900)
        F = flat(T)
        lab, Gn, C, top, margin = fit(F, K, n_iter, prep, patient, seed=a.seed)
        t_fit = time.time() - t0
        rec = dict(set=name, bins=bins_of(name), bands=T.shape[2], features=F.shape[1], K=K, n=n, seed=a.seed, ari_fit_floor=round(floor, 3), fit_s=round(t_fit, 1))
        # vs today
        if today is not None:
            bm, agree = LCn.match_labels(today, lab)
            rec.update(ari_today=ARI(today, lab), nmi_today=NMI(today, lab), overlap_today=agree)
            if today_hg is not None:
                m = component_match(today, lab)
                Ct = hg_tensor(today_hg.reshape(K, -1), bins_of(name))       # (K, 3, bins)
                Cs = hg_part(name, C, K)
                rec["comp_r_today"] = float(np.mean([np.corrcoef(Cs[j].ravel(), Ct[i].ravel())[0, 1] for j, i in m.items()]))
        # split-half
        T1 = feature_tensor(name, X1[ok_half], hg1)
        T2 = feature_tensor(name, X2[ok_half], hg2)
        l1 = fit(flat(T1), K, n_iter_part, prep, patient[ok_half], seed=a.seed + 1)[0]
        l2 = fit(flat(T2), K, n_iter_part, prep, patient[ok_half], seed=a.seed + 2)[0]
        rec.update(ari_halves=ARI(l1, l2), nmi_halves=NMI(l1, l2), n_halves=int(ok_half.sum()))
        # stability over subsamples
        per, _ = LCn.subsample_stability(lambda idx: fit(F[idx], K, n_iter_part, prep, patient[idx], seed=a.seed)[0], n, lab, n_rounds=rounds, frac=0.8, seed=a.seed)
        sizes = {int(k): int((lab == k).sum()) for k in np.unique(lab)}
        rec["stability"] = float(sum(per[k] * sizes[k] for k in per if np.isfinite(per[k])) / max(sum(sizes[k] for k in per if np.isfinite(per[k])), 1))
        rec["stability_min"] = float(np.nanmin(list(per.values())))
        # coherence, sharpness
        obs, ratio = D.spatial_coherence(lab, xyz)
        rec.update(coherence=obs, coherence_x_chance=ratio, top_weight_median=float(np.median(top)),
                   share_top_gt_0p5=float((top > 0.5).mean()), n_clusters_used=int(len(np.unique(lab))))
        if a.rank_curve:
            cv = D.cv_rank_curve(D.apply_pipeline(F, prep, patient) if prep else F, ks, n_folds=5, n_iter=n_iter_part)
            cv.to_csv(out / f"rank_curve_{name}.tsv", sep="\t", index=False)
            mean = cv.groupby("k")["var_explained"].mean()
            rec.update(rank_peak_k=int(mean.idxmax()), rank_var_at_K=float(mean.get(K, np.nan)))
        rec["total_s"] = round(time.time() - t0, 1)
        rows.append(rec)
        # per-set outputs
        labels[f"{name}_label"] = lab
        labels[f"{name}_top_weight"] = np.round(top, 4)
        labels[f"{name}_margin"] = np.round(margin, 4)
        np.save(out / f"components_{name}.npy", C.astype(np.float32))
        np.save(out / f"G_{name}.npy", Gn.astype(np.float32))
        fp = LCn.fingerprints(X, lab)
        fp["stability"] = fp.cluster.map(per)
        fp["patients"] = [df.loc[lab == k, "patient_id"].nunique() for k in fp.cluster]
        fp.to_csv(out / f"per_cluster_{name}.tsv", sep="\t", index=False)
        LCn.cluster_maps(lab, xyz, out / f"F3_maps_{name}.png", f"F3 - {name}, convex NMF K = {K}: clusters on fsaverage")
        comps[name] = (C, component_match(today, lab) if today is not None else {int(j): int(j) for j in range(K)})
        print(f"[{name:<9}] {F.shape[1]:>4} feats  ari_today {rec.get('ari_today', float('nan')):.3f}  halves {rec['ari_halves']:.3f}  "
              f"stability {rec['stability']:.3f}  coherence x{rec['coherence_x_chance']:.2f}  top_w {rec['top_weight_median']:.2f}  {rec['total_s']:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "summary.tsv", sep="\t", index=False)
        labels.to_csv(out / "labels.csv", index=False)

    summary = pd.DataFrame(rows)
    draw_metrics(summary, out / "F1_metrics.png", K, cache["name"])
    draw_components(comps, a.sets, K, out / "F2_components.png", today_hg)
    manifest = dict(run_id=run_id, question="HG at coarser time, and with the band below, under convex NMF",
                    cache=cache, compared_run=str(run["run"]), K=K, n_iter=n_iter, n_iter_partial=n_iter_part,
                    preprocessing=prep, sets=a.sets, n_electrodes=n, muscle_dropped=bool(a.drop_muscle),
                    compared_labels=today_src, ari_fit_floor=floor, seed=a.seed,
                    n_sub=a.n_sub, rounds=rounds, rank_curve=bool(a.rank_curve), ks=ks if a.rank_curve else None,
                    seconds=round(time.time() - t_start), built=datetime.now().strftime("%Y-%m-%d %H:%M"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=LCn._json_default), encoding="utf-8")
    print(f"\n{summary.drop(columns=['fit_s', 'total_s'], errors='ignore').round(3).to_string(index=False)}\n-> {out}  ({manifest['seconds']} s)")
    return 0


# ---- figures ------------------------------------------------------------------------------
def draw_metrics(s: pd.DataFrame, path: Path, K: int, cache_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("ari_today", "ARI with today's labels"), ("comp_r_today", "matched-component r with today"),
              ("ari_halves", "split-half ARI (odd vs even trials)"), ("stability", "subsample co-association"),
              ("coherence_x_chance", "anatomical coherence, x chance"), ("top_weight_median", "median top weight")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    hg = s[~s.set.str.startswith("lowhg")]
    lo = s[s.set.str.startswith("lowhg")]
    for ax, (col, title) in zip(axes.ravel(), panels):
        if col not in s:
            ax.set_visible(False)
            continue
        ax.plot(hg.bins, hg[col], "o-", color="#c1121f", label="HG only")
        if len(lo):
            ax.plot(lo.bins, lo[col], "s", color="#1f77b4", ms=8, label="13-70 + HG")
        for _, r in s.iterrows():
            ax.annotate(r.set, (r.bins, r[col]), fontsize=7, xytext=(3, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xticks([30, 60, 100, 150, 300])
        ax.set_xticklabels([30, 60, 100, 150, 300])
        ax.set_xlabel("time bins per condition")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"F1 - convex NMF at K = {K} on {cache_name}: the grid against the metrics", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_components(comps: dict, sets: list, K: int, path: Path, today_hg):
    """Rows = today's components (by matching), columns = sets; the HG course per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#c1121f", "#1f77b4", "#2a9d8f"]
    fig, axes = plt.subplots(K, len(sets), figsize=(2.6 * len(sets), 1.5 * K), squeeze=False, sharey="row")
    for j, name in enumerate(sets):
        C, m = comps[name]
        bins = bins_of(name)
        Cs = hg_part(name, C, K)
        low = C.reshape(K, N_COND, 2, bins)[:, :, 0, :] if name.startswith("lowhg") else None
        inv = {}
        for jj, ii in m.items():
            inv.setdefault(ii, jj)
        for i in range(K):
            ax = axes[i, j]
            jj = inv.get(i)
            if jj is None:
                ax.axis("off")
                continue
            t = (np.arange(bins) + 0.5) / bins * 100
            if today_hg is not None and name != "hg300":
                tt = (np.arange(N_TIME) + 0.5) / N_TIME * 100
                for c in range(N_COND):
                    ax.plot(tt, today_hg[i, c] / max(np.abs(today_hg[i]).max(), 1e-9), color="#bbb", lw=0.8)
            for c in range(N_COND):
                ax.plot(t, Cs[jj, c] / max(np.abs(Cs[jj]).max(), 1e-9), color=cols[c], lw=1.2)
                if low is not None:
                    ax.plot(t, low[jj, c] / max(np.abs(Cs[jj]).max(), 1e-9), color=cols[c], lw=0.8, ls="--")
            ax.axvline(50, color="k", lw=0.5)
            ax.axhline(0, color="k", lw=0.3)
            ax.set_xticks([0, 50, 100])
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"today's {i}", fontsize=8)
    fig.suptitle("F2 - the components of every set, matched to today's (grey: today's hg300 course)\n"
                 "audio red, picture blue, reading green; dashed: the 13-70 Hz band of the lowhg sets; each component scaled to its own HG peak",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
