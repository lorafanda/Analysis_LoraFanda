#!/usr/bin/env python3
"""
consensus_stability.py - how stable are the clusters when PATIENTS are resampled?

Consensus clustering (Monti et al. 2003) with the PAC / RCSI corrections of Senbabaoglu,
Michailidis & Li (2014), for every cell of the clustering (feature set x method) on the
CURRENT cohort cache, beside the three K criteria of choose_k.py and the per-cluster bootstrap of
measure_cluster_stability.py (which resamples electrodes; this resamples patients, so the two
answer different questions).

    python consensus_stability.py --estimate                  # time one fit per cell, project the total
    python consensus_stability.py --pilot                     # one cell end to end (concat_bands15 x kmeans)
    python consensus_stability.py                             # all 12 cells, then the cross-cell summary
    python consensus_stability.py --summary-only              # rebuild summary_cells.csv / SUMMARY.md from what exists
    python consensus_stability.py --selftest <dir>            # synthetic data, every function once, nothing of the repo touched

WHAT IS COMPUTED, per cell (feature set x method) and per K in --ks (default 2..12)
  1. CONSENSUS MATRIX. H iterations (default 200). Each samples 80 % of the PATIENTS without
     replacement, keeps all their electrodes, and fits the cell's method on them (MS.fit_any: the
     same fitters as everywhere, unit-norm space, k-means n_init=10, Ward, convex NMF with the
     argmax of the row-normalised loadings). k-means and cNMF get a fresh seed per iteration; Ward
     is deterministic, its only variation is the subsampling, which is what is wanted. Electrodes
     are NEVER resampled individually: contacts on one shaft share a reference and electrode-level
     resampling would inflate consensus.  M(i,j) = co-clustered / co-sampled, over the iterations
     in which both were sampled. Stored as float16 (M_kNN.npy), with the minimum co-sample count.
  2. K SELECTION. PAC(K) = share of off-diagonal pairs with 0.1 < M < 0.9; the consensus CDF and
     Monti's area A(K) and Delta(K), plus the running-max variant Delta^(K) that Senbabaoglu
     recommends when A(K) is not monotone (it is not for k-means and cNMF). NULL: the covariance-
     matched Gaussian surrogate of measure_cluster_stability.surrogate (one blob, the data's own
     covariance, the same generator as the separation null), one draw per (feature set, draw),
     shared by the three methods of that feature set, with the real patient labels attached, run
     through the identical consensus procedure (default 25 draws x 50 iterations; PAC is a global
     statistic and does not need a fine M). RCSI(K) = log(mean PAC_null) - log(PAC_real);
     p = (1 + #{PAC_null <= PAC_real}) / (1 + draws). Chosen K = argmax RCSI among the Ks whose
     consensus partition passes the patient-dominance constraint (no cluster with more than half
     of its electrodes from one patient); the unconstrained argmax is reported too.
  3. CONSENSUS PARTITION. Average linkage on 1 - M, cut at K (lf_decompose.consensus_labels), at
     the chosen K and at the current K of the feature set (bsf_comparison/peak_k.json). ARI against
     the published partition of the newest run (cluster_labels_by_k.csv / loadings_by_k) and
     against the refit reference the statistics script uses (MS.fit_any at seed 42). For k-means
     and cNMF the whole consensus is rerun with a second master seed (real data only) and the ARI
     of the two consensus partitions is reported; below 0.9 the cell is flagged "raise H".
     Cross-method ARI of the consensus partitions at the current K, beside the single-fit values in
     statistics/<fset>_K<K>/stats_agreement.csv.
  4. STABILITY SCORES. Cluster consensus m(k) (mean M over within-cluster pairs) with sizes and a
     rank; item consensus m_i(k) per electrode with own-cluster, best-other-cluster and margin
     (items_kNN.csv); per-patient mean own-cluster consensus and the share below 0.5
     (patients_kNN.csv) with the rank of PAT_3455, PAT_3965, PAT_3975, PAT_5515, PAT_5533, EL046,
     and the Spearman correlation with the LOPO placement accuracy of stats_lopo_predict.csv when
     the statistics exist at that K. Heatmaps at both Ks, rows and columns in optimal leaf order
     of the consensus dendrogram, with a patient strip and a cluster strip.
  5. CROSS-CELL. summary_cells.csv, rcsi_by_k.png (4 x 3), SUMMARY.md with everything above and
     whether the feature-set order of the existing tests holds on RCSI and mean cluster consensus.
  6. VERIFICATION, written to verification.json per cell and repeated in SUMMARY.md: M symmetric
     in [0, 1]; minimum co-samples; the md5 of X_train identical across the three runs of a feature
     set (the statistics script's check); one surrogate draw drawn as a heatmap (null_sanity_kNN.png,
     it should show no blocks and a high PAC); and the md5 of every existing statistics file and
     run file read here, before and after, so nothing published changed.

OUTPUTS  outputs/clustering/consensus/<cache>/<feature_set>/<method>/ (per cell) and
         outputs/clustering/consensus/<cache>/ (cross-cell). Nothing is written anywhere else.
RESUMABLE: a (cell, K) whose M_kNN.npy exists and a null draw whose json exists are skipped
(--force recomputes). Iterations run in parallel with joblib (--jobs, default all cores).
COST: 12 cells x 11 Ks x (200 + 25 x 50) fits; cNMF is the expensive one. Run --estimate first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
sys.path.insert(0, str(ROOT))

CLUST = ROOT / "outputs" / "clustering"
FSETS = ("concat_bands15", "concat_bands5", "concat_bands2", "concat_hfa")
METHODS = ("kmeans", "hierarchical", "cnmf")
LABEL = {"kmeans": "k-means", "hierarchical": "Ward", "cnmf": "convex NMF"}
FSET_LABEL = {"concat_bands15": "15 bands", "concat_bands5": "5 bands", "concat_bands2": "2 bands", "concat_hfa": "HFA"}
PAC_LO, PAC_HI = 0.1, 0.9
DOMINANCE = 0.5                    # a cluster with more than this share from one patient fails the constraint
NAMED = ("PAT_3455", "PAT_3965", "PAT_3975", "PAT_5515", "PAT_5533", "EL046")
UNSTABLE = 0.6                     # m(k) below this is flagged (a convention, stated as such)
SEED_RERUN_OK = 0.9
_SELFTEST = False                  # set by --selftest: no repo modules, synthetic fitters


# ------------------------------------------------------------------ repo access (lazy: the self-test needs none of it)
def _mods():
    import lf_runs as LR
    import lf_decompose as D
    import measure_cluster_stability as MS
    return LR, D, MS


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def fit_any(A, K, seed, method):
    """Labels at K by the cell's method, on the unit-norm matrix. The repo's fitters, or the
    self-test's stand-ins."""
    if _SELFTEST:
        from sklearn.cluster import KMeans, AgglomerativeClustering
        if method == "hierarchical":
            return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(A)
        return KMeans(K, n_init=3, random_state=seed).fit_predict(A)
    return _mods()[2].fit_any(A, K, seed, method)


def surrogate(X, seed):
    if _SELFTEST:
        Xc = X - X.mean(0)
        S, Vt = np.linalg.svd(Xc, full_matrices=False)[1:]
        Z = np.random.default_rng(seed).standard_normal((len(X), len(S)))
        return (Z * (S / np.sqrt(len(X) - 1))) @ Vt
    return _mods()[2].surrogate(X, seed)


def consensus_labels(M, K):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    Dm = 1.0 - M.astype(np.float64)
    np.fill_diagonal(Dm, 0.0)
    Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    Z = linkage(squareform(Dm, checks=False), method="average")
    return fcluster(Z, t=K, criterion="maxclust") - 1, Z


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shaft_of(name: str) -> str:
    return re.sub(r"\d+$", "", str(name))


def norm(s) -> str:
    return str(s).replace("_", "").replace("-", "").upper()


def peak_k() -> dict:
    p = CLUST / "bsf_comparison" / "peak_k.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------ the cohort of one feature set
def load_cell(fset: str) -> dict:
    """X_train of the newest current-cache runs of the three methods, proven identical (the
    statistics script's check), with patients, electrodes, shafts and the md5 of X."""
    LR = _mods()[0]
    runs = {m: LR.newest_run(m, fset, require_files=("X_train.npy", "labels.csv")) for m in METHODS}
    X = np.load(runs["kmeans"] / "X_train.npy").astype(np.float64)
    meta = pd.read_csv(runs["kmeans"] / "labels.csv")
    keys = [f"{p}|{norm(e)}" for p, e in zip(meta["patient_id"], meta["electrode"])]
    md5s = {}
    for m in METHODS:
        Xm = np.load(runs[m] / "X_train.npy").astype(np.float64)
        mm = pd.read_csv(runs[m] / "labels.csv")
        km = [f"{p}|{norm(e)}" for p, e in zip(mm["patient_id"], mm["electrode"])]
        if Xm.shape != X.shape or not np.allclose(Xm, X, atol=1e-6):
            raise SystemExit(f"{fset}/{m}: X_train differs from the k-means run - not the same electrodes")
        if km != keys:
            raise SystemExit(f"{fset}/{m}: electrode order differs from the k-means run")
        md5s[m] = md5_bytes(np.ascontiguousarray(Xm).tobytes())
    return dict(fset=fset, X=X, A=unit(X), pats=meta["patient_id"].astype(str).to_numpy(),
                elec=meta["electrode"].astype(str).to_numpy(),
                shafts=np.array([shaft_of(e) for e in meta["electrode"]]),
                keys=keys, runs=runs, md5=md5s, cache=LR.current_cache())


def published_labels(method: str, run: Path, n: int, k: int):
    """The partition the newest run published at K, or None. Read only: a cNMF K without stored
    loadings is reported as missing, never refitted into the run."""
    if method == "cnmf":
        f = run / "loadings_by_k" / f"G_k{k:02d}.npy"
        if f.exists():
            G = np.load(f)
            if G.shape == (n, k):
                return (G / np.maximum(G.sum(1, keepdims=True), 1e-12)).argmax(1)
    f = run / "cluster_labels_by_k.csv"
    if f.exists():
        df = pd.read_csv(f)
        col = f"k_{k}"
        if col in df.columns and len(df) == n:
            lab = df[col].to_numpy()
            return lab - lab.min()
    return None


# ------------------------------------------------------------------ consensus
def _one_iteration(A, idx_by_pat, n_keep, K, method, seed):
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(idx_by_pat), size=n_keep, replace=False)
    idx = np.concatenate([idx_by_pat[p] for p in sorted(keep)])
    return idx, fit_any(A[idx], K, int(seed), method).astype(np.int16)


def consensus_matrix(A, pats, K, method, H, seed0, frac, jobs):
    """(M float32, co-sample counts int32). Patients are the sampling unit."""
    from joblib import Parallel, delayed
    ups = np.unique(pats)
    idx_by_pat = [np.flatnonzero(pats == p) for p in ups]
    n_keep = max(2, int(round(frac * len(ups))))
    tasks = (delayed(_one_iteration)(A, idx_by_pat, n_keep, K, method, seed0 + h) for h in range(H))
    try:                                   # one BLAS thread per worker, or every fit oversubscribes the cores
        from joblib import parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=jobs)(tasks)
    except ImportError:                    # older joblib
        res = Parallel(n_jobs=jobs, prefer="processes")(tasks)
    n = len(A)
    co = np.zeros((n, n), np.int32)
    tog = np.zeros((n, n), np.int32)
    for idx, lab in res:
        ix = np.ix_(idx, idx)
        tog[ix] += 1
        co[ix] += (lab[:, None] == lab[None, :])
    M = (co / np.maximum(tog, 1)).astype(np.float32)
    M[tog == 0] = 0.0
    return M, tog


def pac_of(M):
    off = M[np.triu_indices(len(M), 1)].astype(np.float64)
    return float(((off > PAC_LO) & (off < PAC_HI)).mean())


def cdf_area(M, bins=100):
    """The consensus CDF on the upper triangle and Monti's area A(K) under it."""
    off = M[np.triu_indices(len(M), 1)].astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    h, _ = np.histogram(off, bins=edges)
    cdf = np.cumsum(h) / max(len(off), 1)
    area = float(np.sum(np.diff(edges) * cdf))
    return edges[1:], cdf, area


def delta_curves(areas: dict) -> dict:
    """Monti's Delta(K) and the running-max version Delta^(K) (Senbabaoglu 2014) from A(K)."""
    ks = sorted(areas)
    out = {}
    run_max = None
    for i, k in enumerate(ks):
        a = areas[k]
        if i == 0:
            out[k] = dict(delta=a, delta_hat=a)
            run_max = a
        else:
            prev = areas[ks[i - 1]]
            out[k] = dict(delta=(a - prev) / prev if prev > 0 else float("nan"),
                          delta_hat=(a - run_max) / run_max if run_max > 0 else float("nan"))
            run_max = max(run_max, a)
    return out


def dominance(labels, pats):
    """Per cluster the largest share held by one patient; the constraint fails above DOMINANCE."""
    rows = []
    for c in np.unique(labels):
        sel = labels == c
        u, cnt = np.unique(pats[sel], return_counts=True)
        j = int(cnt.argmax())
        rows.append(dict(cluster=int(c), n=int(sel.sum()), top_patient=str(u[j]), top_share=float(cnt[j] / sel.sum())))
    worst = max(rows, key=lambda r: r["top_share"])
    return bool(worst["top_share"] <= DOMINANCE), worst, rows


def cluster_consensus(M, labels):
    rows = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            rows.append(dict(cluster=int(c), n=int(len(idx)), m=float("nan")))
            continue
        sub = M[np.ix_(idx, idx)].astype(np.float64)
        rows.append(dict(cluster=int(c), n=int(len(idx)), m=float(sub[np.triu_indices(len(idx), 1)].mean())))
    df = pd.DataFrame(rows).sort_values("m", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def item_consensus(M, labels):
    """m_i(k) for every electrode i and cluster k (Monti: i's own cluster without itself)."""
    K = int(labels.max()) + 1
    n = len(labels)
    out = np.zeros((n, K))
    Mf = M.astype(np.float64)
    for c in range(K):
        idx = np.flatnonzero(labels == c)
        s = Mf[:, idx].sum(1)
        cnt = np.full(n, float(len(idx)))
        own = labels == c
        s[own] -= 0.0                       # the diagonal is 1 by convention when i in k; remove it
        s[own] -= np.diag(Mf)[own]
        cnt[own] -= 1
        out[:, c] = s / np.maximum(cnt, 1)
    return out


def heatmap(M, labels, pats, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import squareform
    _, Z = consensus_labels(M, max(2, int(labels.max()) + 1))
    Dm = 1.0 - M.astype(np.float64)
    np.fill_diagonal(Dm, 0.0)
    Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    order = leaves_list(optimal_leaf_ordering(Z, squareform(Dm, checks=False)))
    ups = list(pd.unique(pats))
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    pcol = np.array([pal[ups.index(p) % 20] for p in pats[order]])
    ccol = plt.get_cmap("tab10")(labels[order] % 10)
    fig = plt.figure(figsize=(9, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 30], height_ratios=[1, 30], wspace=0.02, hspace=0.02)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(M[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    top = fig.add_subplot(gs[0, 1], sharex=ax)
    top.imshow(ccol[None, :, :3], aspect="auto", interpolation="nearest"); top.set_axis_off()
    left = fig.add_subplot(gs[1, 0], sharey=ax)
    left.imshow(pcol[:, None, :3], aspect="auto", interpolation="nearest"); left.set_axis_off()
    fig.suptitle(title + "\ntop strip = consensus cluster, left strip = patient; rows and columns in optimal leaf order of the consensus dendrogram", fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ one cell
def run_cell(cell: dict, method: str, out: Path, a, stats_dir: Path | None, k_current: int | None, log) -> dict:
    """Everything of one (feature set, method). Returns the row of summary_cells.csv."""
    from sklearn.metrics import adjusted_rand_score as ari
    out.mkdir(parents=True, exist_ok=True)
    (out / "null").mkdir(exist_ok=True)
    A, X, pats = cell["A"], cell["X"], cell["pats"]
    n = len(A)
    ks = list(a.ks)
    fset = cell["fset"]
    t_cell = time.time()
    log(f"\n=== {fset} x {LABEL[method]}   n={n}  patients={len(np.unique(pats))}  Ks {ks[0]}..{ks[-1]}  H={a.H}  null {a.null_draws}x{a.null_H}")

    # --- real consensus per K (resumable) --------------------------------------------------
    per_k = {}
    for K in ks:
        fM, fJ = out / f"M_k{K:02d}.npy", out / f"k{K:02d}.json"
        if fM.exists() and fJ.exists() and not a.force:
            per_k[K] = json.loads(fJ.read_text(encoding="utf-8"))
            continue
        t0 = time.time()
        M, tog = consensus_matrix(A, pats, K, method, a.H, a.seed * 1_000_003 + K * 1000, a.frac, a.jobs)
        x, cdf, area = cdf_area(M)
        sym = bool(np.allclose(M, M.T, atol=1e-6)) and bool(M.min() >= 0) and bool(M.max() <= 1)
        off_tog = tog[np.triu_indices(n, 1)]
        rec = dict(K=K, pac=pac_of(M), area=area, cdf_x=x.round(4).tolist(), cdf=cdf.round(5).tolist(),
                   min_cosample=int(off_tog.min()), median_cosample=float(np.median(off_tog)),
                   symmetric_in_range=sym, H=a.H, seconds=round(time.time() - t0, 1))
        np.save(fM, M.astype(np.float16))
        fJ.write_text(json.dumps(rec), encoding="utf-8")
        per_k[K] = rec
        log(f"  K={K:<3} PAC {rec['pac']:.3f}  A {area:.3f}  min co-sample {rec['min_cosample']}  ({rec['seconds']:.0f}s)")

    # --- null per K (resumable per draw; the surrogate is per feature set and draw) ------------
    null = {K: [] for K in ks}
    if not a.skip_null:
        for d in range(a.null_draws):
            Y = None
            for K in ks:
                fJ = out / "null" / f"k{K:02d}_draw{d:02d}.json"
                if fJ.exists() and not a.force:
                    null[K].append(json.loads(fJ.read_text(encoding="utf-8")))
                    continue
                if Y is None:
                    Y = unit(surrogate(X, 7000 + d))
                t0 = time.time()
                Mn, _ = consensus_matrix(Y, pats, K, method, a.null_H, 50_000 + a.seed * 1_000_003 + d * 1000 + K, a.frac, a.jobs)
                _, _, area_n = cdf_area(Mn)
                rec = dict(K=K, draw=d, pac=pac_of(Mn), area=area_n, H=a.null_H, seconds=round(time.time() - t0, 1))
                if d == 0 and k_current is not None and K == k_current and not a.skip_figures:
                    lab_n, _ = consensus_labels(Mn, K)
                    heatmap(Mn, lab_n, pats, out / f"null_sanity_k{K:02d}.png",
                            f"{FSET_LABEL[fset]} x {LABEL[method]}  -  ONE SURROGATE DRAW at K={K}: PAC {rec['pac']:.3f} (no blocks expected)")
                fJ.write_text(json.dumps(rec), encoding="utf-8")
                null[K].append(rec)
            log(f"  null draw {d + 1}/{a.null_draws} done")

    # --- the K curve ---------------------------------------------------------------------------
    areas = {K: per_k[K]["area"] for K in ks}
    deltas = delta_curves(areas)
    curve = []
    for K in ks:
        pr = per_k[K]["pac"]
        pn = np.array([r["pac"] for r in null[K]], float)
        row = dict(K=K, pac_real=pr, area=areas[K], delta=deltas[K]["delta"], delta_hat=deltas[K]["delta_hat"],
                   min_cosample=per_k[K]["min_cosample"])
        if len(pn):
            row.update(pac_null_mean=float(pn.mean()), pac_null_sd=float(pn.std(ddof=1)) if len(pn) > 1 else float("nan"),
                       rcsi=float(np.log(max(pn.mean(), 1e-12)) - np.log(max(pr, 1e-12))),
                       p=float((1 + (pn <= pr).sum()) / (1 + len(pn))), n_null=int(len(pn)))
        M = np.load(out / f"M_k{K:02d}.npy").astype(np.float32)
        lab, _ = consensus_labels(M, K)
        ok, worst, _ = dominance(lab, pats)
        row.update(dominance_ok=ok, worst_share=worst["top_share"], worst_cluster=worst["cluster"], worst_patient=worst["top_patient"])
        curve.append(row)
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "k_curve.csv", index=False)
    score = "rcsi" if "rcsi" in curve else "pac_real"
    best = curve.sort_values(score, ascending=(score == "pac_real"))
    k_unc = int(best.iloc[0].K)
    allowed = best[best.dominance_ok]
    k_con = int(allowed.iloc[0].K) if len(allowed) else k_unc
    log(f"  chosen K: constrained {k_con}, unconstrained {k_unc}  (by {'RCSI' if score == 'rcsi' else 'min PAC, no null'})")

    # --- partitions, stability scores, figures at the chosen K and the current K -------------
    rows_k = {}
    seed_rerun = None
    cross = {}
    targets = sorted({k_con} | ({k_current} if k_current in per_k else set()))
    for K in targets:
        M = np.load(out / f"M_k{K:02d}.npy").astype(np.float32)
        lab, _ = consensus_labels(M, K)
        pd.DataFrame({"sample_idx": np.arange(n), "patient_id": pats, "electrode": cell["elec"], "cluster": lab}).to_csv(out / f"consensus_labels_k{K:02d}.csv", index=False)
        pub = published_labels(method, cell["runs"][method], n, K) if "runs" in cell else None
        ref = fit_any(A, K, 42, method)
        rec = dict(K=K, ari_vs_published=float(ari(pub, lab)) if pub is not None else None,
                   ari_vs_refit42=float(ari(ref, lab)))
        cc = cluster_consensus(M, lab)
        cc.to_csv(out / f"clusters_k{K:02d}.csv", index=False)
        ic = item_consensus(M, lab)
        own = ic[np.arange(n), lab]
        other = ic.copy(); other[np.arange(n), lab] = -np.inf
        best_other = other.max(1)
        items = pd.DataFrame({"electrode": cell["elec"], "patient_id": pats, "shaft": cell["shafts"], "cluster": lab,
                              "own_consensus": own, "best_other_consensus": best_other, "margin": own - best_other})
        items.to_csv(out / f"items_k{K:02d}.csv", index=False)
        pt = items.groupby("patient_id").agg(n=("own_consensus", "size"), mean_own=("own_consensus", "mean"),
                                             frac_below_05=("own_consensus", lambda v: float((v < 0.5).mean()))).reset_index()
        pt = pt.sort_values("mean_own", ascending=False).reset_index(drop=True)
        pt["rank"] = np.arange(1, len(pt) + 1)
        rho = None
        if stats_dir is not None and (stats_dir / "stats_lopo_predict.csv").exists() and K == k_current:
            lp = pd.read_csv(stats_dir / "stats_lopo_predict.csv")
            lp = lp[lp["method"] == method].groupby("group")["acc"].mean()
            j = pt.set_index("patient_id").join(lp.rename("lopo_acc"), how="inner")
            if len(j) >= 5:
                from scipy.stats import spearmanr
                rho = spearmanr(j["mean_own"], j["lopo_acc"])
                pt = pt.merge(lp.rename("lopo_acc"), left_on="patient_id", right_index=True, how="left")
                rec["spearman_patient_consensus_vs_lopo_acc"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=int(len(j)))
        pt.to_csv(out / f"patients_k{K:02d}.csv", index=False)
        rec.update(mean_cluster_consensus=float(cc["m"].mean()), n_unstable=int((cc["m"] < UNSTABLE).sum()),
                   cluster_table=cc.to_dict("records"),
                   named_patients={p: (int(pt.loc[pt.patient_id == p, "rank"].iloc[0]), float(pt.loc[pt.patient_id == p, "mean_own"].iloc[0]))
                                   for p in NAMED if (pt.patient_id == p).any()},
                   dominance=dominance(lab, pats)[1])
        if not a.skip_figures:
            heatmap(M, lab, pats, out / f"heatmap_k{K:02d}.png",
                    f"{FSET_LABEL[fset]} x {LABEL[method]}  -  consensus at K={K} ({'chosen' if K == k_con else 'current'}): PAC {per_k[K]['pac']:.3f}, mean cluster consensus {rec['mean_cluster_consensus']:.2f}")
        rows_k[K] = rec
        if K == k_con and method in ("kmeans", "cnmf") and not a.skip_rerun:
            fR = out / f"seed_rerun_k{K:02d}.json"
            if fR.exists() and not a.force:
                seed_rerun = json.loads(fR.read_text(encoding="utf-8"))
            else:
                t0 = time.time()
                M2, _ = consensus_matrix(A, pats, K, method, a.H, 900_000 + a.seed * 1_000_003 + K * 1000, a.frac, a.jobs)
                lab2, _ = consensus_labels(M2, K)
                seed_rerun = dict(K=K, ari=float(ari(lab, lab2)), pac_second=pac_of(M2), seconds=round(time.time() - t0, 1),
                                  verdict="ok" if ari(lab, lab2) >= SEED_RERUN_OK else "raise H")
                fR.write_text(json.dumps(seed_rerun), encoding="utf-8")
            log(f"  seed rerun at K={K}: ARI {seed_rerun['ari']:.3f} -> {seed_rerun['verdict']}")

    # --- figures of the K curve ---------------------------------------------------------------
    if not a.skip_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(curve.K, curve.pac_real, "o-", label="real")
        if "pac_null_mean" in curve:
            ax[0].errorbar(curve.K, curve.pac_null_mean, yerr=curve.pac_null_sd, fmt="s--", color="0.5", label=f"null (mean ± SD, {a.null_draws} draws)")
        ax[0].set_xlabel("K"); ax[0].set_ylabel("PAC (0.1 < M < 0.9)"); ax[0].legend(fontsize=8); ax[0].set_title("PAC")
        if "rcsi" in curve:
            ax[1].plot(curve.K, curve.rcsi, "o-")
            for _, r in curve.iterrows():
                ax[1].annotate(f"p={r.p:.2f}", (r.K, r.rcsi), fontsize=7, textcoords="offset points", xytext=(0, 5), ha="center")
            for _, r in curve[~curve.dominance_ok].iterrows():
                ax[1].plot(r.K, r.rcsi, "x", color="red", ms=10)
            ax[1].axvline(k_con, color="green", ls="--", lw=1); ax[1].axvline(k_unc, color="orange", ls=":", lw=1)
        ax[1].set_xlabel("K"); ax[1].set_ylabel("RCSI = log PAC_null − log PAC_real"); ax[1].set_title("RCSI (red x = fails the patient constraint; green = chosen, orange = unconstrained)", fontsize=8)
        for K in ks:
            ax[2].plot(per_k[K]["cdf_x"], per_k[K]["cdf"], label=f"K={K}", lw=1)
        ax[2].set_xlabel("consensus value"); ax[2].set_ylabel("CDF"); ax[2].legend(fontsize=6, ncol=2); ax[2].set_title("consensus CDF")
        fig.suptitle(f"{FSET_LABEL[fset]} x {LABEL[method]}  -  patients resampled (80 % without replacement), H={a.H}", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "pac_rcsi.png", dpi=110); plt.close(fig)

    # --- verification + manifest --------------------------------------------------------------
    ver = dict(symmetric_in_range=all(per_k[K]["symmetric_in_range"] for K in ks),
               min_cosample=int(min(per_k[K]["min_cosample"] for K in ks)),
               cosample_ok=bool(min(per_k[K]["min_cosample"] for K in ks) >= 0.25 * a.H),
               x_md5=cell.get("md5"), x_md5_identical=(len(set(cell["md5"].values())) == 1) if cell.get("md5") else None)
    (out / "verification.json").write_text(json.dumps(ver, indent=1), encoding="utf-8")
    man = dict(feature_set=fset, method=method, cache=cell.get("cache"), runs={m: str(r) for m, r in cell.get("runs", {}).items()},
               n_electrodes=n, n_patients=int(len(np.unique(pats))), ks=ks, H=a.H, null_draws=0 if a.skip_null else a.null_draws,
               null_H=a.null_H, frac=a.frac, seed=a.seed, k_current=k_current, k_chosen=k_con, k_unconstrained=k_unc,
               seconds=round(time.time() - t_cell, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(dict(curve=curve.to_dict("records"), at_k=rows_k, seed_rerun=seed_rerun), indent=1, default=float), encoding="utf-8")

    cur = curve.set_index("K")
    rc = rows_k[k_con]
    return dict(feature_set=fset, method=method, k_current=k_current, k_chosen=k_con, k_unconstrained=k_unc,
                constraint_binding=(k_con != k_unc),
                pac_real=float(cur.loc[k_con, "pac_real"]), pac_null=float(cur.loc[k_con, "pac_null_mean"]) if "pac_null_mean" in cur else None,
                rcsi=float(cur.loc[k_con, "rcsi"]) if "rcsi" in cur else None, p=float(cur.loc[k_con, "p"]) if "p" in cur else None,
                mean_cluster_consensus=rc["mean_cluster_consensus"], n_unstable=rc["n_unstable"],
                ari_vs_published=rc["ari_vs_published"], ari_vs_refit42=rc["ari_vs_refit42"],
                seed_rerun_ari=seed_rerun["ari"] if seed_rerun else None,
                mean_cluster_consensus_current=rows_k[k_current]["mean_cluster_consensus"] if k_current in rows_k else None,
                seconds=man["seconds"])


# ------------------------------------------------------------------ cross-cell
def cross_cell(root: Path, a, log):
    from sklearn.metrics import adjusted_rand_score as ari
    rows = []
    for fs in a.fsets:
        for m in a.methods:
            f = root / fs / m / "results.json"
            man = root / fs / m / "manifest.json"
            if not f.exists() or not man.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8")); mf = json.loads(man.read_text(encoding="utf-8"))
            cur = {int(c["K"]): c for c in r["curve"]}
            kc = mf["k_chosen"]; at = r["at_k"][str(kc)] if str(kc) in r["at_k"] else r["at_k"][kc]
            rows.append(dict(feature_set=fs, method=m, k_current=mf["k_current"], k_chosen=kc, k_unconstrained=mf["k_unconstrained"],
                             constraint_binding=kc != mf["k_unconstrained"], pac_real=cur[kc]["pac_real"], pac_null=cur[kc].get("pac_null_mean"),
                             rcsi=cur[kc].get("rcsi"), p=cur[kc].get("p"), mean_cluster_consensus=at["mean_cluster_consensus"],
                             n_unstable=at["n_unstable"], ari_vs_published=at["ari_vs_published"], ari_vs_refit42=at["ari_vs_refit42"],
                             seed_rerun_ari=(r["seed_rerun"] or {}).get("ari"), seconds=mf["seconds"]))
    if not rows:
        log("nothing to summarise"); return
    df = pd.DataFrame(rows)
    df.to_csv(root / "summary_cells.csv", index=False)

    # cross-method ARI of the consensus partitions at the current K, beside the single-fit values
    agree = []
    for fs in a.fsets:
        kcur = peak_k().get(fs)
        labs = {}
        for m in a.methods:
            f = root / fs / m / f"consensus_labels_k{int(kcur):02d}.csv" if kcur else None
            if f is not None and f.exists():
                labs[m] = pd.read_csv(f)["cluster"].to_numpy()
        single = {}
        sf = CLUST / "statistics" / f"{fs}_K{kcur}" / "stats_agreement.csv" if kcur else None
        if sf is not None and sf.exists():
            for _, r in pd.read_csv(sf).iterrows():
                single[tuple(sorted((r.a, r.b)))] = float(r.ari)
        ms = list(labs)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                key = tuple(sorted((ms[i], ms[j])))
                agree.append(dict(feature_set=fs, K=kcur, a=ms[i], b=ms[j], ari_consensus=float(ari(labs[ms[i]], labs[ms[j]])),
                                  ari_single_fit=single.get(key)))
    agree = pd.DataFrame(agree)
    if len(agree):
        agree.to_csv(root / "cross_method_agreement.csv", index=False)

    if not a.skip_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(a.fsets), len(a.methods), figsize=(4 * len(a.methods), 3 * len(a.fsets)), squeeze=False, sharex=True)
        for i, fs in enumerate(a.fsets):
            for j, m in enumerate(a.methods):
                ax = axs[i, j]
                f = root / fs / m / "k_curve.csv"
                if f.exists():
                    c = pd.read_csv(f)
                    if "rcsi" in c:
                        ax.plot(c.K, c.rcsi, "o-")
                        for _, r in c[~c.dominance_ok].iterrows():
                            ax.plot(r.K, r.rcsi, "x", color="red", ms=9)
                        mf = json.loads((root / fs / m / "manifest.json").read_text(encoding="utf-8"))
                        ax.axvline(mf["k_chosen"], color="green", ls="--", lw=1)
                        if mf["k_current"]:
                            ax.axvline(mf["k_current"], color="0.5", ls=":", lw=1)
                    else:
                        ax.plot(c.K, c.pac_real, "o-", color="0.4"); ax.set_ylabel("PAC (no null)")
                ax.set_title(f"{FSET_LABEL[fs]} x {LABEL[m]}", fontsize=9)
                if i == len(a.fsets) - 1: ax.set_xlabel("K")
                if j == 0 and "rcsi" in locals().get("c", pd.DataFrame()): ax.set_ylabel("RCSI")
        fig.suptitle("RCSI by K  (green dashed = chosen K, grey dotted = current K, red x = fails the patient constraint)", fontsize=10)
        fig.tight_layout(); fig.savefig(root / "rcsi_by_k.png", dpi=110); plt.close(fig)

    # SUMMARY.md
    L = [f"# Consensus clustering stability  -  {root.name}", "",
         f"written {time.strftime('%Y-%m-%d %H:%M')} by consensus_stability.py; patients resampled (80 % without replacement), "
         f"H = {a.H}, null = covariance-matched Gaussian surrogate, {a.null_draws} draws x {a.null_H} iterations, Ks {a.ks[0]}..{a.ks[-1]}.", "",
         "## Chosen K per cell", "", "| feature set | method | current K | chosen K (constrained) | unconstrained | constraint binding | PAC real | PAC null | RCSI | p | mean cluster consensus | clusters m(k) < 0.6 | ARI vs published | ARI vs refit (seed 42) | seed-rerun ARI |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, d=3: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    for _, r in df.iterrows():
        L.append(f"| {FSET_LABEL.get(r.feature_set, r.feature_set)} | {LABEL.get(r.method, r.method)} | {fmt(r.k_current)} | {r.k_chosen} | {r.k_unconstrained} | {'yes' if r.constraint_binding else 'no'} | {fmt(r.pac_real)} | {fmt(r.pac_null)} | {fmt(r.rcsi)} | {fmt(r.p)} | {fmt(r.mean_cluster_consensus)} | {r.n_unstable} | {fmt(r.ari_vs_published)} | {fmt(r.ari_vs_refit42)} | {fmt(r.seed_rerun_ari)} |")
    if len(agree):
        L += ["", "## Cross-method agreement of the consensus partitions at the current K (ARI), beside the single-fit values of the statistics script", "",
              "| feature set | K | pair | ARI consensus | ARI single fit |", "|---|---|---|---|---|"]
        for _, r in agree.iterrows():
            L.append(f"| {FSET_LABEL.get(r.feature_set, r.feature_set)} | {r.K} | {LABEL.get(r.a, r.a)} vs {LABEL.get(r.b, r.b)} | {r.ari_consensus:.3f} | {fmt(r.ari_single_fit)} |")
    # feature-set order
    if "rcsi" in df and df.rcsi.notna().any():
        order_r = df.groupby("feature_set").rcsi.mean().reindex(a.fsets)
        order_c = df.groupby("feature_set").mean_cluster_consensus.mean().reindex(a.fsets)
        L += ["", "## Feature-set order", "",
              "The existing tests ordered the sets 15 bands > 5 bands > 2 bands > HFA on placement and agreement. Here (mean over the three methods at the chosen K):", "",
              "| feature set | mean RCSI | mean cluster consensus |", "|---|---|---|"]
        for fs in a.fsets:
            L.append(f"| {FSET_LABEL[fs]} | {fmt(order_r.get(fs))} | {fmt(order_c.get(fs))} |")
        L.append("")
        L.append("RCSI order: " + " > ".join(FSET_LABEL[f] for f in order_r.sort_values(ascending=False).index) +
                 "; cluster-consensus order: " + " > ".join(FSET_LABEL[f] for f in order_c.sort_values(ascending=False).index) + ".")
    # per cell detail
    L += ["", "## Per cell: cluster stability ranking and patients, at the chosen K", ""]
    for fs in a.fsets:
        for m in a.methods:
            f = root / fs / m / "results.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8")); mf = json.loads((root / fs / m / "manifest.json").read_text(encoding="utf-8"))
            kc = mf["k_chosen"]; at = r["at_k"].get(str(kc)) or r["at_k"].get(kc)
            L.append(f"### {FSET_LABEL[fs]} x {LABEL[m]}  (chosen K = {kc}, current K = {mf['k_current']})")
            L.append("clusters by consensus m(k): " + ", ".join(f"c{c['cluster']} n={c['n']} m={c['m']:.2f}" + (" UNSTABLE" if c["m"] < UNSTABLE else "") for c in at["cluster_table"]))
            if at.get("named_patients"):
                L.append("named patients (rank of mean own-cluster consensus, of all patients; value): " + ", ".join(f"{p} #{v[0]} ({v[1]:.2f})" for p, v in at["named_patients"].items()))
            atc = r["at_k"].get(str(mf["k_current"])) or r["at_k"].get(mf["k_current"]) if mf.get("k_current") is not None else None
            if atc and atc.get("spearman_patient_consensus_vs_lopo_acc"):       # computed at the current K, where the LOPO table exists
                s = atc["spearman_patient_consensus_vs_lopo_acc"]
                L.append(f"per-patient consensus vs LOPO placement accuracy at the current K = {mf['k_current']}: Spearman rho = {s['rho']:.2f} (p = {s['p']:.3f}, n = {s['n']})")
            d = at.get("dominance")
            if d:
                L.append(f"patient dominance at the chosen K: worst cluster c{d['cluster']} has {100 * d['top_share']:.0f} % of its {d['n']} electrodes from {d['top_patient']}")
            if r.get("seed_rerun"):
                L.append(f"seed rerun: ARI {r['seed_rerun']['ari']:.3f} -> {r['seed_rerun']['verdict']}")
            v = json.loads((root / fs / m / "verification.json").read_text(encoding="utf-8"))
            L.append(f"verification: symmetric in [0,1] {v['symmetric_in_range']}; min co-samples {v['min_cosample']} ({'ok' if v['cosample_ok'] else 'LOW'}); X_train md5 identical across runs {v['x_md5_identical']}")
            L.append("")
    tot = df.seconds.sum()
    L += ["## Runtime", "", f"total {tot / 3600:.1f} h over {len(df)} cells (sum of the cells' own clocks; a resumed run counts only what it computed).", ""]
    if (root / "existing_outputs_check.json").exists():
        chk = json.loads((root / "existing_outputs_check.json").read_text(encoding="utf-8"))
        L += ["## Existing outputs", "", f"{chk['n_files']} existing files hashed before and after: {'unchanged' if chk['unchanged'] else 'CHANGED - ' + ', '.join(chk['changed'])}.", ""]
    L += ["## Reading", "", f"Cluster consensus below {UNSTABLE} is flagged unstable (a convention). RCSI compares the real PAC with the surrogate's; the surrogate has the data's covariance but no patient structure, so a K whose clusters follow patients can look stable against it - the dominance constraint and the per-patient tables are the guards. Numbers only; no interpretation beyond them."]
    (root / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    log(f"\nwrote {root / 'summary_cells.csv'}, {root / 'SUMMARY.md'}")


# ------------------------------------------------------------------ existing outputs: before / after
def existing_snapshot(fsets, cells_runs) -> dict:
    files = []
    for fs in fsets:
        for d in (CLUST / "statistics").glob(f"{fs}_K*"):
            files += [p for p in d.iterdir() if p.is_file()]
    for runs in cells_runs.values():
        for r in runs.values():
            files += [r / f for f in ("labels.csv", "cluster_labels_by_k.csv", "cluster_statistics.json", "manifest.json", "X_train.npy") if (r / f).exists()]
    return {str(p): md5_file(p) for p in sorted(set(files))}


# ------------------------------------------------------------------ estimate
def estimate(cells, a, log):
    log("timing one subsampled fit per cell at the current K ...")
    rows = []
    for fs in a.fsets:
        cell = cells[fs]
        kc = int(peak_k().get(fs, a.ks[len(a.ks) // 2]))
        ups = np.unique(cell["pats"]); idx_by_pat = [np.flatnonzero(cell["pats"] == p) for p in ups]
        for m in a.methods:
            t0 = time.time(); _one_iteration(cell["A"], idx_by_pat, int(round(a.frac * len(ups))), kc, m, 1); t = time.time() - t0
            fits = len(a.ks) * (a.H + (0 if a.skip_null else a.null_draws * a.null_H)) + (a.H if m in ("kmeans", "cnmf") and not a.skip_rerun else 0)
            rows.append(dict(feature_set=fs, method=m, K=kc, seconds_per_fit=round(t, 2), fits=fits, cpu_hours=t * fits / 3600))
            log(f"  {fs:<16} {LABEL[m]:<11} {t:6.2f} s/fit x {fits} fits = {t * fits / 3600:7.1f} cpu-h")
    df = pd.DataFrame(rows)
    jobs = os.cpu_count() if a.jobs in (-1, 0, None) else a.jobs
    log(f"\ntotal {df.cpu_hours.sum():.1f} cpu-hours; with {jobs} parallel jobs about {df.cpu_hours.sum() / max(jobs, 1):.1f} wall-clock hours "
        f"(joblib across iterations; cNMF rows dominate - cut --null-H, --null-draws or --ks first)")
    return df


# ------------------------------------------------------------------ self-test
def selftest(d: Path, log):
    global _SELFTEST
    _SELFTEST = True
    rng = np.random.default_rng(0)
    n_pat, per, p = 8, 20, 24
    pats = np.repeat([f"P{i:02d}" for i in range(n_pat)], per)
    truth = rng.integers(0, 3, size=n_pat * per)
    centers = rng.standard_normal((3, p)) * 3
    X = centers[truth] + rng.standard_normal((n_pat * per, p))
    cell = dict(fset="concat_bands15", X=X, A=unit(X), pats=pats, elec=np.array([f"e{i}" for i in range(len(X))]),
                shafts=np.array([f"s{i // 4}" for i in range(len(X))]), md5={"kmeans": "x", "hierarchical": "x", "cnmf": "x"}, cache="selftest")
    a = argparse.Namespace(ks=[2, 3, 4, 5], H=12, null_draws=2, null_H=6, frac=0.8, jobs=1, seed=0, force=True, skip_null=False,
                           skip_figures=False, skip_rerun=False, fsets=["concat_bands15"], methods=["kmeans", "hierarchical"])
    root = d / "consensus_selftest"
    for m in a.methods:
        row = run_cell(cell, m, root / "concat_bands15" / m, a, None, 3, log)
        log(f"  {m}: chosen K {row['k_chosen']} (planted 3), ARI vs refit {row['ari_vs_refit42']:.2f}")
        assert row["k_chosen"] == 3, "the self-test must recover the planted K=3"
    cross_cell(root, a, log)
    assert (root / "SUMMARY.md").exists() and (root / "rcsi_by_k.png").exists()
    log("self-test passed")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fsets", nargs="*", default=list(FSETS))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--ks", default="2-12", help="K range, e.g. 2-12 or 2,3,4")
    ap.add_argument("--H", type=int, default=200, help="iterations per (cell, K) on the real data")
    ap.add_argument("--null-draws", type=int, default=25)
    ap.add_argument("--null-H", type=int, default=50, help="iterations per null draw")
    ap.add_argument("--frac", type=float, default=0.8, help="share of patients per iteration")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    ap.add_argument("--force", action="store_true", help="recompute what exists")
    ap.add_argument("--skip-null", action="store_true")
    ap.add_argument("--skip-rerun", action="store_true", help="no second-seed consensus")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="one cell only: concat_bands15 x kmeans")
    ap.add_argument("--estimate", action="store_true", help="time one fit per cell and project the total, then stop")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--selftest", metavar="DIR")
    a = ap.parse_args()
    a.ks = sorted({int(x) for x in a.ks.split(",")} if "," in a.ks else set(range(int(a.ks.split("-")[0]), int(a.ks.split("-")[1]) + 1))) if "-" in a.ks or "," in a.ks else [int(a.ks)]
    log = lambda s: print(s, flush=True)
    if a.selftest:
        selftest(Path(a.selftest), log)
        return 0
    if a.pilot:
        a.fsets, a.methods = ["concat_bands15"], ["kmeans"]
    LR, _, _ = _mods()
    cache = LR.current_cache()
    root = CLUST / "consensus" / cache
    root.mkdir(parents=True, exist_ok=True)
    log(f"cohort cache {cache}   ->   {root}")
    if a.summary_only:
        cross_cell(root, a, log)
        return 0
    cells = {fs: load_cell(fs) for fs in a.fsets}
    for fs, c in cells.items():
        log(f"  {fs:<16} X {c['X'].shape}  patients {len(np.unique(c['pats']))}  X_train md5 identical across runs: {len(set(c['md5'].values())) == 1}")
    if a.estimate:
        estimate(cells, a, log)
        return 0
    before = existing_snapshot(a.fsets, {fs: c["runs"] for fs, c in cells.items()})
    pk = peak_k()
    rows = []
    for fs in a.fsets:
        kc = pk.get(fs)
        kc = int(kc) if kc is not None else None
        stats_dir = CLUST / "statistics" / f"{fs}_K{kc}" if kc else None
        for m in a.methods:
            rows.append(run_cell(cells[fs], m, root / fs / m, a, stats_dir, kc, log))
    after = existing_snapshot(a.fsets, {fs: c["runs"] for fs, c in cells.items()})
    changed = [p for p in before if after.get(p) != before[p]]
    (root / "existing_outputs_check.json").write_text(json.dumps(dict(n_files=len(before), unchanged=not changed, changed=changed), indent=1), encoding="utf-8")
    log(f"existing outputs: {len(before)} files hashed, {'unchanged' if not changed else 'CHANGED: ' + ', '.join(changed)}")
    cross_cell(root, a, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
