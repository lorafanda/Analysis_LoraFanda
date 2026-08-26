#!/usr/bin/env python3
"""
make_bsf_comparison.py - PART 1. BSF against the other two algorithms at matched K.

BSF ("best so far") is a PINNED run, not "the newest":

    kmeans / concat_hg_all / 20260819_235524      K = 8      n = 2946

and it is compared against the two runs on the SAME feature set, which were checked
to hold a bit-identical X_train in the identical electrode order:

    hierarchical / concat_hg_all / 20260819_235654
    cnmf         / concat_hg_all / 20260819_220417

Pinned rather than resolved through lf_runs.newest_run(): the point of a "best so far"
is that it does not move when something else is run.

WHAT IS MEASURED, and why each one is here.

  separation   Silhouette in BOTH dB and unit-norm space, each against its own
               matched null - a Gaussian carrying this feature set's covariance, so
               one correlated blob with no cluster structure. Every method is read in
               its HOME space (cnmf unit-norm, kmeans/Ward dB); scoring all three in
               dB is the error that made the first version of FIG C.7 wrong.

  agreement    ARI and NMI between all three partitions, plus the contingency table
               and cluster sizes. Answers "do they find the same thing", which no
               per-method score can.

  coherence    Neighbours-sharing-label over chance, in fsaverage space. The only
               criterion here that is about the brain rather than about the fit.

  lopo         Leave-one-patient-out ARI against the full-cohort solution, scored
               against a size-matched pseudo-patient null. lf_decompose.lopo_stability
               hard-codes KMeans, so it cannot be used for Ward or cnmf; lopo_any()
               below is the same procedure with the refit delegated to the method.

  gate         Every electrode tagged gated / added, by joining the concat_hg cohort
               into this one. concat_hg_all IS the ungated set, so this is what
               FIG C.8 colours blue and grey.

cnmf on concat_hg_all exists only at K=7 and carries no loadings_by_k/, so K=8 is
FITTED here and cached. The fit is written into the run's own loadings_by_k/ - the
layout concat_hg and concat_rawds already use - so existing tooling finds it. That is
purely additive: no existing file is modified.

    python make_bsf_comparison.py                # full, n_null = 10
    python make_bsf_comparison.py --n-null 3     # quick check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import lf_decompose as D                      # noqa: E402
import measure_cluster_stability as MS        # noqa: E402

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "bsf_comparison"
COORDS = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coords"
          / "ALL_PATIENTS_contacts_fsaverage.csv")

K = 8                      # overridden by --k
FSET = "concat_hg"          # overridden by --feature-set
GATED_FSET = "concat_hg"

# Resolved at run time, not pinned: the cohort was rebuilt, so the old run ids no
# longer describe the data. lf_runs.newest_run picks the run each notebook just wrote.
import lf_runs as LR                            # noqa: E402
RUNS: dict = {}
GATED_RUN = None
LABEL = {"kmeans": "k-means", "hierarchical": "Ward", "cnmf": "convex NMF"}
ORDER = ["kmeans", "hierarchical", "cnmf"]


def norm(s) -> str:
    return str(s).replace("_", "").replace("-", "").upper()


def keys_of(df) -> list:
    return [f"{p}|{norm(e)}" for p, e in zip(df["patient_id"], df["electrode"])]


# ── labels at K ──────────────────────────────────────────────────────────────
def cnmf_G_at_k(run: Path, X: np.ndarray, k: int, *, seed: int = 0) -> np.ndarray:
    """Loadings at k, from loadings_by_k/ if present, else fitted and cached there."""
    store = run / "loadings_by_k"
    f = store / f"G_k{k:02d}.npy"
    if f.exists():
        G = np.load(f)
        if G.shape == (len(X), k):
            return G
        print(f"  ! {f.name} has shape {G.shape}, expected {(len(X), k)} - refitting")
    print(f"  fitting convex NMF K={k} on {X.shape} (absent from this run) ...", end="", flush=True)
    t0 = time.time()
    G = D.convex_nmf(D.unit_norm(X), k, n_iter=300, random_state=seed)[1]
    if not np.isfinite(G).all():
        raise RuntimeError("convex_nmf returned non-finite loadings")
    store.mkdir(parents=True, exist_ok=True)
    np.save(f, G)
    np.save(OUT / f"cnmf_G_k{k:02d}_{FSET}.npy", G)     # a copy we own, outside the run
    print(f" {time.time()-t0:.0f}s -> cached {f.relative_to(CLUST)}")
    return G


def labels_at_k(method: str, run: Path, X: np.ndarray, k: int) -> np.ndarray:
    if method == "cnmf":
        G = cnmf_G_at_k(run, X, k)
        return (G / np.maximum(G.sum(1, keepdims=True), 1e-12)).argmax(1)
    f = run / "cluster_labels_by_k.csv"
    col = f"k_{k}"
    df = pd.read_csv(f)
    if col not in df.columns:
        raise SystemExit(f"{run.name}: {f.name} has no column {col}")
    lab = df[col].to_numpy()
    # written 1..K by some writers and 0..K-1 by others; normalise to 0-based
    return lab - lab.min()


# ── the measurements ─────────────────────────────────────────────────────────
def separation(X: np.ndarray, labs: dict, n_null: int) -> pd.DataFrame:
    from sklearn.metrics import silhouette_score
    spaces = {"dB": X, "unit-norm": MS.unit(X)}
    rows = []
    for m in ORDER:
        for sp, A in spaces.items():
            t0 = time.time()
            obs = float(silhouette_score(A, labs[m]))
            nulls = []
            for i in range(n_null):
                Y = MS.surrogate(X, 400 + i)
                Y = MS.unit(Y) if sp == "unit-norm" else Y
                nulls.append(float(silhouette_score(Y, MS.fit_any(Y, K, i, m))))
            nl = np.array(nulls, float)
            z = (obs - nl.mean()) / max(nl.std(), 1e-9)
            home = MS.SPACE[m] == sp
            rows.append(dict(method=m, method_label=LABEL[m], space=sp, home=home,
                             silhouette=obs, null_mean=float(nl.mean()),
                             null_sd=float(nl.std()), z=float(z), n_null=n_null))
            print(f"  {LABEL[m]:<15} {sp:<10} sil {obs:+.4f}  null {nl.mean():+.4f}"
                  f"+/-{nl.std():.4f}  z {z:+7.1f}{'   <- home' if home else ''}"
                  f"   ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def agreement(labs: dict) -> tuple:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows = []
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            rows.append(dict(a=a, a_label=LABEL[a], b=b, b_label=LABEL[b],
                             ari=float(adjusted_rand_score(labs[a], labs[b])),
                             nmi=float(normalized_mutual_info_score(labs[a], labs[b]))))
            print(f"  {LABEL[a]:<15} vs {LABEL[b]:<15} ARI {rows[-1]['ari']:.3f}"
                  f"   NMI {rows[-1]['nmi']:.3f}")
    cont = {f"{a}__{b}": pd.crosstab(labs[a], labs[b]).to_numpy().tolist()
            for i, a in enumerate(ORDER) for b in ORDER[i + 1:]}
    return pd.DataFrame(rows), cont


def coherence(labs: dict, xyz: np.ndarray) -> pd.DataFrame:
    rows = []
    for m in ORDER:
        obs, ratio = D.spatial_coherence(labs[m], xyz)
        rows.append(dict(method=m, method_label=LABEL[m],
                         neighbours_sharing_label=float(obs), over_chance=float(ratio)))
        print(f"  {LABEL[m]:<15} {obs:.3f} shared  =  {ratio:.2f}x chance")
    return pd.DataFrame(rows)


def lopo_any(X, groups, k, method, *, seed=42):
    """Leave-one-group-out ARI against the full-cohort solution, for ANY method.

    lf_decompose.lopo_stability does this with KMeans hard-coded, which silently makes
    it the wrong test for Ward and cnmf. Here the refit is delegated to MS.fit_any, so
    each method is compared against ITSELF."""
    from sklearn.metrics import adjusted_rand_score
    A = MS.unit(X) if MS.SPACE[method] == "unit-norm" else X
    base = MS.fit_any(A, k, seed, method)
    rows = []
    for g in np.unique(groups):
        keep = groups != g
        ref = MS.fit_any(A[keep], k, seed, method)
        held = base[keep]
        degen = len(np.unique(held)) < 2 or len(np.unique(ref)) < 2
        rows.append(dict(group=str(g), n_left_out=int((~keep).sum()),
                         ari=float(adjusted_rand_score(held, ref)),
                         degenerate=bool(degen)))
    return pd.DataFrame(rows)


def lopo_null_any(X, groups, k, method, *, n_rep=10, seed=0):
    """The same test with FAKE groups of identical sizes - otherwise a leave-one-out
    number cannot be read at all, because removing that many electrodes costs ARI on
    its own."""
    rng = np.random.default_rng(seed)
    sizes = pd.Series(groups).value_counts().to_numpy()
    out = []
    for rep in range(n_rep):
        idx = rng.permutation(len(X))
        fake = np.empty(len(X), dtype=object)
        s = 0
        for i, sz in enumerate(sizes):
            fake[idx[s:s + sz]] = f"F{i}"
            s += sz
        d = lopo_any(X, fake, k, method, seed=seed + rep)
        out.append(dict(rep=rep, mean=float(d["ari"].mean()), min=float(d["ari"].min())))
    return pd.DataFrame(out)


def main() -> int:
    # must precede any use of these names in this function
    global K, FSET, RUNS, GATED_RUN, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--lopo-reps", type=int, default=10)
    ap.add_argument("--skip-lopo", action="store_true")
    ap.add_argument("--feature-set", default=FSET)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out-tag", default=None)
    a = ap.parse_args()
    K, FSET = int(a.k), a.feature_set
    RUNS = {m: LR.newest_run(m, FSET) for m in ORDER}
    GATED_RUN = LR.newest_run("kmeans", FSET)
    OUT = CLUST / "statistics" / f"{FSET}_K{K}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"feature set {FSET}   K={K}")
    for m in ORDER:
        print(f"  {m:<14} {RUNS[m].parent.parent.name}/{RUNS[m].name}")

    # ---- load and prove the cohort is one cohort -----------------------------
    print(f"STATISTICS - {FSET}, K={K}")
    Xs, metas = {}, {}
    for m in ORDER:
        r = RUNS[m]
        if not r.exists():
            raise SystemExit(f"missing run: {r}")
        Xs[m] = np.load(r / "X_train.npy").astype(float)
        metas[m] = pd.read_csv(r / "labels.csv")
    X = Xs["kmeans"]
    meta = metas["kmeans"]
    for m in ORDER[1:]:
        assert Xs[m].shape == X.shape and np.allclose(Xs[m], X, atol=1e-6), \
            f"{m}: X_train differs from BSF - these are NOT the same electrodes"
        assert keys_of(metas[m]) == keys_of(meta), f"{m}: electrode ORDER differs"
    pats = meta["patient_id"].astype(str).to_numpy()
    print(f"  X {X.shape}  |  {len(np.unique(pats))} patients  |  "
          f"X_train identical across all three methods: VERIFIED")

    # ---- gate flag -----------------------------------------------------------
    gated_keys = set(keys_of(pd.read_csv(GATED_RUN / "labels.csv")))
    is_gated = np.array([k_ in gated_keys for k_ in keys_of(meta)])
    print(f"  gate: {int(is_gated.sum())} gated / {int((~is_gated).sum())} added "
          f"(joined from {GATED_FSET})")

    # ---- coordinates ---------------------------------------------------------
    co = pd.read_csv(COORDS)
    co["key"] = [f"{p}|{norm(n_)}" for p, n_ in zip(co["patient"], co["name"])]
    xyz = (pd.DataFrame({"key": keys_of(meta)})
           .merge(co[["key", "x", "y", "z"]].drop_duplicates("key"), on="key", how="left")
           [["x", "y", "z"]].to_numpy(float))
    print(f"  coords: {int((~np.isnan(xyz).any(1)).sum())} / {len(xyz)} matched")

    # ---- labels at K=8 -------------------------------------------------------
    labs = {m: labels_at_k(m, RUNS[m], X, K) for m in ORDER}
    for m in ORDER:
        u = np.unique(labs[m])
        print(f"  {LABEL[m]:<15} K={len(u)}  sizes {np.bincount(labs[m]).tolist()}")

    # ---- statistics ----------------------------------------------------------
    print("\n[1/4] separation vs matched null")
    sep = separation(X, labs, a.n_null)
    sep.to_csv(OUT / "stats_separation.csv", index=False)

    print("\n[2/4] cross-method agreement")
    agr, cont = agreement(labs)
    agr.to_csv(OUT / "stats_agreement.csv", index=False)

    print("\n[3/4] anatomical coherence")
    coh = coherence(labs, xyz)
    coh.to_csv(OUT / "stats_coherence.csv", index=False)

    lopo_rows, lopo_sum = [], {}
    if not a.skip_lopo:
        print("\n[4/4] leave-one-patient-out vs size-matched null")
        for m in ORDER:
            t0 = time.time()
            d = lopo_any(X, pats, K, m)
            d.insert(0, "method", m)
            lopo_rows.append(d)
            nl = lopo_null_any(X, pats, K, m, n_rep=a.lopo_reps)
            z = (d["ari"].min() - nl["min"].mean()) / max(nl["min"].std(), 1e-9)
            lopo_sum[m] = dict(real_mean=float(d["ari"].mean()),
                               real_min=float(d["ari"].min()),
                               null_min_mean=float(nl["min"].mean()),
                               null_min_sd=float(nl["min"].std()), z=float(z),
                               degenerate=int(d["degenerate"].sum()))
            print(f"  {LABEL[m]:<15} min {d['ari'].min():.3f}  null "
                  f"{nl['min'].mean():.3f}+/-{nl['min'].std():.3f}  z {z:+.2f}"
                  f"   ({time.time()-t0:.0f}s)")
        pd.concat(lopo_rows).to_csv(OUT / "stats_lopo.csv", index=False)

    # ---- sizes + gate composition -------------------------------------------
    rows = []
    for m in ORDER:
        for c in range(K):
            sel = labs[m] == c
            rows.append(dict(method=m, method_label=LABEL[m], cluster=c, n=int(sel.sum()),
                             pct=100 * float(sel.mean()),
                             n_gated=int((sel & is_gated).sum()),
                             n_added=int((sel & ~is_gated).sum()),
                             pct_added=100 * float((sel & ~is_gated).sum() / max(sel.sum(), 1))))
    sizes = pd.DataFrame(rows)
    sizes.to_csv(OUT / "stats_cluster_sizes.csv", index=False)
    np.save(OUT / "stats_labels.npy", np.stack([labs[m] for m in ORDER]))
    np.save(OUT / "stats_is_gated.npy", is_gated)
    np.save(OUT / "stats_xyz.npy", xyz)

    summary = dict(
        feature_set=FSET, K=K, n_electrodes=int(X.shape[0]),
        n_features=int(X.shape[1]), n_patients=int(len(np.unique(pats))),
        runs={m: str(RUNS[m].relative_to(CLUST)) for m in ORDER},
        x_train_identical=True, n_gated=int(is_gated.sum()),
        n_added=int((~is_gated).sum()),
        baseline_pct_added=100 * float((~is_gated).mean()),
        n_null=a.n_null, lopo=lopo_sum,
        contingency=cont,
        written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "stats_summary.json").write_text(json.dumps(summary, indent=2))

    # ── one file per RUN, so the clustering visualizer can read it ───────────────
    # The visualizer fetches everything from runBase (it already pulls metrics.json,
    # sweep_metrics.csv and stability_summary.json from there), so a central folder is
    # invisible to it. Each run gets only ITS OWN numbers, plus the pairwise agreement
    # it participates in. Everything here is K-specific and the file says so, because
    # the visualizer lets you switch K and these numbers do not follow.
    for m in ORDER:
        sp_home = sep[(sep.method == m) & (sep.home)]
        sp_away = sep[(sep.method == m) & (~sep.home)]
        co = coh[coh.method == m]
        pairs = {}
        for _, r in agr.iterrows():
            if r.a == m or r.b == m:
                other = r.b if r.a == m else r.a
                pairs[LABEL[other]] = dict(ari=float(r.ari), nmi=float(r.nmi))
        blob = dict(
            schema=1, feature_set=FSET, K=K, method=m, method_label=LABEL[m],
            valid_only_at_K=True,
            n_electrodes=int(X.shape[0]), n_patients=int(len(np.unique(pats))),
            cohort_cache="concat_source_v3",
            separation=(dict(
                space=str(sp_home.iloc[0].space),
                silhouette=float(sp_home.iloc[0].silhouette),
                null_mean=float(sp_home.iloc[0].null_mean),
                null_sd=float(sp_home.iloc[0].null_sd),
                z=float(sp_home.iloc[0].z), n_null=int(a.n_null),
                other_space=str(sp_away.iloc[0].space) if len(sp_away) else None,
                other_space_z=float(sp_away.iloc[0].z) if len(sp_away) else None,
            ) if len(sp_home) else None),
            anatomical_coherence=(dict(
                neighbours_sharing_label=float(co.iloc[0].neighbours_sharing_label),
                over_chance=float(co.iloc[0].over_chance)) if len(co) else None),
            lopo=lopo_sum.get(m),
            agreement_with=pairs,
            cluster_sizes=[int(v) for v in
                           sizes[sizes.method == m].sort_values("cluster").n],
            written=summary["written"])
        p = RUNS[m] / "cluster_statistics.json"
        p.write_text(json.dumps(blob, indent=2))
        print(f"  run stats -> {p.relative_to(CLUST)}")

    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
