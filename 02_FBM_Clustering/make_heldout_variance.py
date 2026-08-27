#!/usr/bin/env python3
"""
make_heldout_variance.py - held-out variance explained as a function of K, for all
three algorithms, on the same electrodes.

WHY NOT cv_rank_curve. The curve already on the site holds out ELECTRODES only, then
fits each held-out electrode its own k loadings by NNLS across the FULL feature set.
Adding a basis vector can only reduce a least-squares residual, so that score is
monotone in K by construction - on concat_hg it climbs 0.389 to 0.674 and never turns
over. choose_k.py says this in its own docstring. A monotone curve cannot compare
methods any more than it can choose a K, so it is not used here.

THE PROTOCOL (bi-cross-validation, Owen & Perry 2009; Wold-style):

    hold out a block of ROWS and a block of COLUMNS
    fit the method on          rows_train x cols_train
    get each held-out row's loadings from   cols_train ONLY
    predict                    cols_test
    score there

The loadings never see the columns they are scored on, so an extra component has to
earn its place. This is D.bicv-style, generalised so the SAME protocol serves all
three methods - the only thing that differs is how a held-out row's loadings are
obtained, which is exactly the difference between the methods:

    convex NMF   graded loadings by NNLS against the train-column components
    k-means      one-hot at the nearest train-column centroid
    Ward         one-hot at the nearest train-column cluster mean

and the extension to the test columns is the matching operation in each case: a least-
squares solve for cnmf, and the same rows' mean on the test columns for the hard
methods. Fitting the method a second time on the test columns would be wrong - two
independent fits share no component order or scale.

SPACE. Each method is fitted AND scored in its home space (cnmf unit-norm, k-means and
Ward raw dB), because silhouette-style quantities are not space-free and scoring
everything in dB is the error that made the first FIG C.7 wrong. That makes the SHAPE
of each curve comparable but NOT its absolute height. A second pass with every method
in unit-norm gives the common ground where heights can be read against each other;
both are written and the figure marks which is which.

    python make_heldout_variance.py --feature-set concat_hg concat_rawds
    python make_heldout_variance.py --feature-set concat_hg_all --spaces home
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

# Pinned, for the same reason as in make_bsf_comparison: a "best so far" must not move.
SOURCES = {
    "concat_hg_all": CLUST / "kmeans" / "concat_hg_all" / "runs" / "20260819_235524",  # BSF
    "concat_hg":     CLUST / "kmeans" / "concat_hg" / "runs" / "20260817_171544",      # SBSF
    "concat_rawds":  CLUST / "kmeans" / "concat_rawds" / "runs" / "20260817_171634",
}
LABEL = {"kmeans": "k-means", "hierarchical": "Ward", "cnmf": "convex NMF"}
ORDER = ["kmeans", "hierarchical", "cnmf"]
DEFAULT_KS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14]


def fit_block(A, k, method, seed, n_iter):
    """Fit on a rows_train x cols_train block.

    Returns (Btr, assign, Gtr) where Btr is cols_train x k, assign maps each TRAINING
    row to its representation, and Gtr is the training loadings needed to extend the
    factorisation to unseen columns."""
    if method == "cnmf":
        _, Gtr, comp = D.convex_nmf(A, k, random_state=seed, n_iter=n_iter)
        return comp.T, None, Gtr
    lab = MS.fit_any(A, k, seed, method)
    present = np.unique(lab)
    Btr = np.stack([A[lab == j].mean(0) for j in present], axis=1)   # cols_train x k'
    return Btr, lab, None


def extend(Xtr_te, method, assign, Gtr):
    """The same factorisation on the held-out COLUMNS: cols_test x k.

    cnmf   least-squares extension using the training rows' loadings
    hard   the same training rows' mean, on the test columns"""
    if method == "cnmf":
        Bte, *_ = np.linalg.lstsq(Gtr, Xtr_te, rcond=None)
        return Bte.T
    present = np.unique(assign)
    return np.stack([Xtr_te[assign == j].mean(0) for j in present], axis=1)


def loadings_of(x_ctr, Btr, method):
    """A held-out row's loadings, from the TRAIN columns only."""
    if method == "cnmf":
        from scipy.optimize import nnls
        return nnls(Btr, x_ctr)[0]
    g = np.zeros(Btr.shape[1])
    g[int(np.argmin(((Btr - x_ctr[:, None]) ** 2).sum(0)))] = 1.0
    return g


def bicv(X, ks, method, *, n_row_folds=4, n_col_folds=4, n_iter=150, seed=0,
         progress=True):
    rng = np.random.default_rng(seed)
    n, m = X.shape
    rperm, cperm = rng.permutation(n), rng.permutation(m)
    rfolds = np.array_split(rperm, n_row_folds)
    cfolds = np.array_split(cperm, n_col_folds)
    rows = []
    for k in ks:
        t0 = time.time()
        for ri, rte in enumerate(rfolds):
            rtr = np.setdiff1d(rperm, rte)
            for ci, cte in enumerate(cfolds):
                ctr = np.setdiff1d(cperm, cte)
                try:
                    Btr, assign, Gtr = fit_block(X[np.ix_(rtr, ctr)], k, method,
                                                 seed, n_iter)
                    Bte = extend(X[np.ix_(rtr, cte)], method, assign, Gtr)
                except Exception as e:
                    print(f"    !! k={k} fold {ri},{ci}: {type(e).__name__}: {e}")
                    continue
                err = num = 0.0
                for i in rte:
                    g = loadings_of(X[i, ctr], Btr, method)
                    pred = Bte @ g
                    err += float(((X[i, cte] - pred) ** 2).sum())
                    num += float((X[i, cte] ** 2).sum())
                rows.append(dict(k=int(k), row_fold=ri, col_fold=ci,
                                 var_explained=1.0 - err / max(num, 1e-12)))
        if progress:
            v = np.mean([r["var_explained"] for r in rows if r["k"] == k])
            print(f"    k={k:<3} var_explained {v:+.4f}   ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", nargs="+",
                    default=["concat_hg", "concat_rawds", "concat_bands5",
                             "concat_bands5z"])
    ap.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    ap.add_argument("--spaces", choices=["home", "unit", "both"], default="both")
    # three parallel kernels each sweep ONE method and write their own files, so they
    # can never race on a shared CSV
    ap.add_argument("--method", nargs="+", default=None,
                    choices=["kmeans", "hierarchical", "cnmf"])
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    # SOURCES below pins the OLD run ids. After the cohort is rebuilt those runs no
    # longer describe the data, so the sweep reads X straight from the dataset cache
    # instead and depends on no run at all.
    ap.add_argument("--from-cache", default=None,
                    help="dataset cache dir, e.g. outputs/_dataset/concat_source_v3")
    ap.add_argument("--row-folds", type=int, default=4)
    ap.add_argument("--col-folds", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=150)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    passes = {"home": ["home"], "unit": ["unit-norm"], "both": ["home", "unit-norm"]}[a.spaces]
    methods = a.method or ORDER
    tag = a.tag or ("_".join(methods) if a.method else "all")
    cached = None
    if a.from_cache:
        import lf_concat as CC
        inp = (ROOT.parent / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY")
        df_c, Xc = CC.build_concat_dataset(inp, conditions=("audio", "picture", "reading"),
                                           require_high_activity=True,
                                           cache_dir=Path(a.from_cache), verbose=False)
        # BUILD ONLY WHAT WAS ASKED FOR. This used to be a fixed dict of the two
        # original sets, so --feature-set concat_bands5 died on a KeyError after the
        # cache had already been read - and it built both sets every time even when
        # only one was wanted.
        builders = {
            "concat_hg": lambda: CC.concat_hg_features(Xc, hg_band=(70.0, 150.0),
                                                       fmax=500.0),
            "concat_rawds": lambda: CC.concat_rawds_features(Xc, n_blocks=3,
                                                             fmax_hz=500.0),
            "concat_bands5": lambda: CC.concat_bands5_features(Xc, n_blocks=3,
                                                               fmax_hz=500.0),
            "concat_bands5z": lambda: CC.concat_bands5z_features(Xc, n_blocks=3,
                                                                 fmax_hz=500.0),
        }
        unknown = [f for f in a.feature_set if f not in builders]
        if unknown:
            raise SystemExit(f"--from-cache cannot build {unknown}; known: "
                             f"{sorted(builders)}")
        cached = {f: builders[f]() for f in a.feature_set}
        print(f"X from cache {a.from_cache}: "
              + "  ".join(f"{k}={v.shape}" for k, v in cached.items()))

    allrows = []
    for fs in a.feature_set:
        if cached is not None:
            X0 = np.asarray(cached[fs], dtype=float)
            run = Path(a.from_cache)
        else:
            run = SOURCES[fs]
            X0 = np.load(run / "X_train.npy").astype(float)
        print(f"\n=== {fs}  X={X0.shape}  (source {run.name}) ===")
        for scheme in passes:
            for m in methods:
                space = MS.SPACE[m] if scheme == "home" else "unit-norm"
                A = MS.unit(X0) if space == "unit-norm" else X0
                print(f"  {LABEL[m]:<11} scheme={scheme:<9} space={space}")
                d = bicv(A, a.ks, m, n_row_folds=a.row_folds,
                         n_col_folds=a.col_folds, n_iter=a.n_iter)
                d["feature_set"], d["method"] = fs, m
                d["method_label"], d["space"], d["scheme"] = LABEL[m], space, scheme
                d["n"], d["p"] = X0.shape
                allrows.append(d)
                pd.concat(allrows).to_csv(OUT / f"heldout_variance_{tag}.csv", index=False)

    df = pd.concat(allrows)
    df.to_csv(OUT / f"heldout_variance_{tag}.csv", index=False)
    g = (df.groupby(["feature_set", "scheme", "method_label", "k"])["var_explained"]
           .agg(["mean", "std", "count"]).reset_index())
    g.to_csv(OUT / f"heldout_variance_summary_{tag}.csv", index=False)

    # where does each curve peak, and does it turn over at all?
    peaks = []
    for (fs, sc, ml), sub in g.groupby(["feature_set", "scheme", "method_label"]):
        s = sub.sort_values("k")
        best = s.loc[s["mean"].idxmax()]
        peaks.append(dict(feature_set=fs, scheme=sc, method_label=ml,
                          k_peak=int(best["k"]), peak=float(best["mean"]),
                          at_k8=float(s.loc[s.k == 8, "mean"].iloc[0]) if (s.k == 8).any() else None,
                          monotone=bool(s["mean"].is_monotonic_increasing),
                          k_max_tested=int(s["k"].max())))
    pd.DataFrame(peaks).to_csv(OUT / f"heldout_peaks_{tag}.csv", index=False)
    (OUT / f"heldout_meta_{tag}.json").write_text(json.dumps(dict(
        ks=a.ks, row_folds=a.row_folds, col_folds=a.col_folds, n_iter=a.n_iter,
        methods=methods,
        feature_sets=a.feature_set, spaces=a.spaces,
        sources={k: str(v.relative_to(CLUST)) for k, v in SOURCES.items()
                 if k in a.feature_set},
        written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print("\n=== peaks ===")
    print(pd.DataFrame(peaks).to_string(index=False))
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
