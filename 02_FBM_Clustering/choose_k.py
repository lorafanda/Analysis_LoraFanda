#!/usr/bin/env python3
"""
choose_k.py - three criteria for K, side by side, because the one in use cannot work.

THE PROBLEM. cv_rank_curve holds out ELECTRODES, then fits each held-out electrode
its own k loadings by NNLS. Adding a basis vector can only reduce a least-squares
residual, so the score is monotone in K by construction: on concat_hg it climbs
0.389 (K=2) to 0.674 (K=23) and never turns over. Argmax picks the largest K
tested, the 1-SE rule picks one below it, and neither is a choice. K=7 is a fixed
setting that was not even in the sweep.

THREE CRITERIA THAT CAN TURN OVER:

  bicv    Bi-cross-validation (Owen & Perry 2009, Wold-style). Hold out a block of
          FEATURES as well as a block of electrodes. Fit components on the
          electrode-train x feature-train block, estimate held-out electrodes'
          loadings on the FEATURE-TRAIN columns only, then predict the
          feature-TEST columns. The loadings never see the columns they are scored
          on, so extra components genuinely cost prediction error. This is the
          fix for the monotonicity above.

  lopo    Stability. Leave one patient out, refit, compare to the full-cohort
          solution (ARI), and score against a size-matched pseudo-patient null.
          A rank that is too high fragments differently in every fold, so the
          real-minus-null margin peaks.

  split   Split-half replication. Fit on two independent halves of the patients
          and correlate the best-matching components. Too many components and the
          extra ones stop replicating.

They answer different questions - prediction, robustness to who is in the cohort,
and reproducibility of the components themselves - so agreement between them is
worth more than any one of them.

    python choose_k.py --feature-set concat_hg
    python choose_k.py --ks 2 3 4 5 6 7 8 10 12 --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_decompose as D  # noqa: E402

CLUST = ROOT / "outputs" / "clustering"
DEC = CLUST / "decomposition"
SOURCES = {
    "concat_hg":    CLUST / "kmeans" / "concat_hg" / "runs" / "20260817_171544",
    "concat_rawds": CLUST / "kmeans" / "concat_rawds" / "runs" / "20260817_171634",
}


def bicv_curve(X, ks, *, n_row_folds=4, n_col_folds=4, n_iter=150, seed=0):
    """Held-out BLOCK prediction error as a function of K.

    For each (row-fold, col-fold): fit on rows_train x cols_train, solve each
    held-out row's loadings on cols_train, predict cols_test, score there. The
    loadings are estimated without ever seeing the scored columns, which is
    exactly what the electrode-only scheme fails to do.
    """
    from scipy.optimize import nnls
    rng = np.random.default_rng(seed)
    n, m = X.shape
    rperm, cperm = rng.permutation(n), rng.permutation(m)
    rfolds = np.array_split(rperm, n_row_folds)
    cfolds = np.array_split(cperm, n_col_folds)
    rows = []
    for k in ks:
        for ri, rte in enumerate(rfolds):
            rtr = np.setdiff1d(rperm, rte)
            for ci, cte in enumerate(cfolds):
                ctr = np.setdiff1d(cperm, cte)
                try:
                    _, Gtr, comp = D.convex_nmf(X[np.ix_(rtr, ctr)], k,
                                                random_state=seed, n_iter=n_iter)
                except Exception:
                    continue
                Btr = comp.T                       # cols_train x k
                # Extend the SAME factorisation to the held-out columns: solve for
                # the test-column loadings using the training rows' G. Fitting a
                # second convex_nmf and borrowing its components is wrong - two
                # independent fits have no shared component order or scale, so the
                # prediction is meaningless and the score comes out negative at
                # every K, which is exactly what the first version of this did.
                Bte, *_ = np.linalg.lstsq(Gtr, X[np.ix_(rtr, cte)], rcond=None)
                Bte = Bte.T                        # cols_test x k
                err = num = 0.0
                for i in rte:
                    g, _ = nnls(Btr, X[i, ctr])
                    pred = Bte @ g
                    err += float(((X[i, cte] - pred) ** 2).sum())
                    num += float((X[i, cte] ** 2).sum())
                rows.append(dict(k=int(k), row_fold=ri, col_fold=ci,
                                 err=err / max(num, 1e-12),
                                 var_explained=1.0 - err / max(num, 1e-12)))
    return pd.DataFrame(rows)


def stability_curve(X, pat, ks, *, n_rep=4, seed=0):
    """Leave-one-patient-out ARI minus its size-matched null, per K."""
    rows = []
    for k in ks:
        try:
            real = D.lopo_stability(X, pat, k)
            null = D.pseudo_group_null(X, pat, k, n_rep=n_rep, random_state=seed)
        except Exception as e:
            print(f"    K={k} stability failed: {type(e).__name__}: {e}")
            continue
        ok = ~real["degenerate"]
        rmin = float(real.loc[ok, "ari"].min()) if ok.any() else np.nan
        rmean = float(real.loc[ok, "ari"].mean()) if ok.any() else np.nan
        nmin, nsd = float(null["min"].mean()), float(null["min"].std())
        rows.append(dict(k=int(k), real_min=rmin, real_mean=rmean,
                         null_min=nmin, null_sd=nsd,
                         margin=rmin - nmin,
                         z=(rmin - nmin) / max(nsd, 1e-9),
                         n_degenerate=int((~ok).sum())))
    return pd.DataFrame(rows)


def split_curve(X, pat, ks, *, n_rep=6, n_iter=150, seed=0):
    """Best-matching component correlation across independent patient halves."""
    rng = np.random.default_rng(seed)
    ps = np.unique(pat)
    rows = []
    for k in ks:
        vals = []
        for rep in range(n_rep):
            sh = rng.permutation(ps)
            A = set(sh[:len(ps) // 2])
            ma = np.isin(pat, list(A))
            try:
                _, _, Ca = D.convex_nmf(X[ma], k, random_state=seed, n_iter=n_iter)
                _, _, Cb = D.convex_nmf(X[~ma], k, random_state=seed, n_iter=n_iter)
            except Exception:
                continue
            M = np.corrcoef(Ca, Cb)[:k, k:]
            vals.append(float(M.max(1).mean()))
        if vals:
            rows.append(dict(k=int(k), replication=float(np.mean(vals)),
                             sd=float(np.std(vals)), n_rep=len(vals)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", action="append", choices=sorted(SOURCES))
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", action="append", choices=["bicv", "lopo", "split"],
                    default=[])
    a = ap.parse_args()
    sets = a.feature_set or ["concat_hg"]
    ks = a.ks[:6] if a.quick else a.ks
    n_iter = 50 if a.quick else 150

    for fs in sets:
        src = SOURCES[fs]
        X = np.load(src / "X_train.npy").astype(np.float64)
        lab = pd.read_csv(src / "labels.csv")
        pat = lab["patient_id"].astype(str).to_numpy()
        Xs = D.unit_norm(X)
        out = DEC / fs
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {fs}: {X.shape[0]} electrodes x {X.shape[1]} features, "
              f"{len(np.unique(pat))} patients | K in {ks}")

        res = {}
        if "bicv" not in a.skip:
            print("  bi-cross-validation ...")
            b = bicv_curve(Xs, ks, n_iter=n_iter,
                           n_row_folds=3 if a.quick else 4,
                           n_col_folds=3 if a.quick else 4)
            b.to_csv(out / "k_bicv.csv", index=False)
            res["bicv"] = b.groupby("k")["var_explained"].agg(["mean", "std"])
            print(res["bicv"].round(4).to_string())
        if "lopo" not in a.skip:
            print("  leave-one-patient-out vs null ...")
            s = stability_curve(Xs, pat, ks, n_rep=2 if a.quick else 4)
            s.to_csv(out / "k_stability.csv", index=False)
            res["lopo"] = s.set_index("k")[["margin", "z", "real_min", "null_min"]]
            print(res["lopo"].round(4).to_string())
        if "split" not in a.skip:
            print("  split-half replication ...")
            r = split_curve(Xs, pat, ks, n_rep=3 if a.quick else 6, n_iter=n_iter)
            r.to_csv(out / "k_split.csv", index=False)
            res["split"] = r.set_index("k")[["replication", "sd"]]
            print(res["split"].round(4).to_string())

        # ---- one figure, three criteria
        n = len(res)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0))
        axes = np.atleast_1d(axes)
        picks = {}
        for ax, (name, tab) in zip(axes, res.items()):
            if name == "bicv":
                m, sd = tab["mean"], tab["std"]
                ax.errorbar(m.index, m, yerr=sd, marker="o", capsize=3)
                picks[name] = int(m.idxmax())
                ax.set_ylabel("held-out BLOCK variance explained")
                ax.set_title("bi-cross-validation\n(loadings never see the scored columns)",
                             fontsize=9.5, loc="left")
            elif name == "lopo":
                ax.plot(tab.index, tab["margin"], marker="o")
                ax.axhline(0, color="#c1121f", ls="--", lw=1)
                picks[name] = int(tab["margin"].idxmax())
                ax.set_ylabel("LOPO min  −  matched null")
                ax.set_title("stability vs a size-matched null\n(0 = indistinguishable "
                             "from random groups)", fontsize=9.5, loc="left")
            else:
                ax.errorbar(tab.index, tab["replication"], yerr=tab["sd"],
                            marker="o", capsize=3)
                picks[name] = int(tab["replication"].idxmax())
                ax.set_ylabel("best-match component r")
                ax.set_title("split-half replication\n(independent patient halves)",
                             fontsize=9.5, loc="left")
            ax.axvline(picks[name], color="#1f77b4", ls=":", lw=1.4)
            ax.text(picks[name], ax.get_ylim()[0], f" K={picks[name]}",
                    fontsize=9, color="#1f77b4", va="bottom")
            ax.axvline(7, color="#8a8f96", ls="-", lw=1, alpha=.7)
            ax.set_xlabel("components")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Choosing K on {fs} — grey line is the K=7 currently in use",
                     x=.02, ha="left", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(out / "K_selection.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

        (out / "k_selection.json").write_text(json.dumps(dict(
            feature_set=fs, ks=list(ks), picks=picks, k_in_use=7, quick=bool(a.quick),
            written=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), indent=2),
            encoding="utf-8")
        print(f"\n  picks: {picks}   (currently using K=7)")
        print(f"  -> {out / 'K_selection.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
