#!/usr/bin/env python3
"""FIG 5 - every solution against every other: one number per pair, and its chance.

    python 00_paper2_figure5_confusion.py                 # K = 8
    python 00_paper2_figure5_confusion.py --k 6 7 8 9 10  # one figure per K, plus the K-by-K panel

THE QUESTION. FIG 4 compares solutions two at a time and only along one axis at a time
(algorithms on one set; feature sets under convex NMF). This puts all twelve solutions -
three algorithms x four feature sets - on both axes of one matrix, so "does the
algorithm matter more than the representation" is read off one picture.

THE NUMBER. For a pair of solutions at the same K, the contingency table C (k x k,
C[i, j] = electrodes in cluster i of one and cluster j of the other) is matched one to
one by the Hungarian algorithm on the shared electrodes themselves - the matching that
maximises the diagonal - and the cell reports

    sum of the matched diagonal / sum of the whole table   (= fraction of the cohort)

which is 1 for identical partitions and falls towards the chance level for unrelated
ones. Unlike FIG 4, the matching here is NOT independent of what it is scored on; that
is what makes it the ordinary confusion-matrix accuracy, and why the null matters.

THE NULL. The electrode labels of the second solution are shuffled across electrodes
(cluster sizes kept) and the whole thing - table, Hungarian matching, share - is redone,
n_perm times. Reported per cell: the null mean, its 95th percentile, an empirical p, and
the chance-corrected share (obs - null) / (1 - null), 0 at chance and 1 at identity.

THE K-BY-K PANEL (when several K are given). For each solution, the same share between
its own partition at K_a and at K_b (rectangular table, min(K_a, K_b) matched pairs):
how much of a partition survives when K moves. One small matrix per solution.

Every run is the newest under outputs/clustering/<method>/<feature_set>/runs; all of them
must have been fitted on the same cohort, which is checked, not assumed. Written to
outputs/clustering/paper_figures/FIG5_*.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "clustering" / "paper_figures"

# the loader and the labels FIG 4 uses, so the two figures cannot disagree on a run
_spec = importlib.util.spec_from_file_location("f4", ROOT / "00_paper2_figure4_correspondence.py")
F4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F4)

# the shared module, for paper mode (fs / pw / paper_dir / out_dpi; 2026-10-27)
_p2spec = importlib.util.spec_from_file_location("p2_shared", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_p2spec)
_p2spec.loader.exec_module(P2)

METHODS = ["kmeans", "hierarchical", "cnmf"]
# the four current feature sets (2026-10-21; was concat_hg / rawds / bands5z, all retired)
FSETS = ["concat_bands15", "concat_bands5", "concat_bands2", "concat_hfa"]
MAIN_FSET = F4.ALGO_FSET                 # names the files, as FIG 4 does
ML = {"kmeans": "k-means", "hierarchical": "Ward", "cnmf": "cNMF"}
FL = {"concat_bands15": "15 bands", "concat_bands5": "5 bands", "concat_bands2": "2 bands",
      "concat_hfa": "HFA"}
INK, MUTED = "#1b1b1b", "#6b6b6b"
CMAP = LinearSegmentedColormap.from_list("share", ["#f7f7f5", "#c9d7e6", "#5b8fc4", "#1f4e79"])


# ---- the number ----------------------------------------------------------------------------
def contingency(l1, l2, k1, k2):
    A = np.zeros((k1, k2), float)
    np.add.at(A, (l1, l2), 1.0)
    return A


def hungarian_share(l1, l2, k1, k2):
    """Diagonal share after the matching that maximises it."""
    A = contingency(l1, l2, k1, k2)
    r, c = linear_sum_assignment(-A)
    return float(A[r, c].sum() / A.sum()), A, (r, c)


def null_shares(l1, l2, k1, k2, n_perm, rng):
    """The same share with the second solution's labels shuffled across electrodes."""
    out = np.empty(n_perm)
    for p in range(n_perm):
        out[p] = hungarian_share(l1, rng.permutation(l2), k1, k2)[0]
    return out


# ---- the solutions -----------------------------------------------------------------------
def load_all(k):
    sols = {}
    for m in METHODS:
        for f in FSETS:
            sols[(m, f)] = F4.solution(m, f, k)
    keys0 = next(iter(sols.values()))["keys"]
    for (m, f), s in sols.items():
        if s["keys"] != keys0:
            raise SystemExit(f"{m}/{f} run {s['run'].name} is on a different cohort "
                             f"({len(s['keys'])} vs {len(keys0)} electrodes) - refit before comparing")
    return sols, len(keys0)


def matrix_at_k(k, n_perm, seed):
    sols, n = load_all(k)
    names = list(sols)
    rng = np.random.default_rng(seed)
    S = np.eye(len(names)); NULL = np.full((len(names), len(names)), np.nan)
    P95 = np.full_like(NULL, np.nan); PV = np.full_like(NULL, np.nan); ADJ = np.eye(len(names))
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            la, lb = sols[a]["lab"], sols[b]["lab"]
            obs, A, _ = hungarian_share(la, lb, k, k)
            nul = null_shares(la, lb, k, k, n_perm, rng)
            p = (np.sum(nul >= obs) + 1) / (n_perm + 1)
            adj = (obs - nul.mean()) / max(1.0 - nul.mean(), 1e-9)
            S[i, j] = S[j, i] = obs; NULL[i, j] = NULL[j, i] = nul.mean()
            P95[i, j] = P95[j, i] = np.percentile(nul, 95); PV[i, j] = PV[j, i] = p
            ADJ[i, j] = ADJ[j, i] = adj
            rows.append(dict(k=k, method_1=a[0], feature_1=a[1], method_2=b[0], feature_2=b[1],
                             same_algorithm=a[0] == b[0], same_feature=a[1] == b[1],
                             diagonal_share=obs, null_mean=nul.mean(), null_p95=np.percentile(nul, 95),
                             p_value=p, chance_corrected=adj, n_electrodes=n, n_perm=n_perm,
                             run_1=sols[a]["run"].name, run_2=sols[b]["run"].name))
    return dict(k=k, names=names, S=S, NULL=NULL, P95=P95, PV=PV, ADJ=ADJ, rows=pd.DataFrame(rows),
                n=n, sols=sols)


# ---- drawing --------------------------------------------------------------------------------
def label(nm):
    if P2.paper():                  # the algorithm is a group label on paper
        return FL[nm[1]]
    return f"{ML[nm[0]]}\n{FL[nm[1]]}"


def group_labels(ax, names):
    """PAPER: one algorithm name per block, under the x ticks and, on the first panel
    only, left of the y ticks."""
    nf = len(FSETS)
    first = ax.get_subplotspec().colspan.start == 0
    for b in range(len(names) // nf):
        c = b * nf + (nf - 1) / 2
        ax.text(c, len(names) - 0.5 + 4.2, ML[names[b * nf][0]], ha="center", va="top",
                fontsize=P2.fs(7.4), color=INK, weight="bold")
        if first:
            ax.text(-3.9, c, ML[names[b * nf][0]], ha="right", va="center", rotation=90,
                    fontsize=P2.fs(7.4), color=INK, weight="bold")


def heat(ax, M, names, title, vmin, vmax, pv=None, fmt="{:.2f}"):
    if P2.paper():                  # the diagonal is 1 by construction: masked white
        M = np.array(M, float).copy()
        M[np.arange(len(names)), np.arange(len(names))] = np.nan
    im = ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=vmax)
    nn = len(names)
    for i in range(nn):
        for j in range(nn):
            v = M[i, j]
            if np.isnan(v):
                continue
            dark = (v - vmin) / (vmax - vmin) > 0.55
            s = fmt.format(v)
            if pv is not None and i != j and pv[i, j] >= 0.05:
                s += "\nns"
            ax.text(j, i, s, ha="center", va="center", fontsize=P2.fs(6.6) if nn <= 9 else P2.fs(5.4),
                    color="white" if dark else INK, fontweight="bold" if i != j else "normal")
    ax.set_xticks(range(nn)); ax.set_yticks(range(nn))
    ax.set_xticklabels([label(n) for n in names], fontsize=P2.fs(6.4),
                       rotation=90 if P2.paper() else 40, ha="center" if P2.paper() else "right")
    ax.set_yticklabels([label(n) for n in names], fontsize=P2.fs(6.4))
    if P2.paper():
        group_labels(ax, names)
    ax.tick_params(length=0)
    # block lines between algorithms (names are method-major: len(FSETS) per method)
    for b in np.arange(len(FSETS), nn - 1, len(FSETS)) - 0.5:
        ax.axhline(b, color="white", lw=P2.pw(2.2)); ax.axvline(b, color="white", lw=P2.pw(2.2))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=P2.fs(9), loc="left", color=INK)
    return im


def draw(res, path, n_perm):
    k, names = res["k"], res["names"]
    if P2.paper():
        # A and C only, twice the size; the chance matrix is one number in the legend
        # (2026-10-27)
        fig, axs = plt.subplots(1, 2, figsize=(P2.canvas(12.4), 6.4))
        im0 = heat(axs[0], res["S"], names, f"A   diagonal share, K = {k}", 0, 1, pv=res["PV"])
        im2 = heat(axs[1], res["ADJ"], names, "B   chance-corrected", 0, 1, pv=res["PV"])
        cb = fig.colorbar(im2, ax=list(axs), fraction=0.02, pad=P2.pw(0.02))
        cb.ax.tick_params(labelsize=P2.fs(7), colors=MUTED, length=2)
        plt.savefig(path, dpi=P2.out_dpi(200), bbox_inches="tight"); plt.close(fig)
        return
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.6) if len(names) <= 9 else (18.5, 6.6))
    im0 = heat(axs[0], res["S"], names, f"A   diagonal share, K = {k}   (Hungarian on shared electrodes)", 0, 1, pv=res["PV"])
    heat(axs[1], res["NULL"], names, f"B   chance: mean of {n_perm} label shuffles", 0, 1)
    im2 = heat(axs[2], res["ADJ"], names, "C   chance-corrected  (share - chance) / (1 - chance)", 0, 1, pv=res["PV"])
    for ax in axs:
        ax.text(-0.02, 1.06, "algorithm x feature set", transform=ax.transAxes, fontsize=P2.fs(6.8), color=MUTED, va="bottom")
    cb = fig.colorbar(im2, ax=axs, fraction=0.012, pad=P2.pw(0.01))
    cb.ax.tick_params(labelsize=P2.fs(7), colors=MUTED, length=2)
    R = res["rows"]
    same_a = R[R.same_algorithm & ~R.same_feature]; same_f = R[R.same_feature & ~R.same_algorithm]; nei = R[~R.same_algorithm & ~R.same_feature]
    fig.suptitle(f"FIG 5   {len(names)} solutions against each other at K = {k}   -   {res['n']} electrodes   -   "
                 f"across algorithms (same features) median {same_f.diagonal_share.median():.2f}, "
                 f"across features (same algorithm) {same_a.diagonal_share.median():.2f}, both differ {nei.diagonal_share.median():.2f}   "
                 f"-   chance {R.null_mean.median():.2f}", fontsize=P2.fs(9.5), color=INK, x=0.02, ha="left")
    # just under the rotated tick labels; at 0.005 it left 0.7 in of white under the matrices
    fig.text(0.02, 0.075, "'ns' = share not above the 95th percentile of the null (p >= 0.05)", fontsize=P2.fs(7), color=MUTED)
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def kk_panel(ks, n_perm, seed, path):
    """Per solution, the share between its own partitions at K_a and K_b."""
    rng = np.random.default_rng(seed)
    sols = {k: load_all(k)[0] for k in ks}
    names = list(sols[ks[0]])
    fig, axs = plt.subplots(len(METHODS), len(FSETS), figsize=(3.6 * len(FSETS), 3.5 * len(METHODS)))
    rows = []
    for ax, nm in zip(axs.ravel(), names):
        M = np.eye(len(ks)); ADJ = np.eye(len(ks))
        for i, ka in enumerate(ks):
            for j, kb in enumerate(ks):
                if j <= i:
                    continue
                la, lb = sols[ka][nm]["lab"], sols[kb][nm]["lab"]
                obs = hungarian_share(la, lb, ka, kb)[0]
                nul = null_shares(la, lb, ka, kb, max(100, n_perm // 10), rng)
                M[i, j] = M[j, i] = obs
                ADJ[i, j] = ADJ[j, i] = (obs - nul.mean()) / max(1 - nul.mean(), 1e-9)
                rows.append(dict(method=nm[0], feature=nm[1], k_a=ka, k_b=kb, diagonal_share=obs, null_mean=nul.mean(), chance_corrected=ADJ[i, j]))
        im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1)
        for i in range(len(ks)):
            for j in range(len(ks)):
                v = M[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=P2.fs(7), color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(ks))); ax.set_yticks(range(len(ks)))
        ax.set_xticklabels([f"K={k}" for k in ks], fontsize=P2.fs(7)); ax.set_yticklabels([f"K={k}" for k in ks], fontsize=P2.fs(7))
        ax.tick_params(length=0); ax.set_title(f"{ML[nm[0]]} · {FL[nm[1]]}", fontsize=P2.fs(9), loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=P2.pw(0.01)); cb.ax.tick_params(labelsize=P2.fs(7), colors=MUTED, length=2)
    fig.suptitle("FIG 5 K-by-K   each solution against itself at another K: diagonal share after Hungarian matching on the "
                 "rectangular table (min(K_a, K_b) pairs)", fontsize=P2.fs(9.5), x=0.02, ha="left", color=INK)
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame(rows)


# ---- main --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[8])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paper", action="store_true",
                    help="draw for the page (174 mm, printed font sizes, no prose or stamp; "
                         "PNG under paper_figures/paper/)")
    a = ap.parse_args()
    if a.paper:
        os.environ["PAPER2_PAPER"] = "1"
    OUT = P2.paper_dir(globals()["OUT"])
    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    for k in a.k:
        t0 = datetime.now()
        res = matrix_at_k(k, a.n_perm, a.seed)
        run_tag = res["sols"][("cnmf", MAIN_FSET)]["run"].name.replace("_", "-")
        tag = f"K{k:02d}_run{run_tag}"
        draw(res, OUT / f"FIG5_{tag}.png", a.n_perm)
        res["rows"].to_csv(OUT / f"FIG5_pairs_{tag}.csv", index=False)
        allrows.append(res["rows"])
        R = res["rows"]
        print(f"K={k}: {res['n']} electrodes, {len(R)} pairs, {a.n_perm} shuffles each, "
              f"{(datetime.now() - t0).seconds}s | share: across algorithms median "
              f"{R[R.same_feature].diagonal_share.median():.3f}, across features "
              f"{R[R.same_algorithm].diagonal_share.median():.3f}, both differ "
              f"{R[~R.same_feature & ~R.same_algorithm].diagonal_share.median():.3f} | chance "
              f"{R.null_mean.median():.3f} | pairs not above chance: {int((R.p_value >= 0.05).sum())}/{len(R)}"
              f" -> FIG5_{tag}.png")
    if len(a.k) > 1:
        # named with the run id like every other paper figure, so the web block keeps one
        rt = "run" + F4.LR.newest_run("cnmf", MAIN_FSET).name.replace("_", "-")
        pd.concat(allrows, ignore_index=True).to_csv(OUT / f"FIG5_pairs_allK_{rt}.csv", index=False)
        kk = kk_panel(a.k, a.n_perm, a.seed, OUT / f"FIG5_KbyK_{rt}.png")
        kk.to_csv(OUT / f"FIG5_KbyK_{rt}.csv", index=False)
        print(f"K-by-K panel over K={a.k} -> FIG5_KbyK_{rt}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
