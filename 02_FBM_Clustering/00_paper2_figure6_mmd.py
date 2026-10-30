#!/usr/bin/env python3
"""FIG 6 - how distinct are the types, and which features set them apart.

Maximum mean discrepancy (MMD, Gretton et al. 2012) between every pair of clusters of a
solution, as the separation measure beside the silhouette test of
make_cluster_statistics.py. Agreement (ARI, NMI, diagonal share, Jaccard: FIG 2, 4, 5) and
stability (consensus, refits, LOPO: consensus_stability.py, the statistics) are covered
elsewhere; this is the pairwise distance between the types as distributions, with the
features that carry it.

    python 00_paper2_figure6_mmd.py                       # the main cell: 15 bands x convex NMF, K 7
    python 00_paper2_figure6_mmd.py --all                 # every cell (4 feature sets x 3 methods) + the overview
    python 00_paper2_figure6_mmd.py --figures-only        # redraw from the saved tables (e.g. with --paper)
    python 00_paper2_figure6_mmd.py --selftest DIR        # synthetic data, every function once, nothing of the repo touched

THE STATISTIC. Unbiased MMD^2 with a Gaussian kernel on the unit-norm rows - the space the
three methods fit in and the silhouette test scores in (measure_cluster_stability.SPACE).
One bandwidth for everything: the median pairwise distance over the cohort, so every pair,
every version and the null are on the same scale. MMD^2 = 0 means the two clusters are the
same distribution; it grows with the distance between them relative to the kernel width.

TWO VERSIONS of every number for a convex NMF solution: "argmax" (every electrode belongs to
the cluster of its largest loading, the membership every count and test of the paper uses)
and "cut" (only the electrodes whose own-cluster loading is >= LOAD_CUT, the set the figures
draw). The gap between them is the sensitivity of the separation to the cut. k-means and
Ward have no loadings, so they have the argmax version only.

THREE ANSWERS per pair.
  1. IN-SAMPLE. MMD^2 between the published clusters, against a WITHIN-PATIENT shuffle of
     the two labels (each patient keeps its own counts of the two clusters; only which of
     its electrodes carry which label moves), Benjamini-Hochberg over the pairs. Stated
     caveat: the labels were fitted on these electrodes, so this p is circular; the shuffle
     removes the patient-composition artefact and nothing else.
  2. HELD-OUT PATIENTS. The patients are split in two halves (balanced by electrode count),
     the method is refitted on half A, its clusters are aligned to the published ones on
     half A (Hungarian matching on the contingency table), half B's electrodes are PLACED by
     the leave-one-patient-out rule of make_cluster_statistics (convex NMF: non-negative
     least squares onto the components; k-means and Ward: nearest centroid) and MMD^2 is
     taken between the placed groups on half B. Any nearest-centroid split of a single
     cloud gives MMD^2 > 0 by construction, so the claim is not p = 0: it is the RATIO to
     the Gaussian cloud (measure_cluster_stability.surrogate - one cloud with the data's
     own covariance and no clusters) pushed through the identical procedure. The within-
     patient shuffle is run on half B as well, where the labels were not fitted. For the
     "cut" version the cloud is cut by the real data's FRACTION of electrodes below the cut,
     not by the threshold itself: the cloud's loadings are flatter (median own loading 0.29
     against 0.41 on the data), and a threshold of 0.3 would remove half of it. A ratio
     below 1 means the two types overlap more than two sectors of a featureless cloud
     carved by the same method; the within-patient test on the held-out half says whether
     they differ at all (the dot in the matrices: p < 0.05 in at least 90 % of the splits).
  3. WHERE. Per bin (condition x band x time) the standardised mean difference in dB
     between the two clusters, with the same within-patient shuffle and Benjamini-Hochberg
     over the bins, drawn like FIG 3b's inside-minus-outside panel; and each bin's share of
     the LINEAR-kernel MMD^2 (the squared mean difference in the kernel space), summed per
     band and per condition. The Gaussian-kernel MMD itself has no per-bin decomposition;
     the map is the linear one, and the caption says so.

HELD-OUT CONFUSION (2026-10-30). Where do a type's electrodes land when a model that never
saw their patient places them? Two versions, both recorded per cell as a K x K confusion
(published type in rows, placed type in columns), per-type recall and precision, the type
most confused with, and per electrode a hit rate and the type it goes to when it misses:
  LOPO     leave one patient out, refit on the other 28, place the held-out patient (the
           statistics' own rule, n_lopo_seeds refits per fold): the model is nearly the
           published one and only the patient is new. The headline.
  halves   the same patient halves as the MMD: refit on 14-15 patients, place the other
           half. Stricter, because it mixes refit instability on half a cohort with the
           placement; about n_splits / 2 placements per electrode.
This is separation in plain units - "a c1 electrode from an unseen patient lands in c1 X %
of the time" - out of sample by construction, no kernel, no cloud.

THE FIGURES (paper_figures/, FIG6_mmd_*; the paper versions under paper/ with --paper).
  main      the main cell: the two matrices (argmax | cut), per type its separation from
            the nearest other type against the per-type consensus of the patient-resampling
            sweep, FIG 4's per-type overlap and the per-type silhouette, and the maps of the
            two least separated pairs in both versions.
  cell      one page per cell: the matrices and, per type, the map of its nearest pair.
  pairs     every pair's map, one grid per version (the main cell by default; --grids for
            every cell drawn).
  overview  the twelve cells' matrices on one page (after --all).
  heldout_confusion   per cell, the two K x K confusions, LOPO and halves (recall on the diagonal).

OUTPUTS  outputs/clustering/mmd/<cache>/<feature_set>/<method>/ (tables, maps, manifest with
the md5 of every input before and after). Nothing published is rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import measure_cluster_stability as MS                    # noqa: E402
import lf_decompose as LD                                 # noqa: E402

_dspec = importlib.util.spec_from_file_location("p2driver", ROOT / "run_paper_figures_v17.py")
DRV = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(DRV)
K_BY_FSET, MAIN, K_MAIN = DRV.K, DRV.MAIN, DRV.K_MAIN

OUT = P2.OUT
MMD_ROOT = ROOT / "outputs" / "clustering" / "mmd"
CONS_ROOT = ROOT / "outputs" / "clustering" / "consensus"
INK, MUTED, GREY, RED, GREEN = P2.INK, P2.MUTED, P2.GREY, P2.RED, P2.GREEN
CUT = P2.LOAD_CUT
METHODS = ["cnmf", "kmeans", "hierarchical"]
FS_SHORT = {"concat_bands15": "15 bands", "concat_bands5": "5 bands",
            "concat_bands2": "2 bands", "concat_hfa": "HFA"}
MIN_GROUP = 5                 # a pair needs at least this many electrodes on each side
D_VLIM = 1.5                  # the maps' colour range, in standardised-difference units


# ---------------------------------------------------------------- small helpers
def bh(p):
    """Benjamini-Hochberg q-values; NaN p stays NaN."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    """A path relative to the repo folder when it is inside it (the selftest is not)."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def pairs_of(K):
    return [(i, j) for i in range(K) for j in range(i + 1, K)]


def median_bandwidth(A, rng, n_max=2000):
    """The median pairwise Euclidean distance, on a subsample when n is large."""
    idx = rng.choice(len(A), min(len(A), n_max), replace=False)
    B = A[idx]
    sq = (B * B).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * B @ B.T
    np.maximum(D2, 0.0, out=D2)
    iu = np.triu_indices(len(B), 1)
    return float(np.sqrt(np.median(D2[iu])))


def rbf_kernel(A, sigma):
    sq = (A * A).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * A @ A.T
    np.maximum(D2, 0.0, out=D2)
    Kmat = np.exp(-D2 / (2.0 * sigma * sigma))
    np.fill_diagonal(Kmat, 1.0)
    return Kmat


def mmd2_batch(Kp, S):
    """Unbiased MMD^2 for B labelings at once.

    Kp  (p, p) kernel over the pair's electrodes, diagonal 1
    S   (p, B) 0/1, column b = which of the p electrodes are sample 1 in labeling b
    """
    S = np.asarray(S, float)
    m = S.sum(0)
    n = S.shape[0] - m
    KS = Kp @ S
    rows = Kp.sum(1)[:, None]
    KT = rows - KS
    s11 = (S * KS).sum(0) - m
    s22 = ((1.0 - S) * KT).sum(0) - n
    s12 = (S * KT).sum(0)
    return s11 / (m * (m - 1)) + s22 / (n * (n - 1)) - 2.0 * s12 / (m * n)


def within_patient_masks(s, pat, B, rng):
    """(p, B) labelings: s shuffled within each patient, B times, column-independent."""
    S = np.repeat(np.asarray(s, float)[:, None], B, axis=1)
    for g in np.unique(pat):
        rows = np.where(pat == g)[0]
        if len(rows) > 1:
            S[rows, :] = rng.permuted(S[rows, :], axis=0)
    return S


def split_patients(pats, counts, rng):
    """Two halves of the patients, balanced by electrode count (greedy on a random order)."""
    order = rng.permutation(len(pats))
    tot, half = [0, 0], [[], []]
    for i in order:
        h = 0 if tot[0] <= tot[1] else 1
        half[h].append(pats[i])
        tot[h] += counts[i]
    return half


def fit_and_place(A_fit, A_new, K, method, seed):
    """Labels on the fitted half, labels and own loadings on the placed half.

    convex NMF: components from the fit; a new electrode gets loadings by non-negative
    least squares onto them, the argmax of the row-normalised loadings is its cluster
    and the maximum its own loading (the pipeline's hardening rule, as in
    make_cluster_statistics.fit_with_predictor).
    k-means / Ward: nearest centroid of the fitted clusters; no loading exists.
    """
    if method == "cnmf":
        from scipy.optimize import nnls
        _, G, comp = LD.convex_nmf(A_fit, K, n_iter=300, random_state=seed)
        Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
        Bm = comp.T
        H = np.vstack([nnls(Bm, x)[0] for x in A_new])
        Hn = H / np.maximum(H.sum(1, keepdims=True), 1e-12)
        return Gn.argmax(1), Gn.max(1), Hn.argmax(1), Hn.max(1)
    lab = MS.fit_any(A_fit, K, seed, method)
    cent = np.vstack([A_fit[lab == c].mean(0) if (lab == c).any()
                      else np.full(A_fit.shape[1], np.inf) for c in range(K)])
    d2 = ((A_new[:, None, :] - cent[None, :, :]) ** 2).sum(2)
    return lab, None, d2.argmin(1), None


def align(lab_fit, lab_pub, K):
    """refit id -> published id, by Hungarian matching on the contingency of the fitted half."""
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((K, K))
    for a_, b_ in zip(lab_fit, lab_pub):
        C[a_, b_] += 1
    r, c = linear_sum_assignment(-C)
    return dict(zip(r.tolist(), c.tolist()))


def versions_of(d):
    return ["argmax", "cut"] if not d["hard"] else ["argmax"]


def members(lab, own, j, version, thr=CUT):
    m = lab == j
    if version == "cut" and own is not None:
        m = m & (own >= thr)
    return m


# ---------------------------------------------------------------- the computation
def compute_cell(d, a, log=print):
    """Every number for one cell; returns the dict the writers and figures read."""
    t0 = time.time()
    X, lab, pat, K = d["X"], d["lab"], d["patient"], d["k"]
    own = d["Gn"].max(1) if not d["hard"] else None
    frac_cut = float((own < CUT).mean()) if own is not None else 0.0
    A = MS.unit(X)
    rng = np.random.default_rng(a.seed)
    sigma = median_bandwidth(A, rng)
    Kmat = rbf_kernel(A, sigma)
    vers = versions_of(d)
    PAIRS = pairs_of(K)
    nb, nc, nt = len(d["bands"]), len(d["conds"]), d["nt"]
    F = X.shape[1]
    log(f"  n {len(X)}  K {K}  pairs {len(PAIRS)}  bandwidth {sigma:.4f}  versions {vers}")

    # ---- 1. in-sample, and 3. the maps (same shuffles) -------------------------------
    ins, maps = [], {}
    for v in vers:
        for (i, j) in PAIRS:
            mi, mj = members(lab, own, i, v), members(lab, own, j, v)
            P = np.where(mi | mj)[0]
            s = mi[P].astype(float)
            m_, n_ = int(s.sum()), int(len(P) - s.sum())
            row = dict(version=v, i=i, j=j, n_i=m_, n_j=n_)
            if min(m_, n_) < MIN_GROUP:
                ins.append(dict(row, mmd2=np.nan, null_mean=np.nan, null_sd=np.nan, z=np.nan, p=np.nan))
                continue
            Kp = Kmat[np.ix_(P, P)]
            obs = float(mmd2_batch(Kp, s[:, None])[0])
            S = within_patient_masks(s, pat[P], a.n_perm, rng)
            null = mmd2_batch(Kp, S)
            p = (1 + int((null >= obs).sum())) / (1 + a.n_perm)
            z = (obs - null.mean()) / max(null.std(), 1e-12)
            ins.append(dict(row, mmd2=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                            z=float(z), p=float(p)))
            # the map: standardised mean difference per bin, its within-patient null, and
            # each bin's share of the linear-kernel MMD^2 in the kernel space
            Xp = X[P]
            mu_i, mu_j = Xp[s > 0].mean(0), Xp[s == 0].mean(0)
            sd_i, sd_j = Xp[s > 0].std(0, ddof=1), Xp[s == 0].std(0, ddof=1)
            pooled = np.sqrt(((m_ - 1) * sd_i ** 2 + (n_ - 1) * sd_j ** 2) / (m_ + n_ - 2))
            dvec = (mu_i - mu_j) / np.maximum(pooled, 1e-12)
            diff_null = (Xp.T @ S) / m_ - (Xp.T @ (1.0 - S)) / n_          # (F, B)
            obs_diff = np.abs(mu_i - mu_j)[:, None]
            pbin = (1 + (np.abs(diff_null) >= obs_diff).sum(1)) / (1 + a.n_perm)
            qbin = bh(pbin)
            Ap = A[P]
            lin = (Ap[s > 0].mean(0) - Ap[s == 0].mean(0)) ** 2
            share = lin / max(lin.sum(), 1e-12)
            sh3 = share.reshape(nb, nc, nt)                                  # band-major storage
            maps[(v, i, j)] = dict(d=dvec.astype(np.float32), q=qbin.astype(np.float32),
                                   share=share.astype(np.float32),
                                   band_share=sh3.sum(axis=(1, 2)), cond_share=sh3.sum(axis=(0, 2)))
    ins = pd.DataFrame(ins)
    for v in vers:
        m = ins.version == v
        ins.loc[m, "q"] = bh(ins.loc[m, "p"].to_numpy())
    log(f"  in-sample: {len(ins)} rows, {a.n_perm} within-patient shuffles each  ({time.time()-t0:.0f} s)")

    # ---- 2. held-out patients, and the Gaussian cloud through the same procedure -----
    pats, counts = np.unique(pat, return_counts=True)
    ho, cloud = [], []
    conf = np.zeros((K, K))                     # published type (rows) x placed type (cols), all splits
    place_count = np.zeros((len(X), K))         # per electrode, how often it was placed in each type
    n_heldout = np.zeros(len(X))
    recall_split = []
    for sidx in range(a.n_splits):
        hA, hB = split_patients(pats, counts, rng)
        inA, inB = np.isin(pat, hA), np.isin(pat, hB)
        lab_fit, _, lab_new, own_new = fit_and_place(A[inA], A[inB], K, d["method"], a.seed + sidx)
        mp = align(lab_fit, lab[inA], K)
        placed = np.array([mp[int(c)] for c in lab_new])
        idxB = np.where(inB)[0]
        pub_B = lab[inB]
        np.add.at(conf, (pub_B, placed), 1.0)
        np.add.at(place_count, (idxB, placed), 1.0)
        n_heldout[idxB] += 1
        recall_split.append([float((placed[pub_B == r_] == r_).mean()) if (pub_B == r_).any() else np.nan
                             for r_ in range(K)])
        for v in vers:
            for (i, j) in PAIRS:
                mi, mj = members(placed, own_new, i, v), members(placed, own_new, j, v)
                P = np.where(mi | mj)[0]
                s = mi[P].astype(float)
                m_, n_ = int(s.sum()), int(len(P) - s.sum())
                row = dict(split=sidx, version=v, i=i, j=j, n_i=m_, n_j=n_)
                if min(m_, n_) < MIN_GROUP:
                    ho.append(dict(row, mmd2=np.nan, p=np.nan))
                    continue
                Kp = Kmat[np.ix_(idxB[P], idxB[P])]
                obs = float(mmd2_batch(Kp, s[:, None])[0])
                S = within_patient_masks(s, pat[idxB[P]], a.n_perm_heldout, rng)
                null = mmd2_batch(Kp, S)
                ho.append(dict(row, mmd2=obs, p=(1 + int((null >= obs).sum())) / (1 + a.n_perm_heldout),
                               z=float((obs - null.mean()) / max(null.std(), 1e-12))))
    ho = pd.DataFrame(ho)
    log(f"  held-out: {a.n_splits} splits  ({time.time()-t0:.0f} s)")
    for didx in range(a.n_cloud):
        Y = MS.unit(MS.surrogate(X, 7000 + a.seed + didx))
        hA, hB = split_patients(pats, counts, rng)
        inA, inB = np.isin(pat, hA), np.isin(pat, hB)
        lab_fit, _, lab_new, own_new = fit_and_place(Y[inA], Y[inB], K, d["method"], a.seed + 500 + didx)
        Kc = rbf_kernel(Y[inB], sigma)
        # the cloud is cut by the data's fraction: its loadings are flatter than the data's
        thr_c = float(np.quantile(own_new, frac_cut)) if own_new is not None else CUT
        for v in vers:
            for (i, j) in PAIRS:                       # the cloud's own cluster ids
                mi, mj = members(lab_new, own_new, i, v, thr_c), members(lab_new, own_new, j, v, thr_c)
                P = np.where(mi | mj)[0]
                s = mi[P].astype(float)
                if min(int(s.sum()), int(len(P) - s.sum())) < MIN_GROUP:
                    continue
                cloud.append(dict(draw=didx, version=v, i=i, j=j,
                                  mmd2=float(mmd2_batch(Kc[np.ix_(P, P)], s[:, None])[0])))
    cloud = pd.DataFrame(cloud)
    log(f"  cloud: {a.n_cloud} draws  ({time.time()-t0:.0f} s)")

    # ---- summaries ----------------------------------------------------------------
    summ = []
    for v in vers:
        cl = cloud[cloud.version == v].mmd2.to_numpy()
        c_med = float(np.nanmedian(cl)) if len(cl) else np.nan
        c_hi = float(np.nanpercentile(cl, 97.5)) if len(cl) else np.nan
        for (i, j) in PAIRS:
            h = ho[(ho.version == v) & (ho.i == i) & (ho.j == j)]
            vals = h.mmd2.to_numpy()
            ok = np.isfinite(vals)
            r = ins[(ins.version == v) & (ins.i == i) & (ins.j == j)].iloc[0]
            summ.append(dict(
                version=v, i=i, j=j, n_i=int(r.n_i), n_j=int(r.n_j),
                mmd2_in=r.mmd2, z_in=r.z, p_in=r.p, q_in=r.q,
                mmd2_ho=float(np.nanmedian(vals)) if ok.any() else np.nan,
                mmd2_ho_q25=float(np.nanpercentile(vals[ok], 25)) if ok.any() else np.nan,
                mmd2_ho_q75=float(np.nanpercentile(vals[ok], 75)) if ok.any() else np.nan,
                n_splits_ok=int(ok.sum()),
                cloud_median=c_med, cloud_p975=c_hi,
                ratio=(float(np.nanmedian(vals)) / c_med) if ok.any() and c_med > 0 else np.nan,
                share_above_cloud=float((vals[ok] > c_hi).mean()) if ok.any() else np.nan,
                p_ho_median=float(np.nanmedian(h.p)) if ok.any() else np.nan,
                z_ho_median=float(np.nanmedian(h.z)) if ok.any() else np.nan,
                share_p_ho_lt05=float((h.p[ok] < 0.05).mean()) if ok.any() else np.nan))
    summ = pd.DataFrame(summ)

    # leave one patient out: refit on the rest, align, place the held-out patient
    conf_l = np.zeros((K, K))
    place_l = np.zeros((len(X), K))
    n_l = np.zeros(len(X))
    recall_l = []
    for sd in range(a.n_lopo_seeds):
        rec = np.full(K, np.nan)
        pl_all = np.full(len(X), -1)
        for g in pats:
            held = pat == g
            lab_fit, _, lab_new, _ = fit_and_place(A[~held], A[held], K, d["method"], a.seed + 900 + sd)
            mp = align(lab_fit, lab[~held], K)
            pl_all[held] = [mp[int(c)] for c in lab_new]
        np.add.at(conf_l, (lab, pl_all), 1.0)
        np.add.at(place_l, (np.arange(len(X)), pl_all), 1.0)
        n_l += 1
        recall_l.append([float((pl_all[lab == r_] == r_).mean()) for r_ in range(K)])
    log(f"  LOPO placement: {a.n_lopo_seeds} seed(s) x {len(pats)} folds  ({time.time()-t0:.0f} s)")

    def confusion_tables(conf_, place_, n_, rec_list):
        row_tot = np.maximum(conf_.sum(1, keepdims=True), 1e-12)
        col_tot = np.maximum(conf_.sum(0, keepdims=True), 1e-12)
        R_ = conf_ / row_tot
        rs = np.array(rec_list)
        rows_ = []
        for r_ in range(K):
            off = R_[r_].copy()
            off[r_] = -1
            rows_.append(dict(cluster=r_, n_placements=int(conf_[r_].sum()), recall=float(R_[r_, r_]),
                              recall_median=float(np.nanmedian(rs[:, r_])), recall_q25=float(np.nanpercentile(rs[:, r_], 25)),
                              recall_q75=float(np.nanpercentile(rs[:, r_], 75)),
                              precision=float(conf_[r_, r_] / col_tot[0, r_]),
                              confused_with=int(off.argmax()), confused_share=float(off.max())))
        hit = np.where(n_ > 0, place_[np.arange(len(X)), lab] / np.maximum(n_, 1), np.nan)
        other = place_.copy()
        other[np.arange(len(X)), lab] = -1
        meta = d["meta"]
        el = pd.DataFrame(dict(patient=meta.patient_id.to_numpy(), electrode=meta.electrode.to_numpy(), type=lab,
                               n_heldout=n_.astype(int), hit_rate=hit, goes_to=other.argmax(1),
                               goes_to_share=np.where(n_ > 0, other.max(1) / np.maximum(n_, 1), np.nan)))
        return pd.DataFrame(rows_), el, R_

    conf_type, conf_el, recall_mat = confusion_tables(conf, place_count, n_heldout, recall_split)
    conf_type_l, conf_el_l, recall_mat_l = confusion_tables(conf_l, place_l, n_l, recall_l)
    log(f"  confusion, LOPO:   overall {np.trace(conf_l) / max(conf_l.sum(), 1):.3f}, recall "
        + " ".join(f"c{r_} {recall_mat_l[r_, r_]:.2f}" for r_ in range(K)))
    log(f"  confusion, halves: overall {np.trace(conf) / max(conf.sum(), 1):.3f}, recall "
        + " ".join(f"c{r_} {recall_mat[r_, r_]:.2f}" for r_ in range(K)))

    # per type: separation from the nearest other type (held-out ratio), per version
    per_type = []
    for v in vers:
        sv = summ[summ.version == v]
        for i in range(K):
            rows = sv[(sv.i == i) | (sv.j == i)]
            other = np.where(rows.i == i, rows.j, rows.i)
            rat = rows.ratio.to_numpy()
            if np.isfinite(rat).any():
                kmin = int(np.nanargmin(rat))
                per_type.append(dict(version=v, cluster=i, n=int((members(lab, own, i, v)).sum()),
                                     nn_cluster=int(other[kmin]), nn_ratio=float(rat[kmin]),
                                     mean_ratio=float(np.nanmean(rat)),
                                     nn_mmd2_in=float(rows.mmd2_in.to_numpy()[kmin])))
            else:
                per_type.append(dict(version=v, cluster=i, n=int((members(lab, own, i, v)).sum()),
                                     nn_cluster=-1, nn_ratio=np.nan, mean_ratio=np.nan, nn_mmd2_in=np.nan))
    per_type = pd.DataFrame(per_type)
    for tag_, ct_ in (("lopo", conf_type_l.set_index("cluster")), ("halves", conf_type.set_index("cluster"))):
        per_type[f"{tag_}_recall"] = [float(ct_.loc[int(c), "recall"]) for c in per_type.cluster]
        per_type[f"{tag_}_precision"] = [float(ct_.loc[int(c), "precision"]) for c in per_type.cluster]
        per_type[f"{tag_}_confused_with"] = [int(ct_.loc[int(c), "confused_with"]) for c in per_type.cluster]
        per_type[f"{tag_}_confused_share"] = [float(ct_.loc[int(c), "confused_share"]) for c in per_type.cluster]
    # the per-type silhouette in the same space, and its z against the cloud's per-cluster
    # silhouettes (10 draws, the method refitted on each): the scorecard's column
    sil, sil_m, sil_sd = per_type_silhouette(d, n_cloud=a.n_cloud_sil, seed=a.seed)
    per_type["silhouette"] = [sil[int(c)] for c in per_type.cluster]
    per_type["sil_cloud_mean"], per_type["sil_cloud_sd"] = sil_m, sil_sd
    per_type["sil_z"] = (per_type.silhouette - sil_m) / max(sil_sd, 1e-12)
    log(f"  done in {time.time()-t0:.0f} s")
    return dict(sigma=sigma, insample=ins, heldout=ho, cloud=cloud, summary=summ,
                per_type=per_type, maps=maps, versions=vers, pairs=PAIRS, frac_cut=frac_cut,
                confusion=conf, confusion_type=conf_type, confusion_electrodes=conf_el,
                confusion_lopo=conf_l, confusion_type_lopo=conf_type_l, confusion_electrodes_lopo=conf_el_l,
                seconds=time.time() - t0)


# ---------------------------------------------------------------- the companions (main cell)
def per_type_consensus(d, cache):
    """Mean own-cluster consensus of each published type's electrodes, from the
    patient-resampling sweep's items table at this K (NaN where the sweep has none)."""
    f = CONS_ROOT / cache / d["fset"] / d["method"] / f"items_k{d['k']:02d}.csv"
    if not f.exists():
        return None
    it = pd.read_csv(f)
    it["key"] = [f"{p}|{P2.norm(e)}" for p, e in zip(it.patient_id, it.electrode)]
    meta = d["meta"]
    key = pd.Series([f"{p}|{P2.norm(e)}" for p, e in zip(meta.patient_id, meta.electrode)])
    own = it.set_index("key").own_consensus
    vals = own.reindex(key).to_numpy()
    if np.isnan(vals).mean() > 0.5:
        return None
    return np.array([float(np.nanmean(vals[d["lab"] == i])) for i in range(d["k"])])


def per_type_fig4(d):
    """FIG 4 panel D: each cluster's mean adjusted overlap over the comparisons that put
    this cell on the first axis (the main cell: two algorithms and three feature sets)."""
    f = OUT / f"FIG4_matched_K{d['k']:02d}{P2.run_tag(d)}.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    m = m[(m.weighting == "hard") & m.comparison.str.startswith(f"{d['method']}_")]
    if m.empty:
        return None
    m = m.assign(adj=(m.overlap - m.null_plain_mean) / (1.0 - m.null_plain_mean))
    g = m.groupby("cluster_1").adj.mean()
    return np.array([float(g.get(i, np.nan)) for i in range(d["k"])])


def per_type_silhouette(d, n_cloud=10, seed=0):
    from sklearn.metrics import silhouette_samples
    A = MS.unit(d["X"])
    s = silhouette_samples(A, d["lab"])
    real = np.array([float(s[d["lab"] == i].mean()) for i in range(d["k"])])
    ref = []
    for i in range(n_cloud):
        Y = MS.unit(MS.surrogate(d["X"], 900 + seed + i))
        lb = MS.fit_any(Y, d["k"], i, d["method"])
        sc = silhouette_samples(Y, lb)
        ref.extend(float(sc[lb == c].mean()) for c in range(d["k"]) if (lb == c).any())
    ref = np.array(ref)
    return real, float(ref.mean()), float(ref.std())


# ---------------------------------------------------------------- files
def cell_dir(cache, fset, method):
    return MMD_ROOT / cache / fset / method


def save_cell(cd: Path, d, res, a, hashes):
    cd.mkdir(parents=True, exist_ok=True)
    P2.save_text(res["insample"].to_csv(index=False), cd / "pairs_insample.csv")
    P2.save_text(res["heldout"].to_csv(index=False), cd / "pairs_heldout_splits.csv")
    P2.save_text(res["cloud"].to_csv(index=False), cd / "cloud_pairs.csv")
    P2.save_text(res["summary"].to_csv(index=False), cd / "pairs_summary.csv")
    P2.save_text(res["per_type"].to_csv(index=False), cd / "per_type.csv")
    for tag_, key in (("halves", ""), ("lopo", "_lopo")):
        if res.get(f"confusion{key}") is not None:
            Kc = res[f"confusion{key}"].shape[0]
            cm = pd.DataFrame(res[f"confusion{key}"], index=[f"published_c{i}" for i in range(Kc)],
                              columns=[f"placed_c{j}" for j in range(Kc)])
            P2.save_text(cm.to_csv(), cd / f"confusion_{tag_}_counts.csv")
            P2.save_text(res[f"confusion_type{key}"].to_csv(index=False), cd / f"confusion_{tag_}_per_type.csv")
            P2.save_text(res[f"confusion_electrodes{key}"].to_csv(index=False), cd / f"confusion_{tag_}_electrodes.csv")
    arrs = {}
    for (v, i, j), m in res["maps"].items():
        for key in ("d", "q", "share", "band_share", "cond_share"):
            arrs[f"{v}_{i}_{j}_{key}"] = m[key]
    np.savez_compressed(cd / "maps.npz", **arrs)
    man = dict(feature_set=d["fset"], method=d["method"], k=int(d["k"]), run=str(d["run"].name),
               n=int(len(d["X"])), n_patients=int(d["n_patients"]), bandwidth=res["sigma"],
               cut=CUT, frac_below_cut=res["frac_cut"], versions=res["versions"], n_perm=a.n_perm,
               n_perm_heldout=a.n_perm_heldout, n_cloud_sil=a.n_cloud_sil, n_lopo_seeds=a.n_lopo_seeds,
               n_splits=a.n_splits, n_cloud=a.n_cloud, seed=a.seed, seconds=round(res["seconds"], 1),
               written=time.strftime("%Y-%m-%d %H:%M"), inputs_md5=hashes)
    P2.save_text(json.dumps(man, indent=1), cd / "manifest.json")


def load_cell(cd: Path):
    man = json.loads((cd / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(cd / "maps.npz")
    maps = {}
    for key in z.files:
        v, i, j, what = key.split("_", 3)
        maps.setdefault((v, int(i), int(j)), {})[what] = z[key]
    cf = {}
    for tag_, key in (("halves", ""), ("lopo", "_lopo")):
        if (cd / f"confusion_{tag_}_counts.csv").exists():
            cf[f"confusion{key}"] = pd.read_csv(cd / f"confusion_{tag_}_counts.csv", index_col=0).to_numpy(float)
            cf[f"confusion_type{key}"] = pd.read_csv(cd / f"confusion_{tag_}_per_type.csv")
            cf[f"confusion_electrodes{key}"] = pd.read_csv(cd / f"confusion_{tag_}_electrodes.csv")
        else:
            cf[f"confusion{key}"] = cf[f"confusion_type{key}"] = cf[f"confusion_electrodes{key}"] = None
    return dict(manifest=man, **cf,
                insample=pd.read_csv(cd / "pairs_insample.csv"),
                heldout=pd.read_csv(cd / "pairs_heldout_splits.csv"),
                cloud=pd.read_csv(cd / "cloud_pairs.csv"), summary=pd.read_csv(cd / "pairs_summary.csv"),
                per_type=pd.read_csv(cd / "per_type.csv"), maps=maps, versions=man["versions"],
                pairs=pairs_of(man["k"]), sigma=man["bandwidth"], frac_cut=man.get("frac_below_cut", 0.0))


def input_hashes(d):
    files = [d["run"] / "X_train.npy", d["run"] / "labels.csv"]
    if d["hard"]:
        files.append(d["run"] / "cluster_labels_by_k.csv")
    else:
        files.append(d["run"] / "loadings_by_k" / f"G_k{d['k']:02d}.npy")
    return {str(f.relative_to(ROOT)): md5_file(f) for f in files if f.exists()}


# ---------------------------------------------------------------- drawing helpers
def cname(d, j):
    nm = P2.type_name(d["fset"], d["k"], j)
    return f"c{j}" + (f"  {nm}" if nm else "")


def display_order(d):
    C = P2.cube(d["X"], d)
    means = np.stack([C[d["lab"] == j].mean(0) for j in range(d["k"])])
    try:
        order = P2.block_order(means, d)[0]          # FIG 1's block order (matched to the main run)
    except Exception:
        order = np.arange(d["k"])
    return [int(j) for j in order]


def matrix_panel(ax, d, summ, version, order, title, what="ratio"):
    """K x K: the held-out ratio to the cloud (colour diverging around 1, number) per pair;
    a dot where the held-out within-patient shuffle gives p < 0.05 in >= 90 % of the splits."""
    K = d["k"]
    M = np.full((K, K), np.nan)
    Q = np.full((K, K), np.nan)
    sv = summ[summ.version == version]
    for _, r in sv.iterrows():
        M[int(r.i), int(r.j)] = M[int(r.j), int(r.i)] = r[what]
        Q[int(r.i), int(r.j)] = Q[int(r.j), int(r.i)] = r.share_p_ho_lt05
    Mo = M[np.ix_(order, order)]
    Qo = Q[np.ix_(order, order)]
    with np.errstate(divide="ignore"):
        L = np.log2(Mo) if what == "ratio" else Mo
    if what == "ratio":
        vmin, vmax, cmap = -1.5, 1.5, "BrBG"
    else:
        vmin, vmax, cmap = 0.0, float(np.nanmax(np.abs(L)) or 1.0), "PuBuGn"
    im = ax.imshow(L, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    for a_ in range(K):
        for b_ in range(K):
            if a_ == b_:
                ax.add_patch(plt.Rectangle((b_ - 0.5, a_ - 0.5), 1, 1, fc="#f2f3f5", ec="none"))
                continue
            v = Mo[a_, b_]
            if np.isfinite(v):
                txt = f"{v:.1f}" if what == "ratio" else f"{v:.2f}"
                dark = (abs(L[a_, b_]) > 0.6 * vmax) if what == "ratio" else (L[a_, b_] > 0.55 * vmax)
                ax.text(b_, a_, txt, ha="center", va="center", fontsize=P2.fs(6.6),
                        color="white" if dark else INK)
                if np.isfinite(Qo[a_, b_]) and Qo[a_, b_] >= 0.9:
                    ax.plot(b_ + 0.36, a_ - 0.36, "o", ms=P2.pw(1.6), color="white" if dark else INK)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"c{j}" for j in order], fontsize=P2.fs(6.8))
    ax.set_yticklabels([f"c{j}" for j in order], fontsize=P2.fs(6.8))
    for t, j in zip(ax.get_xticklabels(), order):
        t.set_color(P2.cluster_col(j, K))
    for t, j in zip(ax.get_yticklabels(), order):
        t.set_color(P2.cluster_col(j, K))
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, fontsize=P2.fs(8.6), loc="left", color=INK, pad=P2.pw(5))
    return im


def map_panel(ax, axb, d, m, title, col=INK):
    """The standardised-difference map of one pair with the q < 0.05 outline, and the
    band shares of the linear MMD^2 beside its rows."""
    nb, nc, nt = len(d["bands"]), len(d["conds"]), d["nt"]
    D3 = m["d"].reshape(nb, nc, nt).transpose(1, 0, 2)          # (ncond, nband, nt)
    Q3 = m["q"].reshape(nb, nc, nt).transpose(1, 0, 2)
    dd = dict(conds=d["conds"], bands=d["bands"], nt=nt, fset=d["fset"])
    if nb > 1:
        P2.draw_mean(ax, D3, dd, vlim=D_VLIM, cmap=P2.ERSP_CMAP, show_axes=False, cond_names=True)
        mask = (Q3 < 0.05).transpose(1, 0, 2).reshape(nb, nc * nt)
        if mask.any():
            P2.outline_cells(ax, mask, nt)
        ax.set_yticks(range(0, nb, 3 if nb > 6 else 1))
        ax.set_yticklabels([b.replace("Hz", "") for b in d["bands"][::3 if nb > 6 else 1]],
                           fontsize=P2.fs(5.8))
        ax.tick_params(axis="y", length=2, colors=MUTED, labelsize=P2.fs(5.8))
    else:
        P2.draw_mean(ax, D3, dd, vlim=D_VLIM, col=col, ylim=(-D_VLIM, D_VLIM), show_axes=False,
                     cond_names=True)
        x = np.arange(nc * nt)
        sig = (Q3 < 0.05).reshape(-1)
        if sig.any():
            ax.plot(x[sig], np.full(sig.sum(), -D_VLIM * 0.92), "s", ms=P2.pw(1.6), color=INK)
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(axis="y", length=2, colors=MUTED, labelsize=P2.fs(5.8))
    ax.set_title(title, fontsize=P2.fs(7.6), loc="left", color=col, pad=P2.pw(17))
    if axb is not None:
        bs = m["band_share"]
        axb.barh(np.arange(nb), bs, color=MUTED, height=0.7)
        axb.set_ylim(-0.5, nb - 0.5)
        axb.set_xlim(0, max(0.05, float(bs.max()) * 1.15))
        axb.set_yticks([])
        axb.set_xticks([0, round(float(bs.max()), 2)] if bs.max() > 0 else [0])
        axb.tick_params(labelsize=P2.fs(5.4), colors=MUTED, length=2)
        axb.spines[["top", "right", "left"]].set_visible(False)
        cs = m["cond_share"]
        axb.set_title("  ".join(f"{P2.COND_LABEL[c].split(' ')[0][:3]} {100*v:.0f}%"
                                for c, v in zip(d["conds"], cs)),
                      fontsize=P2.fs(5.6), color=MUTED, loc="left", pad=P2.pw(3))


def pair_title(d, i, j, summ, version):
    r = summ[(summ.version == version) & (summ.i == i) & (summ.j == j)].iloc[0]
    rat = f"{r.ratio:.1f}× the cloud" if np.isfinite(r.ratio) else "held-out n/a"
    return f"c{i} vs c{j}   ·   MMD² {r.mmd2_in:.3f}   ·   {rat}"


# ---------------------------------------------------------------- figures
def fig_cell(d, res, a):
    """One page per cell: the matrices, the per-type separation, and per type the map of
    its nearest pair."""
    K, vers = d["k"], res["versions"]
    order = display_order(d)
    v_main = "cut" if "cut" in vers else "argmax"
    ncol = 4
    nrow_maps = int(np.ceil(K / ncol))
    fig_h = 4.3 + 3.0 * nrow_maps
    fig = plt.figure(figsize=(P2.canvas(17.6), fig_h), dpi=190)
    gs = GridSpec(1 + nrow_maps, 24, figure=fig, height_ratios=[1.15] + [1.0] * nrow_maps,
                  hspace=0.80, wspace=0.9, left=0.045, right=0.985,
                  top=1.0 - (0.35 if P2.paper() else 0.75) / fig_h, bottom=0.35 / fig_h)
    if not P2.paper():
        fig.suptitle(f"FIG 6   ·   MMD between the types   ·   {P2.METHOD_LABEL[d['method']]}   ·   "
                     f"{P2.FS_LABEL[d['fset']]}   ·   K = {K}   ·   {len(d['X'])} electrodes, "
                     f"{d['n_patients']} patients", x=0.045, y=1.0 - 0.25 / fig_h, ha="left",
                     fontsize=P2.fs(15.5), color=INK)
    axA = fig.add_subplot(gs[0, 0:6])
    matrix_panel(axA, d, res["summary"], "argmax", order,
                 "A   held-out MMD², ratio to the Gaussian cloud   (argmax members)")
    axB = fig.add_subplot(gs[0, 7:13])
    if "cut" in vers:
        im = matrix_panel(axB, d, res["summary"], "cut", order, f"B   the same, loading ≥ {CUT:.1f}")
    else:
        im = matrix_panel(axB, d, res["summary"], "argmax", order,
                          "B   in-sample z against the within-patient shuffle", what="z_in")
    cax = axB.inset_axes([1.04, 0.1, 0.035, 0.8])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log2 (held-out MMD² ÷ cloud)" if "cut" in vers else "z", fontsize=P2.fs(6.2), color=MUTED)
    cb.ax.tick_params(labelsize=P2.fs(5.8), colors=MUTED, length=2)
    cb.outline.set_visible(False)
    axC = fig.add_subplot(gs[0, 16:24])
    pt = res["per_type"]
    x = np.arange(K)
    for vi, v in enumerate(vers):
        pv = pt[pt.version == v].set_index("cluster").reindex(order)
        off = (-0.18 if len(vers) > 1 else 0.0) + 0.36 * vi
        axC.bar(x + off, pv.nn_ratio, width=0.34 if len(vers) > 1 else 0.6,
                color=[P2.cluster_col(j, K) for j in order], alpha=1.0 if v == v_main else 0.45,
                lw=P2.pw(0))
        for xi, (j, r) in zip(x + off, pv.iterrows()):
            if np.isfinite(r.nn_ratio):
                axC.text(xi, r.nn_ratio * 1.03, f"c{int(r.nn_cluster)}", ha="center", va="bottom",
                         fontsize=P2.fs(5.6), color=MUTED)
    axC.axhline(1.0, color=INK, ls=":", lw=P2.pw(1.0))
    axC.set_xticks(x)
    axC.set_xticklabels([f"c{j}" for j in order], fontsize=P2.fs(6.8))
    axC.set_ylabel("nearest other type, ratio to the cloud", fontsize=P2.fs(7.2))
    axC.tick_params(labelsize=P2.fs(6.4), colors=MUTED, length=2)
    axC.spines[["top", "right"]].set_visible(False)
    axC.set_title("C   how far each type is from its nearest neighbour"
                  + ("   (pale: argmax, solid: cut)" if len(vers) > 1 else ""),
                  fontsize=P2.fs(8.6), loc="left", color=INK, pad=P2.pw(5))
    # D: per type the nearest pair's map
    pv = pt[pt.version == v_main].set_index("cluster")
    for pos, j in enumerate(order):
        r, c = divmod(pos, ncol)
        sub = GridSpecFromSubplotSpec(1, 2, gs[1 + r, c * 6:(c + 1) * 6], width_ratios=[4.2, 1.0],
                                      wspace=0.08)
        ax, axb = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        nn = int(pv.loc[j].nn_cluster)
        key = (v_main, min(j, nn), max(j, nn))
        if nn < 0 or key not in res["maps"]:
            ax.axis("off"); axb.axis("off")
            ax.text(0.5, 0.5, f"c{j}: no pair with enough electrodes", ha="center", va="center",
                    fontsize=P2.fs(7), color=MUTED, transform=ax.transAxes)
            continue
        m = res["maps"][key]
        sgn = 1.0 if j == key[1] else -1.0           # draw as "this type minus the neighbour"
        mm = dict(m, d=m["d"] * sgn)
        map_panel(ax, axb, d, mm, f"{cname(d, j)}  minus  c{nn}", col=P2.cluster_col(j, K))
    bb = fig.add_subplot(gs[1, 0:6]).get_position()
    fig.text(0.045, bb.y1 + 0.42 / fig_h,
             f"D   per type, its nearest pair: standardised difference per bin (outlined: q < 0.05, "
             f"within-patient shuffle), band and condition shares of the linear MMD²"
             + (f"   ·   members at loading ≥ {CUT:.1f}" if v_main == "cut" else ""),
             fontsize=P2.fs(8.6), color=INK)
    fig.axes[-1].remove()
    od = P2.paper_dir(OUT)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"FIG6_mmd_cell_{d['fset']}_{d['method']}_K{K}{P2.run_tag(d)}.png"
    P2.stamp(fig, d, extra=f"MMD kernel bandwidth {res['sigma']:.3f}")
    P2.save_png(fig, p, dpi=P2.out_dpi(190), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_pairs(d, res, version, a):
    """Every pair's map, one grid."""
    K = d["k"]
    PAIRS = res["pairs"]
    ncol = 3
    nrow = int(np.ceil(len(PAIRS) / ncol))
    fig_h = 1.0 + 2.75 * nrow
    fig = plt.figure(figsize=(P2.canvas(17.6), fig_h), dpi=190)
    gs = GridSpec(nrow, ncol * 6, figure=fig, hspace=0.80, wspace=0.9, left=0.045, right=0.985,
                  top=1.0 - (0.35 if P2.paper() else 0.75) / fig_h, bottom=0.30 / fig_h)
    if not P2.paper():
        fig.suptitle(f"FIG 6 pairs   ·   {P2.METHOD_LABEL[d['method']]}   ·   {P2.FS_LABEL[d['fset']]}"
                     f"   ·   K = {K}   ·   members: {'loading ≥ ' + format(CUT, '.1f') if version == 'cut' else 'argmax'}"
                     "   ·   standardised difference per bin, outlined q < 0.05",
                     x=0.045, y=1.0 - 0.25 / fig_h, ha="left", fontsize=P2.fs(13.5), color=INK)
    for pos, (i, j) in enumerate(PAIRS):
        r, c = divmod(pos, ncol)
        sub = GridSpecFromSubplotSpec(1, 2, gs[r, c * 6:(c + 1) * 6], width_ratios=[4.2, 1.0], wspace=0.08)
        ax, axb = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        key = (version, i, j)
        if key not in res["maps"]:
            ax.axis("off"); axb.axis("off")
            ax.text(0.5, 0.5, f"c{i} vs c{j}: too few electrodes", ha="center", va="center",
                    fontsize=P2.fs(7), color=MUTED, transform=ax.transAxes)
            continue
        map_panel(ax, axb, d, res["maps"][key], pair_title(d, i, j, res["summary"], version))
    od = P2.paper_dir(OUT)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"FIG6_mmd_pairs_{d['fset']}_{d['method']}_K{K}_{version}{P2.run_tag(d)}.png"
    P2.stamp(fig, d)
    P2.save_png(fig, p, dpi=P2.out_dpi(170), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_main(d, res, comp, a):
    """The comparative figure for the main cell."""
    K, vers = d["k"], res["versions"]
    order = display_order(d)
    summ, pt = res["summary"], res["per_type"]
    v_main = "cut" if "cut" in vers else "argmax"
    fig_h = 14.6
    fig = plt.figure(figsize=(P2.canvas(17.6), fig_h), dpi=190)
    gs = GridSpec(4, 24, figure=fig, height_ratios=[1.0, 0.78, 0.95, 0.95], hspace=0.78, wspace=0.9,
                  left=0.045, right=0.985, top=1.0 - (0.35 if P2.paper() else 0.80) / fig_h,
                  bottom=0.35 / fig_h)
    if not P2.paper():
        fig.suptitle(f"FIG 6   ·   how distinct are the {K} types   ·   MMD between every pair, held-out "
                     f"patients, against the Gaussian cloud   ·   {P2.FS_LABEL[d['fset']]}, "
                     f"{P2.METHOD_LABEL[d['method']]}, K = {K}", x=0.045, y=1.0 - 0.25 / fig_h,
                     ha="left", fontsize=P2.fs(15.5), color=INK)
    # A: the two matrices
    axA1 = fig.add_subplot(gs[0, 0:6])
    matrix_panel(axA1, d, summ, "argmax", order, "A   held-out MMD² ÷ cloud   ·   argmax members")
    axA2 = fig.add_subplot(gs[0, 6:12])
    if "cut" in vers:
        im = matrix_panel(axA2, d, summ, "cut", order, f"      members at loading ≥ {CUT:.1f}")
    else:
        im = matrix_panel(axA2, d, summ, "argmax", order, "      in-sample z", what="z_in")
    cax = axA2.inset_axes([1.03, 0.1, 0.035, 0.8])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log2 (held-out MMD² ÷ cloud)", fontsize=P2.fs(6.2), color=MUTED)
    cb.ax.tick_params(labelsize=P2.fs(5.8), colors=MUTED, length=2)
    cb.outline.set_visible(False)
    # B and C: the per-type comparisons
    pv = {v: pt[pt.version == v].set_index("cluster") for v in vers}
    cols = [P2.cluster_col(j, K) for j in range(K)]

    def scatter(ax, yvals, ylabel, title, ref=None):
        x_main = pv[v_main].nn_ratio.reindex(range(K)).to_numpy()
        for j in range(K):
            if "argmax" in vers and v_main != "argmax":
                xa = pv["argmax"].nn_ratio.reindex(range(K)).to_numpy()[j]
                if np.isfinite(xa) and np.isfinite(yvals[j]):
                    ax.plot([xa, x_main[j]], [yvals[j], yvals[j]], "-", color=cols[j], lw=P2.pw(0.8), alpha=0.6)
                    ax.plot(xa, yvals[j], "o", ms=P2.pw(4.5), mfc="white", mec=cols[j], mew=P2.pw(1.0))
            if np.isfinite(x_main[j]) and np.isfinite(yvals[j]):
                ax.plot(x_main[j], yvals[j], "o", ms=P2.pw(5.5), color=cols[j])
                ax.annotate(f"c{j}", (x_main[j], yvals[j]), xytext=(4, 3), textcoords="offset points",
                            fontsize=P2.fs(6.2), color=cols[j])
        ok = np.isfinite(x_main) & np.isfinite(yvals)
        if ok.sum() >= 4:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(x_main[ok], yvals[ok])
            ax.text(0.98, 0.04, f"Spearman ρ {rho:+.2f}  (p {pval:.2f}, n {int(ok.sum())})",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=P2.fs(6.2), color=MUTED)
        if ref is not None:
            ax.axhspan(ref[0] - ref[1], ref[0] + ref[1], color=GREY, alpha=0.35, lw=0)
            ax.axhline(ref[0], color=MUTED, lw=P2.pw(0.8), ls=":")
        ax.axvline(1.0, color=INK, ls=":", lw=P2.pw(0.9))
        ax.set_xlabel("nearest other type: held-out MMD² ÷ cloud", fontsize=P2.fs(7.0))
        ax.set_ylabel(ylabel, fontsize=P2.fs(7.0))
        ax.tick_params(labelsize=P2.fs(6.2), colors=MUTED, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=P2.fs(8.6), loc="left", color=INK, pad=P2.pw(5))

    axB = fig.add_subplot(gs[0, 15:24])
    cons = comp.get("consensus")
    if cons is not None:
        scatter(axB, cons, "per-type consensus (patients resampled)",
                "B   separation vs stability" + ("   (hollow: argmax)" if len(vers) > 1 else ""))
    else:
        axB.axis("off")
        axB.text(0.5, 0.5, "no consensus sweep at this K", ha="center", va="center",
                 fontsize=P2.fs(7), color=MUTED, transform=axB.transAxes)
    # C: per type, its separation from the nearest other type, both versions
    axC = fig.add_subplot(gs[1, 0:11])
    x = np.arange(K)
    for vi, v in enumerate(vers):
        pvv = pv[v].reindex(order)
        off = (-0.18 if len(vers) > 1 else 0.0) + 0.36 * vi
        axC.bar(x + off, pvv.nn_ratio, width=0.34 if len(vers) > 1 else 0.6,
                color=[P2.cluster_col(j, K) for j in order], alpha=1.0 if v == v_main else 0.45,
                lw=P2.pw(0))
        for xi, (j, r) in zip(x + off, pvv.iterrows()):
            if np.isfinite(r.nn_ratio):
                axC.text(xi, r.nn_ratio * 1.03, f"c{int(r.nn_cluster)}", ha="center", va="bottom",
                         fontsize=P2.fs(5.6), color=MUTED)
    axC.axhline(1.0, color=INK, ls=":", lw=P2.pw(1.0))
    axC.set_xticks(x)
    axC.set_xticklabels([f"c{j}" for j in order], fontsize=P2.fs(6.8))
    for t, j in zip(axC.get_xticklabels(), order):
        t.set_color(P2.cluster_col(j, K))
    axC.set_ylabel("nearest other type: held-out MMD² ÷ cloud", fontsize=P2.fs(7.0))
    axC.tick_params(labelsize=P2.fs(6.2), colors=MUTED, length=2)
    axC.spines[["top", "right"]].set_visible(False)
    axC.set_title("C   each type's distance from its nearest neighbour"
                  + ("   (pale: argmax, solid: cut)" if len(vers) > 1 else ""),
                  fontsize=P2.fs(8.6), loc="left", color=INK, pad=P2.pw(5))
    axD1, axD2 = fig.add_subplot(gs[1, 13:18]), fig.add_subplot(gs[1, 19:24])
    f4 = comp.get("fig4")
    if f4 is not None:
        scatter(axD1, f4, "FIG 4: mean adjusted overlap", "D   vs cross-method matching")
    else:
        axD1.axis("off")
    sil = comp.get("silhouette")
    if sil is not None:
        scatter(axD2, sil[0], "per-type silhouette", "      vs silhouette  (grey: the cloud)", ref=(sil[1], sil[2]))
    else:
        axD2.axis("off")
    # E: the two least separated pairs, both versions
    sv = summ[summ.version == v_main].dropna(subset=["ratio"]).sort_values("ratio")
    weakest = [(int(r.i), int(r.j)) for _, r in sv.head(2).iterrows()]
    for row_, (i, j) in enumerate(weakest):
        for vi, v in enumerate(vers):
            sub = GridSpecFromSubplotSpec(1, 2, gs[2 + row_, vi * 12:(vi + 1) * 12], width_ratios=[4.2, 1.0],
                                          wspace=0.06)
            ax, axb = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            key = (v, i, j)
            if key not in res["maps"]:
                ax.axis("off"); axb.axis("off")
                continue
            lab_v = f"members at loading ≥ {CUT:.1f}" if v == "cut" else "argmax members"
            map_panel(ax, axb, d, res["maps"][key], pair_title(d, i, j, summ, v) + f"   ·   {lab_v}")
    bb = fig.add_subplot(gs[2, 0:6]).get_position()
    fig.text(0.045, bb.y1 + 0.42 / fig_h,
             "E   the two least separated pairs: standardised difference per bin, c_i minus c_j "
             "(outlined: q < 0.05, within-patient shuffle); bars: band share of the linear MMD²",
             fontsize=P2.fs(8.6), color=INK)
    fig.axes[-1].remove()
    od = P2.paper_dir(OUT)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"FIG6_mmd_main_{d['fset']}_{d['method']}_K{K}{P2.run_tag(d)}.png"
    P2.stamp(fig, d, extra=f"MMD kernel bandwidth {res['sigma']:.3f}")
    P2.save_png(fig, p, dpi=P2.out_dpi(190), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p, weakest


def fig_confusion(d, res, a):
    """Two K x K confusions: rows the published type, columns where its electrodes were
    placed by a model that never saw their patient; recall on the diagonal. Left: leave one
    patient out (the model is nearly the published one). Right: the patient halves."""
    if res.get("confusion_lopo") is None and res.get("confusion") is None:
        return None
    K = d["k"]
    order = display_order(d)
    panels = [(res.get("confusion_lopo"), "A   leave one patient out: refit on the other patients, place the held-out one"),
              (res.get("confusion"), f"B   patient halves: refit on half, place the other half ({a.n_splits} splits)")]
    fig, axes = plt.subplots(1, 2, figsize=(P2.canvas(15.0), 7.4), dpi=190)
    for ax, (conf, ttl) in zip(axes, panels):
        if conf is None:
            ax.axis("off")
            continue
        R = conf / np.maximum(conf.sum(1, keepdims=True), 1e-12)
        Ro = R[np.ix_(order, order)]
        acc = float(np.trace(conf) / max(conf.sum(), 1))
        ax.imshow(Ro, cmap="Greys", vmin=0, vmax=1, aspect="equal")
        for a_ in range(K):
            for b_ in range(K):
                v = Ro[a_, b_]
                ax.text(b_, a_, f"{100*v:.0f}", ha="center", va="center", fontsize=P2.fs(8.0),
                        color="white" if v > 0.55 else INK, fontweight="bold" if a_ == b_ else "normal")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([f"c{j}" for j in order], fontsize=P2.fs(8))
        ax.set_yticklabels([f"c{j}" for j in order], fontsize=P2.fs(8))
        for t, j in zip(ax.get_xticklabels(), order):
            t.set_color(P2.cluster_col(j, K))
        for t, j in zip(ax.get_yticklabels(), order):
            t.set_color(P2.cluster_col(j, K))
        ax.set_xlabel("placed in", fontsize=P2.fs(8.6))
        ax.set_ylabel("published type", fontsize=P2.fs(8.6))
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(ttl + chr(10) + f"% of each type's held-out electrodes   ·   overall {100*acc:.0f} %",
                     fontsize=P2.fs(8.8), loc="left", color=INK, pad=P2.pw(8))
    fig.suptitle(f"where a type's electrodes land when the model never saw their patient   ·   "
                 f"{P2.METHOD_LABEL[d['method']]}   ·   {P2.FS_LABEL[d['fset']]}   ·   K = {K}",
                 x=0.05, y=0.995, ha="left", fontsize=P2.fs(10.5), color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    od = P2.paper_dir(OUT)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"FIG6_mmd_heldout_confusion_{d['fset']}_{d['method']}_K{K}{P2.run_tag(d)}.png"
    P2.stamp(fig, d)
    P2.save_png(fig, p, dpi=P2.out_dpi(190), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_overview(cache, cells, a):
    """The twelve cells' matrices (argmax, held-out ratio) on one page."""
    fsets = [f for f in P2.FSETS if any(c[0] == f for c in cells)]
    fig_h = 1.2 + 3.9 * len(fsets)
    fig = plt.figure(figsize=(P2.canvas(17.6), fig_h), dpi=190)
    gs = GridSpec(len(fsets), 3, figure=fig, hspace=0.55, wspace=0.35, left=0.05, right=0.985,
                  top=1.0 - (0.35 if P2.paper() else 0.75) / fig_h, bottom=0.30 / fig_h)
    if not P2.paper():
        fig.suptitle("FIG 6 overview   ·   held-out MMD² ÷ Gaussian cloud, every cell, argmax members",
                     x=0.05, y=1.0 - 0.25 / fig_h, ha="left", fontsize=P2.fs(13.5), color=INK)
    im = None
    for r_, fset in enumerate(fsets):
        for c_, method in enumerate(METHODS):
            ax = fig.add_subplot(gs[r_, c_])
            cd = cell_dir(cache, fset, method)
            if not (cd / "manifest.json").exists():
                ax.axis("off")
                ax.text(0.5, 0.5, f"{FS_SHORT[fset]} × {P2.METHOD_LABEL[method]}: not run", ha="center",
                        va="center", fontsize=P2.fs(7), color=MUTED, transform=ax.transAxes)
                continue
            res = load_cell(cd)
            d = dict(k=res["manifest"]["k"], fset=fset, method=method)
            sv = res["summary"][res["summary"].version == "argmax"]
            med = float(np.nanmedian(sv.ratio))
            weakest = sv.dropna(subset=["ratio"]).sort_values("ratio").head(1)
            wk = (f"weakest c{int(weakest.i.iloc[0])} vs c{int(weakest.j.iloc[0])} "
                  f"({weakest.ratio.iloc[0]:.1f}×)") if len(weakest) else ""
            im = matrix_panel(ax, d, res["summary"], "argmax", list(range(d["k"])),
                              f"{FS_SHORT[fset]} × {P2.METHOD_LABEL[method]}, K {d['k']}   ·   "
                              f"median {med:.1f}×   ·   {wk}")
    if im is not None:
        cax = fig.add_axes([0.955, 0.40, 0.009, 0.25])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("log2 (held-out MMD² ÷ cloud)", fontsize=P2.fs(6.4), color=MUTED)
        cb.ax.tick_params(labelsize=P2.fs(6.0), colors=MUTED, length=2)
        cb.outline.set_visible(False)
    od = P2.paper_dir(OUT)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"FIG6_mmd_overview_{cache}.png"
    P2.stamp(fig)
    P2.save_png(fig, p, dpi=P2.out_dpi(170), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- captions
def caption_cell(d, res, a, comp=None, weakest=None):
    man_in = res.get("manifest", {})
    K, vers = d["k"], res["versions"]
    L = []
    A = L.append
    A(f"FIG 6   ·   MMD between the types   ·   {P2.METHOD_LABEL[d['method']]}   ·   {P2.FS_LABEL[d['fset']]}   ·   K = {K}")
    A("=" * 100)
    A("")
    A("WHAT IS MEASURED")
    A("-" * 100)
    A(f"Unbiased MMD^2 (Gaussian kernel, bandwidth = median pairwise distance over the cohort,")
    A(f"{res['sigma']:.4f}) between every pair of clusters, on the unit-norm rows the methods fit in.")
    A(f"In-sample: against {a.n_perm} within-patient shuffles of the two labels (BH over the pairs).")
    A(f"Held-out: {a.n_splits} random halves of the patients; the method refitted on half A, aligned to the")
    A(f"published clusters on half A (Hungarian), half B placed by the LOPO rule, MMD^2 on half B; the")
    A(f"Gaussian cloud (data covariance, no clusters) through the identical procedure, {a.n_cloud} draws.")
    A("The number in the matrices is the median held-out MMD^2 divided by the median cloud MMD^2. The cloud's")
    A("sectors are convex pieces of one featureless distribution, so they are well apart by construction:")
    A("a ratio below 1 means the two types overlap more than two such sectors, above 1 less. The dot marks")
    A("the within-patient shuffle on the held-out half: p < 0.05 in at least 90 % of the splits (z_ho is its")
    A("median z). The in-sample z and q are circular (the labels were fitted on these electrodes) and are")
    A("reported for completeness.")
    if "cut" in vers:
        A(f"Two versions: argmax members (every count and test of the paper) and members at loading >= {CUT:.1f}")
        A(f"(the drawn set, {100*(1-res.get('frac_cut', 0.0)):.1f} % of the electrodes). The cloud is cut by the same")
        A("FRACTION, not the same threshold: its loadings are flatter and a threshold would remove half of it.")
        A("The gap between the versions is the sensitivity of the separation to the cut.")
    A("The maps are the standardised mean difference in dB per bin, outlined where the within-patient")
    A("shuffle gives q < 0.05 over the bins; the bars are each band's share of the LINEAR-kernel MMD^2")
    A("(squared mean difference in the kernel space). The Gaussian-kernel MMD has no per-bin split.")
    A("")
    for v in vers:
        A(f"PAIRS   ·   {v}" + (f" (loading >= {CUT:.1f})" if v == "cut" else ""))
        A("-" * 100)
        A(f"  {'pair':<10}{'n_i':>6}{'n_j':>6}{'MMD2 in':>10}{'z_in':>8}{'q_in':>9}{'held-out':>10}{'z_ho':>8}{'p_ho<.05':>10}{'cloud':>9}{'ratio':>8}{'>97.5%':>8}")
        sv = res["summary"][res["summary"].version == v]
        for _, r in sv.sort_values("ratio").iterrows():
            A(f"  c{int(r.i)} vs c{int(r.j)}{int(r.n_i):>6}{int(r.n_j):>6}{r.mmd2_in:>10.4f}{r.z_in:>8.1f}{r.q_in:>9.4f}"
              f"{r.mmd2_ho:>10.4f}{r.z_ho_median:>8.1f}{r.share_p_ho_lt05:>10.2f}{r.cloud_median:>9.4f}{r.ratio:>8.2f}{r.share_above_cloud:>8.2f}")
        A("")
        A("  per type: nearest other type (held-out ratio)")
        pv = res["per_type"][res["per_type"].version == v]
        for _, r in pv.sort_values("nn_ratio").iterrows():
            A(f"    c{int(r.cluster):<3} n {int(r.n):<5} nearest c{int(r.nn_cluster)}  ratio {r.nn_ratio:.2f}  mean over pairs {r.mean_ratio:.2f}")
        A("")
    for tag_, key, what in (("LOPO", "_lopo", f"leave one patient out, {a.n_lopo_seeds} refit seed(s) per fold: the model is nearly the published one"),
                            ("HALVES", "", f"the MMD's patient halves, {a.n_splits} splits: refit on half a cohort, so refit instability is in it too")):
        if res.get(f"confusion_type{key}") is None:
            continue
        ct = res[f"confusion_type{key}"]
        conf = res[f"confusion{key}"]
        A(f"HELD-OUT PLACEMENT, {tag_}   ({what})")
        A("-" * 100)
        A(f"  overall: {100*np.trace(conf)/max(conf.sum(), 1):.1f} % of held-out electrodes land in their published type")
        A(f"  {'type':<6}{'placements':>12}{'recall':>9}{'median':>9}{'IQR':>16}{'precision':>11}   most confused with")
        for _, r in ct.sort_values("recall", ascending=False).iterrows():
            A(f"  c{int(r.cluster):<5}{int(r.n_placements):>12}{r.recall:>9.2f}{r.recall_median:>9.2f}"
              f"{r.recall_q25:>8.2f}-{r.recall_q75:<7.2f}{r.precision:>11.2f}   c{int(r.confused_with)} ({100*r.confused_share:.0f} %)")
        A("")
    if comp:
        A("COMPANIONS (main figure, per type)")
        A("-" * 100)
        for j in range(K):
            bits = [f"c{j}"]
            if comp.get("consensus") is not None:
                bits.append(f"consensus {comp['consensus'][j]:.2f}")
            if comp.get("fig4") is not None:
                bits.append(f"FIG 4 overlap {comp['fig4'][j]:.2f}")
            if comp.get("silhouette") is not None:
                bits.append(f"silhouette {comp['silhouette'][0][j]:+.3f}")
            A("  " + "   ".join(bits))
        if comp.get("silhouette") is not None:
            A(f"  cloud per-cluster silhouette {comp['silhouette'][1]:+.3f} +/- {comp['silhouette'][2]:.3f}")
        if weakest:
            A("  least separated pairs drawn: " + ", ".join(f"c{i} vs c{j}" for i, j in weakest))
        A("")
    A("PROVENANCE")
    A("-" * 100)
    A(f"  run        {d['run'].name}")
    A(f"  tables     {rel(cell_dir(a.cache, d['fset'], d['method']))}")
    A(f"  built by   00_paper2_figure6_mmd.py --feature-set {d['fset']} --method {d['method']}")
    A(f"  built on   {time.strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- selftest
def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n_pat, per, K = 16, 24, 3
    conds, bands, nt = ["audio", "picture", "reading"], ["1-20Hz", "20-70Hz", "70-170Hz", "170-270Hz", "270-400Hz"], 10
    F = len(conds) * len(bands) * nt
    n = n_pat * per
    lab = rng.integers(0, K, n)
    centres = rng.normal(0, 1.0, (K, F))
    X = centres[lab] + rng.normal(0, 1.2, (n, F))
    pat = np.repeat([f"P{i:02d}" for i in range(n_pat)], per)
    G = np.exp(rng.normal(0, 0.3, (n, K)))
    G[np.arange(n), lab] += 2.0
    Gn = G / G.sum(1, keepdims=True)
    meta = pd.DataFrame(dict(patient_id=pat, electrode=[f"e{i}" for i in range(n)]))
    return dict(run=Path("selftest_run"), k=K, X=X, G=G, Gn=Gn, lab=Gn.argmax(1), conds=conds, bands=bands,
                method="cnmf", hard=False, nt=nt, patient=pat, n_patients=n_pat, meta=meta, fset="concat_bands5")


def selftest(out: Path, a):
    global OUT, MMD_ROOT
    out.mkdir(parents=True, exist_ok=True)
    OUT = out
    MMD_ROOT = out / "mmd"
    P2.OUT = out
    a.n_perm, a.n_perm_heldout, a.n_splits, a.n_cloud, a.cache = 40, 20, 3, 2, "selftest"
    a.n_cloud_sil = 2
    a.n_lopo_seeds = 1
    d = synthetic()
    res = compute_cell(d, a)
    cd = cell_dir(a.cache, d["fset"], d["method"])
    save_cell(cd, d, res, a, {})
    back = load_cell(cd)
    assert np.allclose(back["summary"].mmd2_in, res["summary"].mmd2_in, equal_nan=True)
    sv = res["summary"][res["summary"].version == "argmax"]
    assert (sv.q_in < 0.05).all(), "three well separated synthetic clusters must all pass"
    assert (sv.ratio > 1.0).all(), "and be more separated than the cloud"
    assert back["confusion"] is not None and back["confusion"].shape == (3, 3)
    assert back["confusion_lopo"] is not None and back["confusion_lopo"].shape == (3, 3)
    assert (np.diag(back["confusion_lopo"]) / back["confusion_lopo"].sum(1)).min() > 0.5, "synthetic clusters must place well"
    assert fig_confusion(d, back, a) is not None
    p1 = fig_cell(d, back, a)
    p2 = fig_pairs(d, back, "argmax", a)
    pt0 = back["per_type"][back["per_type"].version == "argmax"].set_index("cluster")
    comp = dict(consensus=None, fig4=None,
                silhouette=(pt0.silhouette.reindex(range(3)).to_numpy(), float(pt0.sil_cloud_mean.iloc[0]),
                            float(pt0.sil_cloud_sd.iloc[0])))
    p3, weakest = fig_main(d, back, comp, a)
    P2.save_text(caption_cell(d, back, a, comp, weakest), p3.with_name(p3.stem + "_caption.txt"))
    # a hard method through the same path
    dh = dict(d, method="kmeans", hard=True, G=None)
    dh["lab"] = MS.fit_any(MS.unit(dh["X"]), 3, 0, "kmeans")
    dh["Gn"] = np.eye(3)[dh["lab"]]
    rh = compute_cell(dh, a)
    assert rh["versions"] == ["argmax"]
    save_cell(cell_dir(a.cache, dh["fset"], "kmeans"), dh, rh, a, {})
    fig_cell(dh, load_cell(cell_dir(a.cache, dh["fset"], "kmeans")), a)
    fig_overview(a.cache, [(d["fset"], "cnmf"), (d["fset"], "kmeans")], a)
    print("selftest ok:", sorted(str(p.relative_to(out)) for p in out.rglob("*.png")))
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    # the server's console is not UTF-8; no print carries essential non-ASCII, so
    # replace rather than crash
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feature-set", default=MAIN)
    ap.add_argument("--method", default="cnmf", choices=METHODS)
    ap.add_argument("--all", action="store_true", help="every feature set x method, then the overview")
    ap.add_argument("--grids", action="store_true", help="the pair grids for every cell drawn (default: the main cell)")
    ap.add_argument("--figures-only", action="store_true", help="redraw from the saved tables")
    ap.add_argument("--force", action="store_true", help="recompute a cell whose tables exist")
    ap.add_argument("--n-perm", type=int, default=2000, help="within-patient shuffles, in-sample")
    ap.add_argument("--n-perm-heldout", type=int, default=200, help="within-patient shuffles per held-out split")
    ap.add_argument("--n-splits", type=int, default=50, help="random patient halves")
    ap.add_argument("--n-cloud", type=int, default=50, help="Gaussian cloud draws")
    ap.add_argument("--n-cloud-sil", type=int, default=10,
                    help="cloud draws behind the per-type silhouette z")
    ap.add_argument("--n-lopo-seeds", type=int, default=3,
                    help="refit seeds per leave-one-patient-out fold (the LOPO confusion)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paper", action="store_true", help="the page versions (PAPER2_PAPER=1)")
    ap.add_argument("--selftest", metavar="DIR", help="synthetic data into DIR; nothing of the repo touched")
    a = ap.parse_args()
    if a.paper:
        os.environ["PAPER2_PAPER"] = "1"
    if a.selftest:
        return selftest(Path(a.selftest), a)
    import lf_runs as LR
    a.cache = LR.current_cache()
    cells = ([(f, m) for f in P2.FSETS for m in METHODS] if a.all else [(a.feature_set, a.method)])
    print(f"cache {a.cache}   cells {len(cells)}   perms {a.n_perm} / {a.n_perm_heldout}   "
          f"splits {a.n_splits}   cloud {a.n_cloud}" + ("   PAPER" if a.paper else ""))
    for fset, method in cells:
        t0 = time.time()
        K = K_BY_FSET[fset]
        d = P2.load_run(fset, K, method)
        d["fset"] = fset
        cd = cell_dir(a.cache, fset, method)
        print(f"\n===== {FS_SHORT[fset]} x {P2.METHOD_LABEL[method]}  K {K}  run {d['run'].name} =====", flush=True)
        before = input_hashes(d)
        if a.figures_only or ((cd / "manifest.json").exists() and not a.force):
            if not (cd / "manifest.json").exists():
                raise SystemExit(f"{cd} has no tables yet; run without --figures-only")
            res = load_cell(cd)
            print(f"  tables read from {rel(cd)}")
        else:
            res = compute_cell(d, a)
            save_cell(cd, d, res, a, before)
            print(f"  tables -> {rel(cd)}")
        res["manifest"] = json.loads((cd / "manifest.json").read_text(encoding="utf-8"))
        p = fig_cell(d, res, a)
        print(f"  {p.name}")
        pc = fig_confusion(d, res, a)
        if pc is not None:
            print(f"  {pc.name}")
        is_main = (fset == MAIN and method == "cnmf")
        comp, weakest = None, None
        if is_main:
            pt0 = res["per_type"][res["per_type"].version == res["versions"][0]].set_index("cluster")
            comp = dict(consensus=per_type_consensus(d, a.cache), fig4=per_type_fig4(d),
                        silhouette=(pt0.silhouette.reindex(range(K)).to_numpy(),
                                    float(pt0.sil_cloud_mean.iloc[0]), float(pt0.sil_cloud_sd.iloc[0])))
            pm, weakest = fig_main(d, res, comp, a)
            print(f"  {pm.name}")
            P2.save_text(caption_cell(d, res, a, comp, weakest), pm.with_name(pm.stem + "_caption.txt"))
        P2.save_text(caption_cell(d, res, a), p.with_name(p.stem + "_caption.txt"))
        if is_main or a.grids:
            for v in res["versions"]:
                print(f"  {fig_pairs(d, res, v, a).name}")
        after = input_hashes(d)
        if after != before:
            raise SystemExit(f"an input changed while running: {set(before) ^ set(after) or 'contents'}")
        print(f"  {time.time()-t0:.0f} s")
    if a.all:
        print(f"\n{fig_overview(a.cache, cells, a).name}")
    print("\nnothing published rewritten; figures -> paper_figures/ (FIG6_mmd_*)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
