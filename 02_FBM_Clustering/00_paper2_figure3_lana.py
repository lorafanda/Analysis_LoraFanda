#!/usr/bin/env python3
"""
00_paper2_figure3_lana.py - FIG 3, how each cluster sits in the LanA language atlas.

    python 00_paper2_figure3_lana.py                     K = 8, concat_hg
    python 00_paper2_figure3_lana.py --k 11 --feature-set concat_hg
    python 00_paper2_figure3_lana.py --feature-set concat_bands5 --n-perm 2000
    python 00_paper2_figure3_lana.py --all --k 8       all four feature sets, SHARED AXES

    A   clusters RANKED by how much LanA probability their electrodes sit in,
        most to least, against two nulls
    B   those correlations ranked, with confidence intervals and FDR
    C   per cluster, LOADING vs P_lana across every electrode - does belonging more
        strongly to this cluster mean sitting further into language cortex?
    D   what LanA looks like on this coverage - left, right and from above - so
        A to C are not read in a vacuum

Four panels, one per corner: A above B on the narrow left, C above D on the right.

WHAT P_lana IS. The LanA probabilistic language atlas (Lipkin et al.) sampled at each
contact's fsaverage position: the fraction of that atlas's subjects whose language
network covers this point. It is an ATLAS PRIOR about a location, not a measurement in
this patient - a high value does not mean this electrode responded to language, and a
low one does not mean it did not.

TWO THINGS THIS FIGURE REFUSES TO DO QUIETLY.

  Electrodes are not independent. Contacts on one shaft sit millimetres apart, share a
  patient and share an atlas neighbourhood, so a correlation across every electrode has
  nothing like that many degrees of freedom. Two nulls: a SHAFT-SHIFT that rolls each
  shaft's labels along itself, keeping the spatial smoothness that makes neighbouring
  contacts alike, and a within-patient shuffle of P_lana that does not. The first is
  the one that counts. Confidence intervals bootstrap PATIENTS, not electrodes, for the
  same reason.

  LanA coverage is not complete and not missing at random. The atlas run predates part
  of the cohort, so whole patients have no value at all. Per-cluster coverage is drawn
  in panel A and a cluster below MIN_COVERAGE is marked, because a mean over 60% of a
  cluster is not comparable with a mean over 95% of another.

NO CAPTION IS DRAWN ON THE FIGURE - it gets a sibling <name>_caption.txt, plus
_clusters.csv (per cluster) and _electrodes.csv (per electrode).
"""
from __future__ import annotations

import argparse
import glob
import json
import importlib.util
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import spearmanr, gaussian_kde

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                        # noqa: E402

_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)

CLUST, OUT = P2.CLUST, P2.OUT
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
RED, GREEN, BLUE = P2.RED, P2.GREEN, P2.BLUE
ATLAS = CLUST / "atlas"
MIN_COVERAGE = 0.70          # below this a cluster's mean is marked as thinly sampled
# How far two atlas tables may differ about the same contact before the figure refuses
# to pick one. lana_inout (2026-07-28) really does differ from the 2026-08-02 tables, by
# 4.99e-05 - CSV rounding. np.allclose's default tolerance near 0.5 is about 6e-6, so it
# was right to object and the threshold was simply tighter than the file format
# round-trips. 1e-3 is still an order of magnitude finer than any difference that could
# reorder a cluster, and the largest difference actually seen is printed either way.
LANA_TOL = 1e-3
N_PERM = 1000
N_BOOT = 1000
# the LanA network as a set: an electrode is 'inside' when its P_lana reaches the
# threshold. 0.1 is the partition the atlas runs on disk use (thr0p1); 0.2 is the
# core of the network. Both are drawn, so the choice is visible rather than hidden.
THRESHOLDS = (0.10, 0.20)
# THE 0.3 CUT (Lora, 2026-10-26), the one constant FIG 1's renders use: a point below it
# is hollow, at or above it filled with the cluster colour. Display only.
LOAD_CUT = P2.LOAD_CUT


def tkey(t):
    """column suffix for a threshold: 0.1 -> 't10', 0.2 -> 't20'"""
    return f"t{int(round(100 * t)):02d}"


# ---- the atlas ---------------------------------------------------------------
def lana_lookup():
    """contact key -> P_lana, from whichever atlas table covers the most contacts.

    The tables are checked against each other rather than trusted: if two ever disagree
    about a contact by more than LANA_TOL the figure stops rather than pick one
    silently. They are compared only where BOTH have a value, and with a tolerance -
    the tables are written on different dates and round to different precision in CSV,
    so exact equality is stricter than the format can deliver.
    """
    tabs = {}
    for f in glob.glob(str(ATLAS / "lana_*" / "runs" / "*" / "recon"
                           / "*with_fsaverage.csv")):
        p = Path(f)
        d = pd.read_csv(f, usecols=["patient_id", "electrode", "P_lana"])
        d["key"] = [f"{q}|{P2.norm(e)}" for q, e in zip(d.patient_id, d.electrode)]
        # parents[1] is the run, parents[3] the feature set. parents[2] is the
        # literal "runs", which made every lana_* collapse onto the same key.
        tabs[f"{p.parents[3].name}/{p.parents[1].name}"] = (
            d.drop_duplicates("key").set_index("key").P_lana)
    if not tabs:
        raise SystemExit(f"no LanA atlas tables under {ATLAS}")
    # TABLES ARE GROUPED BY THE DATE THEY WERE SAMPLED. The atlas is re-sampled whenever
    # the reconstruction changes (2026-10-12: 36 contacts moved, 12 renamed), and two
    # samplings disagree exactly for the contacts that moved - 0.51 between 20260907 and
    # 20261019 - so only the tables of ONE date can be checked against each other. The
    # newest date is the one on the current coordinates; older ones are named, not used.
    # Until 2026-10-21 the reference was the table with the most rows, which was the OLD
    # sampling (3380 rows against 2946), and every current table failed against it.
    stamp = lambda name: name.split("/")[1].split("_")[0]
    newest = max(stamp(n) for n in tabs)
    older = sorted({stamp(n) for n in tabs} - {newest})
    tabs = {n: v for n, v in tabs.items() if stamp(n) == newest}
    if older:
        print(f"    atlas sampling {newest} used; older samplings ignored: {', '.join(older)}")
    # the most USABLE values, not the most rows - a row whose P_lana is NaN covers
    # nothing
    best = max(tabs, key=lambda t: int(np.isfinite(tabs[t]).sum()))
    ref = tabs[best]
    disagree, cover, seen = [], [], []
    for name, v in tabs.items():
        if name == best:
            continue
        c = ref.index.intersection(v.index)
        if not len(c):
            continue
        a, b = ref[c].to_numpy(float), v[c].to_numpy(float)
        # a VALUE can only disagree where both tables have one. No table currently
        # carries a NaN, but a contact the atlas cannot sample would produce one, and
        # NaN != NaN would then read as a disagreement about a number when it is a
        # difference in coverage.
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            md = float(np.abs(a[both] - b[both]).max())
            seen.append((name, md))
            if md > LANA_TOL:
                disagree.append((name, md))
        n_only = int(np.sum(np.isfinite(a) != np.isfinite(b)))
        if n_only:
            cover.append((name, n_only))
    if disagree:
        raise SystemExit("LanA tables disagree about the VALUE of P_lana, so there is "
                         "no single value to use: "
                         + ", ".join(f"{n} (max diff {d:.4g})" for n, d in disagree))
    worst = max((m for _, m in seen), default=0.0)
    print(f"    {len(tabs)} atlas tables agree on P_lana to {worst:.2g} "
          f"(tolerance {LANA_TOL:g})")
    if cover:
        print("    they differ only in COVERAGE, not in value: "
              + ", ".join(f"{n} ({c} keys)" for n, c in cover))
    return ref, best, sorted(tabs), worst


def load(fset, k):
    """The run, its loadings, and P_lana joined on. Reports what did not join."""
    d = P2.load_run(fset, k)
    d["fset"] = fset
    look, src, all_src, tol_seen = lana_lookup()
    d["shaft"] = shaft_of(d["meta"].get("contact_norm", d["meta"]["electrode"]),
                          d["patient"])
    keys = [f"{q}|{P2.norm(e)}"
            for q, e in zip(d["meta"]["patient_id"], d["meta"]["electrode"])]
    P = pd.Series(keys).map(look).to_numpy(float)
    d.update(P_lana=P, has=np.isfinite(P), lana_src=src, lana_all_src=all_src,
             lana_tol_seen=tol_seen)
    lost = pd.Series(d["patient"][~d["has"]]).value_counts()
    d["lana_missing_by_patient"] = lost
    print(f"  LanA from {src}: {int(d['has'].sum())} of {len(P)} electrodes "
          f"({100*d['has'].mean():.1f}%)")
    if len(lost):
        whole = [p for p, n in lost.items()
                 if n == int((d["patient"] == p).sum())]
        print(f"    {len(lost)} patients partly missing; {len(whole)} entirely: "
              f"{', '.join(map(str, whole[:6]))}")
    return d


# ---- the statistics ----------------------------------------------------------
def within_patient_perm(P, patient, rng):
    """P_lana shuffled WITHIN each patient.

    Keeps every patient's own set of atlas values and its coverage exactly, and breaks
    only the link between a contact's position and its cluster. A null that shuffled
    across patients would also break the fact that patients differ in where they were
    implanted, and would be far too easy to beat.
    """
    out = P.copy()
    for p in np.unique(patient):
        m = np.where(patient == p)[0]
        out[m] = P[rng.permutation(m)]
    return out


def shaft_of(contact_norm, patient):
    """patient + shaft, from the contact name with its trailing number removed.

    AHL10 -> AHL. A shaft is the physical electrode: its contacts are millimetres
    apart, so they share an atlas neighbourhood and are the unit spatial correlation
    lives on.
    """
    import re
    sh = [re.sub(r"\d+$", "", str(c)) for c in contact_norm]
    return np.array([f"{p}|{x}" for p, x in zip(patient, sh)], dtype=object)


def shaft_roll_index(shaft_id, rng):
    """An index that rolls each shaft's contacts by a random offset, in place.

    Applied to the cluster labels AND the loadings, with P_lana left where it is, this
    is a spatially-constrained null: the run structure along each shaft survives intact
    and only its alignment with the atlas is broken. One index drives both statistics,
    so panel A and panel C are testing the same permutation rather than two.
    """
    idx = np.arange(len(shaft_id))
    for sh in np.unique(shaft_id):
        m = np.where(shaft_id == sh)[0]
        if len(m) > 1:
            idx[m] = m[np.roll(np.arange(len(m)), int(rng.integers(1, len(m))))]
    return idx


def bh_fdr(p):
    """Benjamini-Hochberg. K clusters is K tests and the figure ranks them, which is
    exactly the situation that manufactures a best-looking one."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    v = p[ok]
    if not len(v):
        return q
    o = np.argsort(v)
    n = len(v)
    adj = v[o] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def analyse(d, n_perm=N_PERM, n_boot=N_BOOT, seed=0):
    """Per cluster: how much LanA it sits in, and whether its loading tracks LanA."""
    rng = np.random.default_rng(seed)
    K, lab, Gn = d["k"], d["lab"], d["Gn"]
    P, has, pat = d["P_lana"], d["has"], d["patient"]
    Ph, path_, labh = P[has], pat[has], lab[has]
    Gh = Gn[has]
    pats = np.unique(path_)
    shaft = d["shaft"][has]
    sizes_sh = pd.Series(shaft).value_counts()
    # a shaft that is one cluster throughout cannot be broken by a shift
    single = int(sum(1 for sh in np.unique(shaft)
                     if len(np.unique(labh[shaft == sh])) == 1))
    d["n_shafts"], d["n_shafts_single"] = int(len(sizes_sh)), single

    rows = []
    for j in range(K):
        sel = labh == j
        n_all = int((lab == j).sum())
        n_l = int(sel.sum())
        cov = n_l / max(n_all, 1)
        mean = float(Ph[sel].mean()) if n_l else np.nan
        med = float(np.median(Ph[sel])) if n_l else np.nan
        r, _ = spearmanr(Gh[:, j], Ph) if n_l else (np.nan, np.nan)
        row = dict(cluster=j, n=n_all, n_lana=n_l, coverage=cov,
                   mean_P=mean, median_P=med, rho=float(r))
        for t in THRESHOLDS:
            row[f"frac_{tkey(t)}"] = float((Ph[sel] >= t).mean()) if n_l else np.nan
        rows.append(row)
    R = pd.DataFrame(rows)

    # TWO NULLS. The shaft-shift one is spatially constrained and leads; the
    # within-patient one is kept beside it so the reader can see how much the answer
    # depends on which is asked.
    for tag, spatial in (("", False), ("_sh", True)):
        nm = np.zeros((n_perm, K))
        nr = np.zeros((n_perm, K))
        nf = {th: np.zeros((n_perm, K)) for th in THRESHOLDS}
        for t in range(n_perm):
            if spatial:
                idx = shaft_roll_index(shaft, rng)
                Lp, Gp, Pp = labh[idx], Gh[idx], Ph
            else:
                Lp, Gp, Pp = labh, Gh, within_patient_perm(Ph, path_, rng)
            for j in range(K):
                sel = Lp == j
                nm[t, j] = Pp[sel].mean() if sel.any() else np.nan
                nr[t, j] = spearmanr(Gp[:, j], Pp)[0]
                for th in THRESHOLDS:
                    nf[th][t, j] = (Pp[sel] >= th).mean() if sel.any() else np.nan
        R[f"null_mean_P{tag}"] = np.nanmean(nm, 0)
        R[f"p_mean{tag}"] = [(np.sum(nm[:, j] >= R.mean_P[j]) + 1) / (n_perm + 1)
                             for j in range(K)]
        R[f"null_rho{tag}"] = np.nanmean(nr, 0)
        R[f"p_rho{tag}"] = [(np.sum(np.abs(nr[:, j]) >= abs(R.rho[j])) + 1)
                            / (n_perm + 1) for j in range(K)]
        R[f"q_mean{tag}"], R[f"q_rho{tag}"] = (bh_fdr(R[f"p_mean{tag}"]),
                                               bh_fdr(R[f"p_rho{tag}"]))
        for th in THRESHOLDS:
            c = f"frac_{tkey(th)}"
            R[f"null_{c}{tag}"] = np.nanmean(nf[th], 0)
            R[f"p_{c}{tag}"] = [(np.sum(nf[th][:, j] >= R[c][j]) + 1) / (n_perm + 1)
                                for j in range(K)]
            R[f"q_{c}{tag}"] = bh_fdr(R[f"p_{c}{tag}"])

    # PATIENTS are the resampling unit, not electrodes: contacts on one shaft are not
    # independent draws and an electrode bootstrap would give intervals several times
    # too narrow
    bm = np.full((n_boot, K), np.nan)
    br = np.full((n_boot, K), np.nan)
    bf = {th: np.full((n_boot, K), np.nan) for th in THRESHOLDS}
    for t in range(n_boot):
        take = rng.choice(pats, len(pats), replace=True)
        idx = np.concatenate([np.where(path_ == p)[0] for p in take])
        for j in range(K):
            sel = labh[idx] == j
            if sel.sum() > 2:
                bm[t, j] = Ph[idx][sel].mean()
                br[t, j] = spearmanr(Gh[idx][:, j], Ph[idx])[0]
                for th in THRESHOLDS:
                    bf[th][t, j] = (Ph[idx][sel] >= th).mean()
    for nm_, arr in (("mean_P", bm), ("rho", br),
                     *[(f"frac_{tkey(th)}", bf[th]) for th in THRESHOLDS]):
        R[f"{nm_}_lo"] = np.nanpercentile(arr, 2.5, axis=0)
        R[f"{nm_}_hi"] = np.nanpercentile(arr, 97.5, axis=0)
    R["baseline_P"] = float(Ph.mean())
    for th in THRESHOLDS:
        R[f"base_{tkey(th)}"] = float((Ph >= th).mean())   # the cohort rate
    return R.sort_values("mean_P", ascending=False).reset_index(drop=True), Ph, Gh, labh


# ---- panels ------------------------------------------------------------------
def own_loading(d):
    """Each electrode's normalised loading on the cluster it is assigned to: its share
    of the K loadings (rows of Gn sum to 1), 1/K at the least and 1 at the most."""
    return d["Gn"][np.arange(len(d["lab"])), d["lab"]]


def panel_rank(ax, R, d, chars, ext):
    """A: one violin per cluster of P(LanA) over its electrodes, every electrode a point,
    clusters ranked by the mean. A point below LOAD_CUT is an outline only, at or above
    it filled with the cluster colour - the cut every figure applies. The mean, its
    patient bootstrap, the two nulls and the stars are the same statistics the bars
    carried until 2026-10-26."""
    K = d["k"]
    P, has, lab = d["P_lana"], d["has"], d["lab"]
    own = own_loading(d)
    rng = np.random.default_rng(0)                  # the jitter, reproducible
    x = np.arange(len(R))
    top = 0.0
    for i, r in R.iterrows():
        j = int(r.cluster)
        col = P2.cluster_col(j, K)
        m = has & (lab == j)
        v, w = P[m], own[m]
        if len(v) < 2:
            continue
        top = max(top, float(v.max()))
        vp = ax.violinplot([v], positions=[i], widths=0.80, showextrema=False,
                           showmeans=False, showmedians=False)
        for body in vp["bodies"]:
            body.set_facecolor(col); body.set_alpha(0.13)
            body.set_edgecolor(col); body.set_linewidth(0.7); body.set_zorder(1)
        # the points sit inside the body: the jitter is scaled by the same density the
        # body is drawn from, so a point never lands outside the violin
        dens = gaussian_kde(v)(v)
        xs = i + rng.uniform(-1.0, 1.0, len(v)) * (0.37 * dens / dens.max())
        hol = w < LOAD_CUT
        ax.scatter(xs[hol], v[hol], s=9.0, facecolors="none", edgecolors=[col],
                   linewidths=0.55, zorder=3)
        ax.scatter(xs[~hol], v[~hol], s=7.5, color=col, lw=0, zorder=3)
        # the statistics, as before: the mean with its patient bootstrap, the two nulls
        ax.plot([i - 0.27, i + 0.27], [r.mean_P, r.mean_P], color=INK, lw=1.7, zorder=6)
        ax.plot([i, i], [r.mean_P_lo, r.mean_P_hi], color=INK, lw=1.0, zorder=6)
        ax.plot(i, r.null_mean_P_sh, "_", ms=15, color=INK, mew=2.0, zorder=5)
        ax.plot(i, r.null_mean_P, "_", ms=11, color=MUTED, mew=1.4, zorder=4)
        star = "*" if r.q_mean_sh < 0.05 else ("(*)" if r.q_mean < 0.05 else "")
        ax.text(i, float(v.max()) + 0.025, f"{star}\n{100*r.coverage:.0f}%", ha="center",
                va="bottom", fontsize=6.6,
                color=RED if r.coverage < MIN_COVERAGE else MUTED)
    ax.axhline(R.baseline_P.iloc[0], color=MUTED, ls=":", lw=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{int(c)}\nn={int(n)}" for c, n in zip(R.cluster, R.n_lana)],
                       fontsize=7.2)
    ax.set_ylabel("P(LanA) of each electrode", fontsize=8.6)
    ax.set_xlim(-0.6, len(R) - 0.4)
    ax.set_ylim(0, top + 0.17)                        # the stars sit above the tallest violin
    ax.tick_params(labelsize=7.4, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    # the key to the points, top right, where no violin reaches
    ax.scatter([0.70], [0.965], s=14, facecolors="none", edgecolors=[INK],
               linewidths=0.6, transform=ax.transAxes, clip_on=False)
    ax.scatter([0.70], [0.915], s=12, color=INK, lw=0, transform=ax.transAxes,
               clip_on=False)
    ax.text(0.73, 0.965, f"loading < {LOAD_CUT:.1f}", transform=ax.transAxes,
            va="center", fontsize=6.4, color=MUTED)
    ax.text(0.73, 0.915, f"loading \u2265 {LOAD_CUT:.1f}", transform=ax.transAxes,
            va="center", fontsize=6.4, color=MUTED)
    thin = [f"c{int(r.cluster)}" for _, r in R.iterrows() if r.coverage < MIN_COVERAGE]
    n_h = int((own[has] < LOAD_CUT).sum())
    ax.text(0, -0.20, _wrap(
        f"One violin per cluster of P(LanA) over its electrodes, ranked by the mean, "
        f"every electrode a point. Points with a normalised loading below {LOAD_CUT:.1f} "
        f"on their cluster are hollow, from {LOAD_CUT:.1f} filled: {n_h} of "
        f"{int(has.sum())} electrodes ({100*n_h/max(int(has.sum()), 1):.0f}%) are hollow. "
        f"Short dark line = the mean, whiskers a 95% interval from bootstrapping PATIENTS; "
        f"dotted = the cohort mean ({R.baseline_P.iloc[0]:.3f}). Two nulls on the mean: "
        f"the dark dash rolls each electrode's cluster along its own SHAFT, keeping the "
        f"smoothness that makes neighbours alike (*); the pale dash shuffles within "
        f"patient, which destroys it and is too easy to beat ((*) = passes only that "
        f"one). Benjamini-Hochberg over {K}; % over each violin = coverage"
        + (f"; {', '.join(thin)} fall{'' if len(thin) > 1 else 's'} below "
           f"{100*MIN_COVERAGE:.0f}% and {'their means are' if len(thin) > 1 else 'its mean is'} "
           f"not comparable with the rest." if thin else "."),
        chars), transform=ax.transAxes, va="top", fontsize=7.4, color=MUTED,
        linespacing=1.5)


def inside_summary(R, Ph, labh, t):
    """The sentence behind panels B and C at one threshold: the two clusters that keep
    the most of their electrodes inside the network and the two that keep the least,
    and the types most enriched and depleted inside it."""
    c = f"frac_{tkey(t)}"
    base = float(R[c.replace("frac", "base")].iloc[0])
    o = R.sort_values(c, ascending=False)
    hi = [(int(r.cluster), float(r[c])) for _, r in o.head(2).iterrows()]
    lo = [(int(r.cluster), float(r[c])) for _, r in o.tail(2).iterrows()]
    m = Ph >= t
    mix_in = pd.Series(labh[m]).value_counts(normalize=True)
    mix_out = pd.Series(labh[~m]).value_counts(normalize=True)
    ratio = {int(j): (float(mix_in.get(j, 0.0)), float(mix_out.get(j, 0.0)))
             for j in R.cluster}
    enr = sorted(ratio.items(), key=lambda kv: kv[1][0] - kv[1][1], reverse=True)
    return dict(base=base, hi=hi, lo=lo, n_in=int(m.sum()), n_out=int((~m).sum()),
                most=enr[:2], least=enr[-2:])


def panel_inside(ax, R, d, chars, ext):
    """Of each cluster's electrodes, the share inside the LanA network at two thresholds.

    The mean of panel A is set by the mass of electrodes below P = 0.1, where nothing
    happens; the relation lives in the tail of the P distribution, and a threshold
    reads it directly - how much of this type sits in language cortex - against the
    cohort rate, with the same two nulls and the same patient bootstrap as panel A.
    """
    from matplotlib.patches import Patch
    K = d["k"]
    x = np.arange(len(R))
    nT = len(THRESHOLDS)
    w = 0.80 / nT
    cols = [P2.cluster_col(int(c), K) for c in R.cluster]
    for i_t, t in enumerate(THRESHOLDS):
        c = f"frac_{tkey(t)}"
        xs = x - 0.40 + w * (i_t + 0.5)
        alpha = 1.0 if i_t == 0 else 0.45
        ax.bar(xs, R[c], width=w * 0.90, color=cols, alpha=alpha, lw=0, zorder=2)
        ax.errorbar(xs, R[c], yerr=[R[c] - R[f"{c}_lo"], R[f"{c}_hi"] - R[c]],
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.0, zorder=3)
        ax.plot(xs, R[f"null_{c}_sh"], "_", ms=9, color=INK, mew=1.8, zorder=5)
        ax.plot(xs, R[f"null_{c}"], "_", ms=7, color=MUTED, mew=1.2, zorder=4)
        base = float(R[f"base_{tkey(t)}"].iloc[0])
        ax.axhline(base, color=MUTED, ls=":" if i_t == 0 else (0, (1, 2.2)), lw=1.1,
                   zorder=1)
        for i, (_, r) in enumerate(R.iterrows()):
            star = ("*" if r[f"q_{c}_sh"] < 0.05 else
                    "(*)" if r[f"q_{c}"] < 0.05 else "")
            top = max(r[f"{c}_hi"], r[c])
            lab = (f"{star}\n{r[c] / base:.1f}×" if i_t == 0 else star)
            ax.text(xs[i], top + 0.012, lab, ha="center", va="bottom", fontsize=6.4,
                    color=INK if i_t == 0 else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{int(c)}\nn={int(n)}" for c, n in zip(R.cluster, R.n_lana)],
                       fontsize=7.2)
    ax.set_ylabel("share of its electrodes inside LanA, P(LanA) ≥ t", fontsize=8.6)
    ax.set_ylim(0, max(0.5, ext.get("B_top", 0.0) * 1.32))
    ax.tick_params(labelsize=7.4, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend([Patch(color=INK, alpha=1.0), Patch(color=INK, alpha=0.45)],
              [f"P ≥ {t:.1f}   (cohort {100*float(R[f'base_{tkey(t)}'].iloc[0]):.0f}%, dotted)"
               for t in THRESHOLDS], fontsize=6.8,
              frameon=False, loc="upper right", ncol=2, handlelength=1.2)
    sm = inside_summary(R, d["P_lana"][d["has"]], d["lab"][d["has"]], THRESHOLDS[0])
    ax.text(0, -0.20, _wrap(
        f"Of each cluster's electrodes, the share whose atlas probability reaches "
        f"{THRESHOLDS[0]:.1f} (solid) or {THRESHOLDS[1]:.1f} (pale), in panel A's order; "
        f"dotted lines are the cohort rate, whiskers a 95% interval from bootstrapping "
        f"PATIENTS, dashes the two nulls of panel A (dark = shaft shift, the one that "
        f"counts; pale = within patient); * passes the shaft-shift null after "
        f"Benjamini-Hochberg over {K}, (*) only the weaker one. The number over a solid "
        f"bar is its enrichment, the share divided by the cohort rate. At P ≥ "
        f"{THRESHOLDS[0]:.1f}: "
        + " and ".join(f"c{j} {100*f:.0f}%" for j, f in sm["hi"])
        + f" against {100*sm['base']:.0f}% in the cohort; "
        + " and ".join(f"c{j} {100*f:.0f}%" for j, f in sm["lo"]) + ".", chars),
        transform=ax.transAxes, va="top", fontsize=7.4, color=MUTED, linespacing=1.5)


def panel_mix(ax, R, d, labh, Ph, chars):
    """What the network is made of: the type composition of the electrodes inside the
    LanA network at each threshold, of the whole cohort, and of the electrodes outside.
    Descriptive; the per-cluster test is panel B."""
    K = d["k"]
    order = [int(c) for c in R.cluster]          # panel A's order, left to right
    groups = []
    for t in THRESHOLDS[::-1]:
        m = Ph >= t
        groups.append((f"inside, P ≥ {t:.1f}   n={int(m.sum())}", labh[m]))
    groups.append((f"whole cohort   n={len(labh)}", labh))
    m = Ph < THRESHOLDS[0]
    groups.append((f"outside, P < {THRESHOLDS[0]:.1f}   n={int(m.sum())}", labh[m]))
    y = np.arange(len(groups))[::-1]
    for yi, (name, L) in zip(y, groups):
        left = 0.0
        for j in order:
            f = float((L == j).mean()) if len(L) else 0.0
            ax.barh(yi, f, left=left, color=P2.cluster_col(j, K), height=0.64, lw=0)
            if f >= 0.07:
                ax.text(left + f / 2, yi, f"c{j}\n{100*f:.0f}%", ha="center",
                        va="center", fontsize=6.2, color="white")
            left += f
    ax.set_yticks(y)
    ax.set_yticklabels([g[0] for g in groups], fontsize=7.2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_xlabel("share of the electrodes in that group", fontsize=8.6)
    ax.tick_params(colors=MUTED, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    sm = inside_summary(R, Ph, labh, THRESHOLDS[0])
    ax.text(0, -0.26, _wrap(
        f"The same electrodes the other way round: which types the language network "
        f"is made of. Each bar is one group of electrodes and its segments are the "
        f"clusters, in panel A's order, with their share of that group. Inside the "
        f"network at P ≥ {THRESHOLDS[0]:.1f} the most over-represented types are "
        + " and ".join(f"c{j} ({100*a:.0f}% against {100*b:.0f}% outside)"
                       for j, (a, b) in sm["most"])
        + "; the most under-represented "
        + " and ".join(f"c{j} ({100*a:.0f}% against {100*b:.0f}%)"
                       for j, (a, b) in sm["least"])
        + ". A bar that looked like the cohort bar would mean the network has no "
        f"type of its own.", chars),
        transform=ax.transAxes, va="top", fontsize=7.4, color=MUTED, linespacing=1.5)


def _wrap(t, chars):
    import textwrap
    return textwrap.fill(t, chars)


def panel_brain(axL, axR, axT, axKey, d, chars):
    """What LanA looks like on THIS coverage - the context A to C are read against.

    Left, right, and from above, on FIG 1's own brain scenes; the fourth cell, which
    the top view leaves free, carries the colour scale and what the colour means.
    """
    P, has = d["P_lana"], d["has"]
    vmax = float(np.nanpercentile(P[has], 98)) or 1.0
    cm = plt.get_cmap("magma")
    for ax, side, tag in ((axL, "L", "left"), (axR, "R", "right"),
                          (axT, "T", "from above")):
        pl, ok = P2._scene(side, d, 1.30)
        m = ok & has
        actor = None
        if m.sum():
            t = np.clip(P[m] / max(vmax, 1e-9), 0, 1)
            rgba = np.empty((int(m.sum()), 4), np.uint8)
            rgba[:, :3] = np.clip(255 * cm(t)[:, :3], 0, 255).astype(np.uint8)
            rgba[:, 3] = 255
            cloud = P2.pv.PolyData(d["xyz"][m])
            cloud["rgba"] = rgba
            cloud["r"] = np.full(int(m.sum()), P2.BG_RADIUS * 1.25, float)
            g = cloud.glyph(orient=False, scale="r",
                            geom=P2.pv.Sphere(radius=1.0, theta_resolution=12,
                                              phi_resolution=12))
            actor = pl.add_mesh(g, scalars="rgba", rgba=True)
        img = pl.screenshot(return_img=True, transparent_background=True)
        if actor is not None:
            pl.remove_actor(actor)
        ax.imshow(P2._crop_alpha(np.asarray(img)))
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        # the top view fills its axes to the bottom edge, so its count goes to the
        # top-left corner, which the brain's outline leaves free (2026-10-26)
        y_, va_ = (0.98, "top") if side == "T" else (0.02, "bottom")
        ax.text(0.02, y_, f"{tag}  {int(m.sum())}", transform=ax.transAxes,
                ha="left", va=va_, fontsize=7.0, color=MUTED)
    # the scale and the reading, in the cell the top view leaves free
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    axKey.axis("off")
    cax = axKey.inset_axes([0.04, 0.82, 0.62, 0.055])
    cb = axKey.figure.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=cm),
                               cax=cax, orientation="horizontal")
    cb.set_label(f"P(LanA), 0 to the 98th percentile ({vmax:.2f})", fontsize=7.2,
                 color=MUTED, labelpad=3)
    cb.ax.tick_params(labelsize=6.4, colors=MUTED, length=2)
    cb.outline.set_visible(False)
    axKey.text(0.04, 0.60, _wrap(
        f"Every electrode with a LanA value, coloured by it - dark is low, bright is "
        f"high. An ATLAS PRIOR about a location, not a measurement in this patient: "
        f"a bright contact is somewhere language cortex usually is, which is not the "
        f"same as having responded to language. LanA is left-lateralised and this "
        f"coverage is not symmetric; the view from above shows both at once.", chars),
        transform=axKey.transAxes, va="top", ha="left", fontsize=7.4, color=MUTED,
        linespacing=1.5)


def reading_notes(R, d):
    """The paragraphs that used to sit on the figure as a fourth-row panel.

    Generated from the data, so the caption cannot describe a figure that has since
    changed. They go to the caption now because the figure is four panels, one per
    corner, and has no room for a fifth.
    """
    K = d["k"]
    thin = [f"c{int(r.cluster)}" for _, r in R.iterrows() if r.coverage < MIN_COVERAGE]
    return [
        "P(LanA) is an ATLAS PRIOR about a location - the fraction of LanA's subjects "
        "whose language network covers this point. It is not a measurement in this "
        "patient. A cluster high in panel A sits where language cortex usually is; "
        "whether it RESPONDED to language is FIG 1, not this.",
        f"Against the SHAFT-SHIFT null, which preserves the spatial smoothness of the "
        f"atlas, {int((R.q_mean_sh < 0.05).sum())} of {K} clusters sit in more LanA "
        f"than chance (panel A) and {int((R[f'q_frac_{tkey(THRESHOLDS[0])}_sh'] < 0.05).sum())} "
        f"of {K} keep more of their electrodes inside the network at P >= "
        f"{THRESHOLDS[0]:.1f} (panel B). Against the weaker within-patient null it is "
        f"{int((R.q_mean < 0.05).sum())} and "
        f"{int((R[f'q_frac_{tkey(THRESHOLDS[0])}'] < 0.05).sum())}. Where those "
        f"differ, the smoothness was doing the work.",
        (lambda sm: f"Inside the network at P >= {THRESHOLDS[0]:.1f}: "
         + " and ".join(f"c{j} {100*f:.0f}%" for j, f in sm["hi"])
         + f" of their electrodes against {100*sm['base']:.0f}% in the cohort, "
         + " and ".join(f"c{j} {100*f:.0f}%" for j, f in sm["lo"])
         + ". The network itself is "
         + " and ".join(f"{100*a:.0f}% c{j} against {100*b:.0f}% outside"
                        for j, (a, b) in sm["most"])
         + " (panel C). The largest |rho| of loading against P_lana, in the table "
         f"and no longer drawn, is {R.rho.abs().max():.2f}.")
        (inside_summary(R, d["P_lana"][d["has"]], d["lab"][d["has"]], THRESHOLDS[0])),
        (f"Only {100*d['has'].mean():.0f}% of electrodes have a LanA value, and the "
         f"missing ones are WHOLE PATIENTS rather than a random scatter - the atlas run "
         f"predates part of the cohort. " if not bool(d["has"].all()) else
         "Every electrode has a LanA value, so coverage is not a caveat here. ")
        + (f"Coverage is below {100*MIN_COVERAGE:.0f}% for {', '.join(thin)}, whose "
           f"{'means are' if len(thin) > 1 else 'mean is'} not comparable with the rest."
           if thin else "Every cluster is above the coverage floor here."),
        "The shares of panel B are one number per cluster and are not compositional, "
        "unlike a correlation of normalised loadings; the composition bars of panel C "
        "sum to 100% within each group.",
        "LanA is left-lateralised and sEEG coverage is not symmetric, so a left-heavy "
        "cluster scores higher for that reason alone. Neither null removes it: it is a "
        "property of where a cluster is, not of which patients it draws on.",
    ]


# ---- shared axes ---------------------------------------------------------------
EXT_KEYS = ("A_top", "B_top", "B_abs", "C_x", "C_y")   # B_abs, C_x, C_y: the panels drawn until 2026-10-26


def extents_of(R, Gh, Ph):
    """What this feature set's panels reach: A's top (mean or whisker), B's largest
    |rho| including its interval, C's largest loading and largest P(LanA)."""
    fr = [f"frac_{tkey(t)}" for t in THRESHOLDS]
    return dict(A_top=float(max(R.mean_P_hi.max(), R.mean_P.max())),
                B_top=float(max(R[f"{c}_hi"].max() for c in fr)),
                B_abs=float(np.nanmax(np.abs([R.rho_lo.min(), R.rho_hi.max()]))),
                C_x=float(np.nanmax(Gh)), C_y=float(np.nanmax(Ph)))


def merge_extents(exts):
    return {key: max(float(e.get(key, 0.0)) for e in exts.values()) for key in EXT_KEYS}


def extents_path(k):
    return OUT / f"FIG3_extents_K{k:02d}.json"


def resolve_extents(fset, k, own):
    """The axes this figure draws on: the maximum over every feature set that has
    recorded its extents at this K, plus this one. Records this one for the others."""
    p = extents_path(k)
    exts = {}
    if p.exists():
        try:
            exts = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            exts = {}
    exts[fset] = own
    P2.save_text(json.dumps(exts, indent=1) + "\n", p)
    return merge_extents(exts), sorted(exts)


def prepare(fset, k, n_perm, n_boot):
    d = load(fset, k)
    if d["has"].sum() < 50:
        raise SystemExit("too few electrodes carry a LanA value to say anything")
    print(f"  {n_perm} permutations of each null, {n_boot} patient bootstraps ...")
    R, Ph, Gh, labh = analyse(d, n_perm, n_boot)
    return d, R, Ph, Gh, labh


def figure_3_all(fsets, k, n_perm, n_boot):
    """All feature sets in one process: analyse each, take the global extents, then
    render each on them. The only way to get four figures on one set of axes without
    running everything twice."""
    pre = {}
    for f in fsets:
        print(f"\n=== {f} ===")
        pre[f] = prepare(f, k, n_perm, n_boot)
    exts = {f: extents_of(R, Gh, Ph) for f, (d, R, Ph, Gh, labh) in pre.items()}
    P2.save_text(json.dumps(exts, indent=1) + "\n", extents_path(k))
    ext = merge_extents(exts)
    print("\n  shared axes: " + "  ".join(f"{q} {v:.3f}" for q, v in ext.items()))
    out = []
    for f in fsets:
        print(f"\n=== {f}: render ===")
        out.append(figure_3(f, k, n_perm, n_boot, ext=ext, pre=pre[f],
                            shared_with=list(fsets)))
    return out


# ---- the figure --------------------------------------------------------------
def figure_3(fset, k, n_perm, n_boot, ext=None, pre=None, shared_with=None):
    t0 = time.time()
    for _pl, _ in P2._PLOTTER.values():
        _pl.close()
    P2._PLOTTER.clear()
    d, R, Ph, Gh, labh = pre if pre is not None else prepare(fset, k, n_perm, n_boot)
    own = extents_of(R, Gh, Ph)
    if ext is None:
        ext, shared_with = resolve_extents(fset, k, own)
    print("  axes shared with: " + ", ".join(shared_with))

    # FOUR PANELS, ONE PER CORNER. The left column is the narrow one (A above B) and
    # the right the wide one (C above D); the bottom row is the taller, so that D -
    # three views of the coverage - can actually be read.
    fig = plt.figure(figsize=(17.6, 15.6), dpi=190)
    gs = GridSpec(2, 24, figure=fig, height_ratios=[1.0, 2.3], hspace=0.52,
                  wspace=1.2, left=0.055, right=0.985, top=0.905, bottom=0.030)
    fig.suptitle(f"FIG 3   ·   LanA language atlas   ·   {fset}   ·   K = {k}   ·   "
                 f"{int(d['has'].sum())} of {len(d['X'])} electrodes have a LanA value",
                 x=0.055, y=0.978, ha="left", fontsize=15.5, color=INK)
    fig.text(0.055, 0.955,
             "How far into the LanA probabilistic language atlas each cluster sits, how "
             "much of each type lies inside the network, and what the network is made "
             "of.  Two nulls per test: a SHAFT-SHIFT that keeps spatial smoothness (*) "
             "and a within-patient shuffle ((*)); every interval bootstraps patients.",
             fontsize=9.8, color=MUTED, va="top")

    axA = fig.add_subplot(gs[0, 0:7])
    panel_rank(axA, R, d, 66, ext)

    # B beside A, on the same cluster order; the scatter grid of loading against
    # P_lana that stood here until 2026-10-26 is in the table, not on the page
    axB = fig.add_subplot(gs[0, 8:24])
    panel_inside(axB, R, d, 150, ext)

    cC = GridSpecFromSubplotSpec(2, 1, gs[1, 0:7], height_ratios=[0.90, 1.0],
                                 hspace=0.0)
    axC = fig.add_subplot(cC[0])
    panel_mix(axC, R, d, labh, Ph, 66)

    cD = GridSpecFromSubplotSpec(2, 2, gs[1, 8:24], hspace=0.08, wspace=0.03)
    axDL, axDR = fig.add_subplot(cD[0, 0]), fig.add_subplot(cD[0, 1])
    axDT, axDK = fig.add_subplot(cD[1, 0]), fig.add_subplot(cD[1, 1])
    panel_brain(axDL, axDR, axDT, axDK, d, 70)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    # one line of labels per row, above the taller of its two panels
    for row_ in (((axA, "A  ·  P(LanA) of every electrode, clusters ranked by the mean"),
                  (axB, f"B  ·  inside the LanA network at P ≥ {THRESHOLDS[0]:.1f} "
                        f"and ≥ {THRESHOLDS[1]:.1f}")),
                 ((axC, "C  ·  what the network is made of"),
                  (axDL, "D  ·  LanA on this coverage: left, right, from above"))):
        top = max(inv.transform((0, ax.get_tightbbox(r).y1))[1] for ax, _ in row_)
        for ax, label in row_:
            fig.text(ax.get_position().x0, top + 0.006, label, fontsize=10.4,
                     color=INK, va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    _rt = "_run" + str(d["run"].name).replace("_", "-")
    p = OUT / f"FIG3_lana_{fset}_K{k}{_rt}.png"
    P2.stamp(fig, d)
    P2.save_png(fig, p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P2.save_text(R.to_csv(index=False), p.with_name(p.stem + "_clusters.csv"))
    P2.save_text(pd.DataFrame(dict(
        patient=d["patient"], electrode=d["meta"]["electrode"], cluster=d["lab"],
        loading=own_loading(d), P_lana=d["P_lana"], has_lana=d["has"])).to_csv(index=False),
        p.with_name(p.stem + "_electrodes.csv"))
    caption_3(p.with_name(p.stem + "_caption.txt"), d, R, k, fset, n_perm, n_boot,
              ext, shared_with)
    print(f"  {time.time()-t0:.0f}s  -> {p.name}")
    return p


def caption_3(path, d, R, k, fset, n_perm, n_boot, ext, shared_with):
    own = own_loading(d)
    L = []
    A = L.append
    A(f"FIG 3   ·   LanA language atlas   ·   {fset}   ·   K = {k}")
    A("=" * 100)
    A("")
    A("AXES SHARED ACROSS FEATURE SETS")
    A("-" * 100)
    A("B's y axis runs to the same limit in every FIG 3 at this K, so the figures can be")
    A("laid side by side and a bar that looks taller is taller; the limit is the maximum")
    A(f"reached in: {', '.join(shared_with)}. A's y axis spans the full range of P_lana,")
    A("which is the same set of electrodes in every feature set.")
    A(f"  B top {ext['B_top']:.4f}   (drawn with a small margin)")
    A(f"  recorded in {extents_path(k).name}; --all rebuilds every feature set on one set")
    A("")
    A("WHAT P_lana IS")
    A("-" * 100)
    A("The LanA probabilistic language atlas sampled at each contact's fsaverage")
    A("position: the fraction of that atlas's subjects whose language network covers")
    A("this point. It is an ATLAS PRIOR ABOUT A LOCATION, not a measurement in this")
    A("patient. A high value means the contact sits where language cortex usually is;")
    A("it does not mean the contact responded to language, and a low value does not")
    A("mean it did not. Nothing in this figure is evidence that a cluster IS or IS NOT")
    A("language-responsive - only that it does or does not sit where LanA is.")
    A("")
    A("COVERAGE, AND WHY IT IS NOT A DETAIL")
    A("-" * 100)
    A(f"  electrodes with a LanA value   {int(d['has'].sum())} of {len(d['X'])} "
      f"({100*d['has'].mean():.1f}%)")
    A(f"  atlas table used               {d['lana_src']}")
    A(f"  tables checked against it      {', '.join(d['lana_all_src'])}")
    A(f"  largest disagreement between them   {d['lana_tol_seen']:.2g} "
      f"(tolerance {LANA_TOL:g})")
    A("")
    A("The tables are written on different dates and round to different precision in")
    A("CSV, so they are compared with a tolerance rather than for exact equality, and")
    A("only where BOTH have a value - a contact the atlas could not sample is a")
    A("coverage difference, not a disagreement about a number.")
    A("")
    if bool(d["has"].all()):
        A("Every electrode has a LanA value, so coverage is not a caveat on this figure.")
        A("The safeguards stay for a cohort whose atlas run lags its reconstruction:")
        A("per-cluster coverage is printed over every violin in panel A and anything below")
        A(f"{100*MIN_COVERAGE:.0f}% is marked in red.")
    else:
        A("The missing electrodes are NOT missing at random: the atlas run predates part of")
        A("the cohort, so whole patients have no value at all. A cluster drawn mostly from")
        A("those patients would look like it sits outside language cortex when in truth it")
        A("has not been measured against the atlas. Per-cluster coverage is printed under")
        A(f"every violin in panel A and anything below {100*MIN_COVERAGE:.0f}% is marked in red.")
    if len(d["lana_missing_by_patient"]):
        A("")
        A("  electrodes with no LanA value, by patient:")
        for p_, n_ in d["lana_missing_by_patient"].items():
            tot = int((d["patient"] == p_).sum())
            A(f"    {p_:<12} {int(n_):>4} of {tot:<4}"
              + ("   (the whole patient)" if int(n_) == tot else ""))
    A("")
    A("THE NULLS, AND WHY THERE ARE TWO")
    A("-" * 100)
    A("SPATIAL AUTOCORRELATION IS THE WHOLE PROBLEM. A contact and its neighbour 3 mm")
    A("along the same shaft have nearly identical P_lana, because a brain map is smooth;")
    A("that smoothness inflates any correlation computed against it. The field's answer")
    A("is a spatially-constrained null - Alexander-Bloch's spin test for dense surface")
    A("maps, variogram-matched surrogates for sparse points.")
    A("")
    A("The sEEG analogue used here is a CIRCULAR SHIFT ALONG EACH SHAFT: every")
    A("electrode's cluster and loading are rolled by a random offset within its own")
    A("shaft while P_lana stays put. Each shaft's run structure survives exactly - the")
    A("same labels in the same runs, in the same order - and only its alignment with the")
    A("atlas is broken. This is the null that counts, marked * on the figure.")
    A("")
    A(f"  shafts                 {d['n_shafts']}")
    A(f"  invariant to a shift   {d['n_shafts_single']} (one cluster throughout, so a")
    A("                         shift cannot change them; they carry no information")
    A("                         about alignment and contribute no randomness)")
    A("")
    A("THE SECOND NULL, KEPT FOR COMPARISON")
    A("-" * 100)
    A("Contacts on one shaft sit millimetres apart, share a patient, and share an atlas")
    A(f"neighbourhood, so a correlation across {int(d['has'].sum())} electrodes has nothing like that many")
    A("degrees of freedom. The weaker null permutes P_lana WITHIN EACH PATIENT: that")
    A("keeps each patient's own atlas values and coverage exactly as they are and")
    A("breaks only the link between a contact's position and its cluster. A null that")
    A("shuffled across patients would also destroy the fact that patients are implanted")
    A("in different places, and would be far too easy to beat.")
    A("")
    A("Confidence intervals BOOTSTRAP PATIENTS, not electrodes, for the same reason: an")
    A("electrode bootstrap treats contacts on one shaft as independent draws and returns")
    A("intervals several times too narrow.")
    A("")
    A(f"  {n_perm} permutations of EACH null, {n_boot} bootstraps, patients resampled")
    A("  with replacement")
    A("")
    A("Both are reported on the figure and in the table below. Where they disagree, the")
    A("spatial smoothness was doing the work and the weaker null was crediting the")
    A("clustering for it - which is exactly what a within-patient shuffle cannot see.")
    A("")
    A("MULTIPLE COMPARISONS")
    A("-" * 100)
    A(f"K clusters is {k} tests per panel, and the figure RANKS them - which is exactly")
    A("the arrangement that manufactures an impressive-looking best one. Both panels are")
    A("corrected with Benjamini-Hochberg and a star means q < 0.05, not p < 0.05.")
    A("")
    A("PANEL A - P(LanA) of every electrode, clusters ranked by the mean")
    A("-" * 100)
    A("One violin per cluster over its electrodes' P_lana, every electrode a point, ranked")
    A("by the mean (the short dark line, with its patient-bootstrap interval). A point is")
    A(f"hollow below a normalised loading of {LOAD_CUT:.2f} on its cluster (its share of the K")
    A(f"loadings) and filled with the cluster colour from {LOAD_CUT:.2f}: {int((own[d['has']] < LOAD_CUT).sum())} "
      f"of {int(d['has'].sum())} electrodes ({100*float((own[d['has']] < LOAD_CUT).mean()):.0f}%) are hollow -")
    A("the cut FIG 1's renders apply too (grey below, full colour above). Display only:")
    A("every count and test uses the argmax membership.")
    A("Dark dash = the shaft-shift null, pale dash = the within-patient null, dotted line =")
    A("the cohort mean, % over each violin = coverage.")
    A(f"  own-cluster loading   min {own.min():.3f}   median {np.median(own):.3f}   "
      f"max {own.max():.3f}   (1/K = {1/k:.3f})")
    A("")
    A(f"  cohort mean P_lana             {R.baseline_P.iloc[0]:.4f}")
    A(f"  most  c{int(R.cluster.iloc[0])}  {R.mean_P.iloc[0]:.4f} "
      f"[{R.mean_P_lo.iloc[0]:.4f}, {R.mean_P_hi.iloc[0]:.4f}]  "
      f"shaft null {R.null_mean_P_sh.iloc[0]:.4f} q={R.q_mean_sh.iloc[0]:.3g}  "
      f"(within-patient null {R.null_mean_P.iloc[0]:.4f} q={R.q_mean.iloc[0]:.3g})")
    A(f"  least c{int(R.cluster.iloc[-1])}  {R.mean_P.iloc[-1]:.4f} "
      f"[{R.mean_P_lo.iloc[-1]:.4f}, {R.mean_P_hi.iloc[-1]:.4f}]  "
      f"shaft null {R.null_mean_P_sh.iloc[-1]:.4f} q={R.q_mean_sh.iloc[-1]:.3g}")
    A(f"  clusters above the SHAFT-SHIFT null at q<0.05   "
      f"{int((R.q_mean_sh < 0.05).sum())} of {k}")
    A(f"  above the weaker within-patient null              "
      f"{int((R.q_mean < 0.05).sum())} of {k}")
    A("")
    A("PANEL B - inside the LanA network, at two thresholds")
    A("-" * 100)
    A("The mean of panel A is set by the mass of electrodes below P = 0.1, where nothing")
    A("happens; the relation lives in the tail of the P distribution. Panel B reads it")
    A(f"directly: of each cluster's electrodes, the share whose P_lana reaches {THRESHOLDS[0]:.1f}")
    A(f"(solid) or {THRESHOLDS[1]:.1f} (pale), against the cohort rate (dotted), with the same two")
    A("nulls and the same patient bootstrap as panel A. The number over a solid bar is")
    A("the enrichment, the share divided by the cohort rate. The thresholds are a")
    A("declared choice: 0.1 is the partition the atlas runs on disk use (thr0p1), 0.2 the")
    A("core of the network; both are drawn so the choice is visible.")
    A("")
    for th in THRESHOLDS:
        c = f"frac_{tkey(th)}"
        base = float(R[f"base_{tkey(th)}"].iloc[0])
        A(f"  P_lana >= {th:.2f}   cohort rate {100*base:.1f}%   "
          f"({int((R[f'q_{c}_sh'] < 0.05).sum())}"
          f" of {k} above the shaft-shift null at q<0.05, "
          f"{int((R[f'q_{c}'] < 0.05).sum())} above the within-patient one)")
        A("  cluster   n     inside  [95% CI]           x cohort   shaft null  q        within null  q")
        for _, r in R.iterrows():
            A(f"  c{int(r.cluster):<7} {int(r.n_lana):<5} {100*r[c]:5.1f}%  "
              f"[{100*r[f'{c}_lo']:5.1f}, {100*r[f'{c}_hi']:5.1f}]   {r[c]/base:5.2f}   "
              f"{100*r[f'null_{c}_sh']:6.1f}%  {r[f'q_{c}_sh']:<7.3g}  "
              f"{100*r[f'null_{c}']:6.1f}%  {r[f'q_{c}']:<7.3g}")
        A("")
    A("PANEL C - what the network is made of")
    A("-" * 100)
    A("The same electrodes the other way round: the type composition of the electrodes")
    A("inside the network at each threshold, of the whole cohort, and of the electrodes")
    A("outside it, each bar summing to 100%. Descriptive - the per-cluster test is B.")
    A("")
    Ph_, labh_ = d["P_lana"][d["has"]], d["lab"][d["has"]]
    order_ = [int(c_) for c_ in R.cluster]
    groups_ = [(f"inside P >= {th:.1f}", Ph_ >= th) for th in THRESHOLDS[::-1]]
    groups_ += [("whole cohort", np.ones(len(Ph_), bool)),
                (f"outside P < {THRESHOLDS[0]:.1f}", Ph_ < THRESHOLDS[0])]
    A("  group               n     " + "  ".join(f"c{j:<4}" for j in order_))
    for name_, m_ in groups_:
        L_ = labh_[m_]
        A(f"  {name_:<18} {len(L_):<5} "
          + "  ".join(f"{100*float((L_ == j).mean()):4.0f}%" for j in order_))
    sm = inside_summary(R, Ph_, labh_, THRESHOLDS[0])
    A("")
    A(f"  most over-represented inside (P >= {THRESHOLDS[0]:.1f}): "
      + ", ".join(f"c{j} {100*a:.0f}% vs {100*b:.0f}% outside" for j, (a, b) in sm["most"]))
    A(f"  most under-represented: "
      + ", ".join(f"c{j} {100*a:.0f}% vs {100*b:.0f}%" for j, (a, b) in sm["least"]))
    A("")
    A("THE CORRELATION OF LOADING WITH P_lana (Spearman rho over every electrode, with")
    A("both nulls and the patient bootstrap) is still computed and written to the table")
    A("- rho, rho_lo, rho_hi, null_rho, p_rho, q_rho and the _sh variants - but it left")
    A("the page on 2026-10-26: the K rhos are compositional (an electrode's loadings sum")
    A("to 1, so one cluster tracking the atlas forces the others negative) and they are")
    A("dominated by the mass of electrodes below P = 0.1. For the record:")
    A("")
    A("  cluster  rho [95% CI]                 q shaft   q within-pat")
    for _, r in R.iterrows():
        A(f"  c{int(r.cluster):<7} {r.rho:+.3f} [{r.rho_lo:+.3f}, {r.rho_hi:+.3f}]"
          f"   {r.q_rho_sh:>8.3g}   {r.q_rho:>10.3g}")
    A(f"  largest |rho| {R.rho.abs().max():.3f}; {int((R.q_rho_sh < 0.05).sum())} of {k} pass the shaft-shift null.")
    A("")
    A("")
    A("PANEL D - LanA on this coverage")
    A("-" * 100)
    A("Every electrode with a LanA value, coloured by it, on the left hemisphere, the")
    A("right, and both from above (anterior up, left on the left). The scale runs 0 to")
    A("the 98th percentile of the values present, so one bright outlier cannot flatten")
    A("the rest.")
    A("")
    A("HOW TO READ THIS FIGURE")
    A("-" * 100)
    for para in reading_notes(R, d):
        A(_wrap(para, 100))
        A("")
    A("WHAT THIS FIGURE DOES NOT SHOW")
    A("-" * 100)
    A("  - Not evidence of language responsiveness. LanA is a prior about where language")
    A("    cortex usually sits, sampled at a coordinate. The response itself is FIG 1.")
    A("  - Not corrected for the fact that LanA is left-lateralised and sEEG coverage is")
    A("    not symmetric. A cluster that happens to be left-heavy will score higher for")
    A("    that reason alone; the within-patient null does not remove it, because it is")
    A("    a property of where a cluster is, not of which patients it draws on.")
    A("  - Not a statement about K. The clusters are whatever this K produced.")
    n_missing = int((~d["has"]).sum())
    if n_missing:
        A(f"  - The {n_missing} electrodes with no LanA value are excluded from every number here,")
        A("    not counted as zero.")
    else:
        A("  - Every electrode has a LanA value; nothing is excluded.")
    P2.save_text("\n".join(L) + "\n", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="concat_hg")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--all", action="store_true",
                    help="every feature set, on one shared set of axes")
    a = ap.parse_args()
    print(f"=== FIGURE 3 ===  {a.feature_set}  K={a.k}")
    (figure_3_all(P2.FSETS, a.k, a.n_perm, a.n_boot) if a.all
     else figure_3(a.feature_set, a.k, a.n_perm, a.n_boot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
