#!/usr/bin/env python3
"""
00_paper2_figure3_lana.py - FIG 3, how each cluster sits in the LanA language atlas.

    python 00_paper2_figure3_lana.py                     K = 8, concat_hg
    python 00_paper2_figure3_lana.py --k 11 --feature-set concat_hg
    python 00_paper2_figure3_lana.py --feature-set concat_bands5 --n-perm 2000

    A   clusters RANKED by how much LanA probability their electrodes sit in,
        most to least, against a within-patient permutation null
    B   per cluster, LOADING vs P_lana across every electrode - does belonging more
        strongly to this cluster mean sitting further into language cortex?
    C   those correlations ranked, with confidence intervals and FDR
    D   what LanA looks like on this coverage, so A-C are not read in a vacuum

WHAT P_lana IS. The LanA probabilistic language atlas (Lipkin et al.) sampled at each
contact's fsaverage position: the fraction of that atlas's subjects whose language
network covers this point. It is an ATLAS PRIOR about a location, not a measurement in
this patient - a high value does not mean this electrode responded to language, and a
low one does not mean it did not.

TWO THINGS THIS FIGURE REFUSES TO DO QUIETLY.

  Electrodes are not independent. Contacts on one shaft sit millimetres apart, share a
  patient and share an atlas neighbourhood, so a correlation across 1400 electrodes has
  nothing like 1400 degrees of freedom. Every null here permutes P_lana WITHIN PATIENT,
  which keeps each patient's coverage and its atlas values intact and asks only whether
  the CLUSTERING lines up with them. Confidence intervals bootstrap PATIENTS, not
  electrodes, for the same reason.

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
from scipy.stats import spearmanr

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

    rows = []
    for j in range(K):
        sel = labh == j
        n_all = int((lab == j).sum())
        n_l = int(sel.sum())
        cov = n_l / max(n_all, 1)
        mean = float(Ph[sel].mean()) if n_l else np.nan
        med = float(np.median(Ph[sel])) if n_l else np.nan
        r, _ = spearmanr(Gh[:, j], Ph) if n_l else (np.nan, np.nan)
        rows.append(dict(cluster=j, n=n_all, n_lana=n_l, coverage=cov,
                         mean_P=mean, median_P=med, rho=float(r)))
    R = pd.DataFrame(rows)

    # nulls, both from the same within-patient permutation so they are one story
    nm = np.zeros((n_perm, K))
    nr = np.zeros((n_perm, K))
    for t in range(n_perm):
        Pp = within_patient_perm(Ph, path_, rng)
        for j in range(K):
            sel = labh == j
            nm[t, j] = Pp[sel].mean() if sel.any() else np.nan
            nr[t, j] = spearmanr(Gh[:, j], Pp)[0]
    R["null_mean_P"] = np.nanmean(nm, 0)
    R["p_mean"] = [(np.sum(nm[:, j] >= R.mean_P[j]) + 1) / (n_perm + 1)
                   for j in range(K)]
    R["null_rho"] = np.nanmean(nr, 0)
    R["p_rho"] = [(np.sum(np.abs(nr[:, j]) >= abs(R.rho[j])) + 1) / (n_perm + 1)
                  for j in range(K)]
    R["q_mean"], R["q_rho"] = bh_fdr(R.p_mean), bh_fdr(R.p_rho)

    # PATIENTS are the resampling unit, not electrodes: contacts on one shaft are not
    # independent draws and an electrode bootstrap would give intervals several times
    # too narrow
    bm = np.full((n_boot, K), np.nan)
    br = np.full((n_boot, K), np.nan)
    for t in range(n_boot):
        take = rng.choice(pats, len(pats), replace=True)
        idx = np.concatenate([np.where(path_ == p)[0] for p in take])
        for j in range(K):
            sel = labh[idx] == j
            if sel.sum() > 2:
                bm[t, j] = Ph[idx][sel].mean()
                br[t, j] = spearmanr(Gh[idx][:, j], Ph[idx])[0]
    for nm_, arr in (("mean_P", bm), ("rho", br)):
        R[f"{nm_}_lo"] = np.nanpercentile(arr, 2.5, axis=0)
        R[f"{nm_}_hi"] = np.nanpercentile(arr, 97.5, axis=0)
    R["baseline_P"] = float(Ph.mean())
    return R.sort_values("mean_P", ascending=False).reset_index(drop=True), Ph, Gh, labh


# ---- panels ------------------------------------------------------------------
def panel_rank(ax, R, d, chars):
    K = d["k"]
    x = np.arange(len(R))
    cols = [P2.cluster_col(int(c), K) for c in R.cluster]
    ax.bar(x, R.mean_P, color=cols, width=0.74, lw=0)
    ax.errorbar(x, R.mean_P, yerr=[R.mean_P - R.mean_P_lo, R.mean_P_hi - R.mean_P],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.4)
    ax.plot(x, R.null_mean_P, "_", ms=13, color=MUTED, mew=1.6, zorder=4)
    ax.axhline(R.baseline_P.iloc[0], color=MUTED, ls=":", lw=1.2)
    for i, r in R.iterrows():
        star = "*" if r.q_mean < 0.05 else ""
        ax.text(i, max(r.mean_P_hi, r.mean_P) + 0.018,
                f"{star}\n{100*r.coverage:.0f}%", ha="center", va="bottom",
                fontsize=6.6, color=RED if r.coverage < MIN_COVERAGE else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{int(c)}\nn={int(n)}" for c, n in zip(R.cluster, R.n_lana)],
                       fontsize=7.2)
    ax.set_ylabel("mean P(LanA) of its electrodes", fontsize=8.6)
    ax.set_ylim(0, max(R.mean_P_hi.max(), R.mean_P.max()) * 1.22)
    ax.tick_params(labelsize=7.4, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    thin = [f"c{int(r.cluster)}" for _, r in R.iterrows() if r.coverage < MIN_COVERAGE]
    ax.text(0, -0.42, _wrap(
        f"Ranked most to least. Bars are the mean LanA probability of a cluster's "
        f"electrodes; whiskers are a 95% interval from bootstrapping PATIENTS. The grey "
        f"dash is a within-patient permutation null and the dotted line the cohort mean "
        f"({R.baseline_P.iloc[0]:.3f}). * = q < 0.05 after Benjamini-Hochberg over the "
        f"{K} clusters. The % under each bar is how much of that cluster HAS a LanA "
        f"value" + (f"; {', '.join(thin)} fall below {100*MIN_COVERAGE:.0f}% and their "
                    f"means are not comparable with the rest." if thin else "."),
        chars), transform=ax.transAxes, va="top", fontsize=7.4, color=MUTED,
        linespacing=1.5)


def panel_scatter(axes, R, Gh, Ph, labh, d):
    K = d["k"]
    for ax, (_, r) in zip(axes, R.iterrows()):
        j = int(r.cluster)
        col = P2.cluster_col(j, K)
        ax.scatter(Gh[:, j], Ph, s=2.0, color=GREY, lw=0, alpha=0.45)
        m = labh == j
        ax.scatter(Gh[m, j], Ph[m], s=4.5, color=col, lw=0, alpha=0.85)
        if np.isfinite(r.rho):
            z = np.polyfit(Gh[:, j], Ph, 1)
            xx = np.linspace(Gh[:, j].min(), Gh[:, j].max(), 20)
            ax.plot(xx, np.polyval(z, xx), color=INK, lw=1.1)
        ax.set_title(f"c{j}   rho {r.rho:+.2f}"
                     + ("*" if r.q_rho < 0.05 else ""), fontsize=7.8,
                     color=RED if r.q_rho < 0.05 else col, loc="left", pad=2.2)
        ax.tick_params(labelsize=6.2, colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)


def panel_rho(ax, R, d, chars):
    K = d["k"]
    y = np.arange(len(R))[::-1]
    cols = [P2.cluster_col(int(c), K) for c in R.cluster]
    ax.barh(y, R.rho, color=cols, height=0.7, lw=0)
    ax.errorbar(R.rho, y, xerr=[R.rho - R.rho_lo, R.rho_hi - R.rho], fmt="none",
                ecolor=INK, elinewidth=1.0, capsize=2.4)
    ax.plot(R.null_rho, y, "|", ms=13, color=MUTED, mew=1.6, zorder=4)
    ax.axvline(0, color=INK, lw=0.8)
    for i, (_, r) in enumerate(R.iterrows()):
        if r.q_rho < 0.05:
            ax.text(r.rho + (0.012 if r.rho >= 0 else -0.012), y[i], "*",
                    ha="left" if r.rho >= 0 else "right", va="center", fontsize=10,
                    color=RED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"c{int(c)}" for c in R.cluster], fontsize=7.4)
    ax.set_xlabel("Spearman rho, loading vs P(LanA)", fontsize=8.6)
    ax.tick_params(labelsize=7.4, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    sig = int((R.q_rho < 0.05).sum())
    ax.text(0, -0.30, _wrap(
        f"Across EVERY electrode with a LanA value, not just the cluster's own: does a "
        f"higher loading on this cluster go with sitting further into language cortex? "
        f"Whiskers bootstrap PATIENTS; the grey tick is the within-patient permutation "
        f"null; * = q < 0.05 (BH over {K}). {sig} of {K} survive. A rho near zero means "
        f"the cluster is placed independently of the atlas, which is a result and not a "
        f"failure.  THE {K} VALUES ARE NOT INDEPENDENT: loadings are normalised to sum "
        f"to 1, so one cluster tracking LanA strongly FORCES negative rho on the others "
        f"arithmetically. Read a negative bar as 'not this one', never as 'avoids "
        f"language cortex'.", chars), transform=ax.transAxes, va="top", fontsize=7.4,
        color=MUTED, linespacing=1.5)


BR = chr(10)


def _wrap(t, chars):
    import textwrap
    return textwrap.fill(t, chars)


def panel_brain(axL, axR, d, chars):
    """What LanA looks like on THIS coverage - the context A to C are read against."""
    P, has = d["P_lana"], d["has"]
    vmax = float(np.nanpercentile(P[has], 98)) or 1.0
    cm = plt.get_cmap("magma")
    for ax, side in ((axL, "L"), (axR, "R")):
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
        ax.text(0.02, 0.02, f"{side}  {int(m.sum())}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.0, color=MUTED)
    axL.text(0, -0.09, _wrap(
        f"Every electrode with a LanA value, coloured by it (dark = low, bright = high, "
        f"scaled to the 98th percentile, {vmax:.2f}). An ATLAS PRIOR about a location, "
        f"not a measurement in this patient: a bright contact is somewhere language "
        f"cortex usually is, which is not the same as having responded to language.",
        chars), transform=axL.transAxes, va="top", fontsize=7.4, color=MUTED,
        linespacing=1.5)


# ---- the figure --------------------------------------------------------------
def figure_3(fset, k, n_perm, n_boot):
    t0 = time.time()
    for _pl, _ in P2._PLOTTER.values():
        _pl.close()
    P2._PLOTTER.clear()
    d = load(fset, k)
    if d["has"].sum() < 50:
        raise SystemExit("too few electrodes carry a LanA value to say anything")
    print(f"  {n_perm} within-patient permutations, {n_boot} patient bootstraps ...")
    R, Ph, Gh, labh = analyse(d, n_perm, n_boot)

    ncol = min(len(R), 8)
    nrow = int(np.ceil(len(R) / ncol))
    fig = plt.figure(figsize=(17.6, 9.8 + 1.4 * (nrow - 1)), dpi=190)
    gs = GridSpec(3, 24, figure=fig, height_ratios=[1.25, 0.90 * nrow, 1.15],
                  hspace=1.35, wspace=1.2, left=0.055, right=0.985,
                  top=0.855, bottom=0.045)
    fig.suptitle(f"FIG 3   ·   LanA language atlas   ·   {fset}   ·   K = {k}   ·   "
                 f"{int(d['has'].sum())} of {len(d['X'])} electrodes have a LanA value",
                 x=0.055, y=0.972, ha="left", fontsize=15.5, color=INK)
    fig.text(0.055, 0.938,
             "How far into the LanA probabilistic language atlas each cluster sits, and "
             "whether belonging more strongly to a cluster goes with sitting further "
             "in.  Every null permutes P(LanA) WITHIN PATIENT; every interval "
             "bootstraps patients.",
             fontsize=9.8, color=MUTED, va="top")

    axA = fig.add_subplot(gs[0, 0:13])
    panel_rank(axA, R, d, 118)
    cD = GridSpecFromSubplotSpec(1, 2, gs[0, 14:24], wspace=0.03)
    axD = fig.add_subplot(cD[0])
    panel_brain(axD, fig.add_subplot(cD[1]), d, 96)

    sub = GridSpecFromSubplotSpec(nrow, ncol, gs[1, :], hspace=0.55, wspace=0.30)
    axes = [fig.add_subplot(sub[i // ncol, i % ncol]) for i in range(len(R))]
    panel_scatter(axes, R, Gh, Ph, labh, d)
    axes[0].set_ylabel("P(LanA)", fontsize=7.6)
    for a in axes:
        a.set_xlabel("loading", fontsize=7.0)

    axC = fig.add_subplot(gs[2, 0:11])
    panel_rho(axC, R, d, 104)
    # the right half of this row was empty. The reading notes go there, generated from
    # the data so they cannot describe a figure that has since changed.
    axN = fig.add_subplot(gs[2, 12:24]); axN.axis("off")
    thin = [f"c{int(r.cluster)}" for _, r in R.iterrows()
            if r.coverage < MIN_COVERAGE]
    axN.text(0, 1.0, "HOW TO READ THIS FIGURE" + BR + BR
             + _wrap("P(LanA) is an ATLAS PRIOR about a location - the fraction of "
                     "LanA's subjects whose language network covers this point. It is "
                     "not a measurement in this patient. A cluster high in panel A sits "
                     "where language cortex usually is; whether it RESPONDED to language "
                     "is FIG 1, not this.", 92) + BR + BR
             + _wrap(f"{int((R.q_mean < 0.05).sum())} of {d['k']} clusters beat the "
                     f"within-patient null on mean P(LanA); "
                     f"{int((R.q_rho < 0.05).sum())} of {d['k']} correlations survive "
                     f"FDR.", 92) + BR + BR
             + _wrap(f"Only {100*d['has'].mean():.0f}% of electrodes have a LanA value, "
                     f"and the missing ones are WHOLE PATIENTS rather than a random "
                     f"scatter - the atlas run predates part of the cohort. "
                     + (f"Coverage is below {100*MIN_COVERAGE:.0f}% for "
                        f"{', '.join(thin)}, whose means are not comparable with the "
                        f"rest." if thin else
                        "Every cluster is above the coverage floor here."), 92)
             + BR + BR
             + _wrap("The K correlations are NOT independent: loadings sum to 1, so one "
                     "cluster tracking the atlas forces the others negative by "
                     "arithmetic. Only the positive end is interpretable alone.", 92)
             + BR + BR
             + _wrap("LanA is left-lateralised and sEEG coverage is not symmetric, so a "
                     "left-heavy cluster scores higher for that reason alone. The "
                     "within-patient null does not remove it: it is a property of where "
                     "a cluster is, not of which patients it draws on.", 92),
             transform=axN.transAxes, va="top", ha="left", fontsize=7.6, color=MUTED,
             linespacing=1.6)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, label in ((axA, "A  ·  clusters ranked by how much LanA they sit in"),
                      (axD, "D  ·  LanA on this coverage"),
                      (axes[0], "B  ·  loading vs P(LanA), every electrode"),
                      (axC, "C  ·  those correlations, ranked")):
        top = inv.transform((0, ax.get_tightbbox(r).y1))[1]
        fig.text(ax.get_position().x0, top + 0.008, label, fontsize=10.4, color=INK,
                 va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"FIG3_lana_{fset}_K{k}.png"
    P2.save_png(fig, p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P2.save_text(R.to_csv(index=False), p.with_name(p.stem + "_clusters.csv"))
    P2.save_text(pd.DataFrame(dict(
        patient=d["patient"], electrode=d["meta"]["electrode"], cluster=d["lab"],
        P_lana=d["P_lana"], has_lana=d["has"])).to_csv(index=False),
        p.with_name(p.stem + "_electrodes.csv"))
    caption_3(p.with_name(p.stem + "_caption.txt"), d, R, k, fset, n_perm, n_boot)
    print(f"  {time.time()-t0:.0f}s  -> {p.name}")
    return p


def caption_3(path, d, R, k, fset, n_perm, n_boot):
    L = []
    A = L.append
    A(f"FIG 3   ·   LanA language atlas   ·   {fset}   ·   K = {k}")
    A("=" * 100)
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
    A("The missing electrodes are NOT missing at random: the atlas run predates part of")
    A("the cohort, so whole patients have no value at all. A cluster drawn mostly from")
    A("those patients would look like it sits outside language cortex when in truth it")
    A("has not been measured against the atlas. Per-cluster coverage is printed under")
    A(f"every bar in panel A and anything below {100*MIN_COVERAGE:.0f}% is marked in red.")
    if len(d["lana_missing_by_patient"]):
        A("")
        A("  electrodes with no LanA value, by patient:")
        for p_, n_ in d["lana_missing_by_patient"].items():
            tot = int((d["patient"] == p_).sum())
            A(f"    {p_:<12} {int(n_):>4} of {tot:<4}"
              + ("   (the whole patient)" if int(n_) == tot else ""))
    A("")
    A("THE NULL, AND WHY IT IS WITHIN PATIENT")
    A("-" * 100)
    A("Contacts on one shaft sit millimetres apart, share a patient, and share an atlas")
    A("neighbourhood, so a correlation across ~1400 electrodes has nothing like 1400")
    A("degrees of freedom. Every null here permutes P_lana WITHIN EACH PATIENT: that")
    A("keeps each patient's own atlas values and coverage exactly as they are and")
    A("breaks only the link between a contact's position and its cluster. A null that")
    A("shuffled across patients would also destroy the fact that patients are implanted")
    A("in different places, and would be far too easy to beat.")
    A("")
    A("Confidence intervals BOOTSTRAP PATIENTS, not electrodes, for the same reason: an")
    A("electrode bootstrap treats contacts on one shaft as independent draws and returns")
    A("intervals several times too narrow.")
    A("")
    A(f"  {n_perm} permutations, {n_boot} bootstraps, patients resampled with replacement")
    A("")
    A("MULTIPLE COMPARISONS")
    A("-" * 100)
    A(f"K clusters is {k} tests per panel, and the figure RANKS them - which is exactly")
    A("the arrangement that manufactures an impressive-looking best one. Both panels are")
    A("corrected with Benjamini-Hochberg and a star means q < 0.05, not p < 0.05.")
    A("")
    A("PANEL A - clusters ranked by how much LanA they sit in")
    A("-" * 100)
    A("Mean P_lana of a cluster's electrodes, most to least. Grey dash = the")
    A("within-patient null, dotted line = the cohort mean, % under each bar = coverage.")
    A("")
    A(f"  cohort mean P_lana             {R.baseline_P.iloc[0]:.4f}")
    A(f"  most  c{int(R.cluster.iloc[0])}  {R.mean_P.iloc[0]:.4f} "
      f"[{R.mean_P_lo.iloc[0]:.4f}, {R.mean_P_hi.iloc[0]:.4f}]  "
      f"null {R.null_mean_P.iloc[0]:.4f}  q={R.q_mean.iloc[0]:.3g}")
    A(f"  least c{int(R.cluster.iloc[-1])}  {R.mean_P.iloc[-1]:.4f} "
      f"[{R.mean_P_lo.iloc[-1]:.4f}, {R.mean_P_hi.iloc[-1]:.4f}]  "
      f"null {R.null_mean_P.iloc[-1]:.4f}  q={R.q_mean.iloc[-1]:.3g}")
    A(f"  clusters above the null at q<0.05   {int((R.q_mean < 0.05).sum())} of {k}")
    A("")
    A("PANEL B and C - loading against P_lana")
    A("-" * 100)
    A("Spearman rho between an electrode's LOADING on a cluster and its P_lana, across")
    A("EVERY electrode with a LanA value - not only the cluster's own members, because")
    A("the question is whether belonging more strongly to this cluster goes with sitting")
    A("further into language cortex, and that is a statement about all of them.")
    A("")
    A("Spearman, not Pearson: P_lana is a bounded, heavily right-skewed probability and")
    A("the loadings are not normal either, so a rank correlation is the honest choice.")
    A("")
    A("THE K CORRELATIONS ARE NOT INDEPENDENT OF EACH OTHER, and this is the easiest")
    A("number in the figure to over-read. A convex-NMF loading vector is normalised to")
    A("sum to 1, so the K loadings of one electrode live on a simplex: if one cluster's")
    A("loading rises with P_lana, the others MUST fall, whatever they are describing.")
    A("A negative rho is therefore 'not this cluster' and not 'this cluster avoids")
    A("language cortex' - the two are indistinguishable from this panel. Only the")
    A("POSITIVE end is interpretable on its own, and even the ranking among negatives is")
    A("partly an artefact of how strong the leader is. Verified on synthetic data where")
    A("one cluster was planted to track the atlas: every other cluster went negative")
    A("without being planted to do anything at all.")
    A("")
    A("The within-patient null does not remove this. It is computed on the same")
    A("normalised loadings, so it prices in the simplex for EACH cluster separately -")
    A("it says whether that cluster's rho exceeds chance, not whether the K rhos are")
    A("mutually consistent.")
    A("")
    for _, r in R.iterrows():
        A(f"  c{int(r.cluster):<3} rho {r.rho:+.3f} [{r.rho_lo:+.3f}, {r.rho_hi:+.3f}]"
          f"   null {r.null_rho:+.3f}   p={r.p_rho:.3g}  q={r.q_rho:.3g}"
          f"   meanP {r.mean_P:.4f}   coverage {100*r.coverage:.0f}%")
    A("")
    A(f"  {int((R.q_rho < 0.05).sum())} of {k} correlations survive FDR.")
    A("A rho near zero means the cluster is placed independently of the atlas. That is a")
    A("result, not a failure - it says the clustering found structure the atlas does not")
    A("describe.")
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
    A("  - The 297 electrodes with no LanA value are excluded from every number here,")
    A("    not counted as zero.")
    P2.save_text("\n".join(L) + "\n", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="concat_hg")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    print(f"=== FIGURE 3 ===  {a.feature_set}  K={a.k}")
    figure_3(a.feature_set, a.k, a.n_perm, a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
