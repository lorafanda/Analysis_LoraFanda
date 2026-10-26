#!/usr/bin/env python3
"""00_paper2_figure3_spin.py - FIG 3b: the LanA comparison under a SPIN null, and what
the language network itself does during the task.

    python 00_paper2_figure3_spin.py --feature-set concat_bands15 --k 7
    python 00_paper2_figure3_spin.py --feature-set concat_bands15 --k 7 --n-spin 5000

TWO THINGS FIG 3 CANNOT DO.

(1) A SPATIAL NULL WITH POWER. FIG 3's shaft-shift null rolls the cluster labels along
    each shaft and leaves the atlas where it is. Clusters and shafts live at the same
    spatial scale, so for a type that fills whole shafts the roll changes nothing and
    the null reproduces the observed statistic almost exactly: no power. The spin
    test (Alexander-Bloch et al. 2018) does the opposite: it leaves every electrode,
    shaft and label where it is and RANDOMISES THE MAP. Each electrode is anchored to
    its nearest fsaverage pial vertex; that vertex's position on the fsaverage sphere
    is rotated by a random rotation (the mirror rotation on the right hemisphere, so
    the two sides stay symmetric), the rotated point is read back to the nearest
    sphere vertex, and the LanA atlas is sampled at that vertex's pial coordinate.
    The spun map keeps LanA's smoothness, size and left-right asymmetry, and the
    electrodes keep their patients, shafts and clusters. The null asks: how often
    does a random map with LanA's statistics give this cluster this share of its
    electrodes inside the network?

(2) THE RESPONSE OF THE NETWORK ITSELF. FIG 3 asks where the clusters sit relative to
    LanA. The inverse question is the one intracranial data can answer and fMRI
    cannot: what does the LanA network DO over the course of a trial, at this time
    resolution? The mean spectral response (the run's own features, in dB) of the
    electrodes inside the network against the electrodes outside it, per condition,
    with the cells where the difference beats the spin null outlined.

Both are computed on all electrodes and on the LEFT hemisphere alone: LanA is
left-lateralised and the cohort is right-heavy, so a type that leans left is 'inside
the network' for that reason alone unless the hemisphere is held fixed.

READS   the run (00_Paper2_Figures.load_run), the LanA tables (00_paper2_figure3_lana.load),
        the fsaverage pial meshes (outputs/250_recon/fsaverage/meshes/fsaverage_?h.gii),
        the fsaverage spheres (FSAVERAGE_DIR/surf/?h.sphere, the FreeSurfer subject on the
        share) and the atlas volume 04_FBM_Pooling/federenko_atlas/langloc_n806_p_0.05_atlas.nii
WRITES  paper_figures/FIG3_spin_<fset>_K<k>_run<id>.png
        ..._clusters.csv   per stratum x threshold x cluster: the share inside, the cohort
                           rate, the enrichment, the spin null (mean, 2.5-97.5%), p, q
        ..._cells.csv      per stratum x feature: mean inside, mean outside, difference,
                           the spin null and its p, q
        ..._summary.json   the one-number tests and the counts
        ..._caption.txt    full provenance and every number on the figure
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from scipy.spatial import cKDTree
import nibabel as nb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
sys.path.insert(0, str(ROOT.parent / "04_FBM_Pooling" / "functions"))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P2 = _load("p2_for_spin", "00_Paper2_Figures.py")
F3 = _load("f3_for_spin", "00_paper2_figure3_lana.py")
import lf_atlas_corr as A                       # noqa: E402
import lf_recon_shared_config as RC             # noqa: E402

OUT = P2.OUT
MESHES = P2.MESHES
SPHERES = Path(RC.FSAVERAGE_DIR) / "surf"
ATLAS_NII = ROOT.parent / "04_FBM_Pooling" / "federenko_atlas" / "langloc_n806_p_0.05_atlas.nii"
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
N_SPIN = 1000
THRESHOLDS = tuple(F3.THRESHOLDS)               # the same two as FIG 3: 0.1 and 0.2
STRATA = ("all", "L")
REFLECT = np.diag([-1.0, 1.0, 1.0])             # the mirror rotation for the right hemisphere


# ---- surfaces and sampling ----------------------------------------------------------
def surfaces():
    """fsaverage pial (the project's own meshes) and sphere (the FreeSurfer subject on
    the share), one per hemisphere, with the trees that make the nearest-vertex lookups
    fast. The gii pial equals surf/?h.pial vertex for vertex (checked 2026-10-26)."""
    S = {}
    for side, h in (("L", "lh"), ("R", "rh")):
        pial = np.asarray(nb.load(MESHES / f"fsaverage_{h}.gii").darrays[0].data, float)
        sph, _ = nb.freesurfer.read_geometry(str(SPHERES / f"{h}.sphere"))
        sph = np.asarray(sph, float)
        sph /= np.linalg.norm(sph, axis=1, keepdims=True)
        if len(sph) != len(pial):
            raise SystemExit(f"{h}: sphere has {len(sph)} vertices, pial {len(pial)}")
        S[side] = dict(pial=pial, sphere=sph, tree_pial=cKDTree(pial),
                       tree_sph=cKDTree(sph))
    return S


_ATLAS = {}


def _atlas():
    """The LanA volume, loaded once. sample_atlas_at_contacts reloads the NIfTI on
    every call; at 5000 spins x 2 hemispheres that reload would be the run time."""
    if not _ATLAS:
        img = nb.load(str(ATLAS_NII))
        _ATLAS["vol"] = np.asarray(img.dataobj, dtype=float)
        _ATLAS["inv"] = np.linalg.inv(img.affine)
    return _ATLAS["vol"], _ATLAS["inv"]


def sample(xyz):
    """LanA at fsaverage coordinates, the way make_lana_runs sampled it at the contacts
    (nearest 2 mm voxel after the MNI305 -> MNI152 affine); outside the volume -> 0,
    counted. The same arithmetic as lf_atlas_corr.sample_atlas_at_contacts, and
    checked against it once per run (check_sampler)."""
    vol, inv = _atlas()
    xyz = np.asarray(xyz, float)
    x152 = (np.c_[xyz, np.ones(len(xyz))] @ A.MNI305_TO_152.T)[:, :3]
    ijk = np.rint(np.c_[x152, np.ones(len(x152))] @ inv.T)[:, :3].astype(int)
    ok = (ijk >= 0).all(1) & (ijk < np.array(vol.shape[:3])).all(1)
    out = np.zeros(len(xyz))
    out[ok] = vol[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return out, int((~ok).sum())


def check_sampler(xyz):
    """The cached sampler must give what the project's sampler gives, value for value."""
    ref = np.nan_to_num(A.sample_atlas_at_contacts(np.asarray(xyz, float), ATLAS_NII), nan=0.0)
    mine, _ = sample(xyz)
    if not np.array_equal(ref, mine):
        raise SystemExit("the cached atlas sampler disagrees with "
                         "lf_atlas_corr.sample_atlas_at_contacts")


def anchor(d, S):
    """Each electrode's nearest pial vertex on its own hemisphere, and how far it is."""
    n = len(d["xyz"])
    vidx, dist = np.zeros(n, int), np.zeros(n)
    for side in ("L", "R"):
        m = d["hemi"] == side
        if m.any():
            dd, vv = S[side]["tree_pial"].query(d["xyz"][m])
            vidx[m], dist[m] = vv, dd
    return vidx, dist


def surface_P(d, S, vidx):
    """LanA at every electrode's anchor vertex: the observed values the spin is
    compared with, on the same footing as the spun ones (a surface point each)."""
    P = np.zeros(len(vidx))
    n_out = 0
    for side in ("L", "R"):
        m = d["hemi"] == side
        if m.any():
            P[m], o = sample(S[side]["pial"][vidx[m]])
            n_out += o
    return P, n_out


def random_rotation(rng):
    """A rotation drawn uniformly (QR of a Gaussian matrix, signs fixed, det +1)."""
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def spin_once(d, S, vidx, rng):
    """One spun map: the atlas read at where each electrode's anchor vertex lands
    after a random rotation of the sphere (its mirror on the right hemisphere)."""
    Q = random_rotation(rng)
    P = np.zeros(len(vidx))
    for side, Qs in (("L", Q), ("R", REFLECT @ Q @ REFLECT)):
        m = d["hemi"] == side
        if not m.any():
            continue
        pts = S[side]["sphere"][vidx[m]] @ Qs.T
        _, vv = S[side]["tree_sph"].query(pts)
        P[m], _ = sample(S[side]["pial"][vv])
    return P


# ---- statistics -------------------------------------------------------------------
def bh_fdr(p):
    return F3.bh_fdr(p)


def shares(P, lab, K, mask, thr):
    """Per cluster, the share of its electrodes (inside `mask`) at P >= thr, and the
    cohort rate on the same electrodes."""
    out = np.full(K, np.nan)
    inside = P >= thr
    for j in range(K):
        m = mask & (lab == j)
        if m.any():
            out[j] = inside[m].mean()
    base = inside[mask].mean() if mask.any() else np.nan
    return out, base


def mean_P(P, lab, K, mask):
    return np.array([np.nanmean(P[mask & (lab == j)]) if (mask & (lab == j)).any() else np.nan
                     for j in range(K)])


def profiles(X, P, mask, thr):
    """Mean feature vector inside and outside the network, on the stratum's electrodes."""
    inside = (P >= thr) & mask
    outside = (P < thr) & mask
    mi = X[inside].mean(0) if inside.any() else np.full(X.shape[1], np.nan)
    mo = X[outside].mean(0) if outside.any() else np.full(X.shape[1], np.nan)
    return mi, mo, int(inside.sum()), int(outside.sum())


def two_sided_p(null, obs):
    """Permutation p against the null's own centre, two-sided: how often a spun map
    is at least as far from the null mean as the observed value is."""
    c = np.nanmean(null, axis=0)
    dev = np.abs(null - c)
    return (np.sum(dev >= np.abs(obs - c), axis=0) + 1) / (null.shape[0] + 1)


def analyse(d, S, n_spin, seed):
    rng = np.random.default_rng(seed)
    K, lab, X = d["k"], d["lab"], d["X"]
    vidx, dist = anchor(d, S)
    check_sampler(np.vstack([S[h]["pial"][vidx[d["hemi"] == h]] for h in ("L", "R")]))
    P_obs, n_out = surface_P(d, S, vidx)
    masks = {"all": np.ones(len(lab), bool), "L": d["hemi"] == "L"}
    t0 = THRESHOLDS[0]

    obs = {}
    for st, mask in masks.items():
        o = dict(meanP=mean_P(P_obs, lab, K, mask), baseP=float(P_obs[mask].mean()))
        for t in THRESHOLDS:
            o[f"share_{F3.tkey(t)}"], o[f"base_{F3.tkey(t)}"] = shares(P_obs, lab, K, mask, t)
        mi, mo, n_in, n_o = profiles(X, P_obs, mask, t0)
        o.update(mean_in=mi, mean_out=mo, n_in=n_in, n_out=n_o, diff=mi - mo,
                 rms=float(np.sqrt(np.nanmean((mi - mo) ** 2))))
        obs[st] = o

    null = {st: dict(meanP=np.full((n_spin, K), np.nan),
                     **{f"share_{F3.tkey(t)}": np.full((n_spin, K), np.nan) for t in THRESHOLDS},
                     **{f"base_{F3.tkey(t)}": np.full(n_spin, np.nan) for t in THRESHOLDS},
                     diff=np.full((n_spin, X.shape[1]), np.nan), rms=np.full(n_spin, np.nan))
            for st in masks}
    t_start = time.time()
    for s in range(n_spin):
        Ps = spin_once(d, S, vidx, rng)
        for st, mask in masks.items():
            nn = null[st]
            nn["meanP"][s] = mean_P(Ps, lab, K, mask)
            for t in THRESHOLDS:
                nn[f"share_{F3.tkey(t)}"][s], nn[f"base_{F3.tkey(t)}"][s] = shares(Ps, lab, K, mask, t)
            mi, mo, _, _ = profiles(X, Ps, mask, t0)
            nn["diff"][s] = mi - mo
            nn["rms"][s] = np.sqrt(np.nanmean((mi - mo) ** 2))
        if s in (9, 99, 499) or (s + 1) % 1000 == 0:
            print(f"    spin {s + 1}/{n_spin}  {time.time() - t_start:.0f}s")

    # tables
    rows, cells = [], []
    summary = dict(n_spin=n_spin, seed=seed, thresholds=list(THRESHOLDS),
                   n_electrodes=int(len(lab)), n_left=int(masks["L"].sum()),
                   anchor_dist_mm=dict(median=float(np.median(dist)), max=float(dist.max()),
                                       over_2mm=int((dist > 2).sum()),
                                       over_2mm_by_patient={str(k_): int(v_) for k_, v_ in
                                                            pd.Series(d["patient"][dist > 2])
                                                            .value_counts().items()}),
                   atlas_outside_volume=n_out,
                   surface_vs_contact_P=dict(
                       max_abs_diff=float(np.nanmax(np.abs(P_obs - d["P_lana"]))),
                       n_differ_over_0p01=int((np.abs(P_obs - d["P_lana"]) > 0.01).sum())))
    feats = d.get("feature_names")
    for st in masks:
        o, nn = obs[st], null[st]
        p_m = two_sided_p(nn["meanP"], o["meanP"])
        q_m = bh_fdr(p_m)
        for t in THRESHOLDS:
            c = f"share_{F3.tkey(t)}"
            p = two_sided_p(nn[c], o[c])
            q = bh_fdr(p)
            for j in range(K):
                rows.append(dict(
                    stratum=st, threshold=t, cluster=j,
                    n=int((masks[st] & (lab == j)).sum()),
                    inside=float(o[c][j]), cohort=float(o[f"base_{F3.tkey(t)}"]),
                    enrichment=float(o[c][j] / o[f"base_{F3.tkey(t)}"]),
                    null_mean=float(np.nanmean(nn[c][:, j])),
                    null_lo=float(np.nanpercentile(nn[c][:, j], 2.5)),
                    null_hi=float(np.nanpercentile(nn[c][:, j], 97.5)),
                    p=float(p[j]), q=float(q[j]),
                    mean_P=float(o["meanP"][j]), null_mean_P=float(np.nanmean(nn["meanP"][:, j])),
                    p_mean_P=float(p_m[j]), q_mean_P=float(q_m[j])))
        pc = two_sided_p(nn["diff"], o["diff"])
        qc = bh_fdr(pc)
        for i in range(X.shape[1]):
            cells.append(dict(stratum=st, feature=feats[i] if feats else i,
                              mean_in=float(o["mean_in"][i]), mean_out=float(o["mean_out"][i]),
                              diff=float(o["diff"][i]),
                              null_mean=float(np.nanmean(nn["diff"][:, i])),
                              null_sd=float(np.nanstd(nn["diff"][:, i])),
                              p=float(pc[i]), q=float(qc[i])))
        summary[st] = dict(
            n_in=o["n_in"], n_out=o["n_out"],
            rms_diff_db=o["rms"], rms_null_mean=float(np.nanmean(nn["rms"])),
            rms_null_95=float(np.nanpercentile(nn["rms"], 95)),
            p_rms=float((np.sum(nn["rms"] >= o["rms"]) + 1) / (n_spin + 1)),
            cells_q05=int((qc < 0.05).sum()), cells=int(X.shape[1]),
            **{f"clusters_q05_{F3.tkey(t)}": int((bh_fdr(two_sided_p(
                nn[f"share_{F3.tkey(t)}"], o[f"share_{F3.tkey(t)}"])) < 0.05).sum())
               for t in THRESHOLDS})
        obs[st]["q_cells"] = qc
    return (pd.DataFrame(rows), pd.DataFrame(cells), summary, obs, null, P_obs)


# ---- the figure ---------------------------------------------------------------
def block(v, d):
    """A feature vector (band x cond x time) as draw_mean's (ncond, nband, nt)."""
    nb_, nc, nt = len(d["bands"]), len(d["conds"]), d["nt"]
    return v.reshape(nb_, nc, nt).transpose(1, 0, 2)


def wrap(t, chars=100):
    import textwrap
    return textwrap.fill(t, chars)


def panel_profiles(fig, cells, d, obs, st, t0):
    nb_, nc, nt = len(d["bands"]), len(d["conds"]), d["nt"]
    o = obs[st]
    vlim = float(max(np.nanmax(np.abs(o["mean_in"])), np.nanmax(np.abs(o["mean_out"]))))
    vlim = max(0.5, np.ceil(vlim * 4) / 4)
    vdiff = max(0.25, np.ceil(float(np.nanmax(np.abs(o["diff"]))) * 4) / 4)
    axes = [fig.add_subplot(c) for c in cells]
    titles = (f"inside the network, P(LanA) ≥ {t0:.1f}   n = {o['n_in']}",
              f"outside, P(LanA) < {t0:.1f}   n = {o['n_out']}",
              "inside minus outside (dB); outlined = beats the spin null, q < 0.05")
    ims = []
    for ax, v, ttl, vl, cm in zip(axes, (o["mean_in"], o["mean_out"], o["diff"]),
                                  titles, (vlim, vlim, vdiff),
                                  ("RdBu_r", "RdBu_r", "PuOr_r")):
        im = P2.draw_mean(ax, block(v, d), d, vlim=vl, cmap=cm, show_axes=False,
                          cond_names=True)
        ims.append(im)
        ax.set_title(ttl, fontsize=8.6, loc="left", color=INK, pad=14)
    # the bands once, on the left panel; a second and third copy sat on the colour bars
    axes[0].set_yticks(range(nb_))
    axes[0].set_yticklabels([b.replace("Hz", "") for b in d["bands"]], fontsize=6.0)
    axes[0].tick_params(axis="y", labelsize=6.0, colors=MUTED, length=2)
    axes[0].set_ylabel("Hz", fontsize=7.4)
    mask = (o["q_cells"] < 0.05).reshape(nb_, nc, nt).reshape(nb_, nc * nt)
    P2.outline_cells(axes[2], mask, nt)
    for ax, im, lab in ((axes[1], ims[1], "dB re baseline (inside and outside)"),
                        (axes[2], ims[2], "dB, inside − outside")):
        if im is not None:
            cax = ax.inset_axes([1.018, 0.0, 0.024, 1.0])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6.2, colors=MUTED, length=2)
            cb.set_label(lab, fontsize=6.8, color=MUTED)
    return axes


def panel_shares(ax, tab, d, st, order, chars):
    """Per cluster, the share inside the network at each threshold, with the spin
    null's 95% band behind each bar and the cohort rate as a line."""
    K = d["k"]
    T = tab[tab.stratum == st]
    x = np.arange(len(order))
    nT = len(THRESHOLDS)
    w = 0.80 / nT
    for i_t, t in enumerate(THRESHOLDS):
        sub = T[T.threshold == t].set_index("cluster").loc[order]
        xs = x - 0.40 + w * (i_t + 0.5)
        alpha = 1.0 if i_t == 0 else 0.45
        for xi, (j, r) in zip(xs, sub.iterrows()):
            ax.add_patch(Rectangle((xi - w * 0.47, r.null_lo), w * 0.94, r.null_hi - r.null_lo,
                                   fc=GREY, ec="none", alpha=0.55, zorder=1))
            ax.bar(xi, r.inside, width=w * 0.62, color=P2.cluster_col(int(j), K),
                   alpha=alpha, lw=0, zorder=2)
            ax.plot([xi - w * 0.47, xi + w * 0.47], [r.null_mean, r.null_mean], color=INK,
                    lw=1.2, zorder=3)
            star = "*" if r.q < 0.05 else ""
            lab = f"{star}\n{r.enrichment:.1f}×" if i_t == 0 else star
            ax.text(xi, max(r.inside, r.null_hi) + 0.012, lab, ha="center", va="bottom",
                    fontsize=6.3, color=INK if i_t == 0 else MUTED)
        base = float(sub.cohort.iloc[0])
        ax.axhline(base, color=MUTED, ls=":" if i_t == 0 else (0, (1, 2.2)), lw=1.1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{j}" for j in order], fontsize=7.2)
    ax.set_ylabel("share of its electrodes inside LanA", fontsize=8.2)
    top = float(T[["inside", "null_hi"]].max().max())
    ax.set_ylim(0, max(0.5, top * 1.35))
    ax.tick_params(labelsize=7.2, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    base0 = {t: float(T[T.threshold == t].cohort.iloc[0]) for t in THRESHOLDS}
    ax.legend([Patch(color=INK, alpha=1.0), Patch(color=INK, alpha=0.45),
               Patch(color=GREY, alpha=0.55)],
              [f"P ≥ {t:.1f}   (cohort {100*base0[t]:.0f}%, dotted)" for t in THRESHOLDS]
              + ["spin null, 2.5 to 97.5%"],
              fontsize=6.4, frameon=False, loc="upper right", ncol=1, handlelength=1.2)
    n_sig = {t: int((T[T.threshold == t].q < 0.05).sum()) for t in THRESHOLDS}
    ax.text(0, -0.17, wrap(
        f"{'All electrodes' if st == 'all' else 'Left hemisphere only'}: n = "
        f"{int(T[T.threshold == THRESHOLDS[0]].n.sum())}. Bars = the share inside at P ≥ "
        f"{THRESHOLDS[0]:.1f} (solid) and ≥ {THRESHOLDS[1]:.1f} (pale); the grey box behind "
        f"each bar is where 95% of the spun maps put it and the dark dash their mean; "
        f"the number is the enrichment against the cohort rate. * = outside the spin "
        f"null, two-sided, Benjamini-Hochberg over {K}: "
        + ", ".join(f"{n_sig[t]} of {K} at P ≥ {t:.1f}" for t in THRESHOLDS) + ".", chars),
        transform=ax.transAxes, va="top", fontsize=7.2, color=MUTED, linespacing=1.5)


def panel_rms(ax, summary, null, obs, chars):
    """The one-number test: the size of the inside-minus-outside difference, observed
    against the spun maps, for both strata."""
    cols = {"all": INK, "L": "#1f5f8b"}
    for st in STRATA:
        nn = null[st]["rms"]
        ax.hist(nn, bins=30, color=cols[st], alpha=0.25 if st == "all" else 0.35,
                lw=0, density=True, label=f"spun maps, {'all' if st == 'all' else 'left'}")
        ax.axvline(obs[st]["rms"], color=cols[st], lw=1.6,
                   label=f"observed, {'all' if st == 'all' else 'left'}: "
                         f"{obs[st]['rms']:.3f} dB, p = {summary[st]['p_rms']:.3g}")
    ax.set_xlabel("RMS of the inside − outside difference over all features (dB)",
                  fontsize=7.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=7.0, colors=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right")
    ax.text(0, -0.24, wrap(
        "Does the network have a response of its own? One number per map: the root "
        "mean square of the inside-minus-outside difference over every band, bin and "
        "condition. The histogram is that number for the spun maps, the line the real "
        "one; p is the share of spun maps at or above it.", chars),
        transform=ax.transAxes, va="top", fontsize=7.2, color=MUTED, linespacing=1.5)


def figure(fset, k, n_spin, seed):
    t_all = time.time()
    print(f"=== FIGURE 3b (spin null)  {fset}  K={k}  {n_spin} spins ===")
    d = F3.load(fset, k)
    d["feature_names"] = json.loads((d["run"] / "feature_schema.json").read_text())["feature_names"]
    S = surfaces()
    tab, cells, summary, obs, null, P_obs = analyse(d, S, n_spin, seed)
    t0 = THRESHOLDS[0]
    # FIG 3's cluster order: by mean P(LanA) of the contact values, most to least
    all_mask = np.ones(len(d["lab"]), bool)
    order = [int(j) for j in np.argsort(-mean_P(d["P_lana"], d["lab"], k, all_mask))]

    fig = plt.figure(figsize=(17.6, 12.6), dpi=190)
    # two blocks: the heatmap rows (their own, tighter, row gap) and the bar row, whose
    # notes hang below the axes and need the gap above and the margin below
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.0, 1.08], hspace=0.24,
                     left=0.05, right=0.985, top=0.868, bottom=0.105)
    gh = GridSpecFromSubplotSpec(2, 3, outer[0], hspace=0.30, wspace=0.17)
    gs = GridSpecFromSubplotSpec(1, 24, outer[1], wspace=1.0)
    fig.suptitle(f"FIG 3b   ·   LanA under a spin null, and what the network does   ·   "
                 f"{fset}   ·   K = {k}   ·   {len(d['lab'])} electrodes, {d['n_patients']} patients",
                 x=0.05, y=0.975, ha="left", fontsize=15.5, color=INK)
    fig.text(0.05, 0.957,
             "The map is randomised, the electrodes stay: each contact's fsaverage vertex is spun on "
             "the sphere and the atlas read where it lands, so the null keeps LanA's smoothness and "
             f"asymmetry and the electrodes keep their patients, shafts and clusters ({n_spin} spins).\n"
             "Rows A and B: the network's own response, all electrodes and the left hemisphere; "
             "C and D: each cluster's share inside the network against the spun maps; E: one number.",
             fontsize=9.4, color=MUTED, va="top", linespacing=1.4)
    axes_all = panel_profiles(fig, [gh[0, 0], gh[0, 1], gh[0, 2]], d, obs, "all", t0)
    axes_L = panel_profiles(fig, [gh[1, 0], gh[1, 1], gh[1, 2]], d, obs, "L", t0)
    axS1 = fig.add_subplot(gs[0, 0:9])
    panel_shares(axS1, tab, d, "all", order, 92)
    axS2 = fig.add_subplot(gs[0, 9:18])
    panel_shares(axS2, tab, d, "L", order, 92)
    axR = fig.add_subplot(gs[0, 18:24])
    panel_rms(axR, summary, null, obs, 60)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    labels = (
        (axes_all[0], f"A  ·  the network's response, all electrodes "
                      f"(inside n = {obs['all']['n_in']}, outside n = {obs['all']['n_out']})"),
        (axes_L[0], f"B  ·  the same, left hemisphere only "
                    f"(inside n = {obs['L']['n_in']}, outside n = {obs['L']['n_out']})"),
        (axS1, "C  ·  inside the network per cluster, against the spin null"),
        (axS2, "D  ·  the same, left hemisphere only"),
        (axR, "E  ·  one number: the size of the network's own response"))
    for ax, label in labels:
        top = inv.transform((0, ax.get_tightbbox(r).y1))[1]
        fig.text(ax.get_position().x0, top + 0.008, label, fontsize=10.2, color=INK,
                 va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    rt = "_run" + str(d["run"].name).replace("_", "-")
    p = OUT / f"FIG3_spin_{fset}_K{k}{rt}.png"
    P2.stamp(fig, d)
    P2.save_png(fig, p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P2.save_text(tab.to_csv(index=False), p.with_name(p.stem + "_clusters.csv"))
    P2.save_text(cells.to_csv(index=False), p.with_name(p.stem + "_cells.csv"))
    P2.save_text(json.dumps(summary, indent=1) + "\n", p.with_name(p.stem + "_summary.json"))
    caption(p.with_name(p.stem + "_caption.txt"), d, tab, summary, obs, order, n_spin, seed)
    print(f"  {time.time() - t_all:.0f}s  -> {p.name}")
    return p


def caption(path, d, tab, summary, obs, order, n_spin, seed):
    k = d["k"]
    L = []
    A_ = L.append
    A_(f"FIG 3b   ·   LanA under a spin null, and what the network does   ·   {d['fset']}   ·   K = {k}")
    A_("=" * 100)
    A_("")
    A_("WHY A SPIN NULL")
    A_("-" * 100)
    A_("FIG 3 tests each cluster against a shaft-shift null that rolls the cluster labels")
    A_("along each shaft and leaves the atlas in place. Clusters and shafts live at the")
    A_("same spatial scale, so for a type that fills whole shafts the roll changes nothing")
    A_("and the null reproduces the observed share almost exactly - it has no power. The")
    A_("spin test (Alexander-Bloch et al. 2018, NeuroImage 178:540) turns it round: the")
    A_("electrodes, their shafts, patients and labels stay exactly where they are, and the")
    A_("MAP is randomised. Each electrode is anchored to its nearest fsaverage pial vertex;")
    A_("that vertex's point on the fsaverage sphere is rotated by a random rotation (drawn")
    A_("uniformly; on the right hemisphere the mirror of the same rotation, so the two")
    A_("sides stay symmetric), the rotated point is read back to the nearest sphere vertex,")
    A_("and the LanA volume is sampled at that vertex's pial coordinate with the same")
    A_("sampler that gave the contacts their values (nearest 2 mm voxel after the")
    A_("MNI305 -> MNI152 affine; lf_atlas_corr.sample_atlas_at_contacts). The spun map")
    A_("keeps LanA's smoothness, size and left-right asymmetry; what it breaks is only the")
    A_("alignment between the map and the electrodes. Observed values are the atlas at the")
    A_("anchor vertices, so observed and spun are on the same footing (a surface point).")
    A_("")
    sv = summary["surface_vs_contact_P"]
    ad = summary["anchor_dist_mm"]
    A_(f"  spins                      {n_spin}   (seed {seed})")
    A_(f"  anchor distance, contact to nearest pial vertex   median {ad['median']:.2f} mm, "
       f"max {ad['max']:.1f} mm, {ad['over_2mm']} over 2 mm"
       + (": " + ", ".join(f"{k_} {v_}" for k_, v_ in ad["over_2mm_by_patient"].items())
          if ad["over_2mm"] else ""))
    A_(f"  atlas at the anchor vs at the contact   max |diff| {sv['max_abs_diff']:.3f}, "
       f"{sv['n_differ_over_0p01']} electrodes differ by more than 0.01")
    A_(f"  spun points outside the atlas volume   {summary['atlas_outside_volume']} (read as 0)")
    A_("")
    A_("The second thing FIG 3 cannot show is the inverse question: what does the LanA")
    A_("network itself DO during the task. Panels A and B are the mean response (the run's")
    A_("own features, dB re baseline, band x time x condition) of the electrodes inside the")
    A_(f"network at P >= {THRESHOLDS[0]:.1f} against the electrodes outside it, and their difference;")
    A_("the outlined cells are where the difference is beyond what the spun maps give")
    A_("(two-sided, Benjamini-Hochberg over every cell). Panel E is the one-number version:")
    A_("the RMS of that difference over all cells, observed against spun.")
    A_("")
    A_("Everything is computed twice: on all electrodes and on the LEFT hemisphere alone,")
    A_("because LanA is left-lateralised and the cohort is right-heavy, so a type that")
    A_("leans left is 'inside the network' for that reason alone unless the hemisphere is")
    A_("held fixed.")
    A_("")
    for st in STRATA:
        s = summary[st]
        A_(f"PANEL {'A' if st == 'all' else 'B'} and E - the network's response, "
           f"{'all electrodes' if st == 'all' else 'left hemisphere'}")
        A_("-" * 100)
        A_(f"  inside n = {s['n_in']}, outside n = {s['n_out']}")
        A_(f"  cells beyond the spin null at q < 0.05   {s['cells_q05']} of {s['cells']}")
        A_(f"  RMS difference {s['rms_diff_db']:.4f} dB   spun maps mean {s['rms_null_mean']:.4f}, "
           f"95th percentile {s['rms_null_95']:.4f}   p = {s['p_rms']:.3g}")
        o = obs[st]
        nb_, nc, nt = len(d["bands"]), len(d["conds"]), d["nt"]
        D = o["diff"].reshape(nb_, nc, nt)
        A_("  mean inside-minus-outside difference (dB) by band and condition, stimulus half | response half:")
        A_("    band        " + "   ".join(f"{c:<16}" for c in d["conds"]))
        for b_i, b in enumerate(d["bands"]):
            A_(f"    {b:<11} " + "   ".join(f"{D[b_i, c_i, :nt//2].mean():+.2f} | {D[b_i, c_i, nt//2:].mean():+.2f}     "
                                             for c_i in range(nc)))
        A_("")
    for st in STRATA:
        A_(f"PANEL {'C' if st == 'all' else 'D'} - inside the network per cluster, "
           f"{'all electrodes' if st == 'all' else 'left hemisphere'}")
        A_("-" * 100)
        T = tab[tab.stratum == st]
        for t in THRESHOLDS:
            sub = T[T.threshold == t].set_index("cluster").loc[order]
            A_(f"  P >= {t:.2f}   cohort rate {100*float(sub.cohort.iloc[0]):.1f}%   "
               f"{int((sub.q < 0.05).sum())} of {k} outside the spin null at q < 0.05 (two-sided)")
            A_("  cluster   n     inside   x cohort   spin null mean [2.5, 97.5]     p        q")
            for j, r in sub.iterrows():
                A_(f"  c{int(j):<7} {int(r.n):<5} {100*r.inside:5.1f}%   {r.enrichment:5.2f}     "
                   f"{100*r.null_mean:5.1f}%  [{100*r.null_lo:5.1f}, {100*r.null_hi:5.1f}]   "
                   f"{r.p:<7.3g}  {r.q:<7.3g}")
            A_("")
        sub = T[T.threshold == THRESHOLDS[0]].set_index("cluster").loc[order]
        A_("  mean P(LanA) per cluster against the spun maps (the statistic of FIG 3 A, two-sided):")
        for j, r in sub.iterrows():
            A_(f"  c{int(j):<7} {r.mean_P:.4f}   spun mean {r.null_mean_P:.4f}   p {r.p_mean_P:<7.3g} q {r.q_mean_P:<7.3g}")
        A_("")
    A_("HOW TO READ IT")
    A_("-" * 100)
    A_(wrap("A cluster whose bar sits inside its grey box is where a random map with LanA's "
            "statistics would put it: its share of language cortex is what its shafts' "
            "positions give any map of that size and smoothness. A bar above the box is a "
            "type that sits in language cortex more than such a map predicts; below, less. "
            "The outlined cells of A and B are the parts of the network's response that no "
            "spun map reproduces: that is the response the LanA network has and the rest "
            "of the sampled cortex does not, in this task. Read B (left only) before A: on "
            "all electrodes a difference can be the two hemispheres differing.", 100))
    A_("")
    A_("WHAT THIS FIGURE DOES NOT SHOW")
    A_("-" * 100)
    A_("  - Not evidence of language responsiveness per se: LanA is the sentences-versus-")
    A_("    nonwords contrast, which subtracts auditory, visual-form and articulatory cortex.")
    A_("    Types built on those phases are outside the network by construction.")
    A_("  - The spin keeps the map's statistics, not the cohort's: where no patient has a")
    A_("    shaft, no spun map can put an electrode either.")
    A_(f"  - Thresholds are a declared choice ({', '.join(f'{t:.1f}' for t in THRESHOLDS)}); the")
    A_("    cell test of A and B uses the first.")
    A_(f"Written {time.strftime('%Y-%m-%d %H:%M')} by 00_paper2_figure3_spin.py.")
    P2.save_text("\n".join(L) + "\n", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="concat_bands15")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--n-spin", type=int, default=N_SPIN)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    figure(a.feature_set, a.k, a.n_spin, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
