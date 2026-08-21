#!/usr/bin/env python3
"""
make_component_anatomy.py - where does each convex-NMF component live, without ever
taking a hard partition?

THE PROBLEM THIS SOLVES. FIG C.7 established that the K=7 decomposition is real
(bi-cross-validation rises to a peak near K=8) but that the argmax partition taken
from it is not a separation (silhouette z = +1.0 against a one-blob null). That leaves
the loadings with nowhere to go: they describe the data well and they do not license
cluster labels.

Castellucci et al. (2026, Cell Reports 45, 116783) hit the same wall - electrodes did
not form discrete clusters by NMF factor weight - and their answer was to change the
unit of analysis from the ELECTRODE to the VOXEL. Rather than asking which component
an electrode belongs to, ask which cortical locations are weighted more highly for a
component than chance allows. The loadings stay continuous throughout; anatomy
supplies the discreteness the weights do not have.

    for each 1 cm voxel containing >= 3 localised contacts:
        observed = median loading of component j over the contacts inside
        null     = the same after shuffling loadings across contacts, 1000x
        significant if observed exceeds the 95th percentile of that null

TWO NULLS, NOT ONE. The paper shuffles across all electrodes. That does not control
for the fact that one patient's contacts are both spatially clustered AND likely to
load alike, so a single patient with a dense grid in one gyrus can manufacture a
significant voxel. The WITHIN-PATIENT shuffle permutes loadings only among contacts of
the same patient, so it asks whether a location is special given who was recorded
there. Both are reported so the difference is visible rather than assumed.

THREE THRESHOLDS, BECAUSE THE UNCORRECTED ONE DOES NOT SURVIVE CONTACT WITH THE
MULTIPLICITY. Overlapping voxels tested at an uncorrected 95th percentile is ~15,000
tests whose only guard in the paper is the >= 3 contact rule; 5% of them are expected
to pass on noise alone, and that expectation is printed next to the result. Alongside
it:

    BH-FDR      the standard control for a voxel map - expected proportion of
                false discoveries among those declared, q = 0.05
    MAX-STAT    family-wise: the 95th percentile of the per-permutation MAXIMUM,
                so one false positive anywhere is a 5% event

They answer different questions and here they disagree, which is worth seeing rather
than hiding: max-stat rewards a single extreme peak, FDR rewards many moderate voxels.

USE 10000 PERMUTATIONS, NOT 1000. The smallest attainable p is 1/(nperm+1), and
BH-FDR cannot declare a voxel unless enough of them sit at that floor - at nperm=1000
the floor is 0.001 and a component needs ~300 voxels there before FDR fires at all. On
this dataset that alone was the difference between 2 and 5 of the seven components
being localisable; the data did not change, the Monte Carlo resolution did. It costs
four minutes rather than twenty seconds. Max-stat is unaffected either way, since it
thresholds the statistic rather than its rank.

THE FRAME IS fsaverage, not MNI152. Same idea, different template, labelled
accordingly rather than borrowing the paper's wording.

    python make_component_anatomy.py --nperm 10000
    python make_component_anatomy.py --step 3 --nperm 500      # faster
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "component_anatomy"

# Yeo 7-network names, so the composition panel reads as anatomy rather than indices.
YEO7 = {
    "7Networks_1": "visual",
    "7Networks_2": "somatomotor",
    "7Networks_3": "dorsal attn",
    "7Networks_4": "ventral attn",
    "7Networks_5": "limbic",
    "7Networks_6": "frontoparietal",
    "7Networks_7": "default",
    "FreeSurfer_Defined_Medial_Wall": "medial wall",
}
INK, MUTED = "#1b232c", "#68727d"
GREY = "#d8dce0"
Q = 0.05


def load(run_dir):
    csv = next((run_dir / "recon").glob("*__with_fsaverage.csv"), None)
    if csv is None:
        raise FileNotFoundError(f"no __with_fsaverage.csv in {run_dir / 'recon'}")
    d = pd.read_csv(csv).dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    wcols = sorted((c for c in d.columns if c[:1] == "w" and c[1:].isdigit()),
                   key=lambda s: int(s[1:]))
    return d, wcols, csv


def build_voxels(xyz, step, cube, min_elec):
    """{voxel centre -> contact indices}, keeping only voxels with enough contacts.

    Rather than sweeping a grid and searching for contacts in each cell, walk the
    contacts and stamp each one onto every lattice point within half a cube of it.
    The two are equivalent; this one costs (n_contacts x offsets) instead of
    (n_grid_points x n_contacts), and a grid over a whole brain is mostly empty.
    """
    half = cube / 2.0
    k = int(np.floor(half / step))
    offs = np.arange(-k, k + 1) * step
    lat = np.round(xyz / step).astype(int) * step        # nearest lattice point
    vox = defaultdict(list)
    for i in range(len(xyz)):
        for dx in offs:
            for dy in offs:
                for dz in offs:
                    c = (lat[i, 0] + dx, lat[i, 1] + dy, lat[i, 2] + dz)
                    # the lattice rounding does not guarantee the contact is really
                    # inside the cube at the edges, so check
                    if (abs(c[0] - xyz[i, 0]) <= half
                            and abs(c[1] - xyz[i, 1]) <= half
                            and abs(c[2] - xyz[i, 2]) <= half):
                        vox[c].append(i)
    return {c: np.asarray(v) for c, v in vox.items() if len(v) >= min_elec}


def medians_by_group(W, groups_by_size):
    """Median of every column of W over each group of rows.

    Groups are pre-bucketed by SIZE so each bucket is one (n_vox, size, n_comp) array
    and the median is a single vectorised call. Looping voxel by voxel across 1000
    permutations is what makes the naive version unusable.
    """
    n_vox = sum(len(gis) for gis, _ in groups_by_size.values())
    out = np.empty((n_vox, W.shape[1]), float)
    for gis, stack in groups_by_size.values():
        out[gis] = np.median(W[stack], axis=1)
    return out


def permute(W, rng, patients=None):
    """Shuffle loadings across contacts, globally or within patient."""
    if patients is None:
        return W[rng.permutation(len(W))]
    out = np.empty_like(W)
    for p in np.unique(patients):
        m = np.where(patients == p)[0]
        out[m] = W[rng.permutation(m)]
    return out


def bh_fdr(p, q=Q):
    """Benjamini-Hochberg: boolean mask of the voxels declared at level q."""
    m = len(p)
    order = np.argsort(p)
    thresh = np.arange(1, m + 1) / m * q
    below = p[order] <= thresh
    if not below.any():
        return np.zeros(m, bool), 0.0
    k = np.max(np.where(below)[0])
    cut = p[order][k]
    return p <= cut, float(cut)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cnmf/concat_hg/runs/20260818_112939")
    ap.add_argument("--step", type=float, default=2.0, help="grid spacing, mm")
    ap.add_argument("--cube", type=float, default=10.0, help="voxel edge, mm")
    ap.add_argument("--min-elec", type=int, default=3, dest="min_elec")
    ap.add_argument("--nperm", type=int, default=1000)
    a = ap.parse_args()

    run_dir = CLUST / a.run
    d, wcols, csv = load(run_dir)
    K = len(wcols)
    xyz = d[["x", "y", "z"]].to_numpy(float)
    W = d[wcols].to_numpy(float)
    pats = d["patient_id"].astype(str).to_numpy()
    n = len(d)
    print(f"  {csv.name}")
    print(f"  {n} localised contacts, {d['patient_id'].nunique()} patients, K={K}")
    print(f"  loadings row-normalised (row sums {W.sum(1).min():.3f}"
          f"-{W.sum(1).max():.3f})")

    vox = build_voxels(xyz, a.step, a.cube, a.min_elec)
    centres = np.array(list(vox.keys()), float)
    groups = list(vox.values())
    sizes = np.array([len(g) for g in groups])
    n_vox = len(groups)
    by_size = defaultdict(list)
    for gi, idx in enumerate(groups):
        by_size[len(idx)].append(gi)
    gbs = {s: (np.asarray(gis), np.stack([groups[gi] for gi in gis]))
           for s, gis in by_size.items()}
    print(f"  {n_vox} voxels with >= {a.min_elec} contacts "
          f"({a.cube:g} mm cube, {a.step:g} mm spacing), "
          f"median {int(np.median(sizes))} contacts, max {sizes.max()}")

    obs = medians_by_group(W, gbs)

    t0 = time.time()
    cnt_g = np.zeros_like(obs)
    cnt_w = np.zeros_like(obs)
    max_w = np.empty((a.nperm, K))
    for b in range(a.nperm):
        mg = medians_by_group(permute(W, np.random.default_rng(9000 + b)), gbs)
        mw = medians_by_group(
            permute(W, np.random.default_rng(50000 + b), pats), gbs)
        cnt_g += (mg >= obs)
        cnt_w += (mw >= obs)
        max_w[b] = mw.max(0)
        if (b + 1) % 100 == 0:
            print(f"    permutation {b + 1}/{a.nperm}  "
                  f"({time.time() - t0:.0f}s)", end="\r")
    print(" " * 50, end="\r")

    p_g = (cnt_g + 1) / (a.nperm + 1)
    p_w = (cnt_w + 1) / (a.nperm + 1)
    thr_w = np.percentile(max_w, 95, axis=0)

    sig_g = p_g < 0.05
    sig_w = p_w < 0.05
    sig_m = obs > thr_w[None, :]
    sig_f = np.zeros_like(sig_w)
    fdr_cut = np.zeros(K)
    for j in range(K):
        sig_f[:, j], fdr_cut[j] = bh_fdr(p_w[:, j])

    expected = int(0.05 * n_vox)
    yeo = d["yeo7_network"].map(YEO7).fillna("other").to_numpy()

    def contacts_in(mask_col, j):
        inside = np.zeros(n, bool)
        for gi in np.where(mask_col)[0]:
            inside[groups[gi]] = True
        return inside & (W[:, j] > np.median(W[:, j]))

    rows, keeps, keeps_cor = [], {}, {}
    for j in range(K):
        keep = contacts_in(sig_w[:, j], j)                  # the paper's test
        best = contacts_in(sig_f[:, j] | sig_m[:, j], j)    # what actually survives
        keeps[j], keeps_cor[j] = keep, best
        pk = (int(np.argmax(np.where(sig_w[:, j], obs[:, j], -np.inf)))
              if sig_w[:, j].any() else -1)
        peak = centres[pk] if pk >= 0 else np.array([np.nan] * 3)
        top = (pd.Series(yeo[best]).value_counts(normalize=True)
               if best.any() else pd.Series(dtype=float))
        # the medial wall is unassigned cortex, not a network - left in, it gets
        # reported as a component's anatomical home
        real = top.drop(labels=["medial wall"], errors="ignore")
        rows.append(dict(
            component=j,
            vox_sig_global=int(sig_g[:, j].sum()),
            vox_sig_within_patient=int(sig_w[:, j].sum()),
            vox_sig_fdr=int(sig_f[:, j].sum()),
            vox_sig_maxstat=int(sig_m[:, j].sum()),
            survives_correction=bool(sig_f[:, j].any() or sig_m[:, j].any()),
            n_contacts_uncorrected=int(keep.sum()),
            n_contacts_corrected=int(best.sum()),
            n_patients_uncorrected=int(len(set(pats[keep]))) if keep.any() else 0,
            peak_x=round(float(peak[0]), 1), peak_y=round(float(peak[1]), 1),
            peak_z=round(float(peak[2]), 1),
            peak_median_loading=(round(float(obs[pk, j]), 4) if pk >= 0 else None),
            maxstat_threshold=round(float(thr_w[j]), 4),
            top_network=(real.index[0] if len(real) else None),
            top_network_frac=(round(float(real.iloc[0]), 3) if len(real) else None),
            medial_wall_frac=(round(float(top.get("medial wall", 0.0)), 3)
                              if len(top) else None)))
    summ = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT / "component_anatomy_summary.csv", index=False)

    vdf = pd.DataFrame(centres, columns=["x", "y", "z"])
    vdf["n_contacts"] = sizes
    for j in range(K):
        vdf[f"median_w{j}"] = obs[:, j]
        vdf[f"p_global_w{j}"] = p_g[:, j]
        vdf[f"p_within_patient_w{j}"] = p_w[:, j]
        vdf[f"sig_fdr_w{j}"] = sig_f[:, j]
        vdf[f"sig_maxstat_w{j}"] = sig_m[:, j]
    # The full map is 14927 x 39 and 7.6 MB as plain text, over the repo's 5 MB
    # per-file limit. Rounded and gzipped it is 0.33 MB and still lets FDR be
    # re-derived at another q without re-running the permutations. The uncompressed
    # file next to it holds only the voxels that actually passed something.
    any_sig = np.zeros(len(vdf), bool)
    for j in range(K):
        any_sig |= sig_f[:, j] | sig_m[:, j]
    vdf.round(4).to_csv(OUT / "voxel_pmap.csv.gz", index=False, compression="gzip")
    vdf[any_sig].round(4).to_csv(OUT / "significant_voxels.csv", index=False)
    print(f"  {int(any_sig.sum())} voxels significant under at least one correction")

    meta = dict(run=a.run, source=csv.name, n_contacts=int(n), K=K,
                n_patients=int(d["patient_id"].nunique()), frame="fsaverage",
                grid_step_mm=a.step, cube_mm=a.cube, min_contacts=a.min_elec,
                n_voxels=n_vox, n_perm=a.nperm, fdr_q=Q,
                expected_by_chance=expected,
                p_floor=round(1.0 / (a.nperm + 1), 5),
                maxstat_threshold_within_patient=[float(v) for v in thr_w],
                n_survive_correction=int(summ["survives_correction"].sum()))
    (OUT / "component_anatomy_stats.json").write_text(json.dumps(meta, indent=2),
                                                      encoding="utf-8")

    print()
    print(summ[["component", "vox_sig_global", "vox_sig_within_patient",
                "vox_sig_fdr", "vox_sig_maxstat", "survives_correction",
                "peak_median_loading", "top_network"]].to_string(index=False))
    print(f"\n  {expected} voxels of {n_vox} would pass the uncorrected test on noise "
          f"alone (5%).")
    print(f"  {meta['n_survive_correction']} of {K} components survive a correction.")
    draw(xyz, W, keeps, keeps_cor, summ, yeo, meta,
         OUT / "A1_component_anatomy.png")
    print(f"  -> {OUT / 'A1_component_anatomy.png'}")
    return 0


# ==============================================================================
def draw(xyz, W, keeps, keeps_cor, summ, yeo, meta, out_png):
    K = meta["K"]
    cmap = plt.get_cmap("turbo")
    cols = [cmap(0.06 + 0.88 * j / max(K - 1, 1)) for j in range(K)]

    fig = plt.figure(figsize=(14.0, 10.7), dpi=200)
    gs = GridSpec(3, K, height_ratios=[1.02, 1.02, 1.42],
                  hspace=0.26, wspace=0.08,
                  left=0.042, right=0.986, top=0.735, bottom=0.058)

    def panel(ax, ai, bi, keep, cor, col, lab):
        ax.scatter(xyz[:, ai], xyz[:, bi], s=2.6, c=GREY, lw=0, zorder=1)
        if keep.any():                       # uncorrected - present for every component
            ax.scatter(xyz[keep, ai], xyz[keep, bi], s=5.0, color=col, lw=0,
                       alpha=0.22, zorder=2)
        if cor.any():                        # what actually survives
            ax.scatter(xyz[cor, ai], xyz[cor, bi], s=11.0, color=col, lw=0, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ccd2d8")
        if lab:
            ax.set_ylabel(lab, fontsize=8, color=MUTED)

    for j in range(K):
        k = keeps[j]
        r = summ.iloc[j]
        kc = keeps_cor[j]
        ax0 = fig.add_subplot(gs[0, j])
        panel(ax0, 0, 1, k, kc, cols[j], "axial   (x, y)" if j == 0 else None)
        mark = "  ✓" if r["survives_correction"] else ""
        ax0.set_title(f"c{j}{mark}", fontsize=10, pad=3, fontweight="bold",
                      color=cols[j] if r["survives_correction"] else MUTED)
        ax1 = fig.add_subplot(gs[1, j])
        panel(ax1, 1, 2, k, kc, cols[j], "sagittal   (y, z)" if j == 0 else None)
        note = (f"{int(r['vox_sig_fdr'])} FDR / {int(r['vox_sig_maxstat'])} max-stat\n"
                f"{int(r['n_contacts_corrected'])} contacts survive"
                if r["survives_correction"] else
                "nothing survives\nthe faint dots are all it has")
        ax1.set_xlabel(note, fontsize=7.2,
                       color=INK if r["survives_correction"] else MUTED, labelpad=3)

    # ---- how many voxels survive each threshold
    axb = fig.add_subplot(gs[2, :3])
    x = np.arange(K)
    axb.bar(x - 0.30, summ["vox_sig_global"], 0.20, color="#c9ced4",
            label="global shuffle, uncorrected  (the paper's test)")
    axb.bar(x - 0.10, summ["vox_sig_within_patient"], 0.20, color="#4a6fa5",
            label="within-patient shuffle, uncorrected")
    axb.bar(x + 0.10, summ["vox_sig_fdr"], 0.20, color="#e07b39",
            label=f"within-patient + BH-FDR q={meta['fdr_q']}")
    axb.bar(x + 0.30, summ["vox_sig_maxstat"], 0.20, color="#1b7837",
            label="within-patient + max-statistic (family-wise)")
    axb.axhline(meta["expected_by_chance"], color="#c1121f", ls="--", lw=1.1)
    axb.text(-0.45, meta["expected_by_chance"] * 1.05,
             f"{meta['expected_by_chance']} expected on noise alone (5%)",
             color="#c1121f", fontsize=7.4, ha="left")
    axb.set_xticks(x); axb.set_xticklabels([f"c{j}" for j in range(K)], fontsize=8.5)
    axb.set_ylabel("significant voxels", fontsize=8.5)
    axb.tick_params(labelsize=7.5, colors=MUTED)
    axb.set_ylim(0, float(summ["vox_sig_global"].max()) * 1.52)
    axb.legend(fontsize=6.9, frameon=False, loc="upper center", ncol=2,
               columnspacing=1.1, handlelength=1.5)
    axb.spines[["top", "right"]].set_visible(False)
    axb.set_title("Voxels surviving each null and each correction",
                  fontsize=10, loc="left", color=INK, pad=6)

    # ---- Yeo-7 composition of the surviving contacts
    axh = fig.add_subplot(gs[2, 3:])
    nets = [v for v in YEO7.values() if v != "medial wall"] + ["medial wall"]
    M = np.zeros((K, len(nets)))
    for j in range(K):
        k = keeps_cor[j]                     # the surviving contacts, not the faint ones
        if k.any():
            vc = pd.Series(yeo[k]).value_counts(normalize=True)
            for i, nm in enumerate(nets):
                M[j, i] = vc.get(nm, 0.0)
    im = axh.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 0.01), aspect="auto")
    axh.set_xticks(range(len(nets)))
    axh.set_xticklabels(nets, rotation=38, ha="right", fontsize=7.4)
    axh.set_yticks(range(K))
    axh.set_yticklabels(
        [f"c{j} ✓  n={int(summ.iloc[j]['n_contacts_corrected'])}"
         if summ.iloc[j]["survives_correction"] else f"c{j}   -"
         for j in range(K)], fontsize=8)
    for j in range(K):
        for i in range(len(nets)):
            if M[j, i] >= 0.10:
                axh.text(i, j, f"{100 * M[j, i]:.0f}", ha="center", va="center",
                         fontsize=6.8,
                         color="white" if M[j, i] > 0.55 * M.max() else INK)
    axh.set_title("Yeo-7 composition of the SURVIVING contacts (%) - blank rows "
                  "survived nothing", fontsize=10, loc="left", color=INK, pad=6)
    axh.tick_params(length=0)
    plt.colorbar(im, ax=axh, fraction=0.03, pad=0.015).ax.tick_params(labelsize=6.5)

    # ---- header
    ns = meta["n_survive_correction"]
    fdr_c = [f"c{r.component}" for r in summ.itertuples() if r.vox_sig_fdr > 0]
    max_c = [f"c{r.component}" for r in summ.itertuples() if r.vox_sig_maxstat > 0]
    dead = [f"c{r.component}" for r in summ.itertuples()
            if not r.survives_correction]
    lo = int(min(summ["vox_sig_within_patient"].min(),
                 summ["vox_sig_global"].min()))
    hi = int(max(summ["vox_sig_within_patient"].max(),
                 summ["vox_sig_global"].max()))

    def lst(v):
        return ", ".join(v) if v else "none"

    # Hand-broken lines used to run past the canvas, and bbox_inches="tight" then
    # stretched the whole figure to fit them. Wrap to a fixed width instead.
    para = [
        f"The K={meta['K']} loadings are graded, not clustered: FIG C.7 showed the "
        f"argmax partition is not a separation (z = +1.0). So the ELECTRODE is dropped "
        f"as the unit of analysis and the VOXEL is used instead, following Castellucci "
        f"et al. (2026), who hit the same wall. For every {meta['cube_mm']:g} mm cube "
        f"holding {meta['min_contacts']}+ of the {meta['n_contacts']} localised "
        f"contacts: is the median loading higher than chance allows? Loadings stay "
        f"continuous throughout - nothing is partitioned.",

        f"{meta['n_voxels']} voxels, {meta['n_perm']} permutations, "
        f"{meta['grid_step_mm']:g} mm spacing, fsaverage space. Grey = all contacts; "
        f"FAINT = passes the uncorrected within-patient test; SOLID = survives a "
        f"correction. Contacts shown also load above that component's median.",

        f"Uncorrected, every component looks localised: {lo}-{hi} voxels pass against "
        f"{meta['expected_by_chance']} expected on noise alone (5%). That excess is "
        f"real, but across ~{meta['n_voxels'] // 1000},000 overlapping tests it cannot "
        f"localise anything by itself - the difficulty with the paper's test.",

        f"{ns} of {meta['K']} survive a correction: {lst(fdr_c)} under FDR, "
        f"{lst(max_c)} under the max-statistic. They disagree because max-stat rewards "
        f"one extreme peak while FDR rewards many moderate voxels. {lst(dead)} localise "
        f"nowhere. Read the anatomy with the n on each heatmap row - several rest on "
        f"very few contacts.",
    ]
    body = "\n".join(textwrap.fill(t, width=163) for t in para)

    fig.suptitle("Where each convex-NMF component lives - and whether that survives "
                 "a correction", x=0.042, y=0.982, ha="left", fontsize=15, color=INK)
    fig.text(0.042, 0.940, body, fontsize=8.2, color=MUTED, va="top",
             linespacing=1.55)

    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
