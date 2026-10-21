#!/usr/bin/env python3
"""make_coverage_figures.py - is the generalisation test reading coverage or response type?

    python make_coverage_figures.py                 both figures, cached correlation
    python make_coverage_figures.py --recompute     refit leave-one-patient-out first (slow)
    python make_coverage_figures.py --only brain
    python make_coverage_figures.py --contacts all  the whole implant, not just the cohort

WRITES, to outputs/clustering/paper_figures/
    FIG_coverage_density.png              how many PATIENTS sample each point of cortex
    FIG_coverage_vs_generalization.png    that, against how well each patient is placed
    coverage_vs_generalization.csv        the per-patient table behind the second one

WHY PATIENTS AND NOT ELECTRODES. One shaft puts a dozen contacts in the same place, which
says nothing about whether a finding there would survive that patient leaving. The question
a leave-one-patient-out test has to answer is how many DIFFERENT patients sample a point.

WHICH ELECTRODES. By default the 1607 gated electrodes - the pool the test actually uses.
The whole implant (--contacts all) gives a rosier number because there are more neighbours
in it; that one describes the implant, not the test. Both are printed either way.

THE CORRELATION IS SLOW because it refits convex NMF without each patient in turn, for
every seed. It is cached in coverage_vs_generalization.csv; --recompute forces it.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

MESH = ROOT / "outputs" / "250_recon" / "fsaverage" / "meshes"
COORDS = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coords" /
          "ALL_PATIENTS_contacts_fsaverage.csv")
OUT = ROOT / "outputs" / "clustering" / "paper_figures"
CSV = OUT / "coverage_vs_generalization.csv"

FS, K, METHOD, SEEDS = "concat_bands15", 7, "cnmf", (42, 0, 7)
RADII = (6.0, 10.0)
VIEWS = (("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral"))
SHOT = (1000, 820)
RED = LinearSegmentedColormap.from_list(
    "cov", ["#ffffff", "#ffe9d6", "#fdbb84", "#fc8d59", "#e34a33", "#8b0000"])


# ── the cohort and its coordinates ───────────────────────────────────────────────────────
def newest_run(method, fset):
    runs = sorted(glob.glob(str(ROOT / "outputs" / "clustering" / method / fset / "runs" / "*")))
    if not runs:
        raise SystemExit(f"no runs for {method}/{fset} - fit it first (240 / 242)")
    return Path(runs[-1])


def key(p, e):
    return f"{p}|{str(e).replace('_', '').replace('-', '').upper()}"


def load_cohort(which):
    """(xyz, patient) for the electrodes to count, plus the gated labels table."""
    run = newest_run("kmeans", FS)
    lab = pd.read_csv(run / "labels.csv")
    co = pd.read_csv(COORDS)
    co["key"] = [key(p, n) for p, n in zip(co["patient"], co["name"])]
    cmap = co.drop_duplicates("key").set_index("key")

    gated_keys = [key(p, e) for p, e in zip(lab["patient_id"], lab["electrode"])]
    gated_xyz = cmap.reindex(gated_keys)[["x", "y", "z"]].to_numpy(float)
    gated_pat = lab["patient_id"].astype(str).to_numpy()
    ok = np.isfinite(gated_xyz).all(1)
    print(f"gated electrodes with coordinates: {ok.sum()} of {len(ok)}")

    if which == "gated":
        return gated_xyz[ok], gated_pat[ok], lab, run
    keep = set(gated_pat)
    all_ = co[co.patient.astype(str).isin(keep)].dropna(subset=["x", "y", "z"])
    print(f"whole implant of the same patients: {len(all_)} contacts")
    return (all_[["x", "y", "z"]].to_numpy(float), all_.patient.astype(str).to_numpy(),
            lab, run)


def n_other_patients(xyz, pat, r):
    """Per electrode: how many OTHER patients have a contact within r mm."""
    nb = cKDTree(xyz).query_ball_point(xyz, r)
    return np.array([len(set(pat[i]) - {p}) for i, p in zip(nb, pat)])


# ── figure 1 · the density on the cortex ────────────────────────────────────────────────
def surface(hemi):
    g = nib.load(str(MESH / f"fsaverage_{hemi}.gii"))
    import pyvista as pv
    v = np.asarray(g.darrays[0].data, float)
    f = np.asarray(g.darrays[1].data, int)
    return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel()), v


def n_patients_per_vertex(verts, xyz, pat, r):
    near = cKDTree(xyz).query_ball_point(verts, r)
    return np.array([len(set(pat[i])) for i in near], float)


def render(mesh, n, hemi, view, vmax):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=SHOT)
    pl.set_background("white")
    m = mesh.copy()
    m["n"] = n
    pl.add_mesh(m, scalars="n", cmap=RED, clim=(0.5, vmax), below_color="black",
                show_scalar_bar=False, smooth_shading=True, ambient=0.35, diffuse=0.65,
                specular=0.05)
    sign = -1 if (hemi == "lh") == (view == "lateral") else 1
    pl.camera_position = [(sign * 400, 0, 0), (0, 0, 0), (0, 0, 1)]
    # fit the WHOLE hemisphere in frame: zooming in crops the frontal and occipital poles,
    # which are exactly the thinly covered parts this figure is about
    pl.reset_camera()
    img = np.asarray(pl.screenshot(return_img=True))
    pl.close()
    return img


def fig_density(xyz, pat, alone):
    data, vmax = {}, 0.0
    for hemi in ("lh", "rh"):
        mesh, verts = surface(hemi)
        for r in RADII:
            n = n_patients_per_vertex(verts, xyz, pat, r)
            data[(hemi, r)] = (mesh, n)
            vmax = max(vmax, float(np.percentile(n[n > 0], 99)))
    print(f"colour scale capped at the 99th percentile: {vmax:.0f} patients")

    fig, axes = plt.subplots(len(RADII), len(VIEWS), figsize=(17, 8))
    for ri, r in enumerate(RADII):
        for ci, (hemi, view) in enumerate(VIEWS):
            mesh, n = data[(hemi, r)]
            ax = axes[ri, ci]
            ax.imshow(render(mesh, n, hemi, view, vmax))
            ax.axis("off")
            if ri == 0:
                ax.set_title(f"{'left' if hemi == 'lh' else 'right'} · {view}",
                             fontsize=11.5, pad=4)
            if ci == 0:
                allv = np.concatenate([data[(h, r)][1] for h in ("lh", "rh")])
                ax.text(-0.02, 0.5,
                        f"r = {r:.0f} mm\n{100*(allv == 0).mean():.0f}% of cortex unsampled",
                        transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=11, linespacing=1.7)

    sm = plt.cm.ScalarMappable(cmap=RED, norm=plt.Normalize(0.5, vmax))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.030, pad=0.03,
                      aspect=55)
    cb.set_label("number of PATIENTS with a contact within r mm      (black = none)",
                 fontsize=10.5)
    fig.suptitle(f"How many patients sample each point of cortex  ·  "
                 f"{len(set(pat))}-patient cohort", fontsize=14, y=0.985)
    fig.text(0.5, 0.942,
             "Of the electrodes that go into the clustering, "
             + ", ".join(f"{100*alone[r]:.0f}% have no OTHER patient within {r:.0f} mm"
                         for r in RADII)
             + " — there, a leave-one-patient-out test is extrapolation, not generalisation.",
             ha="center", fontsize=10, color="#444")
    p = OUT / "FIG_coverage_density.png"
    fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {p}")


# ── the per-patient table ────────────────────────────────────────────────────────────────
def build_table(lab, run):
    spec = importlib.util.spec_from_file_location("mcs", ROOT / "make_cluster_statistics.py")
    mcs = importlib.util.module_from_spec(spec)
    sys.modules["mcs"] = mcs
    spec.loader.exec_module(mcs)

    X = np.load(run / "X_train.npy").astype(float)
    pats = lab["patient_id"].astype(str).to_numpy()
    co = pd.read_csv(COORDS)
    co["key"] = [key(p, n) for p, n in zip(co["patient"], co["name"])]
    cmap = co.drop_duplicates("key").set_index("key")
    xyz = cmap.reindex([key(p, e) for p, e in
                        zip(lab["patient_id"], lab["electrode"])])[["x", "y", "z"]].to_numpy(float)
    ok = np.isfinite(xyz).all(1)
    cov = {r: n_other_patients(xyz[ok], pats[ok], r) for r in RADII}

    print(f"leave-one-patient-out, {FS} K={K} {METHOD}, seeds {SEEDS} - this is the slow "
          f"part", flush=True)
    d = mcs.lopo_predict(X, pats, K, METHOD, seed=SEEDS)
    per = d.groupby("group")[["acc", "chance", "lift"]].mean()

    rows = []
    for p in sorted(set(pats)):
        sel = pats[ok] == p
        r = dict(patient=p, n_electrodes=int((pats == p).sum()))
        for R in RADII:
            r[f"shared_r{int(R)}"] = float(cov[R][sel].mean())
            r[f"alone_r{int(R)}"] = float((cov[R][sel] == 0).mean())
        r["placed"] = float(per.loc[p, "acc"])
        r["chance"] = float(per.loc[p, "chance"])
        r["lift"] = float(per.loc[p, "lift"])
        rows.append(r)
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(CSV, index=False)
    print(f"wrote -> {CSV}")
    return T


def partial_spearman(T, x, y, z):
    """Spearman of x and y with z held constant - the two effects are separable only
    this way, because electrode count and coverage are different things."""
    X, Y, Z = (rankdata(T[c]) for c in (x, y, z))
    rx = X - np.polyval(np.polyfit(Z, X, 1), Z)
    ry = Y - np.polyval(np.polyfit(Z, Y, 1), Z)
    return spearmanr(rx, ry)


# ── figure 2 · coverage against generalisation ──────────────────────────────────────────
def fig_corr(T):
    panels = [("shared_r6", "other patients within 6 mm of this patient's electrodes\n"
                            "(mean over its electrodes)", "n_electrodes"),
              ("n_electrodes", "electrodes this patient contributes", "shared_r6")]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4))
    for ax, (xc, xlabel, ctrl) in zip(axes, panels):
        x, y = T[xc].to_numpy(float), 100 * T["placed"].to_numpy(float)
        ax.scatter(x, y, s=46, c="#2c6fbb", zorder=3, edgecolor="white", linewidth=0.8)
        for xi, yi, name in zip(x, y, T["patient"]):
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(5, 4),
                        fontsize=7.4, color="#333")
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#c1121f", lw=1.5, ls="--", zorder=2)
        rs, ps = spearmanr(x, T["placed"])
        rp, pp = partial_spearman(T, xc, "placed", ctrl)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("placed where the full fit put them (%)", fontsize=10)
        ax.set_title(f"Spearman rho {rs:+.2f}  (p = {ps:.3f})\n"
                     f"holding {ctrl} constant: rho {rp:+.2f}  (p = {pp:.3f})",
                     fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, zorder=0)
    fig.suptitle("Does a patient's coverage explain whether it generalises?  "
                 f"{len(T)} patients, {FS}, K = {K}", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / "FIG_coverage_vs_generalization.png"
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--only", choices=["brain", "corr", "both"], default="both")
    ap.add_argument("--recompute", action="store_true",
                    help="refit leave-one-patient-out instead of reading the cached CSV")
    ap.add_argument("--contacts", choices=["gated", "all"], default="gated",
                    help="which electrodes the density map counts (default: the clustered set)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    xyz, pat, lab, run = load_cohort(a.contacts)
    alone = {}
    for r in RADII:
        n = n_other_patients(xyz, pat, r)
        alone[r] = float((n == 0).mean())
        print(f"r = {r:4.1f} mm   other patients per electrode: median "
              f"{np.median(n):.0f}, mean {n.mean():.2f}   alone: {100*alone[r]:.1f}%")

    if a.only in ("brain", "both"):
        fig_density(xyz, pat, alone)

    if a.only in ("corr", "both"):
        if CSV.exists() and not a.recompute:
            T = pd.read_csv(CSV)
            print(f"read {len(T)} patients from {CSV.name}  (--recompute to refit)")
        else:
            T = build_table(lab, run)
        fig_corr(T)
        print("\nper-patient, worst coverage first:")
        cols = ["patient", "n_electrodes", "shared_r6", "alone_r6", "placed", "lift"]
        print(T.sort_values("shared_r6")[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
