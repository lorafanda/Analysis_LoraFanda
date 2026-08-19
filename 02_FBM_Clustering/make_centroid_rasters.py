#!/usr/bin/env python3
"""
make_centroid_rasters.py - show what the cluster centroid averages away.

THIS SCRIPT OWNS ONE FOLDER: cluster_rasters/. It adds to cluster_centroids/,
which lf_centroids writes and which this script does not touch.

A centroid is a mean, and a mean cannot say how much its members agreed. Both
views here answer that question, each in the way its feature set allows:

  concat_hg    a sample IS one time course (900 = 3 conditions x 300 bins), so
               every sample gets a row: a true per-sample raster, sorted by how
               strongly it belongs, strongest at the top. A cluster that is one
               thing degrades smoothly downwards; one that is two things breaks
               visibly.

  concat_rawds a sample is 1350 features (15 bands x 3 conditions x 30 bins) and
               has no single time axis, so a per-sample raster is not available.
               This is the ordinary centroid heatmap, annotated: one dot per
               time x frequency bin, invisible where the cluster's electrodes
               agree and darkening to grey where they do not. A plain mean cannot
               separate "the whole cluster sits near zero here" from "the cluster
               is split here and the average cancels" - both come out white. The
               first keeps a clear bin, the second gets a dark dot.

               The dot sits ON the mean rather than tinting it, so the colour
               read off the scale bar is still the value. Its scale is pooled
               over EVERY run of the feature set, not per cluster and not per
               run: X_train is the same matrix whichever method partitioned it,
               so a dot means the same disagreement on convex NMF, k-means and
               Ward, which is the comparison anyone will actually make.

MEMBERSHIP STRENGTH IS NOT THE SAME QUANTITY ACROSS METHODS, and only the hg
raster is sorted by it. Convex NMF has a loading on the sample's own component;
k-means and Ward have only a silhouette, computed here since no run stores it per
sample. Both order samples sensibly but they are different measurements, so the
y-axis says which one it is and two runs should not be read as directly
comparable.

    python make_centroid_rasters.py --dry-run
    python make_centroid_rasters.py                 # the 6 concat runs
    python make_centroid_rasters.py --run <run dir>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
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
import lf_centroids as LC  # noqa: E402   (read only - conventions, not modified)

CLUST = ROOT / "outputs" / "clustering"

# The runs asked for: convex NMF, k-means and Ward, on both concat feature sets.
METHODS = ("cnmf", "kmeans", "hierarchical")
FEATURE_SETS = ("concat_hg", "concat_rawds", "concat_hg_all")

COND_NAMES = ("audio", "picture", "reading")
# The disagreement dot: dark grey, fading in from fully transparent. It sits ON the
# centroid rather than tinting it, so the colour read off the scale bar is still the
# mean. DOT_FILL is the fraction of a bin the dot spans at its widest.
DOT_RGB = (0.20, 0.20, 0.20)
DOT_MAX_ALPHA = 0.92
DOT_FILL = 0.62
CMAP = "bwr"                   # the project's diverging map for signed dB


def newest_per_track(a):
    """One run per (method, feature_set): the newest, unless --run names one."""
    idx = json.loads((CLUST / "index.json").read_text(encoding="utf-8"))
    runs = idx["runs"] if isinstance(idx, dict) else idx
    best = {}
    for r in runs:
        if r["method"] not in METHODS or r["feature_set"] not in FEATURE_SETS:
            continue
        rd = CLUST / r["method"] / r["feature_set"] / "runs" / r["run_id"]
        if not rd.is_dir() or not (rd / "X_train.npy").exists():
            continue
        key = (r["method"], r["feature_set"])
        if key not in best or r["run_id"] > best[key][0]:
            best[key] = (r["run_id"], rd)
    out = [(f"{m}/{f}", rd) for (m, f), (_, rd) in sorted(best.items())]
    if a.run:
        want = Path(a.run).resolve()
        out = [(t, rd) for t, rd in out if rd.resolve() == want]
        if not out:                      # allow a run outside the default six
            rd = Path(a.run)
            out = [(f"{rd.parent.parent.parent.name}/{rd.parent.parent.name}", rd)]
    return out


def read_run(rd: Path):
    """(X, labels, cluster ids, feature_set, strength, strength_name)."""
    X = np.load(rd / "X_train.npy")
    lab_df = pd.read_csv(rd / "labels.csv")
    ccol = next(c for c in lab_df.columns
                if c.startswith("cluster_") and not c.endswith("_ranked"))
    labels = pd.to_numeric(lab_df[ccol], errors="coerce").to_numpy()
    feature_set = rd.parent.parent.name
    method = rd.parent.parent.parent.name

    gpath = rd / "G_loadings.npy"
    if gpath.exists():
        G = np.load(gpath)
        # Row-normalised, then each sample's weight on the component it was
        # assigned to. Raw G is not comparable between samples: a loud electrode
        # loads higher on everything.
        Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
        ok = np.isfinite(labels) & (labels >= 0) & (labels < Gn.shape[1])
        strength = np.full(len(labels), np.nan)
        idx = labels[ok].astype(int)
        strength[ok] = Gn[np.where(ok)[0], idx]
        name = "loading on this component"
    else:
        from sklearn.metrics import silhouette_samples
        good = np.isfinite(labels)
        strength = np.full(len(labels), np.nan)
        if len(np.unique(labels[good])) > 1:
            strength[good] = silhouette_samples(X[good], labels[good].astype(int))
        name = "silhouette"

    ids = sorted({int(v) for v in labels[np.isfinite(labels)] if int(v) >= 0})
    return X, labels, ids, feature_set, method, strength, name


def grid_for(rd: Path, feature_set: str, n_features: int):
    """(n_bands, n_cond*n_time) for a heatmap feature set, from the schema."""
    fs = rd / "feature_schema.json"
    if not fs.exists():
        return None
    names = json.loads(fs.read_text(encoding="utf-8")).get("feature_names")
    if not names:
        return None
    bands, conds, times = [], [], set()
    for f in names:
        c, b, t = f.split("|")
        if b not in bands:
            bands.append(b)
        if c not in conds:
            conds.append(c)
        times.add(t)
    if len(bands) * len(conds) * len(times) != n_features:
        return None
    return bands, conds, sorted(times)


def _decorate_x(ax, n_x, n_blocks, *, heatmap, labels_on):
    """Block seams, the GO cue dash per condition, and condition tick labels.

    Same convention as lf_centroids: each block is one condition warped to 0-100%
    of its trial and 50% IS the GO cue, so the dash belongs at every block's
    midpoint rather than once in the middle of the row.
    """
    per = n_x / max(n_blocks, 1)
    for b in range(1, max(n_blocks, 1)):
        ax.axvline(b * per - (0.5 if heatmap else 0), color="k", lw=1.0, zorder=5)
    for b in range(max(n_blocks, 1)):
        ax.axvline((b + 0.5) * per - (0.5 if heatmap else 0),
                   color="#3a3a3a", lw=0.8, ls=(0, (4, 3)), zorder=6, alpha=0.75)
    if labels_on and n_blocks > 1:
        ax.set_xticks([(b + 0.5) * per for b in range(n_blocks)])
        ax.set_xticklabels(list(COND_NAMES)[:n_blocks], fontsize=7)
    else:
        ax.set_xticks([])


def raster_line(out_png, Xc, strength_c, vlim, n_blocks, cid, sname):
    """concat_hg: one row per sample, strongest membership at the top."""
    order = np.argsort(-strength_c)
    M = Xc[order]
    n, T = M.shape
    fig = plt.figure(figsize=(4.6, 3.1), dpi=220)
    gs = GridSpec(1, 2, width_ratios=[40, 1], wspace=0.04,
                  left=0.13, right=0.9, top=0.86, bottom=0.17)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(M, aspect="auto", cmap=CMAP, vmin=-vlim, vmax=vlim,
                   interpolation="nearest", origin="upper")
    _decorate_x(ax, T, n_blocks, heatmap=True, labels_on=True)
    ax.set_ylabel(f"{n} samples, sorted by\n{sname} (strongest at top)", fontsize=7)
    ax.set_yticks([0, n - 1])
    ax.set_yticklabels([f"{np.nanmax(strength_c):.2f}", f"{np.nanmin(strength_c):.2f}"],
                       fontsize=6.5)
    ax.tick_params(length=2, width=0.6, pad=1.5, colors="#68727d")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#c8cfd6")
    ax.set_title(f"cluster {cid} — every sample", fontsize=8.5, loc="left", pad=4)
    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("HG (dB)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    fig.text(0.13, 0.045, "dashed = GO cue (50% of each condition)  ·  "
                          "solid = condition boundary",
             fontsize=5.8, color="#9aa3ab")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return n


def sd_scale(feature_set, targets):
    """Low/high within-cluster SD, pooled over EVERY run of this feature set.

    Not per cluster, and not per run either. X_train is the same matrix for a given
    feature set whichever method partitioned it, so pooling across the runs makes a
    dot mean the same amount of disagreement whether you are looking at convex NMF,
    k-means or Ward - which is the comparison anyone will actually make. Scaling per
    cluster would do the opposite: every cluster would show its own worst bin as
    fully dark and a tight cluster would look exactly as uncertain as a loose one.

    The 10th and 90th percentiles rather than min/max, so one extreme bin cannot
    flatten the scale for everything else.
    """
    sds = []
    for tag, rd in targets:
        if rd.parent.parent.name != feature_set or not (rd / "X_train.npy").exists():
            continue
        X = np.load(rd / "X_train.npy")
        lab = pd.read_csv(rd / "labels.csv")
        ccol = next((c for c in lab.columns
                     if c.startswith("cluster_") and not c.endswith("_ranked")), None)
        if ccol is None:
            continue
        L = pd.to_numeric(lab[ccol], errors="coerce").to_numpy()
        for cid in sorted({int(v) for v in L[np.isfinite(L)] if int(v) >= 0}):
            Xc = X[L == cid]
            if len(Xc) > 1:
                sds.append(Xc.std(axis=0, ddof=1))
    if not sds:
        return 0.0, 1.0
    A = np.concatenate(sds)
    lo, hi = float(np.percentile(A, 10)), float(np.percentile(A, 90))
    return lo, (hi if hi > lo else lo + 1e-6)


def _dot_size(fig, ax, ncols, nrows):
    """Marker area in points^2, from the axes as actually laid out.

    Hard-coding a size breaks whenever the figure or the grid changes: concat_rawds
    is 90 columns wide, so a bin is about 3 points across and a dot sized for a
    10-column plot would cover its neighbours.
    """
    fig.canvas.draw()
    bb = ax.get_window_extent()
    col_pt = (bb.width * 72.0 / fig.dpi) / max(ncols, 1)
    row_pt = (bb.height * 72.0 / fig.dpi) / max(nrows, 1)
    return (DOT_FILL * min(col_pt, row_pt)) ** 2


def centroid_dotted(out_png, Xc, vlim, grid, cid, sd_lo, sd_hi):
    """concat_rawds: the plain centroid heatmap, with a dot marking disagreement.

    The heatmap is the cluster mean exactly as it always was. Over it sits one dot
    per time x frequency bin, invisible where the cluster's electrodes agree and
    darkening to grey where they do not - so the mean is never altered, only
    annotated. An earlier version tinted the bins themselves, which changed the
    colour being read off the scale bar; a dot leaves the value alone.

    The SD is the spread ACROSS ELECTRODES, the same quantity the HG centroid draws
    as its +/-1 SD band, on a scale pooled across every run of this feature set.
    """
    bands, conds, times = grid
    nb, nc, nt = len(bands), len(conds), len(times)
    n = len(Xc)
    mean = Xc.mean(axis=0).reshape(nb, nc * nt)
    sd = (Xc.std(axis=0, ddof=1) if n > 1
          else np.zeros(Xc.shape[1])).reshape(nb, nc * nt)

    fig = plt.figure(figsize=(5.2, 2.75), dpi=220)
    gs = GridSpec(2, 2, width_ratios=[40, 1.4], height_ratios=[13, 0.8],
                  wspace=0.05, hspace=0.30, left=0.15, right=0.89,
                  top=0.87, bottom=0.16)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mean, aspect="auto", cmap=CMAP, vmin=-vlim, vmax=vlim,
                   origin="lower", interpolation="nearest")

    yy, xx = np.mgrid[0:nb, 0:(nc * nt)]
    a = np.clip((sd - sd_lo) / (sd_hi - sd_lo), 0, 1) * DOT_MAX_ALPHA
    rgba = np.zeros((sd.size, 4))
    rgba[:, :3] = DOT_RGB
    rgba[:, 3] = a.ravel()
    ax.scatter(xx.ravel(), yy.ravel(), s=_dot_size(fig, ax, nc * nt, nb),
               c=rgba, marker="o", linewidths=0, zorder=4)

    _decorate_x(ax, nc * nt, nc, heatmap=True, labels_on=True)
    tick_at = sorted({0, nb // 3, 2 * nb // 3, nb - 1})
    ax.set_yticks(tick_at)
    ax.set_yticklabels([bands[t] for t in tick_at], fontsize=6)
    ax.tick_params(length=2, width=0.6, pad=1.5, colors="#68727d")
    ax.set_xlim(-0.5, nc * nt - 0.5)
    ax.set_ylim(-0.5, nb - 0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#c8cfd6")
    ax.set_title(f"cluster {cid} centroid — {n} electrodes, dots mark disagreement",
                 fontsize=8.5, loc="left", pad=4)

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("power (dB)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)

    # The legend is the dot itself fading in, on white, so the reader compares like
    # with like rather than translating a colour ramp into a dot.
    axl = fig.add_subplot(gs[1, 0])
    steps = 26
    la = np.linspace(0, 1, steps) * DOT_MAX_ALPHA
    lrgba = np.zeros((steps, 4))
    lrgba[:, :3] = DOT_RGB
    lrgba[:, 3] = la
    axl.scatter(np.arange(steps), np.zeros(steps),
                s=_dot_size(fig, axl, steps, 1) * 0.55,
                c=lrgba, marker="o", linewidths=0)
    axl.set_xlim(-0.5, steps - 0.5)
    axl.set_ylim(-0.5, 0.5)
    axl.set_yticks([])
    axl.set_xticks([0, steps - 1])
    axl.set_xticklabels([f"{sd_lo:.2f}", f"{sd_hi:.2f}"], fontsize=6)
    axl.tick_params(length=2, width=0.5, pad=1.5, colors="#68727d")
    axl.set_xlabel("within-cluster SD across electrodes (dB), one scale for every "
                   "cluster and run — darker = less agreement",
                   fontsize=6, labelpad=1.5, color="#68727d")
    for sp in axl.spines.values():
        sp.set_visible(False)

    fig.text(0.15, 0.012, f"{nb} bands, low frequency at bottom  ·  "
                          f"dashed = GO cue  ·  solid = condition boundary",
             fontsize=5.4, color="#9aa3ab")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vlim", type=float, default=None,
                    help="symmetric colour limit; default is the run's 99th "
                         "percentile of |x|, rounded up to 0.5")
    a = ap.parse_args()

    targets = newest_per_track(a)
    # The dot scale is pooled over every run of a feature set, so it must not change
    # depending on which subset --run happens to select. Resolve the full set once.
    import copy
    all_targets = newest_per_track(copy.copy(type("A", (), dict(
        run=None, method=None, feature_set=None))()))
    if not targets:
        print("  no run matched", file=sys.stderr)
        return 1
    print(f"  {len(targets)} run(s):")
    for tag, rd in targets:
        print(f"    {tag}/{rd.name}")
    if a.dry_run:
        print("  (dry run)")
        return 0

    for tag, rd in targets:
        X, labels, ids, feature_set, method, strength, sname = read_run(rd)
        line = LC._is_line_feature_set(feature_set)
        n_blocks = LC._n_condition_blocks(feature_set)
        # ONE colour limit for every cluster of a run, so the panels compare. Taken
        # from the run's own distribution rather than fixed: |x| runs to 13 dB but
        # the 99th percentile is near 3, and a fixed +/-6.5 would wash every raster
        # out to pale blue-white.
        vlim = a.vlim or float(np.ceil(np.nanpercentile(np.abs(X), 99) * 2) / 2)
        grid = None if line else grid_for(rd, feature_set, X.shape[1])
        if not line and grid is None:
            print(f"\n  {tag}: no usable feature grid, skipped")
            continue
        sd_lo, sd_hi = (0.0, 1.0) if line else sd_scale(feature_set, all_targets)
        detail = ("sorted by %s" % sname if line
                  else "dots over SD %.2f-%.2f dB, pooled across this feature set"
                       % (sd_lo, sd_hi))
        out_dir = rd / "cluster_rasters"
        print(f"\n  {tag}/{rd.name}: {X.shape[0]} samples x {X.shape[1]} features, "
              f"K={len(ids)}, colour +/-{vlim:g} dB, {detail}")

        for cid in ids:
            m = labels == cid
            Xc, sc = X[m], strength[m]
            if not np.isfinite(sc).all():
                sc = np.nan_to_num(sc, nan=float(np.nanmin(sc)) if np.isfinite(sc).any()
                                    else 0.0)
            out = out_dir / f"cluster_{cid:02d}.png"
            if line:
                n = raster_line(out, Xc, sc, vlim, n_blocks, cid, sname)
                print(f"    cluster {cid}: {n} samples")
            else:
                nn = centroid_dotted(out, Xc, vlim, grid, cid, sd_lo, sd_hi)
                print(f"    cluster {cid}: {nn} electrodes")
        print(f"    -> {out_dir}")

    print("\n  cluster_rasters/ written. cluster_centroids/ is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
