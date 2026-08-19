#!/usr/bin/env python3
"""
make_centroid_rasters.py - show what the cluster centroid averages away.

THIS SCRIPT OWNS ONE FOLDER: cluster_rasters/. It adds to cluster_centroids/,
which lf_centroids writes and which this script does not touch.

The centroid chip is a mean with a +/-1 SD band. An SD band cannot distinguish a
tight family of similar responses from two different populations averaged
together - both give a wide band. These rasters answer that: every sample in the
cluster, sorted by how strongly it belongs, strongest at the top. If the cluster
is one thing the picture degrades smoothly downwards; if it is two things the
break is visible.

The two feature sets get the view each can actually support:

  concat_hg    a sample IS one time course (900 = 3 conditions x 300 bins), so
               every sample is one row. A true per-sample raster.

  concat_rawds a sample is 1350 features (15 bands x 3 conditions x 30 bins) and
               has no single time axis. Samples are binned by membership strength
               instead, and each bin is drawn as the full band x time heatmap -
               keeping the 2-D structure that is the whole reason to use rawds,
               and still showing whether the response degrades or flips as
               membership weakens.

MEMBERSHIP STRENGTH IS NOT THE SAME QUANTITY ACROSS METHODS. Convex NMF has a
loading on the sample's own component; k-means and Ward have only a silhouette,
computed here since no run stores it per sample. Both order samples sensibly but
they are different measurements, so the y-axis is labelled with which one it is
and two runs' rasters should not be read as directly comparable.

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
FEATURE_SETS = ("concat_hg", "concat_rawds")

COND_NAMES = ("audio", "picture", "reading")
N_BINS = 10                    # deciles of membership strength for the rawds view
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


def raster_bins(out_png, Xc, strength_c, vlim, grid, cid, sname):
    """concat_rawds: samples binned by membership, each bin a band x time heatmap."""
    bands, conds, times = grid
    nb, nc, nt = len(bands), len(conds), len(times)
    order = np.argsort(-strength_c)
    n = len(order)
    nbin = int(min(N_BINS, n))
    chunks = np.array_split(order, nbin)

    fig = plt.figure(figsize=(5.4, 0.62 * nbin + 1.15), dpi=220)
    # Three columns, the middle one empty: it reserves room for the per-bin
    # annotation, which otherwise runs straight into the colorbar.
    gs = GridSpec(nbin, 3, width_ratios=[30, 7, 1.1], wspace=0.06, hspace=0.10,
                  left=0.16, right=0.94, top=0.90, bottom=0.10)
    # Frequency IS the y axis here, so it takes the ticks and the bin's membership
    # range moves to the right as an annotation. Ticked on the bottom panel only:
    # every panel shares the axis, and 15 labels x 10 panels is unreadable.
    tick_at = sorted({0, nb // 3, 2 * nb // 3, nb - 1})
    im = None
    for r, ch in enumerate(chunks):
        ax = fig.add_subplot(gs[r, 0])
        img = Xc[ch].mean(axis=0).reshape(nb, nc * nt)
        im = ax.imshow(img, aspect="auto", cmap=CMAP, vmin=-vlim, vmax=vlim,
                       origin="lower", interpolation="nearest")
        _decorate_x(ax, nc * nt, nc, heatmap=True, labels_on=(r == nbin - 1))
        s = strength_c[ch]
        ax.text(1.012, 0.5, "%.2f-%.2f\nn=%d" % (s.max(), s.min(), len(ch)),
                transform=ax.transAxes, fontsize=5.4, color="#68727d",
                ha="left", va="center")
        if r == nbin - 1:
            ax.set_yticks(tick_at)
            ax.set_yticklabels([bands[t] for t in tick_at], fontsize=5.4)
            ax.tick_params(length=2, width=0.5, pad=1.5, colors="#68727d")
        else:
            ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#c8cfd6")
        if r == 0:
            ax.set_title(f"cluster {cid} — {n} samples in {nbin} bins of {sname}, "
                         f"strongest at top", fontsize=8, loc="left", pad=5)
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("power (dB)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    fig.text(0.16, 0.035,
             f"each panel: {nb} bands, low frequency at bottom  ·  dashed = GO cue"
             f"  ·  solid = condition boundary  ·  right = membership range and n",
             fontsize=5.4, color="#9aa3ab")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return nbin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vlim", type=float, default=None,
                    help="symmetric colour limit; default is the run's 99th "
                         "percentile of |x|, rounded up to 0.5")
    a = ap.parse_args()

    targets = newest_per_track(a)
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
        out_dir = rd / "cluster_rasters"
        print(f"\n  {tag}/{rd.name}: {X.shape[0]} samples x {X.shape[1]} features, "
              f"K={len(ids)}, colour +/-{vlim:g} dB, sorted by {sname}")

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
                nbin = raster_bins(out, Xc, sc, vlim, grid, cid, sname)
                print(f"    cluster {cid}: {len(Xc)} samples in {nbin} bins")
        print(f"    -> {out_dir}")

    print("\n  cluster_rasters/ written. cluster_centroids/ is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
