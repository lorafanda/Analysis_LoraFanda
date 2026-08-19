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
               This is the ordinary centroid heatmap with the agreement written
               into it instead: each time x frequency bin keeps its colour where
               the cluster's electrodes agree and fades toward grey where they
               do not. A plain mean cannot separate "the whole cluster sits near
               zero here" from "the cluster is split here and the average
               cancels" - both come out white. With the tint the first stays
               white and the second turns grey.

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
FEATURE_SETS = ("concat_hg", "concat_rawds")

COND_NAMES = ("audio", "picture", "reading")
# Grey the rawds centroid fades toward where its electrodes disagree, and how far
# it is allowed to go. Short of 1.0 so the sign of a bin is never lost entirely.
TINT_RGB = (0.60, 0.60, 0.60)
TINT_MAX = 0.85
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


def sd_scale(X, labels, ids):
    """The run's own low/high within-cluster SD, used to scale every cluster's tint.

    Taken across all clusters so the grey means the same thing on each of them: a
    tight cluster has to look tighter than a loose one, which it cannot do if each
    is stretched over its own range. The 10th and 90th percentiles rather than
    min/max, so one extreme bin cannot flatten the whole scale.
    """
    sds = []
    for cid in ids:
        Xc = X[labels == cid]
        if len(Xc) > 1:
            sds.append(Xc.std(axis=0, ddof=1))
    if not sds:
        return 0.0, 1.0
    A = np.concatenate(sds)
    lo, hi = float(np.percentile(A, 10)), float(np.percentile(A, 90))
    return lo, (hi if hi > lo else lo + 1e-6)


def centroid_tinted(out_png, Xc, vlim, grid, cid, sd_lo, sd_hi):
    """concat_rawds: the cluster mean, greyed wherever its electrodes disagree.

    This is the ordinary centroid heatmap, not a raster. What it adds is that the
    confidence in each time x frequency bin is carried by how much colour that bin
    keeps: where the electrodes in the cluster agree the bin stays saturated, where
    they disagree it fades toward grey. A plain mean cannot separate "the whole
    cluster sits near zero here" from "the cluster is split here and the average
    cancels"; both come out white. With the tint the first stays white and the
    second turns grey.

    The SD is the spread ACROSS ELECTRODES in the cluster, the same quantity the
    HG centroid draws as its +/-1 SD band, and it is scaled on the run's range
    rather than the cluster's - see sd_scale.
    """
    bands, conds, times = grid
    nb, nc, nt = len(bands), len(conds), len(times)
    n = len(Xc)
    mean = Xc.mean(axis=0).reshape(nb, nc * nt)
    sd = (Xc.std(axis=0, ddof=1) if n > 1
          else np.zeros(Xc.shape[1])).reshape(nb, nc * nt)

    # bwr on the mean, then pulled toward grey by the local SD.
    rgb = plt.get_cmap(CMAP)((np.clip(mean, -vlim, vlim) / vlim + 1) / 2)[..., :3]
    w = (np.clip((sd - sd_lo) / (sd_hi - sd_lo), 0, 1) * TINT_MAX)[..., None]
    img = rgb * (1 - w) + np.asarray(TINT_RGB) * w

    # hspace is a fraction of the AVERAGE axes height, and with a 13:1 split the
    # average is dominated by the heatmap - so a value that looks small here still
    # has to leave room for the condition labels without stranding the legend.
    fig = plt.figure(figsize=(5.2, 2.75), dpi=220)
    gs = GridSpec(2, 2, width_ratios=[40, 1.4], height_ratios=[13, 0.8],
                  wspace=0.05, hspace=0.30, left=0.15, right=0.89,
                  top=0.87, bottom=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, aspect="auto", origin="lower", interpolation="nearest")
    _decorate_x(ax, nc * nt, nc, heatmap=True, labels_on=True)
    tick_at = sorted({0, nb // 3, 2 * nb // 3, nb - 1})
    ax.set_yticks(tick_at)
    ax.set_yticklabels([bands[t] for t in tick_at], fontsize=6)
    ax.tick_params(length=2, width=0.6, pad=1.5, colors="#68727d")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#c8cfd6")
    ax.set_title(f"cluster {cid} centroid — {n} electrodes, greyed where they disagree",
                 fontsize=8.5, loc="left", pad=4)

    cax = fig.add_subplot(gs[0, 1])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vlim, vlim))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("power (dB)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)

    # The tint legend shows the fade applied to a fully saturated bin, which is
    # what the reader has to judge: how much colour loss means how much SD.
    axl = fig.add_subplot(gs[1, 0])
    lw = np.linspace(0, 1, 256)[:, None] * TINT_MAX
    base = np.asarray(plt.get_cmap(CMAP)(1.0)[:3])[None, :]
    strip = base * (1 - lw) + np.asarray(TINT_RGB)[None, :] * lw
    axl.imshow(strip[None, :, :], aspect="auto")
    axl.set_yticks([])
    axl.set_xticks([0, 255])
    axl.set_xticklabels([f"{sd_lo:.2f}", f"{sd_hi:.2f}"], fontsize=6)
    axl.tick_params(length=2, width=0.5, pad=1.5, colors="#68727d")
    axl.set_xlabel("within-cluster SD across electrodes (dB) — greyer = less agreement",
                   fontsize=6, labelpad=1.5, color="#68727d")
    for sp in axl.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#c8cfd6")

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
        sd_lo, sd_hi = (0.0, 1.0) if line else sd_scale(X, labels, ids)
        detail = ("sorted by %s" % sname if line
                  else "grey tint over SD %.2f-%.2f dB" % (sd_lo, sd_hi))
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
                nn = centroid_tinted(out, Xc, vlim, grid, cid, sd_lo, sd_hi)
                print(f"    cluster {cid}: {nn} electrodes")
        print(f"    -> {out_dir}")

    print("\n  cluster_rasters/ written. cluster_centroids/ is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
