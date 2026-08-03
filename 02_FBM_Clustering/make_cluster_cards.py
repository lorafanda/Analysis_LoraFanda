#!/usr/bin/env python3
"""
make_cluster_cards.py — compact per-cluster summary cards for a clustering run.

The raw 252 glassbrains are 1200x1000 with about 19% ink: the brain floats in a wide
white margin, so scaling them down to fit a page shrinks the brain, not the margin, and
they become unreadable. Stacking four of them full-width is worse still.

This crops each view to its actual content and tiles the pieces that belong together
into ONE row per cluster:

    waveform  |  lateral L  |  lateral R  |  dorsal  |  anatomy text

so the temporal profile, the anatomical spread and the region labels are read in a
single glance instead of scattered across four figures. One PNG holds every cluster,
which also makes the clusters directly comparable to each other.

    python make_cluster_cards.py                       # newest kmeans/concat_hg run
    python make_cluster_cards.py --run <path> --out cluster_cards.png
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_TRACK = ROOT / "outputs" / "clustering" / "kmeans" / "concat_hg" / "runs"
VIEWS = ("lateral_L", "lateral_R", "dorsal")


def crop_to_content(path: Path, pad: int = 12, thr: int = 245):
    """Trim the white margin. Returns None when the file is missing or blank."""
    from PIL import Image
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = (a < thr).any(axis=2)
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad)
    return a[y0:y1, x0:x1]


def chance_purity(labels_by_region: pd.Series, n: int, n_draw: int = 400, seed: int = 0) -> float:
    """Purity a cluster of this size would reach by drawing regions at random.

    Raw purity is uninterpretable on its own: with a skewed region distribution a
    random draw already lands ~0.24 here, so a cluster at 0.18 is BELOW chance and one
    at 0.41 is genuinely coherent. Reporting the ratio instead of the raw number is the
    difference between "all our clusters are anatomically weak" and the truth.
    """
    rng = np.random.default_rng(seed)
    vals = labels_by_region.to_numpy()
    if n < 1 or len(vals) < n:
        return float("nan")
    out = np.empty(n_draw)
    for i in range(n_draw):
        s = rng.choice(vals, n, replace=False)
        _, c = np.unique(s, return_counts=True)
        out[i] = c.max() / n
    return float(out.mean())


def top_regions(anat: pd.DataFrame, cid: int, k: int = 3):
    row = anat[anat["cluster_id"] == cid]
    if not len(row):
        return []
    try:
        return ast.literal_eval(str(row.iloc[0]["top_3_json"]))[:k]
    except Exception:
        return []


def lana_cards(dpi: int = 140) -> None:
    """Same treatment for the language-atlas partitions.

    FIG 3a.R1-R3 showed 18 separate 1200x1000 glassbrains, which is unreadable for the
    same reason. One card per cohort instead: rows are the three cuts, columns the three
    views, with the in-network count on each row.
    """
    base = ROOT / "outputs" / "clustering" / "atlas"
    CUTS = [("20260802_thr0p05", "P >= 0.05", "01"),
            ("20260802_thr0p1", "P >= 0.10", "01"),
            ("20260802_pbins", "top band (P >= 0.35)", "04")]
    for cohort, nice in (("lana_all", "all electrodes"), ("lana_concat", "concatenated cohort")):
        runs = base / cohort / "runs"
        if not runs.is_dir():
            print(f"  {cohort}: no runs")
            continue
        rows = [(lbl, runs / rid / "recon" / f"cluster_{cl}", runs / rid)
                for rid, lbl, cl in CUTS if (runs / rid / "recon" / f"cluster_{cl}").is_dir()]
        if not rows:
            print(f"  {cohort}: no rendered recon")
            continue
        fig, axes = plt.subplots(len(rows), len(VIEWS),
                                 figsize=(2.9 * len(VIEWS), 2.15 * len(rows)),
                                 gridspec_kw={"wspace": 0.02, "hspace": 0.10})
        axes = np.atleast_2d(axes)
        for r, (lbl, cdir, rundir) in enumerate(rows):
            n_in = ""
            lp = rundir / "labels.csv"
            if lp.exists():
                d = pd.read_csv(lp)
                col = next((c for c in d.columns if c.startswith("cluster_")), None)
                if col is not None:
                    want = int(cdir.name.split("_")[1])
                    n_in = "\n" + f"n={int((d[col] == want).sum())}"
            for j, v in enumerate(VIEWS):
                img = crop_to_content(cdir / "by_condition" / f"{v}.png")
                ax = axes[r, j]
                if img is not None:
                    ax.imshow(img)
                if r == 0:
                    ax.set_title(v.replace("_", " "), fontsize=9.5)
                if j == 0:
                    ax.set_ylabel(lbl + n_in, fontsize=10, fontweight="bold",
                                  rotation=0, ha="right", va="center", labelpad=52)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
        fig.suptitle(f"LanA language network — {nice}", fontsize=12, y=0.99)
        out = base / cohort / "lana_cards.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {cohort}: {len(rows)} cuts -> {out}  ({out.stat().st_size/1024:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default="cluster_cards.png")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--lana", action="store_true",
                    help="build the language-atlas cards instead")
    a = ap.parse_args()

    if a.lana:
        lana_cards(dpi=a.dpi)
        return 0

    run = Path(a.run) if a.run else sorted(d for d in DEFAULT_TRACK.iterdir() if d.is_dir())[-1]
    recon = run / "recon"
    if not recon.is_dir():
        raise SystemExit(f"no recon/ in {run} — run 252 first")
    clusters = sorted(int(d.name.split("_")[1]) for d in recon.iterdir()
                      if d.is_dir() and d.name.startswith("cluster_"))
    if not clusters:
        raise SystemExit(f"no cluster_* folders in {recon}")

    labels = pd.read_csv(run / "labels.csv")
    ccol = next(c for c in labels.columns if c.startswith("cluster_") and not c.endswith("_ranked"))
    sizes = labels[ccol].value_counts().to_dict()
    # Recompute the region labels here with a NORMALISED join. per_cluster_anatomy.csv
    # is written by 211, and any copy produced before the lf_anatomy fix keyed on the raw
    # electrode string and dropped ~45% of contacts into "unknown".
    def _nz(s):
        return str(s).replace("_", "").replace("-", "").upper()

    apath = ROOT / "outputs" / "250_recon" / "fsaverage" / "aparc_lookup.csv"
    lut, hemi_lut = {}, {}
    if apath.exists():
        _ap = pd.read_csv(apath)
        lut = {(str(q), _nz(e)): str(l)
               for q, e, l in zip(_ap["patient"], _ap["electrode"], _ap["aparc_label"])}
        if "hemi" in _ap.columns:
            hemi_lut = {(str(q), _nz(e)): str(h)
                        for q, e, h in zip(_ap["patient"], _ap["electrode"], _ap["hemi"])}
    labels["_reg"] = [lut.get((str(q), _nz(e)), "unknown")
                      for q, e in zip(labels["patient_id"], labels["electrode"])]
    labels["_hemi"] = [hemi_lut.get((str(q), _nz(e)), "?")
                       for q, e in zip(labels["patient_id"], labels["electrode"])]
    known = labels[labels["_reg"] != "unknown"]
    print(f"  region labels: {len(known)}/{len(labels)} matched "
          f"({100*len(known)/max(len(labels),1):.0f}%)")

    ncol = 1 + len(VIEWS) + 1                       # waveform + views + text
    widths = [1.35] + [1.0] * len(VIEWS) + [1.25]
    fig, axes = plt.subplots(len(clusters), ncol,
                             figsize=(sum(widths) * 2.35, 2.05 * len(clusters)),
                             gridspec_kw={"width_ratios": widths,
                                          "wspace": 0.03, "hspace": 0.12})
    axes = np.atleast_2d(axes)

    for r, cid in enumerate(clusters):
        wave = crop_to_content(run / "cluster_centroids" / f"cluster_{cid:02d}.png", pad=2)
        ax = axes[r, 0]
        if wave is not None:
            ax.imshow(wave)
        ax.set_ylabel(f"c{cid}\nn={sizes.get(cid, 0)}", fontsize=11, fontweight="bold",
                      rotation=0, ha="right", va="center", labelpad=26)
        if r == 0:
            ax.set_title("mean HG  [audio | picture | reading]", fontsize=8.5)

        for j, v in enumerate(VIEWS, start=1):
            img = crop_to_content(recon / f"cluster_{cid:02d}" / "by_condition" / f"{v}.png")
            axb = axes[r, j]
            if img is not None:
                axb.imshow(img)
            else:
                axb.text(.5, .5, "not rendered", ha="center", va="center",
                         fontsize=8, color="#aab2ba", transform=axb.transAxes)
            if r == 0:
                axb.set_title(v.replace("_", " "), fontsize=9)

        axt = axes[r, ncol - 1]
        sub = labels[labels[ccol] == cid]
        kn = sub[sub["_reg"] != "unknown"]
        lines = []
        if len(kn):
            vc = kn["_reg"].value_counts()
            for nm, cnt in vc.head(3).items():
                lines.append(f"{nm[:20]}  {100*cnt/len(kn):.0f}%")
            pur = vc.iloc[0] / len(kn)
            ch = chance_purity(known["_reg"], len(kn))
            ratio = pur / ch if ch and np.isfinite(ch) else float("nan")
            verdict = ("above chance" if ratio > 1.15 else
                       "BELOW chance" if ratio < 0.95 else "at chance")
            lines += ["", f"purity {pur:.2f}  vs chance {ch:.2f}",
                      f"= {ratio:.2f}x  ({verdict})"]
        else:
            lines = ["no region match"]
        h = sub["_hemi"].astype(str).str.upper().str[0]
        nl, nr = int((h == "L").sum()), int((h == "R").sum())
        if nl + nr:
            lines.append(f"{nl} left / {nr} right   ({100*nl/(nl+nr):.0f}% LH)")
        axt.text(0.0, 0.5, "\n".join(lines), fontsize=8.4, va="center", ha="left",
                 transform=axt.transAxes)
        if r == 0:
            axt.set_title("anatomy  ·  purity vs chance  ·  hemisphere", fontsize=8.5)

        for axx in axes[r]:
            axx.set_xticks([]); axx.set_yticks([])
            for s in axx.spines.values():
                s.set_visible(False)

    fig.suptitle(f"Concatenated clusters — waveform, anatomy and location   ·   {run.name}"
                 f"   ·   n={len(labels)}", fontsize=11.5, y=0.995)
    out = run / a.out
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    kb = out.stat().st_size / 1024
    print(f"  {len(clusters)} clusters -> {out}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
