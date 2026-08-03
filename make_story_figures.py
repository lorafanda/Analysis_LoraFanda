#!/usr/bin/env python3
"""
make_story_figures.py — composite figures, one per argument.

The status site grew to 46 numbered figures ordered by when they were made rather
than by what they argue, and several of them are pieces of a single claim. These
composites put each claim in one figure:

    S1  three lenses agree      a-priori ROLE timing vs data-driven CLUSTER timing
    S2  how good is the partition   4 independent criteria, one verdict
    S3  decoding: real, and why     above chance -> HFA carries it -> structured errors -> where
    S4  timing: what/where/when/order
    S5  atlas convergence           where the network is -> which TF bins track it

S1 is computed; S2-S5 re-tile figures that already exist, cropping the dead white
margin so the panels are legible at page size.

    python make_story_figures.py            # all
    python make_story_figures.py --only S1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "02_FBM_Clustering" / "outputs" / "clustering" / "story"
CLUST = ROOT / "02_FBM_Clustering" / "outputs" / "clustering"
POOL = ROOT / "04_FBM_Pooling" / "outputs" / "pooling"
CLS = ROOT / "03_FBM_Classifying" / "outputs" / "classification"
RUN = CLUST / "kmeans" / "concat_hg" / "runs" / "20260802_220720"
CONDS = ("audio", "picture", "reading")
MIN_N = 10          # roles/clusters smaller than this have a meaningless "mean"


def crop(path, pad=10, thr=245):
    from PIL import Image
    path = Path(path)
    if not path.exists():
        return None
    a = np.asarray(Image.open(path).convert("RGB"))
    ink = (a < thr).any(axis=2)
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    return a[max(0, ys.min() - pad):ys.max() + pad,
             max(0, xs.min() - pad):xs.max() + pad]


def _panel(ax, path, title=None, missing="not generated yet"):
    img = crop(path)
    if img is not None:
        ax.imshow(img)
    else:
        ax.text(.5, .5, missing, ha="center", va="center", fontsize=9,
                color="#aab2ba", transform=ax.transAxes)
    if title:
        ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _tag(ax, letter):
    ax.text(-0.012, 1.035, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="right")


# ============================================================ S1
def s1(dpi=150):
    """Do the a-priori roles and the discovered clusters order time the same way?"""
    rp = POOL / "pool_web" / "role_onsets.csv"
    cp = RUN / "timing" / "cluster_timing_by_k.csv"
    if not rp.exists() or not cp.exists():
        print("  S1: need role_onsets.csv (make_role_onsets.py) and cluster_timing_by_k.csv (214)")
        return
    roles = pd.read_csv(rp)
    clus = pd.read_csv(cp)
    K = 10 if 10 in set(clus.k) else sorted(clus.k)[0]
    clus = clus[clus.k == K]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 7.2), sharex=True)
    for ax, cond in zip(axes, CONDS):
        r = (roles[(roles.condition == cond) & (roles.n >= MIN_N)]
             .dropna(subset=["onset_pct"]).copy())
        c = (clus[clus.condition == cond].dropna(subset=["onset_pct"]).copy())
        c = c[(c.n >= MIN_N) & (c.onset_pct < 100)]
        r["kind"], r["label"] = "role", r["role"].str.slice(0, 26)
        c["kind"], c["label"] = "cluster", "c" + c["cluster"].astype(str)
        both = (pd.concat([r[["label", "onset_pct", "n", "kind"]],
                           c[["label", "onset_pct", "n", "kind"]]])
                .sort_values("onset_pct").reset_index(drop=True))
        col = ["#0b7a75" if k == "role" else "#b85c6e" for k in both.kind]
        y = np.arange(len(both))
        ax.axvspan(0, 50, color="#f2f4f7", zorder=0)
        ax.axvline(50, color="0.35", ls="--", lw=1.1, zorder=1)
        ax.barh(y, both.onset_pct, color=col, height=.74, zorder=2)
        for i, (v, n) in enumerate(zip(both.onset_pct, both.n)):
            ax.text(v + 1.2, i, f"{v:.0f}", va="center", fontsize=7, color="#333")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{l}  ({n})" for l, n in zip(both.label, both.n)], fontsize=7.6)
        for tick, k in zip(ax.get_yticklabels(), both.kind):
            tick.set_color("#0b7a75" if k == "role" else "#b85c6e")
            if k == "cluster":
                tick.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 104)
        ax.set_xlabel("onset (% of trial;  50 = GO)")
        ax.set_title(cond.capitalize(), fontsize=12)
        ax.grid(axis="x", alpha=.25)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.suptitle("S1 · Do the two lenses order time the same way?      "
                 "teal = a-priori ROLE (theory)      red = data-driven CLUSTER (data)      "
                 f"K={K}, groups with n≥{MIN_N}", fontsize=12, y=1.002)
    fig.tight_layout()
    p = OUT / "S1_three_lenses.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  S1 -> {p.name} ({p.stat().st_size/1024:.0f} KB)")


# ============================================================ S2..S5 (re-tile)
def _grid(specs, out_name, suptitle, figsize, dpi, nrow, ncol, ratios=None):
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                             gridspec_kw=({"height_ratios": ratios} if ratios else None))
    axes = np.atleast_1d(axes).ravel()
    for ax, (letter, path, title) in zip(axes, specs):
        _panel(ax, path, title)
        _tag(ax, letter)
    for ax in axes[len(specs):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12.5, y=0.998)
    fig.tight_layout()
    p = OUT / out_name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out_name.split('_')[0]} -> {p.name} ({p.stat().st_size/1024:.0f} KB)")


def s2(dpi=150):
    _grid([("a", RUN / "sweep_metrics.png", "every K scored on 4 criteria"),
           ("b", RUN / "silhouette_per_cluster.png", "per-sample silhouette + PCA"),
           ("c", RUN / "consensus_heatmap.png", "consensus across resamples"),
           ("d", RUN / "similarity_heatmap.png", "sample x sample distance")],
          "S2_partition_quality.png",
          "S2 · How good is this partition? Four independent criteria, one verdict",
          (16, 11.5), dpi, 2, 2)


def s3(dpi=150):
    C = CLS / "_compare"
    _grid([("a", C / "permutation_null_summary.png", "above chance? (200 permutations)"),
           ("b", C / "hg_vs_full.png", "HFA vs full spectrum, per patient"),
           ("c", C / "delta_confusion_condition.png", "which classes are confused"),
           ("d", CLS / "condition/full_300/logreg/runs/20260624_184315/coef_ersp_maps.png",
            "where the weights live on the TF plane")],
          "S3_decoding.png",
          "S3 · Decoding: is it real, and why?  above chance → HFA carries it → "
          "structured errors → early modality-specific window",
          (16.5, 11), dpi, 2, 2)


def s4(dpi=150):
    T = RUN / "timing"
    _grid([("a", RUN / "cluster_cards.png", "what each cluster is, and where"),
           ("b", T / "timing_k10_panel.png", "when it comes on (K=10)"),
           ("c", T / "xcorr_K10_ref-c1_response.png", "in what order, vs the auditory cluster")],
          "S4_timing.png",
          "S4 · Timing: what, where, when, and in what order",
          (15, 15), dpi, 3, 1, ratios=[1.15, 1.25, 0.9])


def s5(dpi=150):
    A = POOL / "atlas_corr" / "fedorenko"
    _grid([("a", CLUST / "atlas/lana_concat/lana_cards.png", "where the language network is"),
           ("b", A / "fig9_corr_map_berlin.png", "which TF bins track P(language)"),
           ("c", A / "fig8_bin_scatters.png", "one bin: ERSP vs P(language) across contacts")],
          "S5_atlas.png",
          "S5 · Language-atlas convergence: where the network is → which parts of the "
          "response track it",
          (14, 15), dpi, 3, 1, ratios=[1.25, 0.85, 0.9])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}
    for k, fn in todo.items():
        if a.only and k != a.only:
            continue
        fn(dpi=a.dpi)
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
