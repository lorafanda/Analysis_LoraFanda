#!/usr/bin/env python3
"""
make_gate_split_figures.py - does any cluster consist mostly of the electrodes the
responsiveness gate would have thrown away?

Writes outputs/clustering/comparison/: two figures and the JSON behind them.

G1  Per-electrode confidence, grouped by cluster and coloured by gate - the same
    layout as panel B of the membership figure, but the x axis is SILHOUETTE rather
    than the loading margin, so it applies to k-means and Ward as well as to convex
    NMF. Blue = would pass the gate, grey = only present because it was lifted.

G2  Whether the added electrodes JOIN a cluster's response type or DILUTE it. For
    each cluster, Cohen's d against all other electrodes computed twice - once from
    its gated members only, once from its added members only - and the correlation
    between the two. High r means the added electrodes carry the same feature
    signature; low r means the cluster is two different things wearing one label.

EACH METHOD IS MEASURED IN THE SPACE IT FITS IN. Convex NMF unit-norms each
electrode before fitting; k-means and Ward use raw dB. Scoring all three in dB is
the error that made an earlier version of the separation figure wrong, and
silhouette is not a space-free quantity.

The result this was built to check: on the hard partitions there IS such a cluster -
k-means c2 holds 1696 electrodes at 74% added and Ward c3 holds 2075 at 71% - while
convex NMF has none, its most extreme cluster being 69% against a 57% baseline. The
hard methods in dB are partly separating responsive from non-responsive electrodes,
which is exactly the failure the gate exists to prevent, and the silhouette makes it
visible: for k-means and Ward the ADDED electrodes score HIGHER than the gated ones.

    python make_gate_split_figures.py
"""
from __future__ import annotations

import argparse
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

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "comparison"

RUNS = [
    ("cNMF", "cnmf/concat_hg_all/runs/20260819_220417", "unit-norm"),
    ("k-means", "kmeans/concat_hg_all/runs/20260819_235524", "dB"),
    ("Ward", "hierarchical/concat_hg_all/runs/20260819_235654", "dB"),
]

GATED_COL = "#4a6fa5"
ADDED_COL = "#b0b7be"
INK, MUTED = "#1b232c", "#68727d"
COND_NAMES = ("audio", "picture", "reading")


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def load(rel, space):
    d = CLUST / rel
    X = np.load(d / "X_train.npy").astype(float)
    lab = pd.read_csv(d / "labels.csv")
    ccol = next(c for c in lab.columns
                if c.startswith("cluster_") and not c.endswith("_ranked"))
    L = pd.to_numeric(lab[ccol], errors="coerce").to_numpy()
    gate = pd.to_numeric(lab["n_high_activity"], errors="coerce").fillna(0).to_numpy() > 0
    A = unit(X) if space == "unit-norm" else X
    return X, A, L, gate, d.name


def cohens_d(X, m_in, m_out):
    a, b = X[m_in], X[m_out]
    if len(a) < 2 or len(b) < 2:
        return None
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1))
                 / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / np.maximum(sp, 1e-9)


def blocks(ax, n_x, n_blocks, labels_on):
    per = n_x / n_blocks
    for b in range(1, n_blocks):
        ax.axvline(b * per, color="k", lw=0.8, zorder=5)
    for b in range(n_blocks):
        ax.axvline((b + 0.5) * per, color="#3a3a3a", lw=0.7, ls=(0, (4, 3)),
                   alpha=0.75, zorder=6)
    if labels_on:
        ax.set_xticks([(b + 0.5) * per for b in range(n_blocks)])
        ax.set_xticklabels(list(COND_NAMES)[:n_blocks], fontsize=7)
    else:
        ax.set_xticks([])


# ==============================================================================
def figure_g1(data, out_png, stats):
    from sklearn.metrics import silhouette_samples
    lo = min(d["sil"].min() for d in data.values())
    hi = max(d["sil"].max() for d in data.values())
    pad = 0.05 * (hi - lo)

    fig = plt.figure(figsize=(11.6, 7.6), dpi=200)
    gs = GridSpec(1, len(RUNS), wspace=0.26, left=0.055, right=0.985,
                  top=0.735, bottom=0.075)

    for i, (name, rel, space) in enumerate(RUNS):
        d = data[name]
        L, gate, sil = d["L"], d["gate"], d["sil"]
        ids = sorted(set(L))
        order = np.lexsort((-sil, L))
        s_o, g_o, l_o = sil[order], gate[order], L[order]
        bounds = np.searchsorted(l_o, np.arange(len(ids) + 1))

        ax = fig.add_subplot(gs[0, i])
        ax.barh(np.arange(len(s_o)), s_o, height=1.0, linewidth=0,
                color=np.where(g_o, GATED_COL, ADDED_COL))
        ax.axvline(0, color="#c1121f", lw=0.9)
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="#2bb3c0", lw=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(len(s_o), 0)
        ax.set_xlabel("silhouette", fontsize=9)
        if i == 0:
            ax.set_ylabel(f"{len(s_o)} electrodes, grouped by cluster then silhouette",
                          fontsize=8.5, labelpad=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7, length=2, colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # Both annotations go INSIDE the panel. Outside, each panel's right-hand
        # labels landed on the next panel's cluster names.
        for j, cid in enumerate(ids):
            m = L == cid
            pct = 100 * (~gate[m]).mean()
            mid = (bounds[j] + bounds[j + 1]) / 2
            ax.text(lo, mid, f" c{cid}", fontsize=7.5, ha="left", va="center",
                    color=INK)
            strong = abs(pct - stats["baseline_pct_added"]) > 10
            ax.text(hi, mid, f"{pct:.0f}% added ", ha="right", va="center",
                    fontsize=8.4 if strong else 7.2,
                    color="#5f6a72" if pct > stats["baseline_pct_added"] else GATED_COL,
                    fontweight="bold" if strong else "normal")
        st = stats["methods"][name]
        ax.set_title(f"{name}  ·  fits in {space}\n"
                     f"silhouette: gated {st['sil_gated']:+.3f}, "
                     f"added {st['sil_added']:+.3f}",
                     fontsize=9.5, loc="left", color=INK, pad=6)

    fig.suptitle("Is any cluster mostly electrodes the gate would have removed?",
                 x=0.055, y=0.975, ha="left", fontsize=14, color=INK)
    base = stats["baseline_pct_added"]
    fig.text(0.055, 0.925, "\n".join([
        f"All three run on the same {stats['n']} ungated electrodes. "
        f"{GATED_COL and ''}Blue = would pass the responsiveness gate, "
        f"grey = only present because it was lifted;",
        f"{base:.0f}% of the whole set is grey, so a cluster is only notable if it "
        f"departs from that. Each method is scored in the space it fits in - "
        f"silhouette is not space-free.",
        "",
        "Convex NMF has no such cluster: its most extreme is "
        f"{stats['methods']['cNMF']['max_added_pct']:.0f}% against the {base:.0f}% "
        "baseline. The hard partitions do, and in both directions - a very large",
        "cluster of mostly-added electrodes and a small almost-purely-gated one. For "
        "k-means and Ward the ADDED electrodes score HIGHER silhouette than the gated "
        "ones, which is the",
        "shape of a partition that is separating responsive from non-responsive rather "
        "than one response type from another.",
    ]), fontsize=8.4, color=MUTED, va="top", linespacing=1.45)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ==============================================================================
def figure_g2(data, out_png, stats):
    rows = [(name, cid) for name, _, _ in RUNS for cid in sorted(set(data[name]["L"]))]
    n_rows = len(rows)
    fig = plt.figure(figsize=(9.4, 0.86 * n_rows + 2.6), dpi=200)
    H = 0.86 * n_rows + 2.6
    gs = GridSpec(n_rows, 1, hspace=0.34, left=0.135, right=0.945,
                  top=1 - 2.1 / H, bottom=0.55 / H)

    lim = stats["g2_dlim"]
    for r, (name, cid) in enumerate(rows):
        d = data[name]
        X, L, gate = d["X"], d["L"], d["gate"]
        m = L == cid
        dg = cohens_d(X, m & gate, ~m)
        da = cohens_d(X, m & ~gate, ~m)
        ax = fig.add_subplot(gs[r, 0])
        ax.axhline(0, color="#bbb", lw=0.5)
        if dg is not None:
            ax.plot(dg, lw=1.0, color=GATED_COL, label="gated members")
        if da is not None:
            ax.plot(da, lw=1.0, color="#8e969e", label="added members")
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, X.shape[1] - 1)
        blocks(ax, X.shape[1], 3, labels_on=(r == n_rows - 1))
        ax.set_yticks([-lim, 0, lim])
        ax.set_yticklabels([f"{-lim:g}", "0", f"{lim:g}"], fontsize=6)
        ax.tick_params(length=2, width=0.5, colors=MUTED, labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        key = f"{name}|c{cid}"
        rr = stats["g2"][key]["r"]
        ax.set_ylabel(f"{name}\nc{cid}", fontsize=7.5, rotation=0, ha="right",
                      va="center", labelpad=18, color=INK)
        ax.text(1.008, 0.5, f"r={rr:+.2f}\n{stats['g2'][key]['pct_added']:.0f}% added",
                transform=ax.transAxes, fontsize=6.4, color=MUTED,
                ha="left", va="center")
        if r == 0:
            ax.legend(fontsize=6.6, frameon=False, loc="upper right", ncol=2)

    fig.suptitle("Do the added electrodes join the cluster, or dilute it?",
                 x=0.135, y=1 - 0.28 / H, ha="left", fontsize=13, color=INK)
    fig.text(0.135, 1 - 0.62 / H, "\n".join([
        "For each cluster, Cohen's d against all other electrodes, computed twice: "
        "once from its GATED members only (blue) and once from",
        "its ADDED members only (grey). r is the correlation between the two curves.",
        "",
        "High r means the added electrodes carry the same feature signature and are "
        "genuinely joining that response type. Low r means the",
        "cluster is two different things sharing one label, and its centroid is an "
        "average of both.",
    ]), fontsize=8.2, color=MUTED, va="top", linespacing=1.45)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    from sklearn.metrics import silhouette_samples

    OUT.mkdir(parents=True, exist_ok=True)
    data, stats = {}, {"methods": {}, "g2": {}}
    for name, rel, space in RUNS:
        X, A, L, gate, run_id = load(rel, space)
        sil = silhouette_samples(A, L)
        data[name] = dict(X=X, A=A, L=L, gate=gate, sil=sil, run=run_id, space=space)
        ids = sorted(set(L))
        per = {}
        for cid in ids:
            m = L == cid
            per[str(cid)] = dict(
                n=int(m.sum()), pct_added=float(100 * (~gate[m]).mean()),
                sil=float(sil[m].mean()),
                sil_gated=float(sil[m & gate].mean()) if (m & gate).any() else None,
                sil_added=float(sil[m & ~gate].mean()) if (m & ~gate).any() else None)
        stats["methods"][name] = dict(
            run=run_id, space=space, K=len(ids),
            sil_overall=float(sil.mean()),
            sil_gated=float(sil[gate].mean()), sil_added=float(sil[~gate].mean()),
            max_added_pct=float(max(v["pct_added"] for v in per.values())),
            min_added_pct=float(min(v["pct_added"] for v in per.values())),
            clusters=per)
        print(f"  {name:<8} K={len(ids)} sil {sil.mean():+.3f} "
              f"(gated {sil[gate].mean():+.3f}, added {sil[~gate].mean():+.3f})")

    g0 = data["cNMF"]["gate"]
    stats["n"] = int(len(g0))
    stats["baseline_pct_added"] = float(100 * (~g0).mean())

    # G2 statistics, and one shared d limit so the rows compare
    dmax = []
    for name, _, _ in RUNS:
        d = data[name]
        for cid in sorted(set(d["L"])):
            m = d["L"] == cid
            dg = cohens_d(d["X"], m & d["gate"], ~m)
            da = cohens_d(d["X"], m & ~d["gate"], ~m)
            r = float(np.corrcoef(dg, da)[0, 1]) if (dg is not None and da is not None) else float("nan")
            stats["g2"][f"{name}|c{cid}"] = dict(
                r=r, pct_added=float(100 * (~d["gate"][m]).mean()),
                n_gated=int((m & d["gate"]).sum()), n_added=int((m & ~d["gate"]).sum()))
            for v in (dg, da):
                if v is not None:
                    dmax.append(float(np.percentile(np.abs(v), 99)))
    stats["g2_dlim"] = float(np.ceil(max(dmax) * 4) / 4)

    figure_g1(data, OUT / "G1_silhouette_by_gate.png", stats)
    figure_g2(data, OUT / "G2_drivers_by_gate.png", stats)
    (OUT / "gate_split_stats.json").write_text(json.dumps(stats, indent=2),
                                               encoding="utf-8")
    print()
    print(f"  baseline: {stats['baseline_pct_added']:.1f}% of all {stats['n']} "
          f"electrodes were added by ungating")
    print("  gated-vs-added driver agreement (r), per cluster:")
    for k, v in stats["g2"].items():
        flag = "   <-- added electrodes are a different response" if v["r"] < 0.5 else ""
        print(f"    {k:<14} r={v['r']:+.2f}  ({v['pct_added']:.0f}% added){flag}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
