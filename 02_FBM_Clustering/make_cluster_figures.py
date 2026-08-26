#!/usr/bin/env python3
"""
make_bsf_figures.py - FIG C.3a/b/c and FIG C.8a/b/c, one per algorithm, all at K=8 on
the identical 2946 electrodes.

C.3 was built for convex NMF and two of its panels only exist because cNMF returns
graded loadings. Rebuilt here for three methods, so the panels had to be settled first:

    A   held-out variance vs K          all methods - bi-cross-validation, see
                                        make_heldout_variance.py for why the curve
                                        already on the site cannot be used
    B1  cluster response profiles       all methods - the mean of each cluster in dB
    B2  where those electrodes are      all methods - a 2D fsaverage projection,
                                        drawn identically for every method
    C   how confident is the label      SILHOUETTE for k-means and Ward, largest
                                        normalised LOADING for cnmf
    E   leave-one-patient-out vs null   all methods, each refit with ITSELF

    D   anatomical coherence            dropped, by request
    F   split-half replication          dropped, by request
    B3  loading-weighted glassbrain     cnmf only, and needs a pyvista render that
                                        does not exist at K=8 - not drawn

PANEL C IS NOT ONE SCALE. Silhouette is (b-a)/max(a,b) and runs -1..+1; a normalised
loading runs 0..1 and cannot go negative. They answer the same QUESTION - how much
better is this electrode's assignment than its next best - but the axes are different
quantities and the figure says so on the panel rather than in a caption. What IS
comparable is the shape and where the mass sits relative to the panel's own reference
line.

C.8 keeps the gate colouring: blue = would pass the responsiveness gate, grey = present
only because it was lifted. concat_hg_all IS the ungated set, so every cluster can be
read for whether it is mostly made of electrodes the gate would have removed.

    python make_bsf_figures.py
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import lf_decompose as D                      # noqa: E402
import measure_cluster_stability as MS        # noqa: E402

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "bsf_comparison"
K = 8
CONDS = ["audio", "picture", "reading"]

INK, MUTED = "#1b232c", "#68727d"
GATED_COL, ADDED_COL = "#4a6fa5", "#b0b7be"
# #b0b7be is the house grey for BARS, but at 2946 densely-packed scatter points it is
# invisible against white - a cluster that is 71% added still read as solid blue. The
# scatter uses the darker grey the existing FIG C.8 already uses for this population.
ADDED_DOT = "#8e969e"
RED, GREEN, GREY = "#c1121f", "#1b7837", "#c9ced4"
PAL = ["#4a6fa5", "#c1121f", "#1b7837", "#8c564b", "#9467bd",
       "#e08214", "#17a5b4", "#d64f9a"]

ORDER = ["kmeans", "hierarchical", "cnmf"]
TAG = {"kmeans": "a", "hierarchical": "b", "cnmf": "c"}
LABEL = {"kmeans": "k-means  —  BSF", "hierarchical": "Ward", "cnmf": "convex NMF"}
# resolved at run time: the cohort was rebuilt, so pinned run ids are meaningless
import lf_runs as LR                            # noqa: E402
FSET = "concat_hg"
RUNS: dict = {}


def confidence(method, X, lab, run):
    """How much better is this electrode's assignment than its next best.

    Returns (values, axis label, reference line, note). Two different quantities on
    purpose - the hard methods have no graded membership to report."""
    if method == "cnmf":
        G = np.load(run / "loadings_by_k" / f"G_k{K:02d}.npy")
        Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
        return (Gn.max(1), "largest normalised loading", 1.0 / K,
                f"0..1  ·  a flat mixture sits at 1/K = {1/K:.2f}")
    from sklearn.metrics import silhouette_samples
    A = MS.unit(X) if MS.SPACE[method] == "unit-norm" else X
    return (silhouette_samples(A, lab), "silhouette", 0.0,
            "-1..+1  ·  0 means the next cluster fits equally well")


def panel_A(ax, method, hv):
    """Held-out variance vs K, bi-cross-validated. Marks K=8 and the peak."""
    if hv is None or hv.empty:
        ax.text(.5, .5, "held-out variance not computed yet\n"
                        "(make_heldout_variance.py --feature-set concat_hg_all)",
                ha="center", va="center", fontsize=8, color=RED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("A · held-out variance vs K", fontsize=9.5, loc="left", color=INK)
        return
    for m in ORDER:
        s = (hv[hv.method == m].groupby("k")["var_explained"]
             .agg(["mean", "std"]).reset_index().sort_values("k"))
        if s.empty:
            continue
        me = m == method
        ax.plot(s.k, s["mean"], "-o", ms=3.4 if me else 2.2,
                lw=2.0 if me else 1.0, color=INK if me else MUTED,
                alpha=1.0 if me else 0.55, zorder=3 if me else 2,
                label=LABEL[m].split("  —")[0] + ("  ← this figure" if me else ""))
        if me:
            ax.fill_between(s.k, s["mean"] - s["std"], s["mean"] + s["std"],
                            color=INK, alpha=0.12, lw=0)
            pk = s.loc[s["mean"].idxmax()]
            ax.axvline(K, color=RED, ls="--", lw=1.0)
            ax.annotate(f"K=8\n{s.loc[s.k == K, 'mean'].iloc[0]:.3f}"
                        if (s.k == K).any() else "K=8",
                        xy=(K, ax.get_ylim()[0]), fontsize=7, color=RED,
                        ha="center", va="bottom")
            if int(pk.k) != K:
                ax.annotate(f"peak k={int(pk.k)}", xy=(pk.k, pk["mean"]),
                            xytext=(4, 6), textcoords="offset points",
                            fontsize=7, color=GREEN)
    ax.set_xlabel("K (components / clusters)", fontsize=8.5)
    ax.set_ylabel("held-out variance explained", fontsize=8.5)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A · held-out variance vs K — bi-cross-validated\n"
                 "each method in its home space; heights not comparable across spaces",
                 fontsize=9.5, loc="left", color=INK, pad=5)


def panel_B1(axes, X, lab, nb):
    """Cluster mean response, three conditions concatenated, raw dB."""
    lo = min(X[lab == j].mean(0).min() for j in range(K))
    hi = max(X[lab == j].mean(0).max() for j in range(K))
    for j, ax in enumerate(axes):
        sel = lab == j
        prof = X[sel].mean(0)
        for b in range(3):
            seg = prof[b * nb:(b + 1) * nb]
            ax.plot(np.arange(nb) + b * nb, seg, color=PAL[j], lw=1.1)
            if b:
                ax.axvline(b * nb, color=MUTED, lw=0.6)
            ax.axvline(b * nb + nb / 2, color=MUTED, lw=0.5, ls=":")
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_xticks([])
        if j:                                  # keep the dB scale on the first panel only
            ax.set_yticks([])
        ax.set_title(f"c{j}  n={int(sel.sum())}", fontsize=7.6, color=PAL[j], pad=2)
        if j == 0:
            ax.set_ylabel("dB", fontsize=7.5)
            ax.tick_params(labelsize=6.5, colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)


def panel_B2(axes, xyz, lab):
    """A 2D fsaverage projection per cluster - the same drawing for every method, so
    the three figures can be laid side by side. Sagittal, ANTERIOR ON THE LEFT, which
    is the convention the glassbrains in FIG C.3 B2 use."""
    ok = ~np.isnan(xyz).any(1)
    y, z = xyz[:, 1], xyz[:, 2]
    # one shared, data-driven frame for all eight panels, so cluster extents can be
    # compared by eye; autoscaling per panel would silently rescale each one
    pad = 6.0
    ylim = (np.nanmax(y[ok]) + pad, np.nanmin(y[ok]) - pad)   # inverted: anterior left
    zlim = (np.nanmin(z[ok]) - pad, np.nanmax(z[ok]) + pad)
    for j, ax in enumerate(axes):
        sel = (lab == j) & ok
        ax.scatter(y[ok], z[ok], s=1.0, color="#e6e9ec", lw=0)
        ax.scatter(y[sel], z[sel], s=3.4, color=PAL[j], lw=0)
        ax.set_xlim(*ylim); ax.set_ylim(*zlim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        # bottom-left, not centred-top: centred it sat on the electrodes themselves
        ax.text(0.02, 0.02, f"n={int(sel.sum())}", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=7.2, color=PAL[j])
        if j == 0:
            ax.set_ylabel("sagittal", fontsize=6.8, color=MUTED)


def panel_C(ax, vals, lab, xlabel, ref, note):
    ax.hist(vals, bins=44, color=GATED_COL, alpha=0.85, lw=0)
    med = float(np.median(vals))
    ax.axvline(med, color=INK, lw=1.6)
    ax.axvline(ref, color=RED, ls="--", lw=1.0)
    below = 100 * float((vals <= ref).mean())
    ax.annotate(f"median {med:.3f}", xy=(med, ax.get_ylim()[1] * 0.92),
                xytext=(5, 0), textcoords="offset points", fontsize=7.4, color=INK)
    ax.annotate(f"{below:.0f}% at or below {ref:.2f}",
                xy=(ref, ax.get_ylim()[1] * 0.70), xytext=(5, 0),
                textcoords="offset points", fontsize=7.2, color=RED)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel("electrodes", fontsize=8.5)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"C · how confident is each label\n{note}",
                 fontsize=9.5, loc="left", color=INK, pad=5)


def lopo_verdict(z):
    """Read the leave-one-patient-out z WITH ITS SIGN.

    inside  |z| < 2   losing a patient costs no more than losing that many random
                      electrodes - nobody carries the result. This is the pass.
    above   z >= +2   more robust than the size-matched null. Better than the pass.
    below   z <= -2   losing a PATIENT costs more than losing the same number of
                      random electrodes: an individual is holding it together."""
    if z <= -2:
        return "BELOW the null — an individual patient is carrying this solution", RED
    if z >= 2:
        return "ABOVE the null — more robust than losing the same count at random", GREEN
    return "inside the null — no single patient carries this solution", GREEN


def panel_E(ax, method, lopo, summ):
    d = lopo[lopo.method == method].sort_values("ari") if lopo is not None else None
    if d is None or d.empty:
        ax.text(.5, .5, "LOPO not computed yet\n(make_bsf_comparison.py)",
                ha="center", va="center", fontsize=8, color=RED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("E · leave-one-patient-out", fontsize=9.5, loc="left", color=INK)
        return
    s = (summ or {}).get("lopo", {}).get(method, {})
    ax.bar(np.arange(len(d)), d["ari"], color=GATED_COL, width=0.82, lw=0)
    if s:
        m, sd = s["null_min_mean"], s["null_min_sd"]
        ax.axhspan(m - sd, m + sd, color=MUTED, alpha=0.18, lw=0)
        ax.axhline(m, color=MUTED, lw=1.0, ls="--")
        # SIGN matters here and an earlier version threw it away with abs(z). Sitting
        # inside the null is the GOOD outcome - it means losing a patient costs no more
        # than losing that many random electrodes. Above the null is better still. Only
        # BELOW it means an individual patient was holding the solution together.
        verdict, col = lopo_verdict(s["z"])
        # top-left with a white backing: at the bottom this three-line block sat on the
        # bars, which are sorted ascending and lowest exactly there
        ax.annotate(f"size-matched null (min)  {m:.3f} ± {sd:.3f}\n"
                    f"worst real fold {s['real_min']:.3f}   z = {s['z']:+.2f}\n"
                    f"{verdict}",
                    xy=(0.02, 0.98), xycoords="axes fraction", va="top",
                    fontsize=7, color=col,
                    bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=2.2))
    ax.set_xticks([]); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{len(d)} patients, each left out once", fontsize=8.5)
    ax.set_ylabel("ARI vs full cohort", fontsize=8.5)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("E · leave-one-patient-out, refit with the SAME method\n"
                 "against a size-matched pseudo-patient null",
                 fontsize=9.5, loc="left", color=INK, pad=5)


def fig_c3(method, X, lab, xyz, nb, hv, lopo, summ, sep, sizes):
    fig = plt.figure(figsize=(15.4, 12.2), dpi=170)
    # top is low on purpose: the header is four wrapped lines and an earlier version
    # let panel C's two-line title collide with it.
    gs = GridSpec(3, K, figure=fig, height_ratios=[1.50, 0.90, 0.78],
                  hspace=0.62, wspace=0.30, left=0.05, right=0.98,
                  top=0.800, bottom=0.035)
    panel_A(fig.add_subplot(gs[0, 0:3]), method, hv)
    vals, xlabel, ref, note = confidence(method, X, lab, RUNS[method])
    panel_C(fig.add_subplot(gs[0, 3:6]), vals, lab, xlabel, ref, note)
    panel_E(fig.add_subplot(gs[0, 6:8]), method, lopo, summ)
    ax_b1 = [fig.add_subplot(gs[1, j]) for j in range(K)]
    ax_b2 = [fig.add_subplot(gs[2, j]) for j in range(K)]
    panel_B1(ax_b1, X, lab, nb)
    panel_B2(ax_b2, xyz, lab)
    # Row captions placed from the axes' OWN positions - hard-coded y values put them
    # on top of the panels as soon as the grid changed.
    for axes, txt in ((ax_b1, "B1 · cluster mean response — audio | picture | reading, "
                              "raw dB, one shared y-axis. Solid line = condition "
                              "boundary, dotted = GO cue."),
                      (ax_b2, f"B2 · where those electrodes are — sagittal fsaverage "
                              f"projection, ANTERIOR ON THE LEFT. Pale grey = all "
                              f"{len(lab)} electrodes; colour = this cluster.")):
        bb = axes[0].get_position()
        fig.text(0.05, bb.y1 + 0.028, txt, fontsize=9.5, color=INK)
    head(fig, method, sep, sizes, summ, len(lab))
    p = OUT / f"C3{TAG[method]}_{method}_K{K}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def head(fig, method, sep, sizes, summ, n_fallback):
    import textwrap
    s = sep[(sep.method == method) & (sep.home)]
    sz = sizes[sizes.method == method].sort_values("cluster")
    # fall back to the data rather than printing "0 electrodes" when the summary has
    # not been written yet - the figure must never state a count it did not measure
    n = int((summ or {}).get("n_electrodes") or n_fallback)
    npat = (summ or {}).get("n_patients", 27)
    fig.suptitle(f"FIG C.3{TAG[method]}   ·   {LABEL[method]}   ·   K = {K}",
                 x=0.05, y=0.988, ha="left", fontsize=16, color=INK)
    bits = [
        f"All three C.3 figures are drawn on the SAME {n} electrodes, "
        f"{npat} patients, feature set concat_hg_all "
        f"(ungated high gamma, 3 conditions x 300 bins). The three runs were checked "
        f"to carry a bit-identical X_train in the identical electrode order, so any "
        f"difference between C.3a, C.3b and C.3c is the METHOD and nothing else.",
    ]
    if not s.empty:
        r = s.iloc[0]
        bits.append(
            f"Separation in its home space ({r.space}): silhouette {r.silhouette:+.4f} "
            f"against a matched one-blob null of {r.null_mean:+.4f} ± {r.null_sd:.4f}, "
            f"z = {r.z:+.1f}. Cluster sizes {sz.n.tolist()}, "
            f"largest {sz.pct.max():.0f}% of the cohort.")
    bits.append(
        "Panels D (anatomical coherence) and F (split-half replication) are dropped by "
        "request; they are reported as statistics instead. B3 (loading-weighted "
        "glassbrain) is convex-NMF-only and needs a K=8 pyvista render that does not "
        "exist, so it is not drawn for any method rather than for one.")
    fig.text(0.05, 0.958, "\n".join(textwrap.fill(b, width=168) for b in bits),
             fontsize=8.5, color=MUTED, va="top", linespacing=1.5)


def fig_c8(method, X, lab, is_gated, sizes, base_pct):
    """Per-electrode confidence, grouped by cluster, coloured by the GATE."""
    vals, xlabel, ref, note = confidence(method, X, lab, RUNS[method])
    fig = plt.figure(figsize=(15.4, 7.6), dpi=170)
    gs = GridSpec(2, 1, height_ratios=[2.15, 1.0], hspace=0.42,
                  left=0.055, right=0.98, top=0.74, bottom=0.09)

    ax = fig.add_subplot(gs[0])
    # One interleaved scatter coloured by gate was CORRECT but unreadable: 2946 dots
    # packed into a thin band blend into a single blue-grey smear, so a cluster that is
    # 74% added still looked solid blue. The two populations are drawn as separate
    # sorted curves across the cluster's own width instead - which also makes the more
    # useful comparison visible, whether the added electrodes sit LOWER than the gated
    # ones inside the same cluster.
    x0, centres = 0, []
    for j in range(K):
        sel = np.where(lab == j)[0]
        w = len(sel)
        for keep, col, lw_ in ((is_gated[sel], GATED_COL, 1.9),
                               (~is_gated[sel], ADDED_DOT, 1.9)):
            v = np.sort(vals[sel][keep])
            if len(v) < 2:
                continue
            ax.plot(np.linspace(x0, x0 + w, len(v)), v, color=col, lw=lw_,
                    solid_capstyle="butt", zorder=3)
        centres.append((x0 + w / 2, 100 * float((~is_gated[sel]).mean())))
        if j:
            ax.axvline(x0, color=MUTED, lw=0.6)
        x0 += w
    ax.axhline(ref, color=INK, lw=0.8, ls="--")
    ax.set_xlim(0, x0); ax.set_xticks([])
    # The per-cluster "% added" text used to sit here too and collided on the narrow
    # clusters (c3 at 119 and c4 at 54 overlapped illegibly). It lives in the bar panel
    # below, which has room for it; only the cluster id is kept here, staggered.
    lo, hi = ax.get_ylim()
    for j, (cx, _) in enumerate(centres):
        ax.text(cx, hi - (hi - lo) * (0.015 + 0.055 * (j % 2)), f"c{j}", ha="center",
                va="top", fontsize=8.4, color=PAL[j], fontweight="bold")
    ax.set_ylabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([], [], color=GATED_COL, lw=2.4, label="gated — passes the gate")
    ax.plot([], [], color=ADDED_DOT, lw=2.4, label="added — only here because it was lifted")
    ax.legend(fontsize=7.4, frameon=False, loc="lower right", ncol=2)
    ax.set_title("Each cluster's two populations, sorted separately and drawn across "
                 "that cluster's own width.\n"
                 "A grey curve BELOW its blue one means the added electrodes are held "
                 "less confidently than the gated ones.",
                 fontsize=9.5, loc="left", color=INK)

    axb = fig.add_subplot(gs[1])
    sz = sizes[sizes.method == method].sort_values("cluster")
    axb.bar(np.arange(K), sz.pct_added, color=[RED if p > base_pct else GATED_COL
                                               for p in sz.pct_added], width=0.7, lw=0)
    axb.axhline(base_pct, color=INK, ls="--", lw=1.2)
    # anchored left: at the right edge this label sat on top of the c7 bar, which is
    # one of the two bars the figure is about
    axb.annotate(f"cohort baseline {base_pct:.1f}% added", xy=(-0.42, base_pct),
                 xytext=(0, 4), textcoords="offset points", ha="left",
                 fontsize=7.8, color=INK)
    for j, p in enumerate(sz.pct_added):
        axb.text(j, p + 1.5, f"{p:.0f}%", ha="center", fontsize=7.4,
                 color=RED if p > base_pct else MUTED)
    axb.set_xticks(range(K)); axb.set_xticklabels([f"c{j}" for j in range(K)], fontsize=8)
    axb.set_ylabel("% of cluster that is\ngate-added", fontsize=8.5)
    axb.set_ylim(0, 105)
    axb.tick_params(labelsize=7.5, colors=MUTED)
    axb.spines[["top", "right"]].set_visible(False)
    axb.set_title("Above the dashed line = the cluster is ENRICHED for electrodes the "
                  "gate would have removed", fontsize=9.5, loc="left", color=INK)

    import textwrap
    fig.suptitle(f"FIG C.8{TAG[method]}   ·   {LABEL[method]}   ·   K = {K}   ·   "
                 f"what the responsiveness gate was removing",
                 x=0.055, y=0.985, ha="left", fontsize=15.5, color=INK)
    body = (
        f"concat_hg_all IS the ungated set: {int(is_gated.sum())} electrodes would pass "
        f"the responsiveness gate and {int((~is_gated).sum())} are present only because "
        f"it was lifted, so {base_pct:.1f}% added is the cohort baseline every cluster "
        f"is read against. A cluster far above that line is partly separating RESPONSIVE "
        f"from NON-RESPONSIVE electrodes rather than one response type from another, "
        f"which is the failure the gate exists to prevent. "
        f"The y axis is {xlabel} ({note}) — a different quantity for the hard methods "
        f"than for convex NMF, so read the SHAPE and the colour mix within a panel, not "
        f"the height against the other two figures.")
    fig.text(0.055, 0.945, textwrap.fill(body, width=168), fontsize=8.5,
             color=MUTED, va="top", linespacing=1.5)
    p = OUT / f"C8{TAG[method]}_{method}_K{K}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main() -> int:
    global K, FSET, RUNS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--feature-set", default="concat_hg")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    K, FSET = int(a.k), a.feature_set
    RUNS = {m: LR.newest_run(m, FSET) for m in ORDER}
    OUT = CLUST / "statistics" / f"{FSET}_K{K}"
    print(f"figures for {FSET} at K={K}")

    summ = json.loads((OUT / "stats_summary.json").read_text()) \
        if (OUT / "stats_summary.json").exists() else {}
    sep = pd.read_csv(OUT / "stats_separation.csv")
    sizes = pd.read_csv(OUT / "stats_cluster_sizes.csv")
    lopo = pd.read_csv(OUT / "stats_lopo.csv") if (OUT / "stats_lopo.csv").exists() else None
    labs = np.load(OUT / "stats_labels.npy")
    is_gated = np.load(OUT / "stats_is_gated.npy")
    xyz = np.load(OUT / "stats_xyz.npy")
    X = np.load(RUNS["kmeans"] / "X_train.npy").astype(float)
    nb = X.shape[1] // len(CONDS)
    base_pct = 100 * float((~is_gated).mean())

    hv = None
    f = CLUST / "bsf_comparison" / "heldout_variance_ALL.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d.feature_set == FSET) & (d.scheme == "home")]
        hv = d if not d.empty else None

    made = []
    for i, m in enumerate(ORDER):
        if a.only and m not in a.only:
            continue
        lab = labs[i]
        made.append(fig_c3(m, X, lab, xyz, nb, hv, lopo, summ, sep, sizes))
        made.append(fig_c8(m, X, lab, is_gated, sizes, base_pct))
        print(f"  {LABEL[m]:<18} -> {made[-2].name} , {made[-1].name}")
    print(f"\nwrote {len(made)} figures -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
