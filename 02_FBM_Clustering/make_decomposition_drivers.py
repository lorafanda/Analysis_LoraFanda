#!/usr/bin/env python3
"""
make_decomposition_drivers.py - how convex NMF assigns electrodes, and what each
component is actually made of.

THIS SCRIPT OWNS ONE FOLDER: decomposition_drivers/, inside each cnmf run.

It exists because "which cluster is this electrode in" is the wrong question to
ask of a convex NMF, and the published figures do not make that obvious. cNMF
fits K additive profiles and gives every electrode a weight on each one:

    X  ~  G (W' X)          G >= 0, W >= 0

Nothing partitions anything. The cluster label used everywhere downstream is an
ARGMAX taken over G afterwards, outside the model, and that step is where the
information is lost. On the published concat_hg run the median electrode puts
only 0.43 of its weight on its own component, and for a tenth of electrodes the
top two components are within 0.023 of each other - those labels are close to
coin flips.

Two figures:

  D1  the membership structure. Every electrode's weights, sorted, so the
      question "is this a partition or a continuum" is answered by looking. Plus
      how confident each assignment is, and how similar the components are to
      each other - if two profiles are near mirror images then treating them as
      separate response types is a choice, not a finding.

  D2  what drives each component, three columns per component:
        the profile itself   - what the component IS, in feature space
        loading correlation  - across electrodes, r between the weight on this
                               component and the value of each feature. Never
                               passes through the argmax, so it describes what
                               the model fitted.
        cluster vs rest      - Cohen's d between the electrodes the argmax gave
                               this component and every other electrode. This is
                               what a hard clustering would report.

      The last two answer the same question by different routes, and on this data
      they agree at r = 0.97-0.99 per component. That is worth knowing rather than
      assuming: the argmax is unreliable for any INDIVIDUAL electrode and still
      recovers the same feature signature for the component as a whole.

    python make_decomposition_drivers.py --dry-run
    python make_decomposition_drivers.py
    python make_decomposition_drivers.py --run <run dir>
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
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_centroids as LC  # noqa: E402   (read only)

CLUST = ROOT / "outputs" / "clustering"
COND_NAMES = ("audio", "picture", "reading")

# dB keeps the project's bwr. The driver maps are NOT dB - one is a correlation,
# the other an effect size - so they take a different diverging hue on purpose,
# to stop the reader carrying the dB scale across to them.
CMAP_DB = "bwr"
CMAP_DRIVER = "PRGn"
CMAP_LOAD = "magma_r"

INK = "#1b232c"
MUTED = "#68727d"
# The two populations, once the responsiveness gate is lifted. Grey is deliberate:
# the added electrodes are the ones the gate calls non-responsive, so they should
# read as inactive rather than as a second category of interest.
GATED_COL = "#4a6fa5"
ADDED_COL = "#b0b7be"
# FIXED so the gated and ungated figures can be read side by side. Covers the 99th
# percentile of both runs (0.725 gated, 0.582 ungated) with room to spare.
MARGIN_XLIM = 0.80


def cnmf_runs(a):
    """Every cnmf run with the loadings this figure needs, newest per feature set."""
    idx = json.loads((CLUST / "index.json").read_text(encoding="utf-8"))
    runs = idx["runs"] if isinstance(idx, dict) else idx
    best = {}
    for r in runs:
        if r["method"] != "cnmf":
            continue
        rd = CLUST / r["method"] / r["feature_set"] / "runs" / r["run_id"]
        if not (rd / "G_loadings.npy").exists() or not (rd / "X_train.npy").exists():
            continue
        k = r["feature_set"]
        if k not in best or r["run_id"] > best[k][0]:
            best[k] = (r["run_id"], rd)
    out = [(f"cnmf/{f}", rd) for f, (_, rd) in sorted(best.items())]
    if a.run:
        want = Path(a.run).resolve()
        out = [(t, rd) for t, rd in out if rd.resolve() == want] or \
              [(f"cnmf/{Path(a.run).parent.parent.name}", Path(a.run))]
    return out


def load(rd: Path):
    X = np.load(rd / "X_train.npy")
    G = np.load(rd / "G_loadings.npy")
    comp = np.load(rd / "components.npy") if (rd / "components.npy").exists() else None
    lab = pd.read_csv(rd / "labels.csv")
    ccol = next(c for c in lab.columns
                if c.startswith("cluster_") and not c.endswith("_ranked"))
    L = pd.to_numeric(lab[ccol], errors="coerce").to_numpy()
    # Row-normalised: raw G is not comparable between electrodes, because a loud
    # electrode loads higher on everything. Normalised, a row is a mixture that
    # sums to 1 and "0.43" means the same thing on any electrode.
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    # n_high_activity rides along in labels.csv: >0 means the electrode would have
    # passed the responsiveness gate. On a gated run every row is True, which is
    # correct rather than a special case.
    gate = None
    if "n_high_activity" in lab.columns:
        gate = pd.to_numeric(lab["n_high_activity"], errors="coerce").fillna(0).to_numpy() > 0
    return X, G, Gn, comp, L, rd.parent.parent.name, gate


def grid_for(rd: Path, n_features: int):
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


def drivers(X, Gn, L, j):
    """(loading correlation, cluster-vs-rest Cohen's d) per feature, for component j."""
    Xz = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    g = Gn[:, j]
    gz = (g - g.mean()) / max(g.std(), 1e-9)
    r = (Xz * gz[:, None]).mean(0)

    m = L == j
    a, b = X[m], X[~m]
    if len(a) < 2 or len(b) < 2:
        return r, np.zeros_like(r)
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1))
                 / (len(a) + len(b) - 2))
    d = (a.mean(0) - b.mean(0)) / np.maximum(sp, 1e-9)
    return r, d


def _blocks(ax, n_x, n_blocks, *, labels_on, heatmap=True):
    """Condition seams and the GO cue dash, as lf_centroids draws them."""
    per = n_x / max(n_blocks, 1)
    off = 0.5 if heatmap else 0.0
    for b in range(1, max(n_blocks, 1)):
        ax.axvline(b * per - off, color="k", lw=0.9, zorder=5)
    for b in range(max(n_blocks, 1)):
        ax.axvline((b + 0.5) * per - off, color="#3a3a3a", lw=0.7,
                   ls=(0, (4, 3)), alpha=0.75, zorder=6)
    if labels_on and n_blocks > 1:
        ax.set_xticks([(b + 0.5) * per for b in range(n_blocks)])
        ax.set_xticklabels(list(COND_NAMES)[:n_blocks], fontsize=7)
    else:
        ax.set_xticks([])


# ==============================================================================
# D1 - membership structure
# ==============================================================================
def figure_membership(out_png, Gn, comp, L, tag, run_id, gate=None):
    """Membership structure. `gate` is a boolean per electrode: True = the electrode
    would have passed the responsiveness gate, False = it is only here because the
    gate was lifted. When it carries both classes the figure splits by it.

    EVERY DATA AXIS IS FIXED (see the module constants), not fitted to the run, so
    this figure can be read side by side with the gated one. The margin axis used to
    be max(0.6, p99), which is exactly the kind of per-run fitting that makes two
    panels look comparable while quietly using different scales.
    """
    n, K = Gn.shape
    top = Gn.max(1)
    srt = np.sort(Gn, 1)
    margin = srt[:, -1] - srt[:, -2]
    lead = Gn.argmax(1)
    # sort by assigned component, then by how strongly it is assigned
    order = np.lexsort((-top, lead))
    M, mg, ld = Gn[order], margin[order], lead[order]
    bounds = np.searchsorted(ld, np.arange(K + 1))
    uniform = 1.0 / K

    if gate is None:
        gate = np.ones(n, dtype=bool)
    gate = np.asarray(gate, dtype=bool)
    split = bool(gate.any() and (~gate).any())
    g_ord = gate[order]

    # The header is up to five lines and sits above the panel TITLES, which need
    # their own pad on top of that. At top=0.795 it landed on the B and C titles.
    fig = plt.figure(figsize=(11.4, 8.6), dpi=200)
    # A's colourbar goes UNDER A, not in a column between B and C: there it collided
    # with C's y-axis label and its own annotation had nowhere to sit. The extra narrow
    # first column is the gate strip.
    gs = GridSpec(3, 4, width_ratios=[0.55, 16, 5, 12], height_ratios=[1, 1, 0.05],
                  wspace=0.30, hspace=0.60, left=0.075, right=0.965,
                  top=0.745, bottom=0.065)

    # ---- the gate strip, in the same row order as A
    axg = fig.add_subplot(gs[0:2, 0])
    axg.imshow(np.where(g_ord, 1, 0).reshape(-1, 1), aspect="auto",
               cmap=ListedColormap([ADDED_COL, GATED_COL]), vmin=0, vmax=1,
               interpolation="nearest")
    axg.set_xticks([]); axg.set_yticks([])
    for sp in axg.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#c8cfd6")
    for j in range(K):
        axg.text(-1.1, (bounds[j] + bounds[j + 1]) / 2, f"c{j}", fontsize=7.5,
                 ha="right", va="center", color=INK)
    axg.set_ylabel(f"{n} electrodes, grouped by argmax then top weight",
                   fontsize=8.5, labelpad=26)

    # ---- (a) the membership matrix
    ax = fig.add_subplot(gs[0:2, 1], sharey=axg)
    im = ax.imshow(M, aspect="auto", cmap=CMAP_LOAD, vmin=0, vmax=1,
                   interpolation="nearest")
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="#2bb3c0", lw=1.0)
        axg.axhline(b - 0.5, color="#2bb3c0", lw=1.0)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"c{j}" for j in range(K)], fontsize=7.5)
    ax.set_xlabel("weight on each component", fontsize=8.5)
    ax.set_yticks([])
    ax.tick_params(length=2, width=0.6, colors=MUTED)
    ax.set_title("A — every electrode's mixture, not its membership",
                 fontsize=10, loc="left", color=INK, pad=6)
    cax = fig.add_subplot(gs[2, 1])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("weight on a component (each row sums to 1)", fontsize=7.5,
                 labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.ax.axvline(uniform, color="#2bb3c0", lw=1.4)
    cb.ax.text(uniform, 1.9, f"{uniform:.2f} = even split over {K}", fontsize=6.2,
               ha="center", va="bottom", color="#2bb3c0",
               transform=cb.ax.get_xaxis_transform())

    # ---- (b) how confident each of those assignments is
    axm = fig.add_subplot(gs[0:2, 2], sharey=axg)
    axm.barh(np.arange(n), mg, height=1.0, linewidth=0,
             color=np.where(g_ord, GATED_COL, ADDED_COL))
    axm.axvline(0.05, color="#c1121f", lw=0.9, ls="--")
    axm.set_xlim(0, MARGIN_XLIM)
    axm.invert_yaxis()
    axm.set_xlabel("1st − 2nd weight", fontsize=8.5)
    axm.tick_params(labelsize=6.5, length=2, colors=MUTED)
    plt.setp(axm.get_yticklabels(), visible=False)
    for sp in ("top", "right"):
        axm.spines[sp].set_visible(False)
    near = int((margin < 0.05).sum())
    if split:
        n_g = int((margin[gate] < 0.05).sum())
        n_a = int((margin[~gate] < 0.05).sum())
        # B is narrow, so the breakdown goes INSIDE it: as a second title line it
        # overran into C's title.
        axm.text(0.97, 0.995,
                 "%.0f%% of gated" % (100*n_g/max(gate.sum(), 1)),
                 transform=axm.transAxes, fontsize=6.8, color=GATED_COL,
                 ha="right", va="top")
        axm.text(0.97, 0.975,
                 "%.0f%% of added" % (100*n_a/max((~gate).sum(), 1)),
                 transform=axm.transAxes, fontsize=6.8, color="#8e969e",
                 ha="right", va="top")
    axm.set_title(f"B — {near} near-ties ({100*near/n:.0f}%)",
                  fontsize=10, loc="left", color=INK, pad=6)
    axm.text(0.05, 0.012, " 0.05", transform=axm.get_xaxis_transform(),
             fontsize=6, color="#c1121f", ha="left", va="bottom")

    # ---- (c) how much of the weight the winner actually takes
    axc = fig.add_subplot(gs[0, 3])
    ts = np.linspace(uniform, 1.0, 200)
    axc.plot(ts, [(top > t).mean() * 100 for t in ts], color=INK, lw=1.8,
             label="all" if split else None, zorder=5)
    if split:
        axc.plot(ts, [(top[gate] > t).mean() * 100 for t in ts], color=GATED_COL,
                 lw=1.4, label=f"passes gate (n={int(gate.sum())})")
        axc.plot(ts, [(top[~gate] > t).mean() * 100 for t in ts], color=ADDED_COL,
                 lw=1.4, label=f"added by ungating (n={int((~gate).sum())})")
        axc.legend(fontsize=6.4, frameon=False, loc="upper right")
    axc.axvline(uniform, color="#2bb3c0", lw=1.0)
    axc.axvline(0.5, color="#c1121f", lw=1.0, ls="--")
    axc.text(uniform, 102, f" even split {uniform:.2f}", fontsize=6.5,
             color="#2bb3c0", ha="left", va="bottom")
    axc.text(0.5, 102, " majority 0.50", fontsize=6.5, color="#c1121f",
             ha="left", va="bottom")
    maj = (top > 0.5).mean() * 100
    axc.plot([0.5], [maj], "o", ms=4.5, color="#c1121f", zorder=6)
    axc.annotate(f"{maj:.0f}% have a\nmajority component",
                 xy=(0.5, maj), xytext=(0.58, min(74, maj + 22)),
                 fontsize=7, color=INK,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED))
    axc.set_xlim(uniform - 0.02, 1.0)
    axc.set_ylim(0, 108)
    axc.set_xlabel("top weight threshold", fontsize=8.5)
    axc.set_ylabel("% of electrodes above it", fontsize=8.5)
    axc.tick_params(labelsize=7, length=2, colors=MUTED)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    axc.set_title(f"C — median top weight {np.median(top):.2f}",
                  fontsize=10, loc="left", color=INK, pad=6)

    # ---- (d) are the components even distinct from each other
    axk = fig.add_subplot(gs[1, 3])
    wi = wj = None
    if comp is not None:
        Cn = comp / np.maximum(np.linalg.norm(comp, axis=1, keepdims=True), 1e-12)
        R = Cn @ Cn.T
        imk = axk.imshow(R, cmap=CMAP_DRIVER, vmin=-1, vmax=1)
        axk.set_xticks(range(K)); axk.set_yticks(range(K))
        axk.set_xticklabels([f"c{j}" for j in range(K)], fontsize=7)
        axk.set_yticklabels([f"c{j}" for j in range(K)], fontsize=7)
        iu = np.triu_indices(K, 1)
        worst = int(np.argmin(R[iu]))
        wi, wj = iu[0][worst], iu[1][worst]
        for i in range(K):
            for j2 in range(K):
                if i == j2:
                    continue
                axk.text(j2, i, f"{R[i, j2]:+.2f}", ha="center", va="center",
                         fontsize=5.4,
                         color="k" if abs(R[i, j2]) < 0.55 else "w")
        axk.add_patch(plt.Rectangle((wj - .5, wi - .5), 1, 1, fill=False,
                                    edgecolor="#c1121f", lw=1.6))
        axk.add_patch(plt.Rectangle((wi - .5, wj - .5), 1, 1, fill=False,
                                    edgecolor="#c1121f", lw=1.6))
        cbk = fig.colorbar(imk, ax=axk, fraction=0.045, pad=0.03)
        cbk.set_label("r between profiles", fontsize=7)
        cbk.ax.tick_params(labelsize=6.5, length=2)
        axk.set_title(f"D — c{wi} and c{wj} are near mirror images "
                      f"(r = {R[wi, wj]:+.2f})",
                      fontsize=10, loc="left", color=INK, pad=6)
        axk.tick_params(length=0)

    fig.suptitle(f"How the decomposition assigns electrodes — {tag} · {run_id}",
                 x=0.075, y=0.975, ha="left", fontsize=13.5, color=INK)
    head = ["Convex NMF fits K additive profiles and gives every electrode a weight on "
            "each: X ≈ G(W′X). It never partitions anything.",
            "The cluster label used everywhere else is an argmax over these weights, "
            "taken afterwards — panels A and B are what that step discards."]
    if split:
        head.append("")
        head.append(f"The strip left of A marks the two populations: "
                    f"{int(gate.sum())} electrodes that would pass the responsiveness "
                    f"gate (blue) and {int((~gate).sum())} that only appear because it "
                    f"was lifted (grey).")
        head.append("Every axis here is fixed to the same range as the gated figure, so "
                    "the two can be read side by side.")
    fig.text(0.075, 0.945, "\n".join(head), fontsize=8.4, color=MUTED, va="top",
             linespacing=1.45)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    out = dict(median_top=float(np.median(top)), pct_majority=float(maj),
               near_ties=int(near), n=int(n), K=int(K))
    if wi is not None:
        out["worst_pair"] = [int(wi), int(wj)]
        out["worst_r"] = float(R[wi, wj])
    if split:
        head.append("")
        head.append(f"The strip left of A marks the two populations: "
                    f"{int(gate.sum())} electrodes that would pass the "
                    f"responsiveness gate (blue) and {int((~gate).sum())} that only")
        head.append("appear because it was lifted (grey). Every axis is fixed to the "
                    "same range as the gated figure, so the two read side by side.")
    else:
        # Pad so both figures keep the SAME geometry - the point of the pair is that
        # they are read against each other.
        head += ["", "", ""]
    if split:
        out["gate"] = dict(
            n_gated=int(gate.sum()), n_added=int((~gate).sum()),
            median_top_gated=float(np.median(top[gate])),
            median_top_added=float(np.median(top[~gate])),
            median_margin_gated=float(np.median(margin[gate])),
            median_margin_added=float(np.median(margin[~gate])),
            pct_majority_gated=float((top[gate] > 0.5).mean() * 100),
            pct_majority_added=float((top[~gate] > 0.5).mean() * 100),
            pct_nearties_gated=float((margin[gate] < 0.05).mean() * 100),
            pct_nearties_added=float((margin[~gate] < 0.05).mean() * 100))
    return out


# ==============================================================================
# D2 - what drives each component
# ==============================================================================
def figure_drivers(out_png, X, Gn, comp, L, tag, run_id, feature_set, grid):
    K = Gn.shape[1]
    line = LC._is_line_feature_set(feature_set)
    nblk = LC._n_condition_blocks(feature_set)

    R, D = [], []
    for j in range(K):
        r, d = drivers(X, Gn, L, j)
        R.append(r); D.append(d)
    R, D = np.array(R), np.array(D)
    # components.npy is stored as a UNIT-NORM DIRECTION (|c| 0.52-0.72, values
    # +/-0.05), not a dB profile - drawn on a dB axis it is a flat line. It is also
    # not the same thing as what its electrodes look like: the two agree at r = 0.40
    # to 0.97 depending on the component. So column 1 shows the loading-weighted mean
    # of the data itself, which is in dB and is what "this component looks like"
    # actually means.
    P = (Gn.T @ X) / np.maximum(Gn.sum(0)[:, None], 1e-12)
    agree = [float(np.corrcoef(R[j], D[j])[0, 1]) for j in range(K)]

    # One limit per column across all components, so the rows compare.
    rlim = float(np.ceil(np.percentile(np.abs(R), 99) * 20) / 20)
    dlim = float(np.ceil(np.percentile(np.abs(D), 99) * 4) / 4)
    plim = float(np.ceil(np.percentile(np.abs(P), 99) * 4) / 4)

    rowh = 0.92 if line else 1.05
    # HEAD is the header strip in inches, held constant while the panel stack grows
    # with K, so the caption never eats into the first row.
    HEAD = 2.35
    H = rowh * K + HEAD + 0.7
    fig = plt.figure(figsize=(11.0, H), dpi=200)
    gs = GridSpec(K, 4, width_ratios=[10, 10, 10, 0.5], wspace=0.22, hspace=0.34,
                  left=0.085, right=0.945,
                  top=1 - HEAD / H, bottom=0.55 / H)
    imr = imd = imp = None
    for j in range(K):
        n_j = int((L == j).sum())
        for col, (mat, lim, cmap, ttl) in enumerate((
                (P[j], plim, CMAP_DB, "profile"),
                (R[j], rlim, CMAP_DRIVER, "loading r"),
                (D[j], dlim, CMAP_DRIVER, "cluster d"))):
            ax = fig.add_subplot(gs[j, col])
            if mat is None:
                ax.axis("off"); continue
            if line:
                x = np.arange(mat.size)
                ax.axhline(0, color="#bbb", lw=0.5)
                ax.plot(x, mat, lw=1.0,
                        color=INK if col == 0 else ("#5b2c83" if col == 1 else "#1b7837"))
                ax.set_ylim(-lim, lim)
                ax.set_xlim(0, mat.size - 1)
                _blocks(ax, mat.size, nblk, labels_on=(j == K - 1), heatmap=False)
                ax.set_yticks([-lim, 0, lim])
                ax.set_yticklabels([f"{-lim:g}", "0", f"{lim:g}"], fontsize=6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            else:
                nb, nc, nt = len(grid[0]), len(grid[1]), len(grid[2])
                im = ax.imshow(mat.reshape(nb, nc * nt), aspect="auto",
                               cmap=cmap, vmin=-lim, vmax=lim, origin="lower",
                               interpolation="nearest")
                if col == 0: imp = im
                elif col == 1: imr = im
                else: imd = im
                _blocks(ax, nc * nt, nc, labels_on=(j == K - 1))
                tick = sorted({0, nb // 2, nb - 1})
                ax.set_yticks(tick if col == 0 else [])
                if col == 0:
                    ax.set_yticklabels([grid[0][t] for t in tick], fontsize=5.6)
            ax.tick_params(length=2, width=0.5, colors=MUTED, labelsize=6)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_color("#c8cfd6")
            if j == 0:
                ax.set_title(["mean of its electrodes, weighted by loading (dB)",
                              "loading correlation  (no argmax)",
                              "cluster vs rest  (Cohen's d)"][col],
                             fontsize=9, loc="left", color=INK, pad=5)
            if col == 0:
                ax.set_ylabel(f"c{j}\nn={n_j}", fontsize=7.5, rotation=0,
                              ha="right", va="center", labelpad=20, color=INK)
            if col == 2:
                ax.text(1.015, 0.5, f"r={agree[j]:.2f}", transform=ax.transAxes,
                        fontsize=6.2, color=MUTED, ha="left", va="center")

    if not line:
        for im, lab_, row in ((imp, "dB", 0), (imr, "r", 1), (imd, "d", 2)):
            if im is None:
                continue
            cax = fig.add_subplot(gs[min(row, K - 1), 3])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(lab_, fontsize=7)
            cb.ax.tick_params(labelsize=6, length=2)

    fig.suptitle(f"What each component is made of — {tag} · {run_id}",
                 x=0.085, y=1 - 0.30 / H, ha="left", fontsize=13.5, color=INK)
    # Wrapped by hand. bbox_inches="tight" grows the canvas to whatever this text
    # needs, so one long unwrapped line turned an 11-inch figure into 4085 px of
    # mostly-empty width.
    blurb = "\n".join([
        "Left    the mean of this component's electrodes, each weighted by its loading —",
        "        what the component looks like, in dB.",
        "Middle  across electrodes, the correlation between the weight on this component and",
        "        each feature. Never passes through the argmax, so it describes what the model fitted.",
        "Right   Cohen's d between the electrodes the argmax gave this component and all the",
        "        others — what a hard clustering would report.",
        "",
        f"The two right columns answer the same question by different routes and agree at",
        f"r = {min(agree):.2f}–{max(agree):.2f} (printed per row). The argmax is unreliable for any individual",
        "electrode — see D1 — and still recovers the component's feature signature.",
    ])
    fig.text(0.085, 1 - 0.62 / H, blurb,
             fontsize=8.0, color=MUTED, va="top", linespacing=1.35)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return agree


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    targets = cnmf_runs(a)
    if not targets:
        print("  no cnmf run with G_loadings.npy", file=sys.stderr)
        return 1
    print(f"  {len(targets)} run(s):")
    for tag, rd in targets:
        print(f"    {tag}/{rd.name}")
    if a.dry_run:
        print("  (dry run)")
        return 0

    for tag, rd in targets:
        X, G, Gn, comp, L, feature_set, gate = load(rd)
        grid = None if LC._is_line_feature_set(feature_set) else grid_for(rd, X.shape[1])
        out = rd / "decomposition_drivers"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n  {tag}/{rd.name}: X {X.shape}, G {G.shape}")

        s = figure_membership(out / "D1_membership.png", Gn, comp, L, tag, rd.name,
                              gate=gate)
        print(f"    D1: median top weight {s['median_top']:.2f}, "
              f"{s['pct_majority']:.0f}% with a majority component, "
              f"{s['near_ties']} near-ties")
        if "gate" in s:
            g = s["gate"]
            print(f"        gated  n={g['n_gated']:<5d} median top {g['median_top_gated']:.3f}"
                  f"  majority {g['pct_majority_gated']:.0f}%"
                  f"  near-ties {g['pct_nearties_gated']:.0f}%")
            print(f"        added  n={g['n_added']:<5d} median top {g['median_top_added']:.3f}"
                  f"  majority {g['pct_majority_added']:.0f}%"
                  f"  near-ties {g['pct_nearties_added']:.0f}%")
        (out / "membership_stats.json").write_text(json.dumps(s, indent=2),
                                                   encoding="utf-8")

        ag = figure_drivers(out / "D2_drivers.png", X, Gn, comp, L, tag, rd.name,
                            feature_set, grid)
        print(f"    D2: loading-r vs cluster-d agreement "
              f"{min(ag):.2f}-{max(ag):.2f} across {len(ag)} components")
        print(f"    -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
