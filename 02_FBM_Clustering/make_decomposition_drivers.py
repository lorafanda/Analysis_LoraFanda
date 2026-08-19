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
    return X, G, Gn, comp, L, rd.parent.parent.name


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
def figure_membership(out_png, Gn, comp, L, tag, run_id):
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

    fig = plt.figure(figsize=(11.0, 7.2), dpi=200)
    # A's colourbar goes UNDER A, not in a column between B and C: there it collided
    # with C's y-axis label and its own annotation had nowhere to sit.
    gs = GridSpec(3, 3, width_ratios=[16, 5, 12], height_ratios=[1, 1, 0.05],
                  wspace=0.34, hspace=0.60, left=0.085, right=0.965,
                  top=0.795, bottom=0.075)

    # ---- (a) the membership matrix
    ax = fig.add_subplot(gs[0:2, 0])
    im = ax.imshow(M, aspect="auto", cmap=CMAP_LOAD, vmin=0, vmax=1,
                   interpolation="nearest")
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="#2bb3c0", lw=1.0)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"c{j}" for j in range(K)], fontsize=7.5)
    ax.set_xlabel("weight on each component", fontsize=8.5)
    ax.set_ylabel(f"{n} electrodes, grouped by argmax then top weight",
                  fontsize=8.5, labelpad=22)
    ax.set_yticks([])
    ax.tick_params(length=2, width=0.6, colors=MUTED)
    for j in range(K):
        ax.text(-0.60, (bounds[j] + bounds[j + 1]) / 2, f"c{j}", fontsize=7.5,
                ha="right", va="center", color=INK)
    ax.set_title("A — every electrode's mixture, not its membership",
                 fontsize=10, loc="left", color=INK, pad=6)
    cax = fig.add_subplot(gs[2, 0])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("weight on a component (each row sums to 1)", fontsize=7.5,
                 labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.ax.axvline(uniform, color="#2bb3c0", lw=1.4)
    cb.ax.text(uniform, 1.9, f"{uniform:.2f} = even split over {K}", fontsize=6.2,
               ha="center", va="bottom", color="#2bb3c0",
               transform=cb.ax.get_xaxis_transform())

    # ---- (b) how confident each of those assignments is
    axm = fig.add_subplot(gs[0:2, 1], sharey=ax)
    axm.barh(np.arange(n), mg, height=1.0, color="#4a6fa5", linewidth=0)
    axm.axvline(0.05, color="#c1121f", lw=0.9, ls="--")
    axm.set_xlim(0, max(0.6, float(np.percentile(margin, 99))))
    axm.invert_yaxis()
    axm.set_xlabel("1st − 2nd weight", fontsize=8.5)
    axm.tick_params(labelsize=6.5, length=2, colors=MUTED)
    plt.setp(axm.get_yticklabels(), visible=False)
    for sp in ("top", "right"):
        axm.spines[sp].set_visible(False)
    near = int((margin < 0.05).sum())
    axm.set_title(f"B — {near} electrodes ({100*near/n:.0f}%)\nare near-ties",
                  fontsize=10, loc="left", color=INK, pad=6)
    axm.text(0.05, 0.012, " 0.05", transform=axm.get_xaxis_transform(),
             fontsize=6, color="#c1121f", ha="left", va="bottom")

    # ---- (c) how much of the weight the winner actually takes
    axc = fig.add_subplot(gs[0, 2])
    ts = np.linspace(uniform, 1.0, 200)
    frac = [(top > t).mean() * 100 for t in ts]
    axc.plot(ts, frac, color=INK, lw=1.6)
    axc.axvline(uniform, color="#2bb3c0", lw=1.0)
    axc.axvline(0.5, color="#c1121f", lw=1.0, ls="--")
    axc.text(uniform, 102, f" even split {uniform:.2f}", fontsize=6.5,
             color="#2bb3c0", ha="left", va="bottom")
    axc.text(0.5, 102, " majority 0.50", fontsize=6.5, color="#c1121f",
             ha="left", va="bottom")
    maj = (top > 0.5).mean() * 100
    axc.plot([0.5], [maj], "o", ms=4.5, color="#c1121f")
    axc.annotate(f"{maj:.0f}% of electrodes have a\nmajority component",
                 xy=(0.5, maj), xytext=(0.60, min(88, maj + 26)),
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
    axk = fig.add_subplot(gs[1, 2])
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
                 x=0.085, y=0.975, ha="left", fontsize=13.5, color=INK)
    fig.text(0.085, 0.935,
             "Convex NMF fits K additive profiles and gives every electrode a weight on each: "
             "X ≈ G(W′X). It never partitions anything.\n"
             "The cluster label used everywhere else is an argmax over these weights, taken "
             "afterwards — panels A and B are what that step discards.",
             fontsize=8.4, color=MUTED, va="top")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(median_top=float(np.median(top)), pct_majority=float(maj),
                near_ties=int(near))


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
        X, G, Gn, comp, L, feature_set = load(rd)
        grid = None if LC._is_line_feature_set(feature_set) else grid_for(rd, X.shape[1])
        out = rd / "decomposition_drivers"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n  {tag}/{rd.name}: X {X.shape}, G {G.shape}")

        s = figure_membership(out / "D1_membership.png", Gn, comp, L, tag, rd.name)
        print(f"    D1: median top weight {s['median_top']:.2f}, "
              f"{s['pct_majority']:.0f}% with a majority component, "
              f"{s['near_ties']} near-ties")

        ag = figure_drivers(out / "D2_drivers.png", X, Gn, comp, L, tag, rd.name,
                            feature_set, grid)
        print(f"    D2: loading-r vs cluster-d agreement "
              f"{min(ag):.2f}-{max(ag):.2f} across {len(ag)} components")
        print(f"    -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
