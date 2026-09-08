#!/usr/bin/env python3
"""
00_paper2_figure4_panels.py - FIG 4, drawn from the tables 00_paper2_figure4_correspondence.py
already wrote. No permutations, no fitting: two seconds, so the layout can be argued with.

    python 00_paper2_figure4_panels.py
    python 00_paper2_figure4_panels.py --k 8 --weighting weighted --raw-jaccard

THE NUMBER ON THE FIGURE IS CHANCE-CORRECTED, and that is what makes the six panels
comparable. A Jaccard's chance level depends on the two cluster sizes - two clusters of
340 electrodes overlap by 0.09 for nothing, two of 130 by 0.055 - so a raw 0.45 against
a raw 0.28 is partly a statement about sizes. Each pair already has its own permutation
mean J0, so the figure plots

    adjusted overlap = (J - J0) / (1 - J0)

which is 0 at chance and 1 at identity for every pair whatever its size, the same
construction the adjusted Rand index uses. Raw Jaccards stay in the CSV and in the
caption, and --raw-jaccard draws them instead.

FOUR PANELS, ONE CLAIM EACH.

    A   the six correspondence matrices, drawn identically so they can be read against
        each other. Rows are side 1's clusters in their own id order - five of the six
        share a side 1, so a row means the same cluster across those panels - and
        columns are REORDERED so each matched partner sits on the diagonal. Colour is
        the centroid correlation that made the match; the number is the adjusted
        overlap that tests it. The two are independent, which is the whole design.

    B   adjusted overlap against match rank. The greedy order is meant to decay, and
        where it reaches 0 is where the two solutions stop agreeing.

    C   the headline: algorithms agree with each other more than representations do.

    D   which convex-NMF-on-HFA cluster survives which comparison. A dark row is a
        cluster everything else also found; a pale row is a cluster only this one found.

ONE-PATIENT CLUSTERS ARE MARKED, not silently dropped. A cluster held by a single
patient cannot be moved by the within-patient permutation, so its sd is zero and its z
undefined - the test working, not failing.

WRITES  outputs/paper_figures/FIG4_K08.png and FIG4_K08_caption.txt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                     # noqa: E402

OUT = ROOT / "outputs" / "clustering" / "paper_figures"
INK, MUTED, GREY, PALE = "#1b1b1b", "#6b6b6b", "#b9b9b9", "#ededed"
ALGO_C, FEAT_C = "#1f5f8b", "#a4501f"     # axis A, axis B

TITLE = {
    "cnmf_vs_kmeans_concat_hg":         ("convex NMF", "k-means"),
    "cnmf_vs_hierarchical_concat_hg":   ("convex NMF", "Ward"),
    "kmeans_vs_hierarchical_concat_hg": ("k-means", "Ward"),
    "cnmf_concat_hg_vs_concat_bands5":  ("HFA", "5 bands"),
    "cnmf_concat_hg_vs_concat_bands5z": ("HFA", "5 bands z"),
    "cnmf_concat_hg_vs_concat_rawds":   ("HFA", "15 bands"),
}
ORDER = list(TITLE)


def adjust(j, j0):
    """(J - J0) / (1 - J0): 0 at chance, 1 at identity, comparable across sizes."""
    return (j - j0) / np.maximum(1.0 - j0, 1e-9)


def patient_share(method, fset, k):
    """Largest share of any one patient in each cluster of a run's K = k partition."""
    try:
        rd = LR.newest_run(method, fset)
        lab = pd.read_csv(rd / "cluster_labels_by_k.csv")[f"k_{k}"].to_numpy()
        pid = pd.read_csv(rd / "labels.csv", usecols=["patient_id"]).patient_id.to_numpy()
    except Exception:
        return None
    out = np.zeros(k)
    for c in range(k):
        m = lab == c
        if m.any():
            out[c] = pd.Series(pid[m]).value_counts().iloc[0] / m.sum()
    return out


def load(k, weighting, raw):
    # the analysis writes its tables with the run id in the name, so this takes the
    # NEWEST set rather than a name it would have to be told
    cands = sorted(OUT.glob(f"FIG4_matched_K{k:02d}_run*.csv"))
    if not cands:
        cands = sorted(OUT.glob(f"FIG4_matched_K{k:02d}.csv"))
    if not cands:
        raise SystemExit(f"no FIG4_matched_K{k:02d}*.csv in {OUT} - run "
                         "00_paper2_figure4_correspondence.py first")
    tag = cands[-1].stem[len("FIG4_matched_"):]
    matched = pd.read_csv(OUT / f"FIG4_matched_{tag}.csv")
    matched = matched[matched.weighting == weighting].copy()
    matched["value"] = (matched.overlap if raw
                        else adjust(matched.overlap, matched.null_plain_mean))
    summary = pd.read_csv(OUT / f"FIG4_summary_{tag}.csv")
    summary = summary.set_index("comparison").loc[
        [c for c in ORDER if c in set(summary.comparison)]].reset_index()
    mats = {}
    for name in summary.comparison:
        d = pd.read_csv(OUT / f"FIG4_corr_{name}_{tag}.csv")
        M = np.full((d.cluster_1.max() + 1, d.cluster_2.max() + 1), np.nan)
        M[d.cluster_1, d.cluster_2] = d.r_centred
        mats[name] = M
    return matched, summary, mats, tag


# ---- panels -------------------------------------------------------------------
def panel_matrix(ax, M, rows, share1, share2, name, first_col, diag=None):
    k1, k2 = M.shape
    # BOTH axes follow the match rank: rows in the order the pairs were taken (which is
    # descending correlation), columns follow their partners. The diagonal then decays
    # from top-left, so a number's position says how strong its pairing was.
    rows = rows.sort_values("rank")
    r_ord = [int(r.cluster_1) for _, r in rows.iterrows()]
    c_ord = [int(r.cluster_2) for _, r in rows.iterrows()]
    r_ord += [i for i in range(k1) if i not in r_ord]
    cols = c_ord + [j for j in range(k2) if j not in c_ord]
    M = M[np.ix_(r_ord, cols)]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # THE BOX CARRIES SIGNIFICANCE, the number carries the size of the agreement.
    # Asterisks beside a two-decimal number do not fit in a cell this small, and a line
    # weight is read faster across six panels than a run of stars.
    for pos, (_, r) in enumerate(rows.iterrows()):
        i = pos                       # rows are now in match order, so the pair is (pos, pos)
        both = (r.p_plain < 0.05) and (r.p_within < 0.05)
        plain = (r.p_plain < 0.05) and not both
        st = (dict(lw=1.7, ls="-", edgecolor=INK) if both else
              dict(lw=1.2, ls=(0, (2.4, 1.4)), edgecolor=INK) if plain else
              dict(lw=0.8, ls="-", edgecolor=GREY))
        ax.add_patch(plt.Rectangle((pos - .5, i - .5), 1, 1, fill=False, **st))
        ax.text(pos, i, f"{r.value:.2f}".lstrip("0") or "0", ha="center", va="center",
                fontsize=6.0, color="white" if abs(M[pos, pos]) > .55 else INK)
    if len(cols) > len(c_ord):
        ax.axvline(len(c_ord) - .5, color=INK, lw=0.9, ls=(0, (3, 2)))

    a, b = TITLE[name]
    ttl = f"{a}  vs  {b}"
    if diag is not None and np.isfinite(diag[0]):
        ttl += f"\ndiagonal {diag[0]:.2f}   ·   random pairings {diag[1]:.2f}"
    ax.set_title(ttl, fontsize=9.0, color=INK, pad=4, linespacing=1.4)
    ax.set_xticks(range(k2)); ax.set_xticklabels([str(c) for c in cols], fontsize=5.8)
    ax.set_yticks(range(k1)); ax.set_yticklabels([str(i) for i in r_ord], fontsize=5.8)
    ax.tick_params(length=1.8, colors=MUTED, pad=1.5)
    ax.set_xlabel(f"{b} cluster", fontsize=7.2, color=MUTED, labelpad=1.5)
    if first_col:
        ax.set_ylabel(f"{a} cluster", fontsize=7.2, color=MUTED, labelpad=1.5)
    for sp in ax.spines.values():
        sp.set_color(GREY)
    for pos, i in enumerate(r_ord):
        if share1 is not None and share1[i] > 0.5:
            ax.plot(-0.78, pos, "o", ms=2.5, color=INK, clip_on=False)
    for pos, j in enumerate(cols):
        if share2 is not None and share2[j] > 0.5:
            ax.plot(pos, -0.78, "o", ms=2.5, color=INK, clip_on=False)
    return im


def panel_rank(ax, matched, summary, ylab):
    for name in summary.comparison:
        g = matched[matched.comparison == name].sort_values("rank")
        c = ALGO_C if g.axis.iloc[0].startswith("A") else FEAT_C
        a, b = TITLE[name]
        ax.plot(g["rank"], g.value, "-o", ms=3.2, lw=1.25, color=c, alpha=.85,
                label=f"{a} vs {b}")
    ax.axhline(0, color=GREY, lw=1.0, zorder=0)
    ax.text(7.45, 0.018, "chance", fontsize=6.8, color=MUTED, va="bottom", ha="left")
    ax.set_xlabel("match rank   (greedy, by centroid correlation)", fontsize=8, color=MUTED)
    ax.set_ylabel(ylab, fontsize=8, color=MUTED)
    ax.set_xticks(range(1, int(matched["rank"].max()) + 1))
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.legend(fontsize=6.3, frameon=False, loc="lower left", ncol=2, labelcolor=INK,
              handlelength=1.3, columnspacing=1.0, borderaxespad=0.1,
              bbox_to_anchor=(0.0, 0.02))


def panel_summary(ax, matched, summary):
    ys, vals, cols, labs, nsig = [], [], [], [], []
    for i, name in enumerate(summary.comparison[::-1]):
        g = matched[matched.comparison == name]
        s = summary[summary.comparison == name].iloc[0]
        ys.append(i)
        vals.append(float(g.value.mean()))
        cols.append(ALGO_C if g.axis.iloc[0].startswith("A") else FEAT_C)
        a, b = TITLE[name]
        labs.append(f"{a} vs {b}")
        nsig.append(int(((g.p_plain < .05) & (g.p_within < .05)).sum()))
    ax.barh(ys, vals, color=cols, height=.52, alpha=.9)
    for y, v, n, lb in zip(ys, vals, nsig, labs):
        ax.text(v + .008, y, f"{n}/8", va="center", fontsize=6.6, color=MUTED)
        ax.text(0.004, y + .40, lb, va="bottom", ha="left", fontsize=7, color=INK)
    ax.set_yticks([]); ax.set_ylim(-0.7, len(ys) - 0.15)
    ax.set_xlabel("mean adjusted overlap", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    ax.set_xlim(0, max(vals) * 1.22)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.set_title("agreement, all pairs", fontsize=8.8, color=INK, pad=5)
    ax.text(0.0, -0.62, "n/8 = pairs clearing both nulls", fontsize=6.3,
            color=MUTED, ha="left", va="center")


def panel_left_out(ax, matched, summary, k):
    names = [n for n in summary.comparison
             if n.startswith("cnmf_vs_") or n.startswith("cnmf_concat_hg_vs")]
    G = np.full((k, len(names)), np.nan)
    P = np.zeros((k, len(names)), bool)
    for c, name in enumerate(names):
        for _, r in matched[matched.comparison == name].iterrows():
            G[int(r.cluster_1), c] = r.value
            P[int(r.cluster_1), c] = (r.p_plain < .05) and (r.p_within < .05)
    im = ax.imshow(G, cmap="YlGnBu", vmin=0, vmax=np.nanmax(G), aspect="auto")
    for i in range(k):
        for c in range(len(names)):
            if np.isfinite(G[i, c]):
                ax.text(c, i, f"{G[i, c]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=6.2,
                        color="white" if G[i, c] > .55 * np.nanmax(G) else INK)
            if P[i, c]:
                ax.plot(c + .37, i - .35, "o", ms=2.0, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([TITLE[n][1] for n in names], fontsize=6.6, rotation=32, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels([f"c{i}" for i in range(k)], fontsize=6.6)
    ax.tick_params(length=1.8, colors=MUTED, pad=1.5)
    for sp in ax.spines.values():
        sp.set_color(GREY)
    ax.set_title("every convex-NMF-on-HFA cluster,\nagainst everything else",
                 fontsize=8.8, color=INK, pad=5)
    return im


# ---- caption ------------------------------------------------------------------
def caption(matched, summary, k, weighting, onepat, raw):
    unit = ("raw Jaccard" if raw else
            "adjusted overlap (J - J0)/(1 - J0), 0 at chance and 1 at identity")
    L = [f"FIG 4  Cluster correspondence at K = {k} ({weighting} weighting).", "",
         "Clusters from two solutions are PAIRED by the correlation between their mean "
         "5-band z-scored responses - each electrode z-scored across its own features "
         "first, and the mean across a solution's clusters removed - and the pairing is "
         "TESTED on the electrodes the paired clusters share. Nothing about a shape "
         "correlation forces two clusters to hold the same electrodes, so the overlap "
         "is evidence rather than a restatement of the matching.",
         "",
         "Every cluster is described in the same space whatever feature set produced "
         "it, so a clustering fitted on HFA and one fitted on 15 bands can be compared "
         "at all. The number on the figure is the " + unit + "; a Jaccard's chance "
         "level depends on the two cluster sizes, so raw Jaccards are not comparable "
         "between panels and adjusted ones are.",
         "",
         "A  Colour is the centroid correlation; columns are permuted so each matched "
         "partner lies on the diagonal. The box says how that pair stands against the "
         "nulls: solid above both, dashed above the free null only, grey not above "
         "chance (1000 draws, cluster sizes kept, the matching redone in every draw; "
         "the second null permutes only within patient). A dashed rule separates "
         "columns left unmatched. Rows are side 1's own cluster ids, so a row means the "
         "same cluster in every panel that shares a side 1.",
         "B  Adjusted overlap by match rank. C  Mean over all eight pairs, with the "
         "number clearing both nulls. D  Each convex-NMF-on-HFA cluster against the "
         "five comparisons it appears in; a dot marks a pair clearing both nulls.",
         ""]
    for _, s in summary.iterrows():
        a, b = TITLE[s.comparison]
        g = matched[matched.comparison == s.comparison]
        L.append(f"{a} vs {b}: matched r {s.r_matched_mean:+.2f} against "
                 f"{s.r_unmatched_mean:+.2f} for unmatched pairs; mean adjusted overlap "
                 f"{adjust(g.overlap, g.null_plain_mean).mean():.2f} (raw Jaccard "
                 f"{s[f'overlap_{weighting}_mean']:.2f}); "
                 f"{int(s[f'n_sig_plain_{weighting}'])} of {int(s.n_matched)} pairs "
                 f"above the free null, {int(s[f'n_sig_within_{weighting}'])} above the "
                 f"within-patient null. Runs {s.run_1} and {s.run_2}.")
    L.append("")
    if onepat:
        L.append("Dots mark clusters more than half held by one patient: "
                 + "; ".join(onepat) + ". The within-patient null cannot move a cluster "
                 "confined to a single patient, so its sd is zero and its z undefined; "
                 "such a pair is reported without a within-patient p rather than with a "
                 "p of 1.")
    L.append(f"Written {datetime.now():%Y-%m-%d %H:%M} from FIG4_matched_K{k:02d}.csv.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--weighting", choices=["hard", "weighted"], default="hard")
    ap.add_argument("--raw-jaccard", action="store_true",
                    help="draw raw Jaccards instead of the chance-corrected ones")
    a = ap.parse_args()

    matched, summary, mats, tag = load(a.k, a.weighting, a.raw_jaccard)
    ylab = "raw Jaccard" if a.raw_jaccard else "adjusted overlap"

    shares, onepat = {}, []
    for _, s in summary.iterrows():
        for side in (1, 2):
            key = (s[f"method_{side}"], s[f"feature_set_{side}"])
            shares.setdefault(key, patient_share(*key, a.k))
    for (meth, fset), sh in shares.items():
        for c in (np.flatnonzero(sh > 0.5) if sh is not None else []):
            onepat.append(f"{meth} on {fset} c{c} ({sh[c]:.0%})")

    fig = plt.figure(figsize=(11.4, 10.4), facecolor="white")
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.0, 1.0, 1.02],
                  hspace=0.46, wspace=1.05, left=.085, right=.940, top=.885, bottom=.105)

    im = None
    for idx, name in enumerate(summary.comparison):
        ax = fig.add_subplot(gs[idx // 3, 4 * (idx % 3):4 * (idx % 3) + 4])
        s = summary[summary.comparison == name].iloc[0]
        dg = ((float(s.diag_share), float(s.diag_null_mean))
              if "diag_share" in summary.columns else None)
        im = panel_matrix(ax, mats[name], matched[matched.comparison == name],
                          shares.get((s.method_1, s.feature_set_1)),
                          shares.get((s.method_2, s.feature_set_2)),
                          name, idx % 3 == 0, diag=dg)
    panel_rank(fig.add_subplot(gs[2, 0:5]), matched, summary, ylab)
    panel_summary(fig.add_subplot(gs[2, 5:8]), matched, summary)
    imD = panel_left_out(fig.add_subplot(gs[2, 8:12]), matched, summary, a.k)

    cax = fig.add_axes([0.700, 0.9385, 0.205, 0.0095])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, -.5, 0, .5, 1])
    cb.set_label("centroid correlation  (what pairs the clusters)", fontsize=6.8,
                 color=MUTED, labelpad=2)
    cb.ax.tick_params(labelsize=6, colors=MUTED, length=2)
    cb.outline.set_edgecolor(GREY)

    caxD = fig.add_axes([0.955, 0.135, 0.0085, 0.155])
    cbD = fig.colorbar(imD, cax=caxD, orientation="vertical")
    cbD.set_label(f"{ylab}  (what tests the pair)", fontsize=6.8, color=MUTED, labelpad=3)
    cbD.ax.tick_params(labelsize=6, colors=MUTED, length=2)
    cbD.outline.set_edgecolor(GREY)

    fig.text(.085, .955, "A", fontsize=11, color=INK, weight="bold")
    fig.text(.106, .9565, "clusters paired by response shape, tested on shared "
             "electrodes.  Number in each matched cell: " + ylab + ".",
             fontsize=8.6, color=INK)
    fig.text(.106, .9345, "box:  solid = above both nulls     dashed = above the free "
             "null only     grey = at chance          dot = one patient holds over half "
             "the cluster", fontsize=7.0, color=MUTED)
    if a.weighting == "weighted":
        # A SOFT COLUMN AGAINST A 0/1 COLUMN CANNOT REACH 1. Read the weighted rows
        # down a block, never across the two: the top row pairs a graded solution
        # with a hard one and the bottom row pairs two graded ones.
        fig.text(.106, .9135, "weighted: each electrode counts as P(belonging), and the "
                 "overlap is the weighted Jaccard.\nk-means and Ward have no P, so the "
                 "top row pairs a graded solution with a hard one - read down a block, "
                 "not across.",
                 fontsize=7.0, color=FEAT_C, linespacing=1.5, va="top")
    fig.text(.030, .755, "same features,\ndifferent algorithms", fontsize=8.2,
             color=ALGO_C, rotation=90, ha="center", va="center", linespacing=1.5)
    fig.text(.030, .490, "same algorithm,\ndifferent features", fontsize=8.2,
             color=FEAT_C, rotation=90, ha="center", va="center", linespacing=1.5)
    fig.text(.085, .327, "B", fontsize=11, color=INK, weight="bold")
    fig.text(.455, .327, "C", fontsize=11, color=INK, weight="bold")
    fig.text(.680, .327, "D", fontsize=11, color=INK, weight="bold")

    # THE FILENAME CARRIES THE VARIANT. Without this the weighted render overwrites
    # the hard one at the same path and the figure on disk stops saying which it is.
    sfx = ("" if a.weighting == "hard" else "_weighted") + ("_rawJ" if a.raw_jaccard else "")
    # the six runs are in the caption; the figure carries the one it is anchored to
    s0 = summary.iloc[0]
    fig.text(0.004, 0.004,
             f"side 1 run {s0.run_1}   \u00b7   K = {a.k}   \u00b7   "
             f"{a.weighting} weighting   \u00b7   drawn "
             f"{datetime.now():%Y-%m-%d %H:%M}",
             fontsize=6.0, color=MUTED, ha="left", va="bottom")
    png = OUT / f"FIG4_{tag}{sfx}.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    txt = OUT / f"FIG4_{tag}{sfx}_caption.txt"
    body = caption(matched, summary, a.k, a.weighting, onepat, a.raw_jaccard)
    txt.write_text(body, encoding="utf-8")
    if txt.read_text(encoding="utf-8") != body:
        raise SystemExit(f"{txt} did not survive the write")
    print(f"  {png}\n  {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
