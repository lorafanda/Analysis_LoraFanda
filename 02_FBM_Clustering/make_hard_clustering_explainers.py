#!/usr/bin/env python3
"""
make_hard_clustering_explainers.py - k-means and Ward on the SAME twelve electrodes
that make_cnmf_explainer uses, so the three methods can be read side by side.

Same data, imported from make_cnmf_explainer.build(): twelve invented electrodes, forty
time points, three planted response shapes, each electrode a known mixture plus noise.

THE ONE DIFFERENCE THAT MATTERS. Convex NMF unit-norms each electrode before fitting.
k-means and Ward, as this project runs them, fit on RAW dB - so loudness is part of the
distance. The synthetic electrodes deliberately vary in loudness as well as in shape,
and that is enough to change an answer:

    planted dominant shape   0 0 0 0 1 1 1 1 2 2 2 2
    k-means on raw dB        0 0 0 0 1 0 1 1 2 2 2 2     <- e5 in the wrong group
    Ward on raw dB           0 0 0 0 1 0 1 1 2 2 2 2     <- same, e5
    k-means on unit-norm     0 0 0 0 1 1 1 1 2 2 2 2     <- correct

e5 is quiet (vector norm 2.5) and mixed. In raw dB it is CLOSEST to the quiet group
even though its SHAPE correlates best with its own - the figure prints both numbers
side by side so the disagreement is visible rather than asserted. Unit-norming removes
the loudness and the error goes away. That is FIG C.7 on twelve electrodes.

Honest caveat, stated on both figures: in this toy set loudness and shape are correlated
by construction (r = 0.82), so the raw methods get most assignments right for the wrong
reason. e5 is the case where the two cues disagree, which is why it is the informative
one.

    python make_hard_clustering_explainers.py
"""
from __future__ import annotations

import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import lf_decompose as LD
from make_cnmf_explainer import build, N_ELEC, K

OUT = ROOT / "outputs" / "clustering" / "explainers"

INK, MUTED = "#1b232c", "#68727d"
COMP_COL = ["#5b2c83", "#1f77b4", "#2a9d5c"]
GREY, RED = "#c9ced4", "#c1121f"
MONO = {"family": "DejaVu Sans Mono", "size": 6.6}

PICK = 5  # the electrode where loudness and shape disagree


def align(lab, dom):
    """Relabel clusters to the planted shape they mostly contain, for readable colour."""
    out = np.full_like(lab, -1)
    used = set()
    for c in range(K):
        m = lab == c
        if not m.any():
            continue
        for cand in np.bincount(dom[m], minlength=K).argsort()[::-1]:
            if cand not in used:
                out[m] = cand
                used.add(cand)
                break
    return out


def lloyd(X, k, seed=1, n_iter=6):
    """Lloyd's algorithm, kept only so the ITERATION can be drawn in panel B. The fit
    that is reported is sklearn's; the two are checked to agree in main()."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    hist = [C.copy()]
    lab = np.zeros(len(X), int)
    for _ in range(n_iter):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
        hist.append(C.copy())
    return lab, C, hist


def data_panel(ax, t, X):
    """Panel A. The stacking offset hides amplitude, so the norm is printed too -
    otherwise the panel would claim loudness matters while showing nothing of it."""
    amp = np.linalg.norm(X, axis=1)
    for i in range(N_ELEC):
        ax.plot(t, X[i] + i * 1.6, color=GREY, lw=1.1)
        ax.text(-0.05, i * 1.6, f"e{i}", fontsize=6.6, color=MUTED, ha="right",
                va="center")
        ax.text(1.03, i * 1.6, f"{amp[i]:.1f}", fontsize=6.5,
                color=RED if i == PICK else MUTED, va="center",
                fontweight="bold" if i == PICK else "normal")
    ax.text(1.03, N_ELEC * 1.6, "‖x‖", fontsize=7, color=INK, va="center")
    ax.text(1.17, PICK * 1.6, f"e{PICK} is quiet\nAND mixed", fontsize=7.2, color=RED,
            va="center")
    ax.set_title("A · the data — RAW, not unit-normed", fontsize=9.5, loc="left",
                 color=INK, pad=5)
    ax.set_xlabel("loudness is part of the distance here", fontsize=7.4, color=MUTED,
                  labelpad=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.11, 1.52)
    for s in ax.spines.values():
        s.set_visible(False)


def shapes_panel(ax, t, shapes, cent, tag):
    for j in range(K):
        s_n = shapes[j] / np.linalg.norm(shapes[j])
        c_n = cent[j] / max(np.linalg.norm(cent[j]), 1e-12)
        ax.plot(t, s_n + j * 0.30, color=GREY, lw=2.6)
        ax.plot(t, c_n + j * 0.30, color=COMP_COL[j], lw=1.5)
        ax.text(1.01, j * 0.30 + 0.06, f"{tag}{j}", color=COMP_COL[j], fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1.10)
    for s in ax.spines.values():
        s.set_visible(False)


def why_panel(ax, t, X, cent, lab, dom, tag, verb):
    """Panel E. A distance on its own proves nothing - the SHAPE correlation is the cue
    the raw fit is ignoring, so both numbers are printed and they disagree in print."""
    d = np.linalg.norm(X[PICK] - cent, axis=-1)
    r = np.array([float(np.corrcoef(X[PICK], cent[j])[0, 1]) for j in range(K)])
    ax.plot(t, X[PICK], color=INK, lw=2.4,
            label=f"e{PICK}  ‖x‖ {np.linalg.norm(X[PICK]):.1f}")
    ax.plot([], [], color="none", label="      dist   shape r")
    for j in range(K):
        note = f"  <- {verb}" if j == lab[PICK] else (
            "  <- truth" if j == dom[PICK] else "")
        ax.plot(t, cent[j], color=COMP_COL[j], lw=1.3, ls="--",
                label=f"{tag}{j}    {d[j]:4.1f}    {r[j]:+.2f}{note}")
    ax.set_title(f"E · why e{PICK} went where it did\n"
                 f"CLOSEST in dB is {tag}{lab[PICK]} — most SIMILAR in shape is "
                 f"{tag}{int(r.argmax())}",
                 fontsize=9.5, loc="left", color=INK, pad=5)
    ax.legend(frameon=False, loc="upper left", prop=MONO, labelspacing=0.35)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def truth_panel(ax, lab_raw, lab_unit, dom, method):
    """Panel F. Drawn as rectangles with a gap - as scatter squares the twelve rows
    ran together and read as three solid bars."""
    H = 0.62
    for i in range(N_ELEC):
        for col, lb in ((0, dom[i]), (1, lab_raw[i]), (2, lab_unit[i])):
            bad = col > 0 and lb != dom[i]
            ax.add_patch(plt.Rectangle((col - 0.30, i - H / 2), 0.60, H,
                                       facecolor=COMP_COL[lb],
                                       edgecolor=RED if bad else "none",
                                       lw=2.0 if bad else 0,
                                       hatch="///" if bad else None))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["planted\ntruth", f"{method}\nraw dB", f"{method}\nunit-norm"],
                       fontsize=8)
    # the marker is the red hatch itself - an inline "wrong group" label at x=1.36 ran
    # over the unit-norm column and looked like it belonged to the wrong one
    ax.set_xlabel("red hatch = disagrees with the planted truth", fontsize=7.4,
                  color=RED, labelpad=6)
    ax.set_yticks(range(N_ELEC))
    ax.set_yticklabels([f"e{i}" for i in range(N_ELEC)], fontsize=6.6)
    ax.set_xlim(-0.55, 3.0); ax.set_ylim(N_ELEC - 0.35, -0.65)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"F · against the planted truth\n"
                 f"{int((lab_raw != dom).sum())} misassigned in dB, "
                 f"{int((lab_unit != dom).sum())} in unit-norm",
                 fontsize=9.5, loc="left", color=INK, pad=5)


def header(fig, name, X, dom, lab, lab_u, extra):
    amp = np.linalg.norm(X, axis=1)
    r = float(np.corrcoef(dom, amp)[0, 1])
    bad = [f"e{i}" for i in range(N_ELEC) if lab[i] != dom[i]]
    isare = "is" if len(bad) == 1 else "are"
    fig.suptitle(f"How {name} clusters — the same twelve electrodes",
                 x=0.055, y=0.972, ha="left", fontsize=15, color=INK)
    body = [
        f"The same data as the convex NMF explainer: twelve invented electrodes, forty "
        f"time points, three planted response shapes, each electrode a known mixture "
        f"plus noise. The difference is the SPACE. Convex NMF unit-norms each electrode "
        f"first; {name}, as this project runs it, fits on RAW dB - so how loud an "
        f"electrode is counts toward the distance.",
        f"That is enough to change an answer. {', '.join(bad) if bad else 'No electrode'} "
        f"{isare} put in the wrong group in dB and {isare} correct once the SAME method "
        f"runs on unit-normed data (panel F). Panel E prints both cues for e{PICK} and "
        f"they disagree: it is closest in dB to the quiet group, but its shape "
        f"correlates best with the group it actually belongs to.",
        extra,
        f"Caveat, so the demo is not oversold: in this toy set loudness and shape are "
        f"correlated by construction (r = {r:.2f}), so the raw fit gets most assignments "
        f"right partly for the wrong reason. e{PICK} is the case where the two cues "
        f"disagree, which is exactly why it is the informative one.",
    ]
    fig.text(0.055, 0.930, "\n".join(textwrap.fill(x, width=146) for x in body),
             fontsize=8.4, color=MUTED, va="top", linespacing=1.5)


def new_fig():
    fig = plt.figure(figsize=(13.2, 8.8), dpi=200)
    gs = GridSpec(2, 3, hspace=0.55, wspace=0.30,
                  left=0.055, right=0.975, top=0.625, bottom=0.085)
    return fig, gs


# ==============================================================================
def kmeans_figure(t, shapes, mix, X, Xu, dom):
    from sklearn.cluster import KMeans
    lab = align(KMeans(K, n_init=10, random_state=0).fit_predict(X), dom)
    lab_u = align(KMeans(K, n_init=10, random_state=0).fit_predict(Xu), dom)
    cent = np.stack([X[lab == j].mean(0) for j in range(K)])
    lab_lloyd, _, hist = lloyd(X, K)
    settle = next((s for s in range(1, len(hist))
                   if np.allclose(hist[s], hist[s - 1])), len(hist) - 1)

    fig, gs = new_fig()
    data_panel(fig.add_subplot(gs[0, 0]), t, X)

    axb = fig.add_subplot(gs[0, 1])
    for s, C in enumerate(hist[:4]):
        for j in range(K):
            axb.plot(t, C[j] + s * 4.2, color=GREY if s == 0 else COMP_COL[j], lw=1.4)
        axb.text(-0.05, s * 4.2 + 1.4, "start\n(random)" if s == 0 else f"iter {s}",
                 fontsize=7, color=MUTED, ha="right", va="center")
    axb.set_title(f"B · the algorithm — assign to nearest centroid,\n"
                  f"recompute, repeat (stops moving after {settle})",
                  fontsize=9.5, loc="left", color=INK, pad=5)
    axb.set_xticks([]); axb.set_yticks([]); axb.set_xlim(-0.12, 1.02)
    for s_ in axb.spines.values():
        s_.set_visible(False)

    axc = fig.add_subplot(gs[0, 2])
    shapes_panel(axc, t, shapes, cent, "k")
    axc.set_title("C · the centroids — the MEAN of each group\n"
                  "grey = the shape that was planted",
                  fontsize=9.5, loc="left", color=INK, pad=5)

    axd = fig.add_subplot(gs[1, 0])
    D = np.linalg.norm(X[:, None, :] - cent[None, :, :], axis=-1)
    axd.imshow(D, cmap="Blues_r", aspect="auto")
    for i in range(N_ELEC):
        for j in range(K):
            axd.text(j, i, f"{D[i, j]:.1f}", ha="center", va="center", fontsize=6.4,
                     color="white" if D[i, j] < D.mean() else INK,
                     fontweight="bold" if j == lab[i] else "normal")
        axd.add_patch(plt.Rectangle((lab[i] - .5, i - .5), 1, 1, fill=False,
                                    edgecolor=RED if lab[i] != dom[i] else "#1b7837",
                                    lw=2.0))
    axd.set_xticks(range(K)); axd.set_xticklabels([f"k{j}" for j in range(K)],
                                                  fontsize=8.5)
    axd.set_yticks(range(N_ELEC))
    axd.set_yticklabels([f"e{i}" for i in range(N_ELEC)], fontsize=6.6)
    axd.set_title("D · distance to each centroid\n"
                  "boxed = the argmin, which IS the label",
                  fontsize=9.5, loc="left", color=INK, pad=5)
    axd.tick_params(length=0)

    why_panel(fig.add_subplot(gs[1, 1]), t, X, cent, lab, dom, "k", "picked")
    truth_panel(fig.add_subplot(gs[1, 2]), lab, lab_u, dom, "k-means")
    header(fig, "k-means", X, dom, lab, lab_u,
           "Every electrode gets exactly ONE label and nothing else - no weight, no "
           "runner-up, no measure of how close the call was. Panel D is the closest "
           "analogue of convex NMF's loading matrix, and it holds distances rather "
           "than shares.")
    p = OUT / "E2_kmeans_explained.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p, lab, lab_u, lab_lloyd


# ==============================================================================
def ward_figure(t, shapes, mix, X, Xu, dom):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

    Z = linkage(X, method="ward")
    lab = align(fcluster(Z, K, criterion="maxclust") - 1, dom)
    lab_u = align(AgglomerativeClustering(K, linkage="ward").fit_predict(Xu), dom)
    cent = np.stack([X[lab == j].mean(0) for j in range(K)])
    cut = Z[-(K - 1), 2]

    fig, gs = new_fig()
    data_panel(fig.add_subplot(gs[0, 0]), t, X)

    # scipy's default palette gave panel B its own orange/green/red, which contradicted
    # the colours panels C and F use for the same three groups. Colour each link by the
    # cluster its leaves belong to instead, so one group is one colour on every panel.
    below = {i: {i} for i in range(N_ELEC)}
    link_col = {}
    for i, row in enumerate(Z):
        below[N_ELEC + i] = below[int(row[0])] | below[int(row[1])]
    for node, leaves in below.items():
        seen = {lab[m] for m in leaves}
        link_col[node] = COMP_COL[seen.pop()] if len(seen) == 1 else MUTED

    axb = fig.add_subplot(gs[0, 1])
    dendrogram(Z, labels=[f"e{i}" for i in range(N_ELEC)], ax=axb,
               link_color_func=lambda n: link_col[n])
    axb.axhline(cut, color=RED, ls="--", lw=1.2)
    axb.text(0.01, cut, f" cut for K={K}", color=RED, fontsize=7.4, va="bottom",
             transform=axb.get_yaxis_transform())
    axb.set_title("B · the algorithm — merge the pair that raises\n"
                  "within-cluster variance least, repeatedly",
                  fontsize=9.5, loc="left", color=INK, pad=5)
    axb.tick_params(labelsize=6.8, colors=MUTED)
    axb.set_ylabel("merge cost", fontsize=8, color=MUTED)
    axb.spines[["top", "right"]].set_visible(False)

    axc = fig.add_subplot(gs[0, 2])
    shapes_panel(axc, t, shapes, cent, "w")
    axc.set_title("C · the cluster means after the cut\n"
                  "grey = the shape that was planted",
                  fontsize=9.5, loc="left", color=INK, pad=5)

    axd = fig.add_subplot(gs[1, 0])
    merges = Z[:, 2]
    axd.plot(np.arange(1, len(merges) + 1), merges, "o-", color=INK, ms=4, lw=1.2)
    cut_i = len(merges) - (K - 1)
    axd.axvline(cut_i + 0.5, color=RED, ls="--", lw=1.2)
    axd.text(cut_i + 0.75, merges.max() * 0.5, f"cut here\nleaves K={K}",
             color=RED, fontsize=7.4)
    axd.set_xlabel("merge number", fontsize=8.5)
    axd.set_ylabel("cost of that merge", fontsize=8.5)
    axd.set_title("D · what each merge costs\n"
                  "the jump at the end is why K=3 is a natural cut",
                  fontsize=9.5, loc="left", color=INK, pad=5)
    axd.tick_params(labelsize=7.5, colors=MUTED)
    axd.spines[["top", "right"]].set_visible(False)

    why_panel(fig.add_subplot(gs[1, 1]), t, X, cent, lab, dom, "w", "merged")
    truth_panel(fig.add_subplot(gs[1, 2]), lab, lab_u, dom, "Ward")
    header(fig, "Ward", X, dom, lab, lab_u,
           "Ward is DETERMINISTIC - no random start, so it returns the same answer "
           "every time. That is a property of the algorithm, not evidence the answer is "
           "right, which is why notebook 238 reports its initialisation score as "
           "not-applicable rather than as a pass. It is also GREEDY: e5 is merged early "
           "and no later step can undo it.")
    p = OUT / "E3_ward_explained.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p, lab, lab_u


def main() -> int:
    t, shapes, mix, X = build()
    Xu = LD.unit_norm(X)
    dom = mix.argmax(1)
    OUT.mkdir(parents=True, exist_ok=True)

    pk, lab_k, labu_k, lab_lloyd = kmeans_figure(t, shapes, mix, X, Xu, dom)
    agree = bool((align(lab_lloyd, dom) == lab_k).all())
    print(f"panel B's hand-written Lloyd agrees with sklearn: {agree}")
    pw, lab_w, labu_w = ward_figure(t, shapes, mix, X, Xu, dom)

    print(f"\n   planted        {dom}")
    for nm, a, b, p in (("k-means", lab_k, labu_k, pk), ("Ward", lab_w, labu_w, pw)):
        print(f"{p.name}")
        print(f"   {nm:<9} dB   {a}   misassigned {int((a != dom).sum())}")
        print(f"   {nm:<9} unit {b}   misassigned {int((b != dom).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
