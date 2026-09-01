#!/usr/bin/env python3
"""
00_paper2_figures2_2.py - FIG 2, agreement: what the solutions agree on, and what is
left out.

    python 00_paper2_figures2_2.py                    K = 8, both halves
    python 00_paper2_figures2_2.py --k 12
    python 00_paper2_figures2_2.py --algo-feature-set concat_rawds

TWO HALVES, THE SAME THREE QUESTIONS EACH.

    top     FEATURE SETS   convex NMF on concat_hg / concat_rawds / concat_bands5 /
                           concat_bands5z - does the representation change the answer?
    bottom  ALGORITHMS     convex NMF / k-means / Ward / archetypes on one feature set -
                           does the method change the answer?

    A / D   how much any two solutions agree at all           (pairwise ARI)
    B / E   each CLUSTER against the reference, and what        (Jaccard, plus what
            the other solutions think of the same cluster       the others think)
    C / F   which ELECTRODES are placed consistently          (per-electrode agreement,
                                                               on the brain)

THIS FIGURE IS ABOUT WHAT IS LEFT OUT. A 1:1 assignment always returns a partner for
every cluster whether or not one exists, so B and E report the Jaccard of every pair and
a near-empty row is a cluster that only one solution found. C and F ask the same of each
electrode rather than each cluster, because a cluster can match well on average and
still be built from electrodes nobody else groups together.

The matching machinery is IMPORTED from 00_Paper2_Figures.py rather than reimplemented,
so FIG 1's block order and FIG 2's correspondence can never disagree about which cluster
is which.

NO CAPTION IS DRAWN ON THE FIGURE - it gets a sibling <name>_caption.txt, plus
_cluster_agreement.csv and _electrode_agreement.csv.
"""
from __future__ import annotations

import argparse
import importlib.util
import textwrap
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                        # noqa: E402

# The shared machinery, imported from FIG 1's script. A module whose name starts with a
# digit cannot be imported by name, hence the loader. Nothing here is reimplemented:
# match_clusters, the renderer, the palette, the verified writers and the reference all
# come from there, so the two figures cannot drift apart about which cluster is which.
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)

CLUST, OUT = P2.CLUST, P2.OUT
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
RED, GREEN = P2.RED, P2.GREEN
BG = P2.BG

FEATURE_SETS = P2.FSETS
METHODS = ["cnmf", "kmeans", "hierarchical", "archetypes"]
METHOD_LABEL = {"cnmf": "convex NMF", "kmeans": "k-means", "hierarchical": "Ward",
                "archetypes": "archetypes"}
FS_SHORT = P2.FS_SHORT
# convex NMF is the reference for the algorithm half and concat_bands5 for the feature
# half, matching FIG 1's block order so the two figures number clusters the same way.
REF_METHOD = "cnmf"
REF_FSET = P2.MATCH_REF_FSET
# concat_hg by default for the algorithm half, because it is the only feature set with
# an archetype run - bands5 and bands5z have none, so choosing the FIG 1 reference here
# would silently drop a whole algorithm.
DEFAULT_ALGO_FSET = "concat_hg"


# ---- loading any solution ----------------------------------------------------
def solution(method: str, fset: str, k: int):
    """One clustering solution at K: (loadings or None, hard labels, run dir, keys).

    Graded methods return their membership matrix - convex NMF's G renormalised to sum
    to 1, archetypal analysis's A which already does - and the hard methods return None,
    which is what makes match_clusters fall back from loading correlation to shared
    electrodes. `keys` identifies the electrodes so callers can prove two solutions are
    describing the same ones in the same order rather than assuming it.
    """
    try:
        run = LR.newest_run(method, fset)
    except Exception:
        run = None
    if run is None:
        return None
    G, lab = None, None
    for name, norm in ((f"G_k{k:02d}.npy", True), (f"A_k{k:02d}.npy", False)):
        f = run / "loadings_by_k" / name
        if f.exists():
            W = np.load(f).astype(float)
            G = W / np.maximum(W.sum(1, keepdims=True), 1e-12) if norm else W
            lab = G.argmax(1)
            break
    if lab is None:
        f = run / "cluster_labels_by_k.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
        if f"k_{k}" not in d.columns:
            return None
        lab = d[f"k_{k}"].to_numpy().astype(int)
    keys = None
    lf = run / "labels.csv"
    if lf.exists():
        m = pd.read_csv(lf)
        if {"patient_id", "electrode"} <= set(m.columns):
            keys = [f"{p}|{P2.norm(e)}"
                    for p, e in zip(m["patient_id"], m["electrode"])]
    return dict(G=G, lab=np.asarray(lab, int), run=run, keys=keys,
                method=method, fset=fset)


def gather(pairs, k):
    """Load a set of solutions and REFUSE to compare ones that are not comparable.

    Everything downstream - the assignment, the Jaccards, the per-electrode agreement -
    assumes the solutions label the same electrodes in the same order. That is true of
    every run in this project and it is exactly the assumption that would produce
    confident nonsense if it ever stopped being true, so it is checked rather than
    trusted. Solutions that are absent, that are at a different K, or that describe a
    different electrode set are dropped with a message.
    """
    got, dropped = [], []
    for method, fset in pairs:
        s = solution(method, fset, k)
        if s is None:
            dropped.append((method, fset, "no run, or no solution at this K"))
            continue
        got.append(s)
    if not got:
        raise SystemExit("no solutions to compare")
    n, ref_keys = len(got[0]["lab"]), got[0]["keys"]
    keep = []
    for s in got:
        nk = int(s["lab"].max()) + 1
        if len(s["lab"]) != n:
            dropped.append((s["method"], s["fset"],
                            f"{len(s['lab'])} electrodes, not {n}"))
        elif nk != k:
            dropped.append((s["method"], s["fset"], f"{nk} clusters, not {k}"))
        elif ref_keys is not None and s["keys"] is not None and s["keys"] != ref_keys:
            dropped.append((s["method"], s["fset"], "different electrodes or order"))
        else:
            keep.append(s)
    for m_, f_, why in dropped:
        print(f"    dropped {m_}/{f_}: {why}")
    if not keep:
        raise SystemExit("every solution was dropped - nothing comparable at this K")
    return keep, dropped


# ---- the measurements ---------------------------------------------------------
def _modal_count(T, K):
    """Per column of T, how many rows share the commonest value. Vectorised.

    T is (n_solutions, n_electrodes) of block positions. The obvious loop over 1693
    electrodes is fine once and far too slow inside a 200-permutation null, so both use
    this.
    """
    oneh = np.zeros((K, T.shape[1]), np.int16)
    for c in range(T.shape[0]):
        oneh[T[c], np.arange(T.shape[1])] += 1
    return oneh.max(0).astype(int)


def pairwise_ari(sols):
    """ARI between every pair. Chance-corrected, so 0 is 'no better than random'."""
    n = len(sols)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = adjusted_rand_score(sols[i]["lab"], sols[j]["lab"])
    return M


def ari_by_k(sols, ks):
    """Mean pairwise ARI at every K, so one K is never read in isolation.

    Read from each run's cluster_labels_by_k.csv, which holds every K in one file - the
    alternative is opening one loadings array per K per solution. Returns None if any
    solution cannot supply the whole range, because a curve missing a solution is not
    the same curve.
    """
    tabs = []
    for s in sols:
        f = s["run"] / "cluster_labels_by_k.csv"
        if not f.exists():
            return None
        tabs.append(pd.read_csv(f))
    out_k, out_v = [], []
    for k in ks:
        col = f"k_{k}"
        if any(col not in t.columns for t in tabs):
            continue
        labs = [t[col].to_numpy() for t in tabs]
        vals = [adjusted_rand_score(labs[i], labs[j])
                for i in range(len(labs)) for j in range(i + 1, len(labs))]
        out_k.append(k); out_v.append(float(np.mean(vals)))
    return (np.array(out_k), np.array(out_v)) if out_k else None


def cluster_agreement(sols, ref_i, order):
    """Per-cluster Jaccard against the reference, plus what the OTHERS think.

    `others` is the strongest agreement between two NON-reference solutions about the
    same row. It exists because the rest of this panel is asymmetric: a cluster that
    every other solution finds, but the reference does not, shows up as a row of low
    numbers and reads as 'this cluster did not survive'. If `others` is high while the
    row is low, the reference is the odd one out and the cluster survived fine.
    """
    ref = sols[ref_i]
    K = int(ref["lab"].max()) + 1
    J = np.zeros((K, len(sols)))
    maps = []
    for c, s in enumerate(sols):
        if c == ref_i:
            J[:, c] = 1.0
            maps.append(np.arange(K))
            continue
        m, jac, _ = P2.match_clusters((s["G"], s["lab"]), (ref["G"], ref["lab"]))
        inv = np.full(K, -1, int)
        inv[m] = np.arange(K)                # reference cluster -> this solution's
        maps.append(inv)
        for p in range(K):
            J[p, c] = jac[inv[p]]
    others = np.zeros(K)
    rest = [c for c in range(len(sols)) if c != ref_i]
    for p in range(K):
        r = int(order[p])
        best = 0.0
        for a in range(len(rest)):
            for b in range(a + 1, len(rest)):
                ca, cb = rest[a], rest[b]
                A = sols[ca]["lab"] == maps[ca][r]
                B = sols[cb]["lab"] == maps[cb][r]
                u = float((A | B).sum())
                if u:
                    best = max(best, float((A & B).sum()) / u)
        others[p] = best
    return J[order, :], [mm[order] for mm in maps], others


def electrode_agreement(sols, ref_i, maps_in_ref_space, order):
    """How many solutions put each electrode in the SAME cluster, after matching.

    The modal assignment, so nothing privileges the reference: an electrode where the
    reference is the odd one out still reads as agreement among the rest.
    """
    K = int(sols[ref_i]["lab"].max()) + 1
    inv_order = np.empty(K, int)
    inv_order[order] = np.arange(K)          # reference cluster -> block position
    T = np.empty((len(sols), len(sols[0]["lab"])), int)
    for c, s in enumerate(sols):
        fwd = np.full(K, -1, int)            # this solution's cluster -> block position
        for p, own in enumerate(maps_in_ref_space[c]):
            fwd[own] = p
        T[c] = fwd[s["lab"]]
    return _modal_count(T, K), T


def electrode_agreement_null(sols, ref_i, order, n_perm=200, seed=0):
    """The same count when the other solutions are random. Returns fractions per count.

    Each non-reference solution's labels are PERMUTED - which keeps its cluster sizes
    exactly and destroys its correspondence - and then RE-MATCHED to the reference, the
    way the real ones are. Re-matching is the point: the observed pipeline optimises
    the alignment before counting, so a null that skipped that step would be far too
    easy to beat and the panel would overstate its own result.

    The overlap basis is used throughout because a permuted partition has no loadings
    left to correlate.
    """
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(seed)
    ref = sols[ref_i]
    K = int(ref["lab"].max()) + 1
    n = len(ref["lab"])
    inv_order = np.empty(K, int)
    inv_order[order] = np.arange(K)
    ref_pos = inv_order[ref["lab"]]
    rest = [c for c in range(len(sols)) if c != ref_i]
    acc = np.zeros(K + 1)
    for _ in range(n_perm):
        T = np.empty((len(sols), n), int)
        T[0] = ref_pos
        for i, c in enumerate(rest, start=1):
            pl = rng.permutation(sols[c]["lab"])
            Cm = np.bincount(pl * K + ref["lab"], minlength=K * K).reshape(K, K)
            r_, c_ = linear_sum_assignment(-Cm)
            m = np.empty(K, int)
            m[r_] = c_                        # permuted cluster -> reference cluster
            T[i] = inv_order[m[pl]]
        acc += np.bincount(_modal_count(T, K), minlength=K + 1)[:K + 1]
    f = acc[1:len(sols) + 1]
    return f / max(f.sum(), 1)


def one_patient_clusters(lab, patient, k, danger=0.50):
    """How many clusters are more than `danger` one patient - FIG 1 panel D's number."""
    codes, _ = pd.factorize(patient)
    P = int(codes.max()) + 1
    out = 0
    for j in range(k):
        sel = lab == j
        if sel.sum() and np.bincount(codes[sel], minlength=P).max() / sel.sum() > danger:
            out += 1
    return out


# ---- panels ------------------------------------------------------------------
# graded methods carry a membership matrix, hard ones only a partition. The split runs
# through panel D and naming it is most of what that panel has to say.
GROUP_COL = {"graded": "#5b2c83", "hard": "#e08214"}


def wrap(text, chars):
    return textwrap.fill(text, chars)


def summary_under(ax, text, chars, colour=None):
    """A wrapped summary directly under a panel, in that panel's own width."""
    ax.text(0, -0.055, wrap(text, chars), transform=ax.transAxes, va="top", ha="left",
            fontsize=7.4, color=colour or MUTED, linespacing=1.5)


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def ari_summary(M, names, groups):
    """The structure in a pairwise matrix, in words, computed from the matrix."""
    n = len(M)
    iu = np.triu_indices(n, 1)
    vals, pairs = M[iu], list(zip(*iu))
    hi, lo = int(np.argmax(vals)), int(np.argmin(vals))
    mean_off = np.array([(M[i].sum() - 1) / max(n - 1, 1) for i in range(n)])
    j = int(np.argmin(mean_off))
    parts = [f"strongest {names[pairs[hi][0]]}-{names[pairs[hi][1]]} {vals[hi]:.2f}",
             f"weakest {names[pairs[lo][0]]}-{names[pairs[lo][1]]} {vals[lo]:.2f}",
             f"{names[j]} agrees least with the rest ({mean_off[j]:.2f}, "
             f"a mean of {n - 1})"]
    if groups is not None and len(set(groups)) == 2:
        g = np.array(groups)
        within = [M[a, b] for a in range(n) for b in range(a + 1, n) if g[a] == g[b]]
        across = [M[a, b] for a in range(n) for b in range(a + 1, n) if g[a] != g[b]]
        if within and across:
            parts.append(f"WITHIN a kind {np.mean(within):.2f} vs ACROSS the "
                         f"graded/hard divide {np.mean(across):.2f} "
                         f"(labels: purple = graded, orange = hard)")
    return "  ·  ".join(parts)


def panel_ari(ax, M, names, groups, chars, k, curve):
    """Lower triangle only - the other ten cells of a 4x4 repeat these six.

    The empty upper triangle carries the K curve, because a single K read on its own
    cannot say whether the agreement here is typical or an artefact of where it was
    cut - the lesson FIG 1 panel D exists to make.
    """
    n = len(M)
    show = np.full_like(M, np.nan)
    for i in range(n):
        for j in range(i):
            show[i, j] = M[i, j]
    ax.imshow(show, cmap="Blues", vmin=0, vmax=max(0.6, np.nanmax(show) * 1.05))
    for i in range(n):
        for j in range(i):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9.0,
                    color="white" if M[i, j] > 0.42 else INK)
    ax.set_xticks(range(n - 1)); ax.set_yticks(range(1, n))
    ax.xaxis.tick_top()
    ax.set_xticklabels(names[:-1], fontsize=7.8, rotation=26, ha="left")
    ax.set_yticklabels(names[1:], fontsize=7.8)
    if groups is not None and len(set(groups)) == 2:
        for t, gname in zip(ax.get_xticklabels(), groups[:-1]):
            t.set_color(GROUP_COL[gname])
        for t, gname in zip(ax.get_yticklabels(), groups[1:]):
            t.set_color(GROUP_COL[gname])
    ax.tick_params(length=0, colors=MUTED)
    ax.set_anchor("N")
    for s_ in ax.spines.values():
        s_.set_visible(False)

    extra = ""
    if curve is not None:
        ks, vs = curve
        ins = ax.inset_axes([0.46, 0.52, 0.52, 0.38])
        ins.plot(ks, vs, "-", lw=1.4, color=INK)
        ins.axvline(k, color=RED, ls="--", lw=1.0)
        ins.set_title("mean pairwise ARI vs K", fontsize=6.6, color=MUTED, pad=2)
        ins.tick_params(labelsize=6.0, colors=MUTED, length=2)
        ins.set_ylim(0, max(0.6, vs.max() * 1.12))
        ins.spines[["top", "right"]].set_visible(False)
        best = int(ks[int(np.argmax(vs))])
        here = float(vs[list(ks).index(k)]) if k in list(ks) else float("nan")
        extra = (f"  ·  ACROSS K: agreement peaks at K={best} ({vs.max():.2f}); "
                 f"at K={k} it is {here:.2f}")
    summary_under(ax, "adjusted Rand, chance-corrected.  " +
                  ari_summary(M, names, groups) + extra, chars)


def panel_clusters(axM, axW, J, others, names, order, sizes, ref_i, chars, vmax,
                   thresh=0.30):
    """Every reference cluster against its match, with the reference column dropped.

    NOT 'which clusters survive'. Every number here is agreement with ONE reference, so
    a cluster the reference alone missed and a cluster nobody else found look the same
    from these columns. The `others` column separates them: it is the strongest
    agreement between two NON-reference solutions about that row, so a low row beside a
    high `others` means the reference is the odd one out.
    """
    keep = [c for c in range(J.shape[1]) if c != ref_i]
    Jk, nk = J[:, keep], [names[c] for c in keep]
    K = Jk.shape[0]
    axM.imshow(Jk, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for p in range(K):
        for c in range(Jk.shape[1]):
            v = Jk[p, c]
            axM.text(c, p, f"{v:.2f}", ha="center", va="center", fontsize=8.2,
                     color="white" if v > 0.62 * vmax else (RED if v < thresh else INK))
    axM.set_xticks(range(len(nk))); axM.set_yticks(range(K))
    axM.xaxis.tick_top()
    axM.set_xticklabels(nk, fontsize=8.0, rotation=26, ha="left")
    worst = Jk.min(axis=1)
    axM.set_yticklabels(
        [f"p{p}  c{int(order[p])}  n={int(sizes[int(order[p])])}" for p in range(K)],
        fontsize=7.4)
    for t, w in zip(axM.get_yticklabels(), worst):
        t.set_color(RED if w < thresh else INK)
    axM.tick_params(length=0, colors=MUTED)
    for s_ in axM.spines.values():
        s_.set_visible(False)

    # worst-of-N and what the others think, on the SAME scale as the matrix
    side = np.column_stack([worst, others])
    axW.imshow(side, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for p in range(K):
        for c, v in enumerate(side[p]):
            axW.text(c, p, f"{v:.2f}", ha="center", va="center", fontsize=8.2,
                     color="white" if v > 0.62 * vmax else
                     (RED if (c == 0 and v < thresh) else INK))
        if worst[p] < thresh:
            axW.add_patch(Rectangle((-0.5, p - 0.5), 1, 1, fill=False, ec=RED,
                                    lw=1.4, zorder=4))
        # the reference is the odd one out: the others agree with each other here
        if worst[p] < thresh and others[p] > 2 * worst[p] and others[p] > thresh:
            axW.add_patch(Rectangle((0.5, p - 0.5), 1, 1, fill=False, ec=GREEN,
                                    lw=1.4, zorder=4))
    axW.set_xticks([0, 1]); axW.set_yticks([])
    axW.xaxis.tick_top()
    axW.set_xticklabels([f"worst of {Jk.shape[1]}", "others"], fontsize=7.6,
                        rotation=26, ha="left")
    axW.tick_params(length=0, colors=MUTED)
    for s_ in axW.spines.values():
        s_.set_visible(False)

    lost = int((worst < thresh).sum())
    flip = int(((worst < thresh) & (others > 2 * worst) & (others > thresh)).sum())
    gone = [f"p{p} (c{int(order[p])}, n={int(sizes[int(order[p])])})"
            for p in range(K) if worst[p] < thresh]
    txt = (f"Jaccard against the 1:1 match, reference column dropped, scale 0-{vmax:.2f}."
           f"  A LOW VALUE IS THE FINDING: assignment always returns a partner, so it "
           f"means paired-because-it-had-to-be.  {lost} of {K} fall below "
           f"{thresh:.2f} somewhere")
    if gone:
        txt += ": " + ", ".join(gone[:4])
        if len(gone) > 4:
            txt += f", and {len(gone) - 4} more"
    txt += (f".  OTHERS = best agreement between two non-reference solutions on that "
            f"row; {flip} row(s), outlined green, are low only because the REFERENCE "
            f"disagrees.")
    summary_under(axM, txt, chars, colour=RED if lost else MUTED)


def agreement_colours(n):
    """Red at 1 (nobody agrees) through to green at n (everybody does)."""
    return plt.get_cmap("RdYlGn")(np.linspace(0.10, 0.92, max(n, 2)))[:n]


def stacked_agreement(ax, cnt, N, cols, null_frac):
    """Observed against chance, as two bars that double as the colour key."""
    h = np.bincount(cnt, minlength=N + 1)[1:N + 1].astype(float)
    obs = h / max(h.sum(), 1)
    for row, (frac, y, hgt, tag) in enumerate(
            ((obs, 0.16, 0.62, "observed"), (null_frac, -0.62, 0.42, "chance"))):
        x = 0.0
        for i in range(N):
            ax.barh(y, frac[i], left=x, height=hgt, color=cols[i], lw=0.8, ec="white",
                    alpha=1.0 if row == 0 else 0.45)
            if frac[i] > 0.05:
                ax.text(x + frac[i] / 2, y, f"{i+1} of {N}\n{100*frac[i]:.0f}%"
                        if row == 0 else f"{100*frac[i]:.0f}%",
                        ha="center", va="center", fontsize=7.0 if row == 0 else 6.2,
                        color="white" if (row == 0 and lum(cols[i]) < 0.62) else INK)
            elif frac[i] > 0.002 and row == 0:
                # a sliver with no room for a label is exactly the case a reader wants
                # to read, so it gets one above the bar with a leader
                ax.annotate(f"{i+1} of {N}: {100*frac[i]:.0f}%",
                            xy=(x + frac[i] / 2, y + hgt / 2),
                            xytext=(x + frac[i] / 2, y + hgt / 2 + 0.55),
                            ha="center", va="bottom", fontsize=6.2, color=INK,
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
            x += frac[i]
        ax.text(-0.006, y, tag, ha="right", va="center", fontsize=6.6, color=MUTED,
                transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def panel_electrodes(axL, axR, axBar, cnt, N, d, chars, null_frac, n_onepat, k,
                     ref_name):
    cols = agreement_colours(N)
    for ax, side in ((axL, "L"), (axR, "R")):
        img, nsel = render_counts(side, d, cnt, cols)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.text(0.02, 0.02, f"{side}  {nsel}", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=7.0, color=MUTED)
    stacked_agreement(axBar, cnt, N, cols, null_frac)
    obs_all = float((cnt == N).mean())
    summary_under(axBar,
                  f"How many of the {N} solutions put each electrode in the same "
                  f"cluster - the MODAL assignment, so nothing privileges the "
                  f"reference.  {100*obs_all:.0f}% are placed the same way by all {N}, "
                  f"against {100*null_frac[-1]:.0f}% when the other solutions are "
                  f"PERMUTED and re-matched; {100*(cnt <= 1).mean():.0f}% by no two.  "
                  f"Red = 1, green = {N}.  AGREEMENT IS NOT CORRECTNESS: at K={k}, "
                  f"{n_onepat} of {k} clusters in {ref_name} are over half one patient "
                  f"(FIG 1 panel D).", chars)


def render_counts(side, d, cnt, cols):
    """A hemisphere coloured by per-electrode agreement, on FIG 1's own brain scene."""
    pl, ok = P2._scene(side, d, 1.30)
    actor = None
    if ok.sum():
        c = cnt[ok]
        rgba = np.empty((len(c), 4), np.uint8)
        rgba[:, :3] = np.clip(255 * np.array(cols)[c - 1, :3], 0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        cloud = P2.pv.PolyData(d["xyz"][ok])
        cloud["rgba"] = rgba
        # ONE RADIUS PER POINT, not one scalar: pyvista point data has to be as
        # long as the point set. FIG 1 gets away with the same line because there
        # it multiplies by the per-electrode loading and is already an array.
        cloud["r"] = np.full(len(c), P2.BG_RADIUS * 1.25, float)
        g = cloud.glyph(orient=False, scale="r",
                        geom=P2.pv.Sphere(radius=1.0, theta_resolution=12,
                                          phi_resolution=12))
        actor = pl.add_mesh(g, scalars="rgba", rgba=True)
    img = pl.screenshot(return_img=True, transparent_background=True)
    if actor is not None:
        pl.remove_actor(actor)
    return P2._crop_alpha(np.asarray(img)), int(ok.sum())


# ---- FIGURE 2 ----------------------------------------------------------------
def figure_2(k: int, algo_fset: str):
    t0 = time.time()
    for _pl, _ in P2._PLOTTER.values():
        _pl.close()
    P2._PLOTTER.clear()

    print(f"\n  feature sets, {REF_METHOD}, K={k}")
    fs_sols, fs_dropped = gather([(REF_METHOD, f) for f in FEATURE_SETS], k)
    print(f"  algorithms, {algo_fset}, K={k}")
    al_sols, al_dropped = gather([(m, algo_fset) for m in METHODS], k)

    # geometry, and the reference's block order, both straight from FIG 1
    d = P2.load_run(REF_FSET, k)
    d["fset"] = REF_FSET
    C = P2.cube(d["X"], d)
    means = np.stack([C[d["lab"] == j].mean(0) if (d["lab"] == j).any()
                      else np.zeros(C.shape[1:]) for j in range(k)])
    order = np.argsort(-np.array([P2.condition_similarity(means[j])
                                  for j in range(k)]), kind="stable")

    fs_names = [FS_SHORT.get(s["fset"], s["fset"]) for s in fs_sols]
    al_names = [METHOD_LABEL.get(s["method"], s["method"]) for s in al_sols]
    fs_ref = next(i for i, s in enumerate(fs_sols) if s["fset"] == REF_FSET) \
        if any(s["fset"] == REF_FSET for s in fs_sols) else 0
    al_ref = next(i for i, s in enumerate(al_sols) if s["method"] == REF_METHOD) \
        if any(s["method"] == REF_METHOD for s in al_sols) else 0

    # EACH HALF IS SEQUENCED BY THE SHARED REFERENCE, AND SIZED BY ITS OWN.
    # The feature half's reference IS the shared one, so its order is the shared order.
    # The algorithm half's reference is a different solution, so it is matched to the
    # shared one first: without that, p0 in E was concat_bands5's ranking applied to
    # concat_hg's clusters - an arbitrary permutation wearing the same p numbering as B.
    fs_order = order
    fs_sizes = np.bincount(fs_sols[fs_ref]["lab"], minlength=k)
    ar = al_sols[al_ref]
    if ar["fset"] == REF_FSET and ar["method"] == REF_METHOD:
        al_order = order
    else:
        m_, _, _ = P2.match_clusters((ar["G"], ar["lab"]), (d["Gn"], d["lab"]))
        inv_ = np.full(k, -1, int)
        inv_[m_] = np.arange(k)              # shared-reference cluster -> this half's
        al_order = inv_[order]
    al_sizes = np.bincount(ar["lab"], minlength=k)

    fs_ari, al_ari = pairwise_ari(fs_sols), pairwise_ari(al_sols)
    fs_J, fs_maps, fs_oth = cluster_agreement(fs_sols, fs_ref, fs_order)
    al_J, al_maps, al_oth = cluster_agreement(al_sols, al_ref, al_order)
    fs_cnt, _ = electrode_agreement(fs_sols, fs_ref, fs_maps, fs_order)
    al_cnt, _ = electrode_agreement(al_sols, al_ref, al_maps, al_order)
    print("    permutation null for the electrode agreement ...")
    fs_null = electrode_agreement_null(fs_sols, fs_ref, fs_order)
    al_null = electrode_agreement_null(al_sols, al_ref, al_order)
    ks = list(range(5, 31))
    fs_curve, al_curve = ari_by_k(fs_sols, ks), ari_by_k(al_sols, ks)
    # ONE Jaccard scale for both halves, set from the data: 0-1 left half the bar
    # unused once the reference column was dropped, so every difference looked smaller
    vmax = float(max(0.35, np.nanmax([np.delete(fs_J, fs_ref, axis=1).max(),
                                      np.delete(al_J, al_ref, axis=1).max()])))
    n_op_fs = one_patient_clusters(fs_sols[fs_ref]["lab"], d["patient"], k)
    n_op_al = one_patient_clusters(ar["lab"], d["patient"], k)

    fig = plt.figure(figsize=(17.6, 9.6), dpi=190)
    gs = GridSpec(2, 24, figure=fig, hspace=0.75, wspace=1.1,
                  left=0.055, right=0.985, top=0.820, bottom=0.055)
    fig.suptitle(f"FIG 2   ·   agreement   ·   K = {k}   ·   "
                 f"{len(d['X'])} electrodes, {d['n_patients']} patients",
                 x=0.055, y=0.975, ha="left", fontsize=15.5, color=INK)
    fig.text(0.055, 0.945,
             "Does the answer depend on how the data are represented, or on which "
             "algorithm is used?  Each row asks that three ways: overall, per cluster, "
             "per electrode.",
             fontsize=9.8, color=MUTED, va="top")

    halves = (
        dict(ari=fs_ari, J=fs_J, cnt=fs_cnt, names=fs_names, sols=fs_sols, ref=fs_ref,
             oth=fs_oth, order=fs_order, sizes=fs_sizes, null=fs_null,
             curve=fs_curve, onepat=n_op_fs,
             refname=FS_SHORT.get(REF_FSET, REF_FSET), tags="ABC",
             head=f"FEATURE SETS   ·   convex NMF on four representations   ·   "
                  f"reference {FS_SHORT.get(REF_FSET, REF_FSET)}"),
        dict(ari=al_ari, J=al_J, cnt=al_cnt, names=al_names, sols=al_sols, ref=al_ref,
             oth=al_oth, order=al_order, sizes=al_sizes, null=al_null,
             curve=al_curve, onepat=n_op_al,
             refname=f"{METHOD_LABEL[REF_METHOD]} on {algo_fset}", tags="DEF",
             head=f"ALGORITHMS   ·   four methods on {algo_fset}   ·   "
                  f"reference {METHOD_LABEL[REF_METHOD]}"),
    )
    labelled = []
    for row, h in enumerate(halves):
        groups = ["graded" if x["G"] is not None else "hard" for x in h["sols"]]
        t1, t2, t3 = h["tags"]

        axA = fig.add_subplot(gs[row, 0:5])
        panel_ari(axA, h["ari"], h["names"], groups, 62, k, h["curve"])

        cB = GridSpecFromSubplotSpec(1, 2, gs[row, 6:14], width_ratios=[5.4, 2.0],
                                     wspace=0.08)
        axM = fig.add_subplot(cB[0])
        panel_clusters(axM, fig.add_subplot(cB[1]), h["J"], h["oth"], h["names"],
                       h["order"], h["sizes"], h["ref"], 104, vmax)

        cC = GridSpecFromSubplotSpec(2, 2, gs[row, 15:24], height_ratios=[1.0, 0.30],
                                     hspace=0.16, wspace=0.03)
        axL = fig.add_subplot(cC[0, 0])
        panel_electrodes(axL, fig.add_subplot(cC[0, 1]), fig.add_subplot(cC[1, :]),
                         h["cnt"], len(h["sols"]), d, 118, h["null"], h["onepat"], k,
                         h["refname"])

        labelled.append((
            [(axA, f"{t1}  ·  how much any two agree"),
             (axM, f"{t2}  ·  each cluster against the reference"),
             (axL, f"{t3}  ·  which electrodes are placed consistently")], h["head"]))

    # MEASURE, THEN PLACE. One draw settles every tick label, so the row's three
    # labels can share one line above the tallest panel instead of each floating at
    # whatever height its own decorations happen to reach.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for panels, head in labelled:
        top = max(inv.transform((0, ax.get_tightbbox(r).y1))[1] for ax, _ in panels)
        for ax, label in panels:
            # the PLOT area's left edge, not the tight bbox's - the tight bbox
            # includes the y tick labels, which would set each label at a different
            # x depending on how long its row names happen to be
            fig.text(ax.get_position().x0, top + 0.010, label, fontsize=10.4,
                     color=INK, va="bottom")
        fig.text(0.055, top + 0.040, head, fontsize=11.0, color=INK, va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"FIG2_agreement_K{k}.png"
    P2.save_png(fig, p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = []
    for half, J, names, sols, od, oth in (
            ("feature_set", fs_J, fs_names, fs_sols, fs_order, fs_oth),
            ("algorithm", al_J, al_names, al_sols, al_order, al_oth)):
        for pos in range(J.shape[0]):
            for c, nm in enumerate(names):
                rows.append(dict(half=half, block_position=pos,
                                 reference_cluster=int(od[pos]),
                                 solution=nm, jaccard=round(float(J[pos, c]), 4),
                                 others_best=round(float(oth[pos]), 4)))
    P2.save_text(pd.DataFrame(rows).to_csv(index=False),
                 p.with_name(p.stem + "_cluster_agreement.csv"))
    P2.save_text(pd.DataFrame(dict(
        patient=d["patient"], electrode=d["meta"]["electrode"],
        feature_set_agreement=fs_cnt, n_feature_sets=len(fs_sols),
        algorithm_agreement=al_cnt, n_algorithms=len(al_sols),
    )).to_csv(index=False), p.with_name(p.stem + "_electrode_agreement.csv"))

    caption_2(p.with_name(p.stem + "_caption.txt"), k, d, algo_fset,
              (fs_ari, fs_J, fs_cnt, fs_names, fs_sols, fs_dropped, fs_order,
               fs_oth, fs_null, fs_curve, n_op_fs),
              (al_ari, al_J, al_cnt, al_names, al_sols, al_dropped, al_order,
               al_oth, al_null, al_curve, n_op_al), vmax)
    print(f"  {time.time()-t0:.0f}s  -> {p.name}")
    return p


# ---- the caption -------------------------------------------------------------
def caption_2(path, k, d, algo_fset, fs, al, vmax):
    (fs_ari, fs_J, fs_cnt, fs_names, fs_sols, fs_dropped, fs_order, fs_oth,
     fs_null, fs_curve, n_op_fs) = fs
    (al_ari, al_J, al_cnt, al_names, al_sols, al_dropped, al_order, al_oth,
     al_null, al_curve, n_op_al) = al
    order = fs_order
    L = []
    A = L.append
    A(f"FIG 2   ·   agreement   ·   K = {k}")
    A("=" * 100)
    A("")
    A("WHAT THE FIGURE IS")
    A("-" * 100)
    A("Two halves asking the same three questions. The top half varies the FEATURE SET")
    A(f"with the method fixed at {REF_METHOD}; the bottom varies the ALGORITHM with the")
    A(f"feature set fixed at {algo_fset}. If a result only exists in one representation")
    A("or under one method, this is the figure that says so.")
    A("")
    A(f"  {algo_fset} carries the algorithm half because it is the only feature set with")
    A("  an archetype run - concat_bands5 and concat_bands5z have none, so using FIG 1's")
    A("  reference here would have silently dropped a whole algorithm.")
    A("")
    A("EVERY SOLUTION LABELS THE SAME ELECTRODES IN THE SAME ORDER. That is what makes")
    A("any of this comparable, and it is checked rather than assumed: a solution that is")
    A("absent, at a different K, or over a different electrode set is dropped with a")
    A("message instead of being compared.")
    for half, dropped in (("feature sets", fs_dropped), ("algorithms", al_dropped)):
        if dropped:
            A(f"  dropped from the {half} half:")
            for m_, f_, why in dropped:
                A(f"    {m_}/{f_} - {why}")
    A("")
    A("PANELS A and D - how much any two solutions agree at all")
    A("-" * 100)
    A("Adjusted Rand index between every pair of partitions. Chance-corrected, so 0 is")
    A("'no better than random' and 1 is identical. It is a single number for a whole")
    A("partition and says nothing about WHICH clusters agree - that is B and E. It is")
    A("also dominated by the large clusters, so a low ARI can coexist with the big")
    A("structure agreeing.")
    A("")
    A("THE INSET IS MEAN PAIRWISE ARI AT EVERY K, with this K marked. Agreement read at")
    A("one K cannot say whether that K is a good place to be comparing; the inset can.")
    for nm, cv in (("feature sets", fs_curve), ("algorithms", al_curve)):
        if cv is not None:
            kk, vv = cv
            A(f"  {nm:<14} peaks at K={int(kk[int(np.argmax(vv))])} ({vv.max():.3f}); "
              f"at K={k} it is "
              f"{vv[list(kk).index(k)]:.3f}" if k in list(kk) else "")
    A("")
    for nm, M in (("feature sets", fs_ari), ("algorithms", al_ari)):
        off = M[~np.eye(len(M), dtype=bool)]
        A(f"  {nm:<14} off-diagonal {off.min():.3f} to {off.max():.3f}, "
          f"median {np.median(off):.3f}")
    A("")
    A("PANELS B and E - each cluster against the reference, and what the others think")
    A("-" * 100)
    A("Each solution's clusters are matched 1:1 to the reference's by Hungarian")
    A("assignment - on the correlation between loading columns where both sides are")
    A("graded, on shared electrode counts where either is a hard partition - using the")
    A("same match_clusters() FIG 1 orders its blocks with, so the two figures cannot")
    A("disagree about which cluster is which. Rows are the reference's clusters in FIG")
    A("1's block order, so row p here is the cluster drawn at block position p there.")
    A("")
    A("THE NUMBER IS A JACCARD: shared electrodes over the union of the two clusters, on")
    A("the hard labels whichever basis the match used, so it means one thing everywhere.")
    A("")
    A("A 1:1 ASSIGNMENT ALWAYS RETURNS A PARTNER, whether or not one exists. A low value")
    A("is therefore the interesting case: a cluster that was paired with something")
    A("because it had to be, not because the two describe the same electrodes.")
    A("")
    A("THIS IS NOT 'WHICH CLUSTERS SURVIVE', AND THE PANEL NO LONGER SAYS SO. Every")
    A("number in the matrix is agreement with ONE reference, so a cluster the reference")
    A("alone missed and a cluster nobody else found look identical from these columns.")
    A("The OTHERS column separates them: it is the strongest agreement between two")
    A("NON-reference solutions about the same row. A low row beside a high OTHERS means")
    A("the reference is the odd one out and the cluster survived elsewhere; those rows")
    A("are outlined green. On the algorithm half this is not hypothetical - k-means and")
    A("Ward can agree with each other more than either agrees with convex NMF.")
    A("")
    A(f"The colour scale runs 0 to {vmax:.2f}, shared by both halves and set from the")
    A("data: 0-1 left most of the bar unused once the reference column was dropped, and")
    A("every difference looked smaller than it was.")
    A("")
    for nm, J in (("feature sets", fs_J), ("algorithms", al_J)):
        w = J[:, 1:].min(axis=1) if J.shape[1] > 1 else J[:, 0]
        A(f"  {nm:<14} weakest match per cluster: {w.min():.2f} to {w.max():.2f}, "
          f"median {np.median(w):.2f}")
        bad = [f"pos {p} (c{int(order[p])})" for p in range(len(w)) if w[p] < 0.30]
        A(f"  {'':<14} below 0.30 somewhere: "
          + (", ".join(bad) if bad else "none"))
    A("")
    A("PANELS C and F - which electrodes are placed consistently")
    A("-" * 100)
    A("Every solution's labels are translated into the reference's numbering, then each")
    A("electrode is scored by the size of the largest group of solutions that agree on")
    A("it. The count runs 1..N: N means every solution puts it in the same cluster, 1")
    A("means no two do. Red is 1, green is N.")
    A("")
    A("IT IS THE MODAL ASSIGNMENT, NOT 'AGREES WITH THE REFERENCE'. Nothing privileges")
    A("one solution, so an electrode where the reference is the odd one out still reads")
    A("as agreement among the rest.")
    A("")
    A("WHY THIS IS NOT THE SAME QUESTION AS B AND E. A cluster can match well on average")
    A("and still be assembled from electrodes that no other solution groups together;")
    A("the Jaccard is a property of a cluster, this is a property of an electrode.")
    A("")
    for nm, cnt, sols, nul in (("feature sets", fs_cnt, fs_sols, fs_null),
                               ("algorithms", al_cnt, al_sols, al_null)):
        N = len(sols)
        A(f"  {nm:<14} all {N} agree: {100*(cnt == N).mean():.1f}%   "
          f"(chance {100*nul[-1]:.1f}%)   no two agree: "
          f"{100*(cnt <= 1).mean():.1f}%   median count {int(np.median(cnt))}")
    A("")
    A("THE CHANCE BASELINE is a permutation: each non-reference solution's labels are")
    A("shuffled - keeping its cluster sizes exactly - and then RE-MATCHED to the")
    A("reference, the way the real ones are. Re-matching is the point: the observed")
    A("pipeline optimises the alignment before counting, so a null that skipped that")
    A("step would be far too easy to beat and the panel would overstate itself. The")
    A("overlap basis is used throughout, because a permuted partition has no loadings")
    A("left to correlate. 200 permutations.")
    A("")
    A("AGREEMENT IS NOT CORRECTNESS. Four representations can agree perfectly on one")
    A("patient's electrode strip.")
    A(f"  clusters over half one patient   feature half {n_op_fs} of {k}   "
      f"algorithm half {n_op_al} of {k}   (FIG 1 panel D)")
    A("")
    A("PROVENANCE")
    A("-" * 100)
    for nm, sols in (("feature sets", fs_sols), ("algorithms", al_sols)):
        A(f"  {nm}")
        for s in sols:
            A(f"    {s['method']:<13} {s['fset']:<16} "
              f"{s['run'].relative_to(CLUST).as_posix()}"
              f"{'   (graded)' if s['G'] is not None else '   (hard labels)'}")
    A(f"  geometry and block order from {REF_FSET} ({REF_METHOD}) at K={k}")
    A(f"  built by      00_paper2_figures2_2.py --k {k} --algo-feature-set {algo_fset}")
    A(f"  built on      {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A("")
    A("WHAT THIS FIGURE DOES NOT SHOW")
    A("-" * 100)
    A("  - Nothing here is a statistical test. ARI, Jaccard and the agreement count are")
    A("    descriptive; none is scored against a null, and the 0.30 line in B and E is a")
    A("    readable convention, not a threshold with a false-positive rate.")
    A("  - 1:1 matching needs EQUAL K. Every solution here is cut at the same K, which")
    A("    is why this figure exists at one K at a time. Solutions at different K have")
    A("    no 1:1 correspondence and cannot be put on these panels at all.")
    A("  - Agreement is not correctness. Four representations can agree on a cluster")
    A("    that is one patient's electrode strip; FIG 1 panel D is what answers that.")
    A("  - The archetype runs cover concat_hg and concat_rawds only, so the algorithm")
    A("    half cannot currently be drawn on the 5-band feature sets.")
    P2.save_text("\n".join(L) + "\n", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--algo-feature-set", default=DEFAULT_ALGO_FSET)
    a = ap.parse_args()
    print(f"=== FIGURE 2 ===  K={a.k}")
    figure_2(a.k, a.algo_feature_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
