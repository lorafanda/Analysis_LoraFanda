#!/usr/bin/env python3
"""
make_cnmf_on_data_explainer.py - convex NMF, worked through on THIS cohort's own data.

    python make_cnmf_on_data_explainer.py
    python make_cnmf_on_data_explainer.py --k 3 --bins 10 --n 6

E1_cnmf_explained.png does the same job on twelve invented electrodes with a planted
ground truth, which is the right way to show that the method recovers what it should.
This one gives up the ground truth to gain something else: every number on the figure is
a real measurement from the cohort, so the reader can see what the factorisation does to
data that was not built to be factorised.

THE ONE SIMPLIFICATION, and it is only resolution. concat_hg is the 70-150 Hz mean per
time bin, 300 bins per condition, 900 numbers per electrode. Here each condition's 300
bins are averaged into TEN, so an electrode is 30 numbers instead of 900 and every matrix
fits on the page at readable precision. Nothing else is changed: the values are dB
relative to the (-0.4, -0.1 s) baseline, they are signed, and they go through the
project's own lf_decompose.convex_nmf.

WHAT THE FIGURE HAS TO MAKE OBVIOUS

    X  ~=  G (W' X)          the whole model in one line

  A  the data is SIGNED. Plain NMF requires X >= 0 and cannot be used: on this sample
     more than half the values are below baseline. Convex NMF splits the Gram matrix
     into A+ and A- and needs non-negativity only of W and G (Ding, Li & Jordan 2010).
  B  a COMPONENT is a real response. W's columns sum to 1, so W'X is a weighted AVERAGE
     of recorded electrodes: it keeps dB units and can be plotted beside them.
  C  a LOADING is graded. G says how much of each component an electrode expresses, and
     an electrode that matches nothing gets near-zero everywhere rather than a label.
  D  the argmax is imposed afterwards, and that is where the grading is lost.

CITATIONS, all for the method as used here and not as a general endorsement:
  Ding, Li & Jordan (2010) IEEE TPAMI 32(1):45-55, doi 10.1109/TPAMI.2008.277 - the
    algorithm and the A+/A- split.
  Every reference here was checked against Crossref on 2026-09-27: first author, journal,
    year, volume and pages all match what is printed on the figure.
  Hamilton, Edwards & Chang (2018) Curr Biol 10.1016/j.cub.2018.04.033 - cNMF on
    intracranial high-gamma.
  Hamilton et al. (2021) Cell 10.1016/j.cell.2021.07.019 - on signed z-scored high-gamma.
  Kurteff et al. (2024) J Neurosci 10.1523/JNEUROSCI.1109-24.2024 - across concatenated
    conditions, as here.
  Norman-Haignere et al. (2022) Curr Biol 10.1016/j.cub.2022.01.069 - rank by
    cross-validated prediction rather than an internal index.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_decompose as LD                                             # noqa: E402

OUT = ROOT / "outputs" / "clustering" / "explainers"
INK, MUTED = "#1b232c", "#68727d"
RED, GREEN, BLUE = "#c1121f", "#1b7837", "#2471a3"
COMPC = ["#5b2c83", "#1b7837", "#c1121f", "#2471a3", "#e08214"]
MONO = {"family": "DejaVu Sans Mono"}
HG = (70.0, 150.0)
FMAX = 398.4375                        # 102 x 3.90625 Hz, the last bin of the 0-400 cube


def newest_cache() -> Path:
    d = ROOT / "outputs" / "_dataset"
    caches = sorted(d.glob("concat_source_v*"),
                    key=lambda p: int(p.name.rsplit("v", 1)[1]))
    if not caches:
        raise SystemExit("no concat_source_v<N> cache")
    return caches[-1]


def load(n_elec: int, n_bins: int):
    """n_elec cohort electrodes, HFA in n_bins per condition, from the newest cache."""
    cache = newest_cache()
    params = json.load(open(cache / "params.json"))
    meta = pd.read_parquet(cache / "df_meta.parquet")
    freqs = np.linspace(0, FMAX, params["n_freq"])
    band = (freqs >= HG[0]) & (freqs <= HG[1])
    X3 = np.load(cache / "X_3d.npy", mmap_mode="r")

    meta = meta.assign(key=meta.patient_id.astype(str) + "|" + meta.electrode.astype(str))
    per = meta.groupby("key").condition.nunique()
    hot = meta.groupby("key").high_activity.any()
    cohort = sorted(set(per[per == 3].index) & set(hot[hot].index))

    sel, seen = [], set()                      # one per patient, so it is not one shaft
    for k in cohort:
        p = k.split("|")[0]
        if p not in seen:
            seen.add(p)
            sel.append(k)
        if len(sel) == n_elec:
            break

    conds = params["conditions"]
    rows = []
    for k in sel:
        m = meta[meta.key == k]
        parts = []
        for c in conds:
            i = int(m[m.condition == c].sample_idx.iloc[0])
            hg = np.asarray(X3[i], dtype=np.float64)[band].mean(0)
            parts.append(hg.reshape(n_bins, -1).mean(1))
        rows.append(np.concatenate(parts))
    return np.vstack(rows), sel, conds, cache.name, int(band.sum()), params["n_freq"]


def _fit_scale(X, Wn, G, comp):
    """Per-component factor that makes G @ comp the model the iterations converged to.

    The least-squares rescaling of each component is the one that minimises
    ||X - G diag(a) comp||, and for a correctly-paired (G, comp) that is exactly 1.

    Since the 2026-09-27 fix convex_nmf pairs them itself, so this returns ~1 and the
    multiplication at the call site is a no-op.  It is kept as a guard: it is computed
    from the returned factors rather than assumed, so if the pairing ever regresses the
    figure still draws the real fit instead of silently reporting a shrunken one.
    """
    k = comp.shape[0]
    # normal equations for a: (G'G * comp comp') a = diag(G' X comp')
    M = (G.T @ G) * (comp @ comp.T)
    b = np.diag(G.T @ X @ comp.T)
    a = np.linalg.solve(M, b)
    return a


def heat(ax, M, vmax=None, cmap="RdBu_r", fs=5.4, fmt="{:.1f}"):
    v = vmax if vmax is not None else np.abs(M).max()
    ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, aspect="auto", interpolation="none")
    if M.size <= 200:
        for (r, c), val in np.ndenumerate(M):
            ax.text(c, r, fmt.format(val), ha="center", va="center", fontsize=fs,
                    color=("white" if abs(val) > 0.62 * v else INK), **MONO)
    ax.set_xticks([]); ax.set_yticks([])


def main(k, n_bins, n_elec):
    X, names, conds, cache_name, n_hg, n_freq = load(n_elec, n_bins)
    n, p = X.shape
    neg = int((X < 0).sum())
    W, G, comp = LD.convex_nmf(X, k, random_state=0, n_iter=300)
    # convex_nmf pairs G with the normalised components itself since 2026-09-27, so this
    # is a no-op guard rather than a repair (see _fit_scale). Before that fix G was left
    # on the un-normalised scale and G @ comp came out shrunk - 80.3% of variance on this
    # sample instead of the 86.6% the iterations reached. Keeping the line means panels
    # A/C/D are guaranteed to reconstruct each other whatever upstream does.
    G = G * _fit_scale(X, W, G, comp)
    R = G @ comp
    rel = np.linalg.norm(X - R) / np.linalg.norm(X)
    A = X @ X.T
    Ap, An = (np.abs(A) + A) / 2.0, (np.abs(A) - A) / 2.0
    short = [nm.replace("|", " ") for nm in names]

    fig = plt.figure(figsize=(15.4, 10.6), dpi=200)
    gs = GridSpec(4, 6, figure=fig, height_ratios=[1.30, 0.92, 1.05, 1.02],
                  hspace=0.62, wspace=0.42)

    fig.suptitle("Convex NMF, worked through on this cohort's own high-gamma  —  "
                 f"X  ≈  G (W′X)", fontsize=14.5, color=INK, y=0.985)
    fig.text(0.5, 0.955,
             f"{n} electrodes of {cache_name}, one per patient · high-gamma "
             f"({HG[0]:g}–{HG[1]:g} Hz, {n_hg} of {n_freq} frequency bins) averaged into "
             f"{n_bins} time bins per condition · {p} numbers per electrode, dB re baseline",
             ha="center", fontsize=9.6, color=MUTED)

    # ── A · the data ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    heat(ax, X, fs=5.0)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7.4, **MONO)
    for b in range(1, len(conds)):
        ax.axvline(b * n_bins - 0.5, color=INK, lw=1.6)
    for b, c in enumerate(conds):
        ax.text(b * n_bins + n_bins / 2 - 0.5, -0.92, c, ha="center", fontsize=9, color=INK)
    ax.set_title(f"A · X, the data — {n} × {p}.   "
                 f"{neg} of {X.size} values are NEGATIVE (below baseline), which is why "
                 f"plain NMF cannot be used: it requires X ≥ 0.",
                 fontsize=10.2, color=INK, loc="left", pad=16)

    # ── B · the Gram split ──────────────────────────────────────────────────────
    for j, (M, t, cm) in enumerate([
            (A, "A = X X′   (signed)", "RdBu_r"),
            (Ap, "A⁺ = (|A|+A)/2", "Reds"),
            (An, "A⁻ = (|A|−A)/2", "Reds")]):
        ax = fig.add_subplot(gs[1, j])
        if cm == "Reds":
            ax.imshow(M, cmap=cm, vmin=0, vmax=np.abs(A).max(), aspect="auto",
                      interpolation="none")
            for (r, c), val in np.ndenumerate(M):
                ax.text(c, r, f"{val:.0f}", ha="center", va="center", fontsize=5.6,
                        color=("white" if val > 0.62 * np.abs(A).max() else INK), **MONO)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            heat(ax, M, fs=5.6, fmt="{:.0f}")
        ax.set_title(t, fontsize=9, color=INK, loc="left")
    ax = fig.add_subplot(gs[1, 3:])
    ax.axis("off")
    ax.text(0, 1.00, "B · how the sign is carried", fontsize=10.2, color=INK, va="top")
    # the citation is the paragraph's last line, not a separate text object: a free-floating
    # one at a fixed y lands on top of the prose as soon as the wrap runs a line longer.
    ax.text(0, 0.70, textwrap.fill(
        "Convex NMF never asks X to be positive. It works on A = X X′ and splits it into "
        "two non-negative halves, A = A⁺ − A⁻. Only W and G must stay non-negative — the "
        "DATA keeps its sign, which is what makes the method usable on dB at all.", 58)
        + "\n\nDing, Li & Jordan (2010), IEEE TPAMI 32(1):45–55\ndoi 10.1109/TPAMI.2008.277",
        fontsize=8.8, color=MUTED, va="top", linespacing=1.6)

    # ── C · W and G ─────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(W, cmap="Purples", vmin=0, vmax=W.max(), aspect="auto", interpolation="none")
    for (r, c), val in np.ndenumerate(W):
        ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6.6,
                color=("white" if val > 0.62 * W.max() else INK), **MONO)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=6.6, **MONO)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"c{j}" for j in range(k)], fontsize=8)
    ax.set_title("C · W\ncolumns sum to 1", fontsize=9.2, color=INK, loc="left")

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(G, cmap="Greens", vmin=0, vmax=G.max(), aspect="auto", interpolation="none")
    for (r, c), val in np.ndenumerate(G):
        ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6.6,
                color=("white" if val > 0.62 * G.max() else INK), **MONO)
    ax.set_yticks([]); ax.set_xticks(range(k))
    ax.set_xticklabels([f"c{j}" for j in range(k)], fontsize=8)
    ax.set_title("G\ngraded membership", fontsize=9.2, color=INK, loc="left")

    ax = fig.add_subplot(gs[2, 2:4]); ax.axis("off")
    tot = G.sum(1)
    ax.text(0, 1.02, "what G says, electrode by electrode", fontsize=9.4, color=INK)
    # The SHARE alone is a trap: an electrode with loadings (0.04, 0.00) reads "100% / 0%"
    # exactly like one with (1.61, 0.00). The total is printed beside it so a near-zero
    # row is visible as near-zero rather than as a confident assignment.
    faint = 0.25 * np.median(tot[tot > 1e-9]) if (tot > 1e-9).any() else 0.0
    for i in range(n):
        sh = G[i] / tot[i] if tot[i] > 1e-9 else np.zeros(k)
        j = int(np.argmax(sh))
        txt = "  ".join(f"c{q} {sh[q]:3.0%}" for q in range(k)) + f"   Σ {tot[i]:.2f}"
        if tot[i] <= faint:
            col = "#b8860b"          # a state, not a class: too little loading to assign
            txt += "  ← faint"
        else:
            col = COMPC[j % len(COMPC)] if sh[j] > 0.6 else MUTED
        ax.text(0, 0.90 - i * 0.145, short[i], fontsize=7.6, color=INK, **MONO)
        ax.text(0.42, 0.90 - i * 0.145, txt, fontsize=7.6, color=col, **MONO)
    ax.text(0, 0.90 - n * 0.145 - 0.02, textwrap.fill(
        "Grey means no component holds a majority — the electrode is genuinely mixed. "
        "A hard method must still hand it one label; this one records the mixture, and "
        "records when there is almost nothing to assign.", 62),
        fontsize=8.2, color=MUTED, va="top", linespacing=1.5)

    ax = fig.add_subplot(gs[2, 4:]); ax.axis("off")
    ax.text(0, 0.96, "why the components are plottable", fontsize=10.2, color=INK)
    ax.text(0, 0.72, textwrap.fill(
        "A component is W′X — a weighted average of REAL electrodes, because W's columns "
        "sum to 1. So it is not an abstract axis: it carries dB units and can be drawn on "
        "the same scale as the data it came from. The argmax of G is only a summary "
        "imposed afterwards, to make the result comparable with k-means and Ward, and it "
        "is exactly where the grading is thrown away.", 76),
        fontsize=8.8, color=MUTED, va="top", linespacing=1.55)

    # ── D · the components, and what is left ────────────────────────────────────
    ax = fig.add_subplot(gs[3, :4])
    t = np.arange(p)
    for j in range(k):
        ax.plot(t, comp[j], lw=2.0, color=COMPC[j % len(COMPC)], label=f"component {j}")
    ax.axhline(0, color=MUTED, lw=0.8)
    for b in range(1, len(conds)):
        ax.axvline(b * n_bins - 0.5, color=INK, lw=1.2)
    for b, c in enumerate(conds):
        ax.text(b * n_bins + n_bins / 2 - 0.5, ax.get_ylim()[1], c, ha="center",
                va="bottom", fontsize=8.4, color=MUTED)
    ax.set_xlim(-0.5, p - 0.5); ax.set_xticks([])
    ax.set_ylabel("dB re baseline", fontsize=8.6)
    ax.legend(fontsize=8, frameon=False, ncol=k, loc="upper left",
              bbox_to_anchor=(0.0, -0.02))          # under the axes, off the traces
    ax.set_title(f"D · the {k} components, in the data's own units", fontsize=10.2,
                 color=INK, loc="left", pad=14)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3, 4:]); ax.axis("off")
    ax.text(0, 0.96, "how much it accounts for", fontsize=10.2, color=INK)
    ax.text(0, 0.66,
            f"‖X − G W′X‖ / ‖X‖  =  {rel:.3f}\n"
            f"{100*(1-rel**2):.0f}% of the variance, at k = {k}",
            fontsize=10.4, color=GREEN, va="top", **MONO)
    ax.text(0, 0.42, textwrap.fill(
        "k is not chosen here. The pipeline reads it from held-out reconstruction "
        "(lf_decompose.cv_rank_curve): fit without a fold of electrodes, project that "
        "fold on by non-negative least squares, measure its error. A rank that overfits "
        "scores WORSE — which silhouette cannot do.", 56)
        + "\n\nNorman-Haignere et al. (2022), Curr Biol",
        fontsize=8.6, color=MUTED, va="top", linespacing=1.6)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "E13_cnmf_on_data.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta_out = dict(cache=cache_name, electrodes=names, k=k, bins_per_condition=n_bins,
                    conditions=conds, n_features=p, n_negative=neg, n_values=int(X.size),
                    rel_error=round(float(rel), 4),
                    var_explained=round(float(1 - rel ** 2), 4),
                    W=np.round(W, 4).tolist(), G=np.round(G, 4).tolist(),
                    X=np.round(X, 3).tolist(), components=np.round(comp, 3).tolist())
    (OUT / "E13_cnmf_on_data.json").write_text(json.dumps(meta_out, indent=1))
    print(f"wrote {png}")
    print(f"      {OUT / 'E13_cnmf_on_data.json'}")
    print(f"  {n} electrodes x {p} features, {neg} negative values, "
          f"k={k}, rel err {rel:.4f} ({100*(1-rel**2):.1f}% var)")
    for i, nm in enumerate(names):
        tot_i = G[i].sum()
        sh = "all zero" if tot_i < 1e-9 else " ".join(f"{v:.0%}" for v in G[i] / tot_i)
        print(f"      {nm:<20} G {np.round(G[i], 3)}   share {sh}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--bins", type=int, default=10, help="time bins per condition")
    ap.add_argument("--n", type=int, default=6, help="electrodes")
    a = ap.parse_args()
    main(a.k, a.bins, a.n)
