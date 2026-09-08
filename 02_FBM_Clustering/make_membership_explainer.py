#!/usr/bin/env python3
"""
make_membership_explainer.py - E.12 for the KISS tab: what is done to an electrode
before it is clustered, and what "belonging to a cluster" actually means.

    python make_membership_explainer.py

Four steps, each with the arithmetic beside it and real numbers from the cohort:

    A  two electrodes with the SAME shape and different loudness, as recorded
    B  z-scoring per band, then scaling to unit length: the two become the same point,
       which is the whole purpose - the fit sorts by shape, not by amplitude
    C  what convex NMF returns for an electrode: a weight on every cluster. A perfect
       fit is (1, 0, 0, ...); the real ones are mixtures, and argmax keeps one number
       and throws the rest away
    D  the cohort's own distribution of top loading, with the display thresholds on it

Panels A-C are constructed so each step can be checked by hand; D is measured from the
published run. Every number in the KISS text is read back from the JSON this writes.

WRITES  outputs/clustering/explainers/E12_membership.png
        outputs/clustering/explainers/E12_membership.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                     # noqa: E402

OUT = ROOT / "outputs" / "clustering" / "explainers"
INK, MUTED, GREY, PALE = "#1b1b1b", "#6b6b6b", "#b9b9b9", "#ededed"
LOUD, QUIET, ACC = "#1f5f8b", "#7fa8c4", "#a4501f"
K, FSET = 8, "concat_hg"


def foot(ax, text):
    """A note under a panel, in that panel's own width."""
    ax.text(0, -0.34, text, transform=ax.transAxes, fontsize=7.0, color=MUTED,
            va="top", linespacing=1.6)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- the toy pair: same shape, different loudness ---------------------------
    bands = ["1-20", "20-60", "60-100", "100-200", "200-500"]
    shape = np.array([0.4, 0.9, 2.1, 1.6, 0.6])          # dB against baseline
    loud, quiet = shape * 2.4, shape * 0.7

    def unit(v):
        return v / np.linalg.norm(v)

    u_loud, u_quiet = unit(loud), unit(quiet)

    # ---- the real distribution of top loading -----------------------------------
    rd = LR.newest_run("cnmf", FSET)
    G = np.load(rd / "loadings_by_k" / f"G_k{K:02d}.npy").astype(float)
    P = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    top = P.max(1)
    n = len(top)
    keep30, keep40 = int((top >= 0.30).sum()), int((top >= 0.40).sum())
    # a real mixed electrode and the most confident one in the cohort
    mixed_i = int(np.argmin(np.abs(top - np.median(top))))
    best_i = int(np.argmax(top))

    fig = plt.figure(figsize=(11.4, 6.9), facecolor="white")
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.0, 1.05], hspace=.62, wspace=1.6,
                  left=.065, right=.975, top=.855, bottom=.085)
    x = np.arange(len(bands))

    # ---- A ----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:4])
    ax.bar(x - .19, loud, .36, color=LOUD, label="a loud electrode")
    ax.bar(x + .19, quiet, .36, color=QUIET, label="a quiet one, same shape")
    ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=6.6, rotation=25, ha="right")
    ax.set_ylabel("dB vs baseline", fontsize=8, color=MUTED)
    ax.set_title("A   as recorded", fontsize=9.4, color=INK, loc="left", pad=6)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    foot(ax, f"‖loud‖ = {np.linalg.norm(loud):.2f}   ‖quiet‖ = "
         f"{np.linalg.norm(quiet):.2f}\na squared-Euclidean fit puts these far apart")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- B ----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 4:8])
    ax.bar(x - .19, u_loud, .36, color=LOUD)
    ax.bar(x + .19, u_quiet, .36, color=QUIET)
    ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=6.6, rotation=25, ha="right")
    ax.set_ylabel("unit-normed", fontsize=8, color=MUTED)
    ax.set_title("B   x / ‖x‖ : the same point", fontsize=9.4, color=INK, loc="left",
                 pad=6)
    foot(ax, f"both now have ‖x‖ = 1 and differ by "
         f"{np.abs(u_loud - u_quiet).max():.0e}\namplitude is no longer a cluster")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- C ----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 8:12])
    perfect = np.zeros(K); perfect[2] = 1.0
    real = P[mixed_i]
    ax.bar(np.arange(K) - .19, perfect, .36, color=GREY, label="a perfect fit")
    ax.bar(np.arange(K) + .19, real, .36, color=ACC,
           label=f"a median electrode (top {real.max():.2f})")
    ax.axhline(0.30, color=INK, lw=.8, ls=(0, (3, 2)))
    ax.text(K - .4, 0.315, "0.30", fontsize=6.6, color=INK, ha="right")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"c{i}" for i in range(K)], fontsize=6.4)
    ax.set_ylabel("P(belonging)", fontsize=8, color=MUTED)
    ax.set_title("C   what the fit returns per electrode", fontsize=9.4, color=INK,
                 loc="left", pad=6)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    foot(ax, f"argmax keeps ONE of these eight numbers: c{int(real.argmax())} at "
         f"{real.max():.2f},\nover c{int(np.argsort(real)[-2])} at "
         f"{np.sort(real)[-2]:.2f} — the rest is discarded")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- D ----------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0:12])
    ax.hist(top, bins=60, color=PALE, edgecolor=GREY, lw=.5)
    for thr, col, yf in ((0.30, ACC, .96), (0.40, INK, .80)):
        ax.axvline(thr, color=col, lw=1.4)
        ax.text(thr + .006, ax.get_ylim()[1] * yf,
                f"≥ {thr:.2f}  ·  {int((top >= thr).sum())} of {n} electrodes "
                f"({100*(top >= thr).mean():.0f}%)", fontsize=7.6, color=col, va="top")
    ax.axvline(1 / K, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.text(1 / K + .004, ax.get_ylim()[1] * .45, f"  1/K = {1/K:.3f}\n  no preference "
            "at all", fontsize=7.2, color=MUTED, va="top", linespacing=1.5)
    ax.set_xlabel("the electrode's LARGEST P(belonging), convex NMF on HFA, K = 8",
                  fontsize=8.4, color=MUTED)
    ax.set_ylabel("electrodes", fontsize=8, color=MUTED)
    ax.set_title(f"D   the cohort is graded, not sorted   ·   median top loading "
                 f"{np.median(top):.2f}   ·   run {rd.name}", fontsize=9.4, color=INK,
                 loc="left", pad=6)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)

    fig.text(.065, .935, "What is done to an electrode before it is clustered, and what "
             "belonging to a cluster means", fontsize=11.5, color=INK)
    fig.text(.065, .905, "every step below is applied to all 1688 electrodes; A to C are "
             "constructed so the arithmetic can be checked by hand, D is measured",
             fontsize=8.4, color=MUTED)

    png = OUT / "E12_membership.png"
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)

    js = dict(
        run=rd.name, k=K, feature_set=FSET, n=n,
        norm_loud=round(float(np.linalg.norm(loud)), 3),
        norm_quiet=round(float(np.linalg.norm(quiet)), 3),
        unit_gap=float(f"{np.abs(u_loud - u_quiet).max():.3e}"),
        median_top=round(float(np.median(top)), 3),
        keep30=keep30, keep40=keep40,
        pct30=round(100 * keep30 / n, 1), pct40=round(100 * keep40 / n, 1),
        one_over_k=round(1 / K, 3),
        median_first=round(float(P[mixed_i].max()), 3),
        median_second=round(float(np.sort(P[mixed_i])[-2]), 3),
        best_top=round(float(top[best_i]), 3),
        below_1k=int((top < 1 / K + 1e-12).sum()),
    )
    (OUT / "E12_membership.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
    print(f"  {png}")
    for k_, v in js.items():
        print(f"    {k_:<15} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
