#!/usr/bin/env python3
"""
make_heldout_figure.py - PART 2. Held-out variance explained over components, for all
three algorithms, on the SBSF dataset and the bands feature set.

PART 2 IS A DIFFERENT COHORT FROM PART 1, deliberately. Part 1 (FIG C.3a/b/c, C.8a/b/c)
is the BSF run on concat_hg_all - 2946 electrodes, the gate lifted. Part 2 is the SBSF
cohort, concat_hg - 1266 electrodes, the gate applied - plus concat_rawds, which is the
SAME 1266 electrodes described by 15 bands x 3 conditions x 30 bins instead of high
gamma alone. So within Part 2 the two feature sets are one cohort seen two ways, and
nothing here is mixed with Part 1.

    SBSF   kmeans / concat_hg / 20260817_171544      K = 8   n = 1266
           and the Ward and cnmf runs on the identical X_train

TWO PANELS PER FEATURE SET, because "variance explained" is not space-free:

    home         each method fitted AND scored where it fits - cnmf unit-norm,
                 k-means and Ward raw dB. The SHAPE of each curve is meaningful;
                 the heights are NOT comparable across methods, because they are
                 explaining variance in two different matrices.

    unit-norm    every method in one space. Now the heights ARE comparable, at the
                 cost of scoring k-means and Ward somewhere they were not designed
                 to run. Both are shown rather than picking one and hoping.

    python make_heldout_figure.py
"""
from __future__ import annotations

import json
import sys
import textwrap
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
OUT = ROOT / "outputs" / "clustering" / "bsf_comparison"

INK, MUTED = "#1b232c", "#68727d"
RED, GREEN = "#c1121f", "#1b7837"
MCOL = {"k-means": "#4a6fa5", "Ward": "#e08214", "convex NMF": "#5b2c83"}
K_MARK = 8                       # overwritten per feature set from peak_k.json
FS_LABEL = {"concat_hg": "concat_hg  ·  high gamma, gate APPLIED",
            "concat_rawds": "concat_rawds  ·  same electrodes  ·  15 bands x time",
            "concat_hg_all": "concat_hg_all  ·  gate LIFTED"}
SCHEME_LABEL = {"home": "each method in its HOME space — compare SHAPE, not height",
                "unit-norm": "every method in UNIT-NORM — heights comparable"}


def one(ax, sub, title, show_legend):
    peaks = []
    for ml, d in sub.groupby("method_label"):
        s = (d.groupby("k")["var_explained"].agg(["mean", "std"])
             .reset_index().sort_values("k"))
        c = MCOL.get(ml, MUTED)
        ax.plot(s.k, s["mean"], "-o", ms=3.2, lw=1.7, color=c, label=ml, zorder=3)
        ax.fill_between(s.k, s["mean"] - s["std"], s["mean"] + s["std"],
                        color=c, alpha=0.13, lw=0)
        pk = s.loc[s["mean"].idxmax()]
        mono = bool(s["mean"].is_monotonic_increasing)
        peaks.append((ml, int(pk.k), float(pk["mean"]), mono,
                      float(s.loc[s.k == K_MARK, "mean"].iloc[0]) if (s.k == K_MARK).any() else np.nan))
        if not mono:
            ax.plot([pk.k], [pk["mean"]], marker="v", ms=7, color=c, zorder=4)
    ax.axvline(K_MARK, color=RED, ls="--", lw=1.0, zorder=1)
    ax.annotate(f"K={K_MARK}", xy=(K_MARK, ax.get_ylim()[0]), xytext=(2, 2),
                textcoords="offset points", fontsize=7.4, color=RED)
    ax.set_xlabel("K  (components / clusters)", fontsize=8.5)
    ax.set_ylabel("held-out variance explained", fontsize=8.5)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=9.2, loc="left", color=INK, pad=5)
    if show_legend:
        ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    # the numbers, on the figure rather than in a caption
    txt = []
    for ml, kp, pv, mono, v8 in sorted(peaks):
        turn = f"peak k={kp}" if not mono else f"still rising at k={kp}"
        txt.append(f"{ml:<11} K={K_MARK} {v8:.3f}   {turn} ({pv:.3f})")
    ax.text(0.015, 0.975, "\n".join(txt), transform=ax.transAxes, va="top",
            fontsize=6.8, color=INK, linespacing=1.45, family="DejaVu Sans Mono",
            bbox=dict(facecolor="white", alpha=0.82, lw=0, pad=2.4))
    return peaks


def _krange(ks):
    """5..30, rather than twenty-six comma-separated integers."""
    ks = sorted(set(int(k) for k in ks))
    if len(ks) > 3 and ks == list(range(ks[0], ks[-1] + 1)):
        return f"{ks[0]}..{ks[-1]} in steps of 1"
    return ", ".join(str(k) for k in ks)


def _turnovers(df, fsets, schemes):
    """How many curves turn over inside the range - WITHOUT drawing anything.

    The caption quotes this number, and the caption has to be wrapped before the
    figure can be sized, so it cannot wait for the axes to exist.
    """
    tot = turn = 0
    for fs in fsets:
        for sc in schemes:
            sub = df[(df.feature_set == fs) & (df.scheme == sc)]
            for _, d in sub.groupby("method_label"):
                m = d.groupby("k")["var_explained"].mean().sort_index()
                tot += 1
                turn += 0 if m.is_monotonic_increasing else 1
    return turn, tot


def main() -> int:
    # 249 merges the three sweeps into heldout_variance_ALL.csv. PREFER IT: the older
    # part2_heldout_variance.csv is the 25-Aug sweep on the pre-fix 1266-electrode
    # cohort, and reading it here was the last place a stale cohort still reached the
    # site. Kept as the fallback so tab 07's own Part 2 figure can still be rebuilt.
    f = OUT / "heldout_variance_ALL.csv"
    new = f.exists()
    if not new:
        f = OUT / "part2_heldout_variance.csv"
    if not f.exists():
        raise SystemExit("run make_heldout_variance.py first")
    df = pd.read_csv(f)
    print(f"reading {f.name}  ({df.n.iloc[0]} electrodes, "
          f"K={df.k.min()}..{df.k.max()}, schemes: {sorted(set(df.scheme))})")

    # K is no longer one number for the whole figure: each feature set is cut at its
    # OWN convex-NMF peak, which 249 writes here.
    pkf = OUT / "peak_k.json"
    PEAK = json.loads(pkf.read_text()) if (new and pkf.exists()) else {}

    # one meta per --tag; they agree on folds / n_iter / ks
    meta = {}
    for m in (("cnmf", "kmeans", "hierarchical") if new else ()):
        g = OUT / f"heldout_meta_{m}.json"
        if g.exists():
            meta = json.loads(g.read_text())
            break
    if not meta and (OUT / "part2_meta.json").exists():
        meta = json.loads((OUT / "part2_meta.json").read_text())

    fsets = [x for x in ("concat_hg", "concat_rawds") if x in set(df.feature_set)]
    schemes = [x for x in ("home", "unit-norm") if x in set(df.scheme)]
    if not fsets:
        raise SystemExit("feature sets not present in the CSV yet")

    n_turn, n_tot = _turnovers(df, fsets, schemes)
    body = [
        "Bi-cross-validation, not the electrode-only scheme. A block of ROWS and a "
        "block of COLUMNS is held out; the method is fitted on the remaining block; "
        "each held-out electrode's loadings come from the TRAIN columns only and are "
        "scored on the TEST columns. The loadings therefore never see the values they "
        "are graded on, so an extra component has to earn its place. The curve "
        "elsewhere on the site holds out electrodes only and refits them across the "
        "full feature set, which makes it monotone in K by construction and useless "
        "for choosing K or for comparing methods.",
        "The only thing that differs between the three methods is how a held-out "
        "electrode's loadings are obtained - NNLS against the components for convex "
        "NMF, nearest centroid for k-means and Ward - which is exactly what "
        "distinguishes them. Everything else, including the folds, is identical, and "
        "all three read the same X_train, verified bit-identical.",
        f"{meta.get('row_folds','?')}x{meta.get('col_folds','?')} folds, "
        f"n_iter={meta.get('n_iter','?')} for convex NMF, "
        f"K = {_krange(df.k)}, n={int(df.n.iloc[0])} electrodes. "
        f"A triangle marks a curve that TURNS OVER inside the tested range; "
        f"{n_turn} of {n_tot} do, and the dashed line is the K this feature set is "
        f"cut at. A curve still rising at the largest K tested has not been shown to "
        f"have an optimum - it has been shown that this range did not find one.",
    ]

    # SIZE THE FIGURE AROUND THE CAPTION rather than guessing a fraction of it. The
    # old top=0.995-0.115*rows assumed the text wrapped over TWO columns; with only
    # the home scheme swept there is one column, the same words wrap to twice as many
    # lines, and the header ran straight through panel A1's title.
    W = 7.4 * len(schemes)
    cap = "\n".join(textwrap.fill(b, width=int(15.6 * W)) for b in body)
    cap_in = (cap.count("\n") + 1) * 8.4 * 1.5 / 72 + 0.66        # + the suptitle
    H = 4.9 * len(fsets) + cap_in

    fig = plt.figure(figsize=(W, H), dpi=175)
    gs = GridSpec(len(fsets), len(schemes), figure=fig, hspace=0.46, wspace=0.22,
                  left=0.065, right=0.98,
                  top=1.0 - (cap_in + 0.34) / H, bottom=0.075)
    fig.suptitle("FIG C.13   \u00b7   Held-out variance explained over components   "
                 "\u00b7   three algorithms, one cohort",
                 x=0.065, y=1.0 - 0.10 / H, ha="left", va="top",
                 fontsize=15.5, color=INK)
    fig.text(0.065, 1.0 - 0.52 / H, cap, fontsize=8.4, color=MUTED, va="top",
             linespacing=1.5)

    global K_MARK
    allpk = []
    for r, fs in enumerate(fsets):
        for c, sc in enumerate(schemes):
            sub = df[(df.feature_set == fs) & (df.scheme == sc)]
            if sub.empty:
                continue
            ax = fig.add_subplot(gs[r, c])
            n = int(sub["n"].iloc[0])
            K_MARK = int(PEAK.get(fs, 8))
            title = (f"{'AB'[c]}{r+1} \u00b7 {FS_LABEL.get(fs, fs)}   n={n}\n"
                     f"{SCHEME_LABEL[sc]}")
            for ml, kp, pv, mono, vm in one(ax, sub, title,
                                            show_legend=(r == 0 and c == 0)):
                allpk.append(dict(feature_set=fs, scheme=sc, method_label=ml,
                                  k_peak=kp, peak=pv, k_mark=K_MARK, at_k_mark=vm,
                                  monotone=mono))

    pk = pd.DataFrame(allpk)
    DEST = (ROOT / "outputs" / "clustering" / "statistics") if new else OUT
    DEST.mkdir(parents=True, exist_ok=True)
    pk.to_csv(DEST / "heldout_peaks_figure.csv", index=False)
    p = DEST / "C13_heldout_variance.png"
    fig.savefig(p, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(pk.to_string(index=False))
    print(f"\n-> {p}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
