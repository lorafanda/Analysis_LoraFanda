#!/usr/bin/env python3
"""make_k_tradeoff_figure.py - the two criteria for K, every feature set on one page.

    python make_k_tradeoff_figure.py                    all four sets, cnmf
    python make_k_tradeoff_figure.py --feature-set concat_bands15 concat_hfa
    python make_k_tradeoff_figure.py --method kmeans
    python make_k_tradeoff_figure.py --danger 0.5 --sets-dir <dir>   the time-bin sweep

WHY THIS FIGURE. FIG 1 already draws both of these, but one feature set at a time: panel
A is the held-out variance of every set with one emphasised, panel D is the patient
dominance of that one set. The question "which set, and at which K" needs them for all
sets at once, because the two criteria pull in opposite directions and the crossing point
is what decides. Held-out variance rises and then falls slowly, so it can nominate a peak
but cannot see the failure at all: splitting a cohort until each patient has their own
component fits held-out data perfectly well.

    LEFT    held-out variance explained vs K, one line per feature set. Marker = that
            set's own peak. This is FIG 1's panel A with no set emphasised.
    RIGHT   RED   the share of CLUSTERS more than `danger` one patient
            ORANGE the share of ELECTRODES sitting in those clusters
            one pair per feature set. Two of twenty small clusters failing is not the
            same as two of five big ones, and only the second number tells them apart.
            A filled marker is the largest K at which no cluster is over the line - the
            rule 249 now picks K with.

EVERYTHING IS READ, NOTHING IS FITTED. The variance comes from
outputs/clustering/bsf_comparison/heldout_variance_ALL.csv (240/241/242 write the three
halves of it); the dominance is computed from each run's own labels-by-K, through
00_Paper2_Figures.generalization_by_k so it is the same arithmetic as panel D. A feature
set with no run, or no held-out rows, is skipped with a note rather than failing.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

# the same idiom FIG 2 uses: the module name starts with a digit, so it cannot be
# imported by name, and reimplementing panel D would be a second arithmetic to keep true
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)

OUT = ROOT / "outputs" / "clustering" / "paper_figures"


def variance_curves(method: str, fsets: list[str]) -> dict[str, pd.DataFrame]:
    if not P2.HELDOUT.exists():
        raise SystemExit(f"{P2.HELDOUT} missing - run 240 / 241 / 242 (they write the "
                         "three halves of it) or make_heldout_variance.py")
    hv = pd.read_csv(P2.HELDOUT)
    hv = hv[(hv.method == method) & (hv.scheme == "home")]
    out = {}
    for fs in fsets:
        s = (hv[hv.feature_set == fs].groupby("k")["var_explained"]
             .agg(["mean", "std"]).reset_index().sort_values("k"))
        if not s.empty:
            out[fs] = s
    return out


def dominance_curves(method: str, fsets: list[str], danger: float) -> dict:
    """Panel D's table per feature set, via 00_Paper2_Figures so the arithmetic is one."""
    out = {}
    old = P2.DANGER
    P2.DANGER = float(danger)                 # panel D's threshold is a module constant
    try:
        for fs in fsets:
            try:
                d = P2.load_run(fs, None if method == "cnmf" else 8, method=method)
            except SystemExit as e:
                print(f"  [{fs}] no run to read dominance from - {e}")
                continue
            t = P2.generalization_by_k(d)
            if t.empty:
                print(f"  [{fs}] the run has no labels-by-K table")
                continue
            out[fs] = t
    finally:
        P2.DANGER = old
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--feature-set", nargs="+", default=list(P2.FSETS))
    ap.add_argument("--method", default="cnmf", choices=["cnmf", "kmeans", "hierarchical"])
    ap.add_argument("--danger", type=float, default=0.50,
                    help="a cluster is 'one patient' over this share (default 0.50, the "
                         "same constant panel D draws in red)")
    ap.add_argument("--out", default=None, help="png path (default: outputs/clustering/"
                                                "paper_figures/FIG_K_tradeoff_<method>.png)")
    a = ap.parse_args()
    fsets = list(a.feature_set)

    print(f"held-out variance, {a.method}:")
    var = variance_curves(a.method, fsets)
    for fs, s in var.items():
        pk = s.loc[s["mean"].idxmax()]
        print(f"  {fs:16s} {len(s):3d} K values, peak K = {int(pk.k):2d} at "
              f"{pk['mean']:.4f}")
    for fs in fsets:
        if fs not in var:
            print(f"  {fs:16s} no held-out rows - skipped")

    print(f"\npatient dominance (> {100*a.danger:.0f}% one patient), {a.method}:")
    dom = dominance_curves(a.method, fsets, a.danger)
    picks = {}
    for fs, t in dom.items():
        ok = t.k[t.n_dominated == 0].tolist()
        picks[fs] = int(max(ok)) if ok else None
        # generalization_by_k gives median_share, not the max: the largest single share
        # is not in its table, so the clean K is reported with the median instead
        med = t.median_share[t.k == picks[fs]].iloc[0] if picks[fs] else float("nan")
        print(f"  {fs:16s} K {t.k.min()}..{t.k.max()}   clean up to K = "
              f"{picks[fs] if picks[fs] else 'NONE'}"
              + (f"   (median cluster {100*med:.0f}% one patient there)"
                 if picks[fs] else ""))

    if not var and not dom:
        raise SystemExit("nothing to draw")

    fig, (axV, axD) = plt.subplots(1, 2, figsize=(13.4, 4.8))
    cols = plt.get_cmap("tab10").colors
    # red and orange have to stay the two QUANTITIES, as in panel D, so the feature set
    # is carried by the line style instead - otherwise the orange curves have no identity
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for i, fs in enumerate(fsets):
        c = cols[i % len(cols)]
        s = var.get(fs)
        if s is not None:
            axV.plot(s.k, s["mean"], "-o", ms=3.2, lw=1.8, color=c,
                     label=P2.FS_SHORT.get(fs, fs))
            axV.fill_between(s.k, s["mean"] - s["std"], s["mean"] + s["std"],
                             color=c, alpha=0.10, lw=0)
            pk = s.loc[s["mean"].idxmax()]
            axV.plot([pk.k], [pk["mean"]], marker="v", ms=6.0, color=c, lw=0)
    if var:
        # the std band explodes at the top of the sweep (one fold can go negative), which
        # drags the axis to 1.5 and flattens every curve; the MEANS are what is read here
        lo = min(float(s["mean"].min()) for s in var.values())
        hi = max(float(s["mean"].max()) for s in var.values())
        axV.set_ylim(lo - 0.06 * (hi - lo), hi + 0.10 * (hi - lo))
    axV.set_xlabel("K (components)", fontsize=9)
    axV.set_ylabel("held-out variance explained", fontsize=9)
    axV.legend(fontsize=8, frameon=False, loc="lower right", handlelength=1.6)
    axV.spines[["top", "right"]].set_visible(False)
    axV.set_title(f"held-out variance vs K,  {P2.METHOD_LABEL[a.method]}\n"
                  "marker = each set's own peak", fontsize=10, loc="left")

    for i, fs in enumerate(fsets):
        c = cols[i % len(cols)]
        t = dom.get(fs)
        if t is None:
            continue
        ls = styles[i % len(styles)]
        axD.plot(t.k, 100 * t.frac_clusters, ls=ls, lw=1.9, color=P2.RED, alpha=0.9,
                 zorder=4)
        axD.plot(t.k, 100 * t.frac_electrodes, ls=ls, lw=1.4, color="#e08214",
                 alpha=0.9, zorder=3)
        axD.plot([], [], ls=ls, lw=1.6, color="#555", label=P2.FS_SHORT.get(fs, fs))
        if picks.get(fs):
            axD.plot([picks[fs]], [0], marker="^", ms=8.0, color=c, lw=0, zorder=6,
                     clip_on=False)
    axD.axhline(0, color="#aab", lw=0.8, zorder=1)
    axD.set_xlabel("K (components)", fontsize=9)
    axD.set_ylabel(f"% of the solution that is one patient", fontsize=9)
    axD.set_ylim(-1, 102)
    axD.spines[["top", "right"]].set_visible(False)
    axD.legend(fontsize=8, frameon=False, loc="upper left", handlelength=2.4,
               title="line style = feature set", title_fontsize=7.6)
    axD.set_title(f"red = clusters > {100*a.danger:.0f}% one patient,   "
                  "orange = electrodes in them\ntriangle = largest K with none",
                  fontsize=10, loc="left")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = Path(a.out) if a.out else OUT / f"FIG_K_tradeoff_{a.method}.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nwrote -> {png}")

    cap = png.with_suffix("").with_name(png.stem + "_caption.txt")
    lines = [f"Held-out variance and patient dominance over K, {P2.METHOD_LABEL[a.method]}.",
             "", "Left: held-out variance explained, one line per feature set, marker at "
             "each set's own peak. Right: the share of clusters more than "
             f"{100*a.danger:.0f}% one patient (red) and the share of electrodes in those "
             "clusters (orange); the triangle marks the largest K at which no cluster is "
             "over that line.", ""]
    for fs in fsets:
        s, t = var.get(fs), dom.get(fs)
        bits = [P2.FS_LABEL.get(fs, fs) + ":"]
        if s is not None:
            bits.append(f"peak K {int(s.loc[s['mean'].idxmax(), 'k'])}")
        if t is not None:
            bits.append(f"clean up to K {picks[fs] if picks[fs] else 'none'}")
        lines.append("  " + "  ".join(bits))
    cap.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote -> {cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
