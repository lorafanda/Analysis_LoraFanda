#!/usr/bin/env python3
"""
make_gate_examples.py - what the responsiveness gate actually looks like at the line.

THE GATE, exactly as prepare_dataset computes it. For each electrode-condition ERSP
(129 frequencies x 300 time-normalised bins, 0-400 Hz, dB re pre-stimulus baseline):

    prop_above_pos = fraction of bins >  +2.2 dB
    prop_below_neg = fraction of bins <  -3.0 dB
    high_activity  = (prop_above_pos >= 0.02) OR (prop_below_neg >= 0.04)

n_high_activity counts how many of the three conditions pass; the contact is kept if
at least one does. 1323 of 3002 contacts pass, 1679 do not.

TWO THINGS THIS FIGURE IS FOR. First, the criterion counts BINS OVER A THRESHOLD across
the whole 0-400 Hz cube - it is not a high-gamma measure and it is not a test of whether
the response is repeatable. Second, at the line the pass and fail cases look the same,
which is the argument for replacing the criterion rather than moving it.

Margin m = max(prop_pos/0.02, prop_neg/0.04), so m >= 1 IS the pass rule and the two
criteria become comparable. Verified to reproduce the stored flag on 100% of 9279
electrode-conditions.

    python make_gate_examples.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "functions"))

# v4, not v2: v2 was deleted on 2026-08-28 and this is the only script that read it
# directly rather than through build_concat_dataset.
CACHE = ROOT / "outputs" / "_dataset" / "concat_source_v4" / "df_meta.parquet"
OUT = ROOT / "outputs" / "clustering" / "gate_examples"

THR_P, MIN_P, THR_N, MIN_N = 2.2, 0.02, -3.0, 0.04
VMIN, VMAX = -7.0, 7.0          # the project's own ERSP scale (ERSPParams)
FMAX = 400.0
INK, MUTED = "#1b232c", "#68727d"
GREEN, RED = "#1b7837", "#c1121f"
N_EACH = 3


def load_meta():
    d = pd.read_parquet(CACHE)
    d["m_pos"] = d["prop_above_pos"] / MIN_P
    d["m_neg"] = d["prop_below_neg"] / MIN_N
    d["margin"] = d[["m_pos", "m_neg"]].max(axis=1)
    assert ((d["margin"] >= 1.0) == d["high_activity"]).all(), \
        "margin rule does not reproduce the stored high_activity flag"
    return d


def pick(d):
    """N_EACH contacts just inside the gate and N_EACH just outside.

    Chosen at the CONTACT level (the gate's own unit) and, where possible, paired
    within patient so the comparison is not confounded by who was recorded.
    """
    g = d.groupby(["patient_id", "electrode"])
    full = g["condition"].nunique() == 3
    e = g.agg(best=("margin", "max"), n_high=("high_activity", "sum"))
    e = e.loc[full[full].index].reset_index()
    ins = e[e.n_high >= 1].nsmallest(12, "best")
    out = e[e.n_high == 0].nlargest(12, "best")
    # prefer patients that appear on both sides
    shared = set(ins.patient_id) & set(out.patient_id)
    ins = pd.concat([ins[ins.patient_id.isin(shared)], ins]).drop_duplicates(
        ["patient_id", "electrode"]).head(N_EACH)
    out = pd.concat([out[out.patient_id.isin(shared)], out]).drop_duplicates(
        ["patient_id", "electrode"]).head(N_EACH)
    return ins, out


def best_row(d, pid, elec):
    sub = d[(d.patient_id == pid) & (d.electrode == elec)]
    return sub.loc[sub["margin"].idxmax()]


def panel(ax, r, passed):
    arr = np.load(r["file_path"])
    n_f, n_t = arr.shape
    x = np.linspace(0, 100, n_t)
    f = np.linspace(0, FMAX, n_f)
    ax.pcolormesh(x, f, arr, cmap="bwr", vmin=VMIN, vmax=VMAX, shading="auto",
                  rasterized=True)
    # the bins the criterion actually counts
    ax.contour(x, f, (arr > THR_P).astype(float), levels=[0.5],
               colors=["#111111"], linewidths=0.45)
    if r["prop_below_neg"] > 0:
        ax.contour(x, f, (arr < THR_N).astype(float), levels=[0.5],
                   colors=["#111111"], linewidths=0.45, linestyles="dotted")
    ax.axvline(50, color="#222222", lw=0.9, ls=(0, (4, 3)))
    ax.set_ylim(0, FMAX)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["stim", "GO", "end"], fontsize=7.5)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    col = GREEN if passed else RED
    ax.set_title(f"{r['patient_id']} · {r['electrode']} · {r['condition']}",
                 fontsize=8.6, color=INK, pad=3.5)
    n_bins = n_f * n_t
    n_pos = round(r["prop_above_pos"] * n_bins)
    n_neg = round(r["prop_below_neg"] * n_bins)
    need_p, need_n = round(MIN_P * n_bins), round(MIN_N * n_bins)
    ax.set_xlabel(
        f"above +2.2 dB: {n_pos} bins   (needs {need_p})\n"
        f"below −3.0 dB: {n_neg} bins   (needs {need_n})\n"
        f"→ {'KEPT' if passed else 'DISCARDED'}",
        fontsize=7.6, color=col, labelpad=5,
        fontweight="bold" if passed else "normal")
    for s in ax.spines.values():
        s.set_color(col)
        s.set_linewidth(1.4)


def main() -> int:
    d = load_meta()
    ins, out = pick(d)
    OUT.mkdir(parents=True, exist_ok=True)

    rows_in = [best_row(d, r.patient_id, r.electrode) for r in ins.itertuples()]
    rows_out = [best_row(d, r.patient_id, r.electrode) for r in out.itertuples()]

    fig = plt.figure(figsize=(12.4, 8.6), dpi=200)
    gs = GridSpec(2, N_EACH, hspace=0.60, wspace=0.20,
                  left=0.055, right=0.90, top=0.735, bottom=0.075)

    for j, r in enumerate(rows_in):
        panel(fig.add_subplot(gs[0, j]), r, True)
    for j, r in enumerate(rows_out):
        panel(fig.add_subplot(gs[1, j]), r, False)

    fig.text(0.055, 0.765, "KEPT — just over the line", fontsize=10.5,
             color=GREEN, fontweight="bold")
    fig.text(0.055, 0.378, "DISCARDED — just under it", fontsize=10.5,
             color=RED, fontweight="bold")

    cax = fig.add_axes([0.915, 0.075, 0.014, 0.59])
    sm = plt.cm.ScalarMappable(cmap="bwr",
                               norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("dB re baseline", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)

    fig.suptitle("What the responsiveness gate is actually deciding",
                 x=0.055, y=0.972, ha="left", fontsize=15, color=INK)
    body = [
        "The gate counts BINS OVER A THRESHOLD in the whole 0-400 Hz cube, then keeps a "
        "contact if any one of its three conditions passes: at least 2% of bins above "
        "+2.2 dB, OR at least 4% below -3.0 dB. Outlined regions are the bins being "
        "counted - solid for positive, dotted for suppression.",
        "1323 of 3002 contacts pass, 1679 do not. Every panel below shows that "
        "contact's BEST condition, and all six sit within 0.3% of the line.",
        "Read the bin counts, not the percentages. Each cube holds 129 x 300 = 38,700 "
        "bins, so the 2% rule is a count: 774. EL033 aH_R5 has 774 and is kept; aH_R2, "
        "the SAME SHANK IN THE SAME PATIENT, has 773 and is thrown away. One bin in "
        "38,700 decides it, and nothing here measures whether either response repeats.",
    ]
    fig.text(0.055, 0.930, "\n".join(textwrap.fill(t, width=142) for t in body),
             fontsize=8.5, color=MUTED, va="top", linespacing=1.55)

    p = OUT / "G3_gate_borderline_examples.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    tab = pd.DataFrame([{**{k: r[k] for k in
                            ("patient_id", "electrode", "condition",
                             "prop_above_pos", "prop_below_neg", "margin")},
                         "side": s}
                        for s, rr in (("kept", rows_in), ("discarded", rows_out))
                        for r in rr])
    tab.to_csv(OUT / "gate_borderline_examples.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
