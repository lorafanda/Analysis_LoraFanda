#!/usr/bin/env python3
"""make_reliability_examples.py - what the two gates are actually disagreeing about.

make_reliability_gate.py reports the 2x2 and stops at the counts. This draws the
CONTACTS behind each cell, so the disagreement can be read off the recordings instead
of taken on trust:

    amp keep / reliable        both gates agree to keep it
    amp keep / NOT reliable    loud but the two halves do not look alike - the cell
                               that makes the disagreement big
    amp drop / reliable        quiet but reproducible - what an amplitude gate cannot see
    amp drop / NOT reliable    both gates agree to drop it

WHAT IS DRAWN. For every example, the ERSP averaged over the ODD trials and the ERSP
averaged over the EVEN trials, side by side on the project's own +-7 dB scale. The cube
the analysis uses is their average, so these two panels are the whole of the evidence
the reliability gate has: if the two do not look alike, whatever is in the average is
not repeating across trials.

WHICH CONDITION. The one with the highest split-half r, which is the condition the
per-contact r is taken from (make_reliability_gate aggregates with max, matching the
amplitude gate's "responsive in at least one condition"). Each example is annotated with
that condition's own r AND its amplitude margin, so a contact that is loud in a
different condition from the one drawn cannot be mistaken for a quiet one - the
contact's best margin over all conditions is printed too.

AMPLITUDE MARGIN. max(prop_above_pos / 0.02, prop_below_neg / 0.04), so margin >= 1 IS
the amplitude gate. Same definition FIG 0 uses.

HOW THE EXAMPLES ARE PICKED. Within each cell the contacts are sorted by r and sampled
at even quantiles, so they SPAN the cell rather than showing its extremes. Deterministic:
the same call draws the same contacts. Pass --sort margin to span the amplitude range
instead.

    python make_reliability_examples.py
    python make_reliability_examples.py --cell keep_notrel --n 24
    python make_reliability_examples.py --cell keep_notrel --n 24 --patient EL043
    python make_reliability_examples.py --n 6 --band hg
    python make_reliability_examples.py --sort margin

Read-only apart from the PNG and CSV it writes into outputs/clustering/reliability_gate/.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import make_reliability_gate as RG        # noqa: E402
import lf_concat as CC                    # noqa: E402
from lf_features import (downsample_ersp_to_bands,    # noqa: E402
                         FREQ_BANDS_5_TO_400HZ, FREQ_BANDS_15_TO_400HZ)

OUT = RG.OUT
# THE GRID THE CLUSTERING ACTUALLY USES. concat_bands5 is 5 bands x 30 time bins per
# condition; the gate correlates 129 x 300 instead, where nearly every bin is
# independent noise in both halves. Same halves, two resolutions.
FEATURE_GRIDS = {"bands5": FREQ_BANDS_5_TO_400HZ, "b15": FREQ_BANDS_15_TO_400HZ,
                 "hg30": [(70.0, 150.0)]}
ERSP_FMAX = 500.0        # the frequency axis the project's downsampler assumes
VLIM = 7.0                                # the project's ERSP colour scale, dB
FMAX = 400.0
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
# one colour per cell of the 2x2, used for the scatter and for the example titles
CELLS = [
    ("amp keep  ·  reliable", True, True, "#1b7837",
     "both gates agree to keep it"),
    ("amp keep  ·  NOT reliable", True, False, "#c1121f",
     "loud, but the two halves do not look alike"),
    ("amp drop  ·  reliable", False, True, "#4a6fa5",
     "quiet, but it repeats - an amplitude gate cannot see this"),
    ("amp drop  ·  NOT reliable", False, False, "#68727d",
     "both gates agree to drop it"),
]
# --cell takes these short names
CELL_KEY = {"keep_rel": 0, "keep_notrel": 1, "drop_rel": 2, "drop_notrel": 3}


def margins(band_note=""):
    """Per contact and condition: the amplitude margin the gate thresholds at 1.

    Read from the cohort cache lf_concat currently points at, so this figure and the
    gate table describe the same recordings.
    """
    cache = CC.DEFAULT_CONCAT_CACHE
    t = pd.read_parquet(cache / "df_meta.parquet")
    import json
    p = json.loads((cache / "params.json").read_text())
    t["margin"] = np.maximum(t.prop_above_pos / float(p["min_prop_pos"]),
                             t.prop_below_neg / float(p["min_prop_neg"]))
    t["ckey"] = t["electrode"].map(RG.norm_contact)
    return t[["patient_id", "condition", "ckey", "margin"]], cache.name


def half_files(pid: str, cond: str, contact: str):
    """The odd/even pair for one contact-condition, or None."""
    d = RG.ERSP / pid / "LM" / "ERSP_halves" / cond
    hits = sorted(d.glob(f"*_ERSP_{contact}_TN_half1.npy"))
    if not hits:
        return None
    p1 = hits[0]
    p2 = Path(str(p1)[: -len("_half1.npy")] + "_half2.npy")
    return (p1, p2) if p2.exists() else None


def pick(df: pd.DataFrame, n: int, by: str) -> pd.DataFrame:
    """n rows spanning the cell, at even quantiles of `by`. Deterministic."""
    if df.empty:
        return df
    d = df.sort_values([by, "patient_id", "ckey"]).reset_index(drop=True)
    if len(d) <= n:
        return d
    q = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    idx = np.unique((q * (len(d) - 1)).round().astype(int))
    return d.iloc[idx]


def feature_r(A, B, grid: str):
    """Split-half r after both halves are reduced to the analysed feature grid.

    NOT a smoothed version of the same number: averaging within a band and a time
    bin cancels the independent per-bin noise that dominates the full-resolution
    correlation, and leaves the structure the clustering is fitted on. It is the
    reliability OF THE FEATURES rather than of the raw cube.
    """
    edges = FEATURE_GRIDS[grid]
    a = downsample_ersp_to_bands(A, edges, fmax_hz=ERSP_FMAX, time_bins_out=30)
    b = downsample_ersp_to_bands(B, edges, fmax_hz=ERSP_FMAX, time_bins_out=30)
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def draw_half(ax, A, col, label):
    nf = A.shape[0]
    ax.pcolormesh(np.linspace(0, 1, A.shape[1]), np.linspace(0, FMAX, nf), A,
                  cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, shading="auto", rasterized=True)
    ax.axvline(0.5, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks([])
    ax.set_ylim(0, FMAX)
    ax.set_yticks([0, 200, 400])
    ax.tick_params(labelsize=6, colors=MUTED, length=2)
    for s_ in ax.spines.values():
        s_.set_color(col); s_.set_linewidth(1.0)
    ax.text(0.012, 0.94, label, transform=ax.transAxes, fontsize=6.2, color=MUTED,
            ha="left", va="top")


def scatter(ax, m, r_min):
    """Every matched contact: how loud against how reproducible.

    THE PANEL THAT ANSWERS "why is the disagreement so big". If loudness predicted
    reproducibility the cloud would run along the diagonal and the two gates would keep
    the same contacts; the disagreement is the mass in the off-diagonal quadrants.
    """
    for lab, amp, rel, col, _ in CELLS:
        s = m[(m.amp_gate == amp) & (m.rel_gate == rel)]
        ax.scatter(s.margin_best.clip(0.02, 40), s.r, s=5, c=col, alpha=0.55, lw=0,
                   label=f"{lab}   n={len(s)}")
    ax.axvline(1.0, color=INK, lw=1.0, ls="--")
    ax.axhline(r_min, color=INK, lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("amplitude margin  (>= 1 passes the amplitude gate)", fontsize=8.4)
    ax.set_ylabel(f"split-half r  (> {r_min} passes the reliability gate)", fontsize=8.4)
    ax.tick_params(labelsize=7.4, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.0, frameon=False, loc="lower right", markerscale=2.0)
    rho = m[["margin_best", "r"]].corr(method="spearman").iloc[0, 1]
    ax.set_title("A  ·  loudness against reproducibility, every matched contact"
                 f"   ·   Spearman rho = {rho:.2f}",
                 fontsize=10.0, loc="left", color=INK, pad=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="examples per cell")
    ap.add_argument("--band", choices=["all", "hg"], default="all")
    ap.add_argument("--r-min", type=float, default=0.2, dest="r_min")
    ap.add_argument("--sort", choices=["r", "margin"], default="r",
                    help="which axis the examples span within a cell")
    ap.add_argument("--cell", choices=["all"] + list(CELL_KEY), default="all",
                    help="draw one cell of the 2x2 as a grid, with as many "
                         "examples as --n asks for")
    ap.add_argument("--patient", default=None,
                    help="restrict the examples to one patient")
    ap.add_argument("--cols", type=int, default=6, help="grid width in --cell mode")
    ap.add_argument("--grid", choices=list(FEATURE_GRIDS), default="bands5",
                    help="feature grid the second r is computed on")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else OUT
    pc_f = OUT / "reliability_per_contact.csv"
    cond_f = OUT / "reliability_per_contact_condition.csv"
    if not pc_f.exists() or not cond_f.exists():
        raise SystemExit(f"run make_reliability_gate.py first - {pc_f.name} is missing")

    per_cond = pd.read_csv(cond_f)
    per_cond = per_cond[per_cond.band == a.band].copy()
    per_cond["ckey"] = per_cond["contact"].map(RG.norm_contact)
    # the contact is judged on its BEST condition, so that is the one drawn
    best = (per_cond.sort_values("split_half_r")
                    .drop_duplicates(["patient_id", "ckey"], keep="last"))

    pc = pd.read_csv(pc_f)
    pc = pc[pc.amp_gate.notna()].copy()
    pc["amp_gate"] = pc.amp_gate.astype(str).str.lower().isin(["true", "1", "1.0"])
    if a.band != "all":                      # rebuild the gate on the requested band
        pc = pc.drop(columns=["r", "rel_gate"]).merge(
            best[["patient_id", "ckey", "split_half_r"]].rename(
                columns={"split_half_r": "r"}), on=["patient_id", "ckey"], how="left")
        pc["rel_gate"] = pc.r > a.r_min
    mg, cache_name = margins()
    m = pc.merge(best[["patient_id", "ckey", "condition", "split_half_r"]],
                 on=["patient_id", "ckey"], how="left")
    m = m.merge(mg, on=["patient_id", "condition", "ckey"], how="left")
    # the contact's loudest condition, which may not be the one drawn
    bestm = (mg.sort_values("margin").drop_duplicates(["patient_id", "ckey"], keep="last")
               .rename(columns={"margin": "margin_best", "condition": "cond_loudest"}))
    m = m.merge(bestm, on=["patient_id", "ckey"], how="left")
    # the same contact over HG alone, and the patient it belongs to: 30% of the
    # variance in r is between patients, so a contact's r means little without its
    # patient's own level beside it
    wide = (per_cond.pivot_table(index=["patient_id", "ckey", "condition"],
                                 columns="band", values="split_half_r")
            .reset_index())
    if "hg" in wide.columns:
        m = m.merge(wide[["patient_id", "ckey", "condition", "hg"]],
                    on=["patient_id", "ckey", "condition"], how="left")
    else:
        m["hg"] = np.nan
    m["pat_med_r"] = m.groupby("patient_id").r.transform("median")
    m = m[m.margin.notna() & m.r.notna()]
    print(f"  {len(m)} contacts with both a reliability estimate and an amplitude margin")
    print(f"  cohort cache: {cache_name}   band: {a.band}   r > {a.r_min}")

    rows, chosen, feat_rows = [], {}, []
    cells = CELLS if a.cell == "all" else [CELLS[CELL_KEY[a.cell]]]
    pool = m if not a.patient else m[m.patient_id == a.patient]
    if a.patient and pool.empty:
        raise SystemExit(f"no contacts for {a.patient}")
    for lab, amp, rel, col, _ in cells:
        cell = pool[(pool.amp_gate == amp) & (pool.rel_gate == rel)]
        sel = pick(cell, a.n, "split_half_r" if a.sort == "r" else "margin")
        chosen[lab] = sel
        if len(cell):
            print(f"  {lab:<28} n={len(cell):>5}   median r {cell.r.median():.3f}   "
                  f"median margin {cell.margin_best.median():.2f}   "
                  f"drawing {len(sel)}")
        else:
            print(f"  {lab:<28} n=0")
        for t in sel.itertuples():
            rows.append(dict(cell=lab, patient=t.patient_id, contact=t.contact,
                             condition_drawn=t.condition, r=round(float(t.r), 4),
                             r_hg=None if pd.isna(t.hg) else round(float(t.hg), 4),
                             patient_median_r=round(float(t.pat_med_r), 4),
                             margin_drawn=round(float(t.margin), 3),
                             loudest_condition=t.cond_loudest,
                             margin_best=round(float(t.margin_best), 3)))

    def example(ax1, ax2, t, col):
        """One contact: odd trials over even trials, with the three numbers."""
        f = half_files(t.patient_id, t.condition, t.contact)
        if f is None:
            for ax in (ax1, ax2):
                ax.axis("off")
            ax1.text(0.5, 0.5, "half cubes not found", ha="center", va="center",
                     fontsize=7, color=MUTED, transform=ax1.transAxes)
            return
        A, B = np.load(f[0]), np.load(f[1])
        draw_half(ax1, A, col, "odd trials")
        draw_half(ax2, B, col, "even trials")
        rf = feature_r(A, B, a.grid)
        feat_rows.append(dict(patient=t.patient_id, contact=t.contact,
                              condition=t.condition, r_full=float(t.r),
                              **{f"r_{a.grid}": rf}))
        ax1.set_title(f"{t.patient_id} · {t.contact} · {t.condition}\n"
                      f"r = {t.r:.2f}   {a.grid} r = {rf:.2f}   "
                      f"pat med {t.pat_med_r:.2f}   margin {t.margin_best:.2f}",
                      fontsize=6.8, color=col, pad=3, loc="left")

    if a.cell == "all":
        fig_h = 5.0 + 3.05 * len(cells)
        fig = plt.figure(figsize=(2.55 * a.n + 2.2, fig_h), dpi=190)
        gs = GridSpec(1 + len(cells), 1, figure=fig,
                      height_ratios=[1.55] + [1.0] * len(cells),
                      hspace=0.42, left=0.062, right=0.985,
                      top=1 - 0.95 / fig_h, bottom=0.028)
        head = (f"R2   ·   what the amplitude gate and the reliability gate disagree "
                f"about   ·   {len(m)} contacts")
        scatter(fig.add_subplot(gs[0]), m, a.r_min)
        for ci, (lab, amp, rel, col, why) in enumerate(cells):
            sel = chosen[lab]
            cell_n = int(((m.amp_gate == amp) & (m.rel_gate == rel)).sum())
            inner = GridSpecFromSubplotSpec(2, max(len(sel), 1), gs[1 + ci],
                                            hspace=0.16, wspace=0.14)
            for k, t in enumerate(sel.itertuples()):
                a1, a2 = fig.add_subplot(inner[0, k]), fig.add_subplot(inner[1, k])
                example(a1, a2, t, col)
                if k == 0:
                    a1.set_ylabel("Hz", fontsize=7); a2.set_ylabel("Hz", fontsize=7)
            bb = fig.add_subplot(gs[1 + ci], frame_on=False)
            bb.set_xticks([]); bb.set_yticks([]); bb.patch.set_visible(False)
            bb.set_title(f"{chr(66 + ci)}  ·  {lab}   ·   n = {cell_n}   ·   {why}",
                         fontsize=10.0, loc="left", color=col, pad=26)
    else:
        # ONE CELL, as many examples as asked for. The claim being tested is about a
        # population of a thousand contacts; five of them cannot settle it.
        lab, amp, rel, col, why = cells[0]
        sel = chosen[lab]
        cell_n = int(((pool.amp_gate == amp) & (pool.rel_gate == rel)).sum())
        cols = max(1, min(a.cols, len(sel)))
        nrow = int(np.ceil(len(sel) / cols))
        fig_h = 1.9 + 2.35 * nrow
        fig = plt.figure(figsize=(2.75 * cols + 0.9, fig_h), dpi=190)
        gs = GridSpec(nrow * 2, cols, figure=fig, hspace=0.30, wspace=0.14,
                      left=0.052, right=0.988, top=1 - 1.35 / fig_h, bottom=0.022)
        head = (f"R2 · {lab} · {cell_n} contacts"
                + (f" in {a.patient}" if a.patient else "")
                + f", {len(sel)} drawn, spanning the cell by {a.sort}")
        for k, t in enumerate(sel.itertuples()):
            r_, c_ = divmod(k, cols)
            a1 = fig.add_subplot(gs[2 * r_, c_])
            a2 = fig.add_subplot(gs[2 * r_ + 1, c_])
            example(a1, a2, t, col)
            if c_ == 0:
                a1.set_ylabel("Hz", fontsize=7); a2.set_ylabel("Hz", fontsize=7)
        cell = pool[(pool.amp_gate == amp) & (pool.rel_gate == rel)]
        fig.text(0.052, 1 - 0.72 / fig_h,
                 f"{why}.   median r {cell.r.median():.3f}"
                 + (f", median hg r {cell.hg.median():.3f}" if cell.hg.notna().any()
                    else "")
                 + f", median amplitude margin {cell.margin_best.median():.2f}."
                 f"   Each pair is ODD trials over EVEN trials, +-7 dB.   "
                 "'r' is the gate's own number, over all 129 x 300 bins; "
                 f"'{a.grid} r' is the same halves reduced to the grid the "
                 "clustering is fitted on. 'pat med' is that patient's median r.",
                 fontsize=8.8, color=MUTED, va="top")
    fig.suptitle(head, x=0.052 if a.cell != "all" else 0.062,
                 y=1 - 0.30 / fig_h, ha="left", fontsize=14.0, color=INK)
    if a.cell == "all":
        fig.text(0.062, 1 - 0.63 / fig_h,
                 "Each example is one contact: the ERSP of its ODD trials beside the "
                 "ERSP of its EVEN trials, on the project's +-7 dB scale. The analysed "
                 "cube is their average, so these two panels are all the evidence there "
                 "is that a response repeats.",
                 fontsize=9.2, color=MUTED, va="top")

    out.mkdir(parents=True, exist_ok=True)
    tag = a.band + ("" if a.cell == "all" else f"_{a.cell}") \
        + (f"_{a.patient}" if a.patient else "")
    png = out / f"R2_reliability_examples_{tag}.png"
    fig.savefig(png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    tab = pd.DataFrame(rows)
    if feat_rows:
        ft = pd.DataFrame(feat_rows)
        tab = tab.merge(ft.drop(columns=["r_full"]),
                        left_on=["patient", "contact", "condition_drawn"],
                        right_on=["patient", "contact", "condition"],
                        how="left").drop(columns=["condition"])
        col_ = f"r_{a.grid}"
        print(f"\n  {col_} over the drawn examples: median "
              f"{ft[col_].median():.3f}  (the gate's full-resolution r: "
              f"{ft.r_full.median():.3f})")
    csv = out / f"R2_reliability_examples_{tag}.csv"
    tab.to_csv(csv, index=False)
    print(f"\n  -> {png}")
    print(f"  -> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
