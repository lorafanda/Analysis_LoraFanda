#!/usr/bin/env python3
"""
make_graphical_abstract.py - the Cell Reports graphical abstract, 1200 x 1200 px.

    python make_graphical_abstract.py

Built from the SAME run, centroids, hemisphere renders and palette as FIG 1a, imported
from 00_Paper2_Figures.py, so the abstract cannot drift from the figure it condenses.
Every number on it is read from the CSVs and captions the three figures wrote; nothing
is typed in. The caption beside the PNG says where each one came from.

WHAT IT SAYS, top to bottom. The task (three inputs, one output). Four of the eight
response types, each as its mean time course with +/-1 SD and the electrodes that carry
it, seen from each hemisphere's own side. Then the three tests the paper puts them to -
how many, do they survive other methods, are they anatomy - each answered with one
number. Then the sentence the paper is about.

The four type NAMES are descriptive labels read off the centroid shapes (a rise after
the GO cue in every condition is "production"; a sharp onset at the start of the audio
block is "auditory onset"; a dip below baseline during the stimulus is "suppression"; a
response in the reading block only is "reading only"). They are not computed and the
caption says so. The cluster ids they attach to are checked against the FIG 1a patient
table, so a re-fit that renumbered the clusters fails here rather than mislabelling.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)
OUT = P2.OUT
INK, MUTED, GREY, RED, GREEN = P2.INK, P2.MUTED, P2.GREY, P2.RED, P2.GREEN

FSET, K = "concat_hg", 8
PX = 1200                                  # Cell Reports: 1200 x 1200, minimal text
# cluster id -> the label its shape earns. Checked against FIG 1a below.
SHOW = [(1, "Production"), (3, "Auditory onset"), (4, "Suppression"), (5, "Reading only")]
TILE_BG, TILE_EC = "#f6f7f9", "#dde2e8"


def need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"missing {p.name} - run the figure that writes it first")
    return p


def numbers():
    """Every number on the abstract, from the files the figures wrote."""
    n = {}
    gen = pd.read_csv(need(OUT / f"FIG1a_{FSET}_cnmf_K{K}_generalization.csv"))
    n["gen_k"], n["gen_frac"] = gen.k.to_numpy(), 100 * gen.frac_electrodes.to_numpy()
    pk = pd.read_csv(need(P2.PEAKS))
    n["k_peak"] = int(pk[pk.feature_set == FSET].k_peak.iloc[0])
    first_bad = gen[gen.n_dominated > 0].k.min()
    n["k_first_onepatient"] = int(first_bad) if pd.notna(first_bad) else None

    ea = pd.read_csv(need(OUT / f"FIG2_agreement_K{K}_electrode_agreement.csv"))
    N = int(ea.n_feature_sets.iloc[0])
    n["n_fsets"] = N
    n["pct_all_agree"] = 100 * float((ea.feature_set_agreement == N).mean())
    cap = need(OUT / f"FIG2_agreement_K{K}_caption.txt").read_text(encoding="utf-8")
    m = re.search(r"feature sets\s+all \d agree:\s*([\d.]+)%\s*\(chance ([\d.]+)%\)", cap)
    n["pct_chance"] = float(m.group(2)) if m else None

    la = pd.read_csv(need(OUT / f"FIG3_lana_{FSET}_K{K}_clusters.csv"))
    top = la.loc[la.mean_P.idxmax()]
    n["lana_top_cluster"] = int(top.cluster)
    n["lana_top_mean"] = float(top.mean_P)
    n["lana_top_null_sh"] = float(top.null_mean_P_sh)
    n["lana_top_q_sh"] = float(top.q_mean_sh)
    n["lana_baseline"] = float(la.baseline_P.iloc[0])
    n["lana_n_sig_sh"] = int((la.q_mean_sh < 0.05).sum())
    n["lana_n_clusters"] = int(len(la))
    n["lana_max_abs_rho"] = float(la.rho.abs().max())
    return n


def tile(fig, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=TILE_BG, ec=TILE_EC, lw=1.2, transform=ax.transAxes,
                                clip_on=False))
    return ax


def main() -> int:
    t0 = time.time()
    n = numbers()

    d = P2.load_run(FSET, K)
    d["fset"] = FSET
    C = P2.cube(d["X"], d)
    means = np.stack([C[d["lab"] == j].mean(0) for j in range(K)])
    sds = np.stack([C[d["lab"] == j].std(0, ddof=1) for j in range(K)])
    vlim = float(np.percentile(np.abs(means), 99.0))
    ylim = (float((means - sds).min()) * 1.06, float((means + sds).max()) * 1.06)

    # the labels in SHOW belong to THIS numbering: refuse to draw if the run has moved
    pc = pd.read_csv(need(OUT / f"FIG1a_{FSET}_cnmf_K{K}_patients.csv"))
    sizes = np.bincount(d["lab"], minlength=K)
    want = pc.set_index("cluster").n.reindex(range(K)).to_numpy()
    if not np.array_equal(sizes, want):
        raise SystemExit(f"cluster sizes {sizes.tolist()} do not match FIG 1a's "
                         f"{want.tolist()} - the run changed; re-check SHOW")

    fig = plt.figure(figsize=(PX / 100, PX / 100), dpi=100)
    fig.patch.set_facecolor("white")

    # ---- title ----------------------------------------------------------------
    fig.text(0.5, 0.952, "Eight response types in human language cortex",
             ha="center", va="center", fontsize=24, weight="bold", color=INK)
    fig.text(0.5, 0.916, f"sEEG  ·  {d['n_patients']} patients  ·  {len(d['X'])} "
             f"electrodes  ·  one task, three inputs  ·  convex NMF, K = {K}",
             ha="center", va="center", fontsize=11.5, color=MUTED)

    # ---- the task, exactly as FIG 1 panel C draws it ----------------------------
    P2.trial_strip(fig.add_axes([0.17, 0.828, 0.66, 0.062]), d)

    # ---- four of the eight types -------------------------------------------------
    fig.text(0.05, 0.792, f"Four of the {K} types", fontsize=13, weight="bold",
             color=INK, va="bottom")
    fig.text(0.95, 0.792, "mean ± 1 SD across electrodes   ·   dashed = GO cue   ·   "
             "each hemisphere from its own side", fontsize=9, color=MUTED, ha="right",
             va="bottom")
    gap, x0, x1 = 0.024, 0.05, 0.95
    w = (x1 - x0 - 3 * gap) / 4
    for i, (j, name) in enumerate(SHOW):
        bx = x0 + i * (w + gap)
        col = P2.cluster_col(j, K)
        fig.text(bx, 0.770, name, fontsize=13.5, weight="bold", color=col, va="bottom")
        fig.text(bx + w, 0.770, f"c{j}", fontsize=9, color=MUTED, ha="right",
                 va="bottom")
        axc = fig.add_axes([bx, 0.640, w, 0.098])
        P2.draw_mean(axc, means[j], d, vlim=vlim, col=col, ylim=ylim, sd=sds[j])
        # the three stimulus icons over their own blocks, as in panel C
        for b, cond in enumerate(d["conds"]):
            a = P2._icon_axes(axc, b / 3, (b + 1) / 3, y0=1.04, h=0.26, pad=0.40)
            P2.STIM_ICON[cond](a, col=MUTED)
        for s_, side in enumerate(("L", "R")):
            axb = fig.add_axes([bx + s_ * w / 2, 0.500, w / 2, 0.135])
            img, nsel = P2.render_hemi(side, d, j)
            axb.imshow(img)
            axb.axis("off")
            axb.text(0.04, 0.02, side, transform=axb.transAxes, fontsize=8,
                     color=MUTED, ha="left", va="bottom")
        npat = int(pd.Series(d["patient"][d["lab"] == j]).nunique())
        fig.text(bx + w / 2, 0.492, f"n = {sizes[j]}   ·   {npat} patients",
                 fontsize=9.2, color=MUTED, ha="center", va="top")

    # ---- three tests -------------------------------------------------------------
    fig.text(0.05, 0.446, "Three tests", fontsize=13, weight="bold", color=INK,
             va="bottom")
    tw, tg, ty, th = (0.90 - 2 * 0.03) / 3, 0.03, 0.205, 0.228

    # (a) how many
    t = tile(fig, 0.05, ty, tw, th)
    t.text(0.07, 0.90, "How many?", fontsize=11, color=MUTED, va="top")
    t.text(0.07, 0.76, f"K = {K}", fontsize=27, weight="bold", color=INK, va="top")
    kp, kb = n["k_peak"], n["k_first_onepatient"]
    t.text(0.07, 0.50, f"Held-out fit peaks at K = {kp}.\nFrom K = {kb} the added "
           f"clusters\nare single patients.", fontsize=9.6, color=INK, va="top",
           linespacing=1.35)
    m = t.inset_axes([0.24, 0.09, 0.68, 0.22])
    m.plot(n["gen_k"], n["gen_frac"], color=RED, lw=1.6)
    m.axvline(K, color=INK, lw=1.0, ls=(0, (3, 2)))
    m.axvline(kp, color=GREY, lw=1.0)
    m.set_xlim(n["gen_k"].min(), n["gen_k"].max())
    m.set_ylim(0, max(70, n["gen_frac"].max() * 1.05))
    m.set_yticks([0, 30, 60]); m.set_xticks([5, K, kp, 20, 30])
    m.tick_params(labelsize=6.5, colors=MUTED, length=2, pad=1.5)
    m.set_ylabel("% in one-\npatient\nclusters", fontsize=6.2, color=MUTED,
                 labelpad=2, rotation=0, ha="right", va="center")
    for s_ in m.spines.values():
        s_.set_color(GREY)
    m.patch.set_alpha(0)

    # (b) other methods
    t = tile(fig, 0.05 + tw + tg, ty, tw, th)
    t.text(0.07, 0.90, "Other features and algorithms?", fontsize=11, color=MUTED,
           va="top")
    t.text(0.07, 0.72, f"{n['pct_all_agree']:.0f}%", fontsize=27, weight="bold",
           color=GREEN, va="top")
    ch = f" (chance {n['pct_chance']:g}%)" if n["pct_chance"] is not None else ""
    t.text(0.07, 0.47, f"of electrodes are put in the same type\nby all "
           f"{n['n_fsets']} feature sets{ch}.", fontsize=9.6, color=INK, va="top",
           linespacing=1.35)
    c3 = P2.cluster_col(3, K)
    for q in range(4):
        t.add_patch(Rectangle((0.07 + q * 0.062, 0.155), 0.048, 0.085, fc=c3,
                              ec="none"))
    t.text(0.07 + 4 * 0.062 + 0.015, 0.195, "the auditory-onset type is\nfound by "
           "every feature set\nand every algorithm", fontsize=8.6, color=INK,
           va="center", linespacing=1.3)

    # (c) anatomy
    t = tile(fig, 0.05 + 2 * (tw + tg), ty, tw, th)
    t.text(0.07, 0.90, "Placed by the language atlas?", fontsize=11, color=MUTED,
           va="top")
    t.text(0.07, 0.76, "No", fontsize=27, weight="bold", color=RED, va="top")
    t.text(0.07, 0.50, f"{n['lana_n_sig_sh']} of {n['lana_n_clusters']} types sit "
           f"above a null that\nkeeps each electrode shaft intact.\nLargest |ρ| = "
           f"{n['lana_max_abs_rho']:.2f}.", fontsize=9.6, color=INK, va="top",
           linespacing=1.35)
    b = t.inset_axes([0.30, 0.09, 0.60, 0.20])
    jt = n["lana_top_cluster"]
    vals = [n["lana_top_mean"], n["lana_top_null_sh"], n["lana_baseline"]]
    b.barh([2, 1, 0], vals, color=[P2.cluster_col(jt, K), GREY, "#e9edf1"],
           height=0.72)
    for yy, v in zip([2, 1, 0], vals):
        b.text(v + 0.004, yy, f"{v:.3f}", fontsize=6.6, color=MUTED, va="center")
    b.set_yticks([2, 1, 0])
    b.set_yticklabels([f"c{jt} (highest)", "shaft null", "cohort"], fontsize=6.5,
                      color=MUTED)
    b.set_xlim(0, max(vals) * 1.35); b.set_xticks([])
    for s_ in b.spines.values():
        s_.set_visible(False)
    b.tick_params(length=0, pad=2)
    b.patch.set_alpha(0)

    # ---- the sentence ------------------------------------------------------------
    fig.text(0.5, 0.135, "A response type is what an electrode does — not where it sits.",
             ha="center", va="center", fontsize=18, weight="bold", color=INK)
    fig.text(0.5, 0.082, "high-gamma (70–150 Hz) time courses, time-warped so the GO "
             "cue falls at 50% of every block  ·  nulls: shaft-shift and within-patient "
             "permutation  ·  FDR across types",
             ha="center", va="center", fontsize=8.6, color=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"GA_graphical_abstract_{FSET}_K{K}.png"
    P2.save_png(fig, p, dpi=100, facecolor="white")      # no bbox_inches: stay 1200 px
    plt.close(fig)

    from PIL import Image
    with Image.open(p) as im:
        if im.size != (PX, PX):
            raise SystemExit(f"{p.name} is {im.size}, not {(PX, PX)}")

    names = ", ".join(f"c{j} = {nm}" for j, nm in SHOW)
    P2.save_text("\n".join([
        f"GRAPHICAL ABSTRACT   ·   {PX} x {PX} px   ·   {FSET}, convex NMF, K = {K}",
        "",
        "Built from the same run, centroids, renders and palette as FIG 1a "
        "(00_Paper2_Figures.py imported directly). Nothing on it is typed in.",
        "",
        "TYPE LABELS are descriptive, read off the centroid shapes, not computed:",
        f"  {names}",
        "  Cluster ids were checked against FIG1a_..._patients.csv (sizes match).",
        "",
        "NUMBERS and where each comes from:",
        f"  K = {K}; held-out peak K = {n['k_peak']}      heldout_peaks_cnmf.csv",
        f"  first K with a one-patient cluster = {n['k_first_onepatient']}   "
        f"FIG1a_..._generalization.csv (also the red curve)",
        f"  {n['pct_all_agree']:.1f}% of electrodes placed alike by all "
        f"{n['n_fsets']} feature sets   FIG2_..._electrode_agreement.csv",
        f"  chance {n['pct_chance']}%                              "
        f"FIG2_..._caption.txt",
        f"  c{n['lana_top_cluster']} mean P(LanA) {n['lana_top_mean']:.4f}, shaft null "
        f"{n['lana_top_null_sh']:.4f}, q = {n['lana_top_q_sh']:.3f}, cohort "
        f"{n['lana_baseline']:.4f}   FIG3_..._clusters.csv",
        f"  {n['lana_n_sig_sh']} of {n['lana_n_clusters']} above the shaft null at "
        f"q < 0.05; largest |rho| = {n['lana_max_abs_rho']:.3f}   FIG3_..._clusters.csv",
        "",
        "'The auditory-onset type is found by every feature set and algorithm' is "
        "FIG 2B/E read by eye (c3 is the one row with no low cell); re-check if FIG 2 "
        "is re-run.",
    ]), p.with_name(p.stem + "_caption.txt"))
    print(f"  {time.time() - t0:.0f}s  -> {p.name}  ({PX}x{PX})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
