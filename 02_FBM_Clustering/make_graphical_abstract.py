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

The type NAMES are descriptive labels read off the centroid shapes (a rise after the
GO cue in every condition is "production"; a sharp onset at the start of the audio
block is "auditory onset"; a dip below baseline during the stimulus is "suppression"; a
response in the reading block only is "reading only"), and the caption says so. One is
MEASURED rather than read: "ramp to cue" is the cluster whose mean rises most over the
stretch before the GO cue, averaged over conditions, and the script refuses to run if
the cluster drawn under that name is not the one the measurement picks. The eye picked
the wrong cluster for it the first time, which is why. The cluster ids in SHOW are also
checked against the FIG 1a patient table, so a re-fit that renumbered the clusters
fails here rather than mislabelling.

The ramp cluster is the candidate premotor type. It is named by what it does because the
coordinate table has no gyral labels and because naming a type by its anatomy would
contradict the sentence at the bottom of the abstract.
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

FSET, K = "concat_bands15", 7           # the driver's MAIN / K_MAIN (run_paper_figures_v17.py)
PX = 1200                                  # Cell Reports: 1200 x 1200, minimal text
# cluster id -> the label its shape earns. Checked against FIG 1a below.
# PROVISIONAL for the v17 15-band K = 7 fit (2026-10-21): the four labels were read off
# FIG 1a's blocks by Claude and are not yet Lora's - rename or re-pick before the
# abstract is used. c5 = 70-400 Hz burst after the GO cue in every condition; c0 =
# 13-50 Hz decrease through stimulus and response; c2 = broadband high-frequency
# decrease during the stimulus halves; c4 = 1-13 Hz increase, strongest after the cue.
SHOW = [(5, "Production"), (0, "Beta decrease"), (2, "Broadband decrease"),
        (4, "Low-frequency rise")]
# A SHOW entry carrying this label must be the cluster the pre-cue ramp measurement in
# main() picks, or the script refuses. On the 15-band K = 7 fit no cluster ramps before
# the cue (every HFA pre-cue slope is negative), so no entry carries it.
RAMP_LABEL = "Ramp to cue"
TILE_BG, TILE_EC = "#f6f7f9", "#dde2e8"


def need(p: Path) -> Path:
    """The file, or the newest run-tagged version of it.

    FIG 1's sidecars carry the run id since 2026-09-07, so the name this script asks for
    is a prefix rather than a filename.
    """
    if not p.exists():
        stem, suf = p.stem, p.suffix
        # FIG1a_<fset>_cnmf_K8_generalization.csv -> FIG1a_<fset>_cnmf_K8_run<id>_...
        base, _, tail = stem.partition("_K")
        k, _, rest = tail.partition("_")
        cands = sorted(p.parent.glob(f"{base}_K{k}_run*_{rest}{suf}")) if rest else []
        if cands:
            return cands[-1]
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

    ea = pd.read_csv(need(OUT / f"FIG2_agreement_{FSET}_K{K}_electrode_agreement.csv"))
    N = int(ea.n_feature_sets.iloc[0])
    n["n_fsets"] = N
    n["pct_all_agree"] = 100 * float((ea.feature_set_agreement == N).mean())
    cap = need(OUT / f"FIG2_agreement_{FSET}_K{K}_caption.txt").read_text(encoding="utf-8")
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
    # the share inside the network (FIG 3 B, 2026-10-26): which clusters clear the
    # shaft-shift null on it, so the tile cannot say "no" where FIG 3 says "one"
    if "q_frac_t10_sh" in la.columns:
        sh = la[la.q_frac_t10_sh < 0.05]
        n["lana_share_sig"] = [int(c) for c in sh.cluster]
    else:
        n["lana_share_sig"] = []
    n["lana_n_clusters"] = int(len(la))
    n["lana_max_abs_rho"] = float(la.rho.abs().max())
    # the type every algorithm finds: the reference cluster of FIG 2's algorithm half with
    # the highest minimum Jaccard against k-means and Ward - read from the table, not by eye
    ca = pd.read_csv(need(OUT / f"FIG2_agreement_{FSET}_K{K}_cluster_agreement.csv"))
    alg = ca[(ca.half == "algorithm") & (ca.solution != "convex NMF")]
    jmin = alg.groupby("reference_cluster").jaccard.min()
    n["robust_cluster"], n["robust_minJ"] = int(jmin.idxmax()), float(jmin.max())
    return n


def tile(fig, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=TILE_BG, ec=TILE_EC, lw=P2.pw(1.2), transform=ax.transAxes,
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

    # THE PRE-CUE RAMP, MEASURED: the mean over the last fifth of the block before the
    # GO cue minus the mean over its first fifth, averaged over the three conditions.
    # The cluster labelled "Ramp to cue" must be the one this picks, or the label is
    # an eye's opinion - and the eye picked the wrong cluster the first time (c7's rise
    # is AT the cue and its pre-cue slope is negative; c6 rises before it in all three).
    nt = d["nt"]
    g0, g1 = int(0.20 * nt), int(0.50 * nt)
    # on the HFA rows (>= 70 Hz) whatever the feature set: band 0 of a band set is 1-4 Hz
    hfa = [i for i, b in enumerate(d["bands"]) if P2.band_centre(b) >= 70] or [0]
    ramp_c = np.stack([means[j][:, hfa, int(0.30 * nt):g1].mean(axis=(1, 2))
                       - means[j][:, hfa, :g0].mean(axis=(1, 2)) for j in range(K)])
    ramp = ramp_c.mean(1)
    RAMP_ID = int(ramp.argmax()) if float(ramp.max()) > 0 else None
    ramp_ids = [j for j, nm in SHOW if nm == RAMP_LABEL]
    if ramp_ids and ramp_ids[0] != RAMP_ID:
        best = (f"c{int(ramp.argmax())} ({ramp.max():+.2f} dB)" if RAMP_ID is not None
                else "none - every HFA pre-cue slope is negative")
        raise SystemExit(f"largest pre-cue ramp is {best}, not c{ramp_ids[0]} - "
                         "relabel SHOW before drawing")

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
    fig.text(0.5, 0.952, f"{K} response types in human language cortex",
             ha="center", va="center", fontsize=P2.fs(24), weight="bold", color=INK)
    fig.text(0.5, 0.916, f"sEEG  ·  {d['n_patients']} patients  ·  {len(d['X'])} "
             f"electrodes  ·  one task, three inputs  ·  convex NMF, K = {K}",
             ha="center", va="center", fontsize=P2.fs(11.5), color=MUTED)

    # ---- the task, exactly as FIG 1 panel C draws it ----------------------------
    P2.trial_strip(fig.add_axes([0.17, 0.828, 0.66, 0.062]), d)

    # ---- four of the eight types -------------------------------------------------
    fig.text(0.05, 0.792, f"{len(SHOW)} of the {K} types", fontsize=P2.fs(13), weight="bold",
             color=INK, va="bottom")
    fig.text(0.95, 0.792, "mean ± 1 SD across electrodes   ·   dashed = GO cue   ·   "
             "each hemisphere from its own side", fontsize=P2.fs(9), color=MUTED, ha="right",
             va="bottom")
    gap, x0, x1 = 0.016, 0.05, 0.95
    w = (x1 - x0 - (len(SHOW) - 1) * gap) / len(SHOW)
    for i, (j, name) in enumerate(SHOW):
        bx = x0 + i * (w + gap)
        col = P2.cluster_col(j, K)
        fig.text(bx, 0.770, name, fontsize=P2.fs(12.2), weight="bold", color=col, va="bottom")
        fig.text(bx + w, 0.770, f"c{j}", fontsize=P2.fs(9), color=MUTED, ha="right",
                 va="bottom")
        axc = fig.add_axes([bx, 0.640, w, 0.098])
        P2.draw_mean(axc, means[j], d, vlim=vlim, col=col, ylim=ylim, sd=sds[j])
        # the three stimulus icons over their own blocks, as in panel C
        for b, cond in enumerate(d["conds"]):
            a = P2._icon_axes(axc, b / 3, (b + 1) / 3, y0=1.04, h=0.26, pad=P2.pw(0.40))
            P2.STIM_ICON[cond](a, col=MUTED)
        for s_, side in enumerate(("L", "R")):
            axb = fig.add_axes([bx + s_ * w / 2, 0.500, w / 2, 0.135])
            img, nsel = P2.render_hemi(side, d, j)
            axb.imshow(img)
            axb.axis("off")
            axb.text(0.04, 0.02, side, transform=axb.transAxes, fontsize=P2.fs(8),
                     color=MUTED, ha="left", va="bottom")
        npat = int(pd.Series(d["patient"][d["lab"] == j]).nunique())
        fig.text(bx + w / 2, 0.492, f"n = {sizes[j]}   ·   {npat} patients",
                 fontsize=P2.fs(9.2), color=MUTED, ha="center", va="top")

    # ---- three tests -------------------------------------------------------------
    fig.text(0.05, 0.446, "Three tests", fontsize=P2.fs(13), weight="bold", color=INK,
             va="bottom")
    tw, tg, ty, th = (0.90 - 2 * 0.03) / 3, 0.03, 0.205, 0.228

    # (a) how many
    t = tile(fig, 0.05, ty, tw, th)
    t.text(0.07, 0.90, "How many?", fontsize=P2.fs(11), color=MUTED, va="top")
    t.text(0.07, 0.76, f"K = {K}", fontsize=P2.fs(27), weight="bold", color=INK, va="top")
    kp, kb = n["k_peak"], n["k_first_onepatient"]
    t.text(0.07, 0.50, f"Held-out fit peaks at K = {kp}.\nFrom K = {kb} the added "
           f"clusters\nare single patients.", fontsize=P2.fs(9.6), color=INK, va="top",
           linespacing=1.35)
    m = t.inset_axes([0.24, 0.09, 0.68, 0.22])
    m.plot(n["gen_k"], n["gen_frac"], color=RED, lw=P2.pw(1.6))
    m.axvline(K, color=INK, lw=P2.pw(1.0), ls=(0, (3, 2)))
    m.axvline(kp, color=GREY, lw=P2.pw(1.0))
    m.set_xlim(n["gen_k"].min(), n["gen_k"].max())
    m.set_ylim(0, max(70, n["gen_frac"].max() * 1.05))
    m.set_yticks([0, 30, 60]); m.set_xticks([5, K, kp, 20, 30])
    m.tick_params(labelsize=P2.fs(6.5), colors=MUTED, length=2, pad=P2.pw(1.5))
    m.set_ylabel("% in one-\npatient\nclusters", fontsize=P2.fs(6.2), color=MUTED,
                 labelpad=P2.pw(2), rotation=0, ha="right", va="center")
    for s_ in m.spines.values():
        s_.set_color(GREY)
    m.patch.set_alpha(0)

    # (b) other methods
    t = tile(fig, 0.05 + tw + tg, ty, tw, th)
    t.text(0.07, 0.90, "Other features and algorithms?", fontsize=P2.fs(11), color=MUTED,
           va="top")
    t.text(0.07, 0.72, f"{n['pct_all_agree']:.0f}%", fontsize=P2.fs(27), weight="bold",
           color=GREEN, va="top")
    ch = f" (chance {n['pct_chance']:g}%)" if n["pct_chance"] is not None else ""
    t.text(0.07, 0.47, f"of electrodes are put in the same type\nby all "
           f"{n['n_fsets']} feature sets{ch}.", fontsize=P2.fs(9.6), color=INK, va="top",
           linespacing=1.35)
    jr = n["robust_cluster"]
    jr_name = dict(SHOW).get(jr, f"c{jr}")
    cr = P2.cluster_col(jr, K)
    for q in range(3):
        t.add_patch(Rectangle((0.07 + q * 0.062, 0.155), 0.048, 0.085, fc=cr,
                              ec="none"))
    t.text(0.07 + 3 * 0.062 + 0.015, 0.195, f"the {jr_name.lower()} type (c{jr}) is\n"
           f"found by every algorithm\n(Jaccard \u2265 {n['robust_minJ']:.2f} with k-means and Ward)",
           fontsize=P2.fs(8.6), color=INK, va="center", linespacing=1.3)

    # (c) anatomy
    t = tile(fig, 0.05 + 2 * (tw + tg), ty, tw, th)
    t.text(0.07, 0.90, "Placed by the language atlas?", fontsize=P2.fs(11), color=MUTED,
           va="top")
    t.text(0.07, 0.76, "No", fontsize=P2.fs(27), weight="bold", color=RED, va="top")
    share_line = (("one share effect, " + ", ".join(f"c{c}" for c in n["lana_share_sig"]))
                  if n["lana_share_sig"] else "no share effect")
    t.text(0.07, 0.50, f"{n['lana_n_sig_sh']} of {n['lana_n_clusters']} types above the "
           f"shaft-shift null\nby mean atlas probability;\n{share_line}.\n"
           f"Largest |ρ| = {n['lana_max_abs_rho']:.2f}.",
           fontsize=P2.fs(9.4), color=INK, va="top", linespacing=1.35)
    b = t.inset_axes([0.30, 0.09, 0.60, 0.20])
    jt = n["lana_top_cluster"]
    vals = [n["lana_top_mean"], n["lana_top_null_sh"], n["lana_baseline"]]
    b.barh([2, 1, 0], vals, color=[P2.cluster_col(jt, K), GREY, "#e9edf1"],
           height=0.72)
    for yy, v in zip([2, 1, 0], vals):
        b.text(v + 0.004, yy, f"{v:.3f}", fontsize=P2.fs(6.6), color=MUTED, va="center")
    b.set_yticks([2, 1, 0])
    b.set_yticklabels([f"c{jt} (highest)", "shaft null", "cohort"], fontsize=P2.fs(6.5),
                      color=MUTED)
    b.set_xlim(0, max(vals) * 1.35); b.set_xticks([])
    for s_ in b.spines.values():
        s_.set_visible(False)
    b.tick_params(length=0, pad=P2.pw(2))
    b.patch.set_alpha(0)

    # ---- the sentence ------------------------------------------------------------
    fig.text(0.5, 0.135, "A response type is what an electrode does — not where it sits.",
             ha="center", va="center", fontsize=P2.fs(18), weight="bold", color=INK)
    fig.text(0.5, 0.082, f"{P2.FS_LABEL[FSET]}, time-warped so the GO "
             "cue falls at 50% of every block  ·  nulls: shaft-shift and within-patient "
             "permutation  ·  FDR across types",
             ha="center", va="center", fontsize=P2.fs(8.6), color=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"GA_graphical_abstract_{FSET}_K{K}.png"
    P2.save_png(fig, p, dpi=100, facecolor="white")      # no bbox_inches: stay 1200 px
    plt.close(fig)

    from PIL import Image
    with Image.open(p) as im:
        if im.size != (PX, PX):
            raise SystemExit(f"{p.name} is {im.size}, not {(PX, PX)}")

    if RAMP_ID is not None and ramp_ids:
        ramp_note = [f"  c{RAMP_ID} carries the label; its ramp is positive in "
                     f"{int((ramp_c[RAMP_ID] > 0).sum())} of 3 conditions."]
    else:
        ramp_note = ["  No cluster of this fit ramps before the cue - every HFA pre-cue slope",
                     "  is negative - so no type is labelled 'Ramp to cue'."]
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
        "'Ramp to cue' is the one label that is MEASURED, not read: the mean over the",
        "last fifth of the block before the GO cue minus the mean over its first fifth,",
        "averaged over the three conditions. The cluster drawn under that name must be",
        f"the largest, or this script refuses to run. Ranked (dB):",
        "  " + "   ".join(f"c{j} {ramp[j]:+.2f}" for j in np.argsort(-ramp)),
        *ramp_note,
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
        f"'The {jr_name.lower()} type is found by every algorithm' is read from "
        "FIG2_..._cluster_agreement.csv: the reference cluster with the highest minimum "
        f"Jaccard against k-means and Ward (c{jr}, {n['robust_minJ']:.2f}).",
    ]), p.with_name(p.stem + "_caption.txt"))
    print(f"  {time.time() - t0:.0f}s  -> {p.name}  ({PX}x{PX})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
