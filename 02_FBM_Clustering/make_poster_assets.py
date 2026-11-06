#!/usr/bin/env python3
"""make_poster_assets.py - the poster's panels redrawn at print resolution (2026-11-03).

    python make_poster_assets.py              # everything below
    python make_poster_assets.py --only row1 agreement stability separation neurosynth neurosynthT
    python make_poster_assets.py --only neurosynthT --no-render --maps speech_perception language_comprehension speech_production visual

Writes outputs/clustering/paper_figures/poster/assets/:
  row1_task_cohort.png   five equal tiles: the kept electrodes on the glass brain from the left,
                         from above and from the right (each patient its colour), the task
                         screens, and one electrode as the pipeline sees it (full ERSP over its
                         15-band summary); the tiles also singly as tile_*.png
  agreement_bars_table.png  the mean adjusted overlap of the matched clusters per comparison
                         (greys: dark = algorithms, light = feature sets) with the number of pairs
                         clearing both nulls, and the table of diagonal shares, chance and ARI
  stability_consensus.png / stability_K.png / stability_row.png
                         the consensus matrix at K 7 redrawn from the sweep's M, and the held-out
                         variance against K with the share of a cluster held by one patient
  neurosynth_15band.png / neurosynth_15zband.png
                         the eight-map NeuroSynth label matrix of member_measures/10_neurosynth_labels.py
                         (out_k7/neurosynth_labels/labels.csv, nothing recomputed) at print size, for
                         15 bands and for the z-scored companion; under each map name the map on the
                         fsaverage pial surface (ns_brain_<term>_<side>.png, reused with --no-render)
  neurosynth_<tag>_brains.png
                         the same with the cluster ids only and each cluster's glass brains before its
                         row (render_hemi, ns_cluster_<tag>_c<id>_<side>.png, reused with --no-render)
  neurosynth_T_<tag>.png  the matrix transposed for a wide box (3:1): the maps as rows with their pial
                         surfaces, the clusters as columns with their glass brains, ids only;
                         --maps <names> keeps only those rows (file suffix _<n>maps)
  separation_row.png     the two held-out confusions (one patient out, patient halves) and the
                         silhouette of every electrode per type against the Gaussian cloud;
                         separation_violin_loading.png is the loading alternative
Every panel uses the paper's cluster colours (cluster_col), its ink and greys, DejaVu Sans.
Reads outputs only; VTK renders the brains (screen unlocked).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("p2", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(P2)
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                      # noqa: E402
import lf_concat as CC                                    # noqa: E402

REF, K = "concat_bands15", 7
CL = ROOT / "outputs" / "clustering"
OUT = CL / "paper_figures" / "poster" / "assets"
COV = ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz"
INK, MUTED, GREY, RED, GREEN = P2.INK, P2.MUTED, P2.GREY, P2.RED, P2.GREEN
DARK, LIGHT = "#3a3a3a", "#9a9a9a"                        # FIG 2C's greys: algorithms, feature sets
NL = chr(10)
DPI = 300
EXAMPLE = ("EL035", "TTG_R1")                              # the paper's example electrode (FIG 1 B)
NO_RENDER = False                                           # --no-render: reuse the saved brain tiles (a locked screen crashes VTK)


def cache_input_dir():
    """The 03_ERSP folder the cache was built from, as 242 passes it (the cache key carries it)."""
    bs = chr(92)
    p = Path(bs * 2 + bs.join(["nasac-m2.unige.ch", "m-HumanNeuronLab", "ANALYSIS", "FLM", "Analysis_LoraFanda", "01_FBM_Analysis", "outputs", "03_ERSP"]))
    return p


def load_ref():
    d = P2.load_run(REF, K, "cnmf"); d["fset"] = REF
    C = P2.cube(d["X"], d)
    means = np.stack([C[d["lab"] == j].mean(0) for j in range(K)])
    d["order"] = [int(j) for j in P2.block_order(means, d)[0]]
    d["names"] = {int(k_): v for k_, v in P2.TYPE_NAMES[(REF, K)].items()}
    d["col"] = {j: P2.cluster_col(j, K) for j in range(K)}
    d["C"] = C
    return d


def tick_colours(ax, order, col, axis="x"):
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for tl, j in zip(labels, order):
        tl.set_color(col[j])


def render_all(side, d, cols, rads, alphas, zoom=1.30):
    """Every contact with radius > 0 on the glass brain of this side (the page's renderer)."""
    pl, ok = P2._scene(side, dict(xyz=d["xyz"], hemi=d["hemi"]), zoom)
    m = ok & (rads > 0) & np.isfinite(d["xyz"][:, 0])
    actor = None
    if m.sum():
        rgba = np.zeros((int(m.sum()), 4), np.uint8)
        rgba[:, :3] = np.clip(255 * np.asarray(cols)[m], 0, 255).astype(np.uint8)
        rgba[:, 3] = np.clip(np.asarray(alphas)[m], 0, 255).astype(np.uint8)
        cloud = P2.pv.PolyData(d["xyz"][m]); cloud["rgba"] = rgba; cloud["r"] = np.asarray(rads)[m].astype(float)
        g = cloud.glyph(orient=False, scale="r", geom=P2.pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12))
        actor = pl.add_mesh(g, scalars="rgba", rgba=True)
    img = pl.screenshot(return_img=True, transparent_background=True)
    if actor is not None:
        pl.remove_actor(actor)
    return P2._crop_alpha(np.asarray(img))


def image_axes(fig, spec):
    ax = fig.add_subplot(spec); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    return ax


# ---------------------------------------------------------------- row 1: the task and the cohort
def row1(d):
    tile = 3.0                                              # inches, every tile the same width
    pcol = P2.patient_colours(d)
    n = len(d["xyz"])
    cols = np.array([pcol[p][:3] for p in d["patient"]]); rads = np.full(n, 1.35); al = np.full(n, 235.0)
    brains = {s_: render_all(s_, d, cols, rads, al, zoom=1.25) for s_ in ("L", "T", "R")}
    # the task screens: the header of the opening figure, cropped from its paper render
    fig0 = CL / "paper_figures" / "paper" / "FIG0_cohort_v17.png"
    task0 = Image.open(fig0).convert("RGB"); w0, h0 = task0.size
    # the three conditions' screens, one above the other, so the tile is square
    bands = [(0.165, 0.415), (0.405, 0.655), (0.645, 0.895)]
    crops = [task0.crop((int(x0 * w0), int(0.005 * h0), int(x1 * w0), int(0.125 * h0))) for x0, x1 in bands]
    cw = max(c.width for c in crops); ch = crops[0].height
    task = Image.new("RGB", (cw, ch * 3), "white")
    for i_, c in enumerate(crops):
        task.paste(c, (0, i_ * ch))
    # one electrode as the pipeline sees it
    inp = cache_input_dir()
    df, Xc = CC.build_concat_dataset(inp, conditions=("audio", "picture", "reading"), require_high_activity=True, cache_dir=CC.DEFAULT_CONCAT_CACHE, verbose=False)
    i_raw = int(np.where((df.patient_id.values == EXAMPLE[0]) & (df.electrode.astype(str).values == EXAMPLE[1]))[0][0])
    raw = Xc[i_raw]                                         # (103, 900) dB, 0-400 Hz
    m = d["meta"]; i15 = [k_ for k_, (p, e) in enumerate(zip(m.patient_id, m.electrode)) if p == EXAMPLE[0] and P2.norm(e) == P2.norm(EXAMPLE[1])][0]
    cube = d["C"][i15]                                      # (cond, band, t)
    fig = plt.figure(figsize=(5 * tile, tile + 0.55), dpi=DPI)
    gs = GridSpec(1, 5, figure=fig, left=0.01, right=0.99, top=0.86, bottom=0.04, wspace=0.08)
    for k_, (s_, ttl) in enumerate((("L", f"left, {int((d['hemi'] == 'L').sum())} electrodes"), ("T", f"from above, {n}"), ("R", f"right, {int((d['hemi'] == 'R').sum())}"))):
        ax = image_axes(fig, gs[0, k_]); ax.imshow(brains[s_]); ax.set_title(ttl, fontsize=8, color=MUTED, loc="left")
    ax = image_axes(fig, gs[0, 3]); ax.imshow(task); ax.set_title("the task: stimulus, then a spoken answer", fontsize=8, color=MUTED, loc="left")
    sub = gs[0, 4].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    axr = fig.add_subplot(sub[0, 0])
    axr.imshow(raw, aspect="auto", origin="lower", cmap="bwr", vmin=-7, vmax=7, extent=(0, 900, 0, 400))
    for x in (300, 600):
        axr.axvline(x, color=INK, lw=0.6)
    axr.set_yticks([0, 200, 400]); axr.set_yticklabels(["0", "200", "400"], fontsize=6); axr.set_xticks([]); axr.tick_params(length=0)
    axr.set_title(f"one electrode ({EXAMPLE[0]} {EXAMPLE[1]}): full ERSP, ±7 dB", fontsize=8, color=MUTED, loc="left")
    axc = fig.add_subplot(sub[1, 0])
    P2.draw_mean(axc, cube, d, vlim=7.0, col=INK)
    nt = d["nt"]
    axc.set_xticks([(i + 0.5) * nt for i in range(3)]); axc.set_xticklabels([P2.COND_LABEL.get(c, c) for c in d["conds"]], fontsize=6.5)
    axc.set_yticks([0, 7, 14]); axc.set_yticklabels([d["bands"][i].replace("Hz", "") for i in (0, 7, 14)], fontsize=6); axc.tick_params(length=0, pad=1.5)
    axc.set_title("the same electrode as 15 bands × 30 bins", fontsize=8, color=MUTED, loc="left")
    p = OUT / "row1_task_cohort.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    # the tiles singly
    for s_, img in brains.items():
        Image.fromarray(img).save(OUT / f"tile_brain_{s_}.png")
    task.save(OUT / "tile_task.png")
    return p


# ---------------------------------------------------------------- figure 1 as a column
def row1_column(d):
    """The task screens; under them one electrode's full ERSP and its 15-band summary, condition
    by condition in the same columns; the three cohort brains at the bottom."""
    fig0 = CL / "paper_figures" / "paper" / "FIG0_cohort_v17.png"
    task0 = Image.open(fig0).convert("RGB"); w0, h0 = task0.size
    bands = [(0.170, 0.405), (0.415, 0.650), (0.660, 0.895)]
    screens = [task0.crop((int(x0 * w0), int(0.005 * h0), int(x1 * w0), int(0.125 * h0))) for x0, x1 in bands]
    inp = cache_input_dir()
    df, Xc = CC.build_concat_dataset(inp, conditions=("audio", "picture", "reading"), require_high_activity=True, cache_dir=CC.DEFAULT_CONCAT_CACHE, verbose=False)
    i_raw = int(np.where((df.patient_id.values == EXAMPLE[0]) & (df.electrode.astype(str).values == EXAMPLE[1]))[0][0])
    raw = Xc[i_raw]
    m = d["meta"]; i15 = [k_ for k_, (p, e) in enumerate(zip(m.patient_id, m.electrode)) if p == EXAMPLE[0] and P2.norm(e) == P2.norm(EXAMPLE[1])][0]
    cube = d["C"][i15]
    pcol = P2.patient_colours(d); n = len(d["xyz"])
    xt = lambda ax, c_: (ax.set_xticks([0, 50, 100] if c_ == 2 else [0, 50]), ax.set_xticklabels(["0", "50", "100 %"] if c_ == 2 else ["0", "50"], fontsize=6.5))
    cols = np.array([pcol[p][:3] for p in d["patient"]]); rads = np.full(n, 1.35); al = np.full(n, 235.0)
    brains = {}
    for s_ in ("L", "T", "R"):
        tile = OUT / f"tile_brain_{s_}.png"
        if NO_RENDER:                               # a locked screen crashes VTK outright: reuse the saved tile
            if not tile.exists():
                raise SystemExit(f"--no-render but no saved tile {tile.name}: run once with the screen unlocked")
            brains[s_] = np.asarray(Image.open(tile).convert("RGBA"))
        else:
            brains[s_] = render_all(s_, d, cols, rads, al, zoom=1.25)
            Image.fromarray(brains[s_]).save(tile)
    nt_raw = raw.shape[1] // 3; nt = d["nt"]
    fig = plt.figure(figsize=(7.4, 10.2), dpi=DPI)
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.25, 0.9, 1.7], left=0.09, right=0.93, top=0.985, bottom=0.03, hspace=0.28, wspace=0.05)
    for c_ in range(3):
        ax = image_axes(fig, gs[0, c_]); ax.imshow(screens[c_])
    for c_ in range(3):
        ax = fig.add_subplot(gs[1, c_])
        im = ax.imshow(raw[:, c_ * nt_raw:(c_ + 1) * nt_raw], aspect="auto", origin="lower", cmap="bwr", vmin=-7, vmax=7, extent=(0, 100, 0, 400))
        ax.axvline(50, color=INK, lw=0.7, ls="--"); xt(ax, c_); ax.tick_params(labelsize=6.5, length=2)
        if c_ == 0:
            ax.set_yticks([0, 100, 200, 300, 400]); ax.set_ylabel("Hz", fontsize=7.5)
            ax.set_title(f"the full ERSP of one electrode ({EXAMPLE[0]} {EXAMPLE[1]}), ±7 dB", fontsize=7.5, color=MUTED, loc="left")
        else:
            ax.set_yticks([])
    for c_ in range(3):
        ax = fig.add_subplot(gs[2, c_])
        ax.imshow(cube[c_], aspect="auto", origin="lower", cmap="bwr", vmin=-7, vmax=7, extent=(0, 100, -0.5, 14.5))
        ax.axvline(50, color=INK, lw=0.7, ls="--"); xt(ax, c_); ax.tick_params(length=2)
        if c_ == 0:
            ax.set_yticks([0, 4, 7, 9, 14]); ax.set_yticklabels([d["bands"][i].replace("Hz", "") for i in (0, 4, 7, 9, 14)], fontsize=6.5); ax.set_ylabel("Hz", fontsize=7.5)
            ax.set_title("the same electrode as the pipeline sees it: 15 bands × 30 time bins per condition", fontsize=7.5, color=MUTED, loc="left")
        else:
            ax.set_yticks([])
    cax = fig.add_axes([0.94, 0.40, 0.012, 0.30]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6.5)
    cax.set_title("dB", fontsize=7, color=MUTED, pad=3)
    for c_, (s_, ttl) in enumerate((("L", f"left, {int((d['hemi'] == 'L').sum())} electrodes"), ("T", f"from above, {n}"), ("R", f"right, {int((d['hemi'] == 'R').sum())}"))):
        ax = image_axes(fig, gs[3, c_]); ax.imshow(brains[s_]); ax.set_title(ttl, fontsize=7.5, color=MUTED, loc="left")
    fig.text(0.09, 0.005, "the 1587 electrodes that passed the response gate, each patient its colour", fontsize=7.5, color=MUTED, va="bottom")
    p = OUT / "figure1_column.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ---------------------------------------------------------------- agreement: grey bars and the table
def agreement(d):
    cap = sorted(glob.glob(str(CL / "paper_figures" / "FIG4_K07_run*_caption.txt")))[-1]
    txt = Path(cap).read_text(encoding="utf-8")
    rows = []
    for m in re.finditer(r"^(.+?): matched r ([+-][0-9.]+) against ([+-][0-9.]+) for unmatched pairs; mean adjusted overlap ([0-9.]+) \(raw Jaccard ([0-9.]+)\); (\d+) of (\d+) pairs above the free null, (\d+) above the within-patient null; diagonal share ([0-9.]+) of (\d+) electrodes against ([0-9.]+) \+- ([0-9.]+)", txt, re.M):
        rows.append(dict(name=m.group(1).strip(), r=float(m.group(2)), adj=float(m.group(4)), jac=float(m.group(5)), n_free=int(m.group(6)), n_pairs=int(m.group(7)),
                         n_within=int(m.group(8)), diag=float(m.group(9)), chance=float(m.group(11)), chance_sd=float(m.group(12))))
    nums = json.loads((CL / "paper2_validation" / "numbers.json").read_text(encoding="utf-8"))["values"]
    ari = {**nums["ari_algo"], **nums["ari_feat"]}
    ari_key = {"convex NMF vs k-means": "k-means vs convex NMF", "convex NMF vs Ward": "Ward vs convex NMF", "k-means vs Ward": "k-means vs Ward",
               "15 bands vs 5 bands": "15 bands vs 5 bands", "15 bands vs 2 bands": "15 bands vs 2 bands", "15 bands vs HFA": "15 bands vs HFA"}
    algo = [r for r in rows if "NMF" in r["name"] or "Ward" in r["name"]]
    feat = [r for r in rows if "bands vs" in r["name"]]
    fig = plt.figure(figsize=(7.4, 5.4), dpi=DPI)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0], left=0.30, right=0.97, top=0.93, bottom=0.03, hspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    items = algo + feat
    y = np.arange(len(items))[::-1]
    for yi, r in zip(y, items):
        c = DARK if r in algo else LIGHT
        ax.barh(yi, r["adj"], color=c, height=0.66)
        ax.text(r["adj"] + 0.012, yi, f"{r['adj']:.2f}   ·   {min(r['n_free'], r['n_within'])} of {r['n_pairs']} pairs clear both nulls", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r["name"] for r in items], fontsize=8)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("mean adjusted overlap of the matched clusters (0 chance, 1 identical)", fontsize=7.5, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("dark: another algorithm on the same 15 bands  ·  light: convex NMF on another description", fontsize=8, loc="left", color=INK)
    # the table
    axt = fig.add_subplot(gs[1, 0]); axt.axis("off")
    cells = [[r["name"], f"{r['diag']:.2f}", f"{r['chance']:.2f} ± {r['chance_sd']:.2f}", f"{ari[ari_key[r['name']]]['ari']:.2f}", f"{r['r']:+.2f}"] for r in items]
    tb = axt.table(cellText=cells, colLabels=["comparison", "diagonal share", "random pairings", "ARI", "matched r"], loc="upper left", cellLoc="center", colLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.2); tb.scale(1.0, 1.25)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_edgecolor(GREY); cell.set_linewidth(0.5)
        if ri == 0:
            cell.set_text_props(color=MUTED, fontweight="bold")
        if ci == 0:
            cell.set_text_props(ha="left"); cell.set_width(0.34)
    axt.set_title("the share of electrodes on the matched diagonal, against random pairings of the same two solutions", fontsize=8, loc="left", color=INK)
    p = OUT / "agreement_bars_table.png"; fig.savefig(p, facecolor="white", bbox_inches="tight"); plt.close(fig)
    # the bars alone (the poster's table is built natively from the values printed here)
    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=DPI)
    for yi, r in zip(y, items):
        ax.barh(yi, r["adj"], color=DARK if r in algo else LIGHT, height=0.66)
        ax.text(r["adj"] + 0.012, yi, f"{r['adj']:.2f}   ·   {min(r['n_free'], r['n_within'])} of {r['n_pairs']} pairs clear both nulls", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r["name"] for r in items], fontsize=8)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("mean adjusted overlap of the matched clusters (0 chance, 1 identical)", fontsize=7.5, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("dark: another algorithm on the same 15 bands  ·  light: convex NMF on another description", fontsize=8, loc="left", color=INK)
    fig.savefig(OUT / "agreement_bars.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
    # the same bars, narrower, for the poster's column (2026-11-06)
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=DPI)
    for yi, r in zip(y, items):
        ax.barh(yi, r["adj"], color=DARK if r in algo else LIGHT, height=0.66)
        ax.text(r["adj"] + 0.012, yi, f"{r['adj']:.2f} · {min(r['n_free'], r['n_within'])} of {r['n_pairs']} clear both nulls", va="center", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r["name"] for r in items], fontsize=8)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("mean adjusted overlap of the matched clusters" + NL + "(0 chance, 1 identical)", fontsize=7.5, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("dark: another algorithm on the same 15 bands" + NL + "light: convex NMF on another description", fontsize=8, loc="left", color=INK)
    fig.savefig(OUT / "agreement_bars_narrow.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
    (OUT / "agreement_table.md").write_text(NL.join(["| comparison | diagonal share | random pairings | ARI | matched r |", "|---|---|---|---|---|"]
        + [f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |" for c in cells]) + NL, encoding="utf-8")
    return p


# ---------------------------------------------------------------- stability: the consensus matrix and the K panel
def stability(d):
    cache = LR.current_cache()
    cdir = CL / "consensus" / cache / REF / "cnmf"
    it = pd.read_csv(cdir / f"items_k{K:02d}.csv"); Mfull = np.load(cdir / f"M_k{K:02d}.npy").astype(float)
    it["key"] = [f"{p}|{P2.norm(e)}" for p, e in zip(it.patient_id, it.electrode)]
    keys = [f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"].patient_id, d["meta"].electrode)]
    pos = {k_: i for i, k_ in enumerate(it.key)}
    idx = np.array([pos[k_] for k_ in keys]); M = Mfull[np.ix_(idx, idx)]
    lab = np.asarray(d["lab"])
    # electrodes grouped by published type in the figure order, within a type by their mean consensus with it
    groups = []
    for j in d["order"]:
        members = np.where(lab == j)[0]
        own = np.array([M[i, members[members != i]].mean() if len(members) > 1 else 0 for i in members])
        groups.append(members[np.argsort(-own)])
    perm = np.concatenate(groups); Mo = M[np.ix_(perm, perm)]
    sizes = [len(g) for g in groups]; edges = np.cumsum([0] + sizes)
    # --- the matrix
    fig = plt.figure(figsize=(5.2, 5.4), dpi=DPI)
    ax = fig.add_axes([0.12, 0.16, 0.80, 0.77])
    im = ax.imshow(Mo, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="white", lw=0.7); ax.axvline(e - 0.5, color="white", lw=0.7)
    mids = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(mids); ax.set_xticklabels([f"c{j}" for j in d["order"]], fontsize=8); ax.set_yticks(mids); ax.set_yticklabels([f"c{j}" for j in d["order"]], fontsize=8)
    tick_colours(ax, d["order"], d["col"], "x"); tick_colours(ax, d["order"], d["col"], "y"); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08); cb.ax.tick_params(labelsize=7); cb.set_label("consensus: share of the 200 resamples in which two electrodes land together", fontsize=7.5)
    ax.set_title("every pair of electrodes, grouped by type, after 200 refits on 80 % of the patients", fontsize=8.5, loc="left", color=INK)
    pm = OUT / "stability_consensus.png"; fig.savefig(pm, facecolor="white", bbox_inches="tight"); plt.close(fig)
    # --- held-out variance against K, with the share of a cluster held by one patient
    hv = pd.concat([pd.read_csv(f) for f in glob.glob(str(CL / "bsf_comparison" / "heldout_variance*.csv"))], ignore_index=True)
    hv = hv[(hv.feature_set == REF) & (hv.method == "cnmf") & hv.scheme.isin(P2.HOME_SCHEMES) & (hv.cache == cache)]
    curve = hv.groupby("k").var_explained.agg(["mean", "std"]).reset_index()
    lbk = pd.read_csv(d["run"] / "cluster_labels_by_k.csv"); pats = np.asarray(d["patient"]).astype(str)
    ks, ndom = [], []
    for c_ in lbk.columns:
        if not c_.startswith("k_"):
            continue
        k_ = int(c_[2:]); l_ = lbk[c_].to_numpy(int); cnt = 0
        for c in range(k_):
            m_ = l_ == c
            if m_.any() and float(pd.Series(pats[m_]).value_counts(normalize=True).iloc[0]) > 0.5:
                cnt += 1
        ks.append(k_); ndom.append(cnt)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=DPI)
    ax.fill_between(curve.k, curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=GREY, alpha=0.5, lw=0)
    ax.plot(curve.k, curve["mean"], "-o", color=INK, ms=3.5, lw=1.6, label="held-out variance")
    ax.set_xlabel("K", fontsize=8); ax.set_ylabel("held-out variance", fontsize=8); ax.tick_params(labelsize=7, length=2)
    ax2 = ax.twinx()
    ax2.plot(ks, ndom, "-o", color=RED, ms=3.2, lw=1.4, label="clusters > 50% one patient")
    ax2.set_ylim(-0.3, max(ndom) + 1); ax2.set_yticks(range(0, max(ndom) + 1, max(1, (max(ndom) + 1) // 6)))
    ax2.set_ylabel("clusters owned by one patient", fontsize=8, color=RED); ax2.tick_params(labelsize=7, length=2, colors=RED)
    ax.axvline(K, color=GREEN, lw=1.2, ls="--"); ax.text(K + 0.3, ax.get_ylim()[1] - 0.03 * np.ptp(ax.get_ylim()), "K 7: the last K with no cluster" + NL + "owned by one patient", fontsize=7, color=GREEN, va="top")
    ax.set_xlim(min(curve.k) - 0.5, max(curve.k) + 0.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=6.6, frameon=False, loc="lower right")       # short, inside, bottom right (2026-11-06)
    ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    ax.set_title("the held-out curve never peaks inside the range; the patient rule chooses K", fontsize=8.5, loc="left", color=INK)
    pk = OUT / "stability_K.png"; fig.savefig(pk, facecolor="white", bbox_inches="tight"); plt.close(fig)
    # --- the two side by side
    a_, b_ = Image.open(pm).convert("RGB"), Image.open(pk).convert("RGB")
    h = max(a_.height, b_.height); sheet = Image.new("RGB", (a_.width + b_.width + 80, h), "white")
    sheet.paste(a_, (0, (h - a_.height) // 2)); sheet.paste(b_, (a_.width + 80, (h - b_.height) // 2)); sheet.save(OUT / "stability_row.png")
    return OUT / "stability_row.png"


# ---------------------------------------------------------------- separation: the two confusions and the silhouette violins
def confusion(ax, Mx, order, col, title):
    R = Mx / np.maximum(Mx.sum(1, keepdims=True), 1e-12); Ro = R[np.ix_(order, order)]
    ax.imshow(Ro, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{100 * Ro[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if Ro[i, j] > 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels([f"c{j}" for j in order], fontsize=8); ax.set_yticklabels([f"c{j}" for j in order], fontsize=8)
    tick_colours(ax, order, col, "x"); tick_colours(ax, order, col, "y"); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlabel("placed in", fontsize=7.5, color=MUTED); ax.set_ylabel("published type", fontsize=7.5, color=MUTED)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)


def violin(ax, df, col_name, order, col, title, band=None, lines=()):
    rng = np.random.default_rng(0)
    data = [df.loc[df.cluster == j, col_name].dropna().to_numpy() for j in order]
    parts = ax.violinplot(data, positions=range(len(order)), showmedians=False, showextrema=False, widths=0.9)
    for body, j in zip(parts["bodies"], order):
        body.set_facecolor(col[j]); body.set_alpha(0.28); body.set_edgecolor("none")
    if band is not None:
        ax.axhspan(band[0] - 2 * band[1], band[0] + 2 * band[1], color="#ececec", zorder=0)
    for p, (j, v) in enumerate(zip(order, data)):
        ax.scatter(p + rng.uniform(-0.26, 0.26, len(v)), v, s=7, color=col[j], alpha=0.6, lw=0.3, edgecolor="white")
        ax.plot([p - 0.36, p + 0.36], [np.median(v)] * 2, color=INK, lw=2.0, zorder=3)
    for y_, st in lines:
        ax.axhline(y_, color=MUTED, lw=0.9, ls=st)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"c{j}" for j in order], fontsize=8); tick_colours(ax, order, col, "x")
    ax.tick_params(labelsize=7, length=2); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)


def separation(d):
    rid = f"cnmf__{REF}__{d['run'].name}"
    ho = json.loads((COV / "runs" / rid / "heldout.json").read_text(encoding="utf-8"))
    mm = pd.read_csv(CL / "paper2_validation" / "member_measures" / f"member_measures_{REF}.csv")
    pt = pd.read_csv(CL / "mmd" / LR.current_cache() / REF / "cnmf" / "per_type.csv"); pt = pt[pt.version == "argmax"]
    cloud = (float(pt.sil_cloud_mean.iloc[0]), float(pt.sil_cloud_sd.iloc[0]))
    fig = plt.figure(figsize=(13.2, 4.3), dpi=DPI)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.25], left=0.04, right=0.99, top=0.88, bottom=0.12, wspace=0.32)
    confusion(fig.add_subplot(gs[0, 0]), np.asarray(ho["lopo"]["matrix"], float), d["order"], d["col"], f"leave one patient out: {100 * ho['lopo']['overall']:.0f} % land in their type")
    confusion(fig.add_subplot(gs[0, 1]), np.asarray(ho["halves"]["matrix"], float), d["order"], d["col"], f"refit on half the patients: {100 * ho['halves']['overall']:.0f} %")
    violin(fig.add_subplot(gs[0, 2]), mm, "silhouette", d["order"], d["col"], "silhouette of every electrode; grey: the Gaussian cloud's own range", band=cloud, lines=[(0.0, "-")])
    p = OUT / "separation_row.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    # the leave-one-patient-out matrix alone, ticks in the type colours, for the poster (2026-11-06)
    fig, ax = plt.subplots(figsize=(4.4, 4.3), dpi=DPI)
    confusion(ax, np.asarray(ho["lopo"]["matrix"], float), d["order"], d["col"], f"leave one patient out: {100 * ho['lopo']['overall']:.0f} % land in their type")
    fig.savefig(OUT / "separation_lopo.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=DPI)
    violin(ax, mm, "loading", d["order"], d["col"], "loading of every electrode on its own type (dashed 0.3 draw cut, dotted 0.5)", lines=[(P2.LOAD_CUT, "--"), (0.5, ":")])
    ax.set_ylim(0, 1.02)
    pl_ = OUT / "separation_violin_loading.png"; fig.savefig(pl_, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------------- where they sit: the NeuroSynth label matrix
MM = ROOT / "member_measures" / "out_k7"                   # every feature set at K 7 (member_measures/10_neurosynth_labels.py)
NS_MAPS = [("Perception", "speech perception"), ("Perception", "visual"), ("Reading", "reading"),
           ("Comprehension", "language comprehension"), ("Comprehension", "semantic"),
           ("Retrieval", "retrieval"), ("Production", "speech production"), ("Production", "premotor")]
NS_GREY = "#9aa0a6"


def wrap_name(name, width=18):
    lines, cur = [], ""
    for w in name.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return NL.join(lines + [cur])


NS_VOL = ROOT.parent / "04_FBM_Pooling" / "neurosynth"    # the association-test z maps (MNI152, 2 mm) the pooling samples
MNI305_TO_152 = np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496],
                          [-0.0130, -0.0093, 0.9971, 1.1840], [0.0, 0.0, 0.0, 1.0]])   # lf_pool's, as member_measures/4_neurosynth.py applies it
NS_ZMAX = 8.0
NS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("ns", ["#f7b267", "#d7301f", "#7f0000"])
_NS_Z: dict = {}


def ns_vertex_z(term):
    """The map's z at every pial vertex of both hemispheres: the vertex (MNI305) goes to MNI152 with
    lf_pool's affine and reads the nearest voxel, as the contacts do in member_measures/4_neurosynth.py.
    The thresholded map lives in the grey matter under the pial surface, so the maximum over 0, 1.5 and
    3 mm along the vertex normal, both ways, is kept."""
    if term in _NS_Z:
        return _NS_Z[term]
    import nibabel as nb
    folder = next(p for p in (NS_VOL / term.replace(" ", "_"), NS_VOL / term) if p.is_dir())
    f = sorted(folder.glob("*association-test_z_FDR*.nii.gz"))[0]
    img = nb.load(str(f)); vol = np.asarray(img.dataobj, float); inv = np.linalg.inv(img.affine)
    out = {}
    for side in ("L", "R"):
        m = P2.hemi_mesh(side)
        v = np.asarray(m.points, float); nrm = np.asarray(m.point_normals, float)
        best = np.zeros(len(v))
        for off in (0.0, 1.5, -1.5, 3.0, -3.0):
            xyz = v + off * nrm
            mni = (np.c_[xyz, np.ones(len(xyz))] @ MNI305_TO_152.T)[:, :3]
            ijk = np.round((np.c_[mni, np.ones(len(mni))] @ inv.T)[:, :3]).astype(int)
            ok = (ijk >= 0).all(1) & (ijk < np.array(vol.shape[:3])).all(1)
            z = np.zeros(len(v)); z[ok] = vol[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            best = np.maximum(best, z)
        out[side] = best
    _NS_Z[term] = out
    print(f"    {term}: {int((out['L'] > 0).sum())} left and {int((out['R'] > 0).sum())} right vertices inside")
    return out


def ns_brain_image(term, side):
    """One hemisphere's pial surface from the outside, opaque, with the map's z > 0 painted on it
    (NS_CMAP from 0 to NS_ZMAX); the paper's camera and lighting. Saved as assets/ns_brain_<term>_<side>.png
    and reused with --no-render."""
    p = OUT / f"ns_brain_{term.replace(' ', '_')}_{side}.png"
    if p.exists() and NO_RENDER:
        return np.asarray(Image.open(p).convert("RGBA"))
    z = ns_vertex_z(term)[side]
    m = P2.hemi_mesh(side).copy()
    rgb = np.tile(np.array(matplotlib.colors.to_rgb(P2.RC.BRAIN_COLOR)) * 255, (len(z), 1))
    ins = z > 0
    rgb[ins] = NS_CMAP(np.clip(z[ins] / NS_ZMAX, 0, 1))[:, :3] * 255
    m["rgb"] = rgb.astype(np.uint8)
    pl = P2.pv.Plotter(off_screen=True, window_size=P2.RENDER_PX)
    pl.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True, specular=P2.RC.BRAIN_SPECULAR,
                specular_power=P2.RC.BRAIN_SPECULAR_POWER, ambient=P2.RC.BRAIN_AMBIENT, diffuse=P2.RC.BRAIN_DIFFUSE)
    b = m.bounds
    cx, cy, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2
    dist = 2.4 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    pl.camera_position = ((cx - dist if side == "L" else cx + dist, cy, cz), (cx, cy, cz), (0, 0, 1))
    pl.enable_parallel_projection()
    W, H = P2.RENDER_PX
    vis = 2.0 * pl.camera.parallel_scale
    fit = min(vis / (1.06 * (b[5] - b[4])), vis * (W / H) / (1.06 * (b[3] - b[2])))
    pl.camera.zoom(min(1.30, fit))
    img = P2._crop_alpha(np.asarray(pl.screenshot(return_img=True, transparent_background=True)))
    pl.close()
    Image.fromarray(img).save(p)
    return img


def cluster_brain(tag, c, side, d, zrun):
    """The cluster's electrodes on the glass brain of one hemisphere under the paper's rule (render_hemi:
    loading >= LOAD_CUT drawn, colour, size and opacity from the loading). A z-scored cluster is drawn in
    the colour of the published type it matches. Saved as assets/ns_cluster_<tag>_c<c>_<side>.png and
    reused with --no-render."""
    p = OUT / f"ns_cluster_{tag}_c{c}_{side}.png"
    if p.exists() and NO_RENDER:
        return np.asarray(Image.open(p).convert("RGBA"))
    if tag == "15band":
        img, _ = P2.render_hemi(side, d, c)
    else:
        dz, colmap = zrun()
        orig = P2.cluster_col
        P2.cluster_col = lambda j, kk: colmap.get(j, (0.60, 0.60, 0.60))
        try:
            img, _ = P2.render_hemi(side, dz, c)
        finally:
            P2.cluster_col = orig
    Image.fromarray(img).save(p)
    return img


def neurosynth(d):
    """The eight-map label matrix of member_measures/10_neurosynth_labels.py redrawn at print size,
    for 15 bands and for the z-scored companion: bubble and number = % of the type's electrodes
    inside the NeuroSynth map (z > 0), colour = that share against the cohort's (log scale),
    outlined = the share is above the cohort's in 97.5 % of patient resamples. Everything is read
    from out_k7/neurosynth_labels/labels.csv, nothing is recomputed. The z-scored clusters sit in
    the row of the published type they are matched to (out_k7/matching.csv) and carry its colour;
    a cluster that is more than half one patient is greyed with a cross. Under each map name the
    map itself on the fsaverage pial surface, from the left and from the right (ns_brain_image).
    Two versions per set: neurosynth_<tag>.png with the type names, neurosynth_<tag>_brains.png
    with the cluster ids only and each cluster's glass brains (cluster_brain) before its row."""
    from matplotlib.colors import TwoSlopeNorm
    lab = pd.read_csv(MM / "neurosynth_labels" / "labels.csv")
    match = pd.read_csv(MM / "matching.csv")
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2); cmap = plt.get_cmap("RdBu_r")
    names = [m for _, m in NS_MAPS]
    _z = {}

    def zrun():
        """The z-scored run (its newest cNMF run, K 7) and its clusters' colours = their matched types'."""
        if "d" not in _z:
            dz = P2.load_run("concat_bands15z", K, "cnmf"); dz["fset"] = "concat_bands15z"
            assert len(dz["xyz"]) == len(d["xyz"]) and np.allclose(np.nan_to_num(dz["xyz"]), np.nan_to_num(d["xyz"])), "the z run's electrodes are not in the published run's order"
            mz = match[match.fset == "15zband"]
            _z["d"] = dz; _z["col"] = {int(r.cluster): d["col"][int(r.type_15band)] for r in mz.itertuples()}
        return _z["d"], _z["col"]

    out = {}
    for tag in ("15band", "15zband"):
        t = lab[(lab.fset == tag) & (lab.k == K)]
        meas = pd.read_csv(MM / tag / "measures.csv")
        npat = meas.assign(p=meas.key.str.split("|").str[0]).groupby("type").p.nunique()
        rows = []                                          # (cluster, type name, colour, published type)
        for j in d["order"]:
            if tag == "15band":
                rows.append((j, d["names"].get(j, f"c{j}"), d["col"][j], j))
            else:
                m = match[(match.fset == tag) & (match.type_15band == j)]
                if not m.empty:
                    rows.append((int(m.cluster.iloc[0]), d["names"].get(j, f"c{j}"), d["col"][j], j))
        n_rows = len(rows)
        for brains in (False, True):
            fig_h = 1.4 + 0.46 * n_rows + 1.15                 # 1.15 in for the brains under the names
            fig, ax = plt.subplots(figsize=(7.4, fig_h), dpi=DPI)
            left = 0.07 if brains else 0.20
            fig.subplots_adjust(left=left, right=0.955, top=1 - 0.65 / fig_h, bottom=1.95 / fig_h)
            xl, xn = (-2.95, -3.05) if brains else (-0.6, -0.7)   # the axes' left edge and where the row label sits
            print(f"  neurosynth {tag}{' with the cluster brains' if brains else ''}:")
            brain_rows = []
            for r_, (c, name, col, j) in enumerate(rows):
                y = n_rows - 1 - r_
                tc = t[t.cluster == c].set_index("map")
                top_share = float(tc.top_patient_share.iloc[0]); bad = top_share > 0.5
                for x, mname in enumerate(names):
                    q = tc.loc[mname]
                    pct = float(q.pct_inside); le = float(np.log2(max(float(q.enrichment), 1e-3)))
                    face = NS_GREY if bad else cmap(norm(float(np.clip(le, -2, 2))))
                    sure = bool(q.outlined) and not bad
                    ax.scatter(x, y, s=70 + 360 * min(pct, 60) / 60, c=[face], alpha=0.5 if bad else 1.0,
                               edgecolors=INK if sure else "none", linewidths=1.3 if sure else 0, zorder=3)
                    ax.text(x, y, f"{pct:.0f}", ha="center", va="center", fontsize=6.0, zorder=4,
                            color="white" if (not bad and abs(le) > 1.1) else INK)
                n_ = int(tc.n.iloc[0]); p_ = int(npat.get(c, 0))
                if brains:
                    head = f"c{c}" if tag == "15band" else f"z{c}"
                    sub = ("" if tag == "15band" else f"≈ c{j} · ") + f"{n_} electrodes, {p_} patients"
                    brain_rows.append((y, c))
                else:
                    head = wrap_name(name)
                    sub = ("" if tag == "15band" else f"z{c} · ") + f"{n_} electrodes, {p_} patients"
                if bad:
                    sub += f", {100 * top_share:.0f} % {tc.top_patient.iloc[0]}"
                ax.text(xn, y + 0.04, ("✗  " if bad else "") + head, ha="right", va="bottom", fontsize=7.4,
                        color="#b2182b" if bad else col, fontweight="bold", linespacing=1.05)
                ax.text(xn, y - 0.08, sub, ha="right", va="top", fontsize=6.0, color=MUTED)
                sure_maps = tc[tc.outlined.astype(bool)].sort_values("enrichment", ascending=False).index.tolist()
                if bad:
                    label = "one patient: the profile is" + NL + "that patient's coverage"
                elif sure_maps:
                    label = NL.join(sure_maps[:3])
                else:
                    label = "no map"
                ax.text(len(names) - 0.45, y, label, ha="left", va="center", fontsize=6.8, linespacing=1.15,
                        color=MUTED if (bad or not sure_maps) else INK, style="italic" if (bad or not sure_maps) else "normal")
                print(f"    {head.replace(NL, ' '):36s} {label.replace(NL, ' / ')}")
            stages = [s for s, _ in NS_MAPS]
            for s in dict.fromkeys(stages):
                cs = [i for i, x in enumerate(stages) if x == s]
                ax.text(float(np.mean(cs)), n_rows - 0.38, s, ha="center", va="bottom", fontsize=6.4, color=INK)
                ax.plot([min(cs) - 0.42, max(cs) + 0.42], [n_rows - 0.44] * 2, color=INK, lw=0.6)
            coh = t[t.cluster == rows[0][0]].set_index("map").cohort_pct
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([f"{m.replace(' ', NL)}{NL}{float(coh[m]):.0f} %" for m in names], fontsize=6.0, color=INK, linespacing=1.15)
            ax.set_yticks([]); ax.set_xlim(xl, len(names) + 2.1); ax.set_ylim(-0.7, n_rows + 0.25)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.tick_params(length=0)
            ax.text(len(names) - 0.45, n_rows - 0.38, "where the type sits more often" + NL + "than electrodes in general", ha="left", va="bottom", fontsize=6.2, color=INK, linespacing=1.15)
            aw, ah = 7.4 * (0.955 - left), fig_h - 0.65 - 1.95
            x0_, x1_ = ax.get_xlim(); u = 1.0 / (x1_ - x0_)
            y0_, y1_ = ax.get_ylim(); v = 1.0 / (y1_ - y0_)
            # the cluster's own electrodes on the glass brain, left and right, before its row
            for y, c in brain_rows:
                for k_, side in enumerate(("L", "R")):
                    im = cluster_brain(tag, c, side, d, zrun)
                    w_f = 0.92 * u; h_f = w_f * aw / ah * im.shape[0] / im.shape[1]
                    ia = ax.inset_axes([(-2.45 + 0.95 * k_ - x0_) * u - w_f / 2, (y - y0_) * v - h_f / 2, w_f, h_f])
                    ia.imshow(im); ia.axis("off")
            if brains:
                for k_, side in enumerate(("L", "R")):
                    ax.text(-2.45 + 0.95 * k_, n_rows - 0.38, "left" if side == "L" else "right", ha="center", va="bottom", fontsize=6.4, color=INK)
                ax.plot([-2.45 - 0.42, -1.5 + 0.42], [n_rows - 0.44] * 2, color=INK, lw=0.6)
            # the maps themselves under the names: the pial surface from the left and from the right, z > 0 painted on it
            y_top = -0.34 / ah
            for x, mname in enumerate(names):
                for k_, side in enumerate(("L", "R")):
                    im = ns_brain_image(mname, side)
                    w_f = 0.92 * u; h_f = w_f * aw / ah * im.shape[0] / im.shape[1]
                    y_f = y_top - (k_ + 1) * h_f - k_ * 0.03 / ah
                    ia = ax.inset_axes([(x - x0_) * u - w_f / 2, y_f, w_f, h_f])
                    ia.imshow(im); ia.axis("off")
                    if x == 0:
                        ia.text(-0.1, 0.5, "left" if side == "L" else "right", transform=ia.transAxes, ha="right", va="center", fontsize=6.0, color=MUTED)
            cbx, cbw = (len(names) - 0.1 - x0_) * u, 1.95 * u      # the two colour bars right of the last column
            cax2 = ax.inset_axes([cbx, y_top - 0.62 / ah, cbw, 0.035])
            cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, NS_ZMAX), cmap=NS_CMAP), cax=cax2, orientation="horizontal", ticks=[0, 4, 8])
            cb2.ax.set_xticklabels(["0", "4", "8+"], fontsize=5.8); cb2.ax.tick_params(length=2, pad=1.5)
            cax2.set_title("the map's z on the pial surface", fontsize=6.0, color=MUTED, pad=2)
            cax = ax.inset_axes([cbx, -0.17, cbw, 0.035])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-2, -1, 0, 1, 2])
            cb.ax.set_xticklabels(["¼×", "½×", "1×", "2×", "4×"], fontsize=5.8); cb.ax.tick_params(length=2, pad=1.5)
            cax.set_title("the share against all electrodes", fontsize=6.0, color=MUTED, pad=2)
            ax.set_title("15 bands, K 7" if tag == "15band" else "15 bands z-scored, K 7  ·  each cluster in the row of the published type it matches",
                         fontsize=7.2, loc="left", color=INK, pad=22)
            ax.annotate("bubble and number: % of the type's electrodes inside the map  ·  under each map: % of all electrodes inside" + NL
                        + "outlined: above the cohort in 97.5 % of patient resamples  ·  no spatial null"
                        + ("  ·  glass brains: the cluster's electrodes at loading ≥ 0.3, colour and size by loading" if brains else ""),
                        xy=(0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points", fontsize=5.9, color=MUTED, va="bottom", linespacing=1.25)
            p = OUT / f"neurosynth_{tag}{'_brains' if brains else ''}.png"; fig.savefig(p, facecolor="white", bbox_inches="tight"); plt.close(fig)
            out[(tag, brains)] = p
            im = Image.open(p); print(f"    -> {p.name}  {im.size[0]} x {im.size[1]} px")
    return out[("15band", False)]


# ---------------------------------------------------------------- the same matrix transposed: maps as rows, clusters as columns
_ZRUN: dict = {}


def z_run(d, match):
    """The z-scored run (its newest cNMF run, K 7) and its clusters' colours = their matched published types'."""
    if "d" not in _ZRUN:
        dz = P2.load_run("concat_bands15z", K, "cnmf"); dz["fset"] = "concat_bands15z"
        assert len(dz["xyz"]) == len(d["xyz"]) and np.allclose(np.nan_to_num(dz["xyz"]), np.nan_to_num(d["xyz"])), "the z run's electrodes are not in the published run's order"
        mz = match[match.fset == "15zband"]
        _ZRUN["d"] = dz; _ZRUN["col"] = {int(r.cluster): d["col"][int(r.type_15band)] for r in mz.itertuples()}
    return _ZRUN["d"], _ZRUN["col"]


def neurosynth_T(d, keep=None):
    """neurosynth()'s label matrix transposed for a wide box, three times as wide as high: the maps as
    rows, each with its pial surfaces beside the name, the clusters as columns, each with its glass
    brains under the id (ids only, no type names). keep: the maps to draw, in NS_MAPS' order; default
    all eight. Written as assets/neurosynth_T_<tag>.png, or neurosynth_T_<tag>_<n>maps.png with keep."""
    from matplotlib.colors import TwoSlopeNorm
    lab = pd.read_csv(MM / "neurosynth_labels" / "labels.csv")
    match = pd.read_csv(MM / "matching.csv")
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2); cmap = plt.get_cmap("RdBu_r")
    maps = [(s, m) for s, m in NS_MAPS if keep is None or m in keep]
    names = [m for _, m in maps]; nm = len(names)
    if keep is not None and nm != len(keep):
        raise SystemExit(f"--maps: unknown map(s) in {keep}; the maps are {[m for _, m in NS_MAPS]}")
    out = {}
    for tag in ("15band", "15zband"):
        t = lab[(lab.fset == tag) & (lab.k == K)]
        meas = pd.read_csv(MM / tag / "measures.csv")
        npat = meas.assign(p=meas.key.str.split("|").str[0]).groupby("type").p.nunique()
        cols = []                                          # (cluster, colour, published type)
        for j in d["order"]:
            if tag == "15band":
                cols.append((j, d["col"][j], j))
            else:
                m = match[(match.fset == tag) & (match.type_15band == j)]
                if not m.empty:
                    cols.append((int(m.cluster.iloc[0]), d["col"][j], j))
        nc = len(cols)
        RH, TOP, LB, RB, CAP = 0.42, 0.55, 2.2, 1.15, 0.55     # inches: row pitch, the blocks around the matrix, a glass brain's largest width
        anybad = any(float(t[t.cluster == c].top_patient_share.iloc[0]) > 0.5 for c, _, _ in cols)
        max_lines = (1 if tag == '15band' else 2) + (1 if anybad else 0)      # count lines under the ids
        cw = CAP
        for _ in range(2):                                     # the brains' width follows the column pitch, which follows the height
            BOT = 0.19 + 0.1 * max_lines + 0.10 + 0.73 * cw + 0.12
            mh = RH * nm; H = TOP + mh + BOT; W = 3.0 * H; mw = W - LB - RB - 0.15; pitch = mw / nc
            cw = min(CAP, (0.90 * pitch - 0.04) / 2)
        brain_top = BOT - 0.19 - 0.1 * max_lines - 0.10
        fig = plt.figure(figsize=(W, H), dpi=DPI)

        def ax_in(x0, y0, w, h):                               # an axes placed in inches from the bottom left
            a = fig.add_axes([x0 / W, y0 / H, w / W, h / H]); a.axis("off"); return a

        def txt(x, y, s, **kw):
            return fig.text(x / W, y / H, s, **kw)

        ax = fig.add_axes([LB / W, BOT / H, mw / W, mh / H])
        ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(-0.5, nm - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        print(f"  neurosynth_T {tag}: {nm} maps x {nc} clusters, {W:.1f} x {H:.1f} in")
        for xi, (c, col, j) in enumerate(cols):
            tc = t[t.cluster == c].set_index("map")
            top_share = float(tc.top_patient_share.iloc[0]); bad = top_share > 0.5
            for yi, mname in enumerate(names):
                q = tc.loc[mname]; y = nm - 1 - yi
                pct = float(q.pct_inside); le = float(np.log2(max(float(q.enrichment), 1e-3)))
                face = NS_GREY if bad else cmap(norm(float(np.clip(le, -2, 2))))
                sure = bool(q.outlined) and not bad
                ax.scatter(xi, y, s=70 + 360 * min(pct, 60) / 60, c=[face], alpha=0.5 if bad else 1.0,
                           edgecolors=INK if sure else "none", linewidths=1.3 if sure else 0, zorder=3)
                ax.text(xi, y, f"{pct:.0f}", ha="center", va="center", fontsize=6.0, zorder=4,
                        color="white" if (not bad and abs(le) > 1.1) else INK)
            # the column's foot: the id, the counts, the cluster's glass brains left and right
            xc = LB + (xi + 0.5) * pitch
            n_ = int(tc.n.iloc[0]); p_ = int(npat.get(c, 0))
            head = ("✗ " if bad else "") + (f"c{c}" if tag == "15band" else f"z{c}")
            lines = ([] if tag == "15band" else [f"≈ c{j}"]) + [f"n {n_} · {p_} patients"] + ([f"{100 * top_share:.0f} % {tc.top_patient.iloc[0]}"] if bad else [])
            txt(xc, BOT - 0.05, head, ha="center", va="top", fontsize=7.4, fontweight="bold", color="#b2182b" if bad else col)
            txt(xc, BOT - 0.19, NL.join(lines), ha="center", va="top", fontsize=5.6, color=MUTED, linespacing=1.2)
            for k_, side in enumerate(("L", "R")):
                im = cluster_brain(tag, c, side, d, lambda: z_run(d, match))
                ch = cw * im.shape[0] / im.shape[1]
                ax_in(xc - cw - 0.02 + k_ * (cw + 0.04), brain_top - ch, cw, ch).imshow(im)
            print(f"    {head:6s} {' / '.join(lines)}")
        # the rows' heads: the stage, the map's name and cohort share, the map on the pial surface left and right
        coh = t[t.cluster == cols[0][0]].set_index("map").cohort_pct
        bw = 0.46; bh = None
        for yi, (stage, mname) in enumerate(maps):
            yc = BOT + (nm - 1 - yi + 0.5) * RH
            txt(1.03, yc - 0.05, mname.replace(" ", NL), ha="right", va="bottom", fontsize=7.0, color=INK, linespacing=1.05)
            txt(1.03, yc - 0.07, f"{float(coh[mname]):.0f} % of all electrodes", ha="right", va="top", fontsize=5.4, color=MUTED)
            for k_, side in enumerate(("L", "R")):
                im = ns_brain_image(mname, side)
                bh = bw * im.shape[0] / im.shape[1]
                ax_in(1.12 + k_ * (bw + 0.04), yc - bh / 2, bw, bh).imshow(im)
        for k_, side in enumerate(("left", "right")):
            txt(1.12 + bw / 2 + k_ * (bw + 0.04), BOT + mh + 0.03, side, ha="center", va="bottom", fontsize=6.0, color=MUTED)
        # the two colour bars, right of the matrix
        for i_, (nrm, cm_, ticks, labels_, title_) in enumerate((
                (norm, cmap, [-2, -1, 0, 1, 2], ["¼×", "½×", "1×", "2×", "4×"], "the share against all electrodes"),
                (matplotlib.colors.Normalize(0, NS_ZMAX), NS_CMAP, [0, 4, 8], ["0", "4", "8+"], "the map's z on the pial surface"))):
            cax = fig.add_axes([(W - RB + 0.12) / W, (BOT + mh - 0.32 - 0.6 * i_) / H, 0.9 / W, 0.08 / H])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm_), cax=cax, orientation="horizontal", ticks=ticks)
            cb.ax.set_xticklabels(labels_, fontsize=5.8); cb.ax.tick_params(length=2, pad=1.5)
            cax.set_title(title_, fontsize=6.0, color=MUTED, pad=2)
        txt(LB, H - 0.10, "15 bands, K 7  ·  the NeuroSynth maps as rows, the clusters as columns" if tag == "15band"
            else "15 bands z-scored, K 7  ·  each column in the place of the published type it matches", ha="left", va="top", fontsize=7.2, color=INK)
        txt(LB, H - 0.26, "bubble and number: % of the cluster's electrodes inside the map  ·  outlined: above the cohort in 97.5 % of patient resamples  ·  no spatial null",
            ha="left", va="top", fontsize=5.9, color=MUTED)
        txt(LB, H - 0.37, "beside each map: its z > 0 on the pial surface, left and right  ·  under each cluster: its electrodes on the glass brain at loading ≥ 0.3, left and right, colour and size by loading",
            ha="left", va="top", fontsize=5.9, color=MUTED)
        p = OUT / f"neurosynth_T_{tag}{'' if keep is None else f'_{nm}maps'}.png"
        fig.savefig(p, facecolor="white", bbox_inches="tight", pad_inches=0.04); plt.close(fig)
        out[tag] = p
        im = Image.open(p); print(f"    -> {p.name}  {im.size[0]} x {im.size[1]} px")
    return out["15band"]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="replace")
            except Exception: pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--only", nargs="+", default=["figure1", "agreement", "stability", "separation", "neurosynth", "neurosynthT"], choices=["row1", "figure1", "agreement", "stability", "separation", "neurosynth", "neurosynthT"])
    ap.add_argument("--maps", nargs="+", default=None, help="neurosynthT: only these NeuroSynth maps as rows, e.g. speech_perception visual")
    ap.add_argument("--no-render", action="store_true", help="figure 1 reuses the saved brain tiles instead of rendering (no VTK)")
    a = ap.parse_args()
    global NO_RENDER
    NO_RENDER = a.no_render
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_ref()
    steps = dict(row1=row1, figure1=row1_column, agreement=agreement, stability=stability, separation=separation, neurosynth=neurosynth,
                 neurosynthT=lambda d_: neurosynth_T(d_, keep=[m.replace('_', ' ') for m in a.maps] if a.maps else None))
    for name in a.only:
        p = steps[name](d)
        im = Image.open(p); print(f"  {name:11s} -> {p.name}  {im.size[0]} x {im.size[1]} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
