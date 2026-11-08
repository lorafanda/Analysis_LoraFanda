#!/usr/bin/env python3
"""make_poster_blocks.py - the seven response types as poster blocks, for the published 15-band
run and for the z-scored variant matched to it (2026-11-03).

One block per type: the mean ERSP of its members (the paper's draw_mean, +-2 dB, three
conditions side by side, width : height 3 : 1, no frequency values: the poster's example figure carries
that axis) framed in the type's colour, the glass brains (render_hemi: colour, size and opacity by
loading); ONE rule for every part of the block: the members with own loading >= --min-loading (0.35) as left | top | right (--views L R for the two lateral
views only), the type's name on one line and one caption line (n, patients). The z-scored run's clusters are ordered and coloured by the
published type they are matched to (Hungarian on shared electrodes) and captioned "v<id> -> type".

    python make_poster_blocks.py                      # both runs
    python make_poster_blocks.py --sets concat_bands15

Writes outputs/clustering/paper_figures/poster/<set>/block_<pos>_c<id>.png (300 dpi, 3.6 x 4.8 in)
and <set>_strip.png (the seven blocks in a row). VTK renders the brains: screen unlocked.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("p2", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(P2)
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                      # noqa: E402

REF, REF_K = "concat_bands15", 7
OUT = ROOT / "outputs" / "clustering" / "paper_figures" / "poster"
NL = chr(10)
GREY = (0.60, 0.60, 0.60)
MIN_LOAD = 0.35          # one rule for the whole block: members (argmax) with own loading >= this (main() sets it)


def members_of(d, a):
    """The cluster's members whose own loading is at or above MIN_LOAD."""
    return (d["lab"] == a) & (d["Gn"][:, a] >= MIN_LOAD)


def keys_of(meta):
    return [f"{p}|{P2.norm(e)}" for p, e in zip(meta["patient_id"], meta["electrode"])]


def hungarian(lab_v, kv, lab_r, kr):
    M = np.zeros((kv, kr), int)
    for a, b in zip(lab_v, lab_r):
        M[a, b] += 1
    ri, ci = linear_sum_assignment(-M)
    sv = np.bincount(lab_v, minlength=kv); sr = np.bincount(lab_r, minlength=kr)
    J = M / np.maximum(sv[:, None] + sr[None, :] - M, 1)
    return J, dict(zip(ri.tolist(), ci.tolist()))


def load(fset, ref):
    """The run dict the paper's helpers read, its figure order and its match to the published types."""
    names = P2.TYPE_NAMES[(REF, REF_K)]
    if fset == REF:
        d = dict(ref); d["fset"] = REF
        C = P2.cube(d["X"], d)
        means = np.stack([C[d["lab"] == j].mean(0) for j in range(REF_K)])
        order = [int(j) for j in P2.block_order(means, d)[0]]
        return d, order, {j: (j, 1.0) for j in range(REF_K)}, names
    run = LR.newest_run("cnmf", fset)
    meta = pd.read_csv(run / "labels.csv")
    if keys_of(meta) != keys_of(ref["meta"]):
        raise SystemExit(f"{fset}: not the published run's electrodes")
    G = np.load(run / "loadings_by_k" / f"G_k{REF_K:02d}.npy").astype(float)
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    lab = Gn.argmax(1)
    d = dict(ref); d.update(run=run, k=REF_K, G=G, Gn=Gn, lab=lab, fset=fset)
    J, mp = hungarian(lab, REF_K, ref["lab"], REF_K)
    C = P2.cube(ref["X"], ref)
    rmeans = np.stack([C[ref["lab"] == j].mean(0) for j in range(REF_K)])
    rorder = [int(j) for j in P2.block_order(rmeans, ref)[0]]
    order = sorted(mp, key=lambda a: rorder.index(mp[a])) + [a for a in range(REF_K) if a not in mp]
    return d, order, {a: (mp.get(a), float(J[a, mp[a]]) if a in mp else float(J[a].max())) for a in range(REF_K)}, names


def patients_bar(ax, d, a, pcol):
    """Who the cluster is made of: one segment per patient (FIG 1's colours), widest first, width = that patient's
    electrodes in the cluster; the count in each segment wide enough to hold it."""
    vc = pd.Series(d["patient"][members_of(d, a)]).value_counts()
    n = int(vc.sum())
    x = 0.0
    for p_, c_ in vc.items():
        ax.barh(0, c_ / n, left=x, height=1.0, color=pcol[p_], lw=0.4, ec="white")
        if c_ / n >= 0.055:
            rgb = np.asarray(matplotlib.colors.to_rgb(pcol[p_]))
            ax.text(x + c_ / n / 2, 0, str(int(c_)), ha="center", va="center", fontsize=5.5,
                    color="white" if rgb @ [0.299, 0.587, 0.114] < 0.6 else P2.INK)
        x += c_ / n
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_alpha(0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.text(0.0, -0.75, f"{len(vc)} patients · electrodes per patient", transform=ax.get_xaxis_transform(),
            fontsize=6, color=P2.MUTED, ha="left", va="top")
    top = float(vc.iloc[0]) / n
    ax.text(1.0, -0.75, f"largest {100 * top:.0f}%", transform=ax.get_xaxis_transform(), fontsize=6,
            color=P2.MUTED, ha="right", va="top")


def block(d, a, b, jac, names, Cref, col, out, pos, fset, aspect=3.0, views=("L", "T", "R"), pcol=None, center=False):
    """One type as a poster block: framed mean ERSP (width:height = aspect), the brains in `views`, a caption line;
    with `pcol` (patient colours) a patient bar under the brains, written as ..._patients.png."""
    members = members_of(d, a)
    n = int(members.sum()); npat = int(pd.Series(d["patient"][members]).nunique())
    M = Cref[members].mean(0)
    w = 3.6; h_ersp = 0.92 * w / aspect; h_brain = (0.36 if "T" in views else 0.42) * w; head = 0.5
    h_pat = 0.34 if pcol is not None else 0.0                      # the patient bar and its line of text
    H = h_ersp + h_brain + head + h_pat
    fig = plt.figure(figsize=(w, H), dpi=300)
    fig.patch.set_alpha(0)
    gs = GridSpec(2, len(views), figure=fig, height_ratios=[h_ersp, h_brain],
                  width_ratios=[0.62 if v == "T" else 1.0 for v in views], left=0.04, right=0.96, top=1 - head / H,
                  bottom=(0.01 + h_pat) / H, hspace=0.12 if "T" in views else 0.16, wspace=0.02)
    ax = fig.add_subplot(gs[0, :])
    P2.draw_mean(ax, M, d, vlim=P2.MEAN_VLIM, col=col)
    nt = d["nt"]
    ax.set_xticks([(i + 0.5) * nt for i in range(len(d["conds"]))]); ax.set_xticklabels([P2.COND_LABEL.get(c, c) for c in d["conds"]], fontsize=7)
    ax.set_yticks([])                                              # the frequency axis is on the example figure only
    ax.tick_params(length=0, pad=1.5, colors=P2.MUTED)
    for s_ in ax.spines.values():
        s_.set_edgecolor(col); s_.set_linewidth(1.2)
    for i_, side in enumerate(views):
        axb = fig.add_subplot(gs[1, i_])
        d2 = dict(d)                                               # only the members at the cut are drawn
        G = d["Gn"].copy()
        G[~members, a] = 0.0
        d2["Gn"] = G
        cut0 = P2.LOAD_CUT
        P2.LOAD_CUT = MIN_LOAD
        try:
            img, _ = P2.render_hemi(side, d2, a)
        finally:
            P2.LOAD_CUT = cut0
        nsel = int((members & (d["hemi"] == side)).sum()) if side != "T" else n
        axb.imshow(img); axb.set_xticks([]); axb.set_yticks([])
        for s_ in axb.spines.values():
            s_.set_visible(False)
        if side != "T":                                            # the top view shows both hemispheres
            axb.text(0.03, 0.02, f"{side} {nsel}", transform=axb.transAxes, fontsize=6, color=P2.MUTED, ha="left", va="bottom")
    name = names.get(b, "") if b is not None else "no published partner"
    import textwrap
    # the type's name alone as the title (bigger); its id with n and patients on the line under it (11-08)
    title = name if fset == REF else f"→ {name}" + (f"  (J {jac:.2f})" if b is not None else "")
    fs_ = 11.5 if len(title) <= 30 else max(8.5, 11.5 * 30 / len(title))   # one line, smaller when long
    tx, ha = (0.5, "center") if center else (0.04, "left")         # centred version: title and subtitle centred
    fig.text(tx, 1 - 0.05 / H, title, fontsize=fs_, fontweight="bold", color=col, va="top", ha=ha)
    fig.text(tx, 1 - 0.29 / H, f"{'c' if fset == REF else 'v'}{a}  ·  n {n}  ·  {npat} patients  ·  loading ≥ {MIN_LOAD:g}", fontsize=7.5,
             color=P2.MUTED, va="top", ha=ha)
    if pcol is not None:
        patients_bar(fig.add_axes([0.04, (0.17 + 0.0) / H, 0.92, 0.13 / H]), d, a, pcol)
    for ax_ in fig.axes:
        ax_.patch.set_alpha(0)
    p = out / f"block_{pos:02d}_c{a}{'_patients' if pcol is not None else ''}{'_centered' if center else ''}.png"
    fig.savefig(p, transparent=True); plt.close(fig)
    return p


def strip(paths, out_png, rows=1):
    """The blocks side by side, or in `rows` rows (4 + 3 for seven), each row left-aligned."""
    ims = [Image.open(p).convert("RGBA") for p in paths]
    per = int(np.ceil(len(ims) / rows))
    h = max(im.height for im in ims); w = max(im.width for im in ims)
    gap = int(0.12 * h) if rows > 1 else 0                          # air between the rows of the grid
    sheet = Image.new("RGBA", (w * per, h * rows + gap * (rows - 1)), (255, 255, 255, 0))     # transparent
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % per) * w, (i // per) * (h + gap)))
    sheet.save(out_png, optimize=True)
    return out_png


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="replace")
            except Exception: pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=[REF, "concat_bands15z"])
    ap.add_argument("--aspect", type=float, default=3.0, help="ERSP panel width : height (the poster: 3)")
    ap.add_argument("--views", nargs="+", default=["L", "T", "R"], choices=["L", "T", "R"],
                    help="the brains under the centroid: L R = the two lateral views, L T R adds the top view between")
    ap.add_argument("--min-loading", type=float, default=0.35,
                    help="the block's electrodes: members with own loading >= this (0 = every member)")
    a = ap.parse_args()
    globals()["MIN_LOAD"] = a.min_loading
    ref = P2.load_run(REF, REF_K, "cnmf"); ref["fset"] = REF
    Cref = P2.cube(ref["X"], ref)
    for fset in a.sets:
        d, order, match, names = load(fset, ref)
        out = OUT / fset; out.mkdir(parents=True, exist_ok=True)
        orig = P2.cluster_col
        rorder = [int(j) for j in P2.block_order(np.stack([Cref[ref["lab"] == j].mean(0) for j in range(REF_K)]), ref)[0]]
        colmap = {a_: (orig(match[a_][0], REF_K) if match[a_][0] is not None else GREY) for a_ in range(REF_K)}
        P2.cluster_col = lambda j, kk: colmap.get(j, GREY)
        V = {suf: [] for suf in ("", "_patients", "_centered", "_patients_centered")}
        pcol = P2.patient_colours(d)
        try:
            for pos, a_ in enumerate(order, 1):
                b, jac = match[a_]
                for suf in V:                                       # plain / patient bar, left / centred writing
                    V[suf].append(block(d, a_, b, jac, names, Cref, colmap[a_], out, pos, fset, aspect=a.aspect,
                                        views=tuple(a.views), pcol=pcol if "patients" in suf else None,
                                        center="centered" in suf))
                print(f"  {fset} block {pos}: cluster {a_} -> {names.get(b, '-') if b is not None else '-'}")
        finally:
            P2.cluster_col = orig
        for suf, ps_ in V.items():                                      # the strip (one row) and the grid (4 + 3)
            for rows, kind in ((1, "strip"), (2, "grid")):
                s = strip(ps_, OUT / f"{fset}_{kind}{suf}.png", rows=rows)
                print(f"  {s.name}: {Image.open(s).size[0]} x {Image.open(s).size[1]} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
