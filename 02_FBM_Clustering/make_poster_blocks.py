#!/usr/bin/env python3
"""make_poster_blocks.py - the seven response types as poster blocks, for the published 15-band
run and for the z-scored variant matched to it (2026-11-03).

One block per type: the mean ERSP of its members (the paper's draw_mean, +-2 dB, three
conditions side by side) framed in the type's colour, the two lateral glass brains under the
paper's rule (render_hemi: loading >= 0.3 drawn, colour, size and opacity by loading), and one
caption line (type, n, patients). The z-scored run's clusters are ordered and coloured by the
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


def block(d, a, b, jac, names, Cref, col, out, pos, fset, aspect=2.0):
    """One type as a poster block: framed mean ERSP (width:height = aspect), two brains, a caption line."""
    members = d["lab"] == a
    n = int(members.sum()); npat = int(pd.Series(d["patient"][members]).nunique())
    M = Cref[members].mean(0)
    w = 3.6; h_ersp = 0.88 * w / aspect; h_brain = 0.42 * w
    fig = plt.figure(figsize=(w, h_ersp + h_brain + 0.75), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[h_ersp, h_brain], left=0.06, right=0.94, top=1 - 0.55 / (h_ersp + h_brain + 0.75), bottom=0.02, hspace=0.16, wspace=0.03)
    ax = fig.add_subplot(gs[0, :])
    P2.draw_mean(ax, M, d, vlim=P2.MEAN_VLIM, col=col)
    nt = d["nt"]
    ax.set_xticks([(i + 0.5) * nt for i in range(len(d["conds"]))]); ax.set_xticklabels([P2.COND_LABEL.get(c, c) for c in d["conds"]], fontsize=7)
    ax.set_yticks([0, 4, 7, 9, 14]); ax.set_yticklabels([d["bands"][i].replace("Hz", "") for i in (0, 4, 7, 9, 14)], fontsize=6)
    ax.tick_params(length=0, pad=1.5, colors=P2.MUTED)
    for s_ in ax.spines.values():
        s_.set_edgecolor(col); s_.set_linewidth(2.4)
    for i_, side in enumerate(("L", "R")):
        axb = fig.add_subplot(gs[1, i_])
        img, nsel = P2.render_hemi(side, d, a)
        axb.imshow(img); axb.set_xticks([]); axb.set_yticks([])
        for s_ in axb.spines.values():
            s_.set_visible(False)
        axb.text(0.03, 0.02, f"{side} {nsel}", transform=axb.transAxes, fontsize=6, color=P2.MUTED, ha="left", va="bottom")
    name = names.get(b, "") if b is not None else "no published partner"
    import textwrap
    head = (f"c{a}  {name}" if fset == REF else f"v{a} → {name}" + (f"  (J {jac:.2f})" if b is not None else ""))
    lines = textwrap.wrap(head, 30)[:2]
    fig.text(0.06, 0.975, NL.join(lines), fontsize=9 if len(lines) == 1 else 8.2, fontweight="bold", color=col, va="top", linespacing=1.15)
    fig.text(0.06, 0.975 - (0.045 if len(lines) == 1 else 0.075), f"n {n}  ·  {npat} patients", fontsize=7.5, color=P2.MUTED, va="top")
    p = out / f"block_{pos:02d}_c{a}.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def strip(paths, out_png, rows=1):
    """The blocks side by side, or in `rows` rows (4 + 3 for seven), each row left-aligned."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    per = int(np.ceil(len(ims) / rows))
    h = max(im.height for im in ims); w = max(im.width for im in ims)
    sheet = Image.new("RGB", (w * per, h * rows), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % per) * w, (i // per) * h))
    sheet.save(out_png, optimize=True)
    return out_png


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="replace")
            except Exception: pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=[REF, "concat_bands15z"])
    ap.add_argument("--aspect", type=float, default=2.0, help="ERSP panel width : height (the poster asks 3)")
    ap.add_argument("--rows", type=int, default=1, help="rows of blocks in the sheet (2 = 4 + 3)")
    a = ap.parse_args()
    ref = P2.load_run(REF, REF_K, "cnmf"); ref["fset"] = REF
    Cref = P2.cube(ref["X"], ref)
    for fset in a.sets:
        d, order, match, names = load(fset, ref)
        out = OUT / fset; out.mkdir(parents=True, exist_ok=True)
        orig = P2.cluster_col
        rorder = [int(j) for j in P2.block_order(np.stack([Cref[ref["lab"] == j].mean(0) for j in range(REF_K)]), ref)[0]]
        colmap = {a_: (orig(match[a_][0], REF_K) if match[a_][0] is not None else GREY) for a_ in range(REF_K)}
        P2.cluster_col = lambda j, kk: colmap.get(j, GREY)
        paths = []
        try:
            for pos, a_ in enumerate(order, 1):
                b, jac = match[a_]
                paths.append(block(d, a_, b, jac, names, Cref, colmap[a_], out, pos, fset, aspect=a.aspect))
                print(f"  {fset} block {pos}: cluster {a_} -> {names.get(b, '-') if b is not None else '-'}")
        finally:
            P2.cluster_col = orig
        s = strip(paths, OUT / (f"{fset}_grid.png" if a.rows > 1 else f"{fset}_strip.png"), rows=a.rows)
        print(f"  {s.name}: {Image.open(s).size[0]} x {Image.open(s).size[1]} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
