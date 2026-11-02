#!/usr/bin/env python3
"""make_cluster_cards.py - one CARD per cluster: the dB mean of its members (the paper's
draw_mean), the cluster in its own feature space, the two lateral glass brains under the
paper's rule (render_hemi: colour, size and opacity from the loading, below LOAD_CUT not
drawn), the patient composition bar, and the three most representative electrodes. Built
for the published concat_bands15 run and for every band-equalised variant that has a
convex NMF run; a variant cluster is coloured as the published type it is matched to
(Hungarian assignment on shared electrodes), so the same colour means the same type
across all the decks (2026-11-02).

    python make_cluster_cards.py                        # published K 7 + every variant with a run, at K 7
    python make_cluster_cards.py --sets concat_bands15z concat_bands15t --k 7 5

Writes outputs/clustering/feature_variants/<cache>/cards/<set>_K<k>/card_<pos>_c<id>.png,
one contact sheet <set>_K<k>_cards.png per deck, and cards.md. Reads outputs only.
VTK renders the brains: the screen must be unlocked.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
import time
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
VARIANTS = {"concat_bands15z": ("15 bands, each z-scored across the cohort", "SD within the band"),
            "concat_bands15s": ("15 bands, per-band shape within each electrode", "band-relative amplitude"),
            "concat_bands15t": ("15 bands, ternary map on the z-scored values", "agreement: share with +1 minus share with -1")}
NL = chr(10)
GREY = "#9a9a9a"


def keys_of(meta):
    return [f"{p}|{P2.norm(e)}" for p, e in zip(meta["patient_id"], meta["electrode"])]


def hungarian(lab_v, kv, lab_r, kr):
    M = np.zeros((kv, kr), int)
    for a, b in zip(lab_v, lab_r):
        M[a, b] += 1
    ri, ci = linear_sum_assignment(-M)
    sv = np.bincount(lab_v, minlength=kv); sr = np.bincount(lab_r, minlength=kr)
    J = M / np.maximum(sv[:, None] + sr[None, :] - M, 1)
    return M, J, dict(zip(ri.tolist(), ci.tolist()))


def load_set(fset, k, ref):
    """A run dict in the shape the paper's helpers read, plus the match to the published types."""
    names = P2.TYPE_NAMES[(REF, REF_K)]
    if fset == REF:
        d = P2.load_run(REF, k, "cnmf") if k != REF_K else dict(ref)
        d["fset"] = REF
        lab = d["lab"]
        if k == REF_K:
            mp = {j: j for j in range(k)}; J = np.eye(k)
        else:
            _, J, mp = hungarian(lab, k, ref["lab"], REF_K)
        return d, mp, J, d["X"]
    run = LR.newest_run("cnmf", fset)
    meta = pd.read_csv(run / "labels.csv")
    if keys_of(meta) != keys_of(ref["meta"]):
        raise SystemExit(f"{fset}: {run.name} is not on the published run's electrodes")
    gf = run / "loadings_by_k" / f"G_k{k:02d}.npy"
    if not gf.exists():
        raise SystemExit(f"{fset}: no loadings at K {k} ({gf.name} missing)")
    G = np.load(gf).astype(float)
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    lab = Gn.argmax(1)
    Xv = np.load(run / "X_train.npy").astype(float)
    d = dict(ref)                                   # xyz, hemi, patient, bands, conds, nt, meta ...
    d.update(run=run, k=k, G=G, Gn=Gn, lab=lab, fset=fset, X=ref["X"])
    _, J, mp = hungarian(lab, k, ref["lab"], REF_K)
    return d, mp, J, Xv


def representatives(Cd, Gn, lab, a, n=3):
    """Members with loading >= 0.5 (else all members) ranked by r with the members' dB mean."""
    idx = np.where(lab == a)[0]
    if not len(idx):
        return []
    mu = Cd[idx].mean(0).ravel(); mu = mu - mu.mean()
    r = np.array([np.corrcoef(Cd[i].ravel(), mu)[0, 1] for i in idx])
    strong = Gn[idx, a] >= 0.5
    pool = idx[strong] if strong.sum() >= n else idx
    rr = r[strong] if strong.sum() >= n else r
    top = np.argsort(-rr)[:n]
    return [(int(pool[t]), float(rr[t])) for t in top]


def cond_ticks(ax, d):
    nt = d["nt"]
    ax.set_xticks([(i + 0.5) * nt for i in range(len(d["conds"]))])
    ax.set_xticklabels([P2.COND_LABEL.get(c, c) for c in d["conds"]], fontsize=6.6)
    ax.tick_params(axis="x", length=0, pad=1.5)


def card(fset, k, d, a, mp, J, Cref, Xv, pcol, order_pos, out, col):
    """One cluster, one PNG. `col` is the matched published type's colour (grey if none)."""
    names = P2.TYPE_NAMES[(REF, REF_K)]
    b = mp.get(a)
    members = d["lab"] == a
    n = int(members.sum())
    pats = pd.Series(d["patient"][members]).value_counts()
    top_share = float(pats.iloc[0] / n) if n else 0.0
    own = float(np.median(d["Gn"][members, a])) if n else 0.0
    keeps = float(np.sum(members & (d["lab_ref"] == b)) / max((d["lab_ref"] == b).sum(), 1)) if b is not None else None
    M = Cref[members].mean(0) if n else np.zeros(Cref.shape[1:])

    fig = plt.figure(figsize=(13.2, 4.0), dpi=150)
    gs = GridSpec(2, 6, figure=fig, width_ratios=[2.3, 2.3, 1.55, 1.55, 0.06, 2.1], height_ratios=[1.0, 0.16],
                  left=0.035, right=0.99, top=0.80, bottom=0.08, wspace=0.14, hspace=0.05)
    # A - the dB mean of the members
    axA = fig.add_subplot(gs[0, 0])
    P2.draw_mean(axA, M, d, vlim=P2.MEAN_VLIM, col=col)
    cond_ticks(axA, d)
    axA.set_yticks(range(0, len(d["bands"]), 3)); axA.set_yticklabels([x.replace("Hz", "") for x in d["bands"][::3]], fontsize=6.4)
    axA.tick_params(labelsize=6.4, colors=P2.MUTED, length=2); axA.set_ylabel("Hz", fontsize=7)
    axA.set_title(f"A  dB mean of the {n} members (±{P2.MEAN_VLIM:.0f} dB)", fontsize=7.6, loc="left", color=P2.INK)
    # B - the cluster in its own feature space (the variants) or the unit-norm mean (published)
    axB = fig.add_subplot(gs[0, 1])
    Cv = P2.cube(Xv, d)
    Mv = Cv[members].mean(0) if n else np.zeros(Cv.shape[1:])
    vl = float(np.percentile(np.abs(Mv), 99)) or 1.0
    P2.draw_mean(axB, Mv, d, vlim=vl, col=col)
    cond_ticks(axB, d)
    axB.set_yticks([]); axB.tick_params(labelsize=6.4, colors=P2.MUTED, length=2)
    unit = VARIANTS.get(fset, ("", "dB (the fitted matrix)"))[1]
    axB.set_title(f"B  in its own feature space: {unit} (±{vl:.2g})", fontsize=7.6, loc="left", color=P2.INK)
    # C - the two lateral renders, the paper's rule
    nsel = {}
    for i_, side in enumerate(("L", "R")):
        ax = fig.add_subplot(gs[0, 2 + i_])
        img, ns = P2.render_hemi(side, d, a)
        nsel[side] = ns
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.text(0.03, 0.02, f"{side}  {ns}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color=P2.MUTED)
        if i_ == 0:
            ax.set_title(f"C  electrodes at loading ≥ {P2.LOAD_CUT:.1f}, colour and size by loading", fontsize=7.6, loc="left", color=P2.INK)
    # D - the patient bar under the brains
    axD = fig.add_subplot(gs[1, 2:4])
    P2.patient_bar(axD, d, a, pcol)
    # E - the three most representative electrodes
    reps = representatives(Cref, d["Gn"], d["lab"], a)
    sub = gs[0:2, 5].subgridspec(len(reps) or 1, 1, hspace=0.55)
    for r_, (i, rv) in enumerate(reps):
        ax = fig.add_subplot(sub[r_, 0])
        P2.draw_mean(ax, Cref[i], d, vlim=4.0, col=col)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{d['patient'][i]} {d['meta']['electrode'].iloc[i]}   r {rv:.2f}   loading {d['Gn'][i, a]:.2f}", fontsize=6.6, loc="left", color=P2.INK)
        if r_ == 0:
            ax.text(0, 1.55, "E  most representative members (±4 dB)", transform=ax.transAxes, fontsize=7.6, color=P2.INK, va="bottom")
    # the title
    head = (f"{fset}  ·  K {k}  ·  cluster {a}" if fset != REF else f"{fset}  ·  K {k}  ·  c{a} {names.get(a, '')}")
    if fset != REF:
        head += (f"   →   matched to c{b} {names.get(b, '')}   (Jaccard {J[a, b]:.2f}, keeps {100 * keeps:.0f} % of the type)" if b is not None
                 else f"   →   no published partner (closest c{int(J[a].argmax())}, Jaccard {J[a].max():.2f})")
    sub2 = (f"n {n}  ·  {len(pats)} patients  ·  largest patient {pats.index[0] if n else '-'} {100 * top_share:.0f} %"
            + ("  ·  OVER HALF ONE PATIENT" if top_share > P2.DANGER else "") + f"  ·  median own loading {own:.2f}  ·  drawn L {nsel['L']} / R {nsel['R']}")
    fig.text(0.035, 0.955, head, fontsize=10, color=P2.RED if top_share > P2.DANGER else col, fontweight="bold", va="top")
    fig.text(0.035, 0.885, sub2, fontsize=7.8, color=P2.MUTED, va="top")
    p = out / f"card_{order_pos:02d}_c{a}.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p, dict(set=fset, k=k, cluster=a, pos=order_pos, matched=(f"c{b} {names.get(b, '')}" if b is not None else None),
                   jaccard=(float(J[a, b]) if b is not None else float(J[a].max())), keeps=keeps, n=n, n_patients=int(len(pats)),
                   top_patient=(str(pats.index[0]) if n else None), top_share=top_share, median_own_loading=own,
                   representatives=[f"{d['patient'][i]} {d['meta']['electrode'].iloc[i]}" for i, _ in reps])


def contact_sheet(paths, out_png, title):
    ims = [Image.open(p).convert("RGB") for p in paths]
    W = max(im.width for im in ims); H = sum(im.height for im in ims) + 60
    sheet = Image.new("RGB", (W, H), "white")
    fig = plt.figure(figsize=(W / 150, 60 / 150), dpi=150)
    fig.text(0.01, 0.5, title, fontsize=11, color=P2.INK, va="center")
    fig.canvas.draw()
    strip = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").resize((W, 60))
    plt.close(fig)
    sheet.paste(strip, (0, 0)); y = 60
    for im in ims:
        sheet.paste(im, (0, y)); y += im.height
    sheet.save(out_png, optimize=True)
    return out_png


def deck(fset, k, ref, Cref, ref_order, out_root):
    d, mp, J, Xv = load_set(fset, k, ref)
    d["lab_ref"] = np.asarray(ref["lab"])
    out = out_root / f"{fset}_K{k}"; out.mkdir(parents=True, exist_ok=True)
    # the variant's clusters in the paper's order of the types they match; unmatched last;
    # the published run itself in its own figure order
    if fset == REF and k == REF_K:
        order = list(ref_order)
    else:
        order = [a for a, _ in sorted(mp.items(), key=lambda ab: ref_order.index(ab[1]))] + [a for a in range(k) if a not in mp]
    # colour every variant cluster as the published type it matched: render_hemi asks
    # cluster_col(j, k), so answer with the partner's colour while this deck is drawn
    orig = P2.cluster_col
    colmap = {a: orig(b, REF_K) for a, b in mp.items()}
    P2.cluster_col = lambda j, kk: colmap.get(j, (0.60, 0.60, 0.60))
    pcol = P2.patient_colours(d)
    paths, rows = [], []
    try:
        for pos, a in enumerate(order):
            p, row = card(fset, k, d, a, mp, J, Cref, Xv, pcol, pos + 1, out, colmap.get(a, (0.60, 0.60, 0.60)))
            paths.append(p); rows.append(row)
            print(f"  {fset} K {k} cluster {a}: {row['n']} electrodes, {row['n_patients']} patients, matched {row['matched']}")
    finally:
        P2.cluster_col = orig
    label = VARIANTS.get(fset, ("the published 15-band run", ""))[0]
    sheet = contact_sheet(paths, out_root / f"{fset}_K{k}_cards.png", f"{fset}  ·  K {k}  ·  {label}  ·  clusters in the order of the published types they match")
    return sheet, rows


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="replace")
            except Exception: pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=None, help="feature sets (default: the published run and every variant with a run)")
    ap.add_argument("--k", type=int, nargs="+", default=[REF_K])
    a = ap.parse_args()
    t0 = time.time()
    cache = LR.current_cache()
    out_root = ROOT / "outputs" / "clustering" / "feature_variants" / cache / "cards"
    out_root.mkdir(parents=True, exist_ok=True)
    ref = P2.load_run(REF, REF_K, "cnmf"); ref["fset"] = REF
    Cref = P2.cube(ref["X"], ref)
    ref_means = np.stack([Cref[ref["lab"] == j].mean(0) for j in range(REF_K)])
    ref_order = [int(j) for j in P2.block_order(ref_means, ref)[0]]
    sets = a.sets
    if sets is None:
        sets = [REF]
        for v in VARIANTS:
            try:
                LR.newest_run("cnmf", v); sets.append(v)
            except (FileNotFoundError, LR.StaleRunError):
                print(f"{v}: no run yet, skipped")
    allrows, sheets = [], []
    for fset in sets:
        for k in a.k:
            try:
                sheet, rows = deck(fset, k, ref, Cref, ref_order, out_root)
            except SystemExit as e:
                print(f"{fset} K {k}: {e}"); continue
            sheets.append(sheet); allrows.extend(rows)
    L = [f"# Cluster cards on cache {cache}", "", f"*Built {time.strftime('%Y-%m-%d %H:%M')}. One card per cluster; a variant cluster carries the colour of the published type it is matched to.*", ""]
    for s_ in sheets:
        L += [f"![{s_.stem}]({s_.name})", ""]
    L += ["| set | K | pos | cluster | matched type | Jaccard | keeps | n | patients | largest patient | share | median own loading | representatives |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        keeps_txt = "" if r["keeps"] is None else f"{100 * r['keeps']:.0f} %"
        reps_txt = ", ".join(r["representatives"])
        L.append(f"| {r['set']} | {r['k']} | {r['pos']} | {r['cluster']} | {r['matched'] or '-'} | {r['jaccard']:.2f} | {keeps_txt} | {r['n']} | {r['n_patients']} | {r['top_patient']} | {100 * r['top_share']:.0f} % | {r['median_own_loading']:.2f} | {reps_txt} |")
    (out_root / "cards.md").write_text(NL.join(L) + NL, encoding="utf-8")
    (out_root / "cards.json").write_text(json.dumps(allrows, indent=1, default=str), encoding="utf-8")
    print(f"{len(allrows)} cards, {len(sheets)} decks -> {out_root}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
