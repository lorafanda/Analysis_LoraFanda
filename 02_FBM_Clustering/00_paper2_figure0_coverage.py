#!/usr/bin/env python3
"""
00_paper2_figure0_coverage.py - FIG 0, the cohort: what the gate saw and what it kept,
where those electrodes are, how much of them the atlas covers, and what the gate is
deciding at the line.

    python 00_paper2_figure0_coverage.py                      the cohort every figure uses
    python 00_paper2_figure0_coverage.py --k 8 --feature-set concat_hg

    A   electrodes per patient: the total the gate SAW (hatched) and the number that
        came THROUGH it (solid), each patient in the colour FIG 1 draws it in, labelled
        kept/total
    B   the same electrodes on the brain - left, from above, right: kept in the
        patient's colour, rejected by the gate in grey
    C   of the kept electrodes, the ones with a LanA value (solid) against all kept
        (hatched), per patient, labelled with/kept
    D   the same on the brain: kept electrodes with a LanA value in orange, without one
        in grey
    E   an electrode that JUST passed the gate: its three conditions concatenated, the
        bins the criterion counts outlined, and the count that decided it
    F   one that JUST failed - on the same shaft as E where such a pair exists, so the
        two differ in nothing but that count

WHAT 'TOTAL' MEANS. The electrodes the gate actually saw: the ungated table for the
cohort's patients, minus what lf_concat removes BEFORE the gate - non-neural channels,
grid and microelectrode contacts, excluded patients - and minus electrodes missing a
condition, which the gate never sees whole. Those filters are IMPORTED from the
pipeline rather than re-described, and the result is checked: the total must split
exactly into the cohort and the gate's rejects, or the figure stops.

THE GATE, read from the cache's own params.json rather than retyped. For each
electrode-condition ERSP (n_freq x n_time bins, dB re the pre-stimulus baseline), the
fraction of bins above thr_pos must reach min_prop_pos, OR the fraction below thr_neg
must reach min_prop_neg. An electrode is kept if ANY of its three conditions passes.
The rule is re-applied to the ungated table and must reproduce the stored flag on every
row.

NO CAPTION IS DRAWN ON THE FIGURE - it gets <name>_caption.txt, _patients.csv and
_examples.csv beside it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))


def _load(name, fname):
    """A module whose file name starts with a digit, loaded by path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The cohort, its patient colours, the brain scenes and the writers come from FIG 1;
# the atlas join from FIG 3. Nothing is reimplemented, so this figure cannot describe a
# different cohort, colour a patient differently, or draw a different brain from the
# figures that follow it.
P2 = _load("p2fig1", "00_Paper2_Figures.py")
F3 = _load("p2fig3", "00_paper2_figure3_lana.py")

CLUST, OUT = P2.CLUST, P2.OUT
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
RED, GREEN = P2.RED, P2.GREEN
ORANGE = "#e08214"
CACHE_DIR = ROOT / "outputs" / "_dataset" / "concat_source_v4"
VLIM = 7.0          # the project's own ERSP colour scale, dB
FMAX = 400.0        # Hz, the top of the cube the gate counts over
FATES = ("kept", "failed the gate", "grid contact", "microelectrode",
         "non-neural channel", "missing a condition", "patient not in the cohort")


# ---- the gate ------------------------------------------------------------------
def gate_params():
    """The gate as the cache that applied it recorded it."""
    p = json.loads((CACHE_DIR / "params.json").read_text())
    return dict(thr_pos=float(p["thr_pos"]), min_pos=float(p["min_prop_pos"]),
                thr_neg=float(p["thr_neg"]), min_neg=float(p["min_prop_neg"]),
                n_freq=int(p["n_freq"]), n_time=int(p["n_time"]),
                conds=list(p["conditions"]))


def shaft_of(e):
    """AHL10 -> AHL: the physical electrode, which is the unit a 'pair' lives on."""
    return re.sub(r"\d+$", "", P2.norm(e))


def load_gate(gp):
    """The UNGATED table, one row per electrode-condition, with the gate re-derived.

    margin = max(prop_pos / min_pos, prop_neg / min_neg), so margin >= 1 IS the pass
    rule and the two criteria become one number. It must reproduce the stored flag on
    every row, or the table and the rule have drifted apart and nothing below is safe.
    """
    t = pd.read_parquet(CACHE_DIR / "df_meta.parquet")
    t["margin"] = np.maximum(t.prop_above_pos / gp["min_pos"],
                             t.prop_below_neg / gp["min_neg"])
    if not ((t.margin >= 1.0) == t.high_activity.astype(bool)).all():
        raise SystemExit("the margin rule does not reproduce the stored high_activity "
                         "flag - the gate in params.json is not the one in the table")
    t["key"] = [f"{p}|{P2.norm(e)}" for p, e in zip(t.patient_id, t.electrode)]
    t["shaft"] = [f"{p}|{shaft_of(e)}" for p, e in zip(t.patient_id, t.electrode)]
    return t


def per_electrode(t):
    """One row per electrode: how many conditions it has, how many pass, and the best."""
    return (t.groupby(["patient_id", "electrode", "key", "shaft"])
             .agg(n_high=("high_activity", "sum"), ncond=("condition", "nunique"),
                  best=("margin", "max"))
             .reset_index())


def cohort_keys(d):
    return {f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"].patient_id, d["meta"].electrode)}


def fate(g, d, gp):
    """Every electrode of the table, by what happened to it - the pipeline's own
    filters, applied in the order lf_concat applies them, then the gate."""
    from lf_concat import DEFAULT_EXCLUDE_PATIENTS
    from lf_dataset import is_grid_electrode, is_micro_electrode, is_non_neural_electrode
    keys = cohort_keys(d)
    pats = set(d["patient"])
    kept = g.key.isin(keys)
    grid = np.array([bool(is_grid_electrode(e, p)) for e, p in zip(g.electrode, g.patient_id)])
    micro = np.array([bool(is_micro_electrode(e, p)) for e, p in zip(g.electrode, g.patient_id)])
    nonn = np.array([bool(is_non_neural_electrode(e)) for e in g.electrode])
    excluded = g.patient_id.isin(set(DEFAULT_EXCLUDE_PATIENTS)) | ~g.patient_id.isin(pats)
    why = np.select(
        [kept, excluded, g.ncond < len(gp["conds"]), nonn, grid, micro, g.n_high == 0],
        ["kept", "patient not in the cohort", "missing a condition", "non-neural channel",
         "grid contact", "microelectrode", "failed the gate"], "UNEXPLAINED")
    g = g.copy()
    g["fate"] = why
    g["kept"] = kept
    if (why == "UNEXPLAINED").any():
        bad = g[why == "UNEXPLAINED"][["patient_id", "electrode"]].head(5)
        raise SystemExit("electrodes pass the gate, are not in the cohort, and no "
                         f"pipeline filter explains it:\n{bad.to_string(index=False)}")
    return g


def universe(g):
    """What the gate SAW: kept plus rejected-by-the-gate, nothing else."""
    u = g[g.fate.isin(["kept", "failed the gate"])].copy()
    if not (u[u.kept].n_high >= 1).all() or (u[~u.kept].n_high >= 1).any():
        raise SystemExit("the electrodes the gate saw do not split into the cohort and "
                         "the gate's rejects - the cohort was not gated this way")
    return u


def coords(keys):
    """fsaverage positions for any electrode keys, from the table FIG 1 uses."""
    co = pd.read_csv(P2.COORDS)
    co["key"] = [f"{p}|{P2.norm(n_)}" for p, n_ in zip(co["patient"], co["name"])]
    j = (pd.DataFrame({"key": list(keys)})
         .merge(co[["key", "x", "y", "z", "hemi"]].drop_duplicates("key"), on="key",
                how="left"))
    xyz = j[["x", "y", "z"]].to_numpy(float)
    hemi = j["hemi"].astype(object).to_numpy()
    miss = pd.isna(hemi) & np.isfinite(xyz[:, 0])
    hemi = np.where(miss, np.where(xyz[:, 0] < 0, "L", "R"), hemi)
    return xyz, hemi


def pick_examples(u):
    """The electrode just inside the gate and the one just outside it.

    Both come from what the gate saw with every condition present, so the gate is the
    only thing that separates them from their neighbours. Where the two can be found on
    the SAME shaft they are, closest pair first, because then nothing but the count
    differs between them.
    """
    inside, outside = u[u.kept], u[~u.kept]
    if outside.empty:
        raise SystemExit("no electrode failed the gate - nothing to show on the far side")
    ins, out = inside.nsmallest(60, "best"), outside.nlargest(60, "best")
    pairs = ins.merge(out, on="shaft", suffixes=("_in", "_out"))
    if len(pairs):
        pairs["gap"] = pairs.best_in - pairs.best_out
        p = pairs.sort_values(["gap", "best_in"]).iloc[0]
        a = inside[inside.key == p.key_in].iloc[0]
        b = outside[outside.key == p.key_out].iloc[0]
        how = "same shaft"
    else:
        a = inside.nsmallest(1, "best").iloc[0]
        b = outside.nlargest(1, "best").iloc[0]
        how = "different shafts - no pair straddles the line on one shaft"
    return a, b, how


# ---- rendering -----------------------------------------------------------------
def _fresh_scenes():
    """The scene cache keeps the electrode mask of the FIRST set it saw per view, so it
    is emptied whenever the electrode set changes."""
    for pl, _ in P2._PLOTTER.values():
        pl.close()
    P2._PLOTTER.clear()


def render_set(side, xyz, hemi, rgba, radius):
    """One view of one electrode set on FIG 1's brain scene."""
    pl, ok = P2._scene(side, dict(xyz=xyz, hemi=hemi), 1.30)
    actor = None
    if ok.sum():
        cloud = P2.pv.PolyData(xyz[ok])
        cloud["rgba"] = rgba[ok]
        cloud["r"] = radius[ok]
        g_ = cloud.glyph(orient=False, scale="r",
                         geom=P2.pv.Sphere(radius=1.0, theta_resolution=12,
                                           phi_resolution=12))
        actor = pl.add_mesh(g_, scalars="rgba", rgba=True)
    img = pl.screenshot(return_img=True, transparent_background=True)
    if actor is not None:
        pl.remove_actor(actor)
    return P2._crop_alpha(np.asarray(img)), int(ok.sum())


def panel_brains(axes, xyz, hemi, rgba, radius, key):
    """Left, from above, right; the count drawn under each; the key on the first."""
    _fresh_scenes()
    n = {}
    for ax, side, tag in zip(axes, ("L", "T", "R"), ("left", "from above", "right")):
        img, n[side] = render_set(side, xyz, hemi, rgba, radius)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.text(0.02, 0.02, f"{tag}  {n[side]}", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=7.0, color=MUTED)
    axes[0].text(0.02, 0.98, key, transform=axes[0].transAxes, ha="left", va="top",
                 fontsize=7.4, color=MUTED, linespacing=1.4)
    return n


# ---- panels ---------------------------------------------------------------------
def panel_bars(ax, order, total, part, pcol, ylabel, none_label="none"):
    """Hatched = the total, solid = the part, labelled part/total in the patient's
    colour - the reference style. Same x for every bars panel of the figure."""
    x = np.arange(len(order))
    top = float(max(total)) * 1.18
    for i, p in enumerate(order):
        col = pcol[p]
        ax.bar(i, total[i], width=0.78, facecolor=to_rgba(col, 0.18), edgecolor=col,
               hatch="///", lw=0.9)
        ax.bar(i, part[i], width=0.78, color=col, lw=0)
        if part[i] == 0:
            ax.text(i, total[i] + top * 0.012, f"0/{total[i]}", ha="center", va="bottom",
                    fontsize=6.6, color=RED, fontweight="bold")
        else:
            ax.text(i, total[i] + top * 0.012, f"{part[i]}/{total[i]}", ha="center",
                    va="bottom", fontsize=6.6, color=col)
    ax.set_ylabel(ylabel, fontsize=8.6)
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7.0)
    ax.tick_params(labelsize=7.4, colors=MUTED, length=2)
    ax.spines[["top", "right"]].set_visible(False)


def panel_example(axS, ax, t, row, gp, passed):
    """One electrode: its conditions side by side on the warped axis FIG 1 uses, the
    bins the criterion counts OUTLINED (solid above thr_pos, dotted below thr_neg),
    the GO cue dashed at the middle of each block, and under each block the count
    against the count it needed."""
    conds = gp["conds"]
    rows = {c: t[(t.patient_id == row.patient_id) & (t.electrode == row.electrode)
                 & (t.condition == c)].iloc[0] for c in conds}
    arrs = [np.load(rows[c].file_path) for c in conds]
    nf, nt = arrs[0].shape
    if any(a.shape != (nf, nt) for a in arrs):
        raise SystemExit(f"{row.patient_id} {row.electrode}: conditions differ in shape")
    A = np.concatenate(arrs, axis=1)
    x = np.linspace(0, len(conds), A.shape[1])
    f = np.linspace(0, FMAX, nf)
    ax.pcolormesh(x, f, A, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, shading="auto",
                  rasterized=True)
    need_p, need_n = round(gp["min_pos"] * nf * nt), round(gp["min_neg"] * nf * nt)
    detail = []
    for b, c in enumerate(conds):
        xs, a = x[b * nt:(b + 1) * nt], arrs[b]
        ax.contour(xs, f, (a > gp["thr_pos"]).astype(float), levels=[0.5],
                   colors=[INK], linewidths=0.5)
        if (a < gp["thr_neg"]).any():
            ax.contour(xs, f, (a < gp["thr_neg"]).astype(float), levels=[0.5],
                       colors=[INK], linewidths=0.5, linestyles="dotted")
        ax.axvline(b + 0.5, color=INK, lw=0.8, ls=(0, (4, 3)))
        if b:
            ax.axvline(b, color=INK, lw=1.2)
        r = rows[c]
        n_pos = round(float(r.prop_above_pos) * nf * nt)
        n_neg = round(float(r.prop_below_neg) * nf * nt)
        ok = bool(r.high_activity)
        ax.text(b + 0.5, -0.04,
                f"above {gp['thr_pos']:+.1f} dB   {n_pos} of {need_p} bins\n"
                f"below {gp['thr_neg']:+.1f} dB   {n_neg} of {need_n}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.9, color=GREEN if ok else MUTED, linespacing=1.4,
                fontweight="bold" if ok else "normal")
        detail.append(dict(condition=c, bins_pos=n_pos, need_pos=need_p,
                           bins_neg=n_neg, need_neg=need_n, passes=ok,
                           margin=float(r.margin)))
    ax.set_xlim(0, len(conds)); ax.set_ylim(0, FMAX)
    ax.set_xticks([])
    ax.set_ylabel("Hz", fontsize=7.6)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    col = GREEN if passed else RED
    for s_ in ax.spines.values():
        s_.set_color(col); s_.set_linewidth(1.6)
    P2.trial_strip(axS, dict(conds=conds))

    best = max(detail, key=lambda q: q["margin"])
    by_pos = best["bins_pos"] / need_p >= best["bins_neg"] / need_n
    n, need = (best["bins_pos"], need_p) if by_pos else (best["bins_neg"], need_n)
    thr = f"{gp['thr_pos']:+.1f}" if by_pos else f"{gp['thr_neg']:+.1f}"
    who = f"{row.patient_id} · {row.electrode}"
    if passed:
        over = n - need
        verdict = (f"KEPT  ·  {who}  ·  {best['condition']}: {n} bins "
                   f"{'above' if by_pos else 'below'} {thr} dB, "
                   + (f"exactly the {need} needed" if over == 0
                      else f"{over} over the {need} needed"))
    else:
        verdict = (f"DISCARDED  ·  {who}  ·  best is {best['condition']}: {n} bins "
                   f"{'above' if by_pos else 'below'} {thr} dB, {need - n} short of {need}")
    return dict(who=who, patient=row.patient_id, electrode=row.electrode,
                verdict=verdict, detail=detail, n_bins=nf * nt)


# ---- the figure -------------------------------------------------------------------
def figure_0(fset, k):
    t0 = time.time()
    gp = gate_params()
    d = F3.load(fset, k)                       # the cohort, with P_lana joined
    pcol = P2.patient_colours(d)
    t = load_gate(gp)
    g = fate(per_electrode(t), d, gp)
    u = universe(g)
    a, b, how = pick_examples(u)
    counts = g.fate.value_counts()
    print("  " + "   ".join(f"{f_} {int(counts.get(f_, 0))}" for f_ in FATES))
    print(f"  at the line: {a.patient_id} {a.electrode} (margin {a.best:.4f}) vs "
          f"{b.patient_id} {b.electrode} ({b.best:.4f}) - {how}")

    # per patient: total seen, kept, kept with a LanA value - in patient order
    order = sorted(set(d["patient"]))
    tot = u.groupby("patient_id").size().reindex(order).fillna(0).astype(int).to_numpy()
    kep = u[u.kept].groupby("patient_id").size().reindex(order).fillna(0).astype(int).to_numpy()
    pats = pd.Series(d["patient"])
    lan = pats[np.asarray(d["has"])].value_counts().reindex(order).fillna(0).astype(int).to_numpy()
    if not (kep == pats.value_counts().reindex(order).fillna(0).to_numpy()).all():
        raise SystemExit("the kept count per patient differs from the cohort's")
    tab = pd.DataFrame(dict(patient=order, n_total=tot, n_gated=kep, n_lana=lan,
                            gate_rate=(kep / np.maximum(tot, 1)).round(4),
                            lana_coverage=(lan / np.maximum(kep, 1)).round(4)))

    # coordinates: the universe for B, the cohort for D
    uxyz, uhemi = coords(u.key.to_list())
    u_kept = u.kept.to_numpy()
    rgba_u = np.empty((len(u), 4), np.uint8)
    grey = np.clip(255 * P2.BG, 0, 255).astype(np.uint8)
    for i, (p, kp) in enumerate(zip(u.patient_id, u_kept)):
        rgba_u[i, :3] = (np.clip(255 * np.array(pcol[p]), 0, 255).astype(np.uint8)
                         if kp else grey)
        rgba_u[i, 3] = 255 if kp else 150
    rad_u = np.where(u_kept, P2.BG_RADIUS * 1.25, P2.BG_RADIUS * 0.95)
    has = np.asarray(d["has"])
    rgba_c = np.empty((len(has), 4), np.uint8)
    rgba_c[:, :3] = np.where(has[:, None],
                             np.clip(255 * np.array(to_rgba(ORANGE)[:3]), 0, 255).astype(np.uint8),
                             grey)
    rgba_c[:, 3] = np.where(has, 255, 150)
    rad_c = np.where(has, P2.BG_RADIUS * 1.25, P2.BG_RADIUS * 0.95)

    fig = plt.figure(figsize=(17.6, 21.0), dpi=190)
    gs = GridSpec(6, 1, figure=fig, height_ratios=[1.0, 1.55, 0.95, 1.55, 1.15, 1.15],
                  hspace=0.60, left=0.055, right=0.945, top=0.940, bottom=0.028)
    fig.suptitle(f"FIG 0   ·   the cohort   ·   the gate saw {len(u)} electrodes in "
                 f"{len(order)} patients and kept {int(u_kept.sum())}   ·   "
                 f"{int(has.sum())} of those have a LanA value",
                 x=0.055, y=0.982, ha="left", fontsize=15.5, color=INK)
    fig.text(0.055, 0.964,
             "What the responsiveness gate saw and what it kept, where those electrodes "
             "are, how much of them the language atlas can see, and what the gate is "
             "actually deciding at the line.  Each patient keeps the colour FIG 1 draws "
             "it in.", fontsize=9.8, color=MUTED, va="top")

    axA = fig.add_subplot(gs[0])
    panel_bars(axA, order, tot, kep, pcol, "electrodes: seen (hatched), kept (solid)")
    cB = GridSpecFromSubplotSpec(1, 3, gs[1], wspace=0.03)
    axB = [fig.add_subplot(cB[0, i]) for i in range(3)]
    nB = panel_brains(axB, uxyz, uhemi, rgba_u, rad_u,
                      "kept: the patient's colour\nrejected by the gate: grey")
    axC = fig.add_subplot(gs[2])
    panel_bars(axC, order, kep, lan, pcol, "kept electrodes: all (hatched), with LanA")
    cD = GridSpecFromSubplotSpec(1, 3, gs[3], wspace=0.03)
    axD = [fig.add_subplot(cD[0, i]) for i in range(3)]
    nD = panel_brains(axD, d["xyz"], d["hemi"], rgba_c, rad_c,
                      "kept, with a LanA value: orange\nwithout one: grey")

    ex = {}
    for tag, row, ok, cell in (("E", a, True, gs[4]), ("F", b, False, gs[5])):
        c_ = GridSpecFromSubplotSpec(2, 1, cell, height_ratios=[0.26, 1.0], hspace=0.05)
        axS, axE = fig.add_subplot(c_[0]), fig.add_subplot(c_[1])
        ex[tag] = (axS, axE, panel_example(axS, axE, t, row, gp, ok))

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    pE, pF = ex["E"][1].get_position(), ex["F"][1].get_position()
    cax = fig.add_axes([0.958, pF.y0, 0.010, pE.y1 - pF.y0])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-VLIM, VLIM), cmap="RdBu_r"), cax=cax)
    cb.set_label("dB re baseline", fontsize=7.6, color=MUTED)
    cb.ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
    cb.outline.set_visible(False)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, label, verdict, col in (
            (axA, "A  ·  electrodes the gate saw, and kept, per patient", "", INK),
            (axB[0], "B  ·  the same electrodes on the brain", "", INK),
            (axC, "C  ·  of the kept electrodes, those with a LanA value", "", INK),
            (axD[0], "D  ·  the same on the brain", "", INK),
            (ex["E"][0], "E  ·  just through the gate, and why", ex["E"][2]["verdict"],
             GREEN),
            (ex["F"][0], "F  ·  just not, and why", ex["F"][2]["verdict"], RED)):
        top = inv.transform((0, ax.get_tightbbox(r).y1))[1]
        # at the grid's left margin for every panel: a brain axes shrinks to its
        # image's aspect, so its own left edge would put B and D's labels indented
        fig.text(0.055, top + 0.004, label, fontsize=10.4, color=INK, va="bottom")
        if verdict:
            fig.text(0.945, top + 0.004, verdict, fontsize=9.2, color=col, va="bottom",
                     ha="right", fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "FIG0_cohort.png"
    P2.save_png(fig, p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P2.save_text(tab.to_csv(index=False), p.with_name(p.stem + "_patients.csv"))
    exrows = [dict(side=side, patient=e["patient"], electrode=e["electrode"], **q)
              for side, e in (("kept", ex["E"][2]), ("discarded", ex["F"][2]))
              for q in e["detail"]]
    P2.save_text(pd.DataFrame(exrows).to_csv(index=False),
                 p.with_name(p.stem + "_examples.csv"))
    n_noxyz = dict(kept=int((~np.isfinite(uxyz[:, 0]) & u_kept).sum()),
                   rejected=int((~np.isfinite(uxyz[:, 0]) & ~u_kept).sum()))
    caption_0(p.with_name(p.stem + "_caption.txt"), d, gp, tab, counts, u, nB, nD,
              n_noxyz, ex["E"][2], ex["F"][2], how, fset, k)
    print(f"  {time.time()-t0:.0f}s  -> {p.name}")
    return p


# ---- the caption --------------------------------------------------------------------
def caption_0(path, d, gp, tab, counts, u, nB, nD, n_noxyz, exE, exF, how, fset, k):
    L = []
    A = L.append
    nb = gp["n_freq"] * gp["n_time"]
    A("FIG 0   ·   the cohort")
    A("=" * 100)
    A("")
    A("THE GATE, as the cache that applied it recorded it")
    A("-" * 100)
    A(f"For each electrode-condition ERSP - {gp['n_freq']} frequencies x {gp['n_time']} "
      f"time-normalised bins = {nb} bins,")
    A(f"0-{FMAX:.0f} Hz, dB re the pre-stimulus baseline - the electrode-condition passes if")
    A(f"  at least {100*gp['min_pos']:g}% of bins are above {gp['thr_pos']:+.1f} dB "
      f"({round(gp['min_pos']*nb)} bins), OR")
    A(f"  at least {100*gp['min_neg']:g}% of bins are below {gp['thr_neg']:+.1f} dB "
      f"({round(gp['min_neg']*nb)} bins).")
    A("An electrode is KEPT if any of its three conditions passes. The rule counts bins")
    A("over a threshold across the whole cube: it is not a high-gamma measure and it is")
    A("not a test of whether the response repeats. Re-derived from the stored proportions,")
    A("it reproduces the stored flag on every row of the table.")
    A("")
    A("WHAT 'TOTAL' MEANS, AND WHAT HAPPENED TO EVERY ELECTRODE IN THE TABLE")
    A("-" * 100)
    A("The total in A is what the gate SAW: the ungated table for the cohort's patients,")
    A("minus what lf_concat removes before the gate - non-neural channels, grid and")
    A("microelectrode contacts, excluded patients - and minus electrodes missing a")
    A("condition, which the gate never sees whole. Those filters are the pipeline's own,")
    A("imported, and the total splits exactly into the cohort and the gate's rejects:")
    for f_ in FATES:
        A(f"  {f_:<28} {int(counts.get(f_, 0)):>5}")
    A(f"  {'seen by the gate (A)':<28} {len(u):>5}   = kept + failed the gate")
    A("")
    A("PANEL A - electrodes the gate saw, and kept, per patient")
    A("-" * 100)
    A(f"  {len(tab)} patients in id order. Hatched = seen, solid = kept, label kept/seen.")
    A(f"  seen {int(tab.n_total.sum())}   kept {int(tab.n_gated.sum())} "
      f"({100*tab.n_gated.sum()/tab.n_total.sum():.1f}%)")
    hi, lo = tab.loc[tab.gate_rate.idxmax()], tab.loc[tab.gate_rate.idxmin()]
    A(f"  highest pass rate   {hi.patient}  {int(hi.n_gated)}/{int(hi.n_total)} "
      f"({100*hi.gate_rate:.0f}%)")
    A(f"  lowest              {lo.patient}  {int(lo.n_gated)}/{int(lo.n_total)} "
      f"({100*lo.gate_rate:.0f}%)")
    A("")
    A("PANEL B - the same electrodes on the brain")
    A("-" * 100)
    A("Left, from above (anterior up, left on the left), right. Kept electrodes in the")
    A("patient's colour at full size; rejected ones smaller, grey, and lighter.")
    A(f"  drawn   left {nB['L']}   from above {nB['T']}   right {nB['R']}")
    A(f"  without fsaverage coordinates, so not drawn: {n_noxyz['kept']} kept, "
      f"{n_noxyz['rejected']} rejected")
    A("")
    A("PANEL C - of the kept electrodes, those with a LanA value")
    A("-" * 100)
    A(f"  {int(tab.n_lana.sum())} of {int(tab.n_gated.sum())} "
      f"({100*tab.n_lana.sum()/tab.n_gated.sum():.1f}%). Hatched = kept, solid = with a "
      f"value, label with/kept.")
    none = tab[tab.n_lana == 0]
    if len(none):
        A(f"  no value at all   {', '.join(none.patient)} - the atlas run predates "
          f"{'them' if len(none) > 1 else 'it'}; FIG 3 excludes every one of "
          f"{'their' if len(none) > 1 else 'its'} electrodes.")
    thin = tab[(tab.n_lana > 0) & (tab.lana_coverage < 0.7)]
    if len(thin):
        A("  below 70%         " + ", ".join(f"{p} ({100*c:.0f}%)" for p, c in
                                             zip(thin.patient, thin.lana_coverage)))
    A("")
    A("PANEL D - the same on the brain")
    A("-" * 100)
    A("Kept electrodes only: orange with a LanA value, grey without.")
    A(f"  drawn   left {nD['L']}   from above {nD['T']}   right {nD['R']}")
    A("")
    A("  patient        seen   kept   with LanA   pass rate   LanA coverage")
    for r_ in tab.itertuples():
        A(f"  {r_.patient:<12} {r_.n_total:>5}  {r_.n_gated:>5}   {r_.n_lana:>9}   "
          f"{100*r_.gate_rate:>8.1f}%   {100*r_.lana_coverage:>10.1f}%")
    A("")
    A("PANELS E and F - at the line")
    A("-" * 100)
    A("Three conditions side by side on the warped axis every later figure uses (GO cue")
    A("at the middle of each block, no fixation bins). Outlines are the bins the criterion")
    A("counts: solid above the positive threshold, dotted below the negative one. Under")
    A("each block, the count against the count it needed; the block that passes is bold.")
    A("")
    A(f"  E  {exE['verdict']}")
    A(f"  F  {exF['verdict']}")
    A(f"  chosen as: {how}. Of the {int((~u.kept).sum())} electrodes the gate rejected, F")
    A("  is the one that came closest.")
    for tag, e in (("E", exE), ("F", exF)):
        A(f"  {tag}  {e['who']}")
        for q in e["detail"]:
            A(f"     {q['condition']:<9} above: {q['bins_pos']:>5} of {q['need_pos']}   "
              f"below: {q['bins_neg']:>5} of {q['need_neg']}   "
              f"{'PASSES' if q['passes'] else 'no'}")
    A("")
    A("PROVENANCE")
    A("-" * 100)
    A(f"  cohort          {fset} cnmf run at K={k} - the electrode set is the same at every K")
    A(f"  gate table      {CACHE_DIR.relative_to(ROOT).as_posix()}/df_meta.parquet")
    A(f"  gate params     {CACHE_DIR.relative_to(ROOT).as_posix()}/params.json")
    A("  filters         lf_dataset.is_non_neural_electrode / is_grid_electrode /")
    A("                  is_micro_electrode, lf_concat.DEFAULT_EXCLUDE_PATIENTS")
    A(f"  coordinates     {P2.COORDS.relative_to(ROOT).as_posix()}")
    A(f"  atlas           {d['lana_src']}")
    A(f"  built by        00_paper2_figure0_coverage.py --feature-set {fset} --k {k}")
    A(f"  built on        {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A("")
    A("WHAT THIS FIGURE DOES NOT SHOW")
    A("-" * 100)
    A("  - Whether the gate is the RIGHT gate. E and F show that at the line the kept and")
    A("    the discarded look alike and one bin in tens of thousands decides it; whether a")
    A("    reliability criterion should replace it is a separate analysis.")
    A("  - Electrodes with no fsaverage coordinates are counted in A and C and absent from")
    A("    B and D.")
    P2.save_text("\n".join(L) + "\n", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="concat_hg")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    print(f"=== FIGURE 0 ===  cohort of {a.feature_set} at K={a.k}")
    figure_0(a.feature_set, a.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
