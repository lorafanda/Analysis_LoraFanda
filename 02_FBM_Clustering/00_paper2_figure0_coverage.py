#!/usr/bin/env python3
"""
00_paper2_figure0_coverage.py - FIG 0, the task and the cohort, in two figures.

    python 00_paper2_figure0_coverage.py                      both figures
    python 00_paper2_figure0_coverage.py --k 8 --feature-set concat_hg

THE PAPER'S FIG 0 (FIG0_cohort.png)
    A   the task, shown through four electrodes: for each, where it is on its own
        hemisphere, and its three conditions concatenated on the warped time axis
        every later figure uses - stimulus, then the response cue at mid-block.
        One auditory, one visual, one motor, one preparatory. HAND-PICKED
        ILLUSTRATIONS, named in EXEMPLARS below with the reading Lora gave them;
        the caption records each one's coordinates, gate margins and whether it is in
        the analysed cohort.
    B   every electrode the gate saw, on the brain - left, from above, right - the
        ones it kept in green, the ones it rejected in grey.

THE SUPPLEMENT (FIG0_cohort_supplement.png)
    A   electrodes per patient: the total the gate SAW (hatched) and the number that
        came THROUGH it (solid), each patient in the colour FIG 1 draws it in
    B   the same electrodes on the brain, kept in the patient's colour, rejected grey
    C   of the kept electrodes, the ones with a LanA value (solid) against all kept
    D   the same on the brain: with a LanA value in orange, without one in grey
    E   an electrode that JUST passed the gate, and F one that JUST failed - on the
        same shaft where such a pair exists - with the counted bins outlined and the
        count that decided it

WHAT 'TOTAL' MEANS. The electrodes the gate actually saw: the ungated table for the
cohort's patients, minus what lf_concat removes BEFORE the gate - non-neural channels,
grid and microelectrode contacts, excluded patients - and minus electrodes missing a
condition. Those filters are IMPORTED from the pipeline, and the result is checked: the
total must split exactly into the cohort and the gate's rejects, or the figure stops.

THE GATE, read from the cache's own params.json rather than retyped. For each
electrode-condition ERSP (n_freq x n_time bins, dB re the pre-stimulus baseline), the
fraction of bins above thr_pos must reach min_prop_pos, OR the fraction below thr_neg
must reach min_prop_neg. An electrode is kept if ANY of its three conditions passes.

NO CAPTION IS DRAWN ON EITHER FIGURE - each gets <name>_caption.txt, plus
_exemplars.csv (paper) and _patients.csv / _examples.csv (supplement).
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


# The cohort, its patient colours, the brain scenes, the trial strip and the writers come
# from FIG 1; the atlas join from FIG 3. Nothing is reimplemented, so this figure cannot
# describe a different cohort, colour a patient differently, or draw a different brain
# from the figures that follow it.
P2 = _load("p2fig1", "00_Paper2_Figures.py")
F3 = _load("p2fig3", "00_paper2_figure3_lana.py")

CLUST, OUT = P2.CLUST, P2.OUT
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
RED, GREEN, BLUE = P2.RED, P2.GREEN, P2.BLUE
ORANGE, PURPLE = "#e08214", "#5b2c83"
CACHE_DIR = ROOT / "outputs" / "_dataset" / "concat_source_v4"
VLIM = 7.0          # the project's own ERSP colour scale, dB
FMAX = 400.0        # Hz, the top of the cube the gate counts over
FATES = ("kept", "failed the gate", "grid contact", "microelectrode",
         "non-neural channel", "missing a condition", "patient not in the cohort")

# THE FOUR ELECTRODES OF PANEL A. Chosen by hand as illustrations of the task, not by
# any statistic; the reading in the fourth column is Lora's. A kind whose patient is
# None is skipped with a note, so the figure still builds while a choice is pending.
#   kind, patient, electrode, region, what it shows, colour
EXEMPLARS = [
    ("auditory", "PAT_3455", "OTD7", "superior temporal gyrus",
     "selective to the auditory stimulus, and again when the patient speaks", BLUE),
    ("visual", "EL043", "IOG7", "inferior occipital gyrus",
     "responds to the picture and the written sentence, not to the sound", PURPLE),
    ("motor", "PAT_6953", "TPD10", "sensorimotor cortex, around the central sulcus",
     "the verbal response: a burst after the cue in every condition, nothing to the stimulus", RED),
    ("preparatory", "PAT_6854", "IAG6", "anterior insula",
     "activity building toward the response cue in every condition", ORANGE),
]


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


def _show(ax, img):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def panel_brains(axes, xyz, hemi, rgba, radius):
    """Left, from above, right; the count drawn under each. The colour key goes on the
    panel's label line, not on the image, where the frontal pole's spheres clip it."""
    _fresh_scenes()
    n = {}
    for ax, side, tag in zip(axes, ("L", "T", "R"), ("left", "from above", "right")):
        img, n[side] = render_set(side, xyz, hemi, rgba, radius)
        _show(ax, img)
        ax.text(0.02, 0.02, f"{tag}  {n[side]}", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=7.0, color=MUTED)
    return n


def rgba_of(colour, alpha=255):
    c = np.clip(255 * np.array(to_rgba(colour)[:3]), 0, 255).astype(np.uint8)
    return np.array([*c, alpha], np.uint8)


# ---- the concatenated ERSP ---------------------------------------------------------
def cube_of(t, patient, electrode, conds):
    """The three conditions of one electrode, side by side, plus the rows behind them."""
    rows = {c: t[(t.patient_id == patient) & (t.electrode == electrode)
                 & (t.condition == c)] for c in conds}
    for c, r in rows.items():
        if r.empty:
            raise SystemExit(f"{patient} {electrode}: no ERSP for '{c}'")
    rows = {c: r.iloc[0] for c, r in rows.items()}
    arrs = [np.load(rows[c].file_path) for c in conds]
    nf, nt = arrs[0].shape
    if any(a.shape != (nf, nt) for a in arrs):
        raise SystemExit(f"{patient} {electrode}: conditions differ in shape")
    return np.concatenate(arrs, axis=1), arrs, rows, nf, nt


def draw_cube(ax, A, conds, nf):
    """The concatenated ERSP with the block boundaries and the GO cue dashed."""
    x = np.linspace(0, len(conds), A.shape[1])
    f = np.linspace(0, FMAX, nf)
    ax.pcolormesh(x, f, A, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, shading="auto",
                  rasterized=True)
    for b in range(len(conds)):
        ax.axvline(b + 0.5, color=INK, lw=0.8, ls=(0, (4, 3)))
        if b:
            ax.axvline(b, color=INK, lw=1.2)
    ax.set_xlim(0, len(conds)); ax.set_ylim(0, FMAX)
    ax.set_xticks([])
    ax.set_ylabel("Hz", fontsize=7.6)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    return x


# ---- panels: the paper -----------------------------------------------------------
def panel_exemplar(axH, axB, axS, axC, t, ex, gp, keys):
    """One electrode: a header line, where it sits on its own hemisphere, and its
    three conditions concatenated under the trial strip."""
    kind, patient, electrode, region, reading, col = ex
    conds = gp["conds"]
    A, arrs, rows, nf, nt = cube_of(t, patient, electrode, conds)
    key = f"{patient}|{P2.norm(electrode)}"
    xyz, hemi = coords([key])
    if not np.isfinite(xyz[0, 0]):
        raise SystemExit(f"{patient} {electrode}: no fsaverage coordinate, cannot place it")
    side = str(hemi[0])
    # the brain: this electrode alone, large, on the lateral view of its hemisphere
    _fresh_scenes()
    img, _ = render_set(side, xyz, hemi, rgba_of(col)[None, :],
                        np.array([P2.BG_RADIUS * 3.2]))
    _show(axB, img)
    axB.text(0.02, 0.02, f"{'left' if side == 'L' else 'right'} hemisphere",
             transform=axB.transAxes, ha="left", va="bottom", fontsize=6.8, color=MUTED)
    # the header
    # the strip below draws its condition labels ABOVE itself, into the bottom of this
    # row, so the header's own text stays in the upper 60%
    axH.axis("off")
    axH.text(0, 0.86, kind.upper(), transform=axH.transAxes, ha="left", va="center",
             fontsize=10.2, color=col, fontweight="bold")
    axH.text(0.17, 0.86, f"{patient} · {electrode} · {region}", transform=axH.transAxes,
             ha="left", va="center", fontsize=9.6, color=INK)
    axH.text(0.17, 0.56, reading, transform=axH.transAxes, ha="left", va="center",
             fontsize=8.2, color=MUTED, style="italic")
    # the strip and the cube
    P2.trial_strip(axS, dict(conds=conds))
    draw_cube(axC, A, conds, nf)
    for s_ in axC.spines.values():
        s_.set_color(col); s_.set_linewidth(1.4)
    return dict(kind=kind, patient=patient, electrode=electrode, region=region,
                reading=reading, hemi=side, x=float(xyz[0, 0]), y=float(xyz[0, 1]),
                z=float(xyz[0, 2]), in_cohort=key in keys,
                **{f"margin_{c}": round(float(rows[c].margin), 3) for c in conds},
                peak_dB=round(float(np.nanmax(A)), 2))


def figure_0_paper(t, u, d, gp):
    t0 = time.time()
    keys = cohort_keys(d)
    exs = [e for e in EXEMPLARS if e[1] is not None]
    for e in EXEMPLARS:
        if e[1] is None:
            print(f"    NOTE no electrode chosen yet for the {e[0]} exemplar - skipped")

    # B: what the gate saw, kept in green, rejected in grey
    uxyz, uhemi = coords(u.key.to_list())
    kept = u.kept.to_numpy()
    rgba_u = np.where(kept[:, None], rgba_of(GREEN), rgba_of(P2.BG, 150)).astype(np.uint8)
    rad_u = np.where(kept, P2.BG_RADIUS * 1.25, P2.BG_RADIUS * 0.95)

    ncell = max(len(exs), 1)
    nrow_ex = int(np.ceil(ncell / 2))
    fig = plt.figure(figsize=(17.6, 5.6 + 3.0 * nrow_ex), dpi=190)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0 * nrow_ex, 1.15], hspace=0.30,
                  left=0.055, right=0.945, top=0.905, bottom=0.04)
    fig.suptitle(f"FIG 0   ·   the task and the cohort   ·   the gate saw {len(u)} "
                 f"electrodes in {d['n_patients']} patients and kept {int(kept.sum())}",
                 x=0.055, y=0.975, ha="left", fontsize=15.5, color=INK)
    fig.text(0.055, 0.950,
             "Three naming conditions - auditory, picture, written sentence - each with a "
             "stimulus and then a response cue (dashed), on the warped time axis every "
             "later figure uses.  Four electrodes show what the recordings look like; the "
             "brains below show every electrode the responsiveness gate saw.",
             fontsize=9.8, color=MUTED, va="top")

    gA = GridSpecFromSubplotSpec(nrow_ex, 2, gs[0], hspace=0.40, wspace=0.10)
    rows_out, firsts = [], []
    for i, ex in enumerate(exs):
        cell = GridSpecFromSubplotSpec(3, 2, gA[i // 2, i % 2],
                                       height_ratios=[0.46, 0.30, 1.0],
                                       width_ratios=[1.0, 2.5], hspace=0.06, wspace=0.16)
        axH = fig.add_subplot(cell[0, :])
        axB = fig.add_subplot(cell[1:, 0])
        axS, axC = fig.add_subplot(cell[1, 1]), fig.add_subplot(cell[2, 1])
        rows_out.append(panel_exemplar(axH, axB, axS, axC, t, ex, gp, keys))
        firsts.append((axH, axC))
        print(f"    {ex[0]:<12} {ex[1]} {ex[2]}  {'in cohort' if rows_out[-1]['in_cohort'] else 'NOT in cohort'}")

    cB = GridSpecFromSubplotSpec(1, 3, gs[1], wspace=0.03)
    axB3 = [fig.add_subplot(cB[0, i]) for i in range(3)]
    nB = panel_brains(axB3, uxyz, uhemi, rgba_u, rad_u)

    # one colour bar for the cubes, in the margin the grid leaves free
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if firsts:
        tops = [axc.get_position() for _, axc in firsts]
        y0, y1 = min(p.y0 for p in tops), max(p.y1 for p in tops)
        cax = fig.add_axes([0.958, y0, 0.010, y1 - y0])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-VLIM, VLIM), cmap="RdBu_r"),
                          cax=cax)
        cb.set_label("dB re baseline", fontsize=7.6, color=MUTED)
        cb.ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
        cb.outline.set_visible(False)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    labels = [(axB3[0], "B  ·  every electrode the gate saw",
               "active, through the gate: green   ·   inactive, rejected: grey")]
    if firsts:
        labels.insert(0, (firsts[0][0], "A  ·  the task, through four electrodes", ""))
    for ax, label, right in labels:
        top = inv.transform((0, ax.get_tightbbox(r).y1))[1]
        fig.text(0.055, top + 0.004, label, fontsize=10.4, color=INK, va="bottom")
        if right:
            fig.text(0.945, top + 0.004, right, fontsize=8.8, color=MUTED, va="bottom",
                     ha="right")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "FIG0_cohort.png"
    P2.save_png(fig, p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    ex_df = pd.DataFrame(rows_out)
    P2.save_text(ex_df.to_csv(index=False), p.with_name(p.stem + "_exemplars.csv"))
    n_noxyz = dict(kept=int((~np.isfinite(uxyz[:, 0]) & kept).sum()),
                   rejected=int((~np.isfinite(uxyz[:, 0]) & ~kept).sum()))
    caption_paper(p.with_name(p.stem + "_caption.txt"), d, gp, u, ex_df, nB, n_noxyz)
    print(f"  {time.time()-t0:.0f}s  -> {p.name}")
    return p


# ---- panels: the supplement -------------------------------------------------------
def panel_bars(ax, order, total, part, pcol, ylabel):
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
    """One electrode at the gate's line: its conditions side by side, the bins the
    criterion counts OUTLINED (solid above thr_pos, dotted below thr_neg), and under
    each block the count against the count it needed."""
    conds = gp["conds"]
    A, arrs, rows, nf, nt = cube_of(t, row.patient_id, row.electrode, conds)
    x = draw_cube(ax, A, conds, nf)
    f = np.linspace(0, FMAX, nf)
    need_p, need_n = round(gp["min_pos"] * nf * nt), round(gp["min_neg"] * nf * nt)
    detail = []
    for b, c in enumerate(conds):
        xs, a = x[b * nt:(b + 1) * nt], arrs[b]
        ax.contour(xs, f, (a > gp["thr_pos"]).astype(float), levels=[0.5],
                   colors=[INK], linewidths=0.5)
        if (a < gp["thr_neg"]).any():
            ax.contour(xs, f, (a < gp["thr_neg"]).astype(float), levels=[0.5],
                       colors=[INK], linewidths=0.5, linestyles="dotted")
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


def figure_0_supplement(t, g, u, d, gp):
    t0 = time.time()
    pcol = P2.patient_colours(d)
    a, b, how = pick_examples(u)
    counts = g.fate.value_counts()
    print("  " + "   ".join(f"{f_} {int(counts.get(f_, 0))}" for f_ in FATES))
    print(f"  at the line: {a.patient_id} {a.electrode} (margin {a.best:.4f}) vs "
          f"{b.patient_id} {b.electrode} ({b.best:.4f}) - {how}")

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

    uxyz, uhemi = coords(u.key.to_list())
    u_kept = u.kept.to_numpy()
    grey = rgba_of(P2.BG, 150)
    rgba_u = np.array([rgba_of(pcol[p]) if kp else grey
                       for p, kp in zip(u.patient_id, u_kept)], np.uint8)
    rad_u = np.where(u_kept, P2.BG_RADIUS * 1.25, P2.BG_RADIUS * 0.95)
    has = np.asarray(d["has"])
    rgba_c = np.where(has[:, None], rgba_of(ORANGE), grey).astype(np.uint8)
    rad_c = np.where(has, P2.BG_RADIUS * 1.25, P2.BG_RADIUS * 0.95)

    fig = plt.figure(figsize=(17.6, 21.0), dpi=190)
    gs = GridSpec(6, 1, figure=fig, height_ratios=[1.0, 1.55, 0.95, 1.55, 1.15, 1.15],
                  hspace=0.60, left=0.055, right=0.945, top=0.940, bottom=0.028)
    fig.suptitle(f"FIG S0   ·   the cohort, per patient   ·   the gate saw {len(u)} "
                 f"electrodes in {len(order)} patients and kept {int(u_kept.sum())}   ·   "
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
    nB = panel_brains(axB, uxyz, uhemi, rgba_u, rad_u)
    keyB = "kept: the patient's colour   ·   rejected by the gate: grey"
    axC = fig.add_subplot(gs[2])
    panel_bars(axC, order, kep, lan, pcol, "kept electrodes: all (hatched), with LanA")
    cD = GridSpecFromSubplotSpec(1, 3, gs[3], wspace=0.03)
    axD = [fig.add_subplot(cD[0, i]) for i in range(3)]
    nD = panel_brains(axD, d["xyz"], d["hemi"], rgba_c, rad_c)
    keyD = "kept, with a LanA value: orange   ·   without one: grey"

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
    for ax, label, right, col, bold in (
            (axA, "A  ·  electrodes the gate saw, and kept, per patient", "", INK, False),
            (axB[0], "B  ·  the same electrodes on the brain", keyB, MUTED, False),
            (axC, "C  ·  of the kept electrodes, those with a LanA value", "", INK, False),
            (axD[0], "D  ·  the same on the brain", keyD, MUTED, False),
            (ex["E"][0], "E  ·  just through the gate, and why", ex["E"][2]["verdict"],
             GREEN, True),
            (ex["F"][0], "F  ·  just not, and why", ex["F"][2]["verdict"], RED, True)):
        top = inv.transform((0, ax.get_tightbbox(r).y1))[1]
        fig.text(0.055, top + 0.004, label, fontsize=10.4, color=INK, va="bottom")
        if right:
            fig.text(0.945, top + 0.004, right, fontsize=9.2 if bold else 8.8, color=col,
                     va="bottom", ha="right", fontweight="bold" if bold else "normal")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "FIG0_cohort_supplement.png"
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
    caption_supplement(p.with_name(p.stem + "_caption.txt"), d, gp, tab, counts, u, nB,
                       nD, n_noxyz, ex["E"][2], ex["F"][2], how)
    print(f"  {time.time()-t0:.0f}s  -> {p.name}")
    return p


# ---- the captions ------------------------------------------------------------------
def _gate_lines(A, gp, u, n_in):
    nb = gp["n_freq"] * gp["n_time"]
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
    A(f"it reproduces the stored flag on every row; all {n_in} cohort electrodes come out kept.")
    A("")


def caption_paper(path, d, gp, u, ex_df, nB, n_noxyz):
    L = []
    A = L.append
    A("FIG 0   ·   the task and the cohort")
    A("=" * 100)
    A("")
    A("THE TASK")
    A("-" * 100)
    A("Three naming conditions: an auditory stimulus, a picture, a written sentence. Each")
    A("trial is stimulus, then a response cue. The time axis is WARPED, not clock time:")
    A("stimulus and response each occupy half a block (proportions 0, 0.5, 0.5), the")
    A("response cue falls at mid-block (dashed), and the fixation screen contributes no")
    A("bins - the baseline every dB value is expressed against comes from it. This is the")
    A("axis every centroid and every cube in the paper is drawn on.")
    A("")
    A("PANEL A - four electrodes, chosen by hand as illustrations")
    A("-" * 100)
    A("Not selected by any statistic and not a claim about the cohort: they show what the")
    A("recordings look like and what the three conditions do to a contact of each kind.")
    A("Each is drawn alone on the lateral view of its own hemisphere, and its three")
    A("conditions concatenated on the axis above, 0-400 Hz, on the project's +-7 dB scale.")
    A("")
    for r_ in ex_df.itertuples():
        A(f"  {r_.kind.upper():<12} {r_.patient} {r_.electrode}  {r_.region}  "
          f"({r_.hemi}, x {r_.x:.0f}  y {r_.y:.0f}  z {r_.z:.0f})")
        A(f"  {'':<12} {r_.reading}")
        A(f"  {'':<12} gate margin  "
          + "  ".join(f"{c} {getattr(r_, 'margin_' + c):.2f}" for c in gp["conds"])
          + f"   (>= 1 passes)   peak {r_.peak_dB:+.1f} dB   "
          + ("in the analysed cohort" if r_.in_cohort else
             "NOT IN THE ANALYSED COHORT - removed before the gate"))
        A("")
    A("PANEL B - every electrode the gate saw")
    A("-" * 100)
    A("Left, from above (anterior up, left on the left), right. Electrodes that passed")
    A("the gate - the cohort every later figure is computed on - in green at full size;")
    A("the ones it rejected smaller, grey and lighter.")
    A(f"  seen {len(u)}   kept {int(u.kept.sum())}   rejected {int((~u.kept).sum())}")
    A(f"  drawn   left {nB['L']}   from above {nB['T']}   right {nB['R']}")
    A(f"  without fsaverage coordinates, so not drawn: {n_noxyz['kept']} kept, "
      f"{n_noxyz['rejected']} rejected")
    A("")
    _gate_lines(A, gp, u, int(u.kept.sum()))
    A("PROVENANCE")
    A("-" * 100)
    A(f"  gate table      {CACHE_DIR.relative_to(ROOT).as_posix()}/df_meta.parquet")
    A(f"  gate params     {CACHE_DIR.relative_to(ROOT).as_posix()}/params.json")
    A(f"  coordinates     {P2.COORDS.relative_to(ROOT).as_posix()}")
    A("  exemplars       EXEMPLARS in 00_paper2_figure0_coverage.py")
    A(f"  built on        {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A("")
    A("WHAT THIS FIGURE DOES NOT SHOW")
    A("-" * 100)
    A("  - Who the cohort is made of, patient by patient, how much of it the atlas covers,")
    A("    and what the gate decides at the line: FIG S0, the supplement.")
    A("  - Any claim that these four are typical. They are the clearest examples, picked")
    A("    to explain the task.")
    P2.save_text("\n".join(L) + "\n", path)


def caption_supplement(path, d, gp, tab, counts, u, nB, nD, n_noxyz, exE, exF, how):
    L = []
    A = L.append
    A("FIG S0   ·   the cohort, per patient")
    A("=" * 100)
    A("")
    _gate_lines(A, gp, u, int(u.kept.sum()))
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
    A(f"  cohort          the cnmf run's labels.csv - the electrode set is the same at every K")
    A(f"  gate table      {CACHE_DIR.relative_to(ROOT).as_posix()}/df_meta.parquet")
    A(f"  gate params     {CACHE_DIR.relative_to(ROOT).as_posix()}/params.json")
    A("  filters         lf_dataset.is_non_neural_electrode / is_grid_electrode /")
    A("                  is_micro_electrode, lf_concat.DEFAULT_EXCLUDE_PATIENTS")
    A(f"  coordinates     {P2.COORDS.relative_to(ROOT).as_posix()}")
    A(f"  atlas           {d['lana_src']}")
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


# ---- main --------------------------------------------------------------------------
def prepare(fset, k):
    gp = gate_params()
    d = F3.load(fset, k)                       # the cohort, with P_lana joined
    t = load_gate(gp)
    g = fate(per_electrode(t), d, gp)
    u = universe(g)
    return gp, d, t, g, u


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="concat_hg")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--only", choices=["paper", "supplement"], default=None)
    a = ap.parse_args()
    print(f"=== FIGURE 0 ===  cohort of {a.feature_set} at K={a.k}")
    gp, d, t, g, u = prepare(a.feature_set, a.k)
    if a.only in (None, "paper"):
        print("  the paper's FIG 0 ...")
        figure_0_paper(t, u, d, gp)
    if a.only in (None, "supplement"):
        print("  the supplement, FIG S0 ...")
        figure_0_supplement(t, g, u, d, gp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
