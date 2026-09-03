#!/usr/bin/env python3
"""
00_Paper2_Figures.py - the paper figures for Paper 2, built one at a time.

    python 00_Paper2_Figures.py                        every figure, every feature set
    python 00_Paper2_Figures.py --figure 1             FIG 1a/1b/1c/1d
    python 00_Paper2_Figures.py --figure 1 --feature-set concat_hg
    python 00_Paper2_Figures.py --figure 1 --k 12      override the held-out peak K

FIGURE 1 - one per feature set, convex NMF, each cut at ITS OWN held-out peak K.

    A   held-out variance vs K, all four feature sets, this one emphasised. The panel
        is identical in 1a-1d, so the four can be laid side by side.
    C   how to read a cluster mean - an idealised schematic, not data
    D   is each cluster a response type, or one patient? The largest single patient's
        share of every cluster, against a size-matched random draw from the same
        cohort. THIS PANEL IS DIAGNOSTIC, NOT DECORATIVE - it is drawn from the data
        and it will say a cluster is one patient's electrode strip when it is one.
    B   ONE BLOCK PER CLUSTER: the mean on top, spanning the block, and under it two
        3-D surface renders - the LEFT hemisphere seen from the LEFT and the RIGHT
        hemisphere seen from the RIGHT, so each is a true lateral view.

THE BRAIN RENDERS ARE THE SAME PIPELINE THE VISUALIZER USES. pyvista/VTK offscreen,
the fsaverage pial surfaces, and the house material constants read from
functions/lf_recon_shared_config.py rather than copied - so if that file changes, these
figures change with it instead of quietly disagreeing with every other render.

NO CAPTION IS DRAWN ON A FIGURE. Each one gets a sibling <name>_caption.txt with the
full provenance and construction detail, because a caption baked into a PNG cannot be
edited in the manuscript and cannot be checked against the data.
"""
from __future__ import annotations

import argparse
import colorsys
import io
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nb
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle, PathPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D

pv.OFF_SCREEN = True

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                        # noqa: E402
import lf_recon_shared_config as RC                         # noqa: E402

CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "paper_figures"
COORDS = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coords"
          / "ALL_PATIENTS_contacts_fsaverage.csv")
MESHES = ROOT / "outputs" / "250_recon" / "fsaverage" / "meshes"
HELDOUT = CLUST / "bsf_comparison" / "heldout_variance_ALL.csv"
PEAKS = CLUST / "bsf_comparison" / "heldout_peaks_cnmf.csv"

INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
RED, GREEN, BLUE = "#c1121f", "#1b7837", "#4a6fa5"
# THE VISUALIZER'S OWN RULE, transcribed from clustering_visualizer.html rather than
# invented here. updateElectrodes() reaches FULL cluster colour at a loading of 0.5 and
# fades toward grey 148 only BELOW that; radius and opacity track a separate ramp. An
# earlier version here ramped colour across the whole range, so a typical loading -
# the median is 0.33 - landed a third of the way out of grey and every render came out
# muddy. Matching the JS is what makes these look like the brains on the site.
BG = np.array([148, 148, 148]) / 255.0                      # the visualizer's grey 148
COLOR_FULL = 0.50           # at or above this loading the electrode is FULL cluster colour
ALPHA_FLOOR = 0.10          # opacity at a loading of zero; full opacity at LOAD_CAP
# LOADING CAP for the SIZE ramp. An electrode at or above this is drawn at full radius.
# It is a rendering choice: it changes how a loading is drawn, never the loading, the
# label, or any number reported anywhere.
LOAD_CAP = 0.70
RENDER_PX = (1100, 820)         # cropped to the brain afterwards, so this is headroom
BG_RADIUS = RC.DEPTH_RADIUS     # the electrode that is not in this cluster, in mm
DANGER = 0.50                   # more than half a cluster from one patient
N_PERM = 400                    # permutations for the size-matched random draw

FSETS = ["concat_hg", "concat_rawds", "concat_bands5", "concat_bands5z"]
TAG = {"concat_hg": "a", "concat_rawds": "b", "concat_bands5": "c",
       "concat_bands5z": "d"}
FS_LABEL = {"concat_hg": "high gamma (70-150 Hz) x time",
            "concat_rawds": "15 frequency bands x time",
            "concat_bands5": "5 frequency bands x time",
            "concat_bands5z": "5 frequency bands x time, z-scored per band"}
FS_SHORT = {"concat_hg": "HG", "concat_rawds": "15 bands",
            "concat_bands5": "5 bands", "concat_bands5z": "5 bands, z"}
# bands5z is a per-band z-score, so its values are STANDARD DEVIATIONS, not decibels.
# Labelling that axis "dB" would be a units error on the face of a paper figure.
UNITS = {"concat_hg": "dB vs baseline", "concat_rawds": "dB vs baseline",
         "concat_bands5": "dB vs baseline",
         "concat_bands5z": "SD (per-band z-score)"}
COND_LABEL = {"audio": "audio", "picture": "picture", "reading": "reading (sentences)"}
# The picture stimulus is a real line drawing and cannot be reproduced as a glyph. Crop
# it out of Figure1_FBM_v1.png to this path and it is used automatically; without it a
# framed-image glyph stands in, so the figure builds either way.
STIM_PICTURE_PNG = OUT / "assets" / "stim_picture.png"
# How the blocks are ORDERED on the page.
#   "matched"     position p holds the cluster corresponding to the reference's p-th,
#                 so block 1 is the same population of electrodes in every figure
#   "similarity"  each figure sorts its own clusters, most condition-alike first
#   "cluster"     numeric order
# Display only. Cluster ids, colours and every number are untouched in all three.
BLOCK_ORDER = "matched"
# The solution that defines the sequence, evaluated at whatever K the figure uses.
MATCH_REF_FSET, MATCH_REF_METHOD = "concat_bands5", "cnmf"
UPR = 4        # cluster blocks per row
UCOL = 6       # grid columns per block: the mean spans all 6, each render takes 3


def cluster_col(idx: int, k: int):
    """The colour clustering_visualizer.html paints cluster `idx` of `k`, exactly."""
    return colorsys.hls_to_rgb(idx / max(k, 1), 0.52, 0.62)


def norm(s) -> str:
    return str(s).replace("_", "").replace("-", "").upper()


def band_centre(lab: str) -> float:
    """'70-170Hz' -> 120.0 . 'hg' -> the HG band's own centre."""
    t = lab.replace("Hz", "")
    if "-" not in t:
        return 110.0                                        # concat_hg is 70-150 Hz
    a, b = t.split("-")
    return (float(a) + float(b)) / 2


def loading_t(w, k):
    """Normalised loading -> 0..1 SIZE weight, saturating at LOAD_CAP.

    Defined once and used by the renders AND by the caption's statistics, so the
    numbers printed beside a figure cannot drift from the ones it was drawn with.
    """
    return np.clip((np.asarray(w) - 1 / k) / (LOAD_CAP - 1 / k), 0, 1)


def loading_alpha(w):
    """Loading -> opacity. Full at LOAD_CAP, ALPHA_FLOOR at zero, linear between.

    Deliberately measured from ZERO rather than from 1/K, unlike the colour and size
    ramps: this channel is meant to sink the whole uninvolved population into the
    background, and a floor anchored at 1/K would leave everything below a flat mixture
    at full transparency and everything just above it nearly solid.
    """
    return ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * np.clip(np.asarray(w, float) / LOAD_CAP,
                                                       0, 1)


def loading_rgb(w, k, base):
    """Normalised loading -> RGB, the way clustering_visualizer.html does it.

    FULL cluster colour at or above COLOR_FULL; below it, mixed toward the same grey
    every other electrode is drawn in, reaching it exactly at a flat 1/K mixture. The
    colour therefore says "is this electrode expressing this component at all", and the
    radius says "how strongly" - two channels for two questions, which is the split the
    JS makes and the reason its brains read clearly.
    """
    w = np.asarray(w, float)
    g = np.clip((w - 1 / k) / (COLOR_FULL - 1 / k), 0, 1)[:, None]
    return BG[None, :] + g * (np.asarray(base)[None, :] - BG[None, :])


# ---- loading -----------------------------------------------------------------
def load_run(fset: str, k: int | None):
    run = LR.newest_run("cnmf", fset)
    if run is None:
        raise SystemExit(f"no cnmf run for {fset}")
    if k is None:
        pk = pd.read_csv(PEAKS)
        row = pk[pk.feature_set == fset]
        if row.empty:
            raise SystemExit(f"no held-out peak for {fset}; pass --k")
        k = int(row.k_peak.iloc[0])

    X = np.load(run / "X_train.npy").astype(float)
    gf = run / "loadings_by_k" / f"G_k{k:02d}.npy"
    if not gf.exists():
        raise SystemExit(f"{gf} is missing - 242 did not sweep K={k} for {fset}")
    G = np.load(gf).astype(float)
    if G.shape != (len(X), k):
        raise SystemExit(f"{gf.name} is {G.shape}, expected {(len(X), k)}")
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    lab = Gn.argmax(1)

    feats = json.loads((run / "feature_schema.json").read_text())["feature_names"]
    parts = [f.split("|") for f in feats]
    conds = list(dict.fromkeys(p[0] for p in parts))
    bands = list(dict.fromkeys(p[1] for p in parts))
    nt = len(feats) // (len(conds) * len(bands))
    # assert the order rather than trust it - see cube() for why this one matters
    exp = [f"{c}|{b}|" for b in bands for c in conds]
    got = [f"{p[0]}|{p[1]}|" for p in parts[::nt]]
    if exp != got:
        raise SystemExit("feature order is not band x cond x time; reshape would lie")

    meta = pd.read_csv(run / "labels.csv")
    co = pd.read_csv(COORDS)
    co["key"] = [f"{p}|{norm(n_)}" for p, n_ in zip(co["patient"], co["name"])]
    keys = [f"{p}|{norm(e)}" for p, e in zip(meta["patient_id"], meta["electrode"])]
    j = (pd.DataFrame({"key": keys})
         .merge(co[["key", "x", "y", "z", "hemi"]].drop_duplicates("key"),
                on="key", how="left"))
    xyz = j[["x", "y", "z"]].to_numpy(float)
    hemi = j["hemi"].astype(object).to_numpy()
    # fall back on the sign of x only where the table has no side, and count both, so
    # the caption can state how many electrodes each view actually got
    miss = pd.isna(hemi) & np.isfinite(xyz[:, 0])
    hemi = np.where(miss, np.where(xyz[:, 0] < 0, "L", "R"), hemi)
    return dict(run=run, k=k, X=X, G=G, Gn=Gn, lab=lab, conds=conds, bands=bands,
                nt=nt, xyz=xyz, hemi=hemi, n_hemi_from_x=int(miss.sum()),
                patient=meta["patient_id"].to_numpy(),
                n_patients=int(meta["patient_id"].nunique()), meta=meta)


def cube(X, d):
    """(n, features) -> (n, ncond, nband, nt) - the native shape of the feature set.

    The stored axis is BAND-major (band, cond, time), which the feature NAMES do not
    suggest - they read cond|band|t, so "audio|1-20Hz|t00" invites the reading that
    condition varies slowest, and it does not: audio|1-20Hz|t29 is followed by
    picture|1-20Hz|t00. Reshaping as (cond, band, time) would transpose frequency
    against condition and produce a figure that looks entirely plausible and is wrong.
    load_run() asserts the order before this is ever called.
    """
    nb_, ncd, nt = len(d["bands"]), len(d["conds"]), d["nt"]
    return X.reshape(len(X), nb_, ncd, nt).transpose(0, 2, 1, 3)


# ---- the 3-D surface renders -------------------------------------------------
_MESH: dict = {}
_BGGLYPH: dict = {}
_PLOTTER: dict = {}


# ---- writes to the share are not trusted -------------------------------------
def _verified_write(write_fn, path: Path, check_fn, tries: int = 6):
    """Write to a local temp file, CHECK it, then move it into place and check again.

    Writes to this share have twice truncated a file to zero or half its length and
    reported success - once a source file, once a finished figure. A save that is not
    read back is not a save, so every output here goes through this.
    """
    import shutil, tempfile
    tmp = Path(tempfile.gettempdir()) / f"_p2fig_{path.name}"
    last = ""
    for i in range(tries):
        try:
            write_fn(tmp)
            check_fn(tmp)
            n = tmp.stat().st_size
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp, path)
            if path.stat().st_size != n:
                raise OSError(f"{path.stat().st_size} of {n} bytes landed")
            check_fn(path)
            tmp.unlink(missing_ok=True)
            return n
        except Exception as e:                      # noqa: BLE001 - retry anything
            last = f"{type(e).__name__}: {e}"
            print(f"    write attempt {i+1} for {path.name} failed - {last}")
            time.sleep(2)
    raise SystemExit(f"could not write {path}: {last}")


def save_png(fig, path: Path, **kw):
    def _check(p):
        from PIL import Image
        with Image.open(p) as im:
            im.load()                               # forces a full decode
    return _verified_write(lambda p: fig.savefig(p, **kw), path, _check)


def save_text(text: str, path: Path):
    """Byte-exact. Text mode would translate newlines on the way out AND on the way
    back, so a file that landed perfectly could still compare unequal - which is what
    the first version of this did, six times in a row, on a file that was fine.
    """
    data = text.encode("utf-8")

    def _check(p):
        got = Path(p).read_bytes()
        if got != data:
            raise OSError(f"{len(got)} of {len(data)} bytes landed")
    return _verified_write(lambda p: Path(p).write_bytes(data), path, _check)


def hemi_mesh(side: str) -> pv.PolyData:
    """fsaverage pial surface for one hemisphere, as pyvista geometry.

    Read from the .gii in outputs/250_recon rather than through
    lf_recon_shared.load_fsaverage_meshes(), which wants a full FreeSurfer subject
    directory on \\\\nasac-m2 - a dependency these figures do not otherwise have.
    The vertices are the same surface, and their extents agree with the contact table
    to a few millimetres on all three axes, so nothing is transformed between them.
    """
    if side not in _MESH:
        g = nb.load(MESHES / f"fsaverage_{'lh' if side == 'L' else 'rh'}.gii")
        v = np.asarray(g.darrays[0].data, float)
        f = np.asarray(g.darrays[1].data, np.int64)
        m = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3, np.int64), f]).ravel())
        m.compute_normals(inplace=True)
        _MESH[side] = m
    return _MESH[side]


def _scene(side: str, d: dict, zoom: float):
    """The BRAIN only, built once per view and kept. side is "L", "R" or "T" (top).

    The electrodes are not in here: every one of them now depends on which cluster is
    being drawn, so they go on as a single actor that is added and removed around each
    screenshot. What stays is the expensive part - a 163,842-vertex surface, which
    rebuilding per cluster was costing 16 seconds a render instead of one.
    """
    if side in _PLOTTER:
        return _PLOTTER[side]
    xyz, hemi = d["xyz"], d["hemi"]
    finite = np.isfinite(xyz).all(1)
    if side == "T":
        # THE TOP VIEW: both hemispheres, seen from above, anterior at the top of the
        # image and the left hemisphere on the left - the same neurological convention
        # the two lateral views follow, so the three read as one brain.
        meshes, ok = [hemi_mesh("L"), hemi_mesh("R")], finite
    else:
        meshes, ok = [hemi_mesh(side)], finite & (hemi == side)
    pl = pv.Plotter(off_screen=True, window_size=RENDER_PX)
    for mesh in meshes:
        pl.add_mesh(mesh, color=RC.BRAIN_COLOR, opacity=RC.BRAIN_OPACITY_CLEAN,
                    smooth_shading=True, specular=RC.BRAIN_SPECULAR,
                    specular_power=RC.BRAIN_SPECULAR_POWER, ambient=RC.BRAIN_AMBIENT,
                    diffuse=RC.BRAIN_DIFFUSE)
    # per-electrode opacity needs order-independent transparency, or a faint sphere in
    # front of a solid one erases it depending only on draw order
    try:
        pl.enable_depth_peeling(12)
    except Exception:
        pass
    bb = np.array([m.bounds for m in meshes])
    b = (bb[:, 0].min(), bb[:, 1].max(), bb[:, 2].min(), bb[:, 3].max(),
         bb[:, 4].min(), bb[:, 5].max())
    cx, cy, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2
    dist = 2.4 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    if side == "T":
        eye, up = (cx, cy, cz + dist), (0, 1, 0)
    else:
        eye, up = (cx - dist if side == "L" else cx + dist, cy, cz), (0, 0, 1)
    pl.camera_position = (eye, (cx, cy, cz), up)
    pl.enable_parallel_projection()
    # FIT BEFORE ZOOMING. pyvista sets the parallel scale from the camera distance,
    # not from what is in view, so a fixed zoom that suits a lateral view - 127 mm of
    # brain top to bottom - clips a top view, which puts 174 mm of brain front to
    # back on the same vertical axis. The zoom is capped at what leaves a margin on
    # both axes of this view; the lateral views are well inside it and unchanged.
    W, H = RENDER_PX
    up_ext, across_ext = ((b[3] - b[2], b[1] - b[0]) if side == "T"
                          else (b[5] - b[4], b[3] - b[2]))
    vis = 2.0 * pl.camera.parallel_scale          # mm visible top-to-bottom at zoom 1
    fit = min(vis / (1.06 * up_ext), vis * (W / H) / (1.06 * across_ext))
    pl.camera.zoom(min(zoom, fit))
    _PLOTTER[side] = (pl, ok)
    return _PLOTTER[side]


def _crop_alpha(img, pad: int = 4):
    """Trim the transparent margin VTK letterboxes around the brain.

    Without this every render carries a wide empty border, the brain comes out small
    inside its axes, and the block gains a band of dead space under it.
    """
    if img.shape[2] < 4:
        return img
    ys, xs = np.where(img[:, :, 3] > 0)
    if not len(ys):
        return img
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, img.shape[1])
    return img[y0:y1, x0:x1]


def render_hemi(side: str, d: dict, j: int, zoom: float = 1.30):
    """One hemisphere, seen from its OWN side, with this cluster's electrodes.

    LEFT hemisphere from the LEFT and RIGHT hemisphere from the RIGHT, which is what
    makes each a true lateral view: anterior falls on the left of the image for L and
    on the right for R. Drawing both from the same side, as an earlier version did,
    shows one hemisphere from inside the head.

    EVERY electrode of the hemisphere is drawn by ONE rule applied to its own loading
    on THIS cluster - colour, size and opacity all read from that single number. There
    is no separate background layer and no special case for the argmax winner, so an
    electrode's appearance says what it expresses rather than which column happened to
    be largest, and the two layers can no longer disagree.
    """
    pl, ok = _scene(side, d, zoom)
    sel = ok & (d["lab"] == j)
    actor = None
    if ok.sum():
        w = d["Gn"][ok, j]                     # EVERY electrode's loading on THIS cluster
        base = np.array(cluster_col(j, d["k"]))
        cloud = pv.PolyData(d["xyz"][ok])
        rgba = np.empty((len(w), 4), np.uint8)
        rgba[:, :3] = np.clip(255 * loading_rgb(w, d["k"], base), 0, 255).astype(np.uint8)
        rgba[:, 3] = np.clip(255 * loading_alpha(w), 0, 255).astype(np.uint8)
        cloud["rgba"] = rgba
        cloud["r"] = BG_RADIUS * (1.02 + 1.20 * loading_t(w, d["k"]))
        g = cloud.glyph(orient=False, scale="r",
                        geom=pv.Sphere(radius=1.0, theta_resolution=12,
                                       phi_resolution=12))
        actor = pl.add_mesh(g, scalars="rgba", rgba=True)
    img = pl.screenshot(return_img=True, transparent_background=True)
    if actor is not None:
        pl.remove_actor(actor)
    return _crop_alpha(np.asarray(img)), int(sel.sum())


# ---- the patient bar under each cluster ---------------------------------------
def patient_colours(d):
    """One colour per patient, stable across the whole figure.

    Taken from tab20 + tab20b rather than the hue circle, because the hue circle is
    already spoken for by the cluster palette and two hue-indexed schemes in one figure
    read as one scheme. 40 colours for 27 patients, so no two share one.
    """
    pats = sorted(set(d["patient"]))
    pal = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return {p: pal[i % len(pal)] for i, p in enumerate(pats)}


def patient_bar(ax, d, j, pcol):
    """Who this cluster is made of: one segment per patient, widest first.

    A cluster that is one patient's electrode strip is a single block of colour here,
    and a cluster drawn from the cohort is a fine stripe - which is the whole judgement
    panel D makes across K, made per-cluster and put where the cluster is.
    """
    sel = d["lab"] == j
    n = int(sel.sum())
    vc = pd.Series(d["patient"][sel]).value_counts()        # descending
    x = 0.0
    for p, c in vc.items():
        ax.barh(0, c / n, left=x, height=1.0, color=pcol[p], lw=0.4, ec="white")
        x += c / n
    top = float(vc.iloc[0]) / n
    ax.set_xlim(0, 1); ax.set_ylim(-0.62, 0.62)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.text(0.0, -0.95, f"{len(vc)} patient{'s' if len(vc) != 1 else ''}",
            transform=ax.get_xaxis_transform(), fontsize=6.4, color=MUTED,
            ha="left", va="top")
    ax.text(1.0, -0.95, f"top {100*top:.0f}%", transform=ax.get_xaxis_transform(),
            fontsize=6.4, color=RED if top > DANGER else MUTED, ha="right", va="top")


# ---- panel A -----------------------------------------------------------------
def panel_A(ax, fset, k):
    hv = pd.read_csv(HELDOUT)
    hv = hv[(hv.method == "cnmf") & (hv.scheme == "home")]
    for fs in FSETS:
        s = (hv[hv.feature_set == fs].groupby("k")["var_explained"]
             .agg(["mean", "std"]).reset_index().sort_values("k"))
        if s.empty:
            continue
        me = fs == fset
        ax.plot(s.k, s["mean"], "-o", ms=3.6 if me else 2.0, lw=2.1 if me else 1.0,
                color=INK if me else MUTED, alpha=1.0 if me else 0.50,
                zorder=4 if me else 2,
                label=FS_SHORT[fs] + ("   <- this figure" if me else ""))
        pk = s.loc[s["mean"].idxmax()]
        if me:
            ax.fill_between(s.k, s["mean"] - s["std"], s["mean"] + s["std"],
                            color=INK, alpha=0.12, lw=0, zorder=3)
            ax.axvline(k, color=RED, ls="--", lw=1.1, zorder=1)
            ax.annotate(f"K = {k}", xy=(k, s["mean"].min()), xytext=(3, 0),
                        textcoords="offset points", fontsize=8, color=RED,
                        ha="left", va="bottom")
        ax.plot([pk.k], [pk["mean"]], marker="v", ms=5.0 if me else 3.4,
                color=GREEN if me else MUTED, alpha=1.0 if me else 0.55,
                zorder=5, lw=0)
    ax.set_xlabel("K (components)", fontsize=9)
    ax.set_ylabel("held-out variance explained", fontsize=9)
    ax.legend(fontsize=7.6, frameon=False, loc="lower right", handlelength=1.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A  ·  held-out variance vs K, convex NMF, all four feature sets\n"
                 "bi-cross-validated (rows AND columns held out).  "
                 "Marker = each curve's own peak.",
                 fontsize=10.4, loc="left", color=INK, pad=6)


def agreement(M, sd):
    """Where the cluster's own +/-1 SD across electrodes does not span zero.

    |mean| > SD, elementwise, on whatever shape it is handed. This is exactly the
    statement the shaded band makes on the HG line - the band clearing the zero line -
    so the two representations report the same quantity and the caption can say so.

    A cluster of one electrode has SD 0 everywhere and is given NO agreement rather
    than total agreement: there is no dispersion evidence in a sample of one, and
    |mean| > 0 would otherwise mark every bin as agreed.

    WHAT THIS IS NOT. It is not a test - there is no null and no correction over the
    450 or 1350 bins it is evaluated on. And it is not evidence that a cluster
    generalises: electrodes from ONE patient's electrode strip are highly correlated,
    so a cluster that is one person will tend to show MORE agreement here, not less.
    Panel D is the panel that answers that question.
    """
    sd = np.asarray(sd, float)
    return np.abs(np.asarray(M, float)) > np.where(sd > 0, sd, np.inf)


def outline_cells(ax, mask, nt, lw=0.9, color=INK):
    """A blocky border around the True cells of a (nband, ncond*nt) mask.

    Cell (i, j) spans x in [j, j+1] and y in [i-0.5, i+0.5], matching the extent
    draw_mean gives imshow. Only the edges between a True cell and a False one (or the
    panel edge) are drawn, so the result is one outline per region.

    CONDITION BLOCKS ARE NOT ADJACENT. Two cells either side of a condition boundary
    are neighbours in the array and not in time, so the boundary always counts as an
    edge - otherwise a region would appear to run continuously from the end of the
    audio trial into the start of the picture trial.
    """
    nrow, ncol = mask.shape
    segs = []
    for i in range(nrow):
        for j in range(ncol):
            if not mask[i, j]:
                continue
            blk = j // nt
            if j == 0 or (j - 1) // nt != blk or not mask[i, j - 1]:
                segs.append([(j, i - 0.5), (j, i + 0.5)])
            if j == ncol - 1 or (j + 1) // nt != blk or not mask[i, j + 1]:
                segs.append([(j + 1, i - 0.5), (j + 1, i + 0.5)])
            if i == 0 or not mask[i - 1, j]:
                segs.append([(j, i - 0.5), (j + 1, i - 0.5)])
            if i == nrow - 1 or not mask[i + 1, j]:
                segs.append([(j, i + 0.5), (j + 1, i + 0.5)])
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                         zorder=5, capstyle="projecting"))


# ---- the cluster mean, drawn once and reused by B and C ----------------------
def draw_mean(ax, M, d, *, vlim, cmap="RdBu_r", col=None, ylim=None,
              show_axes=False, cond_names=False, sd=None):
    """M is (ncond, nband, nt). One panel, three conditions concatenated.

    concat_hg has a single band, so it is a line; every other feature set is drawn as
    frequency x time, which is the representation it was actually clustered in.
    """
    ncond, nband, nt = M.shape
    if nband == 1:
        ax.axhline(0, color=MUTED, lw=0.5)
        # each condition is its OWN line: one plot across all three blocks draws a
        # connecting segment between points that are neither adjacent in time nor the
        # same condition, which showed as a vertical spike at the block boundary
        for b_ in range(ncond):
            x = np.arange(nt) + b_ * nt
            if sd is not None:
                # +/-1 SD ACROSS THE ELECTRODES, not the SEM. With 50-400 electrodes a
                # cluster the SEM is a hairline and says only that the mean is well
                # estimated, which was never in doubt; SD says whether the cluster is a
                # tight family or a loose one. Same convention, and the same alphas, as
                # lf_centroids.render_hg_centroid, so a paper panel and a run-report
                # chip cannot disagree about what the shading means.
                m_, s_ = M[b_, 0, :], sd[b_, 0, :]
                ax.fill_between(x, m_ - s_, m_ + s_, color=col, alpha=0.22, lw=0)
                ax.plot(x, m_ - s_, color=col, lw=0.45, alpha=0.7)
                ax.plot(x, m_ + s_, color=col, lw=0.45, alpha=0.7)
            ax.plot(x, M[b_, 0, :], color=col, lw=1.25)
        ax.set_xlim(0, ncond * nt - 1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        im = None
    else:
        M2 = M.transpose(1, 0, 2).reshape(nband, ncond * nt)
        im = ax.imshow(M2, aspect="auto", origin="lower", cmap=cmap,
                       vmin=-vlim, vmax=vlim,
                       extent=(0, ncond * nt, -0.5, nband - 0.5))
        if sd is not None:
            # the heatmap counterpart of the HG band: enclose the bins where the
            # cluster mean exceeds its own +/-1 SD across electrodes
            outline_cells(ax, agreement(M2, sd.transpose(1, 0, 2)
                                        .reshape(nband, ncond * nt)), nt)
        ax.set_xlim(0, ncond * nt)
    for b in range(ncond):
        if b:
            ax.axvline(b * nt, color=INK, lw=1.0)
        ax.axvline((b + 0.5) * nt, color="#4a4f55" if nband > 1 else MUTED,
                   lw=0.9, ls=(0, (4, 3)))
        if cond_names:
            ax.text((b + 0.5) * nt, 1.015, COND_LABEL[d["conds"][b]],
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8.4, color=INK)
    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(GREY)
        return im
    ax.set_xticks([(b + f) * nt for b in range(ncond) for f in (0.0, 0.5)]
                  + [ncond * nt])
    ax.set_xticklabels((["0", "50"] * ncond) + ["100"], fontsize=7.4)
    ax.tick_params(labelsize=7.4, colors=MUTED)
    if nband == 1:
        ax.set_ylabel(UNITS[d["fset"]], fontsize=8.4)
    else:
        ax.set_yticks(range(nband))
        ax.set_yticklabels([b.replace("Hz", "") for b in d["bands"]], fontsize=7.0)
        ax.set_ylabel("frequency band (Hz)", fontsize=8.4)
    ax.set_xlabel("% of trial, per condition   ·   dashed = GO cue at 50%",
                  fontsize=8.4)
    for s_ in ax.spines.values():
        s_.set_color(GREY)
    return im


def condition_similarity(M):
    """How alike a cluster's three condition profiles look - mean pairwise Pearson r.

    M is (ncond, nband, nt). Each condition is flattened and correlated against each
    other, and the three r values are averaged. High means the cluster does much the
    same thing whatever the modality; low means it does not.

    IT IS A SORT KEY, NOT A RESULT. Correlation is scale-free, so two conditions with
    the same shape and very different amplitude score high - which is what "looks the
    same" means for a shape you are comparing by eye, and is not what it would mean if
    this were being reported as a measure of modality invariance. Nothing is computed
    from it and nothing is claimed by it; it decides where a block sits on the page.
    """
    P = np.asarray(M, float).reshape(np.shape(M)[0], -1)
    rs = []
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            if P[a].std() > 0 and P[b].std() > 0:
                rs.append(float(np.corrcoef(P[a], P[b])[0, 1]))
    return float(np.mean(rs)) if rs else 0.0


_REF: dict = {}


def reference_solution(k: int):
    """The solution block positions are numbered from, at this K. None if unavailable.

    Evaluated at the SAME K as the figure being drawn rather than at one fixed K, so
    the two sides of every match have equal K by construction - which 1:1 assignment
    requires. It also means the four peak-K figures are each matched to a DIFFERENT
    reference solution (bands5 at 11, at 12, at 14, at 13), so their positions are not
    comparable with one another. Only a set drawn at one shared K is.
    """
    if k in _REF:
        return _REF[k]
    _REF[k] = None
    try:
        rd = load_run(MATCH_REF_FSET, k)
    except SystemExit:
        return None
    rd["fset"] = MATCH_REF_FSET
    C = cube(rd["X"], rd)
    m = np.stack([C[rd["lab"] == j].mean(0) if (rd["lab"] == j).any()
                  else np.zeros(C.shape[1:]) for j in range(k)])
    sim = np.array([condition_similarity(m[j]) for j in range(k)])
    _REF[k] = dict(Gn=rd["Gn"], lab=rd["lab"], order=np.argsort(-sim, kind="stable"))
    return _REF[k]


def match_clusters(a, b):
    """Match solution a's clusters 1:1 onto solution b's. Returns (map, jaccard, basis).

    LOADINGS WHERE POSSIBLE, OVERLAP OTHERWISE. Two graded solutions are matched on the
    correlation between their loading columns across electrodes, which uses the whole
    membership rather than the argmax that discards it. If either side is a hard
    partition - k-means, Ward - there are no loadings and the score falls back to the
    number of electrodes two clusters share. Both are solved identically, by Hungarian
    assignment on the K x K score matrix, so the two rules differ in what they score
    and not in how it is resolved.

    Requires the same electrodes in the same order on both sides, which every run here
    has, and equal K. The returned Jaccard is always computed on the HARD labels
    whichever basis was used, so the quality number means one thing everywhere.
    """
    from scipy.optimize import linear_sum_assignment
    (Ga, la), (Gb, lb) = a, b
    K = int(la.max()) + 1
    if int(lb.max()) + 1 != K:
        raise ValueError("match_clusters needs equal K on both sides")
    if Ga is not None and Gb is not None:
        A = Ga - Ga.mean(0)
        B = Gb - Gb.mean(0)
        A = A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
        B = B / np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-12)
        S, basis = A.T @ B, "loading correlation"
    else:
        S = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                S[i, j] = float(np.sum((la == i) & (lb == j)))
        basis = "shared electrodes"
    rows, cols = linear_sum_assignment(-S)
    m = np.empty(K, int)
    m[rows] = cols
    inter = np.array([np.sum((la == i) & (lb == m[i])) for i in range(K)], float)
    union = np.array([np.sum((la == i) | (lb == m[i])) for i in range(K)], float)
    return m, inter / np.maximum(union, 1.0), basis


def block_order(means, d=None, how=None):
    """Cluster ids in the order their blocks are drawn.

    A DISPLAY ORDER AND NOTHING ELSE. Cluster ids, colours, panel D, the loadings and
    every CSV keep their own numbering, so c3 is c3 everywhere - it just appears in a
    different position. Renumbering the clusters to match the page would have broken
    the correspondence with the visualizer and with every file already written.
    """
    how = BLOCK_ORDER if how is None else how
    K = len(means)
    sim = np.array([condition_similarity(means[j]) for j in range(K)])
    own = np.argsort(-sim, kind="stable")
    if how == "cluster":
        return np.arange(K), sim, "numeric order", None
    if how != "matched" or d is None or d.get("fset") == MATCH_REF_FSET:
        why = ("this run defines the sequence" if d is not None
               and d.get("fset") == MATCH_REF_FSET else "this run's own")
        return own, sim, f"condition similarity, {why}", None
    ref = reference_solution(K)
    if ref is None:
        return own, sim, ("condition similarity - "
                          f"no {MATCH_REF_FSET} solution at K={K} to match to"), None
    m, jac, basis = match_clusters((d["Gn"], d["lab"]), (ref["Gn"], ref["lab"]))
    inv = np.full(K, -1, int)
    inv[m] = np.arange(K)                       # reference cluster -> this run's cluster
    if (inv < 0).any():                         # cannot happen with a 1:1 assignment
        return own, sim, "condition similarity - the match was not 1:1", None
    return (inv[ref["order"]], sim,
            f"matched to {MATCH_REF_FSET} ({MATCH_REF_METHOD}) by {basis}",
            (m, jac, basis))


# ---- the trial strip: what was on the screen -------------------------------
def _icon_axes(host, x0, x1, y0=0.13, h=0.70, pad=0.16):
    """A square drawing area centred in [x0, x1] of the host, in host axes fraction.

    Square by construction: with equal aspect the axes box shrinks to a square inside
    whatever rectangle it is given, so an icon cannot be stretched by the strip being
    wide and short.
    """
    w = (x1 - x0) * (1 - 2 * pad)
    ax = host.inset_axes([x0 + (x1 - x0) * pad, y0, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_alpha(0)
    return ax


def _glyph_path(ax, ch, col, frac=0.74, weight="bold"):
    """A character drawn as a PATH scaled to the axes, not as text at a point size.

    Text would be a fixed number of points and would need re-tuning every time a panel
    changed size; a path scales with the box and cannot be left the wrong size by a
    layout change nobody re-checked.
    """
    tp = TextPath((0, 0), ch, size=1.0, prop=FontProperties(weight=weight))
    b = tp.get_extents()
    if b.width <= 0 or b.height <= 0:
        return
    sc = frac / max(b.width, b.height)
    tr = (Affine2D()
          .translate(-b.x0 - b.width / 2.0, -b.y0 - b.height / 2.0)
          .scale(sc)
          .translate(0.5, 0.5))
    ax.add_patch(PathPatch(tr.transform_path(tp), fc=col, ec="none", zorder=3))


def icon_audio(ax, col=INK):
    ax.add_patch(Polygon([(0.10, 0.36), (0.30, 0.36), (0.52, 0.13), (0.52, 0.87),
                          (0.30, 0.64), (0.10, 0.64)],
                         closed=True, fc=col, ec=col, lw=0.6, zorder=3))
    for r in (0.15, 0.25, 0.35):
        ax.add_patch(Arc((0.55, 0.50), 2 * r, 2 * r, theta1=-52, theta2=52,
                         ec=col, lw=1.3, zorder=3))


def icon_reading(ax, col=INK):
    for y, x1 in ((0.76, 0.90), (0.60, 0.94), (0.44, 0.82), (0.28, 0.56)):
        ax.plot([0.10, x1], [y, y], color=col, lw=1.6, solid_capstyle="round",
                zorder=3)


def icon_picture(ax, col=INK):
    """The picture stimulus - the crop if it is there, a framed-image glyph if not."""
    if STIM_PICTURE_PNG.exists():
        try:
            ax.imshow(plt.imread(str(STIM_PICTURE_PNG)), extent=(0, 1, 0, 1),
                      zorder=3, interpolation="antialiased")
            return
        except Exception:
            pass                                  # fall through to the glyph
    ax.add_patch(Rectangle((0.08, 0.18), 0.84, 0.64, fill=False, ec=col, lw=1.5,
                           zorder=3))
    ax.add_patch(Circle((0.30, 0.66), 0.075, fc=col, ec="none", zorder=3))
    ax.add_patch(Polygon([(0.11, 0.21), (0.40, 0.56), (0.55, 0.41), (0.75, 0.62),
                          (0.89, 0.44), (0.89, 0.21)],
                         closed=True, fc=col, ec="none", alpha=0.85, zorder=3))


STIM_ICON = {"audio": icon_audio, "picture": icon_picture, "reading": icon_reading}


def trial_strip(ax, d):
    """Two screens per condition: the stimulus, then the response cue.

    Positioned to the SAME x as the schematic below it - stimulus over 0-50% of the
    block, "?" over 50-100% - so the boundary between the two boxes falls exactly on
    the dashed GO-cue line.

    THE FIXATION SCREEN IS DELIBERATELY ABSENT. The paradigm runs fixation -> stimulus
    -> "?", but the warp is proportions (0.0, 0.5, 0.5): the fixation phase is given no
    time bins, so it occupies no part of these panels. The baseline (-0.6 to -0.1 s)
    that every dB value is expressed against does come from it.
    """
    ncond = len(d["conds"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for b, cond in enumerate(d["conds"]):
        x0 = b / ncond
        half = 1.0 / (2 * ncond)
        for k_, (bx0, bx1) in enumerate(((x0, x0 + half), (x0 + half, x0 + 2 * half))):
            ax.add_patch(Rectangle((bx0 + 0.004, 0.10), (bx1 - bx0) - 0.008, 0.78,
                                   fc="white", ec=GREY, lw=0.9, zorder=1))
            a = _icon_axes(ax, bx0, bx1)
            if k_ == 0:
                STIM_ICON[cond](a)
            else:
                _glyph_path(a, "?", INK, frac=0.66)
        ax.text(x0 + half, 1.03, COND_LABEL[cond], ha="center", va="bottom",
                fontsize=8.4, color=INK, transform=ax.transAxes)
    # centred on the FIRST pair of boxes: a box is half a block wide, so their centres
    # are at half/2 and 1.5*half - not at 0.5/ncond and 1.5/ncond, which is half a box
    # to the right and put "response" under the picture stimulus
    half = 1.0 / (2 * ncond)
    ax.text(half / 2, 0.03, "stimulus", ha="center", va="top", fontsize=6.6,
            color=MUTED, transform=ax.transAxes)
    ax.text(1.5 * half, 0.03, "response", ha="center", va="top", fontsize=6.6,
            color=MUTED, transform=ax.transAxes)


# ---- panel C: the key --------------------------------------------------------
def schematic(d, vlim):
    """An idealised response with this feature set's exact geometry.

    Deliberately not any real cluster, so nobody reads the key as a result: a
    low-frequency decrease and a high-frequency increase, both locked to the GO cue,
    with the response growing across audio -> picture -> reading purely so the three
    blocks are distinguishable.
    """
    ncond, nband, nt = len(d["conds"]), len(d["bands"]), d["nt"]
    tt = np.linspace(0, 1, nt)
    M = np.zeros((ncond, nband, nt))
    for c in range(ncond):
        amp = 0.6 + 0.2 * c
        rise = np.exp(-((tt - 0.62) ** 2) / (2 * 0.11 ** 2))
        early = np.exp(-((tt - 0.30) ** 2) / (2 * 0.09 ** 2))
        for b, name in enumerate(d["bands"]):
            f = band_centre(name)
            if nband == 1:
                M[c, b] = amp * vlim * (0.85 * rise + 0.25 * early)
            elif f < 30:
                M[c, b] = -amp * vlim * 0.75 * rise
            elif f < 70:
                M[c, b] = -amp * vlim * 0.25 * rise
            else:
                M[c, b] = amp * vlim * (0.55 + 0.35 * (f > 150)) * rise
    return M


def panel_C(ax_strip, ax, d, vlim, ylim):
    # the condition names live on the strip now - they belong to the screens, and
    # printing them twice one above the other reads as two different things
    draw_mean(ax, schematic(d, vlim), d, vlim=vlim, col=INK, ylim=ylim,
              show_axes=True, cond_names=False)
    trial_strip(ax_strip, d)
    ax_strip.set_title("C  ·  how to read a cluster mean   —   schematic, not data",
                       fontsize=10.4, loc="left", color=INK, pad=15)
    ax.annotate("GO cue", xy=(0.5 * d["nt"], 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -13), textcoords="offset points", ha="center",
                fontsize=7.4, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.4))


# ---- panel D: is a cluster a response type, or one patient? -------------------
def patient_composition(d, n_perm: int = N_PERM, seed: int = 0):
    """Per cluster at the published K: how many patients, and the largest one's share.

    Against a SIZE-MATCHED RANDOM DRAW from the same cohort - the same number of
    electrodes, sampled without replacement from the same electrodes, so the null
    inherits the cohort's own patient imbalance rather than assuming it away.
    """
    lab, k = d["lab"], d["k"]
    codes, _ = pd.factorize(d["patient"])
    P = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(k):
        sel = lab == j
        n = int(sel.sum())
        c = np.bincount(codes[sel], minlength=P)
        top = int(c.max())
        a, b = [], []
        for _ in range(n_perm):
            e = np.bincount(codes[rng.choice(len(codes), n, replace=False)],
                            minlength=P)
            a.append(int((e > 0).sum())); b.append(e.max() / n)
        rows.append(dict(cluster=j, n=n, n_patients=int((c > 0).sum()),
                         top_n=top, top_share=top / n, top_patient=int(c.argmax()),
                         null_patients=float(np.mean(a)),
                         null_share=float(np.mean(b))))
    return pd.DataFrame(rows)


def generalization_by_k(d):
    """At EVERY K the run was swept at: how much of the solution is one patient.

    Two numbers per K, because they answer different questions and can disagree:

      frac_clusters    the share of clusters that are more than DANGER one patient
      frac_electrodes  the share of ELECTRODES sitting in those clusters

    Two of twenty small clusters failing is not the same as two of five big ones, and
    only the second number tells them apart.

    This is the panel that chooses a K. Held-out variance rises and then falls slowly,
    so it can nominate a peak but it cannot see this failure at all: splitting a cohort
    until each patient has their own component fits the held-out data perfectly well.
    """
    codes, _ = pd.factorize(d["patient"])
    P = int(codes.max()) + 1
    rows = []
    for f in sorted((d["run"] / "loadings_by_k").glob("G_k*.npy")):
        k = int(f.stem.split("_k")[-1])
        G = np.load(f).astype(float)
        if G.shape[0] != len(codes):
            continue
        lab = G.argmax(1)                    # argmax is invariant to the row rescale
        share, size = [], []
        for j in range(G.shape[1]):
            sel = lab == j
            n = int(sel.sum())
            if not n:
                continue
            share.append(np.bincount(codes[sel], minlength=P).max() / n)
            size.append(n)
        share, size = np.array(share), np.array(size)
        dom = share > DANGER
        rows.append(dict(k=k, n_clusters=len(share), n_dominated=int(dom.sum()),
                         frac_clusters=float(dom.mean()),
                         frac_electrodes=float(size[dom].sum() / size.sum()),
                         median_share=float(np.median(share))))
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def panel_D(ax, d, gk, fset):
    K = d["k"]
    ax.plot(gk.k, 100 * gk.frac_clusters, "-o", ms=3.4, lw=1.9, color=RED,
            label=f"clusters that are >{100*DANGER:.0f}% one patient", zorder=4)
    ax.plot(gk.k, 100 * gk.frac_electrodes, "-o", ms=2.8, lw=1.4, color="#e08214",
            label="electrodes sitting in those clusters", zorder=3)
    ax.axvline(K, color=INK, ls="--", lw=1.1, zorder=2)
    ax.set_xlabel("K (components)", fontsize=9)
    ax.set_ylabel("% of the solution that is\none patient", fontsize=8.6)
    ax.set_ylim(0, 102)
    ax.tick_params(labelsize=7.6, colors=MUTED)
    ax.spines[["top"]].set_visible(False)

    # the held-out curve on the same x, faint, because the whole point is that the two
    # criteria pull in opposite directions and the reader has to see both at once
    hv = pd.read_csv(HELDOUT)
    hv = hv[(hv.method == "cnmf") & (hv.scheme == "home") & (hv.feature_set == fset)]
    if not hv.empty:
        sv = hv.groupby("k")["var_explained"].mean().reset_index().sort_values("k")
        ax2 = ax.twinx()
        ax2.plot(sv.k, sv["var_explained"], "-", lw=1.4, color=MUTED, alpha=0.55,
                 zorder=1)
        ax2.set_ylabel("held-out variance", fontsize=8.2, color=MUTED)
        ax2.tick_params(labelsize=7.2, colors=MUTED)
        ax2.spines[["top"]].set_visible(False)
        pk = int(sv.loc[sv.var_explained.idxmax(), "k"])
        ax2.plot([pk], [sv.var_explained.max()], marker="v", ms=5.0, color=GREEN, lw=0)

    row = gk[gk.k == K]
    nd = int(row.n_dominated.iloc[0]) if len(row) else 0
    fe = float(row.frac_electrodes.iloc[0]) if len(row) else 0.0
    ax.legend(fontsize=7.0, frameon=False, loc="upper left")
    ax.set_title(f"D  ·  does this K generalize?   at K={K}, {nd} of {K} clusters are "
                 f">{100*DANGER:.0f}% one patient",
                 fontsize=10.0, loc="left", pad=8,
                 color=RED if nd else INK)


# ---- FIGURE 1 ----------------------------------------------------------------
def figure_1(fset: str, k: int | None):
    t0 = time.time()
    for _pl, _ in _PLOTTER.values():
        _pl.close()
    _PLOTTER.clear(); _BGGLYPH.clear()
    d = load_run(fset, k)
    d["fset"] = fset
    K = d["k"]
    C = cube(d["X"], d)
    means = np.stack([C[d["lab"] == j].mean(0) for j in range(K)])
    # SD across the electrodes of each cluster. A singleton cluster has no SD; zero is
    # the honest value there and it draws as no band, which is a statement about n.
    sds = np.stack([C[d["lab"] == j].std(0, ddof=1) if (d["lab"] == j).sum() > 1
                    else np.zeros_like(means[j]) for j in range(K)])
    vlim = float(np.percentile(np.abs(means), 99.0))
    # the shared y range spans mean +/- SD, or the band clips on the loosest cluster
    ylim = (float((means - sds).min()) * 1.06, float((means + sds).max()) * 1.06)
    # one number per cluster: the share of bins where the mean clears its own SD
    agree_frac = np.array([agreement(means[j], sds[j]).mean() for j in range(K)])
    # and the order the blocks are laid out in - display only, ids are unchanged
    order, cond_sim, order_note, match = block_order(means, d)
    print(f"  block order: {order_note}")
    print(f"  {fset}  K={K}  n={len(d['X'])}  patient composition, every K ...")
    pc = patient_composition(d)
    gk = generalization_by_k(d)
    pcol = patient_colours(d)

    nrow = int(np.ceil(K / UPR))
    # hcen is chosen so a render at RENDER_PX's aspect exactly fills its half of the
    # block: any larger and the brains are letterboxed, any smaller and the box crops
    hcen, hbar = 0.86, 0.15
    fig_h = 3.4 + nrow * 4.75
    fig = plt.figure(figsize=(17.6, fig_h), dpi=190)
    gs = GridSpec(1 + nrow, UPR * UCOL, figure=fig,
                  height_ratios=[1.72] + [2.62] * nrow,
                  hspace=0.52, wspace=0.30, left=0.045, right=0.985,
                  top=1.0 - 0.95 / fig_h, bottom=0.022)
    R1 = 1                                                  # row 0 is the header

    fig.suptitle(f"FIG 1{TAG[fset]}   ·   convex NMF   ·   {FS_LABEL[fset]}   ·   "
                 f"K = {K}   ·   {len(d['X'])} electrodes, {d['n_patients']} patients",
                 x=0.045, y=1.0 - 0.28 / fig_h, ha="left", fontsize=15.5, color=INK)

    panel_A(fig.add_subplot(gs[0, 0:8]), fset, K)
    cC = GridSpecFromSubplotSpec(2, 1, gs[0, 9:15], height_ratios=[0.44, 1.0],
                                 hspace=0.06)
    panel_C(fig.add_subplot(cC[0]), fig.add_subplot(cC[1]), d, vlim, ylim)
    panel_D(fig.add_subplot(gs[0, 17:24]), d, gk, fset)

    tops, nrend, first_brain = [], 0, None
    for pos, j in enumerate(order):
        r, c = divmod(pos, UPR)
        blk = GridSpecFromSubplotSpec(
            3, 2, gs[R1 + r, c * UCOL:(c + 1) * UCOL],
            height_ratios=[hcen, 1.0, hbar], hspace=0.10, wspace=0.02)
        axc = fig.add_subplot(blk[0, :])          # the mean, twice a render's width
        draw_mean(axc, means[j], d, vlim=vlim, col=cluster_col(j, K), ylim=ylim,
                  sd=sds[j])
        if c == 0 and len(d["bands"]) == 1:
            axc.set_yticks([round(ylim[0] + 0.08 * (ylim[1] - ylim[0]), 1), 0.0,
                            round(ylim[1] - 0.08 * (ylim[1] - ylim[0]), 1)])
            axc.tick_params(labelsize=6.8, colors=MUTED, length=2)
            axc.set_ylabel(UNITS[d["fset"]], fontsize=7.4)
        elif c == 0:
            axc.set_yticks(range(len(d["bands"])))
            axc.set_yticklabels([b_.replace("Hz", "") for b_ in d["bands"]],
                                fontsize=6.4)
            axc.tick_params(labelsize=6.4, colors=MUTED, length=2)
            axc.set_ylabel("Hz", fontsize=7.4)
        row = pc[pc.cluster == j].iloc[0]
        warn = "  ·  " + f"{100*row.top_share:.0f}% one patient" \
            if row.top_share > DANGER else ""
        axc.set_title(f"c{j}   n={int(row.n)}   agree {100*agree_frac[j]:.0f}%{warn}",
                      fontsize=8.6,
                      color=RED if row.top_share > DANGER else cluster_col(j, K),
                      pad=2.6, loc="left")
        for i_, side in enumerate(("L", "R")):
            axb = fig.add_subplot(blk[1, i_])
            if first_brain is None:
                first_brain = axb
            img, nsel = render_hemi(side, d, j)
            nrend += 1
            axb.imshow(img)
            axb.set_xticks([]); axb.set_yticks([])
            for s_ in axb.spines.values():
                s_.set_visible(False)
            axb.text(0.03, 0.03, f"{side}  {nsel}", transform=axb.transAxes,
                     ha="left", va="bottom", fontsize=6.8, color=MUTED)
        # who this cluster is made of, directly under its own two renders
        patient_bar(fig.add_subplot(blk[2, :]), d, j, pcol)
        tops.append(axc)

    bb = tops[0].get_position()
    fig.text(0.045, bb.y1 + 0.30 / fig_h,
             "B  ·  one block per cluster, ordered by how ALIKE THE THREE CONDITIONS "
             "LOOK",
             fontsize=10.4, color=INK)

    # THE LOADING KEY, in the left margin, vertical, level with the first block's
    # renders and directly under that block's centroid. It reads as a legend for the
    # column of brains it sits beside rather than as a second colour bar floating in
    # the header. Tick labels go on the right so the rotated caption can use the left.
    bb2 = first_brain.get_position()
    kw = 0.008
    kx = 0.012
    kh = bb2.height * 0.66
    ky = bb2.y0 + (bb2.height - kh) / 2.0
    kax = fig.add_axes([kx, ky, kw, kh])
    lo, hi = 1 / K, max(LOAD_CAP, COLOR_FULL) * 1.12
    tt = np.linspace(lo, hi, 256)
    ramp = loading_rgb(tt, K, np.array(cluster_col(int(order[0]), K)))
    kax.imshow(ramp[:, None, :], aspect="auto", origin="lower", extent=(0, 1, lo, hi))
    kax.set_xticks([])
    kax.set_yticks([lo, COLOR_FULL, LOAD_CAP])
    kax.set_yticklabels(["1/K", f"{COLOR_FULL:.2f}", f"{LOAD_CAP:.2f}"],
                        fontsize=5.8, color=MUTED)
    kax.yaxis.tick_right()
    kax.tick_params(length=2, colors=MUTED, pad=1.4)
    kax.set_ylabel("loading on this cluster", fontsize=6.2, color=MUTED, labelpad=1.5)
    for s_ in kax.spines.values():
        s_.set_color(GREY)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"FIG1{TAG[fset]}_{fset}_cnmf_K{K}.png"
    save_png(fig, p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    save_text(pc.to_csv(index=False), p.with_name(p.stem + "_patients.csv"))
    save_text(gk.to_csv(index=False), p.with_name(p.stem + "_generalization.csv"))
    if match is not None:
        m_, jac_, _ = match
        save_text(pd.DataFrame(dict(
            cluster=np.arange(K), reference_cluster=m_, jaccard=np.round(jac_, 4),
            block_position=[int(np.where(order == j)[0][0]) for j in range(K)]
        )).to_csv(index=False), p.with_name(p.stem + "_matching.csv"))
    caption_1(p.with_name(p.stem + "_caption.txt"), d, vlim, means, pc, gk, sds,
              agree_frac, order, cond_sim, order_note, match)
    print(f"  {nrend} renders, {time.time()-t0:.0f}s  -> {p.name}")
    return p, d, pc


# ---- the caption -------------------------------------------------------------
def caption_1(path: Path, d, vlim, means, pc, gk, sds, agree_frac, order, cond_sim,
              order_note, match):
    fset, K = d["fset"], d["k"]
    n = len(d["X"])
    sizes = np.bincount(d["lab"], minlength=K)
    ok = np.isfinite(d["xyz"]).all(1)
    nL = int((ok & (d["hemi"] == "L")).sum())
    nR = int((ok & (d["hemi"] == "R")).sum())
    pk = pd.read_csv(PEAKS)
    pk = pk[pk.feature_set == fset].iloc[0]
    hv = pd.read_csv(HELDOUT)
    hv = hv[(hv.method == "cnmf") & (hv.scheme == "home")]
    curve = hv.groupby(["feature_set", "k"])["var_explained"].mean().reset_index()
    peaks = {f: (int(g.loc[g.var_explained.idxmax(), "k"]),
                 float(g.var_explained.max()))
             for f, g in curve.groupby("feature_set")}
    w_own = d["Gn"][np.arange(n), d["lab"]]
    t = loading_t(w_own, K)
    ylim_lo = float((means - sds).min()) * 1.06
    ylim_hi = float((means + sds).max()) * 1.06
    cache = sorted(p.name for p in (ROOT / "outputs" / "_dataset")
                   .glob("concat_source_v*") if p.is_dir())

    L = []
    A = L.append
    A(f"FIG 1{TAG[fset]}   ·   {FS_LABEL[fset]}   ·   convex NMF   ·   K = {K}")
    A("=" * 100)
    A("")
    A("WHAT THE FIGURE IS")
    A("-" * 100)
    A(f"Convex NMF (Ding, Li & Jordan 2010) fitted to {n} electrodes from "
      f"{d['n_patients']} patients, in the")
    A(f"{FS_LABEL[fset]} feature set ({d['X'].shape[1]} features = "
      f"{len(d['conds'])} conditions x {len(d['bands'])} band"
      f"{'s' if len(d['bands']) > 1 else ''} x {d['nt']} time bins).")
    A(f"Cut at K = {K}. Cluster sizes: {sizes.tolist()} "
      f"(largest {100*sizes.max()/n:.1f}% of the cohort).")
    A("")
    A("PROVENANCE")
    A("-" * 100)
    A(f"  run            {d['run'].relative_to(CLUST).as_posix()}")
    A(f"  cohort cache   {cache[-1] if cache else 'unknown'}")
    A(f"  loadings       loadings_by_k/G_k{K:02d}.npy  ({d['G'].shape[0]} x "
      f"{d['G'].shape[1]})")
    A(f"  features       X_train.npy, raw "
      f"{'per-band z-scores' if fset.endswith('z') else 'dB vs baseline'} "
      f"(NOT the unit-norm space cNMF fits in)")
    A(f"  coordinates    {COORDS.relative_to(ROOT).as_posix()}")
    A("  surfaces       outputs/250_recon/fsaverage/meshes/fsaverage_[lr]h.gii")
    A("  render         pyvista/VTK offscreen, material constants read live from")
    A("                 functions/lf_recon_shared_config.py (the visualizer's own)")
    A(f"  built by       00_Paper2_Figures.py --figure 1 --feature-set {fset}")
    A(f"  built on       {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A("")
    A("HOW K WAS CHOSEN")
    A("-" * 100)
    A("Bi-cross-validation (Owen & Perry 2009): both ROWS (electrodes) and COLUMNS")
    A("(features) are held out, so the score can fall as well as rise and the curve can")
    A("turn over. A curve that only rises cannot choose a K. Each feature set is scored")
    A("in its own home space, so heights are comparable WITHIN a curve and the shape")
    A("and the peak are what compare ACROSS curves.")
    A("")
    A("  feature set        peak K    held-out variance at peak")
    for f in FSETS:
        if f in peaks:
            kk, vv = peaks[f]
            A(f"  {f:<18} {kk:>5}    {vv:.4f}"
              + ("   <- this figure" if f == fset else ""))
    A("")
    A(f"This figure is cut at K = {K}"
      + ("" if K == int(pk.k_peak) else
         f"  (NOTE: the peak for this feature set is K = {int(pk.k_peak)}; K was "
         f"overridden on the command line)"))
    A("")
    A("PANEL A - held-out variance vs K")
    A("-" * 100)
    A("All four feature sets, convex NMF only. Each point is the mean over the")
    A("bi-cross-validation folds; the shaded band on the emphasised curve is +/- 1 SD")
    A("across folds. The triangle on each curve marks that curve's own peak. The red")
    A(f"dashed line is the K this figure is cut at ({K}).")
    A("The panel is IDENTICAL in FIG 1a, 1b, 1c and 1d except for which curve is")
    A("emphasised, so the four figures can be laid side by side.")
    A("")
    A("PANEL C - how to read a cluster mean")
    A("-" * 100)
    A("ABOVE IT, A TRIAL STRIP: what was on the screen, over the time it was there.")
    A("Read off the paradigm figure (Figure1_FBM_v1.png). Each condition runs")
    A("fixation -> stimulus -> \"?\", the \"?\" being the response cue. The stimulus box")
    A("spans 0-50% of the block and the \"?\" box spans 50-100%, so the boundary between")
    A("them falls exactly on the dashed GO-cue line.")
    A("")
    A("THE FIXATION SCREEN IS DELIBERATELY ABSENT from the strip. The paradigm has")
    A("three screens per trial; these panels show two, because the warp is proportions")
    A("(0.0, 0.5, 0.5) and the fixation phase is given NO time bins. It has not been")
    A("dropped for tidiness - it occupies no part of the x axis. The baseline every dB")
    A("value is expressed against (-0.6 to -0.1 s) does come from it.")
    A("")
    A("The audio, reading and \"?\" icons are drawn as vectors and stay sharp at any")
    A("print size. The picture stimulus is a real line drawing: if a crop of it exists")
    A(f"at {STIM_PICTURE_PNG.name} in the paper_figures/assets/ folder it is used, and")
    A("otherwise a framed-image glyph stands in, so the figure builds either way.")
    A("")
    A("BELOW THE STRIP, A SCHEMATIC. It is not any real cluster and contains no data - a low-frequency")
    A("decrease and a high-frequency increase locked to the GO cue, with amplitude")
    A("growing across the three conditions purely so the blocks are distinguishable.")
    A("It exists to name the parts: the condition over each block, the GO cue at 50%,")
    A("the percent-of-trial axis, and " +
      ("the dB axis." if len(d["bands"]) == 1 else "the frequency band edges on y."))
    A("")
    A("PANEL D - does this K generalize?")
    A("-" * 100)
    A("Computed at EVERY K the run was swept at, not only the published one. For each K,")
    A("the electrodes are assigned by argmax of that K's loadings and two numbers are")
    A("taken:")
    A("")
    A(f"  red     the share of CLUSTERS more than {100*DANGER:.0f}% of whose electrodes come from")
    A("          one patient")
    A("  orange  the share of ELECTRODES sitting inside those clusters")
    A("")
    A("They answer different questions and can disagree: two of twenty small clusters")
    A("failing is not the same as two of five big ones, and only the second number tells")
    A("them apart. The grey curve on the right axis is panel A's held-out variance for")
    A("this feature set, on the same x.")
    A("")
    A("WHY BOTH CURVES HAVE TO BE ON ONE PANEL. Held-out variance cannot see this")
    A("failure. Splitting a cohort until each patient has their own component fits")
    A("held-out data perfectly well - the components are real, they are just components")
    A("of individuals rather than of a population. So the held-out peak nominates a K")
    A("and this curve says what that K costs; where they disagree is a decision, not a")
    A("computation.")
    A("")
    row = gk[gk.k == K]
    if len(row):
        r0 = row.iloc[0]
        A(f"  at K = {K}:  {int(r0.n_dominated)} of {int(r0.n_clusters)} clusters "
          f"({100*r0.frac_clusters:.0f}%), holding "
          f"{100*r0.frac_electrodes:.0f}% of the electrodes")
    lo = gk.loc[gk.frac_electrodes.idxmin()]
    A(f"  best K on this criterion: {int(lo.k)}  "
      f"({int(lo.n_dominated)} of {int(lo.n_clusters)} clusters, "
      f"{100*lo.frac_electrodes:.0f}% of electrodes)")
    A(f"  range over K={int(gk.k.min())}..{int(gk.k.max())}: "
      f"{100*gk.frac_electrodes.min():.0f}% to {100*gk.frac_electrodes.max():.0f}% "
      f"of electrodes in one-patient clusters")
    A("")
    A("PER-CLUSTER PATIENT COMPOSITION at the published K, against a SIZE-MATCHED")
    A("RANDOM DRAW - the same number of electrodes sampled without replacement from the")
    A(f"same cohort, averaged over {N_PERM} permutations, so the reference inherits the")
    A("cohort's own patient imbalance instead of assuming it away. Read it as a CEILING,")
    A("not a floor: a real anatomical cluster SHOULD draw on fewer patients than a random")
    A("draw, because sEEG coverage is patient-specific. What matters is the other end.")
    A("")
    A(f"  patients per cluster        {int(pc.n_patients.min())}-{int(pc.n_patients.max())}"
      f"  (median {int(pc.n_patients.median())})   random draw "
      f"{pc.null_patients.mean():.1f} of {d['n_patients']}")
    A(f"  largest patient's share     {100*pc.top_share.min():.0f}%-"
      f"{100*pc.top_share.max():.0f}%  (median {100*pc.top_share.median():.0f}%)"
      f"   random draw {100*pc.null_share.mean():.0f}%")
    A(f"  clusters with under 5 patients {int((pc.n_patients < 5).sum())} of {K}")
    worst = pc.loc[pc.top_share.idxmax()]
    A(f"  WORST: c{int(worst.cluster)} is {100*worst.top_share:.0f}% one patient "
      f"({int(worst.top_n)} of its {int(worst.n)} electrodes), drawing on "
      f"{int(worst.n_patients)} patient{'s' if worst.n_patients != 1 else ''}.")
    A("")
    A("Written alongside the figure as "
      f"FIG1{TAG[fset]}_{fset}_cnmf_K{K}_patients.csv (per cluster) and")
    A(f"FIG1{TAG[fset]}_{fset}_cnmf_K{K}_generalization.csv (per K).")
    A("")
    A("PANEL B, TOP OF EACH BLOCK - the cluster mean")
    A("-" * 100)
    A(f"BLOCK ORDER IS NOT CLUSTER ORDER.  Order used: {order_note}.")
    A("")
    A("Every run labels the SAME 1693 electrodes in the SAME order, so two solutions at")
    A("one K can be matched 1:1 by Hungarian assignment, and block position can be made")
    A(f"to mean the same cluster in every figure. {MATCH_REF_FSET} ({MATCH_REF_METHOD})")
    A("defines the sequence; its own blocks are ordered by how alike a cluster's three")
    A("condition profiles look, most alike first.")
    A("")
    A("THE REFERENCE IS TAKEN AT THIS FIGURE'S OWN K, so both sides of the match have")
    A("equal K by construction - which 1:1 assignment requires. It follows that a set of")
    A("figures drawn at DIFFERENT K (the held-out peaks: 11, 12, 14, 13) is matched to a")
    A("different reference solution in each figure, and their positions are NOT")
    A("comparable with one another. Only a set drawn at one shared K is.")
    A("")
    A("CLUSTER IDS, COLOURS AND EVERY NUMBER ARE UNCHANGED. c3 is c3 in panel D, in the")
    A("CSVs, in the run report and on the 3-D brain; it just appears somewhere else on")
    A("this page. Renumbering to match the page would have broken that correspondence.")
    A("")
    A("THE ORDER IS A SORT KEY, NOT A RESULT. Correlation is scale-free, so two")
    A("conditions with the same shape and very different amplitude score high. That is")
    A("what \"looks the same\" means for a shape compared by eye, and it is not what it")
    A("would mean if this were reported as a measure of modality invariance. Nothing is")
    A("computed from it and nothing is claimed by it.")
    A("")
    A("  order drawn   " + " ".join(f"c{int(j)}" for j in order))
    A("  similarity    " + " ".join(f"{cond_sim[int(j)]:+.2f}" for j in order))
    if match is not None:
        m_, jac_, basis_ = match
        A("  ref cluster   " + " ".join(f"{int(m_[int(j)]):>2d}" for j in order))
        A("  jaccard       " + " ".join(f"{jac_[int(j)]:.2f}" for j in order))
        A("")
        A(f"  matched on {basis_}; the Jaccard above is always on the HARD labels, so")
        A("  the quality number means one thing whichever basis was used.")
        A(f"  weakest match: c{int(np.argmin(jac_))} at Jaccard {jac_.min():.2f}"
          f"   median {np.median(jac_):.2f}")
        A("")
        A("  A 1:1 assignment ALWAYS returns a partner for every cluster, whether or not")
        A("  one exists. A low Jaccard is a cluster that has no real counterpart and was")
        A("  paired anyway; position implies a correspondence that the number denies.")
        A("  The matched ids and these overlaps are deliberately NOT drawn on the figure")
        A("  - which clusters agree, and which are left out, is Figure 2's subject.")
        A("  Per-cluster values are written alongside as ..._matching.csv.")
    A("")
    A("Every cluster gets ONE BLOCK, three rows deep: its mean on top; underneath, the")
    A("two hemisphere renders of the electrodes that carry it; and under those, WHO THE")
    A("CLUSTER IS MADE OF - one horizontal bar, one colour-coded segment per patient,")
    A("width proportional to that patient's electrodes in the cluster, widest first. A")
    A("cluster that is one patient's electrode strip is a single block of colour there;")
    A("a cluster drawn from the cohort is a fine stripe. Patient colours come from")
    A("tab20 + tab20b and are stable across the whole figure, so a patient is the same")
    A("colour in every cluster - but they are NOT stable across figures 1a-1d, because")
    A("each figure sorts the patients present in its own run.")
    A("")
    A("The mean spans the full block width - twice a single render - so both fit under")
    A(f"it. A cluster whose largest patient is over {100*DANGER:.0f}% is titled in red, "
      "with that share.")
    A("")
    A("The mean is drawn in the representation the feature set was actually clustered")
    A(f"in. Each panel holds the three conditions "
      f"({', '.join(COND_LABEL[c] for c in d['conds'])}) concatenated left to right,")
    A("separated by a solid line. Each condition is time-normalised to "
      f"{d['nt']} bins spanning")
    A("0-100% of its own trial, and the GO cue is at 50% of every block (the dashed")
    A("line) because the warp uses proportions (0.0, 0.5, 0.5). THERE IS NO ABSOLUTE")
    A("TIME AXIS: trials differ in length and were warped to a common proportion, so")
    A("the axis is percent-of-trial and not seconds.")
    if len(d["bands"]) == 1:
        A("")
        A("concat_hg is a single band (70-150 Hz mean), so the cluster mean is a LINE:")
        A(f"the mean over the cluster's electrodes, in {UNITS[fset]}, with a shaded")
        A("band at +/- 1 STANDARD DEVIATION ACROSS THOSE ELECTRODES.")
        A("")
        A("SD, NOT SEM. With 50-400 electrodes a cluster the SEM is a hairline and says")
        A("only that the mean is well estimated, which was never in doubt. The SD says")
        A("whether the cluster is a tight family or a loose one - a wide band on a large")
        A("mean is a cluster whose members do not look like their own centroid. Same")
        A("convention, and the same alphas, as lf_centroids.render_hg_centroid, so a")
        A("paper panel and a run-report chip cannot disagree about what shading means.")
        A("A singleton cluster has no SD and draws no band; that is a statement about n.")
        A("")
        A(f"  median SD over all clusters and time bins   {np.median(sds):.3f} dB")
        A(f"  widest cluster (mean SD)  c{int(sds.mean(axis=(1,2,3)).argmax())}   "
          f"{sds.mean(axis=(1,2,3)).max():.3f} dB")
        A(f"  tightest cluster (mean SD) c{int(sds.mean(axis=(1,2,3)).argmin())}   "
          f"{sds.mean(axis=(1,2,3)).min():.3f} dB")
        A("")
        A(f"All {K} panels share one y-axis ({ylim_lo:+.3f} to {ylim_hi:+.3f}), spanning")
        A("mean +/- SD so the band never clips, so cluster amplitudes and spreads can")
        A("both be compared by eye.")
    else:
        A("")
        A(f"The band sets are drawn as FREQUENCY x TIME. Rows are the "
          f"{len(d['bands'])} bands, lowest at the")
        A(f"bottom: {', '.join(d['bands'])}.")
        A(f"Colour is {UNITS[fset]}, on one diverging scale shared by all {K} clusters,")
        A(f"symmetric about zero at +/- {vlim:.3f} (the 99th percentile of |cluster")
        A("mean| over every cluster, band and time bin). Blue is a decrease, red an")
        A("increase.")
    A("")
    A("DISPERSION ACROSS ELECTRODES is shown on every cluster mean, and it is the")
    A("same quantity in both representations: +/- 1 STANDARD DEVIATION across the")
    A("electrodes in the cluster, at each point.")
    A("")
    A("  concat_hg     a shaded band at mean +/- 1 SD")
    A("  the band sets an OUTLINE enclosing the time-frequency bins where |mean| > SD,")
    A("                i.e. exactly the bins where that band would not span zero")
    A("")
    A("The outline is drawn on CELL EDGES, never between them. A contour would")
    A("interpolate between band centres and imply a resolution a 5- or 15-band grid")
    A("does not have - the reason lf_centroids.render_heatmap_centroid uses SD dots")
    A("instead of contours. Condition blocks are treated as non-adjacent, so a region")
    A("cannot appear to run from the end of one trial into the start of the next.")
    A("")
    A(f"The block title carries the share of bins inside the outline: \"agree NN%\".")
    A("")
    A(f"  agreement per cluster   {100*agree_frac.min():.0f}% - "
      f"{100*agree_frac.max():.0f}%  (median {100*np.median(agree_frac):.0f}%)")
    A(f"  tightest c{int(agree_frac.argmax())} at {100*agree_frac.max():.0f}%   "
      f"loosest c{int(agree_frac.argmin())} at {100*agree_frac.min():.0f}%")
    A("")
    A("WHAT AGREEMENT IS NOT. It is not a test: there is no null, and no correction")
    A(f"over the {means[0].size} bins it is evaluated on per cluster. And it is NOT")
    A("evidence that a cluster generalises - the opposite, if anything. Electrodes on")
    A("one patient's electrode strip are highly correlated, so a cluster that is mostly")
    A("one person will tend to show MORE agreement, not less. Agreement asks whether")
    A("the members look like their own centroid; panel D asks whether the members are")
    A("more than one person. A cluster can score well here and still be one patient.")
    A("A singleton cluster is given NO agreement rather than total agreement, because")
    A("there is no dispersion evidence in a sample of one.")
    A("")
    A("Cluster colours are the visualizer's own palette - hue = 360*j/K, s = 0.62,")
    A("l = 0.52 - so a cluster is the same colour here, in the run report and on the")
    A("3-D brain.")
    A("")
    A("PANEL B, BOTTOM OF EACH BLOCK - where those electrodes are")
    A("-" * 100)
    A("Two 3-D surface renders per cluster, on the fsaverage pial surface, produced by")
    A("pyvista/VTK offscreen with the same material constants the coverage and cluster")
    A("renders use - read live from functions/lf_recon_shared_config.py, not copied, so")
    A("these figures cannot drift away from every other render in the project.")
    A("")
    A("THE LEFT HEMISPHERE IS SEEN FROM THE LEFT AND THE RIGHT FROM THE RIGHT, so each")
    A("is a true lateral view: anterior falls on the LEFT of the left render and on the")
    A("RIGHT of the right one. Drawing both from the same side, as an earlier version")
    A("did, shows one hemisphere from inside the head. Each render holds only that")
    A("hemisphere's electrodes; an electrode never appears in both.")
    A("")
    A(f"  electrodes with fsaverage coordinates   {int(ok.sum())} / {n}")
    A(f"  left render                             {nL}")
    A(f"  right render                            {nR}")
    if d["n_hemi_from_x"]:
        A(f"  side taken from sign(x) because the coordinate table had no hemi   "
          f"{d['n_hemi_from_x']}")
    A("")
    A("COLOUR AND SIZE, THE VISUALIZER'S OWN RULE. This is transcribed from")
    A("clustering_visualizer.html's updateElectrodes(), not invented here, so these")
    A("renders and the brains on the site read the same way.")
    A("")
    A(f"      colour: FULL cluster colour at a loading of {COLOR_FULL:.2f} or above; below")
    A("              that, mixed toward grey 148 and reaching it exactly at a flat 1/K")
    A(f"              mixture -   g = clip((w - 1/K) / ({COLOR_FULL:.2f} - 1/K), 0, 1)")
    A(f"      size:   t = clip((w - 1/K) / ({LOAD_CAP:.2f} - 1/K), 0, 1)")
    A(f"              radius = {BG_RADIUS} * (1.02 + 1.20 * t) mm")
    A(f"      opacity: {ALPHA_FLOOR:.2f} at a loading of zero, rising linearly to fully")
    A(f"              opaque at {LOAD_CAP:.2f} -   a = {ALPHA_FLOOR:.2f} + "
      f"{1-ALPHA_FLOOR:.2f} * clip(w / {LOAD_CAP:.2f}, 0, 1)")
    A("              measured from ZERO, not from 1/K, unlike the other two: this")
    A("              channel exists to sink the uninvolved population into the")
    A("              background, and a floor at 1/K would leave everything below a flat")
    A("              mixture invisible and everything just above it nearly solid")
    A("")
    A("where w is the electrode's convex-NMF loading on this cluster after its loading")
    A("vector has been normalised to sum to 1. THREE CHANNELS FOR THREE QUESTIONS:")
    A("colour says whether the electrode is expressing this component at all, size says")
    A("how strongly, opacity sinks the rest of the cohort into the background. An")
    A("earlier version ramped colour across the whole range instead, so a typical")
    A("loading - the median is around a third - landed a third of the way out of grey")
    A("and every render came out muddy.")
    A("")
    A("EVERY ELECTRODE OF THE HEMISPHERE IS DRAWN BY THIS ONE RULE, applied to its own")
    A("loading on this cluster. There is no separate background layer and no special")
    A("case for the argmax winner: an electrode's appearance says what it expresses,")
    A("not which column happened to be largest. Order-independent transparency (VTK")
    A("depth peeling) is enabled, or a faint sphere in front of a solid one would erase")
    A("it depending only on draw order.")
    A("")
    A("Both are RENDERING choices. They change how a loading is drawn, never the")
    A("loading, the label, or any number reported anywhere.")
    A("")
    A(f"  median t over all {n} electrodes at their own cluster   {np.median(t):.3f}")
    A(f"  electrodes at or above {COLOR_FULL:.2f} (drawn in full cluster colour)   "
      f"{int((w_own >= COLOR_FULL).sum())}  ({100*(w_own >= COLOR_FULL).mean():.1f}%)")
    A(f"  electrodes at or above the {LOAD_CAP:.2f} size cap                     "
      f"{int((w_own >= LOAD_CAP).sum())}  ({100*(w_own >= LOAD_CAP).mean():.1f}%)")
    A(f"  median normalised loading at an electrode's own cluster {np.median(w_own):.3f}")
    A("")
    A("WHAT THIS FIGURE DOES NOT SHOW")
    A("-" * 100)
    A("  - No statistical test of the clustering. Separation against a matched null,")
    A("    anatomical coherence and leave-one-patient-out are reported separately, and")
    A("    are valid at ONE K only because each is scored against a null refitted at")
    A("    that K. At the time of writing they exist for concat_hg and concat_rawds")
    A("    only; concat_bands5 and concat_bands5z have no statistics at any K.")
    A("  - Panel D and the patient bars are about PATIENT COMPOSITION, which is")
    A("    necessary for a cluster to be a population response type and is not")
    A("    sufficient. A cluster can draw on every patient and still not replicate.")
    A(f"  - The {100*DANGER:.0f}% line in panel D is a convention, not a test. It is not")
    A("    calibrated against a null; it is a threshold chosen to be readable. So is")
    A("    the 1-SD line the agreement outline draws.")
    A("  - No comparison against k-means, Ward or archetypal analysis. Panel A is")
    A("    convex NMF only.")
    A("  - The renders collapse depth: an electrode deep in the temporal lobe and one")
    A("    on the lateral surface can overlap in the image.")
    A("  - The cluster label is an argmax of a GRADED loading. The fade is there")
    A("    precisely because that argmax hides how weak most memberships are.")
    save_text("\n".join(L) + "\n", path)


# ---- driver ------------------------------------------------------------------
FIGURES = {1: figure_1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=int, nargs="*", default=sorted(FIGURES),
                    help="which figures to build (default: all)")
    ap.add_argument("--feature-set", nargs="*", default=FSETS)
    ap.add_argument("--k", type=int, default=None,
                    help="override the held-out peak K (applies to every feature set)")
    a = ap.parse_args()
    for nf in a.figure:
        if nf not in FIGURES:
            print(f"!! no figure {nf}; have {sorted(FIGURES)}", file=sys.stderr)
            continue
        print(f"\n=== FIGURE {nf} ===")
        for fs in a.feature_set:
            FIGURES[nf](fs, a.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
