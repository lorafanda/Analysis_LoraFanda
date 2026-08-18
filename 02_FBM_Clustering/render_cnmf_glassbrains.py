#!/usr/bin/env python3
"""
render_cnmf_glassbrains.py — per-component glassbrains for the graded decomposition.

THIS SCRIPT OWNS ONE FOLDER: by_loading/.

    by_condition/   <- notebook 252, argmax, the project look
    by_patient/     <- notebook 252, per-patient colours
    by_loading/     <- HERE. The GRADED view: every contact, opacity and radius
                       scaled by its loading on this component. With ~66% of
                       electrodes having no majority component the argmax discards
                       most of the signal, and this is what it discards. 252 cannot
                       draw it, because it keys off the hard cluster column.

It renders through pyvista while 252 renders through MNE/PySurfer add_foci. They read
the same colour constants and cameras but NOT the same engine, so their output is
visibly different — a cooler surface and smaller dots here. When this script also wrote
by_condition/ the folder no longer said which tool had made its contents; 252 then
skipped those files because they already existed, and FIG C.3 panel B2 kept the wrong
renderer's output without any error. Hence one folder, one owner, enforced at write time.

The camera keys in lf_recon_shared_config are left/right/dorsal/frontal; the filenames the
rest of the project expects are lateral_L/lateral_R/dorsal/frontal, so they are mapped.

    python render_cnmf_glassbrains.py
    python render_cnmf_glassbrains.py --run <run dir> --scale 1
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

VIEW_MAP = {"lateral_L": "left", "lateral_R": "right",
            "dorsal": "dorsal", "frontal": "frontal"}

# Below this a contact contributes nothing visible and only costs a sphere. Uniform
# loading at K=7 is 1/7 = 0.143, so this keeps everything down to about half of uniform.
LOADING_FLOOR = 0.08


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: newest cnmf/concat_hg)")
    ap.add_argument("--scale", type=int, default=2, help="screenshot supersampling")
    # THE ARGMAX OPTION IS GONE ON PURPOSE. by_condition/ belongs to notebook 252,
    # which renders it through MNE/PySurfer Brain.add_foci. This script is a pyvista
    # reimplementation and produces a visibly different brain from the same colour
    # constants. When both wrote by_condition/ the folder said nothing about which
    # tool made its contents, 252 then SKIPPED the files because they already
    # existed, and FIG C.3 panel B2 silently kept the wrong renderer's output.
    #
    # One folder, one owner:
    #     by_condition/  <- 252   (argmax, the project look)
    #     by_patient/    <- 252   (per-patient colours)
    #     by_loading/    <- here  (graded, which 252 cannot produce)
    ap.add_argument("--which", choices=["loading"], default="loading",
                    help="which of the two renders to (re)draw")
    a = ap.parse_args()

    import pyvista as pv
    import matplotlib.pyplot as plt
    import lf_recon_shared as RS
    import lf_recon_shared_config as C

    base = ROOT / "outputs" / "clustering" / "cnmf" / "concat_hg" / "runs"
    rd = Path(a.run) if a.run else sorted(d for d in base.iterdir() if d.is_dir())[-1]
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    csv = next((rd / "recon").glob("*__with_fsaverage.csv"), None)
    if csv is None:
        print("  !! no recon CSV — run make_recon_csv.py first", file=sys.stderr)
        return 1

    df = pd.read_csv(csv).dropna(subset=["x", "y", "z"])
    ccol = next(c for c in df.columns if c.startswith("cluster_"))
    wcols = sorted((c for c in df.columns if c.startswith("w") and c[1:].isdigit()),
                   key=lambda s: int(s[1:]))
    K = len(wcols)
    print(f"  {rd.name}: {len(df)} localised contacts, K={K}")

    lh, rh = RS.load_fsaverage_meshes()
    bounds = (min(lh.bounds[0], rh.bounds[0]), max(lh.bounds[1], rh.bounds[1]),
              min(lh.bounds[2], rh.bounds[2]), max(lh.bounds[3], rh.bounds[3]),
              min(lh.bounds[4], rh.bounds[4]), max(lh.bounds[5], rh.bounds[5]))
    cams = RS.compute_cameras(bounds)
    pal = plt.get_cmap("tab10").colors

    def shot(sel, rgb, weights, out_png, view):
        pl = pv.Plotter(off_screen=True, window_size=C.WINDOW_SIZE)
        for m in (lh, rh):
            pl.add_mesh(m, color=C.BRAIN_COLOR, opacity=C.BRAIN_OPACITY_CLEAN,
                        specular=C.BRAIN_SPECULAR, specular_power=C.BRAIN_SPECULAR_POWER,
                        ambient=C.BRAIN_AMBIENT, diffuse=C.BRAIN_DIFFUSE)
        for (x, y, z), w in zip(sel, weights):
            r = C.DEPTH_RADIUS * (0.45 + 1.15 * float(w))
            pl.add_mesh(pv.Sphere(radius=r, center=(x, y, z)), color=rgb,
                        opacity=float(np.clip(0.15 + 0.85 * w, 0, 1)))
        pl.camera_position = cams[VIEW_MAP[view]]
        pl.reset_camera_clipping_range()
        if out_png.parent.name != "by_loading":
            raise RuntimeError(
                f"refusing to write {out_png.parent.name}/ - this script only owns "
                f"by_loading/; by_condition/ and by_patient/ belong to notebook 252")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_png), transparent_background=C.TRANSPARENT_BG, scale=a.scale)
        pl.close()

    W = df[wcols].to_numpy()
    Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    xyz = df[["x", "y", "z"]].to_numpy()

    # ONE scale for every component. Dividing each component by its OWN maximum makes
    # every panel peak at full radius and full opacity, so a component whose strongest
    # electrode loads 0.84 looks exactly as intense as one whose strongest loads 0.99 —
    # the panels stop being comparable, which is the only reason to put them in a row.
    # Here the spread is small (per-component max 0.84-0.99 against a global 0.99, so at
    # most a 15% intensity change) but the fix costs nothing and removes the trap.
    GMAX = float(max(Wn.max(), 1e-9))

    for j in range(K):
        rgb = tuple(pal[j % 10])
        cd = rd / "recon" / f"cluster_{j:02d}"
        if a.which == "loading":
            # GRADED — what the argmax throws away
            keep = Wn[:, j] > LOADING_FLOOR
            for v in VIEW_MAP:
                shot(xyz[keep], rgb, Wn[keep, j] / GMAX, cd / "by_loading" / f"{v}.png", v)
        m = df[ccol].to_numpy() == j
        keep = Wn[:, j] > LOADING_FLOOR
        print(f"    comp {j}: argmax {int(m.sum())} contacts, graded {int(keep.sum())} "
              f"above {LOADING_FLOOR} loading (max {Wn[:, j].max():.2f} of {GMAX:.2f})")

    print(f"\n  -> {rd / 'recon'}")
    print("  by_loading/ = graded. by_condition/ and by_patient/ are 252's to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
