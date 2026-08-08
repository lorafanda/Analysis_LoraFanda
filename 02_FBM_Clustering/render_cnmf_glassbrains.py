#!/usr/bin/env python3
"""
render_cnmf_glassbrains.py — per-component glassbrains for the graded decomposition.

Notebook 252 renders these for a normal clustering run, but it needs a Jupyter kernel and
keys off the hard cluster column. This does the same job headlessly, using the project's
own meshes, cameras, colours and window size from lf_recon_shared / _config, so the output
sits beside every other glassbrain in the project and looks the same.

TWO VIEWS PER COMPONENT, and they mean different things:

    by_condition/   the ARGMAX  — contacts whose leading component is this one.
                    Directly comparable to a 252 render of any clustering run.
    by_loading/     the GRADED view — every contact, opacity and radius scaled by its
                    loading on this component. With 59% of electrodes having no majority
                    component the argmax discards most of the signal, and this is what it
                    discards.

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: newest cnmf/concat_hg)")
    ap.add_argument("--scale", type=int, default=2, help="screenshot supersampling")
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
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_png), transparent_background=C.TRANSPARENT_BG, scale=a.scale)
        pl.close()

    W = df[wcols].to_numpy()
    Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    xyz = df[["x", "y", "z"]].to_numpy()

    for j in range(K):
        rgb = tuple(pal[j % 10])
        cd = rd / "recon" / f"cluster_{j:02d}"
        # ARGMAX — what 252 would draw
        m = df[ccol].to_numpy() == j
        for v in VIEW_MAP:
            shot(xyz[m], rgb, np.ones(m.sum()), cd / "by_condition" / f"{v}.png", v)
        # GRADED — what the argmax throws away
        keep = Wn[:, j] > 0.08
        for v in VIEW_MAP:
            shot(xyz[keep], rgb, Wn[keep, j] / max(Wn[:, j].max(), 1e-9),
                 cd / "by_loading" / f"{v}.png", v)
        print(f"    comp {j}: argmax {int(m.sum())} contacts, graded {int(keep.sum())} "
              f"above 0.08 loading")

    print(f"\n  -> {rd / 'recon'}")
    print("  by_condition/ = argmax (comparable to 252) · by_loading/ = graded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
