#!/usr/bin/env python3
"""
render_cnmf_glassbrains.py — per-cluster glassbrains, across every published run.

THIS SCRIPT OWNS THREE FOLDERS: by_loading/, by_cluster/ and spin/.

    by_condition/   <- notebook 252, argmax, the project look
    by_patient/     <- notebook 252, per-patient colours
    by_loading/     <- HERE. The GRADED view: every contact, opacity and radius
                       scaled by its loading on this component. With ~66% of
                       electrodes having no majority component the argmax discards
                       most of the signal, and this is what it discards. 252 cannot
                       draw it, because it keys off the hard cluster column.
    by_cluster/     <- HERE. This script's own argmax copy.
    spin/           <- HERE. The 60-frame turn the HTML report animates.

It renders through pyvista while 252 renders through MNE/PySurfer add_foci. They read
the same colour constants and cameras but NOT the same engine, so their output is
visibly different — a cooler surface and smaller dots here. When this script also wrote
by_condition/ the folder no longer said which tool had made its contents; 252 then
skipped those files because they already existed, and FIG C.3 panel B2 kept the wrong
renderer's output without any error. Hence one folder, one owner, enforced at write time.

EVERY RUN, NOT JUST THE NEWEST cnmf/concat_hg. This used to hard-code that one
directory and take its newest run, so nine of the ten published runs had no by_cluster/
and no spin/ at all — the report's glass row and its rotation were simply blank for
them, with nothing on the page to say why. Runs now come from index.json.

by_loading/ still needs the w0..wK columns, which only a convex-NMF run has. A hard
partition (k-means, Ward, atlas) has no graded membership to draw, so it gets
by_cluster/ and spin/ and is reported as skipping the graded view — rather than
silently computing K=0 and rendering nothing at all, which is what the old code did
when handed such a run.

    python render_cnmf_glassbrains.py --dry-run       # the plan and the render count
    python render_cnmf_glassbrains.py                 # every eligible run
    python render_cnmf_glassbrains.py --published --skip-existing   # fill the gaps
                                                      # the visualizer actually shows
    python render_cnmf_glassbrains.py --run <run dir>
    python render_cnmf_glassbrains.py --method kmeans --feature-set concat_hg
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

CLUST = ROOT / "outputs" / "clustering"
BUNDLE = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz"
          / "manifest.json")

VIEW_MAP = {"lateral_L": "left", "lateral_R": "right",
            "dorsal": "dorsal", "frontal": "frontal"}

# Below this a contact contributes nothing visible and only costs a sphere. Uniform
# loading at K=7 is 1/7 = 0.143, so this keeps everything down to about half of uniform.
LOADING_FLOOR = 0.08

# Measured, not chosen: at 1.0 the brain renders flat grey, at 0.85 it keeps the
# project's warm translucent surface. See the comment in shot().
MAX_SPHERE_OPACITY = 0.85


def _run_dirs(a):
    """Every run with a recon CSV, read from index.json rather than from a glob.

    index.json is what the site and make_coverage_bundle read, so a run that is not
    in it is not published and has no reason to be rendered.
    """
    idx = json.loads((CLUST / "index.json").read_text(encoding="utf-8"))
    runs = idx["runs"] if isinstance(idx, dict) else idx
    # index.json holds every run ever published, including superseded ones. The
    # visualizer only offers what the coverage bundle carries, so --published is the
    # set whose missing renders anyone will actually notice.
    keep = None
    if a.published:
        man = json.loads(BUNDLE.read_text(encoding="utf-8"))
        keep = {x["id"] for x in man["runs"]}
    out = []
    for r in runs:
        rd = CLUST / r["method"] / r["feature_set"] / "runs" / r["run_id"]
        if a.run and Path(a.run).resolve() != rd.resolve():
            continue
        if a.method and r["method"] not in a.method:
            continue
        if a.feature_set and r["feature_set"] not in a.feature_set:
            continue
        if keep is not None and                 f"{r['method']}__{r['feature_set']}__{r['run_id']}" not in keep:
            continue
        if not rd.is_dir() or not (rd / "recon").is_dir():
            continue
        csv = next((rd / "recon").glob("*__with_fsaverage.csv"), None)
        if csv is None:
            continue
        out.append((f"{r['method']}/{r['feature_set']}/{r['run_id']}", rd, csv))
    return out


def _read(csv):
    """(df, cluster column, w columns, cluster ids actually present)."""
    df = pd.read_csv(csv).dropna(subset=["x", "y", "z"])
    # NOT simply the first cluster_* column: a ranked copy sits beside the real one in
    # most runs, and picking it up would relabel every cluster in the output.
    ccol = next((c for c in df.columns
                 if c.startswith("cluster_") and not c.endswith("_ranked")), None)
    wcols = sorted((c for c in df.columns if re.fullmatch(r"w\d+", c)),
                   key=lambda s: int(s[1:]))
    ids = []
    if ccol is not None:
        vals = pd.to_numeric(df[ccol], errors="coerce").dropna().unique()
        ids = sorted({int(v) for v in vals if int(v) >= 0})
    return df, ccol, wcols, ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="one run dir; default is every run")
    ap.add_argument("--method", action="append",
                    help="restrict to a method (repeatable)")
    ap.add_argument("--feature-set", action="append", dest="feature_set",
                    help="restrict to a feature set (repeatable)")
    ap.add_argument("--published", action="store_true",
                    help="only the runs the coverage bundle exposes to the "
                         "visualizer, rather than every run in index.json")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave a folder alone if it already holds files")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the render count, draw nothing")
    ap.add_argument("--scale", type=int, default=2, help="screenshot supersampling")
    ap.add_argument("--spin-frames", type=int, default=60)
    ap.add_argument("--spin-px", type=int, default=460,
                    help="width of each rotation frame; they are embedded in the "
                         "HTML report, so this drives its size")
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
    ap.add_argument("--which", choices=["loading", "cluster", "spin", "all"],
                    default="all",
                    help="which of the three renders to (re)draw")
    a = ap.parse_args()

    targets = _run_dirs(a)
    if not targets:
        print("  no run matched (each needs a recon/*__with_fsaverage.csv)",
              file=sys.stderr)
        return 1

    # ── Plan first. A bare run over every published run is thousands of renders, and
    #    discovering that by watching it crawl is worse than being told up front.
    plan, total = [], 0
    for tag, rd, csv in targets:
        df, ccol, wcols, ids = _read(csv)
        if ccol is None:
            plan.append((tag, rd, csv, None, [], [], "no cluster column"))
            continue
        graded = bool(wcols)
        idx = list(range(len(wcols))) if graded else ids
        # Count what will ACTUALLY be drawn, honouring --skip-existing. A plan that
        # ignores it reports the full 7160 for a run that is already complete, and
        # the number is then worse than no number at all.
        def _have(d):
            return a.skip_existing and d.is_dir() and any(d.iterdir())
        n = 0
        for j in idx:
            cd = rd / "recon" / f"cluster_{j:02d}"
            if a.which in ("loading", "all") and graded and not _have(cd / "by_loading"):
                n += len(VIEW_MAP)
            if a.which in ("cluster", "all") and not _have(cd / "by_cluster"):
                n += len(VIEW_MAP)
            if a.which in ("spin", "all") and                     not _have(rd / "recon" / "spin" / f"cluster_{j:02d}"):
                n += a.spin_frames
        note = "" if graded else "hard partition, no graded view"
        if not n:
            note = (note + "; nothing to draw").lstrip("; ")
        plan.append((tag, rd, csv, ccol, idx, wcols, note))
        total += n

    print(f"  {len(plan)} run(s), {total} renders at --which {a.which}:")
    for tag, rd, csv, ccol, idx, wcols, note in plan:
        mark = "  " if ccol is not None else "!!"
        print(f"  {mark} {tag:<44s} K={len(idx):<3d} {note}")
    if a.dry_run:
        print("  (dry run — nothing drawn)")
        return 0

    import pyvista as pv
    import matplotlib.pyplot as plt
    import lf_recon_shared as RS
    import lf_recon_shared_config as C

    lh, rh = RS.load_fsaverage_meshes()
    bounds = (min(lh.bounds[0], rh.bounds[0]), max(lh.bounds[1], rh.bounds[1]),
              min(lh.bounds[2], rh.bounds[2]), max(lh.bounds[3], rh.bounds[3]),
              min(lh.bounds[4], rh.bounds[4]), max(lh.bounds[5], rh.bounds[5]))
    cams = RS.compute_cameras(bounds)
    pal = plt.get_cmap("tab10").colors

    def shot(sel, rgb, weights, out_png, view, az=None, px=None):
        ws = C.WINDOW_SIZE if px is None else (px, int(px * C.WINDOW_SIZE[1]
                                                      / C.WINDOW_SIZE[0]))
        pl = pv.Plotter(off_screen=True, window_size=ws)
        for m in (lh, rh):
            pl.add_mesh(m, color=C.BRAIN_COLOR, opacity=C.BRAIN_OPACITY_CLEAN,
                        specular=C.BRAIN_SPECULAR, specular_power=C.BRAIN_SPECULAR_POWER,
                        ambient=C.BRAIN_AMBIENT, diffuse=C.BRAIN_DIFFUSE)
        for (x, y, z), w in zip(sel, weights):
            r = C.DEPTH_RADIUS * (0.45 + 1.15 * float(w))
            # NEVER 1.0. A few hundred FULLY OPAQUE spheres change how VTK composites
            # the translucent brain behind them and the mauve surface washes out to a
            # flat grey - which is exactly why the argmax view (w=1) looked wrong while
            # the graded view (w<1) looked right, from this same function. Capped at
            # MAX_SPHERE_OPACITY, measured: 0.85 keeps the warm surface, 1.0 kills it.
            pl.add_mesh(pv.Sphere(radius=r, center=(x, y, z)), color=rgb,
                        opacity=float(np.clip(0.15 + 0.85 * w, 0, MAX_SPHERE_OPACITY)))
        pl.camera_position = cams[VIEW_MAP[view]] if view else cams["left"]
        pl.reset_camera_clipping_range()
        if az is not None:
            pl.camera.azimuth = az        # sweep a full turn from the left view
        if (out_png.parent.name not in ("by_loading", "by_cluster")
                and out_png.parent.parent.name != "spin"):
            raise RuntimeError(
                f"refusing to write {out_png.parent.name}/ - this script owns "
                f"by_loading/, by_cluster/ and spin/; by_condition/ and "
                f"by_patient/ belong to notebook 252")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_png), transparent_background=C.TRANSPARENT_BG,
                      scale=1 if px else a.scale)
        pl.close()

    def done(d):
        return a.skip_existing and d.is_dir() and any(d.iterdir())

    for tag, rd, csv, ccol, idx, wcols, note in plan:
        if ccol is None:
            print(f"\n  {tag}: skipped — {note}")
            continue
        df, _, _, _ = _read(csv)
        lab = pd.to_numeric(df[ccol], errors="coerce").to_numpy()
        xyz = df[["x", "y", "z"]].to_numpy()
        graded = bool(wcols)
        tail = "" if graded else f"  ({note})"
        print(f"\n  {tag}: {len(df)} localised contacts, K={len(idx)}{tail}")

        Wn = GMAX = None
        if graded:
            W = df[wcols].to_numpy()
            Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
            # ONE scale for every component. Dividing each component by its OWN maximum
            # makes every panel peak at full radius and full opacity, so a component
            # whose strongest electrode loads 0.84 looks exactly as intense as one whose
            # strongest loads 0.99 — the panels stop being comparable, which is the only
            # reason to put them in a row. Here the spread is small (per-component max
            # 0.84-0.99 against a global 0.99, so at most a 15% intensity change) but the
            # fix costs nothing and removes the trap.
            GMAX = float(max(Wn.max(), 1e-9))

        for j in idx:
            rgb = tuple(pal[j % 10])
            cd = rd / "recon" / f"cluster_{j:02d}"
            m = lab == j
            if a.which in ("loading", "all") and graded and not done(cd / "by_loading"):
                keep = Wn[:, j] > LOADING_FLOOR
                for v in VIEW_MAP:
                    shot(xyz[keep], rgb, Wn[keep, j] / GMAX,
                         cd / "by_loading" / f"{v}.png", v)
            if a.which in ("cluster", "all") and not done(cd / "by_cluster"):
                # ARGMAX, this script's own copy — by_condition/ stays 252's
                for v in VIEW_MAP:
                    shot(xyz[m], rgb, np.ones(int(m.sum())),
                         cd / "by_cluster" / f"{v}.png", v)
            sd = rd / "recon" / "spin" / f"cluster_{j:02d}"
            if a.which in ("spin", "all") and not done(sd):
                # a full turn for the HTML report, one folder per component
                for fi in range(a.spin_frames):
                    shot(xyz[m], rgb, np.ones(int(m.sum())),
                         sd / f"f{fi:03d}.png", None,
                         az=(360.0 * fi) / a.spin_frames, px=a.spin_px)
            msg = f"    cluster {j}: argmax {int(m.sum())} contacts"
            if graded:
                keep = Wn[:, j] > LOADING_FLOOR
                msg += (f", graded {int(keep.sum())} above {LOADING_FLOOR} loading "
                        f"(max {Wn[:, j].max():.2f} of {GMAX:.2f})")
            print(msg)

        print(f"    -> {rd / 'recon'}")

    print("\n  by_loading/, by_cluster/ and spin/ written.")
    print("  by_condition/ and by_patient/ are 252's to write.")
    print("  spin/ frames are PNG here; pack_spin_frames.py converts them to the")
    print("  .jpg the HTML report fetches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
