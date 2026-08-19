#!/usr/bin/env python3
"""
pack_spin_frames.py - make the rotation frames small enough to embed in a report.

The raw pyvista frames are ~100 kB PNGs at 460 px, and a 7-cluster x 60-frame
rotation set is ~42 MB. That cannot go in a self-contained HTML report, and it
cannot go in one git commit either (the pre-commit hook caps at 25 MB).

Two reductions, in this order:

  1. FIXED-BOX CROP. Most of each frame is white margin. The box is the UNION of
     the ink across every frame of a cluster, not per-frame - a per-frame box is
     what made the in-browser rotations jitter, because each frame then has its
     own size and the brain jumps. One box per cluster keeps every frame the same
     size and the brain registered.
  2. JPEG. These are photographic renders with soft gradients; JPEG at 82 is
     visually indistinguishable here and roughly a quarter the size.

    python pack_spin_frames.py                  # every run that has spin/ PNGs
    python pack_spin_frames.py --run <run dir>  # just this one
    python pack_spin_frames.py --keep-png       # don't delete the sources

It used to take the newest cnmf/concat_hg run and nothing else, matching what
render_cnmf_glassbrains.py could produce at the time. Both now cover every run,
which matters because the report fetches .jpg: leaving a run's frames as .png
means its rotation stays blank on the page with the renders sitting on disk.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLUST = ROOT / "outputs" / "clustering"


def union_box(paths, thresh=6):
    """Bounding box of the ink across ALL frames, so every frame crops identically."""
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    for p in paths:
        im = Image.open(p).convert("RGBA")
        a = np.asarray(im)
        # ink = anything not (near-)white and not transparent
        alpha = a[..., 3]
        rgb = a[..., :3].astype(int)
        ink = (alpha > 8) & (rgb.sum(2) < 255 * 3 - thresh * 3)
        ys, xs = np.where(ink)
        if not len(xs):
            continue
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    if x1 < 0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--keep-png", action="store_true")
    a = ap.parse_args()

    if a.run:
        runs = [Path(a.run)]
    else:
        runs = sorted({q.parent.parent for q in CLUST.glob("*/*/runs/*/recon/spin")})
    runs = [r for r in runs if (r / "recon" / "spin").is_dir()]
    if not runs:
        print("  no spin/ anywhere - run render_cnmf_glassbrains.py --which spin",
              file=sys.stderr)
        return 1

    todo = [(r, cd) for r in runs
            for cd in sorted(d for d in (r / "recon" / "spin").iterdir() if d.is_dir())]
    print(f"  {len(runs)} run(s), {len(todo)} cluster folder(s)")

    before = after = 0
    for rd, cd in todo:
        pngs = sorted(cd.glob("f*.png"))
        if not pngs:
            continue
        box = union_box(pngs, a.pad)
        if box is None:
            print(f"  {cd.name}: no ink found, skipped")
            continue
        x0, y0, x1, y1 = box
        x0, y0 = max(0, x0 - a.pad), max(0, y0 - a.pad)
        w0 = Image.open(pngs[0]).size
        x1, y1 = min(w0[0] - 1, x1 + a.pad), min(w0[1] - 1, y1 + a.pad)
        for p in pngs:
            before += p.stat().st_size
            im = Image.open(p).convert("RGBA").crop((x0, y0, x1 + 1, y1 + 1))
            # flatten onto white: the report shows these on a white card, and an
            # alpha channel is dead weight in a JPEG anyway
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            q = p.with_suffix(".jpg")
            bg.save(q, "JPEG", quality=a.quality, optimize=True)
            after += q.stat().st_size
            if not a.keep_png:
                p.unlink()
        tag = f"{rd.parent.parent.parent.name}/{rd.parent.parent.name}/{rd.name}"
        print(f"  {tag} {cd.name}: {len(pngs)} frames, box {x1-x0+1}x{y1-y0+1}")

    print(f"\n  {before/1e6:.1f} MB PNG -> {after/1e6:.1f} MB JPEG "
          f"({100*(1-after/max(before,1)):.0f}% smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
