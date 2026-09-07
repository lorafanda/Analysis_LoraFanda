#!/usr/bin/env python3
"""convert_tifs_to_png.py - turn the TIFFs the pipeline used to write into PNGs.

matplotlib's TIFF is uncompressed: a 600-dpi PSD overview is ~54 MB, a 300-dpi ERSP
~10 MB, and a cluster-export patient holds hundreds. From 2026-09-07 every writer saves
PNG; this converts what is already on disk, losslessly (same pixels, PNG compression).

    python convert_tifs_to_png.py --dry-run             # count and size, nothing written
    python convert_tifs_to_png.py --apply               # write a .png beside every .tif
    python convert_tifs_to_png.py --apply --delete      # ...and remove the .tif afterwards

--delete removes a TIFF only after its PNG exists and re-opens with the same size in
pixels. Restrict with --root <dir> (default: outputs/) and --pattern (default: *.tif).
Nothing in the pipeline reads these images, so converting them changes no analysis.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs"))
    ap.add_argument("--pattern", default="*.tif")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delete", action="store_true", help="remove each .tif once its .png is verified")
    a = ap.parse_args()
    if not (a.dry_run or a.apply):
        print("pass --dry-run or --apply")
        return 2
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    root = Path(a.root)
    t0 = time.time()
    files = sorted(root.rglob(a.pattern)) + (sorted(root.rglob("*.tiff")) if a.pattern == "*.tif" else [])
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB under {root}  ({time.time() - t0:.0f}s to list)")
    by_dir = {}
    for f in files:
        by_dir[f.parent] = by_dir.get(f.parent, 0) + 1
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:>5}  {d.relative_to(root)}")
    if a.dry_run:
        print("\nDRY RUN - nothing written.")
        return 0

    done = skipped = removed = failed = 0
    saved = 0
    for i, f in enumerate(files, 1):
        png = f.with_suffix(".png")
        try:
            if not png.exists():
                with Image.open(f) as im:
                    im.load()
                    im.save(png, format="PNG", optimize=True)
                done += 1
            else:
                skipped += 1
            if a.delete:
                with Image.open(f) as im_t, Image.open(png) as im_p:
                    ok = im_t.size == im_p.size
                if ok:
                    saved += f.stat().st_size - png.stat().st_size
                    f.unlink()
                    removed += 1
                else:
                    print(f"  !! size mismatch, TIFF kept: {f}")
        except Exception as e:
            failed += 1
            print(f"  !! {f}: {type(e).__name__}: {e}")
        if i % 200 == 0:
            print(f"  {i}/{len(files)}  converted {done}  already had png {skipped}  removed {removed}  "
                  f"failed {failed}  {time.time() - t0:.0f}s", flush=True)
    print(f"\nconverted {done}, already had a png {skipped}, removed {removed}, failed {failed}, "
          f"{time.time() - t0:.0f}s")
    if a.delete:
        print(f"freed {saved / 1e9:.2f} GB")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
