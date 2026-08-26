#!/usr/bin/env python3
"""
move_halves_out_of_ersp_matrix.py - relocate the split-half cubes to their own folder.

WHY. 140 wrote <stem>_half1.npy / _half2.npy BESIDE the full cube in
ERSP_matrix/<condition>/. Everything that consumes the cubes globs that folder:

    lf_dataset.prepare_dataset   cond_dir.glob("*.npy")
    lf_clustering                its own walk of the same folder

so every real electrode was ingested THREE times - once as itself and once per half -
and because a half-cube averages half the trials it is noisier and trips the
responsiveness threshold far more easily. In the concat_source_v3 cache that produced
19,380 phantom rows against 9,342 real ones, and a gate pass rate of 96.9% for the
halves against 35.1% for the real cubes.

Filtering at read time would fix the one caller that was noticed. Moving the files
fixes every caller, including the ones nobody has written yet.

    ERSP_matrix/<cond>/<stem>_half1.npy   ->   ERSP_halves/<cond>/<stem>_half1.npy

The full cubes are NOT touched. Nothing is deleted.

    python move_halves_out_of_ersp_matrix.py --dry-run
    python move_halves_out_of_ersp_matrix.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "outputs" / "04_ersp_LM_RAWONLY"
SRC_SUB, DST_SUB = "ERSP_matrix", "ERSP_halves"
TASK = "LM"


def is_half(name: str) -> bool:
    return name.endswith("_half1.npy") or name.endswith("_half2.npy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        print("pass --dry-run or --apply")
        return 2
    if not ROOT.exists():
        print(f"!! {ROOT} not found")
        return 1

    moved, kept, clash = Counter(), Counter(), []
    t0 = time.time()
    for pdir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        src_root = pdir / TASK / SRC_SUB
        if not src_root.exists():
            continue
        for cond_dir in sorted(d for d in src_root.iterdir() if d.is_dir()):
            dst_dir = pdir / TASK / DST_SUB / cond_dir.name
            for f in sorted(cond_dir.glob("*.npy")):
                if not is_half(f.name):
                    kept[pdir.name] += 1
                    continue
                dst = dst_dir / f.name
                if dst.exists():
                    # never overwrite: a destination that already holds this name means
                    # a previous partial run, and silently replacing it would hide that
                    clash.append(str(dst))
                    continue
                moved[pdir.name] += 1
                if a.apply:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dst))

    print(f"{'patient':<12}{'full kept':>11}{'halves moved':>14}")
    for pid in sorted(set(moved) | set(kept)):
        print(f"{pid:<12}{kept[pid]:>11}{moved[pid]:>14}")
    print(f"\nTOTAL full cubes left in {SRC_SUB}: {sum(kept.values()):,}")
    print(f"TOTAL halves {'moved' if a.apply else 'to move'} -> {DST_SUB}: "
          f"{sum(moved.values()):,}")
    if clash:
        print(f"\n!! {len(clash)} destination(s) already existed and were LEFT ALONE:")
        for c in clash[:10]:
            print(f"   {c}")

    if a.apply:
        # prove it: no half file may remain anywhere under ERSP_matrix
        left = sum(1 for p in ROOT.glob(f"*/{TASK}/{SRC_SUB}/*/*.npy") if is_half(p.name))
        got = sum(1 for _ in ROOT.glob(f"*/{TASK}/{DST_SUB}/*/*.npy"))
        print(f"\nVERIFY  halves still under {SRC_SUB}: {left}   "
              f"files now under {DST_SUB}: {got:,}")
        print("OK" if left == 0 else "!! some halves remain - investigate before rebuilding")
        print(f"({time.time()-t0:.0f}s)")
    else:
        print("\nDRY RUN - nothing moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
