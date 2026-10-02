#!/usr/bin/env python3
"""
144_move_to_pertrial.py - move the per-trial folders of the patients already run into
<pid>/LM/PerTrial/.

    python 144_move_to_pertrial.py             list what would move - moves nothing
    python 144_move_to_pertrial.py --apply     move, and print what went per patient
    python 144_move_to_pertrial.py --patients EL035 PAT_3455 [--apply]
    python 144_move_to_pertrial.py --tree 04_ersp-trialtestzscore [--apply]     another tree

Since 2026-10-02 the products with one row per trial live together:

    <pid>/LM/HFA/          ->  <pid>/LM/PerTrial/HFA/
    <pid>/LM/Signal/       ->  <pid>/LM/PerTrial/Signal/
    <pid>/LM/TrialScores/  ->  <pid>/LM/PerTrial/TrialScores/

140 writes them there (cfg.HFA_DIR, cfg.SIGNAL_DIR, cfg.TRIALSCORES_DIR) and every reader takes
the folder from the same three names, so after this move nothing looks in the old places.
Report/ is NOT moved: it holds the run's reports (trial filter, notch audit, montage) and nine
scripts read it.

A move is a rename on the same volume: instant, and no file is rewritten. If the destination
already exists (a patient re-run after the change has written PerTrial/HFA while an old HFA/ is
still there), the OLD folder's files are moved in only where the new one has no file of that
name - a newer figure is never replaced by an older one - and what is left over is reported,
not deleted.

RUN IT WHEN NO 140 PROCESS IS WRITING. A run started before the change still writes to the
old places; run this again afterwards (it is safe to repeat).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from functions import config as cfg                      # noqa: E402

# old name at <pid>/LM/  ->  where config says it lives now
MOVES = {"HFA": cfg.HFA_DIR, "Signal": cfg.SIGNAL_DIR, "TrialScores": cfg.TRIALSCORES_DIR}


def n_files(d: Path) -> int:
    return sum(len(fs) for _, _, fs in os.walk(d))


def merge(src: Path, dst: Path, apply: bool) -> tuple[int, int]:
    """Move src's files into an existing dst where dst has no such file. (moved, left)"""
    moved = left = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        for f in files:
            a, b = Path(root) / f, dst / rel / f
            if b.exists():
                left += 1                       # the newer one stays; the old one is reported
                continue
            if apply:
                b.parent.mkdir(parents=True, exist_ok=True)
                os.replace(a, b)
            moved += 1
    if apply and left == 0:                      # nothing left behind: clear the empty shell
        for root, dirs, files in os.walk(src, topdown=False):
            if not files and not dirs:
                try:
                    os.rmdir(root)
                except OSError:
                    pass
    return moved, left


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="move (default: list only)")
    ap.add_argument("--patients", nargs="*", help="folder names in the tree; default: every patient")
    ap.add_argument("--tree", default=cfg.ERSP_TREE, help=f"output tree under outputs/ (default {cfg.ERSP_TREE})")
    a = ap.parse_args()

    tree = Path(cfg.outputs_root) / a.tree
    if not tree.is_dir():
        tree = HERE / "outputs" / a.tree
    pdirs = sorted(p for p in tree.iterdir() if (p / cfg.block_name).is_dir())
    if a.patients:
        pdirs = [p for p in pdirs if p.name in set(a.patients)]
    print(f"tree: {tree}   ({len(pdirs)} patients)   {'MOVING' if a.apply else 'listing only - add --apply to move'}\n")

    tot = 0
    problems = []
    for pdir in pdirs:
        lm = pdir / cfg.block_name
        notes = []
        for old, new in MOVES.items():
            src, dst = lm / old, lm / new
            if not src.is_dir():
                if dst.is_dir():
                    notes.append(f"{old}: already in {new} ({n_files(dst)} files)")
                continue
            n = n_files(src)
            if not dst.exists():
                if a.apply:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.rename(src, dst)
                    except OSError as e:          # a file open in a viewer locks the folder
                        problems.append(f"{pdir.name}/{old}: {e}")
                        notes.append(f"{old}: NOT MOVED ({type(e).__name__})")
                        continue
                notes.append(f"{old} -> {new}  {n} files")
                tot += n
            else:
                moved, left = merge(src, dst, a.apply)
                notes.append(f"{old} -> {new}  {moved} files merged"
                             + (f", {left} older duplicates left in {old}/" if left else ""))
                tot += moved
                if left:
                    problems.append(f"{pdir.name}/{old}: {left} files stay - {new} already has newer ones of the same name")
        print(f"{pdir.name:<10} " + ("  |  ".join(notes) if notes else "nothing to move"))

    print(f"\n{tot} files {'moved' if a.apply else 'would move'}")
    for p in problems:
        print("   ! " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
