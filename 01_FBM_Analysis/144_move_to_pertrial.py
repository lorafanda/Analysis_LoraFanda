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
    <pid>/LM/Report/       ->  <pid>/LM/PerTrial/Report/

140 writes them there (cfg.HFA_DIR, cfg.SIGNAL_DIR, cfg.TRIALSCORES_DIR, cfg.REPORT_DIR) and
every reader asks cfg.product_dir(), which looks in PerTrial/ first and in the old place only
when PerTrial/ has no such folder - so a tree can be read before, during and after this move.

A move is a rename on the same volume: instant, and no file is rewritten. If the destination
already exists the two folders are merged file by file and the NEWER file wins, whichever
folder it is in; the older duplicate is deleted. So it is safe to run again after a process
that was still writing to the old places has finished.

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
MOVES = {"HFA": cfg.HFA_DIR, "Signal": cfg.SIGNAL_DIR, "TrialScores": cfg.TRIALSCORES_DIR,
         "Report": cfg.REPORT_DIR}


def n_files(d: Path) -> int:
    return sum(len(fs) for _, _, fs in os.walk(d))


def merge(src: Path, dst: Path, apply: bool) -> tuple[int, int]:
    """Move src's files into an existing dst; where both have a file of that name THE NEWER ONE
    WINS, whichever side it is on, and the older one is deleted. (moved, older dropped)

    Both directions happen: a patient re-run after the change has newer files in PerTrial/, and
    a process started before the change (149 on 2026-10-02) wrote NEWER figures into the old
    place after the first move - 16 patients had the 200 µV Signal figures in Signal/ and the
    300 µV ones in PerTrial/Signal/."""
    moved = left = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        for f in files:
            a, b = Path(root) / f, dst / rel / f
            if b.exists():
                if a.stat().st_mtime > b.stat().st_mtime:      # the old place holds the newer file
                    if apply:
                        os.replace(a, b)
                    moved += 1
                else:                                           # PerTrial already has the newer one
                    if apply:
                        a.unlink()
                    left += 1
                continue
            if apply:
                b.parent.mkdir(parents=True, exist_ok=True)
                os.replace(a, b)
            moved += 1
    if apply:                                    # every file went one way or the other: clear the empty shell
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
                notes.append(f"{old} -> {new}  {moved} files merged (newer wins)"
                             + (f", {left} older duplicates in {old}/ {'deleted' if a.apply else 'would be deleted'}" if left else ""))
                tot += moved
        print(f"{pdir.name:<10} " + ("  |  ".join(notes) if notes else "nothing to move"))

    print(f"\n{tot} files {'moved' if a.apply else 'would move'}")
    for p in problems:
        print("   ! " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
