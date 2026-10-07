#!/usr/bin/env python3
"""
archive_unused.py - move what has not been touched for N days out of Analysis_LoraFanda.

    python archive_unused.py                       list what would move (nothing happens)
    python archive_unused.py --days 61             the cut-off (default 61: two months)
    python archive_unused.py --only 02_FBM_Clustering/outputs/_dataset/concat_source_v4 --days 0
                                                  (everything in those folders, whatever its dates)
    python archive_unused.py --only ... --skip-tracked   leave files git tracks where they are
    python archive_unused.py --move                move the listed files (asks nothing: list first!)
    python archive_unused.py --restore <manifest>  put everything of one move back

WHAT COUNTS AS UNUSED. A file whose LAST ACCESS and last modification are both older than the
cut-off. The share records access times (reading a cube, importing a module, opening a notebook
all count), so a file nobody read in two months is one nothing used - scripts included, since
a script that ran would have been read. The reverse is not true: a backup, a grep or a git
operation also reads files, so a recent access does not prove use. This script therefore only
ever moves files on the safe side of the rule.

WHERE THEY GO. Not deleted: moved, with their relative path, into a sibling folder of the repo
(ARCHIVE below), and every move is written to a manifest (archive_manifests/<stamp>.tsv) that
--restore replays. Deleting the archive folder is the one irreversible step and is yours to do
by hand once you are sure. Files tracked by git show up as deleted in `git status` after a move;
committing that deletion keeps them in the history.

NEVER TOUCHED: .git, .githooks, .gitattributes, .gitignore, this script, its manifests, and
the folders in PROTECT below (the live tree 03_ERSP, the current scripts' folders are protected
file by file through the access rule, not by name).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / f"_{ROOT.name}_archive"
MANIFESTS = ROOT / "archive_manifests"
PROTECT = {".git", ".githooks", ".gitattributes", ".gitignore", "archive_unused.py", "archive_manifests",
           "01_FBM_Analysis/outputs/03_ERSP"}


def tracked_files() -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True).stdout
    return set(out.decode("utf-8", "replace").split("\0")) - {""}


def unused(days: float, only: list[str] | None, skip_tracked: bool = False):
    cut = time.time() - days * 86400
    tracked = tracked_files() if skip_tracked else set()
    roots = [ROOT / o for o in only] if only else [ROOT]
    for r in roots:
        for dp, dn, fn in os.walk(r):
            rel_dir = os.path.relpath(dp, ROOT).replace("\\", "/")
            if rel_dir == ".":
                rel_dir = ""
            if any(rel_dir == p or rel_dir.startswith(p + "/") for p in PROTECT):
                dn[:] = []
                continue
            dn[:] = [d for d in dn if (f"{rel_dir}/{d}" if rel_dir else d) not in PROTECT]
            for f in fn:
                rel = f"{rel_dir}/{f}" if rel_dir else f
                if rel in PROTECT:
                    continue
                p = Path(dp) / f
                try:
                    st = p.stat()
                except OSError:
                    continue
                if rel in tracked:
                    continue
                if st.st_atime < cut and st.st_mtime < cut:
                    yield rel, st.st_size, st.st_mtime, st.st_atime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=61.0)
    ap.add_argument("--only", nargs="*", help="restrict to these folders (relative to the repo)")
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--skip-tracked", action="store_true", help="never move a file git tracks")
    ap.add_argument("--restore", metavar="MANIFEST")
    a = ap.parse_args()

    if a.restore:
        n = 0
        with open(a.restore, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                src, dst = Path(row["archived_to"]), ROOT / row["file"]
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    n += 1
        print(f"restored {n} files from {a.restore}")
        return 0

    rows = list(unused(a.days, a.only, a.skip_tracked))
    by_top = {}
    for rel, size, mt, at in rows:
        top = "/".join(rel.split("/")[:2])
        d = by_top.setdefault(top, [0, 0])
        d[0] += 1
        d[1] += size
    fmt = lambda t: time.strftime("%Y-%m-%d", time.localtime(t))
    print(f"cut-off: last access AND last modification before {fmt(time.time() - a.days * 86400)}")
    for top, (n, b) in sorted(by_top.items(), key=lambda kv: -kv[1][1]):
        print(f"  {top:<60} {n:7d} files  {b / 1e9:8.2f} GB")
    print(f"  {'TOTAL':<60} {len(rows):7d} files  {sum(r[1] for r in rows) / 1e9:8.2f} GB")
    if not a.move:
        print("\n(nothing moved - add --move)")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    MANIFESTS.mkdir(exist_ok=True)
    man = MANIFESTS / f"{stamp}.tsv"
    n = 0
    with open(man, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["file", "bytes", "modified", "accessed", "archived_to"])
        for rel, size, mt, at in rows:
            src, dst = ROOT / rel, ARCHIVE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
            except OSError as e:
                print(f"  [skip] {rel}: {e}")
                continue
            w.writerow([rel, size, fmt(mt), fmt(at), str(dst)])
            n += 1
    # empty folders left behind
    for dp, dn, fn in os.walk(ROOT, topdown=False):
        if dp != str(ROOT) and not os.listdir(dp) and ".git" not in dp:
            os.rmdir(dp)
    print(f"\nmoved {n} files -> {ARCHIVE}\nmanifest: {man}   (python archive_unused.py --restore {man} puts them back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
