#!/usr/bin/env python3
"""
archive_unused.py - move what is no longer used out of Analysis_LoraFanda, reversibly.

    python archive_unused.py                       list what the age rule would move (nothing happens)
    python archive_unused.py --days 61             the cut-off (default 61: two months)
    python archive_unused.py --only A B ...         only these folders / files (relative to the repo)
    python archive_unused.py --only-file units.txt  the same, one path per line (# comments allowed)
    python archive_unused.py --only ... --days 0   everything in those folders, whatever its dates
    python archive_unused.py ... --move            do it (asks nothing: list first!)
    python archive_unused.py --restore <manifest>  put everything of one move back
    python archive_unused.py --dest <folder>       where the archive is (default below); also for --restore
                                                  after the archive folder was renamed

WHAT COUNTS AS UNUSED (age mode, no --only). A file whose LAST ACCESS and last modification are
both older than the cut-off. The share records access times, but it also sweeps them (seen
2026-09-24), so the age rule is a floor, not the truth: decide by provenance and mtime, and
hand the decision to --only.

WHERE THEY GO. Not deleted: moved, with their relative path, into
    <dest> = <parent of the repo>/Analysis_Lora/Analysis_LoraFanda_archive
(Lora renames Analysis_Lora to "archive" herself; --dest follows the rename). Moves are renames
on the same share - nothing is copied, nothing can be half-written - and a destination that
already exists is never overwritten (the unit is skipped and reported). A folder given with
--only and --days 0 moves as ONE rename; files move one by one. Every move is written to a
manifest (archive_manifests/<stamp>.tsv) that --restore replays; after the renames the archived
files are listed into archive_manifests/<stamp>_files.tsv (skip with --no-filelist).

Files tracked by git show up as deleted in `git status` after a move; committing that deletion
keeps them in the history and is a separate step.

NEVER TOUCHED: .git, .githooks, .gitattributes, .gitignore, this script, its manifests, and the
folders in PROTECT below (the live tree 03_ERSP among them).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / "Analysis_Lora" / f"{ROOT.name}_archive"
MANIFESTS = ROOT / "archive_manifests"
PROTECT = {".git", ".githooks", ".gitattributes", ".gitignore", "archive_unused.py", "archive_manifests",
           "01_FBM_Analysis/outputs/03_ERSP"}


def protected(rel: str) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in PROTECT)


def tracked_files() -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True).stdout
    return set(out.decode("utf-8", "replace").split("\0")) - {""}


def scan(d: Path, rel: str):
    """(rel, size, mtime, atime) for every file under d; scandir keeps the share's round trips low."""
    try:
        with os.scandir(d) as it:
            entries = list(it)
    except OSError as e:
        print(f"  [unreadable] {rel}: {e}")
        return
    for e in entries:
        r = f"{rel}/{e.name}" if rel else e.name
        if protected(r):
            continue
        try:
            if e.is_dir(follow_symlinks=False):
                yield from scan(Path(e.path), r)
            else:
                st = e.stat(follow_symlinks=False)
                yield r, st.st_size, st.st_mtime, st.st_atime
        except OSError:
            continue


def unused(days: float, only: list[str] | None, skip_tracked: bool = False):
    cut = time.time() - days * 86400
    tracked = tracked_files() if skip_tracked else set()
    items = only if only else [""]
    for o in items:
        p = ROOT / o if o else ROOT
        if o and protected(o):
            print(f"  [protected] {o}")
            continue
        if p.is_file():
            st = p.stat()
            rows = [(o, st.st_size, st.st_mtime, st.st_atime)]
        elif p.is_dir():
            rows = scan(p, o)
        else:
            print(f"  [missing] {o}")
            continue
        for rel, size, mt, at in rows:
            if rel in tracked:
                continue
            if st_ok(mt, at, cut):
                yield o, rel, size, mt, at


def st_ok(mt: float, at: float, cut: float) -> bool:
    return mt < cut and at < cut


def read_only_file(path: str) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip().replace("\\", "/").strip("/")
        if s:
            out.append(s)
    return out


def rename(src: Path, dst: Path) -> bool:
    """One rename, never a copy; False (and a line) when it cannot be done."""
    if dst.exists():
        print(f"  [exists, skipped] {dst}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError as e:
        print(f"  [failed] {src}: {e}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=61.0)
    ap.add_argument("--only", nargs="*", help="restrict to these folders / files (relative to the repo)")
    ap.add_argument("--only-file", help="file with one relative path per line")
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--skip-tracked", action="store_true", help="never move a file git tracks")
    ap.add_argument("--no-filelist", action="store_true", help="do not list the archived files after the move")
    ap.add_argument("--restore", metavar="MANIFEST")
    ap.add_argument("--dest", default=str(ARCHIVE), help=f"archive folder (default {ARCHIVE})")
    a = ap.parse_args()
    dest = Path(a.dest)
    only = list(a.only or [])
    if a.only_file:
        only += read_only_file(a.only_file)
    only = [o.replace("\\", "/").strip("/") for o in only]

    if a.restore:
        n = 0
        with open(a.restore, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                src = dest / row["unit"]
                if not src.exists():
                    src = Path(row["archived_to"])
                if src.exists() and rename(src, ROOT / row["unit"]):
                    n += 1
        print(f"restored {n} units from {a.restore}")
        return 0

    whole = a.days == 0 and bool(only) and not a.skip_tracked
    if whole:
        # every folder / file named in --only moves as one unit: no walk before the move
        units = []
        for o in only:
            p = ROOT / o
            if protected(o):
                print(f"  [protected] {o}")
            elif not p.exists():
                print(f"  [missing] {o}")
            else:
                units.append(o)
        tracked = tracked_files()
        print(f"{len(units)} units -> {dest}")
        for o in units:
            nt = sum(1 for t in tracked if t == o or t.startswith(o + "/"))
            print(f"  {o:<80} {'dir ' if (ROOT / o).is_dir() else 'file'}  tracked {nt}")
        if not a.move:
            print("\n(nothing moved - add --move)")
            return 0
        stamp = time.strftime("%Y%m%d_%H%M%S")
        MANIFESTS.mkdir(exist_ok=True)
        man = MANIFESTS / f"{stamp}.tsv"
        n = 0
        with open(man, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["unit", "kind", "archived_to"])
            for o in units:
                src, dst = ROOT / o, dest / o
                kind = "dir" if src.is_dir() else "file"
                if rename(src, dst):
                    w.writerow([o, kind, str(dst)])
                    fh.flush()
                    n += 1
        print(f"\nmoved {n} of {len(units)} units -> {dest}\nmanifest: {man}   "
              f"(python archive_unused.py --restore {man} [--dest <where the archive is now>] puts them back)")
        if not a.no_filelist:
            fl = MANIFESTS / f"{stamp}_files.tsv"
            nf = tb = 0
            with open(fl, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, delimiter="\t")
                w.writerow(["file", "bytes", "modified", "unit"])
                for row in csv.DictReader(open(man, encoding="utf-8"), delimiter="\t"):
                    u = row["unit"]
                    p = dest / u
                    rows = [(u, p.stat().st_size, p.stat().st_mtime, 0)] if p.is_file() else scan(p, u)
                    for rel, size, mt, _ in rows:
                        w.writerow([rel, size, time.strftime("%Y-%m-%d", time.localtime(mt)), u])
                        nf += 1
                        tb += size
            print(f"archived files listed: {nf} files, {tb / 1e9:.2f} GB -> {fl}")
        return 0

    rows = list(unused(a.days, only, a.skip_tracked))
    by_top = {}
    for o, rel, size, mt, at in rows:
        top = o or "/".join(rel.split("/")[:2])
        d = by_top.setdefault(top, [0, 0])
        d[0] += 1
        d[1] += size
    fmt = lambda t: time.strftime("%Y-%m-%d", time.localtime(t))
    print(f"cut-off: last access AND last modification before {fmt(time.time() - a.days * 86400)}")
    for top, (n, b) in sorted(by_top.items(), key=lambda kv: -kv[1][1]):
        print(f"  {top:<60} {n:7d} files  {b / 1e9:8.2f} GB")
    print(f"  {'TOTAL':<60} {len(rows):7d} files  {sum(r[2] for r in rows) / 1e9:8.2f} GB")
    if not a.move:
        print("\n(nothing moved - add --move)")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    MANIFESTS.mkdir(exist_ok=True)
    man = MANIFESTS / f"{stamp}.tsv"
    n = 0
    with open(man, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["unit", "kind", "archived_to", "bytes", "modified", "accessed"])
        for o, rel, size, mt, at in rows:
            if rename(ROOT / rel, dest / rel):
                w.writerow([rel, "file", str(dest / rel), size, fmt(mt), fmt(at)])
                n += 1
    # empty folders left behind, under the folders that were walked
    for o in (only or [""]):
        for dp, dn, fn in os.walk(ROOT / o, topdown=False):
            if dp != str(ROOT) and ".git" not in dp and not os.listdir(dp):
                os.rmdir(dp)
    print(f"\nmoved {n} files -> {dest}\nmanifest: {man}   (python archive_unused.py --restore {man} puts them back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
