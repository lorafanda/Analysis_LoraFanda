#!/usr/bin/env python3
"""
cleanup_stale.py — delete stale run dirs and scratch WITHOUT touching anything in use.

Companion to audit_coverage.py. The rule it enforces:

    a run directory may be deleted only if its OWN timestamp id (e.g. 20260529_185754)
    appears in no index.json, no latest pointer, no notebook, no .py, and none of the
    HTML viewers — including the mirrored lorafanda.github.io site.

On top of that it always keeps the NEWEST run of every track, so even a track that
nothing references yet (a fresh experiment) keeps its current result.

Everything on the NEVER list below is hard-refused regardless of what the scan says.
The two that matter:

  * 04_ersp_LM            — 264 GB, but it is the LIVE output of pipeline A in
                            140_ERSP_analysis_pipeline (ERSP/HG QC plots, Report, PSD).
                            Nothing in 02/03/04 reads it, but deleting it is a
                            deliberate choice about your QC record, not housekeeping.
  * 04_ersp_LM_RAWONLY - Copy
                          — looks like a stale 29-May snapshot, but it holds 396 .npy
                            files (132 contacts x 3 conditions) the live tree does NOT
                            have. Some is EL043's shaft rename (X_L -> X), but e.g.
                            EL038 ITG_R is genuinely absent from live. Keep until that
                            gap is resolved.

DRY RUN BY DEFAULT.

    python cleanup_stale.py            # show what would go
    python cleanup_stale.py --apply    # delete it
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIRROR = Path.home() / "lorafanda.github.io"
APPLY = "--apply" in sys.argv
RX_RUN = re.compile(r"20\d{6}_\d{6}")

STAGE_OUTPUTS = (
    "02_FBM_Clustering/outputs/clustering",
    "03_FBM_Classifying/outputs/classification",
    "04_FBM_Pooling/outputs/pooling",
)

# Hard refusal list — never deleted, whatever the reference scan concludes.
NEVER = (
    "01_FBM_Analysis/outputs/04_ersp_LM",          # also covers 04_ersp_LM_RAWONLY*
    "02_FBM_Clustering/outputs/_dataset",
    "03_FBM_Classifying/outputs/_dataset",
    "04_FBM_Pooling/outputs/_dataset",
    "04_FBM_Pooling/outputs/_anatomy",
    "02_FBM_Clustering/outputs/250_recon",         # incl. the fix_recon .bak_ backups
    "02_FBM_Clustering/x_archive",
    ".git",
)
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", "x_archive"}


def refused(p: Path) -> bool:
    rel = str(p.relative_to(ROOT)).replace("\\", "/")
    return any(rel == n or rel.startswith(n) for n in NEVER)


def size_of(p: Path) -> int:
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    tot = 0
    for dp, _, fn in os.walk(p):
        for f in fn:
            try:
                tot += (Path(dp) / f).stat().st_size
            except OSError:
                pass
    return tot


def mb(n: int) -> str:
    return f"{n/1e6:8.1f} MB"


def referenced_run_ids() -> set[str]:
    """Every run id named by anything that could still read it."""
    ref: set[str] = set()
    for sroot in (ROOT, MIRROR):
        if not sroot.exists():
            continue
        for pat in ("*.py", "*.ipynb", "*.html", "*.md", "*.json", "*.csv"):
            for f in sroot.rglob(pat):
                if any(s in f.parts for s in SKIP_DIRS):
                    continue
                try:
                    if f.stat().st_size > 40_000_000:
                        continue
                    txt = f.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for rid in RX_RUN.findall(txt):
                    # a run dir naming itself inside its own manifest is not a reference
                    if rid not in str(f):
                        ref.add(rid)
    return ref


def collect(ref: set[str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []

    for base in STAGE_OUTPUTS:
        b = ROOT / base
        if not b.exists():
            continue
        for rd in b.rglob("runs"):
            if not rd.is_dir():
                continue
            ds = sorted((d for d in rd.iterdir()
                         if d.is_dir() and RX_RUN.fullmatch(d.name)), key=lambda p: p.name)
            if not ds:
                continue
            newest = ds[-1]
            for d in ds:
                if d is not newest and d.name not in ref:
                    out.append((d, f"unreferenced run (track keeps {newest.name})"))

    for d in ROOT.rglob(".ipynb_checkpoints"):
        if d.is_dir():
            out.append((d, "Jupyter scratch"))

    return [(p, w) for p, w in out if not refused(p)]


def main() -> int:
    ref = referenced_run_ids()
    targets = collect(ref)
    print(f"referenced run ids: {len(ref)}   targets: {len(targets)}")
    if not APPLY:
        print("DRY RUN — nothing will be deleted.\n")

    total = 0
    for p, why in sorted(targets, key=lambda t: -size_of(t[0])):
        s = size_of(p)
        total += s
        print(f"  {mb(s)}  {p.relative_to(ROOT)}")
        print(f"             -> {why}")
        if APPLY:
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError as e:
                print(f"             !! FAILED: {e}")

    print("\n" + "=" * 68)
    print(f"  {'DELETED' if APPLY else 'WOULD DELETE'}: {len(targets)} items, {mb(total)}")
    print("=" * 68)
    if not APPLY:
        print("  add --apply to delete")
    else:
        print("  now re-check:  python audit_coverage.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
