#!/usr/bin/env python3
"""
rename_hg_hfa.py - "HG" becomes "HFA" (high-frequency activity) across the website.

    python rename_hg_hfa.py            report what would change, per file, with samples
    python rename_hg_hfa.py --apply    change it, read back, compare

WHAT CHANGES. In the TEXT of every .html in lorafanda.github.io:
    HG                       -> HFA          (the token, on word boundaries)
    high gamma / high-gamma  -> high-frequency activity   (case preserved)
WHAT DOES NOT. Anything inside a tag - attributes, src/href/data-fig paths, alt text -
and any HG that is part of a path or an identifier (/HG/, HG_, HGtrials, HGA), which
name real files and columns. A replace that reached those would rename the
description of a file without renaming the file.

Regenerated blocks get the same treatment from their generators, whose label tables
were changed alongside; this script is for the hand-written prose and for anything
already on the page. Idempotent: HFA matches nothing here.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SITE_DIR = Path.home() / "lorafanda.github.io"
FILES = ["analysis_status.html", "cluster_visualizer.html", "moba.html",
         "activity_visualizer.html", "results.html", "classification.html",
         "index.html", "pool.html", "poolv2.html"]

TAG = re.compile(r"<[^>]*>", re.S)
HG = re.compile(r"(?<![/\w])HG(?![\w/])")
GAMMA = re.compile(r"\b(H|h)igh[- ]gamma\b")


def fix_text(t: str):
    n = 0
    def r1(m):
        nonlocal n; n += 1; return "HFA"
    def r2(m):
        nonlocal n; n += 1; return ("H" if m.group(1) == "H" else "h") + "igh-frequency activity"
    t = HG.sub(r1, t)
    t = GAMMA.sub(r2, t)
    return t, n


def fix_html(s: str):
    """Replace in text between tags only; tags (and everything in them) pass through."""
    out, n, pos = [], 0, 0
    for m in TAG.finditer(s):
        text, k = fix_text(s[pos:m.start()])
        out.append(text); n += k
        out.append(m.group(0))
        pos = m.end()
    text, k = fix_text(s[pos:])
    out.append(text); n += k
    return "".join(out), n


def samples(old: str, new: str, limit: int = 6):
    """A few before/after snippets, for the dry run."""
    got = []
    for m in HG.finditer(old):
        a, b = max(0, m.start() - 34), min(len(old), m.end() + 34)
        got.append(old[a:b].replace("\n", " "))
        if len(got) >= limit:
            break
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    total = 0
    for name in FILES:
        p = SITE_DIR / name
        if not p.exists():
            continue
        s0 = p.read_text(encoding="utf-8")
        s1, n = fix_html(s0)
        left_tag = len(HG.findall("".join(TAG.findall(s1))))       # untouched, by design
        print(f"  {name:<28} {n:>4} replacements" + (f"   ({left_tag} HG left inside tags)" if left_tag else ""))
        if not a.apply:
            for x in samples(s0, s1, 4 if n else 0):
                print(f"      …{x}…")
        elif n:
            p.write_text(s1, encoding="utf-8")
            if p.read_text(encoding="utf-8") != s1:
                raise SystemExit(f"{name}: read-back differs from what was written")
        total += n
    print(f"  {'applied' if a.apply else 'would apply'}: {total} replacements"
          + ("" if a.apply else "   (pass --apply)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
