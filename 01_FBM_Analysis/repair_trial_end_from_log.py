#!/usr/bin/env python3
"""
repair_trial_end_from_log.py — recover real response durations for G-02 and G-03
without re-running PD extraction.

WHY. lf_micromacro.py builds trial_end from a behavioural-log column named
`response_time`, and falls back to `sample_offsets + fs*2.0` when it cannot find
one. G-02 and G-03's logs spelled it `response_timex`, so the lookup missed and
EVERY trial got a flat 2 s — 4096 samples at 2048 Hz, zero variance, in all three
conditions. The response times were in the file the whole time.

The headers have since been corrected, so a fresh PD extraction would now work.
This script exists so that does not have to be re-run: it rewrites `trial_end` in
the existing prep0 tables directly.

    trial_end = sample_offsets + response_time * fs

VALIDATION. The join is POSITIONAL WITHIN CONDITION, and that is not a guess: run
with --validate to rebuild trial_end for G-01, whose values are already correct,
and compare. It reproduces all 160 trials to within half a sample, which pins the
join, the sampling rate and the formula at once.

DUPLICATES MATTER. prep0 holds byte-identical 05-08 / 05-12 copies of each table,
and collect_trials de-duplicates by content hash. Patching one copy and not the
other would make them differ, both would be ingested, and the double-counting bug
would come straight back. Every copy is therefore patched identically, and the
script asserts that identical inputs remain identical afterwards.

    python repair_trial_end_from_log.py --validate      # prove it on G-01, write nothing
    python repair_trial_end_from_log.py --dry-run       # show what would change
    python repair_trial_end_from_log.py --apply         # write, with .bak backups
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from functions import config as cfg  # noqa: E402

FS = 2048.0
TARGETS = ("G-02", "G-03")
CONTROL = "G-01"
CANON = {"auditory_naming_fre": "audio", "auditory_naming_ger": "audio",
         "auditory_naming_eng": "audio", "picture_naming": "picture",
         "reading_completion": "reading"}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def read_log(pid):
    pre = cfg.MICROEPI_MAT_PRESETS[pid]
    f = os.path.join(pre["data_dir"], pre["tsv_file"])
    raw = open(f, encoding="utf-8", errors="replace").read().splitlines()
    # some logs carry a "SubjectNumber : ..." banner before the header row
    df = pd.read_csv(f, sep="\t", skiprows=(1 if "\t" not in raw[0] else 0))
    low = {c.lower(): c for c in df.columns}
    rt = next((low[c] for c in ("response_time", "responsetime", "rt") if c in low), None)
    if rt is None:
        raise ValueError(f"{pid}: log has no response_time column ({list(df.columns)})")
    df["_cond"] = df[low["category"]].astype(str).str.lower().map(lambda s: CANON.get(s, s))
    df["_rt"] = pd.to_numeric(df[rt], errors="coerce")
    return df, f, rt


def prep_tables(pid):
    pre = cfg.MICROEPI_MAT_PRESETS[pid]
    d0 = os.path.join(os.path.dirname(pre["data_dir"]), "prep0")
    return sorted(glob.glob(os.path.join(d0, "*.tsv"))), d0


def patch_one(path, log, write):
    t = pd.read_csv(path, sep="\t")
    low = {c.lower(): c for c in t.columns}
    cond_raw = str(t[low["condition_name"]].iloc[0]).lower()
    cond = CANON.get(cond_raw, cond_raw)
    sub = log[log["_cond"] == cond].reset_index(drop=True)
    off = t[low["sample_offsets"]].to_numpy(np.int64)
    old = t[low["trial_end"]].to_numpy(np.int64)

    n = min(len(t), len(sub))
    if n < len(t):
        return dict(path=path, cond=cond, err=f"log has {len(sub)} rows for {cond}, "
                                              f"table has {len(t)}")
    rt = sub["_rt"].to_numpy(float)[:len(t)]
    if not np.isfinite(rt).all():
        return dict(path=path, cond=cond, err=f"{int((~np.isfinite(rt)).sum())} "
                                              f"non-finite response_time values")
    new = off + np.round(rt * FS).astype(np.int64)
    new = np.maximum(new, off + 1)                      # same guard the extractor uses

    if write:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        t[low["trial_end"]] = new
        t.to_csv(path, sep="\t", index=False)
    return dict(path=path, cond=cond, err=None, n=len(t),
                old=(old - off) / FS, new=(new - off) / FS)


def report(rows, header):
    print(f"\n  {header}")
    print(f"    {'file':46s} {'cond':8s} {'n':>4s}  {'old (s)':>22s}  {'new (s)':>26s}")
    for r in rows:
        if r.get("err"):
            print(f"    {os.path.basename(r['path'])[:46]:46s} {r['cond'][:8]:8s}  "
                  f"!! {r['err']}")
            continue
        o, nw = r["old"], r["new"]
        print(f"    {os.path.basename(r['path'])[:46]:46s} {r['cond'][:8]:8s} {r['n']:>4d}  "
              f"med {np.median(o):5.2f} uniq {len(np.unique(o)):3d}  ->  "
              f"med {np.median(nw):5.2f} rng {nw.min():5.2f}-{nw.max():6.2f} "
              f"uniq {len(np.unique(nw)):3d}")
        keep = int((nw > 0).sum() - (nw > cfg_max()).sum())
        print(f"    {'':46s} {'':8s}      -> {keep}/{r['n']} within the 5 s response cap")


def cfg_max():
    return 5.0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true",
                   help="rebuild G-01 (already correct) and compare; writes nothing")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    if a.validate:
        log, lf, rtcol = read_log(CONTROL)
        print(f"  control {CONTROL}: log {os.path.basename(lf)}  rt column {rtcol!r}")
        worst = 0.0
        for p in prep_tables(CONTROL)[0]:
            t = pd.read_csv(p, sep="\t")
            low = {c.lower(): c for c in t.columns}
            r = patch_one(p, log, write=False)
            if r.get("err"):
                print(f"    {os.path.basename(p)[:46]:46s} !! {r['err']}")
                continue
            actual = (t[low["trial_end"]].to_numpy(float)
                      - t[low["sample_offsets"]].to_numpy(float)) / FS
            err = np.abs(r["new"] - actual) * FS
            worst = max(worst, err.max())
            print(f"    {os.path.basename(p)[:46]:46s} n={r['n']:3d}  "
                  f"|rebuilt - on disk| max={err.max():.2f} samples, "
                  f"exact={(err < 1).sum()}/{r['n']}")
        print(f"\n  worst error across {CONTROL}: {worst:.2f} samples "
              f"({worst/FS*1000:.2f} ms) -> {'JOIN CONFIRMED' if worst < 2 else 'DO NOT APPLY'}")
        return 0 if worst < 2 else 1

    for pid in TARGETS:
        files, d0 = prep_tables(pid)
        print(f"\n=== {pid} ({cfg.MICROEPI_MAT_PRESETS[pid]['pat_name']}) ===")
        print(f"  prep0: {d0}")
        if not files:
            print("  !! no prep0 TSVs on disk — nothing to patch")
            continue
        before = {p: md5(p) for p in files}
        log, lf, rtcol = read_log(pid)
        print(f"  log  : {os.path.basename(lf)}  rt column {rtcol!r}  "
              f"({log['_rt'].notna().sum()} values, median {log['_rt'].median():.2f} s)")
        rows = [patch_one(p, log, write=a.apply) for p in files]
        report(rows, "APPLIED" if a.apply else "DRY RUN (nothing written)")

        if a.apply:
            after = defaultdict(list)
            for p in files:
                after[md5(p)].append(os.path.basename(p))
            groups_before = defaultdict(list)
            for p, h in before.items():
                groups_before[h].append(os.path.basename(p))
            same_before = {tuple(sorted(v)) for v in groups_before.values() if len(v) > 1}
            same_after = {tuple(sorted(v)) for v in after.values() if len(v) > 1}
            ok = same_before == same_after
            print(f"    duplicate groups preserved: {ok}  "
                  f"({len(same_before)} before, {len(same_after)} after)"
                  f"{'' if ok else '  <-- WOULD RE-INTRODUCE DOUBLE COUNTING'}")
            assert ok, "duplicate structure changed; re-check before running 150"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
