#!/usr/bin/env python3
"""
adjust_fixation_cross_duration.py - rewrite trial_end for the two patients whose
trial_end was synthetic, so it reflects the experiment's fixation-cross period.

WHY THESE TWO AND ONLY THESE TWO. A cohort-wide audit of (next onset - this trial_end)
found 6 of 65 condition-blocks where that gap is IDENTICAL to the sample - i.e.
trial_end = next_onset - constant, carrying no information the next onset does not
already have. They belong to exactly two patients:

    EL033      1.28125 s (picture, reading)   0.78125 s (audio)      fs 1024
    PAT_3965   0.50000 s (all three)                                  fs 2048

Everywhere else trial_end varies over 36-55 distinct values per block and is left alone.

WHAT THIS WRITES.

    trial_end[i] = onset[i+1] - round(fs * (1.8 - U(0, 0.2)))     for i < n-1

1.8 s is the fixation-cross period; the uniform 0-0.2 s is the jitter, drawn per trial
from a SEEDED generator so the result is reproducible. The LAST trial of each block has
no successor, so its trial_end is left exactly as it was - there is nothing to measure
it against, and inventing one would be worse than leaving it visible.

THE COST, measured before writing rather than discovered after. post_s = trial_end -
sample_offsets shortens by the difference between the old constant and ~1.7 s, and
lf_trials drops post_s < 1.0 s. Across the two patients the surviving trial count falls
262 -> 185. No trial_end lands before its own stimulus offset.

THE DUPLICATE HAZARD. lf_trials.collect_trials globs prep_dir/*.tsv and de-duplicates by
MD5 only. EL033's prep0 held "... - Copy.tsv" files byte-identical to the originals, so
they were being skipped; patching the originals would have made them differ and both
would then be ingested, double-counting every trial. They are MOVED to a sibling
prep0_manual_backup/ rather than patched or deleted.

    python adjust_fixation_cross_duration.py --dry-run
    python adjust_fixation_cross_duration.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

FIX_BASE, FIX_JITTER = 1.8, 0.2
SEED = 20260825
MIN_POST, MAX_POST, IQR_K = 1.0, 5.0, 1.5
KEEP = {"correct", "valid", "1"}

RESULTS = (Path(__file__).resolve().parent / "outputs" / "04_ersp_LM_RAWONLY")
TARGETS = {
    "EL033": Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN"
                  r"\EL033\task_FBM\data_LM\prep0"),
    "PAT_3965": Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG"
                     r"\PAT_3965\task_FBM\data_LM\prep0"),
}
COL = "trial_end"


def fs_of(d: pd.DataFrame) -> float:
    on_s, on = d["onset"].to_numpy(float), d["sample"].to_numpy(float)
    g = on_s > 0
    return float(round(float(np.median(on[g] / on_s[g])) / 256) * 256)


def kept_count(post, m_ra):
    surv = ~m_ra & (post >= MIN_POST) & (post <= MAX_POST)
    if surv.sum() > 3:
        q1, q3 = np.percentile(post[surv], [25, 75])
        lo, hi = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
        surv = surv & (post >= lo) & (post <= hi)
    return int(surv.sum())


def patch_text(path: Path, new_end: np.ndarray) -> str:
    """Rewrite ONLY the trial_end field, leaving every other byte of every line alone.

    Round-tripping through pandas would reformat the float columns (onset carries 9-10
    decimals) and produce a diff far larger than the change actually being made."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    head = lines[0].rstrip("\r\n").split("\t")
    j = head.index(COL)
    out, k = [lines[0]], 0
    for ln in lines[1:]:
        body = ln.rstrip("\r\n")
        if not body.strip():
            out.append(ln)                       # trailing blank line, keep verbatim
            continue
        eol = ln[len(body):]
        f = body.split("\t")
        f[j] = str(int(new_end[k]))
        out.append("\t".join(f) + eol)
        k += 1
    assert k == len(new_end), f"{path.name}: rewrote {k} rows, expected {len(new_end)}"
    return "".join(out)


NOTE = """FIXATION-CROSS DURATION ADJUSTMENT — {pid}
{stamp}

trial_end in this patient's prep0 tables was SYNTHETIC. It was not a measurement: it sat
at a fixed distance before the next trial's onset, identical to the sample on every
trial, so it carried no information the next onset did not already have.

    what it was:  trial_end[i] = onset[i+1] - c, with c constant
{was}

It has been adjusted MANUALLY to reflect the experiment's fixation-cross period:

    trial_end[i] = onset[i+1] - round(fs * ({base} - U(0, {jit})))    for i < n-1

    {base} s   the fixation-cross period
    0-{jit} s   uniform jitter, drawn per trial, seed {seed} (reproducible)

The LAST trial of each block has no successor and was left exactly as it was.

WHY ONLY THIS PATIENT AND EL033/PAT_3965. A cohort audit of (next onset - this
trial_end) over 23 patients and 65 condition-blocks found only 6 blocks where that gap
is constant; they belong to these two patients alone. Every other patient's trial_end
varies over 36-55 distinct values per block and was NOT touched.

CONSEQUENCE, measured before the change. post_s = trial_end - sample_offsets is what
lf_trials filters on (min_post_s 1.0, max_post_s 5.0, then an IQR rule). Moving
trial_end earlier shortens post_s, and these blocks have only ~2.8 s of median spacing
between one stimulus ending and the next beginning, so a large share now falls under the
1.0 s floor:

{table}

No trial_end lands before its own stimulus offset, so the annotations remain valid.

FILES CHANGED (originals kept as .bak beside them):
{files}
{extra}
Written by 01_FBM_Analysis/adjust_fixation_cross_duration.py
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true",
                    help="undo: put every .tsv.bak back and remove the notes")
    a = ap.parse_args()
    if not (a.apply or a.dry_run or a.restore):
        print("pass --dry-run, --apply or --restore")
        return 2

    if a.restore:
        n = 0
        for pid, p0 in TARGETS.items():
            for b in sorted(p0.glob("*.tsv.bak")):
                tsv = b.with_suffix("")           # strip .bak -> ....tsv
                print(f"  restore {tsv.name}  <-  {b.name}")
                shutil.copy2(b, tsv)
                b.unlink()
                n += 1
            for note in (RESULTS / pid, p0):
                f = note / f"NOTE_fixation_cross_duration_adjustment_{pid}.txt"
                if f.exists():
                    print(f"  remove {f}")
                    f.unlink()
        print(f"\nrestored {n} file(s). The ' - Copy' files are still in "
              f"prep0_manual_backup/ — move them back only if you want them, they are "
              f"byte-identical to the restored originals and would be de-duplicated.")
        return 0
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rng = np.random.default_rng(SEED)
    grand = []

    for pid, p0 in TARGETS.items():
        if not p0.exists():
            print(f"!! {pid}: {p0} missing")
            return 1
        print(f"\n=== {pid}  {p0} ===")

        # --- IDEMPOTENCE. A .bak is proof this already ran. -----------------------
        # Without this the second run still reached the note-writing block with an
        # empty results table and overwrote a correct note with "0 -> 0", destroying
        # the only record of what was done. The TSVs themselves were always safe (the
        # constant-gap test below rejects them), but the note was not.
        done = [f for f in sorted(p0.glob("*.tsv"))
                if f.with_suffix(".tsv.bak").exists() and " - Copy" not in f.name]
        if done:
            print(f"  ALREADY ADJUSTED — {len(done)} file(s) carry a .bak from an "
                  f"earlier run. Nothing touched, note left intact.")
            for f in done:
                print(f"      {f.name}")
            print(f"  To redo it, undo first:  "
                  f"python {Path(__file__).name} --restore")
            continue

        # --- the duplicate hazard, handled before anything is written -------------
        copies = sorted(f for f in p0.glob("*.tsv") if " - Copy" in f.name)
        if copies:
            bak = p0.parent / "prep0_manual_backup"
            print(f"  {len(copies)} ' - Copy' file(s) inside prep0. collect_trials "
                  f"de-duplicates by MD5, so once the originals change these would "
                  f"stop matching and BOTH would be ingested (double-counting).")
            for c in copies:
                print(f"    move {c.name}  ->  {bak.name}/")
                if a.apply:
                    bak.mkdir(exist_ok=True)
                    shutil.move(str(c), str(bak / c.name))

        rows, changed = [], []
        for f in sorted(p0.glob("*.tsv")):
            if " - Copy" in f.name:
                continue
            m = re.search(r"_LM_([A-Za-z]+)_", f.name)
            cond = m.group(1).lower() if m else "?"
            d = pd.read_csv(f, sep="\t").dropna(how="all").reset_index(drop=True)
            n = len(d)
            fs = fs_of(d)
            on = d["sample"].to_numpy(float)
            off = d["sample_offsets"].to_numpy(float)
            end = d["trial_end"].to_numpy(float)
            ra = d["resp_accuracy"].astype(str).str.strip().str.lower()
            m_ra = ~ra.isin(KEEP).to_numpy()

            gaps = np.unique(np.round((on[1:] - end[:-1]) / fs, 6))
            if len(gaps) != 1:
                print(f"  !! {f.name}: gap is NOT constant ({len(gaps)} values) — "
                      f"this file is not one of the synthetic ones, SKIPPED")
                continue
            c_old = float(gaps[0])

            c_new = FIX_BASE - rng.uniform(0, FIX_JITTER, size=n - 1)
            end_new = end.copy()
            end_new[:-1] = on[1:] - np.round(c_new * fs)
            assert (end_new > off).all(), f"{f.name}: a trial_end precedes its offset"
            assert (end_new[:-1] < on[1:]).all(), f"{f.name}: trial_end past next onset"

            post_o, post_n = (end - off) / fs, (end_new - off) / fs
            k_o, k_n = kept_count(post_o, m_ra), kept_count(post_n, m_ra)
            rows.append((cond, n, c_old, float(np.median(post_o)), k_o,
                         float(np.median(post_n)), k_n))
            grand.append((pid, cond, k_o, k_n))
            print(f"  {cond:<8} n={n:<3} c {c_old:.5f} -> "
                  f"{c_new.min():.3f}..{c_new.max():.3f}s   "
                  f"post_s med {np.median(post_o):.2f} -> {np.median(post_n):.2f}   "
                  f"kept {k_o} -> {k_n} ({k_n-k_o:+d})")

            if a.apply:
                bak = f.with_suffix(".tsv.bak")
                # belt and braces: overwriting a .bak would replace the ORIGINAL with
                # an already-patched copy and make the change unrecoverable
                assert not bak.exists(), f"{bak.name} already exists — refusing"
                shutil.copy2(f, bak)
                f.write_text(patch_text(f, end_new.astype(np.int64)), encoding="utf-8")
                changed.append(f.name)

        # --- the note ------------------------------------------------------------
        if not rows:
            print("  no file qualified — no note written (an empty note would "
                  "overwrite a correct one)")
            continue
        was = "\n".join(f"        {c:<9} c = {r[2]:.5f} s" for c, r in
                        ((r[0], r) for r in rows))
        tbl = (f"    {'cond':<9}{'n':>4}{'post_s med':>12}{'kept':>7}   ->"
               f"{'post_s med':>12}{'kept':>7}\n" +
               "\n".join(f"    {r[0]:<9}{r[1]:>4}{r[3]:>12.2f}{r[4]:>7}   ->"
                         f"{r[5]:>12.2f}{r[6]:>7}" for r in rows))
        extra = ""
        if copies:
            extra = (f"\nMOVED OUT OF prep0 (would have caused double-counting):\n" +
                     "\n".join(f"    {c.name}  ->  ../prep0_manual_backup/"
                               for c in copies) + "\n")
        note = NOTE.format(pid=pid, stamp=stamp, was=was, base=FIX_BASE, jit=FIX_JITTER,
                           seed=SEED, table=tbl,
                           files="\n".join(f"    {p0}/{n}" for n in changed) or
                                 "    (dry run — nothing written)",
                           extra=extra)
        for dest in (RESULTS / pid, p0):
            out = dest / f"NOTE_fixation_cross_duration_adjustment_{pid}.txt"
            print(f"  note -> {out}")
            if a.apply:
                dest.mkdir(parents=True, exist_ok=True)
                out.write_text(note, encoding="utf-8")

    tot_o = sum(r[2] for r in grand)
    tot_n = sum(r[3] for r in grand)
    print(f"\nTOTAL surviving trials {tot_o} -> {tot_n}  ({tot_n-tot_o:+d})")
    print("APPLIED" if a.apply else "DRY RUN — nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
