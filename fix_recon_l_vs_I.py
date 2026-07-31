#!/usr/bin/env python3
"""
fix_recon_l_vs_I.py — correct the lowercase-l / capital-I electrode names.

The recon was labelled with a lowercase "l" where the clinical/ERSP naming uses a
capital "I" (aI = anterior Insula, pI = posterior Insula):

      recon  alR1  ->  a + l (U+006C)      ERSP  aI_R1  ->  a + I (U+0049)

The coordinates are correct; only the LABEL is wrong. Because `<PID>.electrodeNames`
is a plain-text sidecar whose Nth line labels the Nth row of `<PID>.LEPTOVOX`, fixing
it is a pure text edit:

  * NO FreeSurfer rerun. Surfaces, talairach.xfm and the mgz volumes never reference
    these labels, so nothing FreeSurfer produced is affected.
  * NO re-localisation. The xyz values do not move.
  * Line counts are verified unchanged, so the name<->coordinate row alignment holds.

It fixes both ends so you do not have to re-run the (archived) 251 recon notebook:
  1. the LEPTOVOX `.electrodeNames` source, so the error cannot come back
  2. the derived coords CSVs (tkrRAS + per-patient fsaverage + ALL_PATIENTS)

DRY RUN BY DEFAULT. Nothing is written until you pass --apply.

    python fix_recon_l_vs_I.py                # show what would change
    python fix_recon_l_vs_I.py --apply        # do it (writes .bak backups first)
    python fix_recon_l_vs_I.py --apply --derived-only   # skip the DATARAW source
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BERN_RECON = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\Reconstruction")
RECON_OUT = ROOT / "02_FBM_Clustering" / "outputs" / "250_recon"
PATIENTS = ["EL035", "EL037", "EL038", "EL040", "EL042"]

# Only ever touch a name that is <a|p> + lowercase l + <L|R> + digits. Case-SENSITIVE:
# 'alR1' matches, 'ALR1' and 'aIR1' do not. This cannot hit a genuine shaft such as
# 'AL' or 'STGR'.
RX_NAME = re.compile(r"^([ap])l([LR]\d+)$")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def fix_name(n: str) -> str:
    return RX_NAME.sub(r"\1I\2", str(n).strip())


def backup(p: Path, apply: bool) -> None:
    if apply:
        shutil.copy2(p, p.with_suffix(p.suffix + f".bak_{STAMP}"))


# ---------------------------------------------------------------- source
def fix_electrode_names(apply: bool) -> int:
    print("\n=== 1. LEPTOVOX source (.electrodeNames) ===")
    total = 0
    for pid in PATIENTS:
        f = BERN_RECON / pid.lower() / "elec_recon" / f"{pid}.electrodeNames"
        if not f.exists():
            print(f"  {pid}: NOT FOUND at {f}")
            continue
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        lepto = f.with_suffix(".LEPTOVOX")
        n_lepto = len(lepto.read_text(errors="ignore").splitlines()) if lepto.exists() else None

        out, changed = [], []
        for i, ln in enumerate(lines):
            if i < 2 or not ln.strip():            # timestamp + header
                out.append(ln); continue
            parts = ln.split()
            new = fix_name(parts[0])
            if new != parts[0]:
                changed.append((parts[0], new))
                parts[0] = new
                ln = " ".join(parts)
            out.append(ln)

        if not changed:
            print(f"  {pid}: nothing to fix")
            continue
        if n_lepto is not None and n_lepto != len(out):
            print(f"  {pid}: !! ABORT — line count would differ from LEPTOVOX "
                  f"({len(out)} vs {n_lepto})")
            continue
        shafts = sorted({f"{a.rstrip('0123456789')}->{b.rstrip('0123456789')}" for a, b in changed})
        print(f"  {pid}: {len(changed)} names  ({', '.join(shafts)})"
              f"{'  [WROTE]' if apply else '  [dry-run]'}")
        total += len(changed)
        if apply:
            backup(f, apply)
            f.write_text("\n".join(out) + "\n", encoding="utf-8")
    return total


# ---------------------------------------------------------------- derived
def fix_csv(p: Path, col: str, apply: bool, label: str) -> int:
    if not p.exists():
        return 0
    df = pd.read_csv(p)
    if col not in df.columns:
        return 0
    new = df[col].map(fix_name)
    n = int((new != df[col]).sum())
    if n:
        print(f"  {label}: {n} names{'  [WROTE]' if apply else '  [dry-run]'}")
        if apply:
            backup(p, apply)
            df[col] = new
            df.to_csv(p, index=False)
    return n


def fix_derived(apply: bool) -> int:
    print("\n=== 2. derived coords (so you need not re-run 251) ===")
    total = 0
    for pid in PATIENTS:
        total += fix_csv(RECON_OUT / pid / "glassbrain" / "coords" / f"{pid}_contacts_tkrRAS.csv",
                         "name", apply, f"{pid} tkrRAS")
        total += fix_csv(RECON_OUT / "fsaverage" / "coords" / f"{pid}_contacts_fsaverage.csv",
                         "name", apply, f"{pid} fsaverage")
    for extra in sorted((RECON_OUT / "fsaverage" / "coords").glob("ALL_PATIENTS*.csv")):
        total += fix_csv(extra, "name", apply, f"  {extra.name}")
    for extra in sorted((RECON_OUT / "talairach" / "coords").glob("*.csv")) if (RECON_OUT / "talairach" / "coords").is_dir() else []:
        total += fix_csv(extra, "name", apply, f"  talairach/{extra.name}")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--derived-only", action="store_true",
                    help="skip the DATARAW .electrodeNames source, fix only the derived CSVs")
    a = ap.parse_args()

    if not a.apply:
        print("DRY RUN — nothing will be written. Re-run with --apply to commit the changes.")
    else:
        print(f"APPLYING changes. Backups written alongside each file as *.bak_{STAMP}")

    n_src = 0 if a.derived_only else fix_electrode_names(a.apply)
    n_der = fix_derived(a.apply)

    print(f"\n{'='*66}")
    print(f"  source names : {n_src}")
    print(f"  derived rows : {n_der}")
    if a.apply:
        print("\n  Next: re-run 252 (recon renders), then the stages that join coords\n"
              "  (233 concat clustering / 460 pooling). Verify with:  python audit_coverage.py")
    else:
        print("\n  Nothing written. Add --apply when the list above looks right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
