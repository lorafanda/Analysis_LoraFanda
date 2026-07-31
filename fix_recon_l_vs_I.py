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

# Per-patient SHAFT aliases: recon spelling -> ERSP spelling. Unlike the l/I case these
# are not typos, they are different naming CONVENTIONS between the recon and the
# clinical/ERSP side, so they are applied to the derived coords only (never to DATARAW).
# Each was verified by matching contact count and hemisphere:
#   EL034  MFG    12 hemi L  == MFGL 12   (source omits the hemisphere suffix)
#   EL043  pl     18 hemi L  == PINS 18   (pl = pI = posterior Insula, spelled out)
#   EL045  PlanTL 15 hemi L  >= PLATL  6  (ERSP carries only the first 6 of the shaft)
#   PAT_6619 OFAD  5 hemi R  == OFA   5   (trailing D = Droite)
#            OFPD  8 hemi R  == OFP   8
SHAFT_ALIASES: dict[str, dict[str, str]] = {
    "EL034":    {"MFG": "MFGL"},
    "EL043":    {"pl": "PINS"},
    "EL045":    {"PlanTL": "PLATL"},
    "PAT_6619": {"OFAD": "OFA", "OFPD": "OFP"},
}
ALIAS_PATIENTS = list(SHAFT_ALIASES)
RX_SPLIT = re.compile(r"^(.*?)(\d+)$")


def fix_name(n: str) -> str:
    """The l/I typo fix (applies everywhere)."""
    return RX_NAME.sub(r"\1I\2", str(n).strip())


def apply_alias(n: str, pid: str) -> str:
    """Shaft-convention rename for one patient, preserving the contact number."""
    m = RX_SPLIT.match(str(n).strip())
    if not m:
        return n
    shaft, num = m.group(1), m.group(2)
    tgt = SHAFT_ALIASES.get(pid, {}).get(shaft)
    return f"{tgt}{num}" if tgt else n


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
def fix_csv(p: Path, col: str, apply: bool, label: str, pid: str | None = None) -> int:
    """Apply the l/I fix, plus (for the derived coords) the shaft aliases. When `pid`
    is None the file is multi-patient and the alias is chosen per row."""
    if not p.exists():
        return 0
    df = pd.read_csv(p)
    if col not in df.columns:
        return 0
    if pid is not None:
        new = df[col].map(lambda v: apply_alias(fix_name(v), pid))
    elif "patient" in df.columns:
        new = [apply_alias(fix_name(v), str(q)) for v, q in zip(df[col], df["patient"])]
    else:
        new = df[col].map(fix_name)
    new = pd.Series(new, index=df.index)
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
    for pid in sorted(set(PATIENTS) | set(ALIAS_PATIENTS)):
        total += fix_csv(RECON_OUT / pid / "glassbrain" / "coords" / f"{pid}_contacts_tkrRAS.csv",
                         "name", apply, f"{pid} tkrRAS", pid=pid)
        total += fix_csv(RECON_OUT / "fsaverage" / "coords" / f"{pid}_contacts_fsaverage.csv",
                         "name", apply, f"{pid} fsaverage", pid=pid)
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
