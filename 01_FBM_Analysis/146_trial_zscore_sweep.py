#!/usr/bin/env python3
"""
146_trial_zscore_sweep.py - which z removes how many trials, from the 145 tables.

    python 146_trial_zscore_sweep.py                       every patient in the test tree
    python 146_trial_zscore_sweep.py --patient PAT_3455
    python 146_trial_zscore_sweep.py --z 4 --frac 0.30 --list   name the trials it drops

Reads outputs/04_ersp-trialtestzscore/<pid>/LM/TrialScores/*.tsv and answers the question
without re-running anything: 145 computes the scores once, this sweeps them freely.

THE RULE, as agreed.

  score      per channel x trial, the 99th percentile of |dB| inside 70-150 Hz. Computed
             by compute_ersp itself, not re-derived here.
  z          per CHANNEL, across that channel's trials: z = (score - mean) / SD. Each
             channel judges its own trials, because channels differ enormously in
             baseline amplitude and one common threshold would simply select the loud
             ones.
  drop       a TRIAL is dropped for the WHOLE patient-condition when it exceeds z on at
             least `frac` of that condition's channels. A real artefact is montage-wide;
             one channel disagreeing with the rest is a channel problem, not a trial
             problem. Dropping whole trials also keeps every electrode of a patient on
             the same trial count, which the noise-scaled gate depends on (its threshold
             moves with 1/sqrt(N)).

Both knobs matter, so both are swept. The table is trials dropped; the aim is the corner
where it stops changing - a rule that removes the same few trials over a wide range of z
is finding artefacts, one that keeps eating more is just trimming the distribution.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "functions"))
import config as cfg  # noqa: E402

TREE = Path(cfg.outputs_root) / "04_ersp-trialtestzscore"
ZS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
FRACS = [0.10, 0.20, 0.30, 0.50]


def load(patients=None) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(TREE / "*" / "*" / "TrialScores" / "*_trial_scores.tsv"))):
        d = pd.read_csv(f, sep="\t")
        if patients and str(d.patient.iloc[0]) not in patients:
            continue
        rows.append(d)
    if not rows:
        raise SystemExit(f"no trial score tables under {TREE}\n"
                         f"run 145_trial_zscore_test.py --patient <ids> first")
    return pd.concat(rows, ignore_index=True)


def per_channel_z(d: pd.DataFrame, col="score_hg") -> pd.DataFrame:
    """z within each patient x condition x channel, across that channel's trials."""
    g = d.groupby(["patient", "condition", "channel"])[col]
    mu, sd = g.transform("mean"), g.transform("std")
    out = d.copy()
    # a channel whose trials all agree has SD ~ 0; z there is meaningless, not infinite
    out["z"] = np.where(sd > 1e-9, (d[col] - mu) / sd, 0.0)
    return out


def dropped(d: pd.DataFrame, z: float, frac: float) -> pd.DataFrame:
    """Per patient x condition x trial: the share of channels over z, and the verdict."""
    d = d.assign(over=d.z > z)
    g = (d.groupby(["patient", "condition", "trial"])
           .agg(share=("over", "mean"), n_ch=("over", "size"),
                worst=("z", "max"), median_z=("z", "median"))
           .reset_index())
    g["is_drop"] = g.share >= frac
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", nargs="*", default=None)
    ap.add_argument("--score", default="score_hg", choices=["score_hg", "score_map"],
                    help="high-gamma band (default) or the whole map")
    ap.add_argument("--z", type=float, default=None, help="with --list, the z to report at")
    ap.add_argument("--frac", type=float, default=None, help="with --list, the fraction")
    ap.add_argument("--list", action="store_true", help="name the trials dropped at --z/--frac")
    a = ap.parse_args()

    raw = load(set(a.patient) if a.patient else None)
    if raw[a.score].isna().all():
        raise SystemExit(f"{a.score} is all NaN - 145 did not have the scoring turned on")
    d = per_channel_z(raw, a.score)

    n_tr = d.groupby(["patient", "condition"]).trial.nunique().sum()
    print(f"{TREE}")
    print(f"{raw.patient.nunique()} patient(s), {len(raw)} channel-trial rows, "
          f"{n_tr} patient-condition-trials, scoring on {a.score}\n")

    print("TRIALS DROPPED   (rows = z, columns = share of channels over z)")
    hdr = "   z  " + "".join(f"{int(f*100):>8}%" for f in FRACS) + "     of"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for z in ZS:
        cells = []
        for f in FRACS:
            cells.append(int(dropped(d, z, f)["is_drop"].sum()))
        print(f" {z:4.1f}  " + "".join(f"{c:>9}" for c in cells) + f"   {n_tr:>6}")

    print("\nPER PATIENT AND CONDITION, at a few settings")
    for z, f in ((3.0, 0.20), (4.0, 0.20), (5.0, 0.20)):
        g = dropped(d, z, f)
        s = (g.groupby(["patient", "condition"])
               .agg(trials=("trial", "nunique"), n_drop=("is_drop", "sum")).reset_index())
        s["kept"] = s["trials"] - s["n_drop"]
        worst = s.sort_values("n_drop", ascending=False).head(6)
        print(f"  z={z}, frac={f:.0%}:  total dropped {int(s['n_drop'].sum())} of {int(s['trials'].sum())}"
              f"   worst: " + ", ".join(f"{r.patient}/{r.condition} {int(r.n_drop)}"
                                        for r in worst.itertuples() if r.n_drop))
    if a.list and a.z is not None and a.frac is not None:
        g = dropped(d, a.z, a.frac)
        hit = g[g["is_drop"]].sort_values(["patient", "condition", "trial"])
        print(f"\nTRIALS DROPPED at z={a.z}, frac={a.frac:.0%}  ({len(hit)})")
        for r in hit.itertuples():
            print(f"  {r.patient:<10} {r.condition:<8} trial {int(r.trial):>3}  "
                  f"on {r.share:.0%} of {int(r.n_ch)} channels  worst z {r.worst:5.1f}")
    elif a.list:
        print("\n--list needs --z and --frac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
