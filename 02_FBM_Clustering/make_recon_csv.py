#!/usr/bin/env python3
"""
make_recon_csv.py — the coordinate table MOBA needs for a run's 3-D brain.

Notebook 252 does two separate jobs and they have different dependencies:

    recon/<method>_<featureset>_<run_id>__with_fsaverage.csv   <- MOBA's 3-D brain
    recon/cluster_NN/by_condition/*.png                        <- the glassbrain figures

Only the PNGs need pyvista. The CSV is a join of labels.csv onto
ALL_PATIENTS_contacts_fsaverage.csv, which is plain pandas, so a run can get its 3-D
brain without a rendering stack. This writes the CSV; the PNGs still need 252.

The join is on patient + a NORMALISED electrode name (underscores and hyphens stripped,
upper-cased). That normalisation is the thing to watch: an unnormalised join is the bug
class that has already bitten this project three times, most recently taking the anatomy
match from 55% to 96%.

    python make_recon_csv.py --run outputs/clustering/cnmf/concat_hg/runs/<id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
COORDS = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coords"
          / "ALL_PATIENTS_contacts_fsaverage.csv")
KEEP = ["patient", "name", "hemi", "x", "y", "z", "is_wm", "is_cortical",
        "dist_to_pial_mm", "yeo7_network", "yeo17_network"]


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory")
    a = ap.parse_args()

    rd = Path(a.run)
    if not (rd / "labels.csv").exists():
        print(f"  !! no labels.csv in {rd}", file=sys.stderr)
        return 1
    import json
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    method, fset, run_id = man["method"], man["feature_set"], man["run_id"]

    lab = pd.read_csv(rd / "labels.csv")
    co = pd.read_csv(COORDS)
    lab["_k"] = [f"{p}|{norm(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
    co["_k"] = [f"{p}|{norm(n)}" for p, n in zip(co["patient"], co["name"])]
    co["contact_name"] = co["name"]

    out = lab.merge(co[["_k", "contact_name"] + KEEP], on="_k", how="left").drop(columns="_k")
    matched = int(out["x"].notna().sum())
    pct = 100 * matched / max(len(out), 1)
    print(f"  {method}/{fset}/{run_id}: {matched}/{len(out)} contacts matched ({pct:.1f}%)")
    if pct < 90:
        print("  !! low match rate — check the electrode-name normalisation before trusting this",
              file=sys.stderr)

    recon = rd / "recon"
    recon.mkdir(parents=True, exist_ok=True)
    dest = recon / f"{method}_{fset}_{run_id}__with_fsaverage.csv"
    out.to_csv(dest, index=False)
    unmatched = out[out["x"].isna()]
    unmatched.to_csv(recon / "UNMATCHED_contacts.csv", index=False)
    print(f"  -> {dest.name}  ({dest.stat().st_size/1e6:.2f} MB)")
    print(f"  -> UNMATCHED_contacts.csv  ({len(unmatched)} rows)")
    print("  NOTE: the glassbrain PNGs are NOT written here — run 252 for those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
