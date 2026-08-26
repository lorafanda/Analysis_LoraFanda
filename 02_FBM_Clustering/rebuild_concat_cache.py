#!/usr/bin/env python3
"""
rebuild_concat_cache.py - STEP 1, and it must finish before anything else starts.

WHY A NEW CACHE DIRECTORY AND NOT A REBUILD IN PLACE.

lf_dataset.prepare_dataset decides a cache hit with `cached_params == params`, and
`params` holds input_dir, task, conditions, the thresholds and a schema tag. It holds
NO patient list, NO cube timestamps and NO content hash. EL033 and PAT_3965 have new
ERSP cubes but none of those parameters changed, so a rebuild against the existing
concat_source_v2 would print "cache hit" and hand back the OLD df_meta - including the
OLD high_activity flags. The re-gating would silently not happen.

Writing to a NEW directory is the only thing that forces the walk of ERSP_matrix and
therefore forces high_activity to be recomputed from the current cubes.

WHAT high_activity IS. Per electrode and condition, over the full 0-400 Hz cube:
    prop_above_pos >= 0.02  OR  prop_below_neg >= 0.04
An electrode enters the GATED set if that holds in at least one of the three
conditions. That is the gate the whole cohort size depends on.

    python rebuild_concat_cache.py --dry-run
    python rebuild_concat_cache.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

import lf_concat as CC          # noqa: E402
import lf_dataset as LD         # noqa: E402

CACHE_ROOT = ROOT / "outputs" / "_dataset"
NEW_CACHE = CACHE_ROOT / "concat_source_v3"
OLD_CACHE = CACHE_ROOT / "concat_source_v2"
INPUT_DIR = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM"
                 r"\Analysis_LoraFanda\01_FBM_Analysis\outputs\04_ersp_LM_RAWONLY")
CONDITIONS = ("audio", "picture", "reading")
OUT = ROOT / "outputs" / "clustering" / "cohort_v3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        print("pass --dry-run or --apply")
        return 2

    inp = INPUT_DIR if INPUT_DIR.exists() else (
        ROOT.parent / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY").resolve()
    print(f"input   : {inp}")
    print(f"new     : {NEW_CACHE}")
    print(f"old     : {OLD_CACHE}  {'(present)' if OLD_CACHE.exists() else '(absent)'}")

    if NEW_CACHE.exists() and any(NEW_CACHE.iterdir()):
        print(f"\n!! {NEW_CACHE.name} already exists and is not empty.")
        print("   A cache hit there would be just as stale as v2. Delete it or bump "
              "to v4 if the cubes have changed again since it was built.")
        if a.apply:
            return 1

    if a.dry_run:
        print("\nDRY RUN — would build the cache and report the cohort. Nothing written.")
        return 0

    t0 = time.time()
    df, X = CC.build_concat_dataset(inp, conditions=CONDITIONS,
                                    require_high_activity=True,
                                    cache_dir=NEW_CACHE, verbose=True)
    print(f"\nbuilt in {time.time()-t0:.0f}s")
    print(f"GATED cohort : {len(df)} electrodes · {df['patient_id'].nunique()} patients")
    print(f"X_concat     : {X.shape}")

    # the ungated count, from the same fresh cache — this is what concat_hg_all uses
    df_all, X_all = CC.build_concat_dataset(inp, conditions=CONDITIONS,
                                            require_high_activity=False,
                                            cache_dir=NEW_CACHE, verbose=False)
    print(f"UNGATED      : {len(df_all)} electrodes")

    OUT.mkdir(parents=True, exist_ok=True)
    per = (df.groupby("patient_id").size().rename("gated")
           .to_frame().join(df_all.groupby("patient_id").size().rename("ungated")))
    per["pct_gated"] = (100 * per.gated / per.ungated).round(1)

    # what changed against the cohort the published runs were fitted on
    if OLD_CACHE.exists():
        try:
            old_meta = pd.read_parquet(OLD_CACHE / "df_meta.parquet")
            oldg = (old_meta[old_meta.get("high_activity", False).astype(bool)]
                    .assign(cn=lambda d: d["electrode"].map(CC.normalize_label))
                    .drop_duplicates(["patient_id", "cn"])
                    .groupby("patient_id").size().rename("gated_v2"))
            per = per.join(oldg)
            per["delta"] = per.gated - per.gated_v2
        except Exception as e:
            print(f"  (could not read v2 for comparison: {type(e).__name__}: {e})")

    per.to_csv(OUT / "cohort_v3_per_patient.csv")
    print("\nper patient:")
    print(per.to_string())
    if "delta" in per.columns:
        ch = per[per.delta.fillna(0) != 0]
        print(f"\npatients whose GATED count moved vs v2: {len(ch)}")
        if len(ch):
            print(ch[["gated_v2", "gated", "delta"]].to_string())

    (OUT / "cohort_v3_summary.json").write_text(json.dumps(dict(
        input_dir=str(inp), cache=str(NEW_CACHE),
        n_gated=int(len(df)), n_ungated=int(len(df_all)),
        n_patients=int(df["patient_id"].nunique()),
        n_freq=int(X.shape[1]), n_time_concat=int(X.shape[2]),
        conditions=list(CONDITIONS),
        written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(f"\nwrote -> {OUT}")
    print("\nNEXT: run 240 / 241 / 242 in parallel. They all read this cache "
          "read-only and must NOT rebuild it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
