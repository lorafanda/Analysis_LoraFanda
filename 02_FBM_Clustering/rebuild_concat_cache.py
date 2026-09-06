#!/usr/bin/env python3
"""
rebuild_concat_cache.py - STEP 1, and it must finish before anything else starts.

WHY A NEW CACHE DIRECTORY AND NOT A REBUILD IN PLACE.

lf_dataset.prepare_dataset decides a cache hit with `cached_params == params`, and
`params` holds input_dir, task, conditions, the thresholds and a schema tag. It holds
NO patient list, NO cube timestamps and NO content hash. A patient with new ERSP cubes -
or a patient who has just acquired ERSP cubes for the first time - changes none of those
parameters, so a rebuild against the existing cache would print "cache hit" and hand back
the OLD df_meta, including the OLD high_activity flags. The re-gating would silently not
happen.

Writing to a NEW directory is the only thing that forces the walk of ERSP_matrix and
therefore forces high_activity to be recomputed from the current cubes.

WHAT high_activity IS. Per electrode and condition, over the full 0-400 Hz cube:
    prop_above_pos >= 0.02  OR  prop_below_neg >= 0.04
An electrode enters the GATED set if that holds in at least one of the three
conditions. That is the gate the whole cohort size depends on.

WHAT ELSE DECIDES THE COHORT, and is NOT in the cache. The cache is the ungated source:
it holds every electrode that survived the non-neural, microelectrode and noisy-shaft
filters. Subdural grid contacts and whole excluded patients are removed one step later,
by lf_concat.build_concat_dataset reading DEFAULT_EXCLUDE_PATIENTS and GRID_SHAFTS. So
excluding a patient does NOT need a new cache; adding or re-running a patient does.

    python rebuild_concat_cache.py --dry-run
    python rebuild_concat_cache.py --apply

    --version N   build concat_source_v<N> and compare against v<N-1>  (default: 5)

AFTER IT FINISHES the whole chain downstream is stale, because every run was fitted on
the previous cohort: 240 / 241 / 242 (and 243), then 249, then 252, then the figures.
The script says so again at the end.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import lf_concat as CC          # noqa: E402
import lf_dataset as LD         # noqa: E402
import check_recon_name_match as NM   # noqa: E402

CACHE_ROOT = ROOT / "outputs" / "_dataset"
INPUT_DIR = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM"
                 r"\Analysis_LoraFanda\01_FBM_Analysis\outputs\04_ersp_LM_RAWONLY")
CONDITIONS = ("audio", "picture", "reading")


def input_patients(inp: Path, task: str = "LM") -> set:
    """The patients the walk would actually find: a folder with an ERSP_matrix in it."""
    return {d.name for d in inp.iterdir()
            if d.is_dir() and (d / task / "ERSP_matrix").is_dir()}


def cache_patients(cache: Path) -> set:
    f = cache / "df_meta.parquet"
    if not f.exists():
        return set()
    return set(pd.read_parquet(f, columns=["patient_id"]).patient_id.unique())


def name_match(cache: Path, out: Path, label: str):
    """The recorded-vs-reconstructed name check, on this cache. See
    check_recon_name_match.py - the join is silent, so it is checked explicitly."""
    if not NM.COORDS.exists():
        print(f"  (no {NM.COORDS.name}: skipping the name check)")
        return
    r = NM.report(cache)
    pp = r["per_patient"]
    out.mkdir(parents=True, exist_ok=True)
    for k in ("per_patient", "unmatched_recorded", "unmatched_recon", "candidates",
              "shafts"):
        r[k].to_csv(out / f"name_match_{label}_{k}.csv", index=False)
    bad = pp[pp.unmatched > 0]
    print(f"\nNAME MATCH, recorded vs reconstructed ({label})")
    print(f"  {int(pp.unmatched.sum())} recorded electrodes have no recon contact; "
          f"{int(pp.unmatched_in_cohort.sum())} of them are in the cohort and so lose "
          f"their coordinates")
    if len(bad):
        print(bad[["patient", "recorded", "in_cohort", "matched", "unmatched",
                   "unmatched_in_cohort", "recon", "recon_unmatched",
                   "pct_matched"]].to_string(index=False))
        print("\n  likeliest alias per shaft (a suggestion, not a rename):")
        print("  " + r["shafts"].to_string(index=False).replace("\n", "\n  "))
    one = pp[(pp.side != "both") & (pp.recorded > 0)]
    if len(one):
        print("\n  recorded with NO reconstruction at all:")
        print("  " + one[["patient", "recorded"]].to_string(index=False)
              .replace("\n", "\n  "))
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--version", type=int, default=5,
                    help="cache version to build (default 5: concat_source_v5)")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        print("pass --dry-run or --apply")
        return 2

    v = a.version
    new_cache = CACHE_ROOT / f"concat_source_v{v}"
    old_cache = CACHE_ROOT / f"concat_source_v{v-1}"
    out = ROOT / "outputs" / "clustering" / f"cohort_v{v}"

    inp = INPUT_DIR if INPUT_DIR.exists() else (
        ROOT.parent / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY").resolve()
    print(f"input   : {inp}")
    print(f"new     : {new_cache}")
    print(f"old     : {old_cache}  {'(present)' if old_cache.exists() else '(absent)'}")
    print(f"excluded patients : {CC.DEFAULT_EXCLUDE_PATIENTS or '(none)'}")
    print(f"grid shafts       : "
          + (", ".join(f"{p} {'/'.join(s)}" for p, s in LD.GRID_SHAFTS.items()) or "(none)"))

    # WHO IS ABOUT TO CHANGE. The cache carries no patient list, so this comparison -
    # the folders on disk against the patients in the old cache - is the only warning
    # that the cohort is moving, and it is the reason a new directory is being written.
    have, had = input_patients(inp), cache_patients(old_cache)
    added, gone = sorted(have - had), sorted(had - have)
    excl = sorted(set(CC.DEFAULT_EXCLUDE_PATIENTS) & have)
    print(f"\npatient folders with ERSP: {len(have)}   in {old_cache.name}: {len(had)}")
    if added:
        print(f"  NEW, will enter the cohort   : {', '.join(added)}")
    if gone:
        print(f"  gone from the input tree     : {', '.join(gone)}")
    if excl:
        print(f"  present but EXCLUDED by lf_concat: {', '.join(excl)}  "
              f"(in the cache, dropped by build_concat_dataset)")

    if new_cache.exists() and any(new_cache.iterdir()):
        print(f"\n!! {new_cache.name} already exists and is not empty.")
        print("   A cache hit there would be just as stale as the one it replaces. "
              "Delete it, or bump --version if the cubes have changed again since it "
              "was built.")
        if a.apply:
            return 1

    if a.dry_run:
        print("\nDRY RUN - nothing written. The name check below is on the OLD cache; "
              "a patient that has no ERSP yet shows as 'recon only'.")
        if old_cache.exists():
            name_match(old_cache, out, f"dryrun_{old_cache.name}")
        return 0

    t0 = time.time()
    df, X = CC.build_concat_dataset(inp, conditions=CONDITIONS,
                                    require_high_activity=True,
                                    cache_dir=new_cache, verbose=True)
    print(f"\nbuilt in {time.time()-t0:.0f}s")
    print(f"GATED cohort : {len(df)} electrodes · {df['patient_id'].nunique()} patients")
    print(f"X_concat     : {X.shape}")

    # the ungated count, from the same fresh cache - this is what concat_hg_all uses
    df_all, X_all = CC.build_concat_dataset(inp, conditions=CONDITIONS,
                                            require_high_activity=False,
                                            cache_dir=new_cache, verbose=False)
    print(f"UNGATED      : {len(df_all)} electrodes")

    out.mkdir(parents=True, exist_ok=True)
    per = (df.groupby("patient_id").size().rename("gated")
           .to_frame().join(df_all.groupby("patient_id").size().rename("ungated")))
    per["pct_gated"] = (100 * per.gated / per.ungated).round(1)

    # WHO IS IN THE CACHE BUT CONTRIBUTES NOTHING. A patient with no gated electrode,
    # or with a condition missing, never appears in `per` at all - it is simply absent,
    # which is the least visible way for a patient to leave a cohort.
    inc = NM.load_recorded(new_cache, CC.DEFAULT_EXCLUDE_PATIENTS)
    if int(inc.in_cohort.sum()) != len(df):
        print(f"  [WARN] the cohort rule reproduced here gives "
              f"{int(inc.in_cohort.sum())} electrodes, build_concat_dataset gives "
              f"{len(df)} - the two have drifted apart")
    silent = []
    for pid, gp in inc.groupby("patient_id"):
        if int(gp.in_cohort.sum()):
            continue
        if pid in set(CC.DEFAULT_EXCLUDE_PATIENTS):
            why = "excluded by rule"
        elif int(gp.n_cond.max()) < inc.n_cond.max():
            why = (f"only {int(gp.n_cond.max())} of {int(inc.n_cond.max())} conditions "
                   "on disk")
        elif not bool(gp.gated.any()):
            why = "no electrode passes the amplitude gate"
        else:
            why = "every electrode removed by a contact filter"
        silent.append((pid, len(gp), why))
    if silent:
        print("\nIN THE CACHE, NOT IN THE COHORT:")
        for pid, n, why in silent:
            print(f"  {pid:<10} {n:>4} electrodes   {why}")

    # what changed against the cohort the published runs were fitted on. BOTH SIDES BY
    # THE SAME RULE: an earlier version counted any high_activity on the old side and
    # the real cohort on the new one, which reported patients as lost that had never
    # been in the cohort.
    if old_cache.exists():
        try:
            prev = NM.load_recorded(old_cache, CC.DEFAULT_EXCLUDE_PATIENTS)
            oldg = (prev[prev.in_cohort].groupby("patient_id").size()
                    .rename("gated_prev"))
            per = per.join(oldg, how="outer")
            per["delta"] = per.gated.fillna(0) - per.gated_prev.fillna(0)
            # a patient the EXCLUSION LIST removed is not a data change; say so
            # separately rather than showing it as a loss
            was = NM.load_recorded(old_cache, ())
            for pid in CC.DEFAULT_EXCLUDE_PATIENTS:
                n = int(was[(was.patient_id == pid) & was.in_cohort].shape[0])
                if n:
                    print(f"  note: {pid} contributed {n} electrodes to "
                          f"{old_cache.name} and is now excluded by rule, not by data")
        except Exception as e:
            print(f"  (could not read {old_cache.name} for comparison: "
                  f"{type(e).__name__}: {e})")

    per.to_csv(out / f"cohort_v{v}_per_patient.csv")
    print("\nper patient:")
    print(per.to_string())
    if "delta" in per.columns:
        ch = per[per.delta.fillna(0) != 0]
        print(f"\npatients whose GATED count moved vs {old_cache.name}: {len(ch)}")
        if len(ch):
            print(ch[["gated_prev", "gated", "delta"]].to_string())

    name_match(new_cache, out, new_cache.name)

    (out / f"cohort_v{v}_summary.json").write_text(json.dumps(dict(
        input_dir=str(inp), cache=str(new_cache),
        n_gated=int(len(df)), n_ungated=int(len(df_all)),
        n_patients=int(df["patient_id"].nunique()),
        excluded_patients=list(CC.DEFAULT_EXCLUDE_PATIENTS),
        patients_added_vs_previous=added, patients_gone_vs_previous=gone,
        n_freq=int(X.shape[1]), n_time_concat=int(X.shape[2]),
        conditions=list(CONDITIONS),
        written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(f"\nwrote -> {out}")
    print(f"\nNEXT, in order, because every run below was fitted on the OLD cohort:")
    print("  1. point lf_concat.DEFAULT_CONCAT_CACHE at "
          f"{new_cache.name} (do this before anything reads the cache)")
    print("  2. 240 / 241 / 242 (and 243) - they read this cache read-only and must "
          "NOT rebuild it")
    print("  3. 249, then 252 (recon exports), then the paper figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
