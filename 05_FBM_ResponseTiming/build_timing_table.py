#!/usr/bin/env python3
"""
build_timing_table.py - one row per electrode x condition, with response timing
and every group label that electrode already carries.

    python build_timing_table.py
    python build_timing_table.py --band 70 150 --thr-db 1.0 --out outputs/timing

Output: outputs/timing/timing_table.csv  plus  meta.json recording the settings,
the source runs, and the coverage that actually made it in.

Read the caveats in functions/lf_rt.py before interpreting anything here. The two
that bite hardest: t=0 is the GO CUE and not speech onset, and the cubes are
trial-averaged so no per-trial response time can be correlated from them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "functions"))
sys.path.insert(0, str(HERE.parent / "01_FBM_Analysis"))
import lf_rt as R  # noqa: E402
from functions import config as cfg  # noqa: E402

CLUST = HERE.parent / "02_FBM_Clustering" / "outputs" / "clustering"
RUNS = {
    "kmeans_concat_hg":       CLUST / "kmeans" / "concat_hg" / "runs" / "20260817_171544",
    "kmeans_concat_rawds":    CLUST / "kmeans" / "concat_rawds" / "runs" / "20260817_171634",
    "hier_concat_hg":         CLUST / "hierarchical" / "concat_hg" / "runs" / "20260817_171627",
    "hier_concat_rawds":      CLUST / "hierarchical" / "concat_rawds" / "runs" / "20260817_171733",
}
DEC = CLUST / "decomposition" / "concat_hg"
POOL = HERE.parent / "04_FBM_Pooling" / "outputs" / "pooling" / "pool_web" / "contacts_pool.csv"


def response_durations() -> pd.DataFrame:
    """Per patient x condition response-duration stats, straight from the 05 IQR
    reports. Used as a patient-level covariate: a slow responder could shift every
    one of that patient's electrodes, which would masquerade as a group effect."""
    rt_root = os.path.join(cfg.outputs_root, "05_ERSP_LM_RAWONLY_RealTime")
    recs = []
    for pid in sorted(os.listdir(rt_root)):
        p = os.path.join(rt_root, pid, "LM", "Report", f"{pid}_IQR.tsv")
        if os.path.isfile(p):
            try:
                recs.append(pd.read_csv(p, sep="\t"))
            except Exception:
                pass
    if not recs:
        return pd.DataFrame(columns=["patient_id", "condition"])
    d = pd.concat(recs, ignore_index=True)
    keep = [c for c in ("patient_id", "condition", "n_in", "n_kept",
                        "post_min", "post_med", "post_max") if c in d.columns]
    d = d[keep].rename(columns={"post_med": "resp_med_s", "post_min": "resp_min_s",
                                "post_max": "resp_max_s", "n_kept": "n_trials_kept",
                                "n_in": "n_trials_in"})
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", nargs=2, type=float, default=list(R.HG_BAND))
    ap.add_argument("--thr-db", type=float, default=1.0)
    ap.add_argument("--min-ms", type=float, default=100.0)
    ap.add_argument("--resp", nargs=2, type=float, default=[0.0, 5.0])
    ap.add_argument("--tree", choices=["RT", "TN"], default="RT",
                    help="RT = GO-aligned real time (05); TN = warped %% of trial (04)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    TN = a.tree == "TN"
    out = Path(a.out) if a.out else (HERE / "outputs" /
                                     ("timing_tn" if TN else "timing"))
    out.mkdir(parents=True, exist_ok=True)
    rt_root = os.path.join(cfg.outputs_root,
                           "04_ersp_LM_RAWONLY" if TN else "05_ERSP_LM_RAWONLY_RealTime")
    band = tuple(a.band)
    # In warped time the response portion is 50-100%% of the trial and the units
    # are percent, so the resp window and the sustain requirement both change.
    resp = (R.TN_GO_PCT, 100.0) if TN else tuple(a.resp)
    min_units = 2.0 if TN else a.min_ms

    print(f"scanning {rt_root}  [{a.tree}]")
    recs = list(R.iter_cubes_tn(rt_root) if TN else R.iter_cubes(rt_root))
    print(f"  {len(recs)} cubes")

    rows, bad = [], 0
    for i, rec in enumerate(recs):
        try:
            if TN:
                A = np.load(rec["path"])
                tr = R.to_grid_tn(R.band_trace(A, band))
                row = {k2: v for k2, v in rec.items() if k2 != "path"}
                row["n_cols"] = int(A.shape[1])
                from dataclasses import asdict as _asdict
                row.update(_asdict(R.timing_features(
                    tr, grid=R.TN_GRID, resp=resp, thr_db=a.thr_db,
                    min_ms=min_units * 1000.0)))
                rows.append(row)
            else:
                rows.append(R.timing_row(rec, band=band, resp=resp,
                                         thr_db=a.thr_db, min_ms=a.min_ms))
        except Exception as e:
            bad += 1
            if bad <= 5:
                print(f"  !! {rec['patient_id']}/{rec['condition']}/"
                      f"{rec['electrode']}: {type(e).__name__}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(recs)}")
    T = pd.DataFrame(rows)
    print(f"  built {len(T)} rows ({bad} failed)")

    # ---- attach every group label this electrode already has
    runs = {k: str(v) for k, v in RUNS.items() if (v / "labels.csv").exists()}
    cl = R.load_clusters(str(CLUST), runs)
    T = T.merge(cl, on=["patient_id", "contact_norm"], how="left")

    if (DEC / "electrode_loadings.csv").exists():
        T = T.merge(R.load_loadings(str(DEC)), on=["patient_id", "contact_norm"], how="left")
        wcols = [c for c in T.columns if c.startswith("w") and c[1:].isdigit()]
        if wcols:
            # argmax over an all-NaN row returns 0, which would hand every
            # electrode that FAILED the merge a confident membership in
            # component 0. Mask on "did this row actually get weights".
            Wm = T[wcols].to_numpy(dtype=float)
            has = np.isfinite(Wm).any(axis=1)
            lead = np.full(len(T), np.nan)
            top = np.full(len(T), np.nan)
            mar = np.full(len(T), np.nan)
            if has.any():
                sub = Wm[has]
                lead[has] = np.nanargmax(sub, axis=1)
                srt = np.sort(np.nan_to_num(sub, nan=-np.inf), axis=1)
                top[has] = srt[:, -1]
                mar[has] = srt[:, -1] - srt[:, -2]
            T["cnmf_lead"] = lead
            T["cnmf_top_weight"] = top
            T["cnmf_margin"] = mar
            # A lead is only meaningful when it actually leads; the decomposition's
            # own finding is that for most electrodes it does not.
            T.loc[T["cnmf_top_weight"] < 0.5, "cnmf_lead_confident"] = np.nan
            T.loc[T["cnmf_top_weight"] >= 0.5, "cnmf_lead_confident"] = \
                T.loc[T["cnmf_top_weight"] >= 0.5, "cnmf_lead"]
    else:
        print(f"  (no decomposition at {DEC} - skipping graded weights)")

    if POOL.exists():
        T = T.merge(R.load_roles(str(POOL)), on=["patient_id", "contact_norm"], how="left")

    rd = response_durations()
    if len(rd):
        T = T.merge(rd, on=["patient_id", "condition"], how="left")

    T.to_csv(out / "timing_table.csv", index=False)

    cov = {}
    for k in list(runs) + (["cnmf_lead"] if "cnmf_lead" in T.columns else []) + \
            (["role"] if "role" in T.columns else []):
        cov[k] = int(T[k].notna().sum())
    meta = dict(
        rt_root=rt_root, band=list(band), thr_db=a.thr_db, min_ms=a.min_ms,
        tree=a.tree, resp_window=list(resp),
        grid_step=(100.0 / 300.0 if TN else 0.02),
        window=list(R.TN_WINDOW if TN else R.RT_WINDOW),
        units=("percent of warped trial" if TN else "seconds after GO"),
        t0_is=("0%% = stimulus onset, 50%% = GO cue" if TN else
               "GO cue (stimulus offset); no speech-onset event exists in this dataset"),
        cubes_found=len(recs), rows_built=len(T), failed=bad,
        n_patients=int(T["patient_id"].nunique()),
        per_condition={k: int(v) for k, v in T["condition"].value_counts().items()},
        electrodes_per_condition={
            c: int(T.loc[T.condition == c, "contact_norm"].nunique())
            for c in sorted(T["condition"].unique())},
        label_coverage=cov,
        source_runs={k: str(Path(v).relative_to(HERE.parent)).replace("\\", "/")
                     for k, v in runs.items()},
        written=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n  {len(T)} rows, {T['patient_id'].nunique()} patients")
    print(f"  per condition: {meta['per_condition']}")
    print(f"  label coverage: {cov}")
    print(f"  -> {out / 'timing_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
