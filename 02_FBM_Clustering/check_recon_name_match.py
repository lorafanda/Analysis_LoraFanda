#!/usr/bin/env python3
"""check_recon_name_match.py - do the RECORDED electrodes and the RECONSTRUCTED
contacts have the same names, patient by patient?

WHY THIS EXISTS. Every figure that puts an electrode on a brain joins two tables that
were written by different people at different times:

    the dataset      one ERSP cube per electrode and condition, named by the RECORDING
                     montage - the label the amplifier channel carried
    the recon        one contact per row in ALL_PATIENTS_contacts_fsaverage.csv, named
                     by the RECONSTRUCTION - the label in the patient's elec_recon files

The join is `patient + normalize_label(name)`, where normalize_label strips underscores
and dashes and upper-cases (aH_R-1 -> AHR1). It is a SILENT join: an electrode whose two
labels disagree does not error, it simply has no coordinate, drops off every brain and
out of the LanA atlas, and the only trace is a smaller number in a caption. On the
2026-09-05 cohort that was 49 electrodes across four patients - EL034 MFG-L vs MFG,
EL043 pIns vs pl, EL045 PlaT_L vs PlanTL, PAT_6619 OFA/OFP vs OFAD/OFPD.

WHAT IT REPORTS, PER PATIENT, IN BOTH DIRECTIONS:

    recorded but not reconstructed   the ones that lose their coordinates. Split by
                                     whether they are in the analysed cohort, because
                                     only those actually cost a figure anything.
    reconstructed but not recorded   the other side of the same join. A large number
                                     here is NORMAL and not a fault: a patient has many
                                     contacts that were never analysed (white matter,
                                     shafts with no usable ERSP, contacts dropped as
                                     noisy). It is listed so that a patient whose two
                                     sides disagree ENTIRELY - a whole shaft renamed -
                                     is visible as a pair of matching counts.

ALIAS CANDIDATES. For every unmatched recorded electrode the script proposes the recon
contacts of the same patient that carry the SAME CONTACT NUMBER, ranked by string
similarity of the shaft. It is a suggestion for a human to accept or reject, never an
automatic rename: MFGL -> MFG is obvious, pIns -> pl is not, and nothing here can tell
a renamed shaft from a genuinely missing one. The rollup by shaft is the table to read.

    python check_recon_name_match.py
    python check_recon_name_match.py --cache outputs/_dataset/concat_source_v4
    python check_recon_name_match.py --out <dir>          # e.g. a scratch folder

Read-only with respect to the dataset and the recon: it writes CSVs into --out and
nothing else.
"""
from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))

import lf_concat as CC                                          # noqa: E402
from lf_dataset import (is_grid_electrode, is_micro_electrode,  # noqa: E402
                        is_non_neural_electrode)

COORDS = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coords"
          / "ALL_PATIENTS_contacts_fsaverage.csv")
OUT_ROOT = ROOT / "outputs" / "clustering" / "name_match"


def shaft_of(norm_name: str) -> str:
    """AHR1 -> AHR. The physical electrode; a rename happens at this level, not per
    contact, which is why the actionable table is the one rolled up to it."""
    return re.sub(r"\d+$", "", str(norm_name))


def number_of(norm_name: str) -> str:
    m = re.search(r"(\d+)$", str(norm_name))
    return m.group(1) if m else ""


# ---- the two sides -----------------------------------------------------------
def load_recorded(cache: Path, exclude_patients=()) -> pd.DataFrame:
    """One row per recorded electrode, with what the cohort rule would do to it.

    `in_cohort` is the SAME rule the figures apply, in the same order as
    00_paper2_figure0_coverage.fate(): every condition present, the amplitude gate
    passed, and none of the pipeline's filters (excluded patient, subdural grid
    contact, microelectrode, non-neural channel) firing. Reproduced here rather than
    imported so this script can be pointed at a cache that no run has been fitted on
    yet - which is exactly the moment the names need checking.
    """
    t = pd.read_parquet(cache / "df_meta.parquet")
    n_cond_expected = t["condition"].nunique()
    per = (t.groupby(["patient_id", "electrode"], as_index=False)
             .agg(n_cond=("condition", "nunique"),
                  n_high=("high_activity", "sum")))
    per["norm"] = [CC.normalize_label(e) for e in per.electrode]
    per["key"] = per.patient_id + "|" + per.norm
    grid = [bool(is_grid_electrode(e, p))
            for e, p in zip(per.electrode, per.patient_id)]
    micro = [bool(is_micro_electrode(e, p))
             for e, p in zip(per.electrode, per.patient_id)]
    nonn = [bool(is_non_neural_electrode(e)) for e in per.electrode]
    per["excluded_patient"] = per.patient_id.isin(set(exclude_patients))
    per["filtered"] = pd.Series(grid) | pd.Series(micro) | pd.Series(nonn)
    per["gated"] = per.n_high > 0
    per["in_cohort"] = (per.gated & (per.n_cond >= n_cond_expected)
                        & ~per.filtered & ~per.excluded_patient)
    return per


def load_recon(coords: Path) -> pd.DataFrame:
    """One row per reconstructed contact, keyed both ways.

    `name` is what the join uses. `name_raw` is kept because a match that appears ONLY
    through the raw label is a finding in itself: it means the recon's own tidy-up of
    the label is what broke the join, and the fix is a one-line alias rather than a
    re-reconstruction.
    """
    c = pd.read_csv(coords)
    c["norm"] = [CC.normalize_label(n) for n in c["name"]]
    c["norm_raw"] = [CC.normalize_label(n) for n in c.get("name_raw", c["name"])]
    c["key"] = c.patient + "|" + c.norm
    c["key_raw"] = c.patient + "|" + c.norm_raw
    c["has_xyz"] = c[["x", "y", "z"]].notna().all(axis=1)
    return c


# ---- the join ----------------------------------------------------------------
def match(rec: pd.DataFrame, con: pd.DataFrame):
    """Add the match columns to both sides. One rule, stated once, used twice."""
    keys, keys_raw = set(con.key), set(con.key_raw)
    xyz_keys = set(con.loc[con.has_xyz, "key"])
    rec = rec.copy()
    rec["matched"] = rec.key.isin(keys)
    rec["matched_via_raw_only"] = ~rec.matched & rec.key.isin(keys_raw)
    rec["has_xyz"] = rec.key.isin(xyz_keys)
    con = con.copy()
    con["recorded"] = con.key.isin(set(rec.key))
    return rec, con


def per_patient(rec: pd.DataFrame, con: pd.DataFrame) -> pd.DataFrame:
    """The table Lora asked for: the split, one row per patient."""
    a = (rec.groupby("patient_id")
            .agg(recorded=("key", "size"),
                 in_cohort=("in_cohort", "sum"),
                 matched=("matched", "sum"),
                 matched_in_cohort=("matched", lambda s: int((s & rec.loc[s.index, "in_cohort"]).sum())),
                 via_raw_only=("matched_via_raw_only", "sum"),
                 excluded_patient=("excluded_patient", "max"))
            .reset_index().rename(columns={"patient_id": "patient"}))
    a["unmatched"] = a.recorded - a.matched
    a["unmatched_in_cohort"] = a.in_cohort - a.matched_in_cohort
    b = (con.groupby("patient")
            .agg(recon=("key", "size"), recon_recorded=("recorded", "sum"))
            .reset_index())
    b["recon_unmatched"] = b.recon - b.recon_recorded
    t = a.merge(b, on="patient", how="outer").fillna(
        {"recorded": 0, "in_cohort": 0, "matched": 0, "matched_in_cohort": 0,
         "via_raw_only": 0, "unmatched": 0, "unmatched_in_cohort": 0,
         "recon": 0, "recon_recorded": 0, "recon_unmatched": 0})
    for c in ("recorded", "in_cohort", "matched", "matched_in_cohort", "via_raw_only",
              "unmatched", "unmatched_in_cohort", "recon", "recon_recorded",
              "recon_unmatched"):
        t[c] = t[c].astype(int)
    t["excluded_patient"] = t.excluded_patient.fillna(False).astype(bool)
    t["pct_matched"] = (100 * t.matched / t.recorded.where(t.recorded > 0)).round(1)
    # the patients present on one side only - a recon that was never analysed, or a
    # patient recorded with no reconstruction at all, which no per-contact number shows
    t["side"] = ["both" if r and c else ("recorded only" if r else "recon only")
                 for r, c in zip(t.recorded > 0, t.recon > 0)]
    return t.sort_values(["unmatched_in_cohort", "unmatched"], ascending=False)


# ---- what a mismatch probably is ---------------------------------------------
def alias_candidates(rec: pd.DataFrame, con: pd.DataFrame, top: int = 3):
    """For each unmatched recorded electrode, the recon contacts it is most likely to be.

    Restricted to the SAME PATIENT and the SAME CONTACT NUMBER, then ranked by string
    similarity of the shaft. The number restriction is what makes the list short and
    readable: a contact called 7 can only be another shaft's 7.
    """
    rows = []
    con_by_pat = {p: g for p, g in con.groupby("patient")}
    for r in rec[~rec.matched].itertuples():
        g = con_by_pat.get(r.patient_id)
        if g is None:
            rows.append(dict(patient=r.patient_id, electrode=r.electrode, norm=r.norm,
                             in_cohort=bool(r.in_cohort), candidate="",
                             candidate_shaft="", similarity=float("nan"),
                             note="no reconstruction for this patient"))
            continue
        num, sh = number_of(r.norm), shaft_of(r.norm)
        pool = g[g.norm.map(number_of) == num] if num else g
        cand = sorted(((SequenceMatcher(None, sh, shaft_of(n)).ratio(), n, nr)
                       for n, nr in zip(pool.norm, pool["name"])), reverse=True)[:top]
        if not cand:
            rows.append(dict(patient=r.patient_id, electrode=r.electrode, norm=r.norm,
                             in_cohort=bool(r.in_cohort), candidate="",
                             candidate_shaft="", similarity=float("nan"),
                             note=f"no recon contact numbered {num} in this patient"))
            continue
        for s_, n_, raw_ in cand:
            rows.append(dict(patient=r.patient_id, electrode=r.electrode, norm=r.norm,
                             in_cohort=bool(r.in_cohort), candidate=raw_,
                             candidate_shaft=shaft_of(n_), similarity=round(s_, 3),
                             note=""))
    cands = pd.DataFrame(rows, columns=["patient", "electrode", "norm", "in_cohort",
                                        "candidate", "candidate_shaft", "similarity",
                                        "note"])
    # rolled up to the shaft: the level a rename actually happens at, and the only
    # table short enough to act on
    if len(cands):
        best = (cands.sort_values("similarity", ascending=False)
                     .drop_duplicates(["patient", "electrode"]))
        best = best.assign(ds_shaft=[shaft_of(n) for n in best.norm])
        roll = (best.groupby(["patient", "ds_shaft"])
                    .agg(n_unmatched=("electrode", "size"),
                         n_in_cohort=("in_cohort", "sum"),
                         best_recon_shaft=("candidate_shaft",
                                           lambda s: s.mode().iat[0] if len(s.mode()) else ""),
                         similarity=("similarity", "max"),
                         examples=("electrode", lambda s: ", ".join(sorted(s)[:4])))
                    .reset_index()
                    .sort_values(["n_in_cohort", "n_unmatched"], ascending=False))
    else:
        roll = pd.DataFrame(columns=["patient", "ds_shaft", "n_unmatched", "n_in_cohort",
                                     "best_recon_shaft", "similarity", "examples"])
    return cands, roll


# ---- driver ------------------------------------------------------------------
def report(cache: Path, coords: Path = COORDS, exclude_patients=None, top: int = 3):
    """Everything above, as DataFrames. Imported by rebuild_concat_cache.py."""
    exclude_patients = (CC.DEFAULT_EXCLUDE_PATIENTS if exclude_patients is None
                        else exclude_patients)
    rec = load_recorded(Path(cache), exclude_patients)
    con = load_recon(Path(coords))
    rec, con = match(rec, con)
    cands, roll = alias_candidates(rec, con, top)
    return dict(per_patient=per_patient(rec, con),
                unmatched_recorded=rec[~rec.matched].drop(columns=["key"]),
                unmatched_recon=con.loc[~con.recorded,
                                        ["patient", "name", "name_raw", "hemi",
                                         "is_wm", "has_xyz"]],
                candidates=cands, shafts=roll, recorded=rec, recon=con)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CC.DEFAULT_CONCAT_CACHE),
                    help="dataset cache to check (default: the current one)")
    ap.add_argument("--coords", default=str(COORDS))
    ap.add_argument("--out", default=None,
                    help="where the CSVs go (default: outputs/clustering/name_match/<cache>)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--keep-excluded-patients", action="store_true",
                    help="do not treat lf_concat.DEFAULT_EXCLUDE_PATIENTS as excluded")
    a = ap.parse_args()

    cache = Path(a.cache)
    if not (cache / "df_meta.parquet").exists():
        raise SystemExit(f"no df_meta.parquet in {cache}")
    if not Path(a.coords).exists():
        raise SystemExit(f"no coordinate table at {a.coords}")
    out = Path(a.out) if a.out else OUT_ROOT / cache.name
    print(f"dataset : {cache}")
    print(f"recon   : {a.coords}")

    r = report(cache, Path(a.coords),
               exclude_patients=() if a.keep_excluded_patients else None, top=a.top)
    pp = r["per_patient"]

    out.mkdir(parents=True, exist_ok=True)
    for name in ("per_patient", "unmatched_recorded", "unmatched_recon", "candidates",
                 "shafts"):
        r[name].to_csv(out / f"name_match_{name}.csv", index=False)

    print(f"\nPER PATIENT  ({len(pp)} patients; "
          f"{int(pp.unmatched.sum())} recorded electrodes have no recon contact, "
          f"{int(pp.unmatched_in_cohort.sum())} of them in the cohort)")
    cols = ["patient", "side", "excluded_patient", "recorded", "in_cohort", "matched",
            "unmatched", "unmatched_in_cohort", "via_raw_only", "recon",
            "recon_unmatched", "pct_matched"]
    print(pp[cols].to_string(index=False))

    bad = pp[pp.unmatched > 0]
    if len(bad):
        print(f"\nPATIENTS WITH A NAME MISMATCH ({len(bad)}), likeliest alias per shaft:")
        print(r["shafts"].to_string(index=False))
    else:
        print("\nno recorded electrode is missing from the reconstruction.")
    only = pp[pp.side != "both"]
    if len(only):
        print("\nONE SIDE ONLY:")
        print(only[["patient", "side", "recorded", "recon"]].to_string(index=False))
    print(f"\nwrote 5 CSVs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
