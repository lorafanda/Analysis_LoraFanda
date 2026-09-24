"""
lookup_mni_to_fsaverage.py - fsaverage coordinates for a patient who has no FreeSurfer
reconstruction but whose anatomy Lookup workbook carries MNI coordinates (EL051, EL052).

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/lookup_mni_to_fsaverage.py EL051 EL052 [--check]

WHY. 250_recon_fsaverage.ipynb puts a contact on fsaverage from the patient's own FreeSurfer
subject (LEPTOVOX -> tkrRAS -> spherical registration, or the talairach affine). EL051 and
EL052 have none: their anatomy/raw holds the T1, the CT, the plain fsaverage template and
the Lookup workbook. The workbook's mni_x/y/z is a position in MNI, and fsaverage's surface
RAS IS MNI305 (c_ras = 0), so the Lookup coordinate is the fsaverage coordinate - the same
class of placement as 250's affine-only route: the contact sits at its volume position, no
snap onto a patient surface. The Lookup's MNI is presumably MNI152; MNI152 and MNI305
differ by a small affine, at most about 2 mm, which is below what the glass brain or the
LanA sampling resolve. Checked on EL051: its 121 contacts are all right-sided, median 2.5 mm
from the nearest fsaverage rh pial vertex, the far ones the depth ones.

WHAT IT WRITES, in 250's schema and conventions, under 250_recon/fsaverage/coords/:
    <pid>_contacts_fsaverage.csv     patient, cohort, name, name_raw, hemi, x, y, z, is_wm,
                                     is_cortical, projection, dist_to_pial_mm, yeo7_network,
                                     yeo17_network
    ALL_PATIENTS_contacts_fsaverage.csv        rows of the patient replaced
    ALL_PATIENTS_contacts_fsaverage_nowm.csv   idem, is_wm == 0 only
and appends the patient to 250_recon/fsaverage/aparc_lookup.csv (patient, electrode, hemi,
aparc_label, vertex_idx, x, y, z; nearest vertex of the white surface, aparc name without
the hemisphere suffix - lf_anatomy.build_aparc_cache's convention).

    name        = the Lookup's `natus` column, the RECORDING name (what every join uses)
    name_raw    = the Lookup's `name` column
    hemi        = the Lookup's `side`
    is_wm       = the Lookup's isWM (isOut contacts are skipped: outside the brain)
    is_cortical = 1 - is_wm; projection = "lookup-mni"
    dist_to_pial_mm = 0.0 for cortical rows, NaN for WM rows - what the table holds for
                      every other patient (the column is not the measured distance)
    yeo7 / yeo17   = Yeo 2011 label at the nearest fsaverage pial vertex of the contact's
                     hemisphere for cortical rows; "WhiteMatter" for WM rows

The fsaverage surfaces and annots are read from the FreeSurfer template (the copy under
the patient's anatomy/raw/fsaverage, or FSAVERAGE below).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot, read_geometry
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NASAC = "//nasac-m2.unige.ch/m-HumanNeuronLab"
COORDS = os.path.join(REPO, "02_FBM_Clustering", "outputs", "250_recon", "fsaverage", "coords")
APARC = os.path.join(REPO, "02_FBM_Clustering", "outputs", "250_recon", "fsaverage", "aparc_lookup.csv")
FSAVERAGE = f"{NASAC}/DATARAW/SEEG_EXPERIMENTS_BERN/EL051/anatomy/raw/fsaverage"   # the plain template
SCHEMA = ["patient", "cohort", "name", "name_raw", "hemi", "x", "y", "z", "is_wm", "is_cortical",
          "projection", "dist_to_pial_mm", "yeo7_network", "yeo17_network"]


def lookup_rows(pid):
    f = sorted(glob.glob(f"{NASAC}/DATARAW/SEEG_EXPERIMENTS_BERN/{pid}/anatomy/raw/*Lookup*.xlsx"))
    assert f, f"{pid}: no Lookup workbook"
    ch = pd.read_excel(f[0], sheet_name="channels")
    lead = ch[ch["type"].astype(str).str.lower().isin(["lead", "micro"])]
    lead = lead[~lead["natus"].astype(str).str.strip().str.lower().isin(["unplugged", "nan", ""])]
    truthy = lambda v: str(v).strip().lower() in ("1", "1.0", "true")
    out = lead[~lead["isOut"].map(truthy)].copy()
    n_out = len(lead) - len(out)
    ok = out[["mni_x", "mni_y", "mni_z"]].notna().all(axis=1)
    assert ok.all(), f"{pid}: {int((~ok).sum())} plugged contacts without MNI coordinates: {out.loc[~ok, 'natus'].tolist()}"
    return out, os.path.basename(f[0]), n_out


def surfaces():
    S = {}
    for h in ("lh", "rh"):
        S[h] = {"pial": read_geometry(f"{FSAVERAGE}/surf/{h}.pial")[0], "white": read_geometry(f"{FSAVERAGE}/surf/{h}.white")[0]}
        for key, fn in (("aparc", f"{h}_aparc.annot"), ("yeo7", f"{h}_Yeo2011_7Networks_N1000.annot"), ("yeo17", f"{h}_Yeo2011_17Networks_N1000.annot")):
            p = f"{FSAVERAGE}/label/{fn}"
            if not os.path.exists(p):
                p = f"{FSAVERAGE}/label/{fn.replace('_', '.', 1)}"       # lh.aparc.annot spelling
            lab, _, names = read_annot(p)
            S[h][key] = (lab, [n.decode() if isinstance(n, bytes) else str(n) for n in names])
        S[h]["kd_pial"] = cKDTree(S[h]["pial"]); S[h]["kd_white"] = cKDTree(S[h]["white"])
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patients", nargs="+")
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    a = ap.parse_args()
    S = surfaces()
    all_path, nowm_path = os.path.join(COORDS, "ALL_PATIENTS_contacts_fsaverage.csv"), os.path.join(COORDS, "ALL_PATIENTS_contacts_fsaverage_nowm.csv")
    ALL = pd.read_csv(all_path); AP = pd.read_csv(APARC)
    for pid in a.patients:
        rows, src, n_out = lookup_rows(pid)
        xyz = rows[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
        side = rows["side"].astype(str).str.strip().str.upper().str[0].to_numpy()
        is_wm = rows["isWM"].map(lambda v: str(v).strip().lower() in ("1", "1.0", "true")).to_numpy()
        h_of = np.where(side == "L", "lh", "rh")
        yeo7, yeo17, aparc, vidx, dpial = [], [], [], [], []
        for p, h, wm in zip(xyz, h_of, is_wm):
            d, v = S[h]["kd_pial"].query(p); dpial.append(d)
            yeo7.append("WhiteMatter" if wm else S[h]["yeo7"][1][S[h]["yeo7"][0][v]])
            yeo17.append("WhiteMatter" if wm else S[h]["yeo17"][1][S[h]["yeo17"][0][v]])
            _, vw = S[h]["kd_white"].query(p); vidx.append(int(vw))
            aparc.append(S[h]["aparc"][1][S[h]["aparc"][0][vw]] if S[h]["aparc"][0][vw] >= 0 else "unknown")
        df = pd.DataFrame({
            "patient": pid, "cohort": "BERN", "name": rows["natus"].astype(str).str.strip().values,
            "name_raw": rows["name"].astype(str).values, "hemi": side,
            "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            "is_wm": is_wm.astype(int), "is_cortical": (~is_wm).astype(int), "projection": "lookup-mni",
            "dist_to_pial_mm": np.where(is_wm, np.nan, 0.0), "yeo7_network": yeo7, "yeo17_network": yeo17,
        })[SCHEMA]
        dp = np.asarray(dpial)
        print(f"[{pid}] {src}: {len(df)} contacts ({n_out} isOut skipped) · sides {dict(zip(*np.unique(side, return_counts=True)))} · WM {int(is_wm.sum())}"
              f" · distance to the nearest fsaverage pial vertex: median {np.median(dp):.1f} mm, max {dp.max():.1f} mm"
              f" · Yeo7 {dict(pd.Series(yeo7).value_counts())}")
        print(f"   aparc: {dict(pd.Series(aparc).value_counts().head(8))}")
        if a.check:
            continue
        df.to_csv(os.path.join(COORDS, f"{pid}_contacts_fsaverage.csv"), index=False)
        ALL = pd.concat([ALL[ALL["patient"] != pid], df], ignore_index=True)
        ap_rows = pd.DataFrame({"patient": pid, "electrode": df["name"], "hemi": h_of, "aparc_label": aparc, "vertex_idx": vidx,
                                "x": df["x"], "y": df["y"], "z": df["z"]})
        AP = pd.concat([AP[AP["patient"] != pid], ap_rows], ignore_index=True)
        print(f"   wrote {pid}_contacts_fsaverage.csv, {len(df)} rows into the ALL tables, {len(ap_rows)} into aparc_lookup.csv")
    if not a.check:
        ALL.to_csv(all_path, index=False)
        ALL[ALL["is_wm"] == 0].to_csv(nowm_path, index=False)
        AP.to_csv(APARC, index=False)
        print(f"ALL: {len(ALL)} rows, {ALL['patient'].nunique()} patients · nowm: {int((ALL['is_wm'] == 0).sum())} · aparc_lookup: {len(AP)} rows")


if __name__ == "__main__":
    main()
