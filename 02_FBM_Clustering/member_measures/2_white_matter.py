"""Step 2 of member_measures (see 1_measures.py). % white matter around every cohort contact,
from each patient's own FreeSurfer segmentation, in the neighbourhood of the proximal tissue
density of Mercier et al. 2017 (NeuroImage 147: 3 x 3 x 3 voxels around the contact): % WM =
n_WM / (n_WM + n_GM), voxels that are neither (CSF, ventricles, unknown, brainstem, ventral DC)
left out. A 3-mm sphere is computed beside it as a check. Coordinates: the native tkrRAS tables
of 250_recon, mapped to voxels with the segmentation's own vox2ras-tkr. Validated against the
anatomical WM flag of the fsaverage contact table. Reads only; writes out/white_matter.csv."""
import os
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHARE = HERE.parent                                   # 02_FBM_Clustering
RECON = SHARE / "outputs" / "250_recon"
OUT = HERE / "out" / "white_matter.csv"
os.environ["PAPER2_PAPER"] = "0"
P2 = types.ModuleType("p2_wm")
P2.__file__ = str(SHARE / "00_Paper2_Figures.py")
sys.modules["p2_wm"] = P2
exec(compile(Path(P2.__file__).read_text(encoding="utf-8"), P2.__file__, "exec"), P2.__dict__)

WM = np.array([2, 41, 7, 46, 77, 78, 79, 251, 252, 253, 254, 255, 5001, 5002])
GM_SUB = np.array([3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58])


def classes(lab):
    wm = np.isin(lab, WM)
    gm = np.isin(lab, GM_SUB) | ((lab >= 1000) & (lab < 3000))
    return wm, gm


sys.path.insert(0, str(HERE))
import mm_common as C                                 # noqa: E402
cohort = C.cohort_keys()
pats = sorted(cohort.key.str.split("|").str[0].unique())
allc = pd.read_csv(RECON / "fsaverage" / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv")
allc["key"] = [f"{p}|{P2.norm(n)}" for p, n in zip(allc.patient, allc.name)]

sph = np.array([(i, j, k) for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)
                if i * i + j * j + k * k <= 9])
cube = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
rows = []
for p in pats:
    c = pd.read_csv(RECON / p / "glassbrain" / "coords" / f"{p}_contacts_tkrRAS.csv")
    mri = Path(str(c.mgz_used.dropna().iloc[0])).parent
    seg_f = mri / "aparc+aseg.mgz" if (mri / "aparc+aseg.mgz").exists() else mri / "aseg.mgz"
    img = nib.load(str(seg_f))
    lab = np.asarray(img.dataobj).astype(np.int32)
    ras2vox = np.linalg.inv(img.header.get_vox2ras_tkr())
    # the recon name -> the recording name, through the contact table's own pairing
    al = allc[allc.patient == p].drop_duplicates("name_raw").set_index("name_raw")["name"]
    for r in c.itertuples():
        name = al.get(r.name_raw, r.name)
        v = np.rint(ras2vox @ np.array([r.x, r.y, r.z, 1.0]))[:3].astype(int)
        out = dict(key=f"{p}|{P2.norm(name)}", patient=p, seg=seg_f.name)
        for tag, nb in (("cube", cube), ("sph3", sph)):
            ix = v[None, :] + nb
            ok = ((ix >= 0) & (ix < np.array(lab.shape))).all(1)
            lv = lab[ix[ok, 0], ix[ok, 1], ix[ok, 2]]
            wm, gm = classes(lv)
            n = int(wm.sum() + gm.sum())
            out[f"pct_wm_{tag}"] = 100.0 * wm.sum() / n if n else np.nan
            out[f"n_tissue_{tag}"] = n
        out["centre_label"] = int(lab[tuple(np.clip(v, 0, np.array(lab.shape) - 1))])
        rows.append(out)
    print(f"  {p:<9} {len(c):>4} contacts   {seg_f.name}")

w = pd.DataFrame(rows).drop_duplicates("key")
m = cohort.merge(w, on="key", how="left")
print(f"\n  cohort electrodes {len(cohort)}   with a value {m.pct_wm_cube.notna().sum()}   "
      f"no match {m.patient.isna().sum()}   no tissue in the cube {(m.patient.notna() & m.pct_wm_cube.isna()).sum()}")
v = m.merge(allc[["key", "is_wm"]].drop_duplicates("key"), on="key", how="left")
for f in (0, 1):
    s = v[v.is_wm == f].pct_wm_cube
    print(f"  anatomical label {'WM' if f else 'not WM'}: n {len(s):>4}   median % WM {s.median():5.1f}   "
          f"share above 50% {100 * (s > 50).mean():5.1f}%")
print(f"  cube vs 3-mm sphere: r = {m[['pct_wm_cube', 'pct_wm_sph3']].corr().iloc[0, 1]:.3f}")
print("  % WM (cube) quantiles 10/25/50/75/90: "
      + " / ".join(f"{m.pct_wm_cube.quantile(q):.0f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
m.to_csv(OUT, index=False)
print("wrote", OUT)
