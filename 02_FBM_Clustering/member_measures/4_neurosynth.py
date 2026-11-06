"""Step 4 of member_measures (see 1_measures.py). Each electrode's z on EVERY NeuroSynth term map in
04_FBM_Pooling/neurosynth/<term>/ (the association-test file of each folder; any other file is
ignored), sampled exactly as 04_FBM_Pooling/functions/lf_pool.assign_neurosynth_regions does: the
fsaverage (MNI305) xyz of the contact table -> MNI152 with lf_pool's fixed _MNI305_TO_152 -> the
nearest voxel. Inside a map = z > 0 (the pooling's rule).

Per map, out/neurosynth_maps.csv: n_studies (typed by hand into neurosynth_term_studies.csv beside
this script: the files carry no study count; NeuroSynth shows it on each term's page), the
significant volume (voxels with z > 0, cm3, and % of the MNI152 brain), the peak z, the median z of
the significant voxels, and in OUR cohort the electrodes and patients inside. Reads only.

--ns nimare (after 0_nimare_maps.py): the unthresholded maps of 04_FBM_Pooling/neurosynth_nimare/.
Each electrode takes the mean z within a 5-mm sphere (localisation error, map smoothness), then its
percentile among the map's in-mask voxels; the table stores percentile - 90, so "inside" (> 0)
means the top 10% of the brain on that map, for every map whatever its study count. Study counts
come from 0_nimare_maps.py (all >= 200). Written to out/nimare_z.csv and out/nimare_maps.csv."""
import os
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHARE = HERE.parent                                   # 02_FBM_Clustering
NS = SHARE.parent / "04_FBM_Pooling" / "neurosynth"
OUT = HERE / "out"
STUDIES = HERE / "neurosynth_term_studies.csv"
os.environ["PAPER2_PAPER"] = "0"
P2 = types.ModuleType("mm_p2_ns")
P2.__file__ = str(SHARE / "00_Paper2_Figures.py")
sys.modules["mm_p2_ns"] = P2
exec(compile(Path(P2.__file__).read_text(encoding="utf-8"), P2.__file__, "exec"), P2.__dict__)

MNI305_TO_152 = np.array([                            # lf_pool._MNI305_TO_152, as it stands
    [0.9975, -0.0073, 0.0176, -0.0429],
    [0.0146, 1.0009, -0.0024, 1.5496],
    [-0.0130, -0.0093, 0.9971, 1.1840],
    [0.0, 0.0, 0.0, 1.0]])

sys.path.insert(0, str(HERE))
import mm_common as C                                 # noqa: E402
NIMARE = C.ARGS.ns == "nimare"
if NIMARE:
    NS = SHARE.parent / "04_FBM_Pooling" / "neurosynth_nimare"
    PATTERN, SPHERE_MM = "*association-test_z.nii.gz", 5.0
    meta = pd.read_csv(NS / "maps.csv").set_index("name") if (NS / "maps.csv").exists() else pd.DataFrame()
else:
    PATTERN, SPHERE_MM = "*association-test_z_FDR*.nii.gz", 0.0

maps = {}
for sub in sorted(p for p in NS.iterdir() if p.is_dir() and not p.name.startswith("_")):
    zfs = sorted(f for f in sub.glob(PATTERN) if NIMARE or "FDR" in f.name)
    if zfs:
        img = nib.load(str(zfs[0]))
        maps[sub.name] = (np.asarray(img.dataobj, dtype=float), img.affine, zfs[0].name)
    else:
        print(f"  {sub.name}: no association-test file, skipped")
anat_f = next(iter(sorted((SHARE.parent / "04_FBM_Pooling" / "neurosynth").glob("*/anatomical.nii.gz"))), None)
brain_vox = None
if anat_f is not None:
    a = np.asarray(nib.load(str(anat_f)).dataobj, float)
    brain_vox = int((a > 0.2 * np.percentile(a[a > 0], 99)).sum())

cohort = C.cohort_keys()
co = pd.read_csv(SHARE / "outputs" / "250_recon" / "fsaverage" / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv")
co["key"] = [f"{p}|{P2.norm(n)}" for p, n in zip(co.patient, co.name)]
xyz = co.drop_duplicates("key").set_index("key").reindex(cohort.key)[["x", "y", "z"]].to_numpy(float)
ok = np.isfinite(xyz).all(1)
mni = (np.c_[xyz, np.ones(len(xyz))] @ MNI305_TO_152.T)[:, :3]
pat = cohort.key.str.split("|").str[0].to_numpy()

studies = {}
if NIMARE:
    studies = {n: r.n_studies for n, r in meta.iterrows()} if len(meta) else {}
elif STUDIES.exists():
    s_ = pd.read_csv(STUDIES)
    studies = {str(r.term): r.n_studies for r in s_.itertuples() if pd.notna(r.n_studies)}
else:
    pd.DataFrame(dict(term=list(maps), n_studies="")).to_csv(STUDIES, index=False)
    print(f"  wrote {STUDIES.name}: fill in n_studies per term from neurosynth.org and rerun")


def sample(vol, aff, point):
    """Nearest voxel (sphere 0) or the mean over the voxels within SPHERE_MM of the point."""
    inv = np.linalg.inv(aff)
    c = (inv @ np.r_[point, 1.0])[:3]
    if SPHERE_MM <= 0:
        ijk = np.round(c).astype(int)
        if (ijk >= 0).all() and (ijk < np.array(vol.shape[:3])).all():
            return vol[ijk[0], ijk[1], ijk[2]]
        return np.nan
    vs = np.abs(np.diag(aff)[:3])
    r = np.ceil(SPHERE_MM / vs).astype(int)
    lo = np.maximum(np.floor(c).astype(int) - r, 0)
    hi = np.minimum(np.floor(c).astype(int) + r + 1, np.array(vol.shape[:3]))
    if (hi <= lo).any():
        return np.nan
    gi, gj, gk = np.meshgrid(*[np.arange(lo[d], hi[d]) for d in range(3)], indexing="ij")
    d2 = sum(((g - c[d]) * vs[d]) ** 2 for d, g in enumerate((gi, gj, gk)))
    sel = d2 <= SPHERE_MM ** 2
    vals = vol[gi[sel], gj[sel], gk[sel]]
    vals = vals[vals != 0] if NIMARE else vals                 # outside the analysis mask
    return float(vals.mean()) if len(vals) else np.nan


out, rows = cohort.copy(), []
for name, (vol, aff, fname) in maps.items():
    zs = np.array([sample(vol, aff, mni[i]) if ok[i] else np.nan for i in range(len(out))])
    col = f"ns_{name.replace(' ', '_')}"
    vox_cm3 = abs(np.linalg.det(aff[:3, :3])) / 1000.0
    if NIMARE:
        inmask = np.sort(vol[vol != 0])
        pct = 100.0 * np.searchsorted(inmask, zs, side="right") / len(inmask)
        pct[~np.isfinite(zs)] = np.nan
        out[col] = pct - 90.0                                 # > 0 = the map's top 10%
        sig = vol >= np.quantile(inmask, 0.90)
        inside = out[col].to_numpy() > 0
        zin = zs[inside]
    else:
        out[col] = zs
        sig = vol > 0
        inside = zs > 0
        zin = zs[inside]
    rows.append(dict(term=name, column=col, file=fname, n_studies=studies.get(name, np.nan),
                     stage=(meta.loc[name, "stage"] if NIMARE and name in meta.index else ""),
                     kind=(meta.loc[name, "kind"] if NIMARE and name in meta.index else "term"),
                     sig_volume_cm3=round(float(sig.sum() * vox_cm3), 1),
                     sig_pct_brain=round(100.0 * sig.sum() / brain_vox, 1) if brain_vox else np.nan,
                     peak_z=round(float(vol.max()), 2),
                     median_sig_z=round(float(np.median(vol[sig])), 2) if sig.any() else np.nan,
                     electrodes_inside=int(inside.sum()),
                     pct_electrodes_inside=round(100.0 * inside.mean(), 1),
                     patients_inside=int(len(set(pat[inside]))),
                     median_z_inside=round(float(np.median(zin)), 2) if inside.any() else np.nan))
out.to_csv(C.NS_Z, index=False)
mt = pd.DataFrame(rows)
mt.to_csv(C.NS_MAPS, index=False)
pd.set_option("display.width", 220)
print(mt.drop(columns=["column", "file"]).to_string(index=False))
print("wrote", C.NS_Z, "and", C.NS_MAPS)
