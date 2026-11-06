"""Step 3 of member_measures (see 1_measures.py), per feature set (--fset, default all four).
Spatial grouping per electrode: of its K nearest electrodes FROM OTHER PATIENTS (fsaverage mm,
hemispheres mirrored onto one side, |x|), the share that carry its own type. Other patients only,
because neighbours on the same shaft share a type through spatial smoothness and the same
recording. Chance = the type's share of the cohort. Also the unmirrored version and the median
distance (mm) to the K nearest same-type electrodes of other patients.
Reads out/<fset>/measures.csv; writes out/<fset>/spatial_grouping.csv."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import mm_common as C

P2 = C.p2()
K_NN = 10
co = pd.read_csv(C.SHARE / "outputs" / "250_recon" / "fsaverage" / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv")
co["key"] = [f"{p}|{P2.norm(n)}" for p, n in zip(co.patient, co.name)]
co = co.drop_duplicates("key").set_index("key")

for tag in C.tags_from_args():
    t = pd.read_csv(C.OUT / tag / "measures.csv")[["key", "type"]]
    xyz = co.reindex(t.key)[["x", "y", "z"]].to_numpy(float)
    if not np.isfinite(xyz).all():
        raise SystemExit(f"{tag}: electrodes without fsaverage coordinates")
    pat = t.key.str.split("|").str[0].to_numpy()
    lab = t.type.to_numpy()
    share = np.bincount(lab) / len(lab)

    def grouping(X):
        _, idx = cKDTree(X).query(X, k=200)          # enough to find K_NN from other patients
        frac, dist = np.empty(len(X)), np.empty(len(X))
        for i in range(len(X)):
            nb = [j for j in idx[i] if pat[j] != pat[i]][:K_NN]
            frac[i] = np.mean(lab[nb] == lab[i])
            same = np.where((lab == lab[i]) & (pat != pat[i]))[0]
            dist[i] = np.median(np.sort(np.linalg.norm(X[same] - X[i], axis=1))[:K_NN])
        return frac, dist

    mir = xyz.copy()
    mir[:, 0] = np.abs(mir[:, 0])
    t["spatial_frac"], t["spatial_dist_mm"] = grouping(mir)
    t["spatial_frac_unmirrored"], _ = grouping(xyz)

    # distance to the type's centroid (Lora, 2026-11-02): hemispheres folded, members weighted by
    # their own-type loading, the centre built from OTHER patients' members only; the reference is
    # the median distance of every non-member to the same centre (a type without a spatial focus
    # would have its members sit there). One centre is assumed: a type with two foci reads as spread.
    w = pd.read_csv(C.OUT / tag / "measures.csv").set_index("key").loading.reindex(t.key).to_numpy(float)
    cd, cref = np.empty(len(t)), np.empty(len(t))
    ref_cache = {}
    for i in range(len(t)):
        m = (lab == lab[i]) & (pat != pat[i])
        c = (w[m, None] * mir[m]).sum(0) / w[m].sum()
        cd[i] = np.linalg.norm(mir[i] - c)
        key = (lab[i], pat[i])
        if key not in ref_cache:
            ref_cache[key] = float(np.median(np.linalg.norm(mir[lab != lab[i]] - c, axis=1)))
        cref[i] = ref_cache[key]
    t["spatial_centroid_mm"], t["spatial_centroid_ref_mm"] = cd, cref
    t.to_csv(C.OUT / tag / "spatial_grouping.csv", index=False)
    g = t.groupby("type")
    print(f"\n=== {tag}")
    print(pd.DataFrame(dict(share=share.round(3), frac_mirrored=g.spatial_frac.mean().round(3),
                            lift=(g.spatial_frac.mean() / share).round(2),
                            centroid_mm=g.spatial_centroid_mm.median().round(1),
                            nonmember_mm=g.spatial_centroid_ref_mm.median().round(1))).to_string())
