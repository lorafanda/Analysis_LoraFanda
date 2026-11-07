"""Step 17 of member_measures (Lora, 2026-11-07): does a higher K give more functional clusters? For each feature set
at K 6-10 x minimum loading, every cluster x every map against the shaft-shift null, two-sided, and the clusters that
are above chance in at least one / at least two maps, counting clusters that are more than 50% one patient
separately. Replaces the read-only scratchpad checks behind section 4.4 / tab section 5 (23 maps, counting p).

    python 17_maps_by_k.py --ns nimare
    python 17_maps_by_k.py --ns nimare --fsets 15band 5band --ks 7 8 --perm 2000

    out/maps_by_k/maps_by_k.csv        one row per feature set x K x loading
    out/maps_by_k/maps_by_k_best.csv   per feature set, the configuration with the most clean multi-map clusters

THE TEST, per configuration (as step 16 --v4): within each patient the cluster labels are rotated along the shaft
order (--perm shifts, a fresh stream per configuration, so 15 bands K 7 at 0.35 gives step 16's numbers); z = (share
- mean of the shifts) / SD of the shifts, p = 2 x the normal tail, Benjamini-Hochberg over all clusters x maps of the
configuration; above chance = z > 0 and q < 0.05, below chance = z < 0 and q < 0.05.
CLEAN: at most 50% of the cluster's members (at that loading) from one patient.
BEST per feature set: the most clean clusters above chance in >= 2 maps, then in >= 1 map, then the fewest one-patient
clusters, then the lower K."""
import argparse
import re

import numpy as np
import pandas as pd

import mm_common as MC

ap = argparse.ArgumentParser()
ap.add_argument("--fsets", nargs="+", default=["15band", "15zband", "5band", "2band"])
ap.add_argument("--ks", nargs="+", type=int, default=[6, 7, 8, 9, 10])
ap.add_argument("--loadings", nargs="+", type=float, default=[0.0, 0.3, 0.35, 0.4])
ap.add_argument("--perm", type=int, default=2000)
a, _ = ap.parse_known_args()
ONE_PATIENT = 0.5
FSET = {t: f for f, t in MC.TAG.items()}
P2 = MC.p2()
OUTD = MC.SHARED / "maps_by_k"
OUTD.mkdir(parents=True, exist_ok=True)
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore").set_index("key")
order = pd.read_csv(MC.NS_MAPS).column.tolist() if MC.NS_MAPS.exists() else []
MAPS = [c for c in order if c in ns.columns] + [c for c in ns.columns if c.startswith("ns_") and c not in order]


def shaft_key(name):
    r = re.match(r"^(.*?)(\d+)$", str(name))
    return (r.group(1), int(r.group(2))) if r else (str(name), 0)


rows = []
for tag in a.fsets:
    for K in a.ks:
        try:
            d = P2.load_run(FSET[tag], K)
        except SystemExit as e:
            print(f"  skip {tag} K {K}: {e}")
            continue
        keys = [f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"]["patient_id"], d["meta"]["electrode"])]
        Zm = ns.reindex(keys)[MAPS].to_numpy(float)
        has = np.isfinite(Zm).all(1)
        I = (Zm > 0).astype(float)
        lab, pat = d["lab"], d["patient"]
        load = d["Gn"][np.arange(len(lab)), lab]
        sk = [shaft_key(e) for e in d["meta"]["electrode"].to_numpy()]
        for lmin in a.loadings:
            rng = np.random.default_rng(0)                             # fresh per configuration (= steps 13, 15, 16)
            kp = np.where(has & (load >= lmin))[0]
            lk, pk, Ik = lab[kp], pat[kp], I[kp]
            pats = np.unique(pk)
            pidx = pd.Series(pk).map({p: i for i, p in enumerate(pats)}).to_numpy()
            groups = [g for g in (np.array(sorted(np.where(pidx == q)[0], key=lambda i: sk[kp[i]]))
                                  for q in range(len(pats))) if len(g) > 1]
            share = lambda lb: np.array([Ik[lb == j].mean(0) if (lb == j).any() else np.zeros(len(MAPS))
                                         for j in range(K)])
            obs = share(lk)
            s1, s2 = np.zeros_like(obs), np.zeros_like(obs)
            for _ in range(a.perm):
                perm = np.arange(len(kp))
                for g in groups:
                    perm[g] = np.roll(g, rng.integers(1, len(g)))
                sh = share(lk[perm])
                s1 += sh
                s2 += sh ** 2
            empty = np.array([not (lk == j).any() for j in range(K)])
            obs[empty] = np.nan
            T = MC.shaft_z(obs, s1, s2, a.perm)
            n_ = np.array([(lk == j).sum() for j in range(K)])
            top = np.array([pd.Series(pk[lk == j]).value_counts().iloc[0] / n_[j] if n_[j] else 1.0 for j in range(K)])
            ok = top <= ONE_PATIENT
            up, dn = T["above"], T["below"]
            rows.append(dict(fset=tag, K=K, loading=lmin, n=len(kp), clean_clusters=int(ok.sum()),
                             one_patient_clusters=int((~ok).sum()),
                             clean_ge1=int((up[ok].sum(1) >= 1).sum()), clean_ge2=int((up[ok].sum(1) >= 2).sum()),
                             above_clean=int(up[ok].sum()), above_one_patient=int(up[~ok].sum()),
                             below_clean=int(dn[ok].sum()), z_crit=round(T["z_crit"], 2),
                             above=" | ".join(f"c{j}{'' if ok[j] else '(X)'} {int(up[j].sum())}" for j in range(K) if up[j].any()),
                             below=" | ".join(f"c{j}{'' if ok[j] else '(X)'} {int(dn[j].sum())}" for j in range(K) if dn[j].any())))
            r = rows[-1]
            print(f"  {tag} K {K} loading {lmin:g}: clean >=1 map {r['clean_ge1']}, >=2 maps {r['clean_ge2']}, "
                  f"one-patient {r['one_patient_clusters']}; above: {r['above'] or '-'}; below: {r['below'] or '-'}")

res = pd.DataFrame(rows)
res.to_csv(OUTD / "maps_by_k.csv", index=False)
best = (res.sort_values(["fset", "clean_ge2", "clean_ge1", "one_patient_clusters", "K"],
                        ascending=[True, False, False, True, True]).groupby("fset", sort=False).head(1))
best.to_csv(OUTD / "maps_by_k_best.csv", index=False)
pd.set_option("display.width", 220)
print(best[["fset", "K", "loading", "clean_ge1", "clean_ge2", "one_patient_clusters", "above", "below"]].to_string(index=False))
print(f"wrote maps_by_k.csv / maps_by_k_best.csv in {OUTD}")
