"""Step 5 of member_measures (see 1_measures.py): Bayesian ranking of the 7 types on every measure.

THE METHOD. A Bayesian bootstrap over PATIENTS (Rubin 1981): each of 4000 draws gives the 29
patients Dirichlet(1, ..., 1) weights, recomputes every type's mean under those weights, and ranks
the types (1 = highest). The share of draws in which a type takes each rank is its posterior rank
probability. The patient is the unit, as in the bootstrap intervals of the figures.

TWO WEIGHTINGS of a type's mean, both reported:
    electrodes   each electrode inherits its patient's weight: the mean the violins draw, where a
                 patient with many electrodes in a type counts for more
    patients     each patient's own mean in the type first, then the weighted mean of those: every
                 patient that has the type has one vote
Gate margin is ranked on log(margin): its tail would otherwise decide the rank.

WHAT IT IS NOT. The clustering is held fixed (published labels, no refit): this is the uncertainty
of the type averages over patients, not cluster instability (consensus and the halves test measure
that). Ranks are relative: a type can rank first and sit at chance, so the figures keep their
chance lines. The measures are correlated (silhouette and MMD witness are rank-identical), so ranks
across measures are not independent evidence. No shrinkage (a hierarchical model would pull types
with few patients toward the middle).

Writes out/bayes_ranks.csv: measure, weighting, type, mean_rank, rank_lo/rank_hi (95%), p_best,
p_worst, p_rank1..p_rank7, per feature set (--fset, default all four): reads
out/<fset>/measures.csv and spatial_grouping.csv, writes out/<fset>/bayes_ranks.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

import mm_common as C

N_DRAWS = 4000
MEASURES = ["loading", "r_type_mean", "silhouette", "mmd_witness", "consensus", "halves",
            "spatial_centroid_mm", "gate_margin"]


def rank_one(tag):
    OUT = C.OUT / tag
    t = pd.read_csv(OUT / "measures.csv").merge(
        pd.read_csv(OUT / "spatial_grouping.csv").drop(columns="type"), on="key", how="left")
    t["patient"] = t.key.str.split("|").str[0]
    pats = sorted(t.patient.unique())
    pi = t.patient.map({p: i for i, p in enumerate(pats)}).to_numpy()
    lab = t.type.to_numpy()
    K, NP = int(lab.max()) + 1, len(pats)
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(NP), size=N_DRAWS)                       # draws x patients

    rows = []
    for m in MEASURES:
        v = t[m].to_numpy(float)
        if not np.isfinite(v).any():                   # not computed at this K
            continue
        if m == "gate_margin":
            v = np.log(v)
        if m == "spatial_centroid_mm":                 # closer = better: rank 1 = closest
            v = -v
        ok = np.isfinite(v)
        S = np.zeros((NP, K)); N = np.zeros((NP, K))
        np.add.at(S, (pi[ok], lab[ok]), v[ok])
        np.add.at(N, (pi[ok], lab[ok]), 1.0)
        has = N > 0
        Mp = np.where(has, S / np.maximum(N, 1), 0.0)                   # each patient's mean per type
        means = {"electrodes": (W @ S) / (W @ N),
                 "patients": (W @ Mp) / (W @ has.astype(float))}
        for wname, mu in means.items():
            # rank 1 = the highest mean in that draw
            rk = (-mu).argsort(1).argsort(1) + 1                        # draws x types
            for j in range(K):
                r = rk[:, j]
                row = dict(measure=m, weighting=wname, type=j,
                           n_patients=int(has[:, j].sum()),
                           mean_rank=round(float(r.mean()), 2),
                           rank_lo=int(np.quantile(r, 0.025)), rank_hi=int(np.quantile(r, 0.975)),
                           p_best=round(float((r == 1).mean()), 3),
                           p_worst=round(float((r == K).mean()), 3))
                row.update({f"p_rank{q}": round(float((r == q).mean()), 3) for q in range(1, K + 1)})
                rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "bayes_ranks.csv", index=False)
    pd.set_option("display.width", 200)
    for w in ("electrodes", "patients"):
        print(f"\nposterior mean rank, {w}-weighted (1 = highest)")
        print(res[res.weighting == w].pivot(index="type", columns="measure", values="mean_rank")[[m for m in MEASURES if m in set(res.measure)]]
              .rename(index=lambda j: f"c{j}").to_string())
    d = res.pivot_table(index=["measure", "type"], columns="weighting", values="mean_rank")
    d["gap"] = (d.electrodes - d.patients).abs()
    big = d[d.gap >= 1.0]
    print("\nwhere the two weightings disagree by a rank or more:")
    print(big.round(2).to_string() if len(big) else "  nowhere")
    print("wrote", OUT / "bayes_ranks.csv")


for tag in C.tags_from_args():
    print(f"\n=== {tag}")
    rank_one(tag)
