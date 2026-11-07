"""Step 14 of member_measures (Lora, 2026-11-05): which feature set x K x loading carries the most functional
information, many-to-many allowed (one cluster in several maps, several clusters in one map). Reads the cNMF runs
directly (every K in --ks) and step 4's map table; does not depend on --k.

    python 14_functional_information.py --ns nimare
    python 14_functional_information.py --ns nimare --fset 15band --perm 1000

    out/functional_information/functional_information[_fdr].csv   one row per feature set x K x loading
    out/functional_information/functional_information[_fdr].png   z per K x loading, one panel per feature set

THE SCORE, one per configuration: the mutual information between cluster membership and inside/outside each
map (nimare: the map's top 10%; fdr: z > 0), summed over all maps. Every pattern counts, nothing has to be
one-to-one. Configurations:
    loading >= L   hard clusters, only electrodes whose own-type loading is >= L (L = 0: all electrodes)
    weighted       every electrode in every cluster, weighted by its normalised loading (convex NMF's G, rows
                   summing to 1): no threshold
THE NULL (--null shift, the default): within each patient the electrodes are put in shaft order (shaft name, then
contact number) and the cluster memberships are rotated by a random offset against the map values (--perm times).
This keeps every patient's coverage and cluster mix AND the run of neighbouring contacts along a shaft, which share
both a cluster and a map; a plain shuffle (--null shuffle) breaks those runs and counts spatial smoothness as
information, so its z is inflated. Either way the score only counts information beyond "which patient, implanted
where". z = (observed - null mean) / null sd; excess = observed - null mean, in bits, comparable across K and
loading cuts (same electrodes as the null) but NOT between hard and weighted (soft memberships carry fewer bits):
compare those on z.
THE RULE: a configuration is valid when no cluster has more than 50% of its electrodes from one patient; the
weighted version must ALSO pass it with its hard clusters on all electrodes (its own loading-mass share is milder,
since every electrode spreads over all clusters, and alone it passed partitions whose clusters are one patient); invalid cells are greyed and never chosen.
z GROWS WITH THE NUMBER OF ELECTRODES for the same effect: a loading cut that keeps fewer electrodes is handicapped
on z, and a finer K can only add information (refining a partition never loses mutual information), which the null
corrects only for chance. Read z together with excess_bits and n.
THE WINNER: the valid configuration with the highest z, and a patient bootstrap (--boot) of z for the top five
(the null mean and sd held fixed, an approximation): if the intervals overlap, the configurations are
equivalent on function and stability and agreement decide.
COMPANIONS, for reading only: breadth = maps whose own information beats the null at one-sided 2.5% (about
0.025 x the number of maps expected by chance); clusters per informative map and maps per cluster = enrichment >= 1.5x."""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mm_common as MC

ap = argparse.ArgumentParser()
ap.add_argument("--ks", nargs="+", type=int, default=[6, 7, 8, 9, 10])
ap.add_argument("--loadings", nargs="+", type=float, default=[0.0, 0.3, 0.35, 0.4, 0.45, 0.5])
ap.add_argument("--perm", type=int, default=500, help="within-patient shuffles per configuration")
ap.add_argument("--boot", type=int, default=500, help="patient bootstraps for the top five")
ap.add_argument("--null", default="shift", choices=["shift", "shuffle"],
                help="shift: rotate memberships along the shaft order within patient; shuffle: permute within patient")
MC.add_style_arg(ap)
a, _ = ap.parse_known_args()
ONE_PATIENT = 0.5
ENRICH = 1.5

P2 = MC.p2()
OUTD = MC.SHARED / "functional_information"
OUTD.mkdir(parents=True, exist_ok=True)
SUFFIX = ("" if MC.ARGS.ns == "nimare" else f"_{MC.ARGS.ns}") + ("" if a.null == "shift" else f"_{a.null}")
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore").set_index("key")
MAPS = [c for c in ns.columns if c.startswith("ns_")]
rng = np.random.default_rng(0)
FSETS = {f: t for f, t in MC.TAG.items() if MC.ARGS.fset in ("all", t)}


def mi(A, w, I):
    """Mutual information (bits) between membership A (n x K, rows = weights over clusters) and each map's
    inside/outside I (n x M), electrodes weighted by w; returns one value per map."""
    Aw = A * w[:, None]
    tot = Aw.sum()
    p1 = (Aw.T @ I) / tot                                # K x M, P(cluster, inside)
    p0 = (Aw.T @ (1 - I)) / tot                          # K x M, P(cluster, outside)
    pk = Aw.sum(0)[:, None] / tot                        # K x 1
    pi = p1.sum(0, keepdims=True)                        # 1 x M
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(p1 > 0, p1 * np.log2(p1 / (pk * pi)), 0)
        t0 = np.where(p0 > 0, p0 * np.log2(p0 / (pk * (1 - pi))), 0)
    return (t1 + t0).sum(0)


def null_perm(n, groups):
    """A permutation of the electrodes inside each patient: groups are the patients' electrodes in shaft order;
    shift rotates each group by a random offset (neighbours stay neighbours), shuffle permutes it."""
    perm = np.arange(n)
    for g in groups:
        if len(g) < 2:
            continue
        perm[g] = np.roll(g, rng.integers(1, len(g))) if a.null == "shift" else rng.permutation(g)
    return perm


def shaft_key(name):
    """(shaft, contact number) from a contact name: the trailing digits are the contact number."""
    m = re.match(r"^(.*?)(\d+)$", str(name))
    return (m.group(1), int(m.group(2))) if m else (str(name), 0)


rows, boots_in = [], {}
for fset, tag in FSETS.items():
    for K in a.ks:
        try:
            d = P2.load_run(fset, K)
        except SystemExit as e:
            print(f"  skip {tag} K {K}: {e}")
            continue
        keys = [f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"]["patient_id"], d["meta"]["electrode"])]
        Z = ns.reindex(keys)[MAPS].to_numpy(float)
        has = np.isfinite(Z).all(1)
        Gn, lab, pat = d["Gn"][has], d["lab"][has], d["patient"][has]
        sk = [shaft_key(e) for e in d["meta"]["electrode"].to_numpy()[has]]
        I = (Z[has] > 0).astype(float)
        load = Gn[np.arange(len(lab)), lab]
        coh = I.mean(0)
        pats = np.unique(pat)
        pidx = pd.Series(pat).map({p: i for i, p in enumerate(pats)}).to_numpy()
        onehot = np.eye(K)[lab]
        hp = np.zeros((len(pats), K))                                  # the hard clusters, all electrodes
        np.add.at(hp, pidx, onehot)
        hard_ok = bool((hp.max(0) / np.maximum(hp.sum(0), 1)).max() <= ONE_PATIENT)
        configs = [(f"{l:g}", onehot, (load >= l).astype(float)) for l in a.loadings] + [("weighted", Gn, np.ones(len(lab)))]
        for name, A, w in configs:
            keep = w > 0
            Ak, wk, Ik, pk_ = A[keep], w[keep], I[keep], pidx[keep]
            mass = Ak * wk[:, None]                                       # one-patient share per cluster
            per_pat = np.zeros((len(pats), K))
            np.add.at(per_pat, pk_, mass)
            tot_c = per_pat.sum(0)
            share = np.where(tot_c > 0, per_pat.max(0) / np.maximum(tot_c, 1e-12), 1.0)
            obs_m = mi(Ak, wk, Ik)
            skk = [sk[i] for i in np.where(keep)[0]]
            groups = [np.array(sorted(np.where(pk_ == q)[0], key=lambda i: skk[i]))     # each patient, shaft order
                      for q in range(len(pats)) if (pk_ == q).any()]
            null_m = np.array([mi(Ak[null_perm(len(pk_), groups)], wk, Ik) for _ in range(a.perm)])
            obs, null = obs_m.sum(), null_m.sum(1)
            z = (obs - null.mean()) / null.std(ddof=1)
            breadth = int((obs_m > np.quantile(null_m, 0.975, axis=0)).sum())
            enr = ((Ak * wk[:, None]).T @ Ik) / np.maximum((Ak * wk[:, None]).sum(0)[:, None], 1e-12) / coh   # K x M
            inf_maps = obs_m > np.quantile(null_m, 0.975, axis=0)
            hit = enr >= ENRICH
            rows.append(dict(fset=tag, k=K, loading=name, n=int(keep.sum()), max_one_patient=round(float(share.max()), 2),
                             valid=bool(share.max() <= ONE_PATIENT) and (name != "weighted" or hard_ok),
                             mi_bits=round(obs, 4),
                             null_mean=round(null.mean(), 4), null_sd=round(null.std(ddof=1), 4),
                             excess_bits=round(obs - null.mean(), 4), z=round(z, 2),
                             p=round(float((null >= obs).mean()), 4), breadth=breadth,
                             clusters_per_informative_map=round(float(hit[:, inf_maps].sum(0).mean()), 2) if inf_maps.any() else 0.0,
                             maps_per_cluster=round(float(hit[:, inf_maps].sum(1)[hit[:, inf_maps].any(1)].mean()), 2)
                             if hit[:, inf_maps].any() else 0.0))
            boots_in[(tag, K, name)] = (Ak, wk, Ik, pk_, len(pats), null.mean(), null.std(ddof=1))
        print(f"  {tag} K {K}: " + "  ".join(f"{r['loading']} z {r['z']:.1f}{'' if r['valid'] else ' x'}"
                                            for r in rows[-len(configs):]))

res = pd.DataFrame(rows)

# the winner and its rivals: a patient bootstrap of z (the null mean and sd held fixed), on the z scale because
# the excess in bits is not comparable between hard and weighted memberships
val = res[res.valid].sort_values("z", ascending=False)
res["z_boot_lo"] = np.nan
res["z_boot_hi"] = np.nan
for i, r in val.head(5).iterrows():
    Ak, wk, Ik, pk_, npat, nm, nsd = boots_in[(r.fset, r.k, r.loading)]
    ex = []
    for _ in range(a.boot):
        cnt = np.bincount(rng.integers(0, npat, npat), minlength=npat).astype(float)
        wb = wk * cnt[pk_]
        if wb.sum() > 0:
            ex.append((mi(Ak, wb, Ik).sum() - nm) / nsd)
    res.loc[i, ["z_boot_lo", "z_boot_hi"]] = np.quantile(ex, [0.025, 0.975])
res.to_csv(OUTD / f"functional_information{SUFFIX}.csv", index=False)

# the figure: z per loading (rows) x K (columns), one panel per feature set; invalid cells grey
tags = list(dict.fromkeys(res.fset))
lrows = [f"{l:g}" for l in a.loadings] + ["weighted"]
vmax = max(3.0, np.nanpercentile(res.z, 98))
fig, axes = plt.subplots(1, len(tags), figsize=(2.3 * len(tags) + 1.2, 3.4), dpi=200, sharey=True)
axes = np.atleast_1d(axes)
best = val.iloc[0] if len(val) else None
for ax, tag in zip(axes, tags):
    sub = res[res.fset == tag]
    ks = sorted(sub.k.unique())
    M = np.full((len(lrows), len(ks)), np.nan)
    V = np.zeros_like(M, bool)
    for _, r in sub.iterrows():
        M[lrows.index(r.loading), ks.index(r.k)] = r.z
        V[lrows.index(r.loading), ks.index(r.k)] = r.valid
    ax.imshow(np.where(V, M, np.nan), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.imshow(np.where(~V & np.isfinite(M), 1.0, np.nan), cmap="Greys", vmin=0, vmax=4, aspect="auto")
    for yi in range(len(lrows)):
        for xi in range(len(ks)):
            if np.isfinite(M[yi, xi]):
                ax.text(xi, yi, f"{M[yi, xi]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if V[yi, xi] and M[yi, xi] < 0.6 * vmax else ("#3a3f45" if not V[yi, xi] else "black"))
    if best is not None and best.fset == tag:
        ax.add_patch(plt.Rectangle((ks.index(best.k) - 0.5, lrows.index(best.loading) - 0.5), 1, 1,
                                   fill=False, ec="#d7191c", lw=2))
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"K {k}" for k in ks], fontsize=7)
    ax.set_yticks(range(len(lrows)))
    ax.set_yticklabels([("loading ≥ " + l) if l != "weighted" else l for l in lrows], fontsize=7)
    ax.set_title(MC.LABEL[tag], fontsize=9)
    for s_ in ax.spines.values():
        s_.set_visible(False)
sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="viridis")
cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("z against the within-patient " + ("shaft shift" if a.null == "shift" else "shuffle"), fontsize=7.5)
cb.ax.tick_params(labelsize=7)
fig.suptitle(f"Functional information of the partition, null: {a.null} within patient (grey: a cluster > 50% one patient; red box: best valid)",
             fontsize=9.5, x=0.01, ha="left", y=1.06)          # above the panel titles at every size
MC.save_styles(fig, OUTD / f"functional_information{SUFFIX}.png", a.styles, transparent=True, bbox_inches="tight")
plt.close(fig)
print(f"wrote functional_information{SUFFIX}.csv / .png in {OUTD}")
cols = ["fset", "k", "loading", "n", "max_one_patient", "z", "excess_bits", "z_boot_lo", "z_boot_hi", "breadth",
        "clusters_per_informative_map", "maps_per_cluster"]
pd.set_option("display.width", 220)
print("top five valid configurations:")
print(res.loc[val.head(5).index, cols].to_string(index=False))
