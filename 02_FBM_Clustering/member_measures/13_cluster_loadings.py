"""Step 13 of member_measures (Lora, 2026-11-05): per cluster, ALL maps, at several minimum loadings. Run after
steps 1 and 4; takes --k and --ns like the rest (made for --ns nimare).

    python 13_cluster_loadings.py --k 7 --ns nimare
    python 13_cluster_loadings.py --k 7 --ns nimare --loadings 0.2 0.3 0.35 0.4 --fset 15zband
    python 13_cluster_loadings.py --k 7 --ns nimare --fset 15band --loadings 0.35 --outline shaft
        outlines from the shaft-shift null, corrected over every bar (files _load035_shaft)

    out_k7/nimare/cluster_loadings/<fset>_cluster_loadings.png   one panel per cluster, transparent background
    out_k7/nimare/cluster_loadings/cluster_loadings.csv          every bar

One panel per cluster (the violins' order and colour); x = every map in the z table, in maps.csv order; per map
one bar per minimum loading (the pattern tells the loading, legend), only the cluster's electrodes with
loading >= that value:
    bar        % of those electrodes inside the map (nimare: the map's top 10%)
    black tick % of ALL electrodes inside (the cohort, the same at every loading)
    error bar  95% from bootstrapping patients; outline: the whole interval above the tick
The n per loading is under the cluster id. Red dot above a bar: at that loading more than 50% of the cluster's
remaining electrodes come from one patient (that bar describes one patient's coverage).
OUTLINE --outline boot (default): the bootstrap interval above the tick, per bar and NOT corrected for the many
bars (clusters x maps x loadings): at 2.5% one-sided, about 1 bar in 40 is outlined by chance; no spatial null.
--outline shaft: the cluster's share beats the shaft-shift null (memberships rotated along the shaft order within
each patient, step 14's null; --perm rotations) at q < 0.05, Benjamini-Hochberg over EVERY bar of the figure."""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import mm_common as MC

ap = argparse.ArgumentParser()
ap.add_argument("--loadings", nargs="+", type=float, default=[0.2, 0.3, 0.35, 0.4])
ap.add_argument("--outline", default="boot", choices=["boot", "shaft"])
ap.add_argument("--perm", type=int, default=2000, help="shaft-shift rotations (--outline shaft)")
a, _ = ap.parse_known_args()
SUFFIX = ("" if a.loadings == [0.2, 0.3, 0.35, 0.4] else "_load" + "_".join(f"{int(round(100 * l)):03d}" for l in a.loadings))
SUFFIX += "_shaft" if a.outline == "shaft" else ""
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "oo", "----"]
plt.rcParams["hatch.linewidth"] = 0.8
LEGEND_FACE = "#5f6670"
ONE_PATIENT = 0.5
RED = "#d7191c"
if len(a.loadings) > len(HATCH):
    raise SystemExit(f"at most {len(HATCH)} loadings")

P2 = MC.p2()
OUTD = MC.FIG_OUT / "cluster_loadings"
OUTD.mkdir(parents=True, exist_ok=True)
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore")
order = pd.read_csv(MC.NS_MAPS).column.tolist() if MC.NS_MAPS.exists() else []
MAPS = [c for c in order if c in ns.columns] + [c for c in ns.columns if c.startswith("ns_") and c not in order]
NAMES = [m[3:].replace("_vs_", " vs ").replace("_", " ") for m in MAPS]
rng = np.random.default_rng(0)
NB = 1000
rows = []

for g in [t for t in MC.tags_from_args() if (MC.OUT / t / "measures.csv").exists()]:
    K = MC.FSETS[g][1]
    t = pd.read_csv(MC.OUT / g / "measures.csv")[["key", "type", "P_lana", "loading"]].merge(ns, on="key", how="left")
    lab, P, L = t.type.to_numpy(), t.P_lana.to_numpy(float), t.loading.to_numpy(float)
    rank = sorted(range(K), key=lambda j: -P[lab == j].mean())        # the violins' order (6_story_figures)
    pat = t.key.str.split("|").str[0].to_numpy()
    pats = np.unique(pat)
    pidx = pd.Series(pat).map({p: i for i, p in enumerate(pats)}).to_numpy()
    boots = [rng.integers(0, len(pats), len(pats)) for _ in range(NB)]
    INS = {m: (t[m].to_numpy(float) > 0).astype(float) for m in MAPS}
    Q = {}
    if a.outline == "shaft":                                           # the shaft-shift null, BH over every bar
        sk = [(lambda r: (r.group(1), int(r.group(2))) if r else (n, 0))(re.match(r"^(.*?)(\d+)$", n))
              for n in t.key.str.split("|").str[1]]
        IM = np.column_stack([INS[m] for m in MAPS])
        cells, pv = [], []
        for li, lmin in enumerate(a.loadings):
            kp = np.where(L >= lmin)[0]
            lk, pk = lab[kp], pidx[kp]
            groups = [kp_ for kp_ in (np.array(sorted(np.where(pk == q)[0], key=lambda i: sk[kp[i]]))
                                      for q in range(len(pats))) if len(kp_) > 1]
            IMk = IM[kp]
            share_fn = lambda lb: np.array([IMk[lb == j].mean(0) if (lb == j).any() else np.zeros(len(MAPS))
                                         for j in range(K)])
            obs = share_fn(lk)
            ge = np.zeros_like(obs)
            for _ in range(a.perm):
                perm = np.arange(len(kp))
                for gr in groups:
                    perm[gr] = np.roll(gr, rng.integers(1, len(gr)))
                ge += share_fn(lk[perm]) >= obs
            for j in range(K):
                for mi in range(len(MAPS)):
                    if (lk == j).any():
                        cells.append((li, j, mi))
                        pv.append((ge[j, mi] + 1) / (a.perm + 1))
        pv = np.array(pv)
        o = np.argsort(pv)
        qv = np.minimum.accumulate((pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
        qq = np.empty_like(qv)
        qq[o] = np.minimum(qv, 1)
        Q = dict(zip(cells, qq))

    nm, nl = len(MAPS), len(a.loadings)
    w = 0.84 / nl
    fig, axes = plt.subplots(K, 1, figsize=(0.24 * nm * nl + 1.6, 1.9 * K + 1.2), dpi=200, sharex=True, sharey=True)
    fig.patch.set_alpha(0)
    for ax, j in zip(axes, rank):
        ax.patch.set_alpha(0)
        col = np.asarray(P2.cluster_col(j, K))[:3]
        ns_ = []
        for li, lmin in enumerate(a.loadings):
            sel = (lab == j) & (L >= lmin)
            ns_.append(int(sel.sum()))
            vc = pd.Series(pat[sel]).value_counts()
            share = vc.iloc[0] / sel.sum() if sel.any() else 0.0     # the largest one-patient share at this loading
            one = share > ONE_PATIENT
            N = np.bincount(pidx[sel], minlength=len(pats)).astype(float)
            for mi, m in enumerate(MAPS):
                ins = INS[m]
                coh = 100 * ins.mean()
                if not sel.any():
                    continue
                S = np.bincount(pidx[sel], weights=ins[sel], minlength=len(pats))
                bs = [S[b].sum() / N[b].sum() for b in boots if N[b].sum()]
                lo, hi = 100 * np.quantile(bs, [0.025, 0.975])
                pct = 100 * ins[sel].mean()
                sure = (lo > coh) if a.outline == "boot" else bool(Q.get((li, j, mi), 1.0) < 0.05)
                x = mi - 0.42 + w * (li + 0.5)
                ax.bar(x, pct, width=w * 0.92, color=col, hatch=HATCH[li], edgecolor="white", lw=0, zorder=2)
                if sure:
                    ax.bar(x, pct, width=w * 0.92, fill=False, edgecolor=P2.INK, lw=1.2, zorder=3)
                ax.plot([x, x], [lo, hi], color=P2.INK, lw=0.5, zorder=3)
                if one:                                             # red dot: > 50% one patient at this loading
                    ax.plot(x, hi + 3, "o", color=RED, ms=3.2, zorder=5, clip_on=False)
                rows.append(dict(fset=g, k=K, cluster=j, min_loading=lmin, n=int(sel.sum()), map=m[3:],
                                 pct_inside=round(pct, 1), lo=round(lo, 1), hi=round(hi, 1),
                                 cohort_pct=round(coh, 1), outlined=sure, top_patient=vc.index[0],
                                 top_patient_share=round(share, 2), one_patient=bool(one),
                                 q_shaft=round(float(Q[(li, j, mi)]), 4) if (li, j, mi) in Q else np.nan))
        for mi, m in enumerate(MAPS):                                   # the cohort tick, once per map
            ax.plot([mi - 0.44, mi + 0.44], [100 * INS[m].mean()] * 2, color="black", lw=1.1, zorder=4)
        ax.set_ylabel(f"c{j}\n" + " / ".join(str(n_) for n_ in ns_), rotation=0, ha="right", va="center",
                      fontsize=9, color=col, fontweight="bold", labelpad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlim(-0.6, nm - 0.4)
    axes[-1].set_xticks(range(nm))
    axes[-1].set_xticklabels(NAMES, rotation=40, ha="right", fontsize=7.5)
    fig.supylabel("% of the cluster's electrodes inside the map", fontsize=9, x=0.004)
    axes[0].legend(handles=[Patch(facecolor=LEGEND_FACE, hatch=HATCH[i] * 2, edgecolor="white", label=f"loading ≥ {l:g}")
                            for i, l in enumerate(a.loadings)]
                   + [Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50% one patient")], handlelength=2.4, handleheight=1.4, frameon=False,
                   fontsize=7.5, ncol=nl + 1, loc="upper right")
    axes[0].set_title(f"{MC.LABEL[g]}, K = {K}", fontsize=10.5, loc="left")
    fig.tight_layout(h_pad=0.5, rect=(0.015, 0, 1, 1))
    fig.savefig(OUTD / f"{g}_cluster_loadings{SUFFIX}.png", transparent=True)
    plt.close(fig)
    print("wrote", f"{g}_cluster_loadings{SUFFIX}.png")

out = pd.DataFrame(rows)
out.to_csv(OUTD / f"cluster_loadings{SUFFIX}.csv", index=False)
if len(out):
    print(f"wrote cluster_loadings{SUFFIX}.csv: {out.outlined.sum()} of {len(out)} bars outlined"
          + (f" (about {round(0.025 * len(out))} expected by chance)" if a.outline == "boot" else " (q < 0.05, shaft-shift null)"))
