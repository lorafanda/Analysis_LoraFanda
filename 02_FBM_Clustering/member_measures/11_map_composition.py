"""Step 11 of member_measures (Lora, 2026-11-05): the companion to the violin figures that cluster size
and violin shape cannot bias: per cluster, the % of its electrodes inside each selected map, the maps
side by side. Run after steps 1 and 4; takes --k and --ns like the rest (made for --ns nimare).

    python 11_map_composition.py --k 7 --ns nimare
    python 11_map_composition.py --k 7 --ns nimare --set rest     each task stage vs resting state (_rest files)
    python 11_map_composition.py --k 7 --ns nimare --maps sentences_vs_words listening_vs_auditory
    python 11_map_composition.py --k 7 --ns nimare --fset 15band --min-loading 0.35 --outline shaft
        only electrodes loading >= 0.35 on their type (file name _load035); outlines from the shaft-shift null
Every figure of a feature set has the contrasts figure's y range, whatever the maps (bars above it are cut
and listed in the console).

    out_k7/nimare/composition/<fset>_composition.png   transparent background
    out_k7/nimare/composition/composition.csv          every bar

Per cluster (x, FIG 3 A's order and the violins' c-ids, ids in the cluster's colour), one bar per map:
    bar        in the cluster's colour, the map told by its pattern (legend);
               % of the cluster's electrodes inside the map (nimare: the top 10% of the map; fdr: z > 0).
               An electrode inside two maps counts in both: the bars of a cluster do not add up to 100.
    error bar  95% from bootstrapping patients (step 9's rule)
    black tick the cohort's % inside that map (the same for every cluster)
    outline    --outline boot (default): the whole interval above the cohort tick (step 10's rule), per bar,
               uncorrected, no spatial null.
               --outline shaft: the cluster's share beats the shaft-shift null (memberships rotated along the
               shaft order within each patient, neighbours kept together, step 14's null; --perm rotations) at
               q < 0.05 after Benjamini-Hochberg over all the bars of the figure (clusters x maps).
With --min-loading the bars, error bars and the y range use only the electrodes loading >= that value on their
type; the black tick stays the cohort of ALL electrodes.
No text beyond the title, the axes and the map names."""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import mm_common as MC

DEFAULT_MAPS = ["naming_vs_objects", "listening_vs_auditory", "sentences_vs_words", "naming_vs_production",
                "semantic_vs_memory_retrieval", "production_vs_motor",
                "language_vs_resting_state"]                                  # the seven NiMARE contrasts
# every bar in its cluster's colour (as in the violins); the map is told by its pattern, never by a colour
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "oo", "----"]
plt.rcParams["hatch.linewidth"] = 0.9
LEGEND_FACE = "#5f6670"                                 # the legend's swatches: a neutral, not a cluster colour
SETS = {"contrasts": DEFAULT_MAPS,
        # each task stage vs resting-state studies (0_nimare_maps.py), language vs resting state in the same last
        # place as in the contrasts figure
        "rest": ["naming_vs_resting_state", "listening_vs_resting_state", "sentences_vs_resting_state",
                 "pictures_vs_resting_state", "semantic_vs_resting_state", "production_vs_resting_state",
                 "language_vs_resting_state"]}
ap = argparse.ArgumentParser()
ap.add_argument("--set", default="contrasts", choices=list(SETS), help="which maps; the file name gets _<set> unless contrasts")
ap.add_argument("--maps", nargs="+", default=None, help="map names as in the z table, without ns_ (overrides --set)")
ap.add_argument("--min-loading", type=float, default=0.0, help="keep electrodes with loading >= this on their type")
ap.add_argument("--outline", default="boot", choices=["boot", "shaft"])
ap.add_argument("--perm", type=int, default=2000, help="shaft-shift rotations (--outline shaft)")
a, _ = ap.parse_known_args()
a.maps = a.maps or SETS[a.set]
SUFFIX = ("" if a.set == "contrasts" and a.maps == DEFAULT_MAPS else f"_{a.set}")
SUFFIX += (f"_load{int(round(100 * a.min_loading)):03d}" if a.min_loading > 0 else "") + ("_shaft" if a.outline == "shaft" else "")

P2 = MC.p2()
OUTD = MC.FIG_OUT / "composition"
OUTD.mkdir(parents=True, exist_ok=True)
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore")
missing = [m for m in a.maps if f"ns_{m}" not in ns.columns]
if missing:
    raise SystemExit(f"--maps not in {MC.NS_Z.name}: {missing}")
if len(a.maps) > len(HATCH):
    raise SystemExit(f"at most {len(HATCH)} maps on one plot")
NAMES = [m.replace("_vs_", " vs ").replace("_", " ") for m in a.maps]
rng = np.random.default_rng(0)
NB = 1000
rows = []

for g in [t for t in MC.tags_from_args() if (MC.OUT / t / "measures.csv").exists()]:
    K = MC.FSETS[g][1]
    tall = pd.read_csv(MC.OUT / g / "measures.csv")[["key", "type", "P_lana", "loading"]].merge(ns, on="key", how="left")
    Pall = tall.P_lana.to_numpy(float)
    rank = sorted(range(K), key=lambda j: -Pall[tall.type.to_numpy() == j].mean())   # the violins' order, all electrodes
    COH = {m: 100 * (tall[f"ns_{m}"].to_numpy(float) > 0).mean() for m in set(a.maps) | set(DEFAULT_MAPS)
           if f"ns_{m}" in tall.columns}                                # the cohort: ALL electrodes
    t = tall[tall.loading.to_numpy(float) >= a.min_loading].reset_index(drop=True)
    lab = t.type.to_numpy()
    pat = t.key.str.split("|").str[0].to_numpy()
    pats = np.unique(pat)
    pidx = pd.Series(pat).map({p: i for i, p in enumerate(pats)}).to_numpy()
    boots = [rng.integers(0, len(pats), len(pats)) for _ in range(NB)]

    def stat(m, j):
        """% of cluster j inside map m, its patient-bootstrap 95% interval, and the cohort's %."""
        ins = (t[f"ns_{m}"].to_numpy(float) > 0).astype(float)
        sel = lab == j
        S = np.bincount(pidx[sel], weights=ins[sel], minlength=len(pats))
        N = np.bincount(pidx[sel], minlength=len(pats)).astype(float)
        bs = [S[b].sum() / N[b].sum() for b in boots if N[b].sum()]
        lo, hi = 100 * np.quantile(bs, [0.025, 0.975])
        return 100 * ins[sel].mean(), lo, hi, COH[m], int(sel.sum())

    Q = None
    if a.outline == "shaft":                                           # the shaft-shift null, BH over all bars
        sk = [(lambda r: (r.group(1), int(r.group(2))) if r else (n, 0))(re.match(r"^(.*?)(\d+)$", n))
              for n in t.key.str.split("|").str[1]]
        groups = [np.array(sorted(np.where(pidx == q)[0], key=lambda i: sk[i])) for q in range(len(pats))
                  if (pidx == q).any()]
        INS = np.column_stack([(t[f"ns_{m}"].to_numpy(float) > 0) for m in a.maps]).astype(float)
        share = lambda lb: np.array([INS[lb == j].mean(0) if (lb == j).any() else np.zeros(len(a.maps))
                                     for j in range(K)])
        obs = share(lab)
        ge = np.zeros_like(obs)
        for _ in range(a.perm):
            perm = np.arange(len(lab))
            for gr in groups:
                if len(gr) > 1:
                    perm[gr] = np.roll(gr, rng.integers(1, len(gr)))
            ge += share(lab[perm]) >= obs
        pv = ((ge + 1) / (a.perm + 1)).ravel()
        o = np.argsort(pv)
        qv = np.minimum.accumulate((pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
        Q = np.empty_like(qv)
        Q[o] = np.minimum(qv, 1)
        Q = Q.reshape(obs.shape)

    # the y range is always the contrasts figure's (its highest error bar + matplotlib's 5% margin), so the
    # figures of one feature set can be laid side by side
    ytop = 1.05 * max(stat(m, j)[2] for m in DEFAULT_MAPS if f"ns_{m}" in ns.columns for j in range(K))
    nm = len(a.maps)
    w = 0.8 / nm
    fig, ax = plt.subplots(figsize=(1.2 + 0.32 * nm * K, 3.6), dpi=200)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    clipped = []
    for mi, (m, name) in enumerate(zip(a.maps, NAMES)):
        for xi, j in enumerate(rank):
            pct, lo, hi, coh, n_ = stat(m, j)
            if hi > ytop:
                clipped.append(f"c{j} {name} ({hi:.0f}%)")
            x = xi - 0.4 + w * (mi + 0.5)
            sure = (lo > coh) if Q is None else bool(Q[j, mi] < 0.05)
            ax.bar(x, pct, width=w * 0.9, color=np.asarray(P2.cluster_col(j, K))[:3], hatch=HATCH[mi],
                   edgecolor="white", lw=0, zorder=2)
            if sure:
                ax.bar(x, pct, width=w * 0.9, fill=False, edgecolor=P2.INK, lw=1.6, zorder=3)
            ax.plot([x, x], [lo, hi], color=P2.INK, lw=0.7, zorder=3)
            ax.plot([x - w * 0.42, x + w * 0.42], [coh, coh], color="black", lw=1.1, zorder=4)
            rows.append(dict(fset=g, k=K, cluster=j, n=n_, map=m, pct_inside=round(pct, 1),
                             lo=round(lo, 1), hi=round(hi, 1), cohort_pct=round(coh, 1), outlined=sure,
                             q_shaft=round(float(Q[j, mi]), 4) if Q is not None else np.nan,
                             min_loading=a.min_loading))
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"c{j}" for j in rank], fontsize=9, fontweight="bold")
    for lbl, j in zip(ax.get_xticklabels(), rank):
        lbl.set_color(np.asarray(P2.cluster_col(j, K))[:3])
    ax.set_xlim(-0.55, K - 0.45)
    ax.set_ylim(0, ytop)
    ax.set_ylabel("% of the cluster's electrodes inside the map", fontsize=8.5)
    ax.set_xlabel("cluster", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor=LEGEND_FACE, hatch=HATCH[i] * 2, edgecolor="white", label=n_)
                       for i, n_ in enumerate(NAMES)], handlelength=2.6, handleheight=1.6, frameon=False, fontsize=7.5, ncol=min(nm, 3), loc="upper right")
    ax.set_title(f"{MC.LABEL[g]}, K = {K}" + (f", loading ≥ {a.min_loading:g}" if a.min_loading > 0 else ""),
                 fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUTD / f"{g}_composition{SUFFIX}.png", transparent=True)
    plt.close(fig)
    print("wrote", f"{g}_composition{SUFFIX}.png")
    if clipped:
        print(f"   above the contrasts figure's y range ({ytop:.0f}%), cut at the top: {', '.join(clipped)}")

pd.DataFrame(rows).to_csv(OUTD / f"composition{SUFFIX}.csv", index=False)
print(f"wrote composition{SUFFIX}.csv")
