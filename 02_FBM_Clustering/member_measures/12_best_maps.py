"""Step 12 of member_measures (Lora, 2026-11-05): per cluster, the functional maps it belongs to most. Run after
steps 1 and 4; takes --k and --ns like the rest (made for --ns nimare: every map 0_nimare_maps.py fitted).

    python 12_best_maps.py --k 7 --ns nimare
    python 12_best_maps.py --k 7 --ns nimare --min-loading 0.3      only electrodes loading >= 0.3 on their type

    out_k7/nimare/best_maps/<fset>_best_maps[_load030].png   transparent background
    out_k7/nimare/best_maps/best_maps[_load030].csv          every cluster x every map

One row per cluster (the violins' order and colour), its TOP maps among ALL maps in the z table, ranked by
enrichment (% of the cluster inside / % of all electrodes inside):
    bar        enrichment = % of the cluster inside the map (nimare: the map's top 10%) / % of ALL electrodes
               inside (the cohort, also with --min-loading); log axis, bars grow from the black line at 1 = the cohort
    error bar  95% from bootstrapping patients; outline: the whole interval above 1
A cluster whose best maps are not outlined belongs to none of them more than electrodes in general do.
No spatial null."""
import argparse

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mm_common as MC

ap = argparse.ArgumentParser()
ap.add_argument("--min-loading", type=float, default=0.0, help="keep electrodes with loading >= this on their type")
ap.add_argument("--top", type=int, default=3, help="maps shown per cluster")
a, _ = ap.parse_known_args()
SUFFIX = f"_load{int(round(100 * a.min_loading)):03d}" if a.min_loading > 0 else ""

P2 = MC.p2()
OUTD = MC.FIG_OUT / "best_maps"
OUTD.mkdir(parents=True, exist_ok=True)
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore")
MAPS = [c for c in ns.columns if c.startswith("ns_")]
rng = np.random.default_rng(0)
NB = 1000
rows = []

for g in [t for t in MC.tags_from_args() if (MC.OUT / t / "measures.csv").exists()]:
    K = MC.FSETS[g][1]
    t = pd.read_csv(MC.OUT / g / "measures.csv")[["key", "type", "P_lana", "loading"]].merge(ns, on="key", how="left")
    lab, P = t.type.to_numpy(), t.P_lana.to_numpy(float)
    rank = sorted(range(K), key=lambda j: -P[lab == j].mean())        # the violins' order (6_story_figures)
    keep = t.loading.to_numpy(float) >= a.min_loading
    pat = t.key.str.split("|").str[0].to_numpy()
    pats = np.unique(pat)
    pidx = pd.Series(pat).map({p: i for i, p in enumerate(pats)}).to_numpy()
    boots = [rng.integers(0, len(pats), len(pats)) for _ in range(NB)]

    best = {}
    for j in range(K):
        sel = (lab == j) & keep
        cells = []
        for m in MAPS:
            ins = (t[m].to_numpy(float) > 0).astype(float)
            coh = 100 * ins.mean()
            S = np.bincount(pidx[sel], weights=ins[sel], minlength=len(pats))
            N = np.bincount(pidx[sel], minlength=len(pats)).astype(float)
            bs = [S[b].sum() / N[b].sum() for b in boots if N[b].sum()]
            lo, hi = 100 * np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan)
            pct = 100 * ins[sel].mean() if sel.any() else np.nan
            cells.append(dict(fset=g, k=K, cluster=j, n=int(sel.sum()), map=m[3:], pct_inside=round(pct, 1),
                              lo=round(lo, 1), hi=round(hi, 1), cohort_pct=round(coh, 1),
                              enrichment=round(pct / coh, 2) if coh else np.nan, outlined=bool(lo > coh),
                              min_loading=a.min_loading))
        rows += cells
        best[j] = sorted(cells, key=lambda c: -np.nan_to_num(c["enrichment"]))[:a.top]

    nt = a.top
    fig, axes = plt.subplots(K, 1, figsize=(7.2, 0.42 * nt * K + 0.9), dpi=200, sharex=True)
    fig.patch.set_alpha(0)
    xmax = max(c["hi"] / c["cohort_pct"] for j in best for c in best[j] if np.isfinite(c["hi"]))
    for ax, j in zip(axes, rank):
        ax.patch.set_alpha(0)
        col = np.asarray(P2.cluster_col(j, K))[:3]
        for yi, c in enumerate(best[j]):
            y = nt - 1 - yi
            e = c["pct_inside"] / c["cohort_pct"]
            ax.barh(y, e - 1, left=1, height=0.7, color=col, edgecolor=P2.INK if c["outlined"] else "none",
                    lw=1.4 if c["outlined"] else 0)
            ax.plot([c["lo"] / c["cohort_pct"], c["hi"] / c["cohort_pct"]], [y, y], color=P2.INK, lw=0.7)
        ax.set_yticks(range(nt))
        ax.set_yticklabels([c["map"].replace("_vs_", " vs ").replace("_", " ") for c in best[j]][::-1], fontsize=7.5)
        ax.set_ylabel(f"c{j}", rotation=0, ha="right", va="center", fontsize=10, fontweight="bold", color=col, labelpad=14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.axvline(1, color="black", lw=1.2, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.25, max(6, 1.1 * xmax))
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["¼×", "½×", "1×", "2×", "4×"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[-1].set_xlabel("inside the map, against all electrodes", fontsize=8.5)
    ttl = f"{MC.LABEL[g]}, K = {K}" + (f", loading ≥ {a.min_loading:g}" if a.min_loading > 0 else "")
    axes[0].set_title(ttl, fontsize=10.5, loc="left")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(OUTD / f"{g}_best_maps{SUFFIX}.png", transparent=True)
    plt.close(fig)
    print("wrote", f"{g}_best_maps{SUFFIX}.png")
    for j in rank:
        print(f"   c{j}: " + "; ".join(f"{c['map']} {c['enrichment']:.1f}x{'*' if c['outlined'] else ''}" for c in best[j]))

pd.DataFrame(rows).to_csv(OUTD / f"best_maps{SUFFIX}.csv", index=False)
print(f"wrote best_maps{SUFFIX}.csv")
