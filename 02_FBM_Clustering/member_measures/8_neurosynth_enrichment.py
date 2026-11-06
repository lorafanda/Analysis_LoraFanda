"""Step 8 of member_measures (see 1_measures.py): which cluster sits in which NeuroSynth map, and
which feature set does it best. Run after steps 1 and 4.

ENRICHMENT of cluster c on map m = (share of c's electrodes inside m) / (share of the cohort inside m);
inside = z > 0, the pooling's rule. 1 = as often as any electrode, 2 = twice as often. The 95%
interval bootstraps PATIENTS (1000 draws); a cell is outlined when the whole interval is above 1.
No spatial null: neighbouring contacts fall in the same map, so an outlined cell says "more often
than the cohort", not "beyond spatial smoothness" (FIG 3 runs the shaft-shift null for LanA).

    out/neurosynth_enrichment/<fset>_enrichment.png     clusters x maps, one per feature set
    out/neurosynth_enrichment/compare_feature_sets.png  per map, the best cluster of each feature set
    out/neurosynth_enrichment/enrichment.csv            every cell: share, enrichment, 95% interval

Reading the comparison: a feature set with more clusters (2 bands, K 10) has more chances at a
high maximum and smaller clusters, whose enrichment is noisier; the interval and the cluster size
are printed so that is visible. Reads out/<fset>/measures.csv and out/neurosynth_z.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import mm_common as MC

P2 = MC.p2()
OUTD = MC.FIG_OUT / "neurosynth_enrichment"
OUTD.mkdir(parents=True, exist_ok=True)
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore")
maps_tab = pd.read_csv(MC.NS_MAPS).set_index("column")
COLS = [c for c in ns.columns if c.startswith("ns_")]
TERMS = [maps_tab.loc[c, "term"] if c in maps_tab.index else c[3:] for c in COLS]
rng = np.random.default_rng(0)
NB = 1000

rows = []
for tag in MC.FSETS:
    t = pd.read_csv(MC.OUT / tag / "measures.csv")[["key", "type"]].merge(ns, on="key", how="left")
    pat = t.key.str.split("|").str[0].to_numpy()
    pats = np.unique(pat)
    groups = [np.where(pat == p)[0] for p in pats]
    boots = [np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))]) for _ in range(NB)]
    lab = t.type.to_numpy()
    K = MC.FSETS[tag][1]
    for c, term in zip(COLS, TERMS):
        ins = (t[c].to_numpy(float) > 0)
        base = ins.mean()
        for j in range(K):
            m = lab == j
            share = ins[m].mean()
            bs = []
            for b in boots:
                mb = m[b]
                if mb.sum() and ins[b].mean() > 0:
                    bs.append(ins[b][mb].mean() / ins[b].mean())
            lo, hi = np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan)
            rows.append(dict(fset=tag, cluster=j, n=int(m.sum()), term=term, column=c,
                             inside=int(ins[m].sum()), share=share, cohort_share=base,
                             enrichment=share / base if base else np.nan, lo=lo, hi=hi))
res = pd.DataFrame(rows)
res.round(4).to_csv(OUTD / "enrichment.csv", index=False)
COHORT = res[(res.fset == "15band") & (res.cluster == 0)].set_index("term").cohort_share
XLAB = [f"{t_}  ({100 * COHORT[t_]:.0f}% of cohort)" for t_ in TERMS]
NORM = TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
CMAP = "RdBu_r"


def heat(ax, E, lo, txt, row_labels):
    ax.imshow(np.clip(E, 0, 3), cmap=CMAP, norm=NORM, aspect="auto")
    for r in range(E.shape[0]):
        for c in range(E.shape[1]):
            ax.text(c, r, txt[r][c], ha="center", va="center", fontsize=7.4,
                    color="white" if abs(E[r, c] - 1) > 1.2 else P2.INK)
            if lo[r, c] > 1:
                ax.add_patch(plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, ec=P2.INK, lw=1.6))
    ax.set_xticks(range(len(TERMS)))
    ax.set_xticklabels(XLAB, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)


def colourbar(fig, ax):
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, ticks=[0, 0.5, 1, 2, 3])
    cb.set_label("enrichment (1 = cohort rate)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)


# 1. per feature set: its own clusters x maps
for tag in MC.FSETS:
    r = res[res.fset == tag]
    K = MC.FSETS[tag][1]
    E = r.pivot(index="cluster", columns="term", values="enrichment")[TERMS].to_numpy()
    LO = r.pivot(index="cluster", columns="term", values="lo")[TERMS].to_numpy()
    SH = r.pivot(index="cluster", columns="term", values="share")[TERMS].to_numpy()
    ns_ = r.groupby("cluster").n.first().to_numpy()
    names = [f"c{j}  " + (P2.type_name("concat_bands15", 7, j) if tag == "15band" else "") + f"  (n {ns_[j]})"
             for j in range(K)]
    fig, ax = plt.subplots(figsize=(15, 3.2 + 0.5 * K), dpi=150)
    heat(ax, E, LO, [[f"{E[a, b]:.1f}\n{100 * SH[a, b]:.0f}%" for b in range(len(TERMS))] for a in range(K)], names)
    colourbar(fig, ax)
    ax.set_title(f"{MC.LABEL[tag]}, K = {K}: how often each cluster's electrodes fall inside each NeuroSynth map\n"
                 "cell = enrichment, below it the % of the cluster inside   ·   outlined = 95% patient-bootstrap "
                 "interval above 1   ·   colour saturates at 3",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUTD / f"{tag}_enrichment.png", facecolor="white")
    plt.close(fig)
    print("wrote", f"{tag}_enrichment.png")

# 2. feature sets compared: per map, the most enriched cluster of each feature set
best = res.loc[res.groupby(["fset", "term"]).enrichment.idxmax()]
tags = list(MC.FSETS)
E = np.array([[best[(best.fset == g) & (best.term == m)].enrichment.iloc[0] for m in TERMS] for g in tags])
LO = np.array([[best[(best.fset == g) & (best.term == m)].lo.iloc[0] for m in TERMS] for g in tags])
TX = [[(lambda q: f"{q.enrichment:.1f}\nc{int(q.cluster)} · n {int(q.n)}")(
    best[(best.fset == g) & (best.term == m)].iloc[0]) for m in TERMS] for g in tags]
fig, ax = plt.subplots(figsize=(15, 5.6), dpi=150)
heat(ax, E, LO, TX, [f"{MC.LABEL[g]} (K {MC.FSETS[g][1]})" for g in tags])
colourbar(fig, ax)
ax.set_title("Which feature set puts one of its clusters most squarely in each NeuroSynth map: per map, the most "
             "enriched cluster of each feature set\ncell = its enrichment, the cluster and its size   ·   outlined = "
             "95% patient-bootstrap interval above 1   ·   more clusters (higher K) = more chances and smaller, "
             "noisier clusters", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(OUTD / "compare_feature_sets.png", facecolor="white")
plt.close(fig)
best[["fset", "term", "cluster", "n", "inside", "share", "enrichment", "lo", "hi"]].round(3) \
    .to_csv(OUTD / "best_per_map.csv", index=False)
print("wrote compare_feature_sets.png, enrichment.csv, best_per_map.csv")
