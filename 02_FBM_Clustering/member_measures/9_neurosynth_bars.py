"""Step 9 of member_measures (see 1_measures.py): what percentage of each cluster falls inside each
NeuroSynth map, as bars, in the sectioned layout of the violin figures. Run after steps 1 and 4
(and step 7 for the per-cluster figures, which read its out/matching.csv). Takes --k like the rest.

    out[_k7]/neurosynth_bars/<fset>_bars.png   one per feature set: a panel per map, a bar per cluster
                                               (the % of its electrodes inside the map)
    out[_k7]/neurosynth_bars/c<j>_bars.png     one per 15-band type: a panel per map, a bar per feature
                                               set (the % of the matched cluster inside the map)
    out[_k7]/neurosynth_bars/percent_inside.csv  every bar

The maps are grouped as the task runs: perception, comprehension and reading, retrieval,
production. Inside = z > 0 (the pooling's rule). Error bars: 95% from bootstrapping PATIENTS.
Dashed line: the cohort's own % inside that map, so a bar above it is a cluster that sits in the
map more often than electrodes in general. A bar is outlined when its whole interval is above the
line. No spatial null (neighbouring contacts fall in the same map). Maps not in a section below
are drawn in a last "other maps" section."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import mm_common as MC

P2 = MC.p2()
OUTD = MC.FIG_OUT / "neurosynth_bars"
OUTD.mkdir(parents=True, exist_ok=True)
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore")
maps_tab = pd.read_csv(MC.NS_MAPS).set_index("column")
TERM = {c: (maps_tab.loc[c, "term"] if c in maps_tab.index else c[3:]) for c in ns.columns if c.startswith("ns_")}
SECTIONS = [("Perception", ["auditory", "speech_perception", "visual"]),
            ("Comprehension and reading", ["comprehension", "language_comprehension", "reading", "semantic",
                                           "lexical", "phonological"]),
            ("Retrieval", ["retrieval", "recall"]),
            ("Production", ["speech_production", "broca", "motor control", "premotor"])]
if "stage" in maps_tab.columns and maps_tab.stage.fillna("").astype(str).str.len().gt(0).all():
    SECTIONS = [(st, [TERM[c] for c in maps_tab.index[maps_tab.stage == st] if c in TERM])
                for st in dict.fromkeys(maps_tab.stage)]
placed = {t for _, ts in SECTIONS for t in ts}
other = [t for t in TERM.values() if t not in placed]
SECTIONS = [(s, [t for t in ts if t in TERM.values()]) for s, ts in SECTIONS] + ([("Other maps", other)] if other else [])
SECTIONS = [(s, ts) for s, ts in SECTIONS if ts]
COL_OF = {v: k for k, v in TERM.items()}
TAGS = [g for g in MC.FSETS if (MC.OUT / g / "measures.csv").exists()]
rng = np.random.default_rng(0)
NB = 1000
INK_SEC = "#3a3f45"

T = {}
for g in TAGS:
    t = pd.read_csv(MC.OUT / g / "measures.csv")[["key", "type"]].merge(ns, on="key", how="left")
    t["patient"] = t.key.str.split("|").str[0]
    T[g] = t
pats = sorted(T[TAGS[0]].patient.unique())
P_IDX = {g: T[g].patient.map({p: i for i, p in enumerate(pats)}).to_numpy() for g in TAGS}
BOOT = [rng.integers(0, len(pats), len(pats)) for _ in range(NB)]


def pct_inside(g, idx, col):
    """% of electrodes idx inside map col, with a patient-bootstrap 95% interval."""
    ins = (T[g][col].to_numpy(float)[idx] > 0).astype(float)
    pi = P_IDX[g][idx]
    S = np.bincount(pi, weights=ins, minlength=len(pats))
    N = np.bincount(pi, minlength=len(pats)).astype(float)
    bs = []
    for b in BOOT:
        nb = N[b].sum()
        if nb:
            bs.append(S[b].sum() / nb)
    lo, hi = np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan)
    return 100 * ins.mean(), 100 * lo, 100 * hi


def layout(n_panels_per_section):
    """A grid of up to 6 panels per row, one block of rows per section."""
    rows = []
    for s, ts in n_panels_per_section:
        for i in range(0, len(ts), 6):
            rows.append((s if i == 0 else None, ts[i:i + 6]))
    return rows


def draw(rows_spec, bars_for, xlabels, title, sub, out, colours):
    rows = layout(rows_spec)
    H = 2.9 * len(rows) + 2.2
    fig = plt.figure(figsize=(22, H), dpi=140)
    gs = GridSpec(len(rows), 6, figure=fig, hspace=0.95, wspace=0.28, left=0.04, right=0.99,
                  top=1 - 1.85 / H, bottom=0.9 / H)
    fig.canvas.draw()
    first_axes = []
    for r_, (sec, terms) in enumerate(rows):
        for c_, term in enumerate(terms):
            ax = fig.add_subplot(gs[r_, c_])
            col = COL_OF[term]
            vals = bars_for(col)                       # [(pct, lo, hi, cohort) or None, ...]
            cohort = next(v[3] for v in vals if v is not None)
            for x, v in enumerate(vals):
                if v is None:
                    ax.text(x, 1, "no\nmatch", ha="center", va="bottom", fontsize=7, color=P2.MUTED)
                    continue
                pct, lo, hi, _ = v
                sure = lo > cohort
                ax.bar(x, pct, width=0.72, color=colours[x], alpha=0.85,
                       edgecolor=P2.INK if sure else "none", lw=1.6 if sure else 0)
                ax.errorbar(x, pct, yerr=[[pct - lo], [hi - pct]], fmt="none", ecolor=P2.INK, lw=1, capsize=2)
            ax.axhline(cohort, color=P2.MUTED, ls=(0, (4, 3)), lw=1)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, fontsize=7.6, rotation=0 if len(xlabels) <= 7 else 30)
            ax.tick_params(labelsize=8, colors=P2.MUTED)
            ax.spines[["top", "right"]].set_visible(False)
            n_in = int(maps_tab.loc[col, "electrodes_inside"]) if col in maps_tab.index else None
            ax.set_title(f"{term}", fontsize=10.5, loc="left", pad=14)
            ax.annotate(f"cohort {cohort:.0f}% inside" + (f" ({n_in} electrodes)" if n_in is not None else ""),
                        xy=(0, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                        fontsize=7.2, color=P2.MUTED, va="bottom")
            if c_ == 0:
                ax.set_ylabel("% inside the map", fontsize=8.5)
            if sec is not None and c_ == 0:
                first_axes.append((sec, ax, len(terms)))
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for sec, ax, nt in first_axes:
        top = inv.transform((0, ax.get_tightbbox(rr).y1))[1]
        x0 = ax.get_position().x0
        x1 = fig.axes[fig.axes.index(ax) + nt - 1].get_position().x1
        fig.add_artist(Line2D([x0, x1], [top + 0.006] * 2, color=INK_SEC, lw=1.2, transform=fig.transFigure))
        fig.text(x0, top + 0.009, sec, fontsize=12, color=INK_SEC, fontweight="bold", va="bottom")
    fig.suptitle(title, x=0.04, y=1 - 0.25 / H, ha="left", fontsize=15)
    fig.text(0.04, 1 - 0.62 / H, sub, fontsize=9.5, color=P2.MUTED, va="top")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out.name)


rows_csv = []
SUB = ("bar = % of the cluster's electrodes inside the map (z > 0)   ·   error bar = 95% from bootstrapping patients"
       "   ·   dashed = the cohort's % inside   ·   outlined = the whole interval above the cohort line")

# 1. per feature set: a bar per cluster
for g in TAGS:
    K = MC.FSETS[g][1]
    lab = T[g].type.to_numpy()
    allidx = np.arange(len(lab))

    def bars(col, g=g, lab=lab, K=K):
        coh = 100 * (T[g][col].to_numpy(float) > 0).mean()
        out = []
        for j in range(K):
            pct, lo, hi = pct_inside(g, np.where(lab == j)[0], col)
            out.append((pct, lo, hi, coh))
            rows_csv.append(dict(view="feature_set", fset=g, cluster=j, map=TERM[col], n=int((lab == j).sum()),
                                 pct_inside=round(pct, 2), lo=round(lo, 2), hi=round(hi, 2), cohort_pct=round(coh, 2)))
        return out
    cols = [np.asarray(P2.cluster_col(j, K))[:3] for j in range(K)]
    draw(SECTIONS, bars, [f"c{j}" for j in range(K)],
         f"{MC.LABEL[g]}, K = {K} ({MC.K_NOTE}): what share of each cluster sits in each NeuroSynth map",
         SUB, OUTD / f"{g}_bars.png", cols)

# 2. per 15-band type: a bar per feature set (the matched cluster)
mfile = MC.OUT / "matching.csv"
if "15band" in TAGS and mfile.exists():
    match = pd.read_csv(mfile)
    K_REF = MC.FSETS["15band"][1]
    for j in range(K_REF):
        def bars(col, j=j):
            out = []
            for g in TAGS:
                if g == "15band":
                    cl = j
                else:
                    m = match[(match.fset == g) & (match.type_15band == j)]
                    if m.empty:
                        out.append(None)
                        continue
                    cl = int(m.cluster.iloc[0])
                lab = T[g].type.to_numpy()
                coh = 100 * (T[g][col].to_numpy(float) > 0).mean()
                pct, lo, hi = pct_inside(g, np.where(lab == cl)[0], col)
                out.append((pct, lo, hi, coh))
                rows_csv.append(dict(view="per_cluster", fset=g, cluster=cl, type_15band=j, map=TERM[col],
                                     n=int((lab == cl).sum()), pct_inside=round(pct, 2), lo=round(lo, 2),
                                     hi=round(hi, 2), cohort_pct=round(coh, 2)))
            return out
        name = P2.type_name("concat_bands15", K_REF, j)
        col = np.asarray(P2.cluster_col(j, K_REF))[:3]
        draw(SECTIONS, bars, [MC.LABEL[g].replace("15 bands ", "15 ") for g in TAGS],
             f"c{j}" + (f" · {name}" if name else "") + f": what share sits in each NeuroSynth map, per feature set ({MC.K_NOTE})",
             SUB + "\neach feature set shows its cluster matched to this 15-band type (step 7, FIG 4's rule; "
             "the match quality is in matching.csv)", OUTD / f"c{j}_bars.png", [col] * len(TAGS))
else:
    print("  NOTE: run step 7 first for the per-cluster bars (they need out/matching.csv)")
pd.DataFrame(rows_csv).to_csv(OUTD / "percent_inside.csv", index=False)
print("wrote percent_inside.csv")
