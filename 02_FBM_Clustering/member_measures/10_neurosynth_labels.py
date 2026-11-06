"""Step 10 of member_measures (see 1_measures.py): a NeuroSynth label for every cluster. Run after
steps 1 and 4; takes --k like the rest.

ONE BUBBLE MATRIX PER FEATURE SET: rows = its clusters; first column the LanA network core (P(LanA)
>= 0.2, added 2026-11-03), then eight NeuroSynth maps, one per stage of the
task and no near-duplicates (auditory overlaps speech perception, comprehension overlaps language
comprehension, phonological overlaps both comprehension maps; lexical, recall, broca and motor control
are small):
    speech perception, visual | reading | language comprehension, semantic | retrieval |
    speech production, premotor
    colour   enrichment = % of the cluster inside the map / % of the cohort inside, log scale
    size     % of the cluster inside the map (z > 0)
    outline  the 95% patient-bootstrap interval of that % is entirely above the cohort's %
LABEL (right of each row): the outlined columns, strongest enrichment first; "no match"
when none is outlined.
X MARK: a cluster with more than 50% of its electrodes from one patient keeps its row, greyed,
with a cross and that patient's share in its name: it does not meet the "drawn from the cohort"
expectation, and its map profile is that patient's electrode coverage.
No spatial null: neighbouring contacts fall in the same map.

    out[_k7]/neurosynth_labels/<fset>_labels.png
    out[_k7]/neurosynth_labels/labels.csv   every cell and every label
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

import mm_common as MC

P2 = MC.p2()
OUTD = MC.FIG_OUT / "neurosynth_labels"
OUTD.mkdir(parents=True, exist_ok=True)
# first column (Lora, 2026-11-03): the LanA network core, P(LanA) >= 0.2 at the contact, the paper's
# own language reference, judged by the same rules as the maps
LANA_CORE = 0.2
MAPS = [("LanA", "lana_core", "LanA core"),
        ("Perception", "speech_perception", "speech perception"), ("Perception", "visual", "visual"),
        ("Reading", "reading", "reading"),
        ("Comprehension", "language_comprehension", "language comprehension"),
        ("Comprehension", "semantic", "semantic"),
        ("Retrieval", "retrieval", "retrieval"),
        ("Production", "speech_production", "speech production"), ("Production", "premotor", "premotor")]
ONE_PATIENT = 0.50
maps_tab = pd.read_csv(MC.NS_MAPS).set_index("column") if MC.NS_MAPS.exists() else pd.DataFrame()
if MC.ARGS.ns == "nimare":                       # the NiMARE maps, in their stage order, LanA first
    MAPS = [MAPS[0]] + [(r.stage, c[3:], c[3:].replace("_vs_", " vs ").replace("_", " "))
                        for c, r in maps_tab.iterrows()]
N_STUDIES = {c[3:]: maps_tab.loc[c, "n_studies"] for c in maps_tab.index} if len(maps_tab) else {}
FEW = {c for c, n_ in N_STUDIES.items() if pd.notna(n_) and str(n_).strip() != "" and float(n_) < MC.MIN_STUDIES}
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore")
missing = [f"ns_{c}" for _, c, _ in MAPS if c != "lana_core" and f"ns_{c}" not in ns.columns]
if missing:
    raise SystemExit(f"maps not in neurosynth_z.csv (rerun step 4?): {missing}")
TAGS = [g for g in MC.FSETS if (MC.OUT / g / "measures.csv").exists()]
rng = np.random.default_rng(0)
NB = 1000
NORM = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)          # log2 enrichment, 1/4x .. 4x
CMAP = plt.get_cmap("RdBu_r")
GREY_ROW = "#9aa0a6"
rows_csv = []

for g in TAGS:
    K = MC.FSETS[g][1]
    t = pd.read_csv(MC.OUT / g / "measures.csv")[["key", "type", "P_lana"]].merge(ns, on="key", how="left")
    t["ns_lana_core"] = np.where(t.P_lana.to_numpy(float) >= LANA_CORE, 1.0, 0.0)
    pat = t.key.str.split("|").str[0].to_numpy()
    pats = np.unique(pat)
    pidx = pd.Series(pat).map({p: i for i, p in enumerate(pats)}).to_numpy()
    boots = [rng.integers(0, len(pats), len(pats)) for _ in range(NB)]
    lab = t.type.to_numpy()

    info = []
    for j in range(K):
        m = lab == j
        vc = pd.Series(pat[m]).value_counts()
        top_share = vc.iloc[0] / m.sum() if m.sum() else 0
        cells = []
        for _, col, name in MAPS:
            ins = (t[f"ns_{col}"].to_numpy(float) > 0).astype(float)
            coh = ins.mean()
            S = np.bincount(pidx[m], weights=ins[m], minlength=len(pats))
            N = np.bincount(pidx[m], minlength=len(pats)).astype(float)
            bs = [S[b].sum() / N[b].sum() for b in boots if N[b].sum()]
            pct = ins[m].mean()
            lo = np.quantile(bs, 0.025) if bs else np.nan
            enr = pct / coh if coh else np.nan
            cells.append(dict(map=name, pct=pct, lo=lo, cohort=coh, enr=enr,
                              sure=bool(lo > coh) and col not in FEW))   # < 200 studies cannot label
            rows_csv.append(dict(fset=g, k=K, cluster=j, n=int(m.sum()), map=name, pct_inside=round(100 * pct, 1),
                                 lo=round(100 * lo, 1), cohort_pct=round(100 * coh, 1), enrichment=round(enr, 2),
                                 outlined=bool(lo > coh), top_patient=vc.index[0], top_patient_share=round(top_share, 2)))
        sure = sorted([c for c in cells if c["sure"]], key=lambda c: -c["enr"])
        names_ = [c["map"] for c in sure[:4]]
        label = ("\n".join(" · ".join(names_[i:i + 2]) for i in range(0, len(names_), 2))
                 if sure else "no match")
        info.append(dict(j=j, n=int(m.sum()), npat=int(vc.size), top=vc.index[0], share=top_share,
                         cells=cells, label=label))

    fig_h = 2.3 + 0.72 * K
    fig, ax = plt.subplots(figsize=(17, fig_h), dpi=150)
    for r_, inf in enumerate(info):
        y = K - 1 - r_
        bad = inf["share"] > ONE_PATIENT
        for c_, cell in enumerate(inf["cells"]):
            le = np.log2(max(cell["enr"], 1e-3)) if np.isfinite(cell["enr"]) else 0
            face = GREY_ROW if bad else CMAP(NORM(np.clip(le, -2, 2)))
            size = 30 + 1400 * min(cell["pct"], 0.6) / 0.6
            ax.scatter(c_, y, s=size, c=[face], alpha=0.55 if bad else 1.0,
                       edgecolors=P2.INK if cell["sure"] and not bad else "none",
                       linewidths=1.8 if cell["sure"] else 0, zorder=3)
            ax.text(c_, y, f"{100 * cell['pct']:.0f}", ha="center", va="center", fontsize=7.4, zorder=4,
                    color="white" if (not bad and abs(le) > 1.1 and cell["pct"] > 0.08) else P2.INK)
        name = f"c{inf['j']}   n {inf['n']}, {inf['npat']} patient{'' if inf['npat'] == 1 else 's'}"
        if bad:
            name = f"✗  {name}   ({100 * inf['share']:.0f}% {inf['top']})"
        ax.text(-0.75, y, name, ha="right", va="center", fontsize=9.5,
                color="#b2182b" if bad else P2.INK, fontweight="bold" if bad else "normal")
        ax.text(len(MAPS) - 0.35, y, "one patient: map profile = that patient's coverage" if bad else inf["label"],
                ha="left", va="center", fontsize=9.2, color=GREY_ROW if bad else P2.INK,
                style="italic" if bad or inf["label"].startswith("no ") else "normal")
    # stage headers over the columns
    stages = [s for s, _, _ in MAPS]
    for s in dict.fromkeys(stages):
        cs = [i for i, x in enumerate(stages) if x == s]
        ax.text(np.mean(cs), K - 0.25, s, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#3a3f45")
        ax.plot([min(cs) - 0.4, max(cs) + 0.4], [K - 0.32] * 2, color="#3a3f45", lw=1)
    coh_txt = {cell["map"]: cell["cohort"] for cell in info[0]["cells"]}
    ax.set_xticks(range(len(MAPS)))
    ax.set_xticklabels([f"{name.replace(' ', chr(10))}\n(cohort {100 * coh_txt[name]:.0f}%)"
                        + (f"\n< {MC.MIN_STUDIES} studies: no label" if col in FEW else "") for _, col, name in MAPS],
                       fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(-0.6, len(MAPS) + 4.4)
    ax.set_ylim(-0.8, K + 0.35)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    ax.text(len(MAPS) - 0.35, K - 0.25, "label: outlined maps, strongest first", fontsize=9, fontweight="bold",
            color="#3a3f45", va="bottom")
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.0, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["¼×", "½×", "cohort", "2×", "4×"], fontsize=8)
    cb.set_label("enrichment vs the cohort", fontsize=8.5)
    handles = [Line2D([0], [0], marker="o", ls="", mfc="white", mec=P2.INK, mew=1.8, ms=10,
                      label="interval above the cohort (outlined)"),
               Line2D([0], [0], marker="o", ls="", mfc=GREY_ROW, mec="none", ms=10, alpha=0.55,
                      label=f"✗ cluster > {int(100 * ONE_PATIENT)}% one patient")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.09 * 8 / K), ncol=2, frameon=False,
              fontsize=8.6)
    fig.suptitle(f"{MC.LABEL[g]}, K = {K} ({MC.K_NOTE}): where each cluster sits: the LanA language network and NeuroSynth maps",
                 x=0.02, ha="left", fontsize=14)
    fig.text(0.02, 1 - 0.55 / fig_h, "colour = enrichment (log scale)   ·   bubble size and number = % of the cluster inside "
             "the map   ·   95% intervals bootstrap patients   ·   no spatial null",
             fontsize=9, color=P2.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig_h))
    fig.savefig(OUTD / f"{g}_labels.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{g}_labels.png")
    for inf in info:
        flag = f"   X {100 * inf['share']:.0f}% {inf['top']}" if inf["share"] > ONE_PATIENT else ""
        print(f"   c{inf['j']}: {inf['label']}{flag}")

pd.DataFrame(rows_csv).to_csv(OUTD / "labels.csv", index=False)
print("wrote labels.csv")
