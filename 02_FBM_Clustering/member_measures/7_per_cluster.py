"""Step 7 of member_measures (see 1_measures.py): per 15-band type, the same eight measures with the
FOUR FEATURE SETS on the x axis, one figure per type and colouring:
out/figures_per_cluster/c<j>_story_<colour>.png.

MATCHING, exactly FIG 4's rule (00_paper2_figure4_correspondence.py, its own functions): every
cluster is described by the mean concat_bands5 representation of its member electrodes (each
electrode z-scored across its features, members weighted by their convex-NMF loading), each
solution's own mean centroid is subtracted, the clusters of 15 bands and of the other feature set
are correlated, and pairs are taken highest correlation first, each cluster once. With unequal K a
15-band type can stay unmatched (shown as "no match"), and the other set's surplus clusters do not
appear. Each x label gives the centred r of the match and the electrode overlap (Jaccard), so a
weak match is visible: the shape correlation does not force the two clusters to share electrodes.
out/matching.csv keeps every pair.

READING ACROSS x. Loading, r with the type mean, silhouette and MMD witness are measured in each
feature set's OWN space: their levels are not comparable across x, only within it. Consensus, the
held-out hit rate, spatial grouping, gate margin and every colouring (P(LanA), white matter,
NeuroSynth, patient) are comparable. Rank strips (Bayesian bootstrap over patients, electrode-
weighted, 4000 draws) rank the four feature-set versions of the type, on the comparable measures
only. Right column: the colouring per feature set (NeuroSynth: share of the members inside the map;
otherwise the mean), 95% from bootstrapping patients.

    python 7_per_cluster.py                    every type, every colouring
    python 7_per_cluster.py --only ns_         colourings whose name starts with ns_
    python 7_per_cluster.py --type 1           one 15-band type
Reads out/<fset>/*.csv, out/white_matter.csv, out/neurosynth_*.csv."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, rankdata

import mm_common as MC

P2 = MC.p2()
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--type", type=int, default=None)
ap.add_argument("--k", default="published", help="published | a number (e.g. 7): see mm_common")
ap.add_argument("--ns", default="fdr", choices=["fdr", "nimare"], help="see mm_common")
ap.add_argument("--solutions", nargs="+", default=None,
                help="compare chosen solutions, each <fset>:<K>, the first the reference, e.g. "
                     "--solutions 15zband:7 15band:9 (needs steps 1 and 3 at each K)")
args = ap.parse_args()
FSET_OF = {v: k for k, v in MC.TAG.items()}               # tag -> concat_* name
SOL = {}                                                  # solution key -> tag, fset, k, dir, label
if args.solutions:
    for spec in args.solutions:
        tag, k = spec.split(":")
        k = int(k)
        SOL[f"{tag}@{k}"] = dict(tag=tag, fset=FSET_OF[tag], k=k, dir=MC.HERE / f"out_k{k}" / tag,
                                 label=f"{MC.LABEL[tag]} K{k}")
    missing = [key for key, v in SOL.items() if not (v["dir"] / "spatial_grouping.csv").exists()]
    if missing:
        raise SystemExit(f"run steps 1 and 3 first for {missing} (e.g. 1_measures.py --k <K> --fset <tag>)")
    TAGS = list(SOL)
    REF = TAGS[0]
    FIGS = MC.HERE / "out_compare" / "_vs_".join(k_.replace("@", "-k") for k_ in TAGS)
    if MC.ARGS.ns == "nimare":
        FIGS = FIGS / "nimare"
    MODE_NOTE = "chosen solutions"
else:
    for g in MC.FSETS:
        if (MC.OUT / g / "spatial_grouping.csv").exists():
            SOL[g] = dict(tag=g, fset=MC.FSETS[g][0], k=MC.FSETS[g][1], dir=MC.OUT / g, label=MC.LABEL[g])
    _missing = [g for g in MC.FSETS if g not in SOL]
    if _missing:
        print(f"  NOTE: no step 1/3 tables yet for {_missing} at this K - left out of the figures")
    if "15band" not in SOL:
        raise SystemExit("run steps 1 and 3 for 15band first: the clusters are matched to its types")
    TAGS = ["15band"] + [g for g in SOL if g != "15band"]
    REF = "15band"
    FIGS = MC.FIG_OUT / "figures_per_cluster"
    MODE_NOTE = MC.K_NOTE
FIGS.mkdir(parents=True, exist_ok=True)
MATCH_OUT = FIGS.parent / "matching.csv" if not args.solutions else FIGS / "matching.csv"

# ---- matching (FIG 4's functions) ----------------------------------------------------
F4 = MC.load_module(MC.SHARE / "00_paper2_figure4_correspondence.py", "mm_f4")
space = F4.description_space()
Zz = F4.electrode_z(space["Z"])
sols = {}
for tag in TAGS:
    fset, k = SOL[tag]["fset"], SOL[tag]["k"]
    s = F4.solution("cnmf", fset, k)
    F4.align(space, s)
    sols[tag] = dict(sol=s, C=F4.centroids(Zz, s["W"]))
K_REF = SOL[REF]["k"]
match = {REF: {j: (j, 1.0, 1.0) for j in range(K_REF)}}
rows = []
for tag in TAGS[1:]:
    S = F4.corr_matrix(sols[REF]["C"], sols[tag]["C"], centre=True)
    match[tag] = {}
    for i, j, r in F4.greedy_match(S):
        a, b = sols[REF]["sol"]["lab"] == i, sols[tag]["sol"]["lab"] == j
        jac = float((a & b).sum() / max((a | b).sum(), 1))
        match[tag][i] = (j, r, jac)
        rows.append(dict(fset=tag, type_15band=i, cluster=j, r_centred=round(r, 3), jaccard=round(jac, 3),
                         n_15band=int(a.sum()), n_cluster=int(b.sum())))
pd.DataFrame(rows).to_csv(MATCH_OUT, index=False)
print(pd.DataFrame(rows).to_string(index=False))

# ---- the per-feature-set tables ------------------------------------------------------
wm = pd.read_csv(MC.SHARED / "white_matter.csv")[["key", "pct_wm_cube"]]
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore") \
    if MC.NS_Z.exists() else None
ns_maps = pd.read_csv(MC.NS_MAPS).set_index("column") if ns is not None else pd.DataFrame()
T = {}
for tag in TAGS:
    t = pd.read_csv(SOL[tag]["dir"] / "measures.csv")
    t = t.merge(pd.read_csv(SOL[tag]["dir"] / "spatial_grouping.csv").drop(columns="type"), on="key", how="left")
    t = t.merge(wm, on="key", how="left")
    if ns is not None:
        t = t.merge(ns, on="key", how="left")
    t["patient"] = t.key.str.split("|").str[0]
    T[tag] = t
n = len(T[REF])
pats_all = sorted(T[REF].patient.unique())

M = {
    "loading": ("own-type loading", "own space: compare within a feature set", False, False),
    "r_type_mean": ("r with own type's mean", "own space: compare within a feature set", False, False),
    "silhouette": ("silhouette", "own space: compare within a feature set", False, False),
    "mmd_witness": ("MMD witness", "own space: compare within a feature set", False, False),
    "consensus": ("consensus with own type", "comparable across feature sets", True, False),
    "halves": ("held-out hit rate, halves", "comparable across feature sets", True, False),
    "spatial_centroid_mm": ("distance to its type's centroid (mm)", "comparable; lower = closer to the type's centre", True, False),
    "gate_margin": ("gate margin", "comparable (log axis)", True, True),
}
YLIM = {"loading": (0, 1), "r_type_mean": (-0.25, 1), "silhouette": (-0.25, 0.45), "mmd_witness": (-0.15, 0.3),
        "consensus": (0, 1), "halves": (0, 1), "spatial_centroid_mm": (0, 90), "gate_margin": (0.9, 17)}
SECTIONS = [("1   Does it belong to its type?", ["loading", "r_type_mean", "silhouette", "mmd_witness"]),
            ("2   Does it come back?", ["consensus", "halves"]),
            ("3   Grouped in space?", ["spatial_centroid_mm"]),
            ("4   How strong?", ["gate_margin"])]
CELLS = {"loading": (0, 0), "r_type_mean": (0, 1), "silhouette": (0, 2), "mmd_witness": (0, 3),
         "consensus": (1, 0), "halves": (1, 1), "spatial_centroid_mm": (1, 2), "gate_margin": (1, 3)}
DARK = np.array([0.10, 0.10, 0.11])
OUTSIDE = np.array([0.82, 0.82, 0.82])
INK_SEC = "#3a3f45"


def reference(tag, k_, cl):
    """The 'no structure' value for this measure in this feature set (None = none drawn)."""
    t = T[tag]
    K = SOL[tag]["k"]
    share = (t.type == cl).mean()
    return {"loading": 1 / K, "r_type_mean": 0.0, "silhouette": 0.0, "mmd_witness": 0.0,
            "consensus": float(((t.type.value_counts(normalize=True)) ** 2).sum()),
            "halves": share, "gate_margin": 1.0,
            "spatial_centroid_mm": float(np.median(t.spatial_centroid_ref_mm[t.type == cl]))}[k_]


# ---- colourings: per feature set, a value per electrode and a colour weight 0..1 -------
def colourings():
    P = T[REF].P_lana.to_numpy(float)
    cap = float(np.round(np.quantile(P, 0.90) / 0.05) * 0.05)
    out = [dict(name="lana", title="P(LanA)", kind="cont",
                var={g: T[g].P_lana.to_numpy(float) for g in TAGS},
                g={g: np.clip(T[g].P_lana.to_numpy(float) / cap, 0, 1) for g in TAGS},
                summ="mean P(LanA)", k=("0", f"{cap:.2f}+"))]
    out.append(dict(name="white_matter", title="% white matter around the contact", kind="cont",
                    var={g: T[g].pct_wm_cube.to_numpy(float) for g in TAGS},
                    g={g: np.clip(1 - T[g].pct_wm_cube.to_numpy(float) / 100, 0, 1) for g in TAGS},
                    summ="mean % white matter", k=("100% white", "100% grey")))
    for k_ in M:
        out.append(dict(name=f"rank_{k_}", title=f"{M[k_][0]} (rank within its feature set)", kind="cont",
                        var={g: (rankdata(T[g][k_]) - 1) / (n - 1) for g in TAGS},
                        g={g: (rankdata(T[g][k_]) - 1) / (n - 1) for g in TAGS},
                        summ="mean rank (0 lowest, 1 highest)", k=("lowest", "highest")))
    for c in [c for c in T[REF].columns if c.startswith("ns_")]:
        z = T[REF][c].to_numpy(float)
        zcap = float(np.quantile(z[z > 0], 0.90)) if (z > 0).any() else 1.0
        term = ns_maps.loc[c, "term"] if c in ns_maps.index else c[3:]
        out.append(dict(name=c, title=f"the NeuroSynth '{term}' map", kind="ns",
                        var={g: (T[g][c].to_numpy(float) > 0).astype(float) for g in TAGS},
                        z={g: T[g][c].to_numpy(float) for g in TAGS},
                        g={g: np.clip(T[g][c].to_numpy(float) / zcap, 0, 1) for g in TAGS},
                        summ="share of the members inside the map", k=("just inside (z > 0)", f"z {zcap:.1f}+"),
                        stats=ns_maps.loc[c] if c in ns_maps.index else None))
    pcol = P2.patient_colours(dict(patient=np.array(pats_all)))
    out.append(dict(name="patient", title="patient", kind="cat",
                    rgb={g: np.array([pcol[p][:3] for p in T[g].patient]) for g in TAGS},
                    legend=[(p, pcol[p]) for p in pats_all], ltitle="patient"))
    LANA = [("P < 0.1", "#c4c4c4"), ("0.1 ≤ P < 0.2", "#f0a030"), ("P ≥ 0.2", "#b2182b")]

    def cls(g):
        P_ = T[g].P_lana.to_numpy(float)
        return np.where(P_ >= 0.2, 2, np.where(P_ >= 0.1, 1, 0))
    out.append(dict(name="lana_class", title="LanA class", kind="cat",
                    rgb={g: np.array([matplotlib.colors.to_rgb(LANA[c][1]) for c in cls(g)]) for g in TAGS},
                    order={g: np.argsort(cls(g), kind="stable") for g in TAGS},
                    legend=[(nm, col) for nm, col in LANA], ltitle="P(LanA) at the contact"))
    return out


def tint(g, base):
    g = np.asarray(g, float)
    o = DARK[None, :] + np.clip(np.nan_to_num(g), 0, 1)[:, None] * (np.asarray(base)[None, :3] - DARK[None, :])
    o[~np.isfinite(g)] = OUTSIDE
    return o


def members(tag, j):
    """Row indices of the cluster matched to 15-band type j in feature set `tag` (None = no match)."""
    if j not in match[tag]:
        return None, None
    cl = match[tag][j][0]
    return np.where(T[tag].type.to_numpy() == cl)[0], cl


groups = {g: [np.where(T[g].patient.to_numpy() == p)[0] for p in pats_all] for g in TAGS}
BW = np.random.default_rng(0).dirichlet(np.ones(len(pats_all)), size=4000)    # draws x patients


def patient_weighted_means(j, getter):
    """(draws x 4) electrode-weighted means of `getter(tag, idx)` over the matched cluster, under
    Bayesian-bootstrap patient weights; NaN column for an unmatched feature set."""
    res = np.full((len(BW), len(TAGS)), np.nan)
    for c, tag in enumerate(TAGS):
        idx, _ = members(tag, j)
        if idx is None:
            continue
        v = getter(tag, idx)
        p_i = pd.Series(T[tag].patient.to_numpy()[idx]).map({p: i for i, p in enumerate(pats_all)}).to_numpy()
        ok = np.isfinite(v)
        S = np.bincount(p_i[ok], weights=v[ok], minlength=len(pats_all))
        N = np.bincount(p_i[ok], minlength=len(pats_all)).astype(float)
        if N.sum() == 0:                               # not computed for this feature set
            continue
        res[:, c] = (BW @ S) / np.maximum(BW @ N, 1e-12)
    return res


def panel(ax, k_, j, Cc, rng):
    title, note, comparable, logy = M[k_]
    col = np.asarray(P2.cluster_col(j, K_REF))[:3]
    labels = []
    for x, tag in enumerate(TAGS):
        idx, cl = members(tag, j)
        if idx is None:
            labels.append(f"{SOL[tag]['label']}\nno match")
            continue
        v = T[tag][k_].to_numpy(float)[idx]
        okv = np.isfinite(v)
        vv, ii = v[okv], idx[okv]
        if len(vv) < 2:                                # not computed for this feature set at this K
            labels.append(f"{SOL[tag]['label']}\nc{cl}\nnot computed")
            continue
        cat = Cc["kind"] == "cat"
        vp = ax.violinplot([vv], positions=[x], widths=0.80, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor("#9aa0a6" if cat else col); body.set_alpha(0.15 if cat else 0.13)
            body.set_edgecolor("#9aa0a6" if cat else col); body.set_linewidth(0.7)
        dens = gaussian_kde(vv)(vv) if np.ptp(vv) > 0 and len(vv) > 2 else np.ones(len(vv))
        xs = x + rng.uniform(-1, 1, len(vv)) * 0.37 * dens / dens.max()
        if cat:
            o = np.argsort(Cc["order"][tag].argsort()[ii]) if "order" in Cc else rng.permutation(len(ii))
            ax.scatter(xs[o], vv[o], s=6, c=Cc["rgb"][tag][ii][o], lw=0, zorder=3)
        elif Cc["kind"] == "ns":
            ins = Cc["z"][tag][ii] > 0
            ax.scatter(xs[~ins], vv[~ins], s=4, c=[OUTSIDE], lw=0, zorder=3)
            gg = Cc["g"][tag][ii]
            o = np.where(ins)[0][np.argsort(gg[ins])]
            ax.scatter(xs[o], vv[o], s=13, c=tint(0.25 + 0.75 * gg[o], col), lw=0, zorder=4)
        else:
            gg = Cc["g"][tag][ii]
            o = np.argsort(gg)
            ax.scatter(xs[o], vv[o], s=6, c=tint(gg[o], col), lw=0, zorder=3)
        ax.plot([x - 0.27, x + 0.27], [vv.mean()] * 2, color=P2.INK, lw=1.8, zorder=6)
        ref = reference(tag, k_, cl)
        ax.plot([x - 0.42, x + 0.42], [ref] * 2, color=P2.MUTED, ls=(0, (4, 3)), lw=1.0, zorder=2)
        _, r, jac = match[tag][j]
        labels.append(f"{SOL[tag]['label']}\nc{cl}" + ("" if tag == REF else f"\nr {r:.2f} · J {jac:.2f}"))
    if logy:
        ax.set_yscale("log"); ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(["1", "2", "4", "8", "16"])
        ax.minorticks_off()
    ax.set_ylim(*YLIM[k_])
    ax.set_xticks(range(len(TAGS))); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_xlim(-0.6, len(TAGS) - 0.4)
    ax.tick_params(labelsize=8.2, colors=P2.MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", pad=27)
    ax.annotate(f"{note}\ndashed = 'no structure' in that feature set", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points", fontsize=7.4, color=P2.MUTED, va="bottom",
                linespacing=1.25)
    if comparable:                                        # Bayesian ranks of the four versions
        sign = -1.0 if k_ == "spatial_centroid_mm" else 1.0      # closer = better
        mu = patient_weighted_means(j, lambda tag, idx: sign * (np.log(T[tag][k_].to_numpy(float)[idx])
                                                                if k_ == "gate_margin" else T[tag][k_].to_numpy(float)[idx]))
        okc = np.isfinite(mu).all(0)
        rk = np.full(mu.shape, np.nan)
        rk[:, okc] = (-mu[:, okc]).argsort(1).argsort(1) + 1
        sa = ax.inset_axes([0, -0.47, 1, 0.10])
        sa.set_xlim(-0.6, len(TAGS) - 0.4); sa.set_ylim(0, 1); sa.axis("off")
        ink = np.array(matplotlib.colors.to_rgb(P2.INK))
        for x in range(len(TAGS)):
            if not okc[x]:
                continue
            pb = float((rk[:, x] == 1).mean())
            sure = np.quantile(rk[:, x], 0.025) == np.quantile(rk[:, x], 0.975)
            sa.add_patch(plt.Rectangle((x - 0.45, 0.05), 0.9, 0.9, fc=(1 - pb) * np.ones(3) + pb * ink,
                                       ec=P2.INK if sure else "#c9ccd0", lw=1.1 if sure else 0.6))
            sa.text(x, 0.5, f"{np.nanmean(rk[:, x]):.1f}", ha="center", va="center", fontsize=7.2,
                    color="white" if pb > 0.5 else P2.INK)
        sa.text(-0.62, 0.5, "rank", ha="right", va="center", fontsize=7, color=P2.MUTED)


def draw(j, Cc):
    stats = Cc.get("stats")
    fig = plt.figure(figsize=(23.0, 13.0 if stats is not None else 11.8), dpi=150)
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 0.30, 0.95], wspace=0.30, hspace=1.05,
                  left=0.04, right=0.985, top=0.83 if stats is None else 0.815,
                  bottom=0.11 if stats is None else 0.21)
    rng = np.random.default_rng(0)
    axes = {}
    for k_, (r_, c_) in CELLS.items():
        axes[k_] = fig.add_subplot(gs[r_, c_])
        panel(axes[k_], k_, j, Cc, rng)
    axR = fig.add_subplot(gs[:, 5])
    if Cc["kind"] == "cat":
        axR.axis("off")
        h = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", ms=7, label=lb) for lb, c in Cc["legend"]]
        axR.legend(handles=h, loc="upper left", frameon=False, fontsize=8.4 if len(h) > 6 else 9.5,
                   labelspacing=0.55 if len(h) > 6 else 1.0, title=Cc["ltitle"], title_fontsize=9.5, alignment="left")
        sec5 = "5   Legend"
    else:
        mu = patient_weighted_means(j, lambda tag, idx: Cc["var"][tag][idx])
        yy = np.arange(len(TAGS))[::-1]
        for y_, c in zip(yy, range(len(TAGS))):
            if not np.isfinite(mu[0, c]):
                axR.text(0.5, y_, "no match", transform=axR.get_yaxis_transform(), ha="center", va="center",
                         fontsize=8, color=P2.MUTED, style="italic")
                continue
            lo, hi = np.nanquantile(mu[:, c], [0.025, 0.975])
            idx, _ = members(TAGS[c], j)
            obs = float(np.nanmean(Cc["var"][TAGS[c]][idx]))
            axR.plot([lo, hi], [y_, y_], color=P2.INK, lw=1.4)
            axR.plot(obs, y_, "o", color=P2.INK, ms=6)
        axR.set_yticks(yy); axR.set_yticklabels([SOL[g]["label"] for g in TAGS], fontsize=9.5)
        axR.set_ylim(-0.7, len(TAGS) - 0.3)
        axR.set_xlabel(f"{Cc['summ']}\n(dot; line = 95% from bootstrapping patients)", fontsize=9)
        axR.tick_params(labelsize=8.5, colors=P2.MUTED)
        axR.spines[["top", "right"]].set_visible(False)
        axR.set_title(Cc["summ"], fontsize=11, loc="left", pad=15)
        kax = fig.add_axes([0.80, 0.955 if stats is None else 0.935, 0.12, 0.014])
        kax.imshow(tint(np.linspace(0.25 if Cc["kind"] == "ns" else 0, 1, 64),
                        np.array(P2.cluster_col(j, K_REF))[:3])[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
        kax.set_yticks([]); kax.set_xticks([0, 1]); kax.set_xticklabels(list(Cc["k"]), fontsize=8)
        kax.set_title("colour: " + Cc["title"] + ("\nlight grey = outside the map" if Cc["kind"] == "ns" else ""),
                      fontsize=8.5, color=P2.MUTED)
        for s_ in kax.spines.values():
            s_.set_visible(False)
        sec5 = "5   The colour per feature set"

    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def section(label, axs):
        x0 = min(a.get_position().x0 for a in axs)
        x1 = max(a.get_position().x1 for a in axs)
        top = max(inv.transform((0, a.get_tightbbox(rr).y1))[1] for a in axs)
        fig.add_artist(Line2D([x0, x1 - 0.004], [top + 0.010] * 2, color=INK_SEC, lw=1.2, transform=fig.transFigure))
        fig.text(x0, top + 0.015, label, fontsize=12, color=INK_SEC, fontweight="bold", va="bottom")

    for label, keys in SECTIONS:
        section(label, [axes[k] for k in keys])
    section(sec5, [axR])
    if stats is not None:
        ns_ = stats.get("n_studies")
        ns_txt = f"{int(ns_)}" if pd.notna(ns_) and str(ns_).strip() != "" else "not entered"
        fig.text(0.04, 0.012, "the NeuroSynth map\n"
                 f"studies behind the map   {ns_txt}\n"
                 f"significant volume   {stats['sig_volume_cm3']:.0f} cm³  ({stats['sig_pct_brain']:.1f}% of the brain)\n"
                 f"peak z   {stats['peak_z']:.1f}      median z of the significant voxels   {stats['median_sig_z']:.1f}\n"
                 f"cohort electrodes inside   {int(stats['electrodes_inside'])} of {n}, {int(stats['patients_inside'])} patients",
                 fontsize=8.6, color=INK_SEC, va="bottom", linespacing=1.45,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f4f5f6", ec="#d0d3d6", lw=0.8))
    name = P2.type_name(SOL[REF]["fset"], K_REF, j)
    sep = " · " + name if name else ""
    fig.suptitle(f"c{j}{sep}   ·   {SOL[REF]['label']} across {MODE_NOTE}   ·   "
                 f"coloured by {Cc['title']}", x=0.04, y=0.988, ha="left", fontsize=15)
    fig.text(0.04, 0.968 if stats is None else 0.962,
             f"x = the cluster of each solution matched to this {SOL[REF]['label']} cluster by FIG 4's rule (centroid correlation in "
             "the 5-band description space,\ngrand mean removed, greedy one-to-one); r = that correlation, J = shared "
             "electrodes (Jaccard)   ·   rank strips: Bayesian bootstrap over patients, comparable measures only",
             fontsize=9, color=P2.MUTED, va="top", linespacing=1.4)
    out = FIGS / f"c{j}_story_{Cc['name']}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", out.name)


CURATED = {"patient", "lana", "lana_class", "white_matter", "ns_speech_perception", "ns_visual", "ns_reading",
           "ns_language_comprehension", "ns_semantic", "ns_retrieval", "ns_speech_production", "ns_premotor"}
for Cc in colourings():
    if not Cc["name"].startswith(args.only):
        continue
    if args.solutions and not args.only and Cc["name"] not in CURATED:
        continue
    for j in range(K_REF):
        if args.type is None or args.type == j:
            draw(j, Cc)
