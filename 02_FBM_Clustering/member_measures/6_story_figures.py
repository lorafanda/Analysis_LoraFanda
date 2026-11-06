"""Step 6 of member_measures (see 1_measures.py): the sectioned figure, once per colouring.

    1  Does it belong to its type?   own-type loading, r with the type mean, silhouette, MMD witness
    2  Does it come back?            consensus over patient resamples, held-out hit rate (halves)
    3  Grouped in space?             share of its 10 nearest other-patient electrodes in its type
    4  How strong?                   gate margin
    5  Does the colour track it?     every measure's within-type rank correlation with the colour,
                                     95% from bootstrapping patients (a legend for categories)

Colourings (out/figures/story_<name>.png):
    lana             P(LanA), near-black at 0, the type's colour from its 90th percentile
    white_matter     % white matter (2_white_matter.py): full colour = grey matter, dark = white
    rank_<measure>   each measure's rank among all electrodes: dark = lowest, full colour = highest
    ns_<term>        every NeuroSynth map (4_neurosynth.py): light grey = outside (z <= 0); inside
                     shaded from dark to full colour by z (full from the 90th pct of inside z);
                     bottom left: the map's statistics from out/neurosynth_maps.csv
    patient          each patient its colour (FIG S0's), no cluster colour
    lana_class       P(LanA) < 0.1 / 0.1-0.2 / >= 0.2, no cluster colour

    python 6_story_figures.py                 every colouring, all four feature sets
    python 6_story_figures.py --fset 5band    one feature set
    python 6_story_figures.py --only ns_      only colourings whose name starts with ns_
Under every panel, when out/bayes_ranks.csv exists (5_bayes_ranks.py): two strips, the type's
posterior mean rank (1 = highest) weighting electrodes and weighting patients, each cell shaded
by P(rank 1); a cell is outlined where the 95% rank interval is a single rank.
Types in FIG 3 A's order (mean P(LanA)) everywhere. Reads out/*.csv only."""
import argparse
import os
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, rankdata

HERE = Path(__file__).resolve().parent
SHARE = HERE.parent
OUT = HERE / "out"
FIGS = OUT / "figures"
os.environ["PAPER2_PAPER"] = "0"
P2 = types.ModuleType("mm_p2_story")
P2.__file__ = str(SHARE / "00_Paper2_Figures.py")
sys.modules["mm_p2_story"] = P2
exec(compile(Path(P2.__file__).read_text(encoding="utf-8"), P2.__file__, "exec"), P2.__dict__)

sys.path.insert(0, str(HERE))
import mm_common as MC                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="draw only the colourings whose name starts with this")
ap.add_argument("--fset", default="all", choices=["all", *MC.FSETS])
ap.add_argument("--k", default="published", help="published | a number (e.g. 7): see mm_common")
ap.add_argument("--ns", default="fdr", choices=["fdr", "nimare"], help="see mm_common")
args = ap.parse_args()
if args.fset == "all":                                 # one process per feature set
    import subprocess
    for tag_ in MC.FSETS:
        print(f"\n=== {tag_}")
        subprocess.run([sys.executable, __file__, "--fset", tag_, "--only", args.only, "--k", args.k, "--ns", args.ns],
                       check=True)
    sys.exit(0)
TAG = args.fset
FSET, K_PUB = MC.FSETS[TAG]
OUT_F = MC.OUT / TAG
FIGS = MC.FIG_OUT / "figures"

t = pd.read_csv(OUT_F / "measures.csv")
t = t.merge(pd.read_csv(OUT_F / "spatial_grouping.csv").drop(columns="type"), on="key", how="left")
t = t.merge(pd.read_csv(OUT / "white_matter.csv")[["key", "pct_wm_cube"]], on="key", how="left")
ns_maps = pd.DataFrame()
if MC.NS_Z.exists():
    t = t.merge(pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore"),
                on="key", how="left")
    ns_maps = pd.read_csv(MC.NS_MAPS).set_index("column")
K = int(t.type.max()) + 1
lab, P = t.type.to_numpy(), t.P_lana.to_numpy(float)
pat = t.key.str.split("|").str[0].to_numpy()
n = len(t)
share = np.bincount(lab, minlength=K) / n
CHANCE_CO = float((share ** 2).sum())
rank = sorted(range(K), key=lambda j: -P[lab == j].mean())
DARK = np.array([0.10, 0.10, 0.11])
OUTSIDE = np.array([0.82, 0.82, 0.82])
VIOLIN_GREY = "#9aa0a6"
INK_SEC = "#3a3f45"

#  key: title, what higher means, ylim, reference (number | "share"), its label, log axis
M = {
    "loading": ("own-type loading", "higher = more of its activity is its own type's",
                (0, 1), 1 / K, f"1/{K}: no preference", False),
    "r_type_mean": ("r with own type's mean", "higher = looks more like its type's average",
                    (-0.25, 1), 0.0, "0: no resemblance", False),
    "silhouette": ("silhouette", "> 0 closer to its own type than to the next",
                   (-0.2, 0.35), 0.0, "0: in between", False),
    "mmd_witness": ("MMD witness", "> 0 more like its own type than the nearest other",
                    (-0.1, 0.2), 0.0, "0: in between", False),
    "consensus": ("consensus with own type", "higher = stays with its type when patients are resampled",
                  (0, 1), CHANCE_CO, f"chance {CHANCE_CO:.2f}", False),
    "halves": ("held-out hit rate, halves", "higher = put back in its type with half the patients unseen",
               (0, 1), "share", "chance for that type (its share)", False),
    "spatial_centroid_mm": ("distance to its type's centroid (mm)",
                            "lower = closer to the type's centre (other patients, folded, loading-weighted)",
                            (0, 90), "cref", "non-members' median distance to that centre", False),
    "gate_margin": ("gate margin", "higher = further above the responsiveness line (log axis)",
                    (0.9, 17), 1.0, "1: the gate's line", True),
}
SECTIONS = [
    ("1   Does it belong to its type?", ["loading", "r_type_mean", "silhouette", "mmd_witness"]),
    ("2   Does it come back?", ["consensus", "halves"]),
    ("3   Grouped in space?", ["spatial_centroid_mm"]),
    ("4   How strong?", ["gate_margin"]),
]
CELLS = {"loading": (0, 0), "r_type_mean": (0, 1), "silhouette": (0, 2), "mmd_witness": (0, 3),
         "consensus": (1, 0), "halves": (1, 1), "spatial_centroid_mm": (1, 2), "gate_margin": (1, 3)}
ORDER = [k for _, ks in SECTIONS for k in ks]


# ---- the colourings: each a dict the drawing reads
def continuous(name, title, var, g, k_lo, k_hi, ktitle, inside=None, stats=None):
    return dict(name=name, title=title, kind="cont", var=var, g=g, k_lo=k_lo, k_hi=k_hi,
                ktitle=ktitle, inside=inside, stats=stats)


COL = []
cap = float(np.round(np.quantile(P, 0.90) / 0.05) * 0.05)
COL.append(continuous("lana", "P(LanA)", P, np.clip(P / cap, 0, 1), "0", f"{cap:.2f}+",
                      "colour: P(LanA), dark to the type's colour"))
wm = t.pct_wm_cube.to_numpy(float)
COL.append(continuous("white_matter", "% white matter around the contact", wm, np.clip(1 - wm / 100, 0, 1),
                      "100% white matter", "100% grey matter", "colour: % white matter around the contact"))
for k_ in ORDER:
    v = t[k_].to_numpy(float)
    COL.append(continuous(f"rank_{k_}", M[k_][0], v, (rankdata(v) - 1) / (n - 1), "lowest", "highest",
                          f"colour: {M[k_][0]}, as a rank"))
for c in [c for c in t.columns if c.startswith("ns_")]:
    z = t[c].to_numpy(float)
    term = ns_maps.loc[c, "term"] if c in ns_maps.index else c[3:].replace("_", " ")
    inside = z > 0
    zcap = float(np.quantile(z[inside], 0.90)) if inside.any() else 1.0
    COL.append(continuous(c, f"the NeuroSynth '{term}' map", z, np.clip(z / zcap, 0, 1),
                          "just inside (z > 0)", f"z {zcap:.1f}+",
                          f"z on the '{term}' map: {int(inside.sum())} electrodes inside\n"
                          f"light grey = outside the map (z ≤ 0)", inside=inside,
                          stats=ns_maps.loc[c] if c in ns_maps.index else None))
pcol = P2.patient_colours(dict(patient=pat))
COL.append(dict(name="patient", title="patient", kind="cat",
                rgb=np.array([pcol[p][:3] for p in pat]), order=np.random.default_rng(3).permutation(n),
                legend=[(f"{p}   {int((pat == p).sum())}", pcol[p]) for p in sorted(pcol)],
                ltitle="patient   electrodes", how="each patient its own colour (FIG S0's)"))
LANA = [("P < 0.1 · outside", "#c4c4c4"), ("0.1 ≤ P < 0.2 · inside at 0.1", "#f0a030"),
        ("P ≥ 0.2 · the core", "#b2182b")]
cls = np.where(P >= 0.2, 2, np.where(P >= 0.1, 1, 0))
COL.append(dict(name="lana_class", title="LanA class", kind="cat",
                rgb=np.array([matplotlib.colors.to_rgb(LANA[c][1]) for c in cls]),
                order=np.argsort(cls, kind="stable"),
                legend=[(f"{nm}\n{int((cls == c).sum())} electrodes ({100 * (cls == c).mean():.0f}%)", col)
                        for c, (nm, col) in enumerate(LANA)],
                ltitle="P(LanA) at the contact", how="the LanA network's thresholds 0.1 and 0.2"))


def tint(g, base):
    g = np.asarray(g, float)
    out = DARK[None, :] + np.clip(np.nan_to_num(g), 0, 1)[:, None] * (np.asarray(base)[None, :3] - DARK[None, :])
    out[~np.isfinite(g)] = OUTSIDE
    return out


def panel(ax, k_, C, rng):
    title, what, ylim, ref, rlab, logy = M[k_]
    v = t[k_].to_numpy(float)
    if not np.isfinite(v).any():                       # no table for this feature set at this K
        ax.text(0.5, 0.5, "not computed for this feature set at this K", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=P2.MUTED, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(title, fontsize=11, loc="left", pad=27)
        return
    cat = C["kind"] == "cat"
    pos = np.full(n, np.nan)
    for i, j in enumerate(rank):
        col = np.asarray(P2.cluster_col(j, K))[:3]
        m = (lab == j) & np.isfinite(v)
        vv = v[m]
        if len(vv) < 2:                                # e.g. a one-patient type: no centroid from others
            ax.text(i, 0.5, "n/a", transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=7.5, color=P2.MUTED, style="italic")
            continue
        vp = ax.violinplot([vv], positions=[i], widths=0.80, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(VIOLIN_GREY if cat else col); body.set_alpha(0.15 if cat else 0.13)
            body.set_edgecolor(VIOLIN_GREY if cat else col); body.set_linewidth(0.7)
        dens = gaussian_kde(vv)(vv) if np.ptp(vv) > 0 else np.ones(len(vv))
        pos[np.where(m)[0]] = i + rng.uniform(-1, 1, len(vv)) * 0.37 * dens / dens.max()
        if not cat:
            idx = np.where(m)[0]
            if C["inside"] is None:
                o = idx[np.argsort(np.nan_to_num(C["g"][idx], nan=-1.0))]
                ax.scatter(pos[o], v[o], s=6, c=tint(C["g"][o], col), lw=0, zorder=3)
            else:
                ins = C["inside"][idx]
                ax.scatter(pos[idx[~ins]], v[idx[~ins]], s=4, c=[OUTSIDE], lw=0, zorder=3)
                o = idx[ins][np.argsort(C["g"][idx[ins]])]
                ax.scatter(pos[o], v[o], s=13, c=tint(0.25 + 0.75 * C["g"][o], col), lw=0, zorder=4)
        ax.plot([i - 0.27, i + 0.27], [vv.mean()] * 2, color=P2.INK, lw=1.8, zorder=6)
        if ref in ("share", "cref"):
            rv = share[j] if ref == "share" else float(np.median(t.spatial_centroid_ref_mm.to_numpy(float)[m]))
            ax.plot([i - 0.42, i + 0.42], [rv] * 2, color=P2.MUTED, ls=(0, (4, 3)), lw=1.0, zorder=2)
    if cat:
        o = C["order"][np.isfinite(pos[C["order"]])]
        ax.scatter(pos[o], v[o], s=6, c=C["rgb"][o], lw=0, zorder=3)
    if ref not in ("share", "cref"):
        ax.axhline(ref, color=P2.MUTED, ls=(0, (4, 3)), lw=1.0, zorder=2)
    if logy:
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(["1", "2", "4", "8", "16"])
        ax.minorticks_off()
    ax.set_ylim(*ylim)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"c{j}" for j in rank], fontsize=9)
    ax.set_xlim(-0.6, K - 0.4)
    ax.tick_params(labelsize=8.5, colors=P2.MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    is_colour = C["name"] == f"rank_{k_}"
    ax.set_title(title + ("   (the colour)" if is_colour else ""), fontsize=11, loc="left", pad=27)
    ax.annotate(f"{what}\ndashed = {rlab}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                textcoords="offset points", fontsize=7.4, color=P2.MUTED, va="bottom", linespacing=1.25)


RANKS = pd.read_csv(OUT_F / "bayes_ranks.csv") if (OUT_F / "bayes_ranks.csv").exists() else None


def rank_strips(ax, k_):
    """Two strips under the panel: posterior mean rank per type, electrode- and patient-weighted,
    shaded by P(rank 1) (white = never first, ink = always first)."""
    if RANKS is None:
        return
    ink = np.array(matplotlib.colors.to_rgb(P2.INK))
    for row, (wname, lbl) in enumerate((("electrodes", "rank \u00b7 electrodes"),
                                        ("patients", "rank \u00b7 patients"))):
        r = RANKS[(RANKS.measure == k_) & (RANKS.weighting == wname)].set_index("type")
        if r.empty:                                    # measure not computed: no strip
            return
        sa = ax.inset_axes([0, -0.27 - row * 0.115, 1, 0.10])
        sa.set_xlim(-0.6, K - 0.4); sa.set_ylim(0, 1); sa.axis("off")
        for i, j in enumerate(rank):
            q = r.loc[j]
            if not np.isfinite(t[k_].to_numpy(float)[lab == j]).any():   # no values: not ranked
                sa.text(i, 0.5, "n/a", ha="center", va="center", fontsize=7, color=P2.MUTED, style="italic")
                continue
            c = (1 - q.p_best) * np.ones(3) + q.p_best * ink
            sure = q.rank_lo == q.rank_hi
            sa.add_patch(plt.Rectangle((i - 0.45, 0.05), 0.9, 0.9, fc=c,
                                       ec=P2.INK if sure else "#c9ccd0", lw=1.1 if sure else 0.6))
            sa.text(i, 0.5, f"{q.mean_rank:.1f}", ha="center", va="center", fontsize=7.2,
                    color="white" if q.p_best > 0.5 else P2.INK)
        sa.text(-0.62, 0.5, lbl, ha="right", va="center", fontsize=7, color=P2.MUTED)


def within_type_rho(idx, a, b):
    xa, xb = [], []
    for j in range(K):
        s = idx[lab[idx] == j]
        s = s[np.isfinite(a[s]) & np.isfinite(b[s])]
        if len(s) < 3:
            continue
        xa.append(rankdata(a[s]) / len(s)); xb.append(rankdata(b[s]) / len(s))
    if not xa:
        return np.nan
    xa, xb = np.concatenate(xa), np.concatenate(xb)
    return float(np.corrcoef(xa, xb)[0, 1]) if xa.std() and xb.std() else np.nan


groups = [np.where(pat == p)[0] for p in np.unique(pat)]
brng = np.random.default_rng(1)
BOOTS = [np.concatenate([groups[q] for q in brng.integers(0, len(groups), len(groups))]) for _ in range(1000)]


def stats_box(fig, s):
    """Bottom left: the NeuroSynth map's own numbers, to judge how much weight it can carry."""
    ns_ = s.get("n_studies")
    ns_txt = f"{int(ns_)}" if pd.notna(ns_) and str(ns_).strip() != "" else "not entered (neurosynth_term_studies.csv)"
    lines = [f"studies behind the map   {ns_txt}",
             f"significant volume   {s['sig_volume_cm3']:.0f} cm³"
             + (f"  ({s['sig_pct_brain']:.1f}% of the brain)" if pd.notna(s.get("sig_pct_brain")) else ""),
             f"peak z   {s['peak_z']:.1f}      median z of the significant voxels   {s['median_sig_z']:.1f}",
             f"our electrodes inside   {int(s['electrodes_inside'])} of {n} ({s['pct_electrodes_inside']:.1f}%), "
             f"{int(s['patients_inside'])} of {len(groups)} patients, median z {s['median_z_inside']:.1f}",
             f"file   {s['file']}"]
    fig.text(0.04, 0.012, "the NeuroSynth map\n" + "\n".join(lines), fontsize=8.6, color=INK_SEC, va="bottom",
             family="DejaVu Sans", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f4f5f6", ec="#d0d3d6", lw=0.8))


def draw(C):
    stats = C.get("stats")
    fig = plt.figure(figsize=(23.0, 12.4 if stats is not None else 11.2), dpi=150)
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 0.30, 0.95], wspace=0.30, hspace=0.62 if RANKS is None else 1.0,
                  left=0.04, right=0.985, top=0.83 if stats is None else 0.815,
                  bottom=(0.06 if stats is None else 0.17) + (0.06 if RANKS is not None else 0))
    rng = np.random.default_rng(0)
    axes = {}
    for k_, (r_, c_) in CELLS.items():
        axes[k_] = fig.add_subplot(gs[r_, c_])
        panel(axes[k_], k_, C, rng)
        rank_strips(axes[k_], k_)
    axR = fig.add_subplot(gs[:, 5])
    if C["kind"] == "cont":
        rows = []
        for k_ in ORDER:
            if C["name"] == f"rank_{k_}":
                rows.append((M[k_][0], np.nan, np.nan, np.nan))
                continue
            v = t[k_].to_numpy(float)
            if not np.isfinite(v).any():
                rows.append((M[k_][0] + " (not computed)", np.nan, np.nan, np.nan))
                continue
            obs = within_type_rho(np.arange(n), C["var"], v)
            bs = np.array([within_type_rho(b, C["var"], v) for b in BOOTS])
            rows.append((M[k_][0], obs, *np.nanquantile(bs, [0.025, 0.975])))
        yy = np.arange(len(rows))[::-1]
        span = max(0.3, np.ceil(np.nanmax(np.abs([r[2:] for r in rows])) * 10) / 10)
        for y_, (title, obs, lo, hi) in zip(yy, rows):
            if np.isfinite(obs):
                axR.plot([lo, hi], [y_, y_], color=P2.INK, lw=1.4)
                axR.plot(obs, y_, "o", color=P2.INK, ms=6)
            else:
                axR.text(0, y_, "not computed" if title.endswith("(not computed)") else "the colour itself",
                         ha="center", va="center", fontsize=8,
                         color=P2.MUTED, style="italic")
        axR.axvline(0, color=P2.MUTED, ls=(0, (4, 3)), lw=1.0)
        axR.set_yticks(yy); axR.set_yticklabels([r[0] for r in rows], fontsize=9)
        axR.set_ylim(-0.7, len(rows) - 0.3); axR.set_xlim(-span, span)
        axR.set_xlabel("within-type rank correlation with the colour\n(dot; line = 95% from bootstrapping "
                       "patients)", fontsize=9)
        axR.tick_params(labelsize=8.5, colors=P2.MUTED)
        axR.spines[["top", "right"]].set_visible(False)
        axR.set_title("every measure on one axis", fontsize=11, loc="left", pad=15)
        axR.annotate("> 0: the measure is higher where the colour variable is higher", xy=(0, 1),
                     xycoords="axes fraction", xytext=(0, 3), textcoords="offset points", fontsize=7.4,
                     color=P2.MUTED, va="bottom")
        kax = fig.add_axes([0.80, 0.955 if stats is None else 0.935, 0.12, 0.014])
        kax.imshow(tint(np.linspace(0.25 if C["inside"] is not None else 0, 1, 64),
                        np.array([0.30, 0.55, 0.85]))[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
        kax.set_yticks([]); kax.set_xticks([0, 1]); kax.set_xticklabels([C["k_lo"], C["k_hi"]], fontsize=8)
        nod = int((~np.isfinite(C["var"])).sum())
        kax.set_title(C["ktitle"] + (f"\n({nod} without a value: light grey)" if nod else ""),
                      fontsize=8.5, color=P2.MUTED)
        for s_ in kax.spines.values():
            s_.set_visible(False)
        sub = "types in FIG 3 A's order   ·   short line = type mean"
    else:
        axR.axis("off")
        handles = [Line2D([0], [0], marker="o", ls="", mfc=col, mec="none", ms=7, label=lb)
                   for lb, col in C["legend"]]
        axR.legend(handles=handles, loc="upper left", frameon=False,
                   fontsize=9 if len(handles) < 6 else 8.4, labelspacing=0.9 if len(handles) < 6 else 0.55,
                   title=C["ltitle"], title_fontsize=9.5, alignment="left")
        sub = f"{C['how']}   ·   no cluster colour   ·   types in FIG 3 A's order   ·   short line = type mean"

    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def section(label, axs):
        x0 = min(a.get_position().x0 for a in axs)
        x1 = max(a.get_position().x1 for a in axs)
        top = max(inv.transform((0, a.get_tightbbox(rr).y1))[1] for a in axs)
        fig.add_artist(Line2D([x0, x1 - 0.004], [top + 0.010] * 2, color=INK_SEC, lw=1.2,
                              transform=fig.transFigure))
        fig.text(x0, top + 0.015, label, fontsize=12, color=INK_SEC, fontweight="bold", va="bottom")

    for label, keys in SECTIONS:
        section(label, [axes[k] for k in keys])
    section("5   Does the colour track any of it?" if C["kind"] == "cont" else "5   Legend", [axR])
    if stats is not None:
        stats_box(fig, stats)
    fig.suptitle(f"{MC.LABEL[TAG]}   ·   coloured by {C['title']}", x=0.04, y=0.988, ha="left", fontsize=15)
    fig.text(0.04, 0.958 if stats is None else 0.952,
             f"Eight per-electrode measures   ·   {MC.LABEL[TAG]} x convex NMF, K = {K} ({MC.K_NOTE}), all {n} electrodes   ·   {sub}" + ("\nstrips under the panels: posterior mean rank (1 = highest), shaded by P(rank 1), Bayesian bootstrap over patients" if RANKS is not None else ""),
             va="top", linespacing=1.4,
             fontsize=10, color=P2.MUTED)
    out = FIGS / f"{TAG}_story_{C['name']}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", out.name)


FIGS.mkdir(parents=True, exist_ok=True)
for C in COL:
    if C["name"].startswith(args.only):
        draw(C)
