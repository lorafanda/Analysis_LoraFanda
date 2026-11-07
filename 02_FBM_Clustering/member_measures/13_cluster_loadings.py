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
each patient, step 14's null; --perm rotations) at q < 0.05, Benjamini-Hochberg over EVERY bar of the figure.
--v4 (2026-11-07, as step 16 --v4; implies --outline shaft): two-sided. z = (share - mean of the shifts) / SD of the
shifts, p = 2 x the normal tail, BH over every bar; full colour = above chance, pale = neither, white with a coloured
edge = below chance; dark line = chance level, grey box = the corrected chance range (a bar outside it is exactly a
significant one); no bootstrap whisker or cohort tick (files _v4). --styles screen poster paper: _poster / _paper."""
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
ap.add_argument("--v4", action="store_true", help="two-sided test, corrected chance box, opaque bar styles (files _v4)")
MC.add_style_arg(ap)
a, _ = ap.parse_known_args()
if a.v4:
    a.outline = "shaft"
SUFFIX = ("" if a.loadings == [0.2, 0.3, 0.35, 0.4] else "_load" + "_".join(f"{int(round(100 * l)):03d}" for l in a.loadings))
SUFFIX += "_shaft" if a.outline == "shaft" else ""
SUFFIX += "_v4" if a.v4 else ""
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
        # the shaft order from the run's own contact names (the keys are normalised - upper case, no underscores - which
        # reorders the shafts inside a patient); the same order, and so the same shifts, as steps 14-16
        _d = P2.load_run(MC.FSETS[g][0], K)
        if [f"{p_}|{P2.norm(e_)}" for p_, e_ in zip(_d["meta"]["patient_id"], _d["meta"]["electrode"])] != list(t.key):
            raise SystemExit(f"{g}: measures.csv is not in the run's electrode order")
        sk = [(lambda r: (r.group(1), int(r.group(2))) if r else (n, 0))(re.match(r"^(.*?)(\d+)$", n))
              for n in _d["meta"]["electrode"].astype(str)]
        IM = np.column_stack([INS[m] for m in MAPS])
        cells, pv = [], []
        O4, S1, S2 = [], [], []                                        # --v4: per loading, the share and its null sums
        rs = np.random.default_rng(0)                                  # the shifts' own stream: step 16 draws the same
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
            s1, s2 = np.zeros_like(obs), np.zeros_like(obs)
            for _ in range(a.perm):
                perm = np.arange(len(kp))
                for gr in groups:
                    perm[gr] = np.roll(gr, rs.integers(1, len(gr)))
                sh = share_fn(lk[perm])
                ge += sh >= obs
                s1 += sh
                s2 += sh ** 2
            empty = np.array([not (lk == j).any() for j in range(K)])
            obs = obs.copy()
            obs[empty] = np.nan                                         # no members at this loading: no test
            O4.append(obs)
            S1.append(s1)
            S2.append(s2)
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
        T4 = MC.shaft_z(np.stack(O4), np.stack(S1), np.stack(S2), a.perm) if a.v4 else None
        if a.v4:
            print(f"  {g}: two-sided, BH cut-off |z| >= {T4['z_crit']:.2f}; above {int(T4['above'].sum())}, "
                  f"below {int(T4['below'].sum())}")

    nm, nl = len(MAPS), len(a.loadings)
    w = 0.84 / nl
    fig, axes = plt.subplots(K, 1, figsize=((0.42 * nm if a.v4 and nl == 1 else 0.24 * nm * nl) + 1.6, 1.9 * K + 1.2 + (0.4 if a.v4 else 0)), dpi=200, sharex=True, sharey=True)
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
                if a.v4:                                            # as step 16 --v4
                    up, dn = bool(T4["above"][li, j, mi]), bool(T4["below"][li, j, mi])
                    sure = up
                    blo, bhi, bm = T4["lo"][li, j, mi], T4["hi"][li, j, mi], T4["mean"][li, j, mi]
                    ax.add_patch(matplotlib.patches.Rectangle((x - w * 0.43, blo), w * 0.86, bhi - blo,
                                                              facecolor="#e3e5e8", edgecolor="none", zorder=1))
                    face = col if up else ((1, 1, 1) if dn else 0.38 * col + 0.62)
                    ax.bar(x, pct, width=w * 0.58, color=face, hatch=HATCH[li],
                           edgecolor=tuple(0.75 * col) if dn else "white", lw=1.0 if dn else 0, zorder=2)
                    ax.plot([x - w * 0.43, x + w * 0.43], [bm, bm], color="#2b3038", lw=1.1, solid_capstyle="butt",
                            zorder=4)
                    if dn:
                        ax.plot(x, blo, marker="v", color="#2b3038", ms=4, mew=0, zorder=5)
                    if one:
                        ax.plot(x, max(pct, bhi) + 2, "o", color=RED, ms=3.2, zorder=5, clip_on=False)
                else:
                    ax.bar(x, pct, width=w * 0.92, color=col, hatch=HATCH[li], edgecolor="white", lw=0, zorder=2)
                    if sure:
                        ax.bar(x, pct, width=w * 0.92, fill=False, edgecolor=P2.INK, lw=1.2, zorder=3)
                    ax.plot([x, x], [lo, hi], color=P2.INK, lw=0.5, zorder=3)
                    if one:                                         # red dot: > 50% one patient at this loading
                        ax.plot(x, hi + 3, "o", color=RED, ms=3.2, zorder=5, clip_on=False)
                rows.append(dict(fset=g, k=K, cluster=j, min_loading=lmin, n=int(sel.sum()), map=m[3:],
                                 pct_inside=round(pct, 1), lo=round(lo, 1), hi=round(hi, 1),
                                 cohort_pct=round(coh, 1), outlined=sure, top_patient=vc.index[0],
                                 top_patient_share=round(share, 2), one_patient=bool(one),
                                 q_shaft=round(float(Q[(li, j, mi)]), 4) if (li, j, mi) in Q else np.nan,
                                 **(dict(z=round(float(T4["z"][li, j, mi]), 2), p_two_sided=float(f"{T4['p'][li, j, mi]:.3g}"),
                                         q_two_sided=float(f"{T4['q'][li, j, mi]:.3g}"),
                                         above_chance=bool(T4["above"][li, j, mi]), below_chance=bool(T4["below"][li, j, mi]),
                                         chance_mean=round(float(T4["mean"][li, j, mi]), 2),
                                         box_lo=round(float(T4["lo"][li, j, mi]), 2), box_hi=round(float(T4["hi"][li, j, mi]), 2),
                                         z_crit=round(T4["z_crit"], 3)) if a.v4 else {})))
        for mi, m in enumerate(MAPS if not a.v4 else []):              # the cohort tick, once per map (not --v4)
            ax.plot([mi - 0.44, mi + 0.44], [100 * INS[m].mean()] * 2, color="black", lw=1.1, zorder=4)
        ax.set_ylabel(f"c{j}\n" + " / ".join(str(n_) for n_ in ns_), rotation=0, ha="right", va="center",
                      fontsize=9, color=col, fontweight="bold", labelpad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlim(-0.6, nm - 0.4)
    axes[-1].set_xticks(range(nm))
    axes[-1].set_xticklabels(NAMES, rotation=40, ha="right", fontsize=7.5)
    fig.supylabel("% of the cluster's electrodes inside the map", fontsize=9, x=0.004)
    _bl = np.array([0.22, 0.56, 0.82])
    V4H = ([Patch(facecolor=_bl, edgecolor="none", label="above chance"),
            Patch(facecolor=0.38 * _bl + 0.62, edgecolor="none", label="neither"),
            Patch(facecolor="white", edgecolor=0.75 * _bl, lw=1.0, label="below chance"),
            Line2D([0], [0], color="#2b3038", lw=1.1, label="chance level (shaft-shift null)"),
            Patch(facecolor="#e3e5e8", edgecolor="none", label="chance range (corrected)")] if a.v4 else [])
    axes[0].legend(handles=V4H + ([] if (a.v4 and nl == 1) else
                            [Patch(facecolor=LEGEND_FACE, hatch=HATCH[i] * 2, edgecolor="white", label=f"loading ≥ {l:g}")
                             for i, l in enumerate(a.loadings)])
                   + [Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50% one patient")], handlelength=2.4, handleheight=1.4, frameon=False,
                   fontsize=7.5, ncol=4 if a.v4 else nl + 1,
                   **(dict(loc="lower right", bbox_to_anchor=(1.0, 1.04)) if a.v4 else dict(loc="upper right")))
    axes[0].set_title(f"{MC.LABEL[g]}, K = {K}" + (f", members with loading ≥ {a.loadings[0]:g}" if nl == 1 else ""),
                      fontsize=10.5, loc="left")
    _lay = lambda f: f.tight_layout(h_pad=0.5, rect=(0.015, 0, 1, 1))
    _lay(fig)
    if a.v4:
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        MC.save_styles(fig, OUTD / f"{g}_cluster_loadings{SUFFIX}.png", a.styles, relayout=_lay, facecolor="white")
    else:
        MC.save_styles(fig, OUTD / f"{g}_cluster_loadings{SUFFIX}.png", a.styles, relayout=_lay, transparent=True)
    plt.close(fig)
    print("wrote", f"{g}_cluster_loadings{SUFFIX}.png")

out = pd.DataFrame(rows)
out.to_csv(OUTD / f"cluster_loadings{SUFFIX}.csv", index=False)
if len(out):
    print(f"wrote cluster_loadings{SUFFIX}.csv: {out.outlined.sum()} of {len(out)} bars outlined"
          + (f" (about {round(0.025 * len(out))} expected by chance)" if a.outline == "boot" else " (q < 0.05, shaft-shift null)"))
