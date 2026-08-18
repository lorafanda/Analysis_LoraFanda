#!/usr/bin/env python3
"""
make_rt_story_figures.py - the four figures that carry the stage-05 argument.

The first version of this folder produced a competent battery of diagnostics -
five figure families x four groupings x six features - and argued nothing. These
four make one claim each, in order:

    RT-1  The agreement between roles and clusters in WARPED time does not
          survive the move to real time. Same electrodes, same groups, two time
          bases; only the axis differs, so any disagreement is the warp's doing.
    RT-2  Which groupings actually carry timing, all four side by side on one
          shared axis, each panel labelled with the within-patient permutation p.
    RT-3  Onset transfers across conditions about twice as well as peak. When a
          site STARTS responding is a property of where it is; when it PEAKS is a
          property of what the task demands.
    RT-S1 The patient-speed confound, demoted to a control.

    python make_rt_story_figures.py
    python make_rt_story_figures.py --n-perm 2000 --group hier_concat_hg
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "functions"))
import lf_rt as R  # noqa: E402

RT = HERE / "outputs" / "timing" / "timing_table.csv"
TN = HERE / "outputs" / "timing_tn" / "timing_table.csv"
FIGS = HERE / "outputs" / "figures"
TABS = HERE / "outputs" / "tables"
CONDS = ["audio", "picture", "reading"]

GROUPS = [
    ("hier_concat_hg",    "Ward · concat HG"),
    ("kmeans_concat_hg",  "k-means · concat HG"),
    ("cnmf_lead",         "convex-NMF lead"),
    ("role",              "pooling role (a-priori)"),
]
# every grouping an electrode carries, for the per-clustering page on the website
ALL_GROUPS = GROUPS + [
    ("kmeans_concat_rawds", "k-means · concat 15-band"),
    ("hier_concat_rawds",   "Ward · concat 15-band"),
]
CCOL = {"audio": "#2c7fb8", "picture": "#41ab5d", "reading": "#e6773a"}
INK, MUTED, LINE = "#1b232c", "#68727d", "#dfe4e9"


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def group_medians(df, group, feature, min_n=8):
    """Median feature per group level, dropping levels too small to mean anything."""
    d = df[[group, feature]].dropna()
    g = d.groupby(group)[feature].agg(["median", "size"])
    return g[g["size"] >= min_n]


# ---------------------------------------------------------------------------
# RT-1  warped time vs real time, same electrodes
# ---------------------------------------------------------------------------
def fig_rt1(rt, tn, feature="peak_lat", out_rows=None):
    """Slopegraph per grouping: where each group sits in warped %, where it sits
    in seconds, and whether the ORDER survives.

    Warped position is expressed as % of the response portion (50-100% -> 0-100)
    so the two panels are on comparable 0-1 scales and only the ORDER is compared,
    never the absolute numbers - a percentage and a second are not the same unit
    and the figure must not imply they are.
    """
    show = [g for g in GROUPS if g[0] in ("role", "hier_concat_hg")]
    fig, axes = plt.subplots(1, len(show), figsize=(5.6 * len(show), 6.0))
    axes = np.atleast_1d(axes)
    for ax, (gcol, glabel) in zip(axes, show):
        rows = []
        for cond in CONDS:
            a = group_medians(tn[tn.condition == cond], gcol, feature)
            b = group_medians(rt[rt.condition == cond], gcol, feature)
            common = sorted(set(a.index) & set(b.index), key=lambda x: str(x))
            if len(common) < 3:
                continue
            # rank within condition; only the ordering is comparable across bases
            ra = a.loc[common, "median"].rank(pct=True)
            rb = b.loc[common, "median"].rank(pct=True)
            rho, p = stats.spearmanr(a.loc[common, "median"], b.loc[common, "median"])
            for k in common:
                ax.plot([0, 1], [ra[k], rb[k]], "-", color=CCOL[cond], alpha=.55, lw=1.4)
                ax.plot([0, 1], [ra[k], rb[k]], "o", color=CCOL[cond], ms=3.5)
            rows.append(dict(group=gcol, condition=cond, n_levels=len(common),
                             spearman_rho=float(rho), p=float(p), feature=feature))
        ax.set_xlim(-.22, 1.22)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["warped\n(% of trial)", "real\n(s after GO)"], fontsize=9)
        ax.set_ylabel("rank within condition  (0 = earliest)", fontsize=9)
        ax.set_title(glabel, fontsize=11, loc="left")
        txt = "\n".join(f"{r['condition']}: ρ = {r['spearman_rho']:+.2f}"
                        for r in rows if r["group"] == gcol)
        ax.text(.5, -.13, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=MUTED)
        _clean(ax)
        if out_rows is not None:
            out_rows.extend(rows)
    for cond in CONDS:
        axes[0].plot([], [], color=CCOL[cond], lw=2, label=cond)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("RT-1  Does the warped-time ordering survive real time?   "
                 "Lines that cross = the order changed", x=.02, ha="left", fontsize=12.5)
    fig.tight_layout(rect=[0, .03, 1, .97])
    fig.savefig(FIGS / "RT-1_warped_vs_real.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RT-2  which groupings carry timing - one figure, shared axis
# ---------------------------------------------------------------------------
def fig_rt2(rt, feature, n_perm, stat_rows):
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(4.3 * len(GROUPS), 6.4),
                             sharex=True)
    lo = hi = None
    for ax, (gcol, glabel) in zip(axes, GROUPS):
        if gcol not in rt.columns or rt[gcol].notna().sum() == 0:
            ax.axis("off")
            continue
        order = (rt[[gcol, feature]].dropna().groupby(gcol)[feature]
                 .median().sort_values())
        keys = list(order.index)
        y = np.arange(len(keys))
        for oc, cond in enumerate(CONDS):
            med, elo, ehi = [], [], []
            for k in keys:
                v = rt.loc[(rt.condition == cond) & (rt[gcol] == k), feature]
                m, a, b = R.boot_ci(v)
                med.append(m)
                elo.append(m - a if np.isfinite(a) else np.nan)
                ehi.append(b - m if np.isfinite(b) else np.nan)
            ax.errorbar(med, y + (oc - 1) * .24, xerr=[elo, ehi], fmt="o",
                        ms=4, lw=1.2, capsize=2, color=CCOL[cond],
                        label=cond if gcol == GROUPS[0][0] else None)
        ax.set_yticks(y)
        # cluster ids arrive as floats from the merge; "3.0" is not a cluster name
        ax.set_yticklabels([f"{int(k)}" if isinstance(k, (int, float, np.integer,
                                                          np.floating)) and float(k).is_integer()
                            else str(k) for k in keys], fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(0, color=LINE, lw=1, zorder=0)
        ps = []
        for cond in CONDS:
            _, p, n, k = R.within_patient_perm(rt[rt.condition == cond], gcol,
                                               feature, n=n_perm)
            ps.append((cond, p, n, k))
            stat_rows.append(dict(group=gcol, feature=feature, condition=cond,
                                  n_electrodes=n, n_groups=k, perm_p=p))
        sig = sum(1 for _, p, _, _ in ps if np.isfinite(p) and p < .05)
        ax.set_title(f"{glabel}\n" + "  ".join(
            f"{c[:3]} p={p:.3f}" if np.isfinite(p) else f"{c[:3]} p=–"
            for c, p, _, _ in ps),
            fontsize=9.5, loc="left",
            color=INK if sig >= 2 else MUTED)
        _clean(ax)
        cur = ax.get_xlim()
        lo = cur[0] if lo is None else min(lo, cur[0])
        hi = cur[1] if hi is None else max(hi, cur[1])
    for ax in axes:
        if ax.has_data():
            ax.set_xlim(lo, hi)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[len(GROUPS) // 2].set_xlabel(
        "peak latency (s after GO)" if feature == "peak_lat"
        else f"{feature} (s after GO)", fontsize=10)
    fig.suptitle("RT-2  Which groupings order the response in real time?   "
                 "p = label shuffled WITHIN patient", x=.02, ha="left", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(FIGS / f"RT-2_groupings_{feature}.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RT-3  onset vs peak: site property or task property?
# ---------------------------------------------------------------------------
def fig_rt3(rt, rows):
    pairs = [("audio", "picture"), ("audio", "reading"), ("picture", "reading")]
    feats = [("onset_lat", "onset latency"), ("peak_lat", "peak latency")]
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], hspace=.42, wspace=.34)
    summary = {}
    for r, (feat, fname) in enumerate(feats):
        piv = rt.pivot_table(index=["patient_id", "contact_norm"],
                             columns="condition", values=feat, aggfunc="first")
        rhos = []
        for c, (a, b) in enumerate(pairs):
            ax = fig.add_subplot(gs[r, c])
            if a not in piv.columns or b not in piv.columns:
                ax.axis("off")
                continue
            d = piv[[a, b]].dropna()
            rho, p = stats.spearmanr(d[a], d[b])
            rhos.append(rho)
            ax.scatter(d[a], d[b], s=5, alpha=.22, lw=0, color="#4a6fa5")
            lim = [np.nanmin(d.to_numpy()), np.nanmax(d.to_numpy())]
            ax.plot(lim, lim, color=MUTED, lw=.8, ls="--")
            ax.set_title(f"{a} vs {b}\nρ = {rho:+.2f}   n = {len(d)}", fontsize=9)
            ax.set_xlabel(a, fontsize=8.5)
            ax.set_ylabel(fname if c == 0 else "", fontsize=9)
            _clean(ax)
            rows.append(dict(feature=feat, cond_a=a, cond_b=b, n=len(d),
                             spearman_rho=float(rho), p=float(p)))
        summary[feat] = (float(np.mean(rhos)) if rhos else np.nan, fname)
    ax = fig.add_subplot(gs[:, 3])
    ks = list(summary)
    ax.barh(np.arange(len(ks)), [summary[k][0] for k in ks],
            color=["#c1121f", "#4a6fa5"], height=.5)
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels([summary[k][1] for k in ks], fontsize=10)
    ax.set_xlabel("mean Spearman ρ across condition pairs", fontsize=9.5)
    ax.set_xlim(0, max(.5, max(summary[k][0] for k in ks) * 1.25))
    for i, k in enumerate(ks):
        ax.text(summary[k][0] + .012, i, f"{summary[k][0]:.2f}", va="center", fontsize=10)
    ax.set_title("onset travels;\npeak does not", fontsize=10.5, loc="left")
    _clean(ax)
    fig.suptitle("RT-3  Is response timing a property of the SITE or of the TASK?",
                 x=.02, ha="left", fontsize=12.5)
    fig.savefig(FIGS / "RT-3_onset_vs_peak_transfer.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RT-S1  the confound, as a control
# ---------------------------------------------------------------------------
def fig_rts1(rt, feature, rows):
    if "resp_med_s" not in rt.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, cond in zip(axes, CONDS):
        d = rt[rt.condition == cond][["patient_id", feature, "resp_med_s"]].dropna()
        if len(d) < 10:
            ax.axis("off")
            continue
        g = d.groupby("patient_id").agg(lat=(feature, "median"),
                                        resp=("resp_med_s", "first")).dropna()
        rho, p = stats.spearmanr(g["resp"], g["lat"]) if len(g) > 3 else (np.nan, np.nan)
        ax.scatter(d["resp_med_s"], d[feature], s=5, alpha=.18, lw=0, color="#b9c2cb")
        ax.scatter(g["resp"], g["lat"], s=32, color=CCOL[cond], zorder=3)
        ax.set_title(f"{cond}   patient-level ρ = {rho:+.2f}  (n={len(g)})", fontsize=9)
        ax.set_xlabel("patient median response duration (s)", fontsize=8.5)
        ax.set_ylabel(feature if cond == CONDS[0] else "", fontsize=9)
        _clean(ax)
        rows.append(dict(condition=cond, feature=feature, n_patients=len(g),
                         spearman_rho=None if not np.isfinite(rho) else float(rho),
                         p=None if not np.isfinite(p) else float(p)))
    fig.suptitle("RT-S1 (control)  Does electrode timing just track how fast the "
                 "patient answered?", x=.02, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(FIGS / f"RT-S1_patient_confound_{feature}.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)



# ---------------------------------------------------------------------------
# One figure per grouping, for the switchable panel on analysis_status
# ---------------------------------------------------------------------------
def fig_per_group(rt, tn, gcol, glabel, feature, n_perm, rows):
    """Left: this grouping's ordering in real seconds, medians with bootstrap CI.
    Right: the same groups' rank in warped time vs real time, so each clustering
    carries its own answer to 'did the warp create the order?'."""
    if gcol not in rt.columns or rt[gcol].notna().sum() == 0:
        return None
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 6.4),
                                   gridspec_kw=dict(width_ratios=[1.85, 1]))
    order = rt[[gcol, feature]].dropna().groupby(gcol)[feature].median().sort_values()
    keys = list(order.index)
    y = np.arange(len(keys))
    for oc, cond in enumerate(CONDS):
        med, elo, ehi = [], [], []
        for k in keys:
            v = rt.loc[(rt.condition == cond) & (rt[gcol] == k), feature]
            m, lo_, hi_ = R.boot_ci(v)
            med.append(m)
            elo.append(m - lo_ if np.isfinite(lo_) else np.nan)
            ehi.append(hi_ - m if np.isfinite(hi_) else np.nan)
        axL.errorbar(med, y + (oc - 1) * .24, xerr=[elo, ehi], fmt="o", ms=5,
                     lw=1.3, capsize=2.5, color=CCOL[cond], label=cond)
    sizes = rt[rt[gcol].notna()].groupby(gcol)["contact_norm"].nunique()
    axL.set_yticks(y)
    axL.set_yticklabels([(f"{int(k)}" if float(k).is_integer() else str(k))
                         + f"   n={sizes.get(k, 0)}"
                         if isinstance(k, (int, float, np.integer, np.floating))
                         else f"{k}   n={sizes.get(k, 0)}" for k in keys], fontsize=8)
    axL.invert_yaxis()
    axL.axvline(0, color=LINE, lw=1, zorder=0)
    axL.set_xlabel("peak latency (s after GO)", fontsize=10)
    axL.legend(frameon=False, fontsize=9, loc="lower right")
    _clean(axL)

    ps = []
    for cond in CONDS:
        _, p_, n_, k_ = R.within_patient_perm(rt[rt.condition == cond], gcol,
                                              feature, n=n_perm)
        ps.append((cond, p_))
        rows.append(dict(group=gcol, label=glabel, condition=cond, feature=feature,
                         n_electrodes=n_, n_groups=k_, perm_p=p_))
    axL.set_title(glabel + chr(10) + "   ".join(
        f"{c} p={p_:.3f}" if np.isfinite(p_) else f"{c} p=-" for c, p_ in ps),
        fontsize=11, loc="left")

    rho_txt = []
    if tn is not None:
        for cond in CONDS:
            a_ = group_medians(tn[tn.condition == cond], gcol, feature)
            b_ = group_medians(rt[rt.condition == cond], gcol, feature)
            common = sorted(set(a_.index) & set(b_.index), key=lambda x: str(x))
            if len(common) < 3:
                continue
            ra = a_.loc[common, "median"].rank(pct=True)
            rb = b_.loc[common, "median"].rank(pct=True)
            rho, _ = stats.spearmanr(a_.loc[common, "median"], b_.loc[common, "median"])
            for kk in common:
                axR.plot([0, 1], [ra[kk], rb[kk]], "-", color=CCOL[cond], alpha=.55, lw=1.4)
                axR.plot([0, 1], [ra[kk], rb[kk]], "o", color=CCOL[cond], ms=3.5)
            rho_txt.append(f"{cond} rho={rho:+.2f}")
    axR.set_xlim(-.22, 1.22)
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["warped" + chr(10) + "(% of trial)", "real" + chr(10) + "(s after GO)"], fontsize=9)
    axR.set_ylabel("rank within condition", fontsize=9)
    axR.set_title("does the warped order survive?" + chr(10) + "   ".join(rho_txt),
                  fontsize=9.5, loc="left")
    _clean(axR)

    fig.tight_layout()
    fp = FIGS / f"RT-G_{gcol}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--feature", default="peak_lat")
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    TABS.mkdir(parents=True, exist_ok=True)

    rt = pd.read_csv(RT)
    tn = pd.read_csv(TN) if TN.exists() else None
    print(f"RT {len(rt)} rows | TN {0 if tn is None else len(tn)} rows")

    rank_rows, stat_rows, transfer_rows, conf_rows = [], [], [], []
    if tn is not None:
        fig_rt1(rt, tn, a.feature, rank_rows)
        print("  RT-1 written")
    fig_rt2(rt, a.feature, a.n_perm, stat_rows)
    print("  RT-2 written")
    fig_rt3(rt, transfer_rows)
    print("  RT-3 written")
    fig_rts1(rt, a.feature, conf_rows)
    print("  RT-S1 written")

    per_rows = []
    for gcol, glabel in ALL_GROUPS:
        fp = fig_per_group(rt, tn, gcol, glabel, a.feature, a.n_perm, per_rows)
        if fp:
            print(f"  RT-G {gcol}")
    pd.DataFrame(per_rows).to_csv(TABS / "per_group_stats.csv", index=False)

    pd.DataFrame(rank_rows).to_csv(TABS / "warped_vs_real_rank.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(TABS / "story_group_stats.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(TABS / "story_transfer.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(TABS / "story_confound.csv", index=False)
    (TABS / "story_meta.json").write_text(json.dumps(dict(
        feature=a.feature, n_perm=a.n_perm, n_rt=len(rt),
        n_tn=0 if tn is None else len(tn),
        written=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), indent=2),
        encoding="utf-8")

    if stat_rows:
        S = pd.DataFrame(stat_rows)
        print("\n  within-patient permutation p:")
        print(S.pivot_table(index="group", columns="condition",
                            values="perm_p").round(4).to_string())
    if transfer_rows:
        T = pd.DataFrame(transfer_rows)
        print("\n  cross-condition transfer (mean rho):")
        print(T.groupby("feature")["spearman_rho"].mean().round(3).to_string())
    if rank_rows:
        K = pd.DataFrame(rank_rows)
        print("\n  warped vs real rank agreement (rho):")
        print(K.pivot_table(index="group", columns="condition",
                            values="spearman_rho").round(3).to_string())
    print(f"\n  -> {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
