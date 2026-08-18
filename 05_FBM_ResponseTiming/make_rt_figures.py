#!/usr/bin/env python3
"""
make_rt_figures.py - does the functional groups differ in WHEN they respond?

Reads outputs/timing/timing_table.csv (build_timing_table.py) and asks, per
condition, whether the groups an electrode already belongs to - k-means cluster,
hierarchical cluster, convex-NMF leading component, pooling role - predict the
timing of its high-gamma response after the GO cue.

    python make_rt_figures.py
    python make_rt_figures.py --group kmeans_concat_hg --feature peak_lat

WHY THE PERMUTATION TEST IS THE HEADLINE AND KRUSKAL-WALLIS IS NOT
Electrodes within a patient share a brain, a reference, a montage and a response
speed, so they are nowhere near independent. Kruskal-Wallis over pooled
electrodes treats them as if they were and will happily return p < 1e-10 for
structure that is entirely "patient A is slow and contributed 147 contacts". The
permutation here shuffles group labels WITHIN each patient, so it asks the
question that survives that: given this patient's own electrodes, does group
membership still order them in time?

Figures, all written to outputs/figures/:
    RT-T1  per-group latency distributions, per condition
    RT-T2  group ordering with bootstrap CIs - the "who responds when" figure
    RT-T3  cross-condition consistency of an electrode's latency
    RT-T4  graded version: component weight vs latency, no hard labels
    RT-T5  the confound check: electrode latency vs the patient's own response time
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

TIMING = HERE / "outputs" / "timing" / "timing_table.csv"
FIGS = HERE / "outputs" / "figures"
TABS = HERE / "outputs" / "tables"
CONDS = ["audio", "picture", "reading"]
GROUPS = ["kmeans_concat_hg", "hier_concat_hg", "cnmf_lead", "role"]
FEATURES = ["onset_lat", "peak_lat", "com", "dur", "peak_db", "auc"]
NICE = {"onset_lat": "onset latency (s after GO)", "peak_lat": "peak latency (s after GO)",
        "com": "centre of mass (s after GO)", "dur": "suprathreshold duration (s)",
        "peak_db": "peak high-gamma (dB)", "auc": "positive area (dB*s)"}


def within_patient_perm(df, group, feature, n=2000, seed=0):
    """Shuffle group labels within each patient; statistic is the spread of
    group medians. Returns (observed, p, n_electrodes, n_groups)."""
    d = df[["patient_id", group, feature]].dropna()
    if d[group].nunique() < 2 or len(d) < 20:
        return np.nan, np.nan, len(d), int(d[group].nunique())

    def spread(lbl):
        m = pd.DataFrame({"g": lbl, "v": d[feature].to_numpy()}).groupby("g")["v"].median()
        return float(m.max() - m.min())

    obs = spread(d[group].to_numpy())
    rng = np.random.default_rng(seed)
    pid = d["patient_id"].to_numpy()
    lbl = d[group].to_numpy().copy()
    idx_by_pat = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    null = np.empty(n)
    for i in range(n):
        sh = lbl.copy()
        for ix in idx_by_pat:
            sh[ix] = rng.permutation(sh[ix])
        null[i] = spread(sh)
    p = float((np.sum(null >= obs) + 1) / (n + 1))
    return obs, p, len(d), int(d[group].nunique())


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, (n, v.size), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fig_t1(T, group, feature):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    for ax, cond in zip(axes, CONDS):
        d = T[(T.condition == cond)][[group, feature]].dropna()
        if not len(d):
            ax.set_title(f"{cond} - no data"); ax.axis("off"); continue
        keys = sorted(d[group].unique(), key=lambda g: d.loc[d[group] == g, feature].median())
        data = [d.loc[d[group] == k, feature].to_numpy() for k in keys]
        ax.boxplot(data, vert=False, showfliers=False, widths=.6)
        ax.set_yticklabels([f"{k} (n={len(v)})" for k, v in zip(keys, data)], fontsize=7)
        ax.set_title(cond, fontsize=10)
        ax.set_xlabel(NICE.get(feature, feature))
        ax.axvline(0, color="0.6", lw=.8, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"RT-T1  {NICE.get(feature, feature)} by {group}, sorted by median",
                 x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / f"RT-T1_{group}_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_t2(T, group, feature, rows):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    keys = sorted(T[group].dropna().unique(),
                  key=lambda g: T.loc[T[group] == g, feature].median())
    ypos = np.arange(len(keys))
    for oc, cond in enumerate(CONDS):
        med, lo, hi = [], [], []
        for k in keys:
            v = T.loc[(T.condition == cond) & (T[group] == k), feature]
            m, a, b = boot_ci(v)
            med.append(m); lo.append(m - a if np.isfinite(a) else np.nan)
            hi.append(b - m if np.isfinite(b) else np.nan)
            rows.append(dict(group=group, feature=feature, condition=cond, level=str(k),
                             n=int(v.notna().sum()), median=m, ci_lo=a, ci_hi=b))
        ax.errorbar(med, ypos + (oc - 1) * .22, xerr=[lo, hi], fmt="o", capsize=2.5,
                    ms=4, lw=1.2, label=cond)
    ax.set_yticks(ypos)
    ax.set_yticklabels([str(k) for k in keys], fontsize=8)
    ax.set_xlabel(NICE.get(feature, feature))
    ax.axvline(0, color="0.6", lw=.8, ls=":")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"RT-T2  group ordering, median with bootstrap 95% CI ({group})",
                 loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / f"RT-T2_{group}_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_t3(T, feature):
    piv = T.pivot_table(index=["patient_id", "contact_norm"], columns="condition",
                        values=feature, aggfunc="first")
    pairs = [("audio", "picture"), ("audio", "reading"), ("picture", "reading")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    out = []
    for ax, (a, b) in zip(axes, pairs):
        if a not in piv.columns or b not in piv.columns:
            ax.axis("off"); continue
        d = piv[[a, b]].dropna()
        if len(d) < 10:
            ax.axis("off"); continue
        r, p = stats.spearmanr(d[a], d[b])
        ax.scatter(d[a], d[b], s=6, alpha=.28, lw=0, color="#4a6fa5")
        lim = [np.nanmin(d.to_numpy()), np.nanmax(d.to_numpy())]
        ax.plot(lim, lim, color="0.6", lw=.8, ls="--")
        ax.set_xlabel(a); ax.set_ylabel(b)
        ax.set_title(f"rho = {r:+.2f}   n = {len(d)}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        out.append(dict(feature=feature, cond_a=a, cond_b=b, n=len(d),
                        spearman_rho=float(r), p=float(p)))
    fig.suptitle(f"RT-T3  does an electrode keep its timing across conditions? "
                 f"({NICE.get(feature, feature)})", x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / f"RT-T3_consistency_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_t4(T, feature):
    """The graded view. 66% of electrodes have no majority component, so a hard
    label throws away most of the signal; correlating the weight itself does not."""
    wcols = sorted([c for c in T.columns if c.startswith("w") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    if not wcols:
        return []
    fig, axes = plt.subplots(len(wcols), 3, figsize=(11, 1.9 * len(wcols)),
                             sharex=True, sharey=True, squeeze=False)
    out = []
    for j, w in enumerate(wcols):
        for b, cond in enumerate(CONDS):
            ax = axes[j][b]
            d = T[(T.condition == cond)][[w, feature]].dropna()
            if len(d) < 10:
                ax.axis("off"); continue
            r, p = stats.spearmanr(d[w], d[feature])
            ax.scatter(d[w], d[feature], s=5, alpha=.25, lw=0, color=f"C{j % 10}")
            ax.set_title(f"{cond}  rho={r:+.2f}", fontsize=8)
            if b == 0:
                ax.set_ylabel(w, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            out.append(dict(component=w, condition=cond, feature=feature, n=len(d),
                            spearman_rho=float(r), p=float(p)))
    axes[-1][1].set_xlabel("component weight")
    fig.suptitle(f"RT-T4  graded membership vs {NICE.get(feature, feature)}",
                 x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / f"RT-T4_loading_vs_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_t5(T, feature):
    """If electrode latency just tracks how long that PATIENT took to respond,
    then any group difference could be a cohort-composition artefact."""
    if "resp_med_s" not in T.columns:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    out = []
    for ax, cond in zip(axes, CONDS):
        d = T[T.condition == cond][["patient_id", feature, "resp_med_s"]].dropna()
        if len(d) < 10:
            ax.axis("off"); continue
        g = d.groupby("patient_id").agg(lat=(feature, "median"),
                                        resp=("resp_med_s", "first")).dropna()
        r, p = stats.spearmanr(g["resp"], g["lat"]) if len(g) > 3 else (np.nan, np.nan)
        ax.scatter(d["resp_med_s"], d[feature], s=5, alpha=.2, lw=0, color="0.6")
        ax.scatter(g["resp"], g["lat"], s=34, color="#c1121f", zorder=3)
        ax.set_xlabel("patient median response duration (s)")
        ax.set_ylabel(NICE.get(feature, feature))
        ax.set_title(f"{cond}   patient-level rho = {r:+.2f} (n={len(g)})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        out.append(dict(condition=cond, feature=feature, n_patients=len(g),
                        spearman_rho=float(r) if np.isfinite(r) else None,
                        p=float(p) if np.isfinite(p) else None))
    fig.suptitle("RT-T5  is electrode timing just the patient's response speed?",
                 x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / f"RT-T5_patient_confound_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", choices=GROUPS)
    ap.add_argument("--feature", action="append", choices=FEATURES)
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args()
    groups = a.group or GROUPS
    feats = a.feature or ["onset_lat", "peak_lat", "com"]

    FIGS.mkdir(parents=True, exist_ok=True)
    TABS.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(TIMING)
    print(f"{len(T)} rows, {T['patient_id'].nunique()} patients")

    stat_rows, order_rows, cons_rows, load_rows, conf_rows = [], [], [], [], []
    for group in groups:
        if group not in T.columns or T[group].notna().sum() == 0:
            print(f"  skip {group}: no labels")
            continue
        for feature in feats:
            fig_t1(T, group, feature)
            fig_t2(T, group, feature, order_rows)
            for cond in CONDS:
                d = T[T.condition == cond]
                sub = d[[group, feature]].dropna()
                if sub[group].nunique() < 2:
                    continue
                arrs = [g[feature].to_numpy() for _, g in sub.groupby(group)]
                H, pkw = stats.kruskal(*arrs) if len(arrs) > 1 else (np.nan, np.nan)
                obs, pperm, n, k = within_patient_perm(d, group, feature, n=a.n_perm)
                stat_rows.append(dict(group=group, feature=feature, condition=cond,
                                      n_electrodes=n, n_groups=k,
                                      kruskal_H=float(H), kruskal_p=float(pkw),
                                      median_spread_s=obs, perm_p_within_patient=pperm))
                print(f"  {group:20s} {feature:10s} {cond:8s} n={n:5d} k={k:2d} "
                      f"spread={obs if obs is None else round(obs, 3)}  "
                      f"KW p={pkw:.2e}  within-patient perm p={pperm}")

    for feature in feats:
        cons_rows += fig_t3(T, feature)
        load_rows += fig_t4(T, feature)
        conf_rows += fig_t5(T, feature)

    pd.DataFrame(stat_rows).to_csv(TABS / "group_timing_stats.csv", index=False)
    pd.DataFrame(order_rows).to_csv(TABS / "group_timing_order.csv", index=False)
    pd.DataFrame(cons_rows).to_csv(TABS / "cross_condition_consistency.csv", index=False)
    pd.DataFrame(load_rows).to_csv(TABS / "loading_vs_timing.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(TABS / "patient_confound.csv", index=False)
    (TABS / "meta.json").write_text(json.dumps(dict(
        groups=groups, features=feats, n_perm=a.n_perm,
        n_rows=len(T), n_patients=int(T["patient_id"].nunique()),
        written=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf-8")
    print(f"\n  figures -> {FIGS}\n  tables  -> {TABS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
