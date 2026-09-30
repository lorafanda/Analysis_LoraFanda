#!/usr/bin/env python3
"""
147_trial_figure.py - the trials of every patient on one page: kept / removed, and the spread
of the per-trial z from the 145 tables.

    python 147_trial_figure.py                    every patient in the test tree
    python 147_trial_figure.py --z 4 --frac 0.20  mark the trials the sweep rule would drop
    python 147_trial_figure.py --patient PAT_3455 EL045

Reads outputs/04_ersp-trialtestzscore/<pid>/LM/Report/<pid>_IQR.tsv (the trial filters of
lf_trials.collect_trials: response accuracy, discharge spans, the stimulus / response
duration limits and the IQR outlier rule) and TrialScores/<pid>_<cond>_trial_scores.tsv (one
row per channel x trial, the 99th percentile of |dB| over 70-150 Hz). Nothing is re-computed:
146's per-channel z and its drop rule are imported, so the numbers here are the sweep's.

A  per patient x condition, a stacked bar: trials kept (solid) and the three ways a trial
   goes before the ERSP - incorrect response, discharge span, duration (stimulus < 0.5 s,
   response outside 0.2-10 s, or an IQR k=1.5 outlier on the response duration). The
   duration count is n_in - kept - accuracy - spans: the report keeps the first three, so a
   trial failing two tests at once is counted once, under the first.
B  the same grid, a violin per patient x condition of the TRIAL z: for each trial that reached
   the ERSP, the median over channels of its per-channel z (146.per_channel_z, z within each
   channel across that channel's trials). A montage-wide artefact is a trial whose median sits
   far right; a channel problem does not move the median. Every trial is a dot inside its
   violin; the red ones are the trials 146's rule drops at --z / --frac (share of channels
   over z), with their HFA-figure label.

Patients outside the clustering cohort are left out (EXCLUDE): EL044 (audio only, ECoG),
PAT_3301 (picture only), PAT_6684 (excluded).

Writes outputs/04_ersp-trialtestzscore/_summary/trial_counts_and_z.png and the two tables
behind it (trial_counts.tsv, trial_z.tsv).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "functions"))
import config as cfg  # noqa: E402

# 146 starts with a digit, so it is loaded by path rather than imported by name
_spec = importlib.util.spec_from_file_location("sweep146", HERE / "146_trial_zscore_sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

TREE = Path(cfg.outputs_root) / "04_ersp-trialtestzscore"
OUT = TREE / "_summary"
CONDS = ("audio", "picture", "reading")
# outside the clustering cohort: EL044 (audio only, ECoG), PAT_3301 (picture only), PAT_6684 (excluded)
EXCLUDE = {"EL044", "PAT_3301", "PAT_6684"}
# pastel hot pink / blue / brown (Lora, 2026-09-30)
CCOL = {"audio": "#f48fb1", "picture": "#8fb8de", "reading": "#c9a27e"}
REMOVED = [("incorrect response", "n_dropped_accuracy", "#9aa3ab"),
           ("discharge span", "n_dropped_spans", "#5d6770"),
           ("duration filters", "n_dropped_duration", "#d4d9de")]


def patient_order(pids) -> list[str]:
    """EL first, then PAT, as the audit lists them."""
    return sorted(pids, key=lambda p: (not str(p).startswith("EL"), str(p)))


def load_counts(patients=None) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(TREE / "*" / "LM" / "Report" / "*_IQR.tsv"))):
        d = pd.read_csv(f, sep="\t")
        d["patient"] = d["patient_id"].astype(str)
        if patients and d.patient.iloc[0] not in patients:
            continue
        rows.append(d)
    if not rows:
        raise SystemExit(f"no <pid>_IQR.tsv under {TREE} - run 145_trial_zscore_test.py first")
    c = pd.concat(rows, ignore_index=True)
    c["n_dropped_duration"] = (c.n_in - c.n_kept - c.n_dropped_accuracy - c.n_dropped_spans).clip(lower=0)
    return c[["patient", "condition", "n_in", "n_kept", "n_dropped_accuracy",
              "n_dropped_spans", "n_dropped_duration"]]


def trial_z(raw: pd.DataFrame, z: float, frac: float, score: str) -> pd.DataFrame:
    """One row per patient x condition x trial: the median z over channels and 146's verdict."""
    d = sweep.per_channel_z(raw, score)
    g = sweep.dropped(d, z, frac)                       # share, worst, median_z, label, is_drop
    return g[["patient", "condition", "trial", "label", "median_z", "worst", "share", "is_drop"]]


def draw(counts: pd.DataFrame, tz: pd.DataFrame, z: float, frac: float, score: str) -> Path:
    pids = patient_order(set(counts.patient) | set(tz.patient))
    x0 = {p: i for i, p in enumerate(pids)}
    w, off = 0.26, {"audio": -0.28, "picture": 0.0, "reading": 0.28}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, 0.62 * len(pids) + 3), 10.5), dpi=150,
                                   sharex=True, gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.08))

    # A - stacked bars
    for r in counts.itertuples():
        x = x0[r.patient] + off[r.condition]
        ax1.bar(x, r.n_kept, w, color=CCOL[r.condition], lw=0)
        base = r.n_kept
        for _, col, colr in REMOVED:
            n = int(getattr(r, col))
            if n:
                ax1.bar(x, n, w, bottom=base, color=colr, lw=0)
                base += n
        ax1.text(x, base + 0.6, str(int(r.n_in)), ha="center", va="bottom", fontsize=5.2, color="#555")
    ax1.set_ylabel("trials", fontsize=9)
    ax1.set_title("A   trials per patient and condition: kept (colour), removed before the ERSP (grey), "
                  "n in the trigger table on top", loc="left", fontsize=10)
    ax1.legend(handles=[Patch(color=CCOL[c], label=f"{c}, kept") for c in CONDS]
               + [Patch(color=colr, label=f"removed: {lab}") for lab, _, colr in REMOVED],
               fontsize=7.5, ncol=6, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.0))
    ax1.set_ylim(0, counts.n_in.max() * 1.22)
    ax1.grid(axis="y", alpha=0.25); ax1.set_axisbelow(True)

    # B - violins of the per-trial median z, every trial as a dot inside its violin
    rng = np.random.default_rng(0)                      # the jitter, the same on every run
    for cond in CONDS:
        data, pos = [], []
        for p in pids:
            v = tz[(tz.patient == p) & (tz.condition == cond)].median_z.to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size >= 2:
                data.append(v); pos.append(x0[p] + off[cond])
        if not data:
            continue
        parts = ax2.violinplot(data, positions=pos, widths=w * 1.05, showmedians=True,
                               showextrema=False, points=80)
        for b in parts["bodies"]:
            b.set_facecolor(CCOL[cond]); b.set_edgecolor("none"); b.set_alpha(0.28)
        parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.9)
        for v, x in zip(data, pos):
            ax2.plot(x + rng.uniform(-w * 0.32, w * 0.32, v.size), v, "o", ms=1.6,
                     color=CCOL[cond], mec="none", alpha=0.85, zorder=3)
    hit = tz[tz.is_drop]
    for r in hit.itertuples():
        x = x0[r.patient] + off[r.condition]
        ax2.plot(x, r.median_z, "o", ms=3.2, color="#c1121f", mec="white", mew=0.4, zorder=5)
        lab = "" if pd.isna(r.label) else f"{int(r.label)}"
        if lab:
            ax2.annotate(lab, (x, r.median_z), xytext=(3, 2), textcoords="offset points",
                         fontsize=5, color="#c1121f")
    ax2.axhline(0, color="#999", lw=0.6)
    ax2.axhline(z, color="#c1121f", lw=0.6, ls="--")
    ax2.text(len(pids) - 0.5, z, f" z = {z:g}", color="#c1121f", fontsize=7, va="bottom", ha="right")
    ax2.set_ylabel(f"trial z  (median over channels of the per-channel z of {score})", fontsize=8.5)
    ax2.set_title(f"B   spread of the trial z per patient and condition, one dot per trial; red = trials 146's rule drops "
                  f"at z > {z:g} on ≥ {frac:.0%} of channels ({int(hit.is_drop.sum())} of {len(tz)}), "
                  "labelled as on the HFA figure", loc="left", fontsize=10)
    ax2.set_xticks(range(len(pids)))
    ax2.set_xticklabels(pids, rotation=90, fontsize=7.5)
    ax2.set_xlim(-0.7, len(pids) - 0.3)
    ax2.grid(axis="y", alpha=0.25); ax2.set_axisbelow(True)

    fig.suptitle(f"Trials of the 04_ersp-trialtestzscore tree  ·  {len(pids)} patients  ·  "
                 f"{int(counts.n_kept.sum())} kept of {int(counts.n_in.sum())}  ·  147_trial_figure.py",
                 fontsize=11, x=0.01, ha="left")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "trial_counts_and_z.png"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", nargs="*", default=None)
    ap.add_argument("--z", type=float, default=4.0, help="146's z for the red dots (default 4)")
    ap.add_argument("--frac", type=float, default=0.20, help="146's share of channels (default 0.20)")
    ap.add_argument("--score", default="score_hg", choices=["score_hg", "score_map"])
    a = ap.parse_args()
    pats = set(a.patient) if a.patient else None

    counts = load_counts(pats)
    counts = counts[~counts.patient.isin(EXCLUDE)]
    raw = sweep.load(pats)
    raw = raw[~raw.patient.astype(str).isin(EXCLUDE)]
    tz = trial_z(raw, a.z, a.frac, a.score)

    OUT.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT / "trial_counts.tsv", sep="\t", index=False)
    tz.to_csv(OUT / "trial_z.tsv", sep="\t", index=False)
    png = draw(counts, tz, a.z, a.frac, a.score)

    s = counts.groupby("condition")[["n_in", "n_kept", "n_dropped_accuracy", "n_dropped_spans",
                                     "n_dropped_duration"]].sum()
    print(s.to_string())
    d = tz[tz.is_drop]
    print(f"\n{len(d)} trial(s) over z={a.z:g} on >= {a.frac:.0%} of channels"
          + (": " + ", ".join(f"{r.patient}/{r.condition} #{'' if pd.isna(r.label) else int(r.label)}"
                              for r in d.itertuples()) if len(d) else ""))
    print(f"-> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
