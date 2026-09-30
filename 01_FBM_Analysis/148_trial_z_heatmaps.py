#!/usr/bin/env python3
"""
148_trial_z_heatmaps.py - channel x trial z per patient, from the 145 tables.

    python 148_trial_z_heatmaps.py                 every patient in the test tree
    python 148_trial_z_heatmaps.py --patient EL033 PAT_3965
    python 148_trial_z_heatmaps.py --z 4 --frac 0.20

One PNG per patient, the three conditions side by side: rows are channels grouped by shaft
(a line between shafts, the shaft name at the group), columns are the trials that reached the
ERSP (numbered as on the HFA figure), colour is the per-channel z of the 70-150 Hz score
(146.per_channel_z: z within each channel across its trials), clipped at ZMAX. A trial event
is a vertical column; a regional event (a discharge on one or two shafts) is a horizontal
block inside one column. The trials 146's rule drops at --z / --frac carry a marker above
the column. Nothing is re-computed.

Patients outside the clustering cohort are left out (EXCLUDE): EL044 (audio only, ECoG),
PAT_3301 (picture only), PAT_6684 (excluded).

Writes outputs/04_ersp-trialtestzscore/_summary/z_heatmaps/<pid>.png
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "functions"))
import config as cfg  # noqa: E402

_spec = importlib.util.spec_from_file_location("sweep146", HERE / "146_trial_zscore_sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

TREE = Path(cfg.outputs_root) / "04_ersp-trialtestzscore"
OUT = TREE / "_summary" / "z_heatmaps"
CONDS = ("audio", "picture", "reading")
EXCLUDE = {"EL044", "PAT_3301", "PAT_6684"}
ZMAX = 6.0
# the non-neural names that reach the score tables (EKG-, EMG-, photo, E1-E4): not rows here
NON_NEURAL = re.compile(r"^(EKG|EMG|ECG|PHOTO|E\d)[+-]?$", re.I)


def shaft_of(name: str) -> str:
    return re.sub(r"\d+$", "", re.sub(r"[-_]", "", str(name)).upper())


def contact_no(name: str) -> int:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 0


def channel_order(names) -> list[str]:
    """Shafts in order of first appearance, contacts by number inside each."""
    first = {}
    for n in names:
        first.setdefault(shaft_of(n), len(first))
    return sorted(set(names), key=lambda n: (first[shaft_of(n)], contact_no(n), str(n)))


def draw_patient(pid: str, d: pd.DataFrame, flagged: pd.DataFrame) -> Path:
    d = d[~d.channel.astype(str).str.match(NON_NEURAL)]
    conds = [c for c in CONDS if (d.condition == c).any()]
    chans = channel_order(d.channel.astype(str))
    n_ch = len(chans)
    widths = [max(1, d[d.condition == c].trial.nunique()) for c in conds]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.2 + 0.11 * sum(widths) + 0.6 * len(conds),
                                                     1.2 + 0.075 * n_ch), dpi=150,
                             gridspec_kw=dict(width_ratios=widths, wspace=0.06), squeeze=False)
    im = None
    for ax, cond in zip(axes[0], conds):
        s = d[d.condition == cond]
        Z = (s.pivot_table(index="channel", columns="trial", values="z", aggfunc="first")
               .reindex(chans))
        labels = (s.drop_duplicates("trial").set_index("trial")["trial_label"]
                    .reindex(Z.columns) if "trial_label" in s.columns else pd.Series(index=Z.columns))
        im = ax.imshow(Z.to_numpy(float), aspect="auto", cmap="magma_r", vmin=0, vmax=ZMAX,
                       interpolation="none")
        # shaft boundaries and names
        shafts = [shaft_of(c) for c in chans]
        starts = [i for i in range(n_ch) if i == 0 or shafts[i] != shafts[i - 1]]
        for i in starts[1:]:
            ax.axhline(i - 0.5, color="white", lw=0.8)
        if ax is axes[0][0]:
            ends = starts[1:] + [n_ch]
            ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)])
            ax.set_yticklabels([shafts[i] for i in starts], fontsize=6.5)
        else:
            ax.set_yticks([])
        # trial numbers as on the HFA figure, every trial when they fit
        xt = np.arange(Z.shape[1])
        step = 1 if Z.shape[1] <= 60 else 2
        ax.set_xticks(xt[::step])
        ax.set_xticklabels(["" if pd.isna(v) else str(int(v)) for v in labels.to_numpy()][::step],
                           fontsize=4.8, rotation=90)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(cond, fontsize=8, pad=8, loc="left")
        # the trials 146's rule drops
        f = flagged[(flagged.patient == pid) & (flagged.condition == cond)]
        for t in f.trial:
            if t in Z.columns:
                ax.plot(list(Z.columns).index(t), -1.0, marker="v", ms=4, color="#c1121f",
                        clip_on=False)
        ax.set_ylim(n_ch - 0.5, -0.5)        # the marker sits outside; keep every panel the same height
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.012, pad=0.01)
    cb.set_label("z", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.text(0.005, 0.995, pid, fontsize=9, fontweight="bold", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{pid}.png"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", nargs="*", default=None)
    ap.add_argument("--z", type=float, default=4.0)
    ap.add_argument("--frac", type=float, default=0.20)
    ap.add_argument("--score", default="score_hg", choices=["score_hg", "score_map"])
    a = ap.parse_args()

    raw = sweep.load(set(a.patient) if a.patient else None)
    raw = raw[~raw.patient.astype(str).isin(EXCLUDE)]
    d = sweep.per_channel_z(raw, a.score)
    g = sweep.dropped(d, a.z, a.frac)
    flagged = g[g.is_drop]
    pids = sorted(d.patient.astype(str).unique(), key=lambda p: (not p.startswith("EL"), p))
    for pid in pids:
        png = draw_patient(pid, d[d.patient == pid], flagged)
        n = int((flagged.patient == pid).sum())
        print(f"{pid:<10} {d[d.patient == pid].channel.nunique():>4} ch  flagged {n}  -> {png.name}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
