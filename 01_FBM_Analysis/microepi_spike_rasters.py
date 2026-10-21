#!/usr/bin/env python3
"""microepi_spike_rasters.py - per-unit spike rasters against the task, per condition.

Called by microepi_spikes.py --rasters. One page per unit: a raster of spikes around the
cue, split by condition, with the firing rate under it.

t=0 IS THE END OF THE CUE, the same anchor the analog rasters use, because that is when
the patient may answer. The stimulus length differs by condition (picture 1.0 s, audio
3.1 s, reading 3.5 s), so aligning on the cue's onset would put the go signal in three
different places; aligning on its end puts the response at a comparable time and leaves
the stimulus trailing off to the left, marked by the dotted line.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lf_microepi_ns6 as NS

INK, MUTED = "#22272b", "#7c858e"
COND_COL = {"picture": "#1b6ca8", "audio": "#c0392b", "reading": "#2e7d32"}
WIN = (-4.0, 4.0)
BIN = 0.050


def _load_sortings(root: Path):
    """{(sorter, shaft): sorting} for whatever has been sorted so far."""
    import spikeinterface.full as si
    out = {}
    for sorter_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "raw"):
        for shaft_dir in sorted(p for p in sorter_dir.iterdir() if p.is_dir()):
            s = shaft_dir / "sorting"
            if s.exists():
                try:
                    out[(sorter_dir.name, shaft_dir.name)] = si.load_extractor(str(s))
                except Exception as e:
                    print(f"  [skip] {sorter_dir.name}/{shaft_dir.name}: {e}")
    return out


def make_rasters(patient, root: Path, staged, files=None):
    root = Path(root)
    sortings = _load_sortings(root)
    if not sortings:
        raise SystemExit(f"{patient}: nothing sorted under {root} - run --sorter first")

    trials, sync = NS.trials_on_ns6(patient, files=files, verbose=True)
    ok = [t for t in trials if t.get("t_on") is not None]
    if not ok:
        raise SystemExit(f"{patient}: no trial could be placed on the .ns6 clock")
    conds = sorted({t["cond"] for t in ok})
    dur = {c: float(np.median([t["duration"] for t in ok if t["cond"] == c]))
           for c in conds}

    fig_dir = root / "rasters"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_fig = 0
    summary = []
    for (sorter, shaft), sorting in sortings.items():
        fs = float(sorting.get_sampling_frequency())
        for uid in sorting.unit_ids:
            st = np.asarray(sorting.get_unit_spike_train(unit_id=uid)) / fs
            if st.size < 50:                      # too few spikes to read anything from
                continue
            rate = st.size / (st[-1] - st[0]) if st[-1] > st[0] else 0.0
            summary.append(dict(sorter=sorter, shaft=shaft, unit=str(uid),
                                n_spikes=int(st.size), mean_hz=round(float(rate), 2)))
            fig, axes = plt.subplots(2, len(conds), figsize=(3.6 * len(conds), 5.0),
                                     dpi=160, sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 1]))
            axes = np.atleast_2d(axes).reshape(2, -1)
            for ci, c in enumerate(conds):
                rows = [t for t in ok if t["cond"] == c]
                top, bot = axes[0, ci], axes[1, ci]
                allsp = []
                for r, t in enumerate(sorted(rows, key=lambda z: z["t_on"])):
                    zero = t["t_on"] + t["duration"]          # the cue's END
                    s = st[(st >= zero + WIN[0]) & (st <= zero + WIN[1])] - zero
                    if s.size:
                        top.plot(s, np.full(s.size, r), "|", ms=2.6, mew=0.6,
                                 color=COND_COL.get(c, INK))
                        allsp.append(s)
                top.set_ylim(len(rows), -1)
                top.set_title(f"{c}  n={len(rows)}", fontsize=9, color=INK)
                edges = np.arange(WIN[0], WIN[1] + BIN, BIN)
                if allsp:
                    h, _ = np.histogram(np.concatenate(allsp), bins=edges)
                    bot.plot(edges[:-1] + BIN / 2, h / (len(rows) * BIN), lw=1.0,
                             color=COND_COL.get(c, INK))
                for ax in (top, bot):
                    ax.axvline(0, color=INK, lw=1.0)
                    ax.axvline(-dur[c], color=INK, lw=0.8, ls=":", alpha=0.7)
                    ax.tick_params(labelsize=7)
                    for sp in ("top", "right"):
                        ax.spines[sp].set_visible(False)
                bot.set_xlabel("s from the end of the cue", fontsize=8)
                if ci == 0:
                    top.set_ylabel("trial", fontsize=8)
                    bot.set_ylabel("Hz", fontsize=8)
            fig.suptitle(f"{patient} ({NS.SESSIONS[patient]['pat']}) · {shaft} · "
                         f"unit {uid} · {sorter} · {st.size} spikes, {rate:.1f} Hz",
                         fontsize=10.5, color=INK)
            png = fig_dir / f"{sorter}_{shaft}_u{uid}.png"
            fig.savefig(png, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            n_fig += 1
    (root / "units.json").write_text(json.dumps(summary, indent=1))
    print(f"  {n_fig} unit figures -> {fig_dir}")
    print(f"  {len(summary)} units kept (>=50 spikes) -> {root/'units.json'}")
    by = {}
    for s in summary:
        by[s["sorter"]] = by.get(s["sorter"], 0) + 1
    for k, v in sorted(by.items()):
        print(f"    {k}: {v} units")
    return 0
