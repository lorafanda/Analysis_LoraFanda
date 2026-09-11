"""Step 1b - the Micromed clock against the Blackrock clock, one offset PER PART, to the
millisecond, from the sync pulses both systems record.

    python .\\11_mm_sync.py --patient G-05

WHAT WAS FOUND (2026-09-11). The TRC has no photodiode - X1-X6 are floating mains pickup,
MKR1-3 a bare clock - but MKR4+ carries, on top of that clock, a short pulse every
5.785 s. The Blackrock .nev carries a 4-word serial burst every 5.785 s. They are the same
event: the stimulation PC's sync generator, wired to both amplifiers. Matched, 50 of 52
bursts in a part land within 20 ms of a pulse (most within 5), residual sd about 3 ms,
drift about 20 us/s. That is the cross-system alignment the earlier micro/macro cross-correlation
approximated to +-0.35 s.

WHY PER PART. The .ns6 parts are not contiguous: Central closes one and opens the next
with a gap - 6.34 s and 4.81 s in G-05, confirmed three ways (header origins, the sync
cadence across the seam, and one whole trial missing from the photodiode in each gap:
the two 19-trial blocks). The session clock everything else uses is the concatenation,
so a single offset is right only for the part it was measured on and wrong by the gap on
every other one - a whole trial. Here each part gets its own.

HOW THE CYCLE IS FIXED. A 5.785 s pulse train matches itself every 5.785 s, so the pulses
alone cannot say which cycle is right. The seed is align_macro_offset_s (the brain
cross-correlation, good to +-0.35 s) on the part it was measured on (align_anchor_part),
carried to the other parts by the header origins (good to ~0.5 s). Both are far inside
half a cycle (2.9 s), so the search is +-scan s around the seed and the pulses then set
the phase to the millisecond. A different seed would be a different cycle: if the
brain-based offset were ever wrong by more than ~2.5 s this would lock onto the wrong
one, and the residual would look just as good. That is why the seed is reported.

WHAT IS WRITTEN.   outputs/<pat>/sync_parts.tsv  - one row per real part: its start on
the session clock, its start on the TRC clock, the offset (TRC = session + offset),
how many bursts matched, the residual and drift.  20_mm_macro.py reads it and maps each
trial with the offset of its own part; without it, it falls back to the single config
offset and says so.   outputs/<pat>/sync_parts.png - the match per part and the residual
over time.

WHAT IT DOES NOT CHANGE. The photodiode trial table and the micro analysis: cue and LFP
share one clock and one concatenation, so they are consistent inside a part. The only
Blackrock-side casualty of a seam is a trial whose epoch spans it; both analyses now drop
those (crosses_splice) and say which.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
import config_mm as cfg                                                      # noqa: E402
from lf_mm_io import (read_nsx_meta, read_nev_digital, read_trc_meta, pull_channels,  # noqa: E402
                      session_timeline, part_paths, real_parts, part_num, gaps, splices,
                      crosses_splice)

INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
BURST_GAP_S = 1.0          # .nev words closer than this belong to one burst
PAIR_S = 0.5               # MKR pulses closer than this are one event (they come in pairs)
MATCH_S = 0.010            # scan score tolerance; 'matched' in the outputs is within 2x this


# ---------------------------------------------------------------------------
def trc_path(pre):
    """Same lookup 20_mm_macro uses: the named TRC somewhere under the patient's raw tree."""
    t = pre["trc"]
    t = t[-1] if isinstance(t, list) else t
    root = pre["out_dir"].split("task_")[0] if "task_" in pre["out_dir"] else pre["out_dir"]
    for dirpath, _dn, fn in os.walk(root):
        if dirpath.count(os.sep) - root.count(os.sep) > 3:
            continue
        for f in fn:
            if f.lower() == t.lower():
                return os.path.join(dirpath, f)
    raise SystemExit(f"could not find {t} under {root}")


def origin_s(o):
    return dt.datetime.strptime(o, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def nev_bursts(part):
    """Start time (part clock) of every serial burst in a part's .nev."""
    nv = part["path"][:-4] + ".nev"
    if not os.path.exists(nv):
        return np.array([])
    t = read_nev_digital(nv)["t"]
    if len(t) == 0:
        return np.array([])
    return t[np.r_[0, np.flatnonzero(np.diff(t) > BURST_GAP_S) + 1]]


def mkr_pulses(x, fs):
    """Event starts on a marker channel: samples outside the channel's two clock levels,
    grouped so a pulse pair is one event. Returns (starts, n_levels_outside)."""
    x = x.astype(float) - np.median(x)
    lev, cnt = np.unique(np.round(x / 25.0) * 25.0, return_counts=True)
    clk = cnt > 0.05 * len(x)                          # a clock level holds a big share; a pulse never does
    top = lev[clk] if clk.any() else lev[np.argsort(cnt)[-2:]]
    lo, hi = top.min() - 100.0, top.max() + 100.0
    out = (x < lo) | (x > hi)
    ev = np.flatnonzero(out)
    if len(ev) == 0:
        return np.array([]), 0
    starts = [ev[0]]
    for i in ev[1:]:
        if i - starts[-1] > PAIR_S * fs:
            starts.append(i)
    return np.array(starts) / fs, int(((lev < lo) | (lev > hi)).sum())


def find_sync_channel(meta):
    """The marker channel whose extra pulses come at a 4-8 s cadence."""
    cands = [i for i, n in enumerate(meta["names"]) if n.upper().startswith("MKR")]
    if not cands:
        return None, None, {}
    got, fs = pull_channels(meta, cands, target_fs=None, dtype=meta["dtype"])
    report = {}
    best = None
    for i in cands:
        st, _n = mkr_pulses(got[i], fs)
        if len(st) < 20:
            report[meta["names"][i]] = f"{len(st)} pulses"
            continue
        med = float(np.median(np.diff(st)))
        report[meta["names"][i]] = f"{len(st)} pulses, cadence {med:.3f} s"
        if 4.0 <= med <= 8.0 and (best is None or len(st) > len(best[1])):
            best = (meta["names"][i], st)
    if best is None:
        return None, None, report
    return best[0], best[1], report


def match(bursts, pulses, start, dur):
    """Bursts (part clock) + start -> TRC clock, each against its nearest pulse."""
    tt = bursts + start
    inside = (tt > 0) & (tt < dur)
    if inside.sum() == 0:
        return np.array([]), np.array([]), 0
    j = np.clip(np.searchsorted(pulses, tt[inside]), 1, len(pulses) - 1)
    d1, d0 = pulses[j] - tt[inside], pulses[j - 1] - tt[inside]
    dd = np.where(np.abs(d1) < np.abs(d0), d1, d0)
    return tt[inside], dd, int(inside.sum())


def fit_part(bursts, pulses, seed, dur, scan):
    """The TRC start of a part: coarse scan +-scan s around the seed, then the median."""
    if len(bursts) == 0:
        return None
    grid = np.arange(seed - scan, seed + scan, 0.0005)
    score = np.empty(len(grid))
    for k, s in enumerate(grid):
        _t, dd, n = match(bursts, pulses, s, dur)
        score[k] = (np.abs(dd) < MATCH_S).mean() if n else 0.0
    if score.max() == 0:
        return None
    start = grid[np.argmax(score)]
    tt, dd, n = match(bursts, pulses, start, dur)
    ok = np.abs(dd) < 2 * MATCH_S
    start += float(np.median(dd[ok]))                  # centre the residual
    tt, dd, n = match(bursts, pulses, start, dur)
    ok = np.abs(dd) < 2 * MATCH_S
    if ok.sum() < max(3, 0.5 * n):                     # chance hits only: no cycle in the window
        return None
    slope = float(np.polyfit(tt[ok], dd[ok], 1)[0]) if ok.sum() > 5 else float("nan")
    return dict(start=start, n=n, n_ok=int(ok.sum()), tt=tt, dd=dd, ok=ok,
                sd_ms=float(dd[ok].std() * 1000) if ok.any() else float("nan"),
                drift_us_s=slope * 1e6, seed=seed)


# ---------------------------------------------------------------------------
def run(pat, scan=2.8):
    pre = cfg.MM_PRESETS[pat]
    pid = pre["pat_name"]
    off = pre.get("align_macro_offset_s")
    if off is None:
        raise SystemExit(f"{pat}: align_macro_offset_s is not set; the pulses cannot choose "
                         f"the cycle on their own (see the module docstring)")
    out_dir = os.path.join(ROOT, "outputs", pat)
    os.makedirs(out_dir, exist_ok=True)

    parts, total = session_timeline(part_paths(pre))
    rp = real_parts(parts)
    if not rp:
        raise SystemExit(f"{pat}: no .ns6 parts under {pre['blackrock_dir']}")
    print(f"\n=== {pat} ({pid})  sync   {len(rp)} parts, session {total:.0f} s")

    # the seams, and what the header origins say the gaps are
    for a, b in zip(rp[:-1], rp[1:]):
        gap = origin_s(b["origin"]) - (origin_s(a["origin"]) + a["dur"])
        seam = "SEAM" if abs(gap) > 0.5 else "contiguous"
        print(f"  {part_num(a)} -> {part_num(b)}: boundary at session {b['t0']:.1f} s, "
              f"gap {gap:+.2f} s by the header origins - {seam}")

    # the TRC and its sync channel
    path = trc_path(pre)
    meta = read_trc_meta(path)
    ch, pulses, report = find_sync_channel(meta)
    for n, r in report.items():
        print(f"  TRC {n:6s} {r}")
    if ch is None:
        raise SystemExit(f"{pat}: no marker channel carries a 4-8 s pulse train; the macro "
                         f"analysis keeps the single config offset ({off:+.1f} s)")
    cadence = float(np.median(np.diff(pulses)))
    print(f"  sync channel {ch}: {len(pulses)} events over {meta['dur']:.0f} s, "
          f"cadence {cadence:.3f} s")
    if scan >= 0.5 * cadence:
        raise SystemExit(f"{pat}: --scan {scan} s is not below half the {cadence:.3f} s "
                         f"cadence ({0.5 * cadence:.2f} s): two cycles would sit in the "
                         f"search window and the seed could no longer choose between them")

    # the seed per part: the brain offset on its anchor part, carried by the origins
    anchor = pre.get("align_anchor_part")
    if anchor is None:
        mid = 0.5 * (pre["time_range"][0] + pre["time_range"][1]) if pre.get("time_range") \
            else 0.5 * total
        anchor = int(part_num([p for p in rp if p["t0"] <= mid][-1]))
    ap = [p for p in rp if int(part_num(p)) == anchor]
    if not ap:
        raise SystemExit(f"{pat}: align_anchor_part {anchor} is not one of the parts")
    ap = ap[0]
    anchor_start = ap["t0"] + off
    print(f"  seed: part {anchor} starts at TRC {anchor_start:.2f} s "
          f"(align_macro_offset_s {off:+.1f}); other parts by header origin")

    rows, fits = [], {}
    for p in rp:
        name = part_num(p)
        seed = anchor_start + (origin_s(p["origin"]) - origin_s(ap["origin"]))
        overlap = min(seed + p["dur"], meta["dur"]) - max(seed, 0.0)
        if overlap < 4 * cadence:          # fewer than 5 bursts could be inside: skip
            print(f"  part {name}: outside the TRC by the seed - not fitted")
            continue                       # (before reading its .nev, which is 250 MB)
        bursts = nev_bursts(p)
        inside = ((bursts + seed) > 0) & ((bursts + seed) < meta["dur"])
        if len(bursts) == 0 or inside.sum() < 5:
            print(f"  part {name}: {inside.sum()} bursts inside the TRC - not fitted")
            continue
        f = fit_part(bursts, pulses, seed, meta["dur"], scan)
        if f is None:
            print(f"  part {name}: no match within +-{scan} s of the seed {seed:.2f}")
            continue
        fits[name] = f
        rows.append(dict(part=name, t0_session=round(p["t0"], 4), dur_s=round(p["dur"], 4),
                         origin=p["origin"], trc_start=round(f["start"], 4),
                         offset_s=round(f["start"] - p["t0"], 4), seed_trc_start=round(seed, 3),
                         n_bursts=f["n"], n_matched=f["n_ok"], resid_sd_ms=round(f["sd_ms"], 2),
                         drift_us_per_s=round(f["drift_us_s"], 1)))
        print(f"  part {name}: TRC start {f['start']:9.4f} s (seed {seed:8.2f}, "
              f"{f['start'] - seed:+.3f})  offset {f['start'] - p['t0']:+.4f} s  "
              f"matched {f['n_ok']}/{f['n']}  residual sd {f['sd_ms']:.2f} ms  "
              f"drift {f['drift_us_s']:+.1f} us/s")
    if not rows:
        raise SystemExit(f"{pat}: nothing fitted")

    # gaps as the pulses see them, against the origins
    for a, b in zip(rows[:-1], rows[1:]):
        gap_sync = b["trc_start"] - (a["trc_start"] + a["dur_s"])
        gap_orig = origin_s(b["origin"]) - (origin_s(a["origin"]) + a["dur_s"])
        print(f"  gap {a['part']} -> {b['part']}: {gap_sync:+.3f} s by the pulses, "
              f"{gap_orig:+.2f} s by the origins")

    tsv = os.path.join(out_dir, "sync_parts.tsv")
    with open(tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {tsv}")

    # the trials, by part and across seams
    tp = os.path.join(out_dir, f"{pid}_MM_trials.tsv")
    if os.path.exists(tp):
        import pandas as pd
        tr = pd.read_csv(tp, sep="\t")
        on = tr.onset_s.to_numpy()
        seams = splices(parts)
        x = crosses_splice(on, cfg.time_window, parts)
        t0s = [p["t0"] for p in rp]
        for k, p in enumerate(rp):
            hi = t0s[k + 1] if k + 1 < len(t0s) else np.inf
            sel = (on >= p["t0"]) & (on < hi)
            if sel.any():
                conds = ", ".join(f"{c} {int(n)}" for c, n in tr[sel].condition.value_counts().items())
                fitted = "fitted" if part_num(p) in fits else "NOT fitted"
                print(f"  trials in part {part_num(p)} ({fitted}): {int(sel.sum())}  [{conds}]")
        if x.any():
            print(f"  {int(x.sum())} trial(s) whose {cfg.time_window} s epoch crosses a seam "
                  f"(dropped by 20/21): " +
                  ", ".join(f"#{int(t)} {c} @{o:.1f}" for t, c, o in
                            zip(tr.trial[x], tr.condition[x], on[x])))
        print(f"  seams at session {', '.join(f'{s:.1f}' for s in seams) or 'none'} s")

    # figure
    fig, axes = plt.subplots(len(fits) + 1, 1, figsize=(11, 1.9 * len(fits) + 2.6), dpi=130,
                             gridspec_kw=dict(height_ratios=[1.0] * len(fits) + [1.5]))
    axes = np.atleast_1d(axes)
    for ax, (name, f) in zip(axes, fits.items()):
        t0, t1 = f["tt"][0] - 3, f["tt"][0] + 60
        for pt in pulses[(pulses > t0) & (pulses < t1)]:
            ax.axvline(pt, color="#c1121f", lw=0.9, alpha=0.8)
        for bt in f["tt"][(f["tt"] > t0) & (f["tt"] < t1)]:
            ax.axvline(bt, color="#1b232c", lw=0.9, ls=(0, (2, 2)))
        ax.set_xlim(t0, t1)
        ax.set_yticks([])
        ax.set_title(f"part {name}: first 60 s   red = {ch} pulse, dashed = .nev burst at "
                     f"TRC start {f['start']:.3f} s   ({f['n_ok']}/{f['n']} matched, "
                     f"sd {f['sd_ms']:.1f} ms)", fontsize=8.5, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=MUTED)
    ax = axes[-1]
    for name, f in fits.items():
        ax.plot(f["tt"][f["ok"]], f["dd"][f["ok"]] * 1000, ".", ms=3.5, label=f"part {name}")
        bad = ~f["ok"]
        if bad.any():
            ax.plot(f["tt"][bad], np.clip(f["dd"][bad] * 1000, -25, 25), "x", ms=4,
                    color="#c1121f")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("TRC clock (s)", fontsize=8)
    ax.set_ylabel("pulse - burst (ms)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=len(fits))
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title("residual per burst after its part's offset (x = unmatched, clipped)",
                 fontsize=8.5, color=INK, loc="left")
    fig.suptitle(f"{pat}   Micromed vs Blackrock on the shared sync pulses   "
                 f"({ch} vs .nev serial bursts)", fontsize=10, color=INK)
    fig.tight_layout()
    png = os.path.join(out_dir, "sync_parts.png")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"  wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="G-05")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scan", type=float, default=2.8,
                    help="search half-width around the seed, s (must stay below half the "
                         "5.785 s cadence or the cycle is no longer fixed by the seed)")
    a = ap.parse_args()
    for p in (cfg.patient_ids if a.all else [a.patient]):
        try:
            run(p, scan=a.scan)
        except SystemExit as e:
            print(f"  !! {p}: {e}")


if __name__ == "__main__":
    main()
