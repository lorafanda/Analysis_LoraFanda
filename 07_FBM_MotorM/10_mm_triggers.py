"""Step 1 - turn the Blackrock photodiode into a labelled trial table, and show its work.

    python .\10_mm_triggers.py --patient G-05
    python .\10_mm_triggers.py --patient G-05 --force      (re-read the raw parts)
    python .\10_mm_triggers.py --all

WHAT IT DOES. Pulls the photodiode channel out of every .ns6 part, caches it, finds the
cue edges, splits them into the five blocks, labels the blocks, and writes a trial table
plus a QC figure. Nothing here reads a .mat.

WHY LEVEL CROSSINGS AND NOT THE USUAL DERIVATIVE DETECTOR. The detector used elsewhere
normalises the derivative of the trace by its maximum over the analysed window, so its
threshold is a FRACTION OF THE LARGEST EDGE PRESENT. G-05 has a calibration flash 47x the
size of its cue pulses (6661 against 143) sitting in the same recording; with that flash
in the window every real cue falls far below threshold and the detector finds 9 events
instead of ~144. That is not a threshold that needs tuning, it is a threshold defined
against the wrong reference.

This detects crossings of the MIDPOINT BETWEEN THE TWO LEVELS instead, with hysteresis and
a minimum dwell. The cue is a two-level signal, so its own levels are the right reference,
and a flash elsewhere in the file cannot move them. time_range still exists in the config
for excluding a region outright, but it is no longer load-bearing.

THE CACHE. Extracting one channel from an interleaved 30 kHz file costs a full read of
every part - about 4.6 GB for G-05 - so the decimated photodiode is cached next to the
outputs. Detection and plotting then re-run in a second, which is what makes the
thresholds tunable by eye rather than by a five-minute round trip. --force re-reads.

WHAT TO LOOK AT in the figure: the block structure should be five clean groups; the
inter-trial interval should sit at the design's ~6.2 s; and the trial counts per block
should match the behavioural log where one exists. If any of those is wrong the trial
table is wrong, and everything after it inherits that.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
import config_mm as cfg                                   # noqa: E402
from lf_mm_io import (read_nsx_meta, session_timeline, pull_channels,   # noqa: E402
                      find_channel, save_cache, load_cache)

PD_FS = 1000.0          # 1 ms edge resolution is ample for a 3.2 s cue
MIN_DWELL_S = 0.8       # a cue level lasts ~3 s; anything briefer is noise
HYST = 0.25             # of the level separation, to stop chatter on the crossing
BLOCK_GAP_S = 8.0       # longer than a trial (6.2 s), shorter than a block pause (~11 s)
DETREND_S = 20.0        # rolling baseline window - see detrend() for why this exists
DESPIKE_MS = 60         # narrow artefacts to remove before detection - see despike()
CUE_BIN_S = 10.0        # bin size for locating the cue train automatically
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
BLUE, RED, GREEN = "#4a6fa5", "#c1121f", "#1b7837"


# ---------------------------------------------------------------------------
def part_paths(pre):
    folder = pre["blackrock_dir"]
    first, last = pre["blackrock"]
    stem, a = first.rsplit("-", 1)
    b = last.rsplit("-", 1)[1]
    return [os.path.join(folder, f"{stem}-{n}.ns6") for n in range(int(a), int(b) + 1)]


def extract_pd(pat, pre, force=False):
    """The photodiode across the whole session, decimated, cached."""
    cache = os.path.join(ROOT, "outputs", pat, "photodiode.npy")
    if not force:
        arr, meta = load_cache(cache)
        if arr is not None:
            print(f"  cached photodiode: {len(arr)} samples @ {meta['fs']:.0f} Hz "
                  f"({meta['dur']:.1f} s)   [--force to re-read]")
            return arr, meta
    parts, total = session_timeline(part_paths(pre))
    if not parts:
        raise SystemExit(f"{pat}: no .ns6 parts under {pre['blackrock_dir']}")
    print(f"  {len(parts)} parts, {total:.1f} s total - reading the photodiode "
          f"(this is the slow step, ~{sum(os.path.getsize(p['path']) for p in parts)/1e9:.1f} GB)")
    chunks, bounds = [], []
    for k, p in enumerate(parts, 1):
        m = read_nsx_meta(p["path"])
        idx = find_channel(m["names"], [pre.get("bk_pd_channel", "ainp1"),
                                        "photodiode", "ainp1"])
        if idx is None:
            print(f"    !! {p['name']}: no photodiode channel, skipped")
            continue
        got, fs_out = pull_channels(m, [idx], target_fs=PD_FS)
        chunks.append(got[idx])
        bounds.append(dict(name=p["name"], t0=p["t0"], dur=p["dur"], origin=p["origin"]))
        print(f"    [{k}/{len(parts)}] {p['name']}  {p['dur']:7.1f} s  "
              f"ch {m['names'][idx]}")
    arr = np.concatenate(chunks).astype(np.float32)
    meta = dict(fs=float(PD_FS), dur=len(arr) / PD_FS, parts=bounds,
                channel=pre.get("bk_pd_channel", "ainp1"))
    save_cache(cache, arr, meta)
    print(f"  cached -> {cache}")
    return arr, meta


# ---------------------------------------------------------------------------
def detrend(x, fs, win_s=DETREND_S):
    """Remove a slowly drifting baseline, so the cue's own two levels stay the reference.

    G-05's cue is 143 units peak-to-peak on a DC level that wanders further than that
    across the session. Estimating the two levels from one global percentile then puts the
    decision midpoint in the wrong place for whole stretches of the recording, and every
    edge there is missed - 135 found where about 220 exist.

    A rolling median over a window several trials long tracks the drift without following
    the cue: the cue spends roughly half its time at each level, so a median over 20 s
    (about three trials) sits between them and moves only with the baseline. Subtracting it
    leaves a square centred on zero whatever the amplifier was doing underneath.

    Computed on coarse centres and interpolated back - an exact rolling median over a
    million samples costs far more than this needs.
    """
    n = len(x)
    step = max(1, int(0.5 * fs))
    half = max(1, int(win_s * fs / 2))
    centres = np.arange(0, n, step)
    base = np.empty(len(centres), np.float32)
    for i, c in enumerate(centres):
        base[i] = np.median(x[max(0, c - half):min(n, c + half)])
    return x - np.interp(np.arange(n), centres, base).astype(np.float32)


def despike(x, fs, ms=DESPIKE_MS):
    """Remove the narrow spikes that run through the whole recording.

    The photodiode line carries a few-millisecond transient every ~11 s, everywhere,
    including stretches where no cue is being shown. Each one is a level crossing, so a
    naive detector counts it as an edge - and then the minimum-dwell rule suppresses the
    REAL edge that follows within a second, so every spike costs two errors rather than
    one. That is most of the difference between the 150 edges found and the ~200 present.

    A median filter a few tens of milliseconds wide erases anything briefer than itself
    while leaving a 3 s square completely untouched: the cue is three orders of magnitude
    longer than the artefact, so there is no trade-off to tune here.
    """
    from scipy.ndimage import median_filter
    k = int(ms * fs / 1000.0) | 1
    return median_filter(x, size=k, mode="nearest")


def find_cue_region(x, fs, bin_s=CUE_BIN_S):
    """Where the cue train actually is, measured rather than configured.

    G-05's configured window came from the .mat export's clock and points at a stretch of
    the Blackrock session that is flat. Rather than carry a hand-entered number per patient
    that can be silently wrong in the wrong clock, find the train: after despiking, bins
    holding a square have a 5-95 spread of tens of units and bins holding a flat line have
    a spread of about two. The gap between those is enormous, so a threshold placed
    geometrically between the quiet floor and the loud bins separates them without tuning.
    """
    n_bin = max(1, int(len(x) / fs / bin_s))
    spread = np.array([np.subtract(*np.percentile(
        x[int(k * bin_s * fs):int((k + 1) * bin_s * fs)], [95, 5])) for k in range(n_bin)])
    # THE CUE'S OWN AMPLITUDE, robustly. Taking "any bin with spread" as cue would swallow
    # the calibration flashes, which are 47x larger and sit outside the task - that is how
    # the region came back as 100..1500 s when the train is a subset of it. So estimate the
    # typical cue amplitude from the middle of the loud bins, discarding the extreme tail
    # where the flashes live, and then keep only bins that MATCH that amplitude.
    up = spread[spread > np.percentile(spread, 50)]
    if len(up) < 3:
        return None, spread
    amp = float(np.median(up[up < np.percentile(up, 90)])) if len(up) > 4 else float(np.median(up))
    if amp <= 1e-6:
        return None, spread
    like_cue = (spread > 0.4 * amp) & (spread < 4.0 * amp)
    if not like_cue.any():
        return None, spread
    # the longest contiguous run of cue-like bins, so a stray match cannot stretch the window
    best, cur = (0, 0, 0), None
    for i, v in enumerate(like_cue):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best[0]:
                best = (i - cur, cur, i)
            cur = None
    if cur is not None and len(like_cue) - cur > best[0]:
        best = (len(like_cue) - cur, cur, len(like_cue))
    if best[0] == 0:
        return None, spread
    return (float(best[1] * bin_s), float(best[2] * bin_s)), spread


def detect_edges(x, fs, time_range=None, norm_blocks=None):
    """Midpoint crossings of a two-level signal, with hysteresis and a dwell floor."""
    x = despike(x, fs)
    t = np.arange(len(x)) / fs
    keep = np.ones(len(x), bool)
    if time_range:
        keep &= (t >= time_range[0]) & (t <= time_range[1])
    wins = norm_blocks or [(t[keep][0] if keep.any() else 0.0,
                            t[keep][-1] if keep.any() else t[-1])]
    edges, levels = [], []
    for (a, b) in wins:
        m = keep & (t >= a) & (t <= b)
        if m.sum() < int(5 * fs):
            continue
        # DETREND ONLY OVER A LONG WINDOW. A rolling median removes a drifting baseline,
        # but over a window only a few cycles long it starts tracking the CUE instead and
        # flattens the transitions it is meant to expose - measured: with a 20 s detrend a
        # block yielding 38 real transitions returned 21. Where the window is one block
        # (a couple of minutes) the drift within it is small next to an 80-unit square, so
        # the raw levels are both sufficient and safer.
        seg = x[m] if (b - a) < 240.0 else detrend(x[m], fs)
        # the two levels ARE the reference; a flash elsewhere cannot move them
        lo, hi = np.percentile(seg, [5, 95])
        if hi - lo < 1e-9:
            continue
        mid, h = (lo + hi) / 2.0, HYST * (hi - lo)
        state = seg[0] > mid
        last = -np.inf
        base = np.flatnonzero(m)[0]
        for i in range(1, len(seg)):
            v = seg[i]
            if state and v < mid - h:
                state = False
            elif (not state) and v > mid + h:
                state = True
            else:
                continue
            tt = (base + i) / fs
            if tt - last >= MIN_DWELL_S:
                edges.append(tt)
                last = tt
        levels.append((a, b, float(lo), float(hi)))
    return np.array(sorted(edges)), levels


def snap_bounds(edges, bounds, tol_s=25.0, min_gap_s=5.0):
    """Move each given boundary onto the nearest real pause in the cue train.

    The bounds are read off a plot by eye, so they are right to a few seconds - and a few
    seconds is a trial. A boundary landing mid-trial moves whole trials into the
    neighbouring block: it is why foot_left came back with 17 and foot_right with 21 when
    both should be 20.

    The cue itself says where the blocks really end. Between blocks the train stops for
    much longer than the 3 s it pauses between cues, so each real boundary is a large gap
    in the edge times. Snap to the middle of the nearest such gap, and only if one is
    within tol_s - otherwise keep what was given rather than inventing a boundary.
    """
    if not bounds or len(edges) < 3:
        return bounds, []
    gaps = np.diff(edges)
    mids = [(edges[i] + edges[i + 1]) / 2.0
            for i in np.flatnonzero(gaps > min_gap_s)]
    if not mids:
        return bounds, []
    mids = np.array(mids)
    out, moved = [], []
    for k, (a, b) in enumerate(bounds):
        new_a = a
        if k > 0:                       # the first start and last end are not pauses
            j = int(np.argmin(np.abs(mids - a)))
            if abs(mids[j] - a) <= tol_s:
                new_a = float(mids[j])
                if abs(new_a - a) > 0.05:
                    moved.append((a, new_a))
        out.append([new_a, b])
    for k in range(len(out) - 1):       # keep the blocks contiguous
        out[k][1] = out[k + 1][0]
    return [tuple(o) for o in out], moved


def expected_from_log(pre, blocks):
    """Where the behavioural log says each stimulus should start, per block.

    The log records BaselineDuration and StimuliDuration per trial but carries no clock, so
    it has to be anchored to something. It is anchored PER BLOCK, to that block's first
    detected onset, for a specific reason: the pauses BETWEEN blocks are not in the log at
    all, so a single anchor at the start would accumulate those gaps as drift and the
    comparison would look broken by the fifth block when nothing is wrong.

    Anchored this way the two rows answer a real question - within a block, does the cue
    train keep step with the intended design, and is any trial missing?
    """
    d = pre.get("beh_dir")
    t = pre.get("tsv_file")
    if not d or not t:
        return None
    t = t[-1] if isinstance(t, list) else t
    path = os.path.join(d, t)
    if not os.path.exists(path):
        return None
    import pandas as pd
    log = pd.read_csv(path, sep="	")
    if "Block" not in log.columns:
        return None
    out = []
    for gi, (name, grp) in enumerate(log.groupby("Block", sort=False)):
        if gi >= len(blocks) or len(blocks[gi]) == 0:
            break
        anchor = float(blocks[gi][0])
        base = grp["BaselineDuration"].to_numpy(float)
        stim = grp["StimuliDuration"].to_numpy(float)
        # onset_i = onset_1 + sum over previous trials of (their stim + this trial baseline)
        step = stim[:-1] + base[1:] if len(stim) > 1 else np.array([])
        out.append(anchor + np.concatenate([[0.0], np.cumsum(step)]))
    return out


def split_blocks(edges, gap_s=BLOCK_GAP_S, bounds=None):
    """Group edges into blocks - by given bounds if we have them, else by gaps.

    Gap-splitting is a guess: it assumes the pause between blocks is reliably longer than
    the longest within-block interval, and on G-05 it is not - the split turned a 5-block
    session into 12. Bounds, where the experimenter can read them off the QC figure, make
    the assignment exact and turn the per-block trial count into an independent check
    instead of something the splitter decided.
    """
    if len(edges) == 0:
        return []
    if bounds:
        return [edges[(edges >= a) & (edges < b)] for a, b in bounds]
    cuts = np.flatnonzero(np.diff(edges) > gap_s) + 1
    return [g for g in np.split(edges, cuts) if len(g) > 1]


def build_trials(blocks, labels):
    """One row per trial. The cue toggles at both transitions, so edges pair up.

    Trials whose stimulus is not ~3 s are DROPPED rather than kept and flagged. A wrong
    duration means an edge was missed or invented, so the ONSET is wrong too - and a trial
    at the wrong onset contaminates an average in a way a missing trial does not. The
    dropped ones are returned so the count is visible rather than silent.
    """
    rows, dropped = [], []
    for bi, (grp, lab) in enumerate(zip(blocks, labels)):
        # THE LAST TRIAL OF EVERY BLOCK IS DROPPED. A trial's end is taken as the onset
        # of the NEXT trial, which is right everywhere except at a block boundary: there
        # the next onset belongs to a different block, after a pause of unknown length, so
        # the last trial's post period is not its own. Rather than carry a trial whose end
        # is wrong, drop it - the cost is one trial in twenty and the alternative is a
        # contaminated epoch in every block.
        for k in range(0, len(grp) - 3, 2):
            on, off = float(grp[k]), float(grp[k + 1])
            nxt = float(grp[k + 2]) if k + 2 < len(grp) else off + cfg.design_baseline_s
            stim = off - on
            rec = dict(block=bi, condition=lab, trial=len(rows),
                       trial_in_block=k // 2,
                       onset_s=on, offset_s=off, trial_end_s=nxt,
                       stim_s=stim, post_s=nxt - off)
            if not (cfg.trial_stim_min_s <= stim <= cfg.trial_stim_max_s):
                dropped.append(rec)
                continue
            rows.append(rec)
    return rows, dropped


# ---------------------------------------------------------------------------
PANEL_PAD_S = 5.0       # context either side of the block run in the overview panel


def figure(pat, x, meta, edges, blocks, labels, rows, out_png, window=None,
           expected=None):
    fs = meta["fs"]
    t = np.arange(len(x)) / fs
    fig = plt.figure(figsize=(13.5, 7.0), dpi=160)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0], hspace=0.55, wspace=0.28,
                          left=0.06, right=0.985, top=0.90, bottom=0.08)

    ax = fig.add_subplot(gs[0, :])
    # ZOOM TO THE TASK. Over a 1500 s session the 720 s that hold the blocks are half the
    # width and the cue itself is a 90-unit ripple, so the overview showed mostly empty
    # recording. Restrict to the block run with a little context either side, and take the
    # y range from THAT window - the calibration flash is 47x the cue and sits outside it,
    # so including it flattens the thing being inspected into a line.
    tv, xv = t, x
    if window:
        w0, w1 = window[0] - PANEL_PAD_S, window[1] + PANEL_PAD_S
        m = (t >= w0) & (t <= w1)
        if m.any():
            tv, xv = t[m], x[m]
    step = max(1, len(xv) // 40000)
    ax.plot(tv[::step], xv[::step], lw=0.5, color=GREY)
    q1, q99 = np.percentile(xv, [1, 99])
    pad = 0.35 * (q99 - q1) + 1e-6
    ax.set_ylim(q1 - pad, q99 + pad)
    ax.set_xlim(tv[0], tv[-1])
    for b in meta["parts"][:-1]:
        ax.axvline(b["t0"] + b["dur"], color=GREY, lw=0.6, ls=(0, (2, 2)))
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(blocks))))
    for g, c, lab in zip(blocks, cols, labels):
        ax.axvspan(g[0], g[-1], color=c, alpha=0.16, lw=0)
        ax.text((g[0] + g[-1]) / 2, ax.get_ylim()[1], lab, fontsize=7, ha="center",
                va="top", color=INK)
    # TWO MARKER ROWS, kept apart on purpose. The upper row is what the behavioural log
    # says should have happened; the lower is what the photodiode actually did. Drawing
    # them on one row would hide exactly the disagreement they exist to show.
    y_hi = float(np.percentile(xv, 99))
    y_lo = float(np.percentile(xv, 1))
    span = max(y_hi - y_lo, 1e-6)
    if expected is not None:
        allx = np.concatenate([e for e in expected if len(e)]) if expected else np.array([])
        if len(allx):
            ax.plot(allx, np.full(len(allx), y_hi + 0.20 * span), "|", ms=5,
                    color="#1b7837", clip_on=False)
            ax.text(tv[0], y_hi + 0.20 * span, "log ", fontsize=6, color="#1b7837",
                    ha="right", va="center", clip_on=False)
    ax.plot(edges, np.full(len(edges), y_hi + 0.08 * span), "|", ms=5, color=RED,
            clip_on=False)
    ax.text(tv[0], y_hi + 0.08 * span, "PD ", fontsize=6, color=RED, ha="right",
            va="center", clip_on=False)
    # every detected onset AND offset as a dashed line, so a mis-paired edge is visible
    for r in rows:
        ax.axvline(r["onset_s"], color=RED, lw=0.35, ls=(0, (1.5, 2.5)), alpha=0.55)
        ax.axvline(r["offset_s"], color="#5b2c83", lw=0.35, ls=(0, (1.5, 2.5)), alpha=0.5)
    ax.set_title(f"{pat}  ·  Blackrock {meta['channel']}  ·  {len(edges)} edges, "
                 f"{len(blocks)} blocks, {len(rows)} trials", fontsize=10, color=INK,
                 loc="left")
    ax.set_xlabel("seconds since the first part began", fontsize=8, color=MUTED)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", color=GREY, lw=0.35, alpha=0.6)
    for lb in ax.get_xticklabels():
        lb.set_rotation(90)
        lb.set_fontsize(5.4)

    ax = fig.add_subplot(gs[1, 0])
    if len(edges) > 1:
        d = np.diff(edges)
        ax.hist(d[d < 20], bins=60, color=BLUE)
        ax.axvline(cfg.design_stim_s, color=RED, ls="--", lw=1)
        ax.axvline(cfg.design_baseline_s, color=GREEN, ls="--", lw=1)
    ax.set_title("edge intervals (red = stim, green = baseline)", fontsize=8, color=INK)
    ax.set_xlabel("s", fontsize=7.5, color=MUTED)

    ax = fig.add_subplot(gs[1, 1])
    if rows:
        ax.hist([r["stim_s"] for r in rows], bins=40, color=BLUE, alpha=0.75, label="stim")
        ax.hist([r["post_s"] for r in rows], bins=40, color=GREEN, alpha=0.6, label="post")
        ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("per-trial durations", fontsize=8, color=INK)
    ax.set_xlabel("s", fontsize=7.5, color=MUTED)

    ax = fig.add_subplot(gs[1, 2])
    if blocks:
        n = [len(g) // 2 for g in blocks]
        ax.barh(range(len(n)), n, color=BLUE)
        ax.set_yticks(range(len(n)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        for i, v in enumerate(n):
            ax.text(v, i, f" {v}", fontsize=6.5, va="center", color=INK)
    ax.set_title("trials per block", fontsize=8, color=INK)

    ax = fig.add_subplot(gs[2, :])
    if len(blocks):
        g = blocks[0]
        a, b = g[0] - 2, min(g[0] + 40, t[-1])
        m = (t >= a) & (t <= b)
        ax.plot(t[m], x[m], lw=0.7, color=INK)
        for e in edges[(edges >= a) & (edges <= b)]:
            ax.axvline(e, color=RED, lw=0.8, alpha=0.8)
        ax.set_title("first block, first 40 s - every red line is a detected edge",
                     fontsize=8, color=INK, loc="left")
    ax.set_xlabel("s", fontsize=7.5, color=MUTED)

    for a_ in fig.axes:
        a_.tick_params(labelsize=7, colors=MUTED, length=2)
        for sp in a_.spines.values():
            sp.set_color(GREY)
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(pat, force=False):
    pre = cfg.MM_PRESETS[pat]
    print(f"\n=== {pat}  ({pre['pat_name']})  {pre.get('status','')}")
    x, meta = extract_pd(pat, pre, force=force)
    # A CONFIGURED WINDOW WINS. Auto-detection is the fallback for a patient nobody has
    # looked at yet; where the experimenter has read the bounds off the QC figure that is
    # better evidence than any heuristic on the amplitude, and my heuristics have been
    # wrong twice on this patient alone.
    tr = pre.get("time_range")
    bounds = pre.get("block_bounds")
    if tr is None:
        tr, _spread = find_cue_region(despike(x, meta["fs"]), meta["fs"])
        print(f"  cue train auto-detected at {tr[0]:.0f}..{tr[1]:.0f} s" if tr
              else "  could not locate a cue train automatically")
    else:
        print(f"  cue window from config: {tr[0]:.0f}..{tr[1]:.0f} s"
              + (f", {len(bounds)} block bounds given" if bounds else ""))
    # NORMALISE PER BLOCK. Estimating the two levels once across the whole window folds in
    # the flat gaps between blocks, which pulls the 5-95 percentiles inward: the midpoint
    # survives but the hysteresis band shrinks with them, and shallow crossings are missed.
    # Each block is a stretch where the cue is continuously present, so it is the natural
    # unit to measure the levels on.
    norm = pre.get("pd_norm_blocks") or bounds
    edges, levels = detect_edges(x, meta["fs"], tr, norm)
    if bounds:
        bounds, moved = snap_bounds(edges, bounds)
        for a, b in moved:
            print(f"  block bound {a:.0f} s -> {b:.1f} s (snapped to the pause in the train)")
    blocks = split_blocks(edges, bounds=bounds)
    print(f"  {len(edges)} edges -> {len(blocks)} blocks "
          f"({[len(g)//2 for g in blocks]} trials)")
    for lab, g in zip([cfg.COND_ALIAS[c] for c in cfg.BLOCK_ORDER], blocks):
        print(f"      {lab:11s} {len(g):3d} edges -> {len(g)//2:3d} trials")
    if len(blocks) != len(cfg.BLOCK_ORDER):
        print(f"  !! expected {len(cfg.BLOCK_ORDER)} blocks, found {len(blocks)}. "
              f"The block labels below are POSITIONAL and will be wrong if a block was "
              f"split or merged - check the figure before using the table.")
    labels = [cfg.COND_ALIAS[c] for c in cfg.BLOCK_ORDER][:len(blocks)]
    labels += [f"block{ i }" for i in range(len(labels), len(blocks))]
    rows, dropped = build_trials(blocks, labels)
    if dropped:
        by = {}
        for d in dropped:
            by[d["condition"]] = by.get(d["condition"], 0) + 1
        print(f"  dropped {len(dropped)} trials with stimulus outside "
              f"{cfg.trial_stim_min_s}-{cfg.trial_stim_max_s} s  "
              + ", ".join(f"{k} {v}" for k, v in by.items()))
    n_exp = pre.get("n_trials_expected")
    if n_exp:
        exp = n_exp[-1] if isinstance(n_exp, list) else n_exp
        flag = "OK" if len(rows) == exp else "MISMATCH"
        print(f"  trials found {len(rows)}, log says {exp}   <-- {flag}")

    out_dir = os.path.join(ROOT, "outputs", pat)
    os.makedirs(out_dir, exist_ok=True)
    tsv = os.path.join(out_dir, f"{pre['pat_name']}_MM_trials.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        cols = ["trial", "block", "condition", "trial_in_block",
                "onset_s", "offset_s", "trial_end_s", "stim_s", "post_s"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    png = os.path.join(out_dir, f"{pre['pat_name']}_MM_triggers.png")
    expected = expected_from_log(pre, blocks)
    if expected:
        n_log = sum(len(e) for e in expected)
        print(f"  behavioural log: {n_log} trials, anchored per block to the first "
              f"detected onset")
    figure(pat, x, meta, edges, blocks, labels, rows, png,
           window=(bounds[0][0], bounds[-1][1]) if bounds else tr,
           expected=expected)
    print(f"  wrote {tsv}")
    print(f"  wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-read the raw .ns6 parts")
    a = ap.parse_args()
    pats = cfg.patient_ids if a.all else [a.patient or "G-05"]
    for p in pats:
        if p not in cfg.MM_PRESETS:
            print(f"{p}: not in MM_PRESETS")
            continue
        try:
            run(p, force=a.force)
        except Exception as e:                       # keep the loop alive
            print(f"  !! {p} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
